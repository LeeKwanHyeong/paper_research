import copy
import hashlib
import json
import math
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts import mixed_quantity_objective as mixed
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.quantity_objective_comparison import QuantityStatistics
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


@pytest.fixture(autouse=True)
def one_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def statistics():
    return QuantityStatistics(mu=math.log(3.0), raw_scale=5.0)


def make_model():
    torch.manual_seed(17)
    model, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=statistics().mu,
                                     max_seq_len=8, time_intercept_limit=300.0)
    return model


def batch():
    return (torch.tensor([[0., 0., 1., 2.], [0., 1., 1., 3.]]),
            torch.tensor([[False, False, True, True], [False, True, True, True]]),
            torch.tensor([[0., 0., 2., 4.], [0., 1., 2., 3.]]))


def identity():
    return {"source": {"sha256": "a" * 64}, "data": {"dataset": "합성"},
            "runtime": {"device": "cpu"}, "execution_contract_sha256": "b" * 64}


def objective(name="mixed_matched_original", alpha=0.2, scale=0.9):
    return mixed.MixedQuantityObjective(name, alpha, scale, "c" * 64)


def calibrate(model=None, dataset=None, **kwargs):
    return mixed.calibrate_mixed_objective(model or make_model(),
                                          dataset if dataset is not None else TensorDataset(*batch()),
                                          statistics=statistics(), identity=identity(), synthetic=True, **kwargs)


def rehash(receipt):
    receipt["calibration_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in receipt.items() if key != "calibration_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def test_mixed_loss_matches_hand_calculation_and_component_gradient_sum():
    model = make_model().eval()
    outputs = mixed.mixed_joint_causal_batch_objective(model, *batch(), statistics=statistics(), objective=objective())
    expected_raw = ((outputs["pred_qty"] - outputs["true_qty"]) / 5.0).square()
    assert torch.equal(outputs["raw_qty_loss"], expected_raw)
    expected = outputs["time_loss"] + .9 * (outputs["log_qty_loss"] + .2 * expected_raw)
    assert torch.equal(outputs["objective_loss"], expected)
    head = tuple(model.quantity_head.parameters())
    log_grads = torch.autograd.grad(outputs["log_qty_loss"].mean(), head, retain_graph=True)
    raw_grads = torch.autograd.grad(expected_raw.mean(), head, retain_graph=True)
    actual = torch.autograd.grad(outputs["quantity_train_loss"].mean(), head)
    for gl, gr, total in zip(log_grads, raw_grads, actual, strict=True):
        assert torch.allclose(total, .9 * (gl + .2 * gr), atol=1e-7, rtol=1e-6)


def test_b_loss_and_all_parameter_gradients_are_bitwise_unchanged():
    baseline, candidate = make_model(), make_model()
    rng = torch.get_rng_state()
    expected = target_outputs(baseline, *batch(), lambda_log_qty=1.0)
    expected["joint_loss"].mean().backward()
    torch.set_rng_state(rng)
    actual = mixed.mixed_joint_causal_batch_objective(candidate, *batch(), statistics=statistics(),
        objective=objective("B_log_original", 0, 1))
    actual["objective_loss"].mean().backward()
    assert torch.equal(expected["joint_loss"], actual["objective_loss"])
    for (name, p), (_, q) in zip(baseline.named_parameters(), candidate.named_parameters(), strict=True):
        assert (p.grad is None and q.grad is None) or torch.equal(p.grad, q.grad), name


def test_all_arms_use_one_identical_stochastic_graph_and_time_loss(monkeypatch):
    model = make_model()
    original = mixed.b_task_outputs
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mixed, "b_task_outputs", counted)
    rng = torch.get_rng_state()
    results = []
    for arm in (objective("B_log_original", 0, 1), objective("mixed_original", .2, 1), objective()):
        torch.set_rng_state(rng)
        results.append(mixed.mixed_joint_causal_batch_objective(model, *batch(), statistics=statistics(), objective=arm))
    assert len(calls) == 3
    for result in results[1:]:
        for key in ("pred_qty", "true_qty", "time_loss", "log_qty_loss", "raw_qty_loss"):
            assert torch.equal(results[0][key], result[key]), key


def test_target_padding_causality_is_preserved_with_nonzero_head():
    model = make_model().eval()
    with torch.no_grad():
        model.quantity_head.weight.normal_(0, .01)
    before = canonical_state_dict_sha256(model.state_dict())
    args = batch()
    expected = mixed.mixed_joint_causal_batch_objective(model, *args, statistics=statistics(), objective=objective())
    args[0][~args[1]] = 999.0
    args[2][~args[1]] = 123456.0
    args[0][:, -1] = 111.0
    args[2][:, -1] = 222.0
    actual = mixed.mixed_joint_causal_batch_objective(model, *args, statistics=statistics(), objective=objective())
    assert torch.allclose(expected["pred_qty"], actual["pred_qty"], rtol=0, atol=1e-6)
    assert canonical_state_dict_sha256(model.state_dict()) == before


@pytest.mark.parametrize("changes", [
    {"alpha": float("nan")}, {"quantity_scale": float("inf")}, {"alpha": True},
    {"name": "unknown"}, {"calibration_sha256": "no"}, {"quantity_scale": 0},
    {"name": "B_log_original"}, {"name": "mixed_original", "quantity_scale": .9},
])
def test_objective_rejects_invalid_identity(changes):
    value = objective().to_dict()
    value.update(changes)
    with pytest.raises(ValueError):
        mixed.MixedQuantityObjective.from_dict(value)


def test_objective_roundtrip_is_exact_and_frozen():
    arm = objective()
    assert mixed.MixedQuantityObjective.from_dict(arm.to_dict()) == arm
    with pytest.raises(AttributeError):
        arm.alpha = 4
    with pytest.raises(ValueError, match="schema"):
        mixed.MixedQuantityObjective.from_dict({**arm.to_dict(), "other": 1})


def test_calibration_matches_hand_computed_head_moments_despite_zero_encoder_gradient():
    model = make_model()
    receipt = calibrate(model)
    indices = torch.randperm(2, generator=torch.Generator().manual_seed(1042))
    args = [value[indices] for value in batch()]
    torch.manual_seed(1042)
    model.train()
    outputs = mixed.b_task_outputs(model, *args, "joint")
    raw = ((outputs["pred_qty"] - outputs["true_qty"]) / 5).square()
    params = tuple(model.parameters())
    log_grads = torch.autograd.grad(outputs["log_qty_loss"].mean(), params, retain_graph=True, allow_unused=True)
    raw_grads = torch.autograd.grad(raw.mean(), params, allow_unused=True)
    head_indices = [i for i, (name, _) in enumerate(model.named_parameters()) if name.startswith("quantity_head.")]
    for i, gradient in enumerate(log_grads):
        if i not in head_indices:
            assert gradient is None or not torch.count_nonzero(gradient)
    a = sum(log_grads[i].double().square().sum().item() for i in head_indices)
    r = sum(raw_grads[i].double().square().sum().item() for i in head_indices)
    d = sum((log_grads[i].double() * raw_grads[i].double()).sum().item() for i in head_indices)
    assert receipt["moments"] == pytest.approx({"A": a, "R": r, "D": d}, rel=1e-12)
    alpha = .25 * math.sqrt(a / r)
    scale = math.sqrt(a / (a + 2 * alpha * d + alpha * alpha * r))
    assert receipt["alpha"] == pytest.approx(alpha, rel=1e-12)
    assert receipt["quantity_scale"] == pytest.approx(scale, rel=1e-12)
    assert receipt["head_parameter_names"] == ["quantity_head.weight", "quantity_head.bias"]
    assert .8 <= scale <= 4 / 3


def test_calibration_preserves_rng_model_flags_grads_and_training_loader_state(monkeypatch):
    model = make_model().eval()
    list(model.modules())[1].train()
    for i, p in enumerate(model.parameters()):
        if i % 2 == 0:
            p.grad = torch.full_like(p, 13.0)
        p.requires_grad_(i % 3 != 0)
    generator = torch.Generator().manual_seed(912)
    loader = DataLoader(TensorDataset(*batch()), batch_size=2, shuffle=True, generator=generator)
    torch_before = torch.get_rng_state().clone()
    np_before = np.random.get_state()
    py_before = random.getstate()
    loader_before = generator.get_state().clone()
    state_before = canonical_state_dict_sha256(model.state_dict())
    flags = [m.training for m in model.modules()]
    grads = [(p.requires_grad, p.grad, None if p.grad is None else p.grad.clone()) for p in model.parameters()]

    def prohibited(*args, **kwargs):
        raise AssertionError("Optimizer construction is forbidden")

    monkeypatch.setattr(torch.optim, "AdamW", prohibited)
    monkeypatch.setattr(torch.optim, "SGD", prohibited)
    first = calibrate(model, loader.dataset)
    second = calibrate(model, loader.dataset)
    assert first == second
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert random.getstate() == py_before
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert np.random.get_state()[2:] == np_before[2:]
    assert torch.equal(generator.get_state(), loader_before)
    assert canonical_state_dict_sha256(model.state_dict()) == state_before
    assert flags == [m.training for m in model.modules()]
    for p, (enabled, reference, value) in zip(model.parameters(), grads, strict=True):
        assert p.requires_grad == enabled and p.grad is reference
        assert p.grad is None or torch.equal(p.grad, value)


def test_receipt_sha_uses_canonical_unicode_and_detects_tampering():
    receipt = calibrate()
    arms = mixed.objectives_from_calibration(receipt)
    assert [arm.name for arm in arms] == list(mixed.NAMES)
    assert arms[0].alpha == 0 and arms[0].quantity_scale == 1
    assert arms[1].alpha == arms[2].alpha == receipt["alpha"]
    assert all(arm.calibration_sha256 == receipt["calibration_sha256"] for arm in arms)
    normalized = copy.deepcopy(receipt)
    rehash(normalized)
    assert receipt == normalized
    receipt["moments"]["A"] *= 2
    with pytest.raises(ValueError, match="checksum"):
        mixed.objectives_from_calibration(receipt)
    rehash(receipt)
    with pytest.raises(ValueError, match="coefficients|inner product"):
        mixed.objectives_from_calibration(receipt)


@pytest.mark.parametrize("field,value", [
    ("selected_sample_count", 3), ("batch_count", 2), ("validation_accessed", True),
    ("held_out_accessed", True), ("optimizer_updates", 1),
    ("head_parameter_names", ["v_t.weight"]), ("batch_sha256", "bad"),
])
def test_rehashed_receipt_still_rejects_contract_drift(field, value):
    receipt = calibrate()
    receipt[field] = value
    rehash(receipt)
    with pytest.raises(ValueError):
        mixed.objectives_from_calibration(receipt)


@pytest.mark.parametrize("splits", [None, {"validation"}, {"test"}, {"train", "validation"}, {"train", "test"}])
def test_calibration_rejects_heldout_mixed_or_unidentified_dataset(splits):
    dataset = TensorDataset(*batch())
    dataset.target_splits = splits
    with pytest.raises(ValueError, match="train only"):
        mixed.calibrate_mixed_objective(make_model(), dataset, statistics=statistics(), identity=identity())


def test_synthetic_escape_hatch_is_bounded_and_exact_type():
    class Child(TensorDataset):
        pass

    for dataset in (Child(*batch()), TensorDataset(*(value.repeat(5, 1) for value in batch()))):
        with pytest.raises(ValueError, match="at most 8"):
            calibrate(dataset=dataset)
    dataset = TensorDataset(*batch())
    dataset.target_splits = {"validation"}
    with pytest.raises(ValueError, match="provenance"):
        calibrate(dataset=dataset)


def test_train_dataset_indices_unique_and_partial_batch_gradient_mean():
    # 130 targets force a 128-row batch and a 2-row batch. Means are per
    # batch, not gradients of a concatenated sample-weighted objective.
    tensors = [value.repeat(65, 1) for value in batch()]
    tensors[2][:128, -1] = 1.0
    tensors[2][128:, -1] = 7.0
    dataset = TensorDataset(*tensors)
    dataset.target_splits = {"train"}
    stages = []
    receipt = mixed.calibrate_mixed_objective(make_model(), dataset, statistics=statistics(),
                                             identity=identity(), budget_check=stages.append)
    selected = torch.randperm(130, generator=torch.Generator().manual_seed(1042))
    assert len(set(selected.tolist())) == 130
    assert receipt["selected_indices_sha256"] == canonical_state_dict_sha256({"indices": selected})
    assert receipt["selected_sample_count"] == 130 and receipt["batch_count"] == 2
    assert stages == ["mixed_calibration_start", "mixed_calibration_batch", "mixed_calibration_batch", "mixed_calibration_complete"]
    model = make_model().train()
    torch.manual_seed(1042)
    rows = []
    for indices in selected.split(128):
        outputs = mixed.b_task_outputs(model, *(value[indices] for value in tensors), "joint")
        head = tuple(model.quantity_head.parameters())
        gl = torch.autograd.grad(outputs["log_qty_loss"].mean(), head, retain_graph=True)
        raw = ((outputs["pred_qty"] - outputs["true_qty"]) / statistics().raw_scale).square()
        gr = torch.autograd.grad(raw.mean(), head)
        rows.append((sum(g.double().square().sum().item() for g in gl),
                     sum(g.double().square().sum().item() for g in gr),
                     sum((a.double() * b.double()).sum().item() for a, b in zip(gl, gr, strict=True))))
    expected = {name: sum(row[i] for row in rows) / 2 for i, name in enumerate(("A", "R", "D"))}
    assert receipt["moments"] == pytest.approx(expected, rel=1e-12)
    mixed.objectives_from_calibration(receipt)


def test_zero_gradient_calibration_fails_without_fallback_and_restores_rng(monkeypatch):
    original = mixed.b_task_outputs

    def zero(*args, **kwargs):
        result = original(*args, **kwargs)
        result["log_qty_loss"] = result["log_qty_loss"] * 0
        result["pred_qty"] = result["true_qty"] + result["pred_qty"] * 0
        return result

    monkeypatch.setattr(mixed, "b_task_outputs", zero)
    model = make_model().eval()
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="Degenerate"):
        calibrate(model)
    assert not model.training and torch.equal(before, torch.get_rng_state())


def test_budget_failure_restores_state_modes_and_rng():
    model = make_model().eval()
    before = torch.get_rng_state()
    initial = canonical_state_dict_sha256(model.state_dict())

    def budget(stage):
        if stage == "mixed_calibration_batch":
            raise RuntimeError("budget exhausted")

    with pytest.raises(RuntimeError, match="budget"):
        calibrate(model, budget_check=budget)
    assert not model.training and torch.equal(before, torch.get_rng_state())
    assert canonical_state_dict_sha256(model.state_dict()) == initial


def test_mutating_forward_restored_and_rejected(monkeypatch):
    model = make_model().eval()
    model.register_buffer("test_counter", torch.tensor(0.0))
    original = mixed.b_task_outputs

    def mutation(*args, **kwargs):
        args[0].test_counter.add_(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mixed, "b_task_outputs", mutation)
    with pytest.raises(ValueError, match="mutated model state; state restored"):
        calibrate(model)
    assert model.test_counter.item() == 0 and not model.training


def test_calibration_rejects_changed_head_and_batch_nonfinite_objective():
    model = make_model()
    with torch.no_grad():
        model.quantity_head.weight.fill_(1)
    with pytest.raises(ValueError, match="initial zero"):
        calibrate(model)
    model = make_model()
    with torch.no_grad():
        model.quantity_head.bias.add_(1)
    with pytest.raises(ValueError, match="bias differs"):
        calibrate(model)
    model = make_model()
    with torch.no_grad():
        model.quantity_head.bias.fill_(60)
    with pytest.raises(ValueError, match="nonfinite"):
        mixed.mixed_joint_causal_batch_objective(model, *batch(), statistics=statistics(), objective=objective())
    model = make_model()
    with torch.no_grad():
        model.b_t.fill_(1000)
    with pytest.raises(ValueError, match="Nonfinite calibration loss"):
        calibrate(model)
