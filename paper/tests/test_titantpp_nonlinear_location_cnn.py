"""Fabricated tensors/checkpoints only; no real data, remote execution or fits."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest
import torch

from models.TPPs import CountAwareTitanNonlinearLocationCNN as candidate
from models.TPPs.CountAwareTitanCNNGRU import observed_context
from paper.scripts import run_titantpp_nonlinear_location_cnn as route
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def forbid_real_data():
    def no(*args, **kwargs):
        raise AssertionError("No real parquet inputs")
    with patch.object(pl, "read_parquet", no), patch.object(pl, "scan_parquet", no):
        yield


def build(arm, seed=42):
    torch.manual_seed(seed)
    return route.build_model(route.fabricated_data(), arm)


def batch():
    dt = torch.tensor([[0., 1., 3., 2., 4., 2., 3., 5.], [0., 0., 0., 2., 5., 1., 4., 2.]])
    quantity = torch.tensor([[0., 0., 5., 2., 7., 4., 3., 6.], [0., 0., 0., 2., 4., 8., 1., 5.]])
    mask = torch.tensor([[False] + [True] * 7, [False] * 3 + [True] * 5])
    return dt, quantity, mask


def activate(model):
    with torch.no_grad():
        attn = model.encoder.layers[0].attn
        for name in ("q", "k", "v"):
            getattr(attn, "causal_" + name + "_kernel").normal_(0, .1)
        if hasattr(attn, "nonlinear_location_branches"):
            for p in attn.nonlinear_location_branches.parameters():
                p.normal_(0, .1)
        model.quantity_head.weight.normal_(0, .1)
        model.time_scale_weight.weight.normal_(0, .01)
    model.eval()


def payload(arm):
    model, meta = build(arm)
    interface = {"time_head": meta["time_head"]}
    return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "variant": route.VARIANT, "interface_meta": interface,
        "resume_identity": {"backbone": arm, "interface_meta": interface,
            "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": route.role_for_arm(arm)}}}


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
@pytest.mark.parametrize("seed", (42, 52, 62))
def test_common_initial_tensors_rng_parameter_budget_and_output(arm, seed):
    torch.manual_seed(seed)
    baseline, _ = route.original.build_model(route.fabricated_data(), candidate.LEGACY_CNN)
    rng = torch.get_rng_state().clone()
    model, meta = build(arm, seed)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in baseline.state_dict().items())
    extra = candidate.ADDED_PARAMETERS if arm in candidate.ARMS else 0
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == extra
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == meta["active_trainable_parameter_count"]
    assert not any(p.requires_grad for p in model.multilag_detail.parameters())
    baseline.eval(); model.eval()
    dt, quantity, mask = batch()
    with torch.no_grad():
        a = core.target_outputs(baseline, dt, mask, quantity, lambda_log_qty=1.)
        b = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
    assert all(torch.equal(a[key], b[key]) for key in a)


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_observed_lags_equal_grouped_convolution_and_reset(arm):
    branch = candidate.NonlinearLocationBranch(arm=arm)
    with torch.no_grad():
        branch.pointwise.weight.normal_(0, .1)
    values = torch.randn(2, 8, 64)
    mask = torch.tensor([[True, True, False, True, True, False, True, True]] * 2)
    _, sources, available = observed_context(mask)
    taps = candidate.observed_taps(values, sources, available)
    flat = taps.transpose(-1, -2).reshape(-1, 64, 3)
    expected = branch.pointwise(torch.nn.functional.gelu(branch.temporal(flat))).reshape(2, 8, 64)
    actual = branch(values, sources, available)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    keep = [0, 1, 3, 4, 6, 7]
    _, cs, ca = observed_context(mask[:, keep])
    compact = branch(values[:, keep], cs, ca)
    torch.testing.assert_close(actual[:, keep], compact, rtol=1e-5, atol=1e-6)
    write = torch.ones_like(mask); write[:, 3] = False
    _, rs, ra = observed_context(torch.ones_like(mask), memory_write_mask=write)
    reset = branch(values, rs, ra)
    _, fs, fa = observed_context(torch.ones_like(mask[:, 4:]))
    fresh = branch(values[:, 4:], fs, fa)
    torch.testing.assert_close(reset[:, 4:], fresh, rtol=1e-5, atol=1e-6)
    assert torch.equal(reset[:, 3], torch.zeros_like(reset[:, 3]))


@pytest.mark.parametrize("seed", (42, 52, 62))
def test_query_and_value_branches_have_identical_initial_tensors(seed):
    q, _ = build(candidate.QUERY, seed)
    v, _ = build(candidate.VALUE, seed)
    qb = q.encoder.layers[0].attn.nonlinear_location_branches["q"]
    vb = v.encoder.layers[0].attn.nonlinear_location_branches["v"]
    assert set(qb.state_dict()) == set(vb.state_dict())
    assert all(torch.equal(x, vb.state_dict()[k]) for k, x in qb.state_dict().items())
    assert sum(p.numel() for p in qb.parameters()) == 1024
    assert bool((qb.temporal.weight != 0).all())
    assert torch.equal(qb.pointwise.weight, torch.zeros_like(qb.pointwise.weight))


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_only_requested_location_changes_and_memory_is_untouched(arm):
    from models.TPPs.CountAwareTitanCNNGRU import causal_depthwise_residual
    model, _ = build(arm); activate(model)
    attn = model.encoder.layers[0].attn
    location = candidate.LOCATION_BY_ARM[arm]
    assert list(attn.nonlinear_location_branches) == [location]
    assert attn.qkv.weight.requires_grad
    attn._ctx_mem = torch.randn(2, 3, 64)
    x = torch.randn(2, 7, 64)
    mask = torch.ones(2, 7, dtype=torch.bool)
    _, sources, available = observed_context(mask)
    projected = dict(zip(("q", "k", "v"), attn.qkv(x).chunk(3, dim=-1)))
    expected = {k: value + causal_depthwise_residual(value, getattr(attn, "causal_" + k + "_kernel"), sources, available)
                for k, value in projected.items()}
    actual_input = []
    original_split = attn._split_heads
    def capture(value):
        actual_input.append(value.detach().clone())
        return original_split(value)
    before = deepcopy(model.state_dict())
    context_before = attn._ctx_mem.clone()
    with torch.no_grad(), patch.object(attn, "_split_heads", side_effect=capture):
        attn(x, mask, observed_sources=sources, observed_available=available)
    assert len(actual_input) == 3
    memory = torch.cat((context_before, attn.persistent_mem.expand(2, -1, -1)), dim=1)
    for i, key in enumerate(("q", "k", "v")):
        predicted = expected[key]
        if key == location:
            extra = attn.nonlinear_location_branches[key](projected[key], sources, available)
            assert bool((extra != 0).any())
            predicted = predicted + extra
        observed = actual_input[i]
        if key != "q":
            assert torch.equal(observed[:, :memory.shape[1]], memory)
            observed = observed[:, memory.shape[1]:]
        torch.testing.assert_close(observed, predicted, rtol=0, atol=0)
    assert torch.equal(attn._ctx_mem, context_before)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_channel_branch_has_no_cross_channel_or_future_dependence(arm):
    branch = candidate.NonlinearLocationBranch(arm=arm)
    with torch.no_grad():
        branch.temporal.weight.fill_(.1); branch.pointwise.weight.fill_(.1)
    x = torch.randn(1, 8, 64)
    _, sources, available = observed_context(torch.ones(1, 8, dtype=torch.bool))
    altered = x.clone(); altered[..., 9] += 2.
    original = branch(x, sources, available); changed = branch(altered, sources, available)
    assert torch.equal(original[..., 7], changed[..., 7])
    assert not torch.equal(original[..., 9], changed[..., 9])
    altered = x.clone(); altered[:, 5:] += 1000.
    assert torch.equal(original[:, :5], branch(altered, sources, available)[:, :5])


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_matching_gradient_onset_and_all_elements_active(arm):
    torch.manual_seed(52)
    branch = candidate.NonlinearLocationBranch(arm=arm)
    values = torch.randn(2, 9, 64)
    target = torch.randn_like(values)
    _, sources, available = observed_context(torch.ones(2, 9, dtype=torch.bool))
    loss = ((branch(values, sources, available) - target) ** 2).mean()
    loss.backward()
    assert torch.equal(branch.temporal.weight.grad, torch.zeros_like(branch.temporal.weight))
    assert bool((branch.pointwise.weight.grad != 0).all())
    with torch.no_grad():
        branch.pointwise.weight.sub_(.1 * branch.pointwise.weight.grad)
    branch.zero_grad()
    ((branch(values, sources, available) - target) ** 2).mean().backward()
    for parameter in branch.parameters():
        assert torch.isfinite(parameter.grad).all() and bool((parameter.grad != 0).all())


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
def test_target_padding_future_batch_and_evaluation_state(arm):
    model, _ = build(arm); activate(model)
    dt, quantity, mask = batch()
    before = deepcopy(model.state_dict())
    with torch.no_grad(), patch.object(model.multilag_detail, "forward", side_effect=AssertionError("GRU used")):
        output = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
        changed = quantity.clone(); changed[:, -1] = float("nan"); changed[~mask] = float("nan")
        changed_dt = dt.clone(); changed_dt[~mask] = float("nan")
        altered = core.target_outputs(model, changed_dt, mask, changed, lambda_log_qty=1.)
        assert torch.equal(output["pred_qty"], altered["pred_qty"])
        changed_dt = dt.clone(); changed_dt[:, -1] += 1000.
        altered = core.target_outputs(model, changed_dt, mask, quantity, lambda_log_qty=1.)
        assert torch.equal(output["pred_qty"], altered["pred_qty"])
        encoded = model._encode_base(dt, quantity, mask)
        quantity2 = quantity.clone(); quantity2[:, 5:] += 500.
        dt2 = dt.clone(); dt2[:, 5:] += 500.
        altered = model._encode_base(dt2, quantity2, mask)
        torch.testing.assert_close(encoded[:, :5], altered[:, :5], rtol=0, atol=0)
        assert torch.equal(encoded[~mask], torch.zeros_like(encoded[~mask]))
        for index in range(2):
            one = core.target_outputs(model, dt[index:index+1], mask[index:index+1], quantity[index:index+1], lambda_log_qty=1.)
            torch.testing.assert_close(one["pred_qty"], output["pred_qty"][index:index+1], rtol=1e-5, atol=1e-6)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
def test_checkpoint_identity_route_selector_and_finiteness(arm):
    data = payload(arm)
    assert candidate.validate_checkpoint(data, arm)
    rebuilt, _ = build(arm); rebuilt.load_state_dict(data["model_state_dict"])
    for other in candidate.ALL_ARMS:
        if other != arm:
            with pytest.raises(ValueError):
                candidate.validate_checkpoint(data, other)
    bad = deepcopy(data); bad["resume_identity"]["checkpoint_monitor"] = "test_raw_quantity_rmse"
    with pytest.raises(ValueError):
        candidate.validate_checkpoint(bad, arm)
    if arm in candidate.ARMS:
        bad = deepcopy(data)
        bad["model_state_dict"][candidate.EXTRA_PREFIX + candidate.LOCATION_BY_ARM[arm] + ".temporal.weight"].flatten()[0] = float("nan")
        with pytest.raises(ValueError, match="state tensor invalid"):
            candidate.validate_checkpoint(bad, arm)
        with pytest.raises(ValueError, match="relabelled"):
            candidate.validate_checkpoint(data, candidate.LEGACY_CNN)


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
def test_training_arguments_lock_common_science_and_allow_baseline(arm, tmp_path):
    contract = {"training": route.TRAINING.copy(), "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False, "test_evaluation": False}
    stub = SimpleNamespace(force_rerun=False, checkpoint_monitor="validation_raw_quantity_rmse")
    with patch.object(route.base, "training_args", return_value=stub) as mocked:
        args = route.training_args(contract, route.fabricated_data(), tmp_path, arm)
    assert (args.epochs, args.min_epochs, args.early_stopping_patience) == (300, 40, 40)
    assert mocked.call_args.args[0]["model_role"] == route.role_for_arm(arm)
    bad = deepcopy(contract); bad["training"]["patience"] = 20
    with pytest.raises(ValueError, match="budget"):
        route.training_args(bad, route.fabricated_data(), tmp_path, arm)
    bad_data = route.fabricated_data(); bad_data["model"]["lambda_log_qty"] = .5
    with pytest.raises(ValueError, match="loss"):
        route.training_args(contract, bad_data, tmp_path, arm)


def test_baseline_alias_initial_state_matches_original_and_invalid_arms_fail():
    data = route.fabricated_data()
    assert route.initial_states(data, 42)[candidate.CNN] == route.baseline_reference_initial_state_sha(data, 42)
    for arms in ((), (candidate.CNN, candidate.CNN), ("wrong",)):
        with pytest.raises(ValueError):
            route.initial_states(data, 42, arms=arms)


def test_synthetic_qualification_exercises_each_active_weight():
    result = route.synthetic_check("cpu")
    assert result["status"] == "passed"
    assert result["active_correction_gradient_parameter_count"] == {
        candidate.CNN: 576, candidate.QUERY: 1600, candidate.VALUE: 1600}
    assert result["scientific_optimizer_updates"] == 0
    for arm in candidate.ARMS:
        for name, count in result["initial_gradient_nonzero_elements"][arm].items():
            assert count == (0 if ".temporal." in name else 256)


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
@pytest.mark.parametrize("context_length", (84, 32))
def test_parameter_metadata_and_initial_state_follow_actual_context_length(arm, context_length):
    data = route.fabricated_data(); data["loader"]["max_seq_len"] = context_length
    model, meta = route.build_model(data, arm)
    assert sum(p.numel() for p in model.parameters()) == meta["registered_parameter_count"]
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == meta["active_trainable_parameter_count"]
    assert meta["parameter_count_context_length"] == context_length
    assert set(route.initial_states(data, 52)) == set(candidate.ALL_ARMS)
    item = payload(arm); item["encoder_config"] = meta; item["model_state_dict"] = model.state_dict()
    assert candidate.validate_checkpoint(item, arm)


@pytest.mark.parametrize("arm", candidate.ALL_ARMS)
def test_inherited_replay_six_metrics_hand_computation_and_validation_only(arm, tmp_path):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    import math
    item = payload(arm)
    item["model_state_sha256"] = canonical_state_dict_sha256(item["model_state_dict"])
    data = route.fabricated_data()
    data.update(quantity_boundaries_all_train_rows=[1., 2., 4., 4.], history_boundaries=[2., 3.],
        inherited_data_identity={"populations": {"validation": {"target_count": 4}}})
    checkpoint = tmp_path / "fabricated_selected.pt"; torch.save(item, checkpoint)
    dt, quantity, mask = batch()
    fabricated_predictions = {"true_qty": torch.tensor([1., 2., 5., 10.]),
        "pred_qty": torch.tensor([2., 4., 2., 6.]), "time_loss": torch.tensor([.5, 1., 2., 4.]),
        "history_length": torch.tensor([1, 2, 3, 4])}
    with patch("paper.scripts.run_taxi_quantity_interface_ablation.make_loader", return_value=[(None, dt, mask, None, quantity)]) as loader, \
         patch.object(core, "target_outputs", return_value=fabricated_predictions):
        result = route.replay_checkpoint(checkpoint, data, None, lambda: None, device="cpu")
    assert loader.call_args.kwargs["target_split"] == "validation"
    assert result["count"] == 4 and result["tail"]["count"] == 2
    assert result["qty_rmse"] == math.sqrt(7.5)
    assert result["qty_mae"] == 2.5 and result["time_nll"] == 1.875
    assert result["tail"]["qty_rmse"] == math.sqrt(12.5)
    assert result["tail"]["qty_mae"] == 3.5 and result["tail"]["time_nll"] == 3.
    assert result["evaluation_scope"] == "validation_only" and result["held_out_test_evaluated"] is False


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_checkpoint_missing_foreign_extra_and_relabelled_state_rejected(arm):
    data = payload(arm)
    key = candidate.EXTRA_PREFIX + candidate.LOCATION_BY_ARM[arm] + ".temporal.weight"
    bad = deepcopy(data); del bad["model_state_dict"][key]
    with pytest.raises(ValueError): candidate.validate_checkpoint(bad, arm)
    bad = deepcopy(data); bad["model_state_dict"][candidate.EXTRA_PREFIX + "k.temporal.weight"] = torch.zeros(256, 1, 3)
    with pytest.raises(ValueError): candidate.validate_checkpoint(bad, arm)
    other = candidate.VALUE if arm == candidate.QUERY else candidate.QUERY
    other_model, _ = build(other)
    with pytest.raises(ValueError): other_model.load_state_dict(data["model_state_dict"])
    bad = deepcopy(data); bad["encoder_config"]["nonlinear_location_route"] = "k"
    with pytest.raises(ValueError): candidate.validate_checkpoint(bad, arm)


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_common_input_head_loss_optimizer_and_memory_contract(arm, tmp_path):
    torch.manual_seed(42)
    baseline, bm = route.original.build_model(route.fabricated_data(), candidate.LEGACY_CNN)
    model, mm = build(arm)
    assert type(model.quantity_head) is type(baseline.quantity_head)
    assert type(model.time_scale_weight) is type(baseline.time_scale_weight)
    assert mm["time_head"] == bm["time_head"]
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, model.state_dict()[key])
    contract = {"training": route.TRAINING.copy(), "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False, "test_evaluation": False}
    for optkey, badvalue in (("lr", .01), ("weight_decay", 0.), ("grad_clip", 2.)):
        data = route.fabricated_data(); data["optimizer"][optkey] = badvalue
        with pytest.raises(ValueError, match="optimizer"):
            route.training_args(contract, data, tmp_path, arm)
    for key, value in (("lambda_tail", 1.), ("time_head_lr_multiplier", 2.), ("time_head_mode", "wrong")):
        data = route.fabricated_data(); data["model"][key] = value
        with pytest.raises(ValueError, match="loss"):
            route.training_args(contract, data, tmp_path, arm)
