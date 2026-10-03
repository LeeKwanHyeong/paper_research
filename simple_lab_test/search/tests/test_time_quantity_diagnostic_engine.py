"""Synthetic, validation-only contracts for the isolated J/Q/T engine."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.time_quantity_diagnostic import (
    configure_parameters, gradient_diagnostics, run_arm, task_outputs,
    update_selectors,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(seed=42):
    torch.manual_seed(seed)
    result, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=1.0, max_seq_len=8, time_intercept_limit=300.0)
    return result


def batch():
    dts = torch.tensor([[0., 0., 1., 2.], [0., 1., 1., 3.], [0., 0., 2., 1.], [0., 1., 3., 2.]])
    mask = torch.tensor([[False, False, True, True], [False, True, True, True], [False, False, True, True], [False, True, True, True]])
    quantities = torch.tensor([[0., 0., 2., 4.], [0., 1., 2., 3.], [0., 0., 3., 5.], [0., 2., 1., 7.]])
    return dts, mask, quantities


def loaders():
    dataset = TensorDataset(*batch())
    return (
        DataLoader(dataset, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42)),
        DataLoader(dataset, batch_size=2, shuffle=False, generator=torch.Generator().manual_seed(43)),
    )


def run(path, objective="joint", **kwargs):
    training, validation = loaders()
    return run_arm(model=model(), train_loader=training, validation_loader=validation,
                   objective=objective, output_dir=path, epochs=3, seed=42,
                   identity={"data": "synthetic-v1", "source": "test"}, **kwargs)


def test_joint_forward_and_gradients_are_exact_baseline():
    first, second = model(), model()
    args = batch()
    rng = torch.get_rng_state()
    baseline = target_outputs(first, *args, lambda_log_qty=1.0)
    baseline["joint_loss"].mean().backward()
    torch.set_rng_state(rng)
    candidate = task_outputs(second, *args, "joint")
    candidate["objective_loss"].mean().backward()
    for name in ("joint_loss", "time_loss", "quantity_train_loss", "log_qty_loss", "true_qty", "pred_qty", "history_length"):
        assert torch.equal(baseline[name], candidate[name]), name
    for (name, left), (_, right) in zip(first.named_parameters(), second.named_parameters(), strict=True):
        if left.grad is None:
            assert right.grad is None, name
        else:
            assert torch.equal(left.grad, right.grad), name


@pytest.mark.parametrize("objective,inactive", [("quantity_only", "time"), ("time_only", "quantity")])
def test_inactive_head_is_not_called_and_excluded_from_optimizer(objective, inactive):
    candidate = model()
    def unexpected(*args, **kwargs):
        raise AssertionError("Inactive head called")
    setattr(candidate, "log_f_dt" if inactive == "time" else "quantity_outputs", unexpected)
    parameters = configure_parameters(candidate, objective)
    outputs = task_outputs(candidate, *batch(), objective)
    outputs["objective_loss"].mean().backward()
    for name, parameter in candidate.named_parameters():
        excluded = (name in {"b_t", "w_raw"} or name.startswith("v_t.")) if inactive == "time" else name.startswith("quantity_head.")
        assert parameter.requires_grad is not excluded
        assert any(parameter is p for p in parameters) is not excluded
        if excluded:
            assert parameter.grad is None
    assert outputs["time_loss" if inactive == "time" else "quantity_train_loss"] is None


@pytest.mark.parametrize("objective", ["joint", "quantity_only", "time_only"])
def test_epoch_resume_matches_uninterrupted_and_selectors(tmp_path, objective):
    full = run(tmp_path / "full", objective)
    partial = run(tmp_path / "resumed", objective, stop_after_epochs=1)
    resumed = run(tmp_path / "resumed", objective, resume=True)
    assert partial["status"] == "paused_at_epoch_boundary"
    assert full["history"] == resumed["history"]
    assert full["last_state_sha256"] == resumed["last_state_sha256"]
    assert full["selectors"] == resumed["selectors"]
    assert full["global_step"] == 6
    initial = model().state_dict()
    payload = torch.load(tmp_path / "full" / "last_epoch_state.pt", weights_only=False)
    for name, value in initial.items():
        inactive = (objective == "quantity_only" and (name in {"b_t", "w_raw"} or name.startswith("v_t."))) or (objective == "time_only" and name.startswith("quantity_head."))
        if inactive:
            assert torch.equal(value, payload["model_state_dict"][name]), name
    for key, info in full["selectors"].items():
        if info["applicable"]:
            best = min(full["history"], key=lambda row: row[key])
            assert info["best_epoch"] == best["epoch"]
            assert info["best_value"] == best[key]
        else:
            assert info["best_epoch"] is None
            assert all(row[key] is None for row in full["history"])
            assert not (tmp_path / "full" / f"best_{key}_model.pt").exists()


def test_common_initialization_batch_order_and_step_budget_across_arms(tmp_path):
    summaries = [run(tmp_path / objective, objective) for objective in ("joint", "quantity_only", "time_only")]
    assert len({summary["initial_state_sha256"] for summary in summaries}) == 1
    for epoch in range(3):
        assert len({summary["history"][epoch]["train_batch_order_sha256"] for summary in summaries}) == 1
        assert len({summary["history"][epoch]["global_step"] for summary in summaries}) == 1


def test_resume_rejects_identity_initialization_and_settings_drift(tmp_path):
    run(tmp_path, stop_after_epochs=1)
    training, validation = loaders()
    arguments = dict(model=model(), train_loader=training, validation_loader=validation,
                     objective="joint", output_dir=tmp_path, epochs=3, seed=42,
                     identity={"data": "wrong", "source": "test"}, resume=True)
    with pytest.raises(ValueError, match="identity"):
        run_arm(**arguments)
    arguments["identity"] = {"data": "synthetic-v1", "source": "test"}
    arguments["model"] = model(43)
    with pytest.raises(ValueError, match="identity"):
        run_arm(**arguments)
    arguments["model"] = model()
    arguments["grad_clip"] = .5
    with pytest.raises(ValueError, match="identity"):
        run_arm(**arguments)
    with pytest.raises(ValueError, match="Output exists"):
        run(tmp_path)


def test_strict_earliest_ties_and_nonfinite_selector_rejection():
    selectors = {"raw_quantity_rmse": {"applicable": True, "best_value": None}}
    state = {"x": torch.ones(1)}
    for epoch, value in ((1, 3.), (2, 2.), (3, 2.)):
        update_selectors(selectors, {"raw_quantity_rmse": value}, epoch=epoch, global_step=epoch, state=state)
    assert selectors["raw_quantity_rmse"]["best_epoch"] == 2
    with pytest.raises(ValueError, match="Nonfinite selector"):
        update_selectors(selectors, {"raw_quantity_rmse": float("nan")}, epoch=4, global_step=4, state=state)


def test_gradient_probe_preserves_rng_state_modes_flags_and_existing_gradients():
    candidate = model()
    # Avoid the deliberately zero-initialized quantity head's zero encoder gradient.
    with torch.no_grad():
        candidate.quantity_head.weight.fill_(.02)
    next(candidate.parameters()).requires_grad_(False)
    for parameter in candidate.parameters():
        parameter.grad = torch.full_like(parameter, 7.)
    state_before = canonical_state_dict_sha256(candidate.state_dict())
    flags = [p.requires_grad for p in candidate.parameters()]
    gradients = [p.grad.clone() for p in candidate.parameters()]
    rng = torch.get_rng_state().clone()
    report = gradient_diagnostics(candidate, *batch())
    assert torch.equal(torch.get_rng_state(), rng)
    assert state_before == canonical_state_dict_sha256(candidate.state_dict())
    assert candidate.training
    assert flags == [p.requires_grad for p in candidate.parameters()]
    assert all(torch.equal(before, after.grad) for before, after in zip(gradients, candidate.parameters(), strict=True))
    assert report["global_joint_preclip_norm"] > 0
    assert 0 < report["global_joint_clip_factor"] <= 1
    assert report["per_task_gradient_norms"]["quantity"]["time_head"] == 0
    assert report["per_task_gradient_norms"]["time"]["quantity_head"] == 0
    assert report["global_quantity_preclip_norm"] > 0
    assert report["global_time_preclip_norm"] > 0
    assert 0 < report["global_quantity_clip_factor"] <= 1
    assert 0 < report["global_time_clip_factor"] <= 1
    assert 0 <= report["time_head_joint_squared_norm_share"] <= 1


def test_target_and_padding_causality():
    candidate = model().eval()
    with torch.no_grad():
        candidate.quantity_head.weight.fill_(.02)
    dts, mask, quantities = batch()
    original = task_outputs(candidate, dts, mask, quantities, "joint")
    dts_changed, quantities_changed = dts.clone(), quantities.clone()
    dts_changed[~mask] = 999.
    quantities_changed[~mask] = 999.
    dts_changed[:, -1] = 111.
    quantities_changed[:, -1] = 222.
    changed = task_outputs(candidate, dts_changed, mask, quantities_changed, "joint")
    assert torch.allclose(original["pred_qty"], changed["pred_qty"], atol=1e-6, rtol=0)
    time_only_original = task_outputs(candidate, dts, mask, quantities, "time_only")
    quantity_changed_only = quantities.clone()
    quantity_changed_only[:, -1] = 222.
    time_only_changed = task_outputs(candidate, dts, mask, quantity_changed_only, "time_only")
    assert torch.equal(time_only_original["time_loss"], time_only_changed["time_loss"])


def test_cap_and_route_guard():
    candidate = model()
    candidate.time_intercept_limit = 30.
    with pytest.raises(ValueError, match="cap 300"):
        task_outputs(candidate, *batch(), "joint")
    candidate.time_intercept_limit = 300.
    candidate.quantity_memory_gradient_mode = "adapter_only"
    with pytest.raises(ValueError, match="gradient route"):
        task_outputs(candidate, *batch(), "joint")


def test_resume_rejects_optimizer_corruption(tmp_path):
    run(tmp_path, stop_after_epochs=1)
    path = tmp_path / "last_epoch_state.pt"
    payload = torch.load(path, weights_only=False)
    payload["optimizer_state_dict"]["param_groups"][0]["lr"] = .9
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer state hash"):
        run(tmp_path, resume=True)


def test_loader_split_guard(tmp_path):
    training, validation = loaders()
    validation.dataset.target_splits = {"test"}
    with pytest.raises(ValueError, match="Loader targets must be train only|Loader targets must be validation only"):
        run_arm(model=model(), train_loader=training, validation_loader=validation,
                objective="joint", output_dir=tmp_path, epochs=1, seed=42,
                identity={"data": "synthetic-v1"})


@pytest.mark.parametrize("component", ["torch", "python", "numpy", "train.loader", "train.sampler", "validation.loader"])
def test_resume_rejects_valid_but_altered_rng_state(tmp_path, component):
    import random
    import numpy as np

    run(tmp_path, stop_after_epochs=1)
    path = tmp_path / "last_epoch_state.pt"
    payload = torch.load(path, weights_only=False)
    if component == "torch":
        payload["rng_state"]["torch"] = torch.Generator().manual_seed(987).get_state()
    elif component == "python":
        payload["rng_state"]["python"] = random.Random(987).getstate()
    elif component == "numpy":
        payload["rng_state"]["numpy"] = np.random.RandomState(987).get_state()
    else:
        payload["loader_generator_states"][component] = torch.Generator().manual_seed(987).get_state()
    torch.save(payload, path)
    with pytest.raises(ValueError, match="RNG/loader generator state hash"):
        run(tmp_path, resume=True)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "mps"])
def test_unqualified_accelerator_training_is_rejected_before_model_transfer(tmp_path, monkeypatch, device):
    candidate = model()
    def unexpected(*args, **kwargs):
        raise AssertionError("Unqualified device transfer attempted")
    monkeypatch.setattr(candidate, "to", unexpected)
    training, validation = loaders()
    output = tmp_path / "not_created"
    with pytest.raises(ValueError, match="CPU-only until CUDA replay qualification"):
        run_arm(model=candidate, train_loader=training, validation_loader=validation,
                objective="joint", output_dir=output, epochs=1, seed=42,
                identity={"data": "synthetic-v1"}, device=device)
    assert not output.exists()


def test_resume_rejects_generator_alias_drift_with_identical_initial_bytes(tmp_path):
    run(tmp_path, stop_after_epochs=1)
    training, validation = loaders()
    # Both initial states match the original, but sampler no longer shares the
    # loader's generator: iterator creation would now consume a different stream.
    training.sampler.generator = torch.Generator().manual_seed(42)
    with pytest.raises(ValueError, match="Resume identity"):
        run_arm(model=model(), train_loader=training, validation_loader=validation,
                objective="joint", output_dir=tmp_path, epochs=3, seed=42,
                identity={"data": "synthetic-v1", "source": "test"}, resume=True)
