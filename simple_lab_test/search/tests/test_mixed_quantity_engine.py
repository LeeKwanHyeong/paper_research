"""CPU integration checks for opt-in mixed objectives on the existing engine."""

from dataclasses import replace
import json

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.mixed_quantity_objective import (
    MixedQuantityObjective,
    mixed_joint_causal_batch_objective,
)
from paper.scripts.quantity_comparison_engine import run_case
from paper.scripts.quantity_objective_comparison import (
    QuantityCase,
    fit_train_quantity_statistics,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def statistics():
    return fit_train_quantity_statistics(
        torch.tensor([0., 1., 2., 4., 3., 2.]),
        torch.tensor([4., 3., 5., 2.]), split="train")


def make_model(seed=31):
    torch.manual_seed(seed)
    model, _ = build_count_aware_model(
        "titantpp", hidden_dim=8, train_log_mean=statistics().mu,
        max_seq_len=8, time_intercept_limit=300.0)
    return model


def tensors():
    return (
        torch.tensor([[0., 0., 1., 2.], [0., 1., 1., 3.],
                      [0., 0., 2., 2.], [0., 1., 2., 4.]]),
        torch.tensor([[False, False, True, True], [False, True, True, True],
                      [False, False, True, True], [False, True, True, True]]),
        torch.tensor([[0., 0., 2., 4.], [0., 1., 2., 3.],
                      [0., 0., 1., 5.], [0., 2., 1., 2.]]),
    )


def objectives():
    # Synthetic coefficients test execution identity, not data calibration.
    receipt = "c" * 64
    return (
        MixedQuantityObjective("B_log_original", 0., 1., receipt),
        MixedQuantityObjective("mixed_original", .2, 1., receipt),
        MixedQuantityObjective("mixed_matched_original", .2, .75, receipt),
    )


def run(path, objective, *, case=QuantityCase.B_LOG_ORIGINAL, model_seed=31, **kwargs):
    dataset = TensorDataset(*tensors())
    train = DataLoader(dataset, batch_size=2, shuffle=True,
                       generator=torch.Generator().manual_seed(41))
    validation = DataLoader(dataset, batch_size=2, shuffle=False,
                            generator=torch.Generator().manual_seed(42))
    return run_case(
        model=make_model(model_seed), train_loader=train, validation_loader=validation,
        case=case, statistics=statistics(), output_dir=path, epochs=3, seed=42,
        identity={"source": {"revision": "synthetic"},
                  "data": {"sha256": "synthetic"},
                  "runtime": {"name": "synthetic-cpu"},
                  "execution_contract_sha256": "a" * 64},
        mixed_objective=objective, **kwargs)


def load_state(path):
    return torch.load(path / "last_epoch_state.pt", weights_only=False)


def assert_old_values_preserved(old, mixed):
    """Allow mixed diagnostic additions, requiring exact legacy values."""
    if isinstance(old, dict):
        for name, value in old.items():
            assert name in mixed
            assert_old_values_preserved(value, mixed[name])
    elif isinstance(old, list):
        assert len(old) == len(mixed)
        for first, second in zip(old, mixed, strict=True):
            assert_old_values_preserved(first, second)
    else:
        assert old == mixed


def test_zero_auxiliary_b_preserves_original_training_bitwise(tmp_path):
    baseline = run(tmp_path / "legacy", None)
    mixed = run(tmp_path / "mixed-b", objectives()[0])
    assert baseline["condition"] == {
        "case": "B_log_original", "mu": statistics().mu,
        "raw_scale": statistics().raw_scale}
    assert "raw_quantity_scaled_mse" not in baseline["history"][0]
    assert "post_first_update_gradient_diagnostic" not in baseline["history"][0]
    assert_old_values_preserved(baseline["history"], mixed["history"])
    assert baseline["selectors"] == mixed["selectors"]
    first, second = load_state(tmp_path / "legacy"), load_state(tmp_path / "mixed-b")
    for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        assert first[key] == second[key]
    for key in first["model_state_dict"]:
        assert torch.equal(first["model_state_dict"][key], second["model_state_dict"][key])


@pytest.mark.parametrize("objective", objectives(), ids=lambda value: value.name)
def test_mixed_full_and_epoch_resume_restore_all_training_state(tmp_path, objective):
    full = run(tmp_path / "full", objective)
    paused = run(tmp_path / "resumed", objective, stop_after_epochs=1)
    resumed = run(tmp_path / "resumed", objective, resume=True)
    assert paused["status"] == "paused_at_epoch_boundary"
    assert full == resumed
    first, second = load_state(tmp_path / "full"), load_state(tmp_path / "resumed")
    for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
        assert first[key] == second[key]
    timing = json.loads((tmp_path / "resumed" / "timing.json").read_text())
    assert [row["epoch"] for row in timing["epochs"]] == [1, 2, 3]


def test_three_conditions_share_batches_steps_and_record_weighted_components(tmp_path):
    summaries = [run(tmp_path / objective.name, objective) for objective in objectives()]
    assert len({summary["initial_state_sha256"] for summary in summaries}) == 1
    assert {summary["global_step"] for summary in summaries} == {6}
    for epoch in range(3):
        assert len({summary["history"][epoch]["train_batch_order_sha256"]
                    for summary in summaries}) == 1
    for objective, summary in zip(objectives(), summaries, strict=True):
        assert summary["condition"]["mixed_objective"] == objective.to_dict()
        for row in summary["history"]:
            assert row["train_count"] == row["validation_count"] == 4
            assert row["train_batches"] == row["validation_batches"] == 2
            for metrics in (row, row["train"]):
                expected = objective.quantity_scale * (
                    metrics["log_quantity_mse"]
                    + objective.alpha * metrics["raw_quantity_scaled_mse"])
                assert metrics["quantity_train_loss"] == pytest.approx(expected, rel=1e-6)
        first = summary["history"][0]
        initial = first["gradient_diagnostic"]
        post_update = first["post_first_update_gradient_diagnostic"]
        assert initial["quantity"]["encoder"] == 0.
        assert post_update["quantity"]["encoder"] > 0.
        for diagnostic in (initial, post_update):
            assert set(diagnostic["quantity_component_unweighted_norms"]) == {"log", "raw_scaled"}
            assert diagnostic["actual_global_preclip_norm"] == pytest.approx(
                diagnostic["global_joint_preclip_norm"], rel=1e-6)
        assert all(row["post_first_update_gradient_diagnostic"] is None
                   for row in summary["history"][1:])


@pytest.mark.parametrize("changes", [
    {"alpha": .3}, {"quantity_scale": .9}, {"calibration_sha256": "d" * 64},
])
def test_resume_rejects_coefficient_or_calibration_receipt_change(tmp_path, changes):
    objective = objectives()[2]
    run(tmp_path, objective, stop_after_epochs=1)
    before = (tmp_path / "last_epoch_state.pt").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        run(tmp_path, replace(objective, **changes), resume=True)
    assert (tmp_path / "last_epoch_state.pt").read_bytes() == before


def test_mixed_resume_rejects_different_initial_parameters(tmp_path):
    objective = objectives()[1]
    run(tmp_path, objective, stop_after_epochs=1)
    before = (tmp_path / "last_epoch_state.pt").read_bytes()
    with pytest.raises(ValueError, match="initialization"):
        run(tmp_path, objective, model_seed=32, resume=True)
    assert (tmp_path / "last_epoch_state.pt").read_bytes() == before


@pytest.mark.parametrize("case", [case for case in QuantityCase
                                   if case is not QuantityCase.B_LOG_ORIGINAL])
def test_mixed_objective_rejects_other_output_conditions_before_writing(tmp_path, case):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="original B output"):
        run(output, objectives()[1], case=case)
    assert not output.exists()


def test_both_selected_checkpoints_restore_their_recorded_validation_metric(tmp_path):
    objective = objectives()[2]
    summary = run(tmp_path, objective)
    for selector in ("raw_quantity_rmse", "legacy_time_loss"):
        selected = torch.load(tmp_path / f"best_{selector}_model.pt", weights_only=False)
        earliest = min(summary["history"], key=lambda row: row[selector])
        assert selected["best_epoch"] == earliest["epoch"]
        assert selected["condition"]["mixed_objective"] == objective.to_dict()
        model = make_model()
        model.load_state_dict(selected["model_state_dict"], strict=True)
        model.eval()
        with torch.no_grad():
            outputs = mixed_joint_causal_batch_objective(
                model, *tensors(), statistics=statistics(), objective=objective)
        value = (torch.square(outputs["pred_qty"].double() - outputs["true_qty"].double()).mean().sqrt()
                 if selector == "raw_quantity_rmse" else outputs["time_loss"].double().mean())
        assert float(value) == pytest.approx(selected["best_value"], rel=1e-6)


def test_mixed_selectors_keep_first_epoch_when_validation_ties(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *args, **kwargs: None)
    summary = run(tmp_path, objectives()[1])
    for selector in ("raw_quantity_rmse", "legacy_time_loss"):
        assert len({row[selector] for row in summary["history"]}) == 1
        assert summary["selectors"][selector]["best_epoch"] == 1
    resumed = run(tmp_path, objectives()[1], resume=True)
    assert resumed == summary


def test_mixed_budget_callback_counts_exact_updates_and_commits(tmp_path):
    stages = []
    summary = run(tmp_path, objectives()[1], budget_check=stages.append)
    assert stages.count("train_batch") == stages.count("validation_batch") == 6
    assert stages.count("before_epoch_commit") == 3
    assert summary["global_step"] == 6
