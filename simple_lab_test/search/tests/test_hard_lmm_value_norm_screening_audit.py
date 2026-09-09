from __future__ import annotations

import pytest

from paper.scripts import audit_hard_lmm_value_norm_screening as audit


def _history(values: list[float]) -> list[dict[str, float | int]]:
    return [
        {"epoch": epoch, "val_qty_rmse": value}
        for epoch, value in enumerate(values, start=1)
    ]


def test_e300_patience_exhausted_at_budget_is_normal_completion() -> None:
    values = [2.0 - epoch * 0.001 for epoch in range(1, 261)]
    values.extend([values[-1]] * 40)

    result = audit.validate_history_and_stop(
        _history(values),
        expected_epochs=300,
        completed_epochs=300,
        stopped_early=False,
        minimum_epochs=40,
        patience=40,
    )

    assert result["selected_epoch"] == 260
    assert result["first_early_stop_epoch"] == 300
    assert result["stopped_early"] is False


def test_e300_first_stop_before_budget_requires_early_stop_flag() -> None:
    values = [1.0] * 41

    result = audit.validate_history_and_stop(
        _history(values),
        expected_epochs=300,
        completed_epochs=41,
        stopped_early=True,
        minimum_epochs=40,
        patience=40,
    )

    assert result["selected_epoch"] == 1
    assert result["first_early_stop_epoch"] == 41

    with pytest.raises(ValueError, match="first valid stop epoch"):
        audit.validate_history_and_stop(
            _history(values),
            expected_epochs=300,
            completed_epochs=41,
            stopped_early=False,
            minimum_epochs=40,
            patience=40,
        )


def test_e1_is_never_used_as_a_performance_gate() -> None:
    baseline = {
        "raw_rmse": 5.0,
        "overall_mae": 4.0,
        "body_mae": 3.0,
        "gt_p99_mae": 20.0,
        "legacy_time_nll": 2.0,
    }
    result = audit.evaluate_stage1_gate(
        dict(baseline), baseline, expected_epochs=1
    )

    assert result["status"] == "not_applicable_e1"
    assert result["applicable"] is False
    assert result["affects_followup"] is False


def test_stage1_gate_uses_strict_rmse_and_inclusive_guardrails() -> None:
    baseline = {
        "raw_rmse": 5.0,
        "overall_mae": 4.0,
        "body_mae": 3.0,
        "gt_p99_mae": 20.0,
        "legacy_time_nll": 2.0,
    }
    passing = {
        "raw_rmse": 4.999999999,
        "overall_mae": 4.04,
        "body_mae": 3.06,
        "gt_p99_mae": 20.4,
        "legacy_time_nll": 2.01,
    }

    passed = audit.evaluate_stage1_gate(passing, baseline, expected_epochs=300)
    assert passed["status"] == "passed"
    assert all(passed["checks"].values())

    passing["raw_rmse"] = baseline["raw_rmse"]
    failed = audit.evaluate_stage1_gate(passing, baseline, expected_epochs=300)
    assert failed["status"] == "failed"
    assert failed["checks"]["raw_rmse_strictly_better_than_B"] is False


def test_checkpoint_reevaluation_tolerance_matches_training_runner() -> None:
    audit._require_reevaluated_metric_equal(
        5.0 + 5e-9,
        5.0,
        label="within runner tolerance",
    )

    with pytest.raises(ValueError, match="replay drift"):
        audit._require_reevaluated_metric_equal(
            5.0 + 2e-8,
            5.0,
            label="outside runner tolerance",
        )
