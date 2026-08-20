from __future__ import annotations

import pytest

from paper.scripts.compare_count_aware_final_time_t1_integration import (
    evaluate_gate,
    percent_change,
)


def metrics() -> dict[str, float | int]:
    return {
        "joint_objective": 1.0,
        "time_nll": 0.8,
        "quantity_train_loss": 0.2,
        "qty_mae": 0.7,
        "qty_rmse": 1.7,
        "le_p95_qty_mae": 0.5,
        "gt_p99_qty_mae": 5.0,
        "best_epoch": 20,
        "completed_epochs": 60,
        "parameter_count": 90_000,
    }


def test_integrated_gate_accepts_bounded_regressions() -> None:
    reference = metrics()
    candidate = dict(reference)
    candidate.update(
        {
            "time_nll": 0.81,
            "qty_mae": 0.714,
            "qty_rmse": 1.734,
            "le_p95_qty_mae": 0.51,
            "gt_p99_qty_mae": 5.1,
        }
    )

    gate = evaluate_gate(reference, candidate)

    assert gate["status"] == "pass"
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("time_nll", 0.811, "time_nll_regression_at_most_0_01"),
        ("qty_mae", 0.715, "qty_mae_regression_at_most_2pct"),
        ("qty_rmse", 1.735, "qty_rmse_regression_at_most_2pct"),
        ("le_p95_qty_mae", 0.511, "le_p95_qty_mae_regression_at_most_2pct"),
        ("gt_p99_qty_mae", 5.11, "gt_p99_qty_mae_regression_at_most_2pct"),
    ],
)
def test_integrated_gate_rejects_each_safety_violation(
    field: str,
    value: float,
    failed_check: str,
) -> None:
    reference = metrics()
    candidate = dict(reference)
    candidate[field] = value

    gate = evaluate_gate(reference, candidate)

    assert gate["status"] == "fail"
    assert gate["checks"][failed_check] is False


def test_percent_change_rejects_zero_reference() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        percent_change(1.0, 0.0)
