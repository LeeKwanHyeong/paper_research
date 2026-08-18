from __future__ import annotations

import pytest

from paper.scripts.compare_titantpp_memory_backbone_screening import (
    CONTROL_BACKBONE,
    evaluate_gate,
    select_backbone,
)


def metrics(**updates: float | int) -> dict[str, float | int]:
    base: dict[str, float | int] = {
        "joint_objective": -3.0,
        "qty_mae": 1.0,
        "qty_rmse": 2.0,
        "time_nll": -3.5,
        "le_p95_qty_mae": 0.8,
        "parameter_count": 90_000,
        "best_epoch": 100,
        "completed_epochs": 140,
    }
    base.update(updates)
    return base


def test_gate_accepts_mae_improvement_at_frozen_boundary() -> None:
    gate = evaluate_gate(
        metrics(),
        metrics(qty_mae=0.95, qty_rmse=2.02, le_p95_qty_mae=0.816, time_nll=-3.49),
    )

    assert gate["status"] == "pass"
    assert all(gate["checks"].values())


def test_gate_accepts_rmse_improvement_when_mae_does_not_improve() -> None:
    gate = evaluate_gate(
        metrics(),
        metrics(qty_mae=1.01, qty_rmse=1.9),
    )

    assert gate["status"] == "pass"
    assert gate["deltas"]["overall_rmse_improvement_pct"] == pytest.approx(5.0)


def test_gate_rejects_body_or_time_regression() -> None:
    body_failure = evaluate_gate(
        metrics(),
        metrics(qty_mae=0.9, le_p95_qty_mae=0.817),
    )
    time_failure = evaluate_gate(
        metrics(),
        metrics(qty_mae=0.9, time_nll=-3.489),
    )

    assert body_failure["status"] == "fail"
    assert not body_failure["checks"]["le_p95_mae_regression_at_most_2pct"]
    assert time_failure["status"] == "fail"
    assert not time_failure["checks"]["time_nll_regression_at_most_0_01"]


def test_selection_prefers_lowest_mae_among_passing_candidates() -> None:
    values = {
        CONTROL_BACKBONE: metrics(),
        "titantpp_no_memory": metrics(qty_mae=0.94, qty_rmse=1.95),
        "titantpp_gated_soft_memory": metrics(qty_mae=0.92, qty_rmse=1.98),
        "titantpp_surprise_memory": metrics(qty_mae=0.93, qty_rmse=1.90),
    }
    gates = {
        backbone: evaluate_gate(values[CONTROL_BACKBONE], values[backbone])
        for backbone in values
        if backbone != CONTROL_BACKBONE
    }

    assert select_backbone(values, gates) == "titantpp_gated_soft_memory"


def test_selection_retains_control_when_no_candidate_passes() -> None:
    values = {
        CONTROL_BACKBONE: metrics(),
        "titantpp_no_memory": metrics(qty_mae=0.99),
        "titantpp_gated_soft_memory": metrics(qty_mae=0.98),
        "titantpp_surprise_memory": metrics(qty_mae=0.97),
    }
    gates = {
        backbone: evaluate_gate(values[CONTROL_BACKBONE], values[backbone])
        for backbone in values
        if backbone != CONTROL_BACKBONE
    }

    assert select_backbone(values, gates) == CONTROL_BACKBONE
