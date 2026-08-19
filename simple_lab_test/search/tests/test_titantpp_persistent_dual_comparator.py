from __future__ import annotations

import pytest

from paper.scripts.compare_titantpp_persistent_dual_screening import (
    CANDIDATE_BACKBONES,
    CONTROL_BACKBONE,
    evaluate_gate,
    select_backbone,
)


def metrics(**updates: float | int) -> dict[str, float | int]:
    base: dict[str, float | int] = {
        "joint_objective": 0.2,
        "qty_mae": 1.0,
        "qty_rmse": 2.0,
        "time_nll": -1.5,
        "le_p95_qty_mae": 0.8,
        "parameter_count": 100_000,
        "best_epoch": 100,
        "completed_epochs": 140,
    }
    base.update(updates)
    return base


def test_gate_accepts_quantity_improvement_with_bounded_regressions() -> None:
    gate = evaluate_gate(
        metrics(),
        metrics(
            qty_mae=0.95,
            qty_rmse=2.02,
            le_p95_qty_mae=0.816,
            time_nll=-1.49,
        ),
    )

    assert gate["status"] == "pass"
    assert all(gate["checks"].values())


def test_gate_accepts_rmse_improvement_without_mae_improvement() -> None:
    gate = evaluate_gate(metrics(), metrics(qty_mae=1.01, qty_rmse=1.9))

    assert gate["status"] == "pass"
    assert gate["deltas"]["overall_rmse_improvement_pct"] == pytest.approx(5.0)


def test_gate_rejects_body_or_scaled_time_regression() -> None:
    body_failure = evaluate_gate(
        metrics(),
        metrics(qty_mae=0.9, le_p95_qty_mae=0.817),
    )
    time_failure = evaluate_gate(
        metrics(),
        metrics(qty_mae=0.9, time_nll=-1.489),
    )

    assert body_failure["status"] == "fail"
    assert not body_failure["checks"]["le_p95_mae_regression_at_most_2pct"]
    assert time_failure["status"] == "fail"
    assert not time_failure["checks"]["time_nll_regression_at_most_0_01"]


def test_selection_uses_only_passing_m2_m3_candidates() -> None:
    values = {
        CONTROL_BACKBONE: metrics(),
        CANDIDATE_BACKBONES[0]: metrics(qty_mae=0.94),
        CANDIDATE_BACKBONES[1]: metrics(qty_mae=0.92),
        CANDIDATE_BACKBONES[2]: metrics(qty_mae=0.93),
    }
    gates = {
        backbone: evaluate_gate(values[CONTROL_BACKBONE], values[backbone])
        for backbone in CANDIDATE_BACKBONES
    }

    assert select_backbone(values, gates) == CANDIDATE_BACKBONES[1]


def test_selection_retains_hard_lmm_when_no_candidate_passes() -> None:
    values = {
        CONTROL_BACKBONE: metrics(),
        **{
            backbone: metrics(qty_mae=0.99)
            for backbone in CANDIDATE_BACKBONES
        },
    }
    gates = {
        backbone: evaluate_gate(values[CONTROL_BACKBONE], values[backbone])
        for backbone in CANDIDATE_BACKBONES
    }

    assert select_backbone(values, gates) == CONTROL_BACKBONE
