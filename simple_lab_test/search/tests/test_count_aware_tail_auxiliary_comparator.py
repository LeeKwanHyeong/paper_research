from __future__ import annotations

from paper.scripts.compare_count_aware_tail_auxiliary_screening import evaluate_gate


def metrics(**updates: float) -> dict[str, float]:
    base = {
        "qty_mae": 1.0,
        "qty_rmse": 2.0,
        "time_nll": -3.5,
        "le_p95_qty_mae": 0.8,
        "gt_p99_qty_mae": 10.0,
    }
    base.update(updates)
    return base


def test_gate_accepts_rmse_improvement_with_guardrails() -> None:
    gate = evaluate_gate(
        metrics(),
        metrics(qty_mae=1.02, qty_rmse=1.90, le_p95_qty_mae=0.816, time_nll=-3.49),
    )

    assert gate["status"] == "pass"
    assert all(gate["checks"].values())


def test_gate_accepts_tail_improvement_without_rmse_improvement() -> None:
    gate = evaluate_gate(
        metrics(),
        metrics(qty_rmse=2.02, gt_p99_qty_mae=9.5),
    )

    assert gate["status"] == "pass"
    assert gate["deltas"]["gt_p99_mae_improvement_pct"] == 5.0


def test_gate_rejects_body_or_time_regression() -> None:
    body_failure = evaluate_gate(
        metrics(),
        metrics(qty_rmse=1.8, le_p95_qty_mae=0.817),
    )
    time_failure = evaluate_gate(
        metrics(),
        metrics(qty_rmse=1.8, time_nll=-3.489),
    )

    assert body_failure["status"] == "fail"
    assert not body_failure["checks"]["le_p95_mae_regression_at_most_2pct"]
    assert time_failure["status"] == "fail"
    assert not time_failure["checks"]["time_nll_regression_at_most_0_01"]
