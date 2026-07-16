import pytest

from simple_lab_test.search.compare_titantpp_v4_taxi_validation import evaluate_pair


def control_metrics() -> dict[str, float]:
    return {
        "val_nll": 5.0,
        "val_nll_marker": 1.0,
        "val_nll_time": 4.0,
        "dt_mae": 10.0,
        "mark_acc": 0.6,
        "qty_mae": 20.0,
    }


def test_pair_gate_accepts_exact_threshold_boundaries() -> None:
    candidate = {
        "val_nll": 5.025,
        "val_nll_marker": 1.02,
        "val_nll_time": 3.98,
        "dt_mae": 10.1,
        "mark_acc": 0.5975,
        "qty_mae": 21.0,
    }

    gate = evaluate_pair(control_metrics(), candidate)

    assert gate["time_nll_improvement_pct"] == pytest.approx(0.5)
    assert gate["total_nll_regression_pct"] == pytest.approx(0.5)
    assert gate["mark_accuracy_delta_pp"] == pytest.approx(-0.25)
    assert gate["passed"] is True
    assert all(gate["checks"].values())


@pytest.mark.parametrize(
    ("metric", "value", "failed_check"),
    [
        ("val_nll_time", 3.981, "time_nll_improvement"),
        ("val_nll", 5.026, "total_nll_safety"),
        ("dt_mae", 10.101, "dt_mae_safety"),
        ("val_nll_marker", 1.021, "marker_nll_safety"),
        ("mark_acc", 0.5974, "mark_accuracy_safety"),
        ("qty_mae", 21.01, "qty_mae_safety"),
    ],
)
def test_pair_gate_rejects_each_guardrail(metric: str, value: float, failed_check: str) -> None:
    candidate = {
        "val_nll": 4.9,
        "val_nll_marker": 0.99,
        "val_nll_time": 3.9,
        "dt_mae": 9.9,
        "mark_acc": 0.61,
        "qty_mae": 19.0,
    }
    candidate[metric] = value

    gate = evaluate_pair(control_metrics(), candidate)

    assert gate["passed"] is False
    assert gate["checks"][failed_check] is False
