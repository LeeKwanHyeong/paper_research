from __future__ import annotations

import numpy as np

from paper.scripts.analyze_count_aware_tail_loss_audit import (
    audit_quantity,
    huber_gradient,
)


def test_huber_gradient_is_bounded() -> None:
    residual = np.asarray([-3.0, -0.5, 0.0, 0.5, 3.0])
    actual = huber_gradient(residual, delta=1.0)
    np.testing.assert_allclose(actual, [-1.0, -0.5, 0.0, 0.5, 1.0])


def test_audit_uses_train_quantiles_and_reports_finite_gradient_shares() -> None:
    quantity = np.arange(1.0, 10001.0)
    summary, rows = audit_quantity(quantity)

    assert summary["status"] == "continue"
    assert summary["quantiles"] == {"p90": 9000.0, "p95": 9500.0, "p99": 9900.0}
    assert summary["proposed_constants"]["tail_threshold"] == 9500.0
    assert summary["proposed_constants"]["normalization_scale"] == 9500.0
    assert summary["proposed_constants"]["clip_cap"] == 9900.0
    assert sum(row["count"] for row in rows) == quantity.size
    assert np.isclose(sum(row["sample_share"] for row in rows), 1.0)
    assert np.isclose(sum(row["log_mse_share"] for row in rows), 1.0)
    assert np.isclose(sum(row["log_location_gradient_share"] for row in rows), 1.0)


def test_audit_rejects_invalid_quantity() -> None:
    try:
        audit_quantity(np.asarray([1.0, np.nan, 3.0]))
    except ValueError as exc:
        assert "finite nonnegative" in str(exc)
    else:
        raise AssertionError("Expected invalid quantity to fail")
