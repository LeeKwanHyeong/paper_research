import math

import pytest

from simple_lab_test.search.reevaluate_titantpp_validation import (
    assert_finite_applicable_metrics,
    json_safe_metrics,
    undefined_validation_metrics,
)


@pytest.mark.parametrize(
    "decoder_mode,expected",
    [
        (
            "mark_residual",
            {"val_magnitude_loss", "val_log_qty_aux_loss"},
        ),
        (
            "direct_log_qty",
            {"value_mae", "val_value_loss"},
        ),
        (
            "direct_raw_qty",
            {"value_mae", "val_value_loss"},
        ),
    ],
)
def test_undefined_validation_metrics_follow_decoder_contract(
    decoder_mode,
    expected,
) -> None:
    assert undefined_validation_metrics(decoder_mode) == expected


def test_legacy_inactive_metrics_export_as_null_without_masking_active_metrics() -> None:
    metrics = {
        "val_nll": 5.0,
        "val_magnitude_loss": math.nan,
        "val_log_qty_aux_loss": math.nan,
        "_total": 10,
    }
    undefined = undefined_validation_metrics("mark_residual")

    assert_finite_applicable_metrics(metrics, undefined)
    safe = json_safe_metrics(metrics, undefined)

    assert safe["val_nll"] == 5.0
    assert safe["val_magnitude_loss"] is None
    assert safe["val_log_qty_aux_loss"] is None


def test_active_nonfinite_metric_still_fails() -> None:
    with pytest.raises(FloatingPointError, match="val_nll"):
        assert_finite_applicable_metrics(
            {"val_nll": math.nan},
            undefined_validation_metrics("mark_residual"),
        )


def test_unknown_decoder_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unsupported quantity decoder"):
        undefined_validation_metrics("unknown")
