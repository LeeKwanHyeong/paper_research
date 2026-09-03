import pytest

from paper.scripts.validate_hard_lmm_frozen_probe import close, finite_json


def test_event_reconciliation_allows_only_float_rounding():
    close(6.003668946739279, 6.003668948978827, "FP32 SIMD rounding")
    with pytest.raises(AssertionError):
        close(6.0, 6.01, "substantive metric mismatch")


@pytest.mark.parametrize("number", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_nested_metric_rejected(number):
    with pytest.raises(AssertionError):
        finite_json({"datasets": [{"metrics": {"loss": number}}]})


def test_explicit_empty_strata_are_not_nonfinite_errors():
    finite_json({"status": "empty", "count": 0, "metric": None})
