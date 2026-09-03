import pytest

from paper.scripts.summarize_hard_lmm_probe_mechanisms import check_close


def test_reporting_only_permits_rounding_not_substantive_differences():
    check_close(1., 1.00000001, "rounding")
    with pytest.raises(AssertionError):
        check_close(1., 1.01, "substantive")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reporting_rejects_nonfinite_comparisons(value):
    with pytest.raises(AssertionError):
        check_close(value, 1., "nonfinite")
