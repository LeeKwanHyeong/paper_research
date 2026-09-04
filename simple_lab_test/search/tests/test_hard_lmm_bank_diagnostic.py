"""Guard the diagnostic's legacy time path against a misleading clamp."""
import pytest
import torch

from paper.scripts.run_hard_lmm_bank_diagnostic import time_intercept_effect


def test_legacy_upper_only_clamp_and_sum_before_clamp():
    base = torch.tensor([-100., 29., 35., 35.], dtype=torch.float64)
    contribution = torch.tensor([5., 3., -2., -10.], dtype=torch.float64)
    actual = time_intercept_effect(base, contribution, 30.)
    torch.testing.assert_close(actual, torch.tensor([5., 1., 0., -5.], dtype=torch.float64))


@pytest.mark.parametrize("limit", [0., -1., float("inf"), float("nan")])
def test_invalid_time_limit_fails(limit):
    with pytest.raises(ValueError):
        time_intercept_effect(torch.ones(1), torch.ones(1), limit)


def test_nonfinite_or_mismatched_time_input_fails():
    with pytest.raises(ValueError):
        time_intercept_effect(torch.ones(1), torch.ones(2), 30.)
    with pytest.raises(ValueError):
        time_intercept_effect(torch.tensor([float("nan")]), torch.ones(1), 30.)
