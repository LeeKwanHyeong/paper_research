from __future__ import annotations

import math

import pytest
import torch

from models.TPPs.positive_integer_time import positive_integer_log_mass


DTYPE = torch.float64


def test_ordinary_masses_match_independent_distribution_oracle() -> None:
    target = torch.tensor([1, 2, 3, 8, 30], dtype=DTYPE)
    location = torch.tensor([-0.5, 0.0, 0.7, 1.2, 2.0], dtype=DTYPE)
    scale = torch.tensor([0.6, 1.2, 0.9, 1.4, 1.1], dtype=DTYPE)
    distribution = torch.distributions.LogNormal(location + math.log(3.0), scale)
    expected = distribution.cdf(target + 0.5) - distribution.cdf(target - 0.5)
    expected[0] = distribution.cdf(target + 0.5)[0]
    expected[-1] = 1.0 - distribution.cdf(target - 0.5)[-1]
    actual = positive_integer_log_mass(target, location, scale, time_scale=3.0, top_code=30)
    torch.testing.assert_close(actual, expected.log(), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("top_code", [2, 30, 100])
@pytest.mark.parametrize("location,scale", [(-5.0, 0.4), (0.0, 1.0), (6.0, 2.0)])
def test_top_coded_partition_is_normalized_and_has_zero_total_gradient(
    top_code: int, location: float, scale: float
) -> None:
    target = torch.arange(1, top_code + 1, dtype=DTYPE)
    mu = torch.tensor(location, dtype=DTYPE, requires_grad=True)
    sigma = torch.tensor(scale, dtype=DTYPE, requires_grad=True)
    total = positive_integer_log_mass(
        target, mu.expand_as(target), sigma.expand_as(target), time_scale=2.0, top_code=top_code
    ).exp().sum()
    torch.testing.assert_close(total, torch.ones_like(total), atol=2e-14, rtol=0)
    for gradient in torch.autograd.grad(total, (mu, sigma)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=2e-13, rtol=0)


def test_uncensored_bins_plus_remaining_tail_partition_all_mass() -> None:
    target = torch.arange(1, 101, dtype=DTYPE)
    mu = torch.full_like(target, 1.2)
    sigma = torch.full_like(target, 0.9)
    mass = positive_integer_log_mass(target, mu, sigma, time_scale=4.0).exp().sum()
    remainder = torch.special.log_ndtr(
        -(torch.tensor(math.log(100.5 / 4.0), dtype=DTYPE) - mu[0]) / sigma[0]
    ).exp()
    torch.testing.assert_close(mass + remainder, torch.tensor(1.0, dtype=DTYPE), atol=2e-14, rtol=0)


def test_first_bin_includes_mass_below_half_not_old_interval_audit() -> None:
    target = torch.tensor([1.0], dtype=DTYPE)
    mu, sigma = torch.zeros_like(target), torch.ones_like(target)
    actual = positive_integer_log_mass(target, mu, sigma, time_scale=1.0).exp()
    distribution = torch.distributions.LogNormal(mu, sigma)
    expected = distribution.cdf(torch.tensor([1.5], dtype=DTYPE))
    old_lower_half_mass = expected - distribution.cdf(torch.tensor([0.5], dtype=DTYPE))
    torch.testing.assert_close(actual, expected)
    assert (actual - old_lower_half_mass).item() > 0.2


@pytest.mark.parametrize("z", [-80.0, -40.0, 0.0, 40.0, 80.0])
def test_first_regular_and_top_extreme_tails_have_finite_gradients(z: float) -> None:
    target = torch.tensor([1.0, 2.0, 30.0], dtype=DTYPE)
    mu = torch.tensor([math.log(1.5) - z, math.log(1.5) - z, math.log(29.5) - z], dtype=DTYPE, requires_grad=True)
    sigma = torch.ones_like(mu, requires_grad=True)
    output = positive_integer_log_mass(target, mu, sigma, time_scale=1.0, top_code=30)
    assert torch.isfinite(output).all()
    gradients = torch.autograd.grad(output.sum(), (mu, sigma))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert not torch.equal(gradients[0], torch.zeros_like(mu))


@pytest.mark.parametrize("target,location,scale", [(1e12, math.log(1e12), 1.0), (2.0, 0.0, 1e12), (1e8, math.log(1e8) - 40, 1.0)])
def test_narrow_log_interval_does_not_poison_backward(target: float, location: float, scale: float) -> None:
    observed = torch.tensor([target], dtype=DTYPE)
    mu = torch.tensor([location], dtype=DTYPE, requires_grad=True)
    sigma = torch.tensor([scale], dtype=DTYPE, requires_grad=True)
    output = positive_integer_log_mass(observed, mu, sigma, time_scale=1.0)
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(gradient).all() for gradient in torch.autograd.grad(output.sum(), (mu, sigma)))


def test_gradcheck_and_both_branch_boundaries() -> None:
    normal = torch.distributions.Normal(torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE))
    quarter_quantile = normal.icdf(torch.tensor(0.25, dtype=DTYPE)).item()
    target = torch.tensor([1.0, 2.0, 2.0, 3.0, 30.0], dtype=DTYPE)
    mu = torch.tensor([0.1, math.log(1.5), math.log(2.5), -0.5, 1.0], dtype=DTYPE, requires_grad=True)
    sigma = torch.tensor([0.7, 1.0, math.log(2.5 / 1.5) / -quarter_quantile, 1.2, 1.4], dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda location, scale: positive_integer_log_mass(target, location, scale, time_scale=1.0, top_code=30),
        (mu, sigma), eps=1e-6, atol=1e-6, rtol=1e-4
    )


def test_float32_parameters_keep_gradients_inputs_and_rng_unchanged() -> None:
    target = torch.tensor([1, 2, 30])
    mu = torch.tensor([0.0, 0.5, 1.0], requires_grad=True)
    sigma = torch.tensor([0.7, 1.0, 1.4], requires_grad=True)
    original = [value.clone() for value in (target, mu, sigma)]
    rng = torch.random.get_rng_state().clone()
    result = positive_integer_log_mass(target, mu, sigma, time_scale=2.0, top_code=30)
    assert result.dtype == DTYPE
    result.sum().backward()
    assert mu.grad.dtype == sigma.grad.dtype == torch.float32
    assert torch.isfinite(mu.grad).all() and torch.isfinite(sigma.grad).all()
    assert mu.grad.abs().sum() > 0 and sigma.grad.abs().sum() > 0
    assert torch.equal(rng, torch.random.get_rng_state())
    for value, before in zip((target, mu, sigma), original):
        assert torch.equal(value, before)


@pytest.mark.parametrize("bad_target", [0.0, -1.0, 1.0 + 1e-10, float("nan"), float("inf")])
def test_invalid_target_rejected(bad_target: float) -> None:
    with pytest.raises(ValueError, match="targets"):
        positive_integer_log_mass(torch.tensor([bad_target], dtype=DTYPE), torch.zeros(1), torch.ones(1), time_scale=1.0)


@pytest.mark.parametrize("field,value", [("location", float("nan")), ("location", float("inf")), ("scale", float("nan")), ("scale", float("inf")), ("scale", 0.0), ("scale", -1.0)])
def test_invalid_distribution_parameters_rejected(field: str, value: float) -> None:
    args = {"target_dt": torch.ones(1), "location": torch.zeros(1), "scale": torch.ones(1), "time_scale": 1.0}
    args[field] = torch.tensor([value])
    with pytest.raises(ValueError):
        positive_integer_log_mass(**args)


@pytest.mark.parametrize("time_scale", [0.0, -1.0, float("nan"), float("inf"), True])
def test_invalid_time_scale_rejected(time_scale: float) -> None:
    with pytest.raises(ValueError, match="time_scale"):
        positive_integer_log_mass(torch.ones(1), torch.zeros(1), torch.ones(1), time_scale=time_scale)


@pytest.mark.parametrize("top_code", [1, 0, -2, 2.5, True])
def test_invalid_top_code_rejected(top_code: int) -> None:
    with pytest.raises(ValueError, match="top_code"):
        positive_integer_log_mass(torch.ones(1), torch.zeros(1), torch.ones(1), time_scale=1.0, top_code=top_code)


def test_top_code_and_shape_contracts_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds top_code"):
        positive_integer_log_mass(torch.tensor([31.0]), torch.zeros(1), torch.ones(1), time_scale=1.0, top_code=30)
    with pytest.raises(ValueError, match="rank one"):
        positive_integer_log_mass(torch.ones(1, 1), torch.zeros(1), torch.ones(1), time_scale=1.0)
    with pytest.raises(ValueError, match="shapes"):
        positive_integer_log_mass(torch.ones(2), torch.zeros(1), torch.ones(2), time_scale=1.0)


@pytest.mark.parametrize("location,scale", [(100.0, 1e-300), (0.0, 1e300)])
def test_unrepresentable_probability_raises_instead_of_clipping(location: float, scale: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        positive_integer_log_mass(torch.tensor([2.0], dtype=DTYPE), torch.tensor([location], dtype=DTYPE), torch.tensor([scale], dtype=DTYPE), time_scale=1.0)


@pytest.mark.parametrize("target", [1.0, 2.0, 30.0])
def test_overflowing_boundary_rejected_even_when_cdf_would_round_to_one(target: float) -> None:
    with pytest.raises(ValueError, match="standardized duration boundary"):
        positive_integer_log_mass(
            torch.tensor([target], dtype=DTYPE),
            torch.tensor([-1e308], dtype=DTYPE),
            torch.tensor([1e-308], dtype=DTYPE),
            time_scale=1.0,
            top_code=30,
        )


def test_empty_batch_keeps_the_gradient_connection() -> None:
    mu = torch.empty(0, requires_grad=True)
    sigma = torch.empty(0, requires_grad=True)
    result = positive_integer_log_mass(torch.empty(0), mu, sigma, time_scale=1.0)
    assert result.shape == (0,) and result.dtype == DTYPE
    result.sum().backward()
    assert mu.grad is not None and sigma.grad is not None
    assert mu.grad.numel() == sigma.grad.numel() == 0


def test_ordinary_outputs_match_existing_frozen_head_helper() -> None:
    from models.TPPs.CountAwareTPP import CountAwareRMTPP, TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import positive_integer_time_log_likelihood

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(717)
        model = CountAwareRMTPP(8, train_log_mean=1.0, time_head_mode=TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION, time_scale=3.0, time_initial_location=0.2, time_initial_scale=0.9)
        hidden = torch.randn(5, 8)
    target = torch.tensor([1.0, 2.0, 3.0, 8.0, 30.0])
    location, scale, _ = model._lognormal_time_terms(hidden, target)
    old = positive_integer_time_log_likelihood(model, hidden, target, is_right_censored=target == 30)
    new = positive_integer_log_mass(target, location, scale, time_scale=model.time_scale, top_code=30)
    torch.testing.assert_close(new, old, atol=1e-13, rtol=1e-13)
