from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    TIME_HEAD_LOGNORMAL_MODES,
    TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
    TIME_HEAD_MODES,
    CountAwareRMTPP,
    inverse_softplus,
)
from paper.scripts.run_count_aware_tpp_backbone_control import (
    JOINT_TRAINING_TIME_HEAD_MODES,
)


TIME_KWARGS = {
    "time_head_mode": TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
    "time_scale": 3.0,
    "time_initial_location": -0.1,
    "time_initial_scale": 0.6,
    "time_sigma_floor": 1e-3,
}


def build_heteroscedastic_rmtpp(*, hidden_dim: int = 8) -> CountAwareRMTPP:
    torch.manual_seed(20260906)
    return CountAwareRMTPP(
        hidden_dim,
        train_log_mean=1.5,
        **TIME_KWARGS,
    )


@pytest.mark.parametrize("backbone", ["rmtpp", "thp", "titantpp"])
def test_heteroscedastic_contract_is_shared_by_primary_backbones(
    backbone: str,
) -> None:
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
        **TIME_KWARGS,
    )

    assert TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION in TIME_HEAD_MODES
    assert TIME_HEAD_LOGNORMAL_MODES == (
        TIME_HEAD_MODE_LOGNORMAL_DURATION,
        TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
    )
    assert model.time_head_contract() == metadata["time_head"]
    assert metadata["time_head"] == {
        "mode": TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
        "density_family": "heteroscedastic_lognormal_on_scaled_duration",
        "time_scale": 3.0,
        "time_initial_location": -0.1,
        "time_initial_scale": 0.6,
        "time_sigma_floor": 1e-3,
        "time_location_transform": "identity",
        "time_scale_conditioning": "linear_hidden",
        "time_scale_transform": "softplus_plus_floor",
        "time_scale_weight_initialization": "zeros",
        "slope_parameterized": False,
        "jacobian_correction": True,
        "wd_clamp": 0.0,
    }
    assert tuple(name for name, _ in model.time_head_named_parameters()) == (
        "v_t.weight",
        "b_t",
        "w_raw",
        "time_scale_weight.weight",
    )
    assert torch.count_nonzero(model.time_scale_weight.weight) == 0


def test_candidate_mode_is_reserved_for_the_censor_aware_frozen_runner() -> None:
    assert (
        TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION
        not in JOINT_TRAINING_TIME_HEAD_MODES
    )


def test_train_initial_mean_std_and_zero_scale_weight_match_homoscedastic_head() -> None:
    common = {
        "hidden_dim": 8,
        "train_log_mean": 1.5,
        "time_scale": 3.0,
        "time_initial_location": -0.1,
        "time_initial_scale": 0.6,
        "time_sigma_floor": 1e-3,
    }
    torch.manual_seed(71)
    homoscedastic = CountAwareRMTPP(
        **common,
        time_head_mode=TIME_HEAD_MODE_LOGNORMAL_DURATION,
    ).eval()
    homoscedastic_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(71)
    heteroscedastic = CountAwareRMTPP(
        **common,
        time_head_mode=TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
    ).eval()
    heteroscedastic_rng = torch.random.get_rng_state().clone()

    assert torch.equal(homoscedastic_rng, heteroscedastic_rng)
    assert set(heteroscedastic.state_dict()) - set(homoscedastic.state_dict()) == {
        "time_scale_weight.weight"
    }
    for name, value in homoscedastic.state_dict().items():
        assert torch.equal(value, heteroscedastic.state_dict()[name]), name
    assert not hasattr(homoscedastic, "time_scale_weight")
    assert tuple(name for name, _ in homoscedastic.time_head_named_parameters()) == (
        "v_t.weight",
        "b_t",
        "w_raw",
    )

    hidden = torch.randn(5, common["hidden_dim"])
    delta_t = torch.tensor([0.25, 1.0, 3.0, 12.0, 100.0])
    expected_scale = torch.full((5,), common["time_initial_scale"])
    assert torch.allclose(
        heteroscedastic.positive_time_sigma(hidden),
        expected_scale,
        atol=1e-7,
        rtol=0.0,
    )
    heteroscedastic_density = heteroscedastic.log_f_dt(hidden, delta_t)
    heteroscedastic_survival = heteroscedastic.log_survival_dt(hidden, delta_t)
    heteroscedastic_median = heteroscedastic.predict_time_median(hidden)
    assert heteroscedastic_density.dtype == torch.float64
    assert heteroscedastic_survival.dtype == torch.float64
    assert heteroscedastic_median.dtype == torch.float64
    assert torch.allclose(
        homoscedastic.log_f_dt(hidden, delta_t).to(torch.float64),
        heteroscedastic_density,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        homoscedastic.log_survival_dt(hidden, delta_t).to(torch.float64),
        heteroscedastic_survival,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        homoscedastic.predict_time_median(hidden).to(torch.float64),
        heteroscedastic_median,
        atol=1e-6,
        rtol=1e-6,
    )


def test_location_and_scale_are_independently_conditioned_on_hidden() -> None:
    model = build_heteroscedastic_rmtpp().double()
    with torch.no_grad():
        model.v_t.weight.zero_()
        model.v_t.weight[0, 0] = 0.4
        model.b_t.fill_(-0.2)
        model.time_scale_weight.weight.zero_()
        model.time_scale_weight.weight[0, 1] = 0.7
        model.w_raw.fill_(inverse_softplus(0.5 - model.time_sigma_floor))
    hidden = torch.zeros(3, model.hidden_dim, dtype=torch.float64)
    hidden[:, 0] = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    hidden[:, 1] = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)

    expected_location = 0.4 * hidden[:, 0] - 0.2
    expected_scale = model.time_sigma_floor + F.softplus(
        0.7 * hidden[:, 1] + model.w_raw
    )

    assert torch.equal(model.time_location(hidden), expected_location)
    assert torch.equal(model.positive_time_sigma(hidden), expected_scale)
    assert torch.unique(model.time_location(hidden)).numel() == 3
    assert torch.unique(model.positive_time_sigma(hidden)).numel() == 3


def test_density_and_survival_match_float64_closed_forms_in_original_unit() -> None:
    model = build_heteroscedastic_rmtpp().double()
    with torch.no_grad():
        model.time_scale_weight.weight.copy_(
            torch.linspace(-0.15, 0.2, model.hidden_dim).unsqueeze(0)
        )
    hidden = torch.randn(5, model.hidden_dim, dtype=torch.float64)
    delta_t = torch.tensor(
        [0.125, 1.0, 3.0, 16.0, 1.0e4],
        dtype=torch.float64,
    )

    location = model.v_t(hidden).squeeze(-1) + model.b_t
    scale = model.time_sigma_floor + F.softplus(
        model.time_scale_weight(hidden).squeeze(-1) + model.w_raw
    )
    log_dt = torch.log(delta_t)
    standardized = (
        log_dt - math.log(model.time_scale) - location
    ) / scale
    expected_density = (
        -0.5 * torch.square(standardized)
        - torch.log(scale)
        - log_dt
        - 0.5 * math.log(2.0 * math.pi)
    )
    expected_survival = torch.special.log_ndtr(-standardized)

    actual_density = model.log_f_dt(hidden, delta_t)
    actual_survival = model.log_survival_dt(hidden, delta_t)
    assert actual_density.dtype == torch.float64
    assert actual_survival.dtype == torch.float64
    assert torch.allclose(actual_density, expected_density, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        actual_survival,
        expected_survival,
        atol=1e-12,
        rtol=1e-12,
    )


def test_float32_inputs_use_float64_likelihood_affines_and_outputs() -> None:
    model = build_heteroscedastic_rmtpp()
    with torch.no_grad():
        model.time_scale_weight.weight.copy_(
            torch.linspace(-0.15, 0.2, model.hidden_dim).unsqueeze(0)
        )
    hidden = torch.randn(5, model.hidden_dim, dtype=torch.float32)
    delta_t = torch.tensor(
        [0.125, 1.0, 3.0, 16.0, 1.0e8],
        dtype=torch.float32,
    )

    hidden_64 = hidden.to(torch.float64)
    location = F.linear(
        hidden_64,
        model.v_t.weight.to(torch.float64),
    ).squeeze(-1) + model.b_t.to(torch.float64)
    raw_scale = F.linear(
        hidden_64,
        model.time_scale_weight.weight.to(torch.float64),
    ).squeeze(-1) + model.w_raw.to(torch.float64)
    scale = model.time_sigma_floor + F.softplus(raw_scale)
    log_dt = torch.log(delta_t.to(torch.float64))
    standardized = (
        log_dt - math.log(model.time_scale) - location
    ) / scale
    expected_density = (
        -0.5 * torch.square(standardized)
        - torch.log(scale)
        - log_dt
        - 0.5 * math.log(2.0 * math.pi)
    )
    expected_survival = torch.special.log_ndtr(-standardized)

    actual_density = model.log_f_dt(hidden, delta_t)
    actual_survival = model.log_survival_dt(hidden, delta_t)
    actual_median = model.predict_time_median(hidden)
    assert actual_density.dtype == torch.float64
    assert actual_survival.dtype == torch.float64
    assert actual_median.dtype == torch.float64
    assert torch.equal(actual_density, expected_density)
    assert torch.equal(actual_survival, expected_survival)
    assert torch.equal(actual_median, model.time_scale * torch.exp(location))


def test_each_conditional_density_integrates_to_one_in_original_time_unit() -> None:
    model = build_heteroscedastic_rmtpp().double()
    with torch.no_grad():
        model.v_t.weight.zero_()
        model.b_t.fill_(0.15)
        model.w_raw.fill_(inverse_softplus(0.55 - model.time_sigma_floor))
        model.time_scale_weight.weight.zero_()
        model.time_scale_weight.weight[0, 0] = 0.2
    grid = torch.logspace(-5.0, 5.0, 80_001, dtype=torch.float64)

    for first_hidden_value in (-1.0, 1.0):
        hidden = torch.zeros(grid.numel(), model.hidden_dim, dtype=torch.float64)
        hidden[:, 0] = first_hidden_value
        density = torch.exp(model.log_f_dt(hidden, grid))
        integral = torch.trapezoid(density, grid)
        assert torch.isclose(
            integral,
            integral.new_tensor(1.0),
            atol=2e-6,
            rtol=0.0,
        )


def test_survival_at_conditional_median_is_one_half() -> None:
    model = build_heteroscedastic_rmtpp().double()
    with torch.no_grad():
        model.time_scale_weight.weight.copy_(
            torch.linspace(-0.1, 0.1, model.hidden_dim).unsqueeze(0)
        )
    hidden = torch.randn(7, model.hidden_dim, dtype=torch.float64)

    median = model.predict_time_median(hidden)
    log_survival = model.log_survival_dt(hidden, median)

    assert torch.all(median > 0.0)
    assert torch.allclose(
        log_survival,
        torch.full_like(log_survival, -math.log(2.0)),
        atol=1e-10,
        rtol=1e-10,
    )


def test_extreme_duration_gradients_are_finite_and_reach_scale_weight() -> None:
    model = build_heteroscedastic_rmtpp()
    with torch.no_grad():
        model.time_scale_weight.weight.fill_(0.01)
    hidden = torch.randn(5, model.hidden_dim, requires_grad=True)
    delta_t = torch.tensor([1.0e-12, 1.0e-6, 1.0, 1.0e6, 1.0e30])

    log_density = model.log_f_dt(hidden, delta_t)
    log_survival = model.log_survival_dt(hidden, delta_t)
    loss = -log_density.mean() - 0.01 * log_survival.mean()
    loss.backward()

    assert torch.isfinite(log_density).all()
    assert torch.isfinite(log_survival).all()
    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for parameter in (
        model.v_t.weight,
        model.b_t,
        model.w_raw,
        model.time_scale_weight.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_heteroscedastic_sigma_requires_hidden_and_positive_duration() -> None:
    model = build_heteroscedastic_rmtpp()
    hidden = torch.zeros(1, model.hidden_dim)

    with pytest.raises(ValueError, match="requires hidden"):
        model.positive_time_sigma()
    with pytest.raises(ValueError, match="strictly positive"):
        model.log_f_dt(hidden, torch.zeros(1))


@pytest.mark.parametrize(
    "mode",
    [TIME_HEAD_MODE_LEGACY_CLAMPED, TIME_HEAD_MODE_LOGNORMAL_DURATION],
)
def test_existing_time_heads_keep_their_three_parameter_schema(mode: str) -> None:
    kwargs = {"time_head_mode": mode}
    if mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
        kwargs.update(
            time_scale=3.0,
            time_initial_location=-0.1,
            time_initial_scale=0.6,
            time_sigma_floor=1e-3,
        )
    model = CountAwareRMTPP(8, train_log_mean=1.5, **kwargs)

    assert not hasattr(model, "time_scale_weight")
    assert tuple(name for name, _ in model.time_head_named_parameters()) == (
        "v_t.weight",
        "b_t",
        "w_raw",
    )
