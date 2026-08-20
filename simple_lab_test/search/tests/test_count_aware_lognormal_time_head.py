from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
    CountAwareRMTPP,
    inverse_softplus,
)
from paper.scripts.run_count_aware_tpp_backbone_control import (
    derive_train_time_contract,
    prepare_count_frame,
)


LOGNORMAL_TIME_KWARGS = {
    "time_head_mode": TIME_HEAD_MODE_LOGNORMAL_DURATION,
    "time_scale": 3.0,
    "time_initial_location": -0.1,
    "time_initial_scale": 0.6,
    "time_sigma_floor": 1e-3,
}


def build_lognormal_rmtpp(*, hidden_dim: int = 8) -> CountAwareRMTPP:
    torch.manual_seed(23)
    return CountAwareRMTPP(
        hidden_dim,
        train_log_mean=1.5,
        **LOGNORMAL_TIME_KWARGS,
    )


@pytest.mark.parametrize("backbone", ["rmtpp", "thp", "titantpp"])
def test_lognormal_contract_is_shared_by_primary_backbones(backbone: str) -> None:
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
        **LOGNORMAL_TIME_KWARGS,
    )

    assert model.time_head_contract() == metadata["time_head"]
    assert metadata["time_head"] == {
        "mode": TIME_HEAD_MODE_LOGNORMAL_DURATION,
        "density_family": "lognormal_on_scaled_duration",
        "time_scale": 3.0,
        "time_initial_location": -0.1,
        "time_initial_scale": 0.6,
        "time_sigma_floor": 1e-3,
        "time_location_transform": "identity",
        "slope_parameterized": False,
        "jacobian_correction": True,
        "wd_clamp": 0.0,
    }


def test_lognormal_constants_are_derived_from_train_targets_only() -> None:
    frame = pl.DataFrame(
        {
            "oper_part_no": ["a", "a", "a", "a", "b", "b", "b"],
            "seq": [0, 1, 2, 3, 0, 1, 2],
            "delta_t": [999, 1, 3, 100, 999, 5, 200],
            "demand_qty": [1.0] * 7,
            "chronological_split": [
                "train",
                "train",
                "train",
                "validation",
                "train",
                "train",
                "test",
            ],
        }
    )

    contract = derive_train_time_contract(
        prepare_count_frame(frame),
        lookback_weeks=520,
        max_seq_len=8,
    )
    expected = np.log(np.asarray([1.0, 3.0, 5.0]) / 3.0)

    assert contract["target_count"] == 3
    assert contract["time_scale"] == 3.0
    assert contract["target_log_scaled_mean"] == pytest.approx(expected.mean())
    assert contract["target_log_scaled_std"] == pytest.approx(expected.std())


def test_lognormal_density_matches_closed_form_in_original_time_unit() -> None:
    model = build_lognormal_rmtpp().double()
    hidden = torch.randn(4, model.hidden_dim, dtype=torch.float64)
    delta_t = torch.tensor([1.0, 3.0, 7.0, 36.0], dtype=torch.float64)

    actual = model.log_f_dt(hidden, delta_t)
    location = model.v_t(hidden).squeeze(-1) + model.b_t
    scale = model.time_sigma_floor + torch.nn.functional.softplus(model.w_raw)
    standardized = (torch.log(delta_t / model.time_scale) - location) / scale
    expected = (
        -0.5 * torch.square(standardized)
        - torch.log(scale)
        - torch.log(delta_t)
        - 0.5 * math.log(2.0 * math.pi)
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_lognormal_density_numerically_integrates_to_one() -> None:
    model = build_lognormal_rmtpp().double()
    model.v_t.weight.data.zero_()
    model.b_t.data.fill_(0.15)
    model.w_raw.data.fill_(inverse_softplus(0.5 - model.time_sigma_floor))
    grid = torch.logspace(-5.0, 4.0, 80_001, dtype=torch.float64)
    hidden = torch.zeros(grid.numel(), model.hidden_dim, dtype=torch.float64)

    density = torch.exp(model.log_f_dt(hidden, grid))
    integral = torch.trapezoid(density, grid)

    assert torch.isclose(integral, integral.new_tensor(1.0), atol=2e-6, rtol=0.0)


def test_lognormal_survival_and_median_are_consistent() -> None:
    model = build_lognormal_rmtpp().double()
    hidden = torch.randn(6, model.hidden_dim, dtype=torch.float64)

    median = model.predict_time_median(hidden)
    log_survival = model.log_survival_dt(hidden, median)

    assert torch.all(median > 0.0)
    assert torch.allclose(
        log_survival,
        torch.full_like(log_survival, -math.log(2.0)),
        atol=1e-10,
        rtol=1e-10,
    )


def test_lognormal_forward_and_backward_are_finite_on_extreme_range() -> None:
    model = build_lognormal_rmtpp()
    hidden = torch.randn(4, model.hidden_dim, requires_grad=True)
    delta_t = torch.tensor([1.0e-4, 1.0, 36.0, 1.0e6])

    loss = -model.log_f_dt(hidden, delta_t).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert model.w_raw.grad is not None and torch.isfinite(model.w_raw.grad).all()
    assert model.v_t.weight.grad is not None
    assert torch.isfinite(model.v_t.weight.grad).all()
    assert set(model.time_head_telemetry()) == {"train_time_sigma"}


def test_lognormal_rejects_nonpositive_duration_targets() -> None:
    model = build_lognormal_rmtpp()
    hidden = torch.zeros(1, model.hidden_dim)

    with pytest.raises(ValueError, match="strictly positive"):
        model.log_f_dt(hidden, torch.zeros(1))
