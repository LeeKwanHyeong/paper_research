from __future__ import annotations

import math

import pytest
import polars as pl
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import (
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TIME_HEAD_MODE_SCALED_EXACT,
    CountAwareRMTPP,
    inverse_sigmoid,
)
from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.run_count_aware_tpp_backbone_control import (
    derive_train_time_contract,
    prepare_count_frame,
)


SCALED_TIME_KWARGS = {
    "time_head_mode": TIME_HEAD_MODE_SCALED_EXACT,
    "time_scale": 3.0,
    "time_w_max": 10.0 / 3.0,
    "time_intercept_limit": 30.0,
}


def build_scaled_rmtpp(*, hidden_dim: int = 8) -> CountAwareRMTPP:
    torch.manual_seed(13)
    return CountAwareRMTPP(
        hidden_dim,
        train_log_mean=1.5,
        **SCALED_TIME_KWARGS,
    )


@pytest.mark.parametrize("backbone", ["rmtpp", "thp", "titantpp"])
def test_scaled_exact_contract_is_shared_by_primary_backbones(backbone: str) -> None:
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
        **SCALED_TIME_KWARGS,
    )

    assert model.time_head_contract() == metadata["time_head"]
    assert metadata["time_head"] == {
        "mode": TIME_HEAD_MODE_SCALED_EXACT,
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 30.0,
        "jacobian_correction": True,
        "wd_clamp": 0.0,
    }


def test_time_constants_are_derived_from_train_targets_only() -> None:
    frame = pl.DataFrame(
        {
            "oper_part_no": ["a", "a", "a", "a", "a", "b", "b", "b"],
            "seq": [0, 1, 2, 3, 4, 0, 1, 2],
            "delta_t": [999, 1, 3, 100, 200, 999, 5, 100],
            "demand_qty": [1.0] * 8,
            "chronological_split": [
                "train",
                "train",
                "train",
                "validation",
                "test",
                "train",
                "train",
                "validation",
            ],
        }
    )

    contract = derive_train_time_contract(
        prepare_count_frame(frame),
        lookback_weeks=520,
        max_seq_len=8,
    )

    assert contract["target_count"] == 3
    assert contract["target_dt_p50"] == 3.0
    assert contract["target_dt_max"] == 5.0
    assert contract["time_scale"] == 3.0
    assert contract["time_w_max"] == 24.0


def test_scaled_exact_density_matches_closed_form_with_jacobian() -> None:
    model = build_scaled_rmtpp().double()
    hidden = torch.randn(4, model.hidden_dim, dtype=torch.float64)
    delta_t = torch.tensor([1.0, 3.0, 7.0, 36.0], dtype=torch.float64)

    actual = model.log_f_dt(hidden, delta_t)
    intercept = torch.clamp(
        model.v_t(hidden).squeeze(-1) + model.b_t,
        min=-model.time_intercept_limit,
        max=model.time_intercept_limit,
    )
    w = model.time_w_max * torch.sigmoid(model.w_raw)
    scaled_dt = delta_t / model.time_scale
    expected = (
        intercept
        + w * scaled_dt
        - (torch.exp(intercept) / w) * torch.expm1(w * scaled_dt)
        - math.log(model.time_scale)
    )

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_scaled_exact_density_numerically_integrates_to_one() -> None:
    model = build_scaled_rmtpp().double()
    model.v_t.weight.data.zero_()
    model.b_t.data.fill_(math.log(model.time_scale))
    initial_w = 0.15
    model.w_raw.data.fill_(inverse_sigmoid(initial_w / model.time_w_max))
    grid = torch.linspace(0.0, 36.0, 40_001, dtype=torch.float64)
    hidden = torch.zeros(grid.numel(), model.hidden_dim, dtype=torch.float64)

    density = torch.exp(model.log_f_dt(hidden, grid))
    integral = torch.trapezoid(density, grid)

    assert torch.isclose(integral, integral.new_tensor(1.0), atol=2e-6, rtol=0.0)


def test_scaled_exact_survival_and_median_are_consistent() -> None:
    model = build_scaled_rmtpp().double()
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


def test_scaled_exact_slope_is_bounded_without_duration_clamp() -> None:
    model = build_scaled_rmtpp()
    model.w_raw.data.fill_(1.0e6)
    hidden = torch.zeros(2, model.hidden_dim)
    delta_t = torch.tensor([1.0, 36.0])

    slope = model.positive_time_slope()
    log_density = model.log_f_dt(hidden, delta_t)

    assert 0.0 < float(slope) <= model.time_w_max
    assert torch.isfinite(log_density).all()
    assert log_density[0] != log_density[1]


def test_scaled_exact_loss_and_gradients_are_finite_on_train_range() -> None:
    model = build_scaled_rmtpp()
    hidden = torch.randn(4, model.hidden_dim, requires_grad=True)
    delta_t = torch.tensor([1.0, 3.0, 7.0, 36.0])

    loss = -model.log_f_dt(hidden, delta_t).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert model.w_raw.grad is not None and torch.isfinite(model.w_raw.grad).all()
    assert model.v_t.weight.grad is not None
    assert torch.isfinite(model.v_t.weight.grad).all()


def test_legacy_mode_preserves_clamped_formula() -> None:
    model = CountAwareRMTPP(
        8,
        train_log_mean=1.5,
        time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED,
    )
    hidden = torch.randn(3, model.hidden_dim)
    delta_t = torch.tensor([1.0, 3.0, 1.0e6])

    w = F.softplus(model.w_raw) + 1e-3
    intercept = torch.clamp(model.v_t(hidden).squeeze(-1) + model.b_t, max=300.0)
    wd = torch.clamp(w * delta_t, max=10.0)
    expected = intercept + wd - (torch.exp(intercept) / w) * torch.expm1(wd)

    assert torch.equal(model.log_f_dt(hidden, delta_t), expected)


@pytest.mark.parametrize(
    ("name", "value"),
    [("time_scale", 0.0), ("time_w_max", -1.0), ("time_intercept_limit", 0.0)],
)
def test_scaled_exact_rejects_invalid_constants(name: str, value: float) -> None:
    kwargs = dict(SCALED_TIME_KWARGS)
    kwargs[name] = value
    with pytest.raises(ValueError):
        CountAwareRMTPP(8, train_log_mean=1.5, **kwargs)
