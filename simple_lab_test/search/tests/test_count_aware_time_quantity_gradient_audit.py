from __future__ import annotations

import math

import pytest
import torch

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TAIL_SHARED_VARIANT,
    TIME_HEAD_MODE_SCALED_EXACT_STABLE,
)
from paper.scripts.audit_count_aware_time_quantity_gradients import (
    classify_h1_failure,
    gradient_pair_statistics,
    shared_encoder_parameters,
    slope_nll_derivative,
    target_terms,
)


def make_model() -> CountAwareTitanTPP:
    return CountAwareTitanTPP(
        hidden_dim=16,
        train_log_mean=1.5,
        train_log_std=0.8,
        max_seq_len=8,
        quantity_variant=TAIL_SHARED_VARIANT,
        lambda_tail=0.1,
        time_head_mode=TIME_HEAD_MODE_SCALED_EXACT_STABLE,
        time_scale=3.0,
        time_w_max=2.0 / 3.0,
        time_intercept_limit=6.0,
        time_initial_intercept=0.0,
        time_wd_safety_limit=8.0,
    )


def test_gradient_pair_statistics_reports_direction_and_norm_ratio() -> None:
    result = gradient_pair_statistics(
        [torch.tensor([1.0, 0.0]), torch.tensor([2.0])],
        [torch.tensor([-1.0, 0.0]), torch.tensor([-2.0])],
    )

    assert result["time_encoder_grad_norm"] == pytest.approx(math.sqrt(5.0))
    assert result["quantity_encoder_grad_norm"] == pytest.approx(math.sqrt(5.0))
    assert result["time_quantity_grad_cosine"] == pytest.approx(-1.0)
    assert result["quantity_to_time_grad_norm_ratio"] == pytest.approx(1.0)


def test_classification_separates_slope_and_gradient_failures() -> None:
    result = classify_h1_failure(
        {
            "time_slope_ratio": 0.999,
            "slope_upward_pressure_fraction": 0.75,
            "gradient_cosine_median": -0.2,
            "gradient_strong_conflict_fraction": 0.6,
        }
    )

    assert result["slope_contract_failed"] is True
    assert result["strong_time_quantity_gradient_conflict"] is True
    assert result["recommendation"] == (
        "replace_slope_family_and_isolate_time_gradient"
    )


def test_target_terms_mask_target_quantity_and_gradients_are_finite() -> None:
    torch.manual_seed(11)
    model = make_model().eval()
    dts = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 4.0]],
        dtype=torch.float32,
    )
    quantities = torch.tensor(
        [[0.0, 2.0, 4.0, 80.0], [0.0, 0.0, 3.0, 60.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[False, True, True, True], [False, False, True, True]],
        dtype=torch.bool,
    )

    terms = target_terms(model, dts, mask, quantities)
    parameters = tuple(parameter for _, parameter in shared_encoder_parameters(model))
    time_gradients = torch.autograd.grad(
        terms["time_loss"].mean(),
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    quantity_gradients = torch.autograd.grad(
        terms["quantity_loss"].mean(),
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    statistics = gradient_pair_statistics(time_gradients, quantity_gradients)
    derivative = slope_nll_derivative(
        model,
        terms["time_hidden"],
        terms["true_dt"],
    )

    assert terms["true_qty"].tolist() == [80.0, 60.0]
    assert terms["tail_indicator"].tolist() == [1.0, 1.0]
    assert torch.isfinite(terms["time_loss"]).all()
    assert torch.isfinite(terms["quantity_loss"]).all()
    assert torch.isfinite(derivative).all()
    assert all(math.isfinite(float(value)) for value in statistics.values())
