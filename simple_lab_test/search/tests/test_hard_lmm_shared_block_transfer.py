from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts import hard_lmm_shared_block_transfer as transfer


def synthetic_continuous_data(count: int = 96) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20391)
    hidden = torch.randn(count, transfer.HIDDEN_DIM, generator=generator)
    log_duration = 0.35 + 0.8 * hidden[:, 0] - 0.45 * hidden[:, 1]
    duration = 4.0 * torch.exp(log_duration).clamp(max=25.0)
    return hidden, duration


def build_head() -> transfer.FrozenBK1LogNormalHead:
    return transfer.FrozenBK1LogNormalHead(
        time_scale=3.0,
        location_mean=0.2,
        population_std=0.8,
    )


def test_k1_head_has_130_float32_parameters_and_float64_likelihood() -> None:
    head = build_head()
    assert sum(parameter.numel() for parameter in head.parameters()) == 130
    assert all(parameter.dtype == torch.float32 for parameter in head.parameters())
    hidden = torch.zeros(3, transfer.HIDDEN_DIM, dtype=torch.float32)
    target = torch.tensor([1.0, 2.0, 4.0])
    likelihood = head.log_likelihood(
        hidden,
        target,
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
    )
    location, sigma = head.location_and_sigma(hidden)
    assert likelihood.dtype == torch.float64
    assert location.dtype == torch.float64
    assert sigma.dtype == torch.float64
    assert torch.isfinite(likelihood).all()


def test_source_fold_initialization_uses_median_mean_and_population_std() -> None:
    target = torch.tensor([1.0, 2.0, 4.0, 16.0], dtype=torch.float32)
    head, statistics = transfer.FrozenBK1LogNormalHead.from_source_fold_targets(
        target
    )
    scale = float(torch.quantile(target.double(), 0.5))
    log_scaled = torch.log(target.double() / scale)
    assert statistics["time_scale"] == scale
    assert statistics["target_log_scaled_mean"] == pytest.approx(
        float(log_scaled.mean()), abs=1e-15
    )
    assert statistics["target_log_scaled_population_std"] == pytest.approx(
        float(log_scaled.std(unbiased=False)), abs=1e-15
    )
    hidden = torch.zeros(2, transfer.HIDDEN_DIM)
    location, sigma = head.location_and_sigma(hidden)
    assert torch.allclose(
        location,
        torch.full_like(location, float(head.location_bias.detach())),
    )
    # Stored float32 parameters are promoted before softplus, so compare to
    # their promoted value rather than to the unrounded source statistic.
    expected_sigma = head.sigma_floor + torch.nn.functional.softplus(
        head.scale_raw_bias.detach().double()
    )
    assert torch.allclose(sigma, expected_sigma.expand_as(sigma), atol=0, rtol=0)


def test_positive_integer_likelihood_normalizes_and_censor_uses_lower_edge() -> None:
    head = build_head()
    # Codes 1..100 have ordinary bins; code 101 closes the law with S(100.5).
    target = torch.arange(1, 102, dtype=torch.float32)
    censor = torch.zeros_like(target, dtype=torch.bool)
    censor[-1] = True
    hidden = torch.zeros(target.numel(), transfer.HIDDEN_DIM)
    log_probability = head.log_likelihood(
        hidden,
        target,
        observation_mode=transfer.POSITIVE_INTEGER_LOGNORMAL,
        is_right_censored=censor,
    )
    assert torch.isfinite(log_probability).all()
    assert log_probability.exp().sum().item() == pytest.approx(1.0, abs=1e-12)

    top_only = head.log_likelihood(
        hidden[-1:],
        target[-1:],
        observation_mode=transfer.POSITIVE_INTEGER_LOGNORMAL,
        is_right_censored=torch.tensor([True]),
    )
    location, sigma = head.location_and_sigma(hidden[-1:])
    lower_z = (
        math.log(100.5 / head.time_scale) - location
    ) / sigma
    assert torch.allclose(
        top_only, torch.special.log_ndtr(-lower_z), atol=1e-14, rtol=0.0
    )


@pytest.mark.parametrize("location", [-12.0, 12.0])
def test_integer_and_continuous_likelihoods_are_finite_in_tails(
    location: float,
) -> None:
    head = build_head()
    with torch.no_grad():
        head.location_bias.fill_(location)
    hidden = torch.zeros(4, transfer.HIDDEN_DIM)
    integer_target = torch.tensor([1.0, 2.0, 100.0, 101.0])
    censor = torch.tensor([False, False, False, True])
    integer = head.log_likelihood(
        hidden,
        integer_target,
        observation_mode=transfer.POSITIVE_INTEGER_LOGNORMAL,
        is_right_censored=censor,
    )
    continuous = head.log_likelihood(
        hidden,
        torch.tensor([0.01, 1.0, 100.0, 1000.0]),
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
        is_right_censored=censor,
    )
    assert torch.isfinite(integer).all()
    assert torch.isfinite(continuous).all()


def test_epoch_zero_hidden_gradient_is_zero_then_fit_makes_it_nonzero() -> None:
    hidden, target = synthetic_continuous_data()
    head, _ = transfer.FrozenBK1LogNormalHead.from_source_fold_targets(target)
    probe = hidden[:16].detach().clone().requires_grad_(True)
    initial_loss = -head.log_likelihood(
        probe,
        target[:16],
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
    ).mean()
    initial_gradient = torch.autograd.grad(initial_loss, probe)[0]
    assert torch.equal(initial_gradient, torch.zeros_like(initial_gradient))

    transfer.fit_train_only_adamw(
        head,
        hidden,
        target,
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
        epochs=12,
        learning_rate=1e-2,
        batch_size=24,
        seed=77,
    )
    trained_probe = hidden[:16].detach().clone().requires_grad_(True)
    trained_loss = -head.log_likelihood(
        trained_probe,
        target[:16],
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
    ).mean()
    trained_gradient = torch.autograd.grad(trained_loss, trained_probe)[0]
    assert torch.isfinite(trained_gradient).all()
    assert torch.linalg.vector_norm(trained_gradient).item() > 0.0


def test_fixed_train_only_adamw_fit_is_deterministic_and_changes_state() -> None:
    hidden, target = synthetic_continuous_data()
    first, _ = transfer.FrozenBK1LogNormalHead.from_source_fold_targets(target)
    second, _ = transfer.FrozenBK1LogNormalHead.from_source_fold_targets(target)
    initial = copy.deepcopy(first.state_dict())
    first_history = transfer.fit_train_only_adamw(
        first,
        hidden,
        target,
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
        epochs=8,
        learning_rate=5e-3,
        batch_size=32,
        seed=19,
    )
    second_history = transfer.fit_train_only_adamw(
        second,
        hidden,
        target,
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
        epochs=8,
        learning_rate=5e-3,
        batch_size=32,
        seed=19,
    )
    assert first_history == second_history
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    assert any(
        not torch.equal(first.state_dict()[name], initial[name])
        for name in initial
    )


def test_crossfit_runner_api_is_train_only_and_json_safe() -> None:
    hidden, target = synthetic_continuous_data(count=32)
    first, audit = transfer.fit_crossfit_time_head(
        hidden,
        target,
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
        censor_threshold=None,
        seed=8,
        epochs=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
        batch_size=16,
    )
    assert audit["scope"] == "source_fold_train_only"
    assert audit["parameter_count"] == 130
    assert audit["target_count"] == 32
    assert audit["state_changed"] is True
    assert audit["final_nll"] < audit["initial_nll"]
    assert audit["optimizer_betas"] == [0.9, 0.999]
    assert audit["optimizer_eps"] == 1e-8
    assert audit["optimizer_amsgrad"] is False
    assert audit["optimizer_foreach"] is False
    json.dumps(audit, sort_keys=True)
    nll = first.negative_log_likelihood(
        hidden,
        target,
        observation_mode=transfer.CONTINUOUS_LOGNORMAL,
        censor_threshold=None,
    )
    assert nll.shape == target.shape
    assert nll.dtype == torch.float64


def test_analytic_zero_init_adapter_gradient_matches_autograd_and_denominator() -> None:
    generator = torch.Generator().manual_seed(901)
    state = torch.randn(3, 5, 7, generator=generator, dtype=torch.float64)
    credit = torch.randn(3, 5, 7, generator=generator, dtype=torch.float64)
    valid = torch.tensor(
        [
            [True, True, False, False, False],
            [True, False, True, False, False],
            [True, True, True, True, False],
        ]
    )
    denominator = 5.5
    analytic = transfer.zero_init_linear_residual_adapter_gradient(
        state, credit, valid, denominator=denominator
    )
    adapter = torch.zeros((7, 7), dtype=torch.float64, requires_grad=True)
    transformed = state + torch.einsum("ij,btj->bti", adapter, state)
    loss = ((transformed * credit).sum(dim=-1) * valid).sum() / denominator
    autograd = torch.autograd.grad(loss, adapter)[0]
    assert torch.allclose(analytic, autograd, atol=1e-14, rtol=0.0)

    default = transfer.zero_init_linear_residual_adapter_gradient(
        state, credit, valid
    )
    expected = autograd * denominator / float(valid.sum())
    assert torch.allclose(default, expected, atol=1e-14, rtol=0.0)


def conflict_gradients() -> tuple[torch.Tensor, torch.Tensor]:
    quantity = torch.zeros(transfer.HIDDEN_DIM, transfer.HIDDEN_DIM)
    time = torch.zeros_like(quantity)
    quantity[0, 0] = 1.0
    time[0, 0] = -0.8
    time[1, 0] = 0.6
    return quantity, time


def test_update_arms_match_unit_budget_and_all_63_controls_preserve_norm() -> None:
    quantity, time = conflict_gradients()
    arms = transfer.construct_directional_update_arms(quantity, time)
    assert arms["norms"]["equal_weight_shared"] == pytest.approx(1.0)
    assert arms["norms"]["norm_balanced_shared"] == pytest.approx(1.0)
    assert arms["norms"]["task_separated_quantity"] == pytest.approx(
        1.0 / math.sqrt(2.0)
    )
    assert arms["norms"]["task_separated_time"] == pytest.approx(
        1.0 / math.sqrt(2.0)
    )
    assert arms["norms"]["task_separated_total"] == pytest.approx(1.0)
    shifts = transfer.cyclic_output_row_shifts(
        arms["task_separated_quantity"]
    )
    assert shifts.shape == (63, 64, 64)
    expected_norm = torch.linalg.vector_norm(
        arms["task_separated_quantity"]
    )
    assert all(
        torch.equal(torch.linalg.vector_norm(shift), expected_norm)
        for shift in shifts
    )


def test_directional_gate_passes_only_when_separation_beats_both_shared_controls() -> None:
    quantity, time = conflict_gradients()
    held = {
        transfer.LOG_MSE: quantity,
        transfer.RAW_SQUARED_ERROR: quantity * 2.0,
        transfer.BODY_MAE: quantity * 0.5,
        transfer.NORMALIZED_TIME_NLL: time,
    }
    result = transfer.compare_shared_and_task_separated_updates(
        quantity, time, held
    )
    assert result["checks"]["passed"] is True
    assert result["checks"]["exactly_63_shift_controls"] is True
    assert all(
        row["cyclic_output_row_shift_control"]["count"] == 63
        and row["checks"]["passed"]
        for row in result["metrics"].values()
    )
    assert all(
        result["metrics"][name]["source_arm"] == "log_mse"
        for name in transfer.QUANTITY_METRICS
    )
    assert (
        result["metrics"][transfer.NORMALIZED_TIME_NLL]["source_arm"]
        == "normalized_time_nll"
    )

    aligned = torch.zeros_like(quantity)
    aligned[0, 0] = 1.0
    failed = transfer.compare_shared_and_task_separated_updates(
        aligned,
        aligned,
        {
            transfer.LOG_MSE: aligned,
            transfer.RAW_SQUARED_ERROR: aligned,
            transfer.BODY_MAE: aligned,
            transfer.NORMALIZED_TIME_NLL: aligned,
        },
    )
    assert failed["checks"]["passed"] is False
    assert all(
        not row["checks"]["beats_equal_weight_shared"]
        for name, row in failed["metrics"].items()
        if name != transfer.BODY_MAE
    )
    assert failed["metrics"][transfer.BODY_MAE]["checks"][
        "non_conflicting_direction"
    ] is True
