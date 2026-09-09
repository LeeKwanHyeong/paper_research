from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = ROOT / "paper/scripts/hard_lmm_value_update_probe.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load_module("hard_lmm_value_update_probe_test", HELPER_PATH)


def _per_example_autograd(loss: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    gradients = []
    for index in range(loss.numel()):
        gradients.append(
            torch.autograd.grad(loss[index], hidden, retain_graph=True)[0][index]
        )
    return torch.stack(gradients)


def test_reconstructed_quantity_and_time_credits_match_autograd() -> None:
    torch.manual_seed(101)
    dtype = torch.float64
    hidden = torch.randn(6, 5, dtype=dtype, requires_grad=True)
    quantity_weight = torch.randn(1, 5, dtype=dtype)
    quantity_bias = torch.tensor([0.2], dtype=dtype)
    time_weight = torch.tensor([[0.8, -0.4, 0.2, 0.5, -0.3]], dtype=dtype)
    time_bias = torch.tensor(0.15, dtype=dtype)
    target = torch.tensor([0.5, 2.0, 7.0, 13.0, 19.0, 3.0], dtype=dtype)
    duration = torch.tensor([0.2, 20.0, 2.0, 30.0, 1.0, 25.0], dtype=dtype)
    body_threshold = 12.0
    slope = torch.full((hidden.size(0),), 0.7, dtype=dtype)

    quantity_preactivation = hidden @ quantity_weight.squeeze(0) + quantity_bias
    location = torch.nn.functional.softplus(quantity_preactivation)
    prediction = torch.expm1(location)
    raw_intercept = hidden @ time_weight.squeeze(0) + time_bias
    intercept = torch.clamp(raw_intercept, max=0.25)
    wd_raw = slope * duration
    wd = torch.clamp(wd_raw, max=10.0)
    integral = torch.exp(intercept) / slope * torch.expm1(wd)

    observed = probe.reconstruct_hidden_credits(
        prediction.detach(),
        target,
        duration,
        quantity_weight,
        time_weight,
        slope,
        raw_intercept.detach(),
        intercept.detach(),
        integral.detach(),
        (raw_intercept.detach() > 0.25),
        (wd_raw > 10.0),
        body_threshold=body_threshold,
    )

    log_mse = (location - torch.log1p(target)).square()
    raw_squared = (prediction - target).square()
    body_absolute = torch.where(
        target <= body_threshold,
        torch.abs(prediction - target),
        torch.zeros_like(target),
    )
    time_nll = -intercept - wd + integral
    assert torch.allclose(observed["log_mse"], log_mse.detach())
    assert torch.allclose(observed["raw_squared_error"], raw_squared.detach())
    assert torch.allclose(observed["body_absolute_error"], body_absolute.detach())
    assert torch.allclose(observed["legacy_time_nll"], time_nll.detach())
    assert torch.allclose(
        observed["log_mse_hidden_credit"],
        _per_example_autograd(log_mse, hidden),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(
        observed["raw_squared_error_hidden_credit"],
        _per_example_autograd(raw_squared, hidden),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(
        observed["body_mae_hidden_credit"],
        _per_example_autograd(body_absolute, hidden),
        rtol=1e-12,
        atol=1e-12,
    )
    assert torch.allclose(
        observed["legacy_time_nll_hidden_credit"],
        _per_example_autograd(time_nll, hidden),
        rtol=1e-12,
        atol=1e-12,
    )
    assert bool(observed["body_mae_hidden_credit"][target > body_threshold].eq(0).all())
    assert bool(
        observed["legacy_time_nll_hidden_credit"][raw_intercept.detach() > 0.25]
        .eq(0)
        .all()
    )


def test_selected_value_gradient_matches_hard_mean_autograd() -> None:
    torch.manual_seed(103)
    count, top_k, memory_size, hidden_dim = 7, 3, 11, 4
    values = torch.randn(memory_size, hidden_dim, dtype=torch.float64, requires_grad=True)
    indices = torch.stack(
        [torch.randperm(memory_size)[:top_k] for _ in range(count)]
    )
    credit = torch.randn(count, hidden_dim, dtype=torch.float64)
    hard_read = values[indices].mean(dim=1)
    objective = (hard_read * credit).sum() / count
    expected = torch.autograd.grad(objective, values)[0]
    observed = probe.aggregate_selected_value_gradient(
        credit, indices, memory_size=memory_size
    )
    assert torch.allclose(observed, expected, rtol=1e-15, atol=1e-15)


def test_usage_transform_is_identity_for_equal_usage_after_norm_match() -> None:
    torch.manual_seed(107)
    # Every one of four rows is selected exactly four times.
    indices = torch.tensor(
        [
            [0, 1],
            [2, 3],
            [0, 2],
            [1, 3],
            [0, 3],
            [1, 2],
            [0, 1],
            [2, 3],
        ],
        dtype=torch.long,
    )
    credit = torch.randn(8, 3, dtype=torch.float64)
    result = probe.usage_normalized_support_confidence(
        credit, indices, memory_size=4
    )
    assert torch.equal(result["selection_counts"], torch.full((4,), 4.0))
    assert result["average_usage"].item() == 4.0
    assert torch.allclose(
        result["transformed_gradient"],
        result["current_gradient"],
        rtol=1e-14,
        atol=1e-14,
    )
    assert torch.allclose(
        torch.linalg.vector_norm(result["transformed_gradient"]),
        torch.linalg.vector_norm(result["current_gradient"]),
        rtol=0.0,
        atol=1e-15,
    )


def test_usage_transform_keeps_inactive_rows_zero_and_handles_zero_credit() -> None:
    indices = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.long)
    credit = torch.tensor(
        [[1.0, -2.0], [0.5, 0.25], [-1.0, 3.0]], dtype=torch.float64
    )
    result = probe.usage_normalized_support_confidence(
        credit, indices, memory_size=5
    )
    assert bool(result["transformed_gradient"][2:].eq(0).all())
    assert torch.allclose(
        torch.linalg.vector_norm(result["transformed_gradient"]),
        torch.linalg.vector_norm(result["current_gradient"]),
    )

    zeros = probe.usage_normalized_support_confidence(
        torch.zeros_like(credit), indices, memory_size=5
    )
    assert bool(zeros["transformed_gradient"].eq(0).all())
    assert zeros["norm_match_scale"].item() == 0.0


def test_all_63_cyclic_shifts_preserve_rows_and_frobenius_norm() -> None:
    rows = torch.arange(64 * 3, dtype=torch.float64).reshape(64, 3)
    shifts = probe.cyclic_row_shifts(rows)
    assert shifts.shape == (63, 64, 3)
    expected_rows = sorted(tuple(row.tolist()) for row in rows)
    expected_norm = torch.linalg.vector_norm(rows)
    for shift in shifts:
        assert sorted(tuple(row.tolist()) for row in shift) == expected_rows
        assert torch.equal(torch.linalg.vector_norm(shift), expected_norm)
        assert not torch.equal(shift, rows)


def test_direction_and_prospective_gate_use_fixed_tolerance() -> None:
    candidate = probe.directional_dot_cosine(
        torch.tensor([2.0, -1.0]), torch.tensor([3.0, -4.0])
    )
    control = probe.directional_dot_cosine(
        torch.tensor([1.0, 0.0]), torch.tensor([3.0, -4.0])
    )
    assert candidate["dot"] == 10.0
    assert candidate["positive_direction"] is True
    assert candidate["non_conflicting_direction"] is True
    assert probe.prospective_alignment_gate(
        candidate, [control], require_positive=True
    )["passed"] is True

    tiny_conflict = {
        **candidate,
        "dot": -0.5 * probe.DIRECTION_TOLERANCE,
    }
    assert probe.prospective_alignment_gate(
        tiny_conflict, require_positive=False
    )["passed"] is True
    assert probe.prospective_alignment_gate(
        tiny_conflict, require_positive=True
    )["passed"] is False


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("nonfinite_prediction", "finite"),
        ("wrong_intercept_flag", "intercept_saturated"),
        ("wrong_wd_flag", "wd_saturated"),
        ("wrong_integral", "time_integral"),
    ],
)
def test_reconstruction_fails_closed_on_bad_cache(case: str, match: str) -> None:
    prediction = torch.tensor([1.0, 2.0], dtype=torch.float64)
    target = torch.tensor([1.5, 2.5], dtype=torch.float64)
    duration = torch.tensor([1.0, 20.0], dtype=torch.float64)
    slope = torch.tensor([0.7, 0.7], dtype=torch.float64)
    raw = torch.tensor([0.1, 0.4], dtype=torch.float64)
    clamped = torch.tensor([0.1, 0.25], dtype=torch.float64)
    wd = torch.clamp(slope * duration, max=10.0)
    integral = torch.exp(clamped) / slope * torch.expm1(wd)
    intercept_flag = torch.tensor([False, True])
    wd_flag = torch.tensor([False, True])
    if case == "nonfinite_prediction":
        prediction[0] = float("nan")
    elif case == "wrong_intercept_flag":
        intercept_flag[1] = False
    elif case == "wrong_wd_flag":
        wd_flag[1] = False
    elif case == "wrong_integral":
        integral[0] += 1.0
    with pytest.raises(ValueError, match=match):
        probe.reconstruct_hidden_credits(
            prediction,
            target,
            duration,
            torch.ones(1, 3, dtype=torch.float64),
            torch.ones(1, 3, dtype=torch.float64),
            slope,
            raw,
            clamped,
            integral,
            intercept_flag,
            wd_flag,
            body_threshold=3.0,
        )


def test_gradient_helpers_reject_invalid_or_nonfinite_inputs() -> None:
    credit = torch.ones(2, 3)
    duplicate = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="distinct"):
        probe.aggregate_selected_value_gradient(credit, duplicate, memory_size=4)
    with pytest.raises(ValueError, match="finite"):
        probe.aggregate_selected_value_gradient(
            credit.fill_(float("inf")),
            torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            memory_size=4,
        )
    with pytest.raises(ValueError, match="finite"):
        probe.cyclic_row_shifts(torch.tensor([[1.0], [float("nan")]]))
    with pytest.raises(ValueError, match="finite"):
        probe.directional_dot_cosine(
            torch.tensor([1.0, float("nan")]), torch.ones(2)
        )
