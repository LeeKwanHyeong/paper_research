"""Pure algebra for the Hard-LMM selected-value update diagnostic.

The helpers in this module operate on frozen-B cache tensors.  They do not
run a model, read a dataset, or mutate parameters.  The central convention is
that a batch loss is averaged over ``N`` examples and a hard memory read is
the arithmetic mean of ``k`` distinct selected rows.
"""

from __future__ import annotations

import math
from typing import Any

import torch


DIRECTION_TOLERANCE = 1e-12
NORM_TOLERANCE = 1e-30


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    return value


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _vector(name: str, value: torch.Tensor, size: int) -> torch.Tensor:
    value = _require_tensor(name, value)
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape [{size}]")
    _require_finite(name, value)
    return value.to(dtype=torch.float64)


def _head_weight(
    name: str,
    value: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    value = _require_tensor(name, value)
    if value.ndim == 2 and value.size(0) == 1:
        value = value.squeeze(0)
    if value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must have shape [hidden] or [1, hidden]")
    if value.device != device:
        raise ValueError(f"{name} must be on the prediction device")
    _require_finite(name, value)
    return value.to(dtype=torch.float64)


def _flag(name: str, value: torch.Tensor, size: int) -> torch.Tensor:
    value = _require_tensor(name, value)
    if value.shape != (size,) or value.dtype != torch.bool:
        raise ValueError(f"{name} must be a bool tensor with shape [{size}]")
    return value


def reconstruct_hidden_credits(
    prediction: torch.Tensor,
    target_quantity: torch.Tensor,
    duration: torch.Tensor,
    quantity_weight: torch.Tensor,
    time_weight: torch.Tensor,
    time_slope: torch.Tensor,
    time_intercept_raw: torch.Tensor,
    time_intercept_clamped: torch.Tensor,
    time_integral: torch.Tensor,
    intercept_saturated: torch.Tensor,
    wd_saturated: torch.Tensor,
    *,
    body_threshold: float,
    wd_limit: float = 10.0,
    consistency_rtol: float = 1e-8,
    consistency_atol: float = 1e-10,
) -> dict[str, torch.Tensor]:
    """Reconstruct per-example loss gradients with respect to B hidden state.

    ``prediction`` is B's non-negative raw quantity prediction.  The quantity
    head is ``expm1(softplus(W_q h + b_q))``.  The time cache follows the
    legacy clamped RMTPP head::

        A = min(W_t h + b_t, intercept_limit)
        Wd = min(w * duration, 10)
        integral = exp(A) / w * expm1(Wd)
        nll = -A - Wd + integral

    The returned credits are gradients of individual, unreduced losses.  A
    saturated duration term has no hidden-state derivative; a saturated
    intercept explicitly zeros the time credit.  Cache relationships are
    checked before any credit is returned so corrupt telemetry fails closed.
    """
    prediction = _require_tensor("prediction", prediction)
    if prediction.ndim != 1 or prediction.numel() == 0:
        raise ValueError("prediction must be a nonempty rank-one tensor")
    size = prediction.numel()
    device = prediction.device
    numeric_inputs = {
        "target_quantity": target_quantity,
        "duration": duration,
        "time_slope": time_slope,
        "time_intercept_raw": time_intercept_raw,
        "time_intercept_clamped": time_intercept_clamped,
        "time_integral": time_integral,
    }
    for name, value in numeric_inputs.items():
        value = _require_tensor(name, value)
        if value.device != device:
            raise ValueError(f"{name} must be on the prediction device")
    prediction = _vector("prediction", prediction, size)
    target = _vector("target_quantity", target_quantity, size)
    duration_64 = _vector("duration", duration, size)
    slope = _vector("time_slope", time_slope, size)
    intercept_raw = _vector("time_intercept_raw", time_intercept_raw, size)
    intercept = _vector(
        "time_intercept_clamped", time_intercept_clamped, size
    )
    integral = _vector("time_integral", time_integral, size)
    intercept_flag = _flag("intercept_saturated", intercept_saturated, size)
    wd_flag = _flag("wd_saturated", wd_saturated, size)
    if intercept_flag.device != device or wd_flag.device != device:
        raise ValueError("clamp flags must be on the prediction device")
    quantity_w = _head_weight("quantity_weight", quantity_weight, device=device)
    time_w = _head_weight("time_weight", time_weight, device=device)
    if quantity_w.shape != time_w.shape:
        raise ValueError("quantity and time head weights must have equal shape")

    scalars = {
        "body_threshold": body_threshold,
        "wd_limit": wd_limit,
        "consistency_rtol": consistency_rtol,
        "consistency_atol": consistency_atol,
    }
    if any(not math.isfinite(float(value)) for value in scalars.values()):
        raise ValueError("thresholds and consistency tolerances must be finite")
    if body_threshold < 0.0 or wd_limit <= 0.0:
        raise ValueError("body_threshold must be non-negative and wd_limit positive")
    if consistency_rtol < 0.0 or consistency_atol < 0.0:
        raise ValueError("consistency tolerances must be non-negative")
    if bool((prediction < 0.0).any()):
        raise ValueError("prediction must be non-negative")
    if bool((target < 0.0).any()):
        raise ValueError("target_quantity must be non-negative")
    if bool((duration_64 < 0.0).any()):
        raise ValueError("duration must be non-negative")
    if bool((slope <= 0.0).any()):
        raise ValueError("time_slope must be positive")
    if bool((integral < 0.0).any()):
        raise ValueError("time_integral must be non-negative")

    expected_intercept_flag = intercept_raw > intercept
    if not torch.equal(intercept_flag, expected_intercept_flag):
        raise ValueError("intercept_saturated disagrees with cached intercepts")
    if not bool(
        torch.allclose(
            intercept[~intercept_flag],
            intercept_raw[~intercept_flag],
            rtol=consistency_rtol,
            atol=consistency_atol,
        )
    ):
        raise ValueError("unsaturated time intercept changed in the cache")
    if bool((intercept[intercept_flag] >= intercept_raw[intercept_flag]).any()):
        raise ValueError("saturated time intercept must be below its raw value")

    wd_raw = slope * duration_64
    expected_wd_flag = wd_raw > float(wd_limit)
    if not torch.equal(wd_flag, expected_wd_flag):
        raise ValueError("wd_saturated disagrees with slope and duration")
    wd_clamped = torch.clamp(wd_raw, max=float(wd_limit))
    expected_integral = torch.exp(intercept) / slope * torch.expm1(wd_clamped)
    if not bool(torch.isfinite(expected_integral).all()):
        raise ValueError("reconstructed time integral is non-finite")
    if not bool(
        torch.allclose(
            integral,
            expected_integral,
            rtol=consistency_rtol,
            atol=consistency_atol,
        )
    ):
        raise ValueError("time_integral disagrees with cached RMTPP terms")

    log_prediction = torch.log1p(prediction)
    log_error = log_prediction - torch.log1p(target)
    # If y=softplus(a) and prediction=expm1(y), then
    # d y / d a = prediction / (1 + prediction), while
    # d prediction / d a = prediction exactly.
    location_derivative = prediction / (1.0 + prediction)
    raw_derivative = prediction
    body_mask = target <= float(body_threshold)

    log_scalar = 2.0 * log_error * location_derivative
    raw_scalar = 2.0 * (prediction - target) * raw_derivative
    body_scalar = torch.where(
        body_mask,
        torch.sign(prediction - target) * raw_derivative,
        torch.zeros_like(prediction),
    )
    time_scalar = torch.where(
        intercept_flag,
        torch.zeros_like(integral),
        integral - 1.0,
    )

    result = {
        "log_mse": log_error.square(),
        "raw_squared_error": (prediction - target).square(),
        "body_absolute_error": torch.where(
            body_mask,
            torch.abs(prediction - target),
            torch.zeros_like(prediction),
        ),
        "legacy_time_nll": -intercept - wd_clamped + integral,
        "body_mask": body_mask,
        "wd_clamped": wd_clamped,
        "log_mse_hidden_credit": log_scalar.unsqueeze(-1) * quantity_w,
        "raw_squared_error_hidden_credit": (
            raw_scalar.unsqueeze(-1) * quantity_w
        ),
        "body_mae_hidden_credit": body_scalar.unsqueeze(-1) * quantity_w,
        "legacy_time_nll_hidden_credit": time_scalar.unsqueeze(-1) * time_w,
    }
    for name, value in result.items():
        if value.dtype != torch.bool:
            _require_finite(name, value)
    return result


def aggregate_selected_value_gradient(
    hidden_credit: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    memory_size: int,
) -> torch.Tensor:
    """Return ``sum_i,j credit_i / (N*k)`` in each selected memory row."""
    hidden_credit = _require_tensor("hidden_credit", hidden_credit)
    selected_indices = _require_tensor("selected_indices", selected_indices)
    if hidden_credit.ndim != 2 or hidden_credit.size(0) == 0:
        raise ValueError("hidden_credit must have shape [N, hidden] with N > 0")
    if selected_indices.ndim != 2 or selected_indices.size(0) != hidden_credit.size(0):
        raise ValueError("selected_indices must have shape [N, k]")
    if selected_indices.size(1) == 0:
        raise ValueError("selected_indices must select at least one row")
    if selected_indices.dtype != torch.long:
        raise ValueError("selected_indices must use torch.long indices")
    if selected_indices.device != hidden_credit.device:
        raise ValueError("hidden_credit and selected_indices must share a device")
    if not isinstance(memory_size, int) or isinstance(memory_size, bool) or memory_size <= 0:
        raise ValueError("memory_size must be a positive integer")
    _require_finite("hidden_credit", hidden_credit)
    if bool(((selected_indices < 0) | (selected_indices >= memory_size)).any()):
        raise ValueError("selected_indices contains an out-of-range row")
    if selected_indices.size(1) > 1:
        sorted_indices = selected_indices.sort(dim=1).values
        if bool((sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()):
            raise ValueError("selected_indices must be distinct within each example")

    count, top_k = selected_indices.shape
    gradient = hidden_credit.new_zeros((memory_size, hidden_credit.size(1)))
    expanded_credit = hidden_credit[:, None, :].expand(-1, top_k, -1)
    gradient.index_add_(
        0,
        selected_indices.reshape(-1),
        expanded_credit.reshape(-1, hidden_credit.size(1)),
    )
    return gradient / float(count * top_k)


def usage_normalized_support_confidence(
    hidden_credit: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    memory_size: int,
    zero_tolerance: float = NORM_TOLERANCE,
) -> dict[str, torch.Tensor]:
    """Apply the fixed support-confidence transform and Frobenius norm match.

    For row ``j``, with selection count ``n_j`` and average count
    ``nbar=N*k/M``, the unscaled transform is

    ``Traw_j = summed_credit_j / (k * (n_j + nbar))``.

    This equals ``c_j * mean_selected_credit_j / k`` for
    ``c_j=n_j/(n_j+nbar)``.  The final tensor is rescaled to the Frobenius norm
    of the current B quantity-memory gradient.  A zero current gradient maps
    safely to an exact zero tensor.
    """
    if not math.isfinite(float(zero_tolerance)) or zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative")
    current = aggregate_selected_value_gradient(
        hidden_credit, selected_indices, memory_size=memory_size
    )
    count, top_k = selected_indices.shape
    counts = torch.bincount(
        selected_indices.reshape(-1), minlength=memory_size
    ).to(dtype=torch.float64)
    summed = current.to(dtype=torch.float64) * float(count * top_k)
    average_usage = float(count * top_k) / float(memory_size)
    denominator = float(top_k) * (counts + average_usage)
    raw = summed / denominator.unsqueeze(-1)
    current_64 = current.to(dtype=torch.float64)
    current_norm = torch.linalg.vector_norm(current_64)
    raw_norm = torch.linalg.vector_norm(raw)
    if float(current_norm) <= zero_tolerance:
        if float(raw_norm) > zero_tolerance:
            raise ValueError("zero current gradient produced a nonzero transform")
        transformed = torch.zeros_like(raw)
        scale = torch.zeros((), dtype=torch.float64, device=raw.device)
    else:
        if float(raw_norm) <= zero_tolerance:
            raise ValueError("nonzero current gradient produced a zero transform")
        scale = current_norm / raw_norm
        transformed = raw * scale
    if not bool(torch.isfinite(transformed).all() and torch.isfinite(scale)):
        raise ValueError("usage-normalized transform is non-finite")
    return {
        "current_gradient": current_64,
        "transformed_gradient": transformed,
        "raw_transformed_gradient": raw,
        "selection_counts": counts,
        "average_usage": torch.tensor(
            average_usage, dtype=torch.float64, device=raw.device
        ),
        "norm_match_scale": scale,
    }


def cyclic_row_shifts(rows: torch.Tensor) -> torch.Tensor:
    """Return every non-identity cyclic row shift, from 1 through M-1."""
    rows = _require_tensor("rows", rows)
    if rows.ndim < 1 or rows.size(0) < 2:
        raise ValueError("rows must contain at least two rows")
    _require_finite("rows", rows)
    return torch.stack(
        [torch.roll(rows, shifts=shift, dims=0) for shift in range(1, rows.size(0))]
    )


def directional_dot_cosine(
    source_gradient: torch.Tensor,
    held_gradient: torch.Tensor,
    *,
    tolerance: float = DIRECTION_TOLERANCE,
) -> dict[str, float | bool]:
    """Measure the first-order effect of stepping along ``-source_gradient``."""
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    source = _require_tensor("source_gradient", source_gradient)
    held = _require_tensor("held_gradient", held_gradient)
    if source.shape != held.shape or source.numel() == 0:
        raise ValueError("gradient tensors must be nonempty and have equal shape")
    _require_finite("source_gradient", source)
    _require_finite("held_gradient", held)
    source = source.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    held = held.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    dot = float(torch.dot(source, held))
    source_norm = float(torch.linalg.vector_norm(source))
    held_norm = float(torch.linalg.vector_norm(held))
    informative = source_norm > tolerance and held_norm > tolerance
    cosine = dot / (source_norm * held_norm) if informative else 0.0
    return {
        "dot": dot,
        "cosine": cosine,
        "source_norm": source_norm,
        "held_norm": held_norm,
        "predicted_first_order_change_for_negative_source_step": -dot,
        "informative": informative,
        "positive_direction": dot > tolerance,
        "non_conflicting_direction": dot >= -tolerance,
    }


__all__ = [
    "DIRECTION_TOLERANCE",
    "NORM_TOLERANCE",
    "aggregate_selected_value_gradient",
    "cyclic_row_shifts",
    "directional_dot_cosine",
    "reconstruct_hidden_credits",
    "usage_normalized_support_confidence",
]
