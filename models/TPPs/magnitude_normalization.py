from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MagnitudeContext:
    """Causal normalization state shared by magnitude input and decoder."""

    normalized_history: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    context_count: torch.Tensor
    history_mask: torch.Tensor


def history_mask_without_appended_target(mask: torch.Tensor) -> torch.Tensor:
    """Remove the final valid token, which weekly loaders reserve as target."""
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, sequence].")
    history_mask = mask.bool().clone()
    positions = torch.arange(mask.size(1), device=mask.device).view(1, -1)
    last_positions = torch.where(
        mask,
        positions,
        torch.full_like(positions, -1),
    ).max(dim=1).values
    has_target = last_positions >= 0
    if has_target.any():
        history_mask[has_target, last_positions[has_target]] = False
    return history_mask


def reconstruct_log2_quantity(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    *,
    num_real_marks: int,
) -> torch.Tensor:
    """Reconstruct the exact continuous log2 quantity from mark factorization."""
    if marks.shape != residuals.shape:
        raise ValueError("marks and residuals must have the same shape.")
    real_mark_count = int(num_real_marks)
    if real_mark_count < 1:
        raise ValueError("num_real_marks must be positive.")
    safe_marks = marks.clamp(min=0, max=real_mark_count - 1)
    return safe_marks.to(dtype=residuals.dtype) + residuals


def reconstruct_raw_quantity(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    *,
    num_real_marks: int,
) -> torch.Tensor:
    """Reconstruct raw quantity at the marked-data interface."""
    return torch.exp2(
        reconstruct_log2_quantity(
            marks,
            residuals,
            num_real_marks=num_real_marks,
        )
    )


def _validate_context_inputs(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if marks.shape != mask.shape or residuals.shape != marks.shape:
        raise ValueError("marks, residuals, and mask must have identical shapes.")
    if not torch.isfinite(residuals).all():
        raise ValueError("Magnitude residuals must be finite.")


def _raw_history_moments(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_real_marks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw quantity and masked population moments over causal history."""
    _validate_context_inputs(marks, residuals, mask)
    history_mask = history_mask_without_appended_target(mask)
    context_count = history_mask.sum(dim=1, keepdim=True)
    if (context_count < 1).any():
        raise ValueError("Raw magnitude normalization requires at least one history event.")

    raw_log_qty = reconstruct_log2_quantity(
        marks,
        residuals,
        num_real_marks=num_real_marks,
    )
    # Excluded target/padding values must not participate even transiently;
    # replacing them before exp2 also avoids irrelevant overflow.
    raw_qty = torch.exp2(torch.where(history_mask, raw_log_qty, torch.zeros_like(raw_log_qty)))
    if not torch.isfinite(raw_qty[history_mask]).all():
        raise ValueError("Reconstructed raw history quantities must be finite.")
    history_weight = history_mask.to(dtype=raw_qty.dtype)
    count = context_count.to(dtype=raw_qty.dtype)
    history_mean = (raw_qty * history_weight).sum(dim=1, keepdim=True) / count
    centered = raw_qty - history_mean
    history_var = (centered.square() * history_weight).sum(dim=1, keepdim=True) / count
    return raw_qty, history_mask, context_count, history_mean, history_var


def _build_context(
    magnitude: torch.Tensor,
    history_mask: torch.Tensor,
    context_count: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> MagnitudeContext:
    if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
        raise FloatingPointError("Magnitude context center and scale must be finite.")
    if (scale <= 0).any():
        raise ValueError("Magnitude context scale must be positive.")
    normalized = ((magnitude - center) / scale) * history_mask.to(dtype=magnitude.dtype)
    if not torch.isfinite(normalized).all():
        raise FloatingPointError("Normalized magnitude history must be finite.")
    return MagnitudeContext(
        normalized_history=normalized,
        center=center,
        scale=scale,
        context_count=context_count,
        history_mask=history_mask,
    )


def build_global_magnitude_context(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_real_marks: int,
    global_mean: float,
    global_std: float,
    sigma_floor: float,
) -> MagnitudeContext:
    """Build M0 train-global normalization without target or padding leakage."""
    _validate_context_inputs(marks, residuals, mask)
    mean_value = float(global_mean)
    std_value = float(global_std)
    floor_value = float(sigma_floor)
    if not torch.isfinite(torch.tensor([mean_value, std_value, floor_value])).all():
        raise ValueError("Magnitude normalization constants must be finite.")
    if std_value <= 0.0 or floor_value <= 0.0:
        raise ValueError("global_std and sigma_floor must be positive.")

    history_mask = history_mask_without_appended_target(mask)
    log_qty = reconstruct_log2_quantity(
        marks,
        residuals,
        num_real_marks=num_real_marks,
    )
    center = torch.full(
        (marks.size(0), 1),
        mean_value,
        dtype=residuals.dtype,
        device=residuals.device,
    )
    scale = torch.full_like(center, max(std_value, floor_value))
    normalized = (log_qty - center) / scale
    normalized = normalized * history_mask.to(dtype=normalized.dtype)
    return MagnitudeContext(
        normalized_history=normalized,
        center=center,
        scale=scale,
        context_count=history_mask.sum(dim=1, keepdim=True),
        history_mask=history_mask,
    )


def build_raw_global_magnitude_context(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_real_marks: int,
    global_mean: float,
    global_std: float,
    sigma_floor: float,
) -> MagnitudeContext:
    """Build Q0 from fixed train-global raw-quantity moments."""
    _validate_context_inputs(marks, residuals, mask)
    constants = torch.tensor(
        [global_mean, global_std, sigma_floor],
        dtype=torch.float64,
    )
    if not torch.isfinite(constants).all():
        raise ValueError("Raw global normalization constants must be finite.")
    if float(global_std) <= 0.0 or float(sigma_floor) <= 0.0:
        raise ValueError("global_std and sigma_floor must be positive.")

    raw_qty, history_mask, context_count, _, _ = _raw_history_moments(
        marks,
        residuals,
        mask,
        num_real_marks=num_real_marks,
    )
    center = torch.full(
        (marks.size(0), 1),
        float(global_mean),
        dtype=residuals.dtype,
        device=residuals.device,
    )
    scale = torch.full_like(center, max(float(global_std), float(sigma_floor)))
    return _build_context(raw_qty, history_mask, context_count, center, scale)


def build_causal_revin_magnitude_context(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_real_marks: int,
    revin_eps: float,
) -> MagnitudeContext:
    """Build Q1 using causal masked raw mean and population variance."""
    eps = float(revin_eps)
    if not torch.isfinite(torch.tensor(eps)) or eps <= 0.0:
        raise ValueError("revin_eps must be finite and positive.")
    raw_qty, history_mask, context_count, history_mean, history_var = (
        _raw_history_moments(
            marks,
            residuals,
            mask,
            num_real_marks=num_real_marks,
        )
    )
    scale = torch.sqrt(history_var + eps)
    return _build_context(
        raw_qty,
        history_mask,
        context_count,
        history_mean,
        scale,
    )


def build_causal_shrinkage_revin_magnitude_context(
    marks: torch.Tensor,
    residuals: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_real_marks: int,
    global_mean: float,
    global_var: float,
    sigma_floor: float,
    shrinkage_k: float,
) -> MagnitudeContext:
    """Build Q2 by mixing causal history and train-global raw moments."""
    constants = torch.tensor(
        [global_mean, global_var, sigma_floor, shrinkage_k],
        dtype=torch.float64,
    )
    if not torch.isfinite(constants).all():
        raise ValueError("Shrinkage RevIN constants must be finite.")
    if float(global_var) < 0.0:
        raise ValueError("global_var must be non-negative.")
    if float(sigma_floor) <= 0.0 or float(shrinkage_k) <= 0.0:
        raise ValueError("sigma_floor and shrinkage_k must be positive.")

    raw_qty, history_mask, context_count, history_mean, history_var = (
        _raw_history_moments(
            marks,
            residuals,
            mask,
            num_real_marks=num_real_marks,
        )
    )
    count = context_count.to(dtype=raw_qty.dtype)
    alpha = count / (count + float(shrinkage_k))
    global_mean_tensor = torch.as_tensor(
        float(global_mean),
        dtype=raw_qty.dtype,
        device=raw_qty.device,
    )
    global_var_tensor = torch.as_tensor(
        float(global_var),
        dtype=raw_qty.dtype,
        device=raw_qty.device,
    )
    center = alpha * history_mean + (1.0 - alpha) * global_mean_tensor
    second_moment = (
        alpha * (history_var + history_mean.square())
        + (1.0 - alpha) * (global_var_tensor + global_mean_tensor.square())
    )
    variance = second_moment - center.square()
    variance = variance.clamp_min(float(sigma_floor) ** 2)
    scale = torch.sqrt(variance)
    return _build_context(raw_qty, history_mask, context_count, center, scale)


def denormalize_magnitude(
    normalized: torch.Tensor,
    context: MagnitudeContext,
) -> torch.Tensor:
    """Map normalized predictions back to their configured magnitude domain."""
    center = context.center
    scale = context.scale
    if normalized.ndim == 1 and center.ndim == 2 and center.size(-1) == 1:
        center = center.squeeze(-1)
        scale = scale.squeeze(-1)
    while center.ndim < normalized.ndim:
        center = center.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
    return center + scale * normalized


def normalized_magnitude_target(
    magnitude: torch.Tensor,
    context: MagnitudeContext,
) -> torch.Tensor:
    """Normalize targets with the same causal context used by the decoder."""
    center = context.center
    scale = context.scale
    if magnitude.ndim == 1 and center.ndim == 2 and center.size(-1) == 1:
        center = center.squeeze(-1)
        scale = scale.squeeze(-1)
    while center.ndim < magnitude.ndim:
        center = center.unsqueeze(-1)
        scale = scale.unsqueeze(-1)
    return (magnitude - center) / scale


def safe_exp2(
    log_qty: torch.Tensor,
    *,
    clamp_min: float,
    clamp_max: float,
) -> torch.Tensor:
    """Reconstruct positive quantity with explicit numeric-only clamp bounds."""
    minimum = float(clamp_min)
    maximum = float(clamp_max)
    if not minimum < maximum:
        raise ValueError("exp2 clamp_min must be smaller than clamp_max.")
    return torch.exp2(log_qty.clamp(min=minimum, max=maximum))
