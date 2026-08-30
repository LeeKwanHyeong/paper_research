"""Explicit, per-series stability policy for Titans inner-loop gradients."""

from __future__ import annotations

import torch


def clip_associative_gradients(
    gradients: tuple[torch.Tensor, ...], max_norm: float | None,
) -> tuple[torch.Tensor, ...]:
    """Clip the joint norm of all memory gradients independently per row.

    Scaling before squaring avoids overflow in the norm computation. The
    clipping remains differentiable through the inner update; None preserves
    the historical, unbounded Titans recurrence exactly.
    """
    if max_norm is None:
        return gradients
    scale = torch.stack(
        [gradient.flatten(1).abs().amax(dim=1) for gradient in gradients], dim=1,
    ).amax(dim=1).detach().clamp_min(1.0)
    scaled_squares = torch.stack([
        (gradient.flatten(1) / scale[:, None]).square().sum(dim=1)
        for gradient in gradients
    ], dim=1).sum(dim=1)
    scaled_norm = scaled_squares.clamp_min(1e-24).sqrt()
    coefficient = ((max_norm / scale) / scaled_norm).clamp(max=1.0)
    return tuple(
        gradient * coefficient.view(-1, *([1] * (gradient.ndim - 1)))
        for gradient in gradients
    )
