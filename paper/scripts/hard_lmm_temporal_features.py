"""Observed-window temporal summaries and deterministic interior interventions.

Inputs contain observed events only. The caller must remove target and padding
before calling; this module does not load data or infer that boundary.
"""

from __future__ import annotations

import math

import torch


def _validate(dts: torch.Tensor, quantities: torch.Tensor) -> None:
    if not isinstance(dts, torch.Tensor) or not isinstance(quantities, torch.Tensor):
        raise ValueError("dts and quantities must be CPU tensors")
    if dts.ndim != 1 or quantities.ndim != 1 or dts.shape != quantities.shape:
        raise ValueError("dts and quantities must share shape [observed history]")
    if not dts.numel():
        raise ValueError("Observed history must be nonempty")
    for tensor in (dts, quantities):
        if tensor.device.type != "cpu":
            raise ValueError("Observed history tensors must be on CPU")
        if tensor.is_complex() or not bool(torch.isfinite(tensor).all()):
            raise ValueError("Observed history must be real and finite")
        if bool((tensor < 0).any()):
            raise ValueError("Observed gaps and quantities must be nonnegative")


@torch.no_grad()
def observed_features(dts: torch.Tensor, quantities: torch.Tensor
                      ) -> dict[str, float | int | None]:
    """Describe an observed window without treating its boundary gap as age.

    ``age_distortion`` is RMS distance between normalized elapsed age and
    normalized event-index age. It is undefined for fewer than three events or
    zero internal span. Changes and internal-gap summaries are undefined for a
    singleton history; ``first_gap`` and ``last_gap`` still describe that event.
    """
    _validate(dts, quantities)
    gaps, log_qty = dts.double(), quantities.double().log1p()
    length = len(gaps)
    mean_log_qty = log_qty.mean()
    result: dict[str, float | int | None] = {
        "history_length": length,
        "mean_log_quantity": mean_log_qty.item(),
        "latest_deviation": (log_qty[-1] - mean_log_qty).item(),
        "count_change_rms": None,
        "log_mean_internal_gap": None,
        "internal_span": None,
        "first_gap": gaps[0].item(),
        "last_gap": gaps[-1].item(),
        "age_distortion": None,
    }
    if length > 1:
        internal_gaps = gaps[1:]
        span = internal_gaps.sum()
        result["count_change_rms"] = log_qty.diff().square().mean().sqrt().item()
        result["log_mean_internal_gap"] = internal_gaps.mean().log1p().item()
        result["internal_span"] = span.item()
        if length >= 3 and bool(span > 0):
            # age[i] = sum(gaps[i+1:]); last observed event has age exactly zero.
            ages = torch.cat((internal_gaps.flip(0).cumsum(0).flip(0), gaps.new_zeros(1)))
            event_ages = torch.arange(length - 1, -1, -1, dtype=torch.float64) / (length - 1)
            result["age_distortion"] = (ages / span - event_ages).square().mean().sqrt().item()
    for name, value in result.items():
        if value is not None and not math.isfinite(value):
            raise ValueError(f"Observed feature is non-finite: {name}")
    return result


@torch.no_grad()
def reverse_interior(dts: torch.Tensor, quantities: torch.Tensor,
                     modality: str = "gap") -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse only positions 1..H-2 in one modality, preserving both endpoints.

    Returns independent clones, including for short or constant histories where
    the intervention makes no value change. No target or padding is accepted as
    part of this observed-history API.
    """
    _validate(dts, quantities)
    if modality not in {"gap", "quantity"}:
        raise ValueError("modality must be 'gap' or 'quantity'")
    next_dts, next_quantities = dts.clone(), quantities.clone()
    if dts.numel() >= 4:
        selected = next_dts if modality == "gap" else next_quantities
        selected[1:-1] = selected[1:-1].flip(0)
    return next_dts, next_quantities
