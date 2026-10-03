"""Stateless, training-only raw auxiliary gradient control on one existing graph.

This module never performs a forward, backward, optimizer step, or RNG draw.
Only the caller's final backward writes parameter.grad. Norm-only zeros for
unused parameters must not be assigned to .grad (AdamW distinguishes None).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RawAuxGradientControl:
    cap_ratio: float = 1.0
    denominator_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        # This version implements the one frozen policy, not a hyperparameter search.
        _require(type(self.cap_ratio) in (int, float) and self.cap_ratio == 1.0,
                 "raw auxiliary cap_ratio must equal the frozen value 1")
        _require(type(self.denominator_epsilon) in (int, float) and self.denominator_epsilon == 1e-12,
                 "raw auxiliary denominator_epsilon must equal the frozen value 1e-12")

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "global_base_norm_ratio_v1",
            "cap_ratio": float(self.cap_ratio), "denominator_epsilon": float(self.denominator_epsilon),
            "base": "mean(time_loss + log_qty_loss)", "raw": "mean(alpha * raw_qty_loss)",
            "quantity_scale": 1.0, "norm": "all_active_parameters_l2_float64_current_device",
            "unused_gradient": "zero_for_norm_only_preserve_grad_None",
            "detached": True, "training_only": True, "stateful": False,
            "raw_zero_multiplier": 1.0, "base_zero_nonzero_raw_multiplier": 0.0,
            "component_gradient_queries": 2,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }


def finite_gradient_norm(gradients, *, device: torch.device) -> torch.Tensor:
    """Compute the concatenated norm without materializing a concatenated vector."""
    squares = torch.zeros((), dtype=torch.float64, device=device)
    for gradient in gradients:
        if gradient is None:
            continue
        _require(gradient.device == device, "Gradient device mismatch")
        _require(bool(torch.isfinite(gradient).all()), "Nonfinite raw auxiliary component/combined gradient")
        squares = squares + gradient.detach().double().square().sum()
    result = squares.sqrt()
    _require(bool(torch.isfinite(result)), "Nonfinite raw auxiliary gradient norm")
    return result


def _ordered_parameters(named_parameters):
    named = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    _require(bool(named), "Raw auxiliary control requires active parameters")
    _require(len({name for name, _ in named}) == len(named)
             and len({id(parameter) for _, parameter in named}) == len(named),
             "Raw auxiliary control requires ordered unique active parameters")
    device = named[0][1].device
    _require(all(parameter.device == device for _, parameter in named), "Active parameter device mismatch")
    return tuple(parameter for _, parameter in named), device


def apply_raw_aux_control(outputs, named_parameters, *, objective, control: RawAuxGradientControl,
                          training: bool) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return effective training outputs and detached JSON scalar diagnostics.

    ``outputs`` remains unchanged and retains the uncapped reference metrics.
    The returned copy replaces objective/quantity keys for the final backward
    and sparse task-gradient audit. All losses use the same already-built graph.
    """
    _require(type(control) is RawAuxGradientControl, "Frozen RawAuxGradientControl required")
    _require(training and torch.is_grad_enabled(), "Raw auxiliary controller is training-only; validation is disabled")
    _require(objective is not None and objective.name == "mixed_original"
             and objective.quantity_scale == 1.0 and math.isfinite(objective.alpha) and objective.alpha > 0,
             "Raw auxiliary control requires uncapped mixed_original with quantity_scale=1")
    parameters, device = _ordered_parameters(named_parameters)
    time_loss, log_loss, raw_scaled = (outputs[key] for key in ("time_loss", "log_qty_loss", "raw_qty_loss"))
    _require(time_loss.numel() > 0 and time_loss.shape == log_loss.shape == raw_scaled.shape,
             "Aligned nonempty per-target losses required")
    _require(all(value.device == device and bool(torch.isfinite(value).all())
                 for value in (time_loss, log_loss, raw_scaled)), "Nonfinite raw auxiliary component loss")
    base = time_loss + log_loss
    weighted_raw = objective.alpha * raw_scaled
    _require(bool(torch.isfinite(base).all()) and bool(torch.isfinite(weighted_raw).all()),
             "Nonfinite raw auxiliary weighted loss")
    base_grads = torch.autograd.grad(base.mean(), parameters, retain_graph=True,
                                     create_graph=False, allow_unused=True)
    raw_grads = torch.autograd.grad(weighted_raw.mean(), parameters, retain_graph=True,
                                    create_graph=False, allow_unused=True)
    base_norm = finite_gradient_norm(base_grads, device=device)
    raw_norm = finite_gradient_norm(raw_grads, device=device)
    with torch.no_grad():
        s = (torch.ones_like(raw_norm) if raw_norm.item() == 0.0 else torch.minimum(
            torch.ones_like(raw_norm), control.cap_ratio * base_norm / (raw_norm + control.denominator_epsilon)))
        s = s.detach()
        _require(bool(torch.isfinite(s)) and 0.0 <= s.item() <= 1.0, "Nonfinite or amplifying raw multiplier")
        effective_raw_norm = finite_gradient_norm(
            (None if gradient is None else s * gradient for gradient in raw_grads), device=device)
        expected_joint_norm = finite_gradient_norm(
            (None if b is None and r is None else
             (0 if b is None else b) + s * (0 if r is None else r)
             for b, r in zip(base_grads, raw_grads, strict=True)), device=device)
        bound_passed = effective_raw_norm.item() <= control.cap_ratio * base_norm.item() * (1 + 1e-5) + 1e-8
        _require(bound_passed, "Raw auxiliary gradient norm bound failed")
    # Preserve the specified (time+log)+s*raw arithmetic; do not differentiate s.
    effective = base + s * weighted_raw
    quantity = log_loss + s * weighted_raw
    _require(bool(torch.isfinite(effective).all()) and bool(torch.isfinite(quantity).all()),
             "Nonfinite effective raw auxiliary training loss")
    adjusted = dict(outputs)
    adjusted.update(objective_loss=effective, joint_loss=effective,
                    quantity_train_loss=quantity, loss_quantity=quantity,
                    adaptive_objective_loss=effective, adaptive_quantity_train_loss=quantity)
    diagnostics = {
        "count": int(time_loss.numel()), "base_norm": float(base_norm.item()),
        "weighted_raw_norm": float(raw_norm.item()), "effective_raw_norm": float(effective_raw_norm.item()),
        "s": float(s.item()), "effective_alpha": float(s.item() * objective.alpha),
        "predicted_composed_joint_norm_preclip": float(expected_joint_norm.item()),
        "s_less_than_one": bool(s.item() < 1.0), "norm_bound_passed": bool(bound_passed),
        "component_gradient_queries": 2,
    }
    return adjusted, diagnostics


def summarize_epoch(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep sample-weighted scalar means distinct from unweighted batch rates."""
    _require(bool(records) and all(record["count"] > 0 for record in records), "Nonempty raw auxiliary epoch required")
    count = sum(record["count"] for record in records)
    scalars = ("base_norm", "weighted_raw_norm", "effective_raw_norm", "s", "effective_alpha",
               "composed_joint_norm_preclip", "common_clip_factor")
    result = {"count": count, "batches": len(records), "nonfinite_count": 0,
              "sample_weighted": {key: {
                  "mean": math.fsum(record["count"] * record[key] for record in records) / count,
                  "min": min(record[key] for record in records), "max": max(record[key] for record in records),
              } for key in scalars}}
    for key, predicate in (("capped", "s_less_than_one"), ("clipped", "clipped"), ("norm_above_max", "norm_above_max")):
        number = sum(bool(record[predicate]) for record in records)
        result[key + "_batch_count"] = number
        result[key + "_batch_rate"] = number / len(records)
    result["component_gradient_queries"] = sum(record["component_gradient_queries"] for record in records)
    result["norm_bound_passed"] = all(record["norm_bound_passed"] for record in records)
    return result
