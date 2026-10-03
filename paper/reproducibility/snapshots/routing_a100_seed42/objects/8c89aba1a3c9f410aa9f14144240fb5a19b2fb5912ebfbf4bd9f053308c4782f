"""Frozen 2x2 quantity objective/output comparison primitives.

This module is deliberately functional: it neither changes a model nor creates
an optimizer.  The ``B_LOG_ORIGINAL`` branch delegates to the frozen B
arithmetic so it remains a bitwise control for a joint batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
import torch
import torch.nn.functional as F

from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch
from paper.scripts.time_quantity_diagnostic import (
    _validate_model as _validate_b_model,
    task_outputs as b_task_outputs,
)


class QuantityCase(str, Enum):
    """The factorized loss/link conditions, with B as the frozen control."""

    B_LOG_ORIGINAL = "B_log_original"
    RAW_ORIGINAL = "raw_original"
    LOG_SOFTPLUS = "log_softplus"
    RAW_SOFTPLUS = "raw_softplus"

    @property
    def uses_raw_loss(self) -> bool:
        return self in {self.RAW_ORIGINAL, self.RAW_SOFTPLUS}

    @property
    def uses_softplus_link(self) -> bool:
        return self in {self.LOG_SOFTPLUS, self.RAW_SOFTPLUS}


@dataclass(frozen=True)
class QuantityStatistics:
    """Train-only constants; ``mu`` and ``raw_scale`` use different populations."""

    mu: float
    raw_scale: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mu) or self.mu <= 0.0:
            raise ValueError("mu must be finite and positive")
        if not math.isfinite(self.raw_scale) or self.raw_scale < 1.0:
            raise ValueError("raw_scale must be finite and at least one")
        # Validate the derived calibration values at construction, rather than
        # allowing a bad identity to reach an optimizer run.
        try:
            valid_calibration = math.isfinite(self.q0) and self.q0 > 0.0 and math.isfinite(self.z0)
        except (OverflowError, ValueError):
            valid_calibration = False
        if not valid_calibration:
            raise ValueError("mu does not produce finite positive calibration values")

    @property
    def q0(self) -> float:
        return math.expm1(self.mu)

    @property
    def z0(self) -> float:
        return math.log(math.expm1(self.mu))


@dataclass(frozen=True)
class QuantityCaseIdentity:
    """Small immutable resume identity for a selected quantity condition."""

    case: QuantityCase
    statistics: QuantityStatistics

    def to_json(self) -> str:
        return json.dumps(
            {"case": self.case.value, "mu": self.statistics.mu, "raw_scale": self.statistics.raw_scale},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str) -> "QuantityCaseIdentity":
        parsed = json.loads(value)
        if set(parsed) != {"case", "mu", "raw_scale"}:
            raise ValueError("Quantity identity has an unexpected schema")
        return cls(QuantityCase(parsed["case"]), QuantityStatistics(float(parsed["mu"]), float(parsed["raw_scale"])))

    def require_match(self, stored_json: str) -> None:
        if self.to_json() != QuantityCaseIdentity.from_json(stored_json).to_json():
            raise ValueError("Quantity objective resume identity mismatch")


def _as_train_quantity(values: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or values.numel() == 0:
        raise ValueError(f"{name} must be a nonempty tensor")
    if not torch.is_floating_point(values):
        # Train statistics are float64; converting large count integers through
        # float32 would silently alter values above 2**24.
        values = values.double()
    if not bool(torch.isfinite(values).all()) or not bool((values >= 0).all()):
        raise ValueError(f"{name} must be finite and nonnegative")
    return values


def fit_train_quantity_statistics(
    all_train_quantities: torch.Tensor,
    canonical_train_targets: torch.Tensor,
    *,
    split: str,
) -> QuantityStatistics:
    """Fit B's all-row log statistic and raw RMSE scale from train only.

    The two populations intentionally differ: B initializes from every train
    row while raw loss is scaled using canonical next-event train targets.
    """
    if split != "train":
        raise ValueError("Quantity statistics may only be fit on the train split")
    all_rows = _as_train_quantity(all_train_quantities, "all_train_quantities")
    targets = _as_train_quantity(canonical_train_targets, "canonical_train_targets")
    mu = float(torch.log1p(all_rows.double()).mean().item())
    scale = max(1.0, math.sqrt(float(targets.double().square().mean().item())))
    return QuantityStatistics(mu=mu, raw_scale=scale)


def _validate_targets(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.ndim != 1 or targets.ndim != 1 or logits.shape != targets.shape or logits.numel() == 0:
        raise ValueError("logits and targets must be nonempty aligned vectors")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if not bool(torch.isfinite(targets).all()) or not bool((targets >= 0).all()):
        raise ValueError("targets must be finite and nonnegative")


def quantity_prediction(logits: torch.Tensor, statistics: QuantityStatistics, case: QuantityCase) -> torch.Tensor:
    """Return raw quantity without clipping, weighting, or quantile routing."""
    if not isinstance(case, QuantityCase):
        case = QuantityCase(case)
    if logits.ndim != 1 or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be a finite vector")
    if case.uses_softplus_link:
        log_two = math.log(2.0)
        return (statistics.q0 / log_two) * F.softplus((2.0 * log_two) * (logits - statistics.z0))
    # Keep B's tensor operations unchanged for the original output link.
    return torch.expm1(F.softplus(logits))


def quantity_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    statistics: QuantityStatistics,
    case: QuantityCase,
) -> dict[str, torch.Tensor]:
    """Compute an elementwise selected loss plus raw prediction and log MSE."""
    if not isinstance(case, QuantityCase):
        case = QuantityCase(case)
    _validate_targets(logits, targets)
    if case.uses_softplus_link:
        prediction = quantity_prediction(logits, statistics, case)
        log_mse = torch.square(torch.log1p(prediction) - torch.log1p(targets))
    else:
        # B arithmetic deliberately does not round-trip through expm1/log1p.
        # This is also the standalone log-loss definition for raw/original.
        location = F.softplus(logits)
        prediction = torch.expm1(location)
        log_mse = F.mse_loss(
            location,
            torch.log1p(targets.clamp_min(0.0)),
            reduction="none",
        )
    if not bool(torch.isfinite(prediction).all()) or not bool((prediction >= 0).all()):
        raise ValueError("Selected quantity link produced a nonfinite prediction")
    raw_mse = torch.square((prediction - targets) / statistics.raw_scale)
    train_loss = raw_mse if case.uses_raw_loss else log_mse
    if not all(bool(torch.isfinite(value).all()) for value in (log_mse, raw_mse, train_loss)):
        raise ValueError("Selected quantity loss is nonfinite")
    return {"train_loss": train_loss, "point_prediction": prediction, "log_mse": log_mse, "raw_mse": raw_mse}


def joint_causal_batch_objective(
    model: torch.nn.Module,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    *,
    statistics: QuantityStatistics,
    case: QuantityCase,
) -> dict[str, torch.Tensor]:
    """Evaluate ``L_t + L_q`` on B's causal target graph without mutation."""
    if not isinstance(case, QuantityCase):
        case = QuantityCase(case)
    if case is QuantityCase.B_LOG_ORIGINAL:
        # Exact B control: do not reconstruct equivalent arithmetic here.
        outputs = b_task_outputs(model, dts, mask, quantities, "joint")
        required = ("pred_qty", "time_loss", "quantity_train_loss", "objective_loss")
        if not all(bool(torch.isfinite(outputs[key]).all()) for key in required):
            raise ValueError("Selected joint objective is nonfinite")
        return outputs
    _validate_b_model(model, "joint")
    if mask.dtype != torch.bool or dts.shape != mask.shape or quantities.shape != mask.shape or mask.ndim != 2:
        raise ValueError("Expected aligned two-dimensional dts/quantity/bool mask")
    if mask.shape[0] == 0:
        raise ValueError("Empty batch")
    observed_dts, observed_quantities = dts[mask], quantities[mask]
    if not bool(torch.isfinite(observed_dts).all()) or not bool((observed_dts >= 0).all()):
        raise ValueError("Observed durations must be finite and nonnegative")
    if not bool(torch.isfinite(observed_quantities).all()) or not bool((observed_quantities >= 0).all()):
        raise ValueError("Observed quantities must be finite and nonnegative")
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    ids = torch.arange(dts.size(0), device=dts.device)
    target_positions, history_positions = lengths - 1, lengths - 2
    history_quantities = quantities.clone()
    history_quantities[ids, target_positions] = 0.0
    memory_write_mask = mask.clone()
    memory_write_mask[ids, target_positions] = False
    time_encoded, quantity_encoded = model.encode_task_states(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
    true_qty = quantities[ids, target_positions].float()
    logits = model.quantity_head(quantity_encoded[ids, history_positions]).squeeze(-1)
    quantity = quantity_loss(logits, true_qty, statistics, case)
    time_loss = -model.log_f_dt(time_encoded[ids, history_positions], dts[ids, target_positions].float())
    objective = time_loss + quantity["train_loss"]
    if not all(bool(torch.isfinite(value).all()) for value in (time_loss, quantity["train_loss"], objective)):
        raise ValueError("Selected joint objective is nonfinite")
    return {
        "objective_loss": objective,
        "joint_loss": objective,
        "time_loss": time_loss,
        "loss_time": time_loss,
        "quantity_train_loss": quantity["train_loss"],
        "loss_quantity": quantity["train_loss"],
        "log_qty_loss": quantity["log_mse"],
        "raw_qty_loss": quantity["raw_mse"],
        "true_qty": true_qty,
        "pred_qty": quantity["point_prediction"],
        "history_length": lengths - 1,
    }
