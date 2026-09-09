"""Pure helpers for the Frozen-B shared-block objective-transfer audit.

This module deliberately has no dataset, checkpoint, or model imports.  It
provides the two small differentiable instruments needed by the audit:

* a train-only, 130-parameter heteroscedastic log-normal duration head; and
* the exact zero-initialized gradient of ``z' = z + A z`` at a shared block.

The directional comparison is also pure.  Every update arm is normalized to
the same unit update budget, and the task-separated arm has unit norm in the
direct sum of its quantity and time adapters.  A positive gradient dot product
means that a small step along the negative source-fold gradient is predicted
to reduce the held-fold metric.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


HIDDEN_DIM = 64
TIME_HEAD_PARAMETER_COUNT = 130
SIGMA_FLOOR = 1e-3
DIRECTION_TOLERANCE = 1e-12
NORM_TOLERANCE = 1e-30
CONTROL_QUANTILE = 0.95
CONTROL_COUNT = HIDDEN_DIM - 1

CONTINUOUS_LOGNORMAL = "continuous_lognormal_density"
POSITIVE_INTEGER_LOGNORMAL = "positive_integer_round_clamp_lognormal"
OBSERVATION_MODES = {CONTINUOUS_LOGNORMAL, POSITIVE_INTEGER_LOGNORMAL}

LOG_MSE = "log_mse"
RAW_SQUARED_ERROR = "raw_squared_error"
BODY_MAE = "body_mae"
NORMALIZED_TIME_NLL = "normalized_time_nll"
QUANTITY_METRICS = (LOG_MSE, RAW_SQUARED_ERROR, BODY_MAE)
REQUIRED_METRICS = (*QUANTITY_METRICS, NORMALIZED_TIME_NLL)


def _require_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    return value


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse-softplus input must be finite and positive")
    # log(expm1(x)) loses precision near zero and overflows for very large x.
    if value > 20.0:
        return value + math.log1p(-math.exp(-value))
    return math.log(math.expm1(value))


def source_fold_time_statistics(target_dt: torch.Tensor) -> dict[str, float | int]:
    """Return the fixed source-fold initialization statistics.

    ``time_scale`` is the median duration.  Location and scale are the mean
    and population standard deviation of ``log(target_dt / time_scale)``.
    No validation or held-fold tensor is accepted by this function.
    """
    target = _require_tensor("target_dt", target_dt)
    if target.ndim != 1 or target.numel() < 2:
        raise ValueError("target_dt must be rank one with at least two values")
    target = target.detach().to(dtype=torch.float64)
    _require_finite("target_dt", target)
    if bool((target <= 0.0).any()):
        raise ValueError("target_dt must be strictly positive")
    # ``quantile`` gives the conventional midpoint median for even folds;
    # ``torch.median`` would silently choose the lower middle observation.
    time_scale = float(torch.quantile(target, 0.5))
    log_scaled = torch.log(target / time_scale)
    location_mean = float(log_scaled.mean())
    population_std = float(log_scaled.std(unbiased=False))
    if not math.isfinite(population_std) or population_std <= SIGMA_FLOOR:
        raise ValueError(
            "source-fold population log-duration std must exceed sigma floor"
        )
    return {
        "target_count": int(target.numel()),
        "time_scale": time_scale,
        "target_log_scaled_mean": location_mean,
        "target_log_scaled_population_std": population_std,
    }


class FrozenBK1LogNormalHead(nn.Module):
    """A 130-parameter K=1 duration head for frozen 64-wide B states.

    Parameters are stored in float32, matching the model checkpoint contract.
    The affine projections, positive scale, and likelihood are all evaluated
    in float64 so narrow integer bins and survival tails remain stable.
    """

    def __init__(
        self,
        *,
        time_scale: float,
        location_mean: float,
        population_std: float,
        sigma_floor: float = SIGMA_FLOOR,
    ) -> None:
        super().__init__()
        scalars = {
            "time_scale": time_scale,
            "location_mean": location_mean,
            "population_std": population_std,
            "sigma_floor": sigma_floor,
        }
        if any(not math.isfinite(float(value)) for value in scalars.values()):
            raise ValueError("duration-head initialization must be finite")
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        if sigma_floor <= 0.0 or population_std <= sigma_floor:
            raise ValueError("population_std must exceed the positive sigma floor")

        self.time_scale = float(time_scale)
        self.sigma_floor = float(sigma_floor)
        self.location_weight = nn.Parameter(
            torch.zeros((1, HIDDEN_DIM), dtype=torch.float32)
        )
        self.location_bias = nn.Parameter(
            torch.tensor([location_mean], dtype=torch.float32)
        )
        self.scale_weight = nn.Parameter(
            torch.zeros((1, HIDDEN_DIM), dtype=torch.float32)
        )
        raw_scale = _inverse_softplus(population_std - sigma_floor)
        self.scale_raw_bias = nn.Parameter(
            torch.tensor([raw_scale], dtype=torch.float32)
        )
        if sum(parameter.numel() for parameter in self.parameters()) != (
            TIME_HEAD_PARAMETER_COUNT
        ):
            raise RuntimeError("Frozen-B K=1 duration head must have 130 parameters")

    @classmethod
    def from_source_fold_targets(
        cls,
        target_dt: torch.Tensor,
        *,
        sigma_floor: float = SIGMA_FLOOR,
    ) -> tuple["FrozenBK1LogNormalHead", dict[str, float | int]]:
        statistics = source_fold_time_statistics(target_dt)
        head = cls(
            time_scale=float(statistics["time_scale"]),
            location_mean=float(statistics["target_log_scaled_mean"]),
            population_std=float(
                statistics["target_log_scaled_population_std"]
            ),
            sigma_floor=sigma_floor,
        )
        return head, statistics

    def location_and_sigma(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = _require_tensor("hidden", hidden)
        if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_DIM:
            raise ValueError(f"hidden must have shape [N, {HIDDEN_DIM}]")
        if hidden.shape[0] == 0:
            raise ValueError("hidden must contain at least one row")
        _require_finite("hidden", hidden)
        hidden64 = hidden.to(dtype=torch.float64)
        location = F.linear(
            hidden64,
            self.location_weight.to(dtype=torch.float64),
            self.location_bias.to(dtype=torch.float64),
        ).squeeze(-1)
        raw_sigma = F.linear(
            hidden64,
            self.scale_weight.to(dtype=torch.float64),
            self.scale_raw_bias.to(dtype=torch.float64),
        ).squeeze(-1)
        sigma = self.sigma_floor + F.softplus(raw_sigma)
        _require_finite("location", location)
        _require_finite("sigma", sigma)
        return location, sigma

    def log_likelihood(
        self,
        hidden: torch.Tensor,
        target_dt: torch.Tensor,
        *,
        observation_mode: str,
        is_right_censored: torch.Tensor | None = None,
        censor_threshold: float | None = None,
    ) -> torch.Tensor:
        if is_right_censored is not None and censor_threshold is not None:
            raise ValueError(
                "provide either is_right_censored or censor_threshold, not both"
            )
        if censor_threshold is not None:
            is_right_censored = _censor_mask_from_threshold(
                target_dt, censor_threshold
            )
        return normalized_lognormal_log_likelihood(
            self,
            hidden,
            target_dt,
            observation_mode=observation_mode,
            is_right_censored=is_right_censored,
        )

    def negative_log_likelihood(
        self,
        hidden: torch.Tensor,
        target_dt: torch.Tensor,
        *,
        observation_mode: str,
        is_right_censored: torch.Tensor | None = None,
        censor_threshold: float | None = None,
    ) -> torch.Tensor:
        """Return the unreduced per-row normalized negative log likelihood."""
        return -self.log_likelihood(
            hidden,
            target_dt,
            observation_mode=observation_mode,
            is_right_censored=is_right_censored,
            censor_threshold=censor_threshold,
        )


def _validate_likelihood_inputs(
    head: FrozenBK1LogNormalHead,
    hidden: torch.Tensor,
    target_dt: torch.Tensor,
    is_right_censored: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not isinstance(head, FrozenBK1LogNormalHead):
        raise ValueError("head must be a FrozenBK1LogNormalHead")
    hidden = _require_tensor("hidden", hidden)
    target = _require_tensor("target_dt", target_dt)
    if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_DIM:
        raise ValueError(f"hidden must have shape [N, {HIDDEN_DIM}]")
    if target.ndim != 1 or target.shape[0] != hidden.shape[0]:
        raise ValueError("target_dt must have shape [N]")
    if target.device != hidden.device:
        raise ValueError("hidden and target_dt must share a device")
    target = target.to(dtype=torch.float64)
    _require_finite("hidden", hidden)
    _require_finite("target_dt", target)
    if bool((target <= 0.0).any()):
        raise ValueError("target_dt must be strictly positive")
    if is_right_censored is None:
        censor = torch.zeros_like(target, dtype=torch.bool)
    else:
        censor = _require_tensor("is_right_censored", is_right_censored)
        if censor.shape != target.shape or censor.dtype != torch.bool:
            raise ValueError("is_right_censored must be bool with shape [N]")
        if censor.device != hidden.device:
            raise ValueError("censor mask must share the hidden device")
    return hidden, target, censor


def _stable_log_probability_difference(
    log_larger: torch.Tensor, log_smaller: torch.Tensor
) -> torch.Tensor:
    if log_larger.shape != log_smaller.shape:
        raise ValueError("log-probability shapes differ")
    if log_larger.dtype != torch.float64 or log_smaller.dtype != torch.float64:
        raise ValueError("stable log difference requires float64 inputs")
    delta = log_smaller - log_larger
    if bool((delta > 1e-14).any()):
        raise ValueError("subtracted probability exceeds the minuend")
    delta = delta.clamp_max(0.0)
    log_two = math.log(2.0)
    complement = torch.where(
        delta < -log_two,
        torch.log1p(-torch.exp(delta)),
        torch.log(-torch.expm1(delta)),
    )
    result = log_larger + complement
    _require_finite("log probability difference", result)
    return result


def normalized_lognormal_log_likelihood(
    head: FrozenBK1LogNormalHead,
    hidden: torch.Tensor,
    target_dt: torch.Tensor,
    *,
    observation_mode: str,
    is_right_censored: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return continuous, integer-bin, or right-censored log likelihood.

    For positive integer observations the first code is ``F(1.5)``, ordinary
    code ``d`` is ``F(d+0.5)-F(d-0.5)``, and a censored top code is
    ``S(d-0.5)``.  For continuous observations, a censored target is
    ``S(target_dt)`` and an ordinary target uses density in original units.
    """
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError("unsupported duration observation mode")
    hidden, target, censor = _validate_likelihood_inputs(
        head, hidden, target_dt, is_right_censored
    )
    location, sigma = head.location_and_sigma(hidden)
    log_scale = math.log(head.time_scale)

    if observation_mode == CONTINUOUS_LOGNORMAL:
        log_target = torch.log(target)
        z = (log_target - log_scale - location) / sigma
        log_density = (
            -log_target
            - torch.log(sigma)
            - 0.5 * z.square()
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_survival = torch.special.log_ndtr(-z)
        result = torch.where(censor, log_survival, log_density)
    else:
        if bool((target < 1.0).any()) or not bool(
            torch.isclose(target, target.round(), atol=1e-8, rtol=0.0).all()
        ):
            raise ValueError(
                "positive-integer target_dt must be integer valued and at least one"
            )
        if bool((censor & (target < 2.0)).any()):
            raise ValueError("the positive-integer first bin cannot be censored")
        if bool(censor.any()):
            censor_codes = torch.unique(target[censor])
            if censor_codes.numel() != 1:
                raise ValueError("a batch cannot mix right-censor codes")
            if bool((target > censor_codes[0]).any()):
                raise ValueError("target_dt exceeds the right-censor code")

        upper_z = (
            torch.log(target + 0.5) - log_scale - location
        ) / sigma
        lower_z = (
            torch.log((target - 0.5).clamp_min(0.5))
            - log_scale
            - location
        ) / sigma
        result = torch.special.log_ndtr(upper_z)

        regular = (target >= 2.0) & ~censor
        if bool(regular.any()):
            index = torch.nonzero(regular, as_tuple=False).squeeze(-1)
            lower = lower_z[index]
            upper = upper_z[index]
            regular_result = torch.empty_like(lower)
            survival_side = lower > 0.0
            if bool(survival_side.any()):
                sub = torch.nonzero(survival_side, as_tuple=False).squeeze(-1)
                regular_result[sub] = _stable_log_probability_difference(
                    torch.special.log_ndtr(-lower[sub]),
                    torch.special.log_ndtr(-upper[sub]),
                )
            cdf_side = ~survival_side
            if bool(cdf_side.any()):
                sub = torch.nonzero(cdf_side, as_tuple=False).squeeze(-1)
                regular_result[sub] = _stable_log_probability_difference(
                    torch.special.log_ndtr(upper[sub]),
                    torch.special.log_ndtr(lower[sub]),
                )
            result = result.index_copy(0, index, regular_result)

        if bool(censor.any()):
            index = torch.nonzero(censor, as_tuple=False).squeeze(-1)
            result = result.index_copy(
                0, index, torch.special.log_ndtr(-lower_z[index])
            )

    if result.dtype != torch.float64:
        raise RuntimeError("duration likelihood must be calculated in float64")
    _require_finite("duration log likelihood", result)
    return result


def fit_train_only_adamw(
    head: FrozenBK1LogNormalHead,
    hidden: torch.Tensor,
    target_dt: torch.Tensor,
    *,
    observation_mode: str,
    is_right_censored: torch.Tensor | None = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    """Fit exactly ``epochs`` on one source fold with deterministic AdamW.

    The cached hidden tensor and targets are detached.  There is no validation
    input, checkpoint selection, early stopping, or optimizer step outside the
    supplied source fold.
    """
    if not isinstance(head, FrozenBK1LogNormalHead):
        raise ValueError("head must be a FrozenBK1LogNormalHead")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    numeric = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
    }
    if any(not math.isfinite(float(value)) for value in numeric.values()):
        raise ValueError("optimizer values must be finite")
    if learning_rate <= 0.0 or weight_decay < 0.0 or grad_clip_norm <= 0.0:
        raise ValueError("invalid AdamW or gradient-clip value")
    hidden, target, censor = _validate_likelihood_inputs(
        head, hidden.detach(), target_dt.detach(), is_right_censored
    )
    hidden = hidden.detach()
    target = target.detach()
    censor = censor.detach()
    if next(head.parameters()).device != hidden.device:
        raise ValueError("head and source-fold cache must share a device")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    history: list[dict[str, float | int]] = []
    count = hidden.shape[0]
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(count, generator=generator)
        loss_sum = 0.0
        grad_norm_sum = 0.0
        batch_count = 0
        for start in range(0, count, batch_size):
            index = permutation[start : start + batch_size].to(hidden.device)
            optimizer.zero_grad(set_to_none=True)
            loss = -head.log_likelihood(
                hidden.index_select(0, index),
                target.index_select(0, index),
                observation_mode=observation_mode,
                is_right_censored=censor.index_select(0, index),
            ).mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("non-finite source-fold duration loss")
            loss.backward()
            for parameter in head.parameters():
                if parameter.grad is None or not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise ValueError("missing or non-finite duration-head gradient")
            norm = torch.nn.utils.clip_grad_norm_(
                tuple(head.parameters()), max_norm=grad_clip_norm
            )
            if not bool(torch.isfinite(norm)):
                raise ValueError("non-finite duration-head gradient norm")
            optimizer.step()
            for parameter in head.parameters():
                _require_finite("duration-head parameter", parameter)
            size = int(index.numel())
            loss_sum += float(loss.detach()) * size
            grad_norm_sum += float(norm.detach())
            batch_count += 1
        history.append(
            {
                "epoch": epoch,
                "source_fold_normalized_time_nll": loss_sum / count,
                "mean_preclip_gradient_norm": grad_norm_sum / batch_count,
            }
        )
    return history


def fit_source_fold_time_head(
    hidden: torch.Tensor,
    target_dt: torch.Tensor,
    *,
    observation_mode: str,
    is_right_censored: torch.Tensor | None = None,
    sigma_floor: float = SIGMA_FLOOR,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    grad_clip_norm: float = 1.0,
    seed: int = 42,
) -> tuple[FrozenBK1LogNormalHead, dict[str, Any]]:
    """Initialize and fit a head using only the supplied source-fold rows."""
    head, statistics = FrozenBK1LogNormalHead.from_source_fold_targets(
        target_dt, sigma_floor=sigma_floor
    )
    head.to(device=hidden.device)
    initial_state = {
        name: tensor.detach().clone() for name, tensor in head.state_dict().items()
    }
    history = fit_train_only_adamw(
        head,
        hidden,
        target_dt,
        observation_mode=observation_mode,
        is_right_censored=is_right_censored,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        grad_clip_norm=grad_clip_norm,
        seed=seed,
    )
    return head, {
        "statistics": statistics,
        "optimizer": "AdamW",
        "fixed_epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "grad_clip_norm": grad_clip_norm,
        "seed": seed,
        "initial_state": initial_state,
        "history": history,
    }


def _state_sha256(head: FrozenBK1LogNormalHead) -> str:
    digest = hashlib.sha256()
    digest.update(b"frozen_b_k1_time_head_v1\0")
    for name, tensor in sorted(head.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _censor_mask_from_threshold(
    target_dt: torch.Tensor, censor_threshold: float | None
) -> torch.Tensor:
    target = _require_tensor("target_dt", target_dt)
    if censor_threshold is None:
        return torch.zeros_like(target, dtype=torch.bool)
    threshold = float(censor_threshold)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("censor_threshold must be finite and positive")
    target64 = target.to(dtype=torch.float64)
    if bool((target64 > threshold).any()):
        raise ValueError("target_dt exceeds censor_threshold")
    return target64 == threshold


def fit_crossfit_time_head(
    hidden: torch.Tensor,
    target_dt: torch.Tensor,
    *,
    observation_mode: str,
    censor_threshold: float | None,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    batch_size: int = 256,
) -> tuple[FrozenBK1LogNormalHead, dict[str, Any]]:
    """Fit one directional source-fold head and return a JSON-safe audit.

    A two-fold runner calls this once per direction.  The threshold is only
    converted to a mask within the supplied source fold; no held-fold values
    participate in initialization, optimization, or epoch selection.
    """
    censor = _censor_mask_from_threshold(target_dt, censor_threshold)
    head, statistics = FrozenBK1LogNormalHead.from_source_fold_targets(target_dt)
    head.to(device=hidden.device)
    initial_sha256 = _state_sha256(head)
    with torch.no_grad():
        initial_nll = float(
            -head.log_likelihood(
                hidden,
                target_dt,
                observation_mode=observation_mode,
                is_right_censored=censor,
            ).mean()
        )
    history = fit_train_only_adamw(
        head,
        hidden,
        target_dt,
        observation_mode=observation_mode,
        is_right_censored=censor,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        grad_clip_norm=grad_clip,
        seed=seed,
    )
    final_sha256 = _state_sha256(head)
    with torch.no_grad():
        final_nll = float(
            -head.log_likelihood(
                hidden,
                target_dt,
                observation_mode=observation_mode,
                is_right_censored=censor,
            ).mean()
        )
    return head, {
        "scope": "source_fold_train_only",
        "statistics": statistics,
        "observation_mode": observation_mode,
        "censor_threshold": censor_threshold,
        "censored_count": int(censor.sum()),
        "parameter_count": sum(
            parameter.numel() for parameter in head.parameters()
        ),
        "target_count": int(target_dt.numel()),
        "parameter_dtype": "float32",
        "calculation_dtype": "float64",
        "optimizer": "AdamW",
        "optimizer_betas": [0.9, 0.999],
        "optimizer_eps": 1e-8,
        "optimizer_amsgrad": False,
        "optimizer_foreach": False,
        "fixed_epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "grad_clip": grad_clip,
        "seed": seed,
        "initial_state_sha256": initial_sha256,
        "final_state_sha256": final_sha256,
        "state_changed": initial_sha256 != final_sha256,
        "initial_nll": initial_nll,
        "final_nll": final_nll,
        "history": history,
    }


def zero_init_linear_residual_adapter_gradient(
    state: torch.Tensor,
    hidden_credit: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    denominator: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``d loss / d A`` at ``A=0`` for ``z' = z + A z``.

    ``hidden_credit[b, t]`` is the derivative of the *unreduced* token loss
    with respect to ``z'[b, t]``.  The returned matrix is therefore
    ``sum(valid * outer(hidden_credit, state)) / denominator``.  If omitted,
    the denominator is exactly the number of valid tokens.  A caller using a
    body-only mean must pass its body count while leaving non-body credits at
    zero; this keeps denominator semantics explicit.
    """
    state = _require_tensor("state", state)
    credit = _require_tensor("hidden_credit", hidden_credit)
    mask = _require_tensor("valid_mask", valid_mask)
    if state.ndim != 3 or state.shape[-1] == 0:
        raise ValueError("state must have shape [batch, sequence, hidden]")
    if credit.shape != state.shape:
        raise ValueError("hidden_credit must have the same shape as state")
    if mask.shape != state.shape[:2] or mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape [batch, sequence]")
    if credit.device != state.device or mask.device != state.device:
        raise ValueError("state, hidden_credit, and valid_mask must share a device")
    _require_finite("state", state)
    _require_finite("hidden_credit", credit)
    valid_count = int(mask.sum())
    if valid_count <= 0:
        raise ValueError("valid_mask must admit at least one token")
    if denominator is None:
        denominator_value = float(valid_count)
    elif isinstance(denominator, torch.Tensor):
        if denominator.numel() != 1:
            raise ValueError("denominator tensor must be scalar")
        denominator_value = float(denominator.detach().cpu())
    else:
        denominator_value = float(denominator)
    if not math.isfinite(denominator_value) or denominator_value <= 0.0:
        raise ValueError("denominator must be finite and positive")

    selected_state = state[mask].to(dtype=torch.float64)
    selected_credit = credit[mask].to(dtype=torch.float64)
    gradient = selected_credit.transpose(0, 1) @ selected_state
    gradient = gradient / denominator_value
    _require_finite("linear residual adapter gradient", gradient)
    return gradient


def linear_residual_adapter_gradient(
    state: torch.Tensor,
    hidden_credit: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    denominator: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility name for the exact zero-init adapter gradient."""
    return zero_init_linear_residual_adapter_gradient(
        state,
        hidden_credit,
        valid_mask,
        denominator=denominator,
    )


def cyclic_output_row_shifts(gradient: torch.Tensor) -> torch.Tensor:
    """Return all 63 non-identity output-row shifts of a 64-row adapter."""
    gradient = _require_tensor("gradient", gradient)
    if gradient.ndim != 2 or gradient.shape[0] != HIDDEN_DIM:
        raise ValueError(f"gradient must have shape [{HIDDEN_DIM}, input]")
    _require_finite("gradient", gradient)
    return torch.stack(
        [
            torch.roll(gradient, shifts=shift, dims=0)
            for shift in range(1, HIDDEN_DIM)
        ]
    )


def _gradient(name: str, value: torch.Tensor) -> torch.Tensor:
    value = _require_tensor(name, value)
    if value.ndim != 2 or value.shape[0] != HIDDEN_DIM or value.numel() == 0:
        raise ValueError(f"{name} must have shape [{HIDDEN_DIM}, input]")
    _require_finite(name, value)
    return value.detach().to(dtype=torch.float64, device="cpu")


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value))


def _unit(value: torch.Tensor, *, tolerance: float) -> torch.Tensor:
    norm = _norm(value)
    if norm <= tolerance:
        return torch.zeros_like(value)
    return value / norm


def construct_directional_update_arms(
    source_log_gradient: torch.Tensor,
    source_time_gradient: torch.Tensor,
    *,
    tolerance: float = DIRECTION_TOLERANCE,
) -> dict[str, Any]:
    """Build unit-budget shared and task-separated adapter updates.

    Equal-weight shared first sums the two raw objective gradients.  The
    norm-balanced shared control first gives each objective unit norm.  The
    ideal separated arm assigns ``1/sqrt(2)`` norm to each task adapter, so
    the concatenated two-adapter update has total norm one.
    """
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    quantity = _gradient("source_log_gradient", source_log_gradient)
    time = _gradient("source_time_gradient", source_time_gradient)
    quantity_unit = _unit(quantity, tolerance=tolerance)
    time_unit = _unit(time, tolerance=tolerance)
    equal_shared = _unit(quantity + time, tolerance=tolerance)
    balanced_shared = _unit(
        quantity_unit + time_unit, tolerance=tolerance
    )
    separated_quantity = quantity_unit / math.sqrt(2.0)
    separated_time = time_unit / math.sqrt(2.0)
    total_separated_norm = math.sqrt(
        _norm(separated_quantity) ** 2 + _norm(separated_time) ** 2
    )
    return {
        "equal_weight_shared": equal_shared,
        "norm_balanced_shared": balanced_shared,
        "task_separated_quantity": separated_quantity,
        "task_separated_time": separated_time,
        "norms": {
            "source_log": _norm(quantity),
            "source_time": _norm(time),
            "equal_weight_shared": _norm(equal_shared),
            "norm_balanced_shared": _norm(balanced_shared),
            "task_separated_quantity": _norm(separated_quantity),
            "task_separated_time": _norm(separated_time),
            "task_separated_total": total_separated_norm,
        },
        "checks": {
            "source_log_informative": _norm(quantity) > tolerance,
            "source_time_informative": _norm(time) > tolerance,
            "equal_weight_shared_informative": _norm(equal_shared) > tolerance,
            "norm_balanced_shared_informative": _norm(balanced_shared) > tolerance,
            "task_separated_unit_total_norm": math.isclose(
                total_separated_norm,
                1.0,
                rel_tol=0.0,
                abs_tol=max(tolerance, 1e-12),
            ),
        },
    }


def directional_dot_cosine(
    source_gradient: torch.Tensor,
    held_gradient: torch.Tensor,
    *,
    tolerance: float = DIRECTION_TOLERANCE,
) -> dict[str, float | bool]:
    """Measure held-fold transfer for a negative source-gradient step."""
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    source = _gradient("source_gradient", source_gradient)
    held = _gradient("held_gradient", held_gradient)
    if source.shape != held.shape:
        raise ValueError("source and held gradients must have equal shape")
    dot = float(torch.sum(source * held))
    source_norm = _norm(source)
    held_norm = _norm(held)
    informative = source_norm > tolerance and held_norm > tolerance
    cosine = dot / (source_norm * held_norm) if informative else 0.0
    direction_epsilon = tolerance * max(1.0, source_norm * held_norm)
    return {
        "dot": dot,
        "cosine": cosine,
        "source_norm": source_norm,
        "held_norm": held_norm,
        "direction_epsilon": direction_epsilon,
        "predicted_first_order_change_for_negative_source_step": -dot,
        "informative": informative,
        "positive_direction": dot > direction_epsilon,
    }


def compare_shared_and_task_separated_updates(
    source_log_gradient: torch.Tensor,
    source_time_gradient: torch.Tensor,
    held_metric_gradients: Mapping[str, torch.Tensor],
    *,
    tolerance: float = DIRECTION_TOLERANCE,
    control_quantile: float = CONTROL_QUANTILE,
) -> dict[str, Any]:
    """Compare shared updates with an ideal two-adapter upper bound.

    Quantity evaluation metrics always use the source log-loss adapter arm;
    normalized Time NLL uses the source normalized-time-loss arm.  The ideal
    arm passes a metric only if its direction is positive and it strictly
    beats both shared controls and the fixed 95th-percentile deterministic
    row-shift alignment control.
    """
    if set(held_metric_gradients) != set(REQUIRED_METRICS):
        raise ValueError(
            "held_metric_gradients must contain exactly log, raw, body, and time"
        )
    if (
        not math.isfinite(float(control_quantile))
        or not 0.0 < control_quantile < 1.0
    ):
        raise ValueError("control_quantile must be strictly between zero and one")
    arms = construct_directional_update_arms(
        source_log_gradient, source_time_gradient, tolerance=tolerance
    )
    shifted_quantity = cyclic_output_row_shifts(
        arms["task_separated_quantity"]
    )
    shifted_time = cyclic_output_row_shifts(arms["task_separated_time"])
    metrics: dict[str, Any] = {}
    for metric_name in REQUIRED_METRICS:
        held = _gradient(
            f"held_metric_gradients[{metric_name!r}]",
            held_metric_gradients[metric_name],
        )
        separated = (
            arms["task_separated_time"]
            if metric_name == NORMALIZED_TIME_NLL
            else arms["task_separated_quantity"]
        )
        shifted = (
            shifted_time
            if metric_name == NORMALIZED_TIME_NLL
            else shifted_quantity
        )
        equal_result = directional_dot_cosine(
            arms["equal_weight_shared"], held, tolerance=tolerance
        )
        balanced_result = directional_dot_cosine(
            arms["norm_balanced_shared"], held, tolerance=tolerance
        )
        separated_result = directional_dot_cosine(
            separated, held, tolerance=tolerance
        )
        control_dots = torch.einsum("kij,ij->k", shifted, held)
        control_q = float(
            torch.quantile(
                control_dots,
                q=control_quantile,
                interpolation="higher",
            )
        )
        comparison_scale = max(
            1.0,
            abs(float(separated_result["dot"])),
            abs(float(equal_result["dot"])),
            abs(float(balanced_result["dot"])),
            abs(control_q),
            float(separated_result["source_norm"])
            * float(separated_result["held_norm"]),
        )
        comparison_epsilon = tolerance * comparison_scale
        gate_threshold = max(
            comparison_epsilon,
            float(equal_result["dot"]) + comparison_epsilon,
            float(balanced_result["dot"]) + comparison_epsilon,
            control_q + comparison_epsilon,
        )
        if metric_name == BODY_MAE:
            checks = {
                "informative": bool(separated_result["informative"]),
                "non_conflicting_direction": (
                    float(separated_result["dot"]) >= -comparison_epsilon
                ),
            }
        else:
            checks = {
                "informative": bool(separated_result["informative"]),
                "positive_direction": (
                    float(separated_result["dot"]) > comparison_epsilon
                ),
                "beats_equal_weight_shared": (
                    float(separated_result["dot"])
                    > float(equal_result["dot"]) + comparison_epsilon
                ),
                "beats_norm_balanced_shared": (
                    float(separated_result["dot"])
                    > float(balanced_result["dot"]) + comparison_epsilon
                ),
                "beats_shifted_control_q95": (
                    float(separated_result["dot"])
                    > control_q + comparison_epsilon
                ),
                "above_explicit_gate_threshold": (
                    float(separated_result["dot"]) > gate_threshold
                ),
            }
        checks["passed"] = all(checks.values())
        metrics[metric_name] = {
            "source_arm": (
                "normalized_time_nll"
                if metric_name == NORMALIZED_TIME_NLL
                else "log_mse"
            ),
            "equal_weight_shared": equal_result,
            "norm_balanced_shared": balanced_result,
            "task_separated": separated_result,
            "comparison_epsilon": comparison_epsilon,
            "gate_threshold": gate_threshold,
            "cyclic_output_row_shift_control": {
                "count": int(control_dots.numel()),
                "dot_q95": control_q,
                "dot_max": float(control_dots.max()),
                "dot_min": float(control_dots.min()),
            },
            "checks": checks,
        }

    structural_checks = dict(arms["checks"])
    structural_checks["exactly_63_shift_controls"] = (
        shifted_quantity.shape[0] == CONTROL_COUNT
        and shifted_time.shape[0] == CONTROL_COUNT
    )
    structural_checks["all_metrics_pass"] = all(
        row["checks"]["passed"] for row in metrics.values()
    )
    structural_checks["passed"] = all(structural_checks.values())
    return {
        "comparison_contract": {
            "aligned_equal_weight_shared": "equal normalized-K1 objective coefficients before normalization",
            "aligned_norm_balanced_shared": "unit source arm per task before shared sum",
            "task_separated_total_norm": 1.0,
            "task_separated_per_task_norm": 1.0 / math.sqrt(2.0),
            "control": "all 63 non-identity cyclic output-row shifts; deterministic alignment control, not a statistical null",
            "control_quantile": control_quantile,
            "direction_tolerance": tolerance,
        },
        "arm_norms": arms["norms"],
        "metrics": metrics,
        "checks": structural_checks,
    }


def compare_shared_and_separated(
    source_gradients: Mapping[str, torch.Tensor],
    held_gradients: Mapping[str, torch.Tensor],
    *,
    tolerance: float = DIRECTION_TOLERANCE,
) -> dict[str, Any]:
    """Mapping-based runner API for one source-to-held fold direction."""
    source_keys = set(source_gradients)
    if not {LOG_MSE, NORMALIZED_TIME_NLL} <= source_keys or not source_keys <= set(
        REQUIRED_METRICS
    ):
        raise ValueError(
            "source_gradients must contain log_mse and normalized_time_nll, "
            "with only declared audit metric keys"
        )
    return compare_shared_and_task_separated_updates(
        source_gradients[LOG_MSE],
        source_gradients[NORMALIZED_TIME_NLL],
        held_gradients,
        tolerance=tolerance,
    )


__all__ = [
    "BODY_MAE",
    "CONTINUOUS_LOGNORMAL",
    "CONTROL_COUNT",
    "CONTROL_QUANTILE",
    "DIRECTION_TOLERANCE",
    "FrozenBK1LogNormalHead",
    "HIDDEN_DIM",
    "LOG_MSE",
    "NORMALIZED_TIME_NLL",
    "POSITIVE_INTEGER_LOGNORMAL",
    "QUANTITY_METRICS",
    "RAW_SQUARED_ERROR",
    "REQUIRED_METRICS",
    "SIGMA_FLOOR",
    "TIME_HEAD_PARAMETER_COUNT",
    "compare_shared_and_task_separated_updates",
    "compare_shared_and_separated",
    "construct_directional_update_arms",
    "cyclic_output_row_shifts",
    "directional_dot_cosine",
    "fit_source_fold_time_head",
    "fit_crossfit_time_head",
    "fit_train_only_adamw",
    "normalized_lognormal_log_likelihood",
    "source_fold_time_statistics",
    "linear_residual_adapter_gradient",
    "zero_init_linear_residual_adapter_gradient",
]
