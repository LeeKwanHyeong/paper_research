"""Shared decoder and baseline encoders for mark-free count-aware TPPs.

The count-aware experiments predict event time and raw quantity without a
categorical quantity mark. Every backbone in this module exposes the same
``encode(dts, history_quantities, mask)`` contract and reuses the same time and
quantity heads, so comparisons isolate the history encoder.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TPPs.TransformerHawkesTPP import THPEncoderLayer
from models.TPPs.config import THPConfig
from models.Titan.backbone import MemoryEncoder
from models.Titan.common.memory import (
    GatedSoftMemory,
    HardLocalMemoryMatcher,
    SimilarityWeightedLocalMemoryMatcher,
    SurpriseGatedMemory,
)
from models.Titan.common.titans_mac import TitansMACEncoder, TitansMemoryState
from models.Titan.common.tpp_gated_memory import (
    TPPGatedMemoryState,
    TPPSpecificGatedMemory,
)


LOG_MSE_VARIANT = "count_only_log_regression"
LOGNORMAL_VARIANT = "count_only_lognormal_k1"
TAIL_SHARED_VARIANT = "count_only_log_mse_tail_shared"
TAIL_HEAD_ONLY_VARIANT = "count_only_log_mse_tail_head_only"
TAIL_VARIANTS = (TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT)
TIME_HEAD_MODE_LEGACY_CLAMPED = "legacy_clamped_rmtpp"
TIME_HEAD_MODE_SCALED_EXACT = "scaled_exact_rmtpp"
TIME_HEAD_MODE_SCALED_EXACT_STABLE = "scaled_exact_stable_rmtpp"
TIME_HEAD_MODE_LOGNORMAL_DURATION = "lognormal_duration"
TIME_HEAD_EXACT_MODES = (
    TIME_HEAD_MODE_SCALED_EXACT,
    TIME_HEAD_MODE_SCALED_EXACT_STABLE,
)
TIME_HEAD_MODES = (
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    *TIME_HEAD_EXACT_MODES,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
)
TITAN_MEMORY_MODE_NONE = "none"
TITAN_MEMORY_MODE_PERSISTENT_ONLY = "persistent_only"
TITAN_MEMORY_MODE_STATIC_HARD = "static_hard_lmm"
TITAN_MEMORY_MODE_STATIC_WEIGHTED = "static_weighted_lmm"
TITAN_MEMORY_MODE_STATIC_SOFT_GATED = "static_soft_gated"
TITAN_MEMORY_MODE_SURPRISE_GATED = "surprise_gated"
TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED = "persistent_surprise_gated"
TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE = "dual_hard_surprise"
TITAN_MEMORY_MODE_TITANS_MAC = "titans_mac"
TITAN_MEMORY_MODE_TPP_GATED = "tpp_gated_memory"
TITAN_MEMORY_MODES = (
    TITAN_MEMORY_MODE_NONE,
    TITAN_MEMORY_MODE_PERSISTENT_ONLY,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_MEMORY_MODE_STATIC_WEIGHTED,
    TITAN_MEMORY_MODE_STATIC_SOFT_GATED,
    TITAN_MEMORY_MODE_SURPRISE_GATED,
    TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
    TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
    TITAN_MEMORY_MODE_TITANS_MAC,
    TITAN_MEMORY_MODE_TPP_GATED,
)
TITAN_QUANTITY_GRADIENT_SHARED = "shared"
TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY = "adapter_only"
TITAN_QUANTITY_GRADIENT_MODES = (
    TITAN_QUANTITY_GRADIENT_SHARED,
    TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY,
)


def inverse_softplus(value: float) -> float:
    """Return the pre-activation that maps to ``value`` through softplus."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse_softplus requires a finite positive value")
    return value + math.log(-math.expm1(-value))


def inverse_sigmoid(value: float) -> float:
    """Return the logit of a finite probability strictly between zero and one."""
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("inverse_sigmoid requires a finite value in (0, 1)")
    return math.log(value) - math.log1p(-value)


class SharedTimeCountModel(nn.Module):
    """Common time-density and continuous-count heads for every backbone."""

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        *,
        train_log_std: float = 1.0,
        quantity_variant: str = LOG_MSE_VARIANT,
        quantity_sigma_floor: float = 1e-3,
        lambda_location_huber: float = 1.0,
        location_huber_delta: float = 0.25,
        lambda_tail: float = 0.0,
        tail_threshold: float = 46.0,
        tail_normalization_scale: float = 46.0,
        tail_clip_cap: float = 187.0,
        tail_huber_delta: float = 1.0,
        time_head_mode: str = TIME_HEAD_MODE_LEGACY_CLAMPED,
        time_scale: float = 3.0,
        time_w_max: float = 10.0 / 3.0,
        time_intercept_limit: float = 30.0,
        time_initial_intercept: float | None = None,
        time_wd_safety_limit: float = 40.0,
        time_initial_location: float | None = None,
        time_initial_scale: float | None = None,
        time_sigma_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        if not math.isfinite(train_log_mean) or train_log_mean <= 0.0:
            raise ValueError("train_log_mean must be finite and positive")
        if not math.isfinite(train_log_std) or train_log_std <= 0.0:
            raise ValueError("train_log_std must be finite and positive")
        if quantity_variant not in {
            LOG_MSE_VARIANT,
            LOGNORMAL_VARIANT,
            *TAIL_VARIANTS,
        }:
            raise ValueError(f"Unsupported quantity_variant: {quantity_variant}")
        if quantity_sigma_floor <= 0.0:
            raise ValueError("quantity_sigma_floor must be positive")
        if lambda_location_huber < 0.0:
            raise ValueError("lambda_location_huber must be nonnegative")
        if location_huber_delta <= 0.0:
            raise ValueError("location_huber_delta must be positive")
        if lambda_tail < 0.0:
            raise ValueError("lambda_tail must be nonnegative")
        if not 0.0 < tail_threshold < tail_clip_cap:
            raise ValueError("tail_threshold must be positive and below tail_clip_cap")
        if tail_normalization_scale <= 0.0:
            raise ValueError("tail_normalization_scale must be positive")
        if tail_huber_delta <= 0.0:
            raise ValueError("tail_huber_delta must be positive")
        if time_head_mode not in TIME_HEAD_MODES:
            raise ValueError(f"Unsupported time_head_mode: {time_head_mode}")
        if not math.isfinite(time_scale) or time_scale <= 0.0:
            raise ValueError("time_scale must be finite and positive")
        if not math.isfinite(time_w_max) or time_w_max <= 0.0:
            raise ValueError("time_w_max must be finite and positive")
        if not math.isfinite(time_intercept_limit) or time_intercept_limit <= 0.0:
            raise ValueError("time_intercept_limit must be finite and positive")
        if time_initial_intercept is not None and not math.isfinite(
            time_initial_intercept
        ):
            raise ValueError("time_initial_intercept must be finite when provided")
        if not math.isfinite(time_wd_safety_limit) or time_wd_safety_limit <= 0.0:
            raise ValueError("time_wd_safety_limit must be finite and positive")
        if time_initial_location is not None and not math.isfinite(
            time_initial_location
        ):
            raise ValueError("time_initial_location must be finite when provided")
        if time_initial_scale is not None and (
            not math.isfinite(time_initial_scale) or time_initial_scale <= 0.0
        ):
            raise ValueError("time_initial_scale must be finite and positive")
        if not math.isfinite(time_sigma_floor) or time_sigma_floor <= 0.0:
            raise ValueError("time_sigma_floor must be finite and positive")

        self.hidden_dim = int(hidden_dim)
        self.quantity_variant = quantity_variant
        self.quantity_sigma_floor = float(quantity_sigma_floor)
        self.lambda_location_huber = float(lambda_location_huber)
        self.location_huber_delta = float(location_huber_delta)
        self.lambda_tail = float(lambda_tail)
        self.tail_threshold = float(tail_threshold)
        self.tail_normalization_scale = float(tail_normalization_scale)
        self.tail_clip_cap = float(tail_clip_cap)
        self.tail_huber_delta = float(tail_huber_delta)
        self.time_head_mode = time_head_mode
        self.time_scale = float(time_scale)
        self.time_w_max = float(time_w_max)
        self.time_intercept_limit = float(time_intercept_limit)
        self.time_wd_safety_limit = float(time_wd_safety_limit)
        self.time_sigma_floor = float(time_sigma_floor)

        self.v_t = nn.Linear(self.hidden_dim, 1, bias=False)
        if self.time_head_mode in TIME_HEAD_EXACT_MODES:
            initial_w = min(0.05 * self.time_scale, 0.25 * self.time_w_max)
            if self.time_head_mode == TIME_HEAD_MODE_SCALED_EXACT_STABLE:
                bounded_initial_intercept = (
                    0.0
                    if time_initial_intercept is None
                    else float(time_initial_intercept)
                )
                if abs(bounded_initial_intercept) >= self.time_intercept_limit:
                    raise ValueError(
                        "time_initial_intercept must lie inside the stable "
                        "intercept range"
                    )
                raw_initial_intercept = self.time_intercept_limit * math.atanh(
                    bounded_initial_intercept / self.time_intercept_limit
                )
            else:
                bounded_initial_intercept = math.log(self.time_scale)
                raw_initial_intercept = bounded_initial_intercept
            self.time_initial_intercept = bounded_initial_intercept
            self.b_t = nn.Parameter(torch.full((1,), raw_initial_intercept))
            self.w_raw = nn.Parameter(
                torch.full((1,), inverse_sigmoid(initial_w / self.time_w_max))
            )
            self.time_initial_location = 0.0
            self.time_initial_scale = 0.0
        elif self.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
            initial_location = (
                0.0
                if time_initial_location is None
                else float(time_initial_location)
            )
            initial_scale = (
                1.0 if time_initial_scale is None else float(time_initial_scale)
            )
            if initial_scale <= self.time_sigma_floor:
                raise ValueError(
                    "time_initial_scale must exceed time_sigma_floor"
                )
            self.time_initial_intercept = 0.0
            self.time_initial_location = initial_location
            self.time_initial_scale = initial_scale
            self.b_t = nn.Parameter(torch.full((1,), initial_location))
            self.w_raw = nn.Parameter(
                torch.full(
                    (1,),
                    inverse_softplus(initial_scale - self.time_sigma_floor),
                )
            )
        else:
            self.time_initial_intercept = 0.0
            self.time_initial_location = 0.0
            self.time_initial_scale = 0.0
            self.b_t = nn.Parameter(torch.zeros(1))
            self.w_raw = nn.Parameter(torch.full((1,), -3.0))
        self.quantity_head = nn.Linear(self.hidden_dim, 1)
        if self.quantity_variant == LOGNORMAL_VARIANT:
            # Preserve the random stream used by the original deterministic
            # log-MSE model when the optional scale head is present.
            rng_state = torch.random.get_rng_state()
            self.quantity_scale_head = nn.Linear(self.hidden_dim, 1)
            torch.random.set_rng_state(rng_state)

        nn.init.normal_(self.v_t.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.quantity_head.weight)
        nn.init.constant_(
            self.quantity_head.bias,
            float(np.log(np.expm1(train_log_mean))),
        )
        if self.quantity_variant == LOGNORMAL_VARIANT:
            initial_scale = max(train_log_std - self.quantity_sigma_floor, 1e-4)
            nn.init.zeros_(self.quantity_scale_head.weight)
            nn.init.constant_(
                self.quantity_scale_head.bias,
                inverse_softplus(initial_scale),
            )

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode an observed event history into one state per event."""
        raise NotImplementedError

    def encode_task_states(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
        *,
        memory_write_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return time and quantity states, shared by default."""
        del memory_write_mask
        encoded = self.encode(dts, history_quantities, mask)
        return encoded, encoded

    @staticmethod
    def continuous_features(
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build the shared log-time and log-quantity event features."""
        features = torch.stack(
            [
                torch.log1p(dts.float().clamp_min(0.0)),
                torch.log1p(history_quantities.float().clamp_min(0.0)),
            ],
            dim=-1,
        )
        return features * mask.unsqueeze(-1).to(dtype=features.dtype)

    def log_f_dt(self, hidden: torch.Tensor, dt_next: torch.Tensor) -> torch.Tensor:
        """Evaluate the shared RMTPP-style next-event time log density."""
        if self.time_head_mode == TIME_HEAD_MODE_LEGACY_CLAMPED:
            w = F.softplus(self.w_raw) + 1e-3
            intercept = torch.clamp(
                self.v_t(hidden).squeeze(-1) + self.b_t,
                max=self.time_intercept_limit,
            )
            exp_intercept = torch.exp(intercept)
            wd = torch.clamp(w * dt_next, max=10.0)
            return intercept + wd - (exp_intercept / w) * torch.expm1(wd)

        if self.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
            output_dtype = hidden.dtype
            location, scale, log_dt = self._lognormal_time_terms(hidden, dt_next)
            standardized = (
                log_dt - math.log(self.time_scale) - location
            ) / scale
            log_density = (
                -0.5 * torch.square(standardized)
                - torch.log(scale)
                - log_dt
                - 0.5 * math.log(2.0 * math.pi)
            )
            return log_density.to(dtype=output_dtype)

        output_dtype = hidden.dtype
        intercept, w, scaled_dt = self._scaled_exact_time_terms(hidden, dt_next)
        wd = w * scaled_dt
        cumulative_hazard = (torch.exp(intercept) / w) * torch.expm1(wd)
        log_density = (
            intercept + wd - cumulative_hazard - math.log(self.time_scale)
        )
        return log_density.to(dtype=output_dtype)

    def log_survival_dt(
        self,
        hidden: torch.Tensor,
        dt_next: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate log survival under the configured time-head contract."""
        if self.time_head_mode == TIME_HEAD_MODE_LEGACY_CLAMPED:
            w = F.softplus(self.w_raw) + 1e-3
            intercept = torch.clamp(
                self.v_t(hidden).squeeze(-1) + self.b_t,
                max=self.time_intercept_limit,
            )
            wd = torch.clamp(w * dt_next, max=10.0)
            return -(torch.exp(intercept) / w) * torch.expm1(wd)

        if self.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
            output_dtype = hidden.dtype
            location, scale, log_dt = self._lognormal_time_terms(hidden, dt_next)
            standardized = (
                log_dt - math.log(self.time_scale) - location
            ) / scale
            return torch.special.log_ndtr(-standardized).to(dtype=output_dtype)

        output_dtype = hidden.dtype
        intercept, w, scaled_dt = self._scaled_exact_time_terms(hidden, dt_next)
        log_survival = -(torch.exp(intercept) / w) * torch.expm1(w * scaled_dt)
        return log_survival.to(dtype=output_dtype)

    def predict_time_median(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the conditional median inter-event time in the original unit."""
        output_dtype = hidden.dtype
        if self.time_head_mode == TIME_HEAD_MODE_LEGACY_CLAMPED:
            w = (F.softplus(self.w_raw) + 1e-3).to(dtype=torch.float64)
            intercept = torch.clamp(
                self.v_t(hidden).squeeze(-1) + self.b_t,
                min=-300.0,
                max=self.time_intercept_limit,
            ).to(dtype=torch.float64)
            median = torch.log1p(w * math.log(2.0) * torch.exp(-intercept)) / w
            return median.to(dtype=output_dtype)

        if self.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
            location = self.time_location(hidden).to(dtype=torch.float64)
            median = self.time_scale * torch.exp(location)
            return median.to(dtype=output_dtype)

        intercept, w, _ = self._scaled_exact_time_terms(
            hidden,
            torch.zeros(hidden.shape[:-1], device=hidden.device, dtype=hidden.dtype),
        )
        scaled_median = (
            torch.log1p(w * math.log(2.0) * torch.exp(-intercept)) / w
        )
        return (scaled_median * self.time_scale).to(dtype=output_dtype)

    def positive_time_slope(self) -> torch.Tensor:
        """Return the positive slope in the active time coordinate."""
        if self.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
            raise RuntimeError("Log-normal duration head has no RMTPP slope")
        if self.time_head_mode == TIME_HEAD_MODE_LEGACY_CLAMPED:
            return F.softplus(self.w_raw) + 1e-3
        return (self.time_w_max * torch.sigmoid(self.w_raw)).clamp_min(1e-6)

    def positive_time_sigma(self) -> torch.Tensor:
        """Return the positive log-duration scale for the log-normal head."""
        if self.time_head_mode != TIME_HEAD_MODE_LOGNORMAL_DURATION:
            raise RuntimeError("Only the log-normal duration head has sigma")
        return self.time_sigma_floor + F.softplus(self.w_raw)

    def time_location(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the conditional log-duration location."""
        if self.time_head_mode != TIME_HEAD_MODE_LOGNORMAL_DURATION:
            raise RuntimeError("Only the log-normal duration head has location")
        return self.v_t(hidden).squeeze(-1) + self.b_t

    def time_head_telemetry(self) -> dict[str, float]:
        """Expose the active scalar time-shape parameter without mislabeling it."""
        if self.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
            return {
                "train_time_sigma": float(
                    self.positive_time_sigma().detach().cpu().item()
                )
            }
        return {
            "train_time_slope": float(
                self.positive_time_slope().detach().cpu().item()
            )
        }

    def time_head_contract(self) -> dict[str, float | str | bool]:
        """Return serializable time-head metadata for experiment manifests."""
        if self.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
            return {
                "mode": self.time_head_mode,
                "density_family": "lognormal_on_scaled_duration",
                "time_scale": self.time_scale,
                "time_initial_location": self.time_initial_location,
                "time_initial_scale": self.time_initial_scale,
                "time_sigma_floor": self.time_sigma_floor,
                "time_location_transform": "identity",
                "slope_parameterized": False,
                "jacobian_correction": True,
                "wd_clamp": 0.0,
            }
        contract: dict[str, float | str | bool] = {
            "mode": self.time_head_mode,
            "time_scale": self.time_scale,
            "time_w_max": self.time_w_max,
            "time_intercept_limit": self.time_intercept_limit,
            "jacobian_correction": (
                self.time_head_mode in TIME_HEAD_EXACT_MODES
            ),
            "wd_clamp": (
                10.0
                if self.time_head_mode == TIME_HEAD_MODE_LEGACY_CLAMPED
                else 0.0
            ),
        }
        if self.time_head_mode == TIME_HEAD_MODE_SCALED_EXACT_STABLE:
            contract.update(
                {
                    "time_initial_intercept": self.time_initial_intercept,
                    "time_intercept_transform": "scaled_tanh",
                    "time_wd_safety_limit": self.time_wd_safety_limit,
                }
            )
        return contract

    def time_head_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return the parameters optimized by the shared event-time head."""
        return (
            ("v_t.weight", self.v_t.weight),
            ("b_t", self.b_t),
            ("w_raw", self.w_raw),
        )

    def bounded_time_intercept(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map the raw intensity intercept into its configured finite range."""
        raw_intercept = self.v_t(hidden).squeeze(-1) + self.b_t
        if self.time_head_mode == TIME_HEAD_MODE_SCALED_EXACT_STABLE:
            return self.time_intercept_limit * torch.tanh(
                raw_intercept / self.time_intercept_limit
            )
        return torch.clamp(
            raw_intercept,
            min=-self.time_intercept_limit,
            max=self.time_intercept_limit,
        )

    def _scaled_exact_time_terms(
        self,
        hidden: torch.Tensor,
        dt_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        intercept = self.bounded_time_intercept(hidden).to(dtype=torch.float64)
        w = self.positive_time_slope().to(dtype=torch.float64)
        scaled_dt = dt_next.to(dtype=torch.float64).clamp_min(0.0) / self.time_scale
        return intercept, w, scaled_dt

    def _lognormal_time_terms(
        self,
        hidden: torch.Tensor,
        dt_next: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool((dt_next <= 0.0).any()):
            raise ValueError("Log-normal duration targets must be strictly positive")
        location = self.time_location(hidden).to(dtype=torch.float64)
        scale = self.positive_time_sigma().to(dtype=torch.float64)
        log_dt = torch.log(dt_next.to(dtype=torch.float64))
        return location, scale, log_dt

    def predict_quantity(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return log1p-space location and reconstructed raw quantity."""
        log_quantity = F.softplus(self.quantity_head(hidden).squeeze(-1))
        return log_quantity, torch.expm1(log_quantity)

    def quantity_outputs(
        self,
        hidden: torch.Tensor,
        true_quantity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Calculate variant-specific quantity losses and shared predictions."""
        location, point_prediction = self.predict_quantity(hidden)
        target = torch.log1p(true_quantity.clamp_min(0.0))
        log_mse = F.mse_loss(location, target, reduction="none")
        zeros = torch.zeros_like(log_mse)
        if self.quantity_variant == LOG_MSE_VARIANT:
            return {
                "train_loss": log_mse,
                "log_mse": log_mse,
                "distribution_nll": zeros,
                "location_huber": zeros,
                "scale": zeros,
                "location": location,
                "point_prediction": point_prediction,
                "tail_aux_loss": zeros,
                "tail_indicator": zeros,
            }

        if self.quantity_variant in TAIL_VARIANTS:
            tail_hidden = (
                hidden
                if self.quantity_variant == TAIL_SHARED_VARIANT
                else hidden.detach()
            )
            _, tail_prediction = self.predict_quantity(tail_hidden)
            normalized_prediction = tail_prediction.clamp(
                min=0.0,
                max=self.tail_clip_cap,
            ) / self.tail_normalization_scale
            normalized_target = true_quantity.clamp(
                min=0.0,
                max=self.tail_clip_cap,
            ) / self.tail_normalization_scale
            tail_indicator = (true_quantity > self.tail_threshold).to(log_mse.dtype)
            tail_aux_loss = tail_indicator * F.huber_loss(
                normalized_prediction,
                normalized_target,
                reduction="none",
                delta=self.tail_huber_delta,
            )
            return {
                "train_loss": log_mse + self.lambda_tail * tail_aux_loss,
                "log_mse": log_mse,
                "distribution_nll": zeros,
                "location_huber": zeros,
                "scale": zeros,
                "location": location,
                "point_prediction": point_prediction,
                "tail_aux_loss": tail_aux_loss,
                "tail_indicator": tail_indicator,
            }

        scale = self.quantity_sigma_floor + F.softplus(
            self.quantity_scale_head(hidden).squeeze(-1)
        )
        distribution_nll = 0.5 * torch.square((target - location) / scale)
        distribution_nll = (
            distribution_nll
            + torch.log(scale)
            + 0.5 * math.log(2.0 * math.pi)
        )
        location_huber = F.huber_loss(
            location,
            target,
            reduction="none",
            delta=self.location_huber_delta,
        )
        train_loss = distribution_nll + self.lambda_location_huber * location_huber
        return {
            "train_loss": train_loss,
            "log_mse": log_mse,
            "distribution_nll": distribution_nll,
            "location_huber": location_huber,
            "scale": scale,
            "location": location,
            "point_prediction": point_prediction,
            "tail_aux_loss": zeros,
            "tail_indicator": zeros,
        }


class CountAwareRMTPP(SharedTimeCountModel):
    """GRU history encoder under the shared count-aware output heads."""

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        **quantity_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.input_dropout = nn.Dropout(0.1)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_dropout(
            self.input_projection(
                self.continuous_features(dts, history_quantities, mask)
            )
        )
        encoded, _ = self.encoder(x)
        return encoded * mask.unsqueeze(-1).to(dtype=encoded.dtype)


class CountAwareTHP(SharedTimeCountModel):
    """THP encoder under the shared count-aware output heads."""

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        **quantity_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        self.input_projection = nn.Linear(2, hidden_dim)
        self.encoder_config = THPConfig(
            d_model=hidden_dim,
            d_inner=hidden_dim * 4,
            n_layers=2,
            n_heads=4,
            dropout=0.1,
            normalize_before=False,
            add_temporal_encoding_each_layer=False,
            use_rnn=False,
            d_rnn=hidden_dim,
        )
        self.layers = nn.ModuleList(
            [
                THPEncoderLayer(self.encoder_config)
                for _ in range(self.encoder_config.n_layers)
            ]
        )

    @staticmethod
    def blocked_attention_mask(mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = mask.shape
        future = torch.triu(
            torch.ones(seq_len, seq_len, device=mask.device, dtype=torch.bool),
            diagonal=1,
        ).unsqueeze(0)
        key_padding = (~mask).unsqueeze(1).expand(batch_size, seq_len, seq_len)
        blocked = future | key_padding
        positions = torch.arange(seq_len, device=mask.device)
        blocked[:, positions, positions] = False
        return blocked

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.input_projection(
            self.continuous_features(dts, history_quantities, mask)
        )
        non_pad = mask.unsqueeze(-1).to(dtype=x.dtype)
        blocked = self.blocked_attention_mask(mask)
        for layer in self.layers:
            x = layer(x, non_pad_mask=non_pad, blocked_mask=blocked)
        return x * non_pad


class CountAwareTitanTPP(SharedTimeCountModel):
    """Titan memory encoder under the shared count-aware output heads."""

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        max_seq_len: int,
        memory_mode: str = TITAN_MEMORY_MODE_STATIC_HARD,
        quantity_memory_gradient_mode: str = TITAN_QUANTITY_GRADIENT_SHARED,
        **quantity_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        if memory_mode not in TITAN_MEMORY_MODES:
            raise ValueError(
                f"Unsupported count-aware Titan memory_mode: {memory_mode}"
            )
        if quantity_memory_gradient_mode not in TITAN_QUANTITY_GRADIENT_MODES:
            raise ValueError(
                "Unsupported Titan quantity_memory_gradient_mode: "
                f"{quantity_memory_gradient_mode}"
            )
        if (
            memory_mode != TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE
            and quantity_memory_gradient_mode != TITAN_QUANTITY_GRADIENT_SHARED
        ):
            raise ValueError(
                "adapter_only quantity routing is valid only for dual memory"
            )
        self.memory_mode = memory_mode
        self.quantity_memory_gradient_mode = quantity_memory_gradient_mode
        uses_titans_mac = memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
        uses_tpp_gated_memory = memory_mode == TITAN_MEMORY_MODE_TPP_GATED
        uses_persistent_memory = memory_mode in {
            TITAN_MEMORY_MODE_PERSISTENT_ONLY,
            TITAN_MEMORY_MODE_STATIC_HARD,
            TITAN_MEMORY_MODE_STATIC_WEIGHTED,
            TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
            TITAN_MEMORY_MODE_TPP_GATED,
        }
        uses_hard_memory = memory_mode in {
            TITAN_MEMORY_MODE_STATIC_HARD,
            TITAN_MEMORY_MODE_STATIC_WEIGHTED,
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
        }
        uses_surprise_memory = memory_mode in {
            TITAN_MEMORY_MODE_SURPRISE_GATED,
            TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
        }
        self.encoder = (
            None
            if uses_titans_mac
            else MemoryEncoder(
                input_dim=2,
                d_model=hidden_dim,
                n_layers=2,
                n_heads=4,
                d_ff=hidden_dim * 2,
                contextual_mem_size=0,
                persistent_mem_size=16 if uses_persistent_memory else 0,
                dropout=0.1,
                use_context_update=False,
                use_pos_emb=True,
                max_len=max_seq_len,
                use_causal=True,
            )
        )
        self.titans_mac_encoder = (
            TitansMACEncoder(
                input_dim=2,
                d_model=hidden_dim,
                n_layers=2,
                n_heads=4,
                d_ff=hidden_dim * 2,
                persistent_memory_size=16,
                segment_size=16,
                max_len=max_seq_len,
                dropout=0.1,
            )
            if uses_titans_mac
            else None
        )
        matcher = (
            SimilarityWeightedLocalMemoryMatcher
            if memory_mode == TITAN_MEMORY_MODE_STATIC_WEIGHTED
            else HardLocalMemoryMatcher
        )
        self.lmm = (
            matcher(d_model=hidden_dim, mem_size=64, topk=4)
            if uses_hard_memory
            else None
        )
        self.soft_memory = (
            GatedSoftMemory(
                d_model=hidden_dim,
                mem_size=64,
                temperature=1.0,
                dropout=0.1,
            )
            if memory_mode == TITAN_MEMORY_MODE_STATIC_SOFT_GATED
            else None
        )
        self.surprise_memory = (
            SurpriseGatedMemory(
                d_model=hidden_dim,
                memory_rank=min(16, hidden_dim),
                chunk_size=32,
                initial_update_rate=0.01,
                initial_retention=0.99,
                initial_momentum=0.5,
                memory_clip=5.0,
                dropout=0.1,
            )
            if uses_surprise_memory
            else None
        )
        self.tpp_gated_memory = (
            TPPSpecificGatedMemory(
                d_model=hidden_dim,
                memory_size=64,
                topk=4,
                temperature=1.0,
                dropout=0.1,
                initial_null_logit=0.0,
                initial_confidence=0.5,
            )
            if uses_tpp_gated_memory
            else None
        )

    def _encode_base(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
        *,
        memory_write_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.continuous_features(dts, history_quantities, mask)
        if self.titans_mac_encoder is not None:
            encoded = self.titans_mac_encoder(
                x,
                mask=mask,
                write_mask=memory_write_mask,
            )
        else:
            if self.encoder is None:
                raise RuntimeError("Titan encoder is not initialized")
            encoded = self.encoder(x, mask=mask, update_context_memory=False)
            if self.tpp_gated_memory is not None:
                encoded = self.tpp_gated_memory(
                    encoded,
                    mask=mask,
                    write_mask=memory_write_mask,
                )
        return encoded * mask.unsqueeze(-1).to(dtype=encoded.dtype)

    def encode_with_memory_state(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
        *,
        state: TitansMemoryState | TPPGatedMemoryState | None = None,
        series_ids: torch.Tensor | None = None,
        memory_write_mask: torch.Tensor | None = None,
        segment_size: int | None = None,
        write_chunk_size: int | None = None,
    ) -> tuple[
        torch.Tensor,
        TitansMemoryState | TPPGatedMemoryState,
        dict[str, torch.Tensor],
    ]:
        """Run B1 or B2 with their shared explicit streaming-state API."""
        features = self.continuous_features(dts, history_quantities, mask)
        if self.titans_mac_encoder is not None:
            if state is not None and not isinstance(state, TitansMemoryState):
                raise TypeError("B1 requires TitansMemoryState")
            encoded, next_state, diagnostics = self.titans_mac_encoder.forward_with_state(
                features,
                mask=mask,
                write_mask=memory_write_mask,
                state=state,
                series_ids=series_ids,
                segment_size=segment_size,
                write_chunk_size=write_chunk_size,
            )
        elif self.tpp_gated_memory is not None:
            if segment_size is not None:
                raise ValueError("segment_size applies only to B1 Titans-MAC")
            if state is not None and not isinstance(state, TPPGatedMemoryState):
                raise TypeError("B2 requires TPPGatedMemoryState")
            prepared_state = self.tpp_gated_memory.prepare_state(
                state,
                batch_size=features.size(0),
                device=features.device,
                dtype=features.dtype,
                series_ids=series_ids,
            )
            if self.encoder is None:
                raise RuntimeError("B2 local encoder is not initialized")
            local_encoded = self.encoder(
                features,
                mask=mask,
                update_context_memory=False,
                position_offset=prepared_state.positions,
            )
            encoded, next_state, diagnostics = (
                self.tpp_gated_memory.forward_with_state(
                    local_encoded,
                    mask=mask,
                    write_mask=memory_write_mask,
                    state=prepared_state,
                    series_ids=series_ids,
                    write_chunk_size=write_chunk_size,
                )
            )
        else:
            raise RuntimeError("Explicit online state is available only for B1 or B2")
        valid = mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return encoded * valid, next_state, diagnostics

    def encode_task_states(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
        *,
        memory_write_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base = self._encode_base(
            dts,
            history_quantities,
            mask,
            memory_write_mask=memory_write_mask,
        )
        valid = mask.unsqueeze(-1).to(dtype=base.dtype)

        if self.memory_mode == TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE:
            if self.lmm is None or self.surprise_memory is None:
                raise RuntimeError("Dual memory requires both hard and surprise paths")
            time_encoded = self.lmm(base) * valid
            if self.quantity_memory_gradient_mode == TITAN_QUANTITY_GRADIENT_SHARED:
                surprise_residual = self.surprise_memory.residual(base, mask=mask)
                quantity_encoded = time_encoded + surprise_residual
            else:
                detached_base = base.detach()
                surprise_residual = self.surprise_memory.residual(
                    detached_base,
                    mask=mask,
                )
                quantity_encoded = time_encoded.detach() + surprise_residual
            return time_encoded, quantity_encoded * valid

        encoded = base
        if self.lmm is not None:
            encoded = self.lmm(encoded)
        if self.soft_memory is not None:
            encoded = self.soft_memory(encoded, mask=mask)
        if self.surprise_memory is not None:
            encoded = self.surprise_memory(encoded, mask=mask)
        encoded = encoded * valid
        return encoded, encoded

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        time_encoded, _ = self.encode_task_states(
            dts,
            history_quantities,
            mask,
        )
        return time_encoded


__all__ = [
    "CountAwareRMTPP",
    "CountAwareTHP",
    "CountAwareTitanTPP",
    "LOG_MSE_VARIANT",
    "LOGNORMAL_VARIANT",
    "SharedTimeCountModel",
    "TAIL_HEAD_ONLY_VARIANT",
    "TAIL_SHARED_VARIANT",
    "TAIL_VARIANTS",
    "TIME_HEAD_MODE_LEGACY_CLAMPED",
    "TIME_HEAD_MODE_LOGNORMAL_DURATION",
    "TIME_HEAD_MODE_SCALED_EXACT",
    "TIME_HEAD_MODE_SCALED_EXACT_STABLE",
    "TIME_HEAD_MODES",
    "TITAN_MEMORY_MODE_NONE",
    "TITAN_MEMORY_MODE_PERSISTENT_ONLY",
    "TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED",
    "TITAN_MEMORY_MODE_STATIC_HARD",
    "TITAN_MEMORY_MODE_STATIC_WEIGHTED",
    "TITAN_MEMORY_MODE_STATIC_SOFT_GATED",
    "TITAN_MEMORY_MODE_SURPRISE_GATED",
    "TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE",
    "TITAN_MEMORY_MODE_TITANS_MAC",
    "TITAN_MEMORY_MODE_TPP_GATED",
    "TITAN_MEMORY_MODES",
    "TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY",
    "TITAN_QUANTITY_GRADIENT_MODES",
    "TITAN_QUANTITY_GRADIENT_SHARED",
    "inverse_sigmoid",
    "inverse_softplus",
]
