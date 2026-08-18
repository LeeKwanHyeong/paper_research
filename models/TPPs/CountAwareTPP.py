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
from models.Titan.common.memory import GatedSoftMemory, LMM, SurpriseGatedMemory


LOG_MSE_VARIANT = "count_only_log_regression"
LOGNORMAL_VARIANT = "count_only_lognormal_k1"
TAIL_SHARED_VARIANT = "count_only_log_mse_tail_shared"
TAIL_HEAD_ONLY_VARIANT = "count_only_log_mse_tail_head_only"
TAIL_VARIANTS = (TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT)
TITAN_MEMORY_MODE_NONE = "none"
TITAN_MEMORY_MODE_STATIC_HARD = "static_hard_lmm"
TITAN_MEMORY_MODE_STATIC_SOFT_GATED = "static_soft_gated"
TITAN_MEMORY_MODE_SURPRISE_GATED = "surprise_gated"
TITAN_MEMORY_MODES = (
    TITAN_MEMORY_MODE_NONE,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_MEMORY_MODE_STATIC_SOFT_GATED,
    TITAN_MEMORY_MODE_SURPRISE_GATED,
)


def inverse_softplus(value: float) -> float:
    """Return the pre-activation that maps to ``value`` through softplus."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse_softplus requires a finite positive value")
    return value + math.log(-math.expm1(-value))


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

        self.v_t = nn.Linear(self.hidden_dim, 1, bias=False)
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
        w = F.softplus(self.w_raw) + 1e-3
        intercept = torch.clamp(self.v_t(hidden).squeeze(-1) + self.b_t, max=300.0)
        exp_intercept = torch.exp(intercept)
        wd = torch.clamp(w * dt_next, max=10.0)
        return intercept + wd - (exp_intercept / w) * torch.expm1(wd)

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
        **quantity_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        if memory_mode not in TITAN_MEMORY_MODES:
            raise ValueError(
                f"Unsupported count-aware Titan memory_mode: {memory_mode}"
            )
        self.memory_mode = memory_mode
        uses_static_memory = memory_mode == TITAN_MEMORY_MODE_STATIC_HARD
        self.encoder = MemoryEncoder(
            input_dim=2,
            d_model=hidden_dim,
            n_layers=2,
            n_heads=4,
            d_ff=hidden_dim * 2,
            contextual_mem_size=0,
            persistent_mem_size=16 if uses_static_memory else 0,
            dropout=0.1,
            use_context_update=False,
            use_pos_emb=True,
            max_len=max_seq_len,
            use_causal=True,
        )
        self.lmm = (
            LMM(d_model=hidden_dim, mem_size=64, topk=4)
            if uses_static_memory
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
            if memory_mode == TITAN_MEMORY_MODE_SURPRISE_GATED
            else None
        )

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.continuous_features(dts, history_quantities, mask)
        encoded = self.encoder(x, mask=mask, update_context_memory=False)
        if self.lmm is not None:
            encoded = self.lmm(encoded)
        if self.soft_memory is not None:
            encoded = self.soft_memory(encoded, mask=mask)
        if self.surprise_memory is not None:
            encoded = self.surprise_memory(encoded, mask=mask)
        return encoded * mask.unsqueeze(-1).to(dtype=encoded.dtype)


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
    "TITAN_MEMORY_MODE_NONE",
    "TITAN_MEMORY_MODE_STATIC_HARD",
    "TITAN_MEMORY_MODE_STATIC_SOFT_GATED",
    "TITAN_MEMORY_MODE_SURPRISE_GATED",
    "TITAN_MEMORY_MODES",
    "inverse_softplus",
]
