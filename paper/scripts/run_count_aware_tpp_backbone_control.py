#!/usr/bin/env python3
"""Run the mark-free count-aware TPP backbone control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RMTPPs.TransformerHawkesTPP import THPEncoderLayer
from models.RMTPPs.config import THPConfig
from models.Titan.backbone import MemoryEncoder
from models.Titan.common.memory import LMM
from paper.scripts.run_intermittent_log_backbone_control import (
    BACKBONES,
    EXPECTED_DATA_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
    HISTORY_BOUNDARIES,
    HISTORY_STRATA,
)
from paper.scripts.run_taxi_quantity_interface_ablation import (
    clone_state_dict,
    make_loader,
    parse_int_tuple,
    parse_str_tuple,
    save_json,
    set_seed,
    sha256_file,
    train_quantile_contract,
)
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


SEEDS = (42, 52, 62)
VARIANT = "count_only_log_regression"
LOGNORMAL_VARIANT = "count_only_lognormal_k1"
TAIL_SHARED_VARIANT = "count_only_log_mse_tail_shared"
TAIL_HEAD_ONLY_VARIANT = "count_only_log_mse_tail_head_only"
TAIL_VARIANTS = (TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT)
QUANTITY_VARIANT_ALIASES = {
    "log_mse": VARIANT,
    VARIANT: VARIANT,
    "lognormal_k1": LOGNORMAL_VARIANT,
    LOGNORMAL_VARIANT: LOGNORMAL_VARIANT,
    "tail_shared": TAIL_SHARED_VARIANT,
    TAIL_SHARED_VARIANT: TAIL_SHARED_VARIANT,
    "tail_head_only": TAIL_HEAD_ONLY_VARIANT,
    TAIL_HEAD_ONLY_VARIANT: TAIL_HEAD_ONLY_VARIANT,
}
BACKBONE_LABELS = {
    "rmtpp": "Count-aware RMTPP",
    "thp": "Count-aware THP",
    "titantpp": "Count-aware TitanTPP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execution-role", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback-weeks", type=int, default=520)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lambda-log-qty", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    parser.add_argument("--quantity-variants", default=VARIANT)
    parser.add_argument("--quantity-sigma-floor", type=float, default=1e-3)
    parser.add_argument("--lambda-location-huber", type=float, default=1.0)
    parser.add_argument("--location-huber-delta", type=float, default=0.25)
    parser.add_argument("--lambda-tail", type=float, default=0.0)
    parser.add_argument("--tail-threshold", type=float, default=46.0)
    parser.add_argument("--tail-normalization-scale", type=float, default=46.0)
    parser.add_argument("--tail-clip-cap", type=float, default=187.0)
    parser.add_argument("--tail-huber-delta", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("inverse_softplus requires a finite positive value")
    return value + math.log(-math.expm1(-value))


def normalize_quantity_variants(raw: str) -> tuple[str, ...]:
    requested = parse_str_tuple(raw)
    try:
        normalized = tuple(QUANTITY_VARIANT_ALIASES[name] for name in requested)
    except KeyError as exc:
        available = ", ".join(sorted(QUANTITY_VARIANT_ALIASES))
        raise ValueError(
            f"Unsupported quantity variant '{exc.args[0]}'. Available: {available}"
        ) from exc
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate quantity variants after alias resolution: {normalized}")
    return normalized


class SharedTimeCountModel(nn.Module):
    """Common time-density and continuous-count heads for every backbone."""

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        *,
        train_log_std: float = 1.0,
        quantity_variant: str = VARIANT,
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
        if quantity_variant not in {VARIANT, LOGNORMAL_VARIANT, *TAIL_VARIANTS}:
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
        raise NotImplementedError

    @staticmethod
    def continuous_features(
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack(
            [
                torch.log1p(dts.float().clamp_min(0.0)),
                torch.log1p(history_quantities.float().clamp_min(0.0)),
            ],
            dim=-1,
        )
        return features * mask.unsqueeze(-1).to(dtype=features.dtype)

    def log_f_dt(self, hidden: torch.Tensor, dt_next: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.w_raw) + 1e-3
        intercept = torch.clamp(self.v_t(hidden).squeeze(-1) + self.b_t, max=300.0)
        exp_intercept = torch.exp(intercept)
        wd = torch.clamp(w * dt_next, max=10.0)
        return intercept + wd - (exp_intercept / w) * torch.expm1(wd)

    def predict_quantity(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        log_quantity = F.softplus(self.quantity_head(hidden).squeeze(-1))
        return log_quantity, torch.expm1(log_quantity)

    def quantity_outputs(
        self,
        hidden: torch.Tensor,
        true_quantity: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        location, point_prediction = self.predict_quantity(hidden)
        target = torch.log1p(true_quantity.clamp_min(0.0))
        log_mse = F.mse_loss(location, target, reduction="none")
        zeros = torch.zeros_like(log_mse)
        if self.quantity_variant == VARIANT:
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
        train_loss = (
            distribution_nll
            + self.lambda_location_huber * location_huber
        )
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
        x = self.input_dropout(self.input_projection(
            self.continuous_features(dts, history_quantities, mask)
        ))
        encoded, _ = self.encoder(x)
        return encoded * mask.unsqueeze(-1).to(dtype=encoded.dtype)


class CountAwareTHP(SharedTimeCountModel):
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
        self.layers = nn.ModuleList([
            THPEncoderLayer(self.encoder_config)
            for _ in range(self.encoder_config.n_layers)
        ])

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
        x = self.input_projection(self.continuous_features(dts, history_quantities, mask))
        non_pad = mask.unsqueeze(-1).to(dtype=x.dtype)
        blocked = self.blocked_attention_mask(mask)
        for layer in self.layers:
            x = layer(x, non_pad_mask=non_pad, blocked_mask=blocked)
        return x * non_pad


class CountAwareTitanTPP(SharedTimeCountModel):
    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        max_seq_len: int,
        **quantity_kwargs: Any,
    ) -> None:
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        self.encoder = MemoryEncoder(
            input_dim=2,
            d_model=hidden_dim,
            n_layers=2,
            n_heads=4,
            d_ff=hidden_dim * 2,
            contextual_mem_size=0,
            persistent_mem_size=16,
            dropout=0.1,
            use_context_update=False,
            use_pos_emb=True,
            max_len=max_seq_len,
            use_causal=True,
        )
        self.lmm = LMM(d_model=hidden_dim, mem_size=64, topk=4)

    def encode(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.continuous_features(dts, history_quantities, mask)
        encoded = self.encoder(x, mask=mask, update_context_memory=False)
        encoded = self.lmm(encoded)
        return encoded * mask.unsqueeze(-1).to(dtype=encoded.dtype)


def build_model(
    backbone: str,
    *,
    hidden_dim: int,
    train_log_mean: float,
    max_seq_len: int,
    train_log_std: float = 1.0,
    quantity_variant: str = VARIANT,
    quantity_sigma_floor: float = 1e-3,
    lambda_location_huber: float = 1.0,
    location_huber_delta: float = 0.25,
    lambda_tail: float = 0.0,
    tail_threshold: float = 46.0,
    tail_normalization_scale: float = 46.0,
    tail_clip_cap: float = 187.0,
    tail_huber_delta: float = 1.0,
) -> tuple[SharedTimeCountModel, dict[str, Any]]:
    quantity_kwargs = {
        "train_log_std": train_log_std,
        "quantity_variant": quantity_variant,
        "quantity_sigma_floor": quantity_sigma_floor,
        "lambda_location_huber": lambda_location_huber,
        "location_huber_delta": location_huber_delta,
        "lambda_tail": lambda_tail,
        "tail_threshold": tail_threshold,
        "tail_normalization_scale": tail_normalization_scale,
        "tail_clip_cap": tail_clip_cap,
        "tail_huber_delta": tail_huber_delta,
    }
    if backbone == "rmtpp":
        return CountAwareRMTPP(hidden_dim, train_log_mean, **quantity_kwargs), {
            "candidate_name": "count_gru_h64",
            "rnn_type": "gru",
            "hidden_dim": hidden_dim,
        }
    if backbone == "thp":
        model = CountAwareTHP(hidden_dim, train_log_mean, **quantity_kwargs)
        return model, {"candidate_name": "count_thp_small", **asdict(model.encoder_config)}
    if backbone == "titantpp":
        return CountAwareTitanTPP(
            hidden_dim,
            train_log_mean,
            max_seq_len,
            **quantity_kwargs,
        ), {
            "candidate_name": "count_titan_small_lmm",
            "d_model": hidden_dim,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": hidden_dim * 2,
            "persistent_mem_size": 16,
            "lmm_mem_size": 64,
            "lmm_topk": 4,
            "max_len": max_seq_len,
        }
    raise ValueError(f"Unsupported backbone: {backbone}")


def prepare_count_frame(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.filter(pl.col("demand_qty") < 0).height:
        raise ValueError("Count-aware input requires nonnegative demand_qty")
    return frame.with_columns([
        pl.lit(0, dtype=pl.Int32).alias("mark"),
        pl.col("demand_qty").cast(pl.Float64).alias("scale_residual"),
    ])


def right_pad_batch(
    dts: torch.Tensor,
    quantities: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len = mask.shape
    positions = torch.arange(seq_len, device=mask.device).expand(batch_size, -1)
    sort_key = (~mask).long() * seq_len + positions
    order = torch.argsort(sort_key, dim=1)
    right_dts = torch.gather(dts, 1, order)
    right_quantities = torch.gather(quantities, 1, order)
    right_mask = torch.gather(mask, 1, order)
    lengths = right_mask.sum(dim=1)
    if bool((lengths < 2).any()):
        raise ValueError("Every next-event sample requires at least one history event")
    return right_dts, right_quantities, right_mask, lengths


def target_outputs(
    model: SharedTimeCountModel,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    *,
    lambda_log_qty: float,
) -> dict[str, torch.Tensor]:
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    encoded = model.encode(dts, history_quantities, mask)
    hidden = encoded[batch_ids, history_positions]
    true_dt = dts[batch_ids, target_positions].float()
    true_qty = quantities[batch_ids, target_positions].float()
    time_loss = -model.log_f_dt(hidden, true_dt)
    quantity = model.quantity_outputs(hidden, true_qty)
    return {
        "joint_loss": time_loss + float(lambda_log_qty) * quantity["train_loss"],
        "time_loss": time_loss,
        "quantity_train_loss": quantity["train_loss"],
        "log_qty_loss": quantity["log_mse"],
        "quantity_distribution_nll": quantity["distribution_nll"],
        "quantity_location_huber": quantity["location_huber"],
        "quantity_scale": quantity["scale"],
        "tail_aux_loss": quantity["tail_aux_loss"],
        "tail_indicator": quantity["tail_indicator"],
        "true_qty": true_qty,
        "pred_qty": quantity["point_prediction"],
        "history_length": lengths - 1,
    }


def empty_accumulator() -> dict[str, float]:
    return {
        "count": 0,
        "joint_sum": 0.0,
        "time_sum": 0.0,
        "quantity_train_sum": 0.0,
        "log_qty_sum": 0.0,
        "distribution_nll_sum": 0.0,
        "location_huber_sum": 0.0,
        "tail_aux_sum": 0.0,
        "tail_count": 0,
        "scale_sum": 0.0,
        "abs_sum": 0.0,
        "sq_sum": 0.0,
        "signed_sum": 0.0,
    }


def update_accumulator(
    accumulator: dict[str, float],
    *,
    joint: np.ndarray,
    time_nll: np.ndarray,
    quantity_train_loss: np.ndarray,
    log_qty_mse: np.ndarray,
    distribution_nll: np.ndarray,
    location_huber: np.ndarray,
    tail_aux_loss: np.ndarray,
    tail_indicator: np.ndarray,
    scale: np.ndarray,
    true_qty: np.ndarray,
    pred_qty: np.ndarray,
) -> None:
    error = pred_qty - true_qty
    accumulator["count"] += int(true_qty.size)
    accumulator["joint_sum"] += float(joint.sum())
    accumulator["time_sum"] += float(time_nll.sum())
    accumulator["quantity_train_sum"] += float(quantity_train_loss.sum())
    accumulator["log_qty_sum"] += float(log_qty_mse.sum())
    accumulator["distribution_nll_sum"] += float(distribution_nll.sum())
    accumulator["location_huber_sum"] += float(location_huber.sum())
    accumulator["tail_aux_sum"] += float(tail_aux_loss.sum())
    accumulator["tail_count"] += int(tail_indicator.sum())
    accumulator["scale_sum"] += float(scale.sum())
    accumulator["abs_sum"] += float(np.abs(error).sum())
    accumulator["sq_sum"] += float(np.square(error).sum())
    accumulator["signed_sum"] += float(error.sum())


def finalize_accumulator(accumulator: dict[str, float]) -> dict[str, Any]:
    count = int(accumulator["count"])
    if count < 1:
        raise ValueError("Cannot finalize an empty accumulator")
    return {
        "count": count,
        "joint_objective": accumulator["joint_sum"] / count,
        "time_nll": accumulator["time_sum"] / count,
        "quantity_train_loss": accumulator["quantity_train_sum"] / count,
        "log_qty_mse": accumulator["log_qty_sum"] / count,
        "quantity_distribution_nll": accumulator["distribution_nll_sum"] / count,
        "quantity_location_huber": accumulator["location_huber_sum"] / count,
        "tail_aux_loss": accumulator["tail_aux_sum"] / count,
        "tail_count": int(accumulator["tail_count"]),
        "quantity_scale_mean": accumulator["scale_sum"] / count,
        "qty_mae": accumulator["abs_sum"] / count,
        "qty_rmse": float(np.sqrt(accumulator["sq_sum"] / count)),
        "qty_bias": accumulator["signed_sum"] / count,
    }


@torch.no_grad()
def evaluate(
    *,
    model: SharedTimeCountModel,
    loader: Any,
    quantity_contract: dict[str, Any],
    device: str,
    lambda_log_qty: float,
    max_batches: int | None,
    include_breakdowns: bool,
) -> dict[str, Any]:
    model.eval()
    overall = empty_accumulator()
    quantity_accumulators = [empty_accumulator() for _ in quantity_contract["strata"]]
    history_accumulators = [empty_accumulator() for _ in HISTORY_STRATA]

    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if quantities is None:
            raise ValueError("Count-aware evaluation requires raw quantities")
        dts = dts.to(device)
        mask = mask.to(device)
        quantities = quantities.to(device)
        outputs = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=lambda_log_qty,
        )
        joint = outputs["joint_loss"].cpu().numpy().astype(np.float64)
        time_nll = outputs["time_loss"].cpu().numpy().astype(np.float64)
        quantity_train_loss = outputs["quantity_train_loss"].cpu().numpy().astype(np.float64)
        log_qty_mse = outputs["log_qty_loss"].cpu().numpy().astype(np.float64)
        distribution_nll = outputs["quantity_distribution_nll"].cpu().numpy().astype(np.float64)
        location_huber = outputs["quantity_location_huber"].cpu().numpy().astype(np.float64)
        tail_aux_loss = outputs["tail_aux_loss"].cpu().numpy().astype(np.float64)
        tail_indicator = outputs["tail_indicator"].cpu().numpy().astype(np.float64)
        scale = outputs["quantity_scale"].cpu().numpy().astype(np.float64)
        true_qty = outputs["true_qty"].cpu().numpy().astype(np.float64)
        pred_qty = outputs["pred_qty"].cpu().numpy().astype(np.float64)
        history_length = outputs["history_length"].cpu().numpy().astype(np.int64)
        update_accumulator(
            overall,
            joint=joint,
            time_nll=time_nll,
            quantity_train_loss=quantity_train_loss,
            log_qty_mse=log_qty_mse,
            distribution_nll=distribution_nll,
            location_huber=location_huber,
            tail_aux_loss=tail_aux_loss,
            tail_indicator=tail_indicator,
            scale=scale,
            true_qty=true_qty,
            pred_qty=pred_qty,
        )
        if not include_breakdowns:
            continue

        quantity_ids = np.searchsorted(
            np.asarray(quantity_contract["boundaries"], dtype=np.float64),
            true_qty,
            side="left",
        )
        history_ids = np.searchsorted(
            np.asarray(HISTORY_BOUNDARIES, dtype=np.int64),
            history_length,
            side="left",
        )
        for ids, accumulators in (
            (quantity_ids, quantity_accumulators),
            (history_ids, history_accumulators),
        ):
            for index, accumulator in enumerate(accumulators):
                selected = ids == index
                if selected.any():
                    update_accumulator(
                        accumulator,
                        joint=joint[selected],
                        time_nll=time_nll[selected],
                        quantity_train_loss=quantity_train_loss[selected],
                        log_qty_mse=log_qty_mse[selected],
                        distribution_nll=distribution_nll[selected],
                        location_huber=location_huber[selected],
                        tail_aux_loss=tail_aux_loss[selected],
                        tail_indicator=tail_indicator[selected],
                        scale=scale[selected],
                        true_qty=true_qty[selected],
                        pred_qty=pred_qty[selected],
                    )

    overall_metrics = finalize_accumulator(overall)
    result: dict[str, Any] = {
        "val_joint_objective": overall_metrics["joint_objective"],
        "val_time_nll": overall_metrics["time_nll"],
        "val_quantity_train_loss": overall_metrics["quantity_train_loss"],
        "val_log_qty_mse": overall_metrics["log_qty_mse"],
        "val_quantity_distribution_nll": overall_metrics["quantity_distribution_nll"],
        "val_quantity_location_huber": overall_metrics["quantity_location_huber"],
        "val_tail_aux_loss": overall_metrics["tail_aux_loss"],
        "val_tail_count": overall_metrics["tail_count"],
        "val_quantity_scale_mean": overall_metrics["quantity_scale_mean"],
        "qty_mae": overall_metrics["qty_mae"],
        "qty_rmse": overall_metrics["qty_rmse"],
        "evaluated_count": overall_metrics["count"],
    }
    if not include_breakdowns:
        return result

    result["quantity_rows"] = [
        {
            **spec,
            "share": int(accumulator["count"]) / overall_metrics["count"],
            **finalize_accumulator(accumulator),
        }
        for spec, accumulator in zip(quantity_contract["strata"], quantity_accumulators)
    ]
    result["history_rows"] = [
        {
            **spec,
            "share": int(accumulator["count"]) / overall_metrics["count"],
            **finalize_accumulator(accumulator),
        }
        for spec, accumulator in zip(HISTORY_STRATA, history_accumulators)
    ]
    return result


def early_stopping_exhausted(
    history: list[dict[str, Any]],
    *,
    min_epochs: int,
    patience: int,
) -> bool:
    if not history or patience < 1:
        return False
    current_epoch = int(history[-1]["epoch"])
    best_epoch = int(min(history, key=lambda row: float(row["val_joint_objective"]))["epoch"])
    return current_epoch >= min_epochs and current_epoch - best_epoch >= patience


def train_one(
    *,
    args: argparse.Namespace,
    frame: pl.DataFrame,
    quantity_contract: dict[str, Any],
    interface_meta: dict[str, Any],
    backbone: str,
    quantity_variant: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = args.output_dir / "runs" / backbone / quantity_variant / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_val_joint_objective_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    if summary_path.exists() and best_path.exists() and not args.force_rerun:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        quantity_rows = payload.pop("quantity_rows")
        history_rows = payload.pop("history_rows")
        return payload, quantity_rows, history_rows

    generator = set_seed(seed)
    train_loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=generator,
    )
    val_loader = make_loader(
        frame,
        target_split="validation",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=False,
        generator=None,
    )
    model, encoder_config = build_model(
        backbone,
        hidden_dim=args.hidden_dim,
        train_log_mean=float(interface_meta["train_target_mean"]),
        train_log_std=float(interface_meta["train_target_std"]),
        max_seq_len=args.max_seq_len,
        quantity_variant=quantity_variant,
        quantity_sigma_floor=args.quantity_sigma_floor,
        lambda_location_huber=args.lambda_location_huber,
        location_huber_delta=args.location_huber_delta,
        lambda_tail=args.lambda_tail,
        tail_threshold=args.tail_threshold,
        tail_normalization_scale=args.tail_normalization_scale,
        tail_clip_cap=args.tail_clip_cap,
        tail_huber_delta=args.tail_huber_delta,
    )
    model.to(args.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history: list[dict[str, Any]] = []
    best_objective = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    source_revision_history = [args.source_revision]
    start_epoch = 1

    if last_path.exists() and not args.force_rerun:
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload.get("history", []))
        best_objective = float(payload.get("best_val_joint_objective", best_objective))
        best_state = payload.get("best_state_dict")
        start_epoch = int(payload["epoch"]) + 1
        source_revision_history = [
            revision for revision in payload.get("source_revision_history", []) if revision
        ]
        if args.source_revision not in source_revision_history:
            source_revision_history.append(args.source_revision)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    started = time.time()
    stopped_early = early_stopping_exhausted(
        history,
        min_epochs=args.min_epochs,
        patience=args.early_stopping_patience,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        if stopped_early:
            break
        model.train()
        running = 0.0
        batches = 0
        for batch_index, (_, dts, mask, _, quantities) in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            if quantities is None:
                raise ValueError("Count-aware training requires raw quantities")
            outputs = target_outputs(
                model,
                dts.to(args.device),
                mask.to(args.device),
                quantities.to(args.device),
                lambda_log_qty=args.lambda_log_qty,
            )
            loss = outputs["joint_loss"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += float(loss.item())
            batches += 1

        validation = evaluate(
            model=model,
            loader=val_loader,
            quantity_contract=quantity_contract,
            device=args.device,
            lambda_log_qty=args.lambda_log_qty,
            max_batches=args.max_val_batches,
            include_breakdowns=False,
        )
        epoch_row = {
            "epoch": epoch,
            "train_joint_objective": running / max(batches, 1),
            "val_joint_objective": float(validation["val_joint_objective"]),
            "val_time_nll": float(validation["val_time_nll"]),
            "val_quantity_train_loss": float(validation["val_quantity_train_loss"]),
            "val_log_qty_mse": float(validation["val_log_qty_mse"]),
            "val_quantity_distribution_nll": float(
                validation["val_quantity_distribution_nll"]
            ),
            "val_quantity_location_huber": float(
                validation["val_quantity_location_huber"]
            ),
            "val_tail_aux_loss": float(validation["val_tail_aux_loss"]),
            "val_tail_count": int(validation["val_tail_count"]),
            "val_quantity_scale_mean": float(validation["val_quantity_scale_mean"]),
            "val_qty_mae": float(validation["qty_mae"]),
            "val_qty_rmse": float(validation["qty_rmse"]),
        }
        history.append(epoch_row)
        line = (
            f"[epoch {epoch:03d}] backbone={backbone} "
            f"variant={quantity_variant} seed={seed} "
            f"train_joint={epoch_row['train_joint_objective']:.8f} "
            f"val_joint={epoch_row['val_joint_objective']:.8f} "
            f"time_nll={epoch_row['val_time_nll']:.8f} "
            f"log_qty_mse={epoch_row['val_log_qty_mse']:.8f} "
            f"tail_aux={epoch_row['val_tail_aux_loss']:.8f} "
            f"qty_mae={epoch_row['val_qty_mae']:.8f}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if epoch_row["val_joint_objective"] < best_objective:
            best_objective = epoch_row["val_joint_objective"]
            best_state = clone_state_dict(model)
        save_json(run_dir / "history.json", {"history": history})
        torch.save({
            "epoch": epoch,
            "backbone": backbone,
            "variant": quantity_variant,
            "seed": seed,
            "model_state_dict": clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_joint_objective": best_objective,
            "best_state_dict": best_state,
            "encoder_config": encoder_config,
            "interface_meta": interface_meta,
            "source_revision": args.source_revision,
            "source_revision_history": source_revision_history,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }, last_path)
        stopped_early = early_stopping_exhausted(
            history,
            min_epochs=args.min_epochs,
            patience=args.early_stopping_patience,
        )
        if stopped_early:
            best_epoch = min(
                history,
                key=lambda row: float(row["val_joint_objective"]),
            )["epoch"]
            print(
                f"[early-stop] backbone={backbone} variant={quantity_variant} seed={seed} "
                f"current_epoch={epoch} best_epoch={best_epoch}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError(
            f"No best checkpoint selected for {backbone}/{quantity_variant}/seed_{seed}"
        )
    model.load_state_dict(best_state, strict=True)
    validation = evaluate(
        model=model,
        loader=val_loader,
        quantity_contract=quantity_contract,
        device=args.device,
        lambda_log_qty=args.lambda_log_qty,
        max_batches=args.max_val_batches,
        include_breakdowns=args.max_val_batches is None,
    )
    state_digest = canonical_state_dict_sha256(best_state)
    if quantity_variant == VARIANT:
        selection_formula = "time_nll + lambda_log_qty * log1p_quantity_mse"
    elif quantity_variant in TAIL_VARIANTS:
        selection_formula = (
            "time_nll + lambda_log_qty * "
            "(log1p_quantity_mse + lambda_tail * tail_raw_huber)"
        )
    else:
        selection_formula = (
            "time_nll + lambda_log_qty * "
            "(gaussian_nll_on_log1p_quantity + lambda_location_huber * location_huber)"
        )
    checkpoint = {
        "selection": "best_validation_joint_objective",
        "selection_formula": selection_formula,
        "backbone": backbone,
        "variant": quantity_variant,
        "seed": seed,
        "model_state_dict": best_state,
        "model_state_sha256": state_digest,
        "encoder_config": encoder_config,
        "interface_meta": interface_meta,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    torch.save(checkpoint, best_path)
    quantity_rows = [{
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": quantity_variant,
        "seed": seed,
        **row,
    } for row in validation.get("quantity_rows", [])]
    history_rows = [{
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": quantity_variant,
        "seed": seed,
        **row,
    } for row in validation.get("history_rows", [])]
    best_epoch = int(min(
        history,
        key=lambda row: float(row["val_joint_objective"]),
    )["epoch"])
    summary = {
        "status": "success",
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": quantity_variant,
        "seed": seed,
        "epochs": args.epochs,
        "completed_epochs": int(history[-1]["epoch"]),
        "stopped_early": int(history[-1]["epoch"]) < args.epochs,
        "best_epoch": best_epoch,
        "best_val_joint_objective": float(validation["val_joint_objective"]),
        "best_val_time_nll": float(validation["val_time_nll"]),
        "best_val_quantity_train_loss": float(validation["val_quantity_train_loss"]),
        "best_val_log_qty_mse": float(validation["val_log_qty_mse"]),
        "best_val_quantity_distribution_nll": float(
            validation["val_quantity_distribution_nll"]
        ),
        "best_val_quantity_location_huber": float(
            validation["val_quantity_location_huber"]
        ),
        "best_val_tail_aux_loss": float(validation["val_tail_aux_loss"]),
        "best_val_tail_count": int(validation["val_tail_count"]),
        "best_val_quantity_scale_mean": float(validation["val_quantity_scale_mean"]),
        "lambda_tail": args.lambda_tail,
        "tail_threshold": args.tail_threshold,
        "tail_normalization_scale": args.tail_normalization_scale,
        "tail_clip_cap": args.tail_clip_cap,
        "tail_huber_delta": args.tail_huber_delta,
        "best_val_qty_mae": float(validation["qty_mae"]),
        "best_val_qty_rmse": float(validation["qty_rmse"]),
        "parameter_count": parameter_count,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_path": str(best_path),
        "checkpoint_state_sha256": state_digest,
        "elapsed_seconds": time.time() - started,
        "encoder_config": encoder_config,
        "interface_meta": interface_meta,
        "quantity_rows": quantity_rows,
        "history_rows": history_rows,
    }
    save_json(summary_path, summary)
    returned = dict(summary)
    returned.pop("quantity_rows")
    returned.pop("history_rows")
    return returned, quantity_rows, history_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_breakdowns(
    rows: list[dict[str, Any]],
    *,
    backbones: tuple[str, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    output = []
    strata = sorted({
        (int(row["stratum_order"]), row["stratum"], row["stratum_label"])
        for row in rows
    })
    for variant in variants:
        for backbone in backbones:
            for order, key, label in strata:
                group = [
                    row for row in rows
                    if row["variant"] == variant
                    and row["backbone"] == backbone
                    and row["stratum"] == key
                ]
                if {int(row["seed"]) for row in group} != set(seeds):
                    raise ValueError(
                        f"Seed contract failed for {variant}/{backbone}/{key}"
                    )
                record = {
                    "backbone": backbone,
                    "backbone_label": BACKBONE_LABELS[backbone],
                    "variant": variant,
                    "stratum_order": order,
                    "stratum": key,
                    "stratum_label": label,
                    "count": int(group[0]["count"]),
                    "share": float(group[0]["share"]),
                    "n_seeds": len(group),
                }
                for metric in (
                    "joint_objective",
                    "time_nll",
                    "quantity_train_loss",
                    "log_qty_mse",
                    "quantity_distribution_nll",
                    "quantity_location_huber",
                    "tail_aux_loss",
                    "quantity_scale_mean",
                    "qty_mae",
                    "qty_rmse",
                    "qty_bias",
                ):
                    values = [float(row[metric]) for row in group]
                    record[f"{metric}_mean"] = statistics.mean(values)
                    record[f"{metric}_std"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )
                output.append(record)
    return output


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    backbones = parse_str_tuple(args.backbones)
    seeds = parse_int_tuple(args.seeds)
    quantity_variants = normalize_quantity_variants(args.quantity_variants)
    if any(backbone not in BACKBONES for backbone in backbones):
        raise ValueError(f"Unsupported backbone selection: {backbones}")
    if not args.allow_partial_contract:
        if set(backbones) != set(BACKBONES) or set(seeds) != set(SEEDS):
            raise ValueError("Qualified run requires all backbones and seeds 42/52/62")
    if args.hidden_dim != 64 or args.max_seq_len != 256:
        raise ValueError("Frozen contract requires hidden_dim=64 and max_seq_len=256")
    if args.lambda_log_qty != 1.0:
        raise ValueError("Frozen contract requires lambda_log_qty=1.0")
    if LOGNORMAL_VARIANT in quantity_variants:
        if args.quantity_sigma_floor != 1e-3:
            raise ValueError("K=1 contract requires quantity_sigma_floor=1e-3")
        if args.lambda_location_huber != 1.0:
            raise ValueError("K=1 contract requires lambda_location_huber=1.0")
        if args.location_huber_delta != 0.25:
            raise ValueError("K=1 contract requires location_huber_delta=0.25")
    if any(variant in TAIL_VARIANTS for variant in quantity_variants):
        if args.lambda_tail <= 0.0:
            raise ValueError("Tail variants require a positive lambda_tail")
        if args.tail_threshold != 46.0:
            raise ValueError("Tail contract requires tail_threshold=46")
        if args.tail_normalization_scale != 46.0:
            raise ValueError("Tail contract requires tail_normalization_scale=46")
        if args.tail_clip_cap != 187.0:
            raise ValueError("Tail contract requires tail_clip_cap=187")
        if args.tail_huber_delta != 1.0:
            raise ValueError("Tail contract requires tail_huber_delta=1")

    data_sha256 = sha256_file(args.data)
    manifest_sha256 = sha256_file(args.split_manifest)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError(f"Unexpected fixed-split SHA-256: {data_sha256}")
    if manifest_sha256 != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise ValueError(f"Unexpected split-manifest SHA-256: {manifest_sha256}")
    raw_frame = pl.read_parquet(args.data).sort(["oper_part_no", "seq"])
    required = {
        "oper_part_no",
        "seq",
        "delta_t",
        "demand_qty",
        "chronological_split",
    }
    missing = sorted(required - set(raw_frame.columns))
    if missing:
        raise ValueError(f"Fixed split is missing columns: {missing}")
    quantity_contract = train_quantile_contract(raw_frame)
    train_qty = raw_frame.filter(
        pl.col("chronological_split") == "train"
    )["demand_qty"].to_numpy().astype(np.float64)
    train_log_qty = np.log1p(train_qty)
    shared_interface = {
        "history_features": ["log1p_delta_t", "log1p_raw_quantity"],
        "target": "log1p_raw_quantity",
        "quantity_output_activation": "softplus",
        "quantity_inverse_transform": "expm1",
        "point_prediction": "distribution_median_expm1_location",
        "point_prediction_shared_by_mae_and_rmse": True,
        "quantity_mark_used": False,
        "quantity_residual_used": False,
        "product_type_used": False,
        "target_quantity_masked_from_history": True,
        "train_target_mean": float(train_log_qty.mean()),
        "train_target_std": float(train_log_qty.std()),
        "fitted_on": "train",
    }
    interface_by_variant = {
        VARIANT: {
            **shared_interface,
            "mode": "mark_free_count_aware_log_regression",
            "quantity_loss": "mse_on_log1p_quantity",
        },
        LOGNORMAL_VARIANT: {
            **shared_interface,
            "mode": "mark_free_count_aware_lognormal_k1",
            "quantity_loss": "gaussian_nll_on_log1p_quantity_plus_location_huber",
            "distribution_components": 1,
            "quantity_sigma_activation": "softplus_plus_floor",
            "quantity_sigma_floor": args.quantity_sigma_floor,
            "lambda_location_huber": args.lambda_location_huber,
            "location_huber_delta": args.location_huber_delta,
        },
        TAIL_SHARED_VARIANT: {
            **shared_interface,
            "mode": "mark_free_count_aware_log_mse_tail_shared",
            "quantity_loss": "log1p_mse_plus_capped_normalized_raw_huber",
            "tail_gradient_route": "quantity_head_and_encoder",
            "lambda_tail": args.lambda_tail,
            "tail_threshold": args.tail_threshold,
            "tail_normalization_scale": args.tail_normalization_scale,
            "tail_clip_cap": args.tail_clip_cap,
            "tail_huber_delta": args.tail_huber_delta,
        },
        TAIL_HEAD_ONLY_VARIANT: {
            **shared_interface,
            "mode": "mark_free_count_aware_log_mse_tail_head_only",
            "quantity_loss": "log1p_mse_plus_capped_normalized_raw_huber",
            "tail_gradient_route": "quantity_head_only_via_detached_hidden",
            "lambda_tail": args.lambda_tail,
            "tail_threshold": args.tail_threshold,
            "tail_normalization_scale": args.tail_normalization_scale,
            "tail_clip_cap": args.tail_clip_cap,
            "tail_huber_delta": args.tail_huber_delta,
        },
    }
    frame = prepare_count_frame(raw_frame)
    split_rows = {
        str(row["chronological_split"]): int(row["len"])
        for row in raw_frame.group_by("chronological_split").agg(pl.len()).iter_rows(named=True)
    }
    contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "mark_free_count_aware_quantity_screening",
        "dataset": "intermittent_frozen_5000",
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha256,
        "split_rows": split_rows,
        "quantity_contract": quantity_contract,
        "history_length_contract": {
            "boundaries": list(HISTORY_BOUNDARIES),
            "strata": list(HISTORY_STRATA),
            "definition": "number of observed events before the validation target",
        },
        "quantity_variants": list(quantity_variants),
        "interfaces": {
            variant: interface_by_variant[variant]
            for variant in quantity_variants
        },
        "backbones": list(backbones),
        "seeds": list(seeds),
        "expected_run_count": len(quantity_variants) * len(backbones) * len(seeds),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_log_qty": args.lambda_log_qty,
        "lambda_tail": args.lambda_tail,
        "tail_contract": {
            "threshold": args.tail_threshold,
            "normalization_scale": args.tail_normalization_scale,
            "clip_cap": args.tail_clip_cap,
            "huber_delta": args.tail_huber_delta,
            "statistics_source_split": "train",
        },
        "grad_clip": args.grad_clip,
        "early_stopping": {
            "monitor": "validation_joint_objective",
            "formula_by_variant": {
                VARIANT: "time_nll + lambda_log_qty * log1p_quantity_mse",
                LOGNORMAL_VARIANT: "time_nll + lambda_log_qty * "
                "(gaussian_nll_on_log1p_quantity + lambda_location_huber * location_huber)",
                TAIL_SHARED_VARIANT: "time_nll + lambda_log_qty * "
                "(log1p_quantity_mse + lambda_tail * tail_raw_huber)",
                TAIL_HEAD_ONLY_VARIANT: "time_nll + lambda_log_qty * "
                "(log1p_quantity_mse + lambda_tail * tail_raw_huber)",
            },
            "min_epochs": args.min_epochs,
            "patience": args.early_stopping_patience,
            "restore": "best_validation_joint_objective",
        },
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "hidden_dim": args.hidden_dim,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "execution_host": os.uname().nodename,
        "execution_role": args.execution_role,
        "partial_smoke": args.max_train_batches is not None or args.max_val_batches is not None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "launch_contract.json", contract)

    summaries: list[dict[str, Any]] = []
    quantity_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    for quantity_variant in quantity_variants:
        for backbone in backbones:
            for seed in seeds:
                summary, run_quantity_rows, run_history_rows = train_one(
                    args=args,
                    frame=frame,
                    quantity_contract=quantity_contract,
                    interface_meta=interface_by_variant[quantity_variant],
                    backbone=backbone,
                    quantity_variant=quantity_variant,
                    seed=seed,
                )
                summaries.append(summary)
                quantity_rows.extend(run_quantity_rows)
                history_rows.extend(run_history_rows)
                write_csv(args.output_dir / "run_summaries.csv", summaries)
                if quantity_rows:
                    write_csv(args.output_dir / "quantity_seed_metrics.csv", quantity_rows)
                if history_rows:
                    write_csv(args.output_dir / "history_seed_metrics.csv", history_rows)

    if quantity_rows:
        write_csv(
            args.output_dir / "quantity_summary.csv",
            summarize_breakdowns(
                quantity_rows,
                backbones=backbones,
                variants=quantity_variants,
                seeds=seeds,
            ),
        )
    if history_rows:
        write_csv(
            args.output_dir / "history_summary.csv",
            summarize_breakdowns(
                history_rows,
                backbones=backbones,
                variants=quantity_variants,
                seeds=seeds,
            ),
        )
    contract["status"] = "complete"
    contract["completed_run_count"] = len(summaries)
    contract["held_out_test_evaluated"] = False
    save_json(args.output_dir / "launch_contract.json", contract)
    print(f"[complete] output_dir={args.output_dir} runs={len(summaries)}", flush=True)


if __name__ == "__main__":
    main()
