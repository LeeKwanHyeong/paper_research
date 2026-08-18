"""Factory for count-aware TPP backbone controls with shared output heads."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    CountAwareRMTPP,
    CountAwareTHP,
    CountAwareTitanTPP,
    SharedTimeCountModel,
)
from models.TPPs.NeuralHawkesTPP import CountAwareNHP
from models.TPPs.SelfAttentiveHawkesTPP import CountAwareSAHP


def build_count_aware_model(
    backbone: str,
    *,
    hidden_dim: int,
    train_log_mean: float,
    max_seq_len: int,
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
) -> tuple[SharedTimeCountModel, dict[str, Any]]:
    """Construct one controlled backbone and its serializable metadata."""
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
    if backbone == "nhp":
        return CountAwareNHP(hidden_dim, train_log_mean, **quantity_kwargs), {
            "candidate_name": "count_nhp_ctlstm_h64",
            "encoder_type": "continuous_time_lstm",
            "hidden_dim": hidden_dim,
            "shared_time_head": True,
        }
    if backbone == "sahp":
        return CountAwareSAHP(hidden_dim, train_log_mean, **quantity_kwargs), {
            "candidate_name": "count_sahp_small",
            "encoder_type": "causal_self_attention_with_continuous_decay",
            "hidden_dim": hidden_dim,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": hidden_dim * 4,
            "shared_time_head": True,
        }
    if backbone == "thp":
        model = CountAwareTHP(hidden_dim, train_log_mean, **quantity_kwargs)
        return model, {
            "candidate_name": "count_thp_small",
            **asdict(model.encoder_config),
        }
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


__all__ = ["build_count_aware_model"]
