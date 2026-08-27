"""Factory for count-aware TPP backbone controls with shared output heads."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    CountAwareRMTPP,
    CountAwareTHP,
    CountAwareTitanTPP,
    SharedTimeCountModel,
    TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
    TITAN_MEMORY_MODE_NONE,
    TITAN_MEMORY_MODE_PERSISTENT_ONLY,
    TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_MEMORY_MODE_STATIC_SOFT_GATED,
    TITAN_MEMORY_MODE_SURPRISE_GATED,
    TITAN_MEMORY_MODE_TITANS_MAC,
    TITAN_MEMORY_MODE_TPP_GATED,
    TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY,
    TITAN_QUANTITY_GRADIENT_SHARED,
)
from models.TPPs.NeuralHawkesTPP import CountAwareNHP
from models.TPPs.SelfAttentiveHawkesTPP import CountAwareSAHP


def with_time_metadata(
    model: SharedTimeCountModel,
    encoder_metadata: dict[str, Any],
) -> tuple[SharedTimeCountModel, dict[str, Any]]:
    """Attach the shared time-head contract to backbone metadata."""
    return model, {**encoder_metadata, "time_head": model.time_head_contract()}


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
    time_head_mode: str = TIME_HEAD_MODE_LEGACY_CLAMPED,
    time_scale: float = 3.0,
    time_w_max: float = 10.0 / 3.0,
    time_intercept_limit: float = 30.0,
    time_initial_intercept: float | None = None,
    time_wd_safety_limit: float = 40.0,
    time_initial_location: float | None = None,
    time_initial_scale: float | None = None,
    time_sigma_floor: float = 1e-3,
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
        "time_head_mode": time_head_mode,
        "time_scale": time_scale,
        "time_w_max": time_w_max,
        "time_intercept_limit": time_intercept_limit,
        "time_initial_intercept": time_initial_intercept,
        "time_wd_safety_limit": time_wd_safety_limit,
        "time_initial_location": time_initial_location,
        "time_initial_scale": time_initial_scale,
        "time_sigma_floor": time_sigma_floor,
    }
    if backbone == "rmtpp":
        model = CountAwareRMTPP(hidden_dim, train_log_mean, **quantity_kwargs)
        return with_time_metadata(
            model,
            {
                "candidate_name": "count_gru_h64",
                "rnn_type": "gru",
                "hidden_dim": hidden_dim,
            },
        )
    if backbone == "nhp":
        model = CountAwareNHP(hidden_dim, train_log_mean, **quantity_kwargs)
        return with_time_metadata(
            model,
            {
                "candidate_name": "count_nhp_ctlstm_h64",
                "encoder_type": "continuous_time_lstm",
                "hidden_dim": hidden_dim,
                "shared_time_head": True,
            },
        )
    if backbone == "sahp":
        model = CountAwareSAHP(hidden_dim, train_log_mean, **quantity_kwargs)
        return with_time_metadata(
            model,
            {
                "candidate_name": "count_sahp_small",
                "encoder_type": "causal_self_attention_with_continuous_decay",
                "hidden_dim": hidden_dim,
                "n_layers": 2,
                "n_heads": 4,
                "d_ff": hidden_dim * 4,
                "shared_time_head": True,
            },
        )
    if backbone == "thp":
        model = CountAwareTHP(hidden_dim, train_log_mean, **quantity_kwargs)
        return with_time_metadata(
            model,
            {
                "candidate_name": "count_thp_small",
                **asdict(model.encoder_config),
            },
        )
    titan_modes = {
        "titantpp": (
            TITAN_MEMORY_MODE_STATIC_HARD,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_no_memory": (
            TITAN_MEMORY_MODE_NONE,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_gated_soft_memory": (
            TITAN_MEMORY_MODE_STATIC_SOFT_GATED,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_surprise_memory": (
            TITAN_MEMORY_MODE_SURPRISE_GATED,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_persistent_only": (
            TITAN_MEMORY_MODE_PERSISTENT_ONLY,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_persistent_surprise_memory": (
            TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_dual_memory_shared": (
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_dual_memory_adapter_only": (
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
            TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY,
        ),
        "titantpp_titans_mac": (
            TITAN_MEMORY_MODE_TITANS_MAC,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
        "titantpp_tpp_gated_memory": (
            TITAN_MEMORY_MODE_TPP_GATED,
            TITAN_QUANTITY_GRADIENT_SHARED,
        ),
    }
    if backbone in titan_modes:
        memory_mode, quantity_memory_gradient_mode = titan_modes[backbone]
        uses_persistent_memory = memory_mode in {
            TITAN_MEMORY_MODE_PERSISTENT_ONLY,
            TITAN_MEMORY_MODE_STATIC_HARD,
            TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
            TITAN_MEMORY_MODE_TITANS_MAC,
            TITAN_MEMORY_MODE_TPP_GATED,
        }
        uses_hard_memory = memory_mode in {
            TITAN_MEMORY_MODE_STATIC_HARD,
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
        }
        uses_soft_memory = memory_mode == TITAN_MEMORY_MODE_STATIC_SOFT_GATED
        uses_surprise_memory = memory_mode in {
            TITAN_MEMORY_MODE_SURPRISE_GATED,
            TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
        }
        model = CountAwareTitanTPP(
            hidden_dim,
            train_log_mean,
            max_seq_len,
            memory_mode=memory_mode,
            quantity_memory_gradient_mode=quantity_memory_gradient_mode,
            **quantity_kwargs,
        )
        candidate_names = {
            "titantpp": "count_titan_small_lmm",
            "titantpp_no_memory": "count_titan_no_memory",
            "titantpp_gated_soft_memory": "count_titan_gated_soft_memory",
            "titantpp_surprise_memory": "count_titan_surprise_memory",
            "titantpp_persistent_only": "count_titan_persistent_only",
            "titantpp_persistent_surprise_memory": (
                "count_titan_persistent_surprise_memory"
            ),
            "titantpp_dual_memory_shared": "count_titan_dual_memory_shared",
            "titantpp_dual_memory_adapter_only": (
                "count_titan_dual_memory_adapter_only"
            ),
            "titantpp_titans_mac": "count_titan_faithful_titans_mac",
            "titantpp_tpp_gated_memory": "count_titan_tpp_specific_gated_memory",
        }
        return with_time_metadata(
            model,
            {
                "candidate_name": candidate_names[backbone],
                "backbone_contract_id": (
                    "B1"
                    if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
                    else "B2"
                    if memory_mode == TITAN_MEMORY_MODE_TPP_GATED
                    else "B0"
                    if backbone == "titantpp"
                    else "historical_ablation"
                ),
                "d_model": hidden_dim,
                "n_layers": 2,
                "n_heads": 4,
                "d_ff": hidden_dim * 2,
                "memory_mode": memory_mode,
                "persistent_mem_size": 16 if uses_persistent_memory else 0,
                "persistent_memory_update_scope": (
                    "outer_loop_only"
                    if memory_mode
                    in {
                        TITAN_MEMORY_MODE_TITANS_MAC,
                        TITAN_MEMORY_MODE_TPP_GATED,
                    }
                    else None
                ),
                "lmm_mem_size": 64 if uses_hard_memory else 0,
                "lmm_topk": 4 if uses_hard_memory else 0,
                "soft_memory_size": 64 if uses_soft_memory else 0,
                "soft_memory_temperature": (
                    1.0 if uses_soft_memory else None
                ),
                "surprise_memory_rank": (
                    min(16, hidden_dim) if uses_surprise_memory else 0
                ),
                "surprise_chunk_size": 32 if uses_surprise_memory else 0,
                "surprise_scan_backend": (
                    "compiled_sequence_cuda" if uses_surprise_memory else None
                ),
                "surprise_update_rate_init": (
                    0.01 if uses_surprise_memory else None
                ),
                "surprise_retention_init": (
                    0.99 if uses_surprise_memory else None
                ),
                "surprise_momentum_init": (
                    0.5 if uses_surprise_memory else None
                ),
                "surprise_state_scope": (
                    "independent_input_sequence"
                    if uses_surprise_memory
                    or memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
                    else None
                ),
                "titans_neural_memory_depth": (
                    2 if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC else 0
                ),
                "titans_neural_memory_hidden_expansion": (
                    2 if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC else 0
                ),
                "titans_mac_segment_size": (
                    16 if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC else 0
                ),
                "titans_scan_backend": (
                    "compiled_sequence_cuda"
                    if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
                    else None
                ),
                "titans_online_update": (
                    "surprise_momentum_adaptive_forgetting"
                    if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
                    else None
                ),
                "titans_event_order": (
                    "segment_read_prediction_then_observed_write"
                    if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
                    else None
                ),
                "tpp_gated_memory_size": (
                    64 if memory_mode == TITAN_MEMORY_MODE_TPP_GATED else 0
                ),
                "tpp_gated_topk": (
                    4 if memory_mode == TITAN_MEMORY_MODE_TPP_GATED else 0
                ),
                "tpp_gated_temperature": (
                    1.0 if memory_mode == TITAN_MEMORY_MODE_TPP_GATED else None
                ),
                "tpp_gated_retrieval": (
                    "similarity_weighted_sparse_topk_with_null"
                    if memory_mode == TITAN_MEMORY_MODE_TPP_GATED
                    else None
                ),
                "tpp_gated_confidence": (
                    "null_mass_times_learned_scalar_gate"
                    if memory_mode == TITAN_MEMORY_MODE_TPP_GATED
                    else None
                ),
                "tpp_gated_write_policy": (
                    "circular_observed_event_after_prediction"
                    if memory_mode == TITAN_MEMORY_MODE_TPP_GATED
                    else None
                ),
                "tpp_gated_state_scope": (
                    "explicit_per_series_state"
                    if memory_mode == TITAN_MEMORY_MODE_TPP_GATED
                    else None
                ),
                "tpp_gated_scan_backend": (
                    "compiled_sequence_cuda"
                    if memory_mode == TITAN_MEMORY_MODE_TPP_GATED
                    else None
                ),
                "memory_residual_gate_init": (
                    0.0 if uses_soft_memory or uses_surprise_memory else None
                ),
                "time_memory_route": (
                    "hard_local_memory_matcher"
                    if uses_hard_memory
                    else "titans_mac"
                    if memory_mode == TITAN_MEMORY_MODE_TITANS_MAC
                    else "tpp_specific_gated_memory"
                    if memory_mode == TITAN_MEMORY_MODE_TPP_GATED
                    else "shared_memory_state"
                ),
                "quantity_memory_route": (
                    "hard_lmm_plus_surprise_residual"
                    if memory_mode == TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE
                    else "shared_memory_state"
                ),
                "quantity_memory_gradient_mode": (
                    quantity_memory_gradient_mode
                    if memory_mode == TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE
                    else "shared_state"
                ),
                "max_len": max_seq_len,
            },
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


__all__ = ["build_count_aware_model"]
