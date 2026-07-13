from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class THPCandidate:
    """
    Compact Transformer Hawkes Process preset.

    TitanTPP already has `TitanCandidate`; this mirrors that idea for THP so
    experiment runners can sweep both model families through the same candidate
    loop.
    """
    name: str
    d_model: int = 128
    d_inner: int = 512
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    normalize_before: bool = False
    add_temporal_encoding_each_layer: bool = True
    use_rnn: bool = False
    d_rnn: int = 128


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Unified runtime config shared by long-epoch style experiments.

    The field names intentionally stay compatible with the existing
    `make_search_cfg(...)` helper, which expects attributes like `base_dir`,
    `lookback_weeks`, `batch_size`, and `val_ratio`.
    """
    base_dir: str
    experiment_mode: str = "long-epoch"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lookback_weeks: int = 52
    max_seq_len: int = 256
    batch_size: int = 128
    lr: float = 1e-3
    val_ratio: float = 0.2
    lambda_value: float = 1.0
    lambda_dt: float = 1.0
    grad_clip: float = 1.0
    epochs: int = 100
    seeds: tuple[int, ...] = (42, 52, 62)
    datasets: tuple[str, ...] = ("intermittent", "yellow_trip_hourly")
    models: tuple[str, ...] = ("rmtpp", "titantpp")
    titan_profile: str = "dataset_best"
    titan_candidates: tuple[str, ...] = ()
    thp_candidates: tuple[str, ...] = ()
    intermittent_max_series: int | None = None
    yellow_max_series: int | None = None
    insta_max_series: int | None = None
    force_rerun: bool = False
    stop_on_error: bool = False
    split_mode: str = "internal"
    rmtpp_rnn_type: str = "gru"
    rmtpp_mark_emb_dim: int = 32
    rmtpp_hidden_dim: int | None = None
    value_head_activation: str = "sigmoid"
    value_head_mode: str = "shared"
    qty_mark_gradient_mode: str = "coupled"
    value_encoder_gradient_mode: str = "coupled"
    marker_loss_mode: str = "ce"
    lambda_ordinal: float = 0.0
    qty_decoder_mode: str = "mark_residual"
    magnitude_norm_mode: str = "global"
    magnitude_input_emb_dim: int = 8
    lambda_magnitude: float = 1.0
    magnitude_encoder_gradient_mode: str = "coupled"
    magnitude_aux_loss_mode: str = "none"
    lambda_log_qty: float = 0.25
    log_qty_huber_delta: float = 1.0
    log_qty_floor: float = 1.0
    magnitude_sigma_floor: float = 0.0014535461338152059
    magnitude_revin_eps: float = 1e-5
    magnitude_shrinkage_k: float = 8.0
    magnitude_center_mode: str = "mean"
    magnitude_revin_affine: bool = False
    magnitude_stat_context_mode: str = "none"
    magnitude_exp_clamp_min: float = -2.0
    magnitude_exp_clamp_max: float = 15.0
    loss_mode: str = "residual_only"
    value_input_mode: str = "none"
    value_input_emb_dim: int = 8
    train_loss_scope: str = "all"
    test_time_memory: str = "none"
    analysis_scale_base: float = 10.0
    analysis_tail_order: int = 4
    eval_selections: tuple[str, ...] = ("best_val_nll",)


@dataclass(frozen=True)
class RunConfig:
    """
    Full identity of one concrete train/eval run.
    """
    dataset_name: str
    dataset_kind: str
    model_name: str
    candidate_name: str
    candidate: Any
    seed: int
    epochs: int
    scale_base: float
    titan_profile: str


@dataclass(frozen=True)
class RunPaths:
    """
    Canonical output layout for one run.
    """
    run_dir: Path
    checkpoint_dir: Path
    metrics_dir: Path
    manifest_dir: Path
    logs_dir: Path
