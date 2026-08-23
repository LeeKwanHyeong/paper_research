#!/usr/bin/env python3
"""Run the mark-free count-aware TPP backbone control."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from models.TPPs.CountAwareTPP import (
    CountAwareRMTPP,
    CountAwareTHP,
    CountAwareTitanTPP,
    SharedTimeCountModel,
    TIME_HEAD_EXACT_MODES,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
    TIME_HEAD_MODE_SCALED_EXACT,
    TIME_HEAD_MODE_SCALED_EXACT_STABLE,
    TIME_HEAD_MODES,
)
from models.TPPs.CountAwareFactory import (
    build_count_aware_model as build_model,
)
from models.TPPs.NeuralHawkesTPP import CountAwareNHP
from models.TPPs.SelfAttentiveHawkesTPP import CountAwareSAHP
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONES,
    BACKBONE_LABELS,
    FROZEN_TAIL_LAMBDA,
    LOGNORMAL_VARIANT,
    MODEL_ROLES,
    MODEL_ROLE_EXPERIMENTAL,
    QUANTITY_VARIANT_ALIASES,
    SEEDS,
    SUPPORTED_BACKBONES,
    TAIL_HEAD_ONLY_VARIANT,
    TAIL_SHARED_VARIANT,
    TAIL_VARIANTS,
    VARIANT,
    validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.datasets import (
    DATASET_CONTRACTS,
    validate_dataset_runtime_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import (
    empty_accumulator,
    evaluate,
    finalize_accumulator,
    prepare_count_frame,
    right_pad_batch,
    target_outputs,
    update_accumulator,
)
from paper.scripts.count_aware_tpp_backbone.reporting import (
    summarize_breakdowns,
    write_csv,
)
from paper.scripts.count_aware_tpp_backbone.training import (
    early_stopping_exhausted,
    train_one,
)
from paper.scripts.run_intermittent_log_backbone_control import (
    HISTORY_BOUNDARIES,
    HISTORY_STRATA,
)
from paper.scripts.run_taxi_quantity_interface_ablation import (
    parse_int_tuple,
    parse_str_tuple,
    save_json,
    sha256_file,
    train_quantile_contract,
)
from simple_lab_test.search.common.experiment_utils import filter_top_series


TIME_WD_SAFETY_LIMIT = 40.0
STABLE_TIME_WD_SAFETY_LIMIT = 8.0
STABLE_TIME_INTERCEPT_LIMIT = 6.0


def derive_train_time_contract(
    frame: pl.DataFrame,
    *,
    lookback_weeks: int,
    max_seq_len: int,
    wd_safety_limit: float = TIME_WD_SAFETY_LIMIT,
) -> dict[str, Any]:
    """Derive scaling constants from train targets under the exact loader path."""
    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback_weeks,
        max_seq_len=max_seq_len,
        val_ratio=0.2,
        mode="all",
        split_col="chronological_split",
        target_splits={"train"},
    )
    target_dts = np.asarray(
        [
            max(1.0, float(dataset.dt_lists[part_index][context_end + 1]))
            for part_index, context_end in dataset.index
        ],
        dtype=np.float64,
    )
    if not target_dts.size or not np.isfinite(target_dts).all():
        raise ValueError("Train-only time targets must be nonempty and finite")
    if not math.isfinite(wd_safety_limit) or wd_safety_limit <= 0.0:
        raise ValueError("wd_safety_limit must be finite and positive")
    time_scale = float(np.quantile(target_dts, 0.50))
    target_mean = float(target_dts.mean())
    target_max = float(target_dts.max())
    time_w_max = float(wd_safety_limit) / (target_max / time_scale)
    time_initial_intercept = math.log(time_scale / target_mean)
    log_scaled_targets = np.log(target_dts / time_scale)
    return {
        "statistics_source_split": "train",
        "target_count": int(target_dts.size),
        "target_dt_min": float(target_dts.min()),
        "target_dt_mean": target_mean,
        "target_dt_p50": time_scale,
        "target_dt_p99": float(np.quantile(target_dts, 0.99)),
        "target_dt_max": target_max,
        "time_scale": time_scale,
        "wd_safety_limit": float(wd_safety_limit),
        "time_w_max": time_w_max,
        "time_initial_intercept": time_initial_intercept,
        "target_log_scaled_mean": float(log_scaled_targets.mean()),
        "target_log_scaled_std": float(log_scaled_targets.std()),
    }


def validate_scaled_time_contract(
    *,
    time_scale: float,
    time_w_max: float,
    train_time_contract: dict[str, Any],
) -> None:
    """Require runtime constants to match the exact train-target statistics."""
    expected_scale = float(train_time_contract["time_scale"])
    expected_w_max = float(train_time_contract["time_w_max"])
    if not math.isclose(
        time_scale,
        expected_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"time_scale={time_scale} does not match train-only value "
            f"{expected_scale}"
        )
    if not math.isclose(
        time_w_max,
        expected_w_max,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"time_w_max={time_w_max} does not match train-only value "
            f"{expected_w_max}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execution-role", required=True)
    parser.add_argument(
        "--dataset-contract",
        choices=sorted(DATASET_CONTRACTS),
        default="intermittent_frozen_5000",
    )
    parser.add_argument("--max-series", type=int, default=None)
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
    parser.add_argument(
        "--model-role",
        choices=MODEL_ROLES,
        default=MODEL_ROLE_EXPERIMENTAL,
        help="Frozen comparison role; experimental preserves legacy launch behavior.",
    )
    parser.add_argument("--quantity-sigma-floor", type=float, default=1e-3)
    parser.add_argument("--lambda-location-huber", type=float, default=1.0)
    parser.add_argument("--location-huber-delta", type=float, default=0.25)
    parser.add_argument("--lambda-tail", type=float, default=0.0)
    parser.add_argument("--tail-threshold", type=float, default=46.0)
    parser.add_argument("--tail-normalization-scale", type=float, default=46.0)
    parser.add_argument("--tail-clip-cap", type=float, default=187.0)
    parser.add_argument("--tail-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--time-head-mode",
        choices=TIME_HEAD_MODES,
        default=TIME_HEAD_MODE_LEGACY_CLAMPED,
    )
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument("--time-w-max", type=float, default=10.0 / 3.0)
    parser.add_argument("--time-intercept-limit", type=float, default=30.0)
    parser.add_argument(
        "--time-wd-safety-limit",
        type=float,
        default=TIME_WD_SAFETY_LIMIT,
    )
    parser.add_argument("--time-head-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--time-sigma-floor", type=float, default=1e-3)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    backbones = parse_str_tuple(args.backbones)
    seeds = parse_int_tuple(args.seeds)
    quantity_variants = normalize_quantity_variants(args.quantity_variants)
    validate_model_role_contract(
        model_role=args.model_role,
        backbones=backbones,
        quantity_variants=quantity_variants,
        time_head_mode=args.time_head_mode,
        lambda_tail=args.lambda_tail,
    )
    if any(backbone not in SUPPORTED_BACKBONES for backbone in backbones):
        raise ValueError(f"Unsupported backbone selection: {backbones}")
    if not args.allow_partial_contract:
        if set(backbones) != set(BACKBONES) or set(seeds) != set(SEEDS):
            raise ValueError("Qualified run requires all backbones and seeds 42/52/62")
    dataset_contract = DATASET_CONTRACTS[args.dataset_contract]
    if args.hidden_dim != 64:
        raise ValueError("Frozen contract requires hidden_dim=64")
    if args.max_series is not None and args.max_series < 1:
        raise ValueError("max_series must be positive")
    if args.dataset_contract == "intermittent_frozen_5000" and args.max_series is not None:
        raise ValueError("The qualified Intermittent contract does not allow max_series")
    if args.dataset_contract == "insta_market_basket":
        if not args.allow_partial_contract or args.max_series is None:
            raise ValueError("Instacart is supported only as an explicit max-series smoke")
    if args.lambda_log_qty != 1.0:
        raise ValueError("Frozen contract requires lambda_log_qty=1.0")
    if args.time_wd_safety_limit <= 0.0:
        raise ValueError("time_wd_safety_limit must be positive")
    if args.time_head_lr_multiplier <= 0.0:
        raise ValueError("time_head_lr_multiplier must be positive")
    if args.time_sigma_floor <= 0.0:
        raise ValueError("time_sigma_floor must be positive")
    if args.time_head_mode in TIME_HEAD_EXACT_MODES:
        if args.time_scale <= 0.0 or args.time_w_max <= 0.0:
            raise ValueError("Scaled-exact time constants must be positive")
        if args.time_intercept_limit <= 0.0:
            raise ValueError("time_intercept_limit must be positive")
        if args.dataset_contract == "intermittent_frozen_5000":
            if not math.isclose(args.time_scale, 3.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Intermittent scaled-exact contract requires time_scale=3")
            if args.time_head_mode == TIME_HEAD_MODE_SCALED_EXACT:
                expected = {
                    "time_w_max": 10.0 / 3.0,
                    "time_intercept_limit": 30.0,
                    "time_wd_safety_limit": TIME_WD_SAFETY_LIMIT,
                }
            else:
                expected = {
                    "time_w_max": 2.0 / 3.0,
                    "time_intercept_limit": STABLE_TIME_INTERCEPT_LIMIT,
                    "time_wd_safety_limit": STABLE_TIME_WD_SAFETY_LIMIT,
                }
            observed = {
                "time_w_max": args.time_w_max,
                "time_intercept_limit": args.time_intercept_limit,
                "time_wd_safety_limit": args.time_wd_safety_limit,
            }
            mismatches = {
                name: {"expected": expected[name], "observed": value}
                for name, value in observed.items()
                if not math.isclose(
                    value,
                    expected[name],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            }
            if mismatches:
                raise ValueError(
                    f"Intermittent scaled-exact contract mismatch: {mismatches}"
                )
    if LOGNORMAL_VARIANT in quantity_variants:
        if args.quantity_sigma_floor != 1e-3:
            raise ValueError("K=1 contract requires quantity_sigma_floor=1e-3")
        if args.lambda_location_huber != 1.0:
            raise ValueError("K=1 contract requires lambda_location_huber=1.0")
        if args.location_huber_delta != 0.25:
            raise ValueError("K=1 contract requires location_huber_delta=0.25")
    if any(variant in TAIL_VARIANTS for variant in quantity_variants):
        if not math.isclose(
            args.lambda_tail,
            FROZEN_TAIL_LAMBDA,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"Tail contract requires lambda_tail={FROZEN_TAIL_LAMBDA}"
            )
    validate_dataset_runtime_contract(
        dataset_id=args.dataset_contract,
        lookback=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        uses_tail_loss=any(
            variant in TAIL_VARIANTS for variant in quantity_variants
        ),
        tail_threshold=args.tail_threshold,
        tail_normalization_scale=args.tail_normalization_scale,
        tail_clip_cap=args.tail_clip_cap,
        tail_huber_delta=args.tail_huber_delta,
    )

    data_sha256 = sha256_file(args.data)
    manifest_sha256 = sha256_file(args.split_manifest)
    if data_sha256 != dataset_contract["data_sha256"]:
        raise ValueError(f"Unexpected fixed-split SHA-256: {data_sha256}")
    if manifest_sha256 != dataset_contract["split_manifest_sha256"]:
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
    if args.max_series is not None:
        raw_frame = filter_top_series(
            raw_frame,
            key_col="oper_part_no",
            max_series=args.max_series,
        ).sort(["oper_part_no", "seq"])
    quantity_contract = train_quantile_contract(raw_frame)
    frame = prepare_count_frame(raw_frame)
    train_time_contract = derive_train_time_contract(
        frame,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        wd_safety_limit=args.time_wd_safety_limit,
    )
    if args.time_head_mode in TIME_HEAD_EXACT_MODES:
        validate_scaled_time_contract(
            time_scale=args.time_scale,
            time_w_max=args.time_w_max,
            train_time_contract=train_time_contract,
        )
    elif args.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
        expected_scale = float(train_time_contract["time_scale"])
        if not math.isclose(
            args.time_scale,
            expected_scale,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Log-normal duration time_scale must match the train-only median"
            )
    if args.time_head_mode == TIME_HEAD_MODE_SCALED_EXACT_STABLE:
        time_initial_intercept = float(
            train_time_contract["time_initial_intercept"]
        )
        time_intercept_transform = "scaled_tanh"
    elif args.time_head_mode == TIME_HEAD_MODE_SCALED_EXACT:
        time_initial_intercept = math.log(args.time_scale)
        time_intercept_transform = "hard_clamp"
    elif args.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION:
        time_initial_intercept = 0.0
        time_intercept_transform = "not_applicable"
    else:
        time_initial_intercept = 0.0
        time_intercept_transform = "legacy_upper_clamp"
    time_initial_location = (
        float(train_time_contract["target_log_scaled_mean"])
        if args.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION
        else None
    )
    time_initial_scale = (
        float(train_time_contract["target_log_scaled_std"])
        if args.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION
        else None
    )
    if (
        args.time_head_mode == TIME_HEAD_MODE_LOGNORMAL_DURATION
        and time_initial_scale is not None
        and time_initial_scale <= args.time_sigma_floor
    ):
        raise ValueError(
            "Train-only log-duration scale must exceed time_sigma_floor"
        )
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
        "time_head": {
            "mode": args.time_head_mode,
            "time_scale": args.time_scale,
            "time_w_max": args.time_w_max,
            "time_intercept_limit": args.time_intercept_limit,
            "time_initial_intercept": time_initial_intercept,
            "time_intercept_transform": time_intercept_transform,
            "time_wd_safety_limit": args.time_wd_safety_limit,
            "time_head_lr_multiplier": args.time_head_lr_multiplier,
            "time_initial_location": time_initial_location,
            "time_initial_scale": time_initial_scale,
            "time_sigma_floor": args.time_sigma_floor,
            "statistics_source_split": "train",
            "train_time_statistics": train_time_contract,
        },
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
    split_rows = {
        str(row["chronological_split"]): int(row["len"])
        for row in raw_frame.group_by("chronological_split").agg(pl.len()).iter_rows(named=True)
    }
    contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "mark_free_count_aware_quantity_screening",
        "model_role": args.model_role,
        "dataset": args.dataset_contract,
        "dataset_time_unit": dataset_contract["time_unit"],
        "max_series": args.max_series,
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
        "time_head": {
            "mode": args.time_head_mode,
            "time_scale": args.time_scale,
            "time_w_max": args.time_w_max,
            "time_intercept_limit": args.time_intercept_limit,
            "time_initial_intercept": time_initial_intercept,
            "time_intercept_transform": time_intercept_transform,
            "time_wd_safety_limit": args.time_wd_safety_limit,
            "time_head_lr_multiplier": args.time_head_lr_multiplier,
            "time_initial_location": time_initial_location,
            "time_initial_scale": time_initial_scale,
            "time_sigma_floor": args.time_sigma_floor,
            "statistics_source_split": "train",
            "density_unit": (
                "original_delta_t_with_jacobian"
                if args.time_head_mode
                in (*TIME_HEAD_EXACT_MODES, TIME_HEAD_MODE_LOGNORMAL_DURATION)
                else "legacy_delta_t_clamped_objective"
            ),
            "wd_clamp": (
                10.0
                if args.time_head_mode == TIME_HEAD_MODE_LEGACY_CLAMPED
                else 0.0
            ),
            "train_time_statistics": train_time_contract,
        },
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
