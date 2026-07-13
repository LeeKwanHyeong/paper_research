#!/usr/bin/env python3
"""Audit Intermittent train-only contexts for raw-quantity Q0/Q1/Q2 RevIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset


DEFAULT_DATASET = PROJECT_ROOT / "sample_data/head_office/marked_target_with_split.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "search_artifacts/model_enhancement_raw_quantity_revin_audit_0713"
REQUIRED_COLUMNS = {
    "oper_part_no",
    "seq",
    "demand_qty",
    "chronological_split",
    "mark",
    "scale_residual",
}
K_CANDIDATES = (1.0, 2.0, 4.0, 8.0, 16.0)
HISTORY_BUCKETS = (
    ("1", 1, 1),
    ("2-4", 2, 4),
    ("5-8", 5, 8),
    ("9+", 9, 10_000),
)
NORMALIZATION_SCOPES = (
    ("all", 1, 10_000),
    ("n_eq_1", 1, 1),
    ("n_le_4", 1, 4),
)
REVIN_EPS = 1e-5
KST = ZoneInfo("Asia/Seoul")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--execution-server", default="unknown")
    parser.add_argument("--tmux-session", default="none")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raw_quantity_revin_audit")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_dir / "audit.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def train_scope_sha256(frame: pl.DataFrame) -> str:
    columns = [
        "oper_part_no",
        "seq",
        "demand_qty",
        "chronological_split",
        "mark",
        "scale_residual",
    ]
    row_hashes = frame.select(columns).hash_rows(seed=0).to_numpy()
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def top_sum_share(values: np.ndarray, fraction: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("top_sum_share requires values and a fraction in (0, 1].")
    count = max(1, int(math.ceil(values.size * fraction)))
    total = float(values.sum())
    return float(np.partition(values, values.size - count)[-count:].sum() / total)


def metric_rows(scope: str, metrics: dict[str, float | int | str | bool]) -> list[dict[str, object]]:
    return [{"scope": scope, "metric": key, "value": value} for key, value in metrics.items()]


def validate_source(frame: pl.DataFrame) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    null_count = int(
        frame.select(
            pl.sum_horizontal(
                [pl.col(column).is_null().cast(pl.Int64) for column in REQUIRED_COLUMNS]
            )
        )
        .to_series()
        .sum()
    )
    duplicate_key_count = int(
        frame.group_by(["oper_part_no", "seq"]).len().filter(pl.col("len") > 1)["len"].sum()
        or 0
    )
    quantity = frame["demand_qty"].cast(pl.Float64).to_numpy()
    marks = frame["mark"].cast(pl.Float64).to_numpy()
    residuals = frame["scale_residual"].cast(pl.Float64).to_numpy()
    reconstructed = np.exp2(marks + residuals)
    abs_error = np.abs(reconstructed - quantity)
    relative_error = abs_error / np.maximum(np.abs(quantity), 1.0)

    non_positive_qty_count = int(np.sum(quantity <= 0.0))
    non_finite_qty_count = int(np.sum(~np.isfinite(quantity)))
    non_train_row_count = int(frame.filter(pl.col("chronological_split") != "train").height)
    max_abs_error = float(abs_error.max(initial=0.0))
    max_relative_error = float(relative_error.max(initial=0.0))

    if null_count:
        raise ValueError(f"Required columns contain {null_count} null cells")
    if duplicate_key_count:
        raise ValueError(f"Found {duplicate_key_count} rows in duplicate part/seq keys")
    if non_positive_qty_count or non_finite_qty_count:
        raise ValueError("Demand quantity must be positive and finite")
    if max_relative_error > 1e-8:
        raise ValueError(f"Raw quantity reconstruction relative error is {max_relative_error}")
    if non_train_row_count:
        raise ValueError(f"Train-only audit decoded {non_train_row_count} non-train rows")

    return {
        "row_count": frame.height,
        "series_count": frame["oper_part_no"].n_unique(),
        "required_null_cells": null_count,
        "duplicate_part_seq_rows": duplicate_key_count,
        "non_positive_qty_rows": non_positive_qty_count,
        "non_finite_qty_rows": non_finite_qty_count,
        "decoded_non_train_rows": non_train_row_count,
        "max_raw_reconstruction_abs_error": max_abs_error,
        "max_raw_reconstruction_relative_error": max_relative_error,
        "quality_gate": "PASS",
    }


def prepare_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(["oper_part_no", "seq"]).with_columns(
        pl.col("demand_qty").cast(pl.Float64).alias("_audit_qty")
    )


def build_train_contexts(
    frame: pl.DataFrame,
    *,
    lookback_weeks: int,
    max_seq_len: int,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    max_context_len = max_seq_len - 1
    if lookback_weeks < 1 or max_context_len < 1:
        raise ValueError("lookback_weeks and max_seq_len must provide at least one context event")

    context_rows: list[dict[str, object]] = []
    series_rows: list[dict[str, object]] = []
    context_split_violations = 0
    grouped = frame.group_by("oper_part_no", maintain_order=True).agg(
        pl.col("seq").alias("seqs"),
        pl.col("chronological_split").alias("splits"),
        pl.col("_audit_qty").alias("qty_values"),
    )

    for row in grouped.iter_rows(named=True):
        part = str(row["oper_part_no"])
        seqs = np.asarray(row["seqs"], dtype=np.int64)
        splits = np.asarray(row["splits"])
        quantities = np.asarray(row["qty_values"], dtype=np.float64)
        train_values = quantities[splits == "train"]
        train_count = int(train_values.size)
        if train_count:
            half = train_count // 2
            if train_count >= 2:
                early = float(train_values[:half].mean())
                late = float(train_values[half:].mean())
                level_shift = late - early
            else:
                early = late = float(train_values[0])
                level_shift = float("nan")
            series_rows.append(
                {
                    "oper_part_no": part,
                    "train_event_count": train_count,
                    "train_mean_qty": float(train_values.mean()),
                    "train_var_qty": float(train_values.var()),
                    "train_std_qty": float(train_values.std()),
                    "train_min_qty": float(train_values.min()),
                    "train_max_qty": float(train_values.max()),
                    "early_mean_qty": early,
                    "late_mean_qty": late,
                    "late_minus_early_qty": level_shift,
                }
            )

        for context_end in range(len(seqs) - 1):
            target_idx = context_end + 1
            if splits[target_idx] != "train":
                continue
            left_seq = int(seqs[context_end]) - (lookback_weeks - 1)
            context_start = int(np.searchsorted(seqs, left_seq, side="left"))
            context_idx = np.arange(context_start, context_end + 1, dtype=np.int64)
            if context_idx.size > max_context_len:
                context_idx = context_idx[-max_context_len:]
            if np.any(splits[context_idx] != "train"):
                context_split_violations += 1

            history = quantities[context_idx]
            count = int(history.size)
            history_mean = float(history.mean())
            history_var = float(history.var())
            target_qty = float(quantities[target_idx])
            recent_minus_prior = float("nan")
            if count >= 4:
                half = count // 2
                recent_minus_prior = float(history[half:].mean() - history[:half].mean())
            relative_gap = abs(target_qty - history_mean) / max(abs(history_mean), 1.0)

            context_rows.append(
                {
                    "oper_part_no": part,
                    "target_seq": int(seqs[target_idx]),
                    "context_count": count,
                    "context_span_weeks": int(seqs[context_idx[-1]] - seqs[context_idx[0]]),
                    "history_mean_qty": history_mean,
                    "history_var_qty": history_var,
                    "history_std_qty": float(math.sqrt(max(history_var, 0.0))),
                    "history_min_qty": float(history.min()),
                    "history_max_qty": float(history.max()),
                    "history_last_qty": float(history[-1]),
                    "target_qty": target_qty,
                    "target_minus_history_mean_qty": target_qty - history_mean,
                    "abs_target_minus_history_mean_qty": abs(target_qty - history_mean),
                    "relative_target_history_mean_gap": relative_gap,
                    "target_minus_last_qty": target_qty - float(history[-1]),
                    "abs_target_minus_last_qty": abs(target_qty - float(history[-1])),
                    "target_outside_history_range": bool(
                        target_qty < float(history.min()) or target_qty > float(history.max())
                    ),
                    "target_above_history_max": bool(target_qty > float(history.max())),
                    "target_below_history_min": bool(target_qty < float(history.min())),
                    "recent_minus_prior_mean_qty": recent_minus_prior,
                }
            )

    if context_split_violations:
        raise ValueError(
            f"Found {context_split_violations} train targets with non-train context events"
        )
    return (
        pl.DataFrame(context_rows),
        pl.DataFrame(series_rows),
        {"context_split_violations": context_split_violations},
    )


def validate_loader_contract(
    frame: pl.DataFrame,
    contexts: pl.DataFrame,
    *,
    lookback_weeks: int,
    max_seq_len: int,
) -> dict[str, object]:
    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback_weeks,
        max_seq_len=max_seq_len,
        mode="all",
        pad_id=int(frame["mark"].max()) + 1,
        target_splits={"train"},
    )
    loader_counts: dict[int, int] = {}
    max_context_len = max_seq_len - 1
    for part_idx, context_end in dataset.index:
        seqs = np.asarray(dataset.seq_lists[part_idx], dtype=np.int64)
        left_seq = int(seqs[context_end]) - (lookback_weeks - 1)
        context_start = int(np.searchsorted(seqs, left_seq, side="left"))
        context_count = min(context_end - context_start + 1, max_context_len)
        loader_counts[context_count] = loader_counts.get(context_count, 0) + 1

    audit_counts = {
        int(row["context_count"]): int(row["len"])
        for row in contexts.group_by("context_count").len().iter_rows(named=True)
    }
    target_count_match = len(dataset) == contexts.height
    distribution_match = loader_counts == audit_counts
    if not target_count_match or not distribution_match:
        raise ValueError(
            "Audit contexts do not match RMTPPWeekLookbackDataset: "
            f"target_count_match={target_count_match}, distribution_match={distribution_match}"
        )
    return {
        "dataloader_target_count": len(dataset),
        "dataloader_target_count_match": target_count_match,
        "dataloader_context_distribution_match": distribution_match,
    }


def summarize_global(
    frame: pl.DataFrame,
    contexts: pl.DataFrame,
    series_stats: pl.DataFrame,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    quantity = frame["_audit_qty"].to_numpy()
    mean = float(quantity.mean())
    variance = float(quantity.var())
    std = float(math.sqrt(variance))
    counts = series_stats["train_event_count"].to_numpy().astype(np.float64)
    means = series_stats["train_mean_qty"].to_numpy()
    variances = series_stats["train_var_qty"].to_numpy()
    within_variance = float(np.sum(counts * variances) / np.sum(counts))
    between_variance = float(np.sum(counts * (means - mean) ** 2) / np.sum(counts))

    global_stats: dict[str, float | int] = {
        "train_event_count": int(quantity.size),
        "train_series_count": int(series_stats.height),
        "train_mean_raw_qty": mean,
        "train_var_raw_qty": variance,
        "train_std_raw_qty": std,
        "train_min_raw_qty": float(quantity.min()),
        "train_median_raw_qty": quantile(quantity, 0.50),
        "train_p90_raw_qty": quantile(quantity, 0.90),
        "train_p95_raw_qty": quantile(quantity, 0.95),
        "train_p99_raw_qty": quantile(quantity, 0.99),
        "train_p999_raw_qty": quantile(quantity, 0.999),
        "train_max_raw_qty": float(quantity.max()),
        "mean_to_median_ratio": mean / max(quantile(quantity, 0.50), 1e-12),
        "p99_to_median_ratio": quantile(quantity, 0.99) / max(quantile(quantity, 0.50), 1e-12),
        "max_to_p99_ratio": float(quantity.max()) / max(quantile(quantity, 0.99), 1e-12),
        "top_1pct_quantity_sum_share": top_sum_share(quantity, 0.01),
        "top_0p1pct_quantity_sum_share": top_sum_share(quantity, 0.001),
        "within_series_raw_variance": within_variance,
        "between_series_raw_variance": between_variance,
        "between_series_raw_variance_share": between_variance / max(variance, 1e-12),
        "variance_decomposition_abs_error": abs(variance - within_variance - between_variance),
    }

    context_count = contexts["context_count"].to_numpy()
    history_std = contexts["history_std_qty"].to_numpy()
    series_count = series_stats["train_event_count"].to_numpy()
    history_summary: dict[str, float | int] = {
        "train_target_count": int(contexts.height),
        "context_count_min": int(context_count.min()),
        "context_count_p25": quantile(context_count, 0.25),
        "context_count_p50": quantile(context_count, 0.50),
        "context_count_p75": quantile(context_count, 0.75),
        "context_count_p90": quantile(context_count, 0.90),
        "context_count_p95": quantile(context_count, 0.95),
        "context_count_max": int(context_count.max()),
        "context_count_eq_1_share": float(np.mean(context_count == 1)),
        "context_count_le_2_share": float(np.mean(context_count <= 2)),
        "context_count_le_4_share": float(np.mean(context_count <= 4)),
        "context_count_le_8_share": float(np.mean(context_count <= 8)),
        "zero_variance_context_share": float(np.mean(history_std <= 1e-12)),
        "history_std_raw_p50": quantile(history_std, 0.50),
        "history_std_raw_p95": quantile(history_std, 0.95),
        "history_std_raw_p99": quantile(history_std, 0.99),
        "series_train_count_eq_1_share": float(np.mean(series_count == 1)),
        "series_train_count_le_4_share": float(np.mean(series_count <= 4)),
        "zero_variance_series_share": float(
            np.mean(series_stats["train_std_qty"].to_numpy() <= 1e-12)
        ),
    }
    return global_stats, history_summary


def history_length_distribution(contexts: pl.DataFrame) -> pl.DataFrame:
    return (
        contexts.group_by("context_count")
        .len(name="sample_count")
        .sort("context_count")
        .with_columns((pl.col("sample_count") / contexts.height).alias("sample_share"))
        .with_columns(pl.col("sample_share").cum_sum().alias("cumulative_share"))
    )


def variance_by_bucket(contexts: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    counts = contexts["context_count"].to_numpy()
    stds = contexts["history_std_qty"].to_numpy()
    target_deviation = contexts["abs_target_minus_history_mean_qty"].to_numpy()
    relative_deviation = contexts["relative_target_history_mean_gap"].to_numpy()
    for label, low, high in HISTORY_BUCKETS:
        mask = (counts >= low) & (counts <= high)
        bucket_std = stds[mask]
        rows.append(
            {
                "history_bucket": label,
                "sample_count": int(mask.sum()),
                "sample_share": float(mask.mean()),
                "zero_variance_share": float(np.mean(bucket_std <= 1e-12)),
                "std_raw_p25": quantile(bucket_std, 0.25),
                "std_raw_p50": quantile(bucket_std, 0.50),
                "std_raw_p75": quantile(bucket_std, 0.75),
                "std_raw_p95": quantile(bucket_std, 0.95),
                "target_abs_raw_deviation_p50": quantile(target_deviation[mask], 0.50),
                "target_abs_raw_deviation_p95": quantile(target_deviation[mask], 0.95),
                "target_relative_deviation_p50": quantile(relative_deviation[mask], 0.50),
                "target_relative_deviation_p95": quantile(relative_deviation[mask], 0.95),
            }
        )
    return pl.DataFrame(rows)


def summarize_level_shift(
    contexts: pl.DataFrame,
    series_stats: pl.DataFrame,
    global_mean: float,
) -> dict[str, float | int]:
    local_mean_gap = np.abs(contexts["history_mean_qty"].to_numpy() - global_mean)
    target_history_gap = contexts["abs_target_minus_history_mean_qty"].to_numpy()
    target_relative_gap = contexts["relative_target_history_mean_gap"].to_numpy()
    target_last_gap = contexts["abs_target_minus_last_qty"].to_numpy()
    window_shift = contexts["recent_minus_prior_mean_qty"].to_numpy()
    series_shift = series_stats["late_minus_early_qty"].to_numpy()
    return {
        "window_mean_abs_gap_vs_global_raw_p50": quantile(local_mean_gap, 0.50),
        "window_mean_abs_gap_vs_global_raw_p95": quantile(local_mean_gap, 0.95),
        "target_abs_gap_vs_history_mean_raw_p50": quantile(target_history_gap, 0.50),
        "target_abs_gap_vs_history_mean_raw_p95": quantile(target_history_gap, 0.95),
        "target_relative_gap_vs_history_mean_p50": quantile(target_relative_gap, 0.50),
        "target_relative_gap_vs_history_mean_p95": quantile(target_relative_gap, 0.95),
        "target_abs_gap_vs_last_raw_p50": quantile(target_last_gap, 0.50),
        "target_abs_gap_vs_last_raw_p95": quantile(target_last_gap, 0.95),
        "target_outside_history_range_share": float(
            contexts["target_outside_history_range"].mean()
        ),
        "target_above_history_max_share": float(contexts["target_above_history_max"].mean()),
        "target_below_history_min_share": float(contexts["target_below_history_min"].mean()),
        "window_half_shift_available_count": int(np.isfinite(window_shift).sum()),
        "window_half_abs_shift_raw_p50": quantile(np.abs(window_shift), 0.50),
        "window_half_abs_shift_raw_p95": quantile(np.abs(window_shift), 0.95),
        "series_half_shift_available_count": int(np.isfinite(series_shift).sum()),
        "series_half_abs_shift_raw_p50": quantile(np.abs(series_shift), 0.50),
        "series_half_abs_shift_raw_p95": quantile(np.abs(series_shift), 0.95),
    }


def normalization_summary_row(
    *,
    variant: str,
    scope: str,
    mask: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    target_norm: np.ndarray,
    global_std: float,
) -> dict[str, object]:
    scoped_scale = scale[mask]
    scoped_center = center[mask]
    scoped_target = np.abs(target_norm[mask])
    return {
        "variant": variant,
        "scope": scope,
        "sample_count": int(mask.sum()),
        "center_p50": quantile(scoped_center, 0.50),
        "scale_min": float(scoped_scale.min(initial=float("inf"))),
        "scale_p01": quantile(scoped_scale, 0.01),
        "scale_p50": quantile(scoped_scale, 0.50),
        "scale_p95": quantile(scoped_scale, 0.95),
        "scale_p50_to_global_std": quantile(scoped_scale, 0.50) / global_std,
        "target_abs_norm_p50": quantile(scoped_target, 0.50),
        "target_abs_norm_p95": quantile(scoped_target, 0.95),
        "target_abs_norm_p99": quantile(scoped_target, 0.99),
        "target_abs_norm_max": float(scoped_target.max(initial=0.0)),
        "target_abs_norm_gt_3_share": float(np.mean(scoped_target > 3.0)),
        "target_abs_norm_gt_5_share": float(np.mean(scoped_target > 5.0)),
        "finite_share": float(
            np.mean(
                np.isfinite(scoped_scale)
                & np.isfinite(scoped_center)
                & np.isfinite(scoped_target)
            )
        ),
    }


def q1_statistics(
    contexts: pl.DataFrame,
    *,
    revin_eps: float = REVIN_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = contexts["history_mean_qty"].to_numpy()
    variance = contexts["history_var_qty"].to_numpy()
    target = contexts["target_qty"].to_numpy()
    scale = np.sqrt(np.maximum(variance, 0.0) + revin_eps)
    return center, scale, (target - center) / scale


def q2_statistics(
    contexts: pl.DataFrame,
    *,
    global_mean: float,
    global_var: float,
    sigma_floor: float,
    shrinkage_k: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = contexts["context_count"].to_numpy().astype(np.float64)
    history_mean = contexts["history_mean_qty"].to_numpy()
    history_var = contexts["history_var_qty"].to_numpy()
    target = contexts["target_qty"].to_numpy()
    alpha = counts / (counts + shrinkage_k)
    center = alpha * history_mean + (1.0 - alpha) * global_mean
    second_moment = alpha * (history_var + history_mean**2) + (1.0 - alpha) * (
        global_var + global_mean**2
    )
    variance = np.maximum(second_moment - center**2, sigma_floor**2)
    scale = np.sqrt(variance)
    return center, scale, (target - center) / scale, alpha


def reference_normalization_summary(
    contexts: pl.DataFrame,
    *,
    global_mean: float,
    global_std: float,
    revin_eps: float = REVIN_EPS,
) -> pl.DataFrame:
    counts = contexts["context_count"].to_numpy()
    target = contexts["target_qty"].to_numpy()
    q0_center = np.full(contexts.height, global_mean, dtype=np.float64)
    q0_scale = np.full(contexts.height, global_std, dtype=np.float64)
    q0_target_norm = (target - q0_center) / q0_scale
    q1_center, q1_scale, q1_target_norm = q1_statistics(contexts, revin_eps=revin_eps)
    rows: list[dict[str, object]] = []
    for scope, low, high in NORMALIZATION_SCOPES:
        mask = (counts >= low) & (counts <= high)
        rows.append(
            normalization_summary_row(
                variant="Q0_global_raw",
                scope=scope,
                mask=mask,
                center=q0_center,
                scale=q0_scale,
                target_norm=q0_target_norm,
                global_std=global_std,
            )
        )
        rows.append(
            normalization_summary_row(
                variant="Q1_causal_revin_raw",
                scope=scope,
                mask=mask,
                center=q1_center,
                scale=q1_scale,
                target_norm=q1_target_norm,
                global_std=global_std,
            )
        )
    return pl.DataFrame(rows)


def shrinkage_candidates(
    contexts: pl.DataFrame,
    *,
    global_mean: float,
    global_var: float,
    global_std: float,
    sigma_floor: float,
    q0_reference: dict[str, object],
) -> pl.DataFrame:
    counts = contexts["context_count"].to_numpy()
    rows: list[dict[str, object]] = []
    for shrinkage_k in K_CANDIDATES:
        center, scale, target_norm, alpha = q2_statistics(
            contexts,
            global_mean=global_mean,
            global_var=global_var,
            sigma_floor=sigma_floor,
            shrinkage_k=shrinkage_k,
        )
        abs_target = np.abs(target_norm)
        one_event = counts == 1
        short = counts <= 4
        finite_share = float(
            np.mean(np.isfinite(center) & np.isfinite(scale) & np.isfinite(target_norm))
        )
        one_event_scale_p50 = quantile(scale[one_event], 0.50)
        alpha_p50 = quantile(alpha, 0.50)
        target_p99 = quantile(abs_target, 0.99)
        gt3_share = float(np.mean(abs_target > 3.0))
        gates = {
            "gate_all_finite": math.isclose(finite_share, 1.0, rel_tol=0.0, abs_tol=0.0),
            "gate_one_event_scale": one_event_scale_p50 >= 0.50 * global_std,
            "gate_alpha_p50": alpha_p50 >= 0.25,
            "gate_target_p99_vs_q0": target_p99
            <= float(q0_reference["target_abs_norm_p99"]),
            "gate_target_gt3_vs_q0": gt3_share
            <= float(q0_reference["target_abs_norm_gt_3_share"]),
        }
        rows.append(
            {
                "shrinkage_k": shrinkage_k,
                "alpha_p50": alpha_p50,
                "alpha_p95": quantile(alpha, 0.95),
                "scale_min": float(scale.min()),
                "scale_p01": quantile(scale, 0.01),
                "scale_p50": quantile(scale, 0.50),
                "scale_p95": quantile(scale, 0.95),
                "scale_max": float(scale.max()),
                "one_event_scale_p50": one_event_scale_p50,
                "one_event_scale_to_global_std": one_event_scale_p50 / global_std,
                "target_abs_norm_p50": quantile(abs_target, 0.50),
                "target_abs_norm_p95": quantile(abs_target, 0.95),
                "target_abs_norm_p99": target_p99,
                "target_abs_norm_max": float(abs_target.max()),
                "target_abs_norm_gt_3_share": gt3_share,
                "target_abs_norm_gt_5_share": float(np.mean(abs_target > 5.0)),
                "short_target_abs_norm_p95": quantile(abs_target[short], 0.95),
                "short_target_abs_norm_p99": quantile(abs_target[short], 0.99),
                "short_target_abs_norm_gt_3_share": float(np.mean(abs_target[short] > 3.0)),
                "finite_share": finite_share,
                "q0_target_abs_norm_p99": float(q0_reference["target_abs_norm_p99"]),
                "q0_target_abs_norm_gt_3_share": float(
                    q0_reference["target_abs_norm_gt_3_share"]
                ),
                **gates,
                "eligible": all(gates.values()),
            }
        )
    return pl.DataFrame(rows).sort("shrinkage_k")


def choose_q2_constants(
    candidates: pl.DataFrame,
    *,
    global_mean: float,
    global_var: float,
    global_std: float,
    sigma_floor: float,
    revin_eps: float = REVIN_EPS,
) -> dict[str, object]:
    eligible = candidates.filter(pl.col("eligible")).sort(
        ["target_abs_norm_p99", "shrinkage_k"]
    )
    base = {
        "status": "frozen" if not eligible.is_empty() else "blocked",
        "selection_scope": "fixed_split_train_only",
        "magnitude_domain": "raw_qty",
        "revin_eps": revin_eps,
        "sigma_floor_raw": sigma_floor,
        "global_mean_raw": global_mean,
        "global_var_raw": global_var,
        "global_std_raw": global_std,
        "selection_rule": {
            "k_candidates": list(K_CANDIDATES),
            "finite_share_required": 1.0,
            "one_event_scale_min_global_std_ratio": 0.50,
            "alpha_p50_min": 0.25,
            "target_abs_norm_p99_max": "Q0 raw/global p99",
            "target_abs_norm_gt_3_share_max": "Q0 raw/global share",
            "rank": "lowest target_abs_norm_p99, then smallest k",
        },
    }
    if eligible.is_empty():
        return {
            **base,
            "shrinkage_k": None,
            "decision": "block_q2_no_train_only_eligible_k",
        }
    selected = eligible.row(0, named=True)
    return {
        **base,
        "shrinkage_k": float(selected["shrinkage_k"]),
        "selected_candidate": {
            key: value
            for key, value in selected.items()
            if isinstance(value, (str, bool, int, float)) or value is None
        },
        "decision": "freeze_q2_constants_for_implementation",
    }


def plot_raw_tail(quantity: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bins = np.logspace(0.0, math.log10(max(float(quantity.max()), 1.0)), 50)
    ax.hist(quantity, bins=bins, color="#B9D6F2", edgecolor="#263238", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Intermittent train raw-quantity tail")
    ax.set_xlabel("Raw demand quantity (log scale)")
    ax.set_ylabel("Event count (log scale)")
    ax.grid(axis="both", color="#D7DCE2", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_variance_by_bucket(contexts: pl.DataFrame, output_path: Path) -> None:
    counts = contexts["context_count"].to_numpy()
    stds = contexts["history_std_qty"].to_numpy()
    values = []
    labels = []
    for label, low, high in HISTORY_BUCKETS:
        mask = (counts >= low) & (counts <= high)
        values.append(stds[mask])
        labels.append(label)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    boxes = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for box in boxes["boxes"]:
        box.set_facecolor("#B9D6F2")
        box.set_edgecolor("#263238")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Raw history standard deviation by context length")
    ax.set_xlabel("History count bucket")
    ax.set_ylabel("Population standard deviation (symlog)")
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_normalization_stability(
    candidates: pl.DataFrame,
    *,
    q0_p99: float,
    selected_k: float | None,
    output_path: Path,
) -> None:
    k = candidates["shrinkage_k"].to_numpy()
    p99 = candidates["target_abs_norm_p99"].to_numpy()
    one_event_ratio = candidates["one_event_scale_to_global_std"].to_numpy()
    fig, left = plt.subplots(figsize=(10, 5.8))
    right = left.twinx()
    left.plot(k, p99, marker="o", color="#2563EB", linewidth=2, label="Q2 target |u| p99")
    left.axhline(q0_p99, color="#6B7280", linestyle=":", linewidth=1.5, label="Q0 p99")
    right.plot(
        k,
        one_event_ratio,
        marker="s",
        color="#D97706",
        linewidth=2,
        label="One-event scale / global std",
    )
    if selected_k is not None:
        left.axvline(
            selected_k,
            color="#263238",
            linestyle="--",
            linewidth=1.3,
            label=f"Selected k={selected_k:g}",
        )
    left.set_xscale("log", base=2)
    left.set_xticks(k, labels=[f"{value:g}" for value in k])
    left.set_title("Raw Q2 shrinkage stability across train-only k")
    left.set_xlabel("shrinkage_k")
    left.set_ylabel("Target absolute normalized quantity p99", color="#2563EB")
    right.set_ylabel("One-event scale / global std", color="#D97706")
    left.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    lines = left.get_lines() + right.get_lines()
    left.legend(lines, [line.get_label() for line in lines], frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    *,
    source_path: Path,
    source_quality: dict[str, object],
    global_stats: dict[str, float | int],
    history_summary: dict[str, float | int],
    level_shift: dict[str, float | int],
    references: pl.DataFrame,
    candidates: pl.DataFrame,
    constants: dict[str, object],
    lookback_weeks: int,
    max_seq_len: int,
) -> str:
    q0 = references.filter(
        (pl.col("variant") == "Q0_global_raw") & (pl.col("scope") == "all")
    ).row(0, named=True)
    q1 = references.filter(
        (pl.col("variant") == "Q1_causal_revin_raw") & (pl.col("scope") == "all")
    ).row(0, named=True)
    selected = None
    if constants["status"] == "frozen":
        selected = candidates.filter(
            pl.col("shrinkage_k") == constants["shrinkage_k"]
        ).row(0, named=True)

    decision = (
        f"Freeze Q2 `shrinkage_k={float(constants['shrinkage_k']):g}` and "
        f"`sigma_floor_raw={float(constants['sigma_floor_raw']):.8f}` for implementation."
        if selected is not None
        else "No Q2 k passed the train-only stability gate; block Q2 implementation."
    )
    lines = [
        "# Intermittent Train-Only Raw-Quantity RevIN Audit",
        "",
        "Status: `completed`  ",
        "Scope: fixed-split train events and exact weekly train-target contexts only  ",
        f"Window: lookback `{lookback_weeks}`, max sequence length `{max_seq_len}` including target",
        "",
        "## Decision",
        "",
        decision,
        "",
        f"Q0 raw/global target |u| p99 is `{float(q0['target_abs_norm_p99']):.4f}`. "
        f"Q1 masked RevIN target |u| p99 is `{float(q1['target_abs_norm_p99']):.4f}` "
        f"with scale p01 `{float(q1['scale_p01']):.6f}`.",
        "",
        "This audit freezes normalization constants only. It does not establish model accuracy or a RevIN benefit.",
        "",
        "## Source And Data Quality",
        "",
        f"- Source: `{source_path}`",
        f"- Train rows/series: `{int(source_quality['row_count']):,}` / `{int(source_quality['series_count']):,}`",
        f"- Train events/targets: `{int(global_stats['train_event_count']):,}` / `{int(history_summary['train_target_count']):,}`",
        "- Required nulls, duplicate keys, non-positive quantity, and decoded non-train rows: `0`",
        f"- Maximum factorized raw reconstruction relative error: `{float(source_quality['max_raw_reconstruction_relative_error']):.3e}`",
        "- Exact RMTPPWeekLookbackDataset target-count and context-distribution gate: `PASS`",
        "- Held-out test read: `false`",
        "",
        "## Raw Quantity Tail",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Mean / median | `{float(global_stats['train_mean_raw_qty']):.4f}` / `{float(global_stats['train_median_raw_qty']):.4f}` |",
        f"| p95 / p99 / p99.9 | `{float(global_stats['train_p95_raw_qty']):.4f}` / `{float(global_stats['train_p99_raw_qty']):.4f}` / `{float(global_stats['train_p999_raw_qty']):.4f}` |",
        f"| Maximum | `{float(global_stats['train_max_raw_qty']):.4f}` |",
        f"| Global population std | `{float(global_stats['train_std_raw_qty']):.4f}` |",
        f"| Top 1% / 0.1% quantity sum share | `{100*float(global_stats['top_1pct_quantity_sum_share']):.2f}%` / `{100*float(global_stats['top_0p1pct_quantity_sum_share']):.2f}%` |",
        f"| Between-series raw variance share | `{100*float(global_stats['between_series_raw_variance_share']):.2f}%` |",
        "",
        "## Context And Shift",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Context p50 / p95 / max | `{float(history_summary['context_count_p50']):.0f}` / `{float(history_summary['context_count_p95']):.0f}` / `{int(history_summary['context_count_max'])}` |",
        f"| One-event / n<=4 share | `{100*float(history_summary['context_count_eq_1_share']):.2f}%` / `{100*float(history_summary['context_count_le_4_share']):.2f}%` |",
        f"| Zero-variance context / series | `{100*float(history_summary['zero_variance_context_share']):.2f}%` / `{100*float(history_summary['zero_variance_series_share']):.2f}%` |",
        f"| Raw history std p50 / p95 / p99 | `{float(history_summary['history_std_raw_p50']):.4f}` / `{float(history_summary['history_std_raw_p95']):.4f}` / `{float(history_summary['history_std_raw_p99']):.4f}` |",
        f"| Target outside history range | `{100*float(level_shift['target_outside_history_range_share']):.2f}%` |",
        f"| Relative target/history gap p50 / p95 | `{float(level_shift['target_relative_gap_vs_history_mean_p50']):.4f}` / `{float(level_shift['target_relative_gap_vs_history_mean_p95']):.4f}` |",
        "",
        "## Q0/Q1 Reference",
        "",
        "| Variant | Scope | Scale p01 | Scale p50 | Target |u| p95 | Target |u| p99 | >3 share |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in references.iter_rows(named=True):
        lines.append(
            f"| {row['variant']} | {row['scope']} | {float(row['scale_p01']):.6f} | "
            f"{float(row['scale_p50']):.6f} | {float(row['target_abs_norm_p95']):.4f} | "
            f"{float(row['target_abs_norm_p99']):.4f} | "
            f"{100*float(row['target_abs_norm_gt_3_share']):.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Q2 Candidate Gate",
            "",
            "| k | Eligible | alpha p50 | one-event scale/global | target |u| p99 | short p99 | >3 share |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in candidates.iter_rows(named=True):
        marker = " (selected)" if selected is not None and row["shrinkage_k"] == selected["shrinkage_k"] else ""
        lines.append(
            f"| {float(row['shrinkage_k']):g}{marker} | {row['eligible']} | "
            f"{float(row['alpha_p50']):.4f} | {float(row['one_event_scale_to_global_std']):.4f} | "
            f"{float(row['target_abs_norm_p99']):.4f} | {float(row['short_target_abs_norm_p99']):.4f} | "
            f"{100*float(row['target_abs_norm_gt_3_share']):.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Frozen Constants",
            "",
            f"- `status={constants['status']}`",
            f"- `shrinkage_k={constants['shrinkage_k']}`",
            f"- `sigma_floor_raw={float(constants['sigma_floor_raw']):.8f}`",
            f"- `revin_eps={float(constants['revin_eps']):.8f}`",
            f"- `global_mean_raw={float(constants['global_mean_raw']):.8f}`",
            f"- `global_var_raw={float(constants['global_var_raw']):.8f}`",
            f"- `global_std_raw={float(constants['global_std_raw']):.8f}`",
            "- All constants were selected from fixed-split train rows and train-target contexts only.",
            "",
            "## Interpretation",
            "",
            "- Q0 is the raw-domain control and is not RevIN or a prerequisite for Q1/Q2.",
            "- Q1 measures unmodified masked mean/std behavior in one-event and constant contexts.",
            "- Q2 is eligible only when its normalized-target tail is no worse than Q0 and one-event scale remains material.",
            "- Raw tail concentration and normalized-target stability are feasibility evidence, not predictive-performance evidence.",
            "",
            "## Next",
            "",
            "Implement `direct_raw_qty` and matched Q0/Q1/Q2 normalization only if Q2 constants are frozen.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    plot_dir = output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    started_at = datetime.now(KST)
    dataset_path = args.dataset.resolve()
    logger.info(
        "Starting train-only raw-quantity RevIN audit | server=%s tmux=%s dataset=%s",
        args.execution_server,
        args.tmux_session,
        dataset_path,
    )

    raw = pl.scan_parquet(dataset_path).filter(
        pl.col("chronological_split") == "train"
    ).collect()
    source_quality = validate_source(raw)
    frame = prepare_frame(raw)
    contexts, series_stats, context_validations = build_train_contexts(
        frame,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
    )
    context_validations.update(
        validate_loader_contract(
            frame,
            contexts,
            lookback_weeks=args.lookback_weeks,
            max_seq_len=args.max_seq_len,
        )
    )
    global_stats, history_summary = summarize_global(frame, contexts, series_stats)
    global_mean = float(global_stats["train_mean_raw_qty"])
    global_var = float(global_stats["train_var_raw_qty"])
    global_std = float(global_stats["train_std_raw_qty"])
    if not math.isfinite(global_std) or global_std <= 0.0:
        raise ValueError("Train raw quantity standard deviation must be finite and positive")
    sigma_floor = max(global_std * 1e-3, 1e-4)
    level_shift = summarize_level_shift(contexts, series_stats, global_mean)
    references = reference_normalization_summary(
        contexts,
        global_mean=global_mean,
        global_std=global_std,
    )
    q0_reference = references.filter(
        (pl.col("variant") == "Q0_global_raw") & (pl.col("scope") == "all")
    ).row(0, named=True)
    candidates = shrinkage_candidates(
        contexts,
        global_mean=global_mean,
        global_var=global_var,
        global_std=global_std,
        sigma_floor=sigma_floor,
        q0_reference=q0_reference,
    )
    constants = choose_q2_constants(
        candidates,
        global_mean=global_mean,
        global_var=global_var,
        global_std=global_std,
        sigma_floor=sigma_floor,
    )

    quality_rows = metric_rows("source", source_quality)
    quality_rows.extend(metric_rows("context", context_validations))
    pl.DataFrame(quality_rows).write_csv(data_dir / "data_quality_summary.csv")
    pl.DataFrame(metric_rows("train_global_raw", global_stats)).write_csv(
        data_dir / "global_raw_statistics.csv"
    )
    pl.DataFrame(metric_rows("history", history_summary)).write_csv(
        data_dir / "history_length_summary.csv"
    )
    history_length_distribution(contexts).write_csv(data_dir / "history_length_distribution.csv")
    variance_by_bucket(contexts).write_csv(data_dir / "variance_by_history_bucket.csv")
    pl.DataFrame(metric_rows("level_shift", level_shift)).write_csv(
        data_dir / "level_shift_summary.csv"
    )
    references.write_csv(data_dir / "normalization_reference_summary.csv")
    candidates.write_csv(data_dir / "shrinkage_candidate_summary.csv")
    contexts.write_parquet(data_dir / "train_target_context_statistics.parquet")
    series_stats.write_csv(data_dir / "train_series_statistics.csv")
    (data_dir / "frozen_q2_constants.json").write_text(
        json.dumps(constants, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    plot_raw_tail(frame["_audit_qty"].to_numpy(), plot_dir / "raw_quantity_tail_distribution.png")
    plot_variance_by_bucket(contexts, plot_dir / "history_raw_variance_by_length.png")
    plot_normalization_stability(
        candidates,
        q0_p99=float(q0_reference["target_abs_norm_p99"]),
        selected_k=(
            float(constants["shrinkage_k"]) if constants["shrinkage_k"] is not None else None
        ),
        output_path=plot_dir / "raw_normalization_stability.png",
    )

    report = build_report(
        source_path=dataset_path,
        source_quality=source_quality,
        global_stats=global_stats,
        history_summary=history_summary,
        level_shift=level_shift,
        references=references,
        candidates=candidates,
        constants=constants,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    summary = {
        "status": "completed",
        "quality_gate": source_quality["quality_gate"],
        "loader_contract_gate": "PASS",
        "train_target_count": history_summary["train_target_count"],
        "context_count_le_4_share": history_summary["context_count_le_4_share"],
        "zero_variance_context_share": history_summary["zero_variance_context_share"],
        "raw_tail": {
            "mean": global_stats["train_mean_raw_qty"],
            "median": global_stats["train_median_raw_qty"],
            "p95": global_stats["train_p95_raw_qty"],
            "p99": global_stats["train_p99_raw_qty"],
            "max": global_stats["train_max_raw_qty"],
            "top_1pct_sum_share": global_stats["top_1pct_quantity_sum_share"],
        },
        "q0_reference": q0_reference,
        "q1_reference": references.filter(
            (pl.col("variant") == "Q1_causal_revin_raw") & (pl.col("scope") == "all")
        ).row(0, named=True),
        "frozen_q2_constants": constants,
        "decision": constants["decision"],
        "held_out_test_read": False,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    finished_at = datetime.now(KST)
    manifest = {
        "status": "completed",
        "analysis": "intermittent_train_only_raw_quantity_revin_audit",
        "started_at_kst": started_at.isoformat(),
        "finished_at_kst": finished_at.isoformat(),
        "execution_server": args.execution_server,
        "execution_host": platform.node(),
        "tmux_session": args.tmux_session,
        "python": sys.executable,
        "dataset_path": str(dataset_path),
        "decoded_train_scope_sha256": train_scope_sha256(raw),
        "output_dir": str(output_dir),
        "split_scope": "train_only",
        "held_out_test_read": False,
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "max_context_len": args.max_seq_len - 1,
        "target_scope": "RMTPPWeekLookbackDataset-compatible train targets",
        "magnitude_domain": "raw_qty",
        "q0_role": "raw_global_control_not_revin_not_prerequisite",
        "q1_role": "causal_masked_revin_diagnostic",
        "q2_role": "causal_moment_shrinkage_primary_candidate",
        "k_candidates": list(K_CANDIDATES),
        "frozen_q2_constants": constants,
        "artifact_order": [
            "audit_manifest.json",
            "logs/audit.log",
            "audit_summary.json",
            "data/data_quality_summary.csv",
            "data/global_raw_statistics.csv",
            "data/history_length_summary.csv",
            "data/normalization_reference_summary.csv",
            "data/shrinkage_candidate_summary.csv",
            "data/train_target_context_statistics.parquet",
            "report.md",
            "plots/*.png",
        ],
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Completed raw audit | train_targets=%d q2_status=%s selected_k=%s sigma_floor_raw=%.8f",
        int(history_summary["train_target_count"]),
        constants["status"],
        constants["shrinkage_k"],
        float(constants["sigma_floor_raw"]),
    )


def main() -> None:
    write_outputs(parse_args())


if __name__ == "__main__":
    main()
