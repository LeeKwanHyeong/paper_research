#!/usr/bin/env python3
"""Audit Intermittent train-only contexts for direct magnitude RevIN design."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
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
DEFAULT_OUTPUT = PROJECT_ROOT / "search_artifacts/model_enhancement_magnitude_revin_audit_0713"
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
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9+", 9, 10_000),
)
KST = ZoneInfo("Asia/Seoul")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--max-seq-len", type=int, default=16)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("magnitude_revin_audit")
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


def metric_rows(scope: str, metrics: dict[str, float | int | str]) -> list[dict[str, object]]:
    return [{"scope": scope, "metric": key, "value": value} for key, value in metrics.items()]


def validate_source(frame: pl.DataFrame) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    null_count = int(
        frame.select(pl.sum_horizontal([pl.col(column).is_null().cast(pl.Int64) for column in REQUIRED_COLUMNS]))
        .to_series()
        .sum()
    )
    duplicate_key_count = int(
        frame.group_by(["oper_part_no", "seq"]).len().filter(pl.col("len") > 1)["len"].sum()
        or 0
    )
    non_positive_qty_count = int(frame.filter(pl.col("demand_qty") <= 0).height)
    non_finite_qty_count = int(frame.filter(~pl.col("demand_qty").is_finite()).height)

    non_train_row_count = int(frame.filter(pl.col("chronological_split") != "train").height)
    checked = frame.with_columns(
        pl.col("demand_qty").log(2.0).alias("_z_log2"),
        (pl.col("mark").cast(pl.Float64) + pl.col("scale_residual")).alias("_z_factorized"),
    ).with_columns(
        (pl.col("_z_log2") - pl.col("_z_factorized")).abs().alias("_z_abs_error"),
    )
    max_reconstruction_error = float(checked["_z_abs_error"].max())

    if null_count:
        raise ValueError(f"Required columns contain {null_count} null cells")
    if duplicate_key_count:
        raise ValueError(f"Found {duplicate_key_count} rows in duplicate part/seq keys")
    if non_positive_qty_count or non_finite_qty_count:
        raise ValueError("Demand quantity must be positive and finite")
    if max_reconstruction_error > 1e-8:
        raise ValueError(f"log2 quantity reconstruction error is {max_reconstruction_error}")
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
        "max_log2_reconstruction_abs_error": max_reconstruction_error,
        "quality_gate": "PASS",
    }


def prepare_frame(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(["oper_part_no", "seq"]).with_columns(
        pl.col("demand_qty").log(2.0).cast(pl.Float64).alias("_audit_z")
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
        pl.col("_audit_z").alias("z_values"),
        pl.col("demand_qty").alias("qty_values"),
    )
    for row in grouped.iter_rows(named=True):
        part = str(row["oper_part_no"])
        seqs = np.asarray(row["seqs"], dtype=np.int64)
        splits = np.asarray(row["splits"])
        z_values = np.asarray(row["z_values"], dtype=np.float64)
        qty_values = np.asarray(row["qty_values"], dtype=np.float64)
        train_values = z_values[splits == "train"]
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
                    "train_mean_z": float(train_values.mean()),
                    "train_var_z": float(train_values.var()),
                    "train_std_z": float(train_values.std()),
                    "train_min_z": float(train_values.min()),
                    "train_max_z": float(train_values.max()),
                    "early_mean_z": early,
                    "late_mean_z": late,
                    "late_minus_early_z": level_shift,
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

            history = z_values[context_idx]
            count = int(history.size)
            history_mean = float(history.mean())
            history_var = float(history.var())
            target_z = float(z_values[target_idx])
            first_half_shift = float("nan")
            if count >= 4:
                half = count // 2
                first_half_shift = float(history[half:].mean() - history[:half].mean())

            context_rows.append(
                {
                    "oper_part_no": part,
                    "target_seq": int(seqs[target_idx]),
                    "context_count": count,
                    "context_span_weeks": int(seqs[context_idx[-1]] - seqs[context_idx[0]]),
                    "history_mean_z": history_mean,
                    "history_var_z": history_var,
                    "history_std_z": float(math.sqrt(max(history_var, 0.0))),
                    "history_min_z": float(history.min()),
                    "history_max_z": float(history.max()),
                    "history_last_z": float(history[-1]),
                    "target_z": target_z,
                    "target_qty": float(qty_values[target_idx]),
                    "target_minus_history_mean_z": target_z - history_mean,
                    "abs_target_minus_history_mean_z": abs(target_z - history_mean),
                    "target_minus_last_z": target_z - float(history[-1]),
                    "abs_target_minus_last_z": abs(target_z - float(history[-1])),
                    "target_outside_history_range": bool(
                        target_z < float(history.min()) or target_z > float(history.max())
                    ),
                    "target_above_history_max": bool(target_z > float(history.max())),
                    "target_below_history_min": bool(target_z < float(history.min())),
                    "recent_minus_prior_mean_z": first_half_shift,
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
            f"target_count_match={target_count_match}, "
            f"distribution_match={distribution_match}"
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
    train = frame.filter(pl.col("chronological_split") == "train")
    z = train["_audit_z"].to_numpy()
    mu = float(z.mean())
    var = float(z.var())
    sigma = float(math.sqrt(var))

    counts = series_stats["train_event_count"].to_numpy().astype(np.float64)
    means = series_stats["train_mean_z"].to_numpy()
    variances = series_stats["train_var_z"].to_numpy()
    within_var = float(np.sum(counts * variances) / np.sum(counts))
    between_var = float(np.sum(counts * (means - mu) ** 2) / np.sum(counts))

    global_stats: dict[str, float | int] = {
        "train_event_count": int(z.size),
        "train_series_count": int(series_stats.height),
        "train_mean_z": mu,
        "train_var_z": var,
        "train_std_z": sigma,
        "train_min_z": float(z.min()),
        "train_max_z": float(z.max()),
        "train_qty_min": float(np.exp2(z.min())),
        "train_qty_max": float(np.exp2(z.max())),
        "within_series_variance": within_var,
        "between_series_variance": between_var,
        "between_series_variance_share": between_var / max(var, 1e-12),
        "variance_decomposition_abs_error": abs(var - within_var - between_var),
    }

    context_count = contexts["context_count"].to_numpy()
    history_std = contexts["history_std_z"].to_numpy()
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
        "std_below_0p1_context_share": float(np.mean(history_std < 0.1)),
        "history_std_p50": quantile(history_std, 0.50),
        "history_std_p95": quantile(history_std, 0.95),
        "series_train_count_eq_1_share": float(np.mean(series_count == 1)),
        "series_train_count_le_4_share": float(np.mean(series_count <= 4)),
        "zero_variance_series_share": float(
            np.mean(series_stats["train_std_z"].to_numpy() <= 1e-12)
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
    stds = contexts["history_std_z"].to_numpy()
    target_deviation = contexts["abs_target_minus_history_mean_z"].to_numpy()
    for label, low, high in HISTORY_BUCKETS:
        mask = (counts >= low) & (counts <= high)
        bucket_std = stds[mask]
        bucket_target = target_deviation[mask]
        rows.append(
            {
                "history_bucket": label,
                "sample_count": int(mask.sum()),
                "sample_share": float(mask.mean()),
                "zero_variance_share": float(np.mean(bucket_std <= 1e-12)),
                "std_p25": quantile(bucket_std, 0.25),
                "std_p50": quantile(bucket_std, 0.50),
                "std_p75": quantile(bucket_std, 0.75),
                "std_p95": quantile(bucket_std, 0.95),
                "target_abs_deviation_p50": quantile(bucket_target, 0.50),
                "target_abs_deviation_p95": quantile(bucket_target, 0.95),
            }
        )
    return pl.DataFrame(rows)


def summarize_level_shift(
    contexts: pl.DataFrame,
    series_stats: pl.DataFrame,
    global_mean: float,
) -> dict[str, float | int]:
    local_mean_gap = np.abs(contexts["history_mean_z"].to_numpy() - global_mean)
    target_history_gap = contexts["abs_target_minus_history_mean_z"].to_numpy()
    target_last_gap = contexts["abs_target_minus_last_z"].to_numpy()
    window_shift = contexts["recent_minus_prior_mean_z"].to_numpy()
    series_shift = series_stats["late_minus_early_z"].to_numpy()
    return {
        "window_mean_abs_gap_vs_global_p50": quantile(local_mean_gap, 0.50),
        "window_mean_abs_gap_vs_global_p95": quantile(local_mean_gap, 0.95),
        "target_abs_gap_vs_history_mean_p50": quantile(target_history_gap, 0.50),
        "target_abs_gap_vs_history_mean_p95": quantile(target_history_gap, 0.95),
        "target_abs_gap_vs_last_p50": quantile(target_last_gap, 0.50),
        "target_abs_gap_vs_last_p95": quantile(target_last_gap, 0.95),
        "target_outside_history_range_share": float(
            contexts["target_outside_history_range"].mean()
        ),
        "target_above_history_max_share": float(contexts["target_above_history_max"].mean()),
        "target_below_history_min_share": float(contexts["target_below_history_min"].mean()),
        "window_half_shift_available_count": int(np.isfinite(window_shift).sum()),
        "window_half_abs_shift_p50": quantile(np.abs(window_shift), 0.50),
        "window_half_abs_shift_p95": quantile(np.abs(window_shift), 0.95),
        "series_half_shift_available_count": int(np.isfinite(series_shift).sum()),
        "series_half_abs_shift_p50": quantile(np.abs(series_shift), 0.50),
        "series_half_abs_shift_p95": quantile(np.abs(series_shift), 0.95),
        "series_half_signed_shift_mean": float(np.nanmean(series_shift)),
    }


def shrinkage_candidates(
    contexts: pl.DataFrame,
    *,
    global_mean: float,
    global_var: float,
    sigma_floor: float,
) -> pl.DataFrame:
    counts = contexts["context_count"].to_numpy().astype(np.float64)
    history_mean = contexts["history_mean_z"].to_numpy()
    history_var = contexts["history_var_z"].to_numpy()
    target = contexts["target_z"].to_numpy()
    rows: list[dict[str, object]] = []
    for k in K_CANDIDATES:
        alpha = counts / (counts + k)
        center = alpha * history_mean + (1.0 - alpha) * global_mean
        second_moment = alpha * (history_var + history_mean**2) + (1.0 - alpha) * (
            global_var + global_mean**2
        )
        variance = np.maximum(second_moment - center**2, sigma_floor**2)
        scale = np.sqrt(variance)
        abs_target_norm = np.abs((target - center) / scale)
        one_event_scale = scale[counts == 1]
        rows.append(
            {
                "shrinkage_k": k,
                "alpha_p50": quantile(alpha, 0.50),
                "alpha_p95": quantile(alpha, 0.95),
                "scale_min": float(scale.min()),
                "scale_p01": quantile(scale, 0.01),
                "scale_p05": quantile(scale, 0.05),
                "scale_p50": quantile(scale, 0.50),
                "scale_p95": quantile(scale, 0.95),
                "scale_max": float(scale.max()),
                "one_event_scale_p50": quantile(one_event_scale, 0.50),
                "target_abs_norm_p50": quantile(abs_target_norm, 0.50),
                "target_abs_norm_p90": quantile(abs_target_norm, 0.90),
                "target_abs_norm_p95": quantile(abs_target_norm, 0.95),
                "target_abs_norm_p99": quantile(abs_target_norm, 0.99),
                "target_abs_norm_max": float(abs_target_norm.max()),
                "target_abs_norm_gt_3_share": float(np.mean(abs_target_norm > 3.0)),
                "target_abs_norm_gt_5_share": float(np.mean(abs_target_norm > 5.0)),
                "finite_share": float(np.mean(np.isfinite(abs_target_norm))),
            }
        )
    return pl.DataFrame(rows).sort("shrinkage_k")


def choose_constants(
    candidates: pl.DataFrame,
    *,
    global_std: float,
    train_min_z: float,
    train_max_z: float,
) -> dict[str, object]:
    eligible = candidates.filter(
        (pl.col("target_abs_norm_p99") <= 2.0)
        & (pl.col("target_abs_norm_gt_3_share") <= 0.001)
        & (pl.col("one_event_scale_p50") >= 0.75 * global_std)
        & (pl.col("alpha_p50") >= 0.25)
    ).sort(["target_abs_norm_p99", "shrinkage_k"])
    if eligible.is_empty():
        raise ValueError("No shrinkage candidate passed the predeclared stability gate")
    selected = eligible.row(0, named=True)
    return {
        "shrinkage_k": float(selected["shrinkage_k"]),
        "sigma_floor": max(global_std * 1e-3, 1e-4),
        "exp2_clamp_min_z": float(math.floor(train_min_z) - 2),
        "exp2_clamp_max_z": float(math.ceil(train_max_z) + 2),
        "selection_scope": "train_only",
        "selection_rule": {
            "target_abs_norm_p99_max": 2.0,
            "target_abs_norm_gt_3_share_max": 0.001,
            "one_event_scale_p50_min_global_std_ratio": 0.75,
            "alpha_p50_min": 0.25,
            "rank": "lowest target_abs_norm_p99, then smallest k",
        },
    }


def plot_history_length(distribution: pl.DataFrame, output_path: Path) -> None:
    x = distribution["context_count"].to_numpy()
    y = 100.0 * distribution["sample_share"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(x, y, color="#2563EB", edgecolor="#263238", linewidth=0.5)
    ax.axvline(4.5, color="#D97706", linestyle="--", linewidth=1.5, label="Short context <= 4")
    ax.set_title("Intermittent train target context length")
    ax.set_xlabel("Valid history events after weekly/max-length truncation")
    ax.set_ylabel("Train target share (%)")
    ax.set_xticks(x)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_variance_by_bucket(contexts: pl.DataFrame, output_path: Path) -> None:
    counts = contexts["context_count"].to_numpy()
    stds = contexts["history_std_z"].to_numpy()
    values = []
    labels = []
    for label, low, high in HISTORY_BUCKETS:
        mask = (counts >= low) & (counts <= high)
        values.append(stds[mask])
        labels.append(label)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    boxes = ax.boxplot(values, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch in boxes["boxes"]:
        patch.set_facecolor("#B9D6F2")
        patch.set_edgecolor("#263238")
    ax.set_title("History log2-quantity standard deviation by context length")
    ax.set_xlabel("History count bucket")
    ax.set_ylabel("Population standard deviation")
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_level_shift(
    contexts: pl.DataFrame,
    series_stats: pl.DataFrame,
    global_mean: float,
    output_path: Path,
) -> None:
    window_global = np.abs(contexts["history_mean_z"].to_numpy() - global_mean)
    target_history = contexts["abs_target_minus_history_mean_z"].to_numpy()
    series_shift = np.abs(series_stats["late_minus_early_z"].to_numpy())
    series_shift = series_shift[np.isfinite(series_shift)]
    bins = np.linspace(0.0, 4.0, 41)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.hist(window_global, bins=bins, density=True, histtype="step", linewidth=2, color="#2563EB", label="|window mean - global mean|")
    ax.hist(target_history, bins=bins, density=True, histtype="step", linewidth=2, color="#D97706", label="|target - window mean|")
    ax.hist(series_shift, bins=bins, density=True, histtype="step", linewidth=2, color="#6B7C23", label="|late series mean - early mean|")
    ax.set_title("Intermittent train-only magnitude level differences")
    ax.set_xlabel("Absolute difference in log2 quantity")
    ax.set_ylabel("Density")
    ax.set_xlim(0.0, 4.0)
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shrinkage_candidates(
    candidates: pl.DataFrame,
    global_std: float,
    selected_k: float,
    output_path: Path,
) -> None:
    k = candidates["shrinkage_k"].to_numpy()
    p99 = candidates["target_abs_norm_p99"].to_numpy()
    scale_ratio = candidates["scale_p01"].to_numpy() / global_std
    fig, left = plt.subplots(figsize=(10, 5.8))
    right = left.twinx()
    left.plot(k, p99, marker="o", color="#2563EB", linewidth=2, label="Target |z_norm| p99")
    right.plot(k, scale_ratio, marker="s", color="#D97706", linewidth=2, label="Scale p01 / global std")
    left.axvline(selected_k, color="#263238", linestyle="--", linewidth=1.3, label=f"Selected k={selected_k:g}")
    left.set_xscale("log", base=2)
    left.set_xticks(k, labels=[f"{value:g}" for value in k])
    left.set_title("Shrinkage stability across train-only k candidates")
    left.set_xlabel("shrinkage_k")
    left.set_ylabel("Target absolute normalized residual p99", color="#2563EB")
    right.set_ylabel("Scale p01 / global standard deviation", color="#D97706")
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
    candidates: pl.DataFrame,
    constants: dict[str, object],
    lookback_weeks: int,
    max_seq_len: int,
) -> str:
    selected = candidates.filter(pl.col("shrinkage_k") == constants["shrinkage_k"]).row(0, named=True)
    lines = [
        "# Intermittent Train-Only Magnitude RevIN Audit",
        "",
        "Status: `completed`  ",
        "Scope: train-only fixed-split magnitude contexts  ",
        f"Window: lookback `{lookback_weeks}`, max sequence length `{max_seq_len}` including target",
        "",
        "## Decision",
        "",
        f"Plain causal RevIN is not safe as the primary Intermittent normalization. "
        f"`{100*float(history_summary['context_count_le_4_share']):.2f}%` of train targets have at most four "
        f"history events and `{100*float(history_summary['zero_variance_context_share']):.2f}%` of contexts have "
        "zero log-magnitude variance.",
        "",
        f"Series scale heterogeneity is material: between-series level differences explain "
        f"`{100*float(global_stats['between_series_variance_share']):.2f}%` of total train log2-quantity variance. "
        "At the same time, local and within-series shifts are non-trivial, so global normalization alone is an "
        "appropriate M0 baseline but not the preferred final candidate.",
        "",
        f"The train-only stability rule selects `shrinkage_k={float(constants['shrinkage_k']):g}`. "
        f"Its target absolute normalized residual p99 is `{float(selected['target_abs_norm_p99']):.3f}` and "
        f"only `{100*float(selected['target_abs_norm_gt_3_share']):.4f}%` exceed three. "
        f"Freeze `sigma_floor={float(constants['sigma_floor']):.8f}` and exp2 clamp "
        f"`[{float(constants['exp2_clamp_min_z']):g}, {float(constants['exp2_clamp_max_z']):g}]` before validation.",
        "",
        "## Source And Data Quality",
        "",
        f"- Source: `{source_path}`",
        f"- Decoded train rows/series: `{int(source_quality['row_count']):,}` / `{int(source_quality['series_count']):,}`",
        f"- Train events/targets: `{int(global_stats['train_event_count']):,}` / `{int(history_summary['train_target_count']):,}`",
        "- Required nulls, duplicate part/seq keys, non-positive quantity, and decoded non-train rows: `0`",
        f"- Maximum `log2(qty)` versus `mark+scale_residual` error: `{float(source_quality['max_log2_reconstruction_abs_error']):.3e}`",
        "- Data quality gate: `PASS`",
        "",
        "## History Length And Variance",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Context count p50 / p95 / max | `{float(history_summary['context_count_p50']):.0f}` / `{float(history_summary['context_count_p95']):.0f}` / `{int(history_summary['context_count_max'])}` |",
        f"| One-event context share | `{100*float(history_summary['context_count_eq_1_share']):.2f}%` |",
        f"| Context count <=2 | `{100*float(history_summary['context_count_le_2_share']):.2f}%` |",
        f"| Context count <=4 | `{100*float(history_summary['context_count_le_4_share']):.2f}%` |",
        f"| Zero-variance context share | `{100*float(history_summary['zero_variance_context_share']):.2f}%` |",
        f"| Zero-variance train-series share | `{100*float(history_summary['zero_variance_series_share']):.2f}%` |",
        f"| History std p50 / p95 | `{float(history_summary['history_std_p50']):.4f}` / `{float(history_summary['history_std_p95']):.4f}` |",
        "",
        "## Scale Heterogeneity And Level Shift",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Train global mean/std z | `{float(global_stats['train_mean_z']):.4f}` / `{float(global_stats['train_std_z']):.4f}` |",
        f"| Train z range | `{float(global_stats['train_min_z']):.4f}` to `{float(global_stats['train_max_z']):.4f}` |",
        f"| Between-series variance share | `{100*float(global_stats['between_series_variance_share']):.2f}%` |",
        f"| Window/global mean absolute gap p50 / p95 | `{float(level_shift['window_mean_abs_gap_vs_global_p50']):.4f}` / `{float(level_shift['window_mean_abs_gap_vs_global_p95']):.4f}` |",
        f"| Target/history mean absolute gap p50 / p95 | `{float(level_shift['target_abs_gap_vs_history_mean_p50']):.4f}` / `{float(level_shift['target_abs_gap_vs_history_mean_p95']):.4f}` |",
        f"| Target outside history range | `{100*float(level_shift['target_outside_history_range_share']):.2f}%` |",
        f"| Window half absolute shift p50 / p95 | `{float(level_shift['window_half_abs_shift_p50']):.4f}` / `{float(level_shift['window_half_abs_shift_p95']):.4f}` |",
        f"| Series early/late absolute shift p50 / p95 | `{float(level_shift['series_half_abs_shift_p50']):.4f}` / `{float(level_shift['series_half_abs_shift_p95']):.4f}` |",
        "",
        "## Shrinkage Candidate Audit",
        "",
        "| k | alpha p50 | scale p01 | one-event scale p50 | target abs(z_norm) p99 | >3 share |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidates.iter_rows(named=True):
        marker = " (selected)" if float(row["shrinkage_k"]) == float(constants["shrinkage_k"]) else ""
        lines.append(
            f"| {float(row['shrinkage_k']):g}{marker} | {float(row['alpha_p50']):.4f} | "
            f"{float(row['scale_p01']):.4f} | {float(row['one_event_scale_p50']):.4f} | "
            f"{float(row['target_abs_norm_p99']):.4f} | "
            f"{100*float(row['target_abs_norm_gt_3_share']):.4f}% |"
        )
    lines.extend(
        [
            "",
            "## Frozen Constants",
            "",
            f"- `shrinkage_k={float(constants['shrinkage_k']):g}`",
            f"- `magnitude_sigma_floor={float(constants['sigma_floor']):.8f}`",
            f"- `magnitude_exp_clamp_min={float(constants['exp2_clamp_min_z']):g}`",
            f"- `magnitude_exp_clamp_max={float(constants['exp2_clamp_max_z']):g}`",
            "- All constants were selected from train rows and train-target contexts only.",
            "",
            "## Interpretation",
            "",
            "- M0 global normalization remains necessary to isolate direct-regression benefit.",
            "- M1 per-series scaling is unstable without fallback because many series have one event or zero variance.",
            "- M2 plain window RevIN is a diagnostic ablation, not the primary candidate.",
            "- M3/M4 shrinkage is justified by the combination of short contexts, zero variance, and large between-series level variation.",
            "- The non-zero early/late and within-window shifts support a reversible local component rather than a static per-series scaler alone.",
            "",
            "## Scope And Caveats",
            "",
            "- No validation or held-out test rows were read for constant selection.",
            "- This audit establishes normalization feasibility; it does not show that M0-M4 improve model metrics.",
            "- The shrinkage gate controls numeric target scale, not predictive accuracy.",
            "- M0 must still pass the predeclared validation gate before any RevIN claim.",
            "",
            "## Next",
            "",
            "Implement M0 direct log2-magnitude baseline and the shared magnitude-context contract with the frozen constants.",
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
    logger.info("Starting train-only magnitude RevIN audit | dataset=%s", dataset_path)

    raw = (
        pl.scan_parquet(dataset_path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
    )
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
    level_shift = summarize_level_shift(contexts, series_stats, float(global_stats["train_mean_z"]))
    sigma_floor = max(float(global_stats["train_std_z"]) * 1e-3, 1e-4)
    candidates = shrinkage_candidates(
        contexts,
        global_mean=float(global_stats["train_mean_z"]),
        global_var=float(global_stats["train_var_z"]),
        sigma_floor=sigma_floor,
    )
    constants = choose_constants(
        candidates,
        global_std=float(global_stats["train_std_z"]),
        train_min_z=float(global_stats["train_min_z"]),
        train_max_z=float(global_stats["train_max_z"]),
    )

    quality_rows = metric_rows("source", source_quality)
    quality_rows.extend(metric_rows("context", context_validations))
    pl.DataFrame(quality_rows).write_csv(data_dir / "data_quality_summary.csv")
    pl.DataFrame(metric_rows("train_global", global_stats)).write_csv(
        data_dir / "global_statistics.csv"
    )
    pl.DataFrame(metric_rows("history", history_summary)).write_csv(
        data_dir / "history_length_summary.csv"
    )
    history_length_distribution(contexts).write_csv(data_dir / "history_length_distribution.csv")
    variance_by_bucket(contexts).write_csv(data_dir / "variance_by_history_bucket.csv")
    pl.DataFrame(metric_rows("level_shift", level_shift)).write_csv(
        data_dir / "level_shift_summary.csv"
    )
    candidates.write_csv(data_dir / "shrinkage_candidate_summary.csv")
    contexts.write_parquet(data_dir / "train_target_context_statistics.parquet")
    series_stats.write_csv(data_dir / "train_series_statistics.csv")
    (data_dir / "recommended_constants.json").write_text(
        json.dumps(constants, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    plot_history_length(
        history_length_distribution(contexts), plot_dir / "history_length_distribution.png"
    )
    plot_variance_by_bucket(contexts, plot_dir / "history_variance_by_length.png")
    plot_level_shift(
        contexts,
        series_stats,
        float(global_stats["train_mean_z"]),
        plot_dir / "level_shift_distributions.png",
    )
    plot_shrinkage_candidates(
        candidates,
        float(global_stats["train_std_z"]),
        float(constants["shrinkage_k"]),
        plot_dir / "shrinkage_candidate_stability.png",
    )

    report = build_report(
        source_path=dataset_path,
        source_quality=source_quality,
        global_stats=global_stats,
        history_summary=history_summary,
        level_shift=level_shift,
        candidates=candidates,
        constants=constants,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    summary = {
        "status": "completed",
        "quality_gate": source_quality["quality_gate"],
        "train_target_count": history_summary["train_target_count"],
        "context_count_le_4_share": history_summary["context_count_le_4_share"],
        "zero_variance_context_share": history_summary["zero_variance_context_share"],
        "zero_variance_series_share": history_summary["zero_variance_series_share"],
        "between_series_variance_share": global_stats["between_series_variance_share"],
        "target_outside_history_range_share": level_shift["target_outside_history_range_share"],
        "recommended_constants": constants,
        "decision": "proceed_with_m0_then_shrinkage_revin_m3_m4",
        "plain_revin_role": "diagnostic_only",
        "per_series_scaler_role": "ablation_with_global_fallback",
        "held_out_test_read": False,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    finished_at = datetime.now(KST)
    manifest = {
        "status": "completed",
        "analysis": "intermittent_train_only_magnitude_revin_audit",
        "started_at_kst": started_at.isoformat(),
        "finished_at_kst": finished_at.isoformat(),
        "execution_environment": "local_analysis",
        "dataset_path": str(dataset_path),
        "decoded_train_scope_sha256": train_scope_sha256(raw),
        "output_dir": str(output_dir),
        "split_scope": "train_only",
        "held_out_test_read": False,
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "max_context_len": args.max_seq_len - 1,
        "target_scope": "RMTPPWeekLookbackDataset-compatible train targets",
        "k_candidates": list(K_CANDIDATES),
        "recommended_constants": constants,
        "artifact_order": [
            "audit_manifest.json",
            "logs/audit.log",
            "audit_summary.json",
            "data/*.csv and data/*.parquet",
            "report.md",
            "plots/*.png",
        ],
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info(
        "Completed audit | train_targets=%d selected_k=%g sigma_floor=%.8f exp2_clamp=[%g,%g]",
        int(history_summary["train_target_count"]),
        float(constants["shrinkage_k"]),
        float(constants["sigma_floor"]),
        float(constants["exp2_clamp_min_z"]),
        float(constants["exp2_clamp_max_z"]),
    )


def main() -> None:
    args = parse_args()
    write_outputs(args)


if __name__ == "__main__":
    main()
