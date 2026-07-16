#!/usr/bin/env python3
"""Audit Taxi train-only next-mark and delta-time dependence for TitanTPP V4."""

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
from typing import Any
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


DEFAULT_DATASET = (
    PROJECT_ROOT / "sample_data/new_york_taxi/yellow_trip_hourly_train.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "search_artifacts/model_enhancement_titantpp_v4_taxi_train_time_audit_0716"
)
REQUIRED_COLUMNS = {
    "oper_part_no",
    "seq",
    "delta_t",
    "demand_qty",
    "mark",
    "scale_residual",
    "chronological_split",
}
KST = ZoneInfo("Asia/Seoul")

# Frozen before reading audit delta-time results.
FIT_RATIO = 0.80
MIN_SUPPORTED_MARK_COUNT = 100
MIN_SUPPORTED_MARK_SHARE = 0.01
MIN_ETA_SQUARED = 0.01
MIN_EVAL_NLL_IMPROVEMENT_PCT = 0.50
MIN_BOOTSTRAP_CI_LOW_PCT = 0.0
MIN_SERIES_IMPROVED_SHARE = 0.55
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42
W_MIN = 1e-3
W_MAX = 1.0
W_GRID_SIZE = 512
WD_CLAMP = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lookback-weeks", type=int, default=168)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--execution-server", default="unknown")
    parser.add_argument("--tmux-session", default="none")
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("taxi_mark_time_audit")
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


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def frame_sha256(frame: pl.DataFrame, columns: list[str]) -> str:
    row_hashes = frame.select(columns).hash_rows(seed=0).to_numpy()
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def metric_rows(scope: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"scope": scope, "metric": key, "value": jsonable(value)}
        for key, value in metrics.items()
        if not isinstance(value, (dict, list, tuple))
    ]


def validate_train_source(frame: pl.DataFrame) -> dict[str, Any]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Taxi train dataset is missing required columns: {missing}")

    null_cells = int(
        frame.select(
            pl.sum_horizontal(
                [pl.col(column).is_null().cast(pl.Int64) for column in REQUIRED_COLUMNS]
            )
        )
        .to_series()
        .sum()
    )
    duplicate_rows = int(
        frame.group_by(["oper_part_no", "seq"])
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum())
        .item()
        or 0
    )
    non_train_rows = int(frame.filter(pl.col("chronological_split") != "train").height)
    invalid_mark_rows = int(
        frame.filter(pl.col("mark").is_null() | (pl.col("mark") < 0)).height
    )
    non_positive_quantity_rows = int(frame.filter(pl.col("demand_qty") <= 0).height)
    source_non_positive_dt_rows = int(frame.filter(pl.col("delta_t") <= 0).height)
    series_count = int(frame["oper_part_no"].n_unique())
    row_count = int(frame.height)

    if null_cells:
        raise ValueError(f"Taxi train dataset has {null_cells} null required cells")
    if duplicate_rows:
        raise ValueError(f"Taxi train dataset has {duplicate_rows} duplicate part/seq rows")
    if non_train_rows:
        raise ValueError(f"Train-only audit received {non_train_rows} non-train rows")
    if invalid_mark_rows:
        raise ValueError(f"Taxi train dataset has {invalid_mark_rows} invalid mark rows")
    if non_positive_quantity_rows:
        raise ValueError(
            f"Taxi train dataset has {non_positive_quantity_rows} non-positive quantities"
        )
    if row_count == 0 or series_count == 0:
        raise ValueError("Taxi train dataset is empty")

    return {
        "row_count": row_count,
        "series_count": series_count,
        "required_null_cells": null_cells,
        "duplicate_part_seq_rows": duplicate_rows,
        "non_train_rows": non_train_rows,
        "invalid_mark_rows": invalid_mark_rows,
        "non_positive_quantity_rows": non_positive_quantity_rows,
        "source_non_positive_delta_t_rows": source_non_positive_dt_rows,
        "quality_gate": "PASS",
    }


def extract_loader_train_targets(
    frame: pl.DataFrame,
    *,
    lookback_weeks: int,
    max_seq_len: int,
) -> tuple[pl.DataFrame, RMTPPWeekLookbackDataset]:
    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback_weeks,
        max_seq_len=max_seq_len,
        mode="all",
        target_splits={"train"},
    )
    rows: list[dict[str, Any]] = []
    for sample_index, (part_index, context_end) in enumerate(dataset.index):
        target_index = int(context_end + 1)
        target_split = str(dataset.split_lists[part_index][target_index])
        rows.append(
            {
                "sample_index": sample_index,
                "oper_part_no": str(dataset.parts[part_index]),
                "target_seq": int(dataset.seq_lists[part_index][target_index]),
                "mark": int(dataset.mk_lists[part_index][target_index]),
                "delta_t": float(max(dataset.dt_lists[part_index][target_index], 1.0)),
                "target_split": target_split,
            }
        )
    targets = pl.DataFrame(rows).sort(["oper_part_no", "target_seq"])
    return targets.with_columns(pl.col("delta_t").log1p().alias("log1p_delta_t")), dataset


def validate_loader_contract(
    frame: pl.DataFrame,
    targets: pl.DataFrame,
    dataset: RMTPPWeekLookbackDataset,
) -> dict[str, Any]:
    expected_target_count = int(
        frame.group_by("oper_part_no")
        .len()
        .select((pl.col("len") - 1).clip(0, None).sum())
        .item()
        or 0
    )
    if targets.height != expected_target_count or len(dataset) != expected_target_count:
        raise ValueError(
            "Taxi train target count does not match one-target-per-event-after-first contract: "
            f"targets={targets.height} loader={len(dataset)} expected={expected_target_count}"
        )

    target_marks = targets["mark"].to_numpy()
    target_dts = targets["delta_t"].to_numpy()
    mark_mismatch_count = 0
    dt_mismatch_count = 0
    min_valid_length = max(dataset.max_len, 1)
    max_valid_length = 0
    for sample_index in range(len(dataset)):
        sample = dataset[sample_index]
        valid_positions = np.flatnonzero(sample["mask"].numpy())
        if valid_positions.size < 2:
            raise ValueError("Week-lookback sample must contain context plus target")
        final_position = int(valid_positions[-1])
        observed_mark = int(sample["marks"][final_position].item())
        observed_dt = float(sample["dts"][final_position].item())
        mark_mismatch_count += int(observed_mark != int(target_marks[sample_index]))
        dt_mismatch_count += int(
            not math.isclose(
                observed_dt,
                float(target_dts[sample_index]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        min_valid_length = min(min_valid_length, int(valid_positions.size))
        max_valid_length = max(max_valid_length, int(valid_positions.size))

    non_train_targets = int(targets.filter(pl.col("target_split") != "train").height)
    non_positive_target_dt = int(targets.filter(pl.col("delta_t") <= 0).height)
    if mark_mismatch_count or dt_mismatch_count:
        raise ValueError(
            "Decoded target rows disagree with RMTPPWeekLookbackDataset: "
            f"mark={mark_mismatch_count} dt={dt_mismatch_count}"
        )
    if non_train_targets or non_positive_target_dt:
        raise ValueError(
            "Train target contract failed: "
            f"non_train={non_train_targets} non_positive_dt={non_positive_target_dt}"
        )

    return {
        "expected_train_target_count": expected_target_count,
        "decoded_train_target_count": int(targets.height),
        "loader_train_target_count": int(len(dataset)),
        "non_train_target_count": non_train_targets,
        "non_positive_target_delta_t_count": non_positive_target_dt,
        "mark_mismatch_count": mark_mismatch_count,
        "delta_t_mismatch_count": dt_mismatch_count,
        "min_valid_sequence_length": min_valid_length,
        "max_valid_sequence_length": max_valid_length,
        "loader_contract_gate": "PASS",
    }


def assign_temporal_audit_partition(
    targets: pl.DataFrame,
    *,
    fit_ratio: float = FIT_RATIO,
) -> pl.DataFrame:
    if not 0.0 < fit_ratio < 1.0:
        raise ValueError("fit_ratio must be in (0, 1)")
    groups: list[pl.DataFrame] = []
    for group in targets.sort(["oper_part_no", "target_seq"]).partition_by(
        "oper_part_no", maintain_order=True
    ):
        count = group.height
        fit_count = count if count < 2 else min(count - 1, max(1, int(math.floor(count * fit_ratio))))
        partition = ["audit_fit"] * fit_count + ["audit_eval"] * (count - fit_count)
        groups.append(
            group.with_columns(
                pl.Series("audit_partition", partition, dtype=pl.String),
                pl.Series("series_target_index", np.arange(count), dtype=pl.Int64),
            )
        )
    if not groups:
        raise ValueError("No Taxi train targets are available for temporal audit")
    result = pl.concat(groups, how="vertical")
    eval_rows = result.filter(pl.col("audit_partition") == "audit_eval").height
    if eval_rows == 0:
        raise ValueError("Temporal audit split produced no audit_eval targets")
    return result


def summarize_mark_delta_time(targets: pl.DataFrame) -> pl.DataFrame:
    total = targets.height
    rows: list[dict[str, Any]] = []
    for mark_frame in targets.partition_by("mark", maintain_order=False):
        mark = int(mark_frame["mark"][0])
        dt = mark_frame["delta_t"].to_numpy().astype(np.float64)
        log_dt = np.log1p(dt)
        count = int(dt.size)
        rows.append(
            {
                "mark": mark,
                "count": count,
                "share": count / total,
                "series_count": int(mark_frame["oper_part_no"].n_unique()),
                "supported": bool(
                    count >= MIN_SUPPORTED_MARK_COUNT
                    and count / total >= MIN_SUPPORTED_MARK_SHARE
                ),
                "delta_t_mean": float(dt.mean()),
                "delta_t_std": float(dt.std()),
                "delta_t_min": float(dt.min()),
                "delta_t_p25": quantile(dt, 0.25),
                "delta_t_median": quantile(dt, 0.50),
                "delta_t_p75": quantile(dt, 0.75),
                "delta_t_p95": quantile(dt, 0.95),
                "delta_t_p99": quantile(dt, 0.99),
                "delta_t_max": float(dt.max()),
                "log1p_delta_t_mean": float(log_dt.mean()),
                "log1p_delta_t_std": float(log_dt.std()),
                "log1p_delta_t_median": quantile(log_dt, 0.50),
                "log1p_delta_t_iqr": quantile(log_dt, 0.75) - quantile(log_dt, 0.25),
            }
        )
    return pl.DataFrame(rows).sort("mark")


def distribution_effect_summary(
    targets: pl.DataFrame,
    mark_summary: pl.DataFrame,
) -> dict[str, Any]:
    marks = targets["mark"].to_numpy().astype(np.int64)
    values = targets["log1p_delta_t"].to_numpy().astype(np.float64)
    global_mean = float(values.mean())
    total_ss = float(np.square(values - global_mean).sum())
    between_ss = 0.0
    within_ss = 0.0
    for mark in np.unique(marks):
        group = values[marks == mark]
        group_mean = float(group.mean())
        between_ss += group.size * (group_mean - global_mean) ** 2
        within_ss += float(np.square(group - group_mean).sum())
    eta_squared = between_ss / total_ss if total_ss > 0.0 else 0.0
    group_count = int(np.unique(marks).size)
    denominator_df = max(values.size - group_count, 1)
    mean_square_within = within_ss / denominator_df
    omega_squared = max(
        0.0,
        (between_ss - (group_count - 1) * mean_square_within)
        / max(total_ss + mean_square_within, np.finfo(np.float64).eps),
    )
    mark_std = float(marks.std())
    value_std = float(values.std())
    pearson = (
        float(np.corrcoef(marks.astype(np.float64), values)[0, 1])
        if mark_std > 0.0 and value_std > 0.0
        else 0.0
    )

    supported = mark_summary.filter(pl.col("supported"))
    supported_marks = supported["mark"].to_list()
    supported_medians = supported["delta_t_median"].to_numpy().astype(np.float64)
    median_ratio = (
        float(supported_medians.max() / supported_medians.min())
        if supported_medians.size >= 2 and supported_medians.min() > 0.0
        else 1.0
    )
    max_pairwise_smd = 0.0
    for left_index, left_mark in enumerate(supported_marks):
        left = values[marks == int(left_mark)]
        for right_mark in supported_marks[left_index + 1 :]:
            right = values[marks == int(right_mark)]
            pooled_variance = (
                ((left.size - 1) * left.var(ddof=1) + (right.size - 1) * right.var(ddof=1))
                / max(left.size + right.size - 2, 1)
            )
            if pooled_variance > 0.0:
                smd = abs(float(left.mean() - right.mean())) / math.sqrt(pooled_variance)
                max_pairwise_smd = max(max_pairwise_smd, smd)

    return {
        "target_count": int(values.size),
        "real_mark_count": group_count,
        "supported_mark_count": int(len(supported_marks)),
        "supported_marks": [int(mark) for mark in supported_marks],
        "eta_squared_log1p_delta_t": eta_squared,
        "omega_squared_log1p_delta_t": omega_squared,
        "pearson_mark_log1p_delta_t": pearson,
        "supported_mark_delta_t_median_ratio": median_ratio,
        "max_pairwise_standardized_log1p_mean_difference": max_pairwise_smd,
    }


def _fit_intercepts_for_w(
    delta_t: np.ndarray,
    marks: np.ndarray,
    *,
    w: float,
    mark_conditioned: bool,
) -> tuple[float, dict[int, float], np.ndarray]:
    z = np.minimum(w * delta_t, WD_CLAMP)
    integrated_term = np.expm1(z)
    fallback_a = math.log(delta_t.size * w / float(integrated_term.sum()))
    intercepts: dict[int, float] = {}
    selected_a = np.full(delta_t.shape, fallback_a, dtype=np.float64)
    if mark_conditioned:
        for mark in np.unique(marks):
            mask = marks == mark
            mark_a = math.log(mask.sum() * w / float(integrated_term[mask].sum()))
            intercepts[int(mark)] = mark_a
            selected_a[mask] = mark_a
    return fallback_a, intercepts, selected_a


def _nll_from_intercepts(
    delta_t: np.ndarray,
    selected_a: np.ndarray,
    *,
    w: float,
) -> np.ndarray:
    z = np.minimum(w * delta_t, WD_CLAMP)
    return -(
        selected_a + z - (np.exp(selected_a) / w) * np.expm1(z)
    )


def fit_intercept_only_rmtpp(
    delta_t: np.ndarray,
    marks: np.ndarray,
    *,
    mark_conditioned: bool,
) -> dict[str, Any]:
    delta_t = np.asarray(delta_t, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.int64)
    if delta_t.size == 0 or marks.size != delta_t.size:
        raise ValueError("RMTPP audit fit requires aligned non-empty delta_t and marks")
    if not np.isfinite(delta_t).all() or np.any(delta_t <= 0.0):
        raise ValueError("RMTPP audit fit requires positive finite delta_t")

    best: tuple[float, float, float, dict[int, float]] | None = None
    for w in np.geomspace(W_MIN, W_MAX, W_GRID_SIZE):
        fallback_a, intercepts, selected_a = _fit_intercepts_for_w(
            delta_t,
            marks,
            w=float(w),
            mark_conditioned=mark_conditioned,
        )
        mean_nll = float(_nll_from_intercepts(delta_t, selected_a, w=float(w)).mean())
        candidate = (mean_nll, float(w), fallback_a, intercepts)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    mean_nll, w, fallback_a, intercepts = best
    return {
        "model": "mark_conditioned" if mark_conditioned else "global_shared",
        "fit_mean_nll": mean_nll,
        "w": w,
        "fallback_a": fallback_a,
        "mark_intercepts": intercepts,
        "w_at_lower_boundary": bool(math.isclose(w, W_MIN, rel_tol=0.0, abs_tol=1e-12)),
        "w_at_upper_boundary": bool(math.isclose(w, W_MAX, rel_tol=0.0, abs_tol=1e-12)),
    }


def score_intercept_only_rmtpp(
    delta_t: np.ndarray,
    marks: np.ndarray,
    fitted: dict[str, Any],
) -> tuple[np.ndarray, int]:
    delta_t = np.asarray(delta_t, dtype=np.float64)
    marks = np.asarray(marks, dtype=np.int64)
    fallback_a = float(fitted["fallback_a"])
    intercepts = {int(key): float(value) for key, value in fitted["mark_intercepts"].items()}
    selected_a = np.full(delta_t.shape, fallback_a, dtype=np.float64)
    unseen_count = 0
    for index, mark in enumerate(marks):
        if int(mark) in intercepts:
            selected_a[index] = intercepts[int(mark)]
        elif fitted["model"] == "mark_conditioned":
            unseen_count += 1
    nll = _nll_from_intercepts(delta_t, selected_a, w=float(fitted["w"]))
    return nll, unseen_count


def compare_train_only_holdout(
    partitioned: pl.DataFrame,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    fit_frame = partitioned.filter(pl.col("audit_partition") == "audit_fit")
    eval_frame = partitioned.filter(pl.col("audit_partition") == "audit_eval")
    fit_dt = fit_frame["delta_t"].to_numpy().astype(np.float64)
    fit_marks = fit_frame["mark"].to_numpy().astype(np.int64)
    eval_dt = eval_frame["delta_t"].to_numpy().astype(np.float64)
    eval_marks = eval_frame["mark"].to_numpy().astype(np.int64)

    global_fit = fit_intercept_only_rmtpp(fit_dt, fit_marks, mark_conditioned=False)
    conditional_fit = fit_intercept_only_rmtpp(fit_dt, fit_marks, mark_conditioned=True)
    global_nll, _ = score_intercept_only_rmtpp(eval_dt, eval_marks, global_fit)
    conditional_nll, unseen_count = score_intercept_only_rmtpp(
        eval_dt, eval_marks, conditional_fit
    )
    if not np.isfinite(global_nll).all() or not np.isfinite(conditional_nll).all():
        raise ValueError("Train-only holdout RMTPP scores must be finite")

    scored = eval_frame.with_columns(
        pl.Series("global_nll", global_nll),
        pl.Series("mark_conditioned_nll", conditional_nll),
    )
    series = (
        scored.group_by("oper_part_no")
        .agg(
            pl.len().alias("target_count"),
            pl.col("global_nll").sum().alias("global_nll_sum"),
            pl.col("mark_conditioned_nll").sum().alias("mark_conditioned_nll_sum"),
            pl.col("global_nll").mean().alias("global_mean_nll"),
            pl.col("mark_conditioned_nll").mean().alias("mark_conditioned_mean_nll"),
        )
        .with_columns(
            (
                100.0
                * (pl.col("global_mean_nll") - pl.col("mark_conditioned_mean_nll"))
                / pl.col("global_mean_nll").abs().clip(1e-12, None)
            ).alias("nll_improvement_pct")
        )
        .sort("oper_part_no")
    )

    global_mean_nll = float(global_nll.mean())
    conditional_mean_nll = float(conditional_nll.mean())
    improvement_pct = 100.0 * (global_mean_nll - conditional_mean_nll) / max(
        abs(global_mean_nll), 1e-12
    )
    series_improved_share = float(
        series.select((pl.col("nll_improvement_pct") > 0.0).mean()).item()
    )

    global_sums = series["global_nll_sum"].to_numpy().astype(np.float64)
    conditional_sums = series["mark_conditioned_nll_sum"].to_numpy().astype(np.float64)
    counts = series["target_count"].to_numpy().astype(np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_improvements = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, series.height, size=series.height)
        denominator = float(counts[indices].sum())
        global_boot = float(global_sums[indices].sum() / denominator)
        conditional_boot = float(conditional_sums[indices].sum() / denominator)
        bootstrap_improvements[replicate] = (
            100.0
            * (global_boot - conditional_boot)
            / max(abs(global_boot), 1e-12)
        )

    parameter_rows = [
        {
            "model": global_fit["model"],
            "mark": None,
            "intercept_a": global_fit["fallback_a"],
            "shared_w": global_fit["w"],
            "fit_mean_nll": global_fit["fit_mean_nll"],
        },
        {
            "model": conditional_fit["model"],
            "mark": "fallback",
            "intercept_a": conditional_fit["fallback_a"],
            "shared_w": conditional_fit["w"],
            "fit_mean_nll": conditional_fit["fit_mean_nll"],
        },
    ]
    for mark, intercept in sorted(conditional_fit["mark_intercepts"].items()):
        parameter_rows.append(
            {
                "model": conditional_fit["model"],
                "mark": str(mark),
                "intercept_a": intercept,
                "shared_w": conditional_fit["w"],
                "fit_mean_nll": conditional_fit["fit_mean_nll"],
            }
        )

    summary = {
        "fit_ratio": FIT_RATIO,
        "fit_target_count": int(fit_frame.height),
        "eval_target_count": int(eval_frame.height),
        "eval_series_count": int(series.height),
        "global_fit_mean_nll": float(global_fit["fit_mean_nll"]),
        "mark_conditioned_fit_mean_nll": float(conditional_fit["fit_mean_nll"]),
        "global_eval_mean_nll": global_mean_nll,
        "mark_conditioned_eval_mean_nll": conditional_mean_nll,
        "eval_nll_improvement_pct": improvement_pct,
        "series_improved_share": series_improved_share,
        "series_nll_improvement_pct_median": float(series["nll_improvement_pct"].median()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_improvement_pct_ci_low": quantile(bootstrap_improvements, 0.025),
        "bootstrap_improvement_pct_ci_high": quantile(bootstrap_improvements, 0.975),
        "unseen_eval_mark_count": int(unseen_count),
        "global_w": float(global_fit["w"]),
        "mark_conditioned_w": float(conditional_fit["w"]),
        "global_w_at_upper_boundary": bool(global_fit["w_at_upper_boundary"]),
        "mark_conditioned_w_at_upper_boundary": bool(
            conditional_fit["w_at_upper_boundary"]
        ),
    }
    return summary, pl.DataFrame(parameter_rows), series


def evaluate_audit_gate(
    *,
    source_quality: dict[str, Any],
    loader_quality: dict[str, Any],
    effects: dict[str, Any],
    holdout: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "source_quality_pass": source_quality["quality_gate"] == "PASS",
        "loader_contract_pass": loader_quality["loader_contract_gate"] == "PASS",
        "at_least_two_supported_marks": effects["supported_mark_count"] >= 2,
        "eta_squared_at_least_0p01": effects["eta_squared_log1p_delta_t"]
        >= MIN_ETA_SQUARED,
        "eval_nll_improvement_at_least_0p5pct": holdout["eval_nll_improvement_pct"]
        >= MIN_EVAL_NLL_IMPROVEMENT_PCT,
        "bootstrap_ci_low_above_zero": holdout["bootstrap_improvement_pct_ci_low"]
        > MIN_BOOTSTRAP_CI_LOW_PCT,
        "series_improved_share_at_least_0p55": holdout["series_improved_share"]
        >= MIN_SERIES_IMPROVED_SHARE,
        "no_unseen_eval_marks": holdout["unseen_eval_mark_count"] == 0,
        "global_w_not_upper_boundary": not holdout["global_w_at_upper_boundary"],
        "conditional_w_not_upper_boundary": not holdout[
            "mark_conditioned_w_at_upper_boundary"
        ],
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "proceed_to_v4_constants_freeze_and_implementation"
            if passed
            else "stop_v4_and_revisit_time_conditioning_hypothesis"
        ),
        "checks": checks,
        "thresholds": {
            "min_supported_mark_count": MIN_SUPPORTED_MARK_COUNT,
            "min_supported_mark_share": MIN_SUPPORTED_MARK_SHARE,
            "min_eta_squared_log1p_delta_t": MIN_ETA_SQUARED,
            "min_eval_nll_improvement_pct": MIN_EVAL_NLL_IMPROVEMENT_PCT,
            "min_bootstrap_ci_low_pct": MIN_BOOTSTRAP_CI_LOW_PCT,
            "min_series_improved_share": MIN_SERIES_IMPROVED_SHARE,
            "w_grid": [W_MIN, W_MAX, W_GRID_SIZE],
            "wd_clamp": WD_CLAMP,
        },
    }


def plot_mark_delta_time(targets: pl.DataFrame, output_path: Path) -> None:
    marks = sorted(int(mark) for mark in targets["mark"].unique().to_list())
    values = [
        targets.filter(pl.col("mark") == mark)["log1p_delta_t"].to_numpy()
        for mark in marks
    ]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    boxes = ax.boxplot(values, tick_labels=[str(mark) for mark in marks], showfliers=False, patch_artist=True)
    for box in boxes["boxes"]:
        box.set_facecolor("#9CC5A1")
        box.set_edgecolor("#213547")
    ax.set_title("Taxi train-only log1p(delta-time) by next mark")
    ax.set_xlabel("Next mark")
    ax.set_ylabel("log1p(delta-time)")
    ax.grid(axis="y", color="#D8DED9", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_mark_support(mark_summary: pl.DataFrame, output_path: Path) -> None:
    marks = mark_summary["mark"].to_numpy()
    counts = mark_summary["count"].to_numpy()
    medians = mark_summary["delta_t_median"].to_numpy()
    fig, left = plt.subplots(figsize=(10, 5.8))
    left.bar(marks, counts, color="#D7A86E", edgecolor="#3A2F24", label="Target count")
    left.set_xlabel("Next mark")
    left.set_ylabel("Train target count")
    right = left.twinx()
    right.plot(marks, medians, color="#1D5D9B", marker="o", linewidth=2, label="Median dt")
    right.set_ylabel("Median delta-time")
    left.set_title("Taxi train-only next-mark support and median delta-time")
    left.grid(axis="y", color="#E3DED7", linewidth=0.7, alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_series_nll_delta(series: pl.DataFrame, output_path: Path) -> None:
    values = series["nll_improvement_pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.hist(values, bins=30, color="#86BBD8", edgecolor="#263238", linewidth=0.5)
    ax.axvline(0.0, color="#B23A48", linestyle="--", linewidth=1.5)
    ax.set_title("Per-series train-only RMTPP NLL improvement from mark conditioning")
    ax.set_xlabel("NLL improvement (%)")
    ax.set_ylabel("Series count")
    ax.grid(axis="y", color="#D7DCE2", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    *,
    source_path: Path,
    source_quality: dict[str, Any],
    loader_quality: dict[str, Any],
    mark_summary: pl.DataFrame,
    effects: dict[str, Any],
    holdout: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    lines = [
        "# Taxi Train-Only Mark And Delta-Time Dependence Audit",
        "",
        "## Scope",
        "",
        f"- Source: `{source_path}`",
        "- Source scope: fixed-split train parquet only",
        "- Target scope: `RMTPPWeekLookbackDataset`-compatible train next events",
        "- Validation/test target data read: `false`",
        "- Purpose: decide whether V4 mark-conditioned time intensity has enough train-only evidence to implement",
        "",
        "## Data Contract",
        "",
        f"- Train rows / series: `{source_quality['row_count']}` / `{source_quality['series_count']}`",
        f"- Train next-event targets: `{loader_quality['decoded_train_target_count']}`",
        f"- Source quality / loader contract: `{source_quality['quality_gate']}` / `{loader_quality['loader_contract_gate']}`",
        "",
        "## Mark-Wise Delta-Time",
        "",
        "| Mark | Count | Share | Series | Median dt | IQR dt | Mean log1p(dt) | Std log1p(dt) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in mark_summary.iter_rows(named=True):
        lines.append(
            f"| {int(row['mark'])} | {int(row['count'])} | {100*float(row['share']):.3f}% | "
            f"{int(row['series_count'])} | {float(row['delta_t_median']):.4f} | "
            f"{float(row['delta_t_p75']) - float(row['delta_t_p25']):.4f} | "
            f"{float(row['log1p_delta_t_mean']):.6f} | {float(row['log1p_delta_t_std']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Distribution Effect",
            "",
            f"- Supported marks: `{effects['supported_marks']}`",
            f"- eta-squared on log1p(dt): `{float(effects['eta_squared_log1p_delta_t']):.6f}`",
            f"- omega-squared on log1p(dt): `{float(effects['omega_squared_log1p_delta_t']):.6f}`",
            f"- Maximum pairwise standardized mean difference: `{float(effects['max_pairwise_standardized_log1p_mean_difference']):.6f}`",
            "",
            "## Train-Only Temporal Holdout",
            "",
            f"- Fit/eval targets: `{holdout['fit_target_count']}` / `{holdout['eval_target_count']}`",
            f"- Global / mark-conditioned eval NLL: `{float(holdout['global_eval_mean_nll']):.6f}` / `{float(holdout['mark_conditioned_eval_mean_nll']):.6f}`",
            f"- Mark-conditioned eval NLL improvement: `{float(holdout['eval_nll_improvement_pct']):.4f}%`",
            f"- Series-bootstrap 95% CI: `[{float(holdout['bootstrap_improvement_pct_ci_low']):.4f}%, {float(holdout['bootstrap_improvement_pct_ci_high']):.4f}%]`",
            f"- Improved-series share: `{100*float(holdout['series_improved_share']):.3f}%`",
            "",
            "## Gate",
            "",
            f"- Status: `{gate['status']}`",
            f"- Decision: `{gate['decision']}`",
            "",
        ]
    )
    for check, passed in gate["checks"].items():
        lines.append(f"- `{check}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This audit measures train-only conditional timing signal, not V4 predictive performance.",
            "- The intercept-only RMTPP is a low-capacity diagnostic nested inside the proposed V4 family.",
            "- True next marks are used only to score the conditional likelihood; deployment dt MAE still depends on predicted marks.",
            "- No fitted audit intercept or diagnostic `w` is transferred into V4 initialization.",
            "",
            "## Next",
            "",
            (
                "Freeze the V4 implementation constants and proceed to `time_head_mode` focused implementation."
                if gate["status"] == "PASS"
                else "Keep V4 unimplemented and revisit the time-conditioning hypothesis."
            ),
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
        "Starting Taxi train-only mark/time audit | server=%s tmux=%s dataset=%s",
        args.execution_server,
        args.tmux_session,
        dataset_path,
    )

    frame = pl.read_parquet(dataset_path).sort(["oper_part_no", "seq"])
    source_quality = validate_train_source(frame)
    targets, dataset = extract_loader_train_targets(
        frame,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
    )
    loader_quality = validate_loader_contract(frame, targets, dataset)
    partitioned = assign_temporal_audit_partition(targets)
    mark_summary = summarize_mark_delta_time(partitioned)
    effects = distribution_effect_summary(partitioned, mark_summary)
    holdout, parameters, series = compare_train_only_holdout(partitioned)
    gate = evaluate_audit_gate(
        source_quality=source_quality,
        loader_quality=loader_quality,
        effects=effects,
        holdout=holdout,
    )

    quality_rows = metric_rows("source", source_quality)
    quality_rows.extend(metric_rows("loader", loader_quality))
    pl.DataFrame(quality_rows).write_csv(data_dir / "data_quality_summary.csv")
    mark_summary.write_csv(data_dir / "mark_delta_time_summary.csv")
    pl.DataFrame(metric_rows("distribution", effects)).write_csv(
        data_dir / "distribution_effect_summary.csv"
    )
    pl.DataFrame(metric_rows("train_temporal_holdout", holdout)).write_csv(
        data_dir / "rmtpp_holdout_summary.csv"
    )
    parameters.write_csv(data_dir / "rmtpp_fitted_parameters.csv")
    series.write_csv(data_dir / "series_holdout_comparison.csv")
    partitioned.write_parquet(data_dir / "train_target_rows.parquet")
    pl.DataFrame(
        [
            {"check": check, "passed": passed}
            for check, passed in gate["checks"].items()
        ]
    ).write_csv(data_dir / "audit_gate.csv")

    plot_mark_delta_time(partitioned, plot_dir / "mark_log1p_delta_time_boxplot.png")
    plot_mark_support(mark_summary, plot_dir / "mark_support_and_median_delta_time.png")
    plot_series_nll_delta(series, plot_dir / "series_rmtpp_nll_improvement.png")

    report = build_report(
        source_path=dataset_path,
        source_quality=source_quality,
        loader_quality=loader_quality,
        mark_summary=mark_summary,
        effects=effects,
        holdout=holdout,
        gate=gate,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    summary = {
        "status": "completed",
        "source_quality": source_quality,
        "loader_contract": loader_quality,
        "distribution_effect": effects,
        "train_temporal_holdout": holdout,
        "audit_gate": gate,
        "held_out_target_data_read": False,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    finished_at = datetime.now(KST)
    manifest = {
        "status": "completed",
        "analysis": "taxi_train_only_next_mark_delta_time_dependence_audit",
        "started_at_kst": started_at.isoformat(),
        "finished_at_kst": finished_at.isoformat(),
        "execution_server": args.execution_server,
        "execution_host": platform.node(),
        "tmux_session": args.tmux_session,
        "python": sys.executable,
        "dataset_path": str(dataset_path),
        "dataset_file_sha256": sha256_file(dataset_path),
        "dataset_scope": "fixed_split_train_parquet_only",
        "held_out_target_data_read": False,
        "output_dir": str(output_dir),
        "lookback_weeks": int(args.lookback_weeks),
        "max_seq_len": int(args.max_seq_len),
        "target_scope": "RMTPPWeekLookbackDataset-compatible train next events",
        "decoded_target_sha256": frame_sha256(
            partitioned,
            ["oper_part_no", "target_seq", "mark", "delta_t", "audit_partition"],
        ),
        "audit_contract": {
            "fit_ratio": FIT_RATIO,
            "minimum_supported_mark_count": MIN_SUPPORTED_MARK_COUNT,
            "minimum_supported_mark_share": MIN_SUPPORTED_MARK_SHARE,
            "minimum_eta_squared": MIN_ETA_SQUARED,
            "minimum_eval_nll_improvement_pct": MIN_EVAL_NLL_IMPROVEMENT_PCT,
            "minimum_bootstrap_ci_low_pct": MIN_BOOTSTRAP_CI_LOW_PCT,
            "minimum_series_improved_share": MIN_SERIES_IMPROVED_SHARE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "w_grid": [W_MIN, W_MAX, W_GRID_SIZE],
            "wd_clamp": WD_CLAMP,
        },
        "decision": gate["decision"],
        "artifact_order": [
            "audit_manifest.json",
            "logs/audit.log",
            "audit_summary.json",
            "data/data_quality_summary.csv",
            "data/mark_delta_time_summary.csv",
            "data/distribution_effect_summary.csv",
            "data/rmtpp_holdout_summary.csv",
            "data/rmtpp_fitted_parameters.csv",
            "data/series_holdout_comparison.csv",
            "data/audit_gate.csv",
            "data/train_target_rows.parquet",
            "report.md",
            "plots/*.png",
        ],
        "not_applicable_artifacts": [
            "test_summary",
            "training_histories",
            "scale_wise_metrics",
        ],
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Completed Taxi mark/time audit | targets=%d eta2=%.6f eval_nll_improvement=%.4f%% gate=%s",
        int(targets.height),
        float(effects["eta_squared_log1p_delta_t"]),
        float(holdout["eval_nll_improvement_pct"]),
        gate["status"],
    )


def main() -> None:
    write_outputs(parse_args())


if __name__ == "__main__":
    main()
