#!/usr/bin/env python3
"""Audit Taxi train-only pre-window support and predictive signal for TitanTPP V6."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset


DEFAULT_DATASET = (
    PROJECT_ROOT / "sample_data/new_york_taxi/yellow_trip_hourly_train.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "search_artifacts/model_enhancement_titantpp_v6_taxi_train_memory_audit_0717"
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

# Frozen before the first Taxi audit result is read.
EXPECTED_SERIES_COUNT = 131
FIT_SHARE = 0.70
SELECTION_SHARE = 0.15
AUDIT_SHARE = 0.15
MEMORY_THRESHOLDS = (1, 8, 16, 32, 64)
MEMORY_ELIGIBILITY_COUNT = 8
MEMORY_BUDGETS = (16, 32, 64, 128)
RETRIEVAL_TOPKS = (4, 8)
QUERY_EVENT_COUNT = 8
MIN_TARGET_COVERAGE = 0.35
MIN_SERIES_COVERAGE = 0.80
MIN_PRIMARY_IMPROVEMENT_PCT = 1.0
MAX_OTHER_REGRESSION_PCT = 1.0
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42
PROBE_SEED = 42
LOGISTIC_C = 1.0
RIDGE_ALPHA = 1.0
METRIC_NAMES = ("marker_ce", "log1p_dt_mae", "log2_qty_mae")


@dataclass(frozen=True)
class SeriesData:
    part_index: int
    oper_part_no: str
    seq: np.ndarray
    delta_t: np.ndarray
    mark: np.ndarray
    quantity: np.ndarray


@dataclass
class ProbeModels:
    scaler: StandardScaler
    marker: LogisticRegression
    time: Ridge
    quantity: Ridge


@dataclass(frozen=True)
class ProbeMatrix:
    sample_index: np.ndarray
    oper_part_no: np.ndarray
    base_features: np.ndarray
    augmented_features: np.ndarray
    target_mark: np.ndarray
    target_log1p_dt: np.ndarray
    target_log2_qty: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--lookback-weeks",
        type=int,
        default=168,
        help="Legacy loader name; Taxi seq units are hourly buckets.",
    )
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--execution-server", default="unknown")
    parser.add_argument("--tmux-session", default="none")
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("SOURCE_REVISION", "unknown"),
    )
    return parser.parse_args()


def configure_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("taxi_pre_window_memory_audit")
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


def frame_sha256(frame: pl.DataFrame, columns: Sequence[str]) -> str:
    row_hashes = frame.select(list(columns)).hash_rows(seed=0).to_numpy()
    digest = hashlib.sha256()
    digest.update("|".join(columns).encode("utf-8"))
    digest.update(np.ascontiguousarray(row_hashes).tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, q)) if finite.size else float("nan")


def metric_rows(scope: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"scope": scope, "metric": key, "value": jsonable(value)}
        for key, value in metrics.items()
        if not isinstance(value, (dict, list, tuple))
    ]


def validate_train_source(
    frame: pl.DataFrame,
    *,
    expected_series_count: int | None = EXPECTED_SERIES_COUNT,
) -> dict[str, Any]:
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
    invalid_mark_rows = int(frame.filter(pl.col("mark").is_null() | (pl.col("mark") < 0)).height)
    non_positive_quantity_rows = int(frame.filter(pl.col("demand_qty") <= 0).height)
    non_finite_quantity_rows = int(frame.filter(~pl.col("demand_qty").is_finite()).height)
    source_non_positive_dt_rows = int(frame.filter(pl.col("delta_t") <= 0).height)
    row_count = int(frame.height)
    series_count = int(frame["oper_part_no"].n_unique())
    marks = sorted(int(mark) for mark in frame["mark"].unique().to_list())
    contiguous_marks = marks == list(range(max(marks) + 1)) if marks else False
    expected_series_match = expected_series_count is None or series_count == expected_series_count

    if null_cells:
        raise ValueError(f"Taxi train dataset has {null_cells} null required cells")
    if duplicate_rows:
        raise ValueError(f"Taxi train dataset has {duplicate_rows} duplicate part/seq rows")
    if non_train_rows:
        raise ValueError(f"Train-only audit received {non_train_rows} non-train rows")
    if invalid_mark_rows or not contiguous_marks:
        raise ValueError(
            "Taxi train marks must be non-negative and contiguous: "
            f"invalid_rows={invalid_mark_rows} marks={marks}"
        )
    if non_positive_quantity_rows or non_finite_quantity_rows:
        raise ValueError(
            "Taxi train quantity must be positive and finite: "
            f"non_positive={non_positive_quantity_rows} non_finite={non_finite_quantity_rows}"
        )
    if row_count == 0 or series_count == 0:
        raise ValueError("Taxi train dataset is empty")
    if not expected_series_match:
        raise ValueError(
            "Taxi train series count changed: "
            f"observed={series_count} expected={expected_series_count}"
        )

    return {
        "row_count": row_count,
        "series_count": series_count,
        "expected_series_count": expected_series_count,
        "expected_series_count_match": expected_series_match,
        "real_mark_count": len(marks),
        "real_marks": marks,
        "required_null_cells": null_cells,
        "duplicate_part_seq_rows": duplicate_rows,
        "non_train_rows": non_train_rows,
        "invalid_mark_rows": invalid_mark_rows,
        "non_positive_quantity_rows": non_positive_quantity_rows,
        "non_finite_quantity_rows": non_finite_quantity_rows,
        "source_non_positive_delta_t_rows": source_non_positive_dt_rows,
        "quality_gate": "PASS",
    }


def build_series_data(frame: pl.DataFrame) -> dict[int, SeriesData]:
    grouped = (
        frame.sort(["oper_part_no", "seq"])
        .group_by("oper_part_no", maintain_order=True)
        .agg(
            pl.col("seq").alias("seqs"),
            pl.col("delta_t").alias("delta_ts"),
            pl.col("mark").alias("marks"),
            pl.col("demand_qty").alias("quantities"),
        )
        .sort("oper_part_no")
    )
    result: dict[int, SeriesData] = {}
    for part_index, row in enumerate(grouped.iter_rows(named=True)):
        result[part_index] = SeriesData(
            part_index=part_index,
            oper_part_no=str(row["oper_part_no"]),
            seq=np.asarray(row["seqs"], dtype=np.int64),
            delta_t=np.maximum(np.asarray(row["delta_ts"], dtype=np.float64), 1.0),
            mark=np.asarray(row["marks"], dtype=np.int64),
            quantity=np.asarray(row["quantities"], dtype=np.float64),
        )
    return result


def build_train_samples(
    frame: pl.DataFrame,
    *,
    lookback_weeks: int,
    max_seq_len: int,
) -> tuple[pl.DataFrame, RMTPPWeekLookbackDataset, dict[int, SeriesData]]:
    if lookback_weeks < 1 or max_seq_len < 2:
        raise ValueError(
            "lookback_weeks must be positive and max_seq_len must include "
            "context plus target"
        )

    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback_weeks,
        max_seq_len=max_seq_len,
        mode="all",
        target_splits={"train"},
    )
    series_data = build_series_data(frame)
    max_context_count = max_seq_len - 1
    rows: list[dict[str, Any]] = []
    for sample_index, (part_index, context_end) in enumerate(dataset.index):
        series = series_data[int(part_index)]
        target_index = int(context_end + 1)
        left_seq = int(series.seq[context_end]) - (lookback_weeks - 1)
        temporal_start = int(np.searchsorted(series.seq, left_seq, side="left"))
        effective_start = max(temporal_start, int(context_end) - max_context_count + 1)
        memory_count = effective_start
        temporal_excluded_count = temporal_start
        max_seq_excluded_count = effective_start - temporal_start
        context_count = int(context_end) - effective_start + 1
        memory_span = (
            int(series.seq[effective_start - 1] - series.seq[0])
            if effective_start >= 2
            else 0
        )
        gap_to_context = (
            int(series.seq[effective_start] - series.seq[effective_start - 1])
            if effective_start >= 1
            else 0
        )
        rows.append(
            {
                "sample_index": sample_index,
                "part_index": int(part_index),
                "oper_part_no": series.oper_part_no,
                "context_end_index": int(context_end),
                "target_index": target_index,
                "temporal_context_start_index": temporal_start,
                "effective_context_start_index": effective_start,
                "target_seq": int(series.seq[target_index]),
                "context_end_seq": int(series.seq[context_end]),
                "context_count": context_count,
                "context_span": int(series.seq[context_end] - series.seq[effective_start]),
                "pre_window_count": memory_count,
                "pre_window_span": memory_span,
                "pre_window_gap_to_context": gap_to_context,
                "temporal_boundary_excluded_count": temporal_excluded_count,
                "max_seq_len_excluded_count": max_seq_excluded_count,
                "target_mark": int(series.mark[target_index]),
                "target_delta_t": float(series.delta_t[target_index]),
                "target_log1p_delta_t": float(np.log1p(series.delta_t[target_index])),
                "target_quantity": float(series.quantity[target_index]),
                "target_log2_quantity": float(np.log2(series.quantity[target_index])),
                "target_split": str(dataset.split_lists[part_index][target_index]),
            }
        )
    if not rows:
        raise ValueError("Taxi train-only audit decoded no next-event samples")
    return pl.DataFrame(rows).sort(["oper_part_no", "target_seq"]), dataset, series_data


def validate_loader_contract(
    frame: pl.DataFrame,
    samples: pl.DataFrame,
    dataset: RMTPPWeekLookbackDataset,
    series_data: dict[int, SeriesData],
) -> dict[str, Any]:
    expected_target_count = int(
        frame.group_by("oper_part_no")
        .len()
        .select((pl.col("len") - 1).clip(0, None).sum())
        .item()
        or 0
    )
    if samples.height != expected_target_count or len(dataset) != expected_target_count:
        raise ValueError(
            "Taxi train target count does not match loader contract: "
            f"samples={samples.height} loader={len(dataset)} expected={expected_target_count}"
        )

    target_mismatch_count = 0
    context_length_mismatch_count = 0
    ordering_violation_count = 0
    non_train_target_count = 0
    for row in samples.iter_rows(named=True):
        sample_index = int(row["sample_index"])
        part_index = int(row["part_index"])
        target_index = int(row["target_index"])
        context_start = int(row["effective_context_start_index"])
        context_end = int(row["context_end_index"])
        series = series_data[part_index]
        loader_sample = dataset[sample_index]
        valid_positions = np.flatnonzero(loader_sample["mask"].numpy())
        expected_valid_count = int(row["context_count"]) + 1
        context_length_mismatch_count += int(valid_positions.size != expected_valid_count)
        final_position = int(valid_positions[-1])
        target_mismatch_count += int(
            int(loader_sample["marks"][final_position].item()) != int(series.mark[target_index])
            or not math.isclose(
                float(loader_sample["dts"][final_position].item()),
                float(series.delta_t[target_index]),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )
        ordering_violation_count += int(
            not (0 <= context_start <= context_end < target_index < len(series.seq))
        )
        non_train_target_count += int(str(row["target_split"]) != "train")

    if target_mismatch_count or context_length_mismatch_count or ordering_violation_count:
        raise ValueError(
            "Taxi pre-window audit disagrees with RMTPPWeekLookbackDataset: "
            f"target={target_mismatch_count} context={context_length_mismatch_count} "
            f"ordering={ordering_violation_count}"
        )
    if non_train_target_count:
        raise ValueError(f"Decoded {non_train_target_count} non-train target samples")

    return {
        "expected_train_target_count": expected_target_count,
        "decoded_train_target_count": int(samples.height),
        "loader_train_target_count": int(len(dataset)),
        "target_mismatch_count": target_mismatch_count,
        "context_length_mismatch_count": context_length_mismatch_count,
        "causal_ordering_violation_count": ordering_violation_count,
        "non_train_target_count": non_train_target_count,
        "loader_contract_gate": "PASS",
    }


def assign_temporal_probe_partition(
    samples: pl.DataFrame,
    *,
    fit_share: float = FIT_SHARE,
    selection_share: float = SELECTION_SHARE,
) -> pl.DataFrame:
    if fit_share <= 0.0 or selection_share <= 0.0 or fit_share + selection_share >= 1.0:
        raise ValueError("fit and selection shares must be positive and leave an audit suffix")

    groups: list[pl.DataFrame] = []
    for group in samples.sort(["oper_part_no", "target_seq"]).partition_by(
        "oper_part_no", maintain_order=True
    ):
        count = group.height
        if count < 3:
            partitions = ["probe_fit"] * count
        else:
            fit_count = max(1, int(math.floor(count * fit_share)))
            selection_count = max(1, int(math.floor(count * selection_share)))
            if fit_count + selection_count >= count:
                fit_count = max(1, count - 2)
                selection_count = 1
            audit_count = count - fit_count - selection_count
            partitions = (
                ["probe_fit"] * fit_count
                + ["probe_selection"] * selection_count
                + ["probe_audit"] * audit_count
            )
        groups.append(
            group.with_columns(
                pl.Series("probe_partition", partitions, dtype=pl.String),
                pl.Series("series_target_index", np.arange(count), dtype=pl.Int64),
            )
        )
    result = pl.concat(groups, how="vertical")
    counts = result.group_by("probe_partition").len()
    observed = set(counts["probe_partition"].to_list())
    required = {"probe_fit", "probe_selection", "probe_audit"}
    if not required.issubset(observed):
        raise ValueError(
            f"Temporal probe split is missing partitions: {sorted(required - observed)}"
        )
    return result.sort(["oper_part_no", "target_seq"])


def summarize_coverage(
    samples: pl.DataFrame,
    *,
    series_count: int,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    memory_counts = samples["pre_window_count"].to_numpy().astype(np.float64)
    memory_spans = samples["pre_window_span"].to_numpy().astype(np.float64)
    coverage_rows: list[dict[str, Any]] = []
    for threshold in MEMORY_THRESHOLDS:
        covered = samples.filter(pl.col("pre_window_count") >= threshold)
        covered_series = int(covered["oper_part_no"].n_unique()) if covered.height else 0
        coverage_rows.append(
            {
                "threshold": threshold,
                "target_count": int(samples.height),
                "covered_target_count": int(covered.height),
                "covered_target_share": float(covered.height / samples.height),
                "covered_series_count": covered_series,
                "covered_series_share": float(covered_series / max(series_count, 1)),
            }
        )
    coverage = pl.DataFrame(coverage_rows).sort("threshold")

    series_rows: list[dict[str, Any]] = []
    for group in samples.partition_by("oper_part_no", maintain_order=False):
        counts = group["pre_window_count"].to_numpy().astype(np.float64)
        row: dict[str, Any] = {
            "oper_part_no": str(group["oper_part_no"][0]),
            "target_count": int(group.height),
            "pre_window_count_mean": float(counts.mean()),
            "pre_window_count_median": quantile(counts, 0.5),
            "pre_window_count_p90": quantile(counts, 0.9),
            "pre_window_count_max": int(counts.max()),
        }
        for threshold in MEMORY_THRESHOLDS:
            row[f"target_share_ge_{threshold}"] = float(np.mean(counts >= threshold))
        series_rows.append(row)
    by_series = pl.DataFrame(series_rows).sort("oper_part_no")

    mark_rows: list[dict[str, Any]] = []
    for group in samples.partition_by("target_mark", maintain_order=False):
        counts = group["pre_window_count"].to_numpy().astype(np.float64)
        row = {
            "target_mark": int(group["target_mark"][0]),
            "target_count": int(group.height),
            "series_count": int(group["oper_part_no"].n_unique()),
            "pre_window_count_mean": float(counts.mean()),
            "pre_window_count_median": quantile(counts, 0.5),
        }
        for threshold in MEMORY_THRESHOLDS:
            row[f"target_share_ge_{threshold}"] = float(np.mean(counts >= threshold))
        mark_rows.append(row)
    by_mark = pl.DataFrame(mark_rows).sort("target_mark")

    temporal_excluded = int(samples["temporal_boundary_excluded_count"].sum())
    max_seq_excluded = int(samples["max_seq_len_excluded_count"].sum())
    total_excluded = temporal_excluded + max_seq_excluded
    ge8 = coverage.filter(pl.col("threshold") == MEMORY_ELIGIBILITY_COUNT).row(0, named=True)
    summary = {
        "target_count": int(samples.height),
        "series_count": int(series_count),
        "pre_window_count_min": int(memory_counts.min()),
        "pre_window_count_median": quantile(memory_counts, 0.5),
        "pre_window_count_p90": quantile(memory_counts, 0.9),
        "pre_window_count_p95": quantile(memory_counts, 0.95),
        "pre_window_count_max": int(memory_counts.max()),
        "pre_window_span_median": quantile(memory_spans, 0.5),
        "pre_window_span_p90": quantile(memory_spans, 0.9),
        "pre_window_span_p95": quantile(memory_spans, 0.95),
        "eligible_target_count_ge_8": int(ge8["covered_target_count"]),
        "eligible_target_share_ge_8": float(ge8["covered_target_share"]),
        "eligible_series_count_ge_8": int(ge8["covered_series_count"]),
        "eligible_series_share_ge_8": float(ge8["covered_series_share"]),
        "temporal_boundary_excluded_event_total": temporal_excluded,
        "max_seq_len_excluded_event_total": max_seq_excluded,
        "temporal_boundary_excluded_share": float(temporal_excluded / max(total_excluded, 1)),
        "max_seq_len_excluded_share": float(max_seq_excluded / max(total_excluded, 1)),
    }
    return summary, coverage, by_series, by_mark


def _event_features(series: SeriesData, num_marks: int) -> np.ndarray:
    one_hot = np.eye(num_marks, dtype=np.float64)[series.mark]
    log_dt = np.log1p(series.delta_t)[:, None]
    log_qty = np.log2(series.quantity)[:, None]
    seq = series.seq.astype(np.float64)
    phase = np.column_stack(
        [
            np.sin(2.0 * np.pi * seq / 24.0),
            np.cos(2.0 * np.pi * seq / 24.0),
            np.sin(2.0 * np.pi * seq / 168.0),
            np.cos(2.0 * np.pi * seq / 168.0),
        ]
    )
    return np.column_stack([one_hot, log_dt, log_qty, phase])


def fit_event_scaler(
    samples: pl.DataFrame,
    series_data: dict[int, SeriesData],
    *,
    partitions: Iterable[str],
    num_marks: int,
) -> StandardScaler:
    allowed = set(partitions)
    selected = samples.filter(pl.col("probe_partition").is_in(sorted(allowed)))
    prefixes: list[np.ndarray] = []
    for group in selected.partition_by("part_index", maintain_order=False):
        part_index = int(group["part_index"][0])
        last_observed = int(group["context_end_index"].max())
        prefixes.append(_event_features(series_data[part_index], num_marks)[: last_observed + 1])
    if not prefixes:
        raise ValueError(f"No event prefixes are available for partitions={sorted(allowed)}")
    return StandardScaler().fit(np.concatenate(prefixes, axis=0))


def _context_features(series: SeriesData, start: int, end: int, num_marks: int) -> np.ndarray:
    indices = np.arange(start, end + 1, dtype=np.int64)
    marks = series.mark[indices]
    mark_histogram = np.bincount(marks, minlength=num_marks).astype(np.float64)
    mark_histogram /= max(indices.size, 1)
    last_mark = np.eye(num_marks, dtype=np.float64)[marks[-1]]
    log_dt = np.log1p(series.delta_t[indices])
    log_qty = np.log2(series.quantity[indices])

    def numeric_summary(values: np.ndarray) -> list[float]:
        return [
            float(values.mean()),
            float(values.std()),
            float(values[-1]),
            float(values[-1] - values[0]),
        ]

    context_seq = series.seq[indices]
    end_seq = float(context_seq[-1])
    metadata = [
        float(np.log1p(indices.size)),
        float(np.log1p(context_seq[-1] - context_seq[0])),
        float(np.sin(2.0 * np.pi * end_seq / 24.0)),
        float(np.cos(2.0 * np.pi * end_seq / 24.0)),
        float(np.sin(2.0 * np.pi * end_seq / 168.0)),
        float(np.cos(2.0 * np.pi * end_seq / 168.0)),
    ]
    return np.concatenate(
        [mark_histogram, last_mark, numeric_summary(log_dt), numeric_summary(log_qty), metadata]
    )


def retrieve_memory_features(
    series: SeriesData,
    *,
    context_start: int,
    context_end: int,
    memory_budget: int,
    topk: int,
    num_marks: int,
    event_scaler: StandardScaler,
) -> np.ndarray:
    if not (0 < topk <= memory_budget):
        raise ValueError("topk must be positive and no larger than memory_budget")
    if context_start < topk:
        raise ValueError(
            f"Sample has only {context_start} pre-window events but topk={topk} was requested"
        )
    candidate_start = max(0, context_start - memory_budget)
    candidate_indices = np.arange(candidate_start, context_start, dtype=np.int64)
    query_start = max(context_start, context_end - QUERY_EVENT_COUNT + 1)
    query_indices = np.arange(query_start, context_end + 1, dtype=np.int64)
    raw_events = _event_features(series, num_marks)
    candidate_keys = event_scaler.transform(raw_events[candidate_indices])
    query = event_scaler.transform(raw_events[query_indices]).mean(axis=0)
    candidate_norm = np.linalg.norm(candidate_keys, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominator = np.maximum(candidate_norm * max(query_norm, 1e-12), 1e-12)
    similarities = candidate_keys @ query / denominator
    # Similarity is primary; a newer event wins exact ties.
    order = np.lexsort((-candidate_indices, -similarities))
    selected_local = order[:topk]
    selected_indices = candidate_indices[selected_local]
    selected_keys = candidate_keys[selected_local]
    selected_similarity = similarities[selected_local]
    ages = np.log1p(series.seq[context_end] - series.seq[selected_indices]).astype(np.float64)
    return np.concatenate(
        [
            selected_keys.mean(axis=0),
            selected_keys.std(axis=0),
            np.asarray(
                [
                    float(selected_similarity.max()),
                    float(selected_similarity.mean()),
                    float(selected_similarity.min()),
                    float(np.log1p(candidate_indices.size)),
                    float(ages.min()),
                    float(ages.mean()),
                    float(ages.max()),
                ],
                dtype=np.float64,
            ),
        ]
    )


def build_probe_matrix(
    samples: pl.DataFrame,
    series_data: dict[int, SeriesData],
    *,
    partitions: Iterable[str],
    memory_budget: int,
    topk: int,
    num_marks: int,
    event_scaler: StandardScaler,
) -> ProbeMatrix:
    allowed = set(partitions)
    selected = samples.filter(
        pl.col("probe_partition").is_in(sorted(allowed))
        & (pl.col("pre_window_count") >= MEMORY_ELIGIBILITY_COUNT)
    ).sort(["oper_part_no", "target_seq"])
    if selected.height == 0:
        raise ValueError(f"No memory-eligible samples for partitions={sorted(allowed)}")

    sample_indices: list[int] = []
    parts: list[str] = []
    base_rows: list[np.ndarray] = []
    augmented_rows: list[np.ndarray] = []
    marks: list[int] = []
    times: list[float] = []
    quantities: list[float] = []
    for row in selected.iter_rows(named=True):
        part_index = int(row["part_index"])
        context_start = int(row["effective_context_start_index"])
        context_end = int(row["context_end_index"])
        target_index = int(row["target_index"])
        series = series_data[part_index]
        if not (context_start <= context_end < target_index):
            raise ValueError("Probe sample violates context/target ordering")
        base = _context_features(series, context_start, context_end, num_marks)
        memory = retrieve_memory_features(
            series,
            context_start=context_start,
            context_end=context_end,
            memory_budget=memory_budget,
            topk=topk,
            num_marks=num_marks,
            event_scaler=event_scaler,
        )
        sample_indices.append(int(row["sample_index"]))
        parts.append(str(row["oper_part_no"]))
        base_rows.append(base)
        augmented_rows.append(np.concatenate([base, memory]))
        marks.append(int(series.mark[target_index]))
        times.append(float(np.log1p(series.delta_t[target_index])))
        quantities.append(float(np.log2(series.quantity[target_index])))

    return ProbeMatrix(
        sample_index=np.asarray(sample_indices, dtype=np.int64),
        oper_part_no=np.asarray(parts, dtype=object),
        base_features=np.vstack(base_rows),
        augmented_features=np.vstack(augmented_rows),
        target_mark=np.asarray(marks, dtype=np.int64),
        target_log1p_dt=np.asarray(times, dtype=np.float64),
        target_log2_qty=np.asarray(quantities, dtype=np.float64),
    )


def fit_probe_models(features: np.ndarray, targets: ProbeMatrix) -> ProbeModels:
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    marker = LogisticRegression(
        C=LOGISTIC_C,
        solver="lbfgs",
        max_iter=1_000,
        tol=1e-8,
        random_state=PROBE_SEED,
    ).fit(transformed, targets.target_mark)
    time = Ridge(alpha=RIDGE_ALPHA, solver="cholesky").fit(
        transformed, targets.target_log1p_dt
    )
    quantity = Ridge(alpha=RIDGE_ALPHA, solver="cholesky").fit(
        transformed, targets.target_log2_qty
    )
    return ProbeModels(scaler=scaler, marker=marker, time=time, quantity=quantity)


def score_probe_models(
    models: ProbeModels,
    features: np.ndarray,
    targets: ProbeMatrix,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    transformed = models.scaler.transform(features)
    probabilities = models.marker.predict_proba(transformed)
    class_to_column = {int(mark): index for index, mark in enumerate(models.marker.classes_)}
    true_probability = np.full(targets.target_mark.shape, 1e-12, dtype=np.float64)
    unseen_mark_count = 0
    for index, mark in enumerate(targets.target_mark):
        column = class_to_column.get(int(mark))
        if column is None:
            unseen_mark_count += 1
        else:
            true_probability[index] = max(float(probabilities[index, column]), 1e-12)
    marker_error = -np.log(true_probability)
    time_error = np.abs(models.time.predict(transformed) - targets.target_log1p_dt)
    quantity_error = np.abs(models.quantity.predict(transformed) - targets.target_log2_qty)
    errors = {
        "marker_ce": marker_error,
        "log1p_dt_mae": time_error,
        "log2_qty_mae": quantity_error,
    }
    if not all(np.isfinite(values).all() for values in errors.values()):
        raise ValueError("Probe errors must be finite")
    metrics: dict[str, float | int] = {
        name: float(values.mean()) for name, values in errors.items()
    }
    metrics["unseen_mark_count"] = unseen_mark_count
    return metrics, errors


def improvement_pct(baseline: float, augmented: float) -> float:
    return 100.0 * (baseline - augmented) / max(abs(baseline), 1e-12)


def select_candidate(rows: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    passed = [row for row in rows if bool(row["selection_pass"])]
    if not passed:
        return None
    return sorted(
        passed,
        key=lambda row: (
            -float(row["worst_improvement_pct"]),
            -float(row["mean_improvement_pct"]),
            int(row["memory_budget"]),
            int(row["topk"]),
        ),
    )[0]


def evaluate_candidate_grid(
    samples: pl.DataFrame,
    series_data: dict[int, SeriesData],
    *,
    num_marks: int,
) -> tuple[pl.DataFrame, dict[str, Any] | None]:
    event_scaler = fit_event_scaler(
        samples,
        series_data,
        partitions={"probe_fit"},
        num_marks=num_marks,
    )
    baseline_fit: ProbeMatrix | None = None
    baseline_selection: ProbeMatrix | None = None
    baseline_models: ProbeModels | None = None
    baseline_metrics: dict[str, float | int] | None = None
    rows: list[dict[str, Any]] = []

    for memory_budget in MEMORY_BUDGETS:
        for topk in RETRIEVAL_TOPKS:
            fit_matrix = build_probe_matrix(
                samples,
                series_data,
                partitions={"probe_fit"},
                memory_budget=memory_budget,
                topk=topk,
                num_marks=num_marks,
                event_scaler=event_scaler,
            )
            selection_matrix = build_probe_matrix(
                samples,
                series_data,
                partitions={"probe_selection"},
                memory_budget=memory_budget,
                topk=topk,
                num_marks=num_marks,
                event_scaler=event_scaler,
            )
            if baseline_fit is None:
                baseline_fit = fit_matrix
                baseline_selection = selection_matrix
                baseline_models = fit_probe_models(fit_matrix.base_features, fit_matrix)
                baseline_metrics, _ = score_probe_models(
                    baseline_models,
                    selection_matrix.base_features,
                    selection_matrix,
                )
            else:
                if not np.array_equal(fit_matrix.sample_index, baseline_fit.sample_index):
                    raise ValueError("Candidate grid changed probe_fit sample identity")
                if not np.array_equal(
                    selection_matrix.sample_index, baseline_selection.sample_index
                ):
                    raise ValueError("Candidate grid changed probe_selection sample identity")

            augmented_models = fit_probe_models(fit_matrix.augmented_features, fit_matrix)
            augmented_metrics, _ = score_probe_models(
                augmented_models,
                selection_matrix.augmented_features,
                selection_matrix,
            )
            assert baseline_metrics is not None
            improvements = {
                name: improvement_pct(
                    float(baseline_metrics[name]), float(augmented_metrics[name])
                )
                for name in METRIC_NAMES
            }
            selection_pass = (
                max(improvements.values()) >= MIN_PRIMARY_IMPROVEMENT_PCT
                and min(improvements.values()) >= -MAX_OTHER_REGRESSION_PCT
                and int(augmented_metrics["unseen_mark_count"]) == 0
            )
            primary_metric = max(
                METRIC_NAMES,
                key=lambda name: (improvements[name], -METRIC_NAMES.index(name)),
            )
            row: dict[str, Any] = {
                "memory_budget": memory_budget,
                "topk": topk,
                "fit_target_count": int(fit_matrix.sample_index.size),
                "selection_target_count": int(selection_matrix.sample_index.size),
                "selection_pass": selection_pass,
                "primary_metric": primary_metric,
                "worst_improvement_pct": min(improvements.values()),
                "mean_improvement_pct": float(np.mean(list(improvements.values()))),
            }
            for name in METRIC_NAMES:
                row[f"baseline_{name}"] = float(baseline_metrics[name])
                row[f"augmented_{name}"] = float(augmented_metrics[name])
                row[f"{name}_improvement_pct"] = improvements[name]
            rows.append(row)

    candidates = pl.DataFrame(rows).sort(["memory_budget", "topk"])
    return candidates, select_candidate(rows)


def bootstrap_metric_improvements(
    parts: np.ndarray,
    baseline_errors: dict[str, np.ndarray],
    augmented_errors: dict[str, np.ndarray],
) -> tuple[dict[str, Any], pl.DataFrame]:
    unique_parts = np.asarray(sorted(set(str(part) for part in parts)), dtype=object)
    if unique_parts.size == 0:
        raise ValueError("Series bootstrap requires at least one series")
    series_rows: list[dict[str, Any]] = []
    for part in unique_parts:
        mask = parts == part
        row: dict[str, Any] = {"oper_part_no": str(part), "target_count": int(mask.sum())}
        for name in METRIC_NAMES:
            baseline_mean = float(baseline_errors[name][mask].mean())
            augmented_mean = float(augmented_errors[name][mask].mean())
            row[f"baseline_{name}"] = baseline_mean
            row[f"augmented_{name}"] = augmented_mean
            row[f"{name}_improvement_pct"] = improvement_pct(
                baseline_mean, augmented_mean
            )
            row[f"baseline_{name}_sum"] = float(baseline_errors[name][mask].sum())
            row[f"augmented_{name}_sum"] = float(augmented_errors[name][mask].sum())
        series_rows.append(row)
    series_frame = pl.DataFrame(series_rows).sort("oper_part_no")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary: dict[str, Any] = {
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "audit_series_count": int(unique_parts.size),
    }
    counts = series_frame["target_count"].to_numpy().astype(np.float64)
    for name in METRIC_NAMES:
        baseline_sums = series_frame[f"baseline_{name}_sum"].to_numpy().astype(np.float64)
        augmented_sums = series_frame[f"augmented_{name}_sum"].to_numpy().astype(np.float64)
        replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
        for replicate in range(BOOTSTRAP_REPLICATES):
            indices = rng.integers(0, unique_parts.size, size=unique_parts.size)
            denominator = float(counts[indices].sum())
            baseline_mean = float(baseline_sums[indices].sum() / denominator)
            augmented_mean = float(augmented_sums[indices].sum() / denominator)
            replicates[replicate] = improvement_pct(baseline_mean, augmented_mean)
        summary[f"{name}_bootstrap_ci_low"] = quantile(replicates, 0.025)
        summary[f"{name}_bootstrap_ci_high"] = quantile(replicates, 0.975)
        summary[f"{name}_series_improved_share"] = float(
            np.mean(series_frame[f"{name}_improvement_pct"].to_numpy() > 0.0)
        )
    return summary, series_frame


def evaluate_final_audit(
    samples: pl.DataFrame,
    series_data: dict[int, SeriesData],
    *,
    num_marks: int,
    selected: dict[str, Any],
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame]:
    memory_budget = int(selected["memory_budget"])
    topk = int(selected["topk"])
    primary_metric = str(selected["primary_metric"])
    event_scaler = fit_event_scaler(
        samples,
        series_data,
        partitions={"probe_fit", "probe_selection"},
        num_marks=num_marks,
    )
    train_matrix = build_probe_matrix(
        samples,
        series_data,
        partitions={"probe_fit", "probe_selection"},
        memory_budget=memory_budget,
        topk=topk,
        num_marks=num_marks,
        event_scaler=event_scaler,
    )
    audit_matrix = build_probe_matrix(
        samples,
        series_data,
        partitions={"probe_audit"},
        memory_budget=memory_budget,
        topk=topk,
        num_marks=num_marks,
        event_scaler=event_scaler,
    )
    baseline_models = fit_probe_models(train_matrix.base_features, train_matrix)
    augmented_models = fit_probe_models(train_matrix.augmented_features, train_matrix)
    baseline_metrics, baseline_errors = score_probe_models(
        baseline_models, audit_matrix.base_features, audit_matrix
    )
    augmented_metrics, augmented_errors = score_probe_models(
        augmented_models, audit_matrix.augmented_features, audit_matrix
    )
    bootstrap, series = bootstrap_metric_improvements(
        audit_matrix.oper_part_no, baseline_errors, augmented_errors
    )

    scored = pl.DataFrame(
        {
            "sample_index": audit_matrix.sample_index,
            "oper_part_no": audit_matrix.oper_part_no,
            "target_mark": audit_matrix.target_mark,
            "target_log1p_delta_t": audit_matrix.target_log1p_dt,
            "target_log2_quantity": audit_matrix.target_log2_qty,
        }
    )
    summary: dict[str, Any] = {
        "memory_budget": memory_budget,
        "topk": topk,
        "primary_metric": primary_metric,
        "probe_train_target_count": int(train_matrix.sample_index.size),
        "probe_audit_target_count": int(audit_matrix.sample_index.size),
        "probe_audit_series_count": int(np.unique(audit_matrix.oper_part_no).size),
        "baseline_unseen_mark_count": int(baseline_metrics["unseen_mark_count"]),
        "augmented_unseen_mark_count": int(augmented_metrics["unseen_mark_count"]),
    }
    for name in METRIC_NAMES:
        summary[f"baseline_{name}"] = float(baseline_metrics[name])
        summary[f"augmented_{name}"] = float(augmented_metrics[name])
        summary[f"{name}_improvement_pct"] = improvement_pct(
            float(baseline_metrics[name]), float(augmented_metrics[name])
        )
        scored = scored.with_columns(
            pl.Series(f"baseline_{name}", baseline_errors[name]),
            pl.Series(f"augmented_{name}", augmented_errors[name]),
        )
    summary.update(bootstrap)
    return summary, series, scored


def evaluate_audit_gate(
    *,
    source_quality: dict[str, Any],
    loader_quality: dict[str, Any],
    coverage_summary: dict[str, Any],
    selected_candidate: dict[str, Any] | None,
    final_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate_selected = selected_candidate is not None
    primary_metric = (
        str(selected_candidate["primary_metric"]) if selected_candidate is not None else None
    )
    primary_improvement = (
        float(final_audit[f"{primary_metric}_improvement_pct"])
        if final_audit is not None and primary_metric is not None
        else float("-inf")
    )
    primary_ci_low = (
        float(final_audit[f"{primary_metric}_bootstrap_ci_low"])
        if final_audit is not None and primary_metric is not None
        else float("-inf")
    )
    worst_improvement = (
        min(float(final_audit[f"{name}_improvement_pct"]) for name in METRIC_NAMES)
        if final_audit is not None
        else float("-inf")
    )
    unseen_marks = (
        int(final_audit["baseline_unseen_mark_count"])
        + int(final_audit["augmented_unseen_mark_count"])
        if final_audit is not None
        else -1
    )
    checks = {
        "source_quality_pass": source_quality["quality_gate"] == "PASS",
        "loader_contract_pass": loader_quality["loader_contract_gate"] == "PASS",
        "target_coverage_ge_8_at_least_0p35": float(
            coverage_summary["eligible_target_share_ge_8"]
        )
        >= MIN_TARGET_COVERAGE,
        "series_coverage_ge_8_at_least_0p80": float(
            coverage_summary["eligible_series_share_ge_8"]
        )
        >= MIN_SERIES_COVERAGE,
        "selection_candidate_passed": candidate_selected,
        "final_primary_improvement_at_least_1pct": primary_improvement
        >= MIN_PRIMARY_IMPROVEMENT_PCT,
        "final_other_metrics_regress_at_most_1pct": worst_improvement
        >= -MAX_OTHER_REGRESSION_PCT,
        "primary_series_bootstrap_ci_low_above_zero": primary_ci_low > 0.0,
        "no_unseen_audit_marks": unseen_marks == 0,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "freeze_v6_memory_constants_and_open_adapter_implementation"
            if passed
            else "close_v6_before_model_implementation"
        ),
        "checks": checks,
        "selected_primary_metric": primary_metric,
        "thresholds": {
            "memory_eligibility_count": MEMORY_ELIGIBILITY_COUNT,
            "minimum_target_coverage": MIN_TARGET_COVERAGE,
            "minimum_series_coverage": MIN_SERIES_COVERAGE,
            "minimum_primary_improvement_pct": MIN_PRIMARY_IMPROVEMENT_PCT,
            "maximum_other_metric_regression_pct": MAX_OTHER_REGRESSION_PCT,
            "bootstrap_ci_low_must_exceed": 0.0,
        },
    }


def plot_coverage(coverage: pl.DataFrame, output_path: Path) -> None:
    thresholds = coverage["threshold"].to_numpy()
    target_share = 100.0 * coverage["covered_target_share"].to_numpy()
    series_share = 100.0 * coverage["covered_series_share"].to_numpy()
    positions = np.arange(thresholds.size)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(positions - width / 2, target_share, width, color="#2A6F97", label="Targets")
    ax.bar(positions + width / 2, series_share, width, color="#E09F3E", label="Series")
    ax.axhline(100.0 * MIN_TARGET_COVERAGE, color="#2A6F97", linestyle="--", linewidth=1)
    ax.axhline(100.0 * MIN_SERIES_COVERAGE, color="#E09F3E", linestyle="--", linewidth=1)
    ax.set_xticks(positions, [f">={int(value)}" for value in thresholds])
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("Available pre-window events")
    ax.set_title("Taxi train-only causal pre-window coverage")
    ax.legend()
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_grid(candidates: pl.DataFrame, output_path: Path) -> None:
    labels = [
        f"M{int(row['memory_budget'])}/k{int(row['topk'])}"
        for row in candidates.iter_rows(named=True)
    ]
    positions = np.arange(len(labels))
    width = 0.25
    colors = ("#335C67", "#E09F3E", "#9E2A2B")
    fig, ax = plt.subplots(figsize=(12, 6.2))
    for index, (metric, color) in enumerate(zip(METRIC_NAMES, colors)):
        values = candidates[f"{metric}_improvement_pct"].to_numpy()
        ax.bar(positions + (index - 1) * width, values, width, label=metric, color=color)
    ax.axhline(1.0, color="#1B4332", linestyle="--", linewidth=1.2)
    ax.axhline(-1.0, color="#7F1D1D", linestyle="--", linewidth=1.2)
    ax.set_xticks(positions, labels, rotation=30, ha="right")
    ax.set_ylabel("Selection improvement (%)")
    ax.set_title("Train-internal V6 memory proxy candidate screen")
    ax.legend()
    ax.grid(axis="y", color="#E1E4E8", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_series(series: pl.DataFrame, primary_metric: str, output_path: Path) -> None:
    values = series[f"{primary_metric}_improvement_pct"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.hist(values, bins=30, color="#7CB518", edgecolor="#263238", linewidth=0.5)
    ax.axvline(0.0, color="#9E2A2B", linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"{primary_metric} improvement (%)")
    ax.set_ylabel("Series count")
    ax.set_title("Per-series final train-only memory-proxy improvement")
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    *,
    source_path: Path,
    source_quality: dict[str, Any],
    coverage_summary: dict[str, Any],
    selected_candidate: dict[str, Any] | None,
    final_audit: dict[str, Any] | None,
    gate: dict[str, Any],
) -> str:
    lines = [
        "# Taxi Train-Only Pre-Window Memory Audit",
        "",
        "## Scope",
        "",
        f"- Source: `{source_path}`",
        "- Source scope: fixed-split Taxi train parquet only",
        "- Validation/test target data read: `false`",
        "- Purpose: decide whether V6 causal series memory is worth implementing",
        "",
        "## Coverage",
        "",
        "- Train rows / series / targets: "
        f"`{source_quality['row_count']}` / `{source_quality['series_count']}` / "
        f"`{coverage_summary['target_count']}`",
        "- Targets with at least 8 pre-window events: "
        f"`{100.0*float(coverage_summary['eligible_target_share_ge_8']):.3f}%`",
        "- Series with an eligible target: "
        f"`{100.0*float(coverage_summary['eligible_series_share_ge_8']):.3f}%`",
        "- Pre-window count median / p90 / p95: "
        f"`{float(coverage_summary['pre_window_count_median']):.2f}` / "
        f"`{float(coverage_summary['pre_window_count_p90']):.2f}` / "
        f"`{float(coverage_summary['pre_window_count_p95']):.2f}`",
        "- Temporal-boundary / max-seq exclusion share: "
        f"`{100.0*float(coverage_summary['temporal_boundary_excluded_share']):.3f}%` / "
        f"`{100.0*float(coverage_summary['max_seq_len_excluded_share']):.3f}%`",
        "",
        "## Candidate Selection",
        "",
    ]
    if selected_candidate is None:
        lines.append("- No `M/topk` candidate passed the train-internal selection guard.")
    else:
        lines.extend(
            [
                f"- Selected: `M={int(selected_candidate['memory_budget'])}`, "
                f"`topk={int(selected_candidate['topk'])}`",
                f"- Predeclared final primary metric: `{selected_candidate['primary_metric']}`",
                "- Selection worst / mean improvement: "
                f"`{float(selected_candidate['worst_improvement_pct']):.4f}%` / "
                f"`{float(selected_candidate['mean_improvement_pct']):.4f}%`",
            ]
        )
    lines.extend(["", "## Final Train-Only Audit", ""])
    if final_audit is None:
        lines.append("- Final audit was not opened because no candidate passed selection.")
    else:
        for name in METRIC_NAMES:
            lines.append(
                f"- `{name}` baseline / augmented / improvement / 95% CI: "
                f"`{float(final_audit[f'baseline_{name}']):.6f}` / "
                f"`{float(final_audit[f'augmented_{name}']):.6f}` / "
                f"`{float(final_audit[f'{name}_improvement_pct']):.4f}%` / "
                f"`[{float(final_audit[f'{name}_bootstrap_ci_low']):.4f}%, "
                f"{float(final_audit[f'{name}_bootstrap_ci_high']):.4f}%]`"
            )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Status: `{gate['status']}`",
            f"- Decision: `{gate['decision']}`",
        ]
    )
    for check, passed in gate["checks"].items():
        lines.append(f"- `{check}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This audit tests support and incremental train-only signal; it is "
            "not TitanTPP model-quality evidence.",
            "- The linear probes use the same eligible targets and differ only "
            "by causal pre-window summaries.",
            "- `M/topk` selection uses the middle train suffix; the last train "
            "suffix is evaluated once.",
            "- No probe parameter or fitted statistic is transferred into TitanTPP.",
            "",
            "## Next",
            "",
            (
                "Freeze the selected memory constants and implement the masked "
                "zero-init V6 adapter."
                if gate["status"] == "PASS"
                else "Retain Taxi V3b and close V6 before model implementation."
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
        "Starting Taxi train-only pre-window audit | server=%s tmux=%s dataset=%s",
        args.execution_server,
        args.tmux_session,
        dataset_path,
    )

    frame = pl.read_parquet(dataset_path).sort(["oper_part_no", "seq"])
    source_quality = validate_train_source(frame)
    samples, dataset, series_data = build_train_samples(
        frame,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
    )
    loader_quality = validate_loader_contract(frame, samples, dataset, series_data)
    samples = assign_temporal_probe_partition(samples)
    coverage_summary, coverage, by_series, by_mark = summarize_coverage(
        samples, series_count=int(source_quality["series_count"])
    )
    logger.info(
        "Coverage decoded | targets=%d ge8=%.4f series_ge8=%.4f",
        int(samples.height),
        float(coverage_summary["eligible_target_share_ge_8"]),
        float(coverage_summary["eligible_series_share_ge_8"]),
    )

    candidates, selected_candidate = evaluate_candidate_grid(
        samples,
        series_data,
        num_marks=int(source_quality["real_mark_count"]),
    )
    final_audit: dict[str, Any] | None = None
    final_series = pl.DataFrame()
    scored_targets = pl.DataFrame()
    if selected_candidate is not None:
        final_audit, final_series, scored_targets = evaluate_final_audit(
            samples,
            series_data,
            num_marks=int(source_quality["real_mark_count"]),
            selected=selected_candidate,
        )
    gate = evaluate_audit_gate(
        source_quality=source_quality,
        loader_quality=loader_quality,
        coverage_summary=coverage_summary,
        selected_candidate=selected_candidate,
        final_audit=final_audit,
    )

    quality_rows = metric_rows("source", source_quality)
    quality_rows.extend(metric_rows("loader", loader_quality))
    pl.DataFrame(quality_rows).write_csv(data_dir / "data_quality_summary.csv")
    pl.DataFrame(metric_rows("coverage", coverage_summary)).write_csv(
        data_dir / "coverage_summary.csv"
    )
    coverage.write_csv(data_dir / "coverage_thresholds.csv")
    by_series.write_csv(data_dir / "coverage_by_series.csv")
    by_mark.write_csv(data_dir / "coverage_by_target_mark.csv")
    candidates.write_csv(data_dir / "candidate_selection_metrics.csv")
    samples.write_parquet(data_dir / "train_target_memory_rows.parquet")
    if final_audit is not None:
        pl.DataFrame(metric_rows("final_train_audit", final_audit)).write_csv(
            data_dir / "final_probe_summary.csv"
        )
        final_series.write_csv(data_dir / "final_probe_by_series.csv")
        scored_targets.write_parquet(data_dir / "final_probe_target_errors.parquet")
    pl.DataFrame(
        [{"check": check, "passed": passed} for check, passed in gate["checks"].items()]
    ).write_csv(data_dir / "audit_gate.csv")

    plot_coverage(coverage, plot_dir / "pre_window_coverage.png")
    plot_candidate_grid(candidates, plot_dir / "candidate_selection_improvement.png")
    if final_audit is not None:
        plot_final_series(
            final_series,
            str(selected_candidate["primary_metric"]),
            plot_dir / "final_primary_metric_by_series.png",
        )

    report = build_report(
        source_path=dataset_path,
        source_quality=source_quality,
        coverage_summary=coverage_summary,
        selected_candidate=selected_candidate,
        final_audit=final_audit,
        gate=gate,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    summary = {
        "status": "completed",
        "source_quality": source_quality,
        "loader_contract": loader_quality,
        "coverage": coverage_summary,
        "selected_candidate": selected_candidate,
        "final_train_audit": final_audit,
        "audit_gate": gate,
        "held_out_target_data_read": False,
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    finished_at = datetime.now(KST)
    artifact_order = [
        "audit_manifest.json",
        "logs/audit.log",
        "audit_summary.json",
        "data/data_quality_summary.csv",
        "data/coverage_summary.csv",
        "data/coverage_thresholds.csv",
        "data/coverage_by_series.csv",
        "data/coverage_by_target_mark.csv",
        "data/candidate_selection_metrics.csv",
        "data/final_probe_summary.csv (only if selection passes)",
        "data/final_probe_by_series.csv (only if selection passes)",
        "data/audit_gate.csv",
        "data/train_target_memory_rows.parquet",
        "report.md",
        "plots/*.png",
    ]
    manifest = {
        "status": "completed",
        "analysis": "taxi_train_only_causal_pre_window_memory_audit",
        "started_at_kst": started_at.isoformat(),
        "finished_at_kst": finished_at.isoformat(),
        "execution_server": args.execution_server,
        "execution_host": platform.node(),
        "tmux_session": args.tmux_session,
        "source_revision": args.source_revision,
        "python": sys.executable,
        "sklearn_version": sklearn_version,
        "runtime_controls": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "numexpr_num_threads": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "dataset_path": str(dataset_path),
        "dataset_file_sha256": sha256_file(dataset_path),
        "dataset_scope": "fixed_split_train_parquet_only",
        "held_out_target_data_read": False,
        "output_dir": str(output_dir),
        "lookback_weeks_legacy_name": int(args.lookback_weeks),
        "lookback_seq_units": "hourly_bucket_index",
        "max_seq_len_including_target": int(args.max_seq_len),
        "decoded_target_sha256": frame_sha256(
            samples,
            [
                "oper_part_no",
                "target_seq",
                "context_end_index",
                "effective_context_start_index",
                "pre_window_count",
                "probe_partition",
            ],
        ),
        "audit_contract": {
            "expected_series_count": EXPECTED_SERIES_COUNT,
            "fit_selection_audit_shares": [FIT_SHARE, SELECTION_SHARE, AUDIT_SHARE],
            "memory_thresholds": MEMORY_THRESHOLDS,
            "memory_eligibility_count": MEMORY_ELIGIBILITY_COUNT,
            "memory_budgets": MEMORY_BUDGETS,
            "retrieval_topks": RETRIEVAL_TOPKS,
            "query_event_count": QUERY_EVENT_COUNT,
            "minimum_target_coverage": MIN_TARGET_COVERAGE,
            "minimum_series_coverage": MIN_SERIES_COVERAGE,
            "minimum_primary_improvement_pct": MIN_PRIMARY_IMPROVEMENT_PCT,
            "maximum_other_regression_pct": MAX_OTHER_REGRESSION_PCT,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "probe_seed": PROBE_SEED,
            "logistic_c": LOGISTIC_C,
            "ridge_alpha": RIDGE_ALPHA,
            "candidate_ranking": [
                "selection_pass",
                "maximum_worst_metric_improvement",
                "maximum_mean_improvement",
                "smaller_memory_budget",
                "smaller_topk",
            ],
        },
        "selected_candidate": selected_candidate,
        "decision": gate["decision"],
        "artifact_order": artifact_order,
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
        "Completed Taxi pre-window audit | selected=%s gate=%s decision=%s",
        None
        if selected_candidate is None
        else f"M{selected_candidate['memory_budget']}/k{selected_candidate['topk']}",
        gate["status"],
        gate["decision"],
    )


def main() -> None:
    write_outputs(parse_args())


if __name__ == "__main__":
    main()
