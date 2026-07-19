#!/usr/bin/env python3
"""Audit whether Taxi pre-window temporal fields explain the V7 time signal."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

os.environ.setdefault("MPLCONFIGDIR", f"/tmp/matplotlib-{os.environ.get('USER', 'codex')}")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simple_lab_test.search.analyze_taxi_pre_window_memory_audit import (
    EXPECTED_SERIES_COUNT,
    SeriesData,
    build_train_samples,
    frame_sha256,
    improvement_pct,
    jsonable,
    metric_rows,
    quantile,
    sha256_file,
    summarize_coverage,
    validate_loader_contract,
    validate_train_source,
)


DEFAULT_DATASET = (
    PROJECT_ROOT / "sample_data/new_york_taxi/yellow_trip_hourly_train.parquet"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "search_artifacts/model_enhancement_titantpp_v7_taxi_time_source_audit_0719"
)
KST = ZoneInfo("Asia/Seoul")

# Frozen before the first V7 source-isolation audit result is read.
MEMORY_ELIGIBILITY_COUNT = 8
ROLLING_TRAIN_END_SHARES = (0.55, 0.70, 0.85)
ROLLING_EVAL_END_SHARES = (0.70, 0.85, 1.00)
MIN_ROLLING_SERIES_TARGETS = 8
MIN_TARGET_COVERAGE = 0.35
MIN_SERIES_COVERAGE = 0.80
MIN_P1_POOLED_IMPROVEMENT_PCT = 1.0
MIN_P1_IMPROVED_FOLDS = 2
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 42
RIDGE_ALPHA = 1.0
POOLING_CONTRACT = "all_strict_pre_window_moments_v1"

TEMPORAL_FEATURE_NAMES = (
    "pre_log_count",
    "pre_log_span",
    "pre_log_gap_to_context",
    "pre_log_dt_mean",
    "pre_log_dt_std",
    "pre_log_dt_median",
    "pre_log_dt_p90",
    "pre_log_dt_last",
    "pre_log_dt_trend",
    "pre_log_age_mean",
    "pre_log_age_std",
    "pre_log_age_min",
    "pre_log_age_max",
    "pre_phase24_sin_mean",
    "pre_phase24_cos_mean",
    "pre_phase168_sin_mean",
    "pre_phase168_cos_mean",
    "pre_phase24_sin_last",
    "pre_phase24_cos_last",
    "pre_phase168_sin_last",
    "pre_phase168_cos_last",
)
QUANTITY_SUMMARY_NAMES = (
    "pre_log_qty_mean",
    "pre_log_qty_std",
    "pre_log_qty_median",
    "pre_log_qty_p90",
    "pre_log_qty_last",
    "pre_log_qty_trend",
)


@dataclass(frozen=True)
class SourceFeatureMatrix:
    sample_index: np.ndarray
    oper_part_no: np.ndarray
    target_seq: np.ndarray
    target_log1p_dt: np.ndarray
    p0_window: np.ndarray
    p1_temporal: np.ndarray
    p2_full: np.ndarray

    @property
    def target_count(self) -> int:
        return int(self.sample_index.size)


@dataclass(frozen=True)
class RollingOriginFold:
    name: str
    train_positions: np.ndarray
    eval_positions: np.ndarray


@dataclass(frozen=True)
class TimeProbe:
    scaler: StandardScaler
    model: Ridge


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
    logger = logging.getLogger("taxi_time_source_isolation_audit")
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


def active_window_feature_names(num_marks: int) -> tuple[str, ...]:
    return (
        *(f"window_mark_share_{mark}" for mark in range(num_marks)),
        *(f"window_last_mark_{mark}" for mark in range(num_marks)),
        "window_log_dt_mean",
        "window_log_dt_std",
        "window_log_dt_last",
        "window_log_dt_trend",
        "window_log_qty_mean",
        "window_log_qty_std",
        "window_log_qty_last",
        "window_log_qty_trend",
        "window_log_count",
        "window_log_span",
        "window_end_phase24_sin",
        "window_end_phase24_cos",
        "window_end_phase168_sin",
        "window_end_phase168_cos",
    )


def full_attribution_feature_names(num_marks: int) -> tuple[str, ...]:
    return (
        *(f"pre_mark_share_{mark}" for mark in range(num_marks)),
        *(f"pre_last_mark_{mark}" for mark in range(num_marks)),
        *QUANTITY_SUMMARY_NAMES,
    )


def _mean_std_last_trend(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(values.mean()),
            float(values.std()),
            float(values[-1]),
            float(values[-1] - values[0]),
        ],
        dtype=np.float64,
    )


def build_active_window_features(
    series: SeriesData,
    *,
    context_start: int,
    context_end: int,
    num_marks: int,
) -> np.ndarray:
    if not (0 <= context_start <= context_end < len(series.seq)):
        raise ValueError("Active-window indices violate causal ordering")
    indices = np.arange(context_start, context_end + 1, dtype=np.int64)
    marks = series.mark[indices]
    if np.any((marks < 0) | (marks >= num_marks)):
        raise ValueError("Active-window marks exceed the real-mark range")
    mark_share = np.bincount(marks, minlength=num_marks).astype(np.float64)
    mark_share /= float(indices.size)
    last_mark = np.eye(num_marks, dtype=np.float64)[marks[-1]]
    log_dt = np.log1p(series.delta_t[indices])
    log_qty = np.log2(series.quantity[indices])
    end_seq = float(series.seq[context_end])
    metadata = np.asarray(
        [
            float(np.log1p(indices.size)),
            float(np.log1p(series.seq[context_end] - series.seq[context_start])),
            float(np.sin(2.0 * np.pi * end_seq / 24.0)),
            float(np.cos(2.0 * np.pi * end_seq / 24.0)),
            float(np.sin(2.0 * np.pi * end_seq / 168.0)),
            float(np.cos(2.0 * np.pi * end_seq / 168.0)),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            mark_share,
            last_mark,
            _mean_std_last_trend(log_dt),
            _mean_std_last_trend(log_qty),
            metadata,
        ]
    )


def build_temporal_pre_window_features(
    series: SeriesData,
    *,
    context_start: int,
    context_end: int,
    minimum_count: int = MEMORY_ELIGIBILITY_COUNT,
) -> np.ndarray:
    if not (minimum_count <= context_start <= context_end < len(series.seq)):
        raise ValueError(
            "Temporal pre-window features require enough strictly past events "
            "and a valid active context"
        )
    indices = np.arange(0, context_start, dtype=np.int64)
    seq = series.seq[indices].astype(np.float64)
    log_dt = np.log1p(series.delta_t[indices])
    context_start_seq = float(series.seq[context_start])
    log_age = np.log1p(context_start_seq - seq)
    phase24_sin = np.sin(2.0 * np.pi * seq / 24.0)
    phase24_cos = np.cos(2.0 * np.pi * seq / 24.0)
    phase168_sin = np.sin(2.0 * np.pi * seq / 168.0)
    phase168_cos = np.cos(2.0 * np.pi * seq / 168.0)
    features = np.asarray(
        [
            float(np.log1p(indices.size)),
            float(np.log1p(seq[-1] - seq[0])),
            float(np.log1p(context_start_seq - seq[-1])),
            float(log_dt.mean()),
            float(log_dt.std()),
            float(np.median(log_dt)),
            quantile(log_dt, 0.90),
            float(log_dt[-1]),
            float(log_dt[-1] - log_dt[0]),
            float(log_age.mean()),
            float(log_age.std()),
            float(log_age.min()),
            float(log_age.max()),
            float(phase24_sin.mean()),
            float(phase24_cos.mean()),
            float(phase168_sin.mean()),
            float(phase168_cos.mean()),
            float(phase24_sin[-1]),
            float(phase24_cos[-1]),
            float(phase168_sin[-1]),
            float(phase168_cos[-1]),
        ],
        dtype=np.float64,
    )
    if features.size != len(TEMPORAL_FEATURE_NAMES):
        raise RuntimeError("Temporal feature names and values are misaligned")
    return features


def build_full_pre_window_attribution_features(
    series: SeriesData,
    *,
    context_start: int,
    num_marks: int,
    minimum_count: int = MEMORY_ELIGIBILITY_COUNT,
) -> np.ndarray:
    if not (minimum_count <= context_start < len(series.seq)):
        raise ValueError("Full pre-window attribution requires enough past events")
    indices = np.arange(0, context_start, dtype=np.int64)
    marks = series.mark[indices]
    if np.any((marks < 0) | (marks >= num_marks)):
        raise ValueError("Pre-window marks exceed the real-mark range")
    mark_share = np.bincount(marks, minlength=num_marks).astype(np.float64)
    mark_share /= float(indices.size)
    last_mark = np.eye(num_marks, dtype=np.float64)[marks[-1]]
    log_qty = np.log2(series.quantity[indices])
    qty_summary = np.asarray(
        [
            float(log_qty.mean()),
            float(log_qty.std()),
            float(np.median(log_qty)),
            quantile(log_qty, 0.90),
            float(log_qty[-1]),
            float(log_qty[-1] - log_qty[0]),
        ],
        dtype=np.float64,
    )
    return np.concatenate([mark_share, last_mark, qty_summary])


def build_source_feature_matrix(
    samples: pl.DataFrame,
    series_data: dict[int, SeriesData],
    *,
    num_marks: int,
    minimum_count: int = MEMORY_ELIGIBILITY_COUNT,
) -> tuple[SourceFeatureMatrix, dict[str, Any]]:
    selected = samples.filter(pl.col("pre_window_count") >= minimum_count).sort(
        ["oper_part_no", "target_seq"]
    )
    if selected.height == 0:
        raise ValueError("No time-source eligible Taxi train targets")

    sample_indices: list[int] = []
    parts: list[str] = []
    target_seqs: list[int] = []
    targets: list[float] = []
    p0_rows: list[np.ndarray] = []
    p1_rows: list[np.ndarray] = []
    p2_rows: list[np.ndarray] = []
    causal_violations = 0
    target_mismatches = 0

    for row in selected.iter_rows(named=True):
        part_index = int(row["part_index"])
        context_start = int(row["effective_context_start_index"])
        context_end = int(row["context_end_index"])
        target_index = int(row["target_index"])
        series = series_data[part_index]
        causal_violations += int(
            not (minimum_count <= context_start <= context_end < target_index < len(series.seq))
        )
        if causal_violations:
            raise ValueError("Source feature matrix violates pre-window causality")

        p0 = build_active_window_features(
            series,
            context_start=context_start,
            context_end=context_end,
            num_marks=num_marks,
        )
        temporal = build_temporal_pre_window_features(
            series,
            context_start=context_start,
            context_end=context_end,
            minimum_count=minimum_count,
        )
        full_extra = build_full_pre_window_attribution_features(
            series,
            context_start=context_start,
            num_marks=num_marks,
            minimum_count=minimum_count,
        )
        target = float(np.log1p(series.delta_t[target_index]))
        target_mismatches += int(
            not math.isclose(
                target,
                float(row["target_log1p_delta_t"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        sample_indices.append(int(row["sample_index"]))
        parts.append(str(row["oper_part_no"]))
        target_seqs.append(int(row["target_seq"]))
        targets.append(target)
        p0_rows.append(p0)
        p1_rows.append(np.concatenate([p0, temporal]))
        p2_rows.append(np.concatenate([p0, temporal, full_extra]))

    matrix = SourceFeatureMatrix(
        sample_index=np.asarray(sample_indices, dtype=np.int64),
        oper_part_no=np.asarray(parts, dtype=object),
        target_seq=np.asarray(target_seqs, dtype=np.int64),
        target_log1p_dt=np.asarray(targets, dtype=np.float64),
        p0_window=np.vstack(p0_rows),
        p1_temporal=np.vstack(p1_rows),
        p2_full=np.vstack(p2_rows),
    )
    tensors = (matrix.p0_window, matrix.p1_temporal, matrix.p2_full, matrix.target_log1p_dt)
    finite = all(np.isfinite(tensor).all() for tensor in tensors)
    aligned = (
        matrix.p0_window.shape[0]
        == matrix.p1_temporal.shape[0]
        == matrix.p2_full.shape[0]
        == matrix.target_count
    )
    if target_mismatches or not finite or not aligned:
        raise ValueError(
            "Source feature contract failed: "
            f"target_mismatches={target_mismatches} finite={finite} aligned={aligned}"
        )

    p0_names = active_window_feature_names(num_marks)
    p1_names = (*p0_names, *TEMPORAL_FEATURE_NAMES)
    p2_names = (*p1_names, *full_attribution_feature_names(num_marks))
    dimensions_match = (
        len(p0_names) == matrix.p0_window.shape[1]
        and len(p1_names) == matrix.p1_temporal.shape[1]
        and len(p2_names) == matrix.p2_full.shape[1]
    )
    if not dimensions_match:
        raise RuntimeError("Source feature names and matrix dimensions are misaligned")

    contract = {
        "status": "PASS",
        "pooling_contract": POOLING_CONTRACT,
        "minimum_pre_window_count": minimum_count,
        "eligible_target_count": matrix.target_count,
        "eligible_series_count": int(np.unique(matrix.oper_part_no).size),
        "p0_feature_count": int(matrix.p0_window.shape[1]),
        "p1_feature_count": int(matrix.p1_temporal.shape[1]),
        "p2_feature_count": int(matrix.p2_full.shape[1]),
        "p0_feature_names": p0_names,
        "p1_added_feature_names": TEMPORAL_FEATURE_NAMES,
        "p2_added_feature_names": full_attribution_feature_names(num_marks),
        "p1_pre_window_mark_features": False,
        "p1_pre_window_quantity_features": False,
        "p1_series_identity_feature": False,
        "causal_ordering_violation_count": causal_violations,
        "target_alignment_mismatch_count": target_mismatches,
        "all_features_finite": finite,
    }
    return matrix, contract


def build_rolling_origin_folds(
    matrix: SourceFeatureMatrix,
    *,
    train_end_shares: Sequence[float] = ROLLING_TRAIN_END_SHARES,
    eval_end_shares: Sequence[float] = ROLLING_EVAL_END_SHARES,
    minimum_series_targets: int = MIN_ROLLING_SERIES_TARGETS,
) -> tuple[list[RollingOriginFold], dict[str, Any]]:
    if len(train_end_shares) != len(eval_end_shares) or not train_end_shares:
        raise ValueError("Rolling train/eval share lists must have the same non-zero length")
    previous_eval_end = 0.0
    for train_end, eval_end in zip(train_end_shares, eval_end_shares):
        if not (0.0 < train_end < eval_end <= 1.0):
            raise ValueError("Each rolling fold needs 0 < train_end < eval_end <= 1")
        if train_end < previous_eval_end - 1e-12:
            raise ValueError("Rolling training prefixes cannot move backward")
        previous_eval_end = eval_end

    unique_parts = np.asarray(sorted(set(str(part) for part in matrix.oper_part_no)), dtype=object)
    train_by_fold: list[list[np.ndarray]] = [[] for _ in train_end_shares]
    eval_by_fold: list[list[np.ndarray]] = [[] for _ in train_end_shares]
    participating_series: set[str] = set()
    ordering_violations = 0

    for part in unique_parts:
        positions = np.flatnonzero(matrix.oper_part_no == part)
        order = positions[np.argsort(matrix.target_seq[positions], kind="stable")]
        count = int(order.size)
        if count < minimum_series_targets:
            continue
        participating_series.add(str(part))
        for fold_index, (train_share, eval_share) in enumerate(
            zip(train_end_shares, eval_end_shares)
        ):
            train_end = max(1, int(math.floor(count * float(train_share))))
            eval_end = min(count, int(math.floor(count * float(eval_share))))
            if eval_end <= train_end:
                eval_end = min(count, train_end + 1)
            if eval_end <= train_end:
                continue
            train_positions = order[:train_end]
            eval_positions = order[train_end:eval_end]
            ordering_violations += int(
                int(matrix.target_seq[train_positions].max())
                >= int(matrix.target_seq[eval_positions].min())
            )
            train_by_fold[fold_index].append(train_positions)
            eval_by_fold[fold_index].append(eval_positions)

    folds: list[RollingOriginFold] = []
    for fold_index in range(len(train_end_shares)):
        if not train_by_fold[fold_index] or not eval_by_fold[fold_index]:
            raise ValueError(f"Rolling fold {fold_index + 1} has no train/eval targets")
        train_positions = np.concatenate(train_by_fold[fold_index]).astype(np.int64)
        eval_positions = np.concatenate(eval_by_fold[fold_index]).astype(np.int64)
        if np.intersect1d(train_positions, eval_positions).size:
            raise ValueError(f"Rolling fold {fold_index + 1} overlaps train and eval targets")
        folds.append(
            RollingOriginFold(
                name=f"fold_{fold_index + 1}",
                train_positions=np.sort(train_positions),
                eval_positions=np.sort(eval_positions),
            )
        )

    all_eval = np.concatenate([fold.eval_positions for fold in folds])
    eval_overlap_count = int(all_eval.size - np.unique(all_eval).size)
    participating_share = float(len(participating_series) / max(unique_parts.size, 1))
    status = (
        "PASS"
        if ordering_violations == 0
        and eval_overlap_count == 0
        and len(folds) == len(train_end_shares)
        else "FAIL"
    )
    summary = {
        "status": status,
        "fold_count": len(folds),
        "train_end_shares": tuple(float(value) for value in train_end_shares),
        "eval_end_shares": tuple(float(value) for value in eval_end_shares),
        "minimum_series_targets": minimum_series_targets,
        "eligible_series_count": int(unique_parts.size),
        "participating_series_count": len(participating_series),
        "participating_series_share": participating_share,
        "evaluation_target_count": int(all_eval.size),
        "evaluation_target_overlap_count": eval_overlap_count,
        "chronological_ordering_violation_count": ordering_violations,
    }
    return folds, summary


def rolling_assignment_frame(
    matrix: SourceFeatureMatrix,
    folds: Sequence[RollingOriginFold],
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for fold in folds:
        for role, positions in (
            ("train", fold.train_positions),
            ("eval", fold.eval_positions),
        ):
            frames.append(
                pl.DataFrame(
                    {
                        "fold": [fold.name] * int(positions.size),
                        "role": [role] * int(positions.size),
                        "sample_index": matrix.sample_index[positions],
                        "oper_part_no": matrix.oper_part_no[positions],
                        "target_seq": matrix.target_seq[positions],
                    }
                )
            )
    return pl.concat(frames, how="vertical").sort(
        ["fold", "role", "oper_part_no", "target_seq"]
    )


def fit_time_probe(features: np.ndarray, targets: np.ndarray) -> TimeProbe:
    if features.ndim != 2 or targets.ndim != 1 or features.shape[0] != targets.size:
        raise ValueError("Time probe features and targets are not aligned")
    if (
        features.shape[0] == 0
        or not np.isfinite(features).all()
        or not np.isfinite(targets).all()
    ):
        raise ValueError("Time probe training data must be non-empty and finite")
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    model = Ridge(alpha=RIDGE_ALPHA, solver="cholesky").fit(transformed, targets)
    return TimeProbe(scaler=scaler, model=model)


def score_time_probe(probe: TimeProbe, features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    predictions = probe.model.predict(probe.scaler.transform(features))
    errors = np.abs(predictions - targets)
    if not np.isfinite(errors).all():
        raise ValueError("Time probe produced NaN or Inf errors")
    return errors.astype(np.float64)


def evaluate_rolling_origin(
    matrix: SourceFeatureMatrix,
    folds: Sequence[RollingOriginFold],
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    source_features = {
        "p0": matrix.p0_window,
        "p1": matrix.p1_temporal,
        "p2": matrix.p2_full,
    }
    fold_rows: list[dict[str, Any]] = []
    oof_frames: list[pl.DataFrame] = []

    for fold in folds:
        train = fold.train_positions
        evaluate = fold.eval_positions
        errors: dict[str, np.ndarray] = {}
        for source, features in source_features.items():
            probe = fit_time_probe(features[train], matrix.target_log1p_dt[train])
            errors[source] = score_time_probe(
                probe,
                features[evaluate],
                matrix.target_log1p_dt[evaluate],
            )
        p0_mae = float(errors["p0"].mean())
        p1_mae = float(errors["p1"].mean())
        p2_mae = float(errors["p2"].mean())
        fold_rows.append(
            {
                "fold": fold.name,
                "train_target_count": int(train.size),
                "eval_target_count": int(evaluate.size),
                "train_series_count": int(np.unique(matrix.oper_part_no[train]).size),
                "eval_series_count": int(np.unique(matrix.oper_part_no[evaluate]).size),
                "p0_log1p_dt_mae": p0_mae,
                "p1_log1p_dt_mae": p1_mae,
                "p2_log1p_dt_mae": p2_mae,
                "p1_improvement_pct": improvement_pct(p0_mae, p1_mae),
                "p2_improvement_pct": improvement_pct(p0_mae, p2_mae),
            }
        )
        oof_frames.append(
            pl.DataFrame(
                {
                    "fold": [fold.name] * int(evaluate.size),
                    "sample_index": matrix.sample_index[evaluate],
                    "oper_part_no": matrix.oper_part_no[evaluate],
                    "target_seq": matrix.target_seq[evaluate],
                    "target_log1p_delta_t": matrix.target_log1p_dt[evaluate],
                    "p0_abs_error": errors["p0"],
                    "p1_abs_error": errors["p1"],
                    "p2_abs_error": errors["p2"],
                }
            )
        )

    fold_metrics = pl.DataFrame(fold_rows).sort("fold")
    oof = pl.concat(oof_frames, how="vertical").sort(
        ["oper_part_no", "target_seq", "fold"]
    )
    duplicate_oof_targets = int(
        oof.group_by("sample_index")
        .len()
        .filter(pl.col("len") > 1)
        .select((pl.col("len") - 1).sum())
        .item()
        or 0
    )
    p0_mae = float(oof["p0_abs_error"].mean())
    p1_mae = float(oof["p1_abs_error"].mean())
    p2_mae = float(oof["p2_abs_error"].mean())
    errors_finite = bool(
        oof.select(
            pl.all_horizontal(
                [
                    pl.col(column).is_finite()
                    for column in (
                        "p0_abs_error",
                        "p1_abs_error",
                        "p2_abs_error",
                    )
                ]
            ).all()
        ).item()
    )
    summary = {
        "fold_count": int(fold_metrics.height),
        "oof_target_count": int(oof.height),
        "oof_series_count": int(oof["oper_part_no"].n_unique()),
        "duplicate_oof_target_count": duplicate_oof_targets,
        "errors_finite": errors_finite,
        "p0_log1p_dt_mae": p0_mae,
        "p1_log1p_dt_mae": p1_mae,
        "p2_log1p_dt_mae": p2_mae,
        "p1_improvement_pct": improvement_pct(p0_mae, p1_mae),
        "p2_improvement_pct": improvement_pct(p0_mae, p2_mae),
        "p1_improved_fold_count": int(
            (fold_metrics["p1_improvement_pct"] > 0.0).sum()
        ),
        "p2_improved_fold_count": int(
            (fold_metrics["p2_improvement_pct"] > 0.0).sum()
        ),
    }
    if duplicate_oof_targets or not errors_finite:
        raise ValueError(
            "Rolling-origin OOF contract failed: "
            f"duplicates={duplicate_oof_targets} finite={errors_finite}"
        )
    return fold_metrics, oof, summary


def bootstrap_oof_improvements(
    oof: pl.DataFrame,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], pl.DataFrame]:
    if replicates < 1:
        raise ValueError("Bootstrap replicate count must be positive")
    series_rows: list[dict[str, Any]] = []
    for group in oof.partition_by("oper_part_no", maintain_order=False):
        p0 = group["p0_abs_error"].to_numpy().astype(np.float64)
        p1 = group["p1_abs_error"].to_numpy().astype(np.float64)
        p2 = group["p2_abs_error"].to_numpy().astype(np.float64)
        p0_mae = float(p0.mean())
        p1_mae = float(p1.mean())
        p2_mae = float(p2.mean())
        series_rows.append(
            {
                "oper_part_no": str(group["oper_part_no"][0]),
                "target_count": int(group.height),
                "p0_error_sum": float(p0.sum()),
                "p1_error_sum": float(p1.sum()),
                "p2_error_sum": float(p2.sum()),
                "p0_log1p_dt_mae": p0_mae,
                "p1_log1p_dt_mae": p1_mae,
                "p2_log1p_dt_mae": p2_mae,
                "p1_improvement_pct": improvement_pct(p0_mae, p1_mae),
                "p2_improvement_pct": improvement_pct(p0_mae, p2_mae),
            }
        )
    by_series = pl.DataFrame(series_rows).sort("oper_part_no")
    if by_series.height == 0:
        raise ValueError("Series bootstrap requires OOF targets")

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, by_series.height, size=(replicates, by_series.height))
    counts = by_series["target_count"].to_numpy().astype(np.float64)
    denominator = counts[draws].sum(axis=1)
    p0_sums = by_series["p0_error_sum"].to_numpy().astype(np.float64)
    p0_mean = p0_sums[draws].sum(axis=1) / denominator
    summary: dict[str, Any] = {
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "series_count": int(by_series.height),
    }
    for source in ("p1", "p2"):
        source_sums = by_series[f"{source}_error_sum"].to_numpy().astype(np.float64)
        source_mean = source_sums[draws].sum(axis=1) / denominator
        improvements = 100.0 * (p0_mean - source_mean) / np.maximum(np.abs(p0_mean), 1e-12)
        summary[f"{source}_bootstrap_ci_low"] = quantile(improvements, 0.025)
        summary[f"{source}_bootstrap_ci_high"] = quantile(improvements, 0.975)
        summary[f"{source}_series_improved_share"] = float(
            np.mean(by_series[f"{source}_improvement_pct"].to_numpy() > 0.0)
        )
    return summary, by_series


def evaluate_time_source_gate(
    *,
    source_quality: dict[str, Any],
    loader_quality: dict[str, Any],
    coverage_summary: dict[str, Any],
    feature_contract: dict[str, Any],
    rolling_contract: dict[str, Any],
    oof_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
) -> dict[str, Any]:
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
        "feature_source_and_causality_contract_pass": feature_contract["status"] == "PASS",
        "rolling_origin_contract_pass": rolling_contract["status"] == "PASS",
        "three_rolling_folds_evaluated": int(oof_summary["fold_count"])
        == len(ROLLING_TRAIN_END_SHARES),
        "oof_targets_unique_and_finite": int(oof_summary["duplicate_oof_target_count"]) == 0
        and bool(oof_summary["errors_finite"]),
        "p1_pooled_improvement_at_least_1pct": float(oof_summary["p1_improvement_pct"])
        >= MIN_P1_POOLED_IMPROVEMENT_PCT,
        "p1_improves_at_least_2_of_3_folds": int(oof_summary["p1_improved_fold_count"])
        >= MIN_P1_IMPROVED_FOLDS,
        "p1_series_bootstrap_ci_low_above_zero": float(
            bootstrap_summary["p1_bootstrap_ci_low"]
        )
        > 0.0,
    }
    passed = all(checks.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "freeze_v7_temporal_source_and_open_adapter_implementation"
            if passed
            else "close_v7_before_model_implementation_and_revisit_v5b"
        ),
        "primary_comparison": "p1_temporal_vs_p0_window",
        "p2_can_pass_gate": False,
        "checks": checks,
        "thresholds": {
            "minimum_pre_window_count": MEMORY_ELIGIBILITY_COUNT,
            "minimum_target_coverage": MIN_TARGET_COVERAGE,
            "minimum_series_coverage": MIN_SERIES_COVERAGE,
            "minimum_p1_pooled_improvement_pct": MIN_P1_POOLED_IMPROVEMENT_PCT,
            "minimum_p1_improved_folds": MIN_P1_IMPROVED_FOLDS,
            "p1_bootstrap_ci_low_must_exceed": 0.0,
        },
    }


def feature_contract_frame(contract: dict[str, Any]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for source, key in (
        ("p0_window", "p0_feature_names"),
        ("p1_temporal_added", "p1_added_feature_names"),
        ("p2_full_added", "p2_added_feature_names"),
    ):
        for order, feature in enumerate(contract[key]):
            rows.append({"source": source, "feature_order": order, "feature": feature})
    return pl.DataFrame(rows)


def plot_coverage(coverage: pl.DataFrame, output_path: Path) -> None:
    thresholds = coverage["threshold"].to_numpy()
    target_share = 100.0 * coverage["covered_target_share"].to_numpy()
    series_share = 100.0 * coverage["covered_series_share"].to_numpy()
    positions = np.arange(thresholds.size)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(positions - width / 2, target_share, width, color="#1D5D6C", label="Targets")
    ax.bar(positions + width / 2, series_share, width, color="#D17A22", label="Series")
    ax.axhline(100.0 * MIN_TARGET_COVERAGE, color="#1D5D6C", linestyle="--")
    ax.axhline(100.0 * MIN_SERIES_COVERAGE, color="#D17A22", linestyle="--")
    ax.set_xticks(positions, [f">={int(value)}" for value in thresholds])
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("Strictly pre-window event count")
    ax.set_title("Taxi train-only V7 temporal-source coverage")
    ax.legend()
    ax.grid(axis="y", color="#D9E1E8", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fold_improvements(fold_metrics: pl.DataFrame, output_path: Path) -> None:
    labels = fold_metrics["fold"].to_list()
    positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.bar(
        positions - width / 2,
        fold_metrics["p1_improvement_pct"].to_numpy(),
        width,
        color="#2A6F97",
        label="P1 temporal only",
    )
    ax.bar(
        positions + width / 2,
        fold_metrics["p2_improvement_pct"].to_numpy(),
        width,
        color="#BC6C25",
        label="P2 full attribution",
    )
    ax.axhline(0.0, color="#202124", linewidth=1.0)
    ax.axhline(MIN_P1_POOLED_IMPROVEMENT_PCT, color="#1B4332", linestyle="--")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Log1p(dt) MAE improvement (%)")
    ax.set_title("Rolling-origin V7 source isolation")
    ax.legend()
    ax.grid(axis="y", color="#E1E4E8", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_series_improvements(by_series: pl.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.hist(
        by_series["p1_improvement_pct"].to_numpy(),
        bins=30,
        alpha=0.72,
        color="#2A6F97",
        label="P1 temporal only",
    )
    ax.hist(
        by_series["p2_improvement_pct"].to_numpy(),
        bins=30,
        alpha=0.45,
        color="#BC6C25",
        label="P2 full attribution",
    )
    ax.axvline(0.0, color="#9E2A2B", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Per-series log1p(dt) MAE improvement (%)")
    ax.set_ylabel("Series count")
    ax.set_title("V7 source-isolation out-of-fold series effects")
    ax.legend()
    ax.grid(axis="y", color="#DCE3E8", linewidth=0.7, alpha=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    *,
    source_path: Path,
    source_quality: dict[str, Any],
    coverage_summary: dict[str, Any],
    rolling_contract: dict[str, Any],
    oof_summary: dict[str, Any],
    bootstrap_summary: dict[str, Any],
    gate: dict[str, Any],
) -> str:
    lines = [
        "# Taxi Train-Only P0/P1/P2 Time-Source Isolation Audit",
        "",
        "## Scope",
        "",
        f"- Source: `{source_path}`",
        "- Source scope: fixed-split Taxi train parquet only",
        "- Validation/test target data read: `false`",
        "- Purpose: decide whether temporal-only pre-window input can open V7 implementation",
        f"- Fixed pooling: `{POOLING_CONTRACT}`; no V6 `M/topk` reuse",
        "",
        "## Coverage",
        "",
        "- Train rows / series / targets: "
        f"`{source_quality['row_count']}` / `{source_quality['series_count']}` / "
        f"`{coverage_summary['target_count']}`",
        "- Targets / series with at least 8 pre-window events: "
        f"`{100.0*float(coverage_summary['eligible_target_share_ge_8']):.3f}%` / "
        f"`{100.0*float(coverage_summary['eligible_series_share_ge_8']):.3f}%`",
        "",
        "## Rolling-Origin Contract",
        "",
        f"- Folds: `{rolling_contract['fold_count']}`",
        "- Train-end shares: `55% / 70% / 85%`; eval-end shares: `70% / 85% / 100%`",
        f"- OOF targets / series: `{oof_summary['oof_target_count']}` / "
        f"`{oof_summary['oof_series_count']}`",
        "",
        "## P0/P1/P2 Result",
        "",
        f"- P0 window-only MAE: `{float(oof_summary['p0_log1p_dt_mae']):.6f}`",
        "- P1 temporal-only MAE / improvement / improved folds / 95% CI: "
        f"`{float(oof_summary['p1_log1p_dt_mae']):.6f}` / "
        f"`{float(oof_summary['p1_improvement_pct']):.4f}%` / "
        f"`{int(oof_summary['p1_improved_fold_count'])}/3` / "
        f"`[{float(bootstrap_summary['p1_bootstrap_ci_low']):.4f}%, "
        f"{float(bootstrap_summary['p1_bootstrap_ci_high']):.4f}%]`",
        "- P2 full-attribution MAE / improvement / improved folds / 95% CI: "
        f"`{float(oof_summary['p2_log1p_dt_mae']):.6f}` / "
        f"`{float(oof_summary['p2_improvement_pct']):.4f}%` / "
        f"`{int(oof_summary['p2_improved_fold_count'])}/3` / "
        f"`[{float(bootstrap_summary['p2_bootstrap_ci_low']):.4f}%, "
        f"{float(bootstrap_summary['p2_bootstrap_ci_high']):.4f}%]`",
        "",
        "## Gate",
        "",
        f"- Status: `{gate['status']}`",
        f"- Decision: `{gate['decision']}`",
    ]
    for check, passed in gate["checks"].items():
        lines.append(f"- `{check}`: `{'PASS' if passed else 'FAIL'}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- P1 versus P0 is the only promotion comparison.",
            "- P2 only tests whether mark/quantity explain the prior V6 time signal.",
            "- This is train-only feasibility evidence, not TitanTPP model-quality evidence.",
            "- No fitted scaler, Ridge parameter, or statistic transfers into TitanTPP.",
            "",
            "## Next",
            "",
            (
                "Freeze the V7 temporal source contract and implement the time-history adapter."
                if gate["status"] == "PASS"
                else "Close V7 before model implementation and revisit the V5b design."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    args: argparse.Namespace,
    *,
    expected_series_count: int | None = EXPECTED_SERIES_COUNT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> None:
    output_dir = args.output_dir.resolve()
    data_dir = output_dir / "data"
    plot_dir = output_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(output_dir)
    started_at = datetime.now(KST)
    dataset_path = args.dataset.resolve()
    logger.info(
        "Starting Taxi V7 time-source isolation audit | server=%s tmux=%s dataset=%s",
        args.execution_server,
        args.tmux_session,
        dataset_path,
    )

    frame = pl.read_parquet(dataset_path).sort(["oper_part_no", "seq"])
    source_quality = validate_train_source(
        frame, expected_series_count=expected_series_count
    )
    samples, dataset, series_data = build_train_samples(
        frame,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
    )
    loader_quality = validate_loader_contract(frame, samples, dataset, series_data)
    coverage_summary, coverage, by_series_coverage, by_mark = summarize_coverage(
        samples, series_count=int(source_quality["series_count"])
    )
    samples = samples.with_columns(
        (pl.col("pre_window_count") >= MEMORY_ELIGIBILITY_COUNT).alias(
            "time_source_eligible"
        )
    )
    logger.info(
        "Coverage decoded | targets=%d ge8=%.4f series_ge8=%.4f",
        int(samples.height),
        float(coverage_summary["eligible_target_share_ge_8"]),
        float(coverage_summary["eligible_series_share_ge_8"]),
    )

    matrix, feature_contract = build_source_feature_matrix(
        samples,
        series_data,
        num_marks=int(source_quality["real_mark_count"]),
    )
    folds, rolling_contract = build_rolling_origin_folds(matrix)
    assignments = rolling_assignment_frame(matrix, folds)
    fold_metrics, oof, oof_summary = evaluate_rolling_origin(matrix, folds)
    bootstrap_summary, oof_by_series = bootstrap_oof_improvements(
        oof,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    gate = evaluate_time_source_gate(
        source_quality=source_quality,
        loader_quality=loader_quality,
        coverage_summary=coverage_summary,
        feature_contract=feature_contract,
        rolling_contract=rolling_contract,
        oof_summary=oof_summary,
        bootstrap_summary=bootstrap_summary,
    )

    quality_rows = metric_rows("source", source_quality)
    quality_rows.extend(metric_rows("loader", loader_quality))
    pl.DataFrame(quality_rows).write_csv(data_dir / "data_quality_summary.csv")
    pl.DataFrame(metric_rows("coverage", coverage_summary)).write_csv(
        data_dir / "coverage_summary.csv"
    )
    coverage.write_csv(data_dir / "coverage_thresholds.csv")
    by_series_coverage.write_csv(data_dir / "coverage_by_series.csv")
    by_mark.write_csv(data_dir / "coverage_by_target_mark.csv")
    feature_contract_frame(feature_contract).write_csv(
        data_dir / "source_feature_contract.csv"
    )
    pl.DataFrame(metric_rows("feature_contract", feature_contract)).write_csv(
        data_dir / "source_feature_summary.csv"
    )
    pl.DataFrame(metric_rows("rolling_contract", rolling_contract)).write_csv(
        data_dir / "rolling_origin_summary.csv"
    )
    assignments.write_parquet(data_dir / "rolling_fold_assignments.parquet")
    fold_metrics.write_csv(data_dir / "rolling_fold_metrics.csv")
    pl.DataFrame(metric_rows("pooled_oof", oof_summary)).write_csv(
        data_dir / "pooled_oof_summary.csv"
    )
    pl.DataFrame(metric_rows("series_bootstrap", bootstrap_summary)).write_csv(
        data_dir / "series_bootstrap_summary.csv"
    )
    oof.write_parquet(data_dir / "oof_target_errors.parquet")
    oof_by_series.write_csv(data_dir / "oof_by_series.csv")
    samples.write_parquet(data_dir / "train_target_source_rows.parquet")
    pl.DataFrame(
        [{"check": check, "passed": passed} for check, passed in gate["checks"].items()]
    ).write_csv(data_dir / "audit_gate.csv")

    plot_coverage(coverage, plot_dir / "pre_window_coverage.png")
    plot_fold_improvements(fold_metrics, plot_dir / "rolling_fold_improvement.png")
    plot_series_improvements(oof_by_series, plot_dir / "series_improvement.png")

    report = build_report(
        source_path=dataset_path,
        source_quality=source_quality,
        coverage_summary=coverage_summary,
        rolling_contract=rolling_contract,
        oof_summary=oof_summary,
        bootstrap_summary=bootstrap_summary,
        gate=gate,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    summary = {
        "status": "completed",
        "source_quality": source_quality,
        "loader_contract": loader_quality,
        "coverage": coverage_summary,
        "feature_contract": feature_contract,
        "rolling_origin": rolling_contract,
        "pooled_out_of_fold": oof_summary,
        "series_bootstrap": bootstrap_summary,
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
        "data/coverage_*.csv",
        "data/source_feature_*.csv",
        "data/rolling_origin_summary.csv",
        "data/rolling_fold_metrics.csv",
        "data/pooled_oof_summary.csv",
        "data/series_bootstrap_summary.csv",
        "data/oof_by_series.csv",
        "data/audit_gate.csv",
        "data/*.parquet",
        "report.md",
        "plots/*.png",
    ]
    manifest = {
        "status": "completed",
        "analysis": "taxi_train_only_p0_p1_p2_time_source_isolation_audit",
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
                "time_source_eligible",
            ],
        ),
        "oof_target_sha256": frame_sha256(
            oof,
            ["fold", "oper_part_no", "target_seq", "sample_index"],
        ),
        "audit_contract": {
            "expected_series_count": expected_series_count,
            "source_variants": {
                "p0": "active_window_summary_only",
                "p1": "p0_plus_strict_pre_window_temporal_fields",
                "p2": "p1_plus_strict_pre_window_mark_and_quantity_attribution",
            },
            "primary_comparison": "p1_vs_p0",
            "p2_can_pass_gate": False,
            "pooling_contract": POOLING_CONTRACT,
            "v6_memory_budget_reused": False,
            "v6_topk_reused": False,
            "memory_eligibility_count": MEMORY_ELIGIBILITY_COUNT,
            "rolling_train_end_shares": ROLLING_TRAIN_END_SHARES,
            "rolling_eval_end_shares": ROLLING_EVAL_END_SHARES,
            "minimum_rolling_series_targets": MIN_ROLLING_SERIES_TARGETS,
            "ridge_alpha": RIDGE_ALPHA,
            "minimum_target_coverage": MIN_TARGET_COVERAGE,
            "minimum_series_coverage": MIN_SERIES_COVERAGE,
            "minimum_p1_pooled_improvement_pct": MIN_P1_POOLED_IMPROVEMENT_PCT,
            "minimum_p1_improved_folds": MIN_P1_IMPROVED_FOLDS,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "feature_contract": {
            "p0_feature_names": feature_contract["p0_feature_names"],
            "p1_added_feature_names": feature_contract["p1_added_feature_names"],
            "p2_added_feature_names": feature_contract["p2_added_feature_names"],
        },
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
        "Completed Taxi V7 time-source isolation audit | p1=%.4f%% p2=%.4f%% "
        "gate=%s decision=%s",
        float(oof_summary["p1_improvement_pct"]),
        float(oof_summary["p2_improvement_pct"]),
        gate["status"],
        gate["decision"],
    )


def main() -> None:
    write_outputs(parse_args())


if __name__ == "__main__":
    main()
