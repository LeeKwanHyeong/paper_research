#!/usr/bin/env python3
"""Analyze paired Instacart validation predictions without fitting a model.

The input is a row-aligned parquet containing the frozen B, FULL causal-QKV,
and BOUNDED-QK quantity predictions.  This script performs descriptive error
attribution only.  In particular, the constant-shift calculations use the
same validation targets that they score and are therefore labelled as oracle
diagnostics rather than performance estimates.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import polars as pl


SCHEMA = "hard_lmm_bounded_qk_error_analysis_v1"
CONTRACT_ID = "hard_lmm_bounded_qk_error_diagnostic_v1"
FOLD_SALT = "hard_lmm_bounded_qk_error_diagnostic_v1:20260908"
TARGET_BOUNDARIES = (8.0, 20.0, 25.0, 35.0)
TARGET_STRATA = ("<=p50", "p50-p90", "p90-p95", "p95-p99", ">p99")
HISTORY_STRATA = ("1", "2-3", "4-7", "8-15", "16-31", "32-63")
RECENT_SIGNAL_STRATA = ("<0", "=0", ">0")
TOP_RESIDUAL_FRACTIONS = (0.001, 0.01, 0.05)
MODEL_COLUMNS = {
    "B": "prediction_b",
    "FULL": "prediction_full",
    "BOUNDED_QK": "prediction_bounded_qk",
}
PRIMARY_COMPARISON = ("BOUNDED_QK", "B")
COMPARISONS = (
    PRIMARY_COMPARISON,
    ("BOUNDED_QK", "FULL"),
    ("FULL", "B"),
)


COLUMN_ALIASES = {
    "series_id": (
        "series_id",
        "oper_part_no",
        "part_id",
    ),
    "target_qty": (
        "target_qty",
        "true_qty",
        "target_quantity",
        "quantity_target",
        "demand_qty",
        "y_true",
    ),
    "prediction_b": (
        "prediction_b",
        "pred_b",
        "b_prediction",
        "prediction_B",
        "b_qty_prediction",
        "qty_prediction_b",
    ),
    "prediction_full": (
        "prediction_full",
        "pred_full",
        "full_prediction",
        "prediction_FULL",
        "full_qty_prediction",
        "qty_prediction_full",
    ),
    "prediction_bounded_qk": (
        "prediction_bounded_qk",
        "pred_bounded_qk",
        "bounded_qk_prediction",
        "prediction_BOUNDED_QK",
        "bounded_qk_qty_prediction",
        "qty_prediction_bounded_qk",
    ),
    "history_length": (
        "history_length",
        "model_history_length",
        "history_count",
        "model_visible_history_count",
    ),
    "recent_signal": (
        "recent_signal",
        "recent3_mean_log_minus_history_mean_log",
        "recent3_minus_history_mean_log",
        "recent3_minus_history_mean_log_qty",
        "recent3_log_qty_minus_history_log_qty",
        "recent3_mean_log_qty_minus_history_mean_log_qty",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--detail-output-dir",
        type=Path,
        help="Optional external directory for high-cardinality per-series rows.",
    )
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--run-audit", type=Path)
    parser.add_argument("--series-column")
    parser.add_argument("--target-column")
    parser.add_argument("--b-column")
    parser.add_argument("--full-column")
    parser.add_argument("--bounded-qk-column")
    parser.add_argument("--history-column")
    parser.add_argument("--recent-signal-column")
    parser.add_argument("--top-series-count", type=int, default=25)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def write_csv_atomic(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_csv(temporary)
    os.replace(temporary, path)


def _series_token(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("numeric series identifiers must be finite integers")
        return f"int:{int(numeric)}"
    if isinstance(value, str):
        if not value:
            raise ValueError("string series identifiers must be nonempty")
        return f"str:{value}"
    raise ValueError("series identifiers must be strings or integer values")


def assign_series_folds(
    series_ids: Sequence[Any], *, salt: str = FOLD_SALT
) -> np.ndarray:
    """Assign whole series deterministically to one of two SHA-256 folds."""
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a nonempty string")
    values = np.asarray(series_ids, dtype=object)
    if values.ndim != 1:
        raise ValueError("series_ids must have shape [N]")
    mapping: dict[str, int] = {}
    result = np.empty(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        token = _series_token(value)
        if token not in mapping:
            digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
            mapping[token] = int.from_bytes(digest[:8], "big") % 2
        result[index] = mapping[token]
    return result


def target_stratum(values: Sequence[float]) -> np.ndarray:
    target = np.asarray(values, dtype=np.float64)
    if target.ndim != 1 or not np.isfinite(target).all():
        raise ValueError("target quantity must be a finite one-dimensional array")
    if np.any(target < 0):
        raise ValueError("target quantity must be nonnegative")
    boundaries = np.asarray(TARGET_BOUNDARIES, dtype=np.float64)
    indices = np.searchsorted(boundaries, target, side="left")
    return np.asarray(TARGET_STRATA, dtype=object)[indices]


def history_stratum(values: Sequence[int]) -> np.ndarray:
    history = np.asarray(values)
    if history.ndim != 1:
        raise ValueError("history length must be one-dimensional")
    try:
        numeric = history.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("history length must be numeric") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("history length must contain finite integers")
    integer = numeric.astype(np.int64)
    if np.any((integer < 1) | (integer > 63)):
        raise ValueError("history length must be within the frozen range [1, 63]")
    result = np.empty(len(integer), dtype=object)
    result[integer == 1] = HISTORY_STRATA[0]
    result[(integer >= 2) & (integer <= 3)] = HISTORY_STRATA[1]
    result[(integer >= 4) & (integer <= 7)] = HISTORY_STRATA[2]
    result[(integer >= 8) & (integer <= 15)] = HISTORY_STRATA[3]
    result[(integer >= 16) & (integer <= 31)] = HISTORY_STRATA[4]
    result[(integer >= 32) & (integer <= 63)] = HISTORY_STRATA[5]
    return result


def recent_signal_stratum(values: Sequence[float]) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError("recent signal must be a finite one-dimensional array")
    result = np.full(len(signal), "=0", dtype=object)
    result[signal < 0.0] = "<0"
    result[signal > 0.0] = ">0"
    return result


def _numeric_vector(frame: pl.DataFrame, column: str, *, nonnegative: bool) -> np.ndarray:
    try:
        value = frame[column].cast(pl.Float64, strict=True).to_numpy()
    except Exception as exc:  # Polars error types vary across supported releases.
        raise ValueError(f"{column} must be numeric") from exc
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"{column} must be finite")
    if nonnegative and np.any(value < 0):
        raise ValueError(f"{column} must be nonnegative")
    return value


def quantity_metrics(target: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if target_array.ndim != 1 or prediction_array.shape != target_array.shape:
        raise ValueError("target and prediction must have the same one-dimensional shape")
    if len(target_array) == 0:
        raise ValueError("metrics require at least one row")
    if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("metrics require finite values")
    error = prediction_array - target_array
    squared_error = np.square(error)
    bias = float(error.mean(dtype=np.float64))
    mse = float(squared_error.mean(dtype=np.float64))
    centered_mse = float(np.square(error - bias).mean(dtype=np.float64))
    return {
        "count": int(len(target_array)),
        "mae": float(np.abs(error).mean(dtype=np.float64)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias": bias,
        "centered_mse": centered_mse,
        "centered_rmse": math.sqrt(centered_mse),
        "bias_squared": bias * bias,
        "mse_decomposition_residual": mse - centered_mse - bias * bias,
        "log1p_mse": float(
            np.square(np.log1p(prediction_array) - np.log1p(target_array)).mean(
                dtype=np.float64
            )
        ),
    }


def _safe_fraction(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def comparison_metrics(
    target: Sequence[float],
    candidate: Sequence[float],
    reference: Sequence[float],
) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    if candidate_array.shape != target_array.shape or reference_array.shape != target_array.shape:
        raise ValueError("paired arrays must have identical shape")
    candidate_metrics = quantity_metrics(target_array, candidate_array)
    reference_metrics = quantity_metrics(target_array, reference_array)
    reference_error = reference_array - target_array
    prediction_shift = candidate_array - reference_array
    delta_se = np.square(candidate_array - target_array) - np.square(
        reference_array - target_array
    )
    delta_ae = np.abs(candidate_array - target_array) - np.abs(
        reference_array - target_array
    )
    positive = delta_se[delta_se > 0.0]
    negative = delta_se[delta_se < 0.0]
    cross_term_sum = float(
        np.multiply(2.0 * reference_error, prediction_shift).sum(dtype=np.float64)
    )
    prediction_shift_squared_sum = float(
        np.square(prediction_shift).sum(dtype=np.float64)
    )
    delta_se_sum = float(delta_se.sum(dtype=np.float64))
    positive_mass = float(positive.sum(dtype=np.float64))
    negative_mass = float(-negative.sum(dtype=np.float64))
    delta_centered_mse = (
        candidate_metrics["centered_mse"] - reference_metrics["centered_mse"]
    )
    delta_bias_squared = (
        candidate_metrics["bias_squared"] - reference_metrics["bias_squared"]
    )
    delta_mse = candidate_metrics["mse"] - reference_metrics["mse"]
    return {
        "count": int(len(target_array)),
        "delta_mae": candidate_metrics["mae"] - reference_metrics["mae"],
        "delta_mse": delta_mse,
        "delta_rmse": candidate_metrics["rmse"] - reference_metrics["rmse"],
        "delta_bias": candidate_metrics["bias"] - reference_metrics["bias"],
        "delta_centered_mse": delta_centered_mse,
        "delta_centered_rmse": (
            candidate_metrics["centered_rmse"] - reference_metrics["centered_rmse"]
        ),
        "delta_bias_squared": delta_bias_squared,
        "delta_mse_decomposition_residual": (
            delta_mse - delta_centered_mse - delta_bias_squared
        ),
        "delta_log1p_mse": (
            candidate_metrics["log1p_mse"] - reference_metrics["log1p_mse"]
        ),
        "delta_se_sum": delta_se_sum,
        "two_base_error_times_prediction_shift_sum": cross_term_sum,
        "prediction_shift_squared_sum": prediction_shift_squared_sum,
        "delta_se_formula_residual_sum": (
            delta_se_sum - cross_term_sum - prediction_shift_squared_sum
        ),
        "delta_ae_sum": float(delta_ae.sum(dtype=np.float64)),
        "gross_positive_harm_mass": positive_mass,
        "gross_negative_help_mass": negative_mass,
        "delta_se_mass_residual": delta_se_sum - positive_mass + negative_mass,
        "harm_row_count": int(np.count_nonzero(delta_se > 0.0)),
        "help_row_count": int(np.count_nonzero(delta_se < 0.0)),
        "tie_row_count": int(np.count_nonzero(delta_se == 0.0)),
        "harm_row_share": float(np.mean(delta_se > 0.0)),
        "help_row_share": float(np.mean(delta_se < 0.0)),
        "tie_row_share": float(np.mean(delta_se == 0.0)),
    }


def validation_oracle_shift_diagnostics(
    target: Sequence[float],
    candidate: Sequence[float],
    reference: Sequence[float],
) -> list[dict[str, Any]]:
    """Return target-derived additive-shift diagnostics, never a score claim."""
    target_array = np.asarray(target, dtype=np.float64)
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    candidate_bias = quantity_metrics(target_array, candidate_array)["bias"]
    reference_bias = quantity_metrics(target_array, reference_array)["bias"]
    definitions = (
        ("match_reference_bias", reference_bias - candidate_bias),
        ("zero_candidate_bias", -candidate_bias),
    )
    rows: list[dict[str, Any]] = []
    for rule, shift in definitions:
        shifted = candidate_array + shift
        metrics = quantity_metrics(target_array, shifted)
        rows.append(
            {
                "rule": rule,
                "shift": float(shift),
                "uses_validation_targets": True,
                "descriptive_only": True,
                "eligible_as_performance_result": False,
                "nonnegative_projection_applied": False,
                "negative_shifted_prediction_count": int(np.count_nonzero(shifted < 0.0)),
                **metrics,
            }
        )
    return rows


def _resolve_column(
    columns: set[str], canonical: str, explicit: str | None
) -> str:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"explicit column is missing: {explicit}")
        return explicit
    matches = [name for name in COLUMN_ALIASES[canonical] if name in columns]
    if not matches:
        raise ValueError(
            f"missing {canonical}; accepted aliases are {COLUMN_ALIASES[canonical]}"
        )
    if len(matches) > 1:
        raise ValueError(f"ambiguous aliases for {canonical}: {matches}")
    return matches[0]


def canonicalize_frame(
    frame: pl.DataFrame, *, overrides: Mapping[str, str | None] | None = None
) -> tuple[pl.DataFrame, dict[str, str]]:
    overrides = dict(overrides or {})
    columns = set(frame.columns)
    resolved = {
        canonical: _resolve_column(columns, canonical, overrides.get(canonical))
        for canonical in COLUMN_ALIASES
    }
    renamed = frame.select(
        [pl.col(source).alias(canonical) for canonical, source in resolved.items()]
        + [
            pl.col(name)
            for name in frame.columns
            if name not in set(resolved.values())
        ]
    )
    if len(renamed) == 0:
        raise ValueError("paired prediction parquet is empty")
    if "split" in renamed.columns:
        splits = set(renamed["split"].cast(pl.String).unique().to_list())
        if splits != {"validation"}:
            raise ValueError(f"only validation rows are allowed, found splits={splits}")
    return renamed, resolved


def prepare_frame(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    target = _numeric_vector(frame, "target_qty", nonnegative=True)
    predictions = {
        role: _numeric_vector(frame, column, nonnegative=True)
        for role, column in MODEL_COLUMNS.items()
    }
    history = _numeric_vector(frame, "history_length", nonnegative=True)
    recent = _numeric_vector(frame, "recent_signal", nonnegative=False)
    series = frame["series_id"].to_numpy()
    folds = assign_series_folds(series)
    if set(np.unique(folds).tolist()) != {0, 1}:
        raise ValueError("deterministic series folds must both be nonempty")

    derived = frame.with_columns(
        pl.Series("target_stratum", target_stratum(target), dtype=pl.String),
        pl.Series("history_stratum", history_stratum(history), dtype=pl.String),
        pl.Series(
            "recent_signal_stratum", recent_signal_stratum(recent), dtype=pl.String
        ),
        pl.Series("series_fold", folds, dtype=pl.Int8),
    )
    for role, prediction in predictions.items():
        key = role.lower()
        error = prediction - target
        derived = derived.with_columns(
            pl.Series(f"error_{key}", error, dtype=pl.Float64),
            pl.Series(f"abs_error_{key}", np.abs(error), dtype=pl.Float64),
            pl.Series(f"squared_error_{key}", np.square(error), dtype=pl.Float64),
            pl.Series(
                f"log1p_squared_error_{key}",
                np.square(np.log1p(prediction) - np.log1p(target)),
                dtype=pl.Float64,
            ),
        )
    derived = derived.with_columns(
        (
            pl.col("squared_error_bounded_qk") - pl.col("squared_error_b")
        ).alias("delta_se_bounded_qk_minus_b"),
        (pl.col("abs_error_bounded_qk") - pl.col("abs_error_b")).alias(
            "delta_ae_bounded_qk_minus_b"
        ),
        (pl.col("squared_error_full") - pl.col("squared_error_b")).alias(
            "delta_se_full_minus_b"
        ),
        (pl.col("abs_error_full") - pl.col("abs_error_b")).alias(
            "delta_ae_full_minus_b"
        ),
    )

    unique_series = len(set(_series_token(value) for value in series))
    audit = {
        "row_count": len(derived),
        "series_count": unique_series,
        "target_and_predictions_finite_nonnegative": True,
        "history_range": [int(np.min(history)), int(np.max(history))],
        "recent_signal_range": [float(np.min(recent)), float(np.max(recent))],
        "series_fold_counts": {
            str(fold): int(np.count_nonzero(folds == fold)) for fold in (0, 1)
        },
        "series_fold_series_counts": {
            str(fold): len(
                {
                    _series_token(value)
                    for value, assigned in zip(series, folds, strict=True)
                    if assigned == fold
                }
            )
            for fold in (0, 1)
        },
    }
    return derived, audit


def _subset_arrays(frame: pl.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = frame["target_qty"].cast(pl.Float64).to_numpy()
    predictions = {
        role: frame[column].cast(pl.Float64).to_numpy()
        for role, column in MODEL_COLUMNS.items()
    }
    return target, predictions


def summarize_subset(frame: pl.DataFrame) -> dict[str, Any]:
    if len(frame) == 0:
        raise ValueError("cannot summarize an empty subset")
    target, predictions = _subset_arrays(frame)
    metrics = {
        role: quantity_metrics(target, prediction)
        for role, prediction in predictions.items()
    }
    comparisons = {}
    for candidate, reference in COMPARISONS:
        key = f"{candidate}_minus_{reference}"
        comparisons[key] = comparison_metrics(
            target, predictions[candidate], predictions[reference]
        )
    return {"count": len(frame), "metrics": metrics, "comparisons": comparisons}


def _flatten_subset_row(
    *, scope: str, group_key: str, group_label: str, summary: Mapping[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scope": scope,
        "group_key": group_key,
        "group_label": group_label,
        "count": int(summary["count"]),
    }
    for role, metrics in summary["metrics"].items():
        for metric, value in metrics.items():
            if metric != "count":
                row[f"{role.lower()}_{metric}"] = value
    for comparison, metrics in summary["comparisons"].items():
        prefix = comparison.lower()
        for metric, value in metrics.items():
            if metric != "count":
                row[f"{prefix}_{metric}"] = value
    return row


def grouped_summary(
    frame: pl.DataFrame,
    *,
    column: str,
    labels: Iterable[str],
    scope: str,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    nested: list[dict[str, Any]] = []
    for order, label in enumerate(labels):
        subset = frame.filter(pl.col(column) == label)
        if len(subset) == 0:
            continue
        summary = summarize_subset(subset)
        flat = _flatten_subset_row(
            scope=scope, group_key=column, group_label=label, summary=summary
        )
        flat["group_order"] = order
        rows.append(flat)
        nested.append({"label": label, **summary})
    return pl.DataFrame(rows), nested


def add_mass_shares(
    frame: pl.DataFrame, *, overall_comparison: Mapping[str, Any]
) -> pl.DataFrame:
    if len(frame) == 0:
        return frame
    prefix = "bounded_qk_minus_b_"
    positive_col = prefix + "gross_positive_harm_mass"
    negative_col = prefix + "gross_negative_help_mass"
    total_positive = float(overall_comparison["gross_positive_harm_mass"])
    total_negative = float(overall_comparison["gross_negative_help_mass"])
    return frame.with_columns(
        (pl.col("count") / pl.col("count").sum()).alias("row_share_within_table"),
        (
            pl.col(positive_col) / total_positive
            if total_positive > 0.0
            else pl.lit(None, dtype=pl.Float64)
        ).alias("share_of_all_gross_positive_harm_mass"),
        (
            pl.col(negative_col) / total_negative
            if total_negative > 0.0
            else pl.lit(None, dtype=pl.Float64)
        ).alias("share_of_all_gross_negative_help_mass"),
    )


def series_contributions(frame: pl.DataFrame, *, scope: str) -> pl.DataFrame:
    if len(frame) == 0:
        return pl.DataFrame()
    delta = pl.col("delta_se_bounded_qk_minus_b")
    result = (
        frame.group_by(["series_id", "series_fold"])
        .agg(
            pl.len().alias("count"),
            pl.col("target_qty").mean().alias("target_qty_mean"),
            pl.col("history_length").mean().alias("history_length_mean"),
            pl.col("recent_signal").mean().alias("recent_signal_mean"),
            delta.sum().alias("net_delta_se_sum"),
            delta.mean().alias("mean_delta_se"),
            pl.col("delta_ae_bounded_qk_minus_b").sum().alias("delta_ae_sum"),
            pl.when(delta > 0.0)
            .then(delta)
            .otherwise(0.0)
            .sum()
            .alias("gross_positive_harm_mass"),
            pl.when(delta < 0.0)
            .then(-delta)
            .otherwise(0.0)
            .sum()
            .alias("gross_negative_help_mass"),
            (delta > 0.0).sum().alias("harm_row_count"),
            (delta < 0.0).sum().alias("help_row_count"),
        )
        .with_columns(pl.lit(scope).alias("scope"))
        .select("scope", pl.all().exclude("scope"))
    )
    positive_total = pl.col("gross_positive_harm_mass").sum().over("scope")
    negative_total = pl.col("gross_negative_help_mass").sum().over("scope")
    row_total = pl.col("count").sum().over("scope")
    return (
        result.with_columns(
            (pl.col("count") / row_total).alias("share_of_scope_rows"),
            pl.when(positive_total > 0.0)
            .then(pl.col("gross_positive_harm_mass") / positive_total)
            .otherwise(None)
            .alias("share_of_scope_gross_positive_harm_mass"),
            pl.when(negative_total > 0.0)
            .then(pl.col("gross_negative_help_mass") / negative_total)
            .otherwise(None)
            .alias("share_of_scope_gross_negative_help_mass"),
        )
        .sort(["net_delta_se_sum", "series_id"], descending=[True, False])
    )


def top_series_rows(contributions: pl.DataFrame, *, count: int) -> pl.DataFrame:
    if count < 1:
        raise ValueError("top series count must be positive")
    if len(contributions) == 0:
        return pl.DataFrame()
    outputs = []
    for scope in contributions["scope"].unique(maintain_order=True).to_list():
        scoped = contributions.filter(pl.col("scope") == scope)
        for direction, descending in (("harm", True), ("help", False)):
            directional = scoped.filter(
                pl.col("net_delta_se_sum") > 0.0
                if direction == "harm"
                else pl.col("net_delta_se_sum") < 0.0
            )
            if len(directional) == 0:
                continue
            ranked = (
                directional.sort(
                    ["net_delta_se_sum", "series_id"],
                    descending=[descending, False],
                )
                .head(count)
                .with_row_index("rank", offset=1)
                .with_columns(pl.lit(direction).alias("direction"))
            )
            outputs.append(ranked)
    if not outputs:
        return contributions.head(0).with_columns(
            pl.lit(None, dtype=pl.UInt32).alias("rank"),
            pl.lit(None, dtype=pl.String).alias("direction"),
        )
    return pl.concat(outputs, how="diagonal_relaxed")


def fold_summaries(frame: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope, scoped in (
        ("all", frame),
        ("p50-p90", frame.filter(pl.col("target_stratum") == "p50-p90")),
    ):
        for fold in (0, 1):
            subset = scoped.filter(pl.col("series_fold") == fold)
            if len(subset) == 0:
                continue
            row = _flatten_subset_row(
                scope=scope,
                group_key="series_fold",
                group_label=str(fold),
                summary=summarize_subset(subset),
            )
            row["series_count"] = subset["series_id"].n_unique()
            rows.append(row)
    return pl.DataFrame(rows)


def top_residual_delta_rows(
    frame: pl.DataFrame,
    *,
    fractions: Sequence[float] = TOP_RESIDUAL_FRACTIONS,
) -> pl.DataFrame:
    delta = frame["delta_se_bounded_qk_minus_b"].cast(pl.Float64).to_numpy()
    n = len(delta)
    if n == 0:
        raise ValueError("top residual analysis requires rows")
    positive_total = float(delta[delta > 0.0].sum(dtype=np.float64))
    negative_total = float(-delta[delta < 0.0].sum(dtype=np.float64))
    absolute_total = float(np.abs(delta).sum(dtype=np.float64))
    rows: list[dict[str, Any]] = []
    modes = {
        "largest_positive_harm": np.argsort(-delta, kind="stable"),
        "largest_negative_help": np.argsort(delta, kind="stable"),
        "largest_absolute_change": np.argsort(-np.abs(delta), kind="stable"),
    }
    for fraction in fractions:
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("top residual fractions must be in (0, 1]")
        k = max(1, int(math.ceil(n * fraction)))
        for mode, order in modes.items():
            selected = delta[order[:k]]
            positive_mass = float(selected[selected > 0.0].sum(dtype=np.float64))
            negative_mass = float(-selected[selected < 0.0].sum(dtype=np.float64))
            absolute_mass = float(np.abs(selected).sum(dtype=np.float64))
            rows.append(
                {
                    "fraction": float(fraction),
                    "selection": mode,
                    "selected_count": k,
                    "selected_row_share": k / n,
                    "net_delta_se_sum": float(selected.sum(dtype=np.float64)),
                    "gross_positive_harm_mass": positive_mass,
                    "gross_negative_help_mass": negative_mass,
                    "share_of_all_gross_positive_harm_mass": _safe_fraction(
                        positive_mass, positive_total
                    ),
                    "share_of_all_gross_negative_help_mass": _safe_fraction(
                        negative_mass, negative_total
                    ),
                    "share_of_all_absolute_delta_mass": _safe_fraction(
                        absolute_mass, absolute_total
                    ),
                }
            )
    return pl.DataFrame(rows)


def _oracle_rows(
    frame: pl.DataFrame, *, scope: str, fold: int | None = None
) -> list[dict[str, Any]]:
    target, prediction = _subset_arrays(frame)
    rows = []
    for candidate, reference in COMPARISONS:
        for diagnostic in validation_oracle_shift_diagnostics(
            target, prediction[candidate], prediction[reference]
        ):
            rows.append(
                {
                    "scope": scope,
                    "series_fold": fold,
                    "candidate": candidate,
                    "reference": reference,
                    **diagnostic,
                }
            )
    return rows


def load_contract(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    actual_id = payload.get("contract_id", payload.get("schema"))
    if actual_id != CONTRACT_ID:
        raise ValueError(f"wrong diagnostic contract: {actual_id!r}")
    if payload.get("status") != "frozen_before_row_level_inference":
        raise ValueError("diagnostic contract was not frozen before paired inference")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("contract scope is missing")
    expected_scope = {
        "dataset": "insta_market_basket",
        "seed": 42,
        "target_split": "validation",
        "held_out_test": False,
        "training": False,
        "checkpoint_selection": False,
        "parameter_updates": False,
        "calibration_fit": False,
        "primary_contrast": "BOUNDED_QK_minus_B",
    }
    for name, expected in expected_scope.items():
        if scope.get(name) != expected:
            raise ValueError(f"contract scope drift: {name}")
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("contract dataset is missing")
    if dataset.get("train_quantity_boundaries") != list(TARGET_BOUNDARIES):
        raise ValueError("target quantity boundaries drift")
    analysis = payload.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("contract analysis definition is missing")
    expected_analysis = {
        "history_bins": list(HISTORY_STRATA),
        "recent_condition": "sign(recent3_mean_log_qty - history_mean_log_qty)",
        "top_positive_delta_se_fractions": list(TOP_RESIDUAL_FRACTIONS),
        "deterministic_series_fold_count": 2,
        "deterministic_series_fold_salt": FOLD_SALT,
    }
    for name, expected in expected_analysis.items():
        if analysis.get(name) != expected:
            raise ValueError(f"contract analysis drift: {name}")
    oracle = analysis.get("oracle_counterfactual")
    if not isinstance(oracle, Mapping) or oracle.get("performance_result") is not False:
        raise ValueError("oracle counterfactual is not explicitly non-performance")
    return payload


def validate_against_contract(
    frame: pl.DataFrame,
    analysis: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    dataset = contract["dataset"]
    expected_rows = int(dataset["expected_validation_targets"])
    if len(frame) != expected_rows:
        raise ValueError(
            f"paired row count drift: expected {expected_rows}, observed {len(frame)}"
        )
    identity_fields = [str(name) for name in dataset.get("identity_fields", [])]
    missing_identity = [name for name in identity_fields if name not in frame.columns]
    if missing_identity:
        raise ValueError(f"paired identity columns are missing: {missing_identity}")
    unique_targets = frame.select(pl.struct(identity_fields).n_unique()).item()
    if int(unique_targets) != len(frame):
        raise ValueError("paired target identities are not unique")

    tolerance = float(
        contract.get("execution", {}).get("aggregate_replay_absolute_tolerance", 1e-5)
    )
    observed_by_role = analysis["overall"]["metrics"]
    differences: dict[str, dict[str, float]] = {}
    metric_map = {
        "qty_mae": "mae",
        "qty_rmse": "rmse",
        "qty_bias": "bias",
        "log_qty_mse": "log1p_mse",
    }
    for role, specification in contract["models"].items():
        expected = specification["expected_validation"]
        if int(expected["count"]) != len(frame):
            raise ValueError(f"contract model count drift: {role}")
        differences[role] = {}
        for expected_name, observed_name in metric_map.items():
            difference = float(observed_by_role[role][observed_name]) - float(
                expected[expected_name]
            )
            differences[role][expected_name] = difference
            if abs(difference) > tolerance:
                raise ValueError(
                    f"aggregate replay drift for {role}.{expected_name}: {difference}"
                )
    return {
        "expected_row_count_matched": True,
        "identity_fields": identity_fields,
        "unique_target_count": int(unique_targets),
        "aggregate_replay_absolute_tolerance": tolerance,
        "aggregate_metric_differences_observed_minus_expected": differences,
        "all_expected_aggregates_matched": True,
    }


def validate_run_audit(
    path: Path,
    *,
    input_parquet: Path,
    contract_path: Path | None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "hard_lmm_bounded_qk_error_diagnostic_run_v1",
        "status": "success",
        "held_out_test_evaluated": False,
        "training_performed": False,
        "calibration_fit_performed": False,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"run audit drift: {name}")
    scope = payload.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("run audit scope is missing")
    for name in (
        "held_out_test",
        "training",
        "checkpoint_selection",
        "parameter_updates",
        "calibration_fit",
    ):
        if scope.get(name) is not False:
            raise ValueError(f"run audit enables forbidden operation: {name}")
    if scope.get("target_split") != "validation":
        raise ValueError("run audit is not validation-only")

    paired = payload.get("paired_predictions")
    if not isinstance(paired, Mapping):
        raise ValueError("run audit paired-prediction record is missing")
    observed_parquet_sha = sha256_file(input_parquet)
    if paired.get("sha256") != observed_parquet_sha:
        raise ValueError("paired parquet SHA-256 does not match the run audit")
    if int(paired.get("row_count", -1)) <= 0:
        raise ValueError("run audit paired row count is invalid")
    if paired.get("canonical_identity_unique") is not True:
        raise ValueError("run audit did not verify canonical identity uniqueness")

    state = payload.get("state_identity")
    if not isinstance(state, Mapping):
        raise ValueError("run audit state-identity record is missing")
    if state.get("unchanged") is not True or state.get("all_gradients_absent") is not True:
        raise ValueError("run audit did not preserve frozen source models")
    if state.get("before") != state.get("after"):
        raise ValueError("run audit source state hashes changed")

    contract_record = payload.get("contract")
    if not isinstance(contract_record, Mapping) or contract_record.get("id") != CONTRACT_ID:
        raise ValueError("run audit contract identity drift")
    if contract_path is not None:
        observed_contract_sha = sha256_file(contract_path)
        if contract_record.get("sha256") != observed_contract_sha:
            raise ValueError("contract SHA-256 does not match the run audit")

    return {
        "file": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "schema": payload["schema"],
        "status": payload["status"],
        "paired_prediction_file_sha256": observed_parquet_sha,
        "paired_prediction_row_count": int(paired["row_count"]),
        "contract_id": contract_record["id"],
        "contract_file_sha256": contract_record.get("sha256"),
        "validation_only": True,
        "held_out_test_evaluated": False,
        "training_performed": False,
        "calibration_fit_performed": False,
        "source_model_states_unchanged": True,
        "source_gradients_absent": True,
    }


def analyze(
    frame: pl.DataFrame,
    *,
    top_series_count: int = 25,
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    prepared, input_audit = prepare_frame(frame)
    overall = summarize_subset(prepared)
    primary = overall["comparisons"]["BOUNDED_QK_minus_B"]

    target_table, target_nested = grouped_summary(
        prepared,
        column="target_stratum",
        labels=TARGET_STRATA,
        scope="all",
    )
    history_table, history_nested = grouped_summary(
        prepared,
        column="history_stratum",
        labels=HISTORY_STRATA,
        scope="all",
    )
    recent_table, recent_nested = grouped_summary(
        prepared,
        column="recent_signal_stratum",
        labels=RECENT_SIGNAL_STRATA,
        scope="all",
    )
    p50_p90 = prepared.filter(pl.col("target_stratum") == "p50-p90")
    p50_history_table, p50_history_nested = grouped_summary(
        p50_p90,
        column="history_stratum",
        labels=HISTORY_STRATA,
        scope="p50-p90",
    )
    p50_recent_table, p50_recent_nested = grouped_summary(
        p50_p90,
        column="recent_signal_stratum",
        labels=RECENT_SIGNAL_STRATA,
        scope="p50-p90",
    )
    tables_to_share = (
        target_table,
        history_table,
        recent_table,
        p50_history_table,
        p50_recent_table,
    )
    target_table, history_table, recent_table, p50_history_table, p50_recent_table = (
        add_mass_shares(table, overall_comparison=primary) for table in tables_to_share
    )

    all_series = series_contributions(prepared, scope="all")
    p50_series = series_contributions(p50_p90, scope="p50-p90")
    contributions = pl.concat([all_series, p50_series], how="diagonal_relaxed")
    top_series = top_series_rows(contributions, count=top_series_count)
    folds = fold_summaries(prepared)
    residual_concentration = top_residual_delta_rows(prepared)

    oracle_rows = _oracle_rows(prepared, scope="all")
    for fold in (0, 1):
        oracle_rows.extend(
            _oracle_rows(
                prepared.filter(pl.col("series_fold") == fold),
                scope="series_fold",
                fold=fold,
            )
        )
    oracle = pl.DataFrame(oracle_rows)

    overall_metrics_rows = []
    for role, metrics in overall["metrics"].items():
        overall_metrics_rows.append({"role": role, **metrics})
    overall_metrics = pl.DataFrame(overall_metrics_rows)
    comparison_rows = [
        {"comparison": comparison, **metrics}
        for comparison, metrics in overall["comparisons"].items()
    ]
    comparisons = pl.DataFrame(comparison_rows)

    decomposition_audit = {}
    for name, values in overall["comparisons"].items():
        formula_tolerance = max(
            1e-8,
            1e-12
            * (
                abs(float(values["delta_se_sum"]))
                + abs(float(values["two_base_error_times_prediction_shift_sum"]))
                + abs(float(values["prediction_shift_squared_sum"]))
            ),
        )
        mass_tolerance = max(
            1e-8,
            1e-12
            * (
                abs(float(values["delta_se_sum"]))
                + abs(float(values["gross_positive_harm_mass"]))
                + abs(float(values["gross_negative_help_mass"]))
            ),
        )
        decomposition_audit[name] = {
            "delta_mse_equals_delta_centered_mse_plus_delta_bias_squared": abs(
                float(values["delta_mse_decomposition_residual"])
            )
            <= 1e-12,
            (
                "delta_se_equals_two_base_error_times_delta_prediction_"
                "plus_delta_prediction_squared"
            ): abs(float(values["delta_se_formula_residual_sum"]))
            <= formula_tolerance,
            "gross_positive_minus_gross_negative_equals_net_delta_se": abs(
                float(values["delta_se_mass_residual"])
            )
            <= mass_tolerance,
            "numeric_residuals": {
                "mse_decomposition": values["delta_mse_decomposition_residual"],
                "delta_se_formula_sum": values["delta_se_formula_residual_sum"],
                "signed_mass_sum": values["delta_se_mass_residual"],
            },
            "numeric_tolerances": {
                "mse_decomposition": 1e-12,
                "delta_se_formula_sum": formula_tolerance,
                "signed_mass_sum": mass_tolerance,
            },
        }
    if not all(
        all(
            value is True
            for key, value in row.items()
            if key not in {"numeric_residuals", "numeric_tolerances"}
        )
        for row in decomposition_audit.values()
    ):
        raise FloatingPointError("a required paired-error decomposition failed")

    primary_candidate = overall["metrics"][PRIMARY_COMPARISON[0]]
    primary_reference = overall["metrics"][PRIMARY_COMPARISON[1]]
    fold_bias_rows = []
    for fold_row in folds.filter(pl.col("scope") == "all").to_dicts():
        fold_bias_rows.append(
            {
                "series_fold": int(fold_row["group_label"]),
                "candidate_more_negative_bias": (
                    fold_row["bounded_qk_bias"] < fold_row["b_bias"]
                ),
                "candidate_centered_mse_lower": (
                    fold_row["bounded_qk_centered_mse"]
                    < fold_row["b_centered_mse"]
                ),
                "candidate_raw_mse_higher": (
                    fold_row["bounded_qk_mse"] > fold_row["b_mse"]
                ),
            }
        )
    bias_explanation = {
        "candidate": PRIMARY_COMPARISON[0],
        "reference": PRIMARY_COMPARISON[1],
        "candidate_more_negative_bias": (
            primary_candidate["bias"] < primary_reference["bias"]
        ),
        "candidate_centered_mse_lower": (
            primary_candidate["centered_mse"] < primary_reference["centered_mse"]
        ),
        "candidate_raw_mse_higher": (
            primary_candidate["mse"] > primary_reference["mse"]
        ),
        "delta_centered_mse": primary["delta_centered_mse"],
        "delta_bias_squared": primary["delta_bias_squared"],
        "delta_mse": primary["delta_mse"],
        "bias_term_reverses_centered_mse_gain": (
            primary["delta_mse"] > 0.0
            and primary["delta_centered_mse"] < 0.0
            and primary["delta_bias_squared"] > primary["delta_mse"]
        ),
        "series_fold_rows": fold_bias_rows,
        "same_pattern_in_both_series_folds": (
            len(fold_bias_rows) == 2
            and all(
                row["candidate_more_negative_bias"]
                and row["candidate_centered_mse_lower"]
                and row["candidate_raw_mse_higher"]
                for row in fold_bias_rows
            )
        ),
        "interpretation_limit": (
            "This algebra localizes the validation MSE gap. It does not validate a "
            "deployable correction or establish held-out improvement."
        ),
    }

    analysis = {
        "schema": SCHEMA,
        "status": "success",
        "created_at_utc": utc_now(),
        "scope": {
            "dataset": "insta_market_basket",
            "split": "validation",
            "held_out_test_evaluated": False,
            "models_frozen": True,
            "predictions_modified": False,
            "descriptive_error_attribution_only": True,
        },
        "definitions": {
            "error": "prediction - target; negative bias means underprediction",
            "primary_delta": (
                "BOUNDED_QK squared error minus B squared error; positive means "
                "BOUNDED_QK is worse"
            ),
            "target_boundaries_train_only": list(TARGET_BOUNDARIES),
            "target_strata": list(TARGET_STRATA),
            "history_strata": list(HISTORY_STRATA),
            "history_length": "number of model-visible prior events; target excluded",
            "recent_signal": (
                "recent3 mean log quantity minus full model-visible history mean log "
                "quantity; target excluded"
            ),
            "recent_signal_strata": list(RECENT_SIGNAL_STRATA),
            "series_fold_rule": (
                "SHA256(f'{salt}:{typed_series_id}') first 8 bytes modulo 2"
            ),
            "series_fold_salt": FOLD_SALT,
            "mass_wording": {
                "gross_positive_harm_mass": "sum of positive per-row delta squared errors",
                "gross_negative_help_mass": "absolute sum of negative per-row delta squared errors",
                "net_delta_se_sum": "gross positive harm minus gross negative help",
            },
        },
        "input_audit": input_audit,
        "overall": overall,
        "required_decomposition_audit": decomposition_audit,
        "primary_bias_explanation_diagnostic": bias_explanation,
        "target_strata": target_nested,
        "history_strata": history_nested,
        "recent_signal_strata": recent_nested,
        "p50_p90_cross_tables": {
            "by_history": p50_history_nested,
            "by_recent_signal": p50_recent_nested,
        },
        "fold_consistency": folds.to_dicts(),
        "top_residual_delta_concentration": residual_concentration.to_dicts(),
        "validation_oracle_constant_shift_diagnostics": {
            "warning": (
                "Shifts are derived from validation targets and scored on those same targets. "
                "They are descriptive bias decompositions, not performance results, calibration "
                "candidates, or evidence of generalization."
            ),
            "eligible_as_performance_result": False,
            "rows": oracle.to_dicts(),
        },
    }
    outputs = {
        "overall_metrics.csv": overall_metrics,
        "comparisons.csv": comparisons,
        "target_strata.csv": target_table,
        "history_strata.csv": history_table,
        "recent_signal_strata.csv": recent_table,
        "p50_p90_by_history.csv": p50_history_table,
        "p50_p90_by_recent_signal.csv": p50_recent_table,
        "series_folds.csv": folds,
        "series_contributions.csv": contributions,
        "top_series.csv": top_series,
        "top_residual_delta.csv": residual_concentration,
        "validation_oracle_shifts.csv": oracle,
    }
    return analysis, outputs


def main() -> None:
    args = parse_args()
    if args.top_series_count < 1:
        raise ValueError("--top-series-count must be positive")
    contract = load_contract(args.contract)
    run_audit = (
        validate_run_audit(
            args.run_audit,
            input_parquet=args.input_parquet,
            contract_path=args.contract,
        )
        if args.run_audit
        else None
    )
    source = pl.read_parquet(args.input_parquet)
    canonical, resolved = canonicalize_frame(
        source,
        overrides={
            "series_id": args.series_column,
            "target_qty": args.target_column,
            "prediction_b": args.b_column,
            "prediction_full": args.full_column,
            "prediction_bounded_qk": args.bounded_qk_column,
            "history_length": args.history_column,
            "recent_signal": args.recent_signal_column,
        },
    )
    analysis, tables = analyze(canonical, top_series_count=args.top_series_count)
    analysis["input"] = {
        "paired_prediction_file": str(args.input_parquet.resolve()),
        "paired_prediction_file_sha256": sha256_file(args.input_parquet),
        "source_columns": source.columns,
        "resolved_columns": resolved,
        "contract_file": str(args.contract.resolve()) if args.contract else None,
        "contract_file_sha256": sha256_file(args.contract) if args.contract else None,
        "contract_loaded": contract is not None,
        "run_audit": run_audit,
    }
    if contract is not None:
        analysis["contract_audit"] = validate_against_contract(
            canonical, analysis, contract
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_manifest: dict[str, Any] = {}
    for name, table in tables.items():
        external_detail = name == "series_contributions.csv" and args.detail_output_dir is not None
        path = (args.detail_output_dir if external_detail else args.output_dir) / name
        write_csv_atomic(path, table)
        output_manifest[name] = {
            "rows": len(table),
            "sha256": sha256_file(path),
            "storage": "external_diagnostic_detail" if external_detail else "result_directory",
            "path": str(path.resolve()),
        }
    analysis["outputs"] = output_manifest
    analysis_path = args.output_dir / "analysis.json"
    write_json_atomic(analysis_path, analysis)
    print(json.dumps({
        "status": "success",
        "analysis": str(analysis_path),
        "analysis_sha256": sha256_file(analysis_path),
        "row_count": analysis["input_audit"]["row_count"],
        "primary_delta_mse": analysis["overall"]["comparisons"]["BOUNDED_QK_minus_B"]["delta_mse"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
