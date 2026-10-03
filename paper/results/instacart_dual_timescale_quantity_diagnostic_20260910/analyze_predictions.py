#!/usr/bin/env python3
"""Descriptive, row-paired quantity-error analysis for the Instacart study.

This module never trains, selects, fits, or calibrates a model.  It consumes a
single row-aligned parquet produced by the separately audited inference step
and attributes candidate-minus-reference quantity errors on the requested
train or validation split.  Hash folds are descriptive replication cuts; they
are not out-of-fold predictions or estimates of seed uncertainty.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


SCHEMA = "instacart_dual_timescale_quantity_analysis_v1"
CONTRACT_ID = "instacart_dual_timescale_quantity_v1"
CONTRACT_STATUS = "frozen_before_row_level_inference"
FOLD_SALT = "instacart_dual_timescale_quantity_v1:20260910"
BOOTSTRAP_SEED = 20260910
BOOTSTRAP_REPLICATES = 1000
CONCENTRATION_FRACTIONS = (0.001, 0.01, 0.05)

MODEL_COLUMNS = {
    "b": "prediction_b",
    "candidate": "prediction_candidate",
    "rmtpp": "prediction_rmtpp",
    "thp": "prediction_thp",
}
REFERENCE_ROLES = ("b", "rmtpp", "thp")
PRIMARY_COMPARISON = "candidate_minus_rmtpp"

QUANTITY_BOUNDARIES = (8.0, 20.0, 25.0, 35.0)
QUANTITY_LABELS = ("<=8", "(8,20]", "(20,25]", "(25,35]", ">35")
HISTORY_LABELS = ("1", "2-3", "4-7", "8-15", "16-31", "32-63")
RECENT_SIGN_LABELS = ("<0", "=0", ">0")

REQUIRED_COLUMNS = (
    "series_id",
    "series_index",
    "context_end",
    "target_position",
    "target_seq",
    "history_length",
    "true_qty",
    "target_dt",
    "last_qty",
    "history_mean_qty",
    "history_mean_log_qty",
    "recent3_minus_history_mean_log_qty",
    "last_minus_history_mean_log_qty",
    "prediction_b",
    "prediction_candidate",
    "prediction_rmtpp",
    "prediction_thp",
)

INTEGER_COLUMNS = (
    "series_index",
    "context_end",
    "target_position",
    "target_seq",
    "history_length",
)

FLOAT_COLUMNS = (
    "true_qty",
    "target_dt",
    "last_qty",
    "history_mean_qty",
    "history_mean_log_qty",
    "recent3_minus_history_mean_log_qty",
    "last_minus_history_mean_log_qty",
    *MODEL_COLUMNS.values(),
)


@dataclass(frozen=True)
class PreparedData:
    frame: pl.DataFrame
    series_id: np.ndarray
    folds: np.ndarray
    target: np.ndarray
    predictions: Mapping[str, np.ndarray]
    quantity_band: np.ndarray
    history_band: np.ndarray
    recent_sign: np.ndarray


@dataclass(frozen=True)
class SeriesAggregates:
    series_id: np.ndarray
    folds: np.ndarray
    row_count: np.ndarray
    candidate_sse: np.ndarray
    candidate_ae: np.ndarray
    reference_sse: Mapping[str, np.ndarray]
    reference_ae: Mapping[str, np.ndarray]
    delta_sse: Mapping[str, np.ndarray]
    delta_ae: Mapping[str, np.ndarray]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-parquet", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument(
        "--detail-dir",
        type=Path,
        help=(
            "Optional search_artifacts directory for the high-cardinality "
            "per-series parquet."
        ),
    )
    return parser.parse_args(argv)


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


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_json_bytes(payload))
    os.replace(temporary, path)


def _write_csv_atomic(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_csv(temporary)
    os.replace(temporary, path)


def _write_parquet_atomic(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    os.replace(temporary, path)


def _require_finite_1d(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _float_column(frame: pl.DataFrame, name: str) -> np.ndarray:
    try:
        values = frame[name].cast(pl.Float64, strict=True).to_numpy()
    except Exception as exc:  # Polars error classes differ across releases.
        raise ValueError(f"{name} must be numeric") from exc
    return _require_finite_1d(values, name)


def _integer_column(frame: pl.DataFrame, name: str) -> np.ndarray:
    values = _float_column(frame, name)
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain integers")
    return values.astype(np.int64)


def assign_series_folds(series_ids: Sequence[str]) -> np.ndarray:
    """Assign a whole series to SHA256(salt|'series_id') mod 2."""
    values = np.asarray(series_ids, dtype=object)
    if values.ndim != 1:
        raise ValueError("series_ids must be one-dimensional")
    mapping: dict[str, int] = {}
    result = np.empty(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value:
            raise ValueError("series_id must contain nonempty strings")
        if value not in mapping:
            payload = f"{FOLD_SALT}|{value}".encode("utf-8")
            mapping[value] = hashlib.sha256(payload).digest()[-1] % 2
        result[index] = mapping[value]
    return result


def quantity_bands(values: Sequence[float]) -> np.ndarray:
    target = _require_finite_1d(values, "true_qty")
    if np.any(target < 0.0):
        raise ValueError("true_qty must be nonnegative")
    indices = np.searchsorted(
        np.asarray(QUANTITY_BOUNDARIES, dtype=np.float64), target, side="left"
    )
    return np.asarray(QUANTITY_LABELS, dtype=object)[indices]


def history_bands(values: Sequence[int]) -> np.ndarray:
    history_float = _require_finite_1d(values, "history_length")
    if not np.equal(history_float, np.floor(history_float)).all():
        raise ValueError("history_length must contain integers")
    history = history_float.astype(np.int64)
    if np.any((history < 1) | (history > 63)):
        raise ValueError("history_length must be within the frozen range [1, 63]")
    result = np.empty(len(history), dtype=object)
    result[history == 1] = HISTORY_LABELS[0]
    result[(history >= 2) & (history <= 3)] = HISTORY_LABELS[1]
    result[(history >= 4) & (history <= 7)] = HISTORY_LABELS[2]
    result[(history >= 8) & (history <= 15)] = HISTORY_LABELS[3]
    result[(history >= 16) & (history <= 31)] = HISTORY_LABELS[4]
    result[(history >= 32) & (history <= 63)] = HISTORY_LABELS[5]
    return result


def recent_signs(values: Sequence[float]) -> np.ndarray:
    recent = _require_finite_1d(
        values, "recent3_minus_history_mean_log_qty"
    )
    result = np.full(len(recent), "=0", dtype=object)
    result[recent < 0.0] = "<0"
    result[recent > 0.0] = ">0"
    return result


def prepare_frame(frame: pl.DataFrame, *, split: str) -> PreparedData:
    if split not in {"train", "validation"}:
        raise ValueError("only train or validation analysis is allowed")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"paired parquet is missing required columns: {missing}")
    if len(frame) == 0:
        raise ValueError("paired parquet is empty")
    if frame["series_id"].dtype != pl.String:
        raise ValueError("series_id must have string dtype")
    if frame["series_id"].null_count() != 0:
        raise ValueError("series_id must not contain nulls")
    series_values = frame["series_id"].to_numpy()
    if any(not isinstance(value, str) or not value for value in series_values):
        raise ValueError("series_id must contain nonempty strings")

    integer_values = {name: _integer_column(frame, name) for name in INTEGER_COLUMNS}
    numeric_values = {name: _float_column(frame, name) for name in FLOAT_COLUMNS}
    if np.any(integer_values["target_position"] < 0):
        raise ValueError("target_position must be nonnegative")
    if np.any(integer_values["history_length"] < 1):
        raise ValueError("history_length must be positive")
    if np.any(numeric_values["true_qty"] < 0.0):
        raise ValueError("true_qty must be nonnegative")
    if np.any(numeric_values["target_dt"] <= 0.0):
        raise ValueError("target_dt must be positive")

    if "split" in frame.columns:
        observed_splits = set(frame["split"].cast(pl.String).unique().to_list())
        if observed_splits != {split}:
            raise ValueError(
                f"parquet split does not match --split={split}: {observed_splits}"
            )

    canonical_count = frame.select(
        pl.struct(["series_id", "target_position"]).n_unique()
    ).item()
    if int(canonical_count) != len(frame):
        raise ValueError("canonical (series_id, target_position) pairs are not unique")

    # Canonical sorting makes all downstream floating-point reductions stable
    # under a harmless input-row reorder.
    ordered = frame.sort(["series_id", "target_position"])
    series = ordered["series_id"].to_numpy()
    target = _float_column(ordered, "true_qty")
    predictions = {
        role: _float_column(ordered, column)
        for role, column in MODEL_COLUMNS.items()
    }
    history = _integer_column(ordered, "history_length")
    recent = _float_column(ordered, "recent3_minus_history_mean_log_qty")
    folds = assign_series_folds(series)
    if len(np.unique(series)) >= 2 and set(np.unique(folds).tolist()) != {0, 1}:
        raise ValueError("whole-series hash folds must both be populated")
    return PreparedData(
        frame=ordered,
        series_id=series,
        folds=folds,
        target=target,
        predictions=predictions,
        quantity_band=quantity_bands(target),
        history_band=history_bands(history),
        recent_sign=recent_signs(recent),
    )


def quantity_metrics(
    target: Sequence[float], prediction: Sequence[float]
) -> dict[str, float | int]:
    target_array = _require_finite_1d(target, "target")
    prediction_array = _require_finite_1d(prediction, "prediction")
    if prediction_array.shape != target_array.shape:
        raise ValueError("target and prediction shapes do not match")
    if len(target_array) == 0:
        raise ValueError("quantity metrics require at least one row")
    error = prediction_array - target_array
    bias = float(error.mean(dtype=np.float64))
    mse = float(np.square(error).mean(dtype=np.float64))
    centered_mse = float(np.square(error - bias).mean(dtype=np.float64))
    residual = mse - centered_mse - bias * bias
    _assert_close(residual, 0.0, "MSE bias/variance decomposition")
    return {
        "count": int(len(target_array)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.abs(error).mean(dtype=np.float64)),
        "bias": bias,
        "bias_squared": bias * bias,
        "centered_mse": centered_mse,
        "mean_prediction": float(prediction_array.mean(dtype=np.float64)),
        "mse_decomposition_residual": residual,
    }


def _assert_close(observed: float, expected: float, label: str) -> None:
    scale = max(1.0, abs(observed), abs(expected))
    tolerance = 1e-10 * scale
    if not math.isfinite(observed) or not math.isfinite(expected):
        raise FloatingPointError(f"{label} is not finite")
    if abs(observed - expected) > tolerance:
        raise AssertionError(
            f"{label} failed: observed={observed}, expected={expected}, "
            f"tolerance={tolerance}"
        )


def comparison_metrics(
    target: Sequence[float],
    candidate: Sequence[float],
    reference: Sequence[float],
) -> dict[str, Any]:
    target_array = _require_finite_1d(target, "target")
    candidate_array = _require_finite_1d(candidate, "candidate")
    reference_array = _require_finite_1d(reference, "reference")
    if candidate_array.shape != target_array.shape or reference_array.shape != target_array.shape:
        raise ValueError("paired target and prediction arrays must have identical shape")
    candidate_metrics = quantity_metrics(target_array, candidate_array)
    reference_metrics = quantity_metrics(target_array, reference_array)
    delta_mse = float(candidate_metrics["mse"] - reference_metrics["mse"])
    delta_bias_squared = float(
        candidate_metrics["bias_squared"] - reference_metrics["bias_squared"]
    )
    delta_centered_mse = float(
        candidate_metrics["centered_mse"] - reference_metrics["centered_mse"]
    )
    decomposition_residual = delta_mse - delta_bias_squared - delta_centered_mse
    _assert_close(
        delta_mse,
        delta_bias_squared + delta_centered_mse,
        "delta MSE bias-squared/centered-MSE decomposition",
    )
    delta_sse = np.square(candidate_array - target_array) - np.square(
        reference_array - target_array
    )
    delta_ae = np.abs(candidate_array - target_array) - np.abs(
        reference_array - target_array
    )
    mean_prediction_shift = float(
        (candidate_array - reference_array).mean(dtype=np.float64)
    )
    _assert_close(
        mean_prediction_shift,
        float(candidate_metrics["bias"] - reference_metrics["bias"]),
        "mean prediction shift / bias change identity",
    )
    return {
        "count": int(len(target_array)),
        "candidate": candidate_metrics,
        "reference": reference_metrics,
        "delta_mse": delta_mse,
        "delta_rmse": float(candidate_metrics["rmse"] - reference_metrics["rmse"]),
        "delta_mae": float(candidate_metrics["mae"] - reference_metrics["mae"]),
        "delta_bias": float(candidate_metrics["bias"] - reference_metrics["bias"]),
        "delta_bias_squared": delta_bias_squared,
        "delta_centered_mse": delta_centered_mse,
        "mean_prediction_shift": mean_prediction_shift,
        "delta_mse_decomposition_residual": decomposition_residual,
        "delta_sse_sum": float(delta_sse.sum(dtype=np.float64)),
        "delta_ae_sum": float(delta_ae.sum(dtype=np.float64)),
    }


def _comparison_name(reference: str) -> str:
    return f"candidate_minus_{reference}"


def _all_comparisons(
    target: np.ndarray, predictions: Mapping[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    return {
        _comparison_name(reference): comparison_metrics(
            target, predictions["candidate"], predictions[reference]
        )
        for reference in REFERENCE_ROLES
    }


def _flat_comparison(
    comparison: Mapping[str, Any], *, reference: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "comparison": _comparison_name(reference),
        "reference_role": reference,
        "is_primary": _comparison_name(reference) == PRIMARY_COMPARISON,
        "count": int(comparison["count"]),
    }
    for prefix in ("candidate", "reference"):
        metrics = comparison[prefix]
        for key in (
            "mse",
            "rmse",
            "mae",
            "bias",
            "bias_squared",
            "centered_mse",
            "mean_prediction",
        ):
            row[f"{prefix}_{key}"] = float(metrics[key])
    for key in (
        "delta_mse",
        "delta_rmse",
        "delta_mae",
        "delta_bias",
        "delta_bias_squared",
        "delta_centered_mse",
        "mean_prediction_shift",
        "delta_mse_decomposition_residual",
        "delta_sse_sum",
        "delta_ae_sum",
    ):
        row[key] = float(comparison[key])
    return row


def _partition_rows(
    data: PreparedData,
    *,
    split: str,
    overall: Mapping[str, Mapping[str, Any]],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    quantity = data.quantity_band.astype(str)
    history = data.history_band.astype(str)
    recent = data.recent_sign.astype(str)
    cross = np.asarray(
        [f"{h}|{q}" for h, q in zip(history, quantity, strict=True)],
        dtype=object,
    )
    families: tuple[tuple[str, np.ndarray, Sequence[str]], ...] = (
        ("quantity_band", quantity, QUANTITY_LABELS),
        ("history_band", history, HISTORY_LABELS),
        (
            "history_x_quantity",
            cross,
            tuple(f"{h}|{q}" for h in HISTORY_LABELS for q in QUANTITY_LABELS),
        ),
        ("recent_sign", recent, RECENT_SIGN_LABELS),
    )
    n = len(data.target)
    rows: list[dict[str, Any]] = []
    replay: dict[str, Any] = {}
    for family, values, labels in families:
        family_rows: list[dict[str, Any]] = []
        for order, label in enumerate(labels):
            mask = values == label
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            subset_predictions = {
                role: prediction[mask] for role, prediction in data.predictions.items()
            }
            comparisons = _all_comparisons(data.target[mask], subset_predictions)
            for reference in REFERENCE_ROLES:
                comparison = comparisons[_comparison_name(reference)]
                row = {
                    "split": split,
                    "partition_family": family,
                    "group_order": order,
                    "group_label": label,
                    "quantity_band": (
                        label
                        if family == "quantity_band"
                        else label.split("|", 1)[1]
                        if family == "history_x_quantity"
                        else None
                    ),
                    "history_band": (
                        label
                        if family == "history_band"
                        else label.split("|", 1)[0]
                        if family == "history_x_quantity"
                        else None
                    ),
                    "recent_sign": label if family == "recent_sign" else None,
                    "history_x_quantity": (
                        label if family == "history_x_quantity" else None
                    ),
                    **_flat_comparison(comparison, reference=reference),
                }
                row["row_fraction"] = count / n
                row["contribution_to_overall_delta_mse"] = (
                    count / n * float(comparison["delta_mse"])
                )
                family_rows.append(row)
                rows.append(row)
        family_replay: dict[str, Any] = {}
        for reference in REFERENCE_ROLES:
            name = _comparison_name(reference)
            selected = [row for row in family_rows if row["comparison"] == name]
            count_sum = sum(int(row["count"]) for row in selected)
            contribution_sum = math.fsum(
                float(row["contribution_to_overall_delta_mse"])
                for row in selected
            )
            expected = float(overall[name]["delta_mse"])
            if count_sum != n:
                raise AssertionError(f"{family}/{name} does not partition all rows")
            _assert_close(
                contribution_sum,
                expected,
                f"{family}/{name} contribution replay",
            )
            family_replay[name] = {
                "count_sum": count_sum,
                "contribution_sum": contribution_sum,
                "overall_delta_mse": expected,
                "replay_residual": contribution_sum - expected,
            }
        replay[family] = family_replay
    # Some nullable label columns first become non-null after more than Polars'
    # default schema-inference prefix.  Inspect every compact row.
    return pl.DataFrame(rows, infer_schema_length=None), replay


def _fold_rows(
    data: PreparedData,
    *,
    split: str,
    overall: Mapping[str, Mapping[str, Any]],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    n = len(data.target)
    rows: list[dict[str, Any]] = []
    for fold in (0, 1):
        mask = data.folds == fold
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        predictions = {
            role: prediction[mask] for role, prediction in data.predictions.items()
        }
        comparisons = _all_comparisons(data.target[mask], predictions)
        for reference in REFERENCE_ROLES:
            comparison = comparisons[_comparison_name(reference)]
            row = {
                "split": split,
                "fold": fold,
                "series_count": int(len(np.unique(data.series_id[mask]))),
                "fold_role": "diagnostic_replication_only",
                "is_oof_generalization": False,
                **_flat_comparison(comparison, reference=reference),
            }
            row["row_fraction"] = count / n
            row["contribution_to_overall_delta_mse"] = (
                count / n * float(comparison["delta_mse"])
            )
            rows.append(row)
    replay: dict[str, Any] = {}
    for reference in REFERENCE_ROLES:
        name = _comparison_name(reference)
        selected = [row for row in rows if row["comparison"] == name]
        count_sum = sum(int(row["count"]) for row in selected)
        contribution_sum = math.fsum(
            float(row["contribution_to_overall_delta_mse"]) for row in selected
        )
        expected = float(overall[name]["delta_mse"])
        if count_sum != n:
            raise AssertionError(f"folds/{name} does not partition all rows")
        _assert_close(contribution_sum, expected, f"folds/{name} replay")
        replay[name] = {
            "count_sum": count_sum,
            "contribution_sum": contribution_sum,
            "overall_delta_mse": expected,
            "replay_residual": contribution_sum - expected,
        }
    return pl.DataFrame(rows), replay


def paired_sign_case_contributions(
    data: PreparedData, overall: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    n = len(data.target)
    candidate_error = data.predictions["candidate"] - data.target
    candidate_under = candidate_error < 0.0
    output: dict[str, list[dict[str, Any]]] = {}
    replay: dict[str, Any] = {}
    for reference in REFERENCE_ROLES:
        name = _comparison_name(reference)
        reference_error = data.predictions[reference] - data.target
        reference_under = reference_error < 0.0
        candidate_sse = np.square(candidate_error)
        reference_sse = np.square(reference_error)
        delta_sse = candidate_sse - reference_sse
        rows: list[dict[str, Any]] = []
        for candidate_flag in (True, False):
            for reference_flag in (True, False):
                mask = (candidate_under == candidate_flag) & (
                    reference_under == reference_flag
                )
                count = int(np.count_nonzero(mask))
                delta_sum = float(delta_sse[mask].sum(dtype=np.float64))
                rows.append(
                    {
                        "candidate_under": candidate_flag,
                        "reference_under": reference_flag,
                        "candidate_sign_case": "under" if candidate_flag else "over_or_equal",
                        "reference_sign_case": "under" if reference_flag else "over_or_equal",
                        "count": count,
                        "row_fraction": count / n,
                        "candidate_sse_sum": float(
                            candidate_sse[mask].sum(dtype=np.float64)
                        ),
                        "reference_sse_sum": float(
                            reference_sse[mask].sum(dtype=np.float64)
                        ),
                        "delta_sse_sum": delta_sum,
                        "contribution_to_overall_delta_mse": delta_sum / n,
                    }
                )
        if sum(row["count"] for row in rows) != n:
            raise AssertionError(f"{name} sign cases are not exhaustive")
        contribution_sum = math.fsum(
            row["contribution_to_overall_delta_mse"] for row in rows
        )
        expected = float(overall[name]["delta_mse"])
        _assert_close(contribution_sum, expected, f"{name} sign-case replay")
        output[name] = rows
        replay[name] = {
            "count_sum": n,
            "contribution_sum": contribution_sum,
            "overall_delta_mse": expected,
            "replay_residual": contribution_sum - expected,
        }
    return output, replay


def aggregate_by_series(data: PreparedData) -> SeriesAggregates:
    series, inverse = np.unique(data.series_id, return_inverse=True)
    number_of_series = len(series)
    row_count = np.bincount(inverse, minlength=number_of_series).astype(
        np.float64, copy=False
    )
    first_row = np.full(number_of_series, -1, dtype=np.int64)
    for row, series_index in enumerate(inverse):
        if first_row[series_index] < 0:
            first_row[series_index] = row
    folds = data.folds[first_row]
    candidate_error = data.predictions["candidate"] - data.target
    candidate_sse = np.bincount(
        inverse, weights=np.square(candidate_error), minlength=number_of_series
    ).astype(np.float64, copy=False)
    candidate_ae = np.bincount(
        inverse, weights=np.abs(candidate_error), minlength=number_of_series
    ).astype(np.float64, copy=False)
    reference_sse: dict[str, np.ndarray] = {}
    reference_ae: dict[str, np.ndarray] = {}
    delta_sse: dict[str, np.ndarray] = {}
    delta_ae: dict[str, np.ndarray] = {}
    for reference in REFERENCE_ROLES:
        error = data.predictions[reference] - data.target
        reference_sse[reference] = np.bincount(
            inverse, weights=np.square(error), minlength=number_of_series
        ).astype(np.float64, copy=False)
        reference_ae[reference] = np.bincount(
            inverse, weights=np.abs(error), minlength=number_of_series
        ).astype(np.float64, copy=False)
        delta_sse[reference] = candidate_sse - reference_sse[reference]
        delta_ae[reference] = candidate_ae - reference_ae[reference]
    return SeriesAggregates(
        series_id=series,
        folds=folds,
        row_count=row_count,
        candidate_sse=candidate_sse,
        candidate_ae=candidate_ae,
        reference_sse=reference_sse,
        reference_ae=reference_ae,
        delta_sse=delta_sse,
        delta_ae=delta_ae,
    )


def series_concentration(
    aggregates: SeriesAggregates,
) -> dict[str, dict[str, Any]]:
    number_of_series = len(aggregates.series_id)
    output: dict[str, dict[str, Any]] = {}
    for reference in REFERENCE_ROLES:
        name = _comparison_name(reference)
        delta = aggregates.delta_sse[reference]
        positive = np.maximum(delta, 0.0)
        negative_signed = np.minimum(delta, 0.0)
        positive_sum = float(positive.sum(dtype=np.float64))
        negative_sum = float(negative_signed.sum(dtype=np.float64))
        net_sum = float(delta.sum(dtype=np.float64))
        # np.lexsort uses the last key as primary: descending delta, then ID.
        order = np.lexsort((aggregates.series_id, -delta))
        fractions: list[dict[str, Any]] = []
        for fraction in CONCENTRATION_FRACTIONS:
            selected_count = max(1, int(math.ceil(number_of_series * fraction)))
            chosen = order[:selected_count]
            chosen_delta = delta[chosen]
            chosen_positive = float(
                np.maximum(chosen_delta, 0.0).sum(dtype=np.float64)
            )
            chosen_negative = float(
                np.minimum(chosen_delta, 0.0).sum(dtype=np.float64)
            )
            fractions.append(
                {
                    "fraction_of_unique_series": fraction,
                    "selected_series_count": selected_count,
                    "selected_series_fraction": selected_count / number_of_series,
                    "ranking": "descending_series_delta_sse",
                    "selected_positive_delta_sse_sum": chosen_positive,
                    "share_of_all_positive_delta_sse": (
                        chosen_positive / positive_sum if positive_sum > 0.0 else None
                    ),
                    "selected_net_delta_sse_sum": float(
                        chosen_delta.sum(dtype=np.float64)
                    ),
                    "selected_negative_delta_sse_sum": chosen_negative,
                }
            )
        output[name] = {
            "series_count": number_of_series,
            "positive_delta_sse_sum": positive_sum,
            "negative_delta_sse_sum": negative_sum,
            "negative_help_magnitude_sum": -negative_sum,
            "net_delta_sse_sum": net_sum,
            "candidate_better_series_count": int(np.count_nonzero(delta < 0.0)),
            "candidate_worse_series_count": int(np.count_nonzero(delta > 0.0)),
            "candidate_tied_series_count": int(np.count_nonzero(delta == 0.0)),
            "fraction_series_candidate_better": float(np.mean(delta < 0.0)),
            "concentration": fractions,
        }
    return output


def series_detail_frame(aggregates: SeriesAggregates, *, split: str) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for reference in REFERENCE_ROLES:
        frames.append(
            pl.DataFrame(
                {
                    "split": [split] * len(aggregates.series_id),
                    "artifact_class": ["search_artifact"] * len(aggregates.series_id),
                    "series_id": aggregates.series_id,
                    "fold": aggregates.folds,
                    "row_count": aggregates.row_count.astype(np.int64),
                    "comparison": [_comparison_name(reference)]
                    * len(aggregates.series_id),
                    "candidate_sse_sum": aggregates.candidate_sse,
                    "reference_sse_sum": aggregates.reference_sse[reference],
                    "delta_sse_sum": aggregates.delta_sse[reference],
                    "candidate_ae_sum": aggregates.candidate_ae,
                    "reference_ae_sum": aggregates.reference_ae[reference],
                    "delta_ae_sum": aggregates.delta_ae[reference],
                }
            )
        )
    return pl.concat(frames).sort(
        ["comparison", "delta_sse_sum", "series_id"],
        descending=[False, True, False],
    )


def paired_series_bootstrap(
    aggregates: SeriesAggregates,
    overall: Mapping[str, Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    draw_chunk_size: int = 8192,
) -> dict[str, Any]:
    if replicates < 2:
        raise ValueError("bootstrap replicates must be at least two")
    number_of_series = len(aggregates.series_id)
    if number_of_series < 2:
        return {
            "available": False,
            "reason": "at_least_two_series_required",
            "interpretation": "case_variation_not_seed_uncertainty",
        }
    references = list(REFERENCE_ROLES)
    delta_sse = np.vstack([aggregates.delta_sse[role] for role in references])
    delta_ae = np.vstack([aggregates.delta_ae[role] for role in references])
    mse_samples = np.empty((len(references), replicates), dtype=np.float64)
    mae_samples = np.empty((len(references), replicates), dtype=np.float64)
    chunk_size = min(draw_chunk_size, number_of_series)
    for replicate in range(replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        sampled_rows = 0.0
        sampled_sse = np.zeros(len(references), dtype=np.float64)
        sampled_ae = np.zeros(len(references), dtype=np.float64)
        remaining = number_of_series
        while remaining:
            draw_size = min(chunk_size, remaining)
            sampled = rng.integers(
                0, number_of_series, size=draw_size, dtype=np.int64
            )
            sampled_rows += float(
                aggregates.row_count[sampled].sum(dtype=np.float64)
            )
            sampled_sse += delta_sse[:, sampled].sum(axis=1, dtype=np.float64)
            sampled_ae += delta_ae[:, sampled].sum(axis=1, dtype=np.float64)
            remaining -= draw_size
        mse_samples[:, replicate] = sampled_sse / sampled_rows
        mae_samples[:, replicate] = sampled_ae / sampled_rows

    comparisons: dict[str, Any] = {}
    for index, reference in enumerate(references):
        name = _comparison_name(reference)
        mse_q = np.quantile(mse_samples[index], (0.025, 0.5, 0.975), method="linear")
        mae_q = np.quantile(mae_samples[index], (0.025, 0.5, 0.975), method="linear")
        comparisons[name] = {
            "delta_mse": {
                "point_estimate": float(overall[name]["delta_mse"]),
                "bootstrap_mean": float(mse_samples[index].mean(dtype=np.float64)),
                "percentile_2_5": float(mse_q[0]),
                "percentile_50": float(mse_q[1]),
                "percentile_97_5": float(mse_q[2]),
                "fraction_replicates_candidate_better": float(
                    np.mean(mse_samples[index] < 0.0)
                ),
            },
            "delta_mae": {
                "point_estimate": float(overall[name]["delta_mae"]),
                "bootstrap_mean": float(mae_samples[index].mean(dtype=np.float64)),
                "percentile_2_5": float(mae_q[0]),
                "percentile_50": float(mae_q[1]),
                "percentile_97_5": float(mae_q[2]),
                "fraction_replicates_candidate_better": float(
                    np.mean(mae_samples[index] < 0.0)
                ),
            },
        }
    return {
        "available": True,
        "protocol": "whole_series_clustered_paired_percentile_bootstrap_v1",
        "resampling_unit": "observed_series_with_replacement_all_rows_retained",
        "interpretation": "case_variation_not_seed_uncertainty",
        "not_oof_generalization": True,
        "seed": seed,
        "replicates": replicates,
        "series_count": number_of_series,
        "comparisons": comparisons,
    }


def load_and_validate_contract(path: Path | None, *, split: str) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contract must be a JSON object")
    if payload.get("contract_id") != CONTRACT_ID:
        raise ValueError("wrong diagnostic contract identity")
    if payload.get("status") != CONTRACT_STATUS:
        raise ValueError("contract was not frozen before row-level inference")
    scope = payload.get("scope", {})
    if scope is not None and not isinstance(scope, Mapping):
        raise ValueError("contract scope must be an object")
    scope = dict(scope or {})
    declared_split = scope.get("target_split", scope.get("split", payload.get("split")))
    if declared_split is not None and declared_split != split:
        raise ValueError("contract split does not match requested analysis split")
    declared_splits = scope.get("splits")
    if declared_splits is not None:
        if not isinstance(declared_splits, list) or split not in declared_splits:
            raise ValueError("requested analysis split is outside contract scope")
        if any(value not in {"train", "validation"} for value in declared_splits):
            raise ValueError("contract scope includes a forbidden split")
    if scope.get("primary_contrast") != PRIMARY_COMPARISON:
        raise ValueError("contract primary contrast drift")
    forbidden_true = (
        "held_out_test",
        "held_out_test_evaluated",
        "test_split",
        "training",
        "training_performed",
        "parameter_updates",
        "calibration_fit",
        "checkpoint_selection",
    )
    for key in forbidden_true:
        if scope.get(key) is True or payload.get(key) is True:
            raise ValueError(f"contract enables forbidden operation: {key}")
    if set(payload.get("models", {})) != set(MODEL_COLUMNS):
        raise ValueError("contract model roles drift")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("contract data definition is missing")
    if [float(value) for value in data.get("train_quantity_boundaries", [])] != list(
        QUANTITY_BOUNDARIES
    ):
        raise ValueError("contract quantity boundaries drift")
    analysis = payload.get("analysis")
    if analysis is not None:
        if not isinstance(analysis, Mapping):
            raise ValueError("contract analysis must be an object")
        expected_if_present = {
            "fold_salt": FOLD_SALT,
            "quantity_boundaries": list(QUANTITY_BOUNDARIES),
            "history_bands": list(HISTORY_LABELS),
            "history_bins": list(HISTORY_LABELS),
            "change_bins": ["negative", "zero", "positive"],
            "series_concentration_fractions": list(CONCENTRATION_FRACTIONS),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "primary_comparison": PRIMARY_COMPARISON,
        }
        for key, expected in expected_if_present.items():
            if key in analysis and analysis[key] != expected:
                raise ValueError(f"contract analysis drift: {key}")
    return payload


def analyze_prepared(
    data: PreparedData,
    *,
    split: str,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    model_metrics = {
        role: quantity_metrics(data.target, prediction)
        for role, prediction in data.predictions.items()
    }
    overall = _all_comparisons(data.target, data.predictions)
    strata, strata_replay = _partition_rows(data, split=split, overall=overall)
    folds, fold_replay = _fold_rows(data, split=split, overall=overall)
    sign_cases, sign_replay = paired_sign_case_contributions(data, overall)
    series = aggregate_by_series(data)
    concentration = series_concentration(series)
    bootstrap = paired_series_bootstrap(
        series, overall, replicates=bootstrap_replicates
    )
    detail = series_detail_frame(series, split=split)
    summary = {
        "schema": SCHEMA,
        "status": "success",
        "split": split,
        "scope": {
            "training_performed": False,
            "inference_performed": False,
            "parameter_updates": False,
            "checkpoint_selection": False,
            "calibration_fit": False,
            "held_out_test_evaluated": False,
            "paired_predictions_only": True,
        },
        "row_count": len(data.target),
        "series_count": len(series.series_id),
        "canonical_identity": ["series_id", "target_position"],
        "canonical_identity_unique": True,
        "float_aggregation_dtype": "float64",
        "fold_definition": {
            "formula": "SHA256(fold_salt + '|' + series_id) mod 2",
            "fold_salt": FOLD_SALT,
            "whole_series": True,
            "fold_counts": {
                str(fold): int(np.count_nonzero(data.folds == fold))
                for fold in (0, 1)
            },
            "interpretation": "diagnostic_replication_not_oof_generalization",
            "train_fold_warning": (
                "train folds diagnose replication across cases; they are not "
                "out-of-fold generalization estimates"
            ),
        },
        "primary_comparison": PRIMARY_COMPARISON,
        "delta_definition": "candidate_minus_reference; negative favors candidate",
        "overall": {
            "models": model_metrics,
            "comparisons": overall,
        },
        "partition_contract": {
            "families_are_separate_complete_partitions": True,
            "do_not_sum_contributions_across_partition_families": True,
            "quantity_bands": list(QUANTITY_LABELS),
            "history_bands": list(HISTORY_LABELS),
            "recent_sign_bands": list(RECENT_SIGN_LABELS),
            "replays": strata_replay,
        },
        "fold_replays": fold_replay,
        "paired_under_over_sse_cases": sign_cases,
        "paired_under_over_replays": sign_replay,
        "series_concentration": concentration,
        "paired_series_bootstrap": bootstrap,
        "limitations": [
            "descriptive paired-case attribution only",
            "no fitting or calibration",
            "no model or checkpoint selection",
            "no held-out test evaluation",
            "bootstrap reflects observed-series case variation, not training-seed uncertainty",
        ],
    }
    return summary, strata, folds, detail


def run_analysis(
    *,
    input_parquet: Path,
    split: str,
    output_dir: Path,
    contract_path: Path | None = None,
    detail_dir: Path | None = None,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    input_parquet = input_parquet.resolve()
    output_dir = output_dir.resolve()
    if not input_parquet.is_file():
        raise FileNotFoundError(input_parquet)
    if detail_dir is not None:
        detail_dir = detail_dir.resolve()
        if "search_artifacts" not in detail_dir.parts:
            raise ValueError("--detail-dir must be inside a search_artifacts path")
    contract = load_and_validate_contract(contract_path, split=split)
    frame = pl.read_parquet(input_parquet)
    data = prepare_frame(frame, split=split)
    if contract is not None:
        population = contract.get("populations", {}).get(split)
        if not isinstance(population, Mapping):
            raise ValueError(f"contract population is missing for split={split}")
        if int(population.get("count", -1)) != len(data.target):
            raise ValueError(
                f"contract population count mismatch for split={split}: "
                f"expected {population.get('count')}, observed {len(data.target)}"
            )
    summary, strata, folds, detail = analyze_prepared(
        data, split=split, bootstrap_replicates=bootstrap_replicates
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    strata_path = output_dir / "strata.csv"
    folds_path = output_dir / "folds.csv"
    _write_csv_atomic(strata_path, strata)
    _write_csv_atomic(folds_path, folds)
    artifacts: dict[str, Any] = {
        "strata_csv": {
            "path": str(strata_path),
            "sha256": sha256_file(strata_path),
            "row_count": len(strata),
        },
        "folds_csv": {
            "path": str(folds_path),
            "sha256": sha256_file(folds_path),
            "row_count": len(folds),
        },
    }
    if detail_dir is not None:
        detail_path = detail_dir / "series_error_contributions.parquet"
        _write_parquet_atomic(detail_path, detail)
        artifacts["series_detail_parquet"] = {
            "path": str(detail_path),
            "sha256": sha256_file(detail_path),
            "row_count": len(detail),
            "artifact_class": "search_artifact",
            "eligible_as_compact_result_table": False,
        }
    summary["input"] = {
        "path": str(input_parquet),
        "sha256": sha256_file(input_parquet),
        "required_columns": list(REQUIRED_COLUMNS),
    }
    summary["contract"] = (
        {
            "path": str(contract_path.resolve()),
            "sha256": sha256_file(contract_path.resolve()),
            "contract_id": contract.get("contract_id", contract.get("schema")),
        }
        if contract is not None and contract_path is not None
        else None
    )
    summary["artifacts"] = artifacts
    summary_path = output_dir / "summary.json"
    _write_json_atomic(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_analysis(
        input_parquet=args.input_parquet,
        split=args.split,
        output_dir=args.output_dir,
        contract_path=args.contract,
        detail_dir=args.detail_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
