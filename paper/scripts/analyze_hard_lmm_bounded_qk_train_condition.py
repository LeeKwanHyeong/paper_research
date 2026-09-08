#!/usr/bin/env python3
"""Pure train-only necessary-condition analysis for BOUNDED-QK.

The analysis compares row-aligned frozen B and BOUNDED-QK quantity
predictions.  Whole series are assigned to two deterministic SHA-256 folds.
The BOUNDED-QK family may continue only when every prospective directional
gate is satisfied independently in both folds.

Target-quantity strata and resampling are intentionally absent: target strata
are descriptive after-the-fact information, and the contract defines the two
fixed series-disjoint folds as the replication unit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "hard_lmm_bounded_qk_train_condition_analysis_v1"
CONTRACT_ID = "hard_lmm_bounded_qk_train_condition_v1"
FOLD_SALT = "hard_lmm_bounded_qk_train_condition_v1:20260908"
FOLD_COUNT = 2
REQUIRED_CONDITION_IDS = (
    "mean_prediction_shift_negative",
    "mean_prediction_shift_material",
    "median_series_mean_prediction_shift_negative",
    "centered_mse_improves",
    "overall_mse_worsens",
    "bias_squared_penalty_increases",
    "bias_penalty_overturns_centered_gain",
    "mae_not_worse",
    "history_2_3_mse_materially_worse",
    "history_2_3_shift_more_negative_than_8_15",
    "history_8_15_mse_improves",
)
HISTORY_STRATA: tuple[tuple[str, int, int], ...] = (
    ("1", 1, 1),
    ("2-3", 2, 3),
    ("4-7", 4, 7),
    ("8-15", 8, 15),
    ("16-31", 16, 31),
    ("32-63", 32, 63),
)


def _series_token(value: Any) -> str:
    """Return a stable, typed token for an allowed series identifier."""
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
    raise ValueError("series identifiers must be strings, booleans, or integer values")


def _fold_for_token(token: str, *, salt: str) -> int:
    digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLD_COUNT


def assign_series_folds(
    series_ids: Sequence[Any], *, salt: str = FOLD_SALT
) -> np.ndarray:
    """Assign every occurrence of a typed series identifier to one fold."""
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a nonempty string")
    values = np.asarray(series_ids, dtype=object)
    if values.ndim != 1:
        raise ValueError("series_id must have shape [N]")
    token_to_fold: dict[str, int] = {}
    folds = np.empty(len(values), dtype=np.int8)
    for index, value in enumerate(values):
        token = _series_token(value)
        fold = token_to_fold.get(token)
        if fold is None:
            fold = _fold_for_token(token, salt=salt)
            token_to_fold[token] = fold
        folds[index] = fold
    return folds


def history_strata(history_length: Sequence[int]) -> np.ndarray:
    """Map valid frozen-model history lengths to the prospective bins."""
    history = np.asarray(history_length)
    if history.ndim != 1:
        raise ValueError("history_length must have shape [N]")
    try:
        numeric = history.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("history_length must be numeric") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("history_length must contain finite integers")
    integer = numeric.astype(np.int64)
    if np.any((integer < 1) | (integer > 63)):
        raise ValueError("history_length must be within [1, 63]")
    result = np.empty(len(integer), dtype=object)
    for label, lower, upper in HISTORY_STRATA:
        result[(integer >= lower) & (integer <= upper)] = label
    return result


def _float_vector(
    values: Sequence[float], *, name: str, nonnegative: bool
) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _numeric_tolerance(*values: float) -> float:
    scale = max(1.0, *(abs(float(value)) for value in values))
    return float(128.0 * np.finfo(np.float64).eps * scale)


def quantity_metrics(
    target: Sequence[float], prediction: Sequence[float]
) -> dict[str, Any]:
    """Calculate raw-scale metrics and audit their bias decomposition."""
    target_array = np.asarray(target, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    if target_array.ndim != 1 or prediction_array.shape != target_array.shape:
        raise ValueError("target and prediction must have the same shape [N]")
    if len(target_array) == 0:
        raise ValueError("metrics require at least one row")
    if not np.isfinite(target_array).all() or not np.isfinite(prediction_array).all():
        raise ValueError("metrics require finite values")

    error = prediction_array - target_array
    bias = float(error.mean(dtype=np.float64))
    mse = float(np.square(error).mean(dtype=np.float64))
    centered_mse = float(np.square(error - bias).mean(dtype=np.float64))
    bias_squared = bias * bias
    residual = mse - centered_mse - bias_squared
    tolerance = _numeric_tolerance(mse, centered_mse, bias_squared)
    passed = abs(residual) <= tolerance
    if not passed:
        raise FloatingPointError("float64 MSE bias decomposition failed")
    return {
        "count": int(len(target_array)),
        "mae": float(np.abs(error).mean(dtype=np.float64)),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "bias": bias,
        "centered_mse": centered_mse,
        "centered_rmse": math.sqrt(centered_mse),
        "bias_squared": bias_squared,
        "numeric_audit": {
            "dtype": "float64",
            "mse_equals_centered_mse_plus_bias_squared": bool(passed),
            "residual": float(residual),
            "tolerance": tolerance,
        },
    }


def comparison_metrics(
    target: Sequence[float],
    prediction_bounded_qk: Sequence[float],
    prediction_b: Sequence[float],
) -> dict[str, Any]:
    """Calculate paired BOUNDED-QK minus B metrics in float64."""
    target_array = np.asarray(target, dtype=np.float64)
    candidate = np.asarray(prediction_bounded_qk, dtype=np.float64)
    reference = np.asarray(prediction_b, dtype=np.float64)
    if candidate.shape != target_array.shape or reference.shape != target_array.shape:
        raise ValueError("paired target and prediction arrays must have identical shape")
    candidate_metrics = quantity_metrics(target_array, candidate)
    reference_metrics = quantity_metrics(target_array, reference)

    prediction_shift = candidate - reference
    reference_error = reference - target_array
    candidate_error = candidate - target_array
    delta_mse = candidate_metrics["mse"] - reference_metrics["mse"]
    delta_centered_mse = (
        candidate_metrics["centered_mse"] - reference_metrics["centered_mse"]
    )
    delta_bias_squared = (
        candidate_metrics["bias_squared"] - reference_metrics["bias_squared"]
    )
    decomposition_residual = delta_mse - delta_centered_mse - delta_bias_squared
    rowwise_delta_mse = float(
        (np.square(candidate_error) - np.square(reference_error)).mean(
            dtype=np.float64
        )
    )
    rowwise_residual = delta_mse - rowwise_delta_mse
    mean_shift = float(prediction_shift.mean(dtype=np.float64))
    bias_shift_residual = (
        candidate_metrics["bias"] - reference_metrics["bias"] - mean_shift
    )

    decomposition_tolerance = _numeric_tolerance(
        delta_mse, delta_centered_mse, delta_bias_squared
    )
    rowwise_tolerance = _numeric_tolerance(delta_mse, rowwise_delta_mse)
    bias_tolerance = _numeric_tolerance(
        candidate_metrics["bias"], reference_metrics["bias"], mean_shift
    )
    checks = {
        "delta_mse_equals_delta_centered_mse_plus_delta_bias_squared": (
            abs(decomposition_residual) <= decomposition_tolerance
        ),
        "delta_mse_equals_mean_rowwise_delta_squared_error": (
            abs(rowwise_residual) <= rowwise_tolerance
        ),
        "delta_bias_equals_mean_prediction_shift": (
            abs(bias_shift_residual) <= bias_tolerance
        ),
    }
    if not all(checks.values()):
        raise FloatingPointError("a float64 paired-error identity failed")
    return {
        "count": int(len(target_array)),
        "mean_prediction_shift": mean_shift,
        "delta_mae": candidate_metrics["mae"] - reference_metrics["mae"],
        "delta_mse": delta_mse,
        "delta_rmse": candidate_metrics["rmse"] - reference_metrics["rmse"],
        "delta_bias": candidate_metrics["bias"] - reference_metrics["bias"],
        "delta_centered_mse": delta_centered_mse,
        "delta_centered_rmse": (
            candidate_metrics["centered_rmse"]
            - reference_metrics["centered_rmse"]
        ),
        "delta_bias_squared": delta_bias_squared,
        "abs_mean_prediction_shift_over_b_rmse": (
            None
            if reference_metrics["rmse"] == 0.0
            else abs(mean_shift) / reference_metrics["rmse"]
        ),
        "delta_mse_over_b_mse": (
            None
            if reference_metrics["mse"] == 0.0
            else delta_mse / reference_metrics["mse"]
        ),
        "numeric_audit": {
            "dtype": "float64",
            **{name: bool(value) for name, value in checks.items()},
            "residuals": {
                "mse_decomposition": float(decomposition_residual),
                "rowwise_delta_mse": float(rowwise_residual),
                "bias_shift": float(bias_shift_residual),
            },
            "tolerances": {
                "mse_decomposition": decomposition_tolerance,
                "rowwise_delta_mse": rowwise_tolerance,
                "bias_shift": bias_tolerance,
            },
        },
    }


def _summary(
    target: np.ndarray, candidate: np.ndarray, reference: np.ndarray
) -> dict[str, Any]:
    return {
        "count": int(len(target)),
        "B": quantity_metrics(target, reference),
        "BOUNDED_QK": quantity_metrics(target, candidate),
        "BOUNDED_QK_minus_B": comparison_metrics(target, candidate, reference),
    }


def _empty_history_summary(label: str, *, fold: int | None) -> dict[str, Any]:
    return {
        "series_fold": fold,
        "history_stratum": label,
        "available": False,
        "count": 0,
        "B": None,
        "BOUNDED_QK": None,
        "BOUNDED_QK_minus_B": None,
    }


def _series_shift_distribution(
    series_tokens: Sequence[str], prediction_shift: np.ndarray
) -> tuple[dict[str, Any], Mapping[str, float]]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for token, shift in zip(series_tokens, prediction_shift, strict=True):
        sums[token] = sums.get(token, 0.0) + float(shift)
        counts[token] = counts.get(token, 0) + 1
    means = {token: sums[token] / counts[token] for token in sums}
    values = np.fromiter(means.values(), dtype=np.float64, count=len(means))
    return (
        {
            "series_count": int(len(values)),
            "mean_of_series_mean_prediction_shifts": float(
                values.mean(dtype=np.float64)
            ),
            "median_of_series_mean_prediction_shifts": float(np.median(values)),
            "minimum_series_mean_prediction_shift": float(values.min()),
            "maximum_series_mean_prediction_shift": float(values.max()),
        },
        means,
    )


def _gate(
    gate_id: str,
    description: str,
    *,
    passed: bool,
    observed: Any,
    criterion: str,
) -> dict[str, Any]:
    return {
        "id": gate_id,
        "description": description,
        "criterion": criterion,
        "observed": observed,
        "passed": bool(passed),
    }


def _fold_gates(
    *,
    fold: int,
    overall: Mapping[str, Any],
    history_by_label: Mapping[str, Mapping[str, Any]],
    median_series_shift: float,
) -> dict[str, Any]:
    comparison = overall["BOUNDED_QK_minus_B"]
    h2 = history_by_label["2-3"]
    h8 = history_by_label["8-15"]
    h2_comparison = h2["BOUNDED_QK_minus_B"]
    h8_comparison = h8["BOUNDED_QK_minus_B"]

    mean_shift = float(comparison["mean_prediction_shift"])
    normalized_shift = comparison["abs_mean_prediction_shift_over_b_rmse"]
    h2_normalized_delta_mse = (
        None if h2_comparison is None else h2_comparison["delta_mse_over_b_mse"]
    )
    h2_shift = (
        None if h2_comparison is None else h2_comparison["mean_prediction_shift"]
    )
    h8_shift = (
        None if h8_comparison is None else h8_comparison["mean_prediction_shift"]
    )
    h8_delta_mse = None if h8_comparison is None else h8_comparison["delta_mse"]

    checks = [
        _gate(
            "mean_prediction_shift_negative",
            "The fold-wide BOUNDED-QK prediction shift is downward.",
            passed=mean_shift < 0.0,
            observed=mean_shift,
            criterion="mean_prediction_shift < 0",
        ),
        _gate(
            "mean_prediction_shift_material",
            "The fold-wide prediction shift is at least 1% of B RMSE.",
            passed=normalized_shift is not None and normalized_shift >= 0.01,
            observed=normalized_shift,
            criterion="abs(mean_prediction_shift) / B_RMSE >= 0.01",
        ),
        _gate(
            "median_series_mean_prediction_shift_negative",
            "The median per-series mean BOUNDED-QK prediction shift is downward.",
            passed=median_series_shift < 0.0,
            observed=median_series_shift,
            criterion="median(per-series mean prediction shift) < 0",
        ),
        _gate(
            "centered_mse_improves",
            "BOUNDED-QK lowers fold-wide centered MSE.",
            passed=float(comparison["delta_centered_mse"]) < 0.0,
            observed=float(comparison["delta_centered_mse"]),
            criterion="delta centered MSE < 0",
        ),
        _gate(
            "overall_mse_worsens",
            "BOUNDED-QK raises fold-wide raw MSE despite its centered-error gain.",
            passed=float(comparison["delta_mse"]) > 0.0,
            observed=float(comparison["delta_mse"]),
            criterion="delta MSE > 0",
        ),
        _gate(
            "bias_squared_penalty_increases",
            "The fold-wide squared-bias term is larger for BOUNDED-QK.",
            passed=float(comparison["delta_bias_squared"]) > 0.0,
            observed=float(comparison["delta_bias_squared"]),
            criterion="delta bias squared > 0",
        ),
        _gate(
            "bias_penalty_overturns_centered_gain",
            "The squared-bias penalty is larger than the centered-MSE gain.",
            passed=float(comparison["delta_bias_squared"])
            > -float(comparison["delta_centered_mse"]),
            observed={
                "delta_bias_squared": float(comparison["delta_bias_squared"]),
                "negative_delta_centered_mse": -float(
                    comparison["delta_centered_mse"]
                ),
            },
            criterion="delta bias squared > -delta centered MSE",
        ),
        _gate(
            "mae_not_worse",
            "BOUNDED-QK does not worsen fold-wide MAE.",
            passed=float(comparison["delta_mae"]) <= 0.0,
            observed=float(comparison["delta_mae"]),
            criterion="delta MAE <= 0",
        ),
        _gate(
            "history_2_3_mse_materially_worse",
            "The history-2-3 raw-MSE penalty is at least 1% of B MSE.",
            passed=(
                h2_normalized_delta_mse is not None
                and h2_normalized_delta_mse >= 0.01
            ),
            observed=h2_normalized_delta_mse,
            criterion="H2-3 delta MSE / H2-3 B MSE >= 0.01",
        ),
        _gate(
            "history_2_3_shift_more_negative_than_8_15",
            "The prediction shift is more downward at history 2-3 than at history 8-15.",
            passed=(
                h2_shift is not None and h8_shift is not None and h2_shift < h8_shift
            ),
            observed={
                "history_2_3_mean_prediction_shift": h2_shift,
                "history_8_15_mean_prediction_shift": h8_shift,
            },
            criterion="H2-3 mean prediction shift < H8-15 mean prediction shift",
        ),
        _gate(
            "history_8_15_mse_improves",
            "BOUNDED-QK lowers raw MSE at history 8-15.",
            passed=h8_delta_mse is not None and h8_delta_mse < 0.0,
            observed=h8_delta_mse,
            criterion="H8-15 delta MSE < 0",
        ),
    ]
    return {
        "series_fold": int(fold),
        "passed": bool(all(check["passed"] for check in checks)),
        "checks": checks,
    }


def analyze_train_condition(
    *,
    target: Sequence[float],
    prediction_b: Sequence[float],
    prediction_bounded_qk: Sequence[float],
    history_length: Sequence[int],
    series_id: Sequence[Any],
) -> dict[str, Any]:
    """Evaluate the frozen, prospective BOUNDED-QK train-only gates."""
    target_array = _float_vector(target, name="target", nonnegative=True)
    reference = _float_vector(prediction_b, name="prediction_b", nonnegative=True)
    candidate = _float_vector(
        prediction_bounded_qk, name="prediction_bounded_qk", nonnegative=True
    )
    history = history_strata(history_length)
    series_values = np.asarray(series_id, dtype=object)

    row_count = len(target_array)
    if row_count == 0:
        raise ValueError("train condition analysis requires at least one row")
    lengths = {
        "target": row_count,
        "prediction_b": len(reference),
        "prediction_bounded_qk": len(candidate),
        "history_length": len(history),
        "series_id": len(series_values),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"all input arrays must be row-aligned: {lengths}")
    if series_values.ndim != 1:
        raise ValueError("series_id must have shape [N]")

    # Intern repeated tokens so a large train artifact stores one string object
    # per unique series plus a compact list of references.
    interned: dict[str, str] = {}
    tokens: list[str] = []
    token_to_fold: dict[str, int] = {}
    folds = np.empty(row_count, dtype=np.int8)
    for index, value in enumerate(series_values):
        raw_token = _series_token(value)
        token = interned.setdefault(raw_token, raw_token)
        fold = token_to_fold.get(token)
        if fold is None:
            fold = _fold_for_token(token, salt=FOLD_SALT)
            token_to_fold[token] = fold
        tokens.append(token)
        folds[index] = fold

    present_folds = set(int(value) for value in np.unique(folds))
    if present_folds != {0, 1}:
        raise ValueError(
            "deterministic series-disjoint analysis requires nonempty folds 0 and 1"
        )
    fold_series = {
        fold: {token for token, assigned in token_to_fold.items() if assigned == fold}
        for fold in (0, 1)
    }
    intersection = fold_series[0].intersection(fold_series[1])
    if intersection:
        raise AssertionError("a series identifier was assigned to both folds")

    shift = candidate - reference
    pooled_series_distribution, per_series_shift = _series_shift_distribution(
        tokens, shift
    )
    fold_series_distributions: dict[int, dict[str, Any]] = {}
    for fold in (0, 1):
        values = np.asarray(
            [per_series_shift[token] for token in fold_series[fold]], dtype=np.float64
        )
        fold_series_distributions[fold] = {
            "series_count": int(len(values)),
            "mean_of_series_mean_prediction_shifts": float(
                values.mean(dtype=np.float64)
            ),
            "median_of_series_mean_prediction_shifts": float(np.median(values)),
            "minimum_series_mean_prediction_shift": float(values.min()),
            "maximum_series_mean_prediction_shift": float(values.max()),
        }

    pooled_metrics = {
        **_summary(target_array, candidate, reference),
        "series_shift_distribution": pooled_series_distribution,
    }
    fold_metrics: list[dict[str, Any]] = []
    history_metrics_pooled: list[dict[str, Any]] = []
    history_metrics_folds: list[dict[str, Any]] = []

    for label, _, _ in HISTORY_STRATA:
        mask = history == label
        if not np.any(mask):
            history_metrics_pooled.append(_empty_history_summary(label, fold=None))
            continue
        history_metrics_pooled.append(
            {
                "series_fold": None,
                "history_stratum": label,
                "available": True,
                **_summary(target_array[mask], candidate[mask], reference[mask]),
            }
        )

    fold_gates: list[dict[str, Any]] = []
    for fold in (0, 1):
        fold_mask = folds == fold
        overall = {
            "series_fold": fold,
            **_summary(
                target_array[fold_mask], candidate[fold_mask], reference[fold_mask]
            ),
            "series_shift_distribution": fold_series_distributions[fold],
        }
        fold_metrics.append(overall)

        history_lookup: dict[str, dict[str, Any]] = {}
        for label, _, _ in HISTORY_STRATA:
            mask = fold_mask & (history == label)
            if not np.any(mask):
                row = _empty_history_summary(label, fold=fold)
            else:
                row = {
                    "series_fold": fold,
                    "history_stratum": label,
                    "available": True,
                    **_summary(target_array[mask], candidate[mask], reference[mask]),
                }
            history_metrics_folds.append(row)
            history_lookup[label] = row

        fold_gates.append(
            _fold_gates(
                fold=fold,
                overall=overall,
                history_by_label=history_lookup,
                median_series_shift=float(
                    fold_series_distributions[fold][
                        "median_of_series_mean_prediction_shifts"
                    ]
                ),
            )
        )

    all_folds_pass = all(row["passed"] for row in fold_gates)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "contract_id": CONTRACT_ID,
        "status": "success",
        "decision": (
            "necessary_condition_met"
            if all_folds_pass
            else "stop_bounded_qk_family"
        ),
        "scope": {
            "split": "train",
            "frozen_predictions_only": True,
            "model_training_performed": False,
            "checkpoint_selection_performed": False,
            "calibration_fit_performed": False,
            "held_out_test_evaluated": False,
            "target_quantity_strata_used_for_decision": False,
            "cluster_bootstrap_used_for_decision": False,
        },
        "input_audit": {
            "row_count": row_count,
            "series_count": int(len(token_to_fold)),
            "target_and_predictions_finite_nonnegative": True,
            "history_length_range": [
                int(np.min(np.asarray(history_length, dtype=np.int64))),
                int(np.max(np.asarray(history_length, dtype=np.int64))),
            ],
            "fold_salt": FOLD_SALT,
            "fold_count": FOLD_COUNT,
            "fold_row_counts": {
                str(fold): int(np.count_nonzero(folds == fold)) for fold in (0, 1)
            },
            "fold_series_counts": {
                str(fold): int(len(fold_series[fold])) for fold in (0, 1)
            },
            "fold_series_intersection_count": int(len(intersection)),
            "series_disjoint": len(intersection) == 0,
        },
        "pooled_metrics": pooled_metrics,
        "fold_metrics": fold_metrics,
        "history_metrics": {
            "pooled": history_metrics_pooled,
            "by_fold": history_metrics_folds,
        },
        "gate_checklist": {
            "prospective": True,
            "required_fold_count": FOLD_COUNT,
            "all_folds_pass": bool(all_folds_pass),
            "failure_policy": "any failed check in either fold stops the BOUNDED-QK family",
            "folds": fold_gates,
        },
    }
    # Fail early if a future edit introduces NumPy scalars, NaN, or Infinity.
    json.dumps(result, allow_nan=False)
    return result


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_b_array_sha256(value: np.ndarray, *, label: str) -> str:
    """Replay the manifest hash used by the previously frozen B cache."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"frozen_raw_affine_array_v1\0")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def bounded_cache_array_sha256(value: np.ndarray, *, label: str) -> str:
    """Replay the manifest hash emitted by the train-inference runner."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"hard_lmm_bounded_qk_train_array_v1\0")
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(str(array.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_contract(contract: Mapping[str, Any]) -> None:
    """Fail closed when the prospective analysis contract has drifted."""
    if contract.get("contract_id") != CONTRACT_ID:
        raise ValueError("wrong train-condition contract")
    if contract.get("status") != "frozen_before_bounded_qk_train_inference":
        raise ValueError("train-condition contract was not frozen prospectively")
    scope = contract.get("scope")
    if not isinstance(scope, Mapping):
        raise ValueError("contract scope is missing")
    expected_scope = {
        "dataset": "insta_market_basket",
        "seed": 42,
        "target_split": "train",
        "input_splits_materialized": ["train"],
        "held_out_test": False,
        "validation_targets": False,
        "training": False,
        "checkpoint_selection": False,
        "parameter_updates": False,
        "calibration_fit": False,
        "primary_contrast": "BOUNDED_QK_minus_B",
    }
    for name, expected in expected_scope.items():
        if scope.get(name) != expected:
            raise ValueError(f"contract scope drift: {name}")

    dataset = contract.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("contract dataset is missing")
    for name in (
        "expected_train_targets",
        "expected_train_series",
        "expected_train_target_series",
    ):
        if type(dataset.get(name)) is not int or int(dataset[name]) <= 0:
            raise ValueError(f"invalid contract dataset field: {name}")

    models = contract.get("models")
    if not isinstance(models, Mapping) or set(models) != {"B", "BOUNDED_QK"}:
        raise ValueError("contract model roles drifted")
    analysis = contract.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("contract analysis is missing")
    if analysis.get("fold_count") != FOLD_COUNT:
        raise ValueError("contract fold count drifted")
    if analysis.get("fold_salt") != FOLD_SALT:
        raise ValueError("contract fold salt drifted")
    expected_bins = [label for label, _, _ in HISTORY_STRATA]
    if analysis.get("history_bins") != expected_bins:
        raise ValueError("contract history bins drifted")
    conditions = analysis.get("required_conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("contract required conditions are missing")
    if tuple(conditions.keys()) != REQUIRED_CONDITION_IDS:
        raise ValueError(
            "contract required-condition IDs or order do not match the analyzer"
        )
    if analysis.get("acceptance") != "all required conditions must pass in both fixed folds":
        raise ValueError("contract acceptance rule drifted")
    if analysis.get("failure_action") != "stop_bounded_qk_family":
        raise ValueError("contract failure action drifted")
    if (
        analysis.get("target_quantity_strata_role")
        != "descriptive_only_and_excluded_from_decision"
    ):
        raise ValueError("target-strata decision role drifted")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise ValueError(f"cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def validate_run_audit(
    audit: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    b_cache_sha256: str,
    bounded_cache_sha256: str,
) -> None:
    """Verify inference scope, state identity, and evidence hashes."""
    if audit.get("schema") != "hard_lmm_bounded_qk_train_condition_run_v1":
        raise ValueError("run-audit schema drifted")
    if audit.get("status") != "success":
        raise ValueError("train inference did not complete successfully")
    audit_contract = audit.get("contract")
    if not isinstance(audit_contract, Mapping):
        raise ValueError("run-audit contract evidence is missing")
    if audit_contract.get("id") != CONTRACT_ID:
        raise ValueError("run-audit contract ID drifted")
    if audit_contract.get("sha256") != contract_sha256:
        raise ValueError("run-audit contract hash drifted")
    if audit.get("scope") != contract["scope"]:
        raise ValueError("run-audit scope differs from the frozen contract")

    forbidden_flags = {
        "training_performed": False,
        "checkpoint_selection_performed": False,
        "calibration_fit_performed": False,
        "validation_targets_evaluated": False,
        "held_out_test_evaluated": False,
    }
    for name, expected in forbidden_flags.items():
        if audit.get(name) is not expected:
            raise ValueError(f"forbidden run-audit operation or access: {name}")

    data = audit.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("run-audit data evidence is missing")
    if data.get("materialized_splits") != ["train"]:
        raise ValueError("run audit admitted a non-train split")
    dataset = contract["dataset"]
    if data.get("sha256") != dataset.get("data_sha256"):
        raise ValueError("run-audit dataset hash drifted")
    if data.get("split_manifest_sha256") != dataset.get("split_manifest_sha256"):
        raise ValueError("run-audit split-manifest hash drifted")
    population = data.get("target_population")
    if not isinstance(population, Mapping):
        raise ValueError("run-audit target population is missing")
    population_expectations = {
        "target_count": dataset.get("expected_train_targets"),
        "target_identity_sha256": dataset.get("expected_train_identity_sha256"),
        "target_quantity_sha256": dataset.get("expected_train_quantity_sha256"),
    }
    for name, expected in population_expectations.items():
        if population.get(name) != expected:
            raise ValueError(f"run-audit target population drifted: {name}")

    b_evidence = audit.get("frozen_b_reuse")
    if not isinstance(b_evidence, Mapping):
        raise ValueError("run-audit Frozen-B reuse evidence is missing")
    if b_evidence.get("cache_file_sha256") != b_cache_sha256:
        raise ValueError("run-audit Frozen-B cache hash drifted")
    if b_cache_sha256 != contract["models"]["B"].get("train_cache_file_sha256"):
        raise ValueError("Frozen-B cache does not match the contract")
    if (
        b_evidence.get("result_file_sha256")
        != contract["models"]["B"].get("train_result_file_sha256")
    ):
        raise ValueError("run-audit Frozen-B result hash drifted")
    if b_evidence.get("all_array_hashes_verified") is not True:
        raise ValueError("run-audit did not verify all Frozen-B arrays")

    checkpoint = audit.get("bounded_qk_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("run-audit BOUNDED-QK checkpoint evidence is missing")
    bounded_spec = contract["models"]["BOUNDED_QK"]
    for name in ("checkpoint_file_sha256", "checkpoint_state_sha256"):
        if checkpoint.get(name) != bounded_spec.get(name):
            raise ValueError(f"run-audit BOUNDED-QK checkpoint drifted: {name}")

    state = audit.get("state_identity")
    if not isinstance(state, Mapping):
        raise ValueError("run-audit state identity is missing")
    if (
        state.get("unchanged") is not True
        or state.get("all_gradients_absent") is not True
        or state.get("before") != state.get("after")
        or state.get("after") != bounded_spec.get("checkpoint_state_sha256")
    ):
        raise ValueError("BOUNDED-QK state or gradients changed during inference")

    source = audit.get("frozen_source")
    if not isinstance(source, Mapping):
        raise ValueError("run-audit frozen-source evidence is missing")
    if source.get("revision") != contract["execution"].get("frozen_source_revision"):
        raise ValueError("run-audit frozen-source revision drifted")
    if (
        source.get("git_worktree_clean") is not True
        or source.get("all_file_hashes_verified") is not True
    ):
        raise ValueError("run-audit frozen-source verification failed")

    predictions = audit.get("bounded_qk_predictions")
    if not isinstance(predictions, Mapping):
        raise ValueError("run-audit prediction evidence is missing")
    if predictions.get("sha256") != bounded_cache_sha256:
        raise ValueError("run-audit BOUNDED-QK cache hash drifted")
    if predictions.get("target_count") != dataset.get("expected_train_targets"):
        raise ValueError("run-audit BOUNDED-QK target count drifted")


def load_verified_inputs(
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
    b_cache_path: Path,
    bounded_cache_path: Path,
    run_audit_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load, hash, and row-align both frozen prediction caches."""
    validate_contract(contract)
    if not run_audit_path.is_file():
        raise ValueError("run audit is missing")
    audit = json.loads(run_audit_path.read_text(encoding="utf-8"))
    contract_sha = sha256_file(contract_path)
    b_cache_sha = sha256_file(b_cache_path)
    bounded_cache_sha = sha256_file(bounded_cache_path)
    validate_run_audit(
        audit,
        contract=contract,
        contract_sha256=contract_sha,
        b_cache_sha256=b_cache_sha,
        bounded_cache_sha256=bounded_cache_sha,
    )

    b_cache = _load_npz(b_cache_path)
    bounded_cache = _load_npz(bounded_cache_path)
    b_keys = {
        "prediction",
        "quantity",
        "time_nll",
        "history_length",
        "series_index",
        "target_index",
        "context_end",
    }
    bounded_keys = b_keys | {"series_parts"}
    if set(b_cache) != b_keys:
        raise ValueError("Frozen-B cache schema drifted")
    if set(bounded_cache) != bounded_keys:
        raise ValueError("BOUNDED-QK cache schema drifted")

    expected_count = int(contract["dataset"]["expected_train_targets"])
    for role, cache in (("B", b_cache), ("BOUNDED_QK", bounded_cache)):
        for name in b_keys:
            if cache[name].shape != (expected_count,):
                raise ValueError(f"{role} cache shape drifted: {name}")
    parts = bounded_cache["series_parts"]
    if parts.ndim != 1 or len(parts) != int(contract["dataset"]["expected_train_series"]):
        raise ValueError("BOUNDED-QK series-parts shape drifted")

    for name, expected in contract["models"]["B"][
        "train_cache_array_sha256"
    ].items():
        observed = frozen_b_array_sha256(b_cache[name], label=name)
        if observed != expected:
            raise ValueError(f"Frozen-B array hash drifted: {name}")

    manifest = audit["bounded_qk_predictions"].get("array_manifest")
    if not isinstance(manifest, Mapping) or set(manifest) != bounded_keys:
        raise ValueError("BOUNDED-QK array manifest schema drifted")
    for name, value in bounded_cache.items():
        item = manifest[name]
        if not isinstance(item, Mapping):
            raise ValueError(f"BOUNDED-QK array manifest entry is invalid: {name}")
        observed_sha = bounded_cache_array_sha256(value, label=name)
        if (
            item.get("dtype") != str(value.dtype)
            or item.get("shape") != list(value.shape)
            or item.get("sha256") != observed_sha
        ):
            raise ValueError(f"BOUNDED-QK array manifest drifted: {name}")

    aligned_fields = (
        "quantity",
        "history_length",
        "series_index",
        "target_index",
        "context_end",
    )
    for name in aligned_fields:
        if not np.array_equal(b_cache[name], bounded_cache[name]):
            raise ValueError(f"B and BOUNDED-QK row alignment drifted: {name}")
    if not np.array_equal(
        b_cache["target_index"], np.arange(expected_count, dtype=np.int64)
    ):
        raise ValueError("canonical target order drifted")

    series_index = b_cache["series_index"]
    if not np.issubdtype(series_index.dtype, np.integer):
        raise ValueError("series_index must be integral")
    if np.any(series_index < 0) or np.any(series_index >= len(parts)):
        raise ValueError("series_index is outside series_parts")
    row_series_id = parts[series_index.astype(np.int64, copy=False)]

    finite_names = ("prediction", "quantity", "time_nll")
    for role, cache in (("B", b_cache), ("BOUNDED_QK", bounded_cache)):
        if not all(np.isfinite(cache[name]).all() for name in finite_names):
            raise ValueError(f"{role} cache contains non-finite numeric values")
        if np.any(cache["prediction"] < 0.0) or np.any(cache["quantity"] < 0.0):
            raise ValueError(f"{role} cache contains negative quantity values")

    inputs = {
        "target": b_cache["quantity"],
        "prediction_b": b_cache["prediction"],
        "prediction_bounded_qk": bounded_cache["prediction"],
        "history_length": b_cache["history_length"],
        "series_id": row_series_id,
    }
    evidence = {
        "contract_sha256": contract_sha,
        "b_cache_sha256": b_cache_sha,
        "bounded_qk_cache_sha256": bounded_cache_sha,
        "run_audit_sha256": sha256_file(run_audit_path),
        "all_b_array_hashes_verified": True,
        "all_bounded_qk_array_hashes_verified": True,
        "paired_row_alignment_verified": True,
        "series_id_reconstruction": "bounded.series_parts[B.series_index]",
    }
    return inputs, evidence


def _fold_csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    gate_by_fold = {
        int(row["series_fold"]): row for row in analysis["gate_checklist"]["folds"]
    }
    rows: list[dict[str, Any]] = []
    for item in analysis["fold_metrics"]:
        fold = int(item["series_fold"])
        comparison = item["BOUNDED_QK_minus_B"]
        series = item["series_shift_distribution"]
        gate = gate_by_fold[fold]
        row = {
            "series_fold": fold,
            "count": item["count"],
            "series_count": series["series_count"],
            "b_mae": item["B"]["mae"],
            "bounded_qk_mae": item["BOUNDED_QK"]["mae"],
            "b_rmse": item["B"]["rmse"],
            "bounded_qk_rmse": item["BOUNDED_QK"]["rmse"],
            "mean_prediction_shift": comparison["mean_prediction_shift"],
            "abs_mean_shift_over_b_rmse": comparison[
                "abs_mean_prediction_shift_over_b_rmse"
            ],
            "median_series_mean_prediction_shift": series[
                "median_of_series_mean_prediction_shifts"
            ],
            "delta_mae": comparison["delta_mae"],
            "delta_mse": comparison["delta_mse"],
            "delta_rmse": comparison["delta_rmse"],
            "delta_centered_mse": comparison["delta_centered_mse"],
            "fold_passed": gate["passed"],
        }
        row.update({check["id"]: check["passed"] for check in gate["checks"]})
        rows.append(row)
    return rows


def _history_csv_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in [
        *analysis["history_metrics"]["pooled"],
        *analysis["history_metrics"]["by_fold"],
    ]:
        comparison = item["BOUNDED_QK_minus_B"]
        rows.append(
            {
                "scope": "pooled" if item["series_fold"] is None else "fold",
                "series_fold": item["series_fold"],
                "history_stratum": item["history_stratum"],
                "available": item["available"],
                "count": item["count"],
                "b_mae": None if item["B"] is None else item["B"]["mae"],
                "bounded_qk_mae": (
                    None if item["BOUNDED_QK"] is None else item["BOUNDED_QK"]["mae"]
                ),
                "b_rmse": None if item["B"] is None else item["B"]["rmse"],
                "bounded_qk_rmse": (
                    None if item["BOUNDED_QK"] is None else item["BOUNDED_QK"]["rmse"]
                ),
                "mean_prediction_shift": (
                    None if comparison is None else comparison["mean_prediction_shift"]
                ),
                "delta_mae": None if comparison is None else comparison["delta_mae"],
                "delta_mse": None if comparison is None else comparison["delta_mse"],
                "delta_mse_over_b_mse": (
                    None if comparison is None else comparison["delta_mse_over_b_mse"]
                ),
                "delta_centered_mse": (
                    None if comparison is None else comparison["delta_centered_mse"]
                ),
            }
        )
    return rows


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV evidence table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0].keys())
    if any(set(row) != set(fieldnames) for row in rows):
        raise ValueError("CSV rows do not share one schema")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    inputs, evidence = load_verified_inputs(
        contract=contract,
        contract_path=args.contract,
        b_cache_path=args.b_cache,
        bounded_cache_path=args.bounded_cache,
        run_audit_path=args.run_audit,
    )
    analysis = analyze_train_condition(**inputs)
    expected_ids = tuple(contract["analysis"]["required_conditions"])
    for fold in analysis["gate_checklist"]["folds"]:
        observed_ids = tuple(check["id"] for check in fold["checks"])
        if observed_ids != expected_ids:
            raise ValueError("analysis gate checklist differs from the frozen contract")
    analysis["evidence_audit"] = evidence
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "analysis.json", analysis)
    write_csv_atomic(args.output_dir / "fold_metrics.csv", _fold_csv_rows(analysis))
    write_csv_atomic(
        args.output_dir / "history_metrics.csv", _history_csv_rows(analysis)
    )
    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--b-cache", type=Path, required=True)
    parser.add_argument("--bounded-cache", type=Path, required=True)
    parser.add_argument("--run-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


# A concise alias for runners that treat this module as a generic analyzer.
analyze = analyze_train_condition


if __name__ == "__main__":
    completed = run_cli(parse_args())
    print(
        json.dumps(
            {
                "status": completed["status"],
                "decision": completed["decision"],
                "all_folds_pass": completed["gate_checklist"]["all_folds_pass"],
            },
            sort_keys=True,
        )
    )
