"""Pure helpers for frozen raw-quantity affine calibration.

The calibrator is deliberately small and auditable.  It fits a positive-slope
ordinary least-squares map on frozen raw predictions and applies a nonnegative
projection only after the affine map::

    calibrated = max(0, slope * frozen_prediction + intercept)

The two-fold audit is train-internal.  Every series is assigned to exactly one
deterministic fold and each row is calibrated with coefficients fitted on the
opposite fold.  These helpers never select a checkpoint or inspect held-out
test data.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


PROTOCOL = "frozen_raw_affine_calibration_v1"
FOLD_SALT = "frozen_raw_affine_calibration_v1:20260907"
MINIMUM_SLOPE = 1e-12
IDENTITY_CALIBRATION = {"slope": 1.0, "intercept": 0.0}
QUANTITY_STRATA = ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")
BOOTSTRAP_PROTOCOL = "series_clustered_paired_bootstrap_v1"
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_REPLICATES = 500
DEFAULT_BOOTSTRAP_DRAW_CHUNK_SIZE = 8192


def _numeric_vector(value: Any, name: str, *, n: int | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if n is not None and len(result) != n:
        raise ValueError(f"{name} row count mismatch")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_vector(value: Any, name: str, *, n: int | None = None) -> np.ndarray:
    result = _numeric_vector(value, name, n=n)
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite positive scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive scalar") from exc
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def _integer_parameter(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        qualifier = (
            f"between {minimum} and {maximum}"
            if maximum is not None
            else f"at least {minimum}"
        )
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _calibration_parameters(calibration: Mapping[str, Any]) -> tuple[float, float]:
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration must be a mapping")
    if not {"slope", "intercept"}.issubset(calibration):
        raise ValueError("calibration must contain slope and intercept")
    slope = _positive_float(calibration["slope"], "slope")
    try:
        intercept = float(calibration["intercept"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("intercept must be a finite scalar") from exc
    if not np.isfinite(intercept):
        raise ValueError("intercept must be a finite scalar")
    return slope, intercept


def identity_calibration() -> dict[str, float]:
    """Return a fresh JSON-serializable identity calibration."""
    return dict(IDENTITY_CALIBRATION)


def fit_positive_affine_ols(
    frozen_prediction: Any,
    target: Any,
    *,
    minimum_slope: float = MINIMUM_SLOPE,
) -> dict[str, Any]:
    """Fit raw-scale OLS with an unpenalized intercept and positive slope.

    The unconstrained centered OLS slope is projected onto the fixed lower
    bound.  The intercept is then the exact conditional optimum for that
    slope.  No validation value or iterative stopping rule enters the fit.
    """
    prediction = _nonnegative_vector(frozen_prediction, "frozen_prediction")
    target = _nonnegative_vector(target, "target", n=len(prediction))
    if len(prediction) < 2:
        raise ValueError("affine OLS requires at least two rows")
    minimum_slope = _positive_float(minimum_slope, "minimum_slope")

    prediction_mean = float(prediction.mean(dtype=np.float64))
    target_mean = float(target.mean(dtype=np.float64))
    centered_prediction = prediction - prediction_mean
    centered_target = target - target_mean
    denominator = float(np.dot(centered_prediction, centered_prediction))
    numerator = float(np.dot(centered_prediction, centered_target))
    if denominator == 0.0:
        unconstrained_slope = 0.0
    else:
        unconstrained_slope = numerator / denominator
    slope = max(float(unconstrained_slope), minimum_slope)
    intercept = target_mean - slope * prediction_mean
    if not np.isfinite((slope, intercept)).all():
        raise FloatingPointError("affine OLS produced nonfinite parameters")

    affine_prediction = slope * prediction + intercept
    calibrated_prediction = np.maximum(0.0, affine_prediction)
    residual = affine_prediction - target
    return {
        "slope": slope,
        "intercept": float(intercept),
        "minimum_slope": minimum_slope,
        "unconstrained_slope": float(unconstrained_slope),
        "slope_was_bounded": bool(unconstrained_slope < minimum_slope),
        "train_rows": int(len(prediction)),
        "train_prediction_mean": prediction_mean,
        "train_target_mean": target_mean,
        "train_prediction_centered_sum_squares": denominator,
        "train_affine_sse": float(np.dot(residual, residual)),
        "train_nonnegative_clamp_count": int(np.count_nonzero(affine_prediction < 0.0)),
        "train_calibrated_sse": float(
            np.square(calibrated_prediction - target).sum(dtype=np.float64)
        ),
    }


def apply_affine_calibration(
    frozen_prediction: Any,
    calibration: Mapping[str, Any],
) -> np.ndarray:
    """Apply an affine calibration and project raw quantities onto [0, inf)."""
    prediction = _nonnegative_vector(frozen_prediction, "frozen_prediction")
    slope, intercept = _calibration_parameters(calibration)
    calibrated = np.maximum(0.0, slope * prediction + intercept)
    if not np.isfinite(calibrated).all():
        raise FloatingPointError("affine calibration produced nonfinite predictions")
    return calibrated


def apply_quantity_only_bundle(
    frozen_prediction: Any,
    time_nll: Any,
    calibration: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Apply the quantity postprocessor while copying the time vector unchanged."""
    prediction = _nonnegative_vector(frozen_prediction, "frozen_prediction")
    time = _numeric_vector(time_nll, "time_nll", n=len(prediction))
    return {
        "quantity_prediction": apply_affine_calibration(prediction, calibration),
        "time_nll": time.copy(),
    }


def _series_token(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("numeric series identifiers must be finite integers")
        return f"int:{int(numeric)}"
    if isinstance(value, str):
        if not value:
            raise ValueError("string series identifiers must be nonempty")
        return f"str:{value}"
    raise ValueError("series identifiers must be strings or integer values")


def _series_tokens(series_id: Any, *, n: int | None = None) -> np.ndarray:
    if hasattr(series_id, "detach"):
        series_id = series_id.detach().cpu().numpy()
    values = np.asarray(series_id, dtype=object)
    if values.ndim != 1:
        raise ValueError("series_id must have shape [N]")
    if n is not None and len(values) != n:
        raise ValueError("series_id row count mismatch")
    # A concrete Unicode dtype keeps any saved NPZ cache pickle-free.
    return np.asarray([_series_token(value) for value in values], dtype=np.str_)


def assign_series_folds(series_id: Any, *, salt: str = FOLD_SALT) -> np.ndarray:
    """Assign each series to one stable SHA-256 fold without touching labels."""
    if not isinstance(salt, str) or not salt:
        raise ValueError("salt must be a nonempty string")
    tokens = _series_tokens(series_id)
    folds_by_series: dict[str, int] = {}
    folds = np.empty(len(tokens), dtype=np.int64)
    for index, token in enumerate(tokens):
        fold = folds_by_series.get(token)
        if fold is None:
            digest = hashlib.sha256(f"{salt}:{token}".encode("utf-8")).digest()
            fold = int.from_bytes(digest[:8], "big") % 2
            folds_by_series[token] = fold
        folds[index] = fold
    return folds


def validate_series_disjoint_folds(
    series_id: Any,
    fold: Any,
    *,
    require_both_folds: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate that no series crosses folds and return normalized arrays."""
    tokens = _series_tokens(series_id)
    raw_folds = _numeric_vector(fold, "fold", n=len(tokens))
    if np.any(raw_folds % 1):
        raise ValueError("fold must contain integer values")
    folds = raw_folds.astype(np.int64)
    expected = {0, 1} if require_both_folds else set(folds.tolist())
    if require_both_folds and set(folds.tolist()) != expected:
        raise ValueError("fold must contain exactly folds 0 and 1")
    if not set(folds.tolist()).issubset({0, 1}):
        raise ValueError("fold values must be 0 or 1")
    fold_by_series: dict[str, int] = {}
    for token, value in zip(tokens.tolist(), folds.tolist(), strict=True):
        previous = fold_by_series.setdefault(token, value)
        if previous != value:
            raise ValueError("series_id values must be disjoint across folds")
    return tokens, folds


def compute_quantity_metrics(prediction: Any, target: Any) -> dict[str, Any]:
    """Compute JSON-safe raw-scale metrics for one nonempty population."""
    prediction = _nonnegative_vector(prediction, "prediction")
    target = _nonnegative_vector(target, "target", n=len(prediction))
    if not len(prediction):
        raise ValueError("quantity metrics require at least one row")
    error = prediction - target
    squared = np.square(error)
    return {
        "count": int(len(prediction)),
        "mae": float(np.abs(error).mean(dtype=np.float64)),
        "mse": float(squared.mean(dtype=np.float64)),
        "rmse": float(np.sqrt(squared.mean(dtype=np.float64))),
        "bias": float(error.mean(dtype=np.float64)),
    }


def _empty_metrics() -> dict[str, Any]:
    return {"count": 0, "mae": None, "mse": None, "rmse": None, "bias": None}


def _validated_boundaries(boundaries: Sequence[float], name: str) -> np.ndarray:
    result = _numeric_vector(boundaries, name)
    if np.any(result < 0) or np.any(np.diff(result) < 0):
        raise ValueError(f"{name} must be ordered and nonnegative")
    return result


def compute_stratified_metrics(
    prediction: Any,
    target: Any,
    stratifier: Any,
    *,
    boundaries: Sequence[float],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Report metrics in fixed lower-inclusive strata.

    ``np.searchsorted(..., side='left')`` places a value equal to a boundary in
    the lower stratum, matching the count-aware experiment evaluator.
    """
    prediction = _nonnegative_vector(prediction, "prediction")
    target = _nonnegative_vector(target, "target", n=len(prediction))
    stratifier = _nonnegative_vector(stratifier, "stratifier", n=len(prediction))
    boundaries_array = _validated_boundaries(boundaries, "boundaries")
    labels = tuple(labels)
    if len(labels) != len(boundaries_array) + 1:
        raise ValueError("labels must contain exactly len(boundaries) + 1 entries")
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("labels must be nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("labels must be unique")
    ids = np.searchsorted(boundaries_array, stratifier, side="left")
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        selected = ids == index
        metrics = (
            compute_quantity_metrics(prediction[selected], target[selected])
            if selected.any()
            else _empty_metrics()
        )
        rows.append({"stratum": label, "stratum_order": index, **metrics})
    return rows


def compute_reporting_metrics(
    prediction: Any,
    target: Any,
    *,
    body_threshold: float,
    extreme_tail_threshold: float,
    quantity_boundaries: Sequence[float] | None = None,
    history_length: Any | None = None,
    history_boundaries: Sequence[float] | None = None,
    history_labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute overall, body, extreme-tail, and optional fixed strata metrics."""
    prediction = _nonnegative_vector(prediction, "prediction")
    target = _nonnegative_vector(target, "target", n=len(prediction))
    body_threshold = float(body_threshold)
    extreme_tail_threshold = float(extreme_tail_threshold)
    if not (
        np.isfinite(body_threshold)
        and np.isfinite(extreme_tail_threshold)
        and 0 <= body_threshold <= extreme_tail_threshold
    ):
        raise ValueError("reporting thresholds must be finite, ordered, and nonnegative")
    body = target <= body_threshold
    extreme_tail = target > extreme_tail_threshold

    def selected_metrics(mask: np.ndarray) -> dict[str, Any]:
        if not mask.any():
            return _empty_metrics()
        return compute_quantity_metrics(prediction[mask], target[mask])

    result: dict[str, Any] = {
        "overall": compute_quantity_metrics(prediction, target),
        "body_le_p95": selected_metrics(body),
        "gt_p99": selected_metrics(extreme_tail),
        "thresholds": {
            "body_max_inclusive": body_threshold,
            "extreme_tail_min_exclusive": extreme_tail_threshold,
        },
    }
    if quantity_boundaries is not None:
        result["quantity_strata"] = compute_stratified_metrics(
            prediction,
            target,
            target,
            boundaries=quantity_boundaries,
            labels=QUANTITY_STRATA,
        )
    requested_history = any(
        value is not None
        for value in (history_length, history_boundaries, history_labels)
    )
    if requested_history:
        if history_length is None or history_boundaries is None or history_labels is None:
            raise ValueError(
                "history_length, history_boundaries, and history_labels must be provided together"
            )
        result["history_strata"] = compute_stratified_metrics(
            prediction,
            target,
            history_length,
            boundaries=history_boundaries,
            labels=history_labels,
        )
    return result


def _relative_improvement(reference: float, candidate: float) -> float | None:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else None
    return float((reference - candidate) / reference)


def _comparison(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mae_relative_improvement": _relative_improvement(
            float(reference["mae"]), float(candidate["mae"])
        ),
        "mse_relative_improvement": _relative_improvement(
            float(reference["mse"]), float(candidate["mse"])
        ),
        "rmse_relative_improvement": _relative_improvement(
            float(reference["rmse"]), float(candidate["rmse"])
        ),
        "bias_change": float(candidate["bias"] - reference["bias"]),
    }


def _bootstrap_interval_summary(
    bootstrap_values: np.ndarray,
    *,
    point_estimate: float,
) -> dict[str, Any]:
    lower, upper = np.quantile(
        bootstrap_values,
        (0.025, 0.975),
        method="linear",
    )
    return {
        "point_estimate": float(point_estimate),
        "bootstrap_mean": float(bootstrap_values.mean(dtype=np.float64)),
        "ci_95_percentile": {
            "lower": float(lower),
            "upper": float(upper),
        },
    }


def series_clustered_paired_bootstrap_ci(
    baseline_prediction: Any,
    candidate_prediction: Any,
    target: Any,
    series_id: Any,
    *,
    seed: int,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    draw_chunk_size: int = DEFAULT_BOOTSTRAP_DRAW_CHUNK_SIZE,
) -> dict[str, Any]:
    """Return deterministic paired percentile intervals clustered by series.

    Both reported deltas are ``candidate - baseline``.  A negative interval
    therefore supports an error reduction.  Each bootstrap replicate samples
    the observed number of series with replacement and includes every row of
    each sampled series.  Unequal series lengths are handled by dividing the
    resampled error sums by the resampled row count.

    The implementation first reduces rows to three sufficient statistics per
    series (row count, absolute-error delta sum, and squared-error delta sum).
    It then streams one replicate in bounded draw chunks.  In particular, it
    never allocates a ``replicates x series`` matrix, which keeps 500-replicate
    audits practical for datasets with hundreds of thousands of series.
    """
    baseline = _nonnegative_vector(baseline_prediction, "baseline_prediction")
    candidate = _nonnegative_vector(
        candidate_prediction, "candidate_prediction", n=len(baseline)
    )
    target_array = _nonnegative_vector(target, "target", n=len(baseline))
    if len(baseline) < 2:
        raise ValueError("paired bootstrap requires at least two rows")
    tokens = _series_tokens(series_id, n=len(baseline))
    seed = _integer_parameter(seed, "seed", minimum=0, maximum=2**64 - 1)
    replicates = _integer_parameter(replicates, "replicates", minimum=2)
    draw_chunk_size = _integer_parameter(
        draw_chunk_size, "draw_chunk_size", minimum=1
    )

    unique_series, inverse = np.unique(tokens, return_inverse=True)
    number_of_series = int(len(unique_series))
    if number_of_series < 2:
        raise ValueError("paired bootstrap requires at least two distinct series")

    baseline_error = baseline - target_array
    candidate_error = candidate - target_array
    mae_delta_by_row = np.abs(candidate_error) - np.abs(baseline_error)
    squared_error_delta_by_row = (
        np.square(candidate_error) - np.square(baseline_error)
    )
    if not (
        np.isfinite(mae_delta_by_row).all()
        and np.isfinite(squared_error_delta_by_row).all()
    ):
        raise FloatingPointError("paired bootstrap error deltas must be finite")

    rows_by_series = np.bincount(
        inverse, minlength=number_of_series
    ).astype(np.float64, copy=False)
    mae_delta_sum_by_series = np.bincount(
        inverse,
        weights=mae_delta_by_row,
        minlength=number_of_series,
    )
    squared_error_delta_sum_by_series = np.bincount(
        inverse,
        weights=squared_error_delta_by_row,
        minlength=number_of_series,
    )
    if not (
        np.isfinite(mae_delta_sum_by_series).all()
        and np.isfinite(squared_error_delta_sum_by_series).all()
    ):
        raise FloatingPointError("paired bootstrap series statistics must be finite")

    mae_bootstrap = np.empty(replicates, dtype=np.float64)
    squared_error_bootstrap = np.empty(replicates, dtype=np.float64)
    bounded_chunk_size = min(draw_chunk_size, number_of_series)
    for replicate in range(replicates):
        # A replicate-specific stream makes the result independent of global
        # NumPy RNG state and preserves each replicate if execution is split.
        replicate_rng = np.random.default_rng(
            np.random.SeedSequence([seed, replicate])
        )
        sampled_rows = 0.0
        sampled_mae_delta = 0.0
        sampled_squared_error_delta = 0.0
        remaining = number_of_series
        while remaining:
            draw_size = min(bounded_chunk_size, remaining)
            sampled_series = replicate_rng.integers(
                0,
                number_of_series,
                size=draw_size,
                dtype=np.int64,
            )
            sampled_rows += float(rows_by_series[sampled_series].sum(dtype=np.float64))
            sampled_mae_delta += float(
                mae_delta_sum_by_series[sampled_series].sum(dtype=np.float64)
            )
            sampled_squared_error_delta += float(
                squared_error_delta_sum_by_series[sampled_series].sum(
                    dtype=np.float64
                )
            )
            remaining -= draw_size
        mae_bootstrap[replicate] = sampled_mae_delta / sampled_rows
        squared_error_bootstrap[replicate] = (
            sampled_squared_error_delta / sampled_rows
        )

    mae_point_estimate = float(mae_delta_by_row.mean(dtype=np.float64))
    squared_error_point_estimate = float(
        squared_error_delta_by_row.mean(dtype=np.float64)
    )
    result = {
        "protocol": BOOTSTRAP_PROTOCOL,
        "resampling_unit": "series",
        "pairing": "same_target_row_and_series",
        "delta_definition": "candidate_minus_baseline",
        "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
        "interval_method": "percentile_linear",
        "seed": seed,
        "replicates": replicates,
        "rows": int(len(baseline)),
        "series": number_of_series,
        "mae_delta_candidate_minus_baseline": _bootstrap_interval_summary(
            mae_bootstrap,
            point_estimate=mae_point_estimate,
        ),
        "squared_error_delta_candidate_minus_baseline": _bootstrap_interval_summary(
            squared_error_bootstrap,
            point_estimate=squared_error_point_estimate,
        ),
        "implementation": {
            "aggregation": "per_series_sufficient_statistics",
            "replicate_streaming": True,
            "draw_chunk_size": bounded_chunk_size,
            "maximum_sampled_series_buffer": bounded_chunk_size,
        },
    }
    # All paths above should already be finite native Python scalars.  Keeping
    # this explicit check makes failure happen here rather than while an audit
    # artifact is being written much later.
    try:
        import json

        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FloatingPointError("paired bootstrap result is not JSON-safe") from exc
    return result


def two_fold_oof_audit(
    frozen_prediction: Any,
    target: Any,
    series_id: Any,
    *,
    fold: Any | None = None,
    fold_salt: str = FOLD_SALT,
    minimum_slope: float = MINIMUM_SLOPE,
    body_threshold: float,
    extreme_tail_threshold: float,
    quantity_boundaries: Sequence[float] | None = None,
    history_length: Any | None = None,
    history_boundaries: Sequence[float] | None = None,
    history_labels: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Cross-fit the calibrator on two deterministic series-disjoint folds."""
    prediction = _nonnegative_vector(frozen_prediction, "frozen_prediction")
    target = _nonnegative_vector(target, "target", n=len(prediction))
    if len(prediction) < 4:
        raise ValueError("two-fold OOF calibration requires at least four rows")
    tokens = _series_tokens(series_id, n=len(prediction))
    folds = assign_series_folds(series_id, salt=fold_salt) if fold is None else fold
    _, folds = validate_series_disjoint_folds(series_id, folds)
    minimum_slope = _positive_float(minimum_slope, "minimum_slope")
    normalized_history = None
    if history_length is not None:
        normalized_history = _nonnegative_vector(
            history_length, "history_length", n=len(prediction)
        )

    oof_prediction = np.empty_like(prediction, dtype=np.float64)
    fits: dict[str, Any] = {}
    fold_summaries: dict[str, Any] = {}
    for held_fold in (0, 1):
        held = folds == held_fold
        train = ~held
        if int(held.sum()) < 1 or int(train.sum()) < 2:
            raise ValueError("each held fold needs rows and each fit fold needs at least two rows")
        calibration = fit_positive_affine_ols(
            prediction[train], target[train], minimum_slope=minimum_slope
        )
        oof_prediction[held] = apply_affine_calibration(prediction[held], calibration)
        fits[str(held_fold)] = {
            "held_fold": held_fold,
            "fit_fold": 1 - held_fold,
            "fit_series": int(len(set(tokens[train].tolist()))),
            "held_series": int(len(set(tokens[held].tolist()))),
            **calibration,
        }

    reporting_arguments: dict[str, Any] = {
        "body_threshold": body_threshold,
        "extreme_tail_threshold": extreme_tail_threshold,
        "quantity_boundaries": quantity_boundaries,
    }
    if normalized_history is not None:
        reporting_arguments.update(
            history_length=normalized_history,
            history_boundaries=history_boundaries,
            history_labels=history_labels,
        )
    baseline = compute_reporting_metrics(prediction, target, **reporting_arguments)
    candidate = compute_reporting_metrics(oof_prediction, target, **reporting_arguments)
    for held_fold in (0, 1):
        selected = folds == held_fold
        selected_arguments = dict(reporting_arguments)
        if normalized_history is not None:
            selected_arguments["history_length"] = normalized_history[selected]
        fold_baseline = compute_reporting_metrics(
            prediction[selected], target[selected], **selected_arguments
        )
        fold_candidate = compute_reporting_metrics(
            oof_prediction[selected], target[selected], **selected_arguments
        )
        fold_summaries[str(held_fold)] = {
            "rows": int(selected.sum()),
            "series": int(len(set(tokens[selected].tolist()))),
            "baseline": fold_baseline,
            "calibrated": fold_candidate,
            "overall_change": _comparison(
                fold_baseline["overall"], fold_candidate["overall"]
            ),
        }

    summary = {
        "protocol": PROTOCOL,
        "scope": "train_internal_series_disjoint_two_fold_oof",
        "rows": int(len(prediction)),
        "series": int(len(set(tokens.tolist()))),
        "fold_rule": (
            "caller_supplied_series_disjoint"
            if fold is not None
            else "sha256_series_id_modulo_2"
        ),
        "fold_salt": fold_salt if fold is None else None,
        "minimum_slope": minimum_slope,
        "fits_by_held_fold": fits,
        "baseline": baseline,
        "calibrated": candidate,
        "overall_change": _comparison(baseline["overall"], candidate["overall"]),
        "folds": fold_summaries,
        "both_folds_raw_mse_improve": bool(
            all(
                (
                    fold_summaries[str(value)]["overall_change"][
                        "mse_relative_improvement"
                    ]
                    is not None
                    and fold_summaries[str(value)]["overall_change"][
                        "mse_relative_improvement"
                    ]
                    > 0
                )
                for value in (0, 1)
            )
        ),
    }
    arrays = {
        "frozen_prediction": prediction.copy(),
        "target": target.copy(),
        "series_token": tokens.copy(),
        "fold": folds.copy(),
        "oof_calibrated_prediction": oof_prediction,
    }
    return summary, arrays
