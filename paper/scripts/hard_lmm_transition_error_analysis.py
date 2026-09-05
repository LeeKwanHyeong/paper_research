"""Fixed train-internal OOF analysis for transition-error correction features.

The analysis cross-fits only a small preactivation correction on two frozen,
series-disjoint train folds.  It does not train or evaluate the backbone and a
positive result is not held-out generalization evidence.

For every arm, the correction is fitted at the same location used by the
proposed implementation::

    corrected_log = softplus(base_pre_activation + intercept + z @ beta)

Here ``z`` uses the opposite fold's mean and population standard deviation.
The fit minimizes summed squared log1p error plus ``1 * ||beta||^2``.  The
intercept is deliberately unpenalized.  ``scipy.optimize.least_squares`` is
used with a zero initialization and an analytic Jacobian; no solver setting or
hyperparameter is selected from the results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
from scipy.optimize import least_squares


PROTOCOL = "hard_lmm_transition_error_oof_v1"
DATASETS = ("intermittent_v2", "yellow_trip_hourly", "insta_market_basket")
CONTEXT_FIELDS = (
    "base_pre_activation",
    "history_length",
    "transition_count",
    "selected_occupied_fraction",
    "selected_count_mean",
    "selected_count_std",
    "last_error",
    "unconditioned_mean_error",
)
ARMS = ("strong_control", "candidate", "sham")
REQUIRED_ARRAYS = (
    "quantity",
    *CONTEXT_FIELDS,
    "prototype_conditioned_error",
    "sham_prototype_conditioned_error",
    "fold",
    "series_id",
)
RIDGE = 1.0
MINIMUM_GAIN = 0.01
MIN_TARGETS_PER_FOLD = 512
MIN_SERIES_PER_FOLD = 30
MAX_TARGETS = 8192
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 20260906
BONFERRONI_QUANTILE = 0.05 / 6


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _vector(value, name: str, n: int | None = None) -> np.ndarray:
    try:
        result = np.asarray(_numpy(value), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must have shape [N]")
    if n is not None and len(result) != n:
        raise ValueError(f"{name} row count mismatch")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if integer != value or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _softplus(value: np.ndarray) -> np.ndarray:
    """Stable NumPy softplus."""
    return np.logaddexp(0.0, value)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    """Stable derivative of softplus."""
    result = np.empty_like(value, dtype=np.float64)
    nonnegative = value >= 0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exponential = np.exp(value[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _validate_cache(cache: Mapping) -> dict[str, np.ndarray | float]:
    if not isinstance(cache, Mapping):
        raise ValueError("cache must be a mapping")
    missing = [name for name in (*REQUIRED_ARRAYS, "body_threshold") if name not in cache]
    if missing:
        raise ValueError(f"cache is missing required fields: {', '.join(missing)}")

    quantity = _vector(cache["quantity"], "quantity")
    n = len(quantity)
    if not 1 <= n <= MAX_TARGETS:
        raise ValueError(f"quantity must contain 1..{MAX_TARGETS} rows")
    data: dict[str, np.ndarray | float] = {"quantity": quantity}
    for name in REQUIRED_ARRAYS[1:]:
        data[name] = _vector(cache[name], name, n)

    body_threshold = np.asarray(_numpy(cache["body_threshold"]), dtype=np.float64)
    if body_threshold.ndim != 0 or not np.isfinite(body_threshold).item() or body_threshold.item() < 0:
        raise ValueError("body_threshold must be a finite nonnegative scalar")
    data["body_threshold"] = float(body_threshold)

    nonnegative_fields = (
        "quantity",
        "history_length",
        "transition_count",
        "selected_occupied_fraction",
        "selected_count_mean",
        "selected_count_std",
    )
    for name in nonnegative_fields:
        if np.any(data[name] < 0):
            raise ValueError(f"{name} must be nonnegative")
    if np.any(data["selected_occupied_fraction"] > 1):
        raise ValueError("selected_occupied_fraction must lie in [0, 1]")

    for name in ("history_length", "transition_count", "fold", "series_id"):
        if np.any(data[name] % 1):
            raise ValueError(f"{name} must contain integer values")
    if np.any(data["history_length"] < 1):
        raise ValueError("history_length must be at least one")
    if np.any(data["transition_count"] > data["history_length"] - 1):
        raise ValueError("transition_count cannot exceed history_length - 1")
    no_transition = data["transition_count"] == 0
    transition_summary_fields = (
        "selected_occupied_fraction",
        "selected_count_mean",
        "selected_count_std",
        "last_error",
        "unconditioned_mean_error",
        "prototype_conditioned_error",
        "sham_prototype_conditioned_error",
    )
    if any(np.any(data[name][no_transition] != 0) for name in transition_summary_fields):
        raise ValueError("rows without transitions must have zero transition-error summaries")
    if np.any(data["series_id"] < 0):
        raise ValueError("series_id must be nonnegative")
    folds = data["fold"].astype(np.int64)
    series = data["series_id"].astype(np.int64)
    if set(folds.tolist()) != {0, 1}:
        raise ValueError("fold must contain exactly folds 0 and 1")
    fold_series = [np.unique(series[folds == fold]) for fold in (0, 1)]
    if np.intersect1d(*fold_series).size:
        raise ValueError("series_id values must be disjoint across folds")
    data["fold"], data["series_id"] = folds, series
    return data


def _designs(data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    columns = []
    for name in CONTEXT_FIELDS:
        value = data[name]
        if name in ("history_length", "transition_count", "selected_count_mean", "selected_count_std"):
            value = np.log1p(value)
        columns.append(value)
    context = np.column_stack(columns).astype(np.float64, copy=False)
    return {
        "strong_control": context,
        "candidate": np.column_stack((context, data["prototype_conditioned_error"])),
        "sham": np.column_stack((context, data["sham_prototype_conditioned_error"])),
    }


def fit_preactivation_correction(
    train_x,
    train_base_pre_activation,
    train_log_target,
    test_x,
    test_base_pre_activation,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit one frozen nonlinear readout and predict one held fold.

    The least-squares residual vector appends the slopes themselves.  Its
    squared norm is therefore exactly log-target SSE plus the fixed unit L2
    penalty, matching standard ridge with ``Gram + I``.  This is the minor
    solver adaptation needed to express penalized regression through
    ``least_squares`` while keeping an analytic Jacobian.
    """
    train_x = np.asarray(_numpy(train_x), dtype=np.float64)
    test_x = np.asarray(_numpy(test_x), dtype=np.float64)
    if train_x.ndim != 2 or test_x.ndim != 2:
        raise ValueError("train_x and test_x must have shape [N,D]")
    if not len(train_x) or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train_x must be nonempty and train/test dimensions must match")
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise ValueError("train_x and test_x must be finite")
    train_base = _vector(train_base_pre_activation, "train_base_pre_activation", len(train_x))
    test_base = _vector(test_base_pre_activation, "test_base_pre_activation", len(test_x))
    target = _vector(train_log_target, "train_log_target", len(train_x))
    if (target < 0).any():
        raise ValueError("train_log_target must be nonnegative")

    mean = train_x.mean(axis=0)
    raw_scale = train_x.std(axis=0)
    zero_variance = raw_scale == 0
    scale = np.where(zero_variance, 1.0, raw_scale)
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
    dimensions = train_x.shape[1]
    def residual(parameters: np.ndarray) -> np.ndarray:
        preactivation = train_base + parameters[0] + train_z @ parameters[1:]
        data_residual = _softplus(preactivation) - target
        return np.concatenate((data_residual, np.sqrt(RIDGE) * parameters[1:]))

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        preactivation = train_base + parameters[0] + train_z @ parameters[1:]
        derivative = _sigmoid(preactivation)
        data_jacobian = np.column_stack((derivative, derivative[:, None] * train_z))
        penalty_jacobian = np.zeros((dimensions, dimensions + 1), dtype=np.float64)
        penalty_jacobian[:, 1:] = np.sqrt(RIDGE) * np.eye(dimensions)
        return np.vstack((data_jacobian, penalty_jacobian))

    fit = least_squares(
        residual,
        np.zeros(dimensions + 1, dtype=np.float64),
        jac=jacobian,
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=2000,
    )
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"least_squares correction fit failed: {fit.message}")

    correction = fit.x[0] + test_z @ fit.x[1:]
    corrected_pre = test_base + correction
    log_prediction = _softplus(corrected_pre)
    if not all(np.isfinite(value).all() for value in (correction, corrected_pre, log_prediction)):
        raise FloatingPointError("nonfinite held-fold correction prediction")
    train_data_residual = _softplus(train_base + fit.x[0] + train_z @ fit.x[1:]) - target
    metadata = {
        "train_rows": int(len(train_x)),
        "input_dimensions": int(dimensions),
        "input_mean": mean.tolist(),
        "input_raw_std": raw_scale.tolist(),
        "input_scale": scale.tolist(),
        "zero_variance_dimensions": np.flatnonzero(zero_variance).tolist(),
        "intercept": float(fit.x[0]),
        "coefficients": fit.x[1:].tolist(),
        "ridge_summed_sse_penalty": RIDGE,
        "train_log_sse": float(train_data_residual @ train_data_residual),
        "train_log_mse": float(np.mean(train_data_residual ** 2)),
        "l2_penalty": float(RIDGE * (fit.x[1:] @ fit.x[1:])),
        "objective_sum_squares": float(2 * fit.cost),
        "optimizer": "scipy.optimize.least_squares",
        "jacobian": "analytic",
        "initialization": "all zeros",
        "nfev": int(fit.nfev),
        "njev": int(fit.njev) if fit.njev is not None else None,
        "status": int(fit.status),
        "message": str(fit.message),
    }
    predictions = {
        "preactivation_correction": correction,
        "corrected_pre_activation": corrected_pre,
        "log_prediction": log_prediction,
    }
    return predictions, metadata


def _oof_arm(
    x: np.ndarray,
    base_pre_activation: np.ndarray,
    log_target: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    output = {
        "preactivation_correction": np.empty(len(x), dtype=np.float64),
        "corrected_pre_activation": np.empty(len(x), dtype=np.float64),
        "log_prediction": np.empty(len(x), dtype=np.float64),
    }
    metadata = {}
    for held_fold in (0, 1):
        held = folds == held_fold
        prediction, metadata[str(held_fold)] = fit_preactivation_correction(
            x[~held], base_pre_activation[~held], log_target[~held],
            x[held], base_pre_activation[held],
        )
        for name in output:
            output[name][held] = prediction[name]
        metadata[str(held_fold)]["training_fold"] = 1 - held_fold
        metadata[str(held_fold)]["held_fold"] = held_fold
    return output, metadata


def _raw_prediction(log_prediction: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        prediction = np.expm1(log_prediction)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite raw quantity prediction")
    return prediction


def _log_mse(squared_errors: np.ndarray, folds: np.ndarray) -> dict:
    return {
        "pooled": float(squared_errors.mean()),
        "folds": {
            str(fold): float(squared_errors[folds == fold].mean()) for fold in (0, 1)
        },
    }


def _body_mae(
    raw_prediction: np.ndarray,
    quantity: np.ndarray,
    folds: np.ndarray,
    body_threshold: float,
) -> dict:
    absolute_error = np.abs(raw_prediction - quantity)
    body = quantity <= body_threshold

    def view(selected: np.ndarray) -> dict:
        rows = selected & body
        return {
            "n": int(rows.sum()),
            "mae": float(absolute_error[rows].mean()) if rows.any() else None,
        }

    return {
        "pooled": view(np.ones(len(quantity), dtype=bool)),
        "folds": {str(fold): view(folds == fold) for fold in (0, 1)},
    }


def _support(data: Mapping[str, np.ndarray | float]) -> tuple[dict, bool]:
    folds, series, quantity = data["fold"], data["series_id"], data["quantity"]
    body_threshold = data["body_threshold"]
    details = {}
    for fold in (0, 1):
        selected = folds == fold
        targets = int(selected.sum())
        n_series = int(len(np.unique(series[selected])))
        body_targets = int((selected & (quantity <= body_threshold)).sum())
        reasons = []
        if targets < MIN_TARGETS_PER_FOLD:
            reasons.append(f"fewer_than_{MIN_TARGETS_PER_FOLD}_targets")
        if n_series < MIN_SERIES_PER_FOLD:
            reasons.append(f"fewer_than_{MIN_SERIES_PER_FOLD}_series")
        if body_targets == 0:
            reasons.append("no_body_targets")
        details[str(fold)] = {
            "targets": targets,
            "series": n_series,
            "body_targets": body_targets,
            "assessable": not reasons,
            "reasons": reasons,
        }
    return details, bool(all(view["assessable"] for view in details.values()))


def _cluster_groups(series: np.ndarray, folds: np.ndarray):
    groups = []
    for fold in (0, 1):
        rows = np.flatnonzero(folds == fold)
        ids, inverse = np.unique(series[rows], return_inverse=True)
        groups.append((rows, inverse, len(ids)))
    return groups


def paired_series_bootstrap_many(
    improvements,
    series,
    folds,
    repeats: int = BOOTSTRAP_REPEATS,
) -> list[dict]:
    """Bootstrap fixed paired predictions with shared series draws.

    The same sampled series indices are used for all supplied contrast columns.
    Series are resampled within each frozen fold and the two folds are pooled by
    their sampled row counts.  Decoders and preprocessing are never refitted.
    """
    repeats = _positive_integer(repeats, "bootstrap_repeats")
    values = np.asarray(_numpy(improvements), dtype=np.float64)
    if values.ndim != 2 or not len(values):
        raise ValueError("improvements must have shape [N,C] with N,C >= 1")
    if values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("improvements must be nonempty and finite")
    series = _vector(series, "series_id", len(values))
    folds = _vector(folds, "fold", len(values))
    if np.any(series % 1) or np.any(folds % 1) or set(folds.tolist()) != {0.0, 1.0}:
        raise ValueError("bootstrap needs integer series_id and exactly folds 0 and 1")
    series, folds = series.astype(np.int64), folds.astype(np.int64)
    if np.intersect1d(series[folds == 0], series[folds == 1]).size:
        raise ValueError("bootstrap series_id values must be disjoint across folds")

    summaries = []
    grouped = []
    for rows, inverse, size in _cluster_groups(series, folds):
        sums = np.column_stack([
            np.bincount(inverse, weights=values[rows, column], minlength=size)
            for column in range(values.shape[1])
        ])
        grouped.append((sums, np.bincount(inverse, minlength=size), size))
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty((repeats, values.shape[1]), dtype=np.float64)
    for start in range(0, repeats, 32):
        batch = min(32, repeats - start)
        total = np.zeros((batch, values.shape[1]), dtype=np.float64)
        count = np.zeros(batch, dtype=np.float64)
        for sums, counts, size in grouped:
            selected = generator.integers(0, size, size=(batch, size))
            total += sums[selected].sum(axis=1)
            count += counts[selected].sum(axis=1)
        draws[start:start + batch] = total / count[:, None]
    if not np.isfinite(draws).all():
        raise FloatingPointError("nonfinite series-bootstrap result")
    for column in range(values.shape[1]):
        summaries.append({
            "unit": "series",
            "repeats": repeats,
            "seed": BOOTSTRAP_SEED,
            "one_sided_bonferroni_quantile": BONFERRONI_QUANTILE,
            "lower_bonferroni": float(np.quantile(draws[:, column], BONFERRONI_QUANTILE)),
            "lower_p05": float(np.quantile(draws[:, column], 0.05)),
            "median": float(np.median(draws[:, column])),
            "shared_draws_across_contrasts": True,
            "fixed_oof_predictions": True,
            "interpretation": (
                "series resampled within frozen folds using shared draws; fixed OOF "
                "predictions, no refitting; descriptive train-internal stability"
            ),
        })
    return summaries


def _body_noninferior(candidate: dict, reference: dict) -> tuple[dict, bool]:
    pooled = bool(candidate["pooled"]["mae"] <= reference["pooled"]["mae"])
    folds = {
        str(fold): bool(candidate["folds"][str(fold)]["mae"] <= reference["folds"][str(fold)]["mae"])
        for fold in (0, 1)
    }
    detail = {"pooled": pooled, "folds": folds}
    return detail, bool(pooled and all(folds.values()))


def _comparison(
    reference_name: str,
    reference_errors: np.ndarray,
    candidate_errors: np.ndarray,
    folds: np.ndarray,
    reference_body: dict,
    candidate_body: dict,
    bootstrap: dict,
) -> dict:
    reference_mse = float(reference_errors.mean())
    candidate_mse = float(candidate_errors.mean())
    relative_gain = (
        (reference_mse - candidate_mse) / reference_mse if reference_mse > 0 else None
    )
    improvement = reference_errors - candidate_errors
    fold_improvements = {
        str(fold): float(improvement[folds == fold].mean()) for fold in (0, 1)
    }
    body_detail, body_passes = _body_noninferior(candidate_body, reference_body)
    conditions = {
        "pooled_log_mse_gain_at_least_one_percent": bool(
            relative_gain is not None and relative_gain >= MINIMUM_GAIN
        ),
        "both_fold_log_mse_improvements_positive": bool(
            all(value > 0 for value in fold_improvements.values())
        ),
        "bonferroni_series_bootstrap_lower_positive": bool(
            bootstrap["lower_bonferroni"] > 0
        ),
        "candidate_body_mae_noninferior_pooled_and_both_folds": body_passes,
    }
    return {
        "reference": reference_name,
        "reference_log_mse": reference_mse,
        "candidate_log_mse": candidate_mse,
        "relative_gain": relative_gain,
        "fold_mean_log_mse_improvements": fold_improvements,
        "body_mae_noninferiority": body_detail,
        "bootstrap": bootstrap,
        "conditions": conditions,
        "passes": bool(all(conditions.values())),
    }


def _base_arrays_and_view(data: Mapping[str, np.ndarray | float]) -> tuple[dict, dict]:
    folds, quantity = data["fold"], data["quantity"]
    log_target = np.log1p(quantity)
    base_log = _softplus(data["base_pre_activation"])
    base_raw = _raw_prediction(base_log)
    base_errors = (base_log - log_target) ** 2
    arrays = {
        name: data[name].copy() for name in REQUIRED_ARRAYS
    }
    arrays.update({
        "log_target": log_target,
        "original_t0__log_prediction": base_log,
        "original_t0__raw_prediction": base_raw,
        "original_t0__log_error": base_log - log_target,
        "original_t0__squared_log_error": base_errors,
        "original_t0__raw_absolute_error": np.abs(base_raw - quantity),
    })
    view = {
        "input_dimensions": 0,
        "log_mse": _log_mse(base_errors, folds),
        "raw_body_mae": _body_mae(
            base_raw, quantity, folds, data["body_threshold"],
        ),
    }
    return arrays, view


def analyze_cache(
    cache: Mapping,
    *,
    bootstrap_repeats: int = BOOTSTRAP_REPEATS,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Analyze one fixed dataset cache and return JSON summary plus row arrays."""
    bootstrap_repeats = _positive_integer(bootstrap_repeats, "bootstrap_repeats")
    data = _validate_cache(cache)
    folds, series, quantity = data["fold"], data["series_id"], data["quantity"]
    fold_support, assessable = _support(data)
    output_arrays, original_t0 = _base_arrays_and_view(data)
    contract = {
        "arms": {
            "strong_control": list(CONTEXT_FIELDS),
            "candidate": [*CONTEXT_FIELDS, "prototype_conditioned_error"],
            "sham": [*CONTEXT_FIELDS, "sham_prototype_conditioned_error"],
        },
        "transforms": {
            name: "log1p" for name in (
                "history_length", "transition_count", "selected_count_mean", "selected_count_std"
            )
        },
        "prediction": "softplus(base_pre_activation + intercept + standardized_X @ beta)",
        "objective": "summed squared log1p error + 1 * squared slopes",
        "intercept_penalized": False,
        "ridge_summed_sse_penalty": RIDGE,
        "standardization": "opposite-fold population mean/std; zero std uses one",
        "optimizer": "scipy.optimize.least_squares, zero init, analytic Jacobian",
        "minimum_relative_gain": MINIMUM_GAIN,
        "minimum_targets_per_fold": MIN_TARGETS_PER_FOLD,
        "minimum_series_per_fold": MIN_SERIES_PER_FOLD,
        "bootstrap_repeats": bootstrap_repeats,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bonferroni_one_sided_quantile": BONFERRONI_QUANTILE,
        "fixed_prediction_bootstrap": True,
        "hyperparameter_search": False,
        "row_scope": (
            "all cached targets, including H=1 rows with zero transition-memory features; "
            "this diagnostic readout does not replace the later H=1 identity contract"
        ),
    }

    if not assessable:
        summary = {
            "protocol": PROTOCOL,
            "status": "UNASSESSABLE",
            "assessable": False,
            "n_targets": int(len(quantity)),
            "body_threshold": float(data["body_threshold"]),
            "folds": fold_support,
            "decoder_contract": contract,
            "arms": {"original_t0": original_t0},
            "gate": {
                "comparisons": {},
                "original_t0_body_mae_noninferiority": None,
                "conditions": {"assessable_support": False},
                "passes": False,
            },
            "interpretation": (
                "Insufficient fixed-fold support for the preregistered analysis; no correction "
                "was fitted and this is a failed common-gate cell."
            ),
        }
        json.dumps(summary, allow_nan=False)
        return summary, output_arrays

    designs = _designs(data)
    log_target = np.log1p(quantity)
    arm_views: dict[str, dict] = {"original_t0": original_t0}
    arm_errors: dict[str, np.ndarray] = {}
    arm_body: dict[str, dict] = {}
    for arm in ARMS:
        predictions, fits = _oof_arm(
            designs[arm], data["base_pre_activation"], log_target, folds,
        )
        raw_prediction = _raw_prediction(predictions["log_prediction"])
        squared_error = (predictions["log_prediction"] - log_target) ** 2
        if not np.isfinite(squared_error).all():
            raise FloatingPointError(f"nonfinite {arm} squared log error")
        body = _body_mae(raw_prediction, quantity, folds, data["body_threshold"])
        arm_errors[arm], arm_body[arm] = squared_error, body
        arm_views[arm] = {
            "input_dimensions": int(designs[arm].shape[1]),
            "log_mse": _log_mse(squared_error, folds),
            "raw_body_mae": body,
            "fit": fits,
        }
        for name, value in predictions.items():
            output_arrays[f"{arm}__{name}"] = value
        output_arrays[f"{arm}__raw_prediction"] = raw_prediction
        output_arrays[f"{arm}__log_error"] = predictions["log_prediction"] - log_target
        output_arrays[f"{arm}__squared_log_error"] = squared_error
        output_arrays[f"{arm}__raw_absolute_error"] = np.abs(raw_prediction - quantity)

    improvements = np.column_stack((
        arm_errors["strong_control"] - arm_errors["candidate"],
        arm_errors["sham"] - arm_errors["candidate"],
    ))
    bootstrap = paired_series_bootstrap_many(
        improvements, series, folds, bootstrap_repeats,
    )
    comparisons = {
        reference: _comparison(
            reference,
            arm_errors[reference],
            arm_errors["candidate"],
            folds,
            arm_body[reference],
            arm_body["candidate"],
            bootstrap[index],
        )
        for index, reference in enumerate(("strong_control", "sham"))
    }
    base_body_detail, base_body_passes = _body_noninferior(
        arm_body["candidate"], original_t0["raw_body_mae"],
    )
    gate_conditions = {
        "assessable_support": True,
        "candidate_passes_vs_strong_control": comparisons["strong_control"]["passes"],
        "candidate_passes_vs_sham": comparisons["sham"]["passes"],
        "candidate_body_mae_noninferior_to_original_t0_pooled_and_both_folds": base_body_passes,
    }
    summary = {
        "protocol": PROTOCOL,
        "status": "PASS" if all(gate_conditions.values()) else "FAIL",
        "assessable": True,
        "n_targets": int(len(quantity)),
        "body_threshold": float(data["body_threshold"]),
        "folds": fold_support,
        "decoder_contract": contract,
        "arms": arm_views,
        "gate": {
            "comparisons": comparisons,
            "original_t0_body_mae_noninferiority": {
                **base_body_detail,
                "passes": base_body_passes,
            },
            "conditions": gate_conditions,
            "passes": bool(all(gate_conditions.values())),
        },
        "interpretation": (
            "Fixed train-internal OOF evidence for a preactivation correction. The frozen T0 "
            "already saw these train series, so this is a candidate-implementation gate rather "
            "than held-out performance evidence."
        ),
    }
    json.dumps(summary, allow_nan=False)
    if any(not np.isfinite(value).all() for value in output_arrays.values()):
        raise FloatingPointError("nonfinite output array")
    return summary, output_arrays


def aggregate_common_gate(results: Mapping) -> dict:
    """Require both fixed comparisons to pass on all three named datasets."""
    if not isinstance(results, Mapping) or set(results) != set(DATASETS):
        raise ValueError(
            "results must contain exactly intermittent_v2, yellow_trip_hourly, "
            "and insta_market_basket"
        )
    cells, comparison_views = {}, {}
    for reference in ("strong_control", "sham"):
        by_dataset = {}
        for dataset in DATASETS:
            result = results[dataset]
            if not isinstance(result, Mapping) or not isinstance(result.get("gate"), Mapping):
                raise ValueError(f"{dataset} must contain an analyze_cache gate")
            if result.get("protocol") != PROTOCOL:
                raise ValueError(f"{dataset} has an unexpected analysis protocol")
            assessable = result.get("assessable")
            gate_passes = result["gate"].get("passes")
            expected_status = "PASS" if assessable and gate_passes else (
                "FAIL" if assessable else "UNASSESSABLE"
            )
            if not isinstance(assessable, (bool, np.bool_)) or not isinstance(
                gate_passes, (bool, np.bool_)
            ) or result.get("status") != expected_status:
                raise ValueError(f"{dataset} has inconsistent assessable/status/gate fields")
            comparisons = result["gate"].get("comparisons")
            if not isinstance(comparisons, Mapping):
                raise ValueError(f"{dataset} gate must contain comparisons")
            comparison = comparisons.get(reference)
            by_dataset[dataset] = bool(
                result.get("assessable", False)
                and isinstance(comparison, Mapping)
                and comparison.get("passes", False)
            )
        comparison_views[reference] = {
            "datasets": by_dataset,
            "passes": bool(all(by_dataset.values())),
        }
    for dataset in DATASETS:
        result = results[dataset]
        original_body = result.get("gate", {}).get("original_t0_body_mae_noninferiority")
        original_body_passes = bool(
            isinstance(original_body, Mapping) and original_body.get("passes", False)
        )
        cells[dataset] = {
            "assessable": bool(result.get("assessable", False)),
            "strong_control": comparison_views["strong_control"]["datasets"][dataset],
            "sham": comparison_views["sham"]["datasets"][dataset],
            "original_t0_body_mae": original_body_passes,
            "passes": bool(result.get("gate", {}).get("passes", False)),
        }
    passes = bool(
        all(view["passes"] for view in comparison_views.values())
        and all(cell["original_t0_body_mae"] and cell["passes"] for cell in cells.values())
    )
    result = {
        "protocol": PROTOCOL,
        "required_datasets": list(DATASETS),
        "required_comparisons": ["strong_control", "sham"],
        "datasets": cells,
        "comparisons": comparison_views,
        "status": "PASS" if passes else "FAIL",
        "passes": passes,
        "interpretation": (
            "Common implementation gate: every named dataset must be assessable and the "
            "candidate must pass both fixed comparisons plus the original-T0 body-MAE guard."
        ),
    }
    json.dumps(result, allow_nan=False)
    return result
