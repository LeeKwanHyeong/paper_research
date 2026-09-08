#!/usr/bin/env python3
"""Analyze frozen Hard-LMM Q/K/V path counterfactuals with NumPy only.

This module never imports or runs a model. It validates the train-only frozen
extraction, aggregates the preregistered QK comparison, and applies the fixed
prospective research-cost gate. Counterfactual results diagnose a checkpoint;
they do not establish what a separately retrained QK model would achieve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CONTRACT_ID = "hard_lmm_qkv_path_diagnostic_v1"
CONTRACT_SHA256 = "a4ae310d254ed124fc8be4c340a10d22c462a33fb08d669afd38504cf978ddea"
VARIANTS = ("B", "FULL", "QK", "QV", "KV", "Q", "K", "V", "ZERO")
COMMON_FIELDS = (
    "fold",
    "dataset_index",
    "series_id",
    "true_quantity",
    "true_duration",
)
SCALAR_FIELDS = (
    "prediction",
    "log_mse",
    "absolute_error",
    "squared_error",
    "legacy_time_loss",
    "time_intercept_raw",
    "time_intercept_clamped",
    "exp_intercept",
    "w",
    "wd_raw",
    "wd_clamped",
    "integral_term",
    "wd_saturated",
    "intercept_saturated",
)
STATE_FIELDS = ("H1", "H2", "fused_state", "memory_residual")
TOP4_FIELD = "top4_indices"
VARIANT_FIELDS = SCALAR_FIELDS + STATE_FIELDS + (TOP4_FIELD,)
PAIR_SPECS = {
    "remove_Q_FULL_to_KV": ("KV", "FULL"),
    "remove_K_FULL_to_QV": ("QV", "FULL"),
    "remove_V_FULL_to_QK": ("QK", "FULL"),
    "QK_vs_FULL": ("QK", "FULL"),
    "QK_vs_ZERO": ("QK", "ZERO"),
    "QK_vs_B": ("QK", "B"),
}
INTERACTION_FIELDS = (
    "prediction",
    "log_mse",
    "absolute_error",
    "squared_error",
    "legacy_time_loss",
    "time_intercept_raw",
    "time_intercept_clamped",
    "integral_term",
)
GRADIENT_VARIANTS = ("B", "FULL")
GRADIENT_GROUPS = (
    "encoder_layer1",
    "encoder_layer2",
    "hard_lmm_bank",
    "new_Q",
    "new_K",
    "new_V",
)
GRADIENT_NUMERIC_FIELDS = (
    "dot_product",
    "time_norm",
    "quantity_norm",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_series_id(value: Any) -> str:
    """Mirror the frozen runner's cross-platform series-ID canonicalization."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and math.isfinite(float(value)):
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".17g")
    text = str(value)
    require(bool(text), "Canonical series ID is empty")
    return text


def fold_for_series(value: Any) -> int:
    payload = ("qkv-path-v1:" + canonical_series_id(value)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2


def load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, value in pairs:
            require(name not in result, f"Duplicate JSON key: {name}")
            result[name] = value
        return result

    require(path.is_file(), f"Missing JSON artifact: {path}")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    require(isinstance(payload, dict), f"JSON root must be an object: {path}")
    return payload


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == 1, "Contract schema drift")
    require(contract.get("contract_id") == CONTRACT_ID, "Contract ID drift")
    require(
        contract.get("status") == "frozen_before_train_counterfactual_outputs",
        "Contract was not frozen before outputs",
    )
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Contract scope is missing")
    for name, expected in {
        "split": "train",
        "validation_materialized": False,
        "held_out_materialized": False,
        "model_or_optimizer_updates": False,
        "head_refit": False,
        "remote_access": False,
        "device": "cpu",
    }.items():
        require(scope.get(name) == expected, f"Scope drift: {name}")
    require(tuple(contract.get("variant_order", [])) == VARIANTS, "Variant drift")
    measurements = contract.get("measurements")
    require(isinstance(measurements, Mapping), "Measurements are missing")
    measured = tuple(measurements.get("per_target", []))
    require(
        measured
        == (
            "prediction",
            "true_quantity",
            "true_duration",
            "log_mse",
            "absolute_error",
            "squared_error",
            "legacy_time_loss",
            "time_intercept_raw",
            "time_intercept_clamped",
            "exp_intercept",
            "w",
            "wd_raw",
            "wd_clamped",
            "integral_term",
            "wd_saturated",
            "intercept_saturated",
            "H1",
            "H2",
            "fused_state",
            "memory_residual",
            "top4_indices",
        ),
        "Per-target measurement schema drift",
    )
    sampling = contract.get("sampling")
    require(isinstance(sampling, Mapping), "Sampling contract is missing")
    require(
        sampling.get("fold_count") == 2
        and sampling.get("targets_per_dataset") == 4096
        and sampling.get("targets_per_fold") == 2048,
        "Sampling size drift",
    )
    gate = contract.get("prospective_QK_gate")
    require(isinstance(gate, Mapping), "Prospective QK gate is missing")
    require(
        contract.get("primary_comparison")
        == "QK (110) vs FULL (111), fixed before outputs; other switches are descriptive, not alternate-candidate selection.",
        "Primary comparison drift",
    )
    require(
        contract.get("interpretation", {}).get(
            "failed_counterfactual_does_not_prove_QK_training_impossible"
        )
        is True,
        "Failure interpretation drift",
    )


def _variant_key(variant: str, field: str) -> str:
    return f"{variant}__{field}"


def expected_npz_keys() -> set[str]:
    return set(COMMON_FIELDS) | {
        _variant_key(variant, field)
        for variant in VARIANTS
        for field in VARIANT_FIELDS
    }


def _as_array(mapping: Mapping[str, Any], name: str) -> np.ndarray:
    require(name in mapping, f"Missing extraction field: {name}")
    return np.asarray(mapping[name])


def _require_finite(array: np.ndarray, *, name: str) -> None:
    require(
        (
            np.issubdtype(array.dtype, np.number)
            or np.issubdtype(array.dtype, np.bool_)
        )
        and bool(np.isfinite(array).all()),
        f"{name} contains nonfinite or nonnumeric values",
    )


def validate_extraction(
    arrays: Mapping[str, Any],
    *,
    dataset_spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    enforce_sampling_contract: bool = True,
) -> dict[str, np.ndarray]:
    """Validate and copy the strict per-dataset NPZ interface."""
    observed_keys = set(arrays)
    required_keys = expected_npz_keys()
    require(
        observed_keys == required_keys,
        "Extraction NPZ keys differ from the frozen interface: "
        f"missing={sorted(required_keys - observed_keys)}, "
        f"extra={sorted(observed_keys - required_keys)}",
    )
    result = {name: _as_array(arrays, name).copy() for name in required_keys}
    fold = result["fold"]
    target_index = result["dataset_index"]
    series = result["series_id"]
    quantity = result["true_quantity"]
    duration = result["true_duration"]
    require(fold.ndim == 1, "fold must have shape [N]")
    n_targets = int(fold.shape[0])
    require(n_targets > 0, "Extraction is empty")
    for name in COMMON_FIELDS[1:]:
        require(result[name].shape == (n_targets,), f"{name} must have shape [N]")
    require(np.issubdtype(fold.dtype, np.integer), "fold must be integer")
    require(
        np.issubdtype(target_index.dtype, np.integer),
        "dataset_index must be integer",
    )
    require(set(np.unique(fold).tolist()) == {0, 1}, "fold must contain 0 and 1")
    require(
        np.unique(target_index).size == n_targets,
        "dataset_index values must be unique",
    )
    require(series.dtype.kind in "iuUS", "series_id dtype is unsupported")
    _require_finite(quantity, name="true_quantity")
    _require_finite(duration, name="true_duration")
    require(bool((quantity >= 0).all()), "true_quantity must be nonnegative")
    require(bool((duration > 0).all()), "true_duration must be positive")
    require(
        bool((target_index >= 0).all())
        and bool((target_index < int(dataset_spec["expected_train_targets"])).all()),
        "dataset_index is outside the train target population",
    )

    sampling = contract["sampling"]
    if enforce_sampling_contract:
        require(
            n_targets == int(sampling["targets_per_dataset"]),
            "Dataset target count drift",
        )
    fold_series: list[set[Any]] = []
    for fold_id in (0, 1):
        selected = fold == fold_id
        if enforce_sampling_contract:
            require(
                int(selected.sum()) == int(sampling["targets_per_fold"]),
                f"fold{fold_id} target count drift",
            )
            require(
                np.unique(series[selected]).size
                >= int(sampling["minimum_unique_series_per_fold"]),
                f"fold{fold_id} has insufficient unique series",
            )
        fold_series.append(set(np.unique(series[selected]).tolist()))
    require(
        not fold_series[0].intersection(fold_series[1]),
        "Series IDs must be disjoint across folds",
    )
    for value, observed_fold in zip(series.tolist(), fold.tolist()):
        require(
            fold_for_series(value) == int(observed_fold),
            "Series fold differs from the frozen hash rule",
        )

    vector_dimensions: dict[str, int] = {}
    for variant in VARIANTS:
        for field in SCALAR_FIELDS:
            name = _variant_key(variant, field)
            value = result[name]
            require(value.shape == (n_targets,), f"{name} must have shape [N]")
            _require_finite(value, name=name)
            if field in ("wd_saturated", "intercept_saturated"):
                require(
                    bool(np.isin(value, (0, 1)).all()),
                    f"{name} must be binary",
                )
        for field in STATE_FIELDS:
            name = _variant_key(variant, field)
            value = result[name]
            require(
                value.ndim == 2
                and value.shape[0] == n_targets
                and value.shape[1] > 0,
                f"{name} must have shape [N,D]",
            )
            _require_finite(value, name=name)
            if field in vector_dimensions:
                require(
                    vector_dimensions[field] == value.shape[1],
                    f"{field} hidden dimension differs by variant",
                )
            else:
                vector_dimensions[field] = int(value.shape[1])
        top4_name = _variant_key(variant, TOP4_FIELD)
        top4 = result[top4_name]
        require(
            top4.shape == (n_targets, 4)
            and np.issubdtype(top4.dtype, np.integer),
            f"{top4_name} must have integer shape [N,4]",
        )
        require(bool((top4 >= 0).all()), f"{top4_name} contains negative IDs")
        sorted_top4 = np.sort(top4, axis=1)
        require(
            bool((np.diff(sorted_top4, axis=1) != 0).all()),
            f"{top4_name} contains duplicate prototype IDs",
        )

        prediction = result[_variant_key(variant, "prediction")]
        require(bool((prediction >= 0).all()), f"{variant} prediction is negative")
        require(
            bool((result[_variant_key(variant, "w")] > 0).all()),
            f"{variant} w must be positive",
        )
        require(
            bool((result[_variant_key(variant, "exp_intercept")] > 0).all())
            and bool((result[_variant_key(variant, "integral_term")] >= 0).all()),
            f"{variant} exponential time telemetry is invalid",
        )
        expected_absolute = np.abs(prediction - quantity)
        expected_squared = np.square(prediction - quantity)
        expected_log_mse = np.square(np.log1p(prediction) - np.log1p(quantity))
        for field, expected in (
            ("absolute_error", expected_absolute),
            ("squared_error", expected_squared),
            ("log_mse", expected_log_mse),
        ):
            observed = result[_variant_key(variant, field)]
            require(
                bool(np.allclose(observed, expected, rtol=1e-5, atol=1e-6)),
                f"{variant} {field} does not match prediction and target",
            )
        w = result[_variant_key(variant, "w")]
        wd_raw = result[_variant_key(variant, "wd_raw")]
        wd_clamped = result[_variant_key(variant, "wd_clamped")]
        exp_intercept = result[_variant_key(variant, "exp_intercept")]
        intercept_clamped = result[
            _variant_key(variant, "time_intercept_clamped")
        ]
        integral = result[_variant_key(variant, "integral_term")]
        require(
            bool(np.allclose(wd_raw, w * duration, rtol=1e-12, atol=1e-12)),
            f"{variant} wd_raw telemetry drift",
        )
        require(
            bool(
                np.allclose(
                    wd_clamped,
                    np.minimum(wd_raw, 10.0),
                    rtol=1e-12,
                    atol=1e-12,
                )
            ),
            f"{variant} wd_clamped telemetry drift",
        )
        require(
            bool(
                np.array_equal(
                    result[_variant_key(variant, "wd_saturated")].astype(bool),
                    wd_raw > 10.0,
                )
            ),
            f"{variant} wd_saturated telemetry drift",
        )
        require(
            bool(
                np.allclose(
                    exp_intercept,
                    np.exp(intercept_clamped),
                    rtol=1e-12,
                    atol=1e-12,
                )
            ),
            f"{variant} exp_intercept telemetry drift",
        )
        require(
            bool(
                np.allclose(
                    integral,
                    (exp_intercept / w) * np.expm1(wd_clamped),
                    rtol=1e-12,
                    atol=1e-12,
                )
            ),
            f"{variant} integral telemetry drift",
        )
    for variant in VARIANTS[2:]:
        for field in ("w", "wd_raw", "wd_clamped", "wd_saturated"):
            require(
                bool(
                    np.array_equal(
                        result[_variant_key(variant, field)],
                        result[_variant_key("FULL", field)],
                    )
                ),
                f"Candidate-switch {field} differs between FULL and {variant}",
            )
    return result


def _safe_ratio(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    if reference == 0.0:
        return 1.0 if candidate == 0.0 else None
    return candidate / reference


def _safe_relative_improvement(
    reference: float | None, candidate: float | None
) -> float | None:
    ratio = _safe_ratio(candidate, reference)
    return None if ratio is None else 1.0 - ratio


def metric_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    variant: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    count = int(mask.sum())
    series_count = int(np.unique(arrays["series_id"][mask]).size)
    if count == 0:
        return {
            "target_count": 0,
            "unique_series": series_count,
            "raw_rmse": None,
            "mae": None,
            "log_mse": None,
            "legacy_time_loss": None,
            "prediction_mean": None,
            "wd_saturated_fraction": None,
            "intercept_saturated_fraction": None,
            "w_mean": None,
            "w_min": None,
            "w_max": None,
            "integral_term_mean": None,
            "unclamped_intercept_score_abs_mean": None,
            "unclamped_intercept_score_target_count": 0,
            "all_positive_wd_saturated_at_10": False,
            "theoretical_wd10_clamp_floor_mean": None,
            "legacy_time_loss_excess_over_wd10_floor": None,
        }
    squared = arrays[_variant_key(variant, "squared_error")][mask].astype(
        np.float64
    )
    w = arrays[_variant_key(variant, "w")][mask].astype(np.float64)
    integral = arrays[_variant_key(variant, "integral_term")][mask].astype(
        np.float64
    )
    wd_saturated = arrays[_variant_key(variant, "wd_saturated")][mask].astype(
        bool
    )
    wd_clamped = arrays[_variant_key(variant, "wd_clamped")][mask].astype(
        np.float64
    )
    intercept_unsaturated = ~arrays[
        _variant_key(variant, "intercept_saturated")
    ][mask].astype(bool)
    all_positive_wd_saturated_at_10 = bool(
        wd_saturated.all()
        and np.allclose(wd_clamped, 10.0, rtol=0.0, atol=1e-6)
    )
    clamp_floor_mean = None
    clamp_floor_excess = None
    if all_positive_wd_saturated_at_10:
        clamp_floor = 1.0 - np.log(w) + np.log1p(-np.exp(-10.0))
        clamp_floor_mean = float(np.mean(clamp_floor, dtype=np.float64))
        observed_time = arrays[_variant_key(variant, "legacy_time_loss")][
            mask
        ].astype(np.float64)
        clamp_floor_excess = float(
            np.mean(observed_time - clamp_floor, dtype=np.float64)
        )
    intercept_score = np.abs(integral[intercept_unsaturated] - 1.0)
    return {
        "target_count": count,
        "unique_series": series_count,
        "raw_rmse": float(np.sqrt(np.mean(squared, dtype=np.float64))),
        "mae": float(
            np.mean(
                arrays[_variant_key(variant, "absolute_error")][mask],
                dtype=np.float64,
            )
        ),
        "log_mse": float(
            np.mean(
                arrays[_variant_key(variant, "log_mse")][mask],
                dtype=np.float64,
            )
        ),
        "legacy_time_loss": float(
            np.mean(
                arrays[_variant_key(variant, "legacy_time_loss")][mask],
                dtype=np.float64,
            )
        ),
        "prediction_mean": float(
            np.mean(
                arrays[_variant_key(variant, "prediction")][mask],
                dtype=np.float64,
            )
        ),
        "wd_saturated_fraction": float(
            np.mean(
                arrays[_variant_key(variant, "wd_saturated")][mask],
                dtype=np.float64,
            )
        ),
        "intercept_saturated_fraction": float(
            np.mean(
                arrays[_variant_key(variant, "intercept_saturated")][mask],
                dtype=np.float64,
            )
        ),
        "w_mean": float(np.mean(w, dtype=np.float64)),
        "w_min": float(np.min(w)),
        "w_max": float(np.max(w)),
        "integral_term_mean": float(np.mean(integral, dtype=np.float64)),
        "unclamped_intercept_score_abs_mean": (
            float(np.mean(intercept_score, dtype=np.float64))
            if intercept_score.size
            else None
        ),
        "unclamped_intercept_score_target_count": int(intercept_score.size),
        "all_positive_wd_saturated_at_10": all_positive_wd_saturated_at_10,
        "theoretical_wd10_clamp_floor_mean": clamp_floor_mean,
        "legacy_time_loss_excess_over_wd10_floor": clamp_floor_excess,
    }


def compare_metric_summaries(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("raw_rmse", "mae", "log_mse"):
        candidate_value = candidate[metric]
        reference_value = reference[metric]
        result[f"{metric}_ratio"] = _safe_ratio(
            candidate_value, reference_value
        )
        result[f"{metric}_relative_improvement"] = _safe_relative_improvement(
            reference_value, candidate_value
        )
        result[f"{metric}_candidate_minus_reference"] = (
            None
            if candidate_value is None or reference_value is None
            else float(candidate_value - reference_value)
        )
    candidate_time = candidate["legacy_time_loss"]
    reference_time = reference["legacy_time_loss"]
    result["legacy_time_candidate_minus_reference"] = (
        None
        if candidate_time is None or reference_time is None
        else float(candidate_time - reference_time)
    )
    return result


def hidden_change_summary(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    reference = reference[mask].astype(np.float64)
    candidate = candidate[mask].astype(np.float64)
    count = int(reference.shape[0])
    if count == 0:
        return {
            "target_count": 0,
            "cosine_valid_count": 0,
            "cosine_mean": None,
            "reference_norm_mean": None,
            "candidate_norm_mean": None,
            "change_norm_mean": None,
            "relative_change_norm_mean": None,
        }
    reference_norm = np.linalg.norm(reference, axis=1)
    candidate_norm = np.linalg.norm(candidate, axis=1)
    change_norm = np.linalg.norm(candidate - reference, axis=1)
    valid = (reference_norm > 0.0) & (candidate_norm > 0.0)
    cosine = None
    if bool(valid.any()):
        cosine_values = np.sum(reference[valid] * candidate[valid], axis=1) / (
            reference_norm[valid] * candidate_norm[valid]
        )
        cosine_values = np.clip(cosine_values, -1.0, 1.0)
        cosine = float(np.mean(cosine_values, dtype=np.float64))
    relative = change_norm / np.maximum(reference_norm, 1e-12)
    return {
        "target_count": count,
        "cosine_valid_count": int(valid.sum()),
        "cosine_mean": cosine,
        "reference_norm_mean": float(np.mean(reference_norm)),
        "candidate_norm_mean": float(np.mean(candidate_norm)),
        "change_norm_mean": float(np.mean(change_norm)),
        "relative_change_norm_mean": float(np.mean(relative)),
    }


def top4_change_summary(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    reference = reference[mask]
    candidate = candidate[mask]
    count = int(reference.shape[0])
    if count == 0:
        return {
            "target_count": 0,
            "ordered_exact_fraction": None,
            "same_set_fraction": None,
            "mean_set_overlap_fraction": None,
            "mean_position_match_fraction": None,
        }
    ordered = np.all(reference == candidate, axis=1)
    sorted_reference = np.sort(reference, axis=1)
    sorted_candidate = np.sort(candidate, axis=1)
    same_set = np.all(sorted_reference == sorted_candidate, axis=1)
    overlap = np.empty(count, dtype=np.float64)
    for index in range(count):
        overlap[index] = np.intersect1d(
            reference[index], candidate[index], assume_unique=True
        ).size / 4.0
    return {
        "target_count": count,
        "ordered_exact_fraction": float(np.mean(ordered)),
        "same_set_fraction": float(np.mean(same_set)),
        "mean_set_overlap_fraction": float(np.mean(overlap)),
        "mean_position_match_fraction": float(np.mean(reference == candidate)),
    }


def interaction_summary(
    arrays: Mapping[str, np.ndarray],
    *,
    field: str,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Return FULL - QK - V + ZERO interaction telemetry."""
    interaction = (
        arrays[_variant_key("FULL", field)]
        - arrays[_variant_key("QK", field)]
        - arrays[_variant_key("V", field)]
        + arrays[_variant_key("ZERO", field)]
    )[mask].astype(np.float64)
    if interaction.shape[0] == 0:
        return {
            "target_count": 0,
            "mean": None,
            "mean_absolute": None,
            "root_mean_square": None,
        }
    if interaction.ndim == 1:
        magnitude = np.abs(interaction)
        square = np.square(interaction)
        mean = float(np.mean(interaction))
    else:
        magnitude = np.linalg.norm(interaction, axis=1)
        square = np.square(magnitude)
        mean = None
    return {
        "target_count": int(interaction.shape[0]),
        "mean": mean,
        "mean_absolute": float(np.mean(magnitude)),
        "root_mean_square": float(np.sqrt(np.mean(square))),
    }


def summarize_scope(
    arrays: Mapping[str, np.ndarray], mask: np.ndarray
) -> dict[str, Any]:
    metrics = {
        variant: metric_summary(arrays, variant=variant, mask=mask)
        for variant in VARIANTS
    }
    pairs: dict[str, Any] = {}
    for label, (candidate, reference) in PAIR_SPECS.items():
        if label == "QK_vs_B":
            hidden = {
                "comparable": False,
                "reason": (
                    "B and the causal-QKV checkpoint are independently trained; "
                    "their hidden coordinates do not share an identified basis."
                ),
                "fields": None,
            }
            top4 = {
                "comparable": False,
                "reason": (
                    "B and the causal-QKV checkpoint have independently learned "
                    "prototype banks; row IDs do not establish semantic identity."
                ),
                "target_count": int(mask.sum()),
                "ordered_exact_fraction": None,
                "same_set_fraction": None,
                "mean_set_overlap_fraction": None,
                "mean_position_match_fraction": None,
            }
        else:
            hidden = {
                "comparable": True,
                "fields": {
                    field: hidden_change_summary(
                        arrays[_variant_key(reference, field)],
                        arrays[_variant_key(candidate, field)],
                        mask,
                    )
                    for field in STATE_FIELDS
                },
            }
            top4 = {
                "comparable": True,
                **top4_change_summary(
                    arrays[_variant_key(reference, TOP4_FIELD)],
                    arrays[_variant_key(candidate, TOP4_FIELD)],
                    mask,
                ),
            }
        pairs[label] = {
            "metrics": compare_metric_summaries(
                metrics[candidate], metrics[reference]
            ),
            "hidden": hidden,
            "top4": top4,
        }
    return {
        "support": {
            "target_count": int(mask.sum()),
            "unique_series": int(np.unique(arrays["series_id"][mask]).size),
        },
        "variants": metrics,
        "pairs": pairs,
        "QK_V_interaction_FULL_minus_QK_minus_V_plus_ZERO": {
            field: interaction_summary(arrays, field=field, mask=mask)
            for field in INTERACTION_FIELDS + STATE_FIELDS
        },
    }


def _duration_masks(
    duration: np.ndarray, boundaries: Sequence[float]
) -> dict[str, np.ndarray]:
    require(len(boundaries) == 3, "Duration quartiles must have three boundaries")
    q25, q50, q75 = (float(value) for value in boundaries)
    require(
        all(math.isfinite(value) for value in (q25, q50, q75))
        and 0.0 < q25 <= q50 <= q75,
        "Duration quartile boundaries are invalid",
    )
    return {
        "duration_q1_le_q25": duration <= q25,
        "duration_q2_q25_to_q50": (duration > q25) & (duration <= q50),
        "duration_q3_q50_to_q75": (duration > q50) & (duration <= q75),
        "duration_q4_gt_q75": duration > q75,
    }


def analyze_dataset_arrays(
    arrays: Mapping[str, Any],
    *,
    dataset_spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    duration_quartile_boundaries: Sequence[float],
    enforce_sampling_contract: bool = True,
) -> dict[str, Any]:
    data = validate_extraction(
        arrays,
        dataset_spec=dataset_spec,
        contract=contract,
        enforce_sampling_contract=enforce_sampling_contract,
    )
    fold = data["fold"]
    quantity = data["true_quantity"]
    boundaries = dataset_spec["quantity_boundaries"]
    require(
        isinstance(boundaries, list)
        and len(boundaries) == 4
        and all(_is_finite_number(value) for value in boundaries),
        "Quantity boundaries are invalid",
    )
    body = quantity <= float(boundaries[2])
    tail = quantity > float(boundaries[3])
    duration_masks = _duration_masks(
        data["true_duration"], duration_quartile_boundaries
    )
    base_masks = {
        "pooled": np.ones(quantity.shape[0], dtype=bool),
        "fold0": fold == 0,
        "fold1": fold == 1,
        "body_le_train_p95": body,
        "tail_gt_train_p99": tail,
        **duration_masks,
    }
    masks = dict(base_masks)
    for fold_id in (0, 1):
        fold_mask = fold == fold_id
        masks[f"fold{fold_id}__body_le_train_p95"] = fold_mask & body
        masks[f"fold{fold_id}__tail_gt_train_p99"] = fold_mask & tail
        for name, duration_mask in duration_masks.items():
            masks[f"fold{fold_id}__{name}"] = fold_mask & duration_mask

    clamp_strata: dict[str, Any] = {}
    for variant in VARIANTS:
        saturated = data[_variant_key(variant, "wd_saturated")].astype(bool)
        clamp_strata[variant] = {
            "wd_saturated": metric_summary(
                data, variant=variant, mask=saturated
            ),
            "wd_unsaturated": metric_summary(
                data, variant=variant, mask=~saturated
            ),
            "folds": {
                str(fold_id): {
                    "wd_saturated": metric_summary(
                        data,
                        variant=variant,
                        mask=(fold == fold_id) & saturated,
                    ),
                    "wd_unsaturated": metric_summary(
                        data,
                        variant=variant,
                        mask=(fold == fold_id) & ~saturated,
                    ),
                }
                for fold_id in (0, 1)
            },
        }
    aggregates = {name: summarize_scope(data, mask) for name, mask in masks.items()}
    taxi_retention: dict[str, Any] | None = None
    if dataset_spec["dataset"] == "yellow_trip_hourly":
        taxi_retention = {}
        for scope_name in ("pooled", "fold0", "fold1"):
            variants = aggregates[scope_name]["variants"]
            B_rmse = variants["B"]["raw_rmse"]
            FULL_rmse = variants["FULL"]["raw_rmse"]
            QK_rmse = variants["QK"]["raw_rmse"]
            full_gain = B_rmse - FULL_rmse
            QK_gain = B_rmse - QK_rmse
            taxi_retention[scope_name] = {
                "FULL_quantity_gain_over_B": full_gain,
                "QK_quantity_gain_over_B": QK_gain,
                "QK_fraction_of_FULL_quantity_gain_retained": (
                    QK_gain / full_gain if full_gain > 0.0 else None
                ),
                "QK_remaining_legacy_time_gap_to_B": (
                    variants["QK"]["legacy_time_loss"]
                    - variants["B"]["legacy_time_loss"]
                ),
            }
    return {
        "dataset": dataset_spec["dataset"],
        "target_count": int(quantity.shape[0]),
        "duration_train_quartile_boundaries": [
            float(value) for value in duration_quartile_boundaries
        ],
        "quantity_body_definition": f"quantity <= train p95 ({boundaries[2]})",
        "quantity_tail_definition": f"quantity > train p99 ({boundaries[3]})",
        "aggregates": aggregates,
        "clamp_strata": clamp_strata,
        "taxi_B_comparison": taxi_retention,
        "B_comparison_is_descriptive": True,
    }


def _criterion(
    *,
    dataset: str,
    fold: int,
    name: str,
    observed: float | None,
    threshold: float,
    operator: str,
) -> dict[str, Any]:
    require(operator in {"<=", ">="}, "Unsupported gate operator")
    evaluable = observed is not None and math.isfinite(float(observed))
    passed = None
    if evaluable:
        value = float(observed)
        on_boundary = math.isclose(
            value, threshold, rel_tol=1e-12, abs_tol=1e-12
        )
        passed = (
            value <= threshold or on_boundary
            if operator == "<="
            else value >= threshold or on_boundary
        )
    return {
        "dataset": dataset,
        "fold": fold,
        "criterion": name,
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "evaluable": evaluable,
        "passed": passed,
    }


def evaluate_prospective_qk_gate(
    analyses: Mapping[str, Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply only the frozen prospective QK gate, in its declared order."""
    policy = contract["prospective_QK_gate"]
    thresholds = policy["every_dataset_and_fold"]
    sampling = contract["sampling"]
    criteria: list[dict[str, Any]] = []
    support_limitations: list[dict[str, Any]] = []
    time_triggered_folds: list[dict[str, Any]] = []
    invalid_cross_b_time_folds: set[tuple[str, int]] = set()
    contributing_datasets: list[str] = []
    for dataset_spec in contract["datasets"]:
        dataset = dataset_spec["dataset"]
        require(dataset in analyses, f"Missing dataset analysis: {dataset}")
        analysis = analyses[dataset]
        fold_contributions: list[bool] = []
        for fold_id in (0, 1):
            overall = analysis["aggregates"][f"fold{fold_id}"]["variants"]
            body = analysis["aggregates"][
                f"fold{fold_id}__body_le_train_p95"
            ]["variants"]
            tail_scope = analysis["aggregates"][
                f"fold{fold_id}__tail_gt_train_p99"
            ]
            tail = tail_scope["variants"]
            tail_support = tail_scope["support"]
            enough_tail = (
                int(tail_support["target_count"])
                >= int(sampling["tail_min_targets_per_fold_for_decision"])
                and int(tail_support["unique_series"])
                >= int(sampling["tail_min_unique_series_per_fold_for_decision"])
            )
            if not enough_tail:
                support_limitations.append(
                    {
                        "dataset": dataset,
                        "fold": fold_id,
                        "stratum": "tail_gt_train_p99",
                        **tail_support,
                        "minimum_targets": sampling[
                            "tail_min_targets_per_fold_for_decision"
                        ],
                        "minimum_unique_series": sampling[
                            "tail_min_unique_series_per_fold_for_decision"
                        ],
                    }
                )

            qk_full_overall = compare_metric_summaries(
                overall["QK"], overall["FULL"]
            )
            qk_zero_overall = compare_metric_summaries(
                overall["QK"], overall["ZERO"]
            )
            qk_full_body = compare_metric_summaries(body["QK"], body["FULL"])
            qk_zero_body = compare_metric_summaries(body["QK"], body["ZERO"])
            gate_values = (
                (
                    "QK_raw_rmse_max_ratio_FULL",
                    qk_full_overall["raw_rmse_ratio"],
                ),
                ("QK_mae_max_ratio_FULL", qk_full_overall["mae_ratio"]),
                (
                    "QK_body_mae_max_ratio_FULL",
                    qk_full_body["mae_ratio"],
                ),
                (
                    "QK_legacy_time_max_increase_FULL",
                    qk_full_overall[
                        "legacy_time_candidate_minus_reference"
                    ],
                ),
                (
                    "QK_raw_rmse_max_ratio_ZERO",
                    qk_zero_overall["raw_rmse_ratio"],
                ),
                (
                    "QK_body_mae_max_ratio_ZERO",
                    qk_zero_body["mae_ratio"],
                ),
                (
                    "QK_legacy_time_max_increase_ZERO",
                    qk_zero_overall[
                        "legacy_time_candidate_minus_reference"
                    ],
                ),
            )
            for name, observed in gate_values:
                criteria.append(
                    _criterion(
                        dataset=dataset,
                        fold=fold_id,
                        name=name,
                        observed=observed,
                        threshold=float(thresholds[name]),
                        operator="<=",
                    )
                )
            if enough_tail:
                qk_full_tail = compare_metric_summaries(
                    tail["QK"], tail["FULL"]
                )
                qk_zero_tail = compare_metric_summaries(
                    tail["QK"], tail["ZERO"]
                )
                for name, observed in (
                    (
                        "QK_tail_mae_max_ratio_FULL",
                        qk_full_tail["mae_ratio"],
                    ),
                    (
                        "QK_tail_mae_max_ratio_ZERO",
                        qk_zero_tail["mae_ratio"],
                    ),
                ):
                    criteria.append(
                        _criterion(
                            dataset=dataset,
                            fold=fold_id,
                            name=name,
                            observed=observed,
                            threshold=float(thresholds[name]),
                            operator="<=",
                        )
                    )

            B_time = overall["B"]["legacy_time_loss"]
            FULL_time = overall["FULL"]["legacy_time_loss"]
            QK_time = overall["QK"]["legacy_time_loss"]
            full_floor = overall["FULL"].get(
                "theoretical_wd10_clamp_floor_mean"
            )
            b_floor = overall["B"].get(
                "theoretical_wd10_clamp_floor_mean"
            )
            cross_b_diagnostic_valid = not (
                overall["FULL"].get("all_positive_wd_saturated_at_10")
                and overall["B"].get("all_positive_wd_saturated_at_10")
                and full_floor is not None
                and b_floor is not None
                and not math.isclose(
                    float(full_floor), float(b_floor), rel_tol=0.0, abs_tol=1e-12
                )
            )
            if not cross_b_diagnostic_valid:
                invalid_cross_b_time_folds.add((dataset, fold_id))
            excess = FULL_time - B_time
            recovery_policy = policy["time_recovery_when_FULL_minus_B_exceeds"]
            triggered = excess > float(recovery_policy["trigger"])
            if triggered:
                absolute_recovery = FULL_time - QK_time
                fractional_recovery = absolute_recovery / excess
                time_triggered_folds.append(
                    {
                        "dataset": dataset,
                        "fold": fold_id,
                        "FULL_minus_B": excess,
                        "FULL_minus_QK": absolute_recovery,
                        "fraction_of_positive_excess_removed": fractional_recovery,
                        "cross_B_trigger_diagnostic_valid": cross_b_diagnostic_valid,
                        "gate_validity_warning": (
                            None
                            if cross_b_diagnostic_valid
                            else "B and FULL are both fully wd-clamped at 10 but have "
                            "different slope-implied clamp floors; their absolute time "
                            "gap cannot diagnose representation quality."
                        ),
                    }
                )
                recovery_rows = (
                    _criterion(
                            dataset=dataset,
                            fold=fold_id,
                            name="minimum_absolute_improvement_FULL_to_QK",
                            observed=absolute_recovery,
                            threshold=float(
                                recovery_policy[
                                    "minimum_absolute_improvement_FULL_to_QK"
                                ]
                            ),
                            operator=">=",
                    ),
                    _criterion(
                            dataset=dataset,
                            fold=fold_id,
                            name=(
                                "minimum_fraction_of_positive_excess_removed"
                            ),
                            observed=fractional_recovery,
                            threshold=float(
                                recovery_policy[
                                    "minimum_fraction_of_positive_excess_removed"
                                ]
                            ),
                            operator=">=",
                    ),
                )
                for row in recovery_rows:
                    row["diagnostic_valid"] = cross_b_diagnostic_valid
                    criteria.append(row)

            improvement = _safe_relative_improvement(
                overall["ZERO"]["raw_rmse"], overall["QK"]["raw_rmse"]
            )
            minimum = float(
                policy["quantity_contribution"]["minimum_fraction_improvement"]
            )
            contribution = _criterion(
                dataset=dataset,
                fold=fold_id,
                name="QK_raw_rmse_improvement_over_ZERO",
                observed=improvement,
                threshold=minimum,
                operator=">=",
            )
            contribution["gating"] = False
            criteria.append(contribution)
            fold_contributions.append(contribution["passed"] is True)
        if all(fold_contributions):
            contributing_datasets.append(dataset)

    minimum_datasets = int(
        policy["quantity_contribution"][
            "minimum_datasets_with_both_folds_QK_raw_rmse_improved_over_ZERO"
        ]
    )
    dataset_contribution_gate = {
        "criterion": "minimum_datasets_with_both_folds_QK_contribution",
        "observed": len(contributing_datasets),
        "operator": ">=",
        "threshold": minimum_datasets,
        "evaluable": True,
        "passed": len(contributing_datasets) >= minimum_datasets,
        "datasets": contributing_datasets,
    }
    criteria.append(dataset_contribution_gate)

    numerical_failures = [
        row
        for row in criteria
        if row["evaluable"]
        and row["passed"] is False
        and row.get("gating", True)
    ]
    meaningful_failures = [
        row
        for row in numerical_failures
        if not (
            row.get("criterion")
            in {
                "minimum_absolute_improvement_FULL_to_QK",
                "minimum_fraction_of_positive_excess_removed",
            }
            and (row.get("dataset"), row.get("fold"))
            in invalid_cross_b_time_folds
        )
    ]
    unevaluable = [
        row
        for row in criteria
        if not row["evaluable"] and row.get("gating", True)
    ]
    if numerical_failures:
        raw_contractual_outcome_type = "failed"
    elif support_limitations or unevaluable or not time_triggered_folds:
        raw_contractual_outcome_type = "inconclusive"
    else:
        raw_contractual_outcome_type = "supported"
    if meaningful_failures:
        outcome_type = "failed"
        classification = (
            "current_counterfactual_does_not_support_QK_candidate_selection"
        )
    elif support_limitations:
        outcome_type = "inconclusive"
        classification = "inconclusive_no_automatic_candidate_selection"
    elif unevaluable:
        outcome_type = "inconclusive"
        classification = "inconclusive_insufficient_numerical_support"
    elif not time_triggered_folds and invalid_cross_b_time_folds:
        outcome_type = "inconclusive"
        classification = "inconclusive_cross_B_time_gap_is_clamp_floor_confounded"
    elif not time_triggered_folds:
        outcome_type = "inconclusive"
        classification = (
            "inconclusive_time_problem_not_reproduced_in_train_sample"
        )
    elif all(
        (row["dataset"], row["fold"]) in invalid_cross_b_time_folds
        for row in time_triggered_folds
    ):
        outcome_type = "inconclusive"
        classification = "inconclusive_cross_B_time_gap_is_clamp_floor_confounded"
    else:
        outcome_type = "supported"
        classification = "supports_drafting_QK_training_contract"
    return {
        "primary_comparison": "QK_vs_FULL",
        "outcome_type": outcome_type,
        "classification": classification,
        "raw_contractual_outcome_type": raw_contractual_outcome_type,
        "criteria": criteria,
        "numerical_failures": numerical_failures,
        "meaningful_numerical_failures": meaningful_failures,
        "unevaluable_criteria": unevaluable,
        "support_limitations": support_limitations,
        "time_triggered_folds": time_triggered_folds,
        "gate_validity": {
            "cross_B_time_trigger_is_diagnostic": not bool(
                invalid_cross_b_time_folds
            ),
            "confounded_dataset_folds": [
                {"dataset": dataset, "fold": fold}
                for dataset, fold in sorted(invalid_cross_b_time_folds)
            ],
            "warning": (
                None
                if not invalid_cross_b_time_folds
                else "Cross-B time trigger/recovery compares different analytic "
                "slope floors under complete wd clamping. Retain the frozen raw "
                "gate flags, but do not use those folds to infer representation "
                "quality or architecture selection. Within-checkpoint QK/FULL/ZERO "
                "time comparisons remain identified because their slope is shared."
            ),
        },
        "failed_counterfactual_does_not_prove_QK_training_impossible": True,
        "counterfactual_is_retrained_model": False,
        "new_candidate_training_authorized": False,
        "B_comparison_is_descriptive_except_fixed_time_trigger": True,
        "alternate_mask_winner_selected": False,
    }


def summarize_gradient_records(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    gradient_contract = contract["gradient_diagnostic"]
    expected_batches = int(gradient_contract["batches_per_fold"])
    expected_keys = {
        (variant, fold, batch, group)
        for variant in GRADIENT_VARIANTS
        for fold in (0, 1)
        for batch in range(expected_batches)
        for group in GRADIENT_GROUPS
    }
    observed: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for record in records:
        require(isinstance(record, Mapping), "Gradient record must be an object")
        key = (
            str(record.get("variant")),
            int(record.get("fold", -1)),
            int(record.get("batch_index", -1)),
            str(record.get("group")),
        )
        require(key in expected_keys, f"Unexpected gradient record identity: {key}")
        require(key not in observed, f"Duplicate gradient record identity: {key}")
        for field in GRADIENT_NUMERIC_FIELDS:
            require(
                _is_finite_number(record.get(field)),
                f"Gradient record {key} has invalid {field}",
            )
        require(
            float(record["time_norm"]) >= 0.0
            and float(record["quantity_norm"]) >= 0.0,
            f"Gradient record {key} has a negative norm",
        )
        require(
            isinstance(record.get("unused_or_zero_gradient"), bool),
            f"Gradient record {key} has invalid zero-gradient flag",
        )
        unused = bool(record["unused_or_zero_gradient"])
        cosine = record.get("cosine")
        if unused:
            require(cosine is None, f"Gradient record {key} must null cosine")
        else:
            require(
                _is_finite_number(cosine) and -1.0 <= float(cosine) <= 1.0,
                f"Gradient record {key} has invalid cosine",
            )
        observed[key] = record
    require(set(observed) == expected_keys, "Gradient record grid is incomplete")

    summaries: dict[str, Any] = {}
    for variant in GRADIENT_VARIANTS:
        summaries[variant] = {}
        for group in GRADIENT_GROUPS:
            selected = [
                record
                for (current_variant, _, _, current_group), record in observed.items()
                if current_variant == variant and current_group == group
            ]
            cosine_values = [
                float(row["cosine"])
                for row in selected
                if row["cosine"] is not None
            ]
            fold_summaries: dict[str, Any] = {}
            for fold in (0, 1):
                fold_rows = [
                    row for row in selected if int(row["fold"]) == fold
                ]
                fold_cosines = [
                    float(row["cosine"])
                    for row in fold_rows
                    if row["cosine"] is not None
                ]
                fold_summaries[str(fold)] = {
                    **{
                        f"{field}_mean": float(
                            np.mean([float(row[field]) for row in fold_rows])
                        )
                        for field in GRADIENT_NUMERIC_FIELDS
                    },
                    "cosine_valid_count": len(fold_cosines),
                    "cosine_mean": (
                        float(np.mean(fold_cosines)) if fold_cosines else None
                    ),
                }
            summaries[variant][group] = {
                "record_count": len(selected),
                **{
                    f"{field}_mean": float(
                        np.mean([float(row[field]) for row in selected])
                    )
                    for field in GRADIENT_NUMERIC_FIELDS
                },
                "cosine_valid_count": len(cosine_values),
                "cosine_mean": (
                    float(np.mean(cosine_values)) if cosine_values else None
                ),
                "unused_or_zero_gradient_fraction": float(
                    np.mean(
                        [bool(row["unused_or_zero_gradient"]) for row in selected]
                    )
                ),
                "folds": fold_summaries,
            }
    return {
        "descriptive_only": True,
        "record_count": len(records),
        "variants": summaries,
    }


def validate_checks(
    checks: Mapping[str, Any],
    *,
    dataset_spec: Mapping[str, Any],
    contract: Mapping[str, Any],
    contract_sha256: str,
    dataset_dir: Path,
    extraction_path: Path,
    arrays: Mapping[str, np.ndarray],
) -> tuple[list[float], list[Mapping[str, Any]]]:
    dataset = str(dataset_spec["dataset"])
    require(checks.get("schema_version") == 1, "Checks schema drift")
    require(
        checks.get("contract_sha256") == contract_sha256,
        "Checks contract digest drift",
    )
    require(checks.get("dataset") == dataset, "Checks dataset drift")
    require(checks.get("status") == "passed", "Extraction checks failed")
    require(
        checks.get("npz_sha256") == sha256_file(extraction_path),
        "Extraction NPZ digest drift",
    )
    require(
        checks.get("npz_keys") == sorted(expected_npz_keys()),
        "Recorded NPZ schema drift",
    )
    require(
        checks.get("train_targets") == dataset_spec["expected_train_targets"]
        and checks.get("sample_targets") == contract["sampling"]["targets_per_dataset"],
        "Train/sample target count drift",
    )
    require(
        checks.get("validation_materialized") is False
        and checks.get("held_out_materialized") is False
        and checks.get("model_or_optimizer_updates") is False,
        "Extraction scope violated the train-only frozen contract",
    )
    require(checks.get("parameter_grad_cleanup") is True, "Gradient cleanup failed")
    source = checks.get("source")
    require(isinstance(source, Mapping), "Frozen source receipt is missing")
    source_keys = {
        "frozen_implementation_commit": "frozen_implementation_commit",
        "archive_sha256": "archive_sha256",
        "manifest_sha256": "pinned_manifest_sha256",
        "verified_python_count": "pinned_python_count",
    }
    for receipt_key, contract_key in source_keys.items():
        require(
            source.get(receipt_key) == contract["source"].get(contract_key),
            f"Source drift: {receipt_key}",
        )
    for key in ("runner_file_sha256", "diagnostic_source_revision", "candidate_common_state_sha256"):
        value = checks.get(key)
        require(
            isinstance(value, str)
            and len(value) in ({40} if key == "diagnostic_source_revision" else {64})
            and all(character in "0123456789abcdef" for character in value.lower()),
            f"Invalid checks digest/revision: {key}",
        )
    before_files = checks.get("checkpoint_file_hashes_before")
    after_files = checks.get("checkpoint_file_hashes_after")
    require(before_files == after_files, "Checkpoint file changed during extraction")
    for role in ("B", "candidate"):
        require(
            isinstance(before_files, Mapping)
            and before_files.get(role)
            == dataset_spec[role]["checkpoint_file_sha256"],
            f"{role} checkpoint receipt drift",
        )
    before_states = checks.get("model_state_hashes_before")
    require(
        isinstance(before_states, Mapping)
        and before_states == checks.get("model_state_hashes_after"),
        "Model state changed during extraction",
    )
    evidence = checks.get("checkpoint_evidence")
    require(isinstance(evidence, Mapping), "Checkpoint evidence is missing")
    for role in ("B", "candidate"):
        row = evidence.get(role)
        require(isinstance(row, Mapping), f"Missing {role} checkpoint evidence")
        require(
            row.get("checkpoint_file_sha256") == before_files[role]
            and row.get("summary_sha256") == dataset_spec[role]["summary_sha256"]
            and row.get("checkpoint_state_sha256") == before_states[role],
            f"{role} checkpoint evidence drift",
        )
    parity = checks.get("official_target_outputs_first_fixed_batch")
    require(isinstance(parity, Mapping) and set(parity) == {"B", "FULL"}, "Parity receipt drift")
    for role, fields in parity.items():
        require(isinstance(fields, Mapping), f"Invalid {role} parity receipt")
        for name in ("prediction", "log_mse", "legacy_time_loss"):
            require(_is_finite_number(fields.get(name)) and fields[name] >= 0.0, f"Invalid parity value: {role}.{name}")
    leakage = checks.get("target_duration_quantity_padding_invariance")
    require(isinstance(leakage, Mapping) and set(leakage) == set(VARIANTS), "Leakage receipt drift")
    for variant, fields in leakage.items():
        require(
            isinstance(fields, Mapping)
            and set(fields)
            == set(STATE_FIELDS) | {TOP4_FIELD, "prediction", "time_intercept_raw", "time_intercept_clamped"}
            and all(value == 0.0 for value in fields.values()),
            f"Target/padding leakage check failed: {variant}",
        )
    mask_checks = checks.get("variant_kernel_mask_checks")
    require(isinstance(mask_checks, Mapping) and set(mask_checks) == set(VARIANTS[1:]), "Mask receipt drift")
    for variant in VARIANTS[1:]:
        expected_switches = dict(zip(("Q", "K", "V"), contract["variants"][variant]))
        require(set(mask_checks[variant]) == {"Q", "K", "V"}, f"Mask fields drift: {variant}")
        for name, enabled in expected_switches.items():
            row = mask_checks[variant][name]
            require(
                isinstance(row, Mapping)
                and row.get("enabled") is bool(enabled)
                and isinstance(row.get("all_three_rows_zero"), bool)
                and (enabled or row["all_three_rows_zero"]),
                f"Mask check failed: {variant}.{name}",
            )

    manifest_path = dataset_dir / "sample_ids.json"
    manifest = load_json(manifest_path)
    require(
        checks.get("sample_ids_sha256") == sha256_file(manifest_path)
        and manifest.get("schema_version") == 1
        and manifest.get("status") == "sample_ids_frozen_before_model_outputs"
        and manifest.get("dataset") == dataset
        and manifest.get("contract_sha256") == contract_sha256
        and manifest.get("selection_seed") == contract["sampling"]["seed"],
        "Sample identity receipt drift",
    )
    rows = manifest.get("rows")
    require(isinstance(rows, list) and len(rows) == len(arrays["fold"]), "Sample identity row count drift")
    identity_bytes = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(identity_bytes).hexdigest() == manifest.get("identity_sha256"), "Sample identity digest drift")
    for index, row in enumerate(rows):
        require(
            isinstance(row, Mapping)
            and int(row.get("dataset_index", -1)) == int(arrays["dataset_index"][index])
            and canonical_series_id(row.get("series_id")) == canonical_series_id(arrays["series_id"][index])
            and int(row.get("fold", -1)) == int(arrays["fold"][index])
            and fold_for_series(row.get("series_id")) == int(row.get("fold", -1)),
            f"Sample identity alignment drift at row {index}",
        )
    for fold in (0, 1):
        selected = arrays["fold"] == fold
        require(
            manifest.get("fold_counts", {}).get(str(fold)) == int(selected.sum())
            and manifest.get("fold_unique_series", {}).get(str(fold))
            == int(np.unique(arrays["series_id"][selected]).size),
            f"Sample manifest fold summary drift: fold{fold}",
        )
    boundaries = checks.get("duration_train_quartile_boundaries")
    require(
        isinstance(boundaries, list) and len(boundaries) == 3,
        "Duration quartile boundaries are missing",
    )
    validated = [float(value) for value in boundaries]
    _duration_masks(np.ones(1, dtype=np.float64), validated)
    records = checks.get("gradient_records")
    require(isinstance(records, list), "Gradient records are missing")
    return validated, records


def render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Hard-LMM Q/K/V Path Diagnostic",
        "",
        "This train-only fixed-checkpoint counterfactual diagnoses Q/K/V paths. "
        "It does not prove that a retrained QK model will or will not work.",
        "",
        "## Prospective QK gate",
        "",
        f"- Outcome: **{result['decision']['outcome_type']}**",
        f"- Classification: `{result['decision']['classification']}`",
        f"- Time-triggered folds: {len(result['decision']['time_triggered_folds'])}",
        f"- Raw contractual numerical failures: {len(result['decision']['numerical_failures'])}",
        f"- Diagnostically meaningful numerical failures: {len(result['decision']['meaningful_numerical_failures'])}",
        f"- Support limitations: {len(result['decision']['support_limitations'])}",
        "",
        "## Pooled metrics",
        "",
        "| Dataset | Variant | Raw RMSE | MAE | Log MSE | Legacy time loss | w | wd saturated | Clamp floor | Excess over floor |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, analysis in result["datasets"].items():
        variants = analysis["aggregates"]["pooled"]["variants"]
        for variant in VARIANTS:
            row = variants[variant]
            lines.append(
                "| "
                + " | ".join(
                    (
                        dataset,
                        variant,
                        _format_number(row["raw_rmse"]),
                        _format_number(row["mae"]),
                        _format_number(row["log_mse"]),
                        _format_number(row["legacy_time_loss"]),
                        _format_number(row["w_mean"]),
                        _format_number(row["wd_saturated_fraction"]),
                        _format_number(row["theoretical_wd10_clamp_floor_mean"]),
                        _format_number(row["legacy_time_loss_excess_over_wd10_floor"]),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## Fold gate audit",
            "",
            "| Dataset | Fold | Criterion | Observed | Rule | Passed |",
            "|---|---:|---|---:|---:|---:|",
        )
    )
    for row in result["decision"]["criteria"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row.get("dataset", "all")),
                    str(row.get("fold", "all")),
                    str(row["criterion"]),
                    _format_number(row["observed"]),
                    f"{row['operator']} {_format_number(row['threshold'])}",
                    str(row["passed"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Interpretation limits",
            "",
            "- Raw contractual gate flags are retained separately from the interpreted outcome.",
            "- B metric comparisons are descriptive; hidden coordinates and prototype row IDs are not comparable across the independently trained checkpoints.",
            (
                "- Gate-validity warning: "
                + str(result["decision"]["gate_validity"]["warning"])
                if result["decision"]["gate_validity"]["warning"] is not None
                else "- Cross-B time-trigger comparisons were not clamp-floor confounded."
            ),
            "- No alternate mask is selected from QV, KV, Q, K, or V results.",
            "- Failure of this off-manifold counterfactual does not prove QK retraining is impossible.",
            "- No validation or held-out test rows are analyzed.",
            "",
        )
    )
    return "\n".join(lines)


def _format_number(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6g}"


def run_analysis(
    *,
    contract_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    contract_sha256 = sha256_file(contract_path)
    require(
        contract_sha256 == CONTRACT_SHA256,
        "Frozen diagnostic contract digest drift",
    )
    contract = load_json(contract_path)
    validate_contract(contract)
    analyses: dict[str, Any] = {}
    input_digests: dict[str, Any] = {}
    gradient_summaries: dict[str, Any] = {}
    for dataset_spec in contract["datasets"]:
        dataset = dataset_spec["dataset"]
        dataset_dir = input_dir / dataset
        extraction_path = dataset_dir / "extraction.npz"
        checks_path = dataset_dir / "checks.json"
        require(extraction_path.is_file(), f"Missing extraction: {extraction_path}")
        checks = load_json(checks_path)
        with np.load(extraction_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        duration_boundaries, gradient_records = validate_checks(
            checks,
            dataset_spec=dataset_spec,
            contract=contract,
            contract_sha256=contract_sha256,
            dataset_dir=dataset_dir,
            extraction_path=extraction_path,
            arrays=arrays,
        )
        analyses[dataset] = analyze_dataset_arrays(
            arrays,
            dataset_spec=dataset_spec,
            contract=contract,
            duration_quartile_boundaries=duration_boundaries,
        )
        gradient_summaries[dataset] = summarize_gradient_records(
            gradient_records, contract
        )
        input_digests[dataset] = {
            "extraction_npz_sha256": sha256_file(extraction_path),
            "checks_json_sha256": sha256_file(checks_path),
            "sample_ids_json_sha256": sha256_file(dataset_dir / "sample_ids.json"),
        }
    decision = evaluate_prospective_qk_gate(analyses, contract)
    result = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "analysis_engine": "pure_numpy_no_model_imports",
        "evaluation_scope": "train_only_fixed_checkpoint_counterfactual",
        "validation_materialized": False,
        "held_out_materialized": False,
        "model_or_optimizer_updates": False,
        "datasets": analyses,
        "gradient_diagnostic": gradient_summaries,
        "decision": decision,
        "input_digests": input_digests,
        "interpretation": {
            "diagnostic_not_training": True,
            "generalization_claim": False,
            "failed_counterfactual_does_not_prove_QK_training_impossible": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "analysis.json", result)
    markdown_path = output_dir / "report.md"
    temporary_markdown = markdown_path.with_suffix(".md.tmp")
    temporary_markdown.write_text(render_markdown(result), encoding="utf-8")
    temporary_markdown.replace(markdown_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_analysis(
        contract_path=args.contract,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["decision"], sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
