"""Paired, raw-quantity diagnostics for the isolated J/Q comparison.

The module deliberately consumes already-aligned arrays.  It neither loads an
artifact nor changes bins, so callers retain ownership of the evaluation split.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


_METRIC_KEYS = ("rmse", "mae", "bias", "log_mse", "under_proportion", "over_proportion", "exact_tie_proportion", "centered_mse")
_DELTA_KEYS = ("mse", "mae", "log_mse")


def _array(name: str, value: Any, count: int | None = None, *, integer: bool = False) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1d array")
    if count is not None and array.size != count:
        raise ValueError(f"{name} must have the same length as truth")
    if integer:
        if not np.issubdtype(array.dtype, np.integer) or np.any(array <= 0):
            raise ValueError(f"{name} must contain positive integers")
        return array.astype(np.int64, copy=False)
    try:
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    if np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _boundaries(name: str, values: Any, *, exact_count: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or (exact_count is not None and array.size != exact_count):
        suffix = f" exactly {exact_count}" if exact_count is not None else ""
        raise ValueError(f"{name} must be a nonempty 1d array with{suffix} boundaries")
    if not np.all(np.isfinite(array)) or np.any(np.diff(array) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return array


def _empty_metrics() -> dict[str, None]:
    return {key: None for key in _METRIC_KEYS}


def _metrics(error: np.ndarray, log_loss: np.ndarray) -> dict[str, float | None]:
    if error.size == 0:
        return _empty_metrics()
    bias = float(np.mean(error, dtype=np.float64))
    return {
        "rmse": float(math.sqrt(np.mean(error * error, dtype=np.float64))),
        "mae": float(np.mean(np.abs(error), dtype=np.float64)),
        "bias": bias,
        "log_mse": float(np.mean(log_loss, dtype=np.float64)),
        "under_proportion": float(np.mean(error < 0)),
        "over_proportion": float(np.mean(error > 0)),
        "exact_tie_proportion": float(np.mean(error == 0)),
        "centered_mse": float(np.mean((error - bias) ** 2, dtype=np.float64)),
    }


def _comparison(error_j: np.ndarray, error_q: np.ndarray, log_j: np.ndarray, log_q: np.ndarray, prediction_shift: np.ndarray) -> dict[str, Any]:
    if error_j.size == 0:
        return {"delta": {key: None for key in _DELTA_KEYS}, "weighted_contribution": {key: None for key in _DELTA_KEYS}, "paired_sse_decomposition": {"cross_term": None, "shift_squared": None, "total_delta_mse": None}}
    delta = {
        "mse": float(np.mean(error_q ** 2, dtype=np.float64) - np.mean(error_j ** 2, dtype=np.float64)),
        "mae": float(np.mean(np.abs(error_q), dtype=np.float64) - np.mean(np.abs(error_j), dtype=np.float64)),
        "log_mse": float(np.mean(log_q, dtype=np.float64) - np.mean(log_j, dtype=np.float64)),
    }
    cross = float(np.mean(2.0 * error_j * prediction_shift, dtype=np.float64))
    shift_squared = float(np.mean(prediction_shift ** 2, dtype=np.float64))
    return {"delta": delta, "weighted_contribution": {key: None for key in _DELTA_KEYS}, "paired_sse_decomposition": {"cross_term": cross, "shift_squared": shift_squared, "total_delta_mse": float(cross + shift_squared)}}


def _cell(mask: np.ndarray, series: np.ndarray, error_j: np.ndarray, error_q: np.ndarray, log_j: np.ndarray, log_q: np.ndarray, shift: np.ndarray, total: int) -> dict[str, Any]:
    n = int(np.count_nonzero(mask))
    result = {
        "n": n,
        "series_count": int(np.unique(series[mask]).size),
        "j": _metrics(error_j[mask], log_j[mask]),
        "q": _metrics(error_q[mask], log_q[mask]),
    }
    result.update(_comparison(error_j[mask], error_q[mask], log_j[mask], log_q[mask], shift[mask]))
    if n:
        result["weighted_contribution"] = {key: float(n / total * result["delta"][key]) for key in _DELTA_KEYS}
    return result


def _audit(name: str, cells: list[dict[str, Any]], overall: dict[str, Any]) -> dict[str, Any]:
    n = sum(cell["n"] for cell in cells)
    totals = {key: sum((cell["weighted_contribution"][key] or 0.0) for cell in cells) for key in _DELTA_KEYS}
    decomp = {key: sum((cell["n"] / overall["n"] * (cell["paired_sse_decomposition"][key] or 0.0)) for cell in cells) for key in ("cross_term", "shift_squared", "total_delta_mse")}
    def close(left: float, right: float) -> bool:
        return bool(np.isclose(left, right, atol=1e-10, rtol=1e-10))
    checks = {"count": n == overall["n"]}
    checks.update({f"delta_{key}": close(totals[key], overall["delta"][key]) for key in _DELTA_KEYS})
    checks.update({f"decomposition_{key}": close(decomp[key], overall["paired_sse_decomposition"][key]) for key in decomp})
    if not all(checks.values()):
        raise AssertionError(f"{name} partition reconstruction failed: {checks}")
    return {"passed": True, "checks": checks}


def _series_concentration(series: np.ndarray, error_j: np.ndarray, error_q: np.ndarray) -> dict[str, Any]:
    values, inverse = np.unique(series, return_inverse=True)
    delta_sse = np.bincount(
        inverse,
        weights=(error_q ** 2 - error_j ** 2),
        minlength=values.size,
    )
    rows = [(str(value), float(delta)) for value, delta in zip(values.tolist(), delta_sse.tolist(), strict=True)]
    worsening = sorted((row for row in rows if row[1] > 0), key=lambda row: (-row[1], row[0]))
    improving = sorted((row for row in rows if row[1] < 0), key=lambda row: (row[1], row[0]))
    positive_gross = float(sum(delta for _, delta in worsening))
    all_count = len(rows)
    def share(percent: float) -> dict[str, float | int]:
        take = int(math.ceil(percent * all_count))
        captured = float(sum(delta for _, delta in worsening[:take]))
        return {"percent": percent, "denominator": "all_series_count", "series_count": take, "captured_positive_sse": captured, "share_of_positive_gross": 0.0 if positive_gross == 0 else float(captured / positive_gross)}
    return {
        "all_series_count": all_count,
        "numbers": {"improve": len(improving), "worsen": len(worsening), "tie": sum(delta == 0 for _, delta in rows)},
        "net_delta_sse": float(sum(delta for _, delta in rows)),
        "positive_gross_delta_sse": positive_gross,
        "negative_gross_delta_sse": float(sum(delta for _, delta in improving)),
        "top_contribution_to_positive_gross": {"top_1_percent": share(0.01), "top_5_percent": share(0.05)},
        "top_10_worsening_ids": [value for value, _ in worsening[:10]],
        "top_10_improving_ids": [value for value, _ in improving[:10]],
    }


def analyze_quantity_pair(dataset_id: str, truth: Any, pred_j: Any, pred_q: Any, log_loss_j: Any, log_loss_q: Any, history_length: Any, series_ids: Any, quantity_boundaries: Any, history_boundaries: Any) -> dict[str, Any]:
    """Return JSON-serializable paired diagnostics using fixed caller-provided bins."""
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a nonempty string")
    truth = _array("truth", truth)
    n = truth.size
    pred_j, pred_q = _array("pred_j", pred_j, n), _array("pred_q", pred_q, n)
    log_j, log_q = _array("log_loss_j", log_loss_j, n), _array("log_loss_q", log_loss_q, n)
    history = _array("history_length", history_length, n, integer=True)
    series = np.asarray(series_ids, dtype=object)
    if series.ndim != 1 or series.size != n or not all(isinstance(value, str) for value in series.tolist()):
        raise ValueError("series_ids must be same-length strings")
    quantity_bounds = _boundaries("quantity_boundaries", quantity_boundaries, exact_count=4)
    history_bounds = _boundaries("history_boundaries", history_boundaries)
    error_j, error_q, shift = pred_j - truth, pred_q - truth, pred_q - pred_j
    quantity_index = np.searchsorted(quantity_bounds, truth, side="left")
    history_index = np.searchsorted(history_bounds, history, side="left")
    overall = _cell(np.ones(n, dtype=bool), series, error_j, error_q, log_j, log_q, shift, n)
    quantity_cells = [_cell(quantity_index == index, series, error_j, error_q, log_j, log_q, shift, n) | {"bin_index": index} for index in range(quantity_bounds.size + 1)]
    history_cells = [_cell(history_index == index, series, error_j, error_q, log_j, log_q, shift, n) | {"bin_index": index} for index in range(history_bounds.size + 1)]
    cross_cells = [_cell((quantity_index == qi) & (history_index == hi), series, error_j, error_q, log_j, log_q, shift, n) | {"quantity_bin_index": qi, "history_bin_index": hi} for qi in range(quantity_bounds.size + 1) for hi in range(history_bounds.size + 1)]
    result = {"dataset_id": dataset_id, "n": n, "binning": {"quantity_boundaries": quantity_bounds.tolist(), "history_boundaries": history_bounds.tolist(), "searchsorted_side": "left"}, "overall": overall, "quantity_cells": quantity_cells, "history_cells": history_cells, "cross_cells": cross_cells, "series_concentration": _series_concentration(series, error_j, error_q)}
    result["audit"] = {"quantity": _audit("quantity", quantity_cells, overall), "history": _audit("history", history_cells, overall), "cross": _audit("cross", cross_cells, overall)}
    return result


def verify_quantity_replay(actual_j: Mapping[str, Any], actual_q: Mapping[str, Any], expected_j: Mapping[str, Any], expected_q: Mapping[str, Any]) -> dict[str, Any]:
    """Verify replay metrics without allowing tiny per-arm drift to hide a delta change."""
    fields = ("raw_quantity_rmse", "quantity_mae", "quantity_train_loss")
    checks: dict[str, Any] = {}
    for field in fields:
        try:
            actual_left, actual_right = float(actual_j[field]), float(actual_q[field])
            expected_left, expected_right = float(expected_j[field]), float(expected_q[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"missing or invalid replay metric: {field}") from error
        if not all(math.isfinite(value) for value in (actual_left, actual_right, expected_left, expected_right)):
            raise ValueError(f"nonfinite replay metric: {field}")
        per_arm = math.isclose(actual_left, expected_left, rel_tol=1e-5, abs_tol=1e-5) and math.isclose(actual_right, expected_right, rel_tol=1e-5, abs_tol=1e-5)
        original_delta, replay_delta = expected_right - expected_left, actual_right - actual_left
        delta_contract = field in {"raw_quantity_rmse", "quantity_mae"}
        sign = ((original_delta > 0) == (replay_delta > 0) and (original_delta < 0) == (replay_delta < 0)) if original_delta != 0 else replay_delta == 0
        distortion = abs(replay_delta - original_delta)
        delta_ok = (sign and distortion <= 0.001 * abs(original_delta)) if original_delta != 0 else sign
        checks[field] = {"per_arm_isclose": per_arm, "delta_contract_applied": delta_contract, "delta_sign_preserved": sign if delta_contract else None, "delta_distortion": distortion if delta_contract else None, "delta_tolerance": 0.001 * abs(original_delta) if delta_contract else None, "passed": per_arm and (delta_ok if delta_contract else True)}
        if not checks[field]["passed"]:
            raise ValueError(f"quantity replay mismatch for {field}: {checks[field]}")
    return {"passed": True, "checks": checks}
