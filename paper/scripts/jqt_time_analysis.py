"""Read-only, per-sample diagnostics for the frozen legacy RMTPP time head.

The helper intentionally reproduces the causal target path used by
``time_quantity_diagnostic.task_outputs``.  It is suitable for synthetic or
validation-only batches; it does not load checkpoints, invoke a quantity head,
or alter model state.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from paper.scripts.count_aware_tpp_backbone.core import right_pad_batch


_LEGACY_MODE = "legacy_clamped_rmtpp"
_LOG_MSE_VARIANT = "count_only_log_regression"
_INTERCEPT_LIMIT = 300.0
_WD_LIMIT = 10.0
_DURATION_UPPERS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all()), f"Nonfinite {name}")


def _validate(model: torch.nn.Module, dts: torch.Tensor, mask: torch.Tensor,
              quantities: torch.Tensor, objective: str) -> None:
    _require(objective in {"joint", "time_only"}, "Only joint and time_only are supported")
    _require(not model.training, "inspect_legacy_time_batch requires model.eval()")
    _require(getattr(model, "quantity_variant", None) == _LOG_MSE_VARIANT,
             "Diagnostic requires count_only_log_regression")
    _require(getattr(model, "time_head_mode", None) == _LEGACY_MODE,
             "Diagnostic requires legacy_clamped_rmtpp")
    _require(getattr(model, "time_intercept_limit", None) == _INTERCEPT_LIMIT,
             "Diagnostic requires the frozen intercept cap of 300")
    _require(mask.dtype == torch.bool and dts.ndim == 2 and dts.shape == mask.shape == quantities.shape,
             "Expected aligned two-dimensional dts, quantities, and bool mask")
    _require(dts.shape[0] > 0, "Empty batch")
    _finite_tensor(dts, "durations")
    _finite_tensor(quantities, "quantities")
    _require(bool((dts >= 0).all()), "Durations must be nonnegative")
    _require(bool((quantities >= 0).all()), "Quantities must be nonnegative")


@torch.no_grad()
def inspect_legacy_time_batch(
    model: torch.nn.Module,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    objective: str,
) -> dict[str, torch.Tensor]:
    """Return detached per-sample legacy-time terms from the baseline graph.

    ``original_legacy_loss`` always comes from ``model.log_f_dt`` so this
    inspection cannot silently substitute a numerically different objective.
    The explicit terms are a detached reconstruction for attribution only.
    """
    _validate(model, dts, mask, quantities, objective)
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    memory_write_mask = mask.clone()
    memory_write_mask[batch_ids, target_positions] = False
    time_encoded, _ = model.encode_task_states(
        dts, history_quantities, mask, memory_write_mask=memory_write_mask,
    )
    hidden = time_encoded[batch_ids, history_positions]
    duration = dts[batch_ids, target_positions].float()
    original_legacy_loss = -model.log_f_dt(hidden, duration)

    raw_intercept = model.v_t(hidden).squeeze(-1) + model.b_t
    capped_intercept = torch.clamp(raw_intercept, max=_INTERCEPT_LIMIT)
    w = F.softplus(model.w_raw) + 1e-3
    raw_w_dt = w * duration
    capped_w_dt = torch.clamp(raw_w_dt, max=_WD_LIMIT)
    term_linear = -capped_intercept - capped_w_dt
    term_integral = torch.exp(capped_intercept) / w * torch.expm1(capped_w_dt)
    reconstruction = term_linear + term_integral
    residual = original_legacy_loss - reconstruction

    results = {
        "original_legacy_loss": original_legacy_loss,
        "raw_intercept": raw_intercept,
        "capped_intercept": capped_intercept,
        "w": w.expand_as(duration),
        "w_dt": raw_w_dt,
        "raw_w_dt": raw_w_dt,
        "capped_w_dt": capped_w_dt,
        "term_linear": term_linear,
        "term_integral": term_integral,
        "duration": duration,
        "history_length": lengths - 1,
        "intercept_cap_hit": raw_intercept > _INTERCEPT_LIMIT,
        "intercept_cap_exact_boundary": raw_intercept == _INTERCEPT_LIMIT,
        "w_dt_cap_hit": raw_w_dt > _WD_LIMIT,
        "w_dt_cap_exact_boundary": raw_w_dt == _WD_LIMIT,
        "residual_reconstruction_error": residual,
    }
    for name, value in results.items():
        if value.is_floating_point():
            _finite_tensor(value, name)
    return {name: value.detach() for name, value in results.items()}


def _as_array(rows: Mapping[str, Any] | Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    if isinstance(rows, Mapping):
        value = rows[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value).reshape(-1)
    values = []
    for row in rows:
        value = row[key]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        values.extend(np.asarray(value).reshape(-1).tolist())
    return np.asarray(values)


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _optional_array(
    rows: Mapping[str, Any] | Sequence[Mapping[str, Any]], key: str,
) -> np.ndarray | None:
    if isinstance(rows, Mapping):
        return _as_array(rows, key) if key in rows else None
    if not rows or not all(key in row for row in rows):
        return None
    return _as_array(rows, key)


def _distribution(values: np.ndarray, name: str, n: int) -> dict[str, float]:
    values = values.astype(np.float64)
    _require(values.size == n and bool(np.isfinite(values).all()),
             f"Invalid {name} diagnostic rows")
    return {
        "mean": float(values.mean()), "min": float(values.min()),
        "max": float(values.max()), "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def summarize_time_rows(rows: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return JSON-safe aggregate diagnostics, retaining every duration bin.

    Each bin has ``mean_loss_contribution = loss_sum / global_n``; the bin
    contributions therefore sum exactly (up to JSON float representation) to
    the global signed mean loss.
    """
    loss = _as_array(rows, "original_legacy_loss").astype(np.float64)
    duration = _as_array(rows, "duration").astype(np.float64)
    hit = _as_array(rows, "intercept_cap_hit").astype(bool)
    boundary = _as_array(rows, "intercept_cap_exact_boundary").astype(bool)
    wd_hit = _as_array(rows, "w_dt_cap_hit").astype(bool)
    wd_boundary = _as_array(rows, "w_dt_cap_exact_boundary").astype(bool)
    w = _as_array(rows, "w").astype(np.float64)
    n = int(loss.size)
    _require(n > 0 and all(array.size == n for array in (duration, hit, boundary, wd_hit, wd_boundary, w)),
             "Rows must contain equally sized nonempty per-sample fields")
    _require(all(np.isfinite(array).all() for array in (loss, duration, w)), "Nonfinite diagnostic rows")
    _require(bool((duration >= 0).all()) and bool((w > 0).all()), "Invalid duration or w")

    positive_total = float(loss[loss > 0].sum())
    positive = np.sort(loss[loss > 0])[::-1]
    def top_share(fraction: float) -> float:
        if not positive.size or positive_total == 0.0:
            return 0.0
        return float(positive[:max(1, math.ceil(n * fraction))].sum() / positive_total)

    bins = []
    lower = 0.0
    for upper in _DURATION_UPPERS:
        selected = (duration >= lower) & (duration <= upper) if lower == 0.0 else (duration > lower) & (duration <= upper)
        count = int(selected.sum())
        loss_sum = float(loss[selected].sum())
        bins.append({"lower_exclusive": None if lower == 0.0 else lower, "upper_inclusive": upper,
                     "n": count, "loss_sum": loss_sum,
                     "mean_loss": float(loss[selected].mean()) if count else None,
                     "mean_loss_contribution": loss_sum / n})
        lower = upper
    selected = duration > lower
    count = int(selected.sum())
    loss_sum = float(loss[selected].sum())
    bins.append({"lower_exclusive": lower, "upper_inclusive": None, "n": count, "loss_sum": loss_sum,
                 "mean_loss": float(loss[selected].mean()) if count else None,
                 "mean_loss_contribution": loss_sum / n})
    term_statistics = {}
    for field in ("raw_intercept", "raw_w_dt", "term_linear", "term_integral",
                  "residual_reconstruction_error"):
        values = _optional_array(rows, field)
        if values is not None:
            term_statistics[field] = _distribution(values, field, n)
    residual = _optional_array(rows, "residual_reconstruction_error")
    result = {
        "n": n, "mean_loss": float(loss.mean()), "signed_loss_sum": float(loss.sum()),
        "positive_loss_sum": positive_total, "negative_loss_sum": float(loss[loss < 0].sum()),
        "intercept_cap_hit_count": int(hit.sum()), "intercept_cap_hit_rate": _rate(int(hit.sum()), n),
        "intercept_cap_exact_boundary_count": int(boundary.sum()), "intercept_cap_exact_boundary_rate": _rate(int(boundary.sum()), n),
        "w_dt_cap_hit_count": int(wd_hit.sum()), "w_dt_cap_hit_rate": _rate(int(wd_hit.sum()), n),
        "w_dt_cap_exact_boundary_count": int(wd_boundary.sum()), "w_dt_cap_exact_boundary_rate": _rate(int(wd_boundary.sum()), n),
        "w": {"min": float(w.min()), "max": float(w.max()), "shared_scalar": bool(np.all(w == w[0]))},
        "positive_loss_top_1pct_share": top_share(0.01), "positive_loss_top_5pct_share": top_share(0.05),
        "duration_bins": bins,
        "term_statistics": term_statistics,
    }
    if residual is not None:
        result["residual_reconstruction_error_max_abs"] = float(np.abs(residual).max())
    return result
