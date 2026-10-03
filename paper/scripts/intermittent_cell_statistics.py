"""Pure float64 accounting for frozen paired quantity/history validation cells.

This module reads no files, checkpoints, runtime state, or data sources. Cell
shares describe signed arithmetic contributions, never causal percentages.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np

SUM_FIELDS = (
    "count", "true_quantity_sum", "predicted_quantity_sum", "signed_error_sum",
    "absolute_error_sum", "raw_squared_error_sum", "under_count", "over_count", "tie_count",
    "under_absolute_error_sum", "over_absolute_error_sum",
)
COUNT_FIELDS = {"count", "under_count", "over_count", "tie_count"}
PAIR_FIELDS = ("series_ids", "target_position", "target_seq", "true_qty", "history_length")
LOSS_FIELDS = ("quantity_train_loss", "log_qty_loss", "time_loss", "raw_qty_loss", "objective_loss")
ABS_TOL = 1e-8
REL_TOL = 1e-10


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, name, *, finite=True):
    array = np.asarray(value)
    _require(array.ndim == 1 and array.dtype.kind in "iuf", f"{name} must be a numeric one-dimensional vector")
    result = array.astype(np.float64, copy=False)
    _require(not finite or bool(np.isfinite(result).all()), f"Nonfinite {name}")
    return result


def _boundaries(quantity_bounds, history_bounds):
    quantity = _vector(quantity_bounds, "quantity boundaries")
    history = _vector(history_bounds, "history boundaries")
    _require(len(quantity) == 4 and len(history) == 2, "Exactly four quantity and two history boundaries required")
    _require(bool((quantity >= 0).all()) and bool((np.diff(quantity) > 0).all()), "Quantity boundaries must be nonnegative and strictly increasing")
    _require(bool((history == np.floor(history)).all()) and bool((history >= 0).all())
             and bool((history <= 255).all()) and bool((np.diff(history) > 0).all()), "History boundaries must be increasing integers in [0,255]")
    return quantity, history


def cell_indices(truth, history, bounds):
    """Return q_bin * 3 + history_bin; equality belongs to the lower bin."""
    _require(isinstance(bounds, Mapping), "bounds must contain quantity_boundaries and history_boundaries")
    q, h = _boundaries(bounds["quantity_boundaries"], bounds["history_boundaries"])
    true = _vector(truth, "truth")
    lengths = _vector(history, "history")
    _require(len(true) == len(lengths), "Truth/history length mismatch")
    _require(bool((true >= 0).all()), "True quantity must be nonnegative")
    _require(bool((lengths == np.floor(lengths)).all()) and bool(((lengths >= 0) & (lengths <= 255)).all()),
             "History length must be integer in [0,255]")
    return (3 * np.searchsorted(q, true, side="left") + np.searchsorted(h, lengths, side="left")).astype(np.int64)


def safe_cosine(left, right):
    """A zero-norm cosine is not applicable, including for a nonempty vector."""
    a, b = _vector(left, "left vector"), _vector(right, "right vector")
    _require(a.shape == b.shape, "Cosine vector shape mismatch")
    # Scaling avoids overflow/underflow without changing the angle.
    scale_a, scale_b = float(np.max(np.abs(a), initial=0.)), float(np.max(np.abs(b), initial=0.))
    if scale_a == 0. or scale_b == 0.:
        return None
    a, b = a / scale_a, b / scale_b
    value = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    _require(math.isfinite(value), "Nonfinite cosine")
    return max(-1., min(1., value))


def _rows(rows):
    _require(isinstance(rows, Mapping), "Rows must be a mapping of ordered vectors")
    required = set(PAIR_FIELDS) | {"pred_qty"}
    _require(required <= set(rows), "Missing prediction or pairing columns")
    out = {key: _vector(rows[key], key) for key in ("true_qty", "pred_qty", "history_length")}
    n = len(out["true_qty"])
    _require(n > 0, "Empty validation population")
    for key in ("target_position", "target_seq"):
        value = np.asarray(rows[key])
        _require(value.ndim == 1 and value.dtype.kind in "iu" and len(value) == n
                 and bool((value >= 0).all()), f"{key} must contain nonnegative integer identities")
        out[key] = value
    ids = np.asarray(rows["series_ids"])
    _require(ids.ndim == 1 and len(ids) == n and ids.dtype.kind in "USiu", "series_ids must be string or integer identities")
    if ids.dtype.kind in "US":
        _require(all(bool(str(value)) for value in ids), "Empty series identity")
    out["series_ids"] = ids
    for key in LOSS_FIELDS:
        if key in rows:
            out[key] = _vector(rows[key], key)
    _require(all(len(value) == n for value in out.values()), "Row column length mismatch")
    # Position/sequence are paired with series: target_seq alone is not an ID.
    identities = [(str(series), int(position), int(sequence)) for series, position, sequence in
                  zip(ids, out["target_position"], out["target_seq"], strict=True)]
    _require(len(set(identities)) == n, "Duplicate validation target identity")
    try:
        with np.errstate(over="raise", invalid="raise"):
            error = out["pred_qty"] - out["true_qty"]
            squared = np.square(error)
    except FloatingPointError as exc:
        raise ValueError("Nonfinite raw error or squared error") from exc
    _require(bool(np.isfinite(error).all()) and bool(np.isfinite(squared).all()), "Nonfinite raw errors")
    return out, error, squared, identities


def _sum(values):
    value = float(np.sum(values, dtype=np.float64))
    _require(math.isfinite(value), "Nonfinite aggregate")
    return value


def _ratio(numerator, denominator):
    if denominator == 0:
        return None
    value = float(numerator / denominator)
    _require(math.isfinite(value), "Nonfinite derived ratio")
    return value


def _cell(rows, error, squared, mask):
    n = int(np.count_nonzero(mask))
    e = error[mask]
    under, over, tie = e < 0, e > 0, e == 0
    total = {
        "count": n, "true_quantity_sum": _sum(rows["true_qty"][mask]),
        "predicted_quantity_sum": _sum(rows["pred_qty"][mask]),
        "signed_error_sum": _sum(e), "absolute_error_sum": _sum(np.abs(e)),
        "raw_squared_error_sum": _sum(squared[mask]),
        "under_count": int(under.sum()), "over_count": int(over.sum()), "tie_count": int(tie.sum()),
        "under_absolute_error_sum": _sum(-e[under]), "over_absolute_error_sum": _sum(e[over]),
    }
    total.update(mean_bias=_ratio(total["signed_error_sum"], n), MAE=_ratio(total["absolute_error_sum"], n),
                 RMSE=math.sqrt(total["raw_squared_error_sum"] / n) if n else None,
                 under_rate=_ratio(total["under_count"], n), over_rate=_ratio(total["over_count"], n),
                 tie_rate=_ratio(total["tie_count"], n))
    total["losses"] = {key: {"sum": _sum(rows[key][mask]), "mean": _ratio(_sum(rows[key][mask]), n)}
                       for key in LOSS_FIELDS if key in rows}
    return total


def _partitions(indices):
    q, h = indices // 3, indices % 3
    return {
        "quantity_cells": [(q == i, {"bin_index": i}) for i in range(5)],
        "history_cells": [(h == i, {"bin_index": i}) for i in range(3)],
        "cross_cells": [(indices == i, {"cell_index": i, "quantity_bin_index": i // 3,
                                        "history_bin_index": i % 3}) for i in range(15)],
    }


def _close(observed, expected):
    return math.isfinite(observed) and math.isfinite(expected) and math.isclose(observed, expected, abs_tol=ABS_TOL, rel_tol=REL_TOL)


def _conservation(overall, partitions):
    audits = {}
    for label, cells in partitions.items():
        checks = {key: (sum(cell[key] for cell in cells) == overall[key] if key in COUNT_FIELDS else
                        _close(math.fsum(cell[key] for cell in cells), overall[key])) for key in SUM_FIELDS}
        for key in overall["losses"]:
            checks["loss_sum:" + key] = _close(math.fsum(cell["losses"][key]["sum"] for cell in cells), overall["losses"][key]["sum"])
        checks["sign_counts"] = all(cell["under_count"] + cell["over_count"] + cell["tie_count"] == cell["count"] for cell in cells)
        checks["absolute_error_sign_partition"] = all(_close(cell["under_absolute_error_sum"] + cell["over_absolute_error_sum"], cell["absolute_error_sum"]) for cell in cells)
        checks["signed_error_sign_partition"] = all(_close(cell["over_absolute_error_sum"] - cell["under_absolute_error_sum"], cell["signed_error_sum"]) for cell in cells)
        _require(all(checks.values()), "Validation partition conservation failed: " + label)
        audits[label] = checks
    return {"passed": True, "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL, "partitions": audits}


def _summarize(rows, error, squared, indices):
    overall = _cell(rows, error, squared, np.ones(len(error), dtype=bool))
    partitions = {key: [_cell(rows, error, squared, mask) | metadata for mask, metadata in masks]
                  for key, masks in _partitions(indices).items()}
    return {"overall": overall, **partitions, "conservation_audit": _conservation(overall, partitions)}


def summarize_rows(rows, quantity_bounds, history_bounds):
    """Return all 5 quantity, 3 history, and 15 cross cells, including empty ones."""
    values, error, squared, _ = _rows(rows)
    indices = cell_indices(values["true_qty"], values["history_length"],
                           {"quantity_boundaries": quantity_bounds, "history_boundaries": history_bounds})
    result = _summarize(values, error, squared, indices)
    return {"schema": "intermittent_validation_cell_summary_v1", "accumulator_dtype": "float64",
            "boundary_equality": "lower_bin", "quantity_boundaries": list(map(float, quantity_bounds)),
            "history_boundaries": list(map(int, history_bounds)), **result}


def _signed_accounting(values):
    positive = math.fsum(float(x) for x in values if x > 0)
    negative = math.fsum(float(x) for x in values if x < 0)
    net = math.fsum(float(x) for x in values)
    _require(all(math.isfinite(x) for x in (positive, negative, net)), "Nonfinite signed contribution aggregate")
    _require(_close(positive + negative, net), "Signed positive/negative conservation failed")
    return {"positive_sum": positive, "negative_sum": negative, "net_sum": net}


def _pair_cell(b, candidate, n):
    delta_sse = candidate["raw_squared_error_sum"] - b["raw_squared_error_sum"]
    delta_ae = candidate["absolute_error_sum"] - b["absolute_error_sum"]
    _require(math.isfinite(delta_sse) and math.isfinite(delta_ae), "Nonfinite paired error change")
    return {"count": b["count"], "fraction": b["count"] / n if b["count"] else None, "B": b, "candidate": candidate,
            "delta_SSE": delta_sse, "delta_MSE_contribution": delta_sse / n,
            "delta_AE": delta_ae, "delta_MAE_contribution": delta_ae / n}


def compare_rows(B, C, quantity_bounds, history_bounds):
    """Pair exact ordered targets and account for additive SSE/MSE and AE/MAE changes."""
    b, e_b, sse_b, ids_b = _rows(B)
    c, e_c, sse_c, _ = _rows(C)
    pairing = {key: bool(np.array_equal(b[key], c[key])) for key in PAIR_FIELDS}
    _require(all(pairing.values()), "Paired target identity/order/truth/history mismatch: " + ", ".join(k for k, v in pairing.items() if not v))
    indices = cell_indices(b["true_qty"], b["history_length"],
                           {"quantity_boundaries": quantity_bounds, "history_boundaries": history_bounds})
    sb, sc = _summarize(b, e_b, sse_b, indices), _summarize(c, e_c, sse_c, indices)
    n = len(e_b)
    overall = _pair_cell(sb["overall"], sc["overall"], n)
    overall["delta_RMSE"] = sc["overall"]["RMSE"] - sb["overall"]["RMSE"]
    partitions, accounting, audits = {}, {}, {}
    for label, definitions in _partitions(indices).items():
        cells = [_pair_cell(x, y, n) | metadata for x, y, (_, metadata) in
                 zip(sb[label], sc[label], definitions, strict=True)]
        account = {key: _signed_accounting([cell[key] for cell in cells]) for key in
                   ("delta_SSE", "delta_MSE_contribution", "delta_AE", "delta_MAE_contribution")}
        for cell in cells:
            cell["shares"] = {}
            for key, sums in account.items():
                value = cell[key]
                cell["shares"][key] = ({"net": None, "positive": None, "negative": None} if cell["count"] == 0 else
                    {"net": _ratio(value, overall[key]),
                     "positive": _ratio(max(value, 0.), sums["positive_sum"]),
                     "negative": _ratio(min(value, 0.), sums["negative_sum"])})
        checks = {key: _close(account[key]["net_sum"], overall[key]) for key in account}
        checks["count"] = sum(cell["count"] for cell in cells) == n
        checks["positive_negative_reconstruct_net"] = all(_close(v["positive_sum"] + v["negative_sum"], v["net_sum"]) for v in account.values())
        for key in account:
            checks["net_shares:" + key] = all(cell["shares"][key]["net"] is None for cell in cells) if overall[key] == 0 else _close(math.fsum(cell["shares"][key]["net"] for cell in cells if cell["count"]), 1.)
        _require(all(checks.values()), "Paired cell conservation failed: " + label)
        partitions[label], accounting[label], audits[label] = cells, account, checks
    row_changes = {"delta_SSE": _signed_accounting(sse_c - sse_b),
                   "delta_AE": _signed_accounting(np.abs(e_c) - np.abs(e_b))}
    _require(all(_close(values["net_sum"], overall[key]) for key, values in row_changes.items()), "Row/aggregate paired change mismatch")
    return {"schema": "intermittent_paired_validation_cell_comparison_v1", "accumulator_dtype": "float64",
            "pairing_audit": {"passed": True, "count": n, "unique_row_identity_count": n,
                              "exact_fields": pairing,
                              "ordered_row_identity_sha256": hashlib.sha256(json.dumps(ids_b, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()},
            "overall": overall, **partitions, "signed_cell_change_accounting": accounting,
            "signed_row_change_accounting": row_changes,
            "conservation_audit": {"passed": True, "absolute_tolerance": ABS_TOL, "relative_tolerance": REL_TOL,
                                   "B": sb["conservation_audit"], "candidate": sc["conservation_audit"], "paired_partitions": audits},
            "definitions": {"error": "prediction - true_quantity", "MAE_contribution": "delta_AE_cell / N_all_paired_rows",
                            "MSE_contribution": "delta_SSE_cell / N_all_paired_rows",
                            "RMSE": "sqrt(total_SSE/N); delta_RMSE is overall only and is not additive across cells",
                            "share_denominators": "net=overall signed change; positive/negative=sum of corresponding signed cell changes within the same partition",
                            "zero_denominator": None,
                            "interpretation": "Offsetting signed arithmetic shares, not causal percentages."}}
