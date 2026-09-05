"""Input-only context selection; target labels have a separate explicit reader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import polars as pl
import torch

from paper.scripts.hard_lmm_temporal_features import observed_features

ROW = "__blind_physical_row_id"
PREFIX = "instacart_balanced_temporal_v1"
SEED = 20260905
LOOKBACK, MAX_LEN = 52, 64


def _rank(kind, series, end=None):
    text = f"{PREFIX}:{kind}:{SEED}:{series}"
    if end is not None:
        text += f":{end}"
    return hashlib.sha256(text.encode()).digest()


def _train_scan(path, columns):
    # Row numbering precedes filtering: IDs refer to physical parquet row order.
    return (pl.scan_parquet(path).with_row_index(ROW)
            .filter(pl.col("chronological_split") == "train")
            .select(pl.col(ROW).cast(pl.Int64), *[pl.col(name) for name in columns]))


def _integer_array(values, name):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    array = np.asarray(values)
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain finite integer IDs")
    return array.astype(np.int64)


def select_contexts(path, excluded_ids, maximum_series=65536):
    """Select using series identity and seq only, without reading gaps or values.

    ``excluded_ids`` may be IDs or the prior input-feature mapping. The latter
    verifies each previous target offset and history length before exclusion.
    Selection arrays retain the complete original train-series ordinal mapping.
    """
    if not isinstance(maximum_series, int) or isinstance(maximum_series, bool) or maximum_series < 1:
        raise ValueError("maximum_series must be a positive integer")
    prior = None
    if isinstance(excluded_ids, Mapping):
        names = ("series_index", "target_index", "context_end", "history_length")
        prior = {name: _integer_array(excluded_ids[name], name) for name in names}
        if len({len(value) for value in prior.values()}) != 1:
            raise ValueError("Prior identity arrays have different lengths")
        excluded = set(prior["series_index"].tolist())
    else:
        excluded = set(_integer_array(list(excluded_ids), "excluded_ids").tolist())
    frame = _train_scan(path, ["oper_part_no", "seq"]).collect()
    if not frame.height or frame.select(pl.any_horizontal(pl.all().is_null()).any()).item():
        raise ValueError("Train identity metadata must be nonempty and non-null")
    frame = frame.sort(["oper_part_no", "seq"]).with_columns(pl.col("seq").cast(pl.Int32))
    if frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().any()).item():
        raise ValueError("Duplicate native series/seq identity is ambiguous")
    groups = frame.group_by("oper_part_no", maintain_order=True).agg(pl.len().alias("n"))
    keys, counts = groups["oper_part_no"].to_list(), groups["n"].to_list()
    if any(series < 0 or series >= len(keys) for series in excluded):
        raise ValueError("Excluded series ID is outside the full train mapping")
    seq, physical = frame["seq"].to_numpy().astype(np.int64), frame[ROW].to_numpy()
    prior_groups = {}
    if prior is not None:
        for position, series in enumerate(prior["series_index"]):
            prior_groups.setdefault(int(series), []).append(position)
    candidates, source_start, target_offset = [], 0, 0
    for series, count in enumerate(counts):
        series_seq = seq[source_start:source_start + count]
        ends = np.arange(max(count - 1, 0))
        starts = np.maximum(np.searchsorted(series_seq, series_seq[:-1] - (LOOKBACK - 1)), ends - (MAX_LEN - 2))
        lengths = ends - starts + 1
        for position in prior_groups.get(series, []):
            end = prior["context_end"][position]
            if not (0 <= end < count - 1):
                raise ValueError("Prior context end is outside its train series")
            if (prior["target_index"][position] != target_offset + end
                    or prior["history_length"][position] != lengths[end]):
                raise ValueError("Prior global target offset or history length mismatch")
        eligible = ends[lengths >= 3]
        if series not in excluded and eligible.size:
            end = min(eligible.tolist(), key=lambda index: _rank("context", series, index))
            candidates.append((series, end, int(starts[end]), source_start, target_offset))
        source_start += count
        target_offset += max(count - 1, 0)
    chosen = sorted(sorted(candidates, key=lambda row: _rank("series", row[0]))[:maximum_series])
    if not chosen:
        raise ValueError("No unused series has an eligible observed context")
    selection = {name: [] for name in ("series_index", "native_series_key", "context_start", "context_end",
        "target_index", "history_length", "target_physical_row_id", "fold")}
    observed_ids, offsets = [], [0]
    for series, end, start, origin, offset in chosen:
        values = (series, keys[series], start, end, offset + end, end - start + 1,
                  int(physical[origin + end + 1]),
                  int.from_bytes(hashlib.sha256(f"{SEED}:{series}".encode()).digest()[:8], "big") % 2)
        for name, value in zip(selection, values):
            selection[name].append(value)
        observed_ids.extend(physical[origin + start:origin + end + 1].tolist())
        offsets.append(len(observed_ids))
    selection = {name: value if name == "native_series_key" else np.asarray(value, dtype=np.int64)
                 for name, value in selection.items()}
    selection["observed_physical_row_ids"] = np.asarray(observed_ids, dtype=np.int64)
    selection["observed_offsets"] = np.asarray(offsets, dtype=np.int64)
    metadata = {"algorithm": PREFIX, "seed": SEED, "lookback": LOOKBACK, "max_seq_len": MAX_LEN,
        "train_rows": frame.height, "train_series": len(keys), "train_targets": target_offset,
        "excluded_series": len(excluded), "eligible_unused_series": len(candidates),
        "selected_series": len(chosen), "maximum_series": maximum_series,
        "prior_identity_rows_verified": 0 if prior is None else len(prior["series_index"]),
        "native_series_dtype": str(frame.schema["oper_part_no"]),
        "series_mapping_sha256": hashlib.sha256(json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest(),
        "selection_columns_read": [ROW, "oper_part_no", "seq"],
        "target_values_read": False, "gap_values_read": False, "duplicate_series_seq": False}
    return metadata, selection


def _join_selected(path, physical_ids, columns):
    ids = _integer_array(physical_ids, "physical row IDs")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Selected physical row IDs must be unique")
    wanted = pl.DataFrame({ROW: ids, "__selected_order": np.arange(len(ids), dtype=np.int64)})
    # Cast value columns only AFTER collecting matched rows, so poisoned or
    # unrequested target cells never enter input conversion or output memory.
    selected = (_train_scan(path, columns).join(wanted.lazy(), on=ROW, how="inner")
                .sort("__selected_order").collect())
    if selected.height != len(ids) or not np.array_equal(selected[ROW].to_numpy(), ids):
        raise ValueError("A selected row is missing or outside the train split")
    return selected


def read_observed(path, selection):
    """Return feature NumPy arrays and right-padded [N,64] observed-only tensors."""
    ids = _integer_array(selection["observed_physical_row_ids"], "observed IDs")
    targets = _integer_array(selection["target_physical_row_id"], "target IDs")
    offsets = _integer_array(selection["observed_offsets"], "observed offsets")
    lengths = _integer_array(selection["history_length"], "history lengths")
    if (len(offsets) != len(targets) + 1 or offsets[0] != 0 or offsets[-1] != len(ids)
            or not np.array_equal(np.diff(offsets), lengths) or np.any((lengths < 3) | (lengths >= MAX_LEN))):
        raise ValueError("Invalid observed row boundaries")
    if np.intersect1d(ids, targets).size:
        raise ValueError("Observed selection includes a selected target row")
    selected = _join_selected(path, ids, ["delta_t", "demand_qty"]).with_columns(
        pl.col("delta_t").cast(pl.Int32).clip(1, None).cast(pl.Float32),
        pl.col("demand_qty").cast(pl.Float64).cast(pl.Float32))
    gaps = torch.from_numpy(selected["delta_t"].to_numpy().copy())
    qty = torch.from_numpy(selected["demand_qty"].to_numpy().copy())
    histories = {"dts": torch.zeros(len(targets), MAX_LEN), "quantities": torch.zeros(len(targets), MAX_LEN),
                 "mask": torch.zeros(len(targets), MAX_LEN, dtype=torch.bool),
                 "history_length": torch.from_numpy(lengths.copy())}
    records = []
    for position, length in enumerate(lengths):
        sl = slice(offsets[position], offsets[position + 1])
        records.append(observed_features(gaps[sl], qty[sl]))
        histories["dts"][position, :length], histories["quantities"][position, :length] = gaps[sl], qty[sl]
        histories["mask"][position, :length] = True
    features = {name: np.asarray([record[name] for record in records], dtype=float) for name in records[0]}
    for name in ("series_index", "fold", "context_start", "context_end", "target_index", "target_physical_row_id"):
        features[name] = np.asarray(selection[name], dtype=np.int64).copy()
    return features, histories


def read_body_labels(path, selection, allowed_target_positions):
    """Explicit post-gate label access; returns quantity only, never target gap.

    The caller must authorize and freeze these positions after its input-only
    design gate. This function does not infer approval or choose a body subset.
    """
    positions = _integer_array(allowed_target_positions, "allowed target positions")
    targets = _integer_array(selection["target_physical_row_id"], "target IDs")
    if not len(positions) or np.any((positions < 0) | (positions >= len(targets))) or len(np.unique(positions)) != len(positions):
        raise ValueError("Allowed target positions must be nonempty, unique and in range")
    ids = targets[positions]
    selected = _join_selected(path, ids, ["demand_qty"])
    quantity = selected["demand_qty"].cast(pl.Float64).cast(pl.Float32).to_numpy().copy()
    if not np.isfinite(quantity).all() or np.any(quantity < 0):
        raise ValueError("Allowed count labels must be finite and nonnegative")
    return {"selection_position": positions.copy(), "target_physical_row_id": ids.copy(), "quantity": quantity}
