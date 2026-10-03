#!/usr/bin/env python3
"""Reconstruct Taxi target/history metadata; no models or held-out row analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


DATA_SHA256 = "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46"
SOURCE_REVISION = "ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile(frame: pd.DataFrame) -> dict:
    dt = frame.dt.to_numpy()
    return {
        "count": len(frame),
        "dt1_count": int((dt == 1).sum()),
        "dt1_share": float((dt == 1).mean()),
        "dt_mean": float(dt.mean()),
        "dt_p50": float(np.quantile(dt, 0.50)),
        "dt_p99": float(np.quantile(dt, 0.99)),
        "dt_max": int(dt.max()),
        "dt_gt15_count": int((dt > 15).sum()),
        "dt_gt50_count": int((dt > 50).sum()),
        "qty_mean": float(frame.qty.mean()),
        "qty_p50": float(frame.qty.median()),
        "qty_le2_count": int((frame.qty <= 2).sum()),
        "history_min": int(frame.history.min()),
        "history_max": int(frame.history.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    observed_hash = sha256(args.data)
    assert observed_hash == DATA_SHA256
    frame = pq.read_table(
        args.data,
        columns=["oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split"],
        filters=[("chronological_split", "in", ["train", "validation"])],
    ).to_pandas().sort_values(["oper_part_no", "seq"])
    assert set(frame.chronological_split) == {"train", "validation"}
    rows = []
    for _, group in frame.groupby("oper_part_no", sort=True):
        seq = group.seq.to_numpy()
        dt = group.delta_t.to_numpy()
        qty = group.demand_qty.to_numpy()
        split = group.chronological_split.to_numpy()
        assert np.all(np.diff(seq) > 0)
        assert np.array_equal(np.diff(seq), dt[1:])
        for i in range(len(seq) - 1):
            left = seq[i] - (168 - 1)
            start = int(np.searchsorted(seq, left, side="left"))
            history = min(i - start + 1, 256 - 1)
            rows.append((split[i + 1], history, max(1, int(dt[i + 1])),
                         float(qty[i + 1]), bool(seq[i] - seq[0] >= 167)))
    targets = pd.DataFrame(rows, columns=["split", "history", "dt", "qty", "full_window"])
    expected = {"train": 38393, "validation": 8268}
    assert targets.groupby("split").size().to_dict() == expected
    assert len(frame) == 46792 and frame.oper_part_no.nunique() == 131
    assert targets.history.max() == 168
    result = {
        "data_path": str(args.data.resolve()),
        "data_sha256": observed_hash,
        "source_revision_reviewed": SOURCE_REVISION,
        "access_scope": "Parquet predicate selects train/validation before table materialization. File SHA hashes opaque bytes; no held-out rows are materialized or analyzed.",
        "filtered_rows": len(frame),
        "series_count": int(frame.oper_part_no.nunique()),
        "assertions": {"target_counts_match": expected, "nonfirst_dt_equals_seq_difference": True, "history_max_is_168": True},
        "reconstruction": {
            "time_unit": "hour",
            "target": "For each series and context end i, target is stored delta_t[i+1], cast to int32 and clipped to minimum 1.",
            "history": "Count rows with seq in [seq[i]-167, seq[i]], capped at 255 after appending one target; no cap binds in Taxi (max history 168).",
            "split": "Target split determines train/validation; validation history includes earlier train and validation rows.",
            "full_window": "Diagnostic only: context seq minus first observed series seq >= 167; excludes early-series startup context.",
            "timestamps": "Absolute demand_dt/time_bucket columns are not passed by this loader; seq chooses the window and delta_t is model input/target.",
            "quantile_method": "numpy.quantile default linear interpolation",
        },
        "source_files": {},
        "profiles": {},
        "interpretation_limits": [
            "Metadata alone cannot attribute the epoch-83 time-loss spike to individual targets or estimate their losses.",
            "history<=64 includes early dense-series startup in train, so an unconditional short-history train/validation comparison is composition-confounded.",
            "Mature train short histories and validation short histories both contain sparse, low-quantity, long-gap events; this does not establish a new validation-only duration regime.",
            "No model checkpoint was opened, no torch.load used, no training/evaluation was run.",
        ],
    }
    refs = {
        "data_loader/event_seq_data_module.py": [[246, 262], [295, 312], [337, 354]],
        "paper/scripts/count_aware_tpp_backbone/core.py": [[19, 25], [69, 87], [101, 101]],
        "paper/scripts/count_aware_tpp_backbone/datasets.py": [[52, 57]],
        "simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb": [[185, 201], [235, 242]],
    }
    for relative, line_ranges in refs.items():
        path = args.source_root / relative
        result["source_files"][relative] = {"path": str(path.resolve()), "sha256": sha256(path), "line_ranges": line_ranges}
    for split, group in targets.groupby("split"):
        short = group[group.history <= 64]
        result["profiles"][split] = {
            "overall": profile(group),
            "history_le64": profile(short),
            "history_65to128": profile(group[(group.history > 64) & (group.history <= 128)]),
            "history_gt128": profile(group[group.history > 128]),
            "short_history_by_elapsed_observation": {
                "full_168h_window" if full else "early_series_startup": profile(subset)
                for full, subset in short.groupby("full_window")
            },
        }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "target_counts": expected}))


if __name__ == "__main__":
    main()
