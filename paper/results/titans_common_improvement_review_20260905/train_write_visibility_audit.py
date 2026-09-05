#!/usr/bin/env python3
"""Train-only metadata audit of effective target histories for MAC chunk-16.

No model execution, fitting, validation/test data decoding, or prediction scoring.
The input parquet is filtered to train and projected to series key/seq before
collection. Whole-file SHA-256 reads bytes for identity, without decoding rows.
Saved train caches include labels, but only target_index/history_length are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def audit(project: Path) -> dict:
    contract_path = project / "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json"
    contract = json.loads(contract_path.read_text())
    results = []
    for row in contract["datasets"]:
        if row["dataset"] == "raf_spare_parts":
            continue
        path = project / row["data_path"]
        data_hash = sha256(path)
        assert data_hash == row["data_sha256"]
        frame = (
            pl.scan_parquet(path)
            .filter(pl.col("chronological_split") == "train")
            .select("oper_part_no", "seq")
            .collect()
            .sort(["oper_part_no", "seq"])
        )
        grouped = (
            frame.group_by("oper_part_no", maintain_order=True)
            .agg(pl.col("seq").cast(pl.Int32))
            .sort("oper_part_no")
        )
        lengths = []
        for seq_list in grouped["seq"].to_list():
            seq = np.asarray(seq_list, dtype=np.int32)
            if len(seq) < 2:
                continue
            ends = np.arange(len(seq) - 1)
            starts = np.searchsorted(seq, seq[:-1] - (row["lookback"] - 1), side="left")
            history = np.minimum(ends + 1 - starts, row["max_seq_len"] - 1)
            assert (history >= 1).all()
            lengths.append(history)
        history = np.concatenate(lengths)
        audit_path = project / "paper/results/count_aware_hard_lmm_frozen_probe_20260903" / row["dataset"] / "baseline_audit.json"
        cache_audit = json.loads(audit_path.read_text())["cache"]["train"]
        assert len(history) == cache_audit["available_targets"]
        cache_path = project / "search_artifacts/hard_lmm_frozen_probe_20260903" / row["dataset"] / "train_cache.pt"
        cache_hash = sha256(cache_path)
        assert cache_hash == cache_audit["sha256"]
        saved_cache = torch.load(cache_path, map_location="cpu", weights_only=True)
        indices = saved_cache["target_index"].numpy()
        cached_history = saved_cache["history_length"].numpy()
        parity = np.array_equal(history[indices], cached_history)
        assert parity
        del saved_cache
        mac_launches = []
        for seed, host in ((52, "5090"), (62, "5080")):
            launch_path = project / f"search_artifacts/count_aware_titantpp_mac_stable_seed{seed}_e300_20260831_{host}/validation_e300/{row['contract_dataset']}/seed_{seed}/launch_contract.json"
            launch = json.loads(launch_path.read_text())
            assert launch["data_sha256"] == data_hash
            assert launch["lookback_weeks"] == row["lookback"]
            assert launch["max_seq_len"] == row["max_seq_len"]
            assert launch["time_head"]["train_time_statistics"]["target_count"] == len(history)
            assert launch["epochs"] == 300 and launch["status"] == "complete"
            mac_launches.append({"seed": seed, "path": str(launch_path), "sha256": sha256(launch_path), "context_data_target_count_match": True})
        results.append({
            "dataset": row["contract_dataset"],
            "data_path": str(path), "data_sha256": data_hash,
            "train_rows": frame.height, "train_series": grouped.height,
            "train_targets": len(history), "lookback": row["lookback"],
            "max_seq_len_including_target": row["max_seq_len"],
            "H_le16": int((history <= 16).sum()),
            "H_gt16": int((history > 16).sum()),
            "H_le16_pct": float((history <= 16).mean() * 100),
            "H_le8": int((history <= 8).sum()),
            "H_le32": int((history <= 32).sum()),
            "H_eq16": int((history == 16).sum()),
            "H_min": int(history.min()), "H_max": int(history.max()),
            "cache_path": str(cache_path), "cache_sha256": cache_hash,
            "cache_audit_path": str(audit_path), "cache_audit_sha256": sha256(audit_path),
            "cache_parity_rows": len(indices), "cache_H_exact_parity": bool(parity),
            "cache_sample_H_le16": int((cached_history <= 16).sum()),
            "mac_stable_completed_e300_launches": mac_launches,
        })
    return {
        "status": "PASS", "schema_version": 1,
        "scope": "train-only metadata; no model forward/backward, fitting, or prediction metrics",
        "project_root": str(project), "script_path": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
        "contract_path": str(contract_path), "contract_sha256": sha256(contract_path),
        "parquet_filter": "chronological_split == train (lazy predicate before collect)",
        "decoded_parquet_columns": ["chronological_split (predicate)", "oper_part_no", "seq"],
        "cache_fields_used": ["target_index", "history_length"],
        "cache_loading_note": "Existing train-only tensors are loaded with weights_only=True; quantity/time/prediction fields are not used.",
        "denominator": "Every eligible next-event TRAIN target, once per epoch; not rows, series, segments, or cached sample size.",
        "target_stride": "One target per consecutive event, i=0..n-2 within each sorted train series; no downsampling.",
        "window_rule": "Observed context seq in [seq_i-(W-1), seq_i], then append target i+1 and retain most recent max_seq_len tokens; H excludes target and padding.",
        "H_formula": "min(i+1-searchsorted(seq, seq_i-(W-1), side=left), max_seq_len-1)",
        "reset_boundary": "This is effective history in each independent input window. No cross-window/cross-batch memory is added. MAC state-reset semantics are a separate code audit.",
        "MAC_relevance_assumption": "For the independently audited segment-16 MAC: output for last observed position cannot read writes from its own segment; H<=16 has no earlier written segment. This metadata audit does not itself execute or prove that code behavior.",
        "not_identified": [
            "A causal explanation of static T0 performance; static memory has no chunk-16 write restriction.",
            "The exact fraction of useful or nonzero gradients when H>16; visibility is necessary, not sufficient.",
            "A guarantee that changing write/read cadence improves held-out prediction.",
            "Validation/test H proportions; only existing launch metadata is read, no validation/test rows or new predictions.",
        ],
        "source_references": {
            "dataset": "data_loader/event_seq_data_module.py:295 (target indexing), :337 (window/target/truncation)",
            "cached_H": "paper/scripts/hard_lmm_frozen_probe.py:52,:72 (right-padded valid length minus target)",
            "cache_sample": "paper/scripts/hard_lmm_frozen_probe.py:91 (seed42 uniform without replacement)",
        },
        "runtime": {"numpy": np.__version__, "polars": pl.__version__, "torch": torch.__version__},
        "datasets": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/Users/igwanhyeong/PycharmProjects/paper_research"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    arguments = parser.parse_args()
    result = audit(arguments.project)
    arguments.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(arguments.output), "datasets": [{k: row[k] for k in ("dataset", "train_targets", "H_le16", "H_gt16", "H_le16_pct", "cache_parity_rows", "cache_H_exact_parity")} for row in result["datasets"]]}, indent=2))
