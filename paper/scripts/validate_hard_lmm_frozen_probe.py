#!/usr/bin/env python3
"""Independently reconcile frozen-probe event artifacts and selection evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.run_hard_lmm_frozen_probe import load_json, save_json, sha256_file, verify_hashes


def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise AssertionError("Non-finite JSON metric")
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    if isinstance(value, list):
        for item in value:
            finite_json(item)


def close(actual, expected, name):
    # The runner evaluates softplus/expm1 in FP32 per stratum, whereas event
    # exports evaluate the full vector. SIMD boundary rounding need not be exact.
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError(f"Event reconciliation failed: {name}: {actual} != {expected}")


def validate(root):
    contract = load_json(ROOT / "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json")
    launch, source, summary = [load_json(root / f"{name}.json") for name in ("launch_contract", "source_manifest", "summary")]
    for key, value in contract.items():
        if launch[key] != value:
            raise AssertionError(f"Frozen contract drift: {key}")
    verify_hashes(ROOT, source["files"])
    assert summary["status"] == "complete" and summary["full_validation"] is True
    assert summary["held_out_test_evaluated"] is False and launch["partial_smoke"] is False
    assert set(summary["datasets"]) == {row["dataset"] for row in contract["datasets"]}
    assert not list(root.rglob("*test*"))
    rows = []
    maximum_event_metric_absolute_difference = 0.0
    for spec in contract["datasets"]:
        name, directory = spec["dataset"], root / spec["dataset"]
        audit = load_json(directory / "baseline_audit.json")
        assert audit["model_state_sha256"] == spec["checkpoint_state_sha256"]
        assert audit["checkpoint_file_sha256"] == spec["checkpoint_file_sha256"]
        assert audit["checkpoint_source_revision"] == spec["checkpoint_source_revision"]
        assert max(audit["replay"].values()) <= 1e-5
        for split, cache in audit["cache"].items():
            assert sha256_file(directory / f"{split}_cache.pt") == cache["sha256"]
            expected = min(cache["available_targets"], 65536) if split == "train" else cache["available_targets"]
            assert cache["selected_targets"] == expected
        ds = summary["datasets"][name]
        finite_json(ds)
        assert ds == load_json(directory / "summary.json")
        for candidate, result in ds["candidates"].items():
            assert candidate in ("calibration", "shrinkage")
            assert result == load_json(directory / f"{candidate}_summary.json")
            assert result["base_unchanged"] and result["adapter_replay_exact"]
            assert result["baseline_state_sha256_before"] == result["baseline_state_sha256_after"] == spec["checkpoint_state_sha256"]
            assert sha256_file(directory / f"{candidate}_best.pt") == result["adapter_sha256"]
            selected = min(result["history"], key=lambda row: row["joint_objective"])
            assert selected["epoch"] == result["selection"]["best_epoch"]
            assert result["history"][-1]["epoch"] == result["selection"]["completed_epochs"]
            events = pl.read_parquet(directory / f"{candidate}_validation_events.parquet")
            assert events.height == audit["cache"]["validation"]["available_targets"]
            assert events["target_index"].n_unique() == events.height
            assert np.isfinite(events.to_numpy()).all()
            q = events["quantity"].to_numpy()
            pred, base = events["candidate_prediction"].to_numpy(), events["baseline_prediction"].to_numpy()
            np.testing.assert_allclose(events["absolute_error_delta"], abs(pred - q) - abs(base - q), rtol=0, atol=1e-9)
            np.testing.assert_allclose(events["squared_error_delta"], (pred - q)**2 - (base - q)**2, rtol=0, atol=1e-9)
            if candidate == "calibration":
                assert np.max(abs(events["logit_correction"].to_numpy())) <= 0.05000001
            else:
                assert events["gate"].min() >= 0.8 and events["gate"].max() <= 1
            boundaries = launch["datasets"][[row["dataset"] for row in launch["datasets"]].index(name)]
            # Baseline launch is hash-pinned; use its train-derived thresholds, not validation quantiles.
            baseline_launch = ROOT / boundaries["artifact_dir"] / "launch_contract.json"
            if not baseline_launch.exists() and name == "intermittent_v2":
                baseline_launch = ROOT / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/launch_contract.json"
            assert sha256_file(baseline_launch) == spec["contract_sha256"]
            p50, p90, p95, p99 = load_json(baseline_launch)["quantity_contract"]["boundaries"]
            h = events["history_length"].to_numpy()
            masks = {"overall": np.ones(len(q), dtype=bool), "body_le_p95": q <= p95,
                "le_p50": q <= p50, "p50_p90": (q > p50) & (q <= p90), "p90_p95": (q > p90) & (q <= p95),
                "p95_p99": (q > p95) & (q <= p99), "gt_p99": q > p99, "tail_gt_p95": q > p95,
                "history_le_64": h <= 64, "history_65_128": (h > 64) & (h <= 128), "history_gt_128": h > 128}
            for scope, mask in masks.items():
                saved = result["scopes"][scope]
                assert saved["count"] == int(mask.sum())
                if not mask.any():
                    assert saved["status"] == "empty"
                    continue
                for role, predictions in (("candidate", pred), ("baseline", base)):
                    error = predictions[mask] - q[mask]
                    mae, rmse = np.abs(error).mean(), np.sqrt(np.square(error).mean())
                    close(mae, saved[role]["qty_mae"], f"{name}/{candidate}/{scope}/MAE")
                    close(rmse, saved[role]["qty_rmse"], f"{name}/{candidate}/{scope}/RMSE")
                    maximum_event_metric_absolute_difference = max(maximum_event_metric_absolute_difference,
                        abs(mae - saved[role]["qty_mae"]), abs(rmse - saved[role]["qty_rmse"]))
                assert saved["candidate"]["time_nll"] == saved["baseline"]["time_nll"]
            rows.append({"dataset": name, "candidate": candidate, "validation_targets": len(q),
                "selected_epoch": selected["epoch"], "decision": result["decision"]["status"],
                "body_change_pct": result["decision"]["body_relative_change"] * 100,
                "rmse_change_pct": result["decision"]["rmse_relative_change"] * 100,
                "p99_change_pct": result["decision"]["p99_relative_change"] * 100,
                "event_metrics_reconciled": True, "base_unchanged": True, "time_unchanged": True})
    assert len(rows) == 8
    return {"status": "verified", "source_revision": source["source_revision"], "all_metrics_finite": True,
        "held_out_test_evaluated": False, "event_and_summary_metrics_match_within_fp32_tolerance": True,
        "event_metric_absolute_tolerance": 1e-6, "event_metric_relative_tolerance": 1e-6,
        "maximum_event_metric_absolute_difference": maximum_event_metric_absolute_difference, "runs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.artifact)
    save_json(args.output, result)
    print(json.dumps(result, indent=2))
