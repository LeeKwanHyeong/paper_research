#!/usr/bin/env python3
"""Verify recorded traces and export compact mechanism diagnostic evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.run_hard_lmm_frozen_probe import load_json, save_json, sha256_file
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json


def check_close(a, b, name, tolerance=1e-6):
    if not math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance):
        raise AssertionError(f"Mismatch {name}: {a} != {b}")


def verify_traces(root, summary):
    rows = []
    for name, result in summary["shrinkage"].items():
        path = root / f"{name}_train_steps.jsonl"
        assert sha256_file(path) == result["trace_sha256"]
        aggregates = {}
        nonzero, last_nonzero, total = 0, 0, 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                finite_json(row)
                agg = aggregates.setdefault(row["epoch"], {"count": 0, "targets": 0, "objective": 0., "zero": 0, "norm": 0.})
                assert row["batch"] == agg["count"]
                for phase in ("before", "after"):
                    values = row[phase]
                    assert .8 - 1e-7 <= values["gate_min"] <= values["gate_mean"] <= 1
                    check_close(values["relative_residual_norm_reduction_mean"], 1 - values["gate_mean"], "residual scaling", 1e-12)
                    assert 0 <= values["identity_fraction"] <= 1
                total += 1
                if row["gradient_norm"] > 0:
                    nonzero += 1
                    last_nonzero = total
                agg["count"] += 1
                agg["targets"] += row["targets"]
                agg["objective"] += row["objective"] * row["targets"]
                agg["zero"] += row["gradient_norm"] == 0
                agg["norm"] += row["gradient_norm"]
        assert set(aggregates) == set(range(1, 11))
        for history in result["history"]:
            agg = aggregates[history["epoch"]]
            assert agg["targets"] == result["train_targets"]
            assert agg["count"] == history["batches"]
            check_close(agg["objective"] / agg["targets"], history["train_joint_objective"], "train loss", 1e-12)
            check_close(agg["zero"] / agg["count"], history["zero_gradient_batch_fraction"], "zero gradient", 1e-12)
            check_close(agg["norm"] / agg["count"], history["gradient_norm_mean"], "gradient norm", 1e-12)
        final = result["snapshots"][-1]["scopes"]["overall"]
        rows.append({"dataset": name, "train_targets": result["train_targets"],
            "total_batches": total, "nonzero_gradient_batches": nonzero,
            "last_nonzero_gradient_step_1based": last_nonzero,
            "final_identity_fraction": final["identity_fraction"],
            "final_gate_mean": final["gate_mean"],
            "final_relative_residual_norm_reduction": final["relative_residual_norm_reduction_mean"],
            "final_projected_residual_abs_reduction": final["projected_residual_reduction_abs_mean"],
            "oracle_shrink_helpful_fraction": final["oracle_shrink_helpful_fraction"],
            "oracle_shrink_derivative_mean": final["oracle_shrink_derivative_mean"],
            "historical_train_objective_max_abs_gap": result["historical_train_objective_max_absolute_difference"],
            "final_train_log_mse_baseline": final["baseline"]["log_qty_mse"],
            "final_train_log_mse_candidate": final["candidate"]["log_qty_mse"],
            "final_train_mae_baseline": final["baseline"]["qty_mae"],
            "final_train_mae_candidate": final["candidate"]["qty_mae"],
            "final_train_rmse_baseline": final["baseline"]["qty_rmse"],
            "final_train_rmse_candidate": final["candidate"]["qty_rmse"],
            "clamp_inactive_at_final_epoch": final["identity_fraction"] == 1 and final["score_max"] < 0 and result["history"][-1]["zero_gradient_batch_fraction"] == 1})
    return rows


def verify_taxi(root, source, summary):
    fit = load_json(root / "taxi_constant_train_selection.json")
    assert fit["selection_split"] == "train" and fit["validation_used_for_selection"] is False
    assert len(fit["curve"]) == 1001
    selected = min(fit["curve"], key=lambda r: (r["train_log_mse"], abs(r["offset"])))
    assert selected == fit["selected"] and selected["offset"] == summary["taxi"]["offset"]
    cache = torch.load(source / "yellow_trip_hourly/train_cache.pt", map_location="cpu", weights_only=True)
    z, target = cache["z"].double().numpy(), np.log1p(cache["quantity"].double().numpy().clip(min=0))
    offsets = np.asarray([r["offset"] for r in fit["curve"]])
    assert offsets.min() == -.05 and offsets.max() == .05 and 0 in offsets
    for start in range(0, 1001, 32):
        block = offsets[start:start + 32]
        mse = np.square(np.logaddexp(0, z[:, None] + block) - target[:, None]).mean(0)
        for observed, row in zip(mse, fit["curve"][start:start + 32]):
            check_close(observed, row["train_log_mse"], "independent train constant curve", 1e-12)
    validation = torch.load(source / "yellow_trip_hourly/validation_cache.pt", map_location="cpu", weights_only=True)
    events = pl.read_parquet(source / "yellow_trip_hourly/calibration_validation_events.parquet")
    assert np.array_equal(events["target_index"].to_numpy(), validation["target_index"].numpy())
    quantity = events["quantity"].to_numpy()
    constant = F.softplus(validation["z"] + selected["offset"]).expm1().double().numpy()
    predictions = {"mlp": events["candidate_prediction"].to_numpy(), "constant": constant}
    observed_baseline = events["baseline_prediction"].to_numpy()
    overall = summary["taxi"]["validation"]
    for name, prediction in predictions.items():
        error = prediction - quantity
        saved = overall["scopes"][name]["overall"]["candidate"]
        check_close(abs(error).mean(), saved["qty_mae"], "independent validation MAE")
        check_close(np.sqrt(np.square(error).mean()), saved["qty_rmse"], "independent validation RMSE")
        check_close(np.sum(abs(observed_baseline - quantity) - abs(error)), overall["bins"]["overall"]["absolute_error_reduction_sum"][name], "absolute error reduction")
        check_close(np.sum(np.square(observed_baseline - quantity) - np.square(error)), overall["bins"]["overall"]["squared_error_reduction_sum"][name], "squared error reduction")
    return {"train_grid_recomputed_with_numpy": True, "validation_event_metrics_reconciled": True}


def main(root, source, output):
    summary = load_json(root / "summary.json")
    finite_json(summary)
    assert summary["status"] == "complete" and summary["held_out_test_evaluated"] is False
    assert not summary["new_candidate_promoted"] and not summary["backbone_training"]
    for path, digest in load_json(root / "input_digests.json").items():
        assert sha256_file(Path(path)) == digest, path
    for name, digest in load_json(root / "output_digests.json").items():
        assert sha256_file(root / name) == digest, name
    assert not list(root.rglob("*test*"))
    gates = verify_traces(root, summary)
    taxi_audit = verify_taxi(root, source, summary)
    output.mkdir(parents=True, exist_ok=False)
    pl.DataFrame(gates).write_csv(output / "gate_summary.csv")
    taxi = summary["taxi"]
    pl.DataFrame([{"model": name, **taxi["validation"]["scopes"][candidate]["overall"][role]}
        for name, candidate, role in (("baseline", "mlp", "baseline"), ("MLP", "mlp", "candidate"), ("constant", "constant", "candidate"))]).write_csv(output / "taxi_validation_table.csv")
    for axis, scopes in {"quantity": ["le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99"],
                         "history": ["history_le_64", "history_65_128", "history_gt_128"]}.items():
        rows = []
        for scope in scopes:
            row = taxi["validation"]["bins"][scope]
            if row["status"] == "empty":
                continue
            rows.append({"scope": scope, "count": row["count"], "mean_mlp_logit_correction": row["mlp_correction"]["mean"],
                "baseline_bias": row["signed_bias"]["baseline"],
                "mlp_bias": row["signed_bias"]["mlp"], "constant_bias": row["signed_bias"]["constant"],
                "MLP_AE_reduction": row["absolute_error_reduction_sum"]["mlp"],
                "constant_AE_reduction": row["absolute_error_reduction_sum"]["constant"],
                "MLP_SE_reduction": row["squared_error_reduction_sum"]["mlp"],
                "constant_SE_reduction": row["squared_error_reduction_sum"]["constant"]})
        pl.DataFrame(rows).write_csv(output / f"taxi_{axis}_decomposition.csv")
    save_json(output / "diagnostic_verification.json", {"status": "verified", "trace_aggregate_reconciled": True,
        "input_output_digests_verified": True, "all_metrics_finite": True, "held_out_test_evaluated": False, **taxi_audit})
    for name in ("execution_manifest", "input_digests", "output_digests", "probe_status", "taxi_constant_comparison"):
        save_json(output / f"{name}.json", load_json(root / f"{name}.json"))
    save_json(output / "gate_epoch_summaries.json", {k: {"history": v["history"], "snapshots": v["snapshots"]}
        for k, v in summary["shrinkage"].items()})
    save_json(output / "constant_train_selection.json", {k: v for k, v in load_json(root / "taxi_constant_train_selection.json").items() if k != "curve"})
    print(json.dumps({"status": "verified", "gate": gates, "taxi_offset": taxi["offset"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.artifact.resolve(), args.source_artifact.resolve(), args.output.resolve())
