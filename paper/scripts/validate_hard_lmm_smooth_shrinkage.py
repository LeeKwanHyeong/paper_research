#!/usr/bin/env python3
"""Reconcile frozen smooth-gate artifacts without fitting or changing candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.hard_lmm_frozen_probe import predict
from paper.scripts.run_hard_lmm_frozen_probe import load_json, save_json, sha256_file, verify_hashes
from paper.scripts.run_hard_lmm_smooth_shrinkage import KINDS, read_cache, restore_checkpoint
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json


def close(a, b, label, tolerance=1e-6):
    if not math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"Mismatch {label}: {a} != {b}")


def check_history(result, policy):
    history = result["history"]
    assert [row["epoch"] for row in history] == list(range(-1, result["completed_epochs"] + 1))
    assert min(history, key=lambda row: row["joint_objective"])["epoch"] == result["best_epoch"]
    assert result["selected_kind"] == ("identity" if result["best_epoch"] == -1 else result["kind"])
    best = min(history[:2], key=lambda row: row["joint_objective"])
    stale = 0
    for row in history[2:]:
        finite_json(row)
        if row["joint_objective"] < best["joint_objective"]:
            best, stale = row, 0
        else:
            stale += 1
        assert row["best_epoch"] == best["epoch"]
        close(row["time_nll"], history[0]["time_nll"], "unchanged time", 0)
        close(row["joint_objective"], row["time_nll"] + row["log_qty_mse"], "joint", 1e-12)
        assert 0 <= row["zero_gradient_batches"] <= row["batches"]
        close(row["zero_gradient_batch_fraction"], row["zero_gradient_batches"] / row["batches"], "gradient fraction", 1e-12)
        close(row["train_gate"]["relative_residual_reduction_mean"], 1 - row["train_gate"]["mean"], "gate scaling", 1e-12)
        if row["epoch"] < result["completed_epochs"]:
            assert row["epoch"] < policy["minimum_epochs"] or stale < policy["patience"]
    assert policy["minimum_epochs"] <= result["completed_epochs"] <= policy["maximum_epochs"]
    assert result["completed_epochs"] == policy["maximum_epochs"] or stale >= policy["patience"]
    return best


def independent_masks(cache, boundaries):
    q, h = cache["quantity"].numpy(), cache["history_length"].numpy()
    p50, p90, p95, p99 = boundaries
    return {"overall": np.ones(len(q), dtype=bool), "le_p50": q <= p50,
        "p50_p90": (q > p50) & (q <= p90), "p90_p95": (q > p90) & (q <= p95),
        "p95_p99": (q > p95) & (q <= p99), "gt_p99": q > p99,
        "body_le_p95": q <= p95, "tail_gt_p95": q > p95,
        "history_le_64": h <= 64, "history_65_128": (h > 64) & (h <= 128), "history_gt_128": h > 128}


def independent_gate_decision(result):
    scopes = result["scopes"]
    def relative(scope, metric):
        row = scopes[scope]
        base, candidate = row["baseline"][metric], row["candidate"][metric]
        return (candidate - base) / base if base else (0 if candidate == 0 else math.inf)
    body = relative("body_le_p95", "qty_mae")
    rmse = relative("overall", "qty_rmse")
    tail = relative("gt_p99", "qty_mae")
    dt = scopes["overall"]["candidate"]["time_nll"] - scopes["overall"]["baseline"]["time_nll"]
    passed = result["best_epoch"] > 0 and body <= -.05 and rmse <= .02 and tail <= .02 and dt <= .01
    assert passed == (result["decision"]["status"] == "exploratory_pass")
    for name, value in (("body_relative_change", body), ("rmse_relative_change", rmse), ("p99_relative_change", tail)):
        close(value, result["decision"][name], name, 1e-12)
    assert result["decision"]["eligible_for_fresh_training"] is False
    return passed


def reconcile_events(events, cache, replay, scopes, boundaries):
    n = len(cache["z"])
    assert events.height == n and events["target_index"].n_unique() == n
    for column in events.columns:
        assert events[column].null_count() == 0 and np.isfinite(events[column].to_numpy()).all()
    for key in ("target_index", "series_index", "context_end", "history_length", "quantity"):
        assert np.array_equal(events[key].to_numpy(), cache[key].numpy()), key
    z, gate, correction = replay
    expected = {"baseline_prediction": F.softplus(cache["z"]).expm1().double().numpy(),
        "candidate_prediction": F.softplus(z).expm1().double().numpy(),
        "gate": gate.numpy(), "logit_correction": correction.numpy()}
    for key, values in expected.items():
        assert np.array_equal(values, events[key].to_numpy()), f"Replay {key}"
    q = cache["quantity"].double().numpy()
    base_error = expected["baseline_prediction"] - q
    error = expected["candidate_prediction"] - q
    np.testing.assert_array_equal(events["absolute_error_delta"].to_numpy(), abs(error) - abs(base_error))
    np.testing.assert_array_equal(events["squared_error_delta"].to_numpy(), error**2 - base_error**2)
    masks = independent_masks(cache, boundaries)
    assert set(masks) == set(scopes)
    quantity_partition = [masks[k] for k in ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")]
    assert np.all(np.stack(quantity_partition).sum(0) == 1)
    assert np.all(np.stack([masks[k] for k in ("history_le_64", "history_65_128", "history_gt_128")]).sum(0) == 1)
    maximum_gap = 0.
    for name, mask in masks.items():
        row = scopes[name]
        assert row["count"] == int(mask.sum())
        if not mask.any():
            assert row["status"] == "empty"
            continue
        assert row["status"] == "evaluated"
        for role, e, logits in (("baseline", base_error, cache["z"]), ("candidate", error, z)):
            # Retain original FP32 elementwise log-MSE; aggregate independently in NumPy.
            squared_log = (F.softplus(logits) - cache["quantity"].clamp_min(0).log1p()).square().double().numpy()
            log_mse = squared_log[mask].mean()
            time_nll = cache["time_nll"].double().numpy()[mask].mean()
            metrics = {"qty_mae": abs(e[mask]).mean(), "qty_rmse": np.sqrt((e[mask]**2).mean()),
                "time_nll": time_nll, "log_qty_mse": log_mse, "joint_objective": log_mse + time_nll}
            for metric, value in metrics.items():
                close(value, row[role][metric], f"{name}/{role}/{metric}")
                maximum_gap = max(maximum_gap, abs(float(value) - row[role][metric]))
        close(expected["gate"][mask].astype(float).mean(), row["gate_mean"], "gate mean", 1e-12)
    return maximum_gap


def main(root, source, output):
    if output.exists():
        raise FileExistsError(output)
    contract = load_json(root / "launch_contract.json")
    assert contract == load_json(ROOT / "paper/contracts/hard_lmm_smooth_shrinkage_v1.json")
    manifest = load_json(root / "source_manifest.json")
    contract_digest = manifest["files"]["paper/contracts/hard_lmm_smooth_shrinkage_v1.json"]
    assert manifest["torch_version"] == "2.7.1" == str(torch.__version__)
    assert manifest["device"] == "cpu" and manifest["threads"] == 1
    verify_hashes(ROOT, manifest["files"])
    verify_hashes(ROOT, manifest["parent_files"])
    for path, digest in manifest["files"].items():
        content = subprocess.check_output(["git", "show", f'{manifest["source_revision"]}:{path}'], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == digest
    for path, digest in load_json(root / "input_digests.json").items():
        assert sha256_file(Path(path)) == digest, path
    digests = load_json(root / "output_digests.json")
    assert set(digests) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        and p.name not in ("probe_status.json", "output_digests.json")}
    for path, digest in digests.items():
        assert sha256_file(root / path) == digest, path
    assert not list(root.rglob("*test*"))
    for path in root.rglob("*.json"):
        finite_json(load_json(path))
    status, summary = load_json(root / "probe_status.json"), load_json(root / "summary.json")
    assert status["status"] == summary["status"] == "complete"
    assert not summary["held_out_test_evaluated"] and not summary["fresh_training_authorized"]
    preflights = load_json(root / "train_only_preflight.json")
    assert set(preflights) == set(summary["datasets"]) == set(contract["datasets"])
    specs = load_json(ROOT / contract["parent_contract"])["datasets"]
    records, diagnostics, comparison, passes, gap, count = [], [], [], [], 0., 0
    for spec in specs:
        name = spec["dataset"]
        validation, audit = read_cache(source, spec, "validation", {})
        train, _ = read_cache(source, spec, "train", {})
        baseline_path = ROOT / spec["artifact_dir"] / "launch_contract.json"
        if not baseline_path.exists() and name == "intermittent_v2":
            baseline_path = ROOT / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/launch_contract.json"
        assert sha256_file(baseline_path) == spec["contract_sha256"]
        boundaries = load_json(baseline_path)["quantity_contract"]["boundaries"]
        dataset = summary["datasets"][name]
        assert dataset == load_json(root / name / "summary.json")
        assert set(preflights[name]) == set(KINDS)
        for kind in KINDS:
            pf = preflights[name][kind]
            assert pf["status"] == "passed" and pf["split"] == "train" and pf["epochs"] == 1
            assert pf["targets"] == len(train["z"]) and not pf["reused_in_main_fit"]
            assert pf["diagnostics"]["first_gradient_norm"] > 0 and pf["parameters_changed"]
            assert pf["maximum_train_gate_change"] > 0
            close(pf["initial_gate_mean"], .99, "matched initial gate", 1e-7)
            result = dataset[kind]
            assert result == load_json(root / name / f"{kind}_summary.json")
            assert result["kind"] == kind and not result["held_out_test_evaluated"]
            assert result["validation_targets"] == len(validation["z"])
            assert result["baseline_state_sha256"] == spec["checkpoint_state_sha256"]
            assert result["boundaries"] == boundaries
            best = check_history(result, contract["training"])
            for metric, value in audit["baseline"].items():
                close(value, result["history"][0][metric], "original validation replay")
            lines = [json.loads(line) for line in (root / name / f"{kind}_history.jsonl").read_text().splitlines()]
            assert lines == result["history"][2:]
            assert all(row["batches"] == math.ceil(len(train["z"]) / 128) for row in lines)
            for tag in ("selected", "final"):
                path = root / name / f"{kind}_{tag}.pt"
                assert sha256_file(path) == result[f"{tag}_checkpoint_sha256"]
                model = restore_checkpoint(path, spec["checkpoint_state_sha256"], contract_digest)
                _, train_gate, _ = predict(model, train)
                expected_gate = result[f"{tag}_train_gate"]
                close(train_gate.double().mean().item(), expected_gate["mean"], "train gate mean", 1e-12)
                close(train_gate.double().std(unbiased=False).item(), expected_gate["std"], "train gate std", 1e-12)
                close((1 - train_gate).double().mean().item(), expected_gate["relative_residual_reduction_mean"], "residual", 1e-12)
                close((train_gate >= 1 - 1e-6).double().mean().item(), expected_gate["near_identity_fraction"], "identity fraction", 0)
                if tag == "selected":
                    replay = predict(model, validation)
                    events = pl.read_parquet(root / name / f"{kind}_validation_events.parquet")
                    gap = max(gap, reconcile_events(events, validation, replay, result["scopes"], result["boundaries"]))
                    for metric in ("qty_mae", "qty_rmse", "log_qty_mse", "time_nll", "joint_objective"):
                        close(result["scopes"]["overall"]["candidate"][metric], best[metric], "selected checkpoint metric", 1e-12)
                else:
                    final_metrics = result["history"][-1]
                    final_z = predict(model, validation)[0]
                    squared = (F.softplus(final_z) - validation["quantity"].clamp_min(0).log1p()).square().double().numpy()
                    close(squared.mean(), final_metrics["log_qty_mse"], "final checkpoint replay", 1e-12)
            independent_gate_decision(result)
            for scope, row in result["scopes"].items():
                if row["status"] == "evaluated":
                    for role in ("baseline", "candidate"):
                        records.append({"dataset": name, "candidate": kind, "scope": scope,
                            "role": role, "count": row["count"], **row[role]})
            overall = result["scopes"]["overall"]
            for role in (("baseline", "candidate") if kind == KINDS[0] else ("candidate",)):
                comparison.append({"dataset": name, "role": "original" if role == "baseline" else kind,
                    "selected_epoch": -1 if role == "baseline" else result["best_epoch"],
                    **overall[role], "body_mae": result["scopes"]["body_le_p95"][role]["qty_mae"],
                    "p99_mae": result["scopes"]["gt_p99"][role]["qty_mae"]})
            diagnostics.append({"dataset": name, "kind": kind, "completed_epochs": result["completed_epochs"],
                "selected_epoch": result["best_epoch"], "total_batches": sum(row["batches"] for row in lines),
                "zero_gradient_batches": sum(row["zero_gradient_batches"] for row in lines),
                "final_gate_mean": result["final_train_gate"]["mean"],
                "final_gate_std": result["final_train_gate"]["std"],
                "final_near_identity_fraction": result["final_train_gate"]["near_identity_fraction"],
                "selected_gate_mean": result["selected_train_gate"]["mean"],
                "final_residual_reduction": result["final_train_gate"]["relative_residual_reduction_mean"],
                "fit_epoch_seconds": sum(row["epoch_seconds"] for row in lines),
                "body_relative_change": result["decision"]["body_relative_change"],
                "rmse_relative_change": result["decision"]["rmse_relative_change"],
                "p99_relative_change": result["decision"]["p99_relative_change"]})
            count += 1
        smooth, scalar = dataset["smooth_shrinkage"], dataset["constant_shrinkage"]
        adaptive = independent_gate_decision(smooth) and (
            smooth["scopes"]["body_le_p95"]["candidate"]["qty_mae"] < scalar["scopes"]["body_le_p95"]["candidate"]["qty_mae"]
            and smooth["scopes"]["overall"]["candidate"]["joint_objective"] <= scalar["scopes"]["overall"]["candidate"]["joint_objective"])
        assert adaptive == dataset["adaptive_decision"]["passes"]
        passes.append(adaptive)
    observed = pl.read_csv(root / "scope_metrics.csv").sort(["dataset", "candidate", "scope", "role"])
    expected = pl.DataFrame(records).select(observed.columns).sort(["dataset", "candidate", "scope", "role"])
    assert observed.equals(expected)
    assert count == summary["completed_fits"] == 8
    assert sum(passes) == summary["adaptive_value_passes"] and all(passes) == summary["cross_dataset_support"]
    assert sum(independent_gate_decision(v["smooth_shrinkage"]) for v in summary["datasets"].values()) == summary["original_body_tail_passes"]
    output.mkdir(parents=True)
    pl.DataFrame(comparison).write_csv(output / "comparison.csv")
    pl.DataFrame(diagnostics).write_csv(output / "gate_diagnostics.csv")
    observed.write_csv(output / "scope_metrics.csv")
    verification = {"status": "verified", "fits": count, "preflights": count,
        "source_revision": manifest["source_revision"], "parent_files_verified": len(manifest["parent_files"]),
        "source_input_output_and_checkpoint_digests_verified": True,
        "full_validation_and_train_subset_verified": True, "selection_and_early_stopping_verified": True,
        "selected_and_final_checkpoint_replay_verified": True,
        "all_metrics_finite": True, "unchanged_time_head_verified": True,
        "numpy_scope_metric_max_abs_gap": gap, "held_out_artifact_absent": True,
        "held_out_test_evaluated": False, "fresh_training_authorized": False,
        "assessment": "share_with_caveats_frozen_single_seed_known_validation",
        "validator_sha256": sha256_file(Path(__file__))}
    save_json(output / "artifact_verification.json", verification)
    for name in ("launch_contract", "source_manifest", "input_digests", "output_digests",
                 "probe_status", "train_only_preflight", "summary"):
        save_json(output / f"{name}.json", load_json(root / f"{name}.json"))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.artifact.resolve(), args.source_artifact.resolve(), args.output.resolve())
