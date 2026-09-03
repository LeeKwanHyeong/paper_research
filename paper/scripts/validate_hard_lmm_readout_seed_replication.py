#!/usr/bin/env python3
"""Reconcile frozen-seed replications and report paired, not pooled, evidence."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import subprocess
import sys

import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.hard_lmm_frozen_probe import predict
from paper.scripts.run_hard_lmm_frozen_probe import load_json, save_json, sha256_file, verify_hashes
from paper.scripts.run_hard_lmm_readout_factorial import SELECTORS, evaluate, new_readout, restore_checkpoint
from paper.scripts.run_hard_lmm_readout_seed_replication import (
    CONTRACT_PATH, checked_model, inherited_contract, read_reference, require_frozen, verify_alignment,
)
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json
from paper.scripts.validate_hard_lmm_readout_factorial import check_history
from paper.scripts.validate_hard_lmm_smooth_shrinkage import close, independent_gate_decision, reconcile_events


def replication_decision(rows):
    by_seed = {r["seed"]: r for r in rows}
    if len(rows) != 3 or set(by_seed) != {42, 52, 62}:
        raise ValueError("Exactly one row per backbone seed is required")
    return {"new_seeds_passed": sum(by_seed[s]["gate_pass"] for s in (52, 62)),
        "replicated_on_both_new_seeds": all(by_seed[s]["gate_pass"] for s in (52, 62)),
        "all_three_pass": all(r["gate_pass"] for r in rows), "fresh_training_authorized": False}


def paired_summary(rows):
    replication_decision(rows)
    metrics = ("qty_mae", "qty_rmse", "body_mae", "p99_mae", "time_nll", "joint_objective")
    summary = {"seeds": 3, "std_ddof": 1}
    for metric in metrics:
        original = [r[f"baseline_{metric}"] for r in rows]
        candidate = [r[metric] for r in rows]
        for label, values in ((f"baseline_{metric}", original), (metric, candidate),
                              (f"paired_{metric}_delta", [b-a for a, b in zip(original, candidate)])):
            summary[f"{label}_mean"] = statistics.mean(values)
            summary[f"{label}_std"] = statistics.stdev(values)
    return summary


def verify_cache(root, contract, seed, split, reference):
    if split not in ("train", "validation"):
        raise ValueError("Held-out split forbidden")
    directory = root / f"seed_{seed}"
    row = next(r for r in contract["checkpoints"] if r["seed"] == seed)
    audit = load_json(directory / "baseline_audit.json")
    assert audit["model_state_sha256"] == row["checkpoint_state_sha256"]
    assert audit["checkpoint_file_sha256"] == row["checkpoint_file_sha256"]
    assert audit["seed"] == seed and audit["frozen_base_verified"] and not audit["held_out_test_evaluated"]
    path = directory / f"{split}_cache.pt"
    assert sha256_file(path) == audit["cache"][split]["sha256"]
    cache = torch.load(path, map_location="cpu", weights_only=True)
    verify_alignment(cache, reference)
    assert cache["features"].shape == (len(reference["z"]), 138)
    for value in cache.values():
        assert torch.isfinite(value).all()
    index_hash = hashlib.sha256(cache["target_index"].numpy().tobytes()).hexdigest()
    assert index_hash == audit["cache"][split]["sample_indices_sha256"]
    assert len(cache["z"]) == audit["cache"][split]["selected_targets"]
    return cache


def main(root, output):
    if output.exists():
        raise FileExistsError(output)
    contract = load_json(root / "launch_contract.json")
    assert contract == load_json(CONTRACT_PATH)
    parent = inherited_contract(contract, {})
    assert parent == load_json(root / "inherited_contract.json")
    manifest = load_json(root / "source_manifest.json")
    assert manifest["torch_version"] == str(torch.__version__) == "2.7.1"
    assert manifest["device"] == "cpu" and manifest["threads"] == 1
    verify_hashes(ROOT, {**manifest["files"], **manifest["parent_files"]})
    for path, digest in manifest["files"].items():
        assert hashlib.sha256(subprocess.check_output(["git", "show", f'{manifest["source_revision"]}:{path}'], cwd=ROOT)).hexdigest() == digest
    for path, digest in load_json(root / "input_digests.json").items():
        assert sha256_file(Path(path)) == digest
    digests = load_json(root / "output_digests.json")
    assert set(digests) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        and p.name not in ("probe_status.json", "output_digests.json")}
    verify_hashes(root, digests)
    assert not list(root.rglob("*test*"))
    for path in root.rglob("*.json"):
        finite_json(load_json(path))
    summary = load_json(root / "summary.json")
    assert summary["status"] == load_json(root / "probe_status.json")["status"] == "complete"
    assert summary["new_fits"] == 8 and summary["reused_seed42_fits"] == 4
    assert not summary["held_out_test_evaluated"] and not summary["fresh_training_authorized"]
    assert set(summary["seeds"]) == {"42", "52", "62"}
    preflights = load_json(root / "train_only_preflight.json")
    assert set(preflights) == {"52", "62"}
    reference_train, _ = read_reference(contract, "train", {})
    reference_val, _ = read_reference(contract, "validation", {})
    launch = load_json(ROOT / contract["baseline_artifact"] / "launch_contract.json")
    boundaries = launch["quantity_contract"]["boundaries"]
    records, scoped, gap, batches, zero = [], [], 0., 0, 0
    for seed in (42, 52, 62):
        if seed == 42:
            directory = ROOT / contract["parent_artifact"] / "intermittent_v2"
            base_digest = load_json(ROOT / contract["parent_feature_artifact"] / "baseline_audit.json")["model_state_sha256"]
            contract_digest = contract["parent_contract_sha256"]
            train, validation = reference_train, reference_val
        else:
            directory = root / f"seed_{seed}"
            row = next(r for r in contract["checkpoints"] if r["seed"] == seed)
            base_digest, contract_digest = row["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH)
            model, _ = checked_model(contract, row, launch)
            require_frozen(model, row)
            train = verify_cache(root, contract, seed, "train", reference_train)
            validation = verify_cache(root, contract, seed, "validation", reference_val)
            assert (root / "train_only_preflight.json").stat().st_mtime <= (directory / "validation_cache.pt").stat().st_mtime
        cells = list(itertools.product(contract["heads"], contract["objectives"]))
        assert set(summary["seeds"][str(seed)]) == {f"{h}_{o}" for h, o in cells}
        if seed != 42:
            assert set(preflights[str(seed)]) == set(summary["seeds"][str(seed)])
        for head, objective in cells:
            cell = f"{head}_{objective}"
            result = load_json(directory / f"{cell}_summary.json")
            assert result == summary["seeds"][str(seed)][cell]
            assert result["head"] == head and result["objective"] == objective and result["boundaries"] == boundaries
            assert result["baseline_state_sha256"] == base_digest and not result["held_out_test_evaluated"]
            assert result["train_targets"] == len(train["z"]) and result["validation_targets"] == len(validation["z"])
            best = check_history(result, parent["training"], len(train["z"]))
            history = [json.loads(line) for line in (directory / f"{cell}_history.jsonl").read_text().splitlines()]
            assert history == result["history"][1:]
            if seed != 42:
                assert result["backbone_seed"] == seed
                pf = preflights[str(seed)][cell]
                assert pf["status"] == "passed" and pf["split"] == "train" and pf["epochs"] == 1
                assert pf["targets"] == len(train["z"]) and pf["initialization_exact"] and pf["parameters_changed"]
                assert pf["max_logit_change"] > 0 and pf["diagnostics"]["first_gradient_norm"] > 0 and not pf["reused_in_main_fit"]
                batches += sum(r["batches"] for r in history)
                zero += sum(r["zero_gradient_batches"] for r in history)
            initial = new_readout(head, train)
            assert torch.equal(predict(initial, validation)[0], validation["z"])
            assert result["trainable_parameters"] == sum(p.numel() for p in initial.parameters())
            for tag in (*SELECTORS, "final"):
                path = directory / f"{cell}_{tag}.pt"
                assert sha256_file(path) == result[f"{tag}_checkpoint_sha256"]
                model = restore_checkpoint(path, base_digest, contract_digest)
                assert torch.equal(model.center, initial.center) and torch.equal(model.scale, initial.scale)
                expected = result["history"][-1] if tag == "final" else best[tag]
                for cache, values in ((validation, expected), (train, expected["train"])):
                    for metric, value in evaluate(model, cache, boundaries[2]).items():
                        close(value, values[metric], f"{seed}/{tag}/{metric}", 1e-12)
                if tag == "final":
                    continue
                selected = result["selections"][tag]
                gap = max(gap, reconcile_events(pl.read_parquet(directory / f"{cell}_{tag}_events.parquet"),
                    validation, predict(model, validation), selected["scopes"], boundaries))
                passed = independent_gate_decision(selected)
                decision, scopes = selected["decision"], selected["scopes"]
                values = {**scopes["overall"]["candidate"],
                    "body_mae": scopes["body_le_p95"]["candidate"]["qty_mae"], "p99_mae": scopes["gt_p99"]["candidate"]["qty_mae"]}
                base = {**scopes["overall"]["baseline"],
                    "body_mae": scopes["body_le_p95"]["baseline"]["qty_mae"], "p99_mae": scopes["gt_p99"]["baseline"]["qty_mae"]}
                records.append({"seed": seed, "evidence_role": "discovery" if seed == 42 else "replication",
                    "head": head, "objective": objective, "selector": tag, "selected_epoch": selected["best_epoch"],
                    **values, **{f"baseline_{k}": v for k, v in base.items()},
                    "body_change_pct": decision["body_relative_change"] * 100,
                    "rmse_change_pct": decision["rmse_relative_change"] * 100,
                    "p99_change_pct": decision["p99_relative_change"] * 100, "gate_pass": passed})
                for scope, item in scopes.items():
                    if item["status"] == "evaluated":
                        for role in ("baseline", "candidate"):
                            scoped.append({"seed": seed, "cell": cell, "selector": tag, "scope": scope,
                                "role": role, "count": item["count"], **item[role]})
    aggregate, decisions = [], []
    for head, objective, selector in itertools.product(contract["heads"], contract["objectives"], SELECTORS):
        group = [r for r in records if (r["head"], r["objective"], r["selector"]) == (head, objective, selector)]
        keys = {"head": head, "objective": objective, "selector": selector}
        aggregate.append({**keys, **paired_summary(group)})
        decisions.append({**keys, **replication_decision(group)})
    scope_table = pl.DataFrame(scoped)
    metrics = ["qty_mae", "qty_rmse", "log_qty_mse", "time_nll", "joint_objective"]
    scope_summary = scope_table.group_by("cell", "selector", "scope", "role").agg(
        pl.col("seed").n_unique().alias("seeds"),
        *[expr for m in metrics for expr in (pl.col(m).mean().alias(f"{m}_mean"), pl.col(m).std(ddof=1).alias(f"{m}_std"))])
    assert scope_summary["seeds"].eq(3).all()
    output.mkdir(parents=True)
    pl.DataFrame(records).write_csv(output / "seed_comparison.csv")
    pl.DataFrame(aggregate).write_csv(output / "three_seed_summary.csv")
    scope_table.write_csv(output / "seed_scope_metrics.csv")
    scope_summary.sort("cell", "selector", "scope", "role").write_csv(output / "scope_summary.csv")
    save_json(output / "replication_decision.json", {"cells": decisions, "fresh_training_authorized": False})
    verification = {"status": "verified", "new_fits": 8, "reused_discovery_fits": 4,
        "checkpoint_replays": 36, "event_scope_reconciliations": 24, "source_revision": manifest["source_revision"],
        "locked_source_files": len(manifest["parent_files"]), "source_input_output_checkpoint_hashes_verified": True,
        "train_sample_and_full_validation_alignment_verified": True, "base_checkpoints_frozen_verified": True,
        "inherited_training_and_both_selectors_verified": True, "unchanged_time_scores_verified": True,
        "all_metrics_finite": True, "new_training_batches": batches, "zero_gradient_batches": zero,
        "independent_numpy_max_metric_gap": gap, "aggregation": "unweighted_seed_mean_sample_std_ddof1",
        "held_out_artifact_absent": True, "held_out_test_evaluated": False,
        "assessment": "checkpoint_seed_replication_on_known_validation_not_independent_data_confirmation"}
    save_json(output / "artifact_verification.json", verification)
    for name in ("source_manifest", "launch_contract", "inherited_contract", "input_digests", "output_digests", "probe_status", "train_only_preflight"):
        save_json(output / f"{name}.json", load_json(root / f"{name}.json"))
    for seed in (52, 62):
        save_json(output / f"baseline_audit_seed{seed}.json", load_json(root / f"seed_{seed}" / "baseline_audit.json"))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.artifact.resolve(), args.output.resolve())
