#!/usr/bin/env python3
"""Independent provenance, checkpoint, selection and event-level reconciliation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.hard_lmm_frozen_probe import predict
from paper.scripts.run_hard_lmm_frozen_probe import load_json, save_json, sha256_file, verify_hashes
from paper.scripts.run_hard_lmm_readout_factorial import (
    CONTRACT_PATH, HEADS, OBJECTIVES, SELECTORS, baseline_contract_path, evaluate,
    new_readout, read_cache, restore_checkpoint,
)
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json
from paper.scripts.validate_hard_lmm_smooth_shrinkage import (
    close, independent_gate_decision, reconcile_events,
)


def check_history(result, policy, targets):
    history = result["history"]
    assert result["completed_epochs"] == policy["maximum_epochs"]
    assert [r["epoch"] for r in history] == list(range(policy["maximum_epochs"] + 1))
    for row in history:
        finite_json(row)
        close(row["time_nll"], history[0]["time_nll"], "unchanged validation time", 0)
        close(row["train"]["time_nll"], history[0]["train"]["time_nll"], "unchanged train time", 0)
        close(row["joint_objective"], row["time_nll"] + row["log_qty_mse"], "joint formula", 1e-12)
        if row["epoch"]:
            assert row["batches"] == math.ceil(targets / policy["batch_size"])
            assert 0 <= row["zero_gradient_batches"] <= row["batches"]
            for tag, key in SELECTORS.items():
                expected = min(history[:row["epoch"] + 1], key=lambda r: r[key])["epoch"]
                assert row["best_epochs"][tag] == expected
    best = {}
    for tag, metric in SELECTORS.items():
        best[tag] = min(history, key=lambda r: r[metric])
        assert best[tag]["epoch"] == result["best_epochs"][tag] == result["selections"][tag]["best_epoch"]
    return best


def contrasts(rows):
    table = {(r["dataset"], r["selector"], r["head"], r["objective"]): r for r in rows}
    output = []
    metrics = ("body_mae", "qty_mae", "qty_rmse", "p99_mae", "log_qty_mse", "joint_objective")
    for dataset, selector in itertools.product(sorted({r["dataset"] for r in rows}), SELECTORS):
        pairs = []
        for objective in OBJECTIVES:
            for head in ("linear", "mlp"):
                pairs.append(("features_vs_constant", (head, objective), ("constant", objective)))
            pairs.append(("capacity_mlp_vs_linear", ("mlp", objective), ("linear", objective)))
        for head in HEADS:
            pairs.append(("objective_mae_vs_logmse", (head, "raw_mae"), (head, "log_mse")))
        for label, candidate, control in pairs:
            c, b = table[dataset, selector, *candidate], table[dataset, selector, *control]
            output.append({"dataset": dataset, "selector": selector, "contrast": label,
                "candidate": "_".join(candidate), "control": "_".join(control),
                **{f"{m}_delta": c[m] - b[m] for m in metrics}})
    return output


def main(root, output):
    if output.exists():
        raise FileExistsError(output)
    contract = load_json(root / "launch_contract.json")
    assert contract == load_json(CONTRACT_PATH)
    manifest = load_json(root / "source_manifest.json")
    assert manifest["torch_version"] == str(torch.__version__) == "2.7.1"
    assert manifest["device"] == "cpu" and manifest["threads"] == 1
    verify_hashes(ROOT, manifest["files"])
    verify_hashes(ROOT, manifest["parent_files"])
    for path, digest in manifest["files"].items():
        assert hashlib.sha256(subprocess.check_output(["git", "show", f'{manifest["source_revision"]}:{path}'], cwd=ROOT)).hexdigest() == digest
    for path, digest in load_json(root / "input_digests.json").items():
        assert sha256_file(Path(path)) == digest
    digests = load_json(root / "output_digests.json")
    assert set(digests) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        and p.name not in ("probe_status.json", "output_digests.json")}
    for path, digest in digests.items():
        assert sha256_file(root / path) == digest
    assert not list(root.rglob("*test*"))
    for path in root.rglob("*.json"):
        finite_json(load_json(path))
    summary = load_json(root / "summary.json")
    assert summary["status"] == load_json(root / "probe_status.json")["status"] == "complete"
    assert summary["completed_fits"] == 24 and not summary["held_out_test_evaluated"] and not summary["fresh_training_authorized"]
    specs = load_json(ROOT / contract["parent_contract"])["datasets"]
    source = ROOT / contract["source_artifact"]
    pf = load_json(root / "train_only_preflight.json")
    assert set(summary["datasets"]) == set(pf) == set(contract["datasets"])
    records, scoped, gap, count, total_batches, zero_batches = [], [], 0., 0, 0, 0
    for spec in specs:
        name = spec["dataset"]
        train, _ = read_cache(source, spec, "train", {})
        validation, audit = read_cache(source, spec, "validation", {})
        boundaries = load_json(baseline_contract_path(spec))["quantity_contract"]["boundaries"]
        cells = {f"{h}_{o}" for h, o in itertools.product(HEADS, OBJECTIVES)}
        assert set(summary["datasets"][name]) == set(pf[name]) == cells
        for head, objective in itertools.product(HEADS, OBJECTIVES):
            cell = f"{head}_{objective}"
            result = load_json(root / name / f"{cell}_summary.json")
            assert result == summary["datasets"][name][cell]
            assert result["head"] == head and result["objective"] == objective
            assert result["validation_targets"] == len(validation["z"]) and result["train_targets"] == len(train["z"])
            assert result["baseline_state_sha256"] == spec["checkpoint_state_sha256"] and result["boundaries"] == boundaries
            assert not result["held_out_test_evaluated"]
            pre = pf[name][cell]
            assert pre["status"] == "passed" and pre["split"] == "train" and pre["targets"] == len(train["z"])
            assert pre["epochs"] == 1 and pre["initialization_exact"] and pre["parameters_changed"]
            assert not pre["reused_in_main_fit"] and pre["max_logit_change"] > 0 and pre["diagnostics"]["first_gradient_norm"] > 0
            best = check_history(result, contract["training"], len(train["z"]))
            history = [json.loads(line) for line in (root / name / f"{cell}_history.jsonl").read_text().splitlines()]
            assert history == result["history"][1:]
            total_batches += sum(r["batches"] for r in history)
            zero_batches += sum(r["zero_gradient_batches"] for r in history)
            initial = new_readout(head, train)
            assert result["trainable_parameters"] == sum(p.numel() for p in initial.parameters())
            assert torch.equal(predict(initial, validation)[0], validation["z"])
            for metric, value in audit["baseline"].items():
                close(result["history"][0][metric], value, "parent baseline replay")
            for tag in (*SELECTORS, "final"):
                path = root / name / f"{cell}_{tag}.pt"
                assert sha256_file(path) == result[f"{tag}_checkpoint_sha256"]
                model = restore_checkpoint(path, spec["checkpoint_state_sha256"], sha256_file(CONTRACT_PATH))
                assert torch.equal(initial.center, model.center) and torch.equal(initial.scale, model.scale)
                expected = result["history"][-1] if tag == "final" else best[tag]
                for cache, expected_metrics in ((validation, expected), (train, expected["train"])):
                    for metric, value in evaluate(model, cache, boundaries[2]).items():
                        close(value, expected_metrics[metric], f"{tag}/{metric}", 1e-12)
                if tag == "final":
                    continue
                selected = result["selections"][tag]
                for metric, value in selected["train"].items():
                    close(value, expected["train"][metric], "selected train", 1e-12)
                replay = predict(model, validation)
                gap = max(gap, reconcile_events(pl.read_parquet(root / name / f"{cell}_{tag}_events.parquet"),
                    validation, replay, selected["scopes"], boundaries))
                passed = independent_gate_decision(selected)
                overall = selected["scopes"]["overall"]["candidate"]
                decision = selected["decision"]
                records.append({"dataset": name, "selector": tag, "head": head, "objective": objective,
                    "selected_epoch": selected["best_epoch"], "parameters": result["trainable_parameters"],
                    **overall, "body_mae": selected["scopes"]["body_le_p95"]["candidate"]["qty_mae"],
                    "p99_mae": selected["scopes"]["gt_p99"]["candidate"]["qty_mae"],
                    "body_change_pct": 100 * decision["body_relative_change"],
                    "rmse_change_pct": 100 * decision["rmse_relative_change"],
                    "p99_change_pct": 100 * decision["p99_relative_change"], "gate_pass": passed})
                for scope, values in selected["scopes"].items():
                    if values["status"] == "evaluated":
                        for role in ("baseline", "candidate"):
                            scoped.append({"dataset": name, "selector": tag, "cell": cell, "scope": scope,
                                "role": role, "count": values["count"], **values[role]})
            count += 1
    assert count == 24 and len(records) == 48
    universal = []
    for selector, head, objective in itertools.product(SELECTORS, HEADS, OBJECTIVES):
        matches = [r for r in records if (r["selector"], r["head"], r["objective"]) == (selector, head, objective)]
        universal.append({"selector": selector, "head": head, "objective": objective,
            "datasets_passed": sum(r["gate_pass"] for r in matches), "all_four_pass": all(r["gate_pass"] for r in matches)})
    output.mkdir(parents=True)
    pl.DataFrame(records).write_csv(output / "comparison.csv")
    pl.DataFrame(scoped).write_csv(output / "scope_metrics.csv")
    pl.DataFrame(contrasts(records)).write_csv(output / "factorial_contrasts.csv")
    save_json(output / "cross_dataset_decision.json", {"registered_cells": universal, "fresh_training_authorized": False})
    verification = {"status": "verified", "fits": count, "train_only_preflights": 24,
        "selected_checkpoint_reconciliations": 48, "final_checkpoint_reconciliations": 24,
        "source_revision": manifest["source_revision"], "parent_files_verified": len(manifest["parent_files"]),
        "source_input_output_checkpoint_hashes_verified": True,
        "full_validation_and_train_subset_verified": True, "train_only_normalizers_verified": True,
        "fixed_40_epochs_and_both_selection_rules_verified": True,
        "all_metrics_finite": True, "unchanged_time_scores_verified": True,
        "training_batches": total_batches, "zero_gradient_batches": zero_batches,
        "independent_numpy_max_metric_gap": gap, "held_out_artifact_absent": True,
        "assessment": "exploratory_single_seed_compressed_cache_known_validation",
        "validator_sha256": sha256_file(Path(__file__))}
    save_json(output / "artifact_verification.json", verification)
    for filename in ("source_manifest", "launch_contract", "input_digests", "output_digests", "probe_status", "train_only_preflight"):
        save_json(output / f"{filename}.json", load_json(root / f"{filename}.json"))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.artifact.resolve(), args.output.resolve())
