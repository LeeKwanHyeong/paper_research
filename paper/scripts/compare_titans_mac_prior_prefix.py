#!/usr/bin/env python3
"""Compare complete validation runs against the pre-training, three-dataset gates."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts.titans_mac_prior_prefix_contract import (
    BACKBONE, BASELINE, CONTRACT_PATH, ROLE, digest, load_contract, metric_gates, summary_metrics,
)

VARIANT = "count_only_log_regression"
METRICS = ("body_mae", "qty_mae", "qty_rmse", "tail_mae", "time_loss", "log_qty_mse")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_run(run_root, dataset, backbone, seed, contract):
    outer = Path(run_root) / dataset / backbone / f"seed_{seed}"
    leaf = outer / "runs" / backbone / VARIANT / f"seed_{seed}"
    path = leaf / "summary.json"
    summary = read(path)
    launch = read(outer / "launch_contract.json")
    history_path = leaf / "history.json"
    history = read(history_path)["history"]
    row = contract["datasets"][dataset]
    revision = summary.get("source_revision", "")
    require(bool(re.fullmatch(r"[0-9a-f]{40}", revision)), "Missing full source revision")
    require(summary.get("status") == "success" and summary.get("backbone") == backbone
            and summary.get("seed") == seed and summary.get("variant") == VARIANT,
            "Summary model, seed, variant or completion mismatch")
    require(summary.get("source_revision_history") == [revision], "Resumed or mixed source run")
    require(summary.get("evaluation_scope") == "validation_only"
            and summary.get("held_out_test_evaluated") is False, "Evaluation scope changed")
    for key, expected in {
        "status": "complete", "completed_run_count": 1, "dataset": dataset,
        "expected_run_count": 1, "quantity_variants": [VARIANT], "max_series": None,
        "model_role": ROLE if backbone == BACKBONE else "experimental",
        "backbones": [backbone], "seeds": [seed], "epochs": 300,
        "source_revision": revision, "partial_smoke": False,
        "held_out_test_evaluated": False, "evaluation_scope": "validation_only",
        "batch_size": 128, "lr": .001, "hidden_dim": 64,
        "grad_clip": 1., "titans_memory_gradient_clip": 1.,
        "lambda_log_qty": 1., "lambda_tail": 0.,
        "lookback_weeks": row["lookback"], "max_seq_len": row["max_seq_len"],
        "data_sha256": row["data_sha256"], "split_manifest_sha256": row["split_manifest_sha256"],
    }.items():
        require(launch.get(key) == expected, f"Launch mismatch: {dataset}/{backbone}/{seed}/{key}")
    require(set(launch.get("split_rows", {})) == {"train", "validation"}, "Unexpected materialized split")
    require(launch["early_stopping"]["min_epochs"] == 40
            and launch["early_stopping"]["patience"] == 40, "Checkpoint policy drift")
    for key, expected in {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10 / 3,
                          "time_intercept_limit": 300., "time_head_lr_multiplier": 1.,
                          "statistics_source_split": "train"}.items():
        require(launch["time_head"].get(key) == expected, f"Shared time head mismatch: {key}")
    quantity_contract = launch["quantity_contract"]
    require(quantity_contract.get("quantiles") == [.5, .9, .95, .99]
            and len(quantity_contract.get("boundaries", [])) == 4
            and quantity_contract["boundaries"][2:] == [row["train_p95_quantity"], row["train_p99_quantity"]],
            "Train quantity threshold drift")
    encoder = summary["encoder_config"]
    require(encoder.get("mac_execution_backend") == "optimized"
            and encoder.get("titans_memory_gradient_clip") == 1., "B1 stability/execution policy drift")
    if backbone == BACKBONE:
        require(encoder.get("titans_prefix_read_contract_id") == contract["contract_id"]
                and encoder.get("titans_output_read_policy") == "prior_prefix"
                and encoder.get("memory_mode") == "titans_mac_prior_prefix", "Candidate read policy drift")
    else:
        require(encoder.get("memory_mode") == "titans_mac"
                and encoder.get("titans_output_read_policy", "segment_start") == "segment_start",
                "Baseline read policy drift")
    require(40 <= len(history) <= 300 and summary["completed_epochs"] == len(history), "Incomplete screening history")
    require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "Nonsequential training")
    require(all(h["train_event_count"] == row["train_targets"]
                and h["train_batch_count"] == math.ceil(row["train_targets"] / 128)
                and h["train_all_finite"] is True and math.isfinite(h["val_joint_objective"])
                for h in history), "Partial or nonfinite training")
    best = min(history, key=lambda h: h["val_joint_objective"])
    require(summary["best_epoch"] == best["epoch"]
            and summary["best_val_joint_objective"] == best["val_joint_objective"], "Checkpoint reselection detected")
    for summary_key, history_key in {
        "best_val_qty_mae": "val_qty_mae", "best_val_qty_rmse": "val_qty_rmse",
        "best_val_time_nll": "val_time_nll", "best_val_log_qty_mse": "val_log_qty_mse",
    }.items():
        # Validation is replayed once after restoring the selected checkpoint.
        # This provenance check tolerates that replay's arithmetic; gate inputs
        # remain the original unrounded summary metrics, with no gate tolerance.
        require(math.isclose(summary[summary_key], best[history_key], rel_tol=1e-7, abs_tol=1e-8),
                f"Summary metric does not belong to the selected epoch: {summary_key}")
    require(len(history) == 300 or len(history) - best["epoch"] >= 40, "Premature training stop")
    metrics = summary_metrics(summary)
    require(metrics["validation_targets"] == row["validation_targets"], "Validation coverage drift")
    require(sum(r["count"] for r in summary["history_rows"]) == row["validation_targets"], "History coverage drift")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", summary.get("checkpoint_state_sha256", ""))), "Checkpoint hash missing")
    evidence = {str(p.resolve()): digest(p) for p in (path, history_path, outer / "launch_contract.json")}
    return {"metrics": metrics, "source_revision": revision, "checkpoint_state_sha256": summary["checkpoint_state_sha256"],
            "best_epoch": summary["best_epoch"], "completed_epochs": len(history), "evidence_files": evidence}


def means(rows):
    """Arithmetic means across paired seeds, never pooled targets or datasets."""
    return {key: statistics.mean(row[key] for row in rows) for key in METRICS}


def assess_dataset(rows, references, contract, dataset, seeds):
    paired = {}
    for seed in seeds:
        candidate, baseline = rows[(dataset, BACKBONE, seed)]["metrics"], rows[(dataset, BASELINE, seed)]["metrics"]
        t0 = references[dataset]["titantpp"][str(seed)]
        external = {b: references[dataset][b][str(seed)] for b in ("rmtpp", "thp")}
        for metrics in (candidate, baseline, *external.values()):
            require(all(metrics[key] == t0[key] for key in (
                "validation_targets", "body_targets", "tail_targets")),
                "Paired or historical quantity strata cover different targets")
        decision = metric_gates(candidate, baseline, t0, external, contract, dataset)
        paired[str(seed)] = {"candidate": candidate, "B1": baseline, "T0": t0, "external": external, **decision}
    mean_candidate = means([v["candidate"] for v in paired.values()])
    mean_baseline = means([v["B1"] for v in paired.values()])
    mean_t0 = means([v["T0"] for v in paired.values()])
    mean_external = {b: means([v["external"][b] for v in paired.values()]) for b in ("rmtpp", "thp")}
    aggregate = metric_gates(mean_candidate, mean_baseline, mean_t0, mean_external, contract, dataset)
    wins = {ref: sum(v["candidate"]["qty_rmse"] < v[ref]["qty_rmse"] for v in paired.values()) for ref in ("B1", "T0")}
    rails = all(v["checks"]["B1_incremental"]["tail_preserved"]
                and v["checks"]["B1_incremental"]["time_preserved"]
                and v["checks"]["T0_legacy_gate"]["tail_guardrail"]
                and v["checks"]["T0_legacy_gate"]["time_guardrail"] for v in paired.values())
    return {"paired_seeds": paired, "mean_candidate": mean_candidate, "mean_B1": mean_baseline,
            "mean_T0": mean_t0, "mean_external": mean_external, "mean_decision": aggregate,
            "strict_rmse_seed_wins": wins, "individual_seed_time_tail_guardrails_pass": rails,
            "screening_pass": paired.get("42", {}).get("screening_pass", False),
            "confirmation_pass": len(seeds) == 3 and paired.get("42", {}).get("screening_pass", False)
                                 and aggregate["screening_pass"]
                                 and all(n >= 2 for n in wins.values()) and rails}


def compare(contract_path, run_root, phase, screening_run_root=None):
    contract = load_contract(Path(contract_path))
    require(phase in {"screening", "confirm"}, "Unknown comparison phase")
    require(phase != "confirm" or screening_run_root is not None, "Confirmation requires seed42 screening root")
    require(len(contract["dataset_order"]) == len(contract["datasets"])
            and set(contract["dataset_order"]) == set(contract["datasets"]),
            "Comparison requires each frozen dataset exactly once")
    reference_path = ROOT / contract["frozen_references"]["path"]
    require(digest(reference_path) == contract["frozen_references"]["sha256"], "Frozen references changed")
    frozen = read(reference_path)
    require(frozen["effective_time_intercept_cap"] == 300, "Historical head contract mismatch")
    references = frozen["references"]
    require(set(references) == set(contract["datasets"]), "Frozen reference dataset grid changed")
    for dataset, models in references.items():
        require(set(models) == {"titantpp", "rmtpp", "thp"}, "Frozen reference model grid changed")
        for model_rows in models.values():
            require(set(model_rows) == {"42", "52", "62"}, "Frozen reference paired-seed grid changed")
            for metrics in model_rows.values():
                require(all(math.isfinite(float(metrics[key])) for key in METRICS),
                        "Nonfinite frozen reference metric")
                require(metrics["validation_targets"] == contract["datasets"][dataset]["validation_targets"],
                        "Frozen reference validation coverage mismatch")
    seeds = (42,) if phase == "screening" else (42, 52, 62)
    rows = {}
    for dataset in contract["dataset_order"]:
        for seed in seeds:
            source = screening_run_root if phase == "confirm" and seed == 42 else run_root
            for backbone in (BASELINE, BACKBONE):
                rows[(dataset, backbone, seed)] = load_run(source, dataset, backbone, seed, contract)
    revisions = {v["source_revision"] for v in rows.values()}
    require(len(revisions) == 1, "Paired comparison requires one source revision")
    datasets = {d: assess_dataset(rows, references, contract, d, seeds) for d in contract["dataset_order"]}
    evidence = {str(reference_path.resolve()): digest(reference_path)}
    for row in rows.values():
        evidence.update(row["evidence_files"])
    result = {"status": "PASS", "execution_status": "complete", "phase": phase,
              "source_revision": revisions.pop(), "contract_sha256": digest(Path(contract_path)),
              "run_root": str(Path(run_root).resolve()), "held_out_test_evaluated": False,
              "screening_pass": all(v["screening_pass"] for v in datasets.values()),
              "datasets": datasets, "evidence_files": evidence,
              "external_TPP_claim_pass": all(v["mean_decision"]["external_TPP_claim_pass"] for v in datasets.values()),
              "comparison_scope": "historical frozen T0/RMTPP/THP references; fresh paired stable B1 controls; validation only",
              "statistical_claim": contract["performance_gates"]["statistical_claim"]}
    if phase == "confirm":
        result["confirmation_pass"] = all(v["confirmation_pass"] for v in datasets.values())
        result["screening_run_root"] = str(Path(screening_run_root).resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("screening", "confirm"), required=True)
    parser.add_argument("--screening-run-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare(args.contract, args.run_root, args.phase, args.screening_run_root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        result = {"status": "FAIL", "phase": args.phase, "error": f"{type(error).__name__}: {error}",
                  "screening_pass": False, "held_out_test_evaluated": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"datasets", "evidence_files"}}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
