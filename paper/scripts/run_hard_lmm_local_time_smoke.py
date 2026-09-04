#!/usr/bin/env python3
"""CUDA contracts followed by full Taxi/RAF e1 only; never performance screening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_weighted_static import (
    baseline, digest, finite, preflight, read, require, run_logged, save,
    validate_quantity_initialization, verify_source,
)

BACKBONE = "titantpp_hard_memory_local_time"
ROLE = "t0_hard_memory_local_time"
VARIANT = "count_only_log_regression"
CONTRACT = ROOT / "paper/contracts/hard_lmm_local_time_v1.json"
CONTRACT_SHA256 = "63ba706026c22b2052c04bc06b27fb7a6b99dbb837945ad029d4f1abcf876a4e"
SUMMARY = Path("runs") / BACKBONE / VARIANT / "seed_42/summary.json"
DATASETS = ("yellow_trip_hourly", "raf_spare_parts")


def reference_route_parity(row, project):
    """Explicit diagnostic-only weight transfer, never initialization for training."""
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    checkpoint_path = (project / row["artifact_dir"] / "runs/titantpp" / VARIANT /
                       "seed_42/best_val_joint_objective_model.pt")
    require(digest(checkpoint_path) == row["checkpoint_file_sha256"], "Reference checkpoint file changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint["backbone"] == "titantpp" and checkpoint["variant"] == VARIANT, "Wrong reference checkpoint")
    require(canonical_state_dict_sha256(checkpoint["model_state_dict"]) == row["checkpoint_state_sha256"],
            "Reference checkpoint state changed")
    models = [build_count_aware_model(name, hidden_dim=64, train_log_mean=1.5,
              max_seq_len=row["max_seq_len"], quantity_variant=VARIANT, lambda_tail=0.,
              time_head_mode="legacy_clamped_rmtpp")[0] for name in ("titantpp", BACKBONE)]
    for model in models:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
    dts = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 0.]])
    qty = torch.tensor([[1., 5., 9., 20.], [2., 6., 12., 0.]])
    mask = dts > 0
    with torch.no_grad():
        original = models[0].encode_task_states(dts, qty, mask)[1]
        time, quantity = models[1].encode_task_states(dts, qty, mask)
        torch.testing.assert_close(original, quantity, rtol=0, atol=0)
        torch.testing.assert_close(models[0].predict_quantity(original)[1],
                                   models[1].predict_quantity(quantity)[1], rtol=0, atol=0)
        torch.testing.assert_close(time, models[0]._encode_base(dts, qty, mask), rtol=0, atol=0)
    return {"dataset": row["dataset"], "status": "passed", "device": "cpu",
            "input": "synthetic_history_no_dataset_rows", "checkpoint_file_sha256": digest(checkpoint_path),
            "checkpoint_state_sha256": row["checkpoint_state_sha256"],
            "quantity_state_max_absolute_difference": float((original - quantity).abs().max()),
            "weights_used_for_training": False}


def command(row, project, output, revision):
    from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS
    require(row["dataset"] in DATASETS, "Only Taxi/RAF smoke is authorized")
    tail = DATASET_CONTRACTS[row["contract_dataset"]]["tail_contract"]
    args = {"data": project / row["data_path"], "split-manifest": project / row["split_manifest_path"],
            "output-dir": output, "source-revision": revision, "execution-role": "hard_local_time_smoke_5080",
            "dataset-contract": row["contract_dataset"], "model-role": ROLE,
            "device": "cuda", "epochs": 1, "min-epochs": 1, "early-stopping-patience": 1,
            "batch-size": 128, "lr": .001, "grad-clip": 1., "hidden-dim": 64,
            "lookback-weeks": row["lookback"], "max-seq-len": row["max_seq_len"],
            "quantity-variants": "log_mse", "backbones": BACKBONE, "seeds": 42,
            "lambda-log-qty": 1., "lambda-tail": 0., "time-head-mode": "legacy_clamped_rmtpp",
            "time-scale": 3., "time-w-max": 10. / 3., "time-intercept-limit": 30.,
            "time-wd-safety-limit": 40., "time-head-lr-multiplier": 1.,
            "tail-threshold": tail["threshold"], "tail-normalization-scale": tail["normalization_scale"],
            "tail-clip-cap": tail["clip_cap"], "tail-huber-delta": tail["huber_delta"]}
    result = [sys.executable, "-s", "-u", str(ROOT / "paper/scripts/run_count_aware_tpp_backbone_control.py")]
    for key, value in args.items():
        result.extend([f"--{key}", str(value)])
    return result + ["--allow-partial-contract"]


def audit_run(output, row, reference, revision, spec, *, screening=False, write_audit=True):
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    c, s = read(output / "launch_contract.json"), read(output / SUMMARY)
    history_payload = read((output / SUMMARY).parent / "history.json")
    require(isinstance(history_payload, dict) and isinstance(history_payload.get("history"), list),
            "Expected the runner's history envelope")
    history = history_payload["history"]
    for payload in (c, s, history):
        finite(payload)
    require(c["status"] == "complete" and c["completed_run_count"] == 1, "Run incomplete")
    require(c["backbones"] == [BACKBONE] and c["seeds"] == [42] and c["model_role"] == ROLE, "Wrong candidate")
    budget, minimum = (300, 40) if screening else (1, 1)
    for key, expected in {"epochs": budget, "batch_size": 128, "lr": .001, "hidden_dim": 64,
                          "grad_clip": 1., "lambda_log_qty": 1., "lambda_tail": 0.,
                          "data_sha256": row["data_sha256"],
                          "split_manifest_sha256": row["split_manifest_sha256"]}.items():
        require(c[key] == expected, f"Launch contract mismatch: {key}")
    require(s["status"] == "success" and s["epochs"] == budget, "Wrong budget/status")
    require(minimum <= len(history) <= budget and
            [r["epoch"] for r in history] == list(range(1, len(history) + 1)), "Unexpected history/resume")
    best = min(history, key=lambda r: r["val_joint_objective"]) if screening else history[0]
    require(s["best_epoch"] == best["epoch"], "Wrong checkpoint selection")
    if screening:
        require(s["completed_epochs"] == len(history), "Completed epoch mismatch")
        require(len(history) == budget or len(history) - best["epoch"] >= 40, "Premature completion")
        for length in range(minimum, len(history)):
            prefix_best = min(history[:length], key=lambda r: r["val_joint_objective"])["epoch"]
            require(length - prefix_best < 40, "Training continued after patience exhausted")
        require(all(r["train_event_count"] == spec["train_targets"] for r in history), "Partial train epoch")
    require(c["source_revision"] == s["source_revision"] == revision and
            s["source_revision_history"] == [revision], "Revision or resume mismatch")
    require(not c["partial_smoke"] and set(c["split_rows"]) == {"train", "validation"}, "Subsampling/test materialization")
    for payload in (c, s):
        require(payload["evaluation_scope"] == "validation_only" and
                not payload["held_out_test_evaluated"], "Held-out evaluated")
    require(not list(output.rglob("*test_summary*")) and not list(output.rglob("*test_metrics*")), "Held-out artifact found")
    for key, expected in {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10. / 3.,
                          "time_intercept_limit": 30., "time_head_lr_multiplier": 1.}.items():
        require(c["time_head"][key] == expected, f"Time launch contract mismatch: {key}")
    for key, expected in {"min_epochs": minimum, "patience": minimum, "monitor": "validation_joint_objective"}.items():
        require(c["early_stopping"][key] == expected, f"Selection mismatch: {key}")
    require(c["time_head"]["train_time_statistics"]["target_count"] == spec["train_targets"], "Train count changed")
    for rows in (s["quantity_rows"], s["history_rows"]):
        require(sum(r["count"] for r in rows) == spec["validation_targets"], "Validation count changed")
    for key in ("quantity_rows", "history_rows"):
        require([(r["stratum"], r["count"]) for r in s[key]] ==
                [(r["stratum"], r["count"]) for r in reference[key]], f"Strata changed: {key}")
    require(s["parameter_count"] == reference["parameter_count"] == spec["parameter_count"], "Parameter count changed")
    validate_quantity_initialization(s, reference, row)
    require(c["lookback_weeks"] == row["lookback"] and c["max_seq_len"] == row["max_seq_len"], "Context changed")
    validate_checkpoint_route(s, BACKBONE)
    checkpoint_path = (output / SUMMARY).parent / "best_val_joint_objective_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    last = torch.load(checkpoint_path.with_name("last_epoch_state.pt"), map_location="cpu", weights_only=False)
    for payload in (checkpoint, last):
        validate_checkpoint_route(payload, BACKBONE)
        require(payload["source_revision"] == revision, "Checkpoint source mismatch")
        require(all(torch.isfinite(t).all() for t in payload["model_state_dict"].values()), "Nonfinite checkpoint")
    state_digest = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(state_digest == s["checkpoint_state_sha256"], "Checkpoint digest mismatch")
    saved_best = last["best_state_dict"] if screening else last["model_state_dict"]
    require(state_digest == canonical_state_dict_sha256(saved_best), "Saved best checkpoint mismatch")
    groups = last["optimizer_state_dict"]["param_groups"]
    require(len(groups) == 1, "Unexpected optimizer groups")
    for key, expected in {"lr": .001, "weight_decay": .01, "eps": 1e-8,
                          "betas": (.9, .999), "amsgrad": False}.items():
        require(groups[0][key] == expected, f"Optimizer mismatch: {key}")
    result = {"status": "passed", "dataset": row["dataset"], "phase": "e300_screening" if screening else "full_e1_smoke",
              "summary_sha256": digest(output / SUMMARY), "history_sha256": digest((output / SUMMARY).parent / "history.json"),
              "checkpoint_file_sha256": digest(checkpoint_path), "checkpoint_state_sha256": state_digest,
              "finite": True, "held_out_test_evaluated": False, "train_validation_counts_match": True,
              "route_and_optimizer_verified": True,
              "performance_acceptance": "pending_comparison" if screening else "not_evaluated_e1_only"}
    if write_audit:
        save(output / "audit.json", result)
    return result


def execute(args):
    # Existing directories are rejected before any write or subprocess call.
    args.output_root.mkdir(parents=True, exist_ok=False)
    status_path = args.output_root / "status.json"
    status = {"status": "running", "phase": "cuda_and_full_e1_only", "source_revision": args.source_revision,
              "started_at": datetime.now(timezone.utc).isoformat(), "completed": [], "current_dataset": None,
              "held_out_test_evaluated": False, "performance_training_authorized": False}
    save(status_path, status)
    try:
        require(digest(CONTRACT) == CONTRACT_SHA256, "Frozen contract changed")
        contract = read(CONTRACT)
        status["source_manifest_sha256"] = verify_source(args.source_revision)
        registry = ROOT / contract["baseline"]["registry"]
        require(digest(registry) == contract["baseline"]["registry_sha256"], "Registry changed")
        rows = {r["dataset"]: r for r in read(registry)["datasets"]}
        specs = {r["dataset"]: r for r in contract["datasets"]}
        references = {name: baseline(rows[name], args.project_root)[1] for name in DATASETS}
        status["reference_route_parity"] = [reference_route_parity(rows[name], args.project_root) for name in DATASETS]
        save(args.output_root / "frozen_contract.json", contract)
        save(args.output_root / "source_manifest.json", read(ROOT / "source_manifest.json"))
        save(args.output_root / "baseline_registry.json", [rows[name] for name in DATASETS])
        status["preflight"] = preflight(12000)
        save(status_path, status)
        env = dict(os.environ, HARD_LOCAL_TIME_TEST_DEVICE="cuda")
        run_logged([sys.executable, "-s", "-m", "pytest", "-q",
                    "simple_lab_test/search/tests/test_hard_lmm_local_time_memory.py",
                    f"--junitxml={args.output_root / 'cuda_contract_tests.xml'}"],
                   args.output_root / "cuda_contract_tests.log", env=env)
        status["cuda_contract_tests"] = "passed"
        for name in DATASETS:
            status["current_dataset"] = name
            status["preflight"] = preflight(12000)
            verify_source(args.source_revision)
            cmd = command(rows[name], args.project_root, args.output_root / name, args.source_revision)
            status["command"] = cmd
            save(status_path, status)
            run_logged(cmd, args.output_root / f"{name}.log")
            result = audit_run(args.output_root / name, rows[name], references[name], args.source_revision, specs[name])
            status["completed"].append(result)
            save(status_path, status)
        status["postflight"] = preflight(12000)
        status.update(status="complete", current_dataset=None, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=datetime.now(timezone.utc).isoformat())
        save(status_path, status)
        raise
    save(status_path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
