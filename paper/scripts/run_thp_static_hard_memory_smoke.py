#!/usr/bin/env python3
"""5080 CUDA contracts and full Taxi/RAF e1 only, with no screening mode."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareTHPStaticMemory import (
    THP_STATIC_MEMORY_BACKBONE as BACKBONE,
    THP_STATIC_MEMORY_ROLE as ROLE,
)
from paper.scripts.count_aware_tpp_backbone.thp_static_contract import frozen_documents
from paper.scripts.run_hard_lmm_weighted_static import (
    digest, finite, preflight, read, require, run_logged, save,
    validate_quantity_initialization, verify_source,
)

VARIANT = "count_only_log_regression"
SUMMARY = Path("runs") / BACKBONE / VARIANT / "seed_42/summary.json"
DATASETS = ("yellow_trip_hourly", "raf_spare_parts")


def finite_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), "Nonfinite checkpoint/optimizer tensor")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite_tensors(item)
    else:
        finite(value)


def baseline(row, project):
    """Recheck pinned inputs; do not create a model or reuse baseline weights."""
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    for path, expected in ((row["data_path"], row["data_sha256"]),
                           (row["split_manifest_path"], row["split_manifest_sha256"]),
                           (row["artifact_dir"] + "/launch_contract.json", row["launch_contract_sha256"])):
        require(digest(project / path) == expected, f"Baseline/input changed: {path}")
    references = {}
    for name, spec in row["baselines"].items():
        for relative, expected in spec["files_sha256"].items():
            require(digest(project / relative) == expected, f"Baseline file changed: {relative}")
        run = project / row["artifact_dir"] / "runs" / name / VARIANT / "seed_42"
        summary = read(run / "summary.json")
        finite(summary)
        require(summary["status"] == "success" and summary["backbone"] == name and summary["seed"] == 42,
                "Baseline identity/status mismatch")
        require(summary["evaluation_scope"] == "validation_only" and summary["held_out_test_evaluated"] is False,
                "Baseline scope mismatch")
        checkpoint = torch.load(run / "best_val_joint_objective_model.pt", map_location="cpu", weights_only=False)
        finite_tensors(checkpoint)
        require(canonical_state_dict_sha256(checkpoint["model_state_dict"]) == spec["checkpoint_state_sha256"],
                "Baseline checkpoint digest mismatch")
        references[name] = summary
    return references["thp"]


def command(row, project, output, revision):
    from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS
    require(row["dataset"] in DATASETS, "Only Taxi/RAF e1 is authorized")
    tail = DATASET_CONTRACTS[row["contract_dataset"]]["tail_contract"]
    args = {"data": project / row["data_path"], "split-manifest": project / row["split_manifest_path"],
            "output-dir": output, "source-revision": revision, "execution-role": "thp_static_memory_e1_5080",
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


def audit_run(output, row, reference, revision, *, write_audit=True):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    c, s = read(output / "launch_contract.json"), read(output / SUMMARY)
    history = read((output / SUMMARY).parent / "history.json")["history"]
    for payload in (c, s, history):
        finite(payload)
    require(c["status"] == "complete" and c["completed_run_count"] == 1, "Run incomplete")
    require(c["backbones"] == [BACKBONE] and c["seeds"] == [42] and c["model_role"] == ROLE, "Wrong candidate")
    for key, expected in {"epochs": 1, "batch_size": 128, "lr": .001, "hidden_dim": 64,
                          "grad_clip": 1., "lambda_log_qty": 1., "lambda_tail": 0.,
                          "data_sha256": row["data_sha256"], "split_manifest_sha256": row["split_manifest_sha256"],
                          "quantity_contract": row["quantity_contract"],
                          "history_length_contract": row["history_length_contract"]}.items():
        require(c[key] == expected, f"Launch contract mismatch: {key}")
    require(s["status"] == "success" and s["epochs"] == s["completed_epochs"] == s["best_epoch"] == 1,
            "Not a completed e1 run")
    require(len(history) == 1 and history[0]["epoch"] == 1 and history[0]["train_all_finite"] is True,
            "Unexpected history/resume or nonfinite training")
    require(history[0]["train_event_count"] == row["train_targets"], "Partial train epoch")
    require(c["source_revision"] == s["source_revision"] == revision and s["source_revision_history"] == [revision],
            "Revision/resume mismatch")
    require(not c["partial_smoke"] and set(c["split_rows"]) == {"train", "validation"}, "Partial/test materialization")
    for payload in (c, s):
        require(payload["evaluation_scope"] == "validation_only" and payload["held_out_test_evaluated"] is False,
                "Held-out evaluated")
    require(not [p for p in output.rglob("*") if p.name.startswith("test_") or
                 "test_summary" in p.name or "test_metrics" in p.name], "Held-out artifact found")
    for key, expected in {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10. / 3.,
                          "time_intercept_limit": 30., "time_wd_safety_limit": 40., "time_head_lr_multiplier": 1.}.items():
        require(c["time_head"][key] == expected, f"Time launch contract mismatch: {key}")
    for key, expected in {"min_epochs": 1, "patience": 1, "monitor": "validation_joint_objective"}.items():
        require(c["early_stopping"][key] == expected, f"Selection mismatch: {key}")
    require(c["time_head"]["train_time_statistics"]["target_count"] == row["train_targets"], "Train count changed")
    for key in ("quantity_rows", "history_rows"):
        rows = s[key]
        require(sum(r["count"] for r in rows) == row["validation_targets"], "Validation count changed")
        require([(r["stratum"], r["count"]) for r in rows] ==
                [(r["stratum"], r["count"]) for r in reference[key]], f"Strata changed: {key}")
        for metric, value in (("qty_mae", sum(r["count"] * r["qty_mae"] for r in rows) / row["validation_targets"]),
                              ("qty_rmse", math.sqrt(sum(r["count"] * r["qty_rmse"]**2 for r in rows) / row["validation_targets"]))):
            require(math.isclose(value, s["best_val_" + metric], rel_tol=1e-10, abs_tol=1e-8), "Stratum aggregate mismatch")
    require(s["parameter_count"] == 104387 and reference["parameter_count"] == 100291, "Wrong parameter count")
    validate_quantity_initialization(s, reference, row)
    require(c["lookback_weeks"] == row["lookback"] and c["max_seq_len"] == row["max_seq_len"], "Context changed")
    validate_checkpoint_route(s, BACKBONE)
    for key in ("qty_mae", "qty_rmse", "time_nll", "log_qty_mse", "joint_objective"):
        require(math.isclose(s["best_val_" + key], history[0]["val_" + key], rel_tol=1e-10, abs_tol=1e-8),
                "Summary/history mismatch")
    checkpoint_path = (output / SUMMARY).parent / "best_val_joint_objective_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    last = torch.load(checkpoint_path.with_name("last_epoch_state.pt"), map_location="cpu", weights_only=False)
    for payload in (checkpoint, last):
        validate_checkpoint_route(payload, BACKBONE)
        require(payload["source_revision"] == revision, "Checkpoint source mismatch")
        finite_tensors(payload)
    state_hash = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(state_hash == s["checkpoint_state_sha256"] == canonical_state_dict_sha256(last["model_state_dict"]) ==
            canonical_state_dict_sha256(last["best_state_dict"]), "e1 checkpoint digest mismatch")
    groups = last["optimizer_state_dict"]["param_groups"]
    require(len(groups) == 1, "Unexpected optimizer groups")
    for key, expected in {"lr": .001, "weight_decay": .01, "eps": 1e-8, "betas": (.9, .999), "amsgrad": False}.items():
        require(groups[0][key] == expected, f"Optimizer mismatch: {key}")
    # Strict-load both saved files and replay without another optimizer step or
    # data read. This parent process never initializes a CUDA context.
    predictions = []
    for payload in (checkpoint, last):
        model, _ = build_count_aware_model(BACKBONE, hidden_dim=64, train_log_mean=1.5,
                                          max_seq_len=row["max_seq_len"], quantity_variant=VARIANT)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        dt, qty = torch.tensor([[1., 2., 3., 0.]]), torch.tensor([[2., 5., 8., 0.]])
        with torch.no_grad():
            predictions.append(target_outputs(model, dt, dt > 0, qty, lambda_log_qty=1.))
    for key in predictions[0]:
        torch.testing.assert_close(predictions[0][key], predictions[1][key], rtol=0, atol=0)
        finite_tensors(predictions[0][key])
    result = {"status": "passed", "dataset": row["dataset"], "phase": "full_e1_only",
              "summary_sha256": digest(output / SUMMARY), "history_sha256": digest(checkpoint_path.with_name("history.json")),
              "checkpoint_file_sha256": digest(checkpoint_path), "checkpoint_state_sha256": state_hash,
              "train_targets": row["train_targets"], "validation_targets": row["validation_targets"],
              "finite": True, "held_out_test_evaluated": False, "checkpoint_replay": "exact_best_last_cpu",
              "performance_acceptance": "not_evaluated_e1_only"}
    if write_audit:
        save(output / "audit.json", result)
    return result


def execute(args):
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "status.json"
    status = {"status": "running", "phase": "cuda_and_full_e1_only", "source_revision": args.source_revision,
              "started_at": datetime.now(timezone.utc).isoformat(), "completed": [], "current_dataset": None,
              "held_out_test_evaluated": False, "performance_training_authorized": False}
    save(path, status)
    try:
        contract, registry = frozen_documents()
        status["source_manifest_sha256"] = verify_source(args.source_revision)
        rows = {r["dataset"]: r for r in registry["datasets"]}
        references = {name: baseline(rows[name], args.project_root) for name in DATASETS}
        save(args.output_root / "frozen_contract.json", contract)
        save(args.output_root / "baseline_registry.json", registry)
        save(args.output_root / "source_manifest.json", read(ROOT / "source_manifest.json"))
        status["preflight"] = preflight(12000)
        save(path, status)
        env = dict(os.environ, THP_STATIC_MEMORY_TEST_DEVICE="cuda")
        run_logged([sys.executable, "-s", "-m", "pytest", "-q",
                    "simple_lab_test/search/tests/test_thp_static_hard_memory_model.py",
                    f"--junitxml={args.output_root / 'cuda_contract_tests.xml'}"],
                   args.output_root / "cuda_contract_tests.log", env=env)
        status["cuda_contract_tests"] = "passed"
        for name in DATASETS:
            status["current_dataset"] = name
            status["preflight"] = preflight(12000)
            verify_source(args.source_revision)
            cmd = command(rows[name], args.project_root, args.output_root / name, args.source_revision)
            status["command"] = cmd
            save(path, status)
            run_logged(cmd, args.output_root / f"{name}.log")
            status["completed"].append(audit_run(args.output_root / name, rows[name], references[name], args.source_revision))
            save(path, status)
        status["postflight"] = preflight(12000)
        status.update(status="complete", current_dataset=None, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=datetime.now(timezone.utc).isoformat())
        save(path, status)
        raise
    save(path, status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
