#!/usr/bin/env python3
"""Isolated 5090 CUDA contracts and full Taxi/Instacart e1, never screening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE as BACKBONE, KEY_VALUE_ROLE as ROLE
from paper.scripts.run_hard_lmm_weighted_static import (
    command as weighted_command, digest, finite, preflight, read, require,
    run_logged, save, validate_quantity_initialization, verify_source,
)

CONTRACT = "paper/contracts/hard_lmm_key_value_cuda_e1_5090_v1.json"
REGISTRY = "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json"
DATASETS = ("yellow_trip_hourly", "insta_market_basket")
VARIANT = "count_only_log_regression"
SUMMARY = Path("runs") / BACKBONE / VARIANT / "seed_42/summary.json"
EXECUTION_ROLE = "key_value_static_e1_5090"


def frozen_documents():
    spec = read(ROOT / CONTRACT)
    for name, expected in spec["frozen_files"].items():
        require(digest(ROOT / name) == expected, f"Frozen model/contract changed: {name}")
    require(tuple(r["dataset"] for r in spec["datasets"]) == DATASETS, "Dataset scope changed")
    registry = {r["dataset"]: r for r in read(ROOT / REGISTRY)["datasets"]}
    return spec, [{**registry[r["dataset"]], **r} for r in spec["datasets"]]


def reference_files(row):
    artifact = Path(row["artifact_dir"])
    run = artifact / "runs/titantpp" / VARIANT / "seed_42"
    return {
        "launch_contract.json": (artifact / "launch_contract.json", row["contract_sha256"]),
        "summary.json": (run / "summary.json", row["summary_sha256"]),
        "best_val_joint_objective_model.pt": (run / "best_val_joint_objective_model.pt", row["checkpoint_file_sha256"]),
    }


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
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    for name in ("data", "split_manifest"):
        require(digest(project / row[name + "_path"]) == row[name + "_sha256"], f"Input changed: {name}")
    root = ROOT / "references" / row["dataset"]
    for name, (_, expected) in reference_files(row).items():
        require(digest(root / name) == expected, f"Reference changed: {name}")
    c, s = read(root / "launch_contract.json"), read(root / "summary.json")
    finite(s)
    require(s["status"] == "success" and s["seed"] == 42 and s["backbone"] == "titantpp", "Wrong reference")
    require(s["evaluation_scope"] == "validation_only" and s["held_out_test_evaluated"] is False, "Reference scope")
    checkpoint = torch.load(root / "best_val_joint_objective_model.pt", map_location="cpu", weights_only=False)
    finite_tensors(checkpoint)
    require(canonical_state_dict_sha256(checkpoint["model_state_dict"]) ==
            s["checkpoint_state_sha256"] == row["checkpoint_state_sha256"], "Reference state mismatch")
    return c, s


def command(row, project, output, revision):
    require(row["dataset"] in DATASETS, "Only Taxi/Instacart e1 is authorized")
    cmd = weighted_command(row, project, output, revision, "smoke")
    for flag, value in (("--backbones", BACKBONE), ("--model-role", ROLE), ("--execution-role", EXECUTION_ROLE)):
        cmd[cmd.index(flag) + 1] = value
    return cmd


def validate_history(history, summary, row, *, screening=False):
    budget, minimum, patience = (300, 40, 40) if screening else (1, 1, 1)
    require(minimum <= len(history) <= budget, "Unexpected history length")
    require([h["epoch"] for h in history] == list(range(1, len(history) + 1)), "Nonconsecutive fresh history")
    require(all(h["train_all_finite"] is True and h["train_event_count"] == row["train_targets"]
                for h in history), "Partial or nonfinite train epoch")
    best = min(history, key=lambda h: h["val_joint_objective"])
    require(summary["epochs"] == budget and summary["completed_epochs"] == len(history) and
            summary["best_epoch"] == best["epoch"], "History/selection mismatch")
    if screening:
        # Verify the first legal stop, not just the final patience distance.
        from paper.scripts.count_aware_tpp_backbone.training import early_stopping_exhausted
        stops = [i for i in range(1, len(history) + 1) if early_stopping_exhausted(
            history[:i], min_epochs=minimum, patience=patience)]
        require(not stops or stops[0] == len(history), "Training continued past early stopping")
        require(len(history) == budget or bool(stops), "Premature training termination")
        require(summary["stopped_early"] == (len(history) < budget), "Wrong stop metadata")
    return best


def audit_run(output, row, reference, revision, *, screening=False, write_audit=True):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    ref_contract, ref_summary = reference
    budget, minimum = (300, 40) if screening else (1, 1)
    c, s = read(output / "launch_contract.json"), read(output / SUMMARY)
    run = (output / SUMMARY).parent
    history = read(run / "history.json")["history"]
    for payload in (c, s, history):
        finite(payload)
    require(c["status"] == "complete" and c["completed_run_count"] == 1, "Run incomplete")
    for key, expected in {
        "backbones": [BACKBONE], "seeds": [42], "model_role": ROLE, "epochs": budget,
        "batch_size": 128, "lr": .001, "hidden_dim": 64, "grad_clip": 1.,
        "lambda_log_qty": 1., "lambda_tail": 0., "data_sha256": row["data_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "lookback_weeks": row["lookback"], "max_seq_len": row["max_seq_len"],
        "quantity_contract": ref_contract["quantity_contract"],
        "history_length_contract": ref_contract["history_length_contract"],
    }.items():
        require(c[key] == expected, f"Launch contract mismatch: {key}")
    require(s["status"] == "success" and s["backbone"] == BACKBONE and s["seed"] == 42, "Not a successful run")
    require(c["source_revision"] == s["source_revision"] == revision and
            s["source_revision_history"] == [revision], "Revision/resume mismatch")
    require(not c["partial_smoke"] and set(c["split_rows"]) == {"train", "validation"}, "Partial/test materialization")
    for payload in (c, s):
        require(payload["evaluation_scope"] == "validation_only" and payload["held_out_test_evaluated"] is False,
                "Held-out evaluated")
    require(not [p for p in output.rglob("*") if p.name.startswith("test_") or
                 "test_summary" in p.name or "test_metrics" in p.name], "Held-out artifact found")
    for key, expected in {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10. / 3.,
                          "time_intercept_limit": 30., "time_wd_safety_limit": 40., "time_head_lr_multiplier": 1.}.items():
        require(c["time_head"][key] == expected, f"Time launch contract mismatch: {key}")
    for key, expected in {"min_epochs": minimum, "patience": minimum, "monitor": "validation_joint_objective"}.items():
        require(c["early_stopping"][key] == expected, f"Selection mismatch: {key}")
    best = validate_history(history, s, row, screening=screening)
    require(c["time_head"]["train_time_statistics"]["target_count"] == row["train_targets"], "Partial train epoch")
    for key in ("quantity_rows", "history_rows"):
        rows = s[key]
        require(sum(r["count"] for r in rows) == row["validation_targets"], "Partial validation epoch")
        require([(r["stratum"], r["count"]) for r in rows] ==
                [(r["stratum"], r["count"]) for r in ref_summary[key]], f"Strata changed: {key}")
        values = {"qty_mae": sum(r["count"] * r["qty_mae"] for r in rows) / row["validation_targets"],
                  "qty_rmse": math.sqrt(sum(r["count"] * r["qty_rmse"]**2 for r in rows) / row["validation_targets"])}
        for metric, value in values.items():
            require(math.isclose(value, s["best_val_" + metric], rel_tol=1e-10, abs_tol=1e-8), "Strata aggregate mismatch")
    require(s["parameter_count"] == row["candidate_parameters"] and
            ref_summary["parameter_count"] == row["baseline_parameters"], "Parameter count mismatch")
    validate_quantity_initialization(s, ref_summary, row)
    validate_checkpoint_route(s, BACKBONE)
    for key in ("qty_mae", "qty_rmse", "time_nll", "log_qty_mse", "joint_objective"):
        require(math.isclose(s["best_val_" + key], best["val_" + key], rel_tol=1e-10, abs_tol=1e-8),
                "Summary/history mismatch")
    checkpoint_path = run / "best_val_joint_objective_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    last = torch.load(run / "last_epoch_state.pt", map_location="cpu", weights_only=False)
    for payload in (checkpoint, last):
        validate_checkpoint_route(payload, BACKBONE)
        require(payload["source_revision"] == revision, "Checkpoint source mismatch")
        finite_tensors(payload)
    state_hash = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(state_hash == s["checkpoint_state_sha256"] == canonical_state_dict_sha256(last["best_state_dict"]),
            "Best checkpoint digest mismatch")
    require(last["epoch"] == len(history) and last["history"] == history, "Last checkpoint history mismatch")
    if not screening:
        require(state_hash == canonical_state_dict_sha256(last["model_state_dict"]), "e1 checkpoint digest mismatch")
    groups = last["optimizer_state_dict"]["param_groups"]
    require(len(groups) == 1, "Unexpected optimizer groups")
    for key, expected in {"lr": .001, "weight_decay": .01, "eps": 1e-8, "betas": (.9, .999), "amsgrad": False}.items():
        require(groups[0][key] == expected, f"Optimizer mismatch: {key}")
    predictions = []
    replay_states = (checkpoint["model_state_dict"], last["best_state_dict"])
    for state in replay_states:
        model, _ = build_count_aware_model(BACKBONE, hidden_dim=64, train_log_mean=1.5,
                                          max_seq_len=row["max_seq_len"], quantity_variant=VARIANT)
        model.load_state_dict(state, strict=True)
        model.eval()
        dt, qty = torch.tensor([[1., 2., 3., 0.]]), torch.tensor([[2., 5., 8., 0.]])
        with torch.no_grad():
            predictions.append(target_outputs(model, dt, dt > 0, qty, lambda_log_qty=1.))
    for key in predictions[0]:
        torch.testing.assert_close(predictions[0][key], predictions[1][key], rtol=0, atol=0)
        finite_tensors(predictions[0][key])
    key_value_distance = (model.lmm.memory_keys - model.lmm.mem).detach().norm().item()
    require(key_value_distance > 0, "Keys and values unexpectedly remained identical")
    result = {"status": "passed", "dataset": row["dataset"], "phase": "seed42_e300" if screening else "full_e1_only",
              "summary_sha256": digest(output / SUMMARY), "history_sha256": digest(run / "history.json"),
              "checkpoint_file_sha256": digest(checkpoint_path), "checkpoint_state_sha256": state_hash,
              "train_targets": row["train_targets"], "validation_targets": row["validation_targets"],
              "finite": True, "held_out_test_evaluated": False, "checkpoint_replay": "exact_best_vs_last_best_state_cpu",
              "key_value_distance": key_value_distance,
              "performance_acceptance": "separate_comparison_required" if screening else "not_evaluated_e1_only"}
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
        spec, rows = frozen_documents()
        status["source_manifest_sha256"] = verify_source(args.source_revision)
        references = {r["dataset"]: baseline(r, args.project_root) for r in rows}
        save(args.output_root / "execution_contract.json", spec)
        save(args.output_root / "source_manifest.json", read(ROOT / "source_manifest.json"))
        status["preflight"] = preflight(spec["gpu_preflight"]["minimum_free_mib"])
        save(path, status)
        run_logged([sys.executable, "-s", "-m", "pytest", "-q",
                    "simple_lab_test/search/tests/test_hard_lmm_key_value_memory.py",
                    f"--junitxml={args.output_root / 'cuda_contract_tests.xml'}"],
                   args.output_root / "cuda_contract_tests.log", env=dict(os.environ, HARD_KEY_VALUE_TEST_DEVICE="cuda"))
        status["cuda_contract_tests"] = "passed"
        for row in rows:
            name = row["dataset"]
            status["current_dataset"] = name
            status["preflight"] = preflight(spec["gpu_preflight"]["minimum_free_mib"])
            verify_source(args.source_revision)
            baseline(row, args.project_root)
            cmd = command(row, args.project_root, args.output_root / name, args.source_revision)
            status["command"] = cmd
            save(path, status)
            start = time.monotonic()
            run_logged(cmd, args.output_root / f"{name}.log")
            audit = audit_run(args.output_root / name, row, references[name], args.source_revision)
            status["completed"].append({**audit, "elapsed_seconds": time.monotonic() - start})
            save(path, status)
        status["postflight"] = preflight(spec["gpu_preflight"]["minimum_free_mib"])
        status.update(status="complete", current_dataset=None, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=datetime.now(timezone.utc).isoformat())
        save(path, status)
        raise
    save(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    execute(parser.parse_args())
