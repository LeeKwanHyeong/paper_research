#!/usr/bin/env python3
"""Isolated elapsed-age CUDA contracts and complete Taxi/Instacart e1 only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import platform
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.Titan.common.elapsed_age import (
    ELAPSED_AGE_BACKBONE as BACKBONE, ELAPSED_AGE_ROLE as ROLE,
    ELAPSED_AGE_CONTRACT as MODEL_CONTRACT,
)
from paper.scripts.run_hard_lmm_key_value_smoke import finite_tensors, reference_files, validate_history
from paper.scripts.run_hard_lmm_weighted_static import (
    command as weighted_command, digest, finite, preflight, read, require,
    run_logged, save, validate_quantity_initialization,
)

CONTRACT = "paper/contracts/hard_lmm_elapsed_age_cuda_e1_5090_v1.json"
REGISTRY = "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json"
DATASETS = ("yellow_trip_hourly", "insta_market_basket")
VARIANT = "count_only_log_regression"
SUMMARY = Path("runs") / BACKBONE / VARIANT / "seed_42/summary.json"
EXECUTION_ROLE = "elapsed_age_static_e1_5090"
BETA_KEY = "encoder.elapsed_age_beta"
CUDA_TESTS = (
    "simple_lab_test/search/tests/test_hard_lmm_elapsed_age_memory.py",
    "simple_lab_test/search/tests/test_hard_lmm_elapsed_age_contract.py",
)


def frozen_documents():
    spec = read(ROOT / CONTRACT)
    for name, expected in spec["frozen_files"].items():
        require(digest(ROOT / name) == expected, f"Frozen reference/contract changed: {name}")
    require(tuple(row["dataset"] for row in spec["datasets"]) == DATASETS, "Dataset scope changed")
    require(spec["run"] == {"seed": 42, "epochs": 1, "min_epochs": 1, "patience": 1,
            "batch_size": 128, "lr": .001, "full_train_validation": True}, "Only fixed full e1 is authorized")
    require(spec["backbone"] == BACKBONE and spec["model_role"] == ROLE, "Wrong candidate identity")
    require(spec["held_out_test"] is False and spec["performance_training_authorized"] is False,
            "Execution scope must remain CUDA/full e1 only")
    registry = {row["dataset"]: row for row in read(ROOT / REGISTRY)["datasets"]}
    return spec, [{**registry[row["dataset"]], **row} for row in spec["datasets"]]


def reference_path(row, name):
    packaged = ROOT / "references" / row["dataset"] / name
    # Local audit-only uses the original pinned artifacts, not downloaded paths.
    return packaged if packaged.is_file() else ROOT / reference_files(row)[name][0]


def verify_source(revision, manifest_path=None):
    path = manifest_path or ROOT / "source_manifest.json"
    manifest = read(path)
    require(manifest["source_revision"] == revision and len(revision) == 40, "Source revision mismatch")
    require(manifest["performance_training_authorized"] is False, "Source package is not e1-only")
    _, rows = frozen_documents()
    references = {str(Path("references") / row["dataset"] / name): reference_path(row, name)
                  for row in rows for name in reference_files(row)}
    for name, expected in manifest["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source manifest path")
        actual = references.get(name, ROOT / name)
        require(digest(actual) == expected, f"Source/reference mismatch: {name}")
    return digest(path)


def baseline(row, project):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    for name in ("data", "split_manifest"):
        require(digest(project / row[name + "_path"]) == row[name + "_sha256"], f"Input changed: {name}")
    for name, (_, expected) in reference_files(row).items():
        require(digest(reference_path(row, name)) == expected, f"Reference changed: {name}")
    contract, summary = (read(reference_path(row, name)) for name in ("launch_contract.json", "summary.json"))
    finite(summary)
    require(summary["status"] == "success" and summary["seed"] == 42 and summary["backbone"] == "titantpp",
            "Wrong original baseline reference")
    require(summary["evaluation_scope"] == "validation_only" and summary["held_out_test_evaluated"] is False,
            "Reference scope mismatch")
    checkpoint = torch.load(reference_path(row, "best_val_joint_objective_model.pt"), map_location="cpu", weights_only=False)
    finite_tensors(checkpoint)
    require(canonical_state_dict_sha256(checkpoint["model_state_dict"]) ==
            summary["checkpoint_state_sha256"] == row["checkpoint_state_sha256"], "Reference state mismatch")
    return contract, summary


def command(row, project, output, revision):
    require(row["dataset"] in DATASETS, "Only Taxi/Instacart e1 is authorized")
    cmd = weighted_command(row, project, output, revision, "smoke")
    for flag, value in (("--backbones", BACKBONE), ("--model-role", ROLE), ("--execution-role", EXECUTION_ROLE)):
        cmd[cmd.index(flag) + 1] = value
    return cmd


def audit_cuda_tests(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    require(bool(cases), "CUDA contract suite contains no executed tests")
    require(not list(root.iter("failure")) and not list(root.iter("error")) and not list(root.iter("skipped")),
            "CUDA contracts contain a failure, error or skipped test")
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    for suite in suites:
        require(all(int(suite.get(key, 0)) == 0 for key in ("failures", "errors", "skipped")),
                "CUDA test totals report failure/error/skip")
    require(sum(int(suite.get("tests", 0)) for suite in suites) == len(cases), "CUDA testcase count mismatch")
    for module in CUDA_TESTS:
        require(any(Path(module).stem in case.get("classname", "") for case in cases),
                f"CUDA test module was not executed: {module}")
    return {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0, "xml_sha256": digest(path)}


def audit_run(output, row, reference, revision, *, write_audit=True, require_cuda=True):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    ref_contract, ref_summary = reference
    contract, summary = read(output / "launch_contract.json"), read(output / SUMMARY)
    run = (output / SUMMARY).parent
    history = read(run / "history.json")["history"]
    for payload in (contract, summary, history):
        finite(payload)
    require(contract["status"] == "complete" and contract["completed_run_count"] == 1, "Run incomplete")
    expected = {"backbones": [BACKBONE], "seeds": [42], "model_role": ROLE, "epochs": 1,
        "batch_size": 128, "lr": .001, "hidden_dim": 64, "grad_clip": 1., "lambda_log_qty": 1., "lambda_tail": 0.,
        "data_sha256": row["data_sha256"], "split_manifest_sha256": row["split_manifest_sha256"],
        "lookback_weeks": row["lookback"], "max_seq_len": row["max_seq_len"],
        "quantity_contract": ref_contract["quantity_contract"], "history_length_contract": ref_contract["history_length_contract"]}
    for key, value in expected.items():
        require(contract[key] == value, f"Launch contract mismatch: {key}")
    require(summary["status"] == "success" and summary["backbone"] == BACKBONE and summary["seed"] == 42,
            "Not a successful candidate run")
    require(contract["source_revision"] == summary["source_revision"] == revision and
            summary["source_revision_history"] == [revision], "Revision/resume mismatch")
    require(not contract["partial_smoke"] and set(contract["split_rows"]) == {"train", "validation"},
            "Partial/test materialization")
    for payload in (contract, summary):
        require(payload["evaluation_scope"] == "validation_only" and payload["held_out_test_evaluated"] is False,
                "Held-out evaluated")
    require(not [path for path in output.rglob("*") if path.name.startswith("test_") or
                 "test_summary" in path.name or "test_metrics" in path.name], "Held-out artifact found")
    for key, value in {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10./3.,
                       "time_intercept_limit": 30., "time_wd_safety_limit": 40., "time_head_lr_multiplier": 1.}.items():
        require(contract["time_head"][key] == value, f"Time launch contract mismatch: {key}")
    for key, value in {"min_epochs": 1, "patience": 1, "monitor": "validation_joint_objective"}.items():
        require(contract["early_stopping"][key] == value, "Selection/epoch contract mismatch")
    best = validate_history(history, summary, row)
    require(history[0]["train_batch_count"] == math.ceil(row["train_targets"] / 128), "Partial train batch count")
    require(contract["time_head"]["train_time_statistics"]["target_count"] == row["train_targets"], "Partial train epoch")
    for key in ("quantity_rows", "history_rows"):
        records = summary[key]
        require(sum(record["count"] for record in records) == row["validation_targets"], "Partial validation epoch")
        require([(r["stratum"], r["count"]) for r in records] ==
                [(r["stratum"], r["count"]) for r in ref_summary[key]], f"Strata changed: {key}")
        mae = sum(r["count"] * r["qty_mae"] for r in records) / row["validation_targets"]
        rmse = math.sqrt(sum(r["count"] * r["qty_rmse"] ** 2 for r in records) / row["validation_targets"])
        for metric, value in (("qty_mae", mae), ("qty_rmse", rmse)):
            require(math.isclose(value, summary["best_val_" + metric], rel_tol=1e-10, abs_tol=1e-8),
                    "Strata aggregate mismatch")
    require(summary["parameter_count"] == row["candidate_parameters"] and
            ref_summary["parameter_count"] == row["baseline_parameters"], "Parameter count mismatch")
    validate_quantity_initialization(summary, ref_summary, row)
    for key, value in {"elapsed_age_contract_id": MODEL_CONTRACT, "elapsed_age_beta_shape": [2, 4],
                       "elapsed_age_parameter_count": 8}.items():
        require(summary["encoder_config"][key] == value, f"Elapsed-age metadata mismatch: {key}")
    for key in ("qty_mae", "qty_rmse", "time_nll", "log_qty_mse", "joint_objective"):
        require(math.isclose(summary["best_val_" + key], best["val_" + key], rel_tol=1e-10, abs_tol=1e-8),
                "Summary/history mismatch")
    checkpoint_path = run / "best_val_joint_objective_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    last = torch.load(run / "last_epoch_state.pt", map_location="cpu", weights_only=False)
    for payload in (checkpoint, last):
        validate_checkpoint_route(payload, BACKBONE)
        require(payload["encoder_config"] == summary["encoder_config"], "Checkpoint/summary encoder metadata mismatch")
        require(payload["source_revision"] == revision and payload["source_revision_history"] == [revision],
                "Checkpoint source/resume mismatch")
        finite_tensors(payload)
    state_hash = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(state_hash == summary["checkpoint_state_sha256"] == canonical_state_dict_sha256(last["best_state_dict"]) ==
            canonical_state_dict_sha256(last["model_state_dict"]), "e1 checkpoint digest mismatch")
    require(last["epoch"] == 1 and last["history"] == history, "Last checkpoint history mismatch")
    groups = last["optimizer_state_dict"]["param_groups"]
    require(len(groups) == 1, "Unexpected optimizer groups")
    for key, value in {"lr": .001, "weight_decay": .01, "eps": 1e-8, "betas": (.9, .999), "amsgrad": False}.items():
        require(groups[0][key] == value, f"Optimizer mismatch: {key}")
    model, _ = build_count_aware_model(BACKBONE, hidden_dim=64, train_log_mean=1.5,
                                      max_seq_len=row["max_seq_len"], quantity_variant=VARIANT)
    parameter_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    require(len(parameter_names) == len(groups[0]["params"]) and len(set(groups[0]["params"])) == len(parameter_names),
            "Optimizer parameter mapping mismatch")
    initial_beta = model.state_dict()[BETA_KEY]
    require(tuple(initial_beta.shape) == (2, 4) and torch.count_nonzero(initial_beta).item() == 0,
            "Fresh candidate beta must initialize at zero")
    beta = checkpoint["model_state_dict"][BETA_KEY]
    require(tuple(beta.shape) == (2, 4) and bool(beta.ne(0).any()), "Beta did not move from fresh zero initialization")
    parameter_id = groups[0]["params"][parameter_names.index(BETA_KEY)]
    optimizer_beta = last["optimizer_state_dict"]["state"][parameter_id]
    for key in ("exp_avg", "exp_avg_sq"):
        require(tuple(optimizer_beta[key].shape) == (2, 4) and bool(optimizer_beta[key].ne(0).any()),
                f"Beta optimizer state is inactive: {key}")
    require(float(optimizer_beta["step"]) == history[0]["train_batch_count"], "Beta optimizer step count mismatch")
    predictions = []
    for state in (checkpoint["model_state_dict"], last["best_state_dict"]):
        model.load_state_dict(state, strict=True)
        model.eval()
        dt, qty = torch.tensor([[1., 2., 5., 3., 0.]]), torch.tensor([[2., 5., 8., 4., 0.]])
        with torch.no_grad():
            predictions.append(target_outputs(model, dt, dt > 0, qty, lambda_log_qty=1.))
    for key in predictions[0]:
        torch.testing.assert_close(predictions[0][key], predictions[1][key], rtol=0, atol=0)
        finite_tensors(predictions[0][key])
    memory = {key: summary.get(key) for key in ("training_device", "cuda_peak_memory_allocated_bytes", "cuda_peak_memory_reserved_bytes")}
    if require_cuda:
        require(str(memory["training_device"]).startswith("cuda"), "CUDA training device not recorded")
        require(all(isinstance(memory[key], int) and memory[key] > 0 for key in
                    ("cuda_peak_memory_allocated_bytes", "cuda_peak_memory_reserved_bytes")), "CUDA peak memory missing")
        require(memory["cuda_peak_memory_reserved_bytes"] >= memory["cuda_peak_memory_allocated_bytes"],
                "CUDA reserved peak smaller than allocated peak")
    result = {"status": "passed", "dataset": row["dataset"], "phase": "full_e1_only",
        "summary_sha256": digest(output / SUMMARY), "history_sha256": digest(run / "history.json"),
        "checkpoint_file_sha256": digest(checkpoint_path), "last_epoch_file_sha256": digest(run / "last_epoch_state.pt"),
        "checkpoint_state_sha256": state_hash, "train_targets": row["train_targets"], "validation_targets": row["validation_targets"],
        "train_batches": history[0]["train_batch_count"], "finite": True, "held_out_test_evaluated": False,
        "checkpoint_replay": "exact_best_vs_last_best_state_cpu", "beta_initial": "all_zero",
        "beta_values": beta.tolist(), "beta_nonzero_count": int(beta.ne(0).sum()),
        "beta_exp_avg": optimizer_beta["exp_avg"].tolist(), "beta_exp_avg_sq": optimizer_beta["exp_avg_sq"].tolist(),
        "beta_optimizer_steps": float(optimizer_beta["step"]), **memory,
        "performance_acceptance": "not_evaluated_e1_only"}
    if write_audit:
        save(output / "audit.json", result)
    return result


def runtime_metadata(spec):
    import polars
    import torch
    actual = {"python": platform.python_version(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "polars": polars.__version__}
    require(actual == spec["runtime"], f"Existing CUDA runtime changed: {actual}")
    require(torch.cuda.is_available(), "CUDA runtime is unavailable")
    return {**actual, "platform": platform.platform(), "executable": sys.executable}


def audit_only(args):
    spec, rows = frozen_documents()
    status = read(args.output_root / "status.json")
    require(status["status"] == "complete" and status["source_revision"] == args.source_revision, "Incomplete or wrong-source run")
    source_hash = verify_source(args.source_revision, args.output_root / "source_manifest.json")
    require(source_hash == status["source_manifest_sha256"], "Run/source manifest mismatch")
    require(read(args.output_root / "execution_contract.json") == spec, "Execution contract changed")
    require({key: status["runtime"][key] for key in spec["runtime"]} == spec["runtime"], "Recorded runtime mismatch")
    require([item["dataset"] for item in status["completed"]] == list(DATASETS), "Run completion scope mismatch")
    tests = audit_cuda_tests(args.output_root / "cuda_contract_tests.xml")
    audits = [audit_run(args.output_root / row["dataset"], row, baseline(row, args.project_root),
                        args.source_revision, write_audit=False) for row in rows]
    result = {"status": "passed", "mode": "local_frozen_outputs", "source_revision": args.source_revision,
              "source_manifest_sha256": source_hash, "cuda_contract_tests": tests, "datasets": audits,
              "held_out_test_evaluated": False, "performance_acceptance": "not_evaluated_e1_only"}
    save(args.output_root / "local_audit.json", result)
    return result


def execute(args):
    if args.audit_only:
        return audit_only(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "status.json"
    status = {"status": "running", "phase": "cuda_and_full_e1_only", "source_revision": args.source_revision,
              "started_at": datetime.now(timezone.utc).isoformat(), "completed": [], "current_dataset": None,
              "held_out_test_evaluated": False, "performance_training_authorized": False}
    save(path, status)
    try:
        spec, rows = frozen_documents()
        status["source_manifest_sha256"] = verify_source(args.source_revision)
        references = {row["dataset"]: baseline(row, args.project_root) for row in rows}
        save(args.output_root / "execution_contract.json", spec)
        save(args.output_root / "source_manifest.json", read(ROOT / "source_manifest.json"))
        status["runtime"] = runtime_metadata(spec)
        status["preflight"] = preflight(spec["gpu_preflight"]["minimum_free_mib"])
        save(path, status)
        run_logged([sys.executable, "-s", "-m", "pytest", "-q", *CUDA_TESTS,
                    f"--junitxml={args.output_root / 'cuda_contract_tests.xml'}"],
                   args.output_root / "cuda_contract_tests.log", env=dict(os.environ, HARD_ELAPSED_AGE_TEST_DEVICE="cuda"))
        status["cuda_contract_tests"] = audit_cuda_tests(args.output_root / "cuda_contract_tests.xml")
        for row in rows:
            status["current_dataset"] = row["dataset"]
            status["preflight"] = preflight(spec["gpu_preflight"]["minimum_free_mib"])
            verify_source(args.source_revision)
            baseline(row, args.project_root)
            cmd = command(row, args.project_root, args.output_root / row["dataset"], args.source_revision)
            status["command"] = cmd
            save(path, status)
            start = time.monotonic()
            run_logged(cmd, args.output_root / f"{row['dataset']}.log")
            audit = audit_run(args.output_root / row["dataset"], row, references[row["dataset"]], args.source_revision)
            status["completed"].append({**audit, "elapsed_seconds": time.monotonic() - start})
            save(path, status)
        status["postflight"] = preflight(spec["gpu_preflight"]["minimum_free_mib"])
        verify_source(args.source_revision)
        status.update(status="complete", current_dataset=None, completed_at=datetime.now(timezone.utc).isoformat())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", failed_at=datetime.now(timezone.utc).isoformat())
        save(path, status)
        raise
    save(path, status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--audit-only", action="store_true", help="Recheck completed frozen outputs locally without training/CUDA execution")
    execute(parser.parse_args())
