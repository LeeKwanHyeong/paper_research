#!/usr/bin/env python3
"""Fresh seed42 elapsed-age e300 with frozen validation gates; no test or retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import run_hard_lmm_elapsed_age_smoke as smoke
from paper.scripts.run_hard_lmm_elapsed_age_smoke import (
    BACKBONE, ROLE, MODEL_CONTRACT, DATASETS, VARIANT, SUMMARY, BETA_KEY,
    finite_tensors, validate_history, digest, finite, preflight, read, require,
    run_logged, save, validate_quantity_initialization,
)

CONTRACT = "paper/contracts/hard_lmm_elapsed_age_screening_5090_v1.json"
MODEL_CONTRACT_PATH = "paper/contracts/hard_lmm_elapsed_age_v1.json"
EXECUTION_ROLE = "elapsed_age_static_e300_5090"


def training_files(manifest):
    # Pin every existing script dependency as well as model/data/common code.
    # Only the new orchestration pair is intentionally absent from e1 source.
    prefixes = ("models/", "data_loader/", "utils/", "simple_lab_test/common/",
                "simple_lab_test/search/common/", "paper/scripts/")
    orchestration = {"paper/scripts/run_hard_lmm_elapsed_age_screening.py",
                     "paper/scripts/package_hard_lmm_elapsed_age_screening.py"}
    return {name: sha for name, sha in manifest["files"].items()
            if name.startswith(prefixes) and name.endswith(".py") and name not in orchestration}


def frozen_documents():
    spec = read(ROOT / CONTRACT)
    for name, expected in spec["frozen_files"].items():
        require(digest(ROOT / name) == expected, f"Frozen document changed: {name}")
    _, rows = smoke.frozen_documents()
    require(spec["dataset_order"] == list(DATASETS) and [row["dataset"] for row in spec["datasets"]] == list(DATASETS),
            "Dataset scope changed")
    for declared, row in zip(spec["datasets"], rows):
        reference_path = smoke.reference_path(row, "launch_contract.json")
        require(digest(reference_path) == row["contract_sha256"], "Original launch contract changed")
        reference = read(reference_path)
        quantities = dict(zip(reference["quantity_contract"]["quantiles"], reference["quantity_contract"]["boundaries"]))
        expected = row | {"time_unit": reference["dataset_time_unit"],
                          "train_p95_quantity": quantities[.95], "train_p99_quantity": quantities[.99]}
        require(all(key in expected and expected[key] == value for key, value in declared.items()), "Dataset contract changed")
    require(spec["run"] == {"seed": 42, "epochs": 300, "min_epochs": 40, "patience": 40,
            "batch_size": 128, "lr": .001, "full_train_validation": True}, "Screening budget changed")
    require(spec["backbone"] == BACKBONE and spec["model_role"] == ROLE, "Candidate identity changed")
    require(spec["performance_training_authorized"] is True and spec["held_out_test"] is False and
            spec["fresh_parameters_and_artifact"] is True and spec["automatic_resume_retry"] is False,
            "Screening authorization/scope changed")
    model = read(ROOT / MODEL_CONTRACT_PATH)
    require(spec["performance_gates"] == model["prospective_performance_gate_not_evaluated_in_e1"],
            "Prospective performance gates changed")
    return spec, rows


def verify_source(revision, manifest_path=None):
    path = manifest_path or ROOT / "source_manifest.json"
    manifest = read(path)
    spec, rows = frozen_documents()
    require(manifest["source_revision"] == revision and len(revision) == 40, "Source revision mismatch")
    require(manifest["performance_training_authorized"] is True and manifest["scope"] == spec["scope"],
            "Source package not authorized for screening")
    required = {CONTRACT, MODEL_CONTRACT_PATH, "models/Titan/common/elapsed_age.py",
                "paper/scripts/run_hard_lmm_elapsed_age_screening.py", *spec["frozen_files"]}
    references = {str(Path("references") / row["dataset"] / name): smoke.reference_path(row, name)
                  for row in rows for name in smoke.reference_files(row)}
    require(required | set(references) <= set(manifest["files"]), "Incomplete source/reference manifest")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe source manifest path")
        require(digest(references.get(name, ROOT / name)) == expected, f"Source/reference mismatch: {name}")
    return digest(path)


def verify_smoke(spec, rows, references):
    snapshot, artifact = Path(spec["smoke_snapshot"]), Path(spec["smoke_output_root"])
    manifest_path = snapshot / "source_manifest.json"
    require(digest(manifest_path) == spec["smoke_source_manifest_sha256"], "Smoke manifest changed")
    old = read(manifest_path)
    require(old["source_revision"] == spec["smoke_source_revision"], "Smoke source revision changed")
    require(training_files(old) == training_files(read(ROOT / "source_manifest.json")),
            "Training source differs from CUDA/e1 validated source")
    for name, expected in old["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe smoke manifest path")
        require(digest(snapshot / name) == expected, f"Smoke source/reference changed: {name}")
    require(digest(artifact / "status.json") == spec["smoke_status_sha256"], "Smoke status changed")
    status = read(artifact / "status.json")
    require(status["status"] == "complete" and status["source_revision"] == spec["smoke_source_revision"] and
            status["source_manifest_sha256"] == spec["smoke_source_manifest_sha256"], "Smoke incomplete/changed")
    require(digest(artifact / "source_manifest.json") == spec["smoke_source_manifest_sha256"], "Smoke output manifest changed")
    require(read(artifact / "execution_contract.json") == read(ROOT / smoke.CONTRACT), "Smoke execution contract changed")
    require({key: status["runtime"][key] for key in spec["runtime"]} == spec["runtime"], "Smoke runtime changed")
    require([item["dataset"] for item in status["completed"]] == list(DATASETS), "Smoke dataset scope changed")
    tests = smoke.audit_cuda_tests(artifact / "cuda_contract_tests.xml")
    require(tests["xml_sha256"] == spec["smoke_cuda_xml_sha256"] and
            tests["tests"] == spec["smoke_cuda_test_count"] and tests == status["cuda_contract_tests"],
            "CUDA evidence changed")
    audits = [smoke.audit_run(artifact / row["dataset"], row, references[row["dataset"]],
                             spec["smoke_source_revision"], write_audit=False) for row in rows]
    for audit, recorded in zip(audits, status["completed"]):
        require(all(recorded[key] == value for key, value in audit.items()), "Smoke audit changed")
    return {"status": "passed", "source_manifest_sha256": digest(manifest_path),
            "status_sha256": digest(artifact / "status.json"), "cuda_contract_tests": tests, "datasets": audits}


def command(row, project, output, revision):
    cmd = smoke.command(row, project, output, revision)
    for flag, value in (("--epochs", "300"), ("--min-epochs", "40"),
                        ("--early-stopping-patience", "40"), ("--execution-role", EXECUTION_ROLE)):
        cmd[cmd.index(flag) + 1] = value
    return cmd


def metrics(summary):
    finite(summary)
    rows = summary["quantity_rows"]
    strata = [row["stratum"] for row in rows]
    require(len(set(strata)) == len(strata), "Duplicate quantity strata")
    body = [row for row in rows if row["stratum"] in {"le_p50", "p50_p90", "p90_p95"}]
    require(len(body) == 3 and sum(row["count"] for row in body) > 0, "Missing/empty body strata")
    tail = [row for row in rows if row["stratum"] == "gt_p99"]
    require(len(tail) == 1 and tail[0]["count"] > 0, "Missing/empty tail stratum")
    return {"body_mae": sum(row["count"] * row["qty_mae"] for row in body) / sum(row["count"] for row in body),
            "quantity_mae": summary["best_val_qty_mae"], "quantity_rmse": summary["best_val_qty_rmse"],
            "gt_p99_mae": tail[0]["qty_mae"], "time_nll": summary["best_val_time_nll"]}


def compare(dataset, candidate, original, separate_key, gates):
    """Inclusive frozen upper bounds; all original and separate-key checks apply."""
    require(dataset in DATASETS, "Unknown screening dataset")
    for payload in (candidate, original, separate_key):
        finite(payload)
        require(all(payload[key] >= 0 for key in ("body_mae", "quantity_mae", "quantity_rmse", "gt_p99_mae")),
                "Invalid error metric")
    require(all(reference[key] > 0 for reference in (original, separate_key)
                for key in ("body_mae", "quantity_mae", "quantity_rmse", "gt_p99_mae")), "Invalid reference denominator")
    original_gate, kv_gate = gates["both_datasets_vs_original"], gates["both_datasets_vs_separate_key"]
    original_bounds = {"body_mae": original["body_mae"] * (1-original_gate["body_le_p95_mae_improvement_min"]),
        "quantity_rmse": original["quantity_rmse"] * (1+original_gate["overall_rmse_regression_max"]),
        "gt_p99_mae": original["gt_p99_mae"] * (1+original_gate["gt_p99_mae_regression_max"]),
        "time_nll": original["time_nll"] + original_gate["time_nll_absolute_increase_max"]}
    body_factor = (1-gates["instacart_vs_separate_key"]["body_le_p95_mae_improvement_min"]
                   if dataset == "insta_market_basket" else 1+gates["taxi_vs_separate_key"]["body_le_p95_mae_regression_max"])
    kv_bounds = {"body_mae": separate_key["body_mae"] * body_factor,
        "quantity_rmse": separate_key["quantity_rmse"] * (1+kv_gate["overall_rmse_regression_max"]),
        "gt_p99_mae": separate_key["gt_p99_mae"] * (1+kv_gate["gt_p99_mae_regression_max"]),
        "time_nll": separate_key["time_nll"] + kv_gate["time_nll_absolute_increase_max"]}
    checks = {name: {key: candidate[key] <= bound for key, bound in bounds.items()}
              for name, bounds in (("vs_original", original_bounds), ("vs_separate_key", kv_bounds))}
    relative = {name: {key: candidate[key]/reference[key]-1 for key in
                      ("body_mae", "quantity_mae", "quantity_rmse", "gt_p99_mae")}
                for name, reference in (("vs_original", original), ("vs_separate_key", separate_key))}
    return {"original": original, "separate_key": separate_key, "candidate": candidate,
        "relative_changes": relative, "time_nll_delta": {"vs_original": candidate["time_nll"]-original["time_nll"],
        "vs_separate_key": candidate["time_nll"]-separate_key["time_nll"]},
        "upper_bounds": {"vs_original": original_bounds, "vs_separate_key": kv_bounds,
                         "simultaneous": {key: min(original_bounds[key], kv_bounds[key]) for key in original_bounds}},
        "gates": checks, "finite": True, "passed": all(all(check.values()) for check in checks.values())}


def compare_run(dataset, candidate, reference, spec):
    model = read(ROOT / MODEL_CONTRACT_PATH)
    pinned = read(ROOT / model["references"]["separate_key_validation_comparison"])["datasets"][dataset]
    original = metrics(reference)
    require(original == pinned["baseline"], "Original metrics differ from full-precision pinned reference")
    return compare(dataset, metrics(candidate), original, pinned["candidate"], spec["performance_gates"])


def comparison_document(comparisons, spec):
    return {"candidate": BACKBONE, "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "gate_source": MODEL_CONTRACT_PATH, "gates": spec["performance_gates"], "datasets": comparisons,
        "overall": {"passed_datasets": sum(item["passed"] for item in comparisons.values()),
                    "evaluated_datasets": len(comparisons),
                    "all_frozen_gates_passed": len(comparisons) == len(DATASETS) and all(item["passed"] for item in comparisons.values()),
                    "performance_adopted": False, "additional_runs_authorized": False}}


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
    expected = {"backbones": [BACKBONE], "seeds": [42], "model_role": ROLE, "epochs": 300,
        "execution_role": EXECUTION_ROLE,
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
    for key, value in {"min_epochs": 40, "patience": 40, "monitor": "validation_joint_objective"}.items():
        require(contract["early_stopping"][key] == value, "Selection/epoch contract mismatch")
    best = validate_history(history, summary, row, screening=True)
    require(all(h["train_batch_count"] == math.ceil(row["train_targets"] / 128) for h in history), "Partial train batch count")
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
        require(payload["evaluation_scope"] == "validation_only" and payload["held_out_test_evaluated"] is False and
                payload["seed"] == 42 and payload["variant"] == VARIANT, "Checkpoint scope mismatch")
        require(payload["encoder_config"] == summary["encoder_config"], "Checkpoint/summary encoder metadata mismatch")
        require(payload["source_revision"] == revision and payload["source_revision_history"] == [revision],
                "Checkpoint source/resume mismatch")
        finite_tensors(payload)
    state_hash = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(state_hash == summary["checkpoint_state_sha256"] == canonical_state_dict_sha256(last["best_state_dict"]), "Best checkpoint digest mismatch")
    require(last["epoch"] == len(history) and last["history"] == history, "Last checkpoint history mismatch")
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
    last_beta = last["model_state_dict"][BETA_KEY]
    require(tuple(last_beta.shape) == (2, 4) and bool(last_beta.ne(0).any()), "Last beta did not move")
    parameter_id = groups[0]["params"][parameter_names.index(BETA_KEY)]
    optimizer_beta = last["optimizer_state_dict"]["state"][parameter_id]
    for key in ("exp_avg", "exp_avg_sq"):
        require(tuple(optimizer_beta[key].shape) == (2, 4) and bool(optimizer_beta[key].ne(0).any()),
                f"Beta optimizer state is inactive: {key}")
    expected_steps = sum(h["train_batch_count"] for h in history)
    require(float(optimizer_beta["step"]) == expected_steps, "Beta optimizer step count mismatch")
    for state in last["optimizer_state_dict"]["state"].values():
        require(float(state["step"]) == expected_steps, "Optimizer step count mismatch")
    model.load_state_dict(last["model_state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer.load_state_dict(last["optimizer_state_dict"])
    require(len(optimizer.state) == len(parameter_names), "Optimizer restoration omitted parameters")
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
    result = {"status": "passed", "dataset": row["dataset"], "phase": "seed42_e300",
        "completed_epochs": len(history), "best_epoch": best["epoch"], "stopped_early": summary["stopped_early"],
        "summary_sha256": digest(output / SUMMARY), "history_sha256": digest(run / "history.json"),
        "checkpoint_file_sha256": digest(checkpoint_path), "last_epoch_file_sha256": digest(run / "last_epoch_state.pt"),
        "checkpoint_state_sha256": state_hash, "train_targets": row["train_targets"], "validation_targets": row["validation_targets"],
        "train_batches_per_epoch": history[0]["train_batch_count"], "total_train_batches": expected_steps, "finite": True, "held_out_test_evaluated": False,
        "checkpoint_replay": "exact_best_vs_last_best_state_cpu", "beta_initial": "all_zero",
        "beta_values": beta.tolist(), "beta_nonzero_count": int(beta.ne(0).sum()),
        "last_beta_values": last_beta.tolist(), "last_beta_nonzero_count": int(last_beta.ne(0).sum()),
        "last_checkpoint_state_sha256": canonical_state_dict_sha256(last["model_state_dict"]),
        "beta_exp_avg": optimizer_beta["exp_avg"].tolist(), "beta_exp_avg_sq": optimizer_beta["exp_avg_sq"].tolist(),
        "beta_optimizer_steps": float(optimizer_beta["step"]), **memory,
        "performance_acceptance": "separate_frozen_comparison_required"}
    if write_audit:
        save(output / "audit.json", result)
    return result


def audit_only(args):
    spec, rows = frozen_documents()
    status = read(args.output_root / "status.json")
    require(status["status"] == "complete" and status["source_revision"] == args.source_revision, "Incomplete/wrong-source run")
    source_hash = verify_source(args.source_revision, args.output_root / "source_manifest.json")
    require(source_hash == status["source_manifest_sha256"], "Run/source manifest mismatch")
    require(read(args.output_root / "execution_contract.json") == spec, "Execution contract changed")
    require({key: status["runtime"][key] for key in spec["runtime"]} == spec["runtime"], "Recorded runtime mismatch")
    require([item["dataset"] for item in status["completed"]] == list(DATASETS), "Run completion scope mismatch")
    comparisons, audits = {}, []
    for row, recorded in zip(rows, status["completed"]):
        reference = smoke.baseline(row, args.project_root)
        audit = audit_run(args.output_root / row["dataset"], row, reference, args.source_revision, write_audit=False)
        require(all(recorded[key] == value for key, value in audit.items()), "Recorded run audit changed")
        audits.append(audit)
        comparisons[row["dataset"]] = compare_run(row["dataset"], read(args.output_root / row["dataset"] / SUMMARY), reference[1], spec)
    comparison = comparison_document(comparisons, spec)
    require(read(args.output_root / "validation_comparison.json") == comparison, "Recorded comparison changed")
    result = {"status": "passed", "mode": "local_frozen_outputs", "source_revision": args.source_revision,
        "source_manifest_sha256": source_hash, "datasets": audits, "comparison": comparison,
        "held_out_test_evaluated": False}
    save(args.output_root / "local_audit.json", result)
    return result


def execute(args):
    if args.audit_only:
        return audit_only(args)
    # The training entry can resume existing state: never enter an existing root.
    args.output_root.mkdir(parents=True, exist_ok=False)
    path = args.output_root / "status.json"
    status = {"status": "running", "phase": "candidate_only_seed42_e300", "source_revision": args.source_revision,
        "started_at": datetime.now(timezone.utc).isoformat(), "completed": [], "current_dataset": None,
        "held_out_test_evaluated": False, "performance_training_authorized": True}
    save(path, status)
    try:
        spec, rows = frozen_documents()
        status["source_manifest_sha256"] = verify_source(args.source_revision)
        status["runtime"] = smoke.runtime_metadata(spec)
        references = {row["dataset"]: smoke.baseline(row, args.project_root) for row in rows}
        status["smoke_gate"] = verify_smoke(spec, rows, references)
        save(args.output_root / "execution_contract.json", spec)
        save(args.output_root / "source_manifest.json", read(ROOT / "source_manifest.json"))
        comparisons = {}
        for row in rows:
            name = row["dataset"]
            status["current_dataset"] = name
            status["preflight"] = preflight(spec["gpu_preflight"]["minimum_free_mib"])
            verify_source(args.source_revision)
            smoke.baseline(row, args.project_root)
            require(not (args.output_root / name).exists(), "Dataset output already exists; refusing resume")
            cmd = command(row, args.project_root, args.output_root / name, args.source_revision)
            status.update(command=cmd, current_started_at=datetime.now(timezone.utc).isoformat())
            save(path, status)
            start = time.monotonic()
            run_logged(cmd, args.output_root / f"{name}.log")
            audit = audit_run(args.output_root / name, row, references[name], args.source_revision)
            comparisons[name] = compare_run(name, read(args.output_root / name / SUMMARY), references[name][1], spec)
            save(args.output_root / "validation_comparison.json", comparison_document(comparisons, spec))
            status["completed"].append({**audit, "elapsed_seconds": time.monotonic()-start, "gate_passed": comparisons[name]["passed"]})
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
    parser.add_argument("--audit-only", action="store_true", help="Audit saved outputs locally without training or CUDA execution")
    execute(parser.parse_args())
