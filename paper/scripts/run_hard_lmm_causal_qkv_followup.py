#!/usr/bin/env python3
"""Run one frozen causal-QKV exploratory follow-up on its assigned CUDA host."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    DATASETS,
    evaluate_gate,
    gpu_preflight,
    job_command,
    read_json,
    require,
    save_json,
    sha256,
)


CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_causal_qkv_followup_v1.json"
PROFILER = ROOT / "paper/scripts/profile_hard_lmm_causal_qkv.py"
ORIGINAL_FAILED_DATASET = "insta_market_basket"
NEW_DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root_file(relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe frozen path: {relative}")
    resolved = (ROOT / path).resolve()
    require(resolved.is_relative_to(ROOT.resolve()), f"Frozen path leaves source root: {relative}")
    require(resolved.is_file(), f"Frozen input is missing: {relative}")
    return resolved


def _dataset_file(relative: str) -> Path:
    """Allow the frozen sample-data symlink while still rejecting path injection."""
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"Unsafe dataset path: {relative}")
    joined = ROOT / path
    require(joined.is_file(), f"Frozen dataset input is missing: {relative}")
    return joined


def _verify_file(spec: Mapping[str, Any], *, label: str) -> Path:
    require(isinstance(spec, Mapping), f"{label} specification is missing")
    path = _root_file(str(spec.get("path", "")))
    require(sha256(path) == spec.get("sha256"), f"{label} hash drift")
    return path


def verify_source_binding(contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = _verify_file(contract["implementation_binding"], label="Source manifest")
    manifest = read_json(manifest_path)
    original_revision = contract["original_contract"]["implementation_source_revision"]
    require(
        manifest.get("implementation_source_revision") == original_revision,
        "Pinned implementation revision drift",
    )
    files = manifest.get("files")
    require(isinstance(files, dict) and bool(files), "Pinned source file map is empty")
    checked: dict[str, str] = {}
    for relative, expected in files.items():
        require(isinstance(relative, str) and isinstance(expected, str), "Invalid source hash entry")
        actual = sha256(_root_file(relative))
        require(actual == expected, f"Pinned implementation source drift: {relative}")
        checked[relative] = actual
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "implementation_source_revision": original_revision,
        "verified_file_count": len(checked),
    }


def _reference_dataset(reference: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    rows = reference.get("datasets")
    require(isinstance(rows, list), "B reference dataset rows are missing")
    matching = [row for row in rows if isinstance(row, dict) and row.get("dataset") == dataset]
    require(len(matching) == 1, f"B reference row is missing or duplicated: {dataset}")
    return matching[0]


def _verify_B_reference(contract: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    spec = contract["B_validation_reference"]
    path = _root_file(str(spec["source"]))
    require(sha256(path) == spec["sha256"], "B reference evidence hash drift")
    reference = read_json(path)
    row = _reference_dataset(reference, dataset)
    binding = contract["data_bindings"][dataset]
    evidence = row.get("evidence", {}).get("B_t0_raw_rmse", {})
    metrics = row.get("metrics", {}).get("B_t0_raw_rmse")
    require(isinstance(metrics, dict), f"B metrics are missing: {dataset}")
    require(evidence.get("summary_sha256") == binding["B_summary_sha256"], "B summary hash drift")
    require(
        row.get("validation_identity", {}).get("target_identity_sha256")
        == binding["validation_target_population"]["target_identity_sha256"],
        "B validation identity drift",
    )
    require(metrics.get("validation_target_count") == binding["validation_target_population"]["target_count"],
            "B validation count drift")
    require(metrics.get("body_target_count") == binding["body_target_count"], "B body count drift")
    require(metrics.get("stratum_counts") == binding["stratum_counts"], "B stratum counts drift")
    expected = contract["B_validation_reference"]["datasets"][dataset]
    normalized = {
        "raw_rmse": metrics.get("raw_rmse"),
        "overall_mae": metrics.get("overall_mae"),
        "body_mae": metrics.get("body_mae"),
        "gt_p99_mae": metrics.get("gt_p99_mae"),
        "clamped_time_loss": metrics.get("time_nll"),
    }
    require(normalized == expected, f"Frozen B metrics drift: {dataset}")
    return {"path": str(path), "sha256": sha256(path), "B_summary_sha256": evidence["summary_sha256"]}


def _verify_original_failure(contract: Mapping[str, Any], original: Mapping[str, Any]) -> dict[str, Any]:
    decision_path = _verify_file(contract["preserved_instacart_decision"], label="Instacart decision")
    decision = read_json(decision_path)
    expected = contract["original_gate_verdict_must_remain"]
    require(decision.get("decision") == expected, "Original Instacart verdict drift")
    require(decision.get("dataset") == ORIGINAL_FAILED_DATASET, "Original failed dataset drift")
    original_revision = contract["original_contract"]["implementation_source_revision"]
    require(decision.get("source_revision") == original_revision,
            "Original decision source drift")
    require(decision.get("gate", {}).get("status") == "failed", "Original failure gate was relabelled")
    checks = decision.get("gate", {}).get("checks")
    require(isinstance(checks, dict) and checks.get("raw_rmse") is False,
            "Original raw-RMSE failure is missing")
    require(decision.get("held_out_test_evaluated") is False, "Original decision evaluated held-out test")
    return {"path": str(decision_path), "sha256": sha256(decision_path), "decision": expected}


def _verify_e1_reuse(
    contract: Mapping[str, Any],
    original: Mapping[str, Any],
    dataset: str,
) -> dict[str, Any]:
    campaign_path = _verify_file(contract["reuse_evidence"]["campaign"], label="Original campaign")
    campaign = read_json(campaign_path)
    original_revision = contract["original_contract"]["implementation_source_revision"]
    require(campaign.get("source_revision") == original_revision,
            "Original campaign source drift")
    require(campaign.get("status") == "stopped_seed42_gate_failed", "Original campaign status drift")
    require(campaign.get("failed_dataset") == ORIGINAL_FAILED_DATASET, "Original campaign failure drift")
    job_id = contract["reuse_evidence"]["datasets"][dataset]["job_id"]
    job = campaign.get("jobs", {}).get(job_id)
    require(isinstance(job, dict) and job.get("status") == "passed", "Assigned e1 did not pass")
    audit = job.get("causal_qkv_audit")
    require(isinstance(audit, dict) and audit.get("status") == "passed", "Assigned e1 strict audit did not pass")
    for key in (
        "binding_verified", "best_artifact_route_verified", "last_artifact_route_verified",
        "strict_best_restore", "strict_last_restore", "optimizer_restore",
        "synthetic_target_outputs_roundtrip_equal",
    ):
        require(audit.get(key) is True, f"Assigned e1 lacks proof: {key}")
    evidence = contract["reuse_evidence"]["datasets"][dataset]
    summary_path = _verify_file(evidence["summary"], label=f"{dataset} e1 summary")
    launch_path = _verify_file(evidence["launch"], label=f"{dataset} e1 launch")
    require(job.get("summary_sha256") == evidence["summary"]["sha256"], "e1 summary receipt drift")
    summary = read_json(summary_path)
    launch = read_json(launch_path)
    binding = contract["data_bindings"][dataset]
    require(summary.get("status") == "success" and summary.get("epochs") == 1
            and summary.get("completed_epochs") == 1, "Assigned e1 is not a completed one-epoch run")
    require(summary.get("backbone") == contract["candidate"]["backbone"], "Assigned e1 backbone drift")
    require(summary.get("source_revision") == original_revision,
            "Assigned e1 source drift")
    require(summary.get("evaluation_scope") == "validation_only"
            and summary.get("held_out_test_evaluated") is False, "Assigned e1 scope drift")
    require(launch.get("status") == "complete", "Assigned e1 launch is incomplete")
    require(launch.get("data_sha256") == binding["data_sha256"], "Assigned e1 data hash drift")
    require(launch.get("split_manifest_sha256") == binding["split_manifest_sha256"],
            "Assigned e1 split hash drift")
    require(launch.get("validation_target_population") == binding["validation_target_population"],
            "Assigned e1 validation population drift")
    require(launch.get("quantity_contract", {}).get("boundaries") == binding["quantity_boundaries"],
            "Assigned e1 quantity boundaries drift")
    require(job.get("validation_target_identity_sha256")
            == binding["validation_target_population"]["target_identity_sha256"],
            "Assigned e1 identity receipt drift")
    require(job.get("validation_target_quantity_sha256")
            == binding["validation_target_population"]["target_quantity_sha256"],
            "Assigned e1 quantity receipt drift")
    return {
        "campaign": str(campaign_path),
        "campaign_sha256": sha256(campaign_path),
        "job_id": job_id,
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "launch": str(launch_path),
        "launch_sha256": sha256(launch_path),
        "strict_audit_status": "passed",
    }


def verify_inputs(
    contract: Mapping[str, Any],
    *,
    host_role: str,
    source_revision: str,
) -> dict[str, Any]:
    require(contract.get("contract_id") == "hard_lmm_causal_qkv_followup_v1", "Contract ID drift")
    require(contract.get("scope") == "exploratory_followup_after_instacart_primary_gate_failure",
            "Follow-up scope drift")
    require(contract.get("evaluation_scope") == "validation_only", "Evaluation scope drift")
    require(contract.get("held_out_test_evaluated") is False, "Held-out evaluation was authorized")
    require(contract.get("additional_seeds_authorized") is False, "Additional seeds were authorized")
    assignment = contract.get("host_assignments", {}).get(host_role)
    require(isinstance(assignment, dict), f"Unknown host role: {host_role}")
    dataset = assignment.get("dataset")
    expected_dataset = {"5080": "yellow_trip_hourly", "5090": "intermittent_frozen_5000"}[host_role]
    require(dataset == expected_dataset, "Host/dataset assignment drift")
    require(dataset in NEW_DATASETS, "Follow-up dataset is outside authorization")

    original_path = _verify_file(contract["original_contract"], label="Original contract")
    original = read_json(original_path)
    require(original.get("contract_id") == "hard_lmm_causal_qkv_screening_v1", "Original contract ID drift")
    require(original.get("candidate", {}).get("backbone") == contract["candidate"]["backbone"],
            "Candidate backbone differs from original contract")
    require(original.get("candidate", {}).get("model_role") == contract["candidate"]["model_role"],
            "Candidate role differs from original contract")
    require(original.get("data_bindings", {}).get(dataset) == contract["data_bindings"][dataset],
            "Dataset binding differs from original contract")
    require(source_revision != contract["original_contract"]["implementation_source_revision"],
            "Follow-up orchestration requires its new source revision")

    source = verify_source_binding(contract)
    failure = _verify_original_failure(contract, original)
    e1 = _verify_e1_reuse(contract, original, dataset)
    cost_path = _verify_file(contract["reuse_evidence"]["cost_profile"], label="Original cost profile")
    cost = read_json(cost_path)
    require(cost.get("status") == "passed", "Original cost profile did not pass")
    B_reference = _verify_B_reference(contract, dataset)
    spec = DATASETS[dataset]
    binding = contract["data_bindings"][dataset]
    current_data = sha256(_dataset_file(spec["data"]))
    current_split = sha256(_dataset_file(spec["manifest"]))
    require(current_data == binding["data_sha256"], "Current data hash drift")
    require(current_split == binding["split_manifest_sha256"], "Current split hash drift")
    return {
        "dataset": dataset,
        "original_contract": {"path": str(original_path), "sha256": sha256(original_path)},
        "pinned_source": source,
        "preserved_failure": failure,
        "reused_e1": e1,
        "reused_5090_cost_profile": {
            "path": str(cost_path), "sha256": sha256(cost_path),
            "scope": "architecture_only_not_host_speed_evidence",
        },
        "B_reference": B_reference,
        "data_sha256": current_data,
        "split_manifest_sha256": current_split,
    }


def runtime_probe(python: str, expected: Mapping[str, Any], expected_gpu: str) -> dict[str, Any]:
    code = (
        "import json,platform,torch;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
        "'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},sort_keys=True))"
    )
    output = subprocess.check_output([python, "-s", "-c", code], cwd=ROOT, text=True).strip()
    payload = json.loads(output.splitlines()[-1])
    for key in ("python", "torch", "cuda"):
        require(payload.get(key) == str(expected[key]), f"Runtime {key} drift")
    require(payload.get("cuda_available") is True, "CUDA is unavailable in the training runtime")
    require(expected_gpu in str(payload.get("device_name")), "Training runtime sees the wrong GPU")
    return payload


def validate_cuda_junit(path: Path, required_tests: list[str]) -> dict[str, Any]:
    require(path.is_file(), "CUDA qualification JUnit receipt is missing")
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    require(bool(cases), "CUDA qualification JUnit contains no tests")
    failed = [case for case in cases if case.find("failure") is not None or case.find("error") is not None]
    skipped = [case for case in cases if case.find("skipped") is not None]
    require(not failed, "CUDA qualification has failed/error tests")
    require(not skipped, "CUDA qualification contains skipped tests")
    names = [str(case.get("name")) for case in cases]
    for required in required_tests:
        require(names.count(required) == 1, f"Required actual CUDA test is missing or duplicated: {required}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "test_count": len(cases),
        "required_actual_cuda_tests": list(required_tests),
        "skipped": 0,
        "failed": 0,
    }


def run_cuda_qualification(
    *, python: str, contract: Mapping[str, Any], output_root: Path,
) -> dict[str, Any]:
    junit = output_root / "cuda_qualification.xml"
    log_path = output_root / "cuda_qualification.log"
    tests = [str(_root_file(path)) for path in contract["qualification"]["tests"]]
    command = [python, "-s", "-m", "pytest", "-q", *tests, f"--junitxml={junit}"]
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = validate_cuda_junit(junit, contract["qualification"]["required_actual_cuda_tests"])
    result.update(command=command, log=str(log_path), log_sha256=sha256(log_path))
    return result


def validate_synthetic_worker(payload: Mapping[str, Any], *, expected_gpu: str) -> dict[str, Any]:
    require(payload.get("status") == "passed", "Synthetic CUDA worker failed")
    require(payload.get("backbone") == "titantpp_hard_memory_causal_qkv", "Synthetic backbone drift")
    require(payload.get("sequence_length") == 256 and payload.get("repeat") == 0,
            "Synthetic worker policy drift")
    require(payload.get("observed_events") == 255 and payload.get("target_events") == 1,
            "Synthetic causal population drift")
    require(payload.get("finite_model_optimizer") is True, "Synthetic state is non-finite")
    require(payload.get("learning_path", {}).get("status") == "passed",
            "Synthetic learning path did not pass")
    require(payload.get("learning_path", {}).get("state_digest_restored") is True,
            "Synthetic counterfactual did not restore state")
    require(expected_gpu in str(payload.get("device_name")), "Synthetic worker used the wrong GPU")
    for key in ("median_step_seconds", "peak_allocated_bytes"):
        value = float(payload.get(key, 0.0))
        require(math.isfinite(value) and value > 0.0, f"Synthetic worker has invalid {key}")
    return {
        "status": "passed", "device_name": payload["device_name"],
        "sequence_length": 256, "repeat": 0,
        "peak_allocated_bytes": int(payload["peak_allocated_bytes"]),
        "finite_model_optimizer": True, "learning_path": payload["learning_path"],
    }


def run_synthetic_worker(
    *, python: str, contract: Mapping[str, Any], assignment: Mapping[str, Any], output_root: Path,
) -> dict[str, Any]:
    spec = contract["qualification"]["synthetic_worker"]
    output = output_root / "synthetic_cuda_worker.json"
    log_path = output_root / "synthetic_cuda_worker.log"
    command = [
        python, "-s", str(PROFILER), "--worker", "--backbone",
        contract["candidate"]["backbone"], "--sequence-length", str(spec["sequence_length"]),
        "--repeat", str(spec["repeat"]), "--output", str(output),
    ]
    with log_path.open("x", encoding="utf-8") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    payload = read_json(output)
    result = validate_synthetic_worker(payload, expected_gpu=assignment["gpu_name_contains"])
    result.update(path=str(output), sha256=sha256(output), command=command,
                  log=str(log_path), log_sha256=sha256(log_path),
                  scope="finite_learning_device_smoke_not_relative_cost_claim")
    return result


def evaluate_replication_review(
    metrics_by_dataset: Mapping[str, Mapping[str, float]],
    baseline_by_dataset: Mapping[str, Mapping[str, float]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    datasets = list(gate["datasets"])
    require(set(metrics_by_dataset) == set(datasets), "Replication review candidate dataset set drift")
    require(set(baseline_by_dataset) == set(datasets), "Replication review baseline dataset set drift")
    ratios: dict[str, dict[str, float]] = {}
    all_checks: dict[str, bool] = {}
    for dataset in datasets:
        candidate = metrics_by_dataset[dataset]
        baseline = baseline_by_dataset[dataset]
        row: dict[str, float] = {}
        for metric in ("raw_rmse", "overall_mae", "body_mae", "gt_p99_mae"):
            numerator, denominator = float(candidate[metric]), float(baseline[metric])
            require(math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0,
                    f"Invalid replication metric: {dataset}.{metric}")
            row[metric] = numerator / denominator
        time_increase = float(candidate["clamped_time_loss"]) - float(baseline["clamped_time_loss"])
        require(math.isfinite(time_increase), f"Invalid replication time metric: {dataset}")
        row["clamped_time_loss_increase"] = time_increase
        ratios[dataset] = row
        all_checks[f"{dataset}.raw_rmse"] = row["raw_rmse"] <= float(gate["all_dataset_max_raw_rmse_ratio"])
        all_checks[f"{dataset}.overall_mae"] = row["overall_mae"] <= float(gate["all_dataset_max_overall_mae_ratio"])
        all_checks[f"{dataset}.body_mae"] = row["body_mae"] <= float(gate["all_dataset_max_body_mae_ratio"])
        all_checks[f"{dataset}.gt_p99_mae"] = row["gt_p99_mae"] <= float(gate["all_dataset_max_gt_p99_mae_ratio"])
        all_checks[f"{dataset}.clamped_time_loss"] = time_increase <= float(gate["all_dataset_max_time_loss_increase"])
    new_improvement = {
        dataset: ratios[dataset]["raw_rmse"]
        <= float(gate["at_least_one_new_dataset_max_raw_rmse_ratio"])
        for dataset in NEW_DATASETS
    }
    passed = all(all_checks.values()) and any(new_improvement.values())
    return {
        "status": gate["eligible_outcome"] if passed else gate["otherwise"],
        "all_dataset_checks": all_checks,
        "new_dataset_raw_rmse_improvement_checks": new_improvement,
        "ratios": ratios,
        "scope": "replication_resource_review_only_not_model_adoption",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-role", choices=("5080", "5090"), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        parser.error("--source-revision must be a full lowercase Git SHA")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from paper.scripts.audit_hard_lmm_causal_qkv import audit_causal_qkv_job

    contract = read_json(CONTRACT_PATH)
    output_root = args.output_root.resolve()
    require(not output_root.exists() or not any(output_root.iterdir()),
            "Output root is occupied; implicit retry/overwrite is forbidden")
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "followup_status.json"
    assignment = contract["host_assignments"][args.host_role]
    dataset = assignment["dataset"]
    status: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256(CONTRACT_PATH),
        "source_revision": args.source_revision,
        "host_role": args.host_role,
        "assigned_dataset": dataset,
        "candidate": contract["candidate"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "additional_seeds_run": False,
        "started_at": utc_now(),
        "status": "preflight",
    }

    def persist() -> None:
        status["updated_at"] = utc_now()
        save_json(status_path, status)

    persist()
    try:
        status["input_evidence"] = verify_inputs(
            contract, host_role=args.host_role, source_revision=args.source_revision,
        )
        save_json(output_root / "frozen_contract.json", dict(contract))
        status["gpu_preflight"] = gpu_preflight(assignment["gpu_name_contains"])
        status["runtime"] = runtime_probe(
            args.python, contract["required_runtime"], assignment["gpu_name_contains"],
        )
        status["status"] = "running_cuda_qualification"
        persist()
        status["cuda_qualification"] = run_cuda_qualification(
            python=args.python, contract=contract, output_root=output_root,
        )
        status["status"] = "running_synthetic_worker"
        persist()
        status["synthetic_worker"] = run_synthetic_worker(
            python=args.python, contract=contract, assignment=assignment, output_root=output_root,
        )

        phase = contract["training_phase"]
        output = output_root / "jobs" / f"seed42_{dataset}"
        output.mkdir(parents=True, exist_ok=False)
        command = job_command(
            python=args.python, candidate=dict(contract["candidate"]), dataset=dataset,
            output=output, source_revision=args.source_revision, phase=dict(phase),
            host_role=args.host_role,
        )
        status["status"] = "running_seed42"
        status["job"] = {
            "status": "running", "started_at": utc_now(), "output": str(output), "command": command,
        }
        persist()
        log_path = output / "followup_runner.log"
        with log_path.open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        audit = audit_causal_qkv_job(
            output, candidate=dict(contract["candidate"]), dataset=dataset,
            source_revision=args.source_revision, expected_epochs=int(phase["epochs"]),
            binding=dict(contract["data_bindings"][dataset]),
        )
        require(audit.get("status") == "passed", "Follow-up strict artifact audit did not pass")
        dataset_gate = evaluate_gate(
            audit["metrics"], contract["B_validation_reference"]["datasets"][dataset],
        )
        status["job"].update(
            audit, status="passed", completed_at=utc_now(), log=str(log_path),
            log_sha256=sha256(log_path), dataset_reporting_gate=dataset_gate,
        )
        status.update(
            status="completed_execution_and_audit",
            execution_completed=True,
            dataset_reporting_gate_status=dataset_gate["status"],
            replication_review_status="pending_parallel_result_merge",
            completed_at=utc_now(),
        )
        persist()
    except BaseException as error:
        status.update(
            status="failed_execution_or_audit", execution_completed=False,
            error=f"{type(error).__name__}: {error}", completed_at=utc_now(),
        )
        persist()
        raise


if __name__ == "__main__":
    main()
