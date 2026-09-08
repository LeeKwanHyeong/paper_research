#!/usr/bin/env python3
"""Run the preregistered BOUNDED-QK campaign on an idle RTX 5090.

The deployment manifest binds the entire committed checkout and runtime before
any GPU output. Joint quantity training and frozen duration fitting are separate
jobs. An execution, quantity, or normalized-time guardrail failure ends the queue.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    DATASETS, gpu_preflight, job_command, read_json, require, save_json, sha256,
)

CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_bounded_qk_screening_v1.json"
PROFILER = ROOT / "paper/scripts/profile_hard_lmm_bounded_qk.py"
DURATION_RUNNER = ROOT / "paper/scripts/run_backbone_normalized_duration.py"
VARIANT = "count_only_log_regression"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_file(path: str) -> Path:
    logical = Path(path)
    require(not logical.is_absolute() and ".." not in logical.parts,
            f"Expected a safe root-relative path: {path}")
    result = ROOT / logical
    require(result.is_file(), f"Missing frozen input: {path}")
    return result


def verify_source(manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema_version") == 1, "Deployment schema drift")
    require(manifest.get("host_role") == "5090", "Deployment is not for 5090")
    require(manifest.get("status") == "frozen_before_gpu_outputs", "Deployment status drift")
    require(manifest.get("source_root") == str(ROOT.resolve()), "Source root drift")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(revision == manifest.get("source_revision"), "Checkout revision drift")
    files = manifest.get("source_files")
    require(isinstance(files, dict) and bool(files), "Source file map is empty")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    require(set(files) == {p for p in tracked if p}, "Tracked source population drift")
    for path, expected in files.items():
        require(sha256(root_file(path)) == expected, f"Source file drift: {path}")
    # Imported local Python code must come from the committed source package.
    for folder in ("models", "paper/scripts", "simple_lab_test"):
        extras = {str(p.relative_to(ROOT)) for p in (ROOT / folder).rglob("*.py")} - set(files)
        require(not extras, f"Unregistered Python source: {sorted(extras)}")
    require(sha256(CONTRACT_PATH) == manifest.get("execution_contract_sha256"),
            "Campaign contract digest drift")
    return {"source_revision": revision, "verified_files": len(files)}


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(contract.get("contract_id") == "hard_lmm_bounded_qk_screening_v1", "Contract ID drift")
    require(contract.get("host_role") == "5090", "Host contract drift")
    require(contract.get("evaluation_scope") == "validation_only"
            and contract.get("held_out_test_evaluated") is False
            and contract.get("additional_seeds_authorized") is False, "Evaluation scope drift")
    require(contract.get("candidate", {}).get("backbone") == "titantpp_hard_memory_bounded_qk"
            and contract["candidate"].get("model_role") == "hard_lmm_bounded_qk_candidate",
            "Candidate route drift")
    require(contract["phases"]["e1"] == {
        "epochs": 1, "minimum_epochs": 1, "patience": 40,
        "datasets_in_order": ["yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket"],
        "performance_gate": False,
    }, "e1 contract drift")
    require(contract["phases"]["seed42_screening"] == {
        "epochs": 300, "minimum_epochs": 40, "patience": 40,
        "datasets_in_order": ["insta_market_basket", "yellow_trip_hourly", "intermittent_frozen_5000"],
        "stop_on_first_quantity_or_normalized_time_guardrail_failure": True,
    }, "Screening contract drift")
    require(contract["quantity_gate"] == {
        "B": {"raw_rmse_ratio_strictly_less_than": 1.0, "overall_mae_ratio_max": 1.01,
              "body_mae_ratio_max": 1.02, "gt_p99_mae_ratio_max": 1.02},
        "FULL": {"raw_rmse_ratio_max": 1.01, "overall_mae_ratio_max": 1.01,
                 "body_mae_ratio_max": 1.02, "gt_p99_mae_ratio_max": 1.02},
    }, "Quantity gate drift")
    require(contract["time_gate"] == {
        "normalized_NLL_delta_vs_aligned_B_max": 0.01,
        "legacy_time_score_used_for_selection_or_adoption": False,
        "taxi_NLL_improvement_vs_FULL_min_for_separate_claim": 0.005,
        "taxi_FULL_claim_failure_alone_stops_campaign": False,
    }, "Time gate drift")


def verify_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate fixed source artifacts and dataset bytes without loading rows."""
    from paper.scripts import run_backbone_normalized_duration as duration
    from simple_lab_test.search.common.runner import torch_load_checkpoint

    for key in ("design_contract", "normalized_duration_contract", "frozen_prior_data_contract", "prior_source_bindings"):
        spec = contract[key]
        require(sha256(root_file(spec["path"])) == spec["sha256"], f"Frozen {key} drift")
    aligned = duration.validate_aligned_contract(read_json(duration.DEFAULT_ALIGNED_CONTRACT))
    evidence = {}
    for dataset, references in contract["datasets"].items():
        binding = contract["data_bindings"][dataset]
        spec = DATASETS[dataset]
        for kind, expected in (("data", "data_sha256"), ("manifest", "split_manifest_sha256")):
            require(sha256(root_file(spec[kind])) == binding[expected], f"{dataset} {kind} drift")
        for role in ("B", "FULL"):
            source = references[role]["source"]
            for path_key, digest_key in (("checkpoint_path", "checkpoint_file_sha256"),
                                         ("summary_path", "summary_sha256"), ("history_path", "history_sha256")):
                require(sha256(root_file(source[path_key])) == source[digest_key], f"{dataset}/{role} {path_key} drift")
            duration.validate_source_artifacts(
                torch_load_checkpoint(root_file(source["checkpoint_path"]), map_location="cpu"),
                read_json(root_file(source["summary_path"])), read_json(root_file(source["history_path"])),
                role=role, source_spec=source, checkpoint_file_sha256=source["checkpoint_file_sha256"],
                aligned_dataset=aligned[dataset],
            )
        reference = references["aligned_B"]
        require(sha256(root_file(reference["summary_path"])) == reference["summary_sha256"]
                and sha256(root_file(reference["checkpoint_path"])) == reference["checkpoint_file_sha256"],
                f"{dataset} recovered aligned-B file drift")
        summary = read_json(root_file(reference["summary_path"]))
        require(summary.get("status") == "success" and summary.get("qualified_full_data") is True
                and summary.get("qualified_full_fit") is True
                and summary.get("quantity_prediction_bitwise_identical") is True,
                "Aligned-B is not a qualified quantity-preserving fit")
        require(summary["source_checkpoint_sha256"] == references["B"]["source"]["checkpoint_file_sha256"]
                and summary["best_validation_proper_time_nll"] == reference["normalized_time_nll"]
                and summary["observation_likelihood_contract"] == reference["observation_likelihood_contract"],
                "Aligned-B reference lineage/likelihood drift")
        evidence[dataset] = {"raw_sources_verified": ["B", "FULL"], "aligned_B_reused": True,
                             "data_sha256": binding["data_sha256"]}
    return evidence


def duration_manifest(*, dataset: str, role: str, source: dict[str, Any],
                      deployment: Mapping[str, Any]) -> dict[str, Any]:
    from paper.scripts import run_backbone_normalized_duration as duration
    dataset_spec = DATASETS[dataset]
    contract = read_json(CONTRACT_PATH)
    binding = contract["data_bindings"][dataset]
    return {
        "schema_version": 1, "manifest_id": duration.JOB_MANIFEST_ID,
        "status": "frozen_before_normalized_duration_outputs",
        "execution_contract_sha256": sha256(duration.DEFAULT_CONTRACT),
        "aligned_contract_sha256": sha256(duration.DEFAULT_ALIGNED_CONTRACT),
        "evaluation_source_revision": deployment["source_revision"],
        "evaluation_runner_file_sha256": sha256(DURATION_RUNNER),
        "calibration_source_revision": duration.ALIGNED_CALIBRATION_SOURCE_REVISION,
        "dataset": dataset, "model_role": role, "runtime": deployment["runtime"],
        "data": {"path": dataset_spec["data"], "sha256": binding["data_sha256"],
                 "split_manifest_path": dataset_spec["manifest"],
                 "split_manifest_sha256": binding["split_manifest_sha256"]},
        "source": source,
    }


def candidate_source(audit: Mapping[str, Any]) -> dict[str, Any]:
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import state_partition_sha256
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    summary_path = Path(audit["summary"])
    checkpoint = summary_path.parent / "best_val_qty_rmse_model.pt"
    history = summary_path.parent / "history.json"
    payload = torch_load_checkpoint(checkpoint, map_location="cpu")
    return {
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_file_sha256": sha256(checkpoint),
        "checkpoint_state_sha256": payload["model_state_sha256"],
        "source_non_time_state_sha256": state_partition_sha256(payload["model_state_dict"], time_head=False),
        "summary_path": str(summary_path.resolve()), "summary_sha256": sha256(summary_path),
        "history_path": str(history.resolve()), "history_sha256": sha256(history),
        "training_source_revision": payload["source_revision"],
        "training_source_revision_history": payload["source_revision_history"],
        "backbone": payload["backbone"], "model_role": "hard_lmm_bounded_qk_candidate",
        "routing_contract_id": "hard_lmm_bounded_qk_causal_v_v1", "quantity_variant": VARIANT,
        "seed": 42, "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
    }


def audit_duration(output: Path, *, dataset: str, role: str, full_fit: bool,
                   source: Mapping[str, Any]) -> dict[str, Any]:
    from paper.scripts import run_backbone_normalized_duration as duration
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
        build_candidate_from_selected_checkpoint, state_partition_sha256,
    )
    from simple_lab_test.search.common.runner import torch_load_checkpoint, canonical_state_dict_sha256
    summary = read_json(output / "summary.json")
    require(summary.get("status") == "success" and summary.get("qualified_full_data") is True
            and summary.get("qualified_full_fit") is full_fit, "Duration fit qualification failed")
    require(summary.get("model_role") == role and summary.get("dataset") == dataset
            and summary.get("held_out_test_evaluated") is False, "Duration scope drift")
    require(summary.get("quantity_prediction_bitwise_identical") is True
            and summary.get("trainable_parameter_count") == 130, "Frozen quantity/head contract failed")
    require(summary["train_cache"]["count"] == DATASETS[dataset]["train_targets"]
            and summary["validation_cache"]["count"] == DATASETS[dataset]["validation_targets"],
            "Duration cache count drift")
    require(summary["source_checkpoint_sha256"] == source["checkpoint_file_sha256"]
            and summary["selected_non_time_state_sha256"] == source["source_non_time_state_sha256"],
            "Duration source/quantity state drift")
    require(summary.get("runtime", {}).get("requested_device") == "cuda"
            and summary["runtime"]["peak_memory_allocated_bytes"] > 0, "Duration CUDA proof missing")
    if not full_fit:
        require(summary.get("completed_epochs") == 1, "Duration e1 did not complete one epoch")
    selected_path = output / duration.SELECTED_CHECKPOINT_NAME
    require(sha256(selected_path) == summary["selected_checkpoint_file_sha256"], "Duration selected file drift")
    selected = torch_load_checkpoint(selected_path, map_location="cpu")
    model = build_candidate_from_selected_checkpoint(selected)
    require(canonical_state_dict_sha256(model.state_dict()) == summary["selected_state_sha256"]
            and state_partition_sha256(model.state_dict(), time_head=False) == source["source_non_time_state_sha256"],
            "Duration independent strict restore failed")
    for key in ("best_validation_proper_time_nll", "time_median_mae", "time_median_rmse"):
        require(math.isfinite(float(summary[key])), f"Nonfinite duration metric: {key}")
    return {"status": "passed", "summary": str(output / "summary.json"),
            "summary_sha256": sha256(output / "summary.json"), "strict_selected_restore": True,
            "quantity_prediction_bitwise_identical": True, "qualified_full_fit": full_fit,
            "best_epoch": summary["best_epoch"], "completed_epochs": summary["completed_epochs"],
            "proper_time_nll": summary["best_validation_proper_time_nll"],
            "time_median_mae": summary["time_median_mae"], "time_median_rmse": summary["time_median_rmse"]}


def execute_phases(contract: Mapping[str, Any], *, joint: Callable, duration: Callable,
                   record_gate: Callable) -> dict[str, Any]:
    """Pure queue policy; an unsuccessful guardrail never launches another fit."""
    from paper.scripts.audit_hard_lmm_bounded_qk import evaluate_quantity_gate, evaluate_time_gate
    for dataset in contract["phases"]["e1"]["datasets_in_order"]:
        joint(dataset, "e1")
    for dataset in contract["phases"]["e1"]["datasets_in_order"]:
        duration(dataset, "FULL", False, None)
    decisions = {}
    for dataset in contract["phases"]["seed42_screening"]["datasets_in_order"]:
        audit = joint(dataset, "seed42_screening")
        refs = contract["datasets"][dataset]
        quantity = evaluate_quantity_gate(audit["metrics"], refs["B"]["metrics"], refs["FULL"]["metrics"])
        decisions[dataset] = {"quantity": quantity}
        record_gate(dataset, decisions[dataset])
        if quantity["status"] != "passed":
            return {"status": "stopped_quantity_gate_failed", "failed_dataset": dataset, "decisions": decisions}
        full = duration(dataset, "FULL", True, None)
        duration(dataset, "BOUNDED", False, audit)
        bounded = duration(dataset, "BOUNDED", True, audit)
        time_gate = evaluate_time_gate(bounded["proper_time_nll"], refs["aligned_B"]["normalized_time_nll"],
                                      full["proper_time_nll"], dataset)
        decisions[dataset]["normalized_time"] = time_gate
        record_gate(dataset, decisions[dataset])
        if time_gate["status"] != "passed":
            return {"status": "stopped_normalized_time_gate_failed", "failed_dataset": dataset, "decisions": decisions}
    return {"status": "seed42_all_datasets_passed", "decisions": decisions,
            "additional_seeds_launched": False, "final_adoption": False}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--expected-deployment-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    require(sha256(args.deployment_manifest) == args.expected_deployment_manifest_sha256,
            "Preregistered deployment digest drift")
    deployment = read_json(args.deployment_manifest)
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)
    source = verify_source(deployment)
    output_root = args.output_root.resolve()
    require(str(output_root) == deployment.get("output_root"), "Output root drift")
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "campaign_status.json"
    require(not status_path.exists(), "Campaign exists; automatic restart is forbidden")
    status = {"schema_version": 1, "contract_id": contract["contract_id"],
              "source_revision": deployment["source_revision"], "host_role": "5090",
              "deployment_manifest_sha256": args.expected_deployment_manifest_sha256,
              "source_verification": source, "status": "preflight", "started_at": utc_now(),
              "held_out_test_evaluated": False, "jobs": {}, "decisions": {}}

    def persist() -> None:
        status["updated_at"] = utc_now()
        save_json(status_path, status)

    def run_command(job_id: str, command: list[str], output: Path) -> None:
        require(sha256(args.deployment_manifest) == args.expected_deployment_manifest_sha256,
                "Deployment manifest changed after outputs")
        verify_source(deployment)
        preflight = gpu_preflight("RTX 5090")
        output.mkdir(parents=True, exist_ok=False)
        status.update(status=f"running_{job_id}", current_job=job_id)
        status["jobs"][job_id] = {"status": "running", "started_at": utc_now(),
                                  "output": str(output), "command": command, "gpu_preflight": preflight}
        save_json(output / "execution_receipt.json", status["jobs"][job_id])
        persist()
        with (output / "runner.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def complete(job_id: str, audit: Mapping[str, Any]) -> None:
        require(audit.get("status") == "passed", f"Job audit failed: {job_id}")
        status["jobs"][job_id].update(audit, completed_at=utc_now())
        save_json(Path(status["jobs"][job_id]["output"]) / "audit.json", dict(audit))
        persist()

    def joint(dataset: str, phase_name: str) -> dict[str, Any]:
        from paper.scripts.audit_hard_lmm_bounded_qk import audit_bounded_qk_job
        phase = contract["phases"][phase_name]
        job_id = f"{phase_name}_{dataset}"
        output = output_root / "jobs" / job_id
        command = job_command(python=args.python, candidate=contract["candidate"], dataset=dataset,
                              output=output, source_revision=deployment["source_revision"], phase=phase, host_role="5090")
        run_command(job_id, command, output)
        audit = audit_bounded_qk_job(output, candidate=contract["candidate"], dataset=dataset,
                                    source_revision=deployment["source_revision"], expected_epochs=phase["epochs"],
                                    binding=contract["data_bindings"][dataset])
        complete(job_id, audit)
        return audit

    def duration(dataset: str, role: str, full_fit: bool, joint_audit: Mapping[str, Any] | None) -> dict[str, Any]:
        source_spec = (contract["datasets"][dataset]["FULL"]["source"] if role == "FULL"
                       else candidate_source(joint_audit))
        manifest = duration_manifest(dataset=dataset, role=role, source=source_spec, deployment=deployment)
        manifest_dir = output_root / "duration_manifests"
        manifest_dir.mkdir(exist_ok=True)
        manifest_path = manifest_dir / f"{role}_{dataset}.json"
        if manifest_path.exists():
            require(read_json(manifest_path) == manifest, "Existing duration manifest drift")
        else:
            save_json(manifest_path, manifest)
        stage = "full" if full_fit else "e1"
        job_id = f"normalized_{stage}_{role}_{dataset}"
        output = output_root / "jobs" / job_id
        command = [args.python, "-s", str(DURATION_RUNNER), "--job-manifest", str(manifest_path),
                   "--expected-job-manifest-sha256", sha256(manifest_path), "--output-dir", str(output),
                   "--device", "cuda", "--source-revision", deployment["source_revision"]]
        if full_fit:
            cache = output_root / "jobs" / f"normalized_e1_{role}_{dataset}" / "cache"
            command += ["--feature-cache-dir", str(cache)]
        else:
            command += ["--allow-partial-contract", "--max-epochs", "1"]
        run_command(job_id, command, output)
        audit = audit_duration(output, dataset=dataset, role=role, full_fit=full_fit, source=source_spec)
        complete(job_id, audit)
        return audit

    def record_gate(dataset: str, decision: dict[str, Any]) -> None:
        status["decisions"][dataset] = decision
        persist()

    persist()
    try:
        status["gpu_preflight"] = gpu_preflight("RTX 5090")
        runtime = subprocess.check_output([
            args.python, "-s", "-c",
            "import json,torch,sys,importlib.metadata; "
            "from paper.scripts.run_backbone_normalized_duration import runtime_identity; "
            "print(json.dumps({'runtime':runtime_identity(torch.device('cuda')), "
            "'executable':sys.executable, 'dependencies':{name:importlib.metadata.version(name) "
            "for name in ['numpy','polars','pyarrow','pytest']}}))",
        ], cwd=ROOT, text=True)
        observed_runtime = json.loads(runtime)
        require(observed_runtime["runtime"] == deployment["runtime"]
                and observed_runtime["dependencies"] == deployment["dependencies"]
                and observed_runtime["executable"] == deployment["python_executable"],
                "Runtime differs from preregistration")
        gpu_uuid = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
        ).strip()
        require(gpu_uuid == deployment["gpu_uuid"], "Physical GPU identity drift")
        status["input_verification"] = verify_inputs(contract)
        persist()
        qualification = output_root / "cuda_qualification"
        tests = ["test_count_aware_titan_bounded_qk.py", "test_count_aware_titan_causal_qkv.py",
                 "test_hard_lmm_bounded_qk_profile.py", "test_backbone_normalized_duration.py",
                 "test_backbone_normalized_duration_integration.py"]
        junit = qualification / "results.xml"
        command = [args.python, "-s", "-m", "pytest", "-q"] + [str(ROOT / "simple_lab_test/search/tests" / name) for name in tests]
        command += [f"--junitxml={junit}"]
        run_command("cuda_qualification", command, qualification)
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        require(cases and all(case.find("failure") is None and case.find("error") is None
                              and case.find("skipped") is None for case in cases), "CUDA suite incomplete")
        required = {
            "test_zero_initialization_preserves_b_full_outputs_common_and_kernel_gradients[64-cuda]",
            "test_operating_extremes_finite_and_optimizer_roundtrip_next_step[cuda]",
        }
        require(required <= {case.attrib["name"] for case in cases}, "Actual BOUNDED CUDA tests missing")
        complete("cuda_qualification", {"status": "passed", "test_count": len(cases),
                                        "required_actual_cuda_tests": sorted(required), "junit_sha256": sha256(junit)})
        profile_output = output_root / "cost_profile"
        run_command("cost_profile", [args.python, "-s", str(PROFILER), "--output", str(profile_output / "profile.json")], profile_output)
        profile = read_json(profile_output / "profile.json")
        complete("cost_profile", {"status": profile["status"], "cost_gate": profile["cost_gate"],
                                  "profile_sha256": sha256(profile_output / "profile.json")})
        decision = execute_phases(contract, joint=joint, duration=duration, record_gate=record_gate)
        status.update(decision, current_job=None, completed_at=utc_now())
        save_json(output_root / "final_decision.json", {**decision, "source_revision": deployment["source_revision"],
                  "contract_sha256": sha256(CONTRACT_PATH), "held_out_test_evaluated": False,
                  "recorded_at": utc_now()})
        persist()
    except BaseException as error:
        status.update(status="failed_execution", error=repr(error), completed_at=utc_now())
        persist()
        raise


if __name__ == "__main__":
    main()
