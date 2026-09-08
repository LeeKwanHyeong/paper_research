#!/usr/bin/env python3
"""Run the frozen LPHC-QKV quantity/mechanism campaign on an RTX 5090.

LPHC and B share the same fresh, seeded initialization contract.  Existing B
and FULL checkpoints are performance references only.  This runner stops after
seed-42 quantity/mechanism screening; it cannot launch normalized-duration fits
or additional seeds.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    DATASETS,
    gpu_preflight,
    job_command as base_job_command,
    read_json,
    require,
    save_json,
    sha256,
)

CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_level_history_qkv_screening_v1.json"
CONTRACT_SHA256 = "a4ba58bb69f40496de20869643843798a13af17ca169ee4587815c55b8d1061d"
PROFILER = ROOT / "paper/scripts/profile_hard_lmm_level_history_qkv.py"
MECHANISM_AUDITOR = ROOT / "paper/scripts/audit_hard_lmm_level_history_qkv_mechanism.py"
VARIANT = "count_only_log_regression"
PENDING_STATUS = "pending_normalized_time_and_additional_seed_contract"
CUDA_TESTS = (
    "test_count_aware_titan_level_history_qkv.py",
    "test_level_history_qkv_training_route.py",
    "test_hard_lmm_level_history_qkv_audit.py",
    "test_hard_lmm_level_history_qkv_profile.py",
)
REQUIRED_ACTUAL_CUDA_TESTS = frozenset({
    "test_zero_initialization_preserves_b_outputs_gradients_and_rng[64-cuda]",
    "test_extremes_finite_and_optimizer_roundtrip_next_step[cuda]",
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def root_file(path: str) -> Path:
    logical = Path(path)
    require(
        not logical.is_absolute() and ".." not in logical.parts
        and str(logical) not in {"", "."},
        f"Expected a safe root-relative path: {path}",
    )
    result = ROOT / logical
    require(result.is_file(), f"Missing frozen input: {path}")
    return result


def verify_source(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the complete checkout and reject unregistered imported Python."""
    require(manifest.get("schema_version") == 1, "Deployment schema drift")
    require(manifest.get("host_role") == "5090", "Deployment is not for 5090")
    require(manifest.get("status") == "frozen_before_gpu_outputs", "Deployment status drift")
    require(manifest.get("source_root") == str(ROOT.resolve()), "Source root drift")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    require(revision == manifest.get("source_revision"), "Checkout revision drift")
    files = manifest.get("source_files")
    require(isinstance(files, dict) and bool(files), "Source file map is empty")
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    require(set(files) == {path for path in tracked if path}, "Tracked source population drift")
    for path, expected in files.items():
        require(sha256(root_file(path)) == expected, f"Source file drift: {path}")
    for folder in ("models", "paper/scripts", "simple_lab_test"):
        extras = {
            str(path.relative_to(ROOT)) for path in (ROOT / folder).rglob("*.py")
        } - set(files)
        require(not extras, f"Unregistered Python source: {sorted(extras)}")
    require(sha256(CONTRACT_PATH) == CONTRACT_SHA256, "Local campaign contract digest drift")
    require(
        manifest.get("execution_contract_sha256") == CONTRACT_SHA256,
        "Deployment campaign contract digest drift",
    )
    return {"source_revision": revision, "verified_files": len(files)}


def validate_contract(contract: Mapping[str, Any]) -> None:
    require(sha256(CONTRACT_PATH) == CONTRACT_SHA256, "LPHC screening contract digest drift")
    require(
        contract.get("schema_version") == 1
        and contract.get("contract_id") == "hard_lmm_level_history_qkv_screening_v1"
        and contract.get("status") == "frozen_before_level_history_qkv_gpu_outputs",
        "Contract identity/status drift",
    )
    require(contract.get("host_role") == "5090", "Host contract drift")
    require(
        contract.get("evaluation_scope") == "validation_only"
        and contract.get("held_out_test_evaluated") is False
        and contract.get("additional_seeds_authorized") is True
        and contract.get("automatic_additional_seed_launch") is False,
        "Evaluation/additional-seed scope drift",
    )
    require(contract.get("candidate") == {
        "backbone": "titantpp_hard_memory_level_history_qkv",
        "model_role": "hard_lmm_level_history_qkv_candidate",
        "routing_contract_id": "hard_lmm_level_preserving_history_confidence_residual_v1",
        "gpu_name_contains": "RTX 5090",
    }, "Candidate route drift")
    require(contract.get("initialization_policy") == {
        "method": "same_seed_fresh_B_identity_v1",
        "pretrained_B_checkpoint_injected": False,
        "equal_training_budget": True,
        "fresh_B_backbone": "titantpp",
        "candidate_new_parameter_count_hidden64": 384,
        "new_parameters_exact_zero": True,
        "required_proof": [
            "fresh_B_and_candidate_common_state_bitwise_equal",
            "fresh_B_and_candidate_outputs_bitwise_equal",
            "post_construction_RNG_state_bitwise_equal",
            "candidate_initial_state_sha256_replays_summary_and_checkpoints",
        ],
    }, "Fresh initialization policy drift")
    require(contract.get("phases", {}).get("e1") == {
        "epochs": 1,
        "minimum_epochs": 1,
        "patience": 40,
        "datasets_in_order": [
            "yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket"
        ],
        "performance_gate": False,
    }, "e1 contract drift")
    require(contract.get("phases", {}).get("seed42_screening") == {
        "seed": 42,
        "epochs": 300,
        "minimum_epochs": 40,
        "patience": 40,
        "datasets_in_order": [
            "insta_market_basket", "yellow_trip_hourly", "intermittent_frozen_5000"
        ],
        "instacart_mechanism_gate_before_other_datasets": True,
        "stop_on_first_quantity_mechanism_or_normalized_time_guardrail_failure": True,
    }, "Seed42 screening contract drift")
    require(contract.get("phases", {}).get("additional_seeds") == {
        "seeds": [52, 62],
        "epochs": 300,
        "minimum_epochs": 40,
        "patience": 40,
        "launch_condition": "all_three_seed42_quantity_mechanism_and_normalized_time_gates_pass",
        "requires_frozen_matched_seed_reference_contract_before_launch": True,
    }, "Additional-seed contract drift")
    require(contract.get("quantity_gate") == {
        "B": {
            "raw_rmse_ratio_strictly_less_than": 1.0,
            "overall_mae_ratio_max": 1.01,
            "body_mae_ratio_max": 1.02,
            "gt_p99_mae_ratio_max": 1.02,
        },
        "FULL": {
            "raw_rmse_ratio_max": 1.01,
            "overall_mae_ratio_max": 1.01,
            "body_mae_ratio_max": 1.02,
            "gt_p99_mae_ratio_max": 1.02,
        },
    }, "Quantity gate drift")
    mechanism = contract.get("mechanism_reference")
    require(
        isinstance(mechanism, Mapping)
        and mechanism.get("dataset") == "insta_market_basket"
        and mechanism.get("baseline_role") == "B",
        "Mechanism reference drift",
    )
    recovery = contract.get("recovery_policy")
    require(
        isinstance(recovery, Mapping)
        and recovery.get("automatic_retry_after_execution_or_screening_failure") is False
        and recovery.get("runtime_or_hyperparameter_changes_after_outputs") is False,
        "Fail-closed recovery policy drift",
    )


def _verify_reference_source(
    dataset: str, role: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify a performance reference without using it as initialization."""
    from simple_lab_test.search.common.runner import (
        canonical_state_dict_sha256, torch_load_checkpoint,
    )

    for path_key, digest_key in (
        ("checkpoint_path", "checkpoint_file_sha256"),
        ("summary_path", "summary_sha256"),
        ("history_path", "history_sha256"),
    ):
        require(
            sha256(root_file(str(source[path_key]))) == source[digest_key],
            f"{dataset}/{role} {path_key} drift",
        )
    checkpoint = torch_load_checkpoint(root_file(source["checkpoint_path"]), map_location="cpu")
    require(isinstance(checkpoint, dict), f"{dataset}/{role} checkpoint is invalid")
    state = checkpoint.get("model_state_dict")
    require(isinstance(state, dict) and bool(state), f"{dataset}/{role} state missing")
    state_digest = canonical_state_dict_sha256(state)
    require(
        state_digest == checkpoint.get("model_state_sha256")
        == source.get("checkpoint_state_sha256"),
        f"{dataset}/{role} checkpoint state drift",
    )
    require(
        checkpoint.get("backbone") == source.get("backbone")
        and checkpoint.get("variant") == VARIANT
        and checkpoint.get("seed") == 42
        and checkpoint.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
        and checkpoint.get("checkpoint_monitor_history_key") == "val_qty_rmse"
        and checkpoint.get("checkpoint_selection") == "best_validation_raw_quantity_rmse"
        and checkpoint.get("selection") == "best_validation_raw_quantity_rmse"
        and checkpoint.get("source_revision") == source.get("training_source_revision")
        and checkpoint.get("evaluation_scope") == "validation_only"
        and checkpoint.get("held_out_test_evaluated") is False,
        f"{dataset}/{role} checkpoint route/selector/scope drift",
    )
    summary = read_json(root_file(source["summary_path"]))
    history = read_json(root_file(source["history_path"])).get("history")
    require(
        summary.get("status") == "success"
        and summary.get("backbone") == source.get("backbone")
        and summary.get("checkpoint_state_sha256") == state_digest
        and summary.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
        and summary.get("evaluation_scope") == "validation_only"
        and summary.get("held_out_test_evaluated") is False,
        f"{dataset}/{role} summary drift",
    )
    require(isinstance(history, list) and bool(history), f"{dataset}/{role} history missing")
    return {
        "checkpoint_file_sha256": source["checkpoint_file_sha256"],
        "checkpoint_state_sha256": state_digest,
        "summary_sha256": source["summary_sha256"],
        "history_sha256": source["history_sha256"],
        "use": "performance_reference_only_not_initialization",
    }


def verify_inputs(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Verify all fixed contracts, data bytes, references, and paired B cache."""
    import polars as pl

    for key in (
        "design_contract", "frozen_prior_screening_contract",
        "normalized_duration_contract", "frozen_prior_data_contract",
        "prior_source_bindings",
    ):
        spec = contract[key]
        require(sha256(root_file(spec["path"])) == spec["sha256"], f"Frozen {key} drift")
    evidence: dict[str, Any] = {}
    for dataset, references in contract["datasets"].items():
        binding = contract["data_bindings"][dataset]
        spec = DATASETS[dataset]
        require(sha256(root_file(spec["data"])) == binding["data_sha256"], f"{dataset} data drift")
        require(
            sha256(root_file(spec["manifest"])) == binding["split_manifest_sha256"],
            f"{dataset} split manifest drift",
        )
        sources = {
            role: _verify_reference_source(dataset, role, references[role]["source"])
            for role in ("B", "FULL")
        }
        aligned = references["aligned_B"]
        require(
            sha256(root_file(aligned["summary_path"])) == aligned["summary_sha256"]
            and sha256(root_file(aligned["checkpoint_path"]))
            == aligned["checkpoint_file_sha256"],
            f"{dataset} aligned-B artifact drift",
        )
        aligned_summary = read_json(root_file(aligned["summary_path"]))
        require(
            aligned_summary.get("status") == "success"
            and aligned_summary.get("qualified_full_data") is True
            and aligned_summary.get("qualified_full_fit") is True
            and aligned_summary.get("quantity_prediction_bitwise_identical") is True
            and aligned_summary.get("source_checkpoint_sha256")
            == references["B"]["source"]["checkpoint_file_sha256"],
            f"{dataset} aligned-B qualification/lineage drift",
        )
        evidence[dataset] = {
            "data_sha256": binding["data_sha256"],
            "split_manifest_sha256": binding["split_manifest_sha256"],
            "performance_references": sources,
            "training_initialization": "fresh_same_seed_not_reference_checkpoint",
            "aligned_B_verified_for_later_stage": True,
        }

    mechanism = contract["mechanism_reference"]["paired_validation_predictions"]
    cache = root_file(mechanism["path"])
    require(sha256(cache) == mechanism["sha256"], "Pinned B cache digest drift")
    lazy = pl.scan_parquet(cache)
    require(
        set(mechanism["required_columns"]).issubset(set(lazy.collect_schema().names())),
        "Pinned B cache required-column drift",
    )
    row_count = int(lazy.select(pl.len()).collect().item())
    require(row_count == mechanism["row_count"], "Pinned B cache row-count drift")
    evidence["mechanism_reference"] = {
        "path": mechanism["path"], "sha256": mechanism["sha256"],
        "row_count": row_count, "required_columns": list(mechanism["required_columns"]),
    }
    return evidence


def job_command(**kwargs: Any) -> list[str]:
    """Build a fresh LPHC fit and reject accidental checkpoint warm-start flags."""
    command = base_job_command(**kwargs)
    require(
        not any(token.startswith("--initial-model") for token in command),
        "LPHC campaign must not initialize from a selected B checkpoint",
    )
    return command


def validate_fresh_initial_identity(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Require the auditor's independent fresh B/LPHC identity reconstruction."""
    require(isinstance(evidence, Mapping), "Fresh initialization proof missing")
    required_fields = {
        "method",
        "seed",
        "B_backbone",
        "candidate_backbone",
        "fresh_B_state_sha256",
        "candidate_common_state_sha256",
        "candidate_initial_state_sha256",
        "new_parameter_state_keys",
        "new_parameters_exact_zero",
        "additional_parameter_count",
        "inherited_state_bitwise_equal",
        "synthetic_outputs_bitwise_equal",
        "post_construction_RNG_state_bitwise_equal",
        "summary_and_checkpoint_initial_state_sha256_equal",
    }
    require(set(evidence) == required_fields, "Fresh initialization proof schema drift")
    require(
        evidence.get("method") == "same_seed_fresh_B_identity_v1",
        "Fresh initialization method drift",
    )
    require(
        evidence.get("seed") == 42
        and evidence.get("B_backbone") == "titantpp"
        and evidence.get("candidate_backbone")
        == "titantpp_hard_memory_level_history_qkv"
        and evidence.get("new_parameter_state_keys") == [
            "encoder.layers.0.attn.level_history_q_kernel",
            "encoder.layers.0.attn.level_history_k_kernel",
            "encoder.layers.0.attn.level_history_v_kernel",
        ]
        and evidence.get("additional_parameter_count") == 384,
        "Fresh LPHC route or parameter contract drift",
    )
    for name in (
        "fresh_B_state_sha256",
        "candidate_common_state_sha256",
        "candidate_initial_state_sha256",
    ):
        value = evidence.get(name)
        require(
            isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"Fresh identity digest is invalid: {name}",
        )
    require(
        evidence.get("fresh_B_state_sha256")
        == evidence.get("candidate_common_state_sha256"),
        "Fresh B and LPHC common-state digests differ",
    )
    proof_names = (
        "new_parameters_exact_zero",
        "inherited_state_bitwise_equal",
        "synthetic_outputs_bitwise_equal",
        "post_construction_RNG_state_bitwise_equal",
        "summary_and_checkpoint_initial_state_sha256_equal",
    )
    require(
        all(evidence.get(name) is True for name in proof_names),
        "Fresh B/LPHC identity proof failed",
    )
    return dict(evidence)


def mechanism_command(
    *, python: str, deployment_manifest: Path, deployment_sha256: str,
    contract: Mapping[str, Any], candidate_audit: Mapping[str, Any], output: Path,
) -> list[str]:
    mechanism = contract["mechanism_reference"]
    dataset = mechanism["dataset"]
    require(dataset == "insta_market_basket", "Mechanism dataset drift")
    summary_path = Path(candidate_audit["summary"])
    checkpoint_path = summary_path.parent / "best_val_qty_rmse_model.pt"
    candidate_job_audit = Path(candidate_audit["campaign_audit_path"])
    require(summary_path.is_file(), "Candidate mechanism summary missing")
    require(checkpoint_path.is_file(), "Candidate mechanism checkpoint missing")
    require(candidate_job_audit.is_file(), "Candidate job audit for mechanism binding missing")
    spec = DATASETS[dataset]
    return [
        python, "-s", str(MECHANISM_AUDITOR),
        "--deployment-manifest", str(deployment_manifest),
        "--expected-deployment-manifest-sha256", deployment_sha256,
        "--data", str(root_file(spec["data"])),
        "--split-manifest", str(root_file(spec["manifest"])),
        "--b-cache", str(root_file(mechanism["paired_validation_predictions"]["path"])),
        "--candidate-checkpoint", str(checkpoint_path),
        "--expected-candidate-checkpoint-sha256", sha256(checkpoint_path),
        "--candidate-summary", str(summary_path),
        "--expected-candidate-summary-sha256", sha256(summary_path),
        "--candidate-job-audit", str(candidate_job_audit),
        "--expected-candidate-job-audit-sha256", sha256(candidate_job_audit),
        "--output-dir", str(output), "--device", "cuda", "--batch-size", "512",
    ]


def execute_phases(
    contract: Mapping[str, Any], *,
    joint: Callable[[str, str], Mapping[str, Any]],
    mechanism: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    record_gate: Callable[[str, Mapping[str, Any]], None],
) -> dict[str, Any]:
    """Execute fixed gates without duration fitting or unmatched extra seeds."""
    from paper.scripts.audit_hard_lmm_level_history_qkv import evaluate_quantity_gate

    for dataset in contract["phases"]["e1"]["datasets_in_order"]:
        joint(dataset, "e1")
    order = contract["phases"]["seed42_screening"]["datasets_in_order"]
    instacart = order[0]
    require(instacart == "insta_market_basket", "Instacart must screen first")
    decisions: dict[str, dict[str, Any]] = {}
    audit = joint(instacart, "seed42_screening")
    refs = contract["datasets"][instacart]
    quantity = evaluate_quantity_gate(audit["metrics"], refs["B"]["metrics"], refs["FULL"]["metrics"])
    decisions[instacart] = {"quantity": quantity}
    record_gate(instacart, decisions[instacart])

    # This is inference from the completed checkpoint. Run it even if the
    # aggregate gate failed, so the short-history mechanism question is closed.
    mechanism_gate = dict(mechanism(audit))
    require(mechanism_gate.get("status") in {"passed", "failed"}, "Mechanism gate status invalid")
    decisions[instacart]["mechanism"] = mechanism_gate
    record_gate(instacart, decisions[instacart])
    failed = [
        name for name, gate in (("quantity", quantity), ("mechanism", mechanism_gate))
        if gate.get("status") != "passed"
    ]
    if failed:
        return _stopped(
            "stopped_instacart_quantity_or_mechanism_gate_failed",
            instacart, failed, decisions,
        )

    for dataset in order[1:]:
        audit = joint(dataset, "seed42_screening")
        refs = contract["datasets"][dataset]
        quantity = evaluate_quantity_gate(audit["metrics"], refs["B"]["metrics"], refs["FULL"]["metrics"])
        decisions[dataset] = {"quantity": quantity}
        record_gate(dataset, decisions[dataset])
        if quantity.get("status") != "passed":
            return _stopped("stopped_quantity_gate_failed", dataset, ["quantity"], decisions)
    return {
        "status": PENDING_STATUS,
        "decisions": decisions,
        "normalized_time_launched": False,
        "additional_seeds_launched": False,
        "final_adoption": False,
        "pending_requirements": [
            "run_matched_normalized_time_gates_for_all_three_seed42_checkpoints",
            "freeze_matched_seed52_62_reference_contract_before_any_additional_seed",
        ],
    }


def _stopped(
    status: str, dataset: str, failed: list[str], decisions: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "status": status, "failed_dataset": dataset, "failed_gates": failed,
        "decisions": dict(decisions), "normalized_time_launched": False,
        "additional_seeds_launched": False, "final_adoption": False,
    }


def record_execution_failure(status: dict[str, Any], error: BaseException) -> dict[str, Any]:
    """Mark the active job and campaign failed before propagating the error."""
    now = utc_now()
    current = status.get("current_job")
    jobs = status.setdefault("jobs", {})
    if current in jobs:
        jobs[current].update(status="failed", error=repr(error), completed_at=now)
    status.update(status="failed_execution", error=repr(error), completed_at=now)
    return status


def _verify_runtime(python: str, deployment: Mapping[str, Any]) -> dict[str, Any]:
    output = subprocess.check_output([
        python, "-s", "-c",
        "import json,torch,sys,importlib.metadata; "
        "from paper.scripts.run_backbone_normalized_duration import runtime_identity; "
        "print(json.dumps({'runtime':runtime_identity(torch.device('cuda')), "
        "'executable':sys.executable, 'dependencies':{name:importlib.metadata.version(name) "
        "for name in ['numpy','polars','pyarrow','pytest']}}))",
    ], cwd=ROOT, text=True)
    observed = json.loads(output)
    require(
        observed.get("runtime") == deployment.get("runtime")
        and observed.get("dependencies") == deployment.get("dependencies")
        and observed.get("executable") == deployment.get("python_executable"),
        "Runtime differs from preregistration",
    )
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()
    require(gpu_uuid == deployment.get("gpu_uuid"), "Physical GPU identity drift")
    return observed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--expected-deployment-manifest-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    require(
        sha256(args.deployment_manifest) == args.expected_deployment_manifest_sha256,
        "Preregistered deployment digest drift",
    )
    deployment = read_json(args.deployment_manifest)
    contract = read_json(CONTRACT_PATH)
    validate_contract(contract)
    source = verify_source(deployment)
    output_root = args.output_root.resolve()
    require(str(output_root) == deployment.get("output_root"), "Output root drift")
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "campaign_status.json"
    require(not status_path.exists(), "Campaign exists; automatic restart is forbidden")
    status: dict[str, Any] = {
        "schema_version": 1, "contract_id": contract["contract_id"],
        "contract_sha256": CONTRACT_SHA256, "source_revision": deployment["source_revision"],
        "host_role": "5090", "deployment_manifest_sha256": args.expected_deployment_manifest_sha256,
        "source_verification": source, "status": "preflight", "started_at": utc_now(),
        "held_out_test_evaluated": False, "normalized_time_launched": False,
        "additional_seeds_launched": False, "jobs": {}, "decisions": {},
    }

    def persist() -> None:
        status["updated_at"] = utc_now()
        save_json(status_path, status)

    def run_command(job_id: str, command: list[str], output: Path) -> None:
        require(
            sha256(args.deployment_manifest) == args.expected_deployment_manifest_sha256,
            "Deployment manifest changed after outputs",
        )
        verify_source(deployment)
        preflight = gpu_preflight("RTX 5090")
        output.mkdir(parents=True, exist_ok=False)
        status.update(status=f"running_{job_id}", current_job=job_id)
        status["jobs"][job_id] = {
            "status": "running", "started_at": utc_now(), "output": str(output),
            "command": command, "gpu_preflight": preflight,
        }
        save_json(output / "execution_receipt.json", status["jobs"][job_id])
        persist()
        with (output / "runner.log").open("x") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)

    def complete(job_id: str, audit: Mapping[str, Any]) -> None:
        require(audit.get("status") == "passed", f"Job audit failed: {job_id}")
        status["jobs"][job_id].update(dict(audit), completed_at=utc_now())
        save_json(Path(status["jobs"][job_id]["output"]) / "campaign_audit.json", dict(audit))
        status["current_job"] = None
        persist()

    def joint(dataset: str, phase_name: str) -> dict[str, Any]:
        from paper.scripts.audit_hard_lmm_level_history_qkv import audit_level_history_qkv_job

        phase = contract["phases"][phase_name]
        job_id = f"{phase_name}_{dataset}"
        output = output_root / "jobs" / job_id
        command = job_command(
            python=args.python, candidate=contract["candidate"], dataset=dataset,
            output=output, source_revision=deployment["source_revision"],
            phase=phase, host_role="5090",
        )
        run_command(job_id, command, output)
        audit = audit_level_history_qkv_job(
            output, candidate=contract["candidate"], dataset=dataset,
            source_revision=deployment["source_revision"], expected_epochs=phase["epochs"],
            binding=contract["data_bindings"][dataset],
        )
        level_audit = audit.get("level_history_qkv_audit")
        require(isinstance(level_audit, Mapping), "LPHC audit evidence missing")
        validate_fresh_initial_identity(level_audit.get("fresh_initial_identity"))
        audit["campaign_audit_path"] = str(output / "campaign_audit.json")
        complete(job_id, audit)
        return audit

    def paired_mechanism(candidate_audit: Mapping[str, Any]) -> dict[str, Any]:
        job_id = "seed42_insta_market_basket_mechanism"
        output = output_root / "jobs" / job_id
        command = mechanism_command(
            python=args.python, deployment_manifest=args.deployment_manifest,
            deployment_sha256=args.expected_deployment_manifest_sha256,
            contract=contract, candidate_audit=candidate_audit, output=output,
        )
        run_command(job_id, command, output)
        native_audit_path = output / "audit.json"
        native_decision_path = output / "decision.json"
        native_audit = read_json(native_audit_path)
        mechanism_scope = native_audit.get("scope")
        require(
            native_audit.get("status") == "passed"
            and isinstance(mechanism_scope, Mapping)
            and mechanism_scope.get("dataset") == "insta_market_basket"
            and mechanism_scope.get("seed") == 42
            and mechanism_scope.get("target_split") == "validation"
            and mechanism_scope.get("held_out_test_evaluated") is False
            and mechanism_scope.get("training_performed") is False,
            "Paired mechanism execution audit failed",
        )
        gate = native_audit.get("gate")
        require(isinstance(gate, Mapping) and gate.get("status") in {"passed", "failed"}, "Paired mechanism gate missing")
        complete(job_id, {
            "status": "passed", "native_audit": str(native_audit_path),
            "native_audit_sha256": sha256(native_audit_path),
            "native_decision": str(native_decision_path),
            "native_decision_sha256": sha256(native_decision_path),
            "gate": dict(gate), "held_out_test_evaluated": False,
        })
        return dict(gate)

    def record_gate(dataset: str, decision: Mapping[str, Any]) -> None:
        status["decisions"][dataset] = dict(decision)
        persist()

    persist()
    try:
        status["gpu_preflight"] = gpu_preflight("RTX 5090")
        status["runtime_verification"] = _verify_runtime(args.python, deployment)
        status["input_verification"] = verify_inputs(contract)
        persist()

        qualification = output_root / "cuda_qualification"
        junit = qualification / "results.xml"
        command = [args.python, "-s", "-m", "pytest", "-q"] + [
            str(ROOT / "simple_lab_test/search/tests" / name) for name in CUDA_TESTS
        ] + [f"--junitxml={junit}"]
        run_command("cuda_qualification", command, qualification)
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        require(
            bool(cases) and all(
                case.find("failure") is None and case.find("error") is None
                and case.find("skipped") is None for case in cases
            ),
            "CUDA qualification suite is incomplete",
        )
        names = {case.attrib["name"] for case in cases}
        require(REQUIRED_ACTUAL_CUDA_TESTS <= names, "Actual LPHC CUDA tests missing")
        complete("cuda_qualification", {
            "status": "passed", "test_count": len(cases),
            "required_actual_cuda_tests": sorted(REQUIRED_ACTUAL_CUDA_TESTS),
            "junit_sha256": sha256(junit),
        })

        profile_output = output_root / "cost_profile"
        profile_path = profile_output / "profile.json"
        run_command(
            "cost_profile", [args.python, "-s", str(PROFILER), "--output", str(profile_path)],
            profile_output,
        )
        profile = read_json(profile_path)
        require(profile.get("status") == "passed", "LPHC cost profile failed")
        complete("cost_profile", {
            "status": "passed", "cost_gate": profile["cost_gate"],
            "profile": str(profile_path), "profile_sha256": sha256(profile_path),
        })

        decision = execute_phases(
            contract, joint=joint, mechanism=paired_mechanism, record_gate=record_gate,
        )
        status.update(decision, current_job=None, completed_at=utc_now())
        save_json(output_root / "final_decision.json", {
            **decision, "source_revision": deployment["source_revision"],
            "contract_sha256": CONTRACT_SHA256, "held_out_test_evaluated": False,
            "recorded_at": utc_now(),
        })
        persist()
    except BaseException as error:
        record_execution_failure(status, error)
        persist()
        raise


if __name__ == "__main__":
    main()
