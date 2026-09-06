#!/usr/bin/env python3
"""Run and audit one selector-aligned RMTPP/THP validation job.

The launcher is intentionally limited to the current next job derived from a
fresh audit of the fixed two-host artifact registry.  An interrupted job is
resumed only through the trainer's exact-identity epoch checkpoint.  Per-job
and host-wide advisory locks prevent duplicate or overlapping GPU work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTRACT_REL = Path("paper/contracts/raw_rmse_baseline_alignment_seed42_v1.json")
TRAINER_REL = Path("paper/scripts/run_count_aware_tpp_backbone_control.py")
LAUNCHER_REL = Path("paper/scripts/run_raw_rmse_baseline_alignment_job.py")
CONTROLLER_REL = Path("paper/scripts/control_raw_rmse_baseline_alignment.py")
PACKAGE_REL = Path("paper/scripts/package_raw_rmse_baseline_alignment.py")
B_REFERENCE_EVIDENCE_REL = Path(
    "paper/results/hard_lmm_raw_rmse_checkpoint_alignment_seed42_20260906/"
    "a_vs_b_metrics.json"
)
B_REFERENCE_EVIDENCE_SHA256 = (
    "17b1c432b58f0620d8e6eda1a291a92eb4c995af9c78d98be8d460be79079db3"
)
REQUIRED_COMMITTED_FILES = (
    CONTRACT_REL,
    B_REFERENCE_EVIDENCE_REL,
    PACKAGE_REL,
    LAUNCHER_REL,
    CONTROLLER_REL,
    TRAINER_REL,
    Path("paper/scripts/count_aware_tpp_backbone/constants.py"),
    Path("paper/scripts/count_aware_tpp_backbone/datasets.py"),
    Path("paper/scripts/count_aware_tpp_backbone/core.py"),
    Path("paper/scripts/count_aware_tpp_backbone/training.py"),
    Path("paper/scripts/count_aware_tpp_backbone/reporting.py"),
    Path("models/TPPs/CountAwareTPP.py"),
    Path("models/TPPs/CountAwareFactory.py"),
    Path("data_loader/event_seq_data_module.py"),
    Path("simple_lab_test/search/tests/test_raw_rmse_baseline_alignment_role.py"),
    Path("simple_lab_test/search/tests/test_raw_rmse_baseline_alignment_orchestration.py"),
    Path("simple_lab_test/search/tests/test_package_raw_rmse_baseline_alignment.py"),
)
SOURCE_POLICY = (
    "git_archive_of_one_clean_commit_plus_three_checksum_pinned_data_split_pairs"
)
SOURCE_MANIFEST_KEYS = frozenset({
    "schema_version",
    "source_revision",
    "contract",
    "contract_sha256",
    "B_reference_evidence",
    "B_reference_evidence_sha256",
    "files",
    "required_committed_files",
    "explicit_checksum_pinned_inputs",
    "explicit_untracked_pinned_inputs",
    "source_policy",
    "evaluation_scope",
    "held_out_test_evaluated",
})

CONTRACT_ID = "raw_rmse_baseline_alignment_seed42_v1"
MODEL_ROLE = "raw_rmse_baseline_alignment"
VARIANT = "count_only_log_regression"
BACKBONES = ("rmtpp", "thp")
DATASETS = (
    "intermittent_frozen_5000",
    "yellow_trip_hourly",
    "insta_market_basket",
)
HOST_ROLES = ("5080", "5090")
SELECTOR = "validation_raw_quantity_rmse"
HISTORY_KEY = "val_qty_rmse"
SELECTION = "best_validation_raw_quantity_rmse"
EXECUTION_ROOT_BASE = Path(
    "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts"
)
GPU_LOCK_PATH = Path("/tmp/paper_research_raw_rmse_baseline_alignment_gpu0.lock")
RUNTIME_FINGERPRINT_FILENAME = "runtime_fingerprint.json"
WRAPPER_TEMP_FILENAMES = frozenset({
    "job_identity.json.tmp",
    "job_status.json.tmp",
    f"{RUNTIME_FINGERPRINT_FILENAME}.tmp",
    "audit_receipt.json.tmp",
})


def fixed_execution_paths(revision: str) -> dict[str, Any]:
    """Derive the only artifact and coordination registry for one source commit."""
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "A full lowercase source revision is required",
    )
    root = EXECUTION_ROOT_BASE / f"{CONTRACT_ID}_{revision}"
    return {
        "root": root,
        "coordination_root": root / "coordination",
        "artifact_roots": {
            "5080": root / "artifacts" / "5080",
            "5090": root / "artifacts" / "5090",
        },
    }


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def save_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def assert_finite_tree(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            assert_finite_tree(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tree(child, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value at {path}: {value}")


def all_jobs(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    schedule = contract.get("schedule")
    require(isinstance(schedule, Mapping), "Contract schedule is missing")
    jobs = [
        *schedule.get("e1_jobs", []),
        *schedule.get("seed42_e300_jobs", []),
    ]
    require(all(isinstance(job, dict) for job in jobs), "Invalid schedule job")
    return jobs


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if the frozen immediate schedule or research scope drifts."""
    require(contract.get("schema_version") == 1, "Unsupported contract schema")
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong contract")
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope is missing")
    expected_scope = {
        "candidate_reference": "B_t0_raw_rmse seed42 (unchanged Hard-LMM architecture and log1p-MSE training loss)",
        "baseline_backbones": list(BACKBONES),
        "quantity_objective": "unweighted MSE on log1p quantity",
        "checkpoint_monitor": SELECTOR,
        "checkpoint_rule": "earliest strict finite minimum validation raw quantity RMSE",
        "early_stopping_monitor": SELECTOR,
        "quantity_variant": VARIANT,
        "time_head": "legacy_clamped_rmtpp with intercept cap 300",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "batch_size": 128,
        "gradient_clip": 1.0,
        "hidden_dim": 64,
        "maximum_epochs": 300,
        "minimum_epochs": 40,
        "patience": 40,
        "seeds": [42],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "seeds_52_62_authorized": False,
    }
    for key, expected in expected_scope.items():
        require(scope.get(key) == expected, f"Scope drift: {key}")

    dataset_rows = contract.get("datasets")
    require(isinstance(dataset_rows, list), "Dataset rows are missing")
    require(
        tuple(row.get("dataset") for row in dataset_rows) == DATASETS,
        "Dataset order/scope drifted",
    )
    datasets: dict[str, dict[str, Any]] = {}
    for row in dataset_rows:
        name = row["dataset"]
        require(int(row.get("expected_train_targets", 0)) > 0, f"Missing train count: {name}")
        require(int(row.get("expected_validation_targets", 0)) > 0, f"Missing validation count: {name}")
        metrics = row.get("B_t0_raw_rmse_validation_metrics")
        require(isinstance(metrics, Mapping), f"Missing B metrics: {name}")
        for metric in ("overall_mae", "raw_rmse"):
            value = float(metrics.get(metric, float("nan")))
            require(math.isfinite(value) and value >= 0.0, f"Invalid B {metric}: {name}")
        for key in (
            "data_sha256",
            "split_manifest_sha256",
            "expected_train_target_identity_sha256",
            "expected_train_target_quantity_sha256",
            "expected_validation_target_identity_sha256",
            "expected_validation_target_quantity_sha256",
            "B_t0_raw_rmse_checkpoint_file_sha256",
            "B_t0_raw_rmse_checkpoint_state_sha256",
        ):
            digest = str(row.get(key, ""))
            require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"Invalid {key}: {name}",
            )
        require(int(row.get("lookback", 0)) > 0, f"Invalid lookback: {name}")
        require(int(row.get("max_sequence_length", 0)) > 0, f"Invalid max sequence length: {name}")
        datasets[name] = row

    jobs = all_jobs(contract)
    require(len(jobs) == 12, "Immediate schedule must contain exactly 12 jobs")
    ids = [job.get("job_id") for job in jobs]
    require(len(set(ids)) == len(ids), "Schedule contains duplicate job IDs")
    expected_ids = [
        f"e1_{index:02d}_{dataset}_{backbone}_seed42"
        for index, (dataset, backbone) in enumerate(
            ((dataset, backbone) for dataset in DATASETS for backbone in BACKBONES),
            start=1,
        )
    ] + [
        f"seed42_e300_{index:02d}_{dataset}_{backbone}"
        for index, (dataset, backbone) in enumerate(
            (
                (dataset, backbone)
                for dataset in (
                    "insta_market_basket",
                    "intermittent_frozen_5000",
                    "yellow_trip_hourly",
                )
                for backbone in BACKBONES
            ),
            start=1,
        )
    ]
    require(ids == expected_ids, "Job IDs/order drifted")
    for job in jobs:
        require(job.get("dataset") in datasets, f"Unknown job dataset: {job}")
        require(job.get("backbone") in BACKBONES, f"Unknown job backbone: {job}")
        require(job.get("host_role") in HOST_ROLES, f"Unknown host role: {job}")
        require(job.get("seed") == 42, f"Only seed42 is authorized: {job}")
    e1 = jobs[:6]
    require(all(job["phase"] == "e1" for job in e1), "First six jobs must be e1")
    require(all(job["host_role"] == "5090" for job in e1), "All e1 jobs must run on 5090")
    require(
        [(job["dataset"], job["backbone"]) for job in e1]
        == [(dataset, backbone) for dataset in DATASETS for backbone in BACKBONES],
        "e1 order drifted",
    )
    require(
        all(
            (job["epochs"], job["min_epochs"], job["patience"]) == (1, 1, 1)
            for job in e1
        ),
        "e1 training contract drifted",
    )
    require(
        all(job["prerequisite"] == "previous_e1_job_passed" for job in e1),
        "e1 prerequisites drifted",
    )
    full = jobs[6:]
    expected_full = [
        (dataset, backbone, "5080" if backbone == "rmtpp" else "5090")
        for dataset in ("insta_market_basket", "intermittent_frozen_5000", "yellow_trip_hourly")
        for backbone in BACKBONES
    ]
    require(
        [(job["dataset"], job["backbone"], job["host_role"]) for job in full]
        == expected_full,
        "seed42 host assignment/order drifted",
    )
    require(
        all(
            (job["epochs"], job["min_epochs"], job["patience"]) == (300, 40, 40)
            for job in full
        ),
        "seed42 e300 training contract drifted",
    )
    require(
        all(job["prerequisite"] == "all_e1_jobs_passed" for job in full[:2]),
        "Instacart prerequisites drifted",
    )
    require(
        all(job["prerequisite"] == "instacart_gate_passed" for job in full[2:]),
        "Post-gate prerequisites drifted",
    )
    schedule = contract["schedule"]
    require(schedule.get("one_job_per_controller_invocation") is True, "Controller cardinality drifted")
    require(
        schedule.get("phase_order") == [
            "e1_all_six_on_5090",
            "instacart_seed42_split_across_hosts",
            "strict_instacart_gate",
            "intermittent_and_taxi_seed42_split_across_hosts",
        ],
        "Phase order drifted",
    )
    require(schedule.get("additional_seeds_in_immediate_stage") == [], "Immediate additional seeds drifted")
    gate = contract.get("instacart_gate")
    require(isinstance(gate, Mapping), "Instacart gate is missing")
    require(gate.get("dataset") == "insta_market_basket", "Gate dataset drifted")
    require(gate.get("baselines") == list(BACKBONES), "Gate baselines drifted")
    require(gate.get("metrics") == ["overall_mae", "raw_rmse"], "Gate metrics drifted")
    require(gate.get("comparison") == "strict_unrounded_less_than", "Gate comparison drifted")
    require(gate.get("candidate") == "B_t0_raw_rmse", "Gate candidate drifted")
    require(gate.get("missing_or_nonfinite_result") == "wait_or_fail_closed", "Gate missing-result rule drifted")
    held_out = contract.get("held_out_lock")
    require(isinstance(held_out, Mapping), "Held-out lock is missing")
    for key in (
        "test_target_loader_allowed",
        "test_metrics_allowed",
        "test_artifacts_allowed",
        "held_out_decision_use_allowed",
    ):
        require(held_out.get(key) is False, f"Held-out lock drift: {key}")
    require(held_out.get("selection_and_reporting_split") == "validation", "Selector split drifted")
    require(held_out.get("train_targets_split") == "train", "Train target split drifted")
    execution = contract.get("execution")
    require(isinstance(execution, Mapping), "Execution authorization is missing")
    require(execution.get("e1_six_on_5090_authorized") is True, "e1 is not authorized")
    require(execution.get("instacart_seed42_on_5080_and_5090_authorized") is True, "Instacart stage is not authorized")
    require(execution.get("conditional_intermittent_and_taxi_seed42_authorized") is True, "Conditional stage is not authorized")
    require(execution.get("seeds52_and62") is False, "Additional seeds are not authorized")
    require(execution.get("held_out_test") is False, "Held-out execution is authorized unexpectedly")
    require(execution.get("overwrite") is False, "Overwrite was authorized unexpectedly")
    require(execution.get("resume_exact_identity_only") is True, "Exact resume was disabled")
    runtime = contract.get("runtime_policy")
    require(isinstance(runtime, Mapping), "Runtime policy is missing")
    expected_runtime = {
        "5080": ("RTX 5080", False, ["rmtpp"]),
        "5090": ("RTX 5090", True, ["rmtpp", "thp"]),
    }
    for host, (gpu_fragment, gdm_inactive, permitted_backbones) in expected_runtime.items():
        policy = runtime.get("hosts", {}).get(host, {})
        require(policy.get("required_gpu_name_contains") == gpu_fragment, f"GPU policy drift: {host}")
        require(policy.get("minimum_free_vram_mib") == 12000, f"VRAM policy drift: {host}")
        require(policy.get("require_gdm_inactive") is gdm_inactive, f"GDM policy drift: {host}")
        require(policy.get("permitted_backbones") == permitted_backbones, f"Host backbone policy drift: {host}")
    require(
        "user-authorized RMTPP assignment" in runtime["hosts"]["5080"].get("active_gdm_exception", ""),
        "5080 desktop exception rationale drifted",
    )
    require(runtime.get("require_exactly_one_visible_gpu") is True, "Visible GPU cardinality drifted")
    require(runtime.get("require_no_existing_compute_process") is True, "GPU exclusivity drifted")
    require(runtime.get("deterministic_environment") == {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONHASHSEED": "42",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }, "Deterministic environment drifted")
    evidence = contract.get("B_reference_evidence")
    require(isinstance(evidence, Mapping), "B reference evidence is missing")
    expected_evidence = {
        "path": B_REFERENCE_EVIDENCE_REL.as_posix(),
        "sha256": B_REFERENCE_EVIDENCE_SHA256,
        "contract_id": "hard_lmm_quantile_checkpoint_alignment_v1",
        "status": "common_raw_rmse_goal_met",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for key, expected in expected_evidence.items():
        require(evidence.get(key) == expected, f"B reference evidence drift: {key}")
    return {"datasets": datasets, "jobs": {job["job_id"]: job for job in jobs}}


def verify_source_manifest(
    project: Path,
    *,
    revision: str,
    contract_path: Path,
) -> dict[str, Any]:
    require(
        len(revision) == 40 and all(character in "0123456789abcdef" for character in revision),
        "A full lowercase source revision is required",
    )
    manifest_path = project / "source_manifest.json"
    manifest = read_json(manifest_path)
    missing_keys = sorted(SOURCE_MANIFEST_KEYS - set(manifest))
    unexpected_keys = sorted(set(manifest) - SOURCE_MANIFEST_KEYS)
    require(
        not missing_keys and not unexpected_keys,
        "Source manifest schema drifted: "
        f"missing={missing_keys}, unexpected={unexpected_keys}",
    )
    require(manifest.get("schema_version") == 1, "Unsupported source manifest schema")
    require(manifest.get("source_revision") == revision, "Source revision mismatch")
    require(manifest.get("evaluation_scope") == "validation_only", "Source manifest scope drifted")
    require(manifest.get("held_out_test_evaluated") is False, "Source manifest opened held-out data")
    require(manifest.get("source_policy") == SOURCE_POLICY, "Source manifest policy drifted")

    project_resolved = project.resolve()
    contract_resolved = contract_path.resolve()
    try:
        contract_relative = contract_resolved.relative_to(project_resolved)
    except ValueError as exc:
        raise ValueError("Contract must be inside the packaged project") from exc
    require(contract_relative == CONTRACT_REL, "Packaged contract path drifted")
    require(manifest.get("contract") == CONTRACT_REL.as_posix(), "Manifest contract path drifted")
    require(
        manifest.get("B_reference_evidence") == B_REFERENCE_EVIDENCE_REL.as_posix(),
        "Manifest B reference evidence path drifted",
    )
    require(
        manifest.get("required_committed_files")
        == [path.as_posix() for path in REQUIRED_COMMITTED_FILES],
        "Manifest required committed file set/order drifted",
    )

    contract = read_json(contract_resolved)
    dataset_rows = contract.get("datasets")
    require(isinstance(dataset_rows, list) and len(dataset_rows) == 3, "Manifest contract dataset scope drifted")
    pinned_inputs: list[str] = []
    pinned_input_digests: dict[str, str] = {}
    for row in dataset_rows:
        require(isinstance(row, Mapping), "Manifest contract dataset row is invalid")
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            name = str(row.get(path_key, ""))
            relative = Path(name)
            require(
                bool(name)
                and not relative.is_absolute()
                and ".." not in relative.parts
                and relative.as_posix() == name,
                f"Unsafe pinned input path: {name}",
            )
            digest = str(row.get(digest_key, ""))
            require(
                len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"Invalid pinned input checksum: {name}",
            )
            pinned_inputs.append(name)
            pinned_input_digests[name] = digest
    expected_pinned_inputs = sorted(pinned_inputs)
    require(
        len(expected_pinned_inputs) == 6 and len(set(expected_pinned_inputs)) == 6,
        "Manifest contract must name six distinct pinned inputs",
    )
    require(
        manifest.get("explicit_checksum_pinned_inputs") == expected_pinned_inputs,
        "Manifest checksum-pinned inputs drifted",
    )
    require(
        manifest.get("explicit_untracked_pinned_inputs") == expected_pinned_inputs,
        "Manifest legacy pinned inputs drifted",
    )

    files = manifest.get("files")
    require(isinstance(files, Mapping) and bool(files), "Source manifest is empty")
    for name, expected in files.items():
        require(isinstance(name, str) and bool(name), "Manifest contains an invalid file name")
        relative = Path(name)
        require(
            not relative.is_absolute()
            and ".." not in relative.parts
            and relative.as_posix() == name
            and name != "source_manifest.json",
            f"Unsafe manifest path: {name}",
        )
        require(
            isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            f"Invalid manifest checksum: {name}",
        )
        actual = project / relative
        require(actual.is_file(), f"Manifest file is missing: {name}")
        actual_resolved = actual.resolve()
        require(
            actual_resolved == project_resolved or project_resolved in actual_resolved.parents,
            f"Manifest file resolves outside packaged project: {name}",
        )
        require(sha256_file(actual) == expected, f"Manifest checksum mismatch: {name}")

    actual_inventory = {
        path.relative_to(project).as_posix()
        for path in project.rglob("*")
        if path.is_file() and path != manifest_path
    }
    declared_inventory = set(files)
    unmanifested = sorted(actual_inventory - declared_inventory)
    absent = sorted(declared_inventory - actual_inventory)
    require(
        not unmanifested and not absent,
        "Packaged project inventory differs from source manifest: "
        f"unmanifested={unmanifested}, absent={absent}",
    )
    for required in REQUIRED_COMMITTED_FILES:
        require(
            required.as_posix() in declared_inventory,
            f"Required committed source is not manifested: {required}",
        )
    for pinned in expected_pinned_inputs:
        require(pinned in declared_inventory, f"Pinned input is not manifested: {pinned}")
        require(
            files.get(pinned) == pinned_input_digests[pinned],
            f"Pinned input manifest checksum differs from contract: {pinned}",
        )

    contract_digest = sha256_file(contract_resolved)
    require(
        manifest.get("contract_sha256") == contract_digest
        and files.get(CONTRACT_REL.as_posix()) == contract_digest,
        "Manifest contract checksum drifted",
    )
    evidence_path = project / B_REFERENCE_EVIDENCE_REL
    evidence_digest = sha256_file(evidence_path)
    require(
        evidence_digest == B_REFERENCE_EVIDENCE_SHA256
        and manifest.get("B_reference_evidence_sha256") == evidence_digest
        and files.get(B_REFERENCE_EVIDENCE_REL.as_posix()) == evidence_digest,
        "Manifest B reference evidence checksum drifted",
    )
    reference = contract.get("B_reference_evidence")
    require(isinstance(reference, Mapping), "Packaged contract B reference evidence is missing")
    require(
        reference.get("path") == B_REFERENCE_EVIDENCE_REL.as_posix()
        and reference.get("sha256") == evidence_digest,
        "Packaged contract B reference evidence pin drifted",
    )
    return manifest


def validate_b_reference_evidence(
    project: Path,
    *,
    contract: Mapping[str, Any],
    datasets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind every B gate scalar and identity to the tracked audit evidence."""
    reference = contract["B_reference_evidence"]
    relative = Path(str(reference["path"]))
    require(
        relative == B_REFERENCE_EVIDENCE_REL
        and not relative.is_absolute()
        and ".." not in relative.parts,
        "Unsafe or unexpected B reference evidence path",
    )
    path = project / relative
    require(path.is_file(), "B reference evidence is missing")
    require(sha256_file(path) == B_REFERENCE_EVIDENCE_SHA256, "B reference evidence checksum drifted")
    evidence = read_json(path)
    expected_top = {
        "schema_version": 1,
        "status": reference["status"],
        "contract_id": reference["contract_id"],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "comparison": "A_t0_joint_vs_B_t0_raw_rmse",
    }
    for key, expected in expected_top.items():
        require(evidence.get(key) == expected, f"B evidence mismatch: {key}")
    rows = evidence.get("datasets")
    require(isinstance(rows, list), "B evidence datasets are missing")
    require(tuple(row.get("dataset") for row in rows) == DATASETS, "B evidence dataset order drifted")
    for evidence_row in rows:
        dataset = str(evidence_row["dataset"])
        contract_row = datasets[dataset]
        b_metrics = evidence_row.get("metrics", {}).get("B_t0_raw_rmse", {})
        expected_metrics = contract_row["B_t0_raw_rmse_validation_metrics"]
        for metric in ("overall_mae", "raw_rmse"):
            require(
                b_metrics.get(metric) == expected_metrics[metric],
                f"B {metric} is not exactly bound to evidence: {dataset}",
            )
        require(
            b_metrics.get("validation_target_count")
            == contract_row["expected_validation_targets"],
            f"B validation count drifted: {dataset}",
        )
        identity = evidence_row.get("validation_identity", {})
        require(identity.get("counts_equal") is True, f"B validation counts were not paired: {dataset}")
        require(identity.get("identities_equal") is True, f"B validation identities were not paired: {dataset}")
        require(
            identity.get("target_identity_sha256")
            == contract_row["expected_validation_target_identity_sha256"],
            f"B validation target identity drifted: {dataset}",
        )
        b_artifact = evidence_row.get("evidence", {}).get("B_t0_raw_rmse", {})
        require(
            b_artifact.get("checkpoint_file_sha256")
            == contract_row["B_t0_raw_rmse_checkpoint_file_sha256"],
            f"B checkpoint file digest drifted: {dataset}",
        )
        require(
            b_artifact.get("checkpoint_state_sha256")
            == contract_row["B_t0_raw_rmse_checkpoint_state_sha256"],
            f"B checkpoint state digest drifted: {dataset}",
        )
        require(
            b_artifact.get("earliest_raw_rmse") == expected_metrics["raw_rmse"],
            f"B selected raw RMSE drifted: {dataset}",
        )
        require(
            int(b_artifact.get("validation_target_count", -1))
            == int(contract_row["expected_validation_targets"]),
            f"B artifact validation count drifted: {dataset}",
        )
    return evidence


def verify_dataset_inputs(project: Path, datasets: Mapping[str, Mapping[str, Any]]) -> None:
    for dataset, row in datasets.items():
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            path = project / str(row[path_key])
            require(path.is_file(), f"Missing {dataset} input: {path_key}")
            require(sha256_file(path) == row[digest_key], f"{dataset} {path_key} checksum mismatch")


def command_output(command: Sequence[str], *, allowed: tuple[int, ...] = (0,)) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if result.returncode not in allowed:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {list(command)!r}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def gpu_preflight(
    contract: Mapping[str, Any],
    host_role: str,
    *,
    backbone: str,
) -> dict[str, Any]:
    require(host_role in HOST_ROLES, f"Unknown host role: {host_role}")
    runtime = contract["runtime_policy"]
    policy = runtime["hosts"][host_role]
    require(backbone in policy["permitted_backbones"], f"Backbone {backbone} is not permitted on {host_role}")
    rows = command_output(
        ["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader,nounits"]
    ).splitlines()
    require(len(rows) == 1, "Expected exactly one visible GPU")
    name, raw_free = (part.strip() for part in rows[0].split(",", 1))
    free_mib = int(raw_free)
    require(policy["required_gpu_name_contains"] in name, f"Wrong GPU for {host_role}: {name}")
    require(free_mib >= int(policy["minimum_free_vram_mib"]), f"Insufficient free VRAM: {free_mib} MiB")
    compute = command_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    )
    require(not compute, f"GPU already has compute processes: {compute}")
    gdm = command_output(["systemctl", "is-active", "gdm"], allowed=(0, 3))
    if policy["require_gdm_inactive"]:
        require(gdm == "inactive", f"GDM must be inactive, got {gdm!r}")
    else:
        require(gdm in {"active", "inactive"}, f"Unexpected GDM state: {gdm!r}")
        if gdm == "active":
            require(host_role == "5080" and backbone == "rmtpp", "Active GDM exception broadened")
    return {
        "gpu_name": name,
        "free_vram_mib": free_mib,
        "compute_processes": [],
        "gdm": gdm,
    }


def validate_runtime_fingerprint(
    payload: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate stable runtime versions and the launch-time GPU preflight record."""
    require(
        set(payload) == {
            "schema_version",
            "host_role",
            "runtime",
            "launch_preflight",
        },
        "Runtime fingerprint schema drifted",
    )
    require(payload.get("schema_version") == 1, "Runtime fingerprint schema is unsupported")
    host_role = str(job["host_role"])
    require(payload.get("host_role") == host_role, "Runtime fingerprint host drifted")
    runtime = payload.get("runtime")
    require(isinstance(runtime, Mapping), "Runtime fingerprint payload is missing")
    require(
        set(runtime) == {
            "interpreter_executable",
            "python_version",
            "python_implementation",
            "torch_version",
            "torch_cuda_version",
            "cudnn_version",
            "numpy_version",
            "polars_version",
            "cuda_available",
            "cuda_device_count",
            "cuda_current_device",
            "gpu_name",
            "gpu_compute_capability",
            "gpu_total_memory_bytes",
        },
        "Runtime version fingerprint schema drifted",
    )
    for key in (
        "interpreter_executable",
        "python_version",
        "python_implementation",
        "torch_version",
        "torch_cuda_version",
        "numpy_version",
        "polars_version",
        "gpu_name",
    ):
        require(isinstance(runtime.get(key), str) and bool(runtime[key]), f"Runtime {key} is missing")
    require(Path(str(runtime["interpreter_executable"])).is_absolute(), "Runtime interpreter is not absolute")
    require(runtime.get("cuda_available") is True, "Runtime CUDA is unavailable")
    require(runtime.get("cuda_device_count") == 1, "Runtime must expose exactly one CUDA GPU")
    require(runtime.get("cuda_current_device") == 0, "Runtime CUDA device index drifted")
    require(
        isinstance(runtime.get("cudnn_version"), int) and int(runtime["cudnn_version"]) > 0,
        "Runtime cuDNN version is missing",
    )
    capability = runtime.get("gpu_compute_capability")
    require(
        isinstance(capability, list)
        and len(capability) == 2
        and all(isinstance(value, int) and value >= 0 for value in capability),
        "Runtime GPU compute capability is invalid",
    )
    require(
        isinstance(runtime.get("gpu_total_memory_bytes"), int)
        and int(runtime["gpu_total_memory_bytes"]) > 0,
        "Runtime GPU memory is invalid",
    )
    policy = contract["runtime_policy"]["hosts"][host_role]
    require(
        str(policy["required_gpu_name_contains"]) in str(runtime["gpu_name"]),
        "Runtime GPU model drifted",
    )

    preflight = payload.get("launch_preflight")
    require(isinstance(preflight, Mapping), "Launch-time GPU preflight is missing")
    require(
        set(preflight) == {
            "gpu_name",
            "free_vram_mib",
            "compute_processes",
            "gdm",
        },
        "Launch-time GPU preflight schema drifted",
    )
    require(preflight.get("gpu_name") == runtime.get("gpu_name"), "Preflight/runtime GPU name differs")
    require(
        isinstance(preflight.get("free_vram_mib"), int)
        and int(preflight["free_vram_mib"]) >= int(policy["minimum_free_vram_mib"]),
        "Launch-time free VRAM evidence is insufficient",
    )
    require(preflight.get("compute_processes") == [], "Launch-time GPU was not idle")
    gdm = preflight.get("gdm")
    if policy["require_gdm_inactive"]:
        require(gdm == "inactive", "Launch-time GDM evidence is not inactive")
    else:
        require(gdm in {"active", "inactive"}, "Launch-time GDM evidence is invalid")
        if gdm == "active":
            require(host_role == "5080" and job["backbone"] == "rmtpp", "GDM exception broadened")
    return dict(payload)


def collect_runtime_fingerprint(
    *,
    python: str,
    project: Path,
    contract: Mapping[str, Any],
    job: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Query the exact trainer interpreter rather than the controller process."""
    program = "\n".join((
        "import json, os, platform, sys",
        "import numpy, polars, torch",
        "assert torch.cuda.is_available()",
        "assert torch.cuda.device_count() == 1",
        "device = torch.cuda.current_device()",
        "properties = torch.cuda.get_device_properties(device)",
        "payload = {",
        "  'interpreter_executable': os.path.realpath(sys.executable),",
        "  'python_version': platform.python_version(),",
        "  'python_implementation': platform.python_implementation(),",
        "  'torch_version': str(torch.__version__),",
        "  'torch_cuda_version': str(torch.version.cuda),",
        "  'cudnn_version': torch.backends.cudnn.version(),",
        "  'numpy_version': str(numpy.__version__),",
        "  'polars_version': str(polars.__version__),",
        "  'cuda_available': True,",
        "  'cuda_device_count': torch.cuda.device_count(),",
        "  'cuda_current_device': device,",
        "  'gpu_name': torch.cuda.get_device_name(device),",
        "  'gpu_compute_capability': [properties.major, properties.minor],",
        "  'gpu_total_memory_bytes': properties.total_memory,",
        "}",
        "print(json.dumps(payload, sort_keys=True))",
    ))
    result = subprocess.run(
        [python, "-c", program],
        cwd=project,
        env=deterministic_env(contract, project),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    require(
        result.returncode == 0,
        f"Runtime fingerprint command failed ({result.returncode}): {result.stderr.strip()}",
    )
    runtime = json.loads(result.stdout, object_pairs_hook=_reject_duplicate_pairs)
    require(isinstance(runtime, dict), "Runtime fingerprint command returned a non-object")
    payload = {
        "schema_version": 1,
        "host_role": job["host_role"],
        "runtime": runtime,
        "launch_preflight": dict(preflight),
    }
    validate_runtime_fingerprint(payload, contract=contract, job=job)
    expected_interpreter = Path(python).resolve()
    require(
        Path(str(runtime["interpreter_executable"])).resolve() == expected_interpreter,
        "Runtime fingerprint used a different interpreter",
    )
    return payload


def audit_runtime_fingerprint_file(
    path: Path,
    *,
    contract: Mapping[str, Any],
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = read_json(path)
    validate_runtime_fingerprint(payload, contract=contract, job=job)
    return payload, sha256_file(path)


def deterministic_env(contract: Mapping[str, Any], project: Path) -> dict[str, str]:
    configured = {
        str(key): str(value)
        for key, value in contract["runtime_policy"]["deterministic_environment"].items()
    }
    return {
        **os.environ,
        **configured,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "PYTHONPATH": str(project),
    }


def training_command(
    *,
    python: str,
    project: Path,
    training_output: Path,
    revision: str,
    dataset_row: Mapping[str, Any],
    job: Mapping[str, Any],
) -> list[str]:
    """Build the only trainer invocation allowed by this experiment contract."""
    return [
        python,
        str(project / TRAINER_REL),
        "--data", str(project / str(dataset_row["data_path"])),
        "--split-manifest", str(project / str(dataset_row["split_manifest_path"])),
        "--output-dir", str(training_output),
        "--source-revision", revision,
        "--execution-role", f"raw_rmse_baseline_alignment_{job['phase']}_{job['host_role']}_{job['job_id']}",
        "--dataset-contract", str(job["dataset"]),
        "--model-role", MODEL_ROLE,
        "--device", "cuda",
        "--epochs", str(job["epochs"]),
        "--min-epochs", str(job["min_epochs"]),
        "--early-stopping-patience", str(job["patience"]),
        "--batch-size", "128",
        "--lr", "0.001",
        "--lookback-weeks", str(dataset_row["lookback"]),
        "--max-seq-len", str(dataset_row["max_sequence_length"]),
        "--hidden-dim", "64",
        "--lambda-log-qty", "1",
        "--lambda-tail", "0",
        "--grad-clip", "1",
        "--backbones", str(job["backbone"]),
        "--seeds", "42",
        "--quantity-variants", "log_mse",
        "--checkpoint-monitor", SELECTOR,
        "--quantile-adaptive-strength", "0",
        "--time-head-mode", "legacy_clamped_rmtpp",
        "--time-scale", "3",
        "--time-w-max", str(10.0 / 3.0),
        "--time-intercept-limit", "300",
        "--allow-partial-contract",
    ]


def _weighted_body_mae(rows: Sequence[Mapping[str, Any]]) -> float:
    names = {"le_p50", "p50_p90", "p90_p95"}
    selected = [row for row in rows if row.get("stratum") in names]
    require({row.get("stratum") for row in selected} == names, "Body quantity strata are incomplete")
    count = sum(int(row["count"]) for row in selected)
    require(count > 0, "Body quantity strata are empty")
    return sum(float(row["qty_mae"]) * int(row["count"]) for row in selected) / count


def audit_selected_quantity_metrics(
    *,
    summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    quantity_rows: Sequence[Mapping[str, Any]],
    expected_count: int,
) -> dict[str, float]:
    """Reconstruct both gate metrics from the disjoint quantity strata."""
    selected = _earliest_minimum(history)
    selected_rmse = float(selected[HISTORY_KEY])
    require(
        float(summary.get("selected_metric_value", float("nan"))) == selected_rmse,
        "Summary selected metric differs from selected history",
    )
    require(
        float(checkpoint.get("selected_metric_value", float("nan"))) == selected_rmse,
        "Checkpoint selected metric differs from selected history",
    )
    summary_rmse = float(summary.get("best_val_qty_rmse", float("nan")))
    require(
        math.isclose(summary_rmse, selected_rmse, rel_tol=1e-12, abs_tol=1e-10),
        "Summary best raw RMSE differs from selected history/checkpoint",
    )

    total_count = 0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    for row in quantity_rows:
        count = int(row.get("count", 0))
        mae = float(row.get("qty_mae", float("nan")))
        rmse = float(row.get("qty_rmse", float("nan")))
        require(count > 0, "Quantity stratum has no observations")
        require(math.isfinite(mae) and mae >= 0.0, "Quantity stratum MAE is invalid")
        require(math.isfinite(rmse) and rmse >= 0.0, "Quantity stratum RMSE is invalid")
        total_count += count
        absolute_error_sum += mae * count
        squared_error_sum += rmse * rmse * count
    require(total_count == int(expected_count), "Validation target count mismatch")
    overall_mae = absolute_error_sum / total_count
    raw_rmse = math.sqrt(squared_error_sum / total_count)
    require(
        math.isclose(
            float(selected.get("val_qty_mae", float("nan"))),
            overall_mae,
            rel_tol=1e-12,
            abs_tol=1e-10,
        ),
        "Selected history MAE differs from quantity strata",
    )
    require(
        math.isclose(
            float(summary.get("best_val_qty_mae", float("nan"))),
            overall_mae,
            rel_tol=1e-12,
            abs_tol=1e-10,
        ),
        "Summary overall MAE differs from quantity strata",
    )
    require(
        math.isclose(summary_rmse, raw_rmse, rel_tol=1e-12, abs_tol=1e-10),
        "Summary raw RMSE differs from quantity strata",
    )
    return {"overall_mae": overall_mae, "raw_rmse": raw_rmse}


def expected_training_artifact_files(job: Mapping[str, Any]) -> set[str]:
    run = f"runs/{job['backbone']}/{VARIANT}/seed_42"
    return {
        "launch_contract.json",
        "run_summaries.csv",
        "quantity_seed_metrics.csv",
        "history_seed_metrics.csv",
        "quantity_summary.csv",
        "history_summary.csv",
        f"{run}/summary.json",
        f"{run}/history.json",
        f"{run}/best_val_qty_rmse_model.pt",
        f"{run}/last_epoch_state.pt",
        f"{run}/train.log",
    }


def audit_completed_job_layout(
    *,
    job_dir: Path,
    job: Mapping[str, Any],
    require_receipt: bool,
) -> None:
    """Reject unknown files and empty directories anywhere in a completed job."""
    symlinks = sorted(
        path.relative_to(job_dir).as_posix()
        for path in job_dir.rglob("*")
        if path.is_symlink()
    )
    require(not symlinks, f"Job artifact contains symlinks: {symlinks}")
    training_files = {
        f"training/{relative}" for relative in expected_training_artifact_files(job)
    }
    stable_files = {
        "job.lock",
        "job_identity.json",
        "job_status.json",
        "launcher.log",
        RUNTIME_FINGERPRINT_FILENAME,
        *training_files,
    }
    allowed_files = {*stable_files, "audit_receipt.json", *WRAPPER_TEMP_FILENAMES}
    required_files = {
        *stable_files,
        *({"audit_receipt.json"} if require_receipt else set()),
    }
    observed_files = {
        path.relative_to(job_dir).as_posix()
        for path in job_dir.rglob("*")
        if path.is_file()
    }
    unexpected_files = sorted(observed_files - allowed_files)
    missing_files = sorted(required_files - observed_files)
    require(not unexpected_files, f"Job artifact file is outside the allowlist: {unexpected_files}")
    require(not missing_files, f"Required job artifact file is missing: {missing_files}")

    run = Path("training/runs") / str(job["backbone"]) / VARIANT / "seed_42"
    allowed_directories = {
        "training",
        "training/runs",
        f"training/runs/{job['backbone']}",
        f"training/runs/{job['backbone']}/{VARIANT}",
        run.as_posix(),
    }
    observed_directories = {
        path.relative_to(job_dir).as_posix()
        for path in job_dir.rglob("*")
        if path.is_dir()
    }
    require(
        observed_directories == allowed_directories,
        "Job artifact directory layout drifted: "
        f"unexpected={sorted(observed_directories - allowed_directories)}, "
        f"missing={sorted(allowed_directories - observed_directories)}",
    )


def audit_incomplete_job_layout(
    *,
    job_dir: Path,
    job: Mapping[str, Any],
) -> None:
    """Reject files that an interrupted invocation could not have produced."""
    symlinks = sorted(
        path.relative_to(job_dir).as_posix()
        for path in job_dir.rglob("*")
        if path.is_symlink()
    )
    require(not symlinks, f"Incomplete job artifact contains symlinks: {symlinks}")

    training_files = {
        f"training/{relative}" for relative in expected_training_artifact_files(job)
    }
    stable_files = {
        "job.lock",
        "job_identity.json",
        "job_status.json",
        "launcher.log",
        RUNTIME_FINGERPRINT_FILENAME,
        *training_files,
    }
    # Atomic writers can leave their fixed temporary neighbor after SIGKILL.
    temporary_files = {
        *WRAPPER_TEMP_FILENAMES,
        "training/launch_contract.failed.tmp",
        *{f"{path}.tmp" for path in training_files},
    }
    observed_files = {
        path.relative_to(job_dir).as_posix()
        for path in job_dir.rglob("*")
        if path.is_file()
    }
    unexpected_files = sorted(observed_files - stable_files - temporary_files)
    require(
        not unexpected_files,
        f"Incomplete job artifact file is outside the allowlist: {unexpected_files}",
    )

    run = Path("training/runs") / str(job["backbone"]) / VARIANT / "seed_42"
    allowed_directories = {
        "training",
        "training/runs",
        f"training/runs/{job['backbone']}",
        f"training/runs/{job['backbone']}/{VARIANT}",
        run.as_posix(),
    }
    observed_directories = {
        path.relative_to(job_dir).as_posix()
        for path in job_dir.rglob("*")
        if path.is_dir()
    }
    unexpected_directories = sorted(observed_directories - allowed_directories)
    require(
        not unexpected_directories,
        "Incomplete job artifact directory is outside the allowlist: "
        f"{unexpected_directories}",
    )


def _earliest_minimum(history: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    require(bool(history), "Training history is empty")
    selected: Mapping[str, Any] | None = None
    best = float("inf")
    for expected_epoch, row in enumerate(history, start=1):
        require(int(row.get("epoch", -1)) == expected_epoch, "History epochs are not consecutive")
        value = float(row.get(HISTORY_KEY, float("nan")))
        require(math.isfinite(value), "History raw RMSE is non-finite")
        if selected is None or value < best:
            selected = row
            best = value
    assert selected is not None
    return selected


def audit_resume_state(
    *,
    resume: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    summary: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    job: Mapping[str, Any],
    dataset_row: Mapping[str, Any],
    revision: str,
) -> dict[str, str]:
    """Verify the full exact-resume identity and both current/best states."""
    import torch

    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    require(resume.get("checkpoint_type") == "epoch_resume", "Resume checkpoint type drifted")
    require(resume.get("checkpoint_schema_version") == 2, "Resume checkpoint schema drifted")
    expected_selector = {
        "checkpoint_monitor": SELECTOR,
        "checkpoint_monitor_history_key": HISTORY_KEY,
        "checkpoint_selection": SELECTION,
    }
    for key, expected in expected_selector.items():
        require(resume.get(key) == expected, f"Resume {key} drifted")
        require(checkpoint.get(key) == expected, f"Selected checkpoint {key} drifted")
        require(summary.get(key) == expected, f"Summary {key} drifted")
    resume_identity = resume.get("resume_identity")
    require(isinstance(resume_identity, Mapping), "Resume exact identity is missing")
    require(checkpoint.get("resume_identity") == resume_identity, "Checkpoint/resume identity mismatch")
    require(summary.get("resume_identity") == resume_identity, "Summary/resume identity mismatch")
    expected_identity = {
        "backbone": job["backbone"],
        "variant": VARIANT,
        "seed": 42,
        "checkpoint_monitor": SELECTOR,
    }
    for key, expected in expected_identity.items():
        require(resume_identity.get(key) == expected, f"Resume identity drifted: {key}")
    arguments = resume_identity.get("arguments")
    require(isinstance(arguments, Mapping), "Resume argument identity is missing")
    expected_arguments = {
        "epochs": job["epochs"],
        "batch_size": 128,
        "lr": 0.001,
        "lookback_weeks": dataset_row["lookback"],
        "max_seq_len": dataset_row["max_sequence_length"],
        "hidden_dim": 64,
        "lambda_log_qty": 1.0,
        "grad_clip": 1.0,
        "early_stopping_patience": job["patience"],
        "min_epochs": job["min_epochs"],
        "lambda_tail": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 300.0,
        "max_train_batches": None,
        "max_val_batches": None,
        "model_role": MODEL_ROLE,
        "dataset_contract": job["dataset"],
        "max_series": None,
        "allow_partial_contract": True,
        "quantile_adaptive_strength": 0.0,
        "device": "cuda",
        "source_revision": revision,
        "data_sha256": dataset_row["data_sha256"],
        "split_manifest_sha256": dataset_row["split_manifest_sha256"],
    }
    for key, expected in expected_arguments.items():
        require(arguments.get(key) == expected, f"Resume argument drifted: {key}")
    require(
        arguments.get("execution_role")
        == f"raw_rmse_baseline_alignment_{job['phase']}_{job['host_role']}_{job['job_id']}",
        "Resume execution role drifted",
    )
    require(int(resume.get("epoch", -1)) == len(history), "Resume epoch/history length mismatch")
    require(resume.get("history") == list(history), "Resume/JSON history mismatch")
    selected = _earliest_minimum(history)
    require(int(resume.get("best_epoch", -1)) == int(selected["epoch"]), "Resume best epoch drifted")
    require(
        float(resume.get("best_selection_value", float("nan")))
        == float(selected[HISTORY_KEY]),
        "Resume best selection value drifted",
    )
    require(int(checkpoint.get("best_epoch", -1)) == int(selected["epoch"]), "Checkpoint best epoch drifted")
    require(
        float(checkpoint.get("selected_metric_value", float("nan")))
        == float(selected[HISTORY_KEY]),
        "Checkpoint selected metric drifted",
    )
    for payload_name, payload in (("resume", resume), ("checkpoint", checkpoint)):
        require(payload.get("source_revision") == revision, f"{payload_name} source revision drifted")
        require(payload.get("source_revision_history") == [revision], f"{payload_name} source history drifted")
        require(payload.get("evaluation_scope") == "validation_only", f"{payload_name} scope drifted")
        require(payload.get("held_out_test_evaluated") is False, f"{payload_name} evaluated held-out test")
    require(summary.get("source_revision_history") == [revision], "Summary source history drifted")
    initial = summary.get("initial_state_sha256")
    require(
        isinstance(initial, str)
        and len(initial) == 64
        and checkpoint.get("initial_state_sha256") == initial
        and resume.get("initial_state_sha256") == initial,
        "Initial state identity drifted",
    )
    current_state = resume.get("model_state_dict")
    best_state = resume.get("best_state_dict")
    selected_state = checkpoint.get("model_state_dict")
    require(isinstance(current_state, dict), "Resume current model state is missing")
    require(isinstance(best_state, dict), "Resume best model state is missing")
    require(isinstance(selected_state, dict), "Selected model state is missing")
    current_digest = canonical_state_dict_sha256(current_state)
    best_digest = canonical_state_dict_sha256(best_state)
    selected_digest = canonical_state_dict_sha256(selected_state)
    require(resume.get("model_state_sha256") == current_digest, "Resume current state digest drifted")
    require(resume.get("best_state_sha256") == best_digest, "Resume best state digest drifted")
    require(checkpoint.get("model_state_sha256") == selected_digest, "Selected state digest drifted")
    require(best_digest == selected_digest, "Resume best state differs from selected checkpoint")
    require(summary.get("checkpoint_state_sha256") == selected_digest, "Summary selected state drifted")
    optimizer = resume.get("optimizer_state_dict")
    require(isinstance(optimizer, Mapping), "Resume optimizer state is missing")
    require(isinstance(optimizer.get("state"), dict) and bool(optimizer["state"]), "Resume optimizer state is empty")
    require(isinstance(optimizer.get("param_groups"), list) and bool(optimizer["param_groups"]), "Resume optimizer groups are missing")
    rng = resume.get("rng_state")
    require(isinstance(rng, Mapping), "Resume RNG state is missing")
    require({"python", "numpy", "torch", "cuda"}.issubset(rng), "Resume RNG state is incomplete")
    require(isinstance(resume.get("train_loader_generator_state"), torch.Tensor), "Resume loader RNG state is missing")
    return {
        "resume_identity_sha256": canonical_sha256(resume_identity),
        "resume_current_state_sha256": current_digest,
        "resume_best_state_sha256": best_digest,
    }


def audit_training_artifact(
    *,
    training_output: Path,
    job: Mapping[str, Any],
    dataset_row: Mapping[str, Any],
    contract: Mapping[str, Any],
    revision: str,
    require_receipt: bool = False,
) -> dict[str, Any]:
    """Audit identity, selector, completeness, checkpoint replay, and test lock."""
    import torch

    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    launch_path = training_output / "launch_contract.json"
    launch = read_json(launch_path)
    assert_finite_tree(launch)
    expected_launch = {
        "status": "complete",
        "model_role": MODEL_ROLE,
        "dataset": job["dataset"],
        "data_sha256": dataset_row["data_sha256"],
        "split_manifest_sha256": dataset_row["split_manifest_sha256"],
        "quantity_variants": [VARIANT],
        "backbones": [job["backbone"]],
        "seeds": [42],
        "expected_run_count": 1,
        "completed_run_count": 1,
        "epochs": job["epochs"],
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": dataset_row["lookback"],
        "max_seq_len": dataset_row["max_sequence_length"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": revision,
        "partial_smoke": False,
    }
    for key, expected in expected_launch.items():
        require(launch.get(key) == expected, f"Launch mismatch for {key}")
    split_rows = launch.get("split_rows")
    require(isinstance(split_rows, Mapping), "Launch split rows are missing")
    require(set(split_rows) == {"train", "validation"}, "Held-out rows were materialized")
    require(int(split_rows.get("train", 0)) > 0, "Train rows are missing")
    require(int(split_rows.get("validation", 0)) > 0, "Validation rows are missing")
    validation_population = launch.get("validation_target_population")
    require(isinstance(validation_population, Mapping), "Validation target identity is missing")
    expected_population = {
        "target_count": dataset_row["expected_validation_targets"],
        "target_identity_sha256": dataset_row["expected_validation_target_identity_sha256"],
        "target_quantity_sha256": dataset_row["expected_validation_target_quantity_sha256"],
    }
    for key, expected in expected_population.items():
        require(validation_population.get(key) == expected, f"Validation target identity mismatch: {key}")
    early = launch.get("early_stopping")
    require(isinstance(early, Mapping), "Early-stopping contract is missing")
    for key, expected in {
        "monitor": SELECTOR,
        "min_epochs": job["min_epochs"],
        "patience": job["patience"],
        "comparison": "earliest_strict_finite_minimum",
        "restore": SELECTION,
    }.items():
        require(early.get(key) == expected, f"Early-stopping mismatch for {key}")
    require(launch.get("time_head", {}).get("time_intercept_limit") == 300.0, "Legacy time cap drifted")

    observed_files = {
        path.relative_to(training_output).as_posix()
        for path in training_output.rglob("*")
        if path.is_file()
    }
    expected_files = expected_training_artifact_files(job)
    unexpected_files = sorted(observed_files - expected_files)
    missing_files = sorted(expected_files - observed_files)
    require(not unexpected_files, f"Artifact is outside the validation-only allowlist: {unexpected_files}")
    require(not missing_files, f"Required validation artifact is missing: {missing_files}")
    job_dir = training_output.parent
    audit_completed_job_layout(
        job_dir=job_dir,
        job=job,
        require_receipt=require_receipt,
    )
    runtime_fingerprint_path = job_dir / RUNTIME_FINGERPRINT_FILENAME
    runtime_fingerprint, runtime_fingerprint_sha256 = audit_runtime_fingerprint_file(
        runtime_fingerprint_path,
        contract=contract,
        job=job,
    )

    run_dir = training_output / "runs" / str(job["backbone"]) / VARIANT / "seed_42"
    summary_path = run_dir / "summary.json"
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best_val_qty_rmse_model.pt"
    resume_path = run_dir / "last_epoch_state.pt"
    summary = read_json(summary_path)
    history_payload = read_json(history_path)
    history = history_payload.get("history")
    require(isinstance(history, list), "History payload is invalid")
    assert_finite_tree(summary)
    assert_finite_tree(history)
    expected_summary = {
        "status": "success",
        "backbone": job["backbone"],
        "variant": VARIANT,
        "seed": 42,
        "epochs": job["epochs"],
        "checkpoint_monitor": SELECTOR,
        "checkpoint_monitor_history_key": HISTORY_KEY,
        "checkpoint_selection": SELECTION,
        "source_revision": revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for key, expected in expected_summary.items():
        require(summary.get(key) == expected, f"Summary mismatch for {key}")
    require(summary.get("source_revision_history") == [revision], "Source/resume revision drifted")
    require(str(summary.get("training_device", "")).startswith("cuda"), "Training did not run on CUDA")
    require(int(summary.get("cuda_peak_memory_allocated_bytes", 0)) > 0, "CUDA allocation was not recorded")
    require(int(summary.get("cuda_peak_memory_reserved_bytes", 0)) > 0, "CUDA reservation was not recorded")
    completed_epochs = int(summary.get("completed_epochs", 0))
    require(completed_epochs == len(history), "Summary/history epoch count mismatch")
    require(completed_epochs >= int(job["min_epochs"]), "Training stopped before minimum epochs")
    require(completed_epochs <= int(job["epochs"]), "Training exceeded maximum epochs")
    require(
        all(int(row.get("train_event_count", -1)) == int(dataset_row["expected_train_targets"]) for row in history),
        "A training epoch did not cover the full train target population",
    )
    selected = _earliest_minimum(history)
    require(int(summary.get("best_epoch", -1)) == int(selected["epoch"]), "Selected epoch is not earliest raw-RMSE minimum")
    require(
        math.isclose(
            float(summary.get("selected_metric_value", float("nan"))),
            float(summary.get("best_val_qty_rmse", float("nan"))),
            rel_tol=1e-10,
            abs_tol=1e-8,
        ),
        "Selected raw RMSE did not replay",
    )
    quantity_rows = summary.get("quantity_rows")
    require(isinstance(quantity_rows, list), "Quantity rows are missing")
    require(
        len(quantity_rows) == 5
        and all(isinstance(row, Mapping) for row in quantity_rows)
        and
        {row.get("stratum") for row in quantity_rows}
        == {"le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99"},
        "Quantity strata are incomplete",
    )
    require(
        sum(int(row["count"]) for row in quantity_rows)
        == int(dataset_row["expected_validation_targets"]),
        "Validation target count mismatch",
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint.get("checkpoint_monitor") == SELECTOR, "Selected checkpoint selector drifted")
    require(checkpoint.get("checkpoint_monitor_history_key") == HISTORY_KEY, "Selected checkpoint history key drifted")
    require(checkpoint.get("checkpoint_selection") == SELECTION, "Selected checkpoint route drifted")
    checkpoint_state = checkpoint.get("model_state_dict")
    require(isinstance(checkpoint_state, dict), "Selected checkpoint state is missing")
    checkpoint_digest = canonical_state_dict_sha256(checkpoint_state)
    require(checkpoint.get("model_state_sha256") == checkpoint_digest, "Selected checkpoint digest mismatch")
    require(summary.get("checkpoint_state_sha256") == checkpoint_digest, "Summary/checkpoint digest mismatch")
    require(checkpoint.get("evaluation_scope") == "validation_only", "Checkpoint scope drifted")
    require(checkpoint.get("held_out_test_evaluated") is False, "Checkpoint evaluated held-out test")
    recomputed_metrics = audit_selected_quantity_metrics(
        summary=summary,
        history=history,
        checkpoint=checkpoint,
        quantity_rows=quantity_rows,
        expected_count=int(dataset_row["expected_validation_targets"]),
    )
    raw_rmse = recomputed_metrics["raw_rmse"]
    overall_mae = recomputed_metrics["overall_mae"]

    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    resume_audit = audit_resume_state(
        resume=resume,
        checkpoint=checkpoint,
        summary=summary,
        history=history,
        job=job,
        dataset_row=dataset_row,
        revision=revision,
    )

    return {
        "schema_version": 1,
        "status": "passed",
        "job_id": job["job_id"],
        "phase": job["phase"],
        "host_role": job["host_role"],
        "dataset": job["dataset"],
        "backbone": job["backbone"],
        "seed": 42,
        "source_revision": revision,
        "checkpoint_monitor": SELECTOR,
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": completed_epochs,
        "metrics": {
            "overall_mae": overall_mae,
            "raw_rmse": raw_rmse,
            "body_mae": _weighted_body_mae(quantity_rows),
            "time_nll": float(summary["best_val_time_nll"]),
        },
        "train_target_count": int(dataset_row["expected_train_targets"]),
        "validation_target_count": int(dataset_row["expected_validation_targets"]),
        "checkpoint_state_sha256": checkpoint_digest,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "launch_gpu_preflight": runtime_fingerprint["launch_preflight"],
        **resume_audit,
        "artifact_sha256": {
            "launch_contract": sha256_file(launch_path),
            "summary": sha256_file(summary_path),
            "history": sha256_file(history_path),
            "selected_checkpoint": sha256_file(checkpoint_path),
            "resume_checkpoint": sha256_file(resume_path),
            "runtime_fingerprint": runtime_fingerprint_sha256,
        },
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def canonical_training_command(
    command: Sequence[str],
    *,
    host_role: str,
    project: Path,
    training_output: Path,
    dataset_row: Mapping[str, Any],
) -> list[str]:
    """Tokenize host-specific paths while preserving every semantic argument."""
    normalized = list(command)
    require(len(normalized) >= 2, "Training command is incomplete")
    require(
        Path(normalized[1]).resolve() == (project / TRAINER_REL).resolve(),
        "Training command uses an unexpected trainer",
    )
    normalized[0] = f"<host-python:{host_role}>"
    normalized[1] = f"<project>/{TRAINER_REL.as_posix()}"
    replacements = {
        "--data": (
            (project / str(dataset_row["data_path"])).resolve(),
            f"<project>/{dataset_row['data_path']}",
        ),
        "--split-manifest": (
            (project / str(dataset_row["split_manifest_path"])).resolve(),
            f"<project>/{dataset_row['split_manifest_path']}",
        ),
        "--output-dir": (training_output.resolve(), "<artifact-root>/jobs/<job-id>/training"),
    }
    for flag, (expected_path, token) in replacements.items():
        require(normalized.count(flag) == 1, f"Training command must contain one {flag}")
        index = normalized.index(flag) + 1
        require(Path(normalized[index]).resolve() == expected_path, f"Training command path drift: {flag}")
        normalized[index] = token
    require(
        not any(str(project.resolve()) in argument or str(training_output.resolve()) in argument for argument in normalized),
        "Canonical command retained an absolute project/artifact path",
    )
    return normalized


def _job_identity(
    *,
    contract_path: Path,
    manifest_path: Path,
    revision: str,
    project: Path,
    job: Mapping[str, Any],
    dataset_row: Mapping[str, Any],
    command: Sequence[str],
    training_output: Path,
) -> dict[str, Any]:
    canonical_command = canonical_training_command(
        command,
        host_role=str(job["host_role"]),
        project=project,
        training_output=training_output,
        dataset_row=dataset_row,
    )
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": sha256_file(contract_path),
        "source_revision": revision,
        "source_manifest_sha256": sha256_file(manifest_path),
        "path_identity": "portable_project_and_artifact_role_tokens_v1",
        "job": dict(job),
        "job_spec_sha256": canonical_sha256(job),
        "dataset_input": {
            "data_path": dataset_row["data_path"],
            "data_sha256": dataset_row["data_sha256"],
            "split_manifest_path": dataset_row["split_manifest_path"],
            "split_manifest_sha256": dataset_row["split_manifest_sha256"],
        },
        "command": canonical_command,
        "command_sha256": canonical_sha256(canonical_command),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def _write_receipt(
    path: Path,
    *,
    audit: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = {
        **dict(audit),
        "identity_sha256": canonical_sha256(identity),
        "contract_sha256": identity["contract_sha256"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "job_spec_sha256": identity["job_spec_sha256"],
        "audited_at": utc_now(),
    }
    save_json_atomic(path, receipt)
    return receipt


def _run_logged(
    command: Sequence[str],
    *,
    log_path: Path,
    project: Path,
    env: Mapping[str, str],
    child_slot: dict[str, subprocess.Popen[str] | None],
    lock_fds: Sequence[int],
) -> None:
    inherited_locks = tuple(int(fd) for fd in lock_fds)
    require(
        len(inherited_locks) == 2
        and len(set(inherited_locks)) == 2
        and all(fd >= 0 for fd in inherited_locks),
        "Exactly two distinct lock file descriptors must be inherited",
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] launch command_sha256={canonical_sha256(list(command))}\n")
        handle.flush()
        child = subprocess.Popen(
            list(command),
            cwd=project,
            env=dict(env),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            pass_fds=inherited_locks,
        )
        child_slot["child"] = child
        code = child.wait()
        child_slot["child"] = None
    if code:
        raise subprocess.CalledProcessError(code, list(command))


def _cleanup_wrapper_temporary_files(job_dir: Path) -> None:
    """Remove only fixed atomic-write leftovers while the job lock is held."""
    for relative in sorted(WRAPPER_TEMP_FILENAMES):
        path = job_dir / relative
        if path.exists():
            require(path.is_file() and not path.is_symlink(), f"Unsafe wrapper temporary path: {path}")
            path.unlink()


def _close_parent_lock_reference(lock: Any) -> None:
    """Close without LOCK_UN so an inherited child descriptor keeps the flock."""
    lock.close()


def _fresh_schedule_snapshot(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Re-audit the fixed two-host registry immediately before one launch."""
    from paper.scripts import control_raw_rmse_baseline_alignment as controller

    return controller.build_snapshot(argparse.Namespace(
        project_root=args.project_root,
        contract=args.contract,
        source_revision=args.source_revision,
        python_5080=args.python_5080,
        python_5090=args.python_5090,
    ))


def execute_job(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve()
    contract_path = args.contract.resolve()
    contract = read_json(contract_path)
    validated = validate_contract(contract)
    jobs = validated["jobs"]
    require(args.job_id in jobs, f"Job is not in the frozen schedule: {args.job_id}")
    job = jobs[args.job_id]
    require(job["host_role"] == args.host_role, "Job is assigned to a different host")
    paths = fixed_execution_paths(args.source_revision)
    artifact_roots = {
        host: path.resolve()
        for host, path in paths["artifact_roots"].items()
    }
    output_root = artifact_roots[args.host_role]
    protected_roots = [*artifact_roots.values(), paths["coordination_root"].resolve()]
    for protected_root in protected_roots:
        require(
            project != protected_root
            and project not in protected_root.parents
            and protected_root not in project.parents,
            "Runtime registry and packaged source tree must be disjoint",
        )
    snapshot, freshly_validated, _ = _fresh_schedule_snapshot(args)
    require(freshly_validated == validated, "Fresh schedule contract validation drifted")
    require(
        snapshot.get("next_job_by_host", {}).get(args.host_role) == args.job_id,
        "Fresh audited schedule does not authorize this job as the current next job",
    )
    require(snapshot.get("held_out_test_evaluated") is False, "Held-out state is not locked")
    require(snapshot.get("additional_seeds_executed") is False, "Additional seed state drifted")
    dataset_row = freshly_validated["datasets"][job["dataset"]]
    training_output = output_root / "jobs" / args.job_id / "training"
    job_dir = training_output.parent
    python_by_host = {"5080": args.python_5080, "5090": args.python_5090}
    python = python_by_host[args.host_role]
    command = training_command(
        python=python,
        project=project,
        training_output=training_output,
        revision=args.source_revision,
        dataset_row=dataset_row,
        job=job,
    )
    identity = _job_identity(
        contract_path=contract_path,
        manifest_path=project / "source_manifest.json",
        revision=args.source_revision,
        project=project,
        job=job,
        dataset_row=dataset_row,
        command=command,
        training_output=training_output,
    )
    if args.verify_only:
        return {
            "status": "verified",
            "job_id": args.job_id,
            "host_role": args.host_role,
            "identity_sha256": canonical_sha256(identity),
            "held_out_test_evaluated": False,
        }

    job_dir.mkdir(parents=True, exist_ok=True)
    lock_path = job_dir / "job.lock"
    identity_path = job_dir / "job_identity.json"
    status_path = job_dir / "job_status.json"
    receipt_path = job_dir / "audit_receipt.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Job is already running: {args.job_id}") from exc
        _cleanup_wrapper_temporary_files(job_dir)
        if identity_path.exists():
            saved_identity = read_json(identity_path)
            require(saved_identity == identity, "Existing job identity drifted; refusing resume")
        else:
            save_json_atomic(identity_path, identity)

        # Recover the narrow crash window after trainer completion but before
        # receipt/status publication without requiring an idle GPU preflight.
        launch_path = training_output / "launch_contract.json"
        if launch_path.is_file() and read_json(launch_path).get("status") == "complete":
            audit = audit_training_artifact(
                training_output=training_output,
                job=job,
                dataset_row=dataset_row,
                contract=contract,
                revision=args.source_revision,
            )
            receipt = _write_receipt(receipt_path, audit=audit, identity=identity)
            save_json_atomic(status_path, {
                "schema_version": 1,
                "status": "complete",
                "job_id": args.job_id,
                "host_role": args.host_role,
                "receipt_sha256": sha256_file(receipt_path),
                "metrics": receipt["metrics"],
                "held_out_test_evaluated": False,
                "updated_at": utc_now(),
            })
            return {**receipt, "crash_recovery": True}

        previous = read_json(status_path) if status_path.exists() else {}
        attempt = int(previous.get("attempt", 0)) + 1
        gpu_lock_path = GPU_LOCK_PATH.resolve()
        require(
            project not in gpu_lock_path.parents and output_root not in gpu_lock_path.parents,
            "Host GPU lock must remain outside source and artifact trees",
        )
        gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
        gpu_lock = gpu_lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            _close_parent_lock_reference(gpu_lock)
            raise RuntimeError(f"Host GPU is locked by another baseline job: {gpu_lock_path}") from exc
        try:
            preflight = gpu_preflight(
                contract,
                args.host_role,
                backbone=str(job["backbone"]),
            )
            runtime_fingerprint = collect_runtime_fingerprint(
                python=python,
                project=project,
                contract=contract,
                job=job,
                preflight=preflight,
            )
            runtime_fingerprint_path = job_dir / RUNTIME_FINGERPRINT_FILENAME
            if runtime_fingerprint_path.exists():
                previous_runtime = read_json(runtime_fingerprint_path)
                validate_runtime_fingerprint(previous_runtime, contract=contract, job=job)
                require(
                    previous_runtime["runtime"] == runtime_fingerprint["runtime"],
                    "Runtime fingerprint changed across resume attempts",
                )
            save_json_atomic(runtime_fingerprint_path, runtime_fingerprint)
            save_json_atomic(status_path, {
                "schema_version": 1,
                "status": "running",
                "job_id": args.job_id,
                "host_role": args.host_role,
                "attempt": attempt,
                "resume_checkpoint_present": (
                    training_output / "runs" / job["backbone"] / VARIANT / "seed_42" / "last_epoch_state.pt"
                ).is_file(),
                "gpu_preflight": preflight,
                "runtime_fingerprint_sha256": sha256_file(runtime_fingerprint_path),
                "gpu_lock_path": str(gpu_lock_path),
                "identity_sha256": canonical_sha256(identity),
                "fresh_schedule_snapshot_sha256": canonical_sha256(snapshot),
                "held_out_test_evaluated": False,
                "updated_at": utc_now(),
            })
        except BaseException:
            _close_parent_lock_reference(gpu_lock)
            raise

        child_slot: dict[str, subprocess.Popen[str] | None] = {"child": None}
        previous_handlers: dict[int, Any] = {}

        def stop(signum: int, _frame: Any) -> None:
            child = child_slot["child"]
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signum)
            raise InterruptedError(f"Launcher received signal {signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, stop)
        try:
            _run_logged(
                command,
                log_path=job_dir / "launcher.log",
                project=project,
                env=deterministic_env(contract, project),
                child_slot=child_slot,
                lock_fds=(lock.fileno(), gpu_lock.fileno()),
            )
            audit = audit_training_artifact(
                training_output=training_output,
                job=job,
                dataset_row=dataset_row,
                contract=contract,
                revision=args.source_revision,
            )
            receipt = _write_receipt(receipt_path, audit=audit, identity=identity)
            save_json_atomic(status_path, {
                "schema_version": 1,
                "status": "complete",
                "job_id": args.job_id,
                "host_role": args.host_role,
                "attempt": attempt,
                "receipt_sha256": sha256_file(receipt_path),
                "metrics": receipt["metrics"],
                "held_out_test_evaluated": False,
                "updated_at": utc_now(),
            })
            return {**receipt, "crash_recovery": False}
        except BaseException as exc:
            resume_path = training_output / "runs" / job["backbone"] / VARIANT / "seed_42" / "last_epoch_state.pt"
            save_json_atomic(status_path, {
                "schema_version": 1,
                "status": "failed",
                "job_id": args.job_id,
                "host_role": args.host_role,
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "exact_resume_checkpoint_present": resume_path.is_file(),
                "held_out_test_evaluated": False,
                "updated_at": utc_now(),
            })
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            _close_parent_lock_reference(gpu_lock)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / CONTRACT_REL)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--host-role", choices=HOST_ROLES, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--python-5080",
        default="/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python",
    )
    parser.add_argument(
        "--python-5090",
        default="/opt/miniconda3/envs/ai_env/bin/python",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = execute_job(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
