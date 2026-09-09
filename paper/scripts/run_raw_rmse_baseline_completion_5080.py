#!/usr/bin/env python3
"""Run the four missing selector-aligned seed42 baselines on one RTX 5080.

This is a narrow completion queue. It reuses the frozen baseline trainer and
artifact auditor, preserves the failed Instacart gate, and never opens test data.
"""

from __future__ import annotations

import argparse
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

from paper.scripts import run_raw_rmse_baseline_alignment_job as alignment


CONTRACT_REL = Path(
    "paper/contracts/raw_rmse_baseline_completion_seed42_5080_v1.json"
)
RUNNER_REL = Path("paper/scripts/run_raw_rmse_baseline_completion_5080.py")
SMOKE_REL = Path("paper/scripts/run_count_aware_cuda_model_test.py")
CONTRACT_ID = "raw_rmse_baseline_completion_seed42_5080_v1"
HOST_ROLE = "5080"
BACKBONES = ("rmtpp", "thp")
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly")
EXECUTION_ROOT_BASE = Path(
    "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts"
)
GPU_LOCK = Path("/tmp/paper_research_raw_rmse_baseline_completion_5080_gpu0.lock")
QUEUE_LOCK = "queue.lock"
QUEUE_STATUS = "queue_status.json"
SOURCE_MANIFEST = "source_manifest.json"
PYTHON_5080 = "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python"

REQUIRED_SOURCE_FILES = (
    CONTRACT_REL,
    Path("paper/contracts/raw_rmse_baseline_completion_seed42_5080_v1.md"),
    RUNNER_REL,
    SMOKE_REL,
    Path("paper/contracts/raw_rmse_baseline_alignment_seed42_v1.json"),
    Path("paper/scripts/run_raw_rmse_baseline_alignment_job.py"),
    Path("paper/scripts/run_count_aware_tpp_backbone_control.py"),
    Path("paper/scripts/count_aware_tpp_backbone/constants.py"),
    Path("paper/scripts/count_aware_tpp_backbone/datasets.py"),
    Path("paper/scripts/count_aware_tpp_backbone/core.py"),
    Path("paper/scripts/count_aware_tpp_backbone/training.py"),
    Path("paper/scripts/count_aware_tpp_backbone/reporting.py"),
    Path("models/TPPs/CountAwareTPP.py"),
    Path("models/TPPs/CountAwareFactory.py"),
    Path("data_loader/event_seq_data_module.py"),
    Path("simple_lab_test/search/tests/test_raw_rmse_baseline_completion_5080.py"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execution_root(revision: str) -> Path:
    require(
        len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "A full lowercase source revision is required",
    )
    return EXECUTION_ROOT_BASE / f"{CONTRACT_ID}_{revision}"


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    require(contract.get("schema_version") == 1, "Unsupported contract schema")
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong completion contract")
    scope = contract.get("scope")
    require(isinstance(scope, Mapping), "Scope is missing")
    expected_scope = {
        "baseline_backbones": list(BACKBONES),
        "quantity_objective": "unweighted MSE on log1p quantity",
        "quantity_variant": "count_only_log_regression",
        "checkpoint_monitor": "validation_raw_quantity_rmse",
        "checkpoint_rule": "earliest strict finite minimum validation raw quantity RMSE",
        "early_stopping_monitor": "validation_raw_quantity_rmse",
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
    }
    require(dict(scope) == expected_scope, "Completion training scope drifted")

    rows = contract.get("datasets")
    require(isinstance(rows, list) and len(rows) == 2, "Exactly two datasets are required")
    require(tuple(row.get("dataset") for row in rows) == DATASETS, "Dataset order drifted")
    datasets: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row["dataset"])
        require(int(row.get("expected_train_targets", 0)) > 0, f"Train count missing: {name}")
        require(int(row.get("expected_validation_targets", 0)) > 0, f"Validation count missing: {name}")
        require(int(row.get("lookback", 0)) > 0, f"Lookback missing: {name}")
        require(int(row.get("max_sequence_length", 0)) > 0, f"Sequence length missing: {name}")
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
            value = str(row.get(key, ""))
            require(len(value) == 64 and all(c in "0123456789abcdef" for c in value), f"Invalid {key}: {name}")
        b_metrics = row.get("B_t0_raw_rmse_validation_metrics")
        require(isinstance(b_metrics, Mapping), f"B metrics missing: {name}")
        require(all(math.isfinite(float(b_metrics[key])) for key in ("raw_rmse", "overall_mae")), f"B metrics invalid: {name}")
        datasets[name] = dict(row)

    jobs = contract.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 4, "Exactly four jobs are required")
    expected_pairs = [
        ("intermittent_frozen_5000", "rmtpp"),
        ("intermittent_frozen_5000", "thp"),
        ("yellow_trip_hourly", "rmtpp"),
        ("yellow_trip_hourly", "thp"),
    ]
    require(
        [(job.get("dataset"), job.get("backbone")) for job in jobs] == expected_pairs,
        "Completion queue order drifted",
    )
    expected_ids = [
        f"completion_{index:02d}_{dataset}_{backbone}_seed42"
        for index, (dataset, backbone) in enumerate(expected_pairs, start=1)
    ]
    require([job.get("job_id") for job in jobs] == expected_ids, "Completion job IDs drifted")
    for job in jobs:
        require(job.get("phase") == "seed42_e300", "Completion phase drifted")
        require(job.get("host_role") == HOST_ROLE, "A job escaped RTX 5080")
        require(job.get("seed") == 42, "Only seed42 is authorized")
        require(
            (job.get("epochs"), job.get("min_epochs"), job.get("patience"))
            == (300, 40, 40),
            "Epoch or early-stopping contract drifted",
        )

    prior = contract.get("prior_campaign")
    require(isinstance(prior, Mapping), "Prior campaign binding is missing")
    require(prior.get("contract_id") == "raw_rmse_baseline_alignment_seed42_v1", "Prior contract drifted")
    require(prior.get("source_revision") == "9440609673cfdf7bcfec5c77d7f677affd04ae49", "Prior source drifted")
    require(prior.get("required_status") == "stopped_gate_failed", "Prior failure status drifted")
    require(prior.get("required_gate_status") == "failed", "Prior gate status drifted")
    require(prior.get("instacart_results_are_preserved_as_failure") is True, "Instacart failure was not preserved")
    require(prior.get("reuse_or_retrain_prior_jobs") is False, "Prior jobs must not be rerun")
    completed = prior.get("required_completed_job_ids")
    require(isinstance(completed, list) and len(completed) == 8, "Prior completed registry drifted")

    runtime = contract.get("runtime_policy", {}).get("hosts", {}).get(HOST_ROLE)
    require(isinstance(runtime, Mapping), "RTX 5080 runtime policy is missing")
    require(runtime.get("required_gpu_name_contains") == "RTX 5080", "GPU model drifted")
    require(runtime.get("minimum_free_vram_mib") == 12000, "VRAM threshold drifted")
    require(runtime.get("permitted_backbones") == list(BACKBONES), "5080 backbone permission drifted")
    require(runtime.get("active_gdm_permitted_backbones") == list(BACKBONES), "Active-GDM authorization drifted")

    execution = contract.get("execution")
    require(isinstance(execution, Mapping), "Execution policy is missing")
    require(execution.get("host") == HOST_ROLE, "Execution host drifted")
    for key in ("queue_order_is_strict", "stop_on_first_failure", "new_output_registry", "resume_exact_identity_only"):
        require(execution.get(key) is True, f"Execution safety drifted: {key}")
    for key in ("overwrite", "additional_seeds", "held_out_test", "retrain_instacart"):
        require(execution.get(key) is False, f"Forbidden execution enabled: {key}")
    held_out = contract.get("held_out_lock")
    require(isinstance(held_out, Mapping), "Held-out lock is missing")
    for key in ("test_target_loader_allowed", "test_metrics_allowed", "test_artifacts_allowed", "held_out_decision_use_allowed"):
        require(held_out.get(key) is False, f"Held-out access enabled: {key}")

    frozen = contract.get("frozen_learning_files_sha256")
    require(isinstance(frozen, Mapping) and len(frozen) == 9, "Frozen learning-file set drifted")
    return {"jobs": [dict(job) for job in jobs], "datasets": datasets}


def verify_frozen_learning_files(project: Path, contract: Mapping[str, Any]) -> None:
    for relative, expected in contract["frozen_learning_files_sha256"].items():
        path = project / str(relative)
        require(path.is_file(), f"Frozen learning file is missing: {relative}")
        require(sha256_file(path) == expected, f"Frozen learning file changed: {relative}")


def verify_source_manifest(project: Path, contract: Mapping[str, Any], revision: str) -> dict[str, Any]:
    path = project / SOURCE_MANIFEST
    manifest = read_json(path)
    require(manifest.get("schema_version") == 1, "Source manifest schema drifted")
    require(manifest.get("source_revision") == revision, "Source manifest revision drifted")
    require(manifest.get("contract") == CONTRACT_REL.as_posix(), "Source manifest contract drifted")
    require(manifest.get("contract_sha256") == sha256_file(project / CONTRACT_REL), "Source contract digest drifted")
    require(manifest.get("evaluation_scope") == "validation_only", "Source scope drifted")
    require(manifest.get("held_out_test_evaluated") is False, "Source manifest opened held-out test")
    required = manifest.get("required_source_files")
    require(required == [value.as_posix() for value in REQUIRED_SOURCE_FILES], "Required source inventory drifted")
    files = manifest.get("files")
    require(isinstance(files, Mapping), "Source file map is missing")
    actual = {
        child.relative_to(project).as_posix(): sha256_file(child)
        for child in sorted(project.rglob("*"))
        if child.is_file() and child != path
    }
    require(dict(files) == actual, "Packaged source inventory or digest drifted")
    return manifest


def verify_dataset_inputs(project: Path, datasets: Mapping[str, Mapping[str, Any]]) -> None:
    alignment.verify_dataset_inputs(project, datasets)


def audit_prior_campaign(contract: Mapping[str, Any]) -> dict[str, Any]:
    prior = contract["prior_campaign"]
    root = Path(str(prior["registry_root"]))
    status_path = root / "coordination/controller_5080_status.json"
    require(status_path.is_file(), "Prior campaign status is missing")
    status = read_json(status_path)
    require(status.get("contract_id") == prior["contract_id"], "Prior campaign contract drifted")
    require(status.get("source_revision") == prior["source_revision"], "Prior campaign source drifted")
    require(status.get("status") == prior["required_status"], "Prior campaign is not the preserved failed gate")
    require(status.get("held_out_test_evaluated") is False, "Prior campaign opened held-out test")
    require(status.get("additional_seeds_executed") is False, "Prior campaign ran extra seeds")
    require(status.get("instacart_gate", {}).get("status") == prior["required_gate_status"], "Prior Instacart failure drifted")
    expected_ids = list(prior["required_completed_job_ids"])
    require(status.get("completed_job_ids") == expected_ids, "Prior completed jobs drifted")
    receipts: dict[str, str] = {}
    for job_id in expected_ids:
        host = "5080" if job_id.endswith("_rmtpp") and "e300" in job_id else "5090"
        receipt_path = root / f"artifacts/{host}/jobs/{job_id}/audit_receipt.json"
        require(receipt_path.is_file(), f"Prior receipt is missing: {job_id}")
        receipt = read_json(receipt_path)
        require(receipt.get("status") == "passed", f"Prior receipt did not pass: {job_id}")
        require(receipt.get("source_revision") == prior["source_revision"], f"Prior receipt source drifted: {job_id}")
        require(receipt.get("held_out_test_evaluated") is False, f"Prior receipt opened held-out test: {job_id}")
        receipts[job_id] = sha256_file(receipt_path)
    return {
        "status_sha256": sha256_file(status_path),
        "gate_status": status["instacart_gate"]["status"],
        "completed_job_count": len(expected_ids),
        "receipt_sha256": receipts,
    }


def audit_cuda_smoke(path: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    require(payload.get("status") == "complete", "CUDA model smoke did not complete")
    require(payload.get("device") == "cuda", "CUDA model smoke used the wrong device")
    require("RTX 5080" in str(payload.get("cuda_device", "")), "CUDA model smoke used the wrong GPU")
    rows = payload.get("results")
    require(isinstance(rows, list), "CUDA model smoke results are missing")
    observed = {
        (row.get("backbone"), row.get("variant"))
        for row in rows
        if row.get("finite") is True
    }
    expected = {
        (row["backbone"], row["variant"])
        for row in contract["cuda_smoke"]["required_cases"]
    }
    require(expected.issubset(observed), "RMTPP/THP CUDA forward-backward smoke is incomplete")
    return payload


def run_cuda_smoke(
    *,
    project: Path,
    root: Path,
    contract: Mapping[str, Any],
    revision: str,
    python: str,
) -> dict[str, Any]:
    smoke_root = root / "cuda_smoke"
    output = smoke_root / "model_test.json"
    receipt_path = smoke_root / "receipt.json"
    if receipt_path.is_file():
        receipt = read_json(receipt_path)
        payload = audit_cuda_smoke(output, contract)
        require(receipt.get("source_revision") == revision, "CUDA smoke source drifted")
        require(receipt.get("model_test_sha256") == sha256_file(output), "CUDA smoke artifact drifted")
        return {**payload, "reused": True}
    require(not output.exists(), "Unreceipted CUDA smoke output exists")
    smoke_root.mkdir(parents=True, exist_ok=True)
    preflight = alignment.gpu_preflight(contract, HOST_ROLE, backbone="thp")
    command = [python, str(project / SMOKE_REL), "--device", "cuda", "--output", str(output)]
    result = subprocess.run(
        command,
        cwd=project,
        env=alignment.deterministic_env(contract, project),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    (smoke_root / "stdout.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    require(result.returncode == 0, f"CUDA model smoke failed: {result.returncode}")
    payload = audit_cuda_smoke(output, contract)
    save_json(receipt_path, {
        "schema_version": 1,
        "status": "passed",
        "source_revision": revision,
        "script_sha256": sha256_file(project / SMOKE_REL),
        "model_test_sha256": sha256_file(output),
        "stdout_sha256": sha256_file(smoke_root / "stdout.log"),
        "gpu_preflight": preflight,
        "held_out_test_evaluated": False,
        "audited_at": alignment.utc_now(),
    })
    return {**payload, "reused": False}


def training_command(
    *,
    python: str,
    project: Path,
    root: Path,
    revision: str,
    job: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> list[str]:
    return alignment.training_command(
        python=python,
        project=project,
        training_output=root / "jobs" / str(job["job_id"]) / "training",
        revision=revision,
        dataset_row=dataset,
        job=job,
    )


def job_identity(
    *,
    project: Path,
    root: Path,
    contract_path: Path,
    revision: str,
    job: Mapping[str, Any],
    dataset: Mapping[str, Any],
    command: Sequence[str],
) -> dict[str, Any]:
    training_output = root / "jobs" / str(job["job_id"]) / "training"
    canonical = alignment.canonical_training_command(
        command,
        host_role=HOST_ROLE,
        project=project,
        training_output=training_output,
        dataset_row=dataset,
    )
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": sha256_file(contract_path),
        "source_revision": revision,
        "source_manifest_sha256": sha256_file(project / SOURCE_MANIFEST),
        "path_identity": "portable_project_and_artifact_role_tokens_v1",
        "job": dict(job),
        "job_spec_sha256": alignment.canonical_sha256(job),
        "dataset_input": {
            "data_path": dataset["data_path"],
            "data_sha256": dataset["data_sha256"],
            "split_manifest_path": dataset["split_manifest_path"],
            "split_manifest_sha256": dataset["split_manifest_sha256"],
        },
        "command": canonical,
        "command_sha256": alignment.canonical_sha256(canonical),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def audit_completed_job(
    *,
    project: Path,
    root: Path,
    contract: Mapping[str, Any],
    revision: str,
    job: Mapping[str, Any],
    dataset: Mapping[str, Any],
    python: str,
) -> dict[str, Any] | None:
    job_dir = root / "jobs" / str(job["job_id"])
    receipt_path = job_dir / "audit_receipt.json"
    if not receipt_path.is_file():
        if job_dir.exists():
            alignment.audit_incomplete_job_layout(job_dir=job_dir, job=job)
        return None
    command = training_command(
        python=python, project=project, root=root, revision=revision, job=job, dataset=dataset
    )
    expected_identity = job_identity(
        project=project,
        root=root,
        contract_path=project / CONTRACT_REL,
        revision=revision,
        job=job,
        dataset=dataset,
        command=command,
    )
    saved_identity = read_json(job_dir / "job_identity.json")
    require(saved_identity == expected_identity, f"Completed job identity drifted: {job['job_id']}")
    fresh = alignment.audit_training_artifact(
        training_output=job_dir / "training",
        job=job,
        dataset_row=dataset,
        contract=contract,
        revision=revision,
        require_receipt=True,
    )
    receipt = read_json(receipt_path)
    for key, value in fresh.items():
        require(receipt.get(key) == value, f"Completed receipt drifted for {job['job_id']}: {key}")
    require(receipt.get("identity_sha256") == alignment.canonical_sha256(expected_identity), "Receipt identity drifted")
    return receipt


def snapshot(
    *,
    project: Path,
    root: Path,
    contract: Mapping[str, Any],
    validated: Mapping[str, Any],
    revision: str,
    python: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    completed: dict[str, dict[str, Any]] = {}
    missing_seen = False
    next_job: str | None = None
    for job in validated["jobs"]:
        receipt = audit_completed_job(
            project=project,
            root=root,
            contract=contract,
            revision=revision,
            job=job,
            dataset=validated["datasets"][job["dataset"]],
            python=python,
        )
        if receipt is None:
            missing_seen = True
            if next_job is None:
                next_job = str(job["job_id"])
        else:
            require(not missing_seen, "Completion artifacts are not a strict queue prefix")
            completed[str(job["job_id"])] = receipt
    return ({
        "schema_version": 1,
        "status": "complete" if next_job is None else "pending",
        "contract_id": CONTRACT_ID,
        "source_revision": revision,
        "host_role": HOST_ROLE,
        "completed_job_ids": list(completed),
        "next_job_id": next_job,
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
        "updated_at": alignment.utc_now(),
    }, completed)


def run_logged(command: Sequence[str], *, log_path: Path, project: Path, env: Mapping[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child: subprocess.Popen[str] | None = None
    previous: dict[int, Any] = {}

    def stop(signum: int, _frame: Any) -> None:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)

    signals = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    for signum in signals:
        previous[signum] = signal.signal(signum, stop)
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{alignment.utc_now()}] launching trainer\n")
            handle.flush()
            child = subprocess.Popen(
                list(command),
                cwd=project,
                env=dict(env),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            code = child.wait()
        if code:
            raise subprocess.CalledProcessError(code, list(command))
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_one(
    *,
    project: Path,
    root: Path,
    contract: Mapping[str, Any],
    revision: str,
    job: Mapping[str, Any],
    dataset: Mapping[str, Any],
    python: str,
) -> dict[str, Any]:
    job_dir = root / "jobs" / str(job["job_id"])
    training_output = job_dir / "training"
    job_dir.mkdir(parents=True, exist_ok=True)
    command = training_command(
        python=python, project=project, root=root, revision=revision, job=job, dataset=dataset
    )
    identity = job_identity(
        project=project,
        root=root,
        contract_path=project / CONTRACT_REL,
        revision=revision,
        job=job,
        dataset=dataset,
        command=command,
    )
    identity_path = job_dir / "job_identity.json"
    if identity_path.exists():
        require(read_json(identity_path) == identity, "Existing job identity drifted; refusing resume")
    else:
        save_json(identity_path, identity)
    (job_dir / "job.lock").touch(exist_ok=True)
    status_path = job_dir / "job_status.json"
    previous_status = read_json(status_path) if status_path.exists() else {}
    attempt = int(previous_status.get("attempt", 0)) + 1
    preflight = alignment.gpu_preflight(contract, HOST_ROLE, backbone=str(job["backbone"]))
    runtime = alignment.collect_runtime_fingerprint(
        python=python,
        project=project,
        contract=contract,
        job=job,
        preflight=preflight,
    )
    runtime_path = job_dir / alignment.RUNTIME_FINGERPRINT_FILENAME
    if runtime_path.exists():
        previous_runtime = read_json(runtime_path)
        alignment.validate_runtime_fingerprint(previous_runtime, contract=contract, job=job)
        require(previous_runtime["runtime"] == runtime["runtime"], "Runtime changed across resume")
    save_json(runtime_path, runtime)
    resume_path = training_output / f"runs/{job['backbone']}/{alignment.VARIANT}/seed_42/last_epoch_state.pt"
    save_json(status_path, {
        "schema_version": 1,
        "status": "running",
        "job_id": job["job_id"],
        "host_role": HOST_ROLE,
        "attempt": attempt,
        "resume_checkpoint_present": resume_path.is_file(),
        "gpu_preflight": preflight,
        "identity_sha256": alignment.canonical_sha256(identity),
        "held_out_test_evaluated": False,
        "started_at": alignment.utc_now(),
    })
    try:
        run_logged(
            command,
            log_path=job_dir / "launcher.log",
            project=project,
            env=alignment.deterministic_env(contract, project),
        )
        save_json(status_path, {
            "schema_version": 1,
            "status": "auditing",
            "job_id": job["job_id"],
            "host_role": HOST_ROLE,
            "attempt": attempt,
            "held_out_test_evaluated": False,
            "updated_at": alignment.utc_now(),
        })
        audit = alignment.audit_training_artifact(
            training_output=training_output,
            job=job,
            dataset_row=dataset,
            contract=contract,
            revision=revision,
        )
        receipt = alignment._write_receipt(job_dir / "audit_receipt.json", audit=audit, identity=identity)
        save_json(status_path, {
            "schema_version": 1,
            "status": "complete",
            "job_id": job["job_id"],
            "host_role": HOST_ROLE,
            "attempt": attempt,
            "receipt_sha256": sha256_file(job_dir / "audit_receipt.json"),
            "metrics": receipt["metrics"],
            "held_out_test_evaluated": False,
            "completed_at": alignment.utc_now(),
        })
        return receipt
    except BaseException as exc:
        save_json(status_path, {
            "schema_version": 1,
            "status": "failed",
            "job_id": job["job_id"],
            "host_role": HOST_ROLE,
            "attempt": attempt,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "exact_resume_checkpoint_present": resume_path.is_file(),
            "held_out_test_evaluated": False,
            "updated_at": alignment.utc_now(),
        })
        raise


def execute(args: argparse.Namespace) -> dict[str, Any]:
    project = args.project_root.resolve()
    contract_path = (project / CONTRACT_REL).resolve()
    contract = read_json(contract_path)
    validated = validate_contract(contract)
    verify_frozen_learning_files(project, contract)
    verify_source_manifest(project, contract, args.source_revision)
    verify_dataset_inputs(project, validated["datasets"])
    prior_audit = audit_prior_campaign(contract)
    root = execution_root(args.source_revision)
    require(project != root and project not in root.parents and root not in project.parents, "Source and output roots overlap")
    root.mkdir(parents=True, exist_ok=True)
    status, completed = snapshot(
        project=project,
        root=root,
        contract=contract,
        validated=validated,
        revision=args.source_revision,
        python=args.python,
    )
    status["prior_campaign_audit"] = prior_audit
    save_json(root / QUEUE_STATUS, status)
    if args.action == "status" or status["status"] == "complete":
        return status

    lock_path = root / QUEUE_LOCK
    with lock_path.open("a+", encoding="utf-8") as queue_lock:
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {**status, "status": "queue_busy"}
        with GPU_LOCK.open("a+", encoding="utf-8") as gpu_lock:
            try:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"RTX 5080 completion GPU lock is busy: {GPU_LOCK}") from exc
            smoke = run_cuda_smoke(
                project=project,
                root=root,
                contract=contract,
                revision=args.source_revision,
                python=args.python,
            )
            save_json(root / QUEUE_STATUS, {
                **status,
                "status": "running",
                "cuda_smoke": {
                    "status": "passed",
                    "device": smoke["cuda_device"],
                    "reused": smoke["reused"],
                },
                "updated_at": alignment.utc_now(),
            })
            try:
                for job in validated["jobs"]:
                    job_id = str(job["job_id"])
                    if job_id in completed:
                        continue
                    save_json(root / QUEUE_STATUS, {
                        **status,
                        "status": "running",
                        "running_job_id": job_id,
                        "completed_job_ids": list(completed),
                        "cuda_smoke": {"status": "passed", "device": smoke["cuda_device"]},
                        "updated_at": alignment.utc_now(),
                    })
                    completed[job_id] = run_one(
                        project=project,
                        root=root,
                        contract=contract,
                        revision=args.source_revision,
                        job=job,
                        dataset=validated["datasets"][job["dataset"]],
                        python=args.python,
                    )
                final, _ = snapshot(
                    project=project,
                    root=root,
                    contract=contract,
                    validated=validated,
                    revision=args.source_revision,
                    python=args.python,
                )
                final["prior_campaign_audit"] = prior_audit
                final["cuda_smoke"] = {"status": "passed", "device": smoke["cuda_device"]}
                save_json(root / QUEUE_STATUS, final)
                return final
            except BaseException as exc:
                failed = {
                    "schema_version": 1,
                    "status": "failed",
                    "contract_id": CONTRACT_ID,
                    "source_revision": args.source_revision,
                    "host_role": HOST_ROLE,
                    "completed_job_ids": list(completed),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "held_out_test_evaluated": False,
                    "additional_seeds_executed": False,
                    "updated_at": alignment.utc_now(),
                }
                save_json(root / QUEUE_STATUS, failed)
                raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", default=PYTHON_5080)
    parser.add_argument("--action", choices=("status", "run-queue"), default="status")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(execute(parse_args(argv)), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
