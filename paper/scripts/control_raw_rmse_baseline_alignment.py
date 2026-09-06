#!/usr/bin/env python3
"""Advance the two-host raw-RMSE baseline schedule by at most one local job.

Both fixed host artifact roots must be visible (or mirrored) when this
controller is called.  It derives the next job exclusively from freshly
audited receipts, so an hourly invocation cannot redirect itself to an empty
registry or rely on a generic process-name search.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.run_raw_rmse_baseline_alignment_job import (
    CONTRACT_REL,
    HOST_ROLES,
    _job_identity,
    audit_incomplete_job_layout,
    audit_training_artifact,
    canonical_sha256,
    execute_job,
    fixed_execution_paths,
    read_json,
    require,
    save_json_atomic,
    sha256_file,
    training_command,
    utc_now,
    validate_contract,
    validate_b_reference_evidence,
    validate_runtime_fingerprint,
    verify_dataset_inputs,
    verify_source_manifest,
)


FRESH_AUDIT_RECEIPT_FIELDS = (
    "status",
    "job_id",
    "phase",
    "host_role",
    "dataset",
    "backbone",
    "seed",
    "source_revision",
    "checkpoint_monitor",
    "best_epoch",
    "completed_epochs",
    "metrics",
    "train_target_count",
    "validation_target_count",
    "checkpoint_state_sha256",
    "runtime_fingerprint",
    "runtime_fingerprint_sha256",
    "launch_gpu_preflight",
    "resume_identity_sha256",
    "resume_current_state_sha256",
    "resume_best_state_sha256",
    "artifact_sha256",
    "evaluation_scope",
    "held_out_test_evaluated",
)


def require_receipt_matches_fresh_audit(
    *,
    job_id: str,
    receipt: Mapping[str, Any],
    fresh: Mapping[str, Any],
) -> None:
    for key in FRESH_AUDIT_RECEIPT_FIELDS:
        require(
            key in receipt and key in fresh,
            f"Receipt/fresh audit field is missing for {job_id}: {key}",
        )
        require(
            receipt.get(key) == fresh.get(key),
            f"Stale/tampered receipt for {job_id}: {key}",
        )


def _job_groups(contract: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    e1 = list(contract["schedule"]["e1_jobs"])
    full = list(contract["schedule"]["seed42_e300_jobs"])
    return e1, full[:2], full[2:]


def _validated_receipt_identity(job: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    expected = {
        "status": "passed",
        "job_id": job["job_id"],
        "phase": job["phase"],
        "host_role": job["host_role"],
        "dataset": job["dataset"],
        "backbone": job["backbone"],
        "seed": 42,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for key, value in expected.items():
        require(receipt.get(key) == value, f"Receipt identity mismatch for {job['job_id']}: {key}")
    metrics = receipt.get("metrics")
    require(isinstance(metrics, Mapping), f"Receipt metrics are missing: {job['job_id']}")
    for metric in ("overall_mae", "raw_rmse"):
        value = float(metrics.get(metric, float("nan")))
        require(math.isfinite(value) and value >= 0.0, f"Invalid receipt {metric}: {job['job_id']}")


def evaluate_instacart_gate(
    contract: Mapping[str, Any],
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Return the prospective strict, unrounded B-vs-each-baseline decision."""
    _, screening, _ = _job_groups(contract)
    missing = [job["job_id"] for job in screening if job["job_id"] not in completed]
    if missing:
        return {
            "status": "pending",
            "missing_job_ids": missing,
            "all_required_comparisons_passed": False,
        }
    dataset = next(
        row for row in contract["datasets"] if row["dataset"] == "insta_market_basket"
    )
    b_metrics = dataset["B_t0_raw_rmse_validation_metrics"]
    comparisons: dict[str, Any] = {}
    passed = True
    for job in screening:
        receipt = completed[job["job_id"]]
        _validated_receipt_identity(job, receipt)
        baseline_metrics = receipt["metrics"]
        metric_results: dict[str, Any] = {}
        for metric in ("overall_mae", "raw_rmse"):
            b_value = float(b_metrics[metric])
            baseline_value = float(baseline_metrics[metric])
            metric_passed = b_value < baseline_value
            passed = passed and metric_passed
            metric_results[metric] = {
                "B": b_value,
                "baseline": baseline_value,
                "B_minus_baseline": b_value - baseline_value,
                "strictly_lower": metric_passed,
            }
        comparisons[str(job["backbone"])] = metric_results
    return {
        "status": "passed" if passed else "failed",
        "comparison": "strict_unrounded_less_than",
        "comparisons": comparisons,
        "all_required_comparisons_passed": passed,
    }


def schedule_snapshot(
    contract: Mapping[str, Any],
    completed: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute the only jobs currently eligible on each host."""
    validated = validate_contract(contract)
    job_index = validated["jobs"]
    unknown = sorted(set(completed) - set(job_index))
    require(not unknown, f"Completed registry has unauthorized jobs: {unknown}")
    for job_id, receipt in completed.items():
        _validated_receipt_identity(job_index[job_id], receipt)

    e1, screening, post_gate = _job_groups(contract)
    completed_ids = set(completed)
    e1_ids = [job["job_id"] for job in e1]
    screening_ids = [job["job_id"] for job in screening]
    post_ids = [job["job_id"] for job in post_gate]

    # e1 is deliberately serial on one GPU.  A completed suffix with a gap is
    # evidence that the frozen schedule was bypassed.
    missing_e1_seen = False
    for job_id in e1_ids:
        if job_id not in completed_ids:
            missing_e1_seen = True
        elif missing_e1_seen:
            raise ValueError("e1 completion is not a prefix of the frozen order")
    if not set(e1_ids).issubset(completed_ids):
        require(not (set(screening_ids) | set(post_ids)) & completed_ids, "Full run completed before all e1 audits")
        next_job = next(job for job in e1 if job["job_id"] not in completed_ids)
        return {
            "schema_version": 1,
            "status": "e1",
            "phase": "e1_all_six_on_5090",
            "completed_job_ids": [job_id for job_id in e1_ids if job_id in completed_ids],
            "next_job_by_host": {"5080": None, "5090": next_job["job_id"]},
            "instacart_gate": {"status": "not_reached", "all_required_comparisons_passed": False},
            "held_out_test_evaluated": False,
            "additional_seeds_executed": False,
        }

    missing_screening = [job for job in screening if job["job_id"] not in completed_ids]
    if missing_screening:
        require(not set(post_ids) & completed_ids, "Post-gate job exists before gate validation")
        next_by_host = {
            host: next(
                (job["job_id"] for job in missing_screening if job["host_role"] == host),
                None,
            )
            for host in HOST_ROLES
        }
        return {
            "schema_version": 1,
            "status": "instacart_screening",
            "phase": "instacart_seed42_split_across_hosts",
            "completed_job_ids": [*e1_ids, *[job_id for job_id in screening_ids if job_id in completed_ids]],
            "next_job_by_host": next_by_host,
            "instacart_gate": evaluate_instacart_gate(contract, completed),
            "held_out_test_evaluated": False,
            "additional_seeds_executed": False,
        }

    gate = evaluate_instacart_gate(contract, completed)
    if gate["status"] != "passed":
        return {
            "schema_version": 1,
            "status": "stopped_gate_failed",
            "phase": "strict_instacart_gate",
            "completed_job_ids": [*e1_ids, *screening_ids],
            "next_job_by_host": {"5080": None, "5090": None},
            "instacart_gate": gate,
            "held_out_test_evaluated": False,
            "additional_seeds_executed": False,
        }

    # Once the gate is open, each host advances its own fixed backbone queue.
    # Intermittent precedes Taxi on both hosts.
    next_by_host: dict[str, str | None] = {}
    for host in HOST_ROLES:
        host_jobs = [job for job in post_gate if job["host_role"] == host]
        missing_seen = False
        for job in host_jobs:
            present = job["job_id"] in completed_ids
            if not present:
                missing_seen = True
            elif missing_seen:
                raise ValueError(f"Post-gate completion is out of order on host {host}")
        next_by_host[host] = next(
            (job["job_id"] for job in host_jobs if job["job_id"] not in completed_ids),
            None,
        )
    all_complete = all(job_id in completed_ids for job_id in post_ids)
    return {
        "schema_version": 1,
        "status": "complete" if all_complete else "post_gate_seed42",
        "phase": "complete" if all_complete else "intermittent_and_taxi_seed42_split_across_hosts",
        "completed_job_ids": [
            job["job_id"]
            for job in [*e1, *screening, *post_gate]
            if job["job_id"] in completed_ids
        ],
        "next_job_by_host": {host: None for host in HOST_ROLES} if all_complete else next_by_host,
        "instacart_gate": gate,
        "held_out_test_evaluated": False,
        "additional_seeds_executed": False,
    }


def audit_job_registry(
    *,
    project: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    validated: Mapping[str, Any],
    artifact_roots: Mapping[str, Path],
    revision: str,
    python_by_host: Mapping[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Re-audit receipts and retain every receipt-less job for eligibility checks."""
    job_index = validated["jobs"]
    completed: dict[str, dict[str, Any]] = {}
    incomplete: dict[str, dict[str, Any]] = {}
    for host, root in artifact_roots.items():
        if not root.exists():
            continue
        root_entries = list(root.iterdir())
        require(
            all(
                entry.name == "jobs"
                and entry.is_dir()
                and not entry.is_symlink()
                for entry in root_entries
            ),
            f"Artifact root contains an unknown entry: {root}",
        )
        jobs_root = root / "jobs"
        if not jobs_root.exists():
            continue
        job_entries = list(jobs_root.iterdir())
        require(
            all(path.is_dir() and not path.is_symlink() for path in job_entries),
            f"Jobs registry contains a non-directory entry: {jobs_root}",
        )
        for job_dir in sorted(job_entries):
            job_id = job_dir.name
            require(job_id in job_index, f"Unauthorized artifact directory: {job_dir}")
            job = job_index[job_id]
            require(job["host_role"] == host, f"Artifact is on the wrong host root: {job_id}")
            receipt_path = job_dir / "audit_receipt.json"
            if not receipt_path.exists():
                require(job_id not in incomplete, f"Duplicate incomplete artifact: {job_id}")
                incomplete[job_id] = {"host_role": host, "job_dir": job_dir}
                continue
            dataset_row = validated["datasets"][job["dataset"]]
            receipt = read_json(receipt_path)
            _validated_receipt_identity(job, receipt)
            fresh = audit_training_artifact(
                training_output=job_dir / "training",
                job=job,
                dataset_row=dataset_row,
                contract=contract,
                revision=revision,
                require_receipt=True,
            )
            require_receipt_matches_fresh_audit(
                job_id=job_id,
                receipt=receipt,
                fresh=fresh,
            )
            command = training_command(
                python=python_by_host[host],
                project=project,
                training_output=job_dir / "training",
                revision=revision,
                dataset_row=dataset_row,
                job=job,
            )
            expected_identity = _job_identity(
                contract_path=contract_path,
                manifest_path=project / "source_manifest.json",
                revision=revision,
                project=project,
                job=job,
                dataset_row=dataset_row,
                command=command,
                training_output=job_dir / "training",
            )
            identity = read_json(job_dir / "job_identity.json")
            require(identity == expected_identity, f"Job identity drifted: {job_id}")
            require(receipt.get("identity_sha256") == canonical_sha256(identity), f"Receipt identity digest drifted: {job_id}")
            require(receipt.get("contract_sha256") == sha256_file(contract_path), f"Receipt contract drifted: {job_id}")
            require(receipt.get("source_manifest_sha256") == sha256_file(project / "source_manifest.json"), f"Receipt source manifest drifted: {job_id}")
            require(receipt.get("job_spec_sha256") == canonical_sha256(job), f"Receipt job spec drifted: {job_id}")
            completed[job_id] = receipt
    return completed, incomplete


def audit_completed_jobs(
    *,
    project: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    validated: Mapping[str, Any],
    artifact_roots: Mapping[str, Path],
    revision: str,
    python_by_host: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper returning completed receipts after a full registry scan."""
    completed, _ = audit_job_registry(
        project=project,
        contract_path=contract_path,
        contract=contract,
        validated=validated,
        artifact_roots=artifact_roots,
        revision=revision,
        python_by_host=python_by_host,
    )
    return completed


def audit_eligible_incomplete_jobs(
    *,
    project: Path,
    contract_path: Path,
    contract: Mapping[str, Any],
    validated: Mapping[str, Any],
    revision: str,
    python_by_host: Mapping[str, str],
    snapshot: Mapping[str, Any],
    incomplete: Mapping[str, Mapping[str, Any]],
) -> None:
    """Allow only the currently eligible interrupted/running job on each host."""
    eligible = {
        str(job_id)
        for job_id in snapshot["next_job_by_host"].values()
        if job_id is not None
    }
    unauthorized = sorted(set(incomplete) - eligible)
    require(
        not unauthorized,
        f"Incomplete artifact is not a current next job: {unauthorized}",
    )

    for job_id, state in incomplete.items():
        job = validated["jobs"][job_id]
        host = str(state["host_role"])
        job_dir = Path(state["job_dir"])
        dataset_row = validated["datasets"][job["dataset"]]
        audit_incomplete_job_layout(job_dir=job_dir, job=job)

        identity_path = job_dir / "job_identity.json"
        status_path = job_dir / "job_status.json"
        runtime_path = job_dir / "runtime_fingerprint.json"
        progressed = any(
            path.exists()
            for path in (
                status_path,
                runtime_path,
                job_dir / "launcher.log",
                job_dir / "training",
            )
        )
        require(
            identity_path.is_file() or not progressed,
            f"Incomplete job progressed without a stable identity: {job_id}",
        )
        if identity_path.is_file():
            command = training_command(
                python=python_by_host[host],
                project=project,
                training_output=job_dir / "training",
                revision=revision,
                dataset_row=dataset_row,
                job=job,
            )
            expected_identity = _job_identity(
                contract_path=contract_path,
                manifest_path=project / "source_manifest.json",
                revision=revision,
                project=project,
                job=job,
                dataset_row=dataset_row,
                command=command,
                training_output=job_dir / "training",
            )
            identity = read_json(identity_path)
            require(identity == expected_identity, f"Incomplete job identity drifted: {job_id}")
        else:
            identity = None

        if status_path.is_file():
            status = read_json(status_path)
            require(status.get("status") in {"running", "failed"}, f"Invalid incomplete status: {job_id}")
            require(status.get("job_id") == job_id, f"Incomplete status job drifted: {job_id}")
            require(status.get("host_role") == host, f"Incomplete status host drifted: {job_id}")
            require(status.get("held_out_test_evaluated") is False, f"Incomplete status scope drifted: {job_id}")
            if "identity_sha256" in status:
                require(identity is not None, f"Incomplete status has no identity: {job_id}")
                require(
                    status["identity_sha256"] == canonical_sha256(identity),
                    f"Incomplete status identity digest drifted: {job_id}",
                )
        if runtime_path.is_file():
            validate_runtime_fingerprint(
                read_json(runtime_path),
                contract=contract,
                job=job,
            )


def build_snapshot(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    project = args.project_root.resolve()
    contract_path = args.contract.resolve()
    contract = read_json(contract_path)
    validated = validate_contract(contract)
    verify_source_manifest(project, revision=args.source_revision, contract_path=contract_path)
    validate_b_reference_evidence(
        project,
        contract=contract,
        datasets=validated["datasets"],
    )
    verify_dataset_inputs(project, validated["datasets"])
    paths = fixed_execution_paths(args.source_revision)
    roots = {
        host: path.resolve()
        for host, path in paths["artifact_roots"].items()
    }
    pythons = {"5080": args.python_5080, "5090": args.python_5090}
    completed, incomplete = audit_job_registry(
        project=project,
        contract_path=contract_path,
        contract=contract,
        validated=validated,
        artifact_roots=roots,
        revision=args.source_revision,
        python_by_host=pythons,
    )
    snapshot = schedule_snapshot(contract, completed)
    audit_eligible_incomplete_jobs(
        project=project,
        contract_path=contract_path,
        contract=contract,
        validated=validated,
        revision=args.source_revision,
        python_by_host=pythons,
        snapshot=snapshot,
        incomplete=incomplete,
    )
    snapshot.update({
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(contract_path),
        "source_revision": args.source_revision,
        "audited_completed_job_count": len(completed),
        "audited_incomplete_job_ids": sorted(incomplete),
        "updated_at": utc_now(),
    })
    return snapshot, validated, completed


def execute(args: argparse.Namespace) -> dict[str, Any]:
    require(args.host_role in HOST_ROLES, f"Unknown host role: {args.host_role}")
    project = args.project_root.resolve()
    paths = fixed_execution_paths(args.source_revision)
    coordination = paths["coordination_root"].resolve()
    require(
        project != coordination
        and project not in coordination.parents
        and coordination not in project.parents,
        "Coordination root and packaged source tree must be disjoint",
    )
    coordination.mkdir(parents=True, exist_ok=True)
    lock_path = coordination / f"controller_{args.host_role}.lock"
    status_path = coordination / f"controller_{args.host_role}_status.json"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            current = read_json(status_path) if status_path.exists() else {
                "status": "controller_busy",
                "host_role": args.host_role,
                "held_out_test_evaluated": False,
            }
            return {**current, "idempotent_busy_return": True}
        try:
            snapshot, validated, _ = build_snapshot(args)
            snapshot["controller_host_role"] = args.host_role
            save_json_atomic(status_path, snapshot)
            if args.action == "status":
                return snapshot
            job_id = snapshot["next_job_by_host"][args.host_role]
            if job_id is None:
                return snapshot
            running = {
                **snapshot,
                "status": "running_job",
                "running_job_id": job_id,
                "updated_at": utc_now(),
            }
            save_json_atomic(status_path, running)
            artifact_roots = paths["artifact_roots"]
            output_root = artifact_roots[args.host_role]
            job_args = argparse.Namespace(
                project_root=args.project_root,
                contract=args.contract,
                source_revision=args.source_revision,
                host_role=args.host_role,
                job_id=job_id,
                python_5080=args.python_5080,
                python_5090=args.python_5090,
                verify_only=False,
            )
            result = execute_job(job_args)
            final_snapshot, _, _ = build_snapshot(args)
            final_snapshot.update({
                "controller_host_role": args.host_role,
                "last_completed_job_id": job_id,
                "last_job_receipt_sha256": sha256_file(
                    output_root.resolve() / "jobs" / job_id / "audit_receipt.json"
                ),
                "last_job_metrics": result["metrics"],
                "updated_at": utc_now(),
            })
            save_json_atomic(status_path, final_snapshot)
            return final_snapshot
        except BaseException as exc:
            save_json_atomic(status_path, {
                "schema_version": 1,
                "status": "blocked",
                "host_role": args.host_role,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "held_out_test_evaluated": False,
                "additional_seeds_executed": False,
                "updated_at": utc_now(),
            })
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path, default=PROJECT_ROOT / CONTRACT_REL)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--host-role", choices=HOST_ROLES, required=True)
    parser.add_argument("--action", choices=("status", "run-next"), default="status")
    parser.add_argument(
        "--python-5080",
        default="/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python",
    )
    parser.add_argument(
        "--python-5090",
        default="/opt/miniconda3/envs/ai_env/bin/python",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    result = execute(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
