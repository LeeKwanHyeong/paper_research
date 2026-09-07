#!/usr/bin/env python3
"""Run the frozen causal-QKV candidate on an idle RTX 5090 only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    DATASETS, evaluate_gate, gpu_preflight, job_command, read_json,
    require, save_json, sha256,
)

CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_causal_qkv_screening_v1.json"
PROFILER = ROOT / "paper/scripts/profile_hard_lmm_causal_qkv.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_inputs(contract: dict[str, Any]) -> dict[str, Any]:
    reference = contract["B_validation_reference"]
    require(sha256(ROOT / reference["source"]) == reference["sha256"],
            "B reference evidence hash drift")
    evidence = {}
    for dataset, binding in contract["data_bindings"].items():
        spec = DATASETS[dataset]
        actual = {
            "data_sha256": sha256(ROOT / spec["data"]),
            "split_manifest_sha256": sha256(ROOT / spec["manifest"]),
        }
        require(all(actual[key] == binding[key] for key in actual),
                f"Data/split checksum drift: {dataset}")
        evidence[dataset] = actual
    return evidence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-role", choices=("5090",), default="5090")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    if re.fullmatch(r"[0-9a-f]{40}", args.source_revision) is None:
        parser.error("--source-revision must be a full lowercase Git SHA")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # Import before creating a campaign, so a missing audit module fails safely.
    from paper.scripts.audit_hard_lmm_causal_qkv import audit_causal_qkv_job

    contract = read_json(CONTRACT_PATH)
    require(contract["host_role"] == args.host_role == "5090", "5090-only contract")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "campaign_status.json"
    require(not status_path.exists(), "Campaign exists; implicit restart is forbidden")
    require(not (output_root / "jobs").exists(), "Job directory already exists")
    candidate = contract["candidate"]
    status: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256(CONTRACT_PATH),
        "source_revision": args.source_revision,
        "host_role": args.host_role,
        "candidate": candidate,
        "started_at": utc_now(),
        "status": "preflight",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "jobs": {},
    }

    def persist() -> None:
        status["updated_at"] = utc_now()
        save_json(status_path, status)

    persist()
    try:
        status["gpu_preflight"] = gpu_preflight(candidate["gpu_name_contains"])
        status["input_checksums"] = verify_inputs(contract)
        save_json(output_root / "frozen_contract.json", contract)
        status["status"] = "running_cost_profile"
        persist()
        profile_path = output_root / "cuda_cost_profile.json"
        with (output_root / "cuda_cost_profile.log").open("x") as log:
            subprocess.run([args.python, "-s", str(PROFILER), "--output", str(profile_path)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        profile = read_json(profile_path)
        require(profile["status"] == "passed", "CUDA cost or learning-path gate failed")
        status["cost_profile"] = {
            "path": str(profile_path), "sha256": sha256(profile_path),
            "cost_gate": profile["cost_gate"],
        }
        persist()

        for phase_name in ("e1", "seed42_screening"):
            phase = contract["phases"][phase_name]
            status["status"] = f"running_{phase_name}"
            for dataset in phase["datasets_in_order"]:
                job_id = f"{phase_name}_{dataset}"
                output = output_root / "jobs" / job_id
                output.mkdir(parents=True, exist_ok=False)
                command = job_command(
                    python=args.python, candidate=candidate, dataset=dataset,
                    output=output, source_revision=args.source_revision,
                    phase=phase, host_role=args.host_role,
                )
                status["current_job"] = job_id
                status["jobs"][job_id] = {
                    "status": "running", "started_at": utc_now(),
                    "output": str(output), "command": command,
                }
                persist()
                with (output / "campaign_runner.log").open("x") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log,
                                   stderr=subprocess.STDOUT, check=True)
                audit = audit_causal_qkv_job(
                    output, candidate=candidate, dataset=dataset,
                    source_revision=args.source_revision,
                    expected_epochs=phase["epochs"], binding=contract["data_bindings"][dataset],
                )
                require(audit.get("status") == "passed", f"Job audit did not pass: {job_id}")
                status["jobs"][job_id].update(audit, completed_at=utc_now())
                if phase_name == "seed42_screening":
                    gate = evaluate_gate(audit["metrics"],
                                         contract["B_validation_reference"]["datasets"][dataset])
                    status["jobs"][job_id]["gate"] = gate
                    if gate["status"] != "passed":
                        status.update(status="stopped_seed42_gate_failed", failed_dataset=dataset,
                                      completed_at=utc_now(), current_job=None)
                        persist()
                        return
                persist()
        status.update(status="seed42_all_datasets_passed", current_job=None, completed_at=utc_now())
        persist()
    except BaseException as error:
        status.update(status="failed_execution", error=repr(error), completed_at=utc_now())
        persist()
        raise


if __name__ == "__main__":
    main()
