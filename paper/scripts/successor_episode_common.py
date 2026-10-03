"""Frozen, 5090-only execution contract for three-arm episode memory comparison."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

from paper.scripts import observed_slot_parallel_common as shared

ROOT = Path(__file__).resolve().parents[2]
DESIGN = "paper/contracts/hard_lmm_successor_episode_memory_design_v1.json"
INHERITED = shared.INHERITED
RUNTIME_REFERENCE = "paper/contracts/observed_slot_parallel_execution_v2.json"
SCHEMA = "successor_episode_execution_5090_v1"
ARMS = ("titantpp", "titantpp_same_event_episode_memory", "titantpp_successor_episode_memory")
ROLES = ("raw_rmse_baseline_alignment", "hard_lmm_same_event_episode_memory_control", "hard_lmm_successor_episode_memory_candidate")
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
LIMITS = {**shared.LIMITS, "total_seconds": 172800}
ENTRYPOINTS = ("paper/scripts/successor_episode_common.py", "paper/scripts/run_successor_episode_comparison.py",
               "paper/scripts/verify_successor_episode_cost.py")
require, sha_json, sha_file, write_json = shared.require, shared.sha_json, shared.sha_file, shared.write_json
apply_environment, gpu_pids, runtime_check = shared.apply_environment, shared.gpu_pids, shared.runtime_check


def source_manifest(root=ROOT):
    return shared.source_manifest(root, entrypoints=ENTRYPOINTS, extra_files=(DESIGN, INHERITED, RUNTIME_REFERENCE))


def verify_source(contract):
    actual = source_manifest()
    require(actual == contract["source"]["files"] and sha_json(actual) == contract["source"]["files_sha256"],
            "Frozen successor execution source changed")


def validate_assignment(contract, host):
    require(host == "5090" and set(contract.get("hosts", {})) == {"5090"}, "Only 5090 is authorized as a proposed host")
    require(contract.get("arms") == list(ARMS) and contract.get("seed") == 42 and contract.get("epochs") == 120,
            "Three-arm/seed/epoch contract changed")
    require(contract["hosts"][host]["assigned_datasets"] == list(DATASETS), "Fixed dataset order changed")
    return contract["hosts"][host]


def read_contract(path, host="5090", check_source=True):
    value = json.loads(Path(path).read_text())
    require(value.get("schema") == SCHEMA, "Wrong successor execution schema")
    entry = validate_assignment(value, host)
    require(value.get("limits") == LIMITS and value.get("policy") == shared.POLICY, "Execution policy/ceiling changed")
    require(value.get("acceptance") == shared.ACCEPTANCE and value.get("cost_gates") == shared.COST_GATES,
            "Frozen quality/cost gates changed")
    require(value.get("total_optimizer_steps") == 6816240 and value.get("endpoint_replays") == 18,
            "Nine-arm exposure or replay budget changed")
    require(value.get("approval_required") is True, "New training requires a separately recorded approval")
    require(value.get("design_sha256") == sha_file(ROOT / DESIGN)
            and value.get("inherited_data_contract_sha256") == sha_file(ROOT / INHERITED), "Design/data provenance differs")
    expected = deepcopy(json.loads((ROOT / INHERITED).read_text())["datasets"])
    for data in expected:
        data["epochs"] = 120
    require(value.get("datasets") == expected, "Frozen train statistics/splits/loader changed")
    root = Path(entry["root"])
    require(root.is_absolute() and root.parent == Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts")
            and root.name.startswith("successor_episode_seed42_5090_"), "Unsafe experiment isolation")
    require(Path(entry["source_root"]) == root / "source" and Path(entry["output_dir"]) == root / "run", "Wrong source/output isolation")
    require(entry["environment"] == shared.ENVIRONMENT
            and entry["python"] == "/opt/miniconda3/envs/ai_env/bin/python3.12"
            and entry["gpu_uuid"] == "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f", "5090 environment changed")
    expected_runtime = json.loads((ROOT / RUNTIME_REFERENCE).read_text())["hosts"]["5090"]["runtime_expected"]
    require(entry["runtime_expected"] == expected_runtime, "Native runtime contract changed")
    require(value["source"]["files_sha256"] == sha_json(value["source"]["files"])
            and len(value["source"]["base_git_revision"]) == 40, "Malformed frozen source identity")
    if check_source:
        verify_source(value)
    return value


def verify_approval(contract, approval):
    require(approval.get("schema") == "successor_episode_approval_v1"
            and approval.get("contract_sha256") == sha_json(contract)
            and approval.get("host") == "5090" and approval.get("approved") is True,
            "Missing explicit approval bound to this 5090 contract")
    require(approval.get("scope") == "synthetic_cuda_qualification_and_fresh_nine_arm_validation"
            and isinstance(approval.get("user_instruction"), str) and bool(approval["user_instruction"].strip()),
            "Approval must record the user's instruction and exact scope")


def verify_permit(contract, permit, host):
    validate_assignment(contract, host)
    require(permit.get("schema") == "successor_episode_start_v1"
            and permit.get("contract_sha256") == sha_json(contract), "Wrong start permit")
    verify_approval(contract, permit.get("approval", {}))
    started, deadline = permit.get("started_at_unix"), permit.get("deadline_unix")
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started <= time.time() < deadline and deadline - started == LIMITS["total_seconds"],
            "Expired, future, or extended qualification-start deadline")
    require(set(permit.get("qualifications", {})) == {"5090"}, "Exactly one native qualification required")
    binding = permit["qualifications"]["5090"]
    path = Path(contract["hosts"][host]["root"]) / "qualification" / "receipt.json"
    require(binding.get("path") == str(path) and binding.get("sha256") == sha_file(path), "Qualification file binding differs")
    receipt = json.loads(path.read_text())
    require(binding.get("receipt") == receipt and binding.get("receipt_sha256") == sha_json(receipt), "Embedded receipt differs")
    require(receipt.get("status") == "passed" and receipt.get("host") == host
            and receipt.get("contract_sha256") == sha_json(contract)
            and receipt.get("source_files_sha256") == contract["source"]["files_sha256"]
            and receipt.get("started_at_unix") == started, "Qualification did not pass for this source/deadline")
    require(receipt["runtime"]["gpu"]["uuid"] == contract["hosts"][host]["gpu_uuid"], "Qualified GPU differs")
    require(all(receipt["runtime"].get(key) == expected for key, expected in contract["hosts"][host]["runtime_expected"].items()),
            "Qualified runtime differs")


def create_permit(contract, approval):
    verify_approval(contract, approval)
    path = Path(contract["hosts"]["5090"]["root"]) / "qualification" / "receipt.json"
    receipt = json.loads(path.read_text())
    value = {"schema": "successor_episode_start_v1", "contract_sha256": sha_json(contract),
             "approval": approval, "started_at_unix": receipt["started_at_unix"],
             "deadline_unix": receipt["started_at_unix"] + LIMITS["total_seconds"],
             "qualifications": {"5090": {"path": str(path), "sha256": sha_file(path),
                 "receipt": receipt, "receipt_sha256": sha_json(receipt)}}}
    verify_permit(contract, value, "5090")
    return value
