"""Two-host frozen successor comparison, with one qualification-start deadline."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time

from paper.scripts import observed_slot_parallel_common as shared
from paper.scripts import successor_episode_common as episode

ROOT = Path(__file__).resolve().parents[2]
DESIGN, INHERITED, RUNTIME_REFERENCE = episode.DESIGN, episode.INHERITED, episode.RUNTIME_REFERENCE
SCHEMA = "successor_episode_parallel_execution_v1"
ARMS, ROLES, DATASETS = episode.ARMS, episode.ROLES, episode.DATASETS
ASSIGNMENTS = {"5090": ["insta_market_basket"],
               "5080": ["intermittent_frozen_5000", "yellow_trip_hourly"]}
EXECUTION_ROLE = "fresh_successor_episode_parallel_validation"
LIMITS = {**shared.LIMITS, "total_seconds": 172800}
POLICY, ACCEPTANCE, COST_GATES = deepcopy(shared.POLICY), deepcopy(shared.ACCEPTANCE), deepcopy(shared.COST_GATES)
ENVIRONMENT = deepcopy(shared.ENVIRONMENT)
ENTRYPOINTS = ("paper/scripts/successor_episode_parallel_common.py",
               "paper/scripts/run_successor_episode_parallel.py",
               "paper/scripts/qualify_successor_episode_parallel.py")
require, sha_json, sha_file, write_json = shared.require, shared.sha_json, shared.sha_file, shared.write_json
apply_environment, gpu_pids, runtime_check = shared.apply_environment, shared.gpu_pids, shared.runtime_check


def source_manifest(root=None):
    return shared.source_manifest(ROOT if root is None else root, entrypoints=ENTRYPOINTS,
                                  extra_files=(DESIGN, INHERITED, RUNTIME_REFERENCE))


def verify_source(contract):
    actual = source_manifest()
    require(actual == contract["source"]["files"]
            and sha_json(actual) == contract["source"]["files_sha256"],
            "Frozen parallel successor source changed")


def validate_assignment(contract, host):
    require(host in ASSIGNMENTS and set(contract.get("hosts", {})) == set(ASSIGNMENTS),
            "Exactly the assigned 5080 and 5090 hosts are required")
    require(contract.get("arms") == list(ARMS) and contract.get("seed") == 42
            and contract.get("epochs") == 120, "Three-arm/seed/epoch contract changed")
    for alias, datasets in ASSIGNMENTS.items():
        require(contract["hosts"][alias]["assigned_datasets"] == datasets,
                "Fixed same-dataset three-arm host assignment changed")
    return contract["hosts"][host]


def read_contract(path, host, check_source=True):
    value = json.loads(Path(path).read_text())
    require(value.get("schema") == SCHEMA, "Wrong parallel successor schema")
    validate_assignment(value, host)
    require(value.get("limits") == LIMITS and value.get("policy") == POLICY,
            "Execution policy/ceiling changed")
    require(value.get("acceptance") == ACCEPTANCE and value.get("cost_gates") == COST_GATES,
            "Frozen acceptance or cost gates changed")
    require(value.get("total_optimizer_steps") == 6816240 and value.get("endpoint_replays") == 18,
            "Nine-arm optimizer/replay budget changed")
    require(value.get("approval_required") is True, "Explicit contract-bound approval is required")
    require(value.get("design_sha256") == sha_file(ROOT / DESIGN)
            and value.get("inherited_data_contract_sha256") == sha_file(ROOT / INHERITED),
            "Frozen design/data provenance changed")
    expected = deepcopy(json.loads((ROOT / INHERITED).read_text())["datasets"])
    for data in expected:
        data["epochs"] = 120
    require(value.get("datasets") == expected, "Frozen train statistics/splits/loader changed")
    reference = json.loads((ROOT / RUNTIME_REFERENCE).read_text())["hosts"]
    for alias, entry in value["hosts"].items():
        root = Path(entry["root"])
        require(root == Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts")
                / "successor_episode_seed42_parallel_20260914_v1", "Wrong isolated experiment root")
        require(Path(entry["source_root"]) == root / "source"
                and Path(entry["output_dir"]) == root / "run", "Wrong source/output isolation")
        require(entry["environment"] == ENVIRONMENT
                and all(entry[key] == reference[alias][key]
                        for key in ("python", "gpu_uuid", "runtime_expected")),
                "Host runtime/Python/device contract changed")
        require(entry["tmux"] == f"episode_seed42_parallel_20260914_{alias}",
                "Wrong owned tmux session")
    source = value["source"]
    revision = source.get("base_git_revision")
    require(isinstance(revision, str) and len(revision) == 40
            and all(char in "0123456789abcdef" for char in revision)
            and source.get("files_sha256") == sha_json(source["files"]), "Malformed source identity")
    if check_source:
        verify_source(value)
    return value


def verify_approval(contract, approval):
    require(approval.get("schema") == "successor_episode_parallel_approval_v1"
            and approval.get("contract_sha256") == sha_json(contract)
            and approval.get("hosts") == ["5080", "5090"] and approval.get("approved") is True,
            "Missing explicit approval bound to both hosts and this contract")
    require(approval.get("scope") == "dual_native_synthetic_cuda_qualification_and_fresh_nine_arm_validation"
            and approval.get("user_instruction") == "5080/5090 병렬로 진행하자.",
            "Approval scope or user instruction differs")


def _qualification_bindings(contract, qualifications):
    """Check both portable receipts; file bytes are checked only by their own host."""
    require(set(qualifications) == set(ASSIGNMENTS), "Both native CUDA qualifications are required")
    starts = []
    expected_checks = ("initial_output_objective_shared_preclip_gradient_exact",
                       "activated_target_prediction_causal", "activated_Q_K_U_gradients",
                       "saved_optimizer_rng_next_step_exact")
    for alias, binding in qualifications.items():
        receipt = binding.get("receipt", {})
        expected_path = Path(contract["hosts"][alias]["root"]) / "qualification" / "receipt.json"
        require(binding.get("path") == str(expected_path)
                and isinstance(binding.get("sha256"), str) and len(binding["sha256"]) == 64
                and binding.get("receipt_sha256") == sha_json(receipt), "Qualification binding differs")
        require(receipt.get("status") == "passed" and receipt.get("host") == alias
                and receipt.get("contract_sha256") == sha_json(contract)
                and receipt.get("source_files_sha256") == contract["source"]["files_sha256"],
                "Both native qualifications must pass for the frozen contract/source")
        runtime = receipt.get("runtime", {})
        require(runtime.get("gpu", {}).get("uuid") == contract["hosts"][alias]["gpu_uuid"]
                and all(runtime.get(key) == value for key, value in
                        contract["hosts"][alias]["runtime_expected"].items()), "Qualified runtime/device differs")
        require(receipt.get("device") == "cuda:0" and receipt.get("real_data_loaded") is False
                and receipt.get("held_out_evaluated") is False, "Qualification must be synthetic native CUDA")
        checks = receipt.get("checks", {})
        require(set(checks) == set(ARMS[1:]) and all(
            checks[name].get(key) is True for name in ARMS[1:] for key in expected_checks),
            "Native model qualification checks did not all pass")
        gates = receipt.get("cost_gate_checks", [])
        require(len(gates) == 4 and {(row.get("backbone"), row.get("length")) for row in gates}
                == {(name, length) for name in ARMS[1:] for length in (64, 256)}
                and all(row.get(key) is True for row in gates
                        for key in ("step", "relative_memory", "device_memory", "parameters")),
                "Native cost gates did not all pass")
        start = receipt.get("started_at_unix")
        require(type(start) in (int, float) and math.isfinite(start) and 0 < start <= time.time(),
                "Qualification start is invalid or in the future")
        starts.append(start)
    return min(starts)


def verify_permit(contract, permit, host):
    validate_assignment(contract, host)
    require(permit.get("schema") == "successor_episode_parallel_start_v1"
            and permit.get("contract_sha256") == sha_json(contract), "Wrong parallel start permit")
    verify_approval(contract, permit.get("approval", {}))
    first_start = _qualification_bindings(contract, permit.get("qualifications", {}))
    started, deadline = permit.get("started_at_unix"), permit.get("deadline_unix")
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started == first_start and deadline == first_start + LIMITS["total_seconds"]
            and started <= time.time() < deadline, "Expired, shifted, or extended common deadline")
    binding = permit["qualifications"][host]
    path = Path(binding["path"])
    require(sha_file(path) == binding["sha256"] and json.loads(path.read_text()) == binding["receipt"],
            "Local native qualification file differs from the embedded receipt")


def create_permit(contract, approval, qualifications):
    """Build a portable permit from both collected native qualification bindings."""
    validate_assignment(contract, "5090")
    verify_approval(contract, approval)
    started = _qualification_bindings(contract, qualifications)
    require(time.time() < started + LIMITS["total_seconds"], "Qualification-start budget has expired")
    return {"schema": "successor_episode_parallel_start_v1", "contract_sha256": sha_json(contract),
            "approval": deepcopy(approval), "started_at_unix": started,
            "deadline_unix": started + LIMITS["total_seconds"],
            "qualifications": deepcopy(qualifications)}
