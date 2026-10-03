"""Immutable scientific and execution identity for the three-arm parallel run.

This module performs no training, synchronization, or launch on import.
The historical design remains unchanged; a new execution overlay records the
user's subsequent authorization and the actual two-host partition.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import time

from paper.scripts.run_quantity_comparison import (
    PACKAGES, read_json, require, sha_file, sha_json, write_json,
)
from paper.scripts.run_raw_aux_gradient_comparison import (
    DESIGN_PATH, DESIGN_SHA256, load_design,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "raw_aux_gradient_parallel_execution_v1"
ASSIGNMENTS = {
    "5090": ["B_log_original", "mixed_raw_capped_original"],
    "5080": ["mixed_original"],
}
LIMITS = {
    "qualification_seconds": 900,
    "training_seconds": 43200,
    "checkpoint_seconds": 1800,
    "preflight_seconds": 1800,
    "total_seconds": 46800,
    "per_host_output_bytes": 2147483648,
    "max_file_bytes": 67108864,
    "min_free_bytes": 5368709120,
    "max_concurrent_gpu_jobs_per_host": 1,
}
ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0",
    "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
    "PYTHONHASHSEED": "42",
}
ENTRYPOINTS = (
    "paper/scripts/run_raw_aux_gradient_partition.py",
    "paper/scripts/run_raw_aux_gradient_cuda.py",
    "paper/scripts/raw_aux_gradient_partition_acceptance.py",
    "paper/scripts/raw_aux_parallel_common.py",
)


def source_manifest(root=ROOT):
    """Resolve all first-party dependencies of every new execution entrypoint."""
    root = Path(root).resolve()
    pending, visited = [root / p for p in ENTRYPOINTS], set()

    def add(module):
        if module.split(".")[0] not in PACKAGES:
            return
        stem = root.joinpath(*module.split("."))
        for candidate in (stem.with_suffix(".py"), stem / "__init__.py"):
            if candidate.is_file():
                pending.append(candidate)
        parent = stem.parent
        while parent != root:
            if (parent / "__init__.py").is_file():
                pending.append(parent / "__init__.py")
            parent = parent.parent

    while pending:
        path = pending.pop().resolve()
        require(path.is_relative_to(root) and path.is_file(), "Missing/escaped frozen source")
        if path in visited:
            continue
        visited.add(path)
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join(prefix + ([node.module] if node.module else []))
                add(module)
                for alias in node.names:
                    if alias.name != "*":
                        add(module + "." + alias.name)
    design = load_design(root / "paper/contracts/raw_aux_gradient_control_design_v1.json")
    visited.add(root / "paper/contracts/raw_aux_gradient_control_design_v1.json")
    measurement = design["evidence"]["measurement_contract"]
    require(sha_file(root / measurement["path"]) == measurement["sha256"], "Frozen coefficient provenance changed")
    visited.add(root / measurement["path"])
    files = {str(p.relative_to(root)): sha_file(p) for p in sorted(visited)}
    return {"files": files, "files_sha256": sha_json(files)}


def data_entry(contract=None):
    design = load_design()
    e = design["initialization_and_exposure"]
    data = {
        "dataset_id": design["scope"]["dataset_id"], "seed": 42,
        "inherited_data_identity": e["data_identity"], "loader": e["loader"],
        "model": e["model"], "optimizer": e["optimizer"], "statistics": e["statistics"],
        "mu_all_train_rows": e["statistics"]["train_log_mean"],
        "expected_global_steps": e["steps_per_arm"], "epochs": 120,
    }
    if contract is not None:
        require(contract["dataset"] == data, "Dataset/initialization/exposure contract changed")
    return deepcopy(data)


def read_contract(path, *, check_source=True):
    c = read_json(path)
    require(c.get("schema") == SCHEMA and c.get("design_sha256") == DESIGN_SHA256,
            "Wrong parallel execution design")
    require(c.get("limits") == LIMITS, "Parallel resource/time limits changed")
    require(set(c.get("hosts", {})) == set(ASSIGNMENTS), "Exactly two hosts required")
    data_entry(c)
    require(c.get("authorization", {}).get("user_instruction") == "5080이랑 병렬로 진행하자"
            and c["authorization"].get("scope") == "fresh_three_arm_parallel_training_after_two_host_qualification",
            "Latest explicit parallel execution authorization missing")
    require(c.get("policy") == {"resume": False, "retry": False, "held_out": False,
            "coefficient_refit": False, "additional_arms": False, "scheduler_changes": False,
            "runtime_mutation": False, "automatic_transfer": False}, "Execution policy changed")
    for alias, ids in ASSIGNMENTS.items():
        h = c["hosts"][alias]
        require(h["assigned_arms"] == ids and h["environment"] == ENVIRONMENT,
                "Host assignment or numerical environment changed")
        root = Path(h["root"])
        require(root.is_absolute() and root.parent == Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts")
                and root.name.startswith("raw_aux_gradient_seed42_parallel_"), "Unsafe/non-isolated run root")
        require(Path(h["source_root"]) == root / "source" and Path(h["output_dir"]) == root / "run",
                "Source/output isolation changed")
        expected_python = ("/opt/miniconda3/envs/ai_env/bin/python3.12" if alias == "5090"
                           else "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12")
        require(h["python"] == expected_python, "Host Python changed")
        require(h["gpu_uuid"] == {"5090": "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f",
                                  "5080": "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"}[alias], "GPU assignment changed")
    require(c["source"]["files_sha256"] == sha_json(c["source"]["files"]), "Source digest mismatch")
    if check_source:
        require(c["source"] == source_manifest(), "Execution source differs from frozen manifest")
    return c


def apply_environment(host_spec):
    require(host_spec["environment"] == ENVIRONMENT, "Strict environment changed")
    os.environ.update(ENVIRONMENT)


def runtime_check(contract, host):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    h = contract["hosts"][host]
    require(Path(sys.executable).resolve() == Path(h["python"]).resolve(), "Unexpected Python executable")
    require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "Strict environment was not applied")
    configure_runtime("cuda:0", threads=4)
    runtime = runtime_identity("cuda:0")
    require(runtime["gpu"]["uuid"] == h["gpu_uuid"], "Unexpected active CUDA GPU")
    if h.get("runtime_expected") is not None:
        require(runtime == h["runtime_expected"], "Frozen runtime changed")
    return runtime


def prepare_inputs(contract=None):
    from paper.scripts.quantity_comparison_data import prepare_quantity_comparison_data
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    data = data_entry(contract)
    frame, metadata = prepare_quantity_comparison_data(data)
    return frame, bind_frozen_statistics(data, metadata), data


def build_inputs(data, frame, metadata):
    from paper.scripts.run_time_quantity_diagnostic import build_arm_inputs
    return build_arm_inputs(data, frame, metadata)


def make_identity(contract, host, runtime):
    return {"execution_contract_sha256": sha_json(contract), "host_alias": host,
            "raw_aux_design_sha256": DESIGN_SHA256, "source": contract["source"],
            "runtime": runtime, "data": contract["dataset"]["inherited_data_identity"],
            "dataset_id": contract["dataset"]["dataset_id"]}


def validate_permit(contract, permit, host, *, now=None):
    now = time.time() if now is None else now
    require(host in ASSIGNMENTS and permit.get("schema") == "raw_aux_parallel_start_permit_v1",
            "Missing common start permit")
    require(permit.get("permit_sha256") == sha_json({k: v for k, v in permit.items() if k != "permit_sha256"}),
            "Start permit digest changed")
    require(permit.get("execution_contract_sha256") == sha_json(contract)
            and permit.get("authorized_arms") == ASSIGNMENTS, "Start permit scope changed")
    start, deadline = permit["started_at_unix"], permit["deadline_unix"]
    require(type(start) in (float, int) and start <= now < deadline
            and deadline - start == LIMITS["total_seconds"], "Start permit expired or budget extended")
    require(set(permit.get("qualifications", {})) == set(ASSIGNMENTS), "Both fresh qualifications required")
    initial = load_design()["initialization_and_exposure"]["initial_state_sha256"]
    for alias, entry in permit["qualifications"].items():
        r = entry["receipt"]
        require(entry["sha256"] == sha_json(r), "Qualification receipt digest changed")
        require(r.get("schema") == "raw_aux_gradient_cuda_qualification_v1"
                and r.get("passed") is True and r.get("qualifies_cuda") is True
                and r.get("host_alias") == alias and r.get("execution_contract_sha256") == sha_json(contract)
                and r.get("source_files_sha256") == contract["source"]["files_sha256"]
                and r.get("initial_state_sha256") == initial, "Qualification identity mismatch")
        require(r["runtime"]["device"] == "cuda:0"
                and r["runtime"]["gpu"]["uuid"] == contract["hosts"][alias]["gpu_uuid"],
                "Qualification GPU mismatch")
        require(type(r.get("checked_at_unix")) in (float, int)
                and 0 <= start - r["checked_at_unix"] <= 7200, "Qualification is not fresh")
        validate_qualification_evidence(r, initial)
    return permit["qualifications"][host]["receipt"]


def validate_qualification_evidence(receipt, initial):
    """Require bounded, non-vacuous proof structure, beyond a passed boolean."""
    r = receipt
    require(type(r.get("elapsed_seconds")) in (float, int)
            and math.isfinite(r["elapsed_seconds"]) and 0 < r["elapsed_seconds"] <= LIMITS["qualification_seconds"]
            and r["checked_at_unix"] <= r["qualification_deadline_unix"], "Qualification exceeded time limit")
    require(r.get("research_optimizer_updates") == 0 and r.get("synthetic_optimizer_updates") == 24
            and r.get("research_data_or_checkpoint_read") is False, "Qualification work scope changed")
    replay, shape = r["tiny_replay"], r["production_shape"]
    arms = {arm for values in ASSIGNMENTS.values() for arm in values}
    require(replay["passed"] is True and replay["fresh_process_exact_resume"] is True
            and replay["synthetic_optimizer_updates"] == 24
            and replay["paired_exposure"] == {"passed": True, "arms": 3, "epochs": 2, "global_steps_per_arm": 4}
            and set(replay["arms"]) == arms, "Three-arm fresh-process replay proof missing")
    require(shape["passed"] is True and shape["initial_state_sha256"] == initial
            and shape["runtime"] == r["runtime"] and shape["batch_size"] == 128
            and shape["sequence_length"] == 256 and shape["hidden_dim"] == 64
            and shape["optimizer_steps"] == 0 and set(shape["arms"]) == arms,
            "Full production shape proof missing")
    for arm in arms:
        p = replay["arms"][arm]
        require(p["runtime"] == r["runtime"] and p["execution_contract_sha256"] == r["execution_contract_sha256"]
                and p["source"]["files_sha256"] == r["source_files_sha256"], "Replay identity differs")
        s = shape["arms"][arm]
        require(s["initial_state_sha256"] == initial and s["synthetic_nonzero_head"] is True,
                "Shape initialization or nonzero probe missing")
        causal = s["causality"]
        require(all(causal[key] is True for key in ("passed", "target_padding_invariance",
                                                    "valid_history_sensitivity", "nonzero_head"))
                and causal["optimizer_steps"] == 0, "Production causal evidence missing")
    require(shape["arms"]["B_log_original"]["B_preservation"]["objective_and_gradients_bitwise_equal"] is True,
            "Original B gradient proof missing")
    capped = shape["arms"]["mixed_raw_capped_original"]["active_gradient_audit"]
    require(capped["passed"] is True and capped["norm_bound_passed"] is True
            and 0 < capped["s"] < 1 and capped["weighted_raw_norm"] > 0
            and capped["per_parameter_composition_passed"] is True
            and capped["postclip_gradient_passed"] is True and capped["state_and_rng_preserved"] is True,
            "Active composed raw gradient proof missing")
    files = r["proof_files"]
    require(isinstance(files, dict) and "production_shape.json" in files,
            "Qualification proof artifact manifest missing")
    for name, digest in files.items():
        require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
                and isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "Invalid proof artifact identity")


def verify_native_proofs(contract, host, receipt):
    """Before research work, verify the native host's original proof bytes."""
    root = (Path(contract["hosts"][host]["root"]) / "qualification").resolve()
    for name, digest in receipt["proof_files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root) and path.is_file() and not path.is_symlink()
                and path.stat().st_size <= LIMITS["max_file_bytes"] and sha_file(path) == digest,
                "Native qualification proof bytes differ: " + name)
    require(read_json(root / "production_shape.json") == receipt["production_shape"],
            "Embedded shape evidence differs from native proof")
    for arm, proof in receipt["tiny_replay"]["arms"].items():
        require(read_json(root / arm / "full_proof.json") == proof,
                "Embedded replay evidence differs from native proof")
