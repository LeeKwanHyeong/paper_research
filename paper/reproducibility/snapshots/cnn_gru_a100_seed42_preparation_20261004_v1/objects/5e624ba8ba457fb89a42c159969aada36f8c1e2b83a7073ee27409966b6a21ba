"""Frozen contracts and identities for the paired observed-slot comparison.

Only standard-library work runs at import time. CUDA initialization follows
source/host validation and an explicit idle-device check in each entrypoint.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "observed_slot_parallel_execution_v1"
DESIGN = "paper/contracts/hard_lmm_observed_slot_memory_v1.json"
INHERITED = "paper/contracts/quantity_comparison_execution_5090_v3.json"
ASSIGNMENTS = {"5090": ["insta_market_basket"], "5080": ["intermittent_frozen_5000", "yellow_trip_hourly"]}
LIMITS = {"qualification_seconds": 900, "total_seconds": 86400,
          "per_host_output_bytes": 4 * 1024**3, "max_file_bytes": 64 * 1024**2,
          "min_free_bytes": 5 * 1024**3, "max_concurrent_gpu_jobs_per_host": 1}
ENVIRONMENT = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "NVIDIA_TF32_OVERRIDE": "0",
               "CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "PYTHONHASHSEED": "42"}
ENTRYPOINTS = ("paper/scripts/observed_slot_parallel_common.py", "paper/scripts/run_observed_slot_partition.py",
              "paper/scripts/qualify_observed_slot_cuda.py")
POLICY = {"resume": False, "retry": False, "held_out": False, "runtime_mutation": False,
          "additional_arms": False, "additional_seeds": False, "automatic_transfer": False, "scheduler_changes": False}
ACCEPTANCE = {"raw_rmse_ratio_strictly_less_than": 1.0, "mae_ratio_max": 1.01,
              "body_mae_ratio_max": 1.02, "tail_mae_ratio_max": 1.02,
              "legacy_time_loss_increase_max": 0.01, "last30_rmse_mean_ratio_max": 1.05,
              "last30_rmse_std_ratio_max": 1.05}
COST_GATES = {"step_ratio_max": 2.5, "peak_allocated_ratio_max": 2.5,
              "peak_device_fraction_max": 0.8, "parameter_ratio_max": 1.05}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path)
    encoded = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    require(len(encoded) < LIMITS["max_file_bytes"], "JSON file ceiling exceeded")
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as handle:
            handle.write(encoded)
    else:
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(encoded)
        os.replace(temporary, path)


def source_manifest(root=ROOT, *, entrypoints=ENTRYPOINTS, extra_files=(DESIGN, INHERITED)):
    root = Path(root).resolve()
    pending, visited = [root / name for name in entrypoints], set()

    def add(module):
        if module.split(".")[0] not in {"models", "paper", "data_loader", "simple_lab_test", "utils"}:
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
        require(path.is_relative_to(root) and path.is_file(), "Missing or escaped first-party source")
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
    visited.update(root / name for name in extra_files)
    return {str(path.relative_to(root)): sha_file(path) for path in sorted(visited)}


def verify_source(contract):
    actual = source_manifest()
    require(actual == contract["source"]["files"], "Frozen execution source changed")
    require(sha_json(actual) == contract["source"]["files_sha256"], "Source manifest digest mismatch")


def read_contract(path, host, check_source=True):
    contract = json.loads(Path(path).read_text())
    require(contract.get("schema") == SCHEMA, "Wrong slot execution schema")
    require(contract.get("limits") == LIMITS and contract.get("policy") == POLICY, "Execution limits or policy changed")
    require(contract.get("acceptance") == ACCEPTANCE and contract.get("cost_gates") == COST_GATES, "Acceptance gates changed")
    require(contract.get("model") == {"baseline": "titantpp", "candidate": "titantpp_observed_slot_memory"}, "Backbones changed")
    require(contract.get("seed") == 42 and contract.get("epochs") == 120, "Seed/epoch budget changed")
    require(host in ASSIGNMENTS and set(contract["hosts"]) == set(ASSIGNMENTS), "Unknown host")
    require(contract.get("authorization", {}).get("scope") == "fresh_six_arm_same_host_paired_training_after_two_native_cuda_qualifications", "Missing parallel training authorization")
    require(contract["authorization"].get("user_instruction") == "이 작업 관련해서 5080/5090 병렬 실행하자", "Current user authorization differs")
    require(sha_file(ROOT / DESIGN) == contract["design_sha256"] and sha_file(ROOT / INHERITED) == contract["inherited_data_contract_sha256"], "Design/data provenance changed")
    expected = deepcopy(json.loads((ROOT / INHERITED).read_text())["datasets"])
    for dataset in expected:
        dataset["epochs"] = 120
    require(contract["datasets"] == expected, "Frozen datasets/statistics/exposure changed")
    require(contract.get("total_optimizer_steps") == 4544160, "Six-arm optimizer budget changed")
    for alias, assignments in ASSIGNMENTS.items():
        entry = contract["hosts"][alias]
        root = Path(entry["root"])
        require(root.is_absolute() and root.parent == Path("/home/leekwanhyeong/workspace/paper_research_experiment_artifacts")
                and root.name.startswith("observed_slot_seed42_parallel_"), "Unsafe isolated run root")
        require(Path(entry["source_root"]) == root / "source" and Path(entry["output_dir"]) == root / "run", "Wrong source/output isolation")
        require(entry["assigned_datasets"] == assignments and entry["environment"] == ENVIRONMENT, "Host assignment/numerics changed")
        expected_python = "/opt/miniconda3/envs/ai_env/bin/python3.12" if alias == "5090" else "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12"
        expected_uuid = "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f" if alias == "5090" else "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"
        require(entry["python"] == expected_python and entry["gpu_uuid"] == expected_uuid, "Host Python/GPU changed")
    source = contract["source"]
    require(len(source["base_git_revision"]) == 40 and source["files_sha256"] == sha_json(source["files"]), "Malformed source identity")
    if check_source:
        verify_source(contract)
    return contract


def apply_environment(host_spec):
    require(host_spec["environment"] == ENVIRONMENT, "Numerical environment drift")
    os.environ.update(ENVIRONMENT)


def gpu_pids(host_spec):
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    pids = set()
    for row in csv.reader(result.stdout.splitlines()):
        if not row:
            continue
        require(len(row) == 2, "Unknown GPU process inventory")
        if row[0].strip() == host_spec["gpu_uuid"]:
            pids.add(int(row[1]))
    return pids


def runtime_check(contract, host):
    entry = contract["hosts"][host]
    require(Path(sys.executable).resolve() == Path(entry["python"]).resolve(), "Wrong host Python executable")
    require(Path(entry["source_root"]).resolve() == ROOT and Path.cwd().resolve() == ROOT, "Run from isolated pinned source")
    require(all(os.environ.get(k) == v for k, v in ENVIRONMENT.items()), "Environment not set before runtime")
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    from paper.scripts.run_time_quantity_diagnostic import validate_import_origins
    configure_runtime("cuda:0", threads=4)
    runtime = runtime_identity("cuda:0")
    validate_import_origins()
    require(runtime["gpu"]["uuid"] == entry["gpu_uuid"], "Wrong CUDA device")
    for key, value in entry["runtime_expected"].items():
        require(runtime.get(key) == value, f"Native runtime changed: {key}")
    return runtime


def verify_permit(contract, permit, host):
    require(permit.get("schema") == "observed_slot_parallel_start_v1", "Wrong start permit")
    require(permit.get("contract_sha256") == sha_json(contract), "Permit contract mismatch")
    started, deadline = permit.get("started_at_unix"), permit.get("deadline_unix")
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started <= time.time() < deadline and deadline - started == LIMITS["total_seconds"], "Expired or extended deadline")
    require(set(permit.get("qualifications", {})) == set(ASSIGNMENTS), "Both native qualifications required")
    # Embed both independently measured receipts in the common permit. Each
    # host can verify the other host's result without polling it over SSH.
    for alias, qualification in permit["qualifications"].items():
        qualified = qualification.get("receipt", {})
        require(qualification.get("receipt_sha256") == sha_json(qualified), "Embedded qualification digest mismatch")
        require(qualified.get("status") == "passed" and qualified.get("host") == alias
                and qualified.get("contract_sha256") == sha_json(contract)
                and qualified.get("source_files_sha256") == contract["source"]["files_sha256"], "Both native CUDA qualifications must pass")
        require(qualified.get("runtime", {}).get("gpu", {}).get("uuid") == contract["hosts"][alias]["gpu_uuid"], "Qualified GPU differs")
        for key, expected_value in contract["hosts"][alias]["runtime_expected"].items():
            require(qualified["runtime"].get(key) == expected_value, "Qualified native runtime differs")
    entry = permit["qualifications"][host]
    expected = Path(contract["hosts"][host]["root"]) / "qualification" / "receipt.json"
    require(Path(entry["path"]) == expected and sha_file(expected) == entry["sha256"], "Native qualification binding mismatch")
    receipt = json.loads(expected.read_text())
    require(receipt == entry["receipt"], "Embedded/local qualification mismatch")
    require(receipt.get("status") == "passed" and receipt.get("host") == host
            and receipt.get("contract_sha256") == sha_json(contract)
            and receipt.get("source_files_sha256") == contract["source"]["files_sha256"], "Native CUDA qualification did not pass")
