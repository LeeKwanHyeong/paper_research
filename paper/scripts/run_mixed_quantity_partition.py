#!/usr/bin/env python3
"""Controlled host partition for an already-approved quantity-comparison run.

This wrapper deliberately does not replace the frozen quantity-comparison
runner.  It only verifies a SHA-bound partition contract, then invokes the
frozen runner's public helpers for an arm explicitly assigned to this host.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import tempfile
import subprocess
import sys
import time

SCHEMA = "mixed_quantity_partition_v1"
PARENT_SHA256 = "fe5f421cfc81486140119d2ed7e0eef67502c22b27c2e59bbdc40addd6c00e87"
APPROVAL_SCOPE = "manual_mixed_quantity_handoff_5090_resume_and_fresh_instacart_5080"
LIMITS = {"qualification_seconds": 900, "calibration_seconds_per_dataset": 600,
          "output_bytes_per_host": 1024 ** 3, "per_file_bytes": 64 * 1024 ** 2,
          "minimum_free_bytes": 5 * 1024 ** 3, "max_concurrent_gpu_jobs_per_host": 1}
CASES = ("B_log_original", "mixed_original", "mixed_matched_original")
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
HOST_DATASET = {"instacart_5080": "insta_market_basket", "continuation_5090": None}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _key(arm):
    return (arm["dataset_id"], arm["case"])


def _parent_map(parent):
    return {data["dataset_id"]: data for data in parent["datasets"]}


def _embedded_parent(contract, parent=None):
    embedded = contract.get("parent", {}).get("contract")
    if parent is None:
        require(isinstance(embedded, dict), "Embedded parent contract required")
        return embedded
    if embedded is not None:
        require(embedded == parent, "Embedded and supplied parent contracts differ")
    return parent


def validate_contract(contract, parent=None, *, wrapper_path=None, now=None):
    """Read-only authority, original deadline, assignment, and host validation."""
    parent = _embedded_parent(contract, parent)
    require(contract["schema"] == SCHEMA and contract["status"] == "frozen_pending_explicit_approval", "Partition schema/status mismatch")
    reference = contract["parent"]
    require(reference["contract_sha256"] == sha_json(parent) == PARENT_SHA256, "Pinned parent contract SHA mismatch")
    require(reference["source_files_sha256"] == parent["source"]["files_sha256"], "Parent source identity mismatch")
    original = reference["execution_receipt"]
    require(original["contract_sha256"] == PARENT_SHA256
            and original["started_at_unix"] == reference["started_at_unix"]
            and original["deadline_unix"] == reference["deadline_unix"] == contract["deadline_unix"]
            and original["started_at_unix"] + parent["limits"]["max_wall_seconds"] == contract["deadline_unix"],
            "Original qualification start/deadline changed")
    require(math.isfinite(contract["deadline_unix"]), "Finite inherited deadline required")
    if now is not None:
        require(now < contract["deadline_unix"], "Inherited suite deadline already reached")
    require(contract["limits"] == LIMITS, "Partition limits mismatch")
    require(contract["policy"] == {"automatic_retry": False, "automatic_resume": False,
            "held_out": False, "budget_extension": False, "cpu_fallback": False}, "Partition policy mismatch")
    receipt = reference["qualification_receipt"]
    require(sha_json(receipt) == reference["qualification_receipt_sha256"]
            and receipt.get("passed") is True and receipt.get("qualifies_cuda") is True
            and receipt["execution_contract_sha256"] == PARENT_SHA256
            and receipt["runtime"] == parent["runtime_expected"], "Parent CUDA qualification mismatch")
    hosts = contract["hosts"]
    require(set(hosts) == {"continuation_5090", "instacart_5080"}, "Exactly two host roles required")
    for role, alias, python, tmux, gpu in (
        ("continuation_5090", "5090", "/opt/miniconda3/envs/ai_env/bin/python3.12", "/opt/miniconda3/envs/ai_env/bin/tmux", "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"),
        ("instacart_5080", "5080", "/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12", "/usr/bin/tmux", "GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6"),
    ):
        host = hosts[role]
        require(host["server_alias"] == alias and host["python"] == python and host["tmux_binary"] == tmux,
                "Pinned host server/Python/tmux required")
        require(host["runtime_expected"]["device"] == "cuda:0" and host["runtime_expected"]["gpu"]["uuid"] == gpu,
                "Pinned host runtime/GPU required")
        for field in ("source_root", "output_dir", "wrapper_path"):
            path = Path(host[field])
            require(path.is_absolute() and ".." not in path.parts and str(path) == host[field], "Canonical absolute host paths required")
        require(bool(host["tmux_session"]), "Pinned tmux session required")
        source, output, wrapper = (Path(host[k]) for k in ("source_root", "output_dir", "wrapper_path"))
        require(not output.is_relative_to(source) and not source.is_relative_to(output)
                and not wrapper.is_relative_to(source) and not wrapper.is_relative_to(output), "Separate source/output/wrapper paths required")
        old_output = Path(parent["execution"]["output_dir"])
        require(not output.is_relative_to(old_output) and not old_output.is_relative_to(output), "Original result directory must be preserved")
    require(hosts["continuation_5090"]["qualification"] == "reused_parent_receipt"
            and hosts["continuation_5090"]["runtime_expected"] == parent["runtime_expected"], "5090 must reuse original qualified runtime")
    require(hosts["instacart_5080"]["qualification"] == "new_cuda_required", "5080 requires fresh CUDA qualification")
    arms = contract["arms"]
    require(len(arms) == 9 and [_key(arm) for arm in arms] == [(d, c) for d in DATASETS for c in CASES], "Exactly nine ordered unique arms required")
    for arm in arms:
        expected = "instacart_5080" if arm["dataset_id"] == "insta_market_basket" else "continuation_5090"
        require(arm["host"] == expected and arm["mode_policy"] == ("fresh" if expected == "instacart_5080" else "inherit_or_fresh"),
                "Fixed dataset ownership/mode policy mismatch")
    require(len(contract["wrapper"]["sha256"]) == 64, "Wrapper identity required")
    if wrapper_path is not None:
        require(contract["wrapper"]["sha256"] == sha_file(wrapper_path), "Partition wrapper SHA mismatch")
    return {"status": "validated_metadata_only", "parent_contract_sha256": PARENT_SHA256, "arms": 9, "gpu_started": False}


def validate_approval(contract, approval):
    require(approval.get("status") == "approved_by_user", "Explicit partition approval required")
    require(approval.get("partition_contract_sha256") == sha_json(contract), "Partition approval SHA mismatch")
    require(approval.get("scope") == APPROVAL_SCOPE, "Partition approval scope mismatch")
    require(bool(approval.get("user_instruction")), "Approval must record the user instruction")


class Budget:
    def __init__(self, contract, output, *, clock=time.time, monotonic=time.monotonic):
        self.contract, self.output, self.clock, self.monotonic = contract, Path(output), clock, monotonic
        self.deadline_mono = monotonic() + max(0.0, contract["deadline_unix"] - clock())
        self.last_scan = -math.inf

    def __call__(self, stage=None):
        if self.clock() >= self.contract["deadline_unix"] or self.monotonic() >= self.deadline_mono:
            raise TimeoutError("Inherited suite deadline reached")
        if stage == "before_epoch_commit" or self.clock() - self.last_scan >= 10:
            self.last_scan = self.clock()
            used = 0
            for path in self.output.rglob("*"):
                try:
                    info = path.stat()
                except FileNotFoundError:
                    continue  # An atomic checkpoint rename can win this race.
                if stat.S_ISREG(info.st_mode):
                    require(info.st_size <= self.contract["limits"]["per_file_bytes"], "Individual artifact file ceiling reached")
                    used += info.st_size
            if used >= self.contract["limits"]["output_bytes_per_host"] - 64 * 1024 ** 2:
                raise RuntimeError("Partition artifact storage ceiling reached")


class CalibrationDeadline:
    """Monotonic watchdog survives hung worker operations and wall-clock rollback."""

    def __init__(self, *, clock=time.time, monotonic=time.monotonic):
        self.clock, self.monotonic = clock, monotonic
        self.phase, self.deadline = None, None

    def remaining(self, status):
        if status.get("status") != "calibrating":
            return math.inf
        phase = status["dataset_id"], status["phase_started_at_unix"]
        if phase != self.phase:
            elapsed = max(0., self.clock() - status["phase_started_at_unix"])
            self.deadline = self.monotonic() + max(0., LIMITS["calibration_seconds_per_dataset"] - elapsed)
            self.phase = phase
        return self.deadline - self.monotonic()


def assigned_arms(contract, role):
    require(role in contract["hosts"], "Unknown host role")
    return [arm for arm in contract["arms"] if arm["host"] == role]


def assert_launch_gates(contract, role, *, disk_free, gpu_pids, executable):
    host = contract["hosts"][role]
    require(Path(sys.executable).resolve() == Path(executable).resolve() == Path(host["python"]).resolve(), "Pinned host Python required")
    require(disk_free >= contract["limits"]["minimum_free_bytes"], "Insufficient free disk")
    require(not gpu_pids, "GPU already has a compute process")


def partition_identity(parent, contract, role, runtime):
    """Preserve parent training identity while recording partition authority externally."""
    require(runtime == contract["hosts"][role]["runtime_expected"], "Actual host runtime differs from partition contract")
    return {"execution_contract_sha256": sha_json(parent), "runtime": runtime,
            "source": parent["source"]}


def verify_source(source_root, parent):
    """Confirm the wrapper will import precisely the frozen 70-file closure."""
    source_root = Path(source_root).resolve()
    expected = parent["source"]["files"]
    require(len(expected) == 70 and sha_json(expected) == parent["source"]["files_sha256"], "Exactly 70 frozen source files required")
    require(all((source_root / name).resolve().is_relative_to(source_root) for name in expected), "Frozen source escapes root")
    observed = {name: sha_file(source_root / name) for name in expected}
    require(observed == expected, "Frozen source closure mismatch")
    for directory in parent["source"].get("required_directories", []):
        require((source_root / directory).is_dir(), "Frozen source sentinel missing")


def validate_handoff(contract, handoff):
    require(handoff.get("schema") == "mixed_quantity_handoff_v1", "Handoff schema mismatch")
    require(handoff.get("partition_contract_sha256") == sha_json(contract), "Handoff partition SHA mismatch")
    require(handoff.get("approved_transfer") is True, "Approved transfer receipt required")
    require(handoff.get("observed_old_process_dead") is True, "Original process ownership not released")
    pids = handoff.get("original_process_ids")
    require(isinstance(pids, list) and pids and len(set(pids)) == len(pids)
            and all(type(pid) is int and pid > 1 for pid in pids), "Original process IDs required")
    original = Path(contract["parent"]["contract"]["execution"]["output_dir"])
    files = handoff.get("persisted_file_hashes")
    require(isinstance(files, dict), "Persisted file hashes required")
    for path, digest in files.items():
        require(Path(path).is_absolute() and ".." not in Path(path).parts and str(Path(path)) == path
                and Path(path).is_relative_to(original) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest), "Invalid inherited file identity")
    arms = handoff.get("arms", [])
    require([_key(a) for a in arms] == [(d, c) for d in DATASETS for c in CASES], "Handoff must assign nine ordered arms")
    allowed = set()
    for arm in arms:
        require(arm["mode"] in {"fresh", "resume", "reuse"}, "Invalid handoff arm mode")
        source = original / arm["dataset_id"] / arm["case"]
        inherited = arm["mode"] != "fresh"
        require(arm["dataset_id"] != "insta_market_basket" or not inherited, "5080 Instacart must start fresh")
        if inherited:
            require(arm.get("source_dir") == str(source), "Inherited source directory mismatch")
            owned = {p for p in files if Path(p).is_relative_to(source)}
            require(str(source / "contract.json") in owned and str(source / "last_epoch_state.pt") in owned,
                    "Inherited epoch checkpoint and contract required")
            allowed.update(owned)
        else:
            require(arm.get("source_dir") is None and not any(Path(p).is_relative_to(source) for p in files), "Fresh arm cannot inherit files")
    calibrations = handoff.get("dataset_calibrations", {})
    require(set(calibrations) == set(DATASETS), "Three dataset calibration modes required")
    for dataset, calibration in calibrations.items():
        require(calibration["mode"] in {"fresh", "reuse"}, "Invalid calibration mode")
        inherited = any(a["dataset_id"] == dataset and a["mode"] != "fresh" for a in arms)
        require(not inherited or calibration["mode"] == "reuse", "Inherited arms require original calibration")
        if calibration["mode"] == "reuse":
            path = str(original / dataset / "calibration.json")
            require(dataset != "insta_market_basket" and calibration["source_path"] == path
                    and files.get(path) == calibration["sha256"], "Original calibration file identity mismatch")
            allowed.add(path)
        else:
            require(calibration.get("source_path") is None and calibration.get("sha256") is None,
                    "Fresh calibration cannot inherit a receipt")
    require(set(files) == allowed, "Unassigned inherited files forbidden")


def verify_inherited_files(contract, handoff):
    """Run only on 5090 after the old process tree has released ownership."""
    validate_handoff(contract, handoff)
    for pid in handoff["original_process_ids"]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise ValueError("Original process still exists; no transfer permitted")
    expected = handoff["persisted_file_hashes"]
    observed = {}
    original = Path(contract["parent"]["contract"]["execution"]["output_dir"])
    for arm in handoff["arms"]:
        source = original / arm["dataset_id"] / arm["case"]
        if arm["mode"] == "fresh":
            require(not source.exists() or not any(source.iterdir()), "Fresh arm has unaccounted original progress")
            continue
        require(source.is_dir() and not source.is_symlink(), "Original arm directory missing or symlinked")
        for path in source.rglob("*"):
            require(not path.is_symlink(), "Inherited symlinks forbidden")
            if path.is_file():
                require(path.stat().st_size <= LIMITS["per_file_bytes"], "Inherited file ceiling exceeded")
                observed[str(path)] = sha_file(path)
    for dataset, calibration in handoff["dataset_calibrations"].items():
        path = original / dataset / "calibration.json"
        if calibration["mode"] == "fresh":
            require(not path.exists(), "Existing calibration must be reused")
        else:
            require(path.is_file() and not path.is_symlink() and path.stat().st_size <= LIMITS["per_file_bytes"], "Inherited calibration missing or oversized")
            observed[str(path)] = sha_file(path)
    require(observed == expected, "Inherited file set or hash changed after stop")


def effective_parent_contract(parent, contract, role):
    """5080 has a native runtime; 5090 keeps the original training identity."""
    effective = copy.deepcopy(parent)
    if role == "continuation_5090":
        return effective
    host = contract["hosts"][role]
    effective["runtime_expected"] = host["runtime_expected"]
    effective["execution"] = {"server_alias": host.get("server_alias"), "python": host["python"],
                              "source_dir": host["source_root"], "output_dir": host["output_dir"],
                              "tmux_session": host["tmux_session"], "tmux_binary": host["tmux_binary"]}
    return effective


def _frozen_runner(source_root):
    source_root = str(Path(source_root).resolve())
    require(source_root not in sys.path or sys.path.index(source_root) == 0, "Frozen source must lead import path")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    module = importlib.import_module("paper.scripts.run_mixed_quantity_comparison")
    require(Path(module.__file__).resolve().is_relative_to(Path(source_root)), "Frozen runner import escapes source root")
    return module


def _child_command(mode, authority_path, *, case=None, stage=None):
    command = [sys.executable, str(Path(__file__).resolve()), mode,
               "--authority", str(authority_path)]
    if case is not None:
        command.extend(["--case", case, "--stage", stage])
    return command


def _qualification(contract, parent, role, frozen, effective, authority_path):
    host = contract["hosts"][role]
    if role == "continuation_5090":
        receipt = contract["parent"]["qualification_receipt"]
        require(sha_json(receipt) == contract["parent"]["qualification_receipt_sha256"], "Parent qualification receipt SHA mismatch")
        require(receipt.get("passed") is True and receipt.get("qualifies_cuda") is True, "Parent CUDA qualification missing")
        require(receipt.get("execution_contract_sha256") == sha_json(parent), "Parent qualification authority mismatch")
        require(frozen.configured_runtime(parent, "cuda:0") == receipt["runtime"] == host["runtime_expected"], "Parent runtime drift")
        return receipt
    # Every child enters this wrapper's authority gates, never the legacy CLI's
    # approval for twelve fresh arms. The children share the worker process group.
    runtime = frozen.configured_runtime(effective, "cuda:0")
    output = Path(host["output_dir"])
    began = time.monotonic()
    budget = Budget(contract, output)
    def child(mode, **kwargs):
        budget()
        remaining = min(contract["limits"]["qualification_seconds"] - (time.monotonic() - began),
                        budget.deadline_mono - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("CUDA qualification ceiling reached")
        subprocess.run(_child_command(mode, authority_path, **kwargs), check=True, timeout=remaining)
    summaries = []
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    for case in CASES:
        for stage in ("full", "first", "resume"):
            child("_probe", case=case, stage=stage)
        folder = output / "qualification" / case
        full, split = read_json(folder / "full/summary.json"), read_json(folder / "split/summary.json")
        frozen.base.compare_replay(full, split)
        # Only the newly generated synthetic qualification checkpoints are read.
        a = torch_load_checkpoint(folder / "full/last_epoch_state.pt", map_location="cpu")
        b = torch_load_checkpoint(folder / "split/last_epoch_state.pt", map_location="cpu")
        for key in ("optimizer_state_sha256", "rng_state_sha256", "model_state_sha256"):
            require(a[key] == b[key], f"Synthetic replay mismatch: {key}")
        summaries.append(full)
    frozen.audit_pairs(summaries)
    child("_shapes")
    budget()
    require(time.monotonic() - began < contract["limits"]["qualification_seconds"], "Qualification ceiling reached")
    require(frozen.configured_runtime(effective, "cuda:0") == runtime, "Runtime drift after qualification")
    frozen.validate_contract(parent)
    return {"passed": True, "qualifies_cuda": True, "runtime": runtime,
            "execution_contract_sha256": sha_json(effective),
            "elapsed_seconds": time.monotonic() - began,
            "checks": ["fresh_process_exact_resume", "optimizer_and_rng_identity", "paired_batches",
                       "production_shapes", "B_objective_identity", "target_padding_causality"]}


def training_identity(parent, contract, role, data, runtime):
    effective = effective_parent_contract(parent, contract, role)
    return {**partition_identity(effective, contract, role, runtime),
            "data": data["inherited_data_identity"], "dataset_id": data["dataset_id"]}


def audit_arm(summary, arm_contract, data, objective, identity):
    require(arm_contract["identity"] == identity, "Arm training identity mismatch")
    require(summary["contract_sha256"] == sha_json(arm_contract), "Arm summary contract SHA mismatch")
    require(summary["status"] == "complete" and summary["epochs_completed"] == summary["epochs_budget"] == 120,
            "Arm is not complete at 120 epochs")
    require(summary["global_step"] == data["expected_global_steps"], "Arm global step mismatch")
    require(summary["condition"] == arm_contract["condition"]
            and summary["condition"]["case"] == "B_log_original"
            and summary["condition"]["mixed_objective"] == objective.to_dict(), "Arm condition mismatch")
    require(summary["evaluation_scope"] == "validation_only" and summary["held_out_test_evaluated"] is False,
            "Arm evaluation scope mismatch")
    require(len(summary["history"]) == 120, "Arm history length mismatch")
    populations = data["inherited_data_identity"]["populations"]
    for epoch, row in enumerate(summary["history"], 1):
        require(row["epoch"] == epoch and row["global_step"] == epoch * (data["expected_global_steps"] // 120), "Arm history step mismatch")
        for split in ("train", "validation"):
            require(row[f"{split}_count"] == populations[split]["target_count"], "Arm population mismatch")


def copy_and_repair_arm(arm, target, handoff, *, identity, objective, engine):
    """Copy hash-bound files, then repair only the new copy from its epoch commit."""
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    source, target = Path(arm["source_dir"]), Path(target)
    require(arm["mode"] in {"resume", "reuse"} and not target.exists(), "Fresh inherited target required")
    files = {p: digest for p, digest in handoff["persisted_file_hashes"].items() if Path(p).is_relative_to(source)}
    observed = {}
    for path in source.rglob("*"):
        require(not path.is_symlink(), "Inherited symlinks forbidden")
        if path.is_file():
            require(path.stat().st_size <= LIMITS["per_file_bytes"], "Inherited file ceiling exceeded")
            observed[str(path)] = sha_file(path)
    require(files and observed == files, "Inherited file set or hash mismatch")
    shutil.copytree(source, target)
    require(all(sha_file(target / Path(path).relative_to(source)) == digest for path, digest in files.items()), "Copied file hash mismatch")
    saved_contract = read_json(target / "contract.json")
    require(saved_contract["identity"] == identity and saved_contract["condition"]["mixed_objective"] == objective.to_dict(), "Inherited objective/identity mismatch")
    payload = torch_load_checkpoint(target / "last_epoch_state.pt", map_location="cpu")
    engine._validate_payload(payload, saved_contract)
    require(saved_contract["epochs"] == 120 and (payload["epoch"] == 120 if arm["mode"] == "reuse" else payload["epoch"] < 120), "Handoff mode disagrees with committed epoch")
    summary = engine._publish(payload, target)
    timing_path = target / "timing.json"
    timing_epochs = []
    if timing_path.exists():
        timing = read_json(timing_path)
        require(timing.get("contract_sha256") == payload["contract_sha256"] and isinstance(timing.get("epochs"), list), "Inherited timing identity mismatch")
        timing["epochs"] = [row for row in timing["epochs"] if type(row.get("epoch")) is int and 1 <= row["epoch"] <= payload["epoch"]]
        timing_epochs = [row["epoch"] for row in timing["epochs"]]
        require(len(timing_epochs) == len(set(timing_epochs)), "Duplicated inherited timing epochs")
        write_json(timing_path, timing)
    write_json(target / "handoff_repair_receipt.json", {
        "source_dir": str(source), "inherited_file_hashes": files,
        "committed_epoch": payload["epoch"], "global_step": payload["global_step"],
        "model_state_sha256": payload["model_state_sha256"], "mode": arm["mode"],
        "repaired_summary_history_selectors_from_epoch_checkpoint": True,
        "timing_epochs_missing": sorted(set(range(1, payload["epoch"] + 1)) - set(timing_epochs)),
        "original_files_modified": False})
    return summary


def load_calibration(calibration, *, target, identity, statistics, initial_state_sha256, primitives):
    """Reuse the exact already-frozen dataset coefficients without recomputation."""
    source = Path(calibration["source_path"])
    require(sha_file(source) == calibration["sha256"], "Inherited calibration file hash mismatch")
    receipt = read_json(source)
    primitives.objectives_from_calibration(receipt)
    require(receipt["identity"] == identity and receipt["statistics"] == {"mu": statistics.mu, "raw_scale": statistics.raw_scale}
            and receipt["initial_model_state_sha256"] == initial_state_sha256, "Inherited calibration initialization/statistics/identity mismatch")
    target = Path(target)
    require(not target.exists(), "Fresh calibration target required")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    require(sha_file(target) == calibration["sha256"], "Copied calibration hash mismatch")
    return receipt


def _run_selected_arms(contract, parent, role, source_root, output, receipt, handoff):
    frozen = _frozen_runner(source_root)
    data_helper = importlib.import_module("paper.scripts.quantity_comparison_data")
    engine = importlib.import_module("paper.scripts.quantity_comparison_engine")
    quantity = importlib.import_module("paper.scripts.quantity_objective_comparison")
    primitives = importlib.import_module("paper.scripts.mixed_quantity_objective")
    acceptance = importlib.import_module("paper.scripts.mixed_quantity_acceptance")
    diagnostic = importlib.import_module("paper.scripts.run_time_quantity_diagnostic")
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    budget, summaries = Budget(contract, output), {}
    effective = effective_parent_contract(parent, contract, role)
    modes = {_key(arm): arm for arm in handoff["arms"]}
    assigned_datasets = list(dict.fromkeys(arm["dataset_id"] for arm in assigned_arms(contract, role)))
    for dataset_id in assigned_datasets:
        budget()
        data = _parent_map(parent)[dataset_id]
        identity = training_identity(parent, contract, role, data, receipt["runtime"])
        calibration_mode = handoff["dataset_calibrations"][dataset_id]
        write_json(Path(output) / "status.json", {"status": "calibrating" if calibration_mode["mode"] == "fresh" else "loading_calibration",
                   "host_role": role, "dataset_id": dataset_id, "phase_started_at_unix": time.time(),
                   "deadline_unix": contract["deadline_unix"]})
        calibration_started = time.monotonic()
        frame, metadata = data_helper.prepare_quantity_comparison_data(data)
        metadata = frozen.base.bind_frozen_statistics(data, metadata)
        model, train, validation, _ = diagnostic.build_arm_inputs({**data, "seed": 42}, frame, metadata)
        statistics = quantity.QuantityStatistics(metadata["train_log_mean"], metadata["raw_scale"])
        initial_hash = canonical_state_dict_sha256(model.state_dict())
        calibration_path = Path(output) / dataset_id / "calibration.json"
        if calibration_mode["mode"] == "reuse":
            calibration = load_calibration(calibration_mode, target=calibration_path, identity=identity,
                                           statistics=statistics, initial_state_sha256=initial_hash, primitives=primitives)
        else:
            def calibration_budget(stage=None):
                budget(stage)
                require(time.monotonic() - calibration_started < LIMITS["calibration_seconds_per_dataset"], "Calibration deadline reached")
            calibration_budget()
            model.to("cuda:0")
            calibration = primitives.calibrate_mixed_objective(model, train.dataset, statistics=statistics, identity=identity,
                                    device="cuda:0", budget_check=calibration_budget)
            calibration_budget()
            write_json(calibration_path, calibration)
        objectives = primitives.objectives_from_calibration(calibration)
        del model, train, validation
        dataset_summaries = []
        for objective in objectives:
            arm = modes[(dataset_id, objective.name)]
            budget()
            require(frozen.configured_runtime(effective, "cuda:0") == receipt["runtime"], "Runtime drift before arm")
            target = Path(output) / dataset_id / objective.name
            model, train, validation, _ = diagnostic.build_arm_inputs({**data, "seed": 42}, frame, metadata)
            require(canonical_state_dict_sha256(model.state_dict()) == calibration["initial_model_state_sha256"], "Calibration/arm initialization mismatch")
            if arm["mode"] != "fresh":
                require(role == "continuation_5090", "Cross-host checkpoint continuation forbidden")
                summary = copy_and_repair_arm(arm, target, handoff, identity=identity, objective=objective, engine=engine)
            write_json(Path(output) / "status.json", {"status": "running", "host_role": role,
                       "dataset_id": dataset_id, "case": objective.name, "mode": arm["mode"],
                       "deadline_unix": contract["deadline_unix"]})
            if arm["mode"] != "reuse":
                summary = engine.run_case(model=model, train_loader=train, validation_loader=validation,
                        case=quantity.QuantityCase.B_LOG_ORIGINAL, mixed_objective=objective, statistics=statistics,
                        output_dir=target, epochs=120, seed=42, identity=identity, device="cuda:0",
                        lr=.001, weight_decay=.01, grad_clip=1., resume=arm["mode"] == "resume", budget_check=budget, cuda_qualification=receipt)
            audit_arm(summary, read_json(target / "contract.json"), data, objective, identity)
            dataset_summaries.append(summary)
            summaries[(dataset_id, objective.name)] = summary
            write_json(Path(output) / "status.json", {"status": "validating_checkpoints", "host_role": role,
                       "dataset_id": dataset_id, "case": objective.name, "deadline_unix": contract["deadline_unix"]})
            evaluation = acceptance.evaluate_checkpoints(model=model, validation_loader=validation, statistics=statistics,
                           objective=objective, arm_dir=target, quantity_boundaries=data["quantity_boundaries_all_train_rows"],
                           device="cuda:0", budget_check=budget)
            write_json(target / "validation_diagnosis.json", evaluation)
            del model, train, validation
            gc.collect()
        write_json(Path(output) / dataset_id / "paired_comparison.json", frozen.audit_pairs(dataset_summaries))
        del frame
        gc.collect()
    return summaries


def _authority_checks(authority, *, now=None):
    contract, parent, role = authority["contract"], authority["parent"], authority["role"]
    validate_contract(contract, parent, wrapper_path=__file__, now=time.time() if now is None else now)
    validate_approval(contract, authority["approval"])
    validate_handoff(contract, authority["handoff"])
    require(role in contract["hosts"], "Unknown host role")
    host = contract["hosts"][role]
    require(Path(authority["source_root"]).resolve() == Path(host["source_root"]).resolve(), "Pinned host source root required")
    require(Path(authority["output"]).resolve() == Path(host["output_dir"]).resolve(), "Pinned host output required")
    require(Path(sys.executable).resolve() == Path(host["python"]).resolve(), "Pinned host Python required")
    require(Path(__file__).resolve() == Path(host["wrapper_path"]).resolve(), "Pinned host wrapper path required")
    require(authority["deadline_unix"] == contract["deadline_unix"], "Child deadline changed")
    verify_source(authority["source_root"], parent)
    return contract, parent, role


def _apply_environment(host):
    for key, value in host["runtime_expected"]["environment"].items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _loaded_frozen(authority):
    contract, parent, role = _authority_checks(authority)
    _apply_environment(contract["hosts"][role])
    frozen = _frozen_runner(authority["source_root"])
    require(Path(frozen.ROOT).resolve() == Path(authority["source_root"]).resolve(), "Frozen runner root mismatch")
    frozen.validate_contract(parent)  # AST closure audit, not only file-list hashes.
    return frozen


def _gpu_pids(host):
    output = subprocess.check_output(["nvidia-smi", "--id=" + host["runtime_expected"]["gpu"]["uuid"],
                "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    return [line.strip() for line in output.splitlines() if line.strip()]


def execute(contract, approval, *, source_root, output, role, handoff, parent=None):
    parent = _embedded_parent(contract, parent)
    authority = {"contract": contract, "parent": parent, "role": role,
                 "approval": approval, "handoff": handoff, "source_root": str(source_root),
                 "output": str(output), "deadline_unix": contract["deadline_unix"]}
    _authority_checks(authority)
    if role == "continuation_5090":
        verify_inherited_files(contract, handoff)
    _loaded_frozen(authority)
    output = Path(output)
    require(not output.exists(), "Fresh partition output required; automatic resume is forbidden")
    host = contract["hosts"][role]
    require(bool(os.environ.get("TMUX")), "Partition execution requires the pinned tmux session")
    session = subprocess.check_output([host["tmux_binary"], "display-message", "-p", "#S"], text=True, timeout=15).strip()
    require(session == host["tmux_session"], "Unexpected partition tmux session")
    import fcntl
    with (output.parent / "mixed_quantity_gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_launch_gates(contract, role, disk_free=shutil.disk_usage(output.parent).free,
                           gpu_pids=_gpu_pids(host), executable=host["python"])
        output.mkdir()
        authority["supervisor_pid"] = os.getpid()
        authority["started_at_unix"] = time.time()
        authority_path = output / "partition_authority.json"
        write_json(authority_path, authority)
        write_json(output / "effective_parent_contract.json", effective_parent_contract(parent, contract, role))
        process = None
        handlers = {}
        def interrupted(signum, frame):
            raise InterruptedError(f"Partition supervisor received signal {signum}")
        try:
            for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                handlers[signum] = signal.signal(signum, interrupted)
            process = subprocess.Popen(_child_command("_worker", authority_path), start_new_session=True)
            budget = Budget(contract, output)
            qualification_deadline = time.monotonic() + contract["limits"]["qualification_seconds"]
            calibration_deadline = CalibrationDeadline()
            while True:
                budget()
                if role == "instacart_5080" and not (output / "qualification/receipt.json").exists():
                    if time.monotonic() >= qualification_deadline:
                        raise TimeoutError("CUDA qualification ceiling reached")
                phase_remaining = math.inf
                status_path = output / "status.json"
                if status_path.exists():
                    phase_remaining = calibration_deadline.remaining(read_json(status_path))
                    if phase_remaining <= 0:
                        raise TimeoutError("Calibration ceiling reached")
                try:
                    code = process.wait(timeout=min(1., max(.001, budget.deadline_mono - time.monotonic()), max(.001, phase_remaining)))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(code == 0, f"Partition failed with exit code {code}; no retry")
            result = read_json(output / "partition_receipt.json")
            require(result["status"] == "host_complete" and result["partition_contract_sha256"] == sha_json(contract),
                    "Worker completion receipt missing or mismatched")
            return result
        except BaseException as error:
            stop_owned_process_group(process)
            previous = read_json(output / "status.json") if (output / "status.json").exists() else {}
            write_json(output / "status.json", {**previous, "status": "timeout" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "failed",
                       "error": str(error), "automatic_retry": False, "deadline_unix": contract["deadline_unix"]})
            raise
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def child_main(mode, authority_path, *, case=None, stage=None):
    authority = read_json(authority_path)
    contract, parent, role = _authority_checks(authority)
    require(Path(authority_path).resolve() == Path(authority["output"]).resolve() / "partition_authority.json", "Pinned child authority path required")
    # Only descendants of the live supervisor may consume an execution receipt.
    os.kill(authority["supervisor_pid"], 0)
    owner_path = Path(authority["output"]) / "worker_owner.json"
    if mode == "_worker":
        require(os.getppid() == authority["supervisor_pid"], "Worker must descend from supervisor")
        require(os.getpgid(0) == os.getpid(), "Worker requires its own process group")
        require(not owner_path.exists(), "Worker already claimed; retry forbidden")
        write_json(owner_path, {"pid": os.getpid(), "partition_contract_sha256": sha_json(contract)})
    else:
        owner = read_json(owner_path)
        require(owner["partition_contract_sha256"] == sha_json(contract), "Probe worker authority mismatch")
        require(os.getppid() == owner["pid"] and os.getpgid(0) == owner["pid"], "Probe must descend from owned worker")
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (contract["limits"]["per_file_bytes"], contract["limits"]["per_file_bytes"]))
    frozen = _loaded_frozen(authority)
    effective = effective_parent_contract(parent, contract, role)
    started = contract["deadline_unix"] - parent["limits"]["max_wall_seconds"]
    if mode in {"_probe", "_shapes"}:
        require(role == "instacart_5080", "Only 5080 may run fresh CUDA qualification")
        if mode == "_probe":
            require(case in CASES and stage in {"full", "first", "resume"}, "Invalid synthetic probe stage")
            return frozen.probe_worker(effective, case, stage, started, "cuda:0")
        frozen.configured_runtime(effective, "cuda:0")
        # Production shape checks use synthetic inputs for the assigned dataset.
        shape_contract = copy.deepcopy(effective)
        shape_contract["datasets"] = [d for d in effective["datasets"] if d["dataset_id"] == HOST_DATASET[role]]
        frozen.shape_checks(shape_contract, started, "cuda:0")
        return {"status": "passed"}
    require(mode == "_worker", "Unknown child mode")
    receipt = _qualification(contract, parent, role, frozen, effective, authority_path)
    write_json(Path(authority["output"]) / "qualification/receipt.json", receipt)
    summaries = _run_selected_arms(contract, parent, role, authority["source_root"], authority["output"], receipt, authority["handoff"])
    result = {"status": "host_complete", "partition_contract_sha256": sha_json(contract),
              "global_suite_complete": False, "parent_contract_sha256": sha_json(parent), "role": role,
              "deadline_unix": contract["deadline_unix"], "qualification": receipt,
              "completed_arms": [{"dataset_id": key[0], "case": key[1]} for key in summaries]}
    write_json(Path(authority["output"]) / "partition_receipt.json", result)
    write_json(Path(authority["output"]) / "status.json", result)
    return result


def stop_owned_process_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "execute", "_worker", "_probe", "_shapes"))
    parser.add_argument("--contract")
    parser.add_argument("--authority")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--stage", choices=("full", "first", "resume"))
    parser.add_argument("--parent-contract")
    parser.add_argument("--approval")
    parser.add_argument("--source-root")
    parser.add_argument("--output")
    parser.add_argument("--role", "--host-role", dest="host_role", choices=("instacart_5080", "continuation_5090"))
    parser.add_argument("--handoff")
    args = parser.parse_args(argv)
    if args.mode.startswith("_"):
        require(args.authority, "Child partition authority required")
        child_main(args.mode, args.authority, case=args.case, stage=args.stage)
        return 0
    require(args.contract, "Partition contract required")
    contract = read_json(args.contract)
    parent = read_json(args.parent_contract) if args.parent_contract else None
    parent = _embedded_parent(contract, parent)
    result = validate_contract(contract, parent, wrapper_path=__file__, now=time.time())
    if args.mode == "execute":
        require(args.approval and args.host_role and args.handoff,
                "Execute requires approval, host role, and handoff receipt")
        host = contract["hosts"][args.host_role]
        result = execute(contract, read_json(args.approval), source_root=args.source_root or host["source_root"], output=args.output or host["output_dir"],
                         role=args.host_role, handoff=read_json(args.handoff), parent=parent)
    elif args.approval:
        validate_approval(contract, read_json(args.approval))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
