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

SCHEMA = "quantity_comparison_partition_v1"
CASES = ("B_log_original", "raw_original", "log_softplus", "raw_softplus")
DATASETS = ("intermittent_frozen_5000", "yellow_trip_hourly", "insta_market_basket")
HOST_DATASET = {"taxi_5080": "yellow_trip_hourly", "continuation_5090": None}


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
    """Validate authority and ownership without importing torch or a runtime."""
    parent = _embedded_parent(contract, parent)
    require(contract["schema"] == SCHEMA, "Partition schema mismatch")
    require(contract["status"] == "frozen_pending_explicit_approval", "Unexpected partition status")
    parent_ref = contract["parent"]
    require(parent_ref["contract_sha256"] == sha_json(parent), "Parent contract SHA mismatch")
    require(parent_ref["source_files_sha256"] == parent["source"]["files_sha256"], "Parent source identity mismatch")
    require(parent_ref["deadline_unix"] == contract["deadline_unix"], "Partition deadline must equal parent deadline")
    require(math.isfinite(contract["deadline_unix"]), "Finite inherited deadline required")
    require(parent_ref["started_at_unix"] + parent["limits"]["max_wall_seconds"] == contract["deadline_unix"],
            "Deadline must follow original qualification start")
    original = parent_ref["execution_receipt"]
    require(original["contract_sha256"] == sha_json(parent)
            and original["started_at_unix"] == parent_ref["started_at_unix"]
            and original["deadline_unix"] == contract["deadline_unix"], "Original execution deadline or identity changed")
    if now is not None:
        require(now < contract["deadline_unix"], "Inherited suite deadline already reached")
    wrapper = contract["wrapper"]
    if wrapper_path is not None:
        require(wrapper["sha256"] == sha_file(wrapper_path), "Partition wrapper SHA mismatch")
    limits = contract["limits"]
    require(limits == {"qualification_seconds": 900, "output_bytes_per_host": 900 * 1024 ** 2,
                       "per_file_bytes": 64 * 1024 ** 2, "minimum_free_bytes": 5 * 1024 ** 3,
                       "max_concurrent_gpu_jobs_per_host": 1}, "Partition limits mismatch")
    require(contract["policy"] == {"automatic_retry": False, "automatic_resume": False,
            "held_out": False, "budget_extension": False, "cpu_fallback": False}, "Partition policy mismatch")
    hosts = contract["hosts"]
    require(set(hosts) == {"taxi_5080", "continuation_5090"}, "Exactly two host roles required")
    require(hosts["taxi_5080"]["qualification"] == "new_cuda_required", "5080 must qualify CUDA anew")
    require(hosts["continuation_5090"]["qualification"] == "reused_parent_receipt", "5090 must reuse parent receipt")
    require(hosts["continuation_5090"]["runtime_expected"] == parent["runtime_expected"], "5090 runtime must retain parent identity")
    require(hosts["taxi_5080"]["server_alias"] == "5080" and hosts["continuation_5090"]["server_alias"] == "5090",
            "Pinned server aliases required")
    for role, host in hosts.items():
        require(host["role"] == role and host["runtime_expected"]["device"] == "cuda:0", "Host runtime identity missing")
        require(host["output_dir"] and host["source_root"], "Host paths required")
    expected = {(dataset, case) for dataset in DATASETS for case in CASES}
    arms = contract["arms"]
    require(len(arms) == 12 and {_key(arm) for arm in arms} == expected, "Exactly one assignment for every arm required")
    require(len({_key(arm) for arm in arms}) == 12, "Duplicate arm ownership")
    require([_key(a) for a in arms] == [(d, c) for d in DATASETS for c in CASES], "Fixed arm order required")
    counts = {role: sum(arm["host"] == role for arm in arms) for role in hosts}
    require(counts == {"taxi_5080": 4, "continuation_5090": 8}, "Host arm split mismatch")
    parent_data = _parent_map(parent)
    for arm in arms:
        require(arm["host"] in hosts and arm["mode"] in {"fresh", "resume", "reuse"}, "Invalid arm route")
        require(arm["dataset_id"] in parent_data and arm["case"] in CASES, "Unknown parent arm")
        if arm["host"] == "taxi_5080":
            require(arm["dataset_id"] == "yellow_trip_hourly" and arm["mode"] == "fresh", "5080 owns fresh Taxi only")
        if arm["mode"] == "resume":
            require(arm["host"] == "continuation_5090" and arm["dataset_id"] == "intermittent_frozen_5000", "Only Intermittent may resume on 5090")
            require(isinstance(arm.get("resume_from"), str) and arm["resume_from"], "Resume source required")
            require(arm.get("resume_identity", {}).get("execution_contract_sha256") == sha_json(parent), "Resume must retain parent identity")
        if arm["mode"] == "reuse":
            require(arm["host"] == "continuation_5090" and arm["dataset_id"] == "intermittent_frozen_5000",
                    "Only prior Intermittent completions may be reused")
            for key in ("reuse_summary", "reuse_contract"):
                receipt = arm.get(key, {})
                require(isinstance(receipt.get("path"), str) and len(receipt.get("sha256", "")) == 64, "Reused completion receipt required")
    return {"status": "validated_metadata_only", "parent_contract_sha256": sha_json(parent), "arms": 12,
            "gpu_started": False}


def validate_approval(contract, approval):
    require(approval.get("status") == "approved_by_user", "Explicit partition approval required")
    require(approval.get("partition_contract_sha256") == sha_json(contract), "Partition approval SHA mismatch")
    require(approval.get("scope") == "partitioned_continuation_taxi_5080_and_remaining_5090_arms", "Partition approval scope mismatch")
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
                    used += info.st_size
            if used >= self.contract["limits"]["output_bytes_per_host"] - 64 * 1024 ** 2:
                raise RuntimeError("Partition artifact storage ceiling reached")


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
    """Confirm the wrapper will import precisely the frozen 67-file closure."""
    source_root = Path(source_root).resolve()
    expected = parent["source"]["files"]
    require(len(expected) == 67, "Exactly 67 frozen source files required")
    require(all((source_root / name).resolve().is_relative_to(source_root) for name in expected), "Frozen source escapes root")
    observed = {name: sha_file(source_root / name) for name in expected}
    require(observed == expected, "Frozen source closure mismatch")
    for directory in parent["source"].get("required_directories", []):
        require((source_root / directory).is_dir(), "Frozen source sentinel missing")


def validate_handoff(contract, handoff):
    require(handoff.get("partition_contract_sha256") == sha_json(contract), "Handoff partition SHA mismatch")
    require(handoff.get("approved_transfer") is True, "Approved transfer receipt required")
    require(handoff.get("observed_old_process_dead") is True, "Original process ownership not released")


def effective_parent_contract(parent, contract, role):
    """5080 has a native runtime; 5090 keeps the original training identity."""
    effective = copy.deepcopy(parent)
    if role == "continuation_5090":
        return effective
    host = contract["hosts"][role]
    effective["runtime_expected"] = host["runtime_expected"]
    effective["execution"] = {"server_alias": host.get("server_alias"), "python": host["python"],
                              "source_dir": host["source_root"], "output_dir": host["output_dir"],
                              "tmux_session": host.get("tmux_session", "partition")}
    return effective


def _frozen_runner(source_root):
    source_root = str(Path(source_root).resolve())
    require(source_root not in sys.path or sys.path.index(source_root) == 0, "Frozen source must lead import path")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    module = importlib.import_module("paper.scripts.run_quantity_comparison")
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
        frozen.compare_replay(full, split)
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
    frozen.validate_contract(effective)
    return {"passed": True, "qualifies_cuda": True, "runtime": runtime,
            "execution_contract_sha256": sha_json(effective),
            "elapsed_seconds": time.monotonic() - began,
            "checks": ["fresh_process_exact_resume", "optimizer_and_rng_identity", "paired_batches",
                       "production_shapes", "B_objective_identity", "target_padding_causality"]}


def training_identity(parent, contract, role, data, runtime):
    effective = effective_parent_contract(parent, contract, role)
    return {**partition_identity(effective, contract, role, runtime),
            "data": data["inherited_data_identity"], "dataset_id": data["dataset_id"]}


def audit_arm(summary, arm_contract, data, case, identity):
    require(arm_contract["identity"] == identity, "Arm training identity mismatch")
    require(summary["contract_sha256"] == sha_json(arm_contract), "Arm summary contract SHA mismatch")
    require(summary["status"] == "complete" and summary["epochs_completed"] == 120
            and summary["epochs_budget"] == 120, "Arm is not complete at 120 epochs")
    require(summary["global_step"] == data["expected_global_steps"], "Arm global step mismatch")
    require(summary["condition"] == arm_contract["condition"] and summary["condition"]["case"] == case,
            "Arm condition mismatch")
    require(summary["evaluation_scope"] == "validation_only" and summary["held_out_test_evaluated"] is False,
            "Arm evaluation scope mismatch")
    require(len(summary["history"]) == 120, "Arm history length mismatch")
    populations = data["inherited_data_identity"]["populations"]
    for epoch, row in enumerate(summary["history"], 1):
        require(row["epoch"] == epoch and row["global_step"] == epoch * (data["expected_global_steps"] // 120),
                "Arm history step mismatch")
        for split in ("train", "validation"):
            require(row[f"{split}_count"] == populations[split]["target_count"], "Arm population mismatch")


def _run_selected_arms(contract, parent, role, source_root, output, receipt, handoff):
    """Run only the assigned fresh/resume arms through the frozen engine."""
    frozen = _frozen_runner(source_root)
    data_helper = importlib.import_module("paper.scripts.quantity_comparison_data")
    engine = importlib.import_module("paper.scripts.quantity_comparison_engine")
    objective = importlib.import_module("paper.scripts.quantity_objective_comparison")
    diagnostic = importlib.import_module("paper.scripts.run_time_quantity_diagnostic")
    budget = Budget(contract, output)
    parent_data = _parent_map(parent)
    summaries = {}
    loaded_dataset = None
    frame = metadata = None
    effective = effective_parent_contract(parent, contract, role)
    for arm in assigned_arms(contract, role):
        dataset_id, case = _key(arm)
        target = Path(output) / dataset_id / case
        data = parent_data[dataset_id]
        budget()
        runtime = frozen.configured_runtime(effective, "cuda:0")
        require(runtime == receipt["runtime"], "Runtime differs from CUDA qualification")
        identity = training_identity(parent, contract, role, data, runtime)
        if arm["mode"] == "reuse":
            path = Path(arm["reuse_summary"]["path"])
            require(sha_file(path) == arm["reuse_summary"]["sha256"], "Reused summary SHA mismatch")
            summary = read_json(path)
            saved_contract = Path(arm["reuse_contract"]["path"])
            require(saved_contract.resolve() == path.parent.resolve() / "contract.json", "Reused contract path mismatch")
            require(sha_file(saved_contract) == arm["reuse_contract"]["sha256"], "Reused contract SHA mismatch")
            audit_arm(summary, read_json(saved_contract), data, case, identity)
            summaries[_key(arm)] = summary
            continue
        if loaded_dataset != dataset_id:
            frame, metadata = data_helper.prepare_quantity_comparison_data(data)
            metadata = frozen.bind_frozen_statistics(data, metadata)
            loaded_dataset = dataset_id
        if arm["mode"] == "resume":
            source = Path(arm["resume_from"])
            require(source.is_dir() and (source / "last_epoch_state.pt").is_file(), "Copied resume checkpoint missing")
            require(not target.exists(), "Resume target must be fresh")
            saved_contract = read_json(source / "contract.json")
            require(saved_contract["identity"] == identity == arm["resume_identity"], "Resume identity changed")
            require(sha_json(saved_contract) == arm["resume_contract_sha256"], "Resume contract SHA mismatch")
            inherited = handoff.get("inherited_files", {})
            observed = {}
            for path in source.rglob("*"):
                require(not path.is_symlink(), "Resume symlinks forbidden")
                if path.is_file():
                    observed[str(path.resolve())] = sha_file(path)
            require(observed and all(inherited.get(path) == digest for path, digest in observed.items()),
                    "Resume transfer file SHA mismatch")
            require(set(observed) == {path for path in inherited if Path(path).is_relative_to(source.resolve())},
                    "Resume transfer file set mismatch")
            shutil.copytree(source, target)
            require(all(sha_file(target / Path(path).relative_to(source.resolve())) == digest for path, digest in observed.items()),
                    "Copied resume file SHA mismatch")
        model, train, validation, _ = diagnostic.build_arm_inputs({**data, "seed": 42}, frame, metadata)
        write_json(Path(output) / "status.json", {"status": "running", "host_role": role,
                   "dataset_id": dataset_id, "case": case, "deadline_unix": contract["deadline_unix"]})
        summary = engine.run_case(model=model, train_loader=train, validation_loader=validation,
                                  case=objective.QuantityCase(case),
                                  statistics=objective.QuantityStatistics(metadata["train_log_mean"], metadata["raw_scale"]),
                                  output_dir=target, epochs=120, seed=42, identity=identity, device="cuda:0",
                                  lr=.001, weight_decay=.01, grad_clip=1., resume=arm["mode"] == "resume",
                                  budget_check=budget, cuda_qualification=receipt)
        audit_arm(summary, read_json(target / "contract.json"), data, case, identity)
        summaries[_key(arm)] = summary
        del model, train, validation
        gc.collect()
    for dataset in dict.fromkeys(arm["dataset_id"] for arm in assigned_arms(contract, role)):
        audit = frozen.audit_pairs([summaries[(dataset, case)] for case in CASES])
        write_json(Path(output) / dataset / "paired_comparison.json", audit)
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
    _loaded_frozen(authority)
    output = Path(output)
    require(not output.exists(), "Fresh partition output required; automatic resume is forbidden")
    host = contract["hosts"][role]
    import fcntl
    with (output.parent / "quantity_comparison_gpu.lock").open("a+") as lock:
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
            while True:
                budget()
                if role == "taxi_5080" and not (output / "qualification/receipt.json").exists():
                    if time.monotonic() >= qualification_deadline:
                        raise TimeoutError("CUDA qualification ceiling reached")
                try:
                    code = process.wait(timeout=min(1., max(.001, budget.deadline_mono - time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(code == 0, f"Partition failed with exit code {code}; no retry")
            result = read_json(output / "partition_receipt.json")
            require(result["status"] == "complete" and result["partition_contract_sha256"] == sha_json(contract),
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
        require(role == "taxi_5080", "Only 5080 may run fresh CUDA qualification")
        if mode == "_probe":
            require(case in CASES and stage in {"full", "first", "resume"}, "Invalid synthetic probe stage")
            return frozen.probe_worker(effective, case, stage, started, "cuda:0")
        frozen.configured_runtime(effective, "cuda:0")
        # Production shape checks use synthetic inputs for the assigned dataset.
        shape_contract = copy.deepcopy(effective)
        shape_contract["datasets"] = [d for d in effective["datasets"] if d["dataset_id"] == HOST_DATASET[role]]
        frozen.production_shape_checks(shape_contract, "cuda:0", Budget(contract, authority["output"]))
        return {"status": "passed"}
    require(mode == "_worker", "Unknown child mode")
    receipt = _qualification(contract, parent, role, frozen, effective, authority_path)
    write_json(Path(authority["output"]) / "qualification/receipt.json", receipt)
    summaries = _run_selected_arms(contract, parent, role, authority["source_root"], authority["output"], receipt, authority["handoff"])
    result = {"status": "complete", "partition_contract_sha256": sha_json(contract),
              "parent_contract_sha256": sha_json(parent), "role": role,
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
    parser.add_argument("--host-role", choices=("taxi_5080", "continuation_5090"))
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
        require(args.approval and args.source_root and args.output and args.host_role and args.handoff,
                "Execute requires approval, source root, output, host role, and handoff receipt")
        result = execute(contract, read_json(args.approval), source_root=args.source_root, output=args.output,
                         role=args.host_role, handoff=read_json(args.handoff), parent=parent)
    elif args.approval:
        validate_approval(contract, read_json(args.approval))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
