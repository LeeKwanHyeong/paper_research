#!/usr/bin/env python3
"""Owned, deadline-bounded production adapter for the raw auxiliary comparison.

Importing or validating this module does not import Torch. Research training is
available only to an inherited-socket worker after a common two-host permit.
The existing run_raw_aux_gradient_comparison CPU guard remains unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts.intermittent_cell_controls import (
    Budget, WorkerAuthority, require, sha_file, sha_json, supervise_child,
)

ASSIGNMENT = {"5090": ["B_log_original", "mixed_raw_capped_original"],
              "5080": ["mixed_original"]}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_exclusive(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_status(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def validate_assignment(contract, host):
    require(contract.get("schema") == "raw_aux_gradient_parallel_execution_v1", "Wrong parallel execution schema")
    require(host in ASSIGNMENT and set(contract["hosts"]) == set(ASSIGNMENT), "Unknown host assignment")
    require(all(contract["hosts"][name]["assigned_arms"] == arms for name, arms in ASSIGNMENT.items()),
            "Fixed 5090 B then capped / 5080 uncapped assignment changed")
    require(all(contract["policy"].get(key) is False for key in ("resume", "retry", "held_out")),
            "Resume/retry/held-out execution is forbidden")
    data = contract["dataset"]
    require(data["dataset_id"] == "intermittent_frozen_5000"
            and data["loader"]["batch_size"] == 128 and data["loader"]["max_seq_len"] == 256
            and data["loader"]["num_workers"] == 0 and data["loader"]["drop_last"] is False,
            "Fixed Intermittent loader contract changed")
    require(data["expected_global_steps"] == 369240, "Fixed 120-epoch step budget changed")
    return contract["hosts"][host]


def supervision_limits(contract):
    limits = contract["limits"]
    require(limits["training_seconds"] == 43200 and limits["total_seconds"] == 46800,
            "Production stage/total time caps changed")
    require(limits["preflight_seconds"] == limits["checkpoint_seconds"] == 1800,
            "Production preflight/replay caps changed")
    # Training includes its ordinary epoch validation. The separate replay
    # stage covers exactly the raw-selected and last120 checkpoint replays.
    return {"max_wall_seconds": limits["total_seconds"], "stage_seconds": {
        "preflight_and_finalize": limits["preflight_seconds"], "training": limits["training_seconds"],
        "checkpoint_replay": limits["checkpoint_seconds"],
    }}


def execution_context(contract, permit, host):
    return sha_json({"contract": sha_json(contract), "permit": sha_json(permit), "host": host})


def fixed_start(permit, *, wall=time.time, monotonic=time.monotonic):
    started, deadline, now = permit["started_at_unix"], permit["deadline_unix"], wall()
    require(type(started) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(started) and math.isfinite(deadline)
            and started <= now < deadline and deadline - started == 46800,
            "Stale, future, or extended common launch deadline")
    return monotonic() - (now - started)


def gpu_pids(host):
    text = subprocess.check_output(["nvidia-smi", "--id=" + host["gpu_uuid"],
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15)
    return {int(line.strip()) for line in text.splitlines() if line.strip()}


def verify_host_paths(host):
    root = Path(host["root"]).resolve()
    require(Path(host["source_root"]).resolve() == root / "source"
            and Path(host["output_dir"]).resolve() == root / "run", "Pinned host root layout differs")
    require(Path(sys.executable).resolve() == Path(host["python"]).resolve(), "Wrong host Python")
    require(Path(__file__).resolve() == root / "source/paper/scripts/run_raw_aux_gradient_partition.py",
            "Run the pinned source adapter")
    require(Path.cwd().resolve() == root / "source", "Pinned source working directory required")


class StorageBudget:
    """Bound cooperative file growth; RLIMIT_FSIZE separately bounds native writes."""
    def __init__(self, budget, contract, output, *, wall=time.time, monotonic=time.monotonic):
        self.budget, self.contract, self.output = budget, contract, Path(output)
        self.wall, self.monotonic, self.last_scan = wall, monotonic, -math.inf

    def check_storage(self):
        limits = self.contract["limits"]
        total = 0
        for path in self.output.rglob("*"):
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            require(not stat.S_ISLNK(info.st_mode), "Output symlinks are forbidden")
            if stat.S_ISREG(info.st_mode):
                require(info.st_size <= limits["max_file_bytes"], "Artifact exceeds per-file byte limit")
                total += info.st_size
        # An epoch may atomically publish several files before returning to a
        # callback. Reserve eight individually bounded files plus terminal JSON.
        reserve = min(8 * limits["max_file_bytes"] + 65536, limits["per_host_output_bytes"] // 2)
        require(total <= limits["per_host_output_bytes"] - reserve, "Host output byte budget reached")
        require(shutil.disk_usage(self.output).free >= limits["min_free_bytes"], "Minimum free disk reached")
        return total

    def __call__(self, stage=None):
        self.budget.check()
        now = self.monotonic()
        if stage == "before_epoch_commit" or now - self.last_scan >= 10:
            self.check_storage()
            self.last_scan = now


def audit_completed_arm(summary, arm, initial_sha, data):
    require(summary["status"] == "complete" and summary["epochs_completed"] == summary["epochs_budget"] == 120,
            "Incomplete fixed-epoch arm")
    require(summary["initial_state_sha256"] == initial_sha and summary["global_step"] == 369240,
            "Arm initialization or optimizer-step mismatch")
    require(summary["evaluation_scope"] == "validation_only" and summary["held_out_test_evaluated"] is False,
            "Arm evaluation split changed")
    require(summary["condition"]["mixed_objective"]["name"] == ("B_log_original" if arm == "B_log_original" else "mixed_original"),
            "Arm objective identity changed")
    require(("raw_aux_control" in summary["condition"]) == (arm == "mixed_raw_capped_original"), "Arm controller identity changed")
    require(len(summary["history"]) == 120, "Arm epoch history incomplete")
    populations = data["inherited_data_identity"]["populations"]
    for epoch, row in enumerate(summary["history"], 1):
        require(row["epoch"] == epoch and row["global_step"] == epoch * 3077,
                "Arm epoch/step exposure differs")
        require(row["train_batches"] == 3077 and row["validation_batches"] == 675,
                "Arm batch budget differs")
        for split in ("train", "validation"):
            require(row[split + "_count"] == populations[split]["target_count"], "Arm population count differs")


def audit_host_exposure(summaries):
    first = next(iter(summaries.values()))
    for value in summaries.values():
        require(value["initial_state_sha256"] == first["initial_state_sha256"], "Paired initial weights differ")
        for left, right in zip(first["history"], value["history"], strict=True):
            require(all(left[key] == right[key] for key in ("epoch", "global_step", "train_count", "train_batches",
                "validation_count", "validation_batches", "train_batch_order_sha256")), "Paired host exposure differs")


def _production_arms(contract, permit, host_name, receipt, authority, budget):
    """Only an authenticated worker reaches model/data imports or run_case."""
    from paper.scripts import raw_aux_parallel_common as common
    from paper.scripts.quantity_comparison_engine import run_case
    from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics
    from paper.scripts.raw_aux_gradient_acceptance import evaluate_checkpoints
    from paper.scripts.run_raw_aux_gradient_comparison import load_design, objective_for_arm
    from paper.scripts.run_time_quantity_diagnostic import validate_import_origins
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    import gc

    host = contract["hosts"][host_name]
    output = Path(host["output_dir"])
    design = load_design()
    expected_initial = design["initialization_and_exposure"]["initial_state_sha256"]
    validate_import_origins()
    runtime = common.runtime_check(contract, host_name)
    require(runtime == receipt["runtime"], "Actual runtime differs from native CUDA qualification")
    require(receipt["initial_state_sha256"] == expected_initial, "Qualified historical initialization differs")
    frame, metadata, data = common.prepare_inputs(contract)
    validate_import_origins()
    require(data == contract["dataset"], "Prepared scientific data contract changed")
    require(metadata["held_out_materialized"] is False
            and metadata["populations"] == data["inherited_data_identity"]["populations"], "Prepared split/population identity differs")
    stats = QuantityStatistics(metadata["train_log_mean"], metadata["raw_scale"])
    identity = common.make_identity(contract, host_name, runtime)
    storage = StorageBudget(budget, contract, output)
    summaries, evaluations = {}, {}
    for arm in host["assigned_arms"]:
        budget.set_stage("preflight_and_finalize")
        authority.check_owner(started=budget.started, limits=budget.limits,
                              context_sha256=execution_context(contract, permit, host_name))
        require(common.source_manifest() == contract["source"], "Source changed before arm")
        require(gpu_pids(host) <= {os.getpid()}, "Another GPU job appeared; do not kill it")
        require(common.runtime_check(contract, host_name) == runtime, "Runtime changed before arm")
        model, train, validation, _ = common.build_inputs(data, frame, metadata)
        initial = canonical_state_dict_sha256(model.state_dict())
        require(initial == expected_initial, "Fresh model does not match frozen historical initialization")
        objective, control = objective_for_arm(arm, design)
        arm_dir = output / arm
        require(not arm_dir.exists(), "Arm output exists; retry and resume are forbidden")
        storage()
        write_status(output / "status.json", {"status": "training", "host": host_name, "arm": arm,
            "execution_contract_sha256": sha_json(contract), "deadline_unix": permit["deadline_unix"]})
        budget.set_stage("training")
        summary = run_case(model=model, train_loader=train, validation_loader=validation,
            case=QuantityCase.B_LOG_ORIGINAL, mixed_objective=objective, raw_aux_control=control,
            statistics=stats, output_dir=arm_dir, epochs=120, seed=42, identity=identity,
            device="cuda:0", lr=.001, weight_decay=.01, grad_clip=1., resume=False,
            stop_after_epochs=None, budget_check=storage, cuda_qualification=receipt)
        audit_completed_arm(summary, arm, initial, data)
        summaries[arm] = summary
        budget.set_stage("checkpoint_replay")
        write_status(output / "status.json", {"status": "checkpoint_replay", "host": host_name, "arm": arm,
            "execution_contract_sha256": sha_json(contract), "deadline_unix": permit["deadline_unix"]})
        evaluation = evaluate_checkpoints(model=model, validation_loader=validation, statistics=stats,
            objective=objective, arm_dir=arm_dir, arm_id=arm, device="cuda:0", budget_check=storage, synthetic=False)
        evaluation["training_contract"] = read_json(arm_dir / "contract.json")
        write_exclusive(arm_dir / "validation_diagnosis.json", evaluation)
        evaluations[arm] = {"path": str(arm_dir / "validation_diagnosis.json"),
                            "sha256": sha_file(arm_dir / "validation_diagnosis.json")}
        del model, train, validation
        gc.collect()
    budget.set_stage("preflight_and_finalize")
    audit_host_exposure(summaries)
    storage()
    return {"status": "host_complete", "host": host_name, "completed_arms": list(summaries),
            "global_suite_complete": False, "global_steps_per_arm": 369240,
            "total_host_optimizer_steps": 369240 * len(summaries), "validation_replays": 2 * len(summaries),
            "evaluations": evaluations, "runtime": runtime, "initial_state_sha256": expected_initial,
            "execution_contract_sha256": sha_json(contract), "permit_sha256": sha_json(permit),
            "deadline_unix": permit["deadline_unix"], "held_out_evaluated": False, "automatic_retry": False,
            "worker_budget": budget.snapshot()}


def worker(contract_path, permit_path, host_name, started):
    # A --worker flag and fabricated PID cannot replace the inherited socket.
    authority = WorkerAuthority.from_environment(started=started)
    try:
        from paper.scripts import raw_aux_parallel_common as common
        contract = common.read_contract(contract_path)
        permit = read_json(permit_path)
        host = validate_assignment(contract, host_name)
        receipt = common.validate_permit(contract, permit, host_name)
        require(time.time() < permit["deadline_unix"], "Common execution deadline expired")
        limits = supervision_limits(contract)
        context = execution_context(contract, permit, host_name)
        authority.check_owner(started=started, limits=limits, context_sha256=context)
        verify_host_paths(host)
        output = Path(host["output_dir"])
        owner = read_json(output / "supervisor_owner.json")
        require(owner["context_sha256"] == context and owner["supervisor_pid"] == os.getppid(), "Output belongs to another supervisor")
        write_exclusive(output / "worker_owner.json", {"pid": os.getpid(), "parent_pid": os.getppid(),
            "pgid": os.getpgrp(), "context_sha256": context, "nonce": authority.nonce})
        common.apply_environment(host)
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE, (contract["limits"]["max_file_bytes"],) * 2)
        budget = Budget(limits, started=started, stage_reporter=authority.set_stage)
        result = _production_arms(contract, permit, host_name, receipt, authority, budget)
        require(result["completed_arms"] == host["assigned_arms"], "Worker arm ownership/completion differs")
        write_exclusive(output / "worker_terminal_status.json", {**result, "context_sha256": context})
        authority.finish()
        return result
    finally:
        authority.close()


@contextmanager
def parent_signals():
    previous = {}
    def interrupted(signum, frame):
        raise InterruptedError(f"Raw auxiliary supervisor interrupted by signal {signum}")
    try:
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def execute(contract_path, permit_path, host_name):
    from paper.scripts import raw_aux_parallel_common as common
    contract = common.read_contract(contract_path)
    permit = read_json(permit_path)
    host = validate_assignment(contract, host_name)
    receipt = common.validate_permit(contract, permit, host_name)
    common.verify_native_proofs(contract, host_name, receipt)
    verify_host_paths(host)
    common.apply_environment(host)
    started = fixed_start(permit)
    context, limits = execution_context(contract, permit, host_name), supervision_limits(contract)
    output = Path(host["output_dir"])
    require(bool(os.environ.get("TMUX")), "Pinned tmux session required")
    session = subprocess.check_output([host["tmux_binary"], "display-message", "-p", "#S"], text=True, timeout=15).strip()
    require(session == host["tmux"], "Wrong tmux session")
    import fcntl
    lock_path = Path(host["root"]) / "raw_aux_training.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not output.exists(), "Fresh output required; resume and retry are forbidden")
        require(shutil.disk_usage(output.parent).free >= contract["limits"]["min_free_bytes"], "Insufficient free disk")
        require(not gpu_pids(host), "GPU is busy; no other process will be stopped")
        output.mkdir()
        owner = {"supervisor_pid": os.getpid(), "context_sha256": context, "host": host_name,
                 "execution_contract_sha256": sha_json(contract), "permit_sha256": sha_json(permit)}
        write_exclusive(output / "supervisor_owner.json", owner)
        command = [sys.executable, str(Path(__file__).resolve()), "_worker", "--contract", str(Path(contract_path).resolve()),
                   "--permit", str(Path(permit_path).resolve()), "--host", host_name, "--started", str(started)]
        try:
            with parent_signals():
                supervision = supervise_child(command, limits, started=started, context_sha256=context,
                    env=dict(os.environ), cwd=host["source_root"], kill_grace=2.)
            completed = read_json(output / "worker_terminal_status.json")
            require(completed.get("status") == "host_complete" and completed.get("context_sha256") == context
                    and completed.get("completed_arms") == host["assigned_arms"], "Authenticated worker completion missing")
            terminal = {**completed, "parent_supervision": supervision, "parent_authoritative": True}
        except BaseException as error:
            terminal = {"status": "timeout" if isinstance(error, TimeoutError) else "failed", "error": str(error),
                **owner, "parent_authoritative": True, "automatic_retry": False,
                "parent_budget": getattr(error, "supervision_budget", None), "held_out_evaluated": False}
            require(read_json(output / "supervisor_owner.json") == owner, "Output ownership changed before terminal write")
            write_exclusive(output / "terminal_status.json", terminal)
            raise
        require(read_json(output / "supervisor_owner.json") == owner, "Output ownership changed before terminal write")
        write_exclusive(output / "terminal_status.json", terminal)
        return terminal


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "execute", "_worker"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--permit", type=Path)
    parser.add_argument("--host", choices=tuple(ASSIGNMENT), required=True)
    parser.add_argument("--started", type=float)
    args = parser.parse_args(argv)
    if args.mode == "_worker":
        require(args.permit is not None and args.started is not None, "Worker needs bound permit/start")
        return worker(args.contract, args.permit, args.host, args.started)
    if args.mode == "validate":
        from paper.scripts import raw_aux_parallel_common as common
        contract = common.read_contract(args.contract)
        validate_assignment(contract, args.host)
        supervision_limits(contract)
        return {"status": "validated_metadata_only", "host": args.host,
                "execution_contract_sha256": sha_json(contract), "gpu_started": False}
    require(args.permit is not None and args.started is None, "Execute requires common permit; start cannot be overridden")
    return execute(args.contract, args.permit, args.host)


if __name__ == "__main__":
    print(json.dumps(main(), allow_nan=False))
