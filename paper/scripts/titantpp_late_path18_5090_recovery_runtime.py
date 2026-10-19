"""One isolated, bounded 5090 recovery after the recorded server reboot.

The original scientific contract, sources and outputs remain read-only. Only
three interrupted Intermittent fits and one unstarted fit may train here. The
six completed RAF fits are Validation references, never new training jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
import signal
from pathlib import Path
import subprocess
import time
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import (
    gpu_snapshot, local_path, now, read, require, runtime_check, sha_file,
    sha_json, terminate_owned_child, write,
)
from paper.scripts.titantpp_cross_fusion_seed_extension_runtime import proc_identity

SCHEMA = "titantpp_late_path18_5090_recovery_v1"
STAGE_SCHEMA = "titantpp_late_path18_recovery_stage_v1"
MODULE = "paper.scripts.titantpp_late_path18_5090_recovery_runtime"
WORKER_MODULE = "paper.scripts.run_titantpp_late_path18_recovery"
ARMS = ("titantpp_mlp16_late_observation_v1", "titantpp_mlp16_late_hidden_v1")
HOST = "5090"
SOURCE_COUNT = 963
SOURCE_SHA256 = "57dca469664c78d008ba9dd091c1b5ee55d4236fa60cbd18a5b0a828a3f5c94a"
DEADLINE = 1792888885.815062
GPU_UUID = "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"
STAGES = ("train", "validation")
SUPERVISOR_CLEANUP_GRACE_SECONDS = 10.0
OWNED_CLEANUP_GRACE_SECONDS = 5.0


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def scientific_contract(contract):
    binding = contract["scientific_contract"]
    path = Path(binding["path"])
    require(path.is_absolute() and sha_file(path) == binding["raw_sha256"], "Original scientific contract bytes changed")
    value = read(path)
    require(sha_json(value) == binding["canonical_sha256"], "Original scientific contract identity changed")
    return value


def validate_contract(contract, host=HOST, *, science=None):
    require(contract.get("schema") == SCHEMA and contract.get("host") == host == HOST, "Recovery is 5090 only")
    require(contract.get("evaluation_scope") == "validation_only" and contract.get("held_out_test_evaluated") is False,
            "Only Validation is authorized")
    require(contract.get("max_workers") == 3, "Recovery concurrency must be three")
    require(contract.get("limits") == {"fit_wall_seconds": 129600, "evaluation_wall_seconds": 7200,
            "queue_wall_seconds": 604800}, "Recovery limits changed")
    science = scientific_contract(contract) if science is None else science
    require(sha_json(science) == contract["scientific_contract"]["canonical_sha256"], "Scientific contract canonical binding changed")
    require(science.get("schema") == "titantpp_late_path18_v1" and science.get("evaluation_scope") == "validation_only"
            and science.get("held_out_test_authorized") is False, "Original scientific scope changed")
    require(science.get("common_condition") == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.},
            "Logged inputs/log-MSE/alpha0 must remain fixed")
    require(all(science["training"].get(k) == v for k, v in {"maximum_epochs": 300, "minimum_epochs": 40,
            "patience": 40, "batch_size": 128, "monitor": "validation_raw_quantity_rmse"}.items()), "Training changed")
    require(science["training"].get("warm_start", False) is False, "Original fresh-fit contract changed")
    a = science["architecture"]
    require(a.get("arms") == list(ARMS) and a.get("late_path_parameter_count") == 1024
            and a.get("time_input_transform") == a.get("quantity_input_transform") == "log1p_nonnegative_float32",
            "Candidate/input architecture changed")
    expected = {(d, a, s) for d in ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
                for a in ARMS for s in (42, 52, 62)}
    require(len(science["jobs"]) == 18 and {(j["dataset"], j["arm"], j["seed"]) for j in science["jobs"]} == expected,
            "Original eighteen-fit matrix changed")
    original = {j["id"]: j for j in science["jobs"] if j["host"] == HOST}
    jobs = contract["jobs"]
    require(len(original) == len(jobs) == len({j["id"] for j in jobs}) == 10
            and {j["id"] for j in jobs} == set(original), "Expected original 5090 ten-job matrix")
    for job in jobs:
        require(all(job.get(k) == v for k, v in original[job["id"]].items()), "Original scientific job changed")
        mode = job["recovery_mode"]
        require(Path(job["id"]).name == job["id"] and job["id"] not in ("", ".", ".."), "Unsafe job ID")
        if job["dataset"] == "raf_spare_parts":
            require(mode == "reference", "RAF must not retrain")
            for key in ("origin_train_receipt", "origin_selected_checkpoint"):
                require(Path(job[key]["path"]).is_absolute() and len(job[key]["sha256"]) == 64, "Missing RAF origin")
        else:
            require(job["dataset"] == "intermittent_frozen_5000" and job["seed"] in (52, 62), "Unapproved recovery fit")
            fresh = job["seed"] == 62 and job["arm"] == ARMS[1]
            require(mode == ("fresh" if fresh else "resume"), "Recovery mode changed")
            if not fresh:
                require(Path(job["origin_resume_checkpoint"]["path"]).is_absolute()
                        and len(job["origin_resume_checkpoint"]["sha256"]) == 64
                        and type(job["resume_epoch"]) is int and job["resume_epoch"] > 0
                        and job["original_started_at"], "Missing exact-resume origin")
    train_ids = [j["id"] for j in jobs if j["recovery_mode"] != "reference"]
    require(len(contract["training_jobs"]) == len(set(contract["training_jobs"])) == 4
            and set(contract["training_jobs"]) == set(train_ids), "Only four Intermittent training jobs are admitted")
    require(len(contract["validation_jobs"]) == len(set(contract["validation_jobs"])) == 10
            and set(contract["validation_jobs"]) == set(original), "Validation must cover ten jobs")
    require(contract["source"]["files"] == science["source"]["files"]
            and len(contract["source"]["files"]) == SOURCE_COUNT
            and sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"] == SOURCE_SHA256
            and contract["source"]["source_revision"] == science["source"]["base_git_revision"], "Original source963 changed")
    operations = contract["operation_files"]
    require(len(operations) == 3 and sha_json(operations) == contract["operation_files_sha256"]
            and all(name.startswith("paper/scripts/") and name.endswith(".py") for name in operations)
            and "paper/scripts/titantpp_late_path18_5090_recovery_runtime.py" in operations
            and "paper/scripts/run_titantpp_late_path18_recovery.py" in operations
            and not set(operations).intersection(contract["source"]["files"]), "Recovery operation closure changed")
    require(contract.get("operation_git_revision"), "Missing operation source revision")
    target = science["hosts"][HOST]
    for key in ("python", "gpu_uuid", "runtime_expected", "absolute_host_deadline_unix"):
        require(contract[key] == target[key], "Original host setting changed: " + key)
    expected_environment = {**target["environment"], "PYTHONPATH": contract["source_root"],
        "MPLCONFIGDIR": contract["root"] + "/cache/matplotlib", "XDG_CACHE_HOME": contract["root"] + "/cache"}
    require(contract["environment"] == expected_environment, "Only isolated source/cache environment paths may change")
    require(contract.get("input_files") == science["input_files"], "Train/Validation input inventory changed")
    require(contract["gpu_uuid"] == GPU_UUID and contract["absolute_host_deadline_unix"] == DEADLINE,
            "GPU/deadline changed")
    for key in ("root", "source_root", "python"):
        require(Path(contract[key]).is_absolute(), "Recovery host paths must be absolute")
    root, old = Path(contract["root"]).resolve(), Path(target["root"]).resolve()
    require(root != old and not root.is_relative_to(old) and not old.is_relative_to(root)
            and Path(contract["source_root"]).resolve() == root / "source", "Recovery must use a separate root")
    immutable = contract["immutable_originals"]
    require(Path(immutable["root"]).resolve() == old and isinstance(immutable["files"], dict)
            and bool(immutable["files"]) and contract.get("boot_id"), "Original inventory/boot binding missing")
    for job in jobs:
        for key in ("origin_train_receipt", "origin_selected_checkpoint", "origin_resume_checkpoint"):
            if key in job:
                path = Path(job[key]["path"])
                require(path.resolve().is_relative_to(old)
                        and immutable["files"].get(str(path.resolve().relative_to(old))) == job[key]["sha256"], "Origin outside immutable inventory")
    return [next(j for j in jobs if j["id"] == key) for key in contract["training_jobs"]]


def verify_originals(contract):
    root = Path(contract["immutable_originals"]["root"])
    inventory = contract["immutable_originals"]["files"]
    require(inventory, "Empty immutable original inventory")
    for name, digest in inventory.items():
        require(sha_file(local_path(root, name)) == digest, "Preserved original changed: " + name)
    return {"root": str(root), "file_count": len(inventory), "files_sha256": sha_json(inventory),
            "verified_at": now(), "original_files_unchanged": True}


def verify_files(contract):
    validate_contract(contract)
    science = scientific_contract(contract)
    source = Path(contract["source_root"])
    old_source = Path(science["hosts"][HOST]["source_root"])
    for name, digest in contract["source"]["files"].items():
        require(sha_file(local_path(source, name)) == digest and sha_file(local_path(old_source, name)) == digest,
                "Original source bytes changed: " + name)
    for name, digest in contract["operation_files"].items():
        require(sha_file(local_path(source, name)) == digest, "Recovery operation changed: " + name)
    for name, digest in science["input_files"].items():
        require(sha_file(local_path(source, name)) == digest and sha_file(local_path(old_source, name)) == digest, "Original Train/Validation input changed: " + name)
    require(boot_id() == contract["boot_id"], "Server boot changed since recovery admission")
    verify_originals(contract)
    return science


def gpu_state(contract, active_pids=()):
    require(len(active_pids) == len(set(active_pids)) <= 3, "Worker concurrency exceeded")
    require(boot_id() == contract["boot_id"], "Server rebooted during recovery")
    gpu = gpu_snapshot(contract["gpu_uuid"])
    require(set(gpu["compute_pids"]) <= set(active_pids), "Unknown GPU workload; no new work")
    return gpu


def worker_command(contract, contract_path, job, stage):
    require(stage in (*STAGES, "qualify"), "Only training, Validation and fabricated qualification are admitted")
    if stage == "train":
        require(job["id"] in contract["training_jobs"] and job["recovery_mode"] != "reference", "RAF training forbidden")
    elif stage == "validation":
        require(job["id"] in contract["validation_jobs"], "Unapproved Validation job")
    return [contract["python"], "-u", "-m", WORKER_MODULE,
            "--recovery-contract", str(contract_path), "--job", job["id"], "--stage", stage]


def verify_receipt(contract, job, stage):
    root = Path(contract["root"])
    path = root / "run" / job["id"] / f"{stage}_receipt.json"
    value = read(path)
    require(stage in STAGES and value.get("schema") == STAGE_SCHEMA and value.get("status") == "complete"
            and value.get("stage") == stage and value.get("job_id") == job["id"]
            and value.get("contract_sha256") == sha_json(contract)
            and value.get("scientific_contract_sha256") == contract["scientific_contract"]["canonical_sha256"]
            and value.get("scientific_contract_raw_sha256") == contract["scientific_contract"]["raw_sha256"], "Recovery stage receipt mismatch")
    require(all(value.get(k) == job[k] for k in ("host", "dataset", "arm", "seed", "time_alpha", "loss_mode", "time_input"))
            and value.get("candidate_name") == job["arm"] and value.get("late_path_parameter_count") == 1024
            and value.get("late_path_mode") == ("observation" if job["arm"] == ARMS[0] else "hidden")
            and value.get("time_input_transform") == value.get("quantity_input_transform") == "log1p_nonnegative_float32"
            and value.get("held_out_test_evaluated") is False and value.get("quantity_only") is True
            and value.get("time_head_trained") is False and value.get("quantity_loss_normalization") == "none",
            "Recovery science labels changed")
    from paper.scripts.titantpp_quantity_loss_mlp18_runtime import verify_time_head_proof
    verify_time_head_proof(value.get("time_head_no_training_verification"))
    files = value.get("files")
    require(isinstance(files, dict) and bool(files), "Empty recovery artifact inventory")
    folder = (root / "run" / job["id"]).resolve()
    for name, digest in files.items():
        p = local_path(root, name)
        require(p.is_relative_to(folder) and p != path.resolve() and sha_file(p) == digest, "Recovery artifact changed: " + name)
    checkpoint = Path(value["selected_checkpoint"])
    require(checkpoint.is_absolute() and checkpoint.resolve().is_relative_to(folder), "Selected copy outside new job root")
    require(files.get(str(checkpoint.relative_to(root))) == value["checkpoint_sha256"]
            and sha_file(checkpoint) == value["checkpoint_sha256"], "Recovery selected binary changed")
    require(type(value.get("selected_epoch")) is int and value["selected_epoch"] > 0
            and isinstance(value.get("state_sha256"), str) and len(value["state_sha256"]) == 64
            and value.get("target_identity") and value.get("metrics"), "Missing selected endpoint proof")
    if job["recovery_mode"] == "reference":
        require(stage == "validation" and value["checkpoint_sha256"] == job["origin_selected_checkpoint"]["sha256"], "RAF selected checkpoint changed")
    if stage == "validation":
        require(value.get("validation_replay_passed") is True, "Native Validation replay failed")
        if job["recovery_mode"] != "reference":
            train = read(path.with_name("train_receipt.json"))
            require(all(train.get(k) == value.get(k) for k in ("selected_checkpoint", "selected_epoch", "checkpoint_sha256",
                    "state_sha256", "target_identity")), "Validation changed training selection")
    return {"path": str(path.relative_to(root)), "sha256": sha_file(path)}


def fit_deadline(contract, job, started_unix):
    """Resume keeps the interrupted fit's original 36-hour wall-clock budget."""
    if job["recovery_mode"] == "resume":
        original = datetime.fromisoformat(job["original_started_at"])
        require(original.tzinfo is not None, "Original resume start requires an explicit timezone")
        started_unix = original.timestamp()
    return min(contract["absolute_host_deadline_unix"], started_unix + contract["limits"]["fit_wall_seconds"])


def _identity(process, command, source, start_ticks=None):
    live = proc_identity(process.pid)
    if live is None:
        require(process.poll() is not None, "Child identity disappeared without observable OS exit")
        return None
    require(live["pgid"] == live["pid"] == process.pid and live["argv"] == command
            and live["cwd"] == str(source) and (start_ticks is None or live["start_ticks"] == start_ticks), "Owned child identity changed")
    return live


def _exit_record(contract, process, prefix, command, *, timeout, started_at, start_ticks):
    value = {"pid": process.pid, "start_ticks": start_ticks, "command": command,
             "actual_os_returncode": process.returncode, "timeout": timeout, "started_at": started_at,
             "observed_exit_at": now(), "boot_id": contract["boot_id"],
             "stdout_sha256": sha_file(prefix.with_name(prefix.name + ".stdout.log")),
             "stderr_sha256": sha_file(prefix.with_name(prefix.name + ".stderr.log")),
             "stdout_bytes": prefix.with_name(prefix.name + ".stdout.log").stat().st_size,
             "stderr_bytes": prefix.with_name(prefix.name + ".stderr.log").stat().st_size}
    return value


def qualify(contract, contract_path):
    validate_contract(contract)
    root, source = Path(contract["root"]), Path(contract["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Qualification cwd mismatch")
    require(not any((root / name).exists() for name in ("qualification_attempt.json", "qualification.json", "start.json", "run")), "Existing qualification; no retry")
    verify_files(contract)
    before = gpu_state(contract)
    require(time.time() < contract["absolute_host_deadline_unix"], "Inherited deadline expired before qualification")
    write(root / "qualification_attempt.json", {"contract_sha256": sha_json(contract), "started_at": now()}, exclusive=True)
    self_identity = proc_identity(os.getpid())
    expected = [contract["python"], "-u", "-m", MODULE, "--contract", str(contract_path), "--mode", "qualify"]
    require(self_identity is not None and self_identity["pid"] == self_identity["pgid"] == os.getpid()
            and self_identity["argv"] == expected and self_identity["cwd"] == str(source), "Qualification own identity changed")
    runtime = runtime_check(contract)
    results, all_pids, peak = {}, [], 0
    qualification_started = time.monotonic()
    for arm in ARMS:
        job = next(j for j in contract["jobs"] if j["arm"] == arm)
        children = []
        try:
            for index in range(3):
                require(time.monotonic() - qualification_started < contract["limits"]["evaluation_wall_seconds"]
                        and time.time() < contract["absolute_host_deadline_unix"], "Qualification deadline exhausted")
                prefix = root / f"qualification_{arm}_{index}"
                command = worker_command(contract, contract_path, job, "qualify")
                started_at = now()
                with prefix.with_name(prefix.name + ".stdout.log").open("xb") as out, prefix.with_name(prefix.name + ".stderr.log").open("xb") as err:
                    child = subprocess.Popen(command, cwd=source, env={**os.environ, **contract["environment"], "PYTHONDONTWRITEBYTECODE": "1"}, stdout=out, stderr=err, start_new_session=True)
                children.append((child, prefix, command, started_at))
                all_pids.append(child.pid)
                live = _identity(child, command, source)
                write(prefix.with_name(prefix.name + ".lease.json"), {"job_id": job["id"], "stage": "qualify", "pid": child.pid,
                    "start_ticks": live["start_ticks"] if live else None, "command": command, "started_at": started_at,
                    "boot_id": contract["boot_id"], "contract_sha256": sha_json(contract)}, exclusive=True)
            rows, began = [], time.monotonic()
            for child, prefix, command, started_at in children:
                timed_out = False
                try:
                    child.wait(timeout=max(.001, min(contract["limits"]["evaluation_wall_seconds"] - (time.monotonic() - qualification_started),
                        contract["absolute_host_deadline_unix"] - time.time())))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_owned_child(child)
                record = _exit_record(contract, child, prefix, command, timeout=timed_out, started_at=started_at, start_ticks=None)
                write(prefix.with_name(prefix.name + ".exit.json"), record, exclusive=True)
                require(child.returncode == 0 and not timed_out, "Native qualification child failed")
                result = json.loads(prefix.with_name(prefix.name + ".stdout.log").read_text())
                require(result.get("status") == "passed" and result.get("real_data_loaded") is False
                        and result.get("candidate_name") == arm, "Qualification candidate mismatch")
                peaks = result["cuda_memory_peaks"]
                require(set(peaks) == {arm}, "Qualification peak candidate changed")
                measured = peaks[arm]["max_memory_reserved_bytes"]
                require(type(measured) is int and measured > 0, "Missing native memory peak")
                peak = max(peak, measured)
                rows.append({**record, "result": result})
            results[arm] = rows
        finally:
            for child, prefix, command, started_at in children:
                cleaned = child.poll() is None
                if cleaned:
                    terminate_owned_child(child)
                if not prefix.with_name(prefix.name + ".exit.json").exists():
                    record = _exit_record(contract, child, prefix, command, timeout=False, started_at=started_at, start_ticks=None)
                    write(prefix.with_name(prefix.name + ".exit.json"), {**record, "owned_cleanup_after_failure": cleaned}, exclusive=True)
        live = proc_identity(os.getpid())
        require(live is not None and all(live[k] == self_identity[k] for k in ("pid", "pgid", "start_ticks", "argv", "cwd")), "Qualification owner changed")
        gpu = gpu_state(contract, [os.getpid()])
        require(not set(all_pids).intersection(gpu["compute_pids"]), "Qualification child GPU owner remains")
    require(all(proc_identity(pid) is None for pid in all_pids), "Qualification child remains")
    require(3 * peak < .85 * runtime["gpu"]["total_memory_bytes"], "Three-worker memory admission failed")
    verify_files(contract)
    receipt = {"status": "passed", "host": HOST, "contract_sha256": sha_json(contract), "verified_at": now(),
               "runtime": runtime, "gpu_before": before, "per_arm": results, "maximum_simultaneous_workers": 3,
               "native_worker_exit0_count": 6, "owned_worker_pids_absent": True, "qualification_owner_identity": self_identity,
               "maximum_single_worker_reserved_bytes": peak, "held_out_test_evaluated": False}
    write(root / "qualification.json", receipt, exclusive=True)
    return receipt


def run_stage(contract, contract_path, stage, queue_started):
    require(stage in STAGES, "Test forbidden")
    validate_contract(contract)
    root, source = Path(contract["root"]), Path(contract["source_root"])
    keys = contract["training_jobs" if stage == "train" else "validation_jobs"]
    pending = [next(j for j in contract["jobs"] if j["id"] == key) for key in keys]
    active, exits, receipts, failure = {}, {}, {}, None
    limit = contract["limits"]["fit_wall_seconds" if stage == "train" else "evaluation_wall_seconds"]
    while active or (pending and failure is None):
        for key, record in list(active.items()):
            child = record["process"]
            timed_out = child.poll() is None and (time.monotonic() - record["began"] >= limit
                or time.monotonic() - queue_started >= contract["limits"]["queue_wall_seconds"]
                or time.time() >= record["deadline_unix"])
            if timed_out:
                terminate_owned_child(child)
            code = child.poll()
            if code is None:
                continue
            exit_record = {**_exit_record(contract, child, record["prefix"], record["command"], timeout=timed_out,
                started_at=record["started_at"], start_ticks=record["start_ticks"]), "job_id": key, "stage": stage}
            write(root / "run" / key / f"{stage}_exit.json", exit_record, exclusive=True)
            exits[key] = exit_record
            del active[key]
            try:
                require(code == 0 and not timed_out, "Scientific worker failed or timed out")
                receipts[key] = verify_receipt(contract, record["job"], stage)
            except Exception:
                failure = failure or {"stage": stage, "job_id": key, "at": now(), "traceback": traceback.format_exc()}
        gpu = None
        if failure is None:
            try:
                require(time.time() < contract["absolute_host_deadline_unix"]
                        and time.monotonic() - queue_started < contract["limits"]["queue_wall_seconds"], "Inherited recovery deadline expired")
                for record in active.values():
                    _identity(record["process"], record["command"], source, record["start_ticks"])
                pids = [r["process"].pid for r in active.values()]
                gpu = gpu_state(contract, pids)
                for _ in range(min(3 - len(active), len(pending))):
                    job = pending.pop(0)
                    folder = root / "run" / job["id"]
                    write(folder / f"{stage}_claim.json", {"job_id": job["id"], "stage": stage,
                        "claimed_at": now(), "contract_sha256": sha_json(contract)}, exclusive=True)
                    deadline_unix = fit_deadline(contract, job, time.time()) if stage == "train" else contract["absolute_host_deadline_unix"]
                    require(time.time() < deadline_unix, "Original fit wall-clock deadline expired")
                    command = worker_command(contract, contract_path, job, stage)
                    prefix = root / "logs" / f"{job['id']}.{stage}"
                    started_at = now()
                    with prefix.with_name(prefix.name + ".stdout.log").open("xb") as out, prefix.with_name(prefix.name + ".stderr.log").open("xb") as err:
                        child = subprocess.Popen(command, cwd=source, env={**os.environ, **contract["environment"],
                            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": str(job["seed"])},
                            stdout=out, stderr=err, start_new_session=True)
                    record = {"process": child, "job": job, "command": command, "prefix": prefix, "started_at": started_at,
                              "began": time.monotonic(), "start_ticks": None, "deadline_unix": deadline_unix}
                    active[job["id"]] = record
                    identity = _identity(child, command, source)
                    record["start_ticks"] = identity["start_ticks"] if identity else None
                    write(folder / f"{stage}_launch.json", {"job_id": job["id"], "stage": stage, "pid": child.pid,
                        "start_ticks": record["start_ticks"], "command": command, "started_at": started_at,
                        "boot_id": contract["boot_id"]}, exclusive=True)
                    write(folder / f"{stage}_lease.json", {"job_id": job["id"], "stage": stage, "pid": child.pid,
                        "start_ticks": record["start_ticks"], "command": command, "started_at": started_at,
                        "boot_id": contract["boot_id"], "contract_sha256": sha_json(contract)}, exclusive=True)
            except Exception:
                failure = failure or {"stage": stage, "at": now(), "traceback": traceback.format_exc()}
        if failure and not (root / "dispatch_failure.json").exists():
            write(root / "dispatch_failure.json", {**failure, "children_left_running": {k: v["process"].pid for k, v in active.items()}}, exclusive=True)
        write(root / "status.json", {"status": "draining_after_failure" if failure else "running", "host": HOST, "stage": stage,
            "observed_at": now(), "contract_sha256": sha_json(contract), "active": {k: {"pid": v["process"].pid,
                "start_ticks": v["start_ticks"], "command": v["command"], "started_at": v["started_at"]} for k, v in active.items()},
            "pending_job_ids": [j["id"] for j in pending], "actual_worker_exits": exits, "gpu": gpu, "failure": failure})
        if active or (pending and failure is None):
            time.sleep(2)
    result = {"stage": stage, "status": "complete" if not failure and len(exits) == len(keys) else "failed",
              "exits": exits, "receipts": receipts, "failure": failure, "pending_job_ids": [j["id"] for j in pending]}
    write(root / f"{stage}_stage_receipt.json", result, exclusive=True)
    return result


def artifact_manifest(root):
    excluded = {"artifact_manifest.json", "final_receipt.json", "dispatcher.stdout.log", "dispatcher.stderr.log", "dispatcher_exit.json", "supervisor_exit.json"}
    inventory = {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob("*")) if p.is_file()
                 and p.relative_to(root).parts[0] not in ("source", "cache") and str(p.relative_to(root)) not in excluded}
    require(inventory, "Empty recovery artifact manifest")
    value = {"schema": SCHEMA, "files": inventory, "excluded_lifecycle_files": sorted(excluded)}
    write(root / "artifact_manifest.json", value, exclusive=True)
    return {"path": "artifact_manifest.json", "sha256": sha_file(root / "artifact_manifest.json"), "file_count": len(inventory)}


def dispatch(contract, contract_path):
    validate_contract(contract)
    root, source = Path(contract["root"]), Path(contract["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Dispatcher cwd mismatch")
    require(not any((root / n).exists() for n in ("dispatch_attempt.json", "start.json", "run", "dispatch_failure.json", "final_receipt.json")), "Existing recovery run; no retry")
    verify_files(contract)
    qualification = read(root / "qualification.json")
    require(qualification.get("status") == "passed" and qualification.get("contract_sha256") == sha_json(contract)
            and qualification.get("host") == HOST and qualification.get("native_worker_exit0_count") == 6,
            "Native qualification incomplete")
    gpu_state(contract)
    write(root / "dispatch_attempt.json", {"contract_sha256": sha_json(contract), "started_at": now()}, exclusive=True)
    write(root / "start.json", {"host": HOST, "pid": os.getpid(), "started_at": now(), "boot_id": contract["boot_id"],
        "contract_sha256": sha_json(contract), "qualification_sha256": sha_file(root / "qualification.json"),
        "training_job_count": 4, "validation_job_count": 10, "maximum_concurrent_workers": 3}, exclusive=True)
    (root / "logs").mkdir(exist_ok=False)
    queue_started = time.monotonic()
    stages = {}
    for stage in STAGES:
        stages[stage] = run_stage(contract, contract_path, stage, queue_started)
        if stages[stage]["status"] != "complete":
            break
    pids = [e["pid"] for stage in stages.values() for e in stage["exits"].values()]
    gpu = gpu_state(contract)
    absent = all(proc_identity(pid) is None for pid in pids)
    originals = verify_originals(contract)
    verify_files(contract)
    complete = len(stages) == 2 and all(s["status"] == "complete" for s in stages.values()) and absent
    terminal = {"schema": SCHEMA, "status": "complete" if complete else "failed", "host": HOST, "completed_at": now(),
        "contract_sha256": sha_json(contract), "scientific_contract_sha256": contract["scientific_contract"]["canonical_sha256"],
        "completed_fit_count": len(stages.get("train", {}).get("receipts", {})),
        "validation_replay_count": len(stages.get("validation", {}).get("receipts", {})),
        "resumed_fit_count": 3, "fresh_fit_count": 1, "stored_raf_reference_count": 6, "test_count": 0,
        "stages": stages, "owned_worker_pids_absent": absent, "gpu_after": gpu,
        "immutable_originals_verification": originals, "held_out_test_evaluated": False,
        "historical_cnn_changed": False, "dispatcher_own_exit_directly_observed": False}
    write(root / "status.json", terminal)
    terminal["artifact_manifest"] = artifact_manifest(root)
    write(root / "final_receipt.json", terminal, exclusive=True)
    return 0 if complete else 1


def cleanup_owned_workers(contract, contract_path, *, evidence_name="supervisor_timeout_cleanup.json"):
    """Last-resort timeout cleanup is restricted to exact new-root worker leases.

    These workers are children of the dispatcher, so this supervisor records
    observed disappearance and signals, never invents their OS return codes.
    """
    root = Path(contract["root"])
    records = []
    by_id = {j["id"]: j for j in contract["jobs"]}
    paths = sorted(root.glob("qualification_*.lease.json")) + sorted((root / "run").glob("*/*_lease.json"))
    for lease_path in paths:
        lease = read(lease_path)
        require(lease.get("job_id") in by_id and lease.get("stage") in (*STAGES, "qualify"), "Unapproved cleanup lease")
        job, stage = by_id[lease["job_id"]], lease["stage"]
        lease = read(lease_path)
        require(lease.get("contract_sha256") == sha_json(contract) and lease.get("boot_id") == contract["boot_id"]
                and lease.get("job_id") == job["id"] and lease.get("stage") == stage, "Cleanup lease binding changed")
        live = proc_identity(lease["pid"])
        value = {"job_id": job["id"], "stage": stage, "pid": lease["pid"], "lease_sha256": sha_file(lease_path),
                 "actual_os_returncode": None, "exit_observer": "dispatcher child; supervisor cannot reap this worker", "signals": []}
        if live is not None:
            expected = worker_command(contract, contract_path, job, stage)
            require(live["pid"] == live["pgid"] == lease["pid"] and live["start_ticks"] == lease["start_ticks"]
                    and live["argv"] == lease["command"] == expected and live["cwd"] == contract["source_root"],
                    "Cleanup ownership is unconfirmed; no signal sent")
            for sig in (signal.SIGTERM, signal.SIGKILL):
                current = proc_identity(lease["pid"])
                if current is None:
                    break
                require(all(current[k] == live[k] for k in ("pid", "pgid", "start_ticks", "argv", "cwd")),
                        "Cleanup worker identity changed; no further signal")
                try:
                    os.killpg(lease["pid"], sig)
                except ProcessLookupError:
                    break
                value["signals"].append(int(sig))
                began = time.monotonic()
                while proc_identity(lease["pid"]) is not None and time.monotonic() - began < OWNED_CLEANUP_GRACE_SECONDS:
                    time.sleep(.05)
        value["owned_pid_absent"] = proc_identity(lease["pid"]) is None
        require(value["owned_pid_absent"], "Owned timeout worker remains after bounded cleanup")
        records.append(value)
    write(root / evidence_name, {"at": now(), "contract_sha256": sha_json(contract),
        "workers": records, "unverified_processes_signalled": False}, exclusive=True)
    return records


def launch_dispatch(contract, contract_path):
    root, source = Path(contract["root"]), Path(contract["source_root"])
    validate_contract(contract)
    require(Path.cwd().resolve() == source.resolve(), "Launch cwd mismatch")
    require(not any((root / n).exists() for n in ("launch_attempt.json", "qualification_attempt.json", "qualification.json", "start.json", "run")), "Existing recovery attempt; no retry")
    verify_files(contract)
    gpu_state(contract)
    write(root / "launch_attempt.json", {"started_at": now(), "contract_sha256": sha_json(contract), "boot_id": contract["boot_id"]}, exclusive=True)
    command = [contract["python"], "-u", "-m", MODULE, "--contract", str(contract_path), "--mode", "qualify"]
    prefix = root / "qualification"
    with prefix.with_name(prefix.name + ".stdout.log").open("xb") as out, prefix.with_name(prefix.name + ".stderr.log").open("xb") as err:
        child = subprocess.Popen(command, cwd=source, env={**os.environ, **contract["environment"], "PYTHONDONTWRITEBYTECODE": "1"}, stdout=out, stderr=err, start_new_session=True)
    qualification_identity = _identity(child, command, source)
    qualification_ticks = qualification_identity["start_ticks"] if qualification_identity else None
    timed_out, began, started = False, time.monotonic(), now()
    try:
        child.wait(timeout=max(.001, min(contract["limits"]["evaluation_wall_seconds"], contract["absolute_host_deadline_unix"] - time.time())))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            child.wait(timeout=SUPERVISOR_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            cleanup_owned_workers(contract, contract_path, evidence_name="qualification_timeout_cleanup.json")
            terminate_owned_child(child)
    record = _exit_record(contract, child, prefix, command, timeout=timed_out, started_at=started, start_ticks=qualification_ticks)
    write(root / "qualification_exit.json", record, exclusive=True)
    require(child.returncode == 0 and not timed_out, "Recovery qualification failed; no automatic retry")
    require(time.monotonic() - began < contract["limits"]["queue_wall_seconds"], "Qualification exhausted queue limit")
    # The launch supervisor directly observes the separate dispatcher child.
    # It does not claim to observe its own OS exit.
    command = [contract["python"], "-u", "-m", MODULE, "--contract", str(contract_path), "--mode", "dispatch"]
    prefix = root / "dispatcher"
    started = now()
    with prefix.with_name(prefix.name + ".stdout.log").open("xb") as out, prefix.with_name(prefix.name + ".stderr.log").open("xb") as err:
        child = subprocess.Popen(command, cwd=source, env={**os.environ, **contract["environment"], "PYTHONDONTWRITEBYTECODE": "1"}, stdout=out, stderr=err, start_new_session=True)
    identity = _identity(child, command, source)
    ticks = identity["start_ticks"] if identity else None
    write(root / "dispatcher_launch.json", {"pid": child.pid, "start_ticks": ticks, "command": command,
        "started_at": started, "contract_sha256": sha_json(contract), "boot_id": contract["boot_id"]}, exclusive=True)
    timed_out = False
    try:
        child.wait(timeout=max(.001, min(contract["limits"]["queue_wall_seconds"] - (time.monotonic() - began),
            contract["absolute_host_deadline_unix"] - time.time())))
    except subprocess.TimeoutExpired:
        timed_out = True
        # The dispatcher normally notices the same deadline and reaps its own
        # independent worker groups. Give that cleanup a bounded opportunity.
        try:
            child.wait(timeout=SUPERVISOR_CLEANUP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            cleanup_owned_workers(contract, contract_path)
            terminate_owned_child(child)
    record = _exit_record(contract, child, prefix, command, timeout=timed_out, started_at=started, start_ticks=ticks)
    write(root / "dispatcher_exit.json", record, exclusive=True)
    require(child.returncode == 0 and not timed_out, "Recovery dispatcher failed; original exit preserved")
    return 0


def status(contract):
    validate_contract(contract)
    verify_files(contract)
    root = Path(contract["root"])
    result = {"host": HOST, "observed_at": now(), "boot_id": boot_id(), "contract_sha256": sha_json(contract), "jobs": []}
    for name in ("launch_attempt.json", "qualification.json", "qualification_exit.json", "start.json", "status.json", "dispatch_failure.json", "final_receipt.json", "dispatcher_exit.json"):
        if (root / name).exists():
            result[name] = read(root / name)
    active = []
    for job in contract["jobs"]:
        entry = {"job": job, "stages": {}}
        for stage in STAGES:
            value = {}
            for kind in ("launch", "exit", "receipt"):
                path = root / "run" / job["id"] / f"{stage}_{kind}.json"
                if path.exists():
                    value[kind] = read(path)
            if "launch" in value and "exit" not in value:
                launch = value["launch"]
                live = proc_identity(launch["pid"])
                value["current_identity"] = live
                if live is not None:
                    require(live["start_ticks"] == launch["start_ticks"] and live["argv"] == launch["command"]
                            and live["cwd"] == contract["source_root"] and live["pgid"] == live["pid"], "Live recovery identity changed")
                    active.append(live["pid"])
            if value:
                entry["stages"][stage] = value
        histories = list((root / "run" / job["id"]).glob("runs/*/*/seed_*/history.json"))
        if len(histories) == 1:
            entry["history"] = read(histories[0]).get("history", [])
        result["jobs"].append(entry)
    result["gpu"] = gpu_state(contract, active)
    if "start.json" in result and "dispatcher_exit.json" not in result:
        result["dispatcher_current_identity"] = proc_identity(result["start.json"]["pid"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--host", choices=(HOST,), default=HOST)
    parser.add_argument("--mode", choices=("qualify", "launch_dispatch", "dispatch", "status", "wait"), required=True)
    args = parser.parse_args(argv)
    require(args.contract.is_absolute(), "Recovery contract must be absolute")
    contract = read(args.contract)
    if args.mode == "launch_dispatch":
        return launch_dispatch(contract, args.contract)
    if args.mode == "dispatch":
        return dispatch(contract, args.contract)
    output = qualify(contract, args.contract) if args.mode == "qualify" else status(contract)
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
