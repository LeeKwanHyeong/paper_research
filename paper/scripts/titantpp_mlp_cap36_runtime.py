"""Bounded Cap-data MLP queue: train, selected Validation replay, then Test.

Each host owns at most three fresh child processes. Existing experiments are
never resumed or modified; only this dispatcher's timed-out child groups may
be terminated. The dispatcher itself stays standard-library-only.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import (
    gpu_snapshot, local_path, now, read, require, runtime_check, sha_file,
    sha_json, terminate_owned_child, verify_files, write,
)
from paper.scripts.titantpp_cross_fusion_seed_extension_runtime import proc_identity

SCHEMA = "titantpp_mlp_cap36_v1"
MODULE = "paper.scripts.titantpp_mlp_cap36_runtime"
WORKER_MODULE = "paper.scripts.run_titantpp_mlp_cap36"
ARMS = ("titantpp_history_mlp", "titantpp_history_mlp_width16",
        "titantpp_all_available_history_mlp", "titantpp_current_only_param_matched")
STAGES = ("train", "validation", "test")


def validate_contract(contract, host):
    require(contract.get("schema") == SCHEMA, "Wrong Cap36 contract")
    require(set(contract["hosts"]) == {"5080", "5090"} and host in contract["hosts"], "Host mismatch")
    datasets = {d["dataset_id"] for d in contract["datasets"]}
    jobs = contract["jobs"]
    require(len(datasets) == len(contract["datasets"]) == 3, "Expected three datasets")
    require(len(jobs) == len({j["id"] for j in jobs}) == 36, "Expected 36 unique fits")
    require({(j["dataset"], j["arm"], j["seed"]) for j in jobs}
            == {(d, a, s) for d in datasets for a in ARMS for s in (42, 52, 62)}, "Fit matrix mismatch")
    for job in jobs:
        require(job["host"] in contract["hosts"] and job["id"]
                == f"{job['dataset']}__{job['arm']}__seed{job['seed']}", "Job identity mismatch")
        require(Path(job["id"]).name == job["id"] and job["id"] not in {".", ".."}, "Unsafe job ID")
    training = contract["training"]
    require(all(training.get(k) == v for k, v in {
        "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40, "batch_size": 128}.items()),
        "Common training budget changed")
    require(training.get("monitor", "validation_raw_quantity_rmse") == "validation_raw_quantity_rmse",
            "Checkpoint selector changed")
    require(contract["limits"] == {"fit_wall_seconds": 129600, "queue_wall_seconds": 604800,
                                   "evaluation_wall_seconds": 7200}, "Fixed limits changed")
    require(contract["source"]["files"] and contract["input_files"], "Empty source/input inventory")
    require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source inventory mismatch")
    target = contract["hosts"][host]
    require(all(Path(target[k]).is_absolute() for k in ("root", "source_root", "python")), "Host path must be absolute")
    require(Path(target["source_root"]) == Path(target["root"]) / "source", "Expected isolated source directory")
    require(contract["hosts"]["5080"]["root"] != contract["hosts"]["5090"]["root"], "Host roots must differ")
    assigned = [j for j in jobs if j["host"] == host]
    require(assigned, "Host has no assigned jobs")
    return assigned


def qualify(contract, host):
    validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Qualification cwd mismatch")
    require(not any((root / p).exists() for p in ("qualification.json", "start.json", "run", "dispatch_attempt.json")),
            "Existing qualification/start/run; no automatic retry")
    verify_files(contract, source)
    write(root / "qualification_attempt.json", {"host": host, "started_at": now(),
          "contract_sha256": sha_json(contract)}, exclusive=True)
    gpu = gpu_snapshot(target["gpu_uuid"])
    require(not gpu["compute_pids"], "GPU is not idle")
    runtime = runtime_check(target)
    from paper.scripts.run_titantpp_mlp_cap36 import synthetic_check
    synthetic = synthetic_check(contract, host)
    require(synthetic["status"] == "passed" and synthetic.get("real_data_loaded") is False,
            "Native synthetic qualification failed")
    peaks = synthetic["cuda_memory_peaks"]
    require(set(peaks) == set(ARMS), "All four arms require native qualification")
    peak = max(p["max_memory_reserved_bytes"] for p in peaks.values())
    require(all(type(p["max_memory_reserved_bytes"]) is int and p["max_memory_reserved_bytes"] > 0
                for p in peaks.values()), "Invalid native memory measurement")
    require(3 * peak < .85 * runtime["gpu"]["total_memory_bytes"], "Three-worker memory admission failed")
    result = {"status": "passed", "host": host, "verified_at": now(), "contract_sha256": sha_json(contract),
        "gpu_before": gpu, "runtime": runtime, "synthetic": synthetic,
        "parallel_memory_admission": {"maximum_workers": 3, "maximum_reserved_bytes_per_worker": peak,
            "three_worker_reserved_estimate_bytes": 3 * peak, "passed": True,
            "limitation": "Sequential full-shape synthetic peaks; not a measured three-process peak."}}
    write(root / "qualification.json", result, exclusive=True)
    return result


def worker_command(contract, host, contract_path, job, stage):
    require(stage in STAGES, "Unknown worker stage")
    return [contract["hosts"][host]["python"], "-u", "-m", WORKER_MODULE,
            "--contract", str(contract_path), "--host", host, "--job", job["id"], "--stage", stage]


def available_slots(active_pids, gpu_pids):
    require(len(active_pids) == len(set(active_pids)) <= 3, "Worker concurrency exceeded")
    require(set(gpu_pids).issubset(active_pids), "Unidentified GPU process; no new jobs may start")
    return 3 - len(active_pids)


def verify_receipt(contract, host, job, stage):
    root = Path(contract["hosts"][host]["root"])
    path = root / "run" / job["id"] / f"{stage}_receipt.json"
    receipt = read(path)
    require(receipt.get("status") == "complete" and receipt.get("job_id") == job["id"]
            and receipt.get("stage") == stage and receipt.get("contract_sha256") == sha_json(contract),
            "Worker receipt identity/completion mismatch")
    files = receipt.get("files")
    require(isinstance(files, dict) and files, "Worker artifact inventory is empty")
    job_root = (root / "run" / job["id"]).resolve()
    for name, digest in files.items():
        artifact = local_path(root, name)
        require(artifact.is_relative_to(job_root) and artifact != path.resolve(), "Artifact outside assigned job or self-reference")
        require(sha_file(artifact) == digest, "Worker artifact SHA mismatch: " + name)
    if stage == "validation":
        require(receipt.get("validation_replay_passed") is True, "Validation replay did not pass")
    return {"path": str(path.relative_to(root)), "sha256": sha_file(path)}


def validation_gate(contract, host, exits):
    jobs = validate_contract(contract, host)
    require(set(exits) == {j["id"] for j in jobs} and all(
        e["actual_os_returncode"] == 0 and not e["timeout"] for e in exits.values()),
        "Every assigned Validation worker must exit successfully")
    receipts = {j["id"]: {"validation_receipt_sha256": verify_receipt(contract, host, j, "validation")["sha256"]}
                for j in jobs}
    result = {"status": "passed", "host": host, "contract_sha256": sha_json(contract),
              "created_at": now(), "jobs": receipts}
    write(Path(contract["hosts"][host]["root"]) / "validation_gate.json", result, exclusive=True)
    return result


def _check_child(record, source):
    identity = proc_identity(record["process"].pid)
    if identity is None:
        return  # poll() reaps a concurrent exit; the slot remains reserved.
    require(identity["start_ticks"] == record["start_ticks"] and identity["pgid"] == identity["pid"]
            and identity["argv"] == record["command"] and identity["cwd"] == str(source),
            "Owned child process identity changed")


def run_stage(contract, host, contract_path, stage, queue_started):
    jobs = validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    pending, active, exits, receipts, failure = list(jobs), {}, {}, {}, None
    timeout = contract["limits"]["fit_wall_seconds" if stage == "train" else "evaluation_wall_seconds"]
    env = {**os.environ, **target["environment"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    while active or (pending and failure is None):
        for job_id, record in list(active.items()):
            process = record["process"]
            timed_out = process.poll() is None and (
                time.monotonic() - record["started_monotonic"] > timeout
                or time.monotonic() - queue_started > contract["limits"]["queue_wall_seconds"])
            if timed_out:
                terminate_owned_child(process)
            code = process.poll()
            if code is None:
                continue
            exit_record = {"job_id": job_id, "stage": stage, "pid": process.pid,
                "start_ticks": record["start_ticks"], "actual_os_returncode": code, "timeout": timed_out,
                "started_at": record["started_at"], "observed_exit_at": now()}
            write(root / "run" / job_id / f"{stage}_exit.json", exit_record, exclusive=True)
            exits[job_id] = exit_record
            del active[job_id]
            try:
                require(code == 0 and not timed_out, "Worker failed or timed out")
                receipts[job_id] = verify_receipt(contract, host, record["job"], stage)
            except Exception:
                failure = failure or {"stage": stage, "job_id": job_id, "at": now(), "traceback": traceback.format_exc()}
        gpu = None
        if failure is None:
            try:
                require(time.monotonic() - queue_started <= contract["limits"]["queue_wall_seconds"], "Queue deadline exceeded")
                for record in active.values():
                    _check_child(record, source)
                gpu = gpu_snapshot(target["gpu_uuid"])
                slots = available_slots([r["process"].pid for r in active.values()], gpu["compute_pids"])
                for _ in range(min(slots, len(pending))):
                    job = pending.pop(0)
                    directory = root / "run" / job["id"]
                    write(directory / f"{stage}_claim.json", {"job_id": job["id"], "stage": stage,
                        "claimed_at": now(), "contract_sha256": sha_json(contract)}, exclusive=True)
                    command = worker_command(contract, host, contract_path, job, stage)
                    with (root / "logs" / f"{job['id']}.{stage}.stdout.log").open("xb") as out, (
                            root / "logs" / f"{job['id']}.{stage}.stderr.log").open("xb") as err:
                        process = subprocess.Popen(command, cwd=source,
                            env={**env, "PYTHONHASHSEED": str(job["seed"])}, stdout=out, stderr=err, start_new_session=True)
                    # Register the child before reading /proc so any read failure drains it.
                    active[job["id"]] = record = {"process": process, "job": job, "command": command,
                        "started_monotonic": time.monotonic(), "started_at": now(), "start_ticks": None}
                    identity = proc_identity(process.pid)
                    record["start_ticks"] = identity["start_ticks"] if identity else None
                    write(directory / f"{stage}_launch.json", {"job_id": job["id"], "stage": stage,
                        "pid": process.pid, "start_ticks": record["start_ticks"], "command": command,
                        "started_at": record["started_at"]}, exclusive=True)
            except Exception:
                failure = failure or {"stage": stage, "at": now(), "traceback": traceback.format_exc()}
        if failure is not None and not (root / "dispatch_failure.json").exists():
            write(root / "dispatch_failure.json", {**failure, "children_left_running": {
                key: r["process"].pid for key, r in active.items()}}, exclusive=True)
        write(root / "status.json", {"status": "draining_after_failure" if failure else "running",
            "host": host, "stage": stage, "observed_at": now(), "contract_sha256": sha_json(contract),
            "active": {key: {"pid": r["process"].pid, "start_ticks": r["start_ticks"], "command": r["command"],
                "started_at": r["started_at"]} for key, r in active.items()},
            "pending_job_ids": [j["id"] for j in pending], "actual_worker_exits": exits, "gpu": gpu,
            "failure": failure})
        if active or (pending and failure is None):
            time.sleep(2)
    result = {"stage": stage, "status": "complete" if not failure and len(exits) == len(jobs) else "failed",
              "exits": exits, "receipts": receipts, "failure": failure, "pending_job_ids": [j["id"] for j in pending]}
    write(root / f"{stage}_stage_receipt.json", result, exclusive=True)
    return result


def artifact_manifest(root):
    # The supervisor's dispatcher_exit.json does not yet exist. The collector
    # binds that receipt plus final_receipt.json and the closed dispatcher log.
    excluded = {"artifact_manifest.json", "final_receipt.json", "dispatcher.log", "dispatcher_exit.json"}
    files = {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob("*"))
             if p.is_file() and p.relative_to(root).parts[0] not in {"source", "cache"}
             and str(p.relative_to(root)) not in excluded}
    require(files, "Empty artifact manifest")
    value = {"schema": "titantpp_mlp_cap36_artifacts_v1", "files": files,
             "excluded_lifecycle_files": sorted(excluded), "excluded_directories": ["source", "cache"]}
    write(root / "artifact_manifest.json", value, exclusive=True)
    return {"path": "artifact_manifest.json", "sha256": sha_file(root / "artifact_manifest.json"), "file_count": len(files)}


def dispatch(contract, host, contract_path, *, queue_started=None):
    jobs = validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Dispatcher cwd mismatch")
    require(not any((root / n).exists() for n in ("start.json", "run", "dispatch_failure.json", "final_receipt.json")),
            "Existing run or failure; no restart/retry")
    verify_files(contract, source)
    qualification = read(root / "qualification.json")
    require(qualification.get("status") == "passed" and qualification.get("host") == host
            and qualification.get("contract_sha256") == sha_json(contract), "Qualification mismatch")
    write(root / "dispatch_attempt.json", {"host": host, "at": now(), "contract_sha256": sha_json(contract)}, exclusive=True)
    require(not gpu_snapshot(target["gpu_uuid"])["compute_pids"], "GPU occupied before dispatch")
    write(root / "start.json", {"host": host, "started_at": now(), "pid": os.getpid(),
        "contract_sha256": sha_json(contract), "qualification_sha256": sha_file(root / "qualification.json"),
        "maximum_concurrent_workers": 3, "jobs": jobs}, exclusive=True)
    (root / "logs").mkdir(exist_ok=False)
    queue_started = time.monotonic() if queue_started is None else queue_started
    stages = {}
    for stage in STAGES:
        result = run_stage(contract, host, contract_path, stage, queue_started)
        stages[stage] = result
        if result["status"] != "complete":
            break
        if stage == "validation":
            validation_gate(contract, host, result["exits"])
    pids = [e["pid"] for result in stages.values() for e in result["exits"].values()]
    gpu = gpu_snapshot(target["gpu_uuid"])
    absent = all(proc_identity(pid) is None for pid in pids) and not set(pids).intersection(gpu["compute_pids"])
    complete = len(stages) == 3 and all(r["status"] == "complete" for r in stages.values()) and absent
    terminal = {"status": "complete" if complete else "failed", "host": host, "completed_at": now(),
        "contract_sha256": sha_json(contract), "expected_job_count": len(jobs),
        "completed_fit_count": len(stages.get("train", {}).get("receipts", {})),
        "validation_replay_count": len(stages.get("validation", {}).get("receipts", {})),
        "test_count": len(stages.get("test", {}).get("receipts", {})), "stages": stages,
        "owned_worker_pids_absent": absent, "gpu_after": gpu, "held_out_test_evaluated": "test" in stages,
        "existing_models_retrained": False, "dispatcher_own_exit_directly_observed": False}
    write(root / "status.json", terminal)
    terminal["artifact_manifest"] = artifact_manifest(root)
    write(root / "final_receipt.json", terminal, exclusive=True)
    return 0 if complete else 1


def qualify_child(contract, host, contract_path, remaining_seconds):
    """A fresh process releases its CUDA context before dispatch checks idleness."""
    target = contract["hosts"][host]
    root = Path(target["root"])
    command = [target["python"], "-u", "-m", MODULE, "--contract", str(contract_path),
               "--host", host, "--mode", "qualify"]
    with (root / "qualification.stdout.log").open("xb") as out, (root / "qualification.stderr.log").open("xb") as err:
        process = subprocess.Popen(command, cwd=target["source_root"],
            env={**os.environ, **target["environment"]}, stdout=out, stderr=err, start_new_session=True)
        timed_out = False
        try:
            code = process.wait(timeout=min(remaining_seconds, contract["limits"]["evaluation_wall_seconds"]))
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_owned_child(process)
            code = process.returncode
    write(root / "qualification_exit.json", {"pid": process.pid, "command": command,
        "actual_os_returncode": code, "timeout": timed_out, "observed_exit_at": now()}, exclusive=True)
    require(code == 0 and not timed_out, "Native qualification child failed or timed out")


def wait_dispatch(contract, host, contract_path):
    """Wait without CUDA imports; retain the same queue deadline after admission."""
    validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Wait cwd mismatch")
    require(not any((root / name).exists() for name in ("qualification_attempt.json", "qualification.json",
        "dispatch_attempt.json", "start.json", "run", "dispatch_failure.json", "final_receipt.json")),
        "Existing attempt/run; wait is only allowed for a fresh bootstrap")
    verify_files(contract, source)
    started = time.monotonic()
    start = {"host": host, "started_at": now(), "contract_sha256": sha_json(contract),
             "queue_wall_seconds": contract["limits"]["queue_wall_seconds"], "poll_seconds": 60}
    write(root / "wait_start.json", start, exclusive=True)
    try:
        while True:
            elapsed = time.monotonic() - started
            require(elapsed < contract["limits"]["queue_wall_seconds"], "Queue deadline expired while waiting for GPU")
            gpu = gpu_snapshot(target["gpu_uuid"])
            write(root / "waiting.json", {**start, "status": "waiting_for_idle_gpu" if gpu["compute_pids"] else "idle_verified",
                "observed_at": now(), "elapsed_seconds": elapsed, "gpu": gpu,
                "training_started": False, "native_synthetic_started": False})
            if not gpu["compute_pids"]:
                break
            time.sleep(min(60, contract["limits"]["queue_wall_seconds"] - elapsed))
        qualify_child(contract, host, contract_path, contract["limits"]["queue_wall_seconds"] - (time.monotonic() - started))
        require(time.monotonic() - started < contract["limits"]["queue_wall_seconds"], "Queue deadline expired during qualification")
        write(root / "waiting.json", {**start, "status": "qualification_passed_dispatching", "observed_at": now(),
            "elapsed_seconds": time.monotonic() - started, "native_synthetic_started": True})
        return dispatch(contract, host, contract_path, queue_started=started)
    except Exception:
        write(root / "wait_failure.json", {**start, "failed_at": now(), "traceback": traceback.format_exc()}, exclusive=True)
        raise


def status(contract, host):
    jobs = validate_contract(contract, host)
    target = contract["hosts"][host]
    root = Path(target["root"])
    result = {"host": host, "observed_at": now(), "contract_sha256": sha_json(contract),
              "gpu": gpu_snapshot(target["gpu_uuid"]), "jobs": []}
    for name in ("waiting.json", "wait_failure.json", "qualification.json", "start.json", "status.json",
                 "dispatch_failure.json", "final_receipt.json", "dispatcher_exit.json"):
        path = root / name
        if path.exists():
            result[name] = read(path)
    if "start.json" in result and "dispatcher_exit.json" not in result:
        result["dispatcher_current_identity"] = proc_identity(result["start.json"]["pid"])
    for job in jobs:
        entry = {"job": job, "stages": {}}
        for stage in STAGES:
            value = {}
            for kind in ("launch", "exit", "receipt"):
                path = root / "run" / job["id"] / f"{stage}_{kind}.json"
                if path.exists():
                    value[kind] = read(path)
            if "launch" in value and "exit" not in value:
                value["current_identity"] = proc_identity(value["launch"]["pid"])
            if value:
                entry["stages"][stage] = value
        # Train history is read only for progress; no re-evaluation is performed.
        histories = list((root / "run" / job["id"]).glob("runs/*/*/seed_*/history.json"))
        if len(histories) == 1:
            entry["history"] = read(histories[0]).get("history", [])
        result["jobs"].append(entry)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("qualify", "dispatch", "wait_dispatch", "status"), required=True)
    args = parser.parse_args(argv)
    require(args.contract.is_absolute(), "Contract path must be absolute")
    contract = read(args.contract)
    if args.mode == "dispatch":
        return dispatch(contract, args.host, args.contract)
    if args.mode == "wait_dispatch":
        return wait_dispatch(contract, args.host, args.contract)
    import json
    output = qualify(contract, args.host) if args.mode == "qualify" else status(contract, args.host)
    print(json.dumps(output, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
