"""Seed52/62 extension with a shared three-fit limit across parent and new roots.

The original seed42 experiment is read-only. A missing parent exit record reserves
its slot; only the new dispatcher's own children may be timed out and terminated.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import time
import traceback

from paper.scripts import titantpp_cross_fusion_precheck_runtime as parent_runtime
from paper.scripts.titantpp_cross_fusion_precheck_runtime import (
    gpu_snapshot, local_path, now, read, require, runtime_check, sha_file,
    sha_json, terminate_owned_child, verify_files, write,
)

SCHEMA = "titantpp_cross_fusion_seed_extension12_v1"
MODULE = "paper.scripts.titantpp_cross_fusion_seed_extension_runtime"


def validate_contract(contract, host):
    require(contract.get("schema") == SCHEMA, "Wrong seed-extension schema")
    require(contract.get("evaluation_scope") == "validation_only"
            and contract.get("held_out_test_evaluated") is False, "Validation only")
    require(set(contract["hosts"]) == {"5080", "5090"} and host in contract["hosts"], "Host mismatch")
    jobs = contract["jobs"]
    datasets = {d["dataset_id"] for d in contract["datasets"]}
    arms = set(contract["architecture"]["arms"])
    require(len(datasets) == 3 and len(arms) == 2 and len(jobs) == 12,
            "Expected three datasets, two arms, and twelve new fits")
    require(len({j["id"] for j in jobs}) == 12, "Duplicate jobs")
    require({(j["dataset"], j["arm"], j["seed"]) for j in jobs}
            == {(d, a, s) for d in datasets for a in arms for s in (52, 62)}, "Seed matrix mismatch")
    require(all(j["id"] == f"{j['dataset']}__{j['arm']}__seed{j['seed']}" for j in jobs), "Job ID mismatch")
    require(all(sum(j["host"] == h for j in jobs) == 6 for h in contract["hosts"]), "Host assignment count")
    require(contract["training"] == {"maximum_epochs": 300, "minimum_epochs": 40,
        "patience": 40, "batch_size": 128, "monitor": "validation_raw_quantity_rmse"}, "Training changed")
    require(contract["limits"] == {"fit_wall_seconds": 129600, "queue_wall_seconds": 259200}, "Limits changed")
    require(contract["source"]["files"] and contract["input_files"], "Empty inventory")
    require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source inventory mismatch")
    target = contract["hosts"][host]
    for key in ("root", "source_root", "parent_root", "python"):
        require(Path(target[key]).is_absolute(), "Host path must be absolute: " + key)
    require(target["root"] != target["parent_root"], "Parent root cannot be reused")
    return [j for j in jobs if j["host"] == host]


def verify_parent_binding(contract, host):
    target = contract["hosts"][host]
    root = Path(target["parent_root"])
    path = root / "execution_contract.json"
    binding = contract["parent"]
    require(sha_file(path) == binding["contract_sha256"], "Parent contract SHA mismatch")
    parent = read(path)
    parent_runtime.validate_contract(parent, host)
    parent_runtime.verify_files(parent, Path(parent["hosts"][host]["source_root"]))
    require(sha_json(parent) == binding["canonical_sha256"], "Parent canonical mismatch")
    require(parent["source"]["files_sha256"] == binding["source_files_sha256"], "Parent source closure mismatch")
    require(parent["hosts"][host]["root"] == str(root), "Wrong parent root")
    for key in ("datasets", "input_files", "architecture", "training"):
        require(contract[key] == parent[key], "Scientific contract changed: " + key)
    for key in ("python", "gpu_uuid", "runtime_expected"):
        require(target[key] == parent["hosts"][host][key], "Host runtime changed: " + key)
    assignments = {(j["dataset"], j["arm"]): j["host"] for j in parent["jobs"]}
    require(all(assignments[(j["dataset"], j["arm"])] == j["host"] for j in contract["jobs"]), "Parent assignment changed")
    for name, digest in parent["source"]["files"].items():
        require(contract["source"]["files"].get(name) == digest, "Inherited source changed: " + name)
        require(sha_file(local_path(Path(target["source_root"]), name)) == digest, "Inherited source bytes changed")
    qualification_path = root / "qualification.json"
    require(sha_file(qualification_path) == binding["qualification_sha256"][host], "Parent qualification SHA mismatch")
    qualification = read(qualification_path)
    require(qualification["status"] == "passed" and qualification["host"] == host
            and qualification["contract_sha256"] == sha_json(parent), "Invalid parent qualification")
    require(qualification["parallel_memory_admission"]["passed"] is True
            and qualification["parallel_memory_admission"]["maximum_workers"] == 3, "Memory admission missing")
    require(qualification["runtime"]["gpu"]["uuid"] == target["gpu_uuid"], "Qualified GPU mismatch")
    for key, value in target["runtime_expected"].items():
        require(qualification["runtime"][key] == value, "Qualified runtime mismatch")
    return parent, qualification


def proc_identity(pid, proc_root=Path("/proc")):
    base = Path(proc_root) / str(pid)
    try:
        stat = (base / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        argv = (base / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
        identity = {"pid": int(pid), "start_ticks": int(fields[19]), "pgid": int(fields[2]),
                    "argv": argv, "cwd": str((base / "cwd").resolve(strict=True))}
        again = (base / "stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None  # A normal exit may occur between any two /proc reads.
    require(int(again[again.rfind(")") + 2:].split()[19]) == identity["start_ticks"], "PID reused during read")
    return identity


def check_worker(identity, worker, job, target, module):
    require(identity["pid"] == worker["pid"] and identity["pgid"] == worker["pgid"] == worker["pid"], "Worker PID/PGID mismatch")
    expected = [target["python"], "-m", module, "--contract", target["root"] + "/execution_contract.json",
                "--host", job["host"], "--mode", "fit", "--job-id", job["id"]]
    require(identity["argv"] == expected and identity["cwd"] == target["source_root"], "Worker argv/cwd mismatch")
    require(worker["job"] == job, "Worker job mismatch")
    if "start_ticks" in worker:
        require(identity["start_ticks"] == worker["start_ticks"], "Worker PID reused")
    else:
        boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime "))
        born = boot + identity["start_ticks"] / os.sysconf("SC_CLK_TCK")
        require(abs(born - datetime.fromisoformat(worker["started_at"]).timestamp()) < 3, "Parent PID birth mismatch")


def parent_occupancy(parent, host):
    target = parent["hosts"][host]
    root = Path(target["root"])
    require(not (root / "dispatch_failure.json").exists(), "Parent dispatcher failure")
    status = read(root / "status.json") if (root / "status.json").exists() else {}
    require(not any(x["actual_os_returncode"] != 0 or x.get("timeout")
                    for x in status.get("actual_worker_exits", {}).values()), "Parent scientific worker failed")
    pids, occupied = [], []
    for job in [j for j in parent["jobs"] if j["host"] == host]:
        run = root / "run" / job["id"]
        exit_path, receipt_path = run / "worker_exit.json", run / "fit_receipt.json"
        if exit_path.exists():
            exit_record = read(exit_path)
            require(exit_record["job_id"] == job["id"] and exit_record["actual_os_returncode"] == 0
                    and not exit_record["timeout"], "Parent worker exit failure")
            if receipt_path.exists():
                receipt = read(receipt_path)
                require(receipt["status"] == "complete" and receipt["job"] == job
                        and receipt["contract_sha256"] == sha_json(parent), "Parent completion mismatch")
                continue
        occupied.append(job["id"])
        if not (run / "worker.json").exists():
            continue  # Parent launch/exit race reserves the slot.
        worker = read(run / "worker.json")
        require(worker["contract_sha256"] == sha_json(parent), "Parent worker contract mismatch")
        identity = proc_identity(worker["pid"])
        if identity is not None:
            check_worker(identity, worker, job, target, parent_runtime.MODULE)
            pids.append(worker["pid"])
    return {"occupied_job_ids": occupied, "live_pids": pids}


def available_slots(parent_state, active_pids, gpu_pids):
    require(len(active_pids) == len(set(active_pids)), "Duplicate new child PID")
    allowed = set(parent_state["live_pids"]) | set(active_pids)
    require(set(gpu_pids) <= allowed, "Unidentified GPU process; pending work stopped")
    occupied = len(parent_state["occupied_job_ids"]) + len(active_pids)
    require(occupied <= 3, "Shared three-fit concurrency cap exceeded")
    return 3 - occupied


def admit(contract, host):
    validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Admission cwd mismatch")
    require(not any((root / name).exists() for name in ("qualification.json", "start.json", "run", "dispatch_failure.json")), "Existing attempt; no retry")
    verify_files(contract, source)
    parent, qualification = verify_parent_binding(contract, host)
    state = parent_occupancy(parent, host)
    gpu = gpu_snapshot(target["gpu_uuid"])
    slots = available_slots(state, [], gpu["compute_pids"])
    receipt = {"status": "passed", "host": host, "contract_sha256": sha_json(contract),
        "verified_at": now(), "gpu_before": gpu, "parent_occupancy": state, "available_slots": slots,
        "runtime": qualification["runtime"], "parallel_memory_admission": qualification["parallel_memory_admission"],
        "inherited_native_qualification": contract["parent"]["qualification_sha256"][host],
        "native_synthetic_rerun": False, "held_out_test_evaluated": False}
    write(root / "qualification.json", receipt, exclusive=True)
    return receipt


def fit(contract, host, job_id):
    jobs = validate_contract(contract, host)
    matches = [j for j in jobs if j["id"] == job_id]
    require(len(matches) == 1, "Job is not assigned to this host")
    job, target = matches[0], contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Fit cwd must be isolated source")
    start = read(root / "start.json")
    require(start["contract_sha256"] == sha_json(contract) and start["host"] == host,
            "Missing or mismatched dispatcher start")
    output = root / "run" / job_id
    output.mkdir(parents=True, exist_ok=False)
    worker = {"status": "starting", "pid": os.getpid(), "pgid": os.getpgrp(),
        "started_at": now(), "start_ticks": proc_identity(os.getpid())["start_ticks"], "job": job, "contract_sha256": sha_json(contract)}
    write(output / "worker.json", worker)
    runtime = runtime_check(target)
    from paper.scripts import run_titantpp_cross_fusion as runner
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.count_aware_tpp_backbone import training
    runner.install_hooks()
    require(job["arm"] in runner.ARMS, "Historical/baseline arm is forbidden")
    data = next(d for d in contract["datasets"] if d["dataset_id"] == job["dataset"])
    frame, inputs = base.prepare_admitted_data(data, data_root=source)
    write(output / "input_receipt.json", inputs, exclusive=True)
    args = runner.training_args(contract, data, output, job["arm"])
    args.seed = job["seed"]
    require(os.environ.get("PYTHONHASHSEED") == str(job["seed"]), "Seed environment mismatch")
    require((args.epochs, args.min_epochs, args.early_stopping_patience, args.batch_size)
            == (300, 40, 40, 128) and args.checkpoint_monitor == "validation_raw_quantity_rmse",
            "Runner changed the common budget/selector")
    require(args.max_train_batches is None and args.max_val_batches is None and not args.force_rerun,
            "Partial batches or existing-run reuse are forbidden")
    interface = runner.time_interface(data, frame, contract, arm=job["arm"])
    quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
        "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(5)]}
    write(output / "worker.json", {**worker, "status": "training", "runtime": runtime})
    print(f"FIT_START {job_id} {now()}", flush=True)
    summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
        interface_meta=interface, backbone=job["arm"], quantity_variant=runner.VARIANT, seed=job["seed"])
    require(summary["seed"] == job["seed"], "Training summary seed mismatch")
    require(summary["status"] == "success" and summary["held_out_test_evaluated"] is False,
            "Scientific fit did not complete Validation-only")
    checkpoint = Path(summary["checkpoint_path"])
    run = checkpoint.parent
    history = read(run / "history.json")["history"]
    selected = training.earliest_strict_minimum(history, metric_key="val_qty_rmse")
    require(selected is not None and selected["epoch"] == summary["best_epoch"],
            "Selected epoch disagrees with the earliest finite Validation RMSE minimum")
    paths = [checkpoint, run / "last_epoch_state.pt", run / "history.json", run / "summary.json",
             output / "input_receipt.json"]
    receipt = {"status": "complete", "job": job, "completed_at": now(),
        "contract_sha256": sha_json(contract), "runtime": runtime,
        "summary": summary, "files": {str(p.relative_to(root)): sha_file(p) for p in paths},
        "held_out_test_evaluated": False, "selected_validation_replayed_by_train_one": True}
    write(output / "fit_receipt.json", receipt, exclusive=True)
    write(output / "worker.json", {**worker, "status": "complete", "completed_at": now(), "runtime": runtime})
    print(f"FIT_COMPLETE {job_id} best_epoch={summary['best_epoch']} "
          f"rmse={summary['best_val_qty_rmse']} mae={summary['best_val_qty_mae']}", flush=True)
    return receipt


def dispatch(contract, host, contract_path):
    jobs = validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Dispatcher cwd mismatch")
    verify_files(contract, source)
    parent, _ = verify_parent_binding(contract, host)
    qualification = read(root / "qualification.json")
    require(qualification["status"] == "passed" and qualification["contract_sha256"] == sha_json(contract), "Qualification mismatch")
    require(not any((root / name).exists() for name in ("start.json", "run", "dispatch_failure.json")), "Existing start/run; no retry")
    start = {"status": "queued", "host": host, "started_at": now(), "pid": os.getpid(),
        "contract_sha256": sha_json(contract), "qualification_sha256": sha_file(root / "qualification.json"),
        "jobs": jobs, "maximum_concurrent_fits_including_parent": 3}
    write(root / "start.json", start, exclusive=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=False)
    pending, active, exits, failure = list(jobs), {}, {}, None
    queue_started = time.monotonic()
    env = {**os.environ, **{k: str(v) for k, v in target.get("environment", {}).items()},
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    while active or (pending and failure is None):
        for job_id, (process, started, started_at, ticks) in list(active.items()):
            timed_out = process.poll() is None and (time.monotonic() - started > contract["limits"]["fit_wall_seconds"]
                or time.monotonic() - queue_started > contract["limits"]["queue_wall_seconds"])
            if timed_out:
                terminate_owned_child(process)
            code = process.poll()
            if code is None:
                continue
            record = {"job_id": job_id, "pid": process.pid, "start_ticks": ticks,
                "actual_os_returncode": code, "timeout": timed_out, "started_at": started_at, "observed_exit_at": now()}
            write(root / "run" / job_id / "worker_exit.json", record, exclusive=True)
            exits[job_id] = record
            del active[job_id]
            if code != 0 or timed_out:
                failure = failure or {"at": now(), "reason": "new_worker_failed", "worker_exit": record}
        state, gpu = None, None
        if failure is None:
            try:
                require(time.monotonic() - queue_started <= contract["limits"]["queue_wall_seconds"], "Queue deadline exceeded")
                state = parent_occupancy(parent, host)
                for job_id, (process, _, _, ticks) in active.items():
                    identity = proc_identity(process.pid)
                    if identity is None:
                        continue  # Direct child poll handles the exit next pass; slot stays occupied.
                    require(identity["start_ticks"] == ticks, "New child PID reused")
                    job = next(j for j in jobs if j["id"] == job_id)
                    check_worker(identity, {"pid": process.pid, "pgid": process.pid, "start_ticks": ticks, "job": job}, job, target, MODULE)
                gpu = gpu_snapshot(target["gpu_uuid"])
                slots = available_slots(state, [x[0].pid for x in active.values()], gpu["compute_pids"])
                for _ in range(min(slots, len(pending))):
                    job = pending.pop(0)
                    with (logs / f"{job['id']}.stdout.log").open("xb") as out, (logs / f"{job['id']}.stderr.log").open("xb") as err:
                        process = subprocess.Popen([target["python"], "-m", MODULE, "--contract", str(contract_path),
                            "--host", host, "--mode", "fit", "--job-id", job["id"]], cwd=source, env={**env, "PYTHONHASHSEED": str(job["seed"])},
                            stdout=out, stderr=err, start_new_session=True)
                    identity = proc_identity(process.pid)
                    ticks = identity["start_ticks"] if identity else None
                    active[job["id"]] = (process, time.monotonic(), now(), ticks)
            except Exception:
                failure = {"at": now(), "reason": "shared_queue_admission_failed", "traceback": traceback.format_exc()}
        if failure is not None and not (root / "dispatch_failure.json").exists():
            write(root / "dispatch_failure.json", {**failure, "children_left_running":
                {key: value[0].pid for key, value in active.items()}}, exclusive=True)
        write(root / "status.json", {**start, "status": "draining_after_failure" if failure else "training" if active else "queued",
            "observed_at": now(), "children": {key: value[0].pid for key, value in active.items()},
            "pending_job_ids": [j["id"] for j in pending], "actual_worker_exits": exits,
            "parent_occupancy": state, "gpu": gpu, "failure": failure})
        if active or (pending and failure is None):
            time.sleep(5)
    receipts = {}
    for job in jobs:
        path = root / "run" / job["id"] / "fit_receipt.json"
        if path.exists():
            receipt = read(path)
            require(receipt["job"] == job and receipt["contract_sha256"] == sha_json(contract), "Fit identity mismatch")
            require(receipt.get("status") == "complete" and receipt.get("files")
                    and receipt.get("held_out_test_evaluated") is False, "Incomplete receipt")
            for name, digest in receipt["files"].items():
                require(sha_file(local_path(root, name)) == digest, "Terminal artifact SHA mismatch")
            receipts[job["id"]] = {"path": str(path.relative_to(root)), "sha256": sha_file(path)}
    terminal_gpu = gpu_snapshot(target["gpu_uuid"])
    own_pids = [record["pid"] for record in exits.values()]
    owned_absent = all(proc_identity(pid) is None for pid in own_pids) and not (set(own_pids) & set(terminal_gpu["compute_pids"]))
    complete = owned_absent and not failure and len(receipts) == len(jobs) == len(exits) and all(
        v["actual_os_returncode"] == 0 and not v["timeout"] for v in exits.values())
    final = {"status": "complete" if complete else "failed", "host": host, "completed_at": now(),
        "contract_sha256": sha_json(contract), "expected_job_count": len(jobs), "completed_fit_count": len(receipts),
        "actual_worker_exits": exits, "fit_receipts": receipts, "pending_job_ids": [j["id"] for j in pending],
        "held_out_test_evaluated": False, "historical_baseline_inferred": False,
        "failure": failure, "owned_worker_pids_absent": owned_absent, "gpu_after": terminal_gpu,
        "dispatcher_own_exit_directly_observed": False}
    write(root / "final_receipt.json", final, exclusive=True)
    write(root / "status.json", final)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("admit", "dispatch", "fit"), required=True)
    parser.add_argument("--job-id")
    args = parser.parse_args(argv)
    require(args.contract.is_absolute(), "Contract path must be absolute")
    contract = read(args.contract)
    validate_contract(contract, args.host)
    require((args.mode == "fit") == (args.job_id is not None), "Only fit requires job ID")
    if args.mode == "admit":
        admit(contract, args.host)
        return 0
    if args.mode == "fit":
        fit(contract, args.host, args.job_id)
        return 0
    return dispatch(contract, args.host, args.contract)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
