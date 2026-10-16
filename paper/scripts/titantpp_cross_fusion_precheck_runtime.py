"""Isolated six-fit Validation pre-check; no baseline or Test execution.

The dispatcher uses only the standard library. Each new fit runs in a fresh
process group, and the dispatcher directly records that child's OS return code.
Only those child groups may be terminated when their fixed time limit expires.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

SCHEMA = "titantpp_cross_fusion_precheck6_v1"
MODULE = "paper.scripts.titantpp_cross_fusion_precheck_runtime"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x") as handle:
            handle.write(content)
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(content)
        os.replace(temporary, path)


def local_path(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Expected relative source/input path: {name}")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Path escapes isolated source: {name}")
    return path


def validate_contract(contract, host):
    require(contract.get("schema") == SCHEMA, "Wrong pre-check contract schema")
    require(contract.get("evaluation_scope") == "validation_only"
            and contract.get("held_out_test_evaluated") is False,
            "Only explicitly Validation-only contracts are admitted")
    require(set(contract["hosts"]) == {"5080", "5090"} and host in contract["hosts"],
            "Expected the two approved hosts")
    jobs = contract["jobs"]
    require(len(jobs) == 6 and len({j["id"] for j in jobs}) == 6, "Expected six unique jobs")
    require(all(j["seed"] == 42 and j["host"] in contract["hosts"] for j in jobs),
            "Only the approved seed42 fits are admitted")
    require(all(j["id"] and Path(j["id"]).name == j["id"] and j["id"] not in {".", ".."}
                for j in jobs), "Invalid job ID")
    datasets = {d["dataset_id"] for d in contract["datasets"]}
    arms = {j["arm"] for j in jobs}
    require(len(contract["datasets"]) == 3 and len(datasets) == 3 and len(arms) == 2,
            "Expected three datasets and two new arms")
    require({(j["dataset"], j["arm"]) for j in jobs}
            == {(d, a) for d in datasets for a in arms}, "Incomplete dataset/arm pairing")
    require(all(sum(j["host"] == h for j in jobs) == 3 for h in contract["hosts"]),
            "Exactly three fits must be assigned to each host")
    training = contract["training"]
    require(all(training.get(k) == v for k, v in {
        "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
        "batch_size": 128, "monitor": "validation_raw_quantity_rmse"}.items()),
        "The common training budget or selector changed")
    require(contract["limits"]["fit_wall_seconds"] == 129600, "Expected fixed 36-hour fit cap")
    require(contract["source"]["files"] and contract["input_files"], "Source/input inventory is empty")
    require(sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"],
            "Source inventory digest mismatch")
    target = contract["hosts"][host]
    for key in ("root", "source_root", "python"):
        require(Path(target[key]).is_absolute(), f"Host {key} must be absolute")
    return [j for j in jobs if j["host"] == host]


def verify_files(contract, source):
    for inventory in (contract["source"]["files"], contract["input_files"]):
        for name, digest in inventory.items():
            require(sha_file(local_path(source, name)) == digest, f"Source/input SHA mismatch: {name}")


def gpu_snapshot(expected_uuid):
    completed = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,memory.total",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=20)
    rows = [[x.strip() for x in line.split(",")] for line in completed.stdout.splitlines() if line.strip()]
    require(len(rows) == 1 and rows[0][0] == expected_uuid, "GPU UUID mismatch or multiple visible GPUs")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, capture_output=True, check=True, timeout=20)
    pids = []
    for line in processes.stdout.splitlines():
        if line.strip():
            uuid, pid = [s.strip() for s in line.split(",")]
            require(uuid == expected_uuid, "Unexpected GPU in compute inventory")
            pids.append(int(pid))
    return {"observed_at": now(), "uuid": rows[0][0], "name": rows[0][1],
            "memory_total_mb": int(rows[0][2]), "compute_pids": pids}


def runtime_check(target):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    configure_runtime("cuda:0", threads=4)
    measured = runtime_identity("cuda:0")
    for key, expected in target["runtime_expected"].items():
        require(measured[key] == expected, f"Native runtime mismatch: {key}")
    require(measured["gpu"]["uuid"] == target["gpu_uuid"], "Runtime GPU UUID mismatch")
    return measured


def qualify(contract, host):
    validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Qualification cwd must be isolated source")
    require(not (root / "qualification.json").exists() and not (root / "start.json").exists(),
            "Qualification/start already exists; no automatic repeat")
    verify_files(contract, source)
    gpu = gpu_snapshot(target["gpu_uuid"])
    require(not gpu["compute_pids"], "GPU has an existing compute workload")
    runtime = runtime_check(target)
    from paper.scripts import run_titantpp_cross_fusion as runner
    runner.install_hooks()
    require({j["arm"] for j in contract["jobs"]} == set(runner.ARMS), "Only the two new arms are allowed")
    length = max(d["loader"]["max_seq_len"] for d in contract["datasets"])
    synthetic = runner.synthetic_check(device="cuda:0", batch_size=128, length=length)
    peaks = synthetic["cuda_memory_peaks"]
    require(set(peaks) == set(runner.ARMS), "Both full-batch candidate memory measurements are required")
    maximum_peak = max(v["max_memory_reserved_bytes"] for v in peaks.values())
    total_memory = runtime["gpu"]["total_memory_bytes"]
    require(0 < 3 * maximum_peak < 0.85 * total_memory,
            "Three simultaneous full-batch fits exceed the conservative memory admission budget")
    admission = {"maximum_workers": 3, "batch_size": 128, "sequence_length": length,
        "maximum_single_worker_reserved_bytes": maximum_peak,
        "three_worker_reserved_estimate_bytes": 3 * maximum_peak,
        "gpu_total_bytes": total_memory, "fraction_limit": 0.85, "passed": True,
        "limitation": "Conservative estimate from sequential full-shape synthetic updates; not measured three-process peak."}
    receipt = {"status": "passed", "host": host, "contract_sha256": sha_json(contract),
        "verified_at": now(), "gpu_before": gpu, "runtime": runtime, "synthetic": synthetic,
        "parallel_memory_admission": admission,
        "source_file_count": len(contract["source"]["files"]),
        "input_file_count": len(contract["input_files"]),
        "held_out_test_evaluated": False, "historical_baseline_inferred": False}
    write(root / "qualification.json", receipt, exclusive=True)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
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
        "started_at": now(), "job": job, "contract_sha256": sha_json(contract)}
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
        interface_meta=interface, backbone=job["arm"], quantity_variant=runner.VARIANT, seed=42)
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


def terminate_owned_child(process):
    """Only a child launched below with start_new_session=True is accepted."""
    if process.poll() is not None:
        return
    require(os.getpgid(process.pid) == process.pid, "Child process group identity changed")
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def dispatch(contract, host, contract_path):
    jobs = validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Dispatcher cwd must be isolated source")
    qualification = read(root / "qualification.json")
    require(qualification["status"] == "passed" and qualification["contract_sha256"] == sha_json(contract),
            "Qualification must pass for this exact contract")
    require(not (root / "start.json").exists() and not (root / "run").exists(),
            "Existing start or run directory: no restart/retry")
    gpu = gpu_snapshot(target["gpu_uuid"])
    require(not gpu["compute_pids"], "GPU occupied between qualification and dispatch")
    start = {"status": "starting", "host": host, "started_at": now(), "pid": os.getpid(),
        "contract_sha256": sha_json(contract), "qualification_sha256": sha_file(root / "qualification.json"),
        "gpu_before": gpu, "jobs": jobs, "maximum_concurrent_fits": 3}
    write(root / "start.json", start, exclusive=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=False)
    active, exits = {}, {}
    env = {**os.environ, **{k: str(v) for k, v in target.get("environment", {}).items()},
           "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    try:
        for job in jobs:
            stdout = (logs / f"{job['id']}.stdout.log").open("xb")
            stderr = (logs / f"{job['id']}.stderr.log").open("xb")
            try:
                process = subprocess.Popen([target["python"], "-m", MODULE, "--contract", str(contract_path),
                    "--host", host, "--mode", "fit", "--job-id", job["id"]], cwd=source, env=env,
                    stdout=stdout, stderr=stderr, start_new_session=True)
            finally:
                stdout.close()
                stderr.close()
            active[job["id"]] = (process, time.monotonic(), now())
            write(root / "status.json", {**start, "status": "training",
                "observed_at": now(), "children": {key: value[0].pid for key, value in active.items()},
                "actual_worker_exits": exits})
        while active:
            for job_id, (process, started, started_at) in list(active.items()):
                timed_out = process.poll() is None and time.monotonic() - started > contract["limits"]["fit_wall_seconds"]
                if timed_out:
                    terminate_owned_child(process)
                code = process.poll()
                if code is None:
                    continue
                exit_record = {"job_id": job_id, "pid": process.pid, "actual_os_returncode": code,
                    "timeout": timed_out, "started_at": started_at, "observed_exit_at": now()}
                write(root / "run" / job_id / "worker_exit.json", exit_record, exclusive=True)
                exits[job_id] = exit_record
                del active[job_id]
                write(root / "status.json", {**start, "status": "training" if active else "finished",
                    "observed_at": now(), "children": {key: value[0].pid for key, value in active.items()},
                    "actual_worker_exits": exits})
            if active:
                time.sleep(1)
    except BaseException:
        write(root / "dispatch_failure.json", {"at": now(), "traceback": traceback.format_exc(),
            "children_left_running": {key: value[0].pid for key, value in active.items()}}, exclusive=True)
        raise
    receipts = {}
    for job in jobs:
        path = root / "run" / job["id"] / "fit_receipt.json"
        if path.exists():
            receipt = read(path)
            require(receipt["job"] == job and receipt["contract_sha256"] == sha_json(contract), "Fit identity mismatch")
            require(receipt.get("status") == "complete" and receipt.get("files")
                    and receipt.get("held_out_test_evaluated") is False, "Incomplete fit receipt")
            for name, digest in receipt["files"].items():
                require(sha_file(local_path(root, name)) == digest, "Terminal artifact SHA mismatch: " + name)
            receipts[job["id"]] = {"path": str(path.relative_to(root)), "sha256": sha_file(path)}
    complete = len(receipts) == len(jobs) and all(v["actual_os_returncode"] == 0 and not v["timeout"] for v in exits.values())
    final = {"status": "complete" if complete else "failed", "host": host,
        "completed_at": now(), "contract_sha256": sha_json(contract), "expected_job_count": len(jobs),
        "completed_fit_count": len(receipts), "actual_worker_exits": exits, "fit_receipts": receipts,
        "held_out_test_evaluated": False, "historical_baseline_inferred": False,
        "dispatcher_own_exit_directly_observed": False}
    write(root / "final_receipt.json", final, exclusive=True)
    print(json.dumps(final, ensure_ascii=False), flush=True)
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("qualify", "dispatch", "fit"), required=True)
    parser.add_argument("--job-id")
    args = parser.parse_args(argv)
    require(args.contract.is_absolute(), "Contract path must be absolute")
    contract = read(args.contract)
    validate_contract(contract, args.host)
    require((args.mode == "fit") == (args.job_id is not None), "Only fit requires --job-id")
    if args.mode == "qualify":
        qualify(contract, args.host)
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
