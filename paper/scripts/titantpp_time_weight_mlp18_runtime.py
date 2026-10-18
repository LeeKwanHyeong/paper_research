"""Bounded Validation-only time-weight MLP16 queue; never resumes the paused campaign.

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
from paper.scripts.titantpp_cross_fusion_seed_extension_runtime import proc_identity as _base_proc_identity

SCHEMA = "titantpp_time_weight_mlp18_v1"
MODULE = "paper.scripts.titantpp_time_weight_mlp18_runtime"
WORKER_MODULE = "paper.scripts.run_titantpp_time_weight_mlp18"
ARMS = ("titantpp_history_mlp_width16",)
TIME_ALPHAS = (0.1, 10.0)
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
STAGES = ("train", "validation")


def proc_identity(pid, proc_root=Path("/proc")):
    live=_base_proc_identity(pid, proc_root=proc_root)
    if live is None: return None
    try: fields=(Path(proc_root)/str(pid)/"stat").read_text().rpartition(")")[2].split()
    except (FileNotFoundError,ProcessLookupError): return None
    require(int(fields[19])==live["start_ticks"],"PID changed during state read")
    return {**live,"state":fields[0],"ppid":int(fields[1])}


def process_parents():
    parents={}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try: parents[int(p.name)]=int((p/"stat").read_text().rpartition(")")[2].split()[1])
            except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    return parents


def alpha_label(alpha):
    require(type(alpha) in (int, float) and alpha in TIME_ALPHAS, "Invalid time alpha")
    return "0p1" if alpha == .1 else "10"


def validate_contract(contract, host):
    require(contract.get("schema") == SCHEMA, "Wrong time-weight contract")
    require(set(contract["hosts"]) == {"5080", "5090"} and host in contract["hosts"], "Host mismatch")
    datasets = {d["dataset_id"] for d in contract["datasets"]}
    require(datasets == set(DATASETS) and len(contract["datasets"]) == 3, "Dataset matrix changed")
    jobs = contract["jobs"]
    require(len(jobs) == len({j["id"] for j in jobs}) == 18, "Expected18 unique fits")
    require({(j["dataset"], j["arm"], j["seed"], j["time_alpha"]) for j in jobs}
        == {(d, ARMS[0], seed, a) for d in DATASETS for seed in (42,52,62) for a in TIME_ALPHAS}, "Fit matrix mismatch")
    for j in jobs:
        expected_host = "5080" if j["dataset"] == "yellow_trip_hourly" or (j["dataset"] == "intermittent_frozen_5000" and j["seed"] == 42) else "5090"
        require(j["host"] == expected_host and j["id"] == f"{j['dataset']}__{j['arm']}__seed{j['seed']}__alpha{alpha_label(j['time_alpha'])}", "Job identity/host mismatch")
    require(all(contract["training"].get(k) == v for k,v in {"maximum_epochs":300,"minimum_epochs":40,"patience":40,"batch_size":128}.items()), "Training budget changed")
    require(contract["training"].get("monitor", "validation_raw_quantity_rmse") == "validation_raw_quantity_rmse", "Selector changed")
    require(contract["limits"] == {"fit_wall_seconds":129600,"queue_wall_seconds":604800,"evaluation_wall_seconds":7200}, "Relative limits changed")
    require(contract["evaluation_scope"] == "validation_only" and contract["held_out_test_authorized"] is False, "Validation-only scope changed")
    require(contract["source"]["files"] and sha_json(contract["source"]["files"]) == contract["source"]["files_sha256"], "Source inventory mismatch")
    expected_inputs={f"inputs/data/{ds}/{name}" for ds in DATASETS for name in ("train_validation.parquet", "split_manifest.json")}
    require(set(contract["input_files"]) == expected_inputs, "Only TrainVal/manifest6 may be packaged")
    baseline=contract["baseline_validation_references"]
    require(len(baseline)==9 and {(x["dataset"],x["seed"]) for x in baseline} == {(d,s) for d in DATASETS for s in (42,52,62)}
        and all(x["time_alpha"] == 1. and x["reevaluated"] is False and x["retrained"] is False and x["sha256"] for x in baseline), "Stored alpha1 baseline binding changed")
    target=contract["hosts"][host]
    require(all(Path(target[k]).is_absolute() for k in ("root","source_root","python")) and Path(target["source_root"]) == Path(target["root"])/"source", "Host path mismatch")
    require(type(target["absolute_host_deadline_unix"]) in (int,float) and target["absolute_host_deadline_unix"] > 0, "Missing absolute host deadline")
    paused=target["paused_processes"]
    require(len(paused)==4 and len({p["pid"] for p in paused})==4 and all(p["state"] in ("T","t") for p in paused), "Expected paused dispatcher plus three workers")
    return [j for j in jobs if j["host"] == host]


def paused_processes(contract, host):
    expected=contract["hosts"][host]["paused_processes"]
    observed=[]
    for p in expected:
        live=proc_identity(p["pid"])
        require(live is not None and all(live[k]==p[k] for k in ("pid","start_ticks","argv","cwd","pgid"))
            and live["state"] in ("T","t"), "Paused process identity/state changed")
        observed.append(live)
    # Any new descendant is unverified; no signal is sent to this process tree.
    parents=process_parents()
    descendants={expected[0]["pid"]}
    while True:
        more={pid for pid,parent in parents.items() if parent in descendants}
        if more <= descendants: break
        descendants.update(more)
    require(descendants == {p["pid"] for p in expected}, "Unknown descendant of paused campaign")
    return observed


def gpu_state(contract,host,active_pids=()):
    stopped=paused_processes(contract,host)
    gpu=gpu_snapshot(contract["hosts"][host]["gpu_uuid"])
    allowed={p["pid"] for p in stopped[1:]} | set(active_pids)
    require(set(gpu["compute_pids"]) <= allowed, "Unidentified GPU process; no new work")
    return gpu,stopped


def free_gpu_bytes(expected_uuid):
    out=subprocess.run(["nvidia-smi","--query-gpu=uuid,memory.free","--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=20)
    rows=[line.split(",") for line in out.stdout.splitlines() if line.strip()]
    require(len(rows)==1 and rows[0][0].strip()==expected_uuid,"Free-memory GPU mismatch")
    free=int(rows[0][1].strip())*1024*1024
    require(free>0,"No free GPU memory")
    return free


def qualify(contract, host):
    validate_contract(contract, host)
    target = contract["hosts"][host]
    root, source = Path(target["root"]), Path(target["source_root"])
    require(Path.cwd().resolve() == source.resolve(), "Qualification cwd mismatch")
    require(not any((root / p).exists() for p in ("qualification.json", "start.json", "run", "dispatch_attempt.json")),
            "Existing qualification/start/run; no automatic retry")
    verify_files(contract, source)
    for name in contract.get("packaging", {}).get("required_empty_source_directories", []):
        require(local_path(source, name).is_dir(), "Required source root directory missing: " + name)
    write(root / "qualification_attempt.json", {"host": host, "started_at": now(),
          "contract_sha256": sha_json(contract)}, exclusive=True)
    require(time.time() < target["absolute_host_deadline_unix"], "Absolute host deadline expired")
    gpu, paused = gpu_state(contract, host)
    free_before = free_gpu_bytes(target["gpu_uuid"])
    runtime = runtime_check(target)
    from paper.scripts.run_titantpp_time_weight_mlp18 import synthetic_check
    synthetic = synthetic_check(contract, host)
    require(synthetic["status"] == "passed" and synthetic.get("real_data_loaded") is False,
            "Native synthetic qualification failed")
    peaks = synthetic["cuda_memory_peaks"]
    require(set(peaks) == set(ARMS), "MLP16 native memory qualification missing")
    peak = max(p["max_memory_reserved_bytes"] for p in peaks.values())
    require(all(type(p["max_memory_reserved_bytes"]) is int and p["max_memory_reserved_bytes"] > 0
                for p in peaks.values()), "Invalid native memory measurement")
    require(3 * peak < .85 * free_before, "Three-worker free-memory admission failed")
    paused_processes(contract,host)
    result = {"status": "passed", "host": host, "verified_at": now(), "contract_sha256": sha_json(contract),
        "gpu_before": gpu, "paused_processes": paused, "runtime": runtime, "synthetic": synthetic,
        "parallel_memory_admission": {"maximum_workers": 3, "maximum_reserved_bytes_per_worker": peak,
            "three_worker_reserved_estimate_bytes": 3 * peak, "free_memory_before_qualification_bytes": free_before, "fraction_limit": .85, "passed": True,
            "limitation": "Sequential full-shape synthetic peaks; not a measured three-process peak."}}
    require(time.time() < target["absolute_host_deadline_unix"], "Deadline exceeded during qualification")
    write(root / "qualification.json", result, exclusive=True)
    return result


def worker_command(contract, host, contract_path, job, stage):
    require(stage in STAGES, "Unknown worker stage")
    return [contract["hosts"][host]["python"], "-u", "-m", WORKER_MODULE,
            "--contract", str(contract_path), "--host", host, "--job", job["id"], "--stage", stage]


def available_slots(active_pids, gpu_pids, paused_pids=()):
    require(len(active_pids) == len(set(active_pids)) <= 3, "Worker concurrency exceeded")
    require(set(gpu_pids).issubset(set(active_pids) | set(paused_pids)), "Unidentified GPU process; no new jobs may start")
    return 3 - len(active_pids)


def verify_receipt(contract, host, job, stage):
    root = Path(contract["hosts"][host]["root"])
    path = root / "run" / job["id"] / f"{stage}_receipt.json"
    receipt = read(path)
    require(stage in STAGES, "Test stage forbidden")
    require(receipt.get("status") == "complete" and receipt.get("job_id") == job["id"]
            and receipt.get("stage") == stage and receipt.get("contract_sha256") == sha_json(contract)
            and all(receipt.get(k) == job[k] for k in ("dataset","arm","seed","time_alpha"))
            and receipt.get("held_out_test_evaluated") is False,
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
    require(stage in STAGES, "Validation-only stages")
    pending, active, exits, receipts, failure = list(jobs), {}, {}, {}, None
    timeout = contract["limits"]["fit_wall_seconds" if stage == "train" else "evaluation_wall_seconds"]
    env = {**os.environ, **target["environment"], "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    while active or (pending and failure is None):
        for job_id, record in list(active.items()):
            process = record["process"]
            timed_out = process.poll() is None and (
                time.monotonic() - record["started_monotonic"] > timeout
                or time.monotonic() - queue_started > contract["limits"]["queue_wall_seconds"]
                or time.time() >= target["absolute_host_deadline_unix"])
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
                require(time.monotonic() - queue_started <= contract["limits"]["queue_wall_seconds"]
                    and time.time() < target["absolute_host_deadline_unix"], "Queue/absolute deadline exceeded")
                for record in active.values():
                    _check_child(record, source)
                gpu, paused = gpu_state(contract, host, [r["process"].pid for r in active.values()])
                slots = available_slots([r["process"].pid for r in active.values()], gpu["compute_pids"], [p["pid"] for p in paused[1:]])
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
    value = {"schema": "titantpp_time_weight_mlp18_artifacts_v1", "files": files,
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
    gpu_state(contract,host)
    require(time.time() < target["absolute_host_deadline_unix"], "Absolute host deadline expired")
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
    gpu, paused_after = gpu_state(contract,host)
    verify_files(contract,source)
    absent = all(proc_identity(pid) is None for pid in pids) and not set(pids).intersection(gpu["compute_pids"])
    complete = len(stages) == 2 and all(r["status"] == "complete" for r in stages.values()) and absent
    terminal = {"status": "complete" if complete else "failed", "host": host, "completed_at": now(),
        "contract_sha256": sha_json(contract), "expected_job_count": len(jobs),
        "completed_fit_count": len(stages.get("train", {}).get("receipts", {})),
        "validation_replay_count": len(stages.get("validation", {}).get("receipts", {})),
        "test_count": 0, "stages": stages, "paused_after": paused_after, "prior_campaign_resumed": False,
        "owned_worker_pids_absent": absent, "gpu_after": gpu, "held_out_test_evaluated": False,
        "existing_models_retrained": False, "dispatcher_own_exit_directly_observed": False}
    write(root / "status.json", terminal)
    terminal["artifact_manifest"] = artifact_manifest(root)
    write(root / "final_receipt.json", terminal, exclusive=True)
    return 0 if complete else 1


def status(contract, host):
    jobs = validate_contract(contract, host)
    target = contract["hosts"][host]
    root = Path(target["root"])
    result = {"host": host, "observed_at": now(), "contract_sha256": sha_json(contract),
              "paused_processes": paused_processes(contract,host), "jobs": []}
    for name in ("waiting.json", "wait_failure.json", "qualification.json", "start.json", "status.json",
                 "dispatch_failure.json", "final_receipt.json", "dispatcher_exit.json"):
        path = root / name
        if path.exists():
            result[name] = read(path)
    if "start.json" in result and "dispatcher_exit.json" not in result:
        result["dispatcher_current_identity"] = proc_identity(result["start.json"]["pid"])
    active_pids=[]
    for job in jobs:
        entry = {"job": job, "stages": {}}
        for stage in STAGES:
            value = {}
            for kind in ("launch", "exit", "receipt"):
                path = root / "run" / job["id"] / f"{stage}_{kind}.json"
                if path.exists():
                    value[kind] = read(path)
            if "launch" in value and "exit" not in value:
                live = proc_identity(value["launch"]["pid"])
                value["current_identity"] = live
                if live is not None:
                    require(live["start_ticks"] == value["launch"]["start_ticks"]
                        and live["argv"] == value["launch"]["command"] and live["cwd"] == target["source_root"], "Active worker identity changed")
                    active_pids.append(live["pid"])
            if value:
                entry["stages"][stage] = value
        # Train history is read only for progress; no re-evaluation is performed.
        histories = list((root / "run" / job["id"]).glob("runs/*/*/seed_*/history.json"))
        if len(histories) == 1:
            entry["history"] = read(histories[0]).get("history", [])
        result["jobs"].append(entry)
    result["gpu"] = gpu_state(contract,host,active_pids)[0]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("qualify", "dispatch", "status"), required=True)
    args = parser.parse_args(argv)
    require(args.contract.is_absolute(), "Contract path must be absolute")
    contract = read(args.contract)
    if args.mode == "dispatch":
        return dispatch(contract, args.host, args.contract)
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
