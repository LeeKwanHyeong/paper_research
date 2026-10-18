"""Bounded Validation-only literal raw-MSE/raw-MAE MLP16 queue; paused campaigns stay stopped.

Each host owns at most three fresh child processes. Existing experiments are
never resumed or modified; only this dispatcher's timed-out child groups may
be terminated. The dispatcher itself stays standard-library-only.
"""
from __future__ import annotations

import argparse
import json
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

SCHEMA = "titantpp_quantity_loss_mlp18_v1"
MODULE = "paper.scripts.titantpp_quantity_loss_mlp18_runtime"
WORKER_MODULE = "paper.scripts.run_titantpp_quantity_loss_mlp18"
ARMS = ("titantpp_history_mlp_width16",)
TIME_ALPHAS = (0.0,)
LOSS_MODES = ("raw_mse", "raw_mae")
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
    require(type(alpha) in (int, float) and alpha == 0., "Only quantity-only alpha0 is admitted")
    return "0"


def validate_contract(contract, host):
    require(contract.get("schema") == SCHEMA, "Wrong quantity-loss contract")
    require(set(contract["hosts"]) == {"5080", "5090"} and host in contract["hosts"], "Host mismatch")
    require({d["dataset_id"] for d in contract["datasets"]} == set(DATASETS) and len(contract["datasets"]) == 3, "Dataset matrix changed")
    jobs=contract["jobs"]
    require(len(jobs) == len({j["id"] for j in jobs}) == 18, "Expected18 unique fits")
    require({(j["dataset"],j["arm"],j["seed"],j["loss_mode"]) for j in jobs}
        == {(d,ARMS[0],s,m) for d in DATASETS for s in (42,52,62) for m in LOSS_MODES}, "Fit matrix mismatch")
    for j in jobs:
        alpha_label(j["time_alpha"])
        require(type(j["seed"]) is int, "Invalid seed type")
        expected="5080" if j["dataset"]=="yellow_trip_hourly" or (j["dataset"]=="intermittent_frozen_5000" and j["seed"]==42) else "5090"
        require(j["host"]==expected and j["id"]==f"{j['dataset']}__{j['arm']}__seed{j['seed']}__loss{j['loss_mode']}", "Job identity/host mismatch")
    require(all(contract["training"].get(k)==v for k,v in {"maximum_epochs":300,"minimum_epochs":40,"patience":40,"batch_size":128}.items()), "Training budget changed")
    require(contract["training"].get("monitor","validation_raw_quantity_rmse")=="validation_raw_quantity_rmse", "Selector changed")
    require(contract["limits"]=={"fit_wall_seconds":129600,"queue_wall_seconds":604800,"evaluation_wall_seconds":7200}, "Relative limits changed")
    require(contract["evaluation_scope"]=="validation_only" and contract["held_out_test_authorized"] is False, "Validation-only scope changed")
    require(contract["source"]["files"] and sha_json(contract["source"]["files"])==contract["source"]["files_sha256"], "Source inventory mismatch")
    expected_inputs={f"inputs/data/{ds}/{name}" for ds in DATASETS for name in ("train_validation.parquet","split_manifest.json")}
    require(set(contract["input_files"])==expected_inputs, "Only TrainVal/manifest6 may be packaged")
    a=contract["architecture"]
    alpha_label(a["time_alpha"])
    require(a["quantity_loss_modes"]==list(LOSS_MODES) and a["quantity_loss_normalization"]=="none" and a["time_alpha"]==0.
        and a["model_architecture_changed"] is False and a["checkpoint_warmstart"] is False, "Literal raw loss scope changed")
    baseline=contract["baseline_validation_references"]
    require(len(baseline)==9 and {(x["dataset"],x["seed"]) for x in baseline}=={(d,s) for d in DATASETS for s in (42,52,62)}
        and all(x["time_alpha"]==0. and x["loss_mode"]=="log_mse" and x["time_head_trained"] is False
            and x["reevaluated"] is False and x["retrained"] is False and x["sha256"] for x in baseline), "Stored alpha0/logMSE binding changed")
    target=contract["hosts"][host]
    require(all(Path(target[k]).is_absolute() for k in ("root","source_root","python"))
        and Path(target["source_root"])==Path(target["root"])/"source", "Host path mismatch")
    require(type(target["absolute_host_deadline_unix"]) in (int,float) and target["absolute_host_deadline_unix"]>0, "Missing absolute host deadline")
    paused=target["paused_processes"]
    require(len(paused)==4 and len({p["pid"] for p in paused})==4 and all(p["state"] in ("T","t") for p in paused), "Expected paused dispatcher plus three workers")
    parent=target["completed_predecessor"]
    require(Path(parent["root"]).is_absolute() and parent["root"]!=target["root"]
        and parent["contract_raw_sha256"]==contract["parent_time_zero9"]["raw_sha256"]
        and parent["contract_canonical_sha256"]==contract["parent_time_zero9"]["canonical_sha256"]
        and parent["source_files_sha256"]==contract["parent_time_zero9"]["source_files_sha256"]
        and parent["dispatcher_identity"]["pid"]>0 and parent["final_receipt_sha256"] and parent["dispatcher_exit_sha256"], "Completed predecessor binding invalid")
    return [j for j in jobs if j["host"]==host]


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


def qualify_three_concurrent(contract, host):
    """Exercise three real synthetic CUDA processes for each literal loss condition."""
    target=contract["hosts"][host];root=Path(target["root"]);source=Path(target["source_root"])
    results={};all_pids=[]
    for mode in LOSS_MODES:
        children=[]
        try:
            for index in range(3):
                prefix=root/f"admission_{mode}_{index}"
                command=[target["python"],"-u","-c",
                    "import json,sys; from paper.scripts.run_titantpp_quantity_loss_mlp18 import synthetic_condition; "
                    "c=json.load(open(sys.argv[1])); r=synthetic_condition(c,sys.argv[2],sys.argv[3]); print(json.dumps(r,allow_nan=False))",
                    str(root/"execution_contract.json"),host,mode]
                with prefix.with_suffix(".stdout").open("xb") as out,prefix.with_suffix(".stderr").open("xb") as err:
                    process=subprocess.Popen(command,cwd=source,env={**os.environ,**target["environment"],"PYTHONDONTWRITEBYTECODE":"1"},
                        stdout=out,stderr=err,start_new_session=True)
                children.append((process,prefix,command));all_pids.append(process.pid)
            rows=[]
            started=time.monotonic()
            for process,prefix,command in children:
                limit=min(contract["limits"]["evaluation_wall_seconds"]-(time.monotonic()-started),target["absolute_host_deadline_unix"]-time.time())
                timed_out=False
                try: code=process.wait(timeout=max(.001,limit))
                except subprocess.TimeoutExpired:timed_out=True;terminate_owned_child(process);code=process.returncode
                record={"pid":process.pid,"command":command,"actual_os_returncode":code,"timeout":timed_out,
                    "stdout_sha256":sha_file(prefix.with_suffix(".stdout")),"stderr_sha256":sha_file(prefix.with_suffix(".stderr")),
                    "stdout_bytes":prefix.with_suffix(".stdout").stat().st_size,"stderr_bytes":prefix.with_suffix(".stderr").stat().st_size,
                    "observed_exit_at":now()}
                write(prefix.with_suffix(".json"),record,exclusive=True)
                require(code==0 and not timed_out,"Three-worker native synthetic failed; original output preserved")
                result=json.loads(prefix.with_suffix(".stdout").read_text())
                require(result.get("status")=="passed" and result.get("real_data_loaded") is False,"Native synthetic condition failed")
                require(result.get("loss_mode")==mode,"Native synthetic loss identity differs")
                rows.append({**record,"result":result})
            results[mode]=rows
        finally:
            for process,prefix,command in children:
                cleaned=process.poll() is None
                if cleaned:terminate_owned_child(process)
                if not prefix.with_suffix(".json").exists():
                    write(prefix.with_suffix(".json"),{"pid":process.pid,"command":command,
                        "actual_os_returncode":process.returncode,"timeout":False,"owned_synthetic_cleanup_after_failure":cleaned,
                        "stdout_bytes":prefix.with_suffix(".stdout").stat().st_size,"stderr_bytes":prefix.with_suffix(".stderr").stat().st_size,
                        "stdout_sha256":sha_file(prefix.with_suffix(".stdout")),"stderr_sha256":sha_file(prefix.with_suffix(".stderr")),
                        "observed_exit_at":now()},exclusive=True)
    require(all(proc_identity(pid) is None for pid in all_pids),"Native synthetic process remains")
    gpu,paused=gpu_state(contract,host)
    require(not set(all_pids).intersection(gpu["compute_pids"]),"Native synthetic GPU owner remains")
    return {"status":"passed","maximum_simultaneous_workers":3,"loss_modes":results,
        "native_worker_exit0_count":6,"owned_worker_pids_absent":True,"gpu_after":gpu,"paused_after":paused}


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
    predecessor_gate(contract,host,require_complete=True)
    gpu, paused = gpu_state(contract, host)
    free_before = free_gpu_bytes(target["gpu_uuid"])
    runtime = runtime_check(target)
    from paper.scripts.run_titantpp_quantity_loss_mlp18 import synthetic_check
    synthetic = synthetic_check(contract, host)
    require(synthetic["status"] == "passed" and synthetic.get("real_data_loaded") is False,
            "Native synthetic qualification failed")
    peaks = synthetic["cuda_memory_peaks"]
    require(set(peaks) == set(ARMS), "MLP16 native memory qualification missing")
    peak = max(p["max_memory_reserved_bytes"] for p in peaks.values())
    require(all(type(p["max_memory_reserved_bytes"]) is int and p["max_memory_reserved_bytes"] > 0
                for p in peaks.values()), "Invalid native memory measurement")
    require(3 * peak < .85 * free_before, "Three-worker free-memory admission failed")
    concurrent = qualify_three_concurrent(contract,host)
    paused_processes(contract,host)
    result = {"status": "passed", "host": host, "verified_at": now(), "contract_sha256": sha_json(contract),
        "gpu_before": gpu, "paused_processes": paused, "runtime": runtime, "synthetic": synthetic,
        "three_worker_native_admission":concurrent,
        "parallel_memory_admission": {"maximum_workers": 3, "maximum_reserved_bytes_per_worker": peak,
            "three_worker_reserved_estimate_bytes": 3 * peak, "free_memory_before_qualification_bytes": free_before, "fraction_limit": .85, "passed": True,
            "limitation": "Sequential full-shape peaks provide the memory estimate; separate three-process checks validate both losses without claiming an aggregate peak."}}
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


def verify_time_head_proof(proof):
    require(isinstance(proof,dict) and proof.get("schema")=="time_head_quantity_only_verification_v1", "Missing actual time-head proof")
    hashes=[proof.get(k) for k in ("initial_time_head_sha256","final_time_head_sha256","checkpoint_time_head_sha256")]
    require(all(isinstance(x,str) and len(x)==64 for x in hashes) and len(set(hashes))==1, "Time head changed")
    require(all(proof.get(k) is True for k in ("all_time_head_gradients_none","optimizer_state_absent","time_head_parameters_unchanged","time_nll_is_diagnostic"))
        and type(proof.get("optimizer_steps_checked")) is int and proof["optimizer_steps_checked"]>0, "Incomplete time-head no-training evidence")
    return proof


def verify_receipt(contract, host, job, stage):
    root = Path(contract["hosts"][host]["root"])
    path = root / "run" / job["id"] / f"{stage}_receipt.json"
    receipt = read(path)
    require(stage in STAGES, "Test stage forbidden")
    require(receipt.get("status") == "complete" and receipt.get("job_id") == job["id"]
            and receipt.get("stage") == stage and receipt.get("contract_sha256") == sha_json(contract)
            and all(receipt.get(k) == job[k] for k in ("dataset","arm","seed","time_alpha","loss_mode"))
            and receipt.get("held_out_test_evaluated") is False,
            "Worker receipt identity/completion mismatch")
    require(receipt.get("schema")=="titantpp_quantity_loss_mlp18_stage_v1" and receipt.get("quantity_only") is True
            and receipt.get("time_head_trained") is False and receipt.get("quantity_loss_normalization")=="none", "Quantity-only literal loss identity mismatch")
    verify_time_head_proof(receipt.get("time_head_no_training_verification"))
    files = receipt.get("files")
    require(isinstance(files, dict) and files, "Worker artifact inventory is empty")
    job_root = (root / "run" / job["id"]).resolve()
    for name, digest in files.items():
        artifact = local_path(root, name)
        require(artifact.is_relative_to(job_root) and artifact != path.resolve(), "Artifact outside assigned job or self-reference")
        require(sha_file(artifact) == digest, "Worker artifact SHA mismatch: " + name)
    selected=Path(receipt["selected_checkpoint"])
    require(selected.is_absolute() and selected.resolve().is_relative_to(job_root), "Selected checkpoint outside assigned job")
    relative=str(selected.relative_to(root))
    require(files.get(relative)==receipt["checkpoint_sha256"] and sha_file(selected)==receipt["checkpoint_sha256"], "Selected binary missing from inventory or changed")
    require(isinstance(receipt.get("state_sha256"),str) and len(receipt["state_sha256"])==64, "Missing selected state binding")
    if stage == "validation":
        train=read(path.with_name("train_receipt.json"))
        require(all(train.get(k)==receipt.get(k) for k in ("selected_checkpoint","selected_epoch","checkpoint_sha256","state_sha256","target_identity","loss_mode")), "Validation replay selection differs from training")
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
    value = {"schema": "titantpp_quantity_loss_mlp18_artifacts_v1", "files": files,
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
    predecessor_gate(contract,host,require_complete=True)
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
    if not (root/"start.json").exists():
        result={"host":host,"observed_at":now(),"contract_sha256":sha_json(contract),"jobs":[],
            "gpu":gpu_snapshot(target["gpu_uuid"]),"paused_processes":paused_processes(contract,host),
            "gpu_ownership_verified_for_training":False}
        for name in ("launch_attempt.json","qualification_attempt.json","qualification.json","qualification_exit.json","dispatcher_exit.json"):
            if (root/name).exists():result[name]=read(root/name)
        if "dispatcher_exit.json" in result: result["status"]="failed" if result["dispatcher_exit.json"]["actual_os_returncode"]!=0 else "launch_exited"
        elif "qualification_attempt.json" in result:result["status"]="qualifying"
        elif "launch_attempt.json" in result:result["status"]="launching"
        else:result["status"]="not_started"
        return result
    result = {"host": host, "observed_at": now(), "contract_sha256": sha_json(contract),
              "paused_processes": paused_processes(contract,host), "jobs": []}
    for name in ("launch_attempt.json", "qualification.json", "start.json", "status.json",
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


def predecessor_gate(contract,host,*,require_complete=False):
    """Read the sealed parent campaign; never signal or change it."""
    from paper.scripts import titantpp_time_zero_mlp9_runtime as parent
    h=contract["hosts"][host]["completed_predecessor"]
    root=Path(h["root"]);cp=root/"execution_contract.json"
    require(sha_file(cp)==h["contract_raw_sha256"],"Parent contract bytes changed")
    c=read(cp);require(parent.sha_json(c)==h["contract_canonical_sha256"],"Parent canonical contract changed")
    jobs=parent.validate_contract(c,host);ids={j["id"] for j in jobs}
    require(c["source"]["files_sha256"]==h["source_files_sha256"],"Parent source binding changed")
    require(c["hosts"][host]["absolute_host_deadline_unix"]==contract["hosts"][host]["absolute_host_deadline_unix"],"Parent deadline changed")
    paused_processes(contract,host)
    if (root/"dispatch_failure.json").exists():raise ValueError("Parent campaign failed; automatic continuation forbidden")
    final_path=root/"final_receipt.json";exit_path=root/"dispatcher_exit.json"
    require(final_path.exists() and exit_path.exists(), "Completed predecessor terminal files missing; no new GPU work")
    require(sha_file(final_path)==h["final_receipt_sha256"] and sha_file(exit_path)==h["dispatcher_exit_sha256"], "Predecessor terminal SHA changed")
    final,exit=read(final_path),read(exit_path)
    require(final["status"]=="complete" and final["host"]==host and final["contract_sha256"]==parent.sha_json(c)
        and final["completed_fit_count"]==final["validation_replay_count"]==len(jobs) and final["test_count"]==0
        and final["held_out_test_evaluated"] is False and final["owned_worker_pids_absent"] is True,"Parent completion evidence invalid")
    require(exit["actual_os_returncode"]==0 and exit["pid"]==h["dispatcher_identity"]["pid"],"Parent actual dispatcher exit missing")
    pids=[exit["pid"]]
    require(set(final["stages"])==set(STAGES),"Parent stage inventory changed")
    for stage in STAGES:
        result=final["stages"][stage]
        require(result["status"]=="complete" and set(result["exits"])==set(result["receipts"])==ids,"Parent stage incomplete")
        for job in jobs:
            e=result["exits"][job["id"]];binding=result["receipts"][job["id"]]
            require(e["actual_os_returncode"]==0 and e["timeout"] is False,"Parent worker failed")
            pids.append(e["pid"])
            receipt_path=local_path(root,binding["path"])
            require(sha_file(receipt_path)==binding["sha256"],"Parent stage receipt changed")
            parent.verify_receipt(c,host,job,stage)
            r=read(receipt_path)
            selected=Path(r["selected_checkpoint"])
            require(selected.is_absolute() and selected.resolve().is_relative_to(root.resolve()),"Parent selection outside root")
            require(sha_file(selected)==r["checkpoint_sha256"],"Parent selected binary changed")
    manifest=final["artifact_manifest"];manifest_path=local_path(root,manifest["path"])
    require(sha_file(manifest_path)==manifest["sha256"],"Parent artifact manifest changed")
    inventory=read(manifest_path)["files"]
    require(inventory and len(inventory)==manifest["file_count"],"Parent artifact inventory incomplete")
    for name,digest in inventory.items():require(sha_file(local_path(root,name))==digest,"Parent artifact changed")
    require(all(proc_identity(pid) is None for pid in pids),"Parent owned process still exists")
    gpu,paused=gpu_state(contract,host)
    require(not set(pids).intersection(gpu["compute_pids"]),"Parent GPU worker remains")
    return {"status":"complete","parent_root":str(root),"parent_final_receipt_sha256":sha_file(final_path),
        "parent_dispatcher_exit_sha256":sha_file(exit_path),"parent_artifact_manifest_sha256":sha_file(manifest_path),
        "parent_owned_pids_absent":True,"parent_gpu_pids_absent":True,"paused_processes":paused,"gpu":gpu}


def qualify_child(contract,host,contract_path,remaining):
    h=contract["hosts"][host];root=Path(h["root"])
    command=[h["python"],"-u","-m",MODULE,"--contract",str(contract_path),"--host",host,"--mode","qualify"]
    with (root/"qualification.stdout.log").open("xb") as out,(root/"qualification.stderr.log").open("xb") as err:
        p=subprocess.Popen(command,cwd=h["source_root"],env={**os.environ,**h["environment"]},stdout=out,stderr=err,start_new_session=True)
        timeout=False
        try:code=p.wait(timeout=min(remaining,contract["limits"]["evaluation_wall_seconds"]))
        except subprocess.TimeoutExpired:timeout=True;terminate_owned_child(p);code=p.returncode
    write(root/"qualification_exit.json",{"pid":p.pid,"command":command,"actual_os_returncode":code,"timeout":timeout,"observed_exit_at":now()},exclusive=True)
    require(code==0 and not timeout,"New qualification child failed or timed out")


def launch_dispatch(contract,host,contract_path):
    """A single supervised lifecycle; no polling or resumption of prior experiments."""
    validate_contract(contract,host);h=contract["hosts"][host];root=Path(h["root"])
    require(Path.cwd().resolve()==Path(h["source_root"]).resolve(),"Launch cwd mismatch")
    require(not any((root/n).exists() for n in ("launch_attempt.json","qualification_attempt.json","qualification.json","dispatch_attempt.json","start.json","run","final_receipt.json")),"Existing launch/attempt; no retry")
    verify_files(contract,Path(h["source_root"]))
    write(root/"launch_attempt.json",{"host":host,"started_at":now(),"contract_sha256":sha_json(contract)},exclusive=True)
    started=time.monotonic()
    gate=predecessor_gate(contract,host,require_complete=True)
    write(root/"parent_terminal_gate.json",gate,exclusive=True)
    remaining=min(contract["limits"]["queue_wall_seconds"],h["absolute_host_deadline_unix"]-time.time())
    require(remaining>0,"Inherited deadline expired")
    qualify_child(contract,host,contract_path,remaining)
    return dispatch(contract,host,contract_path,queue_started=started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("qualify", "dispatch", "launch_dispatch", "status"), required=True)
    args = parser.parse_args(argv)
    require(args.contract.is_absolute(), "Contract path must be absolute")
    contract = read(args.contract)
    if args.mode == "dispatch":
        return dispatch(contract, args.host, args.contract)
    if args.mode == "launch_dispatch":
        return launch_dispatch(contract,args.host,args.contract)
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
