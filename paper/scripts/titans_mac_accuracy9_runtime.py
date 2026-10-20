"""One isolated MAC9 queue; the preserved bounded implementation owns only new children."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import signal
import time
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import (
    gpu_snapshot, local_path, now, read, require, runtime_check, sha_file, sha_json, verify_files, write)

_BASE = Path(__file__).with_name("titantpp_quantity_loss_mlp18_runtime.py")
_BASE_SHA = "03a4d8c07e7ee1af2164e2a7b1cfe088a86bc189dc1e35ba014a4e6b2f28e9ec"
require(sha_file(_BASE) == _BASE_SHA, "Preserved bounded queue changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_queue", _BASE)
_impl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_impl)
SCHEMA = "titans_mac_accuracy9_v1"
MODULE = "paper.scripts.titans_mac_accuracy9_runtime"
WORKER_MODULE = "paper.scripts.run_titans_mac_accuracy9"
ARMS = ("titantpp_titans_mac",)
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
SEEDS, STAGES = (42, 52, 62), ("train", "validation", "test")
GPU_UUID = "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"
DEADLINE = 1792888885.815062
proc_identity, terminate_owned_child = _impl.proc_identity, _impl.terminate_owned_child


def paused_processes(c, host):
    require(host == "5090" and c["hosts"][host]["paused_processes"] == [], "Unexpected paused process contract")
    return []


def gpu_state(c, host, active_pids=()):
    paused = paused_processes(c, host)
    gpu = gpu_snapshot(c["hosts"][host]["gpu_uuid"])
    require(set(gpu["compute_pids"]) <= set(active_pids), "Unidentified GPU process; no new work")
    return gpu, paused


_impl.paused_processes, _impl.gpu_state = paused_processes, gpu_state


def validate_contract(c, host):
    require(c.get("schema") == SCHEMA and host == "5090" and set(c["hosts"]) == {"5090"}, "Only isolated5090 mac9 is admitted")
    require(c.get("common_condition") == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0}
        and c.get("data_variant") == "original_uncapped" and c.get("quantity_capping_applied") is False, "Original joint input/loss differs")
    require(c.get("held_out_test_authorized") is True and c.get("execution_authorization") == {
        "real_data_training": True, "real_data_inference": True, "remote_execution": True, "test_evaluation": True}, "MAC9 approval differs")
    require(c["training"] == {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40, "batch_size": 128,
        "monitor": "validation_raw_quantity_rmse"}, "Fixed budget/selector differs")
    require(c["limits"] == {"fit_wall_seconds": 129600, "queue_wall_seconds": 604800, "evaluation_wall_seconds": 7200}, "Fixed limits differ")
    require(c.get("maximum_parallel_workers") == 3 and c.get("historical_models_retrained") is False
        and c.get("checkpoint_warmstart") is False, "Parallel/fresh-fit scope differs")
    jobs = c["jobs"]
    require(len(jobs) == len({j["id"] for j in jobs}) == 9 and {(j["dataset"], j["arm"], j["seed"]) for j in jobs}
        == {(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}, "Exactly MAC seed42/52/62 dataset3 required")
    for j in jobs:
        require(type(j["seed"]) is int and j["host"] == host and j["id"] == f"{j['dataset']}__{j['arm']}__seed{j['seed']}"
            and all(j.get(k) == v for k, v in c["common_condition"].items()), "Job identity/condition differs")
    source = c["source"]
    require(source["files"] and sha_json(source["files"]) == source["files_sha256"], "Source inventory differs")
    expected = {f"inputs/data/{d}/{n}" for d in DATASETS for n in ("train_validation.parquet", "split_manifest.json")}
    expected |= {f"inputs/full/{d}.parquet" for d in DATASETS}
    require(set(c["input_files"]) == expected and {d["dataset_id"] for d in c["datasets"]} == set(DATASETS)
        and len(c["datasets"]) == 3, "Original dataset/input9 matrix differs")
    for d in c["datasets"]:
        require("cap_value" not in d and d["model"]["hidden_dim"] == 64
            and d["optimizer"] == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.}
            and d["full_data_identity"]["data"]["sha256"] == c["input_files"][f"inputs/full/{d['dataset_id']}.parquet"], "Original model/head/data differs")
    h = c["hosts"][host]
    require(h["gpu_uuid"] == GPU_UUID and h["absolute_host_deadline_unix"] == DEADLINE
        and h["runtime_expected"] == {"python": "3.12.13", "torch": "2.11.0+cu130", "cuda": "13.0", "cudnn": 92000,
            "threads": 4, "interop_threads": 1}, "Native5090 GPU/Runtime/deadline differs")
    require(Path(h["root"]).is_absolute() and Path(h["source_root"]) == Path(h["root"]) / "source"
        and Path(h["python"]).is_absolute(), "Isolated paths differ")
    require(h["paused_processes"] == [], "MAC5090 has no historical paused process")
    return jobs


def predecessor_gate(c, host, *, require_complete=False):
    validate_contract(c, host)
    gpu, paused = gpu_state(c, host)
    require(c.get("prior_owned_pids") and all(proc_identity(p) is None for p in c["prior_owned_pids"]), "Previous selected evaluation/Cap owner remains")
    return {"status": "passed", "observed_at": now(), "gpu": gpu, "paused_processes": paused,
        "prior_owned_pids_absent": True, "historical_campaigns_changed": False}


def verify_receipt(c, host, job, stage, *, originals=None):
    require(stage in STAGES, "Unknown stage")
    remote = Path(c["hosts"][host]["root"]); root = remote if originals is None else Path(originals)
    path = root / "run" / job["id"] / f"{stage}_receipt.json"; r = read(path)
    require(r.get("schema") == "titans_mac_accuracy9_stage_v1" and r.get("status") == "complete"
        and r.get("contract_sha256") == sha_json(c) and r.get("job_id") == job["id"] and r.get("stage") == stage
        and all(r.get(k) == job[k] for k in ("dataset", "arm", "seed", "time_input", "loss_mode", "time_alpha")), "MAC9 worker identity differs")
    require(r.get("time_head_trained") is True and r.get("quantity_only") is False
        and r.get("held_out_test_evaluated") is (stage == "test"), "Joint training/evaluation identity differs")
    files = r.get("files"); require(isinstance(files, dict) and files, "Missing stage original inventory")
    for name, digest in files.items():
        p = local_path(root, name)
        require(p.is_relative_to((root / "run" / job["id"]).resolve()) and p != path.resolve()
            and sha_file(p) == digest, "Stage original changed or escaped job")
    selected = Path(r["selected_checkpoint"])
    require(selected.is_absolute() and selected.is_relative_to(remote / "run" / job["id"]), "Selected checkpoint escaped job")
    relative = str(selected.relative_to(remote))
    require(files.get(relative) == r["checkpoint_sha256"] == sha_file(local_path(root, relative))
        and type(r.get("selected_epoch")) is int and 0 < r["selected_epoch"] <= 300, "Selected checkpoint binding differs")
    if stage != "train":
        train = read(path.with_name("train_receipt.json"))
        require(all(r.get(k) == train.get(k) for k in ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256")), "Frozen selection differs")
    if stage == "validation": require(r.get("validation_replay_passed") is True, "Validation replay did not pass")
    return {"path": str(path.relative_to(root)), "sha256": sha_file(path)}


def validation_gate(c, host, exits):
    root = Path(c["hosts"][host]["root"]); jobs = validate_contract(c, host)
    require(set(exits) == {j["id"] for j in jobs} and all(e["actual_os_returncode"] == 0 and not e["timeout"] for e in exits.values()), "All9 Validation children must exit0")
    records = {j["id"]: {"validation_receipt_sha256": verify_receipt(c, host, j, "validation")["sha256"],
        "train_exit_sha256": sha_file(root / "run" / j["id"] / "train_exit.json"),
        "validation_exit_sha256": sha_file(root / "run" / j["id"] / "validation_exit.json")} for j in jobs}
    result = {"status": "passed", "host": host, "contract_sha256": sha_json(c), "created_at": now(), "jobs": records}
    write(root / "validation_gate.json", result, exclusive=True); return result


def cleanup_owned_synthetic(child, command, source, recorded=None, *, grace=10):
    if child.poll() is not None: return
    live = proc_identity(child.pid)
    require(live is not None and live["argv"] == command and live["cwd"] == str(source)
        and live["pgid"] == child.pid and (recorded is None or live["start_ticks"] == recorded["start_ticks"]),
        "Synthetic child ownership is unconfirmed; no signal sent")
    if grace == 10: terminate_owned_child(child)
    else:
        os.killpg(child.pid, signal.SIGTERM)
        try: child.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)


def _synthetic_group(c, host, arms, prefix):
    h = c["hosts"][host]; root = Path(h["root"]); source = Path(h["source_root"]); children, rows, identities = [], [], []
    try:
        for index, arm in enumerate(arms):
            name = root / f"{prefix}_{index}"
            program = ("import json,sys,os,time,pathlib\n"
                "ready=pathlib.Path(sys.argv[4]);go=pathlib.Path(sys.argv[5]);deadline=float(sys.argv[6])\n"
                "ready.open('x').write(json.dumps({'pid':os.getpid()})+chr(10))\n"
                "while not go.exists():\n if time.time()>=deadline:raise TimeoutError('Synthetic admission barrier expired')\n time.sleep(.05)\n"
                "from paper.scripts.run_titans_mac_accuracy9 import synthetic_condition\n"
                "c=json.load(open(sys.argv[1]));print(json.dumps(synthetic_condition(c,sys.argv[2],'log_mse',arm=sys.argv[3]),allow_nan=False))\n")
            command = [h["python"], "-u", "-c", program, str(root / "execution_contract.json"), host, arm,
                str(name.with_suffix(".ready.json")), str(root / (prefix + ".go")), str(min(time.time()+120., h["absolute_host_deadline_unix"]))]
            with name.with_suffix(".stdout").open("xb") as out, name.with_suffix(".stderr").open("xb") as err:
                p = subprocess.Popen(command, cwd=source, env={**os.environ, **h["environment"], "PYTHONDONTWRITEBYTECODE": "1"},
                    stdout=out, stderr=err, start_new_session=True)
            children.append((p, name, command, arm, time.monotonic()))
        barrier_deadline = min(time.time()+120., h["absolute_host_deadline_unix"])
        while not all(name.with_suffix(".ready.json").exists() for _, name, *_ in children):
            require(time.time() < barrier_deadline and all(p.poll() is None for p, *_ in children), "Native synthetic startup/barrier failed")
            time.sleep(.05)
        for p, name, command, arm, _ in children:
            live = proc_identity(p.pid)
            require(p.poll() is None and read(name.with_suffix(".ready.json"))["pid"] == p.pid and live is not None
                and live["argv"] == command and live["cwd"] == str(source) and live["pgid"] == p.pid, "Synthetic actual process identity differs")
            identities.append(live)
        write(root / (prefix + ".overlap.json"), {"observed_at": now(), "simultaneous_live_count": len(children), "identities": identities}, exclusive=True)
        (root / (prefix + ".go")).open("x").close()
        for p, name, command, arm, started in children:
            timeout = False
            recorded = next(i for i in identities if i["pid"] == p.pid)
            try: code = p.wait(timeout=max(.001, min(c["limits"]["evaluation_wall_seconds"]-(time.monotonic()-started), h["absolute_host_deadline_unix"] - time.time())))
            except subprocess.TimeoutExpired: timeout = True; cleanup_owned_synthetic(p,command,source,recorded); code = p.returncode
            record = {"pid": p.pid, "command": command, "arm": arm, "actual_os_returncode": code, "timeout": timeout,
                "start_ticks": next(i["start_ticks"] for i in identities if i["pid"] == p.pid),
                "stdout_sha256": sha_file(name.with_suffix(".stdout")), "stderr_sha256": sha_file(name.with_suffix(".stderr")),
                "stdout_bytes": name.with_suffix(".stdout").stat().st_size, "stderr_bytes": name.with_suffix(".stderr").stat().st_size}
            write(name.with_suffix(".json"), record, exclusive=True)
            require(code == 0 and not timeout, "Native synthetic failed; original output retained")
            value = json.loads(name.with_suffix(".stdout").read_text())
            require(value.get("status") == "passed" and value.get("real_data_loaded") is False and value.get("arm") == arm, "Synthetic arm/data identity differs")
            rows.append({**record, "result": value})
    finally:
        for p, name, command, arm, _ in children:
            if p.poll() is None: cleanup_owned_synthetic(p,command,source,next((i for i in identities if i["pid"] == p.pid),None))
            if not name.with_suffix(".json").exists(): write(name.with_suffix(".json"), {"pid": p.pid, "command": command,
                "arm": arm, "actual_os_returncode": p.returncode, "cleanup_after_failure": True}, exclusive=True)
    require(all(proc_identity(p.pid) is None for p, *_ in children), "Native synthetic owner remains")
    return rows


def qualify(c, host):
    validate_contract(c, host); h = c["hosts"][host]; root = Path(h["root"]); source = Path(h["source_root"])
    require(Path.cwd().resolve() == source.resolve() and not any((root / n).exists() for n in (
        "qualification_attempt.json", "qualification.json", "start.json", "run", "dispatch_attempt.json")), "Qualification cwd/attempt differs")
    verify_files(c, source); write(root / "qualification_attempt.json", {"at": now(), "contract_sha256": sha_json(c)}, exclusive=True)
    admission = predecessor_gate(c, host); free = _impl.free_gpu_bytes(h["gpu_uuid"])
    rows = [row for index, arm in enumerate(ARMS) for row in _synthetic_group(c, host, [arm], f"qualification_arm{index}")]
    observed = rows[0]["result"]["runtime"]
    def peak(row): return row["result"]["cuda_memory_peak"]["max_memory_reserved_bytes"]
    require(all(type(peak(r)) is int and peak(r) > 0 for r in rows), "Native arm memory measurement missing")
    biggest = max(rows, key=peak); limit = 3
    require(3 * peak(biggest) < .85 * free, "Three-worker native memory admission failed; no scientific fit started")
    concurrent = _synthetic_group(c, host, [biggest["arm"]] * 3, "concurrent_admission")
    gpu, paused = gpu_state(c, host)
    result = {"status": "passed", "host": host, "contract_sha256": sha_json(c), "runtime": observed,
        "admission": admission, "arm_qualifications": rows, "concurrent_admission": concurrent, "native_worker_exit0_count": len(ARMS) + limit,
        "maximum_simultaneous_workers": limit, "maximum_memory_arm": biggest["arm"], "free_memory_before_bytes": free,
        "per_arm_memory_limits": {r["arm"]: 3 for r in rows},
        "gpu_after": gpu, "paused_after": paused, "owned_worker_pids_absent": True}
    write(root / "qualification.json", result, exclusive=True); return result


_impl.SCHEMA, _impl.MODULE, _impl.WORKER_MODULE = SCHEMA, MODULE, WORKER_MODULE
_impl.ARMS, _impl.DATASETS, _impl.STAGES = ARMS, DATASETS, STAGES
_impl.validate_contract, _impl.verify_receipt, _impl.validation_gate = validate_contract, verify_receipt, validation_gate
_impl.predecessor_gate = predecessor_gate
worker_command, available_slots, artifact_manifest = _impl.worker_command, _impl.available_slots, _impl.artifact_manifest
_QUEUE_OWNED_CHILDREN = []
_impl._EXTRA42_OWNED = _QUEUE_OWNED_CHILDREN
_stage_source = inspect.getsource(_impl.run_stage)
_needle = 'identity = proc_identity(process.pid)'
require(_stage_source.count(_needle) == 1, "Preserved queue process registration changed")
_stage_source = _stage_source.replace(_needle, '_EXTRA42_OWNED.append((process, command, source, None))\n                    ' + _needle
    + '\n                    _EXTRA42_OWNED[-1] = (process, command, source, identity)')
exec(compile(_stage_source.replace("def run_stage(", "def _mac9_stage(", 1), str(_BASE), "exec"), _impl.__dict__)


def run_stage(c, host, contract_path, stage, started):
    try: return _impl._mac9_stage(c, host, contract_path, stage, started)
    finally:
        for child,command,source,identity in _QUEUE_OWNED_CHILDREN:
            if child.poll() is None: cleanup_owned_synthetic(child,command,source,identity)


def status(c, host):
    value = _impl.status(c, host)
    root = Path(c["hosts"][host]["root"])
    live = value.get("dispatcher_current_identity")
    if live is not None:
        launched = read(root / "dispatcher_launch.json")["identity"]
        require(all(live[k] == launched[k] for k in ("pid", "start_ticks", "argv", "cwd", "pgid")), "Dispatcher current identity changed")
    return value


def dispatch(c, host, contract_path):
    jobs = validate_contract(c, host); h = c["hosts"][host]; root = Path(h["root"]); source = Path(h["source_root"])
    require(Path.cwd().resolve() == source.resolve() and not any((root / n).exists() for n in (
        "dispatch_attempt.json", "start.json", "run", "dispatch_failure.json", "final_receipt.json")), "Existing scientific attempt; no retry")
    verify_files(c, source); q = read(root / "qualification.json")
    require(q["status"] == "passed" and q["contract_sha256"] == sha_json(c)
        and q["maximum_simultaneous_workers"] == 3 and q["native_worker_exit0_count"] == len(ARMS) + 3, "All native admissions required")
    predecessor_gate(c, host); write(root / "dispatch_attempt.json", {"at": now(), "contract_sha256": sha_json(c)}, exclusive=True)
    identity = proc_identity(os.getpid()); require(identity is not None, "Dispatcher actual identity unavailable")
    write(root / "start.json", {"pid": os.getpid(), "identity": identity, "host": host, "started_at": now(), "jobs": jobs,
        "contract_sha256": sha_json(c), "qualification_sha256": sha_file(root / "qualification.json"), "maximum_concurrent_workers": 3}, exclusive=True)
    (root / "logs").mkdir(exist_ok=False); started = time.monotonic(); stages = {}
    for stage in STAGES:
        stages[stage] = run_stage(c, host, contract_path, stage, started)
        if stages[stage]["status"] != "complete": break
        if stage == "validation": validation_gate(c, host, stages[stage]["exits"])
    pids = [e["pid"] for value in stages.values() for e in value["exits"].values()]
    gpu, paused = gpu_state(c, host); verify_files(c, source)
    absent = all(proc_identity(pid) is None for pid in pids) and not set(pids).intersection(gpu["compute_pids"])
    complete = len(stages) == 3 and all(v["status"] == "complete" for v in stages.values()) and absent
    final = {"status": "complete" if complete else "failed", "host": host, "contract_sha256": sha_json(c), "completed_at": now(),
        "completed_fit_count": len(stages.get("train", {}).get("receipts", {})), "validation_replay_count": len(stages.get("validation", {}).get("receipts", {})),
        "test_count": len(stages.get("test", {}).get("receipts", {})), "stages": stages, "owned_worker_pids_absent": absent,
        "gpu_after": gpu, "paused_after": paused, "prior_campaign_resumed": False, "held_out_test_evaluated": "test" in stages,
        "existing_models_retrained": False, "dispatcher_own_exit_directly_observed": False}
    write(root / "status.json", final); final["artifact_manifest"] = artifact_manifest(root)
    write(root / "final_receipt.json", final, exclusive=True); return 0 if complete else 1


def launch_dispatch(c, host, contract_path):
    """Qualification is a separate child, so its CUDA context exits before fits."""
    h = c["hosts"][host]; root = Path(h["root"])
    write(root / "launch_attempt.json", {"pid": os.getpid(), "at": now()}, exclusive=True)
    command = [h["python"], "-u", "-m", MODULE, "--contract", str(contract_path), "--host", host, "--mode", "qualify"]
    with (root / "qualification.stdout").open("xb") as out, (root / "qualification.stderr").open("xb") as err:
        child = subprocess.Popen(command, cwd=h["source_root"], env={**os.environ, **h["environment"]}, stdout=out, stderr=err, start_new_session=True)
        timeout = False; identity = None
        try:
            identity = proc_identity(child.pid)
            require(identity is not None and identity["argv"] == command and identity["cwd"] == h["source_root"]
                and identity["pgid"] == child.pid, "Native qualification child identity differs")
            try: code = child.wait(timeout=max(.001, min(7200., h["absolute_host_deadline_unix"] - time.time())))
            except subprocess.TimeoutExpired: timeout = True; cleanup_owned_synthetic(child,command,Path(h["source_root"]),identity,grace=90); code = child.returncode
        finally:
            try:
                if child.poll() is None: cleanup_owned_synthetic(child,command,Path(h["source_root"]),identity,grace=90)
            finally:
                write(root / "qualification_exit.json", {"pid": child.pid, "command": command, "start_ticks": identity.get("start_ticks") if identity else None,
                    "actual_os_returncode": child.returncode, "timeout": timeout, "stdout_sha256": sha_file(root / "qualification.stdout"),
                    "stderr_sha256": sha_file(root / "qualification.stderr")}, exclusive=True)
    require(code == 0 and not timeout and proc_identity(child.pid) is None, "Native qualification did not successfully exit")
    return dispatch(c, host, contract_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5090",), required=True); parser.add_argument("--mode", choices=("qualify", "launch_dispatch", "status"), required=True)
    a = parser.parse_args(argv); require(a.contract.is_absolute(), "Absolute contract required"); c = read(a.contract)
    if a.mode in ("launch_dispatch", "qualify"):
        def interrupted(signum, frame): raise KeyboardInterrupt("Only the new mac9 owner was interrupted")
        signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    if a.mode == "launch_dispatch":
        return launch_dispatch(c, a.host, a.contract)
    value = qualify(c, a.host) if a.mode == "qualify" else status(c, a.host)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False)); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception: traceback.print_exc(); raise SystemExit(1)
