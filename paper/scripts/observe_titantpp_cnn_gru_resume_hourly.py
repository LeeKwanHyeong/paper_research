#!/usr/bin/env python3
"""One read-only SSH snapshot per owner of the explicit A100→RTX continuation.

The observer reads named Train/Validation JSON only. It never loads checkpoint
binaries, performs inference, starts/resumes jobs or changes remote files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1"
SCHEMA = "titantpp_cnn_gru_resume_dual_v1"
VARIANT = "count_only_log_regression"
KST = timezone(timedelta(hours=9))
SMALL_NAMES = {"status.json", "failure.json", "input_receipt.json", "initialization.json",
    "source_continuation.json", "history.json", "exposure.json", "epoch_timing.json",
    "server_checkpoint_receipt.json", "endpoint_replays.json", "selected_train_validation_diagnostic.json",
    "width16_candidate_comparison.json", "batch_exposure_comparison.json"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
        allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def kst(unix):
    return datetime.fromtimestamp(unix, KST).strftime("%Y-%m-%d %H:%M:%S KST")


def load_contract(bundle, host=None):
    """Select a sealed host revision without rebinding another running owner.

The historical one-argument API reads the immutable primary contract. A host
pointer may change its dedicated Python/libraries/environment only; original
science, jobs, checkpoint seals and the other host remain identical.
"""
    bundle = Path(bundle).resolve()
    primary = read(bundle / "execution_contract.json")
    c, current = primary, None
    if (bundle / "current.json").exists():
        current = read(bundle / "current.json")
        require(current.get("contract_sha256", canonical(primary)) == canonical(primary),
            "Primary current contract pointer differs")
        if "bundle" in current:
            require(Path(current["bundle"]).resolve() == bundle, "Current bundle pointer differs")
        if host is not None and "host_contracts" in current:
            require(host in ("5080", "5090") and set(current["host_contracts"]) == {"5080", "5090"},
                "Incomplete/foreign host contract pointers")
            pointer = current["host_contracts"][host]
            path = (bundle / pointer["path"]).resolve()
            require(path.is_relative_to(bundle) and path.suffix == ".json", "Host contract path outside bundle")
            c = read(path)
            require(canonical(c) == pointer["contract_sha256"], "Selected host contract pointer differs")
            for key in ("schema", "parent", "source", "worker", "training", "evaluation_scope",
                        "held_out_test_evaluated", "replay_tolerance", "limits", "jobs", "artifacts",
                        "path_mappings", "operation_adapter_sha256", "operation_supervisor_sha256"):
                require(c.get(key) == primary.get(key), "Host revision changed science/ownership: " + key)
            require(set(c["hosts"]) == set(primary["hosts"]) == {"5080", "5090"}, "Host coverage differs")
            for owner in primary["hosts"]:
                if owner != host:
                    require(c["hosts"][owner] == primary["hosts"][owner], "Revision changed another host")
                else:
                    protected = set(primary["hosts"][owner]) | set(c["hosts"][owner])
                    protected -= {"python", "runtime_expected", "environment"}
                    require(all(c["hosts"][owner].get(key) == primary["hosts"][owner].get(key)
                        for key in protected), "Revision changed dedicated root/GPU identity")
    require(c.get("schema") == SCHEMA and set(c["hosts"]) == {"5080", "5090"}, "Foreign continuation target")
    require(c["evaluation_scope"] == "validation_only" and c["held_out_test_evaluated"] is False,
        "Test access is forbidden")
    require(c["limits"]["workers_per_host"] == 1 and c["limits"]["automatic_retry"] is False,
        "Foreign operating limits")
    require(len(c["jobs"]) == len({j["id"] for j in c["jobs"]}), "Duplicate canonical jobs")
    return c


def remote_script(c, host):
    """Entire remote reader uses only standard-library and nvidia-smi queries."""
    spec, jobs = c["hosts"][host], [j for j in c["jobs"] if j["host"] == host]
    constants = {"root": spec["root"], "host": host, "gpu_uuid": spec["gpu_uuid"],
        "python": spec["python"], "jobs": jobs, "contract_sha256": canonical(c),
        "source_epochs": {j["id"]: c["artifacts"][j["id"]]["source_epoch"] for j in jobs},
        "small_names": sorted(SMALL_NAMES), "variant": VARIANT}
    return "cfg=" + repr(constants) + "\n" + r'''
import csv,hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path(cfg['root']).resolve()
out={'host':cfg['host'],'root':str(root),'observed_unix':time.time(),'files':{},'file_sha256':{},'runs':[],'processes':[],'record_errors':{}}
def bytes_sha(raw):return hashlib.sha256(raw).hexdigest()
def safe_json(path):
    path=pathlib.Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('Path outside dedicated root')
    if not path.exists():return None
    try:
        raw=path.read_bytes()
        if len(raw)>32*1024*1024:raise ValueError('JSON exceeds observation size limit')
        value=json.loads(raw)
        def reject_test(x):
            if isinstance(x,dict):
                if x.get('held_out_test_evaluated') is True:raise ValueError('Held-out record forbidden')
                if x.get('evaluation_scope') not in (None,'validation_only','train_validation_only','train_validation'):raise ValueError('Mixed evaluation scope forbidden')
                for k,v in x.items():
                    if k=='test' or k.startswith('test_') or k.startswith('held_out_predictions'):raise ValueError('Test field forbidden')
                    reject_test(v)
            elif isinstance(x,list):
                for y in x:reject_test(y)
        reject_test(value)
        name=str(path.relative_to(root));out['file_sha256'][name]=bytes_sha(raw)
        return value
    except Exception as error:
        out['record_errors'][str(path.relative_to(root))]=type(error).__name__
        return None
for name in ('execution_contract.json','supervisor.claim','supervisor_started.json','supervisor_status.json','supervisor_terminal.json','supervisor_failure.json'):
    out['files'][name]=safe_json(root/name)
for job in cfg['jobs']:
    folder=root/'run'/job['id'];run=folder/'runs'/job['arm']/cfg['variant']/'seed_42'
    row={'job':job,'folder_exists':folder.exists(),'small_records':{},'small_record_checks':{}}
    for label,path in [('status',folder/'status.json'),('failure',folder/'failure.json'),('source_continuation',folder/'source_continuation.json'),('manifest',folder/'terminal_manifest.json'),('history',run/'history.json'),('exposure',run/'exposure.json'),('timing',run/'epoch_timing.json'),('checkpoint_receipt',run/'server_checkpoint_receipt.json'),('endpoints',run/'endpoint_replays.json'),('claim',root/'claims'/(job['id']+'.json')),('progress',root/'progress'/(job['id']+'.json')),('worker_started',root/'worker_started'/(job['id']+'.json'))]:
        row[label]=safe_json(path)
    checkpoint=run/'last_epoch_state.pt'
    row['last_checkpoint_bytes']=checkpoint.stat().st_size if checkpoint.exists() else None
    row['initial_checkpoint_sha256']=None
    # Before its first new saved epoch, verify the copied binary by streaming
    # SHA only. Never deserialize/infer, and never hash routine active epochs.
    if row.get('checkpoint_receipt') is None and checkpoint.exists() and len((row.get('history') or {}).get('history',[]))==cfg['source_epochs'][job['id']]:
        try:
            before=checkpoint.stat()
            if 0<before.st_size<=128*1024*1024:
                digest=hashlib.sha256()
                with checkpoint.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
                after=checkpoint.stat()
                if (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns):
                    row['initial_checkpoint_sha256']=digest.hexdigest()
        except OSError:
            out['record_errors'][str(checkpoint.relative_to(root))]='CheckpointChangedOrUnavailable'
    manifest=row.get('manifest') or {}
    for name,digest in manifest.get('files',{}).items():
        candidate=folder/name
        if pathlib.Path(name).name not in cfg['small_names'] or pathlib.Path(name).suffix!='.json':continue
        if not candidate.resolve().is_relative_to(folder.resolve()):
            row['small_record_checks'][name]=False
            continue
        value=safe_json(candidate);relative=str(candidate.resolve().relative_to(root)) if candidate.resolve().is_relative_to(root) else ''
        row['small_records'][name]=value
        row['small_record_checks'][name]=value is not None and out['file_sha256'].get(relative)==digest
    out['runs'].append(row)
for entry in (pathlib.Path('/proc').iterdir() if pathlib.Path('/proc').exists() else []):
    if not entry.name.isdigit():continue
    try:
        argv=[x.decode(errors='replace') for x in (entry/'cmdline').read_bytes().split(b'\0') if x]
        if len(argv)<2 or argv[0]!=cfg['python']:continue
        allowed=[str(root/'operation/titantpp_cnn_gru_resume_supervisor.py'),str(root/'operation/titantpp_cnn_gru_resume_runtime.py')]
        if argv[1] not in allowed or '--contract' not in argv or argv[argv.index('--contract')+1]!=str(root/'execution_contract.json'):continue
        if '--host' not in argv or argv[argv.index('--host')+1]!=cfg['host']:continue
        status=(entry/'status').read_text();ppid=next(int(x.split()[1]) for x in status.splitlines() if x.startswith('PPid:'))
        out['processes'].append({'pid':int(entry.name),'ppid':ppid,'argv':argv,'cwd':os.readlink(entry/'cwd')})
    except (OSError,ValueError,IndexError,StopIteration):pass
def gpu_query(args):
    try:
        p=subprocess.run(['nvidia-smi',*args],capture_output=True,text=True,timeout=20)
        return {'returncode':p.returncode,'rows':list(csv.reader(p.stdout.splitlines())) if p.returncode==0 else [],'stderr_present':bool(p.stderr)}
    except (OSError,subprocess.TimeoutExpired) as error:
        return {'returncode':None,'rows':[],'error_type':type(error).__name__}
out['gpu']=gpu_query(['--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
out['compute']=gpu_query(['--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'])
print(json.dumps(out,allow_nan=False))
'''


def snapshot_once(c, host, runner=subprocess.run):
    argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, "python3 -"]
    result = runner(argv, input=remote_script(c, host), capture_output=True, text=True, timeout=90)
    require(result.returncode == 0, "Read-only SSH failed (returncode=" + str(result.returncode) + ")")
    return json.loads(result.stdout)


def earliest_best(history):
    eligible = [row for row in history if isinstance(row.get("val_qty_rmse"), (int, float))
        and math.isfinite(row["val_qty_rmse"])]
    return min(eligible, key=lambda row: (row["val_qty_rmse"], row["epoch"])) if eligible else None


def initial_resume_proof(snapshot, c, host, job, row, pids, supervisors, best):
    """Show a live restored fit before first-save receipt, never invent a save."""
    artifact = c["artifacts"][job["id"]]
    source = row.get("source_continuation") or {}
    claim, progress, worker = (row.get(name) or {} for name in ("claim", "progress", "worker_started"))
    status = snapshot["files"].get("supervisor_status.json") or {}
    source_epoch = artifact["source_epoch"]
    history = (row.get("history") or {}).get("history", [])
    prefix = "run/" + job["id"] + "/runs/" + job["arm"] + "/" + VARIANT + "/seed_42/"
    processes = [p for p in snapshot.get("processes", []) if p["pid"] in pids]
    if len(pids) != 1 or len(processes) != 1 or best is None:
        return False
    process, pid = processes[0], pids[0]
    age = snapshot["observed_unix"] - progress.get("updated_unix", 0)
    return (row.get("checkpoint_receipt") is None and len(history) == source_epoch
        and [r.get("epoch") for r in history] == list(range(1, source_epoch + 1))
        and best["epoch"] == artifact["source_best_epoch"]
        and snapshot.get("file_sha256", {}).get(prefix + "history.json") == artifact["files"]["history.json"]
        and row.get("initial_checkpoint_sha256") == artifact["source_checkpoint_sha256"]
        and artifact["origin_contract_sha256"] == c["parent"]["contract_sha256"]
        and source.get("status") == "verified" and source.get("contract_sha256") == canonical(c)
        and source.get("job") == job and source.get("host") == host
        and source.get("source_host") == "a100" and source.get("target_host") == host
        and source.get("optimizer_rng_loader_generator_present") is True
        and source.get("segment_boundary_epoch") == source_epoch and source.get("next_epoch") == source_epoch + 1
        and all(source.get(key) == artifact[key] for key in ("source_epoch", "source_best_epoch",
            "source_checkpoint_sha256", "source_best_state_sha256", "source_current_state_sha256", "source_resume_identity_sha256"))
        and claim.get("job") == job and claim.get("contract_sha256") == canonical(c)
        and claim.get("host") == host and claim.get("pid") == pid
        and claim.get("source_checkpoint_sha256") == artifact["source_checkpoint_sha256"]
        and claim.get("manual_resume_once") is True and claim.get("automatic_retry") is False
        and claim.get("deadline_unix", 0) > snapshot["observed_unix"]
        and progress.get("job") == job and progress.get("contract_sha256") == canonical(c)
        and progress.get("pid") == pid and -5 <= age <= 90
        and worker.get("pid") == pid and worker.get("job") == job
        and worker.get("contract_sha256") == canonical(c) and worker.get("command") == process["argv"]
        and worker.get("supervisor_pid") in supervisors and process.get("ppid") == worker.get("supervisor_pid")
        and status.get("status") == "running" and status.get("contract_sha256") == canonical(c)
        and status.get("active_job") == job and status.get("worker_pid") == pid)


def analyse(snapshot, c, host):
    require(snapshot["host"] == host and snapshot["root"] == c["hosts"][host]["root"], "Wrong live owner root")
    require(snapshot["files"].get("execution_contract.json") == c, "Remote execution contract differs")
    gpu_rows = snapshot.get("gpu", {}).get("rows", [])
    gpu_ok = snapshot.get("gpu", {}).get("returncode") == 0 and any(row and row[0].strip()
        == c["hosts"][host]["gpu_uuid"] for row in gpu_rows)
    compute_ok = snapshot.get("compute", {}).get("returncode") == 0
    gpu_pids = {int(row[1].strip()) for row in snapshot.get("compute", {}).get("rows", [])
        if len(row) > 1 and row[0].strip() == c["hosts"][host]["gpu_uuid"] and row[1].strip().isdigit()}
    supervisors, fit_processes = [], {}
    for process in snapshot.get("processes", []):
        argv = process["argv"]
        if process.get("cwd") != c["hosts"][host]["source_root"]:
            continue
        if argv[1] == c["hosts"][host]["root"] + "/operation/titantpp_cnn_gru_resume_supervisor.py":
            supervisors.append(process["pid"])
        elif "--mode" in argv and argv[argv.index("--mode") + 1] == "fit" and "--job" in argv:
            fit_processes.setdefault(argv[argv.index("--job") + 1], []).append(process["pid"])
    counts = {key: 0 for key in ("completed", "failed", "in_progress", "pending", "unstarted_terminal", "unknown")}
    rows, jobs = [], [j for j in c["jobs"] if j["host"] == host]
    runs = {r["job"]["id"]: r for r in snapshot["runs"]}
    require(len(runs) == len(snapshot["runs"]) and set(runs) == {j["id"] for j in jobs}, "Snapshot job coverage differs")
    for job in jobs:
        r = runs[job["id"]]
        history = (r.get("history") or {}).get("history", [])
        best = earliest_best(history)
        checkpoint_receipt = r.get("checkpoint_receipt") or {}
        saved = checkpoint_receipt.get("epoch")
        checkpoint_prefix = "run/" + job["id"] + "/runs/" + job["arm"] + "/" + VARIANT + "/seed_42/"
        checkpoint_records_ok = bool(checkpoint_receipt) and checkpoint_receipt.get("contract_sha256") == canonical(c) \
            and all(checkpoint_receipt.get("files", {}).get(name) is not None
                and snapshot.get("file_sha256", {}).get(checkpoint_prefix + name) == checkpoint_receipt["files"][name]
                for name in ("history.json", "exposure.json", "epoch_timing.json"))
        status = r.get("status") or {}
        manifest, endpoint = r.get("manifest") or {}, r.get("endpoints") or {}
        issues, pids, phase, initial_verified = [], fit_processes.get(job["id"], []), None, False
        checked_names = {Path(name).name for name in r.get("small_record_checks", {})}
        required_names = {"status.json", "history.json", "exposure.json", "epoch_timing.json", "endpoint_replays.json"}
        small = required_names <= checked_names and all(r["small_record_checks"].values())
        selected_epoch = endpoint.get("best_epoch")
        history_complete = bool(history) and [x.get("epoch") for x in history] == list(range(1, len(history) + 1))
        checks = endpoint.get("replay_checks", {})
        replay = bool(checks) and set(checks) == {"selected", "last"} \
            and all(set(check) == {"qty_rmse", "qty_mae", "time_nll"} for check in checks.values()) and all(value is True
            for check in checks.values() for value in check.values())
        endpoint_verified = (endpoint.get("status") == "complete" and endpoint.get("job") == job
            and endpoint.get("evaluation_scope") == "validation_only" and endpoint.get("held_out_test_evaluated") is False
            and history_complete and best is not None and selected_epoch == best["epoch"]
            and endpoint.get("completed_epochs") == len(history) and replay
            and bool(endpoint.get("selected")) and bool(endpoint.get("last")))
        done = (manifest.get("status") == "complete" and manifest.get("scientific_success") is True
            and manifest.get("job") == job and manifest.get("contract_sha256") == canonical(c)
            and manifest.get("held_out_test_evaluated") is False and small and endpoint_verified)
        if done:
            state = "completed"
        elif (r.get("failure") or {}).get("job") == job:
            state = "failed"
        elif len(pids) == 1 and pids[0] in gpu_pids and gpu_ok and compute_ok:
            if saved is None and initial_resume_proof(snapshot, c, host, job, r, pids, supervisors, best):
                state, phase, initial_verified = "in_progress", "awaiting_first_resumed_checkpoint", True
                saved = c["artifacts"][job["id"]]["source_epoch"]
            elif saved is None or saved != len(history) or not r.get("last_checkpoint_bytes") \
                    or not history_complete or not checkpoint_records_ok:
                state = "unknown"
                issues.append("Checkpoint/history not yet consistent; do not infer a failed fit")
            else:
                state = "in_progress"
        elif pids:
            state = "unknown"
            issues.append("Owned worker/GPU PID missing or ambiguous; initialization/replay may be active")
        elif r.get("folder_exists") or status.get("status") or r.get("claim"):
            state = "unknown"
            issues.append("Owned process absent and scientific terminal incomplete; no automatic restart")
        elif snapshot["files"].get("supervisor_failure.json"):
            state = "unstarted_terminal"
        else:
            state = "pending"
        if not gpu_ok or not compute_ok:
            issues.append("GPU identity/process query is not verified")
            if state == "in_progress": state = "unknown"
        counts[state] += 1
        metrics = ({"epoch": best["epoch"], "qty_rmse": best["val_qty_rmse"], "qty_mae": best.get("val_qty_mae"),
            "time_nll": best.get("val_time_nll"), "best_so_far": not done} if best else None)
        source_epoch = c["artifacts"][job["id"]]["source_epoch"]
        timings = [x["elapsed_seconds"] for x in (r.get("timing") or {}).get("epochs", [])
            if saved is not None and source_epoch < x.get("epoch", 0) <= saved
            and isinstance(x.get("elapsed_seconds"), (int, float))
            and math.isfinite(x["elapsed_seconds"]) and x["elapsed_seconds"] > 0][-10:]
        median = statistics.median(timings) if timings else None
        eta = None
        if state == "in_progress" and not initial_verified and best and saved is not None and median:
            target = min(300, max(40, best["epoch"] + 40))
            remaining = max(0, target - saved)
            eta = {"target_epoch": target, "remaining_epochs": remaining,
                "conditional_finish_kst": kst(snapshot["observed_unix"] + remaining * median),
                "max300_finish_kst": kst(snapshot["observed_unix"] + max(0, 300 - saved) * median),
                "assumptions": "Best does not improve; current target-GPU speed/concurrency stays fixed; replay/retrieval excluded"}
        rows.append({"job": job, "state": state, "saved_epoch": saved, "best_epoch": best["epoch"] if best else None,
            "phase": phase, "initial_source_checkpoint_verified": initial_verified,
            "new_completed_epochs": max(0, saved - source_epoch) if saved is not None else None,
            "source_epoch": source_epoch, "validation_best_same_epoch": metrics, "worker_pids": pids,
            "confirmed_gpu_pids": sorted(set(pids) & gpu_pids), "terminal_verified": bool(done),
            "checkpoint_records_consistent": checkpoint_records_ok,
            "small_record_SHA_verified": bool(small), "selected_last_validation_replay_verified": bool(endpoint_verified),
            "selected_validation": endpoint.get("selected") if done else None,
            "recent_target_epoch_seconds_median": median, "recent_target_epoch_count": len(timings),
            "eta": eta, "issues": issues})
    actual = sum(len(pids) for pids in fit_processes.values())
    status = snapshot["files"].get("supervisor_status.json") or {}
    active = status.get("active_job")
    declared = {active["id"]} if active else set()
    matched = declared == set(fit_processes)
    server_receipt = snapshot["files"].get("supervisor_terminal.json") or {}
    server_terminal = (server_receipt.get("status") == "complete" and server_receipt.get("host") == host
        and server_receipt.get("contract_sha256") == canonical(c)
        and server_receipt.get("all_assigned_jobs_terminal_verified") is True
        and set(server_receipt.get("outcomes", {})) == {j["id"] for j in jobs}
        and all(server_receipt["outcomes"][j["id"]].get("terminal_manifest_sha256")
            == snapshot.get("file_sha256", {}).get("run/" + j["id"] + "/terminal_manifest.json") for j in jobs)
        and counts["completed"] == len(jobs) and not supervisors and actual == 0)
    return {"schema": "titantpp_cnn_gru_resume_observation_v1", "host": host,
        "contract_sha256": canonical(c), "source_closure_sha256": c["source"]["files_sha256"],
        "observed_unix": snapshot["observed_unix"], "observed_kst": kst(snapshot["observed_unix"]),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "status_counts": counts, "rows": rows, "GPU_UUID_verified": gpu_ok,
        "actual_owned_fit_count": actual, "actual_owned_GPU_fit_count": sum(len(set(p) & gpu_pids) for p in fit_processes.values()),
        "supervisor_pids": supervisors, "declared_active_jobs_match_processes": matched,
        "server_terminal_verified": server_terminal, "supervisor_failure": snapshot["files"].get("supervisor_failure.json"),
        "queue_eta": None, "queue_eta_reason": "Sequential pending slot durations or native fit speed may be unavailable",
        "binary_CPU_audit": "Initial copied checkpoint may be SHA-streamed only; never deserialized/inferred; originals retrieval and CPU audit separate",
        "record_errors": snapshot.get("record_errors", {})}


def aggregate_originals(bundle, c, analysis):
    """Bind recovered parent terminals and resumed owners by canonical job ID."""
    path = Path(bundle) / "recovery_inventory.json"
    require(path.exists(), "Original recovery inventory unavailable; whole campaign aggregate unverified")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if "recovery_inventory_sha256" in c:
        require(c["recovery_inventory_sha256"] == digest, "Recovery inventory seal differs")
    inventory = json.loads(raw)
    parent_path = (Path(bundle) / c["parent"]["contract"]).resolve()
    require(parent_path.is_relative_to(Path(bundle).resolve()), "Parent contract path outside local bundle")
    parent = read(parent_path)
    require(canonical(parent) == c["parent"]["contract_sha256"], "Local parent contract changed")
    admitted = {j["id"]: j for j in parent["jobs"]}
    originals = inventory["original_completed_jobs"]
    original_ids = {r["job"]["id"] for r in originals}
    require(len(original_ids) == len(originals), "Duplicated original terminal")
    target_ids = {j["id"] for j in c["jobs"]}
    require(not original_ids & target_ids and original_ids | target_ids == set(admitted),
        "Original/resumed canonical coverage overlaps or omits a parent condition")
    for row in originals:
        require(row["job"] == admitted[row["job"]["id"]] and row["terminal_scientific_success"] is True
            and row["small_record_SHA_verified"] is True and row["selected_last_validation_replay_verified"] is True,
            "Original terminal verification incomplete")
        require(isinstance(row["terminal_manifest_sha256"], str) and len(row["terminal_manifest_sha256"]) == 64
            and isinstance(row["observed_unix"], (int, float)) and math.isfinite(row["observed_unix"]),
            "Original terminal SHA/time provenance missing")
    return {"parent_conditions": len(admitted), "original_completed_count": len(originals),
        "original_completed_jobs": originals, "owned_resume_counts": analysis["status_counts"],
        "other_host_resume_job_ids": [j["id"] for j in c["jobs"] if j["host"] != analysis["host"]],
        "recovery_inventory_sha256": digest,
        "other_host_current_state": "Read its independent observer receipt; never infer from this host"}


def render(analysis):
    lines = ["# CNN·GRU A100→RTX 재개 관측", "", "실제 관측: " + (analysis.get("observed_kst") or "이번 조회 실패"), ""]
    if analysis.get("status") == "observation_failed_not_training_failure":
        return "\n".join(lines + ["조회 실패이며 학습 실패가 아닙니다. 이전 시각을 새 관측으로 표시하지 않습니다.", ""])
    lines += ["| 데이터·모델·seed | 상태 | 저장/best | 조건부 종료 |", "|---|---|---:|---|"]
    for row in analysis["rows"]:
        job = row["job"]
        eta = (row.get("eta") or {}).get("conditional_finish_kst", "미확정" if row["state"] != "completed" else "완료")
        lines.append(f"| {job['dataset']} / {job['arm']} / {job['seed']} | {row['state']} | {row['saved_epoch']}/{row['best_epoch']} | {eta} |")
    lines += ["", "ETA는 목적GPU에서 완료한 최근 최대10epoch의 중앙값으로 계산합니다. A100 구간 시간을 복사하지 않습니다.",
        "best 갱신·병렬 수 변경 시 달라지며 endpoint replay·원본 회수·CPU 감사 시간은 별도입니다.", ""]
    return "\n".join(lines)


def run_once(host, bundle=DEFAULT_BUNDLE, runner=subprocess.run):
    bundle = Path(bundle).resolve()
    requested = time.time()
    folder = bundle / "hourly_monitor" / (datetime.fromtimestamp(requested, timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + host)
    latest = bundle / "hourly_monitor" / ("latest_" + host + ".json")
    calls, snapshot, snapshot_path, digest, reused = 0, None, None, None, False
    try:
        c = load_contract(bundle, host)
        if latest.exists():
            previous = read(latest)
            if previous.get("server_terminal_verified") and previous.get("contract_sha256") == canonical(c):
                path = Path(previous["snapshot"]).resolve()
                require(path.is_relative_to((bundle / "hourly_monitor").resolve()), "Terminal cache outside bundle")
                raw = path.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == previous["snapshot_sha256"], "Cached terminal snapshot changed")
                snapshot = json.loads(raw)
                analysis = analyse(snapshot, c, host)
                require(analysis["server_terminal_verified"], "Cached terminal does not verify")
                reused = True
        if snapshot is None:
            calls = 1
            snapshot = snapshot_once(c, host, runner)
            require(requested - 120 <= snapshot["observed_unix"] <= time.time() + 120, "Stale live snapshot")
            analysis = analyse(snapshot, c, host)
        snapshot_path = folder / "snapshot.json"
        write(snapshot_path, snapshot)
        digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
        analysis["reused_terminal_evidence"] = reused
        try:
            analysis["original_campaign"] = aggregate_originals(bundle, c, analysis)
        except Exception as inventory_error:
            analysis["original_campaign"] = {"status": "aggregate_unverified", "reason": str(inventory_error)[:300]}
    except Exception as error:
        analysis = {"status": "observation_failed_not_training_failure", "host": host,
            "observed_unix": None, "observed_kst": None, "server_terminal_verified": False,
            "error_type": type(error).__name__, "error_message": str(error)[:500],
            "held_out_test_evaluated": False}
    write(folder / "analysis.json", analysis)
    (folder / "report.md").write_text(render(analysis), encoding="utf-8")
    receipt = {"host": host, "requested_unix": requested, "requested_kst": kst(requested),
        "observed_unix": analysis.get("observed_unix"), "observed_kst": analysis.get("observed_kst"),
        "contract_sha256": analysis.get("contract_sha256"), "ssh_calls": calls,
        "status": analysis.get("status", "observed"), "snapshot": str(snapshot_path) if snapshot_path else None,
        "snapshot_sha256": digest, "analysis": str(folder / "analysis.json"), "report": str(folder / "report.md"),
        "status_counts": analysis.get("status_counts"), "server_terminal_verified": analysis["server_terminal_verified"],
        "reused_terminal_evidence": reused, "fresh_observation": snapshot is not None and not reused}
    write(latest, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args(argv)
    print(json.dumps(run_once(args.host, args.bundle), ensure_ascii=False))


if __name__ == "__main__":
    main()
