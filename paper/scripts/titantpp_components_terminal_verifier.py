"""Offline terminal evidence verification without changing the sealed v5 launcher.

Final manifests bind final history after endpoint replay. A stale live checkpoint
receipt cannot replace these terminal facts. Remote binary bytes are not present
in these JSON snapshots: their SHA verification remains an explicit remote witness.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time
from paper.scripts import launch_titantpp_cnn_gru_components_v5 as base

require = base.require
canonical = base.canonical
decode_snapshot = base.decode_snapshot
TERMINAL = "training_complete_remote_verified"


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _require_absent(snapshot, group_pids, message):
    for pid, process in snapshot.get("processes", {}).items():
        associated = int(pid) in group_pids or process.get("pgrp") in group_pids or process.get("ppid") in group_pids
        if associated:
            require(process.get("status") == "disappeared", message)
    require(not set(snapshot.get("gpu_pids", [])) & group_pids, message)


def _dispatch_identity(files, c, host):
    digest = canonical(c)
    launch = json.loads(files["phases/train_launch.json"])
    claim = json.loads(files["supervisor.claim"])
    expected = [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
        "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"]
    require(launch["phase"] == "train" and launch["contract_sha256"] == digest
        and claim["contract_sha256"] == digest and launch["argv"] == claim["argv"] == expected
        and type(launch["pid"]) is int and launch["pid"] > 0
        and launch["pid"] == claim["pid"] and launch["ppid"] == claim["ppid"],
        "Training dispatcher launch/claim PID/PPID/argv mismatch")
    return launch


def verify_completed_job(snapshot, c, host, job):
    """Verify one completed fit even if other fits run or dispatch later fails.

    Does not claim host-wide completion, idle, or dispatcher exit0. The caller
    must separately verify those facts when declaring an entire root terminal.
    """
    files = decode_snapshot(snapshot, c, host)
    require(job in c["jobs"] and job["host"] == host, "Foreign completed job")
    name, digest = job["id"], canonical(c)
    terminal_name = "run/" + name + "/terminal_manifest.json"
    terminal = json.loads(files[terminal_name])
    require(terminal["job"] == job and terminal["contract_sha256"] == digest
        and terminal["status"] == "complete" and terminal["scientific_success"] is True
        and terminal["evaluation_scope"] == "validation_only" and terminal["held_out_test_evaluated"] is False,
        "Foreign training terminal")
    manifest = terminal["files"]
    witness = snapshot.get("terminal_hashes", {}).get(name, {})
    require(bool(manifest) and witness.get("all_files_verified") is True
        and set(witness.get("files", {})) == set(manifest)
        and all(value is True for value in witness["files"].values()), "Terminal file integrity unconfirmed")
    relative_run = "runs/" + job["arm"] + "/count_only_log_regression/seed_" + str(job["seed"])
    required = {"initialization.json", "input_receipt.json", "status.json", "two_epoch_cost_receipt.json"}
    required.update(relative_run + "/" + tail for tail in (
        "best_val_qty_rmse_model.pt", "last_epoch_state.pt", "history.json", "epoch_timing.json",
        "exposure.json", "endpoint_replays.json", "summary.json", "server_checkpoint_receipt.json", "train.log"))
    require(required <= set(manifest), "Terminal binary/final history manifest incomplete")
    prefix = "run/" + name + "/"
    for relative, expected in manifest.items():
        base.safe(Path("/manifest"), relative)
        require(isinstance(expected, str) and len(expected) == 64
            and all(ch in "0123456789abcdef" for ch in expected), "Invalid terminal file SHA")
        captured = prefix + relative
        if captured in files:
            require(hashlib.sha256(files[captured]).hexdigest() == expected, "Final manifest/captured file SHA mismatch")
    history_name, timing_name = prefix + relative_run + "/history.json", prefix + relative_run + "/epoch_timing.json"
    history, timing = json.loads(files[history_name]), json.loads(files[timing_name])
    rows, timed = history["history"], timing["epochs"]
    saved = len(rows)
    require(c["training"]["minimum_epochs"] <= saved <= c["training"]["maximum_epochs"]
        and [row.get("epoch") for row in rows] == list(range(1, saved + 1)), "Terminal saved history epochs inconsistent")
    finite = [(row["val_qty_rmse"], row["epoch"]) for row in rows if _finite(row.get("val_qty_rmse"))]
    require(bool(finite), "No finite terminal full Validation selector history")
    best = min(finite, key=lambda item: (item[0], item[1]))[1]
    require(timing["scope"] == "train_iteration_through_validation_and_checkpoint_save"
        and len(timed) == saved and [row.get("epoch") for row in timed] == list(range(1, saved + 1))
        and all(_finite(row.get("elapsed_seconds")) and row["elapsed_seconds"] > 0 for row in timed),
        "Terminal complete epoch timing inconsistent")
    status = json.loads(files[prefix + "status.json"])
    require(status["status"] == "complete" and status["job"] == job, "Final fit status/job mismatch")
    worker_name, claim_name, lease_name = "worker_exits/" + name + ".json", "claims/" + name + ".json", "leases/" + name + ".json"
    worker, claim, lease = [json.loads(files[key]) for key in (worker_name, claim_name, lease_name)]
    parent = _dispatch_identity(files, c, host)
    require(worker["actual_child_exit_observed"] is True and worker["returncode"] == 0
        and worker["job"] == job and worker["contract_sha256"] == digest
        and worker["supervisor_pid"] == parent["pid"] and type(worker["worker_group_pid"]) is int
        and worker["worker_group_pid"] > 0, "Worker actual OS exit/job/parent mismatch")
    argv = worker["argv"]
    require(claim["job"] == job and claim["contract_sha256"] == digest and len(argv) == 16
        and Path(argv[0]).name == "timeout" and argv[1:3] == ["--signal=TERM", "--kill-after=15s"]
        and argv[4:14] == [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
            "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host,
            "--mode", "fit", "--job", name] and argv[14] == "--deadline"
        and _finite(claim["deadline_unix"]) and float(argv[15]) == claim["deadline_unix"]
        and float(argv[3]) > 0 and float(argv[3]) <= c["limits"]["per_condition_seconds"],
        "Worker timeout argv/claim identity mismatch")
    require(lease["job"] == name and lease["contract_sha256"] == digest
        and lease["worker_group_pid"] == worker["worker_group_pid"]
        and lease["supervisor_pid"] == parent["pid"] and _finite(lease["issued_unix"])
        and _finite(lease["expires_unix"]) and lease["issued_unix"] < lease["expires_unix"] <= claim["deadline_unix"],
        "Completed job lease identity mismatch")
    require(_finite(terminal["completed_unix"]) and _finite(worker["observed_unix"])
        and terminal["completed_unix"] <= worker["observed_unix"] <= snapshot["observed_unix"]
        and worker["observed_unix"] <= claim["deadline_unix"], "Completed job timestamps exceed observed lease")
    _require_absent(snapshot, {worker["worker_group_pid"]}, "Completed worker remains observed")
    state = json.loads(files.get("status.json", b"{}"))
    binding = state.get("completed", {}).get(name)
    require(binding is not None and binding["worker_exit_sha256"] == hashlib.sha256(files[worker_name]).hexdigest()
        and binding["terminal_manifest_sha256"] == hashlib.sha256(files[terminal_name]).hexdigest(),
        "Dispatcher completed-job rawSHA binding mismatch")
    return {"job": job, "status": "complete", "saved_epoch": saved, "best_epoch": best,
        "best_epoch_source": "final terminal-manifest-bound full Validation history; strict earliest finite selector",
        "selected_checkpoint_file_present": True, "terminal_manifest_raw_sha256": hashlib.sha256(files[terminal_name]).hexdigest(),
        "worker_exit_raw_sha256": hashlib.sha256(files[worker_name]).hexdigest(),
        "remote_all_terminal_file_hashes_verified": True, "local_binary_bytes_retrieved": False}


def training_status(snapshot, c, host):
    """Report saved evidence and live ownership without inferring fit failures."""
    files = decode_snapshot(snapshot, c, host)
    digest = canonical(c)
    def record(name):
        return json.loads(files[name]) if name in files else None
    failure = next((name for name in files if name == "failure.json" or name == "phases/train_failure.json"), None)
    state = record("status.json") or {}
    active = state.get("active_jobs", {})
    owned = {job["id"]: job for job in c["jobs"] if job["host"] == host}
    jobs = []
    for name, job in owned.items():
        terminal = record("run/" + name + "/terminal_manifest.json")
        item = {"job": job, "status": "waiting", "saved_epoch": None, "best_epoch": None}
        run = "run/" + name + "/runs/" + job["arm"] + "/count_only_log_regression/seed_" + str(job["seed"])
        if terminal:
            item.update(verify_completed_job(snapshot, c, host, job))
        elif name in active:
            require(active[name]["job"] == job, "Foreign active job")
            item["status"] = "running_before_first_checkpoint"
        saved = snapshot.get("checkpoints", {}).get(name)
        if saved and not terminal:
            if saved["snapshot_complete"] is True:
                history = record(run + "/history.json")
                require(history is not None and len(history["history"]) == saved["saved_epoch"], "Saved history/epoch mismatch")
                rows = history["history"]
                finite = [(row["val_qty_rmse"], row["epoch"]) for row in rows
                    if type(row.get("val_qty_rmse")) in (int, float) and math.isfinite(row["val_qty_rmse"])]
                require(bool(finite), "No finite saved full Validation selector history")
                item.update(saved_epoch=saved["saved_epoch"], best_epoch=min(finite, key=lambda row: (row[0], row[1]))[1],
                    best_epoch_source="saved full Validation history; original strict earliest selector",
                    selected_checkpoint_file_present=saved["best_selected_file_present"])
                if item["status"] == "running_before_first_checkpoint":
                    item["status"] = "running"
            else:
                item["status"] = "checkpoint_snapshot_unconfirmed"
        jobs.append(item)
    completed = sum(item["status"] == "complete" for item in jobs)
    report = {"jobs": jobs, "completed": completed, "total": len(owned),
        "gpu_pids": snapshot["gpu_pids"], "observed_unix": snapshot.get("observed_unix"),
        "whole_queue_eta": "unconfirmed", "pipeline_os_exit_directly_observed": False}
    if failure:
        return {**report, "status": "training_failed", "failure": record(failure), "evidence": failure,
            "scientific_failure_inferred": False}
    complete, exit_record = record("phases/train_complete.json"), record("phases/train_exit.json")
    if complete:
        launch, supervisor_claim = record("phases/train_launch.json"), record("supervisor.claim")
        expected_phase_argv = [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
            "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"]
        require(complete["phase"] == "train" and complete["host"] == host and complete["contract_sha256"] == digest
            and complete["subprocess_os_exit_code"] == 0 and not complete["gpu_compute_pids"]
            and exit_record is not None and exit_record["phase"] == "train"
            and exit_record["os_exit_code"] == 0 and exit_record["contract_sha256"] == digest,
            "Training phase completion/actual child exit mismatch")
        require(launch is not None and launch["phase"] == "train" and launch["contract_sha256"] == digest
            and launch["pid"] == exit_record["pid"] and launch["ppid"] == exit_record["ppid"]
            and launch["argv"] == exit_record["argv"] == expected_phase_argv,
            "Training phase launch/exit PID/PPID/argv mismatch")
        require(supervisor_claim is not None and supervisor_claim["pid"] == exit_record["pid"]
            and supervisor_claim["ppid"] == exit_record["ppid"] and supervisor_claim["argv"] == expected_phase_argv
            and supervisor_claim["contract_sha256"] == digest, "Training dispatcher ownership claim mismatch")
        require(state.get("status") == "complete" and set(state.get("completed", {})) == set(owned)
            and state.get("completed_conditions") == len(owned) and state.get("held_out_test_evaluated") is False,
            "Training dispatcher completion identities incomplete")
        for name, job in owned.items():
            worker_name, terminal_name = "worker_exits/" + name + ".json", "run/" + name + "/terminal_manifest.json"
            require(worker_name in files and terminal_name in files, "Worker actual exit/terminal raw evidence missing")
            binding = state["completed"][name]
            require(binding["worker_exit_sha256"] == hashlib.sha256(files[worker_name]).hexdigest()
                and binding["terminal_manifest_sha256"] == hashlib.sha256(files[terminal_name]).hexdigest(),
                "Dispatcher worker-exit/terminal rawSHA mismatch")
            worker, claim = record(worker_name), record("claims/" + name + ".json")
            require(worker["actual_child_exit_observed"] is True and worker["returncode"] == 0
                and worker["job"] == job and worker["contract_sha256"] == digest
                and worker["supervisor_pid"] == exit_record["pid"] and type(worker["worker_group_pid"]) is int
                and worker["worker_group_pid"] > 0, "Worker actual OS exit/job/parent mismatch")
            argv = worker["argv"]
            require(claim is not None and claim["job"] == job and claim["contract_sha256"] == digest
                and len(argv) == 16 and Path(argv[0]).name == "timeout"
                and argv[1:3] == ["--signal=TERM", "--kill-after=15s"]
                and argv[4:14] == [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
                    "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host,
                    "--mode", "fit", "--job", name] and argv[14] == "--deadline"
                and float(argv[15]) == claim["deadline_unix"], "Worker timeout argv/claim identity mismatch")
        _require_absent(snapshot, {exit_record["pid"]}, "Completed dispatcher remains observed")
        require(not snapshot["gpu_pids"] and completed == len(owned), "Terminal completion or GPU idle unconfirmed")
        return {**report, "status": "training_complete_remote_verified", "immutable_local_binary_retrieval": "not_performed"}
    if state.get("status") == "running":
        try:
            processes = snapshot["processes"]
            parent = processes[str(state["supervisor_pid"])]
            require(parent["argv"] == state["supervisor_argv"], "Supervisor PID/argv mismatch")
            require(state["supervisor_argv"] == [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
                "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host, "--mode", "dispatch"],
                "Supervisor argv outside approved study")
            require(len(active) <= 3 and time.time() < state["deadline_unix"], "Worker ceiling or lease expired")
            for name, worker in active.items():
                proc = processes[str(worker["worker_group_pid"])]
                require(proc["argv"] == worker["worker_guard_argv"] and proc["ppid"] == state["supervisor_pid"]
                    and proc["pgrp"] == worker["worker_group_pid"], "Worker timeout PID/PPID/argv mismatch")
                lease = record("leases/" + name + ".json")
                require(lease and lease["job"] == name and lease["contract_sha256"] == digest
                    and lease["worker_group_pid"] == worker["worker_group_pid"] and lease["supervisor_pid"] == state["supervisor_pid"]
                    and snapshot["observed_unix"] < lease["expires_unix"] <= worker["condition_deadline_unix"], "Active job lease mismatch")
            for pid in snapshot["gpu_pids"]:
                proc = processes[str(pid)]
                owners = [worker for worker in active.values() if proc["ppid"] == worker["worker_group_pid"]]
                require(len(owners) == 1 and proc["pgrp"] == owners[0]["worker_group_pid"]
                    and proc["argv"] == owners[0]["worker_guard_argv"][4:], "Foreign or unresolved GPU process ownership")
        except (KeyError, ValueError) as exc:
            return {**report, "status": "training_identity_unconfirmed", "reason": str(exc), "user_action_required": True}
        return {**report, "status": "training_running", "active_jobs": len(active), "identity_verified": True}
    return {**report, "status": "training_phase_pending_or_identity_unconfirmed",
        "reason": "A launch intent does not prove a live scientific fit"}


def qualification_bridge(bundle, c, host, snapshot):
    """Bind live/terminal q bytes to the original qualified idle observation.

    The remote status protocol does not capture training_permit.json. This proof
    therefore pins the local issued permit, observed start/approval, and identical
    native q bytes without claiming a direct remote permit raw-SHA observation.
    """
    bundle = Path(bundle).resolve()
    approval = base.authorization(bundle, c, host)
    files = decode_snapshot(snapshot, c, host)
    qpath = bundle / "control" / "qualification_cache" / host / "snapshot.json"
    cached = base.read(qpath)
    binding = base.qualification_binding(cached, c, host)
    original_files = decode_snapshot(cached, c, host)
    native_launch = json.loads(original_files["phases/qualify_launch.json"])
    require(native_launch["argv"] == [c["hosts"][host]["python"], c["hosts"][host]["operation_root"] + "/campaign.py",
        "--contract", c["hosts"][host]["root"] + "/execution_contract.json", "--host", host, "--mode", "qualify"],
        "Native qualification argv outside approved root")
    evidence = {"qualification/receipt.json": binding["raw_sha256"], **binding["receipt"]["qualification_files"]}
    evidence.update({name: hashlib.sha256(original_files[name]).hexdigest() for name in
        ("phases/qualify_launch.json", "phases/qualify_exit.json", "phases/qualify_complete.json")})
    for name, expected in evidence.items():
        require(name in files and hashlib.sha256(files[name]).hexdigest() == expected
            and files[name] == original_files[name], "Observed qualification bridge rawSHA mismatch")
    observed_approval = json.loads(files["approval.json"])
    start = json.loads(files["start_permit.json"])
    digest = canonical(c)
    require(observed_approval == approval and start["host"] == host and start["contract_sha256"] == digest
        and start["approval_sha256"] == canonical(approval), "Observed approval/start bridge mismatch")
    require(_finite(start["started_at_unix"]) and _finite(start["deadline_unix"])
        and start["deadline_unix"] - start["started_at_unix"] == c["limits"]["total_wall_seconds"]
        and start["started_at_unix"] <= cached["observed_unix"] <= snapshot["observed_unix"],
        "Original host lease/qualification observation mismatch")
    inherited = c.get("operational_replacement", {}).get("host_leases", {}).get(host)
    if inherited:
        require(start["started_at_unix"] == inherited["started_at_unix"]
            and start["deadline_unix"] == inherited["deadline_unix"], "Inherited original host lease extended")
    deployment_path = bundle / "control" / "deployment" / (host + ".json")
    deployment = base.read(deployment_path)
    require(deployment["status"] == "deployed" and deployment["host"] == host
        and deployment["root"] == c["hosts"][host]["root"] and deployment["contract_sha256"] == digest
        and deployment["start_permit"] == start and deployment["transport_os_exit_code"] == 0,
        "Deployment/start lease bridge mismatch")
    permit_path = bundle / "control" / "training_permits" / (host + ".json")
    permit = base.read(permit_path)
    require(permit["contract_sha256"] == digest and permit["start_permit_sha256"] == canonical(start)
        and permit["qualification_sha256"] == binding["raw_sha256"] and permit["qualification"] == binding["receipt"]
        and permit["qualifications"].get(host) == binding and permit["fresh_fits"] is True
        and permit["automatic_retry"] is False and _finite(permit["issued_unix"])
        and cached["observed_unix"] <= permit["issued_unix"] < start["deadline_unix"],
        "Issued training permit/native qualification bridge mismatch")
    complete_raw = files.get("phases/train_complete.json")
    if complete_raw:
        complete = json.loads(complete_raw)
        require(_finite(complete["completed_unix"]) and permit["issued_unix"] <= complete["completed_unix"]
            <= start["deadline_unix"], "Terminal completion outside original host lease")
    proof_paths = [qpath, permit_path, deployment_path, bundle / ("launch_authorization_" + host + ".json"),
        bundle / ("launch_release_" + host + ".json"), bundle / "launch_hold.json"]
    return {"host": host, "contract_sha256": digest, "qualification_raw_sha256": binding["raw_sha256"],
        "qualification_original_observed_unix": cached["observed_unix"],
        "observed_start_permit_raw_sha256": hashlib.sha256(files["start_permit.json"]).hexdigest(),
        "observed_start_permit_sha256": canonical(start), "original_host_lease": {
            "started_at_unix": start["started_at_unix"], "deadline_unix": start["deadline_unix"]},
        "local_proof_files": {str(path): base.sha(path) for path in proof_paths},
        "remote_training_permit_raw_directly_observed": False,
        "qualification_scope": "original idle native qualification observation; same root/GPU/Runtime/source"}


def seed_terminal_cache(bundle, output, host, snapshot_path, *, checker=None):
    """Exclusively cache a fully verified original root; no SSH or overwrite."""
    bundle, output, path = Path(bundle).resolve(), Path(output).resolve(), Path(snapshot_path).resolve()
    c, checked = (checker or base.check_bundle)(bundle)
    require(checked["contract_sha256"] == canonical(c), "Cache checker/contract mismatch")
    base.require(host in c["hosts"], "Unknown cache host")
    require(path.is_relative_to(bundle / "control" / "status" / host), "Terminal cache snapshot outside original host status root")
    snapshot = base.read(path)
    bridge = qualification_bridge(bundle, c, host, snapshot)
    report = training_status(snapshot, c, host)
    require(report["status"] == TERMINAL, "Terminal cache requires complete verified original root")
    item = {"schema": "titantpp_components_terminal_cache_v2", "host": host,
        "contract_sha256": checked["contract_sha256"], "root": c["hosts"][host]["root"],
        "source_files_sha256": c["source"]["files_sha256"], "gpu_uuid": c["hosts"][host]["gpu_uuid"],
        "snapshot_path": str(path), "snapshot_raw_sha256": base.sha(path),
        "original_observed_unix": snapshot["observed_unix"], "qualification_bridge": bridge,
        "scope": "past terminal state of this owned root; not current global GPU idle"}
    cache = output / "terminal_cache" / (host + ".json")
    if cache.exists():
        require(base.read(cache) == item, "Existing terminal cache differs; preserve original")
    else:
        base.write_exclusive(cache, item)
    return cache


def verified_terminal(bundle, output, c, host):
    """Reverify immutable cache, retaining original time even after lease expiry."""
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    cache = output / "terminal_cache" / (host + ".json")
    if not cache.exists():
        return None
    item = base.read(cache)
    spec = c["hosts"][host]
    require(item["schema"] == "titantpp_components_terminal_cache_v2" and item["host"] == host
        and item["contract_sha256"] == canonical(c) and item["root"] == spec["root"]
        and item["source_files_sha256"] == c["source"]["files_sha256"] and item["gpu_uuid"] == spec["gpu_uuid"],
        "Terminal cache identity changed; SSH0")
    path = Path(item["snapshot_path"])
    require(path.is_relative_to(bundle / "control" / "status" / host)
        and base.sha(path) == item["snapshot_raw_sha256"], "Terminal cache raw snapshot SHA changed; SSH0")
    snapshot = base.read(path)
    require(snapshot["observed_unix"] == item["original_observed_unix"], "Terminal cache original observation time changed")
    bridge = qualification_bridge(bundle, c, host, snapshot)
    require(bridge == item["qualification_bridge"], "Terminal cache qualification/permit proof changed; SSH0")
    report = training_status(snapshot, c, host)
    require(report["status"] == TERMINAL, "Terminal cache is not fully verified; SSH0")
    return path, snapshot, report
