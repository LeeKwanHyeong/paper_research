#!/usr/bin/env python3
"""Run the approved immutable CNN/GRU continuations once, one fit per host.

Importing this standard-library supervisor performs no remote or GPU work.
Only its own newly spawned process group may be stopped by timeout/signal.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SCHEMA = "titantpp_cnn_gru_resume_dual_v1"
VARIANT = "count_only_log_regression"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    else:
        temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        os.replace(temporary, path)


def confined(root, name):
    path = (Path(root) / name).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Path escaped campaign root")
    return path


def permits(contract_path, host, *, now=None):
    contract_path = Path(contract_path).resolve()
    c = read(contract_path)
    require(c.get("schema") == SCHEMA and host in ("5080", "5090"), "Foreign continuation/host")
    spec = c["hosts"][host]
    root = Path(spec["root"]).resolve()
    require(contract_path == root / "execution_contract.json", "Wrong dedicated contract root")
    require(c["evaluation_scope"] == "validation_only" and c["held_out_test_evaluated"] is False,
        "Test evaluation is forbidden")
    limits = c["limits"]
    require(limits["workers_per_host"] == 1 and limits["automatic_retry"] is False
        and limits["manual_resume_once"] is True, "Only once-only single-worker continuation is admitted")
    require(0 < limits["per_condition_seconds"] <= 129600
        and limits["per_condition_seconds"] <= limits["total_wall_seconds"] <= 604800,
        "Resource ceilings enlarged")
    digest = canonical(c)
    approval, start = read(root / "approval.json"), read(root / "start_permit.json")
    require(approval.get("approved") is True and approval.get("manual_resume_once") is True
        and approval["contract_sha256"] == digest
        and approval["parent_contract_sha256"] == c["parent"]["contract_sha256"]
        and set(approval["hosts"]) == set(c["hosts"])
        and set(approval["job_ids"]) == {j["id"] for j in c["jobs"]}
        and bool(approval.get("user_instruction", "").strip()), "Missing/foreign explicit approval")
    require(start["contract_sha256"] == digest and start["approval_sha256"] == canonical(approval),
        "Start permit does not bind approval")
    now = time.time() if now is None else now
    require(start["started_at_unix"] <= now < start["deadline_unix"]
        and 0 < start["deadline_unix"] - start["started_at_unix"] <= limits["total_wall_seconds"],
        "Absolute campaign deadline exhausted/enlarged")
    jobs = [j for j in c["jobs"] if j["host"] == host]
    require(jobs and len({j["id"] for j in jobs}) == len(jobs), "Empty/duplicate assigned jobs")
    require(all(j["id"] == f"{j['dataset']}__42__{j['arm']}" and j["seed"] == 42 for j in jobs),
        "Foreign job identity")
    permit = read(root / ("training_permit_" + host + ".json"))
    ids = {j["id"] for j in jobs}
    require(permit["host"] == host and permit["contract_sha256"] == digest
        and permit["start_permit_sha256"] == canonical(start)
        and permit["manual_resume_once"] is True and permit["automatic_retry"] is False
        and set(permit["job_ids"]) == set(permit["qualification_files"]) == ids,
        "Training permit does not bind assigned jobs")
    for job in jobs:
        path = confined(root, "qualification/" + job["id"] + "/receipt.json")
        require(sha(path) == permit["qualification_files"][job["id"]], "Qualification receipt changed")
        q = read(path)
        require(q.get("status") == "passed" and q["host"] == host and q["job"] == job
            and q["contract_sha256"] == digest and q["full_validation_replay"] is True
            and q["held_out_test_evaluated"] is False, "Missing native checkpoint qualification")
    adapter = confined(root, "operation/titantpp_cnn_gru_resume_runtime.py")
    require(sha(adapter) == c["operation_adapter_sha256"], "Resume adapter changed")
    if "operation_supervisor_sha256" in c:
        require(sha(__file__) == c["operation_supervisor_sha256"], "Supervisor changed")
    return c, root, jobs, start


def fit_argv(c, root, host, job):
    return [c["hosts"][host]["python"], str(root / "operation/titantpp_cnn_gru_resume_runtime.py"),
        "--contract", str(root / "execution_contract.json"), "--host", host,
        "--job", job["id"], "--mode", "fit"]


def terminal(root, c, job):
    folder = confined(root, "run/" + job["id"])
    manifest = read(folder / "terminal_manifest.json")
    require(manifest["status"] == "complete" and manifest["scientific_success"] is True
        and manifest["job"] == job and manifest["contract_sha256"] == canonical(c)
        and manifest["held_out_test_evaluated"] is False, "Missing/foreign scientific terminal")
    require(manifest.get("files"), "Empty terminal manifest")
    for name, digest in manifest["files"].items():
        require(sha(confined(folder, name)) == digest, "Terminal original SHA differs: " + name)
    endpoint = read(folder / "runs" / job["arm"] / VARIANT / "seed_42/endpoint_replays.json")
    require(endpoint["status"] == "complete" and endpoint["evaluation_scope"] == "validation_only"
        and endpoint["held_out_test_evaluated"] is False
        and set(endpoint["replay_checks"]) == {"selected", "last"}
        and all(set(row) == {"qty_rmse", "qty_mae", "time_nll"} for row in endpoint["replay_checks"].values())
        and all(value is True for row in endpoint["replay_checks"].values() for value in row.values()),
        "Selected/last Validation replay is not verified")
    return {"status": "complete", "scientific_success": True,
        "terminal_manifest_sha256": sha(folder / "terminal_manifest.json"),
        "best_epoch": endpoint["best_epoch"], "completed_epochs": endpoint["completed_epochs"],
        "updated_unix": time.time()}


def stop_owned(child):
    """The process group belongs exclusively to a Popen started here."""
    if child is None or child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=15)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=15)


def supervise(contract_path, host):
    c, root, jobs, start = permits(contract_path, host)
    require(Path(sys.executable).resolve() == Path(c["hosts"][host]["python"]).resolve(),
        "Wrong supervisor Python")
    require(Path.cwd().resolve() == Path(c["hosts"][host]["source_root"]).resolve(), "Wrong frozen source cwd")
    require(all(os.environ.get(k) == v for k, v in c["hosts"][host]["environment"].items()),
        "Supervisor environment drift")
    digest, outcomes, child, active, started = canonical(c), {}, None, None, time.time()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(root / "supervisor.claim", {"pid": os.getpid(), "host": host, "contract_sha256": digest,
            "started_unix": started, "argv": sys.argv, "automatic_retry": False}, exclusive=True)
        write(root / "supervisor_started.json", {"pid": os.getpid(), "host": host,
            "contract_sha256": digest, "command": [sys.executable, *sys.argv],
            "started_unix": started, "deadline_unix": start["deadline_unix"],
            "starts_training": True, "workers_per_host": 1}, exclusive=True)

        def record(state, pending):
            write(root / "supervisor_status.json", {"status": state, "host": host,
                "contract_sha256": digest, "supervisor_pid": os.getpid(), "active_job": active,
                "worker_pid": child.pid if child is not None and child.poll() is None else None,
                "worker_command": fit_argv(c, root, host, active) if active else None,
                "outcomes": outcomes, "pending_jobs": pending, "concurrency": 1,
                "deadline_unix": start["deadline_unix"], "updated_unix": time.time(),
                "automatic_retry": False, "mac_required_for_training": False})

        def interrupted(signum, _frame):
            raise InterruptedError("Owned supervisor interrupted by signal " + str(signum))

        previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            record("running", jobs)
            for index, job in enumerate(jobs):
                active = job
                latest, checked_root, _, checked_start = permits(contract_path, host)
                require(latest == c and checked_root == root and checked_start == start, "Live contract/permit changed")
                require(not (root / "claims" / (job["id"] + ".json")).exists()
                    and not (root / "run" / job["id"]).exists(), "Condition already claimed; retry forbidden")
                limit = min(start["deadline_unix"], time.time() + c["limits"]["per_condition_seconds"])
                require(limit > time.time(), "No remaining fit time")
                env = {**os.environ, **c["hosts"][host]["environment"]}
                log = root / "logs" / (job["id"] + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                argv = fit_argv(c, root, host, job)
                with log.open("x", encoding="utf-8") as stream:
                    child = subprocess.Popen(argv, cwd=c["hosts"][host]["source_root"], env=env,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    write(root / "worker_started" / (job["id"] + ".json"), {"pid": child.pid,
                        "supervisor_pid": os.getpid(), "job": job, "contract_sha256": digest,
                        "command": argv, "started_unix": time.time(), "deadline_unix": limit}, exclusive=True)
                    record("running", jobs[index + 1:])
                    try:
                        code = child.wait(timeout=max(.001, limit - time.time()))
                    except subprocess.TimeoutExpired:
                        stop_owned(child)
                        raise TimeoutError("Owned fit absolute/per-condition deadline reached") from None
                require(code == 0, "Owned fit exited with code " + str(code))
                outcomes[job["id"]] = terminal(root, c, job)
                child, active = None, None
                record("running", jobs[index + 1:])
            write(root / "supervisor_terminal.json", {"status": "complete", "host": host,
                "contract_sha256": digest, "outcomes": outcomes, "completed_unix": time.time(),
                "all_assigned_jobs_terminal_verified": True, "automatic_retry": False}, exclusive=True)
            record("complete", [])
        except BaseException as error:
            stop_owned(child)
            write(root / "supervisor_failure.json", {"status": "failed", "host": host,
                "contract_sha256": digest, "type": type(error).__name__, "message": str(error),
                "traceback": traceback.format_exc(), "active_job": active, "outcomes": outcomes,
                "automatic_retry": False, "source_originals_preserved": True, "updated_unix": time.time()},
                exclusive=True)
            record("failed", [j for j in jobs if j["id"] not in outcomes and j != active])
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            stop_owned(child)
    return outcomes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    args = parser.parse_args(argv)
    result = supervise(args.contract, args.host)
    print(json.dumps({"host": args.host, "status": "complete", "conditions": len(result)}))


if __name__ == "__main__":
    main()
