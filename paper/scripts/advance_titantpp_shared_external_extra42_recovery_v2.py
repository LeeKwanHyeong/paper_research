"""Short authorized completion checks for existing extra42 scientific work.

One advance opens at most one 45-second read-only status connection. The native
dispatcher remains responsible for Train/Validation/Test. Only verified terminal
exit0 permits the preserved collector, followed by local five-seed aggregation.
No background process, native launch, resume, signal or scientific retry exists.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from paper.scripts import manage_titantpp_shared_external_extra42 as original
from paper.scripts import compare_titantpp_shared_external_five_seed as comparison

ROOT = original.ROOT
REPORT = original.REPORT / "recovery_v2"
BUNDLE = original.BUNDLE / "recovery_v2"
IDENTITY_KEYS = ("pid", "start_ticks", "argv", "cwd", "pgid")
CHECK_TIMEOUT_SECONDS = 45
MINIMUM_CHECK_INTERVAL_SECONDS = 900
read, sha, require, write = original.read, original.sha, original.require, original.write


def verify(binding_sha, authorization_sha):
    path = REPORT / "CONTROL_BINDINGS.json"
    require(sha(path) == binding_sha, "Recovery v2 binding SHA differs")
    b = read(path)
    require(b.get("recovery_root") == str(REPORT) and b.get("recovery_bundle_root") == str(BUNDLE),
        "Recovery v2 path differs")
    files = b.get("files")
    require(isinstance(files, dict) and files, "Empty recovery file bindings")
    for name, digest in files.items():
        p = Path(name)
        p = p if p.is_absolute() else ROOT / p
        require(not p.is_symlink() and sha(p) == digest, "Recovery bound file changed: " + name)
    require(b.get("authorization_sha256") == authorization_sha
        and sha(REPORT / "AUTHORIZATION.json") == authorization_sha, "Recovery approval SHA differs")
    a = read(REPORT / "AUTHORIZATION.json")
    require(a.get("status") == "approved" and all(a.get(k) is True for k in
        ("recovery_only", "short_terminal_checks", "old_finisher_absent", "read_only_transport_recheck_authorized"))
        and type(a.get("maximum_consecutive_transport_errors")) is int
        and a["maximum_consecutive_transport_errors"] == 3, "Short-check replacement approval missing")
    for path, key in ((original.REPORT / "QUEUE_FAILURE.json", "expected_original_failure_sha256"),
                      (original.REPORT / "recovery_v1/FAILURE.json", "expected_previous_failure_sha256")):
        require(sha(path) == b[key] and read(path).get("status") == "failed", "Historical failure changed")
    c = original.verify_local(b["original_integrity_sha256"])
    require(type(b.get("deadline")) in (int, float) and math.isfinite(b["deadline"])
        and b["deadline"] == c["hosts"]["5080"]["absolute_host_deadline_unix"], "Inherited deadline differs")
    i = b.get("dispatcher_identity")
    require(isinstance(i, dict) and set(i) == set(IDENTITY_KEYS)
        and type(i["pid"]) is int and type(i["start_ticks"]) is int
        and isinstance(i["argv"], list) and i["argv"], "Original dispatcher identity missing")
    return b, c


@contextmanager
def advance_lock():
    """The lock covers checking, collection and comparison; a loser writes nothing."""
    with (REPORT / "advance.lock").open("a+") as stream:
        acquired = False
        try:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            yield acquired
        finally:
            if acquired:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def terminal_program(c, b):
    h = c["hosts"]["5080"]
    return f'''import json,pathlib,time
from paper.scripts import titantpp_shared_external_extra42_runtime as r
root=pathlib.Path({h['root']!r});expected={b['dispatcher_identity']!r};keys={IDENTITY_KEYS!r}
c=r.read(root/'execution_contract.json')
r.require(r.sha_json(c)=={original.canonical(c)!r},'Original native contract changed')
s=r.read(root/'start.json');l=r.read(root/'dispatcher_launch.json')
r.require(s['contract_sha256']==r.sha_json(c),'Original start contract differs')
r.require(s['pid']==l['pid']==expected['pid'] and l['command']==expected['argv'],'Original launch identity differs')
r.require(all(s['identity'][k]==l['identity'][k]==expected[k] for k in keys),'Original dispatcher start identity differs')
result={{'observed_at':r.now(),'contract_sha256':r.sha_json(c),'dispatcher_identity':expected,'read_only_check':True,'deadline':{b['deadline']!r}}}
if (root/'dispatcher_exit.json').exists():
 x=r.read(root/'dispatcher_exit.json')
 r.require(x['pid']==expected['pid'] and x['start_ticks']==expected['start_ticks'] and x['command']==expected['argv'] and x['launch_sha256']==r.sha_file(root/'dispatcher_launch.json'),'Terminal identity differs')
 r.require(x.get('actual_child_exit_observed') is True,'Actual dispatcher exit was not observed')
 result.update(status='terminal' if x['actual_os_returncode']==0 and x.get('timeout') is False else 'terminal_failure',dispatcher_exit=x)
else:
 live=r.proc_identity(expected['pid'])
 r.require(live is not None and all(live[k]==expected[k] for k in keys),'Current dispatcher identity differs')
 r.require(time.time()<{b['deadline']!r},'Inherited host deadline expired without terminal evidence')
 result.update(status='waiting',dispatcher_current_identity=live)
print(json.dumps(result),flush=True)
'''


def short_check(c, b, round_report, round_bundle):
    h = c["hosts"]["5080"]
    native = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join([h["python"], "-u", "-c", terminal_program(c, b)])
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5080", native]
    round_bundle.mkdir(parents=True, exist_ok=False)
    raw_out, raw_err, code, timed_out, transport_error = b"", b"", None, False, None
    try:
        response = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=CHECK_TIMEOUT_SECONDS)
        raw_out, raw_err, code = response.stdout, response.stderr, response.returncode
    except subprocess.TimeoutExpired as error:
        raw_out, raw_err, timed_out = error.stdout or b"", error.stderr or b"", True
        transport_error = str(error)
    except OSError as error:
        transport_error = str(error)
    out, err = round_bundle / "check.stdout", round_bundle / "check.stderr"
    out.write_bytes(raw_out); err.write_bytes(raw_err)
    record = {"at": original.runtime.now(), "command": command, "timeout_seconds": CHECK_TIMEOUT_SECONDS,
        "actual_transport_exit": code, "transport_timeout": timed_out, "transport_error": transport_error,
        "stdout_path": str(out), "stdout_bytes": len(raw_out), "stdout_sha256": sha(out),
        "stderr_path": str(err), "stderr_bytes": len(raw_err), "stderr_sha256": sha(err),
        "remote_scientific_process_signaled": False}
    write(round_report / "TRANSPORT.json", record)
    if code == 255 or timed_out:
        return {"status": "transport_error", "observed_at": record["at"], "actual_transport_exit": code,
            "transport_timeout": timed_out, "automatic_in_round_retry": False}
    require(code == 0 and not timed_out and transport_error is None, "Short terminal SSH failed; raw transport preserved")
    lines = [line for line in raw_out.decode("utf8").splitlines() if line.strip()]
    require(len(lines) == 1, "Expected one terminal-check JSON response")
    value = json.loads(lines[0])
    require(value.get("contract_sha256") == original.canonical(c) and value.get("dispatcher_identity") == b["dispatcher_identity"]
        and value.get("read_only_check") is True and value.get("deadline") == b["deadline"], "Terminal-check identity differs")
    require(isinstance(value.get("observed_at"), str) and value["observed_at"], "Actual observation time missing")
    require(value.get("status") in ("waiting", "terminal", "terminal_failure"), "Unknown terminal-check status")
    if value["status"] == "waiting":
        current = value.get("dispatcher_current_identity")
        require(isinstance(current, dict) and all(current.get(k) == b["dispatcher_identity"][k] for k in IDENTITY_KEYS),
            "Current dispatcher identity missing")
    else:
        e = value.get("dispatcher_exit", {})
        require(e.get("pid") == b["dispatcher_identity"]["pid"] and e.get("start_ticks") == b["dispatcher_identity"]["start_ticks"]
            and e.get("command") == b["dispatcher_identity"]["argv"] and e.get("actual_child_exit_observed") is True,
            "Actual terminal exit identity missing")
        if value["status"] == "terminal":
            require(e.get("actual_os_returncode") == 0 and e.get("timeout") is False, "Terminal exit0 missing")
    return value


def publish_round(folder, value):
    intent = read(folder / "INTENT.json")
    value = {**value, "round_number": intent["round_number"], "check_started_unix": intent["check_started_unix"]}
    write(folder / "ROUND.json", value)
    pointer = {"round_path": str(folder / "ROUND.json"), "round_sha256": sha(folder / "ROUND.json"),
        "status": value["status"], "observed_at": value.get("observed_at")}
    temporary = REPORT / (".LATEST." + uuid.uuid4().hex + ".json")
    write(temporary, pointer)
    os.replace(temporary, REPORT / "LATEST.json")
    return value


def preserve_failure(error, phase, folder=None, *, collection_complete=False):
    value = {"status": "failed", "at": original.runtime.now(), "phase": phase, "error": str(error),
        "collection_complete": collection_complete, "automatic_retry": False}
    if folder is not None:
        value["round_path"] = str(folder)
        transport = folder / "TRANSPORT.json"
        if transport.exists():
            value["transport_path"], value["transport_sha256"] = str(transport), sha(transport)
    if phase == "transport_error_limit":
        value.update(consecutive_transport_errors=3, maximum_consecutive_transport_errors=3)
    if not (REPORT / "FAILURE.json").exists():
        write(REPORT / "FAILURE.json", value)
    return value


@contextmanager
def collector_transport_records():
    previous = original._transport.BUNDLE
    original._transport.BUNDLE = BUNDLE
    try:
        yield
    finally:
        original._transport.BUNDLE = previous


def completion_files(compared):
    fixed = [original.BUNDLE / "retrieval/final_receipt.json", original.BUNDLE / "retrieval/5080/receipt.json",
        original.BUNDLE / "retrieval/5080/original.tar.gz", original.BUNDLE / "retrieval/5080/manifest.json",
        original.REPORT / "NEW_SEED_RESULTS.json", original.REPORT / "QUEUE_COMPLETE.json",
        REPORT / "COLLECT_INTENT.json", REPORT / "COLLECTION_COMPLETE.json"]
    return {str(p): sha(p) for p in (*fixed, *(Path(x) for x in compared["output_files"]))}


def verify_complete_cache(b, c, binding_sha):
    complete = read(REPORT / "COMPLETE.json")
    require(complete.get("status") == "complete" and complete.get("binding_sha256") == binding_sha
        and complete.get("original_failure_sha256") == b["expected_original_failure_sha256"]
        and complete.get("previous_failure_sha256") == b["expected_previous_failure_sha256"], "Completion lineage differs")
    compared = complete["comparison"]
    require(compared.get("status") == "complete" and len(compared.get("output_files", [])) == 2,
        "Five-seed completion output missing")
    expected = completion_files(compared)
    require(complete.get("files") == expected and expected, "Completion archive/report SHA differs")
    original.verify_collected(c, original.BUNDLE / "retrieval/5080/originals")
    checked = comparison.run(ROOT, check_only=True)
    require(checked.get("status") == "verified_ready", "Completed five-seed inputs no longer verify")
    return {"status": "complete", "cached": True, "remote_calls": 0,
        "complete_path": str(REPORT / "COMPLETE.json"), "complete_sha256": sha(REPORT / "COMPLETE.json")}


def guard_partial_work():
    require(not any(p.exists() for p in (REPORT / "COLLECT_INTENT.json", original.BUNDLE / "retrieval",
        original.REPORT / "QUEUE_COMPLETE.json", original.REPORT / "NEW_SEED_RESULTS.json")),
        "Existing collection attempt/partial retrieval; no retry")
    previous = REPORT / "rounds"
    if previous.exists():
        rounds = []
        for folder in previous.iterdir():
            require(folder.is_dir() and (folder / "ROUND.json").exists(), "Incomplete prior check round; no retry")
            rounds.append((folder, read(folder / "ROUND.json")))
        if rounds:
            require({value.get("round_number") for _, value in rounds} == set(range(1, len(rounds) + 1)),
                "Round publication sequence differs")
            require(latest_round().get("round_number") == len(rounds), "Unpublished prior round; no retry")


def latest_round():
    pointer = read(REPORT / "LATEST.json")
    path = Path(pointer["round_path"])
    require(path.resolve().is_relative_to((REPORT / "rounds").resolve())
        and sha(path) == pointer["round_sha256"], "Latest round pointer/SHA differs")
    value = read(path)
    require(value["status"] == pointer["status"], "Latest round status differs")
    return value


def advance(binding_sha, authorization_sha):
    b, c = verify(binding_sha, authorization_sha)
    with advance_lock() as acquired:
        if not acquired:
            return {"status": "busy", "remote_calls": 0, "failure_record_created": False}
        if (REPORT / "FAILURE.json").exists():
            return {"status": "blocked_failure", "remote_calls": 0, "failure_path": str(REPORT / "FAILURE.json"),
                "failure_sha256": sha(REPORT / "FAILURE.json")}
        if (REPORT / "COMPLETE.json").exists():
            return verify_complete_cache(b, c, binding_sha)
        folder, phase, collected = None, "existing_work_guard", False
        try:
            guard_partial_work()
            previous = latest_round() if (REPORT / "LATEST.json").exists() else None
            if previous is not None:
                started = previous.get("check_started_unix")
                require(type(started) in (int, float) and math.isfinite(started), "Previous check start time missing")
                age = time.time() - started
                if age < MINIMUM_CHECK_INTERVAL_SECONDS:
                    return {"status": "waiting_next_check", "remote_calls": 0, "previous_status": previous["status"],
                        "remaining_seconds": MINIMUM_CHECK_INTERVAL_SECONDS - age,
                        "minimum_check_interval_seconds": MINIMUM_CHECK_INTERVAL_SECONDS}
            stamp = original.runtime.now().replace(":", "").replace("+", "").replace("-", "").replace(".", "")
            round_id = stamp + "_" + uuid.uuid4().hex[:12]
            folder = REPORT / "rounds" / round_id
            folder.mkdir(parents=True, exist_ok=False)
            write(folder / "INTENT.json", {"at": original.runtime.now(), "binding_sha256": binding_sha,
                "authorization_sha256": authorization_sha, "read_only_check": True,
                "round_number": 1 if previous is None else previous["round_number"] + 1,
                "check_started_unix": time.time()})
            phase = "short_terminal_check"
            observed = short_check(c, b, folder, BUNDLE / "transport/5080" / round_id)
            if observed["status"] == "transport_error":
                old = latest_round() if (REPORT / "LATEST.json").exists() else {}
                previous = old.get("consecutive_transport_errors", 0) if old.get("status") == "transport_error" else 0
                require(type(previous) is int and 0 <= previous < 3, "Transport error counter differs")
                count = previous + 1
                if count == 3:
                    phase = "transport_error_limit"
                    raise ValueError("Three consecutive short-check transport errors; user action required")
                return publish_round(folder, {**observed, "remote_calls": 1, "collection_calls": 0,
                    "consecutive_transport_errors": count, "maximum_consecutive_transport_errors": 3,
                    "transport_path": str(folder / "TRANSPORT.json"), "transport_sha256": sha(folder / "TRANSPORT.json"),
                    "next_action": "next_authorized_periodic_check", "training_restarted": False})
            if observed["status"] == "waiting":
                return publish_round(folder, {"status": "waiting", "observed_at": observed["observed_at"],
                    "remote_calls": 1, "collection_calls": 0, "observation": observed,
                    "consecutive_transport_errors": 0,
                    "transport_path": str(folder / "TRANSPORT.json"), "transport_sha256": sha(folder / "TRANSPORT.json"),
                    "next_action": "next_authorized_periodic_check", "training_restarted": False})
            phase = "terminal_scientific_gate"
            require(observed["status"] == "terminal", "Dispatcher scientific exit failed; no collection")
            verify(binding_sha, authorization_sha)
            write(REPORT / "COLLECT_INTENT.json", {"at": original.runtime.now(), "round_path": str(folder),
                "terminal_observation": observed, "binding_sha256": binding_sha, "automatic_retry": False})
            phase = "terminal_verification_and_collection"
            with collector_transport_records():
                result = original.collect(c)
            require(result.get("status") == "complete", "Preserved collector did not complete")
            collected = True
            write(REPORT / "COLLECTION_COMPLETE.json", result)
            write(original.REPORT / "QUEUE_COMPLETE.json", {**result, "recovery_control": str(REPORT),
                "historical_failures_preserved": True})
            phase = "five_seed_comparison"
            require(comparison.run(ROOT, check_only=True).get("status") == "verified_ready", "Five-seed inputs not ready")
            compared = comparison.run(ROOT)
            require(compared.get("status") == "complete", "Five-seed comparison did not complete")
            complete = {"status": "complete", "at": original.runtime.now(), "collection": result, "comparison": compared,
                "binding_sha256": binding_sha, "original_failure_sha256": b["expected_original_failure_sha256"],
                "previous_failure_sha256": b["expected_previous_failure_sha256"], "files": completion_files(compared),
                "training_restarted": False, "supervisor_own_exit_directly_observed": False}
            write(REPORT / "COMPLETE.json", complete)
            return publish_round(folder, {"status": "complete", "observed_at": observed["observed_at"], "remote_calls": 2,
                "collection_calls": 1, "complete_path": str(REPORT / "COMPLETE.json"),
                "complete_sha256": sha(REPORT / "COMPLETE.json"), "training_restarted": False})
        except Exception as error:
            failure = preserve_failure(error, phase, folder, collection_complete=collected)
            if folder is not None and not (folder / "ROUND.json").exists():
                publish_round(folder, {**failure, "failure_path": str(REPORT / "FAILURE.json"),
                    "failure_sha256": sha(REPORT / "FAILURE.json")})
            raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("verify", "advance"), required=True)
    p.add_argument("--binding-sha", required=True)
    p.add_argument("--authorization-sha", required=True)
    a = p.parse_args(argv)
    try:
        if a.mode == "verify":
            verify(a.binding_sha, a.authorization_sha)
            value = {"status": "verified", "remote_calls": 0}
        else:
            value = advance(a.binding_sha, a.authorization_sha)
    except Exception as error:
        print(json.dumps({"status": "failed", "error": str(error), "automatic_retry": False}, ensure_ascii=False), flush=True)
        return 1
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
