"""One authorized replacement of the extra42 completion wait, never its training.

The original manager and failure records are immutable. This local control owns
one read-only SSH wait and delegates terminal verification/collection to the
preserved collector. It cannot launch, resume or signal remote scientific work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from paper.scripts import manage_titantpp_shared_external_extra42 as original
from paper.scripts import compare_titantpp_shared_external_five_seed as comparison

ROOT = original.ROOT
REPORT = original.REPORT / "recovery_v1"
BUNDLE = original.BUNDLE / "recovery_v1"
MODULE = "paper.scripts.recover_titantpp_shared_external_extra42"
IDENTITY_KEYS = ("pid", "start_ticks", "argv", "cwd", "pgid")
read, sha, require, write = original.read, original.sha, original.require, original.write


def verify(binding_sha, authorization_sha):
    """Validate every immutable binding before any connection or collection."""
    path = REPORT / "CONTROL_BINDINGS.json"
    require(sha(path) == binding_sha, "Recovery control binding SHA differs")
    b = read(path)
    require(b.get("recovery_root") == str(REPORT)
        and b.get("recovery_bundle_root") == str(BUNDLE), "Recovery path differs")
    files = b.get("files")
    require(isinstance(files, dict) and files, "Empty recovery file bindings")
    for name, digest in files.items():
        source = Path(name)
        if not source.is_absolute():
            source = ROOT / source
        require(not source.is_symlink() and sha(source) == digest, "Recovery bound file changed: " + name)
    require(b.get("authorization_sha256") == authorization_sha
        and sha(REPORT / "AUTHORIZATION.json") == authorization_sha, "Recovery approval SHA differs")
    a = read(REPORT / "AUTHORIZATION.json")
    require(a.get("status") == "approved" and a.get("recovery_only") is True
        and a.get("old_finisher_absent") is True, "Explicit replacement approval/old finisher absence missing")
    failure = original.REPORT / "QUEUE_FAILURE.json"
    require(sha(failure) == b["expected_original_failure_sha256"], "Original failure changed")
    require(read(failure).get("status") == "failed", "Original failure is missing")
    c = original.verify_local(b["original_integrity_sha256"])
    deadline = c["hosts"]["5080"]["absolute_host_deadline_unix"]
    require(type(b.get("deadline")) in (int, float) and math.isfinite(b["deadline"])
        and b["deadline"] == deadline, "Inherited deadline differs")
    identity = b.get("dispatcher_identity")
    require(isinstance(identity, dict) and set(identity) == set(IDENTITY_KEYS)
        and type(identity["pid"]) is int and type(identity["start_ticks"]) is int
        and isinstance(identity["argv"], list) and identity["argv"], "Original dispatcher identity missing")
    return b, c


def guard_new_attempt():
    names = ("START_INTENT.json", "STARTED.json", "FINISH_CLAIM.json", "WAIT_STARTED.json", "FAILURE.json", "COMPLETE.json")
    require(not any((REPORT / name).exists() for name in names), "Existing recovery attempt; no retry")
    require(not any(path.exists() for path in (original.BUNDLE / "retrieval",
        original.REPORT / "QUEUE_COMPLETE.json", original.REPORT / "NEW_SEED_RESULTS.json")),
        "Existing original retrieval/result; no overwrite or retry")


def start(binding_sha, authorization_sha):
    b, _ = verify(binding_sha, authorization_sha)
    guard_new_attempt()
    require(time.time() < b["deadline"], "Inherited deadline expired")
    write(REPORT / "START_INTENT.json", {"at": original.runtime.now(), "binding_sha256": binding_sha,
        "authorization_sha256": authorization_sha, "automatic_retry": False, "training_restart": False})
    command = [sys.executable, "-u", "-m", MODULE, "--mode", "finish", "--binding-sha", binding_sha,
        "--authorization-sha", authorization_sha]
    try:
        with (REPORT / "completion.stdout.log").open("xb") as out, (REPORT / "completion.stderr.log").open("xb") as err:
            child = subprocess.Popen(command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=out, stderr=err, start_new_session=True)
        value = {"status": "connection_requested_not_yet_verified", "pid": child.pid, "command": command,
            "at": original.runtime.now(), "binding_sha256": binding_sha, "automatic_retry": False,
            "own_os_exit_directly_observed": False}
        write(REPORT / "STARTED.json", value)
        return value
    except Exception as error:
        preserve_failure(error, "local_start")
        raise


def wait_program(c, b):
    """Only read identities/contract/exit and sleep; no scientific invocation."""
    h = c["hosts"]["5080"]
    return f'''import json,pathlib,time,sys
from paper.scripts import titantpp_shared_external_extra42_runtime as r
root=pathlib.Path({h['root']!r});deadline={b['deadline']!r};expected={b['dispatcher_identity']!r}
keys={IDENTITY_KEYS!r}
contract=r.read(root/'execution_contract.json')
r.require(r.sha_json(contract)=={original.canonical(c)!r},'Original native contract changed')
s=r.read(root/'start.json');l=r.read(root/'dispatcher_launch.json')
r.require(s['contract_sha256']==r.sha_json(contract),'Native start contract differs')
r.require(s['pid']==l['pid']==expected['pid'] and l['command']==expected['argv'],'Original dispatcher launch differs')
r.require(all(s['identity'][k]==l['identity'][k]==expected[k] for k in keys),'Original start identity differs')
def check_exit():
 x=r.read(root/'dispatcher_exit.json')
 r.require(x['pid']==expected['pid'] and x['start_ticks']==expected['start_ticks'] and x['command']==expected['argv'] and x['launch_sha256']==r.sha_file(root/'dispatcher_launch.json'),'Terminal dispatcher identity differs')
 r.require(x.get('actual_child_exit_observed') is True,'Actual dispatcher exit not observed')
 return x
if (root/'dispatcher_exit.json').exists():
 check_exit()
else:
 live=r.proc_identity(expected['pid'])
 r.require(live is not None and all(live[k]==expected[k] for k in keys),'Current dispatcher identity differs')
print(json.dumps({{'status':'READY','observed_at':r.now(),'contract_sha256':r.sha_json(contract),'dispatcher_identity':expected,'read_only_wait':True,'deadline':deadline}}),flush=True)
while not (root/'dispatcher_exit.json').exists():
 if time.time()>=deadline:raise TimeoutError('Inherited host deadline expired')
 time.sleep(min(30.,max(.01,deadline-time.time())))
x=check_exit();print(json.dumps({{'status':'TERMINAL','dispatcher_exit':x}}),flush=True)
sys.exit(0 if x['actual_os_returncode']==0 and x.get('timeout') is False else 1)
'''


def wait_connection(c, b):
    h = c["hosts"]["5080"]
    native = "cd " + shlex.quote(h["source_root"]) + " && " + shlex.join([h["python"], "-u", "-c", wait_program(c, b)])
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=6", "5080", native]
    folder = BUNDLE / "transport" / "5080"
    folder.mkdir(parents=True, exist_ok=True)
    outpath, errpath = folder / "wait.stdout", folder / "wait.stderr"
    write(REPORT / "WAIT_STARTED.json", {"at": original.runtime.now(), "command": command,
        "deadline": b["deadline"], "read_only_wait": True})
    timed_out = False
    with outpath.open("xb") as out, errpath.open("xb") as err:
        child = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err)
        try:
            code = child.wait(timeout=max(.01, b["deadline"] - time.time() + 60))
        except subprocess.TimeoutExpired:
            timed_out = True
            child.kill()  # Only this local SSH transport child, never remote processes.
            code = child.wait(timeout=15)
    value = {"observed_at": original.runtime.now(), "actual_transport_exit": code, "transport_timeout": timed_out,
        "local_transport_pid": child.pid, "stdout_path": str(outpath), "stdout_bytes": outpath.stat().st_size,
        "stdout_sha256": sha(outpath), "stderr_path": str(errpath), "stderr_bytes": errpath.stat().st_size,
        "stderr_sha256": sha(errpath), "remote_scientific_process_signaled": False}
    write(REPORT / "WAIT_TRANSPORT.json", value)
    require(code == 0 and not timed_out, "Replacement completion SSH failed; raw transport preserved")
    lines = [json.loads(line) for line in outpath.read_text().splitlines() if line.strip()]
    require(len(lines) == 2 and lines[0].get("status") == "READY" and lines[1].get("status") == "TERMINAL",
        "Ready/terminal wait evidence missing")
    ready, exited = lines[0], lines[1]["dispatcher_exit"]
    require(ready.get("contract_sha256") == original.canonical(c) and ready.get("dispatcher_identity") == b["dispatcher_identity"]
        and ready.get("read_only_wait") is True and ready.get("deadline") == b["deadline"], "Ready identity differs")
    require(exited.get("actual_os_returncode") == 0 and exited.get("timeout") is False
        and exited.get("actual_child_exit_observed") is True and exited.get("pid") == b["dispatcher_identity"]["pid"]
        and exited.get("start_ticks") == b["dispatcher_identity"]["start_ticks"]
        and exited.get("command") == b["dispatcher_identity"]["argv"], "Actual terminal exit0 missing")
    return value


@contextmanager
def recovery_transport_records():
    """Keep collector destinations original; only new connection logs move."""
    previous = original._transport.BUNDLE
    original._transport.BUNDLE = BUNDLE
    try:
        yield
    finally:
        original._transport.BUNDLE = previous


def preserve_failure(error, phase, *, collection_complete=False):
    path = REPORT / "FAILURE.json"
    if not path.exists():
        write(path, {"status": "failed", "at": original.runtime.now(), "phase": phase, "error": str(error),
            "collection_complete": collection_complete, "automatic_retry": False})


def finish(binding_sha, authorization_sha):
    phase, collected, owns_finish = "binding_validation", False, False
    try:
        b, c = verify(binding_sha, authorization_sha)
        intent = read(REPORT / "START_INTENT.json")
        require(intent["binding_sha256"] == binding_sha and intent["authorization_sha256"] == authorization_sha,
            "Recovery exclusive intent differs")
        require(not any((REPORT / p).exists() for p in ("FINISH_CLAIM.json", "WAIT_STARTED.json", "FAILURE.json", "COMPLETE.json")),
            "Existing replacement wait; no retry")
        require(not (original.BUNDLE / "retrieval").exists(), "Existing collection; no retry")
        write(REPORT / "FINISH_CLAIM.json", {"pid": os.getpid(), "at": original.runtime.now(),
            "binding_sha256": binding_sha, "authorization_sha256": authorization_sha})
        owns_finish = True
        phase = "completion_wait"
        transport = wait_connection(c, b)
        verify(binding_sha, authorization_sha)
        phase = "terminal_verification_and_collection"
        with recovery_transport_records():
            result = original.collect(c)
        require(result.get("status") == "complete", "Original collection did not complete")
        collected = True
        write(REPORT / "COLLECTION_COMPLETE.json", {**result, "wait_transport_sha256": sha(REPORT / "WAIT_TRANSPORT.json")})
        write(original.REPORT / "QUEUE_COMPLETE.json", {**result, "recovery_control": str(REPORT),
            "original_failure_preserved": True})
        phase = "five_seed_comparison"
        checked = comparison.run(ROOT, check_only=True)
        require(checked.get("status") == "verified_ready", "Five-seed verification is not ready")
        compared = comparison.run(ROOT)
        require(compared.get("status") == "complete", "Five-seed comparison did not complete")
        files = {name: sha(name) for name in compared["output_files"]}
        value = {"status": "complete", "at": original.runtime.now(), "collection": result, "comparison": compared,
            "comparison_files": files, "original_failure_sha256": b["expected_original_failure_sha256"],
            "binding_sha256": binding_sha, "training_restarted": False,
            "supervisor_own_exit_directly_observed": False, "wait_transport": transport}
        write(REPORT / "COMPLETE.json", value)
        return value
    except Exception as error:
        if owns_finish:
            preserve_failure(error, phase, collection_complete=collected)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("verify", "start", "finish"), required=True)
    p.add_argument("--binding-sha", required=True)
    p.add_argument("--authorization-sha", required=True)
    a = p.parse_args(argv)
    if a.mode == "verify":
        verify(a.binding_sha, a.authorization_sha)
        value = {"status": "verified", "remote_calls": 0}
    else:
        value = (start if a.mode == "start" else finish)(a.binding_sha, a.authorization_sha)
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
