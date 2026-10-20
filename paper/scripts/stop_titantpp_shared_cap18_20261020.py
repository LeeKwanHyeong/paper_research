"""One explicitly authorized stop of the isolated 5080 Cap18 campaign."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_mlp_shared_cap_20261019_v1_execution_v3_5080"
CONTRACT_SHA = "a9278125b8a8e01e56298d13ee74c0c628f779020c26950923998e04f05d5226"
LOCAL_CONTRACT = Path("search_artifacts/titantpp_history_mlp_shared_cap_20261019_v1/execution_v3/execution_contract.json")
REPORT = Path("reports/titantpp_shared_raf_taxi6_test_20261020_v1/stop_cap5080")
PROTECTED = {351034, 351048, 351049, 353905}
FINISHER = 16494
FINISHER_SUFFIX = "-u -m paper.scripts.manage_titantpp_shared_history_cap_execution_v3 --mode finish"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(pid, proc_root=Path("/proc")):
    base = Path(proc_root) / str(pid)
    try:
        fields = (base / "stat").read_text().rpartition(")")[2].split()
        raw = (base / "cmdline").read_bytes()
        if fields[0] == "Z":
            return None
        out = {"pid": pid, "start_ticks": int(fields[19]), "ppid": int(fields[1]),
               "pgid": int(fields[2]), "state": fields[0],
               "argv": [os.fsdecode(x) for x in raw.split(b"\0") if x],
               "cwd": str((base / "cwd").resolve(strict=True))}
        again = (base / "stat").read_text().rpartition(")")[2].split()
        if int(again[19]) != out["start_ticks"]:
            raise RuntimeError("PID reused during identity read")
        return out
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(expected, current):
    return current is not None and all(expected[k] == current[k]
        for k in ("pid", "start_ticks", "argv", "cwd", "pgid"))


def campaign_owned(value):
    if not value or value["pid"] in PROTECTED:
        return False
    argv = value["argv"]
    return (value["cwd"] == ROOT + "/source" and
            (ROOT + "/execution_contract.json" in argv or
             any("subprocess.Popen" in arg and ROOT in arg for arg in argv)))


def discover():
    values = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) in PROTECTED or int(p.name) == os.getpid():
            continue
        try:
            if os.fsencode(ROOT) not in (p / "cmdline").read_bytes():
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        live = identity(int(p.name))
        if campaign_owned(live):
            values.append(live)
    return values


def checked_signal(expected, sig, events):
    if not campaign_owned(expected):
        raise RuntimeError("Signal target is not this campaign")
    live = identity(expected["pid"])
    if live is None:
        events.append({"pid": expected["pid"], "already_exited": True})
        return
    if not same_process(expected, live):
        raise RuntimeError("PID identity changed; signal refused")
    os.kill(expected["pid"], sig)
    events.append({"pid": expected["pid"], "signal": signal.Signals(sig).name,
                   "identity": live, "sent_at": now()})


def gpu():
    cmd = ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]
    value = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
    free = subprocess.run(["nvidia-smi", "--query-gpu=uuid,memory.free,memory.used",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
    return {"processes_raw": value.stdout, "memory_raw": free.stdout,
            "compute_pids": [int(line.split(",")[0]) for line in value.stdout.splitlines() if line.strip()]}


def remote_stop():
    result = {"started_at": now(), "status": "in_progress", "reason": "explicit_user_stop",
              "scientific_failure": False, "root": ROOT, "signals": []}
    try:
        root = Path(ROOT)
        if sha(root / "execution_contract.json") != CONTRACT_SHA:
            raise RuntimeError("Contract SHA changed")
        contract = json.loads((root / "execution_contract.json").read_text())
        if contract["hosts"]["5080"]["root"] != ROOT:
            raise RuntimeError("Contract root mismatch")
        protected_before = [identity(pid) for pid in sorted(PROTECTED)]
        if any(v is None or v["state"] not in ("T", "t") for v in protected_before):
            raise RuntimeError("Pre-existing paused identities changed")
        for v in protected_before:
            expected = next(p for p in contract["hosts"]["5080"]["paused_processes"] if p["pid"] == v["pid"])
            if not same_process(expected, v):
                raise RuntimeError("Protected identity mismatch")
        result["protected_before"] = protected_before
        result["gpu_before"] = gpu()
        result["owned_before"] = discover()
        dispatcher = json.loads((root / "start.json").read_text())["pid"]
        live_dispatcher = next((v for v in result["owned_before"] if v["pid"] == dispatcher), None)
        if identity(dispatcher) is not None and live_dispatcher is None:
            raise RuntimeError("Live dispatcher ownership unconfirmed")
        # Stop the queue first, so no further jobs can start.
        if live_dispatcher:
            checked_signal(live_dispatcher, signal.SIGTERM, result["signals"])
        until = time.monotonic() + 5
        while identity(dispatcher) is not None and time.monotonic() < until:
            time.sleep(.1)
        if identity(dispatcher) is not None:
            raise RuntimeError("Dispatcher did not stop; no further action")
        # Parent supervisor is allowed to reap dispatcher and preserve real exit.
        time.sleep(.3)
        remaining = discover()
        workers = [v for v in remaining if ROOT + "/execution_contract.json" in v["argv"]]
        for value in workers:
            checked_signal(value, signal.SIGTERM, result["signals"])
        until = time.monotonic() + 15
        while discover() and time.monotonic() < until:
            time.sleep(.25)
        # Only verified same campaign processes may receive the bounded fallback.
        for value in discover():
            checked_signal(value, signal.SIGKILL, result["signals"])
        until = time.monotonic() + 5
        while discover() and time.monotonic() < until:
            time.sleep(.1)
        result["owned_after"] = discover()
        result["gpu_after"] = gpu()
        result["protected_after"] = [identity(pid) for pid in sorted(PROTECTED)]
        if result["owned_after"] or any(not same_process(a, b) or b["state"] not in ("T", "t")
                for a, b in zip(protected_before, result["protected_after"])):
            raise RuntimeError("Post-stop ownership/protected-process check failed")
        owned_pids = {x["pid"] for x in result["owned_before"] + remaining}
        if owned_pids.intersection(result["gpu_after"]["compute_pids"]):
            raise RuntimeError("Stopped campaign still owns a GPU context")
        result["checkpoint_manifest"] = {
            str(p.relative_to(root)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted((root / "run").rglob("*"))
            if p.is_file() and p.suffix in (".pt", ".pth", ".ckpt")}
        result["receipt_manifest"] = {str(p.relative_to(root)): sha(p)
            for p in sorted((root / "run").glob("*/*_receipt.json"))}
        result["dispatcher_exit_raw"] = (root / "dispatcher_exit.json").read_text() if (root / "dispatcher_exit.json").exists() else None
        result["checkpoint_files_deleted_or_rewritten_by_stop"] = False
        result["status"] = "stopped_verified"
        result["completed_at"] = now()
        result["wrapper_own_exit_directly_observed"] = False
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        result["status"] = "action_required"
        result["error"] = traceback.format_exc()
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1


def local_finisher_stop():
    def inspect():
        p = subprocess.run(["ps", "-p", str(FINISHER), "-o", "pid=,lstart=,command="], capture_output=True, text=True)
        return p.stdout.strip()
    before = inspect()
    if not before:
        return {"pid": FINISHER, "already_absent": True}
    if not before.endswith(FINISHER_SUFFIX) or "Mon Oct 19 22:24:11 2026" not in before:
        raise RuntimeError("Local finisher identity/start time mismatch")
    if inspect() != before:
        raise RuntimeError("Local finisher changed before signal")
    os.kill(FINISHER, signal.SIGTERM)
    until = time.monotonic() + 5
    while inspect() and time.monotonic() < until:
        time.sleep(.1)
    after = inspect()
    if after:
        raise RuntimeError("Local finisher remains")
    return {"pid": FINISHER, "identity_before": before, "signal": "SIGTERM", "absent_after": True,
            "actual_os_exit_directly_observed": False}


def main():
    if "--remote" in sys.argv:
        return remote_stop()
    if sha(LOCAL_CONTRACT) != CONTRACT_SHA:
        raise RuntimeError("Local contract SHA mismatch")
    contract = json.loads(LOCAL_CONTRACT.read_text())
    if contract["hosts"]["5080"]["root"] != ROOT:
        raise RuntimeError("Local root mismatch")
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "INTENT.json").open("x") as f:
        json.dump({"at": now(), "user_authorized_stop": True, "root": ROOT, "automatic_retry": False}, f)
    local = local_finisher_stop()
    (REPORT / "LOCAL_FINISHER.json").write_text(json.dumps(local, indent=2) + "\n")
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "5080", "/usr/bin/python3", "-", "--remote"]
    run = subprocess.run(command, input=Path(__file__).read_bytes(), capture_output=True, timeout=120)
    (REPORT / "remote.stdout").write_bytes(run.stdout)
    (REPORT / "remote.stderr").write_bytes(run.stderr)
    result = json.loads(run.stdout)
    (REPORT / "RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    receipt = {"actual_transport_exit": run.returncode, "local_finisher": local, "status": result["status"],
        "files": {p.name: sha(p) for p in REPORT.iterdir() if p.is_file()}, "observed_at": now()}
    (REPORT / "RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "actual_transport_exit": run.returncode,
                     "report": str(REPORT), "gpu_after": result.get("gpu_after")}))
    return run.returncode


if __name__ == "__main__":
    raise SystemExit(main())
