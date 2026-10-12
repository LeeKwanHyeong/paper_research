"""One-shot native phase wrapper for the prepared, user-held dual-host study.

This module does nothing at import. Qualification and training have separate
exclusive phase claims and actual subprocess exit receipts. No automatic retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def gpu_pids(uuid):
    output = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=15).stdout
    result = []
    for line in output.splitlines():
        parts = [x.strip() for x in line.split(",")]
        require(len(parts) == 2, "Unknown GPU process inventory")
        if parts[0] == uuid:
            result.append(int(parts[1]))
    return result


def run_phase(c, host, phase):
    spec = c["hosts"][host]
    root = Path(spec["root"])
    require(Path(__file__).resolve() == root / "operation/pipeline.py", "Foreign wrapper root")
    require(Path.cwd().resolve() == root / "source", "Foreign source cwd")
    require(Path(sys.executable).resolve() == Path(spec["python"]).resolve(), "Foreign Python")
    require(all(os.environ.get(k) == v for k, v in spec["environment"].items()), "Environment drift")
    approval = read(root / "approval.json")
    require(approval["approved"] is True and approval["contract_sha256"] == canonical(c)
        and set(approval["hosts"]) == set(c["hosts"])
        and approval.get("explicit_new_user_instruction") is True,
        "Explicit dual-host approval missing")
    require(not (root / "failure.json").exists(), "Preserved failure; no automatic retry")
    require(not gpu_pids(spec["gpu_uuid"]), "Fixed GPU is occupied before phase")
    argv = [spec["python"], str(root / "operation/campaign.py"), "--contract",
        str(root / "execution_contract.json"), "--host", host, "--mode",
        "qualify" if phase == "qualify" else "dispatch"]
    started = time.time()
    write(root / "phases" / (phase + "_intent.json"), {"phase": phase,
        "contract_sha256": canonical(c), "host": host, "argv": argv,
        "wrapper_pid": os.getpid(), "wrapper_ppid": os.getppid(), "started_unix": started})
    log = root / "logs" / (phase + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    child = None
    try:
        with log.open("xb") as stream:
            child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                env=os.environ.copy(), start_new_session=True)
            write(root / "phases" / (phase + "_launch.json"), {"phase": phase,
                "argv": argv, "pid": child.pid, "ppid": os.getpid(),
                "contract_sha256": canonical(c), "started_unix": started})
            code = child.wait()
        write(root / "phases" / (phase + "_exit.json"), {"phase": phase, "argv": argv,
            "pid": child.pid, "ppid": os.getpid(), "os_exit_code": code,
            "contract_sha256": canonical(c), "completed_unix": time.time()})
        require(code == 0, "Native phase failed; no automatic retry")
        require(not gpu_pids(spec["gpu_uuid"]), "GPU processes remain after native phase")
        if phase == "qualify":
            q = read(root / "qualification/receipt.json")
            require(q["status"] == "passed" and q["contract_sha256"] == canonical(c), "Qualification receipt mismatch")
        else:
            for job in (j for j in c["jobs"] if j["host"] == host):
                terminal = read(root / "run" / job["id"] / "terminal_manifest.json")
                require(terminal["status"] == "complete" and terminal["scientific_success"] is True
                    and terminal["contract_sha256"] == canonical(c) and terminal["job"] == job,
                    "Terminal job mismatch")
        write(root / "phases" / (phase + "_complete.json"), {"phase": phase,
            "host": host, "contract_sha256": canonical(c), "subprocess_os_exit_code": 0,
            "gpu_compute_pids": [], "completed_unix": time.time(),
            "pipeline_os_exit_directly_observed": False})
        return 0
    except BaseException as exc:
        if child is not None and child.poll() is None:
            require(os.getpgid(child.pid) == child.pid, "Owned phase process group changed")
            os.killpg(child.pid, signal.SIGTERM)
            try:
                # The campaign owns up to three separate timeout sessions.
                # Allow their bounded cleanup before escalating its group.
                child.wait(timeout=75)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=15)
        write(root / "phases" / (phase + "_failure.json"), {"phase": phase,
            "contract_sha256": canonical(c), "exception": type(exc).__name__,
            "message": str(exc), "traceback": traceback.format_exc(),
            "child_pid": child.pid if child else None, "retry": False})
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--host", choices=("5080", "5090"), required=True)
    p.add_argument("--phase", choices=("qualify", "train"), required=True)
    a = p.parse_args()
    c = read(a.contract)
    require(a.contract.resolve() == Path(c["hosts"][a.host]["root"]) / "execution_contract.json", "Foreign contract path")
    previous = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise InterruptedError("Owned phase received SIGTERM")
    signal.signal(signal.SIGTERM, terminated)
    try:
        return run_phase(c, a.host, a.phase)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
