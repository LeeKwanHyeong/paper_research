"""Local-only successor queue fixing macOS Python argv display aliases.

The sealed training contract/archive/worker and failed first queue are preserved.
All remote attempt/claim/collection paths still use the same exclusive bundle.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

_PATH = Path(__file__).with_name("manage_titantpp_shared_history_raw9.py")
_SHA = "d387fd00d0ebd1b026e7524577e1add45c1ddf85482f44d85db878650a91a8c4"
if hashlib.sha256(_PATH.read_bytes()).hexdigest() != _SHA:
    raise ValueError("Preserved raw9 manager changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_manager", _PATH)
_manager = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manager)
BASE_REPORT = _manager.REPORT
REPORT = BASE_REPORT / "queue_v2"
EXPECTED_TAIL = ["-u", "-m", "paper.scripts.manage_titantpp_shared_history_cap_execution_v3", "--mode", "finish"]


def executable_matches(expected, actual):
    expected, actual = Path(expected).resolve(), Path(actual).resolve()
    if expected != Path(sys.executable).resolve():
        return False
    if expected == actual:
        return True
    framework = Path(sys.base_prefix).resolve()
    return (sys.platform == "darwin" and "Python.framework/Versions/3.12" in str(framework)
        and expected.is_relative_to(framework)
        and actual == framework / "Resources/Python.app/Contents/MacOS/Python")


def inspect_finisher(path):
    marker = _manager.read(path)
    expected = marker.get("command", [])
    _manager.require(type(marker.get("pid")) is int and marker["pid"] > 0
        and len(expected) == 6 and expected[1:] == EXPECTED_TAIL, "Unexpected Cap18 finisher identity")
    process = subprocess.run(["/bin/ps", "-p", str(marker["pid"]), "-o", "lstart=,command="],
        capture_output=True, text=True, timeout=20, env={**os.environ, "LC_ALL": "C", "TZ": "UTC"})
    if process.returncode != 0:
        return {"alive": False, "pid": marker["pid"], "actual_ps_exit": process.returncode,
            "stdout": process.stdout, "stderr": process.stderr}
    fields = shlex.split(process.stdout.strip())
    _manager.require(len(fields) == 11, "Unrecognized finisher process command")
    started = datetime.strptime(" ".join(fields[:5]), "%a %b %d %H:%M:%S %Y").replace(tzinfo=timezone.utc)
    recorded = datetime.fromisoformat(marker["started_at"]).astimezone(timezone.utc)
    argv = fields[5:]
    matched = (argv[1:] == expected[1:] and executable_matches(expected[0], argv[0])
        and abs((started - recorded).total_seconds()) <= 5.)
    return {"alive": matched, "pid": marker["pid"], "actual_ps_exit": process.returncode,
        "observed_process_started_at_utc": started.isoformat(), "recorded_started_at": marker["started_at"],
        "actual_argv": argv, "expected_argv": expected, "stdout": process.stdout, "stderr": process.stderr}


def _finisher_alive(path):
    return inspect_finisher(path)["alive"]


_manager.REPORT = _manager._impl.REPORT = REPORT
_manager._finisher_alive = _finisher_alive


def start_queue():
    contract = _manager.verify_local(archive=True)
    old = _manager.read(BASE_REPORT / "QUEUE_FAILURE.json")
    _manager.require(old.get("error") == "Cap18 finite finisher stopped before verified completion",
        "Different unresolved failure; do not repeat")
    _manager.require(not (_manager.BUNDLE / "attempts/5080.json").exists()
        and not (_manager.BUNDLE / "retrieval/5080").exists()
        and not (BASE_REPORT / "LAUNCHED_5080.json").exists(), "A scientific/remote attempt already exists")
    _manager.require(not (REPORT / "QUEUE_STARTED.json").exists()
        and not (REPORT / "WAIT_STARTED.json").exists(), "Version2 queue already started")
    if not Path(contract["predecessor"]["receipt_path"]).exists():
        identity = inspect_finisher(contract["predecessor"]["finisher_receipt_path"])
        _manager.require(identity["alive"], "Cap18 actual finisher identity not confirmed")
        _manager.write(REPORT / "PREDECESSOR_IDENTITY.json", identity)
    command = [sys.executable, "-u", "-m", "paper.scripts.manage_titantpp_shared_history_raw9_queue_v2", "--mode", "wait-predecessor"]
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / "queue.stdout.log").open("xb") as out, (REPORT / "queue.stderr.log").open("xb") as err:
        process = subprocess.Popen(command, cwd=_manager.ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, stdout=out, stderr=err, start_new_session=True)
    result = {"pid": process.pid, "command": command, "started_at": _manager.now(),
        "recurring_automation": False, "preserved_failed_queue": str(BASE_REPORT / "QUEUE_FAILURE.json"),
        "shared_exclusive_scientific_bundle": str(_manager.BUNDLE),
        "contract_raw_sha256": _manager.sha(_manager.BUNDLE / "execution_contract.json"),
        "local_queue_source_sha256": _manager.sha(__file__), "own_os_exit_directly_observed": False}
    _manager.write(REPORT / "QUEUE_STARTED.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("start-queue", "wait-predecessor"), required=True)
    args = parser.parse_args(argv)
    try:
        result = start_queue() if args.mode == "start-queue" else _manager.wait_predecessor()
    except Exception as error:
        if args.mode == "wait-predecessor" and not (REPORT / "QUEUE_FAILURE.json").exists():
            _manager.write(REPORT / "QUEUE_FAILURE.json", {"status": "failed", "at": _manager.now(),
                "error": str(error), "automatic_retry": False})
        raise
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
