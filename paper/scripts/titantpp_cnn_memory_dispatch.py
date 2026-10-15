"""Bounded, single-attempt dispatch of frozen CNN Train/Validation diagnostics.

This module is stdlib-only and has no deployment, training, or held-out route.
Each host finishes every Validation replay before admitting any Train worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


SCHEMA = "titantpp_cnn_memory_sensitivity_v1"
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def confined(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and bool(relative), "invalid relative path")
    rel = Path(relative)
    require(not rel.is_absolute() and ".." not in rel.parts, "path escapes root")
    path = root / rel
    require(path.resolve().is_relative_to(root.resolve()), "resolved path escapes root")
    for part in [path, *path.parents]:
        if part == root:
            break
        require(not part.is_symlink(), f"symlink is not permitted: {relative}")
    return path


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_contract(root: Path, expected_sha: str, host: str) -> dict:
    require(SHA.fullmatch(expected_sha) is not None, "invalid contract SHA")
    path = confined(root, "contract.json")
    require(digest(path) == expected_sha, "contract SHA mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    require(contract.get("schema") == SCHEMA, "contract schema mismatch")
    require(host in ("5080", "5090"), "unsupported host")
    require(type(contract.get("max_parallel_per_host")) is int
            and contract["max_parallel_per_host"] == 3, "parallel limit must be 3")
    deadline = contract.get("deadline_unix")
    require(type(deadline) in (int, float) and math.isfinite(deadline), "invalid deadline")
    require(deadline > time.time(), "contract deadline already expired")
    files = contract.get("operation_files")
    require(isinstance(files, dict) and "operation/evaluate.py" in files,
            "operation/evaluate.py must be SHA pinned")
    for relative, expected in files.items():
        require(isinstance(expected, str) and SHA.fullmatch(expected) is not None,
                f"invalid operation SHA: {relative}")
        require(digest(confined(root, relative)) == expected, f"operation SHA mismatch: {relative}")
    jobs = contract.get("jobs")
    require(isinstance(jobs, list) and bool(jobs), "jobs must be nonempty")
    ids = set()
    for job in jobs:
        require(isinstance(job, dict), "invalid job")
        job_id = job.get("id")
        require(isinstance(job_id, str) and SAFE_ID.fullmatch(job_id) is not None,
                "invalid job id")
        require(job_id not in ids, "duplicate job id")
        require(job.get("host") in ("5080", "5090"), "invalid job host")
        ids.add(job_id)
    require(any(job["host"] == host for job in jobs), "host has no assigned jobs")
    python = contract["runtimes"][host]["python"]
    require(isinstance(python, str) and Path(python).is_absolute(), "Python path must be absolute")
    require(Path(python).is_file() and os.access(python, os.X_OK), "Python is not executable")
    return contract


def validate_worker_receipt(root: Path, job_id: str, split: str) -> dict:
    path = confined(root, f"results/{job_id}/{split}/receipt.json")
    raw = path.read_bytes()
    receipt = json.loads(raw)
    require(receipt.get("status") == "complete", "worker receipt is not complete")
    require(receipt.get("job_id") == job_id and receipt.get("split") == split,
            "worker receipt identity mismatch")
    if split == "validation":
        require(receipt.get("baseline_validation_reproduced") is True,
                "Validation baseline replay gate failed")
    return {"path": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest()}


def _terminate_owned(children: dict[int, dict]) -> None:
    """Signal only live direct children created here in their own sessions."""
    live = [item["process"] for item in children.values() if item["process"].poll() is None]
    groups = []
    for process in live:
        try:
            require(os.getpgid(process.pid) == process.pid and os.getsid(process.pid) == process.pid,
                    "owned child lost its private session/process group")
            os.killpg(process.pid, signal.SIGTERM)
            groups.append(process.pid)
        except ProcessLookupError:
            pass
    until = time.monotonic() + 1.0
    while any(process.poll() is None for process in live) and time.monotonic() < until:
        time.sleep(0.02)
    # A worker may exit on SIGTERM while one of its children remains in the group.
    # These group IDs were proved to be private sessions above; no external IDs
    # are discovered, accepted from files, or signalled.
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for process in live:
        process.wait(timeout=1)


def dispatch(root: Path, contract_sha: str, host: str) -> dict:
    root = root.resolve()
    contract = verify_contract(root, contract_sha, host)
    jobs = [job["id"] for job in contract["jobs"] if job["host"] == host]
    runtime = confined(root, f"runtime/{host}")
    require(not (runtime / "attempt.json").exists(), "existing attempt; replay/resume is forbidden")
    for job_id in jobs:
        for split in ("validation", "train"):
            require(not confined(root, f"results/{job_id}/{split}").exists(),
                    f"existing worker output; replay is forbidden: {job_id}/{split}")
    runtime.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_json(runtime / "attempt.json", {
        "schema": SCHEMA, "contract_sha256": contract_sha, "host": host,
        "dispatcher_pid": os.getpid(), "started_unix": started,
        "deadline_unix": contract["deadline_unix"], "max_parallel_per_host": 3,
        "job_ids": jobs, "automatic_retry": False,
    })
    events = (runtime / "events.jsonl").open("x", encoding="utf-8")
    children: dict[int, dict] = {}
    outcomes: list[dict] = []
    failure: str | None = None
    timed_out = False
    prior_handlers = {}

    def event(kind: str, **fields: Any) -> None:
        events.write(json.dumps({"event": kind, "time_unix": time.time(), **fields},
                                sort_keys=True, allow_nan=False) + "\n")
        events.flush()
        os.fsync(events.fileno())

    def interrupted(signum: int, _frame: Any) -> None:
        raise RuntimeError(f"dispatcher interrupted by signal {signum}")

    def finish(pid: int, *, cleanup_reason: str | None = None) -> None:
        nonlocal failure
        item = children.pop(pid)
        process = item["process"]
        code = process.wait()
        for stream in (item["stdout"], item["stderr"]):
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
        result = {
            "job_id": item["job_id"], "split": item["split"], "pid": pid,
            "actual_worker_exit_code": code, "started_unix": item["started_unix"],
            "finished_unix": time.time(), "cleanup_reason": cleanup_reason,
            "stdout_sha256": digest(item["stdout_path"]),
            "stderr_sha256": digest(item["stderr_path"]),
        }
        error = cleanup_reason
        if code == 0 and error is None:
            try:
                result["receipt"] = validate_worker_receipt(root, item["job_id"], item["split"])
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        elif error is None:
            error = f"worker exited {code}"
        result["status"] = "complete" if error is None else "failed"
        result["error"] = error
        write_json(item["exit_path"], result)
        outcomes.append(result)
        event("finish", **result)
        if error and failure is None:
            failure = f"{item['job_id']}/{item['split']}: {error}"

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            prior_handlers[sig] = signal.signal(sig, interrupted)
        for split in ("validation", "train"):
            pending = iter(jobs)
            next_job = next(pending, None)
            while next_job is not None or children:
                if time.time() >= contract["deadline_unix"]:
                    timed_out = True
                    raise TimeoutError("fixed contract deadline reached")
                # Reap before admission so a known failure never admits another worker.
                for pid, item in list(children.items()):
                    if item["process"].poll() is not None:
                        finish(pid)
                while failure is None and next_job is not None and len(children) < 3:
                    if time.time() >= contract["deadline_unix"]:
                        timed_out = True
                        raise TimeoutError("fixed contract deadline reached")
                    # Never execute an operation changed after the initial audit.
                    require(digest(root / "contract.json") == contract_sha,
                            "contract changed before admission")
                    for relative, expected in contract["operation_files"].items():
                        require(digest(confined(root, relative)) == expected,
                                "operation changed before admission")
                    job_id = next_job
                    output = runtime / "workers" / job_id
                    output.mkdir(parents=True, exist_ok=True)
                    stdout_path, stderr_path = output / f"{split}.stdout", output / f"{split}.stderr"
                    stdout = stdout_path.open("xb")
                    try:
                        stderr = stderr_path.open("xb")
                    except BaseException:
                        stdout.close()
                        raise
                    argv = [contract["runtimes"][host]["python"], str(root / "operation/evaluate.py"),
                            "--root", str(root), "--contract-sha", contract_sha,
                            "--job", job_id, "--split", split]
                    launched = time.time()
                    try:
                        process = subprocess.Popen(argv, cwd=root, stdin=subprocess.DEVNULL,
                                                   stdout=stdout, stderr=stderr, start_new_session=True,
                                                   env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
                    except Exception as exc:
                        stdout.close()
                        stderr.close()
                        failure = f"{job_id}/{split}: launch failed: {type(exc).__name__}: {exc}"
                        write_json(output / f"{split}.launch_failure.json", {
                            "status": "failed", "error": failure, "argv": argv,
                            "actual_worker_exit_code": None, "worker_started": False,
                            "time_unix": time.time(),
                        })
                        event("launch_failure", job_id=job_id, split=split, error=failure)
                        break
                    except BaseException:
                        stdout.close()
                        stderr.close()
                        raise
                    children[process.pid] = dict(process=process, job_id=job_id, split=split,
                        stdout=stdout, stderr=stderr, stdout_path=stdout_path, stderr_path=stderr_path,
                        started_unix=launched, exit_path=output / f"{split}.exit.json")
                    event("launch", job_id=job_id, split=split, pid=process.pid,
                          active_child_count=len(children), argv=argv)
                    next_job = next(pending, None)
                if failure is not None and not children:
                    break
                if children:
                    time.sleep(0.05)
            if failure is not None:
                break
            stage = [item for item in outcomes if item["split"] == split]
            require(len(stage) == len(jobs) and all(item["status"] == "complete" for item in stage),
                    f"incomplete {split} stage")
            # Re-read SHA and content before the Validation -> Train transition.
            for item in stage:
                current = validate_worker_receipt(root, item["job_id"], split)
                require(current == item["receipt"], "worker receipt changed after completion")
            event("stage_complete", split=split, worker_count=len(stage))
        if failure is None:
            require(len(outcomes) == 2 * len(jobs), "host diagnostic count mismatch")
            for item in outcomes:
                require(validate_worker_receipt(root, item["job_id"], item["split"]) == item["receipt"],
                        "worker receipt changed after completion")
            require(digest(root / "contract.json") == contract_sha, "contract changed during dispatch")
            for relative, expected in contract["operation_files"].items():
                require(digest(confined(root, relative)) == expected, "operation changed during dispatch")
    except BaseException as exc:
        failure = failure or f"{type(exc).__name__}: {exc}"
        _terminate_owned(children)
        for pid in list(children):
            finish(pid, cleanup_reason=failure)
    finally:
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)
        events.close()
    receipt = {
        "schema": SCHEMA, "status": "complete" if failure is None else "failed",
        "host": host, "contract_sha256": contract_sha, "started_unix": started,
        "finished_unix": time.time(), "deadline_unix": contract["deadline_unix"],
        "timed_out": timed_out, "error": failure, "job_count": len(jobs),
        "expected_worker_count": 2 * len(jobs), "actual_worker_count": len(outcomes),
        "actual_worker_exit0_count": sum(item["actual_worker_exit_code"] == 0 for item in outcomes),
        "workers": outcomes, "events_sha256": digest(runtime / "events.jsonl"),
        "max_parallel_per_host": 3, "automatic_retry": False,
        "dispatcher_own_os_exit_directly_observed": False,
    }
    write_json(runtime / "receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--host", required=True, choices=("5080", "5090"))
    args = parser.parse_args()
    try:
        receipt = dispatch(args.root, args.contract_sha, args.host)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({key: receipt[key] for key in ("status", "host", "error", "actual_worker_count")}))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
