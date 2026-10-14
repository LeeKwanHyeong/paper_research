"""Supervise only approved diagnostic workers in a new artifact root."""
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as f:
        json.dump(v, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def main():
    root = Path(sys.argv[1]).resolve()
    cpath = root / "contract.json"
    contract_bytes = cpath.read_bytes()
    contract_sha = hashlib.sha256(contract_bytes).hexdigest()
    c = json.loads(contract_bytes)
    assert c["new_training"] is False and c["frozen_selection"] is True
    assert str(root) == c["authorized_remote_root"]
    assert sha(Path(__file__)) == c["pipeline_sha256"]
    assert sha(root / "operation/evaluate.py") == c["evaluator_sha256"]
    assert sha(root / c["approval"]["path"]) == c["approval"]["sha256"]
    write(root / "pipeline_claim.json", {"pid": os.getpid(), "argv": sys.argv,
        "started_unix": time.time(), "contract_sha256": contract_sha, "automatic_retry": False})
    rt = c["runtime"]
    query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    assert not query.stdout.strip(), "GPU has existing compute processes; no worker launched"
    results = []

    def worker(pair):
        job, split = pair
        assert sha(cpath) == contract_sha, "Contract changed before worker dispatch"
        argv = [rt["python"], str(root / "operation/evaluate.py"), "--root", str(root),
            "--job", job["id"], "--split", split, "--contract-sha", contract_sha]
        started = time.time()
        output = root / "results" / job["id"] / split
        output.parent.mkdir(parents=True, exist_ok=True)
        log = root / "logs" / (job["id"] + "__" + split + ".log")
        log.parent.mkdir(exist_ok=True)
        process = None
        with log.open("xb") as f:
            process = subprocess.Popen(argv, stdout=f, stderr=subprocess.STDOUT, cwd=root,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, start_new_session=True)
            try:
                code = process.wait(timeout=c["worker_deadline_seconds"])
            except subprocess.TimeoutExpired:
                # Only this worker is terminated after its declared hard deadline.
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = process.returncode
                write(output.parent / (split + "_hard_timeout.json"), {"pid": process.pid, "deadline_seconds": c["worker_deadline_seconds"]})
        receipt = {"id": job["id"], "job_id": job["id"], "split": split, "pid": process.pid, "argv": argv,
            "actual_child_exit_observed": True, "exit_code": code, "started_unix": started,
            "completed_unix": time.time(), "contract_sha256": contract_sha, "log_sha256": sha(log)}
        write(output.parent / (split + "_worker_exit.json"), receipt)
        print(json.dumps({"id": job["id"], "split": split, "exit_code": code}), flush=True)
        return receipt

    def stage(pairs):
        # At most two fixed workers share the GPU; a failure stops further dispatch.
        it = iter(pairs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=c["parallel_workers"]) as pool:
            pending = {}
            for _ in range(c["parallel_workers"]):
                pair = next(it, None)
                if pair is not None:
                    pending[pool.submit(worker, pair)] = pair
            while pending:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                failed = False
                for f in done:
                    pending.pop(f)
                    result = f.result()
                    results.append(result)
                    failed |= result["exit_code"] != 0
                if failed:
                    for f in pending:
                        results.append(f.result())
                    return False
                for _ in done:
                    pair = next(it, None)
                    if pair is not None:
                        pending[pool.submit(worker, pair)] = pair
        return True

    assert stage([(j, "validation") for j in c["jobs"]]), "Validation stage failed"
    assert sha(cpath) == contract_sha, "Contract changed before Validation gate"
    gate = {"status": "complete", "contract_sha256": contract_sha, "jobs": {}, "actual_worker_exit0": True}
    for j in c["jobs"]:
        p = root / "results" / j["id"] / "validation" / "receipt.json"
        v = json.loads(p.read_text())
        assert v["status"] == "complete"
        gate["jobs"][j["id"]] = {"receipt_sha256": sha(p),
            "worker_exit_sha256": sha(p.parent.parent / "validation_worker_exit.json")}
    write(root / "validation_gate.json", gate)
    assert stage([(j, s) for s in ["train", "test"] for j in c["jobs"] if s in j["allowed_splits"]]), "Diagnostic stage failed"
    assert sha(cpath) == contract_sha, "Contract changed before completion"
    write(root / "pipeline_completion.json", {"status": "complete", "actual_worker_count": len(results),
        "worker_exit0": all(r["exit_code"] == 0 for r in results), "worker_exits": results,
        "contract_sha256": contract_sha, "completed_unix": time.time(), "new_training": False})
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        root = Path(sys.argv[1]).resolve()
        write(root / "pipeline_failure.json", {"error": repr(e), "automatic_retry": False, "completed_unix": time.time()})
        raise
