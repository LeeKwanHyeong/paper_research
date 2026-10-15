"""Synthetic subprocess tests: no model, GPU, or research data is read."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import pytest

from paper.scripts import titantpp_cnn_memory_dispatch as dispatcher
from paper.scripts.titantpp_cnn_memory_dispatch import dispatch


WORKER = r'''
import argparse, json, os, pathlib, sys, time
p = argparse.ArgumentParser()
for key in ("root", "contract-sha", "job", "split"):
    p.add_argument("--" + key, required=True)
a = p.parse_args()
root = pathlib.Path(a.root)
c = json.loads((root / "contract.json").read_text())
output = root / "results" / a.job / a.split
output.mkdir(parents=True)
with (output / "pid.json").open("x") as f:
    json.dump({"pid": os.getpid(), "pgid": os.getpgrp()}, f)
print("synthetic worker stdout", flush=True)
print("synthetic worker stderr", file=sys.stderr, flush=True)
if a.job == c.get("fail_job") and a.split == "validation":
    sys.exit(7)
time.sleep(c.get("worker_sleep", 0.1))
receipt = {"status": "complete", "job_id": a.job, "split": a.split,
           "baseline_validation_reproduced": a.job != c.get("bad_gate_job")}
if c.get("wrong_receipt_id"):
    receipt["job_id"] = "different_job"
with (output / "receipt.json").open("x") as f:
    json.dump(receipt, f)
'''


def bundle(tmp_path, count=5, **options):
    root = tmp_path / "bundle"
    (root / "operation").mkdir(parents=True)
    worker = root / "operation" / "evaluate.py"
    worker.write_text(WORKER)
    contract = {
        "schema": "titantpp_cnn_memory_sensitivity_v1",
        "jobs": [{"id": f"j{i}", "host": "5080"} for i in range(count)],
        "runtimes": {"5080": {"python": sys.executable}},
        "max_parallel_per_host": 3,
        "deadline_unix": time.time() + 20,
        "operation_files": {"operation/evaluate.py": hashlib.sha256(worker.read_bytes()).hexdigest()},
        **options,
    }
    path = root / "contract.json"
    path.write_text(json.dumps(contract))
    return root, hashlib.sha256(path.read_bytes()).hexdigest()


def events(root):
    return [json.loads(line) for line in (root / "runtime/5080/events.jsonl").read_text().splitlines()]


def test_parallel_limit_and_validation_gate(tmp_path):
    root, sha = bundle(tmp_path)
    receipt = dispatch(root, sha, "5080")
    assert receipt["status"] == "complete"
    assert receipt["actual_worker_exit0_count"] == 10
    assert receipt["actual_worker_count"] == 10
    assert receipt["dispatcher_own_os_exit_directly_observed"] is False
    log = events(root)
    launches = [e for e in log if e["event"] == "launch"]
    assert max(e["active_child_count"] for e in launches) == 3
    val_finish = [e["time_unix"] for e in log if e["event"] == "finish" and e["split"] == "validation"]
    train_launch = [e["time_unix"] for e in launches if e["split"] == "train"]
    assert len(val_finish) == len(train_launch) == 5
    assert min(train_launch) > max(val_finish)
    for item in receipt["workers"]:
        raw = (root / item["receipt"]["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item["receipt"]["sha256"]
        path = root / "runtime/5080/workers" / item["job_id"]
        assert b"synthetic worker stdout" in (path / f"{item['split']}.stdout").read_bytes()
        assert b"synthetic worker stderr" in (path / f"{item['split']}.stderr").read_bytes()


def test_first_failure_stops_admission_and_drains_existing_children(tmp_path):
    root, sha = bundle(tmp_path, fail_job="j0", worker_sleep=0.3)
    receipt = dispatch(root, sha, "5080")
    assert receipt["status"] == "failed"
    assert receipt["actual_worker_count"] == 3
    assert receipt["actual_worker_exit0_count"] == 2
    assert {row["job_id"] for row in receipt["workers"]} == {"j0", "j1", "j2"}
    assert next(row for row in receipt["workers"] if row["job_id"] == "j0")["actual_worker_exit_code"] == 7
    assert all(row["cleanup_reason"] is None for row in receipt["workers"])
    assert not any(e.get("split") == "train" for e in events(root))


@pytest.mark.parametrize("options", [{"bad_gate_job": "j0"}, {"wrong_receipt_id": True}])
def test_receipt_mismatch_blocks_train(tmp_path, options):
    root, sha = bundle(tmp_path, count=2, **options)
    receipt = dispatch(root, sha, "5080")
    assert receipt["status"] == "failed"
    assert receipt["actual_worker_exit0_count"] == 2
    assert all(item["split"] == "validation" for item in receipt["workers"])


def test_exclusive_attempt_prevents_replay(tmp_path):
    root, sha = bundle(tmp_path, count=1)
    assert dispatch(root, sha, "5080")["status"] == "complete"
    attempt = (root / "runtime/5080/attempt.json").read_bytes()
    with pytest.raises(ValueError, match="existing attempt"):
        dispatch(root, sha, "5080")
    assert (root / "runtime/5080/attempt.json").read_bytes() == attempt


@pytest.mark.parametrize("damage", ["contract", "operation"])
def test_sha_rejection_before_any_runtime_action(tmp_path, damage):
    root, sha = bundle(tmp_path)
    target = root / ("contract.json" if damage == "contract" else "operation/evaluate.py")
    with target.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA mismatch"):
        dispatch(root, sha, "5080")
    assert not (root / "runtime").exists()


def test_deadline_cleans_owned_child_groups(tmp_path):
    root, sha = bundle(tmp_path, count=4, worker_sleep=10, deadline_unix=time.time() + 0.7)
    started = time.monotonic()
    receipt = dispatch(root, sha, "5080")
    assert time.monotonic() - started < 4
    assert receipt["status"] == "failed" and receipt["timed_out"] is True
    assert receipt["actual_worker_count"] == 3
    assert all(item["actual_worker_exit_code"] != 0 for item in receipt["workers"])
    assert all(item["cleanup_reason"] is not None for item in receipt["workers"])
    for item in receipt["workers"]:
        with pytest.raises(ProcessLookupError):
            os.kill(item["pid"], 0)
    assert not (root / "results/j3").exists()


@pytest.mark.parametrize("options,match", [
    ({"deadline_unix": 1}, "deadline already expired"),
    ({"max_parallel_per_host": 1}, "parallel limit must be 3"),
    ({"jobs": [{"id": "../unsafe", "host": "5080"}]}, "invalid job id"),
    ({"jobs": [{"id": "x", "host": "5080"}, {"id": "x", "host": "5080"}]}, "duplicate job id"),
])
def test_invalid_contract_rejected_without_attempt(tmp_path, options, match):
    root, sha = bundle(tmp_path, **options)
    with pytest.raises(ValueError, match=match):
        dispatch(root, sha, "5080")
    assert not (root / "runtime").exists()


def test_existing_worker_outputs_preserved(tmp_path):
    root, sha = bundle(tmp_path, count=1)
    old = root / "results/j0/validation"
    old.mkdir(parents=True)
    marker = old / "failure.json"
    marker.write_text("original failure")
    with pytest.raises(ValueError, match="existing worker output"):
        dispatch(root, sha, "5080")
    assert marker.read_text() == "original failure"
    assert not (root / "runtime").exists()


def test_launch_failure_does_not_terminate_started_workers(tmp_path, monkeypatch):
    root, sha = bundle(tmp_path, count=5, worker_sleep=0.2)
    original = dispatcher.subprocess.Popen
    launch_count = 0

    def launch(*args, **kwargs):
        nonlocal launch_count
        launch_count += 1
        if launch_count == 3:
            raise OSError("synthetic process creation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatcher.subprocess, "Popen", launch)
    receipt = dispatch(root, sha, "5080")
    assert receipt["status"] == "failed"
    assert receipt["actual_worker_count"] == receipt["actual_worker_exit0_count"] == 2
    assert all(item["cleanup_reason"] is None for item in receipt["workers"])
    failure = root / "runtime/5080/workers/j2/validation.launch_failure.json"
    assert json.loads(failure.read_text())["worker_started"] is False
    assert not any(e.get("split") == "train" for e in events(root))


def test_host_assignment_isolated(tmp_path):
    root, _ = bundle(tmp_path, count=3)
    path = root / "contract.json"
    contract = json.loads(path.read_text())
    contract["jobs"] += [{"id": "other42", "host": "5090"}, {"id": "other52", "host": "5090"}]
    contract["runtimes"]["5090"] = {"python": sys.executable}
    path.write_text(json.dumps(contract))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = dispatch(root, sha, "5090")
    assert receipt["status"] == "complete"
    assert receipt["actual_worker_count"] == 4
    assert {item["job_id"] for item in receipt["workers"]} == {"other42", "other52"}
    assert not (root / "runtime/5080").exists()
    assert not (root / "results/j0").exists()
