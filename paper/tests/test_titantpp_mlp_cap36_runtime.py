"""CPU-only queue contracts: no model, real data, SSH, or GPU execution."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.scripts import titantpp_mlp_cap36_runtime as r


@pytest.fixture
def contract(tmp_path):
    hosts = {}
    for host in ("5080", "5090"):
        root = tmp_path / host
        source = root / "source"
        source.mkdir(parents=True)
        (source / "science.py").write_text("same science\n")
        (source / "data.txt").write_text("synthetic input\n")
        hosts[host] = {"root": str(root), "source_root": str(source), "python": "/python",
            "gpu_uuid": "GPU-" + host, "environment": {}, "runtime_expected": {"python": "3.12.13"}}
    source = Path(hosts["5080"]["source_root"])
    inventory = {"science.py": r.sha_file(source / "science.py")}
    datasets = ("taxi", "raf", "inter")
    jobs = [{"id": f"{d}__{a}__seed{s}", "dataset": d, "arm": a, "seed": s,
             "host": "5080" if d == "taxi" or (d == "inter" and s == 42) else "5090"}
            for d in datasets for a in r.ARMS for s in (42, 52, 62)]
    return {"schema": r.SCHEMA, "hosts": hosts, "datasets": [{"dataset_id": d} for d in datasets], "jobs": jobs,
        "source": {"files": inventory, "files_sha256": r.sha_json(inventory)},
        "input_files": {"data.txt": r.sha_file(source / "data.txt")},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40, "batch_size": 128},
        "limits": {"fit_wall_seconds": 129600, "queue_wall_seconds": 604800, "evaluation_wall_seconds": 7200}}


def receipt(c, host, job, stage, *, replay=True):
    root = Path(c["hosts"][host]["root"])
    directory = root / "run" / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    data = directory / f"{stage}.txt"
    data.write_text("fixed selection\n")
    value = {"status": "complete", "job_id": job["id"], "stage": stage,
             "contract_sha256": r.sha_json(c), "files": {str(data.relative_to(root)): r.sha_file(data)}}
    if stage == "validation":
        value["validation_replay_passed"] = replay
    r.write(directory / f"{stage}_receipt.json", value, exclusive=True)
    return value


def test_contract_matrix_and_fixed_budget(contract):
    assert len(r.validate_contract(contract, "5080")) == 16
    assert len(r.validate_contract(contract, "5090")) == 20
    for field, value in (("seed", 32), ("arm", "historical_cnn"), ("host", "unknown"), ("id", "../escape")):
        broken = deepcopy(contract)
        broken["jobs"][0][field] = value
        with pytest.raises(ValueError):
            r.validate_contract(broken, "5080")
    broken = deepcopy(contract)
    broken["training"]["maximum_epochs"] = 301
    with pytest.raises(ValueError):
        r.validate_contract(broken, "5080")


@pytest.mark.parametrize("active,gpu,expected", [([], [], 3), ([1], [], 2), ([1, 2], [2], 1), ([1, 2, 3], [1, 2, 3], 0)])
def test_slots_include_cpu_loading_workers(active, gpu, expected):
    assert r.available_slots(active, gpu) == expected


def test_foreign_gpu_and_overcapacity_stop_admission():
    with pytest.raises(ValueError, match="Unidentified"):
        r.available_slots([1], [1, 99])
    with pytest.raises(ValueError, match="concurrency"):
        r.available_slots([1, 2, 3, 4], [])


def test_receipt_detects_corruption_empty_inventory_and_replay_failure(contract):
    job = r.validate_contract(contract, "5080")[0]
    root = Path(contract["hosts"]["5080"]["root"])
    value = receipt(contract, "5080", job, "validation")
    assert r.verify_receipt(contract, "5080", job, "validation")["sha256"]
    path = root / "run" / job["id"] / "validation_receipt.json"
    for update in ({"files": {}}, {"validation_replay_passed": False}, {"job_id": "wrong"},
                   {"files": {"../escape": "invalid"}}):
        r.write(path, {**value, **update})
        with pytest.raises(ValueError):
            r.verify_receipt(contract, "5080", job, "validation")
    r.write(path, value)
    (root / next(iter(value["files"]))).write_text("changed")
    with pytest.raises(ValueError, match="SHA"):
        r.verify_receipt(contract, "5080", job, "validation")


def test_validation_gate_requires_every_host_worker(contract):
    jobs = r.validate_contract(contract, "5080")
    exits = {j["id"]: {"actual_os_returncode": 0, "timeout": False} for j in jobs}
    for job in jobs:
        receipt(contract, "5080", job, "validation")
    with pytest.raises(ValueError, match="Every"):
        r.validation_gate(contract, "5080", {j: e for j, e in exits.items() if j != jobs[0]["id"]})
    failed = deepcopy(exits)
    failed[jobs[0]["id"]]["actual_os_returncode"] = 1
    with pytest.raises(ValueError, match="Every"):
        r.validation_gate(contract, "5080", failed)
    gate = r.validation_gate(contract, "5080", exits)
    assert len(gate["jobs"]) == 16
    with pytest.raises(FileExistsError):
        r.validation_gate(contract, "5080", exits)


def fake_queue(monkeypatch, c, host="5080", failure_job=None, bad_replay=False):
    target = c["hosts"][host]
    root = Path(target["root"])
    (root / "logs").mkdir(exist_ok=True)
    jobs = {j["id"]: j for j in r.validate_contract(c, host)}
    instances, launched, maximum = [], [], [0]

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(instances)
            self.command, self.cwd = command, str(kwargs["cwd"])
            self.job = jobs[command[command.index("--job") + 1]]
            self.stage = command[-1]
            self.returncode, self.ticks = None, 0
            self.env = kwargs["env"]
            assert self.env["PYTHONHASHSEED"] == str(self.job["seed"])
            assert kwargs["start_new_session"] is True
            instances.append(self)
            launched.append((self.stage, self.job["id"]))
            maximum[0] = max(maximum[0], sum(p.returncode is None for p in instances))

        def poll(self):
            if self.returncode is None:
                self.ticks += 1
                if self.ticks >= 4:
                    self.returncode = 1 if self.job["id"] == failure_job else 0
                    if self.returncode == 0:
                        receipt(c, host, self.job, self.stage, replay=not bad_replay)
            return self.returncode

    monkeypatch.setattr(r.subprocess, "Popen", Child)
    monkeypatch.setattr(r, "gpu_snapshot", lambda uuid: {"compute_pids": [p.pid for p in instances if p.returncode is None], "uuid": uuid})
    monkeypatch.setattr(r, "proc_identity", lambda pid: next(({"pid": p.pid, "pgid": p.pid,
        "start_ticks": p.pid, "argv": p.command, "cwd": p.cwd} for p in instances if p.pid == pid and p.returncode is None), None))
    monkeypatch.setattr(r.time, "sleep", lambda _: None)
    return launched, maximum


def test_queue_runs_all_jobs_with_no_more_than_three_workers(contract, monkeypatch):
    launched, maximum = fake_queue(monkeypatch, contract)
    result = r.run_stage(contract, "5080", Path("/contract.json"), "train", r.time.monotonic())
    assert result["status"] == "complete" and len(result["exits"]) == 16
    assert len(launched) == 16 and maximum[0] == 3
    assert all(e["actual_os_returncode"] == 0 for e in result["exits"].values())


def test_failure_drains_owned_children_and_stops_new_jobs(contract, monkeypatch):
    job = r.validate_contract(contract, "5080")[0]
    launched, maximum = fake_queue(monkeypatch, contract, failure_job=job["id"])
    result = r.run_stage(contract, "5080", Path("/contract.json"), "train", r.time.monotonic())
    assert result["status"] == "failed" and len(launched) == 3
    assert len(result["exits"]) == 3 and len(result["pending_job_ids"]) == 13
    assert maximum[0] == 3


def test_existing_claim_is_not_reused(contract, monkeypatch):
    fake_queue(monkeypatch, contract)
    job = r.validate_contract(contract, "5080")[0]
    path = Path(contract["hosts"]["5080"]["root"]) / "run" / job["id"] / "train_claim.json"
    r.write(path, {"preserved": True}, exclusive=True)
    result = r.run_stage(contract, "5080", Path("/contract.json"), "train", r.time.monotonic())
    assert result["status"] == "failed" and not result["exits"]
    assert r.read(path) == {"preserved": True}


def test_full_dispatch_stages_and_manifest(contract, monkeypatch):
    host, target = "5080", contract["hosts"]["5080"]
    root = Path(target["root"])
    r.write(root / "qualification.json", {"status": "passed", "host": host, "contract_sha256": r.sha_json(contract)})
    monkeypatch.chdir(target["source_root"])
    launched, maximum = fake_queue(monkeypatch, contract)
    (root / "logs").rmdir()  # The dispatcher creates this itself.
    assert r.dispatch(contract, host, root / "execution_contract.json") == 0
    assert [stage for stage, _ in launched] == [s for s in r.STAGES for _ in range(16)]
    assert maximum[0] == 3
    final = r.read(root / "final_receipt.json")
    assert final["test_count"] == final["validation_replay_count"] == final["completed_fit_count"] == 16
    assert final["owned_worker_pids_absent"] is True
    manifest = r.read(root / "artifact_manifest.json")
    assert all(not p.startswith("source/") for p in manifest["files"])
    for path, digest in manifest["files"].items():
        assert r.sha_file(root / path) == digest
    with pytest.raises(ValueError, match="Existing"):
        r.dispatch(contract, host, root / "execution_contract.json")


def test_replay_failure_prevents_test(contract, monkeypatch):
    target = contract["hosts"]["5080"]
    root = Path(target["root"])
    r.write(root / "qualification.json", {"status": "passed", "host": "5080", "contract_sha256": r.sha_json(contract)})
    monkeypatch.chdir(target["source_root"])
    launched, _ = fake_queue(monkeypatch, contract, bad_replay=True)
    (root / "logs").rmdir()
    assert r.dispatch(contract, "5080", root / "execution_contract.json") == 1
    assert not any(stage == "test" for stage, _ in launched)
    assert not (root / "validation_gate.json").exists()


def test_wait_does_not_qualify_until_idle_and_keeps_original_deadline(contract, monkeypatch):
    target = contract["hosts"]["5080"]
    root = Path(target["root"])
    monkeypatch.chdir(target["source_root"])
    clock, calls, samples = [100.], [], iter(([999], [999], []))
    monkeypatch.setattr(r.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(r.time, "sleep", lambda value: (calls.append(("sleep", value)), clock.__setitem__(0, clock[0] + value)))
    monkeypatch.setattr(r, "gpu_snapshot", lambda uuid: {"uuid": uuid, "compute_pids": next(samples)})
    monkeypatch.setattr(r, "qualify_child", lambda *args: calls.append(("qualify", args[-1])))
    monkeypatch.setattr(r, "dispatch", lambda *args, **kwargs: calls.append(("dispatch", kwargs["queue_started"])) or 0)
    assert r.wait_dispatch(contract, "5080", root / "execution_contract.json") == 0
    assert calls == [("sleep", 60), ("sleep", 60), ("qualify", 604680.), ("dispatch", 100.)]
    with pytest.raises(FileExistsError):
        r.wait_dispatch(contract, "5080", root / "execution_contract.json")


def test_wait_refuses_existing_failure(contract, monkeypatch):
    target = contract["hosts"]["5080"]
    monkeypatch.chdir(target["source_root"])
    r.write(Path(target["root"]) / "dispatch_failure.json", {"preserved": True})
    with pytest.raises(ValueError, match="Existing"):
        r.wait_dispatch(contract, "5080", Path("/contract.json"))


def test_wait_deadline_does_not_start_qualification(contract, monkeypatch):
    target = contract["hosts"]["5080"]
    monkeypatch.chdir(target["source_root"])
    clock, qualification = [0.], []
    monkeypatch.setattr(r.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(r.time, "sleep", lambda _: clock.__setitem__(0, 604800.))
    monkeypatch.setattr(r, "gpu_snapshot", lambda _: {"compute_pids": [999]})
    monkeypatch.setattr(r, "qualify_child", lambda *args: qualification.append(args))
    with pytest.raises(ValueError, match="deadline"):
        r.wait_dispatch(contract, "5080", Path("/contract.json"))
    assert not qualification
    assert (Path(target["root"]) / "wait_failure.json").exists()


def test_stage_timeout_cleans_only_its_own_children(contract, monkeypatch):
    launched, _ = fake_queue(monkeypatch, contract)
    clock, cleaned = [0.], []
    monkeypatch.setattr(r.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(r.time, "sleep", lambda _: clock.__setitem__(0, 129601.))

    def terminate(process):
        cleaned.append(process.pid)
        process.returncode = -15

    monkeypatch.setattr(r, "terminate_owned_child", terminate)
    result = r.run_stage(contract, "5080", Path("/contract.json"), "train", 0.)
    assert result["status"] == "failed" and len(launched) == 3
    assert cleaned == [1000, 1001, 1002]
    assert all(e["timeout"] is True and e["actual_os_returncode"] == -15 for e in result["exits"].values())
