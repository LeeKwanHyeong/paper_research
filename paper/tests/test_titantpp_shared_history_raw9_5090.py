"""Fabricated host-only migration checks; no SSH, Cap mutation or data fit."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from paper.scripts import titantpp_shared_history_raw9_5090_runtime as r
from paper.scripts import manage_titantpp_shared_history_raw9_5090 as m
from paper.tests.test_titantpp_shared_history_raw9 import contract as old_fixture, make_receipt as old_receipt


@pytest.fixture
def contract(tmp_path):
    c = old_fixture.__wrapped__(tmp_path)
    h = c["hosts"].pop("5080")
    h.update(paused_processes=[], absolute_host_deadline_unix=1792888885.815062,
        gpu_uuid=r._GPU_UUID, runtime_expected={"python": "3.12.13", "torch": "2.11.0+cu130", "cuda": "13.0",
            "cudnn": 92000, "threads": 4, "interop_threads": 1}, tmux_binary="/usr/bin/tmux", tmux="fabricated-5090")
    c.update(schema=r.SCHEMA, hosts={"5090": h}, maximum_parallel_workers=3)
    for job in c["jobs"]: job["host"] = "5090"
    return c


def make_receipt(c, job, stage):
    view = deepcopy(c)
    view["hosts"] = {"5080": view["hosts"]["5090"]}
    old = old_receipt(view, job, stage)
    old.update(schema="titantpp_shared_history_raw9_5090_stage_v1", contract_sha256=r.sha_json(c))
    path = Path(c["hosts"]["5090"]["root"]) / "run" / job["id"] / f"{stage}_receipt.json"
    r.write(path, old)
    return old


def test_exact_original_science9_only_native5090_no_cap_wait(contract):
    assert len(r.validate_contract(contract, "5090")) == 9
    assert r.WORKER_MODULE == "paper.scripts.run_titantpp_shared_history_raw9_5090"
    assert r.STAGES == ("train", "validation") and r.ARMS == ("titantpp_history_mlp_shared15_v1",)
    assert contract["common_condition"] == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.}
    with pytest.raises(ValueError): r.validate_contract(contract, "5080")


@pytest.mark.parametrize("mutation", ("host", "paused", "deadline", "gpu", "cudnn", "threads", "parallel", "alpha", "cap", "seed", "test", "arm"))
def test_host_scope_and_science_mutations_rejected(contract, mutation):
    c = deepcopy(contract)
    if mutation == "host": c["jobs"][0]["host"] = "5080"
    if mutation == "paused": c["hosts"]["5090"]["paused_processes"] = [{"pid": 111, "state": "T"}]
    if mutation == "deadline": c["hosts"]["5090"]["absolute_host_deadline_unix"] += 1
    if mutation == "gpu": c["hosts"]["5090"]["gpu_uuid"] = "GPU-wrong"
    if mutation == "cudnn": c["hosts"]["5090"]["runtime_expected"]["cudnn"] = 91900
    if mutation == "threads": c["hosts"]["5090"]["runtime_expected"]["threads"] = 8
    if mutation == "parallel": c["maximum_parallel_workers"] = 4
    if mutation == "alpha": c["common_condition"]["time_alpha"] = 0.
    if mutation == "cap": c["datasets"][0]["cap_value"] = 20
    if mutation == "seed": c["jobs"][0]["seed"] = 32
    if mutation == "test": c["held_out_test_authorized"] = True
    if mutation == "arm": c["jobs"][0]["arm"] = "titantpp_history_mlp_prefix_capacity_v1"
    with pytest.raises((ValueError, KeyError)): r.validate_contract(c, "5090")


def test_fresh_gate_without_cap_files_allows_empty_gpu_only(contract, monkeypatch):
    # The fixture's Cap predecessor paths deliberately do not exist.
    assert not Path(contract["predecessor"]["receipt_path"]).exists()
    monkeypatch.setattr(r._impl, "gpu_snapshot", lambda uuid: {"uuid": uuid, "compute_pids": []})
    gate = r.predecessor_gate(contract, "5090", require_complete=True)
    assert gate["fresh_idle_verified"] and gate["cap5080_completion_required"] is False
    assert gate["old_campaigns_changed_or_resumed"] is False
    monkeypatch.setattr(r._impl, "gpu_snapshot", lambda uuid: {"uuid": uuid, "compute_pids": [999]})
    with pytest.raises(ValueError): r.predecessor_gate(contract, "5090")
    assert r.gpu_state(contract, "5090", [999])[0]["compute_pids"] == [999]


def test_joint_receipt_schema_and_binary_preserved_with_local_originals(contract, tmp_path):
    job = contract["jobs"][0]
    make_receipt(contract, job, "train")
    value = make_receipt(contract, job, "validation")
    binding = r.verify_receipt(contract, "5090", job, "validation")
    assert binding["sha256"]
    import shutil
    originals = tmp_path / "originals"
    shutil.copytree(contract["hosts"]["5090"]["root"], originals)
    assert r.verify_receipt(contract, "5090", job, "validation", originals=originals) == binding
    path = Path(contract["hosts"]["5090"]["root"]) / binding["path"]
    for change in ({"schema": "titantpp_shared_history_raw9_stage_v1"}, {"time_alpha": 0.},
        {"time_head_training_verification": {}}, {"target_identity": {}}, {"validation_replay_passed": False}):
        r.write(path, {**value, **change})
        with pytest.raises(ValueError): r.verify_receipt(contract, "5090", job, "validation")


def test_all9_train_then9validation_bounded3_and_no_restarting(contract, monkeypatch):
    root = Path(contract["hosts"]["5090"]["root"])
    monkeypatch.chdir(contract["hosts"]["5090"]["source_root"])
    r.write(root / "qualification.json", {"status": "passed", "host": "5090", "contract_sha256": r.sha_json(contract)})
    jobs = {j["id"]: j for j in contract["jobs"]}
    children, stages, maximum = [], [], [0]
    class Child:
        def __init__(self, command, **kwargs):
            self.pid, self.command, self.cwd = len(children) + 1000, command, str(kwargs["cwd"])
            self.returncode, self.ticks = None, 0
            self.job, self.stage = jobs[command[command.index("--job") + 1]], command[-1]
            assert command[command.index("--host") + 1] == "5090"
            children.append(self); stages.append(self.stage)
            maximum[0] = max(maximum[0], sum(p.returncode is None for p in children))
        def poll(self):
            if self.returncode is None:
                self.ticks += 1
                if self.ticks >= 4:
                    self.returncode = 0; make_receipt(contract, self.job, self.stage)
            return self.returncode
    monkeypatch.setattr(r._impl.subprocess, "Popen", Child)
    monkeypatch.setattr(r._impl, "gpu_state", lambda c, h, pids=(): ({"compute_pids": pids}, []))
    monkeypatch.setattr(r._impl, "proc_identity", lambda pid: next(({"pid": p.pid, "pgid": p.pid, "start_ticks": p.pid,
        "argv": p.command, "cwd": p.cwd, "state": "R"} for p in children if p.pid == pid and p.returncode is None), None))
    monkeypatch.setattr(r._impl, "predecessor_gate", lambda *a, **kw: {"status": "passed"})
    monkeypatch.setattr(r._impl.time, "sleep", lambda _: None)
    assert r.dispatch(contract, "5090", root / "execution_contract.json") == 0
    assert stages == ["train"] * 9 + ["validation"] * 9 and maximum[0] == 3
    assert r.read(root / "final_receipt.json")["held_out_test_evaluated"] is False
    with pytest.raises(ValueError): r.dispatch(contract, "5090", root / "execution_contract.json")


def test_qualification_uses_new_validator_and_no_cap_dependency(contract, monkeypatch):
    monkeypatch.chdir(contract["hosts"]["5090"]["source_root"])
    monkeypatch.setattr(r._impl, "gpu_state", lambda *a: ({"compute_pids": []}, []))
    monkeypatch.setattr(r._impl, "paused_processes", lambda *a: [])
    monkeypatch.setattr(r._impl, "free_gpu_bytes", lambda *a: 1000)
    monkeypatch.setattr(r._impl, "runtime_check", lambda *a: {"cudnn": 92000})
    monkeypatch.setattr(r._impl, "predecessor_gate", lambda *a, **kw: {"fresh_idle_verified": True})
    monkeypatch.setattr(r._impl, "qualify_three_concurrent", lambda *a: {"native_worker_exit0_count": 3})
    worker = ModuleType(r.WORKER_MODULE)
    worker.synthetic_check = lambda *a: {"status": "passed", "real_data_loaded": False,
        "cuda_memory_peaks": {r.ARMS[0]: {"max_memory_reserved_bytes": 100}}}
    monkeypatch.setitem(sys.modules, r.WORKER_MODULE, worker)
    qualification = r.qualify(contract, "5090")
    assert qualification["three_worker_native_admission"]["native_worker_exit0_count"] == 3
    assert qualification["parallel_memory_admission"]["three_worker_reserved_estimate_bytes"] == 300


def test_collector_and_preflight_use5090_and_no_cap_read(contract, monkeypatch):
    program = m.collection_program(contract, "5090")
    compile(program, "<5090collector>", "exec")
    assert "titantpp_shared_history_raw9_5090_runtime as r" in program
    calls = []
    monkeypatch.setattr(m._raw._impl, "ssh", lambda host, command: calls.append((host, command)) or object())
    monkeypatch.setattr(m._raw._impl, "save_transport", lambda *a: json.dumps({"status": "passed", "host": "5090",
        "compute_pids": [], "gpu_uuid": r._GPU_UUID}))
    monkeypatch.setattr(m, "write", lambda *a: None)
    assert m.fresh_preflight(contract)["host"] == "5090"
    assert calls[0][0] == "5090" and "Cap18" not in calls[0][1] and "cap5080" in calls[0][1]
    import shlex
    compile(shlex.split(calls[0][1])[-1], "<5090preflight>", "exec")


def test_5090_first_finite_finish_and_repeated_failure_guard(contract, tmp_path, monkeypatch):
    bundle, report = tmp_path / "bundle5090", tmp_path / "report5090"
    bundle.mkdir(); report.mkdir()
    monkeypatch.setattr(m._raw, "BUNDLE", bundle)
    monkeypatch.setattr(m._raw, "REPORT", report)
    calls = []
    monkeypatch.setattr(m._raw, "verify_local", lambda **kwargs: contract)
    monkeypatch.setattr(m._raw._impl, "ssh", lambda host, *a, **kw: calls.append(("wait", host)) or object())
    monkeypatch.setattr(m._raw._impl, "save_transport", lambda *a: None)
    monkeypatch.setattr(m._raw._impl, "collect", lambda host: calls.append(("collect", host)) or {"status": "complete"})
    compare = ModuleType("paper.scripts.compare_titantpp_shared_history_raw9_5090")
    compare.main = lambda: calls.append(("compare", "5090"))
    monkeypatch.setitem(sys.modules, compare.__name__, compare)
    assert m.finish()["status"] == "complete"
    assert calls == [("wait", "5090"), ("collect", "5090"), ("compare", "5090")]
    with pytest.raises(ValueError, match="no repeated remote work"): m.finish()
    assert len(calls) == 3


def test_no_wait_queue_cli_and_frozen5080_modules_unchanged():
    from paper.scripts import titantpp_shared_history_raw9_runtime as old
    from paper.scripts import manage_titantpp_shared_history_raw9 as old_manager
    assert old.SCHEMA == "titantpp_shared_history_raw9_v1" and old._impl is not r._impl
    assert old_manager.BUNDLE != m.BUNDLE and old_manager.MODULE != m.MODULE
    assert r.sha_file(r._PATH) == r._SHA and m.sha(m._PATH) == m._SHA
    assert r._impl.run_stage.__globals__ is r._impl.__dict__
    with pytest.raises(SystemExit): m.main(["--mode", "wait-predecessor"])
