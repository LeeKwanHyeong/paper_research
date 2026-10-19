"""Fabricated-only scope, lifecycle and ownership checks; no SSH or data fit."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from paper.scripts import titantpp_shared_history_cap_execution_runtime as r


def stopped():
    return [{"pid": p, "pgid": p, "start_ticks": 2 * p, "argv": ["old", str(p)],
             "cwd": "/old", "state": "T"} for p in (10, 11, 12, 13)]


@pytest.fixture
def contract(tmp_path):
    root, source = tmp_path / "5080", tmp_path / "5080/source"
    source.mkdir(parents=True)
    (source / "science.py").write_text("preserved = True\n")
    files = {"science.py": r.sha_file(source / "science.py")}
    inputs = {}
    for dataset in r.DATASETS:
        for name in ("train_validation.parquet", "split_manifest.json"):
            path = source / "inputs/data" / dataset / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fabricated input only\n")
            inputs[str(path.relative_to(source))] = r.sha_file(path)
    architecture = json.loads((Path(__file__).resolve().parents[2] /
        "reports/titantpp_history_mlp_shared_cap_20261019_v1/SCIENCE_SPEC.json").read_text())["architecture"]
    architecture.update(arms=list(r.ARMS), time_alpha=0., quantity_loss_mode="log_mse", quantity_loss_modes=["log_mse"],
        quantity_loss_normalization="none", model_architecture_changed=True, checkpoint_warmstart=False,
        time_input_transform="log1p_nonnegative_float32", quantity_input_transform="log1p_nonnegative_float32")
    return {"schema": r.SCHEMA, "scope": "execution", "execution_authorization": deepcopy(r._AUTHORIZATION),
        "hosts": {"5080": {"root": str(root), "source_root": str(source), "python": sys.executable,
            "gpu_uuid": "GPU-fabricated", "environment": {}, "runtime_expected": {},
            "absolute_host_deadline_unix": r._HOST_DEADLINE, "paused_processes": stopped(),
            "completed_predecessor": {"root": str(tmp_path / "parent"), "contract_raw_sha256": r._PARENT_RAW_SHA,
                "contract_canonical_sha256": "parent-canonical", "source_files_sha256": "parent-source",
                "dispatcher_identity": {"pid": 500}, "dispatcher_exit_sha256": "old-exit"}}},
        "datasets": [{"dataset_id": d, "cap_value": {"yellow_trip_hourly": 65., "raf_spare_parts": 10., "intermittent_frozen_5000": 20.}[d],
                      "loader": {"batch_size": 128}, "model": {"hidden_dim": 64},
                      "optimizer": {"grad_clip": 1., "lr": .001, "name": "AdamW", "weight_decay": .01}} for d in r.DATASETS],
        "source": {"files": files, "files_sha256": r.sha_json(files)}, "input_files": inputs,
        "architecture": architecture, "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 0.},
        "jobs": [{"id": f"{d}__{a}__seed{s}__losslog_mse", "dataset": d, "arm": a, "seed": s,
                  "loss_mode": "log_mse", "time_alpha": 0., "time_input": "log1p", "host": "5080"}
                 for d in r.DATASETS for a in r.ARMS for s in (42, 52, 62)],
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40, "batch_size": 128, "monitor": "validation_raw_quantity_rmse"},
        "limits": {"fit_wall_seconds": 129600, "queue_wall_seconds": 604800, "evaluation_wall_seconds": 7200},
        "evaluation_scope": "validation_only", "held_out_test_authorized": False,
        "baseline_validation_references": [{"dataset": d, "seed": s, "time_alpha": 0., "loss_mode": "log_mse",
            "time_input": "log1p", "time_input_transform": "log1p_nonnegative_float32", "time_head_trained": False,
            "retrained": False, "reevaluated": False, "sha256": "bound"} for d in r.DATASETS for s in (42, 52, 62)]}


def test_exact18_two_arms_three_datasets_and_only5080(contract):
    assert len(r.validate_contract(contract, "5080")) == 18
    assert r.STAGES == ("train", "validation") and r.LOSS_MODES == ("log_mse",)
    assert r.MODULE == "paper.scripts.titantpp_shared_history_cap_execution_runtime"
    assert r.WORKER_MODULE == "paper.scripts.run_titantpp_shared_history_cap_execution"
    with pytest.raises(ValueError):
        r.validate_contract(contract, "5090")


@pytest.mark.parametrize("mutation", ("host", "extra_host", "test", "permission", "duplicate", "seed", "alpha_bool",
    "loss", "time_input", "cap", "head", "optimizer", "budget", "deadline", "shared_width", "control_width",
    "normalization", "warmstart", "baseline_retrain", "input_test", "source_hash", "paused", "predecessor"))
def test_scope_mutations_rejected(contract, mutation):
    c = deepcopy(contract)
    if mutation == "host": c["jobs"][0]["host"] = "5090"
    if mutation == "extra_host": c["hosts"]["5090"] = deepcopy(c["hosts"]["5080"])
    if mutation == "test": c["held_out_test_authorized"] = True
    if mutation == "permission": c["execution_authorization"]["real_data_training"] = False
    if mutation == "duplicate": c["jobs"][-1] = deepcopy(c["jobs"][0])
    if mutation == "seed": c["jobs"][0]["seed"] = 32
    if mutation == "alpha_bool": c["jobs"][0]["time_alpha"] = False
    if mutation == "loss": c["common_condition"]["loss_mode"] = "raw_mae"
    if mutation == "time_input": c["jobs"][0]["time_input"] = "raw"
    if mutation == "cap": c["datasets"][0]["cap_value"] = 999
    if mutation == "head": c["datasets"][0]["model"]["hidden_dim"] = 32
    if mutation == "optimizer": c["datasets"][0]["optimizer"]["lr"] = .1
    if mutation == "budget": c["training"]["maximum_epochs"] = 301
    if mutation == "deadline": c["hosts"]["5080"]["absolute_host_deadline_unix"] += 1
    if mutation == "shared_width": c["architecture"]["shared"]["input_width"] = 16
    if mutation == "control_width": c["architecture"]["independent"]["branch_widths"][-1] = 6
    if mutation == "normalization": c["architecture"]["quantity_loss_normalization"] = "cap"
    if mutation == "warmstart": c["architecture"]["checkpoint_warmstart"] = True
    if mutation == "baseline_retrain": c["baseline_validation_references"][0]["retrained"] = True
    if mutation == "input_test": c["input_files"]["inputs/test.parquet"] = "forbidden"
    if mutation == "source_hash": c["source"]["files_sha256"] = "wrong"
    if mutation == "paused": c["hosts"]["5080"]["paused_processes"][1]["state"] = "R"
    if mutation == "predecessor": c["hosts"]["5080"]["completed_predecessor"]["dispatcher_exit_sha256"] = ""
    with pytest.raises((ValueError, KeyError)):
        r.validate_contract(c, "5080")


def time_proof():
    return {"schema": "time_head_quantity_only_verification_v1", "optimizer_steps_checked": 2,
        **{k: "a" * 64 for k in ("initial_time_head_sha256", "final_time_head_sha256", "checkpoint_time_head_sha256")},
        **{k: True for k in ("all_time_head_gradients_none", "optimizer_state_absent", "time_head_parameters_unchanged", "time_nll_is_diagnostic")}}


def make_receipt(c, job, stage):
    root = Path(c["hosts"]["5080"]["root"])
    folder = root / "run" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "selected.pt"
    if not binary.exists(): binary.write_bytes(b"fabricated fixed selection")
    value = {"schema": "titantpp_shared_history_cap_execution_stage_v1", "status": "complete", "job_id": job["id"], "stage": stage,
        "contract_sha256": r.sha_json(c), **{k: job[k] for k in ("dataset", "arm", "seed", "time_alpha", "loss_mode", "time_input")},
        "held_out_test_evaluated": False, "quantity_only": True, "time_head_trained": False, "quantity_loss_normalization": "none",
        "time_head_no_training_verification": time_proof(), "candidate_name": job["arm"],
        "history_relation_sharing": r._MODE_BY_ARM[job["arm"]], "history_relation_parameter_count": 9600,
        "architecture_metadata": {"active_prefix_parameter_counts": r._PREFIX_COUNTS},
        "time_input_transform": "log1p_nonnegative_float32", "quantity_input_transform": "log1p_nonnegative_float32",
        "selected_checkpoint": str(binary), "checkpoint_sha256": r.sha_file(binary), "selected_epoch": 3, "state_sha256": "s" * 64,
        "target_identity": {"target_count": 4}, "files": {str(binary.relative_to(root)): r.sha_file(binary)}}
    if stage == "validation": value["validation_replay_passed"] = True
    r.write(folder / f"{stage}_receipt.json", value, exclusive=True)
    return value


def test_selected_binary_scientific_identity_and_same_target_replay(contract):
    job = contract["jobs"][0]
    make_receipt(contract, job, "train")
    value = make_receipt(contract, job, "validation")
    path = Path(contract["hosts"]["5080"]["root"]) / "run" / job["id"] / "validation_receipt.json"
    assert r.verify_receipt(contract, "5080", job, "validation")["sha256"] == r.sha_file(path)
    for change in ({"history_relation_sharing": "independent"}, {"history_relation_parameter_count": 7680},
                   {"architecture_metadata": {}}, {"target_identity": {}}, {"selected_epoch": 4}, {"files": {}},
                   {"time_head_no_training_verification": {}}, {"validation_replay_passed": False}, {"held_out_test_evaluated": True}):
        r.write(path, {**value, **change})
        with pytest.raises(ValueError): r.verify_receipt(contract, "5080", job, "validation")
    with pytest.raises(ValueError): r.worker_command(contract, "5080", Path("/c"), job, "test")


def test_paused_campaign_read_only_identity_and_gpu_allowlist(contract, monkeypatch):
    identities = {p["pid"]: deepcopy(p) for p in stopped()}
    monkeypatch.setattr(r._impl, "proc_identity", lambda pid: identities.get(pid))
    monkeypatch.setattr(r._impl, "process_parents", lambda: {11: 10, 12: 10, 13: 10})
    monkeypatch.setattr(r._impl, "gpu_snapshot", lambda uuid: {"compute_pids": [11, 12, 13, 100]})
    before = deepcopy(identities)
    assert r.gpu_state(contract, "5080", [100])[0]["compute_pids"] == [11, 12, 13, 100]
    assert identities == before
    with pytest.raises(ValueError): r.gpu_state(contract, "5080")
    identities[11]["state"] = "R"
    with pytest.raises(ValueError): r.gpu_state(contract, "5080", [100])
    identities[11]["state"] = "T"
    monkeypatch.setattr(r._impl, "process_parents", lambda: {11: 10, 12: 10, 13: 10, 14: 11})
    with pytest.raises(ValueError): r.gpu_state(contract, "5080", [100])


def fake_queue(c, monkeypatch, fail=False):
    root = Path(c["hosts"]["5080"]["root"])
    (root / "logs").mkdir()
    jobs = {job["id"]: job for job in c["jobs"]}
    instances, launched, maximum = [], [], [0]
    bad = c["jobs"][0]["id"] if fail else None
    class Child:
        def __init__(self, command, **kwargs):
            self.pid, self.command, self.cwd = 1000 + len(instances), command, str(kwargs["cwd"])
            self.returncode, self.ticks = None, 0
            self.job, self.stage = jobs[command[command.index("--job") + 1]], command[-1]
            assert kwargs["env"]["PYTHONHASHSEED"] == str(self.job["seed"]) and kwargs["start_new_session"] is True
            instances.append(self)
            launched.append((self.stage, self.job["id"]))
            maximum[0] = max(maximum[0], sum(p.returncode is None for p in instances))
        def poll(self):
            if self.returncode is None:
                self.ticks += 1
                if self.ticks >= 4:
                    self.returncode = 1 if self.job["id"] == bad else 0
                    if self.returncode == 0: make_receipt(c, self.job, self.stage)
            return self.returncode
    monkeypatch.setattr(r._impl.subprocess, "Popen", Child)
    monkeypatch.setattr(r._impl, "gpu_state", lambda c, h, pids=(): ({"compute_pids": [11, 12, 13, *pids]}, stopped()))
    monkeypatch.setattr(r._impl, "proc_identity", lambda pid: next(({"pid": p.pid, "pgid": p.pid,
        "start_ticks": p.pid, "argv": p.command, "cwd": p.cwd, "state": "R"}
        for p in instances if p.pid == pid and p.returncode is None), None))
    monkeypatch.setattr(r._impl.time, "sleep", lambda _: None)
    monkeypatch.setattr(r._impl, "predecessor_gate", lambda *a, **kw: {"status": "passed"})
    return launched, maximum


def test_all18_train_then18_validation_concurrency3_and_no_repeat(contract, monkeypatch):
    root = Path(contract["hosts"]["5080"]["root"])
    monkeypatch.chdir(contract["hosts"]["5080"]["source_root"])
    r.write(root / "qualification.json", {"status": "passed", "host": "5080", "contract_sha256": r.sha_json(contract)})
    launched, maximum = fake_queue(contract, monkeypatch)
    (root / "logs").rmdir()
    assert r.dispatch(contract, "5080", root / "execution_contract.json") == 0
    assert [stage for stage, _ in launched] == ["train"] * 18 + ["validation"] * 18
    assert maximum[0] == 3
    final = r.read(root / "final_receipt.json")
    assert final["completed_fit_count"] == final["validation_replay_count"] == 18
    assert final["test_count"] == 0 and final["held_out_test_evaluated"] is False
    assert final["prior_campaign_resumed"] is False and final["owned_worker_pids_absent"] is True
    with pytest.raises(ValueError, match="Existing"): r.dispatch(contract, "5080", root / "execution_contract.json")


def test_failure_drains_only_current3_and_starts_no_more(contract, monkeypatch):
    launched, maximum = fake_queue(contract, monkeypatch, fail=True)
    result = r.run_stage(contract, "5080", Path("/c"), "train", r.time.monotonic())
    assert result["status"] == "failed" and len(launched) == len(result["exits"]) == 3
    assert len(result["pending_job_ids"]) == 15 and maximum[0] == 3


def test_expired_inherited_deadline_zero_launch_and_test_rejected(contract, monkeypatch):
    launched, _ = fake_queue(contract, monkeypatch)
    monkeypatch.setattr(r._impl.time, "time", lambda: r._HOST_DEADLINE + 1)
    result = r.run_stage(contract, "5080", Path("/c"), "train", r.time.monotonic())
    assert result["status"] == "failed" and not launched
    with pytest.raises(ValueError): r.run_stage(contract, "5080", Path("/c"), "test", r.time.monotonic())


def predecessor_fixture(c, monkeypatch):
    root = Path(c["hosts"]["5080"]["completed_predecessor"]["root"])
    parent = deepcopy(c)
    parent["hosts"]["5080"]["root"] = str(root)
    parent["hosts"]["5080"]["source_root"] = str(root / "source")
    cp = root / "execution_contract.json"
    r.write(cp, parent, exclusive=True)
    monkeypatch.setattr(r, "_PARENT_RAW_SHA", r.sha_file(cp))
    binding = c["hosts"]["5080"]["completed_predecessor"]
    binding.update(contract_raw_sha256=r.sha_file(cp), contract_canonical_sha256=r.sha_json(parent), source_files_sha256=parent["source"]["files_sha256"])
    xp = root / "dispatcher_exit.json"
    r.write(xp, {"pid": 500, "actual_os_returncode": 0}, exclusive=True)
    binding["dispatcher_exit_sha256"] = r.sha_file(xp)
    monkeypatch.setattr(r._impl, "process_parents", lambda: {})
    monkeypatch.setattr(r._impl, "proc_identity", lambda pid: None)
    monkeypatch.setattr(r._impl, "gpu_state", lambda *a, **kw: ({"compute_pids": [11, 12, 13]}, stopped()))
    return parent, root


def test_predecessor_only5080_terminal_no_other_host_dependency(contract, monkeypatch):
    parent, root = predecessor_fixture(contract, monkeypatch)
    before = {str(p): r.sha_file(p) for p in root.rglob("*") if p.is_file()}
    result = r.predecessor_gate(contract, "5080", require_complete=True)
    assert result["parent_dispatcher_actual_exit0"] and result["parent_owned_python_pids_absent"]
    assert result["other_host_inspected_or_changed"] is False
    assert before == {str(p): r.sha_file(p) for p in root.rglob("*") if p.is_file()}
    assert "final_receipt.json" not in {p.name for p in root.iterdir()}


@pytest.mark.parametrize("mutation", ("exit_fail", "exit_changed", "owner", "unrecorded_python", "data", "runtime"))
def test_predecessor_gate_rejects_failure_live_owner_and_common_changes(contract, monkeypatch, mutation):
    _, root = predecessor_fixture(contract, monkeypatch)
    if mutation == "exit_fail":
        r.write(root / "dispatcher_exit.json", {"pid": 500, "actual_os_returncode": 1})
        contract["hosts"]["5080"]["completed_predecessor"]["dispatcher_exit_sha256"] = r.sha_file(root / "dispatcher_exit.json")
    if mutation == "exit_changed": r.write(root / "dispatcher_exit.json", {})
    if mutation == "owner": monkeypatch.setattr(r._impl, "proc_identity", lambda pid: {"pid": pid})
    if mutation == "unrecorded_python":
        monkeypatch.setattr(r._impl, "process_parents", lambda: {777: 1})
        monkeypatch.setattr(r._impl, "proc_identity", lambda pid: None if pid == 500 else {"argv": ["python", "worker"], "cwd": str(root / "source")})
    if mutation == "data": contract["datasets"][0]["loader"]["extra_changed"] = True
    if mutation == "runtime": contract["hosts"]["5080"]["runtime_expected"]["cudnn"] = 1
    with pytest.raises(ValueError): r.predecessor_gate(contract, "5080", require_complete=True)


def test_native_qualification_two_arm_memory_and_repeat_block(contract, monkeypatch):
    root = Path(contract["hosts"]["5080"]["root"])
    monkeypatch.chdir(contract["hosts"]["5080"]["source_root"])
    monkeypatch.setattr(r._impl, "gpu_state", lambda *a: ({"compute_pids": [11, 12, 13]}, stopped()))
    monkeypatch.setattr(r._impl, "paused_processes", lambda *a: stopped())
    monkeypatch.setattr(r._impl, "free_gpu_bytes", lambda *a: 1000)
    monkeypatch.setattr(r._impl, "runtime_check", lambda *a: {"runtime": "same"})
    monkeypatch.setattr(r._impl, "predecessor_gate", lambda *a, **kw: {"status": "passed"})
    calls = []
    monkeypatch.setattr(r._impl, "qualify_three_concurrent", lambda *a: calls.append("both") or {"status": "passed"})
    worker = ModuleType(r.WORKER_MODULE)
    worker.synthetic_check = lambda *a: {"status": "passed", "real_data_loaded": False,
        "cuda_memory_peaks": {arm: {"max_memory_reserved_bytes": 100} for arm in r.ARMS}}
    monkeypatch.setitem(sys.modules, r.WORKER_MODULE, worker)
    got = r.qualify(contract, "5080")
    assert calls == ["both"] and got["parallel_memory_admission"]["three_worker_reserved_estimate_bytes"] == 300
    assert r.read(root / "qualification.json")["status"] == "passed"
    with pytest.raises(ValueError): r.qualify(contract, "5080")


def test_real_six_fabricated_local_children_preserve_scope(contract, monkeypatch):
    root, source = (Path(contract["hosts"]["5080"][k]) for k in ("root", "source_root"))
    package = source / "paper/scripts"
    package.mkdir(parents=True)
    (source / "paper/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "run_titantpp_shared_history_cap_execution.py").write_text(
        "import time\ndef synthetic_condition(c,h,m,*,arm):\n time.sleep(.02)\n return {'status':'passed','real_data_loaded':False,'candidate_name':arm,'loss_mode':m,'time_input':'log1p'}\n")
    contract["hosts"]["5080"]["environment"]["PYTHONPATH"] = str(source)
    r.write(root / "execution_contract.json", contract, exclusive=True)
    owner = {"pid": 123, "start_ticks": 456, "pgid": 123, "argv": ["qualification"], "cwd": str(source)}
    monkeypatch.setattr(r, "qualification_identity", lambda *a: deepcopy(owner))
    monkeypatch.setattr(r._impl, "proc_identity", lambda pid: None)
    monkeypatch.setattr(r._impl, "gpu_state", lambda c, h, pids=(): ({"compute_pids": [11, 12, 13, *pids]}, stopped()))
    result = r.qualify_three_concurrent(contract, "5080")
    assert result["native_worker_exit0_count"] == 6 and result["maximum_simultaneous_workers"] == 3
    assert set(result["per_arm"]) == set(r.ARMS) and all(len(v) == 3 for v in result["per_arm"].values())
    assert result["qualification_owner_identity"] == owner
    assert len(list(root.glob("admission_*.json"))) == 6


def test_private_queue_preserves_original_module_and_source_sha():
    from paper.scripts import titantpp_quantity_loss_mlp18_runtime as old
    assert r._impl is not old and old.SCHEMA != r.SCHEMA
    assert old.WORKER_MODULE == "paper.scripts.run_titantpp_quantity_loss_mlp18"
    assert r._impl.run_stage.__globals__ is r._impl.__dict__
    assert r._impl.validate_contract is r.validate_contract
    assert r.sha_file(r._BASE_PATH) == r._BASE_SHA
