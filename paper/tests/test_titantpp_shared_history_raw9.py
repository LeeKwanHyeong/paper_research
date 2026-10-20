"""Fabricated-only original-data joint raw9 lifecycle and finite follow-up checks."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from paper.scripts import titantpp_shared_history_raw9_runtime as r
from paper.scripts import manage_titantpp_shared_history_raw9 as m
from paper.tests.test_titantpp_shared_history_cap_execution_runtime import contract as cap_fixture, stopped, make_receipt as cap_receipt


@pytest.fixture
def contract(tmp_path):
    c = cap_fixture.__wrapped__(tmp_path)
    c.update(schema=r.SCHEMA, data_variant="original_uncapped", quantity_capping_applied=False,
        predecessor={"contract_path": str(tmp_path / "cap/execution_contract.json"), "contract_raw_sha256": r._CAP_RAW_SHA,
            "receipt_path": str(tmp_path / "cap/retrieval/final_receipt.json"), "expected_count": 18,
            "finisher_receipt_path": str(tmp_path / "cap/FINISHER_STARTED.json"), "failure_path": str(tmp_path / "cap/FINISH_FAILURE.json")})
    c["common_condition"]["time_alpha"] = 1.
    c["architecture"].update(arms=list(r.ARMS), time_alpha=1.)
    for d in c["datasets"]: d.pop("cap_value")
    c["jobs"] = [{**job, "time_alpha": 1.} for job in c["jobs"] if job["arm"] == r.ARMS[0]]
    for ref in c["baseline_validation_references"]: ref.update(time_alpha=1., time_head_trained=True)
    return c


def proof():
    return {"schema": "time_head_joint_training_verification_v1", "initial_time_head_sha256": "a" * 64,
        "final_time_head_sha256": "b" * 64, "checkpoint_time_head_sha256": "c" * 64,
        "time_head_parameter_names": ["time_head.weight"], "optimizer_steps_checked": 2, "joint_loss_observations": 2, "time_alpha": 1.,
        **{key: True for key in ("finite_time_head_gradient_observed", "nonzero_time_head_gradient_observed",
        "time_head_parameters_changed", "optimizer_state_present", "joint_loss_includes_time")}}


def make_receipt(c, job, stage):
    root = Path(c["hosts"]["5080"]["root"])
    folder = root / "run" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "selected.pt"
    if not binary.exists(): binary.write_bytes(b"fabricated selected joint checkpoint")
    value = {"schema": "titantpp_shared_history_raw9_stage_v1", "status": "complete", "job_id": job["id"], "stage": stage,
        "contract_sha256": r.sha_json(c), **{key: job[key] for key in ("dataset", "arm", "seed", "time_alpha", "loss_mode", "time_input")},
        "held_out_test_evaluated": False, "quantity_only": False, "time_head_trained": True, "quantity_loss_normalization": "none",
        "time_head_training_verification": proof(), "candidate_name": job["arm"], "history_relation_sharing": "shared",
        "history_relation_parameter_count": 9600, "architecture_metadata": {"active_prefix_parameter_counts": r._PREFIX_COUNTS},
        "time_input_transform": "log1p_nonnegative_float32", "quantity_input_transform": "log1p_nonnegative_float32",
        "selected_checkpoint": str(binary), "checkpoint_sha256": r.sha_file(binary), "selected_epoch": 3,
        "state_sha256": "d" * 64, "target_identity": {"target_count": 4},
        "files": {str(binary.relative_to(root)): r.sha_file(binary)}}
    if stage == "validation": value["validation_replay_passed"] = True
    r.write(folder / f"{stage}_receipt.json", value, exclusive=True)
    return value


def test_original9_single_shared_arm_joint1_only5080(contract):
    assert len(r.validate_contract(contract, "5080")) == 9
    assert r.STAGES == ("train", "validation") and r.ARMS == ("titantpp_history_mlp_shared15_v1",)
    assert r.WORKER_MODULE == "paper.scripts.run_titantpp_shared_history_raw9"
    assert r.alpha_label(1.) == "1"
    with pytest.raises(ValueError): r.validate_contract(contract, "5090")


@pytest.mark.parametrize("mutation", ("cap", "cap_marker", "alpha0", "alpha_bool", "second_arm", "seed", "job_host",
    "test", "permission", "warmstart", "budget", "deadline", "baseline_alpha0", "baseline_trained", "baseline_retrained",
    "input_count", "input_escape", "test_input", "source_hash", "parent_count", "parent_sha", "width", "head"))
def test_raw9_scope_mutations_rejected(contract, mutation):
    c = deepcopy(contract)
    if mutation == "cap": c["datasets"][0]["cap_value"] = 20
    if mutation == "cap_marker": c["quantity_capping_applied"] = True
    if mutation == "alpha0": c["common_condition"]["time_alpha"] = 0.
    if mutation == "alpha_bool": c["jobs"][0]["time_alpha"] = True
    if mutation == "second_arm": c["jobs"][0]["arm"] = "titantpp_history_mlp_prefix_capacity_v1"
    if mutation == "seed": c["jobs"][0]["seed"] = 32
    if mutation == "job_host": c["jobs"][0]["host"] = "5090"
    if mutation == "test": c["held_out_test_authorized"] = True
    if mutation == "permission": c["execution_authorization"]["real_data_training"] = False
    if mutation == "warmstart": c["architecture"]["checkpoint_warmstart"] = True
    if mutation == "budget": c["training"]["minimum_epochs"] = 1
    if mutation == "deadline": c["hosts"]["5080"]["absolute_host_deadline_unix"] += 1
    if mutation == "baseline_alpha0": c["baseline_validation_references"][0]["time_alpha"] = 0.
    if mutation == "baseline_trained": c["baseline_validation_references"][0]["time_head_trained"] = False
    if mutation == "baseline_retrained": c["baseline_validation_references"][0]["retrained"] = True
    if mutation == "input_count": c["input_files"].pop(next(iter(c["input_files"])))
    if mutation == "input_escape": c["input_files"]["../escape"] = c["input_files"].pop(next(iter(c["input_files"])))
    if mutation == "test_input": c["input_files"]["inputs/data/test.parquet"] = c["input_files"].pop(next(iter(c["input_files"])))
    if mutation == "source_hash": c["source"]["files_sha256"] = "changed"
    if mutation == "parent_count": c["predecessor"]["expected_count"] = 6
    if mutation == "parent_sha": c["predecessor"]["contract_raw_sha256"] = "changed"
    if mutation == "width": c["architecture"]["shared"]["input_width"] = 16
    if mutation == "head": c["datasets"][0]["model"]["hidden_dim"] = 32
    with pytest.raises((ValueError, KeyError)): r.validate_contract(c, "5080")


@pytest.mark.parametrize("key", ("initial_time_head_sha256", "final_time_head_sha256", "checkpoint_time_head_sha256",
    "finite_time_head_gradient_observed", "nonzero_time_head_gradient_observed", "time_head_parameters_changed", "optimizer_state_present",
    "joint_loss_includes_time", "optimizer_steps_checked", "joint_loss_observations", "time_head_parameter_names", "time_alpha"))
def test_joint_actual_training_evidence_required(key):
    value = proof()
    assert r.verify_time_head_training(value) == value
    value.pop(key)
    with pytest.raises(ValueError): r.verify_time_head_training(value)


def test_joint_receipt_selected_binary_same_population_replay_and_local_originals(contract, tmp_path):
    job = contract["jobs"][0]
    make_receipt(contract, job, "train")
    value = make_receipt(contract, job, "validation")
    path = Path(contract["hosts"]["5080"]["root"]) / "run" / job["id"] / "validation_receipt.json"
    assert r.verify_receipt(contract, "5080", job, "validation")["sha256"] == r.sha_file(path)
    import shutil
    originals = tmp_path / "originals"
    shutil.copytree(Path(contract["hosts"]["5080"]["root"]), originals)
    assert r.verify_receipt(contract, "5080", job, "validation", originals=originals)["sha256"] == r.sha_file(path)
    for changed in ({"quantity_only": True}, {"time_head_trained": False}, {"time_head_training_verification": {}},
        {"target_identity": {}}, {"selected_epoch": 4}, {"validation_replay_passed": False}, {"files": {}}):
        r.write(path, {**value, **changed})
        with pytest.raises(ValueError): r.verify_receipt(contract, "5080", job, "validation")
    with pytest.raises(ValueError): r.worker_command(contract, "5080", Path("/c"), job, "test")


def fake_queue(c, monkeypatch, fail=False):
    root = Path(c["hosts"]["5080"]["root"])
    (root / "logs").mkdir()
    jobs = {job["id"]: job for job in c["jobs"]}
    children, launched, maximum = [], [], [0]
    class Child:
        def __init__(self, command, **kwargs):
            self.pid, self.command, self.cwd, self.returncode, self.ticks = 1000 + len(children), command, str(kwargs["cwd"]), None, 0
            self.job, self.stage = jobs[command[command.index("--job") + 1]], command[-1]
            children.append(self); launched.append((self.stage, self.job["id"]))
            maximum[0] = max(maximum[0], sum(p.returncode is None for p in children))
        def poll(self):
            if self.returncode is None:
                self.ticks += 1
                if self.ticks >= 4:
                    self.returncode = 1 if fail and self.job["id"] == c["jobs"][0]["id"] else 0
                    if self.returncode == 0: make_receipt(c, self.job, self.stage)
            return self.returncode
    monkeypatch.setattr(r._impl.subprocess, "Popen", Child)
    monkeypatch.setattr(r._impl, "gpu_state", lambda c, h, pids=(): ({"compute_pids": [11, 12, 13, *pids]}, stopped()))
    monkeypatch.setattr(r._impl, "proc_identity", lambda pid: next(({"pid": p.pid, "pgid": p.pid, "start_ticks": p.pid,
        "argv": p.command, "cwd": p.cwd, "state": "R"} for p in children if p.pid == pid and p.returncode is None), None))
    monkeypatch.setattr(r._impl.time, "sleep", lambda _: None)
    monkeypatch.setattr(r._impl, "predecessor_gate", lambda *a, **kw: {"status": "complete"})
    return launched, maximum


def test9train_then9_validation_max3_no_restarting(contract, monkeypatch):
    root = Path(contract["hosts"]["5080"]["root"])
    monkeypatch.chdir(contract["hosts"]["5080"]["source_root"])
    r.write(root / "qualification.json", {"status": "passed", "host": "5080", "contract_sha256": r.sha_json(contract)})
    launched, maximum = fake_queue(contract, monkeypatch)
    (root / "logs").rmdir()
    assert r.dispatch(contract, "5080", root / "execution_contract.json") == 0
    assert [stage for stage, _ in launched] == ["train"] * 9 + ["validation"] * 9 and maximum[0] == 3
    final = r.read(root / "final_receipt.json")
    assert final["completed_fit_count"] == final["validation_replay_count"] == 9 and final["test_count"] == 0
    assert final["prior_campaign_resumed"] is False
    with pytest.raises(ValueError): r.dispatch(contract, "5080", root / "execution_contract.json")


def test_raw9_failure_no_retry_pending6_preserved(contract, monkeypatch):
    launched, maximum = fake_queue(contract, monkeypatch, fail=True)
    result = r.run_stage(contract, "5080", Path("/c"), "train", r.time.monotonic())
    assert result["status"] == "failed" and len(launched) == len(result["exits"]) == 3
    assert len(result["pending_job_ids"]) == 6 and maximum[0] == 3


def test_single_arm_native3_admission_count(monkeypatch, contract):
    calls = []
    def old(c, h):
        calls.append((c, h))
        return {"per_arm": {r.ARMS[0]: [{"actual_os_returncode": 0}] * 3}, "native_worker_exit0_count": 6}
    monkeypatch.setattr(r, "_original_three", old)
    got = r.qualify_three_concurrent(contract, "5080")
    assert got["native_worker_exit0_count"] == 3 and calls == [(contract, "5080")]


def completed_cap(c, tmp_path, monkeypatch):
    from paper.scripts import titantpp_shared_history_cap_execution_v3_runtime as parent
    old = cap_fixture.__wrapped__(tmp_path / "cap-parent")
    root = Path(old["hosts"]["5080"]["root"])
    cp = root / "execution_contract.json"
    r.write(cp, old, exclusive=True)
    monkeypatch.setattr(r, "_CAP_ROOT", str(root))
    monkeypatch.setattr(r, "_CAP_RAW_SHA", r.sha_file(cp))
    monkeypatch.setattr(r, "_CAP_CANONICAL_SHA", r.sha_json(old))
    c["predecessor"]["contract_raw_sha256"] = r.sha_file(cp)
    c["hosts"]["5080"].update({key: deepcopy(old["hosts"]["5080"][key]) for key in
        ("python", "gpu_uuid", "runtime_expected", "paused_processes", "absolute_host_deadline_unix")})
    stages, pid = {}, 500
    for stage in parent.STAGES:
        exits, receipts = {}, {}
        for job in old["jobs"]:
            cap_receipt(old, job, stage)
            receipts[job["id"]] = parent.verify_receipt(old, "5080", job, stage)
            exits[job["id"]] = {"pid": pid, "actual_os_returncode": 0, "timeout": False}; pid += 1
        stages[stage] = {"status": "complete", "exits": exits, "receipts": receipts}
    inventory = {str(path.relative_to(root)): r.sha_file(path) for path in root.rglob("*") if path.is_file() and path.relative_to(root).parts[0] != "source"}
    mp = root / "artifact_manifest.json"
    r.write(mp, {"files": inventory}, exclusive=True)
    final = {"status": "complete", "host": "5080", "contract_sha256": r.sha_json(old), "completed_fit_count": 18,
        "validation_replay_count": 18, "test_count": 0, "held_out_test_evaluated": False, "owned_worker_pids_absent": True,
        "stages": stages, "artifact_manifest": {"path": "artifact_manifest.json", "sha256": r.sha_file(mp), "file_count": len(inventory)}}
    r.write(root / "final_receipt.json", final, exclusive=True)
    r.write(root / "dispatcher_exit.json", {"pid": 999, "actual_os_returncode": 0}, exclusive=True)
    monkeypatch.setattr(r._impl, "proc_identity", lambda pid: None)
    monkeypatch.setattr(r._impl, "gpu_state", lambda *a: ({"compute_pids": [11, 12, 13]}, stopped()))
    monkeypatch.setattr(parent, "predecessor_process_candidates", lambda root: {})
    return root, final


def test_cap18_all36_exits_fullmanifest_fresh_absence_and_no_old_mutation(contract, tmp_path, monkeypatch):
    root, _ = completed_cap(contract, tmp_path, monkeypatch)
    before = {str(path): r.sha_file(path) for path in root.rglob("*") if path.is_file()}
    gate = r.predecessor_gate(contract, "5080", require_complete=True)
    assert gate["parent_completed_fit_count"] == gate["parent_validation_replay_count"] == 18
    assert gate["parent_owned_pids_absent"] is True
    assert before == {str(path): r.sha_file(path) for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("mutation", ("fit_partial", "validation_partial", "worker_fail", "dispatcher_fail", "manifest_empty", "owner_alive", "artifact_changed"))
def test_cap18_incomplete_failure_or_changed_original_blocks_raw9(contract, tmp_path, monkeypatch, mutation):
    root, final = completed_cap(contract, tmp_path, monkeypatch)
    if mutation == "fit_partial": final["completed_fit_count"] = 17
    if mutation == "validation_partial": final["validation_replay_count"] = 17
    if mutation == "worker_fail": next(iter(final["stages"]["train"]["exits"].values()))["actual_os_returncode"] = 1
    if mutation == "dispatcher_fail": r.write(root / "dispatcher_exit.json", {"pid": 999, "actual_os_returncode": 1})
    if mutation == "manifest_empty":
        r.write(root / "artifact_manifest.json", {"files": {}})
        final["artifact_manifest"].update(sha256=r.sha_file(root / "artifact_manifest.json"), file_count=0)
    if mutation == "owner_alive": monkeypatch.setattr(r._impl, "proc_identity", lambda pid: {"pid": pid})
    if mutation == "artifact_changed": next((root / "run").glob("*/selected.pt")).write_bytes(b"changed")
    r.write(root / "final_receipt.json", final)
    with pytest.raises(ValueError): r.predecessor_gate(contract, "5080", require_complete=True)


def queue_paths(c, monkeypatch, tmp_path):
    report = tmp_path / "raw-report"
    report.mkdir()
    monkeypatch.setattr(m, "REPORT", report)
    monkeypatch.setattr(m, "verify_local", lambda **kw: c)
    return report


def test_waits_existing_finisher_then_one_verify_deploy_finish(contract, monkeypatch, tmp_path):
    report = queue_paths(contract, monkeypatch, tmp_path)
    predecessor = contract["predecessor"]
    marker = Path(predecessor["finisher_receipt_path"])
    m.write(marker, {"pid": 123, "command": ["existing", "finish"]})
    calls = []
    monkeypatch.setattr(m, "_finisher_alive", lambda _: True)
    def sleep(_):
        calls.append("wait")
        m.write(predecessor["receipt_path"], {"status": "complete"})
    monkeypatch.setattr(m.time, "sleep", sleep)
    monkeypatch.setattr(m, "deploy", lambda: calls.append("deploy") or {"status": "launched"})
    monkeypatch.setattr(m, "finish", lambda: calls.append("finish") or {"status": "complete"})
    assert m.wait_predecessor()["status"] == "complete" and calls == ["wait", "deploy", "finish"]
    assert m.read(report / "QUEUE_COMPLETE.json")["recurring_automation"] is False
    with pytest.raises(FileExistsError): m.wait_predecessor()


@pytest.mark.parametrize("reason", ("failed", "dead", "missing", "expired", "partial_receipt"))
def test_waiter_never_deploys_on_parent_failure_dead_process_or_deadline(contract, monkeypatch, tmp_path, reason):
    queue_paths(contract, monkeypatch, tmp_path)
    parent = contract["predecessor"]
    if reason != "missing": m.write(parent["finisher_receipt_path"], {"pid": 123, "command": ["old", "finish"]})
    if reason == "failed": m.write(parent["failure_path"], {"status": "failed"})
    if reason == "partial_receipt": m.write(parent["receipt_path"], {"status": "partial"})
    if reason == "expired": monkeypatch.setattr(m.time, "time", lambda: r._HOST_DEADLINE + 1)
    monkeypatch.setattr(m, "_finisher_alive", lambda _: reason != "dead")
    monkeypatch.setattr(m, "deploy", lambda: pytest.fail("Deployment must not be attempted"))
    with pytest.raises(ValueError): m.wait_predecessor()


def test_collection_and_fresh_preflight_programs_compile_with_correct_module(contract, monkeypatch):
    compile(m.collection_program(contract, "5080"), "<raw9-collection>", "exec")
    assert "titantpp_shared_history_raw9_runtime as r" in m.collection_program(contract, "5080")
    programs = []
    monkeypatch.setattr(m._impl, "ssh", lambda host, command: programs.append(command) or object())
    monkeypatch.setattr(m._impl, "save_transport", lambda *a: json.dumps({"status": "passed", "cap18_owned_pids_absent": True}))
    # Capture the embedded -c program through shlex and compile without running it.
    monkeypatch.setattr(m, "write", lambda *a: None)
    assert m.fresh_preflight(contract)["status"] == "passed"
    import shlex
    words = shlex.split(programs[0])
    compile(words[-1], "<raw9-preflight>", "exec")


@pytest.mark.parametrize("marker", ("partial_collection", "final_receipt", "root_final_receipt", "queue_failure", "queue_complete", "finish_started"))
def test_finish_existing_attempt_partial_failure_or_complete_stops_before_ssh(tmp_path, monkeypatch, marker):
    bundle, report = tmp_path / "raw-bundle", tmp_path / "raw-report"
    bundle.mkdir(); report.mkdir()
    monkeypatch.setattr(m, "BUNDLE", bundle)
    monkeypatch.setattr(m, "REPORT", report)
    paths = {"partial_collection": bundle / "retrieval/5080", "final_receipt": bundle / "retrieval/final_receipt.json",
        "root_final_receipt": bundle / "final_receipt.json", "queue_failure": report / "QUEUE_FAILURE.json",
        "queue_complete": report / "QUEUE_COMPLETE.json", "finish_started": report / "FINISH_STARTED.json"}
    path = paths[marker]
    if marker == "partial_collection": path.mkdir(parents=True)
    else: m.write(path, {"preserved": True})
    before = {str(p): m.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(m, "verify_local", lambda **kwargs: pytest.fail("An existing attempt must stop before verification/SSH"))
    monkeypatch.setattr(m._impl, "ssh", lambda *a, **kw: pytest.fail("No repeated remote wait is allowed"))
    with pytest.raises(ValueError, match="no repeated remote work"): m.finish()
    assert before == {str(p): m.sha(p) for p in tmp_path.rglob("*") if p.is_file()}


def test_first_finish_is_allowed_once_and_records_intent_before_remote_work(contract, tmp_path, monkeypatch):
    bundle, report = tmp_path / "raw-bundle", tmp_path / "raw-report"
    bundle.mkdir(); report.mkdir()
    monkeypatch.setattr(m, "BUNDLE", bundle)
    monkeypatch.setattr(m, "REPORT", report)
    calls = []
    monkeypatch.setattr(m, "verify_local", lambda **kwargs: calls.append("verify_complete") or contract)
    def ssh(*args, **kwargs):
        assert m.read(report / "FINISH_STARTED.json")["automatic_retry"] is False
        calls.append("wait_once")
        return object()
    monkeypatch.setattr(m._impl, "ssh", ssh)
    monkeypatch.setattr(m._impl, "save_transport", lambda *a: calls.append("preserve_transport"))
    monkeypatch.setattr(m._impl, "collect", lambda host: calls.append("collect_once") or {"status": "complete"})
    comparator = ModuleType("paper.scripts.compare_titantpp_shared_history_raw9")
    comparator.main = lambda: calls.append("compare_once")
    monkeypatch.setitem(sys.modules, comparator.__name__, comparator)
    assert m.finish()["status"] == "complete"
    assert calls == ["verify_complete", "wait_once", "preserve_transport", "collect_once", "compare_once"]
    with pytest.raises(ValueError, match="no repeated remote work"): m.finish()
    assert calls == ["verify_complete", "wait_once", "preserve_transport", "collect_once", "compare_once"]


def test_reuses_frozen_queue_without_changing_cap_runtime():
    from paper.scripts import titantpp_shared_history_cap_execution_v3_runtime as cap
    assert r._impl is not cap._impl and cap.SCHEMA == "titantpp_shared_history_cap_execution_v1"
    assert r._impl.run_stage.__globals__ is r._impl.__dict__
    assert r._impl.qualify is r._adapter.qualify and r._adapter.validate_contract is r.validate_contract
    assert r.sha_file(r._PATH) == r._SHA and m.sha(m._BASE) == m._BASE_SHA
