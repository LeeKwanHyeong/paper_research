"""Fabricated frozen RAF3 operation checks; no SSH, real input or Test inference."""
from pathlib import Path
import sys
from types import ModuleType

import pytest

from paper.scripts import manage_titantpp_shared_raw_raf3_test as m


@pytest.fixture
def contract(tmp_path, monkeypatch):
    root = tmp_path / "remote"
    (root / "source").mkdir(parents=True)
    c = {"schema": "titantpp_shared_raw_raf3_frozen_v1", "remote_root": str(root), "source_root": "source",
        "native": {"python": sys.executable, "gpu_uuid": "GPU-fabricated", "environment": {}},
        "absolute_host_deadline_unix": m.time.time() + 10000, "child_timeout_seconds": 100,
        "jobs": [{"id": f"raf_shared_seed{seed}", "seed": seed, "selected_epoch": epoch} for seed, epoch in m.EPOCHS.items()]}
    m.write(root / "evaluation_contract.json", c)
    monkeypatch.setattr(m, "_OWNED_CHILDREN", [])
    return c, root


def fake_children(c, root, monkeypatch, *, failed_stage=None, no_exit=False):
    children, launched, maximum = [], [], [0]
    class Child:
        def __init__(self, command, **kwargs):
            self.pid, self.returncode, self.ticks = 1000 + len(children), None, 0
            self.job = next(job for job in c["jobs"] if job["id"] == command[command.index("--job") + 1])
            self.stage = command[command.index("--split") + 1] if "--split" in command else "qualification"
            assert kwargs["start_new_session"] is True and kwargs["cwd"] == root / "source"
            children.append(self); launched.append(self.stage)
            maximum[0] = max(maximum[0], sum(child.returncode is None for child in children))
        def poll(self):
            if self.returncode is None and not no_exit:
                self.ticks += 1
                if self.ticks >= 3:
                    self.returncode = 1 if self.stage == failed_stage else 0
                    if self.returncode == 0:
                        receipt = {"status": "passed" if self.stage == "qualification" else "complete",
                            "job_id": self.job["id"], "contract_sha256": m.canonical(c), "checks": {"target_blocked": True}, "real_data_loaded": False}
                        m.write(m._receipt_path(root, self.job, self.stage), receipt)
            return self.returncode
    monkeypatch.setattr(m.subprocess, "Popen", Child)
    monkeypatch.setattr(m, "gpu_snapshot", lambda uuid: {"compute_pids": [child.pid for child in children if child.returncode is None]})
    monkeypatch.setattr(m, "_owned_pid_absent", lambda pid: all(child.returncode is not None for child in children if child.pid == pid))
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    return children, launched, maximum


def test_three_fresh_process_stages_validation_gate_then_test_only(contract, monkeypatch):
    c, root = contract
    children, launched, maximum = fake_children(c, root, monkeypatch)
    monkeypatch.setattr(m.signal, "signal", lambda *a: None)
    worker = ModuleType(m.WORKER)
    calls = []
    def gate_check(path, actual):
        assert launched == ["qualification"] * 3 + ["validation"] * 3
        gate = m.read(root / "validation_gate.json")
        assert gate["status"] == "PASS" and gate["job_count"] == 3
        assert set(gate["validation_receipts"]) == set(gate["validation_worker_exits"]) == {job["id"] for job in c["jobs"]}
        calls.append("gate")
    worker.verify_validation_gate = gate_check
    monkeypatch.setitem(sys.modules, m.WORKER, worker)
    result = m.remote_run(root)
    assert calls == ["gate"] and launched == ["qualification"] * 3 + ["validation"] * 3 + ["test"] * 3
    assert maximum[0] == 3 and len({child.pid for child in children}) == 9
    assert result["qualification_exit0_count"] == 3 and result["actual_data_worker_exit0_count"] == 6
    assert result["actual_native_worker_exit0_count"] == 9 and result["owned_pids_absent"] is True
    assert m.read(root / "remote_final_receipt.json") == m.read(root / "pipeline_receipt.json")
    for job in c["jobs"]:
        for split in ("validation", "test"):
            exit_record = m.read(root / "results" / job["id"] / f"{split}_exit.json")
            assert exit_record["actual_os_returncode"] == 0 and exit_record["actual_child_exit_observed"] is True
    with pytest.raises(ValueError, match="Existing evaluation attempt"): m.remote_run(root)


@pytest.mark.parametrize("stage", ("qualification", "validation"))
def test_failed_stage_never_starts_test_and_preserves_failure(contract, monkeypatch, stage):
    c, root = contract
    _, launched, maximum = fake_children(c, root, monkeypatch, failed_stage=stage)
    monkeypatch.setattr(m.signal, "signal", lambda *a: None)
    with pytest.raises(ValueError, match="stage failed"): m.remote_run(root)
    assert "test" not in launched and maximum[0] == 3 and (root / "pipeline_failure.json").exists()
    assert not (root / "remote_final_receipt.json").exists()


def test_validation_gate_rejection_prevents_all_test_processes(contract, monkeypatch):
    c, root = contract
    _, launched, _ = fake_children(c, root, monkeypatch)
    monkeypatch.setattr(m.signal, "signal", lambda *a: None)
    worker = ModuleType(m.WORKER)
    worker.verify_validation_gate = lambda *a: (_ for _ in ()).throw(ValueError("Predictions do not reproduce"))
    monkeypatch.setitem(sys.modules, m.WORKER, worker)
    with pytest.raises(ValueError, match="do not reproduce"): m.remote_run(root)
    assert "test" not in launched


def test_timeout_cleanup_only_new_owned_evaluation_children(contract, monkeypatch):
    c, root = contract
    c["child_timeout_seconds"] = 0
    children, launched, _ = fake_children(c, root, monkeypatch, no_exit=True)
    killed = []
    def cleanup(child):
        assert child in children
        killed.append(child.pid); child.returncode = -15
    monkeypatch.setattr(m, "terminate_owned_child", cleanup)
    with pytest.raises(ValueError, match="stage failed"): m.run_remote_stage(root, c, "qualification")
    assert len(killed) == 3 and set(killed) == {child.pid for child in children}
    assert launched == ["qualification"] * 3 and not any(child.returncode is None for child in children)


def test_unknown_gpu_blocks_before_any_new_child(contract, monkeypatch):
    c, root = contract
    monkeypatch.setattr(m, "gpu_snapshot", lambda uuid: {"compute_pids": [99999]})
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Must not launch on occupied GPU"))
    with pytest.raises(ValueError, match="not fresh idle"): m.remote_run(root)
    assert not (root / "results").exists()


def test_wait_is_local_until_raw9_complete_and_existing_finisher_exits(tmp_path, monkeypatch):
    bundle, report, target_report = tmp_path / "parent", tmp_path / "parent-report", tmp_path / "new-report"
    bundle.mkdir(); report.mkdir(); target_report.mkdir()
    m.write(bundle / "execution_contract.json", {"hosts": {"5090": {"absolute_host_deadline_unix": m.time.time() + 1000}}})
    m.write(report / "FINISHER_STARTED.json", {"pid": 123})
    monkeypatch.setattr(m, "TRAIN_BUNDLE", bundle)
    monkeypatch.setattr(m, "TRAIN_REPORT", report)
    monkeypatch.setattr(m, "REPORT", target_report)
    present, calls = [True], []
    monkeypatch.setattr(m, "_pid_present", lambda pid: present[0])
    def sleep(_):
        calls.append("local_wait")
        m.write(bundle / "retrieval/final_receipt.json", {"status": "complete"})
        present[0] = False
    monkeypatch.setattr(m.time, "sleep", sleep)
    monkeypatch.setattr(m, "prepare", lambda: calls.append("prepare_after_complete"))
    monkeypatch.setattr(m, "advance", lambda: calls.append("advance_after_complete") or {"status": "complete"})
    assert m.wait_and_advance()["status"] == "complete"
    assert calls == ["local_wait", "prepare_after_complete", "advance_after_complete"]
    with pytest.raises(FileExistsError): m.wait_and_advance()


@pytest.mark.parametrize("reason", ("parent_failure", "parent_dead", "deadline"))
def test_wait_failure_preserves_parent_and_starts_no_evaluation(tmp_path, monkeypatch, reason):
    bundle, report, target = tmp_path / "parent", tmp_path / "parent-report", tmp_path / "new"
    bundle.mkdir(); report.mkdir(); target.mkdir()
    deadline = m.time.time() - 1 if reason == "deadline" else m.time.time() + 1000
    m.write(bundle / "execution_contract.json", {"hosts": {"5090": {"absolute_host_deadline_unix": deadline}}})
    m.write(report / "FINISHER_STARTED.json", {"pid": 123})
    if reason == "parent_failure": m.write(report / "QUEUE_FAILURE.json", {"status": "failed"})
    before = {str(p): m.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(m, "TRAIN_BUNDLE", bundle); monkeypatch.setattr(m, "TRAIN_REPORT", report); monkeypatch.setattr(m, "REPORT", target)
    monkeypatch.setattr(m, "_pid_present", lambda pid: reason != "parent_dead")
    monkeypatch.setattr(m, "prepare", lambda: pytest.fail("Must not prepare after parent failure"))
    with pytest.raises(ValueError): m.wait_and_advance()
    assert all(m.sha(path) == digest for path, digest in before.items())


def test_unsupported_stage_cannot_train_or_reselect(contract):
    c, root = contract
    for stage in ("train", "resume", "fit", "search"):
        with pytest.raises(ValueError): m.run_remote_stage(root, c, stage)


def test_source_and_transport_primitives_preserved():
    assert m.sha(m._BASE) == m._BASE_SHA
    assert m.EPOCHS == {42: 9, 52: 5, 62: 19}
    assert m.TRAIN_SHA == "4ed35f0054995f51dadb9b1d6227fc9c09194f9414a9f832bb723dc1d1c058f1"


def test_emitted_remote_supervisor_program_has_owned_timeout_and_compiles(contract, tmp_path, monkeypatch):
    c, root = contract
    bundle = tmp_path / "bundle"; bundle.mkdir()
    (bundle / "deployment.tar.gz").write_bytes(b"fabricated tar")
    monkeypatch.setattr(m, "BUNDLE", bundle)
    monkeypatch.setattr(m, "training_complete", lambda: None)
    monkeypatch.setattr(m, "verify_local", lambda **kw: c)
    commands = []
    monkeypatch.setattr(m._transport, "ssh", lambda host, command, **kw: commands.append((host, command)) or object())
    monkeypatch.setattr(m._transport, "save_transport", lambda *a: None)
    monkeypatch.setattr(m, "collect_and_compare", lambda c: {"status": "complete"})
    assert m.advance()["status"] == "complete" and len(commands) == 3
    import shlex
    fresh, supervisor = shlex.split(commands[0][1])[-1], shlex.split(commands[2][1])[-1]
    compile(fresh, "<fresh5090>", "exec"); compile(supervisor, "<owned-supervisor>", "exec")
    assert "p.wait(timeout=limit)" in supervisor and "os.killpg(p.pid,signal.SIGTERM)" in supervisor
    assert "os.killpg(p.pid,signal.SIGKILL)" in supervisor and "supervisor_own_exit_directly_observed" in supervisor
    with pytest.raises(FileExistsError): m.advance()
    assert len(commands) == 3


def test_prepare_copies_original_final_validation_predictions_and_selected_artifacts(tmp_path, monkeypatch):
    project, bundle, report = tmp_path / "project", tmp_path / "bundle", tmp_path / "report"
    parent_bundle, originals = tmp_path / "parent", tmp_path / "originals"
    project.mkdir(); report.mkdir(); parent_bundle.mkdir(); originals.mkdir()
    monkeypatch.setattr(m, "ROOT", project); monkeypatch.setattr(m, "BUNDLE", bundle); monkeypatch.setattr(m, "REPORT", report)
    monkeypatch.setattr(m, "TRAIN_BUNDLE", parent_bundle)
    monkeypatch.setattr(m, "verify_reviewed_implementation", lambda: {"status": "fabricated review boundary passed"})
    monkeypatch.setattr(m, "SPEC", report / "SCIENCE_SPEC.json"); monkeypatch.setattr(m, "REFERENCES", report / "REFERENCES.json")
    source = project / "science.py"; source.write_text("unchanged = True\n")
    for name in m.NEW_SOURCE:
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("new operation\n")
    training = {"source": {"files": {"science.py": m.sha(source)}}, "jobs": [], "datasets": [{"dataset_id": "raf_spare_parts"}],
        "hosts": {"5090": {"root": "/remote/train", "python": "/python", "gpu_uuid": m.GPU_UUID,
            "runtime_expected": {"cudnn": 92000}, "environment": {}, "absolute_host_deadline_unix": m.time.time() + 10000}}, "input_files": {}}
    selected = []
    for seed, epoch in m.EPOCHS.items():
        job = {"id": f"raf_seed{seed}", "dataset": "raf_spare_parts", "seed": seed, "arm": "titantpp_history_mlp_shared15_v1"}
        training["jobs"].append(job)
        folder = originals / "run" / job["id"]; checkpoint_dir = folder / "runs/seed"; checkpoint_dir.mkdir(parents=True)
        binary = checkpoint_dir / "selected.pt"; binary.write_bytes(b"fabricated checkpoint")
        (checkpoint_dir / "history.json").write_text('{"history":[]}\n')
        (checkpoint_dir / "summary.json").write_text('{"original":true}\n')
        first, final = folder / "training_validation_predictions.jsonl.gz", folder / "validation_predictions.jsonl.gz"
        first.write_bytes(b"first gzip bytes"); final.write_bytes(b"final gzip bytes")
        value = {"selected_epoch": epoch, "selected_checkpoint": "/remote/train/" + str(binary.relative_to(originals)),
            "checkpoint_sha256": m.sha(binary), "state_sha256": "a" * 64,
            "validation_reference": {"prediction_path": "/remote/train/" + str(first.relative_to(originals))},
            "files": {str(path.relative_to(originals)): m.sha(path) for path in folder.rglob("*") if path.is_file() and path != final}}
        m.write(folder / "train_receipt.json", value)
        m.write(folder / "validation_receipt.json", {**value, "prediction_path": "/remote/train/" + str(final.relative_to(originals))})
        selected.append({"seed": seed, "checkpoint_sha256": m.sha(binary), "state_sha256": "a" * 64})
    for name in ("train_validation.parquet", "split_manifest.json"):
        relative = "inputs/data/raf_spare_parts/" + name
        path = parent_bundle / "input_snapshot" / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"frozen data")
        training["input_files"][relative] = m.sha(path)
    cp = parent_bundle / "execution_contract.json"; m.write(cp, training)
    monkeypatch.setattr(m, "TRAIN_SHA", m.sha(cp))
    m.write(parent_bundle / "retrieval/final_receipt.json", {"status": "complete"})
    full = project / "raw_full.parquet"; full.write_bytes(b"frozen full RAF")
    m.write(m.SPEC, {"selected_checkpoints": selected, "full_data": {"path": "raw_full.parquet", "sha256": m.sha(full)},
        "populations": {"validation": {"target_count": 1}, "test": {"target_count": 1}}})
    m.write(m.REFERENCES, {"references": [{} for _ in range(21)]})
    monkeypatch.setattr(m, "training_complete", lambda: (training, originals, {}))
    def git(command, **kwargs):
        if command[1] == "ls-files": return "\n".join(m.NEW_SOURCE)
        if command[1] == "diff": return ""
        raise AssertionError(command)
    monkeypatch.setattr(m.subprocess, "check_output", git)
    prepared = m.prepare()
    actual = m.read(bundle / "evaluation_contract.json")
    assert prepared["actual_native_evaluation_count"] == 0 and prepared["candidate_count"] == 3
    assert len(actual["jobs"]) == 3 and actual["new_training_count"] == 0
    assert m.read(bundle / actual["references"]["path"])["references"] == [{}] * 21
    for job in actual["jobs"]:
        assert job["selected_epoch"] == m.EPOCHS[job["seed"]]
        assert (bundle / job["validation_predictions"]["path"]).read_bytes() == b"final gzip bytes"
        assert Path(job["selection_history"]["path"]).parent == Path(job["checkpoint"]["path"]).parent
    with pytest.raises(ValueError, match="Existing evaluation bundle"): m.prepare()


@pytest.mark.parametrize("mutation", ("none", "missing_file", "changed_source", "wrong_digest", "not_passed"))
def test_reviewed_source_tests_and_science_must_match_current_files(tmp_path, monkeypatch, mutation):
    project = tmp_path / "project"
    report = project / "reports" / m.NAME
    report.mkdir(parents=True)
    monkeypatch.setattr(m, "ROOT", project); monkeypatch.setattr(m, "REPORT", report)
    reviewed = {}
    for name in m.REVIEW_FILES:
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("reviewed version\n")
        reviewed[name] = m.sha(path)
    value = {"status": "PASS", "files_reviewed_sha256": reviewed}
    if mutation == "missing_file": reviewed.pop(m.REVIEW_FILES[0])
    if mutation == "changed_source": (project / m.REVIEW_FILES[0]).write_text("later unreviewed edit\n")
    if mutation == "wrong_digest": reviewed[m.REVIEW_FILES[1]] = "0" * 64
    if mutation == "not_passed": value["status"] = "PENDING"
    m.write(report / "INDEPENDENT_QA.json", value)
    if mutation == "none":
        result = m.verify_reviewed_implementation()
        assert result["sha256"] == m.sha(report / "INDEPENDENT_QA.json")
    else:
        with pytest.raises(ValueError): m.verify_reviewed_implementation()


def test_missing_review_stops_prepare_and_followup_before_other_work(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "REPORT", tmp_path)
    monkeypatch.setattr(m, "training_complete", lambda: pytest.fail("Do not inspect training before reviewed-version guard"))
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Do not launch without reviewed version"))
    with pytest.raises(FileNotFoundError): m.prepare()
    with pytest.raises(FileNotFoundError): m.start_followup()


def test_review_can_bind_additional_reference_documents_and_checks_their_sha(tmp_path, monkeypatch):
    project = tmp_path / "project"; report = project / "reports" / m.NAME
    report.mkdir(parents=True)
    monkeypatch.setattr(m, "ROOT", project); monkeypatch.setattr(m, "REPORT", report)
    extra = ("reports/" + m.NAME + "/REFERENCES.json", "reports/" + m.NAME + "/REFERENCES.md")
    reviewed = {}
    for name in (*m.REVIEW_FILES, *extra):
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("reviewed version\n")
        reviewed[name] = m.sha(path)
    m.write(report / "INDEPENDENT_QA.json", {"status": "PASS", "files_reviewed_sha256": reviewed})
    assert m.verify_reviewed_implementation()["sha256"] == m.sha(report / "INDEPENDENT_QA.json")
    (project / extra[0]).write_text("changed reference\n")
    with pytest.raises(ValueError, match="changed after independent review"): m.verify_reviewed_implementation()
