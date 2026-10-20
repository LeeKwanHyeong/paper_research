"""Fabricated selected-fit collection and frozen evaluation; no SSH or real data."""
from pathlib import Path
import io
import shlex
import sys
import tarfile
from types import ModuleType

import pytest

from paper.scripts import manage_titantpp_shared_raf_taxi6_test as m


@pytest.fixture
def contract(tmp_path, monkeypatch):
    root = tmp_path / "remote"; (root / "source").mkdir(parents=True)
    c = {"schema": "titantpp_shared_raf_taxi6_frozen_v1", "host": "5080", "remote_root": str(root), "source_root": "source",
        "native": {"python": sys.executable, "gpu_uuid": "GPU-fabricated", "environment": {},
            "paused_processes": [{"pid": p} for p in (700, 701, 702, 703)]},
        "cap_stop": {"owned_pids": [800, 801]}, "absolute_host_deadline_unix": m.time.time() + 10000,
        "child_timeout_seconds": 100, "jobs": [{"id": f"{d}_seed{s}", "dataset": d, "seed": s,
            "selected_epoch": 9} for d in ("raf_spare_parts", "yellow_trip_hourly") for s in (42, 52, 62)]}
    m.write(root / "evaluation_contract.json", c)
    monkeypatch.setattr(m, "_OWNED_CHILDREN", [])
    monkeypatch.setattr(m, "_paused", lambda c: c["native"]["paused_processes"])
    return c, root


def fake_children(c, root, monkeypatch, *, failed_stage=None, no_exit=False):
    children, launched, maximum = [], [], [0]
    class Child:
        def __init__(self, command, **kwargs):
            self.pid, self.returncode, self.ticks = 1000 + len(children), None, 0
            self.job = next(j for j in c["jobs"] if j["id"] == command[command.index("--job") + 1])
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
                        m.write(m._receipt_path(root, self.job, self.stage), {
                            "status": "passed" if self.stage == "qualification" else "complete", "job_id": self.job["id"],
                            "contract_sha256": m.canonical(c), "checks": {"target_blocked": True}, "real_data_loaded": False})
            return self.returncode
    monkeypatch.setattr(m.subprocess, "Popen", Child)
    def snapshot(uuid):
        return {"compute_pids": [701, 702, 703] + [p.pid for p in children if p.returncode is None]}
    monkeypatch.setattr(m, "gpu_snapshot", snapshot); monkeypatch.setattr(m, "_plain_gpu_snapshot", snapshot)
    monkeypatch.setattr(m, "_owned_pid_absent", lambda pid: all(p.returncode is not None for p in children if p.pid == pid))
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    return children, launched, maximum


def test_six_per_stage_maximum_three_and_whole_validation_gate(contract, monkeypatch):
    c, root = contract
    children, launched, maximum = fake_children(c, root, monkeypatch)
    monkeypatch.setattr(m.signal, "signal", lambda *a: None)
    worker = ModuleType(m.WORKER); gates = []
    def gate(path, actual):
        assert launched == ["qualification"] * 6 + ["validation"] * 6
        value = m.read(root / "validation_gate.json")
        assert value["job_count"] == 6 and value["schema"] == "titantpp_shared_raf_taxi6_validation_gate_v1"
        assert set(value["validation_receipts"]) == set(value["validation_worker_exits"]) == {j["id"] for j in c["jobs"]}
        gates.append(True)
    worker.verify_validation_gate = gate; monkeypatch.setitem(sys.modules, m.WORKER, worker)
    result = m.remote_run(root)
    assert gates == [True] and launched == ["qualification"] * 6 + ["validation"] * 6 + ["test"] * 6
    assert maximum[0] == 3 and len({p.pid for p in children}) == 18
    assert result["qualification_exit0_count"] == 6 and result["actual_data_worker_exit0_count"] == 12
    assert result["actual_native_worker_exit0_count"] == 18 and result["whole_raw9_complete"] is False
    assert result["owned_pids_absent"] is result["owned_gpu_pids_absent"] is result["paused_external_preserved"] is True
    assert result["gpu_compute_pids"] == [701, 702, 703]
    assert m.read(root / "remote_final_receipt.json") == m.read(root / "pipeline_receipt.json")
    for job in c["jobs"]:
        for split in ("validation", "test"):
            exited = m.read(root / "results" / job["id"] / f"{split}_exit.json")
            assert exited["actual_os_returncode"] == 0 and exited["actual_child_exit_observed"] is True
    with pytest.raises(ValueError, match="Existing evaluation attempt"): m.remote_run(root)


@pytest.mark.parametrize("stage", ("qualification", "validation"))
def test_failure_preserves_attempt_and_never_starts_test(contract, monkeypatch, stage):
    c, root = contract; _, launched, maximum = fake_children(c, root, monkeypatch, failed_stage=stage)
    monkeypatch.setattr(m.signal, "signal", lambda *a: None)
    with pytest.raises(ValueError, match="stage failed"): m.remote_run(root)
    assert "test" not in launched and maximum[0] == 3 and (root / "pipeline_failure.json").exists()
    assert not (root / "remote_final_receipt.json").exists()


def test_prediction_gate_failure_blocks_test(contract, monkeypatch):
    c, root = contract; _, launched, _ = fake_children(c, root, monkeypatch)
    monkeypatch.setattr(m.signal, "signal", lambda *a: None)
    worker = ModuleType(m.WORKER)
    worker.verify_validation_gate = lambda *a: (_ for _ in ()).throw(ValueError("Prediction replay differs"))
    monkeypatch.setitem(sys.modules, m.WORKER, worker)
    with pytest.raises(ValueError, match="replay differs"): m.remote_run(root)
    assert "test" not in launched


def test_timeout_cleanup_only_new_children(contract, monkeypatch):
    c, root = contract; c["child_timeout_seconds"] = 0
    children, launched, _ = fake_children(c, root, monkeypatch, no_exit=True); killed = []
    def stop(child):
        assert child in children and child.pid not in {700, 701, 702, 703, 800, 801}
        killed.append(child.pid); child.returncode = -15
    monkeypatch.setattr(m, "terminate_owned_child", stop)
    with pytest.raises(ValueError, match="stage failed"): m.run_remote_stage(root, c, "qualification")
    assert len(killed) == 3 and launched == ["qualification"] * 3


def test_unknown_gpu_and_remaining_cap_block_new_children(contract, monkeypatch):
    c, root = contract
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Unexpected native launch"))
    monkeypatch.setattr(m, "_plain_gpu_snapshot", lambda _: {"compute_pids": [701, 999]})
    with pytest.raises(ValueError, match="Unknown active5080"): m.remote_run(root)
    assert not (root / "results").exists()


def test_known_paused_cuda_contexts_are_preserved_in_raw_observation(contract, monkeypatch):
    c, _ = contract
    monkeypatch.setattr(m, "_plain_gpu_snapshot", lambda _: {"compute_pids": [701, 702, 703]})
    assert m._gpu_allowed(c)["compute_pids"] == [701, 702, 703]
    monkeypatch.setattr(m, "_paused", lambda _: (_ for _ in ()).throw(ValueError("Stopped identity changed")))
    with pytest.raises(ValueError, match="identity changed"): m._gpu_allowed(c)


def test_readonly_selected_collection_program_has_no_inter_dependency_or_remote_write():
    jobs = [{"id": f"{d}_seed{s}", "dataset": d, "seed": s} for d in ("raf_spare_parts", "yellow_trip_hourly") for s in (42, 52, 62)]
    program = m.subset_collection_program({"hosts": {"5090": {"root": "/original/raw9"}}}, jobs)
    compile(program, "<readonly-selected-collection>", "exec")
    assert "intermittent" not in program and "'cuda'" not in program and "map_location='cpu'" in program
    assert "r.verify_receipt(c,'5090',job,'train')" in program and "train_exit.json" in program
    assert "('job_id','stage','pid','start_ticks')" in program and "original_exit" in program and "original_launch" in program
    assert "r.write" not in program and ".mkdir" not in program and "run_stage" not in program and "torch.save" not in program


def test_new_5080_scope_and_preserved_dependencies():
    assert m.REMOTE.endswith(m.NAME + "_5080") and m.MODULE.endswith("manage_titantpp_shared_raf_taxi6_test")
    assert m.WORKER.endswith("evaluate_titantpp_shared_raf_taxi6_frozen") and len(m.NEW_SOURCE) == 3
    assert len(m.PRESERVED_OPERATIONS) == 3 and all(m.sha(m.ROOT / name) == digest for name, digest in m.PRESERVED_OPERATIONS.items())
    assert m.sha(m._FROZEN_PATH) == m._FROZEN_SHA


@pytest.mark.parametrize("mutation", ("none", "nonzero_exit", "live_pid", "launch_identity", "tensor", "epoch"))
def test_selected_collection_executes_readonly_and_rejects_invalid_fit(tmp_path, monkeypatch, mutation):
    root = tmp_path / "original"; root.mkdir()
    job = {"id": "raf_seed42", "seed": 42}; folder = root / "run" / job["id"]; folder.mkdir(parents=True)
    checkpoint = folder / "selected.pt"; checkpoint.write_bytes(b"frozen selected tensor")
    m.write(folder / "history.json", {"history": [{"epoch": 9}]})
    m.write(folder / "summary.json", {"immutable": True})
    prediction = folder / "training_validation_predictions.jsonl.gz"; prediction.write_bytes(b"unchanged predictions")
    receipt = {"selected_checkpoint": str(checkpoint), "selected_epoch": 9, "state_sha256": "tensor-digest",
        "files": {str(p.relative_to(root)): m.sha(p) for p in folder.iterdir()}}
    m.write(folder / "train_receipt.json", receipt)
    launched = {"job_id": job["id"], "stage": "train", "pid": 123, "start_ticks": 456,
        "command": ["/python", "-m", "scientific_worker", "--job", job["id"], "--stage", "train"]}
    exited = {k: launched[k] for k in ("job_id", "stage", "pid", "start_ticks")}
    exited.update(actual_os_returncode=1 if mutation == "nonzero_exit" else 0, timeout=False)
    if mutation == "launch_identity": exited["start_ticks"] += 1
    m.write(folder / "train_launch.json", launched); m.write(folder / "train_exit.json", exited)
    m.write(root / "execution_contract.json", {"unchanged": True})
    monkeypatch.setattr(m, "TRAIN_SHA", m.sha(root / "execution_contract.json"))
    before = {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}
    runtime = ModuleType("paper.scripts.titantpp_shared_history_raw9_5090_runtime")
    runtime.require, runtime.read, runtime.sha_file, runtime.local_path = m.require, m.read, m.sha, m.local_path
    runtime.verify_receipt = lambda *a: None
    runtime.proc_identity = lambda _: {"pid": 123} if mutation == "live_pid" else None
    loader = ModuleType("simple_lab_test.search.common.runner")
    cpu_calls = []
    def load(path, *, map_location):
        cpu_calls.append(map_location)
        return {"model_state_dict": {"frozen": True}, "model_state_sha256": "tensor-digest", "seed": 42,
            "best_epoch": 10 if mutation == "epoch" else 9}
    loader.torch_load_checkpoint = load
    loader.canonical_state_dict_sha256 = lambda _: "wrong-tensor" if mutation == "tensor" else "tensor-digest"
    scientific = ModuleType("paper.scripts.run_titantpp_shared_history_raw9_5090")
    scientific.prepared = type("P", (), {"select_best_epoch": staticmethod(lambda history: 9)})
    for module in (runtime, loader, scientific): monkeypatch.setitem(sys.modules, module.__name__, module)
    import paper.scripts
    monkeypatch.setattr(paper.scripts, "titantpp_shared_history_raw9_5090_runtime", runtime, raising=False)
    monkeypatch.setattr(paper.scripts, "run_titantpp_shared_history_raw9_5090", scientific, raising=False)
    output = io.BytesIO(); monkeypatch.setattr(sys, "stdout", type("Output", (), {"buffer": output})())
    program = m.subset_collection_program({"hosts": {"5090": {"root": str(root)}}}, [job])
    if mutation == "none":
        exec(compile(program, "<readonly-selected-originals>", "exec"), {})
        with tarfile.open(fileobj=io.BytesIO(output.getvalue()), mode="r:gz") as archive:
            derived = m.json.load(archive.extractfile("verified_exits/raf_seed42.json"))
            assert derived["actual_os_returncode"] == 0 and derived["command"] == launched["command"]
            assert derived["original_exit"]["sha256"] == m.sha(folder / "train_exit.json")
            assert "run/raf_seed42/train_launch.json" in archive.getnames()
        assert cpu_calls == ["cpu"]
    else:
        with pytest.raises(ValueError): exec(compile(program, "<invalid-selected-originals>", "exec"), {})
        assert not output.getvalue()
    assert {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()} == before


def test_unsupported_stage_cannot_train_resume_search(contract):
    c, root = contract
    for stage in ("train", "resume", "fit", "search"):
        with pytest.raises(ValueError): m.run_remote_stage(root, c, stage)


def test_emitted_supervisor_has_owned_timeout_and_5080_only(contract, tmp_path, monkeypatch):
    c, _ = contract; bundle = tmp_path / "bundle"; bundle.mkdir()
    (bundle / "deployment.tar.gz").write_bytes(b"fabricated archive")
    monkeypatch.setattr(m, "BUNDLE", bundle); monkeypatch.setattr(m, "verify_selected_subset", lambda: c)
    monkeypatch.setattr(m, "verify_local", lambda **kw: c)
    commands = []
    monkeypatch.setattr(m._transport, "ssh", lambda host, command, **kw: commands.append((host, command)) or object())
    monkeypatch.setattr(m._transport, "save_transport", lambda *a: None)
    monkeypatch.setattr(m, "collect_and_compare", lambda c: {"status": "complete"})
    assert m.advance()["status"] == "complete" and len(commands) == 3 and {x[0] for x in commands} == {"5080"}
    fresh, supervisor = shlex.split(commands[0][1])[-1], shlex.split(commands[2][1])[-1]
    compile(fresh, "<fresh5080>", "exec"); compile(supervisor, "<owned-timeout>", "exec")
    assert "p.wait(timeout=limit)" in supervisor and "os.killpg(p.pid,signal.SIGTERM)" in supervisor
    assert "os.killpg(p.pid,signal.SIGKILL)" in supervisor and "supervisor_own_exit_directly_observed" in supervisor
    assert "allowed={701, 702, 703}" in fresh
    with pytest.raises(FileExistsError): m.advance()
    assert len(commands) == 3


def test_prepare_collects_only_six_completed_training_references(tmp_path, monkeypatch):
    project, bundle, report, parent = (tmp_path / x for x in ("project", "bundle", "report", "parent"))
    project.mkdir(); report.mkdir(); parent.mkdir()
    monkeypatch.setattr(m, "ROOT", project); monkeypatch.setattr(m, "BUNDLE", bundle); monkeypatch.setattr(m, "REPORT", report)
    monkeypatch.setattr(m, "TRAIN_BUNDLE", parent); monkeypatch.setattr(m, "SPEC", report / "SCIENCE_SPEC.json")
    monkeypatch.setattr(m, "REFERENCES", report / "REFERENCES.json"); monkeypatch.setattr(m, "verify_reviewed_implementation", lambda: None)
    source = project / "science.py"; source.write_text("unchanged = True\n")
    deps = {}
    for name in (*m.NEW_SOURCE, *m.PRESERVED_OPERATIONS):
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("fabricated operation\n")
        if name in m.PRESERVED_OPERATIONS: deps[name] = m.sha(path)
    monkeypatch.setattr(m, "PRESERVED_OPERATIONS", deps)
    originals = tmp_path / "originals"; originals.mkdir()
    training = {"source": {"files": {"science.py": m.sha(source)}}, "jobs": [], "hosts": {"5090": {
        "root": "/old/raw9", "source_root": "/old/raw9/source", "python": "/python"}}, "input_files": {}}
    selected, datasets = [], {}
    for dataset in ("raf_spare_parts", "yellow_trip_hourly"):
        for seed in (42, 52, 62):
            job = {"id": f"{dataset}_seed{seed}", "dataset": dataset, "seed": seed, "arm": "shared"}
            training["jobs"].append(job); folder = originals / "run" / job["id"]; cpdir = folder / "runs/seed"; cpdir.mkdir(parents=True)
            checkpoint = cpdir / "selected.pt"; checkpoint.write_bytes(b"frozen checkpoint")
            (cpdir / "history.json").write_text('{"history":[]}\n'); (cpdir / "summary.json").write_text('{"original":true}\n')
            prediction = folder / "training_validation_predictions.jsonl.gz"; prediction.write_bytes(b"training reference bytes")
            value = {"selected_epoch": seed, "selected_checkpoint": "/old/raw9/" + str(checkpoint.relative_to(originals)),
                "checkpoint_sha256": m.sha(checkpoint), "state_sha256": "a" * 64,
                "validation_reference": {"prediction_path": "/old/raw9/" + str(prediction.relative_to(originals))},
                "files": {str(p.relative_to(originals)): m.sha(p) for p in folder.rglob("*") if p.is_file()}}
            m.write(folder / "train_receipt.json", value)
            m.write(originals / "verified_exits" / (job["id"] + ".json"), {"actual_os_returncode": 0})
            selected.append({**job, "selected_epoch": seed, "checkpoint_sha256": m.sha(checkpoint), "state_sha256": "a" * 64})
        for name in ("train_validation.parquet", "split_manifest.json"):
            relative = "inputs/data/" + dataset + "/" + name
            path = parent / "input_snapshot" / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"frozen training input")
            training["input_files"][relative] = m.sha(path)
        full = project / (dataset + ".parquet"); full.write_bytes(b"frozen full data")
        datasets[dataset] = {"full_data": {"path": str(full), "sha256": m.sha(full)}, "populations": {"validation": {}, "test": {}}}
    training["jobs"].append({"id": "untouched_inter", "dataset": "intermittent_frozen_5000", "seed": 42})
    m.write(parent / "execution_contract.json", training); monkeypatch.setattr(m, "TRAIN_SHA", m.sha(parent / "execution_contract.json"))
    m.write(originals / "execution_contract.json", training)
    stop = report / "stop/POST_STOP_VERIFIED.json"
    m.write(stop, {"status": "user_stop_verified", "owned_after": [], "known_processes": {"800": {"exists": False}}})
    stopped_receipt = stop.with_name("FINAL_RECEIPT.json")
    m.write(stopped_receipt, {"status": "user_stop_verified", "actual_transport_exit": 0, "files": {stop.name: m.sha(stop)}})
    m.write(m.SPEC, {"selected_checkpoints": selected, "datasets": datasets,
        "native": {"environment": {}, "absolute_host_deadline_unix": m.time.time()+10000},
        "cap_stop": {"path": str(stop), "sha256": m.sha(stop), "owned_pids": [800],
            "receipt_path": str(stopped_receipt), "receipt_sha256": m.sha(stopped_receipt)}})
    m.write(m.REFERENCES, {"references": [{}]*42})
    parent_module = ModuleType("paper.scripts.manage_titantpp_shared_history_raw9_5090"); parent_module.verify_local = lambda: training
    monkeypatch.setitem(sys.modules, parent_module.__name__, parent_module)
    import paper.scripts
    monkeypatch.setattr(paper.scripts, "manage_titantpp_shared_history_raw9_5090", parent_module, raising=False)
    blob = io.BytesIO()
    with tarfile.open(fileobj=blob, mode="w:gz") as archive:
        for path in originals.rglob("*"):
            if path.is_file(): archive.add(path, arcname=str(path.relative_to(originals)))
    queries = []
    def ssh(host, command, **kw):
        queries.append((host, command)); return type("R", (), {"stdout": blob.getvalue()})()
    monkeypatch.setattr(m._transport, "ssh", ssh); monkeypatch.setattr(m._transport, "save_transport", lambda *a: None)
    def git(command, **kw):
        return "\n".join(m.NEW_SOURCE) if command[1] == "ls-files" else ""
    monkeypatch.setattr(m.subprocess, "check_output", git)
    result = m.prepare(); actual = m.read(bundle / "evaluation_contract.json")
    assert result["candidate_count"] == 6 and result["baseline_reference_count"] == 42 and result["actual_native_evaluation_count"] == 0
    assert len(queries) == 1 and queries[0][0] == "5090" and "untouched_inter" not in queries[0][1]
    assert actual["whole_raw9_complete"] is False and actual["new_training_count"] == 0 and actual["host"] == "5080"
    assert len(actual["operation_files"]) == 6 and actual["remote_root"].endswith("_5080")
    assert actual["cap_stop"]["path"] == "references/cap_stop.json"
    assert actual["cap_stop"]["receipt_path"] == "references/cap_stop_receipt.json"
    for job in actual["jobs"]:
        assert job["training_validation_reference"] == job["train_receipt"] and "validation_receipt" not in job
        assert (bundle / job["validation_predictions"]["path"]).read_bytes() == b"training reference bytes"
    with pytest.raises(ValueError, match="Existing six-fit evaluation"): m.prepare()


@pytest.mark.parametrize("mutation", ("none", "missing", "source", "reference", "not_passed"))
def test_reviewed_version_guard_covers_sources_tests_spec_and_extra_refs(tmp_path, monkeypatch, mutation):
    report = tmp_path / "reports" / m.NAME; report.mkdir(parents=True)
    monkeypatch.setattr(m, "ROOT", tmp_path); monkeypatch.setattr(m, "REPORT", report)
    extra = "reports/" + m.NAME + "/REFERENCES.json"; reviewed = {}
    for name in (*m.REVIEW_FILES, extra):
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("reviewed\n"); reviewed[name] = m.sha(path)
    if mutation == "missing": reviewed.pop(m.REVIEW_FILES[0])
    if mutation == "source": (tmp_path / m.REVIEW_FILES[0]).write_text("unreviewed\n")
    if mutation == "reference": (tmp_path / extra).write_text("new reference\n")
    m.write(report / "INDEPENDENT_QA.json", {"status": "PENDING" if mutation == "not_passed" else "PASS", "files_reviewed_sha256": reviewed})
    if mutation == "none": assert m.verify_reviewed_implementation()["sha256"] == m.sha(report / "INDEPENDENT_QA.json")
    else:
        with pytest.raises(ValueError): m.verify_reviewed_implementation()
