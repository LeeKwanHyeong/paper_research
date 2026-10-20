"""Fabricated interim snapshot and native queue checks; no SSH or real data."""
from pathlib import Path
import io
import json
import shlex
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest

from paper.scripts import manage_titantpp_shared_inter3_interim_test as m


def test_capture_cli_dispatch_is_independent_of_preparation(monkeypatch, capsys):
    monkeypatch.setattr(m, "capture", lambda: {"status": "frozen_interim_captured"})
    for name in ("prepare", "advance", "verify_local", "remote_run"):
        monkeypatch.delattr(m, name)
    assert m.main(["--mode", "capture"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "frozen_interim_captured"


@pytest.mark.parametrize("mutation", ("none", "append_history", "checkpoint_race", "prefix", "best_epoch", "best_tensor", "nonfinite"))
def test_capture_preserves_raw_resume_bytes_and_freezes_embedded_selection(tmp_path, monkeypatch, mutation):
    root = tmp_path / "original"; root.mkdir()
    source = root / "source"; source.mkdir()
    sources = {}
    for index in range(983):
        name = f"science{index}.py"; path = source / name; path.write_bytes(b"immutable")
        sources[name] = m.sha(path)
    job = {"id": "inter_seed42", "seed": 42, "dataset": m.DATASET, "arm": "shared"}
    folder = root / "run" / job["id"] / "runs" / "shared" / "log_mse" / "seed_42"; folder.mkdir(parents=True)
    cp = folder / "last_epoch_state.pt"; cp.write_bytes(b"raw checkpoint with optimizer and best state")
    history = [{"epoch": 1, "val_qty_rmse": 4., "val_qty_mae": 3., "val_time_nll": 2.},
        {"epoch": 2, "val_qty_rmse": 2., "val_qty_mae": 1., "val_time_nll": .5},
        {"epoch": 3, "val_qty_rmse": 2., "val_qty_mae": 1., "val_time_nll": .5}]
    m.write(folder / "history.json", {"history": history})
    c = {"source": {"files": sources, "files_sha256": "source-digest"}, "input_files": {},
        "datasets": [{"dataset_id": m.DATASET, "inherited_data_identity": {"populations": {"validation": {"target_count": 8}}}}]}
    m.write(root / "execution_contract.json", c); monkeypatch.setattr(m, "TRAIN_SHA", m.sha(root / "execution_contract.json"))
    before = {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}
    runtime = ModuleType("paper.scripts.titantpp_shared_history_raw9_5090_runtime")
    runtime.require, runtime.read, runtime.sha_file, runtime.local_path = m.require, m.read, m.sha, m.local_path
    loader = ModuleType("simple_lab_test.search.common.runner"); calls = []
    def load(path, *, map_location):
        calls.append(map_location); assert path.read() == cp.read_bytes()
        if mutation == "checkpoint_race": cp.write_bytes(b"new epoch checkpoint")
        if mutation == "prefix":
            (folder / "history.json").write_text(json.dumps({"history": [{**row, "val_qty_rmse": 7.} for row in history]}))
        if mutation == "append_history":
            (folder / "history.json").write_text(json.dumps({"history": history + [{"epoch": 4, "val_qty_rmse": .1}]}))
        embedded = [dict(row) for row in history]
        if mutation == "nonfinite": embedded[1]["val_time_nll"] = float("nan")
        return {"checkpoint_type": "epoch_resume", "seed": 42, "backbone": "shared", "checkpoint_monitor": "validation_raw_quantity_rmse",
            "evaluation_scope": "validation_only", "held_out_test_evaluated": False, "epoch": 3, "history": embedded,
            "best_epoch": 1 if mutation == "best_epoch" else 2, "best_selection_value": 2.,
            "model_state_dict": "current", "model_state_sha256": "current", "best_state_dict": "best",
            "best_state_sha256": "wrong" if mutation == "best_tensor" else "best", "time_head_training_verification": {"current_only": True}}
    loader.torch_load_checkpoint = load; loader.canonical_state_dict_sha256 = lambda value: value
    scientific = ModuleType("paper.scripts.run_titantpp_shared_history_raw9_5090")
    scientific.prepared = SimpleNamespace(select_best_epoch=lambda rows: min(rows, key=lambda x: x["val_qty_rmse"])["epoch"])
    scientific.verify_time_head_proof = lambda proof, state: m.require(proof["current_only"] and state == "current", "Proof bound to wrong state")
    import paper.scripts
    for module in (runtime, loader, scientific): monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(paper.scripts, "titantpp_shared_history_raw9_5090_runtime", runtime, raising=False)
    monkeypatch.setattr(paper.scripts, "run_titantpp_shared_history_raw9_5090", scientific, raising=False)
    output = io.BytesIO(); monkeypatch.setattr(sys, "stdout", SimpleNamespace(buffer=output))
    program = m.snapshot_collection_program({"hosts": {"5090": {"root": str(root)}}}, [job])
    if mutation in ("none", "append_history"):
        exec(compile(program, "<snapshot>", "exec"), {})
        with tarfile.open(fileobj=io.BytesIO(output.getvalue()), mode="r:gz") as archive:
            capture = json.load(archive.extractfile("snapshot_capture.json"))["jobs"][0]
            assert capture["selected_epoch"] == 2 and capture["snapshot_epoch"] == 3
            assert archive.extractfile(capture["checkpoint_path"]).read() == cp.read_bytes()
            assert capture["checkpoint_before_sha256"] == capture["checkpoint_after_sha256"] == m.sha(cp)
            assert not any("train_exit" in name or "train_receipt" in name for name in archive.getnames())
    else:
        with pytest.raises(ValueError): exec(compile(program, "<invalid-snapshot>", "exec"), {})
        assert output.getvalue() == b""
    assert calls == ["cpu"]
    after = {str(p): m.sha(p) for p in root.rglob("*") if p.is_file()}
    if mutation not in ("append_history", "checkpoint_race", "prefix"): assert after == before


def test_capture_failure_is_one_attempt_and_keeps_raw_transport(tmp_path, monkeypatch):
    bundle, report = tmp_path / "bundle", tmp_path / "report"; report.mkdir()
    monkeypatch.setattr(m, "BUNDLE", bundle); monkeypatch.setattr(m, "REPORT", report)
    monkeypatch.setattr(m, "SPEC", report / "SCIENCE_SPEC.json")
    m.write(m.SPEC, {"schema": "shared_inter3_interim_test_science_v1", "training_contract_raw_sha256": m.TRAIN_SHA})
    jobs = [{"id": f"inter{s}", "dataset": m.DATASET, "seed": s} for s in (42, 52, 62)]
    training = {"jobs": jobs, "hosts": {"5090": {"source_root": "/original/source", "root": "/original", "python": "/cpu/python",
        "environment": {"PYTHONPATH": "/original/source", "LD_LIBRARY_PATH": "/original/libraries", "PYTHONDONTWRITEBYTECODE": "1"}}}}
    monkeypatch.setattr(m, "_training", lambda: training)
    calls = []
    def ssh(host, command, **kwargs):
        calls.append((host, command)); return SimpleNamespace(returncode=1, stdout=b"partial raw stream", stderr=b"checkpoint race")
    monkeypatch.setattr(m._transport, "ssh", ssh)
    old_bundle = m._transport.BUNDLE; monkeypatch.setattr(m._transport, "BUNDLE", bundle)
    with pytest.raises(ValueError, match="raw transport preserved"): m.capture()
    assert len(calls) == 1 and calls[0][0] == "5090" and "LD_LIBRARY_PATH=/original/libraries" in calls[0][1]
    assert (bundle / "selected/original.tar.gz").read_bytes() == b"partial raw stream"
    assert (bundle / "snapshot_collection.intent.json").exists() and not (bundle / "selected/receipt.json").exists()
    assert next((bundle / "transport/5090").glob("*.stderr")).read_bytes() == b"checkpoint race"
    with pytest.raises(ValueError, match="no overwrite/retry"): m.capture()
    assert len(calls) == 1


def test_capture_program_never_writes_or_trains_on_original_host():
    program = m.snapshot_collection_program({"hosts": {"5090": {"root": "/original"}}}, [])
    compile(program, "<snapshot>", "exec")
    for forbidden in (".mkdir", "r.write", "torch.save", "run_stage", "train_exit.json", "verify_receipt", "map_location='cuda'"):
        assert forbidden not in program
    assert "map_location='cpu'" in program and "payload['best_state_dict']" in program
    assert "external[:len(history)]==history" in program and "before==after" in program


def test_preserved_dependencies_are_the_original_bytes():
    assert m.REMOTE.endswith(m.NAME + "_5080")
    assert len(m.NEW_SOURCE) == 3 and all(m.sha(m.ROOT / name) == digest for name, digest in m.PRESERVED_OPERATIONS.items())


@pytest.fixture
def native_contract(tmp_path, monkeypatch):
    root = tmp_path / "native"; (root / "source").mkdir(parents=True)
    c = {"schema": m.SCHEMA, "host": "5080", "remote_root": str(root), "source_root": "source",
        "native": {"python": sys.executable, "gpu_uuid": "GPU-fabricated", "environment": {},
            "paused_processes": [{"pid": p} for p in (700, 701, 702, 703)]},
        "completed_raf_taxi6": {"root": "/old/eval", "pipeline_receipt_sha256": "old-final", "pipeline_exit_sha256": "old-exit"},
        "cap_stop": {"owned_pids": [800, 801]}, "absolute_host_deadline_unix": m.time.time() + 10000,
        "child_timeout_seconds": 100, "jobs": [{"id": f"inter_seed{s}", "dataset": m.DATASET, "seed": s,
            "selected_epoch": 5, "snapshot_epoch": 40} for s in (42, 52, 62)]}
    m.write(root / "evaluation_contract.json", c)
    monkeypatch.setattr(m, "_OWNED_CHILDREN", [])
    monkeypatch.setattr(m, "_paused", lambda _: c["native"]["paused_processes"])
    monkeypatch.setattr(m, "_verify_completed_remote", lambda _: [])
    return c, root


def native_children(c, root, monkeypatch, *, failed_stage=None, no_exit=False):
    children, launched, maximum = [], [], [0]
    class Child:
        def __init__(self, command, **kwargs):
            self.pid, self.returncode, self.ticks = 1000 + len(children), None, 0
            self.job = next(j for j in c["jobs"] if j["id"] == command[command.index("--job") + 1])
            self.stage = command[command.index("--split") + 1] if "--split" in command else "qualification"
            assert kwargs["start_new_session"] is True and kwargs["cwd"] == root / "source"
            children.append(self); launched.append(self.stage)
            maximum[0] = max(maximum[0], sum(p.returncode is None for p in children))
        def poll(self):
            if self.returncode is None and not no_exit:
                self.ticks += 1
                if self.ticks >= 3:
                    self.returncode = 1 if self.stage == failed_stage else 0
                    if self.returncode == 0:
                        m.write(m._receipt_path(root, self.job, self.stage), {"status": "passed" if self.stage == "qualification" else "complete",
                            "job_id": self.job["id"], "contract_sha256": m.canonical(c), "checks": {"causal": True}, "real_data_loaded": False})
            return self.returncode
    monkeypatch.setattr(m.subprocess, "Popen", Child)
    def gpu(_): return {"compute_pids": [701, 702, 703] + [p.pid for p in children if p.returncode is None]}
    monkeypatch.setattr(m, "_plain_gpu_snapshot", gpu)
    monkeypatch.setattr(m, "_owned_pid_absent", lambda pid: all(p.returncode is not None for p in children if p.pid == pid))
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    monkeypatch.setattr(m.signal, "signal", lambda *a: None)
    return children, launched, maximum


def test_qualification3_validation3_gate_test3_maximum_three(native_contract, monkeypatch):
    c, root = native_contract; children, launched, maximum = native_children(c, root, monkeypatch)
    worker = ModuleType(m.WORKER); gates = []
    def gate(path, actual):
        assert launched == ["qualification"] * 3 + ["validation"] * 3
        value = m.read(root / "validation_gate.json")
        assert value["job_count"] == 3 and value["schema"] == "titantpp_shared_inter3_interim_validation_gate_v1"
        assert set(value["validation_receipts"]) == set(value["validation_worker_exits"]) == {j["id"] for j in c["jobs"]}
        gates.append(True)
    worker.verify_validation_gate = gate; monkeypatch.setitem(sys.modules, m.WORKER, worker)
    result = m.remote_run(root)
    assert gates == [True] and launched == ["qualification"] * 3 + ["validation"] * 3 + ["test"] * 3
    assert maximum[0] == 3 and len(children) == 9
    assert (result["qualification_exit0_count"], result["actual_data_worker_exit0_count"], result["actual_native_worker_exit0_count"]) == (3, 6, 9)
    assert result["gpu_compute_pids"] == [701, 702, 703] and result["whole_raw9_complete"] is False
    assert result["paused_external_preserved"] is result["owned_pids_absent"] is result["owned_gpu_pids_absent"] is True
    with pytest.raises(ValueError, match="Existing evaluation attempt"): m.remote_run(root)


@pytest.mark.parametrize("failed_stage", ("qualification", "validation"))
def test_native_failure_never_starts_test(native_contract, monkeypatch, failed_stage):
    c, root = native_contract; _, launched, maximum = native_children(c, root, monkeypatch, failed_stage=failed_stage)
    with pytest.raises(ValueError, match="stage failed"): m.remote_run(root)
    assert "test" not in launched and maximum[0] == 3 and (root / "pipeline_failure.json").exists()


def test_history_gate_failure_blocks_test(native_contract, monkeypatch):
    c, root = native_contract; _, launched, _ = native_children(c, root, monkeypatch)
    worker = ModuleType(m.WORKER); worker.verify_validation_gate = lambda *args: (_ for _ in ()).throw(ValueError("Recorded history differs"))
    monkeypatch.setitem(sys.modules, m.WORKER, worker)
    with pytest.raises(ValueError, match="history differs"): m.remote_run(root)
    assert "test" not in launched


def test_timeout_cleans_only_owned_children(native_contract, monkeypatch):
    c, root = native_contract; c["child_timeout_seconds"] = 0
    children, launched, _ = native_children(c, root, monkeypatch, no_exit=True); killed = []
    def stop(child):
        assert child in children and child.pid not in {700, 701, 702, 703, 800, 801}
        killed.append(child.pid); child.returncode = -15
    monkeypatch.setattr(m, "terminate_owned_child", stop)
    with pytest.raises(ValueError, match="stage failed"): m.run_remote_stage(root, c, "qualification")
    assert len(killed) == 3 and launched == ["qualification"] * 3


def test_unknown_gpu_and_old_owner_block_before_new_child(native_contract, monkeypatch):
    c, root = native_contract
    monkeypatch.setattr(m.subprocess, "Popen", lambda *a, **kw: pytest.fail("Unexpected child"))
    monkeypatch.setattr(m, "_plain_gpu_snapshot", lambda _: {"compute_pids": [701, 999]})
    with pytest.raises(ValueError, match="Unknown active5080"): m.remote_run(root)


def test_advance_is_once_5080_only_and_has_inherited_owned_timeout(native_contract, tmp_path, monkeypatch):
    c, _ = native_contract; bundle = tmp_path / "bundle"; bundle.mkdir(); (bundle / "deployment.tar.gz").write_bytes(b"archive")
    monkeypatch.setattr(m, "BUNDLE", bundle); monkeypatch.setattr(m, "verify_local", lambda **kw: c)
    monkeypatch.setattr(m, "collect_and_compare", lambda _: {"status": "complete"})
    calls = []; monkeypatch.setattr(m._transport, "ssh", lambda host, command, **kw: calls.append((host, command)) or SimpleNamespace())
    monkeypatch.setattr(m._transport, "save_transport", lambda *a: None)
    # Production observations contain exact paused identities; this test checks emitted program syntax and scope.
    for p in c["native"]["paused_processes"]: p.update(start_ticks=12, pgid=9, argv=["paused"], cwd="/old")
    assert m.advance()["status"] == "complete" and len(calls) == 3 and {h for h, _ in calls} == {"5080"}
    fresh, supervisor = shlex.split(calls[0][1])[-1], shlex.split(calls[2][1])[-1]
    compile(fresh, "<fresh5080>", "exec"); compile(supervisor, "<owned-timeout>", "exec")
    assert "RAF/Taxi6 worker remains" in fresh and "Paused identity changed" in fresh
    assert "p.wait(timeout=limit)" in supervisor and "os.killpg(p.pid,signal.SIGTERM)" in supervisor
    assert "os.killpg(p.pid,signal.SIGKILL)" in supervisor and "time.time()" in supervisor
    with pytest.raises(FileExistsError): m.advance()
    assert len(calls) == 3


def test_prepare_reuses_capture_without_ssh_and_binds_actual_selected_epochs(tmp_path, monkeypatch):
    project, bundle, report, parent = (tmp_path / x for x in ("project", "bundle", "report", "parent"))
    for path in (project, bundle, report, parent): path.mkdir()
    for key, value in (("ROOT", project), ("BUNDLE", bundle), ("REPORT", report), ("TRAIN_BUNDLE", parent),
            ("SPEC", report / "SCIENCE_SPEC.json"), ("REFERENCES", report / "REFERENCES.json")):
        monkeypatch.setattr(m, key, value)
    monkeypatch.setattr(m, "verify_reviewed_implementation", lambda: None)
    sources = project / "science.py"; sources.write_text("preserved = True\n")
    dependencies = {}
    for name in (*m.NEW_SOURCE, *m.PRESERVED_OPERATIONS):
        path = project / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("isolated operations\n")
        if name in m.PRESERVED_OPERATIONS: dependencies[name] = m.sha(path)
    monkeypatch.setattr(m, "PRESERVED_OPERATIONS", dependencies)
    originals = bundle / "selected/originals"; originals.mkdir(parents=True)
    m.write(bundle / "selected/receipt.json", {"captured_once": True})
    training = {"source": {"files": {"science.py": m.sha(sources)}, "files_sha256": "original-source-digest"},
        "jobs": [], "input_files": {}, "datasets": [{"dataset_id": m.DATASET,
        "inherited_data_identity": {"populations": {"validation": {"target_count": 8}}}}]}
    captured = {"jobs": []}; payloads = {}
    for seed, selected, saved in ((42, 7, 43), (52, 6, 41), (62, 12, 39)):
        job = {"id": f"inter_seed{seed}", "dataset": m.DATASET, "seed": seed, "arm": "shared"}; training["jobs"].append(job)
        folder = originals / "run" / job["id"]; folder.mkdir(parents=True)
        cp, history = folder / "last_epoch_state.pt", folder / "history.json"
        cp.write_bytes(f"raw epoch resume seed{seed}".encode()); row = {"epoch": selected, "val_qty_rmse": 2., "val_qty_mae": 1., "val_time_nll": .5}
        m.write(history, {"history": [row]})
        payloads[cp] = {"best_epoch": selected, "epoch": saved, "best_state_sha256": f"best{seed}", "row": row}
        captured["jobs"].append({"job_id": job["id"], "selected_epoch": selected, "snapshot_epoch": saved,
            "checkpoint_path": str(cp.relative_to(originals)), "history_path": str(history.relative_to(originals)),
            "state_sha256": f"best{seed}", "selected_history_row": row})
    for name in ("train_validation.parquet", "split_manifest.json"):
        relative = "inputs/data/" + m.DATASET + "/" + name; path = parent / "input_snapshot" / relative
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"original raw input"); training["input_files"][relative] = m.sha(path)
    full = project / "full.parquet"; full.write_bytes(b"full original split")
    stop = report / "stop/POST_STOP_VERIFIED.json"; m.write(stop, {"status": "user_stop_verified"})
    stop_receipt = stop.with_name("FINAL_RECEIPT.json")
    m.write(stop_receipt, {"status": "user_stop_verified", "actual_transport_exit": 0, "files": {stop.name: m.sha(stop)}})
    m.write(m.SPEC, {"schema": "shared_inter3_interim_test_science_v1", "training_contract_raw_sha256": m.TRAIN_SHA,
        "datasets": {m.DATASET: {"full_data": {"path": str(full), "sha256": m.sha(full)}, "populations": {split: {"target_count": count, "target_identity_sha256": "a"*64, "target_quantity_sha256": "b"*64, "recorded_time_sha256": "c"*64, "component_time_sha256": "d"*64} for split, count in (("validation", 8), ("test", 7))}}},
        "native": {"environment": {}, "absolute_host_deadline_unix": m.time.time()+10000},
        "completed_raf_taxi6": {"root": "/old/eval", "pipeline_receipt_sha256": "prior", "pipeline_exit_sha256": "prior-exit"},
        "cap_stop": {"path": str(stop), "sha256": m.sha(stop), "receipt_path": str(stop_receipt), "receipt_sha256": m.sha(stop_receipt), "owned_pids": [800]}})
    m.write(m.REFERENCES, {"references": [{}]*42})
    monkeypatch.setattr(m, "_training", lambda: training); monkeypatch.setattr(m, "_verify_capture", lambda: captured)
    monkeypatch.setattr(m._transport, "ssh", lambda *a, **kw: pytest.fail("Snapshot must never be recaptured"))
    worker = ModuleType(m.WORKER)
    worker.verify_snapshot_payload = lambda payload, *args: (payload["row"], {"proof_current_only": True})
    worker.checkpoint_snapshot_metadata = lambda payload: {"derived_original": payload}
    monkeypatch.setitem(sys.modules, m.WORKER, worker)
    loader = ModuleType("simple_lab_test.search.common.runner"); loader.torch_load_checkpoint = lambda path, **kw: payloads[path]
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    result = m.prepare(); actual = m.read(bundle / "evaluation_contract.json")
    assert result["candidate_count"] == 3 and result["baseline_reference_count"] == 42 and result["actual_native_evaluation_count"] == 0
    assert actual["schema"] == m.SCHEMA and actual["whole_raw9_complete"] is False and actual["new_training_count"] == 0
    assert actual["evaluation_plan"] == ["qualification3", "validation3", "gate3", "test3"]
    assert all(set(population) == {"target_count", "target_identity_sha256", "target_quantity_sha256"} for population in actual["data"][m.DATASET]["populations"].values())
    assert [j["selected_epoch"] for j in actual["jobs"]] == [7, 6, 12]
    for job in actual["jobs"]:
        assert "train_exit" not in job and "train_receipt" not in job and "validation_predictions" not in job
        assert job["snapshot_receipt"] == job["training_validation_reference"]
        receipt = m.read(bundle / job["snapshot_receipt"]["path"])
        assert receipt["history_reference"]["metrics"]["overall"] == {"qty_rmse": 2., "qty_mae": 1., "time_nll": .5}
        assert receipt["original_training_terminal_required"] is receipt["predictions_available"] is False
    with pytest.raises(ValueError, match="no retry"): m.prepare()


def test_projected_population_identity_is_admitted_by_native_worker(tmp_path, monkeypatch):
    from paper.scripts import evaluate_titantpp_shared_inter3_interim_frozen as worker
    from paper.scripts import run_titantpp_shared_history_raw9_5090 as frozen
    source = tmp_path / "source"; source.mkdir()
    files = {}
    for index in range(983):
        path = source / f"f{index}.py"; path.write_bytes(b"preserved source")
        files[path.name] = m.sha(path)
    definition = {"quantity_boundaries_all_train_rows": [0, 1, 3, 187.0], "inherited_data_identity": {
        "data": {"sha256": "t"*64}, "populations": {"validation": {"target_count": 8,
            "target_identity_sha256": "a"*64, "target_quantity_sha256": "b"*64}}}}
    training = {"source": {"files": files, "files_sha256": m.canonical(files)}, "hosts": {"5090": {"runtime_expected": {"cudnn": 92000}}}}
    contract_path = tmp_path / "execution_contract.json"; m.write(contract_path, training)
    training_input, full = tmp_path / "training.parquet", tmp_path / "full.parquet"
    training_input.write_bytes(b"training data"); full.write_bytes(b"full original split")
    spec_populations = {split: {"target_count": count, "target_identity_sha256": "a"*64, "target_quantity_sha256": "b"*64,
        "recorded_time_sha256": "c"*64, "component_time_sha256": "d"*64} for split, count in (("validation", 8), ("test", 7))}
    c = {"schema": m.SCHEMA, "host": "5080", "evaluation_authorized": True, "source_root": "source",
        "training_contract": {"path": contract_path.name, "sha256": m.sha(contract_path)},
        "native": {"gpu_uuid": m.GPU_UUID, "runtime_expected": {"cudnn": 91900}},
        "jobs": [{"id": f"{m.DATASET}__{worker.ARM}__seed{seed}__losslog_mse", "dataset": m.DATASET, "arm": worker.ARM,
            "seed": seed, "selected_epoch": 7, "snapshot_epoch": 40, "state_sha256": "f"*64} for seed in (42, 52, 62)],
        "data": {m.DATASET: {"training": {"path": training_input.name, "sha256": "t"*64},
            "full": {"path": full.name, "sha256": m.sha(full)}, "populations": m._population_bindings(spec_populations)}}}
    # Use a real byte binding while matching the fabricated inherited training data identity.
    c["data"][m.DATASET]["training"]["sha256"] = definition["inherited_data_identity"]["data"]["sha256"] = m.sha(training_input)
    monkeypatch.setattr(worker, "ORIGINAL_TRAINING_CONTRACT_SHA256", m.sha(contract_path))
    monkeypatch.setattr(frozen, "ROOT", source); monkeypatch.setattr(frozen, "verify_frozen_dependencies", lambda: None)
    monkeypatch.setattr(frozen, "select_job", lambda actual, host, job_id: (next(j for j in c["jobs"] if j["id"] == job_id), definition, tmp_path, tmp_path))
    assert worker.verify_contract(tmp_path, c)[0] == c
    assert all(len(p) == 5 for p in spec_populations.values())
    assert all(len(p) == 3 for p in c["data"][m.DATASET]["populations"].values())
