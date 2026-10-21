"""Synthetic completion evidence; no SSH, real training, or model evaluation."""
from copy import deepcopy
from pathlib import Path
import time

import pytest

from paper.scripts import titans_mac_accuracy9_completion_bridge as b
from paper.scripts import run_titans_mac_accuracy9_recovery as worker


def identity(pid=123, argv=None, cwd="/source", ticks=99):
    return {"pid": pid, "pgid": pid, "start_ticks": ticks,
            "argv": argv or ["python", "-m", "worker"], "cwd": cwd}


class Child:
    pid = 123
    returncode = None
    def poll(self):
        return self.returncode


def test_initial_exec_race_accepts_only_later_exact_direct_child(monkeypatch):
    wanted = identity(); forked = {**wanted, "argv": ["parent"]}
    rows = iter([forked, wanted])
    monkeypatch.setattr(b.r, "proc_identity", lambda _: next(rows))
    monkeypatch.setattr(b, "_parent_pid", lambda _: b.os.getpid())
    sleeps = []; monkeypatch.setattr(b.time, "sleep", sleeps.append)
    assert b.initial_identity(Child(), wanted["argv"], Path("/source")) == wanted
    assert sleeps == [.01]


def test_running_identity_mismatch_has_no_retry(monkeypatch):
    wanted = identity(); live = {**wanted, "argv": ["unexpected"]}
    monkeypatch.setattr(b.r, "proc_identity", lambda _: live)
    monkeypatch.setattr(b.r.bounded, "proc_identity", lambda _: live)
    monkeypatch.setattr(b.time, "sleep", lambda _: pytest.fail("Running mismatch must not retry"))
    with pytest.raises(ValueError, match="Owned child identity changed"):
        b.initial_identity(Child(), wanted["argv"], Path("/source"), 99)


def test_initial_exec_rejects_non_child_without_wait(monkeypatch):
    live = identity()
    monkeypatch.setattr(b.r, "proc_identity", lambda _: live)
    monkeypatch.setattr(b, "_parent_pid", lambda _: b.os.getpid() + 1)
    with pytest.raises(ValueError, match="live direct child"):
        b.initial_identity(Child(), live["argv"], Path("/source"))


def test_initial_exec_rejects_pid_reuse(monkeypatch):
    rows = iter([identity(argv=["parent"]), identity(ticks=100)])
    monkeypatch.setattr(b.r, "proc_identity", lambda _: next(rows))
    monkeypatch.setattr(b, "_parent_pid", lambda _: b.os.getpid())
    monkeypatch.setattr(b.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="PID was reused"):
        b.initial_identity(Child(), ["python", "-m", "worker"], Path("/source"))


def test_initial_exec_timeout_is_bounded_and_never_signals(monkeypatch):
    monkeypatch.setattr(b.r, "proc_identity", lambda _: identity(argv=["parent"]))
    monkeypatch.setattr(b, "_parent_pid", lambda _: b.os.getpid())
    clock = iter([0., 0., 2.1]); monkeypatch.setattr(b.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(b.time, "sleep", lambda _: None)
    monkeypatch.setattr(b.r, "terminate_owned_child", lambda *_: pytest.fail("Unknown identity must never be signalled"))
    with pytest.raises(ValueError, match="handoff timed out"):
        b.initial_identity(Child(), ["python", "-m", "worker"], Path("/source"))


def test_exited_initial_child_preserves_actual_exit(monkeypatch):
    child = Child(); child.returncode = 17
    monkeypatch.setattr(b.r, "proc_identity", lambda _: None)
    assert b.initial_identity(child, [], Path("/source")) is None
    assert child.returncode == 17


def test_transient_empty_cmdline_is_not_reported_as_os_exit(monkeypatch):
    wanted = identity(); rows = iter([None, wanted])
    monkeypatch.setattr(b.r, "proc_identity", lambda _: next(rows))
    monkeypatch.setattr(b, "_parent_pid", lambda _: b.os.getpid())
    monkeypatch.setattr(b, "_birth_ticks", lambda _: wanted["start_ticks"])
    monkeypatch.setattr(b.time, "sleep", lambda _: None)
    assert b.initial_identity(Child(), wanted["argv"], Path("/source")) == wanted


FAILURE = {"stage": "train", "traceback": "Traceback\n in _identity\nValueError: Owned child identity changed\n"}


@pytest.mark.parametrize("failure", [
    {"stage": "validation", "traceback": FAILURE["traceback"]},
    {**FAILURE, "job_id": "failed_science"},
    {"stage": "train", "traceback": "ValueError: Native CUDA failure"},
    {"stage": "train", "traceback": "ValueError: Owned child identity changed"},
])
def test_only_pinned_initial_identity_failure_is_admitted(failure):
    with pytest.raises(ValueError):
        b._race_failure(failure)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    old_root, root = tmp_path / "v3", tmp_path / "v4"
    (old_root / "source").mkdir(parents=True); (root / "source").mkdir(parents=True)
    jobs = [{"id": f"{d}_{s}", "seed": s, "dataset": d, "recovery_mode": "resume" if d == "inter" else "reference"}
            for d in ("raf", "taxi", "inter") for s in (42, 52, 62)]
    old = {"root": str(old_root), "source_root": str(old_root / "source"), "jobs": jobs,
        "training_jobs": [j["id"] for j in jobs if j["dataset"] == "inter"],
        "reference_jobs": [j["id"] for j in jobs if j["dataset"] != "inter"],
        "validation_jobs": [j["id"] for j in jobs], "test_jobs": [j["id"] for j in jobs],
        "python": "/python", "boot_id": "boot", "absolute_host_deadline_unix": time.time() + 500,
        "limits": {"queue_wall_seconds": 500}, "environment": {}, "marker": "frozen"}
    b.write(old_root / "recovery_contract.json", old)
    c = {**deepcopy(old), "root": str(root), "source_root": str(root / "source"),
        "continuation_only": True, "fresh_training_authorized": False}
    c["continuation_from"] = {"root": str(old_root), "contract_sha256": b.sha_json(old),
        "contract_raw_sha256": b.sha_file(old_root / "recovery_contract.json"),
        "initial_live_observation": {"path": str(root / "proofs/live.json"), "sha256": "proof"}}
    identities, stages = {}, {s: {"status": "complete" if s == "reference" else "failed",
        "exits": {}, "receipts": {}, "pending_job_ids": [], "failure": None if s == "reference" else FAILURE}
        for s in ("reference", "train")}
    for index, job in enumerate(jobs):
        key = job["id"]; stage = "train" if key in old["training_jobs"] else "reference"
        folder = old_root / "run" / key
        selected = folder / "runs/best.pt"; selected.parent.mkdir(parents=True)
        selected.write_bytes((key + "-selected").encode())
        b.write(folder / "input_receipt.json", {"input": key})
        prediction = folder / "training_validation_predictions.jsonl.gz"; prediction.write_bytes(b"predictions")
        b.write(folder / "runs/history.json", {"epoch": 77})
        files = {str(p.relative_to(old_root)): b.sha_file(p) for p in folder.rglob("*") if p.is_file()}
        receipt = {"schema": "science", "status": "complete", "stage": "train", "job_id": key,
            "contract_sha256": b.sha_json(old), "selected_checkpoint": str(selected), "selected_epoch": 77,
            "checkpoint_sha256": b.sha_file(selected), "state_sha256": "a" * 64,
            "summary": {"checkpoint_path": str(selected)}, "files": files,
            "validation_reference": {"prediction_path": str(prediction), "metrics": {"qty_rmse": 1.}}}
        b.write(folder / "train_receipt.json", receipt)
        command = b.r.worker_command(old, old_root / "recovery_contract.json", job, stage)
        pid, ticks = 1000 + index, 2000 + index
        exited = {"job_id": key, "stage": stage, "pid": pid, "start_ticks": ticks,
            "command": command, "started_at": "start", "observed_exit_at": "end", "actual_os_returncode": 0,
            "timeout": False, "boot_id": "boot"}
        for stream in ("stdout", "stderr"):
            log = old_root / "logs" / (key + "." + stage + "." + stream + ".log")
            log.parent.mkdir(exist_ok=True); log.write_bytes((key + stream).encode())
            exited[stream + "_sha256"] = b.sha_file(log); exited[stream + "_bytes"] = log.stat().st_size
        launch = {k: exited[k] for k in ("job_id", "stage", "pid", "start_ticks", "command", "started_at", "boot_id")}
        if stage == "train":
            identities[key] = identity(pid, command, old["source_root"], ticks)
            if job["seed"] == 62:
                exited["start_ticks"] = None
            else:
                b.write(folder / (stage + "_launch.json"), launch)
        else:
            b.write(folder / (stage + "_launch.json"), launch)
            b.write(folder / "train_exit.json", {"historical": "exit0", "pid": pid - 100})
        b.write(folder / (stage + "_exit.json"), exited)
        b.write(folder / (stage + "_lease.json"), {"original_operational_lease": True})
        wrapper = {**receipt, "schema": b.r.STAGE_SCHEMA, "stage": stage,
            "scientific_contract_sha256": b.r.ORIGINAL_CANONICAL, "scientific_contract_raw_sha256": "raw",
            "lineage": {"resume_checkpoint": {"copy_path": str(folder / "original.pt")}},
            "files": {str(p.relative_to(old_root)): b.sha_file(p) for p in folder.rglob("*") if p.is_file()}}
        b.write(folder / (stage + "_recovery_receipt.json"), wrapper)
        stages[stage]["exits"][key] = exited
        stages[stage]["receipts"][key] = b.r.verify_receipt(old, job, stage)
    b.write(old_root / "dispatch_failure.json", {**FAILURE,
        "children_left_running": {key: value["pid"] for key, value in identities.items()}})
    live = {"boot_id": "boot", "jobs": [{"job_id": key, "actual_identity": value,
        "exact_command_cwd_pgid_match": True} for key, value in identities.items()],
        "status": {"status": "draining_after_failure", "stage": "train", "pending_job_ids": [],
                   "contract_sha256": b.sha_json(old), "failure": FAILURE}}
    b.write(root / "proofs/live.json", live)
    c["continuation_from"]["initial_live_observation"]["sha256"] = b.sha_file(root / "proofs/live.json")
    b.write(old_root / "final_receipt.json", {"schema": b.r.SCHEMA, "status": "failed", "contract_sha256": b.sha_json(old),
        "stages": stages, "held_out_test_evaluated": False})
    launch = {"identity": identity(888, ["dispatcher"], old["source_root"], 333),
              "command": ["dispatcher"], "contract_sha256": b.sha_json(old)}
    b.write(old_root / "dispatcher_launch.json", launch)
    dispatcher = {"pid": 888, "start_ticks": 333, "command": ["dispatcher"], "boot_id": "boot",
                  "actual_os_returncode": 1, "timeout": False, "cleanup_error": None}
    for stream in ("stdout", "stderr"):
        log = old_root / ("dispatcher." + stream + ".log"); log.write_bytes(b"closed")
        dispatcher[stream + "_sha256"] = b.sha_file(log); dispatcher[stream + "_bytes"] = log.stat().st_size
    b.write(old_root / "dispatcher_exit.json", dispatcher)
    monkeypatch.setattr(b.r, "proc_identity", lambda _: None)
    monkeypatch.setattr(b.r, "boot_id", lambda: "boot")
    monkeypatch.setattr(b.r, "gpu_state", lambda *_: {"compute_pids": []})
    def effective(contract):
        return {**contract, "hosts": {"5090": {"root": contract["root"]}}}
    monkeypatch.setattr(worker, "effective_scientific", effective)
    monkeypatch.setattr(worker.science, "base_receipt", lambda contract, host, job, stage: {
        "schema": "science", "status": "complete", "stage": stage, "job_id": job["id"], "contract_sha256": b.sha_json(contract)})
    def verify(contract, host, job, where, receipt):
        assert Path(receipt["selected_checkpoint"]).is_relative_to(where)
        for name, digest in receipt["files"].items():
            assert b.sha_file(Path(where) / name) == digest
    monkeypatch.setattr(worker.science, "verify_train_receipt", verify)
    return c, old, identities


def test_terminal_accepts_only_drained_actual_success(completed):
    c, old, identities = completed
    assert b.terminal_ready(c, old, identities) is True


def test_terminal_waits_without_launching_or_signalling(completed, monkeypatch):
    c, old, identities = completed
    (Path(old["root"]) / "dispatcher_exit.json").unlink()
    monkeypatch.setattr(b.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("No new worker during waiting"))
    monkeypatch.setattr(b.r, "terminate_owned_child", lambda *_: pytest.fail("Original training cannot be signalled"))
    assert b.terminal_ready(c, old, identities) is False


def test_partial_original_scientific_failure_blocks_before_queue_drains(completed, monkeypatch):
    c, old, identities = completed; root = Path(old["root"])
    (root / "dispatcher_exit.json").unlink()
    key = old["training_jobs"][0]; path = root / "run" / key / "train_exit.json"
    exited = b.read(path); exited["actual_os_returncode"] = 7; b.write(path, exited)
    monkeypatch.setattr(b.r, "terminate_owned_child", lambda *_: pytest.fail("Other original training must remain untouched"))
    with pytest.raises(ValueError, match="actual successful OS exit"):
        b.terminal_ready(c, old, identities)
    observed = b.read(Path(c["root"]) / "upstream_failure_observation.json")
    assert observed["actual_parent_exit_record"] == exited
    assert observed["original_sha256"] == b.sha_file(path)
    assert observed["original_workers_signalled"] is False


@pytest.mark.parametrize("field,value", [("actual_os_returncode", 2), ("actual_os_returncode", True), ("timeout", True)])
def test_terminal_rejects_scientific_failure_even_with_receipt(completed, field, value):
    c, old, identities = completed; key = old["training_jobs"][0]
    root = Path(old["root"]); path = root / "run" / key / "train_exit.json"
    exited = b.read(path); exited[field] = value; b.write(path, exited)
    final = b.read(root / "final_receipt.json"); final["stages"]["train"]["exits"][key] = exited
    b.write(root / "final_receipt.json", final)
    with pytest.raises(ValueError, match="actual successful OS exit"):
        b.terminal_ready(c, old, identities)


def test_import_preserves_originals_all_nine_and_never_trains(completed, monkeypatch):
    c, old, identities = completed; root, original = Path(c["root"]), Path(old["root"])
    before = {str(p.relative_to(original)): b.sha_file(p) for p in original.rglob("*") if p.is_file()}
    monkeypatch.setattr(b.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Import never launches workers"))
    stages = b.import_completed(c, old, identities)
    assert len(stages["reference"]["exits"]) == 6 and len(stages["train"]["exits"]) == 3
    report = b.read(root / "import_receipt.json")
    assert report["new_training_worker_count"] == 0 and len(report["jobs"]) == 9
    assert before == {str(p.relative_to(original)): b.sha_file(p) for p in original.rglob("*") if p.is_file()}
    assert all(b.sha_file(root / "continuation_v3" / name) == digest for name, digest in before.items())
    assert not list((root / "run").glob("*/*_lease.json"))
    key = "inter_62"; imported_exit = b.read(root / "run" / key / "train_exit.json")
    raw_exit = b.read(original / "run" / key / "train_exit.json")
    assert raw_exit["start_ticks"] is None
    assert imported_exit == {**raw_exit, "start_ticks": identities[key]["start_ticks"]}
    assert report["jobs"][key]["exit"]["identity_supplemented"] is True
    assert report["jobs"][key]["launch"]["original_sha256"] is None
    assert report["jobs"][key]["start_ticks_proof"]["observation_path"] == "proofs/live.json"
    for job in c["jobs"]:
        path = root / "run" / job["id"]
        receipt = b.read(path / "train_receipt.json")
        assert Path(receipt["selected_checkpoint"]).is_relative_to(root)
        assert b.sha_file(receipt["selected_checkpoint"]) == receipt["checkpoint_sha256"]
        assert (path / "input_receipt.json").exists()
    with pytest.raises(ValueError, match="Existing import attempt"):
        b.import_completed(c, old, identities)


def test_relocation_changes_only_bound_absolute_paths():
    source = {"path": "/v3/run/a", "relative": "run/a", "foreign": "/v30/keep", "metric": 1.5}
    assert b._relocate(source, Path("/v3"), Path("/v4")) == {**source, "path": "/v4/run/a"}


def test_copy_rejects_symlink_and_source_mutation(tmp_path):
    src = tmp_path / "data"; src.write_bytes(b"data")
    symlink = tmp_path / "link"; symlink.symlink_to(src)
    with pytest.raises(ValueError, match="regular original"):
        b._copy(symlink, tmp_path / "out")
    with pytest.raises(ValueError, match="changed before copy"):
        b._copy(src, tmp_path / "out", "0" * 64)


def test_continuation_contract_preserves_science_and_seals_live_proof(completed, monkeypatch):
    c, old, identities = completed
    bridge_path = Path(c["source_root"]) / b.BRIDGE_FILE
    bridge_path.parent.mkdir(parents=True); bridge_path.write_text("sealed synthetic bridge")
    c["bridge_files"] = {b.BRIDGE_FILE: b.sha_file(bridge_path)}
    c["bridge_files_sha256"] = b.sha_json(c["bridge_files"])
    monkeypatch.setattr(b.r, "validate_contract", lambda _: None)
    assert b.validate_continuation(c) == (old, identities)
    for mutate in (lambda x: x.update(marker="changed"),
                   lambda x: x.update(fresh_training_authorized=True),
                   lambda x: x["continuation_from"].update(contract_sha256="wrong"),
                   lambda x: x["continuation_from"]["initial_live_observation"].update(sha256="wrong"),
                   lambda x: x.update(bridge_files_sha256="wrong")):
        changed = deepcopy(c); mutate(changed)
        with pytest.raises(ValueError):
            b.validate_continuation(changed)


def test_same_error_text_from_different_failure_is_rejected(completed):
    c, old, identities = completed
    path = Path(old["root"]) / "dispatch_failure.json"
    failure = b.read(path); failure["at"] = "later-running-identity-failure"; b.write(path, failure)
    with pytest.raises(ValueError, match="pinned initial exec race"):
        b.terminal_ready(c, old, identities)


@pytest.mark.parametrize("validation_status,gate_raises", [("complete", False), ("failed", False), ("complete", True)])
def test_dispatch_never_trains_and_test_requires_all_validation_and_gate(completed, monkeypatch, validation_status, gate_raises):
    c, old, identities = completed; root = Path(c["root"]); events = []
    monkeypatch.chdir(c["source_root"])
    monkeypatch.setattr(b, "verify_files", lambda _: (old, identities))
    monkeypatch.setattr(b, "terminal_ready", lambda *_: True)
    def imported(*_):
        b.write(root / "import_receipt.json", {"new_training_worker_count": 0})
        return {stage: {"status": "complete", "exits": {}, "receipts": {}}
                for stage in ("reference", "train")}
    monkeypatch.setattr(b, "import_completed", imported)
    def run_stage(contract, path, stage, began):
        assert stage in ("validation", "test")
        events.append(stage)
        return {"status": validation_status if stage == "validation" else "complete", "exits": {},
                "receipts": {key: {} for key in c[stage + "_jobs"]}}
    monkeypatch.setattr(b.r, "run_stage", run_stage)
    def gate(*_):
        events.append("gate")
        if gate_raises:
            raise ValueError("Synthetic replay gate failed")
    monkeypatch.setattr(b.r, "validation_gate", gate)
    monkeypatch.setattr(b.r, "verify_originals", lambda _: {"original_files_unchanged": True})
    if gate_raises:
        with pytest.raises(ValueError, match="replay gate failed"):
            b.dispatch(c, root / "recovery_contract.json")
        assert events == ["validation", "gate"]
    else:
        assert b.dispatch(c, root / "recovery_contract.json") == (0 if validation_status == "complete" else 1)
        assert events == (["validation", "gate", "test"] if validation_status == "complete" else ["validation"])
        final = b.read(root / "final_receipt.json")
        assert final["current_new_training_count"] == 0
        assert final["imported_completed_fit_count"] == 9
        assert final["held_out_test_evaluated"] is (validation_status == "complete")


def test_supervisor_early_nonzero_exit_cleans_only_new_evaluation_leases(completed, monkeypatch):
    c, _, _ = completed; root = Path(c["root"]); cleaned = []
    monkeypatch.setattr(b, "preflight", lambda _: {"status": "passed"})
    class Dispatcher:
        pid, returncode = 555, 1
        def wait(self, timeout):
            return 1
    child = Dispatcher()
    monkeypatch.setattr(b.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(b, "initial_identity", lambda *args: identity(child.pid, args[1], c["source_root"], 333))
    monkeypatch.setattr(b.r, "cleanup_leased_workers", lambda contract, path: cleaned.append((contract["root"], path)))
    assert b.supervise(c, root / "recovery_contract.json") == 1
    assert cleaned == [(c["root"], root / "recovery_contract.json")]
    exited = b.read(root / "dispatcher_exit.json")
    assert exited["actual_os_returncode"] == 1 and exited["cleanup_error"] is None


def test_exclusive_json_creation_window_waits_then_reads(tmp_path, monkeypatch):
    path = tmp_path / "exit.json"; path.write_bytes(b"")
    def finish_write(_):
        path.write_text('{"actual_os_returncode": 0}')
    monkeypatch.setattr(b.time, "sleep", finish_write)
    assert b._read_ready_json(path, {"root": str(tmp_path)}) == {"actual_os_returncode": 0}


def test_still_young_incomplete_json_waits_for_next_local_check(tmp_path, monkeypatch):
    path = tmp_path / "exit.json"; path.write_bytes(b"{")
    clock = iter([0., .3]); monkeypatch.setattr(b.time, "monotonic", lambda: next(clock))
    assert b._read_ready_json(path, {"root": str(tmp_path)}) is None
    assert not (tmp_path / "upstream_invalid_json.raw").exists()


def test_persistently_invalid_json_preserves_bytes_without_signalling(tmp_path, monkeypatch):
    path = tmp_path / "exit.json"; path.write_bytes(b"incomplete-origin")
    b.os.utime(path, (time.time() - 10, time.time() - 10))
    clock = iter([0., .3]); monkeypatch.setattr(b.time, "monotonic", lambda: next(clock))
    with pytest.raises(ValueError, match="raw bytes preserved"):
        b._read_ready_json(path, {"root": str(tmp_path)})
    assert (tmp_path / "upstream_invalid_json.raw").read_bytes() == b"incomplete-origin"
    assert b.read(tmp_path / "upstream_invalid_json.json")["original_workers_signalled"] is False


def test_historical_pid_reuse_is_not_original_worker_ownership(monkeypatch):
    monkeypatch.setattr(b.r, "proc_identity", lambda pid: identity(pid, ticks=999))
    assert b.historical_identity_absent({"pid": 123, "start_ticks": 555}) is True
    assert b.historical_identity_absent({"pid": 123, "start_ticks": 999}) is False
