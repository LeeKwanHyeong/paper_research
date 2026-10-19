"""Fabricated process ownership regression checks; no SSH, GPU calls or fits."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.scripts import titantpp_shared_history_cap_execution_v2_runtime as v2


@pytest.fixture
def gate(monkeypatch):
    r = v2._impl
    host = {"root": "/new", "source_root": "/new/source", "python": "/python",
        "gpu_uuid": "fake", "runtime_expected": {}, "paused_processes": [],
        "absolute_host_deadline_unix": 123,
        "completed_predecessor": {"root": "/parent", "contract_raw_sha256": "contract",
            "contract_canonical_sha256": "canonical", "source_files_sha256": "source",
            "dispatcher_exit_sha256": "exit", "dispatcher_identity": {"pid": 500}}}
    contract = {"hosts": {"5080": host}, "datasets": [], "training": {}, "input_files": {}, "common_condition": {}}
    parent_host = {**host, "root": "/parent", "source_root": "/parent/source"}
    parent = {**contract, "hosts": {"5080": parent_host}, "source": {"files_sha256": "source"}}
    exit_record = {"pid": 500, "actual_os_returncode": 0, "timeout": False}
    monkeypatch.setattr(v2._v1, "validate_contract", lambda *a: None)
    monkeypatch.setattr(r, "sha_file", lambda path: "exit" if Path(path).name == "dispatcher_exit.json" else "contract")
    monkeypatch.setattr(r, "sha_json", lambda value: "canonical")
    monkeypatch.setattr(r, "read", lambda path: exit_record if Path(path).name == "dispatcher_exit.json" else parent)
    original_parents = lambda: {1: 0, 21: 1}
    monkeypatch.setattr(r, "process_parents", original_parents)
    monkeypatch.setattr(r, "gpu_state", lambda *args: ({"compute_pids": []}, []))
    seen = []
    def identity(pid):
        seen.append(pid)
        if pid == 1:
            raise PermissionError("/proc/1/cwd")
        return None if pid == 500 else {"argv": ["/python"], "cwd": "/unrelated", "pid": pid}
    monkeypatch.setattr(r, "proc_identity", identity)
    monkeypatch.setattr(v2, "owned_process_parents", lambda: {21: 1})
    return contract, r, original_parents, seen


def test_old_failure_reproduced_new_gate_skips_unrelated_system_process(gate):
    contract, r, original, seen = gate
    with pytest.raises(PermissionError, match="/proc/1/cwd"):
        v2._original_gate(contract, "5080")
    seen.clear()
    assert v2.predecessor_gate(contract, "5080")["status"] == "passed"
    assert seen == [500, 21]
    assert r.process_parents is original


def test_owned_predecessor_still_blocks(gate, monkeypatch):
    contract, r, original, _ = gate
    monkeypatch.setattr(r, "proc_identity", lambda pid: None if pid == 500 else {"argv": ["/python"], "cwd": "/parent/source"})
    with pytest.raises(ValueError, match="predecessor-owned"):
        v2.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_owned_identity_permission_failure_not_ignored_and_hook_restored(gate, monkeypatch):
    contract, r, original, _ = gate
    def identity(pid):
        if pid != 500:
            raise PermissionError("owned cwd unreadable")
    monkeypatch.setattr(r, "proc_identity", identity)
    with pytest.raises(PermissionError, match="owned cwd"):
        v2.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_original_enumeration_restored_before_gpu_and_unknown_gpu_blocks(gate, monkeypatch):
    contract, r, original, _ = gate
    actual_gpu = v2.gpu_state
    monkeypatch.setattr(r, "paused_processes", lambda *a: [])
    monkeypatch.setattr(r, "gpu_snapshot", lambda *a: {"compute_pids": [1]})
    def gpu(*args):
        assert r.process_parents is original
        assert 1 in r.process_parents()
        return actual_gpu(*args)
    monkeypatch.setattr(r, "gpu_state", gpu)
    with pytest.raises(ValueError, match="Unidentified GPU"):
        v2.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_hook_restored_when_validation_fails_before_enumeration(gate, monkeypatch):
    contract, r, original, _ = gate
    monkeypatch.setattr(v2._v1, "validate_contract", lambda *a: (_ for _ in ()).throw(ValueError("bad contract")))
    with pytest.raises(ValueError, match="bad contract"):
        v2.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_main_launch_path_uses_replacement_gate_globals(gate, monkeypatch):
    contract, r, original, seen = gate
    original_read = r.read
    monkeypatch.setattr(r, "read", lambda p: contract if str(p) == "/new/contract.json" else original_read(p))
    monkeypatch.setattr(r, "launch_dispatch", lambda c, h, p: r.predecessor_gate(c, h)["status"])
    assert v2.main(["--contract", "/new/contract.json", "--host", "5080", "--mode", "launch_dispatch"]) == "passed"
    assert seen == [500, 21]
    assert r.main.__globals__["predecessor_gate"] is v2.predecessor_gate
    assert v2._v1.qualify.__globals__["_impl"] is r
    assert r.process_parents is original


def fake_entry(name, uid=7, *, stat_error=None, read_error=None):
    class Entry:
        def __init__(self):
            self.name = name
        def stat(self):
            if stat_error:
                raise stat_error
            return SimpleNamespace(st_uid=uid)
        def __truediv__(self, part):
            assert part == "stat"
            return self
        def read_text(self):
            if read_error:
                raise read_error
            return f"{name} (process name) S 5 0"
    return Entry()


def run_owned(monkeypatch, entries):
    monkeypatch.setattr(v2.os, "getuid", lambda: 7)
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(entries))
    return v2.owned_process_parents(Path("/fabricated/proc"))


def test_uid_filter_precedes_owned_stat_read(monkeypatch):
    entries = [fake_entry("1", uid=0, read_error=PermissionError("root stat")), fake_entry("21"), fake_entry("self")]
    assert run_owned(monkeypatch, entries) == {21: 5}


@pytest.mark.parametrize("where", ["stat_error", "read_error"])
@pytest.mark.parametrize("error", [FileNotFoundError(), ProcessLookupError()])
def test_vanished_process_races_skipped(monkeypatch, where, error):
    assert run_owned(monkeypatch, [fake_entry("21", **{where: error}), fake_entry("22")]) == {22: 5}


@pytest.mark.parametrize("where", ["stat_error", "read_error"])
def test_unreadable_owned_or_unestablished_owner_fails_closed(monkeypatch, where):
    with pytest.raises(PermissionError):
        run_owned(monkeypatch, [fake_entry("21", **{where: PermissionError("denied")})])


def test_frozen_science_worker_limits_schema_and_original_runtime_unchanged():
    assert v2.sha_file(v2._V1_PATH) == v2._V1_SHA
    assert v2.SCHEMA == "titantpp_shared_history_cap_execution_v1"
    assert v2.WORKER_MODULE == "paper.scripts.run_titantpp_shared_history_cap_execution"
    assert v2.MODULE == "paper.scripts.titantpp_shared_history_cap_execution_v2_runtime"
    assert v2.STAGES == ("train", "validation")
    assert v2.LOSS_MODES == ("log_mse",)
