"""Fabricated process ownership regression checks; no SSH, GPU calls or fits."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.scripts import titantpp_shared_history_cap_execution_v3_runtime as v3


@pytest.fixture
def gate(monkeypatch):
    r = v3._impl
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
    monkeypatch.setattr(v3._v1, "validate_contract", lambda *a: None)
    monkeypatch.setattr(r, "sha_file", lambda path: "exit" if Path(path).name == "dispatcher_exit.json" else "contract")
    monkeypatch.setattr(r, "sha_json", lambda value: "canonical")
    monkeypatch.setattr(r, "read", lambda path: exit_record if Path(path).name == "dispatcher_exit.json" else parent)
    original_parents = lambda: {1: 0, 1611: 1, 21: 1}
    monkeypatch.setattr(r, "process_parents", original_parents)
    monkeypatch.setattr(r, "gpu_state", lambda *args: ({"compute_pids": []}, []))
    seen = []
    def identity(pid):
        seen.append(pid)
        if pid in (1, 1611):
            raise PermissionError(f"/proc/{pid}/cwd")
        return None if pid == 500 else {"argv": ["/python"], "cwd": "/unrelated", "pid": pid}
    monkeypatch.setattr(r, "proc_identity", identity)
    monkeypatch.setattr(v3, "predecessor_process_candidates", lambda root: {21: None})
    return contract, r, original_parents, seen


def test_old_failure_reproduced_new_gate_skips_unrelated_system_process(gate):
    contract, r, original, seen = gate
    with pytest.raises(PermissionError, match="/proc/1/cwd"):
        v3._original_gate(contract, "5080")
    seen.clear()
    assert v3.predecessor_gate(contract, "5080")["status"] == "passed"
    assert seen == [500, 21]
    assert r.process_parents is original


def test_owned_predecessor_still_blocks(gate, monkeypatch):
    contract, r, original, _ = gate
    monkeypatch.setattr(r, "proc_identity", lambda pid: None if pid == 500 else {"argv": ["/python"], "cwd": "/parent/source"})
    with pytest.raises(ValueError, match="predecessor-owned"):
        v3.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_owned_identity_permission_failure_not_ignored_and_hook_restored(gate, monkeypatch):
    contract, r, original, _ = gate
    def identity(pid):
        if pid != 500:
            raise PermissionError("owned cwd unreadable")
    monkeypatch.setattr(r, "proc_identity", identity)
    with pytest.raises(PermissionError, match="owned cwd"):
        v3.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_original_enumeration_restored_before_gpu_and_unknown_gpu_blocks(gate, monkeypatch):
    contract, r, original, _ = gate
    actual_gpu = v3.gpu_state
    monkeypatch.setattr(r, "paused_processes", lambda *a: [])
    monkeypatch.setattr(r, "gpu_snapshot", lambda *a: {"compute_pids": [1]})
    def gpu(*args):
        assert r.process_parents is original
        assert 1 in r.process_parents()
        return actual_gpu(*args)
    monkeypatch.setattr(r, "gpu_state", gpu)
    with pytest.raises(ValueError, match="Unidentified GPU"):
        v3.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_hook_restored_when_validation_fails_before_enumeration(gate, monkeypatch):
    contract, r, original, _ = gate
    monkeypatch.setattr(v3._v1, "validate_contract", lambda *a: (_ for _ in ()).throw(ValueError("bad contract")))
    with pytest.raises(ValueError, match="bad contract"):
        v3.predecessor_gate(contract, "5080")
    assert r.process_parents is original


def test_main_launch_path_uses_replacement_gate_globals(gate, monkeypatch):
    contract, r, original, seen = gate
    original_read = r.read
    monkeypatch.setattr(r, "read", lambda p: contract if str(p) == "/new/contract.json" else original_read(p))
    monkeypatch.setattr(r, "launch_dispatch", lambda c, h, p: r.predecessor_gate(c, h)["status"])
    assert v3.main(["--contract", "/new/contract.json", "--host", "5080", "--mode", "launch_dispatch"]) == "passed"
    assert seen == [500, 21]
    assert r.main.__globals__["predecessor_gate"] is v3.predecessor_gate
    assert v3._v1.qualify.__globals__["_impl"] is r
    assert r.process_parents is original


def test_same_uid_unrelated_protected_process_failure_reproduced_then_excluded(gate, monkeypatch):
    contract, r, original, seen = gate
    monkeypatch.setattr(r, "process_parents", lambda: {1611: 1, 21: 1})
    with pytest.raises(PermissionError, match="/proc/1611/cwd"):
        v3._original_gate(contract, "5080")
    seen.clear()
    assert v3.predecessor_gate(contract, "5080")["status"] == "passed"
    assert seen == [500, 21]


def fake_entry(name, raw=b"", error=None):
    class Entry:
        def __init__(self):
            self.name = name
        def __truediv__(self, part):
            assert part == "cmdline", "Unrelated process stat/cwd read attempted"
            return self
        def read_bytes(self):
            if error:
                raise error
            return raw
    return Entry()


def run_candidates(monkeypatch, entries):
    monkeypatch.setattr(Path, "iterdir", lambda self: iter(entries))
    return v3.predecessor_process_candidates("/old/campaign", Path("/fabricated/proc"))


@pytest.mark.parametrize("raw", [
    b"python\x00--contract\x00/old/campaign/execution_contract.json\x00",
    b"python\x00/old/campaign/source/runner.py\x00",
    b"python\x00/old/campaign\x00",
    b"python\x00-c\x00child(['python','--contract','/old/campaign/execution_contract.json'])",
    b'python\x00--contract="/old/campaign/execution_contract.json"\x00',
    b"python\x00/old/campaign_extra/execution_contract.json\x00",
    b"python\x00/prefix/old/campaign/execution_contract.json\x00",
    b"python\x00/old/campaign2\x00",
])
def test_root_marker_including_quoted_or_similar_paths_is_conservatively_inspected(monkeypatch, raw):
    assert run_candidates(monkeypatch, [fake_entry("21", raw)]) == {21: None}


@pytest.mark.parametrize("raw", [
    b"systemd\x00", b"unrelated-user-daemon\x00", b"",
    b"python\x00/old/camp/execution_contract.json\x00",
])
def test_unrelated_system_or_same_user_processes_without_full_marker_excluded(monkeypatch, raw):
    assert run_candidates(monkeypatch, [fake_entry("1611", raw)]) == {}


@pytest.mark.parametrize("error", [FileNotFoundError(), ProcessLookupError()])
def test_vanished_cmdline_process_races_skipped(monkeypatch, error):
    assert run_candidates(monkeypatch, [fake_entry("21", error=error), fake_entry("22", b"/old/campaign\x00")]) == {22: None}


def test_unreadable_cmdline_fails_closed(monkeypatch):
    with pytest.raises(PermissionError):
        run_candidates(monkeypatch, [fake_entry("21", error=PermissionError("cmdline unreadable"))])


def test_non_pid_proc_entries_not_read(monkeypatch):
    assert run_candidates(monkeypatch, [fake_entry("self", error=AssertionError("not numeric"))]) == {}


def test_frozen_science_worker_schema_and_original_runtime_unchanged():
    assert v3.sha_file(v3._V1_PATH) == v3._V1_SHA
    assert v3.SCHEMA == "titantpp_shared_history_cap_execution_v1"
    assert v3.WORKER_MODULE == "paper.scripts.run_titantpp_shared_history_cap_execution"
    assert v3.MODULE == "paper.scripts.titantpp_shared_history_cap_execution_v3_runtime"
    assert v3.STAGES == ("train", "validation")
    assert v3.LOSS_MODES == ("log_mse",)


