"""Fabricated evidence tests: no native training, inference, SSH or collection."""
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from paper.scripts import advance_titantpp_shared_external_extra42_recovery_v2 as recovery


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    report, bundle = root / "reports/extra42", root / "search_artifacts/extra42"
    report.mkdir(parents=True); bundle.mkdir(parents=True)
    control = report / "recovery_v2"; control.mkdir()
    monkeypatch.setattr(recovery, "ROOT", root)
    monkeypatch.setattr(recovery, "REPORT", control)
    monkeypatch.setattr(recovery, "BUNDLE", bundle / "recovery_v2")
    monkeypatch.setattr(recovery.original, "ROOT", root)
    monkeypatch.setattr(recovery.original, "REPORT", report)
    monkeypatch.setattr(recovery.original, "BUNDLE", bundle)
    clock = [1000000.]
    monkeypatch.setattr(recovery.time, "time", lambda: clock[0])
    identity = {"pid": 616624, "start_ticks": 87087471, "argv": ["python", "-m", "original_dispatcher"],
        "cwd": "/native/source", "pgid": 616624}
    contract = {"hosts": {"5080": {"root": "/native", "source_root": "/native/source", "python": "/env/python",
        "absolute_host_deadline_unix": 500.}}, "science": "unchanged"}
    monkeypatch.setattr(recovery.original, "verify_local", lambda value: contract if value == "original-seal" else (_ for _ in ()).throw(ValueError("bad original seal")))
    for path in (report / "QUEUE_FAILURE.json", report / "recovery_v1/FAILURE.json"):
        dump(path, {"status": "failed", "automatic_retry": False, "original_error": str(path)})
    for name in ("FINISH_STARTED.json", "ACTION_REQUIRED.json", "recovery_v1/WAIT_TRANSPORT.json", "recovery_v1/FINISH_CLAIM.json"):
        dump(report / name, {"historical": name})
    (report / "finish.stderr.log").write_text("original broken pipe raw\n")
    (report / "recovery_v1/completion.stderr.log").write_text("second broken pipe raw\n")
    source = root / "control.py"; source.write_text("unchanged control")
    dump(control / "AUTHORIZATION.json", {"status": "approved", "recovery_only": True,
        "short_terminal_checks": True, "old_finisher_absent": True,
        "read_only_transport_recheck_authorized": True, "maximum_consecutive_transport_errors": 3})
    auth_sha = recovery.sha(control / "AUTHORIZATION.json")
    binding = {"files": {"control.py": recovery.sha(source)}, "original_integrity_sha256": "original-seal",
        "expected_original_failure_sha256": recovery.sha(report / "QUEUE_FAILURE.json"),
        "expected_previous_failure_sha256": recovery.sha(report / "recovery_v1/FAILURE.json"), "deadline": 500.,
        "recovery_root": str(control), "recovery_bundle_root": str(bundle / "recovery_v2"),
        "authorization_sha256": auth_sha, "dispatcher_identity": identity}
    dump(control / "CONTROL_BINDINGS.json", binding)
    old = {p: p.read_bytes() for p in report.rglob("*") if p.is_file() and not p.is_relative_to(control)}
    return SimpleNamespace(root=root, report=report, bundle=bundle, control=control, contract=contract,
        binding=binding, binding_sha=recovery.sha(control / "CONTROL_BINDINGS.json"), auth_sha=auth_sha, old=old, clock=clock)


def terminal(p, code=0):
    i = p.binding["dispatcher_identity"]
    return {"actual_os_returncode": code, "timeout": False, "actual_child_exit_observed": True,
        "pid": i["pid"], "start_ticks": i["start_ticks"], "command": i["argv"], "launch_sha256": "launch"}


def payload(p, status="waiting"):
    value = {"status": status, "observed_at": "2026-10-21T00:00:00+00:00", "contract_sha256": recovery.original.canonical(p.contract),
        "dispatcher_identity": deepcopy(p.binding["dispatcher_identity"]), "read_only_check": True, "deadline": 500.}
    if status == "waiting": value["dispatcher_current_identity"] = deepcopy(p.binding["dispatcher_identity"])
    else: value["dispatcher_exit"] = terminal(p, 1 if status == "terminal_failure" else 0)
    return value


def mock_check(monkeypatch, p, status="waiting", *, code=0, malformed=False, timeout=False):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        p.clock[0] += 900.
        if timeout: raise subprocess.TimeoutExpired(command, 45, output=b"partial raw output", stderr=b"network stall")
        stdout = b"malformed" if malformed else (json.dumps(payload(p, status)) + "\n").encode()
        return SimpleNamespace(returncode=code, stdout=stdout, stderr=b"broken pipe" if code else b"")
    monkeypatch.setattr(recovery.subprocess, "run", run)
    return calls


def no_collect(monkeypatch):
    monkeypatch.setattr(recovery.original, "collect", lambda *a: pytest.fail("Collection was not authorized by terminal exit0"))
    monkeypatch.setattr(recovery.comparison, "run", lambda *a, **kw: pytest.fail("No comparison before collection"))


def test_waiting_uses_one_short_ssh_and_returns_immediately(prepared, monkeypatch):
    p = prepared; calls = mock_check(monkeypatch, p); no_collect(monkeypatch)
    result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "waiting" and result["remote_calls"] == 1 and result["collection_calls"] == 0
    assert result["consecutive_transport_errors"] == 0 and len(calls) == 1
    assert calls[0][1]["timeout"] == 45
    program = recovery.terminal_program(p.contract, p.binding)
    for forbidden in ("sleep", "while ", "Popen", "run_stage(", "kill(", "killpg", "optimizer", "r.write(", "launch_dispatch"):
        assert forbidden not in program
    assert all(path.read_bytes() == raw for path, raw in p.old.items())
    pointer = recovery.read(p.control / "LATEST.json")
    assert recovery.sha(pointer["round_path"]) == pointer["round_sha256"]
    assert not (p.control / "COLLECT_INTENT.json").exists()


def test_waiting_permits_next_authorized_round_without_overwriting(prepared, monkeypatch):
    p = prepared; calls = mock_check(monkeypatch, p); no_collect(monkeypatch)
    first = recovery.advance(p.binding_sha, p.auth_sha)
    original_round = recovery.read(p.control / "LATEST.json")["round_path"]
    original_bytes = Path(original_round).read_bytes()
    second = recovery.advance(p.binding_sha, p.auth_sha)
    assert first["status"] == second["status"] == "waiting" and len(calls) == 2
    assert len(list((p.control / "rounds").iterdir())) == 2
    assert Path(original_round).read_bytes() == original_bytes


@pytest.mark.parametrize("transport_error", [False, True])
def test_early_manual_or_heartbeat_call_waits_locally(prepared, monkeypatch, transport_error):
    p = prepared; calls = mock_check(monkeypatch, p, code=255 if transport_error else 0); no_collect(monkeypatch)
    recovery.advance(p.binding_sha, p.auth_sha)
    p.clock[0] -= 840.
    result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "waiting_next_check" and result["remote_calls"] == 0 and len(calls) == 1
    assert result["remaining_seconds"] == 840.
    assert not (p.control / "FAILURE.json").exists()


def test_round_written_before_pointer_update_cannot_roll_back_error_count(prepared, monkeypatch):
    p = prepared; calls = mock_check(monkeypatch, p, code=255); no_collect(monkeypatch)
    recovery.advance(p.binding_sha, p.auth_sha)
    pointer = (p.control / "LATEST.json").read_bytes()
    recovery.advance(p.binding_sha, p.auth_sha)
    (p.control / "LATEST.json").write_bytes(pointer)  # Publication crashed after ROUND2 but before pointer2.
    with pytest.raises(ValueError, match="Unpublished prior round"):
        recovery.advance(p.binding_sha, p.auth_sha)
    assert len(calls) == 2
    assert recovery.read(p.control / "FAILURE.json")["phase"] == "existing_work_guard"


@pytest.mark.parametrize("mutation", ["wrong_binding", "empty_inventory", "source_changed", "authorization_changed",
    "previous_failure_changed", "original_failure_changed", "deadline_changed", "identity_missing", "recheck_unapproved"])
def test_bad_binding_blocks_before_remote(prepared, monkeypatch, mutation):
    p = prepared; b = deepcopy(p.binding); binding_sha = p.binding_sha
    if mutation == "wrong_binding": binding_sha = "0" * 64
    elif mutation == "empty_inventory": b["files"] = {}
    elif mutation == "source_changed": (p.root / "control.py").write_text("tampered")
    elif mutation == "authorization_changed": dump(p.control / "AUTHORIZATION.json", {"status": "unapproved"})
    elif mutation == "previous_failure_changed": dump(p.report / "recovery_v1/FAILURE.json", {"status": "failed", "error": "tampered"})
    elif mutation == "original_failure_changed": dump(p.report / "QUEUE_FAILURE.json", {"status": "failed", "error": "tampered"})
    elif mutation == "deadline_changed": b["deadline"] = 501.
    elif mutation == "identity_missing": b.pop("dispatcher_identity")
    elif mutation == "recheck_unapproved":
        auth = recovery.read(p.control / "AUTHORIZATION.json"); auth["read_only_transport_recheck_authorized"] = False
        dump(p.control / "AUTHORIZATION.json", auth)
        p.auth_sha = b["authorization_sha256"] = recovery.sha(p.control / "AUTHORIZATION.json")
    if b != p.binding:
        dump(p.control / "CONTROL_BINDINGS.json", b); binding_sha = recovery.sha(p.control / "CONTROL_BINDINGS.json")
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **kw: pytest.fail("Remote must not occur"))
    with pytest.raises(ValueError): recovery.advance(binding_sha, p.auth_sha)
    assert not (p.control / "rounds").exists()


def test_busy_advance_returns_without_remote_or_failure(prepared, monkeypatch):
    p = prepared
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **kw: pytest.fail("Busy call must not connect"))
    with (p.control / "advance.lock").open("a+") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "busy" and result["remote_calls"] == 0
    assert not (p.control / "FAILURE.json").exists() and not (p.control / "rounds").exists()


def test_transport255_is_preserved_without_in_round_retry(prepared, monkeypatch):
    p = prepared; calls = mock_check(monkeypatch, p, code=255); no_collect(monkeypatch)
    result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "transport_error" and result["consecutive_transport_errors"] == 1 and len(calls) == 1
    assert not (p.control / "FAILURE.json").exists()
    t = recovery.read(recovery.latest_round()["transport_path"])
    assert t["actual_transport_exit"] == 255 and t["transport_timeout"] is False
    assert Path(t["stderr_path"]).read_bytes() == b"broken pipe"
    assert all(path.read_bytes() == raw for path, raw in p.old.items())


def test_timeout_preserves_partial_raw_and_defers_next_check(prepared, monkeypatch):
    p = prepared; calls = mock_check(monkeypatch, p, timeout=True); no_collect(monkeypatch)
    result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "transport_error" and result["transport_timeout"] is True and len(calls) == 1
    transport = recovery.read(result["transport_path"])
    assert transport["actual_transport_exit"] is None and transport["transport_timeout"] is True
    assert Path(transport["stdout_path"]).read_bytes() == b"partial raw output"
    assert Path(transport["stderr_path"]).read_bytes() == b"network stall"


def test_three_consecutive_transport_errors_persistently_block_future_remote(prepared, monkeypatch):
    p = prepared; calls = mock_check(monkeypatch, p, code=255); no_collect(monkeypatch)
    assert recovery.advance(p.binding_sha, p.auth_sha)["consecutive_transport_errors"] == 1
    assert recovery.advance(p.binding_sha, p.auth_sha)["consecutive_transport_errors"] == 2
    with pytest.raises(ValueError, match="Three consecutive"): recovery.advance(p.binding_sha, p.auth_sha)
    assert len(calls) == 3 and recovery.read(p.control / "FAILURE.json")["phase"] == "transport_error_limit"
    result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "blocked_failure" and len(calls) == 3


def test_successful_waiting_resets_transport_error_counter(prepared, monkeypatch):
    p = prepared; mock_check(monkeypatch, p, code=255); no_collect(monkeypatch)
    assert recovery.advance(p.binding_sha, p.auth_sha)["consecutive_transport_errors"] == 1
    mock_check(monkeypatch, p)
    assert recovery.advance(p.binding_sha, p.auth_sha)["consecutive_transport_errors"] == 0
    mock_check(monkeypatch, p, code=255)
    assert recovery.advance(p.binding_sha, p.auth_sha)["consecutive_transport_errors"] == 1


@pytest.mark.parametrize("kind", ["remote_error", "malformed", "terminal_failure", "identity_mismatch"])
def test_nontransport_failure_blocks_immediately_without_collection(prepared, monkeypatch, kind):
    p = prepared; no_collect(monkeypatch)
    calls = mock_check(monkeypatch, p, status="terminal_failure" if kind == "terminal_failure" else "waiting",
        code=1 if kind == "remote_error" else 0, malformed=kind == "malformed")
    if kind == "identity_mismatch":
        def run(*a, **kwargs):
            calls.append((a, kwargs)); value = payload(p); value["dispatcher_current_identity"]["pid"] = 1
            return SimpleNamespace(returncode=0, stdout=json.dumps(value).encode(), stderr=b"")
        monkeypatch.setattr(recovery.subprocess, "run", run)
    with pytest.raises((ValueError, json.JSONDecodeError)): recovery.advance(p.binding_sha, p.auth_sha)
    assert (p.control / "FAILURE.json").exists() and len(calls) == 1
    assert recovery.advance(p.binding_sha, p.auth_sha)["status"] == "blocked_failure" and len(calls) == 1


@pytest.mark.parametrize("partial", ["retrieval", "collect_intent", "new_results", "incomplete_round"])
def test_partial_work_rejects_without_remote(prepared, monkeypatch, partial):
    p = prepared
    if partial == "retrieval": (p.bundle / "retrieval/5080").mkdir(parents=True)
    elif partial == "collect_intent": dump(p.control / "COLLECT_INTENT.json", {"exclusive": True})
    elif partial == "new_results": dump(p.report / "NEW_SEED_RESULTS.json", {"partial": True})
    else: dump(p.control / "rounds/abandoned/INTENT.json", {"uncertain": True})
    monkeypatch.setattr(recovery.subprocess, "run", lambda *a, **kw: pytest.fail("Partial work must block before remote"))
    with pytest.raises(ValueError): recovery.advance(p.binding_sha, p.auth_sha)
    assert recovery.read(p.control / "FAILURE.json")["phase"] == "existing_work_guard"


def mock_collection(monkeypatch, p, *, comparison_failure=False):
    calls = []
    def collect(c):
        calls.append("collect")
        assert recovery.original.BUNDLE == p.bundle and recovery.original._transport.BUNDLE == recovery.BUNDLE
        assert (p.control / "COLLECT_INTENT.json").exists()
        # A competing call during collection must see the lock, never connect/write FAILURE.
        assert recovery.advance(p.binding_sha, p.auth_sha)["status"] == "busy"
        for name in ("retrieval/final_receipt.json", "retrieval/5080/receipt.json", "retrieval/5080/manifest.json"):
            dump(p.bundle / name, {"status": "complete"})
        (p.bundle / "retrieval/5080/originals").mkdir()
        (p.bundle / "retrieval/5080/original.tar.gz").write_bytes(b"synthetic archive")
        dump(p.report / "NEW_SEED_RESULTS.json", {"status": "complete"})
        return {"status": "complete", "new_fit_count": 42}
    def compare(root, *, check_only=False):
        calls.append("check" if check_only else "compare")
        if comparison_failure: raise ValueError("Synthetic aggregate failure")
        if check_only: return {"status": "verified_ready"}
        paths = [p.report / "five_seed_aggregation_v1/FIVE_SEED_RESULTS.json", p.report / "five_seed_aggregation_v1/FIVE_SEED_RESULTS.md"]
        dump(paths[0], {"status": "complete"}); paths[1].write_text("synthetic aggregate")
        return {"status": "complete", "output_files": [str(path) for path in paths]}
    def verify(c, originals):
        calls.append("verify_collected")
        assert originals == p.bundle / "retrieval/5080/originals"
        return {"status": "complete"}
    monkeypatch.setattr(recovery.original, "collect", collect)
    monkeypatch.setattr(recovery.original, "verify_collected", verify)
    monkeypatch.setattr(recovery.comparison, "run", compare)
    return calls


def test_terminal_exit0_collects_and_compares_exactly_once(prepared, monkeypatch):
    p = prepared; remote = mock_check(monkeypatch, p, status="terminal"); calls = mock_collection(monkeypatch, p)
    original_transport_bundle = recovery.original._transport.BUNDLE
    result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "complete" and result["collection_calls"] == 1
    assert len(remote) == 1 and calls == ["collect", "check", "compare"]
    assert recovery.original._transport.BUNDLE == original_transport_bundle
    assert (p.control / "COMPLETE.json").exists() and (p.report / "QUEUE_COMPLETE.json").exists()
    assert all(path.read_bytes() == raw for path, raw in p.old.items())
    assert not (p.control / "FAILURE.json").exists()


def test_complete_cache_reverifies_locally_without_any_ssh(prepared, monkeypatch):
    p = prepared; remote = mock_check(monkeypatch, p, status="terminal"); calls = mock_collection(monkeypatch, p)
    recovery.advance(p.binding_sha, p.auth_sha)
    result = recovery.advance(p.binding_sha, p.auth_sha)
    assert result["status"] == "complete" and result["cached"] is True and result["remote_calls"] == 0
    assert len(remote) == 1 and calls == ["collect", "check", "compare", "verify_collected", "check"]


@pytest.mark.parametrize("file", ["archive", "new_results", "comparison", "empty_inventory"])
def test_complete_cache_rejects_tampered_evidence_without_remote(prepared, monkeypatch, file):
    p = prepared; remote = mock_check(monkeypatch, p, status="terminal"); mock_collection(monkeypatch, p)
    recovery.advance(p.binding_sha, p.auth_sha)
    if file == "archive": (p.bundle / "retrieval/5080/original.tar.gz").write_bytes(b"tampered")
    elif file == "new_results": dump(p.report / "NEW_SEED_RESULTS.json", {"tampered": True})
    elif file == "comparison": (p.report / "five_seed_aggregation_v1/FIVE_SEED_RESULTS.md").write_text("tampered")
    else:
        complete = recovery.read(p.control / "COMPLETE.json"); complete["files"] = {}
        dump(p.control / "COMPLETE.json", complete)
    with pytest.raises(ValueError): recovery.advance(p.binding_sha, p.auth_sha)
    assert len(remote) == 1


def test_comparison_failure_preserves_collection_and_blocks_all_remote(prepared, monkeypatch):
    p = prepared; remote = mock_check(monkeypatch, p, status="terminal"); mock_collection(monkeypatch, p, comparison_failure=True)
    with pytest.raises(ValueError, match="Synthetic aggregate failure"): recovery.advance(p.binding_sha, p.auth_sha)
    failure = recovery.read(p.control / "FAILURE.json")
    assert failure["phase"] == "five_seed_comparison" and failure["collection_complete"] is True
    assert (p.report / "QUEUE_COMPLETE.json").exists() and (p.control / "COLLECTION_COMPLETE.json").exists()
    assert recovery.advance(p.binding_sha, p.auth_sha)["status"] == "blocked_failure" and len(remote) == 1


def test_collection_failure_never_reenters_collector(prepared, monkeypatch):
    p = prepared; remote = mock_check(monkeypatch, p, status="terminal"); calls = []
    def collect(c):
        calls.append("collect"); (p.bundle / "retrieval/5080").mkdir(parents=True)
        raise ValueError("Synthetic collection failure")
    monkeypatch.setattr(recovery.original, "collect", collect)
    with pytest.raises(ValueError, match="Synthetic collection failure"): recovery.advance(p.binding_sha, p.auth_sha)
    assert recovery.advance(p.binding_sha, p.auth_sha)["status"] == "blocked_failure"
    assert calls == ["collect"] and len(remote) == 1 and (p.control / "COLLECT_INTENT.json").exists()


@pytest.mark.parametrize("exists", [False, True])
def test_native_program_is_single_check_and_terminal_precedes_deadline(prepared, monkeypatch, capsys, exists):
    p = prepared; i = p.binding["dispatcher_identity"]
    values = {"execution_contract.json": p.contract,
        "start.json": {"pid": i["pid"], "identity": i, "contract_sha256": recovery.original.canonical(p.contract)},
        "dispatcher_launch.json": {"pid": i["pid"], "identity": i, "command": i["argv"]},
        "dispatcher_exit.json": terminal(p)}
    runtime = recovery.original.runtime
    monkeypatch.setattr(runtime, "read", lambda path: values[Path(path).name])
    monkeypatch.setattr(runtime, "sha_file", lambda path: "launch")
    monkeypatch.setattr(runtime, "proc_identity", lambda pid: i)
    monkeypatch.setattr(Path, "exists", lambda self: exists)
    # A past deadline may prevent a fresh waiting state, but never collection of a successful exit.
    import time
    monkeypatch.setattr(time, "time", lambda: 1000. if exists else 1.)
    exec(recovery.terminal_program(p.contract, p.binding), {})
    response = json.loads(capsys.readouterr().out)
    assert response["status"] == ("terminal" if exists else "waiting")
