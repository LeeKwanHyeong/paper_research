"""Recovery control tests use fabricated evidence and mocked SSH/collection only."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from paper.scripts import recover_titantpp_shared_external_extra42 as recovery


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path
    report = root / "reports/extra42"
    bundle = root / "search_artifacts/extra42"
    report.mkdir(parents=True); bundle.mkdir(parents=True)
    control = report / "recovery_v1"; control.mkdir()
    monkeypatch.setattr(recovery, "ROOT", root)
    monkeypatch.setattr(recovery, "REPORT", control)
    monkeypatch.setattr(recovery, "BUNDLE", bundle / "recovery_v1")
    monkeypatch.setattr(recovery.original, "ROOT", root)
    monkeypatch.setattr(recovery.original, "REPORT", report)
    monkeypatch.setattr(recovery.original, "BUNDLE", bundle)
    monkeypatch.setattr(recovery.time, "time", lambda: 100.)
    identity = {"pid": 1234, "start_ticks": 99, "argv": ["python", "-m", "original_dispatcher"],
        "cwd": "/native/source", "pgid": 1234}
    contract = {"hosts": {"5080": {"root": "/native", "source_root": "/native/source", "python": "/env/python",
        "absolute_host_deadline_unix": 500.}}, "science": "unchanged"}
    monkeypatch.setattr(recovery.original, "verify_local", lambda value: contract if value == "original-seal" else (_ for _ in ()).throw(ValueError("bad original seal")))
    dump(report / "QUEUE_FAILURE.json", {"status": "failed", "automatic_retry": False, "error": "old broken pipe"})
    for name in ("FINISH_STARTED.json", "ACTION_REQUIRED.json"):
        dump(report / name, {"historical": name})
    (report / "finish.stderr.log").write_text("old broken pipe raw\n")
    source = root / "new_control.py"; source.write_text("unchanged control")
    dump(control / "AUTHORIZATION.json", {"status": "approved", "recovery_only": True, "old_finisher_absent": True})
    auth_sha = recovery.sha(control / "AUTHORIZATION.json")
    binding = {"files": {"new_control.py": recovery.sha(source)}, "original_integrity_sha256": "original-seal",
        "expected_original_failure_sha256": recovery.sha(report / "QUEUE_FAILURE.json"), "deadline": 500.,
        "recovery_root": str(control), "recovery_bundle_root": str(bundle / "recovery_v1"),
        "authorization_sha256": auth_sha, "dispatcher_identity": identity}
    dump(control / "CONTROL_BINDINGS.json", binding)
    return SimpleNamespace(root=root, report=report, bundle=bundle, control=control, contract=contract,
        binding=binding, binding_sha=recovery.sha(control / "CONTROL_BINDINGS.json"), auth_sha=auth_sha,
        original_bytes={p: p.read_bytes() for p in report.iterdir() if p.is_file()})


def intent(p):
    dump(p.control / "START_INTENT.json", {"binding_sha256": p.binding_sha, "authorization_sha256": p.auth_sha})


def terminal(p):
    i = p.binding["dispatcher_identity"]
    return {"actual_os_returncode": 0, "timeout": False, "actual_child_exit_observed": True,
        "pid": i["pid"], "start_ticks": i["start_ticks"], "command": i["argv"], "launch_sha256": "launch"}


def wait_lines(p):
    return [{"status": "READY", "contract_sha256": recovery.original.canonical(p.contract),
        "dispatcher_identity": p.binding["dispatcher_identity"], "read_only_wait": True, "deadline": 500.},
        {"status": "TERMINAL", "dispatcher_exit": terminal(p)}]


def mock_wait_child(monkeypatch, p, *, code=0, lines=None, timeout=False):
    commands = []
    class Child:
        pid = 9191
        killed = False
        def __init__(self, command, **kwargs):
            commands.append(command)
            value = wait_lines(p) if lines is None else lines
            kwargs["stdout"].write(("\n".join(json.dumps(x) for x in value) + "\n").encode())
            kwargs["stdout"].flush()
            kwargs["stderr"].write(b"broken pipe" if code else b"")
        def wait(self, timeout=None):
            if timeout is not None and timeout > 15 and timeout_flag[0]:
                timeout_flag[0] = False
                raise subprocess.TimeoutExpired("ssh", timeout)
            return code
        def kill(self):
            self.killed = True
    timeout_flag = [timeout]
    monkeypatch.setattr(recovery.subprocess, "Popen", Child)
    return commands


def test_verify_checks_original_seal_without_remote(prepared):
    b, c = recovery.verify(prepared.binding_sha, prepared.auth_sha)
    assert b == prepared.binding and c == prepared.contract


@pytest.mark.parametrize("mutation", ["wrong_binding", "empty_inventory", "source_changed", "authorization_changed",
    "failure_changed", "deadline_changed", "identity_missing", "old_finisher_present", "original_seal"])
def test_binding_rejects_before_any_process(prepared, monkeypatch, mutation):
    p = prepared
    b = deepcopy(p.binding)
    binding_sha = p.binding_sha
    if mutation == "wrong_binding": binding_sha = "0" * 64
    elif mutation == "empty_inventory": b["files"] = {}
    elif mutation == "source_changed": (p.root / "new_control.py").write_text("tampered")
    elif mutation == "authorization_changed": dump(p.control / "AUTHORIZATION.json", {"status": "unapproved"})
    elif mutation == "failure_changed": dump(p.report / "QUEUE_FAILURE.json", {"status": "failed", "error": "tampered"})
    elif mutation == "deadline_changed": b["deadline"] = 501.
    elif mutation == "identity_missing": b.pop("dispatcher_identity")
    elif mutation == "old_finisher_present":
        dump(p.control / "AUTHORIZATION.json", {"status": "approved", "recovery_only": True, "old_finisher_absent": False})
        p.auth_sha = b["authorization_sha256"] = recovery.sha(p.control / "AUTHORIZATION.json")
    elif mutation == "original_seal": b["original_integrity_sha256"] = "changed"
    if b != p.binding:
        dump(p.control / "CONTROL_BINDINGS.json", b)
        binding_sha = recovery.sha(p.control / "CONTROL_BINDINGS.json")
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **kw: pytest.fail("No process before verification"))
    with pytest.raises(ValueError): recovery.start(binding_sha, p.auth_sha)
    assert not (p.control / "START_INTENT.json").exists()


@pytest.mark.parametrize("existing", ["intent", "retrieval", "queue_complete", "new_results"])
def test_start_refuses_duplicate_or_partial_collection(prepared, monkeypatch, existing):
    p = prepared
    if existing == "intent": intent(p)
    elif existing == "retrieval": (p.bundle / "retrieval").mkdir()
    else: dump(p.report / ("QUEUE_COMPLETE.json" if existing == "queue_complete" else "NEW_SEED_RESULTS.json"), {})
    monkeypatch.setattr(recovery.subprocess, "Popen", lambda *a, **kw: pytest.fail("Duplicate must not spawn"))
    with pytest.raises(ValueError): recovery.start(p.binding_sha, p.auth_sha)


def test_start_spawns_only_local_finish_once_preserving_old_failure(prepared, monkeypatch):
    p = prepared; calls = []
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=9191)
    monkeypatch.setattr(recovery.subprocess, "Popen", popen)
    result = recovery.start(p.binding_sha, p.auth_sha)
    assert result["pid"] == 9191 and len(calls) == 1
    assert calls[0][0][calls[0][0].index("--mode") + 1] == "finish"
    assert calls[0][1]["start_new_session"] is True
    with pytest.raises(ValueError): recovery.start(p.binding_sha, p.auth_sha)
    assert len(calls) == 1
    assert all(path.read_bytes() == raw for path, raw in p.original_bytes.items())


def test_wait_records_ready_terminal_keepalive_and_no_training_command(prepared, monkeypatch):
    p = prepared; commands = mock_wait_child(monkeypatch, p)
    result = recovery.wait_connection(p.contract, p.binding)
    assert result["actual_transport_exit"] == 0 and not result["transport_timeout"]
    assert "ServerAliveInterval=30" in commands[0] and "ServerAliveCountMax=6" in commands[0]
    program = recovery.wait_program(p.contract, p.binding)
    for forbidden in ("Popen", "launch_dispatch", "run_stage(", "kill(", "killpg", "optimizer", "r.write("):
        assert forbidden not in program
    assert "proc_identity" in program and "start.json" in program and "dispatcher_launch.json" in program
    assert json.loads((recovery.BUNDLE / "transport/5080/wait.stdout").read_text().splitlines()[0])["status"] == "READY"


@pytest.mark.parametrize("kind", ["nonzero", "timeout", "missing_ready", "wrong_ready", "failed_dispatcher"])
def test_wait_failure_never_collects_and_preserves_raw(prepared, monkeypatch, kind):
    p = prepared; intent(p); lines = wait_lines(p)
    if kind == "missing_ready": lines = lines[1:]
    if kind == "wrong_ready": lines[0]["dispatcher_identity"]["pid"] = 7777
    if kind == "failed_dispatcher": lines[-1]["dispatcher_exit"]["actual_os_returncode"] = 1
    mock_wait_child(monkeypatch, p, code=255 if kind == "nonzero" else 0, lines=lines, timeout=kind == "timeout")
    monkeypatch.setattr(recovery.original, "collect", lambda *a: pytest.fail("Failed SSH must not collect"))
    with pytest.raises(ValueError): recovery.finish(p.binding_sha, p.auth_sha)
    failure = recovery.read(p.control / "FAILURE.json")
    assert failure["phase"] == "completion_wait" and failure["automatic_retry"] is False
    assert (p.control / "WAIT_TRANSPORT.json").exists()
    assert all(path.read_bytes() == raw for path, raw in p.original_bytes.items())


def test_success_collects_once_and_compares_locally(prepared, monkeypatch):
    p = prepared; intent(p); mock_wait_child(monkeypatch, p)
    calls = []
    original_transport_bundle = recovery.original._transport.BUNDLE
    def collect(c):
        calls.append("collect")
        assert recovery.original.BUNDLE == p.bundle
        assert recovery.original._transport.BUNDLE == recovery.BUNDLE
        (p.bundle / "retrieval").mkdir()
        return {"status": "complete", "new_fit_count": 42}
    def compare(root, *, check_only=False):
        calls.append("check" if check_only else "compare")
        if check_only: return {"status": "verified_ready"}
        output = p.report / "five_seed_aggregation_v1/FIVE_SEED_RESULTS.json"
        dump(output, {"complete": True})
        return {"status": "complete", "output_files": [str(output)]}
    monkeypatch.setattr(recovery.original, "collect", collect)
    monkeypatch.setattr(recovery.comparison, "run", compare)
    result = recovery.finish(p.binding_sha, p.auth_sha)
    assert result["status"] == "complete" and calls == ["collect", "check", "compare"]
    assert recovery.original._transport.BUNDLE == original_transport_bundle
    assert recovery.read(p.report / "QUEUE_COMPLETE.json")["original_failure_preserved"] is True
    assert (p.control / "COLLECTION_COMPLETE.json").exists() and (p.control / "COMPLETE.json").exists()
    assert all(path.read_bytes() == raw for path, raw in p.original_bytes.items())
    with pytest.raises(ValueError): recovery.finish(p.binding_sha, p.auth_sha)
    assert calls == ["collect", "check", "compare"]
    assert not (p.control / "FAILURE.json").exists()


def test_duplicate_finish_cannot_poison_active_owner(prepared, monkeypatch):
    p = prepared; intent(p)
    dump(p.control / "FINISH_CLAIM.json", {"pid": 9191, "binding_sha256": p.binding_sha})
    monkeypatch.setattr(recovery, "wait_connection", lambda *a: pytest.fail("Duplicate must not connect"))
    with pytest.raises(ValueError, match="Existing replacement wait"):
        recovery.finish(p.binding_sha, p.auth_sha)
    assert not (p.control / "FAILURE.json").exists()


def test_racing_finish_claim_loser_does_not_write_failure(prepared, monkeypatch):
    p = prepared; intent(p); real_write = recovery.write
    def raced_write(path, value):
        if path.name == "FINISH_CLAIM.json":
            real_write(path, {"pid": 9191})
        return real_write(path, value)
    monkeypatch.setattr(recovery, "write", raced_write)
    monkeypatch.setattr(recovery, "wait_connection", lambda *a: pytest.fail("Claim loser must not connect"))
    with pytest.raises(FileExistsError): recovery.finish(p.binding_sha, p.auth_sha)
    assert not (p.control / "FAILURE.json").exists()


def test_comparison_failure_is_separate_from_successful_collection(prepared, monkeypatch):
    p = prepared; intent(p); mock_wait_child(monkeypatch, p)
    monkeypatch.setattr(recovery.original, "collect", lambda c: {"status": "complete"})
    monkeypatch.setattr(recovery.comparison, "run", lambda *a, **kw: (_ for _ in ()).throw(ValueError("old reference missing")))
    with pytest.raises(ValueError, match="old reference missing"): recovery.finish(p.binding_sha, p.auth_sha)
    assert (p.report / "QUEUE_COMPLETE.json").exists() and (p.control / "COLLECTION_COMPLETE.json").exists()
    assert not (p.control / "COMPLETE.json").exists()
    f = recovery.read(p.control / "FAILURE.json")
    assert f["collection_complete"] is True and f["phase"] == "five_seed_comparison"


def test_read_only_native_wait_checks_original_identity_before_ready(prepared, monkeypatch, capsys):
    p = prepared; i = p.binding["dispatcher_identity"]
    values = {"execution_contract.json": p.contract,
        "start.json": {"pid": i["pid"], "identity": i, "contract_sha256": recovery.original.canonical(p.contract)},
        "dispatcher_launch.json": {"pid": i["pid"], "identity": i, "command": i["argv"]},
        "dispatcher_exit.json": terminal(p)}
    runtime = recovery.original.runtime
    monkeypatch.setattr(runtime, "read", lambda path: values[Path(path).name])
    monkeypatch.setattr(runtime, "sha_file", lambda path: "launch")
    monkeypatch.setattr(runtime, "proc_identity", lambda pid: {**i, "start_ticks": 123456})
    monkeypatch.setattr(Path, "exists", lambda self: False)
    with pytest.raises(ValueError, match="Current dispatcher identity differs"):
        exec(recovery.wait_program(p.contract, p.binding), {})
    assert capsys.readouterr().out == ""
