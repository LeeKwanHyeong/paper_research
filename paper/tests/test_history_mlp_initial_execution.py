"""Local-only deployment inventory and one-call initial-control fault tests."""

import importlib.util
import inspect
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

PATH = (
    Path(__file__).resolve().parents[2]
    / "reports/shared_history_mlp_control_20261022_v1/initial_execution.py"
)
SPEC = importlib.util.spec_from_file_location("history_mlp_initial_execution", PATH)
d = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d
SPEC.loader.exec_module(d)


def archive_fixture(extra=(), *, wrong_member=False):
    data = b"preserved source bytes"
    c = {
        "source": {
            "files": {"paper/source.py": __import__("hashlib").sha256(data).hexdigest()}
        },
        "input_files": {},
        "legacy_files": {},
    }
    raw = json.dumps(c).encode()
    entries = [
        ("execution_contract.json", raw, tarfile.REGTYPE),
        (
            "source/paper/source.py",
            b"changed" if wrong_member else data,
            tarfile.REGTYPE,
        ),
        ("source/sample_data", b"", tarfile.DIRTYPE),
        *extra,
    ]
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for name, value, kind in entries:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = len(value) if kind == tarfile.REGTYPE else 0
            if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "source/paper/source.py"
            tar.addfile(member, io.BytesIO(value) if member.isfile() else None)
    return (
        out.getvalue(),
        __import__("hashlib").sha256(raw).hexdigest(),
        d.canonical(c),
        c,
    )


def test_deployment_inventory_requires_exact_original_bytes():
    raw, digest, canonical, c = archive_fixture()
    assert d.check_archive(raw, digest, canonical) == c
    bad, digest, canonical, _ = archive_fixture(wrong_member=True)
    with pytest.raises(RuntimeError, match="member SHA"):
        d.check_archive(bad, digest, canonical)


def test_injected_archive_checker_executes_without_local_module_globals():
    raw, digest, canonical, c = archive_fixture()
    namespace = {}
    exec(inspect.getsource(d.check_archive), namespace)
    assert namespace["check_archive"](raw, digest, canonical) == c


@pytest.mark.parametrize(
    "name,kind",
    [
        ("../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("source/a/../escape", tarfile.REGTYPE),
        ("source/symlink", tarfile.SYMTYPE),
        ("source/hardlink", tarfile.LNKTYPE),
        ("source/device", tarfile.CHRTYPE),
        ("source/extra", tarfile.REGTYPE),
        ("source/paper/source.py", tarfile.REGTYPE),
    ],
)
def test_unsafe_unbound_or_duplicate_tar_members_are_rejected(name, kind):
    raw, digest, canonical, _ = archive_fixture([(name, b"x", kind)])
    with pytest.raises(RuntimeError):
        d.check_archive(raw, digest, canonical)


@pytest.mark.parametrize("pin", ["raw", "canonical"])
def test_contract_pins_are_checked_before_extraction(pin):
    raw, digest, canonical, _ = archive_fixture()
    with pytest.raises(RuntimeError, match="SHA"):
        d.check_archive(
            raw,
            "0" * 64 if pin == "raw" else digest,
            "0" * 64 if pin == "canonical" else canonical,
        )


def fixture(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "execution_contract.json").write_text("{}")
    (bundle / "deployment.tar.gz").write_bytes(b"archive")
    c = {
        "hosts": {
            host: {
                "root": "/isolated/" + host,
                "source_root": "/isolated/" + host + "/source",
                "python": "/fixed/python",
                "tmux_binary": "/fixed/tmux",
                "tmux": "isolated_" + host,
                "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                "boot_id": "boot-" + host,
            }
            for host in d.HOSTS
        }
    }
    integrity = {
        "archive_sha256": d.sha(bundle / "deployment.tar.gz"),
        "contract_sha256": d.sha(bundle / "execution_contract.json"),
    }
    monkeypatch.setattr(d, "BUNDLE", bundle)
    monkeypatch.setattr(d.controller, "BUNDLE", bundle)
    monkeypatch.setattr(d, "STATE", tmp_path / "state")
    monkeypatch.setattr(d, "verify_local", lambda *args: (c, integrity))
    return c, integrity


def invoke(name, host="5080"):
    return d.action(
        name, host, integrity_sha="i", authorization_sha="a", qa_sha="q", driver_sha="d"
    )


def forbid_transport(*args, **kwargs):
    raise AssertionError("No SSH may be called")


def test_all_generated_programs_compile_and_keep_initial_scope(tmp_path, monkeypatch):
    c, integrity = fixture(tmp_path, monkeypatch)
    for host in d.HOSTS:
        programs = [d.deployment_program(c, host, integrity)]
        for phase in ("qualification", "fit_validation"):
            programs.append(d.launch_program(c, host, phase))
        programs.append(
            d.qualification_program(
                c, host, {"identity": {}, "supervisor_identity": {}}
            )
        )
        for program in programs:
            compile(program, "<remote-program>", "exec")
        assert "m.verify_phase_receipt" in programs[-1]
        assert "m.pack(" not in programs[-1]
        assert "sys.stdin.buffer.read" in programs[0]
        assert "root.mkdir(parents=False,exist_ok=False)" in programs[0]
        assert "time.monotonic()+10" in programs[1]


def test_local_gate_failure_never_calls_ssh(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(d, "transport", forbid_transport)

    def fail(*args):
        raise RuntimeError("source changed")

    monkeypatch.setattr(d, "verify_local", fail)
    with pytest.raises(RuntimeError, match="source changed"):
        invoke("deploy")
    assert not list(d.STATE.rglob("ATTEMPT.json"))


@pytest.mark.parametrize("marker", ["FAILURE.json", "ATTEMPT.json"])
def test_prior_failure_or_partial_attempt_blocks_all_new_calls(
    tmp_path, monkeypatch, marker
):
    fixture(tmp_path, monkeypatch)
    d.write(d.STATE / "5090/previous" / marker, {})
    monkeypatch.setattr(d, "transport", forbid_transport)
    with pytest.raises(RuntimeError, match="[Pp]reserv|[Ii]ncomplete"):
        invoke("deploy")


def test_waiting_cooldown_is_zero_ssh(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    d.write(
        d.STATE / "5080/check-qualification/000/RESULT.json",
        {"status": "waiting", "checked_unix": d.time.time()},
    )
    monkeypatch.setattr(d, "transport", forbid_transport)
    assert invoke("check-qualification") == {
        "status": "waiting_next_check",
        "ssh_calls": 0,
    }


def test_fit_requires_both_hosts_actual_qualification_gate(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    d.write(d.STATE / "5080/deploy/RESULT.json", {"status": "deployed"})
    calls = []

    def gate(c, host):
        calls.append(host)
        if host == "5090":
            raise RuntimeError("second native gate missing")

    monkeypatch.setattr(d, "verify_pass", gate)
    monkeypatch.setattr(d, "transport", forbid_transport)
    with pytest.raises(RuntimeError, match="second native gate"):
        invoke("launch-fit")
    assert calls == ["5080", "5090"]
    assert (d.STATE / "5080/launch-fit/FAILURE.json").is_file()


def test_deployment_is_one_bounded_binary_stdin_transfer(tmp_path, monkeypatch):
    c, integrity = fixture(tmp_path, monkeypatch)
    calls = []

    def transfer(command, folder, stem, **kwargs):
        calls.append((command, kwargs))
        return {}, json.dumps(
            {
                "status": "deployed",
                "host": "5080",
                "contract_sha256": d.canonical(c),
                "archive_sha256": integrity["archive_sha256"],
                "boot_id": "boot-5080",
            }
        ).encode()

    monkeypatch.setattr(d, "transport", transfer)
    assert invoke("deploy")["status"] == "deployed"
    assert len(calls) == 1 and calls[0][1] == {
        "timeout": 900,
        "input_bytes": b"archive",
    }
    with pytest.raises(FileExistsError):
        invoke("deploy")
    assert len(calls) == 1


def test_transport_failure_is_preserved_and_never_retried(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("transport failed; bytes preserved")

    monkeypatch.setattr(d, "transport", fail)
    with pytest.raises(RuntimeError, match="transport failed"):
        invoke("deploy")
    assert (d.STATE / "5080/deploy/FAILURE.json").is_file()
    with pytest.raises(RuntimeError, match="Preserved failure"):
        invoke("deploy", "5090")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "fault", ["authorization", "preservation", "implementation_gate"]
)
def test_sealed_gate_fields_are_bound_before_any_remote_program(
    tmp_path, monkeypatch, fault
):
    root = tmp_path / "project"
    report, bundle = root / "report", root / "bundle"
    report.mkdir(parents=True)
    bundle.mkdir()
    monkeypatch.setattr(d, "ROOT", root)
    monkeypatch.setattr(d, "REPORT", report)
    monkeypatch.setattr(d, "BUNDLE", bundle)
    monkeypatch.setattr(d.manager, "validate", lambda c: None)
    monkeypatch.setattr(d, "transport", forbid_transport)
    driver = root / d.SELF
    driver.parent.mkdir(parents=True)
    driver.write_text("fixed driver")
    for name in ("AUTHORIZATION.json", "INDEPENDENT_QA.json", "PRESERVATION.json"):
        (report / name).write_text("{}")
    (bundle / "integrity.json").write_text("{}")
    c = {
        "authorization_sha256": d.sha(report / "AUTHORIZATION.json"),
        "preservation_sha256": d.sha(report / "PRESERVATION.json"),
        "implementation_qualification": {"independent_qa_sha256": "invalid"},
    }
    if fault == "authorization":
        c["authorization_sha256"] = "wrong"
    if fault == "preservation":
        c["preservation_sha256"] = "wrong"
    (bundle / "execution_contract.json").write_text(json.dumps(c))
    with pytest.raises(
        RuntimeError,
        match={
            "authorization": "authorization binding",
            "preservation": "Pinned local file",
            "implementation_gate": "implementation gate binding",
        }[fault],
    ):
        d.verify_local(
            d.sha(bundle / "integrity.json"),
            d.sha(report / "AUTHORIZATION.json"),
            d.sha(report / "INDEPENDENT_QA.json"),
            d.sha(driver),
        )
