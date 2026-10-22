"""Local-only tests of the separately admitted initial setup recovery."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROJECT = Path(__file__).resolve().parents[3]
PATH = (
    PROJECT
    / "reports/shared_history_mlp_control_20261022_v1/recovery_v1/recover_initial_deployment.py"
)
SPEC = importlib.util.spec_from_file_location("history_mlp_recovery_under_test", PATH)
d = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = d
SPEC.loader.exec_module(d)


def load_initial():
    spec = importlib.util.spec_from_file_location(
        "frozen_initial_for_recovery_tests",
        PROJECT / "reports/shared_history_mlp_control_20261022_v1/initial_execution.py",
    )
    initial = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = initial
    spec.loader.exec_module(initial)
    return initial


def fixture(tmp_path, monkeypatch):
    state, bundle = tmp_path / "recovery", tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "execution_contract.json").write_text("{}")
    (bundle / "deployment.tar.gz").write_bytes(b"fixed archive")
    c = {
        "hosts": {
            host: {
                "root": "/isolated/" + host,
                "source_root": "/isolated/" + host + "/source",
                "python": "/fixed/python",
                "tmux_binary": "/fixed/tmux",
                "tmux": "fixed_" + host,
                "environment": {"PYTHONDONTWRITEBYTECODE": "1"},
                "boot_id": "boot_" + host,
            }
            for host in ("5080", "5090")
        },
        "source": {"files_sha256": "fixed_source"},
    }
    integrity = {
        "archive_sha256": d.sha(bundle / "deployment.tar.gz"),
        "contract_sha256": d.sha(bundle / "execution_contract.json"),
    }
    initial = load_initial()
    monkeypatch.setattr(initial, "BUNDLE", bundle)
    monkeypatch.setattr(initial.controller, "BUNDLE", bundle)
    monkeypatch.setattr(d, "BUNDLE", bundle)
    monkeypatch.setattr(d, "STATE", state)
    monkeypatch.setattr(
        d, "verify_admission", lambda *args: ({}, c, integrity, initial)
    )
    return c, integrity, initial


def invoke(name, host="5080"):
    return d.action(
        name,
        host,
        admission_path=Path("admission"),
        admission_sha="a",
        self_sha="s",
        recovery_qa_sha="q",
    )


def forbidden(*args, **kwargs):
    raise AssertionError("No SSH or scientific launch is allowed")


def admissions(c, integrity):
    d.write(
        d.STATE / "5080/verify-existing/RESULT.json",
        {
            "status": "existing_deployment_verified",
            "host": "5080",
            "contract_sha256": d.canonical(c),
            "archive_sha256": integrity["archive_sha256"],
        },
    )
    d.write(
        d.STATE / "5090/deploy/RESULT.json",
        {
            "status": "deployed",
            "host": "5090",
            "contract_sha256": d.canonical(c),
            "archive_sha256": integrity["archive_sha256"],
        },
    )


def test_recovery_generated_programs_compile_and_existing_admission_stays_readonly(
    tmp_path, monkeypatch
):
    c, integrity, initial = fixture(tmp_path, monkeypatch)
    program = d.remote_program(c, integrity, initial)
    compile(program, "<readonly-existing-5080>", "exec")
    assert "importlib.invalidate_caches()" in program
    assert (
        "from paper.scripts.quantity_comparison_runtime import configure_runtime,runtime_identity"
        in program
    )
    assert "run_shared_history_mlp_control" not in program
    assert (
        ".mkdir(" not in program
        and ".open('w" not in program
        and ".open('x" not in program
    )
    assert (
        ".supervise(" not in program
        and ".dispatch(" not in program
        and ".pack(" not in program
    )
    assert "root.glob('qualification.*')" in program
    assert "root.glob('fit_validation.*')" in program
    assert "root.glob('cost.*')" in program
    assert "p.is_symlink()" in program and "expected" in program
    deployment = d.deployment_5090_program(c, integrity, initial)
    compile(deployment, "<first-5090-deployment>", "exec")
    assert deployment.count("importlib.invalidate_caches()") == 1
    assert deployment.index("importlib.invalidate_caches()") < deployment.index(
        "from paper.scripts import manage_"
    )
    for host in ("5080", "5090"):
        for phase in ("qualification", "fit_validation"):
            compile(initial.launch_program(c, host, phase), "<launch>", "exec")
        compile(
            initial.qualification_program(
                c, host, {"identity": {}, "supervisor_identity": {}}
            ),
            "<qualification-check>",
            "exec",
        )


@pytest.mark.parametrize("name,host", [("deploy", "5080"), ("verify-existing", "5090")])
def test_existing_5080_can_never_be_reuploaded_or_5090_adopted(
    tmp_path, monkeypatch, name, host
):
    fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(RuntimeError):
        invoke(name, host)
    assert (d.STATE / host / name / "FAILURE.json").is_file()


def test_first_5090_transfer_requires_successful_existing_5080_admission(
    tmp_path, monkeypatch
):
    fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(FileNotFoundError):
        invoke("deploy", "5090")


def test_existing_5080_is_checked_once_with_45_second_bound_and_no_payload(
    tmp_path, monkeypatch
):
    c, integrity, _ = fixture(tmp_path, monkeypatch)
    calls = []

    def transport(command, folder, stem, **kwargs):
        calls.append((command, kwargs))
        value = {
            "status": "existing_deployment_verified",
            "host": "5080",
            "contract_sha256": d.canonical(c),
            "read_only_verification": True,
            "archive_sha256": integrity["archive_sha256"],
            "source_files_sha256": c["source"]["files_sha256"],
            "boot_id": "boot_5080",
            "scientific_processes_launched": 0,
            "files_reuploaded": 0,
            "files_removed_or_overwritten": 0,
        }
        return {}, json.dumps(value).encode()

    monkeypatch.setattr(d.boundary, "transport", transport)
    assert invoke("verify-existing")["status"] == "existing_deployment_verified"
    assert len(calls) == 1 and calls[0][1] == {"timeout": 45, "input_bytes": None}
    with pytest.raises(FileExistsError):
        invoke("verify-existing")
    assert len(calls) == 1


def test_5090_is_first_transferred_once_with_fixed_archive_and_cache_fix(
    tmp_path, monkeypatch
):
    c, integrity, _ = fixture(tmp_path, monkeypatch)
    d.write(
        d.STATE / "5080/verify-existing/RESULT.json",
        {"status": "existing_deployment_verified"},
    )
    calls = []

    def transport(command, folder, stem, **kwargs):
        calls.append((command, kwargs))
        return {}, json.dumps(
            {
                "status": "deployed",
                "host": "5090",
                "contract_sha256": d.canonical(c),
                "archive_sha256": integrity["archive_sha256"],
                "boot_id": "boot_5090",
            }
        ).encode()

    monkeypatch.setattr(d.boundary, "transport", transport)
    assert invoke("deploy", "5090")["status"] == "deployed"
    assert len(calls) == 1 and calls[0][1] == {
        "timeout": 900,
        "input_bytes": b"fixed archive",
    }
    assert "invalidate_caches" in calls[0][0][-1]


@pytest.mark.parametrize(
    "name", ["launch-qualification", "launch-fit", "check-qualification"]
)
def test_native_actions_need_both_deployment_admissions(tmp_path, monkeypatch, name):
    fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(FileNotFoundError):
        invoke(name)


def test_initial_fit_requires_both_hosts_raw_native_qualification(
    tmp_path, monkeypatch
):
    c, integrity, _ = fixture(tmp_path, monkeypatch)
    admissions(c, integrity)
    calls = []

    def gate(c, host):
        calls.append(host)
        if host == "5090":
            raise RuntimeError("second native proof missing")

    monkeypatch.setattr(d, "verify_pass", gate)
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(RuntimeError, match="second native proof"):
        invoke("launch-fit")
    assert calls == ["5080", "5090"]


@pytest.mark.parametrize("marker", ["FAILURE.json", "ATTEMPT.json"])
def test_preserved_recovery_failure_or_partial_blocks_any_other_host(
    tmp_path, monkeypatch, marker
):
    fixture(tmp_path, monkeypatch)
    d.write(d.STATE / "5080/old" / marker, {})
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(RuntimeError, match="[Ff]ailure|[Pp]artial"):
        invoke("deploy", "5090")


def test_failed_admission_does_not_reach_any_ssh(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)

    def gate(*args):
        raise RuntimeError("recovery QA changed")

    monkeypatch.setattr(d, "verify_admission", gate)
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(RuntimeError, match="recovery QA"):
        invoke("verify-existing")
    assert not list(d.STATE.rglob("ATTEMPT.json"))


def test_native_waiting_is_zero_ssh_until_cooldown(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    d.write(
        d.STATE / "5080/check-qualification/000/RESULT.json",
        {"status": "waiting", "checked_unix": d.time.time()},
    )
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    assert invoke("check-qualification") == {
        "status": "waiting_next_check",
        "ssh_calls": 0,
    }


def test_frozen_initial_state_is_not_changed_by_recovery_generation(
    tmp_path, monkeypatch
):
    c, integrity, initial = fixture(tmp_path, monkeypatch)
    before = initial.STATE
    d.remote_program(c, integrity, initial)
    d.deployment_5090_program(c, integrity, initial)
    assert initial.STATE == before and d.STATE != before


def test_qualification_raw_inventory_tampering_is_detected(tmp_path, monkeypatch):
    c, _, _ = fixture(tmp_path, monkeypatch)
    snapshot = d.STATE / "5080/snapshot"
    d.write(snapshot / "proof.json", {"preserved": True})
    d.write(
        d.STATE / "5080/QUALIFICATION_PASS.json",
        {
            "status": "PASS",
            "host": "5080",
            "contract_sha256": d.canonical(c),
            "snapshot": "snapshot",
            "files": {"proof.json": d.sha(snapshot / "proof.json")},
        },
    )
    (snapshot / "proof.json").write_text("tampered")
    monkeypatch.setattr(d.manager, "verify_phase_receipt", forbidden)
    with pytest.raises(RuntimeError, match="raw evidence"):
        d.verify_pass(c, "5080")


def test_archive_bootstrap_marker_must_match_frozen_generator(tmp_path, monkeypatch):
    c, integrity, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="bootstrap marker"):
        d.deployment_5090_program(
            c, integrity, SimpleNamespace(deployment_program=lambda *args: "changed")
        )
