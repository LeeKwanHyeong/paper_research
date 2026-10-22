"""Local-only metadata v2 contract, bootstrap and one-call admission checks."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/shared_history_mlp_control_20261022_v1/execution_v2"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "execution_metadata_v2_" + name, REPORT / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


p, d = load("prepare_execution"), load("initial_execution")


def old_contract():
    return p.read(p.OLD_BUNDLE / "execution_contract.json")


def contract():
    return p.version_contract(
        old_contract(), p.sha(p.OLD_BUNDLE / "execution_contract.json")
    )


def test_metadata_fix_preserves_every_scientific_condition_and_deadline():
    old = old_contract()
    before = deepcopy(old)
    c = contract()
    assert old == before and "design_sha256" not in old
    assert c["design_sha256"] == c["science_spec_sha256"]
    assert c["execution_version"] == 2
    for field in (
        "source",
        "input_files",
        "legacy_files",
        "datasets",
        "jobs",
        "training",
        "limits",
        "common_condition",
        "cost",
        "science",
    ):
        assert c[field] == old[field]
    for host in d.HOSTS:
        assert (
            c["hosts"][host]["absolute_host_deadline_unix"]
            == old["hosts"][host]["absolute_host_deadline_unix"]
        )
        assert c["hosts"][host]["root"].endswith("_execution_v2_" + host)
        assert (
            c["hosts"][host]["environment"]["SOURCE_REVISION"]
            == old["hosts"][host]["environment"]["SOURCE_REVISION"]
        )
        assert (
            c["hosts"][host]["environment"]["PYTHONPATH"]
            == c["hosts"][host]["source_root"]
        )
    p.validate_version(c, old, p.sha(p.OLD_BUNDLE / "execution_contract.json"))


@pytest.mark.parametrize(
    "fault",
    [
        "design",
        "dataset",
        "source",
        "seed",
        "loss",
        "epochs",
        "deadline",
        "root",
        "environment",
        "test",
        "revision",
    ],
)
def test_metadata_contract_rejects_any_unapproved_scientific_or_deadline_change(fault):
    old, c = old_contract(), contract()
    if fault == "design":
        c["design_sha256"] = "0" * 64
    elif fault == "dataset":
        c["datasets"][0]["dataset_id"] = "changed"
    elif fault == "source":
        c["source"]["files_sha256"] = "0" * 64
    elif fault == "seed":
        c["jobs"][0]["seed"] = 43
    elif fault == "loss":
        c["common_condition"]["loss_mode"] = "changed"
    elif fault == "epochs":
        c["training"]["maximum_epochs"] = 301
    elif fault == "deadline":
        c["hosts"]["5080"]["absolute_host_deadline_unix"] += 1
    elif fault == "root":
        c["hosts"]["5080"]["root"] = old["hosts"]["5080"]["root"]
    elif fault == "environment":
        c["hosts"]["5080"]["environment"]["OMP_NUM_THREADS"] = "8"
    elif fault == "test":
        c["held_out_test_authorized"] = True
    else:
        c["source"]["base_git_revision"] = "changed"
    with pytest.raises(RuntimeError):
        p.validate_version(c, old, p.sha(p.OLD_BUNDLE / "execution_contract.json"))


def fixture(tmp_path, monkeypatch):
    c = contract()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "deployment.tar.gz").write_bytes(b"new archive bytes")
    integrity = {
        "archive_sha256": d.sha(bundle / "deployment.tar.gz"),
        "contract_sha256": "new-raw-contract-sha",
    }
    initial = p.original_helper()
    monkeypatch.setattr(d, "BUNDLE", bundle)
    monkeypatch.setattr(d, "STATE", tmp_path / "initial")
    monkeypatch.setattr(d, "verify_local", lambda *args: (c, integrity, initial))
    return c, integrity, initial


def forbidden(*args, **kwargs):
    raise AssertionError("No remote call allowed")


def invoke(action, host="5080"):
    return d.action(
        action, host, admission_sha="a", integrity_sha="i", qa_sha="q", self_sha="s"
    )


def test_generated_bootstrap_invalidates_cache_before_import_and_all_programs_compile(
    tmp_path, monkeypatch
):
    c, integrity, initial = fixture(tmp_path, monkeypatch)
    for host in d.HOSTS:
        program = d.deployment_program(c, host, integrity, initial)
        compile(program, "<deployment-v2>", "exec")
        assert program.index("importlib.invalidate_caches()") < program.index(
            "from paper.scripts import manage_"
        )
        assert "root.mkdir(parents=False,exist_ok=False)" in program
        assert "new-raw-contract-sha" in program
        for phase in ("qualification", "fit_validation"):
            program = d.launch_program(c, host, phase, integrity["contract_sha256"])
            compile(program, "<launch-v2>", "exec")
            assert integrity["contract_sha256"] in program
            assert "time.monotonic()+10" in program
        compile(
            initial.qualification_program(
                c, host, {"identity": {}, "supervisor_identity": {}}
            ),
            "<check-v2>",
            "exec",
        )


def test_expected_command_uses_new_raw_contract_not_original_global():
    c = contract()
    expected = d.expected_command(c, "5080", "fit_validation", "v2-raw")
    assert expected[expected.index("--contract-sha") + 1] == "v2-raw"
    assert (
        expected[expected.index("--contract") + 1]
        == c["hosts"]["5080"]["root"] + "/execution_contract.json"
    )
    with pytest.raises(RuntimeError, match="Test"):
        d.expected_command(c, "5080", "test", "v2-raw")


@pytest.mark.parametrize("marker", ["FAILURE.json", "ATTEMPT.json"])
def test_existing_failure_or_partial_attempt_blocks_other_host(
    tmp_path, monkeypatch, marker
):
    fixture(tmp_path, monkeypatch)
    d.write(d.STATE / "5080/old" / marker, {})
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(RuntimeError, match="[Ff]ailure|[Pp]artial"):
        invoke("deploy", "5090")


def test_failed_local_integrity_never_calls_remote(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)

    def fail(*args):
        raise RuntimeError("frozen source changed")

    monkeypatch.setattr(d, "verify_local", fail)
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    with pytest.raises(RuntimeError, match="source changed"):
        invoke("deploy")
    assert not list(d.STATE.rglob("ATTEMPT.json"))


def test_deployment_is_one_900_second_new_archive_transfer(tmp_path, monkeypatch):
    c, integrity, _ = fixture(tmp_path, monkeypatch)
    calls = []

    def transfer(command, folder, stem, **kwargs):
        calls.append((command, kwargs))
        return {}, json.dumps(
            {
                "status": "deployed",
                "host": "5080",
                "contract_sha256": d.canonical(c),
                "archive_sha256": integrity["archive_sha256"],
                "boot_id": c["hosts"]["5080"]["boot_id"],
            }
        ).encode()

    monkeypatch.setattr(d.boundary, "transport", transfer)
    assert invoke("deploy")["status"] == "deployed"
    assert len(calls) == 1 and calls[0][1] == {
        "timeout": 900,
        "input_bytes": b"new archive bytes",
    }
    with pytest.raises(FileExistsError):
        invoke("deploy")
    assert len(calls) == 1


def test_qualification_waiting_has_zero_ssh_during_cooldown(tmp_path, monkeypatch):
    fixture(tmp_path, monkeypatch)
    d.write(
        d.STATE / "5080/check-qualification/0/RESULT.json",
        {"status": "waiting", "checked_unix": d.time.time()},
    )
    monkeypatch.setattr(d.boundary, "transport", forbidden)
    assert invoke("check-qualification") == {
        "status": "waiting_next_check",
        "ssh_calls": 0,
    }


def test_first_fit_requires_both_native_gates_under_new_contract(tmp_path, monkeypatch):
    c, _, _ = fixture(tmp_path, monkeypatch)
    for host in d.HOSTS:
        d.write(
            d.STATE / host / "deploy/RESULT.json",
            {"status": "deployed", "contract_sha256": d.canonical(c)},
        )
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


@pytest.mark.parametrize(
    "field",
    [
        "actual_os_returncode",
        "source_unchanged_during_tests",
        "actual_cpu_training_e2e_passed",
    ],
)
def test_seal_requires_actual_source_bound_cpu_training_e2e(
    tmp_path, monkeypatch, field
):
    c = contract()
    reviewed = {p.SELF: p.sha(ROOT / p.SELF), p.INITIAL: p.sha(ROOT / p.INITIAL)}
    qa = {
        "status": "PASS",
        "independent_reviewer": True,
        "files_reviewed_sha256": reviewed,
    }
    receipt = {
        "actual_os_returncode": 0,
        "source_unchanged_during_tests": True,
        "actual_cpu_training_e2e_passed": True,
        "files_checked_sha256": reviewed,
    }
    receipt[field] = 1 if field == "actual_os_returncode" else False
    monkeypatch.setattr(
        p,
        "read",
        lambda path: qa if str(path).endswith("INDEPENDENT_QA.json") else receipt,
    )
    monkeypatch.setattr(p, "pinned", lambda *args: None)
    with pytest.raises(RuntimeError, match="Actual successful CPU training"):
        p.verify_qa_tests(c, "qa", tmp_path / "fabricated_unit_receipt.json")


def test_prepare_preserves_existing_partial_version_directory(tmp_path, monkeypatch):
    folder = tmp_path / "execution_v2"
    folder.mkdir()
    (folder / "untouched").write_text("original partial bytes")
    monkeypatch.setattr(p, "BUNDLE", folder)
    monkeypatch.setattr(p, "verify_admission", lambda *_: ({}, old_contract()))
    with pytest.raises(RuntimeError, match="already exists"):
        p.prepare("admission")
    assert (folder / "untouched").read_text() == "original partial bytes"


@pytest.mark.parametrize("script", ["prepare_execution.py", "initial_execution.py"])
def test_direct_file_cli_imports_project_from_fresh_python_outside_project(
    tmp_path, script
):
    import os
    import subprocess

    result = subprocess.run(
        [sys.executable, str(REPORT / script), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode()
    assert b"usage:" in result.stdout
    assert b"ModuleNotFoundError" not in result.stderr
    assert not list(tmp_path.iterdir())
