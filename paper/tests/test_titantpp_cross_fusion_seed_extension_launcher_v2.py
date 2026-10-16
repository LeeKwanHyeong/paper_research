"""Synthetic receipt-based first-launch checks; no host connection or training."""
from copy import deepcopy
import json
import subprocess

import pytest

from paper.scripts import launch_titantpp_cross_fusion_seed_extension_v2 as launcher
from paper.scripts import titantpp_cross_fusion_seed_extension_runtime as runtime


@pytest.fixture
def admission(tmp_path, monkeypatch):
    m = launcher.m
    monkeypatch.setattr(m, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(m, "REPORT", tmp_path / "report")
    c = {"parent": {"qualification_sha256": {"5090": "parent-qualification"}},
         "hosts": {"5090": {"root": "/isolated/new", "source_root": "/isolated/new/source",
                    "python": "/runtime/python", "environment": {},
                    "tmux_binary": "/runtime/tmux", "tmux": "isolated-session"}}}
    monkeypatch.setattr(m, "verify_local", lambda: c)
    monkeypatch.setattr(m, "ssh", lambda *a, **k: pytest.fail("Unexpected remote call"))
    m.write(m.REPORT / "PREPARED.json", {"contract_sha256": "raw-contract"})
    m.write(m.BUNDLE / "attempts/5090.json", {"host": "5090", "contract_sha256": m.canonical(c)})
    for phase in ("install", "admit"):
        m.save_transport("5090", phase, subprocess.CompletedProcess([], 0, b"", b""))
    q = {"status": "passed", "host": "5090", "contract_sha256": m.canonical(c),
         "inherited_native_qualification": "parent-qualification"}
    snapshot = {"qualification": q, "existing": [], "tmux_exit": 1, "processes": [],
                "contract_raw_sha256": "raw-contract"}
    return c, snapshot


def test_actual_runtime_main_admit_returns_zero_without_stdout(tmp_path, monkeypatch, capsys):
    contract = tmp_path / "execution_contract.json"
    contract.write_text("{}")
    monkeypatch.setattr(runtime, "validate_contract", lambda c, h: None)
    def admit(c, host):
        runtime.write(tmp_path / "qualification.json", {"status": "passed", "host": host}, exclusive=True)
        return {"status": "passed"}
    monkeypatch.setattr(runtime, "admit", admit)
    assert runtime.main(["--contract", str(contract), "--host", "5090", "--mode", "admit"]) == 0
    assert capsys.readouterr().out == ""
    assert json.loads((tmp_path / "qualification.json").read_text())["status"] == "passed"


def test_empty_admit_stdout_is_valid_with_verified_receipts(admission):
    c, snapshot = admission
    launcher.check_install_admit("5090")
    launcher.check_snapshot(snapshot, c, "5090")


def test_snapshot_identity_or_prior_start_or_process_or_tmux_errors_rejected(admission):
    c, good = admission
    cases = []
    for key, value in (("host", "5080"), ("contract_sha256", "wrong"),
                       ("inherited_native_qualification", "wrong"), ("status", "failed")):
        bad = deepcopy(good)
        bad["qualification"][key] = value
        cases.append(bad)
    for key, value in (("existing", ["start.json"]), ("existing", ["run"]),
                       ("processes", [123]), ("tmux_exit", 0), ("tmux_exit", 2),
                       ("contract_raw_sha256", "wrong")):
        bad = deepcopy(good)
        bad[key] = value
        cases.append(bad)
    for snapshot in cases:
        with pytest.raises(ValueError, match="absence check"):
            launcher.check_snapshot(snapshot, c, "5090")


def test_changed_transport_evidence_rejected(admission):
    path = next((launcher.m.BUNDLE / "transport/5090").glob("*_admit.stdout"))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="evidence changed"):
        launcher.check_install_admit("5090")


def test_duplicate_launch_intent_rejected_before_remote(admission):
    launcher.m.write(launcher.m.BUNDLE / "launch_v2/5090.json", {"preserve": True})
    with pytest.raises(FileExistsError, match="already attempted"):
        launcher.launch("5090")


def test_verified_installed_host_launches_without_reinstall_or_readmit(admission, monkeypatch):
    _, snapshot = admission
    calls = []
    def fake_ssh(host, command, **kwargs):
        calls.append((host, command))
        assert "mkdir " not in command
        assert "--mode admit" not in command
        if len(calls) == 1:
            assert "qualification.json" in command
            return subprocess.CompletedProcess([], 0, json.dumps(snapshot).encode(), b"")
        assert "new-session" in command
        assert "dispatch" in command
        assert (launcher.m.BUNDLE / "launch_v2/5090.json").exists()
        return subprocess.CompletedProcess([], 0, b"", b"")
    monkeypatch.setattr(launcher.m, "ssh", fake_ssh)
    result = launcher.launch("5090")
    assert len(calls) == 2
    assert result["status"] == "queue_dispatched_not_yet_verified"
    assert result["qualification"] == snapshot["qualification"]
    assert launcher.m.read(launcher.m.BUNDLE / "launch_v2/5090.json")["qualification_reexecuted"] is False


def test_failed_or_additional_transport_blocks_first_launch(admission):
    m = launcher.m
    path = next((m.BUNDLE / "transport/5090").glob("*_admit.json"))
    value = m.read(path)
    value["actual_transport_exit"] = 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Failed transport"):
        launcher.check_install_admit("5090")
    value["actual_transport_exit"] = 0
    path.write_text(json.dumps(value))
    m.save_transport("5090", "launch", subprocess.CompletedProcess([], 0, b"", b""))
    with pytest.raises(ValueError, match="exactly one"):
        launcher.check_install_admit("5090")
