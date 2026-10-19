"""Local recovery preservation and fail-closed transport regression checks."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from paper.scripts import manage_titantpp_late_path18_5090_recovery as manager


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def original(tmp_path):
    origin = tmp_path / "origin_archive"
    root = origin / "originals"
    root.mkdir(parents=True)
    raw = b'{"schema":"science","held_out_test_authorized":false}'
    (root / "execution_contract.json").write_bytes(raw)
    (root / "unchanged.pt").write_bytes(b"immutable checkpoint bytes")
    files = {"execution_contract.json": digest(raw), "unchanged.pt": digest(b"immutable checkpoint bytes")}
    (origin / "originals.tar.gz").write_bytes(b"sealed archive bytes")
    manifest = {"files": files, "files_sha256": manager.canonical(files), "source_contract_raw_sha256": digest(raw),
                "archive_sha256": digest(b"sealed archive bytes")}
    (origin / "manifest.json").write_text(json.dumps(manifest))
    with patch.object(manager, "ORIGIN", origin):
        yield origin, root, manifest


def test_original_inventory_verifies_without_modification(original):
    origin, root, before = original
    old = {p: p.read_bytes() for p in origin.rglob("*") if p.is_file()}
    manifest, science = manager.verify_origin()
    assert manifest == before and science["held_out_test_authorized"] is False
    assert old == {p: p.read_bytes() for p in old}


def test_changed_checkpoint_is_rejected(original):
    _, root, _ = original
    (root / "unchanged.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Original copy changed"):
        manager.verify_origin()


def test_changed_archive_is_rejected(original):
    origin, _, _ = original
    (origin / "originals.tar.gz").write_bytes(b"different archive")
    with pytest.raises(ValueError, match="archive changed"):
        manager.verify_origin()


def test_empty_origin_inventory_is_rejected(original):
    origin, _, manifest = original
    manifest["files"] = {}
    (origin / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        manager.verify_origin()


def test_origin_path_escape_is_rejected(original):
    origin, _, manifest = original
    manifest["files"] = {"../outside.pt": digest(b"outside")}
    (origin / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Invalid original path"):
        manager.verify_origin()


def test_transport_failure_keeps_exact_raw_evidence(tmp_path):
    response = subprocess.CompletedProcess([], 255, b"partial\x00stdout", b"Broken pipe\n")
    with patch.object(manager, "BUNDLE", tmp_path):
        with pytest.raises(ValueError, match="failed; original transport preserved"):
            manager.transport("wait", response)
    prefix = next((tmp_path / "transport").iterdir())
    assert (prefix / "stdout").read_bytes() == b"partial\x00stdout"
    assert (prefix / "stderr").read_bytes() == b"Broken pipe\n"
    receipt = json.loads((prefix / "receipt.json").read_text())
    assert receipt["actual_transport_exit"] == 255
    assert receipt["stdout_sha256"] == digest(b"partial\x00stdout")


def test_timeout_preserves_partial_transport_and_unknown_exit(tmp_path):
    failure = subprocess.TimeoutExpired(["ssh"], 1, output=b"partial stdout", stderr=b"partial error")
    with patch.object(manager.subprocess, "run", side_effect=failure):
        response = manager.ssh("read-only status", timeout=1)
    assert response.returncode is None and response.timed_out is True
    with patch.object(manager, "BUNDLE", tmp_path):
        with pytest.raises(ValueError):
            manager.transport("wait", response)
    prefix = next((tmp_path / "transport").iterdir())
    receipt = json.loads((prefix / "receipt.json").read_text())
    assert receipt["actual_transport_exit"] is None and receipt["transport_timeout"] is True
    assert (prefix / "stdout").read_bytes() == b"partial stdout"
    assert (prefix / "stderr").read_bytes() == b"partial error"


def test_intent_is_exclusive_and_blocks_second_deployment_before_ssh(tmp_path):
    report = tmp_path / "report"
    contract = {"root": "/remote/recovery"}
    manager.write(report / "deployment.intent.json", {"existing_attempt": True})
    before = (report / "deployment.intent.json").read_bytes()
    with patch.object(manager, "REPORT", report), patch.object(manager, "verify_local", return_value=contract), patch.object(manager, "ssh") as remote:
        with pytest.raises(FileExistsError):
            manager.deploy()
        remote.assert_not_called()
    assert (report / "deployment.intent.json").read_bytes() == before


def test_invocation_keeps_science_source_revision_and_new_output_environment():
    contract = {"source_root": "/new/source", "python": "/runtime/python", "root": "/new/root",
                "environment": {"SOURCE_REVISION": "old-scientific-revision", "PYTHONPATH": "/new/source"}}
    result = manager.invocation(contract, "status")
    assert "SOURCE_REVISION=old-scientific-revision" in result
    assert "--contract /new/root/recovery_contract.json --mode status" in result


def test_preserved_contract_json_cannot_be_overwritten(tmp_path):
    target = tmp_path / "recovery_contract.json"
    manager.write(target, {"first": 1})
    raw = target.read_bytes()
    with pytest.raises(FileExistsError):
        manager.write(target, {"second": 2})
    assert target.read_bytes() == raw
