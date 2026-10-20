import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from unittest.mock import patch

import pytest

from paper.scripts import manage_titantpp_shared_efficiency_5090 as ops


def test_supervisor_records_real_failed_child_exit_without_retry(tmp_path):
    (tmp_path / "source").mkdir()
    (tmp_path / "benchmark.py").write_text("import sys;sys.exit(7)\n")
    (tmp_path / "supervise.py").write_text(ops.SUPERVISOR)
    result = subprocess.run([sys.executable, str(tmp_path / "supervise.py")], capture_output=True, timeout=10)
    evidence = json.loads((tmp_path / "controller_exit.json").read_text())
    assert result.returncode == evidence["actual_os_returncode"] == evidence["returncode"] == 7
    assert evidence["actual_controller_exit_observed"] is True
    assert evidence["pid_absent_after_wait"] is True
    assert evidence["supervisor_own_os_exit_observed"] is False


def test_transport_failure_keeps_exact_streams_and_does_not_retry(tmp_path):
    r = subprocess.CompletedProcess(["ssh"], 255, b"original stdout", b"connection error\r\n")
    with patch.object(ops, "BUNDLE", tmp_path), patch.object(ops.subprocess, "run", return_value=r) as run:
        with pytest.raises(ValueError, match="no automatic retry"):
            ops.transport("one", ["ssh"])
    assert run.call_count == 1
    assert (tmp_path / "transport/one.stdout").read_bytes() == r.stdout
    assert (tmp_path / "transport/one.stderr").read_bytes() == r.stderr
    assert json.loads((tmp_path / "transport/one.json").read_text())["actual_transport_exit"] == 255


def test_exclusive_receipts_cannot_replace_old_evidence(tmp_path):
    p = tmp_path / "receipt.json"
    ops.write(p, {"old": True})
    original = p.read_bytes()
    with pytest.raises(FileExistsError): ops.write(p, {"old": False})
    assert p.read_bytes() == original


def test_verify_rejects_empty_deployment_inventory(tmp_path):
    c = {"scope": "test"}; digest = ops.canonical(c)
    ops.write(tmp_path / "contract.json", c)
    ops.write(tmp_path / "approval.json", {"contract_sha256": digest})
    ops.write(tmp_path / "deployment_manifest.json", {"contract_sha256": digest, "files": {}})
    with patch.object(ops, "BUNDLE", tmp_path), pytest.raises(ValueError, match="Empty inventory"):
        ops.verify()


def test_verify_rejects_changed_operation(tmp_path):
    c = {"scope": "test"}; digest = ops.canonical(c)
    ops.write(tmp_path / "contract.json", c)
    ops.write(tmp_path / "approval.json", {"contract_sha256": digest})
    (tmp_path / "benchmark.py").write_text("changed")
    ops.write(tmp_path / "deployment_manifest.json", {"contract_sha256": digest,
        "files": {"benchmark.py": hashlib.sha256(b"original").hexdigest()}})
    with patch.object(ops, "BUNDLE", tmp_path), pytest.raises(ValueError, match="Local deployment changed"):
        ops.verify()


def test_collector_rejects_traversal_and_preserves_original_archive(tmp_path):
    bundle = tmp_path / "bundle"; bundle.mkdir(); report = tmp_path / "report"
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("../../escaped"); info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    data = stream.getvalue()
    with patch.object(ops, "BUNDLE", bundle), patch.object(ops, "REPORT", report), \
         patch.object(ops, "verify", return_value={"host": {"python": "python"}}), \
         patch.object(ops, "transport", return_value=data), pytest.raises(tarfile.OutsideDestinationError):
        ops.finish()
    assert (bundle / "original.tar.gz").read_bytes() == data
    assert not (tmp_path / "escaped").exists()
