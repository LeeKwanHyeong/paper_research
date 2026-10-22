"""Independent synthetic checks; never executes paper imports or remote bodies."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "independent_recovery_candidate", HERE / "recover_initial_deployment.py"
)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)


def staged_fixture(tmp_path):
    root = tmp_path / "deployed"
    source = b"fixture data, no real dataset\n"
    c = {
        "hosts": {"5080": {"root": str(root)}},
        "source": {"files": {"fixture.txt": hashlib.sha256(source).hexdigest()}},
        "input_files": {},
        "legacy_files": {},
    }
    raw = json.dumps(c).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data, kind in (
            ("execution_contract.json", raw, tarfile.REGTYPE),
            ("source/fixture.txt", source, tarfile.REGTYPE),
            ("source/sample_data", b"", tarfile.DIRTYPE),
        ):
            item = tarfile.TarInfo(name)
            item.type, item.size = kind, len(data)
            tar.addfile(item, io.BytesIO(data) if item.isfile() else None)
    (root / "source/sample_data").mkdir(parents=True)
    (root / "source/fixture.txt").write_bytes(source)
    (root / "execution_contract.json").write_bytes(raw)
    (root / "deployment.tar.gz").write_bytes(buf.getvalue())
    integrity = {
        "archive_sha256": d.sha(root / "deployment.tar.gz"),
        "contract_sha256": d.sha(root / "execution_contract.json"),
    }
    return root, c, integrity


@pytest.mark.parametrize("fault", (None, "member", "extra", "symlink", "archive"))
def test_generated_readonly_prefix_executes_checks_and_blocks_writes(tmp_path, fault):
    root, c, integrity = staged_fixture(tmp_path)
    if fault == "member":
        (root / "source/fixture.txt").write_bytes(b"tampered")
    elif fault == "extra":
        (root / "unexpected.txt").write_bytes(b"unexpected")
    elif fault == "symlink":
        (root / "link").symlink_to(root / "source/fixture.txt")
    elif fault == "archive":
        with (root / "deployment.tar.gz").open("ab") as stream:
            stream.write(b"changed")
    initial = d.initial_module()
    program = d.remote_program(c, integrity, initial)
    marker = "from paper.scripts import manage_shared_history_mlp_control as m\n"
    assert program.count(marker) == 1
    prefix = program.split(marker)[0]
    assert "cuda" not in prefix and "subprocess" not in prefix
    # Stop before every actual paper import, runtime call or GPU operation.
    prefix += "\ntry:\n (root/'forbidden-write').write_bytes(b'blocked')\nexcept RuntimeError:\n print('verified readonly prefix and write rejection')\nelse:\n raise AssertionError('write allowed')\n"
    before = {
        str(p.relative_to(root)): d.sha(p)
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    result = subprocess.run(
        [sys.executable, "-c", prefix],
        capture_output=True,
        timeout=20,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    if fault is None:
        assert result.returncode == 0, result.stderr.decode()
        assert result.stdout == b"verified readonly prefix and write rejection\n"
    else:
        assert result.returncode != 0
        assert b"RuntimeError" in result.stderr
    after = {
        str(p.relative_to(root)): d.sha(p)
        for p in root.rglob("*")
        if p.is_file() and not p.is_symlink()
    }
    assert before == after and not (root / "forbidden-write").exists()


def test_recovery_changes_only_one_process_local_bootstrap_line(tmp_path, monkeypatch):
    root, c, integrity = staged_fixture(tmp_path)
    c["hosts"]["5090"] = {"root": str(root)}
    initial = d.initial_module()
    monkeypatch.setattr(initial, "BUNDLE", root)
    original = initial.deployment_program(c, "5090", integrity)
    updated = d.deployment_5090_program(c, integrity, initial)
    assert (
        updated.replace("import importlib;importlib.invalidate_caches()\n", "", 1)
        == original
    )
    assert updated.count("importlib.invalidate_caches()") == 1
    assert initial.deployment_program(c, "5090", integrity) == original
