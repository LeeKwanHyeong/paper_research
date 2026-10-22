"""Independent CPU-only v2 bootstrap and runtime gate checks."""

from copy import deepcopy
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


def load(name):
    spec = importlib.util.spec_from_file_location(
        "qa_v2_" + name, HERE / (name + ".py")
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


d = load("initial_execution")
p = load("prepare_execution")


@pytest.mark.parametrize("fault", [None, "member", "extra", "existing"])
def test_real_generated_deploy_prefix_is_exclusive_and_checks_bytes(
    tmp_path, monkeypatch, fault
):
    root = tmp_path / "native"
    source = b"synthetic fixture only"
    c = {
        "hosts": {"5080": {"root": str(root)}},
        "source": {"files": {"fixture": hashlib.sha256(source).hexdigest()}},
        "input_files": {},
        "legacy_files": {},
    }
    raw = json.dumps(c).encode()
    buff = io.BytesIO()
    with tarfile.open(fileobj=buff, mode="w:gz") as archive:
        entries = [
            ("execution_contract.json", raw),
            ("source/fixture", b"tamper" if fault == "member" else source),
        ]
        if fault == "extra":
            entries.append(("source/unexpected", b"extra"))
        for name, data in entries:
            item = tarfile.TarInfo(name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
        item = tarfile.TarInfo("source/sample_data")
        item.type = tarfile.DIRTYPE
        archive.addfile(item)
    payload = buff.getvalue()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "deployment.tar.gz").write_bytes(payload)
    integrity = {
        "archive_sha256": hashlib.sha256(payload).hexdigest(),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
    }
    monkeypatch.setattr(d, "BUNDLE", bundle)
    if fault == "existing":
        root.mkdir()
        (root / "preserved").write_bytes(b"keep")
    generated = d.deployment_program(c, "5080", integrity, p.original_helper())
    marker = "from paper.scripts import manage_shared_history_mlp_control as m\n"
    prefix = generated.split(marker)[0] + "\nprint('bootstrap passed')\n"
    result = subprocess.run(
        [sys.executable, "-c", prefix],
        input=payload,
        capture_output=True,
        timeout=20,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(root / "source"),
        },
    )
    if fault is None:
        assert result.returncode == 0, result.stderr.decode()
        assert (root / "source/fixture").read_bytes() == source
        assert result.stdout == b"bootstrap passed\n"
    else:
        assert result.returncode != 0
        if fault == "existing":
            assert (root / "preserved").read_bytes() == b"keep"
            assert list(root.iterdir()) == [root / "preserved"]
        else:
            assert not root.exists()


@pytest.mark.parametrize(
    "fault", [None, "missing", "deterministic", "gpu", "extra_file", "symlink"]
)
def test_every_capacity_runtime_and_exact_snapshot_required(
    tmp_path, monkeypatch, fault
):
    state = tmp_path / "state"
    snapshot = state / "5080/snapshot"
    expected = {"torch": "fixture", "deterministic_algorithms": True, "threads": 4}
    c = {"hosts": {"5080": {"runtime_expected": expected, "gpu_uuid": "fixture-gpu"}}}
    monkeypatch.setattr(d, "STATE", state)
    entries = {}
    for index, arm in enumerate(d.manager.ARMS):
        runtime = {**deepcopy(expected), "gpu": {"uuid": "fixture-gpu"}}
        if index == 1:
            if fault == "missing":
                runtime = {}
            elif fault == "deterministic":
                runtime["deterministic_algorithms"] = False
            elif fault == "gpu":
                runtime["gpu"]["uuid"] = "other"
        name = "cost/" + arm + ".json"
        d.write(snapshot / name, {"runtime": runtime})
        entries[arm] = {"path": name}
    d.write(
        state / "5080/QUALIFICATION_PASS.json",
        {
            "status": "PASS",
            "host": "5080",
            "contract_sha256": d.canonical(c),
            "snapshot": "snapshot",
            "files": {
                str(x.relative_to(snapshot)): d.sha(x)
                for x in snapshot.rglob("*")
                if x.is_file()
            },
        },
    )
    if fault == "extra_file":
        (snapshot / "extra").write_text("unexpected")
    elif fault == "symlink":
        (snapshot / "link").symlink_to(snapshot / next(iter(entries.values()))["path"])
    monkeypatch.setattr(
        d.manager,
        "verify_phase_receipt",
        lambda *args: ({"capacity_receipts": entries}, {}),
    )
    if fault is None:
        assert d.verify_pass(c, "5080")["status"] == "PASS"
    else:
        with pytest.raises(RuntimeError, match="runtime|inventory"):
            d.verify_pass(c, "5080")


def test_version_imports_do_not_mutate_frozen_module_globals():
    from paper.scripts import advance_shared_history_mlp_control as original
    from paper.scripts import compare_shared_history_mlp_control as comparison

    before = (
        original.BUNDLE,
        original.REPORT,
        original.COMPARATOR,
        comparison.manager.BUNDLE,
    )
    load("advance").comparator_module()
    load("compare")
    assert before == (
        original.BUNDLE,
        original.REPORT,
        original.COMPARATOR,
        comparison.manager.BUNDLE,
    )
