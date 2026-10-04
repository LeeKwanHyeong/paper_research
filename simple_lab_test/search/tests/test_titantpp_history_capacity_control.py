"""Terminal original retrieval integrity; synthetic files and no SSH execution."""
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.scripts import control_titantpp_history_capacity as control


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def original(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "B", tmp_path / "local")
    job = {"id": "synthetic__42__width8", "host": "5080", "seed": 42}
    c = {"jobs": [job], "hosts": {"5080": {"root": "/synthetic-isolated-remote"}}}
    source = tmp_path / "original"
    source.mkdir()
    (source / "selected.pt").write_bytes(b"synthetic original checkpoint bytes")
    write(source / "endpoint.json", {"evaluation_scope": "validation_only"})
    manifest = {"scientific_success": True, "status": "complete", "job": job,
                "contract_sha256": control.common.sha_json(c),
                "files": {p.name: control.common.sha_file(p) for p in source.iterdir()}}
    write(source / "terminal_manifest.json", manifest)
    proof = {"terminal_manifest_sha256": control.common.sha_file(source / "terminal_manifest.json")}
    observation = {"files": {"status.json": {"completed": {job["id"]: proof}}}}
    write(control.B / "observations/20261004T070000Z/5080.json", observation)
    return c, job, source, observation


def install_transfer(monkeypatch, source, job, *, corrupt=False, unsafe=False):
    calls = []

    def transfer(args, **kwargs):
        calls.append(args)
        assert args[0] == "ssh" and kwargs["timeout"] == 180
        with tarfile.open(fileobj=kwargs["stdout"], mode="w:gz") as stream:
            for p in source.iterdir():
                if corrupt and p.name == "selected.pt":
                    data = b"changed checkpoint"
                    info = tarfile.TarInfo(job["id"] + "/selected.pt")
                    info.size = len(data)
                    stream.addfile(info, io.BytesIO(data))
                else:
                    stream.add(p, arcname=job["id"] + "/" + p.name)
            if unsafe:
                data = b"unsafe"
                info = tarfile.TarInfo("../escaped.json")
                info.size = len(data)
                stream.addfile(info, io.BytesIO(data))
        return SimpleNamespace(returncode=0, stderr=b"")

    monkeypatch.setattr(control.subprocess, "run", transfer)
    return calls


def test_retrieved_binaries_are_verified_and_never_overwritten(original, monkeypatch):
    c, job, source, _ = original
    calls = install_transfer(monkeypatch, source, job)
    control.retrieve(c, "5080")
    target = control.B / "retrieved/5080/run" / job["id"]
    assert (target / "selected.pt").read_bytes() == (source / "selected.pt").read_bytes()
    receipt = next(control.B.glob("retrieval_staging/*/receipt.json"))
    proof = json.loads(receipt.read_text())["conditions"][job["id"]]
    assert proof["verified_files"] == 2 and proof["original_binary_retrieved"] is True
    assert proof["cpu_endpoint_replay_completed"] is False
    control.retrieve(c, "5080")
    assert len(calls) == 1
    (target / "selected.pt").write_bytes(b"local corruption")
    with pytest.raises(ValueError, match="original changed"):
        control.retrieve(c, "5080")
    assert len(calls) == 1


@pytest.mark.parametrize("corrupt,unsafe,error", [(True, False, "SHA mismatch"), (False, True, "Unsafe")])
def test_bad_copy_or_archive_path_never_promotes_to_originals(original, monkeypatch, corrupt, unsafe, error):
    c, job, source, _ = original
    install_transfer(monkeypatch, source, job, corrupt=corrupt, unsafe=unsafe)
    with pytest.raises(ValueError, match=error):
        control.retrieve(c, "5080")
    assert not (control.B / "retrieved/5080/run" / job["id"]).exists()


def test_foreign_or_unconfirmed_condition_is_not_downloaded(original, monkeypatch):
    c, job, _, observation = original
    monkeypatch.setattr(control.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("Unexpected SSH"))
    observation["files"]["status.json"]["completed"] = {}
    p = control.B / "observations/20261004T070000Z/5080.json"
    write(p, observation)
    control.retrieve(c, "5080")
    observation["files"]["status.json"]["completed"] = {"foreign_job": {"terminal_manifest_sha256": "a" * 64}}
    write(p, observation)
    with pytest.raises(ValueError, match="Foreign"):
        control.retrieve(c, "5080")
