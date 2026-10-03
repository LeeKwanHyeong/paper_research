"""Filesystem-only regression tests: restored source is never executed."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "paper/reproducibility/archive.py"
SPEC = importlib.util.spec_from_file_location("reproducibility_archive", SOURCE)
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def artifact(tmp_path, files=None):
    folder = tmp_path / "artifact"
    (folder / "objects").mkdir(parents=True)
    rows = []
    for path, data in (files or {"one.py": b"raise RuntimeError('must not execute')\n"}).items():
        digest = hashlib.sha256(data).hexdigest()
        (folder / "objects" / digest).write_bytes(data)
        rows.append({"path": path, "sha256": digest, "bytes": len(data)})
    manifest = {"schema": 1, "files": rows, "description": "opaque metadata"}
    save(folder, manifest)
    return folder, manifest


def save(folder, manifest):
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_exact_roundtrip_including_binary_empty_unicode_and_shared_objects(tmp_path):
    files = {"models/example.py": b"raise AssertionError('never execute')\n",
             "data.bin": bytes(range(256)), "nested/빈 파일": b"", "copy.bin": bytes(range(256))}
    folder, manifest = artifact(tmp_path, files)
    assert archive.verify(folder) == manifest
    target = tmp_path / "restored"
    assert archive.restore(folder, target) == target
    actual = {p.relative_to(target).as_posix(): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    assert actual == files


def test_corrupt_later_object_leaves_no_partial_restore(tmp_path):
    folder, manifest = artifact(tmp_path, {"first": b"valid", "second": b"original"})
    (folder / "objects" / manifest["files"][1]["sha256"]).write_bytes(b"corrupt!")
    target = tmp_path / "restored"
    with pytest.raises(archive.ArchiveError, match="SHA256 mismatch"):
        archive.restore(folder, target)
    assert not target.exists()


@pytest.mark.parametrize("path", ["../escape", "a/../../escape", "/absolute", "a/../b",
                                 "a/./b", "a//b", "a/", "", ".", "C:/absolute",
                                 "C:relative", "a\\b", "nul\x00name"])
def test_malicious_or_noncanonical_paths_are_rejected_before_writes(tmp_path, path):
    folder, manifest = artifact(tmp_path)
    manifest["files"][0]["path"] = path
    save(folder, manifest)
    with pytest.raises(archive.ArchiveError):
        archive.restore(folder, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("paths", [("same", "same"), ("a", "a/b"), ("a/b", "a")])
def test_duplicates_and_file_directory_conflicts(tmp_path, paths):
    folder, manifest = artifact(tmp_path)
    row = manifest["files"][0]
    manifest["files"] = [{**row, "path": path} for path in paths]
    save(folder, manifest)
    with pytest.raises(archive.ArchiveError, match="Duplicate|conflict"):
        archive.verify(folder)


@pytest.mark.parametrize("field,value", [("sha256", "A" * 64), ("sha256", "0" * 63),
                                        ("sha256", "../object"), ("sha256", None),
                                        ("bytes", -1), ("bytes", True), ("bytes", 1.5)])
def test_bad_object_metadata(tmp_path, field, value):
    folder, manifest = artifact(tmp_path)
    manifest["files"][0][field] = value
    save(folder, manifest)
    with pytest.raises(archive.ArchiveError):
        archive.verify(folder)


def test_size_mismatch_is_rejected_before_restore(tmp_path):
    folder, manifest = artifact(tmp_path)
    manifest["files"][0]["bytes"] += 1
    save(folder, manifest)
    with pytest.raises(archive.ArchiveError, match="byte count"):
        archive.restore(folder, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("kind", ["directory", "file", "symlink", "dangling_symlink"])
def test_existing_destination_is_never_overwritten(tmp_path, kind):
    folder, _ = artifact(tmp_path)
    target = tmp_path / "restored"
    if kind == "directory":
        target.mkdir()
    elif kind == "file":
        target.write_bytes(b"keep")
    else:
        target.symlink_to(folder if kind == "symlink" else tmp_path / "missing", target_is_directory=True)
    with pytest.raises(archive.ArchiveError):
        archive.restore(folder, target)
    assert target.is_symlink() if "symlink" in kind else target.exists()
    if kind == "file":
        assert target.read_bytes() == b"keep"


def test_symlink_destination_parent_is_rejected(tmp_path):
    folder, _ = artifact(tmp_path)
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(archive.ArchiveError, match="Symlink"):
        archive.restore(folder, alias / "restored")
    assert not (actual / "restored").exists()


@pytest.mark.parametrize("component", ["object", "objects", "manifest"])
def test_symlinks_in_artifact_are_rejected(tmp_path, component):
    folder, manifest = artifact(tmp_path)
    path = (folder / "objects" / manifest["files"][0]["sha256"] if component == "object"
            else folder / ("objects" if component == "objects" else "manifest.json"))
    real = tmp_path / "real"
    path.rename(real)
    path.symlink_to(real, target_is_directory=component == "objects")
    with pytest.raises(archive.ArchiveError, match="Symlink"):
        archive.verify(folder)


def test_missing_object_and_duplicate_json_keys(tmp_path):
    folder, manifest = artifact(tmp_path)
    (folder / "objects" / manifest["files"][0]["sha256"]).unlink()
    with pytest.raises(archive.ArchiveError, match="required file"):
        archive.verify(folder)
    (folder / "manifest.json").write_text('{"schema":1,"schema":1,"files":[]}')
    with pytest.raises(archive.ArchiveError, match="Duplicate JSON key"):
        archive.verify(folder)


def test_cli_success_and_failure_do_not_execute_source(tmp_path, capsys):
    folder, manifest = artifact(tmp_path)
    assert archive.main(["verify", str(folder)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "verified"
    target = tmp_path / "restored"
    assert archive.main(["restore", str(folder), str(target)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "restored"
    assert archive.main(["restore", str(folder), str(target)]) == 1
    assert "already exists" in capsys.readouterr().err
