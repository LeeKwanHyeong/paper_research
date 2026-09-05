#!/usr/bin/env python3
"""Package one committed source tree for an isolated prior-prefix experiment."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PREFIXES = ("models/", "data_loader/", "utils/", "paper/scripts/", "paper/contracts/",
                   "simple_lab_test/common/", "simple_lab_test/search/common/",
                   "simple_lab_test/search/tests/")
SOURCE_ROOT_FILES = {"pytest.ini", "pyproject.toml", "setup.cfg", "conftest.py",
                     "paper/__init__.py", "simple_lab_test/__init__.py",
                     "simple_lab_test/search/__init__.py",
                     "paper/results/titans_mac_prior_prefix_20260905/frozen_references.json"}
REQUIRED = ("paper/contracts/titans_mac_prior_prefix_v1.json",
            "paper/scripts/run_titans_mac_prior_prefix_5090.py",
            "paper/scripts/validate_titans_mac_prior_prefix_cuda.py",
            "paper/scripts/compare_titans_mac_prior_prefix.py",
            "paper/results/titans_mac_prior_prefix_20260905/frozen_references.json",
            "simple_lab_test/search/tests/test_titans_mac_prior_prefix_contract.py")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_relative(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def selected_source(name: str) -> bool:
    return (safe_relative(name) and not name.startswith("scripts/")
            and "__pycache__" not in Path(name).parts
            and not name.endswith((".pyc", ".pyo"))
            and (name in SOURCE_ROOT_FILES or name.startswith(SOURCE_PREFIXES)))


def committed_files(project: Path, revision: str) -> dict[str, bytes]:
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("A full committed Git revision is required")
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", revision + "^{commit}"], cwd=project,
        text=True).strip()
    if resolved != revision:
        raise ValueError("Revision did not resolve exactly")
    names = subprocess.check_output(
        ["git", "ls-tree", "-rz", "--name-only", revision], cwd=project).decode().split("\0")
    selected = [name for name in names if selected_source(name)]
    if not selected:
        raise ValueError("The commit contains no packageable source")
    raw = subprocess.check_output(
        ["git", "archive", "--format=tar", revision, "--", *selected], cwd=project)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            if not selected_source(member.name) or member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Source package rejects links or special files: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Cannot read committed source: {member.name}")
            files[member.name] = stream.read()
    missing = set(REQUIRED) - files.keys()
    if missing:
        raise ValueError(f"Required files are not committed: {sorted(missing)}")
    return files


def package(project: Path, revision: str, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite package: {output}")
    files = committed_files(project, revision)
    manifest = {"schema_version": 1, "contract_id": "titans_mac_prior_prefix_v1",
                "source_revision": revision, "origin": "git_archive_committed_tree",
                "held_out_test_evaluated": False,
                "scaffold_directories": ["sample_data"],
                "root_scripts_excluded": True,
                "files": {name: digest_bytes(data) for name, data in sorted(files.items())}}
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    files["source_manifest.json"] = manifest_bytes
    output.parent.mkdir(parents=True, exist_ok=True)
    # Uncompressed tar has a deterministic header; it does not embed gzip timestamps.
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:") as archive:
        # Root discovery imports require this directory. Data remain at the
        # separately checksum-verified --data-project path, outside the source.
        directory = tarfile.TarInfo("sample_data")
        directory.type, directory.mode, directory.mtime = tarfile.DIRTYPE, 0o755, 0
        archive.addfile(directory)
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            archive.addfile(info, io.BytesIO(data))
    with output.open("rb") as stream:
        package_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"source_revision": revision, "package": str(output.resolve()),
            "package_sha256": package_hash, "source_file_count": len(manifest["files"]),
            "source_manifest_sha256": digest_bytes(manifest_bytes),
            "contract_sha256": manifest["files"][REQUIRED[0]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.project_root.resolve(), args.source_revision, args.output), indent=2))


if __name__ == "__main__":
    main()
