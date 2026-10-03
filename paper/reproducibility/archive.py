"""Verify or restore a source snapshot without importing or executing its files.

The artifact contains ``manifest.json`` and ``objects/<sha256>`` byte objects::

    {"schema": 1, "files": [
      {"path": "models/example.py", "sha256": "<64 lowercase hex>", "bytes": 123}
    ]}

Additional manifest metadata is preserved but never interpreted as instructions.
Paths inside the manifest are canonical relative POSIX paths. A restore target
must not exist, and its parent directory must already exist without symlinks.

Examples::

    python paper/reproducibility/archive.py verify /path/to/artifact
    python paper/reproducibility/archive.py restore /path/to/artifact /path/to/new-checkout

No network, package installation, source import, or subprocess execution occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import sys


class ArchiveError(ValueError):
    """The artifact or restore destination violates the snapshot contract."""


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_symlinks(path):
    """Inspect the supplied path components without resolving away symlinks."""
    path = Path(path)
    if ".." in path.parts:
        raise ArchiveError(f"Parent traversal is forbidden: {path}")
    if not path.is_absolute():
        path = Path.cwd() / path
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ArchiveError(f"Symlink is forbidden: {current}")
    return path


def _regular_file(path):
    try:
        info = path.lstat()
    except OSError as error:
        raise ArchiveError(f"Cannot read required file: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ArchiveError(f"Expected a regular file: {path}")


def _manifest_path(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ArchiveError(f"Invalid manifest path: {value!r}")
    parts = value.split("/")
    if (PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in ("", ".", "..") for part in parts)):
        raise ArchiveError(f"Manifest path must be canonical and relative: {value!r}")
    return value


def _load_manifest(artifact):
    path = _no_symlinks(artifact / "manifest.json")
    _regular_file(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArchiveError(f"Invalid manifest: {path}") from error
    if not isinstance(manifest, dict) or type(manifest.get("schema")) is not int or manifest["schema"] != 1:
        raise ArchiveError("Manifest schema must be integer 1")
    if not isinstance(manifest.get("files"), list):
        raise ArchiveError("Manifest files must be a list")
    paths = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not {"path", "sha256", "bytes"} <= entry.keys():
            raise ArchiveError("Each file requires path, sha256 and bytes")
        relative = _manifest_path(entry["path"])
        if relative in paths:
            raise ArchiveError(f"Duplicate manifest path: {relative}")
        paths.add(relative)
        if not isinstance(entry["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None:
            raise ArchiveError(f"Invalid SHA256: {relative}")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ArchiveError(f"Invalid byte count: {relative}")
    for relative in paths:
        for parent in PurePosixPath(relative).parents:
            if parent.as_posix() in paths:
                raise ArchiveError(f"File/directory path conflict: {relative}")
    return manifest


def _verify(artifactdir, *, capture):
    artifact = _no_symlinks(artifactdir)
    if not artifact.is_dir():
        raise ArchiveError(f"Artifact is not a directory: {artifact}")
    manifest = _load_manifest(artifact)
    objects = _no_symlinks(artifact / "objects")
    if not objects.is_dir():
        raise ArchiveError(f"Object store is not a directory: {objects}")
    sizes, contents = {}, {}
    for entry in manifest["files"]:
        expected = entry["sha256"]
        if expected not in sizes:
            path = _no_symlinks(objects / expected)
            _regular_file(path)
            digest, size, chunks = hashlib.sha256(), 0, []
            # O_NOFOLLOW closes the final-component symlink race where supported.
            try:
                descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(descriptor, "rb") as stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise ArchiveError(f"Expected a regular object: {path}")
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                        if capture:
                            chunks.append(chunk)
            except OSError as error:
                raise ArchiveError(f"Cannot read object: {path}") from error
            if digest.hexdigest() != expected:
                raise ArchiveError(f"Object SHA256 mismatch: {expected}")
            sizes[expected] = size
            if capture:
                contents[expected] = b"".join(chunks)
        if sizes[expected] != entry["bytes"]:
            raise ArchiveError(f"Object byte count mismatch: {entry['path']}")
    return manifest, contents


def verify(artifactdir):
    """Validate all paths and referenced object bytes; return the manifest dict."""
    return _verify(artifactdir, capture=False)[0]


def restore(artifactdir, destination):
    """Restore verified bytes into a new directory, returning its absolute Path.

    All referenced bytes are checked and retained before creating the destination.
    This deliberately uses memory proportional to the unique source objects so a
    corrupt object cannot produce a partially restored tree.
    """
    target = _no_symlinks(destination)
    if os.path.lexists(target):
        raise ArchiveError(f"Restore destination already exists: {target}")
    if not target.parent.is_dir():
        raise ArchiveError(f"Restore parent must already be a directory: {target.parent}")
    manifest, contents = _verify(artifactdir, capture=True)
    # Recheck after verification, then claim the target exclusively.
    _no_symlinks(target)
    try:
        target.mkdir(mode=0o700)
    except OSError as error:
        raise ArchiveError(f"Cannot create new restore destination: {target}") from error
    try:
        for entry in manifest["files"]:
            path = target.joinpath(*entry["path"].split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            _no_symlinks(path)
            with path.open("xb") as stream:
                stream.write(contents[entry["sha256"]])
    except BaseException:
        shutil.rmtree(target)
        raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify", help="Check manifest paths and object hashes")
    check.add_argument("artifactdir", type=Path)
    extract = commands.add_parser("restore", help="Restore source bytes to a new directory")
    extract.add_argument("artifactdir", type=Path)
    extract.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            manifest = verify(args.artifactdir)
            print(json.dumps({"status": "verified", "files": len(manifest["files"])}))
        else:
            target = restore(args.artifactdir, args.destination)
            print(json.dumps({"status": "restored", "destination": str(target)}))
    except (ArchiveError, OSError) as error:
        print(f"archive: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
