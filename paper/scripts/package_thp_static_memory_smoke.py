#!/usr/bin/env python3
"""Export committed source plus the hash-pinned pre-change factory for 5080."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = "frozen_references/CountAwareFactory_7f0bf8d.py"
REFERENCE_SHA256 = "0937ccf3219f5c75f5d8ec01cee6a5716e2472cdf546734029f021f9bfc8c588"


def package(output):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = ["models", "data_loader", "simple_lab_test", "paper/scripts", "paper/contracts", "utils"]
    dirty = subprocess.check_output(["git", "diff", "HEAD", "--", *paths], cwd=ROOT)
    if dirty:
        raise ValueError("Commit source changes before packaging")
    archive = subprocess.check_output(["git", "archive", revision, "--", *paths], cwd=ROOT)
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if member.isfile():
                if Path(member.name).suffix in {".py", ".json", ".md", ".sh", ".toml", ".yaml", ".yml", ".txt"}:
                    contents[member.name] = source.extractfile(member).read()
            elif not member.isdir():
                raise ValueError("No links or special files in a source package")
    reference = subprocess.check_output(["git", "show", "7f0bf8d:models/TPPs/CountAwareFactory.py"], cwd=ROOT)
    if hashlib.sha256(reference).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Frozen factory mismatch")
    contents[REFERENCE_PATH] = reference
    manifest = {"source_revision": revision, "scope": "5080_cuda_and_full_taxi_raf_e1_only",
                "reference_revision": "7f0bf8de81a627db3df7609b6d00c3e9617769cc",
                "reference_file": REFERENCE_PATH, "performance_training_authorized": False,
                "empty_directories": ["sample_data"],
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}}
    contents["source_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    # Exclusive creation avoids silently replacing a previously approved package.
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as target:
        # Root discovery requires this sentinel, but datasets stay in the existing
        # server project and are passed to the runner by explicit absolute paths.
        directory = tarfile.TarInfo("sample_data")
        directory.type, directory.mode = tarfile.DIRTYPE, 0o755
        target.addfile(directory)
        for name, data in sorted(contents.items()):
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            target.addfile(item, io.BytesIO(data))
    print(json.dumps({"source_revision": revision, "files": len(manifest["files"]),
                      "package": str(output), "bytes": output.stat().st_size,
                      "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args().output)
