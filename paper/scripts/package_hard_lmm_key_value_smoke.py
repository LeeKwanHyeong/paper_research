#!/usr/bin/env python3
"""Export committed source and pinned audit-only references for isolated 5090 e1."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.run_hard_lmm_key_value_smoke import frozen_documents, reference_files


def package(output):
    spec, rows = frozen_documents()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = ["models", "data_loader", "simple_lab_test", "paper/scripts", "paper/contracts", "utils"]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *paths], cwd=ROOT):
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
    for row in rows:
        for name, (path, expected) in reference_files(row).items():
            data = (ROOT / path).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Frozen reference mismatch: {path}")
            contents[f"references/{row['dataset']}/{name}"] = data
    manifest = {"source_revision": revision, "scope": spec["scope"],
                "frozen_model_revision": spec["frozen_model_revision"],
                "performance_training_authorized": False, "empty_directories": ["sample_data"],
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}}
    contents["source_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as target:
        # Root discovery needs the sentinel; real datasets remain outside the snapshot.
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
