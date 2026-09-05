#!/usr/bin/env python3
"""Export committed elapsed-age e300 source and pinned audit-only references."""
from __future__ import annotations

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
from paper.scripts.run_hard_lmm_elapsed_age_screening import CONTRACT, frozen_documents
from paper.scripts.run_hard_lmm_elapsed_age_smoke import CUDA_TESTS, reference_files

REQUIRED_SOURCE = (
    CONTRACT, "paper/contracts/hard_lmm_elapsed_age_v1.json", "models/Titan/common/elapsed_age.py",
    "models/TPPs/CountAwareTPP.py", "models/TPPs/CountAwareFactory.py",
    "paper/scripts/run_hard_lmm_elapsed_age_smoke.py",
    "paper/scripts/run_hard_lmm_elapsed_age_screening.py",
    "paper/scripts/package_hard_lmm_elapsed_age_screening.py",
    "simple_lab_test/search/tests/test_hard_lmm_elapsed_age_screening.py",
    *CUDA_TESTS,
)


def package(output):
    spec, rows = frozen_documents()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    paths = ["models", "data_loader", "simple_lab_test", "paper/scripts", "paper/contracts", "utils"]
    paths += [name for name in spec["frozen_files"] if not any(Path(parent) in Path(name).parents for parent in paths)]
    if subprocess.check_output(["git", "diff", "HEAD", "--", *paths], cwd=ROOT):
        raise ValueError("Commit source changes before packaging")
    archive = subprocess.check_output(["git", "archive", revision, "--", *paths], cwd=ROOT)
    contents = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            name = Path(member.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Unsafe archive path")
            if member.isfile():
                if member.name in spec["frozen_files"] or name.suffix in {".py", ".json", ".md", ".sh", ".toml", ".yaml", ".yml", ".txt"}:
                    contents[member.name] = source.extractfile(member).read()
            elif not member.isdir():
                raise ValueError("No links or special files in a source package")
    for name in REQUIRED_SOURCE:
        if name not in contents:
            raise ValueError(f"Required source is not committed in this snapshot: {name}")
    for name, expected in spec["frozen_files"].items():
        if name not in contents or hashlib.sha256(contents[name]).hexdigest() != expected:
            raise ValueError(f"Frozen committed document mismatch: {name}")
    for row in rows:
        for name, (path, expected) in reference_files(row).items():
            data = (ROOT / path).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Frozen reference mismatch: {path}")
            contents[f"references/{row['dataset']}/{name}"] = data
    manifest = {"source_revision": revision, "scope": spec["scope"],
                "frozen_model_revision": spec["frozen_model_revision"],
                "performance_training_authorized": True, "empty_directories": ["sample_data"],
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(contents.items())}}
    contents["source_manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as target:
        sentinel = tarfile.TarInfo("sample_data")
        sentinel.type, sentinel.mode = tarfile.DIRTYPE, 0o755
        target.addfile(sentinel)
        for name, data in sorted(contents.items()):
            item = tarfile.TarInfo(name)
            item.size, item.mode = len(data), 0o644
            target.addfile(item, io.BytesIO(data))
    result = {"source_revision": revision, "files": len(manifest["files"]), "package": str(output),
              "bytes": output.stat().st_size, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    package(parser.parse_args().output)
