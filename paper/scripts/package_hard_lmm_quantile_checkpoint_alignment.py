#!/usr/bin/env python3
"""Package one clean committed source revision plus the three pinned datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path("paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.json")
REQUIRED_COMMITTED = (
    CONTRACT,
    Path("paper/contracts/hard_lmm_quantile_checkpoint_alignment_v1.md"),
    Path("paper/scripts/compare_hard_lmm_quantile_checkpoint_alignment.py"),
    Path("paper/scripts/package_hard_lmm_quantile_checkpoint_alignment.py"),
    Path("paper/scripts/run_hard_lmm_quantile_checkpoint_alignment_5090.py"),
    Path("paper/scripts/run_count_aware_tpp_backbone_control.py"),
    Path("paper/scripts/count_aware_tpp_backbone/training.py"),
    Path("paper/scripts/count_aware_tpp_backbone/constants.py"),
    Path("models/TPPs/CountAwareTPP.py"),
    Path("models/TPPs/CountAwareFactory.py"),
    Path("simple_lab_test/search/tests/test_count_aware_quantile_cuda_contract.py"),
    Path("simple_lab_test/search/tests/test_count_aware_quantile_adaptive_loss.py"),
    Path("simple_lab_test/search/tests/test_count_aware_checkpoint_selection.py"),
    Path("simple_lab_test/search/tests/test_count_aware_quantile_alignment_runner.py"),
    Path("simple_lab_test/search/tests/test_compare_hard_lmm_quantile_checkpoint_alignment.py"),
    Path("simple_lab_test/search/tests/test_hard_lmm_quantile_alignment_5090.py"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=text)


def read_contract(revision: str) -> dict[str, Any]:
    raw = git("show", f"{revision}:{CONTRACT.as_posix()}")
    assert isinstance(raw, str)
    return json.loads(raw)


def package(output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite package: {output}")
    revision = str(git("rev-parse", "HEAD")).strip()
    if len(revision) != 40:
        raise ValueError("A full committed source revision is required")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT).returncode:
        raise ValueError("Tracked worktree changes must be committed before packaging")
    for relative in REQUIRED_COMMITTED:
        subprocess.check_call(
            ["git", "cat-file", "-e", f"{revision}:{relative.as_posix()}"],
            cwd=ROOT,
        )
    contract = read_contract(revision)
    with tempfile.TemporaryDirectory(prefix="hard_lmm_quantile_package_") as temporary:
        temp = Path(temporary)
        archive_path = temp / "source.tar"
        snapshot = temp / "snapshot"
        snapshot.mkdir()
        subprocess.check_call(
            ["git", "archive", "--format=tar", "--output", str(archive_path), revision],
            cwd=ROOT,
        )
        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(snapshot, filter="data")

        copied_inputs: list[str] = []
        for row in contract["datasets"]:
            for key in ("data_path", "split_manifest_path"):
                relative = Path(row[key])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe input path: {relative}")
                source = ROOT / relative
                if not source.is_file():
                    raise FileNotFoundError(source)
                expected = row["data_sha256" if key == "data_path" else "split_manifest_sha256"]
                if sha256_file(source) != expected:
                    raise ValueError(f"Pinned input checksum mismatch: {relative}")
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                copied_inputs.append(relative.as_posix())

        files = {
            path.relative_to(snapshot).as_posix(): sha256_file(path)
            for path in sorted(snapshot.rglob("*"))
            if path.is_file() and path.name != "source_manifest.json"
        }
        manifest = {
            "schema_version": 1,
            "source_revision": revision,
            "contract": CONTRACT.as_posix(),
            "contract_sha256": files[CONTRACT.as_posix()],
            "files": files,
            "explicit_untracked_pinned_inputs": sorted(set(copied_inputs)),
            "source_policy": "git_archive_of_one_clean_commit_plus_checksum_pinned_inputs",
            "held_out_test_evaluated": False,
        }
        (snapshot / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(snapshot.rglob("*")):
                archive.add(path, arcname=path.relative_to(snapshot), recursive=False)
    return {
        "output": str(output),
        "source_revision": revision,
        "archive_sha256": sha256_file(output),
        "file_count": len(manifest["files"]),
        "copied_inputs": manifest["explicit_untracked_pinned_inputs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(package(parser.parse_args().output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
