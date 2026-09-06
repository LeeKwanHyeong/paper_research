#!/usr/bin/env python3
"""Package one clean committed baseline-alignment snapshot and pinned inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path("paper/contracts/raw_rmse_baseline_alignment_seed42_v1.json")
B_REFERENCE_EVIDENCE = Path(
    "paper/results/hard_lmm_raw_rmse_checkpoint_alignment_seed42_20260906/"
    "a_vs_b_metrics.json"
)
B_REFERENCE_EVIDENCE_SHA256 = (
    "17b1c432b58f0620d8e6eda1a291a92eb4c995af9c78d98be8d460be79079db3"
)
PACKAGE_SCRIPT = Path("paper/scripts/package_raw_rmse_baseline_alignment.py")
REQUIRED_COMMITTED = (
    CONTRACT,
    B_REFERENCE_EVIDENCE,
    PACKAGE_SCRIPT,
    Path("paper/scripts/run_raw_rmse_baseline_alignment_job.py"),
    Path("paper/scripts/control_raw_rmse_baseline_alignment.py"),
    Path("paper/scripts/run_count_aware_tpp_backbone_control.py"),
    Path("paper/scripts/count_aware_tpp_backbone/constants.py"),
    Path("paper/scripts/count_aware_tpp_backbone/datasets.py"),
    Path("paper/scripts/count_aware_tpp_backbone/core.py"),
    Path("paper/scripts/count_aware_tpp_backbone/training.py"),
    Path("paper/scripts/count_aware_tpp_backbone/reporting.py"),
    Path("models/TPPs/CountAwareTPP.py"),
    Path("models/TPPs/CountAwareFactory.py"),
    Path("data_loader/event_seq_data_module.py"),
    Path("simple_lab_test/search/tests/test_raw_rmse_baseline_alignment_role.py"),
    Path("simple_lab_test/search/tests/test_raw_rmse_baseline_alignment_orchestration.py"),
    Path("simple_lab_test/search/tests/test_package_raw_rmse_baseline_alignment.py"),
)
SOURCE_POLICY = (
    "git_archive_of_one_clean_commit_plus_three_checksum_pinned_data_split_pairs"
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str, text: bool = True) -> str | bytes:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=ROOT,
        text=text,
        stderr=subprocess.PIPE,
    )


def read_committed_contract(revision: str) -> dict[str, Any]:
    raw = git("show", f"{revision}:{CONTRACT.as_posix()}")
    if not isinstance(raw, str):
        raise TypeError("Committed contract must be UTF-8 text")
    value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise ValueError("Committed contract must be a JSON object")
    return value


def _require_clean_tracked_head() -> str:
    revision = str(git("rev-parse", "HEAD")).strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("A full lowercase committed source revision is required")
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
    )
    if dirty.returncode != 0:
        raise ValueError("Tracked worktree changes must be committed before packaging")
    missing: list[str] = []
    for relative in REQUIRED_COMMITTED:
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{revision}:{relative.as_posix()}"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            missing.append(relative.as_posix())
    if missing:
        raise ValueError(
            "Required baseline-alignment files are not committed at HEAD: "
            + ", ".join(missing)
        )
    return revision


def _pinned_inputs(contract: dict[str, Any]) -> list[tuple[Path, str]]:
    rows = contract.get("datasets")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("Contract must pin exactly three datasets")
    expected_names = (
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    )
    if tuple(row.get("dataset") for row in rows) != expected_names:
        raise ValueError("Contract dataset order/scope drifted")
    inputs: list[tuple[Path, str]] = []
    for row in rows:
        for path_key, digest_key in (
            ("data_path", "data_sha256"),
            ("split_manifest_path", "split_manifest_sha256"),
        ):
            relative = Path(str(row[path_key]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe pinned input path: {relative}")
            expected = str(row[digest_key])
            if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
                raise ValueError(f"Invalid pinned SHA-256 for {relative}")
            inputs.append((relative, expected))
    if len({relative.as_posix() for relative, _ in inputs}) != 6:
        raise ValueError("The three data/split pairs must use six distinct paths")
    return inputs


def package(output: Path) -> dict[str, Any]:
    """Write a self-auditing tar.gz without consulting uncommitted source."""
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite package: {output}")
    revision = _require_clean_tracked_head()
    contract = read_committed_contract(revision)
    if contract.get("contract_id") != "raw_rmse_baseline_alignment_seed42_v1":
        raise ValueError("Wrong committed baseline-alignment contract")
    reference = contract.get("B_reference_evidence")
    if not isinstance(reference, dict):
        raise ValueError("Committed contract has no B reference evidence")
    if reference.get("path") != B_REFERENCE_EVIDENCE.as_posix():
        raise ValueError("Committed B reference evidence path drifted")
    evidence_path = ROOT / B_REFERENCE_EVIDENCE
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    if reference.get("sha256") != B_REFERENCE_EVIDENCE_SHA256:
        raise ValueError("Committed B reference evidence pin drifted")
    if sha256_file(evidence_path) != B_REFERENCE_EVIDENCE_SHA256:
        raise ValueError("Committed B reference evidence checksum drifted")
    inputs = _pinned_inputs(contract)

    with tempfile.TemporaryDirectory(prefix="raw_rmse_baseline_alignment_package_") as temporary:
        temporary_root = Path(temporary)
        source_tar = temporary_root / "source.tar"
        snapshot = temporary_root / "snapshot"
        snapshot.mkdir()
        subprocess.check_call(
            ["git", "archive", "--format=tar", "--output", str(source_tar), revision],
            cwd=ROOT,
        )
        with tarfile.open(source_tar, "r") as archive:
            archive.extractall(snapshot, filter="data")

        copied_inputs: list[str] = []
        for relative, expected in inputs:
            source = ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            if sha256_file(source) != expected:
                raise ValueError(f"Pinned input checksum mismatch: {relative}")
            destination = snapshot / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied_inputs.append(relative.as_posix())

        files = {
            path.relative_to(snapshot).as_posix(): sha256_file(path)
            for path in sorted(snapshot.rglob("*"))
            if path.is_file() and path != snapshot / "source_manifest.json"
        }
        missing_manifest_entries = [
            relative.as_posix()
            for relative in REQUIRED_COMMITTED
            if relative.as_posix() not in files
        ]
        if missing_manifest_entries:
            raise ValueError(
                "Required committed files are absent from the archive manifest: "
                + ", ".join(missing_manifest_entries)
            )
        manifest = {
            "schema_version": 1,
            "source_revision": revision,
            "contract": CONTRACT.as_posix(),
            "contract_sha256": files[CONTRACT.as_posix()],
            "B_reference_evidence": B_REFERENCE_EVIDENCE.as_posix(),
            "B_reference_evidence_sha256": files[B_REFERENCE_EVIDENCE.as_posix()],
            "files": files,
            "required_committed_files": [path.as_posix() for path in REQUIRED_COMMITTED],
            "explicit_checksum_pinned_inputs": sorted(copied_inputs),
            # Preserve the legacy field consumed by older package audit tools.
            "explicit_untracked_pinned_inputs": sorted(copied_inputs),
            "source_policy": SOURCE_POLICY,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }
        (snapshot / "source_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w:gz") as archive:
            for path in sorted(snapshot.rglob("*")):
                archive.add(
                    path,
                    arcname=path.relative_to(snapshot),
                    recursive=False,
                )

    return {
        "output": str(output),
        "source_revision": revision,
        "archive_sha256": sha256_file(output),
        "file_count": len(manifest["files"]),
        "copied_input_count": len(copied_inputs),
        "copied_inputs": manifest["explicit_checksum_pinned_inputs"],
        "held_out_test_evaluated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(package(parse_args(argv).output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
