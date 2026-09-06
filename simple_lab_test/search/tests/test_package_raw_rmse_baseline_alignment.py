"""Clean-commit and manifest tests for baseline-alignment packaging."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts import package_raw_rmse_baseline_alignment as packaging
from paper.scripts import run_raw_rmse_baseline_alignment_job as launcher


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def run_git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


@pytest.fixture
def clean_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    root = tmp_path / "repository"
    root.mkdir()
    subprocess.check_call(["git", "init", "-q"], cwd=root)
    subprocess.check_call(["git", "config", "user.name", "Package Test"], cwd=root)
    subprocess.check_call(["git", "config", "user.email", "package@test.invalid"], cwd=root)

    input_rows = []
    dataset_names = (
        "intermittent_frozen_5000",
        "yellow_trip_hourly",
        "insta_market_basket",
    )
    for index, dataset in enumerate(dataset_names):
        data_relative = Path(f"sample_data/{dataset}/data.parquet")
        split_relative = Path(f"sample_data/{dataset}/split.json")
        data_bytes = f"data-{index}\n".encode()
        split_bytes = f'{{"split": {index}}}\n'.encode()
        for relative, payload in ((data_relative, data_bytes), (split_relative, split_bytes)):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        input_rows.append({
            "dataset": dataset,
            "data_path": data_relative.as_posix(),
            "data_sha256": digest(data_bytes),
            "split_manifest_path": split_relative.as_posix(),
            "split_manifest_sha256": digest(split_bytes),
        })

    for relative in packaging.REQUIRED_COMMITTED:
        if relative == packaging.CONTRACT:
            continue
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"committed fixture for {relative.as_posix()}\n", encoding="utf-8")
    evidence_sha256 = packaging.sha256_file(root / packaging.B_REFERENCE_EVIDENCE)
    monkeypatch.setattr(
        packaging,
        "B_REFERENCE_EVIDENCE_SHA256",
        evidence_sha256,
    )
    monkeypatch.setattr(
        launcher,
        "B_REFERENCE_EVIDENCE_SHA256",
        evidence_sha256,
    )
    contract_path = root / packaging.CONTRACT
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps({
            "schema_version": 1,
            "contract_id": "raw_rmse_baseline_alignment_seed42_v1",
            "datasets": input_rows,
            "B_reference_evidence": {
                "path": packaging.B_REFERENCE_EVIDENCE.as_posix(),
                "sha256": evidence_sha256,
            },
            "held_out_lock": {"held_out_test_evaluated": False},
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    nested_manifest = root / "paper/results/historical/source_manifest.json"
    nested_manifest.parent.mkdir(parents=True, exist_ok=True)
    nested_manifest.write_text('{"historical": true}\n', encoding="utf-8")
    for relative in packaging.REQUIRED_COMMITTED:
        subprocess.check_call(["git", "add", relative.as_posix()], cwd=root)
    subprocess.check_call(
        ["git", "add", "paper/results/historical/source_manifest.json"],
        cwd=root,
    )
    subprocess.check_call(["git", "commit", "-q", "-m", "fixture"], cwd=root)
    (root / "untracked-scratch.txt").write_text("must not enter package\n", encoding="utf-8")
    monkeypatch.setattr(packaging, "ROOT", root)
    return {
        "root": root,
        "revision": run_git(root, "rev-parse", "HEAD"),
        "rows": input_rows,
    }


def test_package_archives_clean_head_adds_six_pinned_inputs_and_is_runner_accepted(
    clean_repository: dict,
    tmp_path: Path,
) -> None:
    output = tmp_path / "baseline-alignment.tar.gz"
    result = packaging.package(output)
    assert result["source_revision"] == clean_repository["revision"]
    assert result["copied_input_count"] == 6
    assert result["held_out_test_evaluated"] is False

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    with tarfile.open(output, "r:gz") as archive:
        archive.extractall(snapshot, filter="data")
    manifest = json.loads((snapshot / "source_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_revision"] == clean_repository["revision"]
    assert manifest["held_out_test_evaluated"] is False
    assert manifest["evaluation_scope"] == "validation_only"
    assert manifest["source_policy"] == packaging.SOURCE_POLICY == launcher.SOURCE_POLICY
    assert manifest["B_reference_evidence"] == packaging.B_REFERENCE_EVIDENCE.as_posix()
    assert manifest["B_reference_evidence_sha256"] == packaging.B_REFERENCE_EVIDENCE_SHA256
    assert len(manifest["explicit_checksum_pinned_inputs"]) == 6
    assert "untracked-scratch.txt" not in manifest["files"]
    assert "paper/results/historical/source_manifest.json" in manifest["files"]
    assert set(manifest["required_committed_files"]) == {
        path.as_posix() for path in packaging.REQUIRED_COMMITTED
    }
    assert manifest["required_committed_files"] == [
        path.as_posix() for path in packaging.REQUIRED_COMMITTED
    ]
    assert set(manifest["files"]) == {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and path != snapshot / "source_manifest.json"
    }
    for row in clean_repository["rows"]:
        assert launcher.sha256_file(snapshot / row["data_path"]) == row["data_sha256"]
        assert launcher.sha256_file(snapshot / row["split_manifest_path"]) == row["split_manifest_sha256"]

    accepted = launcher.verify_source_manifest(
        snapshot,
        revision=clean_repository["revision"],
        contract_path=snapshot / packaging.CONTRACT,
    )
    assert accepted == manifest


def package_and_extract(clean_repository: dict, tmp_path: Path, name: str) -> Path:
    output = tmp_path / f"{name}.tar.gz"
    packaging.package(output)
    snapshot = tmp_path / name
    snapshot.mkdir()
    with tarfile.open(output, "r:gz") as archive:
        archive.extractall(snapshot, filter="data")
    return snapshot


def verify_fixture_snapshot(snapshot: Path, clean_repository: dict) -> dict:
    return launcher.verify_source_manifest(
        snapshot,
        revision=clean_repository["revision"],
        contract_path=snapshot / packaging.CONTRACT,
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_policy", "weaker_policy"),
        ("required_committed_files", []),
        ("explicit_checksum_pinned_inputs", []),
        ("explicit_untracked_pinned_inputs", []),
        ("contract_sha256", "0" * 64),
        ("B_reference_evidence_sha256", "0" * 64),
    ),
)
def test_runner_refuses_manifest_metadata_drift(
    clean_repository: dict,
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    snapshot = package_and_extract(clean_repository, tmp_path, f"metadata-{field}")
    manifest_path = snapshot / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = replacement
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|Manifest"):
        verify_fixture_snapshot(snapshot, clean_repository)


@pytest.mark.parametrize(
    "field",
    (
        "required_committed_files",
        "explicit_checksum_pinned_inputs",
        "explicit_untracked_pinned_inputs",
    ),
)
def test_runner_refuses_manifest_list_order_drift(
    clean_repository: dict,
    tmp_path: Path,
    field: str,
) -> None:
    snapshot = package_and_extract(clean_repository, tmp_path, f"order-{field}")
    manifest_path = snapshot / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = list(reversed(manifest[field]))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|Manifest"):
        verify_fixture_snapshot(snapshot, clean_repository)


def test_runner_refuses_unmanifested_project_file(
    clean_repository: dict,
    tmp_path: Path,
) -> None:
    snapshot = package_and_extract(clean_repository, tmp_path, "unmanifested")
    (snapshot / "rogue.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        verify_fixture_snapshot(snapshot, clean_repository)


def test_runner_refuses_reduced_files_mapping(
    clean_repository: dict,
    tmp_path: Path,
) -> None:
    snapshot = package_and_extract(clean_repository, tmp_path, "reduced")
    manifest_path = snapshot / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop(clean_repository["rows"][0]["data_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        verify_fixture_snapshot(snapshot, clean_repository)


def test_runner_refuses_source_manifest_schema_reduction_or_extension(
    clean_repository: dict,
    tmp_path: Path,
) -> None:
    snapshot = package_and_extract(clean_repository, tmp_path, "schema")
    manifest_path = snapshot / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("evaluation_scope")
    manifest["unreviewed_field"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema drifted"):
        verify_fixture_snapshot(snapshot, clean_repository)


def test_package_refuses_dirty_tracked_state(
    clean_repository: dict,
    tmp_path: Path,
) -> None:
    tracked = clean_repository["root"] / packaging.REQUIRED_COMMITTED[1]
    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Tracked worktree changes"):
        packaging.package(tmp_path / "dirty.tar.gz")


def test_package_refuses_pinned_input_checksum_drift(
    clean_repository: dict,
    tmp_path: Path,
) -> None:
    data_path = clean_repository["root"] / clean_repository["rows"][0]["data_path"]
    data_path.write_bytes(b"changed untracked data\n")
    with pytest.raises(ValueError, match="Pinned input checksum mismatch"):
        packaging.package(tmp_path / "drift.tar.gz")


def test_package_refuses_existing_output_before_writing(
    clean_repository: dict,
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing.tar.gz"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        packaging.package(output)
    assert output.read_bytes() == b"preserve"


def test_package_refuses_required_source_that_is_not_at_head(
    clean_repository: dict,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = Path("paper/scripts/uncommitted_required.py")
    path = clean_repository["root"] / missing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("untracked\n", encoding="utf-8")
    monkeypatch.setattr(
        packaging,
        "REQUIRED_COMMITTED",
        (*packaging.REQUIRED_COMMITTED, missing),
    )
    with pytest.raises(ValueError, match="not committed at HEAD"):
        packaging.package(tmp_path / "missing.tar.gz")
