"""Capture regressions using synthetic source bundles, never research artifacts."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[3] / "paper/reproducibility/capture_campaigns.py"
SPEC = importlib.util.spec_from_file_location("reproducibility_capture", SOURCE)
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def canonical(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(capture, "ROOT", tmp_path / "project")
    monkeypatch.setattr(capture, "HERE", tmp_path / "archive")
    root = capture.ROOT / "search_artifacts" / "synthetic"
    source = root / "frozen_source"
    scientific = {
        "models/vendor/s2p2/LICENCE": b"Synthetic license bytes\n",
        "models/coefficients.unusual-scientific-extension": bytes(range(256)),
        "models/opaque_source": b"\x00\xffsource with no suffix\n",
        "runner.py": b"raise RuntimeError('source must never execute')\n",
    }
    for relative, data in scientific.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    hashes = {path: hashlib.sha256(data).hexdigest() for path, data in scientific.items()}
    contract = {"source": {"files": hashes, "files_sha256": canonical(hashes)}}
    expected = canonical(contract)
    write_json(root / "execution_contract.json", contract)
    write_json(root / "current.json", {"contract_sha256": expected})
    return {"root": root, "source": source, "scientific": scientific,
            "contract": contract, "expected": expected,
            "destination": capture.HERE / "snapshots" / "synthetic"}


def execute(bundle, expected=None):
    capture.capture_campaign("synthetic", "synthetic", "frozen_source",
                             bundle["expected"] if expected is None else expected)


def test_scientific_manifest_preserves_licence_and_arbitrary_extensions(bundle):
    execute(bundle)
    destination = bundle["destination"]
    manifest = json.loads((destination / "manifest.json").read_text())
    rows = {row["path"]: row for row in manifest["files"]}
    assert manifest["scientific_source_file_count"] == len(bundle["scientific"])
    assert manifest["scientific_source_closure_sha256"] == bundle["contract"]["source"]["files_sha256"]
    assert manifest["canonical_contract_sha256"] == bundle["expected"]
    assert set(rows) == {"execution_contract.json", "current.json"} | {
        f"frozen_source/{relative}" for relative in bundle["scientific"]}
    for relative, expected_bytes in bundle["scientific"].items():
        row = rows[f"frozen_source/{relative}"]
        assert row["bytes"] == len(expected_bytes)
        assert row["sha256"] == hashlib.sha256(expected_bytes).hexdigest()
        assert (destination / "objects" / row["sha256"]).read_bytes() == expected_bytes


@pytest.mark.parametrize("corruption", ["file_bytes", "closure_digest"])
def test_source_mismatch_creates_no_capture(bundle, corruption):
    if corruption == "file_bytes":
        (bundle["source"] / "models/opaque_source").write_bytes(b"changed")
    else:
        # Keep the outer contract/current binding valid, isolating closure checks.
        contract = bundle["contract"]
        contract["source"]["files_sha256"] = "0" * 64
        bundle["expected"] = canonical(contract)
        write_json(bundle["root"] / "execution_contract.json", contract)
        write_json(bundle["root"] / "current.json", {"contract_sha256": bundle["expected"]})
    with pytest.raises(ValueError, match="Frozen source differs|Source closure changed"):
        execute(bundle)
    assert not bundle["destination"].exists()
    assert not capture.HERE.exists()


@pytest.mark.parametrize("corruption", ["contract_bytes", "current_binding"])
def test_contract_mismatch_creates_no_capture(bundle, corruption):
    if corruption == "contract_bytes":
        contract = bundle["contract"]
        contract["unexpected_change"] = True
        write_json(bundle["root"] / "execution_contract.json", contract)
    else:
        write_json(bundle["root"] / "current.json", {"contract_sha256": "0" * 64})
    with pytest.raises(ValueError, match="Contract changed"):
        execute(bundle)
    assert not bundle["destination"].exists()
    assert not capture.HERE.exists()


def test_existing_capture_is_rejected_without_overwriting(bundle):
    execute(bundle)
    destination = bundle["destination"]
    before = {path.relative_to(destination): path.read_bytes()
              for path in destination.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        execute(bundle)
    after = {path.relative_to(destination): path.read_bytes()
             for path in destination.rglob("*") if path.is_file()}
    assert after == before
