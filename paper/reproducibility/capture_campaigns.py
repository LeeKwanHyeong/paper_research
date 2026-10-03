"""Capture the explicitly listed 2026-10-03 source/contract bundles, offline.

No dataset, checkpoint, prediction, live status or secret file is selected.
This is an archival operation, never a launcher. Existing captures are immutable.
Run from any directory: python paper/reproducibility/capture_campaigns.py
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CAMPAIGNS = {
    "width16_seed42": (
        "titantpp_history_width16_dual_20261003_v1", "frozen_source",
        "7b0681473fcde54a16b564821448490a5d6a5561c9d51b60903c9b7f2981e548"),
    "width16_seed52_62": (
        "titantpp_history_width16_replication_20261003_v1", "frozen_source",
        "dea8ccb93ccfdd3b9c76319dc7a0f894e67e02fbd2ea546d668843dd71251a7a"),
    "routing_a100_seed42": (
        "titantpp_routing_placement_a100_20261003_v1", "source",
        "1db5a7e0cf97b8a667439b86cca8455a6e98158775c46ef1d19efdd26a5cbea0"),
}
TOP_LEVEL = (
    "README.md", "design.json", "execution_contract.json", "current.json",
    "approval.json", "start_permit.json", "training_permit.json",
    "launch_receipt.json", "launch_confirmation.json", "launch_report.md",
    "source_lineage.json", "expected_initialization.json", "user_authorization.json",
    "transfer_authorization.json", "deployment_manifest.json", "package_manifest.json",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False).encode())


def write_capture(destination: Path, paths: dict[str, Path], metadata: dict) -> dict:
    """Read and verify twice before creating a new content-addressed capture."""
    if destination.exists():
        raise FileExistsError(destination)
    blobs = {}
    rows = []
    for rel, src in sorted(paths.items()):
        if src.is_symlink() or not src.is_file():
            raise ValueError(f"Not an ordinary file: {src}")
        data = src.read_bytes()
        sha = digest(data)
        rows.append({"path": rel, "sha256": sha, "bytes": len(data)})
        blobs[sha] = data
    for row in rows:
        if digest(paths[row["path"]].read_bytes()) != row["sha256"]:
            raise ValueError(f"Source changed during capture: {row['path']}")
    manifest = {"schema": 1, "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                **metadata, "files": rows}
    destination.mkdir(parents=True, exist_ok=False)
    objects = destination / "objects"
    objects.mkdir()
    for sha, data in blobs.items():
        (objects / sha).write_bytes(data)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def capture_campaign(name: str, bundle_name: str, source_dir: str, expected: str):
    bundle = ROOT / "search_artifacts" / bundle_name
    contract = json.loads((bundle / "execution_contract.json").read_text())
    current = json.loads((bundle / "current.json").read_text())
    if canonical(contract) != expected or current["contract_sha256"] != expected:
        raise ValueError(f"Contract changed: {name}")
    source = contract["source"]
    if canonical(source["files"]) != source["files_sha256"]:
        raise ValueError(f"Source closure changed: {name}")
    for rel, sha in source["files"].items():
        if digest((bundle / source_dir / rel).read_bytes()) != sha:
            raise ValueError(f"Frozen source differs: {name}/{rel}")
    # The scientific manifest is authoritative, including names with no suffix
    # and alternative license spellings. Never filter this closure by extension.
    paths = {f"{source_dir}/{rel}": bundle / source_dir / rel
             for rel in source["files"]}
    paths.update({f: bundle / f for f in TOP_LEVEL if (bundle / f).is_file()})
    # Manifest omissions (root markers, licenses, contracts, unit tests) are
    # preserved as well, without changing the original scientific closure.
    for p in (bundle / source_dir).rglob("*"):
        if (p.is_file() and not any(x in p.parts for x in ("__pycache__", ".git"))
                and (p.suffix in (".py", ".json", ".md", ".txt")
                     or p.name in (".keep", "LICENSE", "LICENCE", "NOTICE"))):
            paths[str(p.relative_to(bundle))] = p
    for subdir in ("control", "verification"):
        for p in (bundle / subdir).glob("*"):
            if p.is_file() and p.suffix in (".py", ".sh"):
                paths[str(p.relative_to(bundle))] = p
    # Runtime/qualification receipts record environment provenance. No live
    # monitor pointer, run outputs or raw data are captured here.
    for p in (bundle / "qualification").glob("*.json"):
        paths[str(p.relative_to(bundle))] = p
    for rel in ("control/allocation.json", "verification/native_cpu_5080.json",
                "verification/native_cpu_5090.json", "verification/baseline_binding.json"):
        if (bundle / rel).is_file():
            paths[rel] = bundle / rel
    result = write_capture(HERE / "snapshots" / name, paths, {
        "kind": "historical_source_and_contract_snapshot",
        "original_bundle": str(bundle.relative_to(ROOT)),
        "canonical_contract_sha256": expected,
        "scientific_source_closure_sha256": source["files_sha256"],
        "scientific_source_file_count": len(source["files"]),
        "not_included": ["data", "checkpoints", "predictions", "live_monitor_state"],
        "execution_authority": "None. Historical approvals/permits are evidence, not permission to relaunch.",
    })
    print(name, len(result["files"]), "files preserved")


if __name__ == "__main__":
    for campaign, args in CAMPAIGNS.items():
        capture_campaign(campaign, *args)
