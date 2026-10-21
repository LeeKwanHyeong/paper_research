"""Immutable local comparison after the evaluation-only MAC recovery collection.

Only stored, verified endpoints are aggregated.  This module has no remote or
training entry point and never changes a collected artifact or an old report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from paper.scripts import collect_titans_mac_accuracy9_recovery as collector
from paper.scripts import compare_titans_mac_accuracy9 as comparison

SCHEMA = "titans_mac_recovery_shared_three_seed_comparison_v1"
CONTRACT_SHA = "191d2534245ab6c02efff817fe9e21bb43e6589415b1839e22f1a7dec40797c2"
ORIGINAL_RAW_SHA = "05b646251dbf831f4a91034111fc7a7d8f66d659a681b789c59825bbd5cfc16f"
ORIGINAL_CANONICAL_SHA = "7e1cc94dcf913f8451cc8b9a9d39b6db4baf441ab0cf32aed25e85953f6137d0"
JSON_NAME, MARKDOWN_NAME = "MAC_SHARED_THREE_SEED.json", "MAC_SHARED_THREE_SEED.md"
OUTPUT_NAMES = {"ATTEMPT.json", JSON_NAME, MARKDOWN_NAME}
read, sha, canonical, require, write = collector.read, collector.sha, collector.canonical, collector.require, collector.write


def _result(contract, collection, *, created_at):
    evidence = collector.retrieval_dir() / "evidence"
    original_path = collector.mapped(evidence / "original", contract["immutable_originals"]["root"],
                                     contract["scientific_contract"]["path"])
    require(sha(original_path) == ORIGINAL_RAW_SHA == contract["scientific_contract"]["raw_sha256"],
            "Frozen original scientific contract bytes changed")
    original = read(original_path)
    require(canonical(original) == ORIGINAL_CANONICAL_SHA == contract["scientific_contract"]["canonical_sha256"],
            "Frozen original scientific contract identity changed")
    mac_records = collector.comparison_records(evidence, contract)
    table, reproduction, shared_proof = comparison.verify_shared_evidence(
        comparison.PAPER, expected_manifest_sha=comparison.PAPER_MANIFEST_SHA)
    comparison.verify_common_conditions(original, reproduction)
    populations = {(d, s): comparison.expected_population(reproduction, d, s)
                   for d in comparison.DATASETS for s in comparison.SPLITS}
    source_files = {str(path): sha(path) for path in (
        Path(__file__), Path(collector.__file__), Path(comparison.__file__),
        collector.manager.BUNDLE / "recovery_contract.json",
        collector.report_dir() / "COLLECTION_COMPLETE.json",
        collector.retrieval_dir() / "receipt.json", collector.retrieval_dir() / "manifest.json")}
    return {"schema": SCHEMA, "status": "complete", "created_at": created_at,
            "data_variant": "original_uncapped", "models": [comparison.SHARED, comparison.MAC],
            "seeds": list(comparison.SEEDS), "remote_calls": 0, "new_training": 0, "new_inference": 0,
            "held_out_test_evaluated": True,
            "runtime_context": comparison.evaluation_runtime_context(original, reproduction),
            "source_proof": {"recovery_contract_sha256": canonical(contract),
                "original_scientific_contract_raw_sha256": sha(original_path),
                "original_scientific_contract_sha256": canonical(original),
                "mac_archive_sha256": collection["archive_sha256"],
                "mac_manifest_sha256": collection["manifest_sha256"],
                "mac_receipt_sha256": collection["receipt_sha256"],
                "collection_verified_locally": True, "files_sha256": source_files, "shared": shared_proof},
            **comparison.compare_records(table["records"], mac_records, populations)}


def _safe_file(directory, name):
    path = directory / name
    require(path.is_file() and not path.is_symlink(), "Missing or symbolic comparison file: " + name)
    return path


def _verify_outputs(destination, contract, collection):
    require(destination.is_dir() and not destination.is_symlink(), "Comparison directory changed")
    actual = {p.name for p in destination.iterdir()}
    require(actual == OUTPUT_NAMES | {"MANIFEST.json", "COMPLETE.json"},
            "Partial or unbound comparison output; preserve existing attempt")
    manifest_path = _safe_file(destination, "MANIFEST.json")
    manifest = read(manifest_path)
    files = manifest.get("files")
    require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete"
            and isinstance(files, dict) and bool(files) and set(files) == OUTPUT_NAMES,
            "Empty or incomplete comparison inventory")
    for name, digest in files.items():
        require(sha(_safe_file(destination, name)) == digest, "Comparison output bytes changed: " + name)
    result = read(destination / JSON_NAME)
    stamp = result["created_at"]
    require(isinstance(stamp, str) and datetime.fromisoformat(stamp).tzinfo is not None,
            "Comparison creation time changed")
    require(read(destination / "ATTEMPT.json") == {"schema": SCHEMA, "at": stamp,
        "contract_sha256": canonical(contract), "automatic_retry": False}, "Comparison attempt changed")
    expected = _result(contract, collection, created_at=stamp)
    require(result == expected, "Pure comparison recomputation differs from cached result")
    require((destination / MARKDOWN_NAME).read_text(encoding="utf-8") == comparison.render_markdown(expected),
            "Pure comparison Markdown recomputation differs")
    complete = read(_safe_file(destination, "COMPLETE.json"))
    require(complete == {"schema": SCHEMA, "status": "comparison_complete", "contract_sha256": canonical(contract),
        "manifest_sha256": sha(manifest_path), "file_count": len(files),
        "source_proof_sha256": canonical(expected["source_proof"])}, "Comparison completion binding changed")
    return {"status": "comparison_complete_verified", "report": str(destination), "cache_verified": True,
            "seed_metric_comparison_count": 108, "remote_calls": 0, "outputs_created": False}


def run(expected, *, check_only=False):
    require(expected == CONTRACT_SHA, "Pinned evaluation-only recovery contract required")
    with collector.controller_context(True):
        contract = collector.manager.verify_local(expected)
        collection = collector.advance(expected, check_local=True)
        destination = collector.manager.REPORT / "comparison_v1"
        if collection.get("status") != "collection_complete":
            require(not destination.exists(), "Partial comparison exists before verified collection")
            require(check_only, "MAC recovery is not collected; no comparison output is allowed")
            return {"status": "waiting_for_verified_MAC_collection", "remote_calls": 0, "outputs_created": False}
        require(collection.get("cache_verified") is True and collection.get("remote_calls") == 0,
                "Comparison requires a locally verified collection cache")
        if destination.exists():
            return _verify_outputs(destination, contract, collection)
        stamp = datetime.now(timezone.utc).isoformat()
        result = _result(contract, collection, created_at=stamp)
        if check_only:
            return {"status": "comparison_ready_outputs_not_written", "remote_calls": 0, "outputs_created": False,
                    "seed_metric_comparison_count": 108}
        destination.mkdir(parents=True, exist_ok=False)
        write(destination / "ATTEMPT.json", {"schema": SCHEMA, "at": stamp,
            "contract_sha256": canonical(contract), "automatic_retry": False}, exclusive=True)
        write(destination / JSON_NAME, result, exclusive=True)
        with (destination / MARKDOWN_NAME).open("x", encoding="utf-8") as stream:
            stream.write(comparison.render_markdown(result))
        write(destination / "MANIFEST.json", {"schema": SCHEMA, "status": "complete",
            "files": {name: sha(destination / name) for name in sorted(OUTPUT_NAMES)}}, exclusive=True)
        write(destination / "COMPLETE.json", {"schema": SCHEMA, "status": "comparison_complete",
            "contract_sha256": canonical(contract), "manifest_sha256": sha(destination / "MANIFEST.json"),
            "file_count": len(OUTPUT_NAMES), "source_proof_sha256": canonical(result["source_proof"])}, exclusive=True)
        return {**_verify_outputs(destination, contract, collection), "outputs_created": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.contract_sha, check_only=args.check_only), ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
