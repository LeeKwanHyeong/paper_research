"""Synthetic recovery comparison; no stored scientific results or remote access."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from paper.scripts import compare_titans_mac_accuracy9_recovery as wrapper
from paper.scripts import manage_titans_mac_accuracy9_completion_bridge as bridge


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    collector, cmp = wrapper.collector, wrapper.comparison
    report, bundle = tmp_path / "report", tmp_path / "bundle"
    retrieval = bundle / "collection_v1/retrieval_v1"
    evidence = retrieval / "evidence"
    runtime = {"python": "3.12.13", "torch": "2.11.0+cu130", "cuda": "13.0", "cudnn": 92000}
    original = {"common_condition": {"loss": "fixed"}, "training": {"max_epochs": 300},
                "hosts": {"5090": {"runtime_expected": runtime}}, "datasets": [], "input_files": {}}
    reproduction = {"data_variant": "original_uncapped", "quantity_capping_applied": False,
                    "training_condition": original["common_condition"], "budget": original["training"],
                    "shared_training_runtime": runtime, "shared_test_runtime": {**runtime, "cudnn": 91900},
                    "datasets": {}}
    shared, mac = [], []
    for dataset in cmp.DATASETS:
        populations = {split: {"target_count": 20, "target_identity_sha256": dataset + split,
                              "target_quantity_sha256": "b" * 64} for split in cmp.SPLITS}
        identity = {"data": {"sha256": "d" * 64}, "split_manifest": {"sha256": "e" * 64},
                    "populations": {"validation": populations["validation"]}}
        original["datasets"].append({"dataset_id": dataset, "loader": {"kind": "frozen"},
            "model": {"head": "fixed"}, "quantity_boundaries_all_train_rows": [100.],
            "full_data_identity": {"populations": {"test": populations["test"]}}})
        reproduction["datasets"][dataset] = {"loader": {"kind": "frozen"}, "model_head": {"head": "fixed"},
            "tail_threshold": 100., "training_validation_identity": identity,
            "full_data": {"sha256": "f" * 64}, "test_population": populations["test"]}
        for suffix, digest in (("train_validation.parquet", "d" * 64), ("split_manifest.json", "e" * 64)):
            original["input_files"][f"inputs/data/{dataset}/{suffix}"] = digest
        original["input_files"][f"inputs/full/{dataset}.parquet"] = "f" * 64
        for split in cmp.SPLITS:
            for index, seed in enumerate(cmp.SEEDS):
                def metrics(value):
                    return {"overall": {"count": 20, "qty_rmse": value, "qty_mae": value / 2, "time_nll": value / 10},
                        "tail": {"count": 2, "qty_rmse": value * 2, "qty_mae": value, "time_nll": value / 5},
                        "tail_threshold": 100., "tail_rule": "q > frozen_Train_large_quantity_threshold",
                        "time_metric": "recorded_positive_integer_time_nll"}
                shared.append({"dataset": dataset, "split": split, "seed": seed, "model": cmp.SHARED,
                    "selected_epoch": index + 1, "metrics": metrics(8. if seed == 52 else 1. + index)})
                mac.append({"dataset": dataset, "split": split, "seed": seed, "model": cmp.MAC,
                    "selected_epoch": index + 2, "metrics": metrics(2. + index),
                    "target_identity": populations[split]})
    origin = evidence / "original/execution_contract.json"
    put(origin, original)
    contract = {"continuation_only": True, "immutable_originals": {"root": "/immutable"},
        "scientific_contract": {"path": "/immutable/execution_contract.json", "raw_sha256": wrapper.sha(origin),
                                "canonical_sha256": wrapper.canonical(original)}}
    expected = wrapper.canonical(contract)
    monkeypatch.setattr(wrapper, "CONTRACT_SHA", expected)
    monkeypatch.setattr(wrapper, "ORIGINAL_RAW_SHA", wrapper.sha(origin))
    monkeypatch.setattr(wrapper, "ORIGINAL_CANONICAL_SHA", wrapper.canonical(original))
    monkeypatch.setattr(bridge, "BUNDLE", bundle)
    monkeypatch.setattr(bridge, "REPORT", report)
    put(bundle / "recovery_contract.json", contract)
    verified_calls = []

    def verify_local(value):
        assert value == expected
        verified_calls.append(value)
        assert wrapper.canonical(wrapper.read(bundle / "recovery_contract.json")) == expected
        return contract

    monkeypatch.setattr(bridge, "verify_local", verify_local)
    for name, value in (("receipt.json", {"status": "collection_complete"}), ("manifest.json", {"files": {"bound": "x"}})):
        put(retrieval / name, value)
    (retrieval / "archive.tar.gz").write_bytes(b"synthetic immutable archive")
    collection = {"status": "collection_complete", "contract_sha256": expected,
        "cache_verified": True, "remote_calls": 0,
        "archive_sha256": wrapper.sha(retrieval / "archive.tar.gz"),
        "manifest_sha256": wrapper.sha(retrieval / "manifest.json"),
        "receipt_sha256": wrapper.sha(retrieval / "receipt.json")}
    marker = report / "collection_v1/COLLECTION_COMPLETE.json"
    put(marker, collection)
    initial_marker_sha = wrapper.sha(marker)
    advance_calls = []

    def advance(value, *, check_local=False):
        assert value == expected and check_local is True and collector.manager is bridge
        advance_calls.append(check_local)
        assert wrapper.sha(marker) == initial_marker_sha, "Synthetic collection marker changed"
        for name, key in (("archive.tar.gz", "archive_sha256"), ("manifest.json", "manifest_sha256"), ("receipt.json", "receipt_sha256")):
            assert wrapper.sha(retrieval / name) == collection[key], "Synthetic verified cache changed"
        return collection

    monkeypatch.setattr(collector, "advance", advance)

    def records(path, c):
        assert path == evidence and c == contract
        return deepcopy(mac)

    monkeypatch.setattr(collector, "comparison_records", records)
    monkeypatch.setattr(collector, "transport", lambda *a, **k: pytest.fail("Comparison must never connect remotely"))
    monkeypatch.setattr(bridge, "transport", lambda *a, **k: pytest.fail("Comparison must never connect remotely"))
    directory = tmp_path / "shared"
    binding = tmp_path / "shared_original.json"
    put(binding, {"preserved": True})
    table = {"schema": "shared_final_three_seed_paper_evidence_v1", "status": "verified_existing_results_reaggregated",
        "intermittent_final_best_verified": True, "seeds": list(cmp.SEEDS), "data_variant": "original_uncapped", "records": shared}
    verification = {"status": "PASS", "remote_calls": 0, "new_training": 0, "new_inference": 0,
        "file_bindings": {str(binding): wrapper.sha(binding)}, "final_receipts": {str(binding): wrapper.sha(binding)}}
    for name, value in (("RESULT_TABLES.json", table), ("SOURCE_VERIFICATION.json", verification), ("REPRODUCIBILITY.json", reproduction)):
        put(directory / name, value)
    put(directory / "MANIFEST.json", {"files": {p.name: wrapper.sha(p) for p in directory.iterdir()}})
    monkeypatch.setattr(cmp, "ROOT", tmp_path)
    monkeypatch.setattr(cmp, "PAPER", directory)
    monkeypatch.setattr(cmp, "PAPER_MANIFEST_SHA", wrapper.sha(directory / "MANIFEST.json"))
    return {"expected": expected, "report": report, "bundle": bundle, "retrieval": retrieval,
        "destination": report / "comparison_v1", "marker": marker, "origin": origin,
        "binding": binding, "original": original, "contract": contract,
        "advance_calls": advance_calls, "verified_calls": verified_calls}


def test_check_only_is_local_and_writes_no_comparison(fixture):
    result = wrapper.run(fixture["expected"], check_only=True)
    assert result == {"status": "comparison_ready_outputs_not_written", "remote_calls": 0,
                      "outputs_created": False, "seed_metric_comparison_count": 108}
    assert not fixture["destination"].exists()
    assert fixture["advance_calls"] == [True] and fixture["verified_calls"]


def test_create_then_cache_recompute_preserves_bytes_adverse_seeds_and_runtime(fixture):
    result = wrapper.run(fixture["expected"])
    destination = fixture["destination"]
    assert result["outputs_created"] is True and result["remote_calls"] == 0
    output = wrapper.read(destination / wrapper.JSON_NAME)
    assert output["aggregate_comparison_count"] == 36 and output["seed_metric_comparison_count"] == 108
    assert output["aggregates"][0]["shared_worse_seeds"] == [52]
    assert output["runtime_context"]["all_accuracy_endpoints_same_gpu_runtime"] is False
    assert "5080/cuDNN91900" in (destination / wrapper.MARKDOWN_NAME).read_text()
    before = {p.name: p.read_bytes() for p in destination.iterdir()}
    assert wrapper.run(fixture["expected"], check_only=True)["cache_verified"] is True
    assert wrapper.run(fixture["expected"])["outputs_created"] is False
    assert before == {p.name: p.read_bytes() for p in destination.iterdir()}


def test_uncollected_check_only_never_opens_shared_or_original(fixture, monkeypatch):
    monkeypatch.setattr(wrapper.collector, "advance", lambda *a, **k: {"status": "not_collected", "remote_calls": 0})
    monkeypatch.setattr(wrapper.comparison, "verify_shared_evidence", lambda *a, **k: pytest.fail("No endpoints until collection"))
    assert wrapper.run(fixture["expected"], check_only=True)["status"] == "waiting_for_verified_MAC_collection"
    assert not fixture["destination"].exists()
    with pytest.raises(ValueError, match="not collected"):
        wrapper.run(fixture["expected"])


@pytest.mark.parametrize("check_only", [False, True])
def test_partial_attempt_is_preserved_and_never_overwritten(fixture, check_only):
    path = fixture["destination"] / "ATTEMPT.json"
    put(path, {"partial": True})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Partial or unbound"):
        wrapper.run(fixture["expected"], check_only=check_only)
    assert path.read_bytes() == before and set(p.name for p in path.parent.iterdir()) == {path.name}


@pytest.mark.parametrize("name", [wrapper.JSON_NAME, wrapper.MARKDOWN_NAME, "MANIFEST.json", "COMPLETE.json", "ATTEMPT.json"])
def test_cache_missing_or_changed_files_are_rejected(fixture, name):
    wrapper.run(fixture["expected"])
    target = fixture["destination"] / name
    target.unlink()
    with pytest.raises(ValueError, match="Partial or unbound"):
        wrapper.run(fixture["expected"], check_only=True)


def rebind(fixture, name, value):
    destination = fixture["destination"]
    put(destination / name, value)
    manifest = wrapper.read(destination / "MANIFEST.json")
    manifest["files"][name] = wrapper.sha(destination / name)
    put(destination / "MANIFEST.json", manifest)
    complete = wrapper.read(destination / "COMPLETE.json")
    complete["manifest_sha256"] = wrapper.sha(destination / "MANIFEST.json")
    put(destination / "COMPLETE.json", complete)


def test_coherently_rebound_result_still_requires_pure_recomputation(fixture):
    wrapper.run(fixture["expected"])
    output = wrapper.read(fixture["destination"] / wrapper.JSON_NAME)
    output["seed_differences"][0]["mac"] += 1.
    rebind(fixture, wrapper.JSON_NAME, output)
    with pytest.raises(ValueError, match="recomputation differs"):
        wrapper.run(fixture["expected"], check_only=True)


def test_rebound_markdown_still_requires_pure_rendering(fixture):
    wrapper.run(fixture["expected"])
    destination = fixture["destination"]
    (destination / wrapper.MARKDOWN_NAME).write_text("changed report", encoding="utf-8")
    manifest = wrapper.read(destination / "MANIFEST.json")
    manifest["files"][wrapper.MARKDOWN_NAME] = wrapper.sha(destination / wrapper.MARKDOWN_NAME)
    put(destination / "MANIFEST.json", manifest)
    complete = wrapper.read(destination / "COMPLETE.json")
    complete["manifest_sha256"] = wrapper.sha(destination / "MANIFEST.json")
    put(destination / "COMPLETE.json", complete)
    with pytest.raises(ValueError, match="Markdown recomputation differs"):
        wrapper.run(fixture["expected"], check_only=True)


def test_changed_source_proof_cannot_be_hidden_by_a_new_manifest(fixture):
    wrapper.run(fixture["expected"])
    output = wrapper.read(fixture["destination"] / wrapper.JSON_NAME)
    output["source_proof"]["mac_archive_sha256"] = "0" * 64
    rebind(fixture, wrapper.JSON_NAME, output)
    with pytest.raises(ValueError, match="recomputation differs"):
        wrapper.run(fixture["expected"], check_only=True)


def test_empty_inventory_cannot_be_complete(fixture):
    wrapper.run(fixture["expected"])
    manifest = wrapper.read(fixture["destination"] / "MANIFEST.json")
    manifest["files"] = {}
    put(fixture["destination"] / "MANIFEST.json", manifest)
    with pytest.raises(ValueError, match="Empty or incomplete"):
        wrapper.run(fixture["expected"], check_only=True)


@pytest.mark.parametrize("target", ["marker", "origin", "binding"])
def test_collected_cache_original_and_shared_bindings_are_rechecked(fixture, target):
    wrapper.run(fixture["expected"])
    put(fixture[target], {"changed": True})
    with pytest.raises((AssertionError, ValueError)):
        wrapper.run(fixture["expected"], check_only=True)


def test_condition_mismatch_fails_before_any_output(fixture, monkeypatch):
    changed = deepcopy(fixture["original"])
    changed["training"]["max_epochs"] = 299
    put(fixture["origin"], changed)
    fixture["contract"]["scientific_contract"]["raw_sha256"] = wrapper.sha(fixture["origin"])
    fixture["contract"]["scientific_contract"]["canonical_sha256"] = wrapper.canonical(changed)
    new_expected = wrapper.canonical(fixture["contract"])
    put(fixture["bundle"] / "recovery_contract.json", fixture["contract"])
    monkeypatch.setattr(wrapper, "ORIGINAL_RAW_SHA", wrapper.sha(fixture["origin"]))
    monkeypatch.setattr(wrapper, "ORIGINAL_CANONICAL_SHA", wrapper.canonical(changed))
    monkeypatch.setattr(wrapper, "CONTRACT_SHA", new_expected)
    monkeypatch.setattr(bridge, "verify_local", lambda expected: fixture["contract"])
    monkeypatch.setattr(wrapper.collector, "advance", lambda *a, **k: {
        "status": "collection_complete", "cache_verified": True, "remote_calls": 0})
    with pytest.raises(ValueError, match="selection budget differs"):
        wrapper.run(new_expected, check_only=True)
    assert not fixture["destination"].exists()


def test_unpinned_contract_fails_before_collection(fixture):
    with pytest.raises(ValueError, match="Pinned"):
        wrapper.run("0" * 64, check_only=True)
    assert not fixture["advance_calls"]
