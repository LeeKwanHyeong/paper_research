"""Frozen proposal rejects scope drift before any native execution."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from paper.scripts import prepare_state_transport_execution as prep


@pytest.fixture
def contract(monkeypatch):
    # The repository contract references are real; only verification receipt
    # results and manifest collection are fabricated to keep this read-only.
    read = prep.read
    def read_fixture(path):
        if Path(path).name == "synthetic_cost.json":
            return {"status": "complete"}
        if Path(path).name == "synthetic_implementation.json":
            return {"verification": {"failed": 0, "passed": 1},
                    "native_execution_verification": {"status": "passed_synthetic_cpu"}}
        return read(path)
    monkeypatch.setattr(prep, "read", read_fixture)
    monkeypatch.setattr(prep, "sha_file", lambda _: "1" * 64)
    monkeypatch.setattr(prep, "source_manifest", lambda *a, **k: {"synthetic.py": "1" * 64})
    return prep.build_contract(prep.ROOT/"synthetic_cost.json", prep.ROOT/"synthetic_implementation.json")


def test_fixed_budget_and_train_time_statistics(contract):
    prep.validate_contract(contract, verify_source=False)
    assert contract["total_optimizer_steps"] == 1215720
    by_name = {d["dataset_id"]: d for d in contract["datasets"]}
    assert by_name["yellow_trip_hourly"]["model"]["time_scale"] == 1.0
    assert by_name["intermittent_frozen_5000"]["model"]["time_scale"] == 3.0
    assert contract["approved"] is False
    assert contract["qualification"]["executed"] is False
    assert contract["launch"]["estimated_completion_hours"] is None


@pytest.mark.parametrize("mutation", ["budget", "host", "head", "scale", "batch", "optimizer", "held_out", "resume", "gate", "extra_dataset", "cost_gate", "time_identity", "runtime"])
def test_rejects_contract_drift(contract, mutation):
    changed = deepcopy(contract)
    if mutation == "budget": changed["limits"]["total_wall_seconds"] += 1
    elif mutation == "host": changed["hosts"]["5080"]["assigned_datasets"] = ["intermittent_frozen_5000"]
    elif mutation == "head": changed["datasets"][0]["model"]["time_head_mode"] = "legacy_clamped_rmtpp"
    elif mutation == "scale": changed["datasets"][0]["model"]["time_scale"] = 3.0
    elif mutation == "batch": changed["datasets"][0]["loader"]["batch_size"] = 64
    elif mutation == "optimizer": changed["datasets"][0]["optimizer"]["lr"] = 0.01
    elif mutation == "held_out": changed["policy"]["held_out"] = True
    elif mutation == "resume": changed["policy"]["resume"] = True
    elif mutation == "gate": changed["acceptance"]["mae_ratio_max"] = 1.2
    elif mutation == "extra_dataset": changed["datasets"].append(deepcopy(changed["datasets"][0]))
    elif mutation == "cost_gate": changed["cost_gates"]["median_cuda_step_ratio_max"] = 9.0
    elif mutation == "time_identity": changed["datasets"][0]["time_statistics"]["expected_train_target_dt_sha256"] = "0"*64
    elif mutation == "runtime": changed["hosts"]["5090"]["runtime_expected"]["torch"] = "unexpected"
    with pytest.raises(ValueError):
        prep.validate_contract(changed, verify_source=False)


def test_modified_source_cannot_be_frozen(contract, monkeypatch):
    monkeypatch.setattr(prep, "source_manifest", lambda *a, **k: {"unlisted.py": "2" * 64})
    with pytest.raises(ValueError, match="source"):
        prep.validate_contract(contract)


def test_design_identity_is_bound_even_without_runtime_source_scan(contract, monkeypatch):
    monkeypatch.setattr(prep, "sha_file", lambda _: "2" * 64)
    with pytest.raises(ValueError, match="Design"):
        prep.validate_contract(contract, verify_source=False)


def test_source_closure_cannot_be_omitted_or_rehashed(contract):
    contract["source"]["files"] = {}
    contract["source"]["files_sha256"] = prep.sha_json({})
    with pytest.raises(ValueError, match="closure"):
        prep.validate_contract(contract)
