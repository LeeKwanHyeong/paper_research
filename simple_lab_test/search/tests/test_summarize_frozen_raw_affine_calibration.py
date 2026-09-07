"""Tests for staged raw-affine gate summaries."""

from __future__ import annotations

import json
import hashlib

import pytest

from paper.scripts.summarize_frozen_raw_affine_calibration import (
    B_ROWS,
    summarize_b_train,
    summarize_b_validation,
    summarize_final,
)


def result(dataset: str, role: str, phase: str, *, gate: bool = True) -> dict:
    payload = {
        "status": "success",
        "phase": phase,
        "contract_id": "frozen_raw_affine_calibration_v1",
        "contract_sha256": "a" * 64,
        "source_revision": "b" * 40,
        "dataset": dataset,
        "model_role": role,
        "source_checkpoint_file_sha256": f"{len(dataset):064x}",
        "held_out_test_evaluated": False,
        "evaluation_scope": "validation_only",
        "smoke": False,
        "runtime": {"gpu_name": "NVIDIA GeForce RTX 5080"},
        "source_model_state_unchanged": True,
        "source_gradients_absent": True,
        "identity_control": {"exact_prediction_identity": True},
        "train_audit": {"gate_passed": gate},
        "source_manifest": None,
        "input_digests": {"data_sha256": "1" * 64, "split_manifest_sha256": "2" * 64},
        "target_population": {
            phase: {
                "target_count": 10,
                "target_identity_sha256": "3" * 64,
                "target_quantity_sha256": "4" * 64,
            }
        },
        "cache": {phase: {"target_count": 10, "full_population_target_count": 10}},
        "prerequisite_decision": None,
        "train_result": None,
        "validation": None,
    }
    if phase == "validation":
        base_rmse = {"B": 6.0, "rmtpp": 6.2, "thp": 6.1}[role]
        base_mae = {"B": 4.0, "rmtpp": 4.2, "thp": 4.1}[role]
        payload["validation"] = {
            "gate_passed": gate,
            "baseline": {
                "mae": base_mae + 0.1,
                "mse": (base_rmse + 0.1) ** 2,
                "rmse": base_rmse + 0.1,
                "bias": -0.2,
                "log1p_mse": 0.2,
                "time_nll": 1.0,
            },
            "calibrated": {
                "mae": base_mae,
                "mse": base_rmse**2,
                "rmse": base_rmse,
                "bias": -0.1,
                "log1p_mse": 0.19,
                "time_nll": 1.0,
            },
            "calibration_activated_by_train_rule": gate,
            "deployed_calibration_parameters": {"slope": 1.1, "intercept": 0.5},
            "relative_changes": {"rmse": -0.01, "time_nll": 0.0},
            "time_nll_identity": {"exact": True, "distinct_array_storage": True},
            "paired_series_bootstrap": {"seed": 20260907, "replicates": 500},
        }
    return payload


def write_rows(tmp_path, rows):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source_manifest = tmp_path / "source_manifest.json"
    source_manifest.write_text("source manifest")
    source_manifest_sha = hashlib.sha256(source_manifest.read_bytes()).hexdigest()
    paths = []
    for index, payload in enumerate(rows):
        phase = payload["phase"]
        cache_path = tmp_path / f"cache_{index}.npz"
        cache_path.write_bytes(f"cache {index}".encode())
        payload["cache"][phase].update(
            file=cache_path.name,
            file_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        )
        payload["source_manifest"] = {
            "file": str(source_manifest),
            "file_sha256": source_manifest_sha,
            "all_file_hashes_verified": True,
        }
        if phase == "validation":
            train_result = tmp_path / f"train_result_{index}.json"
            train_result.write_text("train result")
            payload["train_result"] = {
                "file": str(train_result),
                "file_sha256": hashlib.sha256(train_result.read_bytes()).hexdigest(),
            }
            prerequisite = tmp_path / f"prerequisite_{index}.json"
            prerequisite.write_text("prerequisite")
            payload["prerequisite_decision"] = {
                "file": str(prerequisite),
                "file_sha256": hashlib.sha256(prerequisite.read_bytes()).hexdigest(),
                "stage": (
                    "B_three_dataset_train_gate"
                    if payload["model_role"] == "B"
                    else "B_three_dataset_validation_gate"
                ),
            }
        path = tmp_path / f"result_{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    return paths


def b_rows(phase: str, *, failed_dataset: str | None = None):
    return [
        result(dataset, role, phase, gate=dataset != failed_dataset)
        for dataset, role in sorted(B_ROWS)
    ]


def test_b_train_failure_is_a_valid_terminal_decision(tmp_path):
    summary = summarize_b_train(
        write_rows(tmp_path, b_rows("train", failed_dataset="yellow_trip_hourly"))
    )
    assert not summary["accepted"]
    assert summary["decision"] == "rejected_stop_before_validation"


def test_b_validation_pass_unlocks_fairness_only(tmp_path):
    summary = summarize_b_validation(write_rows(tmp_path, b_rows("validation")))
    assert summary["accepted"]
    assert summary["decision"] == "accepted_for_instacart_fairness"
    assert len(summary["result_identities"]) == 3


def test_final_fairness_requires_B_to_beat_both_comparators(tmp_path):
    rows = b_rows("validation") + [
        result("insta_market_basket", "rmtpp", "validation"),
        result("insta_market_basket", "thp", "validation"),
    ]
    summary = summarize_final(write_rows(tmp_path, rows))
    assert summary["accepted"]
    assert all(summary["gates"]["instacart_fairness_checks"].values())


def test_stage_rejects_smoke_or_wrong_row_scope(tmp_path):
    rows = b_rows("train")
    rows[0]["smoke"] = True
    with pytest.raises(ValueError, match="Smoke"):
        summarize_b_train(write_rows(tmp_path, rows))

    with pytest.raises(ValueError, match="scope"):
        summarize_b_train(write_rows(tmp_path / "missing", b_rows("train")[:2]))
