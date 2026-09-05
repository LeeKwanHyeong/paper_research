"""Frozen contract checks for the transition-error Hard-LMM study."""

import hashlib
import json
from pathlib import Path

import torch

from simple_lab_test.search.common.runner import canonical_state_dict_sha256


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "paper/contracts/hard_lmm_transition_error_v1.json"


def _read(path: Path):
    return json.loads(path.read_text())


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_signal_scope_gate_and_deferred_follow_up_are_frozen():
    contract = _read(CONTRACT)
    assert contract["contract_id"] == "hard_lmm_transition_error_v1"
    assert [row["dataset"] for row in contract["datasets"]] == [
        "intermittent_v2", "yellow_trip_hourly", "insta_market_basket"
    ]
    scope = contract["diagnostic_scope"]
    assert scope["allowed_split"] == "train_only"
    assert scope["maximum_targets_per_dataset"] == 8192
    assert scope["minimum_targets_per_fold"] == 512
    assert scope["minimum_unique_series_per_fold"] == 30
    assert not scope["validation_rows_materialized"]
    assert not scope["held_out_test_evaluated"]
    gate = contract["signal_gate"]
    assert gate["minimum_pooled_relative_improvement_each_comparison"] == .01
    assert gate["paired_series_bootstrap"]["repeats"] == 10000
    assert gate["paired_series_bootstrap"]["lower_quantile"] == .05 / 6
    assert gate["failure_action"] == "do_not_implement_or_train_the_candidate"
    assert contract["authorization"]["rtx5090_cuda_e1_seed42_e300"]
    assert not contract["authorization"]["held_out_test"]
    assert contract["deferred_follow_up"]["name"] == (
        "quantile-adaptive loss and checkpoint alignment"
    )
    assert contract["deferred_follow_up"]["must_be_repeated_in_each_progress_update"]


def test_registry_and_local_intermittent_override_are_exact_state_equivalents():
    contract = _read(CONTRACT)
    registry_path = ROOT / contract["reference"]["registry"]
    assert _sha(registry_path) == contract["reference"]["registry_sha256"]
    registry = _read(registry_path)
    expected = next(row for row in registry["datasets"] if row["dataset"] == "intermittent_v2")
    override = next(row for row in contract["datasets"] if row["dataset"] == "intermittent_v2")[
        "local_checkpoint_override"
    ]
    artifact = ROOT / override["artifact_dir"]
    launch = artifact / "launch_contract.json"
    checkpoint = artifact / (
        "runs/titantpp/count_only_log_regression/seed_42/"
        "best_val_joint_objective_model.pt"
    )
    summary = checkpoint.with_name("summary.json")
    assert _sha(launch) == override["launch_contract_sha256"]
    assert _sha(checkpoint) == override["checkpoint_file_sha256"]
    assert _sha(summary) == override["summary_sha256"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    observed = canonical_state_dict_sha256(payload["model_state_dict"])
    assert observed == payload["model_state_sha256"]
    assert observed == override["checkpoint_state_sha256"]
    assert observed == expected["checkpoint_state_sha256"]
    assert payload["evaluation_scope"] == "validation_only"
    assert payload["held_out_test_evaluated"] is False
