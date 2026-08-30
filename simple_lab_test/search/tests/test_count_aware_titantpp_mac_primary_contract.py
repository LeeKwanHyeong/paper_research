from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import TIME_HEAD_MODE_LEGACY_CLAMPED
from paper.scripts.count_aware_tpp_backbone.constants import (
    VARIANT,
)
from paper.scripts.count_aware_titantpp_mac_contract import (
    MODEL_ROLE_TITANTPP_MAC_PRIMARY,
    TITANTPP_MAC_PRIMARY_BACKBONES,
    TITANTPP_MAC_PAPER_NAME,
    validate_titantpp_mac_primary_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titantpp_mac_primary_v1.json"
)


def test_primary_contract_freezes_name_causality_and_unseen_seed_boundary() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["paper_model_name"] == "Count-aware TitanTPP-MAC"
    assert contract["implementation_backbone"] == "titantpp_titans_mac"
    assert contract["positioning"] == "primary_candidate"
    causal = contract["causal_memory_contract"]
    assert causal["event_order"] == "predict_before_observed_write"
    assert causal["current_prediction_target_write"] is False
    assert causal["cross_series_state_sharing"] is False
    boundary = contract["decision_information_boundary"]
    assert "seed_52_validation" in boundary["not_observed_before_freeze"]
    assert "seed_62_validation" in boundary["not_observed_before_freeze"]
    assert "held_out_test" in boundary["not_observed_before_freeze"]


def test_primary_acceptance_gate_is_frozen_for_four_datasets_and_three_seeds() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    gate = contract["three_seed_acceptance_gate"]

    assert contract["shared_training_contract"]["seeds"] == [42, 52, 62]
    assert len(contract["dataset_contexts"]) == 4
    assert gate["primary"]["macro_dataset_relative_improvement_minimum"] == 0.05
    assert gate["primary"]["dataset_mean_improvement_minimum_count"] == 3
    assert gate["primary"]["seed_pair_improvement_minimum_count"] == 8
    assert gate["compute_guardrail"]["optimized_epoch_cost_ratio_vs_B0_maximum"] == 3.0
    assert gate["integrity_guardrails"]["held_out_test_unused"] is True


def test_primary_model_role_rejects_contract_drift() -> None:
    assert MODEL_ROLE_TITANTPP_MAC_PRIMARY == "titantpp_mac_primary"
    validate_titantpp_mac_primary_contract(
        backbones=TITANTPP_MAC_PRIMARY_BACKBONES,
        quantity_variants=(VARIANT,),
        time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED,
        lambda_tail=0.0,
    )

    with pytest.raises(ValueError, match="direct log-MSE"):
        validate_titantpp_mac_primary_contract(
            backbones=TITANTPP_MAC_PRIMARY_BACKBONES,
            quantity_variants=("tail_shared_log_regression",),
            time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED,
            lambda_tail=0.0,
        )


def test_primary_name_is_separate_from_frozen_b1_factory_identity() -> None:
    _, metadata = build_count_aware_model(
        "titantpp_titans_mac",
        hidden_dim=8,
        train_log_mean=1.0,
        max_seq_len=16,
    )

    assert TITANTPP_MAC_PAPER_NAME == "Count-aware TitanTPP-MAC"
    assert metadata["candidate_name"] == "count_titan_faithful_titans_mac"
    assert metadata["backbone_contract_id"] == "B1"
    assert "paper_model_name" not in metadata
    assert "model_positioning" not in metadata
