from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PARENT_PATH = (
    PROJECT_ROOT / "paper/contracts/count_aware_titantpp_mac_primary_v1.json"
)
AMENDMENT_PATH = (
    PROJECT_ROOT
    / "paper/contracts/count_aware_titantpp_mac_primary_v1_amendment_1.json"
)


def load_contracts() -> tuple[dict, dict]:
    parent = json.loads(PARENT_PATH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    return parent, amendment


def test_amendment_is_bound_to_immutable_parent_before_unseen_results() -> None:
    _, amendment = load_contracts()

    assert hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest() == amendment[
        "parent_contract_sha256"
    ]
    boundary = amendment["decision_information_boundary"]
    assert boundary["not_observed_before_amendment"] == [
        "seed_52_validation",
        "seed_62_validation",
        "instacart_titantpp_mac_validation",
        "held_out_test",
    ]


def test_forward_name_and_compute_effect_are_amended_without_threshold_drift() -> None:
    parent, amendment = load_contracts()

    terminology = amendment["terminology"]
    assert terminology["short_model_name"] == "TitanTPP-MAC"
    assert terminology["forward_facing_role_labels_are_forbidden"] is True
    compute = amendment["compute_amendment"]
    assert compute["reference_target_ratio_vs_hard_lmm"] == 3.0
    assert compute["validation_expansion_blocked_by_compute_target"] is False
    assert compute["model_selection_blocked_by_compute_target"] is False
    assert compute["target_failure_must_be_reported_as_limitation"] is True
    assert compute["performance_or_integrity_thresholds_changed"] is False

    unchanged = amendment["unchanged_parent_rules"]
    for section in (
        "three_seed_acceptance_gate.primary",
        "three_seed_acceptance_gate.aggregate_guardrails",
        "three_seed_acceptance_gate.per_dataset_guardrails",
        "three_seed_acceptance_gate.integrity_guardrails",
    ):
        assert section in unchanged
    assert parent["three_seed_acceptance_gate"]["primary"][
        "macro_dataset_relative_improvement_minimum"
    ] == 0.05
    assert parent["three_seed_acceptance_gate"]["integrity_guardrails"][
        "held_out_test_unused"
    ] is True


def test_official_grid_uses_frozen_training_revision_for_nine_missing_runs() -> None:
    _, amendment = load_contracts()
    execution = amendment["official_validation_execution"]

    assert execution["training_source_revision"] == (
        "08e59880cd61cbd27cec40aa04636452b87bebfc"
    )
    assert execution["implementation_backbone"] == "titantpp_titans_mac"
    assert execution["semantic_optimization_adapter_used"] is False
    assert execution["held_out_test"] == "locked"
    assert execution["fresh_or_missing_run_count"] == 9
    assert sum(
        len(seeds) for seeds in execution["fresh_or_missing_runs"].values()
    ) == 9
