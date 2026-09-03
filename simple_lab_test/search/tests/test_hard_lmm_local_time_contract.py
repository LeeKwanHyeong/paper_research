"""Validate the prospective document contract, not an implemented model."""

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_local_time_v1.json"


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(), parse_constant=reject_constant, object_pairs_hook=unique_keys
    )


@pytest.fixture
def contract():
    return read_json(CONTRACT_PATH)


def test_frozen_document_is_not_model_or_launch_authorization(contract):
    assert contract["contract_id"] == "hard_lmm_local_time_v1"
    assert contract["status"] == "contract_frozen_implementation_not_started"
    assert contract["authorization"] == {
        "contract_and_document_validation": True,
        "model_implementation": False,
        "server_sync_or_smoke": False,
        "training": False,
        "scheduler": False,
        "push": False,
    }
    candidate = contract["candidate"]
    assert candidate["reference_backbone"] == "titantpp"
    assert candidate["backbone_id_reserved"] == "titantpp_hard_memory_local_time"
    assert candidate["model_role_reserved"] != "t0_common_control"
    assert not candidate["registered_in_factory"]
    assert not candidate["adopted_primary_model"]
    assert contract["future_execution"]["run_commands"] == []
    assert contract["future_execution"]["implementation_source_revision"] is None


def test_exactly_time_route_changes_with_persistent_memory_preserved(contract):
    route = contract["routing"]
    before, after = route["reference_states"], route["candidate_states"]
    assert before == {"time": "h+r", "quantity": "h+r"}
    assert after == {"time": "h", "quantity": "h+r"}
    assert [key for key in before if before[key] != after[key]] == ["time"]
    assert route["encode_task_states_order"] == ["time", "quantity"]
    assert route["encode_returns"] == "time"
    assert route["base_encoder_calls_per_forward"] == 1
    assert route["mask_both_states"] and route["persistent_tokens_in_time_path"]
    assert not route["detach_any_shared_or_quantity_state"]
    assert route["time_loss_gradient_to_prototype_bank"] == "none_or_exact_zero"
    assert route["quantity_loss_gradient_to_prototype_bank"] == "preserved"
    assert "not after joint optimizer steps" in route["quantity_identity_scope"]


def test_original_architecture_not_weighted_or_no_memory_variant(contract):
    architecture = contract["unchanged_architecture"]
    assert contract["candidate"]["additional_parameters"] == 0
    for key, expected in {
        "input_dim": 2, "hidden_dim": 64, "layers": 2, "attention_heads": 4,
        "ff_dim": 128, "dropout": 0.1, "persistent_tokens_per_attention_block": 16,
        "prototype_count": 64, "topk": 4, "contextual_memory_size": 0,
        "aggregation": "arithmetic_mean", "normalization": "original_pre_norm",
        "time_head": "legacy_clamped_rmtpp",
    }.items():
        assert architecture[key] == expected
    assert not architecture["online_memory_updates"]
    assert not architecture["new_gate_projection_adapter_calibration_or_temperature"]


def test_registry_and_diagnostic_evidence_are_hash_pinned(contract):
    for path, expected in (
        (contract["baseline"]["registry"], contract["baseline"]["registry_sha256"]),
        (contract["evidence"]["audit"], contract["evidence"]["audit_sha256"]),
    ):
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    baseline = contract["baseline"]
    assert baseline["reuse_original_results"]
    assert not baseline["retrain_benchmarks"]
    assert not baseline["reuse_weighted_candidate_or_mac_as_baseline"]
    assert not baseline["load_reference_weights_for_training"]
    assert baseline["load_reference_weights_for_route_tests"]


def test_dataset_identity_context_and_counts_match_existing_contracts(contract):
    registry = read_json(ROOT / contract["baseline"]["registry"])
    reference = {row["dataset"]: row for row in registry["datasets"]}
    weighted = read_json(ROOT / "paper/contracts/hard_lmm_weighted_static_v1.json")
    rows = contract["datasets"]
    assert len(rows) == len({row["dataset"] for row in rows}) == 4
    assert {row["dataset"] for row in rows} == set(reference)
    for row in rows:
        name = row["dataset"]
        for key in ("contract_dataset", "lookback", "max_seq_len"):
            assert row[key] == reference[name][key]
        assert row["train_targets"] == weighted["train_target_counts"][name]
        assert row["validation_targets"] == weighted["validation_target_counts"][name]
        assert row["parameter_count"] > 0


def test_learning_objective_optimizer_and_selector_are_not_retuned(contract):
    training = contract["prospective_training"]
    assert (training["maximum_epochs"], training["minimum_epochs"], training["patience"]) == (300, 40, 40)
    assert (training["seed"], training["batch_size"], training["learning_rate"]) == (42, 128, 0.001)
    assert training["optimizer"] == {
        "type": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8,
        "weight_decay": 0.01, "amsgrad": False, "parameter_groups": 1,
        "time_head_lr_multiplier": 1.0,
    }
    assert training["gradient_clip"]["max_norm"] == 1.0
    assert training["quantity_objective"] == "direct_mse_on_log1p_quantity"
    assert (training["lambda_log_qty"], training["lambda_tail"]) == (1.0, 0.0)
    assert training["checkpoint_monitor"] == "validation_joint_objective"
    assert training["checkpoint_comparison"] == "strict_less_than; ties keep earliest epoch"
    for flag in ("checkpoint_epoch_zero_eligible", "posthoc_raw_mae_or_rmse_selection",
                 "lr_schedule_or_pcgrad", "amp", "new_compile_policy"):
        assert training[flag] is False
    assert training["time_launch_arguments"]["time_scale"] == 3.0
    assert training["time_launch_arguments"]["time_w_max"] == 10.0 / 3.0
    assert "separate from launch arguments" in training["time_statistics_policy"]


def test_causal_and_held_out_requirements_are_explicit(contract):
    evaluation = contract["data_and_evaluation"]
    assert evaluation["allowed_materialized_splits"] == ["train", "validation"]
    assert evaluation["evaluation_scope"] == "validation_only"
    for flag in ("held_out_test_evaluated", "train_or_validation_subsampling",
                 "target_memory_write", "static_parameters_updated_in_validation"):
        assert evaluation[flag] is False
    assert evaluation["target_quantity_in_encoder"] == "masked_zero"
    assert "length-2" in evaluation["prediction_position"]
    assert "causal future masking required" in evaluation["target_dt_policy"]
    assert evaluation["body_strata"] == ["le_p50", "p50_p90", "p90_p95"]
    assert evaluation["body_formula"].startswith("sum(count*qty_mae)/sum(count)")
    assert evaluation["extreme_tail_stratum"] == "gt_p99"


def test_performance_gate_matches_previous_thresholds_not_previous_outcomes(contract):
    previous = read_json(ROOT / "paper/contracts/hard_lmm_weighted_static_v1.json")
    gate = contract["performance_gate"]
    for key, expected in previous["per_dataset_gate"].items():
        assert gate[key] == expected
    assert gate["missing_empty_or_zero_reference_metric"].startswith("not_evaluable")
    assert not gate["gate_relaxation_after_results"]
    assert not gate["test_data_before_gate"]


def test_small_stage_reduces_datasets_not_matched_epoch_budget(contract):
    execution = contract["future_execution"]
    first, later = execution["first_stage"], execution["conditional_stage"]
    assert first["datasets"] == ["yellow_trip_hourly", "raf_spare_parts"]
    assert first["candidate_training_runs"] == 2
    assert first["cuda_contracts_before_smoke"]
    assert first["full_e1_train_validation_smoke_per_dataset"]
    assert first["early_stopping_contract_unchanged"]
    assert not first["smoke_weights_used_for_screening"]
    assert later["datasets"] == ["intermittent_v2", "insta_market_basket"]
    assert later["requires_first_stage_pass"] and later["requires_separate_approval"]
    assert later["requires_own_full_e1_smoke"]
    assert "all four datasets pass" in execution["universal_followup_requires"]
    assert execution["fresh_artifact_only"] and execution["failure_status_must_be_atomic"]
    assert not execution["automatic_retry_or_resume"]
    assert execution["minimum_free_vram_mib"] == 12000
    assert execution["server_if_separately_approved"] == "5080"


def test_implementation_evidence_is_pending_not_passed(contract):
    acceptance = contract["implementation_acceptance"]
    assert acceptance["state"] == "not_run_candidate_not_implemented"
    requirements = acceptance["requirements"]
    assert len(requirements) == len(set(requirements)) == 16
    for required in (
        "identical_weight_quantity_state_prediction_and_quantity_only_gradients",
        "quantity_gradient_reaches_bank_with_nondegenerate_quantity_head",
        "checkpoint_route_metadata_and_strict_state_replay_reject_wrong_candidate",
        "target_future_padding_perturbation_cannot_change_history_prediction",
    ):
        assert required in requirements
    assert acceptance["cpu_identity_tolerance"] == {"rtol": 0.0, "atol": 0.0}
    markdown = CONTRACT_PATH.with_suffix(".md").read_text()
    assert "AdamW" in markdown and "not plain Adam" in markdown
    assert "16 required implementation checks" in markdown
    assert "not_run_candidate_not_implemented" in markdown
