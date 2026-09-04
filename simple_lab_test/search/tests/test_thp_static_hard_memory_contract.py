"""Document/identity checks only: the prospective candidate is not implemented."""

import hashlib
import json
import math
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = ROOT / "paper/contracts/count_aware_thp_static_hard_memory_v1.json"


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON: {value}")

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


@pytest.fixture
def registry(contract):
    path = ROOT / contract["baseline_reuse"]["registry"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == contract["baseline_reuse"]["registry_sha256"]
    return read_json(path)


def test_authorization_stops_at_local_documents_and_baseline_audit(contract):
    assert contract["status"] == "contract_frozen_implementation_not_started"
    assert {key for key, value in contract["authorization"].items() if value} == {
        "local_contract_and_document_validation", "local_baseline_metadata_and_checkpoint_audit"
    }
    assert contract["candidate"]["backbone_id_reserved"] == "thp_static_hard_memory"
    assert contract["candidate"]["model_role_reserved"] == "t0_thp_static_hard_memory"
    for key in ("registered_in_factory_at_freeze", "adopted_primary_model",
                "original_titans_reproduction", "hypothesis_confirmed"):
        assert contract["candidate"][key] is False


def test_original_project_thp_is_not_replaced_with_original_paper_adapter(contract):
    encoder = contract["encoder"]
    assert encoder["implementation_reference"].endswith("::CountAwareTHP")
    assert encoder["input_features"] == ["log1p_delta_t", "log1p_raw_quantity"]
    for key, value in {"hidden_dim": 64, "n_layers": 2, "n_heads": 4, "ff_dim": 256,
                       "dropout": .1, "activation": "gelu", "normalize_before": False,
                       "layernorm_epsilon": 1e-6, "persistent_tokens": 0,
                       "contextual_memory_size": 0}.items():
        assert encoder[key] == value
    for key in ("learned_positional_embedding", "sinusoidal_temporal_encoding",
                "add_temporal_encoding_each_layer", "use_rnn"):
        assert encoder[key] is False


def test_one_original_static_bank_and_explicit_capacity_delta(contract):
    memory, candidate = contract["memory"], contract["candidate"]
    assert memory["implementation_reference"].endswith("::HardLocalMemoryMatcher")
    assert memory["shape"] == [1, 64, 64] and memory["topk"] == 4
    assert memory["aggregation"] == "arithmetic_mean"
    assert "unnormalized" in memory["selected_values"]
    assert memory["train_update"] == "outer_optimizer_only"
    for key in ("online_writes", "persistent_attention_tokens", "mutable_series_state",
                "validation_parameter_update"):
        assert memory[key] is False
    assert candidate["new_parameters"] == math.prod(memory["shape"]) == 4096
    assert candidate["reference_parameter_count"] == 100291
    assert candidate["expected_parameter_count"] == 100291 + 4096
    assert "not_measured_on_candidate" in candidate["parameter_count_status"]
    assert {"similarity_weighting", "confidence_gate", "null_memory", "shrinkage",
            "calibration", "MAC", "surprise_update"} <= set(memory["forbidden_additions"])


def test_both_heads_receive_same_residual_without_local_time_combination(contract):
    route = contract["routing"]
    assert route["reference"] == {"time": "h", "quantity": "h"}
    assert route["candidate"] == {"time": "h+r", "quantity": "h+r"}
    assert route["encode_task_states_order"] == ["time", "quantity"]
    assert route["encoder_calls_per_forward"] == route["matcher_calls_per_forward"] == 1
    assert route["mask_after_residual"]
    assert not route["detach_any_state"] and not route["local_time_route_combined"]
    assert "both" in route["time_and_quantity_gradient_to_bank"]


def test_fresh_initialization_preserves_base_rng_not_nonzero_output_identity(contract):
    init = contract["initialization"]
    assert init["all_parameters_fresh_and_trainable"]
    assert init["preserve_base_parameter_names_shapes_and_init_order"]
    assert "fork_rng(devices=[])" in init["bank_construction"]
    assert "diagnostic-only" in init["zero_residual_control"]
    for key in ("reference_weights_used_for_training", "smoke_weights_used_for_screening",
                "full_candidate_output_identity_required", "historical_cuda_trajectory_identity_claimed"):
        assert init[key] is False


def test_training_head_and_selection_match_previously_frozen_conditions(contract):
    previous = read_json(ROOT / "paper/contracts/hard_lmm_local_time_v1.json")["prospective_training"]
    training = contract["training"]
    for key in ("seed", "maximum_epochs", "minimum_epochs", "patience", "batch_size",
                "learning_rate", "optimizer", "gradient_clip", "quantity_objective",
                "lambda_log_qty", "lambda_tail", "checkpoint_monitor", "checkpoint_comparison",
                "loader", "lr_schedule_or_pcgrad"):
        assert training[key] == previous[key]
    for key, value in previous["time_launch_arguments"].items():
        assert training["time_head"][key] == value
    assert training["time_head"]["time_initial_intercept"] == 0.
    assert training["time_head"]["w_raw_initial"] == -3.
    assert not training["epoch_zero_eligible"] and not training["posthoc_mae_or_rmse_selection"]
    assert not training["new_amp_or_compile_policy"]
    assert "distinct from legacy launch arguments" in training["derived_time_statistics"]


def test_causality_and_heldout_scope_do_not_make_unsupported_historical_claim(contract):
    data, reuse = contract["data_and_evaluation"], contract["baseline_reuse"]
    assert data["allowed_materialized_splits"] == ["train", "validation"]
    assert data["filter_test_before_materialization"]
    assert data["evaluation_scope"] == "validation_only"
    assert not data["held_out_test_evaluated"] and not data["subsample_train_or_validation"]
    assert data["target_quantity_in_encoder"] == "masked_zero"
    assert "length-2" in data["prediction_position"]
    assert "causal mask blocks" in data["target_dt_policy"]
    assert data["body_strata"] == ["le_p50", "p50_p90", "p90_p95"]
    assert data["extreme_tail_stratum"] == "gt_p99"
    assert "materialization is not claimed" in reuse["historical_test_metadata"]
    assert "block comparison" in reuse["missing_or_mismatched_reference"]


def test_registry_has_exactly_four_pinned_seed42_references(contract, registry):
    original = read_json(ROOT / registry["source_registry"])
    rows = {row["dataset"]: row for row in original["datasets"]}
    assert hashlib.sha256((ROOT / registry["source_registry"]).read_bytes()).hexdigest() == registry["source_registry_sha256"]
    assert [row["dataset"] for row in registry["datasets"]] == contract["data_and_evaluation"]["datasets"]
    for row in registry["datasets"]:
        old = rows[row["dataset"]]
        for key in ("data_path", "data_sha256", "split_manifest_path", "split_manifest_sha256",
                    "artifact_dir", "lookback", "max_seq_len"):
            assert row[key] == old[key]
        context = contract["data_and_evaluation"]["context"][row["dataset"]]
        for key in ("lookback", "max_seq_len", "train_targets", "validation_targets"):
            assert context[key] == row[key]
        assert set(row["baselines"]) == {"thp", "titantpp"}
        for reference in row["baselines"].values():
            assert len(reference["files_sha256"]) == 4
            assert all("/seed_42/" in path for path in reference["files_sha256"])
            assert all(len(value) == 64 for value in reference["files_sha256"].values())
            assert reference["completed_epochs"] - reference["best_epoch"] == 40
            assert math.isfinite(reference["body_mae"]) and reference["body_mae"] > 0
        assert row["baselines"]["thp"]["parameter_count"] == 100291
        assert row["baselines"]["titantpp"]["checkpoint_state_sha256"] == old["checkpoint_state_sha256"]
    reuse = contract["baseline_reuse"]
    for key in ("retrain_benchmarks_automatically", "compare_candidate_seed42_to_baseline_three_seed_mean",
                "local_time_weighted_or_mac_as_reference"):
        assert reuse[key] is False


def test_direct_thp_criteria_are_prospective_and_prior_hard_gate_is_preserved(contract):
    decision = contract["decision"]
    assert decision["direct_thp_gate"] == {
        "body_mae_regression_max": 0., "overall_mae_regression_max": 0.,
        "overall_rmse_improvement_min": .02, "gt_p99_mae_improvement_min": .02,
        "time_nll_absolute_increase_max": .01,
    }
    previous = read_json(ROOT / "paper/contracts/hard_lmm_local_time_v1.json")["performance_gate"]
    for key, value in decision["hard_lmm_compatibility_gate"].items():
        assert previous[key] == value
    assert "not a significance threshold" in decision["new_gate_rationale"]
    assert decision["all_datasets_required"]
    assert not decision["thresholds_changed_after_candidate_results"]
    assert not decision["historical_local_time_decision_changed"]
    assert decision["invalid_missing_empty_nonfinite_or_zero_reference"].startswith("not_evaluable")
    assert decision["any_performance_failure"] == "hold_no_expansion"
    assert "no_automatic_retry" in decision["technical_failure"]


def test_absolute_limits_use_both_references_and_unrounded_values(contract, registry):
    direct, hard_gate = (contract["decision"][key] for key in ("direct_thp_gate", "hard_lmm_compatibility_gate"))
    for row in registry["datasets"]:
        thp, hard = (row["baselines"][key] for key in ("thp", "titantpp"))
        expected = {
            "body_mae_max": min(thp["body_mae"] * (1 + direct["body_mae_regression_max"]),
                                hard["body_mae"] * (1 - hard_gate["body_le_p95_mae_improvement_min"])),
            "overall_mae_max": thp["best_val_qty_mae"] * (1 + direct["overall_mae_regression_max"]),
            "overall_rmse_max": min(thp["best_val_qty_rmse"] * (1 - direct["overall_rmse_improvement_min"]),
                                    hard["best_val_qty_rmse"] * (1 + hard_gate["overall_rmse_regression_max"])),
            "gt_p99_mae_max": min(thp["gt_p99_mae"] * (1 - direct["gt_p99_mae_improvement_min"]),
                                  hard["gt_p99_mae"] * (1 + hard_gate["gt_p99_mae_regression_max"])),
            "time_nll_max": min(thp["best_val_time_nll"] + direct["time_nll_absolute_increase_max"],
                                hard["best_val_time_nll"] + hard_gate["time_nll_absolute_increase_max"]),
        }
        assert row["prospective_absolute_gate_limits"] == expected


def test_future_implementation_tests_and_server_execution_remain_pending(contract):
    acceptance, execution = contract["implementation_acceptance"], contract["future_execution"]
    assert acceptance["state"] == "not_run_candidate_not_implemented"
    assert len(acceptance["requirements"]) == len(set(acceptance["requirements"])) == 16
    assert acceptance["cpu_identity_tolerance"] == {"rtol": 0., "atol": 0.}
    assert execution["implementation_source_revision"] is None and execution["run_commands"] == []
    assert len(execution["stages"]) == 3 and all("approval" in stage for stage in execution["stages"])
    assert execution["candidate_screening_runs"] == 2
    assert execution["server_if_separately_approved"] == "5080"
    assert execution["minimum_free_vram_mib"] == 12000
    for key in ("cuda_model_tests_before_full_e1", "e1_is_feasibility_not_performance",
                "fresh_artifact_only", "independent_process_per_dataset", "atomic_failed_status",
                "service_changes_require_explicit_approval"):
        assert execution[key]
    for key in ("automatic_retry_resume_or_expansion", "sync_delete_allowed", "scheduler_now"):
        assert execution[key] is False


@pytest.mark.parametrize("path,expected", read_json(CONTRACT_PATH)["evidence"]["reviewed_source_sha256"].items())
def test_reviewed_source_is_pinned_to_git_objects_not_future_working_tree(contract, path, expected):
    revision = contract["evidence"]["reviewed_code_revision"]
    source = subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT)
    assert hashlib.sha256(source).hexdigest() == expected


def test_human_document_discloses_scope_parameter_confounds_and_pending_tests():
    markdown = CONTRACT_PATH.with_suffix(".md").read_text()
    for required in ("NOT candidate", "4,096 parameters", "not retrieval versus a parameter-matched alternative",
                     "engineering screening criterion", "NOT those model tests",
                     "not_run_candidate_not_implemented"):
        assert required in markdown
