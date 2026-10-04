"""Build CPU-only implementation proof from the targeted passing JUnit suite.

This script constructs initialized models only. It reads no research data or
checkpoint, starts no trainer, and performs no remote/GPU operation.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

import torch
from models.TPPs.CountAwareTitanCNNGRU import ARMS, BASELINE, metadata
from models.TPPs.CountAwareTitanHistoryWidth import identity as width_identity
from paper.scripts import run_titantpp_cnn_gru as runner

test_suites = ET.parse(OUT / "cpu_contract_tests.xml").getroot().findall("testsuite")
tests = sum(int(suite.get("tests", 0)) for suite in test_suites)
failures = sum(int(suite.get("failures", 0)) for suite in test_suites)
errors = sum(int(suite.get("errors", 0)) for suite in test_suites)
skipped = sum(int(suite.get("skipped", 0)) for suite in test_suites)
assert tests >= 56 and failures == errors == skipped == 0
cases = [case for suite in test_suites for case in suite.findall("testcase")]
cnn_gru_cases = [case for case in cases if case.get("classname", "").endswith("test_titantpp_cnn_gru")]
replays = [case for case in cnn_gru_cases if case.get("name", "").startswith("test_actual_shared_trainer_selected_last_validation_replay")]
assert len(replays) == 3

def data(length):
    return {"model": {"hidden_dim": 64, "quantity_variant": runner.VARIANT,
        "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 7.,
        "time_initial_location": .2, "time_initial_scale": .8,
        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None}},
        "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": length}}

torch.set_num_threads(2)
counts, correction, identities = {}, {}, {}
for length in (84, 256):
    counts[str(length)], correction[str(length)] = {}, {}
    for arm in (BASELINE, *ARMS):
        torch.manual_seed(42)
        model, _ = runner.build_model(data(length), arm)
        counts[str(length)][arm] = sum(parameter.numel() for parameter in model.parameters())
        correction[str(length)][arm] = sum(parameter.numel() for parameter in model.multilag_detail.parameters())
    identities[str(length)] = {str(seed): runner.initial_states(data(length), seed) for seed in (42, 52, 62)}
assert counts["256"] == dict(zip((BASELINE, *ARMS), (114435, 115011, 114591, 115167)))
assert counts["84"] == dict(zip((BASELINE, *ARMS), (103427, 104003, 103583, 104159)))
baseline_identity = bytes(width_identity().tolist()).hex()
assert baseline_identity == "1309671f1eb463992c1ec3b37209d73cc81501cc81d9e513029566e84bd8277a"

source_files = [
    "models/TPPs/CountAwareTitanCNNGRU.py", "paper/scripts/run_titantpp_cnn_gru.py",
    "simple_lab_test/search/tests/test_titantpp_cnn_gru.py",
    "models/TPPs/CountAwareTitanHistoryWidth.py", "models/TPPs/CountAwareTitanCoreAblation.py",
    "models/TPPs/CountAwareTitanMultiLagDetail.py", "models/TPPs/CountAwareTPP.py",
    "models/TPPs/CountAwareFactory.py", "models/Titan/common/memory.py",
    "paper/scripts/count_aware_tpp_backbone/core.py", "paper/scripts/count_aware_tpp_backbone/training.py",
    "reports/titantpp_cnn_gru_width16_design_20261004_v1/design.json",
    "reports/titantpp_cnn_gru_implementation_20261004_v1/verify_implementation.py",
    "reports/titantpp_cnn_gru_implementation_20261004_v1/cpu_contract_tests.xml",
]
result = {
    "schema_version": 1,
    "status": "implemented_cpu_contracts_verified_native_gpu_qualification_pending",
    "observed_at_utc": datetime.now(timezone.utc).isoformat(),
    "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": "cpu"},
    "test_suite": {"tests": tests, "failures": failures, "errors": errors, "skipped": skipped,
        "cnn_gru_cases": len(cnn_gru_cases), "synthetic_shared_trainer_arms": len(replays),
        "synthetic_epochs_per_arm": 2, "selected_and_last_validation_replays": 6,
        "process_route_hooks_restored_after_each_candidate_test": True},
    "full_model_parameter_counts": counts,
    "correction_parameter_counts": correction,
    "cnn_parameter_count": 576,
    "new_arm_metadata": {arm: metadata(arm=arm) for arm in ARMS},
    "initial_state_sha256_by_length_and_seed": identities,
    "width16_legacy_architecture_identity": baseline_identity,
    "checks": {name: True for name in (
        "cnn_observed_segment_padding_skip", "cnn_current_lag1_lag2_manual_formula_and_gradients",
        "persistent_qkv_attention_parameters_reused_and_persistent_tokens_unfiltered",
        "all_cnn_kernel_rows_receive_actual_joint_loss_gradient_at_zero_initialization",
        "gru_manual_recurrence_and_gradient", "gru_withheld_reset_and_first_event_zero_correction",
        "future_target_and_nonobserved_invariance", "sample_independence_no_cross_call_state",
        "common_rng_and_initial_model_tensors_all_three_seeds", "zero_added_path_initial_output_identity",
        "checkpoint_identity_and_incomplete_state_fail_before_load", "candidate_relabel_scope_and_head_rejection",
        "frozen_quantity_time_selector_role_contracts", "synthetic_optimizer_and_checkpoint_roundtrip",
        "synthetic_actual_shared_training_selected_last_validation_replay",
        "same_process_width_capacity_and_time_diagnostic_regression")},
    "research_data_read": False, "research_checkpoint_read": False,
    "research_training_launched": False, "synthetic_training_executed": True,
    "held_out_test_evaluated": False, "test_results_read": False,
    "deterministic_gpu_qualification": "not_performed",
    "unexecuted_checks": ["native GPU torch2.11 execution and deterministic GRU qualification",
        "GPU peak memory and step/latency measurements", "real-data CNN/GRU training and endpoint audit"],
    "limitations": ["CPU torch2.14 proof does not replace remote torch2.11 qualification",
        "parameter closeness does not match compute or expressive capacity",
        "whole GRU module replacement also changes history context, eligibility and residual scale",
        "no improvement, generalization or paper efficiency claim follows from these contract tests"],
    "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_files},
}
(OUT / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"status": result["status"], "test_suite": result["test_suite"],
    "full_model_parameter_counts": counts}, ensure_ascii=False, indent=2))
