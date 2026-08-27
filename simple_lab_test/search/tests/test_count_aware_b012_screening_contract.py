from __future__ import annotations

import json
from pathlib import Path

from paper.scripts.compare_count_aware_b012_seed42_screening import (
    SCREENING_DATASETS,
    evaluate_b2_gate,
    weighted_body_mae,
)
from paper.scripts.count_aware_tpp_backbone.constants import TITAN_B012_BACKBONES
from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def metric_payload(
    *, body_mae: float, rmse: float, p99_mae: float, time_nll: float
) -> dict[str, float | int]:
    return {
        "body_le_p95_mae": body_mae,
        "body_le_p95_count": 95,
        "overall_qty_mae": body_mae,
        "overall_qty_rmse": rmse,
        "gt_p99_mae": p99_mae,
        "gt_p99_count": 1,
        "time_nll": time_nll,
        "joint_objective": time_nll + 1.0,
        "completed_epochs": 80,
        "best_epoch": 40,
        "elapsed_seconds": 10.0,
    }


def test_weighted_body_mae_uses_all_train_only_strata_through_p95() -> None:
    rows = [
        {"stratum": "le_p50", "count": "50", "qty_mae": "1.0"},
        {"stratum": "p50_p90", "count": "40", "qty_mae": "2.0"},
        {"stratum": "p90_p95", "count": "5", "qty_mae": "5.0"},
        {"stratum": "p95_p99", "count": "4", "qty_mae": "50.0"},
        {"stratum": "gt_p99", "count": "1", "qty_mae": "500.0"},
    ]

    value, count = weighted_body_mae(rows)

    assert count == 95
    assert value == (50.0 + 80.0 + 25.0) / 95.0


def test_b2_gate_requires_primary_and_every_guardrail() -> None:
    baseline = metric_payload(body_mae=100.0, rmse=100.0, p99_mae=100.0, time_nll=2.0)
    passing = metric_payload(body_mae=94.0, rmse=101.0, p99_mae=101.0, time_nll=2.009)
    primary_failure = metric_payload(
        body_mae=96.0, rmse=101.0, p99_mae=101.0, time_nll=2.009
    )
    time_failure = metric_payload(
        body_mae=94.0, rmse=101.0, p99_mae=101.0, time_nll=2.011
    )

    assert evaluate_b2_gate(baseline, passing)["passed"] is True
    assert evaluate_b2_gate(baseline, primary_failure)["passed"] is False
    assert evaluate_b2_gate(baseline, time_failure)["passed"] is False


def test_frozen_json_matches_runtime_backbone_and_dataset_contracts() -> None:
    contract = json.loads(
        (
            PROJECT_ROOT
            / "paper/contracts/count_aware_titan_b012_screening_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert tuple(contract["backbones"]) == TITAN_B012_BACKBONES
    assert tuple(contract["screening"]["datasets"]) == SCREENING_DATASETS
    assert contract["shared_t0_contract"]["held_out_test"] == "locked"
    assert contract["b2_acceptance_gate"]["b1_selection_status"] == (
        "reference_only_not_selectable"
    )
    speed_policy = contract["preflight"]["speed_policy"]
    assert speed_policy["timing_batch_size"] == 128
    assert speed_policy["timing_sequence_length"] == 64
    assert speed_policy["b2_maximum_training_step_ratio_vs_b0"] == 3.0
    assert speed_policy["b1_segment_size"] == 16
    assert speed_policy["b1_policy"] == (
        "reference_only_ratio_must_be_disclosed_after_optimization"
    )
    for dataset_id, context in contract["dataset_contexts"].items():
        runtime = DATASET_CONTRACTS[dataset_id]
        assert context["lookback"] == runtime["lookback"]
        assert context["max_sequence_length"] == runtime["max_seq_len"]
