from __future__ import annotations

import json

from paper.scripts.count_aware_tpp_backbone.constants import MODEL_ROLES
from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    CONTRACT_PATH,
    DATASETS,
    evaluate_gate,
    job_command,
)


def _candidate(host: str) -> dict[str, str]:
    if host == "5090":
        return {
            "backbone": "titantpp_hard_memory_interlayer",
            "model_role": "hard_lmm_interlayer_memory_candidate",
        }
    return {
        "backbone": "titantpp_hard_memory_film",
        "model_role": "hard_lmm_memory_film",
    }


def test_frozen_candidate_roles_are_accepted_by_training_cli() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    for candidate in contract["candidates"].values():
        assert candidate["model_role"] in MODEL_ROLES


def test_job_command_fixes_candidate_selector_and_test_lock_inputs(tmp_path) -> None:
    command = job_command(
        python="/runtime/python",
        candidate=_candidate("5090"),
        dataset="insta_market_basket",
        output=tmp_path,
        source_revision="1" * 40,
        phase={"epochs": 300, "minimum_epochs": 40, "patience": 40},
        host_role="5090",
    )
    joined = " ".join(command)
    assert "titantpp_hard_memory_interlayer" in command
    assert "hard_lmm_interlayer_memory_candidate" in command
    assert "validation_raw_quantity_rmse" in command
    assert "legacy_clamped_rmtpp" in command
    assert "--time-intercept-limit 300" in joined
    assert "--quantile-adaptive-strength 0" in joined
    assert "--target-split" not in command
    assert DATASETS["insta_market_basket"]["validation_targets"] == 503733


def test_seed42_gate_requires_rmse_and_all_guardrails() -> None:
    baseline = {
        "raw_rmse": 10.0,
        "overall_mae": 5.0,
        "body_mae": 4.0,
        "gt_p99_mae": 20.0,
        "clamped_time_loss": 1.0,
    }
    passing = {
        "raw_rmse": 9.9,
        "overall_mae": 5.05,
        "body_mae": 4.08,
        "gt_p99_mae": 20.4,
        "clamped_time_loss": 1.01,
    }
    assert evaluate_gate(passing, baseline)["status"] == "passed"
    for metric in baseline:
        failing = dict(passing)
        failing[metric] = 10.0 if metric == "raw_rmse" else (
            baseline[metric] + 0.011
            if metric == "clamped_time_loss"
            else baseline[metric] * (1.011 if metric == "overall_mae" else 1.021)
        )
        result = evaluate_gate(failing, baseline)
        assert result["status"] == "failed"
        assert result["checks"][metric] is False
