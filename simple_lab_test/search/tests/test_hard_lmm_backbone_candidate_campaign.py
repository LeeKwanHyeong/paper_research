from __future__ import annotations

import json
import math

import pytest

from paper.scripts.count_aware_tpp_backbone.constants import MODEL_ROLES
from paper.scripts.run_hard_lmm_backbone_candidate_campaign import (
    ALL_QUANTITY_STRATA,
    BODY_STRATA,
    B_VALIDATION_POPULATION_REFERENCES,
    CONTRACT_PATH,
    DATASETS,
    aggregate_quantity_metrics,
    evaluate_gate,
    job_command,
    validate_validation_population,
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


def _quantity_summary() -> dict[str, object]:
    counts = {
        "le_p50": 10,
        "p50_p90": 20,
        "p90_p95": 30,
        "p95_p99": 40,
        "gt_p99": 50,
    }
    maes = {
        "le_p50": 1.0,
        "p50_p90": 2.0,
        "p90_p95": 3.0,
        "p95_p99": 100.0,
        "gt_p99": 5.0,
    }
    rmses = {
        "le_p50": 1.5,
        "p50_p90": 2.5,
        "p90_p95": 3.5,
        "p95_p99": 100.5,
        "gt_p99": 5.5,
    }
    time_losses = {name: 0.1 * (index + 1) for index, name in enumerate(ALL_QUANTITY_STRATA)}
    rows = [
        {
            "stratum": name,
            "count": counts[name],
            "qty_mae": maes[name],
            "qty_rmse": rmses[name],
            "time_nll": time_losses[name],
        }
        for name in ALL_QUANTITY_STRATA
    ]
    count = sum(counts.values())
    return {
        "quantity_rows": rows,
        "best_val_qty_mae": sum(counts[name] * maes[name] for name in counts) / count,
        "best_val_qty_rmse": math.sqrt(
            sum(counts[name] * rmses[name] ** 2 for name in counts) / count
        ),
        "best_val_time_nll": (
            sum(counts[name] * time_losses[name] for name in counts) / count
        ),
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


def test_body_aggregation_excludes_p95_to_p99_stratum() -> None:
    summary = _quantity_summary()

    metrics = aggregate_quantity_metrics(summary, expected_count=150)

    expected = (10 * 1.0 + 20 * 2.0 + 30 * 3.0) / 60
    wrongly_including_p95_p99 = (10 * 1.0 + 20 * 2.0 + 30 * 3.0 + 40 * 100.0) / 100
    assert BODY_STRATA == ("le_p50", "p50_p90", "p90_p95")
    assert metrics["body_target_count"] == 60
    assert metrics["body_mae"] == pytest.approx(expected)
    assert metrics["body_mae"] != pytest.approx(wrongly_including_p95_p99)


def test_quantity_aggregation_rejects_duplicate_or_missing_strata() -> None:
    summary = _quantity_summary()
    rows = summary["quantity_rows"]
    assert isinstance(rows, list)
    rows[-1] = dict(rows[0])

    with pytest.raises(ValueError, match="Duplicate quantity stratum"):
        aggregate_quantity_metrics(summary, expected_count=150)


@pytest.mark.parametrize(
    ("metric", "message"),
    (
        ("best_val_qty_mae", "Overall quantity MAE disagrees"),
        ("best_val_qty_rmse", "Overall quantity RMSE disagrees"),
    ),
)
def test_quantity_aggregation_rejects_summary_aggregate_mismatch(
    metric: str,
    message: str,
) -> None:
    summary = _quantity_summary()
    summary[metric] = float(summary[metric]) + 0.01

    with pytest.raises(ValueError, match=message):
        aggregate_quantity_metrics(summary, expected_count=150)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    (
        ("qty_mae", None, "is not numeric"),
        ("qty_rmse", float("nan"), "is not finite"),
        ("time_nll", float("inf"), "is not finite"),
    ),
)
def test_quantity_aggregation_rejects_missing_or_nonfinite_row_metrics(
    metric: str,
    value: object,
    message: str,
) -> None:
    summary = _quantity_summary()
    rows = summary["quantity_rows"]
    assert isinstance(rows, list)
    rows[0][metric] = value

    with pytest.raises(ValueError, match=message):
        aggregate_quantity_metrics(summary, expected_count=150)


def test_validation_population_rejects_identity_and_quantity_hash_mismatch() -> None:
    expected = B_VALIDATION_POPULATION_REFERENCES["insta_market_basket"]
    assert validate_validation_population(
        dataset="insta_market_basket",
        population=dict(expected),
    ) == expected

    for key in ("target_identity_sha256", "target_quantity_sha256"):
        mismatched = dict(expected)
        mismatched[key] = "0" * 64
        with pytest.raises(ValueError, match=key):
            validate_validation_population(
                dataset="insta_market_basket",
                population=mismatched,
            )
