from __future__ import annotations

import math

from paper.scripts.audit_online_retail_time_scale_gradients import (
    build_scale_variants,
    evaluate_stability_gate,
    jacobian_corrected_hour_nll,
    recommend_next_action,
)


def _summary(variant: str, passed: bool) -> dict:
    return {"variant": variant, "stability_gate": {"passed": passed}}


def test_scale_variants_use_only_constants_and_train_statistics() -> None:
    variants = build_scale_variants({"p50": 21.0, "mean": 55.25, "p95": 215.0})
    assert variants["S0_raw_hour"]["divisor"] == 1.0
    assert variants["S1_calendar_day"]["divisor"] == 24.0
    assert variants["S2_train_target_median"]["divisor"] == 21.0
    assert variants["S3_train_target_mean"]["divisor"] == 55.25
    assert variants["S4_train_target_p95"]["divisor"] == 215.0


def test_jacobian_correction_reports_density_per_original_hour() -> None:
    assert math.isclose(
        jacobian_corrected_hour_nll(2.0, 24.0),
        2.0 + math.log(24.0),
    )


def test_stability_gate_rejects_full_gradient_clipping() -> None:
    history = [
        {
            "train_joint_objective": 2.0,
            "train_max_per_event_time_nll": 20.0,
            "batch_count": 4,
            "gradient_clip_count": 4,
            "time_only_gradient_exceed_count": 4,
            "quantity_only_gradient_exceed_count": 0,
            "all_finite": True,
        }
    ]
    gate = evaluate_stability_gate(history, run_status="success")
    assert not gate["passed"]
    assert not gate["checks"]["time_only_gradient_exceed_fraction_within_limit"]


def test_recommendation_prefers_interpretable_calendar_day_candidate() -> None:
    summaries = [
        _summary("S0_raw_hour", False),
        _summary("S1_calendar_day", True),
        _summary("S2_train_target_median", True),
        _summary("S3_train_target_mean", True),
        _summary("S4_train_target_p95", True),
    ]
    decision = recommend_next_action(summaries)
    assert decision["decision"] == "retain_online_retail_for_scaled_time_followup"
    assert decision["primary_candidate"] == "S1_calendar_day"


def test_recommendation_stops_when_all_scaled_variants_fail() -> None:
    summaries = [
        _summary("S0_raw_hour", False),
        _summary("S1_calendar_day", False),
        _summary("S2_train_target_median", False),
        _summary("S3_train_target_mean", False),
        _summary("S4_train_target_p95", False),
    ]
    assert recommend_next_action(summaries)["decision"] == "stop_online_retail_under_legacy_time_head"
