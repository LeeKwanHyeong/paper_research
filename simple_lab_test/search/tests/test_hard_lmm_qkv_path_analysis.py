from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts import analyze_hard_lmm_qkv_path_diagnostic as analysis


CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_qkv_path_diagnostic_v1.json"


def contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def series_for_fold(fold: int, count: int) -> list[str]:
    values: list[str] = []
    candidate = 0
    while len(values) < count:
        value = f"series-{fold}-{candidate}"
        if analysis.fold_for_series(value) == fold:
            values.append(value)
        candidate += 1
    return values


def extraction_arrays() -> dict[str, np.ndarray]:
    series = np.asarray(series_for_fold(0, 6) + series_for_fold(1, 6))
    n = len(series)
    fold = np.asarray([analysis.fold_for_series(value) for value in series])
    quantity = np.arange(1, n + 1, dtype=np.float64)
    duration = np.ones(n, dtype=np.float64)
    arrays: dict[str, np.ndarray] = {
        "fold": fold,
        "dataset_index": np.arange(n, dtype=np.int64),
        "series_id": series,
        "true_quantity": quantity,
        "true_duration": duration,
    }
    for variant_index, variant in enumerate(analysis.VARIANTS):
        prediction = quantity + 0.1 * (variant_index + 1)
        w = np.full(n, 3.0 if variant == "B" else 2.0)
        wd_raw = w * duration
        wd_clamped = np.minimum(wd_raw, 10.0)
        intercept = np.zeros(n)
        exp_intercept = np.exp(intercept)
        integral = exp_intercept / w * np.expm1(wd_clamped)
        values = {
            "prediction": prediction,
            "log_mse": np.square(np.log1p(prediction) - np.log1p(quantity)),
            "absolute_error": np.abs(prediction - quantity),
            "squared_error": np.square(prediction - quantity),
            "legacy_time_loss": -intercept - wd_raw + integral,
            "time_intercept_raw": intercept,
            "time_intercept_clamped": intercept,
            "exp_intercept": exp_intercept,
            "w": w,
            "wd_raw": wd_raw,
            "wd_clamped": wd_clamped,
            "integral_term": integral,
            "wd_saturated": (wd_raw > 10.0).astype(bool),
            "intercept_saturated": np.zeros(n, dtype=bool),
        }
        for name, value in values.items():
            arrays[f"{variant}__{name}"] = value
        for field_index, field in enumerate(analysis.STATE_FIELDS):
            arrays[f"{variant}__{field}"] = np.full(
                (n, 3), variant_index + field_index / 10.0
            )
        arrays[f"{variant}__top4_indices"] = np.tile(
            np.asarray([0, 1, 2, 3], dtype=np.int64), (n, 1)
        )
    return arrays


def metric(raw: float, mae: float, time: float) -> dict:
    return {
        "raw_rmse": raw,
        "mae": mae,
        "log_mse": 1.0,
        "legacy_time_loss": time,
        "all_positive_wd_saturated_at_10": False,
        "theoretical_wd10_clamp_floor_mean": None,
    }


def fake_analyses() -> dict:
    result = {}
    for dataset_row in contract()["datasets"]:
        scopes = {}
        for fold in (0, 1):
            variants = {
                variant: metric(1.0, 1.0, 0.0)
                for variant in analysis.VARIANTS
            }
            variants["B"] = metric(1.0, 1.0, -0.025)
            variants["FULL"] = metric(1.0, 1.0, 0.0)
            variants["QK"] = metric(1.01, 1.01, -0.005)
            variants["ZERO"] = metric(1.01 / 0.99, 1.01, -0.01)
            scopes[f"fold{fold}"] = {"variants": variants}

            body = copy.deepcopy(variants)
            body["QK"]["mae"] = 1.02
            body["ZERO"]["mae"] = 1.0
            scopes[f"fold{fold}__body_le_train_p95"] = {"variants": body}

            tail = copy.deepcopy(variants)
            tail["QK"]["mae"] = 1.02
            tail["ZERO"]["mae"] = 1.0
            scopes[f"fold{fold}__tail_gt_train_p99"] = {
                "support": {"target_count": 10, "unique_series": 5},
                "variants": tail,
            }
        result[dataset_row["dataset"]] = {"aggregates": scopes}
    return result


def gradient_records() -> list[dict]:
    rows = []
    for variant in analysis.GRADIENT_VARIANTS:
        for fold in (0, 1):
            for batch in range(4):
                for group in analysis.GRADIENT_GROUPS:
                    unused = variant == "B" and group.startswith("new_")
                    rows.append(
                        {
                            "variant": variant,
                            "fold": fold,
                            "batch_index": batch,
                            "group": group,
                            "dot_product": 0.0,
                            "cosine": None if unused else 0.25,
                            "time_norm": 0.0 if unused else 2.0,
                            "quantity_norm": 0.0 if unused else 4.0,
                            "unused_or_zero_gradient": unused,
                        }
                    )
    return rows


def test_gate_includes_exact_boundaries_and_null_gradient_cosines() -> None:
    decision = analysis.evaluate_prospective_qk_gate(fake_analyses(), contract())
    assert decision["outcome_type"] == "supported"
    assert not decision["numerical_failures"]
    rows = analysis.summarize_gradient_records(gradient_records(), contract())
    assert rows["variants"]["B"]["new_Q"]["cosine_valid_count"] == 0
    assert rows["variants"]["B"]["new_Q"]["cosine_mean"] is None


def test_negative_time_nll_uses_additive_not_ratio_threshold() -> None:
    data = fake_analyses()
    dataset = contract()["datasets"][0]["dataset"]
    data[dataset]["aggregates"]["fold0"]["variants"]["FULL"][
        "legacy_time_loss"
    ] = -2.0
    data[dataset]["aggregates"]["fold0"]["variants"]["QK"][
        "legacy_time_loss"
    ] = -1.989
    decision = analysis.evaluate_prospective_qk_gate(data, contract())
    failed = {row["criterion"] for row in decision["numerical_failures"]}
    assert "QK_legacy_time_max_increase_FULL" in failed


def test_missing_tail_is_inconclusive_but_other_numeric_failure_still_fails() -> None:
    data = fake_analyses()
    dataset = contract()["datasets"][0]["dataset"]
    data[dataset]["aggregates"]["fold0__tail_gt_train_p99"]["support"] = {
        "target_count": 9,
        "unique_series": 5,
    }
    decision = analysis.evaluate_prospective_qk_gate(data, contract())
    assert decision["outcome_type"] == "inconclusive"
    data[dataset]["aggregates"]["fold0"]["variants"]["QK"]["mae"] = 1.02
    decision = analysis.evaluate_prospective_qk_gate(data, contract())
    assert decision["outcome_type"] == "failed"


def test_cross_b_time_gap_is_marked_nondiagnostic_under_complete_clamping() -> None:
    data = fake_analyses()
    for dataset in data.values():
        for fold in (0, 1):
            variants = dataset["aggregates"][f"fold{fold}"]["variants"]
            for variant, floor in (("B", -3.0), ("FULL", -1.0)):
                variants[variant]["all_positive_wd_saturated_at_10"] = True
                variants[variant]["theoretical_wd10_clamp_floor_mean"] = floor
    decision = analysis.evaluate_prospective_qk_gate(data, contract())
    assert decision["raw_contractual_outcome_type"] == "supported"
    assert decision["outcome_type"] == "inconclusive"
    assert decision["gate_validity"]["cross_B_time_trigger_is_diagnostic"] is False
    assert "different analytic slope floors" in decision["gate_validity"]["warning"]


def test_bool_telemetry_is_accepted_and_nonfinite_is_rejected() -> None:
    arrays = extraction_arrays()
    validated = analysis.validate_extraction(
        arrays,
        dataset_spec=contract()["datasets"][0],
        contract=contract(),
        enforce_sampling_contract=False,
    )
    assert validated["FULL__wd_saturated"].dtype == np.bool_
    bad = extraction_arrays()
    bad["QK__prediction"][0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        analysis.validate_extraction(
            bad,
            dataset_spec=contract()["datasets"][0],
            contract=contract(),
            enforce_sampling_contract=False,
        )


def test_interaction_and_top4_order_vs_set_are_distinct() -> None:
    arrays = {
        "FULL__prediction": np.asarray([10.0, 4.0]),
        "QK__prediction": np.asarray([3.0, 1.0]),
        "V__prediction": np.asarray([4.0, 1.0]),
        "ZERO__prediction": np.asarray([1.0, 0.0]),
    }
    result = analysis.interaction_summary(
        arrays, field="prediction", mask=np.asarray([True, True])
    )
    assert result["mean"] == pytest.approx(3.0)
    assert result["root_mean_square"] == pytest.approx(np.sqrt(10.0))

    reference = np.asarray([[1, 2, 3, 4], [1, 2, 3, 4]])
    candidate = np.asarray([[4, 3, 2, 1], [1, 2, 5, 6]])
    top4 = analysis.top4_change_summary(
        reference, candidate, np.asarray([True, True])
    )
    assert top4["ordered_exact_fraction"] == 0.0
    assert top4["same_set_fraction"] == 0.5
    assert top4["mean_set_overlap_fraction"] == 0.75
