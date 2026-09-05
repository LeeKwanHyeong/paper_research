"""Synthetic tests for the frozen transition-error OOF analysis."""

import copy
import json

import numpy as np
import pytest

from paper.scripts.hard_lmm_transition_error_analysis import (
    BONFERRONI_QUANTILE,
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    DATASETS,
    _body_noninferior,
    _comparison,
    aggregate_common_gate,
    analyze_cache,
)


def _cache(*, signal: bool = True, n: int = 1200) -> dict:
    rng = np.random.default_rng(811)
    rows = np.arange(n)
    series = rows // 10
    folds = series % 2
    prototype = rng.normal(size=n)
    sham = rng.normal(size=n)
    base_pre_activation = np.full(n, 1.25)
    corrected_pre_activation = base_pre_activation + (0.85 * prototype if signal else 0.0)
    log_target = np.logaddexp(0.0, corrected_pre_activation)
    return {
        "quantity": np.expm1(log_target),
        "base_pre_activation": base_pre_activation,
        "history_length": np.full(n, 5),
        "transition_count": np.full(n, 4),
        "selected_occupied_fraction": np.ones(n),
        "selected_count_mean": np.full(n, 1.5),
        "selected_count_std": np.full(n, 0.25),
        "last_error": np.zeros(n),
        "unconditioned_mean_error": np.zeros(n),
        "prototype_conditioned_error": prototype,
        "sham_prototype_conditioned_error": sham,
        "fold": folds,
        "series_id": series,
        "body_threshold": float(np.max(np.expm1(log_target)) + 1.0),
    }


def test_prototype_signal_passes_fixed_gate_and_reports_predictions():
    summary, arrays = analyze_cache(_cache(), bootstrap_repeats=10_000)

    assert summary["status"] == "PASS"
    assert summary["assessable"] and summary["gate"]["passes"]
    assert summary["decoder_contract"]["bonferroni_one_sided_quantile"] == pytest.approx(0.05 / 6)
    assert BONFERRONI_QUANTILE == pytest.approx(0.05 / 6)
    assert BOOTSTRAP_REPEATS == 10_000
    assert BOOTSTRAP_SEED == 20260906
    assert summary["decoder_contract"]["bootstrap_seed"] == 20260906
    for reference in ("strong_control", "sham"):
        comparison = summary["gate"]["comparisons"][reference]
        assert comparison["passes"]
        assert comparison["relative_gain"] > 0.5
        assert comparison["bootstrap"]["repeats"] == 10_000
        assert comparison["bootstrap"]["lower_bonferroni"] > 0
        assert comparison["bootstrap"]["shared_draws_across_contrasts"]
    assert summary["gate"]["original_t0_body_mae_noninferiority"]["passes"]
    assert arrays["candidate__log_prediction"].shape == (1200,)
    assert arrays["candidate__squared_log_error"].mean() < arrays[
        "strong_control__squared_log_error"
    ].mean()
    for fit in summary["arms"]["candidate"]["fit"].values():
        assert fit["objective_sum_squares"] == pytest.approx(
            fit["train_log_sse"] + fit["l2_penalty"], rel=1e-12, abs=1e-12
        )
    assert all(np.isfinite(value).all() for value in arrays.values())
    json.dumps(summary, allow_nan=False)


def test_no_signal_fails_without_manufacturing_gain():
    summary, arrays = analyze_cache(_cache(signal=False), bootstrap_repeats=100)

    assert summary["status"] == "FAIL"
    assert summary["assessable"] and not summary["gate"]["passes"]
    for reference in ("strong_control", "sham"):
        comparison = summary["gate"]["comparisons"][reference]
        assert not comparison["passes"]
        assert not comparison["conditions"]["pooled_log_mse_gain_at_least_one_percent"]
    np.testing.assert_allclose(
        arrays["candidate__log_prediction"], arrays["original_t0__log_prediction"], atol=1e-12,
    )


def test_held_fold_labels_and_inputs_do_not_enter_its_fit_statistics():
    original = _cache()
    changed_labels = copy.deepcopy(original)
    held = original["fold"] == 0
    changed_labels["quantity"][held] *= 10.0
    changed_labels["body_threshold"] = float(changed_labels["quantity"].max() + 1)
    before, before_arrays = analyze_cache(original, bootstrap_repeats=20)
    after, after_arrays = analyze_cache(changed_labels, bootstrap_repeats=20)
    np.testing.assert_array_equal(
        before_arrays["candidate__log_prediction"][held],
        after_arrays["candidate__log_prediction"][held],
    )
    assert before["arms"]["candidate"]["fit"]["0"] == after["arms"]["candidate"]["fit"]["0"]

    changed_inputs = copy.deepcopy(original)
    changed_inputs["prototype_conditioned_error"][held] += 100.0
    changed_inputs["last_error"][held] -= 50.0
    changed, _ = analyze_cache(changed_inputs, bootstrap_repeats=20)
    for field in ("input_mean", "input_raw_std", "input_scale"):
        assert before["arms"]["candidate"]["fit"]["0"][field] == changed["arms"]["candidate"]["fit"]["0"][field]


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("nonfinite", "finite"),
        ("negative_count", "nonnegative"),
        ("bad_fraction", r"\[0, 1\]"),
        ("zero_history", "at least one"),
        ("impossible_count", "cannot exceed"),
        ("zero_count_summary", "zero transition-error"),
        ("mixed_series", "disjoint"),
    ],
)
def test_invalid_cache_is_rejected(fault, message):
    data = _cache()
    if fault == "nonfinite":
        data["prototype_conditioned_error"][0] = np.nan
    elif fault == "negative_count":
        data["transition_count"][0] = -1
    elif fault == "bad_fraction":
        data["selected_occupied_fraction"][0] = 1.01
    elif fault == "zero_history":
        data["history_length"][0] = 0
    elif fault == "impossible_count":
        data["transition_count"][0] = data["history_length"][0]
    elif fault == "zero_count_summary":
        data["transition_count"][0] = 0
        data["prototype_conditioned_error"][0] = 0.1
    else:
        first_fold_one = np.flatnonzero(data["fold"] == 1)[0]
        data["series_id"][first_fold_one] = data["series_id"][0]
    with pytest.raises(ValueError, match=message):
        analyze_cache(data, bootstrap_repeats=2)


def test_one_fold_body_regression_vetoes_an_otherwise_positive_log_gate():
    folds = np.arange(20) % 2
    reference_error = np.ones(20)
    candidate_error = np.full(20, 0.5)
    reference_body = {
        "pooled": {"n": 20, "mae": 1.0},
        "folds": {"0": {"n": 10, "mae": 1.0}, "1": {"n": 10, "mae": 1.0}},
    }
    candidate_body = {
        "pooled": {"n": 20, "mae": 1.0},
        "folds": {"0": {"n": 10, "mae": 0.9}, "1": {"n": 10, "mae": 1.1}},
    }
    bootstrap = {"lower_bonferroni": 0.1}
    result = _comparison(
        "strong_control", reference_error, candidate_error, folds,
        reference_body, candidate_body, bootstrap,
    )
    assert all(
        result["conditions"][name]
        for name in (
            "pooled_log_mse_gain_at_least_one_percent",
            "both_fold_log_mse_improvements_positive",
            "bonferroni_series_bootstrap_lower_positive",
        )
    )
    assert not result["conditions"]["candidate_body_mae_noninferior_pooled_and_both_folds"]
    assert not result["passes"]
    _, original_t0_guard = _body_noninferior(candidate_body, reference_body)
    assert not original_t0_guard


def test_underpowered_valid_cache_is_unassessable_and_not_fitted():
    summary, arrays = analyze_cache(_cache(n=1000), bootstrap_repeats=2)
    assert summary["status"] == "UNASSESSABLE"
    assert not summary["assessable"] and not summary["gate"]["passes"]
    assert "fewer_than_512_targets" in summary["folds"]["0"]["reasons"]
    assert set(summary["arms"]) == {"original_t0"}
    assert "candidate__log_prediction" not in arrays
    json.dumps(summary, allow_nan=False)

    too_few_series = _cache()
    too_few_series["series_id"] = too_few_series["fold"]
    summary, _ = analyze_cache(too_few_series, bootstrap_repeats=2)
    assert not summary["assessable"]
    assert "fewer_than_30_series" in summary["folds"]["0"]["reasons"]


def test_results_are_bitwise_deterministic_and_global_rng_is_unchanged():
    data = _cache()
    numpy_state = np.random.get_state()
    first, first_arrays = analyze_cache(data, bootstrap_repeats=80)
    second, second_arrays = analyze_cache(data, bootstrap_repeats=80)
    assert first == second
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
    after_state = np.random.get_state()
    assert numpy_state[0] == after_state[0]
    np.testing.assert_array_equal(numpy_state[1], after_state[1])
    assert numpy_state[2:] == after_state[2:]


def test_aggregate_requires_all_named_datasets_and_both_comparisons():
    passed, _ = analyze_cache(_cache(), bootstrap_repeats=80)
    results = {dataset: copy.deepcopy(passed) for dataset in DATASETS}
    aggregate = aggregate_common_gate(results)
    assert aggregate["passes"] and aggregate["status"] == "PASS"

    results["yellow_trip_hourly"]["gate"]["comparisons"]["sham"]["passes"] = False
    results["yellow_trip_hourly"]["gate"]["passes"] = False
    results["yellow_trip_hourly"]["status"] = "FAIL"
    aggregate = aggregate_common_gate(results)
    assert not aggregate["passes"] and not aggregate["comparisons"]["sham"]["passes"]

    underpowered, _ = analyze_cache(_cache(n=1000), bootstrap_repeats=2)
    results = {dataset: copy.deepcopy(passed) for dataset in DATASETS}
    results["intermittent_v2"] = underpowered
    aggregate = aggregate_common_gate(results)
    assert not aggregate["passes"] and not aggregate["datasets"]["intermittent_v2"]["assessable"]

    contradictory = {dataset: copy.deepcopy(passed) for dataset in DATASETS}
    contradictory["intermittent_v2"]["status"] = "FAIL"
    with pytest.raises(ValueError, match="inconsistent"):
        aggregate_common_gate(contradictory)
    foreign = {dataset: copy.deepcopy(passed) for dataset in DATASETS}
    foreign["intermittent_v2"]["protocol"] = "some_other_analysis"
    with pytest.raises(ValueError, match="protocol"):
        aggregate_common_gate(foreign)
    with pytest.raises(ValueError, match="exactly"):
        aggregate_common_gate({"yellow_trip_hourly": passed})
