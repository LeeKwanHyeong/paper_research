"""Tests for frozen raw affine calibration primitives."""

import copy
import json

import numpy as np
import pytest

from paper.scripts.frozen_raw_affine_calibration import (
    BOOTSTRAP_PROTOCOL,
    FOLD_SALT,
    IDENTITY_CALIBRATION,
    MINIMUM_SLOPE,
    apply_affine_calibration,
    apply_quantity_only_bundle,
    assign_series_folds,
    compute_reporting_metrics,
    compute_stratified_metrics,
    fit_positive_affine_ols,
    identity_calibration,
    series_clustered_paired_bootstrap_ci,
    two_fold_oof_audit,
    validate_series_disjoint_folds,
)


def _oof_data() -> dict:
    series = np.repeat(np.arange(40), 5)
    fold = series % 2
    prediction = np.linspace(1.0, 50.0, len(series))
    target = 1.35 * prediction + 2.25
    return {
        "prediction": prediction,
        "target": target,
        "series": series,
        "fold": fold,
        "history": 1 + np.arange(len(series)) % 12,
    }


def _audit(data: dict) -> tuple[dict, dict[str, np.ndarray]]:
    return two_fold_oof_audit(
        data["prediction"],
        data["target"],
        data["series"],
        fold=data["fold"],
        body_threshold=55.0,
        extreme_tail_threshold=65.0,
        quantity_boundaries=(20.0, 40.0, 50.0, 65.0),
        history_length=data["history"],
        history_boundaries=(2, 5, 9),
        history_labels=("le2", "3to5", "6to9", "gt9"),
    )


def test_identity_is_exact_and_returns_fresh_parameters():
    prediction = np.asarray([0.0, 1.0, 5.5, 100.0], dtype=np.float64)
    calibrated = apply_affine_calibration(prediction, IDENTITY_CALIBRATION)
    np.testing.assert_array_equal(calibrated, prediction)

    first = identity_calibration()
    first["slope"] = 2.0
    assert identity_calibration() == {"slope": 1.0, "intercept": 0.0}


def test_quantity_bundle_changes_quantity_and_copies_time_exactly():
    prediction = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    time_nll = np.asarray([-1.0, 0.5, 2.0], dtype=np.float64)
    bundle = apply_quantity_only_bundle(
        prediction,
        time_nll,
        {"slope": 2.0, "intercept": 1.0},
    )
    np.testing.assert_array_equal(bundle["quantity_prediction"], [3.0, 5.0, 7.0])
    np.testing.assert_array_equal(bundle["time_nll"], time_nll)
    assert bundle["time_nll"] is not time_nll


def test_positive_affine_ols_recovers_known_raw_map():
    prediction = np.linspace(0.0, 20.0, 101)
    target = 1.75 * prediction + 3.5
    fit = fit_positive_affine_ols(prediction, target)

    assert fit["slope"] == pytest.approx(1.75, abs=1e-14)
    assert fit["intercept"] == pytest.approx(3.5, abs=1e-14)
    assert not fit["slope_was_bounded"]
    assert fit["train_affine_sse"] == pytest.approx(0.0, abs=1e-24)
    np.testing.assert_allclose(
        apply_affine_calibration(prediction, fit), target, rtol=0, atol=1e-13
    )
    json.dumps(fit, allow_nan=False)


def test_nonpositive_unconstrained_slope_uses_fixed_positive_boundary():
    prediction = np.arange(5.0)
    target = prediction[::-1].copy()
    fit = fit_positive_affine_ols(prediction, target)

    assert fit["unconstrained_slope"] < 0
    assert fit["slope"] == MINIMUM_SLOPE
    assert fit["slope_was_bounded"]
    assert fit["intercept"] == pytest.approx(target.mean() - MINIMUM_SLOPE * prediction.mean())


def test_application_clamps_negative_raw_predictions_and_rejects_bad_parameters():
    prediction = np.asarray([0.0, 1.0, 4.0])
    calibrated = apply_affine_calibration(
        prediction, {"slope": 2.0, "intercept": -3.0}
    )
    np.testing.assert_array_equal(calibrated, [0.0, 0.0, 5.0])

    with pytest.raises(ValueError, match="positive"):
        apply_affine_calibration(prediction, {"slope": 0.0, "intercept": 0.0})
    with pytest.raises(ValueError, match="slope and intercept"):
        apply_affine_calibration(prediction, {"slope": 1.0, "extra": 2.0})
    with pytest.raises(ValueError, match="nonnegative"):
        apply_affine_calibration([-1.0], IDENTITY_CALIBRATION)
    with pytest.raises(ValueError, match="finite"):
        fit_positive_affine_ols([0.0, np.nan], [0.0, 1.0])


def test_series_fold_assignment_is_deterministic_and_series_disjoint():
    series = np.asarray(["p1", "p2", "p1", "p3", "p2", "p4"])
    first = assign_series_folds(series)
    second = assign_series_folds(series, salt=FOLD_SALT)
    np.testing.assert_array_equal(first, second)
    assert first[0] == first[2]
    assert first[1] == first[4]

    _, normalized = validate_series_disjoint_folds(series, first)
    np.testing.assert_array_equal(normalized, first)
    crossed = first.copy()
    crossed[2] = 1 - crossed[0]
    with pytest.raises(ValueError, match="disjoint"):
        validate_series_disjoint_folds(series, crossed)

    data = _oof_data()
    _, arrays = two_fold_oof_audit(
        data["prediction"],
        data["target"],
        data["series"],
        body_threshold=55.0,
        extreme_tail_threshold=65.0,
    )
    np.testing.assert_array_equal(arrays["fold"], assign_series_folds(data["series"]))


def test_strata_are_lower_inclusive_and_empty_rows_are_json_safe():
    target = np.asarray([1.0, 2.0, 3.0, 5.0])
    prediction = np.asarray([2.0, 2.0, 2.0, 2.0])
    rows = compute_stratified_metrics(
        prediction,
        target,
        target,
        boundaries=(2.0, 3.0, 4.0),
        labels=("le2", "le3", "le4", "gt4"),
    )
    assert [row["count"] for row in rows] == [2, 1, 0, 1]
    assert rows[0]["mae"] == pytest.approx(0.5)
    assert rows[2] == {
        "stratum": "le4",
        "stratum_order": 2,
        "count": 0,
        "mae": None,
        "mse": None,
        "rmse": None,
        "bias": None,
    }
    json.dumps(rows, allow_nan=False)


def test_reporting_metrics_use_train_threshold_contract():
    target = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    prediction = target + np.asarray([1.0, -1.0, 2.0, -2.0, 4.0])
    report = compute_reporting_metrics(
        prediction,
        target,
        body_threshold=8.0,
        extreme_tail_threshold=8.0,
        quantity_boundaries=(2.0, 4.0, 8.0, 12.0),
    )
    assert report["overall"]["count"] == 5
    assert report["body_le_p95"]["count"] == 4
    assert report["gt_p99"]["count"] == 1
    assert [row["count"] for row in report["quantity_strata"]] == [2, 1, 1, 0, 1]


def test_two_fold_oof_audit_recovers_signal_in_both_folds():
    summary, arrays = _audit(_oof_data())

    assert summary["protocol"] == "frozen_raw_affine_calibration_v1"
    assert summary["scope"] == "train_internal_series_disjoint_two_fold_oof"
    assert summary["fold_rule"] == "caller_supplied_series_disjoint"
    assert summary["both_folds_raw_mse_improve"]
    assert summary["overall_change"]["mse_relative_improvement"] > 0.999999
    np.testing.assert_allclose(arrays["oof_calibrated_prediction"], arrays["target"], atol=1e-12)
    for held_fold in (0, 1):
        fit = summary["fits_by_held_fold"][str(held_fold)]
        assert fit["held_fold"] == held_fold
        assert fit["fit_fold"] == 1 - held_fold
        assert fit["slope"] == pytest.approx(1.35)
        assert fit["intercept"] == pytest.approx(2.25)
        assert summary["folds"][str(held_fold)]["overall_change"][
            "mse_relative_improvement"
        ] > 0
    json.dumps(summary, allow_nan=False)


def test_held_fold_targets_do_not_enter_its_fit_or_prediction():
    original = _oof_data()
    before, before_arrays = _audit(original)
    changed = copy.deepcopy(original)
    held = original["fold"] == 0
    changed["target"][held] += 1000.0
    after, after_arrays = _audit(changed)

    assert before["fits_by_held_fold"]["0"] == after["fits_by_held_fold"]["0"]
    np.testing.assert_array_equal(
        before_arrays["oof_calibrated_prediction"][held],
        after_arrays["oof_calibrated_prediction"][held],
    )


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("crossed_series", "disjoint"),
        ("one_fold", "exactly"),
        ("short", "at least four"),
        ("negative_target", "nonnegative"),
        ("bad_thresholds", "ordered"),
        ("incomplete_history", "provided together"),
    ],
)
def test_invalid_oof_contract_fails_closed(fault, message):
    data = _oof_data()
    kwargs = dict(
        fold=data["fold"],
        body_threshold=55.0,
        extreme_tail_threshold=65.0,
    )
    if fault == "crossed_series":
        data["fold"] = data["fold"].copy()
        data["fold"][1] = 1 - data["fold"][0]
        kwargs["fold"] = data["fold"]
    elif fault == "one_fold":
        kwargs["fold"] = np.zeros_like(data["fold"])
    elif fault == "short":
        for key in data:
            data[key] = data[key][:3]
        kwargs["fold"] = data["fold"]
    elif fault == "negative_target":
        data["target"] = data["target"].copy()
        data["target"][0] = -1.0
    elif fault == "bad_thresholds":
        kwargs["body_threshold"] = 100.0
        kwargs["extreme_tail_threshold"] = 50.0
    else:
        kwargs["history_length"] = data["history"]

    with pytest.raises(ValueError, match=message):
        two_fold_oof_audit(
            data["prediction"], data["target"], data["series"], **kwargs
        )


def test_oof_result_is_bitwise_deterministic_without_global_rng_changes():
    data = _oof_data()
    numpy_state = np.random.get_state()
    first, first_arrays = _audit(data)
    second, second_arrays = _audit(data)
    assert first == second
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])
    after_state = np.random.get_state()
    assert numpy_state[0] == after_state[0]
    np.testing.assert_array_equal(numpy_state[1], after_state[1])
    assert numpy_state[2:] == after_state[2:]


def _naive_series_clustered_bootstrap(
    baseline, candidate, target, series, *, seed, replicates
):
    unique_series = np.unique(series)
    mae = np.empty(replicates, dtype=np.float64)
    squared_error = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, replicate]))
        sampled = unique_series[
            rng.integers(0, len(unique_series), size=len(unique_series))
        ]
        row_indices = np.concatenate(
            [np.flatnonzero(series == sampled_series) for sampled_series in sampled]
        )
        baseline_error = baseline[row_indices] - target[row_indices]
        candidate_error = candidate[row_indices] - target[row_indices]
        mae[replicate] = np.mean(
            np.abs(candidate_error) - np.abs(baseline_error), dtype=np.float64
        )
        squared_error[replicate] = np.mean(
            np.square(candidate_error) - np.square(baseline_error), dtype=np.float64
        )
    return mae, squared_error


def test_series_clustered_paired_bootstrap_matches_naive_cluster_resampling():
    series = np.asarray([10, 10, 10, 20, 30, 30, 40, 40, 40, 40])
    target = np.asarray([2.0, 4.0, 8.0, 3.0, 2.0, 9.0, 1.0, 3.0, 7.0, 12.0])
    baseline = target + np.asarray([2.0, -1.0, 3.0, 4.0, -2.0, 2.0, 5.0, -3.0, 1.0, 4.0])
    candidate = target + np.asarray([1.0, -0.5, 1.5, 2.0, -1.0, 1.0, 2.5, -1.5, 0.5, 2.0])
    seed = 271828
    replicates = 101

    result = series_clustered_paired_bootstrap_ci(
        baseline,
        candidate,
        target,
        series,
        seed=seed,
        replicates=replicates,
        draw_chunk_size=2,
    )
    naive_mae, naive_squared_error = _naive_series_clustered_bootstrap(
        baseline,
        candidate,
        target,
        series,
        seed=seed,
        replicates=replicates,
    )

    assert result["protocol"] == BOOTSTRAP_PROTOCOL
    assert result["delta_definition"] == "candidate_minus_baseline"
    assert result["rows"] == len(target)
    assert result["series"] == 4
    np.testing.assert_allclose(
        result["mae_delta_candidate_minus_baseline"]["bootstrap_mean"],
        naive_mae.mean(),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        [
            result["mae_delta_candidate_minus_baseline"]["ci_95_percentile"]["lower"],
            result["mae_delta_candidate_minus_baseline"]["ci_95_percentile"]["upper"],
        ],
        np.quantile(naive_mae, [0.025, 0.975], method="linear"),
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        [
            result["squared_error_delta_candidate_minus_baseline"]["ci_95_percentile"]["lower"],
            result["squared_error_delta_candidate_minus_baseline"]["ci_95_percentile"]["upper"],
        ],
        np.quantile(naive_squared_error, [0.025, 0.975], method="linear"),
        rtol=0,
        atol=1e-14,
    )
    assert result["mae_delta_candidate_minus_baseline"]["point_estimate"] < 0
    assert result["squared_error_delta_candidate_minus_baseline"]["point_estimate"] < 0
    json.dumps(result, allow_nan=False)


def test_series_clustered_paired_bootstrap_is_deterministic_and_chunk_bounded():
    # The number of stored bootstrap values depends only on replicates; sampled
    # series are streamed through the explicitly bounded buffer.  This uses the
    # contract's expected high-cardinality audit size rather than a toy proxy.
    number_of_series = 200_000
    target = 1.0 + np.arange(number_of_series, dtype=np.float64) % 17
    baseline = target + 2.0
    candidate = target + 1.0
    series = np.arange(number_of_series)
    kwargs = {"seed": 42, "replicates": 500, "draw_chunk_size": 257}
    numpy_state = np.random.get_state()

    first = series_clustered_paired_bootstrap_ci(
        baseline, candidate, target, series, **kwargs
    )
    second = series_clustered_paired_bootstrap_ci(
        baseline, candidate, target, series, **kwargs
    )

    assert first == second
    assert first["implementation"] == {
        "aggregation": "per_series_sufficient_statistics",
        "replicate_streaming": True,
        "draw_chunk_size": 257,
        "maximum_sampled_series_buffer": 257,
    }
    assert first["mae_delta_candidate_minus_baseline"]["point_estimate"] == -1.0
    assert (
        first["squared_error_delta_candidate_minus_baseline"]["point_estimate"]
        == -3.0
    )
    after_state = np.random.get_state()
    assert numpy_state[0] == after_state[0]
    np.testing.assert_array_equal(numpy_state[1], after_state[1])
    assert numpy_state[2:] == after_state[2:]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidate_prediction": [1.0]}, "row count mismatch"),
        ({"target": [1.0, np.inf]}, "finite"),
        ({"series_id": [1, 1]}, "distinct series"),
        ({"seed": -1}, "between"),
        ({"seed": True}, "integer"),
        ({"replicates": 1}, "at least 2"),
        ({"draw_chunk_size": 0}, "at least 1"),
    ],
)
def test_series_clustered_paired_bootstrap_rejects_invalid_contract(kwargs, message):
    arguments = {
        "baseline_prediction": [2.0, 3.0],
        "candidate_prediction": [1.0, 2.0],
        "target": [1.0, 1.0],
        "series_id": [1, 2],
        "seed": 7,
        "replicates": 10,
    }
    arguments.update(kwargs)
    with pytest.raises((ValueError, FloatingPointError), match=message):
        series_clustered_paired_bootstrap_ci(**arguments)
