"""Synthetic checks; no real histories, cache results or fitting are used."""
import json

import numpy as np
import pytest

from paper.scripts.hard_lmm_temporal_analysis import analyze


def fixture(n=2048):
    rows = np.arange(n)
    series = rows // 8
    fold = series % 2
    j = np.tile(np.array([.0, .0, .1, .2, .3, .4, .5, .5]), n // 8)
    features = dict(age_distortion=j, mean_log_quantity=np.full(n, 2.),
        history_length=np.full(n, 12), count_change_rms=np.full(n, .2),
        log_mean_internal_gap=np.full(n, 1.), series_index=series, fold=fold)
    quantity = np.full(n, 10.)
    caches = {}
    for model in ("original", "separate_key"):
        prediction = quantity - (1 + 2 * j)
        caches[model] = dict(prediction=prediction, quantity=quantity.copy(),
                            log_residual=np.log1p(quantity) - np.log1p(prediction))
    return features, caches


def test_controlled_positive_association_passes_and_is_reproducible():
    features, caches = fixture()
    summary, assignments = analyze(features, caches, 25)
    repeat, repeated_assignments = analyze(features, caches, 25)
    assert summary == repeat
    assert summary["assessable"] and summary["relevance_gate"]["passes"]
    assert summary["pooled"]["separate_key"]["raw_body_mae"]["relative_excess"] == pytest.approx(1.)
    assert summary["cluster_bootstrap"]["lower_p05_diff"] == pytest.approx(1.)
    for key in assignments:
        assert np.array_equal(assignments[key], repeated_assignments[key])
    json.dumps(summary, allow_nan=False)


def test_balance_failure_blocks_association_even_inside_same_coarse_cell():
    features, caches = fixture()
    # Both high/low values occupy the lower median cell, but are imbalanced there.
    pattern = np.tile(np.array([0., 0., 2., 2., 2., 2., 1., 1.]), 256)
    features["count_change_rms"] = pattern
    summary, _ = analyze(features, caches, 25)
    assert not summary["assessable"] and not summary["relevance_gate"]["passes"]
    assert "insufficient_covariate_balance" in summary["folds"]["0"]["reasons"]
    json.dumps(summary, allow_nan=False)


def test_confounding_without_common_support_holds():
    features, caches = fixture()
    features["mean_log_quantity"] = 1 + features["age_distortion"]
    summary, assignments = analyze(features, caches, 25)
    assert not summary["assessable"]
    assert not assignments["weight"].any()
    assert not summary["relevance_gate"]["passes"]


def test_constant_J_holds_without_fallback():
    features, caches = fixture()
    features["age_distortion"][:] = 0.
    summary, assignment = analyze(features, caches, 25)
    assert not summary["assessable"]
    assert summary["cluster_bootstrap"] is None
    assert (assignment["arm"] == -1).all()
    assert "no_J_separation" in summary["folds"]["0"]["reasons"]
    json.dumps(summary, allow_nan=False)


def test_unequal_arm_counts_get_equal_cell_mass_without_label_fits():
    features, caches = fixture()
    features["age_distortion"] = np.tile(np.array([0., 0., .1, .2, .3, .5, .5, .5]), 256)
    _, assignment = analyze(features, caches, 25)
    assert np.all(assignment["weight"][assignment["arm"] == 0] == 1.)
    assert np.all(assignment["weight"][assignment["arm"] == 1] == 2 / 3)
    for fold in (0, 1):
        selected = assignment["fold"] == fold
        low = assignment["weight"][selected & (assignment["arm"] == 0)].sum()
        high = assignment["weight"][selected & (assignment["arm"] == 1)].sum()
        assert low == pytest.approx(high)


def test_ineligible_nan_J_is_excluded_and_all_ineligible_is_strict_json():
    features, caches = fixture()
    features["history_length"][:] = 2
    features["age_distortion"][:] = np.nan
    summary, assignment = analyze(features, caches, 25)
    assert summary["excluded"]["invalid_history"] == 2048
    assert not summary["assessable"]
    assert (assignment["arm"] == -1).all()
    assert not assignment["weight"].any()
    json.dumps(summary, allow_nan=False)


def test_sparse_independent_series_holds_even_with_many_rows():
    features, caches = fixture()
    features["series_index"] = features["fold"].copy()
    summary, _ = analyze(features, caches, 25)
    assert not summary["assessable"]
    assert "insufficient_arm_series" in summary["folds"]["0"]["reasons"]


def test_label_changes_do_not_change_input_cutpoints_or_assignments():
    features, caches = fixture()
    before, before_rows = analyze(features, caches, 25)
    for cache in caches.values():
        cache["quantity"][:] = 100.
        cache["prediction"][:] = 90.
        cache["log_residual"][:] = np.log1p(100.) - np.log1p(90.)
    after, after_rows = analyze(features, caches, 25)
    assert before["cuts"] == after["cuts"]
    for name in ("cell", "arm"):
        assert np.array_equal(before_rows[name], after_rows[name])
    assert not after["assessable"]


def test_held_inputs_do_not_change_their_own_cutpoints():
    features, caches = fixture()
    before, _ = analyze(features, caches, 25)
    features["mean_log_quantity"][features["fold"] == 0] += 100.
    after, _ = analyze(features, caches, 25)
    assert before["cuts"]["0"] == after["cuts"]["0"]


@pytest.mark.parametrize("fault", ["nonfinite", "mixed_series", "different_labels"])
def test_bad_inputs_rejected(fault):
    features, caches = fixture()
    if fault == "nonfinite":
        features["mean_log_quantity"][0] = np.nan
    elif fault == "mixed_series":
        features["series_index"][8] = 0
    else:
        caches["separate_key"]["quantity"][0] += 1
    with pytest.raises(ValueError):
        analyze(features, caches, 25)
