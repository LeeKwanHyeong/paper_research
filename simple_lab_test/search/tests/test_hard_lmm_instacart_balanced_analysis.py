"""Synthetic staged-design contracts; no real rows, models or caches loaded."""
import json

import numpy as np
import pytest

from paper.scripts.hard_lmm_instacart_balanced_analysis import body_recheck, design, evaluate


def fixture(n=2048):
    rows = np.arange(n)
    # Every fold has the same J pattern at two exact history lengths.
    j = np.tile(np.array([0., 0., .1, .2, .3, .4, .5, .5]), n // 8)
    return dict(age_distortion=j, mean_log_quantity=np.full(n, 2.),
        history_length=np.where((rows // 8) % 2, 12, 8), count_change_rms=np.full(n, .2),
        log_mean_internal_gap=np.full(n, 1.), series_index=rows, fold=(rows // 16) % 2)


def labels(features, assignment):
    result = np.full(len(features["fold"]), np.nan)
    result[assignment["arm"] >= 0] = 10.
    return result


def predictions(features, assignment, *, excess=1.):
    retained = assignment["rowweight"] > 0
    quantity = np.full(len(retained), np.nan)
    quantity[retained] = 10.
    error = np.where(assignment["arm"] == 1, 1. + excess, 1.)
    prediction = quantity - error
    return {name: dict(quantity=quantity.copy(), prediction=prediction.copy(),
                       log_residual=np.log1p(quantity) - np.log1p(prediction))
            for name in ("original", "separate_key")}


def test_full_staged_positive_design_exact_H_balance_and_error_gate():
    features = fixture()
    plan, initial = design(features)
    assert plan["passes"]
    assert all(f["balance"]["controls"]["history_length"]["smd"] == 0 for f in plan["folds"].values())
    checked, body = body_recheck(features, initial, labels(features, initial), plan)
    assert checked["passes"]
    assert not np.any(body["body_admitted"] & ~body["input_admitted"])
    result = evaluate(features, body, predictions(features, body))
    assert result["relevance_gate"]["passes"]
    assert result["pooled"]["separate_key"]["raw_body_mae"]["relative_excess"] == pytest.approx(1.)
    assert result["cluster_bootstrap"]["lower_p05_diff"] == pytest.approx(1.)
    json.dumps([plan, checked, result], allow_nan=False)


def test_input_design_has_no_label_api_and_failed_design_never_reads_labels():
    features = fixture()
    with pytest.raises(TypeError):
        design(features, target_quantities=np.ones(2048))
    features["age_distortion"][:] = 0.
    plan, initial = design(features)
    class Unread:
        def __array__(self, *args, **kwargs):
            raise AssertionError("labels must remain unread")
    with pytest.raises(ValueError, match="before body-label"):
        body_recheck(features, initial, Unread(), plan)


def test_input_covariate_imbalance_fails_even_with_exact_H():
    features = fixture()
    features["count_change_rms"] = np.tile([0., 0., 2., 2., 2., 2., 1., 1.], 256)
    plan, _ = design(features)
    assert not plan["passes"]
    assert "insufficient_covariate_balance" in plan["folds"]["0"]["reasons"]


def test_body_cannot_add_cells_and_denominator_includes_unsupported_pool_extremes():
    features = fixture(n=4096)
    # Twenty rows have singleton exact-H cells and cannot enter input support.
    features["history_length"][:20] = np.arange(100, 120)
    plan, initial = design(features)
    assert plan["passes"]
    quantities = labels(features, initial)
    checked, body = body_recheck(features, initial, quantities, plan)
    assert not body["body_admitted"][:20].any()
    for fold in (0, 1):
        expected = int(((initial["arm"] >= 0) & (features["fold"] == fold)).sum())
        assert checked["folds"][str(fold)]["all_pool_eligible_extremes"] == expected
    assert np.array_equal(initial["cell"], body["cell"])
    assert np.array_equal(initial["arm"], body["arm"])
    assert np.array_equal(initial["input_admitted"], body["input_admitted"])


def test_body_filter_can_destroy_support_and_predictions_remain_unread():
    features = fixture()
    plan, initial = design(features)
    quantities = labels(features, initial)
    quantities[initial["arm"] == 1] = 26.
    checked, body = body_recheck(features, initial, quantities, plan)
    assert not checked["passes"]
    class Unread(dict):
        def __iter__(self):
            raise AssertionError("predictions must remain unread")
    with pytest.raises(ValueError, match="before prediction/error"):
        evaluate(features, body, Unread())


def test_body_filter_can_break_other_control_balance_despite_exact_H():
    features = fixture(n=4096)
    rows = np.arange(4096)
    extreme = (features["age_distortion"] == 0) | (features["age_distortion"] == .5)
    features["count_change_rms"] = np.where(extreme, (rows // 32) % 2, 2.)
    plan, initial = design(features)
    assert plan["passes"]
    quantities = labels(features, initial)
    quantities[(initial["arm"] == 1) & (features["count_change_rms"] == 1)] = 26.
    checked, body = body_recheck(features, initial, quantities, plan)
    assert not checked["passes"]
    assert "insufficient_covariate_balance" in checked["folds"]["0"]["reasons"]
    assert checked["folds"]["0"]["balance"]["controls"]["history_length"]["smd"] == 0.
    assert np.array_equal(initial["cell"], body["cell"])


def test_small_excess_fails_frozen_five_percent_error_gate():
    features = fixture()
    plan, initial = design(features)
    _, body = body_recheck(features, initial, labels(features, initial), plan)
    result = evaluate(features, body, predictions(features, body, excess=.04))
    assert not result["relevance_gate"]["passes"]
    assert not result["relevance_gate"]["conditions"]["separate_key_pooled_excess_at_least_minimum"]


@pytest.mark.parametrize("veto", ["original_direction", "candidate_one_fold"])
def test_direction_requirements_are_independent_vetoes(veto):
    features = fixture()
    plan, initial = design(features)
    _, body = body_recheck(features, initial, labels(features, initial), plan)
    caches = predictions(features, body)
    target_model = "original" if veto == "original_direction" else "separate_key"
    retained = body["rowweight"] > 0
    selected = retained if veto == "original_direction" else retained & (features["fold"] == 1)
    caches[target_model]["prediction"][selected] = 10 - np.where(body["arm"][selected] == 1, .5, 1.)
    caches[target_model]["log_residual"][selected] = np.log1p(10.) - np.log1p(caches[target_model]["prediction"][selected])
    result = evaluate(features, body, caches)
    assert not result["relevance_gate"]["passes"]
    key = "original_both_folds_positive" if veto == "original_direction" else "separate_key_both_folds_positive"
    assert not result["relevance_gate"]["conditions"][key]


def test_unselected_prediction_and_label_values_are_not_scored():
    features = fixture()
    plan, initial = design(features)
    _, body = body_recheck(features, initial, labels(features, initial), plan)
    caches = predictions(features, body)
    before = evaluate(features, body, caches)
    outside = body["rowweight"] == 0
    for cache in caches.values():
        for values in cache.values():
            values[outside] = float("inf")
    after = evaluate(features, body, caches)
    assert before == after


def test_body_uses_labels_for_every_extreme_and_frozen_input_digests():
    features = fixture()
    plan, initial = design(features)
    quantities = labels(features, initial)
    quantities[np.flatnonzero(initial["arm"] >= 0)[0]] = np.nan
    with pytest.raises(ValueError, match="ALL pool extremes"):
        body_recheck(features, initial, quantities, plan)
    changed = {name: value.copy() for name, value in initial.items()}
    changed["arm"][0] = 1
    with pytest.raises(ValueError, match="frozen input"):
        body_recheck(features, changed, labels(features, initial), plan)


def test_unique_series_and_frozen_thresholds_are_enforced():
    features = fixture()
    features["series_index"][1] = features["series_index"][0]
    with pytest.raises(ValueError, match="unique series"):
        design(features)
    with pytest.raises(ValueError, match="frozen policy"):
        design(fixture(), policy={"body_threshold": 26})
    with pytest.raises(ValueError, match="frozen policy"):
        design(fixture(), policy={"max_smd": .3})
