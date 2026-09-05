"""Synthetic checks only: no actual-data fitting or candidate selection."""
import copy
import json

import numpy as np
import pytest

from paper.scripts.hard_lmm_information_analysis import (
    CONTRASTS, FAMILY_QUANTILE, SEED, SHAM_SEED, aggregate_common_evidence, analyze_cache,
    evaluate_gate, fit_predict, oof_predict, raw_metrics,
)


def fixture(n=160):
    rng = np.random.default_rng(145)
    features = rng.normal(size=(n, 12))
    latent = rng.normal(size=(n, 64))
    # F0 is available early, deliberately absent from the last encoder stage.
    early = latent.copy()
    early[:, 0] = features[:, 0]
    log_qty = 2. + .4 * features[:, 0]
    stages = dict(input_last=early, input_mean=early.copy(),
                  layer1_last=early.copy(), layer1_mean=early.copy(),
                  layer2_last=latent.copy(), layer2_mean=latent.copy(),
                  h=latent.copy(), r=np.zeros_like(latent), fused=latent.copy())
    stages["concat_hr"] = np.concatenate((stages["h"], stages["r"]), axis=1)
    return dict(**stages, history_features=features, quantity=np.expm1(log_qty),
                log_quantity=log_qty, base_logpred=np.full(n, 2.), series_id=np.arange(n),
                fold=np.arange(n) % 2, body_threshold=100., tail_threshold=200.)


@pytest.mark.parametrize("family", ["linear", "random128"])
def test_training_only_statistics_and_held_labels_do_not_affect_predictions(family):
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=(100, 4)), rng.normal(size=100)
    folds = np.arange(100) % 2
    before, meta = oof_predict(x, y, folds, family)
    changed_y = y.copy()
    changed_y[folds == 0] += 1000.
    after, changed_meta = oof_predict(x, changed_y, folds, family)
    np.testing.assert_array_equal(before[folds == 0], after[folds == 0])
    assert meta["0"] == changed_meta["0"]
    changed_x = x.copy()
    changed_x[folds == 0] += 9999.
    _, changed_meta = oof_predict(changed_x, y, folds, family)
    assert meta["0"] == changed_meta["0"]
    np.testing.assert_allclose(meta["0"]["input_mean"], x[folds == 1].mean(axis=0))


@pytest.mark.parametrize("family", ["linear", "random128"])
def test_constant_collinear_reproducible_finite_and_intercept_unpenalized(family):
    x = np.column_stack((np.ones(60), np.arange(60), np.arange(60)))
    y = np.full(60, 3.14)
    pred, meta = fit_predict(x, y, x, family)
    repeated, repeated_meta = fit_predict(x, y, x, family)
    np.testing.assert_allclose(pred, y, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(pred, repeated)
    assert meta == repeated_meta and meta["input_scale"][0] == 1.
    assert np.isfinite(pred).all()


def test_sse_ridge_unit_and_unpenalized_intercept_match_closed_form():
    x = np.arange(10, dtype=float).reshape(-1, 1)
    y = 8. + 3. * x[:, 0]
    pred, _ = fit_predict(x, y, x)
    standardized = (x[:, 0] - x.mean()) / x.std()
    coef = standardized @ (y - y.mean()) / (standardized @ standardized + 1.)
    np.testing.assert_allclose(pred, standardized * coef + y.mean(), rtol=1e-12)


def test_positive_known_feature_reconstruction_and_json():
    cache = fixture()
    result, arrays = analyze_cache(cache, bootstrap_repeats=20)
    assert result["history_usefulness_passes"]
    assert result["reconstruction"]["input_mean"]["per_feature_oof_r2"][0] > .9
    assert result["reconstruction"]["h"]["per_feature_oof_r2"][0] < .1
    assert result["reconstruction"]["Ftruth"]["per_feature_mse"] == [0.] * 12
    np.testing.assert_array_equal(arrays["reconstruct__Ftruth"], cache["history_features"])
    assert result["decoders"]["linear"]["h+F"]["residual_mse"]["pooled"] < result["decoders"]["linear"]["h"]["residual_mse"]["pooled"]
    json.dumps(result, allow_nan=False)


def test_permuted_labels_do_not_pass_history_usefulness():
    cache = fixture(240)
    rng = np.random.default_rng(666)
    permutation = rng.permutation(len(cache["quantity"]))
    cache["quantity"] = cache["quantity"][permutation]
    cache["log_quantity"] = cache["log_quantity"][permutation]
    result, _ = analyze_cache(cache, bootstrap_repeats=20)
    assert not result["history_usefulness_passes"]
    assert not result["contrasts"]["history_at_h"]["passes"]


def test_gate_exact_threshold_fold_sign_and_body_conflict():
    folds, series = np.arange(20) % 2, np.arange(20)
    ref = np.full(20, 100.)
    result = evaluate_gate(ref, np.full(20, 99.), folds, series, bootstrap_repeats=20)
    assert result["passes"] and result["relative_gain"] == .01
    below = evaluate_gate(ref, np.full(20, 99.000001), folds, series, bootstrap_repeats=20)
    assert not below["passes"]
    candidate = np.where(folds == 0, 90., 101.)
    one_fold = evaluate_gate(ref, candidate, folds, series, bootstrap_repeats=20)
    assert one_fold["relative_gain"] > .01 and not one_fold["passes"]
    conflict = evaluate_gate(ref, np.full(20, 90.), folds, series, bootstrap_repeats=20,
                             reference_body_mae=1., candidate_body_mae=1.001)
    assert conflict["log_vs_body_direction_conflict"] and not conflict["passes"]
    assert FAMILY_QUANTILE == .05 / 32


def test_raw_report_clipping_does_not_change_primary_and_tail_is_strict():
    folds = np.array([0, 1, 0, 1])
    qty = np.array([1., 2., 3., 4.])
    result = raw_metrics(np.array([-2., 21., 1., 2.]), qty, folds, 2., 3.)
    assert result["clamps"] == dict(below_zero=1, above_twenty=1)
    assert result["pooled"]["body_n"] == 2 and result["pooled"]["tail_n"] == 1


def test_crossfold_series_leakage_rejected():
    cache = fixture()
    cache["series_id"] = np.arange(len(cache["quantity"])) // 2
    with pytest.raises(ValueError, match="disjoint"):
        analyze_cache(cache, bootstrap_repeats=2)


def test_common_evidence_requires_identical_contrast_all_four_cells_and_usefulness():
    result = dict(history_usefulness_passes=True,
                  contrasts={name: dict(passes=True) for name in CONTRASTS})
    full = {ds: {model: copy.deepcopy(result) for model in ("original", "separate_key")}
            for ds in ("yellow_trip_hourly", "insta_market_basket")}
    assert aggregate_common_evidence(full)["any_common_evidence"]
    full["insta_market_basket"]["original"]["history_usefulness_passes"] = False
    common = aggregate_common_evidence(full)
    assert not common["contrasts"]["history_at_h"]["passes"]
    assert not common["contrasts"]["history_at_fused"]["passes"]
    assert common["contrasts"]["fusion_accessibility"]["passes"]
    assert common["contrasts"]["head_accessibility"]["passes"]
    assert common["any_common_evidence"]
    full["insta_market_basket"]["original"]["history_usefulness_passes"] = True
    for i, (_, models) in enumerate(full.items()):
        for j, (_, value) in enumerate(models.items()):
            value["contrasts"][list(CONTRASTS)[i * 2 + j]]["passes"] = False
    assert not aggregate_common_evidence(full)["any_common_evidence"]


def test_constant_reconstruction_denominator_returns_null():
    cache = fixture(40)
    cache["history_features"][:, 7] = 1.
    result, _ = analyze_cache(cache, bootstrap_repeats=2)
    for stage in ("input_mean", "h", "fused"):
        assert result["reconstruction"][stage]["per_feature_constant_mse"][7] == 0.
        assert result["reconstruction"][stage]["per_feature_oof_r2"][7] is None
    json.dumps(result, allow_nan=False)


def test_random_family_contains_original_linear_design_and_fixed_random_columns():
    rng = np.random.default_rng(24)
    train_x, test_x = rng.normal(size=(90, 3)), rng.normal(size=(10, 3))
    target = rng.normal(size=90)
    got, meta = fit_predict(train_x, target, test_x, "random128")
    seeded = np.random.default_rng(20260905)
    weights = seeded.normal(size=(3, 128)) / np.sqrt(3)
    bias = seeded.normal(size=128)
    train = (train_x - train_x.mean(axis=0)) / train_x.std(axis=0)
    test = (test_x - train_x.mean(axis=0)) / train_x.std(axis=0)
    design = np.concatenate((train, np.tanh(train @ weights + bias)), axis=1)
    held_design = np.concatenate((test, np.tanh(test @ weights + bias)), axis=1)
    centered = design - design.mean(axis=0)
    coef = np.linalg.solve(centered.T @ centered + np.eye(131), centered.T @ (target - target.mean()))
    expected = (held_design - design.mean(axis=0)) @ coef + target.mean()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    assert meta["design_dimensions"] == 131


def test_sham_uses_independent_seed_and_is_reproducible():
    cache = fixture(40)
    first, first_arrays = analyze_cache(cache, bootstrap_repeats=2)
    second, second_arrays = analyze_cache(cache, bootstrap_repeats=2)
    assert SHAM_SEED == 20260906 and SHAM_SEED != SEED
    independent_sham = np.random.default_rng(SHAM_SEED).normal(size=(40, 64))
    residual = cache["log_quantity"] - cache["base_logpred"]
    expected, _ = oof_predict(independent_sham, residual, cache["fold"], "linear")
    np.testing.assert_array_equal(first_arrays["linear__sham64__residual_prediction"], expected)
    for family in ("linear", "random128"):
        key = f"{family}__sham64__residual_prediction"
        np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
        assert first["decoders"][family]["sham64"] == second["decoders"][family]["sham64"]
