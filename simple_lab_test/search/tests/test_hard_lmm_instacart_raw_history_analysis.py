"""Synthetic contract tests for the frozen Instacart raw-history diagnosis."""
import json

import numpy as np
import pytest

from paper.scripts.hard_lmm_instacart_raw_history_analysis import (
    RAW_DIMENSIONS,
    SIMULTANEOUS_LOWER_QUANTILE,
    analyze_models,
    build_h_only64,
    build_raw64,
)


def _cache_pair(n=320, *, h_contains_signal=False):
    index = np.arange(n)
    history_length = np.full(n, 3)
    history_dt = np.zeros((n, 32))
    history_quantity = np.zeros((n, 32))
    x = np.random.default_rng(412).uniform(.15, 1.85, n)
    history_dt[:, :3] = np.array([3., 1., 2.])
    history_quantity[:, 0] = 1.
    history_quantity[:, 1] = 2.
    history_quantity[:, 2] = np.expm1(x)
    raw = build_raw64(history_dt, history_quantity, history_length)
    h64 = raw.copy() if h_contains_signal else np.zeros((n, 64))
    log_quantity = 2. + .65 * x
    common = dict(
        history_dt=history_dt,
        history_quantity=history_quantity,
        history_length=history_length,
        h64=h64,
        quantity=np.expm1(log_quantity),
        log_quantity=log_quantity,
        base_logpred=np.full(n, 2.),
        series_id=index,
        fold=index % 2,
        body_threshold=100.,
        tail_threshold=200.,
    )
    return {"original": dict(common), "separate_key": dict(common)}


def test_raw_order_latest_alignment_first_boundary_and_h_only_control():
    dt = np.full((2, 32), 999.)
    qty = np.full((2, 32), 888.)
    dt[0, :3], qty[0, :3] = [3., 1., 2.], [4., 5., 6.]
    dt[1, :4], qty[1, :4] = [7., 8., 9., 10.], [11., 12., 13., 14.]
    lengths = np.array([3, 4])
    raw = build_raw64(dt, qty, lengths).reshape(2, 32, 2)
    assert raw.reshape(2, -1).shape == (2, RAW_DIMENSIONS)
    np.testing.assert_array_equal(raw[0, :-3], 0.)
    np.testing.assert_allclose(raw[0, -3:, 0], np.log1p([3., 1., 2.]))
    np.testing.assert_allclose(raw[0, -3:, 1], np.log1p([4., 5., 6.]))
    np.testing.assert_allclose(raw[1, -4:, 0], np.log1p([7., 8., 9., 10.]))
    h_only = build_h_only64(lengths).reshape(2, 32, 2)
    np.testing.assert_array_equal(h_only[0, :-3], 0.)
    np.testing.assert_array_equal(h_only[0, -3:, 0], 1.)
    np.testing.assert_array_equal(h_only[:, :, 1], 0.)


def test_poison_padding_is_ignored_and_target_is_not_a_builder_input():
    dt = np.zeros((2, 32))
    qty = np.zeros((2, 32))
    dt[:, :3], qty[:, :3] = [1., 2., 3.], [4., 5., 6.]
    clean = build_raw64(dt, qty, [3, 3])
    dt[:, 3:] = np.nan
    qty[:, 3:] = -999.
    poisoned = build_raw64(dt, qty, [3, 3])
    np.testing.assert_array_equal(clean, poisoned)
    target_a, target_b = np.array([10., 20.]), np.array([999., 888.])
    assert not np.array_equal(target_a, target_b)
    np.testing.assert_array_equal(poisoned, build_raw64(dt, qty, [3, 3]))


def test_positive_synthetic_encoder_loss_passes_both_families_and_is_json_finite():
    result, arrays = analyze_models(_cache_pair(), bootstrap_repeats=120)
    assert result["decision"]["separate_key_encoder_bottleneck_evidence"]
    assert result["decision"]["checkpoint_common_encoder_bottleneck_evidence"]
    assert result["decision"]["classification"] == "checkpoint_common_encoder_bottleneck_evidence"
    for model in ("original", "separate_key"):
        for family in ("linear", "random128"):
            family_result = result["decision"]["by_model"][model]["families"][family]
            assert family_result["passes_all_references"]
            assert family_result["sham_sanity"]["valid"]
    assert result["decoder_contract"]["primary_cells"] == 12
    assert SIMULTANEOUS_LOWER_QUANTILE == pytest.approx(.05 / 24)
    json.dumps(result, allow_nan=False)
    assert all(np.isfinite(value).all() for value in arrays.values())
    assert "separate_key__residual__linear__raw64__prediction" in arrays
    assert "separate_key__residual__linear__raw64__squared_error" in arrays


def test_raw_signal_without_raw_advantage_over_h_is_inconclusive_not_encoder_loss():
    result, _ = analyze_models(_cache_pair(h_contains_signal=True), bootstrap_repeats=80)
    current = result["decision"]["by_model"]["separate_key"]
    assert current["raw_signal_beyond_controls"]
    assert not current["raw_beats_h"]
    assert not current["encoder_bottleneck_evidence"]
    assert result["decision"]["classification"] == "raw_signal_found_but_encoder_comparison_inconclusive"


def test_cross_fold_series_leakage_is_rejected():
    caches = _cache_pair(80)
    leaked = np.arange(80) // 2
    for cache in caches.values():
        cache["series_id"] = leaked
    with pytest.raises(ValueError, match="disjoint"):
        analyze_models(caches, bootstrap_repeats=2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cache: cache.update(h64=np.zeros((len(cache["quantity"]), 63))), "exactly 64"),
        (lambda cache: cache.update(history_dt=np.zeros((len(cache["quantity"]), 31))), "shape"),
        (lambda cache: cache["history_length"].__setitem__(0, 2), "3..32"),
        (lambda cache: cache["history_dt"].__setitem__((0, 0), -1.), "nonnegative"),
        (lambda cache: cache["log_quantity"].__setitem__(0, 999.), "log1p"),
    ],
)
def test_dimension_and_observed_data_corruption_fail_strictly(mutation, message):
    caches = _cache_pair(80)
    mutation(caches["separate_key"])
    with pytest.raises(ValueError, match=message):
        analyze_models(caches, bootstrap_repeats=2)


def test_model_cohort_mismatch_is_rejected():
    caches = _cache_pair(80)
    caches["separate_key"]["history_quantity"] = caches["separate_key"]["history_quantity"].copy()
    caches["separate_key"]["history_quantity"][0, 2] += 1.
    with pytest.raises(ValueError, match="identical cohort field raw64"):
        analyze_models(caches, bootstrap_repeats=2)
