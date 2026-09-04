"""Synthetic protocol checks only; no real cache or backbone is loaded."""

import json

import numpy as np
import pytest
import torch

from paper.scripts.hard_lmm_query_diagnostic_analysis import analyze_cache


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def cache(*, signal=True, n=512):
    generator = torch.Generator().manual_seed(917)
    stats = torch.randn(n, 2, generator=generator, dtype=torch.float64)
    h = torch.randn(n, 6, generator=generator, dtype=torch.float64)
    series = torch.arange(n) // 16
    folds = series % 2
    residual = (1.2 * stats[:, 0] - 0.7 * stats[:, 1] + 0.3) if signal else torch.full(
        (n,), 0.75, dtype=torch.float64
    )
    return {"h": h, "stats": stats, "log_residual": residual,
            "series_index": series, "fold": folds, "target_index": torch.arange(n)}


def test_signal_crossfits_beats_both_controls_and_serializes():
    summary, predictions = analyze_cache(cache(), return_predictions=True)
    assert summary["pooled"]["two_stat_relative_improvement_vs_h_only"] > 0.99
    assert summary["pooled"]["two_stat_relative_improvement_vs_constant"] > 0.99
    assert all(fold["two_stat_relative_improvement_vs_h_only"] > 0.99
               for fold in summary["folds"].values())
    assert summary["cluster_bootstrap"]["paired_delta_lower_p05"] > 0
    assert predictions["two_stat_correction"].shape == (512,)
    assert summary["neighbor_cosine"]["targets_with_mean_at_least_0_9"] == 0
    json.dumps(summary, allow_nan=False)


def test_constant_no_signal_and_zero_variance_remain_identity():
    data = cache(signal=False)
    data["stats"][:] = 4.0
    data["h"].zero_()
    summary, predictions = analyze_cache(data, return_predictions=True)
    assert summary["pooled"]["mse"] == {"constant": 0.0, "h_only": 0.0, "two_stat": 0.0}
    assert summary["pooled"]["two_stat_relative_improvement_vs_h_only"] == 0.0
    assert summary["cluster_bootstrap"]["paired_delta_lower_p05"] == 0.0
    assert summary["zero_norm_h_count"] == 512
    assert summary["folds"]["0"]["zero_variance_stat_dimensions"] == [0, 1]
    assert torch.equal(predictions["two_stat_correction"], data["log_residual"])


def test_collinear_features_have_finite_regularized_fit():
    data = cache()
    data["stats"][:, 1] = data["stats"][:, 0] * 1e6
    summary = analyze_cache(data)
    json.dumps(summary, allow_nan=False)
    assert summary["folds"]["0"]["max_regularized_local_condition_number"] > 1.0


def test_fold_prediction_uses_only_opposite_fold_targets_and_statistics():
    original = cache()
    changed = {name: value.clone() for name, value in original.items()}
    changed["log_residual"][changed["fold"] == 0] += 1000.0
    _, before = analyze_cache(original, return_predictions=True)
    _, after = analyze_cache(changed, return_predictions=True)
    held = original["fold"] == 0
    for name in ("constant_correction", "h_only_correction", "two_stat_correction"):
        torch.testing.assert_close(before[name][held], after[name][held], rtol=0, atol=0)


def test_standardization_excludes_held_fold_statistics():
    original = cache()
    changed = {name: value.clone() for name, value in original.items()}
    changed["stats"][changed["fold"] == 0] += 100.0
    before = analyze_cache(original)
    after = analyze_cache(changed)
    for field in ("reference_stat_mean", "reference_stat_raw_std"):
        assert before["folds"]["0"][field] == after["folds"]["0"][field]
        assert before["folds"]["1"][field] != after["folds"]["1"][field]


def test_local_ridge_matches_independent_numpy_closed_form_for_one_target():
    data = cache()
    _, prediction = analyze_cache(data, return_predictions=True)
    query = 0
    reference = np.flatnonzero(data["fold"].numpy() != data["fold"][query].item())
    h = data["h"].numpy()
    unit = h / np.linalg.norm(h, axis=1, keepdims=True)
    score = unit[reference] @ unit[query]
    selected = reference[np.argsort(-score, kind="stable")[:64]]
    stats = data["stats"].numpy()
    mean, scale = stats[reference].mean(0), stats[reference].std(0)
    standardized = (stats - mean) / scale
    local = standardized[selected]
    center = local.mean(0)
    x = local - center
    y = data["log_residual"].numpy()[selected]
    beta = np.linalg.solve(x.T @ x + np.eye(2), x.T @ (y - y.mean()))
    expected = y.mean() + (standardized[query] - center) @ beta
    assert prediction["two_stat_correction"][query].item() == pytest.approx(expected, abs=1e-12)


def test_reproducible_does_not_mutate_input_or_global_rng_and_ties_are_stable():
    data = cache()
    data["h"].fill_(1.0)
    originals = {name: value.clone() for name, value in data.items()}
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    one, predictions = analyze_cache(data, return_predictions=True)
    two = analyze_cache(data)
    assert one == two
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    for name in data:
        assert torch.equal(data[name], originals[name])
    permutation = torch.arange(511, -1, -1)
    _, reordered = analyze_cache({name: value[permutation] for name, value in data.items()},
                                 return_predictions=True)
    torch.testing.assert_close(predictions["two_stat_correction"][permutation],
                               reordered["two_stat_correction"], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["h", "stats", "log_residual"])
def test_nonfinite_rejected(name):
    data = cache()
    data[name].reshape(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        analyze_cache(data)


def test_shared_series_fails_closed():
    data = cache()
    data["series_index"][16] = 0
    with pytest.raises(ValueError, match="group separation"):
        analyze_cache(data)


def test_short_folds_and_one_series_per_fold_fail_closed():
    with pytest.raises(ValueError, match="at least 64"):
        analyze_cache(cache(n=96))
    data = cache(n=128)
    data["series_index"] = data["fold"].clone()
    with pytest.raises(ValueError, match="at least 10"):
        analyze_cache(data)


def test_duplicate_targets_fail_closed():
    data = cache()
    data["target_index"][1] = data["target_index"][0]
    with pytest.raises(ValueError, match="duplicate"):
        analyze_cache(data)
