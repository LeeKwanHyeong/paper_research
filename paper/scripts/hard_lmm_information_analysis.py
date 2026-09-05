"""Fixed train-internal OOF decoders for frozen Hard-LMM representations.

These bounded decoder checks describe conditional accessibility, not information
entropy, causality, held-out generalization, or architecture adoption. The two
series folds, decoder families and comparisons are fixed before inspecting fits.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

SEED = 20260905
SHAM_SEED = 20260906
RIDGE = 1.0
RANDOM_WIDTH = 128
MINIMUM_GAIN = .01
BOOTSTRAP_REPEATS = 10000
FAMILY_QUANTILE = .05 / 32
FAMILIES = ("linear", "random128")
STAGES = ("input_last", "input_mean", "layer1_last", "layer1_mean",
          "layer2_last", "layer2_mean", "h", "r", "fused", "concat_hr")
CONTRASTS = {
    "history_at_h": ("h+F", "h", "h+shamF"),
    "history_at_fused": ("fused+F", "fused", "fused+shamF"),
    "fusion_accessibility": ("concat_hr", "fused", "fused+sham64"),
    "head_accessibility": ("fused", "constant", "sham64"),
}


def _array(value, name, ndim=None, n=None):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    out = np.asarray(value, dtype=np.float64)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if n is not None and len(out) != n:
        raise ValueError(f"{name} row count mismatch")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _folds_and_series(folds, series, n):
    folds = _array(folds, "fold", 1, n)
    series = _array(series, "series_id", 1, n)
    if set(folds) != {0., 1.}:
        raise ValueError("fold must contain exactly 0 and 1")
    if (series % 1).any() or np.intersect1d(series[folds == 0], series[folds == 1]).size:
        raise ValueError("series must be integer identifiers disjoint across folds")
    return folds.astype(np.int64), series.astype(np.int64)


def fit_predict(train_x, train_y, test_x, family="linear"):
    """Float64 SSE ridge=1 with train-only scaling and unpenalized intercept.

    The random family concatenates standardized inputs with 128 frozen tanh
    features. Both families center their final design by its training mean.
    Returns predictions and training-only fit metadata for leakage audits.
    """
    train_x, test_x = _array(train_x, "train_x", 2), _array(test_x, "test_x", 2)
    train_y = _array(train_y, "train_y", n=len(train_x))
    if train_y.ndim not in (1, 2) or not len(train_x) or train_x.shape[1] != test_x.shape[1]:
        raise ValueError("incompatible nonempty train/test arrays")
    if family not in FAMILIES:
        raise ValueError("unknown frozen decoder family")
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale = np.where(scale == 0, 1., scale)
    train_z, test_z = (train_x - mean) / scale, (test_x - mean) / scale
    if family == "random128" and train_x.shape[1]:
        generator = np.random.default_rng(SEED)
        weights = generator.normal(size=(train_x.shape[1], RANDOM_WIDTH)) / np.sqrt(train_x.shape[1])
        bias = generator.normal(size=RANDOM_WIDTH)
        train_z = np.concatenate((train_z, np.tanh(train_z @ weights + bias)), axis=1)
        test_z = np.concatenate((test_z, np.tanh(test_z @ weights + bias)), axis=1)
    center = train_z.mean(axis=0)
    train_z, test_z = train_z - center, test_z - center
    target_mean = train_y.mean(axis=0)
    if train_z.shape[1]:
        gram = train_z.T @ train_z
        gram.flat[::len(gram) + 1] += RIDGE
        coefficients = np.linalg.solve(gram, train_z.T @ (train_y - target_mean))
        prediction = test_z @ coefficients + target_mean
    else:
        prediction = np.broadcast_to(target_mean, (len(test_x),) + train_y.shape[1:]).copy()
    if not np.isfinite(prediction).all():
        raise FloatingPointError("nonfinite ridge prediction")
    metadata = dict(input_mean=mean.tolist(), input_scale=scale.tolist(),
                    design_mean=center.tolist(), target_mean=np.asarray(target_mean).tolist(),
                    train_rows=len(train_x), input_dimensions=train_x.shape[1],
                    design_dimensions=train_z.shape[1], ridge=RIDGE, family=family, seed=SEED)
    return prediction, metadata


def oof_predict(x, y, folds, family="linear"):
    x, y = _array(x, "x", 2), _array(y, "y")
    folds = _array(folds, "fold", 1, len(x))
    if len(y) != len(x) or set(folds) != {0., 1.}:
        raise ValueError("OOF requires matching rows and both frozen folds")
    predictions = np.empty_like(y, dtype=np.float64)
    metadata = {}
    for fold in (0, 1):
        held = folds == fold
        predictions[held], metadata[str(fold)] = fit_predict(x[~held], y[~held], x[held], family)
    return predictions, metadata


def _mse_summary(errors, folds):
    return dict(pooled=float(errors.mean()),
                folds={str(f): float(errors[folds == f].mean()) for f in (0, 1)})


def raw_metrics(log_prediction, quantity, folds, body_threshold, tail_threshold):
    """Quantity metrics use clipped corrected log predictions for reporting only."""
    log_prediction = _array(log_prediction, "log_prediction", 1, len(quantity))
    clipped = np.clip(log_prediction, 0., 20.)
    prediction = np.expm1(clipped)
    absolute, squared = np.abs(prediction - quantity), (prediction - quantity) ** 2
    body, tail = quantity <= body_threshold, quantity > tail_threshold

    def view(selected):
        body_selected, tail_selected = selected & body, selected & tail
        return dict(n=int(selected.sum()), body_n=int(body_selected.sum()), tail_n=int(tail_selected.sum()),
            body_mae=float(absolute[body_selected].mean()) if body_selected.any() else None,
            rmse=float(np.sqrt(squared[selected].mean())),
            tail_mae=float(absolute[tail_selected].mean()) if tail_selected.any() else None)

    return dict(pooled=view(np.ones(len(quantity), dtype=bool)),
                folds={str(f): view(folds == f) for f in (0, 1)},
                clamps=dict(below_zero=int((log_prediction < 0).sum()), above_twenty=int((log_prediction > 20).sum())))


def _cluster_groups(series, folds):
    groups = []
    for fold in (0, 1):
        rows = np.flatnonzero(folds == fold)
        ids, inverse = np.unique(series[rows], return_inverse=True)
        groups.append((rows, inverse, len(ids)))
    return groups


def paired_bootstrap_many(improvements, series, folds, repeats=BOOTSTRAP_REPEATS):
    """Batched paired series resampling shared by all fixed contrast cells.

    At most 32 replicate-by-series-by-comparison values are materialized at a
    time. No decoder is refitted: quantiles are descriptive stability measures.
    """
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
        raise ValueError("bootstrap repeats must be a positive integer")
    improvements = _array(improvements, "improvements", 2)
    folds, series = _folds_and_series(folds, series, len(improvements))
    generator = np.random.default_rng(SEED)
    groups = []
    for rows, inverse, size in _cluster_groups(series, folds):
        sums = np.column_stack([np.bincount(inverse, weights=improvements[rows, column], minlength=size)
                                for column in range(improvements.shape[1])])
        groups.append((sums, np.bincount(inverse, minlength=size), size))
    values = np.empty((int(repeats), improvements.shape[1]), dtype=np.float64)
    for start in range(0, int(repeats), 32):
        size_batch = min(32, int(repeats) - start)
        total = np.zeros((size_batch, improvements.shape[1]))
        count = np.zeros(size_batch)
        for sums, counts, size in groups:
            selected = generator.integers(0, size, size=(size_batch, size))
            total += sums[selected].sum(axis=1)
            count += counts[selected].sum(axis=1)
        values[start:start + size_batch] = total / count[:, None]
    return [dict(repeats=int(repeats), seed=SEED, quantile=FAMILY_QUANTILE,
                 lower_p05=float(np.quantile(values[:, column], .05)),
                 lower_family_quantile=float(np.quantile(values[:, column], FAMILY_QUANTILE)),
                 median=float(np.median(values[:, column])),
                 interpretation="series bootstrap within frozen folds; shared batched draws; fixed OOF predictions; stability quantiles, not formal p-values")
            for column in range(improvements.shape[1])]


def paired_bootstrap(improvement, series, folds, repeats=BOOTSTRAP_REPEATS):
    improvement = _array(improvement, "improvement", 1)
    return paired_bootstrap_many(improvement[:, None], series, folds, repeats)[0]


def evaluate_gate(reference_errors, candidate_errors, folds, series, *, bootstrap_repeats=BOOTSTRAP_REPEATS,
                  reference_body_mae=None, candidate_body_mae=None, bootstrap_summary=None):
    reference_errors = _array(reference_errors, "reference_errors", 1)
    candidate_errors = _array(candidate_errors, "candidate_errors", 1, len(reference_errors))
    if (reference_errors < 0).any() or (candidate_errors < 0).any():
        raise ValueError("MSE contributions must be nonnegative")
    folds, series = _folds_and_series(folds, series, len(reference_errors))
    difference = reference_errors - candidate_errors
    ref, candidate = float(reference_errors.mean()), float(candidate_errors.mean())
    gain = (ref - candidate) / ref if ref > 0 else None
    bootstrap = (paired_bootstrap(difference, series, folds, bootstrap_repeats)
                 if bootstrap_summary is None else bootstrap_summary)
    fold_diff = {str(f): float(difference[folds == f].mean()) for f in (0, 1)}
    conflict = bool(gain is not None and gain > 0 and reference_body_mae is not None
                    and candidate_body_mae is not None and candidate_body_mae > reference_body_mae)
    conditions = dict(pooled_gain_at_least_one_percent=bool(gain is not None and gain >= MINIMUM_GAIN),
                      both_folds_positive=all(v > 0 for v in fold_diff.values()),
                      bootstrap_p05_positive=bootstrap["lower_p05"] > 0,
                      family_quantile_positive=bootstrap["lower_family_quantile"] > 0,
                      no_log_vs_body_direction_conflict=not conflict)
    return dict(reference_mse=ref, candidate_mse=candidate, relative_gain=gain,
                fold_mean_improvements=fold_diff, bootstrap=bootstrap,
                log_vs_body_direction_conflict=conflict, conditions=conditions,
                passes=bool(all(conditions.values())))


def analyze_cache(cache: Mapping, *, bootstrap_repeats=BOOTSTRAP_REPEATS):
    """Analyze one fixed dataset/model cache; return summary and OOF arrays.

    Required keys: STAGES [N,D], history_features [N,12], quantity,
    log_quantity, base_logpred, series_id, fold [N], body_threshold and
    tail_threshold. The caller must enforce matched rows across models/datasets.
    """
    quantity = _array(cache["quantity"], "quantity", 1)
    n = len(quantity)
    if n < 4 or n > 8192 or (quantity < 0).any():
        raise ValueError("bounded 4..8192 nonnegative quantities required")
    folds, series = _folds_and_series(cache["fold"], cache["series_id"], n)
    log_quantity = _array(cache["log_quantity"], "log_quantity", 1, n)
    base = _array(cache["base_logpred"], "base_logpred", 1, n)
    if not np.allclose(log_quantity, np.log1p(quantity), rtol=1e-6, atol=1e-6):
        raise ValueError("log_quantity must be log1p(quantity)")
    history = _array(cache["history_features"], "history_features", 2, n)
    if history.shape[1] != 12:
        raise ValueError("frozen history feature count is 12")
    body_threshold, tail_threshold = float(cache["body_threshold"]), float(cache["tail_threshold"])
    if not (np.isfinite(body_threshold) and np.isfinite(tail_threshold) and 0 <= body_threshold <= tail_threshold):
        raise ValueError("ordered finite nonnegative train thresholds required")
    packs = {name: _array(cache[name], name, 2, n) for name in STAGES}
    if any(v.shape[1] == 0 for v in packs.values()):
        raise ValueError("stage representations cannot be empty")
    if packs["h"].shape[1] != 64 or packs["r"].shape[1] != 64 or packs["fused"].shape[1] != 64:
        raise ValueError("frozen h/r/fused dimensions must equal 64")
    if not np.array_equal(packs["concat_hr"], np.concatenate((packs["h"], packs["r"]), axis=1)):
        raise ValueError("concat_hr must exactly concatenate h and r")
    generator = np.random.default_rng(SHAM_SEED)
    sham = generator.normal(size=(n, 64))
    packs.update({"F": history, "constant": np.empty((n, 0)), "sham64": sham})
    for stage in ("h", "fused"):
        packs[f"{stage}+F"] = np.concatenate((packs[stage], history), axis=1)
        packs[f"{stage}+shamF"] = np.concatenate((packs[stage], sham[:, :12]), axis=1)
    packs["fused+sham64"] = np.concatenate((packs["fused"], sham), axis=1)
    residual = log_quantity - base
    predictions = {"fold": folds, "series_id": series, "quantity": quantity,
                   "log_quantity": log_quantity, "base_logpred": base, "history_features": history}
    decoders, squared_errors = {}, {}
    for family in FAMILIES:
        views, errors = {}, {}
        for name, x in packs.items():
            prediction, fit = oof_predict(x, residual, folds, family)
            corrected_log = base + prediction
            errors[name] = (residual - prediction) ** 2
            predictions[f"{family}__{name}__residual_prediction"] = prediction
            views[name] = dict(input_dimensions=x.shape[1], residual_mse=_mse_summary(errors[name], folds),
                raw_metrics=raw_metrics(corrected_log, quantity, folds, body_threshold, tail_threshold), fit=fit)
        decoders[family], squared_errors[family] = views, errors
    contrasts, usefulness = {}, {}
    for family in FAMILIES:
        constant, feat = squared_errors[family]["constant"], squared_errors[family]["F"]
        mean_constant = float(constant.mean())
        gain = float((constant.mean() - feat.mean()) / mean_constant) if mean_constant > 0 else None
        fold_diff = {str(f): float((constant - feat)[folds == f].mean()) for f in (0, 1)}
        usefulness[family] = dict(relative_gain=gain, fold_mean_improvements=fold_diff,
                                 passes=bool(gain is not None and gain >= MINIMUM_GAIN and all(x > 0 for x in fold_diff.values())))
    bootstrap_keys, improvements = [], []
    for name, (candidate_name, base_name, sham_name) in CONTRASTS.items():
        for family in FAMILIES:
            for reference_name in (base_name, sham_name):
                bootstrap_keys.append((name, family, reference_name))
                improvements.append(squared_errors[family][reference_name] - squared_errors[family][candidate_name])
    bootstrap_values = paired_bootstrap_many(np.column_stack(improvements), series, folds, bootstrap_repeats)
    bootstrap_summaries = dict(zip(bootstrap_keys, bootstrap_values, strict=True))
    for name, (candidate_name, base_name, sham_name) in CONTRASTS.items():
        by_family = {}
        for family in FAMILIES:
            by_reference = {}
            for reference_name in (base_name, sham_name):
                by_reference[reference_name] = evaluate_gate(
                    squared_errors[family][reference_name], squared_errors[family][candidate_name], folds, series,
                    bootstrap_repeats=bootstrap_repeats,
                    bootstrap_summary=bootstrap_summaries[(name, family, reference_name)],
                    reference_body_mae=decoders[family][reference_name]["raw_metrics"]["pooled"]["body_mae"],
                    candidate_body_mae=decoders[family][candidate_name]["raw_metrics"]["pooled"]["body_mae"])
            by_family[family] = dict(comparisons=by_reference, passes=all(x["passes"] for x in by_reference.values()))
        contrasts[name] = dict(candidate=candidate_name, references=[base_name, sham_name], families=by_family,
                               passes=all(x["passes"] for x in by_family.values()))
    reconstruction, constant_history = {}, np.empty_like(history)
    for f in (0, 1):
        constant_history[folds == f] = history[folds != f].mean(axis=0)
    constant_error = (history - constant_history) ** 2
    for stage in STAGES:
        predicted_history, fit = oof_predict(packs[stage], history, folds, "linear")
        errors = (history - predicted_history) ** 2
        baseline = constant_error.mean(axis=0)
        r2 = np.divide(baseline - errors.mean(axis=0), baseline,
                       out=np.zeros_like(baseline), where=baseline > 0)
        reconstruction[stage] = dict(per_feature_mse=errors.mean(axis=0).tolist(),
            per_feature_constant_mse=baseline.tolist(),
            per_feature_oof_r2=[float(value) if denominator > 0 else None for value, denominator in zip(r2, baseline, strict=True)],
            constant_features=(baseline == 0).tolist(),
            folds={str(f): dict(per_feature_mse=errors[folds == f].mean(axis=0).tolist(),
                               per_feature_constant_mse=constant_error[folds == f].mean(axis=0).tolist()) for f in (0, 1)}, fit=fit)
        predictions[f"reconstruct__{stage}"] = predicted_history
    reconstruction["Ftruth"] = dict(per_feature_mse=np.zeros(12).tolist(),
        interpretation="identity-copy sanity only; not a fitted decoder or gate")
    predictions["reconstruct__Ftruth"] = history.copy()
    summary = dict(protocol="hard_lmm_information_access_v1", n_targets=n,
        folds={str(f): dict(rows=int((folds == f).sum()), series=int(len(np.unique(series[folds == f])))) for f in (0, 1)},
        decoder_contract=dict(families=list(FAMILIES), ridge=RIDGE, random_width=RANDOM_WIDTH, seed=SEED,
            minimum_gain=MINIMUM_GAIN, bootstrap_repeats=bootstrap_repeats, family_quantile=FAMILY_QUANTILE,
            feature_scaling="training-fold mean/std; zero std replaced by one", target="log1p(quantity)-frozen base_logpred"),
        body_threshold=body_threshold, tail_threshold=tail_threshold,
        base_raw_metrics=raw_metrics(base, quantity, folds, body_threshold, tail_threshold),
        uncorrected_residual_mse=_mse_summary(residual ** 2, folds),
        decoders=decoders, reconstruction=reconstruction, history_usefulness=usefulness,
        history_usefulness_passes=all(x["passes"] for x in usefulness.values()), contrasts=contrasts,
        interpretation="Fixed train-internal OOF bounded-decoder accessibility; conditional descriptive evidence, not information loss proof, causality, significance testing, held-out generalization, or model adoption")
    json.dumps(summary, allow_nan=False)
    return summary, predictions


def aggregate_common_evidence(results):
    """Require identical contrast across both datasets and both frozen models."""
    if set(results) != {"yellow_trip_hourly", "insta_market_basket"} or any(set(v) != {"original", "separate_key"} for v in results.values()):
        raise ValueError("exactly yellow_trip_hourly/insta_market_basket and original/separate_key required")
    common = {}
    for contrast in CONTRASTS:
        cells = {f"{dataset}/{model}": bool(result["contrasts"][contrast]["passes"] and (contrast not in ("history_at_h", "history_at_fused") or result["history_usefulness_passes"]))
                 for dataset, models in results.items() for model, result in models.items()}
        common[contrast] = dict(cells=cells, passes=all(cells.values()))
    return dict(contrasts=common, any_common_evidence=any(v["passes"] for v in common.values()),
                interpretation="Only a common bounded-decoder accessibility pattern; cannot establish architecture causality or generalization")
