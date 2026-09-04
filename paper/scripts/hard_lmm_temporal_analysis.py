"""Fixed train-only conditional association; no model or correction is fitted.

analyze(features, caches, body_threshold, policy=None) returns JSON summary and
per-row arm (low=0/high=1/other=-1), cell, weight, fold and bodyeligible arrays.
Cuts use opposite-fold observed inputs, never labels. Cell support and metrics
use the fixed body population. Bootstrap freezes cuts, support and rowweights:
it measures conditional fixed-standardization variability, not re-matching or
held-out inference. Frozen models already saw these training rows.
"""
from __future__ import annotations

import json

import numpy as np

CONTROLS = ("mean_log_quantity", "history_length", "count_change_rms", "log_mean_internal_gap")
DEFAULT_POLICY = dict(min_cell_rows=20, min_cell_series=5, min_fold_arm_rows=100,
                      min_fold_arm_series=20, min_retained_fraction=.5, max_smd=.25,
                      minimum_relative_excess=.05, bootstrap_samples=500,
                      bootstrap_seed=20260905, j_min_separation=1e-6)


def _vector(value, name, n=None, allow_nan=False):
    result = np.asarray(value)
    if result.ndim != 1 or (n is not None and len(result) != n):
        raise ValueError(f"{name} must have shape [N]")
    if np.isinf(result).any() or (not allow_nan and not np.isfinite(result).all()):
        raise ValueError(f"nonfinite {name}")
    return result


def _moments(x, w):
    mass = w.sum()
    if mass <= 0:
        return None
    anchor = x[np.flatnonzero(w > 0)[0]]
    mean = anchor + np.dot(w, x - anchor) / mass
    variance = np.dot(w, (x - mean) ** 2) / mass
    if not np.isfinite(mean) or not np.isfinite(variance):
        raise FloatingPointError("nonfinite weighted moments")
    return float(mean), float(variance)


def _balance(features, arm, weight, selected):
    detail, maximum, valid = {}, 0., True
    for name in CONTROLS:
        moments = [_moments(features[name], weight * selected * (arm == a)) for a in (0, 1)]
        if any(moment is None for moment in moments):
            return dict(controls={}, max_abs_smd=None, defined=False)
        low, high = moments
        denominator = np.sqrt((low[1] + high[1]) / 2)
        smd = (high[0] - low[0]) / denominator if denominator > 0 else (0. if high[0] == low[0] else None)
        detail[name] = dict(low_mean=low[0], high_mean=high[0], smd=None if smd is None else float(smd))
        valid = valid and smd is not None
        maximum = max(maximum, abs(smd)) if smd is not None else maximum
    return dict(controls=detail, max_abs_smd=float(maximum) if valid else None, defined=bool(valid))


def _metrics(caches, arm, weight, selected):
    output = {}
    for model, cache in caches.items():
        views = {}
        errors = dict(raw_body_mae=np.abs(cache["prediction"] - cache["quantity"]),
                      signed_log_residual=cache["log_residual"],
                      absolute_log_residual=np.abs(cache["log_residual"]))
        for name, error in errors.items():
            means = []
            for a in (0, 1):
                w = weight * selected * (arm == a)
                means.append(float(np.dot(w, error) / w.sum()) if w.sum() > 0 else None)
            low, high = means
            diff = high - low if low is not None and high is not None else None
            views[name] = dict(low=low, high=high, high_minus_low=diff,
                               relative_excess=diff / low if name != "signed_log_residual" and diff is not None and low > 0 else None)
        output[model] = views
    return output


def _bootstrap(error, arm, weight, series, folds, policy):
    generator = np.random.default_rng(policy["bootstrap_seed"])
    groups = []
    for fold in (0, 1):
        rows = np.flatnonzero(folds == fold)
        ids, inverse = np.unique(series[rows], return_inverse=True)
        numerator, denominator = [], []
        for a in (0, 1):
            w = weight[rows] * (arm[rows] == a)
            denominator.append(np.bincount(inverse, weights=w, minlength=len(ids)))
            numerator.append(np.bincount(inverse, weights=w * error[rows], minlength=len(ids)))
        groups.append((np.array(numerator), np.array(denominator), len(ids)))
    draws = []
    for _ in range(policy["bootstrap_samples"]):
        numerator, denominator = np.zeros(2), np.zeros(2)
        for sums, masses, n_groups in groups:
            sampled = generator.integers(0, n_groups, size=n_groups)
            numerator += sums[:, sampled].sum(axis=1)
            denominator += masses[:, sampled].sum(axis=1)
        if (denominator > 0).all():
            means = numerator / denominator
            draws.append(float(means[1] - means[0]))
    return dict(samples=policy["bootstrap_samples"], valid_samples=len(draws),
                seed=policy["bootstrap_seed"], lower_p05_diff=float(np.quantile(draws, .05)) if draws else None,
                median_diff=float(np.median(draws)) if draws else None,
                interpretation="series resampled within folds; frozen cuts/support/rowweights; each pooled arm renormalized")


def analyze(features, caches, body_threshold, policy=None):
    """Return conditional body-error association, support and predeclared gate."""
    policy = dict(policy or {})
    if set(policy) - set(DEFAULT_POLICY):
        raise ValueError("unknown policy keys")
    policy = {**DEFAULT_POLICY, **policy}
    if any(not np.isfinite(value) or value < 0 for value in policy.values()):
        raise ValueError("policy values must be finite and nonnegative")
    for key in ("min_cell_rows", "min_cell_series", "min_fold_arm_rows", "min_fold_arm_series", "bootstrap_samples"):
        if policy[key] < 1 or int(policy[key]) != policy[key]:
            raise ValueError(f"{key} must be a positive integer")
        policy[key] = int(policy[key])
    j = _vector(features["age_distortion"], "age_distortion", allow_nan=True).astype(float)
    n = len(j)
    if not 1 <= n <= 8192 or not np.isfinite(body_threshold) or body_threshold < 0:
        raise ValueError("bounded nonempty sample and finite nonnegative body threshold required")
    features = {name: _vector(features[name], name, n).astype(float) for name in CONTROLS + ("series_index", "fold")}
    series, folds = features["series_index"], features["fold"]
    if (series % 1).any():
        raise ValueError("series_index must contain integer identifiers")
    if set(folds) != {0., 1.} or np.intersect1d(series[folds == 0], series[folds == 1]).size:
        raise ValueError("folds must be 0/1 with disjoint series")
    if (features["history_length"] < 1).any() or (features["history_length"] % 1).any():
        raise ValueError("history_length must be a positive integer")
    if set(caches) != {"original", "separate_key"}:
        raise ValueError("paired original and separate_key caches required")
    caches = {model: {name: _vector(cache[name], f"{model}.{name}", n).astype(float)
                      for name in ("prediction", "quantity", "log_residual")} for model, cache in caches.items()}
    quantity = caches["original"]["quantity"]
    if not np.array_equal(quantity, caches["separate_key"]["quantity"]) or (quantity < 0).any():
        raise ValueError("paired nonnegative target quantities must be identical")
    valid = (features["history_length"] >= 3) & np.isfinite(j)
    body = quantity <= body_threshold
    arm, cell, weight = np.full(n, -1, dtype=int), np.full(n, -1, dtype=int), np.zeros(n)
    cut_details, fold_details = {}, {}
    for fold in (0, 1):
        reference, held = valid & (folds != fold), valid & (folds == fold)
        if not reference.any():
            cut_details[str(fold)] = dict(j_separated=False, j_bounds=None, controls={})
            continue
        low, high = np.quantile(j[reference], [.25, .75])
        separated = bool(high - low > policy["j_min_separation"])
        cuts, code, radix = {}, np.zeros(n, dtype=int), 1
        for index, name in enumerate(CONTROLS):
            values = features[name][reference]
            boundaries = np.unique(np.quantile(values, [1/3, 2/3] if index == 0 else [.5]))
            cuts[name] = boundaries.tolist()
            code += radix * np.searchsorted(boundaries, features[name], side="right")
            radix *= len(boundaries) + 1
        cut_details[str(fold)] = dict(j_separated=separated, j_bounds=[float(low), float(high)], controls=cuts)
        cell[held] = code[held]
        if not separated:
            continue
        arm[held & (j <= low)] = 0
        arm[held & (j >= high)] = 1
        for current_cell in np.unique(code[held]):
            selected = held & body & (code == current_cell)
            rows = [np.flatnonzero(selected & (arm == a)) for a in (0, 1)]
            counts = [len(part) for part in rows]
            groups = [len(np.unique(series[part])) for part in rows]
            if min(counts) >= policy["min_cell_rows"] and min(groups) >= policy["min_cell_series"]:
                for part in rows:
                    weight[part] = min(counts) / len(part)
    for fold in (0, 1):
        selected, retained = folds == fold, (folds == fold) & (weight > 0)
        extremes = selected & valid & body & (arm >= 0)
        rows = [int((retained & (arm == a)).sum()) for a in (0, 1)]
        groups = [len(np.unique(series[retained & (arm == a)])) for a in (0, 1)]
        fraction = float(retained.sum() / extremes.sum()) if extremes.any() else 0.
        balance = _balance(features, arm, weight, selected)
        reasons = []
        if not cut_details[str(fold)]["j_separated"]: reasons.append("no_J_separation")
        if min(rows) < policy["min_fold_arm_rows"]: reasons.append("insufficient_arm_rows")
        if min(groups) < policy["min_fold_arm_series"]: reasons.append("insufficient_arm_series")
        if fraction < policy["min_retained_fraction"]: reasons.append("insufficient_retained_fraction")
        if not balance["defined"] or balance["max_abs_smd"] > policy["max_smd"]: reasons.append("insufficient_covariate_balance")
        fold_details[str(fold)] = dict(assessable=not reasons, reasons=reasons, low_high_rows=rows,
            low_high_series=groups, body_valid_extremes=int(extremes.sum()), retained_fraction=fraction,
            excluded_from_extremes_by_cell_support=int((extremes & ~retained).sum()),
            retained_cells=int(len(np.unique(cell[retained]))), balance=balance,
            metrics=_metrics(caches, arm, weight, selected))
    assessable = all(detail["assessable"] for detail in fold_details.values())
    pooled = _metrics(caches, arm, weight, np.ones(n, dtype=bool))
    bootstrap = _bootstrap(np.abs(caches["separate_key"]["prediction"] - quantity),
                           arm, weight, series, folds, policy) if assessable else None
    excess = pooled["separate_key"]["raw_body_mae"]["relative_excess"]
    conditions = dict(support_and_balance=assessable,
        separate_key_pooled_excess_at_least_minimum=excess is not None and excess >= policy["minimum_relative_excess"],
        separate_key_both_folds_positive=all((fold_details[str(f)]["metrics"]["separate_key"]["raw_body_mae"]["high_minus_low"] or 0) > 0 for f in (0, 1)),
        original_both_folds_positive=all((fold_details[str(f)]["metrics"]["original"]["raw_body_mae"]["high_minus_low"] or 0) > 0 for f in (0, 1)),
        bootstrap_lower_positive=bool(bootstrap and bootstrap["valid_samples"] == policy["bootstrap_samples"] and (bootstrap["lower_p05_diff"] or 0) > 0))
    summary = dict(protocol="hard_lmm_temporal_conditional_association_v1", policy=policy,
        n_targets=n, body_threshold=float(body_threshold), assessable=assessable,
        weighting="each cell gives each arm mass min(n_high,n_low); row weight is that mass divided by arm count",
        cuts=cut_details, folds=fold_details, pooled=pooled, cluster_bootstrap=bootstrap,
        excluded=dict(invalid_history=int((~valid).sum()), outside_body=int((valid & ~body).sum()),
                      body_valid_nonextreme_or_unseparated=int((valid & body & (arm < 0)).sum()),
                      body_extremes_without_cell_support=int((valid & body & (arm >= 0) & (weight == 0)).sum())),
        relevance_gate=dict(conditions=conditions, passes=bool(all(conditions.values())),
                            interpretation="Apply only to preregistered primary dataset; diagnostic relevance, not architecture performance"),
        scope="Train-internal fixed-standardization association; no fitting or held-out generalization claim")
    json.dumps(summary, allow_nan=False)
    return summary, dict(arm=arm, cell=cell, weight=weight, fold=folds.astype(int), bodyeligible=valid & body)
