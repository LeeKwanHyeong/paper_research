"""Input-only design -> body-support recheck -> frozen paired error evaluation.

One context per independent series; exact history length and opposite-fold
3/2/2 bins for the other controls. No fitting, resampling searches or revised
thresholds. The entire pool supplies each retention denominator. Bootstrap
keeps cells and weights fixed and resamples series within original pool folds.
These remain training observations of the frozen backbones, not held-out data.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from paper.scripts.hard_lmm_temporal_analysis import (
    CONTROLS, DEFAULT_POLICY as TEMPORAL_POLICY, _balance, _bootstrap, _metrics,
)

DEFAULT_POLICY = {**TEMPORAL_POLICY, "body_threshold": 25.}
FEATURES = ("age_distortion", *CONTROLS, "series_index", "fold")


def _policy(policy):
    policy = dict(policy or {})
    if set(policy) - set(DEFAULT_POLICY) or any(value != DEFAULT_POLICY[key] for key, value in policy.items()):
        raise ValueError("only the frozen policy defaults, including body25, are permitted")
    return dict(DEFAULT_POLICY)


def _features(features):
    result = {name: np.asarray(features[name]) for name in FEATURES}
    n = len(result["age_distortion"])
    if not 1 <= n <= 65536:
        raise ValueError("pool must contain 1..65536 rows")
    for name, values in result.items():
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a finite [N] array")
        integer = name in ("history_length", "series_index", "fold")
        if integer and (values % 1).any():
            raise ValueError(f"{name} must contain integers")
        result[name] = values.astype(np.int64 if integer else np.float64)
    if np.unique(result["series_index"]).size != n:
        raise ValueError("exactly one row per unique series is required")
    if set(result["fold"]) != {0, 1} or (result["history_length"] < 3).any():
        raise ValueError("both folds and legal histories H>=3 are required")
    return result


def _digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(f"{name}:{value.dtype.str}:{value.shape}:".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _assignment(assignment, features, stage):
    n = len(features["fold"])
    required = ("arm", "cell", "input_admitted", "rowweight", "fold", "stage")
    if any(name not in assignment for name in required) or np.asarray(assignment["stage"]).shape != ():
        raise ValueError("incomplete frozen assignment")
    result = {name: np.asarray(value) for name, value in assignment.items()}
    if result["stage"].item() != stage:
        raise ValueError("assignment belongs to the wrong design stage")
    for name in required[:-1]:
        if result[name].shape != (n,) or not np.isfinite(result[name]).all():
            raise ValueError(f"invalid assignment {name}")
    if not np.array_equal(result["fold"], features["fold"]):
        raise ValueError("assignment fold mismatch")
    if not np.isin(result["arm"], [-1, 0, 1]).all() or (result["rowweight"] < 0).any():
        raise ValueError("invalid arm or rowweight")
    if ((result["rowweight"] > 0) & ((result["arm"] < 0) | ~result["input_admitted"].astype(bool))).any():
        raise ValueError("weighted rows must be input-admitted extremes")
    return result


def _weights(features, assignment, eligible, allowed_cells, policy):
    weights = np.zeros(len(eligible))
    allowed_cells = set(allowed_cells.tolist())
    candidates = np.flatnonzero(eligible & (assignment["arm"] >= 0))
    ordered = candidates[np.argsort(assignment["cell"][candidates], kind="stable")]
    boundaries = np.flatnonzero(np.diff(assignment["cell"][ordered])) + 1
    for selected in np.split(ordered, boundaries):
        if not len(selected) or assignment["cell"][selected[0]] not in allowed_cells:
            continue
        rows = [selected[assignment["arm"][selected] == arm] for arm in (0, 1)]
        counts = [len(part) for part in rows]
        groups = [np.unique(features["series_index"][part]).size for part in rows]
        if min(counts) >= policy["min_cell_rows"] and min(groups) >= policy["min_cell_series"]:
            for part in rows:
                weights[part] = min(counts) / len(part)
    return weights


def _support(features, assignment, eligible, policy):
    arm, weight = assignment["arm"], assignment["rowweight"]
    details = {}
    for fold in (0, 1):
        selected = features["fold"] == fold
        retained = selected & (weight > 0)
        potential = selected & eligible & (arm >= 0)
        rows = [int((retained & (arm == a)).sum()) for a in (0, 1)]
        groups = [int(np.unique(features["series_index"][retained & (arm == a)]).size) for a in (0, 1)]
        fraction = float(retained.sum() / potential.sum()) if potential.any() else 0.
        balance = _balance(features, arm, weight, selected)
        reasons = []
        if min(rows) < policy["min_fold_arm_rows"]: reasons.append("insufficient_arm_rows")
        if min(groups) < policy["min_fold_arm_series"]: reasons.append("insufficient_arm_series")
        if fraction < policy["min_retained_fraction"]: reasons.append("insufficient_retained_fraction")
        if not balance["defined"] or balance["max_abs_smd"] > policy["max_smd"]: reasons.append("insufficient_covariate_balance")
        details[str(fold)] = dict(passes=not reasons, reasons=reasons, low_high_rows=rows,
            low_high_series=groups, all_pool_eligible_extremes=int(potential.sum()),
            retained_fraction=fraction, balance=balance,
            retained_cells=int(np.unique(assignment["cell"][retained]).size))
    return dict(passes=all(item["passes"] for item in details.values()), folds=details)


def design(features, policy=None):
    """Read no labels; return frozen input design summary and assignment arrays."""
    policy, features = _policy(policy), _features(features)
    n, j = len(features["fold"]), features["age_distortion"]
    arm, raw_cells, cuts = np.full(n, -1, dtype=int), np.zeros((n, 5), dtype=np.int64), {}
    raw_cells[:, :2] = np.column_stack((features["fold"], features["history_length"]))
    for fold in (0, 1):
        reference, held = features["fold"] != fold, features["fold"] == fold
        low, high = np.quantile(j[reference], [.25, .75])
        separated = bool(high - low > policy["j_min_separation"])
        controls = {}
        for index, name in enumerate(("mean_log_quantity", "count_change_rms", "log_mean_internal_gap")):
            boundaries = np.unique(np.quantile(features[name][reference], [1/3, 2/3] if index == 0 else [.5]))
            controls[name] = boundaries.tolist()
            raw_cells[held, index + 2] = np.searchsorted(boundaries, features[name][held], side="right")
        cuts[str(fold)] = dict(j_bounds=[float(low), float(high)], j_separated=separated,
                              history_matching="exact integer H", controls=controls)
        if separated:
            arm[held & (j <= low)] = 0
            arm[held & (j >= high)] = 1
    definitions, cell = np.unique(raw_cells, axis=0, return_inverse=True)
    assignment = dict(arm=arm, cell=cell, fold=features["fold"].copy(), stage=np.array(0, dtype=np.int8))
    weight = _weights(features, assignment, arm >= 0, np.unique(cell), policy)
    assignment.update(input_admitted=weight > 0, rowweight=weight)
    summary = dict(protocol="hard_lmm_instacart_balanced_design_v1", policy=policy, n_pool=n,
        cuts=cuts, cell_definitions=definitions.tolist(),
        cell_columns=["fold", "history_length", "mean_log_quantity_bin", "count_change_rms_bin", "log_mean_internal_gap_bin"],
        **_support(features, assignment, arm >= 0, policy),
        feature_digest=_digest(features), assignment_digest=_digest(assignment))
    for fold in (0, 1):
        if not cuts[str(fold)]["j_separated"]:
            summary["folds"][str(fold)]["reasons"].append("no_J_separation")
            summary["folds"][str(fold)]["passes"] = False
            summary["passes"] = False
    json.dumps(summary, allow_nan=False)
    return summary, assignment


def body_recheck(features, assignment, target_quantities, design_summary, policy=None):
    """After input pass, admit all pool-extreme body labels; never add cells."""
    policy = _policy(policy)
    if not design_summary.get("passes", False) or design_summary.get("policy") != policy:
        raise ValueError("input comparability gate must pass before body-label access")
    features = _features(features)
    assignment = _assignment(assignment, features, 0)
    if _digest(features) != design_summary["feature_digest"] or _digest(assignment) != design_summary["assignment_digest"]:
        raise ValueError("frozen input features or assignments changed")
    quantities = np.asarray(target_quantities, dtype=float)
    extremes = assignment["arm"] >= 0
    if quantities.shape != extremes.shape or not np.isfinite(quantities[extremes]).all() or (quantities[extremes] < 0).any():
        raise ValueError("finite nonnegative body labels required for ALL pool extremes")
    eligible = extremes & (quantities <= policy["body_threshold"])
    allowed = np.unique(assignment["cell"][assignment["input_admitted"]])
    weights = _weights(features, assignment, eligible & assignment["input_admitted"], allowed, policy)
    result = {name: value.copy() for name, value in assignment.items()}
    result.update(rowweight=weights, stage=np.array(1, dtype=np.int8),
                  bodyeligible=eligible, body_admitted=weights > 0)
    summary = dict(protocol="hard_lmm_instacart_balanced_body_v1", policy=policy,
        **_support(features, result, eligible, policy),
        feature_digest=_digest(features), input_assignment_digest=design_summary["assignment_digest"],
        assignment_digest=_digest(result), body_label_digest=_digest({"extreme_rows": np.flatnonzero(extremes), "quantities": quantities[extremes]}),
        retention_denominator="all candidate-pool body extremes, including input-unsupported cells",
        input_admitted_cells=int(len(allowed)), body_admitted_cells=int(np.unique(result["cell"][weights > 0]).size))
    json.dumps(summary, allow_nan=False)
    return summary, result


def evaluate(features, assignment, caches, policy=None):
    """Score retained body rows only, after frozen body support has passed."""
    policy, features = _policy(policy), _features(features)
    assignment = _assignment(assignment, features, 1)
    n = len(features["fold"])
    if "bodyeligible" not in assignment or assignment["bodyeligible"].shape != (n,):
        raise ValueError("frozen body eligibility is required")
    support = _support(features, assignment, assignment["bodyeligible"], policy)
    if not support["passes"]:
        raise ValueError("body support must pass before prediction/error access")
    retained = assignment["rowweight"] > 0
    if set(caches) != {"original", "separate_key"}:
        raise ValueError("paired original/separate_key predictions required")
    clean = {}
    for model, cache in caches.items():
        clean[model] = {}
        for name in ("prediction", "quantity", "log_residual"):
            values = np.asarray(cache[name])
            if values.shape != (n,) or not np.isfinite(values[retained]).all():
                raise ValueError(f"finite {model}.{name} required on retained rows")
            clean[model][name] = np.zeros(n)
            clean[model][name][retained] = values[retained]
    quantity = clean["original"]["quantity"]
    if not np.array_equal(quantity[retained], clean["separate_key"]["quantity"][retained]) or ((quantity[retained] < 0) | (quantity[retained] > 25)).any():
        raise ValueError("retained paired labels must agree and belong to body25")
    arm, weights = assignment["arm"], assignment["rowweight"]
    pooled = _metrics(clean, arm, weights, np.ones(n, dtype=bool))
    folds = {str(fold): _metrics(clean, arm, weights, features["fold"] == fold) for fold in (0, 1)}
    band = _bootstrap(np.abs(clean["separate_key"]["prediction"] - quantity), arm, weights,
                      features["series_index"], features["fold"], policy)
    excess = pooled["separate_key"]["raw_body_mae"]["relative_excess"]
    conditions = dict(body_support_and_balance=True,
        separate_key_pooled_excess_at_least_minimum=excess is not None and excess >= policy["minimum_relative_excess"],
        separate_key_both_folds_positive=all((folds[str(f)]["separate_key"]["raw_body_mae"]["high_minus_low"] or 0) > 0 for f in (0, 1)),
        original_both_folds_positive=all((folds[str(f)]["original"]["raw_body_mae"]["high_minus_low"] or 0) > 0 for f in (0, 1)),
        bootstrap_lower_positive=band["valid_samples"] == policy["bootstrap_samples"] and (band["lower_p05_diff"] or 0) > 0)
    summary = dict(protocol="hard_lmm_instacart_balanced_evaluation_v1", policy=policy,
        n_scored=int(retained.sum()), pooled=pooled, folds=folds, body_support=support,
        cluster_bootstrap=band, relevance_gate=dict(conditions=conditions, passes=bool(all(conditions.values()))),
        assignment_digest=_digest(assignment),
        scope="one-context-per-unused-series train-internal association; conditional fixed weights, no fitting or held-out performance claim")
    json.dumps(summary, allow_nan=False)
    return summary
