"""Frozen Instacart raw-history accessibility diagnosis.

The diagnostic asks whether a bounded decoder can recover frozen model
residuals from the complete observed history more reliably than from the
encoder state.  It is train-internal, uses series-disjoint OOF folds, and does
not establish information-theoretic absence, causality, or held-out gains.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from paper.scripts.hard_lmm_information_analysis import (
    FAMILIES,
    RANDOM_WIDTH,
    RIDGE,
    SEED,
    fit_predict,
    raw_metrics,
)


PROTOCOL = "hard_lmm_instacart_raw_history_access_v1"
MODEL_NAMES = ("original", "separate_key")
PACK_NAMES = ("raw64", "H_only64", "h64", "constant", "sham64")
REFERENCES = ("constant", "H_only64", "h64")
HISTORY_SLOTS = 32
RAW_DIMENSIONS = 64
MIN_HISTORY = 3
MAX_HISTORY = 32
MAX_TARGETS = 65525
MINIMUM_GAIN = .01
BOOTSTRAP_REPEATS = 10000
SHAM_SEED = 20260906
PRIMARY_CELLS = len(MODEL_NAMES) * len(FAMILIES) * len(REFERENCES)
SIMULTANEOUS_LOWER_QUANTILE = .05 / (2 * PRIMARY_CELLS)


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _array(value, name, ndim=None, n=None):
    out = np.asarray(_numpy(value), dtype=np.float64)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if n is not None and len(out) != n:
        raise ValueError(f"{name} row count mismatch")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _history_lengths(value, n=None):
    lengths = _array(value, "history_length", 1, n)
    if (lengths % 1).any():
        raise ValueError("history_length must contain integers")
    lengths = lengths.astype(np.int64)
    if ((lengths < MIN_HISTORY) | (lengths > MAX_HISTORY)).any():
        raise ValueError("history_length must be in the inclusive range 3..32")
    return lengths


def build_raw64(history_dt, history_quantity, history_length):
    """Create the fixed 64-D raw pack from an observed, right-padded prefix.

    Source rows contain all observed events, oldest to latest, in columns
    ``[:H]``.  The first elapsed-time value is the boundary gap and is retained.
    Observed pairs are transformed as ``[log1p(dt), log1p(quantity)]`` and
    latest-aligned in 32 output slots.  Source padding is deliberately never
    read, so poison values after ``H`` cannot affect the result.
    """
    dt, quantity = _numpy(history_dt), _numpy(history_quantity)
    if dt.ndim != 2 or dt.shape[1] != HISTORY_SLOTS:
        raise ValueError("history_dt must have shape [N, 32]")
    if quantity.shape != dt.shape:
        raise ValueError("history_quantity must match history_dt shape [N, 32]")
    lengths = _history_lengths(history_length, len(dt))
    output = np.zeros((len(dt), HISTORY_SLOTS, 2), dtype=np.float64)
    for row, length in enumerate(lengths):
        observed_dt = np.asarray(dt[row, :length], dtype=np.float64)
        observed_quantity = np.asarray(quantity[row, :length], dtype=np.float64)
        if not np.isfinite(observed_dt).all() or not np.isfinite(observed_quantity).all():
            raise ValueError("observed history values must be finite")
        if (observed_dt < 0).any() or (observed_quantity < 0).any():
            raise ValueError("observed history values must be nonnegative")
        output[row, HISTORY_SLOTS - length:, 0] = np.log1p(observed_dt)
        output[row, HISTORY_SLOTS - length:, 1] = np.log1p(observed_quantity)
    flattened = output.reshape(len(dt), RAW_DIMENSIONS)
    if flattened.shape[1] != RAW_DIMENSIONS or not np.isfinite(flattened).all():
        raise AssertionError("raw history construction violated its 64-D finite contract")
    return flattened


def build_h_only64(history_length):
    """Encode only H: valid slots are ``[1, 0]`` and padding is ``[0, 0]``."""
    lengths = _history_lengths(history_length)
    output = np.zeros((len(lengths), HISTORY_SLOTS, 2), dtype=np.float64)
    for row, length in enumerate(lengths):
        output[row, HISTORY_SLOTS - length:, 0] = 1.
    return output.reshape(len(lengths), RAW_DIMENSIONS)


def _folds_and_series(folds, series, n):
    folds = _array(folds, "fold", 1, n)
    series = _array(series, "series_id", 1, n)
    if set(folds) != {0., 1.} or (folds % 1).any():
        raise ValueError("fold must contain exactly integer folds 0 and 1")
    if (series % 1).any():
        raise ValueError("series_id must contain integer identifiers")
    folds, series = folds.astype(np.int64), series.astype(np.int64)
    if np.intersect1d(np.unique(series[folds == 0]), np.unique(series[folds == 1])).size:
        raise ValueError("series identifiers must be disjoint across folds")
    if any((folds == fold).sum() < 2 for fold in (0, 1)):
        raise ValueError("each fold must contain at least two rows")
    return folds, series


def _mse_summary(errors, folds):
    return dict(
        pooled=float(errors.mean()),
        folds={str(fold): float(errors[folds == fold].mean()) for fold in (0, 1)},
    )


def _oof_predict(x, y, folds, family):
    predictions = np.empty(len(y), dtype=np.float64)
    metadata = {}
    for fold in (0, 1):
        held = folds == fold
        predictions[held], metadata[str(fold)] = fit_predict(
            x[~held], y[~held], x[held], family,
        )
    if not np.isfinite(predictions).all():
        raise FloatingPointError("nonfinite OOF prediction")
    return predictions, metadata


def _cluster_groups(series, folds):
    groups = []
    for fold in (0, 1):
        rows = np.flatnonzero(folds == fold)
        _, inverse = np.unique(series[rows], return_inverse=True)
        groups.append((rows, inverse, int(inverse.max()) + 1))
    return groups


def _paired_bootstrap_many(improvements, series, folds, repeats):
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats < 1:
        raise ValueError("bootstrap_repeats must be a positive integer")
    improvements = _array(improvements, "bootstrap improvements", 2, len(series))
    generator = np.random.default_rng(SEED)
    groups = []
    for rows, inverse, size in _cluster_groups(series, folds):
        sums = np.column_stack([
            np.bincount(inverse, weights=improvements[rows, column], minlength=size)
            for column in range(improvements.shape[1])
        ])
        groups.append((sums, np.bincount(inverse, minlength=size), size))
    values = np.empty((int(repeats), improvements.shape[1]), dtype=np.float64)
    for start in range(0, int(repeats), 32):
        batch = min(32, int(repeats) - start)
        total = np.zeros((batch, improvements.shape[1]), dtype=np.float64)
        count = np.zeros(batch, dtype=np.float64)
        for sums, counts, size in groups:
            selected = generator.integers(0, size, size=(batch, size))
            total += sums[selected].sum(axis=1)
            count += counts[selected].sum(axis=1)
        values[start:start + batch] = total / count[:, None]
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite bootstrap result")
    return [
        dict(
            repeats=int(repeats),
            seed=SEED,
            lower_p05=float(np.quantile(values[:, column], .05)),
            lower_simultaneous=float(np.quantile(
                values[:, column], SIMULTANEOUS_LOWER_QUANTILE,
            )),
            median=float(np.median(values[:, column])),
            simultaneous_lower_quantile=SIMULTANEOUS_LOWER_QUANTILE,
            interpretation=(
                "series bootstrap within frozen folds with shared draws and fixed OOF "
                "predictions; stability interval, not a formal refitted-probe p-value"
            ),
        )
        for column in range(improvements.shape[1])
    ]


def _validate_cache(cache, model_name):
    if not isinstance(cache, Mapping):
        raise ValueError(f"{model_name} cache must be a mapping")
    quantity = _array(cache["quantity"], "quantity", 1)
    n = len(quantity)
    if n < 4 or n > MAX_TARGETS or (quantity < 0).any():
        raise ValueError("quantity must contain 4..65525 nonnegative rows")
    folds, series = _folds_and_series(cache["fold"], cache["series_id"], n)
    log_quantity = _array(cache["log_quantity"], "log_quantity", 1, n)
    if not np.allclose(log_quantity, np.log1p(quantity), rtol=1e-6, atol=1e-6):
        raise ValueError("log_quantity must equal log1p(quantity)")
    base = _array(cache["base_logpred"], "base_logpred", 1, n)
    h64 = _array(cache["h64"], "h64", 2, n)
    if h64.shape[1] != RAW_DIMENSIONS:
        raise ValueError("h64 must have exactly 64 dimensions")
    lengths = _history_lengths(cache["history_length"], n)
    raw64 = build_raw64(cache["history_dt"], cache["history_quantity"], lengths)
    h_only = build_h_only64(lengths)
    body_threshold = float(cache["body_threshold"])
    tail_threshold = float(cache["tail_threshold"])
    if not (np.isfinite(body_threshold) and np.isfinite(tail_threshold)
            and 0 <= body_threshold <= tail_threshold):
        raise ValueError("ordered finite nonnegative train thresholds required")
    if not (quantity <= body_threshold).any():
        raise ValueError("body threshold must select at least one row")
    return dict(
        quantity=quantity,
        log_quantity=log_quantity,
        base_logpred=base,
        h64=h64,
        raw64=raw64,
        H_only64=h_only,
        history_length=lengths,
        folds=folds,
        series=series,
        body_threshold=body_threshold,
        tail_threshold=tail_threshold,
    )


def _require_matched_cohort(caches):
    reference = caches[MODEL_NAMES[0]]
    for model_name in MODEL_NAMES[1:]:
        candidate = caches[model_name]
        for key in ("quantity", "log_quantity", "raw64", "H_only64",
                    "history_length", "folds", "series"):
            if not np.array_equal(reference[key], candidate[key]):
                raise ValueError(f"model caches must use identical cohort field {key}")
        for key in ("body_threshold", "tail_threshold"):
            if reference[key] != candidate[key]:
                raise ValueError(f"model caches must use identical {key}")


def _pack_views(data, packs, target, prediction_base, folds):
    family_views, family_errors, family_predictions = {}, {}, {}
    for family in FAMILIES:
        views, errors, predictions = {}, {}, {}
        for pack_name in PACK_NAMES:
            prediction, fit = _oof_predict(packs[pack_name], target, folds, family)
            error = (target - prediction) ** 2
            log_prediction = prediction_base + prediction
            views[pack_name] = dict(
                input_dimensions=int(packs[pack_name].shape[1]),
                mse=_mse_summary(error, folds),
                raw_metrics=raw_metrics(
                    log_prediction,
                    data["quantity"],
                    folds,
                    data["body_threshold"],
                    data["tail_threshold"],
                ),
                fit=fit,
            )
            errors[pack_name], predictions[pack_name] = error, prediction
        family_views[family] = dict(packs=views)
        family_errors[family], family_predictions[family] = errors, predictions
    return family_views, family_errors, family_predictions


def _comparison(reference_errors, candidate_errors, folds, reference_body, candidate_body,
                bootstrap, procedural_sham_valid=True):
    reference_mse = float(reference_errors.mean())
    candidate_mse = float(candidate_errors.mean())
    gain = ((reference_mse - candidate_mse) / reference_mse
            if reference_mse > 0 else None)
    difference = reference_errors - candidate_errors
    fold_difference = {
        str(fold): float(difference[folds == fold].mean()) for fold in (0, 1)
    }
    conditions = dict(
        pooled_gain_at_least_one_percent=bool(gain is not None and gain >= MINIMUM_GAIN),
        both_folds_positive=bool(all(value > 0 for value in fold_difference.values())),
        bootstrap_simultaneous_lower_positive=bool(bootstrap["lower_simultaneous"] > 0),
        no_log_vs_body_direction_conflict=bool(candidate_body <= reference_body),
        procedural_sham_valid=bool(procedural_sham_valid),
    )
    return dict(
        reference_mse=reference_mse,
        candidate_mse=candidate_mse,
        relative_gain=gain,
        fold_mean_improvements=fold_difference,
        reference_body_mae=float(reference_body),
        candidate_body_mae=float(candidate_body),
        log_vs_body_direction_conflict=bool(candidate_body > reference_body),
        bootstrap=bootstrap,
        conditions=conditions,
        passes=bool(all(conditions.values())),
    )


def _descriptive_comparison(reference_errors, candidate_errors, folds,
                            reference_body, candidate_body):
    reference_mse, candidate_mse = float(reference_errors.mean()), float(candidate_errors.mean())
    difference = reference_errors - candidate_errors
    return dict(
        reference_mse=reference_mse,
        candidate_mse=candidate_mse,
        relative_gain=((reference_mse - candidate_mse) / reference_mse
                       if reference_mse > 0 else None),
        fold_mean_improvements={
            str(fold): float(difference[folds == fold].mean()) for fold in (0, 1)
        },
        reference_body_mae=float(reference_body),
        candidate_body_mae=float(candidate_body),
        log_vs_body_direction_conflict=bool(candidate_body > reference_body),
        interpretation="secondary direct-log-target context; excluded from the encoder decision",
    )


def analyze_models(model_caches: Mapping, *, bootstrap_repeats=BOOTSTRAP_REPEATS):
    """Analyze matched original and separate-key Instacart caches.

    Each model cache must provide ``history_dt`` and ``history_quantity`` with
    shape ``[N, 32]``; ``history_length``; ``h64``; frozen ``base_logpred``;
    target ``quantity``/``log_quantity``; series-disjoint ``fold`` and
    ``series_id``; and train-derived body/tail thresholds.
    """
    if not isinstance(model_caches, Mapping) or set(model_caches) != set(MODEL_NAMES):
        raise ValueError("exactly original and separate_key caches are required")
    caches = {name: _validate_cache(model_caches[name], name) for name in MODEL_NAMES}
    _require_matched_cohort(caches)
    common = caches["original"]
    n, folds, series = len(common["quantity"]), common["folds"], common["series"]
    sham64 = np.random.default_rng(SHAM_SEED).normal(size=(n, RAW_DIMENSIONS))

    model_results, residual_errors, direct_errors = {}, {}, {}
    output_arrays = dict(
        raw64=common["raw64"],
        H_only64=common["H_only64"],
        sham64=sham64,
        fold=folds,
        series_id=series,
        quantity=common["quantity"],
        log_quantity=common["log_quantity"],
        history_length=common["history_length"],
    )
    for model_name in MODEL_NAMES:
        data = caches[model_name]
        packs = dict(
            raw64=data["raw64"],
            H_only64=data["H_only64"],
            h64=data["h64"],
            constant=np.empty((n, 0), dtype=np.float64),
            sham64=sham64,
        )
        residual = data["log_quantity"] - data["base_logpred"]
        residual_views, residual_errors[model_name], residual_predictions = _pack_views(
            data, packs, residual, data["base_logpred"], folds,
        )
        direct_views, direct_errors[model_name], direct_predictions = _pack_views(
            data, packs, data["log_quantity"], np.zeros(n, dtype=np.float64), folds,
        )
        output_arrays[f"{model_name}__base_logpred"] = data["base_logpred"]
        output_arrays[f"{model_name}__residual_label"] = residual
        output_arrays[f"{model_name}__h64"] = data["h64"]
        for family in FAMILIES:
            for pack_name in PACK_NAMES:
                output_arrays[
                    f"{model_name}__residual__{family}__{pack_name}__prediction"
                ] = residual_predictions[family][pack_name]
                output_arrays[
                    f"{model_name}__residual__{family}__{pack_name}__squared_error"
                ] = residual_errors[model_name][family][pack_name]
                output_arrays[
                    f"{model_name}__direct__{family}__{pack_name}__prediction"
                ] = direct_predictions[family][pack_name]
                output_arrays[
                    f"{model_name}__direct__{family}__{pack_name}__squared_error"
                ] = direct_errors[model_name][family][pack_name]
        model_results[model_name] = dict(
            base_raw_metrics=raw_metrics(
                data["base_logpred"], data["quantity"], folds,
                data["body_threshold"], data["tail_threshold"],
            ),
            uncorrected_residual_mse=_mse_summary(residual ** 2, folds),
            residual=dict(families=residual_views),
            direct_log_target=dict(families=direct_views),
        )

    bootstrap_keys, improvement_columns = [], []
    for model_name in MODEL_NAMES:
        for family in FAMILIES:
            for reference in REFERENCES:
                bootstrap_keys.append(("raw", model_name, family, reference))
                improvement_columns.append(
                    residual_errors[model_name][family][reference]
                    - residual_errors[model_name][family]["raw64"]
                )
            bootstrap_keys.append(("sham", model_name, family, "constant"))
            improvement_columns.append(
                residual_errors[model_name][family]["constant"]
                - residual_errors[model_name][family]["sham64"]
            )
    bootstrap_values = _paired_bootstrap_many(
        np.column_stack(improvement_columns), series, folds, bootstrap_repeats,
    )
    bootstrap = dict(zip(bootstrap_keys, bootstrap_values, strict=True))

    decision_models = {}
    for model_name in MODEL_NAMES:
        family_decisions = {}
        for family in FAMILIES:
            constant_error = residual_errors[model_name][family]["constant"]
            sham_error = residual_errors[model_name][family]["sham64"]
            sham_difference = constant_error - sham_error
            sham_reference = float(constant_error.mean())
            sham_gain = ((sham_reference - float(sham_error.mean())) / sham_reference
                         if sham_reference > 0 else None)
            sham_folds = {
                str(fold): float(sham_difference[folds == fold].mean()) for fold in (0, 1)
            }
            sham_bootstrap = bootstrap[("sham", model_name, family, "constant")]
            unexpected_sham_signal = bool(
                sham_gain is not None
                and sham_gain >= MINIMUM_GAIN
                and all(value > 0 for value in sham_folds.values())
                and sham_bootstrap["lower_simultaneous"] > 0
            )
            sham_sanity = dict(
                reference="constant",
                relative_gain=sham_gain,
                fold_mean_improvements=sham_folds,
                bootstrap=sham_bootstrap,
                unexpected_accessible_signal=unexpected_sham_signal,
                valid=not unexpected_sham_signal,
            )
            comparisons = {}
            for reference in REFERENCES:
                reference_view = model_results[model_name]["residual"]["families"][family]["packs"][reference]
                raw_view = model_results[model_name]["residual"]["families"][family]["packs"]["raw64"]
                comparisons[reference] = _comparison(
                    residual_errors[model_name][family][reference],
                    residual_errors[model_name][family]["raw64"],
                    folds,
                    reference_view["raw_metrics"]["pooled"]["body_mae"],
                    raw_view["raw_metrics"]["pooled"]["body_mae"],
                    bootstrap[("raw", model_name, family, reference)],
                    procedural_sham_valid=not unexpected_sham_signal,
                )
            family_pass = bool(all(value["passes"] for value in comparisons.values()))
            family_decisions[family] = dict(
                comparisons=comparisons,
                sham_sanity=sham_sanity,
                passes_all_references=family_pass,
            )
            model_results[model_name]["residual"]["families"][family].update(
                comparisons=comparisons,
                sham_sanity=sham_sanity,
                passes_all_references=family_pass,
            )
            direct_family = model_results[model_name]["direct_log_target"]["families"][family]
            direct_family["comparisons"] = {}
            for reference in REFERENCES:
                reference_view = direct_family["packs"][reference]
                raw_view = direct_family["packs"]["raw64"]
                direct_family["comparisons"][reference] = _descriptive_comparison(
                    direct_errors[model_name][family][reference],
                    direct_errors[model_name][family]["raw64"],
                    folds,
                    reference_view["raw_metrics"]["pooled"]["body_mae"],
                    raw_view["raw_metrics"]["pooled"]["body_mae"],
                )
        controls_pass = bool(all(
            family_decisions[family]["comparisons"][reference]["passes"]
            for family in FAMILIES for reference in ("constant", "H_only64")
        ))
        h_pass = bool(all(
            family_decisions[family]["comparisons"]["h64"]["passes"]
            for family in FAMILIES
        ))
        decision_models[model_name] = dict(
            families=family_decisions,
            raw_signal_beyond_controls=controls_pass,
            raw_beats_h=h_pass,
            encoder_bottleneck_evidence=bool(controls_pass and h_pass),
        )

    separate = decision_models["separate_key"]
    original = decision_models["original"]
    current_encoder_evidence = separate["encoder_bottleneck_evidence"]
    common_encoder_evidence = bool(
        current_encoder_evidence and original["encoder_bottleneck_evidence"]
    )
    if common_encoder_evidence:
        classification = "checkpoint_common_encoder_bottleneck_evidence"
    elif current_encoder_evidence:
        classification = "separate_key_encoder_bottleneck_evidence"
    elif separate["raw_signal_beyond_controls"] and not separate["raw_beats_h"]:
        classification = "raw_signal_found_but_encoder_comparison_inconclusive"
    else:
        classification = "bounded_probes_do_not_establish_missing_accessible_residual_signal"

    summary = dict(
        protocol=PROTOCOL,
        n_targets=n,
        folds={
            str(fold): dict(
                rows=int((folds == fold).sum()),
                series=int(len(np.unique(series[folds == fold]))),
            ) for fold in (0, 1)
        },
        cohort_contract=dict(
            dataset="insta_market_basket",
            history_length_inclusive=[MIN_HISTORY, MAX_HISTORY],
            source_layout="oldest-to-latest observed prefix in [N,32]; source padding ignored",
            raw64=(
                "all H observed events including first boundary gap; each slot "
                "[log1p(dt),log1p(quantity)]; latest-aligned zero padding"
            ),
            H_only64="valid slot [1,0], padding [0,0], latest-aligned",
            target_exclusion=(
                "raw constructor accepts only the caller-supplied observed prefix; the target "
                "is excluded by extraction contract and is not an input to build_raw64"
            ),
            dimensions=RAW_DIMENSIONS,
            matched_models=list(MODEL_NAMES),
        ),
        decoder_contract=dict(
            families=list(FAMILIES),
            ridge=RIDGE,
            random_width=RANDOM_WIDTH,
            decoder_seed=SEED,
            sham_seed=SHAM_SEED,
            target="log1p(quantity)-frozen base_logpred",
            direct_target_context="log1p(quantity), secondary and excluded from decision",
            bootstrap_repeats=int(bootstrap_repeats),
            primary_cells=PRIMARY_CELLS,
            simultaneous_lower_quantile=SIMULTANEOUS_LOWER_QUANTILE,
            minimum_relative_gain=MINIMUM_GAIN,
        ),
        models=model_results,
        decision=dict(
            by_model=decision_models,
            separate_key_encoder_bottleneck_evidence=current_encoder_evidence,
            checkpoint_common_encoder_bottleneck_evidence=common_encoder_evidence,
            classification=classification,
            required_for_encoder_evidence=(
                "raw64 beats constant, H_only64, and h64 for both fixed decoder families: "
                ">=1% pooled residual-MSE gain, positive improvement in both folds, positive "
                "simultaneous series-bootstrap lower bound, no body-MAE direction conflict, "
                "and a valid sham sanity check"
            ),
        ),
        interpretation=(
            "A positive result is bounded train-internal accessibility evidence. A negative "
            "result means these bounded probes did not find missing accessible residual signal; "
            "it is not proof that the raw history contains no additional information."
        ),
    )
    json.dumps(summary, allow_nan=False)
    if any(not np.isfinite(value).all() for value in output_arrays.values()):
        raise FloatingPointError("nonfinite output array")
    return summary, output_arrays
