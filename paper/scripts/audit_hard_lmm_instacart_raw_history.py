#!/usr/bin/env python3
"""Independently audit the saved Instacart raw-history OOF evidence.

This audit deliberately consumes the frozen predictions rather than refitting
the probes.  It verifies the artifact chain, recomputes every saved OOF MSE and
quantity metric, and checks the result directions used in the reader-facing
conclusion.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "paper/results/hard_lmm_instacart_raw_history_20260905"
PREDICTIONS = ROOT / "search_artifacts/hard_lmm_instacart_raw_history_20260905/oof_predictions.npz"
AUDIT = RESULT / "independent_audit.json"
MODELS = ("original", "separate_key")
FAMILIES = ("linear", "random128")
PACKS = ("constant", "H_only64", "raw64", "h64", "sham64")
REFERENCES = ("constant", "H_only64", "h64")
BOOTSTRAP_REPEATS = 10000
BOOTSTRAP_SEED = 20260905
SIMULTANEOUS_LOWER_QUANTILE = 0.05 / (2 * 12)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, *, atol=1e-12):
    require(np.isclose(actual, expected, rtol=1e-12, atol=atol),
            f"{message}: {actual!r} != {expected!r}")


def raw_metrics(log_prediction, quantity, folds, body_threshold, tail_threshold):
    prediction = np.expm1(np.clip(log_prediction, 0.0, 20.0))
    absolute = np.abs(prediction - quantity)
    squared = (prediction - quantity) ** 2
    body = quantity <= body_threshold
    tail = quantity > tail_threshold

    def view(selected):
        selected_body = selected & body
        selected_tail = selected & tail
        return {
            "n": int(selected.sum()),
            "body_n": int(selected_body.sum()),
            "tail_n": int(selected_tail.sum()),
            "body_mae": float(absolute[selected_body].mean()),
            "rmse": float(np.sqrt(squared[selected].mean())),
            "tail_mae": float(absolute[selected_tail].mean()),
        }

    return {
        "pooled": view(np.ones(len(quantity), dtype=bool)),
        "folds": {str(fold): view(folds == fold) for fold in (0, 1)},
        "clamps": {
            "below_zero": int((log_prediction < 0).sum()),
            "above_twenty": int((log_prediction > 20).sum()),
        },
    }


def compare_metric_views(actual, expected, label):
    for scope in ("pooled", "folds"):
        actual_scope = actual[scope]
        expected_scope = expected[scope]
        if scope == "pooled":
            pairs = [("pooled", actual_scope, expected_scope)]
        else:
            pairs = [(fold, actual_scope[fold], expected_scope[fold]) for fold in ("0", "1")]
        for sublabel, left, right in pairs:
            for name in ("n", "body_n", "tail_n"):
                require(left[name] == right[name], f"{label}/{sublabel}/{name} changed")
            for name in ("body_mae", "rmse", "tail_mae"):
                close(left[name], right[name], f"{label}/{sublabel}/{name}")
    require(actual["clamps"] == expected["clamps"], f"{label}/clamps changed")


def paired_series_bootstrap(improvements, series, folds):
    """Recompute the shared-draw fixed-OOF bootstrap without analysis imports."""
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    groups = []
    for fold in (0, 1):
        rows = np.flatnonzero(folds == fold)
        _, inverse = np.unique(series[rows], return_inverse=True)
        size = int(inverse.max()) + 1
        sums = np.column_stack([
            np.bincount(inverse, weights=improvements[rows, column], minlength=size)
            for column in range(improvements.shape[1])
        ])
        counts = np.bincount(inverse, minlength=size)
        groups.append((sums, counts, size))
    values = np.empty((BOOTSTRAP_REPEATS, improvements.shape[1]), dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPEATS, 32):
        batch = min(32, BOOTSTRAP_REPEATS - start)
        total = np.zeros((batch, improvements.shape[1]), dtype=np.float64)
        count = np.zeros(batch, dtype=np.float64)
        for sums, counts, size in groups:
            selected = generator.integers(0, size, size=(batch, size))
            total += sums[selected].sum(axis=1)
            count += counts[selected].sum(axis=1)
        values[start:start + batch] = total / count[:, None]
    return [{
        "lower_p05": float(np.quantile(values[:, column], 0.05)),
        "lower_simultaneous": float(np.quantile(
            values[:, column], SIMULTANEOUS_LOWER_QUANTILE,
        )),
        "median": float(np.median(values[:, column])),
    } for column in range(improvements.shape[1])]


def verify_artifact_chain(manifest, analysis, decision):
    require(manifest["status"] == "complete", "execution is not complete")
    for flag in ("validation_rows_materialized", "held_out_rows_materialized",
                 "model_parameter_updates", "server_accessed", "target_gaps_read"):
        require(manifest[flag] is False, f"forbidden execution flag is true: {flag}")
    require(manifest["target_quantities_read"] is True, "target extraction was not recorded")
    require(manifest["checkpoints_restored"] is True, "checkpoint extraction was not recorded")
    require(manifest["cohort"]["kept_rows"] == 65525, "cohort size changed")
    require(manifest["cohort"]["unique_series"] == 65525, "series are not unique")
    require(manifest["cohort"]["history_min"] == 3, "minimum history changed")
    require(manifest["cohort"]["history_max"] == 32, "maximum history changed")

    hashed = {
        "paper/contracts/hard_lmm_instacart_raw_history_access_v1.json": manifest["contract_sha256"],
        "paper/results/hard_lmm_instacart_raw_history_20260905/input_manifest.json": manifest["input_manifest_sha256"],
        "paper/results/hard_lmm_instacart_raw_history_20260905/analysis.json": manifest["analysis_sha256"],
        "paper/results/hard_lmm_instacart_raw_history_20260905/evidence_decision.json": manifest["evidence_decision_sha256"],
        manifest["input_cache"]["path"]: manifest["input_cache"]["sha256"],
        manifest["probe_cache"]["path"]: manifest["probe_cache"]["sha256"],
        manifest["oof_predictions"]["path"]: manifest["oof_predictions"]["sha256"],
    }
    for model in MODELS:
        record = manifest["checkpoints"][model]
        require(record["state_unchanged"] is True, f"{model} state changed during inference")
        hashed[record["path"]] = record["file_sha256"]
    for relative, expected in hashed.items():
        path = ROOT / relative
        require(path.is_file(), f"missing recorded artifact: {relative}")
        require(digest(path) == expected, f"recorded hash changed: {relative}")
    require(analysis["decision"] == decision, "standalone decision differs from analysis")
    return hashed


def main():
    manifest = read(RESULT / "execution_manifest.json")
    analysis = read(RESULT / "analysis.json")
    decision = read(RESULT / "evidence_decision.json")
    hashed = verify_artifact_chain(manifest, analysis, decision)

    body_threshold = 25.0
    tail_threshold = 35.0
    verified_cells = 0
    verified_bootstrap_cells = 0
    key_metrics = {}
    with np.load(PREDICTIONS) as arrays:
        n = analysis["n_targets"]
        require(n == 65525, "analysis target count changed")
        quantity = arrays["quantity"]
        log_quantity = arrays["log_quantity"]
        folds = arrays["fold"]
        series = arrays["series_id"]
        require(quantity.shape == log_quantity.shape == folds.shape == series.shape == (n,),
                "saved row vectors have the wrong shape")
        require(np.isfinite(quantity).all() and np.isfinite(log_quantity).all(),
                "saved targets are nonfinite")
        require(np.allclose(log_quantity, np.log1p(quantity), rtol=1e-6, atol=1e-6),
                "saved log targets changed")
        require(set(folds.tolist()) == {0, 1}, "saved folds changed")
        require(np.unique(series).size == n, "saved series are not one-context unique")
        require(not np.intersect1d(series[folds == 0], series[folds == 1]).size,
                "series overlap across folds")
        require(arrays["raw64"].shape == arrays["H_only64"].shape == (n, 64),
                "raw/control dimensions changed")
        require(np.isfinite(arrays["raw64"]).all(), "raw64 is nonfinite")

        for model in MODELS:
            base = arrays[f"{model}__base_logpred"]
            residual = arrays[f"{model}__residual_label"]
            require(arrays[f"{model}__h64"].shape == (n, 64), f"{model} h64 dimensions changed")
            require(np.isfinite(arrays[f"{model}__h64"]).all(), f"{model} h64 is nonfinite")
            require(np.array_equal(residual, log_quantity - base), f"{model} residual label changed")
            base_metrics = raw_metrics(base, quantity, folds, body_threshold, tail_threshold)
            compare_metric_views(base_metrics, analysis["models"][model]["base_raw_metrics"],
                                 f"{model}/base")
            close(float(np.mean(residual ** 2)),
                  analysis["models"][model]["uncorrected_residual_mse"]["pooled"],
                  f"{model}/uncorrected_residual_mse")

            for task, target, prediction_base, section in (
                ("direct", log_quantity, np.zeros(n), "direct_log_target"),
                ("residual", residual, base, "residual"),
            ):
                for family in FAMILIES:
                    packs = analysis["models"][model][section]["families"][family]["packs"]
                    for pack in PACKS:
                        prefix = f"{model}__{task}__{family}__{pack}"
                        prediction = arrays[f"{prefix}__prediction"]
                        saved_error = arrays[f"{prefix}__squared_error"]
                        expected_error = (target - prediction) ** 2
                        require(np.array_equal(saved_error, expected_error),
                                f"{prefix} squared errors changed")
                        view = packs[pack]
                        close(float(saved_error.mean()), view["mse"]["pooled"], f"{prefix}/mse")
                        for fold in (0, 1):
                            close(float(saved_error[folds == fold].mean()),
                                  view["mse"]["folds"][str(fold)], f"{prefix}/fold{fold}/mse")
                        recomputed = raw_metrics(prediction_base + prediction, quantity, folds,
                                                 body_threshold, tail_threshold)
                        compare_metric_views(recomputed, view["raw_metrics"], prefix)
                        verified_cells += 1

            direct = analysis["models"][model]["direct_log_target"]["families"]
            residual_views = analysis["models"][model]["residual"]["families"]
            key_metrics[model] = {}
            for family in FAMILIES:
                direct_packs = direct[family]["packs"]
                residual_packs = residual_views[family]["packs"]
                direct_raw = direct_packs["raw64"]["mse"]["pooled"]
                direct_constant = direct_packs["constant"]["mse"]["pooled"]
                direct_h = direct_packs["h64"]["mse"]["pooled"]
                require(direct_raw < direct_constant, f"{model}/{family}: raw lacks direct signal")
                require(direct_h < direct_raw, f"{model}/{family}: raw unexpectedly beats h")
                residual_raw = residual_packs["raw64"]["mse"]["pooled"]
                residual_constant = residual_packs["constant"]["mse"]["pooled"]
                residual_h = residual_packs["h64"]["mse"]["pooled"]
                require(residual_raw > residual_constant,
                        f"{model}/{family}: raw unexpectedly beats residual constant")
                require(residual_raw > residual_h,
                        f"{model}/{family}: raw unexpectedly beats residual h")
                h_comparison = decision["by_model"][model]["families"][family]["comparisons"]["h64"]
                require(h_comparison["passes"] is False, f"{model}/{family}: h gate unexpectedly passed")
                require(all(value < 0 for value in h_comparison["fold_mean_improvements"].values()),
                        f"{model}/{family}: raw-vs-h fold direction changed")
                require(h_comparison["bootstrap"]["lower_simultaneous"] < 0,
                        f"{model}/{family}: raw-vs-h bootstrap direction changed")
                require(decision["by_model"][model]["families"][family]["sham_sanity"]["valid"] is True,
                        f"{model}/{family}: sham guard invalid")
                key_metrics[model][family] = {
                    "direct_log_mse": {
                        "constant": direct_constant,
                        "raw64": direct_raw,
                        "h64": direct_h,
                        "raw_gain_vs_constant": (direct_constant - direct_raw) / direct_constant,
                        "raw_gain_vs_h64": (direct_h - direct_raw) / direct_h,
                    },
                    "frozen_base_residual_mse": {
                        "constant": residual_constant,
                        "raw64": residual_raw,
                        "h64": residual_h,
                        "raw_gain_vs_constant": (residual_constant - residual_raw) / residual_constant,
                        "raw_gain_vs_h64": (residual_h - residual_raw) / residual_h,
                    },
                }

        bootstrap_keys = []
        bootstrap_columns = []
        for model in MODELS:
            for family in FAMILIES:
                prefix = f"{model}__residual__{family}"
                raw_error = arrays[f"{prefix}__raw64__squared_error"]
                for reference in REFERENCES:
                    bootstrap_keys.append(("raw", model, family, reference))
                    bootstrap_columns.append(
                        arrays[f"{prefix}__{reference}__squared_error"] - raw_error
                    )
                bootstrap_keys.append(("sham", model, family, "constant"))
                bootstrap_columns.append(
                    arrays[f"{prefix}__constant__squared_error"]
                    - arrays[f"{prefix}__sham64__squared_error"]
                )
        bootstrap = paired_series_bootstrap(np.column_stack(bootstrap_columns), series, folds)
        for key, recomputed in zip(bootstrap_keys, bootstrap, strict=True):
            kind, model, family, reference = key
            family_decision = decision["by_model"][model]["families"][family]
            if kind == "raw":
                saved = family_decision["comparisons"][reference]["bootstrap"]
            else:
                saved = family_decision["sham_sanity"]["bootstrap"]
            for metric in ("lower_p05", "lower_simultaneous", "median"):
                close(recomputed[metric], saved[metric],
                      f"bootstrap/{kind}/{model}/{family}/{reference}/{metric}")
            require(saved["repeats"] == BOOTSTRAP_REPEATS, "bootstrap repeats changed")
            require(saved["seed"] == BOOTSTRAP_SEED, "bootstrap seed changed")
            verified_bootstrap_cells += 1

    require(decision["separate_key_encoder_bottleneck_evidence"] is False,
            "separate-key encoder decision changed")
    require(decision["checkpoint_common_encoder_bottleneck_evidence"] is False,
            "checkpoint-common encoder decision changed")
    require(decision["classification"] ==
            "bounded_probes_do_not_establish_missing_accessible_residual_signal",
            "classification changed")

    reader_artifacts = {}
    for relative in (
        "paper/results/hard_lmm_instacart_raw_history_20260905/README.md",
        "paper/results/hard_lmm_instacart_raw_history_20260905/build_summary_tables.py",
        "paper/results/hard_lmm_instacart_raw_history_20260905/decoder_metrics.csv",
        "paper/results/hard_lmm_instacart_raw_history_20260905/raw64_comparisons.csv",
        "paper/results/hard_lmm_instacart_raw_history_20260905/pytest.xml",
    ):
        path = ROOT / relative
        require(path.is_file(), f"missing reader artifact: {relative}")
        reader_artifacts[relative] = digest(path)

    audit = {
        "audit_completed_at": datetime.now(timezone.utc).isoformat(),
        "auditor": "independent saved-OOF arithmetic and artifact-chain audit",
        "auditor_script_sha256": digest(Path(__file__)),
        "passed": True,
        "verified_oof_metric_cells": verified_cells,
        "verified_bootstrap_cells": verified_bootstrap_cells,
        "verified_artifact_hashes": hashed,
        "reader_artifact_hashes": reader_artifacts,
        "cohort": {
            "rows": analysis["n_targets"],
            "folds": analysis["folds"],
            "one_context_per_series": True,
            "history_length_inclusive": [3, 32],
        },
        "key_metrics": key_metrics,
        "decision": {
            "classification": decision["classification"],
            "separate_key_encoder_bottleneck_evidence": False,
            "checkpoint_common_encoder_bottleneck_evidence": False,
        },
        "conclusion": (
            "Raw64 carries direct next-quantity signal but is worse than h64 in both fixed "
            "decoder families. It also fails to improve frozen-base residual MSE over h64 "
            "or the constant correction. These bounded train-internal probes do not support "
            "an encoder information-loss bottleneck."
        ),
        "limits": [
            "The audit recomputes saved OOF arithmetic; it does not refit probes or rerun checkpoint inference.",
            "The negative result is bounded to latest-aligned linear and fixed-random-feature probes.",
            "Train-internal accessibility does not establish held-out generalization or irreducible noise.",
        ],
    }
    AUDIT.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "verified_oof_metric_cells": verified_cells,
                      "verified_bootstrap_cells": verified_bootstrap_cells,
                      "classification": decision["classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()
