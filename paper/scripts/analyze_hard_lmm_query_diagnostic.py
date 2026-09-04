#!/usr/bin/env python3
"""Apply the preregistered evidence gate to verified frozen train caches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch

from paper.scripts.hard_lmm_query_diagnostic_analysis import analyze_cache
from paper.scripts.run_hard_lmm_query_diagnostic import (
    CONTRACT, RAW, RESULT, check, digest, read, save, source_hashes,
)


def decide(analyses, policy):
    insta = analyses["insta_market_basket"]["separate_key"]
    taxi = analyses["yellow_trip_hourly"]["separate_key"]
    gain = insta["pooled"]["two_stat_relative_improvement_vs_h_only"]
    taxi_gain = taxi["pooled"]["two_stat_relative_improvement_vs_h_only"]
    checks = {
        "instacart_pooled_improvement_at_least_1pct": gain is not None and gain >= policy["instacart_pooled_relative_mse_improvement_vs_h_only_minimum"],
        "instacart_both_folds_improve": all(
            fold["two_stat_relative_improvement_vs_h_only"] is not None and fold["two_stat_relative_improvement_vs_h_only"] > 0
            for fold in insta["folds"].values()),
        "instacart_beats_constant": insta["pooled"]["mse"]["two_stat"] < insta["pooled"]["mse"]["constant"],
        "instacart_cluster_lower_band_positive": insta["cluster_bootstrap"]["paired_delta_lower_p05"] > 0,
        "taxi_regression_at_most_1pct": taxi_gain is not None and taxi_gain >= -policy["taxi_pooled_relative_mse_regression_vs_h_only_maximum"],
    }
    passed = all(checks.values())
    return {"evidence_passed": passed, "checks": checks,
            "decision": "define_conditional_query_candidate" if passed else "hold_query_implementation",
            "instacart_relative_mse_improvement": gain, "taxi_relative_mse_improvement": taxi_gain,
            "candidate_performance_proven": False, "held_out_test_evaluated": False,
            "interpretation": "Fixed train-internal decodability gate; not an independent model-generalization result."}


def independent_mse_check(cache, summary, predictions):
    y = cache["log_residual"].double().numpy()
    for scope in ("pooled", "0", "1"):
        selected = np.ones(len(y), dtype=bool) if scope == "pooled" else cache["fold"].numpy() == int(scope)
        metrics = summary["pooled"] if scope == "pooled" else summary["folds"][scope]
        for control in ("constant", "h_only", "two_stat"):
            predicted = predictions[f"{control}_correction"].numpy()
            actual = np.mean((y[selected] - predicted[selected]) ** 2)
            np.testing.assert_allclose(actual, metrics["mse"][control], rtol=1e-12, atol=1e-12)
    for field in ("fold", "series_index", "target_index"):
        check(torch.equal(cache[field], predictions[field]), f"Prediction identity mismatch: {field}")


def execute():
    check(not (RESULT / "analysis.json").exists(), "Refusing to overwrite diagnostic analysis")
    contract = read(CONTRACT)
    manifest = read(RESULT / "execution_manifest.json")
    check(manifest["status"] == "extracted", "Extraction must be complete and verified")
    check(digest(CONTRACT) == manifest["contract_sha256"], "Preregistered contract changed")
    check(source_hashes() == manifest["source_files"], "Source changed since extraction")
    for file, expected in manifest["reference_document_hashes"].items():
        check(digest(ROOT / file) == expected, f"Reference changed: {file}")
    torch.set_num_threads(contract["sampling"]["torch_threads"])
    started = time.monotonic()
    analyses, rows, output_hashes = {}, [], {}
    for name in contract["datasets"]:
        analyses[name] = {}
        for label in ("original", "separate_key"):
            entry = manifest["datasets"][name]["models"][label]
            path = ROOT / entry["cache_path"]
            check(digest(path) == entry["cache_sha256"], "Frozen cache changed")
            cache = torch.load(path, map_location="cpu", weights_only=True)
            summary, predictions = analyze_cache(cache, return_predictions=True)
            check(summary["neighbors"] == contract["crossfit"]["neighbor_count"] and
                  summary["ridge_summed_sse_penalty"] == contract["crossfit"]["ridge"] and
                  summary["min_series_per_fold"] == contract["sampling"]["minimum_distinct_series_per_fold"], "Analysis constants differ from contract")
            independent_mse_check(cache, summary, predictions)
            summary["independent_numpy_metric_check_passed"] = True
            prediction_path = RAW / name / f"{label}_crossfit.pt"
            check(not prediction_path.exists(), "Crossfit output already exists")
            torch.save(predictions, prediction_path)
            output_hashes[str(prediction_path.relative_to(ROOT))] = digest(prediction_path)
            analyses[name][label] = summary
            for scope, metrics in [("pooled", summary["pooled"]), *summary["folds"].items()]:
                rows.append({"dataset": name, "model": label, "scope": scope,
                    "targets": metrics["n_targets"], **{f"{key}_mse": value for key, value in metrics["mse"].items()},
                    "two_stat_relative_improvement_vs_h_only": metrics["two_stat_relative_improvement_vs_h_only"]})
            print(f"Analyzed {name}/{label}", flush=True)
    decision = decide(analyses, contract["evidence_gate"])
    decision["integrity_verified"] = True
    check(digest(CONTRACT) == manifest["contract_sha256"] and source_hashes() == manifest["source_files"], "Source or contract changed during analysis")
    save(RESULT / "analysis.json", analyses)
    save(RESULT / "evidence_decision.json", decision)
    pl.DataFrame(rows).write_csv(RESULT / "crossfit_metrics.csv")
    save(RESULT / "analysis_verification.json", {"status": "passed", "completed_at": datetime.now(timezone.utc).isoformat(),
        "seconds": time.monotonic()-started, "contract_sha256": manifest["contract_sha256"],
        "output_hashes": output_hashes, "independent_numpy_metric_checks": 4,
        "all_metrics_finite": True, "validation_or_test_inference": False, "backbone_parameter_updates": False,
        "plots_generated": False})
    print(decision, flush=True)


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    execute()
