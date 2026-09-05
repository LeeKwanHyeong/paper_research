#!/usr/bin/env python3
"""Export the frozen raw-history diagnostic into compact review tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ANALYSIS = json.loads((HERE / "analysis.json").read_text())
MODELS = ("original", "separate_key")
TASKS = (("direct_log_target", "direct"), ("residual", "frozen_base_residual"))
FAMILIES = ("linear", "random128")
PACKS = ("constant", "H_only64", "raw64", "h64", "sham64")
REFERENCES = ("constant", "H_only64", "h64")


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


metrics = []
comparisons = []
for model in MODELS:
    for section, task in TASKS:
        for family in FAMILIES:
            result = ANALYSIS["models"][model][section]["families"][family]
            constant_mse = result["packs"]["constant"]["mse"]["pooled"]
            for pack in PACKS:
                view = result["packs"][pack]
                metrics.append({
                    "model": model,
                    "task": task,
                    "family": family,
                    "pack": pack,
                    "input_dimensions": view["input_dimensions"],
                    "log_mse": view["mse"]["pooled"],
                    "gain_vs_constant_pct": 100 * (constant_mse - view["mse"]["pooled"]) / constant_mse,
                    "fold0_log_mse": view["mse"]["folds"]["0"],
                    "fold1_log_mse": view["mse"]["folds"]["1"],
                    "body_mae": view["raw_metrics"]["pooled"]["body_mae"],
                    "rmse": view["raw_metrics"]["pooled"]["rmse"],
                    "tail_mae": view["raw_metrics"]["pooled"]["tail_mae"],
                    "below_zero_clamps": view["raw_metrics"]["clamps"]["below_zero"],
                    "above_twenty_clamps": view["raw_metrics"]["clamps"]["above_twenty"],
                })
            for reference in REFERENCES:
                compared = result["comparisons"][reference]
                conditions = compared.get("conditions", {})
                bootstrap = compared.get("bootstrap", {})
                comparisons.append({
                    "model": model,
                    "task": task,
                    "family": family,
                    "candidate": "raw64",
                    "reference": reference,
                    "candidate_log_mse": compared["candidate_mse"],
                    "reference_log_mse": compared["reference_mse"],
                    "relative_gain_pct": 100 * compared["relative_gain"],
                    "fold0_mse_improvement": compared["fold_mean_improvements"]["0"],
                    "fold1_mse_improvement": compared["fold_mean_improvements"]["1"],
                    "bootstrap_simultaneous_lower": bootstrap.get("lower_simultaneous"),
                    "pooled_gain_at_least_one_percent": conditions.get("pooled_gain_at_least_one_percent"),
                    "both_folds_positive": conditions.get("both_folds_positive"),
                    "bootstrap_simultaneous_lower_positive": conditions.get(
                        "bootstrap_simultaneous_lower_positive"
                    ),
                    "body_mae_not_worse": conditions.get("no_log_vs_body_direction_conflict"),
                    "procedural_sham_valid": conditions.get("procedural_sham_valid"),
                    "passes_primary_gate": compared.get("passes"),
                    "interpretation": compared.get("interpretation", "primary frozen-base residual comparison"),
                })

write_csv("decoder_metrics.csv", metrics)
write_csv("raw64_comparisons.csv", comparisons)
print(json.dumps({"decoder_metric_rows": len(metrics), "comparison_rows": len(comparisons)}))
