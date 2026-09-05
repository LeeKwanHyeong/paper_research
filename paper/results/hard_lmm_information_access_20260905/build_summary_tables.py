"""Export frozen diagnostic results without refitting or changing decisions."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
analysis = json.loads((HERE / "analysis.json").read_text())
decision = json.loads((HERE / "evidence_decision.json").read_text())


def write_csv(name, rows):
    with (HERE / name).open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


metrics, reconstruction, comparisons, usefulness = [], [], [], []
# Order is the frozen F12 order, not selected from reconstruction results.
features = (
    "log_history_length", "mean_log_quantity", "last_log_quantity",
    "std_log_quantity", "recent3_mean_log_quantity",
    "last_minus_mean_log_quantity", "normalized_rank_log_quantity_slope",
    "mean_internal_log_gap", "last_internal_log_gap",
    "std_internal_log_gap", "log_internal_span", "age_distortion",
)
for dataset, models in analysis.items():
    for model, result in models.items():
        cell = dict(dataset=dataset, model=model)
        for family, decoders in result["decoders"].items():
            baseline = decoders["constant"]["residual_mse"]["pooled"]
            for pack, decoded in decoders.items():
                metrics.append(dict(
                    **cell, family=family, pack=pack,
                    input_dimensions=decoded["input_dimensions"],
                    residual_mse=decoded["residual_mse"]["pooled"],
                    gain_vs_constant_pct=100 * (
                        1 - decoded["residual_mse"]["pooled"] / baseline),
                    fold0_mse=decoded["residual_mse"]["folds"]["0"],
                    fold1_mse=decoded["residual_mse"]["folds"]["1"],
                    **decoded["raw_metrics"]["pooled"],
                    **decoded["raw_metrics"]["clamps"],
                ))
            useful = result["history_usefulness"][family]
            usefulness.append(dict(
                **cell, family=family,
                gain_vs_constant_pct=100 * useful["relative_gain"],
                fold0_mse_improvement=useful["fold_mean_improvements"]["0"],
                fold1_mse_improvement=useful["fold_mean_improvements"]["1"],
                family_passes=useful["passes"],
                both_families_pass=result["history_usefulness_passes"],
            ))
        for stage, decoded in result["reconstruction"].items():
            if stage == "Ftruth":
                continue
            for index, feature in enumerate(features):
                reconstruction.append(dict(
                    **cell, stage=stage, feature=feature,
                    oof_r2=decoded["per_feature_oof_r2"][index],
                    mse=decoded["per_feature_mse"][index],
                    constant_mse=decoded["per_feature_constant_mse"][index],
                ))
        for name, contrast in result["contrasts"].items():
            for family, compared in contrast["families"].items():
                for reference, comparison in compared["comparisons"].items():
                    comparisons.append(dict(
                        **cell, contrast=name, family=family,
                        candidate=contrast["candidate"], reference=reference,
                        residual_mse_gain_pct=100 * comparison["relative_gain"],
                        candidate_mse=comparison["candidate_mse"],
                        reference_mse=comparison["reference_mse"],
                        fold0_mse_improvement=comparison["fold_mean_improvements"]["0"],
                        fold1_mse_improvement=comparison["fold_mean_improvements"]["1"],
                        bootstrap_lower_family_quantile=comparison["bootstrap"]["lower_family_quantile"],
                        **comparison["conditions"],
                        comparison_passes=comparison["passes"],
                        eligible_cell_passes=decision["contrasts"][name]["cells"][f"{dataset}/{model}"],
                        common_passes=decision["contrasts"][name]["passes"],
                    ))

write_csv("decoder_metrics.csv", metrics)
write_csv("feature_reconstruction.csv", reconstruction)
write_csv("contrast_checks.csv", comparisons)
write_csv("history_usefulness.csv", usefulness)
print(json.dumps(dict(decoder_rows=len(metrics), reconstruction_rows=len(reconstruction),
                      comparison_rows=len(comparisons), usefulness_rows=len(usefulness))))
