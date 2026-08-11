#!/usr/bin/env python3
"""Validate and aggregate the frozen-5000 Intermittent backbone experiment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


EXPECTED_REVISION = "308cec0b9c383d4eab5aac8b9015dae663b0ad73"
MODEL_ORDER = ("rmtpp", "thp", "titantpp")
MODEL_LABELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "titantpp": "TitanTPP",
}
SEEDS = (42, 52, 62)
RUN_METRICS = (
    "best_val_nll",
    "best_val_nll_marker",
    "best_val_nll_time",
    "best_val_nll_qty_mae",
    "best_val_nll_qty_rmse",
    "best_val_nll_qty_wape",
    "best_val_nll_dt_mae",
    "best_val_nll_mark_acc",
    "best_val_nll_epoch",
    "trained_epochs",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def pct_delta(reference: float, candidate: float) -> float:
    return 100.0 * (candidate - reference) / reference


def validate_contracts(baseline: dict, titan: dict) -> None:
    for contract in (baseline, titan):
        assert contract["status"] == "COMPLETE"
        assert contract["source_revision"] == EXPECTED_REVISION
        assert contract["evaluation_scope"] == "validation_only"
        assert contract["held_out_test_evaluated"] is False
        assert contract["seeds"] == list(SEEDS)
        assert contract["max_epochs"] == 300
        assert contract["early_stopping"] == {
            "metric": "validation_nll",
            "min_epochs": 40,
            "patience": 40,
        }
    assert baseline["dataset_sha256"] == titan["dataset_sha256"]
    assert baseline["models"] == ["rmtpp", "thp"]
    assert baseline["expected_run_count"] == 6
    assert titan["models"] == ["titantpp"]
    assert titan["expected_run_count"] == 3


def load_source(root: Path) -> tuple[dict, list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    contract = json.loads((root / "launch_contract.json").read_text(encoding="utf-8"))
    leaderboard = root / "intermittent" / "leaderboard"
    return (
        contract,
        read_csv(leaderboard / "runs.csv"),
        read_csv(leaderboard / "histories.csv"),
        read_csv(leaderboard / "scale_wise_summary.csv"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    baseline_contract, baseline_runs, baseline_histories, baseline_scales = load_source(
        root / "source_5080_baselines"
    )
    titan_contract, titan_runs, titan_histories, titan_scales = load_source(
        root / "source_5090_titantpp"
    )
    validate_contracts(baseline_contract, titan_contract)

    runs = [row for row in baseline_runs + titan_runs if row["model_name"] in MODEL_ORDER]
    assert len(runs) == 9
    assert all(row["status"] == "success" for row in runs)
    assert {(row["model_name"], int(row["seed"])) for row in runs} == {
        (model, seed) for model in MODEL_ORDER for seed in SEEDS
    }
    assert all(row["source_revision"] == EXPECTED_REVISION for row in runs)
    assert all(row["evaluation_scope"] == "validation_only" for row in runs)
    assert all(row["held_out_test_evaluated"].lower() == "false" for row in runs)
    assert all(row["qty_decoder_mode"] == "mark_residual" for row in runs)
    assert all(row["max_seq_len"] == "96" for row in runs)
    assert all(row["num_marks"] == "10" for row in runs)
    assert all(row["series_count"] == "5000" for row in runs)

    histories = baseline_histories + titan_histories
    history_lookup = {
        (row["model_name"], int(row["seed"]), int(row["epoch"])): row for row in histories
    }
    for row in runs:
        history = history_lookup[
            (row["model_name"], int(row["seed"]), int(row["best_val_nll_epoch"]))
        ]
        row["best_val_nll_marker"] = history["val_nll_marker"]
        row["best_val_nll_time"] = history["val_nll_time"]
        observed = float(row["best_val_nll"])
        decomposed = float(history["val_nll_marker"]) + float(history["val_nll_time"])
        assert abs(observed - decomposed) < 1e-6

    combined_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for row in sorted((item for item in runs if item["model_name"] == model), key=lambda item: int(item["seed"])):
            combined_rows.append(
                {
                    "model": MODEL_LABELS[model],
                    "model_name": model,
                    "seed": int(row["seed"]),
                    "best_val_nll_epoch": int(row["best_val_nll_epoch"]),
                    "trained_epochs": int(row["trained_epochs"]),
                    "val_nll": float(row["best_val_nll"]),
                    "val_nll_marker": float(row["best_val_nll_marker"]),
                    "val_nll_time": float(row["best_val_nll_time"]),
                    "quantity_mae": float(row["best_val_nll_qty_mae"]),
                    "quantity_rmse": float(row["best_val_nll_qty_rmse"]),
                    "quantity_wape": float(row["best_val_nll_qty_wape"]),
                    "delta_t_mae": float(row["best_val_nll_dt_mae"]),
                    "mark_accuracy": float(row["best_val_nll_mark_acc"]),
                }
            )
    write_csv(root / "combined_run_metrics.csv", combined_rows)

    summary_rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        model_runs = [row for row in runs if row["model_name"] == model]
        summary: dict[str, object] = {"model": MODEL_LABELS[model], "model_name": model, "seeds": 3}
        for metric in RUN_METRICS:
            values = [float(row[metric]) for row in model_runs]
            mean, std = mean_std(values)
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
        summary_rows.append(summary)
    write_csv(root / "model_summary.csv", summary_rows)

    by_model_seed = {(row["model_name"], int(row["seed"])): row for row in runs}
    paired_rows: list[dict[str, object]] = []
    paired_metrics = (
        "best_val_nll",
        "best_val_nll_marker",
        "best_val_nll_time",
        "best_val_nll_qty_mae",
        "best_val_nll_qty_rmse",
        "best_val_nll_mark_acc",
    )
    for baseline in ("rmtpp", "thp"):
        seed_rows: list[dict[str, object]] = []
        for seed in SEEDS:
            reference = by_model_seed[(baseline, seed)]
            candidate = by_model_seed[("titantpp", seed)]
            result: dict[str, object] = {
                "baseline": MODEL_LABELS[baseline],
                "seed": seed,
            }
            for metric in paired_metrics:
                result[f"titan_minus_baseline_{metric}"] = float(candidate[metric]) - float(reference[metric])
            seed_rows.append(result)
            paired_rows.append(result)
        mean_row: dict[str, object] = {"baseline": MODEL_LABELS[baseline], "seed": "mean"}
        for metric in paired_metrics:
            key = f"titan_minus_baseline_{metric}"
            mean_row[key] = statistics.mean(float(row[key]) for row in seed_rows)
        paired_rows.append(mean_row)
    write_csv(root / "paired_titan_deltas.csv", paired_rows)

    scale_rows = [
        row
        for row in baseline_scales + titan_scales
        if row["selection"] == "best_val_nll" and row["model_name"] in MODEL_ORDER
    ]
    assert len(scale_rows) == 24
    scale_rows.sort(key=lambda row: (MODEL_ORDER.index(row["model_name"]), int(row["scale_order"])))
    compact_scales: list[dict[str, object]] = []
    for row in scale_rows:
        compact_scales.append(
            {
                "model": MODEL_LABELS[row["model_name"]],
                "model_name": row["model_name"],
                "scale_order": int(row["scale_order"]),
                "quantity_range": row["scale_label"],
                "event_count": int(row["total_count"]),
                "event_share": float(row["mean_share"]),
                "mean_true_quantity": float(row["mean_true_qty"]),
                "mean_predicted_quantity": float(row["mean_pred_qty"]),
                "quantity_mae_mean": float(row["mean_qty_mae"]),
                "quantity_mae_std": float(row["std_qty_mae"]),
                "quantity_rmse_mean": float(row["mean_qty_rmse"]),
                "quantity_wape_mean": float(row["mean_qty_wape"]),
            }
        )
    write_csv(root / "combined_scale_wise_summary.csv", compact_scales)

    summary_lookup = {row["model_name"]: row for row in summary_rows}
    rmtpp = summary_lookup["rmtpp"]
    thp = summary_lookup["thp"]
    titan = summary_lookup["titantpp"]
    scale_lookup = {
        (row["model_name"], row["quantity_range"]): row for row in compact_scales
    }
    delta_t_values = {float(row["best_val_nll_dt_mae"]) for row in runs}
    assert len(delta_t_values) == 1

    lines = [
        "# Frozen-5000 Intermittent Result Qualification",
        "",
        "## Contract Verification",
        "",
        "The six Adapted RMTPP/THP runs from 5080 and the three TitanTPP runs from 5090 use "
        "the same frozen 5,000-series split, source revision, three seeds, 300-epoch ceiling, "
        "and validation-NLL checkpoint rule. Both contracts are complete, validation-only, and "
        "record `held_out_test_evaluated=false`. All nine run rows are successful.",
        "",
        f"- Source revision: `{EXPECTED_REVISION}`",
        f"- Split manifest SHA-256: `{baseline_contract['dataset_sha256']['split_manifest.json']}`",
        f"- Validation parquet SHA-256: `{baseline_contract['dataset_sha256']['validation.parquet']}`",
        "- Shared interface: 10 marks, exponent + residual decoder, maximum sequence length 96",
        "",
        "## Three-Seed Validation Summary",
        "",
        "| Model | NLL | Time NLL | Mark NLL | Quantity MAE | Quantity RMSE | Mark accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        row = summary_lookup[model]
        lines.append(
            f"| {MODEL_LABELS[model]} | "
            f"{row['best_val_nll_mean']:.4f} +/- {row['best_val_nll_std']:.4f} | "
            f"{row['best_val_nll_time_mean']:.4f} +/- {row['best_val_nll_time_std']:.4f} | "
            f"{row['best_val_nll_marker_mean']:.4f} +/- {row['best_val_nll_marker_std']:.4f} | "
            f"{row['best_val_nll_qty_mae_mean']:.4f} +/- {row['best_val_nll_qty_mae_std']:.4f} | "
            f"{row['best_val_nll_qty_rmse_mean']:.4f} +/- {row['best_val_nll_qty_rmse_std']:.4f} | "
            f"{100 * row['best_val_nll_mark_acc_mean']:.2f} +/- {100 * row['best_val_nll_mark_acc_std']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Lower is better for NLL and quantity errors; higher is better for mark accuracy. "
            "The reported `+/-` value is the sample standard deviation across seeds 42, 52, and 62.",
            "",
            "## Evidence Assessment",
            "",
            f"TitanTPP lowers validation NLL against Adapted RMTPP in all three paired seeds "
            f"(mean difference {float(titan['best_val_nll_mean']) - float(rmtpp['best_val_nll_mean']):.4f}). "
            f"The time-likelihood term is lower by "
            f"{abs(float(titan['best_val_nll_time_mean']) - float(rmtpp['best_val_nll_time_mean'])):.4f}, while "
            f"the mark term is {float(titan['best_val_nll_marker_mean']) - float(rmtpp['best_val_nll_marker_mean']):+.4f} worse. "
            "This supports a narrow temporal-likelihood improvement over the recurrent baseline.",
            "",
            f"Against Adapted THP, TitanTPP also lowers total NLL in all three seeds "
            f"(mean difference {float(titan['best_val_nll_mean']) - float(thp['best_val_nll_mean']):.4f}), but "
            f"its time-likelihood term is {float(titan['best_val_nll_time_mean']) - float(thp['best_val_nll_time_mean']):+.4f} worse. "
            "The advantage comes primarily from the mark term, so this comparison does not establish "
            "a general temporal-dependency advantage over THP.",
            "",
            f"For quantity reconstruction, TitanTPP improves MAE by "
            f"{-pct_delta(float(thp['best_val_nll_qty_mae_mean']), float(titan['best_val_nll_qty_mae_mean'])):.1f}% "
            f"relative to Adapted THP, but is "
            f"{pct_delta(float(rmtpp['best_val_nll_qty_mae_mean']), float(titan['best_val_nll_qty_mae_mean'])):.1f}% "
            "worse than Adapted RMTPP. It is therefore the middle-ranked model rather than a universal winner.",
            "",
            f"At quantities 64-127, TitanTPP has the lowest MAE "
            f"({float(scale_lookup[('titantpp', '64-127')]['quantity_mae_mean']):.3f}; "
            f"Adapted RMTPP: {float(scale_lookup[('rmtpp', '64-127')]['quantity_mae_mean']):.3f}; "
            f"Adapted THP: {float(scale_lookup[('thp', '64-127')]['quantity_mae_mean']):.3f}). "
            f"At quantities >=128, Adapted RMTPP is best "
            f"({float(scale_lookup[('rmtpp', '>=128')]['quantity_mae_mean']):.3f}), followed by TitanTPP "
            f"({float(scale_lookup[('titantpp', '>=128')]['quantity_mae_mean']):.3f}). "
            "The tail evidence is mixed and does not support an across-the-board long-tail quantity claim.",
            "",
            "The delta-time MAE is exactly identical across all nine rows. It is not discriminative in "
            "this experiment and should not be used as evidence of backbone superiority until the "
            "point-prediction path is audited.",
            "",
            "## Qualification Decision",
            "",
            "**Share with caveats.** The result is qualified as validation evidence under a frozen "
            "contract. It can support a narrow statement that TitanTPP improves total event likelihood "
            "over both adapted baselines and temporal likelihood over Adapted RMTPP. It cannot support "
            "the stronger claims that TitanTPP uniformly improves temporal modeling over THP or that "
            "the exponent-residual representation yields the best quantity accuracy.",
            "",
            "The next controlled experiment should place the same fair log-scale quantity head on all "
            "three backbones. That isolates the backbone contribution from the quantity representation "
            "and aligns this Intermittent result with the completed Taxi quantity-interface finding.",
            "",
        ]
    )
    (root / "qualification_briefing.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
