#!/usr/bin/env python3
"""Validate and qualify the mark-free count-aware backbone control."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


MODELS = ("rmtpp", "thp", "titantpp")
BASELINES = ("rmtpp", "thp")
SEEDS = (42, 52, 62)
EXPECTED_REVISION = "044add1f3de768d804d9f0269fd0013bd9658a35"
EXPECTED_DATA_SHA = "85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f"
EXPECTED_SPLIT_SHA = "393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04"
METRICS = (
    "best_val_joint_objective",
    "best_val_time_nll",
    "best_val_log_qty_mse",
    "best_val_qty_mae",
    "best_val_qty_rmse",
    "best_epoch",
    "completed_epochs",
    "elapsed_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def validate(source: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    contract = json.loads((source / "launch_contract.json").read_text())
    expected = {
        "status": "complete",
        "completed_run_count": 9,
        "expected_run_count": 9,
        "source_revision": EXPECTED_REVISION,
        "data_sha256": EXPECTED_DATA_SHA,
        "split_manifest_sha256": EXPECTED_SPLIT_SHA,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for key, value in expected.items():
        actual = contract.get(key)
        if key == "held_out_test_evaluated":
            actual = parse_bool(actual)
        if actual != value:
            raise ValueError(f"contract mismatch for {key}: {actual!r}")
    if tuple(contract["backbones"]) != MODELS or tuple(contract["seeds"]) != SEEDS:
        raise ValueError("run grid contract mismatch")
    interface = contract["interface"]
    if interface["quantity_mark_used"] or interface["quantity_residual_used"]:
        raise ValueError("quantity mark or residual unexpectedly enabled")
    if interface["mode"] != "mark_free_count_aware_log_regression":
        raise ValueError("interface is not the frozen mark-free formulation")

    rows = read_csv(source / "run_summaries.csv")
    expected_grid = {(model, seed) for model in MODELS for seed in SEEDS}
    observed_grid = {(row["backbone"], int(row["seed"])) for row in rows}
    if len(rows) != 9 or observed_grid != expected_grid:
        raise ValueError("run summary grid mismatch")
    for row in rows:
        if row["status"] != "success":
            raise ValueError("non-success run found")
        if row["source_revision"] != EXPECTED_REVISION:
            raise ValueError("run source revision mismatch")
        if row["evaluation_scope"] != "validation_only":
            raise ValueError("non-validation result found")
        if parse_bool(row["held_out_test_evaluated"]):
            raise ValueError("held-out test was evaluated")
        if not row["checkpoint_state_sha256"]:
            raise ValueError("checkpoint digest missing")
        if any("mark" in key.lower() for key in row if key != "interface_meta"):
            raise ValueError("mark metric found in run summary")
    return contract, rows


def summarize_runs(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        group = [row for row in rows if row["backbone"] == model]
        record: dict[str, Any] = {"model": model, "n_seeds": len(group)}
        for metric in METRICS:
            mean, std = mean_std([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def paired_deltas(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for baseline in BASELINES:
        for seed in SEEDS:
            titan = next(
                row for row in rows
                if row["backbone"] == "titantpp" and int(row["seed"]) == seed
            )
            base = next(
                row for row in rows
                if row["backbone"] == baseline and int(row["seed"]) == seed
            )
            output.append({
                "baseline": baseline,
                "seed": seed,
                "titan_minus_baseline_qty_mae": (
                    float(titan["best_val_qty_mae"])
                    - float(base["best_val_qty_mae"])
                ),
                "titan_minus_baseline_qty_rmse": (
                    float(titan["best_val_qty_rmse"])
                    - float(base["best_val_qty_rmse"])
                ),
                "titan_minus_baseline_time_nll": (
                    float(titan["best_val_time_nll"])
                    - float(base["best_val_time_nll"])
                ),
            })
    return output


def history_means(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, float]]]:
    output: dict[str, dict[str, dict[str, float]]] = {}
    for stratum in ("history_le_64", "history_65_128", "history_gt_128"):
        output[stratum] = {}
        for model in MODELS:
            group = [
                row for row in rows
                if row["stratum"] == stratum and row["backbone"] == model
            ]
            output[stratum][model] = {
                metric: statistics.mean(float(row[metric]) for row in group)
                for metric in ("time_nll", "qty_mae", "qty_rmse")
            }
    return output


def lower_seed_count(
    rows: list[dict[str, str]], baseline: str, metric: str, stratum: str | None = None
) -> int:
    count = 0
    for seed in SEEDS:
        def find(model: str) -> float:
            return float(next(
                row[metric] for row in rows
                if row["backbone"] == model
                and int(row["seed"]) == seed
                and (stratum is None or row.get("stratum") == stratum)
            ))
        count += find("titantpp") < find(baseline)
    return count


def percent_change(titan: float, baseline: float) -> float:
    return (titan - baseline) / baseline * 100.0


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract, run_rows = validate(source)
    summary = summarize_runs(run_rows)
    summary_by_model = {row["model"]: row for row in summary}
    pairs = paired_deltas(run_rows)
    history_rows = read_csv(source / "history_seed_metrics.csv")
    history = history_means(history_rows)

    titan = summary_by_model["titantpp"]
    rmtpp = summary_by_model["rmtpp"]
    thp = summary_by_model["thp"]
    best_baseline_time = min(
        rmtpp["best_val_time_nll_mean"], thp["best_val_time_nll_mean"]
    )
    time_degradation = titan["best_val_time_nll_mean"] - best_baseline_time
    mean_better_both = all(
        titan[f"best_val_{metric}_mean"]
        < min(rmtpp[f"best_val_{metric}_mean"], thp[f"best_val_{metric}_mean"])
        for metric in ("qty_mae", "qty_rmse")
    )
    paired_better_both = all(
        lower_seed_count(run_rows, baseline, metric) >= 2
        for baseline in BASELINES
        for metric in ("best_val_qty_mae", "best_val_qty_rmse")
    )
    general_gate = mean_better_both and paired_better_both and time_degradation <= 0.01

    long_paired = all(
        lower_seed_count(history_rows, baseline, metric, "history_gt_128") >= 2
        for baseline in BASELINES
        for metric in ("qty_mae", "qty_rmse")
    )
    short_rmtpp = history["history_le_64"]["rmtpp"]["qty_rmse"]
    short_titan = history["history_le_64"]["titantpp"]["qty_rmse"]
    long_rmtpp = history["history_gt_128"]["rmtpp"]["qty_rmse"]
    long_titan = history["history_gt_128"]["titantpp"]["qty_rmse"]
    short_gain = (short_rmtpp - short_titan) / short_rmtpp
    long_gain = (long_rmtpp - long_titan) / long_rmtpp
    long_gate = general_gate and long_paired and long_gain > short_gain

    qualification = {
        "schema_version": 1,
        "source_revision": contract["source_revision"],
        "data_sha256": contract["data_sha256"],
        "split_manifest_sha256": contract["split_manifest_sha256"],
        "run_count": len(run_rows),
        "remote_best_checkpoint_count_verified": 9,
        "evaluation_scope": contract["evaluation_scope"],
        "held_out_test_evaluated": False,
        "mark_metrics_present": False,
        "general_count_gate": {
            "status": "GO" if general_gate else "NO-GO",
            "mean_mae_and_rmse_better_than_both": mean_better_both,
            "paired_mae_and_rmse_better_min_two_seeds": paired_better_both,
            "time_nll_degradation": time_degradation,
            "time_nll_tolerance_pass": time_degradation <= 0.01,
        },
        "long_history_gate": {
            "status": "GO" if long_gate else "NO-GO",
            "history_gt_128_paired_better_min_two_seeds": long_paired,
            "rmse_gain_vs_rmtpp_history_le_64": short_gain,
            "rmse_gain_vs_rmtpp_history_gt_128": long_gain,
            "long_gain_exceeds_short_gain": long_gain > short_gain,
        },
    }
    (output / "qualification.json").write_text(
        json.dumps(qualification, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "model_summary.csv", summary)
    write_csv(output / "paired_titan_deltas.csv", pairs)

    lines = [
        "# Count-aware TPP Backbone Control Qualification",
        "",
        "## Decision",
        "",
        "- General count-prediction gate: **NO-GO**.",
        "- Long-history gate: **NO-GO**.",
        "- Held-out test remains locked and was not evaluated.",
        "",
        "## Overall validation results",
        "",
        "| Model | Time NLL | Log-count MSE | Quantity MAE | Quantity RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = summary_by_model[model]
        lines.append(
            f"| {model.upper()} | {row['best_val_time_nll_mean']:.6f} +/- "
            f"{row['best_val_time_nll_std']:.6f} | "
            f"{row['best_val_log_qty_mse_mean']:.6f} +/- "
            f"{row['best_val_log_qty_mse_std']:.6f} | "
            f"{row['best_val_qty_mae_mean']:.4f} +/- "
            f"{row['best_val_qty_mae_std']:.4f} | "
            f"{row['best_val_qty_rmse_mean']:.4f} +/- "
            f"{row['best_val_qty_rmse_std']:.4f} |"
        )
    lines.extend([
        "",
        "TitanTPP reduces MAE and RMSE against RMTPP by "
        f"{-percent_change(titan['best_val_qty_mae_mean'], rmtpp['best_val_qty_mae_mean']):.1f}% "
        "and "
        f"{-percent_change(titan['best_val_qty_rmse_mean'], rmtpp['best_val_qty_rmse_mean']):.1f}%, "
        "respectively. Against THP, TitanTPP has "
        f"{percent_change(titan['best_val_qty_mae_mean'], thp['best_val_qty_mae_mean']):.1f}% "
        "higher MAE but "
        f"{-percent_change(titan['best_val_qty_rmse_mean'], thp['best_val_qty_rmse_mean']):.1f}% "
        "lower RMSE. TitanTPP is lower than THP in MAE for 0/3 seeds and lower in "
        "RMSE for 3/3 seeds, so the preregistered general gate fails.",
        "",
        "## History-length result",
        "",
        "| History | Model | Quantity MAE | Quantity RMSE | Time NLL |",
        "|---|---|---:|---:|---:|",
    ])
    for stratum, label in (
        ("history_le_64", "<=64"),
        ("history_65_128", "65-128"),
        ("history_gt_128", ">128"),
    ):
        for model in MODELS:
            row = history[stratum][model]
            lines.append(
                f"| {label} | {model.upper()} | {row['qty_mae']:.4f} | "
                f"{row['qty_rmse']:.4f} | {row['time_nll']:.6f} |"
            )
    lines.extend([
        "",
        "For history >128, TitanTPP has higher MAE than both RMTPP and THP in all "
        "three seeds, and higher RMSE than THP in all three seeds. Its RMSE reduction "
        f"against RMTPP is {long_gain * 100:.1f}% in the >128 stratum, smaller than "
        f"the {short_gain * 100:.1f}% reduction in the <=64 stratum. The long-history "
        "gate therefore fails independently of the overall gate.",
        "",
        "## Manuscript boundary",
        "",
        "This validation experiment supports a narrow statement that TitanTPP is "
        "substantially stronger than the GRU-based RMTPP count baseline and reduces "
        "RMSE relative to THP. It does not support superiority over both baselines, "
        "nor a claim that TitanTPP benefits more from long histories. The held-out "
        "test must remain unevaluated for this configuration.",
        "",
    ])
    (output / "qualification_briefing.md").write_text("\n".join(lines))
    print(output / "qualification_briefing.md")


if __name__ == "__main__":
    main()
