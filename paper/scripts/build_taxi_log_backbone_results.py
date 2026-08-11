#!/usr/bin/env python3
"""Aggregate and qualify the Taxi log-regression backbone control."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 52, 62)
MODELS = ("rmtpp", "thp", "titantpp")
MODEL_LABELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "titantpp": "TitanTPP",
}
STRATUM_ORDER = ("all", "le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")
RUN_METRICS = (
    "best_val_nll",
    "best_val_nll_marker",
    "best_val_nll_time",
    "best_val_qty_mae",
    "best_val_qty_rmse",
    "mark_acc",
    "best_epoch",
)
QUANTITY_METRICS = ("qty_mae", "qty_rmse", "qty_bias")
EXPECTED_DATA_SHA = "b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--rmtpp-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def validate_contracts(
    new_root: Path,
    rmtpp_root: Path,
    new_contract: dict[str, Any],
    rmtpp_contract: dict[str, Any],
    new_runs: list[dict[str, str]],
    rmtpp_runs: list[dict[str, str]],
) -> None:
    for name, contract in (("new", new_contract), ("rmtpp", rmtpp_contract)):
        if contract.get("status") != "complete":
            raise ValueError(f"{name} contract is not complete")
        if contract.get("data_sha256") != EXPECTED_DATA_SHA:
            raise ValueError(f"{name} fixed-data SHA mismatch")
        if contract.get("evaluation_scope") != "validation_only":
            raise ValueError(f"{name} result is not validation-only")
        if parse_bool(contract.get("held_out_test_evaluated", True)):
            raise ValueError(f"{name} held-out test flag is unlocked")
        if tuple(contract.get("seeds", [])) != SEEDS:
            raise ValueError(f"{name} seed contract mismatch")
        if int(contract.get("epochs", -1)) != 300:
            raise ValueError(f"{name} epoch budget mismatch")
        if contract.get("split_rows") != new_contract.get("split_rows"):
            raise ValueError(f"{name} split-row contract mismatch")
        interface = contract.get("interface") or contract.get("interfaces", {}).get("log_regression")
        expected_interface = {
            "mode": "log_regression",
            "target": "log1p_demand_qty",
            "loss": "mse_on_log1p_quantity",
            "output_activation": "softplus",
            "inverse_transform": "expm1",
            "support": "nonnegative",
            "fitted_on": "train",
        }
        if any(interface.get(key) != value for key, value in expected_interface.items()):
            raise ValueError(f"{name} log-regression interface mismatch")
        early_stopping = contract.get("early_stopping", {})
        if int(early_stopping.get("min_epochs", -1)) != 50:
            raise ValueError(f"{name} early-stopping minimum mismatch")
        if int(early_stopping.get("patience", -1)) != 60:
            raise ValueError(f"{name} early-stopping patience mismatch")
        if early_stopping.get("restore") != "best_val_nll":
            raise ValueError(f"{name} checkpoint restoration mismatch")

    if len(new_runs) != 6 or len(rmtpp_runs) != 3:
        raise ValueError("Expected six new runs and three RMTPP reference runs")
    expected_new = {(model, seed) for model in ("thp", "titantpp") for seed in SEEDS}
    observed_new = {(row["backbone"], int(row["seed"])) for row in new_runs}
    if observed_new != expected_new:
        raise ValueError("New backbone run grid mismatch")
    if {int(row["seed"]) for row in rmtpp_runs} != set(SEEDS):
        raise ValueError("RMTPP reference seed grid mismatch")
    for row in new_runs + rmtpp_runs:
        if row["status"] != "success":
            raise ValueError("At least one run is not successful")
        if row["evaluation_scope"] != "validation_only":
            raise ValueError("At least one run is not validation-only")
        if parse_bool(row["held_out_test_evaluated"]):
            raise ValueError("At least one run has a held-out test evaluation")
        if float(row["preclamp_negative_share"]) != 0.0:
            raise ValueError("At least one run violates nonnegative output support")

    test_artifacts = [
        path
        for root in (new_root, rmtpp_root)
        for path in root.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    if test_artifacts:
        raise ValueError(f"Unexpected held-out test artifacts: {test_artifacts}")


def combine_runs(
    new_runs: list[dict[str, str]],
    rmtpp_runs: list[dict[str, str]],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for row in rmtpp_runs:
        combined.append({
            "model": "rmtpp",
            "model_label": MODEL_LABELS["rmtpp"],
            "seed": int(row["seed"]),
            "best_epoch": int(row["best_epoch"]),
            "best_val_nll": float(row["best_val_nll"]),
            "best_val_nll_marker": "",
            "best_val_nll_time": "",
            "best_val_qty_mae": float(row["best_val_qty_mae"]),
            "best_val_qty_rmse": float(row["best_val_qty_rmse"]),
            "mark_acc": float(row["mark_acc"]),
            "preclamp_negative_share": float(row["preclamp_negative_share"]),
            "source_revision": row["source_revision"],
        })
    for row in new_runs:
        model = row["backbone"]
        combined.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "seed": int(row["seed"]),
            "best_epoch": int(row["best_epoch"]),
            "best_val_nll": float(row["best_val_nll"]),
            "best_val_nll_marker": float(row["best_val_nll_marker"]),
            "best_val_nll_time": float(row["best_val_nll_time"]),
            "best_val_qty_mae": float(row["best_val_qty_mae"]),
            "best_val_qty_rmse": float(row["best_val_qty_rmse"]),
            "mark_acc": float(row["mark_acc"]),
            "preclamp_negative_share": float(row["preclamp_negative_share"]),
            "source_revision": row["source_revision"],
        })
    return sorted(combined, key=lambda row: (MODELS.index(row["model"]), row["seed"]))


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model in MODELS:
        group = [row for row in rows if row["model"] == model]
        if {row["seed"] for row in group} != set(SEEDS):
            raise ValueError(f"Seed coverage failed for {model}")
        output: dict[str, Any] = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "n_seeds": len(group),
        }
        for metric in RUN_METRICS:
            values = [row[metric] for row in group if row[metric] != ""]
            if not values:
                output[f"{metric}_mean"] = ""
                output[f"{metric}_std"] = ""
                continue
            mean, std = mean_std([float(value) for value in values])
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
        summaries.append(output)
    return summaries


def combine_quantity_rows(
    new_rows: list[dict[str, str]],
    rmtpp_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for row in rmtpp_rows:
        if row["variant"] != "log_regression":
            continue
        combined.append({
            "model": "rmtpp",
            "model_label": MODEL_LABELS["rmtpp"],
            "seed": int(row["seed"]),
            "stratum_order": int(row["stratum_order"]),
            "stratum": row["stratum"],
            "stratum_label": row["stratum_label"],
            "share": float(row["share"]),
            "count": int(row["count"]),
            **{metric: float(row[metric]) for metric in QUANTITY_METRICS},
        })
    for row in new_rows:
        model = row["backbone"]
        combined.append({
            "model": model,
            "model_label": MODEL_LABELS[model],
            "seed": int(row["seed"]),
            "stratum_order": int(row["stratum_order"]),
            "stratum": row["stratum"],
            "stratum_label": row["stratum_label"],
            "share": float(row["share"]),
            "count": int(row["count"]),
            **{metric: float(row[metric]) for metric in QUANTITY_METRICS},
        })
    expected = {(model, seed, stratum) for model in MODELS for seed in SEEDS for stratum in STRATUM_ORDER}
    observed = {(row["model"], row["seed"], row["stratum"]) for row in combined}
    if observed != expected or len(combined) != len(expected):
        raise ValueError("Quantity stratum grid mismatch")
    validation_count = next(row["count"] for row in combined if row["stratum"] == "all")
    reference_counts = {
        row["stratum"]: row["count"]
        for row in combined
        if row["model"] == "rmtpp" and row["seed"] == SEEDS[0]
    }
    if sum(reference_counts[key] for key in STRATUM_ORDER if key != "all") != validation_count:
        raise ValueError("Stratum counts do not sum to validation count")
    for row in combined:
        if row["count"] != reference_counts[row["stratum"]]:
            raise ValueError("Stratum counts differ across model or seed")
    return sorted(
        combined,
        key=lambda row: (MODELS.index(row["model"]), row["seed"], row["stratum_order"]),
    )


def summarize_quantity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model in MODELS:
        for stratum in STRATUM_ORDER:
            group = [row for row in rows if row["model"] == model and row["stratum"] == stratum]
            output: dict[str, Any] = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "stratum_order": group[0]["stratum_order"],
                "stratum": stratum,
                "stratum_label": group[0]["stratum_label"],
                "count": group[0]["count"],
                "share": group[0]["share"],
                "n_seeds": len(group),
            }
            for metric in QUANTITY_METRICS:
                mean, std = mean_std([row[metric] for row in group])
                output[f"{metric}_mean"] = mean
                output[f"{metric}_std"] = std
            summaries.append(output)
    return summaries


def paired_deltas(run_rows: list[dict[str, Any]], quantity_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    run_index = {(row["model"], row["seed"]): row for row in run_rows}
    quantity_index = {
        (row["model"], row["seed"], row["stratum"]): row for row in quantity_rows
    }
    for baseline in ("rmtpp", "thp"):
        for metric in ("best_val_nll", "best_val_qty_mae", "best_val_qty_rmse", "mark_acc"):
            deltas = [
                run_index[("titantpp", seed)][metric] - run_index[(baseline, seed)][metric]
                for seed in SEEDS
            ]
            baseline_mean = statistics.mean(run_index[(baseline, seed)][metric] for seed in SEEDS)
            output.append({
                "baseline": baseline,
                "baseline_label": MODEL_LABELS[baseline],
                "scope": "overall",
                "metric": metric,
                "titan_minus_baseline_mean": statistics.mean(deltas),
                "titan_minus_baseline_std": statistics.stdev(deltas),
                "relative_delta_pct": 100.0 * statistics.mean(deltas) / baseline_mean,
                "titan_better_seeds": sum(
                    delta < 0 if metric != "mark_acc" else delta > 0 for delta in deltas
                ),
            })
        for stratum in STRATUM_ORDER:
            for metric in ("qty_mae", "qty_rmse"):
                deltas = [
                    quantity_index[("titantpp", seed, stratum)][metric]
                    - quantity_index[(baseline, seed, stratum)][metric]
                    for seed in SEEDS
                ]
                baseline_mean = statistics.mean(
                    quantity_index[(baseline, seed, stratum)][metric] for seed in SEEDS
                )
                output.append({
                    "baseline": baseline,
                    "baseline_label": MODEL_LABELS[baseline],
                    "scope": stratum,
                    "metric": metric,
                    "titan_minus_baseline_mean": statistics.mean(deltas),
                    "titan_minus_baseline_std": statistics.stdev(deltas),
                    "relative_delta_pct": 100.0 * statistics.mean(deltas) / baseline_mean,
                    "titan_better_seeds": sum(delta < 0 for delta in deltas),
                })
    return output


def metric_lookup(rows: list[dict[str, Any]], model: str, metric: str) -> tuple[float, float]:
    row = next(item for item in rows if item["model"] == model)
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def quantity_lookup(
    rows: list[dict[str, Any]], model: str, stratum: str, metric: str
) -> tuple[float, float]:
    row = next(item for item in rows if item["model"] == model and item["stratum"] == stratum)
    return float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])


def fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def write_briefing(
    output_dir: Path,
    run_summary: list[dict[str, Any]],
    quantity_summary: list[dict[str, Any]],
    new_contract: dict[str, Any],
    rmtpp_contract: dict[str, Any],
) -> None:
    lines = [
        "# Taxi Log-Regression Backbone Control",
        "",
        "## Qualification",
        "",
        "- Status: **qualified validation-only backbone control**.",
        "- All three backbones use the same fixed Taxi split, seeds 42/52/62, log1p-MSE quantity target, softplus output, expm1 reconstruction, and best-validation-NLL checkpoint rule.",
        "- Held-out test evaluation remains locked; no test artifacts are present.",
        f"- Fixed data SHA-256: `{new_contract['data_sha256']}`.",
        f"- THP/TitanTPP source revision: `{new_contract['source_revision']}`; RMTPP reference source revision: `{rmtpp_contract['source_revision']}`.",
        "- RMTPP marker/time NLL components were not persisted by the earlier runner, so those two cells remain unavailable. Total NLL is directly comparable.",
        "",
        "## Overall Validation Results",
        "",
        "| Model | NLL | Quantity MAE | Quantity RMSE | Mark accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        nll = metric_lookup(run_summary, model, "best_val_nll")
        mae = metric_lookup(run_summary, model, "best_val_qty_mae")
        rmse = metric_lookup(run_summary, model, "best_val_qty_rmse")
        acc = metric_lookup(run_summary, model, "mark_acc")
        lines.append(
            f"| {MODEL_LABELS[model]} | {fmt(*nll, 4)} | {fmt(*mae)} | {fmt(*rmse)} | {fmt(*acc, 4)} |"
        )
    lines.extend([
        "",
        "## Validation NLL Decomposition",
        "",
        "| Model | Marker NLL | Time NLL |",
        "|---|---:|---:|",
        "| Adapted RMTPP | Not persisted | Not persisted |",
    ])
    for model in ("thp", "titantpp"):
        marker = metric_lookup(run_summary, model, "best_val_nll_marker")
        time = metric_lookup(run_summary, model, "best_val_nll_time")
        lines.append(f"| {MODEL_LABELS[model]} | {fmt(*marker, 4)} | {fmt(*time, 4)} |")
    lines.extend([
        "",
        "## Quantity MAE by Train-Derived Stratum",
        "",
        "| Model | <=p50 | p50-p90 | p90-p95 | p95-p99 | >p99 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for model in MODELS:
        values = [fmt(*quantity_lookup(quantity_summary, model, stratum, "qty_mae")) for stratum in STRATUM_ORDER[1:]]
        lines.append(f"| {MODEL_LABELS[model]} | " + " | ".join(values) + " |")

    rmtpp_mae, _ = metric_lookup(run_summary, "rmtpp", "best_val_qty_mae")
    thp_mae, _ = metric_lookup(run_summary, "thp", "best_val_qty_mae")
    titan_mae, _ = metric_lookup(run_summary, "titantpp", "best_val_qty_mae")
    rmtpp_nll, _ = metric_lookup(run_summary, "rmtpp", "best_val_nll")
    thp_nll, _ = metric_lookup(run_summary, "thp", "best_val_nll")
    titan_nll, _ = metric_lookup(run_summary, "titantpp", "best_val_nll")
    lines.extend([
        "",
        "## Decision",
        "",
        f"TitanTPP improves overall quantity MAE by `{100.0 * (thp_mae - titan_mae) / thp_mae:.1f}%` and NLL by `{100.0 * (thp_nll - titan_nll) / thp_nll:.1f}%` relative to Adapted THP.",
        "The NLL improvement over THP is consistent in all three seeds; the quantity MAE and RMSE improvements occur in two of three seeds.",
        f"However, Adapted RMTPP remains better than TitanTPP: TitanTPP has `{100.0 * (titan_mae - rmtpp_mae) / rmtpp_mae:.1f}%` higher overall quantity MAE and `{100.0 * (titan_nll - rmtpp_nll) / rmtpp_nll:.1f}%` higher NLL.",
        "RMTPP is better than TitanTPP on NLL, quantity MAE, and quantity RMSE in all three paired seeds.",
        "The stratum results do not rescue a broad Taxi backbone claim: Adapted RMTPP has the lowest mean quantity MAE in every train-derived quantity stratum.",
        "Therefore, Taxi does **not** support the claim that the Titan backbone is generally superior under a controlled log-regression head. It only supports the narrower statement that TitanTPP outperforms the tested THP backbone on this dataset.",
        "A central long-sequence contribution requires the same head-controlled comparison on a dataset where long histories are the defining condition. The new Intermittent dataset is the remaining go/no-go experiment.",
        "",
        "## Manuscript Use",
        "",
        "- Do not present Taxi as evidence of universal Titan-backbone superiority.",
        "- The result may be reported as a controlled negative/mixed finding or retained as an internal qualification result.",
        "- Do not combine this result with the exponent-plus-residual runs as if the quantity heads were identical.",
        "- This is a three-seed validation comparison, not a held-out test result or a statistical significance claim.",
    ])
    (output_dir / "qualification_briefing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    new_contract = read_json(args.new_root / "launch_contract.json")
    rmtpp_contract = read_json(args.rmtpp_root / "launch_contract.json")
    new_runs = read_csv(args.new_root / "run_summaries.csv")
    rmtpp_runs = read_csv(args.rmtpp_root / "run_summaries.csv")
    new_quantity = read_csv(args.new_root / "quantity_seed_metrics.csv")
    rmtpp_quantity = read_csv(args.rmtpp_root / "quantity_interface_seed_metrics.csv")

    validate_contracts(
        args.new_root,
        args.rmtpp_root,
        new_contract,
        rmtpp_contract,
        new_runs,
        rmtpp_runs,
    )
    combined_runs = combine_runs(new_runs, rmtpp_runs)
    run_summary = summarize_runs(combined_runs)
    combined_quantity = combine_quantity_rows(new_quantity, rmtpp_quantity)
    quantity_summary = summarize_quantity(combined_quantity)
    deltas = paired_deltas(combined_runs, combined_quantity)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "combined_seed_metrics.csv", combined_runs)
    write_csv(args.output_dir / "combined_model_summary.csv", run_summary)
    write_csv(args.output_dir / "combined_quantity_seed_metrics.csv", combined_quantity)
    write_csv(args.output_dir / "combined_quantity_summary.csv", quantity_summary)
    write_csv(args.output_dir / "paired_titan_deltas.csv", deltas)
    write_briefing(args.output_dir, run_summary, quantity_summary, new_contract, rmtpp_contract)
    print(f"Qualified Taxi log-backbone results written to {args.output_dir}")


if __name__ == "__main__":
    main()
