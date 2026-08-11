#!/usr/bin/env python3
"""Aggregate and qualify the Intermittent log-head backbone control."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


MODELS = ("rmtpp", "thp", "titantpp")
LABELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "titantpp": "TitanTPP",
}
RUN_METRICS = (
    "best_val_nll",
    "best_val_nll_marker",
    "best_val_nll_time",
    "best_val_qty_mae",
    "best_val_qty_rmse",
    "mark_acc",
    "best_epoch",
    "parameter_count",
)
BREAKDOWN_METRICS = (
    "nll",
    "nll_marker",
    "nll_time",
    "mark_acc",
    "qty_mae",
    "qty_rmse",
    "qty_bias",
)
EXPECTED_DATA_SHA = "85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f"
EXPECTED_MANIFEST_SHA = "393158a54a8ca703dbf7e9311b9dff6d2825ef737e3e3de1c30a1f3ff64c1c04"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--titan-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=300)
    parser.add_argument("--seeds", default="42,52,62")
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


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def validate_contracts(
    baseline_root: Path,
    titan_root: Path,
    baseline: dict[str, Any],
    titan: dict[str, Any],
    seeds: tuple[int, ...],
    expected_epochs: int,
) -> None:
    expected_common = {
        "status": "complete",
        "experiment": "intermittent_log_backbone_control",
        "dataset": "intermittent_frozen_5000",
        "data_sha256": EXPECTED_DATA_SHA,
        "split_manifest_sha256": EXPECTED_MANIFEST_SHA,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "max_seq_len": 256,
        "hidden_dim": 64,
        "epochs": expected_epochs,
    }
    for name, contract in (("baseline", baseline), ("titan", titan)):
        for key, value in expected_common.items():
            actual = contract.get(key)
            if key == "held_out_test_evaluated":
                actual = parse_bool(actual)
            if actual != value:
                raise ValueError(f"{name} contract mismatch for {key}: {actual!r}")
        if tuple(contract.get("seeds", [])) != seeds:
            raise ValueError(f"{name} seed contract mismatch")
        interface = contract.get("interface", {})
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
            raise ValueError(f"{name} quantity-interface mismatch")
        early = contract.get("early_stopping", {})
        if early.get("restore") != "best_val_nll":
            raise ValueError(f"{name} checkpoint restoration mismatch")
    for key in (
        "source_revision",
        "data_sha256",
        "split_manifest_sha256",
        "split_rows",
        "history_length_contract",
        "quantity_contract",
        "batch_size",
        "lr",
        "lambda_log",
        "lookback_weeks",
        "max_seq_len",
        "hidden_dim",
    ):
        if baseline.get(key) != titan.get(key):
            raise ValueError(f"Cross-server contract mismatch for {key}")
    baseline_interface = baseline["interface"]
    titan_interface = titan["interface"]
    for key in (
        "mode",
        "target",
        "loss",
        "output_activation",
        "inverse_transform",
        "history_quantity_input",
        "support",
        "fitted_on",
    ):
        if baseline_interface.get(key) != titan_interface.get(key):
            raise ValueError(f"Cross-server interface mismatch for {key}")
    for key in ("train_min", "train_max", "train_target_mean"):
        left = float(baseline_interface[key])
        right = float(titan_interface[key])
        if abs(left - right) > 1e-12 * max(1.0, abs(left), abs(right)):
            raise ValueError(f"Cross-server interface statistic mismatch for {key}")
    if set(baseline.get("backbones", [])) != {"rmtpp", "thp"}:
        raise ValueError("Baseline server does not contain RMTPP and THP")
    if set(titan.get("backbones", [])) != {"titantpp"}:
        raise ValueError("Titan server does not contain only TitanTPP")

    test_artifacts = [
        path
        for root in (baseline_root, titan_root)
        for path in root.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    if test_artifacts:
        raise ValueError(f"Unexpected test artifacts: {test_artifacts}")


def combine_rows(
    baseline_rows: list[dict[str, str]],
    titan_rows: list[dict[str, str]],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = baseline_rows + titan_rows
    expected = {(model, seed) for model in MODELS for seed in seeds}
    observed = {(row["backbone"], int(row["seed"])) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Run grid mismatch")
    output = []
    for row in rows:
        if row["status"] != "success":
            raise ValueError("At least one run is not successful")
        if row["evaluation_scope"] != "validation_only":
            raise ValueError("At least one run is not validation-only")
        if parse_bool(row["held_out_test_evaluated"]):
            raise ValueError("At least one run evaluated held-out test")
        if float(row["preclamp_negative_share"]) != 0.0:
            raise ValueError("At least one run violates nonnegative quantity support")
        output.append({
            "model": row["backbone"],
            "model_label": LABELS[row["backbone"]],
            "seed": int(row["seed"]),
            **{metric: float(row[metric]) for metric in RUN_METRICS},
            "completed_epochs": int(row["completed_epochs"]),
            "source_revision": row["source_revision"],
            "checkpoint_state_sha256": row["checkpoint_state_sha256"],
        })
    return sorted(output, key=lambda row: (MODELS.index(row["model"]), row["seed"]))


def summarize_runs(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    output = []
    for model in MODELS:
        group = [row for row in rows if row["model"] == model]
        if {row["seed"] for row in group} != set(seeds):
            raise ValueError(f"Run seed coverage failed for {model}")
        record: dict[str, Any] = {
            "model": model,
            "model_label": LABELS[model],
            "n_seeds": len(group),
        }
        for metric in RUN_METRICS:
            mean, std = mean_std([float(row[metric]) for row in group])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_std"] = std
        output.append(record)
    return output


def combine_breakdowns(
    baseline_rows: list[dict[str, str]],
    titan_rows: list[dict[str, str]],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows = baseline_rows + titan_rows
    strata = sorted({row["stratum"] for row in rows})
    expected = {
        (model, seed, stratum)
        for model in MODELS
        for seed in seeds
        for stratum in strata
    }
    observed = {(row["backbone"], int(row["seed"]), row["stratum"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError("Breakdown grid mismatch")
    output = []
    for row in rows:
        output.append({
            "model": row["backbone"],
            "model_label": LABELS[row["backbone"]],
            "seed": int(row["seed"]),
            "stratum_order": int(row["stratum_order"]),
            "stratum": row["stratum"],
            "stratum_label": row["stratum_label"],
            "share": float(row["share"]),
            "count": int(row["count"]),
            **{metric: float(row[metric]) for metric in BREAKDOWN_METRICS},
        })
    reference_counts = {
        row["stratum"]: row["count"]
        for row in output
        if row["model"] == "rmtpp" and row["seed"] == seeds[0]
    }
    for row in output:
        if row["count"] != reference_counts[row["stratum"]]:
            raise ValueError("Breakdown counts differ across model or seed")
    return sorted(
        output,
        key=lambda row: (MODELS.index(row["model"]), row["seed"], row["stratum_order"]),
    )


def summarize_breakdowns(rows: list[dict[str, Any]], seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    output = []
    strata = sorted({
        (row["stratum_order"], row["stratum"], row["stratum_label"])
        for row in rows
    })
    for model in MODELS:
        for order, key, label in strata:
            group = [row for row in rows if row["model"] == model and row["stratum"] == key]
            if {row["seed"] for row in group} != set(seeds):
                raise ValueError(f"Breakdown seed coverage failed for {model}/{key}")
            record: dict[str, Any] = {
                "model": model,
                "model_label": LABELS[model],
                "stratum_order": order,
                "stratum": key,
                "stratum_label": label,
                "count": group[0]["count"],
                "share": group[0]["share"],
                "n_seeds": len(group),
            }
            for metric in BREAKDOWN_METRICS:
                mean, std = mean_std([row[metric] for row in group])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
            output.append(record)
    return output


def paired_deltas(
    run_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    output = []
    run_index = {(row["model"], row["seed"]): row for row in run_rows}
    history_index = {
        (row["model"], row["seed"], row["stratum"]): row for row in history_rows
    }
    for baseline in ("rmtpp", "thp"):
        for metric in ("best_val_nll", "best_val_nll_time", "best_val_qty_mae"):
            deltas = [
                run_index[("titantpp", seed)][metric] - run_index[(baseline, seed)][metric]
                for seed in seeds
            ]
            output.append({
                "baseline": baseline,
                "baseline_label": LABELS[baseline],
                "scope": "overall",
                "metric": metric,
                "titan_minus_baseline_mean": statistics.mean(deltas),
                "titan_minus_baseline_std": mean_std(deltas)[1],
                "titan_better_seeds": sum(delta < 0 for delta in deltas),
            })
        for stratum in ("history_le_64", "history_65_128", "history_gt_128"):
            for metric in ("nll", "nll_time", "qty_mae"):
                deltas = [
                    history_index[("titantpp", seed, stratum)][metric]
                    - history_index[(baseline, seed, stratum)][metric]
                    for seed in seeds
                ]
                output.append({
                    "baseline": baseline,
                    "baseline_label": LABELS[baseline],
                    "scope": stratum,
                    "metric": metric,
                    "titan_minus_baseline_mean": statistics.mean(deltas),
                    "titan_minus_baseline_std": mean_std(deltas)[1],
                    "titan_better_seeds": sum(delta < 0 for delta in deltas),
                })
    return output


def find_mean(rows: list[dict[str, Any]], model: str, metric: str) -> float:
    row = next(item for item in rows if item["model"] == model)
    return float(row[f"{metric}_mean"])


def find_breakdown_mean(
    rows: list[dict[str, Any]], model: str, stratum: str, metric: str
) -> float:
    row = next(item for item in rows if item["model"] == model and item["stratum"] == stratum)
    return float(row[f"{metric}_mean"])


def fmt(mean: float, std: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def write_briefing(
    output_dir: Path,
    run_summary: list[dict[str, Any]],
    history_summary: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    seeds: tuple[int, ...],
    contract: dict[str, Any],
) -> None:
    index = {(row["model"], row["seed"]): row for row in run_rows}
    best_baseline_mae = min(
        find_mean(run_summary, "rmtpp", "best_val_qty_mae"),
        find_mean(run_summary, "thp", "best_val_qty_mae"),
    )
    titan_mae = find_mean(run_summary, "titantpp", "best_val_qty_mae")
    quantity_guardrail = titan_mae <= best_baseline_mae * 1.10
    overall_nll_gate = all(
        find_mean(run_summary, "titantpp", "best_val_nll")
        < find_mean(run_summary, baseline, "best_val_nll")
        for baseline in ("rmtpp", "thp")
    )
    overall_time_gate = all(
        find_mean(run_summary, "titantpp", "best_val_nll_time")
        < find_mean(run_summary, baseline, "best_val_nll_time")
        for baseline in ("rmtpp", "thp")
    )
    long_nll_gate = all(
        find_breakdown_mean(history_summary, "titantpp", "history_gt_128", "nll")
        < find_breakdown_mean(history_summary, baseline, "history_gt_128", "nll")
        for baseline in ("rmtpp", "thp")
    )
    long_time_gate = all(
        find_breakdown_mean(history_summary, "titantpp", "history_gt_128", "nll_time")
        < find_breakdown_mean(history_summary, baseline, "history_gt_128", "nll_time")
        for baseline in ("rmtpp", "thp")
    )
    overall_seed_gate = all(
        sum(
            index[("titantpp", seed)]["best_val_nll"]
            < index[(baseline, seed)]["best_val_nll"]
            for seed in seeds
        ) >= max(1, len(seeds) - 1)
        for baseline in ("rmtpp", "thp")
    )
    if overall_nll_gate and overall_time_gate and overall_seed_gate and quantity_guardrail:
        decision = "BROAD BACKBONE CLAIM QUALIFIED ON VALIDATION"
    elif long_nll_gate and long_time_gate and quantity_guardrail:
        decision = "CONDITIONAL LONG-HISTORY CLAIM QUALIFIED ON VALIDATION"
    else:
        decision = "BACKBONE CLAIM NOT QUALIFIED"

    lines = [
        "# Intermittent Log-Head Backbone Qualification",
        "",
        "## Contract",
        "",
        "- Same frozen 5,000-series split, log1p regression head, seeds, optimizer, and checkpoint rule across all backbones.",
        "- History ranges are fixed at <=64, 65-128, and >128 observed events.",
        "- Held-out test remains locked; this report uses validation only.",
        f"- Source revision: `{contract['source_revision']}`.",
        f"- Data SHA-256: `{contract['data_sha256']}`.",
        "",
        "## Overall Results",
        "",
        "| Model | NLL | Time NLL | Quantity MAE | Mark accuracy | Parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = next(item for item in run_summary if item["model"] == model)
        lines.append(
            f"| {LABELS[model]} | "
            f"{fmt(row['best_val_nll_mean'], row['best_val_nll_std'])} | "
            f"{fmt(row['best_val_nll_time_mean'], row['best_val_nll_time_std'])} | "
            f"{fmt(row['best_val_qty_mae_mean'], row['best_val_qty_mae_std'])} | "
            f"{fmt(row['mark_acc_mean'], row['mark_acc_std'])} | "
            f"{row['parameter_count_mean']:.0f} |"
        )
    lines.extend([
        "",
        "## History-Length NLL",
        "",
        "| Model | <=64 | 65-128 | >128 |",
        "|---|---:|---:|---:|",
    ])
    for model in MODELS:
        values = [
            find_breakdown_mean(history_summary, model, stratum, "nll")
            for stratum in ("history_le_64", "history_65_128", "history_gt_128")
        ]
        lines.append(f"| {LABELS[model]} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    lines.extend([
        "",
        "## Pre-Registered Gates",
        "",
        f"- Overall NLL best: **{overall_nll_gate}**",
        f"- Overall time NLL best: **{overall_time_gate}**",
        f"- Paired overall NLL consistency: **{overall_seed_gate}**",
        f"- Long-history NLL best: **{long_nll_gate}**",
        f"- Long-history time NLL best: **{long_time_gate}**",
        f"- Quantity MAE within 10% of the best baseline: **{quantity_guardrail}**",
        "",
        f"## Decision: **{decision}**",
        "",
        "This decision is a validation-stage go/no-go result. A qualified claim still requires one locked held-out test evaluation after the manuscript configuration is frozen.",
    ])
    (output_dir / "qualification_briefing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = tuple(int(token) for token in args.seeds.split(",") if token.strip())
    baseline_contract = read_json(args.baseline_root / "launch_contract.json")
    titan_contract = read_json(args.titan_root / "launch_contract.json")
    validate_contracts(
        args.baseline_root,
        args.titan_root,
        baseline_contract,
        titan_contract,
        seeds,
        args.expected_epochs,
    )
    run_rows = combine_rows(
        read_csv(args.baseline_root / "run_summaries.csv"),
        read_csv(args.titan_root / "run_summaries.csv"),
        seeds,
    )
    history_rows = combine_breakdowns(
        read_csv(args.baseline_root / "history_seed_metrics.csv"),
        read_csv(args.titan_root / "history_seed_metrics.csv"),
        seeds,
    )
    quantity_rows = combine_breakdowns(
        read_csv(args.baseline_root / "quantity_seed_metrics.csv"),
        read_csv(args.titan_root / "quantity_seed_metrics.csv"),
        seeds,
    )
    run_summary = summarize_runs(run_rows, seeds)
    history_summary = summarize_breakdowns(history_rows, seeds)
    quantity_summary = summarize_breakdowns(quantity_rows, seeds)
    deltas = paired_deltas(run_rows, history_rows, seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "combined_seed_metrics.csv", run_rows)
    write_csv(args.output_dir / "model_summary.csv", run_summary)
    write_csv(args.output_dir / "combined_history_seed_metrics.csv", history_rows)
    write_csv(args.output_dir / "history_summary.csv", history_summary)
    write_csv(args.output_dir / "combined_quantity_seed_metrics.csv", quantity_rows)
    write_csv(args.output_dir / "quantity_summary.csv", quantity_summary)
    write_csv(args.output_dir / "paired_titan_deltas.csv", deltas)
    write_briefing(
        args.output_dir,
        run_summary,
        history_summary,
        run_rows,
        seeds,
        baseline_contract,
    )
    print(f"Qualified Intermittent log-backbone results written to {args.output_dir}")


if __name__ == "__main__":
    main()
