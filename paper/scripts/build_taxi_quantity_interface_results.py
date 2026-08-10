#!/usr/bin/env python3
"""Qualify the Taxi quantity-interface ablation and build its figure candidate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/titantpp-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEEDS = (42, 52, 62)
VARIANTS = (
    "uniform_categorical",
    "quantile_categorical",
    "direct_raw_mse",
    "mark_residual",
)
ALTERNATIVES = VARIANTS[:-1]
LABELS = {
    "uniform_categorical": "Uniform-bin categorical",
    "quantile_categorical": "Quantile-bin categorical",
    "direct_raw_mse": "Direct raw-scale MSE",
    "mark_residual": "Magnitude + residual",
}
COLORS = {
    "uniform_categorical": "#4477AA",
    "quantile_categorical": "#228833",
    "direct_raw_mse": "#CC6677",
    "mark_residual": "#AA3377",
}
TAIL_STRATA = ("p90_p95", "p95_p99", "gt_p99")
CUMULATIVE_GROUPS = {
    "p90_plus": TAIL_STRATA,
    "p95_plus": ("p95_p99", "gt_p99"),
}
METRICS = ("qty_mae", "qty_rmse", "qty_bias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=300)
    parser.add_argument("--expected-source-revision", default=None)
    parser.add_argument(
        "--expected-data-sha256",
        default="b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46",
    )
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


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def format_mean_std(mean: float, std: float, digits: int = 2) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def validate_contract(
    contract: dict[str, Any],
    run_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
    args: argparse.Namespace,
) -> None:
    if contract.get("status") != "complete":
        raise ValueError(f"Ablation is not complete: status={contract.get('status')!r}")
    if int(contract.get("epochs", -1)) != args.expected_epochs:
        raise ValueError("Epoch contract mismatch")
    if contract.get("evaluation_scope") != "validation_only":
        raise ValueError("Ablation is not validation-only")
    if bool(contract.get("held_out_test_evaluated", True)):
        raise ValueError("Held-out test flag is not locked")
    if contract.get("data_sha256") != args.expected_data_sha256:
        raise ValueError("Fixed-split data SHA-256 mismatch")
    if args.expected_source_revision is not None:
        if contract.get("source_revision") != args.expected_source_revision:
            raise ValueError("Source revision mismatch")
    if int(contract.get("max_train_batches") or 0) != 0:
        raise ValueError("Qualified result cannot use a train batch limit")
    if int(contract.get("max_val_batches") or 0) != 0:
        raise ValueError("Qualified result cannot use a validation batch limit")

    expected_pairs = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    observed_pairs = {(row["variant"], int(row["seed"])) for row in run_rows}
    if observed_pairs != expected_pairs or len(run_rows) != len(expected_pairs):
        raise ValueError("Run summary contract failed")
    if any(row["status"] != "success" for row in run_rows):
        raise ValueError("At least one run is not successful")
    if any(row["evaluation_scope"] != "validation_only" for row in run_rows):
        raise ValueError("At least one run is not validation-only")
    if any(row["held_out_test_evaluated"].lower() != "false" for row in run_rows):
        raise ValueError("At least one run has an unlocked test flag")
    for row in run_rows:
        if row["variant"] in ALTERNATIVES:
            if int(row["epochs"]) != args.expected_epochs:
                raise ValueError("New interface epoch contract failed")
            if row["source_revision"] != contract["source_revision"]:
                raise ValueError("New interface source revision mismatch")
        else:
            if int(row["epochs"]) != 300:
                raise ValueError("Mark-residual proposal must use its completed e300 checkpoint")
            if row["source_revision"] not in contract["proposal_source_revisions"]:
                raise ValueError("Mark-residual proposal source revision mismatch")
    unexpected_test_artifacts = [
        path for path in args.input_dir.rglob("*")
        if path.is_file() and "test" in path.name.lower()
    ]
    if unexpected_test_artifacts:
        raise ValueError(f"Unexpected test artifacts: {unexpected_test_artifacts}")

    expected_strata = {"all", "le_p50", "p50_p90", *TAIL_STRATA}
    for variant, seed in expected_pairs:
        observed = {
            row["stratum"]
            for row in seed_rows
            if row["variant"] == variant and int(row["seed"]) == seed
        }
        if observed != expected_strata:
            raise ValueError(f"Stratum contract failed for {variant}/seed_{seed}")
    if len(seed_rows) != len(expected_pairs) * len(expected_strata):
        raise ValueError("Seed metric row count mismatch")

    validation_count = int(contract["split_rows"]["validation"])
    reference_counts = {
        row["stratum"]: int(row["count"])
        for row in seed_rows
        if row["variant"] == "mark_residual" and int(row["seed"]) == SEEDS[0]
    }
    if reference_counts["all"] != validation_count:
        raise ValueError("Overall validation count mismatch")
    if sum(reference_counts[key] for key in expected_strata if key != "all") != validation_count:
        raise ValueError("Validation stratum counts do not sum to the fixed split")
    for row in seed_rows:
        if int(row["count"]) != reference_counts[row["stratum"]]:
            raise ValueError("Validation stratum counts differ across interfaces")


def typed_seed_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        output.append({
            **row,
            "seed": int(row["seed"]),
            "stratum_order": int(row["stratum_order"]),
            "count": int(row["count"]),
            "share": float(row["share"]),
            **{metric: float(row[metric]) for metric in METRICS},
        })
    return output


def cumulative_seed_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            indexed = {
                row["stratum"]: row
                for row in seed_rows
                if row["variant"] == variant and row["seed"] == seed
            }
            for order, (group_name, strata) in enumerate(CUMULATIVE_GROUPS.items(), start=10):
                count = sum(indexed[key]["count"] for key in strata)
                mae = sum(indexed[key]["qty_mae"] * indexed[key]["count"] for key in strata) / count
                mse = sum(
                    indexed[key]["qty_rmse"] ** 2 * indexed[key]["count"]
                    for key in strata
                ) / count
                bias = sum(indexed[key]["qty_bias"] * indexed[key]["count"] for key in strata) / count
                output.append({
                    "variant": variant,
                    "variant_label": LABELS[variant],
                    "seed": seed,
                    "stratum_order": order,
                    "stratum": group_name,
                    "stratum_label": "Above train p90" if group_name == "p90_plus" else "Above train p95",
                    "count": count,
                    "share": count / indexed["all"]["count"],
                    "qty_mae": mae,
                    "qty_rmse": float(np.sqrt(mse)),
                    "qty_bias": bias,
                })
    return output


def summarize(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    strata = sorted({
        (row["stratum_order"], row["stratum"], row["stratum_label"])
        for row in seed_rows
    })
    for variant in VARIANTS:
        for order, key, label in strata:
            group = [
                row for row in seed_rows
                if row["variant"] == variant and row["stratum"] == key
            ]
            if {row["seed"] for row in group} != set(SEEDS):
                raise ValueError(f"Seed contract failed for {variant}/{key}")
            record = {
                "variant": variant,
                "variant_label": LABELS[variant],
                "stratum_order": order,
                "stratum": key,
                "stratum_label": label,
                "count": group[0]["count"],
                "share": group[0]["share"],
                "n_seeds": len(group),
            }
            for metric in METRICS:
                mean, std = mean_std([row[metric] for row in group])
                record[f"{metric}_mean"] = mean
                record[f"{metric}_std"] = std
            output.append(record)
    return output


def paired_deltas(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["variant"], row["seed"], row["stratum"]): row
        for row in seed_rows
    }
    strata = sorted({
        (row["stratum_order"], row["stratum"], row["stratum_label"])
        for row in seed_rows
    })
    output: list[dict[str, Any]] = []
    for alternative in ALTERNATIVES:
        for order, key, label in strata:
            for metric in ("qty_mae", "qty_rmse"):
                proposal_values = [
                    indexed[("mark_residual", seed, key)][metric] for seed in SEEDS
                ]
                alternative_values = [
                    indexed[(alternative, seed, key)][metric] for seed in SEEDS
                ]
                deltas = [
                    proposal - baseline
                    for proposal, baseline in zip(proposal_values, alternative_values)
                ]
                alternative_mean = statistics.mean(alternative_values)
                output.append({
                    "alternative": alternative,
                    "alternative_label": LABELS[alternative],
                    "stratum_order": order,
                    "stratum": key,
                    "stratum_label": label,
                    "metric": metric,
                    "proposal_mean": statistics.mean(proposal_values),
                    "alternative_mean": alternative_mean,
                    "delta_mean": statistics.mean(deltas),
                    "delta_std": statistics.stdev(deltas),
                    "relative_delta_pct": 100.0 * statistics.mean(deltas) / alternative_mean,
                    "proposal_better_seeds": sum(delta < 0.0 for delta in deltas),
                    "seed_42_delta": deltas[0],
                    "seed_52_delta": deltas[1],
                    "seed_62_delta": deltas[2],
                })
    return output


def qualification(delta_rows: list[dict[str, Any]]) -> tuple[str, str]:
    required_strata = {*TAIL_STRATA, *CUMULATIVE_GROUPS}
    relevant = [
        row for row in delta_rows
        if row["metric"] == "qty_mae" and row["stratum"] in required_strata
    ]
    mean_consistent = all(row["relative_delta_pct"] < 0.0 for row in relevant)
    seed_consistent = all(row["proposal_better_seeds"] == len(SEEDS) for row in relevant)
    if mean_consistent and seed_consistent:
        return (
            "main_figure_candidate",
            "The magnitude-plus-residual interface has lower upper-tail MAE than all three alternatives in every seed.",
        )
    if mean_consistent:
        return (
            "auxiliary_sensitivity",
            "The magnitude-plus-residual interface improves mean upper-tail MAE, but the ranking is not seed-consistent.",
        )
    return (
        "diagnostic_only",
        "The interface ranking changes across upper-tail ranges; retain the model-level Taxi quantile chart as Figure 2.",
    )


def lookup_summary(
    rows: list[dict[str, Any]],
    variant: str,
    stratum: str,
) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["variant"] == variant and row["stratum"] == stratum
    )


def lookup_delta(
    rows: list[dict[str, Any]],
    alternative: str,
    stratum: str,
) -> dict[str, Any]:
    return next(
        row for row in rows
        if row["alternative"] == alternative
        and row["stratum"] == stratum
        and row["metric"] == "qty_mae"
    )


def write_briefing(
    path: Path,
    contract: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
    decision: str,
    rationale: str,
) -> None:
    boundaries = contract["quantile_contract"]["boundaries"]
    lines = [
        "# Taxi quantity-interface ablation",
        "",
        "## Contract",
        "",
        "- Encoder family: RMTPP with matched hidden size and training budget",
        "- Bin fitting and raw-target normalization: fixed training split only",
        "- Evaluation: fixed validation split only",
        "- Checkpoint selection: best validation event NLL",
        "- Held-out test evaluated: false",
        "- Seeds: 42, 52, 62",
        "- Train-derived boundaries: "
        + f"p50={boundaries[0]:g}, p90={boundaries[1]:g}, "
        + f"p95={boundaries[2]:g}, p99={boundaries[3]:g}",
        "",
        "## Decision",
        "",
        f"- Classification: `{decision}`",
        f"- {rationale}",
        "",
        "## Validation quantity MAE",
        "",
        "| Range | Interface | MAE | RMSE | Bias |",
        "|---|---|---:|---:|---:|",
    ]
    shown_strata = (*TAIL_STRATA, *CUMULATIVE_GROUPS)
    for stratum in shown_strata:
        for variant in VARIANTS:
            row = lookup_summary(summary_rows, variant, stratum)
            lines.append(
                f"| {row['stratum_label']} | {LABELS[variant]} | "
                f"{format_mean_std(row['qty_mae_mean'], row['qty_mae_std'])} | "
                f"{format_mean_std(row['qty_rmse_mean'], row['qty_rmse_std'])} | "
                f"{format_mean_std(row['qty_bias_mean'], row['qty_bias_std'])} |"
            )

    lines.extend([
        "",
        "## Magnitude-plus-residual paired MAE changes",
        "",
        "Negative values indicate lower error than the alternative.",
        "",
        "| Range | Alternative | Relative change | Better seeds |",
        "|---|---|---:|---:|",
    ])
    for stratum in shown_strata:
        for alternative in ALTERNATIVES:
            row = lookup_delta(delta_rows, alternative, stratum)
            lines.append(
                f"| {row['stratum_label']} | {LABELS[alternative]} | "
                f"{row['relative_delta_pct']:+.2f}% | "
                f"{row['proposal_better_seeds']}/3 |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_qualification(
    path: Path,
    decision: str,
    rationale: str,
    contract: dict[str, Any],
    delta_rows: list[dict[str, Any]],
) -> None:
    p90_rows = [lookup_delta(delta_rows, alternative, "p90_plus") for alternative in ALTERNATIVES]
    lines = [
        "# Taxi quantity-interface result qualification",
        "",
        "## Decision",
        "",
        f"- Classification: `{decision}`",
        f"- {rationale}",
        "",
        "## Evidence boundary",
        "",
        "This experiment holds the RMTPP encoder family, hidden size, optimizer, seed set, fixed split, and epoch budget constant. It changes the quantity interface, so it can support a representation-level comparison within RMTPP. It does not isolate TitanTPP's history encoder.",
        "",
        "## Above-p90 paired comparison",
        "",
        "| Alternative | Relative MAE change | Better seeds |",
        "|---|---:|---:|",
    ]
    for row in p90_rows:
        lines.append(
            f"| {row['alternative_label']} | {row['relative_delta_pct']:+.2f}% | "
            f"{row['proposal_better_seeds']}/3 |"
        )
    lines.extend([
        "",
        "## Figure rule",
        "",
    ])
    if decision == "main_figure_candidate":
        lines.append(
            "Use the interface figure as the mechanism-focused Figure 2 candidate. Its caption must state that the comparison uses RMTPP encoders and validation targets."
        )
    else:
        lines.append(
            "Keep `F2_taxi_validation_quantile_mae` as the main Figure 2. Report this interface ablation as an auxiliary table or sensitivity analysis without claiming universal superiority."
        )
    lines.extend([
        "",
        "## Integrity checks",
        "",
        f"- Source revision: `{contract['source_revision']}`",
        f"- Data SHA-256: `{contract['data_sha256']}`",
        "- Evaluation scope: validation only",
        "- Held-out test evaluated: false",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_chart_contract(
    path: Path,
    decision: str,
    rationale: str,
) -> None:
    recommendation = (
        "Replace the model-level Taxi quantile figure with this mechanism-focused figure."
        if decision == "main_figure_candidate"
        else "Keep the model-level Taxi quantile figure as Figure 2 and use this chart only as an auxiliary analysis."
    )
    lines = [
        "# Chart contract: Taxi quantity-interface tail analysis",
        "",
        "## Analytical question",
        "",
        "Does the magnitude-plus-residual representation reduce upper-tail quantity error relative to uniform bins, train-quantile bins, and direct raw-scale MSE when the RMTPP encoder family is held fixed?",
        "",
        "## Decision rule",
        "",
        "The chart becomes the main mechanism figure only if magnitude plus residual has lower mean MAE than every alternative in the p90-p95, p95-p99, above-p99, cumulative above-p90, and cumulative above-p95 ranges, with the same ranking in all three seeds.",
        "",
        "## Current qualification",
        "",
        f"- Classification: `{decision}`",
        f"- {rationale}",
        f"- {recommendation}",
        "",
        "## Data and visual form",
        "",
        "- Quantity boundaries are fitted on the fixed training split.",
        "- Metrics use fixed validation targets only; the held-out test is not evaluated.",
        "- Panel A compares absolute upper-tail MAE for four quantity interfaces.",
        "- Panel B reports the paired relative MAE change of magnitude plus residual against each alternative.",
        "- Error bars show sample standard deviation over seeds 42, 52, and 62.",
        "- PNG, PDF, and SVG are emitted for review and publication workflows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    delta_rows: list[dict[str, Any]],
) -> None:
    tail_labels = [
        lookup_summary(summary_rows, "mark_residual", stratum)["stratum_label"]
        for stratum in TAIL_STRATA
    ]
    comparison_strata = (*TAIL_STRATA, *CUMULATIVE_GROUPS)
    comparison_labels = [
        lookup_summary(summary_rows, "mark_residual", stratum)["stratum_label"]
        for stratum in comparison_strata
    ]
    fig, (ax_abs, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.7),
        gridspec_kw={"width_ratios": [1.08, 1.12]},
    )

    x_abs = np.arange(len(TAIL_STRATA), dtype=np.float64)
    width_abs = 0.19
    for index, variant in enumerate(VARIANTS):
        rows = [lookup_summary(summary_rows, variant, key) for key in TAIL_STRATA]
        means = [row["qty_mae_mean"] for row in rows]
        stds = [row["qty_mae_std"] for row in rows]
        offset = (index - 1.5) * width_abs
        ax_abs.bar(
            x_abs + offset,
            means,
            width=width_abs,
            yerr=stds,
            capsize=2.5,
            color=COLORS[variant],
            label=LABELS[variant],
        )

    x_delta = np.arange(len(comparison_strata), dtype=np.float64)
    width_delta = 0.24
    for index, alternative in enumerate(ALTERNATIVES):
        rows = [lookup_delta(delta_rows, alternative, key) for key in comparison_strata]
        values = [row["relative_delta_pct"] for row in rows]
        offset = (index - 1) * width_delta
        ax_delta.bar(
            x_delta + offset,
            values,
            width=width_delta,
            color=COLORS[alternative],
            label=f"vs. {LABELS[alternative]}",
        )

    ax_abs.set_xticks(x_abs, tail_labels)
    ax_delta.set_xticks(x_delta, comparison_labels)
    ax_abs.set_ylabel("Validation quantity MAE")
    ax_abs.set_title("(a) Upper-tail error by quantity interface")
    ax_delta.set_ylabel("Magnitude + residual MAE change (%)")
    ax_delta.set_title("(b) Relative change against alternatives")
    ax_delta.axhline(0.0, color="#333333", linewidth=1.0)
    ax_delta.text(
        0.01,
        0.98,
        "negative is better",
        transform=ax_delta.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4B5563",
    )
    for axis in (ax_abs, ax_delta):
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelsize=8.2)
    ax_abs.legend(frameon=False, fontsize=7.9, loc="upper left")
    ax_delta.legend(frameon=False, fontsize=7.8, loc="best")
    fig.suptitle(
        "Taxi upper-tail quantity error under controlled RMTPP interfaces",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Ranges use training-split quantiles. Bars show mean +/- sample standard deviation over seeds 42, 52, and 62; validation only.",
        ha="center",
        fontsize=8.4,
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.94))
    base = output_dir / "F2_taxi_quantity_interface_tail"
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(
        base.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg_path = base.with_suffix(".svg")
    svg_path.write_text(svg_path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads(
        (args.input_dir / "launch_contract.json").read_text(encoding="utf-8")
    )
    run_rows = read_csv(args.input_dir / "run_summaries.csv")
    raw_seed_rows = read_csv(args.input_dir / "quantity_interface_seed_metrics.csv")
    validate_contract(contract, run_rows, raw_seed_rows, args)
    seed_rows = typed_seed_rows(raw_seed_rows)
    cumulative_rows = cumulative_seed_metrics(seed_rows)
    combined_rows = [*seed_rows, *cumulative_rows]
    summary_rows = summarize(combined_rows)
    delta_rows = paired_deltas(combined_rows)
    decision, rationale = qualification(delta_rows)

    write_csv(args.output_dir / "quantity_interface_cumulative_seed_metrics.csv", cumulative_rows)
    write_csv(args.output_dir / "quantity_interface_qualified_summary.csv", summary_rows)
    write_csv(args.output_dir / "quantity_interface_paired_deltas.csv", delta_rows)
    write_briefing(
        args.output_dir / "quantity_interface_briefing.md",
        contract,
        summary_rows,
        delta_rows,
        decision,
        rationale,
    )
    write_qualification(
        args.output_dir / "result_qualification.md",
        decision,
        rationale,
        contract,
        delta_rows,
    )
    write_chart_contract(
        args.output_dir / "chart_contract.md",
        decision,
        rationale,
    )
    chart_contract = {
        "schema_version": 1,
        "decision": decision,
        "rationale": rationale,
        "figure": "F2_taxi_quantity_interface_tail",
        "quantile_source_split": "train",
        "evaluation_split": "validation",
        "held_out_test_evaluated": False,
        "seeds": list(SEEDS),
        "source_revision": contract["source_revision"],
        "data_sha256": contract["data_sha256"],
    }
    (args.output_dir / "chart_contract.json").write_text(
        json.dumps(chart_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plot_results(args.output_dir, summary_rows, delta_rows)
    print(f"[complete] decision={decision} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
