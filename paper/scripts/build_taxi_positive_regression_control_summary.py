#!/usr/bin/env python3
"""Combine the Taxi quantity-interface controls into paper-ready evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIANT_ORDER = (
    "uniform_categorical",
    "quantile_categorical",
    "direct_raw_mse",
    "minmax_sigmoid",
    "log_regression",
    "mark_residual",
)
VARIANT_LABELS = {
    "uniform_categorical": "Uniform categorical",
    "quantile_categorical": "Quantile categorical",
    "direct_raw_mse": "Raw MSE (diagnostic)",
    "minmax_sigmoid": "Min-max + sigmoid",
    "log_regression": "Log-scale regression",
    "mark_residual": "Exponent + residual",
}
STRATUM_ORDER = ("all", "le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reduction(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    args = parser.parse_args()

    rows = read_rows(args.existing)
    rows.extend(read_rows(args.control_root / "minmax_sigmoid" / "quantity_interface_summary.csv"))
    rows.extend(read_rows(args.control_root / "log_regression" / "quantity_interface_summary.csv"))
    lookup = {(row["variant"], row["stratum"]): row for row in rows}
    combined = [lookup[(variant, stratum)] for variant in VARIANT_ORDER for stratum in STRATUM_ORDER]
    write_rows(args.control_root / "combined_quantity_interface_summary.csv", combined)

    mark = {stratum: lookup[("mark_residual", stratum)] for stratum in STRATUM_ORDER}
    log = {stratum: lookup[("log_regression", stratum)] for stratum in STRATUM_ORDER}
    lines = [
        "# Taxi Positive-Regression Control Briefing",
        "",
        "## Qualification",
        "",
        "The two added controls completed all three seeds under the frozen Taxi split. "
        "Both use train-fitted transforms, produce nonnegative quantities by construction, "
        "and were evaluated on validation data only.",
        "",
        "The fair log-scale regression baseline has the lowest overall MAE. It reduces overall "
        f"MAE by {reduction(float(mark['all']['qty_mae_mean']), float(log['all']['qty_mae_mean'])):.1f}% "
        "relative to exponent + residual and is also better from p90 through p99. However, "
        "exponent + residual is better above p99. The Taxi control therefore does not support "
        "a general claim that exponent + residual is more accurate than a properly constrained "
        "regression baseline.",
        "",
        "Raw MSE remains diagnostic only because its unclipped output violated nonnegative support. "
        "It is retained in the table to explain the earlier observation, not as the fair final baseline.",
        "",
        "## Mean Validation Error Across Three Seeds",
        "",
        "| Interface | Overall MAE | Overall RMSE | p90-p95 MAE | p95-p99 MAE | >p99 MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANT_ORDER:
        values = {stratum: lookup[(variant, stratum)] for stratum in STRATUM_ORDER}
        lines.append(
            f"| {VARIANT_LABELS[variant]} | "
            f"{float(values['all']['qty_mae_mean']):.3f} | "
            f"{float(values['all']['qty_rmse_mean']):.3f} | "
            f"{float(values['p90_p95']['qty_mae_mean']):.3f} | "
            f"{float(values['p95_p99']['qty_mae_mean']):.3f} | "
            f"{float(values['gt_p99']['qty_mae_mean']):.3f} |"
        )
    lines.extend([
        "",
        "## Manuscript Decision",
        "",
        "Use log-scale regression as the primary fair quantity baseline. Do not claim that the "
        "exponent-residual interface solves long-tail quantity prediction on Taxi. A narrower "
        "statement is defensible: the representation guarantees valid support and changes the "
        "error trade-off, but its advantage is not uniform and disappears against log-scale regression.",
        "",
    ])
    (args.control_root / "qualification_briefing.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
