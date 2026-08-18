"""CSV persistence and cross-seed summaries for count-aware experiments."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from paper.scripts.count_aware_tpp_backbone.constants import BACKBONE_LABELS


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows using the first row as the stable output schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_breakdowns(
    rows: list[dict[str, Any]],
    *,
    backbones: tuple[str, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Aggregate per-stratum metrics after enforcing the seed contract."""
    output = []
    strata = sorted(
        {
            (int(row["stratum_order"]), row["stratum"], row["stratum_label"])
            for row in rows
        }
    )
    for variant in variants:
        for backbone in backbones:
            for order, key, label in strata:
                group = [
                    row
                    for row in rows
                    if row["variant"] == variant
                    and row["backbone"] == backbone
                    and row["stratum"] == key
                ]
                if {int(row["seed"]) for row in group} != set(seeds):
                    raise ValueError(
                        f"Seed contract failed for {variant}/{backbone}/{key}"
                    )
                record = {
                    "backbone": backbone,
                    "backbone_label": BACKBONE_LABELS[backbone],
                    "variant": variant,
                    "stratum_order": order,
                    "stratum": key,
                    "stratum_label": label,
                    "count": int(group[0]["count"]),
                    "share": float(group[0]["share"]),
                    "n_seeds": len(group),
                }
                for metric in (
                    "joint_objective",
                    "time_nll",
                    "quantity_train_loss",
                    "log_qty_mse",
                    "quantity_distribution_nll",
                    "quantity_location_huber",
                    "tail_aux_loss",
                    "quantity_scale_mean",
                    "qty_mae",
                    "qty_rmse",
                    "qty_bias",
                ):
                    values = [float(row[metric]) for row in group]
                    record[f"{metric}_mean"] = statistics.mean(values)
                    record[f"{metric}_std"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )
                output.append(record)
    return output


__all__ = ["summarize_breakdowns", "write_csv"]
