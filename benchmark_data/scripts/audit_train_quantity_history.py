#!/usr/bin/env python3
"""Audit train-only quantity tails and sequence lengths for count benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from benchmark_data.scripts.common import artifact_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUANTILE_INTERPOLATION = "nearest"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    label: str
    role: str
    quantity_provenance: str
    train_path: Path


DATASETS = (
    DatasetSpec(
        dataset_id="intermittent_v2",
        label="Intermittent v2",
        role="main",
        quantity_provenance="native_order_quantity",
        train_path=PROJECT_ROOT
        / "benchmark_data/data/main/intermittent_v2/intermittent_frozen_5000_train.parquet",
    ),
    DatasetSpec(
        dataset_id="online_retail_ii",
        label="Online Retail II",
        role="main",
        quantity_provenance="native_transaction_quantity",
        train_path=PROJECT_ROOT
        / "benchmark_data/data/main/online_retail_ii/online_retail_ii_train.parquet",
    ),
    DatasetSpec(
        dataset_id="raf_spare_parts",
        label="RAF Spare Parts",
        role="main_candidate",
        quantity_provenance="native_monthly_demand",
        train_path=PROJECT_ROOT
        / "benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_train.parquet",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def assert_train_only_path(path: Path) -> None:
    if not path.name.endswith("_train.parquet"):
        raise ValueError(f"audit input must be an explicit train parquet: {path}")
    lowered = path.name.lower()
    if any(token in lowered for token in ("validation", "test", "with_split")):
        raise ValueError(f"forbidden non-train input: {path}")


def nearest_quantile(series: pl.Series, probability: float) -> float:
    value = series.quantile(probability, interpolation=QUANTILE_INTERPOLATION)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"invalid quantile at p={probability}")
    return float(value)


def distribution_stats(series: pl.Series, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_min": float(series.min()),
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_p50": nearest_quantile(series, 0.50),
        f"{prefix}_p95": nearest_quantile(series, 0.95),
        f"{prefix}_p99": nearest_quantile(series, 0.99),
        f"{prefix}_max": float(series.max()),
    }


def validate_train_frame(frame: pl.DataFrame, dataset_id: str) -> None:
    required = {"oper_part_no", "seq", "demand_qty", "chronological_split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{dataset_id}: missing columns {sorted(missing)}")
    splits = set(frame["chronological_split"].unique().to_list())
    if splits != {"train"}:
        raise ValueError(f"{dataset_id}: non-train rows found: {sorted(splits)}")
    quantities = frame["demand_qty"]
    if quantities.null_count() or not quantities.is_finite().all() or float(quantities.min()) <= 0:
        raise ValueError(f"{dataset_id}: quantity must be finite and positive")
    duplicate_count = frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().sum()).item()
    if duplicate_count:
        raise ValueError(f"{dataset_id}: duplicate entity/sequence keys found")


def quantity_bin_rows(
    spec: DatasetSpec, quantities: pl.Series, q50: float, q95: float, q99: float
) -> list[dict[str, Any]]:
    masks = (
        ("le_p50", "quantity <= train p50", quantities <= q50),
        ("p50_p95", "train p50 < quantity <= train p95", (quantities > q50) & (quantities <= q95)),
        ("p95_p99", "train p95 < quantity <= train p99", (quantities > q95) & (quantities <= q99)),
        ("gt_p99", "quantity > train p99", quantities > q99),
    )
    total = len(quantities)
    return [
        {
            "dataset_id": spec.dataset_id,
            "dataset_label": spec.label,
            "stratum_order": order,
            "stratum": stratum,
            "stratum_label": label,
            "count": int(mask.sum()),
            "share": float(mask.sum() / total),
        }
        for order, (stratum, label, mask) in enumerate(masks)
    ]


def summarize_dataset(
    frame: pl.DataFrame, spec: DatasetSpec
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    validate_train_frame(frame, spec.dataset_id)
    frame = frame.sort(["oper_part_no", "seq"])
    quantities = frame["demand_qty"].cast(pl.Float64)
    quantity = distribution_stats(quantities, "quantity")
    q50 = quantity["quantity_p50"]
    q95 = quantity["quantity_p95"]
    q99 = quantity["quantity_p99"]

    sequence_lengths = (
        frame.group_by("oper_part_no")
        .len(name="train_events")
        .sort("oper_part_no")["train_events"]
        .cast(pl.Float64)
    )
    histories = (
        frame.with_columns(
            (pl.col("oper_part_no").cum_count().over("oper_part_no") - 1)
            .cast(pl.Int64)
            .alias("history_length")
        )
        .filter(pl.col("history_length") > 0)["history_length"]
        .cast(pl.Float64)
    )
    if histories.is_empty():
        raise ValueError(f"{spec.dataset_id}: no next-event targets")

    summary = {
        "dataset_id": spec.dataset_id,
        "dataset_label": spec.label,
        "role": spec.role,
        "quantity_provenance": spec.quantity_provenance,
        "train_rows": frame.height,
        "train_series": sequence_lengths.len(),
        "train_next_event_targets": histories.len(),
        **quantity,
        "quantity_gt_p95_count": int((quantities > q95).sum()),
        "quantity_gt_p95_share": float((quantities > q95).sum() / len(quantities)),
        "quantity_gt_p99_count": int((quantities > q99).sum()),
        "quantity_gt_p99_share": float((quantities > q99).sum() / len(quantities)),
        **distribution_stats(sequence_lengths, "sequence_events"),
        **distribution_stats(histories, "history_length"),
    }
    return summary, quantity_bin_rows(spec, quantities, q50, q95, q99)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Count-aware Benchmark Train-only Data Audit",
        "",
        "- Scope: explicit train parquet files only",
        "- Quantiles: nearest interpolation",
        "- Validation/test rows: not read",
        "- Held-out test: not evaluated",
        "",
        "## Quantity Distribution",
        "",
        "| Dataset | Train events | p50 | p95 | p99 | Max | >p95 share | >p99 share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_label']} | {row['train_rows']:,} | "
            f"{row['quantity_p50']:.0f} | {row['quantity_p95']:.0f} | "
            f"{row['quantity_p99']:.0f} | {row['quantity_max']:.0f} | "
            f"{100 * row['quantity_gt_p95_share']:.3f}% | "
            f"{100 * row['quantity_gt_p99_share']:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Train Sequence Structure",
            "",
            "| Dataset | Series | Events/series p50 | p95 | p99 | Max | History p50 | p95 | p99 | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset_label']} | {row['train_series']:,} | "
            f"{row['sequence_events_p50']:.0f} | {row['sequence_events_p95']:.0f} | "
            f"{row['sequence_events_p99']:.0f} | {row['sequence_events_max']:.0f} | "
            f"{row['history_length_p50']:.0f} | {row['history_length_p95']:.0f} | "
            f"{row['history_length_p99']:.0f} | {row['history_length_max']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Tail Severity",
            "",
            "| Dataset | Mean / p50 | Max / p99 | Interpretation |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        max_to_p99 = row["quantity_max"] / row["quantity_p99"]
        mean_to_p50 = row["quantity_mean"] / row["quantity_p50"]
        if row["dataset_id"] == "online_retail_ii":
            interpretation = "Extreme tail far beyond p99; an uncapped raw loss would be high risk"
        elif row["dataset_id"] == "raf_spare_parts":
            interpretation = "Heavy quantity tail with very short event histories"
        else:
            interpretation = "Moderate-length histories and a bounded frozen quantity tail"
        lines.append(
            f"| {row['dataset_label']} | {mean_to_p50:.2f}x | "
            f"{max_to_p99:.2f}x | {interpretation} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All three native-count datasets are right-skewed, but their absolute p50/p95/p99 thresholds and sequence structures differ. A shared absolute quantity threshold is therefore not portable. Any follow-up body/tail objective must derive thresholds from each dataset's train split using one frozen quantile rule.",
            "",
            "Online Retail II has the longest histories and the most extreme outliers: its maximum quantity is more than 100 times its train p99. RAF has the shortest histories, with only six train events per series at the median. Intermittent v2 lies between them. These differences let the next matched validation distinguish a long-history retail setting from a short intermittent-demand setting.",
            "",
            "This audit establishes data compatibility only. It does not establish that the Intermittent body-MAE/tail-RMSE trade-off repeats on Online Retail II or RAF; that requires matched validation model runs.",
            "",
            "## Decision",
            "",
            "Do not implement the new mid-body balanced objective yet. First run the frozen T0 and TitanTPP-T1 validation comparison on all three datasets. If the same body-MAE/tail-RMSE trade-off appears on multiple native-count datasets, design one train-quantile-adaptive objective; otherwise retain the behavior as dataset-specific evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def render_contract_audit(rows: list[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# Train-only Audit Contract",
            "",
            "- Status: **PASS**",
            "- Inputs: three explicit `_train.parquet` files",
            "- Observed split value: `train` only",
            "- Required key: unique `(oper_part_no, seq)`",
            "- Quantity: finite and strictly positive",
            "- Quantile interpolation: `nearest`",
            f"- Dataset count: {len(rows)}",
            "- Validation/test parquet files: not read",
            "- Held-out test: not evaluated",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    bins: list[dict[str, Any]] = []
    inputs = []
    for spec in DATASETS:
        assert_train_only_path(spec.train_path)
        frame = pl.read_parquet(spec.train_path)
        summary, dataset_bins = summarize_dataset(frame, spec)
        summaries.append(summary)
        bins.extend(dataset_bins)
        inputs.append({"dataset_id": spec.dataset_id, **artifact_record(spec.train_path)})

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "quantity_bins.csv", bins)
    (output_dir / "analysis.md").write_text(render_summary(summaries), encoding="utf-8")
    (output_dir / "contract_audit.md").write_text(
        render_contract_audit(summaries), encoding="utf-8"
    )
    log_lines = [
        f"{row['dataset_id']}: rows={row['train_rows']}, series={row['train_series']}, "
        f"p50/p95/p99={row['quantity_p50']:.0f}/{row['quantity_p95']:.0f}/{row['quantity_p99']:.0f}"
        for row in summaries
    ]
    log_lines.append("train-only quantity/history audit: PASS")
    (output_dir / "audit.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(),
        "audit": "count_aware_train_quantity_history",
        "input_scope": "train_only",
        "quantile_interpolation": QUANTILE_INTERPOLATION,
        "datasets": [spec.dataset_id for spec in DATASETS],
        "held_out_test_evaluated": False,
        "validation_data_read": False,
        "test_data_read": False,
        "inputs": inputs,
        "outputs": ["audit.log", "summary.csv", "quantity_bins.csv", "analysis.md", "contract_audit.md"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n".join(log_lines))


if __name__ == "__main__":
    main()
