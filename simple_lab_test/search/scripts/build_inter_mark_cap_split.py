#!/usr/bin/env python3
"""Build an Intermittent fixed split with a capped log2 magnitude mark."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="marked_target_cap5")
    parser.add_argument("--max-order", type=int, default=5)
    parser.add_argument("--scale-base", type=float, default=2.0)
    return parser.parse_args()


def rewrite_marks(frame: pl.DataFrame, *, max_order: int) -> pl.DataFrame:
    required = {
        "oper_part_no",
        "demand_dt",
        "seq",
        "delta_t",
        "demand_qty",
        "chronological_split",
        "log_qty",
        "log10_qty",
        "raw_order",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source split is missing required columns: {missing}")

    return (
        frame.with_columns(
            pl.col("raw_order").clip(None, int(max_order)).cast(pl.Int32).alias("mark")
        )
        .with_columns(
            (pl.col("log_qty") - pl.col("mark"))
            .cast(pl.Float64)
            .alias("scale_residual")
        )
        .with_columns(pl.col("log_qty").alias("z"))
        .select(
            [
                "oper_part_no",
                "demand_dt",
                "seq",
                "delta_t",
                "demand_qty",
                "chronological_split",
                "log_qty",
                "log10_qty",
                "raw_order",
                "mark",
                "scale_residual",
                "z",
            ]
        )
        .sort(["oper_part_no", "seq"])
    )


def split_summary(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.group_by("chronological_split")
        .agg(
            [
                pl.len().alias("rows"),
                pl.col("oper_part_no").n_unique().alias("series"),
                pl.col("demand_qty").median().alias("qty_median"),
                pl.col("demand_qty").quantile(0.95).alias("qty_p95"),
                pl.col("demand_qty").max().alias("qty_max"),
            ]
        )
        .sort("chronological_split")
        .to_dicts()
    )


def mark_summary(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return (
        frame.group_by(["chronological_split", "mark"])
        .len()
        .sort(["chronological_split", "mark"])
        .to_dicts()
    )


def write_split(frame: pl.DataFrame, output_dir: Path, prefix: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "with_split": output_dir / f"{prefix}_with_split.parquet",
        "train": output_dir / f"{prefix}_train.parquet",
        "validation": output_dir / f"{prefix}_validation.parquet",
        "test": output_dir / f"{prefix}_test.parquet",
        "manifest": output_dir / f"{prefix}_split_manifest.json",
    }
    frame.write_parquet(paths["with_split"])
    for split in ("train", "validation", "test"):
        frame.filter(pl.col("chronological_split") == split).write_parquet(paths[split])
    return {key: str(path) for key, path in paths.items()}


def main() -> None:
    args = parse_args()
    source = args.source_dir / "marked_target_with_split.parquet"
    source_manifest = args.source_dir / "marked_target_split_manifest.json"
    frame = rewrite_marks(pl.read_parquet(source), max_order=args.max_order)
    paths = write_split(frame, args.output_dir, args.prefix)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "source_artifacts": {
            "with_split": str(source),
            "manifest": str(source_manifest),
        },
        "config": {
            "dataset_name": "head_office",
            "mark_design": "log2_magnitude_capped_tail",
            "scale_base": float(args.scale_base),
            "max_order": int(args.max_order),
            "tail_definition": f"demand_qty >= {float(args.scale_base) ** int(args.max_order):.0f}",
        },
        "magnitude_rule": {
            "scale_base": float(args.scale_base),
            "min_order": 0,
            "max_order": int(args.max_order),
            "fitted_on": "train",
            "tail_policy": "clip raw_order above max_order into final mark",
        },
        "artifacts": paths,
        "summary": {
            "split_counts": split_summary(frame),
            "mark_counts": mark_summary(frame),
            "num_marks": int(frame.select(pl.col("mark").max()).item()) + 1,
            "scale_residual_min": float(frame.select(pl.col("scale_residual").min()).item()),
            "scale_residual_max": float(frame.select(pl.col("scale_residual").max()).item()),
        },
    }
    Path(paths["manifest"]).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "success", "paths": paths, "summary": manifest["summary"]}, indent=2))


if __name__ == "__main__":
    main()
