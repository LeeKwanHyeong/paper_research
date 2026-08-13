from __future__ import annotations

import json
from pathlib import Path
import sys

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.scripts.run_taxi_quantity_interface_ablation import make_loader


DATASETS = {
    "online_retail_ii": {
        "data": PROJECT_ROOT / "benchmark_data/data/main/online_retail_ii/online_retail_ii_with_split.parquet",
        "manifest": PROJECT_ROOT / "benchmark_data/data/main/online_retail_ii/online_retail_ii_split_manifest.json",
        "lookback": 24 * 365,
        "max_len": 256,
    },
    "yellow_trip_hourly": {
        "data": PROJECT_ROOT / "benchmark_data/data/auxiliary/yellow_trip_hourly/yellow_trip_hourly_with_split.parquet",
        "manifest": PROJECT_ROOT / "benchmark_data/data/auxiliary/yellow_trip_hourly/yellow_trip_hourly_split_manifest.json",
        "lookback": 24 * 30,
        "max_len": 256,
    },
    "instacart": {
        "data": PROJECT_ROOT / "benchmark_data/data/auxiliary/instacart/instacart_marked_target_with_split.parquet",
        "manifest": PROJECT_ROOT / "benchmark_data/data/auxiliary/instacart/instacart_marked_target_split_manifest.json",
        "lookback": 365,
        "max_len": 256,
    },
}


def source_target_counts(manifest: dict[str, object]) -> dict[str, int] | None:
    counts = manifest.get("next_event_target_counts")
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items()}
    return None


def validation_series_sample(frame: pl.DataFrame, limit: int = 64) -> pl.DataFrame:
    entity_ids = (
        frame.filter(pl.col("chronological_split") == "validation")
        .select("oper_part_no")
        .unique(maintain_order=True)
        .head(limit)["oper_part_no"]
    )
    return frame.filter(pl.col("oper_part_no").is_in(entity_ids.to_list()))


def main() -> None:
    for dataset_id, spec in DATASETS.items():
        manifest = json.loads(Path(spec["manifest"]).read_text())
        frame = pl.read_parquet(Path(spec["data"])).sort(["oper_part_no", "seq"])
        sampled = validation_series_sample(frame)
        sampled = sampled.with_columns(
            [
                pl.lit(0, dtype=pl.Int32).alias("mark"),
                pl.col("demand_qty").cast(pl.Float64).alias("scale_residual"),
            ]
        )
        loader = make_loader(
            sampled,
            target_split="validation",
            batch_size=64,
            lookback_weeks=int(spec["lookback"]),
            max_seq_len=int(spec["max_len"]),
            shuffle=False,
            generator=None,
        )
        if not len(loader.dataset):
            raise AssertionError(f"{dataset_id}: no validation targets")
        marks, dts, mask, _, quantities = next(iter(loader))
        if quantities is None or len({marks.shape, dts.shape, mask.shape, quantities.shape}) != 1:
            raise AssertionError(f"{dataset_id}: malformed batch")
        if int(mask.sum(dim=1).min()) < 2 or float(quantities[mask].min()) <= 0:
            raise AssertionError(f"{dataset_id}: invalid history or quantity")

        targets = source_target_counts(manifest)
        target_note = f", frozen_validation_targets={targets['validation']}" if targets else ""
        print(
            f"{dataset_id}: sampled_targets={len(loader.dataset)}, "
            f"batch_shape={tuple(marks.shape)}{target_note}"
        )
    print("benchmark model inputs: PASS")


if __name__ == "__main__":
    main()
