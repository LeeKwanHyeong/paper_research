from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import polars as pl

from common import ROOT, artifact_record, write_json


DATASETS = {
    "yellow_trip_hourly": {
        "root": ROOT / "data" / "auxiliary" / "yellow_trip_hourly",
        "prefix": "yellow_trip_hourly",
        "role": "auxiliary",
        "quantity_provenance": "derived grid-hour pickup count",
    },
    "instacart": {
        "root": ROOT / "data" / "auxiliary" / "instacart",
        "prefix": "instacart_marked_target",
        "role": "optional_auxiliary",
        "quantity_provenance": "derived order basket size",
    },
}
REQUIRED_COLUMNS = {
    "oper_part_no",
    "demand_dt",
    "seq",
    "delta_t",
    "demand_qty",
    "chronological_split",
}
SPLITS = ("train", "validation", "test")


def audit_dataset(dataset_id: str, spec: dict[str, object]) -> dict[str, object]:
    root = Path(spec["root"])
    prefix = str(spec["prefix"])
    with_split = root / f"{prefix}_with_split.parquet"
    split_manifest = root / f"{prefix}_split_manifest.json"
    frame = pl.read_parquet(with_split).sort(["oper_part_no", "seq"])
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_id}: missing columns {missing}")
    if frame.filter(pl.col("demand_qty") <= 0).height:
        raise ValueError(f"{dataset_id}: quantity must be positive")
    duplicate_count = (
        frame.group_by(["oper_part_no", "seq"])
        .len()
        .filter(pl.col("len") > 1)
        .height
    )
    if duplicate_count:
        raise ValueError(f"{dataset_id}: duplicate entity/seq keys={duplicate_count}")

    event_rank = pl.int_range(0, pl.len()).over("oper_part_no")
    split_counts = {}
    target_counts = {}
    series_counts = {}
    artifacts = {
        "with_split": artifact_record(with_split),
        "split_manifest": artifact_record(split_manifest),
    }
    for split in SPLITS:
        split_frame = frame.filter(pl.col("chronological_split") == split)
        split_counts[split] = split_frame.height
        series_counts[split] = split_frame["oper_part_no"].n_unique()
        target_counts[split] = frame.filter(
            (pl.col("chronological_split") == split) & (event_rank > 0)
        ).height
        split_path = root / f"{prefix}_{split}.parquet"
        if pl.read_parquet(split_path).height != split_counts[split]:
            raise ValueError(f"{dataset_id}: {split} artifact count mismatch")
        artifacts[split] = artifact_record(split_path)

    source_manifest = json.loads(split_manifest.read_text())
    source_counts = {
        str(row["chronological_split"]): int(row["rows"])
        for row in source_manifest["summary"]["split_counts"]
    }
    if source_counts != split_counts:
        raise ValueError(f"{dataset_id}: source manifest count mismatch")

    return {
        "dataset_id": dataset_id,
        "role": spec["role"],
        "status": "frozen_experiment_ready",
        "quantity_provenance": spec["quantity_provenance"],
        "rows": frame.height,
        "series": frame["oper_part_no"].n_unique(),
        "split_counts": split_counts,
        "next_event_target_counts": target_counts,
        "series_counts": series_counts,
        "schema": {name: str(dtype) for name, dtype in frame.schema.items()},
        "checks": {
            "positive_quantity": True,
            "unique_entity_sequence_key": True,
            "split_artifacts_match": True,
            "held_out_test_evaluated": False,
        },
        "artifacts": artifacts,
    }


def main() -> None:
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [
            audit_dataset(dataset_id, spec) for dataset_id, spec in DATASETS.items()
        ],
    }
    output = ROOT / "manifests" / "auxiliary_frozen_v1.json"
    write_json(output, payload)
    print(output)


if __name__ == "__main__":
    main()
