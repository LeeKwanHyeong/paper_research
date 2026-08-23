from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from benchmark_data.scripts.audit_train_quantity_history import (
    DatasetSpec,
    assert_train_only_path,
    summarize_dataset,
)


def spec() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="toy",
        label="Toy",
        role="test",
        quantity_provenance="native",
        train_path=Path("toy_train.parquet"),
    )


def train_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "oper_part_no": ["A", "A", "A", "B", "B"],
            "seq": [1, 2, 3, 1, 2],
            "demand_qty": [1.0, 2.0, 10.0, 2.0, 20.0],
            "chronological_split": ["train"] * 5,
        }
    )


def test_summarize_dataset_uses_train_events_and_next_event_histories():
    summary, bins = summarize_dataset(train_frame(), spec())

    assert summary["train_rows"] == 5
    assert summary["train_series"] == 2
    assert summary["train_next_event_targets"] == 3
    assert summary["sequence_events_p50"] == 3
    assert summary["history_length_max"] == 2
    assert sum(row["count"] for row in bins) == 5


def test_summarize_dataset_rejects_non_train_rows():
    frame = train_frame().with_columns(
        pl.when(pl.col("seq") == 3)
        .then(pl.lit("validation"))
        .otherwise(pl.col("chronological_split"))
        .alias("chronological_split")
    )

    with pytest.raises(ValueError, match="non-train rows"):
        summarize_dataset(frame, spec())


@pytest.mark.parametrize(
    "path",
    [
        Path("dataset_validation.parquet"),
        Path("dataset_test.parquet"),
        Path("dataset_with_split.parquet"),
    ],
)
def test_assert_train_only_path_rejects_other_artifacts(path: Path):
    with pytest.raises(ValueError, match="explicit train parquet"):
        assert_train_only_path(path)


def test_assert_train_only_path_accepts_train_artifact():
    assert_train_only_path(Path("dataset_train.parquet"))
