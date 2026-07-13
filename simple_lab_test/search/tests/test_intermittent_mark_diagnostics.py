from __future__ import annotations

import math

import numpy as np
import polars as pl

from simple_lab_test.search.analyze_intermittent_mark_diagnostics import (
    accuracy_contribution_rows,
    build_distribution_tables,
    confusion_metrics,
    load_target_events,
)


def test_load_target_events_excludes_first_event_per_series(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "oper_part_no": ["A", "A", "A", "B", "B"],
            "seq": [1, 2, 3, 1, 2],
            "mark": [0, 1, 2, 1, 0],
            "chronological_split": ["train", "train", "validation", "train", "test"],
        }
    )
    path = tmp_path / "split.parquet"
    frame.write_parquet(path)

    targets = load_target_events(path)

    assert targets.height == 3
    assert targets.group_by("chronological_split").len().sort("chronological_split").to_dicts() == [
        {"chronological_split": "test", "len": 1},
        {"chronological_split": "train", "len": 1},
        {"chronological_split": "validation", "len": 1},
    ]


def test_confusion_metrics_preserve_ordinal_error_direction() -> None:
    matrix = np.asarray([[8, 2], [1, 9]], dtype=np.int64)

    metrics = confusion_metrics(matrix)

    assert math.isclose(float(metrics["accuracy"]), 0.85)
    assert math.isclose(float(metrics["adjacent_accuracy"]), 1.0)
    assert math.isclose(float(metrics["adjacent_share_of_errors"]), 1.0)
    assert math.isclose(float(metrics["mark_mae"]), 0.15)
    assert math.isclose(float(metrics["signed_mark_error"]), 0.05)


def test_accuracy_contributions_reconcile_to_accuracy_delta() -> None:
    baseline = np.asarray([[8, 2], [1, 9]], dtype=np.int64)
    variant = np.asarray([[9, 1], [3, 7]], dtype=np.int64)
    matrices = {
        "validation": {"V2": baseline, "V3c": variant},
        "test": {"V2": baseline, "V3c": variant},
    }

    rows = accuracy_contribution_rows(matrices)
    test_rows = [row for row in rows if row["eval_split"] == "test"]

    assert [row["delta_correct_count"] for row in test_rows] == [1, -2]
    assert math.isclose(sum(float(row["delta_accuracy_pp"]) for row in test_rows), -5.0)


def test_distribution_summary_reconciles_split_totals() -> None:
    targets = pl.DataFrame(
        {
            "chronological_split": ["train", "train", "validation", "test"],
            "mark": [0, 1, 0, 1],
        }
    )

    distribution, summary = build_distribution_tables(targets, num_marks=2)

    assert distribution["count"].sum() == 4
    assert summary.select(pl.col("target_count").sum()).item() == 4
