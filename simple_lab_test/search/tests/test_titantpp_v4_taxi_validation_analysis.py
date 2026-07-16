from pathlib import Path

import polars as pl
import pytest

from simple_lab_test.search.analyze_titantpp_v4_taxi_validation import (
    confusion_summary,
    lower_improvement_pct,
    validate_held_out_lock,
)


def test_lower_improvement_pct_uses_lower_is_better_direction() -> None:
    assert lower_improvement_pct(candidate=9.0, control=10.0) == pytest.approx(10.0)


def test_confusion_summary_separates_adjacent_and_nonadjacent_errors() -> None:
    frame = pl.DataFrame(
        {
            "true_mark": [0, 0, 1, 2],
            "pred_mark": [0, 1, 3, 1],
            "count": [5, 2, 1, 2],
        }
    )

    summary = confusion_summary(frame)

    assert summary["total"] == 10
    assert summary["correct"] == 5
    assert summary["adjacent_error_count"] == 4
    assert summary["nonadjacent_error_count"] == 1
    assert summary["upward_error_count"] == 3
    assert summary["downward_error_count"] == 2
    assert summary["mark_mae"] == pytest.approx(0.6)


def test_confusion_summary_handles_a_perfect_classifier() -> None:
    frame = pl.DataFrame(
        {"true_mark": [0, 1], "pred_mark": [0, 1], "count": [3, 2]}
    )

    summary = confusion_summary(frame)

    assert summary["errors"] == 0
    assert summary["adjacent_share_of_errors"] == 0.0


def test_held_out_lock_rejects_test_artifacts(tmp_path: Path) -> None:
    (tmp_path / "test_summary.csv").write_text("metric,value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Held-out artifacts exist"):
        validate_held_out_lock(tmp_path)


def test_held_out_lock_accepts_validation_only_tree(tmp_path: Path) -> None:
    (tmp_path / "validation_summary.csv").write_text(
        "metric,value\n", encoding="utf-8"
    )

    assert validate_held_out_lock(tmp_path)["held_out_lock"] == "PASS"
