from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_predictions as analysis  # noqa: E402


def make_frame(
    *,
    series_ids: list[str],
    targets: list[float] | None = None,
    candidate: list[float] | None = None,
    reference: list[float] | None = None,
) -> pl.DataFrame:
    n = len(series_ids)
    targets = targets or [10.0 + index for index in range(n)]
    candidate = candidate or [value + ((index % 3) - 1) for index, value in enumerate(targets)]
    reference = reference or [value + 1.0 for value in targets]
    within_series_position: dict[str, int] = {}
    positions = []
    for series_id in series_ids:
        positions.append(within_series_position.get(series_id, 0))
        within_series_position[series_id] = positions[-1] + 1
    history_values = [1, 2, 4, 8, 16, 32]
    recent_values = [-0.5, 0.0, 0.5]
    return pl.DataFrame(
        {
            "series_id": series_ids,
            "series_index": list(range(n)),
            "context_end": [position + 1 for position in positions],
            "target_position": positions,
            "target_seq": positions,
            "history_length": [history_values[index % len(history_values)] for index in range(n)],
            "true_qty": targets,
            "target_dt": [1.0 + index % 30 for index in range(n)],
            "last_qty": targets,
            "history_mean_qty": targets,
            "history_mean_log_qty": np.log1p(np.asarray(targets, dtype=np.float64)),
            "recent3_minus_history_mean_log_qty": [
                recent_values[index % len(recent_values)] for index in range(n)
            ],
            "last_minus_history_mean_log_qty": [0.0] * n,
            "prediction_b": reference,
            "prediction_candidate": candidate,
            "prediction_rmtpp": reference,
            "prediction_thp": [value + 2.0 for value in targets],
        }
    )


def mixed_fold_series(count: int = 12) -> list[str]:
    values = [f"series-{index}" for index in range(count)]
    assert set(analysis.assign_series_folds(values).tolist()) == {0, 1}
    return values


def test_known_bias_centered_mse_decomposition() -> None:
    target = np.full(4, 10.0, dtype=np.float64)
    candidate = np.asarray([11.0, 13.0, 9.0, 7.0], dtype=np.float64)
    reference = np.full(4, 11.0, dtype=np.float64)

    result = analysis.comparison_metrics(target, candidate, reference)

    assert result["candidate"]["mse"] == pytest.approx(5.0)
    assert result["candidate"]["bias"] == pytest.approx(0.0)
    assert result["candidate"]["centered_mse"] == pytest.approx(5.0)
    assert result["reference"]["mse"] == pytest.approx(1.0)
    assert result["reference"]["bias_squared"] == pytest.approx(1.0)
    assert result["reference"]["centered_mse"] == pytest.approx(0.0)
    assert result["delta_mse"] == pytest.approx(4.0)
    assert result["delta_bias_squared"] == pytest.approx(-1.0)
    assert result["delta_centered_mse"] == pytest.approx(5.0)
    assert result["mean_prediction_shift"] == pytest.approx(-1.0)
    assert result["delta_mse_decomposition_residual"] == pytest.approx(0.0)


def test_series_concentration_and_fraction_better() -> None:
    series = mixed_fold_series(10)
    targets = [10.0] * 10
    reference = [11.0] * 10
    candidate = [20.0] + [10.0] * 9
    data = analysis.prepare_frame(
        make_frame(
            series_ids=series,
            targets=targets,
            candidate=candidate,
            reference=reference,
        ),
        split="validation",
    )
    concentration = analysis.series_concentration(analysis.aggregate_by_series(data))
    result = concentration["candidate_minus_rmtpp"]

    assert result["positive_delta_sse_sum"] == pytest.approx(99.0)
    assert result["negative_delta_sse_sum"] == pytest.approx(-9.0)
    assert result["net_delta_sse_sum"] == pytest.approx(90.0)
    assert result["fraction_series_candidate_better"] == pytest.approx(0.9)
    for row in result["concentration"][:2]:
        assert row["selected_series_count"] == 1
        assert row["selected_positive_delta_sse_sum"] == pytest.approx(99.0)
        assert row["share_of_all_positive_delta_sse"] == pytest.approx(1.0)
        assert row["selected_net_delta_sse_sum"] == pytest.approx(99.0)
        assert row["selected_negative_delta_sse_sum"] == pytest.approx(0.0)


def test_whole_series_fold_is_exact_and_stable_under_reorder() -> None:
    series = ["z", "a", "z", "b", "a", "c"]
    observed = analysis.assign_series_folds(series)
    expected = np.asarray(
        [
            int(
                hashlib.sha256(
                    f"{analysis.FOLD_SALT}|{series_id}".encode("utf-8")
                ).hexdigest(),
                16,
            )
            % 2
            for series_id in series
        ],
        dtype=np.int8,
    )
    assert np.array_equal(observed, expected)
    assert observed[0] == observed[2]
    assert observed[1] == observed[4]

    reversed_series = list(reversed(series))
    reversed_folds = analysis.assign_series_folds(reversed_series)
    original_map = dict(zip(series, observed, strict=True))
    reversed_map = dict(zip(reversed_series, reversed_folds, strict=True))
    assert original_map == reversed_map


def test_malformed_pair_and_nonfinite_prediction_are_rejected() -> None:
    duplicate = make_frame(series_ids=["same", "same"])
    duplicate = duplicate.with_columns(pl.lit(0).alias("target_position"))
    with pytest.raises(ValueError, match="canonical .* pairs are not unique"):
        analysis.prepare_frame(duplicate, split="validation")

    nonfinite = make_frame(series_ids=mixed_fold_series(4)).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(float("nan"))
        .otherwise(pl.col("prediction_candidate"))
        .alias("prediction_candidate")
    )
    with pytest.raises(ValueError, match="prediction_candidate must contain only finite"):
        analysis.prepare_frame(nonfinite, split="train")


def test_every_partition_family_fold_and_sign_case_replays_overall() -> None:
    # Thirty rows cover every 6x5 history/quantity cell.  The resulting 132
    # compact rows also guard against nullable-label schema inference that
    # only fails beyond Polars' default 100-row inference prefix.
    series = mixed_fold_series(30)
    targets = [5.0, 9.0, 21.0, 26.0, 36.0] * 6
    candidate = [value + ((index % 5) - 2) for index, value in enumerate(targets)]
    reference = [value + ((index % 4) - 1) for index, value in enumerate(targets)]
    data = analysis.prepare_frame(
        make_frame(
            series_ids=series,
            targets=targets,
            candidate=candidate,
            reference=reference,
        ),
        split="train",
    )
    summary, strata, folds, _ = analysis.analyze_prepared(
        data, split="train", bootstrap_replicates=20
    )

    overall = summary["overall"]["comparisons"]
    for family in (
        "quantity_band",
        "history_band",
        "history_x_quantity",
        "recent_sign",
    ):
        family_rows = strata.filter(pl.col("partition_family") == family)
        for reference in analysis.REFERENCE_ROLES:
            name = f"candidate_minus_{reference}"
            rows = family_rows.filter(pl.col("comparison") == name)
            assert rows["count"].sum() == len(data.target)
            assert rows["contribution_to_overall_delta_mse"].sum() == pytest.approx(
                overall[name]["delta_mse"]
            )
    for reference in analysis.REFERENCE_ROLES:
        name = f"candidate_minus_{reference}"
        rows = folds.filter(pl.col("comparison") == name)
        assert rows["count"].sum() == len(data.target)
        assert rows["contribution_to_overall_delta_mse"].sum() == pytest.approx(
            overall[name]["delta_mse"]
        )
        cases = summary["paired_under_over_sse_cases"][name]
        assert len(cases) == 4
        assert sum(row["count"] for row in cases) == len(data.target)
        assert sum(row["contribution_to_overall_delta_mse"] for row in cases) == pytest.approx(
            overall[name]["delta_mse"]
        )
    assert summary["partition_contract"][
        "do_not_sum_contributions_across_partition_families"
    ] is True
    assert summary["paired_series_bootstrap"]["interpretation"] == (
        "case_variation_not_seed_uncertainty"
    )


def test_end_to_end_writes_only_compact_outputs_and_optional_search_detail(
    tmp_path: Path,
) -> None:
    frame = make_frame(series_ids=mixed_fold_series(12))
    input_path = tmp_path / "paired.parquet"
    frame.write_parquet(input_path)
    output_dir = tmp_path / "validation"
    detail_dir = tmp_path / "search_artifacts" / "validation"

    result = analysis.run_analysis(
        input_parquet=input_path,
        split="validation",
        output_dir=output_dir,
        detail_dir=detail_dir,
        bootstrap_replicates=20,
    )

    assert set(path.name for path in output_dir.iterdir()) == {
        "summary.json",
        "strata.csv",
        "folds.csv",
    }
    assert (detail_dir / "series_error_contributions.parquet").is_file()
    written = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert written["schema"] == analysis.SCHEMA
    assert written["row_count"] == len(frame)
    assert written["scope"]["held_out_test_evaluated"] is False
    assert written["artifacts"]["series_detail_parquet"]["artifact_class"] == (
        "search_artifact"
    )
    assert result["input"]["sha256"] == analysis.sha256_file(input_path)

    with pytest.raises(ValueError, match="search_artifacts"):
        analysis.run_analysis(
            input_parquet=input_path,
            split="validation",
            output_dir=tmp_path / "bad-output",
            detail_dir=tmp_path / "ordinary-details",
            bootstrap_replicates=20,
        )
