from argparse import Namespace
from pathlib import Path

import numpy as np
import polars as pl

from simple_lab_test.search.analyze_taxi_pre_window_memory_audit import SeriesData
from simple_lab_test.search.analyze_taxi_time_source_isolation_audit import (
    SourceFeatureMatrix,
    bootstrap_oof_improvements,
    build_active_window_features,
    build_full_pre_window_attribution_features,
    build_rolling_origin_folds,
    build_temporal_pre_window_features,
    evaluate_rolling_origin,
    evaluate_time_source_gate,
    write_outputs,
)


def make_series() -> SeriesData:
    size = 24
    return SeriesData(
        part_index=0,
        oper_part_no="taxi-00",
        seq=np.arange(size, dtype=np.int64),
        delta_t=np.asarray([0, *[1 + index % 5 for index in range(1, size)]]),
        mark=np.asarray([index % 3 for index in range(size)], dtype=np.int64),
        quantity=np.asarray([2.0 ** (1 + index % 4) for index in range(size)]),
    )


def copy_series(series: SeriesData) -> SeriesData:
    return SeriesData(
        part_index=series.part_index,
        oper_part_no=series.oper_part_no,
        seq=series.seq.copy(),
        delta_t=series.delta_t.copy(),
        mark=series.mark.copy(),
        quantity=series.quantity.copy(),
    )


def make_probe_matrix(*, series_count: int = 4, target_count: int = 20) -> SourceFeatureMatrix:
    sample_index = []
    parts = []
    target_seq = []
    targets = []
    p0_rows = []
    p1_rows = []
    p2_rows = []
    for part_index in range(series_count):
        for sequence in range(target_count):
            sample_index.append(part_index * target_count + sequence)
            parts.append(f"taxi-{part_index:02d}")
            target_seq.append(sequence)
            target = 0.2 + 0.05 * sequence
            targets.append(target)
            p0 = np.asarray([1.0, float(sequence % 2)], dtype=np.float64)
            p0_rows.append(p0)
            p1_rows.append(np.concatenate([p0, [float(sequence)]]))
            p2_rows.append(
                np.concatenate([p0, [float(sequence), float((sequence + part_index) % 3)]])
            )
    return SourceFeatureMatrix(
        sample_index=np.asarray(sample_index, dtype=np.int64),
        oper_part_no=np.asarray(parts, dtype=object),
        target_seq=np.asarray(target_seq, dtype=np.int64),
        target_log1p_dt=np.asarray(targets, dtype=np.float64),
        p0_window=np.vstack(p0_rows),
        p1_temporal=np.vstack(p1_rows),
        p2_full=np.vstack(p2_rows),
    )


def make_train_frame(*, series_count: int = 4, event_count: int = 64) -> pl.DataFrame:
    rows = []
    for part_index in range(series_count):
        for seq in range(event_count):
            mark = (seq + part_index) % 3
            rows.append(
                {
                    "oper_part_no": f"taxi-{part_index:02d}",
                    "seq": seq,
                    "delta_t": 0 if seq == 0 else 1 + (seq + part_index) % 7,
                    "demand_qty": float(2 ** (mark + 1)),
                    "mark": mark,
                    "scale_residual": 0.0,
                    "chronological_split": "train",
                }
            )
    return pl.DataFrame(rows)


def passing_gate_inputs() -> dict[str, dict]:
    return {
        "source_quality": {"quality_gate": "PASS"},
        "loader_quality": {"loader_contract_gate": "PASS"},
        "coverage_summary": {
            "eligible_target_share_ge_8": 0.40,
            "eligible_series_share_ge_8": 0.85,
        },
        "feature_contract": {"status": "PASS"},
        "rolling_contract": {"status": "PASS"},
        "oof_summary": {
            "fold_count": 3,
            "duplicate_oof_target_count": 0,
            "errors_finite": True,
            "p1_improvement_pct": 1.2,
            "p1_improved_fold_count": 2,
        },
        "bootstrap_summary": {"p1_bootstrap_ci_low": 0.1},
    }


def test_source_features_exclude_target_and_future_fields() -> None:
    series = make_series()
    expected_window = build_active_window_features(
        series,
        context_start=10,
        context_end=17,
        num_marks=3,
    )
    expected_temporal = build_temporal_pre_window_features(
        series,
        context_start=10,
        context_end=17,
    )
    expected_attribution = build_full_pre_window_attribution_features(
        series,
        context_start=10,
        num_marks=3,
    )

    changed = copy_series(series)
    changed.delta_t[18:] = 999
    changed.mark[18:] = 2
    changed.quantity[18:] = 10_000.0

    np.testing.assert_array_equal(
        build_active_window_features(
            changed,
            context_start=10,
            context_end=17,
            num_marks=3,
        ),
        expected_window,
    )
    np.testing.assert_array_equal(
        build_temporal_pre_window_features(
            changed,
            context_start=10,
            context_end=17,
        ),
        expected_temporal,
    )
    np.testing.assert_array_equal(
        build_full_pre_window_attribution_features(
            changed,
            context_start=10,
            num_marks=3,
        ),
        expected_attribution,
    )


def test_p1_isolates_temporal_source_from_pre_window_mark_and_quantity() -> None:
    series = make_series()
    expected_temporal = build_temporal_pre_window_features(
        series,
        context_start=10,
        context_end=17,
    )
    expected_attribution = build_full_pre_window_attribution_features(
        series,
        context_start=10,
        num_marks=3,
    )

    changed = copy_series(series)
    changed.mark[:10] = 0
    changed.quantity[:10] = np.arange(100.0, 110.0)

    np.testing.assert_array_equal(
        build_temporal_pre_window_features(
            changed,
            context_start=10,
            context_end=17,
        ),
        expected_temporal,
    )
    assert not np.allclose(
        build_full_pre_window_attribution_features(
            changed,
            context_start=10,
            num_marks=3,
        ),
        expected_attribution,
    )

    changed.delta_t[:10] += 5
    assert not np.allclose(
        build_temporal_pre_window_features(
            changed,
            context_start=10,
            context_end=17,
        ),
        expected_temporal,
    )


def test_rolling_origin_folds_are_chronological_and_oof_disjoint() -> None:
    matrix = make_probe_matrix()
    folds, contract = build_rolling_origin_folds(matrix)

    assert contract["status"] == "PASS"
    assert len(folds) == 3
    all_eval = np.concatenate([fold.eval_positions for fold in folds])
    assert np.unique(all_eval).size == all_eval.size
    for fold in folds:
        for part in np.unique(matrix.oper_part_no):
            train = fold.train_positions[matrix.oper_part_no[fold.train_positions] == part]
            evaluate = fold.eval_positions[matrix.oper_part_no[fold.eval_positions] == part]
            assert matrix.target_seq[train].max() < matrix.target_seq[evaluate].min()


def test_rolling_probe_and_series_bootstrap_are_finite_and_deterministic() -> None:
    matrix = make_probe_matrix()
    folds, _ = build_rolling_origin_folds(matrix)
    fold_metrics, oof, summary = evaluate_rolling_origin(matrix, folds)
    first_bootstrap, by_series = bootstrap_oof_improvements(
        oof,
        replicates=100,
        seed=42,
    )
    second_bootstrap, _ = bootstrap_oof_improvements(
        oof,
        replicates=100,
        seed=42,
    )

    assert fold_metrics.height == 3
    assert summary["duplicate_oof_target_count"] == 0
    assert summary["errors_finite"] is True
    assert summary["p1_improvement_pct"] > 0.0
    assert first_bootstrap == second_bootstrap
    assert by_series.height == 4
    assert all(
        np.isfinite(first_bootstrap[key])
        for key in (
            "p1_bootstrap_ci_low",
            "p1_bootstrap_ci_high",
            "p2_bootstrap_ci_low",
            "p2_bootstrap_ci_high",
        )
    )


def test_p2_cannot_substitute_for_a_failing_p1_gate() -> None:
    inputs = passing_gate_inputs()
    assert evaluate_time_source_gate(**inputs)["status"] == "PASS"

    inputs["oof_summary"]["p1_improvement_pct"] = -1.0
    inputs["oof_summary"]["p2_improvement_pct"] = 20.0
    inputs["oof_summary"]["p2_improved_fold_count"] = 3
    inputs["bootstrap_summary"]["p2_bootstrap_ci_low"] = 10.0
    gate = evaluate_time_source_gate(**inputs)

    assert gate["status"] == "FAIL"
    assert gate["p2_can_pass_gate"] is False
    assert gate["primary_comparison"] == "p1_temporal_vs_p0_window"


def test_write_outputs_emits_train_only_artifact_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "taxi_train.parquet"
    output_dir = tmp_path / "audit"
    make_train_frame().write_parquet(dataset_path)
    args = Namespace(
        dataset=dataset_path,
        output_dir=output_dir,
        lookback_weeks=8,
        max_seq_len=16,
        execution_server="local-test",
        tmux_session="none",
        source_revision="0" * 40,
    )

    write_outputs(args, expected_series_count=None, bootstrap_replicates=50)

    manifest = pl.read_json(output_dir / "audit_manifest.json").to_dicts()[0]
    summary = pl.read_json(output_dir / "audit_summary.json").to_dicts()[0]
    fold_metrics = pl.read_csv(output_dir / "data/rolling_fold_metrics.csv")
    assert manifest["held_out_target_data_read"] is False
    assert manifest["audit_contract"]["primary_comparison"] == "p1_vs_p0"
    assert manifest["audit_contract"]["p2_can_pass_gate"] is False
    assert manifest["audit_contract"]["v6_memory_budget_reused"] is False
    assert summary["status"] == "completed"
    assert fold_metrics.height == 3
    assert (output_dir / "data/oof_target_errors.parquet").is_file()
    assert (output_dir / "plots/rolling_fold_improvement.png").is_file()
    assert (output_dir / "report.md").is_file()
