import numpy as np
import polars as pl
import pytest
from sklearn.preprocessing import StandardScaler

from simple_lab_test.search.analyze_taxi_pre_window_memory_audit import (
    METRIC_NAMES,
    ProbeMatrix,
    SeriesData,
    assign_temporal_probe_partition,
    build_train_samples,
    evaluate_audit_gate,
    fit_probe_models,
    retrieve_memory_features,
    score_probe_models,
    select_candidate,
    summarize_coverage,
    validate_loader_contract,
    validate_train_source,
    _event_features,
)


def make_train_frame(*, series_count: int = 2, event_count: int = 24) -> pl.DataFrame:
    rows = []
    for part_index in range(series_count):
        for seq in range(event_count):
            mark = (seq + part_index) % 3
            rows.append(
                {
                    "oper_part_no": f"p{part_index:02d}",
                    "seq": seq,
                    "delta_t": 0 if seq == 0 else 1,
                    "demand_qty": float(2 ** (mark + 1)),
                    "mark": mark,
                    "scale_residual": 0.0,
                    "chronological_split": "train",
                }
            )
    return pl.DataFrame(rows)


def test_pre_window_boundary_matches_temporal_and_max_length_contract() -> None:
    frame = make_train_frame(series_count=1, event_count=12)
    samples, dataset, series = build_train_samples(
        frame,
        lookback_weeks=4,
        max_seq_len=5,
    )
    row = samples.filter(pl.col("target_seq") == 8).row(0, named=True)

    assert row["temporal_context_start_index"] == 4
    assert row["effective_context_start_index"] == 4
    assert row["pre_window_count"] == 4
    assert row["context_count"] == 4
    assert row["max_seq_len_excluded_count"] == 0
    assert validate_loader_contract(frame, samples, dataset, series)[
        "loader_contract_gate"
    ] == "PASS"

    short_samples, _, _ = build_train_samples(
        frame,
        lookback_weeks=4,
        max_seq_len=3,
    )
    short = short_samples.filter(pl.col("target_seq") == 8).row(0, named=True)
    assert short["temporal_context_start_index"] == 4
    assert short["effective_context_start_index"] == 6
    assert short["pre_window_count"] == 6
    assert short["context_count"] == 2
    assert short["max_seq_len_excluded_count"] == 2


def test_temporal_probe_partition_keeps_fit_selection_audit_ordered() -> None:
    samples, _, _ = build_train_samples(
        make_train_frame(series_count=2, event_count=30),
        lookback_weeks=8,
        max_seq_len=16,
    )
    partitioned = assign_temporal_probe_partition(samples)

    for group in partitioned.partition_by("oper_part_no"):
        fit = group.filter(pl.col("probe_partition") == "probe_fit")["target_seq"]
        selection = group.filter(pl.col("probe_partition") == "probe_selection")[
            "target_seq"
        ]
        audit = group.filter(pl.col("probe_partition") == "probe_audit")["target_seq"]
        assert fit.max() < selection.min() < audit.min()


def test_memory_retrieval_excludes_target_and_future_but_uses_valid_memory() -> None:
    base = SeriesData(
        part_index=0,
        oper_part_no="a",
        seq=np.arange(12, dtype=np.int64),
        delta_t=np.ones(12, dtype=np.float64),
        mark=np.asarray([0, 1] * 6, dtype=np.int64),
        quantity=np.asarray([2.0, 4.0] * 6, dtype=np.float64),
    )
    scaler = StandardScaler().fit(_event_features(base, 2)[:10])
    expected = retrieve_memory_features(
        base,
        context_start=6,
        context_end=9,
        memory_budget=6,
        topk=2,
        num_marks=2,
        event_scaler=scaler,
    )

    future_changed = SeriesData(
        part_index=0,
        oper_part_no="a",
        seq=base.seq.copy(),
        delta_t=base.delta_t.copy(),
        mark=base.mark.copy(),
        quantity=base.quantity.copy(),
    )
    future_changed.mark[10:] = 1
    future_changed.quantity[10:] = 1_000.0
    observed_after_future_change = retrieve_memory_features(
        future_changed,
        context_start=6,
        context_end=9,
        memory_budget=6,
        topk=2,
        num_marks=2,
        event_scaler=scaler,
    )
    np.testing.assert_allclose(observed_after_future_change, expected, atol=0.0, rtol=0.0)

    memory_changed = SeriesData(
        part_index=0,
        oper_part_no="a",
        seq=base.seq.copy(),
        delta_t=base.delta_t.copy(),
        mark=base.mark.copy(),
        quantity=base.quantity.copy(),
    )
    memory_changed.mark[:6] = 0
    memory_changed.quantity[:6] = np.arange(100.0, 106.0)
    observed_after_memory_change = retrieve_memory_features(
        memory_changed,
        context_start=6,
        context_end=9,
        memory_budget=6,
        topk=2,
        num_marks=2,
        event_scaler=scaler,
    )
    assert not np.allclose(observed_after_memory_change, expected)


def test_coverage_uses_all_targets_and_series_at_ge8() -> None:
    samples, _, _ = build_train_samples(
        make_train_frame(series_count=2, event_count=24),
        lookback_weeks=4,
        max_seq_len=8,
    )
    summary, thresholds, by_series, by_mark = summarize_coverage(samples, series_count=2)

    expected_eligible = samples.filter(pl.col("pre_window_count") >= 8).height
    assert summary["eligible_target_count_ge_8"] == expected_eligible
    assert summary["eligible_series_count_ge_8"] == 2
    covered_target_count = thresholds.filter(pl.col("threshold") == 8)[
        "covered_target_count"
    ][0]
    assert covered_target_count == expected_eligible
    assert by_series.height == 2
    assert by_mark.height == 3


def test_candidate_selection_uses_worst_then_mean_then_compute_tiebreak() -> None:
    rows = [
        {
            "memory_budget": 64,
            "topk": 8,
            "selection_pass": True,
            "worst_improvement_pct": 0.2,
            "mean_improvement_pct": 1.5,
        },
        {
            "memory_budget": 32,
            "topk": 4,
            "selection_pass": True,
            "worst_improvement_pct": 0.3,
            "mean_improvement_pct": 1.0,
        },
        {
            "memory_budget": 16,
            "topk": 4,
            "selection_pass": False,
            "worst_improvement_pct": 10.0,
            "mean_improvement_pct": 10.0,
        },
    ]
    selected = select_candidate(rows)
    assert selected is not None
    assert selected["memory_budget"] == 32
    assert selected["topk"] == 4

    rows[0]["worst_improvement_pct"] = 0.3
    rows[0]["mean_improvement_pct"] = 1.0
    selected = select_candidate(rows)
    assert selected is not None
    assert selected["memory_budget"] == 32


def test_probe_models_produce_finite_aligned_metrics() -> None:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(80, 5))
    marks = (features[:, 0] > 0.0).astype(np.int64)
    matrix = ProbeMatrix(
        sample_index=np.arange(80),
        oper_part_no=np.asarray([f"p{index % 4}" for index in range(80)], dtype=object),
        base_features=features,
        augmented_features=np.column_stack([features, features[:, 0] ** 2]),
        target_mark=marks,
        target_log1p_dt=1.0 + 0.5 * features[:, 1],
        target_log2_qty=2.0 - 0.25 * features[:, 2],
    )
    models = fit_probe_models(matrix.base_features, matrix)
    metrics, errors = score_probe_models(models, matrix.base_features, matrix)

    assert metrics["unseen_mark_count"] == 0
    for name in METRIC_NAMES:
        assert np.isfinite(metrics[name])
        assert errors[name].shape == (80,)


def test_audit_gate_requires_coverage_selection_final_gain_and_ci() -> None:
    source = {"quality_gate": "PASS"}
    loader = {"loader_contract_gate": "PASS"}
    coverage = {
        "eligible_target_share_ge_8": 0.40,
        "eligible_series_share_ge_8": 0.85,
    }
    selected = {"primary_metric": "marker_ce"}
    final = {
        "marker_ce_improvement_pct": 1.2,
        "log1p_dt_mae_improvement_pct": -0.5,
        "log2_qty_mae_improvement_pct": 0.1,
        "marker_ce_bootstrap_ci_low": 0.2,
        "baseline_unseen_mark_count": 0,
        "augmented_unseen_mark_count": 0,
    }
    assert evaluate_audit_gate(
        source_quality=source,
        loader_quality=loader,
        coverage_summary=coverage,
        selected_candidate=selected,
        final_audit=final,
    )["status"] == "PASS"

    final["log1p_dt_mae_improvement_pct"] = -1.01
    assert evaluate_audit_gate(
        source_quality=source,
        loader_quality=loader,
        coverage_summary=coverage,
        selected_candidate=selected,
        final_audit=final,
    )["status"] == "FAIL"


def test_train_source_rejects_non_train_rows() -> None:
    frame = make_train_frame(series_count=1, event_count=8).with_columns(
        pl.when(pl.col("seq") == 7)
        .then(pl.lit("validation"))
        .otherwise(pl.col("chronological_split"))
        .alias("chronological_split")
    )
    with pytest.raises(ValueError, match="non-train"):
        validate_train_source(frame, expected_series_count=None)
