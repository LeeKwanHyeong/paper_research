import numpy as np
import polars as pl

from simple_lab_test.search.analyze_taxi_mark_time_audit import (
    assign_temporal_audit_partition,
    compare_train_only_holdout,
    distribution_effect_summary,
    evaluate_audit_gate,
    extract_loader_train_targets,
    summarize_mark_delta_time,
)


def make_train_frame() -> pl.DataFrame:
    rows = []
    for part in ("a", "b"):
        for seq, (mark, dt) in enumerate(((0, 0), (0, 1), (1, 4), (1, 5))):
            rows.append(
                {
                    "oper_part_no": part,
                    "seq": seq,
                    "delta_t": dt,
                    "demand_qty": float(10**mark),
                    "mark": mark,
                    "scale_residual": 0.0,
                    "chronological_split": "train",
                }
            )
    return pl.DataFrame(rows)


def test_loader_target_extraction_excludes_each_series_first_event() -> None:
    targets, dataset = extract_loader_train_targets(
        make_train_frame(),
        lookback_weeks=168,
        max_seq_len=16,
    )

    assert targets.height == 6
    assert len(dataset) == 6
    assert targets["delta_t"].min() >= 1.0
    assert targets["target_split"].unique().to_list() == ["train"]


def test_temporal_audit_partition_keeps_fit_before_eval() -> None:
    targets, _ = extract_loader_train_targets(
        make_train_frame(),
        lookback_weeks=168,
        max_seq_len=16,
    )
    partitioned = assign_temporal_audit_partition(targets, fit_ratio=0.5)

    for group in partitioned.partition_by("oper_part_no"):
        fit_seq = group.filter(pl.col("audit_partition") == "audit_fit")["target_seq"]
        eval_seq = group.filter(pl.col("audit_partition") == "audit_eval")["target_seq"]
        assert fit_seq.max() < eval_seq.min()


def test_distribution_effect_detects_mark_separation() -> None:
    targets = pl.DataFrame(
        {
            "oper_part_no": ["a"] * 400,
            "target_seq": list(range(400)),
            "mark": [0] * 200 + [1] * 200,
            "delta_t": [1.0] * 200 + [5.0] * 200,
            "log1p_delta_t": [np.log1p(1.0)] * 200 + [np.log1p(5.0)] * 200,
        }
    )
    summary = summarize_mark_delta_time(targets)
    effects = distribution_effect_summary(targets, summary)

    assert effects["supported_mark_count"] == 2
    assert effects["eta_squared_log1p_delta_t"] > 0.9
    assert effects["supported_mark_delta_t_median_ratio"] == 5.0


def test_mark_conditioned_rmtpp_improves_temporal_holdout() -> None:
    rows = []
    for part_index in range(20):
        for target_index in range(40):
            mark = target_index % 2
            dt = 1.0 if mark == 0 else 5.0
            rows.append(
                {
                    "oper_part_no": f"p{part_index:02d}",
                    "target_seq": target_index,
                    "mark": mark,
                    "delta_t": dt,
                    "log1p_delta_t": float(np.log1p(dt)),
                }
            )
    partitioned = assign_temporal_audit_partition(pl.DataFrame(rows))
    holdout, _, series = compare_train_only_holdout(partitioned)

    assert holdout["eval_nll_improvement_pct"] > 0.5
    assert holdout["bootstrap_improvement_pct_ci_low"] > 0.0
    assert series["nll_improvement_pct"].min() > 0.0


def test_audit_gate_requires_every_predeclared_check() -> None:
    source = {"quality_gate": "PASS"}
    loader = {"loader_contract_gate": "PASS"}
    effects = {"supported_mark_count": 2, "eta_squared_log1p_delta_t": 0.02}
    holdout = {
        "eval_nll_improvement_pct": 1.0,
        "bootstrap_improvement_pct_ci_low": 0.2,
        "series_improved_share": 0.60,
        "unseen_eval_mark_count": 0,
        "global_w_at_upper_boundary": False,
        "mark_conditioned_w_at_upper_boundary": False,
    }

    assert evaluate_audit_gate(
        source_quality=source,
        loader_quality=loader,
        effects=effects,
        holdout=holdout,
    )["status"] == "PASS"

    holdout["eval_nll_improvement_pct"] = 0.49
    assert evaluate_audit_gate(
        source_quality=source,
        loader_quality=loader,
        effects=effects,
        holdout=holdout,
    )["status"] == "FAIL"
