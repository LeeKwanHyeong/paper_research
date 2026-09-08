"""Pure tests for the paired BOUNDED-QK error attribution."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from paper.scripts.analyze_hard_lmm_bounded_qk_error_diagnostic import (
    FOLD_SALT,
    HISTORY_STRATA,
    RECENT_SIGNAL_STRATA,
    TARGET_STRATA,
    analyze,
    assign_series_folds,
    canonicalize_frame,
    comparison_metrics,
    history_stratum,
    quantity_metrics,
    recent_signal_stratum,
    target_stratum,
    validation_oracle_shift_diagnostics,
    validate_run_audit,
    write_csv_atomic,
    write_json_atomic,
)


def paired_frame() -> pl.DataFrame:
    target = np.asarray([1, 8, 9, 20, 21, 25, 26, 35, 36, 50, 15, 30], dtype=float)
    b_error = np.asarray([1, -2, 2, -1, 1, -3, 2, -2, 4, -5, 1, -1], dtype=float)
    bounded_error = np.asarray([0.5, -1, 3, -2, 0.5, -2, 3, -1, 5, -4, 2, -0.5])
    full_error = np.asarray([1.5, -1, 2.5, -0.5, 0.5, -4, 1, -3, 3, -6, 0.5, -2])
    # All predictions remain nonnegative while both series folds are populated.
    return pl.DataFrame(
        {
            "series_id": ["a", "c"] * 6,
            "target_qty": target,
            "prediction_b": target + b_error,
            "prediction_full": target + full_error,
            "prediction_bounded_qk": target + bounded_error,
            "history_length": [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 1],
            "recent_signal": [-1.0, 0.0, 0.5, -0.0, -0.2, 0.2, -2, 2, 0, -3, 3, 0],
            "split": ["validation"] * len(target),
        }
    )


def test_frozen_strata_include_boundaries_in_lower_bucket() -> None:
    assert target_stratum([0, 8, 8.1, 20, 20.1, 25, 25.1, 35, 35.1]).tolist() == [
        "<=p50",
        "<=p50",
        "p50-p90",
        "p50-p90",
        "p90-p95",
        "p90-p95",
        "p95-p99",
        "p95-p99",
        ">p99",
    ]
    assert history_stratum([1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63]).tolist() == [
        "1",
        "2-3",
        "2-3",
        "4-7",
        "4-7",
        "8-15",
        "8-15",
        "16-31",
        "16-31",
        "32-63",
        "32-63",
    ]
    assert recent_signal_stratum([-1e-12, -0.0, 0.0, 1e-12]).tolist() == [
        "<0",
        "=0",
        "=0",
        ">0",
    ]
    with pytest.raises(ValueError, match=r"\[1, 63\]"):
        history_stratum([0, 64])


def test_series_folds_are_deterministic_and_series_disjoint() -> None:
    series = np.asarray(["a", "c", "a", "c", "a"], dtype=object)
    first = assign_series_folds(series, salt=FOLD_SALT)
    second = assign_series_folds(series[::-1], salt=FOLD_SALT)[::-1]
    assert np.array_equal(first, second)
    assert first[0] == first[2] == first[4]
    assert first[1] == first[3]
    assert set(first.tolist()) == {0, 1}


def test_mse_bias_decomposition_and_signed_mass_are_exact() -> None:
    metrics = quantity_metrics([0, 0, 0], [1, 2, 3])
    assert metrics["mse"] == pytest.approx(
        metrics["centered_mse"] + metrics["bias_squared"], abs=1e-15
    )
    assert metrics["mse_decomposition_residual"] == pytest.approx(0.0, abs=1e-15)

    comparison = comparison_metrics(
        target=[0, 0, 0],
        candidate=[2, 1, 1],
        reference=[1, 2, 1],
    )
    assert comparison["delta_se_sum"] == pytest.approx(0.0)
    assert comparison["gross_positive_harm_mass"] == pytest.approx(3.0)
    assert comparison["gross_negative_help_mass"] == pytest.approx(3.0)
    assert comparison["harm_row_count"] == 1
    assert comparison["help_row_count"] == 1
    assert comparison["tie_row_count"] == 1
    assert comparison["delta_mse_decomposition_residual"] == pytest.approx(
        0.0, abs=1e-15
    )
    assert comparison["delta_se_formula_residual_sum"] == pytest.approx(
        0.0, abs=1e-15
    )
    assert comparison["delta_se_mass_residual"] == pytest.approx(0.0, abs=1e-15)


def test_validation_oracle_shift_is_explicitly_non_performance() -> None:
    rows = validation_oracle_shift_diagnostics(
        target=[10, 20, 30],
        candidate=[8, 18, 28],
        reference=[9, 19, 29],
    )
    by_rule = {row["rule"]: row for row in rows}
    assert by_rule["match_reference_bias"]["shift"] == pytest.approx(1.0)
    assert by_rule["match_reference_bias"]["bias"] == pytest.approx(-1.0)
    assert by_rule["zero_candidate_bias"]["shift"] == pytest.approx(2.0)
    assert by_rule["zero_candidate_bias"]["bias"] == pytest.approx(0.0)
    assert all(row["uses_validation_targets"] for row in rows)
    assert all(row["descriptive_only"] for row in rows)
    assert not any(row["eligible_as_performance_result"] for row in rows)


def test_analysis_covers_p50_p90_situations_folds_and_mass() -> None:
    analysis, tables = analyze(paired_frame(), top_series_count=1)

    assert analysis["status"] == "success"
    assert analysis["scope"]["held_out_test_evaluated"] is False
    assert analysis["input_audit"]["row_count"] == 12
    assert set(analysis["input_audit"]["series_fold_counts"]) == {"0", "1"}
    assert [row["label"] for row in analysis["target_strata"]] == list(TARGET_STRATA)
    assert [row["label"] for row in analysis["history_strata"]] == list(HISTORY_STRATA)
    assert [row["label"] for row in analysis["recent_signal_strata"]] == list(
        RECENT_SIGNAL_STRATA
    )
    assert all(
        row["delta_mse_equals_delta_centered_mse_plus_delta_bias_squared"]
        and row[
            "delta_se_equals_two_base_error_times_delta_prediction_plus_delta_prediction_squared"
        ]
        and row["gross_positive_minus_gross_negative_equals_net_delta_se"]
        for row in analysis["required_decomposition_audit"].values()
    )

    p50_history = tables["p50_p90_by_history.csv"]
    p50_recent = tables["p50_p90_by_recent_signal.csv"]
    assert p50_history["count"].sum() == 3
    assert p50_recent["count"].sum() == 3
    assert tables["series_folds.csv"].filter(pl.col("scope") == "all").height == 2
    assert set(tables["series_contributions.csv"]["scope"].unique()) == {
        "all",
        "p50-p90",
    }
    top_series = tables["top_series.csv"]
    assert top_series.height == 3
    assert (
        top_series.filter(pl.col("direction") == "harm")["net_delta_se_sum"] > 0
    ).all()
    assert (
        top_series.filter(pl.col("direction") == "help")["net_delta_se_sum"] < 0
    ).all()

    concentration = tables["top_residual_delta.csv"]
    assert set(concentration["fraction"].to_list()) == {0.001, 0.01, 0.05}
    assert set(concentration["selection"].to_list()) == {
        "largest_positive_harm",
        "largest_negative_help",
        "largest_absolute_change",
    }
    comparison = analysis["overall"]["comparisons"]["BOUNDED_QK_minus_B"]
    assert comparison["delta_se_sum"] == pytest.approx(
        comparison["gross_positive_harm_mass"]
        - comparison["gross_negative_help_mass"]
    )


def test_column_aliases_split_guard_and_atomic_outputs(tmp_path) -> None:
    aliased = paired_frame().rename(
        {
            "series_id": "oper_part_no",
            "target_qty": "target_quantity",
            "prediction_b": "b_qty_prediction",
            "prediction_full": "full_qty_prediction",
            "prediction_bounded_qk": "bounded_qk_qty_prediction",
            "history_length": "model_history_length",
            "recent_signal": "recent3_mean_log_minus_history_mean_log",
        }
    )
    canonical, resolved = canonicalize_frame(aliased)
    assert set(resolved) == {
        "series_id",
        "target_qty",
        "prediction_b",
        "prediction_full",
        "prediction_bounded_qk",
        "history_length",
        "recent_signal",
    }
    assert "series_id" in canonical.columns

    bad = aliased.with_columns(pl.lit("test").alias("split"))
    with pytest.raises(ValueError, match="only validation"):
        canonicalize_frame(bad)

    json_path = tmp_path / "nested" / "result.json"
    csv_path = tmp_path / "nested" / "result.csv"
    write_json_atomic(json_path, {"finite": 1.0})
    write_csv_atomic(csv_path, pl.DataFrame({"x": [1, 2]}))
    assert json.loads(json_path.read_text()) == {"finite": 1.0}
    assert pl.read_csv(csv_path)["x"].to_list() == [1, 2]
    assert not list(json_path.parent.glob("*.tmp"))


def test_run_audit_pins_input_and_rejects_training(tmp_path) -> None:
    parquet = tmp_path / "paired.parquet"
    paired_frame().write_parquet(parquet)
    from paper.scripts.analyze_hard_lmm_bounded_qk_error_diagnostic import sha256_file

    before = {"B": "a" * 64, "FULL": "b" * 64, "BOUNDED_QK": "c" * 64}
    payload = {
        "schema": "hard_lmm_bounded_qk_error_diagnostic_run_v1",
        "status": "success",
        "held_out_test_evaluated": False,
        "training_performed": False,
        "calibration_fit_performed": False,
        "scope": {
            "target_split": "validation",
            "held_out_test": False,
            "training": False,
            "checkpoint_selection": False,
            "parameter_updates": False,
            "calibration_fit": False,
        },
        "paired_predictions": {
            "sha256": sha256_file(parquet),
            "row_count": 12,
            "canonical_identity_unique": True,
        },
        "state_identity": {
            "before": before,
            "after": before,
            "unchanged": True,
            "all_gradients_absent": True,
        },
        "contract": {"id": "hard_lmm_bounded_qk_error_diagnostic_v1", "sha256": "d" * 64},
    }
    audit = tmp_path / "run_audit.json"
    audit.write_text(json.dumps(payload))
    result = validate_run_audit(audit, input_parquet=parquet, contract_path=None)
    assert result["paired_prediction_row_count"] == 12
    assert result["source_model_states_unchanged"] is True

    payload["training_performed"] = True
    audit.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training_performed"):
        validate_run_audit(audit, input_parquet=parquet, contract_path=None)
