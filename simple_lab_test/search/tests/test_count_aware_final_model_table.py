import copy

import pytest

from paper.scripts.build_count_aware_final_model_table import (
    build_claims,
    build_table_rows,
    validate_metadata,
)


METRIC_BASES = (
    "best_val_joint_objective",
    "best_val_time_nll",
    "best_val_log_qty_mse",
    "best_val_qty_mae",
    "best_val_qty_rmse",
)


def summary(model: str, mae: float, rmse: float) -> dict[str, str]:
    row = {"model": model, "n_seeds": "3"}
    for metric in METRIC_BASES:
        value = -3.5 if "nll" in metric or "objective" in metric else 0.01
        row[f"{metric}_mean"] = str(value)
        row[f"{metric}_std"] = "0.001"
    row["best_val_qty_mae_mean"] = str(mae)
    row["best_val_qty_rmse_mean"] = str(rmse)
    return row


def sources():
    rmtpp = summary("rmtpp", 3.0, 10.0)
    thp = summary("thp", 0.6, 2.1)
    external = {
        "rmtpp": rmtpp,
        "thp": thp,
        "nhp": summary("nhp", 5.0, 15.0),
        "sahp": summary("sahp", 1.1, 3.8),
    }
    t1 = {
        "rmtpp": copy.deepcopy(rmtpp),
        "thp": copy.deepcopy(thp),
        "titantpp": summary("titantpp", 0.75, 1.9),
        "titantpp_t1": summary("titantpp_t1", 0.7, 1.8),
    }
    return external, t1


def test_build_table_separates_t0_and_proposed_roles():
    external, t1 = sources()
    rows = build_table_rows(external, t1)

    assert [row["model"] for row in rows] == [
        "rmtpp", "thp", "nhp", "sahp", "titantpp", "titantpp_t1"
    ]
    assert rows[-2]["quantity_objective"] == "Direct log-MSE"
    assert rows[-1]["table_role"] == "Proposed method"
    assert rows[-1]["quantity_objective"] == "Log-MSE + tail-aware auxiliary"


def test_build_table_rejects_overlapping_metric_mismatch():
    external, t1 = sources()
    t1["thp"]["best_val_qty_mae_mean"] = "0.7"

    with pytest.raises(ValueError, match="overlapping thp metric mismatch"):
        build_table_rows(external, t1)


def test_claims_preserve_mae_rmse_tradeoff():
    external, t1 = sources()
    claims = build_claims(build_table_rows(external, t1))

    assert claims["versus_thp_t0"]["quantity_mae_improvement_pct"] < 0
    assert claims["versus_thp_t0"]["quantity_rmse_improvement_pct"] > 0
    assert claims["versus_titantpp_t0"]["quantity_mae_improvement_pct"] > 0


def test_metadata_rejects_held_out_test_usage():
    external = {
        "status": "complete",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "model_role": "t0_common_control",
    }
    t1 = {
        "status": "complete",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": True,
        "model_roles": {"titantpp": "t0_common_control", "titantpp_t1": "t1_incumbent"},
    }

    with pytest.raises(ValueError, match="held-out test"):
        validate_metadata(external, t1)
