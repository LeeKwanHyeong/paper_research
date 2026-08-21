from __future__ import annotations

import pytest

from paper.scripts.build_count_aware_external_t0_results import (
    collect_runs,
    summarize_runs,
    validate_contracts,
)


def contract(*, models, role=None, data_sha="data"):
    value = {
        "status": "complete",
        "dataset": "intermittent_frozen_5000",
        "data_sha256": data_sha,
        "split_manifest_sha256": "split",
        "split_rows": {"train": 10, "validation": 4, "test": 3},
        "epochs": 300,
        "batch_size": 128,
        "lr": 1e-3,
        "lookback_weeks": 520,
        "max_seq_len": 256,
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "seeds": [42, 52, 62],
        "backbones": list(models),
    }
    if role is not None:
        value.update({
            "model_role": role,
            "quantity_variants": ["count_only_log_regression"],
            "time_head": {"mode": "legacy_clamped_rmtpp"},
            "early_stopping": {
                "min_epochs": 40,
                "patience": 40,
                "restore": "best_validation_joint_objective",
            },
        })
    return value


def run(model, seed, mae):
    return {
        "status": "success",
        "backbone": model,
        "variant": "count_only_log_regression",
        "seed": str(seed),
        "best_val_joint_objective": "-3.5",
        "best_val_time_nll": "-3.6",
        "best_val_log_qty_mse": "0.1",
        "best_val_qty_mae": str(mae),
        "best_val_qty_rmse": "2.0",
        "best_epoch": "20",
        "completed_epochs": "60",
        "elapsed_seconds": "10",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": "False",
        "checkpoint_state_sha256": "digest",
    }


def test_contract_requires_official_t0_extension():
    base = contract(models=("rmtpp", "thp", "titantpp"))
    extension = contract(models=("nhp", "sahp"), role="t0_common_control")
    validate_contracts(base, [extension])
    extension["time_head"]["mode"] = "lognormal_duration"
    with pytest.raises(ValueError, match="time-head"):
        validate_contracts(base, [extension])


def test_contract_rejects_data_mismatch():
    base = contract(models=("rmtpp", "thp", "titantpp"))
    extension = contract(
        models=("nhp", "sahp"), role="t0_common_control", data_sha="other"
    )
    with pytest.raises(ValueError, match="data_sha256"):
        validate_contracts(base, [extension])


def test_collects_exact_four_model_three_seed_grid():
    base_rows = [run(model, seed, 1.0) for model in ("rmtpp", "thp") for seed in (42, 52, 62)]
    extension_rows = [run(model, seed, 0.5) for model in ("nhp", "sahp") for seed in (42, 52, 62)]
    rows = collect_runs(base_rows, [extension_rows])
    summary = {row["model"]: row for row in summarize_runs(rows)}
    assert len(rows) == 12
    assert summary["nhp"]["best_val_qty_mae_mean"] == pytest.approx(0.5)
    assert set(summary) == {"rmtpp", "thp", "nhp", "sahp"}


def test_accepts_non_overlapping_extension_shards():
    base = contract(models=("rmtpp", "thp", "titantpp"))
    nhp_42_52 = contract(models=("nhp",), role="t0_common_control")
    nhp_42_52["seeds"] = [42, 52]
    nhp_62 = contract(models=("nhp",), role="t0_common_control")
    nhp_62["seeds"] = [62]
    sahp = contract(models=("sahp",), role="t0_common_control")
    validate_contracts(base, [nhp_42_52, nhp_62, sahp])


def test_rejects_duplicate_extension_shards():
    base = contract(models=("rmtpp", "thp", "titantpp"))
    first = contract(models=("nhp", "sahp"), role="t0_common_control")
    second = contract(models=("nhp",), role="t0_common_control")
    with pytest.raises(ValueError, match="duplicate"):
        validate_contracts(base, [first, second])
