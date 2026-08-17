import pytest

from paper.scripts.build_count_aware_tail_shared_multiseed_results import (
    collect_run_rows,
    summarize_runs,
    validate_contracts,
)


def contract(*, seeds, status="complete", data_sha="data"):
    return {
        "status": status,
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
        "seeds": list(seeds),
    }


def run_row(backbone, variant, seed, mae, rmse):
    return {
        "status": "success",
        "backbone": backbone,
        "variant": variant,
        "seed": str(seed),
        "best_val_joint_objective": "-3.5",
        "best_val_time_nll": "-3.6",
        "best_val_log_qty_mse": "0.1",
        "best_val_qty_mae": str(mae),
        "best_val_qty_rmse": str(rmse),
        "best_epoch": "20",
        "completed_epochs": "60",
        "elapsed_seconds": "10",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": "False",
        "checkpoint_state_sha256": "digest",
    }


def test_contract_validation_rejects_mismatched_data():
    with pytest.raises(ValueError, match="data_sha256"):
        validate_contracts(
            contract(seeds=(42, 52, 62)),
            contract(seeds=(42,)),
            contract(seeds=(52, 62), data_sha="other"),
        )


def test_collect_and_summarize_three_seed_grid():
    baseline = []
    for backbone, base_mae in (("rmtpp", 3.0), ("thp", 1.0)):
        for index, seed in enumerate((42, 52, 62)):
            baseline.append(
                run_row(backbone, "count_only_log_regression", seed, base_mae + index, 2.0 + index)
            )
    seed42 = [
        run_row("titantpp", "count_only_log_mse_tail_shared", 42, 0.7, 1.7)
    ]
    extension = [
        run_row("titantpp", "count_only_log_mse_tail_shared", 52, 0.8, 1.8),
        run_row("titantpp", "count_only_log_mse_tail_shared", 62, 0.9, 1.9),
    ]

    rows = collect_run_rows(baseline, seed42, extension)
    summaries = {row["model"]: row for row in summarize_runs(rows)}

    assert len(rows) == 9
    assert summaries["titantpp_t1"]["best_val_qty_mae_mean"] == pytest.approx(0.8)
    assert summaries["titantpp_t1"]["best_val_qty_mae_std"] == pytest.approx(0.1)
