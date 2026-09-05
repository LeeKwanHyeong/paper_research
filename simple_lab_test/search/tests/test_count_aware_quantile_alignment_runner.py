"""Train-target fitting and leakage contracts for quantile alignment."""

from __future__ import annotations

import math
import json
import sys
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest

from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts.run_count_aware_tpp_backbone_control import (
    QUANTILE_ADAPTIVE_QUANTILES,
    QUANTILE_ADAPTIVE_RAW_WEIGHTS,
    derive_quantile_adaptive_contract,
    exact_target_population,
)
from paper.scripts.count_aware_tpp_backbone.core import (
    load_train_validation_frame,
    prepare_count_frame,
)


def synthetic_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for part_index, part in enumerate(("a", "b")):
        for seq in range(1, 111):
            split = "train" if seq <= 100 else "validation" if seq <= 105 else "test"
            quantity = float(seq + 100 * part_index)
            if split == "test":
                quantity = 1_000_000.0 + quantity
            rows.append({
                "oper_part_no": part,
                "seq": seq,
                "delta_t": float((seq % 7) + 1),
                "demand_qty": quantity,
                "chronological_split": split,
            })
    return pl.DataFrame(rows)


def test_exact_population_excludes_first_event_and_held_out_rows(tmp_path) -> None:
    path = tmp_path / "fixed_split.parquet"
    synthetic_frame().write_parquet(path)
    materialized = load_train_validation_frame(path)
    assert set(materialized["chronological_split"].to_list()) == {"train", "validation"}
    prepared = prepare_count_frame(materialized)

    train, train_population = exact_target_population(
        prepared,
        target_split="train",
        lookback_weeks=520,
        max_seq_len=256,
    )
    validation, validation_population = exact_target_population(
        prepared,
        target_split="validation",
        lookback_weeks=520,
        max_seq_len=256,
    )

    assert train_population["target_count"] == 198
    assert validation_population["target_count"] == 10
    assert 1.0 not in train and 101.0 not in train and 102.0 in train
    assert float(max(train.max(), validation.max())) < 1_000_000.0
    assert train_population["target_identity_sha256"] != validation_population[
        "target_identity_sha256"
    ]


def test_quantile_contract_uses_frozen_shared_rule_and_mean_one() -> None:
    frame = prepare_count_frame(
        synthetic_frame().filter(pl.col("chronological_split") != "test")
    )
    targets, _ = exact_target_population(
        frame,
        target_split="train",
        lookback_weeks=520,
        max_seq_len=256,
    )
    contract = derive_quantile_adaptive_contract(
        frame,
        lookback_weeks=520,
        max_seq_len=256,
    )

    expected_boundaries = np.quantile(
        targets,
        np.asarray(QUANTILE_ADAPTIVE_QUANTILES),
        method="nearest",
    )
    assert contract["quantiles"] == list(QUANTILE_ADAPTIVE_QUANTILES)
    assert contract["raw_bin_weights"] == list(QUANTILE_ADAPTIVE_RAW_WEIGHTS)
    assert contract["boundaries"] == expected_boundaries.tolist()
    assert sum(contract["bin_counts"]) == targets.size
    assert math.isclose(
        contract["normalized_train_target_mean"],
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert len(contract["contract_sha256"]) == 64


def test_target_identity_is_deterministic_and_seq_sensitive() -> None:
    frame = prepare_count_frame(
        synthetic_frame().filter(pl.col("chronological_split") != "test")
    )
    _, first = exact_target_population(
        frame,
        target_split="validation",
        lookback_weeks=520,
        max_seq_len=256,
    )
    _, replay = exact_target_population(
        frame,
        target_split="validation",
        lookback_weeks=520,
        max_seq_len=256,
    )
    changed = frame.with_columns(
        pl.when(
            (pl.col("oper_part_no") == "a")
            & (pl.col("chronological_split") == "validation")
        )
        .then(pl.col("seq") + 1_000)
        .otherwise(pl.col("seq"))
        .alias("seq")
    )
    _, changed_population = exact_target_population(
        changed,
        target_split="validation",
        lookback_weeks=520,
        max_seq_len=256,
    )

    assert first == replay
    assert first["target_identity_sha256"] != changed_population[
        "target_identity_sha256"
    ]
    assert first["target_quantity_sha256"] == changed_population[
        "target_quantity_sha256"
    ]


def test_real_paired_cpu_e1_uses_train_only_contract_and_raw_selector(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "fixed_split.parquet"
    split = tmp_path / "split.json"
    output = tmp_path / "fresh"
    synthetic_frame().write_parquet(data)
    split.write_text("{}", encoding="utf-8")
    dataset_contract = entry.DATASET_CONTRACTS["intermittent_frozen_5000"]
    monkeypatch.setattr(
        entry,
        "sha256_file",
        lambda path: (
            dataset_contract["data_sha256"]
            if path == data
            else dataset_contract["split_manifest_sha256"]
        ),
    )
    monkeypatch.setattr(
        entry.pl,
        "read_parquet",
        Mock(side_effect=AssertionError("held-out rows must not use eager read_parquet")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--data", str(data),
            "--split-manifest", str(split),
            "--output-dir", str(output),
            "--source-revision", "a" * 40,
            "--execution-role", "test_quantile_alignment_e1",
            "--dataset-contract", "intermittent_frozen_5000",
            "--device", "cpu",
            "--epochs", "1",
            "--min-epochs", "1",
            "--early-stopping-patience", "1",
            "--lookback-weeks", "520",
            "--max-seq-len", "256",
            "--backbones", "titantpp",
            "--seeds", "42",
            "--quantity-variants", "log_mse,quantile_adaptive",
            "--model-role", "quantile_checkpoint_alignment",
            "--checkpoint-monitor", "validation_raw_quantity_rmse",
            "--quantile-adaptive-strength", "1",
            "--time-intercept-limit", "300",
            "--allow-partial-contract",
        ],
    )

    entry.main()

    launch = json.loads((output / "launch_contract.json").read_text())
    assert set(launch["split_rows"]) == {"train", "validation"}
    assert launch["quantile_adaptive_contract"]["population"]["target_count"] == 198
    assert launch["validation_target_population"]["target_count"] == 10
    assert launch["early_stopping"]["monitor"] == "validation_raw_quantity_rmse"
    initial_state_digests = []
    for variant in (
        "count_only_log_regression",
        "count_only_quantile_adaptive_log_regression",
    ):
        run = output / "runs" / "titantpp" / variant / "seed_42"
        summary = json.loads((run / "summary.json").read_text())
        assert summary["checkpoint_monitor"] == "validation_raw_quantity_rmse"
        assert summary["best_epoch"] == 1
        initial_state_digests.append(summary["initial_state_sha256"])
        assert (run / "best_val_qty_rmse_model.pt").is_file()
    assert len(set(initial_state_digests)) == 1
