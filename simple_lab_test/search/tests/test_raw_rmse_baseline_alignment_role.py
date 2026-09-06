"""Dedicated validation-only role contracts for selector-aligned baselines."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from unittest.mock import Mock

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareTPP import (
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TIME_HEAD_MODE_LOGNORMAL_DURATION,
)
from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts.count_aware_tpp_backbone.constants import (
    MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT,
    TAIL_SHARED_VARIANT,
    VARIANT,
    validate_model_role_contract,
)
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.run_count_aware_tpp_backbone_control import exact_target_population


def _validate(**overrides) -> None:
    values = {
        "model_role": MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT,
        "backbones": ("rmtpp",),
        "quantity_variants": (VARIANT,),
        "time_head_mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
        "lambda_tail": 0.0,
    }
    values.update(overrides)
    validate_model_role_contract(**values)


def _synthetic_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for part_index, part in enumerate(("a", "b")):
        for seq in range(1, 111):
            split = "train" if seq <= 100 else "validation" if seq <= 105 else "test"
            quantity = float(seq + 100 * part_index)
            if split == "test":
                quantity += 1_000_000.0
            rows.append({
                "oper_part_no": part,
                "seq": seq,
                "delta_t": float((seq % 7) + 1),
                "demand_qty": quantity,
                "chronological_split": split,
            })
    return pl.DataFrame(rows)


def test_role_accepts_only_rmtpp_thp_direct_log_mse_and_legacy_time() -> None:
    _validate(backbones=("rmtpp", "thp"))
    for invalid in (("titantpp",), ("rmtpp", "titantpp"), ()):
        with pytest.raises(ValueError, match="RMTPP and THP"):
            _validate(backbones=invalid)
    with pytest.raises(ValueError, match="direct log-MSE"):
        _validate(quantity_variants=(TAIL_SHARED_VARIANT,))
    with pytest.raises(ValueError, match="legacy_clamped_rmtpp"):
        _validate(time_head_mode=TIME_HEAD_MODE_LOGNORMAL_DURATION)


def test_cpu_e1_keeps_held_out_unmaterialized_and_records_target_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = tmp_path / "fixed_split.parquet"
    split = tmp_path / "split.json"
    output = tmp_path / "output"
    source = _synthetic_frame()
    source.write_parquet(data)
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
        Mock(side_effect=AssertionError("dedicated role must not eagerly materialize held-out rows")),
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
            "--execution-role", "test_raw_rmse_baseline_alignment_e1",
            "--dataset-contract", "intermittent_frozen_5000",
            "--device", "cpu",
            "--epochs", "1",
            "--min-epochs", "1",
            "--early-stopping-patience", "1",
            "--lookback-weeks", "520",
            "--max-seq-len", "256",
            "--backbones", "rmtpp",
            "--seeds", "42",
            "--quantity-variants", "log_mse",
            "--model-role", MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT,
            "--checkpoint-monitor", "validation_raw_quantity_rmse",
            "--quantile-adaptive-strength", "0",
            "--time-intercept-limit", "300",
            "--allow-partial-contract",
        ],
    )

    entry.main()

    launch = json.loads((output / "launch_contract.json").read_text(encoding="utf-8"))
    prepared = prepare_count_frame(source.filter(pl.col("chronological_split") != "test"))
    _, expected = exact_target_population(
        prepared,
        target_split="validation",
        lookback_weeks=520,
        max_seq_len=256,
    )
    assert set(launch["split_rows"]) == {"train", "validation"}
    assert launch["validation_target_population"] == expected
    assert launch["quantile_adaptive_contract"] is None
    assert launch["early_stopping"]["monitor"] == "validation_raw_quantity_rmse"
    assert launch["held_out_test_evaluated"] is False
    summary = json.loads(
        (
            output
            / "runs/rmtpp/count_only_log_regression/seed_42/summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["checkpoint_monitor"] == "validation_raw_quantity_rmse"
    assert summary["held_out_test_evaluated"] is False


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    (
        ("--checkpoint-monitor", "validation_joint_objective", "raw-RMSE checkpoint"),
        ("--quantile-adaptive-strength", "1", "strength=0"),
        ("--time-intercept-limit", "30", "cap=300"),
    ),
)
def test_entrypoint_rejects_role_runtime_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    value: str,
    message: str,
) -> None:
    data = tmp_path / "data.parquet"
    split = tmp_path / "split.json"
    _synthetic_frame().write_parquet(data)
    split.write_text("{}", encoding="utf-8")
    dataset_contract = entry.DATASET_CONTRACTS["intermittent_frozen_5000"]
    monkeypatch.setattr(
        entry,
        "sha256_file",
        lambda path: dataset_contract[
            "data_sha256" if path == data else "split_manifest_sha256"
        ],
    )
    arguments = [
        "test",
        "--data", str(data),
        "--split-manifest", str(split),
        "--output-dir", str(tmp_path / "output"),
        "--source-revision", "a" * 40,
        "--execution-role", "test",
        "--dataset-contract", "intermittent_frozen_5000",
        "--device", "cpu",
        "--epochs", "1",
        "--min-epochs", "1",
        "--early-stopping-patience", "1",
        "--lookback-weeks", "520",
        "--max-seq-len", "256",
        "--backbones", "rmtpp",
        "--seeds", "42",
        "--quantity-variants", "log_mse",
        "--model-role", MODEL_ROLE_RAW_RMSE_BASELINE_ALIGNMENT,
        "--checkpoint-monitor", "validation_raw_quantity_rmse",
        "--quantile-adaptive-strength", "0",
        "--time-intercept-limit", "300",
        "--allow-partial-contract",
    ]
    index = arguments.index(flag)
    arguments[index + 1] = value
    monkeypatch.setattr(sys, "argv", arguments)
    with pytest.raises(ValueError, match=message):
        entry.main()
