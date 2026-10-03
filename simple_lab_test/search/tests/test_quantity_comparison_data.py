import json
import math

import numpy as np
import polars as pl
import pytest

from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.quantity_comparison_data import (
    prepare_comparison_contract,
    prepare_dataset_statistics,
    prepare_quantity_comparison_data,
    sha256_file,
)
from paper.scripts.run_count_aware_tpp_backbone_control import exact_target_population


def _write_contract(tmp_path, frame: pl.DataFrame):
    tmp_path.mkdir(parents=True, exist_ok=True)
    data = tmp_path / "events.parquet"
    manifest = tmp_path / "split_manifest.json"
    frame.write_parquet(data)
    manifest.write_text('{"split":"fixed"}\n', encoding="utf-8")
    admitted = frame.filter(pl.col("chronological_split").is_in(["train", "validation"])).sort(["oper_part_no", "seq"])
    prepared = prepare_count_frame(admitted)
    populations = {}
    for split in ("train", "validation"):
        _, population = exact_target_population(prepared, target_split=split, lookback_weeks=10, max_seq_len=8)
        populations[split] = {key: population[key] for key in ("target_count", "target_identity_sha256", "target_quantity_sha256")}
    train = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy().astype(np.float64)
    dataset = {
        "dataset_id": "synthetic",
        "inherited_data_identity": {
            "data": {"path": str(data), "sha256": sha256_file(data)},
            "split_manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
            "populations": populations,
        },
        "loader": {"lookback_weeks": 10, "max_seq_len": 8},
        "mu_all_train_rows": float(np.log1p(train).mean()),
    }
    contract = {"schema": "quantity_objective_output_comparison_v1", "datasets": [dataset]}
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path, dataset


def _frame(*, validation_quantity=8.0, include_test=True):
    rows = [
        ("a", 0, 1.0, 0.0, "train"),
        ("a", 1, 1.0, 2.0, "train"),
        ("a", 2, 1.0, 6.0, "train"),
        ("a", 3, 1.0, validation_quantity, "validation"),
    ]
    if include_test:
        rows.append(("a", 4, 1.0, 999.0, "test"))
    return pl.DataFrame(rows, schema=["oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split"], orient="row")


def test_train_statistic_uses_all_train_rows_but_raw_scale_uses_canonical_targets(tmp_path):
    _, dataset = _write_contract(tmp_path, _frame())
    frame, metadata = prepare_quantity_comparison_data(dataset)
    result = prepare_dataset_statistics(dataset)
    # all train rows are [0, 2, 6]; canonical train targets are [2, 6].
    assert result["all_train_rows"]["count"] == 3
    assert result["all_train_rows"]["log1p_mean"] == pytest.approx(np.log1p([0, 2, 6]).mean())
    assert result["canonical_train_targets"]["target_count"] == 2
    assert result["canonical_train_targets"]["raw_rms"] == pytest.approx(math.sqrt((2**2 + 6**2) / 2))
    assert result["canonical_train_targets"]["raw_scale"] == result["canonical_train_targets"]["raw_rms"]
    assert result["held_out_materialized"] is False
    assert set(frame["chronological_split"].unique()) == {"train", "validation"}
    assert metadata["train_log_mean"] == result["all_train_rows"]["log1p_mean"]
    assert metadata["populations"]["train"]["target_count"] == 2
    assert metadata["held_out_materialized"] is False


def test_held_out_and_validation_values_do_not_change_train_statistics(tmp_path):
    first_path, first_dataset = _write_contract(tmp_path / "first", _frame(validation_quantity=8.0, include_test=True))
    second_path, second_dataset = _write_contract(tmp_path / "second", _frame(validation_quantity=123.0, include_test=False))
    first = prepare_comparison_contract(first_path)["datasets"][0]
    second = prepare_comparison_contract(second_path)["datasets"][0]
    assert first["all_train_rows"] == second["all_train_rows"]
    assert first["canonical_train_targets"] == second["canonical_train_targets"]
    assert first_dataset["inherited_data_identity"]["populations"]["validation"] != second_dataset["inherited_data_identity"]["populations"]["validation"]


def test_rejects_input_hash_schema_and_target_population_drift(tmp_path):
    path, dataset = _write_contract(tmp_path, _frame())
    bad_hash = json.loads(path.read_text())
    bad_hash["datasets"][0]["inherited_data_identity"]["data"]["sha256"] = "0" * 64
    path.write_text(json.dumps(bad_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="digest mismatch"):
        prepare_comparison_contract(path)

    path, dataset = _write_contract(tmp_path / "count", _frame())
    dataset["inherited_data_identity"]["populations"]["train"]["target_count"] += 1
    with pytest.raises(ValueError, match="target population mismatch"):
        prepare_dataset_statistics(dataset)

    path, dataset = _write_contract(tmp_path / "schema", _frame())
    malformed = pl.DataFrame({"oper_part_no": ["x"], "seq": [0], "demand_qty": [1.0], "chronological_split": ["train"]})
    data = path.parent / "events.parquet"
    malformed.write_parquet(data)
    dataset["inherited_data_identity"]["data"]["sha256"] = sha256_file(data)
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_dataset_statistics(dataset)


def test_rejects_validation_before_train_and_negative_duration(tmp_path):
    path, dataset = _write_contract(tmp_path / "chronology", _frame())
    data = path.parent / "events.parquet"
    out_of_order = _frame().with_columns(
        pl.when(pl.col("seq") == 2)
        .then(pl.lit("validation"))
        .when(pl.col("seq") == 3)
        .then(pl.lit("train"))
        .otherwise(pl.col("chronological_split"))
        .alias("chronological_split")
    )
    out_of_order.write_parquet(data)
    dataset["inherited_data_identity"]["data"]["sha256"] = sha256_file(data)
    with pytest.raises(ValueError, match="Validation precedes a training target"):
        prepare_dataset_statistics(dataset)

    path, dataset = _write_contract(tmp_path / "negative_dt", _frame())
    data = path.parent / "events.parquet"
    negative_dt = _frame().with_columns(
        pl.when(pl.col("seq") == 1).then(pl.lit(-1.0)).otherwise(pl.col("delta_t")).alias("delta_t")
    )
    negative_dt.write_parquet(data)
    dataset["inherited_data_identity"]["data"]["sha256"] = sha256_file(data)
    with pytest.raises(ValueError, match="Negative delta_t"):
        prepare_dataset_statistics(dataset)
