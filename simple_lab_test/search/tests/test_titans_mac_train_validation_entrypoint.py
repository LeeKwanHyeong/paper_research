"""Exercise actual CLI-to-loader routing without constructing/training a model."""

from __future__ import annotations

import copy
import json
import sys

import numpy as np
import polars as pl
import pytest
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts import titans_mac_prior_prefix_contract as prefix_contract


B1 = "titantpp_titans_mac"
CANDIDATE = "titantpp_titans_mac_prior_prefix"


class ReachedTrainingBoundary(Exception):
    """Stop after real input preparation and launch-contract creation."""


def target_samples(frame, context, split):
    dataset = RMTPPWeekLookbackDataset(
        frame, lookback_weeks=context["lookback"], max_seq_len=context["max_seq_len"],
        val_ratio=0.2, mode="all", split_col="chronological_split", target_splits={split},
    )
    return [dataset[index] for index in range(len(dataset))]


@pytest.mark.parametrize("dataset_id", ["yellow_trip_hourly", "insta_market_basket", "intermittent_frozen_5000"])
@pytest.mark.parametrize("backbones,role", [
    (B1, "experimental"),
    (f"rmtpp,{B1}", "experimental"),
    (CANDIDATE, prefix_contract.ROLE),
])
def test_actual_main_excludes_heldout_before_materialization_and_preserves_targets(
    monkeypatch, tmp_path, dataset_id, backbones, role,
):
    raw = pl.DataFrame({
        "oper_part_no": ["a"] * 24 + ["b"] * 24,
        "seq": list(range(1, 25)) * 2,
        "delta_t": ([1., 2., 3., 4.] * 6) * 2,
        "demand_qty": (list(map(float, range(1, 24))) + [-999999.]) * 2,
        "chronological_split": (["train"] * 20 + ["validation"] * 3 + ["test"]) * 2,
    }).reverse()
    data = tmp_path / "synthetic.parquet"
    manifest = tmp_path / "split.json"
    output = tmp_path / "not_trained"
    raw.write_parquet(data)
    manifest.write_text('{"synthetic": true}')

    # Bind the real CLI guards to this fixture's bytes, rather than bypassing
    # role checks, checksums, candidate launch validation, or the loader branch.
    context = copy.deepcopy(entry.DATASET_CONTRACTS[dataset_id])
    context.update(data_sha256=entry.sha256_file(data), split_manifest_sha256=entry.sha256_file(manifest))
    monkeypatch.setitem(entry.DATASET_CONTRACTS, dataset_id, context)
    frozen = copy.deepcopy(prefix_contract.load_contract())
    frozen["datasets"][dataset_id].update(
        data_sha256=context["data_sha256"], split_manifest_sha256=context["split_manifest_sha256"],
    )
    monkeypatch.setattr(prefix_contract, "load_contract", lambda: frozen)

    allowed_raw = raw.filter(pl.col("chronological_split").is_in(["train", "validation"])).sort(["oper_part_no", "seq"])
    expected_frame = entry.prepare_count_frame(allowed_raw)
    # A normal full-frame reference establishes unchanged legitimate targets and
    # fitted statistics. Only the forbidden rows' synthetic quantity is repaired.
    full_reference = raw.with_columns(
        pl.when(pl.col("chronological_split") == "test")
        .then(pl.lit(999999.)).otherwise(pl.col("demand_qty")).alias("demand_qty"),
    ).sort(["oper_part_no", "seq"])
    reference_frame = entry.prepare_count_frame(full_reference)
    expected_quantity_contract = entry.train_quantile_contract(full_reference)
    expected_time_contract = entry.derive_train_time_contract(
        reference_frame, lookback_weeks=context["lookback"], max_seq_len=context["max_seq_len"],
    )
    reference_samples = {split: target_samples(reference_frame, context, split) for split in ("train", "validation")}
    assert {split: len(values) for split, values in reference_samples.items()} == {"train": 38, "validation": 6}

    monkeypatch.setattr(sys, "argv", [
        "runner", "--data", str(data), "--split-manifest", str(manifest),
        "--output-dir", str(output), "--source-revision", "a" * 40,
        "--execution-role", "local_synthetic_input_contract_only",
        "--dataset-contract", dataset_id, "--model-role", role,
        "--backbones", backbones, "--seeds", "42", "--device", "cpu",
        "--allow-partial-contract", "--epochs", "1", "--min-epochs", "1",
        "--early-stopping-patience", "1", "--lookback-weeks", str(context["lookback"]),
        "--max-seq-len", str(context["max_seq_len"]),
        "--titans-memory-gradient-clip", "1", "--titans-mac-execution-backend", "optimized",
        "--time-intercept-limit", "300",
    ])
    loader_calls = []
    actual_loader = entry.load_train_validation_frame

    def observed_loader(path):
        loaded = actual_loader(path)
        loader_calls.append(path)
        assert loaded.equals(allowed_raw)
        return loaded

    def forbid_eager_read(*args, **kwargs):
        pytest.fail("The actual main entrypoint must not eagerly materialize held-out parquet rows")

    monkeypatch.setattr(entry, "load_train_validation_frame", observed_loader)
    monkeypatch.setattr(entry.pl, "read_parquet", forbid_eager_read)
    training_calls = []

    def stop_before_training(**kwargs):
        training_calls.append(kwargs["backbone"])
        frame = kwargs["frame"]
        assert frame.equals(expected_frame)
        assert set(frame["chronological_split"].unique().to_list()) == {"train", "validation"}
        assert frame["demand_qty"].min() > 0
        assert kwargs["quantity_contract"] == expected_quantity_contract
        interface = kwargs["interface_meta"]
        assert interface["time_head"]["train_time_statistics"] == expected_time_contract
        train_log = np.log1p(allowed_raw.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy())
        assert interface["train_target_mean"] == float(train_log.mean())
        assert interface["train_target_std"] == float(train_log.std())
        for split, expected in reference_samples.items():
            actual = target_samples(frame, context, split)
            assert len(actual) == len(expected)
            for observed, reference in zip(actual, expected, strict=True):
                assert observed.keys() == reference.keys()
                for name in observed:
                    assert torch.equal(observed[name], reference[name]), (split, name)
        raise ReachedTrainingBoundary

    monkeypatch.setattr(entry, "train_one", stop_before_training)
    with pytest.raises(ReachedTrainingBoundary):
        entry.main()
    assert loader_calls == [data]
    assert training_calls == [backbones.split(",")[0]]
    launch = json.loads((output / "launch_contract.json").read_text())
    assert launch["split_rows"] == {"train": 40, "validation": 6}
    assert launch["quantity_contract"] == expected_quantity_contract
    assert launch["time_head"]["train_time_statistics"] == expected_time_contract
    assert launch["time_head"]["train_time_statistics"]["target_count"] == 38
    assert launch["evaluation_scope"] == "validation_only"
    assert launch["held_out_test_evaluated"] is False
    assert not list(output.rglob("*.pt"))
