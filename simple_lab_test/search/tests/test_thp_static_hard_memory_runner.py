"""Candidate scope, dataset isolation and local synthetic runner integration."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import numpy as np
import polars as pl
import pytest
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from models.TPPs.CountAwareFactory import validate_checkpoint_route
from models.TPPs.CountAwareTHPStaticMemory import (
    THP_STATIC_MEMORY_BACKBONE as CANDIDATE,
    THP_STATIC_MEMORY_ROLE as ROLE,
)
from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts.count_aware_tpp_backbone import thp_static_contract as guard
from paper.scripts.count_aware_tpp_backbone.core import load_train_validation_frame, prepare_count_frame
from paper.scripts.count_aware_tpp_backbone.training import train_one


ROOT = Path(__file__).resolve().parents[3]


def parsed_args(monkeypatch, tmp_path, dataset="yellow_trip_hourly"):
    context = entry.DATASET_CONTRACTS[dataset]
    monkeypatch.setattr(sys, "argv", ["test", "--data", str(tmp_path / "data.parquet"),
        "--split-manifest", str(tmp_path / "split.json"), "--output-dir", str(tmp_path / "fresh"),
        "--source-revision", "a" * 40, "--execution-role", "local_synthetic_contract_test",
        "--dataset-contract", dataset, "--model-role", ROLE, "--backbones", CANDIDATE,
        "--seeds", "42", "--device", "cpu", "--allow-partial-contract",
        "--lookback-weeks", str(context["lookback"]), "--max-seq-len", str(context["max_seq_len"])])
    return entry.parse_args()


@pytest.mark.parametrize("dataset", ["yellow_trip_hourly", "raf_spare_parts"])
@pytest.mark.parametrize("budget", [(1, 1, 1), (300, 40, 40)])
def test_cli_accepts_only_separate_full_e1_or_frozen_screening(monkeypatch, tmp_path, dataset, budget):
    args = parsed_args(monkeypatch, tmp_path, dataset)
    args.epochs, args.min_epochs, args.early_stopping_patience = budget
    row = guard.validate_static_memory_launch(args, (42,))
    assert row["dataset"] == dataset and args.model_role == ROLE and args.backbones == CANDIDATE
    assert not args.output_dir.exists()


@pytest.mark.parametrize("change", [{"dataset_contract": "insta_market_basket"}, {"lr": .002},
    {"batch_size": 64}, {"hidden_dim": 32}, {"max_series": 20}, {"max_train_batches": 1},
    {"max_val_batches": 1}, {"grad_clip": .5}, {"epochs": 100}, {"min_epochs": 1},
    {"early_stopping_patience": 10}, {"force_rerun": True}, {"lambda_tail": .01},
    {"lambda_log_qty": .5}, {"time_scale": 1.}, {"time_w_max": .347826},
    {"time_head_lr_multiplier": .5}, {"max_seq_len": 64}, {"lookback_weeks": 52},
    {"titans_memory_gradient_clip": 1.}])
def test_launch_drift_is_rejected_before_data_or_output(monkeypatch, tmp_path, change):
    args = parsed_args(monkeypatch, tmp_path)
    vars(args).update(change)
    with pytest.raises(ValueError):
        guard.validate_static_memory_launch(args, (42,))
    assert not args.output_dir.exists()


def test_unapproved_seeds_existing_directory_and_tampered_contract_are_rejected(monkeypatch, tmp_path):
    args = parsed_args(monkeypatch, tmp_path)
    for seeds in ((52,), (42, 62)):
        with pytest.raises(ValueError, match="seed42"):
            guard.validate_static_memory_launch(args, seeds)
    args.output_dir.mkdir()
    sentinel = args.output_dir / "sentinel"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        guard.validate_static_memory_launch(args, (42,))
    assert sentinel.read_text() == "keep"
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    path = tmp_path / "paper/contracts" / f"{guard.THP_STATIC_MEMORY_CONTRACT}.json"
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(ValueError, match="digest"):
        guard.frozen_documents()


@pytest.mark.parametrize("force", [False, True])
def test_low_level_training_never_reuses_or_overwrites_candidate_cache(tmp_path, force, monkeypatch):
    run = tmp_path / "runs" / CANDIDATE / entry.VARIANT / "seed_42"
    run.mkdir(parents=True)
    sentinel = run / "last_epoch_state.pt"
    sentinel.write_bytes(b"old checkpoint, never touch")
    model_build = Mock(side_effect=AssertionError("Must not construct a model"))
    monkeypatch.setattr("paper.scripts.count_aware_tpp_backbone.training.build_model", model_build)
    with pytest.raises(FileExistsError, match="fresh"):
        train_one(args=argparse.Namespace(output_dir=tmp_path, force_rerun=force), frame=None,
                  quantity_contract={}, interface_meta={}, backbone=CANDIDATE,
                  quantity_variant=entry.VARIANT, seed=42)
    model_build.assert_not_called()
    assert sentinel.read_bytes() == b"old checkpoint, never touch"


def history_contract():
    return {"boundaries": list(entry.HISTORY_BOUNDARIES), "strata": list(entry.HISTORY_STRATA),
            "definition": "number of observed events before the validation target"}


def synthetic_frame():
    return pl.DataFrame({
        "oper_part_no": ["a"] * 5 + ["b"] * 5,
        "seq": [1, 2, 3, 4, 5] * 2,
        "delta_t": [1, 2, 1, 2, 999999] * 2,
        "demand_qty": [1., 3., 5., 7., -999999.] * 2,
        "chronological_split": ["train", "train", "validation", "validation", "test"] * 2,
    })


@pytest.mark.parametrize("failure", [None, "training", "reporting"])
def test_actual_entrypoint_filters_heldout_and_persists_identity_or_failed_status(monkeypatch, tmp_path, failure):
    args = parsed_args(monkeypatch, tmp_path)
    args.epochs = args.min_epochs = args.early_stopping_patience = 1
    pl.DataFrame({
        "oper_part_no": ["a"] * 24 + ["b"] * 24,
        "seq": list(range(1, 25)) * 2,
        "delta_t": [1.] * 48,
        "demand_qty": (list(map(float, range(1, 24))) + [-999999.]) * 2,
        "chronological_split": (["train"] * 20 + ["validation"] * 3 + ["test"]) * 2,
    }).write_parquet(args.data)
    args.split_manifest.write_text('{}')
    frame = load_train_validation_frame(args.data)
    row = dict(dataset="yellow_trip_hourly", lookback=168, max_seq_len=256,
               train_targets=38, validation_targets=6, quantity_contract=entry.train_quantile_contract(frame),
               history_length_contract=history_contract(),
               **{key: entry.DATASET_CONTRACTS["yellow_trip_hourly"][key]
                  for key in ("data_sha256", "split_manifest_sha256")})
    monkeypatch.setattr(entry, "parse_args", lambda: args)
    monkeypatch.setattr(entry, "validate_static_memory_launch", lambda *_: row)
    monkeypatch.setattr(entry, "sha256_file", lambda path: row[
        "data_sha256" if path == args.data else "split_manifest_sha256"])
    # A test-only negative quantity would fail before training if eagerly loaded.
    monkeypatch.setattr(entry.pl, "read_parquet", Mock(side_effect=AssertionError("No eager read")))
    if failure == "training":
        monkeypatch.setattr(entry, "train_one", Mock(side_effect=RuntimeError("synthetic training failure")))
    if failure == "reporting":
        monkeypatch.setattr(entry, "write_csv", Mock(side_effect=RuntimeError("synthetic reporting failure")))
    if failure:
        with pytest.raises(RuntimeError, match=f"synthetic {failure} failure"):
            entry.main()
    else:
        entry.main()
    launch = json.loads((args.output_dir / "launch_contract.json").read_text())
    assert launch["status"] == ("failed" if failure else "complete")
    assert launch["validated_target_counts"] == {"train": 38, "validation": 6}
    assert launch["routing_contract_id"] == guard.THP_STATIC_MEMORY_CONTRACT
    assert set(launch["split_rows"]) == {"train", "validation"}
    assert not list(args.output_dir.rglob("test_*"))
    assert not list(args.output_dir.rglob("*.tmp"))
    if failure:
        assert failure in launch["error"]
    else:
        run = args.output_dir / "runs" / CANDIDATE / entry.VARIANT / "seed_42"
        summary = json.loads((run / "summary.json").read_text())
        validate_checkpoint_route(summary, CANDIDATE)
        assert summary["completed_epochs"] == summary["best_epoch"] == 1
        for name in ("best_val_joint_objective_model.pt", "last_epoch_state.pt"):
            checkpoint = torch.load(run / name, map_location="cpu", weights_only=False)
            validate_checkpoint_route(checkpoint, CANDIDATE)
        assert sum(r["count"] for r in summary["quantity_rows"]) == 6


def test_failure_context_does_not_relabel_preexisting_artifacts(tmp_path):
    launch = tmp_path / "launch_contract.json"
    launch.write_text('{"status":"complete"}')
    original = launch.read_bytes()
    with pytest.raises(RuntimeError):
        with guard.record_static_memory_failure(tmp_path):
            raise RuntimeError("rejected before launch")
    assert launch.read_bytes() == original


def test_failure_context_does_not_claim_a_racing_foreign_output(tmp_path):
    output = tmp_path / "foreign"
    with pytest.raises(FileExistsError):
        with guard.record_static_memory_failure(output):
            output.mkdir()
            (output / "launch_contract.json").write_text('{"status":"running"}')
            raise FileExistsError("A different process created this output")
    assert json.loads((output / "launch_contract.json").read_text())["status"] == "running"


@pytest.mark.parametrize("field", ["data_sha256", "quantity_contract", "history_length_contract",
                                   "evaluation_scope", "held_out_test_evaluated", "split_rows", "time_head"])
def test_materialized_contract_rejects_wrong_scope_identity_counts_and_cutoffs(field):
    frame = prepare_count_frame(synthetic_frame().filter(pl.col("chronological_split") != "test"))
    row = dict(lookback=168, max_seq_len=256, train_targets=2, validation_targets=4,
               data_sha256="data", split_manifest_sha256="split",
               quantity_contract={}, history_length_contract={})
    launch = {key: row[key] for key in ("data_sha256", "split_manifest_sha256", "quantity_contract", "history_length_contract")}
    launch.update(evaluation_scope="validation_only", held_out_test_evaluated=False,
                  split_rows={"train": 4, "validation": 4}, time_head={"train_time_statistics": {"target_count": 2}})
    if field == "time_head":
        launch[field]["train_time_statistics"]["target_count"] = 999
    elif field == "split_rows":
        launch[field]["test"] = 2
    else:
        launch[field] = "wrong"
    with pytest.raises(ValueError):
        guard.validate_static_memory_data(launch, frame, row)


@pytest.mark.parametrize("dataset_name", ["yellow_trip_hourly", "raf_spare_parts"])
def test_real_filtered_data_counts_cutoffs_and_validation_strata_match_pinned_baselines(dataset_name):
    # No model is constructed or evaluated: inspect train/validation loader indices only.
    _, registry = guard.frozen_documents()
    row = next(row for row in registry["datasets"] if row["dataset"] == dataset_name)
    path = ROOT / row["data_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["data_sha256"]
    frame = prepare_count_frame(load_train_validation_frame(path))
    launch = dict(data_sha256=row["data_sha256"], split_manifest_sha256=row["split_manifest_sha256"],
        quantity_contract=entry.train_quantile_contract(frame), history_length_contract=history_contract(),
        evaluation_scope="validation_only", held_out_test_evaluated=False,
        split_rows={name: frame.filter(pl.col("chronological_split") == name).height for name in ("train", "validation")},
        time_head={"train_time_statistics": entry.derive_train_time_contract(
            frame, lookback_weeks=row["lookback"], max_seq_len=row["max_seq_len"])})
    assert guard.validate_static_memory_data(launch, frame, row)["validated_target_counts"] == {
        "train": row["train_targets"], "validation": row["validation_targets"]}
    dataset = RMTPPWeekLookbackDataset(frame, lookback_weeks=row["lookback"], max_seq_len=row["max_seq_len"],
                                      mode="all", split_col="chronological_split", target_splits={"validation"})
    quantity_counts = np.zeros(5, dtype=int)
    history_counts = np.zeros(3, dtype=int)
    for part, end in dataset.index:
        quantity = dataset.val_lists[part][end + 1]
        quantity_counts[np.searchsorted(row["quantity_contract"]["boundaries"], quantity, side="left")] += 1
        seq = dataset.seq_lists[part]
        left = np.searchsorted(seq, seq[end] - (row["lookback"] - 1), side="left")
        length = min(end - left + 1, row["max_seq_len"] - 1)
        history_counts[np.searchsorted(entry.HISTORY_BOUNDARIES, length, side="left")] += 1
    for model in ("thp", "titantpp"):
        summary = json.loads((ROOT / row["artifact_dir"] / "runs" / model / entry.VARIANT / "seed_42/summary.json").read_text())
        for counts, strata, key in (
            (quantity_counts, row["quantity_contract"]["strata"], "quantity_rows"),
            (history_counts, entry.HISTORY_STRATA, "history_rows"),
        ):
            # The historical reporter omits empty bins; do not drop positive bins.
            expected = {s["stratum"]: int(n) for s, n in zip(strata, counts) if n > 0}
            assert expected == {r["stratum"]: r["count"] for r in summary[key]}
