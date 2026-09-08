"""Exercise the LPHC trainer on a tiny synthetic split, never benchmark data."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import polars as pl
import numpy as np
import pytest
import torch

from models.TPPs.CountAwareFactory import validate_checkpoint_route
from models.TPPs.CountAwareTitanLevelHistoryQKV import (
    LEVEL_HISTORY_QKV_BACKBONE, LEVEL_HISTORY_QKV_CONTRACT_ID,
    LEVEL_HISTORY_QKV_ROLE,
)
from paper.scripts import run_count_aware_tpp_backbone_control as trainer
from paper.scripts.count_aware_tpp_backbone import training
from paper.scripts.run_hard_lmm_backbone_candidate_campaign import DATASETS, job_command
from simple_lab_test.search.common.runner import torch_load_checkpoint


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def arguments(monkeypatch, output: Path, *, dataset="yellow_trip_hourly", epochs=2):
    command = job_command(
        python=sys.executable,
        candidate={"backbone": LEVEL_HISTORY_QKV_BACKBONE, "model_role": LEVEL_HISTORY_QKV_ROLE},
        dataset=dataset, output=output, source_revision="a" * 40,
        phase={"epochs": epochs, "minimum_epochs": 1, "patience": 40},
        host_role="local_synthetic_test",
    )
    monkeypatch.setattr(sys, "argv", command[2:])
    args = trainer.parse_args()
    args.device = "cpu"
    return args


@pytest.mark.parametrize("dataset", list(DATASETS))
@pytest.mark.parametrize("epochs", [1, 300])
def test_trainer_accepts_level_history_identity_and_fixed_selector(monkeypatch, tmp_path, dataset, epochs):
    args = arguments(monkeypatch, tmp_path, dataset=dataset, epochs=epochs)
    trainer.validate_model_role_contract(
        model_role=args.model_role, backbones=(args.backbones,),
        quantity_variants=trainer.normalize_quantity_variants(args.quantity_variants),
        time_head_mode=args.time_head_mode, lambda_tail=args.lambda_tail,
    )
    assert args.checkpoint_monitor == "validation_raw_quantity_rmse"
    assert args.max_train_batches is args.max_val_batches is args.max_series is None
    assert args.time_intercept_limit == 300
    assert args.quantile_adaptive_strength == 0


@pytest.mark.parametrize("attribute,value,error", [
    ("checkpoint_monitor", "validation_joint_objective", "raw-RMSE checkpoint"),
    ("quantile_adaptive_strength", 1.0, "adaptive strength=0"),
    ("time_intercept_limit", 30.0, "cap=300"),
    ("time_head_mode", "heteroscedastic_lognormal_duration", "dedicated runner"),
])
def test_wrong_training_contract_fails_before_data_access(monkeypatch, tmp_path, attribute, value, error):
    args = arguments(monkeypatch, tmp_path)
    setattr(args, attribute, value)
    monkeypatch.setattr(trainer, "load_train_validation_frame",
                        lambda _: pytest.fail("Data must not be loaded after contract drift"))
    with pytest.raises(ValueError, match=error):
        trainer.run(args)
    assert not (tmp_path / "runs").exists()


def synthetic_inputs(monkeypatch, tmp_path):
    rows = []
    for series in range(12):
        for seq in range(8):
            split = "train" if seq < 5 else "validation" if seq < 7 else "test"
            rows.append({"oper_part_no": f"synthetic_{series}", "seq": seq,
                         "delta_t": float(1 + seq % 3),
                         "demand_qty": float(1 + (series * 7 + seq * 3) % 61),
                         "chronological_split": split})
    path = tmp_path / "synthetic.parquet"
    pl.DataFrame(rows).write_parquet(path)
    manifest = tmp_path / "synthetic_split.json"
    manifest.write_text('{"synthetic_only": true}\n')
    spec = dict(trainer.DATASET_CONTRACTS["yellow_trip_hourly"])
    spec.update(data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                split_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    monkeypatch.setitem(trainer.DATASET_CONTRACTS, "yellow_trip_hourly", spec)
    return path, manifest


def assert_state_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_state_tree_equal(first, second)
    else:
        assert left == right


def test_real_level_history_training_interruption_resume_and_selected_restore(monkeypatch, tmp_path):
    data, manifest = synthetic_inputs(monkeypatch, tmp_path)
    base_args = arguments(monkeypatch, tmp_path / "continuous")
    base_args.data, base_args.split_manifest = data, manifest
    base_args.batch_size = 16
    admitted = trainer.load_train_validation_frame
    admitted_counts = []

    def record_admitted(path):
        frame = admitted(path)
        assert "test" not in frame["chronological_split"].unique().to_list()
        admitted_counts.append(frame.height)
        return frame

    monkeypatch.setattr(trainer, "load_train_validation_frame", record_admitted)
    trainer.run(base_args)
    resume_args = copy.copy(base_args)
    resume_args.output_dir = tmp_path / "interrupted"
    original_train_epoch = training.train_epoch_with_telemetry
    call_count = 0

    def interrupt_second_epoch(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("synthetic interruption after one completed epoch")
        return original_train_epoch(**kwargs)

    monkeypatch.setattr(training, "train_epoch_with_telemetry", interrupt_second_epoch)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        trainer.run(resume_args)
    monkeypatch.setattr(training, "train_epoch_with_telemetry", original_train_epoch)
    trainer.run(resume_args)
    run_suffix = Path("runs") / LEVEL_HISTORY_QKV_BACKBONE / "count_only_log_regression" / "seed_42"
    left = base_args.output_dir / run_suffix
    right = resume_args.output_dir / run_suffix
    selected = "best_val_qty_rmse_model.pt"
    continuous = torch_load_checkpoint(left / selected, map_location="cpu")
    resumed = torch_load_checkpoint(right / selected, map_location="cpu")
    validate_checkpoint_route(resumed, LEVEL_HISTORY_QKV_BACKBONE)
    assert continuous["best_epoch"] == resumed["best_epoch"]
    assert continuous["model_state_sha256"] == resumed["model_state_sha256"]
    for key, value in continuous["model_state_dict"].items():
        assert torch.equal(value, resumed["model_state_dict"][key]), key
    # A selected epoch1 alone could conceal an incorrect resumed epoch2.
    # Compare the final state and optimizer as well as the selected checkpoint.
    last_left = torch_load_checkpoint(left / "last_epoch_state.pt", map_location="cpu")
    last_right = torch_load_checkpoint(right / "last_epoch_state.pt", map_location="cpu")
    for key in (
        "model_state_dict",
        "optimizer_state_dict",
        "history",
        "best_state_dict",
        "best_state_sha256",
        "best_epoch",
        "best_selection_value",
        "rng_state",
        "train_loader_generator_state",
    ):
        assert_state_tree_equal(last_left[key], last_right[key])
    for folder in (left, right):
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["checkpoint_monitor"] == "validation_raw_quantity_rmse"
        assert summary["completed_epochs"] == 2
        assert summary["held_out_test_evaluated"] is False
        assert summary["encoder_config"]["backbone_contract_id"] == LEVEL_HISTORY_QKV_CONTRACT_ID
    assert admitted_counts == [84, 84, 84]
