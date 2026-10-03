"""Exercise the real candidate training/checkpoint path with synthetic CPU rows."""

from __future__ import annotations

import copy
import json
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.TPPs.CountAwareTitanSlotMemory import SLOT_MEMORY_BACKBONE, SLOT_MEMORY_ROLE
from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT
from paper.scripts import run_count_aware_tpp_backbone_control as cli
from paper.scripts.count_aware_tpp_backbone import training
from paper.scripts.count_aware_tpp_backbone.constants import validate_model_role_contract
from simple_lab_test.search.common.runner import torch_load_checkpoint


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _arguments(monkeypatch, root, output):
    monkeypatch.setattr(sys, "argv", [
        "runner", "--data", str(root / "synthetic_unmaterialized.parquet"),
        "--split-manifest", str(root / "synthetic_unmaterialized.json"),
        "--output-dir", str(output), "--source-revision", "1" * 40,
        "--execution-role", "local_synthetic_contract_test", "--device", "cpu",
        "--backbones", SLOT_MEMORY_BACKBONE, "--model-role", SLOT_MEMORY_ROLE,
        "--quantity-variants", LOG_MSE_VARIANT, "--lambda-tail", "0",
        "--checkpoint-monitor", "validation_raw_quantity_rmse",
        "--time-intercept-limit", "300", "--allow-partial-contract",
        "--epochs", "3", "--min-epochs", "3", "--early-stopping-patience", "3",
        "--batch-size", "4", "--max-seq-len", "8",
    ])
    return cli.parse_args()


def _loader(_frame, *, target_split, batch_size, shuffle, generator, **_):
    assert target_split in {"train", "validation"}
    rng = torch.Generator().manual_seed(77 if target_split == "train" else 88)
    dts = 0.1 + 2 * torch.rand(12, 7, generator=rng)
    quantities = torch.randint(0, 25, (12, 7), generator=rng).float()
    mask = torch.arange(7)[None, :] < (torch.arange(12) % 6 + 2)[:, None]
    dts[~mask] = 0
    quantities[~mask] = 0
    dataset = TensorDataset(torch.zeros_like(dts), dts, mask, torch.zeros_like(dts), quantities)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def _run(args):
    return training.train_one(
        args=args, frame=None, quantity_contract={"boundaries": [], "strata": [{"label": "synthetic_all"}]},
        interface_meta={
            "train_target_mean": 1.5, "train_target_std": 1.0,
            "time_head": {"time_initial_intercept": 0.0},
            "data_scope": "synthetic_cpu_only",
        },
        backbone=SLOT_MEMORY_BACKBONE, quantity_variant=LOG_MSE_VARIANT, seed=42,
    )


def _assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_equal(a, b)
    else:
        assert left == right


def test_real_training_selector_replay_cache_and_interrupted_resume(monkeypatch, tmp_path):
    # Only the data source is substituted: actual model, loss, optimizer,
    # evaluator, selector, checkpoint writer and resume validator all execute.
    monkeypatch.setattr(training, "make_loader", _loader)
    full_args = _arguments(monkeypatch, tmp_path, tmp_path / "full")
    full, _, _ = _run(full_args)
    resumed_args = _arguments(monkeypatch, tmp_path, tmp_path / "resumed")
    save = training.atomic_torch_save

    def save_then_interrupt(payload, path):
        save(payload, path)
        if path.name == "last_epoch_state.pt" and payload["epoch"] == 1:
            raise RuntimeError("synthetic interruption after committed epoch")

    monkeypatch.setattr(training, "atomic_torch_save", save_then_interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run(resumed_args)
    monkeypatch.setattr(training, "atomic_torch_save", save)
    resumed, _, _ = _run(resumed_args)
    assert full["completed_epochs"] == resumed["completed_epochs"] == 3
    assert full["checkpoint_state_sha256"] == resumed["checkpoint_state_sha256"]
    assert full["best_epoch"] == resumed["best_epoch"]
    full_dir = full_args.output_dir / "runs" / SLOT_MEMORY_BACKBONE / LOG_MSE_VARIANT / "seed_42"
    resumed_dir = resumed_args.output_dir / "runs" / SLOT_MEMORY_BACKBONE / LOG_MSE_VARIANT / "seed_42"
    full_last = torch_load_checkpoint(full_dir / "last_epoch_state.pt", map_location="cpu")
    resumed_last = torch_load_checkpoint(resumed_dir / "last_epoch_state.pt", map_location="cpu")
    for key in ("model_state_dict", "best_state_dict", "optimizer_state_dict", "train_loader_generator_state"):
        _assert_equal(full_last[key], resumed_last[key])
    assert torch.equal(full_last["rng_state"]["torch"], resumed_last["rng_state"]["torch"])
    rows = full_last["history"]
    assert all(row["train_event_count"] == 12 and row["train_batch_count"] == 3 for row in rows)
    assert full["best_epoch"] == training.earliest_strict_minimum(rows, metric_key="val_qty_rmse")["epoch"]
    cached, _, _ = _run(resumed_args)
    assert cached["checkpoint_state_sha256"] == resumed["checkpoint_state_sha256"]
    validate_checkpoint_route(resumed_last, SLOT_MEMORY_BACKBONE)
    selected = torch_load_checkpoint(resumed_dir / "best_val_qty_rmse_model.pt", map_location="cpu")
    validate_checkpoint_route(selected, SLOT_MEMORY_BACKBONE)
    validate_checkpoint_route(json.loads((resumed_dir / "summary.json").read_text()), SLOT_MEMORY_BACKBONE)
    with pytest.raises(ValueError):
        validate_checkpoint_route(selected, "titantpp")
    wrong = copy.deepcopy(selected)
    wrong["encoder_config"]["max_len"] += 1
    with pytest.raises(ValueError):
        validate_checkpoint_route(wrong, SLOT_MEMORY_BACKBONE)


@pytest.mark.parametrize("change", [
    {"checkpoint_monitor": "validation_joint_objective"},
    {"time_intercept_limit": 30.0},
    {"quantile_adaptive_strength": 0.1},
])
def test_cli_rejects_comparison_drift_before_accessing_data(monkeypatch, tmp_path, change):
    args = _arguments(monkeypatch, tmp_path, tmp_path / "output")
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises(ValueError, match="Raw-RMSE-aligned"):
        cli.run(args)
    assert not args.output_dir.exists()


def test_dedicated_role_and_existing_factory_defaults(monkeypatch, tmp_path):
    args = _arguments(monkeypatch, tmp_path, tmp_path / "output")
    validate_model_role_contract(
        model_role=args.model_role, backbones=(args.backbones,),
        quantity_variants=(LOG_MSE_VARIANT,), time_head_mode=args.time_head_mode, lambda_tail=0,
    )
    with pytest.raises(ValueError, match="dedicated"):
        validate_model_role_contract(
            model_role="experimental", backbones=(SLOT_MEMORY_BACKBONE,),
            quantity_variants=(LOG_MSE_VARIANT,), time_head_mode=args.time_head_mode, lambda_tail=0,
        )
    kwargs = dict(hidden_dim=64, train_log_mean=1.5, max_seq_len=8, time_intercept_limit=300.0)
    baseline, _ = build_count_aware_model("titantpp", **kwargs)
    candidate, meta = build_count_aware_model(SLOT_MEMORY_BACKBONE, **kwargs)
    assert candidate.time_head_contract() == baseline.time_head_contract()
    assert sum(p.numel() for p in candidate.parameters()) - sum(p.numel() for p in baseline.parameters()) == 1601
    assert meta["additional_parameter_count"] == 1601
    assert meta["max_len"] == 8
    assert training.DEFAULT_CHECKPOINT_MONITOR == "validation_joint_objective"


@pytest.mark.parametrize("corruption", ["missing_address", "foreign_parameter", "nan_gate", "wrong_slot_count", "wrong_mass_epsilon", "wrong_split"])
def test_checkpoint_rejects_changed_structure_or_scope(corruption):
    model, metadata = build_count_aware_model(
        SLOT_MEMORY_BACKBONE, hidden_dim=16, train_log_mean=1.5,
        max_seq_len=8, time_intercept_limit=300.0,
    )
    payload = {
        "backbone": SLOT_MEMORY_BACKBONE, "variant": LOG_MSE_VARIANT,
        "encoder_config": metadata, "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False, "model_state_dict": copy.deepcopy(model.state_dict()),
    }
    validate_checkpoint_route(payload, SLOT_MEMORY_BACKBONE)
    if corruption == "missing_address":
        del payload["model_state_dict"]["slot_memory.addresses"]
    elif corruption == "foreign_parameter":
        payload["model_state_dict"]["interlayer_alpha_raw"] = torch.zeros(())
    elif corruption == "nan_gate":
        payload["model_state_dict"]["slot_memory_alpha_raw"].fill_(float("nan"))
    elif corruption == "wrong_slot_count":
        payload["encoder_config"]["slot_memory_slots"] = 16
    elif corruption == "wrong_mass_epsilon":
        payload["encoder_config"]["slot_memory_mass_epsilon"] = 0.0
    elif corruption == "wrong_split":
        payload["evaluation_scope"] = "unsupported_scope"
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, SLOT_MEMORY_BACKBONE)
