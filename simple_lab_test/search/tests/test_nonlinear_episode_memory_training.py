"""Nonlinear pre/post pooling through the real shared trainer, synthetic CPU rows only.

Local loader fixtures follow the existing episode training lifecycle contract.
No other test module or its mutable arm list is imported.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.TPPs.CountAwareTitanNonlinearEpisodeMemory import (
    POST_POOL_MEMORY_BACKBONE,
    POST_POOL_MEMORY_ROLE,
    PRE_POOL_MEMORY_BACKBONE,
    PRE_POOL_MEMORY_ROLE,
)
from paper.scripts import run_count_aware_tpp_backbone_control as cli
from paper.scripts.count_aware_tpp_backbone import training
from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint


VARIANT = "count_only_log_regression"
ARMS = ("titantpp", POST_POOL_MEMORY_BACKBONE, PRE_POOL_MEMORY_BACKBONE)
ROLES = {
    "titantpp": "raw_rmse_baseline_alignment",
    POST_POOL_MEMORY_BACKBONE: POST_POOL_MEMORY_ROLE,
    PRE_POOL_MEMORY_BACKBONE: PRE_POOL_MEMORY_ROLE,
}


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def arguments(monkeypatch, root, output, backbone, epochs=3):
    monkeypatch.setattr(sys, "argv", [
        "runner", "--data", str(root / "synthetic_unmaterialized.parquet"),
        "--split-manifest", str(root / "synthetic_unmaterialized.json"),
        "--output-dir", str(output), "--source-revision", "1" * 40,
        "--execution-role", "local_synthetic_nonlinear_contract_test", "--device", "cpu",
        "--backbones", backbone, "--model-role", ROLES[backbone],
        "--quantity-variants", VARIANT, "--lambda-tail", "0",
        "--checkpoint-monitor", "validation_raw_quantity_rmse", "--time-intercept-limit", "300",
        "--allow-partial-contract", "--epochs", str(epochs), "--min-epochs", str(epochs),
        "--early-stopping-patience", str(epochs), "--batch-size", "4", "--hidden-dim", "16",
        "--max-seq-len", "8",
    ])
    return cli.parse_args()


def synthetic_dataset(split):
    count = 9 if split == "train" else 7
    row = torch.arange(count, dtype=torch.float32)[:, None]
    position = torch.arange(6, dtype=torch.float32)[None]
    dts = 0.1 + row / 4 + position / 2
    quantities = 2.0 + row * 3 + position * 2
    mask = torch.arange(6)[None] < (torch.arange(count) % 5 + 2)[:, None]
    dts[~mask] = 0
    quantities[~mask] = 0
    return TensorDataset(torch.zeros_like(dts), dts, mask, torch.zeros_like(dts), quantities)


class RecordingLoader:
    def __init__(self, loader, split, records):
        self.loader, self.split, self.records = loader, split, records

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        digest, count, batches = hashlib.sha256(), 0, 0
        for batch in self.loader:
            for value in batch:
                header = json.dumps([str(value.dtype), list(value.shape)]).encode()
                raw = value.detach().contiguous().numpy().tobytes()
                for encoded in (header, raw):
                    digest.update(len(encoded).to_bytes(8, "little"))
                    digest.update(encoded)
            count += len(batch[1])
            batches += 1
            yield batch
        self.records[self.split].append({"count": count, "batches": batches, "sha256": digest.hexdigest()})


def install_loader(monkeypatch, records):
    def make_loader(_frame, *, target_split, batch_size, shuffle, generator, **_):
        assert target_split in {"train", "validation"}
        loader = DataLoader(synthetic_dataset(target_split), batch_size=batch_size, shuffle=shuffle, generator=generator)
        return RecordingLoader(loader, target_split, records)
    monkeypatch.setattr(training, "make_loader", make_loader)


def run(args, backbone):
    return training.train_one(
        args=args, frame=None,
        quantity_contract={"boundaries": [], "strata": [{"label": "synthetic_all"}]},
        interface_meta={
            "train_target_mean": 1.5, "train_target_std": 1.0,
            "time_head": {"time_initial_intercept": 0.0}, "data_scope": "synthetic_cpu_only",
        },
        backbone=backbone, quantity_variant=VARIANT, seed=42,
    )


def run_directory(args, backbone):
    return args.output_dir / "runs" / backbone / VARIANT / "seed_42"


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_equal(a, b)
    else:
        assert left == right


def test_real_three_arm_training_exposure_endpoints_selector_and_cache(monkeypatch, tmp_path):
    traces = {}
    for backbone in ARMS:
        records = {"train": [], "validation": []}
        install_loader(monkeypatch, records)
        args = arguments(monkeypatch, tmp_path, tmp_path / "three_arms", backbone)
        summary, _, _ = run(args, backbone)
        directory = run_directory(args, backbone)
        selected = torch_load_checkpoint(directory / "best_val_qty_rmse_model.pt", map_location="cpu")
        last = torch_load_checkpoint(directory / "last_epoch_state.pt", map_location="cpu")
        cached_json = json.loads((directory / "summary.json").read_text())
        assert summary["completed_epochs"] == 3
        assert summary["stopped_early"] is False
        assert len(records["train"]) == 3
        assert len(records["validation"]) == 4  # three epochs, then selected replay
        assert all(row["count"] == 9 and row["batches"] == 3 for row in records["train"])
        assert all(row["count"] == 7 and row["batches"] == 2 for row in records["validation"])
        for payload in (selected, last, cached_json):
            validate_checkpoint_route(payload, backbone)
            assert payload["evaluation_scope"] == "validation_only"
            assert payload["held_out_test_evaluated"] is False
        assert last["epoch"] == 3
        minimum = min(row["val_qty_rmse"] for row in last["history"])
        first_minimum = next(row for row in last["history"] if row["val_qty_rmse"] == minimum)
        assert selected["best_epoch"] == summary["best_epoch"] == first_minimum["epoch"]
        assert selected["selected_metric_value"] == minimum
        assert summary["checkpoint_state_sha256"] == canonical_state_dict_sha256(selected["model_state_dict"])
        assert_equal(selected["model_state_dict"], last["best_state_dict"])
        for state in last["optimizer_state_dict"]["state"].values():
            assert state["step"].item() == 9
        before_cache = copy.deepcopy(records)
        cached, _, _ = run(args, backbone)
        assert cached["checkpoint_state_sha256"] == summary["checkpoint_state_sha256"]
        assert records == before_cache
        traces[backbone] = records
    assert traces[ARMS[0]] == traces[ARMS[1]] == traces[ARMS[2]]


@pytest.mark.parametrize("backbone", [POST_POOL_MEMORY_BACKBONE, PRE_POOL_MEMORY_BACKBONE])
def test_real_model_training_strict_tie_keeps_first_selected_state(monkeypatch, tmp_path, backbone):
    records = {"train": [], "validation": []}
    install_loader(monkeypatch, records)
    args = arguments(monkeypatch, tmp_path, tmp_path / "ties", backbone)
    actual_evaluate = training.evaluate

    def tie_selector_metric(**kwargs):
        result = actual_evaluate(**kwargs)
        # Keep genuine model training/evaluation, but make only the selector's
        # scalar tie exactly. Later states must not replace the first optimum.
        return {**result, "qty_rmse": 42.0}

    monkeypatch.setattr(training, "evaluate", tie_selector_metric)
    summary, _, _ = run(args, backbone)
    directory = run_directory(args, backbone)
    selected = torch_load_checkpoint(directory / "best_val_qty_rmse_model.pt", map_location="cpu")
    last = torch_load_checkpoint(directory / "last_epoch_state.pt", map_location="cpu")
    assert summary["best_epoch"] == selected["best_epoch"] == last["best_epoch"] == 1
    assert [row["val_qty_rmse"] for row in last["history"]] == [42.0, 42.0, 42.0]
    assert selected["model_state_sha256"] != last["model_state_sha256"]
    assert_equal(selected["model_state_dict"], last["best_state_dict"])


@pytest.mark.parametrize("backbone", [POST_POOL_MEMORY_BACKBONE, PRE_POOL_MEMORY_BACKBONE])
def test_full_vs_interrupted_resume_restores_optimizer_rng_loader_and_selector(monkeypatch, tmp_path, backbone):
    full_records = {"train": [], "validation": []}
    install_loader(monkeypatch, full_records)
    full_args = arguments(monkeypatch, tmp_path, tmp_path / "full", backbone)
    full, _, _ = run(full_args, backbone)

    resumed_records = {"train": [], "validation": []}
    install_loader(monkeypatch, resumed_records)
    resumed_args = arguments(monkeypatch, tmp_path, tmp_path / "resumed", backbone)
    original_save = training.atomic_torch_save

    def committed_epoch_interrupt(payload, path):
        original_save(payload, path)
        if path.name == "last_epoch_state.pt" and payload["epoch"] == 1:
            raise RuntimeError("synthetic interruption after committed epoch")

    monkeypatch.setattr(training, "atomic_torch_save", committed_epoch_interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(resumed_args, backbone)
    monkeypatch.setattr(training, "atomic_torch_save", original_save)
    resumed, _, _ = run(resumed_args, backbone)
    full_last = torch_load_checkpoint(run_directory(full_args, backbone) / "last_epoch_state.pt", map_location="cpu")
    resumed_last = torch_load_checkpoint(run_directory(resumed_args, backbone) / "last_epoch_state.pt", map_location="cpu")
    assert full["checkpoint_state_sha256"] == resumed["checkpoint_state_sha256"]
    assert full["best_epoch"] == resumed["best_epoch"]
    assert full_records == resumed_records
    for key in (
        "model_state_dict", "best_state_dict", "optimizer_state_dict", "rng_state",
        "train_loader_generator_state", "best_epoch", "best_selection_value", "initial_state_sha256",
    ):
        assert_equal(full_last[key], resumed_last[key])
    # Wall-clock telemetry is deliberately excluded; all epoch objectives and
    # selector values must reproduce exactly after optimizer/RNG restoration.
    for full_row, resumed_row in zip(full_last["history"], resumed_last["history"], strict=True):
        for key in full_row:
            if key.startswith("val_") or key in {"epoch", "train_event_count", "train_batch_count"}:
                assert_equal(full_row[key], resumed_row[key])


def checkpoint_payload(backbone):
    torch.manual_seed(42)
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=16, train_log_mean=1.5, train_log_std=1.0,
        max_seq_len=8, quantity_variant=VARIANT, lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp", time_intercept_limit=300.0,
    )
    return {
        "backbone": backbone, "variant": VARIANT, "encoder_config": metadata,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
    }


@pytest.mark.parametrize("source", [POST_POOL_MEMORY_BACKBONE, PRE_POOL_MEMORY_BACKBONE])
def test_state_pooling_prevents_control_candidate_relabeling_even_with_replaced_metadata(source):
    payload = checkpoint_payload(source)
    validate_checkpoint_route(payload, source)
    target = PRE_POOL_MEMORY_BACKBONE if source == POST_POOL_MEMORY_BACKBONE else POST_POOL_MEMORY_BACKBONE
    for expected in ("titantpp", target):
        with pytest.raises(ValueError):
            validate_checkpoint_route(payload, expected)
    relabelled = copy.deepcopy(payload)
    relabelled["backbone"] = target
    relabelled["encoder_config"] = checkpoint_payload(target)["encoder_config"]
    with pytest.raises(ValueError):
        validate_checkpoint_route(relabelled, target)


@pytest.mark.parametrize("corruption", [
    "missing_output", "pooling_dtype", "foreign_state", "wrong_split", "wrong_rank",
    "nonfinite_projection", "foreign_objective", "activation", "tail_objective",
    "value_bias_width", "value_width", "design_digest", "held_out_flag",
])
def test_episode_checkpoint_rejects_incomplete_foreign_or_inactive_contract_state(corruption):
    payload = checkpoint_payload(PRE_POOL_MEMORY_BACKBONE)
    if corruption == "missing_output":
        del payload["model_state_dict"]["nonlinear_episode_memory.output_projection.weight"]
    elif corruption == "pooling_dtype":
        payload["model_state_dict"]["nonlinear_episode_memory.pooling_code"] = torch.tensor(1.0)
    elif corruption == "foreign_state":
        payload["model_state_dict"]["slot_memory_alpha_raw"] = torch.zeros(())
    elif corruption == "wrong_split":
        payload["evaluation_scope"] = "unsupported_scope"
    elif corruption == "wrong_rank":
        payload["model_state_dict"]["nonlinear_episode_memory.query_projection.weight"] = torch.zeros(8, 16)
    elif corruption == "nonfinite_projection":
        payload["model_state_dict"]["nonlinear_episode_memory.output_projection.weight"][0, 0] = float("nan")
    elif corruption == "foreign_objective":
        payload["training_config"] = {"quantile_adaptive_strength": 0.5}
    elif corruption == "activation":
        payload["encoder_config"]["nonlinear_episode_activation"] = "relu"
    elif corruption == "value_bias_width":
        payload["model_state_dict"]["nonlinear_episode_memory.value_projection.bias"] = torch.zeros(2)
    elif corruption == "value_width":
        payload["model_state_dict"]["nonlinear_episode_memory.value_projection.weight"] = torch.zeros(8, 2)
    elif corruption == "design_digest":
        payload["encoder_config"]["design_contract_sha256"] = "0" * 64
    elif corruption == "held_out_flag":
        payload["held_out_test_evaluated"] = True
    else:
        payload["lambda_tail"] = 1.0
    with pytest.raises(ValueError):
        validate_checkpoint_route(payload, PRE_POOL_MEMORY_BACKBONE)


def test_nonfinite_selector_history_is_rejected():
    history = [{"epoch": 1, "val_qty_rmse": 2.0}, {"epoch": 2, "val_qty_rmse": float("nan")}]
    with pytest.raises(ValueError, match="finite"):
        training.earliest_strict_minimum(history, metric_key="val_qty_rmse")


@pytest.mark.parametrize("changed_argument,new_value", [
    ("source_revision", "2" * 40), ("batch_size", 3),
    ("checkpoint_monitor", "validation_joint_objective"),
])
def test_resume_rejects_changed_source_batches_or_selector_before_more_training(
    monkeypatch, tmp_path, changed_argument, new_value,
):
    records = {"train": [], "validation": []}
    install_loader(monkeypatch, records)
    backbone = PRE_POOL_MEMORY_BACKBONE
    args = arguments(monkeypatch, tmp_path, tmp_path / "identity", backbone)
    save = training.atomic_torch_save

    def interrupt_after_saved_epoch(payload, path):
        save(payload, path)
        if path.name == "last_epoch_state.pt" and payload["epoch"] == 1:
            raise RuntimeError("synthetic committed interruption")

    monkeypatch.setattr(training, "atomic_torch_save", interrupt_after_saved_epoch)
    with pytest.raises(RuntimeError, match="synthetic committed"):
        run(args, backbone)
    monkeypatch.setattr(training, "atomic_torch_save", save)
    directory = run_directory(args, backbone)
    saved_bytes = (directory / "last_epoch_state.pt").read_bytes()
    before = copy.deepcopy(records)
    setattr(args, changed_argument, new_value)
    with pytest.raises(ValueError, match="identity|selector|monitor"):
        run(args, backbone)
    assert records == before
    assert (directory / "last_epoch_state.pt").read_bytes() == saved_bytes
