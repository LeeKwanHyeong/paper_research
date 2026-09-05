from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT, QUANTILE_ADAPTIVE_VARIANT
from paper.scripts.count_aware_tpp_backbone import training
from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    restore_rng_state,
    restore_train_loader_generator_state,
    torch_load_checkpoint,
)


def test_strict_finite_minimum_and_matching_early_stop() -> None:
    history = [
        {"epoch": 1, "val_joint_objective": 4.0, "val_qty_rmse": 5.0},
        {"epoch": 2, "val_joint_objective": 3.0, "val_qty_rmse": 3.0},
        {"epoch": 3, "val_joint_objective": 1.0, "val_qty_rmse": 3.0},
        {"epoch": 4, "val_joint_objective": 2.0, "val_qty_rmse": 4.0},
    ]

    assert training.earliest_strict_minimum(
        history, metric_key="val_joint_objective"
    )["epoch"] == 3
    assert training.earliest_strict_minimum(
        history, metric_key="val_qty_rmse"
    )["epoch"] == 2
    assert not training.early_stopping_exhausted(
        history[:3], min_epochs=1, patience=2, metric_key="val_qty_rmse"
    )
    assert training.early_stopping_exhausted(
        history, min_epochs=1, patience=2, metric_key="val_qty_rmse"
    )
    # The public legacy call remains tied to validation joint objective.
    assert not training.early_stopping_exhausted(
        history, min_epochs=1, patience=2
    )

    for invalid in (float("nan"), float("inf"), -float("inf")):
        bad = copy.deepcopy(history)
        bad[0]["val_qty_rmse"] = invalid
        with pytest.raises(ValueError, match="finite"):
            training.earliest_strict_minimum(bad, metric_key="val_qty_rmse")
    missing = copy.deepcopy(history)
    del missing[-1]["val_qty_rmse"]
    with pytest.raises(ValueError, match="missing"):
        training.earliest_strict_minimum(missing, metric_key="val_qty_rmse")


def test_legacy_joint_defaults_and_cache_cannot_be_relabelled(tmp_path: Path) -> None:
    joint = training.checkpoint_monitor_spec(
        training.VALIDATION_JOINT_OBJECTIVE
    )
    rmse = training.checkpoint_monitor_spec(
        training.VALIDATION_RAW_QUANTITY_RMSE
    )
    assert training.DEFAULT_CHECKPOINT_MONITOR == "validation_joint_objective"
    assert joint == {
        "history_key": "val_joint_objective",
        "evaluation_key": "val_joint_objective",
        "selection": "best_validation_joint_objective",
        "filename": "best_val_joint_objective_model.pt",
    }
    assert rmse["history_key"] == "val_qty_rmse"
    assert rmse["selection"] == "best_validation_raw_quantity_rmse"
    assert rmse["filename"] == "best_val_qty_rmse_model.pt"

    with pytest.raises(ValueError, match="selector identity mismatch"):
        training._validate_selector_metadata(
            {"checkpoint_monitor": training.VALIDATION_JOINT_OBJECTIVE},
            monitor=training.VALIDATION_JOINT_OBJECTIVE,
            allow_legacy_joint=True,
            artifact="Partial artifact",
        )

    run_dir = tmp_path / "runs" / "rmtpp" / LOG_MSE_VARIANT / "seed_42"
    run_dir.mkdir(parents=True)
    model_state = {"weight": torch.tensor([1.0])}
    state_digest = canonical_state_dict_sha256(model_state)
    (run_dir / "summary.json").write_text(
        json.dumps({
            "backbone": "rmtpp",
            "variant": LOG_MSE_VARIANT,
            "seed": 42,
            "checkpoint_state_sha256": state_digest,
            "quantity_rows": [],
            "history_rows": [],
        }),
        encoding="utf-8",
    )
    torch.save(
        {
            "selection": "best_validation_joint_objective",
            "backbone": "rmtpp",
            "variant": LOG_MSE_VARIANT,
            "seed": 42,
            "model_state_dict": model_state,
            "model_state_sha256": state_digest,
            "encoder_config": {},
        },
        run_dir / joint["filename"],
    )

    legacy, quantity_rows, history_rows = training.train_one(
        args=argparse.Namespace(output_dir=tmp_path, force_rerun=False),
        frame=None,
        quantity_contract={},
        interface_meta={},
        backbone="rmtpp",
        quantity_variant=LOG_MSE_VARIANT,
        seed=42,
    )
    assert legacy["backbone"] == "rmtpp"
    assert quantity_rows == history_rows == []

    with pytest.raises(ValueError, match="selector identity"):
        training.train_one(
            args=argparse.Namespace(
                output_dir=tmp_path,
                force_rerun=False,
                checkpoint_monitor=training.VALIDATION_RAW_QUANTITY_RMSE,
            ),
            frame=None,
            quantity_contract={},
            interface_meta={},
            backbone="rmtpp",
            quantity_variant=LOG_MSE_VARIANT,
            seed=42,
        )


def test_adaptive_model_kwargs_are_variant_scoped() -> None:
    interface = {
        "quantile_adaptive_strength": 1.0,
        "quantile_adaptive_contract": {
            "boundaries": [10.0, 20.0, 30.0, 40.0],
            "normalized_bin_weights": [0.5, 0.75, 1.0, 1.5, 2.0],
        },
    }
    assert training._quantity_variant_model_kwargs(LOG_MSE_VARIANT, interface) == {}
    assert training._quantity_variant_model_kwargs(
        QUANTILE_ADAPTIVE_VARIANT,
        interface,
    ) == {
        "quantile_adaptive_strength": 1.0,
        "quantile_adaptive_boundaries": (10.0, 20.0, 30.0, 40.0),
        "quantile_adaptive_weights": (0.5, 0.75, 1.0, 1.5, 2.0),
    }
    with pytest.raises(ValueError, match="requires quantile_adaptive_contract"):
        training._quantity_variant_model_kwargs(QUANTILE_ADAPTIVE_VARIANT, {})


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(()))
        self.register_buffer("epoch_index", torch.tensor(0, dtype=torch.int64))

    def to(self, *args: Any, **kwargs: Any) -> _TinyModel:
        # The focused CUDA metadata test does not execute CUDA kernels.
        return self


class _TinyLoader:
    def __init__(self, generator: torch.Generator | None) -> None:
        self.generator = generator


def _args(
    output_dir: Path,
    *,
    epochs: int,
    monitor: str | None,
) -> argparse.Namespace:
    values = {
        "output_dir": output_dir,
        "force_rerun": False,
        "batch_size": 2,
        "lr": 0.01,
        "lookback_weeks": 8,
        "max_seq_len": 16,
        "hidden_dim": 4,
        "lambda_log_qty": 1.0,
        "grad_clip": 1.0,
        "early_stopping_patience": 99,
        "min_epochs": 1,
        "quantity_sigma_floor": 1e-3,
        "lambda_location_huber": 1.0,
        "location_huber_delta": 0.25,
        "lambda_tail": 0.0,
        "tail_threshold": 46.0,
        "tail_normalization_scale": 46.0,
        "tail_clip_cap": 187.0,
        "tail_huber_delta": 1.0,
        "time_head_mode": "legacy_clamped_rmtpp",
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 30.0,
        "time_wd_safety_limit": 40.0,
        "time_head_lr_multiplier": 1.0,
        "time_sigma_floor": 1e-3,
        "titans_memory_gradient_clip": None,
        "max_train_batches": None,
        "max_val_batches": 1,
        "device": "cpu",
        "source_revision": "a" * 40,
        "data_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "epochs": epochs,
    }
    if monitor is not None:
        values["checkpoint_monitor"] = monitor
    return argparse.Namespace(**values)


def _install_tiny_training(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, int | None]:
    control: dict[str, int | None] = {"fail_after_completed_epochs": None}

    def build_model(*args: Any, **kwargs: Any) -> tuple[_TinyModel, dict[str, Any]]:
        return _TinyModel(), {}

    def make_loader(*args: Any, **kwargs: Any) -> _TinyLoader:
        return _TinyLoader(kwargs["generator"])

    def train_epoch_with_telemetry(
        *,
        model: _TinyModel,
        loader: _TinyLoader,
        optimizer: torch.optim.Optimizer,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert loader.generator is not None
        fail_after = control["fail_after_completed_epochs"]
        if fail_after is not None and int(model.epoch_index) >= fail_after:
            raise RuntimeError("synthetic interruption")
        draws = (
            random.random(),
            float(np.random.random()),
            float(torch.rand(())),
            float(torch.rand((), generator=loader.generator)),
        )
        target = model.weight.new_tensor(sum(draws))
        optimizer.zero_grad(set_to_none=True)
        loss = torch.square(model.weight - target)
        loss.backward()
        optimizer.step()
        model.epoch_index.add_(1)
        loss_value = float(loss.detach())
        return {
            "train_joint_objective": loss_value,
            "train_time_nll": loss_value / 2.0,
            "train_quantity_loss": loss_value / 2.0,
            "train_batch_joint_p50": loss_value,
            "train_batch_joint_p95": loss_value,
            "train_batch_joint_p99": loss_value,
            "train_batch_joint_max": loss_value,
            "train_max_per_event_time_nll": loss_value,
            "train_pre_clip_grad_norm_mean": loss_value,
            "train_pre_clip_grad_norm_max": loss_value,
            "train_gradient_clip_count": 0,
            "train_gradient_clip_fraction": 0.0,
            "train_event_count": 1,
            "train_batch_count": 1,
            "train_all_finite": True,
        }

    def evaluate(*, model: _TinyModel, **kwargs: Any) -> dict[str, Any]:
        epoch = int(model.epoch_index)
        joint_by_epoch = {1: 4.0, 2: 3.0, 3: 1.0, 4: 2.0}
        rmse_by_epoch = {1: 4.0, 2: 1.0, 3: 2.0, 4: 3.0}
        joint = joint_by_epoch[epoch]
        rmse = rmse_by_epoch[epoch]
        return {
            "val_joint_objective": joint,
            "val_time_nll": joint / 2.0,
            "val_quantity_train_loss": joint / 2.0,
            "val_log_qty_mse": joint / 2.0,
            "val_quantity_distribution_nll": 0.0,
            "val_quantity_location_huber": 0.0,
            "val_tail_aux_loss": 0.0,
            "val_tail_count": 0,
            "val_quantity_scale_mean": 1.0,
            "qty_mae": rmse / 2.0,
            "qty_rmse": rmse,
        }

    monkeypatch.setattr(training, "build_model", build_model)
    monkeypatch.setattr(training, "make_loader", make_loader)
    monkeypatch.setattr(
        training,
        "train_epoch_with_telemetry",
        train_epoch_with_telemetry,
    )
    monkeypatch.setattr(training, "evaluate", evaluate)
    return control


def _interface_meta() -> dict[str, Any]:
    return {
        "train_target_mean": 1.0,
        "train_target_std": 1.0,
        "time_head": {
            "time_initial_intercept": 0.0,
            "time_initial_location": None,
            "time_initial_scale": None,
        },
    }


def _run(
    output_dir: Path,
    *,
    epochs: int,
    monitor: str | None,
    force_rerun: bool = False,
) -> dict[str, Any]:
    args = _args(output_dir, epochs=epochs, monitor=monitor)
    args.force_rerun = force_rerun
    summary, _, _ = training.train_one(
        args=args,
        frame=object(),
        quantity_contract={"strata": []},
        interface_meta=_interface_meta(),
        backbone="rmtpp",
        quantity_variant=LOG_MSE_VARIANT,
        seed=42,
    )
    return summary


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_monitor_selection_and_exact_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _install_tiny_training(monkeypatch)
    monitor = training.VALIDATION_RAW_QUANTITY_RMSE
    resumed_root = tmp_path / "resumed"
    full_root = tmp_path / "full"

    control["fail_after_completed_epochs"] = 2
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run(resumed_root, epochs=4, monitor=monitor)
    resumed_run = resumed_root / "runs" / "rmtpp" / LOG_MSE_VARIANT / "seed_42"
    assert not (resumed_run / "summary.json").exists()
    assert not (resumed_run / "best_val_qty_rmse_model.pt").exists()

    control["fail_after_completed_epochs"] = None
    resumed = _run(resumed_root, epochs=4, monitor=monitor)
    uninterrupted = _run(full_root, epochs=4, monitor=monitor)

    assert resumed["best_epoch"] == uninterrupted["best_epoch"] == 2
    assert resumed["checkpoint_selection"] == "best_validation_raw_quantity_rmse"
    assert resumed["selected_metric_value"] == 1.0
    assert Path(resumed["checkpoint_path"]).name == "best_val_qty_rmse_model.pt"
    assert resumed["checkpoint_state_sha256"] == uninterrupted[
        "checkpoint_state_sha256"
    ]
    selected_checkpoint = torch_load_checkpoint(
        resumed_run / "best_val_qty_rmse_model.pt",
        map_location="cpu",
    )
    assert selected_checkpoint["best_epoch"] == 2
    assert selected_checkpoint["selected_metric_value"] == 1.0

    full_run = full_root / "runs" / "rmtpp" / LOG_MSE_VARIANT / "seed_42"
    resumed_history = json.loads(
        (resumed_run / "history.json").read_text(encoding="utf-8")
    )
    full_history = json.loads(
        (full_run / "history.json").read_text(encoding="utf-8")
    )
    assert resumed_history == full_history

    resumed_last = torch_load_checkpoint(
        resumed_run / "last_epoch_state.pt", map_location="cpu"
    )
    full_last = torch_load_checkpoint(
        full_run / "last_epoch_state.pt", map_location="cpu"
    )
    assert resumed_last["model_state_sha256"] == full_last["model_state_sha256"]
    assert resumed_last["best_state_sha256"] == full_last["best_state_sha256"]
    _assert_nested_equal(
        resumed_last["optimizer_state_dict"],
        full_last["optimizer_state_dict"],
    )
    _assert_nested_equal(resumed_last["rng_state"], full_last["rng_state"])
    assert torch.equal(
        resumed_last["train_loader_generator_state"],
        full_last["train_loader_generator_state"],
    )
    assert resumed_last["best_epoch"] == 2
    assert resumed_last["best_val_joint_objective"] == 3.0
    assert canonical_state_dict_sha256(resumed_last["best_state_dict"]) == resumed[
        "checkpoint_state_sha256"
    ]

    restored_a = torch.Generator().manual_seed(1)
    restored_b = torch.Generator().manual_seed(2)
    restore_train_loader_generator_state(
        restored_a, resumed_last["train_loader_generator_state"]
    )
    restore_train_loader_generator_state(
        restored_b, full_last["train_loader_generator_state"]
    )
    assert torch.equal(
        torch.randperm(17, generator=restored_a),
        torch.randperm(17, generator=restored_b),
    )
    restore_rng_state(resumed_last["rng_state"])
    next_resumed = (random.random(), float(np.random.random()), float(torch.rand(())))
    restore_rng_state(full_last["rng_state"])
    next_full = (random.random(), float(np.random.random()), float(torch.rand(())))
    assert next_resumed == next_full

    for missing_field, message in (
        ("checkpoint_type", "type"),
        ("checkpoint_schema_version", "schema"),
        ("rng_state", "exact-replay"),
    ):
        invalid = copy.deepcopy(resumed_last)
        del invalid[missing_field]
        with pytest.raises(ValueError, match=message):
            training._validate_resume_payload(
                invalid,
                expected_identity=resumed_last["resume_identity"],
                expected_initial_state_sha256=resumed_last["initial_state_sha256"],
                monitor=monitor,
            )
    for rng_key in ("python", "numpy", "torch"):
        invalid_rng = copy.deepcopy(resumed_last)
        del invalid_rng["rng_state"][rng_key]
        with pytest.raises(ValueError, match="RNG state"):
            training._validate_resume_payload(
                invalid_rng,
                expected_identity=resumed_last["resume_identity"],
                expected_initial_state_sha256=resumed_last["initial_state_sha256"],
                monitor=monitor,
            )

    invalid_digest = copy.deepcopy(selected_checkpoint)
    invalid_digest["model_state_dict"]["weight"].add_(1.0)
    with pytest.raises(ValueError, match="model digest mismatch"):
        training._validate_cached_checkpoint_digest(resumed, invalid_digest)

    wrong_identity = copy.deepcopy(resumed_last["resume_identity"])
    wrong_identity["arguments"]["batch_size"] = 999
    with pytest.raises(ValueError, match="training identity"):
        training._validate_resume_payload(
            resumed_last,
            expected_identity=wrong_identity,
            expected_initial_state_sha256=resumed_last["initial_state_sha256"],
            monitor=monitor,
        )

    with pytest.raises(ValueError, match="initial model state mismatch"):
        training._validate_resume_payload(
            resumed_last,
            expected_identity=resumed_last["resume_identity"],
            expected_initial_state_sha256="0" * 64,
            monitor=monitor,
        )


def test_force_rerun_ignores_stale_cache_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_training(monkeypatch)
    _run(tmp_path, epochs=1, monitor=None)
    run_dir = tmp_path / "runs" / "rmtpp" / LOG_MSE_VARIANT / "seed_42"
    stale_summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    stale_summary["checkpoint_monitor"] = "stale_selector"
    (run_dir / "summary.json").write_text(
        json.dumps(stale_summary),
        encoding="utf-8",
    )
    (run_dir / "last_epoch_state.pt").write_bytes(b"not a checkpoint")

    rerun = _run(tmp_path, epochs=1, monitor=None, force_rerun=True)

    assert rerun["checkpoint_monitor"] == training.VALIDATION_JOINT_OBJECTIVE
    assert rerun["best_epoch"] == 1


def test_legacy_default_training_still_selects_joint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_training(monkeypatch)
    summary = _run(tmp_path, epochs=4, monitor=None)

    assert summary["checkpoint_monitor"] == "validation_joint_objective"
    assert summary["checkpoint_selection"] == "best_validation_joint_objective"
    assert summary["best_epoch"] == 3
    assert Path(summary["checkpoint_path"]).name == (
        "best_val_joint_objective_model.pt"
    )


def test_cuda_memory_metadata_applies_to_every_backbone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_tiny_training(monkeypatch)
    reset_devices: list[torch.device] = []
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: reset_devices.append(device),
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 456)
    args = _args(tmp_path, epochs=1, monitor=None)
    args.device = "cuda:0"

    summary, _, _ = training.train_one(
        args=args,
        frame=object(),
        quantity_contract={"strata": []},
        interface_meta=_interface_meta(),
        backbone="titantpp",
        quantity_variant=LOG_MSE_VARIANT,
        seed=42,
    )

    assert reset_devices == [torch.device("cuda:0")]
    assert summary["training_device"] == "cuda:0"
    assert summary["cuda_peak_memory_allocated_bytes"] == 123
    assert summary["cuda_peak_memory_reserved_bytes"] == 456
    assert training._training_device_metadata("rmtpp", "cpu") == {}
