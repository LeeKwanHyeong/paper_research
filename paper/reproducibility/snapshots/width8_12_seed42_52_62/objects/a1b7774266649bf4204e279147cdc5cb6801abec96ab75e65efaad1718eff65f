#!/usr/bin/env python3
"""Refit only the legacy RMTPP time head of a frozen Hard-LMM B checkpoint.

The encoder, memories, and quantity head are evaluated once and then frozen.
Epoch zero is a real validation candidate, so selection cannot make validation
Time NLL worse than the supplied B checkpoint.  This script never materializes
held-out rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    SharedTimeCountModel,
)
from paper.scripts.count_aware_tpp_backbone.core import (
    load_train_validation_frame,
    prepare_count_frame,
    right_pad_batch,
)
from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


CONTRACT_ID = "hard_lmm_time_head_refit_v1"
DEFAULT_CONTRACT = PROJECT_ROOT / "paper/contracts/hard_lmm_time_head_refit_v1.json"
TIME_HEAD_PARAMETER_NAMES = ("v_t.weight", "b_t", "w_raw")
SOURCE_BACKBONE = "titantpp"
SOURCE_VARIANT = LOG_MSE_VARIANT
SOURCE_MONITOR = "validation_raw_quantity_rmse"
B_TRAINING_SOURCE_REVISION = "f75243473adc25d622319dbca9bda7e076d8240f"
SELECTION_RULE = "earliest_strict_finite_minimum_validation_time_nll"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def state_partition(
    state: Mapping[str, torch.Tensor], *, time_head: bool
) -> dict[str, torch.Tensor]:
    names = set(TIME_HEAD_PARAMETER_NAMES)
    return {
        name: value
        for name, value in state.items()
        if (name in names) is time_head
    }


def state_partition_sha256(
    state: Mapping[str, torch.Tensor], *, time_head: bool
) -> str:
    partition = state_partition(state, time_head=time_head)
    require(bool(partition), "Requested model-state partition is empty")
    return canonical_state_dict_sha256(partition)


def tensor_sha256(name: str, tensor: torch.Tensor) -> str:
    return canonical_state_dict_sha256({name: tensor.detach().cpu()})


def finite_tensor_mapping(values: Mapping[str, Any], *, label: str) -> None:
    for name, value in values.items():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all()):
            raise ValueError(f"{label} contains a non-finite tensor: {name}")


def finite_nested_tensors(value: Any, *, label: str) -> None:
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f"{label} contains a non-finite tensor")
    elif isinstance(value, Mapping):
        for name, child in value.items():
            finite_nested_tensors(child, label=f"{label}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_nested_tensors(child, label=f"{label}[{index}]")


def freeze_time_head_only(model: SharedTimeCountModel) -> tuple[torch.nn.Parameter, ...]:
    """Freeze every state-producing path and expose exactly three parameters."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    named = dict(model.named_parameters())
    require(
        set(TIME_HEAD_PARAMETER_NAMES).issubset(named),
        "Model does not expose the frozen legacy time-head parameter boundary",
    )
    for name in TIME_HEAD_PARAMETER_NAMES:
        named[name].requires_grad_(True)
    trainable = tuple(name for name, value in named.items() if value.requires_grad)
    require(
        set(trainable) == set(TIME_HEAD_PARAMETER_NAMES)
        and len(trainable) == len(TIME_HEAD_PARAMETER_NAMES),
        f"Unexpected trainable parameters: {trainable}",
    )
    return tuple(named[name] for name in TIME_HEAD_PARAMETER_NAMES)


def assert_time_head_gradient_boundary(model: SharedTimeCountModel) -> None:
    named = dict(model.named_parameters())
    for name, parameter in named.items():
        if name in TIME_HEAD_PARAMETER_NAMES:
            require(parameter.grad is not None, f"Missing time-head gradient: {name}")
            require(
                bool(torch.isfinite(parameter.grad).all()),
                f"Non-finite time-head gradient: {name}",
            )
        else:
            require(parameter.grad is None, f"Frozen parameter received gradient: {name}")


@dataclass(frozen=True)
class FrozenFeatureCache:
    """Detached states and targets produced by one immutable B checkpoint."""

    time_hidden: torch.Tensor
    target_dt: torch.Tensor
    quantity_hidden: torch.Tensor | None = None
    target_quantity: torch.Tensor | None = None
    source_quantity_prediction: torch.Tensor | None = None

    @property
    def count(self) -> int:
        return int(self.time_hidden.shape[0])

    def validate(self, *, require_quantity: bool) -> None:
        require(self.time_hidden.ndim == 2, "time_hidden must be rank two")
        require(self.target_dt.shape == (self.count,), "target_dt shape mismatch")
        require(self.count > 0, "Frozen feature cache is empty")
        tensors: dict[str, torch.Tensor] = {
            "time_hidden": self.time_hidden,
            "target_dt": self.target_dt,
        }
        if require_quantity:
            require(self.quantity_hidden is not None, "quantity_hidden is required")
            require(self.target_quantity is not None, "target_quantity is required")
            require(
                self.source_quantity_prediction is not None,
                "source_quantity_prediction is required",
            )
            assert self.quantity_hidden is not None
            assert self.target_quantity is not None
            assert self.source_quantity_prediction is not None
            require(
                self.quantity_hidden.shape == self.time_hidden.shape,
                "quantity_hidden shape mismatch",
            )
            require(
                self.target_quantity.shape == (self.count,),
                "target_quantity shape mismatch",
            )
            require(
                self.source_quantity_prediction.shape == (self.count,),
                "source quantity prediction shape mismatch",
            )
            tensors.update(
                quantity_hidden=self.quantity_hidden,
                target_quantity=self.target_quantity,
                source_quantity_prediction=self.source_quantity_prediction,
            )
        finite_tensor_mapping(tensors, label="Frozen feature cache")
        require(bool((self.target_dt >= 0.0).all()), "Time targets must be nonnegative")
        if self.target_quantity is not None:
            require(
                bool((self.target_quantity >= 0.0).all()),
                "Quantity targets must be nonnegative",
            )

    def digest(self) -> str:
        values = {
            "time_hidden": self.time_hidden,
            "target_dt": self.target_dt,
        }
        for name in (
            "quantity_hidden",
            "target_quantity",
            "source_quantity_prediction",
        ):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        return canonical_state_dict_sha256(values)

    def to_payload(self, *, identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "cache_schema_version": 1,
            "identity": dict(identity),
            "cache_sha256": self.digest(),
            "time_hidden": self.time_hidden,
            "target_dt": self.target_dt,
            "quantity_hidden": self.quantity_hidden,
            "target_quantity": self.target_quantity,
            "source_quantity_prediction": self.source_quantity_prediction,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        expected_identity: Mapping[str, Any],
        require_quantity: bool,
    ) -> "FrozenFeatureCache":
        require(payload.get("cache_schema_version") == 1, "Wrong feature-cache schema")
        require(payload.get("identity") == dict(expected_identity), "Feature-cache identity drift")
        cache = cls(
            time_hidden=payload["time_hidden"],
            target_dt=payload["target_dt"],
            quantity_hidden=payload.get("quantity_hidden"),
            target_quantity=payload.get("target_quantity"),
            source_quantity_prediction=payload.get("source_quantity_prediction"),
        )
        cache.validate(require_quantity=require_quantity)
        require(cache.digest() == payload.get("cache_sha256"), "Feature-cache digest drift")
        return cache


@torch.no_grad()
def extract_frozen_features(
    *,
    model: SharedTimeCountModel,
    loader: Iterable[Any],
    device: str | torch.device,
    include_quantity: bool,
    max_batches: int | None = None,
) -> FrozenFeatureCache:
    """Encode causal histories once with dropout disabled and targets hidden."""
    model.eval()
    time_states: list[torch.Tensor] = []
    target_dts: list[torch.Tensor] = []
    quantity_states: list[torch.Tensor] = []
    target_quantities: list[torch.Tensor] = []
    source_predictions: list[torch.Tensor] = []
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if quantities is None:
            raise ValueError("Time-head refit requires raw quantity histories")
        dts = dts.to(device)
        quantities = quantities.to(device)
        mask = mask.to(device)
        right_dts, right_quantities, right_mask, lengths = right_pad_batch(
            dts, quantities, mask
        )
        batch_ids = torch.arange(right_dts.size(0), device=right_dts.device)
        target_positions = lengths - 1
        history_positions = lengths - 2
        history_quantities = right_quantities.clone()
        history_quantities[batch_ids, target_positions] = 0.0
        memory_write_mask = right_mask.clone()
        memory_write_mask[batch_ids, target_positions] = False
        time_encoded, quantity_encoded = model.encode_task_states(
            right_dts,
            history_quantities,
            right_mask,
            memory_write_mask=memory_write_mask,
        )
        time_hidden = time_encoded[batch_ids, history_positions].detach()
        quantity_hidden = quantity_encoded[batch_ids, history_positions].detach()
        target_dt = right_dts[batch_ids, target_positions].float().detach()
        time_states.append(time_hidden.cpu())
        target_dts.append(target_dt.cpu())
        if include_quantity:
            target_quantity = (
                right_quantities[batch_ids, target_positions].float().detach()
            )
            prediction = model.predict_quantity(quantity_hidden)[1].detach()
            quantity_states.append(quantity_hidden.cpu())
            target_quantities.append(target_quantity.cpu())
            source_predictions.append(prediction.cpu())
    if not time_states:
        raise ValueError("No batches were available for frozen feature extraction")
    cache = FrozenFeatureCache(
        time_hidden=torch.cat(time_states, dim=0).contiguous(),
        target_dt=torch.cat(target_dts, dim=0).contiguous(),
        quantity_hidden=(
            torch.cat(quantity_states, dim=0).contiguous()
            if include_quantity else None
        ),
        target_quantity=(
            torch.cat(target_quantities, dim=0).contiguous()
            if include_quantity else None
        ),
        source_quantity_prediction=(
            torch.cat(source_predictions, dim=0).contiguous()
            if include_quantity else None
        ),
    )
    cache.validate(require_quantity=include_quantity)
    require(not model.training, "Encoder left evaluation mode during feature extraction")
    return cache


def load_or_extract_feature_cache(
    *,
    path: Path,
    identity: Mapping[str, Any],
    model: SharedTimeCountModel,
    loader: Iterable[Any],
    device: str | torch.device,
    include_quantity: bool,
    max_batches: int | None,
) -> FrozenFeatureCache:
    if path.exists():
        payload = torch_load_checkpoint(path, map_location="cpu")
        return FrozenFeatureCache.from_payload(
            payload,
            expected_identity=identity,
            require_quantity=include_quantity,
        )
    cache = extract_frozen_features(
        model=model,
        loader=loader,
        device=device,
        include_quantity=include_quantity,
        max_batches=max_batches,
    )
    atomic_torch_save(cache.to_payload(identity=identity), path)
    return cache


@torch.no_grad()
def evaluate_cached_time_nll(
    *,
    model: SharedTimeCountModel,
    cache: FrozenFeatureCache,
    device: str | torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, cache.count, batch_size):
        end = min(start + batch_size, cache.count)
        hidden = cache.time_hidden[start:end].to(device)
        target_dt = cache.target_dt[start:end].to(device)
        losses = -model.log_f_dt(hidden, target_dt)
        require(bool(torch.isfinite(losses).all()), "Non-finite validation Time NLL")
        total += float(losses.to(torch.float64).sum().item())
        count += int(losses.numel())
    require(count == cache.count, "Validation cache count drift")
    result = total / count
    require(math.isfinite(result), "Validation Time NLL is non-finite")
    return result


@torch.no_grad()
def cached_quantity_predictions(
    *,
    model: SharedTimeCountModel,
    cache: FrozenFeatureCache,
    device: str | torch.device,
    batch_size: int,
) -> torch.Tensor:
    cache.validate(require_quantity=True)
    assert cache.quantity_hidden is not None
    predictions = []
    model.eval()
    for start in range(0, cache.count, batch_size):
        end = min(start + batch_size, cache.count)
        hidden = cache.quantity_hidden[start:end].to(device)
        predictions.append(model.predict_quantity(hidden)[1].detach().cpu())
    return torch.cat(predictions, dim=0).contiguous()


def quantity_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = prediction.to(torch.float64) - target.to(torch.float64)
    return {
        "mae": float(error.abs().mean().item()),
        "rmse": float(torch.square(error).mean().sqrt().item()),
    }


def stratified_quantity_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    body_max: float,
    tail_min_exclusive: float,
    require_nonempty: bool,
) -> dict[str, float | int | None]:
    prediction = prediction.to(torch.float64)
    target = target.to(torch.float64)
    body = target <= float(body_max)
    tail = target > float(tail_min_exclusive)
    if require_nonempty:
        require(bool(body.any()), "Body quantity stratum is empty")
        require(bool(tail.any()), "Tail quantity stratum is empty")
    return {
        "body_count": int(body.sum().item()),
        "body_mae": (
            float((prediction[body] - target[body]).abs().mean().item())
            if bool(body.any()) else None
        ),
        "gt_p99_count": int(tail.sum().item()),
        "gt_p99_mae": (
            float((prediction[tail] - target[tail]).abs().mean().item())
            if bool(tail.any()) else None
        ),
    }


def earliest_strict_minimum(history: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_value = float("inf")
    for row in history:
        value = float(row["val_time_nll"])
        require(math.isfinite(value), "History contains non-finite validation Time NLL")
        if value < best_value:
            best = row
            best_value = value
    if best is None:
        raise ValueError("No finite validation Time NLL candidate")
    return best


def early_stopping_exhausted(
    history: list[dict[str, Any]], *, min_epochs: int, patience: int
) -> bool:
    current_epoch = int(history[-1]["epoch"])
    best_epoch = int(earliest_strict_minimum(history)["epoch"])
    return current_epoch >= min_epochs and current_epoch - best_epoch >= patience


def _finite_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    for state in optimizer.state.values():
        finite_tensor_mapping(state, label="Optimizer state")


def _train_cached_epoch(
    *,
    model: SharedTimeCountModel,
    cache: FrozenFeatureCache,
    optimizer: torch.optim.Optimizer,
    parameters: tuple[torch.nn.Parameter, ...],
    device: str | torch.device,
    batch_size: int,
    grad_clip: float,
    seed: int,
    epoch: int,
) -> dict[str, float]:
    require(not model.training, "Frozen encoder must remain in evaluation mode")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) * 1_000_003 + int(epoch))
    order = torch.randperm(cache.count, generator=generator)
    total = 0.0
    count = 0
    pre_clip_norms = []
    for start in range(0, cache.count, batch_size):
        indices = order[start : start + batch_size]
        hidden = cache.time_hidden[indices].to(device)
        target_dt = cache.target_dt[indices].to(device)
        losses = -model.log_f_dt(hidden, target_dt)
        require(bool(torch.isfinite(losses).all()), "Non-finite train Time NLL")
        loss = losses.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert_time_head_gradient_boundary(model)
        pre_clip = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
        require(bool(torch.isfinite(pre_clip)), "Non-finite time-head gradient norm")
        optimizer.step()
        _finite_optimizer_state(optimizer)
        finite_tensor_mapping(model.state_dict(), label="Model state")
        total += float(losses.detach().to(torch.float64).sum().item())
        count += int(losses.numel())
        pre_clip_norms.append(float(pre_clip.detach().cpu().item()))
    return {
        "train_time_nll": total / count,
        "pre_clip_gradient_norm_mean": float(np.mean(pre_clip_norms)),
        "pre_clip_gradient_norm_max": float(np.max(pre_clip_norms)),
    }


def _resume_identity(
    *,
    contract_sha256: str,
    dataset: str,
    source_checkpoint_sha256: str,
    source_state_sha256: str,
    train_cache_sha256: str,
    validation_cache_sha256: str,
    train_cache_count: int,
    validation_cache_count: int,
    calibration_source_revision: str,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    quantity_replay_batch_size: int,
    grad_clip: float,
    min_epochs: int,
    patience: int,
    planned_epochs: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "contract_sha256": contract_sha256,
        "dataset": dataset,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "train_cache_sha256": train_cache_sha256,
        "validation_cache_sha256": validation_cache_sha256,
        "train_cache_count": int(train_cache_count),
        "validation_cache_count": int(validation_cache_count),
        "calibration_source_revision": calibration_source_revision,
        "seed": int(seed),
        "optimizer": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "quantity_replay_batch_size": int(quantity_replay_batch_size),
        "grad_clip": float(grad_clip),
        "min_epochs": int(min_epochs),
        "patience": int(patience),
        "planned_epochs": int(planned_epochs),
        "selection": SELECTION_RULE,
        "trainable_parameter_names": list(TIME_HEAD_PARAMETER_NAMES),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def _validate_resume_payload(
    payload: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    source_non_time_sha256: str,
) -> None:
    require(payload.get("checkpoint_type") == "time_head_refit_resume", "Wrong resume type")
    require(payload.get("checkpoint_schema_version") == 1, "Wrong resume schema")
    require(payload.get("resume_identity") == dict(identity), "Resume identity drift")
    require(payload.get("evaluation_scope") == "validation_only", "Resume scope drift")
    require(payload.get("held_out_test_evaluated") is False, "Held-out scope drift")
    history = payload.get("history")
    require(isinstance(history, list) and bool(history), "Resume history is missing")
    require(int(history[0]["epoch"]) == 0, "Resume history must begin at epoch zero")
    require(
        [int(row["epoch"]) for row in history] == list(range(len(history))),
        "Resume history epochs are not contiguous",
    )
    require(int(payload.get("epoch", -1)) == len(history) - 1, "Resume epoch drift")
    selected = earliest_strict_minimum(history)
    require(int(payload.get("best_epoch", -1)) == int(selected["epoch"]), "Resume best epoch drift")
    model_state = payload.get("model_state_dict")
    best_state = payload.get("best_state_dict")
    require(isinstance(model_state, dict), "Resume current model state is missing")
    require(isinstance(best_state, dict), "Resume best model state is missing")
    finite_tensor_mapping(model_state, label="Resume current model state")
    finite_tensor_mapping(best_state, label="Resume best model state")
    for label, state in (("current", model_state), ("best", best_state)):
        require(
            state_partition_sha256(state, time_head=False) == source_non_time_sha256,
            f"Resume {label} non-time state drift",
        )
    require(
        canonical_state_dict_sha256(model_state)
        == payload.get("model_state_sha256"),
        "Resume current-state digest drift",
    )
    require(
        canonical_state_dict_sha256(best_state) == payload.get("best_state_sha256"),
        "Resume best-state digest drift",
    )
    optimizer_state = payload.get("optimizer_state_dict")
    require(isinstance(optimizer_state, dict), "Resume optimizer state is missing")
    state_rows = optimizer_state.get("state")
    parameter_groups = optimizer_state.get("param_groups")
    require(
        isinstance(state_rows, dict) and len(state_rows) == len(TIME_HEAD_PARAMETER_NAMES),
        "Resume optimizer does not cover all time-head parameters",
    )
    require(
        isinstance(parameter_groups, list) and len(parameter_groups) == 1,
        "Resume optimizer parameter-group drift",
    )
    parameter_ids = parameter_groups[0].get("params", [])
    require(
        parameter_ids == list(range(len(TIME_HEAD_PARAMETER_NAMES))),
        "Resume optimizer parameter list drift",
    )
    expected_group_values = {
        "lr": float(identity["learning_rate"]),
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": float(identity["weight_decay"]),
        "amsgrad": False,
        "maximize": False,
    }
    for name, expected in expected_group_values.items():
        require(
            parameter_groups[0].get(name) == expected,
            f"Resume AdamW parameter-group drift: {name}",
        )
    optional_group_values = {
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": True,
    }
    for name, expected in optional_group_values.items():
        if name in parameter_groups[0]:
            require(
                parameter_groups[0][name] == expected,
                f"Resume AdamW parameter-group drift: {name}",
            )
    require(
        set(state_rows) == set(parameter_ids),
        "Resume optimizer state parameter IDs drift",
    )
    expected_optimizer_steps = int(payload["epoch"]) * math.ceil(
        int(identity["train_cache_count"]) / int(identity["batch_size"])
    )
    for parameter_id, parameter_name in zip(parameter_ids, TIME_HEAD_PARAMETER_NAMES):
        parameter_state = state_rows[parameter_id]
        require(
            isinstance(parameter_state, dict)
            and {"step", "exp_avg", "exp_avg_sq"}.issubset(parameter_state),
            "Resume AdamW moment state is incomplete",
        )
        expected = model_state[parameter_name]
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            require(
                isinstance(moment, torch.Tensor)
                and moment.shape == expected.shape
                and moment.dtype == expected.dtype,
                f"Resume AdamW {moment_name} shape or dtype drift: {parameter_name}",
            )
        step = parameter_state["step"]
        require(
            isinstance(step, torch.Tensor)
            and step.numel() == 1
            and step.dtype == torch.float32
            and float(step.item()) == float(expected_optimizer_steps),
            f"Resume AdamW step drift: {parameter_name}",
        )
    finite_nested_tensors(optimizer_state, label="Resume optimizer state")


def fit_time_head_from_cache(
    *,
    model: SharedTimeCountModel,
    train_cache: FrozenFeatureCache,
    validation_cache: FrozenFeatureCache,
    output_dir: Path,
    dataset: str,
    contract_sha256: str,
    source_checkpoint_sha256: str,
    source_state_sha256: str,
    calibration_source_revision: str,
    seed: int,
    planned_epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    quantity_replay_batch_size: int,
    grad_clip: float,
    min_epochs: int,
    patience: int,
    device: str | torch.device,
    source_metadata: Mapping[str, Any] | None = None,
    run_epoch_limit: int | None = None,
) -> dict[str, Any]:
    """Fit the isolated head with exact resume and epoch-zero fallback."""
    require(planned_epochs >= 0, "planned_epochs must be nonnegative")
    require(batch_size > 0, "batch_size must be positive")
    require(quantity_replay_batch_size > 0, "quantity replay batch size must be positive")
    require(learning_rate > 0.0 and math.isfinite(learning_rate), "Invalid learning rate")
    require(weight_decay >= 0.0 and math.isfinite(weight_decay), "Invalid weight decay")
    require(grad_clip > 0.0 and math.isfinite(grad_clip), "Invalid gradient clip")
    require(min_epochs >= 0, "min_epochs must be nonnegative")
    require(patience > 0, "patience must be positive")
    train_cache.validate(require_quantity=False)
    validation_cache.validate(require_quantity=True)
    parameters = freeze_time_head_only(model)
    source_state = clone_state_dict(model)
    require(
        canonical_state_dict_sha256(source_state) == source_state_sha256,
        "Loaded source model does not match the declared B state",
    )
    source_non_time_sha256 = state_partition_sha256(source_state, time_head=False)
    source_time_head_sha256 = state_partition_sha256(source_state, time_head=True)
    identity = _resume_identity(
        contract_sha256=contract_sha256,
        dataset=dataset,
        source_checkpoint_sha256=source_checkpoint_sha256,
        source_state_sha256=source_state_sha256,
        train_cache_sha256=train_cache.digest(),
        validation_cache_sha256=validation_cache.digest(),
        train_cache_count=train_cache.count,
        validation_cache_count=validation_cache.count,
        calibration_source_revision=calibration_source_revision,
        seed=seed,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        quantity_replay_batch_size=quantity_replay_batch_size,
        grad_clip=grad_clip,
        min_epochs=min_epochs,
        patience=patience,
        planned_epochs=planned_epochs,
    )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    last_path = output_dir / "last_epoch_state.pt"
    selected_path = output_dir / "best_validation_time_nll_model.pt"
    summary_path = output_dir / "summary.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        require(summary.get("resume_identity") == identity, "Cached summary identity drift")
        require(
            summary.get("evaluation_scope") == "validation_only",
            "Cached summary evaluation scope drift",
        )
        require(
            summary.get("held_out_test_evaluated") is False,
            "Cached summary held-out scope drift",
        )
        require(
            summary.get("quantity_prediction_bitwise_identical") is True,
            "Cached summary did not preserve quantity predictions",
        )
        require(
            summary.get("selected_non_time_state_sha256")
            == source_non_time_sha256,
            "Cached summary non-time state drift",
        )
        require(selected_path.is_file(), "Cached selected checkpoint is missing")
        selected = torch_load_checkpoint(selected_path, map_location="cpu")
        require(
            selected.get("resume_identity") == identity,
            "Cached selected checkpoint identity drift",
        )
        require(
            selected.get("evaluation_scope") == "validation_only",
            "Cached selected checkpoint evaluation scope drift",
        )
        require(
            selected.get("held_out_test_evaluated") is False,
            "Cached selected checkpoint held-out scope drift",
        )
        selected_state = selected.get("model_state_dict")
        require(isinstance(selected_state, dict), "Cached selected state is missing")
        finite_tensor_mapping(selected_state, label="Cached selected model state")
        require(
            state_partition_sha256(selected_state, time_head=False)
            == source_non_time_sha256,
            "Cached selected checkpoint non-time state drift",
        )
        require(
            canonical_state_dict_sha256(selected_state)
            == summary.get("selected_state_sha256"),
            "Cached selected checkpoint digest drift",
        )
        require(
            int(selected.get("best_epoch", -1)) == int(summary.get("best_epoch", -2)),
            "Cached selected checkpoint epoch drift",
        )
        return summary

    history: list[dict[str, Any]]
    best_state: dict[str, torch.Tensor]
    start_epoch: int
    if last_path.exists():
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        _validate_resume_payload(
            payload,
            identity=identity,
            source_non_time_sha256=source_non_time_sha256,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _finite_optimizer_state(optimizer)
        history = list(payload["history"])
        best_state = payload["best_state_dict"]
        start_epoch = int(payload["epoch"]) + 1
    else:
        epoch_zero_nll = evaluate_cached_time_nll(
            model=model,
            cache=validation_cache,
            device=device,
            batch_size=batch_size,
        )
        history = [
            {
                "epoch": 0,
                "train_time_nll": None,
                "val_time_nll": epoch_zero_nll,
                "pre_clip_gradient_norm_mean": None,
                "pre_clip_gradient_norm_max": None,
            }
        ]
        best_state = clone_state_dict(model)
        start_epoch = 1

    stopped_early = early_stopping_exhausted(
        history, min_epochs=min_epochs, patience=patience
    )
    call_last_epoch = planned_epochs
    if run_epoch_limit is not None:
        require(run_epoch_limit >= 0, "run_epoch_limit must be nonnegative")
        call_last_epoch = min(planned_epochs, start_epoch + run_epoch_limit - 1)
    for epoch in range(start_epoch, call_last_epoch + 1):
        if stopped_early:
            break
        telemetry = _train_cached_epoch(
            model=model,
            cache=train_cache,
            optimizer=optimizer,
            parameters=parameters,
            device=device,
            batch_size=batch_size,
            grad_clip=grad_clip,
            seed=seed,
            epoch=epoch,
        )
        require(not model.training, "Encoder mode changed during time-head optimization")
        non_time_sha256 = state_partition_sha256(model.state_dict(), time_head=False)
        require(non_time_sha256 == source_non_time_sha256, "Non-time model state changed")
        val_time_nll = evaluate_cached_time_nll(
            model=model,
            cache=validation_cache,
            device=device,
            batch_size=batch_size,
        )
        history.append(
            {
                "epoch": epoch,
                **telemetry,
                "val_time_nll": val_time_nll,
            }
        )
        selected_row = earliest_strict_minimum(history)
        if int(selected_row["epoch"]) == epoch:
            best_state = clone_state_dict(model)
        current_state = clone_state_dict(model)
        atomic_torch_save(
            {
                "checkpoint_type": "time_head_refit_resume",
                "checkpoint_schema_version": 1,
                "epoch": epoch,
                "model_state_dict": current_state,
                "model_state_sha256": canonical_state_dict_sha256(current_state),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_epoch": int(selected_row["epoch"]),
                "best_state_dict": best_state,
                "best_state_sha256": canonical_state_dict_sha256(best_state),
                "source_non_time_state_sha256": source_non_time_sha256,
                "resume_identity": identity,
                "evaluation_scope": "validation_only",
                "held_out_test_evaluated": False,
            },
            last_path,
        )
        stopped_early = early_stopping_exhausted(
            history, min_epochs=min_epochs, patience=patience
        )

    current_epoch = int(history[-1]["epoch"])
    paused = (
        run_epoch_limit is not None
        and current_epoch < planned_epochs
        and not stopped_early
    )
    if paused:
        return {
            "status": "paused",
            "completed_epochs": current_epoch,
            "resume_identity": identity,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }

    selected_row = earliest_strict_minimum(history)
    model.load_state_dict(best_state, strict=True)
    model.eval()
    selected_state = clone_state_dict(model)
    selected_non_time_sha256 = state_partition_sha256(selected_state, time_head=False)
    require(selected_non_time_sha256 == source_non_time_sha256, "Selected non-time state drift")
    selected_time_head_sha256 = state_partition_sha256(selected_state, time_head=True)
    reevaluated_nll = evaluate_cached_time_nll(
        model=model,
        cache=validation_cache,
        device=device,
        batch_size=batch_size,
    )
    require(
        math.isclose(
            reevaluated_nll,
            float(selected_row["val_time_nll"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Selected Time NLL replay drift",
    )
    final_quantity_prediction = cached_quantity_predictions(
        model=model,
        cache=validation_cache,
        device=device,
        batch_size=quantity_replay_batch_size,
    )
    assert validation_cache.source_quantity_prediction is not None
    assert validation_cache.target_quantity is not None
    quantity_bitwise_identical = torch.equal(
        final_quantity_prediction,
        validation_cache.source_quantity_prediction,
    )
    require(quantity_bitwise_identical, "Quantity predictions changed during time-head refit")
    source_quantity_sha256 = tensor_sha256(
        "quantity_prediction", validation_cache.source_quantity_prediction
    )
    selected_quantity_sha256 = tensor_sha256(
        "quantity_prediction", final_quantity_prediction
    )
    require(source_quantity_sha256 == selected_quantity_sha256, "Quantity digest drift")
    selected_state_sha256 = canonical_state_dict_sha256(selected_state)
    checkpoint = {
        "checkpoint_type": "selected_time_head_refit",
        "checkpoint_schema_version": 1,
        "selection": SELECTION_RULE,
        "selection_formula": "mean(-legacy_clamped_rmtpp_log_density)",
        "best_epoch": int(selected_row["epoch"]),
        "selected_metric_value": reevaluated_nll,
        "model_state_dict": selected_state,
        "model_state_sha256": selected_state_sha256,
        "source_state_sha256": source_state_sha256,
        "source_non_time_state_sha256": source_non_time_sha256,
        "source_time_head_state_sha256": source_time_head_sha256,
        "selected_time_head_state_sha256": selected_time_head_sha256,
        "backbone": (
            source_metadata.get("backbone", SOURCE_BACKBONE)
            if source_metadata is not None else SOURCE_BACKBONE
        ),
        "variant": (
            source_metadata.get("variant", SOURCE_VARIANT)
            if source_metadata is not None else SOURCE_VARIANT
        ),
        "seed": int(
            source_metadata.get("seed", seed)
            if source_metadata is not None else seed
        ),
        "encoder_config": (
            source_metadata.get("encoder_config")
            if source_metadata is not None else None
        ),
        "interface_meta": (
            source_metadata.get("interface_meta")
            if source_metadata is not None else None
        ),
        "train_target_population": (
            source_metadata.get("train_target_population")
            if source_metadata is not None else None
        ),
        "validation_target_population": (
            source_metadata.get("validation_target_population")
            if source_metadata is not None else None
        ),
        "source_checkpoint_lineage": {
            "B_checkpoint_file_sha256": source_checkpoint_sha256,
            "B_checkpoint_state_sha256": source_state_sha256,
            "B_training_source_revision": (
                source_metadata.get("source_revision")
                if source_metadata is not None else None
            ),
            "B_training_source_revision_history": (
                source_metadata.get("source_revision_history")
                if source_metadata is not None else None
            ),
            "calibration_source_revision": calibration_source_revision,
        },
        "resume_identity": identity,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    atomic_torch_save(checkpoint, selected_path)
    epoch_zero_nll = float(history[0]["val_time_nll"])
    summary = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": "success",
        "dataset": dataset,
        "seed": int(seed),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "source_non_time_state_sha256": source_non_time_sha256,
        "source_time_head_state_sha256": source_time_head_sha256,
        "selected_state_sha256": selected_state_sha256,
        "selected_non_time_state_sha256": selected_non_time_sha256,
        "selected_time_head_state_sha256": selected_time_head_sha256,
        "trainable_parameter_names": list(TIME_HEAD_PARAMETER_NAMES),
        "trainable_parameter_count": int(sum(value.numel() for value in parameters)),
        "quantity_replay_batch_size": int(quantity_replay_batch_size),
        "source_checkpoint_lineage": checkpoint["source_checkpoint_lineage"],
        "encoder_config": checkpoint["encoder_config"],
        "interface_meta": checkpoint["interface_meta"],
        "train_target_population": checkpoint["train_target_population"],
        "validation_target_population": checkpoint["validation_target_population"],
        "encoder_mode_during_refit": "eval",
        "hidden_state_gradient": "detached_cache",
        "selection": SELECTION_RULE,
        "best_epoch": int(selected_row["epoch"]),
        "completed_epochs": current_epoch,
        "stopped_early": bool(stopped_early),
        "epoch_zero_val_time_nll": epoch_zero_nll,
        "best_val_time_nll": reevaluated_nll,
        "time_nll_improvement_from_B": epoch_zero_nll - reevaluated_nll,
        "time_nll_non_worse_than_B": reevaluated_nll <= epoch_zero_nll,
        "quantity_prediction_bitwise_identical": quantity_bitwise_identical,
        "source_quantity_prediction_sha256": source_quantity_sha256,
        "selected_quantity_prediction_sha256": selected_quantity_sha256,
        "quantity_metrics": quantity_metrics(
            final_quantity_prediction, validation_cache.target_quantity
        ),
        "train_cache": {
            "count": train_cache.count,
            "sha256": train_cache.digest(),
        },
        "validation_cache": {
            "count": validation_cache.count,
            "sha256": validation_cache.digest(),
        },
        "history": history,
        "resume_identity": identity,
        "selected_checkpoint_path": str(selected_path),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    save_json(summary_path, summary)
    return summary


def validate_source_checkpoint(
    payload: Mapping[str, Any], *, dataset_spec: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    """Reject anything except the exact audited seed42 B checkpoint contract."""
    required = {
        "backbone": SOURCE_BACKBONE,
        "variant": SOURCE_VARIANT,
        "seed": 42,
        "checkpoint_monitor": SOURCE_MONITOR,
        "checkpoint_monitor_history_key": "val_qty_rmse",
        "checkpoint_selection": "best_validation_raw_quantity_rmse",
        "selection": "best_validation_raw_quantity_rmse",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    for name, expected in required.items():
        require(payload.get(name) == expected, f"Source checkpoint {name} mismatch")
    require(
        payload.get("source_revision") == B_TRAINING_SOURCE_REVISION,
        "Source checkpoint training revision mismatch",
    )
    require(
        payload.get("source_revision_history") == [B_TRAINING_SOURCE_REVISION],
        "Source checkpoint training revision history mismatch",
    )
    require(
        math.isclose(
            float(payload.get("selected_metric_value")),
            float(dataset_spec["B_metrics"]["raw_rmse"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "Source checkpoint selected raw RMSE mismatch",
    )
    state = payload.get("model_state_dict")
    require(isinstance(state, dict), "Source checkpoint model state is missing")
    finite_tensor_mapping(state, label="Source checkpoint")
    state_sha256 = canonical_state_dict_sha256(state)
    require(state_sha256 == payload.get("model_state_sha256"), "Source state digest drift")
    require(
        state_sha256 == dataset_spec["B_checkpoint_state_sha256"],
        "Source checkpoint is not the pinned dataset B state",
    )
    encoder = payload.get("encoder_config")
    require(isinstance(encoder, dict), "Source encoder metadata is missing")
    expected_encoder = {
        "backbone_contract_id": "B0",
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "memory_mode": "static_hard_lmm",
        "persistent_mem_size": 16,
        "lmm_mem_size": 64,
        "lmm_topk": 4,
        "time_memory_route": "hard_local_memory_matcher",
        "quantity_memory_route": "shared_memory_state",
    }
    for name, expected in expected_encoder.items():
        require(encoder.get(name) == expected, f"Source encoder {name} mismatch")
    require(
        int(encoder.get("max_len", -1)) == int(dataset_spec["max_sequence_length"]),
        "Source encoder max length mismatch",
    )
    time_head = encoder.get("time_head")
    require(isinstance(time_head, dict), "Source time-head metadata is missing")
    require(
        time_head.get("mode") == TIME_HEAD_MODE_LEGACY_CLAMPED,
        "Source does not use the legacy clamped RMTPP head",
    )
    require(
        math.isclose(float(time_head.get("time_intercept_limit")), 300.0),
        "Source legacy intercept limit mismatch",
    )
    interface = payload.get("interface_meta")
    require(isinstance(interface, dict), "Source interface metadata is missing")
    require(
        interface.get("mode") == "mark_free_count_aware_log_regression",
        "Source quantity interface mismatch",
    )
    require(
        interface.get("target_quantity_masked_from_history") is True,
        "Source target-quantity masking contract mismatch",
    )
    require(set(TIME_HEAD_PARAMETER_NAMES).issubset(state), "Source time-head tensors missing")
    return state


def build_source_model(payload: Mapping[str, Any]) -> SharedTimeCountModel:
    encoder = payload["encoder_config"]
    interface = payload["interface_meta"]
    time_head = encoder["time_head"]
    model, rebuilt_encoder = build_count_aware_model(
        SOURCE_BACKBONE,
        hidden_dim=int(encoder["d_model"]),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface["train_target_std"]),
        max_seq_len=int(encoder["max_len"]),
        quantity_variant=SOURCE_VARIANT,
        lambda_tail=0.0,
        time_head_mode=str(time_head["mode"]),
        time_scale=float(time_head["time_scale"]),
        time_w_max=float(time_head["time_w_max"]),
        time_intercept_limit=float(time_head["time_intercept_limit"]),
        time_initial_intercept=float(
            interface.get("time_head", {}).get("time_initial_intercept", 0.0)
        ),
        time_wd_safety_limit=float(
            interface.get("time_head", {}).get("time_wd_safety_limit", 40.0)
        ),
        time_sigma_floor=float(
            interface.get("time_head", {}).get("time_sigma_floor", 1e-3)
        ),
    )
    require(rebuilt_encoder == encoder, "Rebuilt model metadata differs from checkpoint")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    require(
        canonical_state_dict_sha256(model.state_dict())
        == payload["model_state_sha256"],
        "Strictly loaded model state digest drift",
    )
    return model


def exact_target_population_contract(
    frame: pl.DataFrame,
    *,
    target_split: str,
    lookback: int,
    max_seq_len: int,
) -> dict[str, Any]:
    require(target_split in {"train", "validation"}, "Held-out targets are prohibited")
    dataset = RMTPPWeekLookbackDataset(
        frame,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        val_ratio=0.2,
        mode="all",
        split_col="chronological_split",
        target_splits={target_split},
    )
    require(bool(dataset.index), f"{target_split} target population is empty")
    part_indices = np.fromiter(
        (part_index for part_index, _ in dataset.index),
        dtype=np.int64,
        count=len(dataset.index),
    )
    target_positions = np.fromiter(
        (context_end + 1 for _, context_end in dataset.index),
        dtype=np.int64,
        count=len(dataset.index),
    )
    target_sequences = np.fromiter(
        (
            int(dataset.seq_lists[part_index][context_end + 1])
            for part_index, context_end in dataset.index
        ),
        dtype=np.int64,
        count=len(dataset.index),
    )
    target_quantities = np.fromiter(
        (
            float(dataset.val_lists[part_index][context_end + 1])
            for part_index, context_end in dataset.index
        ),
        dtype=np.float64,
        count=len(dataset.index),
    )
    identity_hasher = hashlib.sha256()
    identity_hasher.update(b"hard_lmm_target_identity_v1\0")
    identity_hasher.update(target_split.encode("utf-8") + b"\0")
    identity_hasher.update(
        json.dumps(
            [str(part) for part in dataset.parts],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for values in (part_indices, target_positions, target_sequences):
        identity_hasher.update(values.astype("<i8", copy=False).tobytes())
    quantity_hasher = hashlib.sha256()
    quantity_hasher.update(b"hard_lmm_target_quantity_v1\0")
    quantity_hasher.update(target_quantities.astype("<f8", copy=False).tobytes())
    return {
        "split": target_split,
        "target_count": len(dataset.index),
        "target_identity_sha256": identity_hasher.hexdigest(),
        "target_quantity_sha256": quantity_hasher.hexdigest(),
    }


def validate_target_population(
    observed: Mapping[str, Any], *, dataset_spec: Mapping[str, Any]
) -> None:
    split = str(observed["split"])
    prefix = f"expected_{split}_target"
    require(
        int(observed["target_count"]) == int(dataset_spec[f"{prefix}s"]),
        f"{split} target count mismatch",
    )
    require(
        observed["target_identity_sha256"]
        == dataset_spec[f"{prefix}_identity_sha256"],
        f"{split} target identity mismatch",
    )
    require(
        observed["target_quantity_sha256"]
        == dataset_spec[f"{prefix}_quantity_sha256"],
        f"{split} target quantity mismatch",
    )


def dataset_by_id(contract: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    rows = [row for row in contract.get("datasets", []) if row.get("dataset") == dataset]
    require(len(rows) == 1, f"Expected one contract row for dataset {dataset!r}")
    return dict(rows[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-source-revision", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=None,
        help="Reuse an already validated train/validation feature cache directory",
    )
    parser.add_argument("--allow-partial-contract", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.perf_counter()
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
        device_name = torch.cuda.get_device_name(args.device)
    else:
        device_name = str(args.device)
    require(
        len(args.calibration_source_revision) == 40,
        "calibration source revision must be a 40-character Git SHA",
    )
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    require(contract.get("contract_id") == CONTRACT_ID, "Wrong time-head refit contract")
    contract_sha256 = sha256_file(args.contract)
    dataset_spec = dataset_by_id(contract, args.dataset)
    require(sha256_file(args.data) == dataset_spec["data_sha256"], "Dataset digest mismatch")
    require(
        sha256_file(args.split_manifest) == dataset_spec["split_manifest_sha256"],
        "Split-manifest digest mismatch",
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    require(
        checkpoint_sha256 == dataset_spec["B_checkpoint_file_sha256"],
        "B checkpoint file digest mismatch",
    )
    source_payload = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    validate_source_checkpoint(source_payload, dataset_spec=dataset_spec)
    model = build_source_model(source_payload).to(args.device)
    time_parameters = freeze_time_head_only(model)
    require(
        sum(value.numel() for value in time_parameters) == 66,
        "Qualified Hard-LMM refit requires exactly 66 time-head parameters",
    )
    frame = prepare_count_frame(load_train_validation_frame(args.data))
    require(
        not bool(frame["chronological_split"].is_in(["test", "held_out"]).any()),
        "Held-out rows entered the refit frame",
    )
    lookback = int(dataset_spec["lookback"])
    max_seq_len = int(dataset_spec["max_sequence_length"])
    train_population = exact_target_population_contract(
        frame,
        target_split="train",
        lookback=lookback,
        max_seq_len=max_seq_len,
    )
    validation_population = exact_target_population_contract(
        frame,
        target_split="validation",
        lookback=lookback,
        max_seq_len=max_seq_len,
    )
    validate_target_population(train_population, dataset_spec=dataset_spec)
    validate_target_population(validation_population, dataset_spec=dataset_spec)

    settings = dict(contract["optimization"])
    partial_values = {
        "max_train_batches": args.max_train_batches,
        "max_validation_batches": args.max_validation_batches,
        "max_epochs": args.max_epochs,
    }
    if any(value is not None for value in partial_values.values()):
        require(
            args.allow_partial_contract,
            "Batch or epoch limits require --allow-partial-contract",
        )
    planned_epochs = int(settings["epochs"])
    if args.max_epochs is not None:
        require(args.max_epochs >= 0, "max epochs must be nonnegative")
        planned_epochs = min(planned_epochs, args.max_epochs)
    encoder_batch_size = int(settings["encoder_batch_size"])
    train_loader = make_loader(
        frame,
        target_split="train",
        batch_size=encoder_batch_size,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        shuffle=False,
        generator=None,
    )
    validation_loader = make_loader(
        frame,
        target_split="validation",
        batch_size=encoder_batch_size,
        lookback_weeks=lookback,
        max_seq_len=max_seq_len,
        shuffle=False,
        generator=None,
    )
    source_state_sha256 = str(source_payload["model_state_sha256"])
    cache_identity_base = {
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "dataset": args.dataset,
        "data_sha256": dataset_spec["data_sha256"],
        "split_manifest_sha256": dataset_spec["split_manifest_sha256"],
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_state_sha256": source_state_sha256,
        "encoder_mode": "eval",
        "target_quantity_masked": True,
        "memory_target_write_masked": True,
        "evaluation_scope": "train_and_validation_only",
        "held_out_test_evaluated": False,
    }
    cache_dir = (
        args.feature_cache_dir
        if args.feature_cache_dir is not None
        else args.output_dir / "cache"
    )
    if args.feature_cache_dir is not None:
        require(
            (cache_dir / "train_features.pt").is_file()
            and (cache_dir / "validation_features.pt").is_file(),
            "Requested shared feature cache is incomplete",
        )
    train_cache = load_or_extract_feature_cache(
        path=cache_dir / "train_features.pt",
        identity={
            **cache_identity_base,
            "split": "train",
            "target_count": train_population["target_count"],
            "target_identity_sha256": train_population["target_identity_sha256"],
            "target_quantity_sha256": train_population["target_quantity_sha256"],
            "max_batches": args.max_train_batches,
        },
        model=model,
        loader=train_loader,
        device=args.device,
        include_quantity=False,
        max_batches=args.max_train_batches,
    )
    validation_cache = load_or_extract_feature_cache(
        path=cache_dir / "validation_features.pt",
        identity={
            **cache_identity_base,
            "split": "validation",
            "target_count": validation_population["target_count"],
            "target_identity_sha256": validation_population["target_identity_sha256"],
            "target_quantity_sha256": validation_population["target_quantity_sha256"],
            "max_batches": args.max_validation_batches,
        },
        model=model,
        loader=validation_loader,
        device=args.device,
        include_quantity=True,
        max_batches=args.max_validation_batches,
    )
    if not args.allow_partial_contract:
        require(
            train_cache.count == train_population["target_count"],
            "Full train cache count mismatch",
        )
        require(
            validation_cache.count == validation_population["target_count"],
            "Full validation cache count mismatch",
        )
    summary = fit_time_head_from_cache(
        model=model,
        train_cache=train_cache,
        validation_cache=validation_cache,
        output_dir=args.output_dir,
        dataset=args.dataset,
        contract_sha256=contract_sha256,
        source_checkpoint_sha256=checkpoint_sha256,
        source_state_sha256=source_state_sha256,
        calibration_source_revision=args.calibration_source_revision,
        seed=int(settings["seed"]),
        planned_epochs=planned_epochs,
        learning_rate=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
        batch_size=int(settings["cached_state_batch_size"]),
        quantity_replay_batch_size=encoder_batch_size,
        grad_clip=float(settings["gradient_clip"]),
        min_epochs=min(int(settings["minimum_epochs"]), planned_epochs),
        patience=int(settings["early_stopping_patience"]),
        device=args.device,
        source_metadata={
            **{
                name: source_payload.get(name)
                for name in (
                "backbone",
                "variant",
                "seed",
                "encoder_config",
                "interface_meta",
                "source_revision",
                "source_revision_history",
                )
            },
            "train_target_population": train_population,
            "validation_target_population": validation_population,
        },
    )
    qualified_full_data = not args.allow_partial_contract
    summary["qualified_full_data"] = qualified_full_data
    summary["feature_cache_dir"] = str(cache_dir.resolve())
    summary["train_target_population"] = train_population
    summary["validation_target_population"] = validation_population
    assert validation_cache.source_quantity_prediction is not None
    assert validation_cache.target_quantity is not None
    summary["quantity_metrics"].update(
        stratified_quantity_metrics(
            validation_cache.source_quantity_prediction,
            validation_cache.target_quantity,
            body_max=float(dataset_spec["reporting_body_max_train_p95"]),
            tail_min_exclusive=float(
                dataset_spec["reporting_tail_min_exclusive_train_p99"]
            ),
            require_nonempty=qualified_full_data,
        )
    )
    stability = contract["identity_and_stability"]
    time_tolerance = float(stability["time_nll_replay_absolute_tolerance"])
    summary["time_guardrails"] = {
        "qualified_comparison": qualified_full_data,
        "B_time_nll": float(dataset_spec["B_metrics"]["time_nll"]),
        "A_plus_0_01_max": float(dataset_spec["time_nll_target_max"]),
        "non_worse_than_B": (
            None
            if not qualified_full_data
            else float(summary["best_val_time_nll"])
            <= float(dataset_spec["B_metrics"]["time_nll"]) + time_tolerance
        ),
        "restores_A_plus_0_01": (
            None
            if not qualified_full_data
            else float(summary["best_val_time_nll"])
            <= float(dataset_spec["time_nll_target_max"]) + time_tolerance
        ),
    }
    if not args.allow_partial_contract:
        require(
            math.isclose(
                float(summary["epoch_zero_val_time_nll"]),
                float(dataset_spec["B_metrics"]["time_nll"]),
                rel_tol=0.0,
                abs_tol=time_tolerance,
            ),
            "Epoch-zero Time NLL did not replay the pinned B metric",
        )
        quantity_absolute_tolerance = float(
            stability["reported_quantity_metric_absolute_tolerance"]
        )
        quantity_relative_tolerance = float(
            stability["reported_quantity_metric_relative_tolerance"]
        )
        for observed_name, expected_name in (
            ("mae", "overall_mae"),
            ("rmse", "raw_rmse"),
            ("body_mae", "body_mae"),
            ("gt_p99_mae", "gt_p99_mae"),
        ):
            require(
                math.isclose(
                    float(summary["quantity_metrics"][observed_name]),
                    float(dataset_spec["B_metrics"][expected_name]),
                    rel_tol=quantity_relative_tolerance,
                    abs_tol=quantity_absolute_tolerance,
                ),
                f"Final quantity {observed_name} did not replay the pinned B metric",
            )
        summary["full_metric_replay"] = {
            "time_nll_absolute_tolerance": time_tolerance,
            "reported_quantity_metric_absolute_tolerance": quantity_absolute_tolerance,
            "reported_quantity_metric_relative_tolerance": quantity_relative_tolerance,
            "epoch_zero_time_nll_matches_B": True,
            "quantity_overall_mae_matches_B": True,
            "quantity_raw_rmse_matches_B": True,
            "quantity_body_mae_matches_B": True,
            "quantity_gt_p99_mae_matches_B": True,
        }
    summary["runtime"] = {
        "requested_device": str(args.device),
        "device_name": device_name,
        "cuda_available": bool(torch.cuda.is_available()),
        "peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(args.device))
            if str(args.device).startswith("cuda") else 0
        ),
        "peak_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(args.device))
            if str(args.device).startswith("cuda") else 0
        ),
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    save_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps({
        "status": summary["status"],
        "dataset": summary["dataset"],
        "best_epoch": summary["best_epoch"],
        "best_val_time_nll": summary["best_val_time_nll"],
        "quantity_prediction_bitwise_identical": summary[
            "quantity_prediction_bitwise_identical"
        ],
        "held_out_test_evaluated": False,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
