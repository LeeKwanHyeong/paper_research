"""Training and checkpoint lifecycle for one count-aware backbone run."""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Any

import numpy as np
import polars as pl
import torch

from models.TPPs.CountAwareFactory import (
    build_count_aware_model,
    validate_checkpoint_route,
)
from models.TPPs.CountAwareTHPStaticMemory import THP_STATIC_MEMORY_BACKBONE
from models.Titan.common.key_value_memory import KEY_VALUE_BACKBONE
from models.Titan.common.elapsed_age import ELAPSED_AGE_BACKBONE
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONE_LABELS,
    CHECKPOINT_HISTORY_RAW_QUANTITY_RMSE,
    CHECKPOINT_MONITOR_JOINT,
    CHECKPOINT_MONITOR_RAW_QUANTITY_RMSE,
    QUANTILE_ADAPTIVE_VARIANT,
    TAIL_VARIANTS,
    VARIANT,
)
from paper.scripts.count_aware_tpp_backbone.core import evaluate, target_outputs
from paper.scripts.run_taxi_quantity_interface_ablation import (
    clone_state_dict,
    make_loader,
    save_json,
    set_seed,
)
from simple_lab_test.search.common.runner import (
    atomic_torch_save,
    canonical_state_dict_sha256,
    capture_rng_state,
    restore_rng_state,
    restore_train_loader_generator_state,
    torch_load_checkpoint,
)


# Keep the historical factory name inside this experiment layer.
build_model = build_count_aware_model


VALIDATION_JOINT_OBJECTIVE = CHECKPOINT_MONITOR_JOINT
VALIDATION_RAW_QUANTITY_RMSE = CHECKPOINT_MONITOR_RAW_QUANTITY_RMSE
DEFAULT_CHECKPOINT_MONITOR = VALIDATION_JOINT_OBJECTIVE

_CHECKPOINT_MONITOR_SPECS = {
    VALIDATION_JOINT_OBJECTIVE: {
        "history_key": "val_joint_objective",
        "evaluation_key": "val_joint_objective",
        "selection": "best_validation_joint_objective",
        "filename": "best_val_joint_objective_model.pt",
    },
    VALIDATION_RAW_QUANTITY_RMSE: {
        "history_key": CHECKPOINT_HISTORY_RAW_QUANTITY_RMSE,
        "evaluation_key": "qty_rmse",
        "selection": "best_validation_raw_quantity_rmse",
        "filename": "best_val_qty_rmse_model.pt",
    },
}

_RESUME_IDENTITY_ARGUMENTS = (
    "epochs",
    "batch_size",
    "lr",
    "lookback_weeks",
    "max_seq_len",
    "hidden_dim",
    "lambda_log_qty",
    "grad_clip",
    "early_stopping_patience",
    "min_epochs",
    "quantity_sigma_floor",
    "lambda_location_huber",
    "location_huber_delta",
    "lambda_tail",
    "tail_threshold",
    "tail_normalization_scale",
    "tail_clip_cap",
    "tail_huber_delta",
    "time_head_mode",
    "time_scale",
    "time_w_max",
    "time_intercept_limit",
    "time_wd_safety_limit",
    "time_head_lr_multiplier",
    "time_sigma_floor",
    "titans_memory_gradient_clip",
    "max_train_batches",
    "max_val_batches",
    "model_role",
    "dataset_contract",
    "execution_role",
    "max_series",
    "allow_partial_contract",
    "quantile_adaptive_strength",
    "device",
    "source_revision",
    "data",
    "data_sha256",
    "split_manifest",
    "split_manifest_sha256",
)


def checkpoint_monitor_spec(monitor: str) -> dict[str, str]:
    """Resolve one registered checkpoint monitor without implicit fallback."""
    try:
        return dict(_CHECKPOINT_MONITOR_SPECS[monitor])
    except KeyError as exc:
        raise ValueError(
            f"Unsupported checkpoint monitor {monitor!r}; expected one of "
            f"{sorted(_CHECKPOINT_MONITOR_SPECS)}"
        ) from exc


def _checkpoint_monitor(args: argparse.Namespace) -> str:
    monitor = getattr(args, "checkpoint_monitor", DEFAULT_CHECKPOINT_MONITOR)
    if not isinstance(monitor, str):
        raise ValueError("checkpoint_monitor must be a string")
    checkpoint_monitor_spec(monitor)
    return monitor


def _selection_formula(quantity_variant: str, monitor: str) -> str:
    if monitor == VALIDATION_RAW_QUANTITY_RMSE:
        return "sqrt(mean((predicted_raw_quantity - raw_quantity)^2))"
    if quantity_variant == VARIANT:
        return "time_nll + lambda_log_qty * log1p_quantity_mse"
    if quantity_variant == QUANTILE_ADAPTIVE_VARIANT:
        return (
            "time_nll + lambda_log_qty * "
            "train_quantile_weighted_log1p_quantity_mse"
        )
    if quantity_variant in TAIL_VARIANTS:
        return (
            "time_nll + lambda_log_qty * "
            "(log1p_quantity_mse + lambda_tail * tail_raw_huber)"
        )
    return (
        "time_nll + lambda_log_qty * "
        "(gaussian_nll_on_log1p_quantity + lambda_location_huber * location_huber)"
    )


def _json_identity(value: Any) -> Any:
    """Normalize paths, tuples, and NumPy scalars for stable identity checks."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _resume_identity(
    *,
    args: argparse.Namespace,
    backbone: str,
    quantity_variant: str,
    seed: int,
    monitor: str,
    quantity_contract: dict[str, Any],
    interface_meta: dict[str, Any],
) -> dict[str, Any]:
    return _json_identity({
        "schema_version": 1,
        "backbone": backbone,
        "variant": quantity_variant,
        "seed": int(seed),
        "checkpoint_monitor": monitor,
        "arguments": {
            name: getattr(args, name, None)
            for name in _RESUME_IDENTITY_ARGUMENTS
        },
        "quantity_contract": quantity_contract,
        "interface_meta": interface_meta,
    })


def _finite_history_value(row: dict[str, Any], metric_key: str) -> float:
    if metric_key not in row:
        raise ValueError(f"History row is missing checkpoint metric {metric_key!r}")
    try:
        value = float(row[metric_key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"History metric {metric_key!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"History metric {metric_key!r} must be finite")
    return value


def earliest_strict_minimum(
    history: list[dict[str, Any]],
    *,
    metric_key: str,
) -> dict[str, Any] | None:
    """Return the earliest row attaining the strict finite minimum."""
    best_row: dict[str, Any] | None = None
    best_value = float("inf")
    for row in history:
        value = _finite_history_value(row, metric_key)
        if best_row is None or value < best_value:
            best_row = row
            best_value = value
    return best_row


def _validate_selector_metadata(
    payload: dict[str, Any],
    *,
    monitor: str,
    allow_legacy_joint: bool,
    artifact: str,
) -> None:
    spec = checkpoint_monitor_spec(monitor)
    expected = {
        "checkpoint_monitor": monitor,
        "checkpoint_monitor_history_key": spec["history_key"],
        "checkpoint_selection": spec["selection"],
    }
    new_keys = tuple(expected)
    if any(key in payload for key in new_keys):
        for key in new_keys:
            value = payload.get(key)
            if value == expected[key]:
                continue
            raise ValueError(
                f"{artifact} checkpoint selector identity mismatch for {key}: "
                f"expected {expected[key]!r}, got {value!r}"
            )
        return

    legacy_selection = payload.get("selection")
    if (
        allow_legacy_joint
        and monitor == VALIDATION_JOINT_OBJECTIVE
        and legacy_selection in (None, spec["selection"])
    ):
        return
    raise ValueError(f"{artifact} has no compatible checkpoint selector identity")


def _validate_cached_checkpoint_digest(
    summary: dict[str, Any],
    checkpoint: dict[str, Any],
) -> None:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Cached checkpoint has no model state")
    actual = canonical_state_dict_sha256(state)
    if checkpoint.get("model_state_sha256") != actual:
        raise ValueError("Cached checkpoint model digest mismatch")
    if summary.get("checkpoint_state_sha256") != actual:
        raise ValueError("Cached summary/checkpoint model digest mismatch")


def _validate_artifact_identity(
    payload: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
    allow_legacy_identity: bool,
    artifact: str,
) -> None:
    for key in ("backbone", "variant", "seed"):
        if payload.get(key) != expected_identity[key]:
            raise ValueError(
                f"{artifact} run identity mismatch for {key}: expected "
                f"{expected_identity[key]!r}, got {payload.get(key)!r}"
            )
    saved_identity = payload.get("resume_identity")
    if saved_identity is None:
        if not allow_legacy_identity:
            raise ValueError(f"{artifact} has no exact run identity")
        return
    if saved_identity != expected_identity:
        raise ValueError(f"{artifact} training identity mismatch")


def _validate_resume_payload(
    payload: dict[str, Any],
    *,
    expected_identity: dict[str, Any],
    expected_initial_state_sha256: str,
    monitor: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if payload.get("checkpoint_type") != "epoch_resume":
        raise ValueError("Resume checkpoint type is invalid")
    if payload.get("checkpoint_schema_version") != 2:
        raise ValueError("Resume checkpoint schema version is invalid")
    _validate_selector_metadata(
        payload,
        monitor=monitor,
        allow_legacy_joint=False,
        artifact="Resume checkpoint",
    )
    _validate_artifact_identity(
        payload,
        expected_identity=expected_identity,
        allow_legacy_identity=False,
        artifact="Resume checkpoint",
    )
    for key in (
        "epoch",
        "model_state_dict",
        "model_state_sha256",
        "optimizer_state_dict",
        "history",
        "best_selection_value",
        "best_epoch",
        "best_state_dict",
        "best_state_sha256",
        "rng_state",
        "train_loader_generator_state",
        "initial_state_sha256",
    ):
        if key not in payload or payload[key] is None:
            raise ValueError(f"Resume checkpoint is missing exact-replay field {key!r}")
    if payload["initial_state_sha256"] != expected_initial_state_sha256:
        raise ValueError("Resume checkpoint initial model state mismatch")
    history = payload["history"]
    if not isinstance(history, list) or not history:
        raise ValueError("Resume checkpoint history must be nonempty")
    saved_epoch = int(payload["epoch"])
    if saved_epoch != int(history[-1].get("epoch", -1)):
        raise ValueError("Resume checkpoint epoch/history mismatch")
    if [int(row.get("epoch", -1)) for row in history] != list(
        range(1, saved_epoch + 1)
    ):
        raise ValueError("Resume checkpoint history epochs must be consecutive")
    spec = checkpoint_monitor_spec(monitor)
    best_row = earliest_strict_minimum(history, metric_key=spec["history_key"])
    if best_row is None:
        raise ValueError("Resume checkpoint has no selectable history row")
    best_value = _finite_history_value(best_row, spec["history_key"])
    if int(payload["best_epoch"]) != int(best_row["epoch"]):
        raise ValueError("Resume checkpoint best epoch is not the earliest strict minimum")
    if float(payload["best_selection_value"]) != best_value:
        raise ValueError("Resume checkpoint best metric/history mismatch")
    if canonical_state_dict_sha256(payload["model_state_dict"]) != payload[
        "model_state_sha256"
    ]:
        raise ValueError("Resume checkpoint current model digest mismatch")
    if canonical_state_dict_sha256(payload["best_state_dict"]) != payload[
        "best_state_sha256"
    ]:
        raise ValueError("Resume checkpoint best model digest mismatch")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, dict) or not {
        "python",
        "numpy",
        "torch",
    }.issubset(rng_state):
        raise ValueError("Resume checkpoint RNG state is invalid")
    resume_device = torch.device(
        str(expected_identity["arguments"].get("device", "cpu"))
    )
    if resume_device.type == "cuda" and "cuda" not in rng_state:
        raise ValueError("CUDA resume checkpoint has no CUDA RNG state")
    generator_state = payload["train_loader_generator_state"]
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("Resume checkpoint train-loader generator state is invalid")
    return history, best_row


def _training_device_metadata(backbone: str, device: str) -> dict[str, Any]:
    device_object = torch.device(device)
    is_cuda = device_object.type == "cuda"
    if not is_cuda and backbone != ELAPSED_AGE_BACKBONE:
        return {}
    return {
        "training_device": str(device),
        "cuda_peak_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(device_object) if is_cuda else None
        ),
        "cuda_peak_memory_reserved_bytes": (
            torch.cuda.max_memory_reserved(device_object) if is_cuda else None
        ),
    }


def _quantity_variant_model_kwargs(
    quantity_variant: str,
    interface_meta: dict[str, Any],
) -> dict[str, Any]:
    if quantity_variant != QUANTILE_ADAPTIVE_VARIANT:
        return {}
    contract = interface_meta.get("quantile_adaptive_contract")
    if not isinstance(contract, dict):
        raise ValueError(
            "Quantile-adaptive training requires quantile_adaptive_contract metadata"
        )
    try:
        strength = float(interface_meta["quantile_adaptive_strength"])
        boundaries = tuple(float(value) for value in contract["boundaries"])
        weights = tuple(
            float(value) for value in contract["normalized_bin_weights"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid quantile-adaptive interface metadata") from exc
    return {
        "quantile_adaptive_strength": strength,
        "quantile_adaptive_boundaries": boundaries,
        "quantile_adaptive_weights": weights,
    }


def build_optimizer(
    model: torch.nn.Module,
    *,
    lr: float,
    time_head_lr_multiplier: float = 1.0,
) -> torch.optim.AdamW:
    """Build AdamW while optionally lowering only the shared time-head LR."""
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if (
        not math.isfinite(time_head_lr_multiplier)
        or time_head_lr_multiplier <= 0.0
    ):
        raise ValueError("time_head_lr_multiplier must be finite and positive")

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if math.isclose(time_head_lr_multiplier, 1.0, rel_tol=0.0, abs_tol=1e-15):
        # Preserve the historical one-group optimizer contract for H0/H1.
        return torch.optim.AdamW(trainable, lr=lr)

    if not hasattr(model, "time_head_named_parameters"):
        raise TypeError("Model does not expose time_head_named_parameters()")
    time_parameters = [
        parameter
        for _, parameter in model.time_head_named_parameters()
        if parameter.requires_grad
    ]
    time_parameter_ids = {id(parameter) for parameter in time_parameters}
    base_parameters = [
        parameter for parameter in trainable if id(parameter) not in time_parameter_ids
    ]
    if not base_parameters or not time_parameters:
        raise ValueError(
            "Optimizer split requires nonempty base and time parameter groups"
        )
    return torch.optim.AdamW(
        [
            {
                "params": base_parameters,
                "lr": lr,
                "group_name": "backbone_and_quantity",
            },
            {
                "params": time_parameters,
                "lr": lr * time_head_lr_multiplier,
                "group_name": "time_head",
            },
        ],
        lr=lr,
    )


def optimizer_group_contract(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    """Return optimizer-group metadata without serializing parameter objects."""
    return [
        {
            "index": index,
            "group_name": str(group.get("group_name", "all_parameters")),
            "lr": float(group["lr"]),
            "parameter_count": int(
                sum(parameter.numel() for parameter in group["params"])
            ),
        }
        for index, group in enumerate(optimizer.param_groups)
    ]


def train_epoch_with_telemetry(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: str,
    lambda_log_qty: float,
    grad_clip: float,
    max_batches: int | None,
) -> dict[str, Any]:
    """Train one epoch and record pre-clipping stability telemetry."""
    if not math.isfinite(grad_clip) or grad_clip <= 0.0:
        raise ValueError("grad_clip must be finite and positive")

    model.train()
    event_count = 0
    joint_sum = 0.0
    time_sum = 0.0
    quantity_sum = 0.0
    batch_joint_means: list[float] = []
    gradient_norms: list[float] = []
    clipping_count = 0
    max_per_event_time_nll = -float("inf")

    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        if quantities is None:
            raise ValueError("Count-aware training requires raw quantities")
        outputs = target_outputs(
            model,
            dts.to(device),
            mask.to(device),
            quantities.to(device),
            lambda_log_qty=lambda_log_qty,
        )
        tracked_outputs = {
            "joint_loss": outputs["joint_loss"],
            "time_loss": outputs["time_loss"],
            "quantity_train_loss": outputs["quantity_train_loss"],
        }
        nonfinite_outputs = [
            name
            for name, value in tracked_outputs.items()
            if not bool(torch.isfinite(value).all())
        ]
        if nonfinite_outputs:
            raise FloatingPointError(
                "Non-finite train outputs at batch "
                f"{batch_index}: {','.join(nonfinite_outputs)}; "
                f"time_head={model.time_head_telemetry()}"
            )

        loss = outputs["joint_loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_clip_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
            error_if_nonfinite=False,
        )
        pre_clip_norm_value = float(pre_clip_norm.detach().cpu().item())
        if not math.isfinite(pre_clip_norm_value):
            raise FloatingPointError(
                f"Non-finite pre-clipping gradient norm at batch {batch_index}"
            )
        optimizer.step()
        if not all(
            bool(torch.isfinite(parameter).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError(
                f"Non-finite model parameter after batch {batch_index}"
            )

        joint = outputs["joint_loss"].detach().double()
        time_loss = outputs["time_loss"].detach().double()
        quantity_loss = outputs["quantity_train_loss"].detach().double()
        current_count = int(joint.numel())
        event_count += current_count
        joint_sum += float(joint.sum().item())
        time_sum += float(time_loss.sum().item())
        quantity_sum += float(quantity_loss.sum().item())
        batch_joint_means.append(float(joint.mean().item()))
        gradient_norms.append(pre_clip_norm_value)
        clipping_count += int(pre_clip_norm_value > grad_clip)
        max_per_event_time_nll = max(
            max_per_event_time_nll,
            float(time_loss.max().item()),
        )

    if event_count < 1 or not batch_joint_means:
        raise ValueError("No train batches were evaluated")
    batch_means = np.asarray(batch_joint_means, dtype=np.float64)
    grad_norms = np.asarray(gradient_norms, dtype=np.float64)
    telemetry = {
        "train_joint_objective": joint_sum / event_count,
        "train_time_nll": time_sum / event_count,
        "train_quantity_loss": quantity_sum / event_count,
        "train_batch_joint_p50": float(np.quantile(batch_means, 0.50)),
        "train_batch_joint_p95": float(np.quantile(batch_means, 0.95)),
        "train_batch_joint_p99": float(np.quantile(batch_means, 0.99)),
        "train_batch_joint_max": float(batch_means.max()),
        "train_max_per_event_time_nll": max_per_event_time_nll,
        "train_pre_clip_grad_norm_mean": float(grad_norms.mean()),
        "train_pre_clip_grad_norm_max": float(grad_norms.max()),
        "train_gradient_clip_count": clipping_count,
        "train_gradient_clip_fraction": clipping_count / len(batch_joint_means),
        "train_event_count": event_count,
        "train_batch_count": len(batch_joint_means),
        "train_all_finite": True,
        **model.time_head_telemetry(),
    }
    if not all(
        math.isfinite(float(value))
        for key, value in telemetry.items()
        if key
        not in {
            "train_gradient_clip_count",
            "train_event_count",
            "train_batch_count",
            "train_all_finite",
        }
    ):
        raise FloatingPointError("Non-finite train telemetry")
    return telemetry


def early_stopping_exhausted(
    history: list[dict[str, Any]],
    *,
    min_epochs: int,
    patience: int,
    metric_key: str = "val_joint_objective",
) -> bool:
    if not history or patience < 1:
        return False
    current_epoch = int(history[-1]["epoch"])
    best_row = earliest_strict_minimum(history, metric_key=metric_key)
    if best_row is None:
        return False
    best_epoch = int(best_row["epoch"])
    return current_epoch >= min_epochs and current_epoch - best_epoch >= patience


def train_one(
    *,
    args: argparse.Namespace,
    frame: pl.DataFrame,
    quantity_contract: dict[str, Any],
    interface_meta: dict[str, Any],
    backbone: str,
    quantity_variant: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    monitor = _checkpoint_monitor(args)
    monitor_spec = checkpoint_monitor_spec(monitor)
    monitor_history_key = monitor_spec["history_key"]
    selection_formula = _selection_formula(quantity_variant, monitor)
    resume_identity = _resume_identity(
        args=args,
        backbone=backbone,
        quantity_variant=quantity_variant,
        seed=seed,
        monitor=monitor,
        quantity_contract=quantity_contract,
        interface_meta=interface_meta,
    )
    run_dir = args.output_dir / "runs" / backbone / quantity_variant / f"seed_{seed}"
    if backbone == ELAPSED_AGE_BACKBONE and run_dir.exists():
        raise FileExistsError("Elapsed-age candidate requires a fresh run; automatic reuse/resume/overwrite is forbidden")
    if backbone == KEY_VALUE_BACKBONE and run_dir.exists():
        raise FileExistsError("Separate-key candidate requires a fresh run; automatic reuse/resume/overwrite is forbidden")
    if backbone == THP_STATIC_MEMORY_BACKBONE and run_dir.exists():
        raise FileExistsError("THP static memory requires a fresh run; cache reuse/resume/overwrite is forbidden")
    summary_path = run_dir / "summary.json"
    best_path = run_dir / monitor_spec["filename"]
    last_path = run_dir / "last_epoch_state.pt"
    other_best_paths = [
        run_dir / spec["filename"]
        for registered_monitor, spec in _CHECKPOINT_MONITOR_SPECS.items()
        if registered_monitor != monitor
    ]
    if not args.force_rerun and summary_path.exists():
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validate_checkpoint_route(cached_summary, backbone)
        _validate_selector_metadata(
            cached_summary,
            monitor=monitor,
            allow_legacy_joint=True,
            artifact="Cached summary",
        )
        _validate_artifact_identity(
            cached_summary,
            expected_identity=resume_identity,
            allow_legacy_identity=(monitor == VALIDATION_JOINT_OBJECTIVE),
            artifact="Cached summary",
        )
        if not best_path.exists():
            raise ValueError(
                "Cached summary is missing its selected checkpoint; refusing reuse"
            )
        cached_checkpoint = torch_load_checkpoint(best_path, map_location="cpu")
        validate_checkpoint_route(cached_checkpoint, backbone)
        _validate_selector_metadata(
            cached_checkpoint,
            monitor=monitor,
            allow_legacy_joint=True,
            artifact="Cached checkpoint",
        )
        _validate_artifact_identity(
            cached_checkpoint,
            expected_identity=resume_identity,
            allow_legacy_identity=(monitor == VALIDATION_JOINT_OBJECTIVE),
            artifact="Cached checkpoint",
        )
        _validate_cached_checkpoint_digest(cached_summary, cached_checkpoint)
        payload = cached_summary
        quantity_rows = payload.pop("quantity_rows")
        history_rows = payload.pop("history_rows")
        return payload, quantity_rows, history_rows
    if not args.force_rerun and any(path.exists() for path in other_best_paths):
        raise ValueError(
            "Run directory contains a checkpoint from a different selector; "
            "use a distinct output directory"
        )

    generator = set_seed(seed)
    train_loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=generator,
    )
    val_loader = make_loader(
        frame,
        target_split="validation",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=False,
        generator=None,
    )
    model, encoder_config = build_model(
        backbone,
        hidden_dim=args.hidden_dim,
        train_log_mean=float(interface_meta["train_target_mean"]),
        train_log_std=float(interface_meta["train_target_std"]),
        max_seq_len=args.max_seq_len,
        quantity_variant=quantity_variant,
        quantity_sigma_floor=args.quantity_sigma_floor,
        lambda_location_huber=args.lambda_location_huber,
        location_huber_delta=args.location_huber_delta,
        lambda_tail=args.lambda_tail,
        tail_threshold=args.tail_threshold,
        tail_normalization_scale=args.tail_normalization_scale,
        tail_clip_cap=args.tail_clip_cap,
        tail_huber_delta=args.tail_huber_delta,
        time_head_mode=args.time_head_mode,
        time_scale=args.time_scale,
        time_w_max=args.time_w_max,
        time_intercept_limit=args.time_intercept_limit,
        time_initial_intercept=float(
            interface_meta["time_head"].get("time_initial_intercept", 0.0)
        ),
        time_wd_safety_limit=args.time_wd_safety_limit,
        time_initial_location=interface_meta["time_head"].get(
            "time_initial_location"
        ),
        time_initial_scale=interface_meta["time_head"].get(
            "time_initial_scale"
        ),
        time_sigma_floor=args.time_sigma_floor,
        titans_memory_gradient_clip=getattr(args, "titans_memory_gradient_clip", None),
        **_quantity_variant_model_kwargs(quantity_variant, interface_meta),
    )
    initial_state_sha256 = canonical_state_dict_sha256(model.state_dict())
    model.to(args.device)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        time_head_lr_multiplier=args.time_head_lr_multiplier,
    )
    optimizer_contract = optimizer_group_contract(optimizer)
    history: list[dict[str, Any]] = []
    best_selection_value = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    source_revision_history = [args.source_revision]
    start_epoch = 1

    if last_path.exists() and not args.force_rerun:
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        validate_checkpoint_route(payload, backbone)
        history, saved_best_row = _validate_resume_payload(
            payload,
            expected_identity=resume_identity,
            expected_initial_state_sha256=initial_state_sha256,
            monitor=monitor,
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(history)
        best_selection_value = float(payload["best_selection_value"])
        best_epoch = int(saved_best_row["epoch"])
        best_state = payload["best_state_dict"]
        start_epoch = int(payload["epoch"]) + 1
        source_revision_history = [
            revision for revision in payload.get("source_revision_history", []) if revision
        ]
        if args.source_revision not in source_revision_history:
            source_revision_history.append(args.source_revision)
        restore_rng_state(payload["rng_state"])
        restore_train_loader_generator_state(
            generator,
            payload["train_loader_generator_state"],
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    started = time.time()
    stopped_early = early_stopping_exhausted(
        history,
        min_epochs=args.min_epochs,
        patience=args.early_stopping_patience,
        metric_key=monitor_history_key,
    )
    for epoch in range(start_epoch, args.epochs + 1):
        if stopped_early:
            break
        train_telemetry = train_epoch_with_telemetry(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            lambda_log_qty=args.lambda_log_qty,
            grad_clip=args.grad_clip,
            max_batches=args.max_train_batches,
        )

        validation = evaluate(
            model=model,
            loader=val_loader,
            quantity_contract=quantity_contract,
            device=args.device,
            lambda_log_qty=args.lambda_log_qty,
            max_batches=args.max_val_batches,
            include_breakdowns=False,
        )
        epoch_row = {
            "epoch": epoch,
            **train_telemetry,
            "val_joint_objective": float(validation["val_joint_objective"]),
            "val_time_nll": float(validation["val_time_nll"]),
            "val_quantity_train_loss": float(validation["val_quantity_train_loss"]),
            "val_log_qty_mse": float(validation["val_log_qty_mse"]),
            "val_quantity_distribution_nll": float(
                validation["val_quantity_distribution_nll"]
            ),
            "val_quantity_location_huber": float(
                validation["val_quantity_location_huber"]
            ),
            "val_tail_aux_loss": float(validation["val_tail_aux_loss"]),
            "val_tail_count": int(validation["val_tail_count"]),
            "val_quantity_scale_mean": float(validation["val_quantity_scale_mean"]),
            "val_qty_mae": float(validation["qty_mae"]),
            "val_qty_rmse": float(validation["qty_rmse"]),
        }
        history.append(epoch_row)
        line = (
            f"[epoch {epoch:03d}] backbone={backbone} "
            f"variant={quantity_variant} seed={seed} "
            f"train_joint={epoch_row['train_joint_objective']:.8f} "
            f"train_time={epoch_row['train_time_nll']:.8f} "
            f"batch_p99={epoch_row['train_batch_joint_p99']:.8f} "
            f"grad_norm={epoch_row['train_pre_clip_grad_norm_mean']:.8f} "
            f"clip_fraction={epoch_row['train_gradient_clip_fraction']:.6f} "
            f"val_joint={epoch_row['val_joint_objective']:.8f} "
            f"time_nll={epoch_row['val_time_nll']:.8f} "
            f"log_qty_mse={epoch_row['val_log_qty_mse']:.8f} "
            f"tail_aux={epoch_row['val_tail_aux_loss']:.8f} "
            f"qty_mae={epoch_row['val_qty_mae']:.8f}"
        )
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        current_selection_value = _finite_history_value(
            epoch_row,
            monitor_history_key,
        )
        if current_selection_value < best_selection_value:
            best_selection_value = current_selection_value
            best_epoch = epoch
            best_state = clone_state_dict(model)
        save_json(run_dir / "history.json", {"history": history})
        current_state = clone_state_dict(model)
        if best_state is None or best_epoch is None:
            raise RuntimeError("No finite checkpoint metric was available")
        selected_best_row = earliest_strict_minimum(
            history,
            metric_key=monitor_history_key,
        )
        if selected_best_row is None:
            raise RuntimeError("No finite checkpoint history was available")
        atomic_torch_save({
            "checkpoint_type": "epoch_resume",
            "checkpoint_schema_version": 2,
            "epoch": epoch,
            "backbone": backbone,
            "variant": quantity_variant,
            "seed": seed,
            "checkpoint_monitor": monitor,
            "checkpoint_monitor_history_key": monitor_history_key,
            "checkpoint_selection": monitor_spec["selection"],
            "selection": monitor_spec["selection"],
            "selection_formula": selection_formula,
            "resume_identity": resume_identity,
            "initial_state_sha256": initial_state_sha256,
            "model_state_dict": current_state,
            "model_state_sha256": canonical_state_dict_sha256(current_state),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_selection_value": best_selection_value,
            "best_epoch": best_epoch,
            "best_val_joint_objective": _finite_history_value(
                selected_best_row,
                "val_joint_objective",
            ),
            "best_state_dict": best_state,
            "best_state_sha256": canonical_state_dict_sha256(best_state),
            "encoder_config": encoder_config,
            "interface_meta": interface_meta,
            "optimizer_group_contract": optimizer_contract,
            "source_revision": args.source_revision,
            "source_revision_history": source_revision_history,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "rng_state": capture_rng_state(),
            "train_loader_generator_state": generator.get_state(),
        }, last_path)
        stopped_early = early_stopping_exhausted(
            history,
            min_epochs=args.min_epochs,
            patience=args.early_stopping_patience,
            metric_key=monitor_history_key,
        )
        if stopped_early:
            selected_row = earliest_strict_minimum(
                history,
                metric_key=monitor_history_key,
            )
            if selected_row is None:
                raise RuntimeError("No checkpoint metric was available for early stopping")
            print(
                f"[early-stop] backbone={backbone} variant={quantity_variant} seed={seed} "
                f"monitor={monitor} current_epoch={epoch} "
                f"best_epoch={selected_row['epoch']}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError(
            f"No best checkpoint selected for {backbone}/{quantity_variant}/seed_{seed}"
        )
    model.load_state_dict(best_state, strict=True)
    validation = evaluate(
        model=model,
        loader=val_loader,
        quantity_contract=quantity_contract,
        device=args.device,
        lambda_log_qty=args.lambda_log_qty,
        max_batches=args.max_val_batches,
        include_breakdowns=args.max_val_batches is None,
    )
    state_digest = canonical_state_dict_sha256(best_state)
    selected_row = earliest_strict_minimum(
        history,
        metric_key=monitor_history_key,
    )
    if selected_row is None:
        raise RuntimeError("No checkpoint metric was available after training")
    best_epoch = int(selected_row["epoch"])
    selected_metric_value = _finite_history_value(
        selected_row,
        monitor_history_key,
    )
    reevaluated_metric_value = float(validation[monitor_spec["evaluation_key"]])
    if not math.isfinite(reevaluated_metric_value):
        raise ValueError("Selected checkpoint metric must be finite")
    if not math.isclose(
        reevaluated_metric_value,
        selected_metric_value,
        rel_tol=1e-10,
        abs_tol=1e-8,
    ):
        raise ValueError("Selected checkpoint metric does not replay its history value")
    checkpoint = {
        "selection": monitor_spec["selection"],
        "selection_formula": selection_formula,
        "checkpoint_monitor": monitor,
        "checkpoint_monitor_history_key": monitor_history_key,
        "checkpoint_selection": monitor_spec["selection"],
        "best_epoch": best_epoch,
        "selected_metric_value": selected_metric_value,
        "resume_identity": resume_identity,
        "initial_state_sha256": initial_state_sha256,
        "backbone": backbone,
        "variant": quantity_variant,
        "seed": seed,
        "model_state_dict": best_state,
        "model_state_sha256": state_digest,
        "encoder_config": encoder_config,
        "interface_meta": interface_meta,
        "optimizer_group_contract": optimizer_contract,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }
    atomic_torch_save(checkpoint, best_path)
    quantity_rows = [{
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": quantity_variant,
        "seed": seed,
        **row,
    } for row in validation.get("quantity_rows", [])]
    history_rows = [{
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": quantity_variant,
        "seed": seed,
        **row,
    } for row in validation.get("history_rows", [])]
    summary = {
        "status": "success",
        "backbone": backbone,
        "backbone_label": BACKBONE_LABELS[backbone],
        "variant": quantity_variant,
        "seed": seed,
        "epochs": args.epochs,
        "completed_epochs": int(history[-1]["epoch"]),
        "stopped_early": int(history[-1]["epoch"]) < args.epochs,
        "best_epoch": best_epoch,
        "checkpoint_monitor": monitor,
        "checkpoint_monitor_history_key": monitor_history_key,
        "checkpoint_selection": monitor_spec["selection"],
        "selection_formula": selection_formula,
        "selected_metric_value": selected_metric_value,
        "best_val_joint_objective": float(validation["val_joint_objective"]),
        "best_val_time_nll": float(validation["val_time_nll"]),
        "best_val_quantity_train_loss": float(validation["val_quantity_train_loss"]),
        "best_val_log_qty_mse": float(validation["val_log_qty_mse"]),
        "best_val_quantity_distribution_nll": float(
            validation["val_quantity_distribution_nll"]
        ),
        "best_val_quantity_location_huber": float(
            validation["val_quantity_location_huber"]
        ),
        "best_val_tail_aux_loss": float(validation["val_tail_aux_loss"]),
        "best_val_tail_count": int(validation["val_tail_count"]),
        "best_val_quantity_scale_mean": float(validation["val_quantity_scale_mean"]),
        "lambda_tail": args.lambda_tail,
        "tail_threshold": args.tail_threshold,
        "tail_normalization_scale": args.tail_normalization_scale,
        "tail_clip_cap": args.tail_clip_cap,
        "tail_huber_delta": args.tail_huber_delta,
        "best_val_qty_mae": float(validation["qty_mae"]),
        "best_val_qty_rmse": float(validation["qty_rmse"]),
        "parameter_count": parameter_count,
        "source_revision": args.source_revision,
        "source_revision_history": source_revision_history,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checkpoint_path": str(best_path),
        "checkpoint_state_sha256": state_digest,
        "elapsed_seconds": time.time() - started,
        "encoder_config": encoder_config,
        "interface_meta": interface_meta,
        "optimizer_group_contract": optimizer_contract,
        "resume_identity": resume_identity,
        "initial_state_sha256": initial_state_sha256,
        "quantity_rows": quantity_rows,
        "history_rows": history_rows,
    }
    summary.update(_training_device_metadata(backbone, args.device))
    save_json(summary_path, summary)
    returned = dict(summary)
    returned.pop("quantity_rows")
    returned.pop("history_rows")
    return returned, quantity_rows, history_rows




__all__ = [
    "DEFAULT_CHECKPOINT_MONITOR",
    "VALIDATION_JOINT_OBJECTIVE",
    "VALIDATION_RAW_QUANTITY_RMSE",
    "build_optimizer",
    "checkpoint_monitor_spec",
    "earliest_strict_minimum",
    "early_stopping_exhausted",
    "optimizer_group_contract",
    "train_epoch_with_telemetry",
    "train_one",
]
