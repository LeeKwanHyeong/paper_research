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

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONE_LABELS,
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
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


# Keep the historical factory name inside this experiment layer.
build_model = build_count_aware_model


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
        tracked_outputs = (
            outputs["joint_loss"],
            outputs["time_loss"],
            outputs["quantity_train_loss"],
        )
        if not all(bool(torch.isfinite(value).all()) for value in tracked_outputs):
            raise FloatingPointError(
                f"Non-finite train loss at batch {batch_index}"
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
    slope = float(model.positive_time_slope().detach().cpu().item())
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
        "train_time_slope": slope,
        "train_all_finite": True,
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
) -> bool:
    if not history or patience < 1:
        return False
    current_epoch = int(history[-1]["epoch"])
    best_epoch = int(min(history, key=lambda row: float(row["val_joint_objective"]))["epoch"])
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
    run_dir = args.output_dir / "runs" / backbone / quantity_variant / f"seed_{seed}"
    summary_path = run_dir / "summary.json"
    best_path = run_dir / "best_val_joint_objective_model.pt"
    last_path = run_dir / "last_epoch_state.pt"
    if summary_path.exists() and best_path.exists() and not args.force_rerun:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        quantity_rows = payload.pop("quantity_rows")
        history_rows = payload.pop("history_rows")
        return payload, quantity_rows, history_rows

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
            interface_meta["time_head"]["time_initial_intercept"]
        ),
        time_wd_safety_limit=args.time_wd_safety_limit,
    )
    model.to(args.device)
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
    best_objective = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    source_revision_history = [args.source_revision]
    start_epoch = 1

    if last_path.exists() and not args.force_rerun:
        payload = torch_load_checkpoint(last_path, map_location="cpu")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        history = list(payload.get("history", []))
        best_objective = float(payload.get("best_val_joint_objective", best_objective))
        best_state = payload.get("best_state_dict")
        start_epoch = int(payload["epoch"]) + 1
        source_revision_history = [
            revision for revision in payload.get("source_revision_history", []) if revision
        ]
        if args.source_revision not in source_revision_history:
            source_revision_history.append(args.source_revision)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    started = time.time()
    stopped_early = early_stopping_exhausted(
        history,
        min_epochs=args.min_epochs,
        patience=args.early_stopping_patience,
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
        if epoch_row["val_joint_objective"] < best_objective:
            best_objective = epoch_row["val_joint_objective"]
            best_state = clone_state_dict(model)
        save_json(run_dir / "history.json", {"history": history})
        torch.save({
            "epoch": epoch,
            "backbone": backbone,
            "variant": quantity_variant,
            "seed": seed,
            "model_state_dict": clone_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_val_joint_objective": best_objective,
            "best_state_dict": best_state,
            "encoder_config": encoder_config,
            "interface_meta": interface_meta,
            "optimizer_group_contract": optimizer_contract,
            "source_revision": args.source_revision,
            "source_revision_history": source_revision_history,
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
        }, last_path)
        stopped_early = early_stopping_exhausted(
            history,
            min_epochs=args.min_epochs,
            patience=args.early_stopping_patience,
        )
        if stopped_early:
            best_epoch = min(
                history,
                key=lambda row: float(row["val_joint_objective"]),
            )["epoch"]
            print(
                f"[early-stop] backbone={backbone} variant={quantity_variant} seed={seed} "
                f"current_epoch={epoch} best_epoch={best_epoch}",
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
    if quantity_variant == VARIANT:
        selection_formula = "time_nll + lambda_log_qty * log1p_quantity_mse"
    elif quantity_variant in TAIL_VARIANTS:
        selection_formula = (
            "time_nll + lambda_log_qty * "
            "(log1p_quantity_mse + lambda_tail * tail_raw_huber)"
        )
    else:
        selection_formula = (
            "time_nll + lambda_log_qty * "
            "(gaussian_nll_on_log1p_quantity + lambda_location_huber * location_huber)"
        )
    checkpoint = {
        "selection": "best_validation_joint_objective",
        "selection_formula": selection_formula,
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
    torch.save(checkpoint, best_path)
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
    best_epoch = int(min(
        history,
        key=lambda row: float(row["val_joint_objective"]),
    )["epoch"])
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
        "quantity_rows": quantity_rows,
        "history_rows": history_rows,
    }
    save_json(summary_path, summary)
    returned = dict(summary)
    returned.pop("quantity_rows")
    returned.pop("history_rows")
    return returned, quantity_rows, history_rows




__all__ = [
    "build_optimizer",
    "early_stopping_exhausted",
    "optimizer_group_contract",
    "train_epoch_with_telemetry",
    "train_one",
]
