#!/usr/bin/env python3
"""Audit time-slope pressure and multi-task gradients on train rows only."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    SharedTimeCountModel,
    TAIL_SHARED_VARIANT,
    TIME_HEAD_MODE_SCALED_EXACT,
    TIME_HEAD_MODE_SCALED_EXACT_STABLE,
)
from paper.scripts.count_aware_tpp_backbone.constants import FROZEN_TAIL_LAMBDA
from paper.scripts.count_aware_tpp_backbone.core import (
    prepare_count_frame,
    right_pad_batch,
)
from paper.scripts.count_aware_tpp_backbone.reporting import write_csv
from paper.scripts.count_aware_tpp_backbone.training import (
    build_optimizer,
    train_epoch_with_telemetry,
)
from paper.scripts.run_count_aware_tpp_backbone_control import (
    DATASET_CONTRACTS,
    STABLE_TIME_INTERCEPT_LIMIT,
    STABLE_TIME_WD_SAFETY_LIMIT,
    TIME_WD_SAFETY_LIMIT,
    derive_train_time_contract,
)
from paper.scripts.run_taxi_quantity_interface_ablation import (
    make_loader,
    save_json,
    set_seed,
    sha256_file,
)


BACKBONE = "titantpp"
SEED = 42
VARIANT_SPECS = {
    "H0": {
        "description": "scaled_exact_budget_40",
        "time_head_mode": TIME_HEAD_MODE_SCALED_EXACT,
        "time_wd_safety_limit": TIME_WD_SAFETY_LIMIT,
        "time_intercept_limit": 30.0,
    },
    "H1": {
        "description": "stable_exact_budget_8",
        "time_head_mode": TIME_HEAD_MODE_SCALED_EXACT_STABLE,
        "time_wd_safety_limit": STABLE_TIME_WD_SAFETY_LIMIT,
        "time_intercept_limit": STABLE_TIME_INTERCEPT_LIMIT,
    },
}
SLOPE_SATURATION_RATIO = 0.98
SLOPE_UPWARD_PRESSURE_FRACTION = 0.50
GRADIENT_CONFLICT_COSINE = -0.10
GRADIENT_CONFLICT_FRACTION = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--execution-role", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback-weeks", type=int, default=520)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--audit-batches", type=int, default=32)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def load_train_only_frame(path: Path) -> pl.DataFrame:
    """Read only the official train partition before any model selection."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )


def shared_encoder_parameters(
    model: torch.nn.Module,
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Return the Titan encoder and Hard-LMM parameters shared by both tasks."""
    selected = tuple(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("encoder.") or name.startswith("lmm."))
    )
    if not selected:
        raise ValueError("No shared Titan encoder parameters were found")
    return selected


def gradient_pair_statistics(
    first: Iterable[torch.Tensor | None],
    second: Iterable[torch.Tensor | None],
) -> dict[str, float]:
    """Calculate norm and cosine statistics without flattening large tensors."""
    dot = torch.zeros((), dtype=torch.float64)
    first_sq = torch.zeros((), dtype=torch.float64)
    second_sq = torch.zeros((), dtype=torch.float64)
    used = 0
    for left, right in zip(first, second, strict=True):
        if left is None or right is None:
            continue
        left64 = left.detach().to(dtype=torch.float64)
        right64 = right.detach().to(dtype=torch.float64)
        dot += torch.sum(left64 * right64).cpu()
        first_sq += torch.sum(torch.square(left64)).cpu()
        second_sq += torch.sum(torch.square(right64)).cpu()
        used += 1
    first_norm = float(torch.sqrt(first_sq).item())
    second_norm = float(torch.sqrt(second_sq).item())
    denominator = first_norm * second_norm
    cosine = float(dot.item() / denominator) if denominator > 0.0 else 0.0
    return {
        "time_encoder_grad_norm": first_norm,
        "quantity_encoder_grad_norm": second_norm,
        "time_quantity_grad_dot": float(dot.item()),
        "time_quantity_grad_cosine": cosine,
        "quantity_to_time_grad_norm_ratio": (
            second_norm / first_norm if first_norm > 0.0 else float("inf")
        ),
        "shared_parameter_tensor_count": float(used),
    }


def target_terms(
    model: SharedTimeCountModel,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Expose target hidden states for diagnostics without target leakage."""
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    time_states, quantity_states = model.encode_task_states(
        dts,
        history_quantities,
        mask,
    )
    time_hidden = time_states[batch_ids, history_positions]
    quantity_hidden = quantity_states[batch_ids, history_positions]
    true_dt = dts[batch_ids, target_positions].float()
    true_qty = quantities[batch_ids, target_positions].float()
    quantity = model.quantity_outputs(quantity_hidden, true_qty)
    return {
        "time_hidden": time_hidden,
        "true_dt": true_dt,
        "true_qty": true_qty,
        "time_loss": -model.log_f_dt(time_hidden, true_dt),
        "quantity_loss": quantity["train_loss"],
        "tail_indicator": quantity["tail_indicator"],
    }


def slope_nll_derivative(
    model: SharedTimeCountModel,
    hidden: torch.Tensor,
    dt: torch.Tensor,
) -> torch.Tensor:
    """Return per-event d(NLL)/dw for the exact RMTPP slope coordinate."""
    intercept = model.bounded_time_intercept(hidden).to(dtype=torch.float64)
    slope = model.positive_time_slope().to(dtype=torch.float64)
    scaled_dt = dt.to(dtype=torch.float64) / model.time_scale
    exponent = torch.exp(slope * scaled_dt)
    derivative = -scaled_dt + torch.exp(intercept) * (
        scaled_dt * exponent / slope
        - torch.expm1(slope * scaled_dt) / torch.square(slope)
    )
    return derivative


def audit_model(
    *,
    model: SharedTimeCountModel,
    loader: Any,
    device: str,
    max_batches: int,
    variant: str,
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit a fixed model state on deterministic train-only mini-batches."""
    model.eval()
    shared_parameters = tuple(
        parameter for _, parameter in shared_encoder_parameters(model)
    )
    rows: list[dict[str, Any]] = []
    for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
        if batch_index >= max_batches:
            break
        if quantities is None:
            raise ValueError("Gradient audit requires raw quantities")
        terms = target_terms(
            model,
            dts.to(device),
            mask.to(device),
            quantities.to(device),
        )
        time_mean = terms["time_loss"].mean()
        quantity_mean = terms["quantity_loss"].mean()
        time_gradients = torch.autograd.grad(
            time_mean,
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        quantity_gradients = torch.autograd.grad(
            quantity_mean,
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        gradient_stats = gradient_pair_statistics(
            time_gradients,
            quantity_gradients,
        )
        slope_derivative = slope_nll_derivative(
            model,
            terms["time_hidden"],
            terms["true_dt"],
        )
        intercept = model.bounded_time_intercept(terms["time_hidden"])
        slope = float(model.positive_time_slope().detach().cpu().item())
        row = {
            "variant": variant,
            "stage": stage,
            "batch_index": batch_index,
            "event_count": int(terms["true_dt"].numel()),
            "time_nll": float(time_mean.detach().cpu().item()),
            "quantity_loss": float(quantity_mean.detach().cpu().item()),
            "tail_count": int(terms["tail_indicator"].detach().sum().cpu().item()),
            "time_slope": slope,
            "time_slope_ratio": slope / model.time_w_max,
            "slope_upward_pressure_fraction": float(
                (slope_derivative < 0.0).double().mean().cpu().item()
            ),
            "slope_nll_derivative_mean": float(slope_derivative.mean().cpu().item()),
            "slope_nll_derivative_median": float(
                slope_derivative.median().cpu().item()
            ),
            "intercept_saturation_fraction": float(
                (intercept.detach().abs() >= 0.95 * model.time_intercept_limit)
                .double()
                .mean()
                .cpu()
                .item()
            ),
            **gradient_stats,
        }
        finite_values = [
            float(value)
            for key, value in row.items()
            if key not in {"variant", "stage"}
        ]
        if not all(math.isfinite(value) for value in finite_values):
            raise FloatingPointError(
                f"Non-finite gradient audit metric at {variant}/{stage}/batch {batch_index}"
            )
        rows.append(row)
    if not rows:
        raise ValueError("No batches were audited")
    summary = summarize_audit_rows(rows, variant=variant, stage=stage)
    return summary, rows


def summarize_audit_rows(
    rows: list[dict[str, Any]],
    *,
    variant: str,
    stage: str,
) -> dict[str, Any]:
    """Aggregate batch metrics into one train-only diagnostic stage."""
    cosine = np.asarray(
        [row["time_quantity_grad_cosine"] for row in rows],
        dtype=np.float64,
    )
    slope_ratio = np.asarray(
        [row["time_slope_ratio"] for row in rows],
        dtype=np.float64,
    )
    upward = np.asarray(
        [row["slope_upward_pressure_fraction"] for row in rows],
        dtype=np.float64,
    )
    summary = {
        "variant": variant,
        "stage": stage,
        "audited_batches": len(rows),
        "audited_events": int(sum(row["event_count"] for row in rows)),
        "time_slope": float(np.median([row["time_slope"] for row in rows])),
        "time_slope_ratio": float(np.median(slope_ratio)),
        "slope_upward_pressure_fraction": float(np.mean(upward)),
        "intercept_saturation_fraction": float(
            np.mean([row["intercept_saturation_fraction"] for row in rows])
        ),
        "time_encoder_grad_norm_median": float(
            np.median([row["time_encoder_grad_norm"] for row in rows])
        ),
        "quantity_encoder_grad_norm_median": float(
            np.median([row["quantity_encoder_grad_norm"] for row in rows])
        ),
        "quantity_to_time_grad_norm_ratio_median": float(
            np.median([row["quantity_to_time_grad_norm_ratio"] for row in rows])
        ),
        "gradient_cosine_mean": float(cosine.mean()),
        "gradient_cosine_median": float(np.median(cosine)),
        "gradient_negative_fraction": float((cosine < 0.0).mean()),
        "gradient_strong_conflict_fraction": float(
            (cosine <= GRADIENT_CONFLICT_COSINE).mean()
        ),
        "all_finite": True,
    }
    return summary


def classify_h1_failure(final_summary: dict[str, Any]) -> dict[str, Any]:
    """Classify H1 using frozen train-only thresholds."""
    slope_saturated = float(final_summary["time_slope_ratio"]) >= SLOPE_SATURATION_RATIO
    upward_pressure = (
        float(final_summary["slope_upward_pressure_fraction"])
        >= SLOPE_UPWARD_PRESSURE_FRACTION
    )
    strong_gradient_conflict = (
        float(final_summary["gradient_cosine_median"]) <= GRADIENT_CONFLICT_COSINE
        and float(final_summary["gradient_strong_conflict_fraction"])
        >= GRADIENT_CONFLICT_FRACTION
    )
    slope_contract_failed = slope_saturated and upward_pressure
    if slope_contract_failed and strong_gradient_conflict:
        recommendation = "replace_slope_family_and_isolate_time_gradient"
    elif slope_contract_failed:
        recommendation = "replace_slope_family_keep_shared_gradient"
    elif strong_gradient_conflict:
        recommendation = "keep_time_family_isolate_time_gradient"
    else:
        recommendation = "retain_h1_contract"
    return {
        "slope_saturated": slope_saturated,
        "continued_upward_slope_pressure": upward_pressure,
        "strong_time_quantity_gradient_conflict": strong_gradient_conflict,
        "slope_contract_failed": slope_contract_failed,
        "recommendation": recommendation,
        "thresholds": {
            "slope_saturation_ratio": SLOPE_SATURATION_RATIO,
            "slope_upward_pressure_fraction": SLOPE_UPWARD_PRESSURE_FRACTION,
            "gradient_conflict_cosine": GRADIENT_CONFLICT_COSINE,
            "gradient_conflict_fraction": GRADIENT_CONFLICT_FRACTION,
        },
    }


def append_log(path: Path, line: str) -> None:
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    if args.epochs < 1 or args.audit_batches < 1:
        raise ValueError("epochs and audit_batches must be positive")
    if args.hidden_dim != 64 or args.max_seq_len != 256:
        raise ValueError(
            "Frozen Intermittent contract requires hidden_dim=64/max_seq_len=256"
        )
    status_path = args.output_dir / "status.json"
    if status_path.exists() and not args.force_rerun:
        raise FileExistsError(f"Audit already has a status artifact: {status_path}")

    dataset_contract = DATASET_CONTRACTS["intermittent_frozen_5000"]
    data_sha256 = sha256_file(args.data)
    manifest_sha256 = sha256_file(args.split_manifest)
    if data_sha256 != dataset_contract["data_sha256"]:
        raise ValueError(f"Unexpected fixed-split SHA-256: {data_sha256}")
    if manifest_sha256 != dataset_contract["split_manifest_sha256"]:
        raise ValueError(f"Unexpected split-manifest SHA-256: {manifest_sha256}")

    train_raw = load_train_only_frame(args.data)
    if train_raw.height < 1 or set(train_raw["chronological_split"]) != {"train"}:
        raise ValueError("Audit must load train rows exclusively")
    train_log_qty = np.log1p(train_raw["demand_qty"].to_numpy().astype(np.float64))
    frame = prepare_count_frame(train_raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "audit.log"
    contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "count_aware_time_quantity_gradient_audit",
        "dataset": "intermittent_frozen_5000",
        "loaded_split": "train_only",
        "data_sha256": data_sha256,
        "split_manifest_sha256": manifest_sha256,
        "backbone": BACKBONE,
        "quantity_variant": TAIL_SHARED_VARIANT,
        "lambda_tail": FROZEN_TAIL_LAMBDA,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "audit_batches": args.audit_batches,
        "lr": args.lr,
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "hidden_dim": args.hidden_dim,
        "variants": VARIANT_SPECS,
        "decision_thresholds": {
            "slope_saturation_ratio": SLOPE_SATURATION_RATIO,
            "slope_upward_pressure_fraction": SLOPE_UPWARD_PRESSURE_FRACTION,
            "gradient_conflict_cosine": GRADIENT_CONFLICT_COSINE,
            "gradient_conflict_fraction": GRADIENT_CONFLICT_FRACTION,
        },
        "selection_source": "train_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "execution_host": os.uname().nodename,
        "execution_role": args.execution_role,
    }
    save_json(args.output_dir / "launch_contract.json", contract)

    stage_summaries: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    started = time.time()
    for variant, spec in VARIANT_SPECS.items():
        generator = set_seed(SEED)
        train_loader = make_loader(
            frame,
            target_split="train",
            batch_size=args.batch_size,
            lookback_weeks=args.lookback_weeks,
            max_seq_len=args.max_seq_len,
            shuffle=True,
            generator=generator,
        )
        audit_loader = make_loader(
            frame,
            target_split="train",
            batch_size=args.batch_size,
            lookback_weeks=args.lookback_weeks,
            max_seq_len=args.max_seq_len,
            shuffle=False,
            generator=None,
        )
        time_contract = derive_train_time_contract(
            frame,
            lookback_weeks=args.lookback_weeks,
            max_seq_len=args.max_seq_len,
            wd_safety_limit=float(spec["time_wd_safety_limit"]),
        )
        initial_intercept = (
            float(time_contract["time_initial_intercept"])
            if spec["time_head_mode"] == TIME_HEAD_MODE_SCALED_EXACT_STABLE
            else math.log(float(time_contract["time_scale"]))
        )
        model, _ = build_count_aware_model(
            BACKBONE,
            hidden_dim=args.hidden_dim,
            train_log_mean=float(train_log_qty.mean()),
            train_log_std=float(train_log_qty.std()),
            max_seq_len=args.max_seq_len,
            quantity_variant=TAIL_SHARED_VARIANT,
            lambda_tail=FROZEN_TAIL_LAMBDA,
            time_head_mode=str(spec["time_head_mode"]),
            time_scale=float(time_contract["time_scale"]),
            time_w_max=float(time_contract["time_w_max"]),
            time_intercept_limit=float(spec["time_intercept_limit"]),
            time_initial_intercept=initial_intercept,
            time_wd_safety_limit=float(spec["time_wd_safety_limit"]),
        )
        model.to(args.device)
        optimizer = build_optimizer(model, lr=args.lr)
        summary, rows = audit_model(
            model=model,
            loader=audit_loader,
            device=args.device,
            max_batches=args.audit_batches,
            variant=variant,
            stage="initial",
        )
        stage_summaries.append(summary)
        batch_rows.extend(rows)
        append_log(
            log_path,
            f"[audit {variant} initial] slope_ratio={summary['time_slope_ratio']:.6f} "
            f"upward={summary['slope_upward_pressure_fraction']:.6f} "
            f"cosine={summary['gradient_cosine_median']:.6f}",
        )
        for epoch in range(1, args.epochs + 1):
            telemetry = train_epoch_with_telemetry(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=args.device,
                lambda_log_qty=1.0,
                grad_clip=args.grad_clip,
                max_batches=args.max_train_batches,
            )
            train_rows.append({"variant": variant, "epoch": epoch, **telemetry})
            summary, rows = audit_model(
                model=model,
                loader=audit_loader,
                device=args.device,
                max_batches=args.audit_batches,
                variant=variant,
                stage=f"epoch_{epoch}",
            )
            stage_summaries.append(summary)
            batch_rows.extend(rows)
            append_log(
                log_path,
                f"[audit {variant} epoch {epoch:03d}] "
                f"train_joint={telemetry['train_joint_objective']:.8f} "
                f"slope_ratio={summary['time_slope_ratio']:.6f} "
                f"upward={summary['slope_upward_pressure_fraction']:.6f} "
                f"cosine={summary['gradient_cosine_median']:.6f} "
                f"conflict={summary['gradient_strong_conflict_fraction']:.6f}",
            )

    h1_final = next(
        row
        for row in stage_summaries
        if row["variant"] == "H1" and row["stage"] == f"epoch_{args.epochs}"
    )
    decision = {
        **classify_h1_failure(h1_final),
        "diagnosed_variant": "H1",
        "diagnosed_stage": f"epoch_{args.epochs}",
        "backbone": BACKBONE,
        "quantity_variant": TAIL_SHARED_VARIANT,
        "selection_source": "train_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
    }
    write_csv(args.output_dir / "gradient_batch_metrics.csv", batch_rows)
    write_csv(args.output_dir / "gradient_stage_summary.csv", stage_summaries)
    write_csv(args.output_dir / "train_epoch_metrics.csv", train_rows)
    save_json(args.output_dir / "decision.json", decision)
    contract.update(
        {
            "status": "complete",
            "elapsed_seconds": time.time() - started,
            "recommendation": decision["recommendation"],
        }
    )
    save_json(args.output_dir / "launch_contract.json", contract)
    save_json(
        status_path,
        {
            "status": "complete",
            "recommendation": decision["recommendation"],
            "validation_evaluated": False,
            "held_out_test_evaluated": False,
            "source_revision": args.source_revision,
        },
    )
    append_log(
        log_path,
        f"[complete] recommendation={decision['recommendation']}",
    )


if __name__ == "__main__":
    main()
