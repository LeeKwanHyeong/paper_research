#!/usr/bin/env python3
"""Audit Online Retail II time scaling and gradient clipping on train rows only."""

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

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT, TIME_HEAD_MODE_LEGACY_CLAMPED
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame, target_outputs
from paper.scripts.count_aware_tpp_backbone.datasets import DATASET_CONTRACTS
from paper.scripts.count_aware_tpp_backbone.reporting import write_csv
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
from paper.scripts.run_taxi_quantity_interface_ablation import (
    make_loader,
    save_json,
    set_seed,
    sha256_file,
)


SEED = 42
BACKBONE = "titantpp"
STABILITY_THRESHOLDS = {
    "max_epoch_train_joint_objective": 100.0,
    "max_per_event_time_nll": 10_000.0,
    "max_time_only_gradient_exceed_fraction": 0.25,
}


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
    parser.add_argument("--lookback-hours", type=int, default=8760)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=16)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def load_train_only_frame(path: Path) -> pl.DataFrame:
    """Read the official train partition without materializing validation/test rows."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )


def train_target_delta_times(dataset: Any) -> np.ndarray:
    """Extract exactly the next-event targets exposed by the lookback dataset."""
    return np.asarray(
        [dataset.dt_lists[part_index][context_end + 1] for part_index, context_end in dataset.index],
        dtype=np.float64,
    )


def summarize_target_delta_times(values: np.ndarray) -> dict[str, float | int]:
    if values.size < 1 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("Train target delta-times must be positive and finite")
    return {
        "count": int(values.size),
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(values.max()),
    }


def build_scale_variants(target_summary: dict[str, float | int]) -> dict[str, dict[str, Any]]:
    """Freeze all scale candidates from constants or train targets only."""
    variants = {
        "S0_raw_hour": {
            "description": "raw_hour_negative_control",
            "divisor_source": "constant_hour",
            "divisor": 1.0,
        },
        "S1_calendar_day": {
            "description": "calendar_day_rescaling",
            "divisor_source": "constant_24_hours",
            "divisor": 24.0,
        },
        "S2_train_target_median": {
            "description": "train_target_median_rescaling",
            "divisor_source": "train_target_p50",
            "divisor": float(target_summary["p50"]),
        },
        "S3_train_target_mean": {
            "description": "train_target_mean_rescaling",
            "divisor_source": "train_target_mean",
            "divisor": float(target_summary["mean"]),
        },
        "S4_train_target_p95": {
            "description": "train_target_p95_upper_scale_probe",
            "divisor_source": "train_target_p95",
            "divisor": float(target_summary["p95"]),
        },
    }
    if any(not math.isfinite(spec["divisor"]) or spec["divisor"] <= 0.0 for spec in variants.values()):
        raise ValueError("Every time-scale divisor must be positive and finite")
    return variants


def gradient_statistics(gradients: Iterable[torch.Tensor | None]) -> dict[str, float]:
    squares = torch.zeros((), dtype=torch.float64)
    used = 0
    for gradient in gradients:
        if gradient is None:
            continue
        value = gradient.detach().to(dtype=torch.float64)
        squares += torch.sum(torch.square(value)).cpu()
        used += 1
    return {
        "norm": float(torch.sqrt(squares).item()),
        "used_parameter_tensors": float(used),
    }


def jacobian_corrected_hour_nll(scaled_nll: float, divisor: float) -> float:
    """Convert NLL in the scaled coordinate back to density per original hour."""
    if divisor <= 0.0:
        raise ValueError("divisor must be positive")
    return float(scaled_nll + math.log(divisor))


def append_log(path: Path, line: str) -> None:
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def train_variant(
    *,
    args: argparse.Namespace,
    frame: pl.DataFrame,
    train_log_mean: float,
    train_log_std: float,
    target_dts: np.ndarray,
    variant: str,
    spec: dict[str, Any],
    log_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    generator = set_seed(SEED)
    loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_hours,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=generator,
    )
    model, encoder_config = build_count_aware_model(
        BACKBONE,
        hidden_dim=args.hidden_dim,
        train_log_mean=train_log_mean,
        train_log_std=train_log_std,
        max_seq_len=args.max_seq_len,
        quantity_variant=LOG_MSE_VARIANT,
        time_head_mode=TIME_HEAD_MODE_LEGACY_CLAMPED,
    )
    model.to(args.device)
    optimizer = build_optimizer(model, lr=args.lr)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    divisor = float(spec["divisor"])
    initial_slope = float((torch.nn.functional.softplus(model.w_raw) + 1e-3).detach().cpu().item())
    initial_saturation_fraction = float(np.mean(initial_slope * target_dts / divisor >= 10.0))
    history: list[dict[str, Any]] = []
    started = time.time()
    append_log(
        log_path,
        f"[variant-start] variant={variant} divisor={divisor:.10f} "
        f"initial_slope={initial_slope:.10f} saturation_fraction={initial_saturation_fraction:.8f}",
    )
    status = "success"
    failure: dict[str, str] | None = None
    try:
        for epoch in range(1, args.epochs + 1):
            event_count = 0
            joint_sum = 0.0
            time_sum = 0.0
            quantity_sum = 0.0
            max_time_nll = -float("inf")
            time_grad_norms: list[float] = []
            quantity_grad_norms: list[float] = []
            joint_grad_norms: list[float] = []
            clipped = 0
            time_grad_exceeded = 0
            quantity_grad_exceeded = 0
            for batch_index, (_, dts, mask, _, quantities) in enumerate(loader):
                if batch_index >= args.max_train_batches:
                    break
                if quantities is None:
                    raise ValueError("Count-aware audit requires raw quantities")
                scaled_dts = dts.to(args.device) / divisor
                outputs = target_outputs(
                    model,
                    scaled_dts,
                    mask.to(args.device),
                    quantities.to(args.device),
                    lambda_log_qty=1.0,
                )
                if not all(
                    bool(torch.isfinite(outputs[key]).all())
                    for key in ("joint_loss", "time_loss", "quantity_train_loss")
                ):
                    raise FloatingPointError(f"Non-finite loss at {variant}/epoch={epoch}/batch={batch_index}")

                time_mean = outputs["time_loss"].mean()
                quantity_mean = outputs["quantity_train_loss"].mean()
                time_grad = torch.autograd.grad(
                    time_mean, parameters, retain_graph=True, allow_unused=True
                )
                quantity_grad = torch.autograd.grad(
                    quantity_mean, parameters, retain_graph=True, allow_unused=True
                )
                time_grad_norm = gradient_statistics(time_grad)["norm"]
                quantity_grad_norm = gradient_statistics(quantity_grad)["norm"]
                time_grad_norms.append(time_grad_norm)
                quantity_grad_norms.append(quantity_grad_norm)
                time_grad_exceeded += int(time_grad_norm > args.grad_clip)
                quantity_grad_exceeded += int(quantity_grad_norm > args.grad_clip)
                del time_grad, quantity_grad

                optimizer.zero_grad(set_to_none=True)
                outputs["joint_loss"].mean().backward()
                pre_clip = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip, error_if_nonfinite=False
                )
                pre_clip_value = float(pre_clip.detach().cpu().item())
                if not math.isfinite(pre_clip_value):
                    raise FloatingPointError(f"Non-finite joint gradient at {variant}/epoch={epoch}/batch={batch_index}")
                optimizer.step()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                    raise FloatingPointError(f"Non-finite parameter at {variant}/epoch={epoch}/batch={batch_index}")

                joint = outputs["joint_loss"].detach().double()
                time_loss = outputs["time_loss"].detach().double()
                quantity_loss = outputs["quantity_train_loss"].detach().double()
                count = int(joint.numel())
                event_count += count
                joint_sum += float(joint.sum().item())
                time_sum += float(time_loss.sum().item())
                quantity_sum += float(quantity_loss.sum().item())
                max_time_nll = max(max_time_nll, float(time_loss.max().item()))
                joint_grad_norms.append(pre_clip_value)
                clipped += int(pre_clip_value > args.grad_clip)

            if event_count < 1:
                raise ValueError("No train batches were evaluated")
            time_grad_array = np.asarray(time_grad_norms, dtype=np.float64)
            quantity_grad_array = np.asarray(quantity_grad_norms, dtype=np.float64)
            joint_grad_array = np.asarray(joint_grad_norms, dtype=np.float64)
            scaled_time_nll = time_sum / event_count
            row = {
                "variant": variant,
                "epoch": epoch,
                "divisor": divisor,
                "train_joint_objective": joint_sum / event_count,
                "train_time_nll_scaled_coordinate": scaled_time_nll,
                "train_time_nll_per_original_hour": jacobian_corrected_hour_nll(scaled_time_nll, divisor),
                "train_quantity_loss": quantity_sum / event_count,
                "train_max_per_event_time_nll": max_time_nll,
                "time_only_grad_norm_mean": float(time_grad_array.mean()),
                "time_only_grad_norm_max": float(time_grad_array.max()),
                "quantity_only_grad_norm_mean": float(quantity_grad_array.mean()),
                "quantity_only_grad_norm_max": float(quantity_grad_array.max()),
                "joint_pre_clip_grad_norm_mean": float(joint_grad_array.mean()),
                "joint_pre_clip_grad_norm_max": float(joint_grad_array.max()),
                "gradient_clip_count": clipped,
                "gradient_clip_fraction": clipped / len(joint_grad_norms),
                "time_only_gradient_exceed_count": time_grad_exceeded,
                "time_only_gradient_exceed_fraction": time_grad_exceeded / len(joint_grad_norms),
                "quantity_only_gradient_exceed_count": quantity_grad_exceeded,
                "quantity_only_gradient_exceed_fraction": quantity_grad_exceeded / len(joint_grad_norms),
                "event_count": event_count,
                "batch_count": len(joint_grad_norms),
                "time_slope": float((torch.nn.functional.softplus(model.w_raw) + 1e-3).detach().cpu().item()),
                "all_finite": True,
            }
            if not all(math.isfinite(float(value)) for key, value in row.items() if key not in {"variant", "epoch", "event_count", "batch_count", "gradient_clip_count", "time_only_gradient_exceed_count", "quantity_only_gradient_exceed_count", "all_finite"}):
                raise FloatingPointError(f"Non-finite epoch telemetry at {variant}/epoch={epoch}")
            history.append(row)
            append_log(
                log_path,
                f"[train-only {variant} epoch {epoch:03d}] joint={row['train_joint_objective']:.8f} "
                f"time_scaled={row['train_time_nll_scaled_coordinate']:.8f} "
                f"time_hour={row['train_time_nll_per_original_hour']:.8f} "
                f"time_grad={row['time_only_grad_norm_mean']:.8f} "
                f"qty_grad={row['quantity_only_grad_norm_mean']:.8f} "
                f"joint_grad={row['joint_pre_clip_grad_norm_mean']:.8f} "
                f"joint_clip={row['gradient_clip_fraction']:.6f} "
                f"time_exceed={row['time_only_gradient_exceed_fraction']:.6f} "
                f"qty_exceed={row['quantity_only_gradient_exceed_fraction']:.6f}",
            )
    except (FloatingPointError, RuntimeError, ValueError) as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        append_log(log_path, f"[variant-failed] variant={variant} type={type(exc).__name__} message={exc}")

    gate = evaluate_stability_gate(history, run_status=status)
    summary = {
        "variant": variant,
        "description": spec["description"],
        "divisor_source": spec["divisor_source"],
        "divisor": divisor,
        "status": status,
        "failure": failure,
        "completed_epochs": len(history),
        "elapsed_seconds": time.time() - started,
        "initial_time_slope": initial_slope,
        "initial_wd_clamp_saturation_fraction": initial_saturation_fraction,
        "stability_gate": gate,
        "encoder_config": encoder_config,
        "evaluation_scope": "train_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
    }
    run_dir = args.output_dir / "runs" / variant
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "history.json", {"history": history})
    save_json(run_dir / "summary.json", summary)
    append_log(log_path, f"[variant-complete] variant={variant} status={status} gate_passed={gate['passed']}")
    return summary, history


def evaluate_stability_gate(history: list[dict[str, Any]], *, run_status: str) -> dict[str, Any]:
    if not history:
        return {"passed": False, "checks": {"run_completed": False}, "metrics": {}, "thresholds": STABILITY_THRESHOLDS}
    max_joint = max(float(row["train_joint_objective"]) for row in history)
    max_time = max(float(row["train_max_per_event_time_nll"]) for row in history)
    total_batches = sum(int(row["batch_count"]) for row in history)
    total_clipped = sum(int(row["gradient_clip_count"]) for row in history)
    total_time_exceeded = sum(int(row["time_only_gradient_exceed_count"]) for row in history)
    total_quantity_exceeded = sum(int(row["quantity_only_gradient_exceed_count"]) for row in history)
    clip_fraction = total_clipped / max(total_batches, 1)
    time_exceed_fraction = total_time_exceeded / max(total_batches, 1)
    quantity_exceed_fraction = total_quantity_exceeded / max(total_batches, 1)
    all_finite = all(bool(row["all_finite"]) for row in history)
    checks = {
        "run_completed": run_status == "success",
        "all_finite": all_finite,
        "epoch_joint_within_limit": max_joint <= STABILITY_THRESHOLDS["max_epoch_train_joint_objective"],
        "per_event_time_nll_within_limit": max_time <= STABILITY_THRESHOLDS["max_per_event_time_nll"],
        "time_only_gradient_exceed_fraction_within_limit": time_exceed_fraction <= STABILITY_THRESHOLDS["max_time_only_gradient_exceed_fraction"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "max_epoch_train_joint_objective": max_joint,
            "max_per_event_time_nll": max_time,
            "aggregate_gradient_clip_fraction": clip_fraction,
            "aggregate_time_only_gradient_exceed_fraction": time_exceed_fraction,
            "aggregate_quantity_only_gradient_exceed_fraction": quantity_exceed_fraction,
            "total_batches": total_batches,
            "total_clipped_batches": total_clipped,
        },
        "thresholds": STABILITY_THRESHOLDS,
    }


def recommend_next_action(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = {summary["variant"]: summary for summary in summaries}
    passing = [
        variant
        for variant in ("S1_calendar_day", "S2_train_target_median", "S3_train_target_mean", "S4_train_target_p95")
        if by_variant[variant]["stability_gate"]["passed"]
    ]
    raw_passed = bool(by_variant["S0_raw_hour"]["stability_gate"]["passed"])
    if passing:
        return {
            "decision": "retain_online_retail_for_scaled_time_followup",
            "primary_candidate": passing[0],
            "passing_scaled_variants": passing,
            "raw_hour_passed": raw_passed,
            "validation_required_before_model_comparison": True,
        }
    return {
        "decision": "stop_online_retail_under_legacy_time_head",
        "primary_candidate": None,
        "passing_scaled_variants": [],
        "raw_hour_passed": raw_passed,
        "validation_required_before_model_comparison": False,
    }


def summary_row(summary: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    final = history[-1] if history else {}
    gate_metrics = summary["stability_gate"].get("metrics", {})
    return {
        "variant": summary["variant"],
        "divisor": summary["divisor"],
        "status": summary["status"],
        "gate_passed": summary["stability_gate"]["passed"],
        "initial_wd_clamp_saturation_fraction": summary["initial_wd_clamp_saturation_fraction"],
        "final_time_nll_scaled_coordinate": final.get("train_time_nll_scaled_coordinate"),
        "final_time_nll_per_original_hour": final.get("train_time_nll_per_original_hour"),
        "final_quantity_loss": final.get("train_quantity_loss"),
        "final_time_only_grad_norm_mean": final.get("time_only_grad_norm_mean"),
        "final_quantity_only_grad_norm_mean": final.get("quantity_only_grad_norm_mean"),
        "final_joint_pre_clip_grad_norm_mean": final.get("joint_pre_clip_grad_norm_mean"),
        "aggregate_gradient_clip_fraction": gate_metrics.get("aggregate_gradient_clip_fraction"),
        "aggregate_time_only_gradient_exceed_fraction": gate_metrics.get("aggregate_time_only_gradient_exceed_fraction"),
        "aggregate_quantity_only_gradient_exceed_fraction": gate_metrics.get("aggregate_quantity_only_gradient_exceed_fraction"),
        "max_per_event_time_nll": gate_metrics.get("max_per_event_time_nll"),
    }


def write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["variant"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(11, 5))
    width = 0.27
    axis.bar(x - width, [float(row["aggregate_gradient_clip_fraction"]) for row in rows], width=width, label="joint clip")
    axis.bar(x, [float(row["aggregate_time_only_gradient_exceed_fraction"]) for row in rows], width=width, label="time-only > threshold")
    axis.bar(x + width, [float(row["aggregate_quantity_only_gradient_exceed_fraction"]) for row in rows], width=width, label="quantity-only > threshold")
    axis.axhline(STABILITY_THRESHOLDS["max_time_only_gradient_exceed_fraction"], color="tab:red", linestyle="--", label="time gate")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Batch fraction")
    axis.set_ylim(0.0, 1.05)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "gradient_clipping_by_time_scale.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x, [float(row["max_per_event_time_nll"]) for row in rows])
    axis.axhline(STABILITY_THRESHOLDS["max_per_event_time_nll"], color="tab:red", linestyle="--", label="gate")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Maximum per-event Time NLL")
    axis.set_yscale("symlog", linthresh=1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "max_time_nll_by_time_scale.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    if args.epochs < 1 or args.max_train_batches < 1:
        raise ValueError("epochs and max-train-batches must be positive")
    if args.lookback_hours != 8760 or args.max_seq_len != 256 or args.hidden_dim != 64:
        raise ValueError("Frozen Online Retail contract requires lookback=8760/max_seq_len=256/hidden_dim=64")
    status_path = args.output_dir / "status.json"
    if status_path.exists() and not args.force_rerun:
        raise FileExistsError(f"Audit already has a status artifact: {status_path}")

    dataset_contract = DATASET_CONTRACTS["online_retail_ii"]
    data_sha = sha256_file(args.data)
    manifest_sha = sha256_file(args.split_manifest)
    if data_sha != dataset_contract["data_sha256"]:
        raise ValueError(f"Unexpected Online Retail II SHA-256: {data_sha}")
    if manifest_sha != dataset_contract["split_manifest_sha256"]:
        raise ValueError(f"Unexpected split-manifest SHA-256: {manifest_sha}")

    train_raw = load_train_only_frame(args.data)
    if train_raw.height < 1 or set(train_raw["chronological_split"]) != {"train"}:
        raise ValueError("Audit must load train rows exclusively")
    frame = prepare_count_frame(train_raw)
    reference_loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_hours,
        max_seq_len=args.max_seq_len,
        shuffle=False,
        generator=None,
    )
    target_dts = train_target_delta_times(reference_loader.dataset)
    target_summary = summarize_target_delta_times(target_dts)
    variants = build_scale_variants(target_summary)
    train_log_qty = np.log1p(train_raw["demand_qty"].to_numpy().astype(np.float64))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "audit.log"
    launch_contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "online_retail_train_only_time_scale_gradient_audit",
        "dataset": "online_retail_ii",
        "loaded_split": "train_only",
        "train_rows": train_raw.height,
        "train_target_count": int(target_dts.size),
        "data_sha256": data_sha,
        "split_manifest_sha256": manifest_sha,
        "backbone": BACKBONE,
        "model_role": "t0_common_control",
        "quantity_variant": LOG_MSE_VARIANT,
        "time_head_mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lookback_hours": args.lookback_hours,
        "max_seq_len": args.max_seq_len,
        "hidden_dim": args.hidden_dim,
        "grad_clip": args.grad_clip,
        "max_train_batches": args.max_train_batches,
        "variants": variants,
        "stability_thresholds": STABILITY_THRESHOLDS,
        "selection_source": "train_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "execution_host": os.uname().nodename,
        "execution_role": args.execution_role,
    }
    save_json(args.output_dir / "launch_contract.json", launch_contract)
    save_json(args.output_dir / "target_delta_time_summary.json", target_summary)

    summaries: list[dict[str, Any]] = []
    histories: dict[str, list[dict[str, Any]]] = {}
    started = time.time()
    for variant, spec in variants.items():
        summary, history = train_variant(
            args=args,
            frame=frame,
            train_log_mean=float(train_log_qty.mean()),
            train_log_std=float(train_log_qty.std()),
            target_dts=target_dts,
            variant=variant,
            spec=spec,
            log_path=log_path,
        )
        summaries.append(summary)
        histories[variant] = history

    rows = [summary_row(summary, histories[summary["variant"]]) for summary in summaries]
    write_csv(args.output_dir / "variant_summary.csv", rows)
    decision = recommend_next_action(summaries)
    save_json(args.output_dir / "decision.json", decision)
    save_json(
        args.output_dir / "evaluation_scope.json",
        {
            "train_evaluated": True,
            "validation_evaluated": False,
            "held_out_test_evaluated": False,
            "test_artifact_present": False,
        },
    )
    write_plots(args.output_dir, rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "experiment": "online_retail_train_only_time_scale_gradient_audit",
        "source_revision": args.source_revision,
        "execution_server": "5080",
        "execution_host": os.uname().nodename,
        "elapsed_seconds": time.time() - started,
        "variant_count": len(summaries),
        "successful_variant_count": sum(summary["status"] == "success" for summary in summaries),
        "passing_variant_count": sum(summary["stability_gate"]["passed"] for summary in summaries),
        "decision": decision["decision"],
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "artifact_reading_order": [
            "manifest.json",
            "audit.log",
            "variant_summary.csv and decision.json",
            "evaluation_scope.json",
            "runs/*/history.json",
            "target_delta_time_summary.json",
            "plots/*.png",
        ],
    }
    save_json(args.output_dir / "manifest.json", manifest)
    save_json(status_path, {"status": "complete", "decision": decision["decision"]})
    append_log(log_path, f"[complete] decision={decision['decision']} primary={decision['primary_candidate']}")


if __name__ == "__main__":
    main()
