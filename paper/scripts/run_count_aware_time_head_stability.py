#!/usr/bin/env python3
"""Select a stable exact time head using Intermittent train data only."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    TIME_HEAD_MODE_SCALED_EXACT,
    TIME_HEAD_MODE_SCALED_EXACT_STABLE,
)
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.count_aware_tpp_backbone.reporting import write_csv
from paper.scripts.count_aware_tpp_backbone.training import (
    build_optimizer,
    optimizer_group_contract,
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


BACKBONE = "titantpp_persistent_only"
SEED = 42
STABILITY_THRESHOLDS = {
    "max_epoch_train_joint_objective": 100.0,
    "max_batch_joint_p99": 100.0,
    "max_per_event_time_nll": 10_000.0,
    "max_gradient_clip_fraction": 0.25,
}
VARIANT_SPECS = {
    "H0": {
        "description": "current_scaled_exact_negative_control",
        "time_head_mode": TIME_HEAD_MODE_SCALED_EXACT,
        "time_wd_safety_limit": TIME_WD_SAFETY_LIMIT,
        "time_intercept_limit": 30.0,
        "time_head_lr_multiplier": 1.0,
    },
    "H1": {
        "description": "stable_exact_primary",
        "time_head_mode": TIME_HEAD_MODE_SCALED_EXACT_STABLE,
        "time_wd_safety_limit": STABLE_TIME_WD_SAFETY_LIMIT,
        "time_intercept_limit": STABLE_TIME_INTERCEPT_LIMIT,
        "time_head_lr_multiplier": 1.0,
    },
    "H2": {
        "description": "stable_exact_lower_time_lr_fallback",
        "time_head_mode": TIME_HEAD_MODE_SCALED_EXACT_STABLE,
        "time_wd_safety_limit": STABLE_TIME_WD_SAFETY_LIMIT,
        "time_intercept_limit": STABLE_TIME_INTERCEPT_LIMIT,
        "time_head_lr_multiplier": 0.1,
    },
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
    parser.add_argument("--lookback-weeks", type=int, default=520)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def evaluate_stability_gate(
    history: list[dict[str, Any]],
    *,
    run_status: str,
) -> dict[str, Any]:
    """Apply the frozen train-only gate without consulting validation data."""
    if not history:
        return {
            "passed": False,
            "checks": {
                "run_completed": False,
                "all_finite": False,
                "epoch_joint_within_limit": False,
                "batch_p99_within_limit": False,
                "per_event_time_nll_within_limit": False,
                "gradient_clip_fraction_within_limit": False,
            },
            "metrics": {},
            "thresholds": STABILITY_THRESHOLDS,
        }

    numeric_keys = (
        "train_joint_objective",
        "train_time_nll",
        "train_quantity_loss",
        "train_batch_joint_p99",
        "train_max_per_event_time_nll",
        "train_pre_clip_grad_norm_mean",
        "train_pre_clip_grad_norm_max",
        "train_gradient_clip_fraction",
        "train_time_slope",
    )
    all_finite = all(
        bool(row.get("train_all_finite", False))
        and all(math.isfinite(float(row[key])) for key in numeric_keys)
        for row in history
    )
    max_epoch_joint = max(float(row["train_joint_objective"]) for row in history)
    max_batch_p99 = max(float(row["train_batch_joint_p99"]) for row in history)
    max_time_nll = max(
        float(row["train_max_per_event_time_nll"]) for row in history
    )
    total_batches = sum(int(row["train_batch_count"]) for row in history)
    total_clipped = sum(int(row["train_gradient_clip_count"]) for row in history)
    clip_fraction = total_clipped / max(total_batches, 1)
    checks = {
        "run_completed": run_status == "success",
        "all_finite": all_finite,
        "epoch_joint_within_limit": (
            max_epoch_joint
            <= STABILITY_THRESHOLDS["max_epoch_train_joint_objective"]
        ),
        "batch_p99_within_limit": (
            max_batch_p99 <= STABILITY_THRESHOLDS["max_batch_joint_p99"]
        ),
        "per_event_time_nll_within_limit": (
            max_time_nll <= STABILITY_THRESHOLDS["max_per_event_time_nll"]
        ),
        "gradient_clip_fraction_within_limit": (
            clip_fraction
            <= STABILITY_THRESHOLDS["max_gradient_clip_fraction"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "max_epoch_train_joint_objective": max_epoch_joint,
            "max_batch_joint_p99": max_batch_p99,
            "max_per_event_time_nll": max_time_nll,
            "aggregate_gradient_clip_fraction": clip_fraction,
            "total_batches": total_batches,
            "total_clipped_batches": total_clipped,
        },
        "thresholds": STABILITY_THRESHOLDS,
    }


def append_log(path: Path, line: str) -> None:
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_train_only_frame(path: Path) -> pl.DataFrame:
    """Load only train rows so calibration cannot inspect validation targets."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )


def should_run_h2(h1_summary: dict[str, Any]) -> bool:
    """Open the lower-LR fallback only after an H1 train-gate failure."""
    return not bool(h1_summary["stability_gate"]["passed"])


def select_stable_variant(
    summaries: dict[str, dict[str, Any]],
) -> str | None:
    """Select H1 first, then H2, without considering H0 or validation."""
    if summaries["H1"]["stability_gate"]["passed"]:
        return "H1"
    if summaries.get("H2", {}).get("stability_gate", {}).get("passed", False):
        return "H2"
    return None


def run_variant(
    *,
    args: argparse.Namespace,
    frame: pl.DataFrame,
    train_log_mean: float,
    train_log_std: float,
    variant: str,
    log_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    spec = VARIANT_SPECS[variant]
    generator = set_seed(SEED)
    loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=generator,
    )
    time_contract = derive_train_time_contract(
        frame,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        wd_safety_limit=float(spec["time_wd_safety_limit"]),
    )
    time_initial_intercept = (
        float(time_contract["time_initial_intercept"])
        if spec["time_head_mode"] == TIME_HEAD_MODE_SCALED_EXACT_STABLE
        else math.log(float(time_contract["time_scale"]))
    )
    model, encoder_config = build_count_aware_model(
        BACKBONE,
        hidden_dim=args.hidden_dim,
        train_log_mean=train_log_mean,
        train_log_std=train_log_std,
        max_seq_len=args.max_seq_len,
        time_head_mode=str(spec["time_head_mode"]),
        time_scale=float(time_contract["time_scale"]),
        time_w_max=float(time_contract["time_w_max"]),
        time_intercept_limit=float(spec["time_intercept_limit"]),
        time_initial_intercept=time_initial_intercept,
        time_wd_safety_limit=float(spec["time_wd_safety_limit"]),
    )
    model.to(args.device)
    optimizer = build_optimizer(
        model,
        lr=args.lr,
        time_head_lr_multiplier=float(spec["time_head_lr_multiplier"]),
    )
    history: list[dict[str, Any]] = []
    status = "success"
    failure: dict[str, str] | None = None
    started = time.time()
    append_log(
        log_path,
        f"[variant-start] variant={variant} mode={spec['time_head_mode']} "
        f"time_lr_multiplier={spec['time_head_lr_multiplier']}",
    )
    try:
        for epoch in range(1, args.epochs + 1):
            telemetry = train_epoch_with_telemetry(
                model=model,
                loader=loader,
                optimizer=optimizer,
                device=args.device,
                lambda_log_qty=1.0,
                grad_clip=args.grad_clip,
                max_batches=args.max_train_batches,
            )
            row = {"variant": variant, "epoch": epoch, **telemetry}
            history.append(row)
            append_log(
                log_path,
                f"[train-only {variant} epoch {epoch:03d}] "
                f"joint={row['train_joint_objective']:.8f} "
                f"time={row['train_time_nll']:.8f} "
                f"batch_p99={row['train_batch_joint_p99']:.8f} "
                f"max_time={row['train_max_per_event_time_nll']:.8f} "
                f"grad={row['train_pre_clip_grad_norm_mean']:.8f} "
                f"clip_fraction={row['train_gradient_clip_fraction']:.6f}",
            )
    except (FloatingPointError, RuntimeError) as exc:
        status = "failed"
        failure = {"type": type(exc).__name__, "message": str(exc)}
        append_log(
            log_path,
            f"[variant-failed] variant={variant} type={type(exc).__name__} "
            f"message={exc}",
        )

    gate = evaluate_stability_gate(history, run_status=status)
    variant_dir = args.output_dir / "runs" / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_json(variant_dir / "history.json", {"history": history})
    torch.save(
        {
            "variant": variant,
            "seed": SEED,
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "evaluation_scope": "train_only",
            "validation_evaluated": False,
            "held_out_test_evaluated": False,
            "source_revision": args.source_revision,
        },
        variant_dir / "last_train_state.pt",
    )
    summary = {
        "variant": variant,
        "description": spec["description"],
        "status": status,
        "failure": failure,
        "seed": SEED,
        "completed_epochs": len(history),
        "elapsed_seconds": time.time() - started,
        "time_head": model.time_head_contract(),
        "time_head_lr_multiplier": spec["time_head_lr_multiplier"],
        "train_time_contract": time_contract,
        "optimizer_groups": optimizer_group_contract(optimizer),
        "encoder_config": encoder_config,
        "stability_gate": gate,
        "evaluation_scope": "train_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
    }
    save_json(variant_dir / "summary.json", summary)
    append_log(
        log_path,
        f"[variant-complete] variant={variant} status={status} "
        f"gate_passed={gate['passed']}",
    )
    return summary, history


def summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary["stability_gate"].get("metrics", {})
    return {
        "variant": summary["variant"],
        "description": summary["description"],
        "status": summary["status"],
        "gate_passed": summary["stability_gate"]["passed"],
        "completed_epochs": summary["completed_epochs"],
        "time_head_mode": summary["time_head"]["mode"],
        "time_w_max": summary["time_head"]["time_w_max"],
        "time_intercept_limit": summary["time_head"]["time_intercept_limit"],
        "time_head_lr_multiplier": summary["time_head_lr_multiplier"],
        "max_epoch_train_joint_objective": metrics.get(
            "max_epoch_train_joint_objective"
        ),
        "max_batch_joint_p99": metrics.get("max_batch_joint_p99"),
        "max_per_event_time_nll": metrics.get("max_per_event_time_nll"),
        "aggregate_gradient_clip_fraction": metrics.get(
            "aggregate_gradient_clip_fraction"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if args.hidden_dim != 64 or args.max_seq_len != 256:
        raise ValueError(
            "Frozen Intermittent contract requires hidden_dim=64/max_seq_len=256"
        )
    if args.output_dir.exists() and not args.force_rerun:
        status_path = args.output_dir / "status.json"
        if status_path.exists():
            raise FileExistsError(
                f"Output already has a status artifact: {status_path}"
            )

    dataset_contract = DATASET_CONTRACTS["intermittent_frozen_5000"]
    data_sha256 = sha256_file(args.data)
    manifest_sha256 = sha256_file(args.split_manifest)
    if data_sha256 != dataset_contract["data_sha256"]:
        raise ValueError(f"Unexpected fixed-split SHA-256: {data_sha256}")
    if manifest_sha256 != dataset_contract["split_manifest_sha256"]:
        raise ValueError(f"Unexpected split-manifest SHA-256: {manifest_sha256}")

    required = {
        "oper_part_no",
        "seq",
        "delta_t",
        "demand_qty",
        "chronological_split",
    }
    train_raw = load_train_only_frame(args.data)
    missing = sorted(required - set(train_raw.columns))
    if missing:
        raise ValueError(f"Fixed split is missing columns: {missing}")
    if (
        train_raw.height < 1
        or train_raw["chronological_split"].unique().to_list() != ["train"]
    ):
        raise ValueError("Train-only runner must contain train rows exclusively")

    train_log_qty = np.log1p(
        train_raw["demand_qty"].to_numpy().astype(np.float64)
    )
    frame = prepare_count_frame(train_raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_stability.log"
    launch_contract = {
        "schema_version": 1,
        "status": "running",
        "experiment": "count_aware_time_head_v2_train_only_stability",
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha256,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha256,
        "loaded_split": "train_only",
        "train_rows": train_raw.height,
        "backbone": BACKBONE,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lookback_weeks": args.lookback_weeks,
        "max_seq_len": args.max_seq_len,
        "hidden_dim": args.hidden_dim,
        "grad_clip": args.grad_clip,
        "variants": VARIANT_SPECS,
        "execution_order": ["H0", "H1", "H2_only_if_H1_fails"],
        "stability_thresholds": STABILITY_THRESHOLDS,
        "selection_source": "train_stability_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "execution_host": os.uname().nodename,
        "execution_role": args.execution_role,
    }
    save_json(args.output_dir / "launch_contract.json", launch_contract)

    summaries: list[dict[str, Any]] = []
    histories: list[dict[str, Any]] = []
    for variant in ("H0", "H1"):
        summary, history = run_variant(
            args=args,
            frame=frame,
            train_log_mean=float(train_log_qty.mean()),
            train_log_std=float(train_log_qty.std()),
            variant=variant,
            log_path=log_path,
        )
        summaries.append(summary)
        histories.extend(history)

    if should_run_h2(summaries[-1]):
        summary, history = run_variant(
            args=args,
            frame=frame,
            train_log_mean=float(train_log_qty.mean()),
            train_log_std=float(train_log_qty.std()),
            variant="H2",
            log_path=log_path,
        )
        summaries.append(summary)
        histories.extend(history)

    by_variant = {summary["variant"]: summary for summary in summaries}
    selected_variant = select_stable_variant(by_variant)
    decision = {
        "selected_variant": selected_variant,
        "validation_screening_opened": selected_variant is not None,
        "h2_executed": "H2" in by_variant,
        "h2_execution_reason": (
            "H1_failed_train_only_stability_gate"
            if "H2" in by_variant
            else "not_run_because_H1_passed"
        ),
        "selection_source": "train_stability_only",
        "validation_evaluated": False,
        "held_out_test_evaluated": False,
        "variant_gates": {
            variant: summary["stability_gate"]
            for variant, summary in by_variant.items()
        },
    }
    save_json(args.output_dir / "decision.json", decision)
    if histories:
        write_csv(args.output_dir / "epoch_metrics.csv", histories)
    write_csv(
        args.output_dir / "variant_summaries.csv",
        [summary_row(summary) for summary in summaries],
    )
    decision_lines = [
        "# Time Head v2 Train-Only Decision",
        "",
        f"- Selected variant: `{selected_variant or 'none'}`",
        f"- H2 executed: `{decision['h2_executed']}`",
        f"- Validation screening opened: `{decision['validation_screening_opened']}`",
        "- Validation evaluated: `False`",
        "- Held-out test evaluated: `False`",
    ]
    (args.output_dir / "decision.md").write_text(
        "\n".join(decision_lines) + "\n",
        encoding="utf-8",
    )
    launch_contract.update(
        {
            "status": "complete",
            "executed_variants": list(by_variant),
            "selected_variant": selected_variant,
            "validation_screening_opened": selected_variant is not None,
        }
    )
    save_json(args.output_dir / "launch_contract.json", launch_contract)
    save_json(
        args.output_dir / "status.json",
        {
            "status": "complete",
            "selected_variant": selected_variant,
            "validation_evaluated": False,
            "held_out_test_evaluated": False,
            "source_revision": args.source_revision,
        },
    )
    append_log(
        log_path,
        f"[complete] selected_variant={selected_variant or 'none'} "
        f"executed_variants={','.join(by_variant)}",
    )


if __name__ == "__main__":
    main()
