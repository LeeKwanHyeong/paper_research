#!/usr/bin/env python3
"""Freeze tail-loss weight from Intermittent train-only gradient norms."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import torch

from paper.scripts.run_count_aware_tpp_backbone_control import (
    TAIL_HEAD_ONLY_VARIANT,
    build_model,
    prepare_count_frame,
    target_outputs,
)
from paper.scripts.run_intermittent_log_backbone_control import (
    EXPECTED_DATA_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
)
from paper.scripts.run_taxi_quantity_interface_ablation import (
    make_loader,
    save_json,
    set_seed,
    sha256_file,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lookback-weeks", type=int, default=520)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--warmup-train-batches", type=int, default=128)
    parser.add_argument("--probe-train-batches", type=int, default=64)
    parser.add_argument("--target-gradient-ratio", type=float, default=0.10)
    parser.add_argument("--lambda-min", type=float, default=1e-4)
    parser.add_argument("--lambda-max", type=float, default=100.0)
    parser.add_argument("--tail-threshold", type=float, default=46.0)
    parser.add_argument("--tail-normalization-scale", type=float, default=46.0)
    parser.add_argument("--tail-clip-cap", type=float, default=187.0)
    parser.add_argument("--tail-huber-delta", type=float, default=1.0)
    return parser.parse_args()


def calibrate_lambda(
    main_gradient_norm: float,
    tail_gradient_norm: float,
    *,
    target_ratio: float,
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    values = (main_gradient_norm, tail_gradient_norm, target_ratio, minimum, maximum)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Gradient norms, ratio, and bounds must be finite and positive")
    if minimum > maximum:
        raise ValueError("minimum must not exceed maximum")
    raw = target_ratio * main_gradient_norm / tail_gradient_norm
    return float(raw), float(min(max(raw, minimum), maximum))


def gradient_norm(
    loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> float:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        retain_graph=retain_graph,
        allow_unused=False,
    )
    squared = torch.zeros((), device=loss.device)
    for gradient in gradients:
        squared = squared + torch.square(gradient).sum()
    return float(torch.sqrt(squared).detach().cpu().item())


def write_briefing(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Count-aware Tail Lambda Train-only Calibration",
        "",
        f"- 상태: **{payload['status'].upper()}**",
        f"- Warm-up / probe batches: `{payload['warmup_train_batches']}` / "
        f"`{payload['probe_train_batches']}`",
        f"- Mean log-MSE head gradient norm: `{payload['mean_log_mse_head_gradient_norm']:.8f}`",
        f"- Mean unweighted tail head gradient norm: "
        f"`{payload['mean_tail_head_gradient_norm']:.8f}`",
        f"- Raw lambda: `{payload['raw_lambda_tail']:.8f}`",
        f"- Frozen lambda: `{payload['frozen_lambda_tail']:.8f}`",
        f"- Weighted tail/main gradient ratio: "
        f"`{payload['weighted_tail_to_main_gradient_ratio']:.8f}`",
        f"- Probe tail targets: `{payload['probe_tail_count']}` / "
        f"`{payload['probe_target_count']}`",
        "- Validation/test rows read: `0 / 0`",
        "",
        "계수는 train split의 quantity-head gradient만 사용해 한 번 산출했다. "
        "T1과 T2 screening에 같은 값을 사용한다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    if args.warmup_train_batches < 1 or args.probe_train_batches < 1:
        raise ValueError("Warm-up and probe batch counts must be positive")
    if args.hidden_dim != 64 or args.max_seq_len != 256:
        raise ValueError("Frozen calibration requires hidden_dim=64 and max_seq_len=256")

    data_sha = sha256_file(args.data)
    manifest_sha = sha256_file(args.split_manifest)
    if data_sha != EXPECTED_DATA_SHA256:
        raise ValueError(f"Unexpected data SHA-256: {data_sha}")
    if manifest_sha != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise ValueError(f"Unexpected split manifest SHA-256: {manifest_sha}")
    raw = pl.read_parquet(args.data).sort(["oper_part_no", "seq"])
    train_raw = raw.filter(pl.col("chronological_split") == "train")
    if train_raw.height == 0:
        raise ValueError("Train split is empty")
    frame = prepare_count_frame(train_raw)
    train_log = np.log1p(
        train_raw["demand_qty"].to_numpy().astype(np.float64)
    )

    generator = set_seed(args.seed)
    warmup_loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=generator,
    )
    probe_generator = torch.Generator().manual_seed(args.seed + 10_000)
    probe_loader = make_loader(
        frame,
        target_split="train",
        batch_size=args.batch_size,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=args.max_seq_len,
        shuffle=True,
        generator=probe_generator,
    )
    model, encoder_config = build_model(
        "titantpp",
        hidden_dim=args.hidden_dim,
        train_log_mean=float(train_log.mean()),
        max_seq_len=args.max_seq_len,
        quantity_variant=TAIL_HEAD_ONLY_VARIANT,
        lambda_tail=1.0,
        tail_threshold=args.tail_threshold,
        tail_normalization_scale=args.tail_normalization_scale,
        tail_clip_cap=args.tail_clip_cap,
        tail_huber_delta=args.tail_huber_delta,
    )
    model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    warmup_completed = 0
    for batch_index, (_, dts, mask, _, quantities) in enumerate(warmup_loader):
        if batch_index >= args.warmup_train_batches:
            break
        if quantities is None:
            raise ValueError("Calibration requires raw quantities")
        outputs = target_outputs(
            model,
            dts.to(args.device),
            mask.to(args.device),
            quantities.to(args.device),
            lambda_log_qty=1.0,
        )
        baseline_joint = (outputs["time_loss"] + outputs["log_qty_loss"]).mean()
        optimizer.zero_grad(set_to_none=True)
        baseline_joint.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        warmup_completed += 1
    if warmup_completed != args.warmup_train_batches:
        raise RuntimeError(
            f"Warm-up loader exhausted at {warmup_completed} batches"
        )

    model.eval()
    head_parameters = tuple(model.quantity_head.parameters())
    main_norms: list[float] = []
    tail_norms: list[float] = []
    probe_tail_count = 0
    probe_target_count = 0
    for batch_index, (_, dts, mask, _, quantities) in enumerate(probe_loader):
        if batch_index >= args.probe_train_batches:
            break
        if quantities is None:
            raise ValueError("Calibration requires raw quantities")
        outputs = target_outputs(
            model,
            dts.to(args.device),
            mask.to(args.device),
            quantities.to(args.device),
            lambda_log_qty=1.0,
        )
        main_norms.append(gradient_norm(
            outputs["log_qty_loss"].mean(),
            head_parameters,
            retain_graph=True,
        ))
        tail_norms.append(gradient_norm(
            outputs["tail_aux_loss"].mean(),
            head_parameters,
            retain_graph=False,
        ))
        probe_tail_count += int(outputs["tail_indicator"].sum().detach().cpu().item())
        probe_target_count += int(outputs["tail_indicator"].numel())
    if len(main_norms) != args.probe_train_batches:
        raise RuntimeError(f"Probe loader exhausted at {len(main_norms)} batches")

    mean_main = float(np.mean(main_norms))
    mean_tail = float(np.mean(tail_norms))
    raw_lambda, frozen_lambda = calibrate_lambda(
        mean_main,
        mean_tail,
        target_ratio=args.target_gradient_ratio,
        minimum=args.lambda_min,
        maximum=args.lambda_max,
    )
    payload = {
        "schema_version": 1,
        "status": "complete",
        "scope": "intermittent_train_only",
        "source_revision": args.source_revision,
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha,
        "train_rows_loaded": int(train_raw.height),
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "seed": args.seed,
        "device": args.device,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_train_batches": warmup_completed,
        "probe_train_batches": len(main_norms),
        "probe_shuffle": True,
        "probe_loader_seed": args.seed + 10_000,
        "probe_target_count": probe_target_count,
        "probe_tail_count": probe_tail_count,
        "probe_tail_share": probe_tail_count / probe_target_count,
        "mean_log_mse_head_gradient_norm": mean_main,
        "mean_tail_head_gradient_norm": mean_tail,
        "target_gradient_ratio": args.target_gradient_ratio,
        "raw_lambda_tail": raw_lambda,
        "frozen_lambda_tail": frozen_lambda,
        "weighted_tail_to_main_gradient_ratio": frozen_lambda * mean_tail / mean_main,
        "lambda_bounds": [args.lambda_min, args.lambda_max],
        "tail_contract": {
            "threshold": args.tail_threshold,
            "normalization_scale": args.tail_normalization_scale,
            "clip_cap": args.tail_clip_cap,
            "huber_delta": args.tail_huber_delta,
        },
        "encoder_config": encoder_config,
        "warmup_state_sha256": canonical_state_dict_sha256(model.state_dict()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "calibration.json", payload)
    write_briefing(args.output_dir / "briefing.md", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
