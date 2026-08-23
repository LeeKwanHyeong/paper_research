#!/usr/bin/env python3
"""Run finite forward/backward checks for official count-aware model roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT, TAIL_SHARED_VARIANT
from paper.scripts.count_aware_tpp_backbone.constants import (
    FROZEN_TAIL_LAMBDA,
    T0_COMMON_BACKBONES,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_case(backbone: str, variant: str, device: str) -> dict[str, object]:
    is_t1 = variant == TAIL_SHARED_VARIANT
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=64,
        train_log_mean=1.0,
        train_log_std=1.0,
        max_seq_len=16,
        quantity_variant=variant,
        lambda_tail=FROZEN_TAIL_LAMBDA if is_t1 else 0.0,
        tail_threshold=46.0,
        tail_normalization_scale=46.0,
        tail_clip_cap=187.0,
        tail_huber_delta=1.0,
    )
    model.to(device)
    dts = torch.tensor(
        [[1, 2, 1, 3, 2, 1], [1, 1, 2, 2, 4, 1]],
        dtype=torch.float32,
        device=device,
    )
    quantities = torch.tensor(
        [[2, 4, 8, 16, 64, 100], [1, 3, 7, 12, 50, 200]],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones_like(dts, dtype=torch.bool)
    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["joint_loss"].mean().backward()
    if not all(torch.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError(f"Non-finite output: {backbone}/{variant}")
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError(f"Non-finite gradient: {backbone}/{variant}")
    return {
        "backbone": backbone,
        "variant": variant,
        "candidate": metadata["candidate_name"],
        "joint_loss": float(outputs["joint_loss"].mean().detach().cpu()),
        "finite": True,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    cases = [(backbone, LOG_MSE_VARIANT) for backbone in T0_COMMON_BACKBONES]
    cases.append(("titantpp", TAIL_SHARED_VARIANT))
    results = [run_case(backbone, variant, args.device) for backbone, variant in cases]
    payload = {
        "status": "complete",
        "device": args.device,
        "cuda_device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "case_count": len(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
