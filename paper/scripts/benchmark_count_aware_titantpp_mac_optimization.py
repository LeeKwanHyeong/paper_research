#!/usr/bin/env python3
"""Verify semantic equivalence and profile optimized TitanTPP-MAC on CUDA."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from models.TPPs.CountAwareFactory import build_count_aware_model  # noqa: E402
from models.TPPs.CountAwareTPP import CountAwareTitanTPP  # noqa: E402
from models.Titan.common.titans_mac_optimized import (  # noqa: E402
    optimization_metadata,
)
from paper.scripts.count_aware_tpp_backbone.core import (  # noqa: E402
    right_pad_batch,
    target_outputs,
)
from paper.scripts.count_aware_titantpp_mac_runtime import (  # noqa: E402
    build_count_aware_titantpp_mac_primary,
)
from simple_lab_test.search.common.runner import (  # noqa: E402
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


B0 = "titantpp"
B1 = "titantpp_titans_mac"
VARIANT = "count_only_log_regression"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[64, 84, 256])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=5)
    parser.add_argument("--maximum-ratio", type=float, default=3.0)
    parser.add_argument("--frozen-b1-checkpoint", type=Path)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def make_batch(
    batch_size: int,
    sequence_length: int,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260830 + sequence_length)
    dts = torch.rand(batch_size, sequence_length, generator=generator, device=device)
    dts = 0.01 + 4.0 * dts
    quantities = torch.poisson(
        3.0 + 10.0 * torch.rand(
            batch_size,
            sequence_length,
            generator=generator,
            device=device,
        ),
        generator=generator,
    )
    mask = torch.ones(batch_size, sequence_length, device=device, dtype=torch.bool)
    if sequence_length > 8:
        mask[-1, -3:] = False
    return dts, quantities, mask


def build_model(
    backbone: str,
    max_seq_len: int,
    device: str,
    *,
    optimize_b1: bool = True,
) -> CountAwareTitanTPP:
    torch.manual_seed(123)
    kwargs = {
        "hidden_dim": 64,
        "train_log_mean": 2.0,
        "train_log_std": 1.0,
        "max_seq_len": max_seq_len,
    }
    if backbone == B1:
        model, _ = build_count_aware_titantpp_mac_primary(
            optimize_execution=optimize_b1,
            **kwargs,
        )
    else:
        model, _ = build_count_aware_model(backbone, **kwargs)
    if not isinstance(model, CountAwareTitanTPP):
        raise TypeError("Expected CountAwareTitanTPP")
    return model.to(device)


def diagnostic_target_outputs(
    model: CountAwareTitanTPP,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if model.titans_mac_encoder is None:
        raise TypeError("Diagnostic target path requires B1")
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    batch_ids = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2
    history_quantities = quantities.clone()
    history_quantities[batch_ids, target_positions] = 0.0
    write_mask = mask.clone()
    write_mask[batch_ids, target_positions] = False
    encoded, _, diagnostics = model.encode_with_memory_state(
        dts,
        history_quantities,
        mask,
        memory_write_mask=write_mask,
    )
    hidden = encoded[batch_ids, history_positions]
    true_dt = dts[batch_ids, target_positions].float()
    true_quantity = quantities[batch_ids, target_positions].float()
    quantity = model.quantity_outputs(hidden, true_quantity)
    time_loss = -model.log_f_dt(hidden, true_dt)
    return {
        "joint_loss": time_loss + quantity["train_loss"],
        "time_loss": time_loss,
        "pred_qty": quantity["point_prediction"],
        "diagnostic_write_count": diagnostics["write_applied"].sum(),
    }


def semantic_equivalence(device: str) -> dict[str, Any]:
    sequence_length = 33
    dts, quantities, mask = make_batch(7, sequence_length, device=device)
    optimized = build_model(B1, sequence_length, device, optimize_b1=True).eval()
    diagnostic = build_model(B1, sequence_length, device, optimize_b1=False).eval()
    diagnostic.load_state_dict(optimized.state_dict())
    if diagnostic.titans_mac_encoder is None:
        raise AssertionError("B1 encoder missing")
    diagnostic.titans_mac_encoder.neural_memory.compile_cuda_scan = False

    optimized_outputs = target_outputs(
        optimized, dts, mask, quantities, lambda_log_qty=1.0
    )
    optimized_outputs["joint_loss"].mean().backward()
    diagnostic_outputs = diagnostic_target_outputs(
        diagnostic, dts, mask, quantities
    )
    diagnostic_outputs["joint_loss"].mean().backward()
    output_difference = max(
        float(
            (optimized_outputs[name] - diagnostic_outputs[name])
            .abs()
            .max()
            .detach()
            .cpu()
        )
        for name in ("joint_loss", "time_loss", "pred_qty")
    )
    optimized_gradients = {
        name: parameter.grad
        for name, parameter in optimized.named_parameters()
        if parameter.grad is not None
    }
    diagnostic_gradients = {
        name: parameter.grad
        for name, parameter in diagnostic.named_parameters()
        if parameter.grad is not None
    }
    if optimized_gradients.keys() != diagnostic_gradients.keys():
        raise AssertionError("State-only and diagnostic gradient coverage differs")
    gradient_difference = max(
        float(
            (optimized_gradients[name] - diagnostic_gradients[name])
            .abs()
            .max()
            .detach()
            .cpu()
        )
        for name in optimized_gradients
    )
    passed = output_difference <= 1e-5 and gradient_difference <= 1e-4
    if not passed:
        raise AssertionError(
            f"Semantic mismatch: output={output_difference}, gradient={gradient_difference}"
        )
    return {
        "passed": True,
        "maximum_output_difference": output_difference,
        "maximum_gradient_difference": gradient_difference,
    }


def strict_checkpoint_replay(
    checkpoint_path: Path | None,
    *,
    device: str,
) -> dict[str, Any] | None:
    if checkpoint_path is None:
        return None
    payload = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    if payload.get("backbone") != B1 or payload.get("variant") != VARIANT:
        raise ValueError("Frozen checkpoint is not a B1 direct log-MSE checkpoint")
    encoder = payload["encoder_config"]
    interface = payload["interface_meta"]
    model, _ = build_count_aware_titantpp_mac_primary(
        hidden_dim=int(encoder["d_model"]),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface.get("train_target_std", 1.0)),
        max_seq_len=int(encoder["max_len"]),
    )
    missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=True)
    if missing or unexpected:
        raise AssertionError("Strict checkpoint load changed parameter keys")
    state_sha = canonical_state_dict_sha256(model.state_dict())
    if state_sha != payload["model_state_sha256"]:
        raise AssertionError("Strict checkpoint state digest changed")
    model.to(device).eval()
    dts, quantities, mask = make_batch(4, min(32, int(encoder["max_len"])), device=device)
    with torch.no_grad():
        before = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored, _ = build_count_aware_titantpp_mac_primary(
        hidden_dim=int(encoder["d_model"]),
        train_log_mean=float(interface["train_target_mean"]),
        train_log_std=float(interface.get("train_target_std", 1.0)),
        max_seq_len=int(encoder["max_len"]),
    )
    restored.load_state_dict(torch.load(buffer, map_location="cpu"), strict=True)
    restored.to(device).eval()
    with torch.no_grad():
        after = target_outputs(restored, dts, mask, quantities, lambda_log_qty=1.0)
    difference = float((before["pred_qty"] - after["pred_qty"]).abs().max().cpu())
    if difference > 1e-6:
        raise AssertionError("Optimized checkpoint replay changed predictions")
    return {
        "passed": True,
        "model_state_sha256": state_sha,
        "prediction_maximum_absolute_difference": difference,
    }


def training_step(
    model: CountAwareTitanTPP,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    dts, quantities, mask = batch
    optimizer.zero_grad(set_to_none=True)
    outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    outputs["joint_loss"].mean().backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()


def profile_backbone(
    backbone: str,
    *,
    batch_size: int,
    sequence_length: int,
    warmup_steps: int,
    timed_steps: int,
    device: str,
) -> dict[str, Any]:
    model = build_model(backbone, sequence_length, device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = make_batch(batch_size, sequence_length, device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    cold_started = time.perf_counter()
    training_step(model, optimizer, batch)
    synchronize(device)
    cold_seconds = time.perf_counter() - cold_started
    for _ in range(warmup_steps):
        training_step(model, optimizer, batch)
    samples = []
    for _ in range(timed_steps):
        synchronize(device)
        started = time.perf_counter()
        training_step(model, optimizer, batch)
        synchronize(device)
        samples.append(time.perf_counter() - started)
    median = statistics.median(samples)
    result = {
        "backbone": backbone,
        "sequence_length": sequence_length,
        "cold_training_step_seconds": cold_seconds,
        "estimated_compile_overhead_seconds": max(0.0, cold_seconds - median),
        "steady_training_step_samples_seconds": samples,
        "steady_training_step_median_seconds": median,
        "steady_training_step_p95_seconds": sorted(samples)[
            min(len(samples) - 1, math.ceil(0.95 * len(samples)) - 1)
        ],
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.startswith("cuda")
            else None
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / (1024**2)
            if device.startswith("cuda")
            else None
        ),
    }
    del optimizer, model, batch
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.batch_size < 1 or args.warmup_steps < 0 or args.timed_steps < 1:
        raise ValueError("Invalid timing arguments")
    if any(length < 4 for length in args.sequence_lengths):
        raise ValueError("Every sequence length must be at least four")
    if len(args.source_revision) != 40:
        raise ValueError("source_revision must be a full commit SHA")

    semantic = semantic_equivalence(args.device)
    checkpoint = strict_checkpoint_replay(
        args.frozen_b1_checkpoint,
        device=args.device,
    )
    timing = []
    target_met = True
    for sequence_length in args.sequence_lengths:
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        b0 = profile_backbone(
            B0,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            device=args.device,
        )
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
        b1 = profile_backbone(
            B1,
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            device=args.device,
        )
        ratio = (
            b1["steady_training_step_median_seconds"]
            / b0["steady_training_step_median_seconds"]
        )
        passed = ratio <= args.maximum_ratio
        target_met = target_met and passed
        timing.append(
            {
                "sequence_length": sequence_length,
                "b0": b0,
                "b1": b1,
                "b1_b0_steady_training_step_ratio": ratio,
                "maximum_ratio": args.maximum_ratio,
                "target_met": passed,
            }
        )
    payload = {
        "status": "complete" if target_met else "complete_target_not_met",
        "source_revision": args.source_revision,
        "device": args.device,
        "cuda_device": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
        ),
        "batch_size": args.batch_size,
        "optimization": optimization_metadata(),
        "semantic_equivalence": semantic,
        "frozen_checkpoint_replay": checkpoint,
        "timing": timing,
        "three_x_target_met": target_met,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
