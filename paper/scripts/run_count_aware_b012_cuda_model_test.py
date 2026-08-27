#!/usr/bin/env python3
"""Validate B0/B1/B2 CUDA numerics, state replay, and training-step cost."""

from __future__ import annotations

import argparse
from dataclasses import fields
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

import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import LOG_MSE_VARIANT
from paper.scripts.count_aware_tpp_backbone.constants import TITAN_B012_BACKBONES
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=3)
    parser.add_argument("--maximum-step-ratio", type=float, default=3.0)
    parser.add_argument(
        "--compile-model-backbones",
        default="",
        help="Comma-separated backbones to compile in-place for timing only.",
    )
    return parser.parse_args()


def build_model(backbone: str, *, max_seq_len: int, device: str):
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=64,
        train_log_mean=1.0,
        train_log_std=1.0,
        max_seq_len=max_seq_len,
        quantity_variant=LOG_MSE_VARIANT,
        lambda_tail=0.0,
        tail_threshold=46.0,
        tail_normalization_scale=46.0,
        tail_clip_cap=187.0,
        tail_huber_delta=1.0,
        time_head_mode="legacy_clamped_rmtpp",
    )
    return model.to(device), metadata


def make_batch(
    *, batch_size: int, sequence_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row = torch.arange(sequence_length, device=device, dtype=torch.float32)
    dts = 1.0 + torch.remainder(row, 7.0)
    quantities = 1.0 + torch.remainder(row.square() + 3.0 * row, 211.0)
    dts = dts.unsqueeze(0).repeat(batch_size, 1)
    quantities = quantities.unsqueeze(0).repeat(batch_size, 1)
    row_offsets = torch.arange(batch_size, device=device).unsqueeze(1)
    quantities = quantities + torch.remainder(row_offsets, 5).float()
    mask = torch.ones_like(dts, dtype=torch.bool)
    return dts, quantities, mask


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def assert_finite_outputs_and_gradients(
    backbone: str,
    outputs: dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> None:
    tensor_outputs = [value for value in outputs.values() if torch.is_tensor(value)]
    if not tensor_outputs or not all(torch.isfinite(value).all() for value in tensor_outputs):
        raise FloatingPointError(f"Non-finite output for {backbone}")
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError(f"Non-finite gradient for {backbone}")


def state_payload(state: Any) -> dict[str, torch.Tensor | None]:
    return {
        field.name: getattr(state, field.name).detach()
        if torch.is_tensor(getattr(state, field.name))
        else None
        for field in fields(state)
    }


def restore_state(template: Any, payload: dict[str, Any], device: str) -> Any:
    values = {
        field.name: (
            None
            if payload[field.name] is None
            else payload[field.name].to(device=device)
        )
        for field in fields(template)
    }
    return type(template)(**values)


def maximum_state_difference(left: Any, right: Any) -> float:
    maximum = 0.0
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if left_value is None or right_value is None:
            if left_value is not right_value:
                raise AssertionError(f"State optional field differs: {field.name}")
            continue
        if left_value.dtype.is_floating_point:
            difference = float((left_value - right_value).abs().max().detach().cpu())
            maximum = max(maximum, difference)
        elif not torch.equal(left_value, right_value):
            raise AssertionError(f"State discrete field differs: {field.name}")
    return maximum


def checkpoint_roundtrip(
    backbone: str,
    model: torch.nn.Module,
    dts: torch.Tensor,
    quantities: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str,
) -> dict[str, float | bool]:
    model.eval()
    with torch.no_grad():
        before = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
    buffer = io.BytesIO()
    torch.save({"model_state": model.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location=device)
    restored, _ = build_model(
        backbone,
        max_seq_len=dts.size(1),
        device=device,
    )
    restored.load_state_dict(checkpoint["model_state"])
    restored.eval()
    with torch.no_grad():
        after = target_outputs(
            restored,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
    prediction_diff = float(
        (before["pred_qty"] - after["pred_qty"]).abs().max().detach().cpu()
    )
    time_nll_diff = float(
        (before["time_loss"] - after["time_loss"]).abs().max().detach().cpu()
    )
    if prediction_diff > 1e-6 or time_nll_diff > 1e-6:
        raise AssertionError(f"Model checkpoint replay failed for {backbone}")
    return {
        "passed": True,
        "maximum_prediction_difference": prediction_diff,
        "maximum_time_nll_difference": time_nll_diff,
    }


def compiled_eager_equivalence(
    backbone: str,
    dts: torch.Tensor,
    quantities: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str,
) -> dict[str, float | bool] | None:
    if backbone == "titantpp":
        return None
    torch.manual_seed(777)
    compiled_model, _ = build_model(
        backbone,
        max_seq_len=dts.size(1),
        device=device,
    )
    eager_model, _ = build_model(
        backbone,
        max_seq_len=dts.size(1),
        device=device,
    )
    eager_model.load_state_dict(compiled_model.state_dict())
    if backbone == "titantpp_titans_mac":
        if eager_model.titans_mac_encoder is None:
            raise AssertionError("B1 Titans-MAC encoder is missing")
        eager_model.titans_mac_encoder.neural_memory.compile_cuda_scan = False
        for layer in eager_model.titans_mac_encoder.layers:
            layer.compile_cuda_block = False
    else:
        if eager_model.tpp_gated_memory is None:
            raise AssertionError("B2 gated memory is missing")
        eager_model.tpp_gated_memory.compile_cuda_scan = False
    compiled_model.eval()
    eager_model.eval()

    compiled_outputs = target_outputs(
        compiled_model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    compiled_outputs["joint_loss"].mean().backward()
    eager_outputs = target_outputs(
        eager_model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    eager_outputs["joint_loss"].mean().backward()

    output_difference = max(
        float(
            (compiled_outputs[name] - eager_outputs[name])
            .abs()
            .max()
            .detach()
            .cpu()
        )
        for name in ("joint_loss", "time_loss", "pred_qty")
    )
    compiled_gradients = {
        name: parameter.grad
        for name, parameter in compiled_model.named_parameters()
        if parameter.grad is not None
    }
    eager_gradients = {
        name: parameter.grad
        for name, parameter in eager_model.named_parameters()
        if parameter.grad is not None
    }
    if compiled_gradients.keys() != eager_gradients.keys():
        raise AssertionError(f"Compiled/eager gradient coverage differs for {backbone}")
    gradient_difference = max(
        float(
            (compiled_gradients[name] - eager_gradients[name])
            .abs()
            .max()
            .detach()
            .cpu()
        )
        for name in compiled_gradients
    )
    if output_difference > 1e-5 or gradient_difference > 1e-4:
        raise AssertionError(
            f"Compiled/eager mismatch for {backbone}: "
            f"output={output_difference} gradient={gradient_difference}"
        )
    return {
        "passed": True,
        "maximum_output_difference": output_difference,
        "maximum_gradient_difference": gradient_difference,
    }


def online_state_roundtrip(
    backbone: str,
    model: torch.nn.Module,
    dts: torch.Tensor,
    quantities: torch.Tensor,
    mask: torch.Tensor,
    *,
    device: str,
) -> dict[str, float | bool] | None:
    if backbone == "titantpp":
        return None
    model.eval()
    split = max(1, dts.size(1) // 2)
    series_ids = torch.arange(dts.size(0), device=device, dtype=torch.long) + 100
    with torch.no_grad():
        _, prefix_state, _ = model.encode_with_memory_state(
            dts[:, :split],
            quantities[:, :split],
            mask[:, :split],
            series_ids=series_ids,
        )
        expected, expected_state, _ = model.encode_with_memory_state(
            dts[:, split:],
            quantities[:, split:],
            mask[:, split:],
            state=prefix_state,
            series_ids=series_ids,
        )

    buffer = io.BytesIO()
    torch.save(
        {
            "model_state": model.state_dict(),
            "memory_state": state_payload(prefix_state),
        },
        buffer,
    )
    buffer.seek(0)
    checkpoint = torch.load(buffer, map_location=device)
    restored_model, _ = build_model(
        backbone,
        max_seq_len=dts.size(1),
        device=device,
    )
    restored_model.load_state_dict(checkpoint["model_state"])
    restored_model.eval()
    restored_state = restore_state(prefix_state, checkpoint["memory_state"], device)
    with torch.no_grad():
        actual, actual_state, _ = restored_model.encode_with_memory_state(
            dts[:, split:],
            quantities[:, split:],
            mask[:, split:],
            state=restored_state,
            series_ids=series_ids,
        )
    output_diff = float((expected - actual).abs().max().detach().cpu())
    state_diff = maximum_state_difference(expected_state, actual_state)
    if output_diff > 1e-6 or state_diff > 1e-6:
        raise AssertionError(f"Online memory-state replay failed for {backbone}")
    return {
        "passed": True,
        "maximum_output_difference": output_diff,
        "maximum_state_difference": state_diff,
    }


def correctness_case(
    backbone: str,
    *,
    batch_size: int,
    sequence_length: int,
    device: str,
) -> dict[str, Any]:
    torch.manual_seed(42)
    model, metadata = build_model(
        backbone,
        max_seq_len=sequence_length,
        device=device,
    )
    model.train()
    dts, quantities, mask = make_batch(
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    outputs["joint_loss"].mean().backward()
    assert_finite_outputs_and_gradients(backbone, outputs, model)

    model.zero_grad(set_to_none=True)
    extreme_dts = dts.clone()
    extreme_quantities = quantities.clone()
    extreme_dts[:, -4:] = torch.tensor(
        [1e-6, 1e3, 1e6, 1.0], device=device
    )
    extreme_quantities[:, -4:] = torch.tensor(
        [0.0, 1e6, 1e9, 1.0], device=device
    )
    extreme_outputs = target_outputs(
        model,
        extreme_dts,
        mask,
        extreme_quantities,
        lambda_log_qty=1.0,
    )
    extreme_outputs["joint_loss"].mean().backward()
    assert_finite_outputs_and_gradients(backbone, extreme_outputs, model)

    model.zero_grad(set_to_none=True)
    model_checkpoint = checkpoint_roundtrip(
        backbone,
        model,
        dts,
        quantities,
        mask,
        device=device,
    )
    memory_checkpoint = online_state_roundtrip(
        backbone,
        model,
        dts[:2],
        quantities[:2],
        mask[:2],
        device=device,
    )
    compiled_eager = compiled_eager_equivalence(
        backbone,
        dts,
        quantities,
        mask,
        device=device,
    )
    return {
        "backbone": backbone,
        "backbone_contract_id": metadata["backbone_contract_id"],
        "candidate": metadata["candidate_name"],
        "finite_forward_backward": True,
        "extreme_input_finite": True,
        "model_checkpoint_roundtrip": model_checkpoint,
        "online_memory_state_roundtrip": memory_checkpoint,
        "compiled_eager_equivalence": compiled_eager,
    }


def training_step_times(
    backbone: str,
    *,
    batch_size: int,
    sequence_length: int,
    warmup_steps: int,
    timed_steps: int,
    device: str,
    compile_model: bool,
) -> list[float]:
    torch.manual_seed(123)
    model, _ = build_model(
        backbone,
        max_seq_len=sequence_length,
        device=device,
    )
    if compile_model:
        if backbone == "titantpp_titans_mac":
            if model.titans_mac_encoder is None:
                raise AssertionError("B1 Titans-MAC encoder is missing")
            for layer in model.titans_mac_encoder.layers:
                layer.compile_cuda_block = False
        model.compile(fullgraph=False, dynamic=False, mode="reduce-overhead")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dts, quantities, mask = make_batch(
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=device,
    )
    elapsed: list[float] = []
    for step in range(warmup_steps + timed_steps):
        synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        outputs["joint_loss"].mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        synchronize(device)
        if step >= warmup_steps:
            elapsed.append(time.perf_counter() - started)
    if len(elapsed) != timed_steps or not all(
        math.isfinite(value) and value > 0.0 for value in elapsed
    ):
        raise RuntimeError(f"Invalid timing samples for {backbone}: {elapsed}")
    return elapsed


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.batch_size < 1 or args.sequence_length < 4:
        raise ValueError("batch-size must be positive and sequence-length at least four")
    if args.warmup_steps < 0 or args.timed_steps < 1:
        raise ValueError("warmup-steps must be nonnegative and timed-steps positive")
    if args.maximum_step_ratio <= 0.0:
        raise ValueError("maximum-step-ratio must be positive")
    compiled_backbones = {
        value.strip()
        for value in args.compile_model_backbones.split(",")
        if value.strip()
    }
    unsupported_compiled = compiled_backbones.difference(TITAN_B012_BACKBONES)
    if unsupported_compiled:
        raise ValueError(
            "Unsupported compile-model backbones: "
            + ", ".join(sorted(unsupported_compiled))
        )

    correctness = [
        correctness_case(
            backbone,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            device=args.device,
        )
        for backbone in TITAN_B012_BACKBONES
    ]
    timings = {
        backbone: training_step_times(
            backbone,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            device=args.device,
            compile_model=backbone in compiled_backbones,
        )
        for backbone in TITAN_B012_BACKBONES
    }
    medians = {
        backbone: statistics.median(samples)
        for backbone, samples in timings.items()
    }
    b0_seconds = medians["titantpp"]
    timing_rows = []
    speed_gate_passed = True
    for backbone in TITAN_B012_BACKBONES:
        ratio = medians[backbone] / b0_seconds
        passed = backbone == "titantpp" or ratio <= args.maximum_step_ratio
        speed_gate_passed = speed_gate_passed and passed
        timing_rows.append(
            {
                "backbone": backbone,
                "samples_seconds": timings[backbone],
                "median_seconds": medians[backbone],
                "ratio_vs_b0": ratio,
                "maximum_allowed_ratio": args.maximum_step_ratio,
                "passed": passed,
            }
        )

    payload = {
        "status": "complete" if speed_gate_passed else "failed_speed_gate",
        "device": args.device,
        "cuda_device": (
            torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None
        ),
        "backbones": list(TITAN_B012_BACKBONES),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "compiled_model_backbones": sorted(compiled_backbones),
        "correctness": correctness,
        "training_step_timing": timing_rows,
        "speed_gate_passed": speed_gate_passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    if not speed_gate_passed:
        raise RuntimeError("B1/B2 training-step speed gate failed")


if __name__ == "__main__":
    main()
