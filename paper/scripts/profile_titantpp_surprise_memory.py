#!/usr/bin/env python3
"""Profile Count-aware TitanTPP memory backbones on fixed synthetic shapes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


BACKBONES = (
    "titantpp",
    "titantpp_no_memory",
    "titantpp_gated_soft_memory",
    "titantpp_surprise_memory",
)
REFERENCE_BACKBONE = "titantpp"
PROFILE_BACKBONE = "titantpp_surprise_memory"


@dataclass(frozen=True)
class ShapeSpec:
    name: str
    batch_size: int
    seq_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-label", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument(
        "--shapes",
        default="intermittent:128:16,instacart:128:64,long:32:256",
        help="Comma-separated name:batch_size:sequence_length entries.",
    )
    parser.add_argument("--baseline-timings", type=Path)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_shapes(raw: str) -> tuple[ShapeSpec, ...]:
    shapes: list[ShapeSpec] = []
    for item in raw.split(","):
        name, batch_size, seq_len = item.split(":")
        shape = ShapeSpec(name, int(batch_size), int(seq_len))
        if shape.batch_size < 1 or shape.seq_len < 2:
            raise ValueError(f"Invalid profile shape: {item}")
        shapes.append(shape)
    if not shapes:
        raise ValueError("At least one profile shape is required")
    return tuple(shapes)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batch(
    shape: ShapeSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(
        20260818 + shape.batch_size * 1000 + shape.seq_len
    )
    dts = 0.05 + torch.rand(
        shape.batch_size,
        shape.seq_len,
        generator=generator,
    )
    quantities = torch.randint(
        1,
        96,
        (shape.batch_size, shape.seq_len),
        generator=generator,
    ).float()
    mask = torch.ones(shape.batch_size, shape.seq_len, dtype=torch.bool)
    return dts.to(device), mask.to(device), quantities.to(device)


def open_memory_gate(model: torch.nn.Module) -> None:
    for attribute in ("soft_memory", "surprise_memory"):
        memory = getattr(model, attribute, None)
        if memory is not None:
            memory.residual_scale.data.fill_(0.5)


def mean_time_ms(
    function: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def profile_one_model(
    backbone: str,
    shape: ShapeSpec,
    *,
    hidden_dim: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(20260818)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260818)
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=hidden_dim,
        train_log_mean=1.5,
        max_seq_len=shape.seq_len,
    )
    model = model.to(device)
    open_memory_gate(model)
    dts, mask, quantities = make_batch(shape, device)

    def forward_step() -> None:
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.encode(dts, quantities, mask)

    def train_step() -> None:
        model.zero_grad(set_to_none=True)
        output = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
        output["joint_loss"].mean().backward()

    model.eval()
    forward_ms = mean_time_ms(
        forward_step,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_step_ms = mean_time_ms(
        train_step,
        warmup=warmup,
        iterations=iterations,
        device=device,
    )
    peak_memory_mb = (
        torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        if device.type == "cuda"
        else 0.0
    )
    return {
        **asdict(shape),
        "backbone": backbone,
        "candidate_name": metadata["candidate_name"],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "forward_ms": forward_ms,
        "train_step_ms": train_step_ms,
        "peak_memory_mb": peak_memory_mb,
    }


def add_reference_ratios(rows: list[dict[str, Any]]) -> None:
    for shape_name in {row["name"] for row in rows}:
        group = [row for row in rows if row["name"] == shape_name]
        reference = next(
            row for row in group if row["backbone"] == REFERENCE_BACKBONE
        )
        for row in group:
            row["forward_ratio_to_hard_lmm"] = (
                row["forward_ms"] / reference["forward_ms"]
            )
            row["train_ratio_to_hard_lmm"] = (
                row["train_step_ms"] / reference["train_step_ms"]
            )


def profile_operators(
    shape: ShapeSpec,
    *,
    hidden_dim: int,
    iterations: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    torch.manual_seed(20260818)
    model, _ = build_count_aware_model(
        PROFILE_BACKBONE,
        hidden_dim=hidden_dim,
        train_log_mean=1.5,
        max_seq_len=shape.seq_len,
    )
    model = model.to(device).train()
    open_memory_gate(model)
    dts, mask, quantities = make_batch(shape, device)

    def train_step() -> None:
        model.zero_grad(set_to_none=True)
        output = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
        output["joint_loss"].mean().backward()

    for _ in range(2):
        train_step()
    synchronize(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
    ) as profiler:
        for _ in range(iterations):
            train_step()
    synchronize(device)

    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        device_total = getattr(
            event,
            "device_time_total",
            getattr(event, "cuda_time_total", 0.0),
        )
        self_device_total = getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
        rows.append({
            "operator": event.key,
            "calls": event.count,
            "cpu_time_total_us": event.cpu_time_total,
            "self_cpu_time_total_us": event.self_cpu_time_total,
            "device_time_total_us": device_total,
            "self_device_time_total_us": self_device_total,
        })
    rows.sort(
        key=lambda row: (
            row["device_time_total_us"],
            row["cpu_time_total_us"],
        ),
        reverse=True,
    )
    return rows[:50]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_with_baseline(
    rows: list[dict[str, Any]],
    baseline_path: Path,
) -> dict[str, Any]:
    baseline_rows = json.loads(baseline_path.read_text(encoding="utf-8"))["rows"]
    baseline_map = {
        (row["name"], row["backbone"]): row for row in baseline_rows
    }
    comparisons = []
    for row in rows:
        baseline = baseline_map[(row["name"], row["backbone"])]
        comparisons.append({
            "name": row["name"],
            "backbone": row["backbone"],
            "forward_speedup": baseline["forward_ms"] / row["forward_ms"],
            "train_step_speedup": (
                baseline["train_step_ms"] / row["train_step_ms"]
            ),
            "peak_memory_ratio": (
                row["peak_memory_mb"] / baseline["peak_memory_mb"]
                if baseline["peak_memory_mb"]
                else 0.0
            ),
        })
    surprise_rows = [
        row for row in rows if row["backbone"] == PROFILE_BACKBONE
    ]
    return {
        "comparisons": comparisons,
        "acceptance": {
            "max_train_ratio_to_hard_lmm": 3.0,
            "observed_max_train_ratio_to_hard_lmm": max(
                row["train_ratio_to_hard_lmm"] for row in surprise_rows
            ),
            "passed": all(
                row["train_ratio_to_hard_lmm"] <= 3.0
                for row in surprise_rows
            ),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but CUDA is unavailable")
    shapes = parse_shapes(args.shapes)
    manifest = {
        "status": "running",
        "started_at": utc_now(),
        "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
        "implementation_label": args.implementation_label,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "hidden_dim": args.hidden_dim,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "shapes": [asdict(shape) for shape in shapes],
        "held_out_test_evaluated": False,
        "command": sys.argv,
    }
    manifest_path = output_dir / "profile_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        rows = [
            profile_one_model(
                backbone,
                shape,
                hidden_dim=args.hidden_dim,
                warmup=args.warmup,
                iterations=args.iterations,
                device=device,
            )
            for shape in shapes
            for backbone in BACKBONES
        ]
        add_reference_ratios(rows)
        write_csv(output_dir / "timings.csv", rows)
        (output_dir / "timings.json").write_text(
            json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        operator_rows = profile_operators(
            next(shape for shape in shapes if shape.name == "instacart"),
            hidden_dim=args.hidden_dim,
            iterations=args.profile_iterations,
            device=device,
        )
        write_csv(output_dir / "operator_profile.csv", operator_rows)
        if args.baseline_timings is not None:
            comparison = compare_with_baseline(rows, args.baseline_timings)
            (output_dir / "comparison.json").write_text(
                json.dumps(comparison, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["completed_at"] = utc_now()
        manifest["error"] = repr(error)
        raise
    finally:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
