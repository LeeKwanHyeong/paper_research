#!/usr/bin/env python3
"""Measure bounded synthetic CPU updates in fresh processes; never access a dataset."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = (
    "titantpp",
    "titantpp_nonlinear_episode_post_pool",
    "titantpp_nonlinear_episode_pre_pool",
)
SOURCES = (
    "models/TPPs/CountAwareTitanNonlinearEpisodeMemory.py",
    "models/TPPs/CountAwareTitanSuccessorMemory.py",
    "models/TPPs/CountAwareTPP.py",
    "models/TPPs/CountAwareFactory.py",
    "paper/contracts/hard_lmm_nonlinear_episode_value_design_v1.json",
    "paper/scripts/count_aware_tpp_backbone/constants.py",
    "paper/scripts/count_aware_tpp_backbone/core.py",
    "paper/scripts/count_aware_tpp_backbone/training.py",
    "paper/scripts/run_count_aware_tpp_backbone_control.py",
    "paper/scripts/verify_observed_slot_memory_cpu.py",
    "paper/scripts/verify_nonlinear_episode_cpu.py",
)


def source_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def measure(backbone: str, length: int) -> dict:
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from paper.scripts.verify_observed_slot_memory_cpu import _step

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    model, _ = build_count_aware_model(
        backbone, hidden_dim=64, train_log_mean=1.5, max_seq_len=length,
        quantity_variant="count_only_log_regression", lambda_tail=0.,
        time_head_mode="legacy_clamped_rmtpp", time_intercept_limit=300.,
    )
    with torch.no_grad():
        # Nonzero paths exercise the full backward graph, not only zero-init U.
        model.quantity_head.weight.copy_(torch.linspace(-.1, .1, 64).reshape(1, 64))
        if backbone != "titantpp":
            model.nonlinear_episode_memory.output_projection.weight.copy_(
                torch.linspace(-.05, .05, 64 * 16).reshape(64, 16))
    generator = torch.Generator().manual_seed(400 + length)
    batch = (
        .1 + 2 * torch.rand(128, length, generator=generator),
        torch.ones(128, length, dtype=torch.bool),
        torch.randint(0, 32, (128, length), generator=generator).float(),
    )
    optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
    samples = []
    for step in range(7):
        torch.manual_seed(1000 + step)
        started = time.perf_counter()
        _step(model, optimizer, batch)
        if step >= 2:
            samples.append(time.perf_counter() - started)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "backbone": backbone, "batch": 128, "length": length,
        "max_seq_len": length, "hidden_dim": 64, "device": "cpu",
        "parameters": sum(p.numel() for p in model.parameters()),
        "step_seconds": samples, "median_step_seconds": statistics.median(samples),
        "process_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "synthetic_optimizer_updates": 7, "all_updates_finite": True,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "threads": torch.get_num_threads()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, choices=(64, 256), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        if args.output is not None or args.arm is None or args.length is None:
            parser.error("child requires one arm/length and no output path")
        print(json.dumps(measure(args.arm, args.length), allow_nan=False))
        return
    if args.output is None or args.output.exists() or args.arm is not None or args.length is not None:
        parser.error("a fresh output path is required")
    before = source_hashes()
    rows = []
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONHASHSEED": "42"}
    for length in (64, 256):
        # Separate processes keep each peak-RSS measurement independent.
        order = ARMS if length == 64 else tuple(reversed(ARMS))
        measurements = {}
        for arm in order:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--child", "--arm", arm, "--length", str(length)],
                cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180, check=True,
            )
            measurements[arm] = json.loads(result.stdout)
        baseline = measurements[ARMS[0]]
        for row in measurements.values():
            row["step_ratio_to_B"] = row["median_step_seconds"] / baseline["median_step_seconds"]
            row["parameter_ratio_to_B"] = row["parameters"] / baseline["parameters"]
            row["process_peak_rss_ratio_to_B"] = row["process_peak_rss_bytes"] / baseline["process_peak_rss_bytes"]
        rows.append({"length": length, "batch": 128, "measurement_order": order, "measurements": measurements})
    if source_hashes() != before:
        raise RuntimeError("Source changed during the CPU measurement")
    receipt = {
        "status": "synthetic_cpu_measurement_complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows, "source_sha256": before, "synthetic_optimizer_updates": 42,
        "gpu_executed": False, "real_data_loaded": False, "held_out_evaluated": False,
        "limitations": [
            "CPU timing is not CUDA qualification or a training-hours estimate.",
            "Peak RSS is fresh-process resident memory including runtime; not CUDA allocated memory.",
            "Step includes objective/backward/clip/AdamW and finite-state checks, excludes loader/validation/checkpoint I/O.",
            "One process per arm/shape with two warmups and five measured steps; hardware/order variability remains.",
            "Synthetic activated parameters exercise computation; they are not trained checkpoints or performance evidence.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
