#!/usr/bin/env python3
"""Bounded synthetic CPU cost check; never loads data or launches GPU work."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTitanSlotMemory import SLOT_MEMORY_BACKBONE
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.training import build_optimizer


def _step(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    dts, mask, quantities = batch
    outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    loss = outputs["joint_loss"].mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite synthetic objective")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite model state after synthetic step")
    return float(loss.detach()), float(norm)


def verify():
    rows = []
    for length in (8, 64, 256):
        generator = torch.Generator().manual_seed(400 + length)
        batch = (
            0.1 + 2.0 * torch.rand(8, length, generator=generator),
            torch.ones(8, length, dtype=torch.bool),
            torch.randint(0, 32, (8, length), generator=generator).float(),
        )
        models = {}
        for backbone in ("titantpp", SLOT_MEMORY_BACKBONE):
            torch.manual_seed(42)
            model, _ = build_count_aware_model(
                backbone, hidden_dim=64, train_log_mean=1.5, max_seq_len=256,
                time_intercept_limit=300.0,
            )
            if backbone == SLOT_MEMORY_BACKBONE:
                with torch.no_grad():
                    model.slot_memory_alpha_raw.fill_(0.25)
            optimizer = build_optimizer(model, lr=0.001, time_head_lr_multiplier=1.0)
            models[backbone] = (model, optimizer)
        durations = {name: [] for name in models}
        # Alternate measurement order to reduce systematic warmup/order bias.
        for iteration in range(7):
            order = list(models) if iteration % 2 == 0 else list(reversed(models))
            for name in order:
                model, optimizer = models[name]
                torch.manual_seed(1000 + iteration)
                started = time.perf_counter()
                _step(model, optimizer, batch)
                seconds = time.perf_counter() - started
                if iteration >= 2:
                    durations[name].append(seconds)
        counts = {name: sum(p.numel() for p in pair[0].parameters()) for name, pair in models.items()}
        medians = {name: statistics.median(samples) for name, samples in durations.items()}
        rows.append({
            "batch": 8, "length": length, "hidden_dim": 64,
            "parameters": counts, "step_seconds": durations, "median_step_seconds": medians,
            "candidate_over_b_cpu_step_ratio": medians[SLOT_MEMORY_BACKBONE] / medians["titantpp"],
            "one_prefix_value_accumulator_bytes_float32": 8 * length * 8 * 64 * 4,
            "synthetic_updates_per_model": 7, "all_updates_finite": True,
        })
    return {
        "status": "synthetic_cpu_cost_check_passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "synthetic_cpu_only; no_dataset_quality_evidence",
        "runtime": {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(), "threads": torch.get_num_threads()},
        "gpu_executed": False, "real_data_loaded": False,
        "rows": rows,
        "limitations": [
            "CPU batch8 timing is not a CUDA or full training time estimate",
            "prefix accumulator bytes are a single tensor calculation, not peak memory",
            "candidate gate was activated for cost/finite-update qualification",
            "synthetic updates do not measure dataset or benchmark performance",
        ],
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in (
                "models/TPPs/CountAwareTitanSlotMemory.py",
                "models/TPPs/CountAwareFactory.py",
                "paper/contracts/hard_lmm_observed_slot_memory_v1.json",
                "paper/scripts/count_aware_tpp_backbone/constants.py",
                "paper/scripts/count_aware_tpp_backbone/core.py",
                "paper/scripts/count_aware_tpp_backbone/training.py",
                "paper/scripts/run_count_aware_tpp_backbone_control.py",
                "paper/scripts/verify_observed_slot_memory_cpu.py",
            )
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    before_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            receipt = verify()
    finally:
        torch.set_num_threads(before_threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
