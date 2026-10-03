#!/usr/bin/env python3
"""Measure synthetic CPU training cost in isolated processes, without datasets.

This is a local cost observation, not CUDA qualification or a GPU ETA. Each
backbone/length pair owns a fresh process so its process peak RSS is attributable
to that case (including Python/PyTorch imports and the baseline model).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
    "titantpp_pair_message_post_pool",
    "titantpp_pair_message_pre_pool",
)
LENGTHS = (64, 256)
BATCH_SIZE = 128
HIDDEN_DIM = 64
THREADS = 4
WARMUP_STEPS = 1
MEASURED_STEPS = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _tensor_digest(items) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        data = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(data.dtype).encode())
        digest.update(str(tuple(data.shape)).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def _worker(backbone: str, length: int) -> dict:
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    _require(backbone in ARMS and length in LENGTHS, "Unrecognized synthetic workload")
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    # Seed only the CPU generator: torch.manual_seed also touches CUDA generators.
    torch.random.default_generator.manual_seed(42)
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=HIDDEN_DIM, train_log_mean=1.5,
        max_seq_len=length,
        time_head_mode="heteroscedastic_lognormal_duration",
        time_scale=3.0, time_initial_location=0.1, time_initial_scale=0.7,
        time_observation_contract={
            "mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None,
        },
    )
    _require(all(p.device.type == "cpu" for p in model.parameters()), "CPU-only model required")
    if hasattr(model, "pair_message"):
        core = model.pair_message
        _require(core.rank == 8 and core.chunk_size == 32 and core.recompute is True,
                 "Pair-message rank/chunk/recompute differs from the synthetic cost contract")
        _require(metadata.get("pair_message_rank") == core.rank
                 and metadata.get("pair_message_chunk_size") == core.chunk_size
                 and metadata.get("pair_message_recompute") is core.recompute,
                 "Pair-message cost metadata differs from the active module")
    shared_initial_sha = _tensor_digest(
        (name, value) for name, value in model.state_dict().items()
        if not name.startswith("pair_message.")
    )
    model.train()
    with torch.no_grad():
        # Activate the common quantity path identically and the new branch for
        # realistic backward cost; zero-U initialization understates that work.
        model.quantity_head.weight.copy_(
            torch.linspace(-0.1, 0.1, HIDDEN_DIM).reshape(1, HIDDEN_DIM)
        )
        if hasattr(model, "pair_message"):
            weight = model.pair_message.output_projection.weight
            weight.copy_(torch.linspace(-0.05, 0.05, weight.numel()).reshape_as(weight))
    generator = torch.Generator(device="cpu").manual_seed(400 + length)
    dts = torch.randint(1, 31, (BATCH_SIZE, length), generator=generator).float()
    mask = torch.ones(BATCH_SIZE, length, dtype=torch.bool)
    quantities = torch.randint(0, 64, (BATCH_SIZE, length), generator=generator).float()
    batch_sha = _tensor_digest(zip(("dts", "mask", "quantities"), (dts, mask, quantities)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    timings, losses, norms = [], [], []
    added_gradients = {}
    for step in range(WARMUP_STEPS + MEASURED_STEPS):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        values = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        loss = values["joint_loss"].mean()
        _require(bool(torch.isfinite(loss)), "Nonfinite synthetic joint loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        if step == WARMUP_STEPS + MEASURED_STEPS - 1:
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    _require(bool(torch.isfinite(parameter.grad).all()), "Nonfinite gradient: " + name)
                if name.startswith("pair_message."):
                    added_gradients[name] = (
                        parameter.grad is not None
                        and bool(torch.count_nonzero(parameter.grad))
                    )
            if hasattr(model, "pair_message"):
                _require(bool(added_gradients) and all(added_gradients.values()),
                         "An activated pair-message parameter has no gradient")
        optimizer.step()
        elapsed = time.perf_counter() - started
        if step >= WARMUP_STEPS:
            timings.append(elapsed)
            losses.append(float(loss.detach()))
            norms.append(float(norm.detach()))
    _require(all(bool(torch.isfinite(p).all()) for p in model.parameters()),
             "A parameter became nonfinite after synthetic updates")
    peak_rss_native = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_factor = 1 if sys.platform == "darwin" else 1024
    return {
        "backbone": backbone,
        "sequence_length": length,
        "batch_size": BATCH_SIZE,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "added_parameter_count": sum(p.numel() for n, p in model.named_parameters()
                                     if n.startswith("pair_message.")),
        "shared_initial_state_sha256": shared_initial_sha,
        "synthetic_batch_sha256": batch_sha,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "step_seconds": timings,
        "median_step_seconds": statistics.median(timings),
        "mean_step_seconds": statistics.mean(timings),
        "joint_losses": losses,
        "preclip_gradient_norms": norms,
        "activated_added_parameter_gradients_nonzero": added_gradients,
        "all_final_parameters_finite": True,
        "process_peak_rss_bytes": int(peak_rss_native * rss_factor),
        "process_peak_rss_mib": peak_rss_native * rss_factor / (1024 ** 2),
        "peak_rss_scope": "whole isolated process, including imports; not tensor-only peak",
        "pair_message_activation_recompute": (
            "query chunks recomputed during backward" if backbone.endswith("pre_pool")
            else "not applicable: no pair-message tensor"
        ),
        "encoder_metadata": metadata,
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "platform": platform.platform(), "device": "cpu",
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
    }


def run(output: Path, *, max_wall_seconds: float) -> dict:
    _require(math.isfinite(max_wall_seconds) and 30 <= max_wall_seconds <= 600,
             "max-wall-seconds must be finite and between 30 and 600")
    _require(not output.exists(), "Refusing to overwrite an existing receipt")
    started = time.monotonic()
    receipt = {
        "schema": "pair_message_synthetic_cpu_cost_v1",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {"synthetic_only": True, "real_data_read": False,
                  "held_out_read": False, "gpu_used": False, "remote_access": False},
        "status": "running",
        "max_wall_seconds": max_wall_seconds,
        "workload": {"batch_size": BATCH_SIZE, "lengths": list(LENGTHS),
                     "hidden_dim": HIDDEN_DIM, "rank": 8, "query_chunk_size": 32,
                     "checkpoint_recompute": True, "seed": 42,
                     "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.01,
                     "gradient_clip_norm": 1.0, "quantity_objective": "log_mse",
                     "lambda_log_qty": 1.0, "time_objective": "recorded_positive_integer_time_nll",
                     "input_padding": "none; every synthetic row has full declared length",
                     "branch_activation": "identical nonzero quantity weights; nonzero added output weights"},
        "cases": [], "overhead_vs_B": [],
        "limitations": [
            "Three timed CPU updates per case are a local observation, not a GPU estimate.",
            "No data loading, validation pass, checkpoint serialization or epoch timing is measured.",
            "Full-length synthetic sequences are not an empirical distribution of real sequence lengths.",
            "Process peak RSS includes runtime allocations and cannot be called added tensor memory.",
        ],
    }
    source_paths = (
        Path(__file__).relative_to(ROOT),
        Path("models/TPPs/CountAwareTitanPairMessage.py"),
        Path("models/TPPs/CountAwareFactory.py"),
        Path("paper/scripts/count_aware_tpp_backbone/core.py"),
    )
    receipt["direct_source_sha256"] = {
        str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths
    }
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    try:
        for length in LENGTHS:
            for backbone in ARMS:
                remaining = max_wall_seconds - (time.monotonic() - started)
                _require(remaining > 0, "Synthetic CPU wall-clock bound exhausted")
                process_started = time.monotonic()
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--internal-worker", backbone,
                     "--length", str(length)],
                    cwd=ROOT, env=environment, text=True, capture_output=True, timeout=remaining,
                    check=False,
                )
                _require(result.returncode == 0,
                         f"Synthetic worker failed for {backbone}/{length}: {result.stderr[-4000:]}")
                case = json.loads(result.stdout)
                case["isolated_process_wall_seconds"] = time.monotonic() - process_started
                receipt["cases"].append(case)
            group = receipt["cases"][-len(ARMS):]
            base = group[0]
            _require(len({c["synthetic_batch_sha256"] for c in group}) == 1,
                     "Synthetic batches differ across arms")
            _require(len({c["shared_initial_state_sha256"] for c in group}) == 1,
                     "Shared initialization differs across arms")
            _require(group[1]["parameter_count"] == group[2]["parameter_count"],
                     "Candidate and post-pool control parameter counts differ")
            for case in group[1:]:
                receipt["overhead_vs_B"].append({
                    "backbone": case["backbone"], "sequence_length": length,
                    "median_step_time_ratio": case["median_step_seconds"] / base["median_step_seconds"],
                    "whole_process_peak_rss_ratio": case["process_peak_rss_bytes"] / base["process_peak_rss_bytes"],
                    "parameter_ratio": case["parameter_count"] / base["parameter_count"],
                })
        receipt["status"] = "complete"
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        receipt["status"] = "failed"
        receipt["error"] = str(error)
    receipt["total_wall_seconds"] = time.monotonic() - started
    receipt["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-wall-seconds", type=float, default=300.0)
    parser.add_argument("--internal-worker", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--length", type=int, choices=LENGTHS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_worker:
        if args.length is None or args.output is not None:
            parser.error("Internal worker requires --length and no --output")
        print(json.dumps(_worker(args.internal_worker, args.length), allow_nan=False))
        return 0
    if args.output is None or args.length is not None:
        parser.error("CPU cost measurement requires --output")
    receipt = run(args.output, max_wall_seconds=args.max_wall_seconds)
    print(json.dumps({"status": receipt["status"], "output": str(args.output.resolve()),
                      "total_wall_seconds": receipt["total_wall_seconds"]}))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
