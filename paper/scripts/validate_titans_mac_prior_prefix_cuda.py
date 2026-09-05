#!/usr/bin/env python3
"""Check the frozen 5090 runtime and synthetic prior-prefix training-step cost.

CUDA correctness tests run separately. This validator never reads a dataset or
checkpoint, and its synthetic optimizer steps cannot establish model quality.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import traceback
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import polars as pl  # noqa: E402
import torch  # noqa: E402


T0 = "titantpp"
B1 = "titantpp_titans_mac"
CANDIDATE = "titantpp_titans_mac_prior_prefix"
MODELS = (T0, B1, CANDIDATE)
DEFAULT_CONTRACT = PROJECT_ROOT / "paper/contracts/titans_mac_prior_prefix_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameter_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def runtime_version_checks(
    expected: dict[str, Any], observed: dict[str, Any],
) -> dict[str, bool]:
    """Pure exact-version checks, also used by CPU fixture tests."""
    return {
        "python_version": observed.get("python") == expected["expected_python"],
        "torch_version": observed.get("torch") == expected["expected_torch"],
        "cuda_version": observed.get("cuda") == expected["expected_cuda"],
        "polars_version": observed.get("polars") == expected["expected_polars"],
        "cuda_available": observed.get("cuda_available") is True,
        "device_is_5090": "5090" in str(observed.get("device_name", "")),
        "minimum_free_memory": (
            isinstance(observed.get("free_bytes"), int)
            and observed["free_bytes"] >= expected["minimum_free_mib"] * 1024**2
        ),
    }


def evaluate_cost_gate(
    rows: list[dict[str, Any]], cost: dict[str, Any],
) -> dict[str, Any]:
    """Reject incomplete/nonfinite measurements and apply every per-H gate."""
    errors: list[str] = []
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    expected_keys = {(model, h) for model in MODELS for h in cost["history_lengths"]}
    for row in rows:
        key = (row.get("backbone"), row.get("history_length"))
        if key not in expected_keys or key in indexed:
            errors.append(f"Unexpected or duplicate benchmark row: {key}")
            continue
        indexed[key] = row
        samples = row.get("step_seconds", [])
        if not isinstance(samples, list) or len(samples) != cost["measured_steps"]:
            errors.append(f"Incomplete measured steps: {key}")
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in samples):
            errors.append(f"Nonpositive/nonfinite timing: {key}")
            continue
        if row.get("median_step_seconds") != statistics.median(samples):
            errors.append(f"Median does not reconstruct from timings: {key}")
        if row.get("warmup_steps_completed") != cost["warmup_steps"]:
            errors.append(f"Incomplete warmup: {key}")
        if row.get("measured_steps_completed") != cost["measured_steps"]:
            errors.append(f"Incomplete measured step count: {key}")
        if row.get("batch_size") != cost["batch_size"] or row.get("sequence_length") != key[1] + 1:
            errors.append(f"Wrong benchmark shape: {key}")
        peak = row.get("peak_allocated_bytes")
        if isinstance(peak, bool) or not isinstance(peak, int) or peak <= 0:
            errors.append(f"Invalid peak allocation: {key}")
        if row.get("finite_steps") is not True or row.get("parameters_finite") is not True:
            errors.append(f"Missing finite checks: {key}")
        if row.get("allocated_before_model_bytes") != 0 or row.get("allocated_after_cleanup_bytes") != 0:
            errors.append(f"GPU allocation was not isolated: {key}")
    if set(indexed) != expected_keys:
        errors.append("Expected complete 3-model x 3-history matrix")
    comparisons: list[dict[str, Any]] = []
    if not errors:
        for history_length in cost["history_lengths"]:
            candidate = indexed[(CANDIDATE, history_length)]
            baseline = indexed[(B1, history_length)]
            t0 = indexed[(T0, history_length)]
            step_ratio = candidate["median_step_seconds"] / baseline["median_step_seconds"]
            memory_ratio = candidate["peak_allocated_bytes"] / baseline["peak_allocated_bytes"]
            same_initial = (
                isinstance(candidate.get("initial_parameter_sha256"), str)
                and len(candidate["initial_parameter_sha256"]) == 64
                and candidate["initial_parameter_sha256"] == baseline.get("initial_parameter_sha256")
                and isinstance(candidate.get("parameter_count"), int)
                and candidate["parameter_count"] > 0
                and candidate.get("parameter_count") == baseline.get("parameter_count")
            )
            same_inputs = (
                isinstance(candidate.get("input_sha256"), str)
                and len(candidate["input_sha256"]) == 64
                and candidate["input_sha256"] == baseline.get("input_sha256") == t0.get("input_sha256")
            )
            checks = {
                "same_B1_initial_parameters": same_initial,
                "same_synthetic_inputs": same_inputs,
                "step_ratio": step_ratio <= cost["candidate_to_B1_step_ratio_max"],
                "peak_allocated_ratio": memory_ratio <= cost["candidate_to_B1_peak_allocated_ratio_max"],
                "absolute_peak_allocated": candidate["peak_allocated_bytes"] <= cost["candidate_peak_allocated_bytes_max"],
            }
            comparisons.append({
                "history_length": history_length,
                "candidate_to_B1_step_ratio": step_ratio,
                "candidate_to_B1_peak_allocated_ratio": memory_ratio,
                "candidate_to_T0_step_ratio": candidate["median_step_seconds"] / t0["median_step_seconds"],
                "B1_to_T0_step_ratio": baseline["median_step_seconds"] / t0["median_step_seconds"],
                "candidate_to_T0_peak_allocated_ratio": candidate["peak_allocated_bytes"] / t0["peak_allocated_bytes"],
                "candidate_peak_allocated_bytes": candidate["peak_allocated_bytes"],
                "checks": checks, "passed": all(checks.values()),
                "T0_ratios_are_report_only": True,
            })
    return {
        "passed": not errors and len(comparisons) == len(cost["history_lengths"]) and all(row["passed"] for row in comparisons),
        "errors": errors, "comparisons": comparisons,
    }


def make_synthetic_batch(history_length: int, batch_size: int) -> tuple[torch.Tensor, ...]:
    """Identical positive CPU-generated inputs; final token is the target."""
    generator = torch.Generator(device="cpu").manual_seed(91000 + history_length)
    dts = 0.25 + 3.75 * torch.rand(batch_size, history_length + 1, generator=generator)
    quantities = torch.randint(1, 21, (batch_size, history_length + 1), generator=generator).float()
    mask = torch.ones_like(dts, dtype=torch.bool)
    return dts, mask, quantities


def _build_model(backbone: str, contract: dict[str, Any]) -> torch.nn.Module:
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from models.Titan.common.titans_mac_optimized import apply_titantpp_mac_semantic_optimization

    training = contract["training"]
    torch.manual_seed(42)
    model, _ = build_count_aware_model(
        backbone, hidden_dim=contract["architecture"]["hidden_dim"],
        train_log_mean=2.0, train_log_std=1.0,
        max_seq_len=max(contract["cost_gate"]["history_lengths"]) + 1,
        quantity_variant=training["quantity_variant"],
        lambda_tail=training["lambda_tail"],
        time_head_mode=training["time_head"],
        time_scale=training["time_scale"],
        time_w_max=training["time_w_max"],
        time_intercept_limit=training["time_intercept_limit"],
        time_wd_safety_limit=training["time_wd_safety_limit"],
        titans_memory_gradient_clip=(training["inner_gradient_clip"] if backbone != T0 else None),
    )
    if backbone != T0:
        apply_titantpp_mac_semantic_optimization(model)
        expected_policy = "prior_prefix" if backbone == CANDIDATE else "segment_start"
        if model.titans_mac_encoder.output_read_policy != expected_policy:
            raise AssertionError("Optimized encoder lost the required output-read policy")
        if not model.titans_mac_encoder.neural_memory.compile_cuda_scan:
            raise AssertionError("CUDA scan compilation is required")
    return model


def _step(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, ...], contract: dict[str, Any],
) -> dict[str, float]:
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    optimizer.zero_grad(set_to_none=True)
    outputs = target_outputs(model, *batch, lambda_log_qty=contract["training"]["lambda_log_qty"])
    for name in ("joint_loss", "time_loss", "quantity_train_loss", "pred_qty"):
        if not bool(torch.isfinite(outputs[name]).all()):
            raise FloatingPointError(f"Synthetic benchmark {name} is nonfinite")
    loss = outputs["joint_loss"].mean()
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), contract["training"]["outer_gradient_clip"], error_if_nonfinite=True,
    )
    optimizer.step()
    return {"joint_loss": float(loss.detach().cpu()), "pre_clip_gradient_norm": float(gradient_norm.detach().cpu())}


def _release_cuda() -> None:
    # Dynamo references can otherwise retain compiled model/optimizer tensors.
    torch._dynamo.reset()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def observe_runtime() -> dict[str, Any]:
    observed = {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "polars": pl.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if observed["cuda_available"]:
        observed["device_name"] = torch.cuda.get_device_name(0)
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        observed.update(free_bytes=int(free_bytes), total_bytes=int(total_bytes))
        observed["capability"] = list(torch.cuda.get_device_capability(0))
    return observed


def _profile_one(row: dict[str, Any], contract: dict[str, Any]) -> None:
    _release_cuda()
    row["allocated_before_model_bytes"] = int(torch.cuda.memory_allocated())
    if row["allocated_before_model_bytes"] != 0:
        raise RuntimeError("Previous GPU tensor allocation survived model cleanup")
    model = optimizer = batch = None
    try:
        model = _build_model(row["backbone"], contract)
        row["initial_parameter_sha256"] = parameter_digest(model)
        row["parameter_count"] = sum(p.numel() for p in model.parameters())
        model = model.cuda().train()
        batch = tuple(t.cuda() for t in make_synthetic_batch(row["history_length"], row["batch_size"]))
        training = contract["training"]
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=training["lr"], weight_decay=training["weight_decay"],
            betas=tuple(training["betas"]), eps=training["eps"],
        )
        row["input_sha256"] = hashlib.sha256(b"".join(t.detach().cpu().numpy().tobytes() for t in batch)).hexdigest()
        row["execution_backend"] = "static_T0" if row["backbone"] == T0 else "optimized_compiled_cuda"
        row["warmup_steps_completed"] = 0
        row["measured_steps_completed"] = 0
        row["step_seconds"] = []
        row["step_telemetry"] = []

        def synchronized_step() -> tuple[float, dict[str, float]]:
            torch.cuda.synchronize()
            started = time.perf_counter()
            telemetry = _step(model, optimizer, batch, contract)
            torch.cuda.synchronize()
            return time.perf_counter() - started, telemetry

        metric_offsets = {
            name: len(values)
            for name, values in getattr(torch._dynamo.utils, "compilation_time_metrics", {}).items()
            if isinstance(values, (list, tuple))
        }
        torch.cuda.reset_peak_memory_stats()
        row["cold_compile_and_first_step_seconds"], cold_telemetry = synchronized_step()
        row["cold_step_telemetry"] = cold_telemetry
        row["cold_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        row["cold_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        row["compile_time_note"] = "Cold wall includes the first training step; it is excluded from warmup and measured timings."
        compilation_metrics = getattr(torch._dynamo.utils, "compilation_time_metrics", {})
        row["compiler_timing_metrics_seconds"] = {
            str(name): [float(value) for value in values[metric_offsets.get(name, 0):] if isinstance(value, (int, float)) and math.isfinite(value)]
            for name, values in compilation_metrics.items()
            if isinstance(values, (list, tuple))
        }
        print(json.dumps({"phase": "cold_complete", "backbone": row["backbone"], "history_length": row["history_length"], "seconds": row["cold_compile_and_first_step_seconds"]}), flush=True)
        torch.cuda.reset_peak_memory_stats()
        row["warmup_step_seconds"] = []
        for _ in range(contract["cost_gate"]["warmup_steps"]):
            elapsed, telemetry = synchronized_step()
            row["warmup_step_seconds"].append(elapsed)
            row["warmup_steps_completed"] += 1
        row["warmup_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        for _ in range(contract["cost_gate"]["measured_steps"]):
            elapsed, telemetry = synchronized_step()
            row["step_seconds"].append(elapsed)
            row["step_telemetry"].append(telemetry)
            row["measured_steps_completed"] += 1
        row["measured_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        row["measured_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        row["peak_allocated_bytes"] = max(row[f"{phase}_peak_allocated_bytes"] for phase in ("cold", "warmup", "measured"))
        row["median_step_seconds"] = statistics.median(row["step_seconds"])
        row["finite_steps"] = True
        row["parameters_finite"] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
        if not row["parameters_finite"]:
            raise FloatingPointError("Synthetic optimizer produced nonfinite parameters")
        if row["backbone"] != T0:
            row["quantity_head_opened"] = bool(torch.count_nonzero(model.quantity_head.weight))
            if not row["quantity_head_opened"]:
                raise AssertionError("Quantity head did not open during synthetic warmup")
    finally:
        model = optimizer = batch = None
        _release_cuda()
        row["allocated_after_cleanup_bytes"] = int(torch.cuda.memory_allocated())


def validate(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "FAIL", "source_revision": args.source_revision,
        "contract_sha256": None, "device": "cuda", "cuda_available": False,
        "checks": {}, "cost_gate_pass": False, "benchmark_rows": [],
        "cost_comparisons": [], "held_out_test_evaluated": False,
        "real_data_used": False, "checkpoint_loaded": False,
        "scope": "runtime and synthetic full training-step cost; no performance adoption",
    }
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
            raise ValueError("--source-revision must be a full lowercase git SHA")
        report["checks"]["source_revision_format"] = True
        contract = json.loads(args.contract.read_text(encoding="utf-8"))
        report["contract_sha256"] = sha256(args.contract)
        if contract.get("contract_id") != "titans_mac_prior_prefix_v1":
            raise ValueError("Unexpected prior-prefix contract identity")
        if contract["cost_gate"]["history_lengths"] != [16, 64, 255] or contract["cost_gate"]["batch_size"] != 128:
            raise ValueError("Unexpected cost benchmark shapes")
        report["checks"]["contract_identity"] = True
        observed = observe_runtime()
        report["cuda_available"] = observed["cuda_available"]
        report["runtime"] = observed
        checks = runtime_version_checks(contract["runtime"], observed)
        report["checks"].update(checks)
        if not all(checks.values()):
            raise RuntimeError(f"Frozen CUDA runtime check failed: {[k for k, v in checks.items() if not v]}")
        import torch._dynamo.config as dynamo_config
        dynamo_config.recompile_limit = contract["runtime"]["dynamo_recompile_limit"]
        dynamo_config.accumulated_recompile_limit = contract["runtime"]["dynamo_accumulated_recompile_limit"]
        dynamo_config.suppress_errors = False
        report["dynamo_policy"] = {
            "recompile_limit": dynamo_config.recompile_limit,
            "accumulated_recompile_limit": dynamo_config.accumulated_recompile_limit,
            "suppress_errors": dynamo_config.suppress_errors,
        }
        source_files = [
            "models/TPPs/CountAwareFactory.py", "models/TPPs/CountAwareTPP.py",
            "models/Titan/common/titans_mac.py", "models/Titan/common/titans_mac_prior_prefix.py",
            "models/Titan/common/titans_mac_optimized.py", "models/Titan/common/titans_memory_stability.py",
            "paper/scripts/count_aware_tpp_backbone/core.py",
            "paper/scripts/validate_titans_mac_prior_prefix_cuda.py",
        ]
        report["source_file_sha256"] = {name: sha256(PROJECT_ROOT / name) for name in source_files}
        report["source_binding_note"] = "Launcher manifest binds these isolated source hashes to source_revision; no .git is required here."
        for history_length in contract["cost_gate"]["history_lengths"]:
            for backbone in MODELS:
                row = {
                    "backbone": backbone, "history_length": history_length,
                    "sequence_length": history_length + 1, "batch_size": 128,
                }
                report["benchmark_rows"].append(row)
                print(json.dumps({"phase": "profile_start", **row}), flush=True)
                _profile_one(row, contract)
        gate = evaluate_cost_gate(report["benchmark_rows"], contract["cost_gate"])
        report["cost_gate_pass"] = gate["passed"]
        report["cost_comparisons"] = gate["comparisons"]
        report["cost_gate_errors"] = gate["errors"]
        report["checks"]["cost_gate"] = gate["passed"]
        report["checks"]["complete_measurement_matrix"] = not gate["errors"]
        report["status"] = "PASS" if all(report["checks"].values()) else "FAIL"
    except Exception as error:
        report["status"] = "FAIL"
        report["cost_gate_pass"] = False
        report["checks"]["execution_completed"] = False
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["traceback"] = traceback.format_exc()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; automatic overwrite/retry is prohibited")
    report = validate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "cost_gate_pass": report["cost_gate_pass"]}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
