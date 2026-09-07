#!/usr/bin/env python3
"""CUDA-only cost and learning-path audit for the frozen causal-QKV contract.

Every model/repeat/length runs in a fresh process. Synthetic inputs contain an
observed prefix and one target and use the production ``target_outputs`` path.
The profile is execution evidence, not evidence of predictive improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINE = "titantpp"
CANDIDATE = "titantpp_hard_memory_causal_qkv"
KERNEL_KEYS = tuple(
    f"encoder.layers.0.attn.causal_{kind}_kernel" for kind in ("q", "k", "v")
)
LENGTHS = (8, 64, 256)
BATCH_SIZE = 128
HIDDEN_DIM = 64
WARMUP_STEPS = 5
MEASURED_STEPS = 15
REPEATS = 3
STEP_LIMIT = 1.5
PEAK_LIMIT = 1.25


def save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def positive_ratio(numerator: float, denominator: float) -> float:
    values = (float(numerator), float(denominator))
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Cost ratios require finite, strictly positive measurements")
    return values[0] / values[1]


def measurement_order(repeat: int) -> tuple[str, str]:
    if type(repeat) is not int or not 0 <= repeat < REPEATS:
        raise ValueError("Unexpected repeat index")
    return (BASELINE, CANDIDATE) if repeat % 2 == 0 else (CANDIDATE, BASELINE)


def summarize_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reject incomplete/duplicate proofs before computing ratios of medians."""
    expected = {(length, repeat, name) for length in LENGTHS
                for repeat in range(REPEATS) for name in (BASELINE, CANDIDATE)}
    identities = [(row.get("sequence_length"), row.get("repeat"), row.get("backbone"))
                  for row in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Cost proof must contain exactly the complete paired grid")
    for row in rows:
        if row.get("status") != "passed":
            raise ValueError("Failed worker cannot qualify a cost proof")
        if (row.get("batch_size") != BATCH_SIZE or row.get("hidden_dim") != HIDDEN_DIM
                or row.get("warmup_steps") != WARMUP_STEPS
                or row.get("measured_steps") != MEASURED_STEPS):
            raise ValueError("Worker measurement policy differs from the frozen policy")
        times = row.get("step_seconds", [])
        if len(times) != MEASURED_STEPS:
            raise ValueError("Every worker must retain every measured step")
        for value in times:
            positive_ratio(value, 1.0)
        positive_ratio(row["median_step_seconds"], 1.0)
        positive_ratio(row["peak_allocated_bytes"], 1.0)
        if not math.isclose(statistics.median(times), row["median_step_seconds"],
                            rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("Worker median does not reconstruct from measured steps")
        if row.get("finite_model_optimizer") is not True:
            raise ValueError("Worker has no finite model/optimizer proof")
        if row["backbone"] == CANDIDATE and row.get("learning_path", {}).get("status") != "passed":
            raise ValueError("Candidate has no successful learning-path audit")
    summaries = []
    for length in LENGTHS:
        medians: dict[str, dict[str, float]] = {}
        for name in (BASELINE, CANDIDATE):
            matching = [row for row in rows if row["sequence_length"] == length
                        and row["backbone"] == name]
            medians[name] = {
                "step_seconds": statistics.median(row["median_step_seconds"] for row in matching),
                "peak_allocated_bytes": statistics.median(row["peak_allocated_bytes"] for row in matching),
            }
        step_ratio = positive_ratio(medians[CANDIDATE]["step_seconds"], medians[BASELINE]["step_seconds"])
        peak_ratio = positive_ratio(medians[CANDIDATE]["peak_allocated_bytes"], medians[BASELINE]["peak_allocated_bytes"])
        summaries.append({
            "sequence_length": length, "models": medians,
            "candidate_over_baseline_step": step_ratio,
            "candidate_over_baseline_peak_allocated": peak_ratio,
            "step_passed": step_ratio <= STEP_LIMIT,
            "peak_passed": peak_ratio <= PEAK_LIMIT,
        })
    return {
        "status": "passed" if all(row["step_passed"] and row["peak_passed"] for row in summaries) else "failed",
        "step_limit": STEP_LIMIT, "peak_limit": PEAK_LIMIT,
        "aggregation": "ratio_of_medians_over_three_independent_paired_repeats",
        "lengths": summaries,
    }


def synthetic_batch(sequence_length: int, device: Any) -> tuple[Any, Any, Any]:
    import torch

    if sequence_length not in LENGTHS:
        raise ValueError("Sequence length is outside the frozen profile grid")
    generator = torch.Generator(device="cpu").manual_seed(98173 + sequence_length)
    dts = torch.randint(1, 12, (BATCH_SIZE, sequence_length), generator=generator).float()
    quantities = torch.randint(1, 35, (BATCH_SIZE, sequence_length), generator=generator).float()
    # Last column is a target; production target_outputs controls its visibility.
    quantities[:, -1] = (0.6 * quantities[:, -2] + 0.25 * quantities[:, -3]
                         + dts[:, -2].remainder(3.0)).round().clamp_min(1.0)
    mask = torch.ones_like(dts, dtype=torch.bool)
    return dts.to(device), quantities.to(device), mask.to(device)


def tensor_tree_finite(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tensor_tree_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_tree_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def state_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def kernel_gradient_audit(model: Any) -> dict[str, Any]:
    import torch

    parameters = dict(model.named_parameters())
    result = {}
    for name in KERNEL_KEYS:
        parameter = parameters[name]
        if tuple(parameter.shape) != (3, HIDDEN_DIM) or parameter.grad is None:
            raise ValueError(f"Missing expected three-lag kernel gradient: {name}")
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError(f"Non-finite new kernel gradient: {name}")
        norms = [float(gradient[index].norm().item()) for index in range(3)]
        if any(value <= 0.0 for value in norms):
            raise ValueError(f"New kernel current/lag1/lag2 gradient must all be nonzero: {name}")
        result[name] = {"gradient_norm_by_current_lag1_lag2": norms}
    return result


def learned_path_audit(model: Any, batch: tuple[Any, Any, Any], gradients: dict[str, Any]) -> dict[str, Any]:
    import torch

    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    parameters = dict(model.named_parameters())
    saved = {name: parameters[name].detach().clone() for name in KERNEL_KEYS}
    weights = {}
    for name, tensor in saved.items():
        norms = [float(tensor[index].norm().item()) for index in range(3)]
        if any(not math.isfinite(value) or value <= 0.0 for value in norms):
            raise ValueError(f"Every new kernel lag must have learned finite nonzero weights: {name}")
        weights[name] = {"weight_norm_by_current_lag1_lag2": norms}
    dts, quantities, mask = batch
    training_before = model.training
    digest_before = state_digest(model)
    model.eval()

    def observe() -> tuple[Any, Any]:
        history_quantities = quantities.clone()
        history_quantities[:, -1] = 0.0
        writes = mask.clone()
        writes[:, -1] = False
        _time, quantity_states = model.encode_task_states(
            dts, history_quantities, mask, memory_write_mask=writes,
        )
        outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        return quantity_states[:, -2].detach().clone(), outputs["pred_qty"].detach().clone()

    try:
        with torch.no_grad():
            learned_state, learned_prediction = observe()
            for name in KERNEL_KEYS:
                parameters[name].zero_()
            zero_state, zero_prediction = observe()
    finally:
        with torch.no_grad():
            for name, tensor in saved.items():
                parameters[name].copy_(tensor)
        model.train(training_before)
    restored = digest_before == state_digest(model)
    hidden_delta = float((learned_state - zero_state).abs().max().item())
    prediction_delta = float((learned_prediction - zero_prediction).abs().max().item())
    if (not restored or not math.isfinite(hidden_delta) or not math.isfinite(prediction_delta)
            or hidden_delta <= 0.0 or prediction_delta <= 0.0):
        raise ValueError("Learned convolution must affect the final causal state and quantity prediction")
    return {
        "status": "passed", "first_backward": gradients, "learned_weights": weights,
        "zero_kernel_counterfactual_final_state_max_abs_delta": hidden_delta,
        "zero_kernel_counterfactual_quantity_max_abs_delta": prediction_delta,
        "state_digest_restored": restored,
    }


def run_worker(backbone: str, sequence_length: int, repeat: int) -> dict[str, Any]:
    import torch

    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    if backbone not in (BASELINE, CANDIDATE) or sequence_length not in LENGTHS:
        raise ValueError("Worker identity differs from the frozen model/length grid")
    measurement_order(repeat)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU result may qualify this profile")
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    torch.manual_seed(42 + repeat)
    torch.cuda.manual_seed_all(42 + repeat)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=HIDDEN_DIM, train_log_mean=2.0,
        max_seq_len=256, quantity_variant="count_only_log_regression", lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp", time_intercept_limit=300.0,
    )
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = synthetic_batch(sequence_length, device)
    if backbone == CANDIDATE:
        parameters = dict(model.named_parameters())
        if any(name not in parameters or tuple(parameters[name].shape) != (3, HIDDEN_DIM)
               or bool(parameters[name].detach().count_nonzero()) for name in KERNEL_KEYS):
            raise ValueError("Candidate must start with the exact zero-initialized kernel contract")
    gradients: dict[str, Any] = {}

    def step(*, inspect_gradients: bool = False) -> tuple[Any, Any]:
        nonlocal gradients
        dts, quantities, mask = batch
        outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        loss = outputs["joint_loss"].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if inspect_gradients:
            gradients = kernel_gradient_audit(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        optimizer.step()
        return loss.detach(), grad_norm.detach()

    for index in range(WARMUP_STEPS):
        loss, grad_norm = step(inspect_gradients=backbone == CANDIDATE and index == 0)
        if not tensor_tree_finite((loss, grad_norm)):
            raise FloatingPointError("Non-finite warmup loss/gradient")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(MEASURED_STEPS):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        loss, grad_norm = step()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        if not tensor_tree_finite((loss, grad_norm)):
            raise FloatingPointError("Non-finite measured loss/gradient")
    peak = int(torch.cuda.max_memory_allocated(device))
    finite = tensor_tree_finite((model.state_dict(), optimizer.state_dict()))
    if not finite:
        raise FloatingPointError("Model or optimizer contains non-finite values after measured steps")
    learning = learned_path_audit(model, batch, gradients) if backbone == CANDIDATE else None
    return {
        "status": "passed", "backbone": backbone, "sequence_length": sequence_length,
        "observed_events": sequence_length - 1, "target_events": 1,
        "batch_size": BATCH_SIZE, "hidden_dim": HIDDEN_DIM, "repeat": repeat,
        "seed": 42 + repeat, "warmup_steps": WARMUP_STEPS, "measured_steps": MEASURED_STEPS,
        "step_seconds": timings, "median_step_seconds": statistics.median(timings),
        "peak_allocated_bytes": peak, "finite_model_optimizer": finite,
        "learning_path": learning, "model_metadata": metadata,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "tf32_matmul": False,
        "measurement_scope": "train_forward_backward_clip_AdamW_synchronized_wall_time",
        "model_state_after_profile_sha256": state_digest(model),
    }


def run_profile(output: Path) -> dict[str, Any]:
    workers = output.parent / (output.stem + "_workers")
    if output.exists() or workers.exists():
        raise FileExistsError("Refusing to overwrite an existing cost proof or workers directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    workers.mkdir()
    payload: dict[str, Any] = {
        "status": "running", "profile_contract": "hard_lmm_causal_qkv_cuda_profile_v1",
        "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_scope": "synthetic_only_no_dataset_or_checkpoint_loaded",
        "worker_process_isolation": True, "rows": [],
    }
    save_json(output, payload)
    try:
        import torch
        if not torch.cuda.is_available() or "5090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("This approved profile requires an RTX 5090 CUDA device")
        for length in LENGTHS:
            for repeat in range(REPEATS):
                for name in measurement_order(repeat):
                    worker_output = workers / f"L{length}_repeat{repeat}_{name}.json"
                    command = [sys.executable, "-s", str(Path(__file__).resolve()),
                               "--worker", "--backbone", name, "--sequence-length", str(length),
                               "--repeat", str(repeat), "--output", str(worker_output)]
                    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
                    log_path = worker_output.with_suffix(".log")
                    log_path.write_text(process.stdout + process.stderr)
                    if process.returncode != 0:
                        raise RuntimeError(f"Cost worker failed ({name}, L{length}, repeat {repeat}); see {log_path}")
                    row = json.loads(worker_output.read_text())
                    if (row.get("backbone"), row.get("sequence_length"), row.get("repeat")) != (name, length, repeat):
                        raise ValueError("Worker output identity differs from its command")
                    payload["rows"].append(row)
                    save_json(output, payload)
        payload["cost_gate"] = summarize_cost(payload["rows"])
        payload["status"] = payload["cost_gate"]["status"]
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
    save_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backbone", choices=(BASELINE, CANDIDATE), help=argparse.SUPPRESS)
    parser.add_argument("--sequence-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.output.exists():
            parser.error("Refusing to overwrite a worker proof")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = run_worker(args.backbone, args.sequence_length, args.repeat)
        except Exception as error:
            result = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        save_json(args.output, result)
    else:
        result = run_profile(args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
