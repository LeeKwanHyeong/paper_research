#!/usr/bin/env python3
"""RTX 5090 cost/learning audit for B, FULL and bounded-Q/K Hard-LMM.

The profile uses only fixed synthetic next-event batches. Each grid cell runs
in a fresh process; the three repeat orders place each model in each position.
Timing excludes the post-training counterfactual audits.
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

from paper.scripts import profile_hard_lmm_causal_qkv as original


BASELINE = "titantpp"
FULL = "titantpp_hard_memory_causal_qkv"
CANDIDATE = "titantpp_hard_memory_bounded_qk"
BACKBONES = (BASELINE, FULL, CANDIDATE)
PROFILE_CONTRACT = "hard_lmm_bounded_qk_cuda_profile_v1"
LENGTHS = original.LENGTHS
BATCH_SIZE = original.BATCH_SIZE
HIDDEN_DIM = original.HIDDEN_DIM
WARMUP_STEPS = original.WARMUP_STEPS
MEASURED_STEPS = original.MEASURED_STEPS
REPEATS = original.REPEATS
STEP_LIMIT = original.STEP_LIMIT
PEAK_LIMIT = original.PEAK_LIMIT
save_json = original.save_json
positive_ratio = original.positive_ratio
synthetic_batch = original.synthetic_batch
tensor_tree_finite = original.tensor_tree_finite
state_digest = original.state_digest
KERNEL_KEYS = {
    FULL: tuple(f"encoder.layers.0.attn.causal_{kind}_kernel" for kind in ("q", "k", "v")),
    CANDIDATE: tuple(f"encoder.layers.0.attn.bounded_{kind}_kernel" for kind in ("q", "k", "v")),
}
COMMON_GRADIENT_KEYS = (
    "encoder.layers.0.attn.qkv.weight", "encoder.layers.1.attn.qkv.weight",
    "lmm.mem", "quantity_head.weight", "v_t.weight",
)


def measurement_order(repeat: int) -> tuple[str, ...]:
    if type(repeat) is not int or not 0 <= repeat < REPEATS:
        raise ValueError("Unexpected repeat index")
    return BACKBONES[repeat:] + BACKBONES[:repeat]


def require_5090() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU cannot qualify this cost proof")
    device = torch.device("cuda:0")
    if "RTX 5090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("This approved profile requires an RTX 5090 CUDA device")
    return device


def gradient_audit(model: Any, backbone: str) -> dict[str, Any]:
    import torch

    if backbone not in BACKBONES:
        raise ValueError("Unknown gradient-audit backbone")
    parameters = dict(model.named_parameters())
    shared = {}
    for name in COMMON_GRADIENT_KEYS:
        parameter = parameters.get(name)
        if parameter is None or parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
            raise ValueError(f"Missing or non-finite shared-path gradient: {name}")
        norm = float(parameter.grad.detach().norm().item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"Shared-path gradient must be nonzero: {name}")
        shared[name] = norm
    kernel_rows = {}
    for name in KERNEL_KEYS.get(backbone, ()):
        parameter = parameters.get(name)
        if parameter is None or tuple(parameter.shape) != (3, HIDDEN_DIM) or parameter.grad is None:
            raise ValueError(f"Missing current/lag1/lag2 kernel gradient: {name}")
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"Non-finite kernel gradient: {name}")
        norms = [float(row.norm().item()) for row in gradient]
        if any(value <= 0.0 or not math.isfinite(value) for value in norms):
            raise ValueError(f"Every current/lag1/lag2 gradient must be nonzero: {name}")
        kernel_rows[name] = norms
    return {"status": "passed", "shared_gradient_norms": shared, "kernel_gradient_norm_by_lag": kernel_rows}


def initial_backbone_state(model: Any) -> dict[str, Any]:
    """Keep CPU copies outside CUDA memory/timing measurements."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(("encoder.", "lmm."))
    }


def counterfactual_path_audit(model: Any, batch: tuple[Any, Any, Any], replacement: dict[str, Any]) -> dict[str, Any]:
    import torch

    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    state = model.state_dict()
    if not replacement or not set(replacement).issubset(state):
        raise ValueError("Counterfactual replacement must identify existing backbone tensors")
    saved = {name: state[name].detach().clone() for name in replacement}
    digest_before = state_digest(model)
    training_before = model.training
    model.eval()
    dts, quantities, mask = batch

    def observe() -> tuple[Any, Any]:
        history = quantities.clone()
        history[:, -1] = 0.0
        writes = mask.clone()
        writes[:, -1] = False
        _, hidden = model.encode_task_states(dts, history, mask, memory_write_mask=writes)
        output = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        return hidden[:, -2].detach().clone(), output["pred_qty"].detach().clone()

    try:
        with torch.no_grad():
            learned_hidden, learned_prediction = observe()
            for name, value in replacement.items():
                state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))
            reference_hidden, reference_prediction = observe()
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                state[name].copy_(value)
        model.train(training_before)
    hidden_delta = float((learned_hidden - reference_hidden).abs().max().item())
    prediction_delta = float((learned_prediction - reference_prediction).abs().max().item())
    restored = state_digest(model) == digest_before
    if not restored or any(not math.isfinite(value) or value <= 0.0 for value in (hidden_delta, prediction_delta)):
        raise ValueError("Learned backbone must affect hidden state and prediction and restore exactly")
    return {
        "status": "passed", "state_digest_restored": restored,
        "final_state_max_abs_delta": hidden_delta,
        "quantity_prediction_max_abs_delta": prediction_delta,
    }


def bounded_gain_audit(model: Any, batch: tuple[Any, Any, Any]) -> dict[str, Any]:
    import torch

    from models.TPPs.CountAwareTitanBoundedQK import BOUNDED_QK_EPSILON

    attention = model.bounded_qk_attention
    projections = []
    handle = attention.qkv.register_forward_hook(lambda _module, _inputs, output: projections.append(output.detach()))
    training_before = model.training
    model.eval()
    dts, quantities, mask = batch
    try:
        with torch.no_grad():
            history = quantities.clone()
            history[:, -1] = 0.0
            model.encode(dts, history, mask)
            q, k, v = projections[-1].chunk(3, dim=-1)
            rows = {}
            for kind, p in (("q", q), ("k", k)):
                residual = attention.causal_depthwise_residual(p * mask.unsqueeze(-1), getattr(attention, f"bounded_{kind}_kernel"))
                bounded = attention.bound_qk_residual(p, residual)
                shape = (*p.shape[:-1], attention.n_heads, attention.head_dim)
                scale_squared = p.reshape(shape).square().mean(-1) + BOUNDED_QK_EPSILON
                residual_squared = residual.reshape(shape).square().mean(-1)
                gain = (scale_squared / (scale_squared + residual_squared)).sqrt()
                ratio = (bounded.reshape(shape).square().mean(-1) / scale_squared).sqrt()
                if not tensor_tree_finite((gain, ratio)) or not bool((ratio <= 1.000001).all()) or not bool((gain < 1.0).any()):
                    raise ValueError(f"Learned {kind.upper()} must have an active finite per-head bound")
                rows[kind] = {
                    "gain_min": float(gain.min()), "gain_max": float(gain.max()),
                    "gain_mean": float(gain.mean()),
                    "fraction_gain_below_one": float((gain < 1.0).float().mean()),
                    "bounded_residual_rms_over_projection_rms_max": float(ratio.max()),
                }
            expected_v = v + attention.causal_depthwise_residual(v * mask.unsqueeze(-1), attention.bounded_v_kernel)
            actual_v = attention._adapt_event_projections(q, k, v, mask, input_dtype=dts.dtype)[2]
            if not torch.equal(expected_v, actual_v):
                raise ValueError("Bounded-QK changed the original causal-V operation")
    finally:
        handle.remove()
        model.train(training_before)
    return {"status": "passed", "qk": rows, "v_operation_bitwise_identical": True}


def learned_path_audit(model: Any, backbone: str, batch: tuple[Any, Any, Any], initial: dict[str, Any]) -> dict[str, Any]:
    import torch

    result = {
        "status": "passed",
        "initial_backbone_counterfactual": counterfactual_path_audit(model, batch, initial),
    }
    if backbone in KERNEL_KEYS:
        parameters = dict(model.named_parameters())
        norms = {}
        for name in KERNEL_KEYS[backbone]:
            norms[name] = [float(row.detach().norm()) for row in parameters[name]]
            if any(not math.isfinite(value) or value <= 0.0 for value in norms[name]):
                raise ValueError(f"Every current/lag1/lag2 weight must learn: {name}")
        result["learned_kernel_norm_by_lag"] = norms
        result["zero_kernel_counterfactual"] = counterfactual_path_audit(
            model, batch, {name: torch.zeros_like(parameters[name]) for name in KERNEL_KEYS[backbone]},
        )
    if backbone == CANDIDATE:
        result["bounded_gain"] = bounded_gain_audit(model, batch)
    return result


def summarize_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = {(length, repeat, name) for length in LENGTHS for repeat in range(REPEATS) for name in BACKBONES}
    identities = [(row.get("sequence_length"), row.get("repeat"), row.get("backbone")) for row in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Cost proof must contain exactly the complete three-model grid")
    runtime_identities = set()
    for row in rows:
        if row.get("status") != "passed" or row.get("profile_contract") != PROFILE_CONTRACT:
            raise ValueError("Failed or foreign worker cannot qualify a cost proof")
        if any(row.get(key) != value for key, value in {
            "batch_size": BATCH_SIZE, "hidden_dim": HIDDEN_DIM,
            "warmup_steps": WARMUP_STEPS, "measured_steps": MEASURED_STEPS,
            "seed": 42 + row["repeat"], "tf32_matmul": False,
            "amp_enabled": False, "measurement_dtype": "float32",
        }.items()):
            raise ValueError("Worker measurement policy differs from the frozen policy")
        if "RTX 5090" not in str(row.get("device_name", "")):
            raise ValueError("Cost proof must use RTX 5090")
        runtime_identities.add((row.get("device_name"), row.get("torch_version"), row.get("cuda_version")))
        times = row.get("step_seconds", [])
        if len(times) != MEASURED_STEPS:
            raise ValueError("Every measured step must be retained")
        for value in times:
            positive_ratio(value, 1.0)
        positive_ratio(row.get("median_step_seconds", 0), 1.0)
        positive_ratio(row.get("peak_allocated_bytes", 0), 1.0)
        if not math.isclose(statistics.median(times), row["median_step_seconds"], rel_tol=1e-12, abs_tol=0):
            raise ValueError("Worker median does not reconstruct from measured steps")
        if row.get("finite_model_optimizer") is not True:
            raise ValueError("Worker has no finite model/optimizer proof")
        if row.get("gradient_audit", {}).get("status") != "passed" or row.get("learning_path", {}).get("status") != "passed":
            raise ValueError("Every model needs gradient and learned-path proof")
        if row["backbone"] == CANDIDATE and row["learning_path"].get("bounded_gain", {}).get("status") != "passed":
            raise ValueError("Candidate has no active Q/K bound proof")
    if len(runtime_identities) != 1:
        raise ValueError("Cost grid mixes runtimes")
    summaries = []
    for length in LENGTHS:
        medians = {}
        counts = {}
        for name in BACKBONES:
            matching = [row for row in rows if row["sequence_length"] == length and row["backbone"] == name]
            medians[name] = {
                "step_seconds": statistics.median(row["median_step_seconds"] for row in matching),
                "peak_allocated_bytes": statistics.median(row["peak_allocated_bytes"] for row in matching),
            }
            model_counts = {row.get("parameter_count") for row in matching}
            if len(model_counts) != 1 or not all(type(value) is int and value > 0 for value in model_counts):
                raise ValueError("Parameter counts differ between repeats")
            counts[name] = model_counts.pop()
        if counts[CANDIDATE] != counts[FULL] or counts[CANDIDATE] != counts[BASELINE] + 576:
            raise ValueError("B/FULL/BOUNDED parameter-count contract drift")
        ratios = {}
        for label, numerator, denominator in (
            ("candidate_over_baseline", CANDIDATE, BASELINE),
            ("candidate_over_full", CANDIDATE, FULL),
            ("full_over_baseline", FULL, BASELINE),
        ):
            ratios[label + "_step"] = positive_ratio(medians[numerator]["step_seconds"], medians[denominator]["step_seconds"])
            ratios[label + "_peak_allocated"] = positive_ratio(medians[numerator]["peak_allocated_bytes"], medians[denominator]["peak_allocated_bytes"])
        summaries.append({
            "sequence_length": length, "models": medians, "parameter_counts": counts, **ratios,
            "step_passed": ratios["candidate_over_baseline_step"] <= STEP_LIMIT,
            "peak_passed": ratios["candidate_over_baseline_peak_allocated"] <= PEAK_LIMIT,
        })
    return {
        "status": "passed" if all(row["step_passed"] and row["peak_passed"] for row in summaries) else "failed",
        "step_limit": STEP_LIMIT, "peak_limit": PEAK_LIMIT,
        "aggregation": "ratio_of_medians_over_three_independent_balanced_repeats",
        "gate_reference": BASELINE, "full_ratios_are_reported_not_separately_gated": True,
        "lengths": summaries,
    }


def run_worker(backbone: str, sequence_length: int, repeat: int) -> dict[str, Any]:
    import torch

    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    if backbone not in BACKBONES or sequence_length not in LENGTHS:
        raise ValueError("Worker identity differs from the frozen profile grid")
    measurement_order(repeat)
    device = require_5090()
    torch.set_num_threads(1)
    torch.manual_seed(42 + repeat)
    torch.cuda.manual_seed_all(42 + repeat)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, metadata = build_count_aware_model(
        backbone, hidden_dim=HIDDEN_DIM, train_log_mean=2.0,
        max_seq_len=256, quantity_variant="count_only_log_regression", lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp", time_intercept_limit=300.0,
    )
    initial = initial_backbone_state(model)
    for name in KERNEL_KEYS.get(backbone, ()):
        if tuple(initial[name].shape) != (3, HIDDEN_DIM) or bool(initial[name].count_nonzero()):
            raise ValueError("Candidate must start with exactly zero causal kernels")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = synthetic_batch(sequence_length, device)
    gradients = {}

    def step(inspect_gradients=False):
        nonlocal gradients
        dts, quantities, mask = batch
        outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss = outputs["joint_loss"].mean()
        loss.backward()
        if inspect_gradients:
            gradients = gradient_audit(model, backbone)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        return loss.detach(), norm.detach()

    for index in range(WARMUP_STEPS):
        if not tensor_tree_finite(step(inspect_gradients=index == 0)):
            raise FloatingPointError("Non-finite warmup loss/gradient")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(MEASURED_STEPS):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        observed = step()
        torch.cuda.synchronize(device)
        timings.append(time.perf_counter() - started)
        if not tensor_tree_finite(observed):
            raise FloatingPointError("Non-finite measured loss/gradient")
    peak = int(torch.cuda.max_memory_allocated(device))
    finite = tensor_tree_finite((model.state_dict(), optimizer.state_dict()))
    if not finite:
        raise FloatingPointError("Model/optimizer contains non-finite state")
    learning = learned_path_audit(model, backbone, batch, initial)
    return {
        "status": "passed", "profile_contract": PROFILE_CONTRACT,
        "backbone": backbone, "sequence_length": sequence_length, "repeat": repeat,
        "observed_events": sequence_length - 1, "target_events": 1,
        "batch_size": BATCH_SIZE, "hidden_dim": HIDDEN_DIM, "seed": 42 + repeat,
        "warmup_steps": WARMUP_STEPS, "measured_steps": MEASURED_STEPS,
        "step_seconds": timings, "median_step_seconds": statistics.median(timings),
        "peak_allocated_bytes": peak, "finite_model_optimizer": finite,
        "gradient_audit": gradients, "learning_path": learning,
        "model_metadata": metadata, "parameter_count": sum(p.numel() for p in model.parameters()),
        "device_name": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda, "tf32_matmul": False,
        "amp_enabled": False, "measurement_dtype": "float32",
        "measurement_scope": "train_forward_backward_clip_AdamW_synchronized_wall_time",
        "model_state_after_profile_sha256": state_digest(model),
    }


def run_profile(output: Path) -> dict[str, Any]:
    workers = output.parent / (output.stem + "_workers")
    if output.exists() or workers.exists():
        raise FileExistsError("Refusing to overwrite an existing cost proof or workers directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    workers.mkdir()
    payload = {
        "status": "running", "profile_contract": PROFILE_CONTRACT,
        "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_profile_file_sha256": hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        "data_scope": "synthetic_only_no_dataset_or_checkpoint_loaded",
        "worker_process_isolation": True,
        "measurement_orders": [list(measurement_order(repeat)) for repeat in range(REPEATS)],
        "rows": [],
    }
    save_json(output, payload)
    try:
        require_5090()
        for length in LENGTHS:
            for repeat in range(REPEATS):
                for name in measurement_order(repeat):
                    worker_output = workers / f"L{length}_repeat{repeat}_{name}.json"
                    process = subprocess.run(
                        [sys.executable, "-s", str(Path(__file__).resolve()), "--worker",
                         "--backbone", name, "--sequence-length", str(length),
                         "--repeat", str(repeat), "--output", str(worker_output)],
                        cwd=ROOT, capture_output=True, text=True, timeout=600,
                    )
                    log_path = worker_output.with_suffix(".log")
                    log_path.write_text(process.stdout + process.stderr)
                    if process.returncode != 0:
                        raise RuntimeError(f"Cost worker failed ({name}, L{length}, repeat {repeat}); see {log_path}")
                    row = json.loads(worker_output.read_text())
                    if (row.get("backbone"), row.get("sequence_length"), row.get("repeat")) != (name, length, repeat):
                        raise ValueError("Worker identity differs from its command")
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
    parser.add_argument("--backbone", choices=BACKBONES, help=argparse.SUPPRESS)
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
