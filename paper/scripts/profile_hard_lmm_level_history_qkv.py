#!/usr/bin/env python3
"""RTX 5090 cost and mechanism audit for the LPHC-QKV Hard-LMM candidate.

The profile compares the common Hard-LMM B route, the earlier unrestricted
causal-QKV route, and the level-preserving history-confidence (LPHC) route on
fixed synthetic next-event batches.  Every grid cell runs in a fresh process;
the balanced repeat orders place each model in each measurement position.
Timing excludes the post-training counterfactual and mechanism audits.
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
CANDIDATE = "titantpp_hard_memory_level_history_qkv"
BACKBONES = (BASELINE, FULL, CANDIDATE)
PROFILE_CONTRACT = "hard_lmm_level_history_qkv_cuda_profile_v1"
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
    FULL: tuple(
        f"encoder.layers.0.attn.causal_{kind}_kernel"
        for kind in ("q", "k", "v")
    ),
    CANDIDATE: tuple(
        f"encoder.layers.0.attn.level_history_{kind}_kernel"
        for kind in ("q", "k", "v")
    ),
}
KERNEL_ROWS = {FULL: 3, CANDIDATE: 2}
COMMON_GRADIENT_KEYS = (
    "encoder.layers.0.attn.qkv.weight",
    "encoder.layers.1.attn.qkv.weight",
    "lmm.mem",
    "quantity_head.weight",
    "v_t.weight",
)


def measurement_order(repeat: int) -> tuple[str, ...]:
    """Rotate the three models so each occupies every timing position once."""
    if type(repeat) is not int or not 0 <= repeat < REPEATS:
        raise ValueError("Unexpected repeat index")
    return BACKBONES[repeat:] + BACKBONES[:repeat]


def require_5090() -> Any:
    """Reject CPU and non-5090 measurements as cost evidence."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU cannot qualify this cost proof")
    device = torch.device("cuda:0")
    if "RTX 5090" not in torch.cuda.get_device_name(device):
        raise RuntimeError("This approved profile requires an RTX 5090 CUDA device")
    return device


def gradient_audit(model: Any, backbone: str) -> dict[str, Any]:
    """Require finite shared gradients and every available kernel row to learn."""
    import torch

    if backbone not in BACKBONES:
        raise ValueError("Unknown gradient-audit backbone")
    parameters = dict(model.named_parameters())
    shared: dict[str, float] = {}
    for name in COMMON_GRADIENT_KEYS:
        parameter = parameters.get(name)
        if (
            parameter is None
            or parameter.grad is None
            or not bool(torch.isfinite(parameter.grad).all())
        ):
            raise ValueError(f"Missing or non-finite shared-path gradient: {name}")
        norm = float(parameter.grad.detach().norm().item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"Shared-path gradient must be nonzero: {name}")
        shared[name] = norm

    kernel_rows: dict[str, list[float]] = {}
    expected_rows = KERNEL_ROWS.get(backbone)
    for name in KERNEL_KEYS.get(backbone, ()):
        parameter = parameters.get(name)
        if (
            expected_rows is None
            or parameter is None
            or tuple(parameter.shape) != (expected_rows, HIDDEN_DIM)
            or parameter.grad is None
        ):
            raise ValueError(f"Missing kernel gradient with the contracted shape: {name}")
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise ValueError(f"Non-finite kernel gradient: {name}")
        norms = [float(row.norm().item()) for row in gradient]
        if any(value <= 0.0 or not math.isfinite(value) for value in norms):
            raise ValueError(f"Every contracted kernel row must have a nonzero gradient: {name}")
        kernel_rows[name] = norms
    return {
        "status": "passed",
        "shared_gradient_norms": shared,
        "kernel_gradient_norm_by_row": kernel_rows,
    }


def initial_backbone_state(model: Any) -> dict[str, Any]:
    """Keep CPU copies outside CUDA memory and timing measurements."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(("encoder.", "lmm."))
    }


def counterfactual_path_audit(
    model: Any,
    batch: tuple[Any, Any, Any],
    replacement: dict[str, Any],
) -> dict[str, Any]:
    """Replace a path temporarily and prove both effect and exact restoration."""
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
        _, hidden = model.encode_task_states(
            dts,
            history,
            mask,
            memory_write_mask=writes,
        )
        output = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
        return hidden[:, -2].detach().clone(), output["pred_qty"].detach().clone()

    try:
        with torch.no_grad():
            learned_hidden, learned_prediction = observe()
            for name, value in replacement.items():
                state[name].copy_(
                    value.to(device=state[name].device, dtype=state[name].dtype)
                )
            reference_hidden, reference_prediction = observe()
    finally:
        with torch.no_grad():
            for name, value in saved.items():
                state[name].copy_(value)
        model.train(training_before)
    hidden_delta = float((learned_hidden - reference_hidden).abs().max().item())
    prediction_delta = float(
        (learned_prediction - reference_prediction).abs().max().item()
    )
    restored = state_digest(model) == digest_before
    if not restored or any(
        not math.isfinite(value) or value <= 0.0
        for value in (hidden_delta, prediction_delta)
    ):
        raise ValueError(
            "Learned backbone must affect hidden state and prediction and restore exactly"
        )
    return {
        "status": "passed",
        "state_digest_restored": restored,
        "final_state_max_abs_delta": hidden_delta,
        "quantity_prediction_max_abs_delta": prediction_delta,
    }


def _transition_scale_squared(
    attention: Any,
    lag1: Any,
    lag2: Any,
    available1: Any,
    available2: Any,
) -> Any:
    """Reconstruct the contract's per-head transition-only scale."""
    import torch

    from models.TPPs.CountAwareTitanLevelHistoryQKV import (
        LEVEL_HISTORY_QKV_EPSILON,
    )

    norm_dtype = (
        torch.float32
        if lag1.dtype in (torch.float16, torch.bfloat16)
        else lag1.dtype
    )
    shape = (*lag1.shape[:-1], attention.n_heads, attention.head_dim)
    first = lag1.to(dtype=norm_dtype).reshape(shape)
    second = lag2.to(dtype=norm_dtype).reshape(shape)
    first_available = available1.to(dtype=norm_dtype)[..., None, None]
    second_available = available2.to(dtype=norm_dtype)[..., None, None]
    count = (first_available + second_available).clamp_min(1.0)
    return (
        first_available * first.square().mean(dim=-1, keepdim=True)
        + second_available * second.square().mean(dim=-1, keepdim=True)
    ) / count + LEVEL_HISTORY_QKV_EPSILON


def transition_mechanism_audit(
    model: Any,
    batch: tuple[Any, Any, Any],
) -> dict[str, Any]:
    """Audit learned confidence, transition RMS bounds, DC-null, and V route."""
    import torch

    attention = model.level_history_qkv_attention
    projections: list[Any] = []
    handle = attention.qkv.register_forward_hook(
        lambda _module, _inputs, output: projections.append(output.detach())
    )
    training_before = model.training
    model.eval()
    dts, quantities, mask = batch
    try:
        with torch.no_grad():
            history = quantities.clone()
            history[:, -1] = 0.0
            model.encode(dts, history, mask)
            q, k, v = projections[-1].chunk(3, dim=-1)
            rows: dict[str, Any] = {}
            expected_confidence = (
                torch.arange(mask.size(1), device=mask.device, dtype=q.dtype)
                / torch.arange(
                    1,
                    mask.size(1) + 1,
                    device=mask.device,
                    dtype=q.dtype,
                )
            ).expand(mask.size(0), -1)
            observed_confidence = None
            for kind, projection in (("q", q), ("k", k), ("v", v)):
                kernel = getattr(attention, f"level_history_{kind}_kernel")
                residual, lag1, lag2, available1, available2, confidence = (
                    attention.transition_residual(projection, kernel, mask)
                )
                if observed_confidence is None:
                    observed_confidence = confidence
                elif not torch.equal(observed_confidence, confidence):
                    raise ValueError("Q/K/V confidence paths differ")
                if not torch.equal(confidence, expected_confidence):
                    raise ValueError("Contiguous-history confidence differs from (H-1)/H")
                added = residual * confidence.to(dtype=residual.dtype).unsqueeze(-1)
                row: dict[str, Any] = {
                    "confidence_min": float(confidence.min()),
                    "confidence_max": float(confidence.max()),
                    "confidence_final": float(confidence[0, -1]),
                    "lag1_available_fraction": float(available1.float().mean()),
                    "lag2_available_fraction": float(available2.float().mean()),
                }
                if kind in ("q", "k"):
                    bounded = attention.bound_transition_residual(
                        residual,
                        lag1,
                        lag2,
                        available1,
                        available2,
                    )
                    scale_squared = _transition_scale_squared(
                        attention,
                        lag1,
                        lag2,
                        available1,
                        available2,
                    )
                    shape = (
                        *projection.shape[:-1],
                        attention.n_heads,
                        attention.head_dim,
                    )
                    raw_squared = residual.to(scale_squared.dtype).reshape(shape).square().mean(
                        dim=-1,
                        keepdim=True,
                    )
                    gain = (scale_squared / (scale_squared + raw_squared)).sqrt()
                    bounded_added = (
                        bounded
                        * confidence.to(dtype=bounded.dtype).unsqueeze(-1)
                    )
                    added_rms = bounded_added.to(scale_squared.dtype).reshape(shape).square().mean(
                        dim=-1,
                        keepdim=True,
                    ).sqrt()
                    confidence_head = confidence.to(scale_squared.dtype)[..., None, None]
                    limit = confidence_head * scale_squared.sqrt()
                    valid_limit = (available1 | available2)[..., None, None]
                    ratio = torch.where(
                        valid_limit,
                        added_rms / limit.clamp_min(1e-30),
                        torch.zeros_like(added_rms),
                    )
                    if (
                        not tensor_tree_finite((gain, ratio))
                        or not bool((ratio <= 1.000001).all())
                        or not bool((gain[valid_limit.expand_as(gain)] < 1.0).any())
                    ):
                        raise ValueError(
                            f"Learned {kind.upper()} lacks an active finite transition-RMS bound"
                        )
                    row.update(
                        {
                            "gain_min": float(gain.min()),
                            "gain_max": float(gain.max()),
                            "gain_mean": float(gain.mean()),
                            "fraction_valid_gain_below_one": float(
                                (gain[valid_limit.expand_as(gain)] < 1.0)
                                .float()
                                .mean()
                            ),
                            "added_rms_over_confidence_transition_rms_max": float(
                                ratio.max()
                            ),
                        }
                    )
                    added = bounded_added
                rows[kind] = row

                shifted = projection + torch.linspace(
                    -3.0,
                    3.0,
                    projection.size(-1),
                    device=projection.device,
                    dtype=projection.dtype,
                ).view(1, 1, -1)
                shifted_result = attention.transition_residual(shifted, kernel, mask)
                shifted_residual = shifted_result[0]
                if kind in ("q", "k"):
                    shifted_residual = attention.bound_transition_residual(
                        shifted_residual,
                        shifted_result[1],
                        shifted_result[2],
                        shifted_result[3],
                        shifted_result[4],
                    )
                shifted_added = (
                    shifted_residual
                    * shifted_result[5].to(dtype=shifted_residual.dtype).unsqueeze(-1)
                )
                torch.testing.assert_close(shifted_added, added, rtol=1e-5, atol=1e-6)

            constant = torch.ones_like(q) * 7.0
            for kind in ("q", "k", "v"):
                kernel = getattr(attention, f"level_history_{kind}_kernel")
                constant_result = attention.transition_residual(constant, kernel, mask)
                if bool(constant_result[0].count_nonzero()):
                    raise ValueError("Constant valid projection produced a nonzero residual")

            v_result = attention.transition_residual(
                v,
                attention.level_history_v_kernel,
                mask,
            )
            expected_v = v + v_result[0] * v_result[5].to(
                dtype=v_result[0].dtype
            ).unsqueeze(-1)
            actual_v = attention._adapt_event_projections(
                q,
                k,
                v,
                mask,
                input_dtype=dts.dtype,
            )[2]
            if not torch.equal(expected_v, actual_v):
                raise ValueError("LPHC candidate changed the contracted unbounded V route")
    finally:
        handle.remove()
        model.train(training_before)
    return {
        "status": "passed",
        "confidence_contract": "valid_adjacent_pairs_over_valid_events",
        "qkv": rows,
        "constant_projection_dc_null": True,
        "constant_level_translation_invariant": True,
        "v_operation_bitwise_identical": True,
    }


def learned_path_audit(
    model: Any,
    backbone: str,
    batch: tuple[Any, Any, Any],
    initial: dict[str, Any],
) -> dict[str, Any]:
    """Prove every route learned, affects output, and restores after audits."""
    import torch

    result = {
        "status": "passed",
        "initial_backbone_counterfactual": counterfactual_path_audit(
            model,
            batch,
            initial,
        ),
    }
    if backbone in KERNEL_KEYS:
        parameters = dict(model.named_parameters())
        norms: dict[str, list[float]] = {}
        expected_rows = KERNEL_ROWS[backbone]
        for name in KERNEL_KEYS[backbone]:
            if tuple(parameters[name].shape) != (expected_rows, HIDDEN_DIM):
                raise ValueError(f"Learned kernel shape drift: {name}")
            norms[name] = [float(row.detach().norm()) for row in parameters[name]]
            if any(not math.isfinite(value) or value <= 0.0 for value in norms[name]):
                raise ValueError(f"Every contracted kernel row must learn: {name}")
        result["learned_kernel_norm_by_row"] = norms
        result["zero_kernel_counterfactual"] = counterfactual_path_audit(
            model,
            batch,
            {
                name: torch.zeros_like(parameters[name])
                for name in KERNEL_KEYS[backbone]
            },
        )
    if backbone == CANDIDATE:
        result["transition_mechanism"] = transition_mechanism_audit(model, batch)
    return result


def summarize_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate the full proof grid and apply the fixed B-relative cost gates."""
    expected = {
        (length, repeat, name)
        for length in LENGTHS
        for repeat in range(REPEATS)
        for name in BACKBONES
    }
    identities = [
        (
            row.get("sequence_length"),
            row.get("repeat"),
            row.get("backbone"),
        )
        for row in rows
    ]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Cost proof must contain exactly the complete three-model grid")
    runtime_identities = set()
    for row in rows:
        if (
            row.get("status") != "passed"
            or row.get("profile_contract") != PROFILE_CONTRACT
        ):
            raise ValueError("Failed or foreign worker cannot qualify a cost proof")
        fixed = {
            "batch_size": BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "seed": 42 + row["repeat"],
            "tf32_matmul": False,
            "amp_enabled": False,
            "measurement_dtype": "float32",
        }
        if any(row.get(key) != value for key, value in fixed.items()):
            raise ValueError("Worker measurement policy differs from the frozen policy")
        if "RTX 5090" not in str(row.get("device_name", "")):
            raise ValueError("Cost proof must use RTX 5090")
        runtime_identities.add(
            (
                row.get("device_name"),
                row.get("torch_version"),
                row.get("cuda_version"),
            )
        )
        times = row.get("step_seconds", [])
        if len(times) != MEASURED_STEPS:
            raise ValueError("Every measured step must be retained")
        for value in times:
            positive_ratio(value, 1.0)
        positive_ratio(row.get("median_step_seconds", 0), 1.0)
        positive_ratio(row.get("peak_allocated_bytes", 0), 1.0)
        if not math.isclose(
            statistics.median(times),
            row["median_step_seconds"],
            rel_tol=1e-12,
            abs_tol=0,
        ):
            raise ValueError("Worker median does not reconstruct from measured steps")
        if row.get("finite_model_optimizer") is not True:
            raise ValueError("Worker has no finite model/optimizer proof")
        if (
            row.get("gradient_audit", {}).get("status") != "passed"
            or row.get("learning_path", {}).get("status") != "passed"
        ):
            raise ValueError("Every model needs gradient and learned-path proof")
        if row["backbone"] == CANDIDATE and (
            row["learning_path"]
            .get("transition_mechanism", {})
            .get("status")
            != "passed"
        ):
            raise ValueError("Candidate has no transition/confidence mechanism proof")
    if len(runtime_identities) != 1:
        raise ValueError("Cost grid mixes runtimes")

    summaries = []
    for length in LENGTHS:
        medians: dict[str, dict[str, float]] = {}
        counts: dict[str, int] = {}
        for name in BACKBONES:
            matching = [
                row
                for row in rows
                if row["sequence_length"] == length and row["backbone"] == name
            ]
            medians[name] = {
                "step_seconds": statistics.median(
                    row["median_step_seconds"] for row in matching
                ),
                "peak_allocated_bytes": statistics.median(
                    row["peak_allocated_bytes"] for row in matching
                ),
            }
            model_counts = {row.get("parameter_count") for row in matching}
            if len(model_counts) != 1 or not all(
                type(value) is int and value > 0 for value in model_counts
            ):
                raise ValueError("Parameter counts differ between repeats")
            counts[name] = model_counts.pop()
        if (
            counts[FULL] != counts[BASELINE] + 576
            or counts[CANDIDATE] != counts[BASELINE] + 384
        ):
            raise ValueError("B/FULL/LPHC parameter-count contract drift")
        ratios: dict[str, float] = {}
        for label, numerator, denominator in (
            ("candidate_over_baseline", CANDIDATE, BASELINE),
            ("candidate_over_full", CANDIDATE, FULL),
            ("full_over_baseline", FULL, BASELINE),
        ):
            ratios[label + "_step"] = positive_ratio(
                medians[numerator]["step_seconds"],
                medians[denominator]["step_seconds"],
            )
            ratios[label + "_peak_allocated"] = positive_ratio(
                medians[numerator]["peak_allocated_bytes"],
                medians[denominator]["peak_allocated_bytes"],
            )
        summaries.append(
            {
                "sequence_length": length,
                "models": medians,
                "parameter_counts": counts,
                **ratios,
                "step_passed": (
                    ratios["candidate_over_baseline_step"] <= STEP_LIMIT
                ),
                "peak_passed": (
                    ratios["candidate_over_baseline_peak_allocated"] <= PEAK_LIMIT
                ),
            }
        )
    return {
        "status": (
            "passed"
            if all(row["step_passed"] and row["peak_passed"] for row in summaries)
            else "failed"
        ),
        "step_limit": STEP_LIMIT,
        "peak_limit": PEAK_LIMIT,
        "aggregation": "ratio_of_medians_over_three_independent_balanced_repeats",
        "gate_reference": BASELINE,
        "full_ratios_are_reported_not_separately_gated": True,
        "lengths": summaries,
    }


def run_worker(
    backbone: str,
    sequence_length: int,
    repeat: int,
) -> dict[str, Any]:
    """Run one isolated CUDA timing cell plus post-timing path audits."""
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
        backbone,
        hidden_dim=HIDDEN_DIM,
        train_log_mean=2.0,
        max_seq_len=256,
        quantity_variant="count_only_log_regression",
        lambda_tail=0.0,
        time_head_mode="legacy_clamped_rmtpp",
        time_intercept_limit=300.0,
    )
    initial = initial_backbone_state(model)
    expected_rows = KERNEL_ROWS.get(backbone)
    for name in KERNEL_KEYS.get(backbone, ()):
        if (
            expected_rows is None
            or tuple(initial[name].shape) != (expected_rows, HIDDEN_DIM)
            or bool(initial[name].count_nonzero())
        ):
            raise ValueError("Candidate must start with exactly zero contracted kernels")
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = synthetic_batch(sequence_length, device)
    gradients: dict[str, Any] = {}

    def step(inspect_gradients: bool = False) -> tuple[Any, Any]:
        nonlocal gradients
        dts, quantities, mask = batch
        outputs = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = outputs["joint_loss"].mean()
        loss.backward()
        if inspect_gradients:
            gradients = gradient_audit(model, backbone)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
            error_if_nonfinite=True,
        )
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
        "status": "passed",
        "profile_contract": PROFILE_CONTRACT,
        "backbone": backbone,
        "sequence_length": sequence_length,
        "repeat": repeat,
        "observed_events": sequence_length - 1,
        "target_events": 1,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "seed": 42 + repeat,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURED_STEPS,
        "step_seconds": timings,
        "median_step_seconds": statistics.median(timings),
        "peak_allocated_bytes": peak,
        "finite_model_optimizer": finite,
        "gradient_audit": gradients,
        "learning_path": learning,
        "model_metadata": metadata,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32_matmul": False,
        "amp_enabled": False,
        "measurement_dtype": "float32",
        "measurement_scope": (
            "train_forward_backward_clip_AdamW_synchronized_wall_time"
        ),
        "model_state_after_profile_sha256": state_digest(model),
    }


def run_profile(output: Path) -> dict[str, Any]:
    """Run and atomically aggregate the complete isolated worker grid."""
    workers = output.parent / (output.stem + "_workers")
    if output.exists() or workers.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing cost proof or workers directory"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    workers.mkdir()
    payload: dict[str, Any] = {
        "status": "running",
        "profile_contract": PROFILE_CONTRACT,
        "source_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_profile_file_sha256": hashlib.sha256(
            Path(original.__file__).read_bytes()
        ).hexdigest(),
        "data_scope": "synthetic_only_no_dataset_or_checkpoint_loaded",
        "worker_process_isolation": True,
        "measurement_orders": [
            list(measurement_order(repeat)) for repeat in range(REPEATS)
        ],
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
                        [
                            sys.executable,
                            "-s",
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--backbone",
                            name,
                            "--sequence-length",
                            str(length),
                            "--repeat",
                            str(repeat),
                            "--output",
                            str(worker_output),
                        ],
                        cwd=ROOT,
                        capture_output=True,
                        text=True,
                        timeout=600,
                    )
                    log_path = worker_output.with_suffix(".log")
                    log_path.write_text(process.stdout + process.stderr)
                    if process.returncode != 0:
                        raise RuntimeError(
                            "Cost worker failed "
                            f"({name}, L{length}, repeat {repeat}); see {log_path}"
                        )
                    row = json.loads(worker_output.read_text())
                    if (
                        row.get("backbone"),
                        row.get("sequence_length"),
                        row.get("repeat"),
                    ) != (name, length, repeat):
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
            result = run_worker(
                args.backbone,
                args.sequence_length,
                args.repeat,
            )
        except Exception as error:
            result = {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        save_json(args.output, result)
    else:
        result = run_profile(args.output)
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
