#!/usr/bin/env python3
"""Synthetic nonlinear-memory qualification; the public CLI is CPU-only.

Native CUDA is called only by the contract-bound parallel supervisor. No real
training samples or existing checkpoints are opened by this module.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import io
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import nonlinear_episode_parallel_common as common

OWNER_ENV = "NONLINEAR_EPISODE_QUAL_OWNER_PID"
STARTED_ENV = "NONLINEAR_EPISODE_QUAL_STARTED"
CHECK_KEYS = (
    "initial_output_objective_shared_preclip_gradient_exact",
    "activated_target_prediction_causal",
    "activated_Q_K_V_b_U_gradients",
    "saved_optimizer_rng_next_step_exact",
)


def require_native_owner():
    """Reject accidental standalone CUDA calls before touching the device."""
    common.require(os.environ.get(OWNER_ENV) == str(os.getppid()),
                   "Native CUDA requires its contract-bound supervising parent")
    try:
        started = float(os.environ.get(STARTED_ENV, "nan"))
    except ValueError:
        started = float("nan")
    now = time.time()
    common.require(math.isfinite(started) and 0 < started <= now < started + 900,
                   "Native qualification deadline is missing or expired")
    return started


def _cpu_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


def _exact(left, right):
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.cpu(), right.cpu()))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


@contextmanager
def _deterministic_rng(device):
    import torch
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        with torch.random.fork_rng(devices=[0] if device == "cuda:0" else []):
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic


def verify(device="cpu", *, batch_size=128):
    """Run 42 timed updates and six exact-replay updates on synthetic inputs."""
    common.require(device in ("cpu", "cuda:0"), "Only CPU or the pinned CUDA device is supported")
    common.require(type(batch_size) is int and 1 <= batch_size <= 128,
                   "Synthetic batch must be a positive integer at most 128")
    if device == "cuda:0":
        require_native_owner()
        common.require(batch_size == 128, "Native qualification requires the frozen batch128 workload")
    with _deterministic_rng(device):
        return _verify(device, batch_size=batch_size)


def _verify(device, *, batch_size):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from paper.scripts.verify_observed_slot_memory_cpu import _step
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    def seed(value):
        # torch.manual_seed would seed all visible CUDA devices even in CPU mode.
        torch.random.default_generator.manual_seed(value)
        if device == "cuda:0":
            torch.cuda.default_generators[0].manual_seed(value)

    def rng_state():
        result = {"cpu_rng": torch.get_rng_state().clone()}
        if device == "cuda:0":
            result["cuda_rng"] = torch.cuda.get_rng_state(0).clone()
        return result

    def restore_rng(state):
        torch.set_rng_state(state["cpu_rng"])
        if device == "cuda:0":
            torch.cuda.set_rng_state(state["cuda_rng"], 0)

    def build(name):
        seed(42)
        return build_count_aware_model(
            name, hidden_dim=64, train_log_mean=1.5, max_seq_len=256,
            time_intercept_limit=300.,
        )[0]

    def batch(size, length):
        generator = torch.Generator().manual_seed(400 + length)
        return tuple(value.to(device) for value in (
            .1 + 2 * torch.rand(size, length, generator=generator),
            torch.ones(size, length, dtype=torch.bool),
            torch.randint(0, 32, (size, length), generator=generator).float(),
        ))

    def sync():
        if device == "cuda:0":
            torch.cuda.synchronize(0)

    def activate(model):
        with torch.no_grad():
            # B uses the same quantity weights, so its shared backward path is
            # activated identically to both added-memory arms during timing.
            model.quantity_head.weight.copy_(
                torch.linspace(-.1, .1, model.hidden_dim, device=device).reshape(1, -1))
            if hasattr(model, "nonlinear_episode_memory"):
                model.nonlinear_episode_memory.output_projection.weight.copy_(
                    torch.linspace(-.05, .05, model.hidden_dim * 16, device=device)
                    .reshape(model.hidden_dim, 16))

    updates = 0
    baseline = build(common.ARMS[0])
    initial_rng = rng_state()
    base_state = _cpu_copy(baseline.state_dict())
    digest = canonical_state_dict_sha256(base_state)
    baseline.to(device).train()
    tiny = batch(4, 8)
    seed(79)
    outputs = target_outputs(baseline, *tiny, lambda_log_qty=1.)
    outputs["joint_loss"].mean().backward()
    expected = _cpu_copy(outputs)
    base_gradients = {name: _cpu_copy(param.grad) for name, param in baseline.named_parameters()}
    del outputs, baseline
    checks, added_initial = {}, None
    empty_checks = {}
    for name in common.ARMS[1:]:
        model = build(name)
        common.require(_exact(initial_rng, rng_state()), "Initial CPU/CUDA RNG differs")
        state = model.state_dict()
        common.require(all(torch.equal(value, state[key]) for key, value in base_state.items()),
                       "Shared initial state differs")
        added = {key: value.clone() for key, value in state.items()
                 if key.startswith("nonlinear_episode_memory.") and not key.endswith("pooling_code")}
        if added_initial is not None:
            common.require(_exact(added_initial, added), "Pre/post new parameter initialization differs")
        else:
            added_initial = added
        del state, added
        model.to(device).train()
        seed(79)
        actual = target_outputs(model, *tiny, lambda_log_qty=1.)
        common.require(_exact(expected, _cpu_copy(actual)), "Initial device outputs/objective differ")
        actual["joint_loss"].mean().backward()
        parameters = dict(model.named_parameters())
        common.require(all(_exact(gradient, _cpu_copy(parameters[key].grad))
                           for key, gradient in base_gradients.items()),
                       "Initial shared preclip gradient differs")
        del actual, parameters
        model.zero_grad(set_to_none=True)
        activate(model)
        model.eval()
        prediction = target_outputs(model, *tiny, lambda_log_qty=1.)["pred_qty"]
        changed = tuple(value.clone() for value in tiny)
        changed[0][:, -1] += 10000
        changed[2][:, -1] += 10000
        with torch.no_grad():
            altered = target_outputs(model, *changed, lambda_log_qty=1.)["pred_qty"]
        common.require(torch.equal(prediction, altered), "Activated future target leakage")
        prediction.sum().backward()
        common.require(all(param.grad is not None and bool(torch.isfinite(param.grad).all())
                           and bool(torch.count_nonzero(param.grad))
                           for param in model.nonlinear_episode_memory.parameters()),
                       "Activated Q/K/V/b/U gradient is absent or nonfinite")
        del prediction, altered, changed

        # Native kernels must also preserve an empty read when the trained value
        # bias becomes nonzero, especially for the post-pool phi(0) path.
        core = model.nonlinear_episode_memory
        saved_bias = core.value_projection.bias.detach().clone()
        with torch.no_grad():
            core.value_projection.bias.fill_(.2)
            hidden = torch.zeros(2, 3, 64, device=device)
            features = torch.zeros(2, 3, 2, device=device)
            mask = torch.tensor([[True, False, False], [False, False, False]], device=device)
            read = core.read(hidden, features, mask)
            common.require(bool(torch.isfinite(read).all()) and not bool(torch.count_nonzero(read)),
                           "Empty nonlinear memory read differs from exact zero")
            core.value_projection.bias.copy_(saved_bias)
        empty_checks[name] = True
        del core, saved_bias, hidden, features, mask, read

        optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
        _step(model, optimizer, tiny)
        updates += 1
        stream = io.BytesIO()
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), **rng_state()}
        torch.save(saved, stream)
        next_values = _step(model, optimizer, tiny)
        updates += 1
        continued = {"model": _cpu_copy(model.state_dict()),
                     "optimizer": _cpu_copy(optimizer.state_dict()), **rng_state()}
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved)
        replay_values = _step(model, optimizer, tiny)
        updates += 1
        replay = {"model": _cpu_copy(model.state_dict()),
                  "optimizer": _cpu_copy(optimizer.state_dict()), **rng_state()}
        common.require(_exact(continued, replay) and _exact(next_values, replay_values),
                       "Restored model/optimizer/RNG/next-step objective differs")
        checks[name] = {key: True for key in CHECK_KEYS}
        del model, optimizer, saved, stream, continued, replay
    del expected, base_state, base_gradients, tiny, added_initial, initial_rng
    gc.collect()
    if device == "cuda:0":
        torch.cuda.empty_cache()
    rows = []
    for length in (64, 256):
        data = batch(batch_size, length)
        measurements = {}
        order = common.ARMS if length == 64 else tuple(reversed(common.ARMS))
        for name in order:
            model = build(name).to(device)
            activate(model)
            optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
            samples = []
            if device == "cuda:0":
                torch.cuda.reset_peak_memory_stats(0)
            for repeat in range(7):
                seed(110 + repeat)
                sync()
                started = time.perf_counter()
                _step(model, optimizer, data)
                sync()
                updates += 1
                if repeat >= 2:
                    samples.append(time.perf_counter() - started)
            measurements[name] = {
                "parameters": sum(param.numel() for param in model.parameters()),
                "step_seconds": samples, "median_step_seconds": statistics.median(samples),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0) if device == "cuda:0" else None,
            }
            del model, optimizer
            gc.collect()
            if device == "cuda:0":
                torch.cuda.empty_cache()
        for measure in measurements.values():
            measure["step_ratio_to_B"] = measure["median_step_seconds"] / measurements[common.ARMS[0]]["median_step_seconds"]
        rows.append({"batch": batch_size, "length": length, "max_seq_len": 256,
                     "measurements": measurements, "updates_per_arm": 7, "measurement_order": list(order)})
        del data
    common.require(updates == 48, "Synthetic qualification update budget differs")
    if device == "cuda:0":
        require_native_owner()
    return {
        "status": "synthetic_checks_passed", "device": device,
        "real_data_loaded": False, "held_out_evaluated": False,
        "shared_initial_state_sha256": digest, "checks": checks,
        "empty_rows_exact_zero_after_value_bias": empty_checks,
        "new_arm_initial_parameters_exact": True, "costs": rows,
        "synthetic_optimizer_updates": updates,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "threads": torch.get_num_threads()},
        "limitations": [
            "Synthetic timing excludes data loader and validation/checkpoint I/O.",
            "CPU timing is not native CUDA qualification or a full training-hours estimate.",
            "Sequential timings can contain order effects; shape256 reverses the arm order.",
            "B and new arms share activated quantity weights; only new arms have an activated memory projection.",
            "Pre-pool and post-pool have equal parameter counts but different compute paths.",
            "Synthetic finite updates provide no dataset or benchmark performance evidence.",
        ],
        "completed_at_unix": time.time(),
    }


def cost_gates(receipt, contract, runtime):
    """Evaluate the four frozen native resource gates for both shapes and arms."""
    common.require(receipt.get("device") == "cuda:0", "Cost gates require native CUDA allocation measurements")
    common.require(receipt.get("synthetic_optimizer_updates") == 48, "Wrong synthetic update budget")
    rows = receipt.get("costs", [])
    common.require(len(rows) == 2 and {row.get("length") for row in rows} == {64, 256},
                   "Native cost shape coverage differs")
    total_memory = runtime.get("gpu", {}).get("total_memory_bytes")
    common.require(type(total_memory) is int and total_memory > 0, "Device total memory must be positive")
    checks = []
    for row in rows:
        common.require(row.get("batch") == 128 and row.get("updates_per_arm") == 7,
                       "Native cost batch/update budget differs")
        measurements = row.get("measurements", {})
        common.require(set(measurements) == set(common.ARMS), "Native cost arm coverage differs")
        for value in measurements.values():
            samples = value.get("step_seconds", [])
            common.require(isinstance(samples, list) and len(samples) == 5
                           and all(type(item) in (int, float) and math.isfinite(item) and item > 0 for item in samples)
                           and value.get("median_step_seconds") == statistics.median(samples),
                           "Native cost timing must contain five finite positive samples and their exact median")
            common.require(all(type(value.get(key)) is int and value[key] > 0
                               for key in ("parameters", "peak_allocated_bytes")),
                           "Native cost parameter/memory measurements must be positive integers")
        baseline = measurements[common.ARMS[0]]
        for name in common.ARMS[1:]:
            value, gates = measurements[name], contract["cost_gates"]
            checks.append({"backbone": name, "length": row["length"],
                "step": value["median_step_seconds"] <= baseline["median_step_seconds"] * gates["step_ratio_max"],
                "relative_memory": value["peak_allocated_bytes"] <= baseline["peak_allocated_bytes"] * gates["peak_allocated_ratio_max"],
                "device_memory": value["peak_allocated_bytes"] <= total_memory * gates["peak_device_fraction_max"],
                "parameters": value["parameters"] <= baseline["parameters"] * gates["parameter_ratio_max"]})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    common.require(not args.output.exists(), "Fresh CPU output required")
    import torch
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        receipt = verify()
    finally:
        torch.set_num_threads(previous_threads)
    receipt["source_files_sha256"] = common.sha_json(common.source_manifest())
    common.write_json(args.output, receipt, exclusive=True)
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
