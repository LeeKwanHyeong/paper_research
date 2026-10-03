#!/usr/bin/env python3
"""Synthetic task-routed-memory qualification; the public CLI is CPU-only.

Native CUDA is called only by the contract-bound parallel supervisor. No real
training samples or existing checkpoints are opened by this module.
"""
from __future__ import annotations

import argparse
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
from paper.scripts import task_routed_episode_parallel_common as common
from paper.scripts.verify_nonlinear_episode_cost import _cpu_copy, _exact, _deterministic_rng

OWNER_ENV = "TASK_ROUTED_EPISODE_QUAL_OWNER_PID"
STARTED_ENV = "TASK_ROUTED_EPISODE_QUAL_STARTED"
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

    def build(name, max_seq_len=256):
        seed(42)
        return build_count_aware_model(
            name, hidden_dim=64, train_log_mean=1.5, max_seq_len=max_seq_len,
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
            if hasattr(model, "task_routed_episode_memory"):
                model.task_routed_episode_memory.time_output_projection.weight.copy_(
                    torch.linspace(-.05, .05, model.hidden_dim * 16, device=device)
                    .reshape(model.hidden_dim, 16))
                model.task_routed_episode_memory.quantity_output_projection.weight.copy_(
                    torch.linspace(.04, -.03, model.hidden_dim * 16, device=device)
                    .reshape(model.hidden_dim, 16))

    def prediction_states(model, data):
        # Match target_outputs' exclusion of the final target quantity/write;
        # the synthetic batch has all-valid rows and the target at the end.
        dts, mask, quantities = data
        history_quantities = quantities.clone()
        history_quantities[:, -1] = 0.
        write = mask.clone()
        write[:, -1] = False
        return tuple(state[:, -2] for state in model.encode_task_states(
            dts, history_quantities, mask, memory_write_mask=write))

    def nonzero_gradient(parameter):
        gradient = parameter.grad
        common.require(gradient is None or bool(torch.isfinite(gradient).all()),
                       "Task-routed gradient is nonfinite")
        return gradient is not None and bool(torch.count_nonzero(gradient))

    def branch_gradient_check(model, data):
        core = model.task_routed_episode_memory
        results = {"routing": core.routing}
        for task, objective in (("time", "time_loss"), ("quantity", "quantity_train_loss")):
            model.zero_grad(set_to_none=True)
            target_outputs(model, *data, lambda_log_qty=1.)[objective].mean().backward()
            actual = {
                "time_query": nonzero_gradient(core.time_query_projection.weight),
                "quantity_query": nonzero_gradient(core.quantity_query_projection.weight),
                "shared_key_value": all(nonzero_gradient(parameter) for parameter in (
                    core.key_projection.weight, core.value_projection.weight, core.value_projection.bias)),
                "time_output": nonzero_gradient(core.time_output_projection.weight),
                "quantity_output": nonzero_gradient(core.quantity_output_projection.weight),
            }
            expected = {"time_query": task == "time" or core.routing == "shared",
                        "quantity_query": task == "quantity" or core.routing == "shared",
                        "shared_key_value": True, "time_output": task == "time",
                        "quantity_output": task == "quantity"}
            common.require(actual == expected, "Task query/output gradient routing differs: " + task)
            results[task] = actual
        model.zero_grad(set_to_none=True)
        return results

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
    empty_checks, task_route_gradient_checks = {}, {}
    for name in common.ARMS[1:]:
        model = build(name)
        common.require(_exact(initial_rng, rng_state()), "Initial CPU/CUDA RNG differs")
        state = model.state_dict()
        common.require(all(torch.equal(value, state[key]) for key, value in base_state.items()),
                       "Shared initial state differs")
        added = {key: value.clone() for key, value in state.items()
                 if key.startswith("task_routed_episode_memory.") and not key.endswith("routing_code")}
        if added_initial is not None:
            common.require(_exact(added_initial, added), "Shared/split new parameter initialization differs")
        else:
            added_initial = added
        del state, added
        common.require(not torch.equal(model.task_routed_episode_memory.time_query_projection.weight,
                                       model.task_routed_episode_memory.quantity_query_projection.weight),
                       "Time and quantity queries must have distinct initial weights")
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
        activated_outputs = target_outputs(model, *tiny, lambda_log_qty=1.)
        changed = tuple(value.clone() for value in tiny)
        changed[0][:, -1] += 10000
        changed[2][:, -1] += 10000
        with torch.no_grad():
            altered = target_outputs(model, *changed, lambda_log_qty=1.)["pred_qty"]
            states = prediction_states(model, tiny)
            changed_states = prediction_states(model, changed)
            common.require(all(torch.equal(a, b) for a, b in zip(states, changed_states))
                           and torch.equal(model.predict_time_median(states[0]),
                                           model.predict_time_median(changed_states[0])),
                           "Activated time/quantity prediction state reads the future target")
        common.require(torch.equal(activated_outputs["pred_qty"], altered), "Activated future target leakage")
        # Quantity-only loss correctly leaves the time query inactive in split
        # routing. Joint loss must reach both queries and both output paths.
        activated_outputs["joint_loss"].mean().backward()
        common.require(all(param.grad is not None and bool(torch.isfinite(param.grad).all())
                           and bool(torch.count_nonzero(param.grad))
                           for param in model.task_routed_episode_memory.parameters()),
                       "Activated Qt/Qq/K/V/b/Ut/Uq gradient is absent or nonfinite")
        del activated_outputs, altered, changed, states, changed_states
        task_route_gradient_checks[name] = branch_gradient_check(model, tiny)

        # Native kernels must also preserve an empty read when the trained value
        # bias becomes nonzero, for both task reads and output corrections.
        core = model.task_routed_episode_memory
        saved_bias = core.value_projection.bias.detach().clone()
        with torch.no_grad():
            core.value_projection.bias.fill_(.2)
            hidden = torch.zeros(2, 3, 64, device=device)
            features = torch.zeros(2, 3, 2, device=device)
            mask = torch.tensor([[True, False, False], [False, False, False]], device=device)
            read = core.read(hidden, features, mask)
            correction = core(hidden, features, mask)
            common.require(all(bool(torch.isfinite(value).all()) and not bool(torch.count_nonzero(value))
                               for value in (*read, *correction)),
                           "Empty task-routed memory reads differ from exact zero")
            core.value_projection.bias.copy_(saved_bias)
        empty_checks[name] = True
        del core, saved_bias, hidden, features, mask, read, correction

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
            model = build(name, max_seq_len=length).to(device)
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
        rows.append({"batch": batch_size, "length": length, "max_seq_len": length,
                     "measurements": measurements, "updates_per_arm": 7, "measurement_order": list(order)})
        del data
    common.require(updates == 48, "Synthetic qualification update budget differs")
    if device == "cuda:0":
        require_native_owner()
    return {
        "status": "synthetic_checks_passed", "device": device,
        "real_data_loaded": False, "held_out_evaluated": False,
        "shared_initial_state_sha256": digest, "checks": checks,
        "empty_read_checks": empty_checks,
        "task_route_gradient_checks": task_route_gradient_checks,
        "new_arm_initial_parameters_exact": True, "costs": rows,
        "synthetic_optimizer_updates": updates,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "platform": platform.platform(), "threads": torch.get_num_threads()},
        "limitations": [
            "Synthetic timing excludes data loader and validation/checkpoint I/O.",
            "CPU timing is not native CUDA qualification or a full training-hours estimate.",
            "Sequential timings can contain order effects; shape256 reverses the arm order.",
            "B and new arms share activated quantity weights; only new arms have two activated memory projections.",
            "Shared and split query routing have equal parameter counts and major operations; their wall times can differ.",
            "Synthetic finite updates provide no dataset or benchmark performance evidence.",
        ],
        "completed_at_unix": time.time(),
    }


def cost_gates(receipt, contract, runtime):
    """Use the same strict numerical audit as portable start-permit validation."""
    common.require(receipt.get("device") == "cuda:0", "Cost gates require native CUDA allocation measurements")
    return common.verified_cost_gates({**receipt, "runtime": runtime}, contract)


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
