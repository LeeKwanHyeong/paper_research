#!/usr/bin/env python3
"""Synthetic multi-lag correctness and training-cost qualification.

No dataset or checkpoint is read. CPU measurements are local observations, never
CUDA qualification or epoch-time predictions. Native CUDA execution is an
explicit caller choice; the execution runner owns approval/host/deadline checks.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("titantpp", "titantpp_local_detail", "titantpp_multilag_detail")
QUALIFICATION_CHECKS = (
    "shared_initial_state_rng", "initial_output_objective_common_gradient",
    "local_candidate_initial_tensors", "staged_all_branch_gradients",
    "prefix_target_padding_causality", "pmf_boundary_tail_topcode_gradients",
    "model_optimizer_rng_exact_resume", "projected_reference_forward_backward", "finite_native_updates",
)
DEFAULT_COST_GATES = {
    "parameter_ratio_max": 1.10,
    "median_cuda_step_ratio_max": 1.50,
    "peak_cuda_allocated_ratio_max": 1.25,
    "device_memory_fraction_max": .80,
}
ADDED_PREFIX = "multilag_detail."


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _same(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        data = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(data.dtype).encode())
        digest.update(str(tuple(data.shape)).encode())
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def _validate_options(device, batch_size, lengths, warmup_steps, measured_steps, max_wall_seconds, cost_gates):
    _require(device in ("cpu", "cuda", "cuda:0"), "Only explicit CPU or native cuda:0 is supported")
    _require(type(batch_size) is int and 1 <= batch_size <= 128, "batch_size must be 1..128")
    _require(isinstance(lengths, (list, tuple)) and 1 <= len(lengths) <= 2
             and len(set(lengths)) == len(lengths)
             and all(type(x) is int and 2 <= x <= 256 for x in lengths), "Invalid distinct sequence lengths")
    _require(type(warmup_steps) is int and 1 <= warmup_steps <= 5, "warmup_steps must be 1..5")
    _require(type(measured_steps) is int and 1 <= measured_steps <= 10, "measured_steps must be 1..10")
    _require(type(max_wall_seconds) in (int, float) and math.isfinite(max_wall_seconds)
             and 0 < max_wall_seconds <= 1800, "max_wall_seconds must be finite in (0,1800]")
    _require(set(cost_gates) == set(DEFAULT_COST_GATES), "Cost gate identity changed")
    _require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in cost_gates.values()),
             "Cost gates must be finite and positive")
    _require(cost_gates["device_memory_fraction_max"] <= 1, "Invalid device memory fraction")


def _cost_checks(measurements, gates, *, device_memory=None):
    base = measurements[ARMS[0]]
    checks = {}
    for arm in ARMS[1:]:
        value = measurements[arm]
        checks[arm + ":parameters"] = value["parameters"] <= base["parameters"] * gates["parameter_ratio_max"]
        if device_memory is not None:
            checks[arm + ":step"] = value["median_step_seconds"] <= base["median_step_seconds"] * gates["median_cuda_step_ratio_max"]
            checks[arm + ":memory"] = value["peak_allocated_bytes"] <= base["peak_allocated_bytes"] * gates["peak_cuda_allocated_ratio_max"]
            checks[arm + ":device"] = value["peak_allocated_bytes"] <= device_memory * gates["device_memory_fraction_max"]
    return checks


def _direct_reference_check(module, device):
    """Compare the production gather against independently indexed raw pairs."""
    import torch
    from torch.nn import functional as F
    length = 137
    valid_cpu = torch.ones(2, length, dtype=torch.bool)
    write_cpu = valid_cpu.clone()
    valid_cpu[0, [3, 19, 42]] = False
    write_cpu[1, [4, 100]] = False
    valid, write = valid_cpu.to(device), write_cpu.to(device)
    hidden = torch.linspace(-1., 1., 2 * length * 64, device=device).reshape(2, length, 64).requires_grad_()
    source = torch.zeros(2, length, 8, dtype=torch.long)
    available = torch.zeros(2, length, 8, dtype=torch.bool)
    for row in range(2):
        segment = []
        for pos in range(length):
            if not valid_cpu[row, pos]:
                continue
            if not write_cpu[row, pos]:
                segment = []
                continue
            for branch, lag in enumerate((1, 2, 4, 8, 16, 32, 64, 128)):
                if len(segment) >= lag:
                    available[row, pos, branch] = True
                    source[row, pos, branch] = segment[-lag if module.mode == "multilag" else -1]
            segment.append(pos)
    source, available = source.to(device), available.to(device)
    safe = torch.where((valid & write).unsqueeze(-1), hidden, 0.)
    reference = torch.zeros_like(safe)
    for branch, (level, detail, output) in enumerate(zip(
            module.level_projections, module.detail_projections, module.output_projections)):
        past = safe.gather(1, source[..., branch].unsqueeze(-1).expand(-1, -1, 64))
        active = available[..., branch].unsqueeze(-1)
        mean = torch.where(active, (safe + past) * .5, 0.)
        difference = torch.where(active, safe - past, 0.)
        correction = output(F.gelu(level(mean), approximate="none") + F.gelu(detail(difference), approximate="none"))
        reference = reference + torch.where(active, correction, 0.)
    reference = reference / 8
    actual = module(hidden, valid, memory_write_mask=write)
    _require(torch.allclose(actual, reference, rtol=2e-5, atol=2e-7), "Projected-gather forward differs from raw-pair oracle")
    weights = torch.linspace(-.8, 1.2, actual.numel(), device=device).reshape_as(actual)
    inputs = (hidden, *module.parameters())
    actual_grads = torch.autograd.grad((actual * weights).sum(), inputs)
    reference_grads = torch.autograd.grad((reference * weights).sum(), inputs)
    _require(all(torch.allclose(a, b, rtol=2e-5, atol=2e-6) for a, b in zip(actual_grads, reference_grads)),
             "Projected-gather backward differs from raw-pair oracle")


def run(output=None, *, device="cpu", batch_size=128, lengths=(64, 256),
        warmup_steps=2, measured_steps=5, max_wall_seconds=600., budget=None, cost_gates=None):
    """Run synthetic checks; raise on correctness errors and preserve no input state.

    6 restoration optimizer updates plus 3*len(lengths)*(warmup+measured)
    measured-workload updates. Standard CUDA workload therefore performs 48.
    An external runner must provide its stricter shared-deadline ``budget``.
    """
    gates = dict(DEFAULT_COST_GATES if cost_gates is None else cost_gates)
    _validate_options(device, batch_size, lengths, warmup_steps, measured_steps, max_wall_seconds, gates)
    output = None if output is None else Path(output)
    _require(output is None or not output.exists(), "Refusing to overwrite a qualification receipt")
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    verifier_source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def guard():
        _require(time.monotonic() - started < max_wall_seconds, "Synthetic qualification time limit exceeded")
        if budget is not None:
            budget()

    guard()
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from models.TPPs.positive_integer_time import positive_integer_log_mass
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    native_cuda = device != "cpu"
    if native_cuda:
        _require(torch.cuda.is_available(), "Native CUDA requested but not available; no CPU fallback")
        torch.cuda.set_device(0)
        device = "cuda:0"
    checks = {name: False for name in QUALIFICATION_CHECKS}
    updates = 0
    original_threads = torch.get_num_threads()
    if not native_cuda:
        torch.set_num_threads(min(4, original_threads))
    saved_cpu_rng = torch.get_rng_state().clone()
    saved_cuda_rng = torch.cuda.get_rng_state(0).clone() if native_cuda else None

    def seed(value):
        torch.random.default_generator.manual_seed(value)
        if native_cuda:
            torch.cuda.manual_seed(value)

    def rng():
        return {"cpu": torch.get_rng_state().clone(),
                "cuda": torch.cuda.get_rng_state(0).clone() if native_cuda else None}

    def restore_rng(state):
        torch.set_rng_state(state["cpu"])
        if native_cuda:
            torch.cuda.set_rng_state(state["cuda"], 0)

    kwargs = dict(hidden_dim=64, train_log_mean=1.5, max_seq_len=256,
        time_head_mode="heteroscedastic_lognormal_duration", time_scale=3.,
        time_initial_location=.1, time_initial_scale=.7,
        time_observation_contract={"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None})

    def build(arm):
        guard(); seed(42)
        model, _ = build_count_aware_model(arm, **kwargs)
        _require(all(p.device.type == "cpu" for p in model.parameters()), "Factory must initialize on CPU")
        return model

    def batch(size, length):
        generator = torch.Generator(device="cpu").manual_seed(77 + length)
        values = (torch.randint(1, 31, (size, length), generator=generator).float(),
                  torch.ones(size, length, dtype=torch.bool),
                  torch.randint(0, 64, (size, length), generator=generator).float())
        return tuple(t.to(device) for t in values)

    def objective(model, data):
        return target_outputs(model, *data, lambda_log_qty=1.)

    def added_parameters(model):
        return {n: p for n, p in model.named_parameters() if n.startswith(ADDED_PREFIX)}

    def activate(model):
        with torch.no_grad():
            model.quantity_head.weight.copy_(torch.linspace(-.1, .1, 64, device=device).reshape(1, 64))
            for name, parameter in added_parameters(model).items():
                if ".output_projections." in name:
                    parameter.copy_(torch.linspace(-.05, .05, parameter.numel(), device=device).reshape_as(parameter))

    def step(model, optimizer, data):
        nonlocal updates
        guard(); model.train(); optimizer.zero_grad(set_to_none=True)
        loss = objective(model, data)["joint_loss"].mean()
        _require(bool(torch.isfinite(loss)), "Nonfinite synthetic joint objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); updates += 1
        _require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite updated parameter")
        guard()
        return float(loss.detach()), float(norm.detach())

    try:
        # Probability masses are checked at the low boundary, far tails and the
        # Instacart top code independently of any learned head initialization.
        for top_code, targets in ((None, [1., 2., 30., 1000.]), (30, [1., 2., 29., 30.])):
            guard()
            mu = torch.tensor([-80., 80., 0., 2.], device=device, requires_grad=True)
            sigma = torch.ones(4, device=device, requires_grad=True)
            values = positive_integer_log_mass(torch.tensor(targets, device=device), mu, sigma,
                                              time_scale=3., top_code=top_code)
            (-values.sum()).backward()
            _require(bool(torch.isfinite(values).all()) and bool(torch.isfinite(mu.grad).all())
                     and bool(torch.isfinite(sigma.grad).all()), "PMF boundary/tail gradient failed")
        checks["pmf_boundary_tail_topcode_gradients"] = True
        del values, mu, sigma
        tiny = batch(2, 130)  # At prediction index128 all eight lag branches exist.
        base = build(ARMS[0]); base_rng = rng()
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        initial_sha = _tensor_digest(base_state.items())
        base.to(device)
        local_initial_added = None
        gradient_receipts = {}
        for arm in ARMS[1:]:
            candidate = build(arm)
            _require(_same(base_rng, rng()) and all(torch.equal(v, candidate.state_dict()[k]) for k, v in base_state.items()),
                     "Common initial tensors or caller RNG differ")
            added = {n: p.detach().clone() for n, p in added_parameters(candidate).items()}
            _require(bool(added) and sum(p.numel() for p in added.values()) == 6144,
                     "Expected exactly 6144 added level/detail/output parameters")
            if local_initial_added is None:
                local_initial_added = added
            else:
                _require(_same(local_initial_added, added), "Local/candidate initial tensors differ")
            candidate.to(device)
            for training in (False, True):
                base.train(training); candidate.train(training)
                base.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
                seed(79); expected = objective(base, tiny)
                expected["joint_loss"].mean().backward()
                seed(79); observed = objective(candidate, tiny)
                _require(_same(expected, observed), "Initial output or objective differs from B")
                observed["joint_loss"].mean().backward()
                params = dict(candidate.named_parameters())
                _require(all(_same(p.grad, params[n].grad) for n, p in base.named_parameters()),
                         "Initial common preclip gradient differs")
            initial_gradient_nonzero = {
                n: p.grad is not None and bool(torch.count_nonzero(p.grad))
                for n, p in added_parameters(candidate).items()
            }
            _require(all(initial_gradient_nonzero[n] for n in initial_gradient_nonzero if ".output_projections." in n),
                     "Zero-output initialization lost an active output gradient")
            candidate.zero_grad(set_to_none=True); activate(candidate); candidate.eval()
            _direct_reference_check(candidate.multilag_detail, device)
            objective(candidate, tiny)["joint_loss"].mean().backward()
            active_gradient_nonzero = {
                n: p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.count_nonzero(p.grad))
                for n, p in added_parameters(candidate).items()
            }
            _require(all(active_gradient_nonzero.values()), "Activated L/D/U parameter gradient absent or nonfinite")
            gradient_receipts[arm] = {"initial_nonzero": initial_gradient_nonzero,
                                     "activated_finite_nonzero": active_gradient_nonzero}
            candidate.zero_grad(set_to_none=True)
            with torch.no_grad():
                dts, mask, quantities = tiny
                write = mask.clone(); write[:, -1] = False
                tq = quantities.clone(); tq[:, -1] = 0
                original = candidate.encode_task_states(dts, tq, mask, memory_write_mask=write)
                changed = tuple(v.clone() for v in tiny)
                changed[0][:, -1] += 10000; changed[2][:, -1] += 10000
                changed_tq = changed[2].clone(); changed_tq[:, -1] = 0
                future = candidate.encode_task_states(changed[0], changed_tq, mask, memory_write_mask=write)
                _require(_same(original, future), "Withheld target leaked into predictor states")
                orig_result, changed_result = objective(candidate, tiny), objective(candidate, changed)
                _require(torch.equal(orig_result["pred_qty"], changed_result["pred_qty"])
                         and not torch.equal(orig_result["joint_loss"], changed_result["joint_loss"]),
                         "Target isolation must preserve prediction while changing supervision")
                # Same-shape future perturbation avoids changing position IDs.
                later_dts, later_q = dts.clone(), quantities.clone()
                later_dts[:, 65:] += 10; later_q[:, 65:] += 100
                full = candidate.encode_task_states(dts, quantities, mask)
                later = candidate.encode_task_states(later_dts, later_q, mask)
                _require(all(torch.equal(x[:, :65], y[:, :65]) for x, y in zip(full, later)), "Future suffix leaked into prefix")
                padded_mask = mask.clone(); padded_mask[:, 120:] = False
                clean_dts, clean_q = dts.clone(), quantities.clone()
                clean_dts[:, 120:] = 0; clean_q[:, 120:] = 0
                poison_dts, poison_q = clean_dts.clone(), clean_q.clone()
                poison_dts[:, 120:] = float("nan"); poison_q[:, 120:] = float("inf")
                clean = candidate.encode_task_states(clean_dts, clean_q, padded_mask)
                poisoned = candidate.encode_task_states(poison_dts, poison_q, padded_mask)
                _require(_same(clean, poisoned) and all(bool(torch.isfinite(x).all()) for x in poisoned),
                         "Padding poison changed finite predictor states")
                _require(torch.equal(candidate.time_location(original[0][:, -2]), candidate.time_location(future[0][:, -2]))
                         and torch.equal(candidate.positive_time_sigma(original[0][:, -2]), candidate.positive_time_sigma(future[0][:, -2])),
                         "Target changed predicted time distribution")
            optimizer = torch.optim.AdamW(candidate.parameters(), lr=.001, weight_decay=.01)
            step(candidate, optimizer, tiny)
            stream = io.BytesIO()
            torch.save({"model": candidate.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng()}, stream)
            step(candidate, optimizer, tiny)
            expected_state = copy.deepcopy(candidate.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict()); expected_rng = rng()
            stream.seek(0); saved = torch.load(stream, map_location="cpu", weights_only=False)
            candidate.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); restore_rng(saved["rng"])
            step(candidate, optimizer, tiny)
            _require(_same(expected_state, candidate.state_dict()) and _same(expected_optimizer, optimizer.state_dict())
                     and _same(expected_rng, rng()), "Model/optimizer/RNG interrupted-next-update replay differs")
            del candidate, optimizer, stream, saved, expected_state, expected_optimizer, expected_rng
            del original, future, full, later, clean, poisoned, params, expected, observed, orig_result, changed_result
        for name in QUALIFICATION_CHECKS[:-1]:
            checks[name] = True
        del base, base_state, tiny, local_initial_added
        gc.collect()
        if native_cuda:
            torch.cuda.empty_cache()
        device_memory = torch.cuda.get_device_properties(0).total_memory if native_cuda else None
        costs = []
        for length in lengths:
            data = batch(batch_size, length)
            batch_sha = _tensor_digest(zip(("dts", "mask", "quantities"), data))
            measurements = {}
            for arm in ARMS:
                model = build(arm).to(device); activate(model)
                optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                times, losses, norms = [], [], []
                if native_cuda:
                    torch.cuda.reset_peak_memory_stats(0)
                for index in range(warmup_steps + measured_steps):
                    seed(110 + index)
                    if native_cuda:
                        torch.cuda.synchronize(0)
                    tick = time.perf_counter()
                    loss, norm = step(model, optimizer, data)
                    if native_cuda:
                        torch.cuda.synchronize(0)
                    elapsed = time.perf_counter() - tick
                    if index >= warmup_steps:
                        times.append(elapsed); losses.append(loss); norms.append(norm)
                _require(all(math.isfinite(t) and t > 0 for t in times), "Invalid measured step time")
                measurements[arm] = {"step_seconds": times, "median_step_seconds": statistics.median(times),
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)) if native_cuda else None,
                    "joint_losses": losses, "preclip_gradient_norms": norms}
                del model, optimizer
                gc.collect()
                if native_cuda:
                    torch.cuda.empty_cache()
            _require(measurements[ARMS[1]]["parameters"] == measurements[ARMS[2]]["parameters"], "Control/candidate capacities differ")
            costs.append({"length": length, "batch_size": batch_size, "synthetic_batch_sha256": batch_sha,
                          "measurements": measurements, "checks": _cost_checks(measurements, gates, device_memory=device_memory)})
            del data
            gc.collect()
            if native_cuda:
                torch.cuda.empty_cache()
        _require(updates == 6 + len(lengths) * 3 * (warmup_steps + measured_steps), "Synthetic optimizer update accounting mismatch")
        checks["finite_native_updates"] = True
        all_cost_pass = all(all(row["checks"].values()) for row in costs)
        # A CPU success is deliberately impossible to mistake for a native permit.
        status = ("passed" if native_cuda else "cpu_observation_complete") if all_cost_pass else "cost_gate_failed"
        rss_factor = 1 if sys.platform == "darwin" else 1024
        result = {"schema": "multilag_detail_synthetic_qualification_v1", "status": status,
            "verifier_source_sha256": verifier_source_sha,
            "device": device, "cuda_qualification": native_cuda, "checks": checks, "costs": costs,
            "synthetic_optimizer_updates": updates, "shared_initial_state_sha256": initial_sha,
            "real_data_loaded": False, "held_out_evaluated": False, "cost_gates": gates,
            "warmup_steps": warmup_steps, "measured_steps": measured_steps,
            "workload": {"arms": list(ARMS), "batch_size": batch_size, "lengths": list(lengths),
                         "correctness_batch_size": 2, "correctness_sequence_length": 130,
                         "hidden_dim": 64, "rank": 4, "branch_count": 8, "model_max_seq_len": 256,
                         "optimizer": "AdamW", "learning_rate": .001, "weight_decay": .01,
                         "gradient_clip_norm": 1., "lambda_log_qty": 1.,
                         "objective": "recorded_positive_integer_time_nll_plus_log_quantity_mse",
                         "branch_activation": "identical_nonzero_quantity_weights_and_nonzero_added_output_weights",
                         "timing_scope": "forward_backward_clip_optimizer_finite_checks_synchronized_if_cuda"},
            "branch_gradient_checks": gradient_receipts,
            "started_at_utc": started_at, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_wall_seconds": time.monotonic() - started,
            "runtime": {"python": platform.python_version(), "torch": torch.__version__, "device": device,
                        "platform": platform.platform(), "threads": torch.get_num_threads(),
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()},
            "process_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_factor),
            "limitations": ["Synthetic training steps exclude real data loading, validation, checkpoint writing and epoch duration.",
                            "The process peak RSS is cumulative across cases and cannot serve as a per-arm memory gate.",
                            "CPU step times do not estimate GPU training or qualify native CUDA.",
                            "Checks do not establish prediction quality or multi-seed generalization."]}
        if native_cuda:
            properties = torch.cuda.get_device_properties(0)
            result["runtime"]["gpu"] = {"name": properties.name, "total_memory_bytes": device_memory,
                                          "capability": list(torch.cuda.get_device_capability(0))}
        guard()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write("\n")
        return result
    finally:
        torch.set_rng_state(saved_cpu_rng)
        if native_cuda:
            torch.cuda.set_rng_state(saved_cuda_rng, 0)
        if not native_cuda:
            torch.set_num_threads(original_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lengths", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    parser.add_argument("--max-wall-seconds", type=float, default=600.)
    args = parser.parse_args()
    result = run(args.output, device=args.device, batch_size=args.batch_size, lengths=args.lengths,
                 warmup_steps=args.warmup_steps, measured_steps=args.measured_steps, max_wall_seconds=args.max_wall_seconds)
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()),
                      "total_wall_seconds": result["total_wall_seconds"]}))
    return 0 if result["status"] in ("passed", "cpu_observation_complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
