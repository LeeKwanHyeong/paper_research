#!/usr/bin/env python3
"""Deadline-bounded native CUDA qualification using only synthetic tensors."""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import observed_slot_parallel_common as common


def qualify(contract, host):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    runtime = common.runtime_check(contract, host)
    baseline, candidate = contract["model"]["baseline"], contract["model"]["candidate"]
    kwargs = dict(hidden_dim=64, train_log_mean=1.5, max_seq_len=256, time_intercept_limit=300.0)

    def build(name):
        torch.manual_seed(42)
        model, _ = build_count_aware_model(name, **kwargs)
        return model

    def batch(size, length):
        rng = torch.Generator().manual_seed(77 + length)
        return (0.1 + 2 * torch.rand(size, length, generator=rng),
                torch.ones(size, length, dtype=torch.bool),
                torch.randint(0, 32, (size, length), generator=rng).float())

    def objective(model, data):
        return target_outputs(model, *data, lambda_log_qty=1.0)

    b = build(baseline)
    rng_b = torch.get_rng_state().clone()
    c = build(candidate)
    common.require(torch.equal(rng_b, torch.get_rng_state()), "Initial CPU RNG differs")
    base_state = b.state_dict()
    common.require(all(torch.equal(value, c.state_dict()[name]) for name, value in base_state.items()), "Shared initialization differs")
    shared_initial = canonical_state_dict_sha256(base_state)
    b, c = b.cuda(), c.cuda()
    tiny = tuple(t.cuda() for t in batch(4, 8))
    torch.manual_seed(79)
    expected = objective(b, tiny)
    torch.manual_seed(79)
    actual = objective(c, tiny)
    common.require(all(torch.equal(expected[k], actual[k]) for k in expected), "CUDA zero-gate B outputs differ")
    expected["joint_loss"].mean().backward()
    actual["joint_loss"].mean().backward()
    other = dict(c.named_parameters())
    for name, value in b.named_parameters():
        common.require((value.grad is None and other[name].grad is None)
                       or (value.grad is not None and torch.equal(value.grad, other[name].grad)), "CUDA shared gradient differs: " + name)
    del expected, actual, b
    c.zero_grad(set_to_none=True)
    with torch.no_grad():
        c.slot_memory_alpha_raw.fill_(0.25)
        # B deliberately starts with a zero quantity weight. A nonconstant
        # synthetic readout is required to exercise causality and slot gradients.
        c.quantity_head.weight.copy_(torch.linspace(-0.1, 0.1, c.hidden_dim,
                                                    device="cuda").reshape(1, -1))
    c.eval()
    prediction = objective(c, tiny)["pred_qty"]
    changed = tuple(t.clone() for t in tiny)
    changed[0][:, -1] += 10000
    changed[2][:, -1] += 10000
    common.require(torch.equal(prediction, objective(c, changed)["pred_qty"]), "CUDA future target leakage")
    prediction.sum().backward()
    common.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.count_nonzero(p.grad))
                       for p in c.slot_memory.parameters()), "CUDA active slot gradient missing/nonfinite")
    del prediction, changed

    optimizer = build_optimizer(c, lr=0.001, time_head_lr_multiplier=1.0)
    def step(model, optim, data):
        model.train()
        optim.zero_grad(set_to_none=True)
        loss = objective(model, data)["joint_loss"].mean()
        common.require(bool(torch.isfinite(loss)), "Nonfinite CUDA synthetic loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        common.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite CUDA state")

    step(c, optimizer, tiny)
    stream = io.BytesIO()
    torch.save({"model": c.state_dict(), "optimizer": optimizer.state_dict(),
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, stream)
    step(c, optimizer, tiny)
    continuous = {name: value.detach().cpu().clone() for name, value in c.state_dict().items()}
    stream.seek(0)
    saved = torch.load(stream, map_location="cpu", weights_only=False)
    c.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    step(c, optimizer, tiny)
    common.require(all(torch.equal(continuous[name], value.cpu()) for name, value in c.state_dict().items()), "CUDA save/restore replay differs")
    del c, optimizer, tiny, stream, saved, continuous, base_state, other
    gc.collect()
    torch.cuda.empty_cache()
    costs = []
    for length in (64, 256):
        data = tuple(t.cuda() for t in batch(128, length))
        measurements = {}
        for name in (baseline, candidate):
            model = build(name).cuda()
            if name == candidate:
                with torch.no_grad():
                    model.slot_memory_alpha_raw.fill_(0.25)
            optim = build_optimizer(model, lr=0.001, time_head_lr_multiplier=1.0)
            samples = []
            torch.cuda.reset_peak_memory_stats()
            for repeat in range(7):
                torch.manual_seed(110 + repeat)
                torch.cuda.synchronize()
                start = time.perf_counter()
                step(model, optim, data)
                torch.cuda.synchronize()
                if repeat >= 2:
                    samples.append(time.perf_counter() - start)
            measurements[name] = {
                "median_step_seconds": statistics.median(samples), "step_seconds": samples,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
            del model, optim
            gc.collect()
            torch.cuda.empty_cache()
        bcost, ccost = measurements[baseline], measurements[candidate]
        gates = contract["cost_gates"]
        checks = {
            "step": ccost["median_step_seconds"] <= bcost["median_step_seconds"] * gates["step_ratio_max"],
            "relative_memory": ccost["peak_allocated_bytes"] <= bcost["peak_allocated_bytes"] * gates["peak_allocated_ratio_max"],
            "device_memory": ccost["peak_allocated_bytes"] <= runtime["gpu"]["total_memory_bytes"] * gates["peak_device_fraction_max"],
            "parameters": ccost["parameters"] <= bcost["parameters"] * gates["parameter_ratio_max"],
        }
        costs.append({"batch": 128, "length": length, "measurements": measurements, "checks": checks})
        del data
        gc.collect()
        torch.cuda.empty_cache()
    common.verify_source(contract)
    common.require(common.gpu_pids(contract["hosts"][host]) <= {os.getpid()}, "Other GPU workload appeared during qualification")
    return {
        "status": "passed" if all(all(row["checks"].values()) for row in costs) else "cost_gate_failed",
        "host": host, "runtime": runtime, "contract_sha256": common.sha_json(contract),
        "source_files_sha256": contract["source"]["files_sha256"],
        "shared_initial_cpu_state_sha256": shared_initial,
        "zero_gate_cuda_output_and_shared_gradient_exact": True,
        "target_causality_passed": True, "activated_slot_gradient_passed": True,
        "cuda_saved_optimizer_rng_next_step_exact": True,
        "synthetic_optimizer_updates": 31, "real_data_loaded": False,
        "held_out_evaluated": False, "costs": costs,
        "completed_at_unix": time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    contract = common.read_contract(args.contract, args.host)
    spec = contract["hosts"][args.host]
    common.require(ROOT == Path(spec["source_root"]).resolve() and Path.cwd().resolve() == ROOT, "Pinned source directory required")
    common.apply_environment(spec)
    output = Path(spec["root"]) / "qualification"
    if args.child:
        common.require(os.environ.get("SLOT_QUALIFICATION_OWNER_PID") == str(os.getppid()), "Qualification requires its supervising parent")
        common.require(not common.gpu_pids(spec), "GPU is busy; existing work is preserved")
        receipt = qualify(contract, args.host)
        common.write_json(output / "receipt.json", receipt, exclusive=True)
        raise SystemExit(0 if receipt["status"] == "passed" else 2)
    common.require(not common.gpu_pids(spec), "GPU is busy; existing work is preserved")
    output.mkdir(exist_ok=False)
    environment = {**os.environ, "SLOT_QUALIFICATION_OWNER_PID": str(os.getpid())}
    command = [sys.executable, str(Path(__file__).resolve()), "--contract", str(args.contract.resolve()), "--host", args.host, "--child"]
    with (output / "qualification.log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        try:
            code = child.wait(timeout=contract["limits"]["qualification_seconds"])
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            common.write_json(output / "failure.json", {"status": "timeout", "owned_pid": child.pid}, exclusive=True)
            raise SystemExit(124)
    if code != 0:
        common.write_json(output / "failure.json", {"status": "qualification_failed", "returncode": code}, exclusive=True)
    print(json.dumps({"host": args.host, "returncode": code, "output": str(output)}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
