#!/usr/bin/env python3
"""Synthetic training cost and device qualification; CPU default, CUDA approval-bound."""
from __future__ import annotations

import argparse
import gc
import io
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import successor_episode_common as common


def verify(device="cpu", *, batch_size=128):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.count_aware_tpp_backbone.training import build_optimizer
    from paper.scripts.verify_observed_slot_memory_cpu import _step
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    def build(name):
        torch.manual_seed(42)
        return build_count_aware_model(name, hidden_dim=64, train_log_mean=1.5, max_seq_len=256,
                                      time_intercept_limit=300.)[0]

    def batch(size, length):
        rng = torch.Generator().manual_seed(400 + length)
        return tuple(value.to(device) for value in (
            .1 + 2 * torch.rand(size, length, generator=rng),
            torch.ones(size, length, dtype=torch.bool),
            torch.randint(0, 32, (size, length), generator=rng).float()))

    def sync():
        if device != "cpu":
            torch.cuda.synchronize()

    def activate(model):
        with torch.no_grad():
            model.episode_memory.output_projection.weight.copy_(
                torch.linspace(-.05, .05, model.hidden_dim * 2, device=device).reshape(model.hidden_dim, 2))
            model.quantity_head.weight.copy_(torch.linspace(-.1, .1, model.hidden_dim, device=device).reshape(1, -1))

    baseline = build(common.ARMS[0])
    initial_rng = torch.get_rng_state().clone()
    base_state = {name: value.clone() for name, value in baseline.state_dict().items()}
    digest = canonical_state_dict_sha256(base_state)
    baseline.to(device).train()
    tiny = batch(4, 8)
    torch.manual_seed(79)
    expected = target_outputs(baseline, *tiny, lambda_log_qty=1.)
    expected["joint_loss"].mean().backward()
    checks = {}
    for name in common.ARMS[1:]:
        model = build(name)
        common.require(torch.equal(initial_rng, torch.get_rng_state()), "Initial CPU RNG differs")
        common.require(all(torch.equal(value, model.state_dict()[key]) for key, value in base_state.items()), "Shared initial state differs")
        model.to(device).train()
        torch.manual_seed(79)
        actual = target_outputs(model, *tiny, lambda_log_qty=1.)
        common.require(all(torch.equal(expected[key], actual[key]) for key in expected), "Initial device outputs/objective differ")
        actual["joint_loss"].mean().backward()
        parameters = dict(model.named_parameters())
        for key, value in baseline.named_parameters():
            other = parameters[key].grad
            common.require((value.grad is None and other is None) or
                           (value.grad is not None and other is not None and torch.equal(value.grad, other)), "Initial shared preclip gradient differs: " + key)
        del actual, parameters
        model.zero_grad(set_to_none=True)
        activate(model)
        model.eval()
        prediction = target_outputs(model, *tiny, lambda_log_qty=1.)["pred_qty"]
        changed = tuple(value.clone() for value in tiny)
        changed[0][:, -1] += 10000
        changed[2][:, -1] += 10000
        common.require(torch.equal(prediction, target_outputs(model, *changed, lambda_log_qty=1.)["pred_qty"]), "Activated future target leakage")
        prediction.sum().backward()
        common.require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(torch.count_nonzero(p.grad))
                           for p in model.episode_memory.parameters()), "Activated Q/K/U gradient is absent/nonfinite")
        del prediction, changed
        optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
        _step(model, optimizer, tiny)
        stream = io.BytesIO()
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state()}
        if device != "cpu":
            saved["cuda_rng"] = torch.cuda.get_rng_state()
        torch.save(saved, stream)
        _step(model, optimizer, tiny)
        continued = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stream.seek(0)
        saved = torch.load(stream, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["cpu_rng"])
        if device != "cpu":
            torch.cuda.set_rng_state(saved["cuda_rng"])
        _step(model, optimizer, tiny)
        common.require(all(torch.equal(continued[key], value.cpu()) for key, value in model.state_dict().items()), "Restored next optimizer step differs")
        checks[name] = {"initial_output_objective_shared_preclip_gradient_exact": True,
                       "activated_target_prediction_causal": True, "activated_Q_K_U_gradients": True,
                       "saved_optimizer_rng_next_step_exact": True}
        del model, optimizer, saved, stream, continued
    del baseline, expected, base_state, tiny
    gc.collect()
    if device != "cpu":
        torch.cuda.empty_cache()
    rows = []
    for length in (64, 256):
        data = batch(batch_size, length)
        measures = {}
        for name in common.ARMS:
            model = build(name).to(device)
            if name != common.ARMS[0]:
                activate(model)
            optimizer = build_optimizer(model, lr=.001, time_head_lr_multiplier=1.)
            samples = []
            if device != "cpu":
                torch.cuda.reset_peak_memory_stats()
            for repeat in range(7):
                torch.manual_seed(110 + repeat)
                sync()
                started = time.perf_counter()
                _step(model, optimizer, data)
                sync()
                if repeat >= 2:
                    samples.append(time.perf_counter() - started)
            measures[name] = {"parameters": sum(p.numel() for p in model.parameters()),
                              "step_seconds": samples, "median_step_seconds": statistics.median(samples),
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device != "cpu" else None}
            del model, optimizer
            gc.collect()
            if device != "cpu":
                torch.cuda.empty_cache()
        for name in common.ARMS:
            measures[name]["step_ratio_to_B"] = measures[name]["median_step_seconds"] / measures[common.ARMS[0]]["median_step_seconds"]
        rows.append({"batch": batch_size, "length": length, "measurements": measures, "updates_per_arm": 7})
        del data
    return {"status": "synthetic_checks_passed", "device": device, "real_data_loaded": False, "held_out_evaluated": False,
            "shared_initial_state_sha256": digest, "checks": checks, "costs": rows,
            "synthetic_optimizer_updates": 48, "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                "platform": platform.platform(), "threads": torch.get_num_threads()},
            "limitations": ["Synthetic timing excludes data loader and validation/checkpoint I/O",
                "CPU timing is not a CUDA or full training-time estimate", "Sequential arm timing can contain order effects",
                "Both added routes were activated for cost checks; no dataset performance evidence"],
            "completed_at_unix": time.time()}


def cost_gates(receipt, contract, runtime):
    checks = []
    for row in receipt["costs"]:
        baseline = row["measurements"][common.ARMS[0]]
        for name in common.ARMS[1:]:
            value, gates = row["measurements"][name], contract["cost_gates"]
            checks.append({"backbone": name, "length": row["length"],
                "step": value["median_step_seconds"] <= baseline["median_step_seconds"] * gates["step_ratio_max"],
                "relative_memory": value["peak_allocated_bytes"] <= baseline["peak_allocated_bytes"] * gates["peak_allocated_ratio_max"],
                "device_memory": value["peak_allocated_bytes"] <= runtime["gpu"]["total_memory_bytes"] * gates["peak_device_fraction_max"],
                "parameters": value["parameters"] <= baseline["parameters"] * gates["parameter_ratio_max"]})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device == "cpu":
        common.require(args.output is not None and not args.output.exists() and not args.child, "Fresh CPU output required")
        import torch
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                receipt = verify()
        finally:
            torch.set_num_threads(threads)
        receipt["source_files_sha256"] = common.sha_json(common.source_manifest())
        common.write_json(args.output, receipt, exclusive=True)
        print(json.dumps({"status": receipt["status"], "output": str(args.output)}))
        return
    common.require(args.contract is not None and args.approval is not None and args.output is None, "CUDA requires contract and approval")
    contract = common.read_contract(args.contract)
    approval = json.loads(args.approval.read_text())
    common.verify_approval(contract, approval)
    host = contract["hosts"]["5090"]
    common.require(ROOT == Path(host["source_root"]).resolve() and Path.cwd().resolve() == ROOT
                   and Path(sys.executable).resolve() == Path(host["python"]).resolve(), "Pinned 5090 source/Python required")
    common.apply_environment(host)
    output = Path(host["root"]) / "qualification"
    if args.child:
        common.require(os.environ.get("EPISODE_QUALIFICATION_OWNER_PID") == str(os.getppid()), "Qualification requires its parent")
        common.require(not common.gpu_pids(host), "GPU busy; existing work preserved")
        runtime = common.runtime_check(contract, "5090")
        receipt = verify("cuda:0")
        gates = cost_gates(receipt, contract, runtime)
        receipt.update(status="passed" if all(all(row[key] for key in ("step", "relative_memory", "device_memory", "parameters")) for row in gates) else "cost_gate_failed",
                       runtime=runtime, host="5090", contract_sha256=common.sha_json(contract), cost_gate_checks=gates,
                       source_files_sha256=contract["source"]["files_sha256"],
                       started_at_unix=float(os.environ["EPISODE_QUALIFICATION_STARTED"]))
        common.verify_source(contract)
        common.require(common.gpu_pids(host) <= {os.getpid()}, "Another GPU job appeared")
        common.write_json(output / "receipt.json", receipt, exclusive=True)
        raise SystemExit(0 if receipt["status"] == "passed" else 2)
    common.require(not common.gpu_pids(host), "GPU busy; existing work preserved")
    output.mkdir(exist_ok=False)
    environment = {**os.environ, "EPISODE_QUALIFICATION_OWNER_PID": str(os.getpid()), "EPISODE_QUALIFICATION_STARTED": str(time.time())}
    command = [sys.executable, str(Path(__file__).resolve()), "--device", "cuda:0", "--contract", str(args.contract.resolve()),
               "--approval", str(args.approval.resolve()), "--child"]
    from paper.scripts.run_observed_slot_partition import kill_owned_process_group, _supervisor_terminated
    child = None
    old_signal = signal.signal(signal.SIGTERM, _supervisor_terminated)
    try:
        with (output / "qualification.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
            code = child.wait(timeout=contract["limits"]["qualification_seconds"])
            common.require(code == 0, f"Native qualification failed: {code}")
    except BaseException as error:
        if child is not None:
            kill_owned_process_group(child)
        common.write_json(output / "failure.json", {"status": "qualification_stopped", "error": str(error), "retry": False}, exclusive=True)
        raise
    finally:
        signal.signal(signal.SIGTERM, old_signal)


if __name__ == "__main__":
    main()
