"""Native synthetic admission helper. Importing executes no GPU or data work.

The separate controller records actual native-child exits and three simultaneous
GPU owners. Synthetic updates are admission only, not scientific fits or timing
evidence. No Test materialization, checkpoint or shared Runtime writes occur.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback


def native_qualification(c, host, root, deadline, *, campaign):
    runtime = campaign.common.runtime_check(c, host)
    campaign.require(runtime["python"] == "3.12.13", "Pinned native Python version changed")
    budget = campaign.PulseBudget(root, campaign.sha_json(c), "native_qualification", deadline)
    result = {"schema": "titantpp_component_native_qualification_v1", "status": "checking",
              "host": host, "runtime": runtime, "contract_sha256": campaign.sha_json(c),
              "source_files_sha256": c["source"]["files_sha256"], "inputs": {}, "initialization": {},
              "synthetic_optimizer_updates": 0, "scientific_optimizer_updates": 0,
              "held_out_test_evaluated": False, "real_Validation_predictions": False,
              "timing_is_scientific_efficiency_evidence": False,
              "pid": os.getpid(), "ppid": os.getppid(), "argv": [sys.executable, *sys.argv]}
    # This preflight opens only admitted Train/Validation frame materialization.
    # Metadata may retain an inherited opaque held-out row count, never outcomes.
    for data in c["datasets"]:
        budget()
        frame, receipt = campaign.base.prepare_admitted_data(data, data_root=campaign.ROOT)
        campaign.require(receipt["held_out_materialized"] is False, "Held-out data was materialized")
        data_id = data["dataset_id"]
        result["inputs"][data_id] = {"input_receipt": receipt,
            "input_receipt_sha256": campaign.sha_json(receipt),
            "interface_sha256": {arm: campaign.sha_json(campaign.runner.time_interface(data, frame, c, arm=arm))
                                  for arm in campaign.ARMS}}
        result["initialization"][data_id] = {str(seed): campaign.initial_states(data, seed) for seed in campaign.SEEDS}
        del frame
        gc.collect()
    result["synthetic"] = {}
    for seed in campaign.SEEDS:
        budget()
        check = campaign.runner.synthetic_check("cuda:0", arms=campaign.ARMS, seed=seed,
                                                include_causality=False)
        campaign.require(check["status"] == "passed" and check["seed"] == seed
                         and check["held_out_test_evaluated"] is False and check["real_data_loaded"] is False,
                         "Native three-seed synthetic qualification drift")
        result["synthetic"][str(seed)] = check
        result["synthetic_optimizer_updates"] += check["synthetic_optimizer_updates"]
    budget()
    result["causality_checks"] = campaign.runner.check_synthetic_causality("cuda:0", arms=campaign.ARMS)
    campaign.require(result["causality_checks"]["status"] == "passed", "Native causal check failed")
    result.update(status="passed", completed_unix=time.time())
    campaign.write_json(root / "qualification" / "native_receipt.json", result, exclusive=True)
    return result


def capacity_worker(c, host, root, arm, deadline, *, campaign):
    campaign.require(arm in (campaign.FULL, campaign.CNN, campaign.GRU), "Foreign synthetic admission arm")
    runtime = campaign.common.runtime_check(c, host)
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    torch.cuda.set_per_process_memory_fraction(.30, 0)
    torch.manual_seed(42)
    model, _ = campaign.runner.build_model(campaign.runner.fabricated_data(), arm)
    model.cuda().train()
    length = 256
    dt = torch.ones((128, length), device="cuda:0")
    qty = torch.arange(length, device="cuda:0").remainder(17).add(1).float().expand(128, -1)
    mask = torch.ones_like(dt, dtype=torch.bool)
    mask[:32, :85] = False
    folder = root / "concurrency_admission" / arm
    ready = {"arm": arm, "pid": os.getpid(), "ppid": os.getppid(), "pgrp": os.getpgrp(),
             "argv": [sys.executable, *sys.argv], "runtime": runtime, "shape": [128, 256],
             "contract_sha256": campaign.sha_json(c), "ready_unix": time.time()}
    campaign.write_json(folder / "ready.json", ready, exclusive=True)
    barrier = root / "concurrency_admission" / "begin.json"
    while not barrier.exists():
        campaign.require(time.time() < deadline, "Capacity barrier deadline expired")
        time.sleep(.05)
    gate = campaign.read(barrier)
    campaign.require(gate["contract_sha256"] == campaign.sha_json(c)
                     and gate["deadline_unix"] == deadline, "Foreign capacity barrier")
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01)
    torch.cuda.reset_peak_memory_stats()
    timings = []
    for step in range(25):
        campaign.require(time.time() < deadline, "Capacity worker deadline expired")
        torch.cuda.synchronize()
        begin = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
        loss = output["joint_loss"].mean()
        campaign.require(bool(torch.isfinite(loss)), "Nonfinite capacity loss")
        loss.backward()
        campaign.require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                         "Nonfinite capacity gradient")
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        campaign.require(bool(torch.isfinite(norm)), "Nonfinite native gradient clip")
        optimizer.step()
        torch.cuda.synchronize()
        campaign.require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Nonfinite capacity parameter")
        if step >= 5:
            timings.append(time.perf_counter() - begin)
    model.eval()
    with torch.inference_mode():
        output = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
    campaign.require(all(bool(torch.isfinite(value).all()) for value in output.values()
                         if torch.is_tensor(value) and value.is_floating_point()), "Nonfinite capacity inference")
    receipt = {**ready, "status": "passed", "completed_unix": time.time(), "warmup_steps": 5,
               "measure_steps": 20, "synthetic_optimizer_updates": 25, "scientific_optimizer_updates": 0,
               "step_median_seconds": statistics.median(timings), "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "capacity_admission_only": True, "timing_is_scientific_efficiency_evidence": False,
               "held_out_test_evaluated": False}
    campaign.write_json(folder / "receipt.json", receipt, exclusive=True)
    return receipt


def capacity_admission(c, host, root, deadline, *, campaign):
    """Exactly three native children; no next phase until actual exit and idle."""
    campaign.require(not campaign.common.gpu_pids(c["hosts"][host]), "GPU occupied before capacity admission")
    admission = root / "concurrency_admission"
    admission.mkdir(exist_ok=False)
    campaign.write_json(admission / "claim.json", {"pid": os.getpid(), "ppid": os.getppid(),
                        "contract_sha256": campaign.sha_json(c), "started_unix": time.time()}, exclusive=True)
    script = root / "operation" / "qualification.py"
    active, exits, peak = {}, {}, 0.
    try:
        for arm in (campaign.FULL, campaign.CNN, campaign.GRU):
            command = campaign.timeout_command(script, c, host,
                ["--mode", "capacity-worker", "--arm", arm, "--deadline", str(deadline)], deadline)
            log = admission / (arm + ".log")
            with log.open("xb") as stream:
                child = subprocess.Popen(command, cwd=campaign.ROOT, env=os.environ.copy(),
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            active[arm] = {"child": child, "command": command}
        ready_paths = {arm: admission / arm / "ready.json" for arm in active}
        while not all(path.exists() for path in ready_paths.values()):
            campaign.require(time.time() < deadline, "Capacity ready deadline reached")
            campaign.require(all(item["child"].poll() is None for item in active.values()),
                             "Capacity child exited before simultaneous ready")
            time.sleep(.1)
        ready = {arm: campaign.read(path) for arm, path in ready_paths.items()}
        campaign.require(len({value["pid"] for value in ready.values()}) == 3, "Duplicate capacity PID")
        for arm, value in ready.items():
            campaign.require(value["arm"] == arm and value["contract_sha256"] == campaign.sha_json(c)
                             and value["ppid"] == active[arm]["child"].pid == value["pgrp"]
                             and value["argv"] == active[arm]["command"][4:], "Capacity ready ownership drift")
        witness = campaign.gpu_observation(c, host, active, script)
        campaign.require(set(witness["gpu_pids"]) == {value["pid"] for value in ready.values()},
                         "Three simultaneous native GPU owners not witnessed")
        peak = witness["used_bytes"] / witness["total_bytes"]
        campaign.write_json(admission / "begin.json", {"contract_sha256": campaign.sha_json(c),
                            "deadline_unix": deadline, "issued_unix": time.time()}, exclusive=True)
        while active:
            campaign.require(time.time() < deadline, "Capacity phase deadline reached")
            observation = campaign.gpu_observation(c, host, active, script)
            peak = max(peak, observation["used_bytes"] / observation["total_bytes"])
            finished = []
            for arm, item in active.items():
                if item["child"].poll() is None:
                    continue
                actual = campaign.observe_exit(item["child"], item["command"], campaign.sha_json(c), arm=arm)
                campaign.write_json(admission / arm / "worker_exit.json", actual, exclusive=True)
                exits[arm] = actual
                campaign.require(actual["returncode"] == 0, "Capacity worker failed; no automatic retry")
                receipt = campaign.read(admission / arm / "receipt.json")
                campaign.require(receipt["status"] == "passed" and receipt["contract_sha256"] == campaign.sha_json(c)
                                 and receipt["pid"] == ready[arm]["pid"] and receipt["ppid"] == actual["worker_group_pid"]
                                 and receipt["scientific_optimizer_updates"] == 0, "Capacity receipt drift")
                finished.append(arm)
            for arm in finished:
                del active[arm]
            if active:
                time.sleep(.1)
        final = campaign.gpu_observation(c, host, active, script)
        campaign.require(not final["gpu_pids"], "Capacity GPU owner remains after actual exits")
        result = {"schema": "titantpp_component_capacity_admission_v1", "status": "passed",
                  "host": host, "contract_sha256": campaign.sha_json(c), "workers": 3,
                  "arms": [campaign.FULL, campaign.CNN, campaign.GRU], "shape": [128, 256],
                  "warmup_steps": 5, "measure_steps": 20, "simultaneous_ready_verified": True,
                  "GPU_witness": witness, "final_GPU_observation": final, "actual_child_exits": exits,
                  "aggregate_peak_fraction_observed": peak, "scientific_optimizer_updates": 0,
                  "capacity_admission_only": True, "timing_is_scientific_efficiency_evidence": False,
                  "held_out_test_evaluated": False, "completed_unix": time.time()}
        campaign.write_json(admission / "receipt.json", result, exclusive=True)
        return result
    except BaseException as exc:
        stopped = []
        for arm, item in active.items():
            try:
                campaign.stop_owned(item["child"])
                stopped.append(campaign.observe_exit(item["child"], item["command"], campaign.sha_json(c), arm=arm,
                                                      cancelled_by_owned_failure=True))
            except BaseException as stop_exc:
                stopped.append({"arm": arm, "stop_error": str(stop_exc)})
        campaign.write_json(admission / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                            "actual_exits_before_failure": exits, "owned_children_stopped": stopped,
                            "traceback": traceback.format_exc(), "automatic_retry": False}, exclusive=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--mode", choices=("native", "capacity-worker"), required=True)
    parser.add_argument("--arm")
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    operation = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("component_campaign", operation / "campaign.py")
    campaign = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = campaign
    spec.loader.exec_module(campaign)
    c = campaign.read(args.contract)
    root = Path(c["hosts"][args.host]["root"])
    campaign.require(Path(args.contract).resolve() == root / "execution_contract.json"
                     and operation == root / "operation", "Wrong isolated native entry")
    campaign.require(Path.cwd().resolve() == Path(c["hosts"][args.host]["source_root"]), "Wrong frozen native cwd")
    campaign.require(math.isfinite(args.deadline) and time.time() < args.deadline, "Native deadline expired")
    campaign.require(all(os.environ.get(k) == v for k, v in c["hosts"][args.host]["environment"].items()),
                     "Native environment drift")
    campaign.bootstrap(c, args.host)
    start = campaign.authorization(c, root)
    campaign.require(args.deadline <= start["deadline_unix"], "Native deadline extends host permit")
    if args.mode == "native":
        native_qualification(c, args.host, root, args.deadline, campaign=campaign)
    else:
        capacity_worker(c, args.host, root, args.arm, args.deadline, campaign=campaign)


if __name__ == "__main__":
    main()
