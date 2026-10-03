#!/usr/bin/env python3
"""Bounded synthetic CUDA qualification; never reads research data/checkpoints.

Each CUDA child exits before the next starts. Only the eight-row replay uses
AdamW (24 synthetic updates in total); production shapes use zero updates.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("B_log_original", "mixed_original", "mixed_raw_capped_original")
INITIAL_SHA256 = "2d054932a1b9f1fe8f3342242cb496c362a20c02eaca888d342ce03909a3ae9b"
MAX_SECONDS, MAX_FILE_BYTES, MAX_OUTPUT_BYTES = 900, 64 * 1024**2, 2 * 1024**3


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    require(len(encoded) < MAX_FILE_BYTES, "Qualification JSON exceeds file ceiling")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


class Budget:
    def __init__(self, output, deadline, monotonic_deadline=None):
        self.output, self.deadline = Path(output), deadline
        self.monotonic_deadline = monotonic_deadline

    def __call__(self, stage=None):
        require(time.time() < self.deadline, "CUDA qualification deadline reached")
        require(self.monotonic_deadline is None or time.monotonic() < self.monotonic_deadline,
                "CUDA qualification monotonic deadline reached")
        if self.output.exists():
            total = 0
            for path in self.output.rglob("*"):
                require(not path.is_symlink(), "Qualification output contains symlink")
                if path.is_file():
                    size = path.stat().st_size
                    require(size < MAX_FILE_BYTES, "Qualification file ceiling reached")
                    total += size
            require(total < MAX_OUTPUT_BYTES - MAX_FILE_BYTES, "Qualification output ceiling reached")
            require(shutil.disk_usage(self.output).free >= 5 * 1024**3,
                    "Qualification requires 5 GiB free disk")


def _load(contract_path, host):
    # The shared stdlib gate must run before any torch/CUDA import.
    from paper.scripts import raw_aux_parallel_common as common
    contract = common.read_contract(contract_path, check_source=True)
    require(host in {"5080", "5090"} and host in contract["hosts"], "Unknown qualification host")
    require(Path(contract["hosts"][host]["source_root"]).resolve() == ROOT,
            "CUDA qualification must run from the frozen host source root")
    common.apply_environment(contract["hosts"][host])
    return common, contract


def _output_path(contract, host, output):
    output = Path(output).resolve()
    require(output == Path(contract["hosts"][host]["root"]).resolve() / "qualification",
            "Qualification output must be the contracted host qualification directory")
    return output


def _assert_idle(host_spec):
    """Read-only check before any CUDA context or synthetic update is created."""
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True, timeout=15)
    expected = host_spec["gpu_uuid"]
    require(inventory.stdout.splitlines().count(expected) == 1, "Assigned GPU UUID is not uniquely present")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True, timeout=15)
    for row in csv.reader(processes.stdout.splitlines()):
        if not row or not any(field.strip() for field in row):
            continue
        require(len(row) == 2, "Cannot determine CUDA compute process ownership")
        require(row[0].strip() != expected, "Assigned GPU is busy; existing processes will not be changed")


def _tiny(contract, host, runtime, output, arm, stage, budget, common):
    from paper.scripts.quantity_comparison_engine import run_case
    from paper.scripts.quantity_objective_comparison import QuantityCase, QuantityStatistics
    from paper.scripts.run_raw_aux_gradient_comparison import load_design, objective_for_arm, synthetic_inputs
    from simple_lab_test.search.common.runner import torch_load_checkpoint

    require(arm in ARMS and stage in {"full", "first", "resume"}, "Invalid replay task")
    model, train, validation = synthetic_inputs()
    objective, control = objective_for_arm(arm)
    identity = common.make_identity(contract, host, runtime)
    identity = {**identity, "purpose": "synthetic_qualification",
                "data": {"kind": "synthetic", "train_targets": 8, "validation_targets": 8}}
    bootstrap = {"purpose": "synthetic_probe", "runtime": runtime,
                 "execution_contract_sha256": identity["execution_contract_sha256"]}
    target = Path(output) / arm / ("full" if stage == "full" else "split")
    result = run_case(model=model, train_loader=train, validation_loader=validation,
        case=QuantityCase.B_LOG_ORIGINAL, statistics=QuantityStatistics(1., load_design()["losses"]["raw_scale"]),
        output_dir=target, epochs=2, seed=42, identity=identity, device="cuda:0",
        lr=.001, weight_decay=.01, grad_clip=1., mixed_objective=objective, raw_aux_control=control,
        resume=stage == "resume", stop_after_epochs=1 if stage == "first" else None,
        budget_check=budget, cuda_qualification=bootstrap)
    saved = torch_load_checkpoint(target / "last_epoch_state.pt", map_location="cpu")
    proof = {"arm": arm, "stage": stage, "runtime": runtime,
             "execution_contract_sha256": identity["execution_contract_sha256"],
             "source": identity["source"], "status": result["status"],
             **{key: saved[key] for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256")}}
    write_json(Path(output) / arm / (stage + "_proof.json"), proof)
    return proof


def _same_gradients(left, right, *, exact=False):
    import torch
    require(len(left) == len(right), "Gradient parameter universe differs")
    for first, second in zip(left, right, strict=True):
        require((first is None) == (second is None), "Unused .grad semantics changed")
        if first is not None:
            require(first.dtype == second.dtype and first.shape == second.shape,
                    "Gradient shape/dtype changed")
            torch.testing.assert_close(first, second, rtol=0 if exact else 1e-5,
                                       atol=0 if exact else 1e-7)


def _gradient_audit(model, tensors, statistics, objective, control):
    """Use a nonzero synthetic head and verify the actual composed backward."""
    import torch
    from paper.scripts.mixed_quantity_objective import mixed_joint_causal_batch_objective
    from paper.scripts.raw_aux_gradient_control import apply_raw_aux_control
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, capture_rng_state
    from paper.scripts.time_quantity_diagnostic import _rng_hash

    model.train()
    model.zero_grad(set_to_none=True)
    outputs = mixed_joint_causal_batch_objective(model, *tensors, statistics=statistics, objective=objective)
    parameters = tuple(model.parameters())
    state_before = canonical_state_dict_sha256(model.state_dict())
    rng_before = _rng_hash(capture_rng_state(), {})
    original_grads = [p.grad for p in parameters]
    adjusted, record = apply_raw_aux_control(outputs, model.named_parameters(), objective=objective,
                                             control=control, training=True)
    require(all(p.grad is previous for p, previous in zip(parameters, original_grads, strict=True)),
            "Component queries assigned .grad")
    require(_rng_hash(capture_rng_state(), {}) == rng_before, "Controller consumed RNG")
    require(canonical_state_dict_sha256(model.state_dict()) == state_before, "Controller changed model state")
    # Additional queries are qualification-only and excluded from throughput.
    base = torch.autograd.grad((outputs["time_loss"] + outputs["log_qty_loss"]).mean(), parameters,
                               retain_graph=True, allow_unused=True)
    raw = torch.autograd.grad((objective.alpha * outputs["raw_qty_loss"]).mean(), parameters,
                              retain_graph=True, allow_unused=True)
    expected = [None if b is None and r is None else
                (0 if b is None else b) + record["s"] * (0 if r is None else r)
                for b, r in zip(base, raw, strict=True)]
    adjusted["objective_loss"].mean().backward()
    _same_gradients([p.grad for p in parameters], expected)
    before_clip = [None if p.grad is None else p.grad.detach().clone() for p in parameters]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    factor = min(1., 1. / (float(norm.item()) + 1e-6))
    _same_gradients([p.grad for p in parameters], [None if g is None else factor * g for g in before_clip])
    require(record["weighted_raw_norm"] > 0 and 0 < record["s"] < 1 and record["norm_bound_passed"],
            "Synthetic probe did not exercise active nonzero raw limitation")
    require(_rng_hash(capture_rng_state(), {}) == rng_before, "Gradient audit consumed RNG")
    require(canonical_state_dict_sha256(model.state_dict()) == state_before, "Gradient audit mutated model state")
    return {"passed": True, **record, "per_parameter_composition_passed": True,
            "postclip_gradient_passed": True, "common_clip_factor": factor,
            "state_and_rng_preserved": True, "optimizer_steps": 0,
            "additional_audit_gradient_queries": 2}


def _b_preservation(model, tensors, statistics, objective):
    import torch
    from paper.scripts.mixed_quantity_objective import mixed_joint_causal_batch_objective
    from paper.scripts.time_quantity_diagnostic import task_outputs
    from simple_lab_test.search.common.runner import capture_rng_state, restore_rng_state

    model.train()
    parameters = tuple(model.parameters())
    rng = capture_rng_state()
    previous = task_outputs(model, *tensors, "joint")
    first = torch.autograd.grad(previous["objective_loss"].mean(), parameters, allow_unused=True)
    restore_rng_state(rng)
    current = mixed_joint_causal_batch_objective(model, *tensors, statistics=statistics, objective=objective)
    second = torch.autograd.grad(current["objective_loss"].mean(), parameters, allow_unused=True)
    for key in ("objective_loss", "time_loss", "log_qty_loss", "pred_qty"):
        require(torch.equal(previous[key], current[key]), "B objective/prediction changed: " + key)
    _same_gradients(first, second, exact=True)
    return {"passed": True, "objective_and_gradients_bitwise_equal": True,
            "optimizer_steps": 0, "qualification_only_forward_calls": 2}


def _timed_batches(model, tensors, statistics, objective, control, budget):
    import torch
    from paper.scripts.mixed_quantity_objective import mixed_joint_causal_batch_objective
    from paper.scripts.raw_aux_gradient_control import apply_raw_aux_control

    model.train()
    rows = []
    for index in range(3):  # One warmup, two measured batches; no optimizer.
        budget("production_shape_batch")
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        outputs = mixed_joint_causal_batch_objective(model, *tensors, statistics=statistics, objective=objective)
        if control is not None:
            outputs, record = apply_raw_aux_control(outputs, model.named_parameters(), objective=objective,
                                                    control=control, training=True)
            require(record["norm_bound_passed"] and record["s"] < 1, "Inactive timed limiter probe")
        outputs["objective_loss"].mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - began
        require(math.isfinite(elapsed) and elapsed > 0 and bool(torch.isfinite(norm)), "Nonfinite CUDA timing/gradient")
        rows.append({"warmup": index == 0, "seconds": elapsed,
                     "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                     "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                     "global_preclip_norm": float(norm.item())})
        del outputs
    return {"batches": rows, "measured_mean_seconds": sum(r["seconds"] for r in rows[1:]) / 2,
            "forward_calls": 3, "ordinary_backward_calls": 3,
            "component_gradient_queries": 6 if control is not None else 0,
            "optimizer_steps": 0,
            "timing_scope": "synthetic device forward+component queries+backward+clip only; excludes AdamW,data,validation,save; not training ETA"}


def _causality(model, tensors, statistics, objective):
    import torch
    from paper.scripts.mixed_quantity_objective import mixed_joint_causal_batch_objective
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    model.eval()
    before_state = canonical_state_dict_sha256(model.state_dict())
    dts, mask, quantities = (value.clone() for value in tensors)
    with torch.no_grad():
        before = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
            statistics=statistics, objective=objective)["pred_qty"]
        dts[~mask], quantities[~mask] = 999., 999.
        dts[:, -1], quantities[:, -1] = 42., 123.
        after = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
            statistics=statistics, objective=objective)["pred_qty"]
        dts[:, -2], quantities[:, -2] = 7., 100.
        changed = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
            statistics=statistics, objective=objective)["pred_qty"]
    require(torch.equal(before, after), "Native CUDA target/padding causality failed")
    require(not torch.equal(after, changed), "Native CUDA probe is insensitive to valid history")
    require(canonical_state_dict_sha256(model.state_dict()) == before_state,
            "Causality evaluation mutated probe state")
    return {"passed": True, "target_padding_invariance": True, "valid_history_sensitivity": True,
            "nonzero_head": True, "evaluation_forward_calls": 3, "optimizer_steps": 0}


def _shapes(contract, host, runtime, output, budget):
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.quantity_objective_comparison import QuantityStatistics
    from paper.scripts.run_raw_aux_gradient_comparison import load_design, objective_for_arm
    from paper.scripts.run_time_quantity_diagnostic import reset_seed
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    design = load_design()
    exposure = design["initialization_and_exposure"]
    statistics = QuantityStatistics(exposure["statistics"]["train_log_mean"], design["losses"]["raw_scale"])
    records = {}
    for arm in ARMS:
        budget("production_shape")
        reset_seed(42)
        model, _ = build_count_aware_model(**exposure["model"], train_log_mean=statistics.mu,
            train_log_std=exposure["statistics"]["train_log_std"], max_seq_len=256)
        model.to("cuda:0")
        initial = canonical_state_dict_sha256(model.state_dict())
        require(initial == INITIAL_SHA256 == exposure["initial_state_sha256"],
                "Native production initialization differs; CPU hash substitution is forbidden")
        with torch.no_grad():
            model.quantity_head.weight.copy_(torch.linspace(-.02, .03, model.quantity_head.weight.numel(),
                device="cuda:0").reshape_as(model.quantity_head.weight))
        probe_sha = canonical_state_dict_sha256(model.state_dict())
        dts = torch.ones((128, 256), device="cuda:0")
        mask = torch.ones_like(dts, dtype=torch.bool)
        mask[::2, :64] = False
        mask[1::2, :128] = False
        quantities = torch.full_like(dts, 3.)
        quantities[:, -1] = 10000.  # Fixed artificial target activates raw cap.
        tensors = dts, mask, quantities
        objective, control = objective_for_arm(arm, design)
        record = {"initial_state_sha256": initial, "probe_state_sha256": probe_sha,
                  "synthetic_nonzero_head": True}
        if arm == ARMS[0]:
            record["B_preservation"] = _b_preservation(model, tensors, statistics, objective)
        if control is not None:
            record["active_gradient_audit"] = _gradient_audit(model, tensors, statistics, objective, control)
        record["causality"] = _causality(model, tensors, statistics, objective)
        reset_seed(42)  # Same timed dropout sequence after unequal audit work.
        record["timing"] = _timed_batches(model, tensors, statistics, objective, control, budget)
        require(canonical_state_dict_sha256(model.state_dict()) == probe_sha, "Shape probe changed weights")
        records[arm] = record
        del model, tensors, dts, mask, quantities
        gc.collect()
        torch.cuda.empty_cache()
    result = {"passed": True, "host_alias": host, "runtime": runtime,
              "initial_state_sha256": INITIAL_SHA256, "batch_size": 128,
              "sequence_length": 256, "hidden_dim": 64, "optimizer_steps": 0,
              "arms": records, "research_data_read": False,
              "timing_is_training_eta": False, "cross_host_bitwise_equivalence_claim": False}
    write_json(Path(output) / "production_shape.json", result)
    return result


def _child(contract_path, host, output, task, arm, stage, deadline):
    common, contract = _load(contract_path, host)
    output = _output_path(contract, host, output)
    budget = Budget(output, deadline)
    budget()
    runtime = common.runtime_check(contract, host)
    if task == "tiny":
        return _tiny(contract, host, runtime, output, arm, stage, budget, common)
    require(task == "shapes", "Unknown CUDA child task")
    return _shapes(contract, host, runtime, output, budget)


def _audit_replays(output):
    from paper.scripts.run_quantity_comparison import compare_replay
    from paper.scripts.run_raw_aux_gradient_comparison import audit_pairs
    summaries, proofs = {}, {}
    for arm in ARMS:
        folder = Path(output) / arm
        full, split = read_json(folder / "full/summary.json"), read_json(folder / "split/summary.json")
        compare_replay(full, split)
        require(full["status"] == "complete" and full["epochs_completed"] == 2 and full["global_step"] == 4,
                "Tiny replay exposure changed")
        first, last = read_json(folder / "full_proof.json"), read_json(folder / "resume_proof.json")
        for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256", "runtime",
                    "execution_contract_sha256", "source"):
            require(first[key] == last[key], "Fresh-process CUDA replay differs: " + key)
        summaries[arm], proofs[arm] = full, first
    paired = audit_pairs(summaries)
    return {"passed": True, "paired_exposure": paired, "arms": proofs,
            "synthetic_optimizer_updates": 24, "fresh_process_exact_resume": True}


def qualify(contract_path, host, output_dir):
    began_monotonic = time.monotonic()
    common, contract = _load(contract_path, host)
    output = _output_path(contract, host, output_dir)
    require(not output.exists(), "Qualification requires a fresh output directory; no automatic retry")
    limit = contract["limits"]["qualification_seconds"]
    require(type(limit) in {int, float} and 0 < limit <= MAX_SECONDS, "Invalid qualification time ceiling")
    started = time.time()
    deadline = started + limit
    monotonic_deadline = began_monotonic + limit
    source = common.source_manifest()
    output.mkdir(parents=True)
    budget = Budget(output, deadline, monotonic_deadline)
    tasks = [("tiny", arm, stage) for arm in ARMS for stage in ("full", "first", "resume")]
    tasks.append(("shapes", None, None))
    try:
        budget()
        _assert_idle(contract["hosts"][host])
        for task, arm, stage in tasks:
            budget()
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--contract", str(Path(contract_path).resolve()),
                       "--host", host, "--output", str(output), "--child", task, "--deadline", str(deadline)]
            if arm is not None:
                command.extend(["--arm", arm, "--stage", stage])
            result = subprocess.run(command, check=True, capture_output=True, text=True,
                                    timeout=max(.001, min(deadline - time.time(), monotonic_deadline - time.monotonic())))
            name = f"{arm}_{stage}" if task == "tiny" else "production_shape"
            write_json(output / (name + "_process.json"), {"exit_code": result.returncode,
                "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
        replay = _audit_replays(output)
        shape = read_json(output / "production_shape.json")
        require(shape["passed"] is True and shape["initial_state_sha256"] == INITIAL_SHA256,
                "Production shape qualification missing")
        runtime = shape["runtime"]
        identity = common.make_identity(contract, host, runtime)
        for proof in replay["arms"].values():
            require(proof["runtime"] == runtime and proof["execution_contract_sha256"] == identity["execution_contract_sha256"],
                    "Qualification child runtime/contract drift")
        common.read_contract(contract_path, check_source=True)
        require(common.source_manifest() == source, "Source changed during CUDA qualification")
        budget()
        proof_files = {str(path.relative_to(output)): sha_file(path)
                       for path in sorted(output.rglob("*")) if path.is_file()}
        receipt = {"schema": "raw_aux_gradient_cuda_qualification_v1", "passed": True, "qualifies_cuda": True,
            "host_alias": host, "runtime": runtime, "execution_contract_sha256": identity["execution_contract_sha256"],
            "source_files_sha256": source["files_sha256"], "initial_state_sha256": INITIAL_SHA256,
            "tiny_replay": replay, "production_shape": shape, "proof_files": proof_files,
            "checked_at_unix": time.time(), "elapsed_seconds": time.monotonic() - began_monotonic,
            "wall_clock_elapsed_seconds": time.time() - started,
            "qualification_deadline_unix": deadline, "research_optimizer_updates": 0,
            "synthetic_optimizer_updates": 24, "research_data_or_checkpoint_read": False,
            "cross_host_bitwise_equivalence_claim": False}
        write_json(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        write_json(output / "failure.json", {"passed": False, "host_alias": host,
            "error_type": type(error).__name__, "error": str(error)[:2000],
            "checked_at_unix": time.time(), "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", choices=("tiny", "shapes"), help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("full", "first", "resume"), help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        require(args.deadline is not None and 0 < args.deadline - time.time() <= MAX_SECONDS,
                "Bounded child deadline required")
        result = _child(args.contract, args.host, args.output, args.child, args.arm, args.stage, args.deadline)
    else:
        result = qualify(args.contract, args.host, args.output)
    print(json.dumps({key: result[key] for key in ("passed", "host_alias", "initial_state_sha256", "status") if key in result}))


if __name__ == "__main__":
    main()
