#!/usr/bin/env python3
"""One explicitly approved Instacart last-arm transfer, preserving frozen sources."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SCHEMA = "mixed_quantity_last_arm_transfer_v1"
APPROVAL_SCOPE = "manual_instacart_mixed_resume_5080_and_pending_matched_fresh_5090"
PARTITION_SHA = "045a29adc5da5f65cfea21b42454f2b584275a1de4e4e91c89edab914e7f303e"
OLD_WRAPPER_SHA = "ab23980fa1e34845e03f7c99dc0d53b21fc3ae997f4d35f566c7d445ff0b5722"
OLD_5080_SHA = "85d934db1a291693ddb171b63a181e4b7580c9cf8c19ab6b9da51161df182adc"
DEADLINE = 1789390323.7680202
DATASET = "insta_market_basket"
ROLES = {"mixed_continuation_5080": ("instacart_5080", "mixed_original"),
         "matched_fresh_5090": ("continuation_5090", "mixed_matched_original")}
PROBE_POLICY = {"sample_count": 512, "batch_size": 128, "seed": 1042,
                "seconds": 600, "output_bytes": 268435456, "rtol": 1e-4, "atol": 1e-6,
                "gradient_relative_l2": 1e-4, "gradient_absolute_l2": 1e-6,
                "anchors": ["initial", "B_final_epoch120"], "optimizer_steps": 0, "model_mode": "eval"}
POLICY = {"automatic_retry": False, "automatic_resume": False, "held_out": False,
          "budget_extension": False, "cpu_fallback": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_old(path):
    require(sha_file(path) == OLD_WRAPPER_SHA, "Frozen partition wrapper hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_mixed_partition_transfer_library", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def partition(contract):
    return contract["original_partition"]["contract"]


def parent(contract):
    return partition(contract)["parent"]["contract"]


def effective(contract, role, old):
    return old.effective_parent_contract(parent(contract), partition(contract), ROLES[role][0])


def native_identity(contract, role, old):
    data = next(d for d in parent(contract)["datasets"] if d["dataset_id"] == DATASET)
    return old.training_identity(parent(contract), partition(contract), ROLES[role][0], data,
                                 contract["hosts"][role]["runtime_expected"])


def training_identity(contract, role, old):
    identity = native_identity(contract, role, old)
    if role == "matched_fresh_5090":
        identity.update(transfer_contract_sha256=sha_json(contract),
                        calibration_provenance_identity=contract["calibration"]["receipt"]["identity"],
                        cross_runtime_comparison=True)
    return identity


def validate_contract(contract, old, *, wrapper_path=None, now=None):
    require(contract["schema"] == SCHEMA and contract["status"] == "frozen_pending_explicit_approval", "Transfer schema/status mismatch")
    require(sha_json(partition(contract)) == contract["original_partition"]["sha256"] == PARTITION_SHA,
            "Original partition contract changed")
    old.validate_contract(partition(contract))
    require(contract["deadline_unix"] == partition(contract)["deadline_unix"] == DEADLINE, "Original deadline changed")
    require(now is None or now < DEADLINE, "Original deadline reached")
    require(contract["limits"] == old.LIMITS and contract["policy"] == POLICY, "Transfer limits/policy changed")
    require(contract["probe_policy"] == PROBE_POLICY, "Fixed probe policy changed")
    require(set(contract["hosts"]) == set(ROLES), "Exactly two last-arm host roles required")
    for role, (old_role, _) in ROLES.items():
        host, previous = contract["hosts"][role], partition(contract)["hosts"][old_role]
        require(host["old_role"] == old_role, "Host role mismatch")
        for field in ("server_alias", "python", "tmux_binary", "source_root", "runtime_expected"):
            require(host[field] == previous[field], "Original host runtime/source changed: " + field)
        require(host["old_wrapper_path"] == previous["wrapper_path"], "Frozen wrapper location changed")
        for field in ("wrapper_path", "output_dir", "probe_output_dir"):
            path = Path(host[field])
            require(path.is_absolute() and str(path) == host[field] and ".." not in path.parts, "Canonical host path required")
            for original in (Path(previous["output_dir"]), Path(host["source_root"])):
                require(not path.is_relative_to(original) and not original.is_relative_to(path), "Separate preserved source/results required")
        require(not Path(host["output_dir"]).is_relative_to(host["probe_output_dir"])
                and not Path(host["probe_output_dir"]).is_relative_to(host["output_dir"]), "Separate probe/training output required")
        require(host["tmux_session"] and host["probe_tmux_session"]
                and len({host["tmux_session"], host["probe_tmux_session"], previous["tmux_session"]}) == 3,
                "Fresh separate pinned tmux sessions required")
        q = contract["qualifications"][role]
        require(q["passed"] is True and q["qualifies_cuda"] is True
                and q["runtime"] == host["runtime_expected"]
                and q["execution_contract_sha256"] == sha_json(effective(contract, role, old)), "Native qualification mismatch")
    require(sha_json(effective(contract, "mixed_continuation_5080", old)) == OLD_5080_SHA, "5080 continuation identity changed")
    calibration = contract["calibration"]["receipt"]
    require(calibration["identity"] == native_identity(contract, "mixed_continuation_5080", old), "Original calibration identity changed")
    for spec in (contract["calibration"], contract["anchor"]):
        require(len(spec["file_sha256"]) == 64 and set(spec["paths"]) == set(ROLES), "Pinned evidence hash/host paths required")
        require(all(Path(p).is_absolute() and ".." not in Path(p).parts for p in spec["paths"].values()), "Absolute evidence paths required")
    require(len(contract["anchor"]["state_sha256"]) == 64, "B final state identity required")
    if wrapper_path:
        require(contract["wrapper"]["sha256"] == sha_file(wrapper_path), "Transfer wrapper hash mismatch")
    return {"status": "validated_metadata_only", "new_training_arms": 1, "resumed_arms": 1, "gpu_started": False}


def validate_approval(contract, approval):
    require(approval.get("status") == "approved_by_user" and approval.get("scope") == APPROVAL_SCOPE
            and approval.get("transfer_contract_sha256") == sha_json(contract)
            and bool(approval.get("user_instruction")), "Explicit hash-bound transfer approval required")


def validate_handoff(contract, handoff):
    require(handoff.get("schema") == "mixed_quantity_last_arm_handoff_v1"
            and handoff.get("transfer_contract_sha256") == sha_json(contract), "Handoff authority mismatch")
    require(handoff.get("observed_old_process_dead") is True and handoff.get("matched_never_started") is True,
            "Old ownership must be released and matched arm never started")
    pids = handoff.get("original_process_ids", [])
    require(len(pids) >= 2 and all(type(p) is int and p > 0 for p in pids), "Original supervisor/worker PID evidence required")
    require(type(handoff.get("committed_epoch")) is int and 1 <= handoff["committed_epoch"] < 120,
            "Incomplete mixed arm committed epoch required")
    original = Path(partition(contract)["hosts"]["instacart_5080"]["output_dir"]) / DATASET
    require(handoff.get("source_dir") == str(original / "mixed_original"), "Only original mixed arm can resume")
    files = handoff.get("persisted_file_hashes", {})
    require(files and {str(original / "mixed_original" / n) for n in ("contract.json", "last_epoch_state.pt")} <= set(files),
            "Committed mixed checkpoint and contract required")
    require(all(Path(p).is_relative_to(original / "mixed_original") and ".." not in Path(p).parts
                and len(h) == 64 for p, h in files.items()), "Unassigned inherited file forbidden")


def verify_old_dead(contract, handoff, role):
    if role != "mixed_continuation_5080":
        return
    for pid in handoff["original_process_ids"]:
        require(not Path(f"/proc/{pid}").exists(), "Original 5080 process still exists")
    original = Path(partition(contract)["hosts"]["instacart_5080"]["output_dir"])
    require(not (original / DATASET / "mixed_matched_original").exists(), "Old matched arm has started")


def validate_probe(contract, receipt, role, handoff=None):
    require(isinstance(receipt, dict) and receipt.get("schema") == "mixed_quantity_cross_runtime_probe_v1" and receipt.get("status") == "passed"
            and receipt.get("transfer_contract_sha256") == sha_json(contract) and receipt.get("role") == role,
            "Passed host-bound probe required")
    require(receipt.get("policy") == PROBE_POLICY and receipt.get("optimizer_steps") == 0
            and receipt.get("runtime") == contract["hosts"][role]["runtime_expected"], "Probe policy/runtime mismatch")
    require(receipt.get("calibration_file_sha256") == contract["calibration"]["file_sha256"]
            and receipt.get("anchor_file_sha256") == contract["anchor"]["file_sha256"], "Probe evidence mismatch")
    if handoff is not None:
        require(receipt.get("handoff_sha256") == sha_json(handoff), "Probe belongs to a different handoff")
    require(len(receipt.get("input_tensors_sha256", "")) == 64
            and receipt.get("initial_state_sha256") == contract["calibration"]["receipt"]["initial_model_state_sha256"]
            and receipt.get("anchor_state_sha256") == contract["anchor"]["state_sha256"],
            "Probe input/initialization/anchor identity mismatch")
    require(set(receipt.get("anchors", {})) == set(PROBE_POLICY["anchors"]), "Both probe anchors required")
    for anchor in receipt["anchors"].values():
        require(set(anchor) == {"B_log_original", "mixed_original", "mixed_matched_original"}, "Three probe objectives required")
        for row in anchor.values():
            require(set(row["outputs"]) == {"pred_qty", "time_loss", "log_qty_loss", "raw_qty_loss", "objective_loss"}, "Probe output groups missing")
            require(set(row["gradients"]) == {"encoder", "time_head", "quantity_head"}, "Probe gradient groups missing")
            require(all(len(vector) == PROBE_POLICY["sample_count"] for vector in row["outputs"].values()), "Probe requires exactly 512 output samples")
            for gradient in row["gradients"].values():
                parameters = gradient["parameters"]
                require(parameters and len({p["name"] for p in parameters}) == len(parameters)
                        and all(isinstance(p["name"], str) and isinstance(p["shape"], list)
                                and all(type(n) is int and n > 0 for n in p["shape"]) for p in parameters)
                        and sum(math.prod(p["shape"]) for p in parameters) == len(gradient["values"]), "Invalid gradient parameter layout")
            for vector in [*row["outputs"].values(), *(g["values"] for g in row["gradients"].values())]:
                require(isinstance(vector, list) and len(vector) > 0 and all(type(x) in {int, float} and math.isfinite(x) for x in vector), "Probe nonfinite/empty vector")


def compare_probes(contract, left, right):
    import numpy as np
    validate_probe(contract, left, "mixed_continuation_5080")
    validate_probe(contract, right, "matched_fresh_5090")
    for key in ("handoff_sha256", "input_tensors_sha256", "indices_sha256", "initial_state_sha256", "anchor_state_sha256"):
        require(left.get(key) == right.get(key) and left.get(key) is not None, "Cross-host probe identity mismatch: " + key)
    checks = []
    for anchor in PROBE_POLICY["anchors"]:
        for case, a in left["anchors"][anchor].items():
            b = right["anchors"][anchor][case]
            for key, values in a["outputs"].items():
                x, y = np.asarray(values, dtype=np.float64), np.asarray(b["outputs"][key], dtype=np.float64)
                require(x.shape == y.shape, "Cross-host output shape mismatch")
                ok = bool(np.all(np.abs(x-y) <= PROBE_POLICY["atol"] + PROBE_POLICY["rtol"] * np.maximum(np.abs(x), np.abs(y))))
                checks.append({"anchor": anchor, "case": case, "kind": "output", "group": key, "passed": ok,
                               "max_abs_difference": float(np.max(np.abs(x-y)))})
            for key, vector in a["gradients"].items():
                other = b["gradients"][key]
                require(vector["parameters"] == other["parameters"], "Cross-host gradient parameter layout mismatch")
                x, y = np.asarray(vector["values"], dtype=np.float64), np.asarray(other["values"], dtype=np.float64)
                require(x.shape == y.shape, "Cross-host gradient shape mismatch")
                delta, norm = float(np.linalg.norm(x-y)), max(float(np.linalg.norm(x)), float(np.linalg.norm(y)))
                relative = delta / norm if norm else 0.
                ok = delta <= PROBE_POLICY["gradient_absolute_l2"] or relative <= PROBE_POLICY["gradient_relative_l2"]
                checks.append({"anchor": anchor, "case": case, "kind": "gradient", "group": key,
                               "passed": ok, "absolute_l2": delta, "relative_l2": relative})
    return {"schema": "mixed_quantity_cross_runtime_compatibility_v1", "transfer_contract_sha256": sha_json(contract),
            "handoff_sha256": left["handoff_sha256"], "passed": all(c["passed"] for c in checks), "checks": checks,
            "probe_sha256": {left["role"]: sha_json(left), right["role"]: sha_json(right)},
            "policy": PROBE_POLICY, "long_training_bitwise_equivalence_claimed": False}


def validate_training_gates(contract, role, handoff, own_probe, compatibility=None):
    validate_handoff(contract, handoff)
    validate_probe(contract, own_probe, role, handoff)
    if role == "matched_fresh_5090":
        require(isinstance(compatibility, dict) and compatibility.get("schema") == "mixed_quantity_cross_runtime_compatibility_v1"
                and compatibility.get("passed") is True and compatibility.get("transfer_contract_sha256") == sha_json(contract)
                and compatibility.get("handoff_sha256") == sha_json(handoff) and compatibility.get("policy") == PROBE_POLICY
                and compatibility.get("probe_sha256", {}).get(role) == sha_json(own_probe)
                and set(compatibility.get("probe_sha256", {})) == set(ROLES)
                and len(compatibility.get("checks", [])) == 48 and all(c.get("passed") is True for c in compatibility["checks"]),
                "Matched fresh launch requires passed cross-host compatibility")


def _evidence(contract, role):
    for field in ("calibration", "anchor"):
        spec = contract[field]
        path = Path(spec["paths"][role])
        require(path.is_file() and not path.is_symlink() and path.stat().st_size <= contract["limits"]["per_file_bytes"]
                and sha_file(path) == spec["file_sha256"], "Pinned evidence file mismatch: " + field)
    receipt = read_json(contract["calibration"]["paths"][role])
    require(receipt == contract["calibration"]["receipt"], "Original calibration bytes/content changed")
    return receipt


def _inputs(contract, role, old, frozen):
    from paper.scripts import quantity_comparison_data as data_helper
    from paper.scripts import quantity_objective_comparison as quantity
    from paper.scripts import run_time_quantity_diagnostic as diagnostic
    from paper.scripts import mixed_quantity_objective as primitives
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    data = next(d for d in parent(contract)["datasets"] if d["dataset_id"] == DATASET)
    frame, metadata = data_helper.prepare_quantity_comparison_data(data)
    metadata = frozen.base.bind_frozen_statistics(data, metadata)
    model, train, validation, _ = diagnostic.build_arm_inputs({**data, "seed": 42}, frame, metadata)
    statistics = quantity.QuantityStatistics(metadata["train_log_mean"], metadata["raw_scale"])
    calibration = _evidence(contract, role)
    require(calibration["statistics"] == {"mu": statistics.mu, "raw_scale": statistics.raw_scale}
            and canonical_state_dict_sha256(model.state_dict()) == calibration["initial_model_state_sha256"],
            "Calibration statistics/initialization mismatch")
    objectives = primitives.objectives_from_calibration(calibration)
    require(objectives[1].alpha == 0.160021665648455 and objectives[2].quantity_scale == 0.8340005759765311,
            "Frozen Instacart coefficients changed")
    return data, model, train, validation, statistics, calibration, objectives


def probe_anchors(model, batches, statistics, objectives, states, *, device, check):
    """Optimizer-free deterministic forward/gradient comparison on fixed anchors."""
    import torch
    from paper.scripts import mixed_quantity_objective as primitives
    from paper.scripts.time_quantity_diagnostic import _group
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    results = {}
    model.to(device).eval()
    for anchor, state in states.items():
        model.load_state_dict(state, strict=True)
        results[anchor] = {}
        for objective in objectives:
            check()
            outputs = {key: [] for key in ("pred_qty", "time_loss", "log_qty_loss", "raw_qty_loss", "objective_loss")}
            parameters = list(model.named_parameters())
            accumulated = [torch.zeros_like(p) for _, p in parameters]
            for batch in batches:
                check()
                dts, mask, quantities = (t.to(device) for t in batch)
                values = primitives.mixed_joint_causal_batch_objective(model, dts, mask, quantities,
                                                                         statistics=statistics, objective=objective)
                for key in outputs:
                    outputs[key].extend(values[key].detach().cpu().flatten().tolist())
                grads = torch.autograd.grad(values["objective_loss"].mean(), [p for _, p in parameters], allow_unused=True)
                for total, grad in zip(accumulated, grads, strict=True):
                    if grad is not None:
                        total.add_(grad.detach() / len(batches))
            groups = {key: {"parameters": [], "values": []} for key in ("encoder", "time_head", "quantity_head")}
            for (name, parameter), grad in zip(parameters, accumulated, strict=True):
                group = _group(name)
                groups[group]["parameters"].append({"name": name, "shape": list(parameter.shape)})
                groups[group]["values"].extend(grad.cpu().flatten().tolist())
            results[anchor][objective.name] = {"outputs": outputs, "gradients": groups}
        require(canonical_state_dict_sha256(model.state_dict()) == canonical_state_dict_sha256(state), "Probe changed model state")
    return results


def _probe(contract, role, handoff, old, frozen, output):
    import torch
    from torch.utils.data import DataLoader, Subset
    from paper.scripts import mixed_quantity_objective as primitives
    from paper.scripts import quantity_comparison_engine as engine
    from paper.scripts.time_quantity_diagnostic import _batch_tensors
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    budget = old.Budget({**contract, "limits": {**contract["limits"], "output_bytes_per_host": PROBE_POLICY["output_bytes"]}}, output)
    began = time.monotonic()
    def check():
        budget()
        require(time.monotonic() - began < PROBE_POLICY["seconds"], "Compatibility probe deadline reached")
    check()
    runtime = frozen.configured_runtime(effective(contract, role, old), "cuda:0")
    _, model, train, validation, statistics, calibration, objectives = _inputs(contract, role, old, frozen)
    require(set(train.dataset.target_splits) == {"train"}, "Probe data must be train-only")
    indices = torch.randperm(len(train.dataset), generator=torch.Generator().manual_seed(PROBE_POLICY["seed"]))[:512].tolist()
    loader = DataLoader(Subset(train.dataset, indices), batch_size=128, shuffle=False, num_workers=0,
                        collate_fn=primitives._collate, generator=torch.Generator().manual_seed(1042))
    batches = [_batch_tensors(batch, "cpu") for batch in loader]
    del train, validation
    digest = hashlib.sha256()
    for batch in batches:
        for tensor in batch:
            digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    initial_hash = canonical_state_dict_sha256(initial)
    payload = torch_load_checkpoint(contract["anchor"]["paths"][role], map_location="cpu")
    engine._validate_payload(payload, payload["contract"])
    require(payload["epoch"] == 120 and payload["contract"]["identity"] == native_identity(contract, "mixed_continuation_5080", old)
            and payload["contract"]["condition"]["mixed_objective"]["name"] == "B_log_original"
            and payload["model_state_sha256"] == contract["anchor"]["state_sha256"], "Frozen B final checkpoint mismatch")
    results = probe_anchors(model, batches, statistics, objectives,
                            {"initial": initial, "B_final_epoch120": payload["model_state_dict"]},
                            device="cuda:0", check=check)
    check()
    result = {"schema": "mixed_quantity_cross_runtime_probe_v1", "status": "passed", "role": role,
              "transfer_contract_sha256": sha_json(contract), "handoff_sha256": sha_json(handoff),
              "policy": PROBE_POLICY, "runtime": runtime, "optimizer_steps": 0,
              "calibration_file_sha256": contract["calibration"]["file_sha256"],
              "anchor_file_sha256": contract["anchor"]["file_sha256"], "anchor_state_sha256": payload["model_state_sha256"],
              "initial_state_sha256": initial_hash, "input_tensors_sha256": digest.hexdigest(), "indices_sha256": sha_json(indices),
              "anchors": results, "elapsed_seconds": time.monotonic()-began}
    validate_probe(contract, result, role, handoff)
    return result


def _train(contract, role, handoff, old, frozen, output, own_probe, compatibility):
    from paper.scripts import quantity_comparison_engine as engine
    from paper.scripts import mixed_quantity_acceptance as acceptance
    from paper.scripts.quantity_objective_comparison import QuantityCase
    validate_training_gates(contract, role, handoff, own_probe, compatibility)
    receipt = contract["qualifications"][role]
    require(frozen.configured_runtime(effective(contract, role, old), "cuda:0") == receipt["runtime"], "Runtime drift")
    data, model, train, validation, statistics, calibration, objectives = _inputs(contract, role, old, frozen)
    objective = next(o for o in objectives if o.name == ROLES[role][1])
    identity = training_identity(contract, role, old)
    target = Path(output) / DATASET / objective.name
    calibration_target = Path(output) / DATASET / "calibration.json"
    calibration_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract["calibration"]["paths"][role], calibration_target)
    require(sha_file(calibration_target) == contract["calibration"]["file_sha256"], "Copied calibration changed")
    resume = role == "mixed_continuation_5080"
    if resume:
        old.copy_and_repair_arm({"source_dir": handoff["source_dir"], "mode": "resume"}, target, handoff,
                                identity=identity, objective=objective, engine=engine)
        require(read_json(target / "handoff_repair_receipt.json")["committed_epoch"] == handoff["committed_epoch"], "Committed epoch changed")
    write_json(Path(output) / "status.json", {"status": "running", "host_role": role, "dataset_id": DATASET,
               "case": objective.name, "mode": "resume" if resume else "fresh", "deadline_unix": DEADLINE})
    budget = old.Budget(contract, output)
    summary = engine.run_case(model=model, train_loader=train, validation_loader=validation,
        case=QuantityCase.B_LOG_ORIGINAL, mixed_objective=objective, statistics=statistics, output_dir=target,
        epochs=120, seed=42, identity=identity, device="cuda:0", lr=.001, weight_decay=.01, grad_clip=1.,
        resume=resume, budget_check=budget, cuda_qualification=receipt)
    old.audit_arm(summary, read_json(target / "contract.json"), data, objective, identity)
    write_json(Path(output) / "status.json", {"status": "validating_checkpoints", "host_role": role,
               "dataset_id": DATASET, "case": objective.name, "deadline_unix": DEADLINE})
    diagnosis = acceptance.evaluate_checkpoints(model=model, validation_loader=validation, statistics=statistics,
        objective=objective, arm_dir=target, quantity_boundaries=data["quantity_boundaries_all_train_rows"], device="cuda:0", budget_check=budget)
    write_json(target / "validation_diagnosis.json", diagnosis)
    return {"status": "host_complete", "transfer_contract_sha256": sha_json(contract), "role": role,
            "deadline_unix": DEADLINE, "completed_arms": [{"dataset_id": DATASET, "case": objective.name}],
            "global_suite_complete": False, "calibration_recomputed": False, "old_results_modified": False,
            "cross_runtime_comparison": not resume}


def _authority(authority):
    contract, role = authority["contract"], authority["role"]
    require(role in ROLES, "Unknown role")
    host = contract["hosts"][role]
    old = load_old(host["old_wrapper_path"])
    validate_contract(contract, old, wrapper_path=__file__, now=time.time())
    validate_approval(contract, authority["approval"])
    validate_handoff(contract, authority["handoff"])
    require(Path(__file__).resolve() == Path(host["wrapper_path"]).resolve()
            and Path(sys.executable).resolve() == Path(host["python"]).resolve(), "Pinned wrapper/Python required")
    require(authority["mode"] in {"probe", "execute"}, "Invalid authority mode")
    expected = host["probe_output_dir"] if authority["mode"] == "probe" else host["output_dir"]
    require(authority["output"] == expected, "Pinned output required")
    verify_old_dead(contract, authority["handoff"], role)
    old.verify_source(host["source_root"], parent(contract))
    _evidence(contract, role)
    old._apply_environment(host)
    frozen = old._frozen_runner(host["source_root"])
    frozen.validate_contract(parent(contract))
    return contract, role, old, frozen


def execute_mode(contract, approval, handoff, role, mode, *, own_probe=None, compatibility=None):
    require(mode in {"probe", "execute"}, "Public mode required")
    host = contract["hosts"][role]
    output = Path(host["probe_output_dir"] if mode == "probe" else host["output_dir"])
    authority = {"contract": contract, "approval": approval, "handoff": handoff, "role": role,
                 "mode": mode, "output": str(output), "own_probe": own_probe, "compatibility": compatibility}
    _, _, old, _ = _authority(authority)
    if mode == "execute":
        validate_training_gates(contract, role, handoff, own_probe, compatibility)
    require(not output.exists(), "Fresh output required; no automatic retry")
    require(bool(os.environ.get("TMUX")), "Pinned tmux required")
    observed = subprocess.check_output([host["tmux_binary"], "display-message", "-p", "#S"], text=True, timeout=15).strip()
    require(observed == host["probe_tmux_session"] if mode == "probe" else observed == host["tmux_session"], "Wrong tmux session")
    import fcntl
    # Shared directory for this host's probe and training supervisors.
    with (Path(host["wrapper_path"]).parent / "last_arm_transfer_gpu.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old.assert_launch_gates(contract, role, disk_free=shutil.disk_usage(output.parent).free,
                               gpu_pids=old._gpu_pids(host), executable=host["python"])
        output.mkdir()
        authority["supervisor_pid"] = os.getpid()
        path = output / "transfer_authority.json"
        write_json(path, authority)
        process = None
        handlers = {}
        def interrupted(signum, frame):
            raise InterruptedError(f"Transfer supervisor received {signum}")
        began = time.monotonic()
        try:
            for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                handlers[signum] = signal.signal(signum, interrupted)
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker", "--authority", str(path)], start_new_session=True)
            budget_contract = contract if mode == "execute" else {**contract, "limits": {**contract["limits"], "output_bytes_per_host": PROBE_POLICY["output_bytes"]}}
            budget = old.Budget(budget_contract, output)
            while True:
                budget()
                if mode == "probe" and time.monotonic()-began >= PROBE_POLICY["seconds"]:
                    raise TimeoutError("Compatibility probe ceiling reached")
                try:
                    code = process.wait(timeout=min(1., max(.001, budget.deadline_mono-time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    pass
            require(code == 0, f"Transfer {mode} failed with exit {code}; no retry")
            result = read_json(output / ("probe_receipt.json" if mode == "probe" else "transfer_receipt.json"))
            require(result["status"] == ("passed" if mode == "probe" else "host_complete"), "Missing successful worker receipt")
            return result
        except BaseException as error:
            old.stop_owned_process_group(process)
            previous = read_json(output / "status.json") if (output / "status.json").exists() else {}
            write_json(output / "status.json", {**previous, "status": "timeout" if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) else "failed",
                        "error": str(error), "automatic_retry": False, "deadline_unix": DEADLINE})
            raise
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def child_main(path):
    authority = read_json(path)
    contract, role, old, frozen = _authority(authority)
    require(Path(path).resolve() == Path(authority["output"]).resolve() / "transfer_authority.json", "Wrong child authority path")
    os.kill(authority["supervisor_pid"], 0)
    require(os.getppid() == authority["supervisor_pid"] and os.getpgid(0) == os.getpid(), "Worker ownership mismatch")
    owner = Path(authority["output"]) / "worker_owner.json"
    with owner.open("x") as handle:
        json.dump({"pid": os.getpid(), "transfer_contract_sha256": sha_json(contract)}, handle)
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (contract["limits"]["per_file_bytes"], contract["limits"]["per_file_bytes"]))
    mode = authority["mode"]
    if mode == "probe":
        result = _probe(contract, role, authority["handoff"], old, frozen, authority["output"])
        filename = "probe_receipt.json"
    else:
        result = _train(contract, role, authority["handoff"], old, frozen, authority["output"], authority["own_probe"], authority["compatibility"])
        filename = "transfer_receipt.json"
    write_json(Path(authority["output"]) / filename, result)
    write_json(Path(authority["output"]) / "status.json", {k: v for k, v in result.items() if k != "anchors"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "probe", "compare-probes", "execute", "_worker"))
    for name in ("contract", "approval", "handoff", "authority", "own-probe", "compatibility", "left-probe", "right-probe", "result", "old-wrapper"):
        parser.add_argument("--" + name)
    parser.add_argument("--role", choices=tuple(ROLES))
    args = parser.parse_args(argv)
    if args.mode == "_worker":
        child_main(args.authority)
        return 0
    contract = read_json(args.contract)
    if args.mode == "compare-probes":
        old = load_old(args.old_wrapper)
        validate_contract(contract, old, wrapper_path=__file__, now=time.time())
        result = compare_probes(contract, read_json(args.left_probe), read_json(args.right_probe))
        write_json(args.result, result)
        print(json.dumps({"passed": result["passed"], "checks": len(result["checks"])}))
        return 0 if result["passed"] else 1
    old_path = args.old_wrapper or contract["hosts"][args.role]["old_wrapper_path"]
    old = load_old(old_path)
    if args.mode == "validate":
        print(json.dumps(validate_contract(contract, old, wrapper_path=__file__, now=time.time())))
        return 0
    require(args.approval and args.handoff and args.role, "Explicit approval, handoff, and role required")
    result = execute_mode(contract, read_json(args.approval), read_json(args.handoff), args.role, args.mode,
                          own_probe=read_json(args.own_probe) if args.own_probe else None,
                          compatibility=read_json(args.compatibility) if args.compatibility else None)
    print(json.dumps({k: v for k, v in result.items() if k != "anchors"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
