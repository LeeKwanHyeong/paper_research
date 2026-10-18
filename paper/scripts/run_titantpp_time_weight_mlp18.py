"""Fresh MLP16 Cap fits with only the time-loss weight changed; Validation only.

The sealed Cap36 model, input, head, active quantity loss and training engine
are reused unchanged. Hooks are interpreter-local and restored on exit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2] if "paper/scripts" in str(Path(__file__)) else Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts import run_titantpp_mlp_cap36 as inherited

ARMS = ("titantpp_history_mlp_width16",)
TIME_ALPHAS = (.1, 10.)
SEEDS, DATASETS = inherited.SEEDS, inherited.DATASETS
VARIANT, MONITOR = inherited.VARIANT, inherited.MONITOR
INACTIVE_TAIL = inherited.INACTIVE_TAIL
SCHEMA = "titantpp_time_weight_mlp18_stage_v1"
CONTRACT_SCHEMA = "titantpp_time_weight_mlp18_v1"
require, sha_file, sha_json, read = inherited.require, inherited.sha_file, inherited.sha_json, inherited.read
write_exclusive, verify_inventory = inherited.write_exclusive, inherited.verify_inventory
file_inventory, resolve_under = inherited.file_inventory, inherited.resolve_under
population_identity, verify_population = inherited.population_identity, inherited.verify_population
MetricAccumulator, compare_metrics = inherited.MetricAccumulator, inherited.compare_metrics
compare_prediction_files = inherited.compare_prediction_files
build_model = inherited.build_model
fabricated_data = inherited.fabricated_data
_BASE_TRAINING_ARGS = inherited.training_args
_BASE_RECEIPT = inherited._base_receipt
_BASE_VERIFY_TRAIN = inherited.verify_train_receipt
_BASE_LOAD_SELECTED = inherited._load_selected
_BASE_EVALUATE_SELECTED = inherited.evaluate_selected
_ACTIVE_ALPHA = None


def weight_identity(alpha):
    require(type(alpha) in (float, int) and math.isfinite(float(alpha))
        and float(alpha) in (*TIME_ALPHAS, 1.), "Unapproved time-loss weight")
    return {"schema": "titantpp_time_weight_identity_v1", "time_alpha": float(alpha),
        "quantity_lambda": 1., "joint_formula": "time_alpha * unweighted_time_nll + existing_log1p_quantity_mse",
        "quantity_variant": VARIANT, "checkpoint_monitor": MONITOR,
        "input_head_memory_optimizer_budget_unchanged": True}


def select_job(contract, host, job_id):
    require(contract.get("schema") == CONTRACT_SCHEMA, "Wrong time-weight experiment contract")
    job, data, root, folder = inherited.select_job(contract, host, job_id)
    require(job["arm"] in ARMS and type(job.get("time_alpha")) in (int, float)
        and job["time_alpha"] in TIME_ALPHAS, "Only fresh MLP16 alpha0.1/10 conditions are admitted")
    require(data["model"].get("time_head_lr_multiplier") == 1., "Time head learning rate changed")
    require(data["optimizer"] == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.},
        "Original optimizer changed")
    require(not contract["training"].get("warm_start", False), "Warm start is forbidden")
    return job, data, root, folder


def make_weighted_target_outputs(original, alpha):
    identity = weight_identity(alpha)
    def weighted(*args, **kwargs):
        import torch
        require(kwargs.get("lambda_log_qty") == 1., "Quantity loss weight must remain one")
        value = original(*args, **kwargs)
        require(torch.equal(value["quantity_train_loss"], value["log_qty_loss"]),
            "Active quantity objective changed from existing log-MSE")
        result = dict(value)
        result["unweighted_joint_loss"] = value["joint_loss"]
        # Preserve alpha1 bitwise arithmetic, as well as its gradient and RNG path.
        if identity["time_alpha"] != 1.:
            result["joint_loss"] = identity["time_alpha"] * value["time_loss"] + value["quantity_train_loss"]
        result["weighted_time_loss"] = identity["time_alpha"] * value["time_loss"]
        return result
    return weighted


def validate_weight_binding(payload, alpha):
    identity = weight_identity(alpha)
    require(payload.get("time_alpha") == float(alpha)
        and payload.get("time_weight_identity") == identity, "Time-weight top-level binding mismatch")
    interface, resume = payload.get("interface_meta", {}), payload.get("resume_identity", {})
    require(interface.get("time_weight_identity") == identity
        and resume.get("time_weight_identity") == identity
        and resume.get("arguments", {}).get("time_alpha") == float(alpha)
        and resume.get("interface_meta") == interface, "Time-weight interface/resume binding mismatch")
    return identity


def annotate_history_row(row, alpha):
    identity = weight_identity(alpha)
    row["time_alpha"], row["time_weight_identity"] = float(alpha), identity
    for prefix, qty_key in (("train", "train_quantity_loss"), ("val", "val_quantity_train_loss")):
        time_key, joint_key = prefix + "_time_nll", prefix + "_joint_objective"
        if all(k in row for k in (time_key, qty_key, joint_key)):
            unweighted = float(row[time_key]) + float(row[qty_key])
            expected = float(alpha) * float(row[time_key]) + float(row[qty_key])
            require(math.isclose(float(row[joint_key]), expected, rel_tol=1e-7, abs_tol=1e-7),
                "Weighted objective does not match unweighted components")
            row[prefix + "_joint_weighted"] = float(row[joint_key])
            row[prefix + "_joint_unweighted"] = unweighted
    return row


def annotate_payload(payload, alpha):
    """Mutate only the new experiment's payload before its exclusive save."""
    identity = weight_identity(alpha)
    payload["time_alpha"], payload["time_weight_identity"] = float(alpha), identity
    for row in payload.get("history", []):
        annotate_history_row(row, alpha)
    if "interface_meta" in payload or "resume_identity" in payload:
        validate_weight_binding(payload, alpha)
    if all(k in payload for k in ("best_val_time_nll", "best_val_quantity_train_loss", "best_val_joint_objective")):
        payload["best_val_joint_weighted"] = payload["best_val_joint_objective"]
        payload["best_val_joint_unweighted"] = payload["best_val_time_nll"] + payload["best_val_quantity_train_loss"]
        require(math.isclose(payload["best_val_joint_weighted"],
            alpha * payload["best_val_time_nll"] + payload["best_val_quantity_train_loss"], rel_tol=1e-7, abs_tol=1e-7),
            "Summary weighted objective mismatch")
    return payload


def training_args(contract, data, output, arm, *, time_alpha):
    require(arm in ARMS and time_alpha in TIME_ALPHAS, "Unexpected new condition")
    args = _BASE_TRAINING_ARGS(contract, data, output, arm)
    args.time_alpha = float(time_alpha)
    args.execution_role = "fresh_time_weight_mlp18_validation"
    return args


def base_receipt(contract, host, job, stage):
    require(stage in ("train", "validation"), "Held-out execution is forbidden")
    result = _BASE_RECEIPT(contract, host, job, stage)
    result.update(schema=SCHEMA, time_alpha=float(job["time_alpha"]),
        time_weight_identity=weight_identity(job["time_alpha"]), held_out_test_evaluated=False)
    return result


def verify_train_receipt(contract, host, job, root, receipt):
    require(receipt.get("schema") == SCHEMA and receipt.get("time_alpha") == float(job["time_alpha"])
        and receipt.get("time_weight_identity") == weight_identity(job["time_alpha"]), "Training receipt alpha mismatch")
    validate_weight_binding(receipt["summary"], job["time_alpha"])
    _BASE_VERIFY_TRAIN(contract, host, job, root, receipt)


def _load_selected(data, job, train_receipt, root, device):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    require(train_receipt.get("time_alpha") == float(job["time_alpha"])
        and train_receipt.get("time_weight_identity") == weight_identity(job["time_alpha"]), "Selected receipt alpha mismatch")
    path = resolve_under(root, train_receipt["selected_checkpoint"])
    require(sha_file(path) == train_receipt["checkpoint_sha256"], "Selected binary SHA changed")
    cp = torch_load_checkpoint(path, map_location="cpu")
    validate_weight_binding(cp, job["time_alpha"])
    # Existing route, strict tensor load, binary/state/seed/selector checks remain.
    return _BASE_LOAD_SELECTED(data, job, train_receipt, root, device)


def evaluate_selected(data, job, train_receipt, root, frame, split, output, device="cuda:0"):
    require(split == "validation", "Only fixed-checkpoint Validation replay is admitted")
    return _BASE_EVALUATE_SELECTED(data, job, train_receipt, root, frame, split, output, device=device)


@contextmanager
def runtime_hooks(alpha):
    """Patch only process-local references, never sealed source or model tensors."""
    global _ACTIVE_ALPHA
    require(_ACTIVE_ALPHA is None, "Nested/concurrent loss hooks in one interpreter are forbidden")
    identity = weight_identity(alpha)
    inherited.install_hooks()
    from paper.scripts.count_aware_tpp_backbone import core, training
    from paper.scripts import run_multilag_detail_execution as base
    require(training.target_outputs is core.target_outputs, "Unexpected existing objective hook")
    undo = []
    def replace(module, name, value):
        undo.append((module, name, getattr(module, name)))
        setattr(module, name, value)
    original_time_interface, original_resume = base.time_interface, training._resume_identity
    original_atomic, original_json = training.atomic_torch_save, training.save_json
    original_epoch, original_route = training.train_epoch_with_telemetry, training.validate_checkpoint_route
    weighted = make_weighted_target_outputs(core.target_outputs, alpha)
    def time_interface(*args, **kwargs):
        value = original_time_interface(*args, **kwargs)
        value["time_weight_identity"] = dict(identity)
        return value
    def resume_identity(*args, **kwargs):
        value = original_resume(*args, **kwargs)
        value["time_weight_identity"] = dict(identity)
        value["arguments"]["time_alpha"] = float(alpha)
        return value
    def atomic(value, path):
        return original_atomic(annotate_payload(value, alpha), path)
    def save_json(path, value):
        annotated = annotate_payload(value, alpha)
        result = original_json(path, annotated)
        if Path(path).name == "history.json" and annotated.get("history"):
            row = annotated["history"][-1]
            keys = ("epoch", "time_alpha", "train_joint_weighted", "train_joint_unweighted",
                "train_time_nll", "train_quantity_loss", "val_joint_weighted", "val_joint_unweighted",
                "val_time_nll", "val_quantity_train_loss", "val_log_qty_mse", "val_qty_rmse", "val_qty_mae")
            with (Path(path).parent / "time_weight.log").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({key: row[key] for key in keys if key in row}, sort_keys=True, allow_nan=False) + "\n")
        return result
    def train_epoch(*args, **kwargs):
        value = annotate_history_row(original_epoch(*args, **kwargs), alpha)
        print(f"[time-weight] alpha={alpha:g} train_weighted_joint={value['train_joint_weighted']:.8f} "
            f"train_unweighted_joint={value['train_joint_unweighted']:.8f} "
            f"unweighted_time={value['train_time_nll']:.8f} existing_log_mse={value['train_quantity_loss']:.8f}", flush=True)
        return value
    def route(payload, backbone):
        validate_weight_binding(payload, alpha)
        return original_route(payload, backbone)
    _ACTIVE_ALPHA = float(alpha)
    try:
        replace(core, "target_outputs", weighted)
        replace(training, "target_outputs", weighted)
        replace(base, "time_interface", time_interface)
        replace(training, "_resume_identity", resume_identity)
        replace(training, "atomic_torch_save", atomic)
        replace(training, "save_json", save_json)
        replace(training, "train_epoch_with_telemetry", train_epoch)
        replace(training, "validate_checkpoint_route", route)
        replace(inherited, "SCHEMA", SCHEMA)
        replace(inherited, "training_args", lambda c, d, o, a: training_args(c, d, o, a, time_alpha=alpha))
        replace(inherited, "_base_receipt", base_receipt)
        replace(inherited, "verify_train_receipt", verify_train_receipt)
        replace(inherited, "_load_selected", _load_selected)
        replace(inherited, "evaluate_selected", evaluate_selected)
        yield
    finally:
        for module, name, original in reversed(undo):
            setattr(module, name, original)
        _ACTIVE_ALPHA = None


def run_stage(contract, host, job_id, stage):
    require(stage in ("train", "validation"), "New Test execution is forbidden")
    job, data, root, folder = select_job(contract, host, job_id)
    if stage == "train":
        require(not (folder / "runs").exists(), "Fresh fit required; reuse/resume is forbidden")
    with runtime_hooks(job["time_alpha"]):
        return inherited.run_stage(contract, host, job_id, stage)


def synthetic_check(contract=None, host=None, *, device=None):
    """Fabricated data only; all admitted weights, original head and full shape."""
    results = {}
    for alpha in TIME_ALPHAS:
        with runtime_hooks(alpha):
            # Only the fabricated admission loop needs narrowing. The inherited
            # role_for indexes its original four-arm tuple during actual fits.
            original_arms = inherited.ARMS
            try:
                inherited.ARMS = ARMS
                results[str(alpha)] = inherited.synthetic_check(contract, host, device=device)
            finally:
                inherited.ARMS = original_arms
    state = [r["initial_state_sha256"] for r in results.values()]
    require(all(item == state[0] for item in state), "Alpha changed initial weights/RNG")
    peaks = {}
    for arm in ARMS:
        entries = [r["cuda_memory_peaks"][arm] for r in results.values()]
        peak = dict(entries[0])
        for key in ("max_memory_reserved_bytes", "max_memory_allocated_bytes"):
            if all(key in item for item in entries):
                peak[key] = max(item[key] for item in entries)
        peak["tested_time_alphas"] = list(TIME_ALPHAS)
        peaks[arm] = peak
    first = next(iter(results.values()))
    return {"status": "passed", "schema": SCHEMA, "device": first["device"], "host": host,
        "arms": list(ARMS), "time_alphas": list(TIME_ALPHAS), "per_alpha": results,
        "cuda_memory_peaks": peaks, "parameter_count": first["parameter_count"],
        "initial_state_sha256": first["initial_state_sha256"],
        "real_data_loaded": False, "held_out_test_evaluated": False,
        "checks": {"same_initial_tensors_and_rng": True, "time_weight_only": True,
            "existing_head_checkpoint_roundtrip": True, "target_blocking_and_batch_independence": True}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--host", required=True, choices=("5080", "5090"))
    parser.add_argument("--job")
    parser.add_argument("--stage", required=True, choices=("train", "validation", "synthetic"))
    args = parser.parse_args(argv)
    contract = read(args.contract)
    if args.stage == "synthetic":
        result = synthetic_check(contract, args.host)
    else:
        require(args.job is not None, "Scientific stage requires --job")
        result = run_stage(contract, args.host, args.job, args.stage)
    print(json.dumps({k: result[k] for k in ("status", "job_id", "stage", "host", "time_alpha", "real_data_loaded") if k in result}, sort_keys=True))


if __name__ == "__main__":
    main()
