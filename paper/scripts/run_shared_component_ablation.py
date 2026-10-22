"""Exclusive stage runner for the locked Shared15 A--D component experiment.

The manager owns process admission and child-exit evidence.  This module only
owns one already-admitted job stage and never retries or overwrites artifacts.
"""
from __future__ import annotations

import argparse
from copy import copy
from contextlib import contextmanager, redirect_stdout
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import torch

from paper.scripts import run_titantpp_mlp_cap36 as common
from paper.scripts import run_titantpp_shared_history_raw9 as raw9


SCHEMA = "shared_component_ablation_v1"
STAGE_SCHEMA = "shared_component_ablation_stage_v1"
ARMS = ("full", "no_prototype", "no_history", "encoder_only")
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
VARIANT = "count_only_log_regression"
MONITOR = "validation_raw_quantity_rmse"
ROOT = Path(__file__).resolve().parents[2]
REPLAY_RTOL = REPLAY_ATOL = 1e-5

require = common.require
read = common.read
sha_file = common.sha_file
sha_json = common.sha_json
write_exclusive = common.write_exclusive
resolve_under = common.resolve_under
file_inventory = common.file_inventory
verify_inventory = common.verify_inventory
population_identity = common.population_identity
verify_population = common.verify_population
compare_prediction_files = common.compare_prediction_files


def _model_module():
    # Deliberately the only component-model import.  No MAC worker/runtime is
    # imported by this execution path.
    from models.TPPs import CountAwareComponentAblation
    return CountAwareComponentAblation


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _host_root(contract, host):
    return Path(contract["hosts"][host]["root"]).resolve()


def _under(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            "Artifact path escapes its isolated host root")
    value = (Path(root).resolve() / relative).resolve()
    require(value.is_relative_to(Path(root).resolve()), "Artifact symlink escaped host root")
    return value


def _validate_contract(contract):
    require(isinstance(contract, dict) and contract.get("schema") == SCHEMA
            and contract.get("scope") == "execution", "Locked component contract required")
    require(contract.get("common_condition") == {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
            "Common input or joint loss changed")
    qualification = contract.get("implementation_qualification", {})
    require(qualification.get("status") == "passed" and qualification.get("training_allowed") is True,
            "Implementation qualification has not passed")
    require(set(contract.get("hosts", {})) == {"5080", "5090"}, "Both approved hosts required")
    expected = {(d, a, s) for d in DATASETS for a in ARMS for s in SEEDS}
    jobs = contract.get("jobs", [])
    require(len(jobs) == 36 and {(j.get("dataset"), j.get("arm"), j.get("seed")) for j in jobs} == expected,
            "Expected complete 36-job matched matrix")
    require(len({j.get("id") for j in jobs}) == 36, "Duplicate job id")
    for job in jobs:
        require(job.get("id") == f"{job['dataset']}__{job['arm']}__seed{job['seed']}", "Job identity changed")
        require(job.get("host") == ("5080" if job["seed"] == 42 else "5090"), "Fixed host assignment changed")
        require(bool(job.get("reuse")) == (job["arm"] == "full" and job["seed"] in (52, 62)),
                "Reuse matrix changed")
    expected_training = {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
                         "batch_size": 128, "monitor": MONITOR}
    require(contract.get("training") == expected_training, "Training budget changed")
    require(contract.get("source", {}).get("files") and _canonical(contract["source"]["files"]) == contract["source"].get("files_sha256"),
            "Frozen source inventory changed")
    require(len(contract.get("input_files", {})) == 9, "Nine frozen input files required")


def select_job(contract, host, job_id):
    """Return the contract-bound job and its exclusive artifact folder."""
    _validate_contract(contract)
    require(host in contract["hosts"], "Unapproved host")
    selected = [job for job in contract["jobs"] if job["id"] == job_id]
    require(len(selected) == 1, "Job id must select exactly one condition")
    job = selected[0]
    require(job["host"] == host and Path(job_id).name == job_id, "Job/host scope mismatch")
    data = [item for item in contract.get("datasets", []) if item.get("dataset_id") == job["dataset"]]
    require(len(data) == 1, "Dataset binding is not unique")
    data = data[0]
    model, loader = data.get("model", {}), data.get("loader", {})
    require(model.get("hidden_dim") == 64 and model.get("lambda_log_qty") == 1.0
            and model.get("lambda_tail") == 0.0 and model.get("time_head_lr_multiplier") == 1.0,
            "Shared head/loss condition changed")
    require(loader.get("batch_size") == 128 and loader.get("num_workers") == 0
            and loader.get("drop_last") is False and loader.get("train_shuffle") is True
            and loader.get("validation_shuffle") is False, "Loader condition changed")
    require(data.get("optimizer") == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.0},
            "Optimizer condition changed")
    require(set(data.get("inherited_data_identity", {}).get("populations", {})) == {"train", "validation"},
            "Frozen split identity missing")
    root = _host_root(contract, host)
    return job, data, root, root / "run" / job_id


def _source_root(contract, host):
    return Path(contract["hosts"][host].get("source_root", contract["hosts"][host]["root"])).resolve()


def _native_runtime(contract, host, device):
    """Bind an admitted CUDA stage to its fixed interpreter, runtime and GPU."""
    if str(device) == "cpu":
        return {"device": "cpu", "fabricated_cpu_only": True}
    from paper.scripts import manage_shared_component_ablation as manager
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    spec = contract["hosts"][host]
    require(Path(sys.executable).resolve() == Path(spec["python"]).resolve(), "Native Python differs")
    for key, value in spec["environment"].items():
        require(os.environ.get(key) == value, "Native environment differs: " + key)
    manager.guard(contract, host, active=(os.getpid(),))
    configure_runtime("cuda:0", threads=spec["runtime_expected"]["threads"])
    identity = runtime_identity("cuda:0")
    require(all(identity.get(key) == value for key, value in spec["runtime_expected"].items()), "Native runtime differs")
    require(identity["gpu"]["uuid"] == spec["gpu_uuid"], "GPU UUID differs")
    return identity


def _check_source(contract, host):
    source = _source_root(contract, host)
    for rel, digest in {**contract["source"]["files"], **contract["input_files"], **contract.get("legacy_files", {})}.items():
        require(sha_file(_under(source, rel)) == digest, "Frozen source/input SHA changed: " + rel)


def _candidate_labels(arm):
    module = _model_module()
    metadata = getattr(module, "metadata", None)
    value = (raw9.metadata(mode="shared") if arm == "full"
             else metadata(hidden_dim=64, arm=arm)) if metadata else {"component_arm": arm}
    return {"candidate_name": arm, "component_arm": arm, "architecture_metadata": value,
            **raw9.input_identity(), "quantity_only": False, "time_nll_is_diagnostic": False}


def _build_model(data, arm):
    built = _model_module().build_model(data, arm)
    if isinstance(built, tuple):
        return built
    return built, getattr(built, "encoder_config", {"component_arm": arm})


@contextmanager
def _engine_hooks(data, arm):
    """Patch only construction; preserve the generic core joint target function."""
    from paper.scripts.count_aware_tpp_backbone import core, training
    before_core, before_training = core.target_outputs, training.target_outputs
    old_build, old_optimizer, old_save, old_json, old_labels, old_validate = (training.build_model, training.build_optimizer,
        training.atomic_torch_save, training.save_json, training.BACKBONE_LABELS, training.validate_training_identity)
    tracked = {}
    def build(*_args, **_kwargs):
        model, encoder = _build_model(data, arm)
        require("tracker" not in tracked, "Duplicate fit model")
        tracked["tracker"] = raw9.JointTimeHeadTracker(model)
        return model, encoder
    def optimizer(model, *args, **kwargs):
        tracker = tracked["tracker"]
        require(tracker.model is model, "Optimizer/model identity changed")
        value = tracker.bind_optimizer(old_optimizer(model, *args, **kwargs))
        require(all(group["lr"] == .001 and group["weight_decay"] == .01 for group in value.param_groups),
                "AdamW common settings changed")
        return value
    def save(payload, path):
        return old_save({**payload, **raw9.input_identity(),
            "model_metadata": _candidate_labels(arm)["architecture_metadata"],
            "time_head_training_verification": tracked["tracker"].proof(payload)}, path)
    def save_json(path, payload):
        if "backbone" in payload:
            payload = {**payload, **raw9.input_identity(),
                       "time_head_training_verification": tracked["tracker"].proof()}
        return old_json(path, payload)
    def joint(*args, **kwargs):
        require(kwargs.get("lambda_log_qty") == 1.0, "Quantity loss coefficient changed")
        output = before_core(*args, **kwargs)
        require(torch.equal(output["quantity_train_loss"], output["log_qty_loss"]), "Log-MSE changed")
        require(torch.equal(output["joint_loss"], output["time_loss"] + output["quantity_train_loss"]),
                "Generic alpha1 joint target changed")
        if "tracker" in tracked:
            tracked["tracker"].joint_observations += 1
        return output
    def validate(args, interface, _backbone, variant):
        # Validate every observed-time/head condition through the maintained
        # generic role using its canonical factory label, then bind the actual
        # component arm only at model construction.
        checked = copy(args)
        checked.model_role = "observed_time_joint_v1"
        old_validate(checked, interface, "titantpp", variant)
    training.build_model, training.build_optimizer = build, optimizer
    training.atomic_torch_save, training.save_json = save, save_json
    training.BACKBONE_LABELS = {**old_labels, arm: "Shared15 " + arm}
    training.validate_training_identity = validate
    core.target_outputs = training.target_outputs = joint
    try:
        yield tracked
    finally:
        training.build_model, training.build_optimizer = old_build, old_optimizer
        training.atomic_torch_save, training.save_json, training.BACKBONE_LABELS = old_save, old_json, old_labels
        training.validate_training_identity = old_validate
        core.target_outputs, training.target_outputs = before_core, before_training
        tracker = tracked.get("tracker")
        if tracker is not None and getattr(tracker, "optimizer", None) is not None:
            tracker.optimizer.step = tracker.original_step


def _training_args(contract, data, output, arm):
    from paper.scripts import run_multilag_detail_execution as base
    args = base.training_args({**contract, "model_role": "observed_time_joint_v1"}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.time_alpha, args.execution_role = 1.0, "shared_component_ablation_joint"
    for key, value in {**common.INACTIVE_TAIL, **raw9.input_identity()}.items():
        setattr(args, key, value)
    return args


def _training_interface(contract, data, frame, arm):
    from paper.scripts import run_multilag_detail_execution as base
    interface = base.time_interface(data, frame, contract)
    interface.update(raw9.input_identity())
    interface["backbone_design"] = {"schema": SCHEMA, "arm": arm,
        "training": contract["training"], "component_metadata": _candidate_labels(arm)["architecture_metadata"]}
    return interface


def _receipt_base(contract, host, job, stage):
    return {"schema": STAGE_SCHEMA, "status": "complete", "stage": stage, "host": host,
        "job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
        "contract_sha256": _canonical(contract), "source_files_sha256": contract["source"]["files_sha256"],
        "held_out_test_evaluated": stage == "test", **_candidate_labels(job["arm"])}


def _selected_path(root, receipt):
    return _under(root, receipt["selected_checkpoint"])


def _verify_reuse(contract, data, job, root):
    reuse = job["reuse"]
    required = ("checkpoint", "train_receipt", "validation_receipt", "history", "old_train_exit", "old_validation_exit")
    require(all(key in reuse for key in required), "Reuse provenance incomplete")
    for item in reuse.values():
        if isinstance(item, dict) and "path" in item:
            path = _under(_source_root(contract, job["host"]), item["path"])
            require(sha_file(path) == item["sha256"], "Legacy evidence changed: " + item["path"])
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    checkpoint = torch_load_checkpoint(_under(_source_root(contract, job["host"]), reuse["checkpoint"]["path"]), map_location="cpu")
    require(checkpoint.get("seed") == job["seed"] and checkpoint.get("best_epoch") == reuse["selected_epoch"],
            "Reused selected epoch/seed changed")
    require(canonical_state_dict_sha256(checkpoint["model_state_dict"]) == reuse["canonical_tensor_sha256"],
            "Reused checkpoint tensor changed")
    history = read(_under(_source_root(contract, job["host"]), reuse["history"]["path"]))["history"]
    require(raw9.prepared.select_best_epoch(history) == reuse["selected_epoch"], "Legacy strict-minimum selection changed")
    return checkpoint


def load_selected(contract, host, job_id, root=None, *, receipt=None):
    """Public cost-profiler API; validates selected checkpoint identity before load."""
    job, data, host_root, folder = select_job(contract, host, job_id)
    root = Path(root or host_root)
    receipt = read(folder / "train_receipt.json") if receipt is None else receipt
    checkpoint_path = _selected_path(root, receipt)
    require(sha_file(checkpoint_path) == receipt["checkpoint_sha256"], "Selected checkpoint bytes changed")
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    payload = torch_load_checkpoint(checkpoint_path, map_location="cpu")
    state_sha = canonical_state_dict_sha256(payload["model_state_dict"])
    require(state_sha == receipt["state_sha256"] == payload.get("model_state_sha256"), "Selected tensor identity changed")
    return {"job": job, "data": data, "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": receipt["checkpoint_sha256"], "checkpoint_payload": payload,
            "selected_epoch": receipt["selected_epoch"], "training_origin": receipt["training_origin"],
            "provenance": receipt.get("reuse")}


def _load_model(contract, data, job, receipt, root, device):
    selected = load_selected(contract, receipt["host"], job["id"], root, receipt=receipt)
    model, encoder = _build_model(data, job["arm"])
    payload = selected["checkpoint_payload"]
    if job["arm"] == "full" and receipt["training_origin"] == "reused_verified_fit":
        _model_module().validate_checkpoint(payload, "full")
    elif job["arm"] != "full":
        _model_module().validate_checkpoint(payload, job["arm"])
    require(payload.get("seed") == job["seed"] and payload.get("best_epoch") == receipt["selected_epoch"],
            "Selected checkpoint metadata changed")
    require(payload.get("checkpoint_monitor") == MONITOR
            and payload.get("encoder_config") == encoder, "Selected checkpoint selection/encoder changed")
    if job["arm"] == "full" and receipt["training_origin"] == "fresh_fit":
        require(payload.get("backbone") == "full"
                and payload.get("resume_identity", {}).get("arguments", {}).get("model_role") == "observed_time_joint_v1"
                and payload.get("interface_meta", {}).get("source_files_sha256") == contract["source"]["files_sha256"],
                "Fresh full checkpoint neutral provenance changed")
    proof = payload.get("time_head_training_verification")
    require(isinstance(proof, dict), "Selected time-head proof missing")
    raw9.verify_time_head_proof(proof, payload["model_state_dict"])
    if receipt["training_origin"] == "fresh_fit":
        raw9.verify_checkpoint_fit_proof(proof, receipt["time_head_training_verification"])
    history_path = (root / receipt.get("reused_history_copy", "")).resolve() if receipt.get("reused_history_copy") else selected["checkpoint_path"].parent / "history.json"
    history = read(history_path)["history"]
    require(raw9.prepared.select_best_epoch(history) == receipt["selected_epoch"] == payload["best_epoch"],
            "Selected checkpoint is not the first strict Validation minimum")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), selected


def _metrics(cutoff):
    return raw9.MetricAccumulator(cutoff)


def _evaluation_frame(data, split):
    from paper.scripts import run_multilag_detail_execution as base
    if split == "validation":
        return base.prepare_admitted_data(data)[0]
    require(split == "test", "Unsupported evaluation split")
    import polars as pl
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    full = data["full_data_identity"]["data"]
    require(sha_file(full["path"]) == full["sha256"], "Frozen full input changed")
    full_frame = prepare_count_frame(pl.read_parquet(full["path"]).sort(["oper_part_no", "seq"]))
    admitted = base.prepare_admitted_data(data)[0]
    require(set(admitted.columns) <= set(full_frame.columns)
            and full_frame.filter(pl.col("chronological_split").is_in(["train", "validation"]))
            .select(admitted.columns).equals(admitted), "Full input Train/Validation parent changed")
    return full_frame


def evaluate_selected(contract, data, job, receipt, root, frame, split, output, device="cuda:0"):
    import numpy as np
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    model, selected = _load_model(contract, data, job, receipt, root, device)
    loader = make_loader(frame, target_split=split, shuffle=False, generator=None,
        **{key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")})
    truth, identity = population_identity(loader.dataset, split)
    identities = data["inherited_data_identity"] if split == "validation" else data["full_data_identity"]
    verify_population(identity, identities["populations"][split])
    accumulator, processed = _metrics(data["quantity_boundaries_all_train_rows"][-1]), 0
    with Path(output).open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as zipped:
        with torch.no_grad():
            for _, dts, mask, part_ids, quantities in loader:
                value = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.0)
                arrays = {key: value[key].detach().cpu().numpy() for key in ("true_qty", "pred_qty", "time_loss", "history_length")}
                size = len(arrays["true_qty"])
                require(np.array_equal(arrays["true_qty"].astype(np.float64), truth[processed:processed + size]), "Target truth changed")
                accumulator.add(arrays["true_qty"], arrays["pred_qty"], arrays["time_loss"])
                for index in range(size):
                    part, end = loader.dataset.index[processed + index]
                    require(int(part_ids[index]) == int(part), "Target ordering changed")
                    row = {"target_index": processed + index, "entity_id": str(loader.dataset.parts[part]),
                        "target_seq": int(loader.dataset.seq_lists[part][end + 1]), "split": split,
                        "raw_quantity": float(arrays["true_qty"][index]), "predicted_raw_quantity": float(arrays["pred_qty"][index]),
                        "time_nll": float(arrays["time_loss"][index]), "history_length": int(arrays["history_length"][index])}
                    zipped.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
                processed += size
    require(processed == identity["target_count"], "Incomplete evaluation population")
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    final_state = canonical_state_dict_sha256(model.state_dict())
    require(final_state == selected["checkpoint_payload"]["model_state_sha256"], "Evaluation changed model state")
    return {"metrics": accumulator.finish(), "target_identity": identity, "prediction_path": str(Path(output).resolve()),
            "prediction_rows": processed, "parameters_unchanged": True, "state_sha256": final_state}


def _common_initialization(data, seed):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        model, _ = _build_model(data, arm)
        states[arm] = model.state_dict()
    names = set.intersection(*(set(value) for value in states.values()))
    shared = {}
    for name in sorted(names):
        first = states[ARMS[0]][name]
        require(all(torch.equal(first, states[arm][name]) for arm in ARMS[1:]),
                "Common initial tensor differs across arms: " + name)
        shared[name] = canonical_state_dict_sha256({"tensor": first})
    require(bool(shared), "No common initialized tensors")
    return {"seed": seed, "common_tensor_count": len(shared), "common_tensor_digests": shared,
            "common_initialization_exact": True}


def _write_reference(contract, host, job, root, folder, receipt, frame, device):
    target = folder / "training_validation_predictions.jsonl.gz"
    reference = evaluate_selected(contract, *select_job(contract, host, job["id"])[1:2], job, receipt, root, frame,
                                  "validation", target, device=device)
    receipt["validation_reference"] = reference
    return reference


def _stage_train(contract, host, job, data, root, folder, device):
    from paper.scripts import manage_shared_component_ablation as manager
    manager.verify_native_qualification(contract, host, root)
    runtime_before = _native_runtime(contract, host, device)
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.count_aware_tpp_backbone import training
    frame, input_receipt = base.prepare_admitted_data(data)
    write_exclusive(folder / "input_receipt.json", input_receipt)
    if job.get("reuse"):
        payload = _verify_reuse(contract, data, job, root)
        source = _under(_source_root(contract, host), job["reuse"]["checkpoint"]["path"])
        selected_copy = folder / "reused_selected.pt"
        history_source = _under(_source_root(contract, host), job["reuse"]["history"]["path"])
        history_copy = folder / "reused_history.json"
        require(not selected_copy.exists() and not history_copy.exists(), "Reuse evidence already exists")
        shutil.copyfile(source, selected_copy)
        shutil.copyfile(history_source, history_copy)
        require(sha_file(selected_copy) == sha_file(source)
                and sha_file(history_copy) == sha_file(history_source), "Reuse byte copy changed")
        receipt = {**_receipt_base(contract, host, job, "train"), "training_origin": "reused_verified_fit",
            "new_training_epochs": 0, "selected_checkpoint": str(selected_copy.relative_to(root)),
            "selected_epoch": job["reuse"]["selected_epoch"], "checkpoint_sha256": sha_file(selected_copy),
            "state_sha256": job["reuse"]["canonical_tensor_sha256"], "reuse": job["reuse"],
            "reused_checkpoint_copy": str(selected_copy.relative_to(root)),
            "reused_history_copy": str(history_copy.relative_to(root)),
            "summary": {"status": "reused_verified_fit", "best_epoch": job["reuse"]["selected_epoch"]},
            "common_initialization": _common_initialization(data, job["seed"])}
    else:
        with _engine_hooks(data, job["arm"]) as tracked:
            args = _training_args(contract, data, folder, job["arm"])
            args.device = device
            interface = _training_interface(contract, data, frame, job["arm"])
            quantity = {"boundaries": data["quantity_boundaries_all_train_rows"], "strata": []}
            summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
                interface_meta=interface, backbone=job["arm"], quantity_variant=VARIANT, seed=job["seed"])
            proof = tracked["tracker"].proof()
        selected = Path(summary["checkpoint_path"])
        receipt = {**_receipt_base(contract, host, job, "train"), "training_origin": "fresh_fit",
            "new_training_epochs": summary["completed_epochs"], "selected_checkpoint": str(selected.resolve().relative_to(root)),
            "selected_epoch": summary["best_epoch"], "checkpoint_sha256": sha_file(selected),
            "state_sha256": summary["checkpoint_state_sha256"], "summary": {**summary, "time_head_training_verification": proof},
            "time_head_training_verification": proof, "common_initialization": _common_initialization(data, job["seed"])}
    reference = evaluate_selected(contract, data, job, receipt, root, frame, "validation",
                                  folder / "training_validation_predictions.jsonl.gz", device=device)
    receipt["validation_reference"] = reference
    files = [folder / "input_receipt.json", Path(reference["prediction_path"])]
    if job.get("reuse"):
        files.extend((folder / "reused_selected.pt", folder / "reused_history.json"))
    if not job.get("reuse"):
        files.extend(path for path in (folder / "runs").rglob("*") if path.is_file())
    receipt["files"] = file_inventory(root, files)
    receipt["runtime_before"] = runtime_before
    receipt["runtime_after"] = _native_runtime(contract, host, device)
    return receipt


def _stage_replay(contract, host, job, data, root, folder, stage, device):
    runtime_before = _native_runtime(contract, host, device)
    train = read(folder / "train_receipt.json")
    require(train.get("training_origin") in ("fresh_fit", "reused_verified_fit"), "Train origin changed")
    frame = _evaluation_frame(data, stage)
    prediction = folder / f"{stage}_predictions.jsonl.gz"
    result = evaluate_selected(contract, data, job, train, root, frame, stage, prediction, device=device)
    if stage == "validation":
        require(result["target_identity"] == train["validation_reference"]["target_identity"], "Validation population changed")
        raw9.compare_metrics(result["metrics"], train["validation_reference"]["metrics"])
        compare_prediction_files(prediction, train["validation_reference"]["prediction_path"])
        result["validation_replay_passed"] = True
        result["individual_predictions_match"] = True
    return {**_receipt_base(contract, host, job, stage), **result,
        "training_origin": train["training_origin"], "selected_checkpoint": train["selected_checkpoint"],
        "selected_epoch": train["selected_epoch"], "checkpoint_sha256": train["checkpoint_sha256"],
        "state_sha256": train["state_sha256"], "runtime_before": runtime_before,
        "runtime_after": _native_runtime(contract, host, device),
        "files": file_inventory(root, [prediction, folder / "train_receipt.json"])}


def run_stage(contract, host, job_id, stage, *, device="cuda:0"):
    require(stage in ("synthetic", "train", "reuse", "validation", "test"), "Unsupported stage")
    job, data, root, folder = select_job(contract, host, job_id)
    _check_source(contract, host)
    require((stage == "reuse") == bool(job.get("reuse")) if stage in ("train", "reuse") else True,
            "Use reuse stage only for verified reused fits")
    if stage == "synthetic":
        return synthetic_check(contract, host, job_id, device=device)
    if stage == "test":
        from paper.scripts import manage_shared_component_ablation as manager
        manager.verify_global_gate(contract, root)
    folder.mkdir(parents=True, exist_ok=True)
    receipt_stage = "train" if stage == "reuse" else stage
    receipt_path = folder / f"{receipt_stage}_receipt.json"
    require(not receipt_path.exists(), "Completed stage cannot be retried or overwritten")
    write_exclusive(folder / f"{stage}.intent.json", {"contract_sha256": _canonical(contract), "job_id": job_id,
        "stage": stage, "started_unix": time.time()})
    if stage in ("train", "reuse"):
        receipt = _stage_train(contract, host, job, data, root, folder, device)
    else:
        receipt = _stage_replay(contract, host, job, data, root, folder, stage, device)
    write_exclusive(receipt_path, receipt)
    return receipt


def synthetic_check(contract, host, job_id, *, device="cpu"):
    job, data, _root, _folder = select_job(contract, host, job_id)
    from paper.scripts.count_aware_tpp_backbone import core
    torch.manual_seed(job["seed"])
    model, _ = _build_model(data, job["arm"])
    model = model.to(device).train()
    optimizer = raw9.JointTimeHeadTracker(model).bind_optimizer(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01))
    dt = torch.ones((2, min(8, data["loader"]["max_seq_len"])), device=device)
    qty = torch.ones_like(dt)
    mask = torch.ones_like(dt, dtype=torch.bool)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = core.target_outputs(model, dt, mask, qty, lambda_log_qty=1.0)
        output["joint_loss"].mean().backward(); optimizer.step()
    return {"schema": STAGE_SCHEMA, "status": "qualified", "host": host, "job_id": job_id,
        "arm": job["arm"], "dataset": job["dataset"], "contract_sha256": _canonical(contract),
        "source_files_sha256": contract["source"]["files_sha256"], "device": str(device), "real_data_loaded": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--host", choices=("5080", "5090"), required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--stage", choices=("synthetic", "train", "reuse", "validation", "test"), required=True)
    args = parser.parse_args(argv)
    raw = args.contract.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == args.contract_sha, "Raw contract SHA differs")
    contract = json.loads(raw)
    with redirect_stdout(sys.stderr):
        result = run_stage(contract, args.host, args.job, args.stage)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
