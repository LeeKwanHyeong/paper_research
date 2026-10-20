"""Fresh original-data MAC accuracy fits under the final Shared15 conditions.

A separate 5090-only operation preserves every historical worker and artifact.
Train loads Train/Validation only. Test uses frozen selected checkpoints after
all 9 native Validation replays and actual successful child exits are sealed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import gc
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time

import torch

from paper.scripts import run_titantpp_shared_history_cap as prepared
from paper.scripts import run_titantpp_mlp_cap36 as metrics_parent
from paper.scripts import run_titantpp_shared_history_raw9 as frozen_raw9


SCHEMA = "titans_mac_accuracy9_stage_v1"
CONTRACT_SCHEMA = "titans_mac_accuracy9_v1"
ARM = "titantpp_titans_mac"
ARMS = (ARM,)
DATASETS = ("raf_spare_parts", "yellow_trip_hourly", "intermittent_frozen_5000")
SEEDS = (42, 52, 62)
from paper.scripts import titans_mac_observed_time_runtime as mac_runtime
MAC_RUNTIME_SHA256 = "36ec340041db9b3ea7d2335d30719cfb5c0e1255c72aee4935710cde673082d1"
MAC_CONTRACT_SHA256 = "ac80db7e4190c447ae2a3a8e4c33574886cc7edfca21a2fa272075914e926922"
FROZEN_RAW9_SHA256 = "100de1166d6e59dad906e0df21d3baadddc90d7834e948676cc124b822effb1c"
VARIANT, MONITOR = prepared.VARIANT, prepared.MONITOR
ROOT = Path(__file__).resolve().parents[2]
PREPARATION_WORKER_SHA256 = "508cd9f26d75c911f6c5c10f341679c271c4a2cea5846206dcae6bcc66b5e55f"
MODEL_SHA256 = "eee18d3abe199f0c71d80da35acf01cf19acd8f56291ff7cfd33fd86d8bc82a2"
REPLAY_RTOL = REPLAY_ATOL = 1e-5
TIME_HEAD_NAMES = ("v_t.weight", "b_t", "w_raw", "time_scale_weight.weight")
INACTIVE_TAIL = metrics_parent.INACTIVE_TAIL
require, read, sha_file, sha_json = (
    metrics_parent.require, metrics_parent.read, metrics_parent.sha_file, metrics_parent.sha_json)
write_exclusive, resolve_under = metrics_parent.write_exclusive, metrics_parent.resolve_under
file_inventory, verify_inventory = metrics_parent.file_inventory, metrics_parent.verify_inventory
population_identity, verify_population = metrics_parent.population_identity, metrics_parent.verify_population
compare_prediction_files = metrics_parent.compare_prediction_files


def verify_frozen_dependencies():
    require(sha_file(Path(frozen_raw9.__file__)) == FROZEN_RAW9_SHA256,
            "Original raw9 worker changed")
    require(sha_file(Path(prepared.__file__)) == PREPARATION_WORKER_SHA256,
            "Immutable preparation worker changed")
    require(sha_file(ROOT / "models/TPPs/CountAwareTitanSharedHistoryMLP.py") == MODEL_SHA256,
            "Immutable shared-history model changed")
    require(sha_file(Path(mac_runtime.__file__)) == MAC_RUNTIME_SHA256
            and sha_file(ROOT / "paper/scripts/titans_mac_observed_time_contract.py") == MAC_CONTRACT_SHA256,
            "Frozen optimized MAC construction changed")


def input_identity():
    return {"time_input": "log1p", "time_input_transform": prepared.INPUT_TRANSFORM,
            "quantity_input_transform": prepared.INPUT_TRANSFORM, "time_alpha": 1.0,
            "loss_mode": "log_mse", "quantity_loss_normalization": "none",
            "quantity_head_family": "unchanged_softplus_log1p_location_then_expm1",
            "time_loss_graph_excluded": False, "time_head_trained": True,
            "data_variant": "original_uncapped", "quantity_capping_applied": False}


def candidate_labels(arm):
    require(arm in ARMS, "Only the approved MAC model is admitted")
    from models.Titan.common.titans_mac_optimized import optimization_metadata
    return {"candidate_name": arm, **input_identity(), "quantity_only": False,
        "time_nll_is_diagnostic": False, "stored_baselines_retrained_or_reevaluated": False,
        "model_role": role_for(arm), "titans_memory_gradient_clip": 1.0,
        "architecture_source": "byte_identical_efficiency_benchmark_mac_factory",
        "mac_execution_optimization": optimization_metadata()}


def role_for(arm):
    require(arm in ARMS, "Only the approved MAC model is admitted")
    return "observed_time_joint_v1"


def build_model(data, arm=ARM):
    require(arm in ARMS, "Only the approved MAC model is admitted")
    return mac_runtime.build_model(data, arm)


@contextmanager
def candidate_factory(arm=None):
    """Scope MAC admission to this worker; restore shared routing even on failure."""
    require(arm is None or arm in ARMS, "Only the approved MAC model is admitted")
    from paper.scripts.count_aware_tpp_backbone import observed_time, training
    original_backbones, original_build = observed_time.BACKBONES, training.build_model
    observed_time.BACKBONES = (*original_backbones, *[a for a in ARMS if a not in original_backbones])
    training.build_model = mac_runtime.factory
    try:
        yield
    finally:
        training.build_model, observed_time.BACKBONES = original_build, original_backbones


def verify_common_head_initialization(data, seed=42):
    """Compare exact six output tensors to the frozen final Shared15 constructor."""
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    keys = ("b_t", "w_raw", "v_t.weight", "time_scale_weight.weight",
            "quantity_head.weight", "quantity_head.bias")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        mac, _ = build_model(data, ARM)
        torch.manual_seed(seed)
        shared, _ = frozen_raw9.build_model(data, "titantpp_history_mlp_shared15_v1")
        a, b = mac.state_dict(), shared.state_dict()
        require(all(torch.equal(a[key], b[key]) for key in keys),
                "Common quantity/time head initialization differs from final Shared15")
        require(mac.time_head_contract() == shared.time_head_contract(),
                "Common quantity/time head likelihood contract differs")
        return {"status": "passed", "seed": seed, "parameter_names": list(keys),
                "common_head_sha256": canonical_state_dict_sha256({key: a[key] for key in keys}),
                "shared15_common_head_exact": True}


def _time_head_sha(state, names):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(all(name in state for name in names), "Missing time-head tensor")
    return canonical_state_dict_sha256({name: state[name] for name in names})


class JointTimeHeadTracker:
    """Observe actual joint optimizer updates without suppressing any gradient."""
    def __init__(self, model):
        self.model, self.optimizer = model, None
        self.names = tuple(name for name, _ in model.time_head_named_parameters())
        require(self.names == TIME_HEAD_NAMES, "Invalid time-head inventory")
        self.initial_sha = _time_head_sha(model.state_dict(), self.names)
        self.steps, self.finite_gradient, self.nonzero_gradient = 0, False, False
        self.joint_observations = 0

    def bind_optimizer(self, optimizer):
        require(self.optimizer is None, "Duplicate optimizer binding")
        self.entries = tuple(self.model.time_head_named_parameters())
        require(tuple(name for name, _ in self.entries) == self.names, "Time-head identity changed")
        parameter_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        require(all(id(p) in parameter_ids for _, p in self.entries), "Optimizer omitted time head")
        trainable_ids = {id(p) for p in self.model.parameters() if p.requires_grad}
        require(parameter_ids == trainable_ids, "Joint optimizer omitted or added model parameters")
        self.trainable_names = tuple(name for name, p in self.model.named_parameters() if p.requires_grad)
        self.optimizer, self.original_step = optimizer, optimizer.step
        def step(*args, **kwargs):
            gradients = [parameter.grad for _, parameter in self.entries]
            require(any(g is not None for g in gradients), "Joint time-head gradients absent")
            require(all(g is None or bool(torch.isfinite(g).all()) for g in gradients),
                    "Nonfinite time-head gradient")
            self.finite_gradient = True
            self.nonzero_gradient |= any(g is not None and bool(torch.any(g != 0)) for g in gradients)
            value = self.original_step(*args, **kwargs)
            self.steps += 1
            return value
        optimizer.step = step
        return optimizer

    def proof(self, payload=None):
        require(self.optimizer is not None and self.steps > 0 and self.joint_observations > 0,
                "No actual joint optimizer updates verified")
        state = self.model.state_dict() if payload is None else payload.get("model_state_dict", self.model.state_dict())
        final_sha = _time_head_sha(self.model.state_dict(), self.names)
        checkpoint_sha = _time_head_sha(state, self.names)
        require(self.finite_gradient and self.nonzero_gradient, "No finite nonzero time-head gradient observed")
        require(all(p in self.optimizer.state for _, p in self.entries), "Missing time-head optimizer state")
        require(final_sha != self.initial_sha and checkpoint_sha != self.initial_sha,
                "Joint time-head parameters did not change")
        return {"schema": "time_head_joint_training_verification_v1",
            "initial_time_head_sha256": self.initial_sha, "final_time_head_sha256": final_sha,
            "checkpoint_time_head_sha256": checkpoint_sha, "time_head_parameter_names": list(self.names),
            "optimizer_steps_checked": self.steps, "joint_loss_observations": self.joint_observations,
            "finite_time_head_gradient_observed": True, "nonzero_time_head_gradient_observed": True,
            "time_head_parameters_changed": True, "optimizer_state_present": True,
            "joint_loss_includes_time": True, "time_alpha": 1.0,
            "all_trainable_parameters_in_optimizer": True,
            "trainable_parameter_names": list(self.trainable_names)}


def verify_time_head_proof(proof, state=None):
    require(isinstance(proof, dict) and proof.get("schema") == "time_head_joint_training_verification_v1",
            "Missing actual joint time-head verification")
    for name in ("initial_time_head_sha256", "final_time_head_sha256", "checkpoint_time_head_sha256"):
        value = proof.get(name)
        require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                "Invalid joint time-head digest")
    require(proof["initial_time_head_sha256"] != proof["final_time_head_sha256"]
            and proof["initial_time_head_sha256"] != proof["checkpoint_time_head_sha256"]
            and type(proof.get("optimizer_steps_checked")) is int and proof["optimizer_steps_checked"] > 0
            and type(proof.get("joint_loss_observations")) is int and proof["joint_loss_observations"] > 0
            and proof.get("time_alpha") == 1.0
            and proof.get("time_head_parameter_names") == list(TIME_HEAD_NAMES)
            and all(proof.get(key) is True for key in ("finite_time_head_gradient_observed",
                "nonzero_time_head_gradient_observed", "time_head_parameters_changed",
                "optimizer_state_present", "joint_loss_includes_time")), "Joint time-head proof mismatch")
    require(proof.get("all_trainable_parameters_in_optimizer") is True
            and isinstance(proof.get("trainable_parameter_names"), list)
            and set(TIME_HEAD_NAMES) <= set(proof["trainable_parameter_names"]),
            "Missing complete joint optimizer inventory")
    if state is not None:
        require(_time_head_sha(state, proof["time_head_parameter_names"]) == proof["checkpoint_time_head_sha256"],
                "Selected time-head state differs from proof")
    return proof


def verify_checkpoint_fit_proof(checkpoint_proof, fit_proof):
    verify_time_head_proof(checkpoint_proof)
    verify_time_head_proof(fit_proof)
    require(checkpoint_proof["initial_time_head_sha256"] == fit_proof["initial_time_head_sha256"]
            and checkpoint_proof["checkpoint_time_head_sha256"] == fit_proof["checkpoint_time_head_sha256"]
            and checkpoint_proof["optimizer_steps_checked"] <= fit_proof["optimizer_steps_checked"],
            "Checkpoint/fit joint time-head verification mismatch")


@contextmanager
def runtime_hooks(arm=None, *, tracked=None):
    from paper.scripts.count_aware_tpp_backbone import core, training
    with candidate_factory(arm):
        originals = core.target_outputs, training.target_outputs
        def joint(*args, **kwargs):
            require(kwargs.get("lambda_log_qty") == 1.0, "Quantity coefficient changed")
            value = originals[0](*args, **kwargs)
            require(torch.equal(value["quantity_train_loss"], value["log_qty_loss"]), "Log-MSE loss changed")
            require(torch.equal(value["joint_loss"], value["time_loss"] + value["quantity_train_loss"]),
                    "Original alpha1 joint objective changed")
            if tracked is not None and "tracker" in tracked:
                tracked["tracker"].joint_observations += 1
            return value
        core.target_outputs = training.target_outputs = joint
        try:
            yield
        finally:
            core.target_outputs, training.target_outputs = originals


@contextmanager
def _engine_hooks(arm, *, contract=None, data=None):
    from paper.scripts.count_aware_tpp_backbone import training
    tracked = {}
    with runtime_hooks(arm, tracked=tracked):
        old_build, old_optimizer = training.build_model, training.build_optimizer
        old_save, old_json = training.atomic_torch_save, training.save_json
        def build(*args, **kwargs):
            model, meta = old_build(*args, **kwargs)
            if contract is not None:
                verify_model_conditions(contract, data, arm, model, meta)
            require("tracker" not in tracked, "Duplicate joint fit model")
            tracked["tracker"] = JointTimeHeadTracker(model)
            return model, meta
        def optimizer(model, *args, **kwargs):
            tracker = tracked["tracker"]
            require(tracker.model is model, "Optimizer/model binding changed")
            result = tracker.bind_optimizer(old_optimizer(model, *args, **kwargs))
            require(all(group["lr"] == .001 and group["weight_decay"] == .01 for group in result.param_groups),
                    "Common AdamW settings changed")
            return result
        def save(payload, path):
            return old_save({**payload, **input_identity(),
                "time_head_training_verification": tracked["tracker"].proof(payload)}, path)
        def save_json(path, payload):
            if "backbone" in payload:
                payload = {**payload, **input_identity(),
                    "time_head_training_verification": tracked["tracker"].proof()}
            return old_json(path, payload)
        training.build_model, training.build_optimizer = build, optimizer
        training.atomic_torch_save, training.save_json = save, save_json
        try:
            yield tracked
        finally:
            training.build_model, training.build_optimizer = old_build, old_optimizer
            training.atomic_torch_save, training.save_json = old_save, old_json
            tracker = tracked.get("tracker")
            if tracker is not None and tracker.optimizer is not None:
                tracker.optimizer.step = tracker.original_step


class MetricAccumulator:
    """Uncapped raw quantity errors and strict frozen large-quantity tail; no upper clipping."""
    def __init__(self, threshold):
        self.threshold = float(threshold)
        require(math.isfinite(self.threshold) and self.threshold > 0, "Invalid frozen Train tail threshold")
        self.values = {key: [0, 0., 0., 0., 0.] for key in ("overall", "tail")}

    def add(self, truth, prediction, temporal):
        import numpy as np
        q, p, t = [np.asarray(x, dtype=np.float64).reshape(-1) for x in (truth, prediction, temporal)]
        require(q.shape == p.shape == t.shape and all(np.isfinite(x).all() for x in (q, p, t))
                and bool((q >= 0).all()), "Nonfinite, negative or mismatched raw metric rows")
        for key, mask in (("overall", np.ones(q.shape, dtype=bool)), ("tail", q > self.threshold)):
            error, total = p[mask] - q[mask], self.values[key]
            total[0] += int(mask.sum())
            total[1] += float(np.abs(error).sum())
            total[2] += float(np.square(error).sum())
            total[3] += float(t[mask].sum())
            total[4] += float(error.sum())

    def finish(self):
        def one(total):
            n, ae, se, temporal, signed = total
            return {"count": n, "qty_mae": ae/n if n else None,
                "qty_rmse": math.sqrt(se/n) if n else None, "time_nll": temporal/n if n else None,
                "qty_bias": signed/n if n else None, "qty_sse": se, "qty_absolute_error_sum": ae,
                "time_nll_sum": temporal, "status": "available" if n else "N/A_empty"}
        require(self.values["overall"][0] > 0, "No evaluated targets")
        return {**{key: one(value) for key, value in self.values.items()},
                "tail_rule": "q > frozen_Train_large_quantity_threshold", "tail_threshold": self.threshold,
                "time_metric": "recorded_positive_integer_time_nll"}


def compare_metrics(actual, expected):
    require(actual.get("tail_threshold") == expected.get("tail_threshold")
            and actual.get("tail_rule") == expected.get("tail_rule")
            and actual.get("time_metric") == expected.get("time_metric"),
            "Frozen raw metric definitions changed")
    for group in ("overall", "tail"):
        require(actual[group]["count"] == expected[group]["count"], "Metric support changed")
        for key in ("qty_rmse", "qty_mae", "time_nll"):
            a, b = actual[group][key], expected[group][key]
            if a is None or b is None:
                require(a is None and b is None, "Empty-tail metric changed")
            else:
                require(math.isclose(a, b, rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL),
                        "Validation replay changed: " + group + "." + key)

def select_job(contract, host, job_id):
    """Reject unsupported conditions before resolving or loading any input."""
    require(isinstance(contract, dict) and contract.get("schema") == CONTRACT_SCHEMA
            and contract.get("scope") == "execution",
            "A separate approved shared-history execution contract is required")
    permission = contract.get("execution_authorization", {})
    require(permission == {"real_data_training": True, "real_data_inference": True,
                           "remote_execution": True, "test_evaluation": True},
            "Execution authorization must allow the approved fresh mac9 operation")
    require(host == "5090" and set(contract.get("hosts", {})) == {"5090"},
            "Only the approved 5090 host is admitted")
    require(contract.get("common_condition") == {
        "time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "Common input or objective changed")
    jobs = contract.get("jobs", [])
    expected = {(dataset, arm, seed) for dataset in DATASETS for arm in ARMS for seed in SEEDS}
    require(len(jobs) == 9 and {(j.get("dataset"), j.get("arm"), j.get("seed"))
            for j in jobs} == expected, "The approved 9-job matrix changed")
    for job in jobs:
        require(job.get("host") == host and type(job.get("seed")) is int
                and job.get("time_input") == "log1p" and job.get("loss_mode") == "log_mse"
                and type(job.get("time_alpha")) in (int, float) and job["time_alpha"] == 1.0,
                "A job changed host, seed, input or loss")
        expected_id = f"{job['dataset']}__{job['arm']}__seed{job['seed']}"
        require(job.get("id") == expected_id, "A job ID changed its scientific identity")
    selected = [j for j in jobs if j["id"] == job_id]
    require(len(selected) == 1, "Job ID must select one approved condition")
    job = selected[0]
    datasets = contract.get("datasets", [])
    require(len(datasets) == 3 and {d.get("dataset_id") for d in datasets} == set(DATASETS),
            "Uncapped dataset matrix changed")
    require(all("cap_value" not in value for value in datasets),
            "An uncapped contract cannot contain a capped dataset")
    data = next(d for d in datasets if d["dataset_id"] == job["dataset"])
    require(data["optimizer"] == {"name": "AdamW", "lr": .001,
            "weight_decay": .01, "grad_clip": 1.0}, "Common optimizer changed")
    for key, expected_value in (("maximum_epochs", 300), ("minimum_epochs", 40),
                               ("patience", 40), ("batch_size", 128), ("monitor", MONITOR)):
        require(contract["training"].get(key) == expected_value,
                "Common training setting changed: " + key)
    require(contract["training"].get("warm_start", False) is False,
            "Warm-start/resume training is not admitted")
    model, loader = data["model"], data["loader"]
    require(model["hidden_dim"] == 64 and model["quantity_variant"] == VARIANT
            and model["lambda_log_qty"] == 1.0 and model["lambda_tail"] == 0.0
            and model["time_head_lr_multiplier"] == 1.0
            and model["time_head_mode"] == "heteroscedastic_lognormal_duration",
            "Common head or loss configuration changed")
    require(loader["batch_size"] == 128 and loader["num_workers"] == 0
            and loader["drop_last"] is False and loader["train_shuffle"] is True
            and loader["validation_shuffle"] is False, "Common loader changed")
    require(contract.get("data_variant") == "original_uncapped"
            and contract.get("quantity_capping_applied") is False
            and "cap_value" not in data, "Raw task cannot contain capped quantities")
    bounds = data["quantity_boundaries_all_train_rows"]
    require(len(bounds) == 4 and all(math.isfinite(float(v)) for v in bounds)
            and all(a <= b for a, b in zip(bounds, bounds[1:])) and bounds[2] > 0,
            "Frozen raw quantity boundaries changed")
    identity = data["inherited_data_identity"]
    require(set(identity["populations"]) == {"train", "validation"},
            "Only Train/Validation input populations are allowed")
    require(data["full_data_identity"].get("populations", {}).keys() == {"test"},
            "Full raw input must bind exactly the held-out target population")
    expected_window = {"yellow_trip_hourly": (168, 256), "raf_spare_parts": (84, 84),
                       "intermittent_frozen_5000": (520, 256)}[job["dataset"]]
    require((loader["lookback_weeks"], loader["max_seq_len"]) == expected_window,
            "Original dataset history window changed")
    root = Path(contract["hosts"][host]["root"]).resolve()
    return job, data, root, root / "run" / job_id


def training_args(contract, data, output, arm):
    from paper.scripts import run_multilag_detail_execution as base
    args = base.training_args({**contract, "model_role": role_for(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.time_alpha = 1.0
    args.titans_memory_gradient_clip = 1.0
    args.execution_role = "fresh_titans_mac_accuracy9_validation"
    for key, value in {**INACTIVE_TAIL, **input_identity()}.items():
        setattr(args, key, value)
    return args


def training_interface(contract, data, frame, arm):
    from paper.scripts import run_multilag_detail_execution as base
    interface = base.time_interface(data, frame, contract)
    interface.update(input_identity())
    interface["backbone_design"] = {
        "schema": CONTRACT_SCHEMA, "arm": arm, "architecture": contract["architecture"],
        "design_sha256": contract["design_sha256"], "training": contract["training"],
        "data_variant": "original_uncapped", "quantity_capping_applied": False, "encoder_contract": candidate_labels(arm)}
    return interface


def base_receipt(contract, host, job, stage):
    require(stage in ("train", "validation", "test"), "Unknown mac9 stage")
    return {"schema": SCHEMA, "status": "complete", "stage": stage, "host": host,
        "job_id": job["id"], "dataset": job["dataset"], "arm": job["arm"], "seed": job["seed"],
        "contract_sha256": sha_json(contract), "held_out_test_evaluated": stage == "test",
        "source_files_sha256": contract["source"]["files_sha256"],
        "source_revision": contract["source"]["base_git_revision"],
        "design_sha256": contract["design_sha256"],
        **candidate_labels(job["arm"])}


def verify_train_receipt(contract, host, job, root, receipt):
    require(all(receipt.get(key) == value for key, value in
                base_receipt(contract, host, job, "train").items()), "Training receipt binding changed")
    verify_inventory(root, receipt.get("files"))
    proof = verify_time_head_proof(receipt.get("time_head_training_verification"))
    require(receipt["summary"].get("time_head_training_verification") == proof,
            "Fit and summary time-head proofs differ")


def verify_validation_gate(contract, host, root):
    """Independently bind every replay and its observed child exit before Test."""
    gate = read(root / "validation_gate.json")
    require(gate.get("status") == "passed" and gate.get("host") == host
            and gate.get("contract_sha256") == sha_json(contract)
            and set(gate.get("jobs", {})) == {j["id"] for j in contract["jobs"]},
            "All 9 native Validation replays must pass before any Test")
    for job in contract["jobs"]:
        folder = root / "run" / job["id"]
        entry = gate["jobs"][job["id"]]
        path = folder / "validation_receipt.json"
        require(sha_file(path) == entry.get("validation_receipt_sha256"), "Validation gate artifact changed")
        for stage in ("validation", "train"):
            exit_path = folder / (stage + "_exit.json")
            require(sha_file(exit_path) == entry.get(stage + "_exit_sha256"),
                    "Native child exit evidence changed")
            exited = read(exit_path)
            require(exited.get("job_id") == job["id"] and exited.get("stage") == stage
                    and type(exited.get("actual_os_returncode")) is int
                    and exited["actual_os_returncode"] == 0 and exited.get("timeout") is False
                    and type(exited.get("pid")) is int and exited["pid"] > 0
                    and exited.get("start_ticks") is not None
                    and exited.get("observed_exit_at") is not None,
                    "Native " + stage + " actual child exit0 is missing")
        train = read(folder / "train_receipt.json")
        verify_train_receipt(contract, host, job, root, train)
        replay = read(path)
        require(all(replay.get(k) == v for k, v in base_receipt(contract, host, job, "validation").items())
                and replay.get("validation_replay_passed") is True
                and replay.get("individual_predictions_match") is True
                and replay.get("parameters_unchanged") is True
                and replay.get("context_unchanged") is True
                and replay.get("native_runtime_checked") is True,
                "Missing complete native Validation receipt")
        verify_inventory(root, replay.get("files"))
        for key in ("selected_checkpoint", "selected_epoch", "checkpoint_sha256", "state_sha256", "target_identity"):
            require(replay[key] == train[key], "Validation changed selected checkpoint or target identity")
        compare_metrics(replay["metrics"], train["validation_reference"]["metrics"])
        compare_prediction_files(resolve_under(root, replay["prediction_path"]),
                                 resolve_under(root, train["validation_reference"]["prediction_path"]))
    return gate



# These pure checks preserve the reviewed frozen evaluator semantics without
# importing its separate legacy operation and 5090 runtime dependency.
def validate_frame(frame, split):
    import numpy as np
    import polars as pl
    required = {"oper_part_no", "seq", "delta_t", "demand_qty", "chronological_split"}
    permitted = {"train", "validation"} | ({"test"} if split == "test" else set())
    require(required <= set(frame.columns) and frame.height > 0
            and set(frame["chronological_split"].unique().to_list()) <= permitted, "Foreign or empty population")
    require(not frame.select(pl.struct(["oper_part_no", "seq"]).is_duplicated().any()).item(), "Duplicate target key")
    for name in ("delta_t", "demand_qty", "seq"):
        values = frame[name].to_numpy().astype(np.float64)
        require(bool(np.isfinite(values).all()) and (name == "seq" or bool((values >= 0).all())), "Invalid raw values")
    ranks = frame.with_columns(pl.col("chronological_split").replace_strict(
        {"train": 0, "validation": 1, "test": 2}).alias("_rank")).with_columns(
        pl.col("_rank").cum_max().over("oper_part_no").alias("_max_rank"))
    require(ranks.filter(pl.col("_rank") < pl.col("_max_rank")).height == 0, "Chronological split order changed")


def verify_sequence_local_online_memory(model, dt, mask, quantity):
    """Exercise live writes and prove unrelated forwards cannot persist memory."""
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    memory = model.titans_mac_encoder.neural_memory
    original = memory.write_sequence
    before_sha = canonical_state_dict_sha256(model.state_dict())
    observed = {"write_calls": 0, "finite_state_updates": 0}
    def write(state, *args, **kwargs):
        old = tuple(value.detach().clone() for value in state.memory_tensors())
        updated, diagnostics = original(state, *args, **kwargs)
        new = updated.memory_tensors()
        require(all(bool(torch.isfinite(value).all()) for value in new), "Nonfinite online memory state")
        observed["write_calls"] += 1
        observed["finite_state_updates"] += int(any(not torch.equal(a, b) for a, b in zip(old, new)))
        return updated, diagnostics
    had_instance = "write_sequence" in memory.__dict__
    previous_instance = memory.__dict__.get("write_sequence")
    memory.write_sequence = write
    try:
        with torch.no_grad():
            first = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
            # Deliberately unrelated observations execute normal online updates.
            target_outputs(model, dt + 2, mask, quantity + 5, lambda_log_qty=1.0)
            again = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
            for key in ("pred_qty", "time_loss"):
                torch.testing.assert_close(first[key], again[key], rtol=1e-5, atol=1e-5)
        require(observed["write_calls"] > 0 and observed["finite_state_updates"] > 0,
                "MAC online memory updates were not exercised")
        require(canonical_state_dict_sha256(model.state_dict()) == before_sha,
                "Online memory persisted into parameters or buffers")
        return {**observed, "online_updates_enabled": True, "sequence_local_reset": True,
                "parameters_and_buffers_unchanged": True}
    finally:
        if had_instance:
            memory.write_sequence = previous_instance
        else:
            del memory.write_sequence


def qualification_checks(model, dt, mask, quantity):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    before = canonical_state_dict_sha256(model.state_dict())
    context_before = context_identity(model)
    require(not any(module.training for module in model.modules())
            and not any(parameter.requires_grad for parameter in model.parameters()),
            "Frozen evaluation must disable training and parameter gradients")
    with torch.no_grad():
        result = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
        changed_q, changed_dt = quantity.clone(), dt.clone()
        changed_q[:, -1] += 12345
        changed_dt[:, -1] += 123
        q = target_outputs(model, dt, mask, changed_q, lambda_log_qty=1.0)
        t = target_outputs(model, changed_dt, mask, quantity, lambda_log_qty=1.0)
        single = target_outputs(model, dt[:1], mask[:1], quantity[:1], lambda_log_qty=1.0)
        checks = {
            "target_quantity_blocked_for_prediction": torch.allclose(result["pred_qty"], q["pred_qty"], rtol=1e-5, atol=1e-5),
            "target_quantity_blocked_for_time_loss": torch.allclose(result["time_loss"], q["time_loss"], rtol=1e-5, atol=1e-5),
            "target_gap_blocked_for_quantity": torch.allclose(result["pred_qty"], t["pred_qty"], rtol=1e-5, atol=1e-5),
            "single_batch_prediction_matches": torch.allclose(result["pred_qty"][:1], single["pred_qty"], rtol=1e-5, atol=1e-5),
            "single_batch_time_loss_matches": torch.allclose(result["time_loss"][:1], single["time_loss"], rtol=1e-5, atol=1e-5),
            "finite_outputs": all(bool(torch.isfinite(result[key]).all()) for key in ("pred_qty", "time_loss")),
            "parameters_unchanged": canonical_state_dict_sha256(model.state_dict()) == before,
            "context_unchanged": context_identity(model) == context_before,
        }
    require(all(checks.values()), "Synthetic or first-batch causal qualification failed")
    return checks


def context_identity(model):
    """Include the inherited transient attention context outside state_dict."""
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    result = {}
    for name, module in model.named_modules():
        if hasattr(module, "_ctx_mem"):
            value = module._ctx_mem
            result[name] = None if value is None else canonical_state_dict_sha256({"context": value})
    return result


def prepare_evaluation_frame(data, split):
    """Admit full data only after run_stage has passed the 9-job gate."""
    from paper.scripts import run_multilag_detail_execution as base
    if split == "validation":
        return base.prepare_admitted_data(data)[0]
    require(split == "test", "Unsupported evaluation population")
    import polars as pl
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    full = data["full_data_identity"]["data"]
    require(sha_file(full["path"]) == full["sha256"], "Frozen full raw input changed")
    frame = pl.read_parquet(full["path"]).sort(["oper_part_no", "seq"])
    validate_frame(frame, "test")
    training_frame = base.prepare_admitted_data(data)[0]
    frame = prepare_count_frame(frame)
    require(set(training_frame.columns) <= set(frame.columns)
            and frame.filter(pl.col("chronological_split").is_in(["train", "validation"]))
            .select(training_frame.columns).equals(training_frame),
            "Full raw parent differs from frozen Train/Validation export")
    return frame


def check_native_runtime(contract):
    import platform
    expected = contract["hosts"]["5090"]["runtime_expected"]
    require(platform.python_version() == "3.12.13"
            and torch.__version__ == "2.11.0+cu130"
            and torch.version.cuda == "13.0"
            and torch.backends.cudnn.version() == 92000
            and torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090"
            and torch.get_num_threads() == 4 and torch.get_num_interop_threads() == 1,
            "Native 5090 numerical runtime changed")
    measured = {"python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads()}
    require(measured == expected, "Native runtime differs from sealed contract")
    uuid = str(torch.cuda.get_device_properties(0).uuid)
    require(uuid.lower().removeprefix("gpu-").replace("-", "")
            == contract["hosts"]["5090"]["gpu_uuid"].lower().removeprefix("gpu-").replace("-", ""),
            "Unapproved CUDA device")
    return {**measured, "gpu_uuid": uuid, "gpu": torch.cuda.get_device_name(0)}


def verify_model_conditions(contract, data, arm, model, encoder):
    frozen = contract["frozen_conditions"]["models"][arm]
    dataset = data["dataset_id"]
    require(encoder == frozen["expected_encoder_config_by_dataset"][dataset],
            "Historical model architecture/head metadata changed: " + arm)
    require(sum(p.numel() for p in model.parameters() if p.requires_grad)
            == frozen["expected_parameter_count_by_dataset"][dataset],
            "Historical parameter count changed: " + arm)


def _load_selected(data, job, receipt, root, device):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256, torch_load_checkpoint
    path = resolve_under(root, receipt["selected_checkpoint"])
    require(sha_file(path) == receipt["checkpoint_sha256"], "Selected binary changed")
    checkpoint = torch_load_checkpoint(path, map_location="cpu")
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    validate_checkpoint_route(checkpoint, job["arm"])
    require(checkpoint["encoder_config"].get("titans_memory_gradient_clip") == 1.0
            and checkpoint["encoder_config"].get("backbone_contract_id") == "B1"
            and checkpoint["resume_identity"]["arguments"].get("titans_memory_gradient_clip") == 1.0
            and checkpoint["resume_identity"]["arguments"].get("model_role") == role_for(job["arm"]),
            "MAC clip1 or observed-time checkpoint identity changed")
    require(checkpoint.get("backbone") == job["arm"] and checkpoint.get("variant") == VARIANT
            and checkpoint.get("evaluation_scope") == "validation_only"
            and checkpoint.get("held_out_test_evaluated") is False, "Selected checkpoint route changed")
    require(checkpoint["seed"] == job["seed"] and checkpoint["best_epoch"] == receipt["selected_epoch"]
            and checkpoint["checkpoint_monitor"] == MONITOR, "Selected seed/epoch/monitor changed")
    require(all(checkpoint.get(k) == v for k, v in input_identity().items()),
            "Checkpoint input or objective identity changed")
    arguments = checkpoint["resume_identity"]["arguments"]
    require(arguments.get("dataset_contract") == job["dataset"]
            and arguments.get("data_sha256") == data["inherited_data_identity"]["data"]["sha256"]
            and arguments.get("split_manifest_sha256") == data["inherited_data_identity"]["split_manifest"]["sha256"]
            and checkpoint["interface_meta"].get("source_files_sha256") == receipt["source_files_sha256"],
            "Selected checkpoint data or source binding changed")
    proof = checkpoint["time_head_training_verification"]
    verify_time_head_proof(proof, checkpoint["model_state_dict"])
    verify_checkpoint_fit_proof(proof, receipt["time_head_training_verification"])
    digest = canonical_state_dict_sha256(checkpoint["model_state_dict"])
    require(digest == checkpoint["model_state_sha256"] == receipt["state_sha256"],
            "Selected tensor digest changed")
    history_path = path.parent / "history.json"
    history = read(history_path)["history"]
    require(prepared.select_best_epoch(history) == checkpoint["best_epoch"],
            "Selected epoch differs from first strict finite full-Validation minimum")
    model, encoder = build_model(data, job["arm"])
    require(encoder == checkpoint["encoder_config"], "Reconstructed encoder metadata changed")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False), path, digest


def evaluate_selected(data, job, receipt, root, frame, split, output, device="cuda:0"):
    """Persist exact targets and six metrics with the frozen Validation loader."""
    import gzip
    import numpy as np
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(split in ("validation", "test"), "Unsupported evaluation split")
    model, checkpoint, digest = _load_selected(data, job, receipt, root, device)
    context_before = context_identity(model)
    loader = make_loader(frame, target_split=split, shuffle=False, generator=None,
        **{key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")})
    truth, identity = population_identity(loader.dataset, split)
    expected = data["inherited_data_identity"] if split == "validation" else data["full_data_identity"]
    verify_population(identity, expected["populations"][split])
    metrics, processed = MetricAccumulator(data["quantity_boundaries_all_train_rows"][-1]), 0
    with Path(output).open("xb") as binary, gzip.GzipFile(fileobj=binary, mode="wb", mtime=0) as compressed:
        with torch.no_grad():
            for _, dts, mask, part_ids, quantities in loader:
                value = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.0)
                arrays = {key: value[key].detach().cpu().numpy()
                          for key in ("true_qty", "pred_qty", "time_loss", "history_length")}
                size = len(arrays["true_qty"])
                require(np.array_equal(arrays["true_qty"].astype(np.float64), truth[processed:processed + size]),
                        "Validation target truth changed")
                metrics.add(arrays["true_qty"], arrays["pred_qty"], arrays["time_loss"])
                for index in range(size):
                    part, end = loader.dataset.index[processed + index]
                    require(int(part_ids[index]) == int(part), "Validation target order changed")
                    row = {"target_index": processed + index, "entity_id": str(loader.dataset.parts[part]),
                        "target_seq": int(loader.dataset.seq_lists[part][end + 1]), "split": split,
                        "raw_quantity": float(arrays["true_qty"][index]),
                        "predicted_raw_quantity": float(arrays["pred_qty"][index]),
                        "time_nll": float(arrays["time_loss"][index]),
                        "history_length": int(arrays["history_length"][index])}
                    compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
                processed += size
    require(processed == identity["target_count"], "Incomplete Validation population")
    require(canonical_state_dict_sha256(model.state_dict()) == digest, "Replay changed model state")
    require(context_identity(model) == context_before, "Replay changed transient model context")
    require(sha_file(checkpoint) == receipt["checkpoint_sha256"], "Replay changed checkpoint binary")
    result = {"metrics": metrics.finish(), "target_identity": identity,
        "prediction_path": str(Path(output).resolve()), "prediction_rows": processed,
        "parameters_unchanged": True, "context_unchanged": True, "state_sha256": digest}
    del model
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def run_stage(contract, host, job_id, stage):
    require(stage in ("train", "validation", "test"), "Unsupported mac9 stage")
    verify_frozen_dependencies()
    job, data, root, folder = select_job(contract, host, job_id)
    if stage == "test":
        verify_validation_gate(contract, host, root)
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / f"{stage}_receipt.json"
    require(not receipt_path.exists(), "Completed stages cannot be repeated")
    if stage == "train":
        require(not (folder / "runs").exists(), "Only fresh fits; resume/retry is unavailable")
    write_exclusive(folder / f"{stage}.intent.json", {"contract_sha256": sha_json(contract),
        "job_id": job_id, "stage": stage, "started_unix": time.time()})
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts import run_multilag_detail_execution as base
    configure_runtime("cuda:0", 4)
    runtime_before = check_native_runtime(contract)
    mac_runtime.configure_dynamo()
    if stage == "train":
        from paper.scripts.count_aware_tpp_backbone import training
        frame, input_receipt = base.prepare_admitted_data(data)
        write_exclusive(folder / "input_receipt.json", input_receipt)
        with _engine_hooks(job["arm"], contract=contract, data=data) as tracked:
            args = training_args(contract, data, folder, job["arm"])
            args.seeds = str(job["seed"])
            interface = training_interface(contract, data, frame, job["arm"])
            quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
                "strata": [{"stratum": f"quantity_{index}", "stratum_order": index,
                            "stratum_label": f"Raw Train bin {index}"} for index in range(5)]}
            summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
                interface_meta=interface, backbone=job["arm"], quantity_variant=VARIANT, seed=job["seed"])
            proof = tracked["tracker"].proof()
        summary = {**summary, **input_identity(), "time_head_training_verification": proof}
        checkpoint = resolve_under(root, summary["checkpoint_path"])
        receipt = {**base_receipt(contract, host, job, stage), "selected_checkpoint": str(checkpoint),
            "selected_epoch": summary["best_epoch"], "checkpoint_sha256": sha_file(checkpoint),
            "state_sha256": summary["checkpoint_state_sha256"], "summary": summary,
            "time_head_training_verification": proof}
        prediction = folder / "training_validation_predictions.jsonl.gz"
        with runtime_hooks(job["arm"]):
            reference = evaluate_selected(data, job, receipt, root, frame, "validation", prediction)
        for key, expected in (("qty_rmse", summary["best_val_qty_rmse"]),
                              ("qty_mae", summary["best_val_qty_mae"]),
                              ("time_nll", summary["best_val_time_nll"])):
            require(math.isclose(reference["metrics"]["overall"][key], expected,
                    rel_tol=REPLAY_RTOL, abs_tol=REPLAY_ATOL), "Selected Validation differs from training summary")
        receipt.update(validation_reference=reference, target_identity=reference["target_identity"])
        files = [path for path in (folder / "runs").rglob("*") if path.is_file()]
        receipt["files"] = file_inventory(root, [*files, prediction, folder / "input_receipt.json"])
    else:
        train_path = folder / "train_receipt.json"
        train_receipt = read(train_path)
        verify_train_receipt(contract, host, job, root, train_receipt)
        frame = prepare_evaluation_frame(data, stage)
        prediction = folder / (stage + "_predictions.jsonl.gz")
        with runtime_hooks(job["arm"]):
            evaluated = evaluate_selected(data, job, train_receipt, root, frame, stage, prediction)
        if stage == "validation":
            compare_metrics(evaluated["metrics"], train_receipt["validation_reference"]["metrics"])
            require(evaluated["target_identity"] == train_receipt["validation_reference"]["target_identity"],
                    "Validation replay target identity changed")
            compare_prediction_files(prediction, train_receipt["validation_reference"]["prediction_path"])
        receipt = {**base_receipt(contract, host, job, stage), **evaluated,
            "selected_checkpoint": train_receipt["selected_checkpoint"],
            "selected_epoch": train_receipt["selected_epoch"],
            "checkpoint_sha256": train_receipt["checkpoint_sha256"],
            "time_head_training_verification": train_receipt["time_head_training_verification"],
            "validation_replay_passed": stage == "validation",
            "individual_predictions_match": stage == "validation",
            "files": file_inventory(root, [prediction, train_path,
                     resolve_under(root, train_receipt["selected_checkpoint"])])}
    runtime_after = check_native_runtime(contract)
    require(runtime_before == runtime_after, "Numerical runtime changed during stage")
    receipt.update(native_runtime_checked=True, runtime=runtime_before, runtime_after=runtime_after)
    if stage == "test":
        verify_validation_gate(contract, host, root)
        require(sha_file(data["full_data_identity"]["data"]["path"])
                == data["full_data_identity"]["data"]["sha256"], "Full raw input changed during Test")
    write_exclusive(receipt_path, receipt)
    return receipt


def synthetic_check(contract=None, host=None, *, device=None, arm=None):
    """Fabricated admission on each candidate, including joint gradients and full training shape."""
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import core
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    verify_frozen_dependencies()
    mac_runtime.configure_dynamo()
    require(arm is None or arm in ARMS, "Unknown mac9 synthetic candidate")
    selected_device = device or ("cuda:0" if contract is not None else "cpu")
    configure_runtime(selected_device, 4)
    if contract is not None:
        select_job(contract, host, contract["jobs"][0]["id"])
        if str(selected_device).startswith("cuda"):
            runtime_before = check_native_runtime(contract)
    lengths = [data["loader"]["max_seq_len"] for data in contract["datasets"]] if contract else [256]
    peaks, counts, states, proofs, memory_proofs = {}, {}, {}, {}, {}
    head_proof = verify_common_head_initialization(prepared.fabricated_data())
    head_proofs_by_dataset = ({data["dataset_id"]: {str(seed): verify_common_head_initialization(data, seed)
        for seed in SEEDS} for data in contract["datasets"]} if contract is not None else {})
    selected_arms = ARMS if arm is None else (arm,)
    for arm in selected_arms:
        if contract is not None:
            for data in contract["datasets"]:
                expected_model, expected_encoder = build_model(data, arm)
                verify_model_conditions(contract, data, arm, expected_model, expected_encoder)
                del expected_model
        if str(selected_device).startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        model, encoder = build_model(prepared.fabricated_data(), arm)
        states[arm] = canonical_state_dict_sha256(model.state_dict())
        model = model.to(selected_device).train()
        tracker = JointTimeHeadTracker(model)
        optimizer = tracker.bind_optimizer(torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01))
        dt, quantity, mask = [value.to(selected_device) for value in prepared.fabricated_batch()]
        with runtime_hooks(arm, tracked={"tracker": tracker}):
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
                require(torch.equal(output["joint_loss"], output["time_loss"] + output["log_qty_loss"]), "Joint loss changed")
                output["joint_loss"].mean().backward()
                require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()), "Nonfinite fabricated gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            model.eval()
            memory_proof = verify_sequence_local_online_memory(model, dt, mask, quantity)
            before = canonical_state_dict_sha256(model.state_dict())
            with torch.no_grad():
                original = core.target_outputs(model, dt, mask, quantity, lambda_log_qty=1.0)
                changed_quantity, changed_dt = quantity.clone(), dt.clone()
                changed_quantity[:, -1] += 100
                changed_dt[:, -1] += 100
                for altered in (core.target_outputs(model, dt, mask, changed_quantity, lambda_log_qty=1.0),
                                core.target_outputs(model, changed_dt, mask, quantity, lambda_log_qty=1.0)):
                    torch.testing.assert_close(original["pred_qty"], altered["pred_qty"], rtol=1e-5, atol=1e-5)
                changed_q_output = core.target_outputs(model, dt, mask, changed_quantity, lambda_log_qty=1.0)
                torch.testing.assert_close(original["time_loss"], changed_q_output["time_loss"], rtol=1e-5, atol=1e-5)
                single = core.target_outputs(model, dt[:1], mask[:1], quantity[:1], lambda_log_qty=1.0)
                for key in ("pred_qty", "time_loss"):
                    torch.testing.assert_close(original[key][:1], single[key], rtol=1e-5, atol=1e-5)
                binary = io.BytesIO()
                torch.save(model.state_dict(), binary)
                binary.seek(0)
                restored, restored_encoder = build_model(prepared.fabricated_data(), arm)
                require(encoder == restored_encoder, "Fabricated encoder metadata changed")
                restored.load_state_dict(torch.load(binary, map_location="cpu", weights_only=True), strict=True)
                require(canonical_state_dict_sha256(restored.state_dict()) == before,
                        "Checkpoint roundtrip changed exact parameter bytes")
                restored = restored.to(selected_device).eval()
                replay = core.target_outputs(restored, dt, mask, quantity, lambda_log_qty=1.0)
                for key in ("pred_qty", "time_loss"):
                    torch.testing.assert_close(original[key], replay[key], rtol=1e-5, atol=1e-5)
            require(before == canonical_state_dict_sha256(model.state_dict()), "Synthetic evaluation changed state")
            del restored
            full_shape = [128, max(lengths)] if str(selected_device).startswith("cuda") else None
            if full_shape:
                model.train()
                full_dt = torch.ones(full_shape, device=selected_device)
                full_quantity = torch.randint(1, 11, full_shape, device=selected_device).float()
                full_mask = torch.ones(full_shape, device=selected_device, dtype=torch.bool)
                optimizer.zero_grad(set_to_none=True)
                loss = core.target_outputs(model, full_dt, full_mask, full_quantity, lambda_log_qty=1.0)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite full-shape loss")
                loss.backward()
                require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                            for parameter in model.parameters()), "Nonfinite full-shape gradient")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                del full_dt, full_quantity, full_mask, loss
                # Qualify the full native inference batch on fabricated rows.
                model.eval().requires_grad_(False)
                inference_shape = [128, max(lengths)]
                inference_dt = torch.ones(inference_shape, device=selected_device)
                inference_q = torch.ones(inference_shape, device=selected_device)
                inference_mask = torch.ones(inference_shape, device=selected_device, dtype=torch.bool)
                qualification_checks(model, inference_dt, inference_mask, inference_q)
                torch.cuda.synchronize()
                del inference_dt, inference_q, inference_mask
        counts[arm] = sum(parameter.numel() for parameter in model.parameters())
        proofs[arm] = tracker.proof()
        memory_proofs[arm] = memory_proof
        optimizer.step = tracker.original_step
        peaks[arm] = {"max_memory_reserved_bytes": torch.cuda.max_memory_reserved() if full_shape else 0,
            "max_memory_allocated_bytes": torch.cuda.max_memory_allocated() if full_shape else 0,
            "full_training_shape": full_shape, "includes_optimizer_step": bool(full_shape),
            "full_inference_shape": inference_shape if full_shape else None,
            "inference_batch_size": 128 if full_shape else None}
        del model, optimizer, tracker
        gc.collect()
    if contract is not None and str(selected_device).startswith("cuda"):
        require(check_native_runtime(contract) == runtime_before, "Synthetic numerical runtime changed")
    return {"schema": SCHEMA, "status": "passed", "device": str(selected_device), "host": host,
        "arms": list(selected_arms), "cuda_memory_peaks": peaks, "parameter_count": counts,
        "runtime": runtime_before if contract is not None and str(selected_device).startswith("cuda") else None,
        "initial_state_sha256": states, "time_head_training_verification": proofs,
        "common_head_initialization": head_proof, "common_head_initialization_by_dataset": head_proofs_by_dataset,
        "online_memory_verification": memory_proofs,
        "real_data_loaded": False, "held_out_test_evaluated": False,
        "checks": {"finite_forward_backward": True, "checkpoint_roundtrip": True,
            "target_quantity_blocked": True, "target_time_blocked_for_quantity": True,
            "target_quantity_blocked_for_time": True, "checkpoint_state_bytes_exact": True,
            "batch_independence": True, "state_unchanged_during_evaluation": True,
            "joint_time_and_log_quantity_loss": True, "time_head_received_nonzero_gradient": True,
            "final_shared15_common_heads_exact": True, "online_memory_updates_preserved": True,
            "sequence_local_memory_reset": True, "frozen_preparation_source_unchanged": True}}


def synthetic_condition(contract, host, loss_mode="log_mse", *, arm, device=None):
    require(loss_mode == "log_mse", "Only the approved log-MSE condition can be qualified")
    result = synthetic_check(contract, host, device=device, arm=arm)
    return {**result, **candidate_labels(arm), "arm": arm,
            "cuda_memory_peak": result["cuda_memory_peaks"][arm]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--host", choices=("5090",), required=True)
    parser.add_argument("--job")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--stage", choices=("train", "validation", "test", "synthetic"))
    selector.add_argument("--mode", choices=("synthetic",))
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args(argv)
    contract = read(args.contract)
    with redirect_stdout(sys.stderr):
        stage = args.stage or args.mode
        if stage == "synthetic":
            result = synthetic_check(contract, args.host, arm=args.arm)
        else:
            require(args.job is not None, "A scientific job is required")
            result = run_stage(contract, args.host, args.job, stage)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
