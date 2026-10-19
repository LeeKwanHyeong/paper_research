"""Local preparation checks for shared and matched independent history MLPs.

This entry point accepts fabricated tensors only.  It contains no data-loader,
SSH, deployment, Test, or real-data training command.  Its two-epoch fits check
the actual model, existing heads/objective, AdamW updates, selected binary, and
same-target Validation replay; they are not research experiment results.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import torch

from models.TPPs.CountAwareTitanSharedHistoryMLP import (
    ARMS, MODE_BY_ARM, CountAwareTitanSharedHistoryMLP,
    metadata as history_metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_time_zero_mlp9 as frozen_zero


SCHEMA = "titantpp_shared_history_cap_synthetic_v1"
CONTRACT_SCHEMA = "titantpp_shared_history_cap_preparation_v1"
VARIANT = "count_only_log_regression"
MONITOR = "validation_raw_quantity_rmse"
SEEDS = (42, 52, 62)
DATASETS = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
INPUT_TRANSFORM = "log1p_nonnegative_float32"
FROZEN_ZERO_SHA256 = "2bcc15dec157ff8d0c6aa82198714ba2895e9c5d62d2b1c5d13b930a8ce9c86b"
_ZERO_PATH = Path(frozen_zero.__file__).resolve()
_ACTIVE_ARM = None
_SCOPE_DEPTH = 0
TimeHeadGuard = frozen_zero.TimeHeadGuard
verify_time_head_proof = frozen_zero.verify_time_head_proof


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def input_identity():
    return {"time_input": "log1p", "time_input_transform": INPUT_TRANSFORM,
            "quantity_input_transform": INPUT_TRANSFORM, "time_alpha": 0.0,
            "loss_mode": "log_mse", "quantity_loss_normalization": "none",
            "quantity_head_family": "unchanged_softplus_log1p_location_then_expm1",
            "time_loss_graph_excluded": True, "time_head_trained": False}


def fabricated_data():
    """Numerical configuration only; contains no project dataset paths."""
    return {"model": {"hidden_dim": 64, "quantity_variant": VARIANT,
            "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 1.0,
            "time_initial_location": 0.2, "time_initial_scale": 0.8,
            "time_observation_contract": {"mode": "positive_integer_round_clamp_v1",
                                          "unit": "hour", "top_code": None}},
            "statistics": {"train_log_mean": 1.2, "train_log_std": 0.8},
            "loader": {"max_seq_len": 256}}


def _candidate_factory(arm, **kwargs):
    require(arm in ARMS, "Only the two explicit shared-history candidates are allowed")
    require(kwargs.pop("titans_memory_gradient_clip", None) is None,
            "Memory gradient clipping must remain unchanged")
    from models.TPPs.CountAwareFactory import with_time_metadata
    model = CountAwareTitanSharedHistoryMLP(
        **kwargs, history_sharing_mode=MODE_BY_ARM[arm])
    return with_time_metadata(model, {
        **history_metadata(kwargs["hidden_dim"], mode=MODE_BY_ARM[arm]),
        "max_len": kwargs["max_seq_len"], **input_identity()})


def build_model(data, arm):
    """Build candidates directly; existing registered model routes stay intact."""
    config = {key: value for key, value in data["model"].items()
              if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    # The inactive auxiliary settings equal the unchanged Factory defaults.
    config.update(tail_threshold=46.0, tail_normalization_scale=46.0,
                  tail_clip_cap=187.0, tail_huber_delta=1.0)
    return _candidate_factory(arm, **config,
        train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"],
        max_seq_len=data["loader"]["max_seq_len"])


@contextmanager
def candidate_factory(arm=None):
    """Temporarily route candidate names without editing the shared registry."""
    global _ACTIVE_ARM, _SCOPE_DEPTH
    require(arm is None or arm in ARMS, "Unknown candidate arm")
    if _SCOPE_DEPTH:
        require(arm is None or arm == _ACTIVE_ARM, "Nested candidate scope changed arm")
        _SCOPE_DEPTH += 1
        try:
            yield
        finally:
            _SCOPE_DEPTH -= 1
        return
    from copy import copy
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import constants, observed_time, training
    undo = []
    previous = _ACTIVE_ARM
    _ACTIVE_ARM, _SCOPE_DEPTH = arm, 1

    def replace(module, name, value):
        undo.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    old_build, old_route = factory.build_count_aware_model, factory.validate_checkpoint_route
    old_launch, old_role = observed_time.validate_launch, constants.validate_model_role_contract
    roles = {name: role_for_arm(name) for name in ARMS}

    def build(backbone, **kwargs):
        return _candidate_factory(backbone, **kwargs) if backbone in ARMS else old_build(backbone, **kwargs)

    def route(payload, expected_backbone):
        if expected_backbone in ARMS:
            return validate_checkpoint(payload, expected_backbone)
        # The model validator also rejects a candidate relabelled as an old arm.
        validate_checkpoint(payload, expected_backbone)
        return old_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if any(name in ARMS for name in backbones) or getattr(args, "model_role", None) in roles.values():
            require(len(backbones) == 1 and backbones[0] in ARMS
                    and args.model_role == roles[backbones[0]], "Candidate launch role mismatch")
            proxy = copy(args)
            proxy.model_role = "observed_time_core_ablation_v1"
            return old_launch(proxy, ("titantpp_history_mlp",), variants)
        return old_launch(args, backbones, variants)

    def role_contract(**kwargs):
        names = kwargs["backbones"]
        if any(name in ARMS for name in names) or kwargs["model_role"] in roles.values():
            require(len(names) == 1 and names[0] in ARMS
                    and kwargs["model_role"] == roles[names[0]], "Candidate model role mismatch")
            return old_role(**{**kwargs, "model_role": "observed_time_core_ablation_v1",
                               "backbones": ("titantpp_history_mlp",)})
        return old_role(**kwargs)

    try:
        for module, name, value in (
            (factory, "build_count_aware_model", build),
            (factory, "validate_checkpoint_route", route),
            (training, "build_count_aware_model", build), (training, "build_model", build),
            (training, "validate_checkpoint_route", route),
            (observed_time, "validate_launch", launch),
            (observed_time, "ROLES", (*observed_time.ROLES, *roles.values())),
            (training, "OBSERVED_TIME_ROLES", (*training.OBSERVED_TIME_ROLES, *roles.values())),
            (constants, "validate_model_role_contract", role_contract),
            (constants, "BACKBONE_LABELS", {**constants.BACKBONE_LABELS, **{name: name for name in ARMS}}),
            (training, "BACKBONE_LABELS", {**training.BACKBONE_LABELS, **{name: name for name in ARMS}}),
            (constants, "MODEL_ROLES", (*constants.MODEL_ROLES, *roles.values())),
            (constants, "SUPPORTED_BACKBONES", (*constants.SUPPORTED_BACKBONES, *ARMS))):
            replace(module, name, value)
        yield
    finally:
        for module, name, value in reversed(undo):
            setattr(module, name, value)
        _ACTIVE_ARM, _SCOPE_DEPTH = previous, 0


def quantity_only_outputs(original):
    """Exclude time loss from the graph; do not construct 0 * time_loss."""
    def wrapped(*args, **kwargs):
        require(kwargs.get("lambda_log_qty") == 1.0, "Quantity coefficient must remain one")
        result = original(*args, **kwargs)
        require(torch.equal(result["quantity_train_loss"], result["log_qty_loss"]),
                "The existing log-MSE quantity loss changed")
        return {**result, "joint_loss": result["quantity_train_loss"]}
    return wrapped


@contextmanager
def runtime_hooks(arm=None):
    """Restore temporary quantity-only and candidate aliases even on failure."""
    require(sha_file(_ZERO_PATH) == FROZEN_ZERO_SHA256, "Frozen time-head guard source changed")
    from paper.scripts.count_aware_tpp_backbone import core, training
    with candidate_factory(arm):
        originals = core.target_outputs, training.target_outputs
        core.target_outputs = training.target_outputs = quantity_only_outputs(originals[0])
        try:
            yield
        finally:
            core.target_outputs, training.target_outputs = originals
            require(sha_file(_ZERO_PATH) == FROZEN_ZERO_SHA256, "Frozen source changed during check")


def run_stage(contract, host=None, job_id=None, stage=None):
    """Preparation contracts cannot run real data, even after flag tampering."""
    if stage not in ("train", "validation"):
        raise PermissionError("Test and all other real-data stages are unavailable")
    # Fail before selecting a job, resolving an input path, or opening a dataset.
    if not isinstance(contract, dict) or contract.get("scope") != "preparation_only":
        raise PermissionError("Only a preparation contract is accepted; no data has been read")
    raise PermissionError("Real-data training/inference is not implemented or authorized in this preparation worker")


def fabricated_batch():
    """Four fixed irregular event sequences; target fields never enter history."""
    dts = torch.tensor([[1, 2, 1, 3, 1, 8, 2, 1, 6, 3],
                        [1, 1, 2, 1, 3, 2, 4, 1, 3, 7],
                        [1, 3, 1, 2, 1, 4, 2, 3, 1, 2],
                        [1, 2, 3, 1, 2, 1, 3, 2, 4, 1]], dtype=torch.float32)
    quantities = torch.tensor([[1, 3, 2, 4, 2, 1, 3, 2, 1, 12],
                               [2, 4, 1, 3, 1, 2, 1, 3, 2, 2],
                               [1, 2, 3, 1, 2, 1, 4, 2, 3, 8],
                               [2, 1, 3, 2, 1, 4, 2, 1, 3, 1]], dtype=torch.float32)
    return dts, quantities, torch.ones_like(dts, dtype=torch.bool)


def _target_identity(batch):
    digest = hashlib.sha256(b"shared_history_fabricated_target_population_v1\0")
    for tensor in batch:
        digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return {"scope": "fabricated_same_target_only", "target_count": batch[0].shape[0],
            "target_identity_sha256": digest.hexdigest()}


def _state_sha(model):
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    return canonical_state_dict_sha256(model.state_dict())


@torch.no_grad()
def evaluate_fabricated(model, batch=None):
    """Evaluate six diagnostic metrics using an explicit fabricated tail cut."""
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    batch = fabricated_batch() if batch is None else batch
    model.eval()
    before = _state_sha(model)
    dt, quantity, mask = batch
    value = quantity_only_outputs(target_outputs)(model, dt, mask, quantity, lambda_log_qty=1.0)
    require(before == _state_sha(model), "Evaluation changed a parameter or buffer")
    prediction, truth, time_nll = (value[name].double().cpu()
                                  for name in ("pred_qty", "true_qty", "time_loss"))
    require(all(bool(torch.isfinite(tensor).all()) for tensor in (prediction, truth, time_nll)),
            "Nonfinite fabricated prediction or time loss")
    error = prediction - truth
    tail = truth > 5.0
    require(bool(tail.any()), "Fabricated large-quantity cohort is empty")
    metrics = {"raw_quantity_rmse": float(error.square().mean().sqrt()),
               "raw_quantity_mae": float(error.abs().mean()),
               "large_quantity_rmse": float(error[tail].square().mean().sqrt()),
               "large_quantity_mae": float(error[tail].abs().mean()),
               "recorded_positive_integer_time_nll": float(time_nll.mean()),
               "large_recorded_positive_integer_time_nll": float(time_nll[tail].mean())}
    return {"metrics": metrics, "prediction": prediction.tolist(), "truth": truth.tolist(),
            "time_nll": time_nll.tolist(), "target_identity": _target_identity(batch),
            "large_quantity_threshold": 5.0, "large_quantity_threshold_scope": "fabricated_only",
            "state_sha256": before}


def select_best_epoch(history):
    """First strict finite minimum of full-population raw quantity RMSE."""
    best = None
    for row in history:
        score = row.get("val_qty_rmse")
        if (isinstance(score, bool) or not isinstance(score, (float, int))
                or not math.isfinite(float(score))):
            continue
        if best is None or float(score) < float(best["val_qty_rmse"]):
            best = row
    require(best is not None, "No finite full-Validation RMSE selection exists")
    return best["epoch"]


def _engine_fixture(output, arm):
    """Construct a tiny Train/Validation frame, with Train-only head statistics."""
    import numpy as np
    import polars as pl
    from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
    from paper.scripts.run_count_aware_tpp_backbone_control import parse_args, derive_train_time_contract
    rows = []
    for entity in ("fabricated_a", "fabricated_b", "fabricated_c", "fabricated_d"):
        for index, (dt, quantity) in enumerate(zip(
                (0, 1, 2, 1, 3, 1, 4, 2, 1, 3), (1, 3, 8, 10, 10, 2, 5, 1, 12, 2))):
            rows.append({"oper_part_no": entity, "seq": index + 1, "delta_t": dt,
                         "demand_qty": float(quantity),
                         "chronological_split": "train" if index < 8 else "validation"})
    raw = pl.DataFrame(rows)
    frame = prepare_count_frame(raw)
    data_path, manifest_path = output / "fabricated_train_validation.parquet", output / "fabricated_manifest.json"
    raw.write_parquet(data_path)
    with manifest_path.open("x") as stream:
        json.dump({"scope": "fabricated_only", "train_rows": 32, "validation_rows": 8}, stream)
    old_argv = sys.argv
    try:
        sys.argv = ["shared_history_fabricated_check", "--data", str(data_path),
            "--split-manifest", str(manifest_path), "--output-dir", str(output / "engine"),
            "--source-revision", "fabricated_preparation_only", "--execution-role", "synthetic_cpu_check"]
        args = parse_args()
    finally:
        sys.argv = old_argv
    temporal = derive_train_time_contract(frame, lookback_weeks=84, max_seq_len=84)
    train_q = raw.filter(pl.col("chronological_split") == "train")["demand_qty"].to_numpy()
    data = fabricated_data()
    data["loader"]["max_seq_len"] = 84
    data["statistics"] = {"train_log_mean": float(np.log1p(train_q).mean()),
                           "train_log_std": float(np.log1p(train_q).std())}
    data["model"].update(time_scale=temporal["time_scale"],
        time_initial_location=temporal["target_log_scaled_mean"],
        time_initial_scale=temporal["target_log_scaled_std"], time_sigma_floor=0.001,
        time_observation_contract={"mode": "positive_integer_round_clamp_v1", "unit": "month", "top_code": None})
    settings = {"epochs": 2, "min_epochs": 2, "early_stopping_patience": 2,
        "device": "cpu", "dataset_contract": "raf_spare_parts", "model_role": role_for_arm(arm),
        "checkpoint_monitor": MONITOR, "lookback_weeks": 84, "max_seq_len": 84,
        "data_sha256": sha_file(data_path), "split_manifest_sha256": sha_file(manifest_path),
        "time_head_mode": data["model"]["time_head_mode"], "time_scale": temporal["time_scale"],
        "time_sigma_floor": 0.001, "time_alpha": 0.0, "loss_mode": "log_mse"}
    for name, value in settings.items():
        setattr(args, name, value)
    interface = {"train_target_mean": data["statistics"]["train_log_mean"],
        "train_target_std": data["statistics"]["train_log_std"], **input_identity(),
        "data_scope": "fabricated_train_validation_only",
        "time_head": {"mode": args.time_head_mode, "time_scale": args.time_scale,
            "time_initial_location": data["model"]["time_initial_location"],
            "time_initial_scale": data["model"]["time_initial_scale"],
            "statistics_source_split": "train", "train_time_statistics": temporal,
            "observation_likelihood": data["model"]["time_observation_contract"]}}
    quantity_contract = {"boundaries": [2.0, 5.0, 10.0, 12.0],
        "strata": [{"stratum": "fabricated_" + str(index)} for index in range(5)]}
    return args, frame, data, interface, quantity_contract


@contextmanager
def _engine_hooks(arm):
    """Guard the actual engine optimizer and bind its serialized checkpoint proof."""
    from paper.scripts.count_aware_tpp_backbone import training
    tracked = {}
    with runtime_hooks(arm):
        old_build, old_optimizer = training.build_model, training.build_optimizer
        old_save, old_json = training.atomic_torch_save, training.save_json
        def build(*args, **kwargs):
            model, meta = old_build(*args, **kwargs)
            require("guard" not in tracked, "Duplicate synthetic engine fit model")
            tracked["guard"] = TimeHeadGuard(model)
            return model, meta
        def optimizer(model, *args, **kwargs):
            guard = tracked.get("guard")
            require(guard is not None and guard.model is model, "Engine optimizer/model binding changed")
            result = guard.bind_optimizer(old_optimizer(model, *args, **kwargs))
            require(all(group["lr"] == 0.001 and group["weight_decay"] == 0.01
                        for group in result.param_groups), "Engine AdamW settings changed")
            return result
        def save(payload, path):
            proof = tracked["guard"].proof(payload)
            return old_save({**payload, **input_identity(), "time_head_no_training_verification": proof}, path)
        def save_json(path, payload):
            if "backbone" in payload:
                payload = {**payload, **input_identity(),
                           "time_head_no_training_verification": tracked["guard"].proof()}
            return old_json(path, payload)
        training.build_model, training.build_optimizer = build, optimizer
        training.atomic_torch_save, training.save_json = save, save_json
        try:
            yield tracked
        finally:
            training.build_model, training.build_optimizer = old_build, old_optimizer
            training.atomic_torch_save, training.save_json = old_save, old_json
            if "guard" in tracked and tracked["guard"].optimizer is not None:
                tracked["guard"].optimizer.step = tracked["guard"].original_step


def synthetic_fit_replay(arm, output, *, seed=42):
    """Run the existing train_one engine for two fabricated CPU epochs, then replay."""
    require(arm in ARMS and seed in SEEDS, "Unknown synthetic arm or seed")
    require(sha_file(_ZERO_PATH) == FROZEN_ZERO_SHA256, "Frozen guard source changed")
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    configure_runtime("cpu", 4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    args, frame, data, interface, quantity_contract = _engine_fixture(output, arm)
    with torch.random.fork_rng(devices=[]):
        with _engine_hooks(arm) as tracked:
            summary, _, _ = training.train_one(args=args, frame=frame,
                quantity_contract=quantity_contract, interface_meta=interface,
                backbone=arm, quantity_variant=VARIANT, seed=seed)
            final_proof = tracked["guard"].proof()
            model = tracked["guard"].model
            loader = make_loader(frame, target_split="validation", batch_size=128,
                lookback_weeks=84, max_seq_len=84, shuffle=False, generator=None)
            batches = list(loader)
            require(len(batches) == 1, "Fabricated engine population must fit one full batch")
            _, dts, masks, _, quantities = batches[0]
            batch = dts, quantities, masks
            selected_evaluation = evaluate_fabricated(model, batch)
        checkpoint = Path(summary["checkpoint_path"])
        digest = sha_file(checkpoint)
        loaded = torch_load_checkpoint(checkpoint, map_location="cpu")
        validate_checkpoint(loaded, arm)
        verify_time_head_proof(loaded["time_head_no_training_verification"], loaded["model_state_dict"])
        history = json.loads((checkpoint.parent / "history.json").read_text())["history"]
        require(loaded["best_epoch"] == select_best_epoch(history), "Engine selector differs from strict finite minimum")
        require(summary["completed_epochs"] == 2, "Synthetic engine did not complete two epochs")
        # New model construction and strict load reproduce each target and metric.
        replay_model, replay_metadata = build_model(data, arm)
        require(replay_metadata == loaded["encoder_config"], "Replayed model metadata changed")
        replay_model.load_state_dict(loaded["model_state_dict"], strict=True)
        replay = evaluate_fabricated(replay_model, batch)
        require(replay["target_identity"] == selected_evaluation["target_identity"], "Validation target identity changed")
        for key in ("prediction", "truth", "time_nll"):
            torch.testing.assert_close(torch.tensor(replay[key], dtype=torch.float64),
                torch.tensor(selected_evaluation[key], dtype=torch.float64), rtol=1e-5, atol=1e-5)
        require(replay["metrics"] == selected_evaluation["metrics"], "Six-metric replay differs")
        require(replay["state_sha256"] == loaded["model_state_sha256"], "Selected tensor binding differs")
        require(math.isclose(replay["metrics"]["raw_quantity_rmse"], summary["best_val_qty_rmse"], rel_tol=1e-12)
                and math.isclose(replay["metrics"]["raw_quantity_mae"], summary["best_val_qty_mae"], rel_tol=1e-12),
                "Full engine Validation and independent replay metrics differ")
        require(digest == sha_file(checkpoint), "Selected binary changed during replay")
    receipt = {"schema": SCHEMA, "status": "passed", "candidate_name": arm,
        "history_relation_sharing": MODE_BY_ARM[arm],
        "history_relation_parameter_count": loaded["encoder_config"]["history_relation_parameter_count"],
        "training_engine": "paper.scripts.count_aware_tpp_backbone.training.train_one",
        "actual_training_engine_integration_passed": True,
        "device": "cpu", "seed": seed, "fabricated_epochs": 2,
        "selected_epoch": loaded["best_epoch"], "history": history,
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
        "checkpoint_state_sha256": loaded["model_state_sha256"],
        "target_identity": replay["target_identity"], "six_metrics": replay["metrics"],
        "same_target_prediction_and_time_loss_replay_passed": True,
        "parameter_and_buffer_unchanged_during_replay": True,
        "selected_binary_sha256_unchanged": True,
        "time_head_no_training_verification": final_proof,
        "research_training_budget": {"epochs": 300, "min_epochs": 40, "patience": 40},
        "synthetic_budget_is_research_training": False,
        "real_data_loaded": False, "held_out_test_evaluated": False, **input_identity()}
    with (output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def synthetic_check(*, output=None):
    """Only two fabricated fits; no contract may enable a real-data path."""
    root = Path(output) if output is not None else Path(tempfile.mkdtemp(prefix="shared_history_cap_synthetic_")) / "checks"
    root.mkdir(parents=True, exist_ok=False)
    results = {arm: synthetic_fit_replay(arm, root / arm) for arm in ARMS}
    return {"schema": SCHEMA, "status": "passed", "scope": "fabricated_cpu_preparation_only",
        "device": "cpu", "arms": list(ARMS), "per_arm": results,
        "real_data_loaded": False, "held_out_test_evaluated": False,
        "new_real_data_training_count": 0, "remote_execution_count": 0,
        "output": str(root), **input_identity()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic-check",), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    # The maintained training engine prints progress.  Keep that diagnostic
    # stream separate so callers can parse stdout as one complete JSON value.
    with redirect_stdout(sys.stderr):
        result = synthetic_check(output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
