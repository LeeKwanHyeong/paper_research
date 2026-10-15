"""Execution adapter for the two frozen history-CNN candidates. No fit at import."""
from __future__ import annotations

import argparse
from copy import copy
import inspect
import json
from pathlib import Path
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareTitanAdaptiveHistoryCNN import (
    CNN, ADAPTIVE, CAPACITY, ARMS, CONTRACT_ID, IDENTITY_BUFFER, EXTRA_PREFIX,
    CountAwareTitanAdaptiveHistoryCNN, metadata as candidate_metadata,
    role_for_arm as candidate_role, validate_checkpoint,
)
ALL_ARMS = (CNN, *ARMS)
CANDIDATE_ROLES = {arm: candidate_role(arm) for arm in ARMS}
from paper.scripts import run_titantpp_cnn_gru_components as original

BASELINE, SCHEMA, SUPPORTED_ARMS = CNN, CONTRACT_ID, ALL_ARMS
VARIANT = original.VARIANT
base, shared, common = original.base, original.shared, original.common
require, read = original.require, original.read
TIME_METRIC, audit_replay_accounting = original.TIME_METRIC, original.audit_replay_accounting
ROLE_BY_ARM = {CNN: original.role_for_arm(CNN), **CANDIDATE_ROLES}
TRAINING = {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
    "monitor": "validation_raw_quantity_rmse", "tie": "strict_earliest_finite_minimum", "warm_start": False}
_installed = False


def role_for_arm(arm):
    require(arm in ALL_ARMS, "Unknown History-CNN comparison arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm):
    require(arm in ALL_ARMS, "Unknown History-CNN metadata arm")
    return (candidate_metadata(hidden_dim, arm=arm) if arm in ARMS
            else original.metadata(hidden_dim, arm=arm))


def install_hooks():
    """Opt-in process-local model/identity routes; original files stay untouched."""
    global _installed
    if _installed:
        return
    original.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build = factory.build_count_aware_model
    signature = inspect.getclosurevars(original_build).nonlocals.get("signature")
    if not isinstance(signature, inspect.Signature) or "hidden_dim" not in signature.parameters:
        raise RuntimeError("Original component factory signature unavailable")
    original_route, original_launch = factory.validate_checkpoint_route, observed_time.validate_launch
    original_role = constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone not in ARMS:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("History CNN preserves the original memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = CountAwareTitanAdaptiveHistoryCNN(**values, history_arm=backbone)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=backbone), "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in CANDIDATE_ROLES.values() or any(a in ARMS for a in backbones):
            if len(backbones) != 1 or backbones[0] not in ARMS or getattr(args, "model_role", None) != role_for_arm(backbones[0]):
                raise ValueError("History CNN requires its matching explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = role_for_arm(CNN)
            return original_launch(old_args, (CNN,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        if kwargs["model_role"] in CANDIDATE_ROLES.values() or any(a in ARMS for a in kwargs["backbones"]):
            if len(kwargs["backbones"]) != 1 or kwargs["backbones"][0] not in ARMS or kwargs["model_role"] != role_for_arm(kwargs["backbones"][0]):
                raise ValueError("History CNN requires its matching explicit single-arm role")
            return original_role(**{**kwargs, "model_role": role_for_arm(CNN), "backbones": (CNN,)})
        return original_role(**kwargs)

    factory.build_count_aware_model, factory.validate_checkpoint_route = build, route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = (*observed_time.ROLES, *CANDIDATE_ROLES.values())
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.MODEL_ROLES = (*constants.MODEL_ROLES, *CANDIDATE_ROLES.values())
    constants.SUPPORTED_BACKBONES = (*constants.SUPPORTED_BACKBONES, *ARMS)
    constants.BACKBONE_LABELS.update({a: a for a in ARMS})
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in ALL_ARMS, "Unknown History-CNN model arm")
    if arm not in ARMS:
        return original.build_model(data, arm)
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {key: value for key, value in data["model"].items()
        if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def fabricated_data():
    data = original.fabricated_data()
    data["model"].update(lambda_log_qty=1., lambda_tail=0., time_head_lr_multiplier=1.)
    data["model"]["time_sigma_floor"] = .001
    data["loader"].update(batch_size=128, drop_last=False, num_workers=0,
        train_shuffle=True, validation_shuffle=False, lookback_weeks=168)
    data["optimizer"] = {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.}
    data["dataset_id"] = "yellow_trip_hourly"
    data["inherited_data_identity"] = {
        "data": {"path": "fabricated_no_data_access.parquet", "sha256": "0" * 64},
        "split_manifest": {"path": "fabricated_no_split_access.json", "sha256": "1" * 64}}
    data["quantity_boundaries_all_train_rows"] = [2, 10, 20, 30]
    return data


def initial_states(data, seed, *, arms=ARMS):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(seed in (42, 52, 62) and tuple(arms) == ARMS, "Only two candidates and three seeds")
    result = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference_model, _ = original.build_model(data, CNN)
        reference, rng = reference_model.state_dict(), torch.get_rng_state().clone()
        for arm in arms:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            state = model.state_dict()
            require(torch.equal(rng, torch.get_rng_state()), "Constructor RNG changed")
            require(all(name in state and torch.equal(value, state[name]) for name,value in reference.items()),
                    "Common initial tensors changed")
            require(all(name == IDENTITY_BUFFER or name.startswith(EXTRA_PREFIX) for name in set(state)-set(reference)),
                    "Unexpected parameter addition")
            require(sum(p.numel() for p in model.parameters() if p.requires_grad) == 90678, "Active count changed")
            result[arm] = canonical_state_dict_sha256(state)
    return result


initialization = initial_states


def training_args(contract, data, output, arm=ADAPTIVE):
    """Prepare frozen arguments only; this function never starts a fit."""
    install_hooks()
    require(arm in ARMS, "Existing CNN are read-only references, not new fits")
    require(all(contract.get("training", {}).get(key) == value for key, value in TRAINING.items()),
        "Frozen training budget or Validation checkpoint selector changed")
    require(contract.get("evaluation_scope") == "validation_only"
        and contract.get("held_out_test_evaluated") is False and contract.get("test_evaluation") is False,
        "This experiment is Train/Validation only")
    require(not any(Path(output).glob("*.pt")), "Fresh candidate fit cannot reuse a checkpoint")
    config, optimizer = data["model"], data["optimizer"]
    require(config.get("hidden_dim") == 64 and config.get("lambda_log_qty") == 1.
        and config.get("lambda_tail") == 0. and config.get("time_head_lr_multiplier") == 1.
        and config.get("time_head_mode") == "heteroscedastic_lognormal_duration"
        and config.get("time_observation_contract", {}).get("mode") == "positive_integer_round_clamp_v1"
        and config.get("time_observation_contract", {}).get("top_code") is None,
        "Original quantity loss or recorded-time head changed")
    require(optimizer == {"name": "AdamW", "lr": .001, "weight_decay": .01, "grad_clip": 1.},
        "Original optimizer changed")
    require(data["loader"].get("batch_size") == 128 and data["loader"].get("drop_last") is False
        and data["loader"].get("num_workers") == 0 and data["loader"].get("train_shuffle") is True
        and data["loader"].get("validation_shuffle") is False, "Original loader budget changed")
    args = base.training_args({**contract, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_adaptive_history_cnn_validation"
    require(args.force_rerun is False and args.checkpoint_monitor == "validation_raw_quantity_rmse",
        "Rerun or selector changed")
    return args


def run_fit(*args, **kwargs):
    """No actual-training entry point is authorized in this preparation version."""
    raise PermissionError("Native qualification and a separately approved training permit are required; preparation starts no fit")


def time_interface(data, frame, contract, *, arm=ADAPTIVE):
    require(arm in ARMS, "Only the candidate creates a new fit interface")
    interface = base.time_interface(data, frame, contract)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": arm,
        "model_metadata": metadata(arm=arm), "training": contract["training"]}
    return interface


def _parent_replay_function(function):
    return FunctionType(function.__code__, {**function.__globals__,
        "SUPPORTED_ARMS": SUPPORTED_ARMS, "build_model": build_model,
        "_replay_validation": _replay_validation}, function.__name__, function.__defaults__, function.__closure__)


def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    return _parent_replay_function(original.original.cnn._replay_validation)(
        payload, path, data, frame, budget_check, device=device)


def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None,
        expected_identity=None, expected_initial=None, expected_epoch=None, expected_seed=42):
    install_hooks()
    return _parent_replay_function(original.original.cnn.replay_checkpoint)(path, data, frame, budget,
        device=device, expected_arm=expected_arm, expected_identity=expected_identity,
        expected_initial=expected_initial, expected_epoch=expected_epoch, expected_seed=expected_seed)

