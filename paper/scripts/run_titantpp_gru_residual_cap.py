"""Isolated GRU-cap model adapter and fabricated-input qualification.

Importing performs no model construction, route installation, data access or fit.
The CLI only accepts a CPU synthetic check. Argument preparation is not a native
training permit; deployment, native qualification and real fits remain a separate
explicitly authorized campaign. Existing CNN/FULL model routes are delegated.
"""
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

from models.TPPs.CountAwareTitanCNNGRUResidualCap import (
    FULL, CNN, CAPPED, RHO, ARMS, ALL_ARMS, CONTRACT_ID, IDENTITY_BUFFER,
    ROLE_BY_ARM as CANDIDATE_ROLES, CountAwareTitanCNNGRUResidualCap,
    correction_parameters, metadata as candidate_metadata,
    role_for_arm as candidate_role, validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_gru_components as original

BASELINE, SCHEMA, SUPPORTED_ARMS = FULL, CONTRACT_ID, ALL_ARMS
VARIANT = original.VARIANT
base, shared, common = original.base, original.shared, original.common
require, read = original.require, original.read
TIME_METRIC, audit_replay_accounting = original.TIME_METRIC, original.audit_replay_accounting
REGISTERED_PARAMETERS = 115167
ACTIVE_PARAMETERS = {CNN: 90435, FULL: 115167, CAPPED: 115167}
ACTIVE_CORRECTION_PARAMETERS = {CNN: 576, FULL: 25308, CAPPED: 25308}
ROLE_BY_ARM = {CNN: original.role_for_arm(CNN), FULL: original.role_for_arm(FULL), **CANDIDATE_ROLES}
TRAINING = {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
    "monitor": "validation_raw_quantity_rmse", "tie": "strict_earliest_finite_minimum", "warm_start": False}
_installed = False


def role_for_arm(arm):
    require(arm in ALL_ARMS, "Unknown GRU-cap comparison arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm):
    require(arm in ALL_ARMS, "Unknown GRU-cap metadata arm")
    return (candidate_metadata(hidden_dim, arm=arm) if arm == CAPPED
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
        if backbone != CAPPED:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("GRU cap preserves the original memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = CountAwareTitanCNNGRUResidualCap(**values, residual_cap_arm=backbone, rho=RHO)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=backbone), "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in CANDIDATE_ROLES.values() or CAPPED in backbones:
            if tuple(backbones) != (CAPPED,) or getattr(args, "model_role", None) != role_for_arm(CAPPED):
                raise ValueError("GRU cap requires its matching explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = role_for_arm(FULL)
            return original_launch(old_args, (FULL,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        if kwargs["model_role"] in CANDIDATE_ROLES.values() or CAPPED in kwargs["backbones"]:
            if tuple(kwargs["backbones"]) != (CAPPED,) or kwargs["model_role"] != role_for_arm(CAPPED):
                raise ValueError("GRU cap requires its matching explicit single-arm role")
            return original_role(**{**kwargs, "model_role": role_for_arm(FULL), "backbones": (FULL,)})
        return original_role(**kwargs)

    factory.build_count_aware_model, factory.validate_checkpoint_route = build, route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = (*observed_time.ROLES, *CANDIDATE_ROLES.values())
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.MODEL_ROLES = (*constants.MODEL_ROLES, *CANDIDATE_ROLES.values())
    constants.SUPPORTED_BACKBONES = (*constants.SUPPORTED_BACKBONES, CAPPED)
    constants.BACKBONE_LABELS[CAPPED] = "TitanTPP CNN+GRU fixed residual cap 0.25"
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in ALL_ARMS, "Unknown GRU-cap model arm")
    if arm != CAPPED:
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


def initial_states(data, seed, *, arms=ALL_ARMS):
    """Exact learned FULL tensors and construction RNG; identity buffers differ."""
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    arms = tuple(arms)
    require(type(seed) is int and seed in (42, 52, 62), "Only the frozen three seeds are permitted")
    require(bool(arms) and len(set(arms)) == len(arms) and all(arm in ALL_ARMS for arm in arms),
        "Invalid GRU-cap comparison arms")
    result = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference_model, _ = original.build_model(data, FULL)
        reference, rng = reference_model.state_dict(), torch.get_rng_state().clone()
        for arm in arms:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            state = model.state_dict()
            require(torch.equal(rng, torch.get_rng_state()), "Candidate changed construction RNG")
            require(all(name in state and torch.equal(value, state[name]) for name, value in reference.items()),
                "Candidate changed a parent initial tensor")
            extra = set() if arm == FULL else {IDENTITY_BUFFER} if arm == CAPPED else {"cnn_gru_component_identity"}
            require(set(state) - set(reference) == extra, "Unexpected candidate state addition")
            result[arm] = canonical_state_dict_sha256(state)
    return result


initialization = initial_states


def training_args(contract, data, output, arm=CAPPED):
    """Prepare frozen arguments only; this function never starts a fit."""
    install_hooks()
    require(arm == CAPPED, "Existing CNN/FULL are read-only references, not new fits")
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
    args.execution_role = "fresh_gru_residual_cap_validation"
    require(args.force_rerun is False and args.checkpoint_monitor == "validation_raw_quantity_rmse",
        "Rerun or selector changed")
    return args


def run_fit(*args, **kwargs):
    """No actual-training entry point is authorized in this preparation version."""
    raise PermissionError("Native qualification and a separately approved training permit are required; preparation starts no fit")


def time_interface(data, frame, contract, *, arm=CAPPED):
    require(arm == CAPPED, "Only the candidate creates a new fit interface")
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


def synthetic_check(*, seed=42):
    """Four fabricated updates verify exact initial outputs and trainable paths."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    data = fabricated_data()
    initial = initial_states(data, seed)
    coverage_counts, counts = {}, {}
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator().manual_seed(121)
        dt = torch.randint(1, 8, (3, 12), generator=generator).float()
        qty = torch.randint(0, 20, (3, 12), generator=generator).float()
        dt[:, 0] = 0.
        mask = torch.ones_like(dt, dtype=torch.bool)
        reference_output, reference_rng = None, None
        for arm in ALL_ARMS:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            model.train()
            parameters = correction_parameters(model)
            require(sum(p.numel() for p in model.parameters()) == REGISTERED_PARAMETERS,
                "Registered parameter count changed")
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            require(counts[arm] == ACTIVE_PARAMETERS[arm], "Active parameter count changed")
            before = {name: p.detach().clone() for name, p in parameters.items()}
            torch.manual_seed(314)
            initial_output = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            rng = torch.get_rng_state().clone()
            if reference_output is None:
                reference_output, reference_rng = initial_output, rng
            require(torch.equal(reference_rng, rng), "Initial forward changed dropout RNG")
            require(all(torch.equal(reference_output[key], initial_output[key]) for key in initial_output),
                "Zero-initialized candidate output differs")
            coverage = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in parameters.items()}
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01)
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(131 + step)
                loss = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    "Nonfinite synthetic gradient")
                for name, parameter in parameters.items():
                    if not parameter.requires_grad:
                        require(parameter.grad is None, "Frozen CNN reference GRU received a gradient")
                    elif parameter.grad is not None:
                        coverage[name] |= parameter.grad != 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            coverage_counts[arm] = sum(int(value.sum()) for value in coverage.values())
            require(coverage_counts[arm] == ACTIVE_CORRECTION_PARAMETERS[arm], "Inactive correction parameter element")
            for name, parameter in parameters.items():
                require(bool((before[name] != parameter).any()) if parameter.requires_grad
                    else torch.equal(before[name], parameter), "Synthetic correction update/freeze mismatch")
            restored, _ = build_model(data, arm)
            restored.load_state_dict(model.state_dict(), strict=True)
    return {"status": "passed", "device": "cpu", "seed": seed, "rho": RHO,
        "initial_state_sha256": initial, "common_initial_tensors": "exact_original_FULL",
        "initial_output_and_dropout_rng": "identical", "registered_parameter_count": REGISTERED_PARAMETERS,
        "active_trainable_parameter_count": counts, "active_correction_gradient_parameter_count": coverage_counts,
        "synthetic_optimizer_updates": 12, "real_data_loaded": False, "real_training_started": False,
        "held_out_test_evaluated": False, "native_runtime_qualification": "not_performed",
        "limitation": "Synthetic correctness does not demonstrate improved generalization; CNN has fewer active parameters."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-check", action="store_true", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 52, 62), default=42)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    print(json.dumps(synthetic_check(seed=args.seed), indent=2))
