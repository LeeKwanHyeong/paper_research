"""Sealed-runtime adapter for RAF-first Q/K-location CNN comparison.

Import, argument preparation and qualification do not start scientific fits.
The existing permit/lease-verified native runtime dispatches actual training.
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

from models.TPPs.CountAwareTitanQKLocationCNN import (
    CNN, Q_ONLY, K_ONLY, ARMS, ALL_ARMS, LEGACY_CNN, CONTRACT_ID, IDENTITY_BUFFER, EXTRA_PREFIX,
    LOCATION_BY_ARM, LOCAL_KEYS_BY_ARM, REMOVED_KEYS_BY_ARM, ORIGINAL_LOCAL_KEYS,
    ACTIVE_PARAMETERS, ADDED_PARAMETERS, PARAMETER_DELTA, CountAwareTitanQKLocationCNN, metadata as candidate_metadata,
    role_for_arm as candidate_role, validate_checkpoint,
)
CANDIDATE_ROLES = {arm: candidate_role(arm) for arm in ALL_ARMS}
from paper.scripts import run_titantpp_cnn_gru_components as original

BASELINE, SCHEMA, SUPPORTED_ARMS = CNN, CONTRACT_ID, ALL_ARMS
VARIANT = original.VARIANT
base, shared, common = original.base, original.shared, original.common
require, read = original.require, original.read
TIME_METRIC, audit_replay_accounting = original.TIME_METRIC, original.audit_replay_accounting
ROLE_BY_ARM = CANDIDATE_ROLES.copy()
TRAINING = {"batch_size": 128, "maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
    "monitor": "validation_raw_quantity_rmse", "tie": "strict_earliest_finite_minimum", "warm_start": False}
_installed = False


def role_for_arm(arm):
    require(arm in ALL_ARMS, "Unknown Q/K-location CNN comparison arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm, max_seq_len=256):
    require(arm in ALL_ARMS, "Unknown Q/K-location CNN metadata arm")
    return candidate_metadata(hidden_dim, arm=arm, max_seq_len=max_seq_len)


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
        if backbone not in ALL_ARMS:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("Q/K-location CNN preserves the original memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = (original_build(LEGACY_CNN, **kwargs)[0] if backbone == CNN
                 else CountAwareTitanQKLocationCNN(**values, qk_location_arm=backbone))
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=backbone, max_seq_len=values["max_seq_len"]), "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in CANDIDATE_ROLES.values() or any(a in ALL_ARMS for a in backbones):
            if len(backbones) != 1 or backbones[0] not in ALL_ARMS or getattr(args, "model_role", None) != role_for_arm(backbones[0]):
                raise ValueError("Q/K-location CNN requires its matching explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = original.role_for_arm(LEGACY_CNN)
            return original_launch(old_args, (LEGACY_CNN,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        if kwargs["model_role"] in CANDIDATE_ROLES.values() or any(a in ALL_ARMS for a in kwargs["backbones"]):
            if len(kwargs["backbones"]) != 1 or kwargs["backbones"][0] not in ALL_ARMS or kwargs["model_role"] != role_for_arm(kwargs["backbones"][0]):
                raise ValueError("Q/K-location CNN requires its matching explicit single-arm role")
            return original_role(**{**kwargs, "model_role": original.role_for_arm(LEGACY_CNN), "backbones": (LEGACY_CNN,)})
        return original_role(**kwargs)

    factory.build_count_aware_model, factory.validate_checkpoint_route = build, route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = (*observed_time.ROLES, *CANDIDATE_ROLES.values())
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.MODEL_ROLES = (*constants.MODEL_ROLES, *CANDIDATE_ROLES.values())
    constants.SUPPORTED_BACKBONES = (*constants.SUPPORTED_BACKBONES, *ALL_ARMS)
    constants.BACKBONE_LABELS.update({a: a for a in ALL_ARMS})
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in ALL_ARMS, "Unknown Q/K-location CNN model arm")
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


def baseline_reference_initial_state_sha(data, seed):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(seed in (42, 52, 62), "Only approved seeds")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, _ = original.build_model(data, LEGACY_CNN)
        return canonical_state_dict_sha256(model.state_dict())


def common_reference_initial_state_sha(data, seed, *, arm):
    """Hash original tensors retained by the specified location, excluding removed local kernels."""
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(seed in (42, 52, 62) and arm in ARMS, "Only approved seeds and candidate arms")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model, _ = original.build_model(data, LEGACY_CNN)
        return canonical_state_dict_sha256({k: v for k, v in model.state_dict().items() if k not in REMOVED_KEYS_BY_ARM[arm]})


def initial_states(data, seed, *, arms=ALL_ARMS):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    arms = tuple(arms)
    require(seed in (42, 52, 62) and arms and len(set(arms)) == len(arms)
            and all(arm in ALL_ARMS for arm in arms), "Unknown/duplicate arms or seed")
    result = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference_model, _ = original.build_model(data, LEGACY_CNN)
        reference, rng = reference_model.state_dict(), torch.get_rng_state().clone()
        require(all(bool((reference[key] == 0).all()) for key in ORIGINAL_LOCAL_KEYS), "Original local kernels must initialize to zero")
        for arm in arms:
            torch.manual_seed(seed)
            model, meta = build_model(data, arm)
            state = model.state_dict()
            require(torch.equal(rng, torch.get_rng_state()), "Constructor RNG changed")
            shared = reference if arm == CNN else {k: v for k, v in reference.items() if k not in REMOVED_KEYS_BY_ARM[arm]}
            require(all(name in state and torch.equal(value, state[name]) for name, value in shared.items()),
                    "Common initial tensors changed")
            if arm == CNN:
                require(set(state) == set(reference), "Unexpected baseline state")
            else:
                require(set(state) == (set(reference) - set(REMOVED_KEYS_BY_ARM[arm])) | {IDENTITY_BUFFER},
                        "Unexpected replacement state")
                require(all(tuple(state[k].shape) == (3, 64) and bool((state[k] == 0).all()) for k in LOCAL_KEYS_BY_ARM[arm]),
                        "Q/K must retain the original zero-initialized three taps")
            expected_active = sum(p.numel() for p in reference_model.parameters() if p.requires_grad) + (PARAMETER_DELTA if arm in ARMS else 0)
            require(sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_active
                    == meta["active_trainable_parameter_count"], "Active count changed")
            require(sum(p.numel() for p in model.parameters()) == meta["registered_parameter_count"],
                    "Registered count changed")
            result[arm] = canonical_state_dict_sha256(state)
    return result


initialization = initial_states


def training_args(contract, data, output, arm=Q_ONLY):
    """Prepare frozen arguments only; this function never starts a fit."""
    install_hooks()
    require(arm in ARMS, "Only the new Q/K-location candidate is approved for fitting")
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
    args.execution_role = "fresh_qk_location_cnn_validation"
    require(args.force_rerun is False and args.checkpoint_monitor == "validation_raw_quantity_rmse",
        "Rerun or selector changed")
    return args


def time_interface(data, frame, contract, *, arm=Q_ONLY):
    require(arm in ALL_ARMS, "Unknown approved interface arm")
    interface = base.time_interface(data, frame, contract)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": arm,
        "model_metadata": metadata(arm=arm, max_seq_len=data["loader"]["max_seq_len"]), "training": contract["training"]}
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


def synthetic_check(device="cpu", *, arms=ALL_ARMS, seed=42, include_causality=True):
    """Fabricated inputs only; covers active gradients and frozen/common state."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    data, arms = fabricated_data(), tuple(arms)
    initial = initial_states(data, seed, arms=arms)
    devices = [] if str(device) == "cpu" else [torch.device(device)]
    counts, coverage_counts, initial_gradient_onset, value_gradient_verified = {}, {}, {}, {}
    with torch.random.fork_rng(devices=devices):
        generator = torch.Generator().manual_seed(121)
        dt = torch.randint(1, 8, (3, 12), generator=generator).float().to(device)
        qty = torch.randint(0, 20, (3, 12), generator=generator).float().to(device)
        dt[:, 0] = 0.
        mask = torch.ones_like(dt, dtype=torch.bool)
        reference = reference_rng = None
        for arm in arms:
            torch.manual_seed(seed)
            model, meta = build_model(data, arm)
            model.to(device).train()
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            require(counts[arm] == meta["active_trainable_parameter_count"], "Active parameter count changed")
            require(sum(p.numel() for p in model.parameters()) == meta["registered_parameter_count"],
                    "Registered parameter count changed")
            tracked = {name: p for name, p in model.named_parameters()
                       if name.startswith(EXTRA_PREFIX) or name.startswith("multilag_detail.")
                       or name.startswith("encoder.layers.0.attn.causal_")}
            before = {name: p.detach().clone() for name, p in tracked.items()}
            if arm in ARMS:
                require(set(n for n, _ in model.encoder.layers[0].attn.named_parameters() if n.endswith("_kernel")) == {"causal_" + LOCATION_BY_ARM[arm] + "_kernel"}, "Unexpected local filter or dummy V parameter")
            value_gradient_verified[arm] = False
            coverage = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in tracked.items()}
            torch.manual_seed(314)
            output = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            rng = torch.get_rng_state().clone()
            if reference is None:
                reference, reference_rng = output, rng
            require(torch.equal(reference_rng, rng), "Forward dropout RNG changed")
            require(all(torch.equal(reference[k], output[k]) for k in reference), "Initial predictions differ")
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01)
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(131 + step)
                loss = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                        "Nonfinite synthetic gradient")
                value_gradient = model.encoder.layers[0].attn.qkv.weight.grad
                value_gradient_verified[arm] |= value_gradient is not None and bool((value_gradient[128:] != 0).any())
                for name, parameter in tracked.items():
                    if not parameter.requires_grad:
                        require(parameter.grad is None, "Frozen GRU received gradients")
                    elif parameter.grad is not None:
                        coverage[name] |= parameter.grad != 0
                if step == 0:
                    initial_gradient_onset[arm] = {name: int((p.grad != 0).sum()) if p.grad is not None else 0
                        for name, p in tracked.items() if name.startswith(EXTRA_PREFIX)}
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            require(value_gradient_verified[arm], "Common V projection did not receive a synthetic gradient")
            expected = meta["active_correction_parameter_count"]
            coverage_counts[arm] = sum(int(v.sum()) for v in coverage.values())
            require(coverage_counts[arm] == expected, "Unused correction weights: " + str(
                {name: int(v.sum()) for name, v in coverage.items()}))
            for name, parameter in tracked.items():
                require(bool((before[name] != parameter).any()) if parameter.requires_grad else torch.equal(before[name], parameter),
                        "Correction update/freeze mismatch: " + name)
            restored, _ = build_model(data, arm)
            restored.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)
            if include_causality:
                model.eval()
                with torch.no_grad():
                    before_eval = {k: v.clone() for k, v in model.state_dict().items()}
                    out = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
                    changed = qty.clone(); changed[:, -1] = float("nan")
                    altered = target_outputs(model, dt, mask, changed, lambda_log_qty=1.)
                    require(torch.equal(out["pred_qty"], altered["pred_qty"]), "Target quantity leakage")
                    encoded = model._encode_base(dt, qty, mask)
                    changed = qty.clone(); changed[:, 7:] += 1000.
                    changed_dt = dt.clone(); changed_dt[:, 7:] += 1000.
                    later = model._encode_base(changed_dt, changed, mask)
                    require(torch.equal(encoded[:, :7], later[:, :7]), "Future changed prefix")
                    require(all(torch.equal(v, model.state_dict()[k]) for k, v in before_eval.items()), "Evaluation mutated state")
    return {"status": "passed", "device": str(device), "seed": seed,
        "initial_state_sha256": initial, "active_trainable_parameter_count": counts,
        "active_correction_gradient_parameter_count": coverage_counts,
        "initial_gradient_nonzero_elements": initial_gradient_onset,
        "common_value_projection_gradient_verified": value_gradient_verified,
        "active_local_cnn_parameters": 192, "active_value_cnn_parameters": 0,
        "original_value_kernel_absent": True, "value_operation": "unmixed_common_value_projection",
        "synthetic_optimizer_updates": 4 * len(arms), "scientific_optimizer_updates": 0,
        "real_data_loaded": False, "held_out_test_evaluated": False,
        "budget_limitation": "Q-only/K-only each192, original576, QK-only384; memory-key scoring makes the two insertion locations structurally asymmetric"}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic qualification only; native runtime owns permitted fits")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, choices=(42, 52, 62), default=42)
    arguments = parser.parse_args(argv)
    print(json.dumps(synthetic_check(arguments.device, seed=arguments.seed), sort_keys=True))


if __name__ == "__main__":
    main()

