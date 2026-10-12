"""Four-arm CNN×GRU adapter; importing starts no training or dataset access.

FULL and NONE retain their original model routes/metadata. Only CNN-only and
GRU-only are new process-local factory routes. The separately authorized sealed
campaign owns fresh fit execution, leases, input filtering and artifact guards.
This module's CLI operates on fabricated data only.
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

from models.TPPs.CountAwareTitanCNNGRUComponents import (
    FULL, NONE, CNN, GRU, ALL_ARMS, ARMS, CONTRACT_ID,
    ROLE_BY_ARM as COMPONENT_ROLES, ACTIVE_PARAMETERS,
    ACTIVE_CORRECTION_PARAMETERS, REGISTERED_PARAMETERS,
    CountAwareTitanCNNGRUComponents, correction_parameters,
    metadata as component_metadata, role_for_arm as component_role,
    validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_history_controls as original

BASELINE = FULL
SUPPORTED_ARMS = ALL_ARMS
VARIANT, SCHEMA = original.VARIANT, CONTRACT_ID
base, shared, common = original.base, original.shared, original.common
require, read = original.require, original.read
TIME_METRIC, audit_replay_accounting = original.TIME_METRIC, original.audit_replay_accounting
fabricated_data = original.fabricated_data
ROLE_BY_ARM = {FULL: original.cnn.role_for_arm(FULL), NONE: original.role_for_arm(NONE), **COMPONENT_ROLES}
_installed = False


def role_for_arm(arm):
    require(arm in ALL_ARMS, "Unknown component comparison arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm):
    require(arm in ALL_ARMS, "Unknown component comparison arm")
    if arm == FULL:
        return original.cnn.metadata(hidden_dim, arm=arm)
    if arm == NONE:
        return original.metadata(hidden_dim, arm=arm)
    return component_metadata(hidden_dim, arm=arm)


def install_hooks():
    global _installed
    if _installed:
        return
    original.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build = factory.build_count_aware_model
    signature = inspect.getclosurevars(original_build).nonlocals.get("signature")
    if not isinstance(signature, inspect.Signature) or "hidden_dim" not in signature.parameters:
        raise RuntimeError("Expected the original history-control public factory signature")
    original_route = factory.validate_checkpoint_route
    original_launch = observed_time.validate_launch
    original_role = constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone not in ARMS:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("Component comparison preserves memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = CountAwareTitanCNNGRUComponents(**values, component_arm=backbone)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=backbone), "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in COMPONENT_ROLES.values() or any(arm in ARMS for arm in backbones):
            if (len(backbones) != 1 or backbones[0] not in ARMS
                    or getattr(args, "model_role", None) != role_for_arm(backbones[0])):
                raise ValueError("Component comparison requires matching explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = role_for_arm(FULL)
            return original_launch(old_args, (FULL,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        backbones = kwargs["backbones"]
        if kwargs["model_role"] in COMPONENT_ROLES.values() or any(arm in ARMS for arm in backbones):
            if (len(backbones) != 1 or backbones[0] not in ARMS
                    or kwargs["model_role"] != role_for_arm(backbones[0])):
                raise ValueError("Component comparison requires matching explicit single-arm role")
            return original_role(**{**kwargs, "model_role": role_for_arm(FULL), "backbones": (FULL,)})
        return original_role(**kwargs)

    factory.build_count_aware_model = build
    factory.validate_checkpoint_route = route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = (*observed_time.ROLES, *COMPONENT_ROLES.values())
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.MODEL_ROLES = (*constants.MODEL_ROLES, *COMPONENT_ROLES.values())
    constants.SUPPORTED_BACKBONES = (*constants.SUPPORTED_BACKBONES, *ARMS)
    for arm in ARMS:
        constants.BACKBONE_LABELS[arm] = "TitanTPP CNN/GRU component " + arm.removeprefix(FULL + "_")
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in ALL_ARMS, "Unexpected CNN/GRU component comparison arm")
    if arm in (FULL, NONE):
        return original.build_model(data, arm)
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {key: value for key, value in data["model"].items()
        if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def _comparison_arms(arms):
    arms = tuple(arms)
    require(bool(arms) and len(set(arms)) == len(arms) and all(arm in ALL_ARMS for arm in arms),
        "Invalid CNN/GRU component comparison arms")
    return arms


def initial_states(data, seed, *, arms=ALL_ARMS):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    arms = _comparison_arms(arms)
    states = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        parent, _ = original.build_model(data, FULL)
        reference, rng = parent.state_dict(), torch.get_rng_state().clone()
        for arm in arms:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            state = model.state_dict()
            require(torch.equal(rng, torch.get_rng_state()), "Component constructor changed common RNG")
            for name, value in reference.items():
                require(name in state and torch.equal(value, state[name]), "Common initial tensor changed: " + name)
            extra = (set() if arm == FULL else {"cnn_history_control_identity"} if arm == NONE
                else {"cnn_gru_component_identity"})
            require(set(state) - set(reference) == extra, "Unexpected common state exclusion")
            states[arm] = canonical_state_dict_sha256(state)
    return states


initialization = initial_states


def training_args(c, data, output, arm):
    install_hooks()
    require(arm in ALL_ARMS, "Unknown component fit arm")
    plan = c["training"]
    require(plan.get("maximum_epochs") == 300 and plan.get("minimum_epochs") == 40
        and plan.get("patience") == 40 and plan.get("batch_size") == 128
        and plan.get("monitor") == "validation_raw_quantity_rmse"
        and plan.get("tie") == "strict_earliest_finite_minimum" and plan.get("warm_start") is False,
        "Frozen component training or selector changed")
    require(c.get("held_out_test_evaluated", False) is False
        and c.get("evaluation_scope", "validation_only") == "validation_only"
        and c.get("test_evaluation", False) is False, "Component study is Train/Validation only")
    require(not any(Path(output).glob("*.pt")), "Component fit must be fresh; checkpoint reuse forbidden")
    if arm in (FULL, NONE):
        args = original.training_args(c, data, output, arm)
    else:
        args = base.training_args({**c, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_cnn_gru_component_validation"
    require(args.force_rerun is False and args.checkpoint_monitor == "validation_raw_quantity_rmse",
        "Fresh fit or selector changed")
    return args


def time_interface(data, frame, c, *, arm):
    require(arm in ALL_ARMS, "Unknown component interface arm")
    if arm in (FULL, NONE):
        return original.time_interface(data, frame, c, arm=arm)
    interface = base.time_interface(data, frame, c)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": arm,
        "model_metadata": metadata(arm=arm), "training": c["training"]}
    return interface


def check_synthetic_causality(device="cpu", *, arms=ALL_ARMS):
    import torch
    from models.TPPs.CountAwareTitanCNNGRU import causal_depthwise_residual, observed_context
    arms = _comparison_arms(arms)
    devices = [] if str(device) == "cpu" else [torch.device(device)]
    compact_residuals, reset_residuals = {}, {}
    # Compact and padded inputs have different GEMM shapes. The unchanged
    # parent projection may round differently; same-shape leakage checks below
    # remain exact, and only these shape-changing checks use tight tolerance.
    def same_sequence(actual, expected, message):
        require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
            and torch.allclose(actual, expected, rtol=1e-5, atol=1e-6), message)
        return float((actual - expected).abs().max().item()) if actual.numel() else 0.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(321)
        dt = torch.tensor([[0., 2., 4., 6., 8., 10.]], device=device)
        qty = torch.tensor([[0., 2., 3., 4., 5., 6.]], device=device)
        valid = torch.ones_like(dt, dtype=torch.bool)
        observed = valid.clone()
        observed[:, -1] = False
        for arm in arms:
            model, _ = build_model(fabricated_data(), arm)
            model.to(device).eval()
            with torch.no_grad():
                for parameter in correction_parameters(model).values():
                    parameter.copy_(torch.randn_like(parameter) * .05)
                original_state = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)[0]
                changed_dt, changed_qty = dt.clone(), qty.clone()
                changed_dt[:, -1], changed_qty[:, -1] = float("nan"), float("inf")
                changed = model.encode_task_states(changed_dt, changed_qty, valid, memory_write_mask=observed)[0]
                require(torch.equal(original_state, changed), "Withheld row affected component state")
                require(bool((original_state[:, -1] == 0).all()), "Withheld state not zero")
                changed_dt, changed_qty = dt.clone(), qty.clone()
                changed_dt[:, 4], changed_qty[:, 4] = 1000., 9999.
                changed = model.encode_task_states(changed_dt, changed_qty, valid, memory_write_mask=observed)[0]
                require(torch.equal(original_state[:, :4], changed[:, :4]), "Future row affected component prefix")
                hidden = torch.randn(1, 7, 64, device=device)
                padded_valid = torch.ones(1, 7, dtype=torch.bool, device=device)
                padded_valid[:, 2] = False
                hidden[:, 2] = float("nan")
                if arm in (FULL, GRU):
                    padded = model.multilag_detail(hidden, padded_valid)
                    keep = [0, 1, 3, 4, 5, 6]
                    compact = model.multilag_detail(hidden[:, keep], padded_valid[:, keep])
                    compact_residuals[arm] = same_sequence(padded[:, keep], compact,
                        "GRU padding changed recurrent state")
                    withheld = torch.ones_like(padded_valid)
                    withheld[:, 2] = False
                    reset = model.multilag_detail(hidden, torch.ones_like(padded_valid), memory_write_mask=withheld)
                    require(bool((reset[:, 2:4] == 0).all()), "Withheld valid GRU row did not reset state/count")
                    fresh = model.multilag_detail(hidden[:, 3:], torch.ones_like(padded_valid[:, 3:]))
                    reset_residuals[arm] = same_sequence(reset[:, 3:], fresh,
                        "GRU segment reset differs from fresh prefix")
                if arm in (FULL, CNN):
                    observed_mask, sources, available = observed_context(padded_valid)
                    kernel = model.encoder.layers[0].attn.causal_q_kernel
                    padded = causal_depthwise_residual(hidden, kernel, sources, available)
                    keep = [0, 1, 3, 4, 5, 6]
                    _, sources, available = observed_context(padded_valid[:, keep])
                    compact = causal_depthwise_residual(hidden[:, keep], kernel, sources, available)
                    require(torch.equal(padded[:, keep], compact), "CNN padding changed observed lag context")
    return {"status": "passed", "device": str(device), "arms": list(arms),
        "future_prefix": "invariant", "withheld_target": "invariant",
        "padding": "skip", "withheld_valid": "reset", "real_data_loaded": False,
        "held_out_test_evaluated": False,
        "different_shape_comparison_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "gru_padding_max_absolute_residual": compact_residuals,
        "gru_reset_max_absolute_residual": reset_residuals}


def synthetic_check(device="cpu", *, arms=ALL_ARMS, seed=42, include_causality=True):
    """Every active correction element updates; inactive paths receive no gradient."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    arms, data = _comparison_arms(arms), fabricated_data()
    require(type(seed) is int and seed in (42, 52, 62), "Qualification seed changed")
    initial = initial_states(data, seed, arms=arms)
    counts, active_elements, frozen = {}, {}, {}
    devices = [] if str(device) == "cpu" else [torch.device(device)]
    with torch.random.fork_rng(devices=devices):
        generator = torch.Generator().manual_seed(121)
        dt = torch.randint(1, 8, (3, 12), generator=generator).float().to(device)
        qty = torch.randint(0, 20, (3, 12), generator=generator).float().to(device)
        dt[:, 0] = 0.
        mask = torch.ones_like(dt, dtype=torch.bool)
        reference, reference_rng = None, None
        for arm in arms:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            model.to(device).train()
            parameters = correction_parameters(model)
            require(sum(p.numel() for p in model.parameters()) == REGISTERED_PARAMETERS,
                "Registered component model parameter count changed")
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            require(counts[arm] == ACTIVE_PARAMETERS[arm], "Active model parameter budget changed")
            frozen[arm] = sorted(name for name, p in parameters.items() if not p.requires_grad)
            before = {name: p.detach().clone() for name, p in parameters.items()}
            torch.manual_seed(314)
            initial_output = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
            rng = torch.get_rng_state().clone()
            if reference is None:
                reference, reference_rng = initial_output, rng
            require(torch.equal(reference_rng, rng), "Initial forward changed dropout RNG")
            require(all(torch.equal(reference[key], initial_output[key]) for key in reference),
                "Zero initialized component output differs")
            coverage = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in parameters.items()}
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001, weight_decay=.01)
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(131 + step)
                loss = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite fabricated component loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    "Nonfinite fabricated component gradient")
                for name, p in parameters.items():
                    if not p.requires_grad:
                        require(p.grad is None, "Inactive component received a gradient: " + name)
                    elif p.grad is not None:
                        coverage[name] |= p.grad != 0
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            active = sum(int(value.sum()) for value in coverage.values())
            require(active == ACTIVE_CORRECTION_PARAMETERS[arm], "Inactive correction element: " + str(
                {name: int(value.sum()) for name, value in coverage.items()}))
            for name, p in parameters.items():
                require(bool((before[name] != p).any()) if p.requires_grad else torch.equal(before[name], p),
                    "Correction update/freeze mismatch: " + name)
            active_elements[arm] = active
            restored, _ = build_model(data, arm)
            restored.load_state_dict({name: value.cpu() for name, value in model.state_dict().items()}, strict=True)
    result = {"status": "passed", "device": str(device), "seed": seed,
        "initial_state_sha256": initial, "registered_parameter_count": REGISTERED_PARAMETERS,
        "active_trainable_parameter_count": counts, "active_correction_gradient_parameter_count": active_elements,
        "frozen_correction_tensors": frozen, "synthetic_optimizer_updates": 4 * len(arms),
        "real_data_loaded": False, "held_out_test_evaluated": False,
        "budget_limitation": "Different active counts; no capacity matching or causal synergy proof."}
    if include_causality:
        result["causality"] = check_synthetic_causality(device, arms=arms)
    return result


def _parent_replay_function(function):
    return FunctionType(function.__code__, {**function.__globals__,
        "SUPPORTED_ARMS": SUPPORTED_ARMS, "build_model": build_model,
        "_replay_validation": _replay_validation}, function.__name__, function.__defaults__, function.__closure__)


def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    return _parent_replay_function(original.cnn._replay_validation)(payload, path, data, frame,
        budget_check, device=device)


def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None,
        expected_identity=None, expected_initial=None, expected_epoch=None, expected_seed=42):
    install_hooks()
    return _parent_replay_function(original.cnn.replay_checkpoint)(path, data, frame, budget,
        device=device, expected_arm=expected_arm, expected_identity=expected_identity,
        expected_initial=expected_initial, expected_epoch=expected_epoch, expected_seed=expected_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-check", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, choices=(42, 52, 62), default=42)
    parser.add_argument("--arms", nargs="+", choices=ALL_ARMS, default=ALL_ARMS)
    args = parser.parse_args()
    print(json.dumps(synthetic_check(args.device, arms=args.arms, seed=args.seed), indent=2))
