"""Opt-in process-local routes for the representative CNN history controls.

The full arm is the unchanged original CNN+GRU54 route, also used as retrieval
mean. Importing this module starts no execution and installs no routes. Native
training needs the separately sealed campaign; CLI is fabricated checks only.
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

from models.TPPs.CountAwareTitanCNNHistoryControls import (
    ARMS, BASELINE, CURRENT_ONLY, DISABLED, CONTRACT_ID, ROLE_BY_ARM,
    CORRECTION_PARAMETERS, CountAwareTitanCNNHistoryControl,
    correction_parameters, metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_gru as cnn

ALL_ARMS = SUPPORTED_ARMS = (BASELINE, *ARMS)
VARIANT, SCHEMA = cnn.VARIANT, CONTRACT_ID
base, shared, common, width = cnn.base, cnn.shared, cnn.common, cnn.width
require, read = cnn.require, cnn.read
TIME_METRIC, audit_replay_accounting = cnn.TIME_METRIC, cnn.audit_replay_accounting
_installed = False


def install_hooks():
    global _installed
    if _installed:
        return
    cnn.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build = factory.build_count_aware_model
    signature = inspect.getclosurevars(original_build).nonlocals.get("signature")
    if not isinstance(signature, inspect.Signature) or "hidden_dim" not in signature.parameters:
        raise RuntimeError("Expected original CNN wrapper public factory signature")
    original_route = factory.validate_checkpoint_route
    original_launch = observed_time.validate_launch
    original_role = constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone not in ARMS:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("CNN history controls preserve memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = CountAwareTitanCNNHistoryControl(**values, control_arm=backbone)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=backbone), "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in ROLE_BY_ARM.values() or any(arm in ARMS for arm in backbones):
            if (len(backbones) != 1 or backbones[0] not in ARMS
                    or getattr(args, "model_role", None) != role_for_arm(backbones[0])):
                raise ValueError("CNN history controls require matching single-arm role")
            old_args = copy(args)
            old_args.model_role = cnn.role_for_arm(BASELINE)
            return original_launch(old_args, (BASELINE,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        backbones = kwargs["backbones"]
        if kwargs["model_role"] in ROLE_BY_ARM.values() or any(arm in ARMS for arm in backbones):
            if (len(backbones) != 1 or backbones[0] not in ARMS
                    or kwargs["model_role"] != role_for_arm(backbones[0])):
                raise ValueError("CNN history controls require matching single-arm role")
            return original_role(**{**kwargs, "model_role": cnn.role_for_arm(BASELINE), "backbones": (BASELINE,)})
        return original_role(**kwargs)

    factory.build_count_aware_model = build
    factory.validate_checkpoint_route = route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = (*observed_time.ROLES, *ROLE_BY_ARM.values())
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.MODEL_ROLES = (*constants.MODEL_ROLES, *ROLE_BY_ARM.values())
    constants.SUPPORTED_BACKBONES = (*constants.SUPPORTED_BACKBONES, *ARMS)
    for arm in ARMS:
        constants.BACKBONE_LABELS[arm] = "TitanTPP CNN history control " + arm.removeprefix(BASELINE + "_")
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in SUPPORTED_ARMS, "Unexpected CNN history comparison arm")
    if arm == BASELINE:
        return cnn.build_model(data, BASELINE)
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {key: value for key, value in data["model"].items()
        if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def _comparison_arms(arms):
    arms = tuple(arms)
    require(bool(arms) and len(set(arms)) == len(arms) and all(arm in SUPPORTED_ARMS for arm in arms),
        "Invalid CNN history comparison arms")
    return arms


def initial_states(data, seed, *, arms=ALL_ARMS):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    arms = _comparison_arms(arms)
    states = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        parent, _ = cnn.build_model(data, BASELINE)
        reference, reference_rng = parent.state_dict(), torch.get_rng_state().clone()
        for arm in arms:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            state = model.state_dict()
            require(torch.equal(reference_rng, torch.get_rng_state()), "CNN history control changed construction RNG")
            for name, value in reference.items():
                require(name in state and torch.equal(value, state[name]), "Original common initial tensor changed: " + name)
            require(set(state) - set(reference) == (set() if arm == BASELINE else {"cnn_history_control_identity"}),
                "Unexpected CNN history control common-state exclusion")
            states[arm] = canonical_state_dict_sha256(state)
    return states


initialization = initial_states


def training_args(c, data, output, arm):
    install_hooks()
    require(arm in SUPPORTED_ARMS, "Only explicit CNN history candidates are new fits")
    if arm == BASELINE:
        return cnn.training_args(c, data, output, BASELINE)
    args = base.training_args({**c, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_cnn_history_control_validation"
    return args


def time_interface(data, frame, c, *, arm):
    if arm == BASELINE:
        return cnn.time_interface(data, frame, c, arm=BASELINE)
    interface = base.time_interface(data, frame, c)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": arm,
        "model_metadata": metadata(arm=arm), "training": c["training"]}
    return interface


def fabricated_data():
    return {"model": {"hidden_dim": 64, "quantity_variant": VARIANT,
        "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 1.,
        "time_initial_location": .2, "time_initial_scale": .8,
        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None}},
        "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": 256}}


def check_synthetic_causality(device="cpu"):
    """Fabricated prefix safety and no direct prior-state correction inputs."""
    import torch
    from models.TPPs.CountAwareTitanCNNHistoryControls import current_pointwise_residual
    from models.TPPs.CountAwareTitanCNNGRU import observed_context
    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [torch.device(device)]):
        torch.manual_seed(321)
        dt = torch.tensor([[0., 2., 4., 6., 8., 10.]], device=device)
        qty = torch.tensor([[0., 2., 3., 4., 5., 6.]], device=device)
        valid = torch.ones_like(dt, dtype=torch.bool)
        observed = valid.clone()
        observed[:, -1] = False
        for arm in ARMS:
            model, _ = build_model(fabricated_data(), arm)
            model.to(device).eval()
            with torch.no_grad():
                for parameter in correction_parameters(model).values():
                    parameter.copy_(torch.randn_like(parameter) * .05)
                original = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)[0]
                altered_dt, altered_qty = dt.clone(), qty.clone()
                altered_dt[:, -1], altered_qty[:, -1] = float("nan"), float("inf")
                altered = model.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
                require(torch.equal(original, altered), "Withheld event affected Q1 state")
                require(bool((original[:, -1] == 0).all()), "Withheld state was not zero")
                altered_dt, altered_qty = dt.clone(), qty.clone()
                altered_dt[:, 4], altered_qty[:, 4] = 1000., 9999.
                future = model.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
                require(torch.equal(original[:, :4], future[:, :4]), "Q1 future event affected earlier state")
                if arm == CURRENT_ONLY:
                    hidden = torch.randn(1, 6, 64, device=device)
                    changed = hidden.clone()
                    changed[:, :-1] += 5.
                    require(torch.equal(model.multilag_detail(hidden, valid)[:, -1],
                        model.multilag_detail(changed, valid)[:, -1]), "Current-only GRU consumed past hidden row")
                    _, _, availability = observed_context(valid)
                    kernel = model.encoder.layers[0].attn.causal_q_kernel
                    require(torch.equal(current_pointwise_residual(hidden, kernel, availability)[:, -1],
                        current_pointwise_residual(changed, kernel, availability)[:, -1]),
                        "Current-only CNN consumed past hidden row")
    return {"status": "passed", "device": str(device), "arms": list(ARMS),
        "future_prefix": "invariant", "withheld_target": "invariant",
        "current_only_explicit_prior_hidden_input": "absent", "encoder_prefix_information": "retained",
        "real_data_loaded": False, "held_out_test_evaluated": False}


def synthetic_check(device="cpu", *, arms=ALL_ARMS):
    """Check exact common initialization and per-element active budget on fake data."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    arms, data = _comparison_arms(arms), fabricated_data()
    initial = initial_states(data, 42, arms=arms)
    counts, coverage_counts, gradients, frozen = {}, {}, {}, {}
    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [torch.device(device)]):
        generator = torch.Generator().manual_seed(121)
        dt = torch.randint(1, 8, (3, 12), generator=generator).float().to(device)
        qty = torch.randint(0, 20, (3, 12), generator=generator).float().to(device)
        dt[:, 0] = 0.
        mask = torch.ones_like(dt, dtype=torch.bool)
        for arm in arms:
            torch.manual_seed(42)
            model, _ = build_model(data, arm)
            model.to(device).train()
            parameters = correction_parameters(model)
            require(sum(p.numel() for p in parameters.values()) == CORRECTION_PARAMETERS,
                "Original correction registration budget changed")
            counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            require(counts[arm] == (89859 if arm == DISABLED else 115167), "Active trainable budget changed")
            frozen[arm] = sorted(name for name, p in parameters.items() if not p.requires_grad)
            before = {name: p.detach().clone() for name, p in parameters.items()}
            coverage = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in parameters.items()}
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.001)
            for step in range(4):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(131 + step)
                loss = target_outputs(model, dt, mask, qty, lambda_log_qty=1.)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite fabricated CNN history loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    "Nonfinite fabricated CNN history gradient")
                for name, p in parameters.items():
                    if p.grad is not None:
                        coverage[name] |= p.grad != 0
                optimizer.step()
            active = sum(int(value.sum()) for value in coverage.values())
            require(active == (0 if arm == DISABLED else CORRECTION_PARAMETERS),
                "Inactive CNN history correction element: " + str({name: int(v.sum()) for name, v in coverage.items()}))
            if arm == DISABLED:
                require(all(torch.equal(before[name], p) for name, p in parameters.items()), "Disabled correction changed")
            else:
                require(all(bool((before[name] != p).any()) for name, p in parameters.items()),
                    "An active correction tensor did not update")
            coverage_counts[arm] = active
            gradients[arm] = sorted(name for name, value in coverage.items() if bool(value.any()))
            restored, _ = build_model(data, arm)
            restored.load_state_dict({name: value.cpu() for name, value in model.state_dict().items()}, strict=True)
    return {"status": "passed", "device": str(device), "initial_state_sha256": initial,
        "active_trainable_parameter_count": counts, "active_correction_gradient_parameter_count": coverage_counts,
        "active_correction_gradient_tensors": gradients, "frozen_correction_tensors": frozen,
        "synthetic_optimizer_updates": 4 * len(arms), "real_data_loaded": False,
        "held_out_test_evaluated": False, "causality": check_synthetic_causality(device),
        "budget_limitation": "Current-only pointwise bases/two-step depth differ; native costs must be reported."}


def _parent_replay_function(function):
    return FunctionType(function.__code__, {**function.__globals__,
        "SUPPORTED_ARMS": SUPPORTED_ARMS, "build_model": build_model,
        "_replay_validation": _replay_validation}, function.__name__, function.__defaults__, function.__closure__)


def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    return _parent_replay_function(cnn._replay_validation)(payload, path, data, frame, budget_check, device=device)


def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None,
        expected_identity=None, expected_initial=None, expected_epoch=None, expected_seed=42):
    install_hooks()
    return _parent_replay_function(cnn.replay_checkpoint)(path, data, frame, budget, device=device,
        expected_arm=expected_arm, expected_identity=expected_identity, expected_initial=expected_initial,
        expected_epoch=expected_epoch, expected_seed=expected_seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-check", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--arms", nargs="+", choices=SUPPORTED_ARMS, default=ALL_ARMS)
    args = parser.parse_args()
    print(json.dumps(synthetic_check(args.device, arms=args.arms), indent=2))
