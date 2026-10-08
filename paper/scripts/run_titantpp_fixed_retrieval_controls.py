"""Explicit process-local routing for the prospective active512 retrieval control.

Importing this module reads no dataset and installs no routes. Campaign launch,
lease, source freezing and full Validation accounting stay with the sealed owner.
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

from models.TPPs.CountAwareTitanFixedRetrievalControls import (
    ARMS, BASELINE, GENERIC512, ROLE_BY_ARM, EXTRA_PARAMETERS, CONTRACT_ID,
    CountAwareTitanFixedRetrievalControls, identifiability_contract, metadata,
    role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_contextual_retrieval as contextual

ALL_ARMS = SUPPORTED_ARMS = ARMS
VARIANT, SCHEMA = contextual.VARIANT, CONTRACT_ID
cnn = contextual.cnn
base, shared, common, width = contextual.base, contextual.shared, contextual.common, contextual.width
require, read = contextual.require, contextual.read
TIME_METRIC, audit_replay_accounting = contextual.TIME_METRIC, contextual.audit_replay_accounting
_installed = False


def install_hooks():
    """Add the new explicit route without modifying existing source or routes."""
    global _installed
    if _installed:
        return
    contextual.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build = factory.build_count_aware_model
    signature = inspect.getclosurevars(original_build).nonlocals.get("signature")
    if not isinstance(signature, inspect.Signature) or "hidden_dim" not in signature.parameters:
        raise RuntimeError("Expected contextual wrapper public factory signature")
    original_route = factory.validate_checkpoint_route
    original_launch = observed_time.validate_launch
    original_role = constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone != GENERIC512:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("Generic512 preserves inherited memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = CountAwareTitanFixedRetrievalControls(**values, retrieval_arm=backbone)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=backbone, statistics=model.retrieval_statistics),
            "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if expected_backbone == GENERIC512 or payload.get("backbone") == GENERIC512:
            if validate_checkpoint(payload, expected_backbone):
                return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) == role_for_arm(GENERIC512) or GENERIC512 in backbones:
            if tuple(backbones) != (GENERIC512,) or getattr(args, "model_role", None) != role_for_arm(GENERIC512):
                raise ValueError("Generic512 requires matching explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = contextual.role_for_arm(ARMS[2])
            return original_launch(old_args, (ARMS[2],), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        backbones = kwargs["backbones"]
        if kwargs["model_role"] == role_for_arm(GENERIC512) or GENERIC512 in backbones:
            if tuple(backbones) != (GENERIC512,) or kwargs["model_role"] != role_for_arm(GENERIC512):
                raise ValueError("Generic512 requires matching explicit single-arm role")
            return original_role(**{**kwargs, "model_role": contextual.role_for_arm(ARMS[2]), "backbones": (ARMS[2],)})
        return original_role(**kwargs)

    factory.build_count_aware_model = build
    factory.validate_checkpoint_route = route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = (*observed_time.ROLES, role_for_arm(GENERIC512))
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.MODEL_ROLES = (*constants.MODEL_ROLES, role_for_arm(GENERIC512))
    constants.SUPPORTED_BACKBONES = (*constants.SUPPORTED_BACKBONES, GENERIC512)
    constants.BACKBONE_LABELS[GENERIC512] = "TitanTPP retrieval generic active512"
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in ARMS, "Unexpected fixed retrieval comparison arm")
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {key: value for key, value in data["model"].items()
        if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def _comparison_arms(arms):
    arms = tuple(arms)
    require(bool(arms) and len(set(arms)) == len(arms) and all(arm in ARMS for arm in arms),
        "Invalid fixed retrieval comparison arms")
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
            states[arm] = model.state_dict()
            require(torch.equal(reference_rng, torch.get_rng_state()), "Retrieval changed postconstruction common RNG")
            for name, value in reference.items():
                require(name in states[arm] and torch.equal(value, states[arm][name]),
                    "Frozen CNN+GRU common initial tensor changed: " + name)
            permitted = {"contextual_retrieval_identity"} | {key for key in states[arm]
                if key.startswith("lmm.") and key != "lmm.mem"}
            require(set(states[arm]) - set(reference) == permitted, "Unexpected retrieval common-state exclusion")
    return {arm: canonical_state_dict_sha256(state) for arm, state in states.items()}


initialization = initial_states
fabricated_data = contextual.fabricated_data


def training_args(c, data, output, arm):
    install_hooks()
    require(arm in ARMS, "Only explicit fixed retrieval candidates are new fits")
    args = base.training_args({**c, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_fixed_retrieval_controls_validation"
    return args


def time_interface(data, frame, c, *, arm):
    require(arm in ARMS, "Unexpected fixed retrieval interface arm")
    interface = base.time_interface(data, frame, c)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": arm,
        "model_metadata": metadata(arm=arm, statistics=contextual._data_statistics(data)),
        "identifiability": identifiability_contract(), "training": c["training"]}
    return interface


def retrieval_diagnostics(model, dts, history_quantities, mask, *, memory_write_mask=None):
    """Audit inherited frozen models without replacing their execution path."""
    require(getattr(model, "retrieval_arm", None) in ARMS, "Unexpected retrieval diagnostic model")
    return CountAwareTitanFixedRetrievalControls.retrieval_diagnostics(model, dts, history_quantities, mask,
        memory_write_mask=memory_write_mask)


def check_synthetic_causality(device="cpu"):
    """Bounded fabricated observations only, independent of research splits."""
    import torch
    data = fabricated_data()
    dt = torch.tensor([[0., 2., 4., 6., 8., 10.]], device=device)
    quantity = torch.tensor([[0., 2., 3., 4., 5., 6.]], device=device)
    valid = torch.ones_like(dt, dtype=torch.bool)
    observed = valid.clone()
    observed[:, -1] = False
    for arm in ARMS:
        torch.manual_seed(42)
        model, _ = build_model(data, arm)
        model.to(device).eval()
        with torch.no_grad():
            reference = model.encode_task_states(dt, quantity, valid, memory_write_mask=observed)[0]
            altered_dt, altered_qty = dt.clone(), quantity.clone()
            altered_dt[:, -1], altered_qty[:, -1] = float("nan"), float("inf")
            altered = model.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
            require(torch.equal(reference, altered), "Withheld target changed observed retrieval states")
            altered_dt, altered_qty = dt.clone(), quantity.clone()
            altered_dt[:, 4], altered_qty[:, 4] = 1000., 9999.
            altered = model.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
            require(torch.equal(reference[:, :4], altered[:, :4]), "Future value changed earlier retrieval states")
            padded = valid.clone()
            padded[:, 2] = False
            original_features = model.observed_features(dt, quantity, padded, memory_write_mask=observed)
            keep = [0, 1, 3, 4, 5]
            compact_features = model.observed_features(dt[:, keep], quantity[:, keep], valid[:, keep], memory_write_mask=observed[:, keep])
            require(torch.equal(original_features[:, keep], compact_features), "Padding consumed retrieval context")
            reset = observed.clone()
            reset[:, 2] = False
            reset_features = model.observed_features(dt, quantity, valid, memory_write_mask=reset)
            require(bool((reset_features[:, 2] == 0).all()) and bool((reset_features[:, 3, [2, 3, 6, 7]] == 0).all()),
                "Retrieval summary crossed withheld segment reset")
    return {"status": "passed", "arms": list(ARMS), "device": str(device),
        "future_prefix": "invariant", "withheld_target": "invariant", "padding": "skip",
        "retrieval_segment_reset": "passed", "real_data_loaded": False, "held_out_test_evaluated": False}


def synthetic_check(device="cpu", *, arms=ALL_ARMS):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    arms, data = _comparison_arms(arms), fabricated_data()
    initial = initial_states(data, 42, arms=arms)
    counts, extras, active_counts, outputs = {}, {}, {}, {}
    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [torch.device(device)]):
        torch.manual_seed(42)
        dt = torch.randint(1, 8, (2, 12), device=device).float()
        dt[:, 0] = 0.
        quantity = torch.randint(0, 20, dt.shape, device=device).float()
        dt[:, 9], dt[:, 10], quantity[:, 9], quantity[:, 10] = 2., 7., 2., 9.
        mask = torch.ones_like(dt, dtype=torch.bool)
        for arm in arms:
            torch.manual_seed(42)
            model, _ = build_model(data, arm)
            model.to(device).eval()
            counts[arm] = sum(parameter.numel() for parameter in model.parameters())
            score_parameters = {name: parameter for name, parameter in model.named_parameters()
                if name.startswith("lmm.") and name != "lmm.mem"}
            extras[arm] = sum(parameter.numel() for parameter in score_parameters.values())
            coverage = {name: torch.zeros_like(parameter, dtype=torch.bool) for name, parameter in score_parameters.items()}
            with torch.no_grad():
                outputs[arm] = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            model.train()
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite fabricated loss")
                loss.backward()
                require(all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()),
                    "Nonfinite fabricated gradient")
                for name, parameter in score_parameters.items():
                    if parameter.grad is not None:
                        coverage[name] |= parameter.grad != 0
                optimizer.step()
            active_counts[arm] = sum(int(value.sum()) for value in coverage.values())
            require(counts[arm] == 115167 + EXTRA_PARAMETERS[arm] and extras[arm] == EXTRA_PARAMETERS[arm],
                "Fixed retrieval budget changed")
            require(active_counts[arm] == EXTRA_PARAMETERS[arm], "Dormant retrieval score parameter element")
            restored, _ = build_model(data, arm)
            restored.load_state_dict({key: value.cpu() for key, value in model.state_dict().items()}, strict=True)
        if ARMS[1] in arms:
            for arm in ARMS[2:]:
                if arm in arms:
                    require(all(torch.equal(value, outputs[arm][key]) for key, value in outputs[ARMS[1]].items()),
                        "Zero-initialized learned query changed weighted baseline")
    return {"status": "passed", "device": str(device), "parameter_count": counts,
        "retrieval_extra_parameter_count": extras, "active_score_gradient_parameter_count": active_counts,
        "initial_state_sha256": initial, "causality": check_synthetic_causality(device),
        "identifiability": identifiability_contract(), "synthetic_optimizer_updates": 4 * len(arms),
        "real_data_loaded": False, "held_out_test_evaluated": False}


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
