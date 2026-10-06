"""Explicit process-local routes for five frozen CNN+GRU54 retrieval arms.

Importing this module loads no data, fits no model and installs no routes.
Research execution still requires its separately approved sealed campaign.
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

from models.TPPs.CountAwareTitanContextualRetrieval import (
    ARMS, BASELINE, ROLE_BY_ARM, EXTRA_PARAMETERS, CONTRACT_ID,
    CountAwareTitanContextualRetrieval, metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_gru as cnn

ALL_ARMS = SUPPORTED_ARMS = ARMS
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
        raise RuntimeError("Expected frozen CNN wrapper public factory signature")
    original_route = factory.validate_checkpoint_route
    original_launch = observed_time.validate_launch
    original_role = constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone not in ARMS:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("Retrieval comparison preserves inherited memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = CountAwareTitanContextualRetrieval(**values, retrieval_arm=backbone)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=backbone, statistics=model.retrieval_statistics),
            "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in ROLE_BY_ARM.values() or any(arm in ARMS for arm in backbones):
            if (len(backbones) != 1 or backbones[0] not in ARMS
                    or getattr(args, "model_role", None) != role_for_arm(backbones[0])):
                raise ValueError("Contextual retrieval requires matching explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = cnn.role_for_arm(BASELINE)
            return original_launch(old_args, (BASELINE,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        backbones = kwargs["backbones"]
        if kwargs["model_role"] in ROLE_BY_ARM.values() or any(arm in ARMS for arm in backbones):
            if (len(backbones) != 1 or backbones[0] not in ARMS
                    or kwargs["model_role"] != role_for_arm(backbones[0])):
                raise ValueError("Contextual retrieval requires matching explicit single-arm role")
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
        constants.BACKBONE_LABELS[arm] = "TitanTPP retrieval " + arm.rsplit("_", 1)[1]
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in SUPPORTED_ARMS, "Unexpected contextual retrieval comparison arm")
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {key: value for key, value in data["model"].items()
        if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def _comparison_arms(arms):
    arms = tuple(arms)
    require(bool(arms) and len(set(arms)) == len(arms) and all(arm in ARMS for arm in arms),
        "Invalid contextual retrieval comparison arms")
    return arms


def initial_states(data, seed, *, arms=ALL_ARMS):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    arms = _comparison_arms(arms)
    states, rngs = {}, {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        parent, _ = cnn.build_model(data, BASELINE)
        reference, reference_rng = parent.state_dict(), torch.get_rng_state().clone()
        for arm in arms:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            states[arm], rngs[arm] = model.state_dict(), torch.get_rng_state().clone()
            require(torch.equal(reference_rng, rngs[arm]), "Retrieval changed post-construction common RNG")
            for name, value in reference.items():
                require(name in states[arm] and torch.equal(value, states[arm][name]),
                    "Frozen CNN+GRU common initial tensor changed: " + name)
            permitted = {"contextual_retrieval_identity"} | {key for key in states[arm]
                if key.startswith("lmm.") and key != "lmm.mem"}
            require(set(states[arm]) - set(reference) == permitted, "Unexpected retrieval common-state exclusion")
    return {arm: canonical_state_dict_sha256(state) for arm, state in states.items()}


initialization = initial_states


def training_args(c, data, output, arm):
    install_hooks()
    require(arm in ARMS, "Only explicit contextual retrieval candidates are new fits")
    args = base.training_args({**c, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_contextual_retrieval_validation"
    return args


def _data_statistics(data):
    return {"time_scale": data["model"]["time_scale"],
        "time_log_mean": data["model"]["time_initial_location"],
        "time_log_std": data["model"]["time_initial_scale"],
        "quantity_log_mean": data["statistics"]["train_log_mean"],
        "quantity_log_std": data["statistics"]["train_log_std"]}


def time_interface(data, frame, c, *, arm):
    interface = base.time_interface(data, frame, c)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": arm,
        "model_metadata": metadata(arm=arm, statistics=_data_statistics(data)), "training": c["training"]}
    return interface


def fabricated_data():
    return {"model": {"hidden_dim": 64, "quantity_variant": VARIANT,
        "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 1.,
        "time_initial_location": .2, "time_initial_scale": .8,
        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "hour", "top_code": None}},
        "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": 256}}


def check_synthetic_causality(device="cpu"):
    """Fabricated observed-prefix checks; never read a research data file."""
    import torch
    data = fabricated_data()
    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [torch.device(device)]):
        torch.manual_seed(42)
        dt = torch.tensor([[0., 2., 4., 6., 8., 10.]], device=device)
        qty = torch.tensor([[0., 2., 3., 4., 5., 6.]], device=device)
        valid = torch.ones_like(dt, dtype=torch.bool)
        observed = valid.clone()
        observed[:, -1] = False
        for arm in ARMS:
            model, _ = build_model(data, arm)
            model.to(device).eval()
            with torch.no_grad():
                original = model.encode_task_states(dt, qty, valid, memory_write_mask=observed)[0]
                altered_dt, altered_qty = dt.clone(), qty.clone()
                altered_dt[:, -1], altered_qty[:, -1] = float("nan"), float("inf")
                altered = model.encode_task_states(altered_dt, altered_qty, valid, memory_write_mask=observed)[0]
                require(torch.equal(original, altered), "Withheld target changed observed retrieval states")
                features = model.observed_features(dt, qty, valid, memory_write_mask=observed)
                require(torch.equal(features[:, -1], torch.zeros_like(features[:, -1])), "Withheld features nonzero")
                future_dt, future_qty = dt.clone(), qty.clone()
                future_dt[:, 4], future_qty[:, 4] = 1000., 9999.
                future = model.encode_task_states(future_dt, future_qty, valid, memory_write_mask=observed)[0]
                require(torch.equal(original[:, :4], future[:, :4]), "Future value changed earlier retrieval states")
                # Segment reset concerns retrieval/CNN/GRU context. Parent
                # attention reference scope is deliberately retained unchanged.
                split = observed.clone()
                split[:, 2] = False
                reset_features = model.observed_features(dt, qty, valid, memory_write_mask=split)
                require(bool((reset_features[:, 2] == 0).all()), "Withheld segment features nonzero")
                require(bool((reset_features[:, 3, [2, 3, 6, 7]] == 0).all()), "Summary context crossed reset")
                padded_mask = valid.clone()
                padded_mask[:, 2] = False
                padded_dt, padded_qty = dt.clone(), qty.clone()
                padded_dt[:, 2], padded_qty[:, 2] = float("nan"), float("inf")
                padded = model.observed_features(padded_dt, padded_qty, padded_mask, memory_write_mask=observed)
                compact = model.observed_features(dt[:, [0, 1, 3, 4, 5]], qty[:, [0, 1, 3, 4, 5]],
                    valid[:, [0, 1, 3, 4, 5]], memory_write_mask=observed[:, [0, 1, 3, 4, 5]])
                require(torch.equal(padded[:, [0, 1, 3, 4, 5]], compact), "Padding consumed retrieval context")
    return {"status": "passed", "device": str(device), "arms": list(ARMS),
        "withheld_target": "invariant", "future_prefix": "invariant", "padding": "skip",
        "retrieval_segment_reset": "passed", "real_data_loaded": False, "held_out_test_evaluated": False}


def synthetic_check(device="cpu", *, arms=ALL_ARMS):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    data, arms = fabricated_data(), _comparison_arms(arms)
    initial = initial_states(data, 42, arms=arms)
    counts, extras, outputs, gradients, active_counts, gates = {}, {}, {}, {}, {}, {}
    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [torch.device(device)]):
        torch.manual_seed(42)
        dt = torch.randint(1, 8, (2, 12), device=device).float()
        dt[:, 0] = 0.
        quantity = torch.randint(0, 20, dt.shape, device=device).float()
        # Exercise the final observed difference feature deliberately: a
        # random equality at positions 9/10 would leave an entire query row
        # dormant without indicating an implementation fault.
        dt[:, 9], dt[:, 10] = 2., 7.
        quantity[:, 9], quantity[:, 10] = 2., 9.
        mask = torch.ones_like(dt, dtype=torch.bool)
        torch.manual_seed(42)
        parent, _ = cnn.build_model(data, BASELINE)
        parent.to(device).eval()
        with torch.no_grad():
            parent_outputs = target_outputs(parent, dt, mask, quantity, lambda_log_qty=1.)
        for arm in arms:
            torch.manual_seed(42)
            model, _ = build_model(data, arm)
            counts[arm] = sum(p.numel() for p in model.parameters())
            extras[arm] = sum(p.numel() for name, p in model.named_parameters()
                if name.startswith("lmm.") and name != "lmm.mem")
            model.to(device).eval()
            with torch.no_grad():
                outputs[arm] = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            active = set()
            coverage = {name: torch.zeros_like(p, dtype=torch.bool) for name, p in model.named_parameters()
                if name.startswith("lmm.") and name != "lmm.mem"}
            model.train()
            for _ in range(4):
                optimizer.zero_grad(set_to_none=True)
                loss = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite fabricated loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                    "Nonfinite fabricated gradient")
                for name, p in model.named_parameters():
                    if name.startswith("lmm.") and name != "lmm.mem" and p.grad is not None and bool((p.grad != 0).any()):
                        active.add(name)
                        coverage[name] |= p.grad != 0
                optimizer.step()
            expected_active = {name for name, _ in model.named_parameters() if name.startswith("lmm.") and name != "lmm.mem"}
            require(active == expected_active, "Dormant fabricated retrieval score parameter tensor")
            gradients[arm] = sorted(active)
            active_counts[arm] = sum(int(value.sum()) for value in coverage.values())
            require(active_counts[arm] == EXTRA_PARAMETERS[arm], "Dormant retrieval score parameter element: "
                + arm + " " + str({name: int(value.sum()) for name, value in coverage.items()}))
            if arm in ARMS[3:]:
                require(bool((model.lmm.context_gate_weight != 0).any())
                    and bool(model.lmm.context_gate_bias != 0), "Fabricated gate failed to learn after updates")
                gates[arm] = "nonzero_after_four_fabricated_updates"
            restored, _ = build_model(data, arm)
            restored.load_state_dict({key: value.cpu() for key, value in model.state_dict().items()}, strict=True)
        for arm in arms:
            require(counts[arm] == 115167 + EXTRA_PARAMETERS[arm] and extras[arm] == EXTRA_PARAMETERS[arm],
                "Contextual retrieval active parameter budget changed")
        if ARMS[0] in arms:
            for key in parent_outputs:
                require(torch.equal(parent_outputs[key], outputs[ARMS[0]][key]),
                    "Mean retrieval changed frozen CNN+GRU full output: " + key)
        weighted = next((arm for arm in ARMS[1:] if arm in arms), None)
        if weighted is not None:
            for arm in (candidate for candidate in ARMS[1:] if candidate in arms):
                for key in outputs[weighted]:
                    require(torch.equal(outputs[weighted][key], outputs[arm][key]),
                        "Zero initialized contextual search differs from cosine weighted baseline: " + key)
    return {"status": "passed", "device": str(device), "parameter_count": counts,
        "retrieval_extra_parameter_count": extras, "initial_state_sha256": initial,
        "active_score_gradient_tensors": gradients, "gate_learning": gates,
        "active_score_gradient_parameter_count": active_counts,
        "causality": check_synthetic_causality(device), "synthetic_optimizer_updates": 4 * len(arms),
        "real_data_loaded": False, "held_out_test_evaluated": False,
        "deterministic_gpu_qualification": "not_performed" if str(device) == "cpu" else "candidate_checks_only"}


def _parent_replay_function(function):
    # Reuse the frozen streaming replay bytecode with only its arm registry
    # and builder replaced in a private globals mapping. No parent mutation.
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
