"""Explicit six-arm contribution routing, native checks and Validation replay.

Importing this module performs no fit, data loading or remote operation. The
owned campaign must supply approval, immutable source and native permits.
"""
from __future__ import annotations

from copy import copy
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareTitanArchitectureContribution import (
    ARMS, BACKBONE_BY_ARM, CountAwareTitanArchitectureContribution,
    arm_for_backbone, backbone_for_arm, metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_history_width as width

SUPPORTED_ARMS = tuple(BACKBONE_BY_ARM.values())
VARIANT = width.VARIANT
SCHEMA = "titantpp_architecture_contribution_v1"
base, shared, common = width.base, width.shared, width.common
require, read = width.require, width.read
TIME_METRIC, audit_replay_accounting = width.TIME_METRIC, width.audit_replay_accounting
_installed = False


def design_arm(arm):
    return arm if arm in ARMS else arm_for_backbone(arm)


def install_hooks():
    """Route only explicit new identities in the current interpreter."""
    global _installed
    if _installed:
        return
    width.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build = factory.build_count_aware_model
    signature = inspect.getclosurevars(original_build).nonlocals.get("signature")
    require(isinstance(signature, inspect.Signature), "Public factory signature missing")
    original_route = factory.validate_checkpoint_route
    original_launch = observed_time.validate_launch
    original_role = constants.validate_model_role_contract
    roles = tuple(role_for_arm(a) for a in ARMS)

    def build(backbone, **kwargs):
        if backbone not in SUPPORTED_ARMS:
            return original_build(backbone, **kwargs)
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        require(values.pop("titans_memory_gradient_clip") is None,
                "Contribution intervention changes no memory gradient policy")
        arm = arm_for_backbone(backbone)
        model = CountAwareTitanArchitectureContribution(**values, architecture_arm=arm)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"], arm=arm), "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in roles or any(a in SUPPORTED_ARMS for a in backbones):
            require(len(backbones) == 1 and backbones[0] in SUPPORTED_ARMS
                    and getattr(args, "model_role", None) == role_for_arm(arm_for_backbone(backbones[0])),
                    "Contribution requires its matching explicit single-arm role")
            old = copy(args)
            old.model_role = width.role_for_arm(width.ARM)
            return original_launch(old, (width.ARM,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        backbones = kwargs["backbones"]
        if kwargs["model_role"] in roles or any(a in SUPPORTED_ARMS for a in backbones):
            require(len(backbones) == 1 and backbones[0] in SUPPORTED_ARMS
                    and kwargs["model_role"] == role_for_arm(arm_for_backbone(backbones[0])),
                    "Contribution requires its matching explicit single-arm role")
            return original_role(**{**kwargs, "model_role": width.role_for_arm(width.ARM),
                                    "backbones": (width.ARM,)})
        return original_role(**kwargs)

    factory.build_count_aware_model = build
    factory.validate_checkpoint_route = route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = tuple(dict.fromkeys((*observed_time.ROLES, *roles)))
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.MODEL_ROLES = tuple(dict.fromkeys((*constants.MODEL_ROLES, *roles)))
    constants.SUPPORTED_BACKBONES = tuple(dict.fromkeys((*constants.SUPPORTED_BACKBONES, *SUPPORTED_ARMS)))
    for arm in ARMS:
        constants.BACKBONE_LABELS[backbone_for_arm(arm)] = "TitanTPP contribution " + arm
    _installed = True


def build_model(data, arm):
    install_hooks()
    from models.TPPs.CountAwareFactory import build_count_aware_model
    arm = design_arm(arm)
    config = {k: v for k, v in data["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(backbone_for_arm(arm), **config,
        train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def initial_states(data, seed):
    """Each optional package has forked RNG; shared tensors and RNG must match."""
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states, rngs = {}, {}
    with torch.random.fork_rng(devices=[]):
        for arm in ARMS:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            states[arm] = model.state_dict()
            rngs[arm] = torch.get_rng_state().clone()
        baseline = states["P0_H0"]
        for arm in ARMS:
            require(torch.equal(rngs["P0_H0"], rngs[arm]), "Optional package changed common RNG")
            for name, value in baseline.items():
                if name.endswith("identity"):
                    continue
                require(name in states[arm] and torch.equal(value, states[arm][name]),
                        "Common initial tensor changed " + name)
    return {arm: canonical_state_dict_sha256(state) for arm, state in states.items()}


initialization = initial_states


def training_args(c, data, output, arm):
    install_hooks()
    arm = design_arm(arm)
    args = base.training_args({**c, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_architecture_contribution_validation"
    return args


def time_interface(data, frame, c, *, arm):
    interface = base.time_interface(data, frame, c)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": design_arm(arm),
        "model_metadata": metadata(arm=design_arm(arm)), "training": c["training"],
        "design_sha256": c["design_sha256"]}
    return interface


def synthetic_data(length):
    return {"model": {"hidden_dim": 64, "quantity_variant": VARIANT,
        "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 7.,
        "time_initial_location": .2, "time_initial_scale": .8,
        "time_observation_contract": {"mode": "positive_integer_round_clamp_v1", "unit": "week", "top_code": None}},
        "statistics": {"train_log_mean": 1.2, "train_log_std": .8}, "loader": {"max_seq_len": length}}


def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None,
        expected_identity=None, expected_initial=None, expected_epoch=None, expected_seed=42):
    install_hooks()
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    payload = torch_load_checkpoint(Path(path), map_location="cpu")
    if expected_arm is not None:
        from paper.scripts.count_aware_tpp_backbone.training import checkpoint_monitor_spec
        selector = checkpoint_monitor_spec("validation_raw_quantity_rmse")
        require(payload.get("backbone") == expected_arm and payload.get("seed") == expected_seed
            and payload.get("variant") == VARIANT, "Replay arm/seed/objective mismatch")
        require(payload.get("resume_identity") == expected_identity
            and payload.get("interface_meta") == expected_identity["interface_meta"]
            and payload.get("initial_state_sha256") == expected_initial, "Replay source/interface identity mismatch")
        revision = expected_identity.get("arguments", {}).get("source_revision")
        if revision is not None:
            require(payload.get("source_revision") == revision and payload.get("source_revision_history") == [revision],
                "Replay source revision mismatch")
        require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
            and payload.get("checkpoint_monitor_history_key") == selector["history_key"]
            and payload.get("checkpoint_selection") == selector["selection"], "Replay selector mismatch")
        require(payload.get("epoch", payload.get("best_epoch")) == expected_epoch, "Replay epoch mismatch")
    require(payload.get("interface_meta", {}).get("time_head", {}).get("observation_likelihood")
        == data["model"]["time_observation_contract"], "Replay observation likelihood mismatch")
    result = _replay_validation(payload, path, data, frame, budget, device=device)
    result["time_metric"] = TIME_METRIC
    return result


def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    """Streaming inherited validation metrics at a fixed checkpoint; no records."""
    import numpy as np
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(payload.get("backbone") in SUPPORTED_ARMS, "Unexpected endpoint backbone")
    validate_checkpoint_route(payload, payload["backbone"])
    require(payload["evaluation_scope"] == "validation_only" and payload["held_out_test_evaluated"] is False,
        "Replay split changed")
    state_digest = canonical_state_dict_sha256(payload["model_state_dict"])
    require(payload.get("model_state_sha256") == state_digest, "Checkpoint state digest differs")
    model, _ = build_model(data, payload["backbone"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = make_loader(frame, target_split="validation", **{key: data["loader"][key]
        for key in ("batch_size", "lookback_weeks", "max_seq_len")}, shuffle=False, generator=None)
    q_bounds, h_bounds = data["quantity_boundaries_all_train_rows"], data["history_boundaries"]
    partitions = [("quantity", q_bounds), ("history", h_bounds)]
    if "additional_history_boundaries" in data:
        partitions.append(("additional_history", data["additional_history_boundaries"]))
    accumulator = {key: [0, 0., 0., 0.] for key in ("overall", "body", "tail")}
    for kind, bounds in partitions:
        accumulator.update({f"{kind}_{index}": [0, 0., 0., 0.] for index in range(len(bounds) + 1)})
    with torch.no_grad():
        for _, dts, mask, _, quantities in loader:
            budget_check()
            result = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
            q, pred = result["true_qty"].cpu().numpy().astype(np.float64), result["pred_qty"].cpu().numpy().astype(np.float64)
            temporal, histories = result["time_loss"].cpu().numpy().astype(np.float64), result["history_length"].cpu().numpy()
            require(np.isfinite(q).all() and np.isfinite(pred).all() and np.isfinite(temporal).all(), "Nonfinite checkpoint replay")
            masks = {"overall": np.ones(q.shape, dtype=bool), "body": q <= q_bounds[2], "tail": q > q_bounds[3]}
            for kind, bounds in partitions:
                indices = np.searchsorted(bounds, q if kind == "quantity" else histories, side="left")
                masks.update({f"{kind}_{index}": indices == index for index in range(len(bounds) + 1)})
            error = pred - q
            for key, selected in masks.items():
                acc = accumulator[key]
                acc[0] += int(selected.sum())
                acc[1] += float(np.abs(error[selected]).sum())
                acc[2] += float(np.square(error[selected]).sum())
                acc[3] += float(temporal[selected].sum())
    budget_check()
    def finish(acc):
        count, absolute, squared, temporal = acc
        return {"count": count, "qty_mae": absolute / count if count else None, "qty_sse": squared,
            "qty_rmse": math.sqrt(squared / count) if count else None, "time_nll": temporal / count if count else None}
    result = {**finish(accumulator["overall"]), "body": finish(accumulator["body"]), "tail": finish(accumulator["tail"]),
        "checkpoint_path": str(path), "state_sha256": state_digest, "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False}
    for kind, bounds in partitions:
        result[kind + "_boundaries"] = bounds
        result[kind + "_cells"] = [{"bin": index, **finish(accumulator[f"{kind}_{index}"])} for index in range(len(bounds) + 1)]
    require(result["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
        "Replay validation target count changed")
    audit_replay_accounting(result)
    return result

