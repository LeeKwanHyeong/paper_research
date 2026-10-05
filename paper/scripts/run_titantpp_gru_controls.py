"""Process-local CNN/GRU routing and synthetic checks; no automatic research fit.

Explicit campaign authorization and a sealed source/data/runtime contract are
still required before any real-data training. Importing this file installs no
hooks and does not load a dataset or checkpoint.
"""
from __future__ import annotations

import argparse
from copy import copy
import inspect
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareTitanGRUControls import (
    ARMS, ANCHOR, PAIR, MLP1, MLP8, ROLE_BY_ARM, CountAwareTitanGRUControl,
    metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_cnn_gru as original
BASELINE = original.BASELINE
CONTROL_ARMS = (ANCHOR, *ARMS)
GRU_ARMS = (ANCHOR, PAIR)
CNN_ARMS = ()
from paper.scripts import run_titantpp_history_width as width

ALL_ARMS = SUPPORTED_ARMS = (BASELINE, *CONTROL_ARMS)
VARIANT = width.VARIANT
SCHEMA = "titantpp_gru_matched_controls_v1"
base, shared, common = width.base, width.shared, width.common
require, read = width.require, width.read
TIME_METRIC, audit_replay_accounting = width.TIME_METRIC, width.audit_replay_accounting
_installed = False


def install_hooks():
    """Install explicit candidate routes above existing width hooks once."""
    global _installed
    if _installed:
        return
    original.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build = factory.build_count_aware_model
    # The width wrapper retains the exact public constructor signature in its
    # closure. Reuse it rather than maintaining a second list of defaults.
    signature = inspect.getclosurevars(original_build).nonlocals.get("signature")
    if not isinstance(signature, inspect.Signature) or "hidden_dim" not in signature.parameters:
        raise RuntimeError("Expected the explicit width wrapper with its bound public factory signature")
    original_route = factory.validate_checkpoint_route
    original_launch = observed_time.validate_launch
    original_role = constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone not in ARMS:
            return original_build(backbone, **kwargs)
        if kwargs.get("titans_memory_gradient_clip") is not None:
            raise ValueError("CNN/GRU does not change memory gradient clipping")
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        values.pop("titans_memory_gradient_clip")
        model = CountAwareTitanGRUControl(**values, candidate_arm=backbone)
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
                raise ValueError("CNN/GRU requires matching explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = width.role_for_arm(BASELINE)
            return original_launch(old_args, (BASELINE,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        backbones = kwargs["backbones"]
        if kwargs["model_role"] in ROLE_BY_ARM.values() or any(arm in ARMS for arm in backbones):
            if (len(backbones) != 1 or backbones[0] not in ARMS
                    or kwargs["model_role"] != role_for_arm(backbones[0])):
                raise ValueError("CNN/GRU requires matching explicit single-arm role")
            return original_role(**{**kwargs, "model_role": width.role_for_arm(BASELINE), "backbones": (BASELINE,)})
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
        constants.BACKBONE_LABELS[arm] = "TitanTPP " + arm.removeprefix("titantpp_")
    _installed = True


def build_model(data, arm):
    install_hooks()
    from models.TPPs.CountAwareFactory import build_count_aware_model
    require(arm in SUPPORTED_ARMS, "CNN/GRU comparison model changed")
    config = {key: value for key, value in data["model"].items()
              if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def _comparison_arms(arms):
    arms = tuple(arms)
    require(BASELINE in arms and len(arms) >= 2 and len(set(arms)) == len(arms)
        and all(arm in SUPPORTED_ARMS for arm in arms), "Invalid CNN/GRU comparison arms")
    return arms


def initial_states(data, seed, *, arms=ALL_ARMS):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    arms = _comparison_arms(arms)
    states, rngs = {}, {}
    with torch.random.fork_rng(devices=[]):
        for arm in arms:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            states[arm] = model.state_dict()
            rngs[arm] = torch.get_rng_state().clone()
        for arm in arms:
            require(torch.equal(rngs[BASELINE], rngs[arm]), "CNN/GRU changed common model RNG")
            for name, value in states[BASELINE].items():
                if name.startswith("multilag_detail.") or name.endswith("identity"):
                    continue
                require(torch.equal(value, states[arm][name]), "Common initial tensor changed: " + name)
    return {arm: canonical_state_dict_sha256(state) for arm, state in states.items()}


initialization = initial_states


def training_args(c, data, output, arm):
    install_hooks()
    if arm == ANCHOR: return original.training_args(c, data, output, arm)
    require(arm in ARMS, "Only explicit GRU controls are new fits")
    args = base.training_args({**c, "model_role": role_for_arm(arm)}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_cnn_gru_validation"
    return args


def time_interface(data, frame, c, *, arm):
    if arm == ANCHOR: return original.time_interface(data, frame, c, arm=arm)
    interface = base.time_interface(data, frame, c)
    interface["backbone_design"] = {"schema": SCHEMA, "intervention": arm,
        "model_metadata": metadata(arm=arm), "training": c["training"]}
    return interface


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
