"""Opt-in process-local width16 routing; operational permissions live in the campaign.

No shared Factory or training-engine source is modified. Importing this module
does not install hooks, load research data, or execute a scientific fit.
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

from models.TPPs.CountAwareTitanHistoryWidth import (
    ARM, ARMS, ROLE, CountAwareTitanHistoryWidth, metadata, validate_checkpoint,
)
from models.TPPs.CountAwareTitanCoreAblation import ROLE as BASE_ROLE
from paper.scripts import run_titantpp_core_ablation as core

BASELINE = "titantpp_history_mlp"
ALL_ARMS = (BASELINE, ARM)
VARIANT = "count_only_log_regression"
SCHEMA = "titantpp_history_width_v1"
base, shared, common = core.base, core.shared, core.common
require, read = core.require, core.read
TIME_METRIC, audit_replay_accounting = core.TIME_METRIC, core.audit_replay_accounting
_installed = False


def install_hooks():
    """Register only the explicit new model in this interpreter, idempotently."""
    global _installed
    if _installed:
        return
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build = factory.build_count_aware_model
    signature = inspect.signature(original_build)
    original_route = factory.validate_checkpoint_route
    original_launch = observed_time.validate_launch
    original_role = constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone != ARM:
            return original_build(backbone, **kwargs)
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        if values.pop("titans_memory_gradient_clip") is not None:
            raise ValueError("Width intervention does not change memory gradient clipping")
        model = CountAwareTitanHistoryWidth(**values)
        return factory.with_time_metadata(model, {
            **metadata(values["hidden_dim"]), "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) == ROLE or ARM in backbones:
            if getattr(args, "model_role", None) != ROLE or backbones != (ARM,):
                raise ValueError("Width16 requires its explicit single-arm role")
            old_args = copy(args)
            old_args.model_role = BASE_ROLE
            return original_launch(old_args, (BASELINE,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        if kwargs["model_role"] == ROLE or ARM in kwargs["backbones"]:
            if kwargs["model_role"] != ROLE or kwargs["backbones"] != (ARM,):
                raise ValueError("Width16 requires its explicit single-arm role")
            return original_role(**{**kwargs, "model_role": BASE_ROLE, "backbones": (BASELINE,)})
        return original_role(**kwargs)

    factory.build_count_aware_model = build
    factory.validate_checkpoint_route = route
    training.build_count_aware_model = training.build_model = build
    training.validate_checkpoint_route = route
    observed_time.ROLES = (*observed_time.ROLES, ROLE)
    observed_time.validate_launch = launch
    training.OBSERVED_TIME_ROLES = observed_time.ROLES
    constants.validate_model_role_contract = role_contract
    constants.BACKBONE_LABELS[ARM] = "TitanTPP History MLP width16"
    constants.MODEL_ROLES = (*constants.MODEL_ROLES, ROLE)
    constants.SUPPORTED_BACKBONES = (*constants.SUPPORTED_BACKBONES, ARM)
    _installed = True


def build_model(data, arm):
    install_hooks()
    from models.TPPs.CountAwareFactory import build_count_aware_model
    require(arm in ALL_ARMS, "Width comparison model changed")
    config = {k: v for k, v in data["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(
        arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def initial_states(data, seed):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states, rngs = {}, {}
    with torch.random.fork_rng(devices=[]):
        for arm in ALL_ARMS:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            states[arm] = model.state_dict()
            rngs[arm] = torch.get_rng_state().clone()
        require(torch.equal(rngs[BASELINE], rngs[ARM]), "Width changed common-model RNG")
        for name, value in states[BASELINE].items():
            if name.startswith("multilag_detail."):
                continue
            require(torch.equal(value, states[ARM][name]), "Common initial tensor changed: " + name)
    return {arm: canonical_state_dict_sha256(state) for arm, state in states.items()}


initialization = initial_states


def training_args(c, data, output, arm):
    install_hooks()
    require(arm == ARM, "Only the width16 candidate is a new fit")
    args = base.training_args({**c, "model_role": ROLE}, data, output)
    args.epochs, args.min_epochs, args.early_stopping_patience = 300, 40, 40
    args.execution_role = "fresh_history_width_validation"
    return args


def time_interface(data, frame, c):
    interface = base.time_interface(data, frame, c)
    interface["backbone_design"] = {
        "schema": SCHEMA, "intervention": "history_mlp_bottleneck_width_4_to_16",
        "model_metadata": metadata(), "training": c["training"],
    }
    return interface


def synthetic_check(device="cpu"):
    """Small fabricated-input update/identity check; no research dataset access."""
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    data = {"model": {"hidden_dim": 64, "quantity_variant": VARIANT,
            "time_head_mode": "heteroscedastic_lognormal_duration", "time_scale": 7.,
            "time_initial_location": .2, "time_initial_scale": .8,
            "time_observation_contract": {"mode": "positive_integer_round_clamp_v1",
                                          "unit": "week", "top_code": None}},
            "statistics": {"train_log_mean": 1.2, "train_log_std": .8},
            "loader": {"max_seq_len": 256}}
    initial = initial_states(data, 42)
    counts, corrections, outputs = {}, {}, {}
    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else [torch.device(device)]):
        torch.manual_seed(42)
        dt = torch.randint(1, 8, (2, 12), device=device).float()
        quantity = torch.randint(1, 20, dt.shape, device=device).float()
        mask = torch.ones_like(dt, dtype=torch.bool)
        for arm in ALL_ARMS:
            torch.manual_seed(42)
            model, _ = build_model(data, arm)
            counts[arm] = sum(p.numel() for p in model.parameters())
            corrections[arm] = sum(p.numel() for p in model.multilag_detail.parameters())
            model.to(device).eval()
            with torch.no_grad():
                outputs[arm] = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            model.train()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = target_outputs(model, dt, mask, quantity, lambda_log_qty=1.)["joint_loss"].mean()
                require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
                loss.backward()
                require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
                        "Nonfinite synthetic gradient")
                optimizer.step()
            restored, _ = build_model(data, arm)
            restored.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()}, strict=True)
        for key in outputs[BASELINE]:
            require(torch.equal(outputs[BASELINE][key], outputs[ARM][key]), "Initial outputs differ: " + key)
    require(corrections == {BASELINE: 6144, ARM: 24576}, "Correction parameter count drift")
    require(counts[ARM] - counts[BASELINE] == 18432, "Common parameter count drift")
    return {"status": "passed", "device": str(device), "checks": {
        "common_initial_tensors_and_rng": True, "zero_correction_initial_output_identity": True,
        "finite_synthetic_optimizer_updates": True, "strict_checkpoint_roundtrip": True},
        "initial_state_sha256": initial, "parameter_count": counts,
        "correction_parameter_count": corrections, "synthetic_optimizer_updates": 4,
        "real_data_loaded": False, "held_out_test_evaluated": False}


# Frozen core endpoint logic; only this module's explicit model routing differs.
def replay_checkpoint(path, data, frame, budget, *, device="cuda:0", expected_arm=None, expected_identity=None, expected_initial=None, expected_epoch=None, expected_seed=42):
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
                and payload.get("initial_state_sha256") == expected_initial,
                "Replay source/initialization/interface identity mismatch")
        revision = expected_identity["arguments"]["source_revision"] if "source_revision" in expected_identity.get("arguments", {}) else None
        if revision is not None:
            require(payload.get("source_revision") == revision and payload.get("source_revision_history") == [revision], "Replay source revision mismatch")
        require(payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse"
                and payload.get("checkpoint_monitor_history_key") == selector["history_key"]
                and payload.get("checkpoint_selection") == selector["selection"], "Replay selector mismatch")
        require(payload.get("epoch", payload.get("best_epoch")) == expected_epoch, "Replay epoch mismatch")
    head = payload.get("interface", {}).get("time_head", {})
    # The shared trainer writes its interface under interface_meta.
    if not head:
        head = payload.get("interface_meta", {}).get("time_head", {})
    require(head.get("observation_likelihood") == data["model"]["time_observation_contract"],
            "Replay observation likelihood metadata mismatch")
    result = _replay_validation(payload, path, data, frame, budget, device=device)
    result["time_metric"] = TIME_METRIC
    return result

def _replay_validation(payload, path, data, frame, budget_check, *, device="cuda:0"):
    """One streaming validation pass with fixed train-derived strata; no records."""
    import numpy as np
    import torch
    from models.TPPs.CountAwareFactory import validate_checkpoint_route
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_taxi_quantity_interface_ablation import make_loader
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    require(payload.get("backbone") in ALL_ARMS, "Unexpected endpoint backbone")
    validate_checkpoint_route(payload, payload["backbone"])
    require(payload["evaluation_scope"] == "validation_only"
            and payload["held_out_test_evaluated"] is False, "Replay split changed")
    state_digest = canonical_state_dict_sha256(payload["model_state_dict"])
    require(payload.get("model_state_sha256") == state_digest, "Checkpoint state digest differs")
    model, _ = build_model(data, payload["backbone"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = make_loader(frame, target_split="validation", **{
        key: data["loader"][key] for key in ("batch_size", "lookback_weeks", "max_seq_len")},
        shuffle=False, generator=None)
    q_bounds, h_bounds = data["quantity_boundaries_all_train_rows"], data["history_boundaries"]
    partitions = [("quantity", q_bounds), ("history", h_bounds)]
    if "additional_history_boundaries" in data:
        partitions.append(("additional_history", data["additional_history_boundaries"]))
    accumulator = {"overall": [0, 0., 0., 0.], "body": [0, 0., 0., 0.], "tail": [0, 0., 0., 0.]}
    for kind, bounds in partitions:
        accumulator.update({f"{kind}_{i}": [0, 0., 0., 0.] for i in range(len(bounds) + 1)})
    with torch.no_grad():
        for _, dts, mask, _, quantities in loader:
            budget_check()
            result = target_outputs(model, dts.to(device), mask.to(device), quantities.to(device), lambda_log_qty=1.)
            q = result["true_qty"].cpu().numpy().astype(np.float64)
            pred = result["pred_qty"].cpu().numpy().astype(np.float64)
            t = result["time_loss"].cpu().numpy().astype(np.float64)
            h = result["history_length"].cpu().numpy()
            require(np.isfinite(q).all() and np.isfinite(pred).all() and np.isfinite(t).all(),
                    "Nonfinite checkpoint replay")
            masks = {"overall": np.ones(q.shape, dtype=bool), "body": q <= q_bounds[2], "tail": q > q_bounds[3]}
            for kind, bounds in partitions:
                values = q if kind == "quantity" else h
                ids = np.searchsorted(bounds, values, side="left")
                masks.update({f"{kind}_{i}": ids == i for i in range(len(bounds) + 1)})
            error = pred - q
            for key, selected in masks.items():
                acc = accumulator[key]
                acc[0] += int(selected.sum())
                acc[1] += float(np.abs(error[selected]).sum())
                acc[2] += float(np.square(error[selected]).sum())
                acc[3] += float(t[selected].sum())
    budget_check()
    def finish(acc):
        n, absolute, squared, temporal = acc
        return {"count": n, "qty_mae": absolute / n if n else None, "qty_sse": squared,
            "qty_rmse": math.sqrt(squared / n) if n else None,
            "time_nll": temporal / n if n else None}
    result = {**finish(accumulator["overall"]), "body": finish(accumulator["body"]),
        "tail": finish(accumulator["tail"]), "quantity_boundaries": q_bounds,
        "history_boundaries": h_bounds, "quantity_cells": [], "history_cells": [],
        "checkpoint_path": str(path), "state_sha256": state_digest,
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False}
    for kind, bounds in partitions:
        result[kind + "_boundaries"] = bounds
        result[kind + "_cells"] = [{"bin": i, **finish(accumulator[f"{kind}_{i}"])}
                                    for i in range(len(bounds) + 1)]
    require(result["count"] == data["inherited_data_identity"]["populations"]["validation"]["target_count"],
            "Replay validation target count changed")
    audit_replay_accounting(result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-check", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(synthetic_check(args.device), indent=2))
