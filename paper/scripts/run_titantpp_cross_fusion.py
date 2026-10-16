"""Opt-in process-local routing for the two time/quantity fusion candidates."""
from __future__ import annotations

import argparse
from copy import copy
import gc
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.TPPs.CountAwareTitanCrossFusion import (
    ARMS, POINTWISE, CROSS, ROLE_BY_ARM, CONTRACT_ID, FUSION_PARAMETERS,
    CountAwareTitanCrossFusion, metadata, role_for_arm, validate_checkpoint,
)
from paper.scripts import run_titantpp_history_width as width

base, shared, common = width.base, width.shared, width.common
require, read = width.require, width.read
VARIANT, TIME_METRIC = width.VARIANT, width.TIME_METRIC
SCHEMA = CONTRACT_ID
_installed = False


def install_hooks():
    global _installed
    if _installed:
        return
    width.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    from paper.scripts.count_aware_tpp_backbone import training, constants, observed_time
    original_build, original_route = factory.build_count_aware_model, factory.validate_checkpoint_route
    # Width's process-local wrapper retains the original factory signature,
    # including scientifically relevant head defaults, in its closure.
    signature = inspect.getclosurevars(original_build).nonlocals["signature"]
    original_launch, original_role = observed_time.validate_launch, constants.validate_model_role_contract

    def build(backbone, **kwargs):
        if backbone not in ARMS:
            return original_build(backbone, **kwargs)
        bound = signature.bind(backbone, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop("backbone")
        if values.pop("titans_memory_gradient_clip") is not None:
            raise ValueError("Fusion does not change memory gradient clipping")
        model = CountAwareTitanCrossFusion(**values, fusion_arm=backbone)
        return factory.with_time_metadata(model, {**metadata(values["hidden_dim"], arm=backbone),
                                                  "max_len": values["max_seq_len"]})

    def route(payload, expected_backbone):
        if validate_checkpoint(payload, expected_backbone):
            return
        return original_route(payload, expected_backbone)

    def launch(args, backbones, variants):
        if getattr(args, "model_role", None) in ROLE_BY_ARM.values() or any(a in ARMS for a in backbones):
            require(len(backbones) == 1 and backbones[0] in ARMS
                    and args.model_role == role_for_arm(backbones[0]), "Fusion requires its explicit single-arm role")
            inherited = copy(args)
            inherited.model_role = width.role_for_arm(width.ARM)
            return original_launch(inherited, (width.ARM,), variants)
        return original_launch(args, backbones, variants)

    def role_contract(**kwargs):
        backbones = kwargs["backbones"]
        if kwargs["model_role"] in ROLE_BY_ARM.values() or any(a in ARMS for a in backbones):
            require(len(backbones) == 1 and backbones[0] in ARMS
                    and kwargs["model_role"] == role_for_arm(backbones[0]), "Fusion requires its explicit single-arm role")
            return original_role(**{**kwargs, "model_role": width.role_for_arm(width.ARM), "backbones": (width.ARM,)})
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
        constants.BACKBONE_LABELS[arm] = "TitanTPP History16 " + ("Cross Fusion" if arm == CROSS else "Pointwise Fusion")
    _installed = True


def build_model(data, arm):
    install_hooks()
    require(arm in ARMS, "Only the two new fusion candidates may be built")
    from models.TPPs.CountAwareFactory import build_count_aware_model
    config = {k: v for k, v in data["model"].items()
              if k not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    return build_count_aware_model(arm, **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])


def training_args(c, data, output, arm):
    install_hooks()
    require(arm in ARMS, "Only the two new fusion candidates may be trained")
    args = base.training_args({**c, "model_role": role_for_arm(arm)}, data, output)
    config = c.get("training", {})
    args.epochs = int(config.get("maximum_epochs", 300))
    args.min_epochs = int(config.get("minimum_epochs", 40))
    args.early_stopping_patience = int(config.get("patience", 40))
    require(1 <= args.min_epochs <= args.epochs <= 300 and args.early_stopping_patience >= 1,
            "Invalid bounded pre-check budget")
    args.execution_role = "fresh_history16_cross_fusion_validation"
    return args


def time_interface(data, frame, c, *, arm):
    require(arm in ARMS, "Unexpected fusion interface arm")
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


def initial_states(data, seed):
    import torch
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256
    states, rngs = {}, {}
    with torch.random.fork_rng(devices=[]):
        for arm in ARMS:
            torch.manual_seed(seed)
            model, _ = build_model(data, arm)
            states[arm] = model.state_dict()
            rngs[arm] = torch.get_rng_state().clone()
        require(torch.equal(rngs[POINTWISE], rngs[CROSS]), "Fusion changed shared RNG")
        for name, value in states[POINTWISE].items():
            if name.startswith("input_fusion.") or name == "cross_fusion_identity":
                continue
            require(torch.equal(value, states[CROSS][name]), "Common initial tensor changed: " + name)
    return {arm: canonical_state_dict_sha256(state) for arm, state in states.items()}


initialization = initial_states


def synthetic_check(device="cpu", *, arms=ARMS, batch_size=2, length=12):
    import torch
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    require(tuple(arms) == ARMS, "Synthetic check covers both new arms only")
    require(type(batch_size) is int and 1 <= batch_size <= 128
            and type(length) is int and 2 <= length <= 256, "Invalid synthetic batch shape")
    cuda = torch.device(device).type == "cuda"
    initial = initial_states(fabricated_data(), 42)
    parameter_counts, outputs, peaks = {}, {}, {}
    with torch.random.fork_rng(devices=[torch.device(device)] if cuda else []):
        generator = torch.Generator().manual_seed(123)
        dt = torch.randint(1, 8, (batch_size, length), generator=generator).float().to(device)
        quantity = torch.randint(1, 20, dt.shape, generator=generator).float().to(device)
        mask = torch.ones_like(dt, dtype=torch.bool)
        for arm in ARMS:
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            torch.manual_seed(42)
            model, _ = build_model(fabricated_data(), arm)
            require(sum(p.numel() for p in model.input_fusion.parameters()) == FUSION_PARAMETERS,
                    "Fusion parameter budget drift")
            parameter_counts[arm] = sum(p.numel() for p in model.parameters())
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
            restored, _ = build_model(fabricated_data(), arm)
            restored.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            if cuda:
                torch.cuda.synchronize(device)
                peaks[arm] = {"max_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
                              "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)}
            del restored, optimizer, model, loss
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
        for key in outputs[POINTWISE]:
            require(torch.equal(outputs[POINTWISE][key], outputs[CROSS][key]), "Initial output mismatch: " + key)
    require(len(set(parameter_counts.values())) == 1, "Candidate capacity differs")
    return {"status": "passed", "device": str(device), "arms": list(ARMS),
        "parameter_count": parameter_counts, "fusion_parameter_count": FUSION_PARAMETERS,
        "batch_size": batch_size, "sequence_length": length, "cuda_memory_peaks": peaks,
        "common_initial_state_and_rng": True, "initial_output_equal": True,
        "strict_checkpoint_roundtrip": True, "synthetic_optimizer_updates": 4,
        "initial_state_sha256": initial, "real_data_loaded": False, "held_out_test_evaluated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-check", action="store_true", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(synthetic_check(args.device), indent=2))
