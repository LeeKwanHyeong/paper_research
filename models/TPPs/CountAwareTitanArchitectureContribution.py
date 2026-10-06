"""Prespecified prototype × history interventions on frozen History-MLP8.

Persistent encoder tokens, both common output heads and their initialization
are retained. P0/H0 remove real parameters; no inactive budget padding is used.
The ordinary FFN controls are useful capacity controls, not identical functions
or identical initial residual scales to hard prototype retrieval.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask, lag_source_indices

ARMS = ("P0_H0", "P1_H0", "P0_H1", "P1_H1", "PF_H1", "P1_HC")
CONTRACT_ID = "titantpp_architecture_contribution_v1"
BACKBONE_BY_ARM = {arm: "titantpp_architecture_" + arm.lower() for arm in ARMS}
BACKBONES = tuple(BACKBONE_BY_ARM.values())
MODULES = {
    "P0_H0": ("none", "none"), "P1_H0": ("prototype", "none"),
    "P0_H1": ("none", "history_mlp8"), "P1_H1": ("prototype", "history_mlp8"),
    "PF_H1": ("ordinary_ffn32", "history_mlp8"),
    "P1_HC": ("prototype", "current_ffn12"),
}


def backbone_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported architecture contribution arm")
    return BACKBONE_BY_ARM[arm]


def arm_for_backbone(backbone):
    for arm, name in BACKBONE_BY_ARM.items():
        if name == backbone:
            return arm
    raise ValueError("Unsupported architecture contribution backbone")


def role_for_arm(arm):
    backbone_for_arm(arm)
    return "architecture_contribution_" + arm.lower() + "_v1"


def metadata(hidden_dim=64, *, arm="P1_H1"):
    backbone = backbone_for_arm(arm)
    if type(hidden_dim) is not int or hidden_dim != 64:
        raise ValueError("Architecture contribution fixes hidden_dim=64")
    prototype, correction = MODULES[arm]
    p_count = 0 if prototype == "none" else 4096
    h_count = 0 if correction == "none" else 12288
    return {
        "candidate_name": backbone, "backbone_contract_id": CONTRACT_ID,
        "model_role": role_for_arm(arm), "architecture_arm": arm,
        "base_encoder": "frozen_HistoryMLP8_common_encoder",
        "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128,
        "persistent_mem_size": 16, "persistent_memory_retained": True,
        "prototype_module": prototype, "correction_module": correction,
        "prototype_parameter_count": p_count, "correction_parameter_count": h_count,
        "intervention_parameter_count": p_count + h_count,
        "prototype_position": "after_block2", "prototype_topk": 4 if prototype == "prototype" else 0,
        "prototype_initialization": "native_normal_std_0.02" if prototype == "prototype" else
            "ordinary_Linear_forked_CPU_RNG_nonzero_output" if prototype == "ordinary_ffn32" else "none",
        "correction_position": "between_block1_block2", "correction_branch_count": 8 if h_count else 0,
        "correction_availability_lags": [1, 2, 4, 8, 16, 32, 64, 128] if h_count else [],
        "correction_residual_divisor": 8 if h_count else 0,
        "correction_context": "current_and_previous_observed_state" if correction == "history_mlp8" else
            "current_state_encoder_history_retained" if correction == "current_ffn12" else "none",
        "correction_initialization": "ordinary_Linear_input_zero_output_forked_CPU_RNG" if h_count else "none",
        "shared_final_state_feeds": ["time_head", "quantity_head"],
        "online_writes": False, "dummy_parameters": False,
        "quantity_objective": "count_only_log_regression",
        "initialization_anchor": "frozen_HistoryMLP8_construct_then_remove_or_replace_optional_modules",
    }


def identity(hidden_dim=64, *, arm="P1_H1"):
    digest = hashlib.sha256(json.dumps(metadata(hidden_dim, arm=arm), sort_keys=True).encode()).digest()
    return torch.tensor(list(digest), dtype=torch.uint8)


class NoCorrection(nn.Module):
    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
        return torch.zeros_like(hidden)


class PrototypeBudgetFFN(nn.Module):
    """Ordinary residual 64→32→64, exactly 4096 participating weights."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Linear(64, 32, bias=False)
            self.output_projection = nn.Linear(32, 64, bias=False)

    def forward(self, hidden):
        return hidden + self.output_projection(F.gelu(self.input_projection(hidden), approximate="none"))


class CurrentBudgetCorrection(nn.Module):
    """Current-only 64→12→64 branches with the native history availability."""
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.input_projections = nn.ModuleList(nn.Linear(64, 12, bias=False) for _ in range(8))
            self.output_projections = nn.ModuleList(nn.Linear(12, 64, bias=False) for _ in range(8))
            for projection in self.output_projections:
                nn.init.zeros_(projection.weight)

    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed current state")
        _, available = lag_source_indices(valid, memory_write_mask=memory_write_mask, mode="local")
        current = F.linear(safe, torch.cat([p.weight for p in self.input_projections])).unflatten(-1, (8, 12))
        features = F.gelu(torch.where(available.unsqueeze(-1), current, 0.), approximate="none")
        return F.linear(features.flatten(-2), torch.cat([p.weight for p in self.output_projections], dim=1)) / 8


class CountAwareTitanArchitectureContribution(CountAwareTitanHistoryWidth):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, architecture_arm="P1_H1", **kwargs):
        metadata(hidden_dim, arm=architecture_arm)
        # All arms consume exactly the native anchor's RNG before intervention.
        # Discarded temporary modules are not registered in the resulting model.
        super().__init__(hidden_dim, train_log_mean, max_seq_len, history_mlp_width=8, **kwargs)
        self.architecture_arm = architecture_arm
        self._buffers.pop("core_ablation_identity")
        self._buffers.pop("history_width_identity")
        prototype, correction = MODULES[architecture_arm]
        if prototype == "none":
            self.lmm = nn.Identity()
        elif prototype == "ordinary_ffn32":
            self.lmm = PrototypeBudgetFFN()
        if correction == "none":
            self.multilag_detail = NoCorrection()
        elif correction == "current_ffn12":
            self.multilag_detail = CurrentBudgetCorrection()
        self.register_buffer("architecture_contribution_identity", identity(hidden_dim, arm=architecture_arm))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get("architecture_contribution_identity")
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(arm=self.architecture_arm)):
            raise ValueError("Architecture contribution checkpoint identity mismatch")
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)


def expected_intervention_shapes(arm):
    prototype, correction = MODULES[arm]
    shapes = {}
    if prototype == "prototype":
        shapes["lmm.mem"] = (1, 64, 64)
    elif prototype == "ordinary_ffn32":
        shapes.update({"lmm.input_projection.weight": (32, 64), "lmm.output_projection.weight": (64, 32)})
    if correction != "none":
        width, inputs = (8, 128) if correction == "history_mlp8" else (12, 64)
        for branch in range(8):
            shapes[f"multilag_detail.input_projections.{branch}.weight"] = (width, inputs)
            shapes[f"multilag_detail.output_projections.{branch}.weight"] = (64, width)
    return shapes


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in BACKBONES or "architecture_arm" in meta or
                  any(isinstance(s, Mapping) and "architecture_contribution_identity" in s for s in states))
    if expected_backbone not in BACKBONES:
        if identified:
            raise ValueError("Architecture contribution checkpoint cannot be relabelled")
        return False
    arm = arm_for_backbone(expected_backbone)
    interface, resume = payload.get("interface_meta", {}), payload.get("resume_identity", {})
    head = meta.get("time_head", {})
    observation = head.get("observation_likelihood", {})
    if (payload.get("backbone") != expected_backbone or
            any(meta.get(k) != v for k, v in metadata(meta.get("d_model"), arm=arm).items()) or
            payload.get("evaluation_scope") != "validation_only" or
            payload.get("held_out_test_evaluated") is not False or
            payload.get("variant") != "count_only_log_regression" or
            resume.get("backbone") != expected_backbone or resume.get("interface_meta") != interface or
            resume.get("checkpoint_monitor") != "validation_raw_quantity_rmse" or
            resume.get("arguments", {}).get("model_role") != role_for_arm(arm) or
            head.get("mode") != "heteroscedastic_lognormal_duration" or
            not isinstance(observation, dict) or observation.get("mode") != "positive_integer_round_clamp_v1" or
            observation != interface.get("time_head", {}).get("observation_likelihood")):
        raise ValueError("Architecture contribution metadata, split, selector or role mismatch")
    shapes = expected_intervention_shapes(arm)
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("Architecture contribution state must be a mapping")
        value = state.get("architecture_contribution_identity")
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or
                not torch.equal(value.cpu(), identity(arm=arm))):
            raise ValueError("Architecture contribution state identity mismatch")
        optional = {k for k in state if k.startswith(("lmm.", "multilag_detail."))}
        if optional != set(shapes) or any(k in state for k in ("core_ablation_identity", "history_width_identity")):
            raise ValueError("Architecture contribution intervention parameter presence mismatch")
        for key, shape in shapes.items():
            value = state[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("Architecture contribution parameter shape/finiteness mismatch: " + key)
    if not states and (len(payload.get("checkpoint_state_sha256", "")) != 64 or payload.get("best_epoch", 0) < 1):
        raise ValueError("Unbound architecture contribution summary")
    return True
