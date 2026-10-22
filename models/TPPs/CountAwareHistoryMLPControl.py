"""Parameter-matched single-MLP control for the prototype-free Shared15 arm.

The control sees the same current state, immediate predecessor and eight
eligibility bits as Shared15.  It matches the total 9,600 correction parameters;
it does not match the active-prefix parameter budget or the function class.
Availability bits can produce a length-only correction even at zero hidden input.
The existing encoder, persistent memory, heads, losses and masking stay intact.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs import CountAwareComponentAblation as components
from models.TPPs.CountAwareTitanMultiLagDetail import (
    AVAILABILITY_LAGS, _complete_state, _mask, lag_source_indices,
)
from models.TPPs.CountAwareTitanSharedHistoryMLP import _identity_matches

ARMS = ("shared15", "encoder_only", "single_mlp")
BACKBONE_BY_ARM = {"shared15": "no_prototype", "encoder_only": "encoder_only", "single_mlp": "single_mlp"}
CONTRACT_ID = "shared_history_mlp_control_v1"
CAMPAIGN = "shared_history_mlp_control_20261022_v1"
CORRECTION_PARAMETER_COUNT = 9600
PROJECTION_SHAPES = {
    "multilag_detail.input_projection.weight": (48, 136),
    "multilag_detail.output_projection.weight": (64, 48),
}
_FOREIGN_IDENTITIES = ("component_ablation_identity", "shared_history_identity", "core_ablation_identity")


def metadata(hidden_dim: int = 64, *, arm: str) -> dict[str, Any]:
    if type(hidden_dim) is not int or hidden_dim != 64:
        raise ValueError("History MLP controls fix hidden_dim=64")
    if not isinstance(arm, str) or arm not in ARMS:
        raise ValueError("Unknown history MLP control arm")
    if arm != "single_mlp":
        return components.metadata(hidden_dim, arm=BACKBONE_BY_ARM[arm])
    result = components.metadata(hidden_dim, arm="encoder_only").copy()
    for key in ("component_ablation_arm", "active_prefix_parameter_counts"):
        result.pop(key, None)
    result.update(
        candidate_name="single_mlp", backbone_contract_id=CONTRACT_ID,
        model_role="shared_history_mlp_control_single_mlp", control_arm="single_mlp",
        control_campaign=CAMPAIGN, checkpoint_identity_key="control_identity",
        control_identity_hash_scope="metadata_without_control_identity_sha256",
        prototype_component_enabled=False, static_retrieval_enabled=False,
        history_correction_enabled=True, history_relation_sharing="single",
        history_relation_width=48, history_relation_branch_widths=[48], history_relation_branch_count=1,
        history_relation_input_width=136,
        history_relation_input_features=["current_hidden64", "immediate_previous_hidden64", "availability_bits8"],
        history_relation_input_parameter_count=6528, history_relation_output_parameter_count=3072,
        history_relation_parameter_count=CORRECTION_PARAMETER_COUNT,
        additional_parameter_count=CORRECTION_PARAMETER_COUNT,
        shared_input_projection_count=0, independent_input_projection_count=1, output_projection_count=1,
        parameter_matching="equal_total_correction_parameters_not_active_prefix_or_function_class",
        history_length_only_correction_possible=True,
        history_context="current_and_same_immediate_predecessor_with_eight_availability_bits",
        branch_availability="original_threshold_bits_with_lag1_output_gate",
        multilag_detail_rank=48, multilag_detail_branch_count=1, multilag_detail_lags=[1],
        multilag_detail_availability_lags=list(AVAILABILITY_LAGS),
        multilag_detail_projections="single_bias_free_136_to48_to64",
        multilag_detail_projection_shapes={name: list(shape) for name, shape in PROJECTION_SHAPES.items()},
        multilag_detail_initialization="default_Linear_input_zero_output_forked_cpu_rng",
        multilag_detail_implementation="safe_immediate_previous_concat_current_availability",
        multilag_detail_activation="gelu_exact_erf", multilag_detail_residual_divisor=8,
        residual_divisor=8, initialization="default_Linear_input_zero_output_forked_cpu_rng",
        component_removed_parameters={"prototype": 4096, "history_correction": 0},
    )
    encoded = json.dumps(result, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    result["control_identity_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def identity(hidden_dim: int = 64) -> torch.Tensor:
    value = metadata(hidden_dim, arm="single_mlp")["control_identity_sha256"]
    return torch.tensor(list(bytes.fromhex(value)), dtype=torch.uint8)


class SingleHistoryCorrection(nn.Module):
    """One bias-free 136→48→64 GELU correction; observed lag-one gate only."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        metadata(hidden_dim, arm="single_mlp")
        self.hidden_dim, self.width, self.residual_divisor = hidden_dim, 48, 8
        self.availability_lags = AVAILABILITY_LAGS
        # Only CPU initializers are used; the common model/head RNG is preserved.
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Linear(136, 48, bias=False, device="cpu")
            self.output_projection = nn.Linear(48, 64, bias=False, device="cpu")
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, hidden, valid, *, memory_write_mask=None):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[-1] != self.hidden_dim or not hidden.is_floating_point()):
            raise ValueError("History states must have floating shape [batch,length,64]")
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        valid = valid.to(hidden.device)
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
            memory_write_mask = memory_write_mask.to(hidden.device)
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed history")
        sources, available = lag_source_indices(valid, memory_write_mask=memory_write_mask, mode="local")
        previous = safe.gather(1, sources[..., 0].unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        previous = torch.where(available[..., 0, None], previous, 0.0)
        inputs = torch.cat((safe, previous, available.to(hidden.dtype)), dim=-1)
        features = F.gelu(self.input_projection(inputs), approximate="none")
        correction = self.output_projection(features) / self.residual_divisor
        return torch.where(available[..., 0, None], correction, 0.0)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not isinstance(state_dict, Mapping):
            raise ValueError("History correction state must be a mapping")
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)


class CountAwareHistoryMLPControl(components.CountAwareComponentAblation):
    """Start with the unchanged prototype-free encoder-only construction."""
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, **kwargs):
        metadata(hidden_dim, arm="single_mlp")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, component_arm="encoder_only", **kwargs)
        self.multilag_detail = SingleHistoryCorrection(hidden_dim)
        delattr(self, "component_ablation_identity")
        delattr(self, "component_arm")
        self.control_arm = "single_mlp"
        self.register_buffer("control_identity", identity(hidden_dim))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not isinstance(state_dict, Mapping):
            raise ValueError("History MLP control state must be a mapping")
        if not _identity_matches(state_dict.get("control_identity"), identity(self.core_hidden_dim)):
            raise ValueError("History MLP control checkpoint identity mismatch")
        if any(name in state_dict for name in _FOREIGN_IDENTITIES):
            raise ValueError("History MLP control checkpoint contains a foreign identity")
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)


def validate_checkpoint(payload: Mapping[str, Any], expected_arm: str) -> bool:
    if not isinstance(expected_arm, str) or expected_arm not in ARMS:
        raise ValueError("Unknown history MLP control arm")
    if expected_arm != "single_mlp":
        # Job labels and actual frozen backbones are separate.  Never relabel an
        # existing B/D payload to make the delegated validator accept it.
        return components.validate_checkpoint(payload, BACKBONE_BY_ARM[expected_arm])
    if not isinstance(payload, Mapping) or payload.get("backbone") != "single_mlp":
        raise ValueError("History MLP control checkpoint arm mismatch")
    expected = metadata(64, arm="single_mlp")
    descriptions = [payload[name] for name in ("model_metadata", "encoder_config") if name in payload]
    if (not descriptions or any(not isinstance(value, Mapping)
            or any(value.get(key) != item for key, item in expected.items()) for value in descriptions)):
        raise ValueError("History MLP control checkpoint metadata mismatch")
    if len(descriptions) == 2:
        first, second = descriptions
        if any(first[key] != second[key] for key in set(first) & set(second)):
            raise ValueError("History MLP control metadata aliases disagree")
    states = [payload[name] for name in ("state_dict", "model_state_dict", "best_state_dict") if name in payload]
    if not states:
        raise ValueError("History MLP control checkpoint requires actual state tensors")
    for state in states:
        if (not isinstance(state, Mapping) or not _identity_matches(state.get("control_identity"), identity())
                or any(name in state for name in _FOREIGN_IDENTITIES)):
            raise ValueError("History MLP control state identity mismatch")
        if set(name for name in state if str(name).startswith("multilag_detail.")) != set(PROJECTION_SHAPES):
            raise ValueError("History MLP control projection keys mismatch")
        for name, shape in PROJECTION_SHAPES.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise ValueError("History MLP control projection tensor mismatch: " + name)
        if sum(state[name].numel() for name in PROJECTION_SHAPES) != CORRECTION_PARAMETER_COUNT:
            raise ValueError("History MLP control requires exactly 9600 correction parameters")
        if any(not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()) for value in state.values()):
            raise ValueError("History MLP control state must contain finite tensors")
        if any(str(name).startswith("lmm.") for name in state):
            raise ValueError("History MLP control cannot contain static prototype tensors")
    return True


def build_model(data: Mapping[str, Any], arm: str):
    """Explicit campaign factory; the generic backbone registry stays unchanged."""
    if not isinstance(arm, str) or arm not in ARMS:
        raise ValueError("Unknown history MLP control arm")
    if arm != "single_mlp":
        return components.build_model(data, BACKBONE_BY_ARM[arm])
    from models.TPPs.CountAwareFactory import with_time_metadata
    from paper.scripts import run_titantpp_shared_history_raw9 as raw9
    config = {key: value for key, value in data["model"].items()
        if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    config.update(tail_threshold=46.0, tail_normalization_scale=46.0, tail_clip_cap=187.0, tail_huber_delta=1.0)
    if config.pop("titans_memory_gradient_clip", None) is not None:
        raise ValueError("Memory gradient clipping must remain unchanged")
    model = CountAwareHistoryMLPControl(**config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])
    return with_time_metadata(model, {**metadata(config["hidden_dim"], arm="single_mlp"),
        "max_len": data["loader"]["max_seq_len"], **raw9.input_identity()})
