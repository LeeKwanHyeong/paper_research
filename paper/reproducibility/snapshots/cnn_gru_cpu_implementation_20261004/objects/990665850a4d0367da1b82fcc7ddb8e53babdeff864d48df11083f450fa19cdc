"""Nonlinear observed-successor values with a matched pooling-order control.

The pre-pool candidate transforms each completed successor before retrieval.
The post-pool control transforms the retrieved mean. Both retain the original
Hard-LMM, predecessor addressing, task heads and per-forward memory lifetime.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP, LOG_MSE_VARIANT, TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD, TITAN_QUANTITY_GRADIENT_SHARED,
)
from models.TPPs.CountAwareTitanSuccessorMemory import (
    EPISODE_MEMORY_LN_EPSILON, _is_sha256, _state_shapes as _episode_state_shapes,
    _validate_mask, completed_pairs,
)

PRE_POOL_MEMORY_BACKBONE = "titantpp_nonlinear_episode_pre_pool"
POST_POOL_MEMORY_BACKBONE = "titantpp_nonlinear_episode_post_pool"
PRE_POOL_MEMORY_ROLE = "hard_lmm_nonlinear_episode_pre_pool_candidate"
POST_POOL_MEMORY_ROLE = "hard_lmm_nonlinear_episode_post_pool_control"
NONLINEAR_EPISODE_CONTRACT_ID = "hard_lmm_nonlinear_episode_value_v1"
NONLINEAR_EPISODE_DESIGN_SHA256 = "abd4885ed029613babd66a268845056caed714690f25bdb7e6a0e40191d9759c"
NONLINEAR_EPISODE_RANK = 16
NONLINEAR_EPISODE_VALUE_WIDTH = 16
NONLINEAR_EPISODE_PARAMETER_KEYS = (
    "nonlinear_episode_memory.query_projection.weight",
    "nonlinear_episode_memory.key_projection.weight",
    "nonlinear_episode_memory.value_projection.weight",
    "nonlinear_episode_memory.value_projection.bias",
    "nonlinear_episode_memory.output_projection.weight",
)
NONLINEAR_EPISODE_POOLING_KEY = "nonlinear_episode_memory.pooling_code"
_POOLINGS = {
    "post": (POST_POOL_MEMORY_BACKBONE, POST_POOL_MEMORY_ROLE, 0),
    "pre": (PRE_POOL_MEMORY_BACKBONE, PRE_POOL_MEMORY_ROLE, 1),
}


def _pooling_contract(pooling: str) -> tuple[str, str, int]:
    if not isinstance(pooling, str) or pooling not in _POOLINGS:
        raise ValueError("pooling must be 'pre' or 'post'")
    return _POOLINGS[pooling]


def nonlinear_episode_metadata(hidden_dim: int, pooling: str) -> dict[str, Any]:
    """Structural identity; runners bind source, optimizer, selector and RNG."""
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of four")
    backbone, role, code = _pooling_contract(pooling)
    return {
        "candidate_name": backbone,
        "candidate_status": "experimental_unvalidated",
        "backbone_contract_id": NONLINEAR_EPISODE_CONTRACT_ID,
        "routing_contract_id": NONLINEAR_EPISODE_CONTRACT_ID,
        "design_contract_id": "hard_lmm_nonlinear_episode_value_design_v1",
        "design_contract_sha256": NONLINEAR_EPISODE_DESIGN_SHA256,
        "model_role": role,
        "base_encoder": "CountAwareTitanTPP",
        "d_model": hidden_dim, "n_layers": 2, "n_heads": 4, "d_ff": hidden_dim * 2,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD,
        "persistent_mem_size": 16, "lmm_mem_size": 64, "lmm_topk": 4,
        "static_retrieval_similarity": "cosine",
        "static_retrieval_aggregation": "arithmetic_mean",
        "static_key_value_tied": True,
        "nonlinear_episode_pooling": pooling,
        "nonlinear_episode_pooling_code": code,
        "nonlinear_episode_association": "successor",
        "nonlinear_episode_key_source": "predecessor_hidden",
        "nonlinear_episode_query_source": "observed_query_base_encoder_hidden",
        "nonlinear_episode_feature_source": "existing_two_channel_successor_event_features",
        "nonlinear_episode_feature_transform": "SharedTimeCountModel.continuous_features_unchanged",
        "nonlinear_episode_adjacency": "adjacent_valid_events_skip_padding_no_unobserved_bridge",
        "nonlinear_episode_availability": "both_endpoints_observed_and_successor_at_or_before_observed_query",
        "nonlinear_episode_rank": NONLINEAR_EPISODE_RANK,
        "nonlinear_episode_value_width": NONLINEAR_EPISODE_VALUE_WIDTH,
        "nonlinear_episode_normalization": "affine_free_layer_norm",
        "nonlinear_episode_normalization_epsilon": EPISODE_MEMORY_LN_EPSILON,
        "nonlinear_episode_address_projection_bias": False,
        "nonlinear_episode_value_projection": "linear_2_to_16_bias_true",
        "nonlinear_episode_activation": "gelu_approximate_none",
        "nonlinear_episode_output_projection": "linear_16_to_hidden_bias_false_zero_initialized",
        "nonlinear_episode_empty_row": "exact_zero_read_after_activation_for_both_routes",
        "nonlinear_episode_invalid_record": "sanitize_before_nonlinear_operations_and_exclude_from_reduction",
        "nonlinear_episode_placement": "after_unchanged_encoder_and_final_static_memory_before_both_heads",
        "nonlinear_episode_fusion": "z_B+output_projection(read_context)",
        "nonlinear_episode_state_lifetime": "per_sample_rebuilt_each_forward_no_external_carry",
        "nonlinear_episode_new_dropout": False,
        "nonlinear_episode_compute_dtype": "float64_for_float64_otherwise_float32",
        "nonlinear_episode_initialization": "isolated_cpu_rng_post_B_Q_K_V_U_default_linear_weights_V_bias_zero_U_zero",
        "quantity_variant": LOG_MSE_VARIANT,
        "quantity_memory_gradient_mode": TITAN_QUANTITY_GRADIENT_SHARED,
        "lambda_tail": 0.0,
        "new_parameter_state_keys": list(NONLINEAR_EPISODE_PARAMETER_KEYS),
        "pooling_state_key": NONLINEAR_EPISODE_POOLING_KEY,
        "additional_parameter_count": 3 * hidden_dim * 16 + 2 * 16 + 16,
        "online_writes": True,
    }


def _state_shapes(dim: int, max_len: int) -> dict[str, tuple[int, ...]]:
    # Only the unchanged B keys are reused. No legacy episode state is accepted.
    shapes = {key: shape for key, shape in _episode_state_shapes(dim, max_len).items()
              if not key.startswith("episode_memory.")}
    shapes.update({
        NONLINEAR_EPISODE_PARAMETER_KEYS[0]: (16, dim),
        NONLINEAR_EPISODE_PARAMETER_KEYS[1]: (16, dim),
        NONLINEAR_EPISODE_PARAMETER_KEYS[2]: (16, 2),
        NONLINEAR_EPISODE_PARAMETER_KEYS[3]: (16,),
        NONLINEAR_EPISODE_PARAMETER_KEYS[4]: (dim, 16),
        NONLINEAR_EPISODE_POOLING_KEY: (),
    })
    return shapes


def validate_nonlinear_episode_checkpoint(payload: dict[str, Any], expected_backbone: str) -> bool:
    """Recognize nonlinear state before other validators, including relabeling."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    metadata = payload.get("encoder_config", {})
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    backbones = tuple(value[0] for value in _POOLINGS.values())
    roles = tuple(value[1] for value in _POOLINGS.values())
    recognized = (
        payload.get("backbone") in backbones
        or isinstance(metadata, Mapping) and (
            metadata.get("backbone_contract_id") == NONLINEAR_EPISODE_CONTRACT_ID
            or metadata.get("routing_contract_id") == NONLINEAR_EPISODE_CONTRACT_ID
            or metadata.get("design_contract_sha256") == NONLINEAR_EPISODE_DESIGN_SHA256
            or metadata.get("model_role") in roles
            or metadata.get("candidate_name") in backbones
            or any(isinstance(key, str) and key.startswith("nonlinear_episode_") for key in metadata)
        )
        or any(isinstance(state, Mapping) and any(
            isinstance(key, str) and key.startswith("nonlinear_episode_memory.") for key in state
        ) for state in states)
    )
    if expected_backbone not in backbones:
        if recognized:
            raise ValueError("Nonlinear episode checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, Mapping):
        raise ValueError("Nonlinear episode checkpoint metadata must be a mapping")
    pooling = next(mode for mode, values in _POOLINGS.items() if values[0] == expected_backbone)
    dim, max_len = metadata.get("d_model"), metadata.get("max_len")
    if type(dim) is not int or dim < 1 or dim % 4 or type(max_len) is not int or max_len < 1:
        raise ValueError("Nonlinear episode checkpoint dimensions are invalid")
    if payload.get("backbone") != expected_backbone or any(
        metadata.get(name) != value for name, value in nonlinear_episode_metadata(dim, pooling).items()
    ):
        raise ValueError("Nonlinear episode checkpoint pooling/routing metadata mismatch")
    if any(isinstance(key, str) and key.startswith("episode_") for key in metadata):
        raise ValueError("Nonlinear episode checkpoint contains foreign legacy episode metadata")
    time_head = metadata.get("time_head", {})
    if (payload.get("variant") != LOG_MSE_VARIANT
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or not isinstance(time_head, Mapping)
            or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
            or time_head.get("jacobian_correction") is not False
            or time_head.get("wd_clamp") != 10.0
            or any(type(time_head.get(name)) not in (int, float)
                   or not math.isfinite(time_head[name]) or time_head[name] <= 0
                   for name in ("time_scale", "time_w_max", "time_intercept_limit"))):
        raise ValueError("Nonlinear episode checkpoint scope/head/objective mismatch")
    for config in (payload, payload.get("training_config", {})):
        if not isinstance(config, Mapping) or any(
            name in config and config[name] != expected for name, expected in (
                ("quantity_variant", LOG_MSE_VARIANT), ("lambda_tail", 0.0),
                ("time_head_mode", TIME_HEAD_MODE_LEGACY_CLAMPED),
                ("quantile_adaptive_strength", 0.0),
                ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
            )
        ):
            raise ValueError("Nonlinear episode checkpoint training objective mismatch")
    if not states:
        if (_is_sha256(payload.get("checkpoint_state_sha256"))
                and type(payload.get("best_epoch")) is int and payload["best_epoch"] >= 0
                and ("checkpoint_file_sha256" not in payload or _is_sha256(payload["checkpoint_file_sha256"]))):
            return True
        raise ValueError("Nonlinear episode summary requires selected epoch and canonical state digest")
    shapes = _state_shapes(dim, max_len)
    code = _POOLINGS[pooling][2]
    for state in states:
        if not isinstance(state, Mapping) or set(state) != set(shapes):
            raise ValueError("Nonlinear episode checkpoint state keys are missing or foreign")
        for name, shape in shapes.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError("Nonlinear episode checkpoint state shape is invalid")
            if name == NONLINEAR_EPISODE_POOLING_KEY:
                if tensor.dtype != torch.int64 or tensor.item() != code:
                    raise ValueError("Nonlinear episode checkpoint pooling state mismatch")
            elif not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonlinear episode checkpoint requires finite floating state")
    return True


class NonlinearEpisodeMemory(nn.Module):
    """Retrieve completed successors with fixed pre- or post-pool GELU values."""

    def __init__(self, hidden_dim: int, *, pooling: str = "pre") -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        _, _, code = _pooling_contract(pooling)
        self.hidden_dim = hidden_dim
        self.rank = NONLINEAR_EPISODE_RANK
        self.value_width = NONLINEAR_EPISODE_VALUE_WIDTH
        self._pooling = pooling
        # Preserve the RNG consumed by B, including its subsequent dropout.
        # Both arms use this identical constructor order and initializer set.
        with torch.random.fork_rng(devices=[]):
            self.query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.key_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.value_projection = nn.Linear(2, self.value_width, bias=True)
            self.output_projection = nn.Linear(self.value_width, hidden_dim, bias=False)
            nn.init.zeros_(self.value_projection.bias)
            nn.init.zeros_(self.output_projection.weight)
        self.register_buffer("pooling_code", torch.tensor(code, dtype=torch.int64))

    @property
    def pooling(self) -> str:
        return self._pooling

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        identity = state_dict.get(prefix + "pooling_code")
        if (not isinstance(identity, torch.Tensor) or identity.dtype != torch.int64
                or identity.shape != torch.Size([])
                or identity.item() != _POOLINGS[self.pooling][2]):
            error_msgs.append("Nonlinear episode pooling identity is missing or incompatible")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _value_map(self, values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return F.gelu(F.linear(values, self.value_projection.weight.to(dtype),
                               self.value_projection.bias.to(dtype)), approximate="none")

    def read(self, hidden: torch.Tensor, features: torch.Tensor, mask: torch.Tensor, *,
             memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[-1] != self.hidden_dim or not hidden.is_floating_point()):
            raise ValueError("hidden must have floating shape [batch, length, hidden_dim]")
        shape = tuple(hidden.shape[:2])
        if (not isinstance(features, torch.Tensor) or tuple(features.shape) != (*shape, 2)
                or not features.is_floating_point() or features.device != hidden.device):
            raise ValueError("features must have floating shape [batch, length, 2] on the hidden device")
        _validate_mask(mask, shape, "mask")
        predecessors, pairs, observed = completed_pairs(
            mask.to(hidden.device), memory_write_mask=memory_write_mask)
        safe_hidden = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        safe_features = torch.where(observed.unsqueeze(-1), features, 0.0)
        if not bool(torch.isfinite(safe_hidden).all()) or not bool(torch.isfinite(safe_features).all()):
            raise ValueError("Observed nonlinear episode states and features must be finite")
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            normalized = F.layer_norm(safe_hidden.to(dtype), (self.hidden_dim,), eps=EPISODE_MEMORY_LN_EPSILON)
            queries = F.linear(normalized, self.query_projection.weight.to(dtype))
            keys = F.linear(normalized, self.key_projection.weight.to(dtype))
            keys = keys.gather(1, predecessors.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.rank))
            keys = torch.where(pairs.unsqueeze(-1), keys, 0.0)
            scores = (queries @ keys.transpose(1, 2)) / math.sqrt(self.rank)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("Nonlinear episode addressing scores must be finite")
            positions = torch.arange(shape[1], device=hidden.device)
            available = (observed.unsqueeze(-1) & pairs.unsqueeze(1)
                         & (positions.unsqueeze(0) <= positions.unsqueeze(1)).unsqueeze(0))
            scores = scores.masked_fill(~available, -torch.inf)
            nonempty = available.any(dim=-1)
            weights = torch.zeros_like(scores)
            weights[nonempty] = torch.softmax(scores[nonempty], dim=-1)
            # Unobserved values are sanitized before phi, and non-record values
            # are explicitly excluded. Zero weights must never mask NaNs.
            values = torch.where(pairs.unsqueeze(-1), safe_features.to(dtype), 0.0)
            if self.pooling == "pre":
                transformed = self._value_map(values, dtype)
                transformed = torch.where(pairs.unsqueeze(-1), transformed, 0.0)
                context = weights @ transformed
            else:
                context = self._value_map(weights @ values, dtype)
            # The control's trainable bias makes phi(0) nonzero in general.
            context = torch.where(nonempty.unsqueeze(-1), context, 0.0)
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Nonlinear episode read must be finite")
        return context.to(hidden.dtype)

    def forward(self, hidden: torch.Tensor, features: torch.Tensor, mask: torch.Tensor, *,
                memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        context = self.read(hidden, features, mask, memory_write_mask=memory_write_mask)
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            correction = F.linear(context.to(dtype), self.output_projection.weight.to(dtype)).to(hidden.dtype)
        if not bool(torch.isfinite(correction).all()):
            raise ValueError("Nonlinear episode correction must be finite")
        return correction


class _CountAwareTitanNonlinearEpisodeTPP(CountAwareTitanTPP):
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, *,
                 pooling: str, **quantity_kwargs: Any) -> None:
        nonlinear_episode_metadata(hidden_dim, pooling)
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        for name, expected in (
            ("memory_mode", TITAN_MEMORY_MODE_STATIC_HARD),
            ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
            ("quantity_variant", LOG_MSE_VARIANT), ("time_head_mode", TIME_HEAD_MODE_LEGACY_CLAMPED),
            ("lambda_tail", 0.0), ("quantile_adaptive_strength", 0.0),
            ("quantile_adaptive_boundaries", None), ("quantile_adaptive_weights", None),
        ):
            if quantity_kwargs.get(name, expected) != expected:
                raise ValueError(f"Nonlinear episode route requires {name}={expected!r}")
        quantity_kwargs.pop("memory_mode", None)
        super().__init__(hidden_dim=hidden_dim, train_log_mean=train_log_mean,
                         max_seq_len=max_seq_len, memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
                         **quantity_kwargs)
        self.nonlinear_episode_memory = NonlinearEpisodeMemory(hidden_dim, pooling=pooling)

    def encode_task_states(self, dts: torch.Tensor, history_quantities: torch.Tensor,
                           mask: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch, length]")
        _validate_mask(mask, tuple(dts.shape), "mask")
        if memory_write_mask is not None:
            _validate_mask(memory_write_mask, tuple(dts.shape), "memory_write_mask")
        hidden = super()._encode_base(dts, history_quantities, mask,
                                      memory_write_mask=memory_write_mask)
        if self.lmm is None:
            raise RuntimeError("Nonlinear episode route requires the original static Hard-LMM")
        valid = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
        baseline = self.lmm(hidden) * valid
        features = self.continuous_features(dts, history_quantities, mask)
        correction = self.nonlinear_episode_memory(
            hidden, features, mask, memory_write_mask=memory_write_mask)
        encoded = (baseline + correction) * valid
        return encoded, encoded


class CountAwareTitanNonlinearPrePoolTPP(_CountAwareTitanNonlinearEpisodeTPP):
    """Candidate: transform each observed successor before weighted pooling."""
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
                 **quantity_kwargs: Any) -> None:
        if "pooling" in quantity_kwargs or "association" in quantity_kwargs:
            raise ValueError("Pre-pool backbone fixes pooling and successor association")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, pooling="pre", **quantity_kwargs)


class CountAwareTitanNonlinearPostPoolTPP(_CountAwareTitanNonlinearEpisodeTPP):
    """Capacity control: apply the same transform after weighted pooling."""
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
                 **quantity_kwargs: Any) -> None:
        if "pooling" in quantity_kwargs or "association" in quantity_kwargs:
            raise ValueError("Post-pool backbone fixes pooling and successor association")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, pooling="post", **quantity_kwargs)


__all__ = [
    "NonlinearEpisodeMemory", "CountAwareTitanNonlinearPrePoolTPP", "CountAwareTitanNonlinearPostPoolTPP",
    "PRE_POOL_MEMORY_BACKBONE", "POST_POOL_MEMORY_BACKBONE", "PRE_POOL_MEMORY_ROLE", "POST_POOL_MEMORY_ROLE",
    "NONLINEAR_EPISODE_CONTRACT_ID", "NONLINEAR_EPISODE_DESIGN_SHA256", "NONLINEAR_EPISODE_RANK",
    "NONLINEAR_EPISODE_VALUE_WIDTH", "NONLINEAR_EPISODE_PARAMETER_KEYS", "NONLINEAR_EPISODE_POOLING_KEY",
    "nonlinear_episode_metadata", "validate_nonlinear_episode_checkpoint",
]
