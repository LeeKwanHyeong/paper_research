"""Explicit completed-event association memory on the unchanged Hard-LMM state.

The candidate matches predecessor contexts and reads their observed successor
features. Its same-event control changes only which endpoint supplies the key.
Neither route carries memory across calls or changes the existing output heads.
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

SUCCESSOR_MEMORY_BACKBONE = "titantpp_successor_episode_memory"
SAME_EVENT_MEMORY_BACKBONE = "titantpp_same_event_episode_memory"
SUCCESSOR_MEMORY_ROLE = "hard_lmm_successor_episode_memory_candidate"
SAME_EVENT_MEMORY_ROLE = "hard_lmm_same_event_episode_memory_control"
EPISODE_MEMORY_CONTRACT_ID = "hard_lmm_successor_episode_memory_v1"
EPISODE_MEMORY_DESIGN_SHA256 = "bb4c0fc68cc06c34108a09d217401c2a517530cf24072e1bc8857a6033b52c65"
EPISODE_MEMORY_RANK = 16
EPISODE_MEMORY_LN_EPSILON = 1e-5
EPISODE_MEMORY_PARAMETER_KEYS = (
    "episode_memory.query_projection.weight",
    "episode_memory.key_projection.weight",
    "episode_memory.output_projection.weight",
)
EPISODE_MEMORY_ASSOCIATION_KEY = "episode_memory.association_code"
_ASSOCIATIONS = {
    "same_event": (SAME_EVENT_MEMORY_BACKBONE, SAME_EVENT_MEMORY_ROLE, 0),
    "successor": (SUCCESSOR_MEMORY_BACKBONE, SUCCESSOR_MEMORY_ROLE, 1),
}


def _association_contract(association: str) -> tuple[str, str, int]:
    if not isinstance(association, str) or association not in _ASSOCIATIONS:
        raise ValueError("association must be 'same_event' or 'successor'")
    return _ASSOCIATIONS[association]


def episode_memory_metadata(hidden_dim: int, association: str) -> dict[str, Any]:
    """Immutable structural identity; source/runtime bindings are added by runners."""
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of four")
    backbone, role, code = _association_contract(association)
    return {
        "candidate_name": backbone,
        "candidate_status": "experimental_unvalidated",
        "backbone_contract_id": EPISODE_MEMORY_CONTRACT_ID,
        "routing_contract_id": EPISODE_MEMORY_CONTRACT_ID,
        "design_contract_id": "hard_lmm_successor_episode_memory_design_v1",
        "design_contract_sha256": EPISODE_MEMORY_DESIGN_SHA256,
        "model_role": role,
        "base_encoder": "CountAwareTitanTPP",
        "d_model": hidden_dim, "n_layers": 2, "n_heads": 4,
        "d_ff": hidden_dim * 2,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD,
        "persistent_mem_size": 16, "lmm_mem_size": 64, "lmm_topk": 4,
        "static_retrieval_similarity": "cosine",
        "static_retrieval_aggregation": "arithmetic_mean",
        "static_key_value_tied": True,
        "episode_association": association,
        "episode_association_code": code,
        "episode_key_source": "predecessor_hidden" if association == "successor" else "successor_hidden",
        "episode_query_source": "observed_query_base_encoder_hidden",
        "episode_value_source": "existing_two_channel_successor_event_features",
        "episode_feature_transform": "SharedTimeCountModel.continuous_features_unchanged",
        "episode_adjacency": "adjacent_valid_events_skip_padding_no_unobserved_bridge",
        "episode_availability": "both_endpoints_observed_and_successor_at_or_before_observed_query",
        "episode_rank": EPISODE_MEMORY_RANK,
        "episode_normalization": "affine_free_layer_norm",
        "episode_normalization_epsilon": EPISODE_MEMORY_LN_EPSILON,
        "episode_projection_bias": False,
        "episode_value_projection": "none",
        "episode_output_projection": "direct_2_to_hidden_zero_initialized",
        "episode_empty_row": "zero_read_softmax_only_nonempty_rows",
        "episode_placement": "after_unchanged_encoder_and_final_static_memory_before_both_heads",
        "episode_fusion": "z_B+output_projection(read_context)",
        "episode_state_lifetime": "per_sample_rebuilt_each_forward_no_external_carry",
        "episode_new_dropout": False,
        "episode_compute_dtype": "float64_for_float64_otherwise_float32",
        "quantity_variant": LOG_MSE_VARIANT,
        "quantity_memory_gradient_mode": TITAN_QUANTITY_GRADIENT_SHARED,
        "lambda_tail": 0.0,
        "new_parameter_state_keys": list(EPISODE_MEMORY_PARAMETER_KEYS),
        "association_state_key": EPISODE_MEMORY_ASSOCIATION_KEY,
        "additional_parameter_count": 2 * hidden_dim * EPISODE_MEMORY_RANK + 2 * hidden_dim,
        "online_writes": True,
    }


def _state_shapes(dim: int, max_len: int) -> dict[str, tuple[int, ...]]:
    shapes = {
        "b_t": (1,), "w_raw": (1,), "v_t.weight": (1, dim),
        "quantity_head.weight": (1, dim), "quantity_head.bias": (1,),
        "encoder.pos_emb": (1, max_len, dim),
        "encoder.input_proj.weight": (dim, 2), "encoder.input_proj.bias": (dim,),
        "lmm.mem": (1, 64, dim),
        "episode_memory.query_projection.weight": (EPISODE_MEMORY_RANK, dim),
        "episode_memory.key_projection.weight": (EPISODE_MEMORY_RANK, dim),
        "episode_memory.output_projection.weight": (dim, 2),
        EPISODE_MEMORY_ASSOCIATION_KEY: (),
    }
    for index in range(2):
        prefix = f"encoder.layers.{index}."
        shapes.update({
            prefix + "norm1.weight": (dim,), prefix + "norm1.bias": (dim,),
            prefix + "attn.persistent_mem": (1, 16, dim),
            prefix + "attn.qkv.weight": (3 * dim, dim), prefix + "attn.qkv.bias": (3 * dim,),
            prefix + "attn.out_proj.weight": (dim, dim), prefix + "attn.out_proj.bias": (dim,),
            prefix + "norm2.weight": (dim,), prefix + "norm2.bias": (dim,),
            prefix + "ff.0.weight": (2 * dim, dim), prefix + "ff.0.bias": (2 * dim,),
            prefix + "ff.3.weight": (dim, 2 * dim), prefix + "ff.3.bias": (dim,),
        })
    return shapes


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_episode_memory_checkpoint(payload: dict[str, Any], expected_backbone: str) -> bool:
    """Recognize either route before legacy validators and reject association relabeling."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    metadata = payload.get("encoder_config", {})
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    backbones = tuple(value[0] for value in _ASSOCIATIONS.values())
    roles = tuple(value[1] for value in _ASSOCIATIONS.values())
    recognized = (
        payload.get("backbone") in backbones
        or isinstance(metadata, Mapping) and (
            metadata.get("backbone_contract_id") == EPISODE_MEMORY_CONTRACT_ID
            or metadata.get("routing_contract_id") == EPISODE_MEMORY_CONTRACT_ID
            or metadata.get("model_role") in roles
            or metadata.get("candidate_name") in backbones
            or "episode_association" in metadata
        )
        or any(isinstance(state, Mapping) and any(
            isinstance(key, str) and key.startswith("episode_memory.") for key in state
        ) for state in states)
    )
    if expected_backbone not in backbones:
        if recognized:
            raise ValueError("Episode-memory checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, Mapping):
        raise ValueError("Episode-memory checkpoint metadata must be a mapping")
    association = next(mode for mode, values in _ASSOCIATIONS.items() if values[0] == expected_backbone)
    dim, max_len = metadata.get("d_model"), metadata.get("max_len")
    if type(dim) is not int or dim < 1 or dim % 4 or type(max_len) is not int or max_len < 1:
        raise ValueError("Episode-memory checkpoint dimensions are invalid")
    if payload.get("backbone") != expected_backbone or any(
        metadata.get(name) != value for name, value in episode_memory_metadata(dim, association).items()
    ):
        raise ValueError("Episode-memory checkpoint association/routing metadata mismatch")
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
        raise ValueError("Episode-memory checkpoint scope/head/objective mismatch")
    for config in (payload, payload.get("training_config", {})):
        if not isinstance(config, Mapping) or any(
            name in config and config[name] != expected for name, expected in (
                ("quantity_variant", LOG_MSE_VARIANT), ("lambda_tail", 0.0),
                ("time_head_mode", TIME_HEAD_MODE_LEGACY_CLAMPED),
                ("quantile_adaptive_strength", 0.0),
                ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
            )
        ):
            raise ValueError("Episode-memory checkpoint training objective mismatch")
    if not states:
        if (_is_sha256(payload.get("checkpoint_state_sha256"))
                and type(payload.get("best_epoch")) is int and payload["best_epoch"] >= 0
                and ("checkpoint_file_sha256" not in payload or _is_sha256(payload["checkpoint_file_sha256"]))):
            return True
        raise ValueError("Episode-memory summary requires selected epoch and canonical state digest")
    shapes = _state_shapes(dim, max_len)
    code = _ASSOCIATIONS[association][2]
    for state in states:
        if not isinstance(state, Mapping) or set(state) != set(shapes):
            raise ValueError("Episode-memory checkpoint state keys are missing or foreign")
        for name, shape in shapes.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError("Episode-memory checkpoint state shape is invalid")
            if name == EPISODE_MEMORY_ASSOCIATION_KEY:
                if tensor.dtype != torch.int64 or tensor.item() != code:
                    raise ValueError("Episode-memory checkpoint association state mismatch")
            elif not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Episode-memory checkpoint requires finite floating state")
    return True


def _validate_mask(mask: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must be a boolean tensor with shape {shape}")


def completed_pairs(
    mask: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predecessor positions, completed-pair validity, and observed mask.

    Each record is indexed by its successor's original position. Adjacency uses
    valid events, not just writable events: padding is skipped, but a valid
    nonwritable event breaks the observed pair on either side.
    """
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        raise ValueError("mask must have shape [batch, length]")
    _validate_mask(mask, tuple(mask.shape), "mask")
    if memory_write_mask is None:
        observed = mask
    else:
        _validate_mask(memory_write_mask, tuple(mask.shape), "memory_write_mask")
        observed = mask & memory_write_mask.to(mask.device)
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    latest_valid = torch.where(mask, positions, -1).cummax(dim=1).values
    predecessors = F.pad(latest_valid[:, :-1], (1, 0), value=-1)[:, :mask.shape[1]]
    previous_observed = observed.gather(1, predecessors.clamp_min(0))
    pairs = observed & (predecessors >= 0) & previous_observed
    return predecessors, pairs, observed


class EpisodeMemory(nn.Module):
    """Read individual completed pairs through one fixed association mode.

    For adjacent valid events (j,k), both endpoints must be observed. Query t
    may use the record only when t is observed and k<=t. q_t=W_Q LN(h_t), with
    k_(j,k)=W_K LN(h_j) for successor association or W_K LN(h_k) for the control.
    Softmax(q dot k / sqrt(16)) reads the identical raw two-channel x_k values
    in both arms. The forward correction is U times that read, with U initially
    zero. All association state is rebuilt from this call's tensors.
    """
    def __init__(self, hidden_dim: int, *, association: str = "successor") -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        _, _, code = _association_contract(association)
        self.hidden_dim, self.rank, self._association = hidden_dim, EPISODE_MEMORY_RANK, association
        self.query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.key_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.output_projection = nn.Linear(2, hidden_dim, bias=False)
        nn.init.zeros_(self.output_projection.weight)
        # Identity only, not a trainable parameter or dynamic memory state.
        self.register_buffer("association_code", torch.tensor(code, dtype=torch.int64))

    @property
    def association(self) -> str:
        return self._association

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        identity = state_dict.get(prefix + "association_code")
        if (not isinstance(identity, torch.Tensor) or identity.dtype != torch.int64
                or identity.shape != torch.Size([])
                or identity.item() != _ASSOCIATIONS[self.association][2]):
            error_msgs.append("Episode-memory association identity is missing or incompatible")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

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
            raise ValueError("Observed episode states and features must be finite")
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            normalized = F.layer_norm(safe_hidden.to(dtype), (self.hidden_dim,), eps=EPISODE_MEMORY_LN_EPSILON)
            queries = F.linear(normalized, self.query_projection.weight.to(dtype))
            keys = F.linear(normalized, self.key_projection.weight.to(dtype))
            if self.association == "successor":
                keys = keys.gather(1, predecessors.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.rank))
            keys = torch.where(pairs.unsqueeze(-1), keys, 0.0)
            scores = (queries @ keys.transpose(1, 2)) / math.sqrt(self.rank)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError("Episode addressing scores must be finite")
            positions = torch.arange(shape[1], device=hidden.device)
            available = (observed.unsqueeze(-1) & pairs.unsqueeze(1)
                         & (positions.unsqueeze(0) <= positions.unsqueeze(1)).unsqueeze(0))
            scores = scores.masked_fill(~available, -torch.inf)
            nonempty = available.any(dim=-1)
            weights = torch.zeros_like(scores)
            # Only populated rows enter softmax. In particular, an empty row
            # containing -inf is never evaluated and remains an exact zero read.
            weights[nonempty] = torch.softmax(scores[nonempty], dim=-1)
            context = weights @ safe_features.to(dtype)
        if not bool(torch.isfinite(context).all()):
            raise ValueError("Episode read must be finite")
        return context.to(hidden.dtype)

    def forward(self, hidden: torch.Tensor, features: torch.Tensor, mask: torch.Tensor, *,
                memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        context = self.read(hidden, features, mask, memory_write_mask=memory_write_mask)
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            correction = F.linear(context.to(dtype), self.output_projection.weight.to(dtype)).to(hidden.dtype)
        if not bool(torch.isfinite(correction).all()):
            raise ValueError("Episode correction must be finite")
        return correction


class _CountAwareEpisodeMemoryTPP(CountAwareTitanTPP):
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, *,
                 association: str, **quantity_kwargs: Any) -> None:
        episode_memory_metadata(hidden_dim, association)
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
                raise ValueError(f"Episode-memory route requires {name}={expected!r}")
        quantity_kwargs.pop("memory_mode", None)
        super().__init__(hidden_dim=hidden_dim, train_log_mean=train_log_mean,
                         max_seq_len=max_seq_len, memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
                         **quantity_kwargs)
        with torch.random.fork_rng(devices=[]):
            self.episode_memory = EpisodeMemory(hidden_dim, association=association)

    def encode_task_states(self, dts: torch.Tensor, history_quantities: torch.Tensor,
                           mask: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch, length]")
        _validate_mask(mask, tuple(dts.shape), "mask")
        if memory_write_mask is not None:
            _validate_mask(memory_write_mask, tuple(dts.shape), "memory_write_mask")
        # Execute the original B encoder and final static retrieval exactly once.
        hidden = super()._encode_base(dts, history_quantities, mask,
                                      memory_write_mask=memory_write_mask)
        if self.lmm is None:
            raise RuntimeError("Episode-memory route requires the original static Hard-LMM")
        valid = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
        baseline = self.lmm(hidden) * valid
        features = self.continuous_features(dts, history_quantities, mask)
        correction = self.episode_memory(hidden, features, mask, memory_write_mask=memory_write_mask)
        encoded = (baseline + correction) * valid
        return encoded, encoded


class CountAwareTitanSuccessorMemoryTPP(_CountAwareEpisodeMemoryTPP):
    """Match prior predecessor contexts, then read their observed successor."""
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
                 **quantity_kwargs: Any) -> None:
        if "association" in quantity_kwargs:
            raise ValueError("Successor backbone fixes its association mode")
        super().__init__(hidden_dim, train_log_mean, max_seq_len,
                         association="successor", **quantity_kwargs)


class CountAwareTitanSameEventMemoryTPP(_CountAwareEpisodeMemoryTPP):
    """Capacity/attention control: match the same observed successor event."""
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
                 **quantity_kwargs: Any) -> None:
        if "association" in quantity_kwargs:
            raise ValueError("Same-event backbone fixes its association mode")
        super().__init__(hidden_dim, train_log_mean, max_seq_len,
                         association="same_event", **quantity_kwargs)


__all__ = [
    "EpisodeMemory", "completed_pairs", "CountAwareTitanSuccessorMemoryTPP",
    "CountAwareTitanSameEventMemoryTPP", "SUCCESSOR_MEMORY_BACKBONE", "SAME_EVENT_MEMORY_BACKBONE",
    "SUCCESSOR_MEMORY_ROLE", "SAME_EVENT_MEMORY_ROLE", "EPISODE_MEMORY_CONTRACT_ID",
    "EPISODE_MEMORY_DESIGN_SHA256", "EPISODE_MEMORY_RANK", "EPISODE_MEMORY_LN_EPSILON",
    "EPISODE_MEMORY_PARAMETER_KEYS", "EPISODE_MEMORY_ASSOCIATION_KEY",
    "episode_memory_metadata", "validate_episode_memory_checkpoint",
]
