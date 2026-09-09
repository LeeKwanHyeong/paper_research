"""Bounded local/global transition memory inside the common Hard-LMM encoder.

Prefix statistics are rebuilt from observed events on every forward. The last
8 admitted transitions and the older prefix have disjoint associative states;
there is no cross-forward state, event loop, or inner optimizer.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP, LOG_MSE_VARIANT, TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
from models.TPPs.CountAwareTitanValueNorm import VALUE_NORM_REQUIRED_STATE_KEYS

DUAL_TIMESCALE_BACKBONE = "titantpp_dual_timescale_memory"
DUAL_TIMESCALE_ROLE = "hard_lmm_dual_timescale_candidate"
DUAL_TIMESCALE_CONTRACT_ID = "hard_lmm_dual_timescale_v1"
DUAL_TIMESCALE_RANK = 8
DUAL_TIMESCALE_LOCAL_TRANSITIONS = 8
DUAL_TIMESCALE_EPSILON = 1e-8
DUAL_TIMESCALE_INPUT_BOUND = 1e4
DUAL_TIMESCALE_STATE_PREFIX = "transition_memory."


def _new_state_shapes(hidden_dim: int) -> dict[str, tuple[int, ...]]:
    return {
        "alpha_raw": (), "norm.weight": (hidden_dim,), "norm.bias": (hidden_dim,),
        "query.weight": (8, hidden_dim), "key.weight": (8, hidden_dim),
        "value.weight": (hidden_dim, hidden_dim), "value.bias": (hidden_dim,),
        "write.weight": (1, 1), "write.bias": (1,),
        "mix.weight": (1, 3), "mix.bias": (1,),
    }


def dual_timescale_metadata(hidden_dim: int) -> dict[str, Any]:
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    return {
        "candidate_name": "count_titan_dual_timescale_transition_memory",
        "backbone_contract_id": DUAL_TIMESCALE_CONTRACT_ID,
        "routing_contract_id": DUAL_TIMESCALE_CONTRACT_ID,
        "model_role": DUAL_TIMESCALE_ROLE,
        "base_encoder": "CountAwareTitanTPP", "d_model": hidden_dim,
        "n_layers": 2, "n_heads": 4, "d_ff": 2 * hidden_dim,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD,
        "persistent_mem_size": 16, "lmm_mem_size": 64, "lmm_topk": 4,
        "static_retrieval_aggregation": "arithmetic_mean", "static_key_value_tied": True,
        "memory_placement": "after_encoder_layer_1_before_layer_2",
        "final_memory_placement": "unchanged_static_hard_lmm_after_layer_2",
        "online_writes": True, "mutable_cross_forward_state": False,
        "transition_rank": DUAL_TIMESCALE_RANK,
        "local_transition_count": DUAL_TIMESCALE_LOCAL_TRANSITIONS,
        "global_scope": "all_admitted_prefix_transitions_older_than_local",
        "query": "softmax_linear_layernorm_current_hidden",
        "key": "softmax_linear_layernorm_previous_hidden",
        "value": "tanh_linear_clipped_current_minus_previous_hidden",
        "write_confidence": "sigmoid_linear_mean_absolute_observed_hidden_innovation",
        "write_admission": "both_adjacent_positions_valid_and_observed",
        "read_timing": "inclusive_observed_prefix_for_next_event_prediction",
        "read": "normalized_nonnegative_associative_sufficient_statistics",
        "fusion": "sigmoid_linear_log1p_local_mass_log1p_global_mass_innovation",
        "missing_path_policy": "mask_and_renormalize_available_path",
        "residual": "tanh_zero_initialized_scalar_times_bounded_fused_read",
        "read_value_bound": 1.0, "projection_input_bound": DUAL_TIMESCALE_INPUT_BOUND,
        "epsilon": DUAL_TIMESCALE_EPSILON,
        "accumulation_dtype": "float32_unless_float64_input",
        "shared_time_quantity_state": True,
        "additional_parameter_count": sum(
            int(torch.tensor(shape).prod()) if shape else 1
            for shape in _new_state_shapes(hidden_dim).values()
        ),
        "new_parameter_state_keys": [
            DUAL_TIMESCALE_STATE_PREFIX + key for key in _new_state_shapes(hidden_dim)
        ],
    }


def validate_dual_timescale_checkpoint(payload: dict[str, Any], expected_backbone: str) -> bool:
    metadata = payload.get("encoder_config", {})
    states = [payload[name] for name in ("model_state_dict", "best_state_dict") if name in payload]
    is_candidate = (
        payload.get("backbone") == DUAL_TIMESCALE_BACKBONE
        or isinstance(metadata, dict) and (
            metadata.get("routing_contract_id") == DUAL_TIMESCALE_CONTRACT_ID
            or metadata.get("model_role") == DUAL_TIMESCALE_ROLE)
        or any(isinstance(state, dict) and any(
            name.startswith(DUAL_TIMESCALE_STATE_PREFIX) for name in state
        ) for state in states)
    )
    if expected_backbone != DUAL_TIMESCALE_BACKBONE:
        if is_candidate:
            raise ValueError("Dual-timescale checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, dict):
        raise ValueError("Dual-timescale checkpoint metadata must be a mapping")
    dim = metadata.get("d_model")
    if type(dim) is not int or dim < 1:
        raise ValueError("Dual-timescale checkpoint hidden dimension is invalid")
    if payload.get("backbone") != expected_backbone or any(
        metadata.get(key) != value for key, value in dual_timescale_metadata(dim).items()
    ):
        raise ValueError("Dual-timescale checkpoint routing metadata mismatch")
    head = metadata.get("time_head", {})
    if (payload.get("variant") != LOG_MSE_VARIANT
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or not isinstance(head, dict)
            or head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED):
        raise ValueError("Dual-timescale checkpoint scope/head/objective mismatch")
    if not states:
        digest = payload.get("checkpoint_state_sha256")
        if (isinstance(digest, str) and len(digest) == 64
                and all(c in "0123456789abcdef" for c in digest)
                and type(payload.get("best_epoch")) is int and payload["best_epoch"] >= 0):
            return True
        raise ValueError("Dual-timescale summary requires epoch and canonical state digest")
    shapes = {DUAL_TIMESCALE_STATE_PREFIX + k: v for k, v in _new_state_shapes(dim).items()}
    required = (set(VALUE_NORM_REQUIRED_STATE_KEYS) - {"lmm.alpha_raw"}) | set(shapes)
    for state in states:
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError("Dual-timescale checkpoint state key population is invalid")
        if any(not isinstance(v, torch.Tensor) or not v.is_floating_point()
               or not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError("Dual-timescale checkpoint contains an invalid tensor")
        if any(tuple(state[k].shape) != shape for k, shape in shapes.items()):
            raise ValueError("Dual-timescale checkpoint memory tensor shape mismatch")
        if tuple(state["lmm.mem"].shape) != (1, 64, dim):
            raise ValueError("Dual-timescale checkpoint prototype shape mismatch")
    return True


class DualTimescaleTransitionMemory(nn.Module):
    """Vectorized observed-prefix local/global associative transition store."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, DUAL_TIMESCALE_RANK, bias=False)
        self.key = nn.Linear(hidden_dim, DUAL_TIMESCALE_RANK, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.write = nn.Linear(1, 1)
        self.mix = nn.Linear(3, 1)
        nn.init.constant_(self.write.weight, -0.1)
        nn.init.zeros_(self.write.bias)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_raw.tanh()

    def residual_components(
        self, hidden: torch.Tensor, mask: torch.Tensor,
        memory_write_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("hidden [B,L,D] and boolean mask [B,L] are required")
        if hidden.size(1) < 1:
            raise ValueError("memory requires at least one position")
        if memory_write_mask is not None and (
            memory_write_mask.shape != mask.shape or memory_write_mask.dtype != torch.bool
        ):
            raise ValueError("memory_write_mask must be boolean [B,L]")
        valid = mask.to(device=hidden.device)
        observed = valid if memory_write_mask is None else valid & memory_write_mask.to(hidden.device)
        # Invalid positions are removed before arithmetic, including NaN padding.
        safe = torch.where(valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        safe = safe.clamp(-DUAL_TIMESCALE_INPUT_BOUND, DUAL_TIMESCALE_INPUT_BOUND)
        previous = F.pad(safe[:, :-1], (0, 0, 1, 0))
        admitted = observed & F.pad(observed[:, :-1], (1, 0), value=False)
        innovation = torch.where(admitted.unsqueeze(-1), safe - previous, torch.zeros_like(safe))
        normed = self.norm(safe)
        previous_normed = F.pad(normed[:, :-1], (0, 0, 1, 0))
        queries = self.query(normed).softmax(dim=-1)
        keys = self.key(previous_normed).softmax(dim=-1)
        values = self.value(innovation).tanh()
        innovation_norm = innovation.abs().mean(dim=-1, keepdim=True)
        trust = self.write(innovation_norm).sigmoid() * admitted.unsqueeze(-1)
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        queries, keys, values, trust = (x.to(dtype) for x in (queries, keys, values, trust))
        weighted_keys = keys * trust
        writes = weighted_keys.unsqueeze(-1) * values.unsqueeze(-2)
        key_prefix = weighted_keys.cumsum(dim=1)
        value_prefix = writes.cumsum(dim=1)
        write_count = admitted.long().cumsum(dim=1)
        old_count = (write_count - DUAL_TIMESCALE_LOCAL_TRANSITIONS).clamp_min(0)
        # searchsorted maps transition counts to prefix boundaries. Equal-count
        # positions contain zero writes, so choosing their right edge is causal.
        boundary = torch.searchsorted(write_count.contiguous(), old_count.contiguous(), right=True)
        padded_keys = F.pad(key_prefix, (0, 0, 1, 0))
        padded_values = F.pad(value_prefix, (0, 0, 0, 0, 1, 0))
        global_keys = padded_keys.gather(1, boundary.unsqueeze(-1).expand(-1, -1, keys.size(-1)))
        global_values = padded_values.gather(1, boundary[..., None, None].expand(
            -1, -1, keys.size(-1), values.size(-1)))
        local_keys, local_values = key_prefix - global_keys, value_prefix - global_values

        def read(key_sum: torch.Tensor, value_sum: torch.Tensor):
            mass = (queries * key_sum).sum(-1, keepdim=True).clamp_min(0)
            numerator = (queries.unsqueeze(-1) * value_sum).sum(-2)
            result = (numerator / mass.clamp_min(DUAL_TIMESCALE_EPSILON)).clamp(-1., 1.)
            return torch.where(mass > 0, result, torch.zeros_like(result)), mass

        local, local_mass = read(local_keys, local_values)
        global_read, global_mass = read(global_keys, global_values)
        mix_features = torch.cat((local_mass.log1p(), global_mass.log1p(), innovation_norm.to(dtype)), -1)
        learned_mix = self.mix(mix_features.to(hidden.dtype)).sigmoid().to(dtype)
        local_available = (write_count > 0).unsqueeze(-1)
        global_available = (old_count > 0).unsqueeze(-1)
        local_weight = learned_mix * local_available
        global_weight = (1 - learned_mix) * global_available
        denominator = (local_weight + global_weight).clamp_min(DUAL_TIMESCALE_EPSILON)
        local_weight, global_weight = local_weight / denominator, global_weight / denominator
        residual = (local_weight * local + global_weight * global_read).clamp(-1., 1.)
        residual = torch.where(valid.unsqueeze(-1), residual, torch.zeros_like(residual))
        return {
            "residual": residual.to(hidden.dtype), "local_read": local,
            "global_read": global_read, "local_mass": local_mass,
            "global_mass": global_mass, "local_weight": local_weight,
            "global_weight": global_weight, "admitted": admitted,
            "write_count": write_count, "global_count": old_count,
            "queries": queries, "keys": keys, "values": values, "trust": trust,
        }

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor,
                memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.residual_components(hidden, mask, memory_write_mask)["residual"]
        return hidden + self.alpha.to(hidden.dtype) * residual


class CountAwareTitanDualTimescaleTPP(CountAwareTitanTPP):
    """Original B with a single identity-initialized inter-layer memory route."""
    contract_id = DUAL_TIMESCALE_CONTRACT_ID
    backbone_name = DUAL_TIMESCALE_BACKBONE

    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
                 **quantity_kwargs: Any) -> None:
        if "memory_mode" in quantity_kwargs or "quantity_memory_gradient_mode" in quantity_kwargs:
            raise ValueError("Dual-timescale candidate fixes original shared Hard-LMM route")
        super().__init__(hidden_dim=hidden_dim, train_log_mean=train_log_mean,
                         max_seq_len=max_seq_len, memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
                         **quantity_kwargs)
        # Extra parameters must not consume RNG from B or the subsequent loader.
        with torch.random.fork_rng(devices=[]):
            self.transition_memory = DualTimescaleTransitionMemory(hidden_dim)

    @property
    def additional_parameter_count(self) -> int:
        return sum(p.numel() for p in self.transition_memory.parameters())

    def _encode_base(self, dts: torch.Tensor, history_quantities: torch.Tensor,
                     mask: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.encoder is None or len(self.encoder.layers) != 2:
            raise RuntimeError("Dual-timescale candidate requires original two-layer encoder")
        features = self.continuous_features(dts, history_quantities, mask)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        valid = mask.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(-1)
        encoded = encoded * valid
        encoded = self.encoder.layers[0](encoded, mask=mask)
        encoded = self.transition_memory(encoded, mask, memory_write_mask)
        encoded = self.encoder.layers[1](encoded, mask=mask)
        return encoded * valid
