"""Observed-prefix slot memory between the unchanged Hard-LMM encoder layers.

This is an experimental backbone candidate, with no performance claim. The
memory is rebuilt from the current input on every call and never carries state
between batches, sequences, or prediction calls.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_QUANTITY_GRADIENT_SHARED,
)


SLOT_MEMORY_BACKBONE = "titantpp_observed_slot_memory"
SLOT_MEMORY_ROLE = "hard_lmm_observed_slot_memory_candidate"
SLOT_MEMORY_CONTRACT_ID = "hard_lmm_observed_slot_memory_v1"
SLOT_MEMORY_SLOTS = 8
SLOT_MEMORY_RANK = 8
SLOT_MEMORY_LN_EPSILON = 1e-5
SLOT_MEMORY_MASS_EPSILON = 1e-8
SLOT_MEMORY_PARAMETER_KEYS = (
    "slot_memory_alpha_raw",
    "slot_memory.addresses",
    "slot_memory.write_projection.weight",
    "slot_memory.query_projection.weight",
    "slot_memory.key_projection.weight",
)


def slot_memory_metadata(hidden_dim: int) -> dict[str, Any]:
    """Return the immutable identity of the selected experimental candidate."""
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of four")
    return {
        "candidate_name": "count_titan_hard_lmm_observed_slot_memory",
        "candidate_status": "experimental_unvalidated",
        "backbone_contract_id": SLOT_MEMORY_CONTRACT_ID,
        "routing_contract_id": SLOT_MEMORY_CONTRACT_ID,
        "model_role": SLOT_MEMORY_ROLE,
        "base_encoder": "CountAwareTitanTPP",
        "d_model": hidden_dim,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": hidden_dim * 2,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD,
        "persistent_mem_size": 16,
        "lmm_mem_size": 64,
        "lmm_topk": 4,
        "static_retrieval_similarity": "cosine",
        "static_retrieval_aggregation": "arithmetic_mean",
        "static_key_value_tied": True,
        "final_memory_placement": "after_encoder_layer_2",
        "slot_memory_placement": "after_encoder_layer_1_before_encoder_layer_2",
        "slot_memory_slots": SLOT_MEMORY_SLOTS,
        "slot_memory_rank": SLOT_MEMORY_RANK,
        "slot_memory_value_source": "absolute_first_layer_hidden_states",
        "slot_memory_value_transform": "none",
        "slot_memory_write_scope": "observed_inclusive_prefix",
        "slot_memory_write_mask": "mask_and_memory_write_mask_default_mask",
        "slot_memory_addressing": "affine_free_layer_norm_bias_free_linear",
        "slot_memory_layer_norm_epsilon": SLOT_MEMORY_LN_EPSILON,
        "slot_memory_mass_epsilon": SLOT_MEMORY_MASS_EPSILON,
        "slot_memory_write_rule": "softmax_write_address_cumulative_mass_and_moment",
        "slot_memory_read_rule": "supported_slot_softmax_projected_query_key",
        "slot_memory_empty_rule": "zero_unsupported_slots_and_correction",
        "slot_memory_correction": "slot_read_minus_current_hidden",
        "slot_memory_short_history_rule": "zero_correction_at_most_one_observed_event",
        "slot_memory_combination": "hidden+tanh(slot_memory_alpha_raw)*correction",
        "slot_memory_alpha_initialization": 0.0,
        "slot_memory_accumulation_dtype": "fp32_for_fp16_bfloat16_fp32_fp64_for_fp64",
        "slot_memory_carry": "none_rebuild_per_call",
        "slot_memory_additional_dropout": False,
        "quantity_variant": LOG_MSE_VARIANT,
        "quantity_memory_gradient_mode": TITAN_QUANTITY_GRADIENT_SHARED,
        "lambda_tail": 0.0,
        "new_parameter_state_keys": list(SLOT_MEMORY_PARAMETER_KEYS),
        "additional_parameter_count": 3 * hidden_dim * SLOT_MEMORY_RANK + 64 + 1,
        "online_writes": True,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _state_shapes(hidden_dim: int, max_len: int) -> dict[str, tuple[int, ...]]:
    """Describe fixed B and added state without constructing a random model."""
    dim = hidden_dim
    shapes = {
        "b_t": (1,),
        "w_raw": (1,),
        "v_t.weight": (1, dim),
        "quantity_head.weight": (1, dim),
        "quantity_head.bias": (1,),
        "encoder.pos_emb": (1, max_len, dim),
        "encoder.input_proj.weight": (dim, 2),
        "encoder.input_proj.bias": (dim,),
        "lmm.mem": (1, 64, dim),
        "slot_memory_alpha_raw": (),
        "slot_memory.addresses": (SLOT_MEMORY_SLOTS, SLOT_MEMORY_RANK),
        "slot_memory.write_projection.weight": (SLOT_MEMORY_RANK, dim),
        "slot_memory.query_projection.weight": (SLOT_MEMORY_RANK, dim),
        "slot_memory.key_projection.weight": (SLOT_MEMORY_RANK, dim),
    }
    for index in range(2):
        prefix = f"encoder.layers.{index}."
        shapes.update({
            prefix + "norm1.weight": (dim,),
            prefix + "norm1.bias": (dim,),
            prefix + "attn.persistent_mem": (1, 16, dim),
            prefix + "attn.qkv.weight": (3 * dim, dim),
            prefix + "attn.qkv.bias": (3 * dim,),
            prefix + "attn.out_proj.weight": (dim, dim),
            prefix + "attn.out_proj.bias": (dim,),
            prefix + "norm2.weight": (dim,),
            prefix + "norm2.bias": (dim,),
            prefix + "ff.0.weight": (2 * dim, dim),
            prefix + "ff.0.bias": (2 * dim,),
            prefix + "ff.3.weight": (dim, 2 * dim),
            prefix + "ff.3.bias": (dim,),
        })
    return shapes


def validate_slot_memory_checkpoint(
    payload: dict[str, Any], expected_backbone: str,
) -> bool:
    """Reject relabelled, incomplete, or incompatible candidate artifacts.

    Checkpoint tensors must contain exactly the fixed B state plus this
    candidate's five parameter tensors. A state-free summary is accepted only
    with its canonical state digest and nonnegative selected epoch, matching
    the existing bound-summary artifact path. Digest verification against the
    actual checkpoint remains the responsibility of that path's caller.
    """
    metadata = payload.get("encoder_config", {})
    states = [payload[name] for name in ("model_state_dict", "best_state_dict")
              if name in payload]
    has_candidate_state = any(
        isinstance(state, Mapping) and any(
            name == "slot_memory_alpha_raw"
            or isinstance(name, str) and name.startswith("slot_memory.")
            for name in state
        )
        for state in states
    )
    is_candidate = (
        payload.get("backbone") == SLOT_MEMORY_BACKBONE
        or isinstance(metadata, Mapping) and (
            metadata.get("backbone_contract_id") == SLOT_MEMORY_CONTRACT_ID
            or metadata.get("routing_contract_id") == SLOT_MEMORY_CONTRACT_ID
            or metadata.get("model_role") == SLOT_MEMORY_ROLE
            or metadata.get("candidate_name")
            == "count_titan_hard_lmm_observed_slot_memory"
        )
        or has_candidate_state
    )
    if expected_backbone != SLOT_MEMORY_BACKBONE:
        if is_candidate:
            raise ValueError("Observed-slot checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, Mapping):
        raise ValueError("Observed-slot checkpoint metadata must be a mapping")
    dim, max_len = metadata.get("d_model"), metadata.get("max_len")
    if type(dim) is not int or dim < 1 or dim % 4:
        raise ValueError("Observed-slot checkpoint hidden dimension is invalid")
    if type(max_len) is not int or max_len < 1:
        raise ValueError("Observed-slot checkpoint max_len is invalid")
    if payload.get("backbone") != SLOT_MEMORY_BACKBONE or any(
        metadata.get(name) != value
        for name, value in slot_memory_metadata(dim).items()
    ):
        raise ValueError("Observed-slot checkpoint routing metadata mismatch")
    time_head = metadata.get("time_head", {})
    if (
        payload.get("variant") != LOG_MSE_VARIANT
        or payload.get("evaluation_scope") != "validation_only"
        or payload.get("held_out_test_evaluated") is not False
        or not isinstance(time_head, Mapping)
        or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
        or time_head.get("jacobian_correction") is not False
        or time_head.get("wd_clamp") != 10.0
        or any(
            type(time_head.get(name)) not in (int, float)
            or not math.isfinite(time_head[name]) or time_head[name] <= 0.0
            for name in ("time_scale", "time_w_max", "time_intercept_limit")
        )
    ):
        raise ValueError("Observed-slot checkpoint scope/head/objective mismatch")
    # Reject contradictory optional training metadata as well as the required
    # encoder identity. Older artifacts need not carry these duplicated fields.
    for config in (payload, payload.get("training_config", {})):
        if not isinstance(config, Mapping) or any(
            name in config and config[name] != expected
            for name, expected in (
                ("quantity_variant", LOG_MSE_VARIANT),
                ("time_head_mode", TIME_HEAD_MODE_LEGACY_CLAMPED),
                ("lambda_tail", 0.0),
                ("quantile_adaptive_strength", 0.0),
                ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
            )
        ):
            raise ValueError("Observed-slot checkpoint training objective mismatch")
    if not states:
        if (
            _is_sha256(payload.get("checkpoint_state_sha256"))
            and type(payload.get("best_epoch")) is int
            and payload["best_epoch"] >= 0
            and ("checkpoint_file_sha256" not in payload
                 or _is_sha256(payload.get("checkpoint_file_sha256")))
        ):
            return True
        raise ValueError("Observed-slot summary requires epoch and canonical state digest")
    shapes = _state_shapes(dim, max_len)
    for state in states:
        if not isinstance(state, Mapping) or set(state) != set(shapes):
            raise ValueError("Observed-slot checkpoint state keys are missing or foreign")
        if any(
            not isinstance(state[name], torch.Tensor)
            or not state[name].is_floating_point()
            or tuple(state[name].shape) != shape
            or not bool(torch.isfinite(state[name]).all())
            for name, shape in shapes.items()
        ):
            raise ValueError("Observed-slot checkpoint tensor shape or finite state is invalid")
    return True


class ObservedSlotMemory(nn.Module):
    """Read eight slots from the inclusive observed prefix of hidden states.

    Let o_t = mask_t AND write_mask_t, and LN be affine-free LayerNorm. With
    rank r=8 and learned addresses A[8,8], write weights are
    a_t = softmax((W_write LN(h_t)) A^T / sqrt(r)). Prefix statistics are
    N_ts = sum_{i<=t} o_i a_is and U_ts = sum_{i<=t} o_i a_is h_i.
    Slot values S_ts = U_ts/N_ts retain the raw hidden-state level; unsupported
    slots (N_ts<=1e-8) are zero. Queries q_t=W_query LN(h_t) read keys
    k_ts=W_key LN(S_ts), with softmax(q_t dot k_ts/sqrt(r)) over N_ts>1e-8.
    Return sum_s p_ts S_ts - h_t, exactly zero for an invalid query or an
    observed prefix containing at most one event. No value normalization,
    output projection, dropout, or mutable carry is introduced.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self.num_slots = SLOT_MEMORY_SLOTS
        self.rank = SLOT_MEMORY_RANK
        self.write_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.key_projection = nn.Linear(hidden_dim, self.rank, bias=False)
        self.addresses = nn.Parameter(torch.randn(self.num_slots, self.rank))

    @staticmethod
    def _validate_mask(mask: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor")
        if tuple(mask.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")

    def forward(
        self, hidden: torch.Tensor, mask: torch.Tensor, *,
        memory_write_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[-1] != self.hidden_dim
                or not hidden.is_floating_point()):
            raise ValueError("hidden must be floating point with shape [batch, length, hidden_dim]")
        shape = tuple(hidden.shape[:2])
        self._validate_mask(mask, shape, "mask")
        valid = mask.to(device=hidden.device)
        if memory_write_mask is None:
            observed = valid
        else:
            self._validate_mask(memory_write_mask, shape, "memory_write_mask")
            observed = valid & memory_write_mask.to(device=hidden.device)
        safe_hidden = torch.where(valid.unsqueeze(-1), hidden, 0.0)
        if not bool(torch.isfinite(safe_hidden).all()):
            raise ValueError("hidden states at valid queries must be finite")
        compute_dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        # Disable ambient autocast so statistics and addressing really use the
        # documented compute dtype, including on the half-precision route.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            values = safe_hidden.to(dtype=compute_dtype)
            normalized = F.layer_norm(values, (self.hidden_dim,), eps=SLOT_MEMORY_LN_EPSILON)
            write_query = F.linear(normalized, self.write_projection.weight.to(compute_dtype))
            write_logits = write_query @ self.addresses.to(compute_dtype).transpose(0, 1)
            write_logits = write_logits / math.sqrt(self.rank)
            if not bool(torch.isfinite(write_logits).all()):
                raise ValueError("slot write addressing must be finite")
            assignments = torch.softmax(write_logits, dim=-1)
            writes = torch.where(observed.unsqueeze(-1), assignments, 0.0)
            mass = writes.cumsum(dim=1)
            moment = (writes.unsqueeze(-1) * values.unsqueeze(-2)).cumsum(dim=1)
            # Subnormal positive masses can produce finite U/N but an
            # overflowing 1/N in backward. Exclude numerically unsupported
            # slots before division; the cutoff is fixed independently of data.
            supported = mass > SLOT_MEMORY_MASS_EPSILON
            denominator = torch.where(supported, mass, 1.0)
            slots = moment / denominator.unsqueeze(-1)
            slots = torch.where(supported.unsqueeze(-1), slots, 0.0)
            if not bool(torch.isfinite(slots).all()):
                raise ValueError("supported slot state must be finite")
            queries = F.linear(normalized, self.query_projection.weight.to(compute_dtype))
            normalized_slots = F.layer_norm(slots, (self.hidden_dim,), eps=SLOT_MEMORY_LN_EPSILON)
            keys = F.linear(normalized_slots, self.key_projection.weight.to(compute_dtype))
            logits = (queries.unsqueeze(-2) * keys).sum(dim=-1) / math.sqrt(self.rank)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("slot read addressing must be finite")
            logits = logits.masked_fill(~supported, -torch.inf)
            # A wholly unsupported prefix uses benign logits before softmax;
            # masking afterward makes all its read weights exactly zero.
            logits = torch.where(supported.any(dim=-1, keepdim=True), logits, 0.0)
            weights = torch.where(supported, torch.softmax(logits, dim=-1), 0.0)
            context = (weights.unsqueeze(-1) * slots).sum(dim=-2)
            active = valid & (observed.cumsum(dim=1) > 1)
            correction = torch.where(active.unsqueeze(-1), context - values, 0.0)
            correction = correction.to(dtype=hidden.dtype)
        if not bool(torch.isfinite(correction).all()):
            raise ValueError("slot correction must be finite in the hidden-state dtype")
        return correction


class CountAwareTitanSlotMemoryTPP(CountAwareTitanTPP):
    """Fixed B with a zero-gated observed-slot correction between its layers."""

    def __init__(
        self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
        **quantity_kwargs: Any,
    ) -> None:
        slot_memory_metadata(hidden_dim)
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        for name, expected in (
            ("memory_mode", TITAN_MEMORY_MODE_STATIC_HARD),
            ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
            ("quantity_variant", LOG_MSE_VARIANT),
            ("time_head_mode", TIME_HEAD_MODE_LEGACY_CLAMPED),
            ("lambda_tail", 0.0),
            ("quantile_adaptive_strength", 0.0),
            ("quantile_adaptive_boundaries", None),
            ("quantile_adaptive_weights", None),
        ):
            if quantity_kwargs.get(name, expected) != expected:
                raise ValueError(f"Observed-slot candidate requires {name}={expected!r}")
        quantity_kwargs.pop("memory_mode", None)
        super().__init__(
            hidden_dim=hidden_dim, train_log_mean=train_log_mean,
            max_seq_len=max_seq_len, memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
            **quantity_kwargs,
        )
        if self.encoder is None or len(self.encoder.layers) != 2 or self.lmm is None:
            raise RuntimeError("Observed-slot candidate requires the fixed two-layer Hard-LMM base")
        # B construction completes first. Isolate only new CPU draws to retain
        # same-seed shared parameters and the caller's post-B CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            self.slot_memory = ObservedSlotMemory(hidden_dim)
            self.slot_memory_alpha_raw = nn.Parameter(torch.zeros(()))

    @property
    def slot_memory_alpha(self) -> torch.Tensor:
        return torch.tanh(self.slot_memory_alpha_raw)

    def _encode_base(
        self, dts: torch.Tensor, history_quantities: torch.Tensor,
        mask: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Observed-slot encoder is not initialized")
        ObservedSlotMemory._validate_mask(mask, tuple(dts.shape), "mask")
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("dts and history_quantities must have shape [batch, length]")
        if memory_write_mask is not None:
            ObservedSlotMemory._validate_mask(memory_write_mask, tuple(dts.shape), "memory_write_mask")
        valid = mask.to(device=dts.device)
        # Invalid padding may contain NaN/Inf sentinels; remove it before the
        # inherited logarithm/projection rather than relying on NaN * 0.
        features = self.continuous_features(
            torch.where(valid, dts, 0.0),
            torch.where(valid.to(history_quantities.device), history_quantities, 0.0),
            valid,
        )
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        valid_hidden = valid.to(encoded.device).unsqueeze(-1)
        encoded = torch.where(valid_hidden, encoded, 0.0)
        encoded = self.encoder.layers[0](encoded, mask=valid)
        correction = self.slot_memory(encoded, valid, memory_write_mask=memory_write_mask)
        encoded = encoded + self.slot_memory_alpha.to(encoded.dtype) * correction
        encoded = torch.where(valid_hidden, encoded, 0.0)
        encoded = self.encoder.layers[1](encoded, mask=valid)
        return torch.where(valid_hidden, encoded, 0.0)


__all__ = [
    "CountAwareTitanSlotMemoryTPP", "ObservedSlotMemory",
    "SLOT_MEMORY_BACKBONE", "SLOT_MEMORY_ROLE", "SLOT_MEMORY_CONTRACT_ID",
    "SLOT_MEMORY_SLOTS", "SLOT_MEMORY_RANK", "SLOT_MEMORY_LN_EPSILON",
    "SLOT_MEMORY_MASS_EPSILON",
    "SLOT_MEMORY_PARAMETER_KEYS", "slot_memory_metadata",
    "validate_slot_memory_checkpoint",
]
