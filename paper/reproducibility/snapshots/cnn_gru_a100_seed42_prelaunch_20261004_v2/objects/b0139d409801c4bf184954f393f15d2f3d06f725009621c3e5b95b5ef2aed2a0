"""Headwise RMS-bounded causal Q/K residuals with the original causal V path.

Only the event projections of encoder block 1 are changed. The extra residual
bound is relative to each unmodified projection; it is not a global attention
or model stability bound.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_KERNEL_KEYS,
    CausalQKVMemoryAttention,
    causal_qkv_metadata,
)
from models.Titan.common.memory import MemoryAttention


BOUNDED_QK_BACKBONE = "titantpp_hard_memory_bounded_qk"
BOUNDED_QK_ROLE = "hard_lmm_bounded_qk_candidate"
BOUNDED_QK_CONTRACT_ID = "hard_lmm_bounded_qk_causal_v_v1"
BOUNDED_QK_EPSILON = 1e-8
BOUNDED_QK_RHO = 1.0
BOUNDED_QK_KERNEL_KEYS = (
    "encoder.layers.0.attn.bounded_q_kernel",
    "encoder.layers.0.attn.bounded_k_kernel",
    "encoder.layers.0.attn.bounded_v_kernel",
)


def bounded_qk_metadata(hidden_dim: int) -> dict[str, Any]:
    """Return the fixed model identity, including the parameter-free bound."""
    return {
        **causal_qkv_metadata(hidden_dim),
        "candidate_name": "count_titan_hard_lmm_bounded_qk",
        "backbone_contract_id": BOUNDED_QK_CONTRACT_ID,
        "routing_contract_id": BOUNDED_QK_CONTRACT_ID,
        "model_role": BOUNDED_QK_ROLE,
        "causal_qkv_combination": "headwise_rms_bounded_qk_unbounded_causal_v",
        "new_parameter_state_keys": list(BOUNDED_QK_KERNEL_KEYS),
        "bounded_qk_scope": "first_block_event_qk_residual_only",
        "bounded_qk_reduction": "per_attention_head_feature_mean",
        "bounded_qk_head_dim": hidden_dim // 4,
        "bounded_qk_epsilon": BOUNDED_QK_EPSILON,
        "bounded_qk_rho": BOUNDED_QK_RHO,
        "bounded_qk_detach_projection_rms": False,
        "bounded_qk_learned_gate": False,
        "bounded_qk_norm_dtype": "fp32_for_fp16_bf16_otherwise_promoted_dtype",
        "bounded_qk_addition_dtype": "projection_residual_result_type",
        "bounded_qk_v_operation": "original_full_residual_addition",
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_bounded_qk_checkpoint(
    payload: dict[str, Any], expected_backbone: str,
) -> bool:
    """Reject B/FULL checkpoint relabelling and malformed bounded-Q/K state."""
    metadata = payload.get("encoder_config", {})
    states = [
        payload[name]
        for name in ("model_state_dict", "best_state_dict")
        if name in payload
    ]
    has_candidate_state = any(
        isinstance(state, dict)
        and any(name in state for name in BOUNDED_QK_KERNEL_KEYS)
        for state in states
    )
    is_candidate = (
        payload.get("backbone") == BOUNDED_QK_BACKBONE
        or (
            isinstance(metadata, dict)
            and (
                metadata.get("routing_contract_id") == BOUNDED_QK_CONTRACT_ID
                or metadata.get("model_role") == BOUNDED_QK_ROLE
            )
        )
        or has_candidate_state
    )
    if expected_backbone != BOUNDED_QK_BACKBONE:
        if is_candidate:
            raise ValueError("Bounded-QK checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, dict):
        raise ValueError("Bounded-QK checkpoint metadata must be a mapping")
    hidden_dim = metadata.get("d_model")
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("Bounded-QK checkpoint hidden dimension is invalid")
    required = bounded_qk_metadata(hidden_dim)
    if payload.get("backbone") != BOUNDED_QK_BACKBONE or any(
        metadata.get(name) != value for name, value in required.items()
    ):
        raise ValueError("Bounded-QK checkpoint routing metadata mismatch")
    time_head = metadata.get("time_head", {})
    if (
        payload.get("variant") != LOG_MSE_VARIANT
        or payload.get("evaluation_scope") != "validation_only"
        or payload.get("held_out_test_evaluated") is not False
        or not isinstance(time_head, dict)
        or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
    ):
        raise ValueError("Bounded-QK checkpoint scope/head/objective mismatch")
    if not states:
        if (
            _is_sha256(payload.get("checkpoint_state_sha256"))
            and type(payload.get("best_epoch")) is int
            and payload["best_epoch"] >= 0
            and (
                "checkpoint_file_sha256" not in payload
                or _is_sha256(payload.get("checkpoint_file_sha256"))
            )
        ):
            return True
        raise ValueError("Bounded-QK summary requires epoch and canonical state digest")
    for state in states:
        if not isinstance(state, dict) or any(
            not isinstance(state.get(name), torch.Tensor)
            or tuple(state[name].shape) != (3, hidden_dim)
            or not bool(torch.isfinite(state[name]).all())
            for name in BOUNDED_QK_KERNEL_KEYS
        ):
            raise ValueError("Bounded-QK checkpoint kernels are missing or invalid")
        if any(name in state for name in CAUSAL_QKV_KERNEL_KEYS):
            raise ValueError("FULL checkpoint kernels cannot be relabelled as Bounded-QK")
        lmm = state.get("lmm.mem")
        if (
            not isinstance(lmm, torch.Tensor)
            or tuple(lmm.shape) != (1, 64, hidden_dim)
            or not bool(torch.isfinite(lmm).all())
        ):
            raise ValueError("Bounded-QK checkpoint Hard-LMM state is invalid")
    return True


class BoundedQKMemoryAttention(CausalQKVMemoryAttention):
    """Reuse FULL attention while bounding only event Q/K residuals per head."""

    def __init__(self, base: MemoryAttention) -> None:
        super().__init__(base)
        # Keep FULL's zero allocation and parameter order without consuming RNG.
        # Distinct keys prevent accidentally restoring FULL as this candidate.
        for projection in ("q", "k", "v"):
            parameter = self._parameters.pop(f"causal_{projection}_kernel")
            self.register_parameter(f"bounded_{projection}_kernel", parameter)

    def bound_qk_residual(
        self, projection: torch.Tensor, residual: torch.Tensor,
    ) -> torch.Tensor:
        """Limit each head's added RMS, preserving FULL's promoted add dtype."""
        if projection.shape != residual.shape or projection.ndim != 3:
            raise ValueError("projection and residual must have equal [B,L,D] shape")
        if projection.shape[-1] != self.d_model:
            raise ValueError("projection channels must match attention d_model")
        result_dtype = torch.result_type(projection, residual)
        norm_dtype = (
            torch.float32
            if result_dtype in (torch.float16, torch.bfloat16)
            else result_dtype
        )
        head_shape = (*projection.shape[:-1], self.n_heads, self.head_dim)
        p = projection.to(dtype=norm_dtype).reshape(head_shape)
        r = residual.to(dtype=norm_dtype).reshape(head_shape)
        scale_squared = p.square().mean(dim=-1, keepdim=True) + BOUNDED_QK_EPSILON
        residual_squared = r.square().mean(dim=-1, keepdim=True)
        gain = scale_squared.sqrt() / (scale_squared + residual_squared).sqrt()
        return (r * gain).reshape_as(residual).to(dtype=result_dtype)

    def _adapt_event_projections(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        input_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, length, _ = q.shape
        if mask is not None:
            if mask.shape != (batch_size, length):
                raise ValueError(
                    f"Expected attention mask {(batch_size, length)}, got {tuple(mask.shape)}"
                )
            event_valid = mask.to(
                device=q.device, dtype=input_dtype or q.dtype
            ).unsqueeze(-1)
            causal_q, causal_k, causal_v = q * event_valid, k * event_valid, v * event_valid
        else:
            causal_q, causal_k, causal_v = q, k, v
        q_residual = self.causal_depthwise_residual(causal_q, self.bounded_q_kernel)
        k_residual = self.causal_depthwise_residual(causal_k, self.bounded_k_kernel)
        q = q + self.bound_qk_residual(q, q_residual)
        k = k + self.bound_qk_residual(k, k_residual)
        v = v + self.causal_depthwise_residual(causal_v, self.bounded_v_kernel)
        return q, k, v


class CountAwareTitanBoundedQKTPP(CountAwareTitanTPP):
    """The common two-block Hard-LMM with bounded first-block causal Q/K."""

    contract_id = BOUNDED_QK_CONTRACT_ID
    backbone_name = BOUNDED_QK_BACKBONE

    def __init__(
        self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
        **quantity_kwargs: Any,
    ) -> None:
        if "memory_mode" in quantity_kwargs:
            raise ValueError("CountAwareTitanBoundedQKTPP fixes memory_mode to static_hard_lmm")
        if "quantity_memory_gradient_mode" in quantity_kwargs:
            raise ValueError("CountAwareTitanBoundedQKTPP uses the shared Hard-LMM gradient route")
        super().__init__(
            hidden_dim=hidden_dim,
            train_log_mean=train_log_mean,
            max_seq_len=max_seq_len,
            memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
            **quantity_kwargs,
        )
        if self.encoder is None or len(self.encoder.layers) != 2:
            raise RuntimeError("Bounded-QK requires the two-layer local encoder")
        first_attention = self.encoder.layers[0].attn
        if not isinstance(first_attention, MemoryAttention):
            raise RuntimeError("Bounded-QK requires MemoryAttention in encoder block 1")
        self.encoder.layers[0].attn = BoundedQKMemoryAttention(first_attention)

    @property
    def bounded_qk_attention(self) -> BoundedQKMemoryAttention:
        if self.encoder is None:
            raise RuntimeError("Bounded-QK encoder is not initialized")
        attention = self.encoder.layers[0].attn
        if not isinstance(attention, BoundedQKMemoryAttention):
            raise RuntimeError("Bounded-QK first attention route is missing")
        return attention

    @property
    def additional_parameter_count(self) -> int:
        return 9 * self.hidden_dim

    @torch.no_grad()
    def reset_bounded_qk_identity(self) -> None:
        attention = self.bounded_qk_attention
        attention.bounded_q_kernel.zero_()
        attention.bounded_k_kernel.zero_()
        attention.bounded_v_kernel.zero_()

    def load_hard_lmm_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Explicit B initialization; not a FULL/candidate checkpoint restore."""
        expected = {
            name: tensor for name, tensor in self.state_dict().items()
            if name not in BOUNDED_QK_KERNEL_KEYS
        }
        missing = set(expected) - set(state_dict)
        unexpected = set(state_dict) - set(expected)
        if missing:
            raise RuntimeError(f"Hard-LMM B checkpoint is missing inherited parameters: {sorted(missing)}")
        if unexpected:
            raise RuntimeError(f"Hard-LMM B checkpoint has unexpected parameters: {sorted(unexpected)}")
        for name, tensor in state_dict.items():
            if not isinstance(tensor, torch.Tensor) or tensor.shape != expected[name].shape:
                raise RuntimeError(f"Hard-LMM B checkpoint has an invalid tensor: {name}")
        complete = dict(state_dict)
        complete.update({
            name: torch.zeros_like(self.state_dict()[name])
            for name in BOUNDED_QK_KERNEL_KEYS
        })
        super().load_state_dict(complete, strict=True)


__all__ = [
    "BOUNDED_QK_BACKBONE", "BOUNDED_QK_ROLE", "BOUNDED_QK_CONTRACT_ID",
    "BOUNDED_QK_EPSILON", "BOUNDED_QK_RHO", "BOUNDED_QK_KERNEL_KEYS",
    "BoundedQKMemoryAttention", "CountAwareTitanBoundedQKTPP",
    "bounded_qk_metadata", "validate_bounded_qk_checkpoint",
]
