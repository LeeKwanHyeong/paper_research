"""Hard-LMM backbone with causal depthwise event Q/K/V adaptation.

The candidate changes only the first encoder block's event projections.  Each
projected query, key, and value receives a channelwise causal kernel-3 residual
before persistent memory is concatenated.  All new kernels are exactly zero at
initialization, so the inherited Hard-LMM model is the exact initial route.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
from models.Titan.common.memory import MemoryAttention


CAUSAL_QKV_BACKBONE = "titantpp_hard_memory_causal_qkv"
CAUSAL_QKV_ROLE = "hard_lmm_causal_qkv_candidate"
CAUSAL_QKV_CONTRACT_ID = "hard_lmm_causal_qkv_v1"
CAUSAL_QKV_KERNEL_KEYS = (
    "encoder.layers.0.attn.causal_q_kernel",
    "encoder.layers.0.attn.causal_k_kernel",
    "encoder.layers.0.attn.causal_v_kernel",
)


def causal_qkv_metadata(hidden_dim: int) -> dict[str, Any]:
    """Return the immutable routing identity for checkpoint artifacts."""
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of four")
    return {
        "candidate_name": "count_titan_hard_lmm_causal_qkv",
        "backbone_contract_id": CAUSAL_QKV_CONTRACT_ID,
        "routing_contract_id": CAUSAL_QKV_CONTRACT_ID,
        "model_role": CAUSAL_QKV_ROLE,
        "base_encoder": "CountAwareTitanTPP",
        "d_model": hidden_dim,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": hidden_dim * 2,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD,
        "persistent_mem_size": 16,
        "lmm_mem_size": 64,
        "lmm_topk": 4,
        "static_retrieval_aggregation": "arithmetic_mean",
        "causal_qkv_layer": 0,
        "causal_qkv_scope": "event_qkv_only",
        "causal_qkv_kernel_size": 3,
        "causal_qkv_groups": hidden_dim,
        "causal_qkv_bias": False,
        "causal_qkv_lag_order": ["current", "lag1", "lag2"],
        "causal_qkv_combination": "residual_addition",
        "causal_qkv_initialization": "all_zero_exact_identity",
        "persistent_key_value_adaptation": False,
        "final_memory_placement": "after_encoder_layer_2",
        "new_parameter_state_keys": list(CAUSAL_QKV_KERNEL_KEYS),
        "additional_parameter_count": 9 * hidden_dim,
        "online_writes": False,
    }


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_causal_qkv_checkpoint(
    payload: dict[str, Any],
    expected_backbone: str,
) -> bool:
    """Validate route identity and reject causal-QKV checkpoint relabelling."""
    metadata = payload.get("encoder_config", {})
    states = [
        payload[name]
        for name in ("model_state_dict", "best_state_dict")
        if name in payload
    ]
    has_causal_qkv_state = any(
        isinstance(state, dict)
        and any(name in state for name in CAUSAL_QKV_KERNEL_KEYS)
        for state in states
    )
    is_candidate = (
        payload.get("backbone") == CAUSAL_QKV_BACKBONE
        or (
            isinstance(metadata, dict)
            and (
                metadata.get("routing_contract_id") == CAUSAL_QKV_CONTRACT_ID
                or metadata.get("model_role") == CAUSAL_QKV_ROLE
            )
        )
        or has_causal_qkv_state
    )
    if expected_backbone != CAUSAL_QKV_BACKBONE:
        if is_candidate:
            raise ValueError("Causal-QKV checkpoint cannot be relabelled")
        return False

    if not isinstance(metadata, dict):
        raise ValueError("Causal-QKV checkpoint metadata must be a mapping")
    hidden_dim = metadata.get("d_model")
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("Causal-QKV checkpoint hidden dimension is invalid")
    required = causal_qkv_metadata(hidden_dim)
    if payload.get("backbone") != CAUSAL_QKV_BACKBONE or any(
        metadata.get(name) != value for name, value in required.items()
    ):
        raise ValueError("Causal-QKV checkpoint routing metadata mismatch")

    time_head = metadata.get("time_head", {})
    if (
        payload.get("variant") != LOG_MSE_VARIANT
        or payload.get("evaluation_scope") != "validation_only"
        or payload.get("held_out_test_evaluated") is not False
        or not isinstance(time_head, dict)
        or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
    ):
        raise ValueError("Causal-QKV checkpoint scope/head/objective mismatch")

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
        raise ValueError(
            "Causal-QKV summary requires epoch and canonical state digest"
        )

    expected_shape = (3, hidden_dim)
    for state in states:
        if not isinstance(state, dict) or any(
            not isinstance(state.get(name), torch.Tensor)
            or tuple(state[name].shape) != expected_shape
            or not bool(torch.isfinite(state[name]).all())
            for name in CAUSAL_QKV_KERNEL_KEYS
        ):
            raise ValueError("Causal-QKV checkpoint kernels are missing or invalid")
        lmm = state.get("lmm.mem")
        if (
            not isinstance(lmm, torch.Tensor)
            or tuple(lmm.shape) != (1, 64, hidden_dim)
            or not bool(torch.isfinite(lmm).all())
        ):
            raise ValueError("Causal-QKV checkpoint Hard-LMM state is invalid")
    return True


class CausalQKVMemoryAttention(MemoryAttention):
    """MemoryAttention with event-only channelwise causal Q/K/V residuals."""

    def __init__(self, base: MemoryAttention) -> None:
        # Do not call MemoryAttention.__init__: it would redraw all inherited
        # parameters. Re-register the already initialized modules and parameter
        # objects, then create only deterministic zero-valued candidate kernels.
        nn.Module.__init__(self)
        self.d_model = base.d_model
        self.n_heads = base.n_heads
        self.use_causal = base.use_causal
        self.head_dim = base.head_dim
        self.scale = base.scale
        self.qkv = base.qkv
        self.out_proj = base.out_proj
        self.drop = base.drop
        self.contextual_mem_size = base.contextual_mem_size
        self.persistent_mem_size = base.persistent_mem_size
        if base.persistent_mem is None:
            self.register_parameter("persistent_mem", None)
        else:
            self.persistent_mem = base.persistent_mem
        self._ctx_buf = base._ctx_buf
        self._ctx_mem = base._ctx_mem

        self.causal_q_kernel = nn.Parameter(torch.zeros(3, self.d_model))
        self.causal_k_kernel = nn.Parameter(torch.zeros(3, self.d_model))
        self.causal_v_kernel = nn.Parameter(torch.zeros(3, self.d_model))

    @staticmethod
    def causal_depthwise_residual(
        values: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a left-looking depthwise kernel with rows current/lag1/lag2."""
        if values.ndim != 3:
            raise ValueError("values must have shape [batch, length, channels]")
        if kernel.shape != (3, values.size(-1)):
            raise ValueError("kernel must have shape [3, channels]")
        length = values.size(1)
        lag1 = F.pad(values, (0, 0, 1, 0))[:, :length]
        lag2 = F.pad(values, (0, 0, 2, 0))[:, :length]
        return values * kernel[0] + lag1 * kernel[1] + lag2 * kernel[2]

    def _adapt_event_projections(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        input_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adapt event projections before persistent K/V are concatenated."""
        batch_size, length, _ = q.shape
        if mask is not None:
            if mask.shape != (batch_size, length):
                raise ValueError(
                    f"Expected attention mask {(batch_size, length)}, "
                    f"got {tuple(mask.shape)}"
                )
            event_valid = mask.to(
                device=q.device, dtype=input_dtype or q.dtype
            ).unsqueeze(-1)
            causal_q = q * event_valid
            causal_k = k * event_valid
            causal_v = v * event_valid
        else:
            causal_q, causal_k, causal_v = q, k, v
        q = q + self.causal_depthwise_residual(causal_q, self.causal_q_kernel)
        k = k + self.causal_depthwise_residual(causal_k, self.causal_k_kernel)
        v = v + self.causal_depthwise_residual(causal_v, self.causal_v_kernel)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        *,
        event_attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, length, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = self._adapt_event_projections(
            *torch.chunk(qkv, 3, dim=-1), mask, input_dtype=x.dtype
        )

        memory_parts = []
        if self._ctx_mem is not None and self._ctx_mem.numel() > 0:
            memory_parts.append(self._ctx_mem.to(device=x.device, dtype=x.dtype))
        if self.persistent_mem is not None:
            memory_parts.append(
                self.persistent_mem.to(device=x.device, dtype=x.dtype).expand(
                    batch_size, -1, -1
                )
            )

        memory_length = 0
        if memory_parts:
            memory = torch.cat(memory_parts, dim=1)
            memory_length = memory.size(1)
            # Persistent/context memory is intentionally not convolved.
            k = torch.cat([memory, k], dim=1)
            v = torch.cat([memory, v], dim=1)

        qh = self._split_heads(q)
        kh = self._split_heads(k)
        vh = self._split_heads(v)
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale
        if event_attention_bias is not None:
            expected = (batch_size, self.n_heads, length, length)
            if event_attention_bias.shape != expected:
                raise ValueError(
                    "Event attention bias must have shape "
                    "[batch, heads, length, length]"
                )
            bias = F.pad(
                event_attention_bias.to(
                    dtype=scores.dtype,
                    device=scores.device,
                ),
                (memory_length, 0),
            )
            scores = scores + bias

        full_mask: torch.Tensor | None = None
        if self.use_causal:
            memory_visible = torch.ones(
                length,
                memory_length,
                device=x.device,
                dtype=torch.bool,
            )
            causal_visible = torch.tril(
                torch.ones(length, length, device=x.device, dtype=torch.bool)
            )
            full_mask = torch.cat([memory_visible, causal_visible], dim=1)

        if mask is not None:
            valid_keys = torch.cat(
                [
                    torch.ones(
                        batch_size,
                        memory_length,
                        device=x.device,
                        dtype=torch.bool,
                    ),
                    mask.to(device=x.device, dtype=torch.bool),
                ],
                dim=1,
            )
            key_mask = valid_keys[:, None, None, :]
            if full_mask is None:
                full_mask = key_mask
            else:
                full_mask = full_mask[None, None, :, :] & key_mask

            if memory_length == 0:
                invalid_queries = ~mask.to(device=x.device, dtype=torch.bool)
                if invalid_queries.any():
                    full_mask = full_mask.expand(
                        batch_size, 1, length, length
                    ).clone()
                    batch_ids, positions = invalid_queries.nonzero(as_tuple=True)
                    full_mask[batch_ids, 0, positions, positions] = True

        if full_mask is not None:
            scores = scores.masked_fill(~full_mask, float("-inf"))

        attention = F.softmax(scores, dim=-1)
        attention = self.drop(attention)
        output = torch.matmul(attention, vh)
        output = self._merge_heads(output)
        output = self.out_proj(output)
        if mask is not None:
            output = output * mask.to(
                device=output.device,
                dtype=output.dtype,
            ).unsqueeze(-1)
        return output


class CountAwareTitanCausalQKVTPP(CountAwareTitanTPP):
    """Apply a zero-initialized causal Q/K/V residual in encoder block 1."""

    contract_id = CAUSAL_QKV_CONTRACT_ID
    backbone_name = CAUSAL_QKV_BACKBONE

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        max_seq_len: int,
        **quantity_kwargs: Any,
    ) -> None:
        if "memory_mode" in quantity_kwargs:
            raise ValueError(
                "CountAwareTitanCausalQKVTPP fixes memory_mode to static_hard_lmm"
            )
        if "quantity_memory_gradient_mode" in quantity_kwargs:
            raise ValueError(
                "CountAwareTitanCausalQKVTPP uses the shared Hard-LMM gradient route"
            )
        super().__init__(
            hidden_dim=hidden_dim,
            train_log_mean=train_log_mean,
            max_seq_len=max_seq_len,
            memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
            **quantity_kwargs,
        )
        if self.encoder is None or len(self.encoder.layers) != 2:
            raise RuntimeError("Causal-QKV requires the two-layer local encoder")
        first_attention = self.encoder.layers[0].attn
        if not isinstance(first_attention, MemoryAttention):
            raise RuntimeError("Causal-QKV requires MemoryAttention in encoder block 1")
        self.encoder.layers[0].attn = CausalQKVMemoryAttention(first_attention)

    @property
    def causal_qkv_attention(self) -> CausalQKVMemoryAttention:
        if self.encoder is None:
            raise RuntimeError("Causal-QKV encoder is not initialized")
        attention = self.encoder.layers[0].attn
        if not isinstance(attention, CausalQKVMemoryAttention):
            raise RuntimeError("Causal-QKV first attention route is missing")
        return attention

    @property
    def additional_parameter_count(self) -> int:
        return 9 * self.hidden_dim

    @torch.no_grad()
    def reset_causal_qkv_identity(self) -> None:
        attention = self.causal_qkv_attention
        attention.causal_q_kernel.zero_()
        attention.causal_k_kernel.zero_()
        attention.causal_v_kernel.zero_()

    def load_hard_lmm_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Strictly initialize inherited tensors from a historical B state."""
        self.reset_causal_qkv_identity()
        incompatible = super().load_state_dict(state_dict, strict=False)
        if set(incompatible.missing_keys) != set(CAUSAL_QKV_KERNEL_KEYS):
            inherited_missing = sorted(
                set(incompatible.missing_keys) - set(CAUSAL_QKV_KERNEL_KEYS)
            )
            raise RuntimeError(
                "Hard-LMM B checkpoint is missing inherited parameters: "
                f"{inherited_missing}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Hard-LMM B checkpoint has unexpected parameters: "
                f"{sorted(incompatible.unexpected_keys)}"
            )


__all__ = [
    "CAUSAL_QKV_BACKBONE",
    "CAUSAL_QKV_CONTRACT_ID",
    "CAUSAL_QKV_KERNEL_KEYS",
    "CAUSAL_QKV_ROLE",
    "CausalQKVMemoryAttention",
    "CountAwareTitanCausalQKVTPP",
    "causal_qkv_metadata",
    "validate_causal_qkv_checkpoint",
]
