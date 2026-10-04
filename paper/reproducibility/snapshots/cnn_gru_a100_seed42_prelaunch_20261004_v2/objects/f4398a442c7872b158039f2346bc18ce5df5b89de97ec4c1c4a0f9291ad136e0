"""Level-preserving causal transition residuals for Hard-LMM event Q/K/V.

The candidate changes only the first encoder block's event projections.  It
reads the two most recent valid adjacent transitions, scales the branch by a
parameter-free history-confidence factor, and bounds Q/K with a transition-
only per-head RMS.  All added parameters are exactly zero at initialization,
so the inherited Hard-LMM route is the exact initial model.
"""

from __future__ import annotations

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
from models.TPPs.CountAwareTitanBoundedQK import BOUNDED_QK_KERNEL_KEYS
from models.TPPs.CountAwareTitanCausalQKV import (
    CAUSAL_QKV_KERNEL_KEYS,
    CausalQKVMemoryAttention,
)
from models.Titan.common.memory import MemoryAttention


LEVEL_HISTORY_QKV_BACKBONE = "titantpp_hard_memory_level_history_qkv"
LEVEL_HISTORY_QKV_ROLE = "hard_lmm_level_history_qkv_candidate"
LEVEL_HISTORY_QKV_CONTRACT_ID = (
    "hard_lmm_level_preserving_history_confidence_residual_v1"
)
LEVEL_HISTORY_QKV_EPSILON = 1e-8
LEVEL_HISTORY_QKV_KERNEL_KEYS = (
    "encoder.layers.0.attn.level_history_q_kernel",
    "encoder.layers.0.attn.level_history_k_kernel",
    "encoder.layers.0.attn.level_history_v_kernel",
)
LEVEL_HISTORY_QKV_FOREIGN_STATE_KEYS = (
    "encoder.elapsed_age_beta",
    "film_scale_gain",
    "film_shift_gain",
    "interlayer_alpha_raw",
    "lmm.memory_keys",
)


def level_history_qkv_metadata(hidden_dim: int) -> dict[str, Any]:
    """Return the immutable model and routing identity for this candidate."""
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of four")
    return {
        "candidate_name": "count_titan_hard_lmm_level_history_qkv",
        "backbone_contract_id": LEVEL_HISTORY_QKV_CONTRACT_ID,
        "routing_contract_id": LEVEL_HISTORY_QKV_CONTRACT_ID,
        "model_role": LEVEL_HISTORY_QKV_ROLE,
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
        "level_history_layer": 0,
        "level_history_scope": "event_qkv_only",
        "level_history_transition_order": [
            "current_minus_lag1",
            "lag1_minus_lag2",
        ],
        "level_history_parameterization": "channelwise_diagonal",
        "level_history_confidence": (
            "cumulative_valid_adjacent_pairs_over_cumulative_valid_events"
        ),
        "level_history_confidence_trainable": False,
        "level_history_qk_bound": "headwise_transition_rms_saturation",
        "level_history_qk_epsilon": LEVEL_HISTORY_QKV_EPSILON,
        "level_history_v_operation": "unbounded_transition_residual",
        "level_history_initialization": "all_zero_exact_identity",
        "persistent_key_value_adaptation": False,
        "final_memory_placement": "after_encoder_layer_2",
        "new_parameter_state_keys": list(LEVEL_HISTORY_QKV_KERNEL_KEYS),
        "additional_parameter_count": 6 * hidden_dim,
        "online_writes": False,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_level_history_qkv_checkpoint(
    payload: dict[str, Any], expected_backbone: str,
) -> bool:
    """Reject route relabelling and malformed level-history checkpoint state."""
    metadata = payload.get("encoder_config", {})
    states = [
        payload[name]
        for name in ("model_state_dict", "best_state_dict")
        if name in payload
    ]
    has_candidate_state = any(
        isinstance(state, dict)
        and any(name in state for name in LEVEL_HISTORY_QKV_KERNEL_KEYS)
        for state in states
    )
    is_candidate = (
        payload.get("backbone") == LEVEL_HISTORY_QKV_BACKBONE
        or (
            isinstance(metadata, dict)
            and (
                metadata.get("routing_contract_id")
                == LEVEL_HISTORY_QKV_CONTRACT_ID
                or metadata.get("model_role") == LEVEL_HISTORY_QKV_ROLE
            )
        )
        or has_candidate_state
    )
    if expected_backbone != LEVEL_HISTORY_QKV_BACKBONE:
        if is_candidate:
            raise ValueError("Level-history QKV checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, dict):
        raise ValueError("Level-history QKV checkpoint metadata must be a mapping")
    hidden_dim = metadata.get("d_model")
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("Level-history QKV checkpoint hidden dimension is invalid")
    required = level_history_qkv_metadata(hidden_dim)
    if payload.get("backbone") != LEVEL_HISTORY_QKV_BACKBONE or any(
        metadata.get(name) != value for name, value in required.items()
    ):
        raise ValueError("Level-history QKV checkpoint routing metadata mismatch")
    time_head = metadata.get("time_head", {})
    if (
        payload.get("variant") != LOG_MSE_VARIANT
        or payload.get("evaluation_scope") != "validation_only"
        or payload.get("held_out_test_evaluated") is not False
        or not isinstance(time_head, dict)
        or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
    ):
        raise ValueError("Level-history QKV checkpoint scope/head/objective mismatch")
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
            "Level-history QKV summary requires epoch and canonical state digest"
        )
    forbidden_keys = (
        *CAUSAL_QKV_KERNEL_KEYS,
        *BOUNDED_QK_KERNEL_KEYS,
        *LEVEL_HISTORY_QKV_FOREIGN_STATE_KEYS,
    )
    for state in states:
        if not isinstance(state, dict) or any(
            not isinstance(state.get(name), torch.Tensor)
            or tuple(state[name].shape) != (2, hidden_dim)
            or not bool(torch.isfinite(state[name]).all())
            for name in LEVEL_HISTORY_QKV_KERNEL_KEYS
        ):
            raise ValueError(
                "Level-history QKV checkpoint kernels are missing or invalid"
            )
        if any(name in state for name in forbidden_keys):
            raise ValueError(
                "A FULL or BOUNDED-QK checkpoint cannot be relabelled as level-history QKV"
            )
        lmm = state.get("lmm.mem")
        if (
            not isinstance(lmm, torch.Tensor)
            or tuple(lmm.shape) != (1, 64, hidden_dim)
            or not bool(torch.isfinite(lmm).all())
        ):
            raise ValueError("Level-history QKV checkpoint Hard-LMM state is invalid")
    return True


class LevelHistoryQKVMemoryAttention(CausalQKVMemoryAttention):
    """Add level-free, history-confident transition residuals to event Q/K/V."""

    def __init__(self, base: MemoryAttention) -> None:
        super().__init__(base)
        for projection in ("q", "k", "v"):
            self._parameters.pop(f"causal_{projection}_kernel")
            parameter = nn.Parameter(base.qkv.weight.new_zeros((2, self.d_model)))
            self.register_parameter(f"level_history_{projection}_kernel", parameter)

    @staticmethod
    def transition_components(
        projection: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return two valid adjacent differences, availability, and confidence."""
        if projection.ndim != 3:
            raise ValueError("projection must have shape [batch, length, channels]")
        batch_size, length, _ = projection.shape
        if mask is None:
            valid = torch.ones(
                batch_size, length, dtype=torch.bool, device=projection.device
            )
        else:
            if mask.shape != (batch_size, length):
                raise ValueError(
                    f"Expected attention mask {(batch_size, length)}, got {tuple(mask.shape)}"
                )
            valid = mask.to(device=projection.device, dtype=torch.bool)

        safe_projection = torch.where(
            valid.unsqueeze(-1), projection, torch.zeros_like(projection)
        )
        previous = F.pad(safe_projection, (0, 0, 1, 0))[:, :length]
        previous_two = F.pad(safe_projection, (0, 0, 2, 0))[:, :length]
        previous_valid = F.pad(valid, (1, 0), value=False)[:, :length]
        previous_two_valid = F.pad(valid, (2, 0), value=False)[:, :length]
        lag1_available = valid & previous_valid
        lag2_available = valid & previous_valid & previous_two_valid
        lag1 = torch.where(
            lag1_available.unsqueeze(-1),
            safe_projection - previous,
            torch.zeros_like(projection),
        )
        lag2 = torch.where(
            lag2_available.unsqueeze(-1),
            previous - previous_two,
            torch.zeros_like(projection),
        )

        event_count = valid.cumsum(dim=1)
        transition_count = lag1_available.cumsum(dim=1)
        confidence = torch.where(
            event_count > 0,
            transition_count.to(dtype=projection.dtype)
            / event_count.clamp_min(1).to(dtype=projection.dtype),
            torch.zeros_like(event_count, dtype=projection.dtype),
        )
        return lag1, lag2, lag1_available, lag2_available, confidence

    @classmethod
    def transition_residual(
        cls,
        projection: torch.Tensor,
        kernel: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply the two diagonal transition coefficients without a level term."""
        if kernel.shape != (2, projection.size(-1)):
            raise ValueError("kernel must have shape [2, channels]")
        lag1, lag2, available1, available2, confidence = (
            cls.transition_components(projection, mask)
        )
        residual = lag1 * kernel[0] + lag2 * kernel[1]
        return residual, lag1, lag2, available1, available2, confidence

    def bound_transition_residual(
        self,
        residual: torch.Tensor,
        lag1: torch.Tensor,
        lag2: torch.Tensor,
        lag1_available: torch.Tensor,
        lag2_available: torch.Tensor,
    ) -> torch.Tensor:
        """Bound Q/K per head using only available transition RMS values."""
        if not (
            residual.shape == lag1.shape == lag2.shape
            and residual.ndim == 3
            and residual.shape[-1] == self.d_model
        ):
            raise ValueError("residual and transitions must share [B,L,D] shape")
        if lag1_available.shape != residual.shape[:2] or (
            lag2_available.shape != residual.shape[:2]
        ):
            raise ValueError("transition availability must have shape [B,L]")
        result_dtype = torch.result_type(residual, lag1)
        norm_dtype = (
            torch.float32
            if result_dtype in (torch.float16, torch.bfloat16)
            else result_dtype
        )
        head_shape = (*residual.shape[:-1], self.n_heads, self.head_dim)
        first = lag1.to(dtype=norm_dtype).reshape(head_shape)
        second = lag2.to(dtype=norm_dtype).reshape(head_shape)
        value = residual.to(dtype=norm_dtype).reshape(head_shape)
        first_available = lag1_available.to(dtype=norm_dtype)[..., None, None]
        second_available = lag2_available.to(dtype=norm_dtype)[..., None, None]
        available_count = (first_available + second_available).clamp_min(1.0)
        scale_squared = (
            first_available * first.square().mean(dim=-1, keepdim=True)
            + second_available * second.square().mean(dim=-1, keepdim=True)
        ) / available_count + LEVEL_HISTORY_QKV_EPSILON
        residual_squared = value.square().mean(dim=-1, keepdim=True)
        bounded = value * scale_squared.sqrt() / (
            scale_squared + residual_squared
        ).sqrt()
        return bounded.reshape_as(residual).to(dtype=result_dtype)

    def _adapt_event_projections(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        *,
        input_dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del input_dtype  # Mask validity, rather than its numeric dtype, defines transitions.
        adapted = []
        for kind, projection in (("q", q), ("k", k), ("v", v)):
            kernel = getattr(self, f"level_history_{kind}_kernel")
            residual, lag1, lag2, available1, available2, confidence = (
                self.transition_residual(projection, kernel, mask)
            )
            if kind in ("q", "k"):
                residual = self.bound_transition_residual(
                    residual, lag1, lag2, available1, available2
                )
            adapted.append(
                projection
                + residual * confidence.to(dtype=residual.dtype).unsqueeze(-1)
            )
        return adapted[0], adapted[1], adapted[2]


class CountAwareTitanLevelHistoryQKVTPP(CountAwareTitanTPP):
    """The common two-block Hard-LMM with an LPHC event-Q/K/V branch."""

    contract_id = LEVEL_HISTORY_QKV_CONTRACT_ID
    backbone_name = LEVEL_HISTORY_QKV_BACKBONE

    def __init__(
        self, hidden_dim: int, train_log_mean: float, max_seq_len: int,
        **quantity_kwargs: Any,
    ) -> None:
        if "memory_mode" in quantity_kwargs:
            raise ValueError(
                "CountAwareTitanLevelHistoryQKVTPP fixes memory_mode to static_hard_lmm"
            )
        if "quantity_memory_gradient_mode" in quantity_kwargs:
            raise ValueError(
                "CountAwareTitanLevelHistoryQKVTPP uses the shared Hard-LMM gradient route"
            )
        super().__init__(
            hidden_dim=hidden_dim,
            train_log_mean=train_log_mean,
            max_seq_len=max_seq_len,
            memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
            **quantity_kwargs,
        )
        if self.encoder is None or len(self.encoder.layers) != 2:
            raise RuntimeError("Level-history QKV requires the two-layer local encoder")
        first_attention = self.encoder.layers[0].attn
        if not isinstance(first_attention, MemoryAttention):
            raise RuntimeError(
                "Level-history QKV requires MemoryAttention in encoder block 1"
            )
        self.encoder.layers[0].attn = LevelHistoryQKVMemoryAttention(first_attention)

    @property
    def level_history_qkv_attention(self) -> LevelHistoryQKVMemoryAttention:
        if self.encoder is None:
            raise RuntimeError("Level-history QKV encoder is not initialized")
        attention = self.encoder.layers[0].attn
        if not isinstance(attention, LevelHistoryQKVMemoryAttention):
            raise RuntimeError("Level-history QKV first attention route is missing")
        return attention

    @property
    def additional_parameter_count(self) -> int:
        return 6 * self.hidden_dim

    @torch.no_grad()
    def reset_level_history_qkv_identity(self) -> None:
        attention = self.level_history_qkv_attention
        attention.level_history_q_kernel.zero_()
        attention.level_history_k_kernel.zero_()
        attention.level_history_v_kernel.zero_()

    def load_hard_lmm_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Strictly and atomically initialize inherited tensors from Hard-LMM B."""
        expected = {
            name: tensor
            for name, tensor in self.state_dict().items()
            if name not in LEVEL_HISTORY_QKV_KERNEL_KEYS
        }
        missing = set(expected) - set(state_dict)
        unexpected = set(state_dict) - set(expected)
        if missing:
            raise RuntimeError(
                "Hard-LMM B checkpoint is missing inherited parameters: "
                f"{sorted(missing)}"
            )
        if unexpected:
            raise RuntimeError(
                "Hard-LMM B checkpoint has unexpected parameters: "
                f"{sorted(unexpected)}"
            )
        for name, tensor in state_dict.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise RuntimeError(
                    f"Hard-LMM B checkpoint has an invalid tensor: {name}"
                )
        complete = dict(state_dict)
        complete.update(
            {
                name: torch.zeros_like(self.state_dict()[name])
                for name in LEVEL_HISTORY_QKV_KERNEL_KEYS
            }
        )
        super().load_state_dict(complete, strict=True)


__all__ = [
    "LEVEL_HISTORY_QKV_BACKBONE",
    "LEVEL_HISTORY_QKV_CONTRACT_ID",
    "LEVEL_HISTORY_QKV_EPSILON",
    "LEVEL_HISTORY_QKV_FOREIGN_STATE_KEYS",
    "LEVEL_HISTORY_QKV_KERNEL_KEYS",
    "LEVEL_HISTORY_QKV_ROLE",
    "CountAwareTitanLevelHistoryQKVTPP",
    "LevelHistoryQKVMemoryAttention",
    "level_history_qkv_metadata",
    "validate_level_history_qkv_checkpoint",
]
