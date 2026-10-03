"""Value-norm-consistent static Hard-LMM candidate.

The original cosine top-k search and arithmetic-mean residual remain the base
route.  The candidate computes a second residual from the same selected rows
after equalizing every selected value to the bank-wide mean row L2 norm.  A
single zero-initialized bounded signed contrast combines both residuals, so the
inherited Hard-LMM is the exact initial model.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
from models.Titan.common.memory import HardLocalMemoryMatcher


VALUE_NORM_BACKBONE = "titantpp_hard_memory_value_norm"
VALUE_NORM_ROLE = "hard_lmm_value_norm_candidate"
VALUE_NORM_CONTRACT_ID = "hard_lmm_value_norm_consistent_v1"
VALUE_NORM_ALPHA_KEY = "lmm.alpha_raw"
VALUE_NORM_EPSILON = 1e-8
VALUE_NORM_FOREIGN_STATE_KEYS = (
    "encoder.elapsed_age_beta",
    "lmm.memory_keys",
    "interlayer_alpha_raw",
    "film_scale_gain",
    "film_shift_gain",
    "encoder.layers.0.attn.causal_q_kernel",
    "encoder.layers.0.attn.causal_k_kernel",
    "encoder.layers.0.attn.causal_v_kernel",
    "encoder.layers.0.attn.bounded_q_kernel",
    "encoder.layers.0.attn.bounded_k_kernel",
    "encoder.layers.0.attn.bounded_v_kernel",
    "encoder.layers.0.attn.level_history_q_kernel",
    "encoder.layers.0.attn.level_history_k_kernel",
    "encoder.layers.0.attn.level_history_v_kernel",
)
VALUE_NORM_REQUIRED_STATE_KEYS = (
    "b_t",
    "w_raw",
    "v_t.weight",
    "quantity_head.weight",
    "quantity_head.bias",
    "encoder.pos_emb",
    "encoder.input_proj.weight",
    "encoder.input_proj.bias",
    "encoder.layers.0.norm1.weight",
    "encoder.layers.0.norm1.bias",
    "encoder.layers.0.attn.persistent_mem",
    "encoder.layers.0.attn.qkv.weight",
    "encoder.layers.0.attn.qkv.bias",
    "encoder.layers.0.attn.out_proj.weight",
    "encoder.layers.0.attn.out_proj.bias",
    "encoder.layers.0.norm2.weight",
    "encoder.layers.0.norm2.bias",
    "encoder.layers.0.ff.0.weight",
    "encoder.layers.0.ff.0.bias",
    "encoder.layers.0.ff.3.weight",
    "encoder.layers.0.ff.3.bias",
    "encoder.layers.1.norm1.weight",
    "encoder.layers.1.norm1.bias",
    "encoder.layers.1.attn.persistent_mem",
    "encoder.layers.1.attn.qkv.weight",
    "encoder.layers.1.attn.qkv.bias",
    "encoder.layers.1.attn.out_proj.weight",
    "encoder.layers.1.attn.out_proj.bias",
    "encoder.layers.1.norm2.weight",
    "encoder.layers.1.norm2.bias",
    "encoder.layers.1.ff.0.weight",
    "encoder.layers.1.ff.0.bias",
    "encoder.layers.1.ff.3.weight",
    "encoder.layers.1.ff.3.bias",
    "lmm.mem",
    VALUE_NORM_ALPHA_KEY,
)


def value_norm_metadata(hidden_dim: int) -> dict[str, Any]:
    """Return the immutable identity and routing contract for this candidate."""
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    return {
        "candidate_name": "count_titan_hard_lmm_value_norm",
        "backbone_contract_id": VALUE_NORM_CONTRACT_ID,
        "routing_contract_id": VALUE_NORM_CONTRACT_ID,
        "model_role": VALUE_NORM_ROLE,
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
        "value_norm_target": "mean_l2_norm_over_all_bank_rows",
        "value_norm_scope": (
            "selected_values_with_detached_full_bank_scale_reference"
        ),
        "value_norm_epsilon": VALUE_NORM_EPSILON,
        "value_norm_zero_row_policy": (
            "norm_le_epsilon_zero_output_and_normalization_gradient"
        ),
        "value_norm_reference_detached": True,
        "value_norm_norm_compute_dtype": (
            "fp32_for_fp16_bfloat16_fp32_and_fp64_for_fp64"
        ),
        "value_norm_norm_algorithm": "max_abs_scaled_l2",
        "value_norm_combination": "r_base+tanh(alpha_raw)*(r_norm-r_base)",
        "value_norm_initialization": "alpha_raw_zero_exact_base_identity",
        "shared_search_indices_and_scores": True,
        "shared_base_and_alternative_bank": True,
        "new_parameter_state_keys": [VALUE_NORM_ALPHA_KEY],
        "additional_parameter_count": 1,
        "online_writes": False,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def validate_value_norm_checkpoint(
    payload: dict[str, Any], expected_backbone: str,
) -> bool:
    """Reject relabelled and malformed value-normalized Hard-LMM state."""
    metadata = payload.get("encoder_config", {})
    states = [
        payload[name]
        for name in ("model_state_dict", "best_state_dict")
        if name in payload
    ]
    has_candidate_state = any(
        isinstance(state, dict) and VALUE_NORM_ALPHA_KEY in state
        for state in states
    )
    is_candidate = (
        payload.get("backbone") == VALUE_NORM_BACKBONE
        or (
            isinstance(metadata, dict)
            and (
                metadata.get("routing_contract_id") == VALUE_NORM_CONTRACT_ID
                or metadata.get("model_role") == VALUE_NORM_ROLE
            )
        )
        or has_candidate_state
    )
    if expected_backbone != VALUE_NORM_BACKBONE:
        if is_candidate:
            raise ValueError("Value-norm checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, dict):
        raise ValueError("Value-norm checkpoint metadata must be a mapping")
    hidden_dim = metadata.get("d_model")
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("Value-norm checkpoint hidden dimension is invalid")
    if payload.get("backbone") != VALUE_NORM_BACKBONE or any(
        metadata.get(name) != value
        for name, value in value_norm_metadata(hidden_dim).items()
    ):
        raise ValueError("Value-norm checkpoint routing metadata mismatch")
    time_head = metadata.get("time_head", {})
    if (
        payload.get("variant") != LOG_MSE_VARIANT
        or payload.get("evaluation_scope") != "validation_only"
        or payload.get("held_out_test_evaluated") is not False
        or not isinstance(time_head, dict)
        or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
    ):
        raise ValueError("Value-norm checkpoint scope/head/objective mismatch")
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
            "Value-norm summary requires epoch and canonical state digest"
        )
    for state in states:
        if not isinstance(state, dict) or any(
            name in state for name in VALUE_NORM_FOREIGN_STATE_KEYS
        ):
            raise ValueError("Value-norm checkpoint contains a foreign candidate route")
        if set(state) != set(VALUE_NORM_REQUIRED_STATE_KEYS):
            raise ValueError("Value-norm checkpoint state key population is invalid")
        if any(
            not isinstance(tensor, torch.Tensor)
            or not bool(torch.isfinite(tensor).all())
            for tensor in state.values()
        ):
            raise ValueError("Value-norm checkpoint contains an invalid tensor")
        alpha = state.get(VALUE_NORM_ALPHA_KEY)
        memory = state.get("lmm.mem")
        if (
            not isinstance(alpha, torch.Tensor)
            or tuple(alpha.shape) != ()
            or not alpha.is_floating_point()
            or not bool(torch.isfinite(alpha).all())
            or not isinstance(memory, torch.Tensor)
            or tuple(memory.shape) != (1, 64, hidden_dim)
            or not bool(torch.isfinite(memory).all())
        ):
            raise ValueError("Value-norm checkpoint state is missing or invalid")
    return True


class ValueNormHardLocalMemoryMatcher(HardLocalMemoryMatcher):
    """Hard top-k matcher with an identity-gated signed value-norm contrast."""

    def __init__(self, base: HardLocalMemoryMatcher) -> None:
        if type(base) is not HardLocalMemoryMatcher:
            raise TypeError("value normalization requires the original Hard-LMM matcher")
        # Avoid the parent constructor because it would allocate a new random
        # bank.  Register the exact same Parameter object under the same key.
        nn.Module.__init__(self)
        self.d_model = base.d_model
        self.mem_size = base.mem_size
        self.topk = base.topk
        self.register_parameter("mem", base.mem)
        self.alpha_raw = nn.Parameter(torch.zeros(()))

    @property
    def alpha(self) -> torch.Tensor:
        """Return the bounded interpolation coefficient."""
        return torch.tanh(self.alpha_raw)

    @staticmethod
    def _expanded_memory(
        encoded: torch.Tensor, memory: torch.Tensor,
    ) -> torch.Tensor:
        if memory.dim() == 2:
            memory = memory.unsqueeze(0)
        if memory.dim() != 3:
            raise ValueError("memory must have shape [M, D], [1, M, D], or [B, M, D]")
        if memory.size(0) == 1:
            memory = memory.expand(encoded.size(0), -1, -1)
        if memory.size(0) != encoded.size(0) or memory.size(2) != encoded.size(2):
            raise ValueError("memory must have shape [M, D], [1, M, D], or [B, M, D]")
        return memory

    def retrieve_components(
        self,
        encoded: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return base/normalized residuals from one unchanged search trace."""
        base_residual, trace = super().retrieve(encoded, memory=memory)
        source = self.mem if memory is None else memory
        indices = trace["prototype_indices"]
        if source is None or source.numel() == 0 or indices.size(-1) == 0:
            return base_residual, base_residual, trace

        source = self._expanded_memory(encoded, source)
        expanded = source.unsqueeze(1).expand(-1, encoded.size(1), -1, -1)
        selected = torch.gather(
            expanded,
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, encoded.size(-1)),
        )
        normalized_selected = self._normalize_selected_values(selected, source)
        normalized_residual = normalized_selected.mean(dim=2).to(
            dtype=base_residual.dtype
        )
        return base_residual, normalized_residual, trace

    @staticmethod
    def _normalize_selected_values(
        selected: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """Equalize valid selected rows; tiny rows stay exact zero with zero grad."""
        norm_dtype = (
            torch.float64 if selected.dtype == torch.float64 else torch.float32
        )
        selected_for_norm = selected.to(dtype=norm_dtype)
        source_for_norm = source.to(dtype=norm_dtype)
        selected_norm = ValueNormHardLocalMemoryMatcher._stable_row_l2(
            selected_for_norm
        )
        bank_mean_norm = (
            ValueNormHardLocalMemoryMatcher._stable_row_l2(source_for_norm)
            .squeeze(-1)
            .mean(dim=1)
            .view(source.size(0), 1, 1, 1)
            .detach()
        )
        valid = selected_norm > VALUE_NORM_EPSILON
        safe_denominator = torch.where(
            valid, selected_norm, torch.ones_like(selected_norm)
        )
        scaled = selected_for_norm * (bank_mean_norm / safe_denominator)
        return torch.where(valid, scaled, torch.zeros_like(scaled))

    @staticmethod
    def _stable_row_l2(values: torch.Tensor) -> torch.Tensor:
        """Compute row L2 without overflowing on representable large norms."""
        max_abs = values.abs().amax(dim=-1, keepdim=True).detach()
        safe_scale = torch.where(
            max_abs > 0, max_abs, torch.ones_like(max_abs)
        )
        unit_norm = torch.linalg.vector_norm(
            values / safe_scale, ord=2, dim=-1, keepdim=True
        )
        return max_abs * unit_norm

    def retrieve(
        self,
        encoded: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        base_residual, normalized_residual, trace = self.retrieve_components(
            encoded, memory=memory
        )
        alpha = self.alpha.to(device=base_residual.device, dtype=base_residual.dtype)
        residual = base_residual + alpha * (normalized_residual - base_residual)
        return residual, trace

    def forward(
        self,
        encoded: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual, _ = self.retrieve(encoded, memory=memory)
        return encoded + residual


class CountAwareTitanValueNormTPP(CountAwareTitanTPP):
    """Common Hard-LMM with only the selected-value geometry adapted."""

    contract_id = VALUE_NORM_CONTRACT_ID
    backbone_name = VALUE_NORM_BACKBONE

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        max_seq_len: int,
        **quantity_kwargs: Any,
    ) -> None:
        if "memory_mode" in quantity_kwargs:
            raise ValueError("Value-norm Hard-LMM fixes memory_mode to static_hard_lmm")
        if "quantity_memory_gradient_mode" in quantity_kwargs:
            raise ValueError("Value-norm Hard-LMM uses the shared quantity gradient route")
        super().__init__(
            hidden_dim=hidden_dim,
            train_log_mean=train_log_mean,
            max_seq_len=max_seq_len,
            memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
            **quantity_kwargs,
        )
        if self.lmm is None or type(self.lmm) is not HardLocalMemoryMatcher:
            raise RuntimeError("Value-norm candidate requires the original Hard-LMM")
        self.lmm = ValueNormHardLocalMemoryMatcher(self.lmm)

    @property
    def value_norm_matcher(self) -> ValueNormHardLocalMemoryMatcher:
        if not isinstance(self.lmm, ValueNormHardLocalMemoryMatcher):
            raise RuntimeError("Value-norm Hard-LMM route is missing")
        return self.lmm

    @property
    def additional_parameter_count(self) -> int:
        return 1

    @torch.no_grad()
    def reset_value_norm_identity(self) -> None:
        self.value_norm_matcher.alpha_raw.zero_()

    def load_hard_lmm_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        """Strictly initialize from B, allowing only the new scalar to be absent."""
        expected = {
            name: tensor
            for name, tensor in self.state_dict().items()
            if name != VALUE_NORM_ALPHA_KEY
        }
        missing = set(expected) - set(state_dict)
        unexpected = set(state_dict) - set(expected)
        if missing:
            raise RuntimeError(
                f"Hard-LMM B checkpoint is missing inherited parameters: {sorted(missing)}"
            )
        if unexpected:
            raise RuntimeError(
                f"Hard-LMM B checkpoint has unexpected parameters: {sorted(unexpected)}"
            )
        for name, tensor in state_dict.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype
                or not bool(torch.isfinite(tensor).all())
            ):
                raise RuntimeError(f"Hard-LMM B checkpoint has an invalid tensor: {name}")
        complete = dict(state_dict)
        complete[VALUE_NORM_ALPHA_KEY] = torch.zeros_like(
            self.state_dict()[VALUE_NORM_ALPHA_KEY]
        )
        super().load_state_dict(complete, strict=True)


__all__ = [
    "CountAwareTitanValueNormTPP",
    "VALUE_NORM_ALPHA_KEY",
    "VALUE_NORM_BACKBONE",
    "VALUE_NORM_CONTRACT_ID",
    "VALUE_NORM_EPSILON",
    "VALUE_NORM_FOREIGN_STATE_KEYS",
    "VALUE_NORM_REQUIRED_STATE_KEYS",
    "VALUE_NORM_ROLE",
    "ValueNormHardLocalMemoryMatcher",
    "validate_value_norm_checkpoint",
    "value_norm_metadata",
]
