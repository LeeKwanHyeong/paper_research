"""Inter-layer Hard-LMM candidate for the count-aware TitanTPP backbone.

The candidate keeps the original two-layer persistent-memory encoder and its
final Hard-LMM read.  It additionally reads the same prototype bank after the
first encoder layer and injects that residual before the second layer.  A
zero-initialized scalar gate makes the candidate exactly reduce to the
original static Hard-LMM route at initialization.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD,
)


INTERLAYER_MEMORY_CONTRACT_ID = "hard_lmm_interlayer_memory_v1"
INTERLAYER_MEMORY_BACKBONE = "titantpp_hard_memory_interlayer"
INTERLAYER_MEMORY_ROLE = "hard_lmm_interlayer_memory_candidate"


def interlayer_memory_metadata(hidden_dim: int) -> dict[str, Any]:
    """Return the immutable artifact identity for the inter-layer route."""
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    return {
        "candidate_name": "count_titan_hard_lmm_interlayer",
        "backbone_contract_id": INTERLAYER_MEMORY_CONTRACT_ID,
        "routing_contract_id": INTERLAYER_MEMORY_CONTRACT_ID,
        "model_role": INTERLAYER_MEMORY_ROLE,
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
        "static_key_value_tied": True,
        "shared_intermediate_final_bank": True,
        "intermediate_memory_placement": "after_encoder_layer_1",
        "final_memory_placement": "after_encoder_layer_2",
        "intermediate_combination": "bounded_scalar_gated_residual",
        "intermediate_initialization": "exact_identity_zero_gate",
        "online_writes": False,
        "additional_parameter_count": 1,
    }


def validate_interlayer_memory_checkpoint(
    payload: dict[str, Any], expected_backbone: str
) -> bool:
    """Reject relabelled or incomplete inter-layer candidate checkpoints."""
    metadata = payload.get("encoder_config", {})
    states = [
        payload[name]
        for name in ("model_state_dict", "best_state_dict")
        if name in payload
    ]
    has_state = any(
        isinstance(state, dict) and "interlayer_alpha_raw" in state
        for state in states
    )
    is_candidate = (
        payload.get("backbone") == INTERLAYER_MEMORY_BACKBONE
        or (
            isinstance(metadata, dict)
            and (
                metadata.get("routing_contract_id") == INTERLAYER_MEMORY_CONTRACT_ID
                or metadata.get("model_role") == INTERLAYER_MEMORY_ROLE
            )
        )
        or has_state
    )
    if expected_backbone != INTERLAYER_MEMORY_BACKBONE:
        if is_candidate:
            raise ValueError("Inter-layer checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, dict):
        raise ValueError("Inter-layer checkpoint metadata must be a mapping")
    hidden_dim = metadata.get("d_model")
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("Inter-layer checkpoint hidden dimension is invalid")
    if payload.get("backbone") != INTERLAYER_MEMORY_BACKBONE or any(
        metadata.get(name) != value
        for name, value in interlayer_memory_metadata(hidden_dim).items()
    ):
        raise ValueError("Inter-layer checkpoint routing metadata mismatch")
    time_head = metadata.get("time_head", {})
    if (
        payload.get("variant") != LOG_MSE_VARIANT
        or payload.get("evaluation_scope") != "validation_only"
        or payload.get("held_out_test_evaluated") is not False
        or not isinstance(time_head, dict)
        or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
    ):
        raise ValueError("Inter-layer checkpoint scope/head/objective mismatch")
    if not states:
        if (
            isinstance(payload.get("checkpoint_state_sha256"), str)
            and type(payload.get("best_epoch")) is int
        ):
            return True
        raise ValueError("Inter-layer artifact requires model state or summary binding")
    for state in states:
        if (
            not isinstance(state, dict)
            or not isinstance(state.get("interlayer_alpha_raw"), torch.Tensor)
            or tuple(state["interlayer_alpha_raw"].shape) != ()
            or not bool(torch.isfinite(state["interlayer_alpha_raw"]).all())
            or not isinstance(state.get("lmm.mem"), torch.Tensor)
            or tuple(state["lmm.mem"].shape) != (1, 64, hidden_dim)
            or not bool(torch.isfinite(state["lmm.mem"]).all())
        ):
            raise ValueError("Inter-layer checkpoint state is missing or invalid")
    return True


class CountAwareTitanInterLayerMemoryTPP(CountAwareTitanTPP):
    """Static Hard-LMM with a gated read between encoder layers.

    Only the inter-layer placement is new.  The candidate shares the original
    prototype bank with the final Hard-LMM read and inherits the original time
    head, quantity head, losses, and final memory combination unchanged.
    """

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        max_seq_len: int,
        *,
        interlayer_alpha_init: float = 0.0,
        **quantity_kwargs: Any,
    ) -> None:
        if "memory_mode" in quantity_kwargs:
            raise ValueError(
                "Inter-layer Hard-LMM fixes memory_mode to static_hard_lmm"
            )
        if not math.isfinite(interlayer_alpha_init):
            raise ValueError("interlayer_alpha_init must be finite")
        if not -1.0 < interlayer_alpha_init < 1.0:
            raise ValueError(
                "interlayer_alpha_init must lie strictly inside (-1, 1)"
            )

        super().__init__(
            hidden_dim=hidden_dim,
            train_log_mean=train_log_mean,
            max_seq_len=max_seq_len,
            memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
            **quantity_kwargs,
        )
        if self.encoder is None or len(self.encoder.layers) != 2:
            raise RuntimeError(
                "Inter-layer Hard-LMM requires the two-layer local Titan encoder"
            )
        if self.lmm is None:
            raise RuntimeError(
                "Inter-layer Hard-LMM requires the static prototype bank"
            )

        raw_alpha = math.atanh(float(interlayer_alpha_init))
        # Creating a scalar from a constant consumes no RNG, preserving the
        # original B initialization and the caller's random stream.
        self.interlayer_alpha_raw = nn.Parameter(torch.tensor(raw_alpha))

    @property
    def interlayer_alpha(self) -> torch.Tensor:
        """Return the bounded effective residual coefficient."""
        return torch.tanh(self.interlayer_alpha_raw)

    def _encode_base(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
        *,
        memory_write_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Static Hard-LMM has no online write state.  Keep the argument for the
        # common next-event interface and intentionally ignore it, as B does.
        del memory_write_mask
        if self.encoder is None or self.lmm is None:
            raise RuntimeError("Inter-layer Hard-LMM modules are not initialized")

        features = self.continuous_features(dts, history_quantities, mask)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(
                encoded.size(1),
                encoded.device,
                encoded.dtype,
            )

        valid = mask.to(device=encoded.device).unsqueeze(-1).to(encoded.dtype)
        encoded = encoded * valid

        encoded = self.encoder.layers[0](encoded, mask=mask)
        interlayer_residual, _ = self.lmm.retrieve(encoded)
        alpha = self.interlayer_alpha.to(dtype=encoded.dtype)
        encoded = (encoded + alpha * interlayer_residual) * valid
        encoded = self.encoder.layers[1](encoded, mask=mask)
        return encoded * valid


__all__ = [
    "CountAwareTitanInterLayerMemoryTPP",
    "INTERLAYER_MEMORY_BACKBONE",
    "INTERLAYER_MEMORY_CONTRACT_ID",
    "INTERLAYER_MEMORY_ROLE",
    "interlayer_memory_metadata",
    "validate_interlayer_memory_checkpoint",
]
