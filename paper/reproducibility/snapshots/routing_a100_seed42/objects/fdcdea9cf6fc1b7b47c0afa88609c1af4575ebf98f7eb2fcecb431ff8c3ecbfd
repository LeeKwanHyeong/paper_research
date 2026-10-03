"""Hard-LMM backbone with shared-bank, memory-conditioned feature modulation.

The candidate keeps the count-aware Hard-LMM model's input projection,
persistent attention tokens, output heads, losses, and final prototype read.
It adds one structural path between the two encoder blocks: the first block's
state retrieves from the existing Hard-LMM bank and the retrieved residual
modulates each hidden feature before the second block.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD,
)


MEMORY_FILM_CONTRACT_ID = "hard_lmm_memory_film_v1"
MEMORY_FILM_BACKBONE = "titantpp_hard_memory_film"
MEMORY_FILM_ROLE = "hard_lmm_memory_film"


def memory_film_metadata(hidden_dim: int) -> dict[str, Any]:
    """Return the immutable routing identity expected in candidate artifacts."""
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("hidden_dim must be a positive integer")
    return {
        "candidate_name": "count_titan_hard_memory_film",
        "backbone_contract_id": MEMORY_FILM_CONTRACT_ID,
        "routing_contract_id": MEMORY_FILM_CONTRACT_ID,
        "model_role": MEMORY_FILM_ROLE,
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
        "feature_modulation": "bounded_elementwise_scale_shift",
        "feature_modulation_initialization": "exact_identity_zero_gains",
        "online_writes": False,
        "additional_parameter_count": 2 * hidden_dim,
    }


def validate_memory_film_checkpoint(
    payload: dict[str, Any],
    expected_backbone: str,
) -> bool:
    """Reject relabelled or incomplete Memory-FiLM checkpoint identities."""
    metadata = payload.get("encoder_config", {})
    states = [
        payload[name]
        for name in ("model_state_dict", "best_state_dict")
        if name in payload
    ]
    has_film_state = any(
        isinstance(state, dict)
        and (
            "film_scale_gain" in state
            or "film_shift_gain" in state
        )
        for state in states
    )
    is_candidate = (
        payload.get("backbone") == MEMORY_FILM_BACKBONE
        or (
            isinstance(metadata, dict)
            and (
                metadata.get("routing_contract_id") == MEMORY_FILM_CONTRACT_ID
                or metadata.get("model_role") == MEMORY_FILM_ROLE
            )
        )
        or has_film_state
    )
    if expected_backbone != MEMORY_FILM_BACKBONE:
        if is_candidate:
            raise ValueError("Memory-FiLM checkpoint cannot be relabelled")
        return False

    if not isinstance(metadata, dict):
        raise ValueError("Memory-FiLM checkpoint metadata must be a mapping")
    hidden_dim = metadata.get("d_model")
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("Memory-FiLM checkpoint hidden dimension is invalid")
    required = memory_film_metadata(hidden_dim)
    if payload.get("backbone") != MEMORY_FILM_BACKBONE or any(
        metadata.get(name) != value for name, value in required.items()
    ):
        raise ValueError("Memory-FiLM checkpoint routing metadata mismatch")

    time_head = metadata.get("time_head", {})
    if (
        payload.get("variant") != LOG_MSE_VARIANT
        or payload.get("evaluation_scope") != "validation_only"
        or payload.get("held_out_test_evaluated") is not False
        or not isinstance(time_head, dict)
        or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
    ):
        raise ValueError("Memory-FiLM checkpoint scope/head/objective mismatch")
    if not states:
        if (
            isinstance(payload.get("checkpoint_state_sha256"), str)
            and type(payload.get("best_epoch")) is int
        ):
            return True
        raise ValueError("Memory-FiLM artifact requires model state or summary binding")
    expected_shapes = {
        "lmm.mem": (1, 64, hidden_dim),
        "film_scale_gain": (hidden_dim,),
        "film_shift_gain": (hidden_dim,),
    }
    for state in states:
        if not isinstance(state, dict) or any(
            not isinstance(state.get(name), torch.Tensor)
            or tuple(state[name].shape) != shape
            or not bool(torch.isfinite(state[name]).all())
            for name, shape in expected_shapes.items()
        ):
            raise ValueError("Memory-FiLM checkpoint state is missing or invalid")
    return True


class CountAwareTitanMemoryFiLM(CountAwareTitanTPP):
    """Apply a bounded feature-wise FiLM transform between encoder blocks.

    For the first encoder-block output ``h`` and the existing Hard-LMM
    retrieval residual ``r``, the transform is

    ``h' = h * (1 + tanh(r * scale_gain)) + tanh(r * shift_gain)``.

    ``scale_gain`` and ``shift_gain`` contain one value per hidden feature and
    are zero initialized. Therefore a newly constructed candidate has exactly
    the same inherited parameters, random-number consumption, hidden states,
    and predictions as a same-seed Hard-LMM B model. The bounded modulation
    avoids introducing an unbounded multiplicative path.
    """

    contract_id = MEMORY_FILM_CONTRACT_ID
    backbone_name = MEMORY_FILM_BACKBONE

    def __init__(
        self,
        hidden_dim: int,
        train_log_mean: float,
        max_seq_len: int,
        **quantity_kwargs: Any,
    ) -> None:
        if "memory_mode" in quantity_kwargs:
            raise ValueError(
                "CountAwareTitanMemoryFiLM fixes memory_mode to static_hard_lmm"
            )
        if "quantity_memory_gradient_mode" in quantity_kwargs:
            raise ValueError(
                "CountAwareTitanMemoryFiLM uses the shared Hard-LMM gradient route"
            )
        super().__init__(
            hidden_dim=hidden_dim,
            train_log_mean=train_log_mean,
            max_seq_len=max_seq_len,
            memory_mode=TITAN_MEMORY_MODE_STATIC_HARD,
            **quantity_kwargs,
        )
        if self.encoder is None or len(self.encoder.layers) != 2:
            raise RuntimeError("Memory-FiLM requires the two-layer local encoder")
        if self.lmm is None:
            raise RuntimeError("Memory-FiLM requires the final Hard-LMM bank")

        # Zero tensors do not consume random numbers, preserving the B
        # initialization stream. Two vectors add only 2 * hidden_dim scalars.
        self.film_scale_gain = nn.Parameter(torch.zeros(self.hidden_dim))
        self.film_shift_gain = nn.Parameter(torch.zeros(self.hidden_dim))

    @property
    def additional_parameter_count(self) -> int:
        return 2 * self.hidden_dim

    @torch.no_grad()
    def reset_film_identity(self) -> None:
        """Reset only the candidate path to the exact Hard-LMM identity."""
        self.film_scale_gain.zero_()
        self.film_shift_gain.zero_()

    def load_hard_lmm_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Load a B state while requiring every inherited tensor to match.

        Candidate checkpoints should use ordinary strict ``load_state_dict``.
        This helper is solely for initializing the candidate from a historical
        Hard-LMM B checkpoint that predates the two FiLM parameters.
        """
        self.reset_film_identity()
        incompatible = super().load_state_dict(state_dict, strict=False)
        expected_missing = {"film_scale_gain", "film_shift_gain"}
        if set(incompatible.missing_keys) != expected_missing:
            raise RuntimeError(
                "Hard-LMM B checkpoint is missing inherited parameters: "
                f"{sorted(set(incompatible.missing_keys) - expected_missing)}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Hard-LMM B checkpoint has unexpected parameters: "
                f"{sorted(incompatible.unexpected_keys)}"
            )

    def apply_memory_film(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Retrieve from the shared bank and apply the feature-wise transform."""
        if self.lmm is None:
            raise RuntimeError("Memory-FiLM requires the Hard-LMM bank")
        if hidden.ndim != 3 or hidden.size(-1) != self.hidden_dim:
            raise ValueError("hidden must have shape [batch, length, hidden_dim]")
        if mask.shape != hidden.shape[:2]:
            raise ValueError("mask must have shape [batch, length]")

        valid_bool = mask.to(device=hidden.device, dtype=torch.bool)
        valid = valid_bool.unsqueeze(-1).to(dtype=hidden.dtype)
        retrieval, trace = self.lmm.retrieve(hidden)
        retrieval = retrieval * valid
        scale_delta = torch.tanh(retrieval * self.film_scale_gain)
        shift = torch.tanh(retrieval * self.film_shift_gain)
        modulated = (hidden * (1.0 + scale_delta) + shift) * valid

        prototype_indices = trace["prototype_indices"].masked_fill(
            ~valid_bool.unsqueeze(-1),
            -1,
        )
        topk_similarity = trace["topk_similarity"] * valid
        return modulated, {
            "prototype_indices": prototype_indices,
            "topk_similarity": topk_similarity,
            "retrieval": retrieval,
            "scale_delta": scale_delta,
            "shift": shift,
        }

    def _encode_base(
        self,
        dts: torch.Tensor,
        history_quantities: torch.Tensor,
        mask: torch.Tensor,
        *,
        memory_write_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # This static candidate has no online writes. Accepting the argument
        # preserves the shared target_outputs interface; it cannot expose a
        # future target to either retrieval or the encoder.
        del memory_write_mask
        if self.encoder is None:
            raise RuntimeError("Memory-FiLM local encoder is not initialized")

        features = self.continuous_features(dts, history_quantities, mask)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(
                encoded.size(1),
                encoded.device,
                encoded.dtype,
            )
        valid = mask.to(device=encoded.device, dtype=encoded.dtype).unsqueeze(-1)
        encoded = encoded * valid

        encoded = self.encoder.layers[0](encoded, mask=mask)
        encoded, _ = self.apply_memory_film(encoded, mask)
        encoded = self.encoder.layers[1](encoded, mask=mask)
        return encoded * valid


__all__ = [
    "CountAwareTitanMemoryFiLM",
    "MEMORY_FILM_BACKBONE",
    "MEMORY_FILM_CONTRACT_ID",
    "MEMORY_FILM_ROLE",
    "memory_film_metadata",
    "validate_memory_film_checkpoint",
]
