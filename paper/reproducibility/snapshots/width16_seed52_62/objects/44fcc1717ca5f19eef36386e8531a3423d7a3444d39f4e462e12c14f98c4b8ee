"""Project THP plus one static prototype bank, not Titans neural memory."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from models.TPPs.CountAwareTPP import (
    CountAwareTHP,
    LOG_MSE_VARIANT,
    TIME_HEAD_MODE_LEGACY_CLAMPED,
)
from models.TPPs.config import THPConfig
from models.Titan.common.memory import HardLocalMemoryMatcher


THP_STATIC_MEMORY_BACKBONE = "thp_static_hard_memory"
THP_STATIC_MEMORY_ROLE = "t0_thp_static_hard_memory"
THP_STATIC_MEMORY_CONTRACT = "count_aware_thp_static_hard_memory_v1"
THP_STATIC_MEMORY_CONTRACT_SHA256 = "77ed967b5be633095219b825a117b174b162f84cc4122c50410fb7d1eb2a9158"


def static_memory_metadata() -> dict[str, Any]:
    return {
        "candidate_name": "count_thp_static_hard_memory",
        **asdict(THPConfig(d_model=64, d_inner=256, n_layers=2, n_heads=4,
                          dropout=.1, normalize_before=False,
                          add_temporal_encoding_each_layer=False, use_rnn=False, d_rnn=64)),
        "routing_contract_id": THP_STATIC_MEMORY_CONTRACT,
        "routing_contract_sha256": THP_STATIC_MEMORY_CONTRACT_SHA256,
        "model_role": THP_STATIC_MEMORY_ROLE,
        "base_encoder": "CountAwareTHP",
        "memory_mode": "thp_static_hard_lmm",
        "lmm_mem_size": 64,
        "lmm_topk": 4,
        "persistent_mem_size": 0,
        "static_retrieval_aggregation": "arithmetic_mean",
        "static_retrieval_values": "unnormalized_prototypes",
        "memory_placement": "after_final_thp_layer",
        "time_memory_route": "thp_plus_static_hard_memory",
        "quantity_memory_route": "thp_plus_static_hard_memory",
        "mask_after_residual": True,
        "online_writes": False,
        "additional_parameter_count": 4096,
        "parameter_count": 104387,
    }


class CountAwareTHPStaticMemory(CountAwareTHP):
    """Preserve the complete THP computation and add only the original matcher."""

    def __init__(self, hidden_dim: int, train_log_mean: float, **quantity_kwargs: Any) -> None:
        expected = {
            "quantity_variant": LOG_MSE_VARIANT,
            "lambda_tail": 0.,
            "time_head_mode": TIME_HEAD_MODE_LEGACY_CLAMPED,
            "time_scale": 3.,
            "time_w_max": 10. / 3.,
            "time_intercept_limit": 30.,
            "time_wd_safety_limit": 40.,
        }
        if hidden_dim != 64 or any(quantity_kwargs.get(key, value) != value
                                   for key, value in expected.items()):
            raise ValueError("THP static memory requires frozen h64, direct log-MSE and legacy head")
        if quantity_kwargs.get("time_initial_intercept") not in (None, 0.):
            raise ValueError("THP static memory requires the original zero time intercept")
        super().__init__(hidden_dim, train_log_mean, **quantity_kwargs)
        # Adding the bank must not advance the caller's dropout/initialization RNG.
        with torch.random.fork_rng(devices=[]):
            self.lmm = HardLocalMemoryMatcher(hidden_dim, mem_size=64, topk=4)

    def encode(self, dts: torch.Tensor, history_quantities: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        base = super().encode(dts, history_quantities, mask)
        return self.lmm(base) * mask.unsqueeze(-1).to(dtype=base.dtype)


def validate_static_memory_checkpoint(payload: dict[str, Any], expected_backbone: str) -> None:
    """Reject relabelled, incomplete or incompatible candidate checkpoint identities."""
    metadata = payload.get("encoder_config", {})
    is_candidate = payload.get("backbone") == THP_STATIC_MEMORY_BACKBONE or (
        isinstance(metadata, dict) and (
            metadata.get("routing_contract_id") == THP_STATIC_MEMORY_CONTRACT
            or metadata.get("memory_mode") == "thp_static_hard_lmm"
        )
    )
    if expected_backbone != THP_STATIC_MEMORY_BACKBONE:
        if is_candidate:
            raise ValueError("THP static memory checkpoint cannot be relabelled")
        return
    if (not isinstance(metadata, dict) or payload.get("backbone") != expected_backbone
            or any(metadata.get(key) != value for key, value in static_memory_metadata().items())):
        raise ValueError("THP static memory checkpoint routing metadata mismatch")
    time_head = metadata.get("time_head", {})
    if (payload.get("variant") != LOG_MSE_VARIANT
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or not isinstance(time_head, dict)
            or any(time_head.get(key) != value for key, value in {
                "mode": TIME_HEAD_MODE_LEGACY_CLAMPED, "time_scale": 3.,
                "time_w_max": 10. / 3., "time_intercept_limit": 30.,
                "jacobian_correction": False, "wd_clamp": 10.,
            }.items())):
        raise ValueError("THP static memory checkpoint scope/head/objective mismatch")
    for state_key in ("model_state_dict", "best_state_dict"):
        if state_key in payload:
            state = payload[state_key]
            bank = state.get("lmm.mem") if isinstance(state, dict) else None
            if (not isinstance(bank, torch.Tensor) or tuple(bank.shape) != (1, 64, 64)
                    or not bool(torch.isfinite(bank).all())):
                raise ValueError("THP static memory checkpoint bank is missing or invalid")
