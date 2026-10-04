"""Signed causal elapsed-age geometry for the separate-key Hard-LMM encoder."""

from __future__ import annotations

import torch


ELAPSED_AGE_BACKBONE = "titantpp_elapsed_age_static_memory"
ELAPSED_AGE_ROLE = "t0_elapsed_age_static_retrieval"
ELAPSED_AGE_CONTRACT = "hard_lmm_elapsed_age_v1"
ELAPSED_AGE_MEMORY_MODE = "elapsed_age_static_lmm"
ELAPSED_AGE_BETA_KEY = "encoder.elapsed_age_beta"


def elapsed_age_metadata(hidden_dim: int) -> dict:
    """Candidate identity; the underlying sparse K/V retrieval is unchanged."""
    return {
        "candidate_name": "count_titan_elapsed_age_separate_keys",
        "backbone_contract_id": ELAPSED_AGE_CONTRACT,
        "elapsed_age_contract_id": ELAPSED_AGE_CONTRACT,
        "memory_mode": ELAPSED_AGE_MEMORY_MODE,
        "elapsed_age_geometry": "signed_causal_prefix_age_minus_rank",
        "elapsed_age_normalization": "query_prefix_internal_span",
        "elapsed_age_geometry_dtype": "float64_to_float32",
        "elapsed_age_beta_shape": [2, 4],
        "elapsed_age_beta_initialization": "zeros_no_rng",
        "elapsed_age_parameter_count": 8,
        "elapsed_age_first_gap": "excluded_from_geometry",
        "elapsed_age_persistent_key_bias": 0.0,
        "additional_parameter_count": 64 * hidden_dim + 8,
        "time_memory_route": "elapsed_age_encoder_then_separate_key_static_matcher",
        "quantity_memory_route": "elapsed_age_encoder_then_separate_key_static_matcher",
    }


def causal_elapsed_age_geometry(
    dts: torch.Tensor,
    mask: torch.Tensor,
    *,
    observed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [B,L,L] signed age deviation, using each query's own prefix.

    The first observed gap is a boundary gap and contributes no elapsed age.
    Nonnegative observed gaps are accumulated in float64 outside autocast; the
    bounded geometry is returned in float32. No target values are accepted.
    """
    if dts.ndim != 2 or dts.shape != mask.shape or not dts.numel():
        raise ValueError("Elapsed-age dts and mask require a nonempty [batch, length] shape")
    if mask.dtype != torch.bool or mask.device != dts.device:
        raise ValueError("Elapsed-age mask must be boolean and share the input device")
    observed = mask if observed_mask is None else observed_mask
    if observed.shape != mask.shape or observed.dtype != torch.bool or observed.device != dts.device:
        raise ValueError("Elapsed-age observed mask must match the input mask")
    if bool((observed & ~mask).any()):
        raise ValueError("Elapsed-age observed mask cannot include padding")
    if not bool(torch.isfinite(dts[observed]).all()):
        raise ValueError("Elapsed-age observed gaps must be finite")
    with torch.autocast(device_type=dts.device.type, enabled=False):
        rank = observed.long().cumsum(-1) - 1
        internal = observed & (rank > 0)
        gaps = torch.where(internal, dts.to(torch.float64).clamp_min(0), 0.0)
        span = gaps.cumsum(-1)
        if not bool(torch.isfinite(span).all()):
            raise ValueError("Elapsed-age cumulative span must be finite")
        denominator = torch.where(span > 0, span, 1.0)
        elapsed_fraction = (span.unsqueeze(-1) - span.unsqueeze(-2)) / denominator.unsqueeze(-1)
        rank_fraction = (rank.unsqueeze(-1) - rank.unsqueeze(-2)).to(torch.float64)
        rank_fraction = rank_fraction / rank.clamp_min(1).unsqueeze(-1)
        length = dts.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=dts.device).tril()
        valid = observed.unsqueeze(-1) & observed.unsqueeze(-2) & causal
        valid = valid & (span > 0).unsqueeze(-1) & (rank >= 2).unsqueeze(-1)
        geometry = torch.where(valid, elapsed_fraction - rank_fraction, 0.0)
        return geometry.to(torch.float32)
