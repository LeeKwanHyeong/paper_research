"""Matched late paths on MLP16, prepared locally without changing its backbone.

Both arms retain the parent encoder, memory and heads.  Only the source of the
additional post-encoder, pre-static-memory residual differs.
"""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTitanHistoryWidth import (
    CountAwareTitanHistoryWidth,
    metadata as parent_metadata,
)
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask


MODE_BY_ARM = {
    "titantpp_mlp16_late_observation_v1": "observation",
    "titantpp_mlp16_late_hidden_v1": "hidden",
}
ARMS = tuple(MODE_BY_ARM)
RANK = 8
WIDTH = 16
CONTRACT_ID = "titantpp_mlp16_late_path_v1"


def metadata(hidden_dim=64, *, mode="observation"):
    """Return the fixed structural identity, including the residual's source."""
    if type(hidden_dim) is not int or hidden_dim != 64:
        raise ValueError("Late paths fix hidden_dim=64")
    if not isinstance(mode, str) or mode not in MODE_BY_ARM.values():
        raise ValueError("Late path mode must be observation or hidden")
    result = parent_metadata(hidden_dim, width=WIDTH)
    arm = next(arm for arm, arm_mode in MODE_BY_ARM.items() if arm_mode == mode)
    parent_count = result["additional_parameter_count"]
    result.update(
        candidate_name=arm,
        backbone_contract_id=CONTRACT_ID,
        model_role=f"mlp16_late_{mode}_path_v1",
        parent_backbone="titantpp_history_mlp_width16",
        late_path_mode=mode,
        late_path_rank=RANK,
        late_path_bias=False,
        late_path_activation="gelu_exact_erf",
        late_path_source=("pre_position_input_projection" if mode == "observation"
                          else "post_encoder2_hidden"),
        late_path_insertion="after_encoder2_before_existing_static_hard_lmm",
        late_path_mask="valid_and_memory_write_mask_when_supplied",
        late_path_initialization="default_Linear_input_zero_output_forked_cpu_rng",
        late_path_parameter_count=2 * hidden_dim * RANK,
        parent_additional_parameter_count=parent_count,
        additional_parameter_count=parent_count + 2 * hidden_dim * RANK,
    )
    return result


def identity(hidden_dim=64, *, mode="observation"):
    encoded = json.dumps(metadata(hidden_dim, mode=mode), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


class LatePath(nn.Module):
    """A bias-free 64 -> 8 -> 64 exact-GELU residual, initially zero."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim != 64:
            raise ValueError("Late paths fix hidden_dim=64")
        self.hidden_dim = hidden_dim
        # Neither the new tensors nor the discarded output initialization
        # consume the RNG used by the common model or subsequent callers.
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Linear(hidden_dim, RANK, bias=False)
            self.output_projection = nn.Linear(RANK, hidden_dim, bias=False)
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, source, observed):
        if (not isinstance(source, torch.Tensor) or source.ndim != 3
                or source.shape[-1] != self.hidden_dim):
            raise ValueError("Late path source must have shape [batch,length,64]")
        _mask(observed, tuple(source.shape[:2]), "observed")
        observed = observed.to(source.device)
        safe = torch.where(observed.unsqueeze(-1), source, 0.0)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed late path source")
        value = self.output_projection(
            F.gelu(self.input_projection(safe), approximate="none"))
        return torch.where(observed.unsqueeze(-1), value, 0.0)


class CountAwareTitanLateObservationPath(CountAwareTitanHistoryWidth):
    """Inject a matched residual after encoder2 and before inherited Hard-LMM."""

    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *,
                 late_path_mode="observation", **kwargs):
        metadata(hidden_dim, mode=late_path_mode)
        width = kwargs.pop("history_mlp_width", WIDTH)
        if type(width) is not int or width != WIDTH:
            raise ValueError("Late paths require history_mlp_width=16")
        super().__init__(hidden_dim, train_log_mean, max_seq_len,
                         history_mlp_width=WIDTH, **kwargs)
        self.late_path_mode = late_path_mode
        self.late_path = LatePath(hidden_dim)
        self.register_buffer("late_path_identity",
                             identity(hidden_dim, mode=late_path_mode))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get("late_path_identity")
        expected = identity(self.core_hidden_dim, mode=self.late_path_mode)
        if (not isinstance(value, torch.Tensor) or value.dtype != expected.dtype
                or not torch.equal(value.cpu(), expected)):
            raise ValueError("Late path checkpoint identity mismatch")
        # Reject absent, extra or malformed tensors before parent loading can
        # mutate any common weight, even when strict=False was requested.
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        hidden = super()._encode_base(
            dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        observed = mask.to(hidden.device)
        if memory_write_mask is not None:
            observed = observed & memory_write_mask.to(hidden.device)
        if self.late_path_mode == "observation":
            # This pure recomputation preserves the parent's operation order
            # and RNG while excluding absent/withheld nonfinite inputs before
            # log transforms and the existing input projection.
            safe_dts = torch.where(observed.to(dts.device), dts, 0.0)
            safe_quantities = torch.where(
                observed.to(history_quantities.device), history_quantities, 0.0)
            features = self.continuous_features(safe_dts, safe_quantities, observed)
            source = self.encoder.input_proj(features)
        else:
            source = hidden
        return hidden + self.late_path(source, observed)
