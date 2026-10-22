"""Component-isolated Shared15 ablations.

The full arm deliberately delegates to the frozen raw9 factory.  The other
arms are built from that same constructor path and replace only the requested
post-encoder component with a parameter-free identity/zero correction.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask
from models.TPPs.CountAwareTitanSharedHistoryMLP import (
    CountAwareTitanSharedHistoryMLP,
    _identity_matches,
    metadata as shared_metadata,
)

ARMS = ("full", "no_prototype", "no_history", "encoder_only")
_COMPONENT_ARMS = ARMS[1:]
CONTRACT_ID = "shared_component_ablation_v1"


def metadata(hidden_dim: int = 64, *, arm: str) -> dict[str, Any]:
    """Return an identity-bearing description of the actual retained modules."""
    if arm not in _COMPONENT_ARMS:
        raise ValueError("Component metadata is only defined for B--D arms")
    result = shared_metadata(hidden_dim, mode="shared").copy()
    prototype = arm in ("no_history",)
    history = arm in ("no_prototype",)
    result.update(
        candidate_name=arm,
        backbone_contract_id=CONTRACT_ID,
        model_role="shared15_component_ablation_" + arm,
        component_ablation_arm=arm,
        static_retrieval_enabled=prototype,
        prototype_component_enabled=prototype,
        history_correction_enabled=history,
        persistent_mem_size=16,
        lmm_mem_size=64 if prototype else 0,
        lmm_topk=4 if prototype else 0,
        memory_mode=("static_hard_retrieval_persistent_retained" if prototype
                     else "static_retrieval_removed_persistent_retained"),
        removed_static_parameters=0 if prototype else 4096,
        history_relation_sharing="shared" if history else "none",
        history_relation_width=15 if history else None,
        history_relation_branch_widths=[15] * 8 if history else [],
        history_relation_branch_count=8 if history else 0,
        history_relation_input_parameter_count=1920 if history else 0,
        history_relation_output_parameter_count=7680 if history else 0,
        history_relation_parameter_count=9600 if history else 0,
        shared_input_projection_count=1 if history else 0,
        independent_input_projection_count=0,
        output_projection_count=8 if history else 0,
        additional_parameter_count=9600 if history else 0,
        multilag_detail_rank=15 if history else None,
        multilag_detail_projections=("shared_current_previous_bias_free_input_branch_outputs"
                                    if history else "removed_zero_correction"),
        multilag_detail_initialization=("default_Linear_input_zero_output_forked_cpu_rng"
                                        if history else "not_present"),
        component_removed_parameters={
            "prototype": 0 if prototype else 4096,
            "history_correction": 0 if history else 9600,
        },
    )
    return result


def identity(hidden_dim: int = 64, *, arm: str) -> torch.Tensor:
    value = json.dumps(metadata(hidden_dim, arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(value).digest()), dtype=torch.uint8)


class HistoryZeroCorrection(nn.Module):
    """A shape- and observation-safe replacement with no trainable tensors."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, hidden, valid, *, memory_write_mask=None):
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError("History correction hidden state shape mismatch")
        _mask(valid, tuple(hidden.shape[:2]), "mask")
        observed = valid.to(hidden.device)
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(hidden.shape[:2]), "memory_write_mask")
            observed = observed & memory_write_mask.to(hidden.device)
        safe = torch.where(observed.unsqueeze(-1), hidden, torch.zeros_like(hidden))
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Observed history state must be finite")
        return torch.zeros_like(hidden)


class CountAwareComponentAblation(CountAwareTitanSharedHistoryMLP):
    """B--D only; construction begins with the exact shared full architecture."""
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, component_arm: str, **kwargs):
        if component_arm not in _COMPONENT_ARMS:
            raise ValueError("Component ablation arm must be no_prototype, no_history, or encoder_only")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, history_sharing_mode="shared", **kwargs)
        self.component_arm = component_arm
        if component_arm in ("no_prototype", "encoder_only"):
            self.lmm = nn.Identity()
        if component_arm in ("no_history", "encoder_only"):
            self.multilag_detail = HistoryZeroCorrection(hidden_dim)
        # These identify the inherited full configuration and must never make a
        # component checkpoint look loadable as a full/shared checkpoint.
        delattr(self, "shared_history_identity")
        delattr(self, "core_ablation_identity")
        self.register_buffer("component_ablation_identity", identity(hidden_dim, arm=component_arm))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = identity(self.core_hidden_dim, arm=self.component_arm)
        if not _identity_matches(state_dict.get("component_ablation_identity"), expected):
            raise ValueError("Component-ablation checkpoint identity mismatch")
        if "shared_history_identity" in state_dict or "core_ablation_identity" in state_dict:
            raise ValueError("Component checkpoint contains inherited full-arm identity")
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)


def validate_checkpoint(payload: Mapping[str, Any], expected_arm: str) -> bool:
    if expected_arm not in ARMS:
        raise ValueError("Unknown component ablation arm")
    if expected_arm == "full":
        from paper.scripts import run_titantpp_shared_history_raw9 as raw9
        return raw9.validate_checkpoint(payload, raw9.ARMS[0])
    if not isinstance(payload, Mapping) or payload.get("backbone") != expected_arm:
        raise ValueError("Component checkpoint arm mismatch")
    # Generic training checkpoints name this ``encoder_config`` and may append
    # the shared time-head mapping.  Accept the explicit alias only when both
    # descriptions agree, so a runner cannot relabel a foreign state.
    model_metadata = payload.get("model_metadata", payload.get("encoder_config"))
    expected_metadata = metadata(64, arm=expected_arm)
    if (not isinstance(model_metadata, Mapping)
            or any(model_metadata.get(key) != value for key, value in expected_metadata.items())):
        raise ValueError("Component checkpoint metadata mismatch")
    state = payload.get("state_dict", payload.get("model_state_dict"))
    expected_identity = identity(64, arm=expected_arm)
    if not isinstance(state, Mapping) or not _identity_matches(state.get("component_ablation_identity"), expected_identity):
        raise ValueError("Component checkpoint state identity mismatch")
    if "shared_history_identity" in state or "core_ablation_identity" in state:
        raise ValueError("Component checkpoint contains full-arm identity")
    return True


def build_model(data: Mapping[str, Any], arm: str):
    """Factory consumed by both trainer and profiler.

    `full` is intentionally a byte-for-byte delegation to raw9's approved
    factory.  B--D use the same input/configuration contract.
    """
    from paper.scripts import run_titantpp_shared_history_raw9 as raw9
    if arm not in ARMS:
        raise ValueError("Unknown component ablation arm")
    if arm == "full":
        return raw9.build_model(data, raw9.ARMS[0])
    config = {key: value for key, value in data["model"].items()
              if key not in ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    config.update(tail_threshold=46.0, tail_normalization_scale=46.0,
                  tail_clip_cap=187.0, tail_huber_delta=1.0)
    if config.pop("titans_memory_gradient_clip", None) is not None:
        raise ValueError("Memory gradient clipping must remain unchanged")
    from models.TPPs.CountAwareFactory import with_time_metadata
    model = CountAwareComponentAblation(
        **config, train_log_mean=data["statistics"]["train_log_mean"],
        train_log_std=data["statistics"]["train_log_std"],
        max_seq_len=data["loader"]["max_seq_len"], component_arm=arm,
    )
    return with_time_metadata(model, {**metadata(config["hidden_dim"], arm=arm),
        "max_len": data["loader"]["max_seq_len"], **raw9.input_identity()})
