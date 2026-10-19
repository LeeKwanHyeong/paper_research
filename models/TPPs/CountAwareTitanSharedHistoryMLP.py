"""Prefix-parameter-matched shared15 and independent history corrections.

Only the relation MLP between the two existing encoder blocks changes.  Both
arms retain the eight original eligibility thresholds, immediate predecessor,
exact GELU, bias-free projections and divisor eight.  Equal parameter counts
do not imply equal function classes: the shared arm has one width15 nonlinear
basis, whereas k eligible independent branches can span up to 15+5*(k-1)
outputs.  Independent widths (15,5,5,5,5,5,5,5) match the number of active
parameters at every eligible branch prefix, including RAF's shorter histories.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTitanCoreAblation import (
    CountAwareTitanCoreAblation, identity as core_identity,
    metadata as core_metadata,
)
from models.TPPs.CountAwareTitanMultiLagDetail import (
    AVAILABILITY_LAGS, _complete_state, _mask, lag_source_indices,
)


MODE_BY_ARM = {
    "titantpp_history_mlp_shared15_v1": "shared",
    "titantpp_history_mlp_prefix_capacity_v1": "independent",
}
ARMS = tuple(MODE_BY_ARM)
SHARED_WIDTH = 15
INDEPENDENT_WIDTHS = (15, 5, 5, 5, 5, 5, 5, 5)
CONTRACT_ID = "titantpp_shared_history_mlp_v1"
CORRECTION_PARAMETER_COUNT = 9600


def role_for_arm(arm):
    if not isinstance(arm, str) or arm not in MODE_BY_ARM:
        raise ValueError("Unknown shared-history MLP arm")
    return "observed_time_" + arm.removeprefix("titantpp_")


def metadata(hidden_dim=64, *, mode="shared"):
    if type(hidden_dim) is not int or hidden_dim != 64:
        raise ValueError("Shared-history MLP fixes hidden_dim=64")
    if not isinstance(mode, str) or mode not in MODE_BY_ARM.values():
        raise ValueError("History sharing mode must be shared or independent")
    widths = (SHARED_WIDTH,) * 8 if mode == "shared" else INDEPENDENT_WIDTHS
    arm = next(name for name, value in MODE_BY_ARM.items() if value == mode)
    result = core_metadata(hidden_dim, "mlp")
    result.update(
        candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), history_relation_sharing=mode,
        history_relation_width=SHARED_WIDTH if mode == "shared" else None,
        history_relation_branch_widths=list(widths), history_relation_branch_count=8,
        history_relation_input_parameter_count=(128 * SHARED_WIDTH if mode == "shared" else 128 * sum(widths)),
        history_relation_output_parameter_count=hidden_dim * sum(widths),
        history_relation_parameter_count=CORRECTION_PARAMETER_COUNT,
        parameter_matching="equal_active_trainable_count_at_every_branch_prefix_not_equal_function_capacity",
        active_prefix_parameter_counts=[0] + [hidden_dim * SHARED_WIDTH * (count + 2) for count in range(1, 9)],
        shared_input_projection_count=1 if mode == "shared" else 0,
        independent_input_projection_count=0 if mode == "shared" else 8,
        output_projection_count=8,
        additional_parameter_count=CORRECTION_PARAMETER_COUNT,
        multilag_detail_rank=SHARED_WIDTH if mode == "shared" else None,
        multilag_detail_projections=("shared_current_previous_bias_free_input_branch_outputs"
                                    if mode == "shared" else
                                    "independent_current_previous_bias_free_mlps_prefix_widths15_5x7"),
        multilag_detail_initialization="default_Linear_input_zero_output_forked_cpu_rng",
        multilag_detail_implementation="project_current_previous_before_gather",
        history_context="immediate_predecessor", branch_availability="original_thresholds",
        residual_divisor=8,
        initialization="default_Linear_input_zero_output_forked_cpu_rng",
    )
    return result


def identity(hidden_dim=64, *, mode="shared"):
    value = json.dumps(metadata(hidden_dim, mode=mode), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(value).digest()), dtype=torch.uint8)


def _identity_matches(value, expected):
    return (isinstance(value, torch.Tensor) and value.dtype == torch.uint8
            and value.shape == expected.shape and torch.equal(value.detach().cpu(), expected))


class SharedHistoryCorrection(nn.Module):
    """Use one shared relation basis, or eight equal-count independent bases."""

    def __init__(self, hidden_dim=64, *, mode="shared"):
        super().__init__()
        metadata(hidden_dim, mode=mode)
        self.hidden_dim, self.mode = hidden_dim, mode
        self.width = SHARED_WIDTH if mode == "shared" else None
        self.branch_widths = (SHARED_WIDTH,) * 8 if mode == "shared" else INDEPENDENT_WIDTHS
        self.availability_lags = AVAILABILITY_LAGS
        self.branch_count, self.residual_divisor = 8, 8
        # Preserve the common encoder/head initialization and the caller RNG.
        with torch.random.fork_rng(devices=[]):
            if mode == "shared":
                self.input_projection = nn.Linear(2 * hidden_dim, self.width, bias=False)
            else:
                self.input_projections = nn.ModuleList(
                    nn.Linear(2 * hidden_dim, width, bias=False) for width in self.branch_widths)
            self.output_projections = nn.ModuleList(
                nn.Linear(width, hidden_dim, bias=False) for width in self.branch_widths)
            for projection in self.output_projections:
                nn.init.zeros_(projection.weight)

    def forward(self, hidden, valid, *, memory_write_mask=None):
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[-1] != self.hidden_dim or not hidden.is_floating_point()):
            raise ValueError("History states must have shape [batch,length,64] and floating dtype")
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        valid = valid.to(hidden.device)
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
            memory_write_mask = memory_write_mask.to(hidden.device)
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed history")
        sources, available = lag_source_indices(
            valid, memory_write_mask=memory_write_mask, mode="local")
        if self.mode == "shared":
            weight = self.input_projection.weight
            current = F.linear(safe, weight[:, :self.hidden_dim])
            previous = F.linear(safe, weight[:, self.hidden_dim:])
            # Every eligible branch uses the same immediate predecessor.  Its
            # nonlinear relation is computed once, then eligibility is applied.
            previous = previous.gather(
                1, sources[..., 0].unsqueeze(-1).expand(-1, -1, self.width))
            features = F.gelu(current + previous, approximate="none").unsqueeze(-2)
            features = torch.where(available.unsqueeze(-1), features, 0.0)
        else:
            weight = torch.cat([projection.weight for projection in self.input_projections])
            current = F.linear(safe, weight[:, :self.hidden_dim])
            previous = F.linear(safe, weight[:, self.hidden_dim:])
            features = current + previous.gather(
                1, sources[..., 0].unsqueeze(-1).expand(-1, -1, sum(self.branch_widths)))
            eligible = available.repeat_interleave(
                torch.tensor(self.branch_widths, device=available.device), dim=-1)
            features = F.gelu(torch.where(eligible, features, 0.0), approximate="none")
        if self.mode == "shared":
            features = features.flatten(-2)
        return F.linear(features, torch.cat(
            [projection.weight for projection in self.output_projections], dim=1)) / 8


class CountAwareTitanSharedHistoryMLP(CountAwareTitanCoreAblation):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *,
                 history_sharing_mode="shared", **kwargs):
        metadata(hidden_dim, mode=history_sharing_mode)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, core_mode="mlp", **kwargs)
        self.history_sharing_mode = history_sharing_mode
        self.multilag_detail = SharedHistoryCorrection(hidden_dim, mode=history_sharing_mode)
        self.register_buffer("shared_history_identity", identity(hidden_dim, mode=history_sharing_mode))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = identity(self.core_hidden_dim, mode=self.history_sharing_mode)
        if not _identity_matches(state_dict.get("shared_history_identity"), expected):
            raise ValueError("Shared-history checkpoint identity mismatch")
        if not _identity_matches(state_dict.get("core_ablation_identity"), core_identity(64, "mlp")):
            raise ValueError("Shared-history inherited core identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def _projection_shapes(mode):
    widths = (SHARED_WIDTH,) * 8 if mode == "shared" else INDEPENDENT_WIDTHS
    result = {f"multilag_detail.output_projections.{branch}.weight": (64, width)
              for branch, width in enumerate(widths)}
    if mode == "shared":
        result["multilag_detail.input_projection.weight"] = (15, 128)
    else:
        result.update({f"multilag_detail.input_projections.{branch}.weight": (width, 128)
                       for branch, width in enumerate(widths)})
    return result


def validate_checkpoint(payload, expected_backbone):
    """Bind names, metadata, split, selector and exact arm projection tensors."""
    if not isinstance(payload, Mapping):
        raise ValueError("Shared-history checkpoint must be a mapping")
    meta = payload.get("encoder_config", {})
    states = [payload[name] for name in ("model_state_dict", "best_state_dict") if name in payload]
    identified = (payload.get("backbone") in ARMS
                  or isinstance(meta, Mapping) and (
                      meta.get("backbone_contract_id") == CONTRACT_ID
                      or "history_relation_sharing" in meta or meta.get("candidate_name") in ARMS)
                  or any(isinstance(state, Mapping) and "shared_history_identity" in state for state in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("Shared-history checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping):
        raise ValueError("Shared-history encoder metadata must be a mapping")
    mode = MODE_BY_ARM[expected_backbone]
    expected = metadata(meta.get("d_model"), mode=mode)
    interface, resume, head = payload.get("interface_meta"), payload.get("resume_identity"), meta.get("time_head")
    if not all(isinstance(value, Mapping) for value in (interface, resume, head)):
        raise ValueError("Shared-history checkpoint requires interface, resume and time-head mappings")
    arguments, interface_head = resume.get("arguments"), interface.get("time_head")
    if not isinstance(arguments, Mapping) or not isinstance(interface_head, Mapping):
        raise ValueError("Shared-history resume arguments and interface time head must be mappings")
    observation = head.get("observation_likelihood")
    if (payload.get("backbone") != expected_backbone
            or any(meta.get(key) != value for key, value in expected.items())
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or payload.get("variant") != "count_only_log_regression"
            or resume.get("backbone") != expected_backbone
            or resume.get("interface_meta") != interface
            or resume.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
            or arguments.get("model_role") != role_for_arm(expected_backbone)
            or head.get("mode") != "heteroscedastic_lognormal_duration"
            or not isinstance(observation, Mapping)
            or observation.get("mode") != "positive_integer_round_clamp_v1"
            or observation != interface_head.get("observation_likelihood")):
        raise ValueError("Shared-history metadata, split, head, selector or role mismatch")
    scale = head.get("time_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Shared-history time scale must be finite and positive")
    shapes = _projection_shapes(mode)
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("Shared-history state must be a mapping")
        for name, expected_identity in (("shared_history_identity", identity(mode=mode)),
                                        ("core_ablation_identity", core_identity(64, "mlp"))):
            if not _identity_matches(state.get(name), expected_identity):
                raise ValueError("Shared-history state identity mismatch")
        if (set(key for key in state if key.startswith("multilag_detail.")) != set(shapes)
                or "history_width_identity" in state or "late_path_identity" in state
                or any(key.startswith("late_path.") for key in state)
                or not any(key.startswith("lmm.") for key in state)):
            raise ValueError("Shared-history projection set or retained memory mismatch")
        for name, shape in shapes.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
                raise ValueError("Shared-history projection tensor mismatch: " + name)
    if not states:
        digest, epoch = payload.get("checkpoint_state_sha256"), payload.get("best_epoch")
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or type(epoch) is not int or epoch < 1):
            raise ValueError("Unbound shared-history summary")
    return True
