"""Matched location contrast: one nonlinear causal CNN branch on Q or V.

The original depthwise Q/K/V CNN is retained. Both candidates add the same
1,024-weight grouped branch with identical initialization. The common QKV
projection remains trainable; moving a branch is not freezing Q or K.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F
from models.TPPs.CountAwareTitanNonlinearLocalCNN import (
    NonlinearLocalBranch as _FrozenChannelBranch, CHANNEL as _CHANNEL, observed_taps,
)

from models.TPPs import CountAwareTitanCNNGRUComponents as components
from models.TPPs.CountAwareTitanCNNGRU import ObservedCausalQKVMemoryAttention, causal_depthwise_residual
from models.TPPs.CountAwareTitanCNNHistoryControls import _identity_matches
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

CNN = "titantpp_cnn"
QUERY = "titantpp_cnn_query_nonlinear_v1"
VALUE = "titantpp_cnn_value_nonlinear_v1"
ARMS = (QUERY, VALUE)
LOCATION_BY_ARM = {QUERY: "q", VALUE: "v"}
ALL_ARMS = (CNN, *ARMS)
LEGACY_CNN = components.CNN
CONTRACT_ID = "titantpp_nonlinear_location_cnn_20261015_v1"
IDENTITY_BUFFER = "nonlinear_location_cnn_identity"
EXTRA_PREFIX = "encoder.layers.0.attn.nonlinear_location_branches."
ADDED_PARAMETERS = 1024
ACTIVE_PARAMETERS = {arm: components.ACTIVE_PARAMETERS[LEGACY_CNN] + (ADDED_PARAMETERS if arm in ARMS else 0)
                     for arm in ALL_ARMS}


def role_for_arm(arm):
    if arm not in ALL_ARMS:
        raise ValueError("Unsupported nonlinear-location CNN arm")
    return "observed_time_" + arm.removeprefix("titantpp_") + ("_v1" if arm == CNN else "")


def metadata(hidden_dim=64, *, arm, max_seq_len=256):
    role_for_arm(arm)
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("Context length must be a positive integer")
    result = components.metadata(hidden_dim, arm=LEGACY_CNN)
    candidate = arm in ARMS
    position_difference = hidden_dim * (max_seq_len - 256)
    result.update(candidate_name=arm, model_role=role_for_arm(arm), backbone_contract_id=CONTRACT_ID,
        baseline_backbone=LEGACY_CNN, nonlinear_location_enabled=candidate,
        nonlinear_location_insertion="encoder_block0_event_qkv_after_projection_before_memory_concat",
        nonlinear_location_route=LOCATION_BY_ARM.get(arm),
        nonlinear_location_tap_order=["current", "previous_observed", "second_previous_observed"],
        nonlinear_location_receptive_field=3,
        nonlinear_location_groups=64 if candidate else None,
        nonlinear_location_hidden_channels=256 if candidate else None,
        nonlinear_location_expansion_per_channel=4 if candidate else None,
        nonlinear_location_activation="gelu_exact" if candidate else None,
        nonlinear_location_bias=False, nonlinear_location_dropout=0., nonlinear_location_normalization=False,
        nonlinear_location_original_depthwise_retained=True,
        nonlinear_location_qkv_projection_trainable=True,
        nonlinear_location_extra_parameters=ADDED_PARAMETERS if candidate else 0,
        nonlinear_location_initialization="fork_rng_default_temporal_zero_pointwise" if candidate else "original_CNN",
        nonlinear_location_minimum_observations=1,
        nonlinear_location_padding="skip_observed_lags_zero_unavailable",
        nonlinear_location_withheld_valid="reset_observed_segment_zero_branch",
        nonlinear_location_parameter_budget_matched_between_candidates=True,
        nonlinear_location_budget_limitations="Identical branch weights, width, grouping and three observations; Q versus V placement changes function and optimization",
        active_budget_limitations="Query-only and Value-only match added trainable weights; stored QKV Channel has 3072 added weights and is a secondary comparison",
        original_depthwise_cnn_parameter_count=components.CNN_PARAMETERS,
        cnn_parameter_count=components.CNN_PARAMETERS + (ADDED_PARAMETERS if candidate else 0),
        registered_cnn_parameter_count=components.CNN_PARAMETERS + (ADDED_PARAMETERS if candidate else 0),
        registered_correction_parameter_count=components.CNN_PARAMETERS + components.GRU_PARAMETERS + (ADDED_PARAMETERS if candidate else 0),
        registered_additional_parameter_count=components.CNN_PARAMETERS + components.GRU_PARAMETERS + (ADDED_PARAMETERS if candidate else 0),
        parameter_count_context_length=max_seq_len,
        active_trainable_parameter_count=ACTIVE_PARAMETERS[arm] + position_difference,
        registered_parameter_count=components.REGISTERED_PARAMETERS + position_difference + (ADDED_PARAMETERS if candidate else 0),
        additional_parameter_count=components.CNN_PARAMETERS + (ADDED_PARAMETERS if candidate else 0),
        active_correction_parameter_count=components.CNN_PARAMETERS + (ADDED_PARAMETERS if candidate else 0),
        common_encoder_initialization="exact_original_CNN_tensors_and_rng")
    return result


def identity(*, arm):
    if arm not in ARMS:
        raise ValueError("Only nonlinear candidates have an additional state identity")
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(arm=arm), sort_keys=True).encode()).digest()), dtype=torch.uint8)


class NonlinearLocationBranch(_FrozenChannelBranch):
    """Exactly the frozen Channel branch, independently of its Q/V location."""
    def __init__(self, *, arm):
        if arm not in ARMS:
            raise ValueError("Nonlinear location branch requires Query or Value arm")
        super().__init__(arm=_CHANNEL)


class NonlinearLocationAttention(ObservedCausalQKVMemoryAttention):
    def __init__(self, base, *, arm):
        nn.Module.__init__(self)
        for name in ("d_model", "n_heads", "use_causal", "head_dim", "scale", "qkv", "out_proj", "drop",
                     "contextual_mem_size", "persistent_mem_size", "persistent_mem", "_ctx_buf", "_ctx_mem",
                     "causal_q_kernel", "causal_k_kernel", "causal_v_kernel"):
            setattr(self, name, getattr(base, name))
        self.arm = arm
        with torch.random.fork_rng(devices=[]):
            self.nonlinear_location_branches = nn.ModuleDict({LOCATION_BY_ARM[arm]: NonlinearLocationBranch(arm=arm)})
        self.train(base.training)

    def forward(self, x, mask=None, *, observed_sources, observed_available,
                event_attention_bias=None, return_event_attention=False):
        batch, length, _ = x.shape
        if mask is None:
            raise ValueError("Observed CNN attention requires an explicit mask")
        _mask(mask, (batch, length), "attention mask")
        if not torch.equal(mask, observed_available[..., 0]):
            raise ValueError("CNN context does not match observed attention mask")
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        def filtered(values, name):
            result = values + causal_depthwise_residual(values, getattr(self, "causal_" + name + "_kernel"),
                    observed_sources, observed_available)
            if name in self.nonlinear_location_branches:
                result = result + self.nonlinear_location_branches[name](values, observed_sources, observed_available)
            return result
        q, k, v = filtered(q, "q"), filtered(k, "k"), filtered(v, "v")
        # Frozen attention arithmetic; memory tokens never enter either CNN path.
        memory_parts = []
        if self._ctx_mem is not None and self._ctx_mem.numel() > 0:
            memory_parts.append(self._ctx_mem.to(device=x.device, dtype=x.dtype))
        if self.persistent_mem is not None:
            memory_parts.append(self.persistent_mem.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1))
        memory_length = 0
        if memory_parts:
            memory = torch.cat(memory_parts, dim=1)
            memory_length = memory.size(1)
            k, v = torch.cat([memory, k], dim=1), torch.cat([memory, v], dim=1)
        scores = torch.matmul(self._split_heads(q), self._split_heads(k).transpose(-2, -1)) * self.scale
        if event_attention_bias is not None:
            if event_attention_bias.shape != (batch, self.n_heads, length, length):
                raise ValueError("Event attention bias shape mismatch")
            scores = scores + torch.nn.functional.pad(event_attention_bias.to(dtype=scores.dtype, device=x.device), (memory_length, 0))
        full_mask = None
        if self.use_causal:
            full_mask = torch.cat((torch.ones(length, memory_length, device=x.device, dtype=torch.bool),
                torch.tril(torch.ones(length, length, device=x.device, dtype=torch.bool))), 1)
        valid_keys = torch.cat((torch.ones(batch, memory_length, device=x.device, dtype=torch.bool), mask), 1)
        key_mask = valid_keys[:, None, None, :]
        full_mask = key_mask if full_mask is None else full_mask[None, None, :, :] & key_mask
        if memory_length == 0:
            invalid_queries = ~mask
            if invalid_queries.any():
                full_mask = full_mask.expand(batch, 1, length, length).clone()
                batch_ids, positions = invalid_queries.nonzero(as_tuple=True)
                full_mask[batch_ids, 0, positions, positions] = True
        scores = scores.masked_fill(~full_mask, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        event_attention = attention[..., memory_length:].mean(1) if return_event_attention else None
        attention = self.drop(attention)
        output = self.out_proj(self._merge_heads(torch.matmul(attention, self._split_heads(v))))
        output = output * mask.to(dtype=output.dtype).unsqueeze(-1)
        return (output, event_attention) if return_event_attention else output


class CountAwareTitanNonlinearLocationCNN(components.CountAwareTitanCNNGRUComponents):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, nonlinear_arm, **kwargs):
        if nonlinear_arm not in ARMS:
            raise ValueError("Use original model for the baseline alias")
        metadata(hidden_dim, arm=nonlinear_arm, max_seq_len=max_seq_len)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, component_arm=LEGACY_CNN, **kwargs)
        self.nonlinear_arm = nonlinear_arm
        self.encoder.layers[0].attn = NonlinearLocationAttention(self.encoder.layers[0].attn, arm=nonlinear_arm)
        self.register_buffer(IDENTITY_BUFFER, identity(arm=nonlinear_arm))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not _identity_matches(state_dict.get(IDENTITY_BUFFER), identity(arm=self.nonlinear_arm)):
            raise ValueError("Nonlinear-location CNN identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in ALL_ARMS or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
        or any(isinstance(s, Mapping) and (IDENTITY_BUFFER in s or any(k.startswith(EXTRA_PREFIX) for k in s)) for s in states))
    if expected_backbone not in ALL_ARMS:
        if identified:
            raise ValueError("Nonlinear-location CNN checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping) or not states:
        raise ValueError("Nonlinear-location CNN metadata and state are required")
    required = metadata(meta.get("d_model"), arm=expected_backbone, max_seq_len=meta.get("max_len"))
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(k) != v for k, v in required.items())
            or not isinstance(resume, Mapping) or resume.get("backbone") != expected_backbone
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("Nonlinear-location CNN metadata or route mismatch")
    shapes = {f"{EXTRA_PREFIX}{LOCATION_BY_ARM[expected_backbone]}.{layer}.weight": shape
        for layer, shape in (("temporal", (256, 1, 3)), ("pointwise", (64, 4, 1)))} if expected_backbone in ARMS else {}
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("Nonlinear-location CNN state must be a mapping")
        if expected_backbone in ARMS:
            if not _identity_matches(state.get(IDENTITY_BUFFER), identity(arm=expected_backbone)):
                raise ValueError("Nonlinear-location CNN state identity mismatch")
        elif IDENTITY_BUFFER in state:
            raise ValueError("Nonlinear candidate cannot be relabelled as baseline")
        if {k for k in state if k.startswith(EXTRA_PREFIX)} != set(shapes):
            raise ValueError("Unexpected nonlinear local state")
        for name, shape in shapes.items():
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not torch.is_floating_point(value) or not bool(torch.isfinite(value).all()):
                raise ValueError("Nonlinear location state tensor invalid: " + name)
    parent = {**payload, "backbone": LEGACY_CNN, "encoder_config": {**meta, **components.metadata(arm=LEGACY_CNN)},
        "resume_identity": {**resume, "backbone": LEGACY_CNN,
            "arguments": {**resume.get("arguments", {}), "model_role": components.role_for_arm(LEGACY_CNN)}}}
    for key in ("model_state_dict", "best_state_dict"):
        if key in payload:
            parent[key] = {name: value for name, value in payload[key].items() if name != IDENTITY_BUFFER and not name.startswith(EXTRA_PREFIX)}
    return components.validate_checkpoint(parent, LEGACY_CNN)
