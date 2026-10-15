"""V-only observed-difference constraint of the frozen CNN-only backbone.

Q/K keep the original trainable three-tap CNN. V replaces its three free
coefficients with two adjacent observed differences, including at boundaries.
This is a restricted parameterization, not a parameter-matched new mechanism.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs import CountAwareTitanCNNGRUComponents as components
from models.TPPs.CountAwareTitanCNNGRU import ObservedCausalQKVMemoryAttention, causal_depthwise_residual
from models.TPPs.CountAwareTitanCNNHistoryControls import _identity_matches
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

CNN = "titantpp_cnn"
VALUE_DIFFERENCE = "titantpp_cnn_value_difference_v1"
ARMS = (VALUE_DIFFERENCE,)
ALL_ARMS = (CNN, *ARMS)
LEGACY_CNN = components.CNN
CONTRACT_ID = "titantpp_value_difference_cnn_20261016_v1"
IDENTITY_BUFFER = "value_difference_cnn_identity"
EXTRA_PREFIX = DIFFERENCE_KEY = "encoder.layers.0.attn.value_difference_kernel"
REMOVED_KEY = "encoder.layers.0.attn.causal_v_kernel"
PARAMETER_DELTA = ADDED_PARAMETERS = -64
LOCAL_CNN_PARAMETERS = 512
ACTIVE_PARAMETERS = {arm: components.ACTIVE_PARAMETERS[LEGACY_CNN] + (PARAMETER_DELTA if arm in ARMS else 0)
                     for arm in ALL_ARMS}


def role_for_arm(arm):
    if arm not in ALL_ARMS:
        raise ValueError("Unsupported value-difference CNN arm")
    return "observed_time_" + arm.removeprefix("titantpp_") + ("_v1" if arm == CNN else "")


def metadata(hidden_dim=64, *, arm, max_seq_len=256):
    role_for_arm(arm)
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("Context length must be a positive integer")
    result = components.metadata(hidden_dim, arm=LEGACY_CNN)
    candidate = arm in ARMS
    count = LOCAL_CNN_PARAMETERS if candidate else components.CNN_PARAMETERS
    position_difference = hidden_dim * (max_seq_len - 256)
    result.update(candidate_name=arm, model_role=role_for_arm(arm), backbone_contract_id=CONTRACT_ID,
        baseline_backbone=LEGACY_CNN, value_difference_enabled=candidate,
        value_difference_insertion="encoder_block0_event_v_after_projection_before_memory_concat",
        value_difference_formula="a*(v0-v1)*I(prev)+b*(v1-v2)*I(prev2)" if candidate else None,
        value_difference_kernel_shape=[2, 64] if candidate else None,
        value_difference_tap_order=["current", "previous_observed", "second_previous_observed"],
        value_difference_minimum_observations=2 if candidate else None,
        value_difference_missing_differences="exact_zero",
        value_difference_padding="skip_observed_lags",
        value_difference_withheld_valid="reset_observed_segment_zero_residual",
        value_difference_initialization="zero_coefficients_no_rng_consumption" if candidate else "original_CNN",
        value_difference_qk="original_independent_trainable_three_tap",
        value_difference_parameter_delta=PARAMETER_DELTA if candidate else 0,
        value_difference_unused_parameters=0,
        value_difference_constant_latent_preserved=candidate,
        value_difference_raw_quantity_bias_guaranteed=False,
        value_difference_added_nonlinearity=False, value_difference_added_gate=False,
        value_difference_added_qk_bound=False, value_difference_added_confidence=False,
        cnn_parameter_count=count, registered_cnn_parameter_count=count,
        registered_correction_parameter_count=count + components.GRU_PARAMETERS,
        registered_additional_parameter_count=count + components.GRU_PARAMETERS,
        parameter_count_context_length=max_seq_len,
        active_trainable_parameter_count=ACTIVE_PARAMETERS[arm] + position_difference,
        registered_parameter_count=components.REGISTERED_PARAMETERS + position_difference + (PARAMETER_DELTA if candidate else 0),
        additional_parameter_count=count, active_correction_parameter_count=count,
        common_encoder_initialization="exact_original_CNN_common_tensors_and_rng_except_replaced_V_kernel",
        active_budget_limitations="576 to 512 local coefficients; reduced freedom, short-history boundaries and difference-coordinate weight decay are part of the intervention")
    return result


def identity(*, arm):
    if arm not in ARMS:
        raise ValueError("Only the value-difference candidate has this state identity")
    raw = json.dumps(metadata(arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


def observed_difference_residual(values, kernel, sources, available):
    """Two channelwise differences; unavailable pairs contribute exactly zero.

    Context comes from the original observed_context function. Padding is
    skipped and valid withheld rows reset availability. The current latent V
    itself is unchanged by this helper; callers add the returned residual.
    """
    if values.ndim != 3 or kernel.shape != (2, values.size(-1)):
        raise ValueError("Expected values [batch,length,channels] and difference kernel [2,channels]")
    if (sources.shape != (*values.shape[:2], 3) or sources.dtype != torch.long
            or available.shape != sources.shape or available.dtype != torch.bool
            or sources.device != values.device or available.device != values.device):
        raise ValueError("Difference observed context shape/type/device mismatch")
    safe = torch.where(available[..., 0, None], values, 0.)
    if not bool(torch.isfinite(safe).all()):
        raise ValueError("Nonfinite observed V projection")
    taps = [safe.gather(1, sources[..., lag, None].expand_as(values)) for lag in range(3)]
    first = torch.where(available[..., 1, None], taps[0] - taps[1], 0.)
    second = torch.where(available[..., 2, None], taps[1] - taps[2], 0.)
    return first * kernel[0] + second * kernel[1]


class ValueDifferenceAttention(ObservedCausalQKVMemoryAttention):
    def __init__(self, base):
        nn.Module.__init__(self)
        for name in ("d_model", "n_heads", "use_causal", "head_dim", "scale", "qkv", "out_proj", "drop",
                     "contextual_mem_size", "persistent_mem_size", "persistent_mem", "_ctx_buf", "_ctx_mem",
                     "causal_q_kernel", "causal_k_kernel"):
            setattr(self, name, getattr(base, name))
        # No old V kernel, unused trainable padding, or random construction.
        self.value_difference_kernel = nn.Parameter(base.causal_v_kernel.new_zeros((2, self.d_model)))
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
        q = q + causal_depthwise_residual(q, self.causal_q_kernel, observed_sources, observed_available)
        k = k + causal_depthwise_residual(k, self.causal_k_kernel, observed_sources, observed_available)
        v = v + observed_difference_residual(v, self.value_difference_kernel, observed_sources, observed_available)
        # Frozen attention arithmetic; memory tokens bypass every CNN filter.
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
            scores = scores + F.pad(event_attention_bias.to(dtype=scores.dtype, device=x.device), (memory_length, 0))
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


class CountAwareTitanValueDifferenceCNN(components.CountAwareTitanCNNGRUComponents):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, difference_arm=VALUE_DIFFERENCE, **kwargs):
        if difference_arm not in ARMS:
            raise ValueError("Use the original model for the baseline alias")
        metadata(hidden_dim, arm=difference_arm, max_seq_len=max_seq_len)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, component_arm=LEGACY_CNN, **kwargs)
        self.difference_arm = difference_arm
        self.encoder.layers[0].attn = ValueDifferenceAttention(self.encoder.layers[0].attn)
        self.register_buffer(IDENTITY_BUFFER, identity(arm=difference_arm))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not _identity_matches(state_dict.get(IDENTITY_BUFFER), identity(arm=self.difference_arm)):
            raise ValueError("Value-difference CNN identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in ALL_ARMS or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
        or any(isinstance(s, Mapping) and (IDENTITY_BUFFER in s or DIFFERENCE_KEY in s) for s in states))
    if expected_backbone not in ALL_ARMS:
        if identified:
            raise ValueError("Value-difference CNN checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping) or not states:
        raise ValueError("Value-difference CNN metadata and state are required")
    required = metadata(meta.get("d_model"), arm=expected_backbone, max_seq_len=meta.get("max_len"))
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(k) != v for k, v in required.items())
            or not isinstance(resume, Mapping) or resume.get("backbone") != expected_backbone
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("Value-difference CNN metadata or route mismatch")
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("Value-difference CNN state must be a mapping")
        if expected_backbone in ARMS:
            if not _identity_matches(state.get(IDENTITY_BUFFER), identity(arm=expected_backbone)):
                raise ValueError("Value-difference CNN state identity mismatch")
            value = state.get(DIFFERENCE_KEY)
            if (REMOVED_KEY in state or not isinstance(value, torch.Tensor) or tuple(value.shape) != (2, 64)
                    or not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())):
                raise ValueError("Value-difference CNN replacement tensor invalid")
        elif IDENTITY_BUFFER in state or DIFFERENCE_KEY in state:
            raise ValueError("Value-difference candidate cannot be relabelled as baseline")
    parent = {**payload, "backbone": LEGACY_CNN, "encoder_config": {**meta, **components.metadata(arm=LEGACY_CNN)},
        "resume_identity": {**resume, "backbone": LEGACY_CNN,
            "arguments": {**resume.get("arguments", {}), "model_role": components.role_for_arm(LEGACY_CNN)}}}
    for key in ("model_state_dict", "best_state_dict"):
        if key in payload:
            state = {name: value for name, value in payload[key].items() if name not in (IDENTITY_BUFFER, DIFFERENCE_KEY)}
            if expected_backbone in ARMS:
                # Schema adapter only: the inherited validator requires old V's
                # shape. Never saved, loaded into the candidate, or optimized.
                state[REMOVED_KEY] = payload[key][DIFFERENCE_KEY].new_zeros((3, 64))
            parent[key] = state
    return components.validate_checkpoint(parent, LEGACY_CNN)
