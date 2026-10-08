"""CNN+GRU54 controls for explicit correction versus current-state capacity.

The original full arm stays in CountAwareTitanCNNGRU. Both controls retain its
initial common tensors and static prototype mean. Current-only correction does
not gather previous positions or carry recurrent state between positions. Its
current hidden state still summarizes the encoder's observed causal prefix.

Matching 25,308 active correction parameters does not match functional depth or
activation distribution: current-only uses three fixed pointwise bases in QKV
and two repeated-current GRU steps. Report these limits and measured costs.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTitanCNNGRU import (
    CountAwareTitanCNNGRU, GRUHistoryCorrection, KERNEL_KEYS,
    ObservedCausalQKVMemoryAttention, observed_context,
    identity as parent_identity, metadata as parent_metadata,
    role_for_arm as parent_role, validate_checkpoint as parent_validate,
)
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

BASELINE = "titantpp_cnn_gru54"
DISABLED = "titantpp_cnn_gru54_correction_disabled"
CURRENT_ONLY = "titantpp_cnn_gru54_current_only"
ARMS = (DISABLED, CURRENT_ONLY)
CONTRACT_ID = "titantpp_cnn_history_controls_20261008_v1"
ROLE_BY_ARM = {arm: "observed_time_" + arm.removeprefix("titantpp_") + "_v1" for arm in ARMS}
CORRECTION_PARAMETERS = 25308
RECURRENT_PARAMETERS = 24732
CNN_PARAMETERS = 576


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported CNN history control")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm):
    role_for_arm(arm)
    result = parent_metadata(hidden_dim, arm=BASELINE)
    disabled = arm == DISABLED
    result.update(
        candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), baseline_backbone=BASELINE,
        history_control="disabled" if disabled else "current_state_only",
        correction_parameter_count=RECURRENT_PARAMETERS,
        registered_correction_parameter_count=CORRECTION_PARAMETERS,
        active_correction_parameter_count=0 if disabled else CORRECTION_PARAMETERS,
        registered_additional_parameter_count=CORRECTION_PARAMETERS,
        additional_parameter_count=0 if disabled else CORRECTION_PARAMETERS,
        active_trainable_parameter_count=89859 if disabled else 115167,
        explicit_previous_correction_state=False,
        current_encoder_contains_observed_prefix=True,
        common_encoder_initialization="exact_original_CNNGRU54_tensors_and_rng",
        correction_modules_frozen=disabled,
        correction_paths_bypassed=disabled,
        cnn_enabled=not disabled,
        gru_correction_enabled=not disabled,
        correction_context="disabled" if disabled else "current_hidden_only",
        correction_state_scope="none" if disabled else "reset_each_position_two_current_steps",
        gru_steps_per_position=0 if disabled else 2,
        gru_initial_state="zero_each_position",
        cnn_lags="none" if disabled else "current_only_identity_tanh_gelu_bases",
        cnn_eligibility="original_observed_segment_counts_1_2_3",
        correction_eligibility="original_observed_segment_count_at_least_two",
        active_budget_limitations=("disabled removes active correction capacity" if disabled else
            "equal active parameter count; pointwise basis activations and two-step depth differ "
            "from historical convolution and prefix GRU; measured cost is separate"),
    )
    return result


def identity(hidden_dim=64, *, arm):
    raw = json.dumps(metadata(hidden_dim, arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


def _identity_matches(value, expected):
    return (isinstance(value, torch.Tensor) and value.dtype == torch.uint8
        and value.shape == expected.shape and torch.equal(value.cpu(), expected))


def correction_parameters(model):
    """Name every original CNN/GRU correction parameter, including frozen ones."""
    return {name: value for name, value in model.named_parameters()
            if name.startswith("multilag_detail.") or name in KERNEL_KEYS}


def current_pointwise_residual(values, kernel, available):
    """Use only a position's own value; observed counts gate the three bases.

    Distinct parameter-free bases avoid three identical current-value taps
    collapsing to one summed kernel. Availability is metadata about observation
    count, not an input containing historical state or quantity values.
    """
    if values.ndim != 3 or kernel.shape != (3, values.size(-1)):
        raise ValueError("Expected current values [batch,length,channels] and kernel [3,channels]")
    if (available.shape != (*values.shape[:2], 3) or available.dtype != torch.bool
            or available.device != values.device):
        raise ValueError("Current-only CNN context shape/type/device mismatch")
    safe = torch.where(available[..., 0, None], values, 0.)
    if not bool(torch.isfinite(safe).all()):
        raise ValueError("Nonfinite observed current QKV projection")
    result = torch.zeros_like(values)
    for channel, basis in enumerate((safe, torch.tanh(safe), F.gelu(safe, approximate="none"))):
        result = result + torch.where(available[..., channel, None], basis, 0.) * kernel[channel]
    return result


class ControlledObservedAttention(ObservedCausalQKVMemoryAttention):
    """Original observed attention arithmetic; only correction inputs differ."""

    def __init__(self, base, *, disabled):
        super().__init__(base)
        # Parent constructor draws no random values. Reuse the exact original
        # kernel objects as well, so no initial parameter is replaced.
        for name in ("causal_q_kernel", "causal_k_kernel", "causal_v_kernel"):
            setattr(self, name, getattr(base, name))
        self.disabled = disabled

    def forward(self, x, mask=None, *, observed_sources, observed_available,
                event_attention_bias=None, return_event_attention=False):
        batch, length, _ = x.shape
        if mask is None:
            raise ValueError("Observed current-only attention requires an explicit mask")
        _mask(mask, (batch, length), "attention mask")
        if not torch.equal(mask, observed_available[..., 0]):
            raise ValueError("Current-only context does not match observed attention mask")
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        if not self.disabled:
            q = q + current_pointwise_residual(q, self.causal_q_kernel, observed_available)
            k = k + current_pointwise_residual(k, self.causal_k_kernel, observed_available)
            v = v + current_pointwise_residual(v, self.causal_v_kernel, observed_available)
        # No historical source indices are dereferenced. The common attention
        # itself still reads its inherited causal prefix and static tokens.
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


class CurrentStateGRUCorrection(GRUHistoryCorrection):
    """Two repeated-current GRU steps, with independent zero state per position.

    Two steps let every hidden-hidden recurrent weight affect the output.
    A single step from zero would leave those weights permanently inactive.
    There is no sequence-position state carry, although the encoded input itself
    already contains prefix information and eligibility uses observed counts.
    """

    def __init__(self, base):
        nn.Module.__init__(self)
        self.hidden_dim, self.latent_width = base.hidden_dim, base.latent_width
        self.input_projection = base.input_projection
        self.gru = base.gru
        self.output_projection = base.output_projection

    def forward(self, hidden, valid, *, memory_write_mask=None):
        if hidden.ndim != 3 or hidden.size(-1) != self.hidden_dim:
            raise ValueError("hidden must have shape [batch,length,64]")
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        valid = valid.to(hidden.device)
        observed, _, available = observed_context(valid, memory_write_mask=memory_write_mask)
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed current hidden state")
        if hidden.size(1) == 0:
            return torch.zeros_like(hidden)
        # Treat each [sample,position] as an independent two-step microsequence.
        # This reshape does not mix positions or refer to earlier hidden rows.
        projected = self.input_projection(safe).reshape(-1, 1, self.latent_width)
        repeated_current = projected.expand(-1, 2, -1).contiguous()
        state = hidden.new_zeros((1, projected.size(0), self.latent_width))
        _, state = self.gru(repeated_current, state)
        correction = self.output_projection(state[0]).reshape_as(hidden)
        return torch.where(available[..., 1, None], correction, 0.)


class CountAwareTitanCNNHistoryControl(CountAwareTitanCNNGRU):
    """Preserve original modules/initial RNG and opt in to one Q1 control."""

    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, control_arm, **kwargs):
        metadata(hidden_dim, arm=control_arm)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, candidate_arm=BASELINE, **kwargs)
        self.control_arm = control_arm
        disabled = control_arm == DISABLED
        self.encoder.layers[0].attn = ControlledObservedAttention(self.encoder.layers[0].attn, disabled=disabled)
        if disabled:
            for parameter in correction_parameters(self).values():
                parameter.requires_grad_(False)
        else:
            self.multilag_detail = CurrentStateGRUCorrection(self.multilag_detail)
        self.register_buffer("cnn_history_control_identity", identity(hidden_dim, arm=control_arm))

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if self.control_arm != DISABLED:
            return super()._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        # Keep all common encoder/head modules; bypass both correction paths.
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch, length]")
        _mask(mask, tuple(dts.shape), "mask")
        valid = mask.to(dts.device)
        observed, sources, available = observed_context(valid, memory_write_mask=memory_write_mask)
        safe_dts = torch.where(observed, dts, 0.)
        safe_quantities = torch.where(observed, history_quantities, 0.)
        if (not bool(torch.isfinite(safe_dts).all()) or bool((safe_dts < 0).any())
                or not bool(torch.isfinite(safe_quantities).all()) or bool((safe_quantities < 0).any())):
            raise ValueError("Observed durations and quantities must be finite and nonnegative")
        encoded = self.encoder.input_proj(self.continuous_features(safe_dts, safe_quantities, observed))
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed.unsqueeze(-1), encoded, 0.)
        encoded = self.encoder.layers[0](encoded, mask=observed, observed_sources=sources, observed_available=available)
        return self.encoder.layers[1](encoded, mask=observed)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get("cnn_history_control_identity")
        if not _identity_matches(value, identity(arm=self.control_arm)):
            raise ValueError("CNN history control checkpoint identity mismatch")
        parent = state_dict.get("cnn_gru_identity")
        if not _identity_matches(parent, parent_identity(arm=BASELINE)):
            raise ValueError("CNN history control parent identity mismatch")
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    """Bind control semantics while reusing the original parent shape checks."""
    meta = payload.get("encoder_config", {})
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    identified = (payload.get("backbone") in ARMS
        or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
        or any(isinstance(state, Mapping) and "cnn_history_control_identity" in state for state in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("CNN history control cannot be relabelled")
        return False
    if not isinstance(meta, Mapping):
        raise ValueError("CNN history control metadata must be a mapping")
    required = metadata(meta.get("d_model"), arm=expected_backbone)
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(key) != value for key, value in required.items())
            or not isinstance(resume, Mapping)
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("CNN history control semantics or role mismatch")
    for state in states:
        value = state.get("cnn_history_control_identity") if isinstance(state, Mapping) else None
        if not _identity_matches(value, identity(arm=expected_backbone)):
            raise ValueError("CNN history control state identity mismatch")
        if not _identity_matches(state.get("cnn_gru_identity"), parent_identity(arm=BASELINE)):
            raise ValueError("CNN history control parent state identity mismatch")
    # Project only routing metadata for the inherited architecture validator;
    # original tensors and source/checkpoint evidence stay immutable.
    parent = {**payload, "backbone": BASELINE,
        "encoder_config": {**meta, **parent_metadata(meta.get("d_model"), arm=BASELINE)},
        "resume_identity": {**resume, "backbone": BASELINE,
            "arguments": {**resume.get("arguments", {}), "model_role": parent_role(BASELINE)}}}
    for name in ("model_state_dict", "best_state_dict"):
        if name in payload:
            parent[name] = {key: value for key, value in payload[name].items() if key != "cnn_history_control_identity"}
    if resume.get("backbone") != expected_backbone:
        raise ValueError("CNN history control resume arm mismatch")
    return parent_validate(parent, BASELINE)
