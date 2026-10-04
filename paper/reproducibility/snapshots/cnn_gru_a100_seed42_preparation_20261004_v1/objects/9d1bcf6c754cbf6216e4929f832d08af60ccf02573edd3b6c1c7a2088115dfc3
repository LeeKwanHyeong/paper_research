"""Opt-in observed-time CNN/GRU contrasts of the frozen History-MLP16 model.

Only event Q/K/V in encoder block 1 and/or the correction between blocks are
changed. Recurrent state and convolution context are explicit prefix-local
values; neither is stored between calls. Shared heads and memory stay inherited.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTitanHistoryWidth import (
    CountAwareTitanHistoryWidth, metadata as width_metadata,
)
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask
from models.TPPs.CountAwareTitanCausalQKV import CausalQKVMemoryAttention

BASELINE = "titantpp_history_mlp_width16"
ARMS = ("titantpp_cnn_mlp16", "titantpp_gru54", "titantpp_cnn_gru54")
ROLE_BY_ARM = {arm: "observed_time_" + arm.removeprefix("titantpp_") + "_v1" for arm in ARMS}
CONTRACT_ID = "titantpp_cnn_gru_width16_v1"
CNN_ARMS = (ARMS[0], ARMS[2])
GRU_ARMS = (ARMS[1], ARMS[2])
LATENT_WIDTH = 54
KERNEL_KEYS = tuple("encoder.layers.0.attn.causal_" + name + "_kernel" for name in ("q", "k", "v"))


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported CNN/GRU arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm):
    role_for_arm(arm)
    if type(hidden_dim) is not int or hidden_dim != 64:
        raise ValueError("CNN/GRU comparison fixes hidden_dim=64")
    result = width_metadata(hidden_dim, width=16)
    # Do not retain names that would label a recurrent module as an eight-branch MLP.
    for name in tuple(result):
        if name.startswith("multilag_detail_") or name in (
                "core_ablation_mode", "history_mlp_width", "bottleneck_width",
                "branch_availability", "residual_divisor", "history_context", "initialization"):
            result.pop(name)
    cnn, gru = arm in CNN_ARMS, arm in GRU_ARMS
    result.update(candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), baseline_backbone=BASELINE,
        correction_kind="gru54" if gru else "mlp16", correction_latent_width=54 if gru else 16,
        correction_parameter_count=24732 if gru else 24576,
        correction_insertion="between_encoder_blocks",
        correction_divisor=1 if gru else 8,
        correction_eligibility="observed_segment_count_at_least_two" if gru else "original_eight_thresholds",
        correction_availability_lags=None if gru else [1, 2, 4, 8, 16, 32, 64, 128],
        correction_context="segment_prefix_recurrent" if gru else "current_immediate_predecessor",
        correction_state_scope="sample_prefix_segment_forward_only",
        correction_padding="skip_no_state_or_count_update",
        correction_withheld_valid="reset_state_and_count_zero_correction",
        correction_initialization="forked_default_input_recurrence_zero_output",
        gru_layers=1 if gru else 0, gru_unidirectional=True, gru_bias=True if gru else None,
        gru_dropout=0, gru_initial_state="zero_per_prefix" if gru else None,
        cnn_enabled=cnn, cnn_layer=0 if cnn else None,
        cnn_scope="event_qkv_after_projection_before_persistent_concat" if cnn else None,
        cnn_kernel_size=3 if cnn else None, cnn_channels=64 if cnn else None,
        cnn_groups=64 if cnn else None, cnn_bias=False,
        cnn_lags="current_lag1_lag2_observed_segment" if cnn else None,
        cnn_initialization="all_zero_residual_kernel" if cnn else None,
        cnn_parameter_count=576 if cnn else 0,
        persistent_tokens_filtered=False, attention_reference_scope_changed=False,
        mutable_batch_context_cache=False, main_hidden_dim=64,
        added_layer_norm=False, added_learned_scale=False, fixed_alpha=1,
        additional_parameter_count=(24732 if gru else 24576) + (576 if cnn else 0))
    return result


def identity(hidden_dim=64, *, arm):
    encoded = json.dumps(metadata(hidden_dim, arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


def observed_context(valid, *, memory_write_mask=None):
    """Return current/lag1/lag2 indices and eligibility, skipping padding.

    A valid withheld row resets adapter eligibility; the attention reference
    scope itself is unchanged and continues to use the inherited causal mask.
    """
    if not isinstance(valid, torch.Tensor) or valid.ndim != 2:
        raise ValueError("valid must have shape [batch, length]")
    _mask(valid, tuple(valid.shape), "valid")
    if memory_write_mask is not None:
        _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
    observed = valid if memory_write_mask is None else valid & memory_write_mask.to(valid.device)
    batch, length = valid.shape
    if length == 0:
        return observed, torch.zeros((batch, 0, 3), device=valid.device, dtype=torch.long), torch.zeros(
            (batch, 0, 3), device=valid.device, dtype=torch.bool)
    count = observed.long().cumsum(1)
    local_count = count - torch.where(valid & ~observed, count, 0).cummax(1).values
    positions = torch.arange(length, device=valid.device).expand(batch, -1)
    latest = torch.where(observed, positions, -1).cummax(1).values
    previous = torch.cat((torch.full((batch, 1), -1, device=valid.device, dtype=torch.long), latest[:, :-1]), 1)
    previous2 = previous.gather(1, previous.clamp_min(0))
    previous2 = torch.where(previous >= 0, previous2, -1)
    available = observed.unsqueeze(-1) & (local_count.unsqueeze(-1) > torch.arange(3, device=valid.device))
    sources = torch.stack((positions, previous, previous2), -1)
    return observed, torch.where(available, sources, 0), available


def causal_depthwise_residual(values, kernel, sources, available):
    """Channelwise current/observed-lag filtering with explicit context."""
    if values.ndim != 3 or kernel.shape != (3, values.size(-1)):
        raise ValueError("Expected values [batch,length,channels] and kernel [3,channels]")
    if (sources.shape != (*values.shape[:2], 3) or sources.dtype != torch.long
            or available.shape != sources.shape or available.dtype != torch.bool
            or sources.device != values.device or available.device != values.device):
        raise ValueError("CNN observed context shape/type/device mismatch")
    safe = torch.where(available[..., 0, None], values, 0.)
    if not bool(torch.isfinite(safe).all()):
        raise ValueError("Nonfinite observed QKV projection")
    result = torch.zeros_like(values)
    for lag in range(3):
        gathered = safe.gather(1, sources[..., lag, None].expand_as(values))
        result = result + torch.where(available[..., lag, None], gathered, 0.) * kernel[lag]
    return result


class ObservedCausalQKVMemoryAttention(CausalQKVMemoryAttention):
    """Inherited attention arithmetic with explicit observed-segment CNN input."""
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
        v = v + causal_depthwise_residual(v, self.causal_v_kernel, observed_sources, observed_available)
        # This mirrors MemoryAttention after projection; persistent/context
        # tokens bypass the filter and key visibility remains unchanged.
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


class ObservedCNNEncoderLayer(nn.Module):
    """Copy existing block modules while passing context as arguments."""
    def __init__(self, base):
        super().__init__()
        for name in ("norm1", "norm2", "drop1", "drop2", "ff"):
            setattr(self, name, getattr(base, name))
        self.attn = ObservedCausalQKVMemoryAttention(base.attn)

    def forward(self, x, mask=None, *, observed_sources, observed_available, event_attention_bias=None):
        hidden = self.attn(self.norm1(x), mask=mask, observed_sources=observed_sources,
            observed_available=observed_available, event_attention_bias=event_attention_bias)
        x = x + self.drop1(hidden)
        x = x * mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x * mask.to(dtype=x.dtype).unsqueeze(-1)


class GRUHistoryCorrection(nn.Module):
    """One unidirectional GRU; padding skips and withheld valid rows reset."""
    def __init__(self, hidden_dim=64, *, latent_width=LATENT_WIDTH):
        super().__init__()
        if hidden_dim != 64 or type(latent_width) is not int or latent_width != LATENT_WIDTH:
            raise ValueError("Main GRU contrast fixes hidden_dim64 and latent_width54")
        self.hidden_dim, self.latent_width = hidden_dim, latent_width
        with torch.random.fork_rng(devices=[]):
            self.input_projection = nn.Linear(hidden_dim, latent_width, bias=False)
            self.gru = nn.GRU(latent_width, latent_width, num_layers=1, bias=True,
                batch_first=True, dropout=0, bidirectional=False)
            self.output_projection = nn.Linear(latent_width, hidden_dim, bias=False)
            nn.init.zeros_(self.output_projection.weight)

    def forward(self, hidden, valid, *, memory_write_mask=None):
        if hidden.ndim != 3 or hidden.size(-1) != self.hidden_dim:
            raise ValueError("hidden must have shape [batch,length,64]")
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        valid = valid.to(hidden.device)
        observed, _, _ = observed_context(valid, memory_write_mask=memory_write_mask)
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed recurrent history")
        projected = self.input_projection(safe)
        batch, length = valid.shape
        state = hidden.new_zeros((1, batch, self.latent_width))
        count = torch.zeros(batch, device=hidden.device, dtype=torch.long)
        outputs = []
        for position in range(length):
            reset = valid[:, position] & ~observed[:, position]
            state = torch.where(reset[None, :, None], 0., state)
            count = torch.where(reset, 0, count)
            _, next_state = self.gru(projected[:, position:position + 1], state)
            active = observed[:, position]
            state = torch.where(active[None, :, None], next_state, state)
            count = count + active.long()
            correction = self.output_projection(state[0])
            outputs.append(torch.where((active & (count >= 2)).unsqueeze(-1), correction, 0.))
        return torch.stack(outputs, 1) if outputs else hidden.new_zeros(hidden.shape)


class CountAwareTitanCNNGRU(CountAwareTitanHistoryWidth):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, candidate_arm, **kwargs):
        metadata(hidden_dim, arm=candidate_arm)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, history_mlp_width=16, **kwargs)
        self.candidate_arm = candidate_arm
        # Architecture identities describe the implemented route rather than
        # retaining inherited MLP identifiers on recurrent replacements.
        del self._buffers["history_width_identity"]
        del self._buffers["core_ablation_identity"]
        if candidate_arm in GRU_ARMS:
            self.multilag_detail = GRUHistoryCorrection(hidden_dim)
        if candidate_arm in CNN_ARMS:
            self.encoder.layers[0] = ObservedCNNEncoderLayer(self.encoder.layers[0])
        self.register_buffer("cnn_gru_identity", identity(hidden_dim, arm=candidate_arm))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get("cnn_gru_identity")
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(arm=self.candidate_arm)):
            raise ValueError("CNN/GRU checkpoint identity mismatch")
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if self.candidate_arm not in CNN_ARMS:
            return super()._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
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
        features = self.continuous_features(safe_dts, safe_quantities, observed)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed.unsqueeze(-1), encoded, 0.)
        encoded = self.encoder.layers[0](encoded, mask=observed, observed_sources=sources, observed_available=available)
        encoded = encoded + self.multilag_detail(encoded, valid, memory_write_mask=observed)
        return self.encoder.layers[1](encoded, mask=observed)


def validate_checkpoint(payload, expected_backbone):
    """Reject candidate relabelling and bind exact architecture/split/head/role."""
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in ARMS
        or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
        or any(isinstance(state, Mapping) and ("cnn_gru_identity" in state
            or any(key in state for key in KERNEL_KEYS) and any(
                str(key).startswith("multilag_detail.") for key in state)
            or any(str(key).startswith("multilag_detail.gru.") for key in state)) for state in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("CNN/GRU checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping):
        raise ValueError("CNN/GRU checkpoint metadata must be a mapping")
    required = metadata(meta.get("d_model"), arm=expected_backbone)
    interface, resume = payload.get("interface_meta", {}), payload.get("resume_identity", {})
    head = meta.get("time_head", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(key) != value for key, value in required.items())
        or payload.get("evaluation_scope") != "validation_only" or payload.get("held_out_test_evaluated") is not False
        or payload.get("variant") != "count_only_log_regression"
        or resume.get("backbone") != expected_backbone or resume.get("interface_meta") != interface
        or resume.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
        or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)
        or head.get("mode") != "heteroscedastic_lognormal_duration"
        or not isinstance(head.get("observation_likelihood"), Mapping)
        or head["observation_likelihood"].get("mode") != "positive_integer_round_clamp_v1"
        or head["observation_likelihood"] != interface.get("time_head", {}).get("observation_likelihood")):
        raise ValueError("CNN/GRU metadata, split, head, selector or role mismatch")
    expected_shapes = ({"multilag_detail.input_projection.weight": (54, 64),
        "multilag_detail.output_projection.weight": (64, 54),
        "multilag_detail.gru.weight_ih_l0": (162, 54),
        "multilag_detail.gru.weight_hh_l0": (162, 54),
        "multilag_detail.gru.bias_ih_l0": (162,), "multilag_detail.gru.bias_hh_l0": (162,)}
        if expected_backbone in GRU_ARMS else {f"multilag_detail.{name}_projections.{branch}.weight": shape
            for branch in range(8) for name, shape in (("input", (16, 128)), ("output", (64, 16)))})
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("CNN/GRU checkpoint state must be a mapping")
        value = state.get("cnn_gru_identity")
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(arm=expected_backbone)):
            raise ValueError("CNN/GRU state identity mismatch")
        shapes = dict(expected_shapes)
        if expected_backbone in CNN_ARMS:
            shapes.update({key: (3, 64) for key in KERNEL_KEYS})
        elif any(key in state for key in KERNEL_KEYS):
            raise ValueError("CNN/GRU unexpected convolution state")
        actual_correction_keys = {key for key in state if str(key).startswith("multilag_detail.")}
        if actual_correction_keys != set(expected_shapes):
            raise ValueError("CNN/GRU correction state keys mismatch")
        if any(not isinstance(state.get(key), torch.Tensor) or tuple(state[key].shape) != shape
                or not bool(torch.isfinite(state[key]).all()) for key, shape in shapes.items()):
            raise ValueError("CNN/GRU projection or recurrent state shape mismatch")
        if "history_width_identity" in state or "core_ablation_identity" in state:
            raise ValueError("CNN/GRU contains inherited architecture identity")
    if not states:
        digest = payload.get("checkpoint_state_sha256", "")
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or type(payload.get("best_epoch")) is not int or payload["best_epoch"] < 1):
            raise ValueError("Unbound CNN/GRU summary")
    return True
