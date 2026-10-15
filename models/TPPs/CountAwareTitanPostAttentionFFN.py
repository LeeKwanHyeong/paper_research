"""Final-FFN nonlinear observed-lag expansion of the unchanged CNN backbone.

Both new arms retain the original QKV filters, attention, FFN matrices and
registered inactive GRU. Only access to previous FFN hidden content differs.
No historical checkpoint is loaded and no new random draw is consumed.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs import CountAwareTitanCNNGRUComponents as components
from models.TPPs.CountAwareTitanCNNGRU import observed_context, KERNEL_KEYS
from models.TPPs.CountAwareTitanCNNHistoryControls import _identity_matches
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

CNN = "titantpp_cnn"  # Saved-result label, never an accepted new model arm.
LEGACY_CNN = components.CNN
CURRENT = "titantpp_cnn_ffn_current_v1"
TEMPORAL = "titantpp_cnn_ffn_temporal_v1"
ARMS = ALL_ARMS = (CURRENT, TEMPORAL)
CONTRACT_ID = "titantpp_post_attention_ffn_20261016_v1"
IDENTITY_BUFFER = "post_attention_ffn_identity"
EXTRA_PREFIX = "encoder.layers.1.ffn_observed."
EXTRA_KEYS = tuple(EXTRA_PREFIX + name for name in ("a", "b", "c"))
ADDED_PARAMETERS = PARAMETER_DELTA = 1152
ACTIVE_PARAMETERS = {arm: components.ACTIVE_PARAMETERS[LEGACY_CNN] + ADDED_PARAMETERS for arm in ARMS}
PARENT_SOURCE_CLOSURE = "fc0668016b8eb89752e03d0751e317e77b06d8135c0b82f8b43625e345abf9a3"


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Only the two new post-attention FFN arms are supported")
    return "observed_time_" + arm.removeprefix("titantpp_")


def metadata(hidden_dim=64, *, arm, max_seq_len=256):
    role_for_arm(arm)
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("Context length must be a positive integer")
    result = components.metadata(hidden_dim, arm=LEGACY_CNN)
    position_difference = hidden_dim * (max_seq_len - 256)
    result.update(candidate_name=arm, model_role=role_for_arm(arm), backbone_contract_id=CONTRACT_ID,
        baseline_backbone=LEGACY_CNN, parent_source_closure=PARENT_SOURCE_CLOSURE,
        post_attention_ffn_enabled=True, post_attention_ffn_arm=arm,
        post_attention_ffn_schema="channelwise_nonlinear_observed_lag_v1",
        post_attention_ffn_insertion="encoder.layers.1.after_ff.0_before_ff.1",
        post_attention_ffn_sources=["current", "current", "current"] if arm == CURRENT else
            ["current", "previous_observed", "second_previous_observed"],
        post_attention_ffn_formula="C=sum_l A_l*c_l*GELU(a_l*u_source_l+b_l)/3; W2(dropout(GELU(u+C)))",
        post_attention_ffn_parameter_shapes={name: [3, 128] for name in ("a", "b", "c")},
        post_attention_ffn_added_parameters=ADDED_PARAMETERS,
        post_attention_ffn_initialization={"a": [1., 1., 1.], "b": [-1., 0., 1.], "c": [0., 0., 0.]},
        post_attention_ffn_padding="skip_observed_lags", post_attention_ffn_withheld_valid="reset_local_gather_only",
        post_attention_ffn_divisor=3, post_attention_ffn_added_rng_draws=0,
        post_attention_ffn_context_state="none", post_attention_ffn_added_gate=False,
        post_attention_ffn_added_norm=False, post_attention_ffn_warm_start=False,
        post_attention_ffn_channel_mixing="unchanged_original_ffn_W1_W2",
        post_attention_ffn_initial_ab_task_gradient="zero_until_c_nonzero; AdamW_decay_may_update_from_step1",
        original_qkv_filters_retained=True, original_ffn_matrices_retained=True,
        active_local_cnn_parameters=576, active_value_cnn_parameters=192,
        active_query_cnn_parameters=192, active_key_cnn_parameters=192,
        parameter_count_context_length=max_seq_len,
        active_trainable_parameter_count=ACTIVE_PARAMETERS[arm] + position_difference,
        registered_parameter_count=components.REGISTERED_PARAMETERS + ADDED_PARAMETERS + position_difference,
        active_correction_parameter_count=576 + ADDED_PARAMETERS,
        additional_parameter_count=576 + ADDED_PARAMETERS,
        registered_correction_parameter_count=576 + ADDED_PARAMETERS + components.GRU_PARAMETERS,
        registered_additional_parameter_count=576 + ADDED_PARAMETERS + components.GRU_PARAMETERS,
        common_encoder_initialization="exact_original_CNN_tensors_and_rng_plus_deterministic_FFN_parameters",
        active_budget_limitations="same1152_new_parameters_not_same_function_space_or_optimization; current_hidden_already_contains_attention_prefix")
    return result


def identity(*, arm, max_seq_len=256):
    raw = json.dumps(metadata(arm=arm, max_seq_len=max_seq_len), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


class ObservedFFNResidual(nn.Module):
    """Tapwise nonlinear channel filter with explicit, stateless observation context."""
    def __init__(self, *, arm):
        super().__init__()
        role_for_arm(arm)
        self.arm = arm
        self.a = nn.Parameter(torch.ones(3, 128))
        self.b = nn.Parameter(torch.tensor([-1., 0., 1.])[:, None].expand(3, 128).clone())
        self.c = nn.Parameter(torch.zeros(3, 128))

    def forward(self, values, sources, available):
        if not isinstance(values, torch.Tensor) or values.ndim != 3 or values.size(-1) != 128:
            raise ValueError("Expanded FFN values must be [batch,length,128]")
        shape = (*values.shape[:2], 3)
        if (not isinstance(sources, torch.Tensor) or sources.shape != shape or sources.dtype != torch.long
                or not isinstance(available, torch.Tensor) or available.shape != shape or available.dtype != torch.bool
                or sources.device != values.device or available.device != values.device):
            raise ValueError("FFN observed context shape/type/device mismatch")
        if any(tuple(p.shape) != (3, 128) or not bool(torch.isfinite(p).all()) for p in (self.a, self.b, self.c)):
            raise ValueError("Nonfinite or malformed FFN parameters")
        batch, length = values.shape[:2]
        if length == 0:
            return torch.zeros_like(values)
        positions = torch.arange(length, device=values.device)[None, :].expand(batch, -1)
        if (bool((sources < 0).any()) or bool((sources >= length).any())
                or bool((sources > positions[..., None]).any())
                or bool((sources[~available] != 0).any())
                or not torch.equal(sources[..., 0][available[..., 0]], positions[available[..., 0]])
                or bool((available[..., 1:] & ~available[..., :-1]).any())):
            raise ValueError("Invalid/noncausal FFN observed indices or availability")
        observed = available[..., 0]
        for lag in (1, 2):
            if (bool((available[..., lag] & (sources[..., lag] >= sources[..., lag - 1])).any())
                    or bool((available[..., lag] & ~observed.gather(1, sources[..., lag])).any())):
                raise ValueError("FFN lag must reference an earlier observed position")
        safe = torch.where(observed[..., None], values, 0.)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed FFN projection")
        result = torch.zeros_like(values)
        for lag in range(3):
            selected = safe if self.arm == CURRENT else safe.gather(1, sources[..., lag, None].expand_as(values))
            # Mask before arithmetic, then after GELU: absent taps cannot leak bias.
            selected = torch.where(available[..., lag, None], selected, 0.)
            affine = self.a[lag] * selected + self.b[lag]
            hidden = F.gelu(affine)
            weighted = self.c[lag] * torch.where(available[..., lag, None], hidden, 0.)
            if any(not bool(torch.isfinite(x).all()) for x in (affine, hidden, weighted)):
                raise ValueError("Nonfinite FFN nonlinear intermediate")
            result = result + weighted
        result = result / 3.
        if not bool(torch.isfinite(result).all()):
            raise ValueError("Nonfinite FFN residual sum")
        return result


class PostAttentionFFNBlock(nn.Module):
    def __init__(self, base, *, arm):
        super().__init__()
        if (len(base.ff) != 4 or not isinstance(base.ff[0], nn.Linear)
                or (base.ff[0].in_features, base.ff[0].out_features) != (64, 128)
                or not isinstance(base.ff[1], nn.GELU) or not isinstance(base.ff[2], nn.Dropout)
                or not isinstance(base.ff[3], nn.Linear)
                or (base.ff[3].in_features, base.ff[3].out_features) != (128, 64)):
            raise ValueError("Original final FFN structure changed")
        for name in ("norm1", "attn", "drop1", "norm2", "ff", "drop2"):
            setattr(self, name, getattr(base, name))
        self.ffn_observed = ObservedFFNResidual(arm=arm)
        self.train(base.training)

    def forward(self, x, mask=None, *, observed_sources, observed_available, event_attention_bias=None):
        if mask is None:
            raise ValueError("FFN block requires the observed mask")
        _mask(mask, tuple(x.shape[:2]), "FFN block mask")
        if not torch.equal(mask, observed_available[..., 0]):
            raise ValueError("FFN context and block mask disagree")
        h = self.norm1(x)
        h = (self.attn(h, mask=mask) if event_attention_bias is None else
             self.attn(h, mask=mask, event_attention_bias=event_attention_bias))
        x = x + self.drop1(h)
        x = x * mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        u = self.ff[0](self.norm2(x))
        combined = u + self.ffn_observed(u, observed_sources, observed_available)
        if not bool(torch.isfinite(combined).all()):
            raise ValueError("Nonfinite final FFN expanded hidden")
        h = self.ff[3](self.ff[2](self.ff[1](combined)))
        x = x + self.drop2(h)
        return x * mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)

    @torch.no_grad()
    def update_contextual_memory(self, x_detached):
        self.attn.update_contextual_memory(x_detached.detach())

    @torch.no_grad()
    def reset_contextual_memory(self):
        self.attn.reset_contextual_memory()


class CountAwareTitanPostAttentionFFN(components.CountAwareTitanCNNGRUComponents):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, post_attention_ffn_arm=CURRENT, **kwargs):
        metadata(hidden_dim, arm=post_attention_ffn_arm, max_seq_len=max_seq_len)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, component_arm=LEGACY_CNN, **kwargs)
        self.post_attention_ffn_arm = post_attention_ffn_arm
        self.post_attention_ffn_max_seq_len = max_seq_len
        self.encoder.layers[1] = PostAttentionFFNBlock(self.encoder.layers[1], arm=post_attention_ffn_arm)
        self.register_buffer(IDENTITY_BUFFER, identity(arm=post_attention_ffn_arm, max_seq_len=max_seq_len))

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch,length]")
        _mask(mask, tuple(dts.shape), "mask")
        observed, sources, available = observed_context(mask.to(dts.device), memory_write_mask=memory_write_mask)
        safe_dts = torch.where(observed, dts, 0.)
        safe_quantities = torch.where(observed, history_quantities, 0.)
        if (not bool(torch.isfinite(safe_dts).all()) or bool((safe_dts < 0).any())
                or not bool(torch.isfinite(safe_quantities).all()) or bool((safe_quantities < 0).any())):
            raise ValueError("Observed durations and quantities must be finite and nonnegative")
        encoded = self.encoder.input_proj(self.continuous_features(safe_dts, safe_quantities, observed))
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed.unsqueeze(-1), encoded, 0.)
        encoded = self.encoder.layers[0](encoded, mask=observed,
            observed_sources=sources, observed_available=available)
        return self.encoder.layers[1](encoded, mask=observed,
            observed_sources=sources, observed_available=available)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not _identity_matches(state_dict.get(IDENTITY_BUFFER), identity(
                arm=self.post_attention_ffn_arm, max_seq_len=self.post_attention_ffn_max_seq_len)):
            raise ValueError("Post-attention FFN identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    """Candidate identity and new state bind to the unchanged parent validation contract."""
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in ARMS
        or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
        or any(isinstance(s, Mapping) and (IDENTITY_BUFFER in s or any(str(k).startswith(EXTRA_PREFIX) for k in s)) for s in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("Post-attention FFN checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping) or not states:
        raise ValueError("Post-attention FFN metadata and state required")
    required = metadata(meta.get("d_model"), arm=expected_backbone, max_seq_len=meta.get("max_len"))
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(k) != v for k, v in required.items())
            or not isinstance(resume, Mapping) or resume.get("backbone") != expected_backbone
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("Post-attention FFN arm/source/schema/metadata mismatch")
    parent = {**payload, "backbone": LEGACY_CNN,
        "encoder_config": {**meta, **components.metadata(arm=LEGACY_CNN)},
        "resume_identity": {**resume, "backbone": LEGACY_CNN,
            "arguments": {**resume.get("arguments", {}), "model_role": components.role_for_arm(LEGACY_CNN)}}}
    for name in ("model_state_dict", "best_state_dict"):
        if name not in payload:
            continue
        state = payload[name]
        if not isinstance(state, Mapping) or not _identity_matches(state.get(IDENTITY_BUFFER), identity(
                arm=expected_backbone, max_seq_len=meta["max_len"])):
            raise ValueError("Post-attention FFN state identity mismatch")
        if {k for k in state if str(k).startswith(EXTRA_PREFIX)} != set(EXTRA_KEYS):
            raise ValueError("Post-attention FFN parameter inventory mismatch")
        for key in EXTRA_KEYS:
            value = state[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != (3, 128) or not bool(torch.isfinite(value).all()):
                raise ValueError("Malformed/nonfinite post-attention FFN parameter")
        if any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError("Malformed/nonfinite candidate state")
        parent[name] = {k: v for k, v in state.items() if k != IDENTITY_BUFFER and k not in EXTRA_KEYS}
    return components.validate_checkpoint(parent, LEGACY_CNN)
