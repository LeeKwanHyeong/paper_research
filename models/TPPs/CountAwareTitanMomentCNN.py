"""Preparation-only CNN block normalization and observed-moment conditioning.

No factory registration, data access, learned output statistics, or head change.
This is causal *latent block* RevIN (affine=False), not input/output RevIN.
For the linear CNN, the latent scale cancels algebraically. The conditioned arm
uses separate observed log-quantity moments to modulate the centered filter.
"""
from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs import CountAwareTitanCNNGRUComponents as components
from models.TPPs.CountAwareTitanCNNGRU import observed_context, ObservedCausalQKVMemoryAttention
from models.TPPs.CountAwareTitanCNNHistoryControls import _identity_matches
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

CNN = components.CNN
REVIN = "titantpp_cnn_block_revin_v1"
CONDITIONED = "titantpp_cnn_moment_conditioned_v1"
CAPACITY = "titantpp_cnn_moment_capacity_v1"
ARMS = (REVIN, CONDITIONED, CAPACITY)
CONTRACT_ID = "titantpp_cnn_moment_conditioning_20261015_v1"
IDENTITY_BUFFER = "moment_cnn_identity"
EXTRA_PREFIX = "encoder.layers.0.attn.moment_networks."
EPS = 1e-5
EXTRA_PARAMETERS = {REVIN: 0, CONDITIONED: 195, CAPACITY: 195}


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unknown moment-CNN arm")
    return "observed_time_" + arm.removeprefix("titantpp_")


def metadata(hidden_dim=64, *, arm):
    role_for_arm(arm)
    result = components.metadata(hidden_dim, arm=CNN)
    result.update(
        candidate_name=arm, model_role=role_for_arm(arm), backbone_contract_id=CONTRACT_ID,
        baseline_backbone=CNN, block_normalization="causal_latent_qkv_revin_affine_false",
        normalization_domain="first_block_event_qkv_after_layernorm_and_projection",
        normalization_scope="each_query_observed_segment_prefix; same_query_stats_for_all_three_taps",
        normalization_eps=EPS, normalization_variance_ddof=0,
        normalization_statistics="detached_float64_prefix_sums_cast_to_feature_dtype",
        normalization_withheld="reset_prefix_statistics; preserve_parent_attention_scope",
        normalization_padding="skip; no_statistics_state_persisted",
        normalization_inverse="latent_residual_times_prefix_std; outer_identity_preserved",
        linear_latent_scale_cancels=True, ordinary_input_output_revin=False,
        quantity_input_domain="log1p_observed_quantity_unchanged",
        moment_features=["prefix_mean_log1p_quantity", "log1p_prefix_std_log1p_quantity"],
        moment_network_shape=None if arm == REVIN else [2, 16, 1],
        moment_network_sharing="one_per_qkv_shared_across_channels",
        moment_operation={REVIN: "centered_linear_cnn", CONDITIONED: "relative_tap_weights",
                          CAPACITY: "additive_normalized_feature_residual"}[arm],
        moment_minimum_observations=2, moment_strength=0.5,
        moment_tap_bounds=[0.5, 1.5] if arm == CONDITIONED else None,
        moment_available_tap_mean=1.0 if arm == CONDITIONED else None,
        moment_residual_norm_bound=False,
        moment_initialization="fork_rng_first_layer_zero_final_layer",
        active_trainable_parameter_count=components.ACTIVE_PARAMETERS[CNN] + EXTRA_PARAMETERS[arm],
        registered_parameter_count=components.REGISTERED_PARAMETERS + EXTRA_PARAMETERS[arm],
        active_correction_parameter_count=components.CNN_PARAMETERS + EXTRA_PARAMETERS[arm],
        additional_parameter_count=components.CNN_PARAMETERS + EXTRA_PARAMETERS[arm],
        common_encoder_initialization="exact_original_CNN_tensors_and_rng",
        shared_input_loss_head_unchanged=True,
        active_budget_limitations="Conditioned/capacity match inputs and parameters, not function classes or optimization",
    )
    return result


def identity(*, arm):
    raw = json.dumps(metadata(arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


def prefix_moments(values, valid, observed):
    """Detached per-channel prefix moments; padding skips, withheld rows reset.

Statistics are accumulated in float64. No whole-window statistic can reach an
earlier query. This function has no mutable module/global cache.
"""
    if values.ndim != 3 or not torch.is_floating_point(values):
        raise ValueError("Expected floating [batch,length,channels]")
    _mask(valid, tuple(values.shape[:2]), "valid")
    _mask(observed, tuple(values.shape[:2]), "observed")
    if valid.device != values.device or observed.device != values.device or bool((observed & ~valid).any()):
        raise ValueError("Invalid observed mask or device")
    if values.size(1) == 0:
        return values.detach(), torch.ones_like(values), observed.long()
    safe = torch.where(observed[..., None], values.detach(), 0.).to(torch.float64)
    if not bool(torch.isfinite(safe).all()):
        raise ValueError("Nonfinite observed statistic input")
    count = observed.long().cumsum(1)
    positions = torch.arange(values.size(1), device=values.device).expand_as(valid)
    resets = torch.where(valid & ~observed, positions + 1, 0).cummax(1).values
    totals = torch.cat((safe.new_zeros((values.size(0), 1, values.size(2))), safe.cumsum(1)), 1)
    squares = torch.cat((safe.new_zeros((values.size(0), 1, values.size(2))), safe.square().cumsum(1)), 1)
    counts = F.pad(count, (1, 0))
    n = count - counts.gather(1, resets)
    indices = resets[..., None].expand_as(safe)
    total = totals[:, 1:] - totals.gather(1, indices)
    square = squares[:, 1:] - squares.gather(1, indices)
    denominator = n.clamp_min(1)[..., None]
    mean = total / denominator
    variance = (square / denominator - mean.square()).clamp_min(0.)
    std = torch.sqrt(variance + EPS)
    if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("Nonfinite prefix moments")
    mean = torch.where(observed[..., None], mean, 0.).to(values.dtype)
    std = torch.where(observed[..., None], std, 1.).to(values.dtype)
    return mean, std, torch.where(observed, n, 0)


def normalized_taps(values, mean, std, sources, available):
    if mean.shape != values.shape or std.shape != values.shape or bool((std <= 0).any()):
        raise ValueError("Invalid reversible statistics")
    if (sources.shape != (*values.shape[:2], 3) or sources.dtype != torch.long
            or available.shape != sources.shape or available.dtype != torch.bool):
        raise ValueError("Invalid tap context")
    safe = torch.where(available[..., 0, None], values, 0.)
    if not bool(torch.isfinite(safe).all() and torch.isfinite(mean).all() and torch.isfinite(std).all()):
        raise ValueError("Nonfinite normalized tap input")
    taps = torch.stack([safe.gather(1, sources[..., lag, None].expand_as(values)) for lag in range(3)], -2)
    # All three lag values use THIS query's statistics, not their own past stats.
    return torch.where(available[..., None], (taps - mean[..., None, :]) / std[..., None, :], 0.)


def tap_weights(raw, available):
    u = torch.where(available[..., 1, None], torch.tanh(raw), 0.)
    past_count = available[..., 1:].sum(-1).clamp_min(1).to(raw.dtype)[..., None]
    return torch.stack((1. + .5 * u, 1. - .5 * u / past_count, 1. - .5 * u / past_count), -2)


class MomentController(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.network = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self, moments):
        return self.network(moments)


def reversible_cnn_residual(values, kernel, mean, std, sources, available, *, arm, controller=None, moments=None):
    role_for_arm(arm)
    if kernel.shape != (3, values.size(-1)):
        raise ValueError("Invalid CNN kernel shape")
    taps = normalized_taps(values, mean, std, sources, available)
    if arm == REVIN:
        if controller is not None:
            raise ValueError("Plain block RevIN has no controller")
        weights = torch.ones_like(taps[..., :1])
        raw = None
    else:
        if controller is None or moments is None or moments.shape != (*values.shape[:2], 2):
            raise ValueError("Observed quantity moments and controller required")
        raw = controller(moments)
        weights = tap_weights(raw, available) if arm == CONDITIONED else torch.ones_like(taps[..., :1])
    result = torch.zeros_like(values)
    for lag in range(3):
        result = result + taps[..., lag, :] * kernel[lag] * weights[..., lag, :]
    if arm == CAPACITY:
        result = result + torch.where(available[..., 1, None], .5 * torch.tanh(raw), 0.)
    # denorm(z_current + correction) == values + std * correction.
    # Return only the residual: no duplicate mean addition, exact zero-kernel identity.
    return torch.where(available[..., 0, None], std * result, 0.)


class MomentAttention(ObservedCausalQKVMemoryAttention):
    def __init__(self, base, *, arm):
        role_for_arm(arm)
        nn.Module.__init__(self)
        for name in ("d_model", "n_heads", "use_causal", "head_dim", "scale", "qkv", "out_proj", "drop",
                     "contextual_mem_size", "persistent_mem_size", "persistent_mem", "_ctx_buf", "_ctx_mem",
                     "causal_q_kernel", "causal_k_kernel", "causal_v_kernel"):
            setattr(self, name, getattr(base, name))
        self.moment_arm = arm
        self.moment_networks = nn.ModuleDict({} if arm == REVIN else {name: MomentController() for name in ("q", "k", "v")})
        self.train(base.training)

    def forward(self, x, mask=None, *, observed_sources, observed_available, valid, quantity_log):
        batch, length, _ = x.shape
        _mask(mask, (batch, length), "attention mask")
        if not torch.equal(mask, observed_available[..., 0]) or quantity_log.shape != mask.shape:
            raise ValueError("Moment CNN context mismatch")
        projected = self.qkv(x)
        mean, std, _ = prefix_moments(torch.cat((projected, quantity_log[..., None]), -1), valid, mask)
        condition = torch.cat((mean[..., -1:], torch.log1p(std[..., -1:])), -1)
        outputs = []
        for index, (name, value) in enumerate(zip(("q", "k", "v"), projected.chunk(3, -1))):
            part = slice(index * self.d_model, (index + 1) * self.d_model)
            residual = reversible_cnn_residual(value, getattr(self, "causal_" + name + "_kernel"),
                mean[..., part], std[..., part], observed_sources, observed_available, arm=self.moment_arm,
                controller=None if self.moment_arm == REVIN else self.moment_networks[name], moments=condition)
            outputs.append(value + residual)
        q, k, v = outputs
        # Inherited attention arithmetic: memory tokens and reference scope unchanged.
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
        attention = self.drop(F.softmax(scores, dim=-1))
        output = self.out_proj(self._merge_heads(torch.matmul(attention, self._split_heads(v))))
        return output * mask.to(dtype=output.dtype).unsqueeze(-1)


class MomentEncoderLayer(nn.Module):
    def __init__(self, base, *, arm):
        super().__init__()
        for name in ("norm1", "norm2", "drop1", "drop2", "ff"):
            setattr(self, name, getattr(base, name))
        self.attn = MomentAttention(base.attn, arm=arm)

    def forward(self, x, mask, **context):
        hidden = self.attn(self.norm1(x), mask=mask, **context)
        x = (x + self.drop1(hidden)) * mask.to(x.dtype)[..., None]
        return (x + self.drop2(self.ff(self.norm2(x)))) * mask.to(x.dtype)[..., None]


class CountAwareTitanMomentCNN(components.CountAwareTitanCNNGRUComponents):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, moment_arm, **kwargs):
        metadata(hidden_dim, arm=moment_arm)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, component_arm=CNN, **kwargs)
        self.moment_arm = moment_arm
        self.encoder.layers[0] = MomentEncoderLayer(self.encoder.layers[0], arm=moment_arm)
        self.register_buffer(IDENTITY_BUFFER, identity(arm=moment_arm))

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch,length]")
        _mask(mask, tuple(dts.shape), "mask")
        valid = mask.to(dts.device)
        observed, sources, available = observed_context(valid, memory_write_mask=memory_write_mask)
        dt = torch.where(observed, dts, 0.)
        quantity = torch.where(observed, history_quantities, 0.)
        if (not bool(torch.isfinite(dt).all()) or not bool(torch.isfinite(quantity).all())
                or bool((dt < 0).any()) or bool((quantity < 0).any())):
            raise ValueError("Observed durations and quantities must be finite and nonnegative")
        features = self.continuous_features(dt, quantity, observed)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed[..., None], encoded, 0.)
        encoded = self.encoder.layers[0](encoded, observed, observed_sources=sources,
            observed_available=available, valid=valid, quantity_log=features[..., 1])
        return self.encoder.layers[1](encoded, mask=observed)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not _identity_matches(state_dict.get(IDENTITY_BUFFER), identity(arm=self.moment_arm)):
            raise ValueError("Moment-CNN identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in ARMS or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
                  or any(isinstance(s, Mapping) and IDENTITY_BUFFER in s for s in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("Moment-CNN checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping) or not states:
        raise ValueError("Moment-CNN metadata and state required")
    required = metadata(meta.get("d_model"), arm=expected_backbone)
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(k) != v for k, v in required.items())
            or not isinstance(resume, Mapping) or resume.get("backbone") != expected_backbone
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("Moment-CNN metadata or route mismatch")
    shapes = {} if expected_backbone == REVIN else {
        f"{EXTRA_PREFIX}{q}.network.{k}": shape for q in ("q", "k", "v")
        for k, shape in (("0.weight", (16, 2)), ("0.bias", (16,)), ("2.weight", (1, 16)), ("2.bias", (1,)))}
    for state in states:
        if not isinstance(state, Mapping) or not _identity_matches(state.get(IDENTITY_BUFFER), identity(arm=expected_backbone)):
            raise ValueError("Moment-CNN state identity mismatch")
        if {k for k in state if k.startswith(EXTRA_PREFIX)} != set(shapes):
            raise ValueError("Moment network state keys mismatch")
        for name, shape in shapes.items():
            value = state[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not torch.is_floating_point(value) or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid moment network state: " + name)
    parent = {**payload, "backbone": CNN, "encoder_config": {**meta, **components.metadata(arm=CNN)},
        "resume_identity": {**resume, "backbone": CNN, "arguments": {**resume.get("arguments", {}), "model_role": components.role_for_arm(CNN)}}}
    for key in ("model_state_dict", "best_state_dict"):
        if key in payload:
            parent[key] = {n: v for n, v in payload[key].items() if n != IDENTITY_BUFFER and not n.startswith(EXTRA_PREFIX)}
    return components.validate_checkpoint(parent, CNN)
