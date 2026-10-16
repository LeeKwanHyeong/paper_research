"""Matched time/quantity fusion before the unchanged History-MLP16 encoder.

Both arms retain the original two encoder blocks, persistent memory, inter-block
History-MLP16 and final static prototype read. The new branch sees observed
log-duration/log-quantity only. Its segment mask does not change the inherited
encoder's attention reference scope.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTitanHistoryWidth import (
    CountAwareTitanHistoryWidth, metadata as width_metadata,
    identity as width_identity,
)
from models.TPPs.CountAwareTitanCoreAblation import identity as core_identity
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

POINTWISE = "titantpp_history16_pointwise_fusion_v1"
CROSS = "titantpp_history16_cross_fusion_v1"
ARMS = (POINTWISE, CROSS)
ROLE_BY_ARM = {arm: "observed_time_" + arm for arm in ARMS}
CONTRACT_ID = "history16_time_quantity_fusion_v1"
FUSION_PARAMETERS = 6176


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported time/quantity fusion arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm):
    role_for_arm(arm)
    result = width_metadata(hidden_dim, width=16)
    result.update(
        candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), baseline_backbone="titantpp_history_mlp_width16",
        fusion_kind="bidirectional_causal_cross_attention" if arm == CROSS else "independent_pointwise_mlp",
        fusion_insertion="input_projection_residual_before_encoder_position_and_block1",
        fusion_modality_width=16, fusion_cross_heads=1 if arm == CROSS else 0,
        fusion_mlp_width=32 if arm == POINTWISE else None,
        fusion_parameter_count=FUSION_PARAMETERS,
        fusion_direct_inputs="log1p_observed_dt_and_quantity_separate_bias_free_embeddings",
        fusion_embedding_activation="gelu_exact_erf_before_fixed_position",
        fusion_positions="fixed_sinusoidal_observed_segment_ordinal_zero_based",
        fusion_output="concat_time_quantity_time_context_quantity_context_bias_free_64_to_64",
        fusion_mask="causal_inclusive_observed_prefix_same_withheld_segment_padding_skipped",
        fusion_initialization="forked_cpu_rng_default_inputs_zero_final_output",
        fusion_attention_dropout=0, fusion_added_norm=False,
        fusion_memory="none_new_original_persistent_and_final_static_retained",
        attention_reference_scope_changed=False,
        additional_parameter_count=24576 + FUSION_PARAMETERS,
    )
    return result


def identity(*, arm):
    encoded = json.dumps(metadata(arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


def observed_segments(valid, memory_write_mask=None):
    """Observed flags, segment id and zero-based observed ordinal per segment."""
    if not isinstance(valid, torch.Tensor) or valid.ndim != 2:
        raise ValueError("valid must be boolean [batch,length]")
    _mask(valid, tuple(valid.shape), "valid")
    if memory_write_mask is not None:
        _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
        memory_write_mask = memory_write_mask.to(valid.device)
    observed = valid if memory_write_mask is None else valid & memory_write_mask
    withheld = valid & ~observed
    segment = withheld.long().cumsum(1)
    count = observed.long().cumsum(1)
    reset = torch.where(withheld, count, 0).cummax(1).values
    ordinal = (count - reset - 1).clamp_min(0)
    return observed, segment, ordinal


def sinusoidal_position(ordinal, dtype):
    frequency = torch.exp(torch.arange(0, 16, 2, device=ordinal.device, dtype=torch.float32)
                          * (-math.log(10000.) / 16))
    phase = ordinal.to(torch.float32).unsqueeze(-1) * frequency
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2).to(dtype)


class CrossDirection(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(16, 16, bias=False)
        self.key = nn.Linear(16, 16, bias=False)
        self.value = nn.Linear(16, 16, bias=False)
        self.output = nn.Linear(16, 16, bias=False)

    def forward(self, query, source, allowed):
        score = self.query(query) @ self.key(source).transpose(-1, -2) / 4.
        # No allowed key for a masked query: use finite logits then explicitly
        # zero all weights, avoiding softmax(-inf,...,-inf) and NaN gradients.
        has_key = allowed.any(-1, keepdim=True)
        score = score.masked_fill(~allowed, -torch.inf)
        score = torch.where(has_key, score, 0.)
        weights = torch.softmax(score, dim=-1).masked_fill(~allowed, 0.)
        return self.output(weights @ self.value(source))


class TimeQuantityFusion(nn.Module):
    def __init__(self, arm):
        super().__init__()
        role_for_arm(arm)
        self.arm = arm
        with torch.random.fork_rng(devices=[]):
            self.time_embedding = nn.Linear(1, 16, bias=False)
            self.quantity_embedding = nn.Linear(1, 16, bias=False)
            if arm == CROSS:
                self.time_context = CrossDirection()
                self.quantity_context = CrossDirection()
            else:
                self.time_context = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(),
                                                  nn.Linear(32, 16, bias=False))
                self.quantity_context = nn.Sequential(nn.Linear(16, 32, bias=False), nn.GELU(),
                                                      nn.Linear(32, 16, bias=False))
            self.output = nn.Linear(64, 64, bias=False)
            nn.init.zeros_(self.output.weight)

    def forward(self, log_features, valid, *, memory_write_mask=None):
        if log_features.ndim != 3 or log_features.shape[-1] != 2:
            raise ValueError("Fusion input must be [batch,length,2]")
        _mask(valid, tuple(log_features.shape[:2]), "valid")
        valid = valid.to(log_features.device)
        observed, segment, ordinal = observed_segments(valid, memory_write_mask)
        safe = torch.where(observed.unsqueeze(-1), log_features, 0.)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Nonfinite observed fusion input")
        if safe.shape[1] == 0:
            return safe.new_zeros((*safe.shape[:2], 64))
        position = sinusoidal_position(ordinal, safe.dtype)
        time = torch.where(observed.unsqueeze(-1), F.gelu(self.time_embedding(safe[..., :1])) + position, 0.)
        quantity = torch.where(observed.unsqueeze(-1), F.gelu(self.quantity_embedding(safe[..., 1:])) + position, 0.)
        if self.arm == CROSS:
            length = safe.shape[1]
            causal = torch.ones(length, length, dtype=torch.bool, device=safe.device).tril()
            allowed = (observed[:, :, None] & observed[:, None, :]
                       & (segment[:, :, None] == segment[:, None, :]) & causal)
            time_context = self.time_context(time, quantity, allowed)
            quantity_context = self.quantity_context(quantity, time, allowed)
        else:
            time_context = self.time_context(time)
            quantity_context = self.quantity_context(quantity)
        correction = self.output(torch.cat((time, quantity, time_context, quantity_context), -1))
        return torch.where(observed.unsqueeze(-1), correction, 0.)


class CountAwareTitanCrossFusion(CountAwareTitanHistoryWidth):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, fusion_arm, **kwargs):
        metadata(hidden_dim, arm=fusion_arm)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, history_mlp_width=16, **kwargs)
        self.fusion_arm = fusion_arm
        self.input_fusion = TimeQuantityFusion(fusion_arm)
        self.register_buffer("cross_fusion_identity", identity(arm=fusion_arm))

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch,length]")
        _mask(mask, tuple(dts.shape), "mask")
        valid = mask.to(dts.device)
        observed, _, _ = observed_segments(valid, memory_write_mask)
        safe_dts = torch.where(observed, dts, 0.)
        safe_quantities = torch.where(observed, history_quantities, 0.)
        if (not bool(torch.isfinite(safe_dts).all()) or bool((safe_dts < 0).any())
                or not bool(torch.isfinite(safe_quantities).all()) or bool((safe_quantities < 0).any())):
            raise ValueError("Observed durations and quantities must be finite and nonnegative")
        if self.encoder is None or self.lmm is None or len(self.encoder.layers) != 2:
            raise RuntimeError("Fusion requires the original two blocks and static Hard-LMM")
        features = self.continuous_features(safe_dts, safe_quantities, observed)
        encoded = self.encoder.input_proj(features) + self.input_fusion(
            features, valid, memory_write_mask=observed)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed.unsqueeze(-1), encoded, 0.)
        encoded = self.encoder.layers[0](encoded, mask=observed)
        encoded = encoded + self.multilag_detail(encoded, valid, memory_write_mask=observed)
        return self.encoder.layers[1](encoded, mask=observed)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not isinstance(state_dict.get("cross_fusion_identity"), torch.Tensor) or not torch.equal(
                state_dict["cross_fusion_identity"].cpu(), identity(arm=self.fusion_arm)):
            raise ValueError("Fusion checkpoint arm identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in ARMS or "fusion_kind" in meta
                  or any("cross_fusion_identity" in state for state in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("Fusion candidate cannot be relabelled")
        return False
    expected = metadata(meta.get("d_model"), arm=expected_backbone)
    interface, resume = payload.get("interface_meta", {}), payload.get("resume_identity", {})
    head = meta.get("time_head", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(k) != v for k, v in expected.items())
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or payload.get("variant") != "count_only_log_regression"
            or resume.get("backbone") != expected_backbone or resume.get("interface_meta") != interface
            or resume.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)
            or head.get("mode") != "heteroscedastic_lognormal_duration"
            or head.get("observation_likelihood", {}).get("mode") != "positive_integer_round_clamp_v1"
            or head.get("observation_likelihood") != interface.get("time_head", {}).get("observation_likelihood")):
        raise ValueError("Fusion metadata, split, head, selector or role mismatch")
    for state in states:
        for name, expected_value in (("cross_fusion_identity", identity(arm=expected_backbone)),
                                     ("history_width_identity", width_identity(width=16)),
                                     ("core_ablation_identity", core_identity(64, "mlp"))):
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), expected_value):
                raise ValueError("Fusion state identity mismatch")
        expected_shapes = {"time_embedding.weight": (16, 1), "quantity_embedding.weight": (16, 1),
                           "output.weight": (64, 64)}
        for side in ("time_context", "quantity_context"):
            if expected_backbone == CROSS:
                expected_shapes.update({side + "." + name + ".weight": (16, 16)
                                        for name in ("query", "key", "value", "output")})
            else:
                expected_shapes.update({side + ".0.weight": (32, 16), side + ".2.weight": (16, 32)})
        actual = {k.removeprefix("input_fusion."): v for k, v in state.items() if k.startswith("input_fusion.")}
        if set(actual) != set(expected_shapes) or any(
                not isinstance(actual[k], torch.Tensor) or tuple(actual[k].shape) != shape
                for k, shape in expected_shapes.items()):
            raise ValueError("Fusion state projection shape mismatch")
    if not states and (len(payload.get("checkpoint_state_sha256", "")) != 64 or payload.get("best_epoch", 0) < 1):
        raise ValueError("Unbound fusion summary")
    return True
