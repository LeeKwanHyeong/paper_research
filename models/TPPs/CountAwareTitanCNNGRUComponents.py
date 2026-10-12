"""Isolated CNN-only/GRU-only routes with original CNN+GRU initialization.

Each control constructs the full parent, preserves every registered tensor and
its construction RNG, then freezes and bypasses only the inactive component.
There is no replacement MLP, additional normalization, or trained scale.
Parameter-count differences are explicit; this is not a capacity-matched study.
The unchanged FULL and NONE arms remain in their original model modules.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn

from models.TPPs.CountAwareTitanCNNGRU import (
    CountAwareTitanCNNGRU, KERNEL_KEYS, observed_context,
    identity as parent_identity, metadata as parent_metadata,
    role_for_arm as parent_role, validate_checkpoint as parent_validate,
)
from models.TPPs.CountAwareTitanCNNHistoryControls import (
    ControlledObservedAttention, correction_parameters, _identity_matches,
)
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

FULL = BASELINE = "titantpp_cnn_gru54"
NONE = "titantpp_cnn_gru54_correction_disabled"
CNN = "titantpp_cnn_gru54_cnn_only"
GRU = "titantpp_cnn_gru54_gru_only"
ARMS = (CNN, GRU)
ALL_ARMS = (NONE, CNN, GRU, FULL)
CONTRACT_ID = "titantpp_cnn_gru_components_20261012_v1"
ROLE_BY_ARM = {arm: "observed_time_" + arm.removeprefix("titantpp_") + "_v1" for arm in ARMS}
CNN_PARAMETERS, GRU_PARAMETERS, COMMON_PARAMETERS = 576, 24732, 89859
REGISTERED_PARAMETERS = COMMON_PARAMETERS + CNN_PARAMETERS + GRU_PARAMETERS
ACTIVE_PARAMETERS = {NONE: COMMON_PARAMETERS, CNN: COMMON_PARAMETERS + CNN_PARAMETERS,
    GRU: COMMON_PARAMETERS + GRU_PARAMETERS, FULL: REGISTERED_PARAMETERS}
ACTIVE_CORRECTION_PARAMETERS = {arm: count - COMMON_PARAMETERS for arm, count in ACTIVE_PARAMETERS.items()}


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported CNN/GRU component arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm):
    role_for_arm(arm)
    result = parent_metadata(hidden_dim, arm=FULL)
    cnn, gru = arm == CNN, arm == GRU
    result.update(candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), baseline_backbone=FULL,
        component_control="cnn_only" if cnn else "gru_only",
        cnn_enabled=cnn, gru_correction_enabled=gru,
        correction_kind="none" if cnn else "gru54",
        correction_context="disabled" if cnn else "segment_prefix_recurrent",
        correction_state_scope="none" if cnn else "sample_prefix_segment_forward_only",
        correction_parameter_count=GRU_PARAMETERS if gru else 0,
        cnn_parameter_count=CNN_PARAMETERS if cnn else 0,
        registered_cnn_parameter_count=CNN_PARAMETERS,
        registered_gru_parameter_count=GRU_PARAMETERS,
        registered_correction_parameter_count=CNN_PARAMETERS + GRU_PARAMETERS,
        registered_parameter_count=REGISTERED_PARAMETERS,
        active_trainable_parameter_count=ACTIVE_PARAMETERS[arm],
        active_correction_parameter_count=ACTIVE_CORRECTION_PARAMETERS[arm],
        additional_parameter_count=ACTIVE_CORRECTION_PARAMETERS[arm],
        registered_additional_parameter_count=CNN_PARAMETERS + GRU_PARAMETERS,
        inactive_component="gru" if cnn else "cnn",
        inactive_component_frozen=True, inactive_component_path_bypassed=True,
        common_encoder_initialization="exact_original_CNNGRU54_tensors_and_rng",
        current_encoder_contains_observed_prefix=True,
        explicit_previous_correction_state=gru,
        active_budget_limitations="CNN and GRU active parameter counts differ; no capacity matching",
        cnn_lags="current_lag1_lag2_observed_segment" if cnn else "none",
        gru_layers=1 if gru else 0,
    )
    return result


def identity(hidden_dim=64, *, arm):
    raw = json.dumps(metadata(hidden_dim, arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


class CountAwareTitanCNNGRUComponents(CountAwareTitanCNNGRU):
    """Original full tensors with one correction path disabled by construction."""

    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, component_arm, **kwargs):
        metadata(hidden_dim, arm=component_arm)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, candidate_arm=FULL, **kwargs)
        self.component_arm = component_arm
        if component_arm == GRU:
            self.encoder.layers[0].attn = ControlledObservedAttention(self.encoder.layers[0].attn, disabled=True)
        for name, parameter in correction_parameters(self).items():
            if ((component_arm == CNN and name.startswith("multilag_detail."))
                    or (component_arm == GRU and name in KERNEL_KEYS)):
                parameter.requires_grad_(False)
        self.register_buffer("cnn_gru_component_identity", identity(hidden_dim, arm=component_arm))

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if self.component_arm == GRU:
            # Parent still executes the GRU, while its attention skips CNN.
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
        encoded = self.encoder.input_proj(self.continuous_features(safe_dts, safe_quantities, observed))
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed.unsqueeze(-1), encoded, 0.)
        encoded = self.encoder.layers[0](encoded, mask=observed,
            observed_sources=sources, observed_available=available)
        # No recurrent or MLP correction call in CNN-only.
        return self.encoder.layers[1](encoded, mask=observed)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not _identity_matches(state_dict.get("cnn_gru_component_identity"), identity(arm=self.component_arm)):
            raise ValueError("CNN/GRU component checkpoint identity mismatch")
        if not _identity_matches(state_dict.get("cnn_gru_identity"), parent_identity(arm=FULL)):
            raise ValueError("CNN/GRU component parent identity mismatch")
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    """Bind route, role, inactive path, selector and all original state shapes."""
    meta = payload.get("encoder_config", {})
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    identified = (payload.get("backbone") in ARMS
        or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
        or any(isinstance(state, Mapping) and "cnn_gru_component_identity" in state for state in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("CNN/GRU component checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping):
        raise ValueError("CNN/GRU component metadata must be a mapping")
    required = metadata(meta.get("d_model"), arm=expected_backbone)
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != expected_backbone
            or any(meta.get(key) != value for key, value in required.items())
            or not isinstance(resume, Mapping) or resume.get("backbone") != expected_backbone
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("CNN/GRU component semantics or role mismatch")
    for state in states:
        if (not isinstance(state, Mapping)
                or not _identity_matches(state.get("cnn_gru_component_identity"), identity(arm=expected_backbone))
                or not _identity_matches(state.get("cnn_gru_identity"), parent_identity(arm=FULL))):
            raise ValueError("CNN/GRU component state identity mismatch")
    parent = {**payload, "backbone": FULL,
        "encoder_config": {**meta, **parent_metadata(meta.get("d_model"), arm=FULL)},
        "resume_identity": {**resume, "backbone": FULL,
            "arguments": {**resume.get("arguments", {}), "model_role": parent_role(FULL)}}}
    for name in ("model_state_dict", "best_state_dict"):
        if name in payload:
            parent[name] = {key: value for key, value in payload[name].items() if key != "cnn_gru_component_identity"}
    return parent_validate(parent, FULL)
