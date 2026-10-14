"""One fixed, parameter-free cap on the existing FULL GRU correction.

The parent FULL constructs every learned tensor and consumes exactly its original
RNG. Only its GRU correction output is then limited relative to the detached
block-1 hidden state, before the original residual addition and block-2 path.
CNN, attention, prototypes, heads and objectives are inherited unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn

from models.TPPs.CountAwareTitanCNNGRU import (
    CountAwareTitanCNNGRU, GRUHistoryCorrection, observed_context,
    identity as parent_identity, metadata as parent_metadata,
    role_for_arm as parent_role, validate_checkpoint as parent_validate,
)
from models.TPPs.CountAwareTitanCNNHistoryControls import correction_parameters, _identity_matches
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state

FULL = BASELINE = "titantpp_cnn_gru54"
CNN = "titantpp_cnn_gru54_cnn_only"
CAPPED = "titantpp_cnn_gru54_residual_cap025"
RHO = 0.25
ARMS = (CAPPED,)
ALL_ARMS = (CNN, FULL, CAPPED)
CONTRACT_ID = "titantpp_cnn_gru_residual_cap025_v1"
ROLE_BY_ARM = {CAPPED: "observed_time_cnn_gru54_residual_cap025_v1"}
IDENTITY_BUFFER = "cnn_gru_residual_cap_identity"


def _fixed_rho(rho):
    if type(rho) not in (int, float) or rho != RHO:
        raise ValueError("Residual cap fixes rho=0.25; no tuning or learned scale")


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported GRU residual-cap arm")
    return ROLE_BY_ARM[arm]


def metadata(hidden_dim=64, *, arm=CAPPED, rho=RHO):
    _fixed_rho(rho)
    role_for_arm(arm)
    result = parent_metadata(hidden_dim, arm=FULL)
    result.update(candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), baseline_backbone=FULL,
        residual_cap_enabled=True, residual_cap_rho=RHO,
        residual_cap_scope="each_batch_position_final_hidden_dimension_L2",
        residual_cap_reference="block1_output_immediately_before_GRU_residual_addition",
        residual_cap_radius="rho_times_detached_reference_L2_norm",
        residual_cap_norm_dtype="float64", residual_cap_output_dtype="original_residual_dtype",
        residual_cap_unsaturated="bitwise_identity",
        residual_cap_zero_reference="zero_correction",
        residual_cap_zero_residual="identity_when_reference_norm_positive; finite_valid_gradient",
        residual_cap_denominator="one_for_unsaturated_or_zero_radius; no_epsilon_floor",
        residual_cap_rounding="saturated_overshoot_only; up_to_eight_one_ULP_steps_towards_zero_with_float64_norm_recheck; casting_straight_through_gradient",
        residual_cap_parameter_count=0, residual_cap_learned_scale=False,
        residual_cap_added_normalization=False, residual_cap_added_dropout=False,
        parent_initialization="exact_original_FULL_learned_tensors_and_construction_RNG",
        checkpoint_parent_identity="unchanged_FULL_cnn_gru_identity",
        current_encoder_contains_observed_prefix=True, explicit_previous_correction_state=True,
        cnn_enabled=True, gru_correction_enabled=True)
    return result


def identity(hidden_dim=64, *, arm=CAPPED, rho=RHO):
    raw = json.dumps(metadata(hidden_dim, arm=arm, rho=rho), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


def cap_gru_residual(base_hidden, residual, *, rho=RHO):
    """Preserve direction and cap each token; zero init retains its gradient.

    The detached radius has no gradient path back into the base-state norm.
    The norm denominator is never zero, including inactive ``where`` branches.
    The one-ULP guard only handles floating-point output rounding; the backward
    rule remains that of the scalar norm projection and the ordinary dtype cast.
    """
    _fixed_rho(rho)
    if (not isinstance(base_hidden, torch.Tensor) or not isinstance(residual, torch.Tensor)
            or base_hidden.shape != residual.shape or base_hidden.ndim != 3 or base_hidden.size(-1) != 64
            or base_hidden.device != residual.device or not base_hidden.is_floating_point()
            or not residual.is_floating_point()):
        raise ValueError("Residual cap expects same-device floating [batch,length,64] tensors")
    if not bool(torch.isfinite(base_hidden).all()) or not bool(torch.isfinite(residual).all()):
        raise ValueError("Residual cap received nonfinite hidden state or correction")
    values = residual.to(torch.float64)
    radius = RHO * torch.linalg.vector_norm(base_hidden.detach().to(torch.float64), dim=-1, keepdim=True)
    correction_norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    if not bool(torch.isfinite(radius).all()) or not bool(torch.isfinite(correction_norm).all()):
        raise ValueError("Residual cap norm overflow")
    positive_radius = radius > 0
    saturated = positive_radius & (correction_norm > radius)
    denominator = torch.where(saturated, correction_norm, torch.ones_like(correction_norm))
    scale = torch.where(positive_radius,
        torch.where(saturated, radius / denominator, torch.ones_like(radius)), torch.zeros_like(radius))
    projected = (values * scale).to(residual.dtype)
    # A round-to-nearest dtype cast/reduction can exceed the double radius.
    # Each conservative step changes only saturated overshoot tokens by one ULP;
    # the backward rule remains the original projection/casting rule.
    for _ in range(8):
        overshoot = saturated & (torch.linalg.vector_norm(projected.to(torch.float64), dim=-1, keepdim=True) > radius)
        if not bool(overshoot.any()):
            return projected
        inward = torch.nextafter(projected.detach(), torch.zeros_like(projected))
        inward_with_cast_gradient = projected + (inward - projected).detach()
        projected = torch.where(overshoot, inward_with_cast_gradient, projected)
    if bool((torch.linalg.vector_norm(projected.to(torch.float64), dim=-1, keepdim=True) > radius).any()):
        raise ValueError("Residual cap rounding guard could not satisfy the float64 norm bound")
    return projected


class ResidualCappedGRUCorrection(GRUHistoryCorrection):
    """Reuse the parent's three modules without constructing any new weights."""
    def __init__(self, original, *, rho=RHO):
        _fixed_rho(rho)
        if type(original) is not GRUHistoryCorrection:
            raise ValueError("Residual cap requires the original FULL GRU correction")
        nn.Module.__init__(self)
        self.hidden_dim, self.latent_width = original.hidden_dim, original.latent_width
        self.input_projection = original.input_projection
        self.gru = original.gru
        self.output_projection = original.output_projection
        self.rho = RHO
        self.train(original.training)

    def forward(self, hidden, valid, *, memory_write_mask=None):
        _fixed_rho(self.rho)
        residual = super().forward(hidden, valid, memory_write_mask=memory_write_mask)
        observed, _, _ = observed_context(valid, memory_write_mask=memory_write_mask)
        # The parent permits NaN in masked/padding rows because they are excluded
        # before the GRU; preserve that causal masking policy for the radius.
        reference = torch.where(observed.to(hidden.device).unsqueeze(-1), hidden, 0.)
        return cap_gru_residual(reference, residual, rho=self.rho)


class CountAwareTitanCNNGRUResidualCap(CountAwareTitanCNNGRU):
    """FULL parent architecture plus one nontrainable candidate identity buffer."""
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, residual_cap_arm=CAPPED, rho=RHO, **kwargs):
        metadata(hidden_dim, arm=residual_cap_arm, rho=rho)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, candidate_arm=FULL, **kwargs)
        self.residual_cap_arm = residual_cap_arm
        self.multilag_detail = ResidualCappedGRUCorrection(self.multilag_detail, rho=rho)
        self.register_buffer(IDENTITY_BUFFER, identity(hidden_dim, arm=residual_cap_arm, rho=rho))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if (not _identity_matches(state_dict.get(IDENTITY_BUFFER), identity(arm=self.residual_cap_arm))
                or not _identity_matches(state_dict.get("cnn_gru_identity"), parent_identity(arm=FULL))):
            raise ValueError("GRU residual-cap candidate or parent state identity mismatch")
        _fixed_rho(self.multilag_detail.rho)
        _complete_state(self, state_dict)
        return nn.Module.load_state_dict(self, state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    """Reject candidate relabelling/rho changes, then reuse the FULL validator."""
    meta = payload.get("encoder_config", {})
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    identified = (payload.get("backbone") == CAPPED
        or isinstance(meta, Mapping) and (meta.get("backbone_contract_id") == CONTRACT_ID
            or meta.get("residual_cap_enabled") is True)
        or any(isinstance(state, Mapping) and IDENTITY_BUFFER in state for state in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("GRU residual-cap checkpoint cannot be relabelled as FULL/CNN")
        return False
    if not isinstance(meta, Mapping):
        raise ValueError("GRU residual-cap metadata must be a mapping")
    required = metadata(meta.get("d_model"), arm=expected_backbone)
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != expected_backbone
            or any(meta.get(key) != value for key, value in required.items())
            or not isinstance(resume, Mapping) or resume.get("backbone") != expected_backbone
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("GRU residual-cap semantics, rho or role mismatch")
    for state in states:
        if (not isinstance(state, Mapping)
                or not _identity_matches(state.get(IDENTITY_BUFFER), identity(arm=expected_backbone))
                or not _identity_matches(state.get("cnn_gru_identity"), parent_identity(arm=FULL))):
            raise ValueError("GRU residual-cap state identity mismatch")
    parent = {**payload, "backbone": FULL,
        "encoder_config": {**meta, **parent_metadata(meta.get("d_model"), arm=FULL)},
        "resume_identity": {**resume, "backbone": FULL,
            "arguments": {**resume.get("arguments", {}), "model_role": parent_role(FULL)}}}
    for key in ("model_state_dict", "best_state_dict"):
        if key in payload:
            parent[key] = {name: value for name, value in payload[key].items() if name != IDENTITY_BUFFER}
    return parent_validate(parent, FULL)


__all__ = ["FULL", "CNN", "CAPPED", "RHO", "ARMS", "ALL_ARMS", "CONTRACT_ID", "ROLE_BY_ARM", "IDENTITY_BUFFER",
           "metadata", "identity", "role_for_arm", "correction_parameters", "validate_checkpoint", "cap_gru_residual",
           "ResidualCappedGRUCorrection", "CountAwareTitanCNNGRUResidualCap"]
