"""A width-only intervention on the observed-time History-MLP correction."""
from __future__ import annotations

import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTitanCoreAblation import (
    CountAwareTitanCoreAblation, metadata as base_metadata,
    identity as base_identity,
)
from models.TPPs.CountAwareTitanMultiLagDetail import (
    _complete_state, _mask, lag_source_indices,
)

ARM = "titantpp_history_mlp_width16"
ARMS = (ARM,)
ROLE = "observed_time_history_mlp_width16_v1"
WIDTH = 16


def metadata(hidden_dim=64):
    if hidden_dim != 64:
        raise ValueError("History width comparison fixes hidden_dim=64")
    result = base_metadata(hidden_dim, "mlp")
    result.update(
        candidate_name=ARM, backbone_contract_id="titantpp_history_width_v1",
        model_role=ROLE, history_mlp_width=WIDTH, bottleneck_width=WIDTH,
        multilag_detail_rank=WIDTH,
        multilag_detail_projections="concat_current_previous_bias_free_mlp_width16",
        additional_parameter_count=8 * (2 * hidden_dim * WIDTH + WIDTH * hidden_dim),
        branch_availability="original_thresholds", residual_divisor=8,
        history_context="immediate_predecessor",
        initialization="default_Linear_input_zero_output_forked_cpu_rng",
    )
    return result


def identity(hidden_dim=64):
    encoded = json.dumps(metadata(hidden_dim), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


class HistoryWidthCorrection(nn.Module):
    """Preserve the width4 operation order, branch masks and fixed divisor."""
    def __init__(self, hidden_dim=64, *, width=WIDTH):
        super().__init__()
        if hidden_dim != 64 or width not in (4, WIDTH):
            raise ValueError("Only hidden_dim64 and diagnostic width4/candidate16 are supported")
        self.hidden_dim, self.width = hidden_dim, width
        # Includes Linear's discarded output initialization: preserve the prior
        # initialization procedure while leaving the common-model RNG unchanged.
        with torch.random.fork_rng(devices=[]):
            self.input_projections = nn.ModuleList(
                nn.Linear(2 * hidden_dim, width, bias=False) for _ in range(8))
            self.output_projections = nn.ModuleList(
                nn.Linear(width, hidden_dim, bias=False) for _ in range(8))
            for projection in self.output_projections:
                nn.init.zeros_(projection.weight)

    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.)
        if not torch.isfinite(safe).all():
            raise ValueError("Nonfinite observed history")
        source, available = lag_source_indices(
            valid, memory_write_mask=memory_write_mask, mode="local")
        weight = torch.cat([p.weight for p in self.input_projections])
        current = F.linear(safe, weight[:, :self.hidden_dim]).unflatten(-1, (8, self.width))
        previous = F.linear(safe, weight[:, self.hidden_dim:]).unflatten(-1, (8, self.width))
        features = current + previous.gather(
            1, source.unsqueeze(-1).expand(-1, -1, -1, self.width))
        features = F.gelu(torch.where(available.unsqueeze(-1), features, 0.), approximate="none")
        return F.linear(features.flatten(-2), torch.cat(
            [p.weight for p in self.output_projections], dim=1)) / 8


class CountAwareTitanHistoryWidth(CountAwareTitanCoreAblation):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, **kwargs):
        metadata(hidden_dim)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, core_mode="mlp", **kwargs)
        self.multilag_detail = HistoryWidthCorrection(hidden_dim)
        self.register_buffer("history_width_identity", identity(hidden_dim))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get("history_width_identity")
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(self.core_hidden_dim)):
            raise ValueError("History width checkpoint identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") == ARM or "history_mlp_width" in meta
                  or any("history_width_identity" in state for state in states))
    if expected_backbone != ARM:
        if identified:
            raise ValueError("History width candidate cannot be relabelled")
        return False
    expected = metadata(meta.get("d_model"))
    interface, resume = payload.get("interface_meta", {}), payload.get("resume_identity", {})
    head = meta.get("time_head", {})
    if (payload.get("backbone") != ARM or any(meta.get(k) != v for k, v in expected.items())
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or payload.get("variant") != "count_only_log_regression"
            or resume.get("backbone") != ARM or resume.get("interface_meta") != interface
            or resume.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
            or resume.get("arguments", {}).get("model_role") != ROLE
            or head.get("mode") != "heteroscedastic_lognormal_duration"
            or not isinstance(head.get("observation_likelihood"), dict)
            or head["observation_likelihood"].get("mode") != "positive_integer_round_clamp_v1"
            or head["observation_likelihood"] != interface.get("time_head", {}).get("observation_likelihood")):
        raise ValueError("History width metadata, split, head, selector or role mismatch")
    for state in states:
        for key, expected_value in (("history_width_identity", identity()),
                                    ("core_ablation_identity", base_identity(64, "mlp"))):
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), expected_value):
                raise ValueError("History width state identity mismatch")
        for branch in range(8):
            for name, shape in (("input", (WIDTH, 128)), ("output", (64, WIDTH))):
                value = state.get(f"multilag_detail.{name}_projections.{branch}.weight")
                if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                    raise ValueError("History width projection shape mismatch")
    if not states and (len(payload.get("checkpoint_state_sha256", "")) != 64
                       or payload.get("best_epoch", 0) < 1):
        raise ValueError("Unbound history width summary")
    return True
