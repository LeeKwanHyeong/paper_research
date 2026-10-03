"""Bounded, sample-local affine state transport between B's encoder blocks.

The two arms differ only in their explicit transition clock. The parallel scan
uses Hillis--Steele: O(L log L) work/depth O(log L), not a linear-work scan.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP, LOG_MSE_VARIANT, TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_QUANTITY_GRADIENT_SHARED,
    TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
)

EVENT_STATE_BACKBONE = "titantpp_event_state_transport"
ELAPSED_STATE_BACKBONE = "titantpp_elapsed_state_transport"
STATE_TRANSPORT_BACKBONES = (EVENT_STATE_BACKBONE, ELAPSED_STATE_BACKBONE)
STATE_TRANSPORT_CONTRACT_ID = "hard_lmm_state_transport_v1"
STATE_TRANSPORT_ROLE = "observed_time_state_transport_v1"
_CLOCKS = {"event": EVENT_STATE_BACKBONE, "elapsed": ELAPSED_STATE_BACKBONE}
_SCANS = ("sequential", "parallel")


def state_transport_metadata(hidden_dim: int, clock: str, rank: int = 8,
                             scan: str = "parallel", time_scale: float = 3.0) -> dict[str, Any]:
    if type(hidden_dim) is not int or hidden_dim < 4 or hidden_dim % 4:
        raise ValueError("State transport hidden_dim must be a positive multiple of four")
    if clock not in _CLOCKS or scan not in _SCANS:
        raise ValueError("State transport requires event/elapsed clock and sequential/parallel scan")
    if type(rank) is not int or rank != 8:
        raise ValueError("State transport v1 fixes rank=8")
    if isinstance(time_scale, bool) or not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("State transport requires a finite positive train time scale")
    return {
        "candidate_name": _CLOCKS[clock], "backbone_contract_id": STATE_TRANSPORT_CONTRACT_ID,
        "model_role": STATE_TRANSPORT_ROLE, "base_encoder": "CountAwareTitanTPP",
        "d_model": hidden_dim, "n_layers": 2, "n_heads": 4, "d_ff": 2 * hidden_dim,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD, "persistent_mem_size": 16,
        "lmm_mem_size": 64, "lmm_topk": 4, "static_retrieval_aggregation": "arithmetic_mean",
        "static_key_value_tied": True, "state_transport_clock": clock,
        "state_transport_rank": rank, "state_transport_scan": scan,
        "state_transport_scan_work": "L_log_L" if scan == "parallel" else "L",
        "state_transport_time_scale": float(time_scale),
        "state_transport_insertion": "between_encoder_blocks",
        "state_transport_initial_write_gate": 1.0 / 16.0,
        "state_transport_event_free_half_lives": [2 ** k for k in range(rank)],
        "state_transport_initialization": "zero_output_and_gate_weight_forked_rng",
        "state_transport_observation_mask": "sanitize_before_features_padding_identity_withheld_reset",
        "state_transport_state_lifetime": "sample_local_forward_only",
        "additional_parameter_count": 3 * hidden_dim * rank + 2 * rank,
        "quantity_objective": LOG_MSE_VARIANT, "online_writes": False,
    }


def _mask(mask: torch.Tensor, shape: tuple[int, int], name: str) -> None:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")


def affine_state_scan(a: torch.Tensor, b: torch.Tensor, *, scan: str = "parallel") -> torch.Tensor:
    """Return all states of s_i=a_i*s_(i-1)+b_i with s_start=0.

    Inputs are diagonal affine coefficients [batch, length, state]. Out-of-place
    composition preserves autograd for zero coefficients (resets and padding).
    """
    if scan not in _SCANS or a.ndim != 3 or a.shape != b.shape:
        raise ValueError("Expected matching [batch, length, state] affine coefficients")
    if a.size(1) == 0:
        return b.clone()
    if scan == "sequential":
        state = torch.zeros_like(b[:, 0])
        states = []
        for i in range(a.size(1)):
            state = a[:, i] * state + b[:, i]
            states.append(state)
        return torch.stack(states, dim=1)
    prefix_a, prefix_b = a, b
    offset = 1
    while offset < a.size(1):
        prefix_b = torch.cat((prefix_b[:, :offset],
                              prefix_b[:, offset:] + prefix_a[:, offset:] * prefix_b[:, :-offset]), dim=1)
        prefix_a = torch.cat((prefix_a[:, :offset],
                              prefix_a[:, offset:] * prefix_a[:, :-offset]), dim=1)
        offset *= 2
    return prefix_b


class BoundedStateTransport(nn.Module):
    def __init__(self, hidden_dim: int, *, clock: str, rank: int = 8,
                 scan: str = "parallel", time_scale: float = 3.0) -> None:
        super().__init__()
        metadata = state_transport_metadata(hidden_dim, clock, rank, scan, time_scale)
        self.hidden_dim, self.rank = hidden_dim, rank
        self.clock, self.scan, self.time_scale = clock, scan, float(time_scale)
        self.scan_mode = scan
        # No extra initialization draws escape into the shared B/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.value_projection = nn.Linear(hidden_dim, rank, bias=False)
            self.gate_projection = nn.Linear(hidden_dim, rank, bias=True)
            self.output_projection = nn.Linear(rank, hidden_dim, bias=False)
            nn.init.zeros_(self.gate_projection.weight)
            nn.init.constant_(self.gate_projection.bias, math.log(1.0 / 15.0))
            nn.init.zeros_(self.output_projection.weight)
            rate = math.log(2.0) / (2.0 ** torch.arange(rank))
            self.rho = nn.Parameter(torch.log(torch.expm1(rate)))
        identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).digest()
        self.register_buffer("contract_identity", torch.tensor(list(identity), dtype=torch.uint8))
        self.register_buffer("clock_code", torch.tensor(0 if clock == "event" else 1, dtype=torch.int64))
        self._expected_identity = identity

    def validate_identity(self, state_dict, prefix: str = "") -> None:
        value = state_dict.get(prefix + "contract_identity")
        code = state_dict.get(prefix + "clock_code")
        expected = torch.tensor(list(self._expected_identity), dtype=torch.uint8)
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                or not torch.equal(value.detach().cpu(), expected)
                or not isinstance(code, torch.Tensor) or code.dtype != torch.int64
                or code.shape != torch.Size([]) or code.item() != (0 if self.clock == "event" else 1)):
            raise RuntimeError("State transport clock/scale/rank/scan architecture identity mismatch")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.validate_identity(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def read(self, hidden: torch.Tensor, dts: torch.Tensor, valid: torch.Tensor, *,
             memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.size(-1) != self.hidden_dim:
            raise ValueError("hidden must have shape [batch, length, hidden_dim]")
        shape = tuple(hidden.shape[:2])
        _mask(valid, shape, "valid")
        if tuple(dts.shape) != shape:
            raise ValueError("dts must have shape [batch, length]")
        valid = valid.to(hidden.device)
        observed = valid
        if memory_write_mask is not None:
            _mask(memory_write_mask, shape, "memory_write_mask")
            observed = valid & memory_write_mask.to(hidden.device)
        safe_hidden = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        safe_dts = torch.where(observed, dts.to(hidden.device), 0.0)
        if not bool(torch.isfinite(safe_dts).all()) or bool((safe_dts < 0).any()):
            raise ValueError("Observed durations must be finite and nonnegative")
        u = F.layer_norm(safe_hidden, (self.hidden_dim,), eps=1e-5)
        value = torch.tanh(self.value_projection(u))
        gate = torch.sigmoid(self.gate_projection(u))
        duration = safe_dts.to(hidden.dtype) / self.time_scale
        if self.clock == "event":
            duration = torch.ones_like(duration)
        decay = torch.exp(-F.softplus(self.rho) * duration.unsqueeze(-1))
        a = (1.0 - gate) * decay
        b = gate * value
        # Padding is an identity; an unobserved valid event resets this path.
        a = torch.where(observed.unsqueeze(-1), a, (~valid).unsqueeze(-1).to(a.dtype))
        b = torch.where(observed.unsqueeze(-1), b, 0.0)
        return affine_state_scan(a, b, scan=self.scan)

    def forward(self, hidden: torch.Tensor, dts: torch.Tensor, valid: torch.Tensor, *,
                memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        state = self.read(hidden, dts, valid, memory_write_mask=memory_write_mask)
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        return torch.where(observed.to(hidden.device).unsqueeze(-1), self.output_projection(state), 0.0)


class CountAwareTitanStateTransport(CountAwareTitanTPP):
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, *,
                 state_transport_clock: str = "elapsed", state_transport_rank: int = 8,
                 state_transport_scan: str = "parallel", **quantity_kwargs: Any) -> None:
        time_scale = quantity_kwargs.get("time_scale", 3.0)
        state_transport_metadata(hidden_dim, state_transport_clock, state_transport_rank,
                                 state_transport_scan, time_scale)
        if (quantity_kwargs.get("time_head_mode") != TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION
                or not isinstance(quantity_kwargs.get("time_observation_contract"), Mapping)
                or quantity_kwargs["time_observation_contract"].get("mode") != "positive_integer_round_clamp_v1"):
            raise ValueError("State transport fixes the recorded-duration heteroscedastic lognormal time head")
        for name, expected in (("memory_mode", TITAN_MEMORY_MODE_STATIC_HARD),
                               ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
                               ("quantity_variant", LOG_MSE_VARIANT), ("lambda_tail", 0.0),
                               ("quantile_adaptive_strength", 0.0),
                               ("quantile_adaptive_boundaries", None), ("quantile_adaptive_weights", None)):
            if quantity_kwargs.get(name, expected) != expected:
                raise ValueError(f"State transport fixes {name}={expected!r}")
        quantity_kwargs.pop("memory_mode", None)
        super().__init__(hidden_dim, train_log_mean, max_seq_len,
                         memory_mode=TITAN_MEMORY_MODE_STATIC_HARD, **quantity_kwargs)
        self.state_transport = BoundedStateTransport(
            hidden_dim, clock=state_transport_clock, rank=state_transport_rank,
            scan=state_transport_scan, time_scale=self.time_scale)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Validate before the superclass can mutate any shared B parameter.
        self.state_transport.validate_identity(state_dict, "state_transport.")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch, length]")
        _mask(mask, tuple(dts.shape), "mask")
        valid = mask.to(dts.device)
        observed = valid
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(dts.shape), "memory_write_mask")
            observed = valid & memory_write_mask.to(dts.device)
        safe_dts = torch.where(observed, dts, 0.0)
        safe_quantities = torch.where(observed, history_quantities, 0.0)
        if (not bool(torch.isfinite(safe_dts).all()) or bool((safe_dts < 0).any())
                or not bool(torch.isfinite(safe_quantities).all()) or bool((safe_quantities < 0).any())):
            raise ValueError("Observed durations and quantities must be finite and nonnegative")
        if self.encoder is None or self.lmm is None or len(self.encoder.layers) != 2:
            raise RuntimeError("State transport requires B's two blocks and static Hard-LMM")
        features = self.continuous_features(safe_dts, safe_quantities, observed)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed.unsqueeze(-1), encoded, 0.0)
        encoded = self.encoder.layers[0](encoded, mask=observed)
        encoded = encoded + self.state_transport(encoded, safe_dts, valid, memory_write_mask=observed)
        return self.encoder.layers[1](encoded, mask=observed)

    def encode_task_states(self, dts, history_quantities, mask, *, memory_write_mask=None):
        base = self._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        observed = mask if memory_write_mask is None else mask & memory_write_mask
        encoded = torch.where(observed.to(base.device).unsqueeze(-1), self.lmm(base), 0.0)
        return encoded, encoded


def validate_state_transport_checkpoint(payload: Mapping[str, Any], expected_backbone: str) -> bool:
    """Reject relabelled arms and inconsistent model/interface/resume metadata.

    The execution runner additionally verifies frozen source and data checksums.
    Summaries are allowed only with an explicit checkpoint digest and best epoch.
    """
    metadata = payload.get("encoder_config", {})
    states = [payload[name] for name in ("model_state_dict", "best_state_dict") if name in payload]
    identified = (payload.get("backbone") in STATE_TRANSPORT_BACKBONES
                  or isinstance(metadata, Mapping) and (
                      any(str(k).startswith("state_transport_") for k in metadata)
                      or metadata.get("backbone_contract_id") == STATE_TRANSPORT_CONTRACT_ID
                      or metadata.get("model_role") == STATE_TRANSPORT_ROLE
                      or metadata.get("candidate_name") in STATE_TRANSPORT_BACKBONES)
                  or any(isinstance(s, Mapping) and any(str(k).startswith("state_transport.") for k in s)
                         for s in states))
    if expected_backbone not in STATE_TRANSPORT_BACKBONES:
        if identified:
            raise ValueError("State transport checkpoint cannot be relabelled as another backbone")
        return False
    if not isinstance(metadata, Mapping):
        raise ValueError("State transport encoder metadata must be a mapping")
    clock = next(k for k, v in _CLOCKS.items() if v == expected_backbone)
    head = metadata.get("time_head", {})
    if not isinstance(head, Mapping):
        raise ValueError("State transport requires a time-head mapping")
    scale = head.get("time_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("State transport checkpoint time scale is invalid")
    expected = state_transport_metadata(metadata.get("d_model"), clock,
                                        rank=metadata.get("state_transport_rank"),
                                        scan=metadata.get("state_transport_scan"), time_scale=scale)
    if payload.get("backbone") != expected_backbone or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("State transport checkpoint architecture metadata mismatch")
    if (payload.get("variant") != LOG_MSE_VARIANT or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or head.get("mode") != TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION
            or head.get("reported_metric") != "recorded_positive_integer_time_nll"
            or head.get("density_family") != "heteroscedastic_lognormal_on_scaled_duration"
            or head.get("time_location_transform") != "identity"
            or head.get("time_scale_conditioning") != "linear_hidden"
            or head.get("time_scale_transform") != "softplus_plus_floor"
            or head.get("time_scale_weight_initialization") != "zeros" or head.get("wd_clamp") != 0.0):
        raise ValueError("State transport checkpoint scope/objective/time-head mismatch")
    for field in ("time_initial_location", "time_initial_scale", "time_sigma_floor"):
        value = head.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("State transport time-head initialization is invalid")
    if not 0 < head["time_sigma_floor"] < head["time_initial_scale"]:
        raise ValueError("State transport time-head scale floor is invalid")
    interface, identity = payload.get("interface_meta"), payload.get("resume_identity")
    if not isinstance(interface, Mapping) or not isinstance(identity, Mapping):
        raise ValueError("State transport requires interface and resume identity")
    arguments, interface_head = identity.get("arguments", {}), interface.get("time_head", {})
    if not isinstance(arguments, Mapping) or not isinstance(interface_head, Mapping):
        raise ValueError("State transport interface/resume mappings are invalid")
    units = {"intermittent_frozen_5000": "week", "yellow_trip_hourly": "hour"}
    dataset = arguments.get("dataset_contract")
    if not isinstance(dataset, str) or dataset not in units:
        raise ValueError("State transport v1 dataset must be Intermittent or Taxi")
    observation = {"mode": "positive_integer_round_clamp_v1", "unit": units[dataset], "top_code": None}
    if (head.get("observation_likelihood") != observation
            or interface_head.get("observation_likelihood") != observation
            or interface_head.get("statistics_source_split") != "train"
            or interface_head.get("mode") != head["mode"]
            or identity.get("interface_meta") != interface or identity.get("backbone") != expected_backbone
            or identity.get("variant") != LOG_MSE_VARIANT
            or identity.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
            or arguments.get("model_role") != STATE_TRANSPORT_ROLE
            or arguments.get("time_head_mode") != head["mode"] or arguments.get("time_scale") != scale):
        raise ValueError("State transport observation/resume identity mismatch")
    statistics = interface_head.get("train_time_statistics", {})
    if not isinstance(statistics, Mapping):
        raise ValueError("State transport train-time statistics are invalid")
    for field, statistic in (("time_scale", "time_scale"), ("time_initial_location", "target_log_scaled_mean"),
                             ("time_initial_scale", "target_log_scaled_std")):
        value = statistics.get(statistic)
        # Frozen encoder/interface settings stay exact. Recomputed train
        # statistics use the same absolute tolerance as the execution runner.
        if (interface_head.get(field) != head[field]
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(value, head[field], rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("State transport train-only initialization mismatch")
    if not states:
        digest = payload.get("checkpoint_state_sha256")
        if (isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                and type(payload.get("best_epoch")) is int and payload["best_epoch"] >= 1):
            return True
        raise ValueError("State transport requires model state or a valid summary binding")
    expected_identity = torch.tensor(list(hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).digest()),
                                     dtype=torch.uint8)
    dim, rank = metadata["d_model"], metadata["state_transport_rank"]
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("State transport state must be a mapping")
        code, digest = state.get("state_transport.clock_code"), state.get("state_transport.contract_identity")
        if (not isinstance(code, torch.Tensor) or code.dtype != torch.int64 or code.shape != torch.Size([])
                or code.item() != (0 if clock == "event" else 1)
                or not isinstance(digest, torch.Tensor) or digest.dtype != torch.uint8
                or not torch.equal(digest.detach().cpu(), expected_identity)):
            raise ValueError("State transport tensor identity mismatch")
        for name, shape in (("value_projection.weight", (rank, dim)), ("gate_projection.weight", (rank, dim)),
                            ("gate_projection.bias", (rank,)), ("output_projection.weight", (dim, rank)),
                            ("rho", (rank,))):
            tensor = state.get("state_transport." + name)
            if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
                    or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all())):
                raise ValueError("State transport learned tensor is missing or invalid")
    return True


__all__ = ["CountAwareTitanStateTransport", "BoundedStateTransport", "affine_state_scan",
           "state_transport_metadata", "validate_state_transport_checkpoint", "EVENT_STATE_BACKBONE", "ELAPSED_STATE_BACKBONE",
           "STATE_TRANSPORT_BACKBONES", "STATE_TRANSPORT_CONTRACT_ID", "STATE_TRANSPORT_ROLE"]
