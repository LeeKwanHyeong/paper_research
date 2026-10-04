"""Causal, sample-local level/detail residual between B's encoder blocks.

The candidate reads eight dyadic observed-event distances; the same-capacity
control reads lag one with the candidate's exact branch availability masks.
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
    TITAN_QUANTITY_GRADIENT_SHARED, TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION,
)

LOCAL_DETAIL_BACKBONE = "titantpp_local_detail"
MULTILAG_DETAIL_BACKBONE = "titantpp_multilag_detail"
MULTILAG_DETAIL_BACKBONES = (LOCAL_DETAIL_BACKBONE, MULTILAG_DETAIL_BACKBONE)
MULTILAG_DETAIL_CONTRACT_ID = "hard_lmm_multilag_detail_v1"
MULTILAG_DETAIL_ROLE = "observed_time_multilag_detail_v1"
AVAILABILITY_LAGS = (1, 2, 4, 8, 16, 32, 64, 128)
_MODES = {"local": LOCAL_DETAIL_BACKBONE, "multilag": MULTILAG_DETAIL_BACKBONE}


def multilag_detail_metadata(hidden_dim: int, mode: str, rank: int = 4) -> dict[str, Any]:
    if type(hidden_dim) is not int or hidden_dim < 4 or hidden_dim % 4:
        raise ValueError("Multilag detail hidden_dim must be a positive multiple of four")
    if mode not in _MODES or type(rank) is not int or rank != 4:
        raise ValueError("Multilag detail v1 requires local/multilag mode and rank=4")
    return {
        "candidate_name": _MODES[mode], "backbone_contract_id": MULTILAG_DETAIL_CONTRACT_ID,
        "model_role": MULTILAG_DETAIL_ROLE, "base_encoder": "CountAwareTitanTPP",
        "d_model": hidden_dim, "n_layers": 2, "n_heads": 4, "d_ff": 2 * hidden_dim,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD, "persistent_mem_size": 16,
        "lmm_mem_size": 64, "lmm_topk": 4, "static_retrieval_aggregation": "arithmetic_mean",
        "static_key_value_tied": True, "multilag_detail_mode": mode,
        "multilag_detail_rank": rank, "multilag_detail_branch_count": 8,
        "multilag_detail_lags": list(AVAILABILITY_LAGS if mode == "multilag" else (1,) * 8),
        "multilag_detail_availability_lags": list(AVAILABILITY_LAGS),
        "multilag_detail_residual_divisor": 8,
        "multilag_detail_activation": "gelu_exact_erf",
        "multilag_detail_projections": "independent_bias_free_level_detail_output",
        "multilag_detail_initialization": "ordinary_level_detail_zero_output_forked_rng",
        "multilag_detail_insertion": "between_encoder_blocks",
        "multilag_detail_mask": "observed_count_padding_skipped_withheld_segment_reset",
        "multilag_detail_implementation": "project_before_gather",
        "multilag_detail_lifetime": "sample_local_forward_only",
        "additional_parameter_count": 8 * 3 * hidden_dim * rank,
        "quantity_objective": LOG_MSE_VARIANT, "online_writes": False,
    }


def _mask(mask: torch.Tensor, shape: tuple[int, int], name: str) -> None:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")


def _complete_state(module: nn.Module, state_dict: Mapping[str, Any]) -> None:
    """Reject incomplete substitutions before PyTorch mutates any parameter."""
    expected = module.state_dict()
    if set(state_dict) != set(expected):
        raise RuntimeError("Multilag detail requires a complete exact state mapping")
    for name, target in expected.items():
        value = state_dict[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                or value.dtype != target.dtype or not bool(torch.isfinite(value).all())):
            raise RuntimeError("Multilag detail state tensor mismatch: " + name)


def lag_source_indices(valid: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None,
                       mode: str = "multilag") -> tuple[torch.Tensor, torch.Tensor]:
    """Return safe source indices and shared eligibility [batch,length,8].

    Prefix counts and eight predecessor-doubling gathers use O(batch*length*8)
    integer work/storage. Padding consumes no event; withheld valid rows reset
    eligibility. No query-event matrix, host loop over events or sorting is used.
    """
    if not isinstance(valid, torch.Tensor) or valid.ndim != 2:
        raise ValueError("valid must have shape [batch, length]")
    _mask(valid, tuple(valid.shape), "valid")
    if mode not in _MODES:
        raise ValueError("Unsupported detail mode")
    observed = valid
    if memory_write_mask is not None:
        _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
        observed = valid & memory_write_mask.to(valid.device)
    batch, length = valid.shape
    if length == 0:
        shape = (batch, 0, 8)
        return (torch.zeros(shape, device=valid.device, dtype=torch.long),
                torch.zeros(shape, device=valid.device, dtype=torch.bool))
    count = observed.long().cumsum(dim=1)
    reset_count = torch.where(valid & ~observed, count, 0).cummax(dim=1).values
    local_count = count - reset_count
    lags = torch.tensor(AVAILABILITY_LAGS, device=valid.device)
    available = observed.unsqueeze(-1) & (local_count.unsqueeze(-1) > lags)
    positions = torch.arange(length, device=valid.device).expand(batch, -1)
    latest = torch.where(observed, positions, -1).cummax(dim=1).values
    predecessor = torch.cat((torch.full((batch, 1), -1, device=valid.device, dtype=torch.long),
                             latest[:, :-1]), dim=1)
    sources = [predecessor]
    for _ in range(7):
        previous = sources[-1]
        doubled = previous.gather(1, previous.clamp_min(0))
        sources.append(torch.where(previous >= 0, doubled, -1))
    source = torch.stack(sources if mode == "multilag" else [predecessor] * 8, dim=-1)
    return torch.where(available, source, 0), available


class MultiLagDetail(nn.Module):
    def __init__(self, hidden_dim: int, *, mode: str = "multilag", rank: int = 4) -> None:
        super().__init__()
        metadata = multilag_detail_metadata(hidden_dim, mode, rank)
        self.hidden_dim, self.mode, self.rank = hidden_dim, mode, rank
        self.availability_lags = AVAILABILITY_LAGS
        self.lags = AVAILABILITY_LAGS if mode == "multilag" else (1,) * 8
        self.branch_count, self.residual_divisor = 8, 8
        with torch.random.fork_rng(devices=[]):
            self.level_projections = nn.ModuleList(nn.Linear(hidden_dim, rank, bias=False) for _ in range(8))
            self.detail_projections = nn.ModuleList(nn.Linear(hidden_dim, rank, bias=False) for _ in range(8))
            self.output_projections = nn.ModuleList(nn.Linear(rank, hidden_dim, bias=False) for _ in range(8))
            for projection in self.output_projections:
                nn.init.zeros_(projection.weight)
        identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).digest()
        self._expected_identity = identity
        self.register_buffer("contract_identity", torch.tensor(list(identity), dtype=torch.uint8))
        self.register_buffer("mode_code", torch.tensor(0 if mode == "local" else 1, dtype=torch.int64))

    def validate_identity(self, state_dict, prefix: str = "") -> None:
        digest, code = state_dict.get(prefix + "contract_identity"), state_dict.get(prefix + "mode_code")
        expected = torch.tensor(list(self._expected_identity), dtype=torch.uint8)
        if (not isinstance(digest, torch.Tensor) or digest.dtype != torch.uint8
                or not torch.equal(digest.detach().cpu(), expected)
                or not isinstance(code, torch.Tensor) or code.dtype != torch.int64
                or code.shape != torch.Size([]) or code.item() != (0 if self.mode == "local" else 1)):
            raise RuntimeError("Multilag detail architecture identity mismatch")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.validate_identity(state_dict)
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, hidden: torch.Tensor, valid: torch.Tensor, *,
                memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.size(-1) != self.hidden_dim:
            raise ValueError("hidden must have shape [batch, length, hidden_dim]")
        _mask(valid, tuple(hidden.shape[:2]), "valid")
        valid = valid.to(hidden.device)
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(valid.shape), "memory_write_mask")
            memory_write_mask = memory_write_mask.to(hidden.device)
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        if not bool(torch.isfinite(safe).all()):
            raise ValueError("Observed hidden states must be finite")
        sources, available = lag_source_indices(valid, memory_write_mask=memory_write_mask, mode=self.mode)
        # Pack views of the existing 24 independent parameters for two GEMMs.
        # No parameter, checkpoint key, initialization or architecture identity
        # changes. Autograd distributes packed gradients back to each branch.
        input_weight = torch.cat([p.weight for p in self.level_projections]
                                 + [p.weight for p in self.detail_projections], dim=0)
        projected = F.linear(safe, input_weight).unflatten(-1, (2, self.branch_count, self.rank))
        index = sources.unsqueeze(2).unsqueeze(-1).expand(-1, -1, 2, -1, self.rank)
        past = projected.gather(1, index)
        current_l, current_d = projected.unbind(dim=2)
        past_l, past_d = past.unbind(dim=2)
        eligible = available.unsqueeze(-1)
        projected_l = torch.where(eligible, (current_l + past_l) * .5, 0.0)
        projected_d = torch.where(eligible, current_d - past_d, 0.0)
        features = F.gelu(projected_l, approximate="none") + F.gelu(projected_d, approximate="none")
        output_weight = torch.cat([p.weight for p in self.output_projections], dim=1)
        correction = F.linear(features.flatten(-2), output_weight)
        return correction / self.residual_divisor


class CountAwareTitanMultiLagDetail(CountAwareTitanTPP):
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, *,
                 multilag_detail_mode: str = "multilag", multilag_detail_rank: int = 4,
                 **quantity_kwargs: Any) -> None:
        multilag_detail_metadata(hidden_dim, multilag_detail_mode, multilag_detail_rank)
        if (quantity_kwargs.get("time_head_mode") != TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION
                or not isinstance(quantity_kwargs.get("time_observation_contract"), Mapping)
                or quantity_kwargs["time_observation_contract"].get("mode") != "positive_integer_round_clamp_v1"):
            raise ValueError("Multilag detail fixes the recorded-duration heteroscedastic lognormal head")
        for name, expected in (("memory_mode", TITAN_MEMORY_MODE_STATIC_HARD),
                               ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
                               ("quantity_variant", LOG_MSE_VARIANT), ("lambda_tail", 0.0),
                               ("quantile_adaptive_strength", 0.0),
                               ("quantile_adaptive_boundaries", None), ("quantile_adaptive_weights", None)):
            if quantity_kwargs.get(name, expected) != expected:
                raise ValueError(f"Multilag detail fixes {name}={expected!r}")
        quantity_kwargs.pop("memory_mode", None)
        super().__init__(hidden_dim, train_log_mean, max_seq_len,
                         memory_mode=TITAN_MEMORY_MODE_STATIC_HARD, **quantity_kwargs)
        self.multilag_detail = MultiLagDetail(hidden_dim, mode=multilag_detail_mode, rank=multilag_detail_rank)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.multilag_detail.validate_identity(state_dict, "multilag_detail.")
        _complete_state(self, state_dict)
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
            raise RuntimeError("Multilag detail requires B's two blocks and static Hard-LMM")
        features = self.continuous_features(safe_dts, safe_quantities, observed)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        encoded = torch.where(observed.unsqueeze(-1), encoded, 0.0)
        encoded = self.encoder.layers[0](encoded, mask=observed)
        encoded = encoded + self.multilag_detail(encoded, valid, memory_write_mask=observed)
        return self.encoder.layers[1](encoded, mask=observed)

    def encode_task_states(self, dts, history_quantities, mask, *, memory_write_mask=None):
        base = self._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        observed = mask if memory_write_mask is None else mask & memory_write_mask
        encoded = torch.where(observed.to(base.device).unsqueeze(-1), self.lmm(base), 0.0)
        return encoded, encoded


def validate_multilag_detail_checkpoint(payload: Mapping[str, Any], expected_backbone: str) -> bool:
    """Reject relabelled arms and inconsistent model/interface/resume metadata.

    The execution runner additionally verifies frozen source and data checksums.
    Summaries are allowed only with an explicit checkpoint digest and best epoch.
    """
    metadata = payload.get("encoder_config", {})
    states = [payload[name] for name in ("model_state_dict", "best_state_dict") if name in payload]
    identified = (payload.get("backbone") in MULTILAG_DETAIL_BACKBONES
                  or isinstance(metadata, Mapping) and (
                      any(str(k).startswith("multilag_detail_") for k in metadata)
                      or metadata.get("backbone_contract_id") == MULTILAG_DETAIL_CONTRACT_ID
                      or metadata.get("model_role") == MULTILAG_DETAIL_ROLE
                      or metadata.get("candidate_name") in MULTILAG_DETAIL_BACKBONES)
                  or any(isinstance(s, Mapping) and any(str(k).startswith("multilag_detail.") for k in s)
                         for s in states))
    if expected_backbone not in MULTILAG_DETAIL_BACKBONES:
        if identified:
            raise ValueError("Multilag detail checkpoint cannot be relabelled as another backbone")
        return False
    if not isinstance(metadata, Mapping):
        raise ValueError("Multilag detail encoder metadata must be a mapping")
    mode = next(k for k, v in _MODES.items() if v == expected_backbone)
    head = metadata.get("time_head", {})
    if not isinstance(head, Mapping):
        raise ValueError("Multilag detail requires a time-head mapping")
    scale = head.get("time_scale")
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Multilag detail checkpoint time scale is invalid")
    expected = multilag_detail_metadata(metadata.get("d_model"), mode,
                                        rank=metadata.get("multilag_detail_rank"))
    if payload.get("backbone") != expected_backbone or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Multilag detail checkpoint architecture metadata mismatch")
    if (payload.get("variant") != LOG_MSE_VARIANT or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or head.get("mode") != TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION
            or head.get("reported_metric") != "recorded_positive_integer_time_nll"
            or head.get("density_family") != "heteroscedastic_lognormal_on_scaled_duration"
            or head.get("time_location_transform") != "identity"
            or head.get("time_scale_conditioning") != "linear_hidden"
            or head.get("time_scale_transform") != "softplus_plus_floor"
            or head.get("time_scale_weight_initialization") != "zeros" or head.get("wd_clamp") != 0.0):
        raise ValueError("Multilag detail checkpoint scope/objective/time-head mismatch")
    for field in ("time_initial_location", "time_initial_scale", "time_sigma_floor"):
        value = head.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Multilag detail time-head initialization is invalid")
    if not 0 < head["time_sigma_floor"] < head["time_initial_scale"]:
        raise ValueError("Multilag detail time-head scale floor is invalid")
    interface, identity = payload.get("interface_meta"), payload.get("resume_identity")
    if not isinstance(interface, Mapping) or not isinstance(identity, Mapping):
        raise ValueError("Multilag detail requires interface and resume identity")
    arguments, interface_head = identity.get("arguments", {}), interface.get("time_head", {})
    if not isinstance(arguments, Mapping) or not isinstance(interface_head, Mapping):
        raise ValueError("Multilag detail interface/resume mappings are invalid")
    units = {"intermittent_frozen_5000": "week", "yellow_trip_hourly": "hour", "insta_market_basket": "day"}
    dataset = arguments.get("dataset_contract")
    if not isinstance(dataset, str) or dataset not in units:
        raise ValueError("Multilag detail v1 dataset must be Intermittent, Taxi or Instacart")
    observation = {"mode": "positive_integer_round_clamp_v1", "unit": units[dataset], "top_code": 30 if dataset == "insta_market_basket" else None}
    if (head.get("observation_likelihood") != observation
            or interface_head.get("observation_likelihood") != observation
            or interface_head.get("statistics_source_split") != "train"
            or interface_head.get("mode") != head["mode"]
            or identity.get("interface_meta") != interface or identity.get("backbone") != expected_backbone
            or identity.get("variant") != LOG_MSE_VARIANT
            or identity.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
            or arguments.get("model_role") != MULTILAG_DETAIL_ROLE
            or arguments.get("time_head_mode") != head["mode"] or arguments.get("time_scale") != scale):
        raise ValueError("Multilag detail observation/resume identity mismatch")
    statistics = interface_head.get("train_time_statistics", {})
    if not isinstance(statistics, Mapping):
        raise ValueError("Multilag detail train-time statistics are invalid")
    for field, statistic in (("time_scale", "time_scale"), ("time_initial_location", "target_log_scaled_mean"),
                             ("time_initial_scale", "target_log_scaled_std")):
        value = statistics.get(statistic)
        # Frozen encoder/interface settings stay exact. Recomputed train
        # statistics use the same absolute tolerance as the execution runner.
        if (interface_head.get(field) != head[field]
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not math.isclose(value, head[field], rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError("Multilag detail train-only initialization mismatch")
    if not states:
        digest = payload.get("checkpoint_state_sha256")
        if (isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
                and type(payload.get("best_epoch")) is int and payload["best_epoch"] >= 1):
            return True
        raise ValueError("Multilag detail requires model state or a valid summary binding")
    expected_identity = torch.tensor(list(hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).digest()),
                                     dtype=torch.uint8)
    dim, rank = metadata["d_model"], metadata["multilag_detail_rank"]
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("Multilag detail state must be a mapping")
        code, digest = state.get("multilag_detail.mode_code"), state.get("multilag_detail.contract_identity")
        if (not isinstance(code, torch.Tensor) or code.dtype != torch.int64 or code.shape != torch.Size([])
                or code.item() != (0 if mode == "local" else 1)
                or not isinstance(digest, torch.Tensor) or digest.dtype != torch.uint8
                or not torch.equal(digest.detach().cpu(), expected_identity)):
            raise ValueError("Multilag detail tensor identity mismatch")
        for branch in range(8):
            for projection, shape in (("level_projections", (rank, dim)),
                                      ("detail_projections", (rank, dim)),
                                      ("output_projections", (dim, rank))):
                tensor = state.get(f"multilag_detail.{projection}.{branch}.weight")
                if (not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
                        or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all())):
                    raise ValueError("Multilag detail learned tensor is missing or invalid")
    return True


__all__ = ["CountAwareTitanMultiLagDetail", "MultiLagDetail", "lag_source_indices",
           "multilag_detail_metadata", "validate_multilag_detail_checkpoint",
           "LOCAL_DETAIL_BACKBONE", "MULTILAG_DETAIL_BACKBONE", "MULTILAG_DETAIL_BACKBONES",
           "MULTILAG_DETAIL_CONTRACT_ID", "MULTILAG_DETAIL_ROLE", "AVAILABILITY_LAGS"]
