"""Query-conditioned event messages inside B's second attention block.

The two routes use the same four projections. They differ only in whether
query/event nonlinear interaction happens before or after attention pooling.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP, LOG_MSE_VARIANT, TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_QUANTITY_GRADIENT_SHARED,
)

PAIR_MESSAGE_CONTRACT_ID = "hard_lmm_pair_message_v1"
PAIR_MESSAGE_ROLE = "observed_time_pair_message_v1"
PAIR_MESSAGE_RANK = 8
PAIR_MESSAGE_PRE_POOL_BACKBONE = "titantpp_pair_message_pre_pool"
PAIR_MESSAGE_POST_POOL_BACKBONE = "titantpp_pair_message_post_pool"
PAIR_MESSAGE_BACKBONES = (PAIR_MESSAGE_PRE_POOL_BACKBONE, PAIR_MESSAGE_POST_POOL_BACKBONE)
_ROUTES = {"pre_pool": (PAIR_MESSAGE_PRE_POOL_BACKBONE, 1),
           "post_pool": (PAIR_MESSAGE_POST_POOL_BACKBONE, 2)}


def pair_message_metadata(hidden_dim: int, routing: str) -> dict[str, Any]:
    if type(hidden_dim) is not int or hidden_dim < 4 or hidden_dim % 4:
        raise ValueError("Pair-message hidden_dim must be a positive multiple of four")
    if routing not in _ROUTES:
        raise ValueError("Pair-message routing must be pre_pool or post_pool")
    return {
        "candidate_name": _ROUTES[routing][0], "backbone_contract_id": PAIR_MESSAGE_CONTRACT_ID,
        "routing_contract_id": PAIR_MESSAGE_CONTRACT_ID, "model_role": PAIR_MESSAGE_ROLE,
        "base_encoder": "CountAwareTitanTPP", "d_model": hidden_dim,
        "n_layers": 2, "n_heads": 4, "d_ff": 2 * hidden_dim,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD, "persistent_mem_size": 16,
        "lmm_mem_size": 64, "lmm_topk": 4, "static_retrieval_aggregation": "arithmetic_mean",
        "static_key_value_tied": True, "pair_message_routing": routing,
        "pair_message_rank": PAIR_MESSAGE_RANK, "pair_message_insertion": "second_attention_before_residual",
        "pair_message_weights": "existing_pre_dropout_head_mean_observed_causal_events_without_renormalization",
        "pair_message_gate": "2_sigmoid_query_plus_key", "pair_message_initialization": "zero_output_projection_forked_rng",
        "pair_message_chunk_size": 32, "pair_message_recompute": True,
        "pair_message_observation_mask": "exclude_unobserved_from_both_encoder_blocks_and_message",
        "additional_parameter_count": 4 * hidden_dim * PAIR_MESSAGE_RANK,
        "quantity_objective": LOG_MSE_VARIANT, "online_writes": False,
    }


def _validate_mask(mask: torch.Tensor, shape: tuple[int, int], name: str) -> None:
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError(f"{name} must be boolean with shape {shape}")


def _pre_pool_chunk(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                    weights: torch.Tensor) -> torch.Tensor:
    messages = 2.0 * torch.sigmoid(query.unsqueeze(2) + key.unsqueeze(1)) * value.unsqueeze(1)
    return (weights.unsqueeze(-1) * messages).sum(dim=2)


def aggregate_pair_messages(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            weights: torch.Tensor, *, routing: str,
                            chunk_size: int = 32, recompute: bool = True) -> torch.Tensor:
    """Pool already-masked messages; exposed for independent numerical oracles."""
    if routing not in _ROUTES:
        raise ValueError("Unknown pair-message routing")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if routing == "post_pool":
        mass = weights.sum(dim=-1, keepdim=True)
        # Exact zero branch; avoid dividing by zero in the autograd graph.
        denominator = torch.where(mass > 0, mass, torch.ones_like(mass))
        pooled_key = torch.bmm(weights, key) / denominator
        pooled_value = torch.bmm(weights, value) / denominator
        return mass * (2.0 * torch.sigmoid(query + pooled_key) * pooled_value)
    chunks = []
    for start in range(0, query.size(1), chunk_size):
        q = query[:, start:start + chunk_size]
        w = weights[:, start:start + chunk_size]
        if recompute and torch.is_grad_enabled() and any(t.requires_grad for t in (q, key, value, w)):
            # This function has no RNG draws. Recompute bounds retained
            # pair-tensor storage, not merely the forward temporary size.
            out = checkpoint(_pre_pool_chunk, q, key, value, w,
                             use_reentrant=False, preserve_rng_state=False)
        else:
            out = _pre_pool_chunk(q, key, value, w)
        chunks.append(out)
    return torch.cat(chunks, dim=1) if chunks else query * 0.0


class PairMessage(nn.Module):
    def __init__(self, hidden_dim: int, routing: str, *, chunk_size: int = 32,
                 recompute: bool = True) -> None:
        super().__init__()
        pair_message_metadata(hidden_dim, routing)
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.hidden_dim, self.rank = hidden_dim, PAIR_MESSAGE_RANK
        self._routing, self.chunk_size, self.recompute = routing, chunk_size, bool(recompute)
        with torch.random.fork_rng(devices=[]):
            self.query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.key_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.value_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.output_projection = nn.Linear(self.rank, hidden_dim, bias=False)
            nn.init.zeros_(self.output_projection.weight)
        self.register_buffer("routing_code", torch.tensor(_ROUTES[routing][1], dtype=torch.int64))

    @property
    def routing(self) -> str:
        return self._routing

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        code = state_dict.get(prefix + "routing_code")
        if (not isinstance(code, torch.Tensor) or code.dtype != torch.int64
                or code.shape != torch.Size([]) or code.item() != _ROUTES[self.routing][1]):
            error_msgs.append("Pair-message routing identity is missing or incompatible")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def read(self, hidden: torch.Tensor, event_attention: torch.Tensor, mask: torch.Tensor, *,
             memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.size(-1) != self.hidden_dim or not hidden.is_floating_point():
            raise ValueError("hidden must have floating shape [batch, length, hidden_dim]")
        shape = tuple(hidden.shape[:2])
        _validate_mask(mask, shape, "mask")
        observed = mask.to(hidden.device)
        if memory_write_mask is not None:
            _validate_mask(memory_write_mask, shape, "memory_write_mask")
            observed = observed & memory_write_mask.to(hidden.device)
        if tuple(event_attention.shape) != (*shape, shape[1]) or event_attention.device != hidden.device:
            raise ValueError("event_attention must have shape [batch, length, length] on hidden device")
        safe_hidden = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        causal = torch.ones(shape[1], shape[1], dtype=torch.bool, device=hidden.device).tril()
        allowed = observed.unsqueeze(-1) & observed.unsqueeze(1) & causal
        weights = torch.where(allowed, event_attention, 0.0)
        query, key, value = (layer(safe_hidden) for layer in
                             (self.query_projection, self.key_projection, self.value_projection))
        return aggregate_pair_messages(query, key, value, weights, routing=self.routing,
                                       chunk_size=self.chunk_size, recompute=self.recompute)

    def forward(self, hidden: torch.Tensor, event_attention: torch.Tensor, mask: torch.Tensor, *,
                memory_write_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.output_projection(self.read(hidden, event_attention, mask,
                                                memory_write_mask=memory_write_mask))


class CountAwareTitanPairMessageTPP(CountAwareTitanTPP):
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, *,
                 routing: str, pair_message_chunk_size: int = 32,
                 pair_message_recompute: bool = True, **quantity_kwargs: Any) -> None:
        pair_message_metadata(hidden_dim, routing)
        for name, expected in (("memory_mode", TITAN_MEMORY_MODE_STATIC_HARD),
                               ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
                               ("quantity_variant", LOG_MSE_VARIANT), ("lambda_tail", 0.0),
                               ("quantile_adaptive_strength", 0.0),
                               ("quantile_adaptive_boundaries", None), ("quantile_adaptive_weights", None)):
            if quantity_kwargs.get(name, expected) != expected:
                raise ValueError(f"Pair-message fixes {name}={expected!r}")
        quantity_kwargs.pop("memory_mode", None)
        super().__init__(hidden_dim=hidden_dim, train_log_mean=train_log_mean, max_seq_len=max_seq_len,
                         memory_mode=TITAN_MEMORY_MODE_STATIC_HARD, **quantity_kwargs)
        self.pair_message = PairMessage(hidden_dim, routing, chunk_size=pair_message_chunk_size,
                                        recompute=pair_message_recompute)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        code = state_dict.get("pair_message.routing_code")
        if (not isinstance(code, torch.Tensor) or code.dtype != torch.int64
                or code.shape != torch.Size([]) or code.item() != _ROUTES[self.pair_message.routing][1]):
            raise RuntimeError("Pair-message routing identity is missing or incompatible")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def encode_task_states(self, dts: torch.Tensor, history_quantities: torch.Tensor,
                           mask: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch, length]")
        _validate_mask(mask, tuple(dts.shape), "mask")
        observed = mask.to(dts.device)
        if memory_write_mask is not None:
            _validate_mask(memory_write_mask, tuple(dts.shape), "memory_write_mask")
            observed = observed & memory_write_mask.to(dts.device)
        # Sanitize before log1p, projection and attention, including indirect
        # paths through later first-layer hidden states.
        safe_dts = torch.where(observed, dts, 0.0)
        safe_quantities = torch.where(observed, history_quantities, 0.0)
        return super().encode_task_states(safe_dts, safe_quantities, observed)

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        del memory_write_mask
        if self.encoder is None or self.lmm is None or len(self.encoder.layers) != 2:
            raise RuntimeError("Pair-message requires B's two blocks and static Hard-LMM")
        features = self.continuous_features(dts, history_quantities, mask)
        encoded = self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:
            encoded = encoded + self.encoder._get_pos(encoded.size(1), encoded.device, encoded.dtype)
        valid = mask.to(encoded.device, dtype=encoded.dtype).unsqueeze(-1)
        encoded = self.encoder.layers[0](encoded * valid, mask=mask)
        block = self.encoder.layers[1]
        normalized = block.norm1(encoded)
        attention, event_attention = block.attn(normalized, mask=mask, return_event_attention=True)
        correction = self.pair_message(normalized, event_attention, mask)
        encoded = (encoded + block.drop1(attention + correction)) * valid
        encoded = (encoded + block.drop2(block.ff(block.norm2(encoded)))) * valid
        return encoded


class CountAwareTitanPairMessagePrePoolTPP(CountAwareTitanPairMessageTPP):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, **kwargs):
        if "routing" in kwargs:
            raise ValueError("Pre-pool backbone fixes its routing")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, routing="pre_pool", **kwargs)


class CountAwareTitanPairMessagePostPoolTPP(CountAwareTitanPairMessageTPP):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, **kwargs):
        if "routing" in kwargs:
            raise ValueError("Post-pool backbone fixes its routing")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, routing="post_pool", **kwargs)


def validate_pair_message_checkpoint(payload: dict[str, Any], expected_backbone: str) -> bool:
    metadata = payload.get("encoder_config", {})
    states = [payload[name] for name in ("model_state_dict", "best_state_dict") if name in payload]
    identified = (payload.get("backbone") in PAIR_MESSAGE_BACKBONES
                  or isinstance(metadata, Mapping) and (
                      any(str(key).startswith("pair_message_") for key in metadata)
                      or metadata.get("backbone_contract_id") == PAIR_MESSAGE_CONTRACT_ID
                      or metadata.get("routing_contract_id") == PAIR_MESSAGE_CONTRACT_ID
                      or metadata.get("model_role") == PAIR_MESSAGE_ROLE
                      or metadata.get("candidate_name") in PAIR_MESSAGE_BACKBONES)
                  or any(isinstance(state, Mapping) and any(str(key).startswith("pair_message.") for key in state)
                         for state in states))
    if expected_backbone not in PAIR_MESSAGE_BACKBONES:
        if identified:
            raise ValueError("Pair-message checkpoint cannot be relabelled as another backbone")
        return False
    routing = next(name for name, (backbone, _) in _ROUTES.items() if backbone == expected_backbone)
    if not isinstance(metadata, Mapping):
        raise ValueError("Pair-message checkpoint metadata must be a mapping")
    dim = metadata.get("d_model")
    expected = pair_message_metadata(dim, routing)
    if payload.get("backbone") != expected_backbone or any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Pair-message checkpoint routing metadata mismatch")
    if (payload.get("variant") != LOG_MSE_VARIANT or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False):
        raise ValueError("Pair-message checkpoint scope/objective mismatch")
    head = metadata.get("time_head")
    if (not isinstance(head, Mapping) or head.get("mode") != "heteroscedastic_lognormal_duration"
            or head.get("reported_metric") != "recorded_positive_integer_time_nll"
            or head.get("density_family") != "heteroscedastic_lognormal_on_scaled_duration"
            or head.get("time_location_transform") != "identity"
            or head.get("time_scale_conditioning") != "linear_hidden"
            or head.get("time_scale_transform") != "softplus_plus_floor"
            or head.get("time_scale_weight_initialization") != "zeros"
            or head.get("wd_clamp") != 0.0):
        raise ValueError("Pair-message checkpoint requires the recorded-duration time-head contract")
    for field in ("time_scale", "time_initial_location", "time_initial_scale", "time_sigma_floor"):
        value = head.get(field)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("Pair-message checkpoint time-head numeric contract is invalid")
    if not (head["time_scale"] > 0 and 0 < head["time_sigma_floor"] < head["time_initial_scale"]):
        raise ValueError("Pair-message checkpoint time-head scale contract is invalid")
    interface, identity = payload.get("interface_meta"), payload.get("resume_identity")
    if not isinstance(interface, Mapping) or not isinstance(identity, Mapping):
        raise ValueError("Pair-message checkpoint requires training interface and resume identity")
    arguments = identity.get("arguments", {})
    observations = {"intermittent_frozen_5000": ("week", None), "yellow_trip_hourly": ("hour", None),
                    "insta_market_basket": ("day", 30)}
    if not isinstance(arguments, Mapping):
        raise ValueError("Pair-message checkpoint resume arguments are invalid")
    dataset = arguments.get("dataset_contract")
    if not isinstance(dataset, str) or dataset not in observations:
        raise ValueError("Pair-message checkpoint dataset identity is invalid")
    unit, top_code = observations[dataset]
    observation = {"mode": "positive_integer_round_clamp_v1", "unit": unit, "top_code": top_code}
    interface_head = interface.get("time_head", {})
    if (not isinstance(interface_head, Mapping) or head.get("observation_likelihood") != observation
            or interface_head.get("observation_likelihood") != observation
            or interface_head.get("statistics_source_split") != "train"
            or interface_head.get("mode") != head["mode"]
            or identity.get("interface_meta") != interface
            or identity.get("backbone") != expected_backbone or identity.get("variant") != LOG_MSE_VARIANT
            or identity.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
            or arguments.get("model_role") != PAIR_MESSAGE_ROLE
            or arguments.get("time_head_mode") != head["mode"]
            or arguments.get("time_scale") != head["time_scale"]):
        raise ValueError("Pair-message checkpoint observation/resume identity mismatch")
    statistics = interface_head.get("train_time_statistics", {})
    if not isinstance(statistics, Mapping):
        raise ValueError("Pair-message checkpoint train-time statistics are invalid")
    for field, statistic in (("time_scale", "time_scale"), ("time_initial_location", "target_log_scaled_mean"),
                             ("time_initial_scale", "target_log_scaled_std")):
        if interface_head.get(field) != head[field] or statistics.get(statistic) != head[field]:
            raise ValueError("Pair-message checkpoint train-only initialization mismatch")
    if not states:
        if isinstance(payload.get("checkpoint_state_sha256"), str) and type(payload.get("best_epoch")) is int:
            return True
        raise ValueError("Pair-message artifact requires model state or a summary binding")
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("Pair-message checkpoint state must be a mapping")
        code = state.get("pair_message.routing_code")
        if (not isinstance(code, torch.Tensor) or code.dtype != torch.int64 or code.shape != torch.Size([])
                or code.item() != _ROUTES[routing][1]):
            raise ValueError("Pair-message checkpoint route identity mismatch")
        for name, shape in (("query_projection", (8, dim)), ("key_projection", (8, dim)),
                            ("value_projection", (8, dim)), ("output_projection", (dim, 8))):
            tensor = state.get(f"pair_message.{name}.weight")
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Pair-message checkpoint projection state is missing or invalid")
    return True


__all__ = ["CountAwareTitanPairMessageTPP", "CountAwareTitanPairMessagePrePoolTPP",
           "CountAwareTitanPairMessagePostPoolTPP", "PairMessage", "aggregate_pair_messages",
           "PAIR_MESSAGE_CONTRACT_ID", "PAIR_MESSAGE_ROLE", "PAIR_MESSAGE_RANK",
           "PAIR_MESSAGE_PRE_POOL_BACKBONE", "PAIR_MESSAGE_POST_POOL_BACKBONE", "PAIR_MESSAGE_BACKBONES",
           "pair_message_metadata", "validate_pair_message_checkpoint"]
