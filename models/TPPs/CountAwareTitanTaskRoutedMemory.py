"""Task-specific queries over one shared completed-episode memory bank.

The split candidate assigns each query's read to its task. The matched shared
control gives both tasks the mean of the two reads. Both retain B's encoder,
static Hard-LMM, heads, objectives and per-forward observed-episode lifetime.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP, LOG_MSE_VARIANT, TIME_HEAD_MODE_LEGACY_CLAMPED,
    TITAN_MEMORY_MODE_STATIC_HARD, TITAN_QUANTITY_GRADIENT_SHARED,
)
from models.TPPs.CountAwareTitanSuccessorMemory import (
    EPISODE_MEMORY_LN_EPSILON, _is_sha256, _state_shapes as _episode_state_shapes,
    _validate_mask, completed_pairs,
)

TASK_SHARED_MEMORY_BACKBONE = "titantpp_task_shared_episode_memory"
TASK_SPLIT_MEMORY_BACKBONE = "titantpp_task_split_episode_memory"
TASK_SHARED_MEMORY_ROLE = "hard_lmm_task_shared_episode_control"
TASK_SPLIT_MEMORY_ROLE = "hard_lmm_task_split_episode_candidate"
TASK_ROUTED_MEMORY_CONTRACT_ID = "hard_lmm_task_routed_episode_v1"
TASK_ROUTED_MEMORY_DESIGN_SHA256 = "d4ba8e7a55866a668ac5dfac307acb001e2cfa1a4ef2d4ffd9cdc3589cdb8012"
TASK_ROUTED_MEMORY_RANK = 8
TASK_ROUTED_MEMORY_VALUE_WIDTH = 16
TASK_ROUTED_MEMORY_PARAMETER_KEYS = (
    "task_routed_episode_memory.time_query_projection.weight",
    "task_routed_episode_memory.quantity_query_projection.weight",
    "task_routed_episode_memory.key_projection.weight",
    "task_routed_episode_memory.value_projection.weight",
    "task_routed_episode_memory.value_projection.bias",
    "task_routed_episode_memory.time_output_projection.weight",
    "task_routed_episode_memory.quantity_output_projection.weight",
)
TASK_ROUTED_MEMORY_ROUTING_KEY = "task_routed_episode_memory.routing_code"
_ROUTINGS = {
    "shared": (TASK_SHARED_MEMORY_BACKBONE, TASK_SHARED_MEMORY_ROLE, 0),
    "split": (TASK_SPLIT_MEMORY_BACKBONE, TASK_SPLIT_MEMORY_ROLE, 1),
}


def _routing_contract(routing: str) -> tuple[str, str, int]:
    if not isinstance(routing, str) or routing not in _ROUTINGS:
        raise ValueError("routing must be 'shared' or 'split'")
    return _ROUTINGS[routing]


def task_routed_memory_metadata(hidden_dim: int, routing: str) -> dict[str, Any]:
    """Structural identity; the runner supplies source, optimizer and RNG bindings."""
    if type(hidden_dim) is not int or hidden_dim < 1 or hidden_dim % 4:
        raise ValueError("hidden_dim must be a positive multiple of four")
    backbone, role, code = _routing_contract(routing)
    return {
        "candidate_name": backbone,
        "candidate_status": "experimental_unvalidated",
        "backbone_contract_id": TASK_ROUTED_MEMORY_CONTRACT_ID,
        "routing_contract_id": TASK_ROUTED_MEMORY_CONTRACT_ID,
        "design_contract_id": "hard_lmm_task_routed_episode_design_v1",
        "design_contract_sha256": TASK_ROUTED_MEMORY_DESIGN_SHA256,
        "model_role": role, "base_encoder": "CountAwareTitanTPP",
        "d_model": hidden_dim, "n_layers": 2, "n_heads": 4, "d_ff": hidden_dim * 2,
        "memory_mode": TITAN_MEMORY_MODE_STATIC_HARD,
        "persistent_mem_size": 16, "lmm_mem_size": 64, "lmm_topk": 4,
        "static_retrieval_similarity": "cosine", "static_retrieval_aggregation": "arithmetic_mean",
        "static_key_value_tied": True,
        "task_routed_routing": routing, "task_routed_routing_code": code,
        "task_routed_association": "successor",
        "task_routed_key_source": "one_shared_projection_of_predecessor_hidden",
        "task_routed_query_source": "two_independent_projections_of_observed_base_encoder_hidden",
        "task_routed_feature_source": "existing_two_channel_successor_event_features",
        "task_routed_feature_transform": "SharedTimeCountModel.continuous_features_unchanged",
        "task_routed_adjacency": "adjacent_valid_events_skip_padding_no_unobserved_bridge",
        "task_routed_availability": "both_endpoints_observed_and_successor_at_or_before_observed_query",
        "task_routed_rank": TASK_ROUTED_MEMORY_RANK,
        "task_routed_value_width": TASK_ROUTED_MEMORY_VALUE_WIDTH,
        "task_routed_normalization": "affine_free_layer_norm",
        "task_routed_normalization_epsilon": EPISODE_MEMORY_LN_EPSILON,
        "task_routed_address_projection_bias": False,
        "task_routed_value_projection": "one_shared_linear_2_to_16_bias_true",
        "task_routed_activation": "gelu_approximate_none",
        "task_routed_output_projections": "two_independent_linear_16_to_hidden_bias_false_zero_initialized",
        "task_routed_reads": "two_query_reads_always_computed",
        "task_routed_context_assignment": "both_mean_of_two_reads" if routing == "shared" else "time_read_and_quantity_read_separately",
        "task_routed_empty_row": "both_contexts_exact_zero_even_with_trained_value_bias",
        "task_routed_invalid_record": "sanitize_before_nonlinear_operations_and_exclude_from_reduction",
        "task_routed_placement": "after_unchanged_encoder_and_final_static_memory_before_task_heads",
        "task_routed_fusion": "zt=z_B+Ut(context_time);zq=z_B+Uq(context_quantity)",
        "task_routed_state_lifetime": "per_sample_rebuilt_each_forward_no_external_carry",
        "task_routed_new_dropout": False,
        "task_routed_compute_dtype": "float64_for_float64_otherwise_float32",
        "task_routed_initialization": "isolated_cpu_rng_post_B_Qt_Qq_K_V_Ut_Uq_default_linear_V_bias_zero_both_U_zero",
        "quantity_variant": LOG_MSE_VARIANT,
        "quantity_memory_gradient_mode": TITAN_QUANTITY_GRADIENT_SHARED,
        "lambda_tail": 0.0,
        "new_parameter_state_keys": list(TASK_ROUTED_MEMORY_PARAMETER_KEYS),
        "routing_state_key": TASK_ROUTED_MEMORY_ROUTING_KEY,
        "additional_parameter_count": 3 * hidden_dim * 8 + 2 * hidden_dim * 16 + 2 * 16 + 16,
        "online_writes": True,
    }


def _state_shapes(dim: int, max_len: int) -> dict[str, tuple[int, ...]]:
    shapes = {key: value for key, value in _episode_state_shapes(dim, max_len).items()
              if not key.startswith("episode_memory.")}
    shapes.update({
        TASK_ROUTED_MEMORY_PARAMETER_KEYS[0]: (8, dim),
        TASK_ROUTED_MEMORY_PARAMETER_KEYS[1]: (8, dim),
        TASK_ROUTED_MEMORY_PARAMETER_KEYS[2]: (8, dim),
        TASK_ROUTED_MEMORY_PARAMETER_KEYS[3]: (16, 2),
        TASK_ROUTED_MEMORY_PARAMETER_KEYS[4]: (16,),
        TASK_ROUTED_MEMORY_PARAMETER_KEYS[5]: (dim, 16),
        TASK_ROUTED_MEMORY_PARAMETER_KEYS[6]: (dim, 16),
        TASK_ROUTED_MEMORY_ROUTING_KEY: (),
    })
    return shapes


def validate_task_routed_memory_checkpoint(payload: dict[str, Any], expected_backbone: str) -> bool:
    """Recognize route identity before other validators, including relabeled states."""
    if not isinstance(payload, Mapping):
        raise ValueError("Checkpoint payload must be a mapping")
    metadata = payload.get("encoder_config", {})
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    backbones = tuple(value[0] for value in _ROUTINGS.values())
    roles = tuple(value[1] for value in _ROUTINGS.values())
    recognized = (
        payload.get("backbone") in backbones
        or isinstance(metadata, Mapping) and (
            metadata.get("backbone_contract_id") == TASK_ROUTED_MEMORY_CONTRACT_ID
            or metadata.get("routing_contract_id") == TASK_ROUTED_MEMORY_CONTRACT_ID
            or metadata.get("design_contract_sha256") == TASK_ROUTED_MEMORY_DESIGN_SHA256
            or metadata.get("model_role") in roles or metadata.get("candidate_name") in backbones
            or any(isinstance(key, str) and key.startswith("task_routed_") for key in metadata)
        )
        or any(isinstance(state, Mapping) and any(
            isinstance(key, str) and key.startswith("task_routed_episode_memory.") for key in state
        ) for state in states)
    )
    if expected_backbone not in backbones:
        if recognized:
            raise ValueError("Task-routed memory checkpoint cannot be relabelled")
        return False
    if not isinstance(metadata, Mapping):
        raise ValueError("Task-routed checkpoint metadata must be a mapping")
    routing = next(mode for mode, values in _ROUTINGS.items() if values[0] == expected_backbone)
    dim, max_len = metadata.get("d_model"), metadata.get("max_len")
    if type(dim) is not int or dim < 1 or dim % 4 or type(max_len) is not int or max_len < 1:
        raise ValueError("Task-routed checkpoint dimensions are invalid")
    if payload.get("backbone") != expected_backbone or any(
        metadata.get(key) != value for key, value in task_routed_memory_metadata(dim, routing).items()
    ):
        raise ValueError("Task-routed checkpoint routing metadata mismatch")
    if any(isinstance(key, str) and key.startswith(("episode_", "nonlinear_episode_")) for key in metadata):
        raise ValueError("Task-routed checkpoint contains foreign episode metadata")
    time_head = metadata.get("time_head", {})
    if (payload.get("variant") != LOG_MSE_VARIANT
            or payload.get("evaluation_scope") != "validation_only"
            or payload.get("held_out_test_evaluated") is not False
            or not isinstance(time_head, Mapping)
            or time_head.get("mode") != TIME_HEAD_MODE_LEGACY_CLAMPED
            or time_head.get("jacobian_correction") is not False or time_head.get("wd_clamp") != 10.0
            or any(type(time_head.get(key)) not in (int, float)
                   or not math.isfinite(time_head[key]) or time_head[key] <= 0
                   for key in ("time_scale", "time_w_max", "time_intercept_limit"))):
        raise ValueError("Task-routed checkpoint scope/head/objective mismatch")
    for config in (payload, payload.get("training_config", {})):
        if not isinstance(config, Mapping) or any(
            key in config and config[key] != value for key, value in (
                ("quantity_variant", LOG_MSE_VARIANT), ("lambda_tail", 0.0),
                ("time_head_mode", TIME_HEAD_MODE_LEGACY_CLAMPED),
                ("quantile_adaptive_strength", 0.0),
                ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
                ("quantile_adaptive_boundaries", None), ("quantile_adaptive_weights", None),
            )
        ):
            raise ValueError("Task-routed checkpoint training objective mismatch")
    if not states:
        if (_is_sha256(payload.get("checkpoint_state_sha256"))
                and type(payload.get("best_epoch")) is int and payload["best_epoch"] >= 0
                and ("checkpoint_file_sha256" not in payload or _is_sha256(payload["checkpoint_file_sha256"]))):
            return True
        raise ValueError("Task-routed summary requires selected epoch and canonical state digest")
    shapes = _state_shapes(dim, max_len)
    code = _ROUTINGS[routing][2]
    for state in states:
        if not isinstance(state, Mapping) or set(state) != set(shapes):
            raise ValueError("Task-routed checkpoint state keys are missing or foreign")
        for name, shape in shapes.items():
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
                raise ValueError("Task-routed checkpoint state shape is invalid")
            if name == TASK_ROUTED_MEMORY_ROUTING_KEY:
                if tensor.dtype != torch.int64 or tensor.item() != code:
                    raise ValueError("Task-routed checkpoint routing state mismatch")
            elif not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Task-routed checkpoint requires finite floating state")
    return True


class TaskRoutedEpisodeMemory(nn.Module):
    """Two independently queried reads of a shared causal successor bank."""

    def __init__(self, hidden_dim: int, routing: str = "shared") -> None:
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        _, _, code = _routing_contract(routing)
        self.hidden_dim, self.rank, self.value_width = hidden_dim, TASK_ROUTED_MEMORY_RANK, TASK_ROUTED_MEMORY_VALUE_WIDTH
        self._routing = routing
        # Qt and Qq consume successive draws within one isolated RNG context.
        # Both models instantiate exactly the same parameters in the same order.
        with torch.random.fork_rng(devices=[]):
            self.time_query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.quantity_query_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.key_projection = nn.Linear(hidden_dim, self.rank, bias=False)
            self.value_projection = nn.Linear(2, self.value_width, bias=True)
            self.time_output_projection = nn.Linear(self.value_width, hidden_dim, bias=False)
            self.quantity_output_projection = nn.Linear(self.value_width, hidden_dim, bias=False)
            nn.init.zeros_(self.value_projection.bias)
            nn.init.zeros_(self.time_output_projection.weight)
            nn.init.zeros_(self.quantity_output_projection.weight)
        self.register_buffer("routing_code", torch.tensor(code, dtype=torch.int64))

    @property
    def routing(self) -> str:
        return self._routing

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        identity = state_dict.get(prefix + "routing_code")
        if (not isinstance(identity, torch.Tensor) or identity.dtype != torch.int64
                or identity.shape != torch.Size([]) or identity.item() != _ROUTINGS[self.routing][2]):
            error_msgs.append("Task-routed memory routing identity is missing or incompatible")
            return
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def read(self, hidden: torch.Tensor, features: torch.Tensor, mask: torch.Tensor, *,
             memory_write_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 3
                or hidden.shape[-1] != self.hidden_dim or not hidden.is_floating_point()):
            raise ValueError("hidden must have floating shape [batch, length, hidden_dim]")
        shape = tuple(hidden.shape[:2])
        if shape[1] < 1:
            raise ValueError("Task-routed memory requires a nonempty sequence dimension")
        if (not isinstance(features, torch.Tensor) or tuple(features.shape) != (*shape, 2)
                or not features.is_floating_point() or features.device != hidden.device):
            raise ValueError("features must have floating shape [batch, length, 2] on the hidden device")
        _validate_mask(mask, shape, "mask")
        predecessors, pairs, observed = completed_pairs(mask.to(hidden.device), memory_write_mask=memory_write_mask)
        safe_hidden = torch.where(observed.unsqueeze(-1), hidden, 0.0)
        safe_features = torch.where(observed.unsqueeze(-1), features, 0.0)
        if not bool(torch.isfinite(safe_hidden).all()) or not bool(torch.isfinite(safe_features).all()):
            raise ValueError("Observed task-routed states and features must be finite")
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            normalized = F.layer_norm(safe_hidden.to(dtype), (self.hidden_dim,), eps=EPISODE_MEMORY_LN_EPSILON)
            keys = F.linear(normalized, self.key_projection.weight.to(dtype))
            keys = keys.gather(1, predecessors.clamp_min(0).unsqueeze(-1).expand(-1, -1, self.rank))
            keys = torch.where(pairs.unsqueeze(-1), keys, 0.0)
            values = torch.where(pairs.unsqueeze(-1), safe_features.to(dtype), 0.0)
            transformed = F.gelu(F.linear(values, self.value_projection.weight.to(dtype),
                                           self.value_projection.bias.to(dtype)), approximate="none")
            transformed = torch.where(pairs.unsqueeze(-1), transformed, 0.0)
            positions = torch.arange(shape[1], device=hidden.device)
            available = (observed.unsqueeze(-1) & pairs.unsqueeze(1)
                         & (positions.unsqueeze(0) <= positions.unsqueeze(1)).unsqueeze(0))
            nonempty = available.any(dim=-1)
            contexts = []
            # Both arms compute both attention matrices and both value reads.
            for projection in (self.time_query_projection, self.quantity_query_projection):
                queries = F.linear(normalized, projection.weight.to(dtype))
                scores = (queries @ keys.transpose(1, 2)) / math.sqrt(self.rank)
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("Task-routed addressing scores must be finite")
                scores = scores.masked_fill(~available, -torch.inf)
                weights = torch.zeros_like(scores)
                weights[nonempty] = torch.softmax(scores[nonempty], dim=-1)
                contexts.append(torch.where(nonempty.unsqueeze(-1), weights @ transformed, 0.0))
            if self.routing == "shared":
                shared = (contexts[0] + contexts[1]) * 0.5
                contexts = (shared, shared)
        if any(not bool(torch.isfinite(context).all()) for context in contexts):
            raise ValueError("Task-routed memory reads must be finite")
        return tuple(context.to(hidden.dtype) for context in contexts)

    def forward(self, hidden: torch.Tensor, features: torch.Tensor, mask: torch.Tensor, *,
                memory_write_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = self.read(hidden, features, mask, memory_write_mask=memory_write_mask)
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            corrections = tuple(F.linear(context.to(dtype), projection.weight.to(dtype)).to(hidden.dtype)
                                for context, projection in zip(contexts, (self.time_output_projection,
                                                                          self.quantity_output_projection)))
        if any(not bool(torch.isfinite(correction).all()) for correction in corrections):
            raise ValueError("Task-routed memory corrections must be finite")
        return corrections


class _CountAwareTitanTaskRoutedTPP(CountAwareTitanTPP):
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, *,
                 routing: str, **quantity_kwargs: Any) -> None:
        task_routed_memory_metadata(hidden_dim, routing)
        if type(max_seq_len) is not int or max_seq_len < 1:
            raise ValueError("max_seq_len must be a positive integer")
        for name, expected in (
            ("memory_mode", TITAN_MEMORY_MODE_STATIC_HARD),
            ("quantity_memory_gradient_mode", TITAN_QUANTITY_GRADIENT_SHARED),
            ("quantity_variant", LOG_MSE_VARIANT), ("time_head_mode", TIME_HEAD_MODE_LEGACY_CLAMPED),
            ("lambda_tail", 0.0), ("quantile_adaptive_strength", 0.0),
            ("quantile_adaptive_boundaries", None), ("quantile_adaptive_weights", None),
        ):
            if quantity_kwargs.get(name, expected) != expected:
                raise ValueError(f"Task-routed memory requires {name}={expected!r}")
        quantity_kwargs.pop("memory_mode", None)
        super().__init__(hidden_dim=hidden_dim, train_log_mean=train_log_mean,
                         max_seq_len=max_seq_len, memory_mode=TITAN_MEMORY_MODE_STATIC_HARD, **quantity_kwargs)
        self.task_routed_episode_memory = TaskRoutedEpisodeMemory(hidden_dim, routing=routing)

    def encode_task_states(self, dts: torch.Tensor, history_quantities: torch.Tensor,
                           mask: torch.Tensor, *, memory_write_mask: torch.Tensor | None = None
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch, length]")
        _validate_mask(mask, tuple(dts.shape), "mask")
        if memory_write_mask is not None:
            _validate_mask(memory_write_mask, tuple(dts.shape), "memory_write_mask")
        hidden = super()._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        if self.lmm is None:
            raise RuntimeError("Task-routed memory requires the original static Hard-LMM")
        valid = mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
        baseline = self.lmm(hidden) * valid
        features = self.continuous_features(dts, history_quantities, mask)
        time_correction, quantity_correction = self.task_routed_episode_memory(
            hidden, features, mask, memory_write_mask=memory_write_mask)
        return (baseline + time_correction) * valid, (baseline + quantity_correction) * valid


class CountAwareTitanTaskSharedMemoryTPP(_CountAwareTitanTaskRoutedTPP):
    """Capacity/major-operation control: both tasks receive the mean of two reads."""
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, **quantity_kwargs: Any) -> None:
        if any(name in quantity_kwargs for name in ("routing", "pooling", "association")):
            raise ValueError("Task-shared backbone fixes its routing and successor association")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, routing="shared", **quantity_kwargs)


class CountAwareTitanTaskSplitMemoryTPP(_CountAwareTitanTaskRoutedTPP):
    """Candidate: time and quantity consume their independently queried reads."""
    def __init__(self, hidden_dim: int, train_log_mean: float, max_seq_len: int, **quantity_kwargs: Any) -> None:
        if any(name in quantity_kwargs for name in ("routing", "pooling", "association")):
            raise ValueError("Task-split backbone fixes its routing and successor association")
        super().__init__(hidden_dim, train_log_mean, max_seq_len, routing="split", **quantity_kwargs)


__all__ = [
    "TaskRoutedEpisodeMemory", "CountAwareTitanTaskSharedMemoryTPP", "CountAwareTitanTaskSplitMemoryTPP",
    "TASK_SHARED_MEMORY_BACKBONE", "TASK_SPLIT_MEMORY_BACKBONE", "TASK_SHARED_MEMORY_ROLE", "TASK_SPLIT_MEMORY_ROLE",
    "TASK_ROUTED_MEMORY_CONTRACT_ID", "TASK_ROUTED_MEMORY_DESIGN_SHA256", "TASK_ROUTED_MEMORY_RANK",
    "TASK_ROUTED_MEMORY_VALUE_WIDTH", "TASK_ROUTED_MEMORY_PARAMETER_KEYS", "TASK_ROUTED_MEMORY_ROUTING_KEY",
    "task_routed_memory_metadata", "validate_task_routed_memory_checkpoint",
]
