"""Five static-memory retrieval contrasts on the frozen CNN+GRU54 encoder.

Observed context changes the search query/score only. The 64x64 trainable bank,
fixed additive residual, two shared heads and encoder arithmetic stay inherited.
No state is stored between forwards and no memory slot is written online.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.Titan.common.memory import SimilarityWeightedLocalMemoryMatcher
from models.TPPs.CountAwareTitanCNNGRU import (
    CountAwareTitanCNNGRU, metadata as parent_metadata,
    validate_checkpoint as parent_validate_checkpoint,
)
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

BASELINE = "titantpp_cnn_gru54"
ARMS = tuple("titantpp_retrieval_" + name for name in
    ("mean", "weighted", "fixed", "adaptive", "generic"))
ROLE_BY_ARM = {arm: "observed_time_contextual_retrieval_" + arm.rsplit("_", 1)[1] + "_v1"
    for arm in ARMS}
CONTRACT_ID = "titantpp_contextual_retrieval_cnn_gru54_v1"
WINDOW = 32
EXTRA_PARAMETERS = dict(zip(ARMS, (0, 0, 512, 521, 521)))
NORMALIZATION_KEYS = ("time_scale", "time_log_mean", "time_log_std", "quantity_log_mean", "quantity_log_std")


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported contextual retrieval arm")
    return ROLE_BY_ARM[arm]


def _statistics(values):
    if not isinstance(values, Mapping) or set(values) != set(NORMALIZATION_KEYS):
        raise ValueError("Retrieval normalization requires exact frozen Train statistics")
    try:
        result = {key: float(values[key]) for key in NORMALIZATION_KEYS}
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid frozen retrieval normalization statistics") from error
    if (not all(math.isfinite(value) for value in result.values())
            or result["time_scale"] <= 0 or result["time_log_std"] <= 0
            or result["quantity_log_std"] < 0):
        raise ValueError("Invalid frozen retrieval normalization statistics")
    return result


def metadata(hidden_dim=64, *, arm, statistics=None):
    role_for_arm(arm)
    result = parent_metadata(hidden_dim, arm=BASELINE)
    result.update(candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), baseline_backbone=BASELINE,
        frozen_encoder_contract_id=parent_metadata(hidden_dim, arm=BASELINE)["backbone_contract_id"],
        retrieval_kind=arm.rsplit("_", 1)[1], retrieval_insertion="after_encoder_block_2",
        retrieval_memory_slots=64, retrieval_memory_dim=64, retrieval_topk=4,
        retrieval_bank="inherited_static_tied_cosine_key_value_trainable_mem",
        retrieval_output="hidden_plus_selected_bank_values_fixed_alpha_1_shared_heads",
        retrieval_topk_ties="inherited_torch_topk_native_deterministic_policy",
        retrieval_context_window=WINDOW,
        retrieval_context_order=["time_last", "time_mean", "time_population_std", "time_last_difference",
            "quantity_last", "quantity_mean", "quantity_population_std", "quantity_last_difference"],
        retrieval_context_window_unit="last_32_observed_events_in_current_segment",
        retrieval_time_feature="log_positive_observed_gap_divided_by_time_scale_then_train_zscore",
        retrieval_quantity_feature="log1p_observed_quantity_then_train_zscore",
        retrieval_normalization="inherited_frozen_Train_point_statistics_before_summary_no_feature_refit",
        retrieval_missing="dt_le_zero_excluded_qty_zero_included_missing_summaries_zero",
        retrieval_availability="last_mean_n_ge_1_population_std_difference_n_ge_2",
        retrieval_zero_quantity_std="all_quantity_point_features_zero",
        retrieval_withheld="valid_withheld_resets_segment_and_returns_zero",
        retrieval_padding="skip_no_window_update",
        retrieval_weighting="mean" if arm == ARMS[0] else "softmax_selected_cosine_score",
        retrieval_fixed_temperature=1.0 if arm != ARMS[4] else None,
        retrieval_query="inherited_hidden" if arm in ARMS[:2] else
            "hidden_plus_half_all8_times_W8x64" if arm == ARMS[4] else
            "time_hidden_plus_phi4Wt_and_quantity_hidden_plus_phi4Wq",
        retrieval_score="cosine" if arm in (*ARMS[:2], ARMS[4]) else "alpha_cos_time_plus_one_minus_alpha_cos_quantity",
        retrieval_alpha="sigmoid_phi8_g8_plus_bias" if arm == ARMS[3] else
            "fixed_0.5" if arm == ARMS[2] else None,
        retrieval_temperature="exp_log2_tanh_phi8_g8_plus_bias" if arm == ARMS[4] else "fixed_1",
        retrieval_temperature_bounds=[0.5, 2.0] if arm == ARMS[4] else None,
        retrieval_parameter_initialization="all_added_score_parameters_zero_no_rng_consumption",
        retrieval_extra_parameter_count=EXTRA_PARAMETERS[arm],
        retrieval_parameter_budget_comparison="adaptive_generic_521_actual_active_score_parameters",
        retrieval_gain_learning=False, retrieval_online_writes=False,
        retrieval_added_output_projection=False, additional_parameter_count=25308 + EXTRA_PARAMETERS[arm])
    if statistics is not None:
        result["retrieval_train_statistics"] = _statistics(statistics)
    return result


def identity(*, arm, statistics):
    encoded = json.dumps(metadata(arm=arm, statistics=statistics), sort_keys=True, separators=(",", ":")).encode()
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


def _prefix_summaries(points, eligibility, valid, observed):
    """Summarize last 32 observed event positions, within each withheld segment.

    Padding never consumes a position. Missing gaps are excluded inside the
    common event window; zero quantities are valid observations. All summaries
    use the already standardized points, and unavailable summaries become zero.
    """
    batch, length = points.shape
    if length == 0:
        return points.new_zeros((batch, 0, 4))
    count = observed.long().cumsum(1)
    boundary = torch.where(valid & ~observed, count, 0).cummax(1).values
    local_count = count - boundary
    positions = torch.arange(length, device=points.device).expand(batch, -1)
    # Rank zero is reserved for missing/padding; amax prevents invalid rows
    # from overwriting the latest actual observed position at a shared rank.
    table = torch.full((batch, length + 1), -1, dtype=torch.long, device=points.device)
    table.scatter_reduce_(1, count, torch.where(observed, positions, -1), reduce="amax", include_self=True)
    offsets = torch.arange(WINDOW, device=points.device)
    ranks = count.unsqueeze(-1) - offsets
    active = observed.unsqueeze(-1) & (offsets < local_count.unsqueeze(-1)) & (ranks > 0)
    indices = table.gather(1, ranks.clamp_min(0).reshape(batch, -1)).reshape(batch, length, WINDOW)
    gathered = points.gather(1, indices.clamp_min(0).reshape(batch, -1)).reshape(batch, length, WINDOW)
    available = eligibility.gather(1, indices.clamp_min(0).reshape(batch, -1)).reshape(batch, length, WINDOW) & active
    safe = torch.where(available, gathered, 0.)
    n = available.sum(-1)
    first = available.long().argmax(-1)
    last = safe.gather(-1, first.unsqueeze(-1)).squeeze(-1)
    mean = safe.sum(-1) / n.clamp_min(1)
    centered = torch.where(available, gathered - mean.unsqueeze(-1), 0.)
    std = (centered.square().sum(-1) / n.clamp_min(1)).sqrt()
    remaining = available & (offsets > first.unsqueeze(-1))
    previous = safe.gather(-1, remaining.long().argmax(-1).unsqueeze(-1)).squeeze(-1)
    return torch.stack((torch.where(n >= 1, last, 0.), torch.where(n >= 1, mean, 0.),
        torch.where(n >= 2, std, 0.), torch.where(n >= 2, last - previous, 0.)), -1)


class ContextualLocalMemoryMatcher(SimilarityWeightedLocalMemoryMatcher):
    """Reuse the parent's exact bank object and add only active score tensors."""
    def __init__(self, bank, arm):
        nn.Module.__init__(self)
        if (arm not in ARMS[1:] or bank.mem.shape != (1, 64, 64)
                or (bank.d_model, bank.mem_size, bank.topk) != (64, 64, 4)):
            raise ValueError("Contextual retrieval requires the frozen 64x64 top4 bank")
        self.d_model, self.mem_size, self.topk = bank.d_model, bank.mem_size, bank.topk
        self.mem, self.arm = bank.mem, arm
        if arm in ARMS[2:4]:
            self.time_query_weight = nn.Parameter(torch.zeros(4, 64))
            self.quantity_query_weight = nn.Parameter(torch.zeros(4, 64))
        elif arm == ARMS[4]:
            self.query_weight = nn.Parameter(torch.zeros(8, 64))
        if arm in ARMS[3:]:
            self.context_gate_weight = nn.Parameter(torch.zeros(8))
            self.context_gate_bias = nn.Parameter(torch.zeros(()))

    def retrieve_context(self, encoded, features):
        if features.shape != (*encoded.shape[:2], 8) or not bool(torch.isfinite(features).all()):
            raise ValueError("Retrieval features must be finite [batch,length,8]")
        memory = self.mem.expand(encoded.size(0), -1, -1)
        keys = F.normalize(memory, p=2, dim=-1).transpose(-2, -1)
        tau = encoded.new_ones(encoded.shape[:2])
        alpha = encoded.new_full(encoded.shape[:2], 0.5)
        if self.arm == ARMS[1]:
            # Delegate native cosine/topk/softmax arithmetic exactly.
            residual, trace = self.retrieve(encoded)
            trace.update(weights=trace["topk_similarity"].softmax(-1), alpha=alpha, temperature=tau)
            return residual, trace
        if self.arm == ARMS[4]:
            query = encoded + 0.5 * (features @ self.query_weight)
            scores = F.normalize(query, p=2, dim=-1) @ keys
            gate = features @ self.context_gate_weight + self.context_gate_bias
            tau = torch.exp(math.log(2.) * torch.tanh(gate))
        else:
            time_query = encoded + features[..., :4] @ self.time_query_weight
            quantity_query = encoded + features[..., 4:] @ self.quantity_query_weight
            if self.arm == ARMS[3]:
                alpha = torch.sigmoid(features @ self.context_gate_weight + self.context_gate_bias)
            time_score = F.normalize(time_query, p=2, dim=-1) @ keys
            quantity_score = F.normalize(quantity_query, p=2, dim=-1) @ keys
            scores = alpha.unsqueeze(-1) * time_score + (1. - alpha).unsqueeze(-1) * quantity_score
        top_scores, indices = torch.topk(scores, self.topk, dim=-1)
        values = memory.unsqueeze(1).expand(-1, encoded.size(1), -1, -1)
        selected = values.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 64))
        weights = (top_scores / tau.unsqueeze(-1)).softmax(-1)
        return (selected * weights.unsqueeze(-1)).sum(2), {
            "prototype_indices": indices, "topk_similarity": top_scores, "weights": weights,
            "alpha": alpha, "temperature": tau}


class CountAwareTitanContextualRetrieval(CountAwareTitanCNNGRU):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, retrieval_arm, **kwargs):
        role_for_arm(retrieval_arm)
        stats = _statistics({"time_scale": kwargs.get("time_scale", 3.),
            "time_log_mean": kwargs.get("time_initial_location", 0.),
            "time_log_std": kwargs.get("time_initial_scale", 1.),
            "quantity_log_mean": train_log_mean, "quantity_log_std": kwargs.get("train_log_std", 1.)})
        # Parent _encode_base dispatches on candidate_arm: retain its exact
        # frozen CNN+GRU name and give retrieval its own explicit identity.
        super().__init__(hidden_dim, train_log_mean, max_seq_len, candidate_arm=BASELINE, **kwargs)
        self.retrieval_arm, self.retrieval_statistics = retrieval_arm, stats
        if retrieval_arm != ARMS[0]:
            self.lmm = ContextualLocalMemoryMatcher(self.lmm, retrieval_arm)
        self.register_buffer("contextual_retrieval_identity", identity(arm=retrieval_arm, statistics=stats))

    def observed_features(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if dts.ndim != 2 or history_quantities.shape != dts.shape:
            raise ValueError("Event inputs must have shape [batch,length]")
        _mask(mask, tuple(dts.shape), "mask")
        valid = mask.to(dts.device)
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(dts.shape), "memory_write_mask")
        observed = valid if memory_write_mask is None else valid & memory_write_mask.to(dts.device)
        dt = torch.where(observed, dts, 0.)
        qty = torch.where(observed, history_quantities, 0.)
        if (not bool(torch.isfinite(dt).all()) or bool((dt < 0).any())
                or not bool(torch.isfinite(qty).all()) or bool((qty < 0).any())):
            raise ValueError("Observed durations and quantities must be finite and nonnegative")
        stats = self.retrieval_statistics
        time_available = observed & (dt > 0)
        safe_dt = torch.where(time_available, dt, 1.)
        time_points = (torch.log(safe_dt / stats["time_scale"]) - stats["time_log_mean"]) / stats["time_log_std"]
        quantity_points = ((torch.log1p(qty) - stats["quantity_log_mean"]) / stats["quantity_log_std"]
            if stats["quantity_log_std"] > 0 else torch.zeros_like(qty))
        time = _prefix_summaries(time_points, time_available, valid, observed)
        quantity = _prefix_summaries(quantity_points, observed, valid, observed)
        return torch.cat((time, quantity), -1)

    def encode_task_states(self, dts, history_quantities, mask, *, memory_write_mask=None):
        if self.retrieval_arm == ARMS[0]:
            return super().encode_task_states(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        base = self._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        observed = mask if memory_write_mask is None else mask & memory_write_mask
        if self.retrieval_arm == ARMS[1]:
            combined = self.lmm(base)
        else:
            features = self.observed_features(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
            residual, _ = self.lmm.retrieve_context(base, features.to(dtype=base.dtype))
            combined = base + residual
        encoded = torch.where(observed.to(base.device).unsqueeze(-1), combined, 0.)
        return encoded, encoded

    def retrieval_trace(self, dts, history_quantities, mask, *, memory_write_mask=None):
        base = self._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        features = self.observed_features(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        if self.retrieval_arm == ARMS[0]:
            _, trace = self.lmm.retrieve(base)
            trace.update(weights=torch.full_like(trace["topk_similarity"], 0.25),
                alpha=base.new_full(base.shape[:2], 0.5), temperature=base.new_ones(base.shape[:2]))
        else:
            _, trace = self.lmm.retrieve_context(base, features.to(dtype=base.dtype))
        return {**trace, "features": features}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get("contextual_retrieval_identity")
        expected = identity(arm=self.retrieval_arm, statistics=self.retrieval_statistics)
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), expected):
            raise ValueError("Contextual retrieval checkpoint identity mismatch")
        # Includes every inherited tensor, all new tensors and exact shapes;
        # the inherited loader separately binds frozen CNN+GRU identity.
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get("encoder_config", {})
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    identified = (payload.get("backbone") in ARMS or isinstance(meta, Mapping)
        and meta.get("backbone_contract_id") == CONTRACT_ID or any(isinstance(state, Mapping)
        and ("contextual_retrieval_identity" in state or any(str(key).startswith("lmm.")
            and key != "lmm.mem" for key in state)) for state in states))
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError("Contextual retrieval checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping):
        raise ValueError("Contextual retrieval checkpoint metadata must be a mapping")
    stats = _statistics(meta.get("retrieval_train_statistics"))
    required = metadata(meta.get("d_model"), arm=expected_backbone, statistics=stats)
    resume = payload.get("resume_identity", {})
    head, interface = meta.get("time_head", {}), payload.get("interface_meta", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(key) != value for key, value in required.items())
            or resume.get("backbone") != expected_backbone
            or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)):
        raise ValueError("Contextual retrieval metadata, role or normalization mismatch")
    if (not states or not isinstance(head, Mapping) or not isinstance(interface, Mapping)
            or head.get("time_scale") != stats["time_scale"]
            or head.get("time_initial_location") != stats["time_log_mean"]
            or head.get("time_initial_scale") != stats["time_log_std"]
            or interface.get("train_target_mean") != stats["quantity_log_mean"]
            or interface.get("train_target_std") != stats["quantity_log_std"]):
        raise ValueError("Retrieval normalizer differs from inherited Train head/interface statistics")
    # Delegate inherited split/head/selector and exact CNN/GRU state checks via
    # a projection of labels/metadata only, without modifying actual payload.
    old_meta = {**meta, **parent_metadata(meta.get("d_model"), arm=BASELINE)}
    projected = {**payload, "backbone": BASELINE, "encoder_config": old_meta,
        "resume_identity": {**resume, "backbone": BASELINE,
            "arguments": {**resume.get("arguments", {}),
                "model_role": parent_metadata(arm=BASELINE)["model_role"]}}}
    parent_validate_checkpoint(projected, BASELINE)
    expected_keys = {"lmm.mem": (1, 64, 64)}
    if expected_backbone in ARMS[2:4]:
        expected_keys.update({"lmm.time_query_weight": (4, 64), "lmm.quantity_query_weight": (4, 64)})
    elif expected_backbone == ARMS[4]:
        expected_keys["lmm.query_weight"] = (8, 64)
    if expected_backbone in ARMS[3:]:
        expected_keys.update({"lmm.context_gate_weight": (8,), "lmm.context_gate_bias": ()})
    for state in states:
        value = state.get("contextual_retrieval_identity")
        if (not isinstance(value, torch.Tensor)
                or not torch.equal(value.cpu(), identity(arm=expected_backbone, statistics=stats))):
            raise ValueError("Contextual retrieval state identity mismatch")
        if {key for key in state if str(key).startswith("lmm.")} != set(expected_keys):
            raise ValueError("Contextual retrieval score parameter keys mismatch")
        if any(not isinstance(state.get(key), torch.Tensor) or tuple(state[key].shape) != shape
                or not bool(torch.isfinite(state[key]).all()) for key, shape in expected_keys.items()):
            raise ValueError("Contextual retrieval bank/score state shape mismatch")
    return True
