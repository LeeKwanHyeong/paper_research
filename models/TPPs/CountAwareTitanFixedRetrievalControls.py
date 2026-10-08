"""Exact active-budget controls for the prospective fixed retrieval question.

Mean, weighted and fixed delegate to their frozen implementations unchanged.
The new generic512 arm removes the old generic arm's nine temperature weights;
its only addition is a zero-initialized 8x64 contextual query projection.

Fixed averages two separately normalized cosine queries. Concatenating those
unit queries and duplicating the unit prototype key is algebraically equivalent
when both queries and the key have nonzero norm. Thus a two-view description
alone is not evidence of a novel retrieval family. The comparison with the joint
generic query tests a split-normalization inductive bias, with identical scalar
score range and temperature, rather than semantic independence of the views.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs import CountAwareTitanContextualRetrieval as contextual
from models.TPPs.CountAwareTitanCNNGRU import (
    CountAwareTitanCNNGRU, metadata as cnn_metadata,
    validate_checkpoint as cnn_validate_checkpoint,
)
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state

BASELINE = contextual.BASELINE
GENERIC512 = "titantpp_retrieval_generic512"
ARMS = (*contextual.ARMS[:3], GENERIC512)
CONTRACT_ID = "titantpp_fixed_retrieval_active512_v1"
ROLE_BY_ARM = {**{arm: contextual.role_for_arm(arm) for arm in ARMS[:3]},
    GENERIC512: "observed_time_retrieval_generic512_v1"}
EXTRA_PARAMETERS = dict(zip(ARMS, (0, 0, 512, 512)))


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported fixed retrieval control arm")
    return ROLE_BY_ARM[arm]


def identifiability_contract():
    """Read-only scientific limitations; no performance-derived decision rule."""
    return {
        "fixed_query_normalization": "separate_unit_time_query_and_unit_quantity_query",
        "generic512_query_normalization": "one_unit_joint_query",
        "common_key_normalization": "unit_cosine_prototype_key",
        "common_score_range": [-1.0, 1.0],
        "common_temperature": 1.0,
        "fixed_concat_equivalence": "cos(cat(unit(qt),unit(qq)),cat(unit(k),unit(k)))",
        "concat_equivalence_domain": "nonzero_query_and_key_norms_above_normalization_epsilon",
        "zero_norm_extension": "half_sum_of_individually_normalized_dot_products; concatenated dot divided_by_2",
        "scope": "split_vs_joint_query_normalization_inductive_bias_not_semantic_novelty_proof",
        "selection_vs_aggregation": "diagnose_mean_and_softmax_on_the_same_selected_top4_without_extra_fit",
        "bank_semantics": "train_learned_static_prototypes_not_observed_event_bank",
        "content_improvement_claim": False,
    }


def metadata(hidden_dim=64, *, arm, statistics=None):
    role_for_arm(arm)
    if arm != GENERIC512:
        # Checkpoint compatibility and initial identity of the three frozen arms
        # stay exact. The campaign contract carries the additional audit fields.
        return contextual.metadata(hidden_dim, arm=arm, statistics=statistics)
    result = contextual.metadata(hidden_dim, arm=contextual.ARMS[4], statistics=statistics)
    result.update(candidate_name=arm, backbone_contract_id=CONTRACT_ID,
        model_role=role_for_arm(arm), retrieval_kind="generic512",
        retrieval_fixed_temperature=1.0, retrieval_temperature="fixed_1",
        retrieval_temperature_bounds=None, retrieval_extra_parameter_count=512,
        retrieval_parameter_budget_comparison="fixed_generic512_exact_active_query_budget",
        retrieval_query="hidden_plus_half_all8_times_W8x64",
        retrieval_normalization_comparison="generic_joint_query_vs_fixed_separate_queries",
        retrieval_identifiability=identifiability_contract(),
        additional_parameter_count=25308 + 512)
    return result


def identity(*, arm, statistics):
    if arm != GENERIC512:
        return contextual.identity(arm=arm, statistics=statistics)
    encoded = json.dumps(metadata(arm=arm, statistics=statistics), sort_keys=True,
        separators=(",", ":")).encode()
    return torch.tensor(list(hashlib.sha256(encoded).digest()), dtype=torch.uint8)


class Generic512LocalMemoryMatcher(contextual.ContextualLocalMemoryMatcher):
    """Joint query control: exact active512, fixed tau=1, inherited tied bank."""
    def __init__(self, bank):
        nn.Module.__init__(self)
        if bank.mem.shape != (1, 64, 64) or (bank.d_model, bank.mem_size, bank.topk) != (64, 64, 4):
            raise ValueError("Generic512 requires the frozen 64x64 top4 bank")
        self.d_model, self.mem_size, self.topk = bank.d_model, bank.mem_size, bank.topk
        self.mem, self.arm = bank.mem, GENERIC512
        self.query_weight = nn.Parameter(torch.zeros(8, 64))

    def retrieve_context(self, encoded, features):
        if features.shape != (*encoded.shape[:2], 8) or not bool(torch.isfinite(features).all()):
            raise ValueError("Retrieval features must be finite [batch,length,8]")
        memory = self.mem.expand(encoded.size(0), -1, -1)
        keys = F.normalize(memory, p=2, dim=-1).transpose(-2, -1)
        query = encoded + 0.5 * (features @ self.query_weight)
        scores = F.normalize(query, p=2, dim=-1) @ keys
        top_scores, indices = torch.topk(scores, self.topk, dim=-1)
        values = memory.unsqueeze(1).expand(-1, encoded.size(1), -1, -1)
        selected = values.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, 64))
        weights = top_scores.softmax(-1)
        return (selected * weights.unsqueeze(-1)).sum(2), {
            "prototype_indices": indices, "topk_similarity": top_scores, "weights": weights,
            "alpha": encoded.new_full(encoded.shape[:2], 0.5),
            "temperature": encoded.new_ones(encoded.shape[:2])}


def fixed_concatenation_scores(encoded, features, time_weight, quantity_weight, memory):
    """Diagnostic equivalents including a defined zero-norm extension.

    Direct half-sum and concat dot/2 are equivalent for all finite inputs under
    F.normalize's epsilon convention. A literal concatenated cosine is only
    equivalent when both query blocks and the key are unit length; the returned
    mask prevents a zero norm from being silently called an equivalent cosine.
    """
    time_query = encoded + features[..., :4] @ time_weight
    quantity_query = encoded + features[..., 4:] @ quantity_weight
    time_unit = F.normalize(time_query, p=2, dim=-1)
    quantity_unit = F.normalize(quantity_query, p=2, dim=-1)
    key_unit = F.normalize(memory, p=2, dim=-1)
    fixed = 0.5 * (time_unit @ key_unit.transpose(-2, -1)) + 0.5 * (
        quantity_unit @ key_unit.transpose(-2, -1))
    query_concat = torch.cat((time_unit, quantity_unit), -1)
    key_concat = torch.cat((key_unit, key_unit), -1)
    concat_dot = 0.5 * (query_concat @ key_concat.transpose(-2, -1))
    concat_cosine = F.normalize(query_concat, p=2, dim=-1) @ F.normalize(
        key_concat, p=2, dim=-1).transpose(-2, -1)
    eps = 1e-12
    equivalent_domain = ((time_query.norm(dim=-1) > eps) & (quantity_query.norm(dim=-1) > eps)).unsqueeze(-1)
    equivalent_domain = equivalent_domain & (memory.norm(dim=-1) > eps).unsqueeze(-2)
    return {"fixed_scores": fixed, "concat_dot_half_scores": concat_dot,
        "concat_cosine_scores": concat_cosine, "concat_cosine_equivalent_domain": equivalent_domain}


class CountAwareTitanFixedRetrievalControls(contextual.CountAwareTitanContextualRetrieval):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, retrieval_arm, **kwargs):
        role_for_arm(retrieval_arm)
        # The temporary fixed score tensors are zero allocations, consume no
        # RNG, and are discarded before optimizer construction for generic512.
        inherited_arm = contextual.ARMS[2] if retrieval_arm == GENERIC512 else retrieval_arm
        super().__init__(hidden_dim, train_log_mean, max_seq_len, retrieval_arm=inherited_arm, **kwargs)
        if retrieval_arm == GENERIC512:
            self.lmm = Generic512LocalMemoryMatcher(self.lmm)
            self.retrieval_arm = retrieval_arm
            self.contextual_retrieval_identity = identity(arm=retrieval_arm, statistics=self.retrieval_statistics)

    def retrieval_diagnostics(self, dts, history_quantities, mask, *, memory_write_mask=None):
        """Same-model, same-score selection/aggregation audit; not a fit arm."""
        base = self._encode_base(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        features = self.observed_features(dts, history_quantities, mask, memory_write_mask=memory_write_mask)
        if self.retrieval_arm == ARMS[0]:
            _, trace = self.lmm.retrieve(base)
        else:
            _, trace = self.lmm.retrieve_context(base, features.to(dtype=base.dtype))
        memory = self.lmm.mem.expand(base.size(0), -1, -1)
        values = memory.unsqueeze(1).expand(-1, base.size(1), -1, -1)
        selected = values.gather(2, trace["prototype_indices"].unsqueeze(-1).expand(-1, -1, -1, 64))
        weights = trace["topk_similarity"].softmax(-1)
        result = {"prototype_indices": trace["prototype_indices"],
            "topk_similarity": trace["topk_similarity"], "features": features,
            "mean_on_same_subset": selected.mean(2),
            "softmax_on_same_subset": (selected * weights.unsqueeze(-1)).sum(2),
            "softmax_weights": weights, "identifiability": identifiability_contract(),
            "diagnostic_only_no_checkpoint_reselection": True}
        if self.retrieval_arm == ARMS[2]:
            result.update(fixed_concatenation_scores(base, features.to(base.dtype),
                self.lmm.time_query_weight, self.lmm.quantity_query_weight, memory))
        return result

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.retrieval_arm != GENERIC512:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)
        value = state_dict.get("contextual_retrieval_identity")
        expected = identity(arm=GENERIC512, statistics=self.retrieval_statistics)
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), expected):
            raise ValueError("Generic512 checkpoint identity mismatch")
        _complete_state(self, state_dict)
        return CountAwareTitanCNNGRU.load_state_dict(self, state_dict, strict=strict, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    """Bind new generic identity without relabelling any frozen pilot checkpoint."""
    meta = payload.get("encoder_config", {})
    if expected_backbone != GENERIC512:
        if payload.get("backbone") == GENERIC512 or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID:
            raise ValueError("Generic512 checkpoint cannot be relabelled")
        return contextual.validate_checkpoint(payload, expected_backbone)
    if not isinstance(meta, Mapping):
        raise ValueError("Generic512 checkpoint metadata must be a mapping")
    stats = contextual._statistics(meta.get("retrieval_train_statistics"))
    required = metadata(meta.get("d_model"), arm=GENERIC512, statistics=stats)
    resume = payload.get("resume_identity", {})
    if (payload.get("backbone") != GENERIC512 or any(meta.get(key) != value for key, value in required.items())
            or resume.get("backbone") != GENERIC512
            or resume.get("arguments", {}).get("model_role") != role_for_arm(GENERIC512)):
        raise ValueError("Generic512 metadata, role or normalization mismatch")
    head, interface = meta.get("time_head", {}), payload.get("interface_meta", {})
    if (not isinstance(head, Mapping) or not isinstance(interface, Mapping)
            or head.get("time_scale") != stats["time_scale"]
            or head.get("time_initial_location") != stats["time_log_mean"]
            or head.get("time_initial_scale") != stats["time_log_std"]
            or interface.get("train_target_mean") != stats["quantity_log_mean"]
            or interface.get("train_target_std") != stats["quantity_log_std"]):
        raise ValueError("Generic512 normalization differs from frozen Train head/interface")
    cnn_meta = {**meta, **cnn_metadata(meta.get("d_model"), arm=BASELINE)}
    projected = {**payload, "backbone": BASELINE, "encoder_config": cnn_meta,
        "resume_identity": {**resume, "backbone": BASELINE, "arguments": {
            **resume.get("arguments", {}), "model_role": cnn_meta["model_role"]}}}
    cnn_validate_checkpoint(projected, BASELINE)
    states = [payload[key] for key in ("model_state_dict", "best_state_dict") if key in payload]
    if not states:
        raise ValueError("Generic512 checkpoint requires complete model state")
    expected_keys = {"lmm.mem": (1, 64, 64), "lmm.query_weight": (8, 64)}
    for state in states:
        value = state.get("contextual_retrieval_identity")
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(arm=GENERIC512, statistics=stats)):
            raise ValueError("Generic512 state identity mismatch")
        if {key for key in state if str(key).startswith("lmm.")} != set(expected_keys):
            raise ValueError("Generic512 has unexpected score or temperature parameters")
        if any(not isinstance(state.get(key), torch.Tensor) or tuple(state[key].shape) != shape
                or not bool(torch.isfinite(state[key]).all()) for key, shape in expected_keys.items()):
            raise ValueError("Generic512 score state shape or finiteness mismatch")
    return True
