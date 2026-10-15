"""Q/K-only observed CNN structural ablation; the extra V filter is absent.

Construct the original CNN first, retain initialized Q/K and all common tensors,
then omit V local coefficients without consuming RNG. Q/K and the common QKV
projector train normally. This is a reduced-capacity ablation, not a new operator.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.TPPs import CountAwareTitanCNNGRUComponents as components
from models.TPPs.CountAwareTitanCNNGRU import ObservedCausalQKVMemoryAttention, causal_depthwise_residual
from models.TPPs.CountAwareTitanCNNHistoryControls import _identity_matches
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

CNN = "titantpp_cnn"
QK_ONLY = "titantpp_cnn_qk_only_v1"
ARMS = (QK_ONLY,)
ALL_ARMS = (CNN, *ARMS)
LEGACY_CNN = components.CNN
CONTRACT_ID = "titantpp_qk_only_cnn_20261016_v1"
IDENTITY_BUFFER = "qk_only_cnn_identity"
EXTRA_PREFIX = "encoder.layers.0.attn.causal_"
QK_KEYS = tuple(EXTRA_PREFIX + name + "_kernel" for name in ("q", "k"))
REMOVED_KEY = "encoder.layers.0.attn.causal_v_kernel"
PARAMETER_DELTA = ADDED_PARAMETERS = -192
LOCAL_CNN_PARAMETERS = 384
ACTIVE_PARAMETERS = {arm: components.ACTIVE_PARAMETERS[LEGACY_CNN] + (PARAMETER_DELTA if arm in ARMS else 0)
                     for arm in ALL_ARMS}


def role_for_arm(arm):
    if arm not in ALL_ARMS:
        raise ValueError("Unsupported QK-only CNN arm")
    return "observed_time_" + arm.removeprefix("titantpp_") + ("_v1" if arm == CNN else "")


def metadata(hidden_dim=64, *, arm, max_seq_len=256):
    role_for_arm(arm)
    if type(max_seq_len) is not int or max_seq_len < 1:
        raise ValueError("Context length must be a positive integer")
    result = components.metadata(hidden_dim, arm=LEGACY_CNN)
    candidate = arm in ARMS
    count = LOCAL_CNN_PARAMETERS if candidate else components.CNN_PARAMETERS
    position_difference = hidden_dim * (max_seq_len - 256)
    result.update(candidate_name=arm, model_role=role_for_arm(arm), backbone_contract_id=CONTRACT_ID,
        baseline_backbone=LEGACY_CNN, qk_only_enabled=candidate,
        qk_only_insertion="encoder_block0_event_qk_after_projection_before_memory_concat",
        qk_only_qk="original_independent_trainable_three_tap",
        qk_only_formula="Q'=Q+sum_i a_i*Q_observed_i; K'=K+sum_i b_i*K_observed_i; V'=V" if candidate else None,
        qk_only_kernel_shapes=[[3, 64], [3, 64]] if candidate else [[3, 64]] * 3,
        qk_only_tap_order=["current", "previous_observed", "second_previous_observed"],
        qk_only_padding="skip_observed_lags", qk_only_withheld_valid="reset_observed_segment",
        qk_only_initialization="original_CNN_construct_then_remove_V_no_rng_consumption" if candidate else "original_CNN",
        qk_only_parameter_delta=PARAMETER_DELTA if candidate else 0,
        qk_only_unused_parameters=0, qk_only_v_filter_present=not candidate,
        active_local_cnn_parameters=count, active_value_cnn_parameters=0 if candidate else 192,
        original_value_kernel_absent=candidate,
        value_operation="unmixed_common_value_projection" if candidate else "original_three_tap_value_residual",
        qk_only_v_projection="original_normally_trainable_common_qkv_projection",
        qk_only_operation_time_override=False, qk_only_added_nonlinearity=False,
        qk_only_added_gate=False, qk_only_added_norm=False, qk_only_warm_start=False,
        cnn_parameter_count=count, registered_cnn_parameter_count=count,
        registered_correction_parameter_count=count + components.GRU_PARAMETERS,
        registered_additional_parameter_count=count + components.GRU_PARAMETERS,
        parameter_count_context_length=max_seq_len,
        active_trainable_parameter_count=ACTIVE_PARAMETERS[arm] + position_difference,
        registered_parameter_count=components.REGISTERED_PARAMETERS + position_difference + (PARAMETER_DELTA if candidate else 0),
        additional_parameter_count=count, active_correction_parameter_count=count,
        common_encoder_initialization="exact_original_CNN_common_tensors_and_rng_except_removed_V_kernel",
        active_budget_limitations="576 to 384 local coefficients; V local CNN removal and reduced capacity are the intervention; not parameter matched")
    return result


def identity(*, arm):
    if arm not in ARMS:
        raise ValueError("Only the QK-only candidate has this state identity")
    raw = json.dumps(metadata(arm=arm), sort_keys=True).encode()
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8)


class QKOnlyAttention(ObservedCausalQKVMemoryAttention):
    def __init__(self, base):
        nn.Module.__init__(self)
        for name in ("d_model", "n_heads", "use_causal", "head_dim", "scale", "qkv", "out_proj", "drop",
                     "contextual_mem_size", "persistent_mem_size", "persistent_mem", "_ctx_buf", "_ctx_mem",
                     "causal_q_kernel", "causal_k_kernel"):
            setattr(self, name, getattr(base, name))
        # Reuse the original initialized objects. No V filter, dummy parameter,
        # tensor initialization, or random draw is introduced.
        self.train(base.training)

    def forward(self, x, mask=None, *, observed_sources, observed_available,
                event_attention_bias=None, return_event_attention=False):
        batch, length, _ = x.shape
        if mask is None:
            raise ValueError("Observed CNN attention requires an explicit mask")
        _mask(mask, (batch, length), "attention mask")
        if not torch.equal(mask, observed_available[..., 0]):
            raise ValueError("CNN context does not match observed attention mask")
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        q = q + causal_depthwise_residual(q, self.causal_q_kernel, observed_sources, observed_available)
        k = k + causal_depthwise_residual(k, self.causal_k_kernel, observed_sources, observed_available)
        # Projected event V reaches attention unchanged; no local V filter exists.
        # Frozen attention arithmetic; memory tokens bypass every CNN filter.
        memory_parts = []
        if self._ctx_mem is not None and self._ctx_mem.numel() > 0:
            memory_parts.append(self._ctx_mem.to(device=x.device, dtype=x.dtype))
        if self.persistent_mem is not None:
            memory_parts.append(self.persistent_mem.to(device=x.device, dtype=x.dtype).expand(batch, -1, -1))
        memory_length = 0
        if memory_parts:
            memory = torch.cat(memory_parts, dim=1)
            memory_length = memory.size(1)
            k, v = torch.cat([memory, k], dim=1), torch.cat([memory, v], dim=1)
        scores = torch.matmul(self._split_heads(q), self._split_heads(k).transpose(-2, -1)) * self.scale
        if event_attention_bias is not None:
            if event_attention_bias.shape != (batch, self.n_heads, length, length):
                raise ValueError("Event attention bias shape mismatch")
            scores = scores + F.pad(event_attention_bias.to(dtype=scores.dtype, device=x.device), (memory_length, 0))
        full_mask = None
        if self.use_causal:
            full_mask = torch.cat((torch.ones(length, memory_length, device=x.device, dtype=torch.bool),
                torch.tril(torch.ones(length, length, device=x.device, dtype=torch.bool))), 1)
        valid_keys = torch.cat((torch.ones(batch, memory_length, device=x.device, dtype=torch.bool), mask), 1)
        key_mask = valid_keys[:, None, None, :]
        full_mask = key_mask if full_mask is None else full_mask[None, None, :, :] & key_mask
        if memory_length == 0:
            invalid_queries = ~mask
            if invalid_queries.any():
                full_mask = full_mask.expand(batch, 1, length, length).clone()
                batch_ids, positions = invalid_queries.nonzero(as_tuple=True)
                full_mask[batch_ids, 0, positions, positions] = True
        scores = scores.masked_fill(~full_mask, float("-inf"))
        attention = F.softmax(scores, dim=-1)
        event_attention = attention[..., memory_length:].mean(1) if return_event_attention else None
        attention = self.drop(attention)
        output = self.out_proj(self._merge_heads(torch.matmul(attention, self._split_heads(v))))
        output = output * mask.to(dtype=output.dtype).unsqueeze(-1)
        return (output, event_attention) if return_event_attention else output


class CountAwareTitanQKOnlyCNN(components.CountAwareTitanCNNGRUComponents):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, qk_only_arm=QK_ONLY, **kwargs):
        if qk_only_arm not in ARMS:
            raise ValueError("Use the original model for the baseline alias")
        metadata(hidden_dim, arm=qk_only_arm, max_seq_len=max_seq_len)
        super().__init__(hidden_dim, train_log_mean, max_seq_len, component_arm=LEGACY_CNN, **kwargs)
        self.qk_only_arm = qk_only_arm
        self.encoder.layers[0].attn = QKOnlyAttention(self.encoder.layers[0].attn)
        self.register_buffer(IDENTITY_BUFFER, identity(arm=qk_only_arm))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not _identity_matches(state_dict.get(IDENTITY_BUFFER), identity(arm=self.qk_only_arm)):
            raise ValueError("QK-only CNN identity mismatch")
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    """Bind the reduced architecture without fabricating a removed V parameter."""
    meta = payload.get("encoder_config", {})
    states = [payload[k] for k in ("model_state_dict", "best_state_dict") if k in payload]
    identified = (payload.get("backbone") in ALL_ARMS
        or isinstance(meta, Mapping) and meta.get("backbone_contract_id") == CONTRACT_ID
        or any(isinstance(state, Mapping) and IDENTITY_BUFFER in state for state in states))
    if expected_backbone not in ALL_ARMS:
        if identified:
            raise ValueError("QK-only CNN checkpoint cannot be relabelled")
        return False
    if not isinstance(meta, Mapping) or not states:
        raise ValueError("QK-only CNN metadata and state are required")
    required = metadata(meta.get("d_model"), arm=expected_backbone, max_seq_len=meta.get("max_len"))
    resume, interface = payload.get("resume_identity", {}), payload.get("interface_meta", {})
    head = meta.get("time_head", {})
    if (payload.get("backbone") != expected_backbone or any(meta.get(k) != v for k, v in required.items())
        or not isinstance(resume, Mapping) or resume.get("backbone") != expected_backbone
        or resume.get("arguments", {}).get("model_role") != role_for_arm(expected_backbone)
        or payload.get("evaluation_scope") != "validation_only" or payload.get("held_out_test_evaluated") is not False
        or payload.get("variant") != "count_only_log_regression"
        or resume.get("interface_meta") != interface or resume.get("checkpoint_monitor") != "validation_raw_quantity_rmse"
        or head.get("mode") != "heteroscedastic_lognormal_duration"
        or not isinstance(head.get("observation_likelihood"), Mapping)
        or head["observation_likelihood"].get("mode") != "positive_integer_round_clamp_v1"
        or head["observation_likelihood"] != interface.get("time_head", {}).get("observation_likelihood")):
        raise ValueError("QK-only CNN metadata, split, head, selector or role mismatch")
    if expected_backbone == CNN:
        if any(IDENTITY_BUFFER in state for state in states):
            raise ValueError("QK-only candidate cannot be relabelled as baseline")
        parent = {**payload, "backbone": LEGACY_CNN, "encoder_config": {**meta, **components.metadata(arm=LEGACY_CNN)},
            "resume_identity": {**resume, "backbone": LEGACY_CNN,
                "arguments": {**resume.get("arguments", {}), "model_role": components.role_for_arm(LEGACY_CNN)}}}
        return components.validate_checkpoint(parent, LEGACY_CNN)
    expected_shapes = {"multilag_detail.input_projection.weight": (54, 64),
        "multilag_detail.output_projection.weight": (64, 54),
        "multilag_detail.gru.weight_ih_l0": (162, 54), "multilag_detail.gru.weight_hh_l0": (162, 54),
        "multilag_detail.gru.bias_ih_l0": (162,), "multilag_detail.gru.bias_hh_l0": (162,)}
    for state in states:
        if not isinstance(state, Mapping):
            raise ValueError("QK-only CNN state must be a mapping")
        if (not _identity_matches(state.get(IDENTITY_BUFFER), identity(arm=expected_backbone))
            or not _identity_matches(state.get("cnn_gru_component_identity"), components.identity(arm=LEGACY_CNN))
            or not _identity_matches(state.get("cnn_gru_identity"), components.parent_identity(arm=components.FULL))):
            raise ValueError("QK-only CNN state identity mismatch")
        filters = {k for k in state if str(k).startswith(EXTRA_PREFIX) and str(k).endswith("_kernel")}
        if (filters != set(QK_KEYS) or REMOVED_KEY in state
            or any("value_difference_kernel" in str(k) or "qk_only_kernel" in str(k) for k in state)
            or {k for k in state if str(k).startswith("multilag_detail.")} != set(expected_shapes)):
            raise ValueError("QK-only state has a removed V filter or unexpected correction parameter")
        for key, shape in {**expected_shapes, **{k: (3, 64) for k in QK_KEYS}}.items():
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError("QK-only retained correction tensor invalid")
        if any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in state.values()):
            raise ValueError("QK-only nonfinite or non-tensor state")
        if "history_width_identity" in state or "core_ablation_identity" in state:
            raise ValueError("QK-only contains inherited architecture identity")
    return True
