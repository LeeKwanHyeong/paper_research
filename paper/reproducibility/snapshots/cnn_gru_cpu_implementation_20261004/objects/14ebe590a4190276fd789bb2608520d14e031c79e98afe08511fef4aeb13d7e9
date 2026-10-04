"""History MLP with the fixed divisor replaced by the active branch count.

The eight branches retain the original immediate-predecessor sources and
availability thresholds. This is not an eight-lag history model.
"""
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from models.TPPs.CountAwareTitanCoreAblation import (
    HistoryCorrection, CountAwareTitanCoreAblation, metadata as core_metadata,
)
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices, _mask, _complete_state

ARM = 'titantpp_history_mlp_active_norm'
ARMS = (ARM,)
ROLE = 'observed_time_active_branch_norm_v1'


def metadata(hidden_dim):
    m = core_metadata(hidden_dim, 'mlp')
    m.update(candidate_name=ARM, backbone_contract_id='titantpp_active_branch_norm_v1',
             model_role=ROLE, active_branch_normalization='sum_div_max_active_count_1',
             multilag_detail_residual_divisor='max_active_count_1')
    return m


def identity(hidden_dim):
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(hidden_dim), sort_keys=True).encode()).digest()), dtype=torch.uint8)


class ActiveBranchCorrection(HistoryCorrection):
    def __init__(self, hidden_dim):
        super().__init__(hidden_dim, 'mlp')

    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid, tuple(hidden.shape[:2]), 'valid')
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.)
        if not torch.isfinite(safe).all():
            raise ValueError('Nonfinite observed history')
        source, available = lag_source_indices(valid, memory_write_mask=memory_write_mask, mode='local')
        weight = torch.cat([p.weight for p in self.input_projections])
        current = F.linear(safe, weight[:, :self.hidden_dim]).unflatten(-1, (8, 4))
        previous = F.linear(safe, weight[:, self.hidden_dim:]).unflatten(-1, (8, 4))
        feature = current + previous.gather(1, source.unsqueeze(-1).expand(-1, -1, -1, 4))
        feature = F.gelu(torch.where(available.unsqueeze(-1), feature, 0.), approximate='none')
        output = F.linear(feature.flatten(-2), torch.cat([p.weight for p in self.output_projections], dim=1))
        return output / available.sum(-1, keepdim=True).clamp_min(1).to(output.dtype)


class CountAwareTitanActiveBranchNorm(CountAwareTitanCoreAblation):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, **kwargs):
        super().__init__(hidden_dim, train_log_mean, max_seq_len, core_mode='mlp', **kwargs)
        self.multilag_detail = ActiveBranchCorrection(hidden_dim)
        self.register_buffer('active_branch_norm_identity', identity(hidden_dim))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get('active_branch_norm_identity')
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(self.core_hidden_dim)):
            raise ValueError('Active branch normalization checkpoint identity mismatch')
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get('encoder_config', {})
    states = [payload[k] for k in ('model_state_dict', 'best_state_dict') if k in payload]
    identified = (payload.get('backbone') == ARM or 'active_branch_normalization' in meta
                  or any('active_branch_norm_identity' in s for s in states))
    if expected_backbone != ARM:
        if identified:
            raise ValueError('Active branch normalization cannot be relabelled')
        return False
    if payload.get('backbone') != ARM or any(meta.get(k) != v for k, v in metadata(meta.get('d_model')).items()):
        raise ValueError('Active branch normalization metadata mismatch')
    resume = payload.get('resume_identity', {})
    interface = payload.get('interface_meta', {})
    if (payload.get('evaluation_scope') != 'validation_only' or payload.get('held_out_test_evaluated') is not False
            or payload.get('variant') != 'count_only_log_regression'
            or resume.get('backbone') != ARM or resume.get('interface_meta') != interface
            or resume.get('arguments', {}).get('model_role') != ROLE
            or resume.get('checkpoint_monitor') != 'validation_raw_quantity_rmse'
            or meta.get('time_head', {}).get('mode') != 'heteroscedastic_lognormal_duration'
            or meta.get('time_head', {}).get('observation_likelihood') != interface.get('time_head', {}).get('observation_likelihood')):
        raise ValueError('Active branch normalization split/head/selector mismatch')
    for state in states:
        value = state.get('active_branch_norm_identity')
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(meta['d_model'])):
            raise ValueError('Active branch normalization state identity mismatch')
    if not states and (len(payload.get('checkpoint_state_sha256', '')) != 64 or payload.get('best_epoch', 0) < 1):
        raise ValueError('Unbound normalization summary')
    return True
