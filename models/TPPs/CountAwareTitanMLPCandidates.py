"""Two capacity-matched history corrections for the six-fit A100 screen."""
import hashlib
import json
import torch
from torch.nn import functional as F
from models.TPPs.CountAwareTitanCoreAblation import (
    CountAwareTitanCoreAblation, HistoryCorrection, metadata as base_metadata,
)
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices, _mask, _complete_state

MODES = {'titantpp_history_cross_product': 'cross_product',
         'titantpp_history_recent4_mean': 'recent4_mean'}
ARMS = tuple(MODES)
ROLE = 'observed_time_mlp_candidates_v1'


def metadata(hidden_dim, mode):
    if hidden_dim != 64 or mode not in MODES.values():
        raise ValueError('MLP candidates require width64 and an explicit candidate')
    result = base_metadata(hidden_dim, 'mlp')
    result.update(candidate_name=next(a for a,m in MODES.items() if m == mode),
                  backbone_contract_id='titantpp_mlp_candidates_v1', model_role=ROLE,
                  mlp_candidate=mode, additional_parameter_count=6144,
                  branch_availability='original_thresholds', residual_divisor=8,
                  history_context='immediate_predecessor' if mode == 'cross_product' else 'mean_up_to_four_previous_observed_states_within_segment',
                  multilag_detail_projections='gelu_current_times_previous' if mode == 'cross_product' else 'gelu_current_plus_recent4_mean',
                  bottleneck_width=4, initialization='same_MLP_input_and_zero_output_forked_rng')
    return result


def identity(hidden_dim, mode):
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(hidden_dim, mode), sort_keys=True).encode()).digest()), dtype=torch.uint8)


class CandidateCorrection(HistoryCorrection):
    def __init__(self, hidden_dim, candidate):
        metadata(hidden_dim, candidate)
        super().__init__(hidden_dim, 'mlp')
        self.candidate = candidate

    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid, tuple(hidden.shape[:2]), 'valid')
        if memory_write_mask is not None:
            _mask(memory_write_mask, tuple(valid.shape), 'memory_write_mask')
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.)
        if not torch.isfinite(safe).all():
            raise ValueError('Nonfinite observed history')
        source, available = lag_source_indices(valid, memory_write_mask=memory_write_mask, mode='local')
        weights = torch.cat([p.weight for p in self.input_projections])
        current = F.linear(safe, weights[:, :self.hidden_dim]).unflatten(-1, (8, 4))
        previous = F.linear(safe, weights[:, self.hidden_dim:]).unflatten(-1, (8, 4))
        if self.candidate == 'cross_product':
            context = previous.gather(1, source.unsqueeze(-1).expand(-1, -1, -1, 4))
            features = F.gelu(current, approximate='none') * context
        else:
            # Follow observed predecessors. Padding is skipped; withheld events reset eligibility.
            predecessor = source[:, :, 0]
            predecessor_available = available[:, :, 0]
            index, alive = predecessor, predecessor_available
            context = torch.zeros_like(previous)
            count = torch.zeros_like(predecessor, dtype=previous.dtype)
            for _ in range(4):
                gathered = previous.gather(1, index[:, :, None, None].expand_as(previous))
                context = context + torch.where(alive[:, :, None, None], gathered, 0.)
                count = count + alive.to(count.dtype)
                alive = alive & predecessor_available.gather(1, index)
                index = predecessor.gather(1, index)
            context = context / count.clamp_min(1)[:, :, None, None]
            features = F.gelu(current + context, approximate='none')
        features = torch.where(available.unsqueeze(-1), features, 0.)
        return F.linear(features.flatten(-2), torch.cat([p.weight for p in self.output_projections], dim=1)) / 8


class CountAwareTitanMLPCandidate(CountAwareTitanCoreAblation):
    def __init__(self, hidden_dim, train_log_mean, max_seq_len, *, candidate_mode, **kwargs):
        super().__init__(hidden_dim, train_log_mean, max_seq_len, core_mode='mlp', **kwargs)
        self.candidate_mode = candidate_mode
        self.multilag_detail = CandidateCorrection(hidden_dim, candidate_mode)
        self.register_buffer('mlp_candidate_identity', identity(hidden_dim, candidate_mode))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        value = state_dict.get('mlp_candidate_identity')
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(self.core_hidden_dim, self.candidate_mode)):
            raise ValueError('MLP candidate checkpoint identity mismatch')
        _complete_state(self, state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def validate_checkpoint(payload, expected_backbone):
    meta = payload.get('encoder_config', {})
    states = [payload[k] for k in ('model_state_dict', 'best_state_dict') if k in payload]
    identified = payload.get('backbone') in ARMS or 'mlp_candidate' in meta or any('mlp_candidate_identity' in s for s in states)
    if expected_backbone not in ARMS:
        if identified:
            raise ValueError('MLP candidate cannot be relabelled')
        return False
    mode = MODES[expected_backbone]
    expected = metadata(meta.get('d_model'), mode)
    resume, interface = payload.get('resume_identity', {}), payload.get('interface_meta', {})
    if (payload.get('backbone') != expected_backbone or any(meta.get(k) != v for k,v in expected.items())
            or payload.get('evaluation_scope') != 'validation_only' or payload.get('held_out_test_evaluated') is not False
            or payload.get('variant') != 'count_only_log_regression' or resume.get('backbone') != expected_backbone
            or resume.get('interface_meta') != interface or resume.get('arguments', {}).get('model_role') != ROLE
            or resume.get('checkpoint_monitor') != 'validation_raw_quantity_rmse'
            or meta.get('time_head', {}).get('observation_likelihood') != interface.get('time_head', {}).get('observation_likelihood')):
        raise ValueError('MLP candidate metadata, split, head, selection or role mismatch')
    for state in states:
        value = state.get('mlp_candidate_identity')
        if not isinstance(value, torch.Tensor) or not torch.equal(value.cpu(), identity(meta.get('d_model'), mode)):
            raise ValueError('MLP candidate state identity mismatch')
    if not states and (len(payload.get('checkpoint_state_sha256', '')) != 64 or payload.get('best_epoch', 0) < 1):
        raise ValueError('Unbound MLP candidate summary')
    return True
