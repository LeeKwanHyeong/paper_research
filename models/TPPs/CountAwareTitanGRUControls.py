"""Matched direct-context controls; shared Encoder1 remains contextual."""
from __future__ import annotations
import hashlib
import json
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from models.TPPs.CountAwareTitanCNNGRU import GRUHistoryCorrection, observed_context
from models.TPPs.CountAwareTitanHistoryWidth import CountAwareTitanHistoryWidth, HistoryWidthCorrection, metadata as width_metadata
from models.TPPs.CountAwareTitanMultiLagDetail import _complete_state, _mask

ANCHOR = 'titantpp_gru54'
PAIR = 'titantpp_gru54_pair'
MLP1 = 'titantpp_pair_mlp16_all_alpha1'
MLP8 = 'titantpp_pair_mlp16_all_div8'
ARMS = (PAIR, MLP1, MLP8)
ROLE_BY_ARM = {arm: 'observed_time_' + arm.removeprefix('titantpp_') + '_v1' for arm in ARMS}
CONTRACT_ID = 'titantpp_gru_matched_controls_v1'

def role_for_arm(arm):
    if arm not in ARMS: raise ValueError('Foreign GRU control')
    return ROLE_BY_ARM[arm]

def metadata(hidden_dim=64, *, arm):
    role = role_for_arm(arm)
    result = width_metadata(hidden_dim, width=16)
    result.update(candidate_name=arm, backbone_contract_id=CONTRACT_ID, model_role=role,
        history_context='adapter_direct_immediate_previous_and_current',
        shared_encoder_context='causal_all_observed_history_not_reset_at_withheld_rows',
        branch_availability='all_eight_active_after_two_observed_events_in_segment',
        residual_divisor=8 if arm == MLP8 else 1,
        initialization='inherited_forked_cpu_rng_zero_output',
        additional_parameter_count=24732 if arm == PAIR else 24576,
        correction_family='two_step_GRU54_restarted_per_output' if arm == PAIR else 'eight_bias_free_pair_GELU_MLP16',
        gru_hidden_size=54 if arm == PAIR else None,
        attention_changed=False, cnn_enabled=False)
    if arm == PAIR:
        for key in ('history_mlp_width', 'bottleneck_width', 'multilag_detail_rank'):
            result.pop(key, None)
        result['multilag_detail_projections'] = '64_to54_GRU54_54to64'
    return result

def identity(hidden_dim=64, *, arm):
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(hidden_dim,arm=arm),sort_keys=True).encode()).digest()),dtype=torch.uint8)

class PairGRUHistoryCorrection(GRUHistoryCorrection):
    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid, tuple(hidden.shape[:2]), 'valid')
        if memory_write_mask is not None: _mask(memory_write_mask,tuple(valid.shape),'memory_write_mask')
        observed = valid if memory_write_mask is None else valid & memory_write_mask
        safe = torch.where(observed.unsqueeze(-1), hidden, 0.)
        if not bool(torch.isfinite(safe).all()): raise ValueError('Nonfinite observed history')
        _, sources, available = observed_context(valid, memory_write_mask=memory_write_mask)
        previous = safe.gather(1, sources[:,:,1].unsqueeze(-1).expand_as(safe))
        pairs = torch.stack((self.input_projection(previous),self.input_projection(safe)),dim=2)
        batch,length = hidden.shape[:2]
        values,_ = self.gru(pairs.reshape(batch*length,2,54))
        delta = self.output_projection(values[:,-1]).reshape(batch,length,64)
        return torch.where(available[:,:,1].unsqueeze(-1),delta,0.)

class AllActivePairMLP16(HistoryWidthCorrection):
    def __init__(self,hidden_dim=64,*,divisor=1):
        if divisor not in (1,8): raise ValueError('Fixed divisor must be1 or8')
        super().__init__(hidden_dim,width=16); self.divisor=divisor
    def forward(self,hidden,valid,*,memory_write_mask=None):
        _mask(valid,tuple(hidden.shape[:2]),'valid')
        if memory_write_mask is not None: _mask(memory_write_mask,tuple(valid.shape),'memory_write_mask')
        observed=valid if memory_write_mask is None else valid & memory_write_mask
        safe=torch.where(observed.unsqueeze(-1),hidden,0.)
        if not bool(torch.isfinite(safe).all()): raise ValueError('Nonfinite observed history')
        _,sources,available=observed_context(valid,memory_write_mask=memory_write_mask)
        weights=torch.cat([p.weight for p in self.input_projections])
        current=F.linear(safe,weights[:,:64]).unflatten(-1,(8,16))
        previous=F.linear(safe,weights[:,64:]).unflatten(-1,(8,16))
        index=sources[:,:,1].unsqueeze(-1).unsqueeze(-1).expand(-1,-1,8,16)
        features=current+previous.gather(1,index)
        features=F.gelu(torch.where(available[:,:,1,None,None],features,0.),approximate='none')
        return F.linear(features.flatten(-2),torch.cat([p.weight for p in self.output_projections],dim=1))/self.divisor

class CountAwareTitanGRUControl(CountAwareTitanHistoryWidth):
    def __init__(self,hidden_dim,train_log_mean,max_seq_len,*,candidate_arm,**kwargs):
        metadata(hidden_dim,arm=candidate_arm)
        super().__init__(hidden_dim,train_log_mean,max_seq_len,history_mlp_width=16,**kwargs)
        del self._buffers['history_width_identity'];del self._buffers['core_ablation_identity']
        self.candidate_arm=candidate_arm
        self.multilag_detail=(PairGRUHistoryCorrection(hidden_dim) if candidate_arm==PAIR else
            AllActivePairMLP16(hidden_dim,divisor=8 if candidate_arm==MLP8 else 1))
        self.register_buffer('gru_control_identity',identity(hidden_dim,arm=candidate_arm))
    def load_state_dict(self,state_dict,strict=True,assign=False):
        if not torch.equal(state_dict.get('gru_control_identity',torch.empty(0)).cpu(),identity(arm=self.candidate_arm)):
            raise ValueError('GRU control checkpoint identity mismatch')
        _complete_state(self,state_dict)
        return nn.Module.load_state_dict(self,state_dict,strict=strict,assign=assign)

def validate_checkpoint(payload,expected_backbone):
    meta=payload.get('encoder_config',{});states=[payload[k] for k in ('model_state_dict','best_state_dict') if k in payload]
    identified=payload.get('backbone') in ARMS or meta.get('backbone_contract_id')==CONTRACT_ID or any('gru_control_identity' in s for s in states)
    if expected_backbone not in ARMS:
        if identified: raise ValueError('GRU control cannot be relabelled')
        return False
    expected=metadata(meta.get('d_model'),arm=expected_backbone)
    interface=payload.get('interface_meta',{});resume=payload.get('resume_identity',{});head=meta.get('time_head',{})
    if (payload.get('backbone')!=expected_backbone or any(meta.get(k)!=v for k,v in expected.items())
        or payload.get('evaluation_scope')!='validation_only' or payload.get('held_out_test_evaluated') is not False
        or payload.get('variant')!='count_only_log_regression' or resume.get('backbone')!=expected_backbone
        or resume.get('interface_meta')!=interface or resume.get('checkpoint_monitor')!='validation_raw_quantity_rmse'
        or resume.get('arguments',{}).get('model_role')!=role_for_arm(expected_backbone)
        or head.get('mode')!='heteroscedastic_lognormal_duration'
        or head.get('observation_likelihood',{}).get('mode')!='positive_integer_round_clamp_v1'
        or head.get('observation_likelihood')!=interface.get('time_head',{}).get('observation_likelihood')):
        raise ValueError('GRU control metadata/split/role/selector/head mismatch')
    shapes=({'multilag_detail.input_projection.weight':(54,64),'multilag_detail.output_projection.weight':(64,54),
        'multilag_detail.gru.weight_ih_l0':(162,54),'multilag_detail.gru.weight_hh_l0':(162,54),
        'multilag_detail.gru.bias_ih_l0':(162,),'multilag_detail.gru.bias_hh_l0':(162,)} if expected_backbone==PAIR else
        {f'multilag_detail.{name}_projections.{b}.weight':shape for b in range(8) for name,shape in (('input',(16,128)),('output',(64,16)))})
    for state in states:
        if not torch.equal(state.get('gru_control_identity',torch.empty(0)).cpu(),identity(arm=expected_backbone)):
            raise ValueError('GRU control identity mismatch')
        if set(k for k in state if k.startswith('multilag_detail.'))!=set(shapes): raise ValueError('Control state keys mismatch')
        if any(k in state for k in ('cnn_gru_identity','history_width_identity','core_ablation_identity')): raise ValueError('Foreign state identity')
        for k,shape in shapes.items():
            if not isinstance(state.get(k),torch.Tensor) or tuple(state[k].shape)!=shape or not bool(torch.isfinite(state[k]).all()):
                raise ValueError('Control correction state mismatch')
    if not states and (len(payload.get('checkpoint_state_sha256',''))!=64 or payload.get('best_epoch',0)<1): raise ValueError('Unbound control summary')
    return True
