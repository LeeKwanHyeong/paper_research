"""Prespecified capacity and branch-availability controls for History MLP."""
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from models.TPPs.CountAwareTitanCoreAblation import CountAwareTitanCoreAblation, metadata as base_metadata
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices, _mask, _complete_state

MODES = {'titantpp_current_only_param_matched': 'current_only',
         'titantpp_all_available_history_mlp': 'all_available'}
ARMS = tuple(MODES)
ROLE = 'observed_time_history_controls_v1'

def metadata(hidden_dim, mode):
    if hidden_dim != 64 or mode not in MODES.values():
        raise ValueError('History controls require the prespecified width64')
    m = base_metadata(hidden_dim, 'mlp')
    m.update(candidate_name=next(k for k,v in MODES.items() if v == mode),
             backbone_contract_id='titantpp_history_controls_v1', model_role=ROLE,
             history_control=mode, bottleneck_width=6 if mode=='current_only' else 4,
             explicit_predecessor=mode!='current_only', additional_parameter_count=6144,
             branch_availability='original_thresholds' if mode=='current_only' else 'all_after_first',
             residual_divisor=8)
    return m

def identity(hidden_dim, mode):
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(hidden_dim,mode),sort_keys=True).encode()).digest()),dtype=torch.uint8)

class HistoryControl(nn.Module):
    def __init__(self, hidden_dim, mode):
        super().__init__(); metadata(hidden_dim,mode)
        self.hidden_dim,self.mode=hidden_dim,mode
        self.width=6 if mode=='current_only' else 4
        inputs=hidden_dim if mode=='current_only' else 2*hidden_dim
        with torch.random.fork_rng(devices=[]):
            self.input_projections=nn.ModuleList(nn.Linear(inputs,self.width,bias=False) for _ in range(8))
            self.output_projections=nn.ModuleList(nn.Linear(self.width,hidden_dim,bias=False) for _ in range(8))
            for p in self.output_projections:nn.init.zeros_(p.weight)

    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid,tuple(hidden.shape[:2]),'valid')
        observed=valid if memory_write_mask is None else valid & memory_write_mask
        safe=torch.where(observed.unsqueeze(-1),hidden,0.)
        if not torch.isfinite(safe).all():raise ValueError('Nonfinite observed history')
        source,available=lag_source_indices(valid,memory_write_mask=memory_write_mask,mode='local')
        weight=torch.cat([p.weight for p in self.input_projections])
        if self.mode=='current_only':
            feature=F.linear(safe,weight).unflatten(-1,(8,self.width))
        else:
            # Branch zero always names the real immediate predecessor when eligible.
            source=source[:,:,:1].expand(-1,-1,8)
            available=available[:,:,:1].expand(-1,-1,8)
            current=F.linear(safe,weight[:,:self.hidden_dim]).unflatten(-1,(8,self.width))
            previous=F.linear(safe,weight[:,self.hidden_dim:]).unflatten(-1,(8,self.width))
            feature=current+previous.gather(1,source.unsqueeze(-1).expand(-1,-1,-1,self.width))
        feature=F.gelu(torch.where(available.unsqueeze(-1),feature,0.),approximate='none')
        return F.linear(feature.flatten(-2),torch.cat([p.weight for p in self.output_projections],dim=1))/8

class CountAwareTitanHistoryControl(CountAwareTitanCoreAblation):
    def __init__(self,hidden_dim,train_log_mean,max_seq_len,*,control_mode,**kwargs):
        super().__init__(hidden_dim,train_log_mean,max_seq_len,core_mode='mlp',**kwargs)
        self.control_mode=control_mode
        self.multilag_detail=HistoryControl(hidden_dim,control_mode)
        self.register_buffer('history_control_identity',identity(hidden_dim,control_mode))

    def load_state_dict(self,state_dict,strict=True,assign=False):
        v=state_dict.get('history_control_identity')
        if not isinstance(v,torch.Tensor) or not torch.equal(v.cpu(),identity(self.core_hidden_dim,self.control_mode)):
            raise ValueError('History control checkpoint identity mismatch')
        _complete_state(self,state_dict)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

def validate_checkpoint(payload,expected_backbone):
    meta=payload.get('encoder_config',{});states=[payload[k] for k in ('model_state_dict','best_state_dict') if k in payload]
    identified=payload.get('backbone') in ARMS or 'history_control' in meta or any('history_control_identity' in s for s in states)
    if expected_backbone not in ARMS:
        if identified:raise ValueError('History control cannot be relabelled')
        return False
    mode=MODES[expected_backbone];resume=payload.get('resume_identity',{});interface=payload.get('interface_meta',{})
    if (payload.get('backbone')!=expected_backbone or any(meta.get(k)!=v for k,v in metadata(meta.get('d_model'),mode).items())
        or payload.get('evaluation_scope')!='validation_only' or payload.get('held_out_test_evaluated') is not False
        or payload.get('variant')!='count_only_log_regression' or resume.get('backbone')!=expected_backbone
        or resume.get('arguments',{}).get('model_role')!=ROLE or resume.get('interface_meta')!=interface
        or resume.get('checkpoint_monitor')!='validation_raw_quantity_rmse'
        or meta.get('time_head',{}).get('observation_likelihood')!=interface.get('time_head',{}).get('observation_likelihood')):
        raise ValueError('History control split/head/selection mismatch')
    for state in states:
        v=state.get('history_control_identity')
        if not isinstance(v,torch.Tensor) or not torch.equal(v.cpu(),identity(meta['d_model'],mode)):
            raise ValueError('History control state identity mismatch')
    if not states and (len(payload.get('checkpoint_state_sha256',''))!=64 or payload.get('best_epoch',0)<1):
        raise ValueError('Unbound history control summary')
    return True
