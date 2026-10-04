"""Four prespecified ablations of the frozen lag-one level/change TitanTPP.

Persistent encoder memory is retained by every arm. The no-static arm removes
only the final tied prototype retrieval. All new residuals start at zero.
"""
import hashlib
import json
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F
from models.TPPs.CountAwareTitanMultiLagDetail import (
    CountAwareTitanMultiLagDetail, MultiLagDetail, multilag_detail_metadata,
    lag_source_indices, _mask, _complete_state,
)

MODES = {"titantpp_history_mlp": "mlp", "titantpp_level_only": "level",
         "titantpp_change_only": "change", "titantpp_no_static_lmm": "no_static"}
ARMS = tuple(MODES)
ROLE = "observed_time_core_ablation_v1"


def metadata(hidden_dim, mode):
    if mode not in MODES.values(): raise ValueError("Unknown core ablation")
    m = multilag_detail_metadata(hidden_dim, "local")
    m.update(candidate_name=next(k for k,v in MODES.items() if v == mode),
             backbone_contract_id="titantpp_core_ablation_v1", model_role=ROLE,
             core_ablation_mode=mode, persistent_mem_size=16,
             additional_parameter_count=8*(3 if mode in ('mlp','no_static') else 2)*hidden_dim*4,
             removed_static_parameters=64*hidden_dim if mode=='no_static' else 0,
             static_retrieval_enabled=mode!='no_static')
    m['multilag_detail_projections']={'mlp':'concat_current_previous_bias_free_mlp_width4',
        'level':'bias_free_level_output','change':'bias_free_change_output',
        'no_static':m['multilag_detail_projections']}[mode]
    if mode=='no_static':m.update(lmm_mem_size=0,lmm_topk=0,memory_mode='static_retrieval_removed_persistent_retained')
    return m


def identity(hidden_dim, mode):
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(hidden_dim,mode),sort_keys=True).encode()).digest()),dtype=torch.uint8)


class HistoryCorrection(nn.Module):
    def __init__(self, hidden_dim, mode):
        super().__init__()
        self.hidden_dim,self.mode=hidden_dim,mode
        if mode not in ('mlp','level','change'):raise ValueError(mode)
        with torch.random.fork_rng(devices=[]):
            if mode=='mlp':
                self.input_projections=nn.ModuleList(nn.Linear(2*hidden_dim,4,bias=False) for _ in range(8))
                self.output_projections=nn.ModuleList(nn.Linear(4,hidden_dim,bias=False) for _ in range(8))
                for p in self.output_projections:nn.init.zeros_(p.weight)
            else:
                # Retain the exact corresponding Full initialization, with no dummy parameters.
                full=MultiLagDetail(hidden_dim,mode='local')
                self.input_projections=full.level_projections if mode=='level' else full.detail_projections
                self.output_projections=full.output_projections

    def forward(self, hidden, valid, *, memory_write_mask=None):
        _mask(valid,tuple(hidden.shape[:2]),'valid')
        observed=valid if memory_write_mask is None else valid & memory_write_mask
        safe=torch.where(observed.unsqueeze(-1),hidden,0.)
        if not torch.isfinite(safe).all():raise ValueError('Nonfinite observed history')
        source,available=lag_source_indices(valid,memory_write_mask=memory_write_mask,mode='local')
        weight=torch.cat([p.weight for p in self.input_projections])
        if self.mode=='mlp':
            current=F.linear(safe,weight[:,:self.hidden_dim]).unflatten(-1,(8,4))
            previous=F.linear(safe,weight[:,self.hidden_dim:]).unflatten(-1,(8,4))
            feature=current+previous.gather(1,source.unsqueeze(-1).expand(-1,-1,-1,4))
        else:
            current=F.linear(safe,weight).unflatten(-1,(8,4))
            previous=current.gather(1,source.unsqueeze(-1).expand(-1,-1,-1,4))
            feature=(current+previous)*.5 if self.mode=='level' else current-previous
        feature=F.gelu(torch.where(available.unsqueeze(-1),feature,0.),approximate='none')
        return F.linear(feature.flatten(-2),torch.cat([p.weight for p in self.output_projections],dim=1))/8


class CountAwareTitanCoreAblation(CountAwareTitanMultiLagDetail):
    def __init__(self,hidden_dim,train_log_mean,max_seq_len,*,core_mode,**kwargs):
        super().__init__(hidden_dim,train_log_mean,max_seq_len,multilag_detail_mode='local',**kwargs)
        self.core_mode,self.core_hidden_dim=core_mode,hidden_dim
        if core_mode=='no_static':self.lmm=nn.Identity()
        else:self.multilag_detail=HistoryCorrection(hidden_dim,core_mode)
        self.register_buffer('core_ablation_identity',identity(hidden_dim,core_mode))

    def load_state_dict(self,state_dict,strict=True,assign=False):
        value=state_dict.get('core_ablation_identity')
        if not isinstance(value,torch.Tensor) or not torch.equal(value.cpu(),identity(self.core_hidden_dim,self.core_mode)):
            raise ValueError('Core ablation checkpoint identity mismatch')
        if self.core_mode=='no_static':self.multilag_detail.validate_identity(state_dict,'multilag_detail.')
        _complete_state(self,state_dict)
        return nn.Module.load_state_dict(self,state_dict,strict=True,assign=assign)


def validate_checkpoint(payload,expected_backbone):
    meta=payload.get('encoder_config',{})
    states=[payload[k] for k in ('model_state_dict','best_state_dict') if k in payload]
    identified=(payload.get('backbone') in ARMS or 'core_ablation_mode' in meta
                or any('core_ablation_identity' in s for s in states))
    if expected_backbone not in ARMS:
        if identified:raise ValueError('Core ablation cannot be relabelled')
        return False
    if payload.get('backbone')!=expected_backbone:raise ValueError('Core ablation label mismatch')
    mode=MODES[expected_backbone]; expected=metadata(meta.get('d_model'),mode)
    if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Core ablation metadata mismatch')
    head=meta.get('time_head',{}); interface=payload.get('interface_meta',{}); resume=payload.get('resume_identity',{})
    if (payload.get('evaluation_scope')!='validation_only' or payload.get('held_out_test_evaluated') is not False
        or payload.get('variant')!='count_only_log_regression'
        or resume.get('backbone')!=expected_backbone or resume.get('interface_meta')!=interface
        or resume.get('checkpoint_monitor')!='validation_raw_quantity_rmse'
        or resume.get('arguments',{}).get('model_role')!=ROLE
        or head.get('mode')!='heteroscedastic_lognormal_duration'
        or head.get('observation_likelihood')!=interface.get('time_head',{}).get('observation_likelihood')):
        raise ValueError('Core ablation split, head, selector or resume identity mismatch')
    for s in states:
        value=s.get('core_ablation_identity')
        if not isinstance(value,torch.Tensor) or not torch.equal(value.cpu(),identity(meta['d_model'],mode)):
            raise ValueError('Core ablation state identity mismatch')
        if (any(k.startswith('lmm.') for k in s)) != (mode!='no_static'):
            raise ValueError('Static retrieval state mismatch')
    if not states and (len(payload.get('checkpoint_state_sha256',''))!=64 or payload.get('best_epoch',0)<1):
        raise ValueError('Unbound core ablation summary')
    return True
