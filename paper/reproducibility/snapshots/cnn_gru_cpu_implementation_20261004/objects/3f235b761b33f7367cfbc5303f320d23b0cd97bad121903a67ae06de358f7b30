"""Observed-history scalar routing of the unchanged local-detail residual.

The time+quantity gate adds two explicit observed-quantity coefficients to the
same temporal gate. A multiplier in (0,2) is not a bound on the residual norm.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping

import torch
from torch import nn

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanMultiLagDetail import (
    CountAwareTitanMultiLagDetail, LOCAL_DETAIL_BACKBONE, MULTILAG_DETAIL_ROLE,
    multilag_detail_metadata, validate_multilag_detail_checkpoint, _mask, _complete_state,
)

TIME_GATE_BACKBONE = 'titantpp_local_time_gate'
QUANTITY_GATE_BACKBONE = 'titantpp_local_quantity_gate'
LOCAL_GATE_BACKBONES = (TIME_GATE_BACKBONE, QUANTITY_GATE_BACKBONE)
LOCAL_GATE_ROLE = 'observed_time_local_gate_v1'
LOCAL_GATE_CONTRACT = 'hard_lmm_local_gate_v1'


def gate_metadata(hidden_dim, mode, time_scale, quantity_scale):
    if mode not in ('time','quantity'):
        raise ValueError('Local gate mode must be time or quantity')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
           for v in (time_scale,quantity_scale)):
        raise ValueError('Local gate requires positive frozen train scales')
    meta=multilag_detail_metadata(hidden_dim,'local')
    meta.update(candidate_name=TIME_GATE_BACKBONE if mode=='time' else QUANTITY_GATE_BACKBONE,
        backbone_contract_id=LOCAL_GATE_CONTRACT,model_role=LOCAL_GATE_ROLE,
        local_gate_mode=mode,local_gate_formula='2*sigmoid(b+w_t*t+w_dt*delta_t+w_q*q+w_dq*delta_q)',
        local_gate_features=['time_level','time_change']+(['quantity_level','quantity_change'] if mode=='quantity' else []),
        local_gate_time_scale=float(time_scale),local_gate_quantity_scale=float(quantity_scale),
        local_gate_scale_source='frozen_train_only',local_gate_initialization='zero_coefficients_multiplier_one',
        local_gate_parameter_count=3 if mode=='time' else 5,
        additional_parameter_count=meta['additional_parameter_count']+(3 if mode=='time' else 5))
    return meta


def observation_gate_features(dts,quantities,valid,*,memory_write_mask=None,time_scale,quantity_scale):
    """Only current observed row and its previous observed row; padding is skipped.

    Valid-but-withheld rows reset the added path, matching local-detail semantics.
    Earlier B attention context is not erased by that path-specific reset.
    """
    if dts.ndim!=2 or quantities.shape!=dts.shape:
        raise ValueError('Gate inputs must have equal [batch,length] shapes')
    _mask(valid,tuple(dts.shape),'valid')
    if memory_write_mask is not None:_mask(memory_write_mask,tuple(valid.shape),'memory_write_mask')
    valid=valid.to(dts.device)
    observed=valid if memory_write_mask is None else valid & memory_write_mask.to(dts.device)
    t=torch.where(observed,dts,0.);q=torch.where(observed,quantities,0.)
    if not bool(torch.isfinite(t).all() and torch.isfinite(q).all()) or bool((t<0).any() or (q<0).any()):
        raise ValueError('Observed gate inputs must be finite and nonnegative')
    t=torch.log1p(t/time_scale);q=torch.log1p(q)/quantity_scale
    if dts.size(1)==0:return torch.stack((t,t,q,q),dim=-1),observed
    positions=torch.arange(dts.size(1),device=dts.device).expand_as(dts)
    latest=torch.where(observed,positions,-1).cummax(dim=1).values
    predecessor=torch.cat((torch.full_like(latest[:,:1],-1),latest[:,:-1]),dim=1)
    count=observed.long().cumsum(dim=1)
    reset=torch.where(valid & ~observed,count,0).cummax(dim=1).values
    available=observed & (count-reset>1)
    prior=predecessor.clamp_min(0)
    delta_t=torch.where(available,t-t.gather(1,prior),0.)
    delta_q=torch.where(available,q-q.gather(1,prior),0.)
    features=torch.stack((t,delta_t,q,delta_q),dim=-1)
    return torch.where(observed.unsqueeze(-1),features,0.),observed


class ObservedLocalGate(nn.Module):
    def __init__(self,hidden_dim,mode,*,time_scale,quantity_scale):
        super().__init__();self.mode=mode
        meta=gate_metadata(hidden_dim,mode,time_scale,quantity_scale)
        with torch.random.fork_rng(devices=[]):
            self.temporal=nn.Linear(2,1,bias=True)
            nn.init.zeros_(self.temporal.weight);nn.init.zeros_(self.temporal.bias)
            self.quantity=nn.Linear(2,1,bias=False) if mode=='quantity' else None
            if self.quantity is not None:nn.init.zeros_(self.quantity.weight)
        digest=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).digest()
        self.register_buffer('contract_identity',torch.tensor(list(digest),dtype=torch.uint8))
        self.register_buffer('scales',torch.tensor([time_scale,quantity_scale],dtype=torch.float64))
        self._expected_identity=bytes(digest);self._expected_scales=(float(time_scale),float(quantity_scale))

    def validate_identity(self,state,prefix=''):
        identity=state.get(prefix+'contract_identity');scales=state.get(prefix+'scales')
        if (not isinstance(identity,torch.Tensor) or identity.dtype!=torch.uint8
                or not torch.equal(identity.detach().cpu(),torch.tensor(list(self._expected_identity),dtype=torch.uint8))
                or not isinstance(scales,torch.Tensor) or scales.dtype!=torch.float64
                or not torch.equal(scales.detach().cpu(),torch.tensor(self._expected_scales,dtype=torch.float64))):
            raise RuntimeError('Local gate identity/scales mismatch')

    def forward(self,dts,quantities,valid,*,memory_write_mask=None):
        # Scalars are frozen metadata; no data-dependent device synchronization.
        features,observed=observation_gate_features(dts,quantities,valid,memory_write_mask=memory_write_mask,
            time_scale=self._expected_scales[0],quantity_scale=self._expected_scales[1])
        score=self.temporal(features[...,:2])
        if self.quantity is not None:score=score+self.quantity(features[...,2:])
        return torch.where(observed.unsqueeze(-1),2*torch.sigmoid(score),0.)


class CountAwareTitanLocalGate(CountAwareTitanMultiLagDetail):
    def __init__(self,hidden_dim,train_log_mean,max_seq_len,*,local_gate_mode='quantity',**kwargs):
        meta=gate_metadata(hidden_dim,local_gate_mode,kwargs.get('time_scale',3.),train_log_mean)
        super().__init__(hidden_dim,train_log_mean,max_seq_len,multilag_detail_mode='local',**kwargs)
        self.local_gate=ObservedLocalGate(hidden_dim,local_gate_mode,
            time_scale=meta['local_gate_time_scale'],quantity_scale=meta['local_gate_quantity_scale'])

    def load_state_dict(self,state_dict,strict=True,assign=False):
        self.local_gate.validate_identity(state_dict,'local_gate.')
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def _encode_base(self,dts,history_quantities,mask,*,memory_write_mask=None):
        # Match the frozen local-detail encoder exactly; only residual injection differs.
        gate=self.local_gate(dts,history_quantities,mask,memory_write_mask=memory_write_mask)
        valid=mask.to(dts.device);observed=valid if memory_write_mask is None else valid & memory_write_mask.to(dts.device)
        safe_dts=torch.where(observed,dts,0.);safe_quantities=torch.where(observed,history_quantities,0.)
        features=self.continuous_features(safe_dts,safe_quantities,observed)
        encoded=self.encoder.input_proj(features)
        if self.encoder.use_pos_emb:encoded=encoded+self.encoder._get_pos(encoded.size(1),encoded.device,encoded.dtype)
        encoded=torch.where(observed.unsqueeze(-1),encoded,0.)
        encoded=self.encoder.layers[0](encoded,mask=observed)
        correction=self.multilag_detail(encoded,valid,memory_write_mask=observed)
        encoded=encoded+gate*correction
        return self.encoder.layers[1](encoded,mask=observed)


def validate_local_gate_checkpoint(payload,expected_backbone):
    states=[payload[k] for k in ('model_state_dict','best_state_dict') if k in payload]
    meta=payload.get('encoder_config',{})
    if not isinstance(meta,Mapping):raise ValueError('Invalid encoder metadata')
    identified=(payload.get('backbone') in LOCAL_GATE_BACKBONES or meta.get('backbone_contract_id')==LOCAL_GATE_CONTRACT
        or any(isinstance(s,Mapping) and any(str(k).startswith('local_gate.') for k in s) for s in states))
    if expected_backbone not in LOCAL_GATE_BACKBONES:
        if identified:raise ValueError('Local gate checkpoint cannot be relabelled')
        return False
    mode='time' if expected_backbone==TIME_GATE_BACKBONE else 'quantity'
    expected=gate_metadata(meta.get('d_model'),mode,meta.get('local_gate_time_scale'),meta.get('local_gate_quantity_scale'))
    if payload.get('backbone')!=expected_backbone or any(meta.get(k)!=v for k,v in expected.items()):
        raise ValueError('Local gate architecture mismatch')
    interface=payload.get('interface_meta',{});identity=payload.get('resume_identity',{})
    if (not isinstance(interface,Mapping) or not isinstance(identity,Mapping)
            or not isinstance(identity.get('arguments',{}),Mapping) or not isinstance(meta.get('time_head',{}),Mapping)):
        raise ValueError('Invalid gate checkpoint metadata')
    if (identity.get('backbone')!=expected_backbone or identity.get('arguments',{}).get('model_role')!=LOCAL_GATE_ROLE
            or interface.get('train_target_mean')!=expected['local_gate_quantity_scale']
            or meta.get('time_head',{}).get('time_scale')!=expected['local_gate_time_scale']):
        raise ValueError('Local gate train-scale/resume identity mismatch')
    # Validate the unchanged local-detail body and common head using its existing
    # strict validator on a validation-only view; never mutate the stored artifact.
    base=deepcopy(payload);base['backbone']=LOCAL_DETAIL_BACKBONE
    base_meta=multilag_detail_metadata(meta['d_model'],'local');base_meta.update(max_len=meta.get('max_len'),time_head=meta['time_head'])
    base['encoder_config']=base_meta
    base['resume_identity']['backbone']=LOCAL_DETAIL_BACKBONE
    base['resume_identity']['arguments']['model_role']=MULTILAG_DETAIL_ROLE
    gate=ObservedLocalGate(meta['d_model'],mode,time_scale=expected['local_gate_time_scale'],quantity_scale=expected['local_gate_quantity_scale'])
    for name in ('model_state_dict','best_state_dict'):
        if name not in base:continue
        state=base[name];gate_state={k.removeprefix('local_gate.'):v for k,v in state.items() if k.startswith('local_gate.')}
        gate.validate_identity(gate_state);_complete_state(gate,gate_state)
        base[name]={k:v for k,v in state.items() if not k.startswith('local_gate.')}
    return validate_multilag_detail_checkpoint(base,LOCAL_DETAIL_BACKBONE)
