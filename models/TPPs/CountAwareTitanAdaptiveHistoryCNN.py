"""Opt-in local-history CNN hypotheses; all shared science remains inherited.

Both candidates add 243 active parameters. They differ in how the MLP output
is used, not its input, size, causal scope, or initialization. Equal parameter
count is not equivalence of the function classes or optimization dynamics.
"""
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

CNN = components.CNN
ADAPTIVE = "titantpp_cnn_history_adaptive_v1"
CAPACITY = "titantpp_cnn_history_capacity_v1"
ARMS = (ADAPTIVE, CAPACITY)
CONTRACT_ID = "titantpp_cnn_history_adaptation_20261015_v1"
IDENTITY_BUFFER = "adaptive_history_cnn_identity"
EXTRA_PREFIX = "encoder.layers.0.attn.history_networks."
ADDED_PARAMETERS = 243


def role_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unsupported history-CNN arm")
    return "observed_time_"+arm.removeprefix("titantpp_")


def metadata(hidden_dim=64, *, arm):
    role_for_arm(arm)
    result=components.metadata(hidden_dim,arm=CNN)
    result.update(candidate_name=arm,model_role=role_for_arm(arm),backbone_contract_id=CONTRACT_ID,
        baseline_backbone=CNN,history_network_shape=[3,16,1],history_network_sharing="per_qkv_shared_across_channels",
        history_network_features="projected_current,current_minus_lag1,lag1_minus_lag2; unavailable_deltas_zero",
        history_network_activation="tanh_hidden_and_output",history_network_scale=.5,
        history_operation="relative_tap_reweighting" if arm==ADAPTIVE else "additive_local_nonlinearity",
        history_minimum_observations=2,history_extra_parameters=ADDED_PARAMETERS,
        history_initialization="fork_rng_default_first_layer_zero_final_layer",
        history_tap_bounds=[.5,1.5] if arm==ADAPTIVE else None,
        history_available_tap_mean=1. if arm==ADAPTIVE else None,
        history_adaptive_norm_bound=False,
        active_trainable_parameter_count=components.ACTIVE_PARAMETERS[CNN]+ADDED_PARAMETERS,
        registered_parameter_count=components.REGISTERED_PARAMETERS+ADDED_PARAMETERS,
        additional_parameter_count=components.CNN_PARAMETERS+ADDED_PARAMETERS,
        active_correction_parameter_count=components.CNN_PARAMETERS+ADDED_PARAMETERS,
        active_budget_limitations="Matched added parameters and receptive field; distinct function classes and optimization",
        common_encoder_initialization="exact_original_CNN_tensors_and_rng")
    return result


def identity(*, arm):
    return torch.tensor(list(hashlib.sha256(json.dumps(metadata(arm=arm),sort_keys=True).encode()).digest()),dtype=torch.uint8)


def history_features(values, sources, available):
    # Parent validation also guarantees masked nonfinite values cannot enter.
    causal_depthwise_residual(values,values.new_zeros((3,values.size(-1))),sources,available)
    safe=torch.where(available[...,0,None],values,0.)
    taps=torch.stack([torch.where(available[...,lag,None],
        safe.gather(1,sources[...,lag,None].expand_as(values)),0.) for lag in range(3)],dim=-2)
    now,prev,older=taps.unbind(-2)
    features=torch.stack((now,
        torch.where(available[...,1,None],now-prev,0.),
        torch.where(available[...,2,None],prev-older,0.)),dim=-1)
    return taps,features


def tap_multipliers(raw, available):
    """Mean one over available taps; bounded .5..1.5, with no learned scale."""
    u=torch.where(available[...,1,None],torch.tanh(raw),0.)
    m=available[...,1:].sum(-1).clamp_min(1).to(raw.dtype).unsqueeze(-1)
    return torch.stack((1.+.5*u,1.-.5*u/m,1.-.5*u/m),dim=-2)


class HistoryCNNFilter(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.network=nn.Sequential(nn.Linear(3,16),nn.Tanh(),nn.Linear(16,1))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)

    def forward(self,values,kernel,sources,available,*,arm):
        if arm not in ARMS:
            raise ValueError("Unknown local history operation")
        taps,features=history_features(values,sources,available)
        raw=self.network(features).squeeze(-1)
        if arm==ADAPTIVE:
            weights=tap_multipliers(raw,available)
            # Preserve the original accumulation order, including at u=0.
            result=torch.zeros_like(values)
            for lag in range(3):
                result=result+taps[...,lag,:]*kernel[lag]*weights[...,lag,:]
            return result
        result=causal_depthwise_residual(values,kernel,sources,available)
        return result+torch.where(available[...,1,None],.5*torch.tanh(raw),0.)


class AdaptiveHistoryAttention(ObservedCausalQKVMemoryAttention):
    def __init__(self,base,*,arm):
        role_for_arm(arm)
        nn.Module.__init__(self)
        for name in ("d_model","n_heads","use_causal","head_dim","scale","qkv","out_proj","drop",
                     "contextual_mem_size","persistent_mem_size","persistent_mem","_ctx_buf","_ctx_mem",
                     "causal_q_kernel","causal_k_kernel","causal_v_kernel"):
            setattr(self,name,getattr(base,name))
        self.arm=arm
        self.history_networks=nn.ModuleDict({name:HistoryCNNFilter() for name in ("q","k","v")})
        self.train(base.training)

    def forward(self,x,mask=None,*,observed_sources,observed_available,
                event_attention_bias=None,return_event_attention=False):
        batch,length,_=x.shape
        if mask is None:
            raise ValueError("Observed CNN attention requires an explicit mask")
        _mask(mask,(batch,length),"attention mask")
        if not torch.equal(mask,observed_available[...,0]):
            raise ValueError("CNN context does not match observed attention mask")
        q,k,v=torch.chunk(self.qkv(x),3,dim=-1)
        q=q+self.history_networks["q"](q,self.causal_q_kernel,observed_sources,observed_available,arm=self.arm)
        k=k+self.history_networks["k"](k,self.causal_k_kernel,observed_sources,observed_available,arm=self.arm)
        v=v+self.history_networks["v"](v,self.causal_v_kernel,observed_sources,observed_available,arm=self.arm)
        # Exact inherited attention arithmetic below; memory tokens bypass CNN.
        memory_parts=[]
        if self._ctx_mem is not None and self._ctx_mem.numel()>0:
            memory_parts.append(self._ctx_mem.to(device=x.device,dtype=x.dtype))
        if self.persistent_mem is not None:
            memory_parts.append(self.persistent_mem.to(device=x.device,dtype=x.dtype).expand(batch,-1,-1))
        memory_length=0
        if memory_parts:
            memory=torch.cat(memory_parts,dim=1)
            memory_length=memory.size(1)
            k,v=torch.cat([memory,k],dim=1),torch.cat([memory,v],dim=1)
        scores=torch.matmul(self._split_heads(q),self._split_heads(k).transpose(-2,-1))*self.scale
        if event_attention_bias is not None:
            if event_attention_bias.shape!=(batch,self.n_heads,length,length):
                raise ValueError("Event attention bias shape mismatch")
            scores=scores+F.pad(event_attention_bias.to(dtype=scores.dtype,device=x.device),(memory_length,0))
        full_mask=None
        if self.use_causal:
            full_mask=torch.cat((torch.ones(length,memory_length,device=x.device,dtype=torch.bool),
                torch.tril(torch.ones(length,length,device=x.device,dtype=torch.bool))),1)
        valid_keys=torch.cat((torch.ones(batch,memory_length,device=x.device,dtype=torch.bool),mask),1)
        key_mask=valid_keys[:,None,None,:]
        full_mask=key_mask if full_mask is None else full_mask[None,None,:,:]&key_mask
        if memory_length==0:
            invalid_queries=~mask
            if invalid_queries.any():
                full_mask=full_mask.expand(batch,1,length,length).clone()
                batch_ids,positions=invalid_queries.nonzero(as_tuple=True)
                full_mask[batch_ids,0,positions,positions]=True
        scores=scores.masked_fill(~full_mask,float("-inf"))
        attention=F.softmax(scores,dim=-1)
        event_attention=attention[...,memory_length:].mean(1) if return_event_attention else None
        attention=self.drop(attention)
        output=self.out_proj(self._merge_heads(torch.matmul(attention,self._split_heads(v))))
        output=output*mask.to(dtype=output.dtype).unsqueeze(-1)
        return (output,event_attention) if return_event_attention else output


class CountAwareTitanAdaptiveHistoryCNN(components.CountAwareTitanCNNGRUComponents):
    def __init__(self,hidden_dim,train_log_mean,max_seq_len,*,history_arm,**kwargs):
        metadata(hidden_dim,arm=history_arm)
        super().__init__(hidden_dim,train_log_mean,max_seq_len,component_arm=CNN,**kwargs)
        self.history_arm=history_arm
        self.encoder.layers[0].attn=AdaptiveHistoryAttention(self.encoder.layers[0].attn,arm=history_arm)
        self.register_buffer(IDENTITY_BUFFER,identity(arm=history_arm))

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if not _identity_matches(state_dict.get(IDENTITY_BUFFER),identity(arm=self.history_arm)):
            raise ValueError("History-CNN identity mismatch")
        if not _identity_matches(state_dict.get("cnn_gru_component_identity"),components.identity(arm=CNN)):
            raise ValueError("CNN parent identity mismatch")
        _complete_state(self,state_dict)
        # Parent verifies its two identities after the full candidate-state check.
        return super().load_state_dict(state_dict,strict=True,assign=assign)


def validate_checkpoint(payload,expected_backbone):
    meta=payload.get("encoder_config",{})
    states=[payload[k] for k in ("model_state_dict","best_state_dict") if k in payload]
    identified=(payload.get("backbone") in ARMS or isinstance(meta,Mapping) and meta.get("backbone_contract_id")==CONTRACT_ID
                or any(isinstance(s,Mapping) and IDENTITY_BUFFER in s for s in states))
    if expected_backbone not in ARMS:
        if identified: raise ValueError("History-CNN checkpoint cannot be relabelled")
        return False
    if not isinstance(meta,Mapping) or not states:
        raise ValueError("History-CNN metadata and state are required")
    required=metadata(meta.get("d_model"),arm=expected_backbone)
    resume=payload.get("resume_identity",{})
    if (payload.get("backbone")!=expected_backbone or any(meta.get(k)!=v for k,v in required.items())
            or not isinstance(resume,Mapping) or resume.get("backbone")!=expected_backbone
            or resume.get("arguments",{}).get("model_role")!=role_for_arm(expected_backbone)):
        raise ValueError("History-CNN metadata or route mismatch")
    shapes={f"{EXTRA_PREFIX}{q}.network.{k}":shape for q in ("q","k","v")
            for k,shape in (("0.weight",(16,3)),("0.bias",(16,)),("2.weight",(1,16)),("2.bias",(1,)))}
    for state in states:
        if not isinstance(state,Mapping) or not _identity_matches(state.get(IDENTITY_BUFFER),identity(arm=expected_backbone)):
            raise ValueError("History-CNN state identity mismatch")
        for k,shape in shapes.items():
            v=state.get(k)
            if not isinstance(v,torch.Tensor) or tuple(v.shape)!=shape or not torch.is_floating_point(v) or not bool(torch.isfinite(v).all()):
                raise ValueError("History-CNN network state invalid: "+k)
        if {k for k in state if k.startswith(EXTRA_PREFIX)}!=set(shapes):
            raise ValueError("Unexpected history network state")
    parent={**payload,"backbone":CNN,"encoder_config":{**meta,**components.metadata(arm=CNN)},
            "resume_identity":{**resume,"backbone":CNN,"arguments":{**resume.get("arguments",{}),"model_role":components.role_for_arm(CNN)}}}
    for k in ("model_state_dict","best_state_dict"):
        if k in payload:
            parent[k]={n:v for n,v in payload[k].items() if n!=IDENTITY_BUFFER and not n.startswith(EXTRA_PREFIX)}
    return components.validate_checkpoint(parent,CNN)
