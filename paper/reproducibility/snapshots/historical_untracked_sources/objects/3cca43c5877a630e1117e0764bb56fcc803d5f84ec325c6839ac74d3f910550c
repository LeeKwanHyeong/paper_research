"""Deterministic CPU/CUDA tests; no real data or scientific training."""
from __future__ import annotations
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from models.TPPs.CountAwareAdditionalTPP import ARMS, StableBackwardLLH
from models.TPPs.CountAwareFactory import build_count_aware_model, validate_checkpoint_route
from models.TPPs.vendor.s2p2.models import Int_Backward_LLH
from models.TPPs.vendor.attnhp import xfmr
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def build(arm,device='cpu'):
    m,meta=build_count_aware_model(arm,hidden_dim=64,train_log_mean=1.2,train_log_std=.8,max_seq_len=256,
        time_head_mode='heteroscedastic_lognormal_duration',time_scale=1.,time_initial_location=.5,time_initial_scale=.8,
        time_observation_contract={'mode':'positive_integer_round_clamp_v1','top_code':None,'unit':'week'})
    return m.to(device),meta


def check(device='cpu',upstream=None):
    torch.manual_seed(42);torch.set_num_threads(4);results={}
    for first in (True,False):
        kw=dict(P=16,H=64,dropout_rate=0.,relative_time=True,pre_norm=False,post_norm=True,is_first_layer=first)
        original=Int_Backward_LLH(**kw,for_loop=True).to(device);adapted=StableBackwardLLH(**kw).to(device)
        adapted.load_state_dict(original.state_dict())
        alpha=torch.randn(3,9,64,device=device);dt=torch.rand(3,9,device=device)*4
        left=None if first else torch.randn_like(alpha);right=None if first else torch.randn_like(alpha)
        outs=[]
        for m in (original,adapted):
            a=alpha.clone().requires_grad_();y=m(left,right,a,dt)
            sum(v.abs().square().sum() for v in y).backward()
            outs.append((y,a.grad,{k:p.grad for k,p in m.named_parameters()}))
        for a,b in zip(outs[0][0],outs[1][0]):torch.testing.assert_close(a,b,atol=2e-5,rtol=3e-5)
        torch.testing.assert_close(outs[0][1],outs[1][1],atol=1e-4,rtol=5e-4)
        for k,a in outs[0][2].items():
            b=outs[1][2][k]
            if a is None:assert b is None
            else:torch.testing.assert_close(a,b,atol=1e-3,rtol=5e-4,msg=k)
        results['s2p2_original_recurrence_'+str(first)]='passed'
    for arm in ARMS:
        torch.manual_seed(42);model,meta=build(arm,device);model.eval()
        dt=torch.randint(1,5,(3,12),device=device).float();q=torch.rand_like(dt)*8;mask=torch.ones_like(dt,dtype=torch.bool)
        y=model.encode(dt,q,mask);dt2=dt.clone();q2=q.clone();dt2[:,7:]=400;q2[:,7:]=900
        torch.testing.assert_close(y[:,:7],model.encode(dt2,q2,mask)[:,:7],atol=1e-5,rtol=1e-5)
        for side in ('left','right'):
            def pad(x,zero):return torch.cat((zero,x),1) if side=='left' else torch.cat((x,zero),1)
            pm=pad(mask,torch.zeros_like(mask[:,:3]));pd=pad(dt,dt[:,:3]*0);pq=pad(q,q[:,:3]*0)
            py=model.encode(pd,pq,pm)
            torch.testing.assert_close(py[pm].reshape_as(y),y,atol=1e-5,rtol=1e-5)
            assert torch.equal(py[~pm],torch.zeros_like(py[~pm]))
        o=target_outputs(model,dt,mask,q,lambda_log_qty=1.)
        dt2=dt.clone();q2=q.clone();dt2[:,-1]=10000;q2[:,-1]=20000
        p=target_outputs(model,dt2,mask,q2,lambda_log_qty=1.)
        torch.testing.assert_close(o['pred_qty'],p['pred_qty'],atol=0,rtol=0)
        torch.manual_seed(42);baseline,_=build('titantpp_history_mlp',device)
        head_keys=[k for k in model.state_dict() if 'head' in k or k in ('raw_time_slope','raw_time_sigma')];assert head_keys
        for k in head_keys:torch.testing.assert_close(model.state_dict()[k],baseline.state_dict()[k],atol=0,rtol=0)
        model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001)
        def step(m,optim):
            optim.zero_grad(set_to_none=True)
            loss=target_outputs(m,dt,mask,q,lambda_log_qty=1.)['joint_loss'].mean();assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(norm)
            optim.step();assert all(torch.isfinite(x).all() for x in m.parameters())
            return loss.detach()
        for _ in range(3):step(model,opt)
        state=deepcopy(model.state_dict());os=deepcopy(opt.state_dict());rng=torch.get_rng_state()
        crng=torch.cuda.get_rng_state() if device.startswith('cuda') else None
        expected=step(model,opt);next_state=deepcopy(model.state_dict())
        restored,_=build(arm,device);restored.load_state_dict(state,strict=True)
        ropt=torch.optim.AdamW(restored.parameters(),lr=.001);ropt.load_state_dict(os);torch.set_rng_state(rng)
        if crng is not None:torch.cuda.set_rng_state(crng)
        actual=step(restored,ropt);torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        for k,v in next_state.items():torch.testing.assert_close(v,restored.state_dict()[k],atol=0,rtol=0)
        payload={'backbone':arm,'encoder_config':meta,'variant':'count_only_log_regression','evaluation_scope':'validation_only','held_out_test_evaluated':False}
        validate_checkpoint_route(payload,arm)
        try:validate_checkpoint_route(payload,'titantpp_history_mlp')
        except ValueError:pass
        else:raise AssertionError('Foreign checkpoint accepted')
        results[arm]={'causality':'passed','padding':'passed','target_leak':'passed','shared_head_initialization':'passed','optimizer_rng_roundtrip':'passed','route':'passed',
            'parameter_count':sum(x.numel() for x in model.parameters()),'real_scalar_parameter_count':sum(x.numel()*(2 if x.is_complex() else 1) for x in model.parameters())}
    if upstream is not None:
        sys.modules.setdefault('anhp',types.ModuleType('anhp'));sys.modules.setdefault('anhp.model',types.ModuleType('anhp.model'));sys.modules['anhp.model.xfmr']=xfmr
        spec=importlib.util.spec_from_file_location('author_attnhp',Path(upstream)/'attnhp/anhp/model/xfmr_nhp_fast.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        model,_=build(ARMS[1],device);model.eval()
        oracle=mod.XFMRNHPFast(types.SimpleNamespace(num_types=1,pad_index=0,event_num=1),64,2,1,.1,16).to(device).eval()
        oracle.Emb=deepcopy(model.input_projection);oracle.heads[0].load_state_dict(model.layers.state_dict())
        dt=torch.rand(2,10,device=device)*3;q=torch.rand_like(dt)*4;mask=torch.ones_like(dt,dtype=torch.bool)
        blocked=torch.ones(10,10,device=device,dtype=torch.bool).triu(1)[None].expand(2,-1,-1)
        expected=oracle(model.continuous_features(dt,q,mask),dt.cumsum(1),mask,blocked)
        torch.testing.assert_close(model.encode(dt,q,mask),expected,atol=1e-6,rtol=1e-6)
        results['attnhp_original_forward']='passed'
    return {'status':'passed','device':device,'torch':torch.__version__,'checks':results}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--upstream');p.add_argument('--output',required=True);a=p.parse_args()
    r=check(a.device,a.upstream);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
