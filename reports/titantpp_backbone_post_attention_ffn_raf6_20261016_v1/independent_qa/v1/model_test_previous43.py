"""Synthetic-only FFN contract tests; never load scientific data/checkpoints."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import random
import numpy as np
import polars as pl
import pytest
import torch
from torch.nn import functional as F

from models.TPPs import CountAwareTitanPostAttentionFFN as candidate
from models.TPPs.CountAwareTitanCNNGRU import observed_context
from paper.scripts import run_titantpp_post_attention_ffn as route
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def forbid_real_data():
    def no(*args, **kwargs):
        raise AssertionError("Scientific data is forbidden in model contract tests")
    with patch.object(pl, "read_parquet", no), patch.object(pl, "scan_parquet", no), patch.object(torch, "load", no):
        yield


def build(arm=candidate.CURRENT, seed=42, length=84):
    torch.manual_seed(seed)
    data = route.fabricated_data(); data["loader"]["max_seq_len"] = length
    return route.build_model(data, arm)


def batch():
    dt = torch.tensor([[0., 1., 3., 2., 4., 2., 3., 5.], [0., 0., 0., 2., 5., 1., 4., 2.]])
    qty = torch.tensor([[0., 0., 5., 2., 7., 4., 3., 6.], [0., 0., 0., 2., 4., 8., 1., 5.]])
    mask = torch.tensor([[False] + [True] * 7, [False] * 3 + [True] * 5])
    return dt, qty, mask


def activate(model):
    with torch.no_grad():
        for name in ("q", "k", "v"):
            getattr(model.encoder.layers[0].attn, "causal_" + name + "_kernel").normal_(0, .1)
        model.encoder.layers[1].ffn_observed.c.normal_(0, .2)
        model.quantity_head.weight.normal_(0, .1)
        model.time_scale_weight.weight.normal_(0, .01)
    return model.eval()


def payload(arm=candidate.CURRENT, length=84):
    model, meta = build(arm, length=length)
    interface = {"time_head": meta["time_head"]}
    return {"backbone": arm, "encoder_config": meta, "model_state_dict": model.state_dict(),
        "evaluation_scope": "validation_only", "held_out_test_evaluated": False,
        "variant": route.VARIANT, "interface_meta": interface,
        "resume_identity": {"backbone": arm, "interface_meta": interface,
            "checkpoint_monitor": "validation_raw_quantity_rmse", "arguments": {"model_role": route.role_for_arm(arm)}}}


@pytest.mark.parametrize("arm", candidate.ARMS)
@pytest.mark.parametrize("seed", (42, 52, 62))
@pytest.mark.parametrize("length", (32, 84, 256))
def test_exact_initial_tensors_rng_counts_and_zero_output(arm, seed, length):
    data = route.fabricated_data(); data["loader"]["max_seq_len"] = length
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    parent, _ = route.original.build_model(data, candidate.LEGACY_CNN)
    rng, py, np_rng = torch.get_rng_state().clone(), random.getstate(), np.random.get_state()
    random.seed(seed); np.random.seed(seed)
    model, meta = build(arm, seed, length)
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == py
    nr = np.random.get_state()
    assert nr[0] == np_rng[0] and np.array_equal(nr[1], np_rng[1]) and nr[2:] == np_rng[2:]
    state = model.state_dict()
    assert set(state) == set(parent.state_dict()) | {candidate.IDENTITY_BUFFER, *candidate.EXTRA_KEYS}
    assert all(torch.equal(v, state[k]) for k,v in parent.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) == meta["registered_parameter_count"] == 116319 + 64*(length-256)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == meta["active_trainable_parameter_count"] == 91587 + 64*(length-256)
    assert sum(p.numel() for p in model.multilag_detail.parameters()) == 24732
    assert not any(p.requires_grad for p in model.multilag_detail.parameters())
    assert sum(p.numel() for n,p in model.encoder.layers[0].attn.named_parameters() if n.endswith("_kernel")) == 576
    dt, qty, mask = batch()
    for mode in (False, True):
        parent.train(mode); model.train(mode)
        torch.manual_seed(204); a = core.target_outputs(parent, dt, mask, qty, lambda_log_qty=1.); r = torch.get_rng_state()
        torch.manual_seed(204); b = core.target_outputs(model, dt, mask, qty, lambda_log_qty=1.)
        assert torch.equal(r, torch.get_rng_state())
        assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("arm", candidate.ARMS)
@pytest.mark.parametrize("length", (0,1,2,3,7))
def test_scalar_oracle_and_fixed_divisor_mask(arm,length):
    module = candidate.ObservedFFNResidual(arm=arm).double()
    with torch.no_grad():
        module.a.copy_(torch.linspace(.2,1.9,384).reshape(3,128))
        module.b.copy_(torch.linspace(-1.7,1.2,384).reshape(3,128))
        module.c.copy_(torch.linspace(-.7,.8,384).reshape(3,128))
    x = torch.linspace(-2.,2.,max(1,2*length*128))[:2*length*128].reshape(2,length,128).double()
    valid=torch.ones(2,length,dtype=torch.bool)
    _, src, available = observed_context(valid)
    actual=module(x,src,available)
    expected=torch.zeros_like(x)
    for row in range(2):
        for t in range(length):
            for lag in range(min(t+1,3)):
                source=t if arm==candidate.CURRENT else t-lag
                expected[row,t] += module.c[lag]*F.gelu(module.a[lag]*x[row,source]+module.b[lag])/3
    torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert torch.isfinite(actual).all()
    if length:
        assert torch.allclose(actual[:,0], module.c[0]*F.gelu(module.a[0]*x[:,0]+module.b[0])/3)


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_padding_withheld_and_unobserved_nonfinite_gradient_isolation(arm):
    module=candidate.ObservedFFNResidual(arm=arm)
    with torch.no_grad(): module.c.fill_(.7)
    values=torch.randn(2,8,128)
    valid=torch.tensor([[True,True,False,True,True,False,True,True]]*2)
    _,src,av=observed_context(valid)
    dirty=values.clone();dirty[~valid]=float("nan");dirty.requires_grad_()
    actual=module(dirty,src,av)
    keep=[0,1,3,4,6,7];_,cs,ca=observed_context(valid[:,keep])
    assert torch.equal(actual[:,keep],module(values[:,keep],cs,ca))
    assert torch.equal(actual[~valid],torch.zeros_like(actual[~valid]))
    actual.square().sum().backward()
    assert torch.isfinite(dirty.grad).all() and not dirty.grad[~valid].any()
    write=torch.ones_like(valid);write[:,3]=False
    observed,rs,ra=observed_context(torch.ones_like(valid),memory_write_mask=write)
    reset=module(values,rs,ra)
    _,fs,fa=observed_context(torch.ones_like(valid[:,4:]))
    assert torch.equal(reset[:,4:],module(values[:,4:],fs,fa))
    assert not reset[:,3].any()
    # This isolation claim concerns the new gather, not the parent attention.
    changed=values.clone();changed[:,:4]+=100.
    assert torch.equal(reset[:,4:],module(changed,rs,ra)[:,4:])


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_nonfinite_and_malformed_context_rejected(arm):
    module=candidate.ObservedFFNResidual(arm=arm)
    x=torch.ones(1,4,128);_,s,a=observed_context(torch.ones(1,4,dtype=torch.bool))
    for name in ('a','b','c'):
        p=getattr(module,name);before=p.detach().clone()
        with torch.no_grad():p[0,0]=float('nan')
        with pytest.raises(ValueError):module(x,s,a)
        with torch.no_grad():p.copy_(before)
    bad=x.clone();bad[0,0,0]=float('inf')
    with pytest.raises(ValueError):module(bad,s,a)
    for bads in (s.float(),s[:,:,:2],s+4,s-1):
        with pytest.raises(ValueError):module(x,bads,a)
    future=s.clone();future[0,1,1]=2
    with pytest.raises(ValueError):module(x,future,a)
    with pytest.raises(ValueError):module(x,s,a.float())
    with torch.no_grad():module.a.fill_(torch.finfo(torch.float32).max)
    with pytest.raises(ValueError):module(x*4,s,a)


def test_current_control_is_nonaffine_with_independent_tap_effects():
    m=candidate.ObservedFFNResidual(arm=candidate.CURRENT).double()
    x=torch.linspace(-3,3,31)[:,None,None].expand(-1,3,128).clone().double()
    _,s,a=observed_context(torch.ones(31,3,dtype=torch.bool))
    with torch.no_grad():m.c.fill_(1.)
    y=m(x,s,a)[:,2,0]
    assert (y[2:]-2*y[1:-1]+y[:-2]).abs().max()>.001
    derivatives=[]
    for tap in range(3):
        with torch.no_grad():m.c.zero_();m.c[tap].fill_(1.)
        derivatives.append(m(x,s,a)[:,2,0])
    assert int(torch.linalg.matrix_rank(torch.stack(derivatives,1)))==3
    assert all(not torch.allclose(derivatives[i],derivatives[j]) for i in range(3) for j in range(i))


@pytest.mark.parametrize("arm", candidate.ARMS)
def test_two_steps_generic_task_gradient_and_adamw_decay(arm):
    m=candidate.ObservedFFNResidual(arm=arm).double()
    x=torch.linspace(-1.7,2.,3*7*128).reshape(3,7,128).double()
    _,s,a=observed_context(torch.ones(3,7,dtype=torch.bool))
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01)
    original_a=m.a.detach().clone();original_b=m.b.detach().clone()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=(m(x,s,a)*torch.linspace(.5,1.5,x.numel()).reshape_as(x)).sum()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        if step==0:
            assert not m.a.grad.any() and not m.b.grad.any() and torch.all(m.c.grad!=0)
        else:
            assert all(torch.all(p.grad!=0) for p in m.parameters())
        optimizer.step()
        if step==0:
            torch.testing.assert_close(m.a,original_a*(1-.001*.01),rtol=0,atol=0)
            torch.testing.assert_close(m.b,original_b*(1-.001*.01),rtol=0,atol=0)


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_full_model_future_target_boundary_batch_and_state(arm):
    model,_=build(arm);activate(model)
    dt,q,mask=batch()
    initial={k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        baseline=model._encode_base(dt,q,mask)
        cd=dt.clone();cq=q.clone();cd[:,6:]+=300;cq[:,6:]+=300
        assert torch.equal(baseline[:,:6],model._encode_base(cd,cq,mask)[:,:6])
        dirtyq=q.clone();dirtyd=dt.clone();dirtyq[~mask]=float('nan');dirtyd[~mask]=float('inf')
        assert torch.equal(baseline,model._encode_base(dirtyd,dirtyq,mask))
        bad=q.clone();bad[0,2]=float('nan')
        with pytest.raises(ValueError):model._encode_base(dt,bad,mask)
        write=mask.clone();write[:,4]=False
        clean=model._encode_base(dt,q,mask,memory_write_mask=write)
        cq=q.clone();cd=dt.clone();cq[:,4]=float('nan');cd[:,4]=float('inf')
        assert torch.equal(clean,model._encode_base(cd,cq,mask,memory_write_mask=write))
        outputs=core.target_outputs(model,dt,mask,q,lambda_log_qty=1.)
        cq=q.clone();cq[:,-1]=float('nan')
        assert torch.equal(outputs['pred_qty'],core.target_outputs(model,dt,mask,cq,lambda_log_qty=1.)['pred_qty'])
        for row in range(2):
            one=core.target_outputs(model,dt[row:row+1],mask[row:row+1],q[row:row+1],lambda_log_qty=1.)
            for key in outputs:torch.testing.assert_close(outputs[key][row:row+1],one[key],rtol=1e-5,atol=1e-6)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in initial.items())
    assert model.encoder.layers[1].ffn_observed.__dict__.keys() >= {'arm'}
    assert not list(model.encoder.layers[1].ffn_observed.buffers())


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_checkpoint_strict_identity_and_metadata(arm):
    p=payload(arm);assert candidate.validate_checkpoint(p,arm)
    model,_=build(arm);model.load_state_dict(p['model_state_dict'],strict=False)
    other=next(a for a in candidate.ARMS if a!=arm)
    with pytest.raises(ValueError):candidate.validate_checkpoint(p,other)
    for alias in (candidate.CNN,candidate.LEGACY_CNN,'titantpp_unknown'):
        with pytest.raises(ValueError):candidate.validate_checkpoint(p,alias)
    for key,value in [('parent_source_closure','0'*64),('post_attention_ffn_schema','wrong'),('max_len',83),('backbone_contract_id','wrong')]:
        bad=deepcopy(p);bad['encoder_config'][key]=value
        with pytest.raises((ValueError,RuntimeError)):candidate.validate_checkpoint(bad,arm)
    for key in candidate.EXTRA_KEYS:
        bad=deepcopy(p);bad['model_state_dict'][key]=torch.zeros(128,3)
        with pytest.raises((ValueError,RuntimeError)):candidate.validate_checkpoint(bad,arm)
        with pytest.raises((ValueError,RuntimeError)):model.load_state_dict(bad['model_state_dict'],strict=False)
    bad=deepcopy(p['model_state_dict']);bad['unexpected']=torch.zeros(1)
    with pytest.raises((ValueError,RuntimeError)):model.load_state_dict(bad,strict=False)
    bad=deepcopy(p['model_state_dict']);bad.pop(candidate.EXTRA_KEYS[0])
    with pytest.raises((ValueError,RuntimeError)):model.load_state_dict(bad,strict=False)
    bad=deepcopy(p);bad['encoder_config']['post_attention_ffn_arm']=other
    with pytest.raises((ValueError,RuntimeError)):candidate.validate_checkpoint(bad,arm)
    bad=deepcopy(p);bad['held_out_test_evaluated']=True
    with pytest.raises((ValueError,RuntimeError)):candidate.validate_checkpoint(bad,arm)


def test_no_baseline_training_build_metadata_or_initialization_route(tmp_path):
    data=route.fabricated_data()
    for arm in (candidate.CNN,candidate.LEGACY_CNN):
        for operation in (lambda:route.build_model(data,arm),lambda:route.metadata(arm=arm),
                          lambda:route.initial_states(data,42,arms=(arm,)),lambda:route.training_args({},data,tmp_path,arm)):
            with pytest.raises((ValueError,RuntimeError)):operation()
    assert set(route.SUPPORTED_ARMS)==set(candidate.ARMS)


@pytest.mark.parametrize('seed',(42,52,62))
def test_native_qualification_callable_is_synthetic_and_full_gradient(seed):
    result=route.synthetic_check('cpu',seed=seed)
    assert result['status']=='passed' and result['scientific_optimizer_updates']==0
    assert result['real_data_loaded'] is False and result['held_out_test_evaluated'] is False
    assert all(v==1152 for v in result['active_ffn_gradient_parameter_count'].values())
