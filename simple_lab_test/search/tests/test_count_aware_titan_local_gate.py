"""Synthetic-only checks for observation-conditioned local residual injection."""
from copy import deepcopy
import io

import pytest
import torch
from models.TPPs.CountAwareFactory import build_count_aware_model,validate_checkpoint_route
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanMultiLagDetail import CountAwareTitanMultiLagDetail
from models.TPPs.CountAwareTitanLocalGate import (CountAwareTitanLocalGate,ObservedLocalGate,
    observation_gate_features,LOCAL_GATE_BACKBONES,LOCAL_GATE_ROLE)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from simple_lab_test.search.tests.test_count_aware_titan_multilag_detail import kwargs,inputs,activate

@pytest.fixture(autouse=True)
def one_thread():
    n=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(n)

@pytest.mark.parametrize('mode',['time','quantity'])
@pytest.mark.parametrize('training',[False,True])
def test_initial_B_and_local_outputs_common_preclip_gradient_and_rng(mode,training):
    torch.manual_seed(31);base=CountAwareTitanTPP(**kwargs());rng=torch.get_rng_state().clone()
    torch.manual_seed(31);local=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode='local')
    torch.manual_seed(31);model=CountAwareTitanLocalGate(**kwargs(),local_gate_mode=mode)
    assert torch.equal(rng,torch.get_rng_state())
    dts,qty,mask=inputs();start=torch.get_rng_state().clone();results=[]
    for m in (base,local,model):
        m.train(training);torch.set_rng_state(start)
        r=target_outputs(m,dts,mask,qty,lambda_log_qty=1.)
        r['joint_loss'].mean().backward();results.append(r)
    for r in results[1:]:
        for k,v in results[0].items():torch.testing.assert_close(v,r[k],rtol=0,atol=0)
    for name,p in local.named_parameters():
        q=dict(model.named_parameters())[name]
        if p.grad is None:assert q.grad is None
        else:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    for p in model.local_gate.parameters():assert p.grad is not None and torch.count_nonzero(p.grad)==0


def test_independent_feature_oracle_skip_padding_reset_withheld_and_poison():
    d=torch.tensor([[1.,3.,float('nan'),7.,8.,5.,9.]])
    q=torch.tensor([[2.,5.,float('nan'),10.,11.,12.,16.]])
    valid=torch.tensor([[1,1,0,1,1,1,1]],dtype=torch.bool)
    write=torch.tensor([[1,1,0,1,0,1,1]],dtype=torch.bool)
    features,observed=observation_gate_features(d,q,valid,memory_write_mask=write,time_scale=3.,quantity_scale=1.5)
    previous=None;expected=torch.zeros_like(features)
    for i in range(d.size(1)):
        if not observed[0,i]:
            if valid[0,i]:previous=None
            continue
        t=torch.log1p(d[0,i]/3);z=torch.log1p(q[0,i])/1.5
        expected[0,i]=torch.stack((t,t-previous[0] if previous else t*0,z,z-previous[1] if previous else z*0))
        previous=(t,z)
    torch.testing.assert_close(features,expected,rtol=0,atol=0)
    assert not features[0,5,1] and not features[0,5,3]
    gate=ObservedLocalGate(16,'quantity',time_scale=3.,quantity_scale=1.5)
    with torch.no_grad():gate.temporal.weight.fill_(.3);gate.quantity.weight.fill_(-.2);gate.temporal.bias.fill_(.1)
    actual=gate(d,q,valid,memory_write_mask=write)
    oracle=torch.where(observed.unsqueeze(-1),2*torch.sigmoid(.1+.3*expected[...,:2].sum(-1,keepdim=True)-.2*expected[...,2:].sum(-1,keepdim=True)),0.)
    torch.testing.assert_close(actual,oracle)


def test_time_only_gate_cannot_read_quantity_but_quantity_gate_can():
    d,q,mask=inputs();a=ObservedLocalGate(16,'time',time_scale=3.,quantity_scale=1.5)
    b=ObservedLocalGate(16,'quantity',time_scale=3.,quantity_scale=1.5)
    with torch.no_grad():a.temporal.weight.fill_(.2);b.temporal.weight.fill_(.2);b.quantity.weight.fill_(.4)
    assert torch.equal(a(d,q,mask),a(d,q*3,mask))
    assert not torch.equal(b(d,q,mask),b(d,q*3,mask))
    assert sum(p.numel() for p in a.parameters())==3
    assert sum(p.numel() for p in b.parameters())==5


@pytest.mark.parametrize('mode',['time','quantity'])
def test_staged_gradients_target_future_and_invalid_padding(mode):
    torch.manual_seed(31);m=CountAwareTitanLocalGate(**kwargs(),local_gate_mode=mode).eval();activate(m.multilag_detail)
    with torch.no_grad():
        for p in m.local_gate.parameters():p.fill_(.1)
    d,q,valid=inputs();r=target_outputs(m,d,valid,q,lambda_log_qty=1.);r['joint_loss'].sum().backward()
    for name,p in m.local_gate.named_parameters():assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    dd=d.clone();qq=q.clone();dd[0,-1]=999.;qq[0,-1]=999.
    changed=target_outputs(m,dd,valid,qq,lambda_log_qty=1.)
    for k in ('pred_qty',):torch.testing.assert_close(changed[k],r[k],rtol=0,atol=0)
    full=m.encode_task_states(d,q,valid)[0]
    prefix=m.encode_task_states(d[:,:3],q[:,:3],valid[:,:3])[0]
    torch.testing.assert_close(full[:,:3],prefix,rtol=1e-5,atol=1e-6)
    dd=d.clone();qq=q.clone();dd[~valid]=float('nan');qq[~valid]=float('inf')
    torch.testing.assert_close(m.encode_task_states(dd,qq,valid)[0],full,rtol=0,atol=0)


def test_exact_state_restore_and_wrong_mode_rejected_before_mutation():
    a=CountAwareTitanLocalGate(**kwargs(),local_gate_mode='quantity');activate(a.multilag_detail)
    b=CountAwareTitanLocalGate(**kwargs(),local_gate_mode='quantity');b.load_state_dict(a.state_dict())
    d,q,m=inputs();a.eval();b.eval();assert torch.equal(a.encode_task_states(d,q,m)[0],b.encode_task_states(d,q,m)[0])
    before=deepcopy(b.state_dict());bad=deepcopy(before);bad['local_gate.scales'][0]+=1
    with pytest.raises(RuntimeError,match='identity'):b.load_state_dict(bad)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in before.items())
    other=CountAwareTitanLocalGate(**kwargs(),local_gate_mode='time')
    with pytest.raises(RuntimeError):other.load_state_dict(a.state_dict())


def test_new_roles_and_generic_gpu_cli_are_isolated(monkeypatch):
    from types import SimpleNamespace
    from paper.scripts import run_count_aware_tpp_backbone_control as cli
    from paper.scripts.count_aware_tpp_backbone.constants import validate_model_role_contract
    validate_model_role_contract(model_role=LOCAL_GATE_ROLE,backbones=LOCAL_GATE_BACKBONES,
        quantity_variants=('count_only_log_regression',),time_head_mode='heteroscedastic_lognormal_duration',lambda_tail=0.)
    with pytest.raises(ValueError):validate_model_role_contract(model_role='experimental',backbones=LOCAL_GATE_BACKBONES,
        quantity_variants=('count_only_log_regression',),time_head_mode='heteroscedastic_lognormal_duration',lambda_tail=0.)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:pytest.fail('GPU touched'))
    with pytest.raises(ValueError,match='approved dedicated supervisor'):cli.run(SimpleNamespace(model_role=LOCAL_GATE_ROLE,backbones=LOCAL_GATE_BACKBONES[0]))
