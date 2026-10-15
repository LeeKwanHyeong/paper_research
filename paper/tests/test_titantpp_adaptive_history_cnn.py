"""Fabricated CPU tensors only; no real dataset, checkpoint, or optimization."""
from copy import deepcopy
from unittest.mock import patch

import polars as pl
import pytest
import torch

from models.TPPs import CountAwareTitanAdaptiveHistoryCNN as candidate
from models.TPPs.CountAwareTitanCNNGRU import observed_context, causal_depthwise_residual
from paper.scripts import prepare_titantpp_adaptive_history_cnn as route
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)


@pytest.fixture(autouse=True)
def forbid_real_work():
    def no(*args,**kwargs): raise AssertionError("No real inputs, saved checkpoints, or optimizer steps")
    with patch.object(torch,"load",no),patch.object(pl,"read_parquet",no),patch.object(pl,"scan_parquet",no), \
         patch.object(torch.optim,"AdamW",no),patch.object(torch.optim,"Adam",no),patch.object(torch.optim,"SGD",no):
        yield


def build(arm,seed=42):
    torch.manual_seed(seed)
    data=route.original.fabricated_data()
    return route.build_model(data,arm)


def batch():
    dt=torch.tensor([[0.,1.,3.,2.,4.,2.,3.,5.],[0.,0.,0.,2.,5.,1.,4.,2.]])
    q=torch.tensor([[0.,0.,5.,2.,7.,4.,3.,6.],[0.,0.,0.,2.,4.,8.,1.,5.]])
    mask=torch.tensor([[False]+[True]*7,[False]*3+[True]*5])
    return dt,q,mask


def activate(model):
    """Fabricated weights exercise paths hidden by zero initialization."""
    with torch.no_grad():
        attn=model.encoder.layers[0].attn
        for name in ("q","k","v"):
            getattr(attn,"causal_"+name+"_kernel").normal_(0,.1)
        for p in attn.history_networks.parameters(): p.normal_(0,.1)
        model.quantity_head.weight.normal_(0,.1)
        model.time_scale_weight.weight.normal_(0,.01)
    model.eval()


@pytest.mark.parametrize("arm",candidate.ARMS)
@pytest.mark.parametrize("seed",[42,52,62])
def test_common_initial_tensors_rng_counts_and_neutral_output(arm,seed):
    baseline,_=build(candidate.CNN,seed); rng=torch.get_rng_state().clone()
    model,_=build(arm,seed)
    assert torch.equal(torch.get_rng_state(),rng)
    state=model.state_dict()
    assert all(torch.equal(v,state[k]) for k,v in baseline.state_dict().items())
    active=lambda m:sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert active(model)-active(baseline)==243
    assert sum(p.numel() for p in model.encoder.layers[0].attn.history_networks.parameters())==243
    assert not any(p.requires_grad for p in model.multilag_detail.parameters())
    dt,q,mask=batch(); baseline.eval();model.eval()
    with torch.no_grad():
        a=baseline._encode_base(dt,q,mask);b=model._encode_base(dt,q,mask)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_taps_mean_bound_neutral_and_short_history():
    mask=torch.tensor([[True,True,False,True,True]])
    write=torch.tensor([[True,True,False,False,True]])
    _,sources,available=observed_context(mask,memory_write_mask=write)
    raw=torch.tensor([[[1e8,-1e8]]*5])
    w=candidate.tap_multipliers(raw,available)
    assert w.min()>=.5 and w.max()<=1.5
    sums=(w*available[...,None]).sum(-2)
    torch.testing.assert_close(sums,available.sum(-1)[...,None].expand_as(sums).to(sums.dtype))
    assert torch.equal(w[:,0],torch.ones_like(w[:,0]))
    assert torch.equal(w[:,4],torch.ones_like(w[:,4]))  # reset after withheld


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_nonzero_kernels_neutral_matches_original_filter(arm):
    values=torch.randn(2,7,64); valid=torch.rand(2,7)>.2
    _,sources,available=observed_context(valid)
    kernel=torch.randn(3,64);network=candidate.HistoryCNNFilter()
    actual=network(values,kernel,sources,available,arm=arm)
    expected=causal_depthwise_residual(values,kernel,sources,available)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_target_padding_batch_and_eval_state_invariance(arm):
    model,_=build(arm);activate(model)
    dt,q,mask=batch();before=deepcopy(model.state_dict())
    with torch.no_grad(),patch.object(model.multilag_detail,"forward",side_effect=AssertionError("GRU was used")):
        out=core.target_outputs(model,dt,mask,q,lambda_log_qty=1.)
        changed=q.clone();changed[:,-1]=float("nan");changed[~mask]=float("nan")
        d=dt.clone();d[~mask]=float("nan")
        altered=core.target_outputs(model,d,mask,changed,lambda_log_qty=1.)
        # Target time legitimately affects its likelihood; target quantity cannot affect prediction.
        torch.testing.assert_close(out["pred_qty"],altered["pred_qty"],rtol=0,atol=0)
        d=dt.clone();d[:,-1]+=1000
        shifted=core.target_outputs(model,d,mask,q,lambda_log_qty=1.)
        torch.testing.assert_close(out["pred_qty"],shifted["pred_qty"],rtol=0,atol=0)
        for i in range(2):
            one=core.target_outputs(model,dt[i:i+1],mask[i:i+1],q[i:i+1],lambda_log_qty=1.)
            torch.testing.assert_close(one["pred_qty"],out["pred_qty"][i:i+1],rtol=1e-5,atol=1e-6)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_future_does_not_change_prefix_and_inactive_rows_zero(arm):
    model,_=build(arm);activate(model)
    dt,q,mask=batch()
    with torch.no_grad():
        a=model._encode_base(dt,q,mask)
        q[:,5:]+=999;dt[:,5:]+=444
        b=model._encode_base(dt,q,mask)
    torch.testing.assert_close(a[:,:5],b[:,:5],rtol=0,atol=0)
    assert torch.equal(a[~mask],torch.zeros_like(a[~mask]))


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_all_extra_parameters_receive_finite_nonzero_gradients(arm):
    network=candidate.HistoryCNNFilter()
    # First-layer gradients are initially zero by deliberate zero final-layer init.
    with torch.no_grad(): network.network[-1].weight.normal_(0,.2)
    values=torch.randn(2,9,64,requires_grad=True);kernel=torch.randn(3,64,requires_grad=True)
    _,sources,available=observed_context(torch.ones(2,9,dtype=torch.bool))
    output=network(values,kernel,sources,available,arm=arm)
    output.square().mean().backward()
    for p in network.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and (p.grad.abs()>0).all()
    assert torch.isfinite(values.grad).all() and torch.isfinite(kernel.grad).all()


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_identity_roundtrip_and_wrong_structure_rejected(arm):
    model,meta=build(arm);state=model.state_dict()
    rebuilt,_=build(arm);rebuilt.load_state_dict(state)
    other=candidate.CAPACITY if arm==candidate.ADAPTIVE else candidate.ADAPTIVE
    wrong,_=build(other)
    with pytest.raises(ValueError,match="identity"): wrong.load_state_dict(state)
    baseline,_=build(candidate.CNN)
    with pytest.raises(ValueError): model.load_state_dict(baseline.state_dict())
    broken=deepcopy(state);broken[next(k for k in state if k.startswith(candidate.EXTRA_PREFIX))].flatten()[0]=float("nan")
    with pytest.raises(RuntimeError,match="state tensor mismatch"): model.load_state_dict(broken)
    payload={"backbone":arm,"encoder_config":meta,"model_state_dict":state,
             "resume_identity":{"backbone":arm,"arguments":{"model_role":candidate.role_for_arm(arm)}}}
    # A relabel must be rejected before requiring other training-specific fields.
    with pytest.raises(ValueError,match="relabelled"): candidate.validate_checkpoint(payload,candidate.CNN)
    with pytest.raises(ValueError,match="metadata or route"): candidate.validate_checkpoint(payload,other)


@pytest.mark.parametrize("arm",candidate.ARMS)
def test_complete_checkpoint_contract_passes_and_selector_change_fails(arm):
    model,meta=build(arm)
    interface={"time_head":meta["time_head"]}
    payload={"backbone":arm,"encoder_config":meta,"model_state_dict":model.state_dict(),
        "evaluation_scope":"validation_only","held_out_test_evaluated":False,"variant":"count_only_log_regression",
        "interface_meta":interface,"resume_identity":{"backbone":arm,"interface_meta":interface,
        "checkpoint_monitor":"validation_raw_quantity_rmse","arguments":{"model_role":candidate.role_for_arm(arm)}}}
    assert candidate.validate_checkpoint(payload,arm)
    bad=deepcopy(payload);bad["resume_identity"]["checkpoint_monitor"]="test_raw_quantity_rmse"
    with pytest.raises(ValueError): candidate.validate_checkpoint(bad,arm)
    bad=deepcopy(payload);bad["model_state_dict"][candidate.EXTRA_PREFIX+"q.network.0.weight"][0,0]=float("nan")
    with pytest.raises(ValueError,match="network state invalid"): candidate.validate_checkpoint(bad,arm)


def test_preparation_cannot_start_training():
    with pytest.raises(PermissionError,match="Preparation only"): route.run_fit()
