"""Synthetic CPU verification of the fixed multilag level/detail contract."""
from __future__ import annotations

import copy
import io

import pytest
import torch
from torch.nn import functional as F

from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.TPPs.CountAwareTitanMultiLagDetail import (
    CountAwareTitanMultiLagDetail, MultiLagDetail, lag_source_indices, AVAILABILITY_LAGS,
    multilag_detail_metadata, validate_multilag_detail_checkpoint,
    LOCAL_DETAIL_BACKBONE, MULTILAG_DETAIL_BACKBONE, MULTILAG_DETAIL_ROLE,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def kwargs():
    return dict(hidden_dim=16, train_log_mean=1.5, max_seq_len=140,
        quantity_variant="count_only_log_regression", lambda_tail=0.0,
        time_head_mode="heteroscedastic_lognormal_duration", time_scale=3.0,
        time_initial_location=.1, time_initial_scale=.7,
        time_observation_contract={"mode":"positive_integer_round_clamp_v1","top_code":None,"unit":"week"})


def inputs():
    return (torch.tensor([[0.,1.,3.,2.,7.,4.],[0.,2.,1.,3.,4.,0.]]),
            torch.tensor([[1.,2.,6.,3.,12.,5.],[2.,1.,4.,8.,3.,0.]]),
            torch.tensor([[True]*6,[True]*5+[False]]))


def activate(block):
    with torch.no_grad():
        for branch, p in enumerate(block.output_projections):
            p.weight.copy_(torch.linspace(-.3,.4,p.weight.numel(),dtype=p.weight.dtype,
                                          device=p.weight.device).reshape_as(p.weight) * (1+branch/8))


def reference(block, hidden, valid, write=None):
    """Independent raw-hidden pair oracle; no projected-gather helper is used."""
    observed = valid if write is None else valid & write
    result = []
    for batch in range(hidden.size(0)):
        past, rows = [], []
        for i in range(hidden.size(1)):
            zero = torch.zeros_like(hidden[batch,i])
            if not observed[batch,i]:
                if valid[batch,i]: past=[]
                rows.append(zero); continue
            correction = zero
            for s, availability in enumerate(AVAILABILITY_LAGS):
                if len(past) < availability: continue
                lag = availability if block.mode == 'multilag' else 1
                current, source = hidden[batch,i], hidden[batch,past[-lag]]
                level, detail = (current+source)/2, current-source
                features = F.gelu(F.linear(level,block.level_projections[s].weight),approximate='none')
                features = features + F.gelu(F.linear(detail,block.detail_projections[s].weight),approximate='none')
                correction = correction + F.linear(features,block.output_projections[s].weight)
            rows.append(correction/8); past.append(i)
        result.append(torch.stack(rows) if rows else hidden[batch].clone())
    return torch.stack(result)


@pytest.mark.parametrize('mode',['local','multilag'])
@pytest.mark.parametrize('training',[False,True])
def test_initial_B_output_objective_preclip_gradients_and_dropout_rng(mode,training):
    torch.manual_seed(123); baseline=CountAwareTitanTPP(**kwargs()); expected_rng=torch.get_rng_state().clone()
    torch.manual_seed(123); model=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode=mode)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    common=dict(model.named_parameters())
    assert all(torch.equal(p,common[name]) for name,p in baseline.named_parameters())
    baseline.train(training); model.train(training)
    dts,qty,valid=inputs(); start=torch.get_rng_state().clone()
    a=target_outputs(baseline,dts,valid,qty,lambda_log_qty=1.); after=torch.get_rng_state().clone()
    torch.set_rng_state(start); b=target_outputs(model,dts,valid,qty,lambda_log_qty=1.)
    assert torch.equal(after,torch.get_rng_state())
    for key in a: assert torch.equal(a[key],b[key]),key
    a['joint_loss'].mean().backward(); b['joint_loss'].mean().backward()
    for name,p in baseline.named_parameters():
        q=common[name]
        if p.grad is None: assert q.grad is None
        else: torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)
    assert model.multilag_detail.output_projections[0].weight.grad.abs().sum()>0
    for layer in (*model.multilag_detail.level_projections,*model.multilag_detail.detail_projections):
        assert layer.weight.grad is not None and torch.count_nonzero(layer.weight.grad)==0


def test_equal_parameters_fixed_initialization_and_exact_capacity():
    torch.manual_seed(1); control=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode='local')
    torch.manual_seed(1); candidate=CountAwareTitanMultiLagDetail(**kwargs())
    for name,p in control.named_parameters():assert torch.equal(p,dict(candidate.named_parameters())[name])
    assert sum(p.numel() for p in MultiLagDetail(64).parameters())==6144
    assert all(layer.bias is None for block in (control.multilag_detail,candidate.multilag_detail)
               for layer in (*block.level_projections,*block.detail_projections,*block.output_projections))
    assert all(torch.count_nonzero(p.weight)==0 for p in candidate.multilag_detail.output_projections)
    assert candidate.multilag_detail.lags==AVAILABILITY_LAGS
    assert control.multilag_detail.lags==(1,)*8


@pytest.mark.parametrize('mode',['local','multilag'])
@pytest.mark.parametrize('length',[0,1,2,5,17,130,256])
def test_projected_gather_matches_independent_reference_and_gradients(mode,length):
    torch.manual_seed(111); block=MultiLagDetail(8,mode=mode).double(); activate(block)
    oracle=copy.deepcopy(block)
    h=torch.randn(2,length,8,dtype=torch.float64,requires_grad=True)
    href=h.detach().clone().requires_grad_()
    valid=torch.rand(2,length)>.15; write=torch.ones_like(valid)
    if length>5:write[0,3]=False
    actual=block(h,valid,memory_write_mask=write); expected=reference(oracle,href,valid,write)
    torch.testing.assert_close(actual,expected,rtol=2e-12,atol=2e-12)
    if length<2:return
    probe=torch.randn_like(actual)
    (actual*probe).sum().backward()
    if expected.requires_grad:(expected*probe).sum().backward()
    if href.grad is not None:torch.testing.assert_close(h.grad,href.grad,rtol=2e-11,atol=2e-11)
    for name,p in block.named_parameters():
        other=dict(oracle.named_parameters())[name]
        if other.grad is None:assert p.grad is None or torch.count_nonzero(p.grad)==0
        else:torch.testing.assert_close(p.grad,other.grad,rtol=2e-11,atol=2e-11,msg=name)


def test_index_masks_count_observations_not_padding_and_reset_withheld():
    valid=torch.tensor([[True,False,True,True,True,True,False,True,True]])
    write=valid.clone();write[0,4]=False
    ci,ca=lag_source_indices(valid,memory_write_mask=write,mode='multilag')
    li,la=lag_source_indices(valid,memory_write_mask=write,mode='local')
    assert torch.equal(ca,la)
    assert not ca[0,0].any() and not ca[0,1].any()
    assert ci[0,2,0]==0 and ci[0,3,1]==0 and li[0,3,1]==2
    assert not ca[0,4].any() and not ca[0,5].any()
    assert ci[0,7,0]==5 and not ca[0,7,1]
    assert ci[0,8,1]==5 and li[0,8,1]==7
    assert torch.equal(ci[~ca],torch.zeros_like(ci[~ca]))


def test_padding_insertion_and_reset_are_helper_not_positional_model_properties():
    torch.manual_seed(2); block=MultiLagDetail(8).double();activate(block)
    h=torch.randn(1,7,8,dtype=torch.float64); valid=torch.ones(1,7,dtype=torch.bool)
    original=block(h,valid)
    padded=torch.cat((h[:,:3],torch.full((1,1,8),float('nan')),h[:,3:]),dim=1)
    mask=torch.tensor([[True]*3+[False]+[True]*4])
    actual=block(padded,mask)
    torch.testing.assert_close(actual[:,[0,1,2,4,5,6,7]],original,rtol=1e-13,atol=1e-14)
    assert torch.isfinite(actual).all() and not actual[:,3].any()
    reset=block(padded,torch.ones_like(mask),memory_write_mask=mask)
    torch.testing.assert_close(reset[:,4:],block(h[:,3:],valid[:,3:]),rtol=1e-13,atol=1e-14)


@pytest.mark.parametrize('mode',['local','multilag'])
def test_every_eligible_branch_staged_gradient_and_no_zero_U_skip(mode):
    torch.manual_seed(7); block=MultiLagDetail(8,mode=mode).double()
    h=torch.randn(2,131,8,dtype=torch.float64); valid=torch.ones(2,131,dtype=torch.bool)
    probe=torch.randn_like(h)
    (block(h,valid)*probe).sum().backward()
    for layer in block.output_projections:assert layer.weight.grad.abs().sum()>0
    for layer in (*block.level_projections,*block.detail_projections):
        assert layer.weight.grad is not None and torch.count_nonzero(layer.weight.grad)==0
    block.zero_grad(set_to_none=True);activate(block)
    (block(h,valid)*probe).sum().backward()
    for name,p in block.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name


def test_control_identity_cases_and_actual_long_lag_difference():
    torch.manual_seed(9); a=MultiLagDetail(8,mode='local').double();activate(a)
    torch.manual_seed(9); b=MultiLagDetail(8,mode='multilag').double();activate(b)
    h=torch.randn(1,130,8,dtype=torch.float64); valid=torch.ones(1,130,dtype=torch.bool)
    assert torch.equal(a(h[:,:2],valid[:,:2]),b(h[:,:2],valid[:,:2]))
    constant=h[:,:1].expand(-1,130,-1)
    assert torch.equal(a(constant,valid),b(constant,valid))
    assert not torch.allclose(a(h,valid),b(h,valid),rtol=1e-8,atol=1e-8)
    # Branch1 is divided by8 even when it is the only available branch.
    only=b(h[:,:2],valid[:,:2])[:,1]
    level=(h[:,1]+h[:,0])/2;detail=h[:,1]-h[:,0]
    expected=b.output_projections[0](F.gelu(b.level_projections[0](level))+F.gelu(b.detail_projections[0](detail)))/8
    torch.testing.assert_close(only,expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('mode',['local','multilag'])
def test_helper_future_and_mask_poison_have_zero_gradient(mode):
    torch.manual_seed(6);block=MultiLagDetail(8,mode=mode).double();activate(block)
    h=torch.randn(2,12,8,dtype=torch.float64,requires_grad=True)
    valid=torch.ones(2,12,dtype=torch.bool);valid[1,9:]=False
    write=valid.clone();write[0,5]=False
    expected=block(h,valid,memory_write_mask=write)
    altered=h.detach().clone();altered[:,8:]+=999
    actual=block(altered,valid,memory_write_mask=write)
    assert torch.equal(expected[:,:8],actual[:,:8])
    poison=h.detach().clone();poison[~(valid&write)]=float('nan')
    assert torch.equal(expected,block(poison,valid,memory_write_mask=write))
    expected[:,:8].square().sum().backward()
    assert h.grad[:,8:].count_nonzero()==0
    assert h.grad[~(valid&write)].count_nonzero()==0


@pytest.mark.parametrize('mode',['local','multilag'])
def test_full_model_target_padding_withheld_and_future_causality(mode):
    torch.manual_seed(8); model=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode=mode).eval()
    activate(model.multilag_detail)
    with torch.no_grad():model.quantity_head.weight.copy_(torch.linspace(-.3,.4,16)[None])
    dt,qty,valid=inputs();baseline=target_outputs(model,dt,valid,qty,lambda_log_qty=1.)
    changed_dt,changed_qty=dt.clone(),qty.clone()
    ends=valid.sum(1)-1
    changed_dt[torch.arange(2),ends]+=30;changed_qty[torch.arange(2),ends]+=300
    changed=target_outputs(model,changed_dt,valid,changed_qty,lambda_log_qty=1.)
    assert torch.equal(baseline['pred_qty'],changed['pred_qty'])
    assert not torch.equal(baseline['joint_loss'],changed['joint_loss'])
    write=valid.clone();write[torch.arange(2),ends]=False;write[0,2]=False
    d=dt.clone().requires_grad_();q=qty.clone().requires_grad_()
    h=model.encode_task_states(d,q,valid,memory_write_mask=write)[0]
    poisoned_d=d.detach().clone();poisoned_q=q.detach().clone()
    poisoned_d[~(valid&write)]=float('inf');poisoned_q[~(valid&write)]=float('nan')
    actual=model.encode_task_states(poisoned_d,poisoned_q,valid,memory_write_mask=write)[0]
    assert torch.equal(h,actual)
    assert torch.equal(model.time_location(h),model.time_location(actual))
    h[:,3].square().sum().backward()
    assert d.grad[~(valid&write)].count_nonzero()==0 and q.grad[~(valid&write)].count_nonzero()==0
    assert d.grad[:,4:].count_nonzero()==0 and q.grad[:,4:].count_nonzero()==0
    future_d,future_q=dt.clone(),qty.clone();future_d[:,4:]+=50;future_q[:,4:]+=500
    future=model.encode_task_states(future_d,future_q,valid,memory_write_mask=write)[0]
    assert torch.equal(h[:,:4],future[:,:4])
    # No cross-sample or cross-forward memory state is carried.
    assert torch.equal(actual,model.encode_task_states(poisoned_d,poisoned_q,valid,memory_write_mask=write)[0])
    one=model.encode_task_states(dt[:1],qty[:1],valid[:1],memory_write_mask=write[:1])[0]
    torch.testing.assert_close(h[:1],one,rtol=1e-6,atol=1e-6)


@pytest.mark.parametrize('mode',['local','multilag'])
@pytest.mark.parametrize('component',['time_loss','quantity_train_loss'])
def test_full_model_both_objectives_reach_added_parameters(mode,component):
    torch.manual_seed(61); model=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode=mode).eval()
    activate(model.multilag_detail)
    with torch.no_grad():model.quantity_head.weight.copy_(torch.linspace(-.3,.4,16)[None])
    dt=torch.randint(1,20,(2,131)).float();qty=torch.randint(1,100,(2,131)).float()
    mask=torch.ones_like(dt,dtype=torch.bool)
    result=target_outputs(model,dt,mask,qty,lambda_log_qty=1.)
    result[component].mean().backward()
    for name,p in model.multilag_detail.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name


@pytest.mark.parametrize('mismatch',['mode','missing_identity','code','missing_added','missing_common','shape','nonfinite'])
def test_wrong_or_incomplete_state_rejected_before_parameter_mutation(mismatch):
    torch.manual_seed(31); model=CountAwareTitanMultiLagDetail(**kwargs())
    source=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode='local' if mismatch=='mode' else 'multilag').state_dict()
    if mismatch=='missing_identity':del source['multilag_detail.contract_identity']
    elif mismatch=='code':source['multilag_detail.mode_code']=torch.tensor(0)
    elif mismatch=='missing_added':del source['multilag_detail.level_projections.7.weight']
    elif mismatch=='missing_common':del source['encoder.input_proj.weight']
    elif mismatch=='shape':source['multilag_detail.output_projections.0.weight']=torch.ones(1)
    elif mismatch=='nonfinite':source['multilag_detail.output_projections.0.weight'][0,0]=float('nan')
    before=copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError):model.load_state_dict(source,strict=False)
    for name,value in before.items():assert torch.equal(value,model.state_dict()[name]),name


@pytest.mark.parametrize('mode',['local','multilag'])
def test_model_optimizer_and_rng_restore_reproduces_exact_next_dropout_update(mode):
    torch.manual_seed(29); model=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode=mode).train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    dt,qty,mask=inputs()
    def step(m,opt):
        opt.zero_grad(set_to_none=True)
        loss=target_outputs(m,dt,mask,qty,lambda_log_qty=1.)['joint_loss'].mean()
        loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        return loss.detach().clone()
    step(model,optimizer)
    buffer=io.BytesIO();torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},buffer)
    expected=step(model,optimizer); expected_rng=torch.get_rng_state().clone()
    restored=CountAwareTitanMultiLagDetail(**kwargs(),multilag_detail_mode=mode).train()
    restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=.001,weight_decay=.01)
    buffer.seek(0);saved=torch.load(buffer,weights_only=True)
    restored.load_state_dict(saved['model']);restored_optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    actual=step(restored,restored_optimizer)
    assert torch.equal(expected,actual) and torch.equal(expected_rng,torch.get_rng_state())
    for name,value in model.state_dict().items():assert torch.equal(value,restored.state_dict()[name]),name
    for key,state in optimizer.state_dict()['state'].items():
        for name,value in state.items():assert torch.equal(value,restored_optimizer.state_dict()['state'][key][name])


@pytest.mark.parametrize('field',['duration','quantity'])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.])
def test_invalid_observed_raw_values_fail(field,value):
    model=CountAwareTitanMultiLagDetail(**kwargs());dt,qty,mask=inputs()
    (dt if field=='duration' else qty)[0,1]=value
    with pytest.raises(ValueError,match='finite and nonnegative'):model.encode(dt,qty,mask)


@pytest.mark.parametrize('bad',[
    {'multilag_detail_rank':3}, {'multilag_detail_mode':'adaptive'},
    {'memory_mode':'none'}, {'lambda_tail':.1}, {'quantity_memory_gradient_mode':'detach'},
    {'time_head_mode':'legacy_clamped_rmtpp'}, {'time_observation_contract':None},
])
def test_unsupported_architecture_or_objective_rejected(bad):
    with pytest.raises(ValueError):CountAwareTitanMultiLagDetail(**(kwargs()|bad))


def checkpoint_payload(mode='multilag',*,summary=False,dataset='intermittent_frozen_5000'):
    config=kwargs();unit,cap={'intermittent_frozen_5000':('week',None),'yellow_trip_hourly':('hour',None),'insta_market_basket':('day',30)}[dataset]
    config['time_observation_contract']={'mode':'positive_integer_round_clamp_v1','unit':unit,'top_code':cap}
    model=CountAwareTitanMultiLagDetail(**config,multilag_detail_mode=mode);head=model.time_head_contract()
    metadata=multilag_detail_metadata(16,mode)|{'time_head':head}
    interface={'time_head':dict(head,statistics_source_split='train',train_time_statistics={
        'time_scale':3.,'target_log_scaled_mean':.1,'target_log_scaled_std':.7})}
    backbone=LOCAL_DETAIL_BACKBONE if mode=='local' else MULTILAG_DETAIL_BACKBONE
    identity=dict(backbone=backbone,variant='count_only_log_regression',interface_meta=interface,
        checkpoint_monitor='validation_raw_quantity_rmse',arguments={
            'dataset_contract':dataset,'model_role':MULTILAG_DETAIL_ROLE,'time_head_mode':head['mode'],'time_scale':3.})
    payload=dict(backbone=backbone,variant='count_only_log_regression',encoder_config=metadata,
        evaluation_scope='validation_only',held_out_test_evaluated=False,interface_meta=interface,resume_identity=identity)
    if summary:payload.update(checkpoint_state_sha256='a'*64,best_epoch=1)
    else:payload['model_state_dict']=model.state_dict()
    return payload


@pytest.mark.parametrize('mode',['local','multilag'])
@pytest.mark.parametrize('summary',[False,True])
@pytest.mark.parametrize('dataset',['intermittent_frozen_5000','yellow_trip_hourly','insta_market_basket'])
def test_checkpoint_state_and_summary_accept_all_frozen_dataset_observation_laws(mode,summary,dataset):
    payload=checkpoint_payload(mode,summary=summary,dataset=dataset)
    assert validate_multilag_detail_checkpoint(payload,payload['backbone'])
    assert validate_multilag_detail_checkpoint({'backbone':'titantpp'},'titantpp') is False
    for other in ('titantpp',LOCAL_DETAIL_BACKBONE if mode=='multilag' else MULTILAG_DETAIL_BACKBONE):
        with pytest.raises(ValueError):validate_multilag_detail_checkpoint(payload,other)


@pytest.mark.parametrize('field',['multilag_detail_lags','multilag_detail_availability_lags','multilag_detail_rank',
    'multilag_detail_branch_count','multilag_detail_activation','multilag_detail_residual_divisor','multilag_detail_mask'])
def test_checkpoint_architecture_metadata_cannot_be_relabeled(field):
    payload=checkpoint_payload(summary=True);payload['encoder_config'][field]='changed'
    with pytest.raises(ValueError):validate_multilag_detail_checkpoint(payload,payload['backbone'])


@pytest.mark.parametrize('mutation',['missing_branch','nonfinite_branch','wrong_code','missing_identity','held_out',
    'objective','role','observation_cap','head_scale','missing_interface','summary_digest','bare_state'])
def test_checkpoint_rejects_wrong_identity_scope_and_incomplete_payload(mutation):
    payload=checkpoint_payload(summary=mutation=='summary_digest')
    if mutation=='missing_branch':del payload['model_state_dict']['multilag_detail.detail_projections.7.weight']
    elif mutation=='nonfinite_branch':payload['model_state_dict']['multilag_detail.level_projections.0.weight'][0,0]=float('nan')
    elif mutation=='wrong_code':payload['model_state_dict']['multilag_detail.mode_code']=torch.tensor(0)
    elif mutation=='missing_identity':del payload['model_state_dict']['multilag_detail.contract_identity']
    elif mutation=='held_out':payload['held_out_test_evaluated']=True
    elif mutation=='objective':payload['variant']='other'
    elif mutation=='role':payload['resume_identity']['arguments']['model_role']='other'
    elif mutation=='observation_cap':payload['encoder_config']['time_head']['observation_likelihood']['top_code']=30
    elif mutation=='head_scale':payload['resume_identity']['arguments']['time_scale']=1.
    elif mutation=='missing_interface':del payload['interface_meta']
    elif mutation=='summary_digest':payload['checkpoint_state_sha256']='not a digest'
    else:payload={'model_state_dict':payload['model_state_dict']}
    with pytest.raises(ValueError):validate_multilag_detail_checkpoint(payload,MULTILAG_DETAIL_BACKBONE)


_STATISTICS=(('time_scale','time_scale'),('time_initial_location','target_log_scaled_mean'),('time_initial_scale','target_log_scaled_std'))
@pytest.mark.parametrize('field,statistic',_STATISTICS)
@pytest.mark.parametrize('offset,accepted',[(1e-13,True),(1e-10,False)])
def test_recomputed_train_statistics_only_allow_existing_tolerance(field,statistic,offset,accepted):
    payload=checkpoint_payload(summary=True)
    payload['interface_meta']['time_head']['train_time_statistics'][statistic]+=offset
    if accepted:assert validate_multilag_detail_checkpoint(payload,payload['backbone'])
    else:
        with pytest.raises(ValueError,match='train-only'):validate_multilag_detail_checkpoint(payload,payload['backbone'])


@pytest.mark.parametrize('field,statistic',_STATISTICS)
def test_frozen_interface_settings_require_exact_equality(field,statistic):
    payload=checkpoint_payload(summary=True);payload['interface_meta']['time_head'][field]+=1e-13
    with pytest.raises(ValueError,match='train-only'):validate_multilag_detail_checkpoint(payload,payload['backbone'])


def original_projected_branch_loop(block, hidden, valid, write):
    """Original v1 per-branch arithmetic, for the packed-only optimization check."""
    observed=valid & write
    safe=torch.where(observed.unsqueeze(-1),hidden,0.)
    sources,available=lag_source_indices(valid,memory_write_mask=write,mode=block.mode)
    correction=torch.zeros_like(safe)
    for branch,(level,detail,output) in enumerate(zip(block.level_projections,block.detail_projections,block.output_projections)):
        current_l,current_d=level(safe),detail(safe)
        index=sources[...,branch,None].expand(-1,-1,block.rank)
        past_l,past_d=current_l.gather(1,index),current_d.gather(1,index)
        eligible=available[...,branch,None]
        l=torch.where(eligible,(current_l+past_l)*.5,0.)
        d=torch.where(eligible,current_d-past_d,0.)
        f=F.gelu(l,approximate='none')+F.gelu(d,approximate='none')
        correction=correction+torch.where(eligible,output(f),0.)
    return correction/8


@pytest.mark.parametrize('mode',['local','multilag'])
@pytest.mark.parametrize('length',[64,256])
def test_packed_forward_and_all_parameter_gradients_match_original_float32_loop(mode,length):
    torch.manual_seed(127)
    packed=MultiLagDetail(64,mode=mode);activate(packed)
    loop=copy.deepcopy(packed)
    hidden=torch.randn(2,length,64,requires_grad=True)
    original=hidden.detach().clone().requires_grad_()
    valid=torch.ones(2,length,dtype=torch.bool);valid[1,3]=False
    write=valid.clone();write[1,length//3]=False
    a=packed(hidden,valid,memory_write_mask=write)
    b=original_projected_branch_loop(loop,original,valid,write)
    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
    probe=torch.randn_like(a)
    (a*probe).sum().backward();(b*probe).sum().backward()
    torch.testing.assert_close(hidden.grad,original.grad,rtol=3e-5,atol=3e-6)
    for name,p in packed.named_parameters():
        reference=dict(loop.named_parameters())[name]
        torch.testing.assert_close(p.grad,reference.grad,rtol=3e-5,atol=3e-6,msg=name)
    # Packing never replaces the independent optimizer parameters or state keys.
    assert set(packed.state_dict())==set(loop.state_dict())
    assert len(list(packed.parameters()))==24
    for name,value in packed.state_dict().items():assert torch.equal(value,loop.state_dict()[name]),name
