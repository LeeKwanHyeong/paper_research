from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from paper.scripts import run_titantpp_core_ablation as runner
from models.TPPs.CountAwareTitanCoreAblation import HistoryCorrection, ARMS, MODES
from models.TPPs.CountAwareTitanMultiLagDetail import lag_source_indices
from models.TPPs.CountAwareFactory import validate_checkpoint_route
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract, cpu_training


def draft():
    return json.loads(Path('search_artifacts/titantpp_core_ablation_20260928_v1/draft_contract.json').read_text())


def test_scope_and_budget():
    c=draft();runner.validate_execution_contract(c)
    assert len(runner.scheduled_jobs(c,'5080'))==24
    assert len(runner.scheduled_jobs(c,'5090'))==12
    assert len(c['reuse'])==18
    changed=deepcopy(c);changed['jobs'].append(changed['jobs'][0])
    with pytest.raises(ValueError):runner.validate_execution_contract(changed)


@pytest.mark.parametrize('mode',['mlp','level','change'])
def test_formula_padding_reset_gradients_and_causality(mode):
    torch.manual_seed(42);m=HistoryCorrection(8,mode)
    assert sum(p.numel() for p in m.parameters())==8*8*4*(3 if mode=='mlp' else 2)
    h=torch.randn(2,12,8,requires_grad=True);valid=torch.ones(2,12,dtype=torch.bool);valid[0,3]=False
    observed=valid.clone();observed[1,5]=False
    assert torch.equal(m(h,valid,memory_write_mask=observed),torch.zeros_like(h))
    for p in m.output_projections:torch.nn.init.normal_(p.weight)
    sources,available=lag_source_indices(valid,memory_write_mask=observed,mode='local')
    ref=torch.zeros_like(h)
    for b in range(2):
        for t in range(12):
            for branch in range(8):
                if available[b,t,branch]:
                    prev=h[b,sources[b,t,branch]];cur=h[b,t]
                    x=torch.cat([cur,prev]) if mode=='mlp' else (cur+prev)*.5 if mode=='level' else cur-prev
                    ref[b,t]+=m.output_projections[branch](F.gelu(m.input_projections[branch](x)))/8
    out=m(h,valid,memory_write_mask=observed)
    torch.testing.assert_close(out,ref,rtol=2e-5,atol=1e-6)
    g=torch.autograd.grad(out.sum(),h,retain_graph=True)[0];gref=torch.autograd.grad(ref.sum(),h)[0]
    torch.testing.assert_close(g,gref,rtol=2e-5,atol=2e-6)
    changed=h.detach().clone();changed[:,9:]+=100;changed[~observed]=float('nan')
    torch.testing.assert_close(m(changed,valid,memory_write_mask=observed)[:,:9],out[:,:9])


def test_initialization_capacity_and_no_static_memory():
    c=draft()
    for d in c['datasets']:
        initial=runner.initialization(d,42)
        assert len(initial)==6
        sizes={}
        for a in runner.ALL_ARMS:
            torch.manual_seed(42);m,_=runner.build_model(d,a);sizes[a]=sum(p.numel() for p in m.parameters())
            if a=='titantpp_no_static_lmm':
                assert isinstance(m.lmm,torch.nn.Identity)
                assert not any(k.startswith('lmm.') for k in m.state_dict())
                assert any('persistent' in k for k in m.state_dict())
        assert sizes['titantpp_history_mlp']==sizes['titantpp_local_detail']
        assert sizes['titantpp_level_only']==sizes['titantpp']+4096
        assert sizes['titantpp_change_only']==sizes['titantpp']+4096
        assert sizes['titantpp_no_static_lmm']==sizes['titantpp_local_detail']-4096


@pytest.mark.parametrize('dataset_id',['yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket'])
def test_actual_synthetic_training_and_both_replays(cpu_training,admitted_contract,tmp_path,dataset_id):
    c=draft();c['datasets']=deepcopy(admitted_contract['datasets'])
    d=next(x for x in c['datasets'] if x['dataset_id']==dataset_id)
    d['steps_per_epoch']=(d['inherited_data_identity']['populations']['train']['target_count']+127)//128
    frame,_=runner.base.prepare_admitted_data(d)
    interface=runner.time_interface(d,frame,c);init=runner.initialization(d,42)
    quantity={'boundaries':d['quantity_boundaries_all_train_rows'],'strata':[{'label':f'bin_{i}'} for i in range(5)]}
    from paper.scripts.count_aware_tpp_backbone import training
    exposures=[]
    for arm in ARMS:
        args=runner.training_args(c,d,tmp_path/'train',arm);args.device='cpu';args.epochs=args.min_epochs=args.early_stopping_patience=3
        with runner.shared.audited_training(training,d,lambda:None,lambda *_:None) as exposure:
            summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,interface_meta=interface,
                backbone=arm,quantity_variant=runner.VARIANT,seed=42)
        exposures.append(exposure)
        assert summary['initial_state_sha256']==init[arm]
        run=args.output_dir/'runs'/arm/runner.VARIANT/'seed_42';history=runner.read(run/'history.json')['history']
        runner.audit_arm(history,summary,exposure,d,{'minimum_epochs':3,'maximum_epochs':3,'patience':3})
        identity=training._resume_identity(args=args,backbone=arm,quantity_variant=runner.VARIANT,seed=42,
            monitor=args.checkpoint_monitor,quantity_contract=quantity,interface_meta=interface)
        for name,epoch in [('best_val_qty_rmse_model.pt',summary['best_epoch']),('last_epoch_state.pt',3)]:
            result=runner.replay_checkpoint(run/name,d,frame,lambda:None,device='cpu',expected_arm=arm,
                expected_identity=identity,expected_initial=init[arm],expected_epoch=epoch)
            assert result['qty_rmse']==pytest.approx(history[epoch-1]['val_qty_rmse'],abs=1e-8)
            assert result['time_nll']==pytest.approx(history[epoch-1]['val_time_nll'],abs=1e-8)
            payload=torch.load(run/name,weights_only=False);validate_checkpoint_route(payload,arm)
            with pytest.raises(ValueError):validate_checkpoint_route(payload,'titantpp')
            other=next(a for a in ARMS if a!=arm)
            with pytest.raises(ValueError):validate_checkpoint_route(payload,other)
            payload['model_state_dict']['core_ablation_identity'][0]^=1
            with pytest.raises(ValueError):validate_checkpoint_route(payload,arm)
    assert all(e['train']==exposures[0]['train'] for e in exposures)


def test_native_synthetic_algorithm_on_cpu(cpu_training):
    result=runner.qualify(draft(),device='cpu',batch_size=2,lengths=(8,20))
    assert result['synthetic_optimizer_updates']==60
    assert result['real_data_loaded'] is False

def test_approval_expiry_scope_and_mutation_rejected(tmp_path,monkeypatch):
    import time
    c=draft();approval={'approved':True,'hosts':['5080','5090'],'user_instruction':'5번까지 진행하자','contract_sha256':runner.common.sha_json(c)}
    p={'contract_sha256':runner.common.sha_json(c),'approval_sha256':runner.common.sha_json(approval),'started_at_unix':time.time()-10}
    p['deadline_unix']=p['started_at_unix']+c['limits']['total_wall_seconds']
    runner.verify_authorization(c,approval,p,'5080')
    for mutation in ('budget','hash','scope'):
        changed=deepcopy(p)
        if mutation=='budget':changed['deadline_unix']+=1
        elif mutation=='hash':changed['approval_sha256']='bad'
        else:changed['started_at_unix']=time.time()+20;changed['deadline_unix']=changed['started_at_unix']+c['limits']['total_wall_seconds']
        with pytest.raises(ValueError):runner.verify_authorization(c,approval,changed,'5080')
    bad=deepcopy(approval);bad['approved']=False
    with pytest.raises(ValueError):runner.verify_authorization(c,bad,p,'5080')
    with pytest.raises(ValueError):runner.verify_authorization(c,approval,p,'other_host')


def test_registered_role_rejects_wrong_head_or_encoder():
    from paper.scripts.count_aware_tpp_backbone.constants import validate_model_role_contract
    valid=dict(model_role=runner.ROLE,backbones=(ARMS[0],),quantity_variants=(runner.VARIANT,),time_head_mode='heteroscedastic_lognormal_duration',lambda_tail=0.)
    validate_model_role_contract(**valid)
    for key,value in [('time_head_mode','legacy_clamped'),('backbones',('titantpp',)),('model_role','experimental')]:
        with pytest.raises(ValueError):validate_model_role_contract(**{**valid,key:value})
