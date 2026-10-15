"""Synthetic execution bridge tests; never open research inputs or contact GPU hosts."""
import json
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from paper.scripts import run_titantpp_adaptive_history_cnn as r
import importlib.util
ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT/'search_artifacts/titantpp_cnn_history_raf6_20261015_v1'
spec=importlib.util.spec_from_file_location('raf_op', BUNDLE/'operation/runtime.py')
op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
assert Path(r.__file__).resolve().is_relative_to(BUNDLE/'source')

def raf_data():
    data=r.fabricated_data()
    data['loader']['max_seq_len']=84
    data['loader']['lookback_weeks']=84
    data['dataset_id']='raf_spare_parts'
    return data
from models.TPPs.CountAwareTitanAdaptiveHistoryCNN import validate_checkpoint, EXTRA_PREFIX
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)

@pytest.mark.parametrize('arm',r.ARMS)
def test_training_factory_and_serialized_identity(tmp_path,arm):
    data=raf_data();c={'training':r.TRAINING,'evaluation_scope':'validation_only',
        'held_out_test_evaluated':False,'test_evaluation':False,'source':{'base_git_revision':'a'*40}}
    args=r.training_args(c,data,tmp_path,arm)
    assert (args.epochs,args.min_epochs,args.early_stopping_patience,args.batch_size)==(300,40,40,128)
    from paper.scripts.count_aware_tpp_backbone import training, observed_time
    observed_time.validate_launch(args,(arm,),(r.VARIANT,))
    kwargs={k:v for k,v in data['model'].items() if k not in ('backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier')}
    m,meta=training.build_model(arm,**kwargs,train_log_mean=data['statistics']['train_log_mean'],train_log_std=data['statistics']['train_log_std'],max_seq_len=data['loader']['max_seq_len'])
    interface={'time_head':meta['time_head']}
    payload={'backbone':arm,'encoder_config':meta,'model_state_dict':m.state_dict(),'evaluation_scope':'validation_only',
        'held_out_test_evaluated':False,'variant':r.VARIANT,'interface_meta':interface,
        'resume_identity':{'backbone':arm,'interface_meta':interface,'checkpoint_monitor':args.checkpoint_monitor,'arguments':{'model_role':args.model_role}}}
    path=tmp_path/'synthetic.pt';torch.save(payload,path)
    loaded=torch.load(path,weights_only=False);training.validate_checkpoint_route(loaded,arm);m.load_state_dict(loaded['model_state_dict'])
    wrong=next(a for a in r.ARMS if a!=arm)
    with pytest.raises(ValueError):training.validate_checkpoint_route(loaded,wrong)

@pytest.mark.parametrize('seed',[42,52,62])
def test_all_initializations_preserve_baseline(seed):
    data=raf_data();assert set(r.initial_states(data,seed))==set(r.ARMS)

@pytest.mark.parametrize('arm',r.ARMS)
def test_real_optimizer_reaches_all_added_parameters(arm):
    m,_=r.build_model(raf_data(),arm)
    dt=torch.tensor([[0.,1.,3.,2.,4.,2.,3.,5.],[0.,2.,3.,2.,5.,1.,4.,2.]])
    q=torch.tensor([[0.,0.,5.,2.,7.,4.,3.,6.],[0.,0.,0.,2.,4.,8.,1.,5.]])
    mask=torch.ones_like(dt,dtype=torch.bool)
    optim=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);active=set()
    names={n for n,p in m.named_parameters() if n.startswith(EXTRA_PREFIX)}
    for i in range(5):
        optim.zero_grad(set_to_none=True);loss=core.target_outputs(m,dt,mask,q,lambda_log_qty=1.)['joint_loss'].mean();loss.backward()
        assert torch.isfinite(loss)
        active.update(n for n,p in m.named_parameters() if n in names and p.grad is not None and p.grad.ne(0).any())
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optim.step()
    assert names==active
    assert not any(p.requires_grad for p in m.multilag_detail.parameters())

def test_no_baseline_training_or_test():
    data=raf_data();c={'training':r.TRAINING,'evaluation_scope':'validation_only',
        'held_out_test_evaluated':False,'test_evaluation':False,'source':{'base_git_revision':'a'*40}}
    with pytest.raises(ValueError):r.training_args(c,data,Path('/unused'),r.CNN)
    for key,value in [('test_evaluation',True),('evaluation_scope','test')]:
        with pytest.raises(ValueError):r.training_args({**c,key:value},data,Path('/unused'),r.ARMS[0])

def test_exact_six_jobs_and_parallel_waves():
    jobs=op.expected_jobs();assert len(jobs)==len({j['id'] for j in jobs})==6
    assert {j['host'] for j in jobs}=={'5090'}
    for arm in r.ARMS:assert {j['seed'] for j in jobs if j['arm']==arm}=={42,52,62}
    assert op.PARALLELISM['maximum_workers']==3

def test_failure_never_retries(tmp_path):
    (tmp_path/'failure.json').write_text('{}')
    with patch.object(op,'validate',return_value=None),pytest.raises(ValueError,match='no automatic retry'):
        op.authorization({},tmp_path)

def test_gpu_foreign_worker_rejected():
    from types import SimpleNamespace
    active={'a':{'child':SimpleNamespace(pid=20),'command':['timeout','--signal=TERM','--kill-after=15s','40','python','mine']}}
    with pytest.raises(ValueError,match='Foreign'):
        op.verify_gpu_membership({30},{30:{'ppid':99,'pgrp':99,'argv':['python','foreign']}},active,script='mine',used_bytes=1,total_bytes=10)


def test_raf_contract_and_all_deployment_bytes():
    c=op.read(BUNDLE/'execution_contract.json');op.validate(c,verify_files=False)
    for name,digest in op.read(BUNDLE/'deployment_manifest.json')['files'].items():
        assert op.sha_file(BUNDLE/name)==digest
    assert c['datasets'][0]['loader']['max_seq_len']==84
    assert c['datasets'][0]['inherited_data_identity']['populations']['validation']['target_count']==6690

@pytest.mark.parametrize('field', ['test','budget','host','gpu'])
def test_reject_scope_drift(field):
    c=op.read(BUNDLE/'execution_contract.json')
    if field=='test':c['test_evaluation']=True
    if field=='budget':c['training']['maximum_epochs']=301
    if field=='host':c['jobs'][0]['host']='5080'
    if field=='gpu':c['hosts']['5090']['gpu_uuid']='foreign'
    with pytest.raises(ValueError):op.validate(c,verify_files=False)

@pytest.mark.parametrize('arm', r.ARMS)
def test_raf_exact_active_capacity_and_neutral_output(arm):
    data=raf_data();torch.manual_seed(42);baseline,_=r.build_model(data,r.CNN)
    torch.manual_seed(42);model,_=r.build_model(data,arm)
    assert sum(p.numel() for p in baseline.parameters() if p.requires_grad)==79427
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==79670
    baseline.eval();model.eval();dt=torch.ones(2,84);q=torch.arange(168).reshape(2,84).float();mask=torch.ones_like(dt,dtype=torch.bool)
    before=deepcopy(model.state_dict())
    with torch.no_grad():
        a=core.target_outputs(baseline,dt,mask,q,lambda_log_qty=1.)['pred_qty']
        b=core.target_outputs(model,dt,mask,q,lambda_log_qty=1.)['pred_qty']
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        q[:,-1]=float('nan')
        d=core.target_outputs(model,dt,mask,q,lambda_log_qty=1.)['pred_qty']
        torch.testing.assert_close(b,d,rtol=0,atol=0)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())


def test_six_result_collection_and_paired_differences(tmp_path):
    spec=importlib.util.spec_from_file_location('raf_followup',ROOT/'paper/scripts/advance_titantpp_history_raf6.py')
    followup=importlib.util.module_from_spec(spec);spec.loader.exec_module(followup)
    c=op.read(BUNDLE/'execution_contract.json')
    endpoint={'count':6690,'qty_rmse':4.,'qty_mae':2.,'time_nll':1.,
              'tail':{'qty_rmse':8.,'qty_mae':5.,'time_nll':3.}}
    for job in c['jobs']:
        folder=tmp_path/'originals'/'run'/job['id']
        prefix=folder/'runs'/job['arm']/'count_only_log_regression'/f"seed_{job['seed']}"
        followup.write(prefix/'endpoint_replays.json',{'best_epoch':2,'selected':endpoint})
        followup.write(prefix/'history.json',{'history':[{'epoch':1,'val_qty_rmse':5.},
            {'epoch':2,'val_qty_rmse':4.},{'epoch':3,'val_qty_rmse':4.}]})
        followup.write(folder/'terminal_manifest.json',{'job':job,'held_out_test_evaluated':False,
            'files':{str(p.relative_to(folder)):followup.sha(p) for p in prefix.iterdir()}})
    report=tmp_path/'report';report.mkdir()
    with patch.object(followup,'REPORT',report):result=followup.analyze(c,tmp_path/'originals')
    assert result['seed_count']==3 and len(result['rows'])==9 and len(result['seed_differences'])==9
    assert result['aggregate']['Adaptive']['RMSE']=={'mean':4.,'sample_sd':0.}
    assert result['comparison_runtime_limitation']==c['comparison_runtime_limitation']
    assert all(d['RMSE']['absolute']==0 for d in result['seed_differences'] if d['contrast']=='Adaptive−Capacity')
