"""Synthetic execution bridge tests; never open research inputs or contact GPU hosts."""
import json
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from paper.scripts import run_titantpp_adaptive_history_cnn as r
from paper.scripts import titantpp_history_taxi6_runtime as op
from models.TPPs.CountAwareTitanAdaptiveHistoryCNN import validate_checkpoint, EXTRA_PREFIX
from paper.scripts.count_aware_tpp_backbone import core

torch.set_num_threads(1)

@pytest.mark.parametrize('arm',r.ARMS)
def test_training_factory_and_serialized_identity(tmp_path,arm):
    data=r.fabricated_data();c={'training':r.TRAINING,'evaluation_scope':'validation_only',
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
    data=r.fabricated_data();assert set(r.initial_states(data,seed))==set(r.ARMS)

@pytest.mark.parametrize('arm',r.ARMS)
def test_real_optimizer_reaches_all_added_parameters(arm):
    m,_=r.build_model(r.fabricated_data(),arm)
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
    data=r.fabricated_data();c={'training':r.TRAINING,'evaluation_scope':'validation_only',
        'held_out_test_evaluated':False,'test_evaluation':False,'source':{'base_git_revision':'a'*40}}
    with pytest.raises(ValueError):r.training_args(c,data,Path('/unused'),r.CNN)
    for key,value in [('test_evaluation',True),('evaluation_scope','test')]:
        with pytest.raises(ValueError):r.training_args({**c,key:value},data,Path('/unused'),r.ARMS[0])

def test_exact_six_jobs_and_parallel_waves():
    jobs=op.expected_jobs();assert len(jobs)==len({j['id'] for j in jobs})==6
    assert {j['host'] for j in jobs}=={'5080'}
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
