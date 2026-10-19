"""Fabricated recovery checks; no research prediction, SSH or shared-file mutation."""
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from paper.scripts import run_titantpp_late_path18_recovery as r
from paper.scripts import run_titantpp_late_path18 as s


class InterruptedAtSavedEpoch(RuntimeError):
    pass


def scientific_fixture(tmp_path,arm,*,seed=52,reference=False):
    from paper.tests.test_titantpp_quantity_loss_worker import fabricated_contract
    from paper.tests.test_titantpp_late_path18_worker import common_contract
    old=tmp_path/'old';old.mkdir()
    c,data=fabricated_contract(old)
    common_contract(c,time_input='log1p',mode='log_mse')
    dataset='raf_spare_parts' if reference else 'intermittent_frozen_5000'
    data['dataset_id']=dataset
    data['model']['time_observation_contract']['unit']='month' if reference else 'week'
    job={'id':f'{dataset}__{arm}__seed{seed}__losslog_mse','host':'5090','dataset':dataset,
        'seed':seed,'arm':arm,'time_alpha':0.,'loss_mode':'log_mse','time_input':'log1p'}
    c['jobs']=[job]
    c['source'].update(files={'fixture.py':'f'*64},files_sha256='fixture')
    c['hosts']['5090']['absolute_host_deadline_unix']=2.e9
    cp=old/'execution_contract.json';r.write(cp,c)
    return c,data,job


def tiny_args(monkeypatch):
    original=s.training_args
    def tiny(*args,**kwargs):
        a=original(*args,**kwargs)
        assert (a.epochs,a.min_epochs,a.early_stopping_patience)==(300,40,40)
        a.epochs=a.min_epochs=a.early_stopping_patience=4;a.device='cpu'
        return a
    monkeypatch.setattr(s,'training_args',tiny)


def control_fit(original,data,job,output,*,interrupt_epoch=None):
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts import run_multilag_detail_execution as base
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    configure_runtime('cpu',4)
    atomic=training.atomic_torch_save
    def save(value,path):
        atomic(value,path)
        if value.get('checkpoint_type')=='epoch_resume' and value['epoch']==interrupt_epoch:
            raise InterruptedAtSavedEpoch('fabricated interruption after durable checkpoint')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(training,'atomic_torch_save',save)
        with s.runtime_hooks('log_mse',time_input='log1p',arm=job['arm']):
            frame,_=base.prepare_admitted_data(data)
            args=s.training_args(original,data,output,job['arm'],time_alpha=0.)
            interface=r.interface_for(original,data,frame,job['arm'])
            return training.train_one(args=args,frame=frame,quantity_contract=r.quantity_contract(data),
                interface_meta=interface,backbone=job['arm'],quantity_variant=s.VARIANT,seed=job['seed'])


def operation_contract(original,job,new_root,mode,*,resume_path=None,train_receipt=None):
    new_root.mkdir();(new_root/'source').mkdir()
    old_root=Path(original['hosts']['5090']['root'])
    immutable={str(p.relative_to(old_root)):r.sha_file(p) for p in old_root.rglob('*') if p.is_file()}
    cp=old_root/'execution_contract.json'
    item={**deepcopy(job),'recovery_mode':mode}
    if mode=='resume':
        from simple_lab_test.search.common.runner import torch_load_checkpoint
        payload=torch_load_checkpoint(resume_path,map_location='cpu')
        item.update(origin_resume_checkpoint={'path':str(resume_path),'sha256':r.sha_file(resume_path)},
            resume_epoch=payload['epoch'],original_started_at='2026-10-19T07:00:00+00:00')
    if mode=='reference':
        scientific=r.read(train_receipt)
        item.update(origin_train_receipt={'path':str(train_receipt),'sha256':r.sha_file(train_receipt)},
            origin_selected_checkpoint={'path':scientific['selected_checkpoint'],'sha256':scientific['checkpoint_sha256']})
    return {'schema':r.CONTRACT_SCHEMA,'host':'5090','root':str(new_root),'source_root':str(new_root/'source'),
        'scientific_contract':{'path':str(cp),'raw_sha256':r.sha_file(cp),'canonical_sha256':r.sha_json(original)},
        'source':{'files':original['source']['files'],'files_sha256':original['source']['files_sha256'],
            'source_revision':original['source']['base_git_revision']},
        'absolute_host_deadline_unix':original['hosts']['5090']['absolute_host_deadline_unix'],
        'jobs':[item],'training_jobs':[] if mode=='reference' else [job['id']],'validation_jobs':[job['id']],
        'immutable_originals':{'root':str(old_root),'files':immutable}}


def last_path(folder,job):
    return folder/'runs'/job['arm']/s.VARIANT/f"seed_{job['seed']}"/'last_epoch_state.pt'


def assert_same_tree(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for key in a:assert_same_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):assert_same_tree(x,y)
    else:assert a==b


@pytest.mark.parametrize('arm',s.ARMS)
def test_interrupted_prefix_exactly_resumes_model_optimizer_rng_loader_history_and_headproof(tmp_path,monkeypatch,arm):
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    original,data,job=scientific_fixture(tmp_path,arm)
    tiny_args(monkeypatch)
    uninterrupted=tmp_path/'uninterrupted'
    control_fit(original,data,job,uninterrupted)
    baseline=torch_load_checkpoint(last_path(uninterrupted,job),map_location='cpu')
    prefix=Path(original['hosts']['5090']['root'])/'run'/job['id']
    with pytest.raises(InterruptedAtSavedEpoch):control_fit(original,data,job,prefix,interrupt_epoch=2)
    prefix_path=last_path(prefix,job)
    interrupted=torch_load_checkpoint(prefix_path,map_location='cpu')
    assert interrupted['epoch']==2 and interrupted['time_head_no_training_verification']['optimizer_steps_checked']==2
    contract=operation_contract(original,job,tmp_path/'new','resume',resume_path=prefix_path)
    before={str(p):r.sha_file(p) for p in Path(contract['immutable_originals']['root']).rglob('*') if p.is_file()}
    result=r.train(contract,job['id'],device='cpu')
    from paper.scripts import titantpp_late_path18_5090_recovery_runtime as native_runtime
    assert native_runtime.verify_receipt(contract,contract['jobs'][0],'train')['sha256']
    resumed=torch_load_checkpoint(last_path(Path(contract['root'])/'run'/job['id'],job),map_location='cpu')
    for key in ('model_state_dict','optimizer_state_dict','rng_state','train_loader_generator_state','history',
        'best_state_dict','best_state_sha256','initial_state_sha256','resume_identity','source_revision_history',
        'best_epoch','best_selection_value','time_head_no_training_verification'):
        assert_same_tree(baseline[key],resumed[key])
    assert result['schema']==r.SCHEMA and result['lineage']['next_epoch']==3
    assert result['lineage']['resume_validation']['prefix_optimizer_steps']==2
    assert result['summary']['completed_epochs']==4
    assert result['time_head_no_training_verification']['optimizer_steps_checked']==4
    assert before=={str(p):r.sha_file(p) for p in Path(contract['immutable_originals']['root']).rglob('*') if p.is_file()}
    val=r.validation(contract,job['id'],device='cpu')
    assert native_runtime.verify_receipt(contract,contract['jobs'][0],'validation')['sha256']
    assert val['validation_replay_passed'] and val['target_identity']==result['target_identity']
    assert val['selected_epoch']==result['selected_epoch'] and val['checkpoint_sha256']==result['checkpoint_sha256']
    assert before=={str(p):r.sha_file(p) for p in Path(contract['immutable_originals']['root']).rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):r.train(contract,job['id'],device='cpu')


def test_only_unstarted_seed62_hidden_is_fresh_and_replayable(tmp_path,monkeypatch):
    original,data,job=scientific_fixture(tmp_path,s.ARMS[1],seed=62)
    tiny_args(monkeypatch)
    c=operation_contract(original,job,tmp_path/'fresh','fresh')
    tr=r.train(c,job['id'],device='cpu')
    assert tr['lineage']['fresh_from_original_seed'] and tr['lineage']['resumed_from_epoch'] is None
    assert tr['summary']['completed_epochs']==4 and tr['time_head_no_training_verification']['optimizer_steps_checked']==4
    val=r.validation(c,job['id'],device='cpu')
    assert val['validation_replay_passed'] and val['metrics']==tr['metrics']


@pytest.mark.parametrize('arm',s.ARMS)
def test_completed_raf_is_not_retrained_and_selected_original_copies_replay(tmp_path,monkeypatch,arm):
    from paper.scripts.quantity_comparison_runtime import configure_runtime
    original,data,job=scientific_fixture(tmp_path,arm,seed=52,reference=True)
    tiny_args(monkeypatch)
    evaluate=s.evaluate_selected
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr('paper.scripts.quantity_comparison_runtime.configure_runtime',lambda *args:configure_runtime('cpu',4))
        patch.setattr(s,'evaluate_selected',lambda *args,**kwargs:evaluate(*args,**kwargs,device='cpu'))
        tr=s.run_stage(original,'5090',job['id'],'train')
    origin_receipt=Path(original['hosts']['5090']['root'])/'run'/job['id']/'train_receipt.json'
    c=operation_contract(original,job,tmp_path/'replay-only','reference',train_receipt=origin_receipt)
    before={str(p):r.sha_file(p) for p in Path(c['immutable_originals']['root']).rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='cannot be retrained'):r.train(c,job['id'],device='cpu')
    v=r.validation(c,job['id'],device='cpu')
    from paper.scripts import titantpp_late_path18_5090_recovery_runtime as native_runtime
    assert native_runtime.verify_receipt(c,c['jobs'][0],'validation')['sha256']
    assert v['lineage']['reference_retrained'] is False
    assert v['checkpoint_sha256']==tr['checkpoint_sha256'] and v['selected_epoch']==tr['selected_epoch']
    assert v['metrics']==tr['validation_reference']['metrics']
    assert before=={str(p):r.sha_file(p) for p in Path(c['immutable_originals']['root']).rglob('*') if p.is_file()}
    copied=v['lineage']['reference_checkpoint']
    assert copied['copy_sha256']==copied['original_sha256']==tr['checkpoint_sha256']


def test_reference_guard_tamper_and_old_root_write_blocked(tmp_path):
    original,data,job=scientific_fixture(tmp_path,s.ARMS[1],seed=62)
    c=operation_contract(original,job,tmp_path/'new','fresh')
    r.select_job(c,job['id'])
    for key,value in (('schema','other'),('host','5080'),('absolute_host_deadline_unix',2.e9+1)):
        changed=deepcopy(c);changed[key]=value
        with pytest.raises(ValueError):r.load_scientific(changed)
    changed=deepcopy(c);changed['source']['source_revision']='new-science'
    with pytest.raises(ValueError):r.load_scientific(changed)
    binding={'path':str(Path(c['scientific_contract']['path'])),'sha256':c['scientific_contract']['raw_sha256']}
    with pytest.raises(ValueError):r.copy_origin(c,binding,Path(c['immutable_originals']['root'])/'bad-new-file')
    changed=deepcopy(c);changed['jobs'][0]['seed']=42
    with pytest.raises(ValueError):r.select_job(changed,job['id'])
    changed=deepcopy(c);changed['jobs'][0]['recovery_mode']='reference'
    with pytest.raises(ValueError):r.select_job(changed,job['id'])


@pytest.mark.parametrize('mutation',('original_hash','source_inventory','raw_time','shared_loss','not_original_root'))
def test_scientific_contract_tamper_rejected_before_execution(tmp_path,mutation):
    original,data,job=scientific_fixture(tmp_path,s.ARMS[1],seed=62)
    c=operation_contract(original,job,tmp_path/'new','fresh')
    if mutation=='original_hash':c['scientific_contract']['raw_sha256']='wrong'
    if mutation=='source_inventory':c['source']['files']={'foreign.py':'z'*64}
    if mutation=='raw_time':
        original['common_condition']['time_input']='raw'
        Path(c['scientific_contract']['path']).write_text(__import__('json').dumps(original))
        c['scientific_contract'].update(raw_sha256=r.sha_file(c['scientific_contract']['path']),canonical_sha256=r.sha_json(original))
    if mutation=='shared_loss':
        original['common_condition']['loss_mode']='raw_mae'
        Path(c['scientific_contract']['path']).write_text(__import__('json').dumps(original))
        c['scientific_contract'].update(raw_sha256=r.sha_file(c['scientific_contract']['path']),canonical_sha256=r.sha_json(original))
    if mutation=='not_original_root':c['immutable_originals']['root']=str(tmp_path/'elsewhere')
    with pytest.raises(ValueError):r.load_scientific(c)


def test_resume_is_not_replaced_by_another_fresh_seed(tmp_path):
    original,data,job=scientific_fixture(tmp_path,s.ARMS[0],seed=52)
    c=operation_contract(original,job,tmp_path/'new','fresh')
    with pytest.raises(ValueError,match='never-started'):r.select_job(c,job['id'])


def test_qualification_cli_preserves_complete_native_admission_output(tmp_path,monkeypatch,capsys):
    p=tmp_path/'contract.json';r.write(p,{'fabricated':True})
    expected={'status':'passed','real_data_loaded':False,'candidate_name':s.ARMS[0],
        'cuda_memory_peaks':{s.ARMS[0]:{'max_memory_reserved_bytes':1234}}}
    monkeypatch.setattr(r,'qualify',lambda c,j:deepcopy(expected))
    r.main(['--recovery-contract',str(p),'--stage','qualify','--job','fabricated'])
    assert __import__('json').loads(capsys.readouterr().out)==expected
