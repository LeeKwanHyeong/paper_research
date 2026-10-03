"""No real inputs/GPU: continuation authority and real trainer/audit integration."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from paper.scripts import resume_multilag_detail_instacart as recovery
from simple_lab_test.search.tests.test_multilag_detail_execution import cpu_training, synthetic


@pytest.fixture
def authority():
    root=Path(__file__).resolve().parents[3]
    parent=json.loads((root/'paper/contracts/multilag_detail_observed_time_execution_data_v3.json').read_text())
    prefix=f'run/{recovery.DATASET}/runs/titantpp/{recovery.r.VARIANT}/seed_42/'
    files={prefix+n:'0'*64 for n in ('last_epoch_state.pt','history.json','exposure.json','epoch_timing.json','train.log')}
    files.update({n:'0'*64 for n in ('frozen_execution/execution_contract.json','approval.json','training_permit.json',
        'qualification/receipt.json','run/status.json','run/active_arm.json','run/wrapper_manifest.json',
        f'run/{recovery.DATASET}/input_receipt.json',f'run/{recovery.DATASET}/initialization.json')})
    c={'schema':'multilag_detail_instacart_explicit_resume_v1','parent_contract':parent,
       'parent_contract_sha256':recovery.PARENT_SHA,'scope':deepcopy(recovery.SCOPE),
       'old_root':str(recovery.OLD_ROOT),'new_root':str(recovery.NEW_ROOT),'saved_files':files,
       'deadline_unix':recovery.TOTAL_DEADLINE,'baseline_deadline_unix':recovery.B_DEADLINE,
       'original_started_at_unix':parent['launch']['fixed_started_at_unix']}
    a={'approved':True,'explicit_resume':True,'hosts':['5090'],'user_instruction':'5090 다시 진행하자',
       'contract_sha256':recovery.common.sha_json(c)}
    return c,a


def test_exact_recovery_authority(authority):
    c,a=authority
    assert recovery.validate(c,a,now=1789966707)['seed']==42


@pytest.mark.parametrize('damage',['approval','host','epoch','budget','deadline','b_deadline','preserved','path','expired','parent'])
def test_recovery_rejects_changed_scope(authority,damage):
    c,a=authority;now=1789966707
    if damage=='approval':a['explicit_resume']=False
    elif damage=='host':a['hosts']=['5080']
    elif damage=='epoch':c['scope']['saved_epoch']=83
    elif damage=='budget':c['scope']['new_optimizer_steps']+=1
    elif damage=='deadline':c['deadline_unix']+=1
    elif damage=='b_deadline':c['baseline_deadline_unix']+=1
    elif damage=='preserved':c['saved_files'].pop(next(iter(c['saved_files'])))
    elif damage=='path':c['saved_files']['../elsewhere']='0'*64
    elif damage=='expired':now=recovery.B_DEADLINE
    elif damage=='parent':c['parent_contract']['seed']=43
    a['contract_sha256']=recovery.common.sha_json(c)
    with pytest.raises(ValueError):recovery.validate(c,a,now=now)


def test_baseline_clock_does_not_restart(monkeypatch,tmp_path):
    monkeypatch.setattr(recovery,'NEW_ROOT',tmp_path);(tmp_path/'run').mkdir()
    monkeypatch.setattr(recovery.time,'time',lambda:1789966707.)
    monkeypatch.setattr(recovery.time,'monotonic',lambda:100.)
    c={'old_arm_clock':{'started_at_unix':recovery.B_DEADLINE-86400}}
    clock=recovery.ArmClock(c,'titantpp',lambda:None)
    assert clock.deadline==recovery.B_DEADLINE
    monkeypatch.setattr(recovery.time,'time',lambda:recovery.B_DEADLINE)
    with pytest.raises(ValueError,match='per-arm'):clock()


def test_saved_exposure_is_copied_and_checked():
    d={'loader':{'batch_size':4},'inherited_data_identity':{'populations':{'train':{'target_count':9},'validation':{'target_count':7}}}}
    rows={s:[{'epoch':1,'split':s,'count':n,'batches':(n+3)//4,'batch_order_sha256':'0'*64}]
          for s,n in [('train',9),('validation',7)]}
    history=[{'epoch':1,'train_batch_count':3,'train_event_count':9}]
    records={'train':[],'validation':[]};recovery.seed_exposure(records,rows,history,d)
    records['train'][0]['count']=1
    assert rows['train'][0]['count']==9
    with pytest.raises(ValueError,match='empty'):recovery.seed_exposure(records,rows,history,d)
    rows['validation'][0]['count']=8
    with pytest.raises(ValueError,match='mismatch'):recovery.seed_exposure({'train':[],'validation':[]},rows,history,d)


def test_saved_prefix_exact_resume_with_real_audit_and_trainer(cpu_training,monkeypatch,tmp_path):
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    r=recovery.r;arm=r.ARMS[0]
    data={'loader':{'batch_size':4},'inherited_data_identity':{'populations':{'train':{'target_count':9},'validation':{'target_count':7}}},'expected_global_steps':9}
    h.install_loader(monkeypatch,{'train':[],'validation':[]})
    full,meta=synthetic(monkeypatch,tmp_path,arm,dataset='insta_market_basket',name='full')
    with r.shared.audited_training(training,data,lambda:None,lambda e,x:None) as full_exposure:
        full_summary,_,_=h.run(full,arm,meta=meta)
    partial,_=synthetic(monkeypatch,tmp_path,arm,dataset='insta_market_basket',name='partial')
    saved={}
    def interrupt(epoch,records):
        if epoch==1:
            saved.update(deepcopy(records));raise RuntimeError('synthetic stop after saved epoch')
    with pytest.raises(RuntimeError,match='synthetic stop'):
        with r.shared.audited_training(training,data,lambda:None,interrupt):h.run(partial,arm,meta=meta)
    original=h.directory(partial,arm)
    checkpoint=h.torch_load_checkpoint(original/'last_epoch_state.pt',map_location='cpu')
    digest=r.common.sha_file(original/'last_epoch_state.pt')
    restored,_=synthetic(monkeypatch,tmp_path,arm,dataset='insta_market_basket',name='restored')
    target=h.directory(restored,arm);target.mkdir(parents=True)
    shutil.copy2(original/'last_epoch_state.pt',target/'last_epoch_state.pt')
    with r.shared.audited_training(training,data,lambda:None,lambda e,x:None) as actual_exposure:
        recovery.seed_exposure(actual_exposure,saved,checkpoint['history'],data)
        actual_summary,_,_=h.run(restored,arm,meta=meta)
    assert actual_exposure==full_exposure
    assert r.shared.audit_exposure(actual_exposure,data,epochs=3)==9
    assert actual_summary['checkpoint_state_sha256']==full_summary['checkpoint_state_sha256']
    expected=h.torch_load_checkpoint(h.directory(full,arm)/'last_epoch_state.pt',map_location='cpu')
    actual=h.torch_load_checkpoint(target/'last_epoch_state.pt',map_location='cpu')
    for key in ('model_state_dict','best_state_dict','optimizer_state_dict','rng_state','train_loader_generator_state','history'):
        h.same(expected[key],actual[key])
    assert r.common.sha_file(original/'last_epoch_state.pt')==digest
