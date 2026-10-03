"""Queued Instacart replications: pure guards and CPU integration only."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import types
import pytest
from paper.scripts import local_detail_instacart_replication_contract as contract
from paper.scripts import local_detail_instacart_queue as queue
from paper.scripts import run_local_detail_benchmark as runner
from simple_lab_test.search.tests import test_observed_time_joint_training as helpers
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract, cpu_training


def metadata():
    c=contract.protocol();c['source']={'base_git_revision':'1'*40,'files':{},'files_sha256':runner.common.sha_json({})}
    c['verification']={'status':'passed_synthetic_cpu','failed':0};return c


def test_frozen_instacart_scope():
    c=metadata();contract.validate_contract(c,verify_source=False);old=contract.read(contract.ROOT/contract.BASE)
    assert list(c['hosts']) == ['5090'] and c['total_arms']==12 and c['endpoint_replays']==24
    assert c['maximum_optimizer_steps']==56005200
    assert runner.scheduled_jobs(c,'5090')==[('insta_market_basket',52),('insta_market_basket',62)]
    assert c['datasets']==old['datasets'] and c['architecture']==old['architecture']
    assert c['launch']['fixed_deadline_unix']==old['launch']['fixed_deadline_unix']
    assert c['replication']['dependency']['contract_sha256']==contract.BASE_SHA


@pytest.mark.parametrize('field',['seed','deadline','host','arm'])
def test_scope_drift_rejected(field):
    c=metadata()
    if field=='seed':c['replication']['jobs'][0]['seed']=42
    elif field=='deadline':c['launch']['fixed_deadline_unix']+=1
    elif field=='host':c['launch']['allowed_execution_hosts']=['5080']
    else:c['arms'].pop()
    with pytest.raises(ValueError):contract.validate_contract(c,verify_source=False)


@pytest.fixture
def completed_evidence(tmp_path):
    source=Path('search_artifacts/local_detail_benchmark_intermittent_recovery_20260923_v4/monitor/20260923T235525Z/terminal_5080')
    target=tmp_path/'previous';shutil.copytree(source,target)
    prior=runner.read('paper/contracts/local_detail_benchmark_observed_time_v4.json')
    return target,prior,'intermittent_frozen_5000','5080'


def test_completed_real_seed42_json_audit(completed_evidence):
    result=contract.audit_completed_dataset(*completed_evidence)
    assert result['status']=='passed' and result['endpoint_replays']==12 and result['optimizer_steps']==1375419
    assert result['checkpoint_loaded'] is False


@pytest.mark.parametrize('file,change',[
 ('run/partition_summary.json',lambda x:x.update(endpoint_replays=11)),
 ('run/intermittent_frozen_5000/runs/titantpp/count_only_log_regression/seed_42/history.json',lambda x:x['history'][0].update(val_qty_rmse=0.0001)),
 ('run/intermittent_frozen_5000/runs/titantpp/count_only_log_regression/seed_42/exposure.json',lambda x:x['train'][0].update(count=0)),
])
def test_incomplete_or_corrupt_predecessor_blocks_start(completed_evidence,file,change):
    root,_,_,_=completed_evidence;p=root/file;x=runner.read(p);change(x);runner.common.write_json(p,x)
    with pytest.raises(ValueError):contract.audit_completed_dataset(*completed_evidence)


@pytest.fixture
def queue_env(tmp_path,monkeypatch):
    c=metadata();c['hosts']['5090'].update(root=str(tmp_path),source_root=str(tmp_path/'source'))
    c['replication']['dependency']['tmux']='prior'
    runner.common.write_json(tmp_path/'frozen_execution/execution_contract.json',c)
    runner.common.write_json(tmp_path/'approval.json',{})
    runner.common.write_json(tmp_path/'start_permit.json',{'deadline_unix':9999999999})
    for n in ('verify_authorization','source_and_host','verify_process_environment','verify_start_memory'):
        monkeypatch.setattr(runner,n,lambda *a,**k:None)
    monkeypatch.setattr(runner.shared,'check_storage',lambda *a:None)
    monkeypatch.setattr(contract,'dependency_status',lambda *a:{'status':'ready_for_audit'})
    monkeypatch.setattr(contract,'verify_predecessors',lambda *a:{'status':'passed'})
    calls=[]
    def process(args,**kwargs):
        calls.append(args);return types.SimpleNamespace(returncode=1 if 'has-session' in args else 0)
    monkeypatch.setattr(queue.subprocess,'run',process)
    return tmp_path,c,calls


def test_queue_waits_without_starting(queue_env,monkeypatch):
    root,c,calls=queue_env
    monkeypatch.setattr(contract,'dependency_status',lambda *a:{'status':'waiting_for_seed42'})
    assert queue.check_once(root,{},start=True)['status']=='waiting_for_seed42'
    assert not calls and not (root/'queue_claim.json').exists()


def test_queue_starts_once_and_duplicate_call_cannot_restart(queue_env):
    root,c,calls=queue_env
    assert queue.check_once(root,{},start=False)['status']=='ready_not_started'
    assert not (root/'queue_claim.json').exists()
    assert queue.check_once(root,{},start=True)['status']=='dispatch_started'
    assert queue.check_once(root,{},start=True)['status']=='already_requested'
    assert sum('new-session' in cmd for cmd in calls)==1


def test_uncertain_launch_is_never_automatically_retried(queue_env,monkeypatch):
    root,c,calls=queue_env
    def fail(args,**kw):
        if 'new-session' in args:raise RuntimeError('synthetic launch failure')
        return types.SimpleNamespace(returncode=1)
    monkeypatch.setattr(queue.subprocess,'run',fail)
    with pytest.raises(RuntimeError):queue.check_once(root,{},start=True)
    assert (root/'queue_launch_failure.json').exists()
    assert queue.check_once(root,{},start=True)['status']=='already_requested'


def test_deadline_and_source_checks_run_before_any_start(queue_env,monkeypatch):
    root,c,calls=queue_env
    def reject(*a,**k):raise ValueError('Expired common deadline')
    monkeypatch.setattr(runner,'verify_authorization',reject)
    with pytest.raises(ValueError,match='Expired'):queue.check_once(root,{},start=True)
    assert not calls
    with pytest.raises(ValueError,match='identity'):
        queue.check_once(root,{'approval.json':'wrong'},start=True)



@pytest.mark.parametrize('dataset_id',['insta_market_basket'])
def test_both_seeds_all_models_production_checkpoint_and_batch_contract(cpu_training,admitted_contract,monkeypatch,tmp_path,dataset_id):
    c=contract.protocol();c['datasets']=deepcopy(admitted_contract['datasets']);c['source']={'base_git_revision':'1'*40,'files_sha256':'synthetic'}
    for d in c['datasets']:
        d['steps_per_epoch']=(d['inherited_data_identity']['populations']['train']['target_count']+127)//128
        d['expected_global_steps']=d['steps_per_epoch']*300
    c['replication']['jobs']=[{'host':'5090','dataset':dataset_id,'seed':s} for s in (52,62)]
    c['training']['maximum_epochs']=40
    out=tmp_path/'production';out.mkdir();c['hosts']['5090'].update(output_dir=str(out),assigned_datasets=[dataset_id])
    runtime={'device':'synthetic_cpu'}
    monkeypatch.setattr(runner.common,'runtime_check',lambda *a:runtime)
    monkeypatch.setattr(runner,'verify_process_environment',lambda *a,**k:None)
    monkeypatch.setattr(runner,'validate_execution_contract',lambda *a,**k:None)
    real_args,real_replay=runner.training_args,runner.replay_checkpoint
    def cpu_args(*a):
        args=real_args(*a);args.device='cpu';return args
    def cpu_replay(*a,**kw):
        kw['device']='cpu';return real_replay(*a,**kw)
    monkeypatch.setattr(runner,'training_args',cpu_args);monkeypatch.setattr(runner,'replay_checkpoint',cpu_replay)
    result=runner.production(c,{'deadline_unix':10**12},'5090',{'runtime':runtime},lambda:None)
    assert result['status']=='complete' and result['endpoint_replays']==24 and result['total_optimizer_steps']==480
    assert set(result['datasets'])=={dataset_id+'/seed_52',dataset_id+'/seed_62'}
    hashes=[]
    for seed in (52,62):
        ds=result['datasets'][dataset_id+f'/seed_{seed}'];assert ds['seed']==seed and ds['batch_prefix_equal']
        run=out/dataset_id/f'seed_{seed}/runs/titantpp/count_only_log_regression'/f'seed_{seed}'
        state=helpers.torch_load_checkpoint(run/'last_epoch_state.pt',map_location='cpu');assert state['seed']==seed
        frame,_=runner.base.prepare_admitted_data(next(d for d in c['datasets'] if d['dataset_id']==dataset_id))
        with pytest.raises(ValueError,match='arm/seed'):
            real_replay(run/'last_epoch_state.pt',next(d for d in c['datasets'] if d['dataset_id']==dataset_id),frame,lambda:None,
                        device='cpu',expected_arm='titantpp',expected_seed=42)
        hashes.append(runner.read(run/'exposure.json')['train'][0]['batch_order_sha256'])
    assert hashes[0]!=hashes[1]
    with pytest.raises(FileExistsError):runner.production(c,{'deadline_unix':10**12},'5090',{'runtime':runtime},lambda:None)

