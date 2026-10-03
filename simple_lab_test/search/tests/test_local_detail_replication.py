"""Explicit additional seeds; synthetic CPU training only, no external actions."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
from paper.scripts import local_detail_replication_contract as contract
from paper.scripts import run_local_detail_benchmark as runner
from paper.scripts.count_aware_tpp_backbone import training
from simple_lab_test.search.tests import test_observed_time_joint_training as helpers
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract, cpu_training
from simple_lab_test.search.tests.test_local_detail_benchmark import qualified_receipt


def admitted_metadata():
    c=contract.protocol()
    c['source']={'base_git_revision':'1'*40,'files':{},'files_sha256':runner.common.sha_json({})}
    c['verification']={'status':'passed_synthetic_cpu','failed':0}
    return c


def test_exact_replication_scope_and_unchanged_scientific_settings():
    c=admitted_metadata();old=contract.read(contract.ROOT/contract.BASE)
    contract.validate_contract(c,verify_source=False)
    assert c['total_arms']==24 and c['endpoint_replays']==48 and c['maximum_optimizer_steps']==12157200
    assert c['replication']['seeds']==[52,62] and c['replication']['report_seeds']==[42,52,62]
    assert c['datasets']==old['datasets'] and c['architecture']==old['architecture']
    assert {k:v for k,v in c['training'].items() if k!='seed'}=={k:v for k,v in old['training'].items() if k!='seed'}
    assert runner.scheduled_jobs(c,'5080')==[(ds,seed) for ds in ['yellow_trip_hourly','intermittent_frozen_5000'] for seed in (52,62)]
    assert c['launch']['fixed_deadline_unix']==old['launch']['fixed_deadline_unix']
    assert list(c['hosts'])==['5080'] and c['policy']['held_out'] is False


@pytest.mark.parametrize('change',['seed','model','head','deadline','job'])
def test_replication_contract_rejects_scope_drift(change):
    c=admitted_metadata()
    if change=='seed':c['replication']['seeds'].append(72)
    elif change=='model':c['arms'].pop()
    elif change=='head':c['datasets'][0]['model']['time_head_mode']='legacy_clamped_rmtpp'
    elif change=='deadline':c['launch']['fixed_deadline_unix']+=1
    else:c['replication']['jobs'].append(c['replication']['jobs'][0])
    with pytest.raises(ValueError):contract.validate_contract(c,verify_source=False)


@pytest.mark.parametrize('dataset_id',['yellow_trip_hourly','intermittent_frozen_5000'])
def test_seed_changes_initialization_without_changing_shared_B_local_pair(cpu_training,dataset_id):
    d=next(d for d in contract.protocol()['datasets'] if d['dataset_id']==dataset_id)
    state=torch.get_rng_state().clone();a=runner.initialization(d,52);b=runner.initialization(d,62)
    assert torch.equal(torch.get_rng_state(),state)
    assert a==runner.initialization(d,52) and all(a[k]!=b[k] for k in a)


@pytest.mark.parametrize('dataset_id',['yellow_trip_hourly','intermittent_frozen_5000'])
def test_both_seeds_all_models_production_checkpoint_and_batch_contract(cpu_training,admitted_contract,monkeypatch,tmp_path,dataset_id):
    c=contract.protocol();c['datasets']=deepcopy(admitted_contract['datasets']);c['source']={'base_git_revision':'1'*40,'files_sha256':'synthetic'}
    for d in c['datasets']:
        d['steps_per_epoch']=(d['inherited_data_identity']['populations']['train']['target_count']+127)//128
        d['expected_global_steps']=d['steps_per_epoch']*300
    c['replication']['jobs']=[{'host':'5080','dataset':dataset_id,'seed':s} for s in (52,62)]
    c['training']['maximum_epochs']=40
    out=tmp_path/'production';out.mkdir();c['hosts']['5080'].update(output_dir=str(out),assigned_datasets=[dataset_id])
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
    result=runner.production(c,{'deadline_unix':10**12},'5080',{'runtime':runtime},lambda:None)
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
    with pytest.raises(FileExistsError):runner.production(c,{'deadline_unix':10**12},'5080',{'runtime':runtime},lambda:None)


@pytest.mark.parametrize('seed',[52,62])
@pytest.mark.parametrize('arm',runner.ARMS)
def test_new_seed_checkpoint_resume_is_exact_on_synthetic_cpu(cpu_training,monkeypatch,tmp_path,seed,arm):
    helpers.install_loader(monkeypatch,{'train':[],'validation':[]})
    args=helpers.args_for(monkeypatch,tmp_path,arm,'full');args.seeds=str(seed)
    if arm=='titantpp_local_detail':args.model_role='observed_time_multilag_detail_v1'
    meta=helpers.interface()
    def run(a):
        return training.train_one(args=a,frame=helpers.grid_frame(),quantity_contract={"boundaries":[],"strata":[{"label":"all"}]},interface_meta=meta,
                                  backbone=arm,quantity_variant=runner.VARIANT,seed=seed)
    full,_,_=run(args);resumed=deepcopy(args);resumed.output_dir=tmp_path/'resume'
    original=training.atomic_torch_save
    def interrupt(payload,path):
        original(payload,path)
        if path.name=='last_epoch_state.pt' and payload['epoch']==1:raise RuntimeError('synthetic interrupt')
    monkeypatch.setattr(training,'atomic_torch_save',interrupt)
    with pytest.raises(RuntimeError,match='synthetic interrupt'):run(resumed)
    monkeypatch.setattr(training,'atomic_torch_save',original);restored,_,_=run(resumed)
    assert full['checkpoint_state_sha256']==restored['checkpoint_state_sha256']
    suffix=Path('runs')/arm/runner.VARIANT/f'seed_{seed}'/'last_epoch_state.pt'
    a=helpers.torch_load_checkpoint(args.output_dir/suffix,map_location='cpu');b=helpers.torch_load_checkpoint(resumed.output_dir/suffix,map_location='cpu')
    assert a['seed']==b['seed']==seed
    for name in ('model_state_dict','optimizer_state_dict','rng_state','train_loader_generator_state','history','best_state_dict'):
        helpers.same(a[name],b[name])


def test_native_qualification_seed_and_preserved_deadline(monkeypatch):
    c=admitted_metadata();monkeypatch.setattr(runner.time,'time',lambda:c['launch']['fixed_started_at_unix']+1000)
    a={'approved':True,'hosts':['5080'],'user_instruction':'synthetic test','contract_sha256':runner.common.sha_json(c)}
    permit=runner.make_start_permit(c,a);q=qualified_receipt(c,a,permit,'5080');q['seed']=52
    p=runner.make_training_permit(c,a,permit,{'5080':q});runner.verify_authorization(c,a,p,'5080',training=True)
    q['seed']=42
    with pytest.raises(ValueError,match='Qualification seed'):runner.make_training_permit(c,a,permit,{'5080':q})
    with pytest.raises(ValueError,match='execution scope'):runner.verify_authorization(c,a,p,'5090',training=True)


def test_replication_watchdog_rejects_wrong_seed(tmp_path):
    c=admitted_metadata();c['hosts']['5080']['output_dir']=str(tmp_path)
    budget=runner.ArmBudget(c,'5080','yellow_trip_hourly','titantpp',lambda:None,seed=52)
    runner.verify_active_arm_deadline(c,'5080')
    record=runner.read(tmp_path/'active_arm.json');record['seed']=42;runner.common.write_json(tmp_path/'active_arm.json',record)
    with pytest.raises(ValueError,match='identity'):runner.verify_active_arm_deadline(c,'5080')
