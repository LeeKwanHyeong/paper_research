"""The new shared-head campaign: genuine CPU training, only synthetic data."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from paper.scripts import local_detail_benchmark_contract as contract
from paper.scripts import run_local_detail_benchmark as runner
from paper.scripts.count_aware_tpp_backbone import training
from simple_lab_test.search.tests import test_observed_time_joint_training as helpers
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract, cpu_training


def synthetic_contract(admitted):
    c = contract.protocol()
    c.pop('recovery')  # Synthetic three-dataset integration is independent of real recovery lineage.
    c['datasets'] = deepcopy(admitted['datasets'])
    c['source'] = {'base_git_revision':'1'*40,'files_sha256':'synthetic_source'}
    for d in c['datasets']:
        d['steps_per_epoch'] = (d['inherited_data_identity']['populations']['train']['target_count']+127)//128
        d['expected_global_steps'] = d['steps_per_epoch']*300
    return c


def test_frozen_scope_and_common_arguments(tmp_path):
    c=contract.protocol(); c['source']={'base_git_revision':'1'*40}
    assert c['total_arms']==18 and c['maximum_optimizer_steps']==34081200
    assert c['endpoint_replays']==36 and c['approved'] is False
    assignments=[d for s in c['hosts'].values() for d in s['assigned_datasets']]
    assert assignments == ['intermittent_frozen_5000']
    assert c['launch']['fixed_started_at_unix'] == 1790080128.1799538
    assert c['launch']['fixed_deadline_unix'] == 1790944128.1799538
    assert c['launch']['allowed_execution_hosts'] == ['5080']
    assert c['recovery']['new_training_arms'] == 6
    assert c['recovery']['new_endpoint_replays'] == 12
    assert c['recovery']['maximum_new_optimizer_steps'] == 5538600
    assert c['launch']['requires_both_native_qualifications'] is False
    for d in c['datasets']:
        for arm in runner.ARMS:
            a=runner.training_args(c,d,tmp_path,arm)
            assert (a.epochs,a.min_epochs,a.early_stopping_patience,a.batch_size)==(300,40,40,128)
            assert a.checkpoint_monitor=='validation_raw_quantity_rmse'
            assert a.lambda_log_qty==1. and a.lambda_tail==0.
            assert a.time_head_mode=='heteroscedastic_lognormal_duration'


@pytest.mark.parametrize('dataset_id',['yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket'])
def test_six_models_real_data_adapter_and_checkpoint_replay(cpu_training,admitted_contract,tmp_path,dataset_id):
    c=synthetic_contract(admitted_contract)
    d=next(d for d in c['datasets'] if d['dataset_id']==dataset_id)
    frame,_=runner.base.prepare_admitted_data(d)
    meta=runner.time_interface(d,frame,c); initial=runner.initialization(d)
    quantity={'boundaries':d['quantity_boundaries_all_train_rows'],'strata':[{'label':f'bin_{i}'} for i in range(5)]}
    exposures={}
    for arm in runner.ARMS:
        args=runner.training_args(c,d,tmp_path/'train',arm)
        args.device='cpu';args.epochs=args.min_epochs=args.early_stopping_patience=3
        with runner.shared.audited_training(training,d,lambda:None,lambda *_:None) as exposure:
            summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,
                interface_meta=meta,backbone=arm,quantity_variant=runner.VARIANT,seed=42)
        exposures[arm]=deepcopy(exposure)
        assert summary['initial_state_sha256']==initial[arm]
        run=args.output_dir/'runs'/arm/runner.VARIANT/'seed_42'
        history=runner.read(run/'history.json')['history']
        settings={'minimum_epochs':3,'maximum_epochs':3,'patience':3}
        assert runner.audit_arm(history,summary,exposure,d,settings)==3*d['steps_per_epoch']
        identity=training._resume_identity(args=args,backbone=arm,quantity_variant=runner.VARIANT,seed=42,
            monitor=args.checkpoint_monitor,quantity_contract=quantity,interface_meta=meta)
        for label,path in [('selected',Path(summary['checkpoint_path'])),('last',run/'last_epoch_state.pt')]:
            epoch=summary['best_epoch'] if label=='selected' else 3
            result=runner.replay_checkpoint(path,d,frame,lambda:None,device='cpu',expected_arm=arm,
                expected_identity=identity,expected_initial=initial[arm],expected_epoch=epoch)
            assert result['qty_rmse']==pytest.approx(history[epoch-1]['val_qty_rmse'],abs=1e-9)
            assert result['qty_mae']==pytest.approx(history[epoch-1]['val_qty_mae'],abs=1e-9)
            assert result['time_nll']==pytest.approx(history[epoch-1]['val_time_nll'],abs=1e-9)
            runner.audit_replay_accounting(result)
    assert runner.audit_batch_prefixes(exposures)
    shorter=deepcopy(exposures);shorter[runner.ARMS[-1]]['train']=shorter[runner.ARMS[-1]]['train'][:1]
    assert runner.audit_batch_prefixes(shorter)
    shorter[runner.ARMS[-1]]['train'][0]['batch_order_sha256']='changed'
    with pytest.raises(ValueError,match='prefix'):runner.audit_batch_prefixes(shorter)


@pytest.mark.parametrize('arm',runner.ARMS)
def test_actual_interrupted_resume_model_optimizer_rng_selector(cpu_training,monkeypatch,tmp_path,arm):
    helpers.install_loader(monkeypatch,{'train':[],'validation':[]})
    args=helpers.args_for(monkeypatch,tmp_path,arm,'full')
    if arm=='titantpp_local_detail':args.model_role='observed_time_multilag_detail_v1'
    meta=helpers.interface(); meta.update(execution_contract_sha256='synthetic_300_contract',source_files_sha256='synthetic_source')
    summary,_,_=helpers.run(args,arm,meta)
    resumed=deepcopy(args);resumed.output_dir=tmp_path/'resume'
    original_save=training.atomic_torch_save
    def interrupt(payload,path):
        original_save(payload,path)
        if path.name=='last_epoch_state.pt' and payload['epoch']==1:raise RuntimeError('synthetic interrupt')
    monkeypatch.setattr(training,'atomic_torch_save',interrupt)
    with pytest.raises(RuntimeError,match='synthetic interrupt'):helpers.run(resumed,arm,meta)
    monkeypatch.setattr(training,'atomic_torch_save',original_save)
    restored,_,_=helpers.run(resumed,arm,meta)
    assert summary['checkpoint_state_sha256']==restored['checkpoint_state_sha256']
    full=helpers.torch_load_checkpoint(helpers.directory(args,arm)/'last_epoch_state.pt',map_location='cpu')
    resumed_state=helpers.torch_load_checkpoint(helpers.directory(resumed,arm)/'last_epoch_state.pt',map_location='cpu')
    for k in ['model_state_dict','optimizer_state_dict','rng_state','train_loader_generator_state','history','best_state_dict']:
        helpers.same(full[k],resumed_state[k])
    wrong=deepcopy(meta);wrong['source_files_sha256']='changed'
    with pytest.raises(ValueError,match='identity'):helpers.run(resumed,arm,wrong)


@pytest.mark.parametrize('arm',runner.ARMS)
def test_earliest_tie_drives_exact_early_stop(cpu_training,monkeypatch,tmp_path,arm):
    helpers.install_loader(monkeypatch,{'train':[],'validation':[]})
    args=helpers.args_for(monkeypatch,tmp_path,arm)
    args.epochs=8;args.min_epochs=2;args.early_stopping_patience=2
    if arm=='titantpp_local_detail':args.model_role='observed_time_multilag_detail_v1'
    original=training.evaluate
    monkeypatch.setattr(training,'evaluate',lambda **kw:{**original(**kw),'qty_rmse':42.})
    result,_,_=helpers.run(args,arm)
    assert result['best_epoch']==1 and result['completed_epochs']==3 and result['stopped_early']


def test_early_stop_audit_rejects_extra_epochs():
    history=[{'epoch':i,'val_qty_rmse':1.,'train_all_finite':True} for i in range(1,43)]
    summary={'status':'success','completed_epochs':42,'stopped_early':True,'best_epoch':1,'held_out_test_evaluated':False}
    with pytest.raises(ValueError,match='first stopping epoch'):
        runner.audit_arm(history,summary,{}, {},contract.TRAINING)


def test_unapproved_never_reaches_cuda_or_real_data(monkeypatch):
    c=contract.protocol()
    monkeypatch.setattr(contract,'validate_contract',lambda *a,**kw:None)
    monkeypatch.setattr(runner,'read',lambda path:{'c':c,'a':{'approved':False},'p':{}}[path])
    monkeypatch.setattr(runner,'source_and_host',lambda *a:pytest.fail('Unapproved runtime access'))
    with pytest.raises(ValueError,match='Explicit approval'):runner.supervisor('qualify','c','a','p','5080')


def test_contract_drift_rejected_without_source_access():
    c=contract.protocol()
    c['source']={'base_git_revision':'1'*40,'files':{},'files_sha256':runner.common.sha_json({})}
    c['verification']={'status':'passed_synthetic_cpu','failed':0}
    c['cost_estimate']={'native_current_head_benchmark_timing_measured':False}
    contract.validate_contract(c,verify_source=False)
    for mutate in [lambda x:x['training'].update(maximum_epochs=120),lambda x:x['arms'].pop(),
        lambda x:x['datasets'][0]['model'].update(time_head_mode='legacy_clamped_rmtpp'),
        lambda x:x['policy'].update(resume=True),lambda x:x['limits'].update(total_wall_seconds=900000)]:
        changed=deepcopy(c);mutate(changed)
        with pytest.raises(ValueError):contract.validate_contract(changed,verify_source=False)


@pytest.mark.parametrize('dataset_id',['yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket'])
def test_complete_production_adapter_on_synthetic_cpu(cpu_training,admitted_contract,monkeypatch,tmp_path,dataset_id):
    """Exercise every production aggregation step, including all 12 endpoint replays."""
    c=synthetic_contract(admitted_contract)
    c['training']={**c['training'],'maximum_epochs':40}
    host='5090';out=tmp_path/'production';out.mkdir()
    c['hosts'][host]={**c['hosts'][host],'assigned_datasets':[dataset_id],'output_dir':str(out)}
    runtime={'device':'synthetic_cpu'}
    monkeypatch.setattr(runner.common,'runtime_check',lambda *a:runtime)
    monkeypatch.setattr(runner,'verify_process_environment',lambda *a,**k:None)
    monkeypatch.setattr(contract,'validate_contract',lambda *a,**k:None)
    original_args=runner.training_args; original_replay=runner.replay_checkpoint
    def cpu_args(*a):
        result=original_args(*a);result.device='cpu';return result
    def cpu_replay(*a,**kw):
        kw['device']='cpu';return original_replay(*a,**kw)
    monkeypatch.setattr(runner,'training_args',cpu_args)
    monkeypatch.setattr(runner,'replay_checkpoint',cpu_replay)
    result=runner.production(c,{'deadline_unix':10**12},host,{'runtime':runtime},lambda:None)
    assert result['status']=='complete' and result['endpoint_replays']==12
    assert result['total_optimizer_steps']==240
    ds=result['datasets'][dataset_id]
    assert ds['batch_prefix_equal'] is True and set(ds['arms'])==set(runner.ARMS)
    assert len(ds['comparisons'])==5
    assert all(a['completed_epochs']==40 and a['first40']['count']==40 for a in ds['arms'].values())


def qualified_receipt(c,a,p,host):
    rows=[{'length':length,'batch_size':128,'measurements':{arm:{'parameters':100,
        'step_seconds':[.1,.2,.3],'median_step_seconds':.2,'peak_allocated_bytes':100}
        for arm in runner.ARMS}} for length in (64,256)]
    return {'status':'passed','host':host,'device':'cuda:0','synthetic_optimizer_updates':60,
        'real_data_loaded':False,'held_out_evaluated':False,
        'contract_sha256':runner.common.sha_json(c),'approval_sha256':runner.common.sha_json(a),
        'source_files_sha256':c['source']['files_sha256'],'started_at_unix':p['started_at_unix'],
        'deadline_unix':p['deadline_unix'],'runtime':{**c['hosts'][host]['runtime_expected'],
            'gpu':{'uuid':c['hosts'][host]['gpu_uuid'],'total_memory_bytes':1000}},
        'process_environment':c['process_environment'][host],'library_sha256':c['library_sha256'][host],
        'checks':dict.fromkeys(['finite_joint_updates','B_local_initial_output_equal','shared_head_contract','causality','state_restore'],True),
        'costs':rows}


def test_qualification_and_deadline_identity(monkeypatch):
    c=contract.protocol();c['source']={'files_sha256':'synthetic'}
    c['launch'].update(allowed_execution_hosts=['5080','5090'],fixed_started_at_unix=None,fixed_deadline_unix=None)
    monkeypatch.setattr(contract,'validate_contract',lambda *a,**k:None)
    monkeypatch.setattr(runner.time,'time',lambda:1000.)
    a={'approved':True,'contract_sha256':runner.common.sha_json(c),'hosts':['5080','5090'],'user_instruction':'test only'}
    p=runner.make_start_permit(c,a,started_at_unix=1000.)
    q=runner.make_training_permit(c,a,p,{'5080':qualified_receipt(c,a,p,'5080')})
    for mutate in [lambda x:x['qualifications'].pop('5080'),lambda x:x.update(deadline_unix=x['deadline_unix']+1),
        lambda x:x['qualifications']['5080']['receipt'].update(synthetic_optimizer_updates=59),
        lambda x:x['qualifications']['5080']['receipt']['costs'][0]['measurements'].pop('nhp')]:
        wrong=deepcopy(q);mutate(wrong)
        if '5080' in wrong['qualifications']:
            entry=wrong['qualifications']['5080'];entry['receipt_sha256']=runner.common.sha_json(entry['receipt'])
        with pytest.raises(ValueError):runner.verify_authorization(c,a,wrong,'5080',training=True)


@pytest.mark.parametrize('host',['5080','5090'])
def test_staggered_hosts_keep_same_deadline_and_cannot_borrow_qualification(monkeypatch,host):
    c=contract.protocol();c['source']={'files_sha256':'synthetic'}
    c['launch'].update(allowed_execution_hosts=['5080','5090'],fixed_started_at_unix=None,fixed_deadline_unix=None)
    monkeypatch.setattr(contract,'validate_contract',lambda *a,**k:None)
    monkeypatch.setattr(runner.time,'time',lambda:1000.)
    a={'approved':True,'contract_sha256':runner.common.sha_json(c),'hosts':['5080','5090'],'user_instruction':'test only'}
    start=runner.make_start_permit(c,a,started_at_unix=1000.)
    monkeypatch.setattr(runner.time,'time',lambda:1000.+7*3600)
    permit=runner.make_training_permit(c,a,start,{host:qualified_receipt(c,a,start,host)})
    assert permit['started_at_unix']==start['started_at_unix']
    assert permit['deadline_unix']==start['deadline_unix']
    assert runner.verify_authorization(c,a,permit,host,training=True)==c['hosts'][host]
    other='5090' if host=='5080' else '5080'
    with pytest.raises(ValueError,match="own native qualification"):
        runner.verify_authorization(c,a,permit,other,training=True)
    renamed=deepcopy(permit);entry=renamed['qualifications'].pop(host)
    renamed['training_host']=other;renamed['qualifications'][other]=entry
    with pytest.raises(ValueError,match='Native qualification identity'):
        runner.verify_authorization(c,a,renamed,other,training=True)
    monkeypatch.setattr(runner.time,'time',lambda:start['deadline_unix'])
    with pytest.raises(ValueError,match='Expired'):
        runner.verify_authorization(c,a,permit,host,training=True)


def test_watchdog_accepts_actual_float_rounding_without_relaxing_deadline(tmp_path,monkeypatch):
    import math
    c=contract.protocol();c['hosts']['5080']['output_dir']=str(tmp_path)
    wall,mono=1790088709.4908912,701048.45825594
    monkeypatch.setattr(runner.time,'time',lambda:wall)
    monkeypatch.setattr(runner.time,'monotonic',lambda:mono)
    budget=runner.ArmBudget(c,'5080','intermittent_frozen_5000','titantpp',lambda:None)
    active=runner.read(tmp_path/'active_arm.json')
    assert active['deadline_monotonic']-active['started_monotonic']!=active['limit_seconds']
    runner.verify_active_arm_deadline(c,'5080')
    wrong=deepcopy(active)
    wrong['deadline_monotonic']=math.nextafter(active['deadline_monotonic'],math.inf)
    runner.common.write_json(tmp_path/'active_arm.json',wrong)
    with pytest.raises(ValueError,match='identity'):
        runner.verify_active_arm_deadline(c,'5080')
    runner.common.write_json(tmp_path/'active_arm.json',active)
    monkeypatch.setattr(runner.time,'monotonic',lambda:active['deadline_monotonic'])
    with pytest.raises(ValueError,match='deadline reached'):
        runner.verify_active_arm_deadline(c,'5080')
    with pytest.raises(ValueError,match='deadline reached'):budget()


def test_5080_correction_preserves_old_clock_and_does_not_authorize_5090(monkeypatch):
    c=contract.protocol();c['source']={'files_sha256':'synthetic'}
    monkeypatch.setattr(contract,'validate_contract',lambda *a,**k:None)
    started=c['launch']['fixed_started_at_unix']
    monkeypatch.setattr(runner.time,'time',lambda:started+40000.)
    a={'approved':True,'contract_sha256':runner.common.sha_json(c),'hosts':['5080'],'user_instruction':'test only'}
    p=runner.make_start_permit(c,a)
    assert p['started_at_unix']==started and p['deadline_unix']==c['launch']['fixed_deadline_unix']
    permit=runner.make_training_permit(c,a,p,{'5080':qualified_receipt(c,a,p,'5080')})
    runner.verify_authorization(c,a,permit,'5080',training=True)
    with pytest.raises(ValueError,match='execution scope'):
        runner.verify_authorization(c,a,permit,'5090',training=True)
    with pytest.raises(ValueError,match='original common deadline'):
        runner.make_start_permit(c,a,started_at_unix=started+40000.)


@pytest.fixture
def recovery_evidence(tmp_path):
    c = contract.protocol()
    r = c['recovery']; r['old_root'] = str(tmp_path)
    predecessor = {'synthetic': True}
    r['old_contract_sha256'] = runner.common.sha_json(predecessor)
    runner.common.write_json(tmp_path/'frozen_execution/execution_contract.json', predecessor)
    for name, status in [('run/status.json','stopped'), ('dispatch_status.json','failed')]:
        runner.common.write_json(tmp_path/name, {'status':status, 'error':r['old_error']})
    taxi = tmp_path/'run/yellow_trip_hourly/paired_comparison.json'
    runner.common.write_json(taxi, {'status':'complete', 'synthetic':True})
    r['completed_evidence_sha256'] = {str(taxi.relative_to(tmp_path)):runner.common.sha_file(taxi)}
    r['initial_state_sha256'] = {arm:'synthetic_'+arm for arm in runner.ARMS}
    runner.common.write_json(tmp_path/'run/intermittent_frozen_5000/initialization.json', r['initial_state_sha256'])
    return c


def test_recovery_reads_only_frozen_predecessor_and_rejects_saved_progress(recovery_evidence):
    c = recovery_evidence
    root = runner.verify_recovery_evidence(c, '5080')
    saved = root/'run/intermittent_frozen_5000/runs/titantpp/checkpoint.pt'
    saved.parent.mkdir(parents=True); saved.write_bytes(b'synthetic checkpoint; must not be loaded')
    with pytest.raises(ValueError, match='saved progress'):
        runner.verify_recovery_evidence(c, '5080')


@pytest.mark.parametrize('change', ['taxi','contract','status','scope','initialization'])
def test_recovery_rejects_changed_predecessor(recovery_evidence, change):
    c = recovery_evidence; root = Path(c['recovery']['old_root'])
    if change == 'scope':
        c['hosts']['5080']['assigned_datasets'].append('yellow_trip_hourly')
    else:
        name = {'taxi':'run/yellow_trip_hourly/paired_comparison.json',
                'contract':'frozen_execution/execution_contract.json',
                'status':'run/status.json',
                'initialization':'run/intermittent_frozen_5000/initialization.json'}[change]
        runner.common.write_json(root/name, {'changed':True})
    with pytest.raises(ValueError):
        runner.verify_recovery_evidence(c, '5080')


def test_recovery_rejects_active_predecessor_tmux(recovery_evidence, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k:SimpleNamespace(returncode=0))
    with pytest.raises(ValueError, match='tmux remains active'):
        runner.verify_recovery_predecessor(recovery_evidence, '5080')
