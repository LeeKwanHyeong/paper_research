"""Synthetic CPU contracts and actual trainer integration; no remote/real data."""
from __future__ import annotations

import copy
import math

import pytest

from paper.scripts import run_local_gate_execution as runner


@pytest.fixture
def authority(monkeypatch, tmp_path):
    monkeypatch.setattr(runner.proposal, 'validate_contract', lambda *a, **k: None)
    c = {'limits': {'total_wall_seconds': 172800, 'qualification_seconds_per_host': 900,
                   'per_host_output_bytes': 4*1024**3, 'min_free_bytes': 5*1024**3},
         'cost_gates': {'median_cuda_step_ratio_max': 1.5, 'peak_cuda_allocated_ratio_max': 1.25,
                        'device_memory_fraction_max': .8, 'parameter_ratio_max': 1.1},
         'source': {'files_sha256': 'synthetic_source'}, 'hosts': {},
         'process_environment': {}, 'library_sha256': {}}
    for host in runner.proposal.ASSIGNMENTS:
        c['hosts'][host] = {'root': str(tmp_path/host), 'python': 'synthetic',
                            'runtime_expected': {'torch': 'synthetic'}, 'gpu_uuid': host}
        c['process_environment'][host] = {'LD_LIBRARY_PATH': 'synthetic', 'XDG_CACHE_HOME': 'synthetic'}
        c['library_sha256'][host] = {'synthetic': 'identity'}
    a = {'approved': True, 'contract_sha256': runner.common.sha_json(c),
         'hosts': list(c['hosts']), 'user_instruction': 'synthetic test approval'}
    monkeypatch.setattr(runner.time, 'time', lambda: 1000.)
    return c, a, runner.make_start_permit(c, a, started_at_unix=1000.)


def qualification(c, a, p, host):
    rows = []
    for length in (64, 256):
        measurements = {arm: {'step_seconds': [1.]*5, 'median_step_seconds': 1.,
            'peak_allocated_bytes': 100, 'parameters': 1000 if arm == runner.ARMS[0] else 1068 if arm == runner.ARMS[1] else 1070} for arm in runner.ARMS}
        checks = {arm+':'+key: True for arm in runner.ARMS[1:] for key in ('step','memory','device','parameters')}
        rows.append({'length': length, 'batch_size': 128, 'measurements': measurements, 'checks': checks})
    return {'status': 'passed', 'host': host, 'device': 'cuda:0', 'runtime': {'torch': 'synthetic',
                'gpu': {'uuid': host, 'total_memory_bytes': 1000}}, 'synthetic_optimizer_updates': 48,
        'contract_sha256': runner.common.sha_json(c), 'approval_sha256': runner.common.sha_json(a),
        'source_files_sha256': c['source']['files_sha256'], 'started_at_unix': p['started_at_unix'],
        'deadline_unix': p['deadline_unix'], 'checks': dict.fromkeys(runner.QUALIFICATION_CHECKS, True),
        'costs': rows, 'real_data_loaded': False, 'held_out_evaluated': False,
        'process_environment': c['process_environment'][host], 'library_sha256': c['library_sha256'][host]}


def test_both_qualified_hosts_and_origin_required(authority):
    c,a,p = authority
    receipts = {h: qualification(c,a,p,h) for h in c['hosts']}
    q = runner.make_training_permit(c,a,p,receipts)
    assert q['started_at_unix'] == p['started_at_unix'] and q['deadline_unix'] == p['deadline_unix']
    q['qualifications'].pop(next(iter(c['hosts'])))
    with pytest.raises(ValueError, match='native qualification'):
        runner.verify_authorization(c,a,q,next(iter(c['hosts'])),training=True)


def test_implementation_correction_cannot_reset_budget_origin(authority):
    c,a,p=authority
    c['launch']={'fixed_started_at_unix':990.,'fixed_deadline_unix':990.+172800}
    a['contract_sha256']=runner.common.sha_json(c)
    retained=runner.make_start_permit(c,a)
    assert retained['started_at_unix']==990.
    assert retained['deadline_unix']==990.+172800
    with pytest.raises(ValueError,match='original common deadline'):
        runner.make_start_permit(c,a,started_at_unix=1000.)


@pytest.mark.parametrize('bad', ['approval','deadline','source','cuda','environment','cost','updates','checks'])
def test_invalid_authority_and_qualification_fail_closed(authority, bad):
    c,a,p = authority
    q = runner.make_training_permit(c,a,p,{h: qualification(c,a,p,h) for h in c['hosts']})
    host = next(iter(c['hosts'])); r=q['qualifications'][host]['receipt']
    if bad == 'approval': a['approved'] = False
    elif bad == 'deadline': q['deadline_unix'] += 1
    elif bad == 'source': r['source_files_sha256'] = 'wrong'
    elif bad == 'cuda': r['device'] = 'cpu'
    elif bad == 'environment': r['process_environment'] = {}
    elif bad == 'updates': r['synthetic_optimizer_updates'] = 47
    elif bad == 'checks': r['checks'] = {}
    else: r['costs'][0]['measurements'][runner.ARMS[2]]['step_seconds'] = [2.]*5
    q['qualifications'][host]['receipt_sha256'] = runner.common.sha_json(r)
    with pytest.raises(ValueError): runner.verify_authorization(c,a,q,host,training=True)


def test_unapproved_supervisor_never_touches_gpu(monkeypatch, authority):
    c,a,p=authority; a['approved']=False
    monkeypatch.setattr(runner,'read',lambda k:{'c':c,'a':a,'p':p}[k])
    monkeypatch.setattr(runner,'source_and_host',lambda *a:pytest.fail('reached GPU/source'))
    with pytest.raises(ValueError,match='Explicit approval'):
        runner.supervisor('qualify','c','a','p',next(iter(c['hosts'])))


def test_arm_cap_includes_replays_and_supervisor_watchdog(monkeypatch,tmp_path):
    c={'limits':{'per_arm_wall_seconds':20},'hosts':{'5090':{
        'output_dir':str(tmp_path),'assigned_datasets':['synthetic']}}}
    clock={'wall':100.,'mono':10.}
    monkeypatch.setattr(runner.time,'time',lambda:clock['wall'])
    monkeypatch.setattr(runner.time,'monotonic',lambda:clock['mono'])
    budget=runner.ArmBudget(c,'5090','synthetic',runner.ARMS[2],lambda:None)
    budget();runner.verify_active_arm_deadline(c,'5090')
    clock['wall']=119.;clock['mono']=30.
    with pytest.raises(ValueError,match='Per-arm'):budget.finish()
    with pytest.raises(ValueError,match='Per-arm'):runner.verify_active_arm_deadline(c,'5090')
    clock['wall']=120.;clock['mono']=29.
    with pytest.raises(ValueError,match='Per-arm'):budget()


def test_library_environment_is_process_local_and_hash_bound(monkeypatch,tmp_path):
    lib=tmp_path/'installed';lib.mkdir();file=lib/'libnvrtc.so';file.write_bytes(b'synthetic existing library')
    root=tmp_path/'run';root.mkdir()
    env={'LD_LIBRARY_PATH':str(lib),'XDG_CACHE_HOME':str(root/'cache')}
    c={'hosts':{'5090':{'root':str(root)}},'process_environment':{'5090':env},
       'library_sha256':{'5090':{str(file):runner.common.sha_file(file)}}}
    monkeypatch.setattr(runner.common,'apply_environment',lambda spec:None)
    for key in env:monkeypatch.setenv(key,'foreign')
    runner.apply_process_environment(c,'5090')
    assert runner.verify_process_environment(c,'5090',active=True)==env
    assert file.read_bytes()==b'synthetic existing library'
    file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):runner.verify_process_environment(c,'5090')


def cell(n, error, time=.3):
    return {'count': n, 'qty_mae': error if n else None, 'qty_rmse': error if n else None,
            'qty_sse': n*error**2, 'time_nll': time if n else None}


def measurements(error=1.):
    q=[{'bin':i,**cell(2,error)} for i in range(5)]
    return {'selected': {**cell(10,error), 'body':cell(6,error), 'tail':cell(2,error),
        'quantity_cells':q, 'history_cells':[{'bin':i,**cell(n,error)} for i,n in enumerate((2,4,4))],
        'quantity_boundaries':[1.,2.,3.,4.], 'history_boundaries':[64,128]},
        'last30':{'count':30,'mean':error,'sd':.1,'standard_deviation':'sample_ddof_1'}}


def test_prospective_full_gates_and_zero_reference():
    arms={a:measurements(1. if a!=runner.ARMS[2] else .9) for a in (*runner.ARMS,runner.proposal.UNGATED)}
    assert runner.compare_arms(arms)['accepted'] is True
    assert runner.compare_arms(arms)['goal_fully_met'] is False  # Equal SD is a screen pass, not a reduction.
    arms[runner.ARMS[2]]['last30']['sd']=.09
    assert runner.compare_arms(arms)['goal_fully_met'] is True
    zeros={a:measurements(0.) for a in (*runner.ARMS,runner.proposal.UNGATED)}
    for v in zeros.values():v['last30']['sd']=0.
    result=runner.compare_arms(zeros)
    assert result['verified'] and not result['accepted']
    assert result['against'][runner.ARMS[0]]['checks']['tail_rmse_improves'] is False
    assert result['against'][runner.ARMS[0]]['checks']['last30_rmse_sd_guard'] is True


@pytest.mark.parametrize('damage',['missing','nan','negative','empty_middle','sse','count','last30'])
def test_missing_empty_nonfinite_and_bad_accounting_never_pass(damage):
    arms={a:measurements(1. if a!=runner.ARMS[2] else .9) for a in (*runner.ARMS,runner.proposal.UNGATED)}
    candidate=arms[runner.ARMS[2]]; value=candidate['selected']
    if damage=='missing':value['quantity_cells'].pop()
    elif damage=='nan':value['quantity_cells'][1]['qty_mae']=float('nan')
    elif damage=='negative':value['qty_rmse']=-1.
    elif damage=='sse':value['qty_sse']+=1
    elif damage=='count':value['quantity_cells'][1]['count']=3
    elif damage=='last30':candidate['last30']['count']=29
    else:
        # Repeated quantiles yield a real empty bin; keep all accounting valid.
        for arm in arms.values():
            v=arm['selected'];e=v['qty_mae'];v['quantity_boundaries']=[1.,1.,3.,4.]
            v['quantity_cells'][0]={'bin':0,**cell(4,e)};v['quantity_cells'][1]={'bin':1,**cell(0,e)}
    result=runner.compare_arms(arms)
    assert result['status']=='unverified' and not result['accepted']


@pytest.fixture
def cpu_training():
    import torch
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def admitted_contract(tmp_path):
    """Actual execution schema, with every input replaced by fabricated data."""
    import numpy as np
    import polars as pl
    import torch
    from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
    from paper.scripts import quantity_comparison_data as qdata
    from paper.scripts import run_count_aware_tpp_backbone_control as cli
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256

    contract = runner.read(runner.ROOT/'paper/contracts/multilag_detail_observed_time_execution_packed_v2.json')
    rows = [{'oper_part_no': str(part), 'seq': i, 'delta_t': [0, 1, 2, 3, 4, 7][i % 6],
             'demand_qty': float(1+i+part*20),
             'chronological_split': 'train' if i < 8 else 'validation'}
            for part in range(2) for i in range(12)]
    raw = pl.DataFrame(rows)
    path, manifest = tmp_path/'synthetic.parquet', tmp_path/'synthetic_split.json'
    raw.write_parquet(path); manifest.write_text('{"synthetic":true}')
    frame = cli.prepare_count_frame(raw)
    quantities = raw.filter(pl.col('chronological_split') == 'train')['demand_qty'].to_numpy()
    for data in contract['datasets']:
        assert 'mu_all_train_rows' not in data
        identity = data['inherited_data_identity']
        identity['data'] = {'path': str(path), 'sha256': cli.sha256_file(path)}
        identity['split_manifest'] = {'path': str(manifest), 'sha256': cli.sha256_file(manifest)}
        data['loader']['max_seq_len'] = 8
        loader = {k: data['loader'][k] for k in ('lookback_weeks', 'max_seq_len')}
        identity['populations'] = {split: qdata._population_view(
            qdata.exact_target_population(frame, target_split=split, **loader)[1])
            for split in ('train', 'validation')}
        statistics = qdata.prepare_dataset_statistics(
            {**data, 'mu_all_train_rows': float(np.log1p(quantities).mean())})
        data['statistics'].update(train_log_mean=statistics['all_train_rows']['log1p_mean'],
            train_log_std=statistics['all_train_rows']['log1p_std'],
            raw_scale=statistics['canonical_train_targets']['raw_scale'],
            all_train_quantity_sha256=statistics['all_train_rows']['quantity_sha256'])
        data['quantity_boundaries_all_train_rows'] = np.quantile(quantities, [.5,.9,.95,.99]).tolist()
        time_stats = cli.derive_train_time_contract(frame, **loader)
        for observed, frozen in (('time_scale','train_time_scale'),
                ('target_log_scaled_mean','train_log_scaled_mean'),
                ('target_log_scaled_std','train_log_scaled_std')):
            data['time_statistics'][frozen] = time_stats[observed]
        for split in ('train','validation'):
            targets = RMTPPWeekLookbackDataset(frame, **loader, val_ratio=.2, mode='all',
                split_col='chronological_split', target_splits={split})
            values = torch.tensor([max(1., float(targets.dt_lists[p][i+1]))
                for p,i in targets.index], dtype=torch.float64)
            data['time_statistics'][f'expected_{split}_targets'] = values.numel()
            data['time_statistics'][f'expected_{split}_target_dt_sha256'] = target_dt_sha256(values)
        data['model'].update(time_scale=time_stats['time_scale'],
            time_initial_location=time_stats['target_log_scaled_mean'],
            time_initial_scale=time_stats['target_log_scaled_std'])
    contract['model_role']='observed_time_local_gate_v1'
    return contract


@pytest.mark.parametrize('dataset_id', ['yellow_trip_hourly','intermittent_frozen_5000','insta_market_basket'])
def test_admitted_data_real_helpers_and_training_metadata(cpu_training, admitted_contract, tmp_path, dataset_id):
    from paper.scripts.count_aware_tpp_backbone import training
    contract = admitted_contract
    data = next(d for d in contract['datasets'] if d['dataset_id'] == dataset_id)
    original = copy.deepcopy(data)
    frame, metadata = runner.prepare_admitted_data(data)
    assert data == original and 'mu_all_train_rows' not in data
    assert set(frame['chronological_split'].unique()) == {'train','validation'}
    assert metadata['held_out_materialized'] is False
    assert metadata['populations'] == data['inherited_data_identity']['populations']
    for key in ('train_log_mean','train_log_std','raw_scale'):
        assert metadata[key] == data['statistics'][key]
    interface = runner.time_interface(data, frame, contract)
    assert interface['train_target_mean'] == metadata['train_log_mean']
    assert interface['train_target_std'] == metadata['train_log_std']
    assert interface['statistics'] == data['statistics']
    # Continue through the real loader, model, optimizer, selector and save path.
    # Only fabricated inputs and the one-epoch CPU workload differ from production.
    for arm in runner.ARMS:
        args = runner.training_args(contract, data, tmp_path/'trained')
        args.device = 'cpu'; args.epochs = args.min_epochs = args.early_stopping_patience = 1
        summary, _, _ = training.train_one(args=args, frame=frame,
            quantity_contract={'boundaries':data['quantity_boundaries_all_train_rows'],
                              'strata':[{'label':f'bin_{i}'} for i in range(5)]},
            interface_meta=interface, backbone=arm, quantity_variant=runner.VARIANT, seed=42)
        assert summary['status'] == 'success' and summary['completed_epochs'] == 1
        assert summary['interface_meta'] == interface


@pytest.mark.parametrize('drift', ['mean','std','scale','quantity_hash','population','legacy_alias','nonfinite'])
def test_admitted_data_keeps_frozen_checks(admitted_contract, drift):
    data = copy.deepcopy(admitted_contract['datasets'][0])
    if drift in ('mean','std','scale'):
        data['statistics'][{'mean':'train_log_mean','std':'train_log_std','scale':'raw_scale'}[drift]] += .01
    elif drift == 'quantity_hash': data['statistics']['all_train_quantity_sha256'] = '0'*64
    elif drift == 'population': data['inherited_data_identity']['populations']['validation']['target_count'] += 1
    elif drift == 'legacy_alias': data['mu_all_train_rows'] = data['statistics']['train_log_mean']+.01
    else: data['statistics']['train_log_mean'] = float('nan')
    with pytest.raises(ValueError):
        runner.prepare_admitted_data(data)


def synthetic(monkeypatch,tmp_path,arm,dataset='intermittent_frozen_5000',name='run'):
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    from paper.scripts.count_aware_tpp_backbone.observed_time import observation_contract
    args=h.args_for(monkeypatch,tmp_path,arm,name)
    args.model_role='observed_time_local_gate_v1';args.dataset_contract=dataset
    meta=h.interface();meta['time_head']['observation_likelihood']=observation_contract(dataset)
    meta.update(source_files_sha256='synthetic_source',execution_contract_sha256='synthetic_contract',
        backbone_design={'schema':'hard_lmm_local_gate_design_v1','design_sha256':'synthetic_design',
            'mode_by_arm':{runner.ARMS[0]:None,runner.ARMS[1]:'time',runner.ARMS[2]:'quantity'},
            'architecture':{'branch_rank':4,'candidate_lags':[1,2,4,8,16,32,64,128]}})
    return args,meta


@pytest.mark.parametrize('dataset',['intermittent_frozen_5000','yellow_trip_hourly','insta_market_basket'])
def test_real_cpu_training_equal_batches_and_earliest_selection(cpu_training,monkeypatch,tmp_path,dataset):
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    traces=[]
    for arm in runner.ARMS:
        trace={'train':[],'validation':[]};h.install_loader(monkeypatch,trace)
        args,meta=synthetic(monkeypatch,tmp_path,arm,dataset)
        summary,_,_=h.run(args,arm,meta=meta)
        payload=h.torch_load_checkpoint(h.directory(args,arm)/'last_epoch_state.pt',map_location='cpu')
        assert summary['completed_epochs']==3 and all(r['train_all_finite'] for r in payload['history'])
        assert summary['best_epoch']==min(payload['history'],key=lambda r:r['val_qty_rmse'])['epoch']
        assert payload['resume_identity']['interface_meta']==meta
        traces.append(trace)
    assert traces[0]==traces[1]==traces[2]


@pytest.mark.parametrize('arm',runner.ARMS[1:])
def test_exact_interrupted_resume_model_optimizer_rng_selector(cpu_training,monkeypatch,tmp_path,arm):
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    h.install_loader(monkeypatch,{'train':[],'validation':[]})
    full,meta=synthetic(monkeypatch,tmp_path,arm,name='full');expected,_,_=h.run(full,arm,meta=meta)
    args,_=synthetic(monkeypatch,tmp_path,arm,name='resumed');save=training.atomic_torch_save
    def interrupt(payload,path):
        save(payload,path)
        if path.name=='last_epoch_state.pt' and payload['epoch']==1:raise RuntimeError('synthetic interruption')
    monkeypatch.setattr(training,'atomic_torch_save',interrupt)
    with pytest.raises(RuntimeError,match='synthetic interruption'):h.run(args,arm,meta=meta)
    monkeypatch.setattr(training,'atomic_torch_save',save)
    actual,_,_=h.run(args,arm,meta=meta)
    assert actual['checkpoint_state_sha256']==expected['checkpoint_state_sha256']
    a=h.torch_load_checkpoint(h.directory(full,arm)/'last_epoch_state.pt',map_location='cpu')
    b=h.torch_load_checkpoint(h.directory(args,arm)/'last_epoch_state.pt',map_location='cpu')
    for key in ('model_state_dict','best_state_dict','optimizer_state_dict','rng_state','train_loader_generator_state','history'):
        h.same(a[key],b[key])


def test_strict_earliest_tie_and_identity_rejection(cpu_training,monkeypatch,tmp_path):
    from paper.scripts.count_aware_tpp_backbone import training
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    h.install_loader(monkeypatch,{'train':[],'validation':[]});evaluate=training.evaluate
    monkeypatch.setattr(training,'evaluate',lambda **kw:{**evaluate(**kw),'qty_rmse':42.})
    arm=runner.ARMS[2];args,meta=synthetic(monkeypatch,tmp_path,arm)
    summary,_,_=h.run(args,arm,meta=meta);assert summary['best_epoch']==1
    changed=copy.deepcopy(meta);changed['source_files_sha256']='foreign'
    monkeypatch.setattr(training,'train_epoch_with_telemetry',lambda *a,**k:pytest.fail('identity failure reached training'))
    with pytest.raises(ValueError):h.run(args,arm,meta=changed)


@pytest.mark.parametrize('arm',runner.ARMS)
def test_selected_and_last_replay_real_cpu_model_accounting(cpu_training,monkeypatch,tmp_path,arm):
    from torch.utils.data import DataLoader
    from paper.scripts import run_taxi_quantity_interface_ablation as loaders
    from simple_lab_test.search.tests import test_observed_time_joint_training as h
    h.install_loader(monkeypatch,{'train':[],'validation':[]});args,meta=synthetic(monkeypatch,tmp_path,arm)
    summary,_,_=h.run(args,arm,meta=meta)
    def loader(_frame,*,target_split,**kwargs):
        assert target_split=='validation';return DataLoader(h.dataset('validation'),batch_size=4,shuffle=False)
    monkeypatch.setattr(loaders,'make_loader',loader)
    data={'model':{'hidden_dim':16,'quantity_variant':runner.VARIANT,'time_head_mode':meta['time_head']['mode'],
        'time_scale':3.,'time_initial_location':.1,'time_initial_scale':.7,'time_sigma_floor':args.time_sigma_floor,
        'time_observation_contract':meta['time_head']['observation_likelihood']},
        'statistics':{'train_log_mean':1.5,'train_log_std':1.},'loader':{'max_seq_len':8,'batch_size':4,'lookback_weeks':520},
        'quantity_boundaries_all_train_rows':[2.,4.,8.,16.],'history_boundaries':[2,4],
        'additional_history_boundaries':[1,2,3,4],
        'inherited_data_identity':{'populations':{'validation':{'target_count':7}}}}
    history=runner.read(h.directory(args,arm)/'history.json')['history']
    for label,path,epoch in [('selected',summary['checkpoint_path'],summary['best_epoch']),
                             ('last',h.directory(args,arm)/'last_epoch_state.pt',3)]:
        payload=h.torch_load_checkpoint(path,map_location='cpu')
        result=runner.replay_checkpoint(path,data,h.grid_frame(),lambda:None,device='cpu',expected_arm=arm,
            expected_identity=payload['resume_identity'],expected_initial=summary['initial_state_sha256'],expected_epoch=epoch)
        runner.audit_replay_accounting(result)
        assert sum(c['count'] for c in result['additional_history_cells'])==7
        for metric,key in [('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll')]:
            assert math.isclose(result[metric],history[epoch-1][key],rel_tol=1e-10,abs_tol=1e-8)
        assert result['count']==7 and result['held_out_test_evaluated'] is False


def test_own_host_can_start_before_other_qualification(authority):
    c,a,p=authority
    q=runner.make_training_permit(c,a,p,{'5080':qualification(c,a,p,'5080')})
    runner.verify_authorization(c,a,q,'5080',training=True)
    with pytest.raises(ValueError,match='This host'):runner.verify_authorization(c,a,q,'5090',training=True)


def test_other_gpu_processes_do_not_abort_owned_worker(monkeypatch,tmp_path):
    c={'limits':{'total_wall_seconds':172800},'hosts':{'5080':{}}}
    p={'started_at_unix':runner.time.time(),'deadline_unix':runner.time.time()+172800}
    monkeypatch.setattr(runner.shared,'fixed_start',lambda *a,**k:float('inf'))
    monkeypatch.setattr(runner.shared,'check_storage',lambda *a:None)
    monkeypatch.setattr(runner.common,'gpu_pids',lambda spec:{999})
    budget=runner.Budget(c,p,'5080',runner.os.getppid(),tmp_path);budget()
    assert runner.read(tmp_path/'gpu_observation.json')['other_pids']==[999]


def test_dependency_still_running_blocks_before_cuda(monkeypatch,tmp_path):
    c={'dependency':{'5090':{'root':str(tmp_path),'contract_sha256':'digest'}}}
    monkeypatch.setattr(runner.common,'sha_json',lambda _: 'digest')
    monkeypatch.setattr(runner,'read',lambda p: {'status':'training'})
    with pytest.raises(ValueError,match='not complete'):runner.verify_dependency(c,'5090')


def test_local_gate_malformed_metadata_rejected():
    from models.TPPs.CountAwareTitanLocalGate import validate_local_gate_checkpoint
    with pytest.raises(ValueError):validate_local_gate_checkpoint({'encoder_config':None},runner.ARMS[1])
