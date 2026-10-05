"""CPU-only operation guards; all run/metric fixtures are synthetic."""
import copy
import ast
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
PROJECT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
FROZEN = PROJECT/'search_artifacts/titantpp_gru_controls_3seed_dual_20261005_v1/source'


def load(name):
    spec=importlib.util.spec_from_file_location('operation_'+name,HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


campaign=load('campaign'); diagnostic=load('diagnostic_adapter'); monitor=load('monitor')


@pytest.fixture(autouse=True)
def no_network_or_processes(monkeypatch):
    def forbidden(*a,**k): raise AssertionError('Real subprocess/network/GPU operation forbidden')
    monkeypatch.setattr(campaign.subprocess,'run',forbidden)
    monkeypatch.setattr(campaign.subprocess,'Popen',forbidden)
    monkeypatch.setattr(campaign.os,'killpg',forbidden)


@pytest.fixture
def contract(tmp_path):
    # Only scientific contract/source identities are read; no original history,
    # endpoint, prediction, held-out, or mixed metric file is opened.
    c=json.loads((FROZEN.parent/'execution_contract.json').read_text())
    c['schema']=campaign.SCHEMA;c['arms']=[campaign.ARM];c['jobs']=campaign.expected_jobs()
    c['canonical_conditions']=9;c['new_fits']=6;c['already_terminal_reused']=3
    c['reuse']=[r for r in c['reuse'] if r['seed'] in campaign.SEEDS]
    c['baseline_replays']={d:{s:r for s,r in v.items() if s in ['52','62']} for d,v in c['baseline_replays'].items()}
    c['anchors42']={d:{'checkpoint':'inputs/synthetic_selected.pt','checkpoint_sha256':'a'*64,
        'last_checkpoint':'inputs/synthetic_last.pt','last_checkpoint_sha256':'b'*64,
        'selected_epoch':9,'last_epoch':49,'initial_state_sha256':'c'*64,
        'source_revision':'d'*40,'metrics':{'qty_rmse':1.,'qty_mae':.5,'time_nll':2.},
        'last_metrics':{'qty_rmse':1.1,'qty_mae':.6,'time_nll':2.1},
        'history_metrics':{'qty_rmse':1.,'qty_mae':.5,'time_nll':2.},
        'last_history_metrics':{'qty_rmse':1.1,'qty_mae':.6,'time_nll':2.1}}
        for d in campaign.FROZEN_DATA_SHA}
    c['architecture']={'arms':[campaign.ARM]};c['design_sha256']=campaign.sha_json(c['architecture'])
    c['operation']={'files':{f'operation/{n}.py':'e'*64 for n in ['campaign','diagnostic_adapter','monitor']}}
    c['operation']['files_sha256']=campaign.sha_json(c['operation']['files'])
    for h,spec in c['hosts'].items():
        root=tmp_path/h;spec['root']=str(root);spec['source_root']=str(root/'source');spec['operation_root']=str(root/'operation')
    return c


def test_import_has_no_execution_side_effects():
    assert campaign.engine is None and campaign.ROOT is None
    assert campaign.expected_jobs()[0]['seed']==52
    assert len(campaign.expected_jobs())==6


def test_accepts_exact_six_fit_contract(contract):
    assert campaign.validate(contract,verify_files=False) is contract


@pytest.mark.parametrize('change',[
    lambda c:c.update(schema='old'),
    lambda c:c['arms'].append('titantpp_gru54'),
    lambda c:c['jobs'][0].update(seed=42),
    lambda c:c['jobs'].reverse(),
    lambda c:c['training'].update(minimum_epochs=0),
    lambda c:c['training'].update(monitor='validation_joint_loss'),
    lambda c:c['training'].update(warm_start=True),
    lambda c:c['limits'].update(automatic_retry=True),
    lambda c:c['limits'].update(workers_per_host=2),
    lambda c:c.update(held_out_test_evaluated=True),
    lambda c:c['source'].update(files_sha256='0'*64),
    lambda c:c['datasets'][0]['optimizer'].update(lr=.01),
    lambda c:c['reuse'].pop(),
    lambda c:c['anchors42'].pop('raf_spare_parts'),
    lambda c:c['anchors42']['raf_spare_parts']['metrics'].update(qty_rmse=float('nan')),
    lambda c:c['operation'].update(files_sha256='0'*64),
    lambda c:c['hosts']['5080'].update(operation_root='/outside'),
    lambda c:c.update(new_fits=9),
])
def test_rejects_scope_or_scientific_drift(contract,change):
    change(contract)
    with pytest.raises(ValueError):campaign.validate(contract,verify_files=False)


def test_frozen_source_and_parent117_are_byte_exact():
    old=PROJECT/'search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1'
    a=json.loads((old/'execution_contract.json').read_text())['source']['files']
    b=json.loads((FROZEN.parent/'execution_contract.json').read_text())['source']['files']
    assert len(a)==117 and len(b)==123
    assert all(b[p]==s for p,s in a.items())
    assert all(campaign.sha_file(FROZEN/p)==s for p,s in b.items())


def test_source_operation_and_input_guard(contract,tmp_path,monkeypatch):
    sr=tmp_path/'source';op=tmp_path/'operation';op.mkdir()
    expected={str((sr/p).resolve()):s for p,s in contract['source']['files'].items()}
    expected.update({str((sr/p).resolve()):s for p,s in contract['input_files'].items()})
    expected.update({str((tmp_path/p).resolve()):s for p,s in contract['operation']['files'].items()})
    monkeypatch.setenv('SOURCE_REVISION',contract['source']['git_revision'])
    monkeypatch.setattr(campaign,'sha_file',lambda p:expected[str(Path(p).resolve())])
    campaign.validate(contract,source_root=sr,operation_root=op)
    expected[str((tmp_path/'operation/monitor.py').resolve())]='0'*64
    with pytest.raises(ValueError,match='Operation drift'):
        campaign.validate(contract,source_root=sr,operation_root=op)


def test_unsafe_manifest_paths_rejected(tmp_path):
    for name in ['../escape','/absolute']:
        with pytest.raises(ValueError):campaign.safe_relative(tmp_path,name)
        with pytest.raises(ValueError):monitor.safe(tmp_path,name)


def test_fit_runtime_changes_only_approved_hash_seed(contract):
    runtime={'environment':{'PYTHONHASHSEED':'42'},'torch':'2.11.0+cu130'}
    q={'status':'passed','runtime':runtime};job=contract['jobs'][0]
    actual=copy.deepcopy(runtime);actual['environment']['PYTHONHASHSEED']='52'
    assert campaign.verify_fit_runtime(contract,'5080',q,job,actual)
    actual['torch']='changed'
    with pytest.raises(ValueError):campaign.verify_fit_runtime(contract,'5080',q,job,actual)
    assert runtime['environment']['PYTHONHASHSEED']=='42'


def test_seed_wrapper_preserves_real_frozen_rng_and_restores_exception():
    import numpy as np
    import torch
    source=FROZEN/'paper/scripts/run_taxi_quantity_interface_ablation.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='set_seed')
    scope={'random':random,'np':np,'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    set_seed=scope['set_seed']
    training=SimpleNamespace(set_seed=set_seed)
    enabled=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        generator=set_seed(52)
        expected=(random.random(),np.random.random(4),torch.rand(4),generator.get_state())
        original=training.set_seed
        before_flags=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
        with pytest.raises(RuntimeError,match='synthetic callback'):
            with campaign.strict_training_seed(training):
                generator=training.set_seed(52)
                actual=(random.random(),np.random.random(4),torch.rand(4),generator.get_state())
                assert actual[0]==expected[0] and np.array_equal(actual[1],expected[1])
                assert torch.equal(actual[2],expected[2]) and torch.equal(actual[3],expected[3])
                assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
                raise RuntimeError('synthetic callback')
        assert training.set_seed is original
        assert before_flags==(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled())
    finally:torch.use_deterministic_algorithms(enabled,warn_only=warn)


def payload():
    return {'backbone':campaign.ARM,'seed':52,'evaluation_scope':'validation_only','held_out_test_evaluated':False,
            'variant':campaign.VARIANT,'checkpoint_monitor':'validation_raw_quantity_rmse',
            'checkpoint_monitor_history_key':'val_qty_rmse','checkpoint_selection':'strict-earliest',
            'interface_meta':{'time_head':{'observation_likelihood':{'mode':'synthetic'}}},'model_state_sha256':'state'}


@pytest.mark.parametrize('mutation',[
    lambda p:p.update(backbone='titantpp_gru54_pair'),lambda p:p.update(seed=99),
    lambda p:p.update(evaluation_scope='test'),lambda p:p.update(held_out_test_evaluated=True),
    lambda p:p.update(checkpoint_selection='last'),lambda p:p.update(model_state_sha256='corrupt')])
def test_diagnostic_payload_rejects_unapproved_checkpoint(mutation):
    p=payload();mutation(p)
    with pytest.raises(ValueError):diagnostic.validate_payload(p,{'model':{'time_observation_contract':{'mode':'synthetic'}}},'state',
        {'history_key':'val_qty_rmse','selection':'strict-earliest'})


def test_diagnostic_supports_cnngru_newseeds_without_broad_route():
    p=payload()
    assert diagnostic.validate_payload(p,{'model':{'time_observation_contract':{'mode':'synthetic'}}},'state',
        {'history_key':'val_qty_rmse','selection':'strict-earliest'})


@pytest.fixture
def snapshot(contract):
    rows=[]
    for job in (j for j in contract['jobs'] if j['host']=='5090'):
        history=[{'epoch':e,'val_qty_rmse':1.+abs(e-9)/100,'val_qty_mae':.5,'val_time_nll':2.} for e in range(1,50)]
        data=next(d for d in contract['datasets'] if d['dataset_id']==job['dataset'])
        n=data['inherited_data_identity']['populations']['validation']['target_count']
        cell=lambda epoch:{'count':n,'qty_rmse':history[epoch-1]['val_qty_rmse'],'qty_mae':.5,'time_nll':2.,
            'evaluation_scope':'validation_only','held_out_test_evaluated':False}
        endpoint={'job':job,'best_epoch':9,'completed_epochs':49,'selected':cell(9),'last':cell(49),
                  'evaluation_scope':'validation_only','held_out_test_evaluated':False}
        rows.append({'job':job,'started':True,'history':{'history':history},'endpoint':endpoint,'actual_saved_epoch':49,
                     'terminal':{'status':'complete','scientific_success':True,'job':job,
                         'contract_sha256':campaign.sha_json(contract),'held_out_test_evaluated':False},'SHA_verified':True})
    return {'host':'5090','contract_sha256':campaign.sha_json(contract),'actual_observed_utc':'2026-10-05T15:00:00+00:00',
            'rows':rows,'processes':[],'gpu_pids':'','server_status':None,'supervisor_exit':{'returncode':0}}


def test_terminal_evidence_accepts_full_bound_result(contract,snapshot):
    assert monitor.analyze(contract,snapshot)['counts']['complete']==2


@pytest.mark.parametrize('change',[
    lambda r:r.update(SHA_verified=False),lambda r:r.update(actual_saved_epoch=48),
    lambda r:r['terminal'].update(scientific_success=False),lambda r:r['terminal'].update(contract_sha256='foreign'),
    lambda r:r['endpoint'].update(best_epoch=10),lambda r:r['endpoint'].update(held_out_test_evaluated=True),
    lambda r:r['endpoint']['last'].update(count=1),lambda r:r['endpoint']['selected'].update(qty_mae=100),
    lambda r:r['history']['history'][1].update(epoch=3),
    lambda r:r['history']['history'][4].update(val_qty_rmse=.9),
])
def test_terminal_corruption_is_not_completion(contract,snapshot,change):
    change(snapshot['rows'][0]);result=monitor.analyze(contract,snapshot)
    assert result['counts']['complete']==1 and result['counts']['uncertain']==1


def test_owned_pid_evidence_and_failure_no_retry(contract,snapshot):
    row=snapshot['rows'][0];row['terminal']=None
    snapshot['server_status']={'active_job':row['job'],'status':'running','supervisor_pid':100,'worker_group_pid':101}
    snapshot['processes']=['100 1 python /operation/campaign.py --mode dispatch --host 5090',
                           '102 101 python /operation/campaign.py --mode fit --host 5090 --job '+row['job']['id'],
                           '101 100 timeout --signal=TERM /operation/campaign.py --mode fit --host 5090']
    snapshot['gpu_uuid_verified']=True
    snapshot['gpu_pids']=contract['hosts']['5090']['gpu_uuid']+', 102, python'
    assert monitor.analyze(contract,snapshot)['counts']['running']==1
    snapshot['gpu_pids']=''
    assert monitor.analyze(contract,snapshot)['counts']['uncertain']==1
    snapshot['failure']={'message':'synthetic stop'}
    result=monitor.analyze(contract,snapshot)
    assert result['counts']['failed']==1 and result['automatic_retry'] is False


def test_failure_sentinel_blocks_authorization_before_permit_reads(contract,tmp_path,monkeypatch):
    (tmp_path/'failure.json').write_text('{}')
    monkeypatch.setattr(campaign,'validate',lambda c:c)
    with pytest.raises(ValueError,match='Recorded failure'):campaign.authorization(contract,tmp_path)


def test_anchor_identity_rejects_other_arm_and_epoch(contract):
    ref=contract['anchors42']['raf_spare_parts'];p=payload()
    p.update(seed=42,epoch=9,initial_state_sha256=ref['initial_state_sha256'],source_revision=ref['source_revision'],
             source_revision_history=[ref['source_revision']],resume_identity={'synthetic':True})
    assert campaign.verify_anchor_payload(p,ref,'selected')
    p['epoch']=10
    with pytest.raises(ValueError):campaign.verify_anchor_payload(p,ref,'selected')


def test_native_reset_probe_on_cpu_frozen_cnngru(monkeypatch):
    sys.path.insert(0,str(FROZEN))
    from models.TPPs.CountAwareTitanCNNGRU import CountAwareTitanCNNGRU
    def builder(data,arm):
        return CountAwareTitanCNNGRU(candidate_arm=arm,train_log_mean=data['statistics']['train_log_mean'],
            train_log_std=data['statistics']['train_log_std'],max_seq_len=data['loader']['max_seq_len'],
            **data['model']),{}
    engine=SimpleNamespace(build_model=builder)
    checks=SimpleNamespace(synthetic_data=lambda length:{'model':{'hidden_dim':64,
        'quantity_variant':'count_only_log_regression',
        'time_head_mode':'heteroscedastic_lognormal_duration',
        'time_observation_contract':{'mode':'positive_integer_round_clamp_v1','unit':'week','top_code':None}},
        'statistics':{'train_log_mean':1.2,'train_log_std':.8},'loader':{'max_seq_len':length}})
    monkeypatch.setattr(campaign,'checks',checks)
    result=campaign.reset_check(engine,lambda:None,device='cpu')
    assert result['withheld_valid_resets'] and result['padding_skips_state']
    assert result['held_out_test_evaluated'] is False


def test_monitor_current_host_drift_blocks_query(contract,tmp_path,monkeypatch):
    (tmp_path/'execution_contract.json').write_text(json.dumps(contract))
    current={'canonical_sha256':monitor.digest(contract),'source_closure_sha256':contract['source']['files_sha256'],
             'operation_closure_sha256':contract['operation']['files_sha256'],
             'hosts':{h:v['root'] for h,v in contract['hosts'].items()}}
    current['hosts']['5080']='/oldroot'
    (tmp_path/'current.json').write_text(json.dumps(current))
    monkeypatch.setattr(sys,'argv',['monitor.py','--bundle',str(tmp_path),'--host','5080'])
    with pytest.raises(ValueError,match='no old-root query'):monitor.main()


def test_storage_budget_remains_periodic_under_frequent_pulses(tmp_path,monkeypatch):
    clock=[100.];size=[0]
    monkeypatch.setattr(campaign.time,'time',lambda:clock[0])
    monkeypatch.setattr(campaign.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda p:SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(campaign,'common',SimpleNamespace(write_json=lambda *a,**k:None))
    original=Path.read_text
    monkeypatch.setattr(Path,'read_text',lambda p,*a,**k:'VmRSS: 100 kB\n' if str(p)=='/proc/self/status' else original(p,*a,**k))
    file=SimpleNamespace(is_file=lambda:True,stat=lambda:SimpleNamespace(st_size=size[0]))
    monkeypatch.setattr(Path,'rglob',lambda p,pattern:[file])
    budget=campaign.PulseBudget(tmp_path,'digest','job',1000.)
    budget()
    for t in range(102,132,2):clock[0]=float(t);budget()
    size[0]=17*1024**3;clock[0]=132.
    with pytest.raises(ValueError,match='Owned-root bytes'):budget()


def test_full_validation_guard_rejects_partial_population():
    data={'inherited_data_identity':{'populations':{'validation':{'target_count':100}}}}
    endpoint={'evaluation_scope':'validation_only','held_out_test_evaluated':False,'count':100}
    assert campaign.verify_full_validation(endpoint,data)
    endpoint['count']=99
    with pytest.raises(ValueError,match='full original'):campaign.verify_full_validation(endpoint,data)
