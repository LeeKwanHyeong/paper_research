"""Independent host-only transfer review; no remote or real-data execution."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import hashlib
import types
import pytest
from paper.scripts import titantpp_shared_history_raw9_5090_runtime as r
from paper.scripts import manage_titantpp_shared_history_raw9_5090 as m
from paper.scripts import run_titantpp_shared_history_raw9_5090 as worker
from tests.test_titantpp_shared_history_raw9_independent_qa import fixture_contract


def contract():
    c=fixture_contract();c['schema']=r.SCHEMA
    h=c['hosts'].pop('5080');c['hosts']['5090']=h
    h['root']='/tmp/independent_raw9_5090';h['source_root']=h['root']+'/source'
    h['gpu_uuid']=r._GPU_UUID;h['paused_processes']=[];h['absolute_host_deadline_unix']=1792888885.815062
    h['runtime_expected']={'python':'3.12.13','torch':'2.11.0+cu130','cuda':'13.0','cudnn':92000,'threads':4,'interop_threads':1}
    c['maximum_parallel_workers']=3
    for j in c['jobs']:j['host']='5090'
    return c


def test_new5090_contract_has_identical_scientific_configuration():
    c=contract();assert len(r.validate_contract(c,'5090'))==9
    parent=fixture_contract()
    for name in ('datasets','training','common_condition','architecture','baseline_validation_references','input_files'):
        assert c[name]==parent[name]
    for j in c['jobs']:worker.select_job(c,'5090',j['id'])
    assert r.STAGES==('train','validation')


@pytest.mark.parametrize('case',['host','alpha0','cap','gpu','cudnn','paused','parallel','deadline','baseline','test'])
def test_changed_host_science_runtime_or_limits_rejected(case):
    c=contract()
    if case=='host':c['jobs'][0]['host']='5080'
    elif case=='alpha0':c['common_condition']['time_alpha']=0.
    elif case=='cap':c['quantity_capping_applied']=True
    elif case=='gpu':c['hosts']['5090']['gpu_uuid']='other'
    elif case=='cudnn':c['hosts']['5090']['runtime_expected']['cudnn']=91900
    elif case=='paused':c['hosts']['5090']['paused_processes']=[{'pid':9,'state':'T'}]
    elif case=='parallel':c['maximum_parallel_workers']=4
    elif case=='deadline':c['hosts']['5090']['absolute_host_deadline_unix']+=1
    elif case=='baseline':c['jobs'][0]['arm']='titantpp_history_mlp_width16'
    elif case=='test':c['held_out_test_authorized']=True
    with pytest.raises(ValueError):r.validate_contract(c,'5090')


def test_fresh_idle_gate_never_reads_cap5080_files():
    c=contract()
    with patch.object(r._impl,'gpu_snapshot',return_value={'compute_pids':[]}) as gpu,patch.object(r._raw,'read',side_effect=AssertionError('No Cap receipt read')),patch.object(r._impl,'read',side_effect=AssertionError('No Cap receipt read')):
        out=r.predecessor_gate(c,'5090',require_complete=True)
        assert out['cap5080_completion_required'] is False and out['fresh_idle_verified'] is True
        gpu.assert_called_once_with(r._GPU_UUID)


def test_fresh_gpu_unowned_pid_and_active_allowlist():
    c=contract()
    with patch.object(r._impl,'gpu_snapshot',return_value={'compute_pids':[123]}):
        with pytest.raises(ValueError):r.predecessor_gate(c,'5090')
        assert r.gpu_state(c,'5090',[123])[0]['compute_pids']==[123]
        with pytest.raises(ValueError):r.gpu_state(c,'5090',[456])


def test_manager_modes_do_not_offer_cap_wait_queue_or5080():
    with patch.object(m,'verify_local',side_effect=AssertionError('No local operation')):
        for mode in ('wait-predecessor','start-queue','resume'):
            with pytest.raises(SystemExit):m.main(['--mode',mode])
    assert m.BUNDLE.name==m.REPORT.name=='execution_5090_v1'
    assert m.runtime is r
    assert r._impl.WORKER_MODULE.endswith('_5090')
    assert r._impl.predecessor_gate is r.predecessor_gate


def test_collection_program_routes5090_runtime_only():
    c=contract();source=m.collection_program(c,'5090')
    compile(source,'collection-only-compile','exec')
    assert 'import titantpp_shared_history_raw9_5090_runtime as r' in source
    assert 'import titantpp_shared_history_raw9_runtime as r' not in source
    assert '5080' not in source


def test_frozen_worker_and_adapters_preserve_parent_source():
    assert hashlib.sha256(worker._FROZEN_WORKER_PATH.read_bytes()).hexdigest()==worker.FROZEN_WORKER_SHA256
    assert hashlib.sha256(r._PATH.read_bytes()).hexdigest()==r._SHA
    assert hashlib.sha256(m._PATH.read_bytes()).hexdigest()==m._SHA
    actual=worker._adapted_source
    original=worker._frozen_source.decode()
    restored=actual.replace('titantpp_shared_history_raw9_5090_stage_v1','titantpp_shared_history_raw9_stage_v1').replace('titantpp_shared_history_raw9_5090_v1','titantpp_shared_history_raw9_v1').replace('5090','5080')
    assert restored==original


def test_5090_aggregation_preserves_frozen_arithmetic():
    from paper.scripts import compare_titantpp_shared_history_raw9_5090 as new
    from paper.scripts import compare_titantpp_shared_history_raw9 as old
    from tests.test_titantpp_shared_history_raw9_independent_qa import fabricated_records
    rows=fabricated_records()
    for row in rows:
        if row['model']==new.SHARED:row['host']='5090'
    actual,expected=new.aggregate(rows),old.aggregate(rows)
    actual.pop('interpretation_limits');expected.pop('interpretation_limits')
    assert actual==expected


@pytest.mark.parametrize('mutation',['notcancelled','pidpresent','priorremote','priorfit','host5080','capchanged','gpuactive','wrongruntime','oldpidpresent'])
def test_migration_rejects_unconfirmed_cancel_or_idleproof(monkeypatch,mutation):
    from paper.scripts import prepare_titantpp_shared_history_raw9_5090 as p
    cancel={'status':'cancelled_before_any_scientific_or_remote_attempt','observed_pid_absence':True,'raw9_remote_attempt_count':0,'raw9_fit_count':0,'replacement_host':'5090','Cap_training_modified':False}
    runtime=contract()['hosts']['5090']['runtime_expected']
    initial={'host':'5090','gpu':{'actual_exit':0},'gpu_processes':{'actual_exit':0,'stdout':''},'new_root_exists':False,'runtime':deepcopy(runtime),'old_paused_process_paths_present':{'999':False}}
    if mutation=='notcancelled':cancel['status']='running'
    elif mutation=='pidpresent':cancel['observed_pid_absence']=False
    elif mutation=='priorremote':cancel['raw9_remote_attempt_count']=1
    elif mutation=='priorfit':cancel['raw9_fit_count']=1
    elif mutation=='host5080':cancel['replacement_host']='5080'
    elif mutation=='capchanged':cancel['Cap_training_modified']=True
    elif mutation=='gpuactive':initial['gpu_processes']['stdout']='GPU,999'
    elif mutation=='wrongruntime':initial['runtime']['cudnn']=91900
    elif mutation=='oldpidpresent':initial['old_paused_process_paths_present']['999']=True
    def read(path):
        if path==p.CANCELLATION:return cancel
        if path==p.HOST_PARENT:return {'hosts':{'5090':{'runtime_expected':runtime}}}
        return initial
    with patch.object(p,'read',side_effect=read),patch.object(Path,'exists',return_value=False):
        with pytest.raises(ValueError):p.migration_gate()
