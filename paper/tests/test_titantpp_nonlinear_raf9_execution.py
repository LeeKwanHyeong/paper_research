"""Synthetic protocol tests; no research rows, checkpoints or remote calls."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import pytest
from paper.scripts import titantpp_nonlinear_raf9_runtime as runtime
from paper.scripts import advance_titantpp_nonlinear_raf9 as followup
from paper.scripts import prepare_titantpp_nonlinear_raf9_execution as prep
from paper.scripts import deploy_titantpp_nonlinear_raf9 as deploy


def contract():
    remote='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+runtime.SCHEMA+'_5090'
    return {'schema':runtime.SCHEMA,'jobs':runtime.expected_jobs(),'arms':list(runtime.ARMS),
        'training':deepcopy(runtime.TRAINING),'parallelism':deepcopy(runtime.PARALLELISM),
        'evaluation_scope':'validation_only','held_out_test_evaluated':False,'test_evaluation':False,
        'limits':{'total_wall_seconds':168*3600,'per_condition_seconds':36*3600,'automatic_retry':False},
        'hosts':{'5090':{'root':remote,'source_root':remote+'/source','operation_root':remote+'/operation',
            'gpu_uuid':'GPU-c9adc246-ef66-7906-bc07-6e152fa9952f',
            'python':'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12'}},
        'source':{'files':{},'files_sha256':runtime.sha_json({})},
        'comparison_runtime_limitation':'all three fresh arms on one runtime',
        'primary_contrast':'Joint−Channel'}


def test_nine_unique_fresh_jobs_same_runtime_three_waves():
    c=contract();runtime.validate(c,verify_files=False)
    assert len(c['jobs'])==len({j['id'] for j in c['jobs']})==9
    assert c['parallelism']['maximum_workers']==3
    for arm in runtime.ARMS:
        assert {j['seed'] for j in c['jobs'] if j['arm']==arm}=={42,52,62}
    assert {j['dataset'] for j in c['jobs']}=={'raf_spare_parts'}
    assert {j['host'] for j in c['jobs']}=={'5090'}


@pytest.mark.parametrize('change',['gpu','root','jobs','budget','test','concurrency','retry'])
def test_contract_scope_drift_blocks(change):
    c=contract()
    if change=='gpu':c['hosts']['5090']['gpu_uuid']='foreign'
    elif change=='root':c['hosts']['5090']['root']+='_old'
    elif change=='jobs':c['jobs'][0]['seed']=7
    elif change=='budget':c['training']['maximum_epochs']=301
    elif change=='test':c['test_evaluation']=True
    elif change=='concurrency':c['parallelism']['maximum_workers']=4
    elif change=='retry':c['limits']['automatic_retry']=True
    with pytest.raises(ValueError):runtime.validate(c,verify_files=False)


def test_actual_admission_required_for_training_permit(tmp_path):
    c=contract();q={'status':'passed','contract_sha256':runtime.sha_json(c),
        'capacity_check':{'maximum_workers':3,'passed':True,'simultaneous_synthetic_workers_executed':False}}
    runtime.write_json(tmp_path/'qualification/receipt.json',q,exclusive=True)
    runtime.write_json(tmp_path/'qualification/actual_exit.json',{'returncode':0},exclusive=True)
    runtime.write_json(tmp_path/'start_permit.json',{'fixture':True},exclusive=True)
    runtime.write_json(tmp_path/'training_permit.json',{'contract_sha256':runtime.sha_json(c),'qualification':q,
        'qualification_sha256':runtime.sha_file(tmp_path/'qualification/receipt.json'),
        'start_permit_sha256':runtime.sha_json({'fixture':True})},exclusive=True)
    with pytest.raises(ValueError,match='GPU capacity missing'):runtime.verify_training_permit(c,tmp_path)


def test_foreign_gpu_worker_rejected():
    active={'job':{'child':SimpleNamespace(pid=20),'command':['timeout','--signal=TERM','--kill-after=15s','40','python','mine']}}
    with pytest.raises(ValueError,match='Foreign'):
        runtime.verify_gpu_membership({30},{30:{'ppid':99,'pgrp':99,'argv':['python','foreign']}},active,script='mine',used_bytes=1,total_bytes=10)


def test_exclusive_write_and_preserved_failure(tmp_path):
    prep.write(tmp_path/'failure.json',{'original':True})
    before=(tmp_path/'failure.json').read_bytes()
    with pytest.raises(FileExistsError):prep.write(tmp_path/'failure.json',{'original':False})
    with patch.object(runtime,'validate',return_value=None),pytest.raises(ValueError,match='no automatic retry'):
        runtime.authorization({},tmp_path)
    assert (tmp_path/'failure.json').read_bytes()==before


def test_remote_programs_compile_and_are_new_scope_only():
    compile(deploy.REMOTE_CODE,'deployment','exec')
    compile(followup.REMOTE_CODE,'observation','exec')
    assert 'scientific_worker_count\')==9' in followup.REMOTE_CODE
    assert 'supervisor_wrapper_own_os_exit_directly_observed' in followup.REMOTE_CODE
    assert 'titantpp_backbone_nonlinear_raf9_20261015_v1' in followup.REMOTE


def test_nine_selected_models_use_paired_same_host_baseline(tmp_path):
    c=contract();endpoint={'count':6690,'qty_rmse':4.,'qty_mae':2.,'time_nll':1.,
        'tail':{'qty_rmse':8.,'qty_mae':5.,'time_nll':3.}}
    for j in c['jobs']:
        folder=tmp_path/'originals/run'/j['id']
        prefix=folder/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"
        followup.write(prefix/'endpoint_replays.json',{'best_epoch':2,'selected':endpoint})
        followup.write(prefix/'history.json',{'history':[{'epoch':1,'val_qty_rmse':5.},
            {'epoch':2,'val_qty_rmse':4.},{'epoch':3,'val_qty_rmse':4.}]})
        followup.write(folder/'terminal_manifest.json',{'job':j,'held_out_test_evaluated':False,
            'files':{str(p.relative_to(folder)):followup.sha(p) for p in prefix.iterdir()}})
    report=tmp_path/'report';report.mkdir()
    with patch.object(followup,'REPORT',report):result=followup.analyze(c,tmp_path/'originals')
    assert result['new_fit_count']==9 and len(result['rows'])==9 and len(result['seed_differences'])==9
    assert set(result['aggregate'])=={'CNN','Channel','Joint'}
    assert all(r['best_epoch']==2 for r in result['rows'])
    assert all(d['RMSE']['absolute']==0 for d in result['seed_differences'])


def test_eta_uses_only_current_fit_own_finite_completed_timings():
    snapshot={'observed_unix':100,'all_training_terminal_verified':False,'failure':False,
        'jobs':[{'job':{'arm':'a','seed':42},'state':'running',
            'history':{'history':[{'epoch':1,'val_qty_rmse':8.},{'epoch':2,'val_qty_rmse':7.}]},
            'timing':{'epochs':[{'elapsed_seconds':float('nan')},{'elapsed_seconds':0},
                {'elapsed_seconds':2.},{'elapsed_seconds':4.}]}}]}
    row=followup.brief_round(snapshot)['rows'][0]
    assert row['timing_samples']==2 and row['conditional_stop_epoch']==42
    assert row['conditional_stop_seconds']==120 and row['epoch300_seconds']==894
