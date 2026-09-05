"""Frozen e300 gates, first legal stop, fail-closed launch and saved-state audit."""
from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
from unittest.mock import Mock

import pytest
import torch

from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts import run_hard_lmm_elapsed_age_screening as screening
from paper.scripts import run_hard_lmm_elapsed_age_smoke as smoke
from simple_lab_test.search.tests.test_hard_lmm_elapsed_age_smoke import create_synthetic_e1


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('name', screening.DATASETS)
def test_command_changes_only_e300_budget_and_execution_role(monkeypatch, tmp_path, name):
    spec, rows = screening.frozen_documents()
    row = next(row for row in rows if row['dataset'] == name)
    before = smoke.command(row, tmp_path, tmp_path/'new', 'a'*40)
    after = screening.command(row, tmp_path, tmp_path/'new', 'a'*40)
    assert len(before) == len(after)
    assert {before[i-1] for i, (a,b) in enumerate(zip(before,after)) if a != b} == {
        '--epochs', '--min-epochs', '--early-stopping-patience', '--execution-role'}
    monkeypatch.setattr(sys, 'argv', ['test']+after[4:])
    args = entry.parse_args()
    assert (args.epochs,args.min_epochs,args.early_stopping_patience) == (300,40,40)
    assert args.seeds == '42' and args.device == 'cuda' and args.backbones == screening.BACKBONE
    assert args.execution_role == screening.EXECUTION_ROLE and not args.force_rerun
    assert args.max_series is args.max_train_batches is args.max_val_batches is None
    assert spec['performance_training_authorized'] and not spec['held_out_test']


def pinned():
    spec, _ = screening.frozen_documents()
    model = screening.read(screening.ROOT/screening.MODEL_CONTRACT_PATH)
    refs = screening.read(screening.ROOT/model['references']['separate_key_validation_comparison'])['datasets']
    return spec, refs


@pytest.mark.parametrize('name', screening.DATASETS)
def test_frozen_reference_precision_and_inclusive_simultaneous_bound(name):
    spec, refs = pinned(); ref = refs[name]
    example = screening.compare(name, ref['candidate'], ref['baseline'], ref['candidate'], spec['performance_gates'])
    candidate = ref['candidate'] | example['upper_bounds']['simultaneous']
    result = screening.compare(name, candidate, ref['baseline'], ref['candidate'], spec['performance_gates'])
    assert result['passed']
    for metric, bound in result['upper_bounds']['simultaneous'].items():
        beyond = candidate | {metric:math.nextafter(bound, math.inf)}
        assert not screening.compare(name, beyond, ref['baseline'], ref['candidate'], spec['performance_gates'])['passed']
    if name == 'yellow_trip_hourly':
        assert result['upper_bounds']['simultaneous']['body_mae'] == ref['candidate']['body_mae']*1.01
        assert result['upper_bounds']['simultaneous']['time_nll'] == ref['candidate']['time_nll']+.01
    else:
        assert result['upper_bounds']['simultaneous']['body_mae'] == ref['candidate']['body_mae']*.95
        assert result['upper_bounds']['simultaneous']['time_nll'] == ref['baseline']['time_nll']+.01
        assert result['upper_bounds']['simultaneous']['gt_p99_mae'] == ref['baseline']['gt_p99_mae']*1.02


@pytest.mark.parametrize('invalid', [float('nan'),float('inf'),-1.])
def test_invalid_error_metrics_cannot_pass(invalid):
    spec, refs = pinned(); ref = refs[screening.DATASETS[0]]
    with pytest.raises(ValueError):
        screening.compare(screening.DATASETS[0],ref['candidate']|{'body_mae':invalid},
                          ref['baseline'],ref['candidate'],spec['performance_gates'])


def test_train_dependency_check_covers_indirect_entry_and_audit_helpers():
    names = ['paper/scripts/run_intermittent_log_backbone_control.py',
             'paper/scripts/run_taxi_log_backbone_control.py',
             'paper/scripts/run_hard_lmm_elapsed_age_smoke.py',
             'paper/scripts/run_hard_lmm_weighted_static.py', 'models/Titan/backbone.py']
    files = dict.fromkeys(names,'sha') | {'paper/scripts/run_hard_lmm_elapsed_age_screening.py':'new'}
    assert set(screening.training_files({'files':files})) == set(names)
    assert screening.training_files({'files':files}) != screening.training_files({'files':files|{names[0]:'drift'}})


@pytest.mark.parametrize('failure', [None,'smoke','preflight','training','audit'])
def test_execute_is_fresh_sequential_and_fail_closed(monkeypatch,tmp_path,failure):
    spec,rows = screening.frozen_documents()
    monkeypatch.setattr(screening,'verify_source',lambda *a,**k:'manifest')
    monkeypatch.setattr(smoke,'runtime_metadata',lambda *a:spec['runtime'])
    monkeypatch.setattr(smoke,'baseline',lambda *a:({},{}))
    actual_read=screening.read
    monkeypatch.setattr(screening,'read',lambda p:{} if p == screening.ROOT/'source_manifest.json' or p.name=='summary.json' else actual_read(p))
    calls=[]
    def gate(*args):
        calls.append('smoke')
        if failure=='smoke':raise ValueError('Smoke gate failed')
        return {'status':'passed'}
    def preflight(*args):
        calls.append('preflight')
        if failure=='preflight':raise ValueError('Busy GPU')
        return {}
    def run(cmd,path):
        calls.append(path.name)
        if failure=='training':raise subprocess.CalledProcessError(1,cmd)
    def audit(*args):
        calls.append('audit')
        if failure=='audit':raise ValueError('Audit failed')
        return {'status':'passed','dataset':args[1]['dataset']}
    monkeypatch.setattr(screening,'verify_smoke',gate)
    monkeypatch.setattr(screening,'preflight',preflight)
    monkeypatch.setattr(screening,'run_logged',run)
    monkeypatch.setattr(screening,'audit_run',audit)
    # A failed performance gate is an outcome, not permission to stop/retune/retry.
    monkeypatch.setattr(screening,'compare_run',lambda *a:{'passed':False})
    args=argparse.Namespace(project_root=tmp_path,output_root=tmp_path/'new',source_revision='a'*40,audit_only=False)
    if failure:
        with pytest.raises((ValueError,subprocess.CalledProcessError)):screening.execute(args)
    else:
        result=screening.execute(args)
        assert len(result['completed'])==2 and all(not item['gate_passed'] for item in result['completed'])
    status=actual_read(args.output_root/'status.json')
    assert status['status']==('failed' if failure else 'complete')
    if failure:assert screening.DATASETS[1]+'.log' not in calls
    before=(args.output_root/'status.json').read_bytes()
    with pytest.raises(FileExistsError):screening.execute(args)
    assert before==(args.output_root/'status.json').read_bytes()


def synthetic_screening(monkeypatch,tmp_path):
    # One real CPU step creates tensors. e41 metadata below is an audit fixture,
    # never a claim that these are real e300 training or performance results.
    output,row,launch,reference=create_synthetic_e1(monkeypatch,tmp_path)
    run=(output/smoke.SUMMARY).parent
    summary=smoke.read(output/smoke.SUMMARY)
    old=smoke.read(run/'history.json')['history'][0]
    history=[dict(old,epoch=i,val_joint_objective=old['val_joint_objective']+i-1) for i in range(1,42)]
    launch.update(epochs=300,execution_role=screening.EXECUTION_ROLE)
    launch['early_stopping'].update(min_epochs=40,patience=40)
    summary.update(epochs=300,completed_epochs=41,stopped_early=True)
    smoke.save(output/'launch_contract.json',launch)
    smoke.save(output/smoke.SUMMARY,summary)
    smoke.save(run/'history.json',{'history':history})
    last=torch.load(run/'last_epoch_state.pt',map_location='cpu',weights_only=False)
    last.update(epoch=41,history=history)
    last['model_state_dict'][smoke.BETA_KEY] += .01
    for state in last['optimizer_state_dict']['state'].values():state['step'].fill_(41)
    torch.save(last,run/'last_epoch_state.pt')
    return output,row,launch,reference,last


def test_saved_state_audit_checks_best_last_optimizer_steps_and_first_legal_stop(monkeypatch,tmp_path):
    output,row,launch,reference,last=synthetic_screening(monkeypatch,tmp_path)
    args=(output,row,(launch,reference),'a'*40)
    result=screening.audit_run(*args,require_cuda=False)
    assert result['status']=='passed' and result['completed_epochs']==41 and result['best_epoch']==1
    assert result['beta_optimizer_steps']==41 and result['beta_values']!=result['last_beta_values']
    assert result['checkpoint_state_sha256']!=result['last_checkpoint_state_sha256']
    run=(output/smoke.SUMMARY).parent
    original=copy.deepcopy(last)
    beta_state=next(state for state in last['optimizer_state_dict']['state'].values() if tuple(state['exp_avg'].shape)==(2,4))
    beta_state['step'].fill_(1);torch.save(last,run/'last_epoch_state.pt')
    with pytest.raises(ValueError,match='Beta optimizer step'):screening.audit_run(*args,require_cuda=False)
    last=copy.deepcopy(original);last['best_state_dict'][smoke.BETA_KEY]+=.1;torch.save(last,run/'last_epoch_state.pt')
    with pytest.raises(ValueError,match='digest'):screening.audit_run(*args,require_cuda=False)
    torch.save(original,run/'last_epoch_state.pt')
    history=copy.deepcopy(original['history']);history[1]['train_batch_count']=2
    smoke.save(run/'history.json',{'history':history})
    with pytest.raises(ValueError,match='Partial train batch'):screening.audit_run(*args,require_cuda=False)


def test_earliest_tied_best_and_premature_termination_are_rejected():
    h=[{'epoch':i,'val_joint_objective':1.,'train_event_count':38,'train_all_finite':True} for i in range(1,42)]
    s=dict(epochs=300,completed_epochs=41,best_epoch=1,stopped_early=True)
    assert screening.validate_history(h,s,{'train_targets':38},screening=True)['epoch']==1
    with pytest.raises(ValueError,match='selection'):
        screening.validate_history(h,s|{'best_epoch':2},{'train_targets':38},screening=True)
    with pytest.raises(ValueError,match='Premature'):
        screening.validate_history(h[:-1],s|{'completed_epochs':40},{'train_targets':38},screening=True)


def test_screening_source_requires_authorized_complete_hash_verified_manifest(tmp_path):
    spec,rows=screening.frozen_documents()
    names={screening.CONTRACT,screening.MODEL_CONTRACT_PATH,'models/Titan/common/elapsed_age.py',
           'paper/scripts/run_hard_lmm_elapsed_age_screening.py',*spec['frozen_files']}
    files={name:screening.digest(screening.ROOT/name) for name in names}
    for row in rows:
        for name,(_,sha) in smoke.reference_files(row).items():files[f'references/{row["dataset"]}/{name}']=sha
    manifest={'source_revision':'a'*40,'scope':spec['scope'],'performance_training_authorized':True,'files':files}
    path=tmp_path/'source_manifest.json';screening.save(path,manifest)
    assert screening.verify_source('a'*40,path)==screening.digest(path)
    screening.save(path,manifest|{'performance_training_authorized':False})
    with pytest.raises(ValueError,match='not authorized'):screening.verify_source('a'*40,path)
    screening.save(path,manifest|{'files':{}})
    with pytest.raises(ValueError,match='Incomplete'):screening.verify_source('a'*40,path)
    screening.save(path,manifest|{'files':files|{screening.CONTRACT:'drift'}})
    with pytest.raises(ValueError,match='Source/reference mismatch'):screening.verify_source('a'*40,path)


def test_package_includes_frozen_cuda_xml_and_only_committed_source(monkeypatch,tmp_path):
    import io
    import json
    import tarfile
    from paper.scripts import package_hard_lmm_elapsed_age_screening as packager
    spec,_=screening.frozen_documents()
    names=set(packager.REQUIRED_SOURCE)|set(spec['frozen_files'])
    def archive(omit=None):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as target:
            for name in sorted(names-({omit} if omit else set())):
                data=(screening.ROOT/name).read_bytes();member=tarfile.TarInfo(name);member.size=len(data)
                target.addfile(member,io.BytesIO(data))
        return stream.getvalue()
    source=archive()
    monkeypatch.setattr(packager.subprocess,'check_output',lambda cmd,**kw:{'rev-parse':'a'*40,'diff':b'','archive':source}[cmd[1]])
    output=tmp_path/'source.tar.gz';packager.package(output)
    with tarfile.open(output) as package:
        manifest=json.load(package.extractfile('source_manifest.json'))
        assert manifest['performance_training_authorized'] and manifest['scope']==spec['scope']
        assert package.getmember('sample_data').isdir()
        assert any(name.endswith('cuda_contract_tests.xml') for name in manifest['files'])
        assert not any(name.startswith('scripts/') for name in manifest['files'])
        for name,sha in manifest['files'].items():
            import hashlib
            assert hashlib.sha256(package.extractfile(name).read()).hexdigest()==sha
    with pytest.raises(FileExistsError):packager.package(output)
    source=archive('models/Titan/common/elapsed_age.py')
    with pytest.raises(ValueError,match='not committed'):packager.package(tmp_path/'missing.tar.gz')
