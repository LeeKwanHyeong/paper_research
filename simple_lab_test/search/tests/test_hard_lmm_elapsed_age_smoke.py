"""Fresh e1 boundaries, CUDA evidence, immutable packaging and real-format audit."""
from __future__ import annotations

import argparse
import copy
import io
import json
import subprocess
import sys
import tarfile
from unittest.mock import Mock

import polars as pl
import pytest
import torch

from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts import run_hard_lmm_elapsed_age_smoke as smoke
from paper.scripts import package_hard_lmm_elapsed_age_smoke as packager


def xml(path, *, skipped=False, omit_module=False):
    modules = smoke.CUDA_TESTS[:1] if omit_module else smoke.CUDA_TESTS
    cases = ''.join(f'<testcase classname="{name[:-3].replace(chr(47), chr(46))}" name="contract">'
                    f'{"<skipped/>" if skipped and index == 0 else ""}</testcase>'
                    for index, name in enumerate(modules))
    path.write_text(f'<testsuites><testsuite tests="{len(modules)}" failures="0" errors="0" '
                    f'skipped="{int(skipped)}">{cases}</testsuite></testsuites>')


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('name', smoke.DATASETS)
def test_command_cannot_expand_e1_or_select_partial_data(monkeypatch, tmp_path, name):
    spec, rows = smoke.frozen_documents()
    row = next(row for row in rows if row['dataset'] == name)
    cmd = smoke.command(row, tmp_path, tmp_path/'fresh', 'a'*40)
    monkeypatch.setattr(sys, 'argv', ['test']+cmd[4:])
    args = entry.parse_args()
    assert args.epochs == args.min_epochs == args.early_stopping_patience == 1
    assert args.device == 'cuda' and args.model_role == smoke.ROLE and args.backbones == smoke.BACKBONE
    assert args.execution_role == smoke.EXECUTION_ROLE and args.seeds == '42'
    assert not args.force_rerun and args.max_series is None
    assert args.max_train_batches is args.max_val_batches is None
    assert not spec['performance_training_authorized'] and not spec['held_out_test']
    with pytest.raises(ValueError, match='Only Taxi/Instacart'):
        smoke.command(row|{'dataset':'raf_spare_parts'}, tmp_path, tmp_path, 'a'*40)


@pytest.mark.parametrize('failure', [None, 'preflight', 'cuda', 'skip', 'dataset', 'audit'])
def test_execute_is_sequential_fail_closed_without_retry(monkeypatch, tmp_path, failure):
    spec, rows = smoke.frozen_documents()
    monkeypatch.setattr(smoke, 'frozen_documents', lambda:(spec, rows))
    monkeypatch.setattr(smoke, 'verify_source', lambda *a,**k:'manifest_hash')
    monkeypatch.setattr(smoke, 'baseline', lambda *a:{})
    monkeypatch.setattr(smoke, 'runtime_metadata', lambda *a:spec['runtime'])
    actual_read = smoke.read
    monkeypatch.setattr(smoke, 'read', lambda path: {} if path == smoke.ROOT/'source_manifest.json' else actual_read(path))
    calls=[]
    def preflight(*args):
        calls.append('preflight')
        if failure=='preflight':raise ValueError('Insufficient VRAM')
        return {'gdm':'inactive'}
    def run(cmd, path, **kwargs):
        calls.append(path.name)
        if (failure=='cuda' and path.name.startswith('cuda')) or (failure=='dataset' and path.name.startswith(smoke.DATASETS[0])):
            raise subprocess.CalledProcessError(1,cmd)
        if path.name.startswith('cuda'):
            assert kwargs['env']['HARD_ELAPSED_AGE_TEST_DEVICE']=='cuda'
            assert all(name in cmd for name in smoke.CUDA_TESTS)
            xml(path.parent/'cuda_contract_tests.xml', skipped=failure=='skip')
    def audit(*args):
        if failure=='audit':raise ValueError('Contract mismatch')
        return {'status':'passed','dataset':args[1]['dataset']}
    monkeypatch.setattr(smoke, 'preflight', preflight)
    monkeypatch.setattr(smoke, 'run_logged', run)
    monkeypatch.setattr(smoke, 'audit_run', audit)
    args=argparse.Namespace(project_root=tmp_path,output_root=tmp_path/'new',source_revision='a'*40,audit_only=False)
    if failure:
        with pytest.raises((ValueError,subprocess.CalledProcessError)):smoke.execute(args)
    else:smoke.execute(args)
    status=json.loads((args.output_root/'status.json').read_text())
    assert status['status']==('failed' if failure else 'complete')
    assert not status['performance_training_authorized'] and not status['held_out_test_evaluated']
    if failure:
        assert 'error' in status and not status['completed']
        assert smoke.DATASETS[1]+'.log' not in calls
    else:
        assert calls==['preflight','cuda_contract_tests.log','preflight',smoke.DATASETS[0]+'.log','preflight',smoke.DATASETS[1]+'.log','preflight']
        assert status['cuda_contract_tests']['tests']==2 and len(status['completed'])==2
    before=(args.output_root/'status.json').read_bytes()
    with pytest.raises(FileExistsError):smoke.execute(args)
    assert (args.output_root/'status.json').read_bytes()==before


def test_cuda_xml_requires_both_modules_and_no_skips(tmp_path):
    p=tmp_path/'cuda.xml'
    p.write_text('<testsuites><testsuite tests="0"/></testsuites>')
    with pytest.raises(ValueError,match='no executed tests'):smoke.audit_cuda_tests(p)
    xml(p,omit_module=True)
    with pytest.raises(ValueError,match='not executed'):smoke.audit_cuda_tests(p)
    xml(p,skipped=True)
    with pytest.raises(ValueError,match='skipped test'):smoke.audit_cuda_tests(p)
    xml(p)
    assert smoke.audit_cuda_tests(p)['skipped']==0


def test_audit_only_uses_frozen_artifacts_without_cuda_or_training(monkeypatch,tmp_path):
    spec,rows=smoke.frozen_documents()
    smoke.save(tmp_path/'status.json',{'status':'complete','source_revision':'a'*40,'source_manifest_sha256':'hash',
               'runtime':spec['runtime'],'completed':[{'dataset':row['dataset']} for row in rows]})
    smoke.save(tmp_path/'execution_contract.json',spec)
    xml(tmp_path/'cuda_contract_tests.xml')
    monkeypatch.setattr(smoke,'verify_source',lambda *a,**k:'hash')
    monkeypatch.setattr(smoke,'baseline',lambda *a:{})
    monkeypatch.setattr(smoke,'audit_run',lambda *a,**k:{'status':'passed','dataset':a[1]['dataset']})
    forbidden=Mock(side_effect=AssertionError('audit-only must not execute CUDA/training'))
    monkeypatch.setattr(smoke,'preflight',forbidden)
    monkeypatch.setattr(smoke,'run_logged',forbidden)
    monkeypatch.setattr(smoke,'runtime_metadata',forbidden)
    before=(tmp_path/'status.json').read_bytes()
    args=argparse.Namespace(project_root=tmp_path,output_root=tmp_path,source_revision='a'*40,audit_only=True)
    assert smoke.execute(args)['status']=='passed'
    forbidden.assert_not_called()
    assert (tmp_path/'status.json').read_bytes()==before and (tmp_path/'local_audit.json').is_file()


def fixture_archive(omit=None):
    spec,_=smoke.frozen_documents();stream=io.BytesIO()
    names=set(packager.REQUIRED_SOURCE)|set(spec['frozen_files'])
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for name in sorted(names-({omit} if omit else set())):
            data=(smoke.ROOT/name).read_bytes();member=tarfile.TarInfo(name);member.size=len(data);archive.addfile(member,io.BytesIO(data))
        data=b'{}';member=tarfile.TarInfo('simple_lab_test/ignored.ipynb');member.size=len(data);archive.addfile(member,io.BytesIO(data))
    return stream.getvalue()


def test_package_is_committed_source_with_reference_hashes_and_sentinel(monkeypatch,tmp_path):
    archive=fixture_archive()
    def git(cmd,**kwargs):return {'rev-parse':'a'*40,'diff':b'','archive':archive}[cmd[1]]
    monkeypatch.setattr(packager.subprocess,'check_output',git)
    output=tmp_path/'source.tar.gz';packager.package(output)
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    with tarfile.open(output) as a:a.extractall(snapshot,filter='data')
    assert list((snapshot/'sample_data').iterdir())==[] and not list(snapshot.rglob('*.ipynb'))
    manifest=json.loads((snapshot/'source_manifest.json').read_text())
    assert not manifest['performance_training_authorized'] and len(list((snapshot/'references').rglob('*.pt')))==2
    assert not (snapshot/'scripts').exists()
    for name,expected in manifest['files'].items():assert smoke.digest(snapshot/name)==expected
    with pytest.raises(FileExistsError):packager.package(output)
    archive=fixture_archive(omit=packager.REQUIRED_SOURCE[2])
    with pytest.raises(ValueError,match='not committed'):packager.package(tmp_path/'missing.tar.gz')


def test_verify_source_local_reference_fallback_and_drift(tmp_path):
    _,rows=smoke.frozen_documents();files={smoke.CONTRACT:smoke.digest(smoke.ROOT/smoke.CONTRACT)}
    for row in rows:
        for name,(_,expected) in smoke.reference_files(row).items():files[f'references/{row["dataset"]}/{name}']=expected
    path=tmp_path/'source_manifest.json'
    smoke.save(path,{'source_revision':'a'*40,'performance_training_authorized':False,'files':files})
    assert smoke.verify_source('a'*40,path)==smoke.digest(path)
    files[smoke.CONTRACT]='wrong'
    smoke.save(path,{'source_revision':'a'*40,'performance_training_authorized':False,'files':files})
    with pytest.raises(ValueError,match='Source/reference mismatch'):smoke.verify_source('a'*40,path)


def test_changed_input_rejected_before_checkpoint_load(monkeypatch,tmp_path):
    _,rows=smoke.frozen_documents();monkeypatch.setattr(smoke,'digest',lambda path:'wrong')
    loader=Mock(side_effect=AssertionError('Unverified checkpoint must not load'))
    monkeypatch.setattr(torch,'load',loader)
    with pytest.raises(ValueError,match='Input changed'):smoke.baseline(rows[0],tmp_path)
    loader.assert_not_called()


def create_synthetic_e1(monkeypatch,tmp_path):
    _,rows=smoke.frozen_documents();row=rows[0]
    data,split,output=tmp_path/'data.parquet',tmp_path/'split.json',tmp_path/'fresh'
    pl.DataFrame({'oper_part_no':['a']*24+['b']*24,'seq':list(range(1,25))*2,
        'delta_t':[1.,2.,5.,3.]*12,'demand_qty':(list(map(float,range(1,24)))+[-999999.])*2,
        'chronological_split':(['train']*20+['validation']*3+['test'])*2}).write_parquet(data)
    split.write_text('{}');cmd=smoke.command(row,tmp_path,output,'a'*40)
    for flag,value in (('--data',str(data)),('--split-manifest',str(split)),('--device','cpu')):cmd[cmd.index(flag)+1]=value
    monkeypatch.setattr(sys,'argv',['test']+cmd[4:])
    monkeypatch.setattr(entry,'sha256_file',lambda path:row['data_sha256' if path==data else 'split_manifest_sha256'])
    monkeypatch.setattr(entry.pl,'read_parquet',Mock(side_effect=AssertionError('No held-out materialization')))
    entry.main()
    launch,summary=smoke.read(output/'launch_contract.json'),smoke.read(output/smoke.SUMMARY)
    reference=copy.deepcopy(summary);reference['parameter_count']=row['baseline_parameters']
    return output,row|{'train_targets':38,'validation_targets':6},launch,reference


def test_real_synthetic_e1_audit_and_fresh_beta_optimizer_contract(monkeypatch,tmp_path):
    output,row,launch,reference=create_synthetic_e1(monkeypatch,tmp_path)
    result=smoke.audit_run(output,row,(launch,reference),'a'*40,require_cuda=False)
    assert result['status']=='passed' and result['beta_nonzero_count']>0 and result['beta_optimizer_steps']==1
    with pytest.raises(ValueError,match='CUDA training device'):smoke.audit_run(output,row,(launch,reference),'a'*40)
    for drift in ({'epochs':300},{'partial_smoke':True},{'held_out_test_evaluated':True},{'source_revision':'wrong'},{'lambda_tail':.1}):
        smoke.save(output/'launch_contract.json',launch|drift)
        with pytest.raises(ValueError):smoke.audit_run(output,row,(launch,reference),'a'*40,require_cuda=False)
    smoke.save(output/'launch_contract.json',launch)
    for counts in ({'train_targets':39},{'validation_targets':7}):
        with pytest.raises(ValueError,match='Partial'):smoke.audit_run(output,row|counts,(launch,reference),'a'*40,require_cuda=False)
    last_path=(output/smoke.SUMMARY).parent/'last_epoch_state.pt';last=torch.load(last_path,weights_only=False)
    beta_state=next(state for state in last['optimizer_state_dict']['state'].values() if tuple(state['exp_avg'].shape)==(2,4))
    beta_state['exp_avg'].zero_();torch.save(last,last_path)
    with pytest.raises(ValueError,match='Beta optimizer state is inactive'):
        smoke.audit_run(output,row,(launch,reference),'a'*40,require_cuda=False)
    (output/'test_summary.json').write_text('{}')
    with pytest.raises(ValueError,match='Held-out artifact'):smoke.audit_run(output,row,(launch,reference),'a'*40,require_cuda=False)
