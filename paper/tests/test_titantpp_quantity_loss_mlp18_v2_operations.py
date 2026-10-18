"""Mock-only regression for qualification self-context; no Torch or remote calls."""
from copy import deepcopy
import os
from pathlib import Path
import sys
import pytest

from paper.scripts import titantpp_quantity_loss_mlp18_runtime_v2 as r
from paper.scripts import manage_titantpp_quantity_loss_mlp18_v2 as m
from paper.scripts import titantpp_quantity_loss_mlp18_runtime as sealed_r
from paper.scripts import manage_titantpp_quantity_loss_mlp18 as sealed_m
from paper.tests.test_titantpp_quantity_loss_mlp18_operations import contract as base_contract,stopped


@pytest.fixture
def setup(tmp_path,monkeypatch):
    c=base_contract.__wrapped__(tmp_path);host='5080';h=c['hosts'][host]
    monkeypatch.chdir(h['source_root'])
    expected=[h['python'],'-u','-m',r.MODULE,'--contract',str(Path(h['root'])/'execution_contract.json'),'--host',host,'--mode','qualify']
    identity={'pid':os.getpid(),'pgid':os.getpid(),'start_ticks':1234,'cwd':h['source_root'],'argv':expected,'state':'R'}
    r.write(Path(h['root'])/'qualification_attempt.json',{'host':host,'contract_sha256':r.sha_json(c)},exclusive=True)
    monkeypatch.setattr(r._impl,'proc_identity',lambda pid:deepcopy(identity) if pid==os.getpid() else None)
    monkeypatch.setattr(r._impl,'paused_processes',lambda *a:stopped())
    monkeypatch.setattr(r._impl,'gpu_snapshot',lambda *a:{'compute_pids':[11,12,13,os.getpid()]})
    return c,host,identity


def fake_checks(c,h):
    gpu,paused=r._impl.gpu_state(c,h)
    return {'status':'passed','gpu_after':gpu,'paused_after':paused,'native_worker_exit0_count':6}


def test_self_cuda_context_admitted_only_with_current_qualified_identity(setup,monkeypatch):
    c,h,identity=setup;original=r._impl.gpu_state;torch_before='torch' in sys.modules
    monkeypatch.setattr(r,'_original_three_concurrent',fake_checks)
    with pytest.raises(ValueError,match='Unidentified'):original(c,h)
    result=r.qualify_three_concurrent(c,h)
    assert result['qualification_process_identity']==identity and result['qualification_self_context_allowed'] is True
    assert r._impl.gpu_state is original
    observation=r.read(Path(c['hosts'][h]['root'])/'qualification_gpu_state_observation.json')
    owner=r.read(Path(c['hosts'][h]['root'])/'qualification_owner_identity.json')
    assert observation['qualification_owner_present_in_gpu_observation'] is True
    assert observation['gpu']['compute_pids']==[11,12,13,os.getpid()]
    assert owner['qualification_owner_identity']==identity
    assert ('torch' in sys.modules)==torch_before
    # The same PID is NOT implicitly allowed outside the qualification scope.
    with pytest.raises(ValueError,match='Unidentified'):r._impl.gpu_state(c,h)


def test_unidentified_extra_gpu_pid_remains_rejected_and_hook_restored(setup,monkeypatch):
    c,h,_=setup;original=r._impl.gpu_state
    monkeypatch.setattr(r._impl,'gpu_snapshot',lambda *a:{'compute_pids':[11,12,13,os.getpid(),999999]})
    monkeypatch.setattr(r,'_original_three_concurrent',fake_checks)
    with pytest.raises(ValueError,match='Unidentified'):r.qualify_three_concurrent(c,h)
    assert r._impl.gpu_state is original
    evidence=r.read(Path(c['hosts'][h]['root'])/'qualification_gpu_state_observation.json')
    assert 999999 in evidence['gpu']['compute_pids']


@pytest.mark.parametrize('field,value',[('cwd','/foreign'),('argv',['python','foreign.py']),('pgid',-1),('pid',-1)])
def test_qualification_self_identity_mismatch_rejected(setup,monkeypatch,field,value):
    c,h,identity=setup;identity[field]=value
    monkeypatch.setattr(r,'_original_three_concurrent',lambda *a:pytest.fail('invalid identity must not run synthetic children'))
    with pytest.raises(ValueError,match='identity mismatch'):r.qualify_three_concurrent(c,h)


def test_start_ticks_reuse_during_scoped_gpu_read_rejected(setup,monkeypatch):
    c,h,identity=setup;original=r._impl.gpu_state;calls=[]
    def pid(pid):
        calls.append(pid)
        return {**identity,'start_ticks':1234 if len(calls)==1 else 5678}
    monkeypatch.setattr(r._impl,'proc_identity',pid)
    monkeypatch.setattr(r,'_original_three_concurrent',fake_checks)
    with pytest.raises(ValueError,match='changed during admission'):r.qualify_three_concurrent(c,h)
    assert r._impl.gpu_state is original


def test_wrong_qualification_contract_binding_rejected(setup,monkeypatch):
    c,h,_=setup
    r.write(Path(c['hosts'][h]['root'])/'qualification_attempt.json',{'host':h,'contract_sha256':'foreign'})
    monkeypatch.setattr(r,'_original_three_concurrent',lambda *a:pytest.fail('invalid contract must not run synthetic children'))
    with pytest.raises(ValueError,match='attempt does not match'):r.qualify_three_concurrent(c,h)


def test_scoped_gpu_contract_host_escape_rejected(setup,monkeypatch):
    c,h,_=setup;original=r._impl.gpu_state
    def foreign(*a):return r._impl.gpu_state(deepcopy(c),h)
    monkeypatch.setattr(r,'_original_three_concurrent',foreign)
    with pytest.raises(ValueError,match='escaped'):r.qualify_three_concurrent(c,h)
    assert r._impl.gpu_state is original


def test_v1_modules_and_scientific_worker_are_not_patched(setup,monkeypatch):
    c,h,_=setup
    identities=(sealed_r.MODULE,sealed_r.gpu_state,sealed_r.qualify_three_concurrent,sealed_m.NAME,sealed_m.runtime)
    digest=r.sha_file(Path(r.__file__).with_name('run_titantpp_quantity_loss_mlp18.py'))
    monkeypatch.setattr(r,'_original_three_concurrent',fake_checks)
    assert r.qualify_three_concurrent(c,h)['status']=='passed'
    assert identities==(sealed_r.MODULE,sealed_r.gpu_state,sealed_r.qualify_three_concurrent,sealed_m.NAME,sealed_m.runtime)
    assert digest==r.sha_file(Path(r.__file__).with_name('run_titantpp_quantity_loss_mlp18.py'))
    assert r._impl.SCHEMA==sealed_r.SCHEMA and r._impl.WORKER_MODULE==sealed_r.WORKER_MODULE


def test_manager_v2_isolated_roots_and_runtime_exports():
    assert m.NAME=='titantpp_quantity_loss_mlp18_20261019_v2'
    assert m.BUNDLE.name==m.REPORT.name==m.NAME and m.MODULE==r.MODULE
    assert m._impl.runtime is r and m._impl.prepare.__globals__ is m._impl.__dict__
    assert sealed_m.NAME.endswith('_v1') and sealed_m.runtime is sealed_r
    assert len(m.build_jobs())==18 and {j['loss_mode'] for j in m.build_jobs()}=={'raw_mse','raw_mae'}
    assert '20261019_v2' in m._impl.prepare.__code__.co_consts or any('20261019_v2' in str(x) for x in m._impl.prepare.__code__.co_consts)


def test_source950_preserved_plus_two_v2_operators(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir();files={}
    for i in range(950):
        name=f'old/{i}.py';p=root/name;p.parent.mkdir(exist_ok=True);p.write_text(str(i));files[name]=m.sha(p)
    for name in m.REQUIRED_NEW_SOURCE:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('v2 operation')
    bundle=tmp_path/'v1';report=tmp_path/'report'
    c={'source':{'files':files,'files_sha256':m.canonical(files),'model_science_source_revision':'unchanged','model_science_source_files_sha256':'science941'}}
    m.write(bundle/'execution_contract.json',c)
    m.write(report/'PREPARED.json',{'contract_sha256':m.sha(bundle/'execution_contract.json'),'canonical_contract_sha256':m.canonical(c)})
    monkeypatch.setattr(m,'_V1_BUNDLE',bundle);monkeypatch.setattr(m,'_V1_REPORT',report);monkeypatch.setattr(m._impl,'ROOT',root)
    def git(cmd,**kw):
        if cmd[1]=='ls-files':return '\n'.join(m.REQUIRED_NEW_SOURCE)
        if cmd[1]=='diff':return ''
        if cmd[1]=='rev-parse':return 'newv2commit\n'
        raise AssertionError(cmd)
    monkeypatch.setattr(m._impl.subprocess,'check_output',git)
    out=m.source_snapshot()
    assert len(out['files'])==952 and out['parent_source_file_count']==950
    assert out['model_science_source_revision']=='unchanged'
    assert out['operational_repair_parent']['v1_attempts_preserved'] is True
    (root/'old/1.py').write_text('changed')
    with pytest.raises(ValueError,match='Sealed v1 source changed'):m.source_snapshot()


def test_v1_six_child_checks_run_through_scoped_v2_self_allowance(setup,monkeypatch):
    c,h,identity=setup;root=Path(c['hosts'][h]['root']);source=Path(c['hosts'][h]['source_root'])
    (root/'execution_contract.json').write_text(r.json.dumps(c))
    package=source/'paper/scripts';package.mkdir(parents=True)
    (source/'paper/__init__.py').write_text('');(package/'__init__.py').write_text('')
    (package/'run_titantpp_quantity_loss_mlp18.py').write_text(
        "def synthetic_condition(c,h,m):\n return {'status':'passed','loss_mode':m,'real_data_loaded':False}\n")
    c['hosts'][h]['environment']['PYTHONPATH']=str(source)
    # Registration must bind the exact contract, including this fixture-only environment.
    r.write(root/'qualification_attempt.json',{'host':h,'contract_sha256':r.sha_json(c)})
    out=r.qualify_three_concurrent(c,h)
    assert out['native_worker_exit0_count']==6 and out['qualification_owner_identity']==identity
    assert out['qualification_owner_present_in_gpu_observation'] is True
    assert set(out['loss_modes'])=={'raw_mse','raw_mae'}
    assert len(list(root.glob('admission_*.json')))==6
