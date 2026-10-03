"""Recovery boundaries, genuine progress watchdog, and unchanged scientific arguments."""
from copy import deepcopy
from pathlib import Path
import json
import os
import pytest
from paper.scripts import local_detail_instacart_remaining_contract as contract
from paper.scripts import run_local_detail_instacart_remaining as recovery
from paper.scripts import run_local_detail_benchmark as original
from simple_lab_test.search.tests.test_multilag_detail_execution import admitted_contract, cpu_training


def prepared():
    c = contract.protocol()
    c['source'] = {'base_git_revision':'1'*40, 'files':{}, 'files_sha256':original.common.sha_json({})}
    c['verification'] = {'status':'passed_synthetic_cpu','failed':0}
    return c


def test_recovery_excludes_completed_fits_and_preserves_science(tmp_path):
    c = prepared(); p = original.read(contract.ROOT/contract.BASE)
    c['source']['base_git_revision'] = p['source']['base_git_revision']
    contract.validate_contract(c,verify_source=False)
    assert c['execution_arms'] == ['rmtpp','thp','nhp','sahp']
    assert original.scheduled_jobs(c,'5090') == [('insta_market_basket',62)]
    assert (c['total_arms'],c['endpoint_replays'],c['maximum_optimizer_steps']) == (4,8,18668400)
    for key in ('datasets','architecture','training','comparison','library_sha256','limits'):
        assert c[key] == p[key]
    assert c['launch']['fixed_deadline_unix'] == p['launch']['fixed_deadline_unix']
    assert c['remaining_recovery']['completed_fits'] == 8
    for arm in c['execution_arms']:
        d = next(d for d in c['datasets'] if d['dataset_id']=='insta_market_basket')
        aa = vars(original.training_args(c,d,tmp_path,arm))
        bb = vars(original.training_args(p,d,tmp_path,arm))
        assert aa == bb


@pytest.mark.parametrize('mutation', ['arm','deadline','seed','runtime','training'])
def test_recovery_protocol_tampering_rejected(mutation):
    c = prepared()
    if mutation == 'arm': c['execution_arms'].append('titantpp')
    if mutation == 'deadline': c['launch']['fixed_deadline_unix'] += 1
    if mutation == 'seed': c['replication']['jobs'][0]['seed'] = 72
    if mutation == 'runtime': c['hosts']['5090']['python'] = '/another/python'
    if mutation == 'training': c['training']['patience'] = 41
    with pytest.raises(ValueError):contract.validate_contract(c,verify_source=False)


def test_stalled_worker_rejected_even_when_arm_time_limit_has_not_expired():
    pulse = {'pid':123,'arm':'rmtpp','contract_sha256':'digest','monotonic':100.}
    args = dict(child_pid=123, arm='rmtpp', digest='digest', started=90., timeout=1800.)
    recovery.check_progress(pulse,now=1899.9,**args)
    # The old watchdog waits up to 120 hours; this regression is a live but stalled worker.
    with pytest.raises(ValueError,match='no progress'):
        recovery.check_progress(pulse,now=1900.,**args)


@pytest.mark.parametrize('field,value', [('pid',124),('arm','nhp'),('contract_sha256','wrong'),
                                       ('monotonic',float('nan')),('monotonic',3000.)])
def test_foreign_or_invalid_progress_cannot_keep_worker_alive(field,value):
    pulse = {'pid':123,'arm':'rmtpp','contract_sha256':'digest','monotonic':100.}
    pulse[field] = value
    with pytest.raises(ValueError):recovery.check_progress(pulse,child_pid=123,arm='rmtpp',digest='digest',
                                                        started=90.,now=1000.,timeout=1800.)


def test_progress_only_advances_when_actual_budget_callback_runs(tmp_path,monkeypatch):
    c = prepared(); now = [100.]
    monkeypatch.setattr(recovery.faulthandler,'cancel_dump_traceback_later',lambda:None)
    monkeypatch.setattr(recovery.faulthandler,'dump_traceback_later',lambda *a,**k:None)
    pulse = recovery.ProgressBudget(lambda:None,tmp_path/'pulse.json','rmtpp',c,
                                   clock=lambda:now[0],mono=lambda:now[0])
    pulse(); before = (tmp_path/'pulse.json').read_bytes()
    now[0] = 1000.
    assert (tmp_path/'pulse.json').read_bytes() == before
    pulse()
    assert json.loads((tmp_path/'pulse.json').read_text())['monotonic'] == 1000.


@pytest.mark.parametrize('arm',contract.REMAINING)
def test_remaining_arm_real_cpu_training_and_two_replays(cpu_training,admitted_contract,tmp_path,monkeypatch,arm):
    c = prepared(); c['datasets'] = deepcopy(admitted_contract['datasets'])
    c['hosts']['5090']['output_dir'] = str(tmp_path/'run')
    c['training'].update(maximum_epochs=40,minimum_epochs=40,patience=40)
    c['source']['base_git_revision'] = '1'*40
    for d in c['datasets']:
        d['steps_per_epoch'] = (d['inherited_data_identity']['populations']['train']['target_count']+127)//128
    d = next(d for d in c['datasets'] if d['dataset_id']=='insta_market_basket')
    c['remaining_recovery']['initial_state_sha256'] = original.initialization(d,62)
    monkeypatch.setattr(recovery,'verify_process_environment',lambda *a,**k:None)
    monkeypatch.setattr(recovery.common,'runtime_check',lambda *a,**k:{'synthetic':'cpu'})
    monkeypatch.setattr(recovery,'validate_execution_contract',lambda *a,**k:None)
    original_args = recovery.training_args
    def args(*a,**k):
        value = original_args(*a,**k);value.device = 'cpu';return value
    monkeypatch.setattr(recovery,'training_args',args)
    replay = recovery.replay_checkpoint
    monkeypatch.setattr(recovery,'replay_checkpoint',lambda *a,**k:replay(*a,device='cpu',**k))
    result = recovery.run_arm(c,{'deadline_unix':9999999999},'5090',{'runtime':{'synthetic':'cpu'}},lambda:None,arm)
    assert result['completed_epochs'] == 40
    assert result['global_steps'] == d['steps_per_epoch']*40
    assert result['selected']['held_out_test_evaluated'] is False
    assert result['last']['held_out_test_evaluated'] is False
    fits = list((tmp_path/'run/insta_market_basket/seed_62/runs').iterdir())
    assert [x.name for x in fits] == [arm]
    with pytest.raises(ValueError,match='Fresh arm'):
        recovery.run_arm(c,{'deadline_unix':9999999999},'5090',{'runtime':{'synthetic':'cpu'}},lambda:None,arm)


@pytest.mark.parametrize('fail_at',[None,1])
def test_supervisor_fresh_processes_and_failure_does_not_retry_or_start_next(tmp_path,monkeypatch,fail_at):
    import subprocess
    import sys
    import time
    c=prepared(); spec=c['hosts']['5090']
    spec.update(root=str(tmp_path),output_dir=str(tmp_path/'run'),source_root=str(tmp_path),python=sys.executable)
    started=time.time()-10
    permit={'started_at_unix':started,'deadline_unix':started+c['limits']['total_wall_seconds']}
    cp,ap,tp=[tmp_path/n for n in ('contract.json','approval.json','permit.json')]
    for path,value in ((cp,c),(ap,{}),(tp,permit)):path.write_text(json.dumps(value))
    monkeypatch.setattr(original,'verify_authorization',lambda *a,**k:spec)
    for name in ('source_and_host','verify_previous_run_complete','verify_local_qualification',
                 'apply_process_environment','verify_start_memory','verify_active_arm_deadline'):
        monkeypatch.setattr(original,name,lambda *a,**k:None)
    monkeypatch.setattr(recovery.shared,'check_storage',lambda *a,**k:None)
    monkeypatch.setattr(recovery,'collect_final',lambda *a:{'status':'complete'})
    real_popen=subprocess.Popen; launched=[]
    def spawn(argv,**kwargs):
        arm=argv[argv.index('--arm')+1];fd=argv[argv.index('--authority-fd')+1]
        code="""import os,json,sys
from pathlib import Path
with os.fdopen(int(sys.argv[1]),'rb') as f:a=json.loads(f.read())
if sys.argv[3]=='fail':sys.exit(7)
p=Path(sys.argv[2])/'run/arm_receipts'/f"{a['arm']}.json"
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps({'status':'complete','pid':os.getpid(),'contract_sha256':a['contract_sha256']}))
"""
        mode='fail' if len(launched)==fail_at else 'pass'
        child=real_popen([sys.executable,'-c',code,fd,str(tmp_path),mode],**kwargs)
        launched.append((arm,child.pid))
        return child
    monkeypatch.setattr(recovery.subprocess,'Popen',spawn)
    if fail_at is None:
        assert recovery.supervise(cp,ap,tp)['status']=='complete'
        assert [a for a,_ in launched] == list(contract.REMAINING)
        assert len({p for _,p in launched}) == 4
    else:
        with pytest.raises(ValueError,match='no retry'):recovery.supervise(cp,ap,tp)
        assert [a for a,_ in launched] == ['rmtpp','thp']
        assert original.read(tmp_path/'run/status.json')['status']=='stopped'
    with pytest.raises(FileExistsError):recovery.supervise(cp,ap,tp)
