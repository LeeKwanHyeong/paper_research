"""Fabricated queue and scope checks; no SSH or original model evaluation."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from paper.scripts import manage_titans_mac_accuracy9_recovery as m
from paper.scripts import titans_mac_accuracy9_recovery_runtime as r


@pytest.fixture
def contract(tmp_path, monkeypatch):
    original = m.read(m.ORIGINAL)
    old = Path(original['hosts']['5090']['root'])
    h = original['hosts']['5090']; root = tmp_path / 'new'
    (root / 'source').mkdir(parents=True)
    jobs = []
    files = {'execution_contract.json': 'a' * 64}
    for j in original['jobs']:
        job = {**j, 'recovery_mode': 'resume' if j['dataset'] == 'intermittent_frozen_5000' else 'reference',
               'original_started_at': datetime.now(timezone.utc).isoformat()}
        keys = ('origin_resume_checkpoint',) if job['recovery_mode'] == 'resume' else ('origin_train_receipt', 'origin_train_exit')
        for key in keys:
            relative = 'run/' + j['id'] + '/' + key
            files[relative] = 'b' * 64
            job[key] = {'path': str(old / relative), 'sha256': 'b' * 64}
        if job['recovery_mode'] == 'resume':
            job['resume_epoch'] = {42: 76, 52: 74, 62: 67}[j['seed']]
        jobs.append(job)
    c = {'schema': r.SCHEMA, 'host': '5090', 'root': str(root), 'source_root': str(root / 'source'),
        **{k: deepcopy(h[k]) for k in ('python', 'gpu_uuid', 'runtime_expected', 'absolute_host_deadline_unix')},
        'max_workers': 3, 'limits': original['limits'], 'boot_id': 'fabricated-boot',
        'scientific_contract': {'path': str(old / 'execution_contract.json'), 'raw_sha256': m.sha(m.ORIGINAL),
            'canonical_sha256': m.canonical(original)},
        'source': {**original['source'], 'source_revision': original['source']['base_git_revision']},
        'input_files': original['input_files'], 'jobs': jobs,
        'immutable_originals': {'root': str(old), 'files': files}, 'operation_git_revision': 'fabricated',
        'authorization': {'resume_three_only': True, 'originals_immutable': True},
        'operation_files': {k: m.sha(m.ROOT / k) for k in m.OPERATION_FILES},
        'environment': {**h['environment'], 'PYTHONPATH': str(root / 'source'),
            'MPLCONFIGDIR': str(root / 'cache/matplotlib'), 'XDG_CACHE_HOME': str(root / 'cache'),
            'TORCHINDUCTOR_CACHE_DIR': str(root / 'cache/torchinductor')}}
    c['operation_files_sha256'] = r.sha_json(c['operation_files'])
    for key, mode in (('training_jobs', 'resume'), ('reference_jobs', 'reference')):
        c[key] = [j['id'] for j in jobs if j['recovery_mode'] == mode]
    c['validation_jobs'] = c['test_jobs'] = [j['id'] for j in jobs]
    monkeypatch.setattr(r, 'scientific_contract', lambda _: original)
    return c


def test_original_nine_and_only_three_resume(contract):
    assert len(r.validate_contract(contract)) == 9
    assert len(contract['training_jobs']) == 3 and len(contract['reference_jobs']) == 6


@pytest.mark.parametrize('mutation', ('deadline', 'runtime', 'old_root', 'new_seed', 'reference_retrain',
    'missing_validation', 'source', 'input', 'parallel', 'origin_escape', 'cache'))
def test_reject_scope_changes(contract, mutation):
    c = deepcopy(contract)
    if mutation == 'deadline': c['absolute_host_deadline_unix'] += 1
    if mutation == 'runtime': c['runtime_expected']['threads'] = 8
    if mutation == 'old_root': c['root'] = c['immutable_originals']['root']
    if mutation == 'new_seed': c['jobs'][-1]['seed'] = 72
    if mutation == 'reference_retrain': c['jobs'][0]['recovery_mode'] = 'resume'
    if mutation == 'missing_validation': c['validation_jobs'] = c['validation_jobs'][:-1]
    if mutation == 'source': c['source']['source_revision'] = 'changed'
    if mutation == 'input': c['input_files'] = {}
    if mutation == 'parallel': c['max_workers'] = 4
    if mutation == 'origin_escape': c['jobs'][-1]['origin_resume_checkpoint']['path'] = '/tmp/foreign.pt'
    if mutation == 'cache': c['environment']['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/shared'
    with pytest.raises(ValueError): r.validate_contract(c)


def test_resume_does_not_reset_original_fit_deadline(contract):
    j = next(j for j in contract['jobs'] if j['recovery_mode'] == 'resume')
    j['original_started_at'] = '2026-10-20T09:55:13.611297+00:00'
    expected = datetime.fromisoformat(j['original_started_at']).timestamp() + 129600
    assert r.fit_deadline(contract, j, expected + 10000) == min(expected, contract['absolute_host_deadline_unix'])


def test_queue_only_runs_three_resume_children_and_observes_exits(contract, monkeypatch):
    c = contract; root = Path(c['root']); (root / 'logs').mkdir(); children = []; peak = [0]
    class Child:
        def __init__(self, command, **kwargs):
            self.command, self.pid, self.returncode, self.polls = command, 1000 + len(children), None, 0
            assert kwargs['start_new_session'] is True
            children.append(self); peak[0] = max(peak[0], sum(x.returncode is None for x in children))
        def poll(self):
            if self.returncode is None:
                self.polls += 1
                if self.polls >= 4: self.returncode = 0
            return self.returncode
    def identity(pid):
        x = next((x for x in children if x.pid == pid), None)
        return None if x is None or x.returncode is not None else {'pid': pid, 'start_ticks': pid + 1,
            'argv': x.command, 'cwd': c['source_root'], 'pgid': pid}
    monkeypatch.setattr(r.subprocess, 'Popen', Child)
    monkeypatch.setattr(r, 'proc_identity', identity)
    monkeypatch.setattr(r, '_identity', lambda child, command, source, start_ticks=None: identity(child.pid))
    monkeypatch.setattr(r, 'gpu_state', lambda *a: {'compute_pids': [x.pid for x in children if x.returncode is None]})
    monkeypatch.setattr(r.time, 'sleep', lambda _: None)
    monkeypatch.setattr(r, 'verify_receipt', lambda c,j,s: {'path': j['id'], 'sha256': 'f'*64})
    result = r.run_stage(c, root / 'contract.json', 'train', r.time.monotonic())
    assert result['status'] == 'complete' and len(children) == peak[0] == 3
    assert all(e['actual_os_returncode'] == 0 and not e['timeout'] for e in result['exits'].values())
    assert {x.command[-1] for x in children} == set(c['training_jobs'])


@pytest.mark.parametrize('failure', (None, 'reference', 'train', 'validation'))
def test_all_validation_before_any_test(contract, monkeypatch, failure):
    c = contract; root = Path(c['root']); calls = []
    monkeypatch.chdir(root / 'source')
    monkeypatch.setattr(r, 'verify_files', lambda *a: None)
    monkeypatch.setattr(r, 'verify_originals', lambda *a: {'original_files_unchanged': True})
    monkeypatch.setattr(r, 'gpu_state', lambda *a: {'compute_pids': []})
    monkeypatch.setattr(r, 'proc_identity', lambda pid: None)
    monkeypatch.setattr(r, 'boot_id', lambda: c['boot_id'])
    def stage(c, cp, stage, began):
        if stage == 'test': assert calls[-1] == 'validation_gate'
        calls.append(stage)
        return {'status': 'failed' if stage == failure else 'complete', 'exits': {}, 'receipts': {}}
    monkeypatch.setattr(r, 'run_stage', stage)
    monkeypatch.setattr(r, 'validation_gate', lambda *a: calls.append('validation_gate'))
    code = r.dispatch(c, root / 'contract.json')
    assert (code == 0) == (failure is None)
    assert ('test' in calls) == (failure is None)
    if failure is None: assert calls == ['reference', 'train', 'validation', 'validation_gate', 'test']


def test_original_inventory_corruption_is_rejected_before_resume(tmp_path):
    p = tmp_path / 'original'; p.mkdir(); file = p / 'checkpoint'; file.write_bytes(b'original')
    c = {'immutable_originals': {'root': str(p), 'files': {'checkpoint': r.sha_file(file)}}}
    assert r.verify_originals(c)['original_files_unchanged']
    file.write_bytes(b'changed')
    with pytest.raises(ValueError): r.verify_originals(c)


@pytest.mark.parametrize('reuse,race', ((False, False), (True, False), (False, True)))
def test_cleanup_kills_term_resistant_owned_worker_but_never_reused_pid(contract, monkeypatch, reuse, race):
    import signal
    c = contract; root = Path(c['root']); job = c['jobs'][-1]; cp = root / 'contract.json'
    command = r.worker_command(c, cp, job, 'train'); live = {'pid': 2002, 'pgid': 2002,
        'start_ticks': 33 if reuse else 32, 'argv': command, 'cwd': c['source_root']}
    r.write(root / 'run' / job['id'] / 'train_lease.json', {'pid': 2002, 'start_ticks': 32,
        'job_id': job['id'], 'stage': 'train', 'command': command, 'boot_id': c['boot_id'],
        'contract_sha256': r.sha_json(c)}, exclusive=True)
    state = {'live': True}; signals = []
    monkeypatch.setattr(r, 'boot_id', lambda: c['boot_id'])
    monkeypatch.setattr(r, 'proc_identity', lambda pid: live if state['live'] else None)
    def send(pid, value):
        signals.append((pid, value))
        if race:
            state['live'] = False
            raise ProcessLookupError('already exited')
        if value == signal.SIGKILL: state['live'] = False
    monkeypatch.setattr(r.os, 'killpg', send)
    monkeypatch.setattr(r.bounded, 'gpu_snapshot', lambda _: {'compute_pids': [2002] if state['live'] else []})
    if reuse:
        with pytest.raises(ValueError): r.cleanup_leased_workers(c, cp, grace=0)
        assert signals == []
    else:
        result = r.cleanup_leased_workers(c, cp, grace=0)
        assert signals == ([(2002, signal.SIGTERM)] if race else [(2002, signal.SIGTERM), (2002, signal.SIGKILL)])
        assert result['status'] == 'passed' and not state['live']
        assert result['worker_own_os_exits_directly_observed'] is False


@pytest.mark.parametrize('timeout,cleanup_failure', ((False, False), (True, False), (False, True)))
def test_supervisor_cleans_new_leases_after_timeout_or_dispatcher_crash(contract, monkeypatch, timeout, cleanup_failure):
    c = contract; root = Path(c['root']); calls = []
    class Child:
        pid = 9001
        returncode = None
        def wait(self, **kwargs):
            if timeout: raise r.subprocess.TimeoutExpired('fabricated', 1)
            self.returncode = 1
            return 1
    child = Child()
    monkeypatch.setattr(r.subprocess, 'Popen', lambda *a, **k: child)
    monkeypatch.setattr(r, 'verify_files', lambda *a: None)
    monkeypatch.setattr(r, 'gpu_state', lambda *a: {'compute_pids': []})
    monkeypatch.setattr(r, '_identity', lambda *a, **k: {'pid': 9001, 'start_ticks': 100})
    def cleanup(*a, **k):
        calls.append('cleanup')
        if cleanup_failure: raise ValueError('unknown worker; no signal')
        r.write(root / 'supervisor_worker_cleanup.json', {'status': 'passed'}, exclusive=True)
    monkeypatch.setattr(r, 'cleanup_leased_workers', cleanup)
    monkeypatch.setattr(r, 'terminate_owned_child', lambda p: setattr(p, 'returncode', -15))
    assert r.supervise(c, root / 'contract.json') == 1
    assert calls == ['cleanup']
    assert r.read(root / 'dispatcher_exit.json')['timeout'] is timeout
    assert bool(r.read(root / 'dispatcher_exit.json')['cleanup_error']) is cleanup_failure


def test_import_from_only_sealed_science_and_operation_closure(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    original = m.read(m.ORIGINAL); source = tmp_path / 'source'; source.mkdir()
    for name in {*original['source']['files'], *m.OPERATION_FILES}:
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(m.ROOT / name, target)
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': str(source)}
    program = 'from paper.scripts import titans_mac_accuracy9_recovery_runtime as r; print(r.SCHEMA)'
    p = subprocess.run([sys.executable, '-c', program], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr
    assert r.SCHEMA in p.stdout
