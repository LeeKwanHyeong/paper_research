"""Fabricated recovery lifecycle checks: no SSH, CUDA, data or checkpoint inference."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from paper.scripts import titantpp_late_path18_5090_recovery_runtime as r
from paper.tests.test_titantpp_quantity_loss_mlp18_operations import proof


def test_production_bindings_are_fixed():
    assert r.SOURCE_COUNT == 963
    assert r.SOURCE_SHA256 == '57dca469664c78d008ba9dd091c1b5ee55d4236fa60cbd18a5b0a828a3f5c94a'
    assert r.DEADLINE == 1792888885.815062 and r.HOST == '5090'
    assert r.STAGES == ('train', 'validation')


@pytest.fixture
def contract(tmp_path, monkeypatch):
    # Only the immutable Validation-only contract is read; no research rows are loaded.
    path = Path(__file__).resolve().parents[2] / 'search_artifacts/titantpp_mlp16_late_path18_20261019_v2/execution_contract.json'
    science = deepcopy(json.loads(path.read_text()))
    old, new = tmp_path / 'old', tmp_path / 'recovery'
    old_source, source = old / 'source', new / 'source'
    old_source.mkdir(parents=True); source.mkdir(parents=True)
    for base in (old_source, source):
        (base / 'frozen.py').write_text('old science unchanged\n')
    files = {'frozen.py': r.sha_file(source / 'frozen.py')}
    monkeypatch.setattr(r, 'SOURCE_COUNT', 1)
    monkeypatch.setattr(r, 'SOURCE_SHA256', r.sha_json(files))
    science['source']['files'] = files
    science['source']['files_sha256'] = r.sha_json(files)
    science['hosts']['5090']['root'] = str(old)
    science['hosts']['5090']['source_root'] = str(old_source)
    science['input_files'] = {}
    for dataset in ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000'):
        for name in ('train_validation.parquet', 'split_manifest.json'):
            p = old_source / 'inputs/data' / dataset / name
            p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'fabricated input')
            science['input_files'][str(p.relative_to(old_source))] = r.sha_file(p)
    sp = old / 'execution_contract.json'; r.write(sp, science, exclusive=True)
    jobs = [deepcopy(j) for j in science['jobs'] if j['host'] == '5090']
    for j in jobs:
        folder = old / 'run' / j['id']; folder.mkdir(parents=True)
        if j['dataset'] == 'raf_spare_parts':
            j['recovery_mode'] = 'reference'
            cp, tr = folder / 'selected.pt', folder / 'train_receipt.json'
            cp.write_bytes(b'fixed selected binary'); tr.write_text('{}\n')
            j['origin_train_receipt'] = {'path': str(tr), 'sha256': r.sha_file(tr)}
            j['origin_selected_checkpoint'] = {'path': str(cp), 'sha256': r.sha_file(cp)}
        elif j['seed'] == 62 and j['arm'] == r.ARMS[1]:
            j['recovery_mode'] = 'fresh'
        else:
            j['recovery_mode'] = 'resume'
            cp = folder / 'resume.pt'; cp.write_bytes(b'fixed resume binary')
            j['origin_resume_checkpoint'] = {'path': str(cp), 'sha256': r.sha_file(cp)}
            j['resume_epoch'] = 10; j['original_started_at'] = '2026-10-19T07:00:00+00:00'
    operation_files = {}
    for name in ('titantpp_late_path18_5090_recovery_runtime.py', 'run_titantpp_late_path18_recovery.py', 'manage_recovery.py'):
        p = source / 'paper/scripts' / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('new operation only\n')
        operation_files[str(p.relative_to(source))] = r.sha_file(p)
    target = science['hosts']['5090']
    c = {'schema': r.SCHEMA, 'host': '5090', 'root': str(new), 'source_root': str(source),
         **{k: target[k] for k in ('python', 'environment', 'runtime_expected', 'gpu_uuid', 'absolute_host_deadline_unix')},
         'boot_id': 'new-boot', 'limits': {'fit_wall_seconds': 129600, 'queue_wall_seconds': 604800, 'evaluation_wall_seconds': 7200},
         'max_workers': 3, 'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
         'scientific_contract': {'path': str(sp), 'raw_sha256': r.sha_file(sp), 'canonical_sha256': r.sha_json(science)},
         'source': {'files': files, 'files_sha256': r.sha_json(files), 'source_revision': science['source']['base_git_revision']},
         'input_files': science['input_files'], 'operation_files': operation_files, 'operation_files_sha256': r.sha_json(operation_files), 'operation_git_revision': 'new-op-revision',
         'jobs': jobs, 'training_jobs': [j['id'] for j in jobs if j['recovery_mode'] != 'reference'],
         'validation_jobs': [j['id'] for j in jobs],
         'immutable_originals': {'root': str(old), 'files': {str(p.relative_to(old)): r.sha_file(p) for p in old.rglob('*') if p.is_file()}}}
    c['environment'] = {**target['environment'], 'PYTHONPATH': str(source), 'MPLCONFIGDIR': str(new / 'cache/matplotlib'), 'XDG_CACHE_HOME': str(new / 'cache')}
    for name in science['input_files']:
        p = source / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes((old_source / name).read_bytes())
    monkeypatch.setattr(r, 'boot_id', lambda: 'new-boot')
    return c


def test_exact_four_training_ten_validation_and_source_split(contract):
    jobs = r.validate_contract(contract)
    assert len(jobs) == 4 and sum(j['recovery_mode'] == 'resume' for j in jobs) == 3
    assert sum(j['recovery_mode'] == 'fresh' for j in jobs) == 1
    assert all(j['dataset'] == 'intermittent_frozen_5000' for j in jobs)
    assert len(contract['validation_jobs']) == 10
    assert r.verify_files(contract)['source']['files'] == contract['source']['files']


@pytest.mark.parametrize('mutation', ['host', 'test', 'workers', 'limit', 'train_raf', 'omit_val', 'duplicate_val',
    'mode', 'original_job', 'source', 'source_revision', 'operation', 'deadline', 'gpu', 'root', 'empty_originals', 'origin'])
def test_recovery_scope_mutations_rejected(contract, mutation):
    c = deepcopy(contract)
    if mutation == 'host': c['host'] = '5080'
    if mutation == 'test': c['held_out_test_evaluated'] = True
    if mutation == 'workers': c['max_workers'] = 4
    if mutation == 'limit': c['limits']['fit_wall_seconds'] += 1
    if mutation == 'train_raf': c['training_jobs'][0] = c['validation_jobs'][0]
    if mutation == 'omit_val': c['validation_jobs'].pop()
    if mutation == 'duplicate_val': c['validation_jobs'][-1] = c['validation_jobs'][0]
    if mutation == 'mode': c['jobs'][0]['recovery_mode'] = 'fresh'
    if mutation == 'original_job': c['jobs'][0]['seed'] = 32
    if mutation == 'source': c['source']['files']['frozen.py'] = '0' * 64
    if mutation == 'source_revision': c['source']['source_revision'] = 'changed'
    if mutation == 'operation': c['operation_files']['extra.py'] = '0' * 64
    if mutation == 'deadline': c['absolute_host_deadline_unix'] += 1
    if mutation == 'gpu': c['gpu_uuid'] = 'wrong'
    if mutation == 'root': c['root'] = c['immutable_originals']['root']
    if mutation == 'empty_originals': c['immutable_originals']['files'] = {}
    if mutation == 'origin': c['jobs'][0]['origin_selected_checkpoint']['path'] = '/outside.pt'
    with pytest.raises((ValueError, KeyError)):
        r.validate_contract(c)


@pytest.mark.parametrize('kind', ['original', 'source', 'operation', 'input', 'boot'])
def test_hash_and_boot_changes_rejected(contract, kind, monkeypatch):
    c = contract
    if kind == 'original': Path(c['jobs'][0]['origin_selected_checkpoint']['path']).write_bytes(b'changed')
    if kind == 'source': (Path(c['source_root']) / 'frozen.py').write_text('changed')
    if kind == 'operation': (Path(c['source_root']) / next(iter(c['operation_files']))).write_text('changed')
    if kind == 'input':
        sci = r.scientific_contract(c)
        (Path(sci['hosts']['5090']['source_root']) / next(iter(sci['input_files']))).write_bytes(b'changed')
    if kind == 'boot': monkeypatch.setattr(r, 'boot_id', lambda: 'other-boot')
    with pytest.raises(ValueError): r.verify_files(c)


def test_gpu_no_inherited_paused_allowances(contract, monkeypatch):
    monkeypatch.setattr(r, 'gpu_snapshot', lambda _: {'compute_pids': [100]})
    assert r.gpu_state(contract, [100])['compute_pids'] == [100]
    with pytest.raises(ValueError): r.gpu_state(contract)
    with pytest.raises(ValueError): r.gpu_state(contract, [100, 100])
    with pytest.raises(ValueError): r.gpu_state(contract, [100, 101, 102, 103])


def make_receipt(c, job, stage):
    root = Path(c['root']); folder = root / 'run' / job['id']; folder.mkdir(parents=True, exist_ok=True)
    cp = folder / 'selected.pt'
    if job['recovery_mode'] == 'reference': cp.write_bytes(Path(job['origin_selected_checkpoint']['path']).read_bytes())
    elif not cp.exists(): cp.write_bytes(b'new selected binary')
    value = {'schema': r.STAGE_SCHEMA, 'status': 'complete', 'stage': stage, 'job_id': job['id'],
        'contract_sha256': r.sha_json(c), 'scientific_contract_sha256': c['scientific_contract']['canonical_sha256'],
        'scientific_contract_raw_sha256': c['scientific_contract']['raw_sha256'],
        **{k: job[k] for k in ('host', 'dataset', 'arm', 'seed', 'time_alpha', 'loss_mode', 'time_input')},
        'candidate_name': job['arm'], 'late_path_mode': 'observation' if job['arm'] == r.ARMS[0] else 'hidden',
        'late_path_parameter_count': 1024, 'time_input_transform': 'log1p_nonnegative_float32',
        'quantity_input_transform': 'log1p_nonnegative_float32', 'held_out_test_evaluated': False,
        'quantity_only': True, 'time_head_trained': False, 'quantity_loss_normalization': 'none',
        'time_head_no_training_verification': proof(), 'selected_checkpoint': str(cp), 'checkpoint_sha256': r.sha_file(cp),
        'selected_epoch': 9, 'state_sha256': 's' * 64, 'target_identity': {'target_count': 4},
        'metrics': {'overall': {'qty_rmse': 1., 'qty_mae': .5}}}
    if stage == 'validation': value['validation_replay_passed'] = True
    value['files'] = {str(cp.relative_to(root)): r.sha_file(cp)}
    r.write(folder / f'{stage}_receipt.json', value, exclusive=True)
    return value


def test_receipt_selection_binary_and_time_head_bindings(contract):
    job = r.validate_contract(contract)[0]
    make_receipt(contract, job, 'train'); value = make_receipt(contract, job, 'validation')
    path = Path(contract['root']) / 'run' / job['id'] / 'validation_receipt.json'
    assert r.verify_receipt(contract, job, 'validation')['sha256']
    for change in ({'time_head_trained': True}, {'selected_epoch': 10}, {'scientific_contract_sha256': 'bad'},
                   {'files': {}}, {'validation_replay_passed': False}, {'state_sha256': 'bad'}, {'held_out_test_evaluated': True}):
        r.write(path, {**value, **change})
        with pytest.raises(ValueError): r.verify_receipt(contract, job, 'validation')
    r.write(path, value)
    Path(value['selected_checkpoint']).write_bytes(b'changed')
    with pytest.raises(ValueError): r.verify_receipt(contract, job, 'validation')


def test_raf_reference_only_and_exact_selected_copy(contract):
    job = next(j for j in contract['jobs'] if j['recovery_mode'] == 'reference')
    with pytest.raises(ValueError): r.worker_command(contract, '/c', job, 'train')
    with pytest.raises(ValueError): r.worker_command(contract, '/c', job, 'test')
    make_receipt(contract, job, 'validation')
    assert r.verify_receipt(contract, job, 'validation')['sha256']


def fake_queue(c, monkeypatch, fail=False):
    instances, launches, maximum = [], [], [0]
    jobs = {j['id']: j for j in c['jobs']}
    first = c['training_jobs'][0]
    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 1000 + len(instances); self.command = command; self.returncode = None; self.ticks = 0
            self.cwd = str(kwargs['cwd']); self.job = jobs[command[command.index('--job') + 1]]; self.stage = command[-1]
            assert kwargs['start_new_session'] and kwargs['env']['PYTHONHASHSEED'] == str(self.job['seed'])
            instances.append(self); launches.append((self.stage, self.job['id']))
            maximum[0] = max(maximum[0], sum(p.returncode is None for p in instances))
        def poll(self):
            if self.returncode is None:
                self.ticks += 1
                if self.ticks >= 8:
                    self.returncode = 1 if fail and self.stage == 'train' and self.job['id'] == first else 0
                    if self.returncode == 0: make_receipt(c, self.job, self.stage)
            return self.returncode
    monkeypatch.setattr(r.subprocess, 'Popen', Child)
    monkeypatch.setattr(r, 'gpu_state', lambda c, pids=(): {'compute_pids': list(pids)})
    monkeypatch.setattr(r, 'proc_identity', lambda pid: next(({'pid': p.pid, 'pgid': p.pid, 'start_ticks': p.pid,
        'argv': p.command, 'cwd': p.cwd} for p in instances if p.pid == pid and p.returncode is None), None))
    monkeypatch.setattr(r.time, 'sleep', lambda _: None)
    return launches, maximum


def test_full_recovery_four_train_then_ten_validation_maxthree(contract, monkeypatch):
    root = Path(contract['root']); monkeypatch.chdir(contract['source_root'])
    r.write(root / 'qualification.json', {'status': 'passed', 'host': '5090', 'contract_sha256': r.sha_json(contract), 'native_worker_exit0_count': 6})
    launches, maximum = fake_queue(contract, monkeypatch)
    assert r.dispatch(contract, root / 'execution_contract.json') == 0
    assert [x[0] for x in launches] == ['train'] * 4 + ['validation'] * 10
    assert maximum[0] == 3
    assert all(jobs not in [j['id'] for j in contract['jobs'] if j['recovery_mode'] == 'reference'] for stage, jobs in launches if stage == 'train')
    final = r.read(root / 'final_receipt.json')
    assert final['status'] == 'complete' and final['completed_fit_count'] == 4 and final['validation_replay_count'] == 10
    assert final['owned_worker_pids_absent'] and final['test_count'] == 0
    assert len(list((root / 'logs').glob('*.stdout.log'))) == 14
    manifest = r.read(root / 'artifact_manifest.json')
    for name, digest in manifest['files'].items(): assert r.sha_file(root / name) == digest
    with pytest.raises(ValueError, match='Existing'): r.dispatch(contract, root / 'execution_contract.json')


def test_failure_stops_admission_drains_only_current_children(contract, monkeypatch):
    root = Path(contract['root']); (root / 'logs').mkdir()
    launches, maximum = fake_queue(contract, monkeypatch, fail=True)
    result = r.run_stage(contract, root / 'execution_contract.json', 'train', r.time.monotonic())
    assert result['status'] == 'failed' and len(launches) == 3 and len(result['exits']) == 3
    assert len(result['pending_job_ids']) == 1 and maximum[0] == 3
    assert (root / 'dispatch_failure.json').exists()


def test_expired_deadline_starts_nothing(contract, monkeypatch):
    root = Path(contract['root']); (root / 'logs').mkdir()
    launches, _ = fake_queue(contract, monkeypatch)
    monkeypatch.setattr(r.time, 'time', lambda: r.DEADLINE + 1)
    result = r.run_stage(contract, root / 'execution_contract.json', 'train', r.time.monotonic())
    assert result['status'] == 'failed' and launches == []
    with pytest.raises(ValueError): r.run_stage(contract, '/c', 'test', 0)


def test_status_no_reinference_and_boot_identity(contract, monkeypatch):
    monkeypatch.setattr(r, 'gpu_snapshot', lambda _: {'compute_pids': []})
    result = r.status(contract)
    assert len(result['jobs']) == 10 and result['gpu']['compute_pids'] == []
    assert result['boot_id'] == 'new-boot'


def test_resume_original_wall_clock_budget_is_not_reset(contract):
    from datetime import datetime
    resume = next(j for j in contract['jobs'] if j['recovery_mode'] == 'resume')
    beginning = datetime.fromisoformat(resume['original_started_at']).timestamp()
    assert r.fit_deadline(contract, resume, beginning + 10000) == beginning + 129600
    fresh = next(j for j in contract['jobs'] if j['recovery_mode'] == 'fresh')
    assert r.fit_deadline(contract, fresh, beginning + 10000) == beginning + 10000 + 129600
    with pytest.raises(ValueError): r.fit_deadline(contract, {**resume, 'original_started_at': '2026-10-19T07:00:00'}, beginning)


def test_native_qualification_launches_three_real_fabricated_children_per_arm(contract, monkeypatch):
    """Keep subprocess I/O real; substitute only fabricated calculation and GPU reads."""
    import os
    import sys
    c = contract; root = Path(c['root']); source = Path(c['source_root'])
    science = r.scientific_contract(c)
    science['hosts']['5090']['python'] = sys.executable
    sp = Path(c['scientific_contract']['path']); r.write(sp, science)
    c['python'] = sys.executable
    c['scientific_contract'].update(raw_sha256=r.sha_file(sp), canonical_sha256=r.sha_json(science))
    c['immutable_originals']['files']['execution_contract.json'] = r.sha_file(sp)
    package = source / 'paper/scripts'
    (source / 'paper/__init__.py').write_text(''); (package / '__init__.py').write_text('')
    worker = package / 'run_titantpp_late_path18_recovery.py'
    worker.write_text("import argparse,json,time\np=argparse.ArgumentParser();p.add_argument('--recovery-contract');p.add_argument('--job');p.add_argument('--stage');a=p.parse_args();c=json.load(open(a.recovery_contract));j=next(j for j in c['jobs'] if j['id']==a.job);time.sleep(.03);print(json.dumps({'status':'passed','real_data_loaded':False,'candidate_name':j['arm'],'cuda_memory_peaks':{j['arm']:{'max_memory_reserved_bytes':100}}}))\n")
    c['operation_files'][str(worker.relative_to(source))] = r.sha_file(worker)
    c['operation_files_sha256'] = r.sha_json(c['operation_files'])
    cp = root / 'recovery_contract.json'; r.write(cp, c)
    expected = [c['python'], '-u', '-m', r.MODULE, '--contract', str(cp), '--mode', 'qualify']
    own = {'pid': os.getpid(), 'pgid': os.getpid(), 'start_ticks': 1, 'argv': expected, 'cwd': str(source)}
    spawned = []; original_popen = r.subprocess.Popen
    def popen(command, **kwargs):
        child = original_popen(command, **kwargs); spawned.append((child, command, str(kwargs['cwd']))); return child
    monkeypatch.setattr(r.subprocess, 'Popen', popen)
    monkeypatch.setattr(r, 'proc_identity', lambda pid: own if pid == os.getpid() else next(({'pid': child.pid, 'pgid': child.pid, 'start_ticks': child.pid, 'argv': command, 'cwd': cwd} for child, command, cwd in spawned if child.pid == pid and child.poll() is None), None))
    monkeypatch.setattr(r, 'runtime_check', lambda _: {'gpu': {'total_memory_bytes': 1000}})
    monkeypatch.setattr(r, 'gpu_state', lambda c, active_pids=(): {'compute_pids': list(active_pids)})
    monkeypatch.chdir(source)
    q = r.qualify(c, cp)
    assert q['native_worker_exit0_count'] == 6 and q['maximum_simultaneous_workers'] == 3
    assert all(len(rows) == 3 for rows in q['per_arm'].values())
    assert len(list(root.glob('qualification_*.exit.json'))) == 6
    assert all(row['stdout_bytes'] > 0 and row['stderr_bytes'] == 0 and row['actual_os_returncode'] == 0 for rows in q['per_arm'].values() for row in rows)
    with pytest.raises(ValueError, match='Existing'): r.qualify(c, cp)


def test_local_archived_science_can_validate_without_remote_path_reads(contract):
    science = r.scientific_contract(contract)
    Path(contract['scientific_contract']['path']).unlink()
    assert len(r.validate_contract(contract, science=science)) == 4
    with pytest.raises(FileNotFoundError): r.validate_contract(contract)
    with pytest.raises(ValueError): r.validate_contract(contract, science={**science, 'schema': 'changed'})


@pytest.mark.parametrize('outcome', ['exit0', 'nonzero', 'timeout'])
def test_supervisor_observes_real_dispatch_child_exit_and_never_retries(contract, monkeypatch, outcome):
    """Run real local children; replace only their GPU/data work with tiny stubs."""
    import sys
    c = contract; root = Path(c['root']); source = Path(c['source_root'])
    monkeypatch.setattr(r, 'SUPERVISOR_CLEANUP_GRACE_SECONDS', .05)
    science = r.scientific_contract(c)
    science['hosts']['5090']['python'] = sys.executable
    if outcome == 'timeout':
        deadline = r.time.time() + .45
        science['hosts']['5090']['absolute_host_deadline_unix'] = c['absolute_host_deadline_unix'] = deadline
        monkeypatch.setattr(r, 'DEADLINE', deadline)
    sp = Path(c['scientific_contract']['path']); r.write(sp, science)
    c['python'] = sys.executable
    c['scientific_contract'].update(raw_sha256=r.sha_file(sp), canonical_sha256=r.sha_json(science))
    c['immutable_originals']['files']['execution_contract.json'] = r.sha_file(sp)
    package = source / 'paper/scripts'
    (source / 'paper/__init__.py').write_text(''); (package / '__init__.py').write_text('')
    runtime = package / 'titantpp_late_path18_5090_recovery_runtime.py'
    action = "time.sleep(30)" if outcome == 'timeout' else ("sys.exit(7)" if outcome == 'nonzero' else "sys.exit(0)")
    runtime.write_text("import argparse,sys,time\np=argparse.ArgumentParser();p.add_argument('--contract');p.add_argument('--mode');a=p.parse_args()\nif a.mode=='dispatch':\n " + action + "\n")
    c['operation_files'][str(runtime.relative_to(source))] = r.sha_file(runtime)
    c['operation_files_sha256'] = r.sha_json(c['operation_files'])
    cp = root / 'recovery_contract.json'; r.write(cp, c)
    spawned = []
    original_popen = r.subprocess.Popen
    def popen(command, **kwargs):
        process = original_popen(command, **kwargs)
        spawned.append((process, command, str(kwargs['cwd'])))
        return process
    monkeypatch.setattr(r.subprocess, 'Popen', popen)
    monkeypatch.setattr(r, 'gpu_state', lambda *_: {'compute_pids': []})
    monkeypatch.setattr(r, 'proc_identity', lambda pid: next(({'pid': p.pid, 'pgid': p.pid, 'start_ticks': p.pid,
        'argv': cmd, 'cwd': cwd} for p, cmd, cwd in spawned if p.pid == pid and p.poll() is None), None))
    monkeypatch.chdir(source)
    if outcome == 'exit0': assert r.launch_dispatch(c, cp) == 0
    else:
        with pytest.raises(ValueError, match='dispatcher failed'): r.launch_dispatch(c, cp)
    assert [cmd[-1] for _, cmd, _ in spawned] == ['qualify', 'dispatch']
    record = r.read(root / 'dispatcher_exit.json')
    assert record['pid'] == spawned[1][0].pid
    assert record['command'] == spawned[1][1]
    assert record['actual_os_returncode'] == spawned[1][0].returncode
    assert record['actual_os_returncode'] == (0 if outcome == 'exit0' else (7 if outcome == 'nonzero' else -15))
    assert record['timeout'] == (outcome == 'timeout')
    assert all(p.poll() is not None for p, _, _ in spawned)
    with pytest.raises(ValueError, match='Existing'): r.launch_dispatch(c, cp)
    assert len(spawned) == 2


def test_local_remote_home_path_alias_does_not_break_relative_origin_binding(contract):
    science = r.scientific_contract(contract)
    old = science['hosts']['5090']['root']; remote = '/home/fabricated/workspace/preserved'
    science['hosts']['5090']['root'] = remote
    science['hosts']['5090']['source_root'] = remote + '/source'
    contract['scientific_contract']['path'] = remote + '/execution_contract.json'
    contract['scientific_contract']['canonical_sha256'] = r.sha_json(science)
    contract['immutable_originals']['root'] = remote
    for job in contract['jobs']:
        for key in ('origin_train_receipt', 'origin_selected_checkpoint', 'origin_resume_checkpoint'):
            if key in job: job[key]['path'] = remote + '/' + str(Path(job[key]['path']).relative_to(old))
    assert len(r.validate_contract(contract, science=science)) == 4


@pytest.mark.parametrize('parent_mode', ['qualify', 'dispatch'])
def test_supervisor_timeout_cleans_exact_owned_independent_worker_group(contract, monkeypatch, parent_mode):
    """A real grandchild has its own session, reproducing the orphan regression."""
    import os
    import signal
    import sys
    c = contract; root = Path(c['root']); source = Path(c['source_root'])
    monkeypatch.setattr(r, 'SUPERVISOR_CLEANUP_GRACE_SECONDS', .05)
    monkeypatch.setattr(r, 'OWNED_CLEANUP_GRACE_SECONDS', .5)
    science = r.scientific_contract(c)
    deadline = r.time.time() + .8
    science['hosts']['5090']['python'] = c['python'] = sys.executable
    science['hosts']['5090']['absolute_host_deadline_unix'] = c['absolute_host_deadline_unix'] = deadline
    monkeypatch.setattr(r, 'DEADLINE', deadline)
    sp = Path(c['scientific_contract']['path']); r.write(sp, science)
    c['scientific_contract'].update(raw_sha256=r.sha_file(sp), canonical_sha256=r.sha_json(science))
    c['immutable_originals']['files']['execution_contract.json'] = r.sha_file(sp)
    package = source / 'paper/scripts'
    (source / 'paper/__init__.py').write_text(''); (package / '__init__.py').write_text('')
    worker = package / 'run_titantpp_late_path18_recovery.py'
    worker.write_text('import time\ntime.sleep(30)\n')
    runtime = package / 'titantpp_late_path18_5090_recovery_runtime.py'
    stage = 'qualify' if parent_mode == 'qualify' else 'train'
    lease = root / 'qualification_fabricated.lease.json' if parent_mode == 'qualify' else root / 'run' / c['training_jobs'][0] / 'train_lease.json'
    runtime.write_text("import argparse,json,subprocess,sys,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--contract');p.add_argument('--mode');a=p.parse_args();c=json.load(open(a.contract))\nif a.mode==" + repr(parent_mode) + ":\n j=next(j for j in c['jobs'] if j['id']==c['training_jobs'][0]);cmd=[c['python'],'-u','-m'," + repr(r.WORKER_MODULE) + ",'--recovery-contract',a.contract,'--job',j['id'],'--stage'," + repr(stage) + "];w=subprocess.Popen(cmd,cwd=c['source_root'],env=None,start_new_session=True);lp=Path(" + repr(str(lease)) + ");lp.parent.mkdir(parents=True,exist_ok=True);v={'pid':w.pid,'start_ticks':w.pid,'job_id':j['id'],'stage':" + repr(stage) + ",'command':cmd,'boot_id':c['boot_id'],'contract_sha256':__import__('hashlib').sha256(json.dumps(c,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()};lp.write_text(json.dumps(v));w.wait();time.sleep(30)\n")
    for p in (worker, runtime): c['operation_files'][str(p.relative_to(source))] = r.sha_file(p)
    c['operation_files_sha256'] = r.sha_json(c['operation_files'])
    cp = root / 'recovery_contract.json'; r.write(cp, c)
    spawned = []; original_popen = r.subprocess.Popen
    def popen(command, **kwargs):
        process = original_popen(command, **kwargs); spawned.append((process, command, str(kwargs['cwd']))); return process
    monkeypatch.setattr(r.subprocess, 'Popen', popen)
    monkeypatch.setattr(r, 'gpu_state', lambda *_: {'compute_pids': []})
    def identity(pid):
        for process, command, cwd in spawned:
            if process.pid == pid and process.poll() is None:
                return {'pid': pid, 'pgid': pid, 'start_ticks': pid, 'argv': command, 'cwd': cwd}
        if lease.exists():
            v = r.read(lease)
            if v['pid'] == pid:
                try: os.kill(pid, 0)
                except ProcessLookupError: return None
                return {'pid': pid, 'pgid': pid, 'start_ticks': pid, 'argv': v['command'], 'cwd': str(source)}
        return None
    monkeypatch.setattr(r, 'proc_identity', identity)
    monkeypatch.chdir(source)
    try:
        with pytest.raises(ValueError, match='failed'): r.launch_dispatch(c, cp)
        assert lease.exists()
        value = r.read(lease)
        with pytest.raises(ProcessLookupError): os.kill(value['pid'], 0)
        cleanup_path = root / ('qualification_timeout_cleanup.json' if parent_mode == 'qualify' else 'supervisor_timeout_cleanup.json')
        cleanup = r.read(cleanup_path)
        assert len(cleanup['workers']) == 1
        record = cleanup['workers'][0]
        assert record['pid'] == value['pid'] and record['signals'] == [int(signal.SIGTERM)]
        assert record['owned_pid_absent'] and record['actual_os_returncode'] is None
        parent_exit = r.read(root / ('qualification_exit.json' if parent_mode == 'qualify' else 'dispatcher_exit.json'))
        assert parent_exit['timeout'] and parent_exit['actual_os_returncode'] == -15
        assert all(process.poll() is not None for process, _, _ in spawned)
    finally:
        if lease.exists():
            try: os.killpg(r.read(lease)['pid'], signal.SIGKILL)
            except ProcessLookupError: pass
        for process, _, _ in spawned:
            if process.poll() is None:
                process.kill(); process.wait(timeout=2)
