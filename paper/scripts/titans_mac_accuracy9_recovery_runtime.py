"""Bounded, isolated continuation of the three interrupted MAC9 fits.

The sealed scientific engine and original outputs remain immutable. The queue
reuses the reviewed process ownership, deadline and exit-record implementation.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from paper.scripts import titantpp_late_path18_5090_recovery_runtime as bounded
from paper.scripts import titans_mac_accuracy9_runtime as original_runtime

SCHEMA = 'titans_mac_accuracy9_recovery_v1'
STAGE_SCHEMA = 'titans_mac_accuracy9_recovery_stage_v1'
MODULE = 'paper.scripts.titans_mac_accuracy9_recovery_runtime'
WORKER_MODULE = 'paper.scripts.run_titans_mac_accuracy9_recovery'
HOST = '5090'
STAGES = ('reference', 'train', 'validation', 'test')
ORIGINAL_CANONICAL = '7e1cc94dcf913f8451cc8b9a9d39b6db4baf441ab0cf32aed25e85953f6137d0'
ORIGINAL_SOURCE = 'e89e4916f82471dd2dda406296d59e3951ea3694ba345d50ebfcf33c4c61ea00'
require, read, write = bounded.require, bounded.read, bounded.write
sha_file, sha_json, local_path = bounded.sha_file, bounded.sha_json, bounded.local_path
now, boot_id, proc_identity = bounded.now, bounded.boot_id, bounded.proc_identity
gpu_state, scientific_contract = bounded.gpu_state, bounded.scientific_contract
fit_deadline, _identity, _exit_record = bounded.fit_deadline, bounded._identity, bounded._exit_record
terminate_owned_child = bounded.terminate_owned_child
def artifact_manifest(root):
    excluded = {'artifact_manifest.json', 'final_receipt.json', 'dispatcher.stdout.log',
                'dispatcher.stderr.log', 'dispatcher_exit.json', 'supervisor.stdout.log', 'supervisor.stderr.log'}
    files = {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()
             and p.relative_to(root).parts[0] not in ('source', 'cache') and str(p.relative_to(root)) not in excluded}
    require(files, 'Empty terminal manifest')
    write(root / 'artifact_manifest.json', {'schema': SCHEMA, 'files': files,
        'excluded_lifecycle_files': sorted(excluded)}, exclusive=True)
    return {'path': 'artifact_manifest.json', 'sha256': sha_file(root / 'artifact_manifest.json'), 'file_count': len(files)}


def validate_contract(c, host=HOST):
    require(c.get('schema') == SCHEMA and c.get('host') == host == HOST, 'Only MAC5090 recovery admitted')
    original = scientific_contract(c)
    original_runtime.validate_contract(original, HOST)
    require(c['scientific_contract']['canonical_sha256'] == ORIGINAL_CANONICAL, 'Wrong original campaign')
    require(c['max_workers'] == 3 and c['limits'] == original['limits'], 'Budget/concurrency changed')
    target = original['hosts'][HOST]
    for k in ('python', 'gpu_uuid', 'runtime_expected', 'absolute_host_deadline_unix'):
        require(c[k] == target[k], 'Original Runtime/GPU/deadline changed: ' + k)
    # Host paths are lexical contracts here; resolving Linux /home on macOS
    # would spuriously rewrite it through macOS's /home automount symlink.
    root, old = Path(c['root']), Path(target['root'])
    require(root.is_absolute() and root != old and not root.is_relative_to(old)
            and not old.is_relative_to(root) and Path(c['source_root']) == root / 'source'
            and '..' not in root.parts, 'Unsafe recovery root')
    source = c['source']
    require(source['files'] == original['source']['files'] and len(source['files']) == 989
            and source['files_sha256'] == sha_json(source['files']) == ORIGINAL_SOURCE
            and source['source_revision'] == original['source']['base_git_revision'], 'Scientific source changed')
    require(c['input_files'] == original['input_files'], 'Input bytes changed')
    expected_env = {**target['environment'], 'PYTHONPATH': str(root / 'source'),
                    'MPLCONFIGDIR': str(root / 'cache/matplotlib'), 'XDG_CACHE_HOME': str(root / 'cache'),
                    'TORCHINDUCTOR_CACHE_DIR': str(root / 'cache/torchinductor')}
    require(c['environment'] == expected_env, 'Only private source/cache paths may change')
    operations = c['operation_files']
    require(set(operations) == {'paper/scripts/run_titans_mac_accuracy9_recovery.py',
            'paper/scripts/titans_mac_accuracy9_recovery_runtime.py'}
            and sha_json(operations) == c['operation_files_sha256']
            and not set(operations).intersection(source['files']) and c['operation_git_revision'], 'Unsealed operation files')
    require(c['boot_id'] and c['immutable_originals']['root'] == str(old)
            and c['immutable_originals']['files'] and c['authorization']['resume_three_only'] is True
            and c['authorization']['originals_immutable'] is True, 'Authorization/original inventory missing')
    originals = {j['id']: j for j in original['jobs']}
    require(len(c['jobs']) == len({j['id'] for j in c['jobs']}) == 9, 'Wrong MAC9 matrix')
    for job in c['jobs']:
        require(job['id'] in originals and all(job.get(k) == v for k, v in originals[job['id']].items()), 'Scientific job changed')
        expected = 'resume' if job['dataset'] == 'intermittent_frozen_5000' else 'reference'
        require(job['recovery_mode'] == expected, 'Only interrupted Intermittent may resume')
        keys = ('origin_resume_checkpoint',) if expected == 'resume' else ('origin_train_receipt', 'origin_train_exit')
        for key in keys:
            binding = job[key]; path = Path(binding['path'])
            require(path.is_relative_to(old) and '..' not in path.parts
                    and c['immutable_originals']['files'].get(str(path.relative_to(old))) == binding['sha256'], 'Unbound original')
        if expected == 'resume':
            require(type(job['resume_epoch']) is int and 0 < job['resume_epoch'] < 300 and job['original_started_at'], 'Missing saved epoch/start')
    for stage, mode, count in (('reference', 'reference', 6), ('training', 'resume', 3)):
        expected = [j['id'] for j in c['jobs'] if j['recovery_mode'] == mode]
        require(c[stage + '_jobs'] == expected and len(expected) == count, 'Unapproved work matrix')
    require(c['validation_jobs'] == c['test_jobs'] == list(originals), 'Full nine replay gate required')
    return c['jobs']


def verify_originals(c):
    return bounded.verify_originals(c)


def verify_files(c):
    validate_contract(c)
    original = scientific_contract(c)
    source, old = Path(c['source_root']), Path(original['hosts'][HOST]['source_root'])
    for name, digest in {**c['source']['files'], **c['input_files']}.items():
        require(sha_file(local_path(source, name)) == digest and sha_file(local_path(old, name)) == digest,
                'Frozen source/input changed: ' + name)
    for name, digest in c['operation_files'].items():
        require(sha_file(local_path(source, name)) == digest, 'Recovery source changed')
    require(boot_id() == c['boot_id'], 'Server boot changed since recovery admission')
    verify_originals(c)
    return original


def worker_command(c, contract_path, job, stage):
    require(stage in STAGES and job['id'] in c[{'train': 'training_jobs', 'reference': 'reference_jobs',
            'validation': 'validation_jobs', 'test': 'test_jobs'}[stage]], 'Unapproved stage/job')
    return [c['python'], '-u', '-m', WORKER_MODULE, '--contract', str(contract_path),
            '--contract-sha', sha_json(c), '--phase', stage, '--job-id', job['id']]


def verify_receipt(c, job, stage):
    root = Path(c['root']); folder = root / 'run' / job['id']
    path = folder / (stage + '_recovery_receipt.json'); value = read(path)
    require(value['schema'] == STAGE_SCHEMA and value['status'] == 'complete'
            and value['contract_sha256'] == sha_json(c) and value['scientific_contract_sha256'] == ORIGINAL_CANONICAL
            and value['stage'] == stage and value['job_id'] == job['id'], 'Recovery receipt identity changed')
    require(value['files'] and value['selected_checkpoint'] and value['checkpoint_sha256'], 'Empty result inventory')
    for name, digest in value['files'].items():
        p = local_path(root, name)
        require(p.is_relative_to(folder) and p != path.resolve() and sha_file(p) == digest, 'Recovery artifact changed')
    selected = Path(value['selected_checkpoint'])
    require(selected.is_relative_to(folder) and sha_file(selected) == value['checkpoint_sha256'], 'Selected binary changed')
    if stage in ('validation', 'test'):
        train = read(folder / 'train_receipt.json')
        require(all(value[k] == train[k] for k in ('selected_checkpoint', 'selected_epoch', 'checkpoint_sha256', 'state_sha256')), 'Selection changed')
    if stage == 'validation':
        require(value['validation_replay_passed'] is True, 'Replay failed')
    return {'path': str(path.relative_to(root)), 'sha256': sha_file(path)}


# Reuse the reviewed child ownership/timeout/draining queue byte-for-byte except
# its explicit four-stage job selector. No preserved module globals are changed.
_queue_source = inspect.getsource(bounded.run_stage)
_selector = 'keys = contract["training_jobs" if stage == "train" else "validation_jobs"]'
require(_queue_source.count(_selector) == 1, 'Reviewed bounded queue changed')
_queue_source = _queue_source.replace(_selector,
    'keys = contract[{"train":"training_jobs","reference":"reference_jobs","validation":"validation_jobs","test":"test_jobs"}[stage]]')
exec(compile(_queue_source, __file__ + ':bounded_queue', 'exec'), globals())


def validation_gate(c, stages):
    root = Path(c['root'])
    require(set(stages['validation']['exits']) == set(c['validation_jobs'])
            and stages['validation']['status'] == 'complete', 'All nine observed replay exits required')
    from paper.scripts.run_titans_mac_accuracy9_recovery import effective_scientific
    scientific = effective_scientific(c)
    records = {j['id']: {'validation_receipt_sha256': sha_file(root / 'run' / j['id'] / 'validation_receipt.json'),
        **{s + '_exit_sha256': sha_file(root / 'run' / j['id'] / (s + '_exit.json')) for s in ('train', 'validation')}} for j in c['jobs']}
    write(root / 'validation_gate.json', {'status': 'passed', 'host': HOST,
        'contract_sha256': sha_json(scientific), 'created_at': now(), 'jobs': records}, exclusive=True)


def dispatch(c, path):
    root = Path(c['root']); validate_contract(c)
    require(Path.cwd().resolve() == Path(c['source_root']).resolve(), 'Wrong source cwd')
    require(not any((root / n).exists() for n in ('dispatch_attempt.json', 'run', 'final_receipt.json', 'dispatch_failure.json')), 'Existing attempt; no retry')
    verify_files(c); gpu_state(c)
    write(root / 'dispatch_attempt.json', {'at': now(), 'contract_sha256': sha_json(c)}, exclusive=True)
    write(root / 'start.json', {'at': now(), 'pid': os.getpid(), 'identity': proc_identity(os.getpid()),
        'boot_id': boot_id(), 'contract_sha256': sha_json(c)}, exclusive=True)
    (root / 'logs').mkdir(exist_ok=False)
    began = time.monotonic(); stages = {}
    for stage in STAGES:
        verify_files(c)
        stages[stage] = run_stage(c, path, stage, began)
        if stages[stage]['status'] != 'complete':
            break
        if stage == 'validation':
            validation_gate(c, stages)
    pids = [e['pid'] for v in stages.values() for e in v['exits'].values()]
    gpu = gpu_state(c)
    absent = all(proc_identity(p) is None for p in pids)
    verify_files(c)
    complete = len(stages) == len(STAGES) and all(v['status'] == 'complete' for v in stages.values()) and absent
    final = {'schema': SCHEMA, 'status': 'complete' if complete else 'failed', 'completed_at': now(),
        'contract_sha256': sha_json(c), 'stages': stages, 'resumed_fit_count': len(stages.get('train', {}).get('receipts', {})),
        'preserved_completed_fit_count': len(stages.get('reference', {}).get('receipts', {})),
        'validation_replay_count': len(stages.get('validation', {}).get('receipts', {})),
        'candidate_test_count': len(stages.get('test', {}).get('receipts', {})),
        'held_out_test_evaluated': complete, 'gpu_after': gpu, 'owned_worker_pids_absent': absent,
        'immutable_originals_verification': verify_originals(c), 'dispatcher_own_exit_directly_observed': False}
    final['artifact_manifest'] = artifact_manifest(root)
    write(root / 'final_receipt.json', final, exclusive=True)
    return 0 if complete else 1


def cleanup_leased_workers(c, contract_path, *, grace=5.0):
    """Signal only exact leases of this attempt, then prove their disappearance.

    These workers are not supervisor children: process absence is observed, but
    their OS return codes must never be invented by the supervisor.
    """
    import signal
    root, source = Path(c['root']), Path(c['source_root'])
    require(boot_id() == c['boot_id'], 'Boot changed; no cleanup signals')
    identities, errors, actions = [], [], []
    jobs = {j['id']: j for j in c['jobs']}
    for lease_path in (root / 'run').glob('*/*_lease.json'):
        lease = read(lease_path); live = proc_identity(lease['pid'])
        if live is None:
            continue
        expected = worker_command(c, contract_path, jobs[lease['job_id']], lease['stage'])
        valid = (lease['boot_id'] == c['boot_id'] and lease['contract_sha256'] == sha_json(c)
            and lease['command'] == expected and live['start_ticks'] == lease['start_ticks']
            and live['argv'] == expected and live['cwd'] == str(source) and live['pgid'] == live['pid'])
        if not valid:
            errors.append({'lease_path': str(lease_path), 'error': 'unconfirmed_or_reused_pid', 'pid': lease['pid']})
            continue
        identities.append(live)
        try:
            os.killpg(live['pid'], signal.SIGTERM); actions.append({'pid': live['pid'], 'signal': 'SIGTERM'})
        except ProcessLookupError:
            actions.append({'pid': live['pid'], 'signal': 'SIGTERM', 'already_exited': True})
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(proc_identity(i['pid']) is not None for i in identities):
        time.sleep(.1)
    for identity in identities:
        live = proc_identity(identity['pid'])
        if live is None:
            continue
        if not all(live[k] == identity[k] for k in ('pid', 'start_ticks', 'argv', 'cwd', 'pgid')):
            errors.append({'pid': identity['pid'], 'error': 'identity_changed_during_cleanup'})
            continue
        try:
            os.killpg(live['pid'], signal.SIGKILL); actions.append({'pid': live['pid'], 'signal': 'SIGKILL'})
        except ProcessLookupError:
            actions.append({'pid': live['pid'], 'signal': 'SIGKILL', 'already_exited': True})
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(proc_identity(i['pid']) is not None for i in identities):
        time.sleep(.1)
    remaining = [i['pid'] for i in identities if proc_identity(i['pid']) is not None]
    gpu = bounded.gpu_snapshot(c['gpu_uuid'])
    owned_gpu = sorted({i['pid'] for i in identities}.intersection(gpu['compute_pids']))
    result = {'at': now(), 'actions': actions, 'errors': errors, 'remaining_pids': remaining,
        'owned_gpu_pids_remaining': owned_gpu, 'worker_own_os_exits_directly_observed': False,
        'status': 'passed' if not (errors or remaining or owned_gpu) else 'failed'}
    write(root / 'supervisor_worker_cleanup.json', result, exclusive=True)
    require(result['status'] == 'passed', 'Owned worker cleanup incomplete; evidence retained')
    return result


def supervise(c, path):
    root, source = Path(c['root']), Path(c['source_root'])
    verify_files(c); gpu_state(c)
    write(root / 'launch_attempt.json', {'at': now(), 'contract_sha256': sha_json(c)}, exclusive=True)
    command = [c['python'], '-u', '-m', MODULE, '--contract', str(path), '--contract-sha', sha_json(c), '--mode', 'dispatch']
    prefix = root / 'dispatcher'; started = now()
    with prefix.with_name(prefix.name + '.stdout.log').open('xb') as out, prefix.with_name(prefix.name + '.stderr.log').open('xb') as err:
        child = subprocess.Popen(command, cwd=source, env={**os.environ, **c['environment'], 'PYTHONDONTWRITEBYTECODE': '1'},
            stdout=out, stderr=err, start_new_session=True)
    identity = _identity(child, command, source)
    ticks = identity['start_ticks'] if identity else None
    write(root / 'dispatcher_launch.json', {'at': started, 'identity': identity, 'command': command,
        'contract_sha256': sha_json(c)}, exclusive=True)
    # The dispatcher enforces all worker deadlines. The supervisor waits locally
    # on5090, survives SSH disconnects in tmux, and directly records its child exit.
    timed_out, cleanup_error, cleanup_attempted = False, None, False
    def cleanup_once():
        nonlocal cleanup_error, cleanup_attempted
        cleanup_attempted = True
        try:
            cleanup_leased_workers(c, path)
        except Exception:
            cleanup_error = traceback.format_exc()
    try:
        code = child.wait(timeout=max(.001, min(c['limits']['queue_wall_seconds'], c['absolute_host_deadline_unix'] - time.time())))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            code = child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            cleanup_once()
            _identity(child, command, source, ticks)
            terminate_owned_child(child)
            code = child.returncode
    if code != 0 or timed_out:
        if not cleanup_attempted:
            cleanup_once()
    record = _exit_record(c, child, prefix, command, timeout=timed_out, started_at=started, start_ticks=ticks)
    record['cleanup_error'] = cleanup_error
    write(root / 'dispatcher_exit.json', record, exclusive=True)
    return 0 if code == 0 and not timed_out and cleanup_error is None else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--contract-sha', required=True)
    p.add_argument('--mode', choices=('supervise', 'dispatch', 'check'), required=True)
    a = p.parse_args(argv); c = read(a.contract)
    require(sha_json(c) == a.contract_sha, 'Pinned recovery contract changed')
    if a.mode == 'check':
        verify_files(c); print(json.dumps({'status': 'passed', 'gpu': gpu_state(c)})); return 0
    return supervise(c, a.contract) if a.mode == 'supervise' else dispatch(c, a.contract)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
