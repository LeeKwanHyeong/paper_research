"""Finish only MAC9 Test after a separately verified terminal identity race.

Run as a flat operational script beside terminal_identity.py. The original v4
scientific source and contract are imported unchanged; v5 owns control records.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

SCHEMA = 'titans_mac_accuracy9_terminal_finish_v1'
UPSTREAM_SHA = '191d2534245ab6c02efff817fe9e21bb43e6589415b1839e22f1a7dec40797c2'
RACE_MESSAGE = 'ValueError: Child identity disappeared without observable OS exit'
OPERATION_NAMES = {'source/titans_mac_accuracy9_terminal_runtime.py',
                   'source/titans_mac_accuracy9_terminal_identity.py'}
TEST_NAMES = {'test_claim.json', 'test_launch.json', 'test_lease.json', 'test_exit.json',
              'test.intent.json', 'test_receipt.json', 'test_recovery_receipt.json',
              'test_predictions.jsonl.gz'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as f:
            f.write(content)
    else:
        tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
        tmp.write_text(content)
        os.replace(tmp, path)


def under(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative path')
    result = Path(root) / p
    require(result.resolve().is_relative_to(Path(root).resolve()), 'Path escaped its root')
    return result


def timestamp(value):
    d = datetime.fromisoformat(value)
    require(d.tzinfo is not None, 'Timestamp needs an explicit timezone')
    return d.timestamp()


def queue_deadline(f):
    return min(f['absolute_host_deadline_unix'],
               timestamp(f['queue_started_at']) + f['limits']['queue_wall_seconds'])


def snapshot(root):
    files = {}
    for p in sorted(Path(root).rglob('*')):
        require(not p.is_symlink(), 'Symlink in immutable upstream tree')
        if p.is_file():
            files[str(p.relative_to(root))] = sha_file(p)
    require(files, 'Empty upstream inventory')
    return files


def allowed_additions(c):
    return {'validation_gate.json'} | {
        'run/' + key + '/' + name for key in c['test_jobs'] for name in TEST_NAMES}


def verify_preserved(f, c, baseline, *, completed=False):
    require(baseline, 'Missing preexisting upstream files')
    current = snapshot(Path(f['upstream_root']))
    require(all(current.get(k) == v for k, v in baseline.items()),
            'Preexisting v4 file changed or disappeared')
    added = set(current) - set(baseline)
    allowed = allowed_additions(c)
    require(added <= allowed, 'Unapproved addition to v4 tree')
    if completed:
        require(added == allowed, 'Incomplete fixed Test addition matrix')
    return {k: current[k] for k in sorted(added)}


def validate_finish(f, c):
    root, upstream = Path(f['root']), Path(f['upstream_root'])
    require(f.get('schema') == SCHEMA and root.is_absolute() and upstream.is_absolute()
            and root != upstream and not root.is_relative_to(upstream)
            and not upstream.is_relative_to(root) and '..' not in root.parts,
            'Finish requires a separate operational root')
    require(Path(f['upstream_contract_path']) == upstream / 'recovery_contract.json'
            and f['upstream_contract_sha256'] == sha_json(c) == UPSTREAM_SHA
            and c['root'] == str(upstream), 'Original scientific contract changed')
    require(sha_file(f['upstream_contract_path']) == f['upstream_contract_raw_sha256'],
            'Original scientific contract bytes changed')
    pinned = f['upstream_evidence_files']
    require(isinstance(pinned, dict) and {'dispatcher_launch.json', 'dispatcher_exit.json',
            'dispatch_failure.json', 'validation_stage_receipt.json', 'final_receipt.json'} <= set(pinned),
            'Missing pinned upstream failure evidence')
    for name, digest in pinned.items():
        require(sha_file(under(upstream, name)) == digest, 'Pinned upstream evidence changed: ' + name)
    for k in ('python', 'boot_id', 'gpu_uuid', 'limits', 'absolute_host_deadline_unix'):
        require(f[k] == c[k], 'Original condition changed: ' + k)
    require(set(f['operation_files']) == OPERATION_NAMES, 'Wrong operation closure')
    for name, digest in f['operation_files'].items():
        require(sha_file(under(root, name)) == digest, 'Operation file changed: ' + name)
    auth = f['authorization']
    p = Path(auth['path'])
    require(auth.get('instruction') and p.is_absolute() and p.is_relative_to(root)
            and '..' not in p.parts and not p.is_symlink()
            and sha_file(p) == auth['sha256'], 'Unbound finish authorization')
    require(timestamp(f['queue_started_at']) > 0, 'Missing original queue start')


def dependencies(f):
    """Load only the already sealed upstream source, plus our sibling helper."""
    c = read(f['upstream_contract_path'])
    validate_finish(f, c)
    sys.dont_write_bytecode = True
    sys.path.insert(0, c['source_root'])
    from paper.scripts import titans_mac_accuracy9_completion_bridge as bridge
    from paper.scripts import run_titans_mac_accuracy9_recovery as worker
    require(Path(bridge.__file__).resolve().is_relative_to(Path(c['source_root']).resolve()),
            'Imported a different scientific source tree')
    helper_path = Path(f['root']) / 'source/titans_mac_accuracy9_terminal_identity.py'
    spec = importlib.util.spec_from_file_location('_mac9_terminal_identity', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return c, bridge, worker, helper


def verify_exit(c, bridge, exited, launch, command, prefix, *, key=None, stage=None, code=0):
    r = bridge.r
    require(type(exited.get('actual_os_returncode')) is int
            and exited['actual_os_returncode'] == code and exited.get('timeout') is False
            and exited.get('boot_id') == c['boot_id'] and exited.get('command') == command
            and type(exited.get('start_ticks')) is int and exited['start_ticks'] > 0,
            'Actual owned OS exit evidence is missing')
    if key is not None:
        require(exited.get('job_id') == key and exited.get('stage') == stage,
                'Exit job or stage changed')
    identity = launch.get('identity', launch)
    require(identity and identity['pid'] == exited['pid']
            and identity['start_ticks'] == exited['start_ticks']
            and launch['command'] == command, 'Launch and exit identity differ')
    require(timestamp(exited['observed_exit_at']) >= timestamp(exited['started_at']),
            'Exit predates launch')
    for stream in ('stdout', 'stderr'):
        path = Path(str(prefix) + '.' + stream + '.log')
        require(exited[stream + '_sha256'] == sha_file(path)
                and exited[stream + '_bytes'] == path.stat().st_size, 'Closed worker log changed')
    require(bridge.historical_identity_absent(exited), 'Previously owned process is still alive')


def verify_upstream(f, c, bridge, worker):
    """Admit only the pinned control race after every Validation worker exited 0."""
    r, root = bridge.r, Path(c['root'])
    old, identities = bridge.verify_files(c)
    require(bridge.terminal_ready(c, old, identities), 'Original v3 queue is not terminal')
    r.verify_originals(c)
    failure = read(root / 'dispatch_failure.json')
    require(failure.get('stage') == 'validation' and not failure.get('job_id')
            and failure.get('traceback', '').rstrip().endswith(RACE_MESSAGE)
            and 'in _identity' in failure['traceback'], 'Unapproved upstream failure')
    final = read(root / 'final_receipt.json')
    require(final['schema'] == r.SCHEMA and final['status'] == 'failed'
            and final['contract_sha256'] == sha_json(c)
            and set(final['stages']) == {'reference', 'train', 'validation'}
            and final['candidate_test_count'] == 0 and final['held_out_test_evaluated'] is False,
            'Unexpected upstream terminal state')
    exited = read(root / 'dispatcher_exit.json')
    launch = read(root / 'dispatcher_launch.json')
    require(exited.get('cleanup_error') is None and launch['contract_sha256'] == sha_json(c)
            and f['queue_started_at'] == exited['started_at'], 'Original dispatcher binding changed')
    expected = [c['python'], '-u', '-m', bridge.MODULE, '--contract',
                f['upstream_contract_path'], '--contract-sha', sha_json(c), '--mode', 'dispatch']
    verify_exit(c, bridge, exited, launch, expected, root / 'dispatcher', code=1)
    validation = read(root / 'validation_stage_receipt.json')
    require(validation == final['stages']['validation'], 'Validation stage differs from failed final')
    jobs = {j['id']: j for j in c['jobs']}
    stages = deepcopy(final['stages'])
    for stage, keys in (('reference', c['reference_jobs']), ('train', c['training_jobs']),
                        ('validation', c['validation_jobs'])):
        result = stages[stage]
        require(result['status'] == ('failed' if stage == 'validation' else 'complete')
                and set(result['exits']) == set(result['receipts']) == set(keys)
                and result.get('pending_job_ids') == [], 'Incomplete historical stage')
        if stage == 'validation':
            require(result['failure'] == {k: v for k, v in failure.items() if k != 'children_left_running'},
                    'Validation failure identity changed')
        for key in keys:
            folder = root / 'run' / key
            ex = read(folder / (stage + '_exit.json'))
            require(ex == result['exits'][key], 'Worker exit differs from stage receipt')
            contract, path = ((c, Path(f['upstream_contract_path'])) if stage == 'validation'
                              else (old, Path(old['root']) / 'recovery_contract.json'))
            command = r.worker_command(contract, path, jobs[key], stage)
            verify_exit(c, bridge, ex, read(folder / (stage + '_launch.json')), command,
                        root / 'logs' / (key + '.' + stage), key=key, stage=stage)
            require(r.verify_receipt(c, jobs[key], stage) == result['receipts'][key],
                    'Historical scientific receipt changed')
    children = failure.get('children_left_running', {})
    expected_children = {k for k in c['training_jobs'] if k.endswith(('__seed42', '__seed62'))}
    require(set(children) == expected_children and len(children) == 2, 'Wrong failed live-child set')
    for key, pid in children.items():
        ex = validation['exits'][key]
        require(ex['pid'] == pid and timestamp(ex['observed_exit_at']) >= timestamp(failure['at']),
                'Failed active child did not subsequently drain with exit zero')
    # Check the scientific train/Validation material without running inference.
    scientific = worker.effective_scientific(c)
    for job in scientific['jobs']:
        worker.science.verify_train_receipt(scientific, '5090', job, root,
                                           read(root / 'run' / job['id'] / 'train_receipt.json'))
    stages['validation'] = {**stages['validation'], 'status': 'complete', 'failure': None,
        'imported_without_execution': True, 'control_failure_preserved': {
            'path': str(root / 'validation_stage_receipt.json'),
            'sha256': sha_file(root / 'validation_stage_receipt.json')}}
    r.gpu_state(c)
    return stages


def preflight(f, c, bridge, worker):
    validate_finish(f, c)
    require(time.time() < queue_deadline(f), 'Inherited queue deadline expired')
    stages = verify_upstream(f, c, bridge, worker)
    root, upstream = Path(f['root']), Path(c['root'])
    admission_path = root / 'admission.json'
    require(not any((root / name).exists() for name in
        ('dispatch_attempt.json', 'test_stage_receipt.json', 'final_receipt.json', 'failure.json')),
        'Existing terminal attempt; no retry')
    current = snapshot(upstream)
    require(not (set(current) & allowed_additions(c)), 'Existing gate or Test output; no rerun')
    if admission_path.exists():
        admission = read(admission_path)
        require(admission['contract_sha256'] == sha_json(f)
                and admission['upstream_contract_sha256'] == sha_json(c)
                and admission['preexisting_v4_files'] == current
                and admission['preexisting_v4_files_sha256'] == sha_json(current)
                and admission['validated_stages'] == stages, 'Admission changed before dispatch')
        return admission
    admission = {'schema': SCHEMA, 'status': 'passed', 'created_at': now(),
        'contract_sha256': sha_json(f), 'upstream_contract_sha256': sha_json(c),
        'preexisting_v4_files': current, 'preexisting_v4_files_sha256': sha_json(current),
        'validated_stages': stages,
        'validation_failure_preserved': {'path': str(upstream / 'dispatch_failure.json'),
                                         'sha256': sha_file(upstream / 'dispatch_failure.json')},
        'upstream_dispatcher_exit': {'path': str(upstream / 'dispatcher_exit.json'),
                                    'sha256': sha_file(upstream / 'dispatcher_exit.json')},
        'immutable_originals_verification': bridge.r.verify_originals(c),
        'new_training_worker_count': 0, 'new_validation_worker_count': 0,
        'cache_environment_overrides': {k: process_environment(f, c)[k] for k in
            ('MPLCONFIGDIR', 'XDG_CACHE_HOME', 'TORCHINDUCTOR_CACHE_DIR')}}
    write(admission_path, admission)
    return admission


def process_environment(f, c, seed=None):
    cache = Path(f['root']) / 'cache'
    result = {**os.environ, **c['environment'], 'PYTHONDONTWRITEBYTECODE': '1',
              'PYTHONUNBUFFERED': '1', 'MPLCONFIGDIR': str(cache / 'matplotlib'),
              'XDG_CACHE_HOME': str(cache), 'TORCHINDUCTOR_CACHE_DIR': str(cache / 'torchinductor')}
    if seed is not None:
        result['PYTHONHASHSEED'] = str(seed)
    return result


def test_queue(f, c, bridge, helper):
    """Use original scientific commands; write operational status only in v5."""
    r, root, upstream = bridge.r, Path(f['root']), Path(c['root'])
    source, path = Path(c['source_root']), Path(f['upstream_contract_path'])
    identity = helper.make_terminal_identity(proc_identity=r.proc_identity,
                                             initial_identity=bridge.initial_identity)
    pending = [next(j for j in c['jobs'] if j['id'] == k) for k in c['test_jobs']]
    active, exits, receipts, failure = {}, {}, {}, None
    while active or (pending and failure is None):
        for key, rec in list(active.items()):
            child = rec['process']
            timed_out = child.poll() is None and (
                time.monotonic() - rec['began'] >= c['limits']['evaluation_wall_seconds']
                or time.time() >= queue_deadline(f))
            if timed_out:
                identity(child, rec['command'], source, rec['start_ticks'])
                if child.poll() is None:
                    r.terminate_owned_child(child)
            code = child.poll()
            if code is None:
                continue
            ex = {**r._exit_record(c, child, rec['prefix'], rec['command'], timeout=timed_out,
                  started_at=rec['started_at'], start_ticks=rec['start_ticks']),
                  'job_id': key, 'stage': 'test', 'operational_root': str(root)}
            write(upstream / 'run' / key / 'test_exit.json', ex)
            exits[key] = ex
            del active[key]
            try:
                require(code == 0 and not timed_out, 'Scientific Test worker failed or timed out')
                receipts[key] = r.verify_receipt(c, rec['job'], 'test')
            except Exception:
                failure = failure or {'job_id': key, 'at': now(), 'traceback': traceback.format_exc()}
        if failure is None:
            try:
                require(time.time() < queue_deadline(f), 'Inherited queue deadline expired')
                for rec in active.values():
                    identity(rec['process'], rec['command'], source, rec['start_ticks'])
                r.gpu_state(c, [rec['process'].pid for rec in active.values()])
                for _ in range(min(3 - len(active), len(pending))):
                    require(time.time() < queue_deadline(f), 'Inherited queue deadline expired before launch')
                    job = pending.pop(0)
                    key, folder = job['id'], upstream / 'run' / job['id']
                    write(folder / 'test_claim.json', {'job_id': key, 'stage': 'test',
                        'at': now(), 'contract_sha256': sha_json(c), 'finish_contract_sha256': sha_json(f)})
                    command = r.worker_command(c, path, job, 'test')
                    prefix, started = root / 'logs' / (key + '.test'), now()
                    prefix.parent.mkdir(exist_ok=True)
                    with Path(str(prefix) + '.stdout.log').open('xb') as out, Path(str(prefix) + '.stderr.log').open('xb') as err:
                        child = subprocess.Popen(command, cwd=source, env=process_environment(f, c, job['seed']),
                            stdout=out, stderr=err, start_new_session=True)
                    rec = {'process': child, 'job': job, 'command': command, 'prefix': prefix,
                           'started_at': started, 'began': time.monotonic(), 'start_ticks': None}
                    active[key] = rec
                    live = identity(child, command, source)
                    rec['start_ticks'] = live['start_ticks'] if live else None
                    lease = {'job_id': key, 'stage': 'test', 'pid': child.pid,
                        'start_ticks': rec['start_ticks'], 'command': command, 'started_at': started,
                        'boot_id': c['boot_id'], 'contract_sha256': sha_json(c),
                        'finish_contract_sha256': sha_json(f), 'operational_root': str(root)}
                    write(folder / 'test_launch.json', lease)
                    write(folder / 'test_lease.json', lease)
                    write(root / 'leases' / (key + '.json'), lease)
            except Exception:
                failure = failure or {'at': now(), 'traceback': traceback.format_exc()}
        if failure and not (root / 'failure.json').exists():
            write(root / 'failure.json', failure)
        write(root / 'status.json', {'status': 'draining_after_failure' if failure else 'running',
            'stage': 'test', 'observed_at': now(), 'contract_sha256': sha_json(f),
            'active': {k: {n: v[n] for n in ('start_ticks', 'command', 'started_at')} | {'pid': v['process'].pid}
                       for k, v in active.items()}, 'pending_job_ids': [j['id'] for j in pending],
            'actual_worker_exits': exits, 'failure': failure}, exclusive=False)
        if active or (pending and failure is None):
            time.sleep(2)
    result = {'stage': 'test', 'status': 'complete' if not failure and len(exits) == 9 else 'failed',
              'exits': exits, 'receipts': receipts, 'failure': failure,
              'pending_job_ids': [j['id'] for j in pending]}
    write(root / 'test_stage_receipt.json', result)
    return result


def artifact_manifest(root):
    excluded = {'artifact_manifest.json', 'final_receipt.json', 'dispatcher.stdout.log',
                'dispatcher.stderr.log', 'dispatcher_exit.json', 'supervisor.stdout.log',
                'supervisor.stderr.log'}
    files = {str(p.relative_to(root)): sha_file(p) for p in sorted(root.rglob('*')) if p.is_file()
             and p.relative_to(root).parts[0] != 'cache' and str(p.relative_to(root)) not in excluded}
    require(files, 'Empty finish manifest')
    write(root / 'artifact_manifest.json', {'schema': SCHEMA, 'files': files,
                                          'excluded_lifecycle_files': sorted(excluded)})
    return {'path': 'artifact_manifest.json', 'sha256': sha_file(root / 'artifact_manifest.json'),
            'file_count': len(files)}


def dispatch(f, c, bridge, worker, helper):
    admission = preflight(f, c, bridge, worker)
    r, root, upstream = bridge.r, Path(f['root']), Path(c['root'])
    require(Path.cwd().resolve() == Path(c['source_root']).resolve(), 'Wrong dispatcher cwd')
    write(root / 'dispatch_attempt.json', {'at': now(), 'contract_sha256': sha_json(f)})
    write(root / 'start.json', {'at': now(), 'identity': r.proc_identity(os.getpid()),
                              'boot_id': r.boot_id(), 'contract_sha256': sha_json(f)})
    stages = deepcopy(admission['validated_stages'])
    r.validation_gate(c, stages)
    worker.science.verify_validation_gate(worker.effective_scientific(c), '5090', upstream)
    verify_preserved(f, c, admission['preexisting_v4_files'])
    stages['test'] = test_queue(f, c, bridge, helper)
    complete = stages['test']['status'] == 'complete'
    added = verify_preserved(f, c, admission['preexisting_v4_files'], completed=complete)
    bridge.verify_files(c)
    worker.science.verify_validation_gate(worker.effective_scientific(c), '5090', upstream)
    absent = all(bridge.historical_identity_absent(ex) for stage in stages.values()
                 for ex in stage['exits'].values())
    gpu = r.gpu_state(c)
    complete = complete and absent
    final = {'schema': SCHEMA, 'status': 'complete' if complete else 'failed', 'completed_at': now(),
        'contract_sha256': sha_json(f), 'upstream_contract_sha256': sha_json(c),
        'admission': {'path': 'admission.json', 'sha256': sha_file(root / 'admission.json')},
        'test_stage_receipt': {'path': 'test_stage_receipt.json', 'sha256': sha_file(root / 'test_stage_receipt.json')},
        'stages': stages, 'new_training_worker_count': 0, 'new_validation_worker_count': 0,
        'candidate_test_count': len(stages['test']['receipts']),
        'actual_scientific_worker_exit0_count': 18 + sum(ex['actual_os_returncode'] == 0
            and ex['timeout'] is False for ex in stages['test']['exits'].values()),
        'owned_worker_pids_absent': absent, 'preexisting_v4_files_unchanged': True,
        'added_v4_files': added, 'gpu_after': gpu,
        'immutable_originals_verification': r.verify_originals(c),
        'dispatcher_own_exit_directly_observed': False}
    final['artifact_manifest'] = artifact_manifest(root)
    write(root / 'final_receipt.json', final)
    return 0 if complete else 1


def cleanup_test_leases(f, c, bridge):
    """Signal only exact Test leases created by this v5 dispatcher."""
    r, root = bridge.r, Path(f['root'])
    require(r.boot_id() == f['boot_id'], 'Boot changed; no cleanup')
    actions, errors, owned = [], [], []
    jobs = {j['id']: j for j in c['jobs']}
    for path in sorted((root / 'leases').glob('*.json')):
        lease = read(path)
        key = lease['job_id']
        require(key in c['test_jobs'] and path.name == key + '.json'
                and lease['stage'] == 'test' and lease['operational_root'] == str(root)
                and lease['finish_contract_sha256'] == sha_json(f)
                and lease['contract_sha256'] == sha_json(c) and lease['boot_id'] == f['boot_id'],
                'Lease is not owned by this Test attempt')
        require(lease == read(Path(c['root']) / 'run' / key / 'test_lease.json'), 'Test lease copies differ')
        command = r.worker_command(c, Path(f['upstream_contract_path']), jobs[key], 'test')
        require(lease['command'] == command, 'Cleanup command changed')
        live = r.proc_identity(lease['pid'])
        if live is None:
            continue
        if not (live['pid'] == live['pgid'] == lease['pid'] and live['start_ticks'] == lease['start_ticks']
                and live['argv'] == command and live['cwd'] == c['source_root']):
            errors.append({'pid': lease['pid'], 'error': 'unconfirmed_identity'})
            continue
        owned.append(live)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for item in owned:
            live = r.proc_identity(item['pid'])
            if live is None:
                continue
            if live != item:
                errors.append({'pid': item['pid'], 'error': 'identity_changed'})
                continue
            try:
                os.killpg(item['pid'], sig)
                actions.append({'pid': item['pid'], 'signal': sig.name})
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(r.proc_identity(i['pid']) is not None for i in owned):
            time.sleep(.1)
    remaining = [i['pid'] for i in owned if r.proc_identity(i['pid']) is not None]
    result = {'status': 'passed' if not errors and not remaining else 'failed', 'at': now(),
              'actions': actions, 'errors': errors, 'remaining_pids': remaining,
              'worker_own_os_exits_directly_observed': False}
    write(root / 'supervisor_worker_cleanup.json', result)
    require(result['status'] == 'passed', 'Test-only cleanup could not prove absence')
    return result


def supervise(f, c, bridge, worker, helper, path):
    preflight(f, c, bridge, worker)
    root, source, r = Path(f['root']), Path(c['source_root']), bridge.r
    write(root / 'launch_attempt.json', {'at': now(), 'contract_sha256': sha_json(f)})
    command = [f['python'], '-u', str(root / 'source/titans_mac_accuracy9_terminal_runtime.py'),
               '--contract', str(path), '--contract-sha', sha_json(f), '--mode', 'dispatch']
    prefix, started = root / 'dispatcher', now()
    identity = helper.make_terminal_identity(proc_identity=r.proc_identity,
                                             initial_identity=bridge.initial_identity)
    with Path(str(prefix) + '.stdout.log').open('xb') as out, Path(str(prefix) + '.stderr.log').open('xb') as err:
        child = subprocess.Popen(command, cwd=source, env=process_environment(f, c),
                                 stdout=out, stderr=err, start_new_session=True)
    live = identity(child, command, source)
    ticks = live['start_ticks'] if live else None
    write(root / 'dispatcher_launch.json', {'at': started, 'identity': live, 'command': command,
                                          'contract_sha256': sha_json(f)})
    timed_out, cleanup_error = False, None
    try:
        child.wait(timeout=max(.001, queue_deadline(f) - time.time()))
    except subprocess.TimeoutExpired:
        timed_out = True
        identity(child, command, source, ticks)
        if child.poll() is None:
            r.terminate_owned_child(child)
    if timed_out or child.returncode != 0:
        try:
            cleanup_test_leases(f, c, bridge)
        except Exception:
            cleanup_error = traceback.format_exc()
    ex = r._exit_record(c, child, prefix, command, timeout=timed_out,
                        started_at=started, start_ticks=ticks)
    ex['cleanup_error'] = cleanup_error
    write(root / 'dispatcher_exit.json', ex)
    return 0 if child.returncode == 0 and not timed_out and cleanup_error is None else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--contract-sha', required=True)
    p.add_argument('--mode', choices=('preflight', 'dispatch', 'supervise'), required=True)
    a = p.parse_args(argv)
    f = read(a.contract)
    require(sha_json(f) == a.contract_sha and a.contract.resolve() == Path(f['root']) / 'finish_contract.json',
            'Pinned finish contract changed')
    c, bridge, worker, helper = dependencies(f)
    if a.mode == 'preflight':
        result = preflight(f, c, bridge, worker)
        print(json.dumps({'status': 'passed', 'contract_sha256': sha_json(f),
                          'preexisting_v4_file_count': len(result['preexisting_v4_files'])}))
        return 0
    return dispatch(f, c, bridge, worker, helper) if a.mode == 'dispatch' else supervise(f, c, bridge, worker, helper, a.contract)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
