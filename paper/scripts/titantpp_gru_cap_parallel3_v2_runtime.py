"""Operational-only parallel handoff; the frozen cap trainer remains unchanged."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CONTRACT_RAW = '274ff80ef771788dc4652749bd2875716c7ddeb59ba0a1a0e497358545fe1fdc'
CAMPAIGN_RAW = '7ebd9df9a0a407aae258f5c009145adc33d27d1fa4e6b11f375510d7dd4d3159'
APPROVAL = '그럼 기존 계약처럼 서버당 3개씩 돌려.. 너 요즘 좀 삐꾸가되간다? 새로운 세션을 열어야하냐'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path if exclusive else path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with target.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    if not exclusive:
        target.replace(path)


def proc(pid):
    p = Path('/proc') / str(pid)
    before = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    argv = (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    after = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    need(before[1:3] == after[1:3] and before[19] == after[19], 'Process changed during probe')
    return dict(pid=pid, ppid=int(before[1]), pgrp=int(before[2]), argv=argv, start_ticks=before[19])


def absent(pid):
    try:
        proc(pid)
        return False
    except (FileNotFoundError, ProcessLookupError):
        return True


def owned_jobs(c, host):
    return [j for j in c['jobs'] if j['host'] == host]


def partition(c, host, state):
    jobs = owned_jobs(c, host)
    need(state.get('status') == 'running' and len(state.get('active_jobs', {})) == 1,
        'Handoff requires one verified original active fit')
    done = list(state['completed'])
    active = next(iter(state['active_jobs']))
    need({j['id'] for j in jobs[:len(done)]} == set(done) and jobs[len(done)]['id'] == active,
        'Original sequential prefix changed')
    remaining = jobs[len(done) + 1:]
    need(bool(remaining), 'No unstarted fits remain on this host')
    return [j['id'] for j in jobs[:len(done)]], active, remaining


def parent_terminal(root, c, handoff, host):
    """Only the exact planned queue barrier is an administrative termination."""
    root = Path(root)
    expected = handoff['retained_job_ids']
    failpath = root / 'failure.json'
    if not failpath.exists():
        state = read(root / 'status.json')
        need(set(state.get('active_jobs', {})) <= {expected[-1]}, 'Original queue escaped handoff')
        return False
    failure = read(failpath)
    first = handoff['remaining_jobs'][0]
    valid_reason = ((failure.get('type') == 'ValueError' and failure.get('message') in
        ('GPU occupied before next fit', 'Deadline reached or fresh-run rule failed')) or
        (failure.get('type') == 'FileExistsError' and str(root / 'claims' / (first['id'] + '.json'))
         in failure.get('message', '')))
    need(valid_reason and failure.get('cleanup') is None and failure.get('active_job') == first
        and failure.get('contract_sha256') == canonical(c) and set(failure.get('completed', {})) == set(expected)
        and sha(root / 'claims' / (first['id'] + '.json')) == handoff['first_claim_raw_sha256'],
        'Unexpected original failure: preserve and stop; no retry')
    for name in expected:
        item = read(root / 'worker_exits' / (name + '.json'))
        need(item['actual_child_exit_observed'] is True and item['returncode'] == 0
            and item['contract_sha256'] == canonical(c) and absent(item['pid'])
            and absent(item['worker_group_pid']), 'Original scientific fit not successfully terminal')
    exitpath = root / 'phases/train_exit.json'
    if not exitpath.exists():
        return False
    launch, exited = read(root / 'phases/train_launch.json'), read(exitpath)
    need(exited['os_exit_code'] == 1 and all(exited[k] == launch[k] for k in ('pid', 'ppid', 'argv', 'contract_sha256')),
        'Original administrative dispatcher exit identity differs')
    if not absent(exited['pid']) or not absent(exited['ppid']):
        return False
    return True


def load_campaign(root, c, host):
    path = root / 'operation/campaign.py'
    need(sha(path) == CAMPAIGN_RAW, 'Frozen campaign changed')
    spec = importlib.util.spec_from_file_location('frozen_cap_parallel_bridge', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.bootstrap(c, host)
    return module


def authorization(module, root, c, handoff, host):
    if (root / 'failure.json').exists():
        # Validates the exact administrative reason and successful retained fits;
        # the wrapper may still be writing its exit receipt at this instant.
        parent_terminal(root, c, handoff, host)
    need(not (root / 'qualification/failure.json').exists(), 'Qualification failure preserved')
    approval, start = read(root / 'approval.json'), read(root / 'start_permit.json')
    need(approval['approved'] is True and approval['hosts'] == [host]
        and approval['new_training_authorized'] is True and approval['contract_sha256'] == canonical(c)
        and start['host'] == host and start['contract_sha256'] == canonical(c)
        and start['approval_sha256'] == canonical(approval)
        and start['deadline_unix'] - start['started_at_unix'] == c['limits']['total_wall_seconds']
        and start['started_at_unix'] <= time.time() < start['deadline_unix'], 'Original authorization/deadline changed')
    module.verify_training_permit(c, root)
    return start


def route_frozen(frozen, root, job):
    """Only operational lease and progress paths change, never training logic."""
    original_read = frozen.read
    original_write = frozen.common.write_json
    def read_job(path):
        path = Path(path)
        return original_read(root / 'leases' / (job['id'] + '.json') if path == root / 'server_lease.json' else path)
    def write_job(path, value, *args, **kwargs):
        path = Path(path)
        if path == root / 'progress.json':
            path = root / 'parallel3_v2/progress' / (job['id'] + '.json')
        return original_write(path, value, *args, **kwargs)
    frozen.read = read_job
    frozen.common.write_json = write_job


def fit(module, root, c, host, handoff, name, deadline):
    job = next(j for j in handoff['remaining_jobs'] if j['id'] == name)
    source = inspect.getsource(module.run_fit)
    anchor = 'frozen.PulseBudget = frozen.PulseBudget'
    need(source.count(anchor) == 1, 'Frozen wiring anchor changed')
    module.__dict__['_parallel_route'] = route_frozen
    module.authorization = lambda contract, r, **_: authorization(module, Path(r), contract, handoff, host)
    exec(compile(source.replace(anchor, '_parallel_route(frozen, root, job)'), str(root / 'operation/campaign.py'), 'exec'), module.__dict__)
    module.run_fit(c, host, root, job, deadline)


def initial_handoff(module, root, c, host):
    state = read(root / 'status.json')
    done, active, remaining = partition(c, host, state)
    launch = read(root / 'phases/train_launch.json')
    actual = proc(launch['pid'])
    need(actual['argv'] == launch['argv'] and actual['ppid'] == launch['ppid']
        and state['supervisor_pid'] == launch['pid'], 'Original supervisor identity changed')
    group = state['active_jobs'][active]['worker_group_pid']
    pids = sorted(module.common.gpu_pids(c['hosts'][host]))
    need(len(pids) == 1 and proc(pids[0])['ppid'] == group, 'Original active GPU worker not uniquely verified')
    for j in remaining:
        need(not (root / 'run' / j['id']).exists() and not (root / 'claims' / (j['id'] + '.json')).exists(),
            'Handoff job already started/claimed')
    start = module.authorization(c, root)
    module.verify_training_permit(c, root)
    first = remaining[0]
    claim = {'job': first, 'contract_sha256': canonical(c),
        'deadline_unix': min(start['deadline_unix'], time.time() + c['limits']['per_condition_seconds']),
        'maximum_concurrency': 3, 'owner': 'parallel3_v2'}
    write(root / 'claims' / (first['id'] + '.json'), claim, True)
    h = {'host': host, 'contract_sha256': canonical(c), 'user_instruction': APPROVAL,
        'maximum_concurrency': 3, 'retained_job_ids': [*done, active], 'remaining_jobs': remaining,
        'original_launch': launch, 'original_active_guard': group,
        'original_active_gpu_pid': pids[0], 'original_active_probe': proc(pids[0]),
        'first_claim_raw_sha256': sha(root / 'claims' / (first['id'] + '.json')),
        'start_permit_raw_sha256': sha(root / 'start_permit.json'), 'created_unix': time.time(),
        'automatic_retry': False, 'science_contract_changed': False, 'Test': False}
    write(root / 'parallel3_v2/handoff.json', h, True)
    return h


def worker_exit(module, root, c, job, child, command):
    result = module._observe_exit(child, command, c, job, os.getpid())
    write(root / 'parallel3_v2/worker_exits' / (job['id'] + '.json'), result, True)
    need(result.get('returncode') == 0 and result.get('actual_child_exit_observed') is True
        and type(result.get('pid')) is int and result['ppid'] == child.pid, 'Parallel scientific worker failed; no retry')
    manifest = read(root / 'run' / job['id'] / 'terminal_manifest.json')
    need(manifest['scientific_success'] is True and manifest['job'] == job
        and manifest['contract_sha256'] == canonical(c), 'Parallel terminal identity changed')
    for name, digest in manifest['files'].items():
        need(sha(module.safe_relative(root / 'run' / job['id'], name)) == digest, 'Parallel terminal file SHA mismatch')
    return {'worker_exit_sha256': sha(root / 'parallel3_v2/worker_exits' / (job['id'] + '.json')),
        'terminal_manifest_sha256': sha(root / 'run' / job['id'] / 'terminal_manifest.json')}


def capacity(c, host, root, active, next_job):
    q = read(root / 'qualification/receipt.json')
    def peak(job):
        return q['synthetic'][job['dataset']+'__'+str(job['seed'])]['peak_allocated_bytes']
    out = subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True,timeout=15).stdout
    rows = [r.split(',') for r in out.splitlines()]
    row = next(r for r in rows if r[0].strip()==c['hosts'][host]['gpu_uuid'])
    total,used = float(row[1])*1024**2,float(row[2])*1024**2
    # Conservative admission includes workers still importing before CUDA appears.
    reserve = 1.25 * (peak(next_job) + sum(peak(i['job']) for i in active.values()))
    need(used + reserve < c['limits']['gpu_fraction_max'] * total,
        'Three-slot memory admission exceeds fixed GPU limit; no retry')


def dispatch(module, root, c, host, handoff):
    active, completed, pending = {}, {}, list(handoff['remaining_jobs'])
    started = {'pid': os.getpid(), 'ppid': os.getppid(), 'argv': [sys.executable, *sys.argv], 'contract_sha256': canonical(c)}
    write(root / 'parallel3_v2/supervisor.claim', started, True)
    def status(parent_done):
        write(root / 'parallel3_v2/status.json', {'status': 'running', **started, 'supervisor_pid': os.getpid(),
            'supervisor_argv': started['argv'], 'maximum_concurrency': 3, 'parent_terminal': parent_done,
            'active_jobs': {name: item['record'] for name, item in active.items()}, 'completed': completed,
            'pending_job_ids': [j['id'] for j in pending], 'updated_unix': time.time()})
    try:
        while pending or active or not parent_terminal(root, c, handoff, host):
            start = authorization(module, root, c, handoff, host)
            parent_done = parent_terminal(root, c, handoff, host)
            for name, item in list(active.items()):
                if item['child'].poll() is not None:
                    completed[name] = worker_exit(module, root, c, item['job'], item['child'], item['command'])
                    del active[name]
            # Count every admitted process, including startup before CUDA first appears.
            slots = 3 - len(active) - (0 if parent_done else 1)
            for _ in range(min(slots, len(pending))):
                pids = module.common.gpu_pids(c['hosts'][host])
                groups = {i['child'].pid for i in active.values()}
                if not parent_done:
                    groups.add(handoff['original_active_guard'])
                need(all(proc(pid)['pgrp'] in groups for pid in pids), 'Unowned GPU work detected')
                job = pending.pop(0)
                capacity(c,host,root,active,job)
                need(not (root / 'run' / job['id']).exists(), 'Fresh fit already exists; no retry')
                claimpath = root / 'claims' / (job['id'] + '.json')
                if job == handoff['remaining_jobs'][0]:
                    need(sha(claimpath) == handoff['first_claim_raw_sha256'], 'Queue barrier claim changed')
                    claim = read(claimpath)
                else:
                    claim = {'job': job, 'contract_sha256': canonical(c), 'deadline_unix':
                        min(start['deadline_unix'], time.time() + c['limits']['per_condition_seconds']),
                        'maximum_concurrency': 3, 'owner': 'parallel3_v2'}
                    write(claimpath, claim, True)
                deadline = claim['deadline_unix']
                need(time.time() < deadline, 'Condition budget exhausted')
                cmd = ['timeout', '--signal=TERM', '--kill-after=15s', str(max(1, int(deadline-time.time()))),
                    c['hosts'][host]['python'], str(root / 'parallel3_v2/runtime.py'), '--contract', str(root / 'execution_contract.json'),
                    '--host', host, '--mode', 'fit', '--job', job['id'], '--deadline', str(deadline)]
                log = root / 'parallel3_v2/logs' / (job['id'] + '.log'); log.parent.mkdir(exist_ok=True)
                with log.open('x') as stream:
                    child = subprocess.Popen(cmd, cwd=root / 'source', env={**os.environ, 'PYTHONHASHSEED': str(job['seed'])},
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record = {'job': job, 'worker_group_pid': child.pid, 'worker_guard_argv': cmd, 'condition_deadline_unix': deadline}
                active[job['id']] = {'job': job, 'child': child, 'command': cmd, 'record': record}
            for name, item in active.items():
                now = time.time()
                write(root / 'leases' / (name + '.json'), {'job': name, 'contract_sha256': canonical(c),
                    'issued_unix': now, 'expires_unix': min(now + 90, item['record']['condition_deadline_unix']),
                    'supervisor_pid': os.getpid(), 'worker_group_pid': item['child'].pid})
            status(parent_done)
            if pending or active or not parent_done:
                time.sleep(2)
        need(not module.common.gpu_pids(c['hosts'][host]), 'GPU not empty at parallel completion')
        write(root / 'parallel3_v2/status.json', {'status': 'complete', **started, 'supervisor_pid': os.getpid(),
            'supervisor_argv': started['argv'], 'active_jobs': {}, 'completed': completed, 'parent_terminal': True,
            'maximum_concurrency': 3, 'updated_unix': time.time()})
    except BaseException as exc:
        for item in active.values():
            module.stop_owned(item['child'])
            path = root / 'parallel3_v2/worker_exits' / (item['job']['id'] + '.json')
            if not path.exists():
                write(path, module._observe_exit(item['child'], item['command'], c, item['job'], os.getpid()), True)
        write(root / 'parallel3_v2/failure.json', {'type': type(exc).__name__, 'message': str(exc),
            'contract_sha256': canonical(c), 'automatic_retry': False, 'completed': completed, 'time': time.time()}, True)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--contract', type=Path, required=True); p.add_argument('--host', choices=('5080','5090'), required=True)
    p.add_argument('--mode', choices=('pipeline','dispatch','fit'), required=True)
    p.add_argument('--job'); p.add_argument('--deadline', type=float)
    a = p.parse_args(); c = read(a.contract); root = Path(c['hosts'][a.host]['root'])
    need(sha(a.contract) == CONTRACT_RAW and a.contract == root / 'execution_contract.json', 'Frozen contract changed')
    approval = read(root / 'parallel3_v2/approval.json')
    need(approval['user_instruction'] == APPROVAL and approval['maximum_concurrency'] == 3
        and approval['runtime_raw_sha256'] == sha(__file__) and approval['contract_raw_sha256'] == CONTRACT_RAW,
        'Parallel extension identity changed')
    need(not (root / 'parallel3_v2/failure.json').exists(), 'Parallel failure preserved; no retry')
    need(Path.cwd() == root / 'source' and Path(sys.executable).resolve() == Path(c['hosts'][a.host]['python']).resolve(), 'Native path changed')
    env = {**c['hosts'][a.host]['environment']}
    if a.mode == 'fit':
        env['PYTHONHASHSEED'] = str(next(j['seed'] for j in c['jobs'] if j['id'] == a.job))
    need(all(os.environ.get(k) == v for k,v in env.items()), 'Native environment changed')
    module = load_campaign(root, c, a.host)
    if a.mode == 'pipeline':
        h = initial_handoff(module, root, c, a.host)
        cmd = [c['hosts'][a.host]['python'], str(root/'parallel3_v2/runtime.py'), '--contract', str(a.contract), '--host', a.host, '--mode', 'dispatch']
        with (root/'parallel3_v2/dispatch.log').open('x') as stream:
            child = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            launch = dict(pid=child.pid, ppid=os.getpid(), argv=cmd, contract_sha256=canonical(c), started_unix=time.time())
            write(root/'parallel3_v2/launch.json', launch, True)
            code = child.wait()
        write(root/'parallel3_v2/exit.json', {**launch, 'os_exit_code':code,'completed_unix':time.time()}, True)
        need(code == 0, 'Parallel dispatcher failed; no retry')
        need(parent_terminal(root,c,h,a.host) and not module.common.gpu_pids(c['hosts'][a.host]), 'Terminal ownership unresolved')
        write(root/'parallel3_v2/complete.json', {'contract_sha256':canonical(c),'actual_dispatcher_exit':0,'host':a.host,'completed_unix':time.time()}, True)
    else:
        h = read(root / 'parallel3_v2/handoff.json')
        if a.mode == 'fit':
            fit(module,root,c,a.host,h,a.job,a.deadline)
        else:
            def stopped(signum,frame):
                raise InterruptedError('Parallel supervisor stopped')
            signal.signal(signal.SIGTERM, stopped)
            dispatch(module,root,c,a.host,h)


if __name__ == '__main__':
    main()
