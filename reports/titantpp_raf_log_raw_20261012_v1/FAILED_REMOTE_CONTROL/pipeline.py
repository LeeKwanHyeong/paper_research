#!/usr/bin/env python3
"""Supervise only the authorized 18 frozen RAF Train/Validation evaluations.

Only this isolated bundle is writable. Child exits and unsuccessful claims are
preserved; a second invocation cannot restart or resume an evaluation.
"""
from __future__ import annotations

import argparse
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


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(root, relative):
    root = Path(root).resolve()
    require(not Path(relative).is_absolute(), 'Absolute sealed path')
    target = (root / relative).resolve()
    require(target.is_relative_to(root), 'Sealed path escapes root')
    return target


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def latest(root, value):
    print(json.dumps({'phase':value.get('phase'),'status':value.get('status'),
          'train_completed':len(value.get('train_completed', [])),
          'validation_completed':len(value.get('validation_completed', [])),
          'active_count':len(value.get('active', [])),'observed_unix':time.time()}),flush=True)
    target = root / 'control' / 'pipeline_latest.json'
    temp = target.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)


def verify_seals(root, seal_sha):
    seal_path = root / 'operation_integrity.json'
    require(sha(seal_path) == seal_sha, 'Operation seal raw SHA differs from launch authorization')
    seal = read(seal_path)
    for rel, expected in seal['files'].items():
        require(sha(safe(root, rel)) == expected, 'Operation file changed: ' + rel)
    inputs = read(root / 'input_integrity.json')
    for rel, expected in inputs['files'].items():
        require(sha(safe(root, rel)) == expected, 'Frozen input file changed: ' + rel)
    require(inputs['contract_sha256'] == seal['contract_sha256'], 'Input/operation contract differs')
    return seal


def load_adapter(root):
    path = root / 'control' / 'evaluate.py'
    spec = importlib.util.spec_from_file_location('frozen_test_operational_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def grouped(jobs):
    groups = {}
    for job in jobs:
        groups.setdefault((job['study'], job['dataset'], job['seed']), []).append(job)
    return [sorted(group, key=lambda j: j['arm']) for _, group in sorted(groups.items())]


def child_environment(contract, root, job):
    env = dict(os.environ)
    env.update(contract['runtime']['environment'])
    # Prior training-root values are process-local history only. Every import and
    # runtime cache for this evaluation is confined to its own sealed bundle.
    env['SOURCE_REVISION'] = contract['sources'][job['study']]['revision']
    env['PYTHONPATH'] = str(safe(root, contract['sources'][job['study']]['root']))
    env['MPLCONFIGDIR'] = str(root / 'results' / job['id'] / 'runtime_cache' / 'matplotlib')
    env['XDG_CACHE_HOME'] = str(root / 'results' / job['id'] / 'runtime_cache')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def gpu_health(contract):
    run = lambda args: subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()
    row = run(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']).splitlines()
    require(len(row) == 1 and row[0].split(',')[0].strip() == contract['runtime']['gpu_uuid'],
            'Pinned physical GPU differs')
    compute = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'])
    require(not compute, 'GPU is already in use; no evaluation launched')
    require(not Path('/var/run/reboot-required').exists(), 'Host reboot is pending')
    return {'observed_unix': time.time(), 'gpu': row[0], 'compute_processes': compute,
            'python_executable': sys.executable, 'reboot_required': False}


def native_probes(contract, root):
    routes = {}
    for job in contract['jobs']:
        routes.setdefault((job['study'], job['arm']), job)
    receipts = {}
    for (study, arm), job in sorted(routes.items()):
        name = study + '__' + arm
        dest = root / 'control' / 'native_probes' / name
        dest.mkdir(parents=True, exist_ok=False)
        argv = [contract['runtime']['python'], str(root / 'control' / 'evaluate.py'),
                '--root', str(root), '--native-probe', '--study', study, '--arm', arm, '--device', 'cuda']
        start = time.time()
        with (dest / 'probe.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen(argv, cwd=root, env=child_environment(contract, root, job),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                write_new(dest / 'launch.json', {'pid': proc.pid, 'ppid': os.getpid(), 'argv': argv,
                          'started_unix': start, 'contract_sha256': contract['contract_sha256']})
                rc = proc.wait(timeout=180)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                write_new(dest / 'worker_exit.json', {'exit_code': proc.returncode,
                          'actual_child_exit_observed': True, 'pid': proc.pid, 'ppid': os.getpid(),
                          'argv': argv, 'started_unix': start, 'completed_unix': time.time(),
                          'contract_sha256': contract['contract_sha256']})
        require(rc == 0, 'Native synthetic route probe failed: ' + name)
        lines = (dest / 'probe.log').read_text().splitlines()
        result = json.loads(lines[-1])
        require(result['status'] == 'PASS' and result['contract_sha256'] == contract['contract_sha256']
                and result['native_pinned_contract_checked'] is True
                and result['real_data_loaded'] is False and result['checkpoint_reads'] == 0
                and result['optimizer_updates'] == 0 and all(result['qualification'].values())
                and result.get('hooked_and_unhooked_all_original_outputs_bitwise_equal') is True
                and result.get('telemetry_finite') is True,
                'Incomplete native probe qualification: ' + name)
        write_new(dest / 'receipt.json', result)
        receipts[name] = sha(dest / 'receipt.json')
    require(len(receipts) == 3, 'Missing native history source/arm route')
    write_new(root / 'control' / 'native_qualification.json', {
        'status': 'PASS', 'contract_sha256': contract['contract_sha256'],
        'routes': receipts, 'synthetic_only': True, 'completed_unix': time.time()})


def run_group(contract, root, jobs, split, total_start, state):
    require(len(jobs) == 3 and len({(j['dataset'], j['seed']) for j in jobs}) == 1,
            'Exactly three arms of one dataset/seed per group')
    active = []
    def record_exit(entry, timed_out=False, aborted=False):
        job, proc, log, wall_start, mono_start, launch = entry
        rc = proc.wait()
        log.close()
        dest = root / 'results' / job['id']
        require(split in ('train','validation'), 'Only Train/Validation inference authorized')
        exit_name = split + '_worker_exit.json'
        write_new(dest / exit_name, {'job_id': job['id'], 'contract_sha256': contract['contract_sha256'],
                  'split': split, 'exit_code': rc, 'actual_child_exit_observed': True,
                  'pid': proc.pid, 'ppid': os.getpid(), 'argv': launch['argv'],
                  'started_unix': wall_start, 'completed_unix': time.time(),
                  'timed_out': timed_out, 'supervisor_aborted': aborted})
        active.remove(entry)
        return rc

    def stop(proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    try:
      for job in jobs:
        dest = root / 'results' / job['id']
        dest.mkdir(parents=True, exist_ok=True)
        argv = [contract['runtime']['python'], str(root / 'control' / 'evaluate.py'),
                '--root', str(root), '--job-id', job['id'], '--split', split, '--device', 'cuda']
        start = time.time()
        log = (dest / (split + '_worker.log')).open('x', encoding='utf-8')
        try:
            proc = subprocess.Popen(argv, cwd=root, env=child_environment(contract, root, job),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            log.close()
            raise
        launch = {'job_id': job['id'], 'contract_sha256': contract['contract_sha256'], 'split': split,
                  'pid': proc.pid, 'ppid': os.getpid(), 'argv': argv, 'started_unix': start}
        active.append((job, proc, log, start, time.monotonic(), launch))
        state.update(phase=split, active=[entry[5] for entry in active], observed_unix=time.time())
        write_new(dest / (split + '_launch.json'), launch)
      latest(root, state)
      failed = []
      while active:
        for entry in list(active):
            job, proc, log, wall_start, mono_start, launch = entry
            timed_out = (time.monotonic() - mono_start > contract['execution']['per_job_timeout_seconds']
                         or time.monotonic() - total_start > contract['execution']['total_timeout_seconds'])
            if timed_out and proc.poll() is None:
                # Terminate this exact supervised process group only. Never touch
                # other host owners, sessions, GPUs or administrative claims.
                stop(proc)
            rc = proc.poll()
            if rc is None:
                continue
            record_exit(entry, timed_out=timed_out)
            state['active'] = [e[5] for e in active]
            state['observed_unix'] = time.time()
            if rc != 0:
                failed.append({'job_id': job['id'], 'exit_code': rc, 'timed_out': timed_out})
            else:
                state[split + '_completed'].append(job['id'])
            latest(root, state)
        if active:
            time.sleep(1)
      require(not failed, 'Supervised worker failures (no automatic retry): ' + json.dumps(failed))
    finally:
        # Even launch/receipt/status I/O errors unwind every child launched by
        # this group. Keep observed exits or a cleanup error; never delete claims.
        cleanup_errors = []
        for entry in list(active):
            try:
                stop(entry[1])
                record_exit(entry, aborted=True)
            except BaseException as error:
                entry[2].close()
                cleanup_errors.append({'job_id': entry[0]['id'], 'error': repr(error),
                                       'observed_exit_code': entry[1].poll()})
        if cleanup_errors:
            write_new(root / 'control' / 'cleanup_failure.json', cleanup_errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--operation-seal-sha256', required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise InterruptedError('Supervisor received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    root = args.root.resolve()
    seal = verify_seals(root, args.operation_seal_sha256)
    adapter = load_adapter(root)
    contract = adapter.verify_contract(root)
    require(contract['contract_sha256'] == seal['contract_sha256'], 'Canonical contract differs')
    require(str(root) == contract['execution']['remote_root'], 'Not the authorized isolated remote root')
    require(Path(sys.executable).resolve() == Path(contract['runtime']['python']).resolve(), 'Wrong native Python')
    require(contract['execution']['maximum_workers'] == 3 and contract['execution']['automatic_retry'] is False,
            'Concurrency/retry contract changed')
    claim = {'contract_sha256': contract['contract_sha256'], 'operation_seal_sha256': args.operation_seal_sha256,
             'pid': os.getpid(), 'ppid': os.getppid(), 'argv': sys.argv, 'started_unix': time.time()}
    write_new(root / 'control' / 'pipeline_claim.json', claim)
    state = {**claim, 'status': 'running', 'phase': 'native_health', 'train_completed': [], 'validation_completed': [], 'active': [], 'observed_unix': time.time()}
    total_start = time.monotonic()
    try:
        write_new(root / 'control' / 'native_health.json', gpu_health(contract))
        state.update(phase='native_qualification', observed_unix=time.time())
        latest(root, state)
        native_probes(contract, root)
        groups = grouped(contract['jobs'])
        for split in ('train','validation'):
            for jobs in groups:
                run_group(contract, root, jobs, split, total_start, state)
        identities = {}
        receipts, exits = {}, {}
        for job in contract['jobs']:
            receipts[job['id']], exits[job['id']] = {}, {}
            for split in ('train','validation'):
                dest = root / 'results' / job['id']
                receipt = adapter.verify_receipt(root, contract, job, split)
                identity = (receipt['target_identity_sha256'], receipt['truth_sha256'])
                require(identities.setdefault((job['dataset'],split), identity) == identity,
                        'Population differs across seeds/arms')
                child_exit = read(dest / (split + '_worker_exit.json'))
                require(child_exit['exit_code'] == 0 and child_exit['actual_child_exit_observed'] is True,
                        'Actual worker exit0 missing')
                receipts[job['id']][split] = sha(dest / (split + '_metrics.json'))
                exits[job['id']][split] = sha(dest / (split + '_worker_exit.json'))
        require(len(state['train_completed']) == len(state['validation_completed']) == 9,
                'Train9 + Validation9 completion required')
        verify_seals(root, args.operation_seal_sha256)
        write_new(root / 'control' / 'pipeline_complete.json', {
            'status': 'complete', 'contract_sha256': contract['contract_sha256'],
            'train_count': 9, 'validation_count': 9, 'model_split_passes': 18,
            'actual_all_worker_exit0': True, 'new_fits': 0, 'test_inference': 0,
            'receipts': receipts, 'worker_exits': exits,
            'completed_unix': time.time(), 'elapsed_seconds': time.monotonic() - total_start})
        state.update(status='complete', phase='complete', active=[], observed_unix=time.time())
        latest(root, state)
        return 0
    except BaseException as error:
        write_new(root / 'control' / 'pipeline_failure.json', {
            'status': 'failed', 'phase': state['phase'], 'contract_sha256': contract['contract_sha256'],
            'error': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
            'completed_unix': time.time(), 'automatic_retry': False})
        state.update(status='failed', observed_unix=time.time())
        latest(root, state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
