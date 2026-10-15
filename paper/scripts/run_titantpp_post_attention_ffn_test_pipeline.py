"""Supervise one native worker at a time; all six native Validation replays precede RAF6 Test."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

CLEANUP_TERM_SECONDS = 10
CLEANUP_KILL_SECONDS = 10


class PipelineTermination(RuntimeError):
    """Parent termination cancels only this pipeline's active child group."""
    def __init__(self, signum):
        super().__init__('Pipeline received signal ' + str(signum))
        self.signum = signum


def stop_owned_child(child):
    """Observe exit, or stop just the verified start_new_session child group."""
    code = child.poll()
    actions = []
    if code is not None:
        return {'actual_child_exit_observed': True, 'exit_code': child.wait(), 'cleanup_actions': actions}
    try:
        group = os.getpgid(child.pid)
    except ProcessLookupError:
        return {'actual_child_exit_observed': True, 'exit_code': child.wait(timeout=CLEANUP_KILL_SECONDS),
                'cleanup_actions': actions}
    if group != child.pid:
        raise RuntimeError('Active child process-group ownership changed; no signal sent')
    if child.poll() is None:
        try:
            os.killpg(group, signal.SIGTERM)
            actions.append('TERM_own_child_group')
        except ProcessLookupError:
            pass
    try:
        code = child.wait(timeout=CLEANUP_TERM_SECONDS)
    except subprocess.TimeoutExpired:
        # Poll and verify the same child group again before escalating; never
        # target a group inferred only from stale receipt or another process.
        if child.poll() is None:
            try:
                if os.getpgid(child.pid) != child.pid:
                    raise RuntimeError('Active child process-group changed before KILL; no signal sent')
                os.killpg(child.pid, signal.SIGKILL)
                actions.append('KILL_own_child_group')
            except ProcessLookupError:
                pass
        code = child.wait(timeout=CLEANUP_KILL_SECONDS)
    return {'actual_child_exit_observed': True, 'exit_code': code, 'cleanup_actions': actions}


def preserve_cancelled_child(root, c, active, cause):
    """Preserve cancellation separately; existing normal receipts stay intact."""
    outcome = {'actual_child_exit_observed': False, 'exit_code': None, 'cleanup_actions': []}
    try:
        outcome = stop_owned_child(active['child'])
    except BaseException as error:
        outcome['cleanup_error'] = type(error).__name__ + ': ' + str(error)
        code = active['child'].poll()
        if code is not None:
            outcome.update(actual_child_exit_observed=True, exit_code=code)
    name = active['job_id']
    split = active['split']
    path = root / 'results' / name / (split + '_worker_cancelled_exit_' + uuid.uuid4().hex + '.json')
    record = {'job_id': name, 'split': split, 'contract_sha256': c['contract_sha256'],
              'pid': active['child'].pid, 'ppid': os.getpid(), 'argv': active['argv'],
              'started_unix': active['started_unix'], 'completed_unix': time.time(),
              'cancellation_reason': type(cause).__name__, 'automatic_retry': False,
              'normal_exit_receipt_present': (root / 'results' / name / (split + '_worker_exit.json')).exists(),
              **outcome}
    write(path, record)
    return {'receipt_path': str(path), **outcome}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def load_evaluator(root):
    spec = importlib.util.spec_from_file_location('post_attention_ffn_native_contract', root / 'operation/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(root, host, split, popen=subprocess.Popen):
    root = Path(root).resolve()
    e = load_evaluator(root)
    c = e.verify_contract(root)
    a = e.verify_partial_metadata(root, c)
    e.require(host in c['runtimes'] and str(root) == a['remote_roots'][host], 'Wrong native host/root')
    e.require(split in ('validation', 'test'), 'Only frozen evaluation stages allowed')
    e.require(e.sha(__file__) == c['evaluator_integrity']['files']['operation/pipeline.py'], 'Pipeline seal differs')
    jobs = [j for j in c['jobs'] if j['evaluation_host'] == host]
    if split == 'test':
        e.verify_global_validation_gate(root, c)
    write(root / (split + '_pipeline.claim'), {'contract_sha256': c['contract_sha256'],
          'host': host, 'split': split, 'pid': os.getpid(), 'started_unix': time.time()})
    active = None
    completed = []
    previous = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise PipelineTermination(signum)
    signal.signal(signal.SIGTERM, terminated)
    try:
        for j in jobs:
            e.verify_contract(root)
            permit_path = 'permits/' + host + '_' + split + '.json'
            permit_sha = e.sha(root / permit_path)
            permit = e.verify_permit(root, c, j, split, permit_path, permit_sha)
            gpu = c['runtimes'][host]['gpu_uuid']
            inventory = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                                       check=True, capture_output=True, text=True).stdout.splitlines()
            e.require(inventory.count(gpu) == 1, 'Fixed GPU missing')
            compute = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                                     check=True, capture_output=True, text=True).stdout
            e.require(not any(row.split(',')[0].strip() == gpu for row in compute.splitlines()),
                      'Fixed GPU busy; no evaluation worker launched')
            budget = min(10800., permit['deadline_unix'] - time.time())
            e.require(budget > 0, 'Evaluation permit expired')
            argv = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(budget) + 's',
                    c['runtimes'][host]['python'], str(root / 'operation/evaluate.py'), '--root', str(root),
                    '--mode', 'worker', '--job', j['id'], '--split', split, '--device', 'cuda',
                    '--permit', permit_path, '--permit-sha', permit_sha]
            started = time.time()
            logs = root / 'logs'
            logs.mkdir(exist_ok=True)
            with (logs / (split + '_' + j['id'] + '.log')).open('xb') as stream:
                child = popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active = {'child': child, 'job_id': j['id'], 'split': split, 'argv': argv, 'started_unix': started}
                write(root / 'worker_launches' / (split + '_' + j['id'] + '.json'),
                      {'pid': child.pid, 'ppid': os.getpid(), 'argv': argv, 'started_unix': started,
                       'contract_sha256': c['contract_sha256'], 'job_id': j['id'], 'split': split})
                code = child.wait()
            write(root / 'results' / j['id'] / (split + '_worker_exit.json'),
                  {'job_id': j['id'], 'split': split, 'contract_sha256': c['contract_sha256'],
                   'actual_child_exit_observed': True, 'exit_code': code, 'pid': child.pid,
                   'started_unix': started, 'completed_unix': time.time()})
            active = None
            e.require(type(code) is int and code == 0, 'Evaluation worker failed; automatic retry forbidden')
            completed.append(j['id'])
            print(json.dumps({'stage': split, 'host': host, 'job_id': j['id'], 'actual_exit': code}), flush=True)
        write(root / (split + '_pipeline_complete.json'), {'status': 'complete', 'host': host,
              'split': split, 'contract_sha256': c['contract_sha256'], 'completed': completed,
              'completed_unix': time.time(), 'new_training': False})
        return 0
    except BaseException as error:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cancellation = preserve_cancelled_child(root, c, active, error) if active else None
        write(root / (split + '_pipeline_failure.json'), {'status': 'failed', 'message': str(error),
              'traceback': traceback.format_exc(), 'completed': completed, 'cancelled_active_child': cancellation,
              'automatic_retry': False, 'completed_unix': time.time()})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--host', choices=('5080', '5090'), required=True)
    p.add_argument('--split', choices=('validation', 'test'), required=True)
    args = p.parse_args()
    sys.exit(run(args.root, args.host, args.split))


