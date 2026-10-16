"""Supervise one native worker at a time; all Validation4 precede Test4."""
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
    spec = importlib.util.spec_from_file_location('inter62_native_contract', root / 'operation/evaluate.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_gate(root, c, evaluator):
    receipts, exits = {}, {}
    for job in c['jobs']:
        name = job['id']
        receipt_path = root / 'results' / name / 'validation_replay.json'
        exit_path = root / 'results' / name / 'validation_worker_exit.json'
        receipt = evaluator.verify_validation_receipt(evaluator.read(receipt_path), c, job)
        actual = evaluator.read(exit_path)
        evaluator.require(actual['job_id'] == name and actual['contract_sha256'] == c['contract_sha256']
                          and actual['actual_child_exit_observed'] is True and type(actual['exit_code']) is int
                          and actual['exit_code'] == 0, 'Validation child did not actually exit0')
        receipts[name], exits[name] = evaluator.sha(receipt_path), evaluator.sha(exit_path)
    gate = {'schema': 'titantpp_components_inter62_validation_gate_v1', 'cohort': c['cohort'],
            'status': 'PASS', 'contract_sha256': c['contract_sha256'], 'job_count': 4,
            'validation_receipts': receipts, 'validation_worker_exits': exits,
            'created_unix': time.time(), 'original36_global_gate_modified': False}
    write(root / 'validation_gate.json', gate)
    evaluator.verify_global_validation_gate(root, c)
    return gate


def run(root, evaluator=None, popen=subprocess.Popen):
    root = Path(root).resolve()
    evaluator = evaluator or load_evaluator(root)
    c = evaluator.verify_contract(root)
    approval = evaluator.verify_partial_metadata(root, c)
    evaluator.require(str(root) == approval['authorization_root'], 'Pipeline only runs in approved native root')
    evaluator.require(evaluator.sha(__file__) in c['evaluator_integrity']['files'].values(), 'Pipeline not bound by evaluator integrity')
    python = c['runtimes']['5080']['python']
    write(root / 'pipeline.claim', {'pid': os.getpid(), 'ppid': os.getppid(), 'argv': sys.argv,
          'contract_sha256': c['contract_sha256'], 'started_unix': time.time(), 'automatic_retry': False})
    completed = []
    active_child = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def on_sigterm(signum, frame):
        raise PipelineTermination(signum)
    signal.signal(signal.SIGTERM, on_sigterm)
    try:
        for split in ('validation', 'test'):
            if split == 'test':
                make_gate(root, c, evaluator)
                print(json.dumps({'stage': 'validation4_gate', 'status': 'PASS', 'actual_exit0_count': 4}), flush=True)
            permit_path = 'permits/' + split + '.json'
            permit_sha = evaluator.sha(root / permit_path)
            for job in c['jobs']:
                evaluator.verify_contract(root)
                permit = evaluator.verify_permit(root, c, job, split, permit_path, permit_sha)
                if split == 'test':
                    evaluator.verify_global_validation_gate(root, c)
                inventory = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                                           check=True, capture_output=True, text=True).stdout.splitlines()
                gpu = c['runtimes']['5080']['gpu_uuid']
                evaluator.require(inventory.count(gpu) == 1, 'Fixed GPU missing before worker')
                compute = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                                         check=True, capture_output=True, text=True).stdout
                evaluator.require(not [row for row in compute.splitlines() if row.split(',')[0].strip() == gpu],
                                  'GPU busy before native replay; no worker started')
                argv = [python, str(root / 'operation/evaluate.py'), '--root', str(root), '--mode', 'worker',
                        '--job', job['id'], '--split', split, '--device', 'cuda', '--permit', permit_path,
                        '--permit-sha', permit_sha]
                budget = min(10800., permit['deadline_unix'] - time.time())
                evaluator.require(budget > 0, 'Evaluation permit expired')
                guard = ['/usr/bin/timeout', '--signal=TERM', '--kill-after=15s', str(budget) + 's', *argv]
                logs = root / 'logs'; logs.mkdir(exist_ok=True)
                started = time.time()
                with (logs / (split + '_' + job['id'] + '.log')).open('xb') as stream:
                    child = popen(guard, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    active_child = {'child': child, 'job_id': job['id'], 'split': split,
                                    'argv': guard, 'started_unix': started}
                    write(root / 'worker_launches' / (split + '_' + job['id'] + '.json'),
                          {'pid': child.pid, 'ppid': os.getpid(), 'argv': guard, 'started_unix': started,
                           'contract_sha256': c['contract_sha256'], 'job_id': job['id'], 'split': split})
                    exit_code = child.wait()
                record = {'job_id': job['id'], 'contract_sha256': c['contract_sha256'], 'split': split,
                          'actual_child_exit_observed': True, 'exit_code': exit_code, 'pid': child.pid,
                          'ppid': os.getpid(), 'argv': guard, 'started_unix': started, 'completed_unix': time.time()}
                write(root / 'results' / job['id'] / (split + '_worker_exit.json'), record)
                active_child = None
                evaluator.require(exit_code == 0, 'Native replay actual child exit is nonzero; automatic retry forbidden')
                receipt_name = 'validation_replay.json' if split == 'validation' else 'test_metrics.json'
                receipt = evaluator.read(root / 'results' / job['id'] / receipt_name)
                evaluator.require(receipt['status'] == 'complete' and receipt['job_id'] == job['id']
                                  and receipt['contract_sha256'] == c['contract_sha256']
                                  and receipt['split'] == split, 'Worker receipt identity differs')
                if split == 'validation':
                    evaluator.verify_validation_receipt(receipt, c, job)
                completed.append([split, job['id']])
                print(json.dumps({'stage': split, 'job_id': job['id'], 'actual_exit': exit_code,
                                  'prediction_rows': receipt['prediction_rows'],
                                  'elapsed_seconds': receipt['elapsed_seconds'], 'completed': len(completed)}), flush=True)
        write(root / 'pipeline_complete.json', {'status': 'complete', 'contract_sha256': c['contract_sha256'],
              'completed': completed, 'new_training': False, 'original36_global_gate_modified': False,
              'completed_unix': time.time()})
        return 0
    except BaseException as error:
        # Avoid reentrant parent TERM during the bounded cleanup. Restore the
        # caller's original handler in finally, on every outcome.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cancellation = preserve_cancelled_child(root, c, active_child, error) if active_child else None
        write(root / 'pipeline_failure.json', {'status': 'failed', 'completed': completed,
              'error': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc(),
              'contract_sha256': c['contract_sha256'], 'automatic_retry': False,
              'cancelled_active_child': cancellation, 'completed_unix': time.time()})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
