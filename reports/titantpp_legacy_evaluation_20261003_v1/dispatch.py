"""Single-worker, bounded qualification and frozen legacy evaluation queue."""
import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def condition_id(row):
    return f"{row['dataset']}__{row['model']}__seed{row['seed']}"


def bounded_output(command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def load_campaign(path, contract_sha, timeout_seconds, now_epoch=None):
    """Called under the campaign lock; never reset an existing campaign clock."""
    stamp = time.time() if now_epoch is None else now_epoch
    path = Path(path)
    if path.exists():
        state = read(path)
        require(state['contract_sha256'] == contract_sha, 'Campaign contract changed')
        require(math.isfinite(state['started_epoch']) and math.isfinite(state['deadline_epoch']),
                'Invalid campaign clock')
        require(state['deadline_epoch'] == state['started_epoch'] + timeout_seconds,
                'Campaign deadline differs from contract')
        return state
    state = {'contract_sha256': contract_sha, 'started_epoch': stamp,
             'deadline_epoch': stamp + timeout_seconds, 'started_utc': now()}
    write(path, state)
    return state


def verify_gate(gate, root, seal_sha, contract_sha):
    require(gate.get('status') == 'passed', 'Qualification gate has not passed')
    require(gate.get('code_seal_sha256') == seal_sha, 'Qualification gate has stale code seal')
    require(gate.get('contract_sha256') == contract_sha, 'Qualification gate has stale contract')
    evidence = gate.get('evidence_hashes', {})
    require(bool(evidence), 'Qualification gate has no bound evidence')
    for relative, expected in evidence.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), 'Gate evidence path escapes root')
        require(sha(path) == expected, f'Qualification evidence changed: {relative}')


def output_bytes(folder):
    total = 0
    for base, _, files in os.walk(folder):
        for name in files:
            try:
                total += (Path(base) / name).stat().st_size
            except FileNotFoundError:
                pass  # An atomic part rename can race the scan.
    return total


def stop_owned_group(process):
    """The launcher and worker inherit the session we created, never another job."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The launcher may exit before its worker, so stop any surviving group member.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def supervisor_failure(directory, reason, exit_code):
    directory.mkdir(parents=True, exist_ok=True)
    failure = directory / 'failure.json'
    if not failure.exists():
        progress = read(directory / 'progress.json') if (directory / 'progress.json').exists() else {}
        write(failure, {'status': 'failed', 'exception': 'SupervisorStop', 'message': reason,
              'exit_code': exit_code, 'updated_utc': now(), 'parts': progress.get('parts', []),
              'prediction_rows': progress.get('prediction_rows', 0),
              'receipt_written': (directory / 'receipt.json').exists()})


def run_phase(args, contract, registry, seal_sha, contract_sha, campaign):
    if args.phase != 'qualification_cpu':
        gpu = bounded_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])
        require(gpu == contract['resources']['gpu_uuid'], 'Unexpected GPU')
        pids = bounded_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'])
        require(not pids, 'GPU occupied; do not share or terminate unrelated work')
    output = HERE / 'runs' / args.phase / args.attempt
    output.mkdir(parents=True, exist_ok=False)
    rows = registry['rows']
    if args.phase.startswith('qualification'):
        rows = [r for r in rows if r['seed'] == 42]
    elif args.phase == 'validation_full':
        rows = [r for r in rows if r['seed'] == 42 and r['model'] == 'titantpp_history_mlp']
    dataset_order = ['yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000', 'insta_market_basket']
    model_order = contract['approval']['models']
    rows = sorted(rows, key=lambda r: (dataset_order.index(r['dataset']), model_order.index(r['model']), r['seed']))
    conditions = [{k: r[k] for k in ['dataset', 'model', 'seed']} | {'output_dir': condition_id(r)} for r in rows]
    write(output / 'run_manifest.json', {'schema_version': 1, 'phase': args.phase,
          'scope': 'retrospective_locked_test_reevaluation' if args.phase == 'test' else 'validation_qualification',
          'root': str(ROOT), 'registry_sha256': contract['registry_sha256'],
          'dataset_manifest_sha256': contract['dataset_manifest_sha256'], 'code_seal_sha256': seal_sha,
          'contract_sha256': contract_sha, 'execution_contract_sha256': contract_sha,
          'execution_contract_path': '../../../execution_contract.json',
          'dataset_manifest_path': '../../../dataset_manifest.json',
          'registry_path': '../../../../titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json',
          'campaign_deadline_epoch': campaign['deadline_epoch'],
          'conditions': conditions})
    started = time.monotonic()
    failures, completed, attempted = [], [], []
    disk_sampled_at, disk_bytes = 0., 0
    campaign_stop = None
    limits = contract['resources']
    for row, spec in zip(rows, conditions):
        if time.time() >= campaign['deadline_epoch']:
            campaign_stop = 'campaign_deadline'
            break
        if time.monotonic() - disk_sampled_at >= 60:
            disk_bytes = output_bytes(HERE / 'runs')
            disk_sampled_at = time.monotonic()
        if disk_bytes > limits['output_limit_gib'] * 1024**3:
            campaign_stop = 'campaign_output_limit'
            break
        ident = condition_id(row)
        attempted.append(ident)
        write(output / 'status.json', {'status': 'running', 'updated_utc': now(), 'phase': args.phase,
              'dispatcher_pid': os.getpid(), 'current': ident, 'completed': completed, 'failures': failures,
              'total': len(rows), 'campaign_deadline_epoch': campaign['deadline_epoch']})
        command = [sys.executable, str(HERE / 'evaluate.py'), '--root', str(ROOT),
                   '--registry', contract['registry_path'], '--dataset-manifest', str(HERE / 'dataset_manifest.json'),
                   '--contract', str(HERE / 'execution_contract.json'), '--dataset', row['dataset'],
                   '--model', row['model'], '--seed', str(row['seed']), '--split', 'test' if args.phase == 'test' else 'validation',
                   '--device', 'cpu' if args.phase == 'qualification_cpu' else 'cuda', '--output', str(output / spec['output_dir']),
                   '--deadline-seconds', str(limits['condition_timeout_seconds'])]
        if args.phase.startswith('qualification'):
            command.extend(['--max-batches', '1' if args.phase == 'qualification_cpu' else '4'])
        reason, monitor_error, process = None, None, None
        interrupted = None
        with (output / (ident + '.log')).open('x') as log:
            try:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                condition_started, sampled_at = time.monotonic(), 0.
                while process.poll() is None:
                    if time.time() >= campaign['deadline_epoch']:
                        reason = 'campaign_deadline'
                    elif time.monotonic() - condition_started > limits['condition_timeout_seconds'] + 2:
                        reason = 'condition_deadline'
                    if time.monotonic() - disk_sampled_at >= 60:
                        disk_bytes = output_bytes(HERE / 'runs')
                        disk_sampled_at = time.monotonic()
                    if disk_bytes > limits['output_limit_gib'] * 1024**3:
                        reason = 'campaign_output_limit'
                    if sys.platform.startswith('linux') and time.monotonic() - sampled_at > 10:
                        sampled_at = time.monotonic()
                        ps = bounded_output(['ps', '-eo', 'pgid=,rss='])
                        rss = sum(int(line.split()[1]) * 1024 for line in ps.splitlines()
                                  if len(line.split()) == 2 and int(line.split()[0]) == process.pid)
                        if rss > limits['rss_limit_gib'] * 1024**3:
                            reason = 'owned_process_RSS_limit'
                        if args.phase != 'qualification_cpu':
                            memory = bounded_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'])
                            if int(memory) * 1024**2 > limits['gpu_memory_limit_gib'] * 1024**3:
                                reason = 'GPU_memory_limit'
                    if reason:
                        break
                    time.sleep(2)
            except BaseException as error:
                reason, monitor_error = 'supervisor_error', f'{type(error).__name__}: {error}'
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    interrupted = error
            finally:
                if process is not None and (reason or process.poll() is None):
                    stop_owned_group(process)
        directory = output / spec['output_dir']
        receipt = directory / 'receipt.json'
        exit_code = process.returncode if process is not None else None
        if not reason and exit_code == 0 and receipt.exists() and read(receipt)['status'] == 'complete':
            completed.append(ident)
        else:
            supervisor_failure(directory, monitor_error or reason or 'worker_failure', exit_code)
            failures.append({'condition': ident, 'exit_code': exit_code, 'reason': reason or 'worker_failure',
                             'failure_record': str((directory / 'failure.json').relative_to(ROOT))})
        print(json.dumps({'utc': now(), 'phase': args.phase, 'finished': ident, 'completed': len(completed), 'failed': len(failures), 'total': len(rows)}), flush=True)
        if reason in ('campaign_deadline', 'campaign_output_limit', 'supervisor_error'):
            campaign_stop = reason
            break
        if interrupted:
            break
    state = {'status': 'complete' if len(completed) == len(rows) else 'terminal_with_failures', 'updated_utc': now(),
             'phase': args.phase, 'dispatcher_pid': os.getpid(), 'completed': completed, 'failures': failures,
             'unstarted': [condition_id(r) for r in rows if condition_id(r) not in attempted],
             'total': len(rows), 'elapsed_seconds': time.monotonic() - started,
             'campaign_stop_reason': campaign_stop, 'campaign_deadline_epoch': campaign['deadline_epoch'],
             'run_manifest_sha256': sha(output / 'run_manifest.json')}
    write(output / 'status.json', state)
    write(output / 'terminal_manifest.json', state)
    return 0 if state['status'] == 'complete' else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['qualification_cpu', 'qualification_cuda', 'validation_full', 'test'])
    ap.add_argument('--attempt', default='attempt1')
    args = ap.parse_args()
    require(Path(args.attempt).name == args.attempt and args.attempt not in ('', '.', '..'), 'Invalid attempt name')
    contract = read(HERE / 'execution_contract.json')
    registry = read(ROOT / contract['registry_path'])
    seal = read(HERE / 'code_seal.json')
    seal_sha, contract_sha = sha(HERE / 'code_seal.json'), sha(HERE / 'execution_contract.json')
    for relative, expected in seal['files'].items():
        require(sha(ROOT / relative) == expected, f'Sealed file changed: {relative}')
    require(sha(ROOT / contract['registry_path']) == contract['registry_sha256'], 'Registry identity changed')
    require(sha(HERE / 'dataset_manifest.json') == contract['dataset_manifest_sha256'], 'Dataset identity changed')
    require(str(ROOT) == contract['resources']['root'], 'Dispatcher must run in the isolated approved remote root')
    if args.phase == 'test':
        verify_gate(read(HERE / 'qualification_gate.json'), ROOT, seal_sha, contract_sha)
        require(contract['approval']['test_inference_authorized'] is True, 'Test is not authorized')
    with (HERE / '.campaign.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another phase owns the single campaign worker') from error
        campaign = load_campaign(HERE / 'campaign_state.json', contract_sha,
                                 contract['resources']['campaign_timeout_seconds'])
        require(time.time() < campaign['deadline_epoch'], 'Campaign deadline already passed')
        # Translate termination into the same owned-process cleanup path.
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        # Keep the advisory lock through the entire phase, including supervisor cleanup.
        return run_phase(args, contract, registry, seal_sha, contract_sha, campaign)


if __name__ == '__main__':
    sys.exit(main())
