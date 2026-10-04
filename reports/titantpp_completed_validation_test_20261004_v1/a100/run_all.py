"""Serial 5080 inference: all twelve full Validation gates before twelve Test runs.

Run only after the parent has completed width16 and explicitly released this
campaign. No SSH, rental, training, resumption, retry, or checkpoint mutation.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from verify import (read, sha, require, save, inspect_predictions, compare_identity,
                    validation_gate, check_receipt)

HERE = Path(__file__).resolve().parent


def utc():
    return datetime.now(timezone.utc).isoformat()


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink())


def gpu_processes(uuid):
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True, timeout=10)
    rows = []
    for line in result.stdout.splitlines():
        fields = [s.strip() for s in line.split(',')]
        if len(fields) == 3 and fields[1] == uuid:
            rows.append({'pid': int(fields[0]), 'memory_bytes': int(fields[2]) * 1024**2})
    return rows


def process_group_usage(pgid):
    result = subprocess.run(['ps', '-eo', 'pid=,pgid=,rss='], check=True,
                            capture_output=True, text=True, timeout=10)
    pids, rss = set(), 0
    for line in result.stdout.splitlines():
        pid, group, kib = [int(s) for s in line.split()]
        if group == pgid:
            pids.add(pid)
            rss += kib * 1024
    return pids, rss


def static_gate(root, contract):
    require(sha(HERE / 'evaluation_registry.json') == contract['registry_sha256'], 'Registry changed')
    require(sha(HERE / 'selected_binding.json') == contract['selected_binding_sha256'], 'Selected binding changed')
    require(sha(HERE / 'comparison_binding.json') == contract['comparison_binding_sha256'], 'Comparison binding changed')
    require(sha(HERE / 'dataset_manifest.json') == contract['dataset_manifest_sha256'], 'Dataset manifest changed')
    require(sha(HERE / 'evaluate.py') == contract['evaluator']['sha256'], 'Evaluator changed')
    require(sha(HERE.parent / 'scope_and_approval.json') == contract['scope_and_approval_sha256'], 'Approval changed')
    for file, digest in contract['code_sha256'].items():
        require(sha(HERE / file) == digest, 'Campaign code changed: ' + file)
    registry = read(HERE / 'evaluation_registry.json')
    require(len(registry['rows']) == 12, 'Expected twelve conditions')
    for bundle in registry['bundles'].values():
        for path, digest in bundle['source_files'].items():
            require(sha(root / bundle['source_root'] / path) == digest, 'Frozen source changed: ' + path)
    for row in read(HERE / 'selected_binding.json')['rows']:
        for file, digest in row['original_files'].items():
            require(sha(root / file) == digest, 'Selected original evidence changed: ' + file)
    for row in read(HERE / 'comparison_binding.json')['rows']:
        for file, digest in row['files'].items():
            require(sha(root / file) == digest, 'Existing baseline prediction changed: ' + file)
    return registry


def run_condition(root, contract, row, split, deadline):
    limits = contract['resources']
    name = f"{row['dataset']}__{row['model']}__seed{row['seed']}"
    output = HERE / 'runs' / split / name
    require(not output.exists(), 'No implicit reuse/retry of inference output')
    require(not gpu_processes(limits['gpu_uuid']), 'Approved GPU has another process; wait for release')
    command = [sys.executable, str(HERE / 'evaluate.py'), '--root', str(root),
        '--registry', str(HERE / 'evaluation_registry.json'),
        '--dataset-manifest', str(HERE / 'dataset_manifest.json'),
        '--contract', str(HERE / 'execution_contract.json'), '--dataset', row['dataset'],
        '--model', row['model'], '--seed', str(row['seed']), '--split', split,
        '--device', 'cuda', '--output', str(output),
        '--deadline-seconds', str(limits['condition_timeout_seconds'])]
    log = HERE / 'logs' / f'{split}__{name}.log'
    log.parent.mkdir(exist_ok=True)
    condition_deadline = min(deadline, time.monotonic() + limits['condition_timeout_seconds'] + 5)
    peak_rss = peak_gpu = 0
    with log.open('x') as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                require(time.monotonic() < condition_deadline, 'Condition/campaign deadline reached')
                pids, rss = process_group_usage(child.pid)
                active = gpu_processes(limits['gpu_uuid'])
                require(all(r['pid'] in pids for r in active), 'Unrelated GPU process appeared; owned evaluation stopped')
                used_gpu = sum(r['memory_bytes'] for r in active)
                peak_rss, peak_gpu = max(peak_rss, rss), max(peak_gpu, used_gpu)
                require(rss <= limits['rss_limit_gib'] * 1024**3, 'Owned evaluator RSS cap exceeded')
                require(used_gpu <= limits['gpu_memory_limit_gib'] * 1024**3, 'Owned evaluator GPU cap exceeded')
                require(directory_bytes(HERE / 'runs') <= limits['output_limit_gib'] * 1024**3,
                        'Campaign output size cap exceeded')
                time.sleep(1)
            require(child.returncode == 0, 'Evaluator failed; inspect ' + str(log))
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            raise
    save(output / 'resource_receipt.json', {'peak_owned_process_group_rss_bytes': peak_rss,
        'peak_owned_gpu_bytes': peak_gpu, 'gpu_uuid': limits['gpu_uuid'], 'one_condition_at_a_time': True})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Explicit parent release after width16 completion')
    args = parser.parse_args()
    root = args.root.resolve()
    contract = read(HERE / 'execution_contract.json')
    registry = static_gate(root, contract)
    if not args.execute:
        print(json.dumps({'status': 'static_gate_passed', 'conditions': 12, 'inference_started': False}))
        return
    limits = contract['resources']
    require(str(root) == limits['root'], 'Only the approved dedicated remote workspace may execute')
    require(Path(sys.executable).resolve() == Path(limits['python']).resolve(), 'Unapproved Python Runtime')
    require(sys.platform.startswith('linux'), 'This campaign is restricted to the approved 5080 Linux Runtime')
    require(not (HERE / 'campaign_started.json').exists(), 'Campaign already attempted; no implicit restart')
    require(not (HERE / 'runs').exists(), 'Existing inference directory; no implicit restart')
    actual_uuid = subprocess.run(['nvidia-smi', '--id=' + limits['gpu_uuid'], '--query-gpu=uuid',
        '--format=csv,noheader'], check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    require(actual_uuid == limits['gpu_uuid'] and not gpu_processes(actual_uuid), 'GPU identity/availability failed')
    os.environ.update(CUDA_VISIBLE_DEVICES=limits['gpu_uuid'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                      OPENBLAS_NUM_THREADS='4', MPLCONFIGDIR=str(HERE / 'runtime_cache'))
    lock = (HERE / 'execution.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupted(signum, _frame):
        raise InterruptedError('Owned campaign received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save(HERE / 'campaign_started.json', {'started_utc': utc(), 'pid': os.getpid(),
        'contract_sha256': sha(HERE / 'execution_contract.json'), 'source_gate_passed': True,
        'training_gpu': 'A100 SXM', 'evaluation_gpu': 'RTX 5080', 'native_A100_runtime_reproduction': False})
    deadline = time.monotonic() + limits['campaign_timeout_seconds']
    bindings = {(r['dataset'], r['model'], r['seed']): r for r in read(HERE / 'selected_binding.json')['rows']}
    results, gates = [], []
    try:
        for row in registry['rows']:
            binding = bindings[(row['dataset'], row['model'], row['seed'])]
            output = run_condition(root, contract, row, 'validation', deadline)
            receipt, metrics = inspect_predictions(output, binding['tail_threshold'])
            check_receipt(receipt, row, contract, root)
            checks = validation_gate(metrics['overall'], binding['validation_reference'],
                contract['validation_gate']['rtol'], contract['validation_gate']['atol'])
            tail_checks = validation_gate(metrics['tail'], binding['validation_tail_reference'],
                contract['validation_gate']['rtol'], contract['validation_gate']['atol'])
            gate = {'dataset': row['dataset'], 'model': row['model'], 'seed': row['seed'],
                'passed': True, 'checks': checks, 'tail_checks': tail_checks,
                'receipt_sha256': sha(output / 'receipt.json'), 'selected_epoch': row['selected_epoch']}
            save(output / 'validation_gate.json', gate)
            gates.append(gate)
            results.append({**gate, 'split': 'validation', 'metrics': metrics,
                            'tail_threshold': binding['tail_threshold'], 'output_dir': str(output.relative_to(root))})
        save(HERE / 'validation_gate.json', {'passed': True, 'conditions': 12, 'rows': gates,
             'completed_before_test_utc': utc(), 'source_gate_passed': True,
             'contract_sha256': sha(HERE / 'execution_contract.json')})
        baseline_rows = {r['dataset']: r for r in read(HERE / 'comparison_binding.json')['rows']}
        baselines = {}
        for row in registry['rows']:
            binding = bindings[(row['dataset'], row['model'], row['seed'])]
            dataset = row['dataset']
            if dataset not in baselines:
                baseline = baseline_rows[dataset]
                require(sha(root / baseline['folder'] / 'receipt.json') == baseline['receipt_sha256'],
                        'Baseline receipt changed before Test')
                baselines[dataset] = inspect_predictions(root / baseline['folder'], binding['tail_threshold'])
            output = run_condition(root, contract, row, 'test', deadline)
            receipt, metrics = inspect_predictions(output, binding['tail_threshold'])
            check_receipt(receipt, row, contract, root)
            compare_identity(receipt, baselines[dataset][0])
            results.append({'dataset': dataset, 'model': row['model'], 'seed': 42, 'split': 'test',
                'selected_epoch': row['selected_epoch'], 'metrics': metrics, 'paired_with_width4_seed42': True,
                'width4_seed42_metrics': baselines[dataset][1], 'tail_threshold': binding['tail_threshold'],
                'receipt_sha256': sha(output / 'receipt.json'), 'output_dir': str(output.relative_to(root))})
        static_gate(root, contract)
        save(HERE / 'results.json', {'rows': results, 'training': False, 'selection_changed': False,
            'scope': contract['interpretation'], 'evaluation_gpu': contract['evaluation_gpu'],
            'sample_sd': None, 'not_scheduled': read(HERE / 'selected_binding.json')['not_scheduled']})
        save(HERE / 'completion_receipt.json', {'completed_utc': utc(), 'status': 'complete',
            'validation_conditions': 12, 'test_conditions': 12, 'all_target_truth_loader_pairs_match': True,
            'source_gate_before_and_after': True, 'results_sha256': sha(HERE / 'results.json'),
            'contract_sha256': sha(HERE / 'execution_contract.json'), 'last_binary_cpu_audit': 'not in this scope'})
    except BaseException as error:
        save(HERE / 'campaign_failure.json', {'failed_utc': utc(), 'error': str(error),
            'type': type(error).__name__, 'completed_split_rows': results, 'validation_gates': gates,
            'retry_performed': False, 'training_performed': False})
        raise


if __name__ == '__main__':
    main()
