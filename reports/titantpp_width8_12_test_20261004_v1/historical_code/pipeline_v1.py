"""One serialized aggregate worker; every full Validation gate precedes Test."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def identity(row):
    return f"{row['dataset']}__{row['model']}__seed{row['seed']}"


def output(command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def check_receipt(folder, row, split, contract, campaign):
    receipt = read(folder / 'receipt.json')
    require(receipt['status'] == 'complete' and receipt['full_population'], 'Incomplete population')
    require(receipt['retrained'] is False and receipt['raw_predictions_written'] is False and receipt['parts'] == [], 'Training/private prediction output is forbidden')
    require(not (folder / 'predictions').exists(), 'Private prediction directory is forbidden')
    require(all(receipt['qualification'].values()), 'Causal qualification failed')
    for key in ('dataset', 'model', 'seed', 'selected_epoch', 'checkpoint_file_sha256', 'state_tensor_sha256'):
        require(receipt[key] == row[key], f'Checkpoint receipt identity differs: {key}')
    require(receipt['split'] == split and receipt['contract_sha256'] == sha(campaign / 'execution_contract.json'), 'Split/contract differs')
    for key, contract_key in [('runner_sha256', 'evaluator_sha256'), ('registry_sha256', 'registry_sha256'), ('dataset_manifest_sha256', 'dataset_manifest_sha256')]:
        require(receipt[key] == contract[contract_key], f'Input identity differs: {key}')
    population = contract['population_references'][row['dataset']][split]
    for key in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'expected_target_count'):
        require(receipt[key] == population[key], f'Original population/history differs: {key}')
    metrics = receipt['metrics']
    require(metrics['count'] == receipt['prediction_rows'] == population['expected_target_count'], 'Aggregate count differs')
    require(all(math.isfinite(metrics[key]) for key in ('qty_rmse', 'qty_mae', 'time_nll')), 'Nonfinite aggregate metric')
    require(sum(cell['count'] for cell in metrics['quantity_cells']) == metrics['count'], 'Bin counts differ')
    if split == 'validation':
        reference = contract['validation_references'][identity(row)]
        require(metrics['count'] == reference['count'], 'Validation count differs')
        for current, prior in [(metrics, reference), (metrics['tail'], reference['tail'])]:
            require(current['count'] == prior['count'], 'Validation/tail count differs')
            for key in ('qty_rmse', 'qty_mae', 'time_nll'):
                require((current[key] is None and prior[key] is None) or (current[key] is not None and prior[key] is not None and math.isclose(current[key], prior[key], rel_tol=contract['validation_gate']['rel_tol'], abs_tol=contract['validation_gate']['abs_tol'])), f'Validation endpoint replay differs: {identity(row)} {key}')
    limits = contract['resources']
    require(receipt['runtime']['cpu_peak_rss_bytes'] <= limits['rss_limit_gib'] * 1024**3, 'RSS limit exceeded')
    require(receipt['runtime']['cuda_peak_reserved_bytes'] <= limits['gpu_memory_limit_gib'] * 1024**3, 'GPU memory limit exceeded')
    return {'condition': identity(row), 'split': split, 'receipt_path': str((folder / 'receipt.json').relative_to(campaign)),
            'receipt_sha256': sha(folder / 'receipt.json'), 'metrics': metrics,
            'target_identity_sha256': receipt['target_identity_sha256'], 'truth_sha256': receipt['truth_sha256']}


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_condition(row, split, contract, deadline, campaign, root, device):
    folder = campaign / 'runs' / split / identity(row)
    require(not folder.exists(), 'Existing output preserved; automatic retries forbidden')
    if device == 'cuda':
        require(not output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']), 'GPU occupied')
    command = [sys.executable, str(HERE / 'evaluate.py'), '--root', str(root), '--registry', str(campaign / 'evaluation_registry.json'),
        '--dataset-manifest', str(campaign / 'dataset_manifest.json'), '--contract', str(campaign / 'execution_contract.json'),
        '--dataset', row['dataset'], '--model', row['model'], '--seed', str(row['seed']), '--split', split,
        '--device', device, '--output', str(folder), '--deadline-seconds', str(contract['resources']['condition_timeout_seconds'])]
    log = campaign / 'logs' / f'{split}__{identity(row)}.log'
    log.parent.mkdir(exist_ok=True)
    process = None
    try:
        with log.open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                require(time.time() < deadline, 'Campaign time limit exceeded')
                time.sleep(1)
            require(process.returncode == 0, f'Evaluator failed; inspect {log}')
    finally:
        if process is not None and process.poll() is None:
            stop_group(process)
    return check_receipt(folder, row, split, contract, campaign)


def main(args):
    campaign, root = Path(args.campaign).resolve(), Path(args.root).resolve()
    require(not (campaign / 'pipeline_status.json').exists(), 'Never overwrite an earlier campaign')
    contract = read(campaign / 'execution_contract.json')
    require(str(root) == contract['resources']['root'], 'Execution root differs from sealed deployment root')
    for relative, digest in read(campaign / 'code_seal.json')['files'].items():
        require(sha(root / relative) == digest, f'Evaluation source changed: {relative}')
    for name, key in [('evaluation_registry.json', 'registry_sha256'), ('dataset_manifest.json', 'dataset_manifest_sha256'), ('selected_binding.json', 'selected_binding_sha256')]:
        require(sha(campaign / name) == contract[key], f'Campaign input SHA mismatch: {name}')
    if args.device == 'cuda':
        require(output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']) == contract['resources']['gpu_uuid'], 'Wrong GPU UUID')
    os.environ.update(contract['environment'])
    rows = read(campaign / 'evaluation_registry.json')['rows']
    require(len(rows) == contract['validation_gate']['conditions'] and len({identity(row) for row in rows}) == len(rows), 'Batch condition count/uniqueness differs')
    completed, started = [], time.time()
    deadline = started + contract['resources']['campaign_timeout_seconds']
    try:
        for split in ('validation', 'test'):
            for row in rows:
                write(campaign / 'pipeline_status.json', {'status': 'running', 'split': split, 'condition': identity(row), 'completed': completed, 'pid': os.getpid()})
                result = run_condition(row, split, contract, deadline, campaign, root, args.device)
                completed.append(result)
                print(json.dumps({'completed': len(completed), 'split': split, 'condition': identity(row)}), flush=True)
            if split == 'validation':
                write(campaign / 'validation_gate.json', {'passed': True, 'conditions': len(rows), 'completed_before_test_unix': time.time(), 'contract_sha256': sha(campaign / 'execution_contract.json'), 'rows': completed.copy()})
        result = {'status': 'complete', 'completed': completed, 'conditions': len(rows), 'full_population_splits': len(rows) * 2,
                  'contract_sha256': sha(campaign / 'execution_contract.json'), 'elapsed_seconds': time.time() - started,
                  'new_training': False, 'selection_changed': False, 'raw_predictions_written': False}
        write(campaign / 'inference_completion.json', result)
        write(campaign / 'pipeline_status.json', result)
    except BaseException as error:
        write(campaign / 'pipeline_status.json', {'status': 'failed', 'completed': completed, 'exception': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.time() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--root', default=ROOT, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    with (args.campaign / 'campaign.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main(args)
