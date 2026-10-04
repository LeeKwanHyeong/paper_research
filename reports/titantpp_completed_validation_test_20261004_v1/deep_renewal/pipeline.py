"""One bounded CUDA worker: frozen full-validation gates, then frozen Test."""
from __future__ import annotations

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
ROOT = HERE.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def identity(row):
    return f"{row['dataset']}__{row['model']}__seed{row['seed']}"


def output(command):
    return subprocess.check_output(command, text=True, timeout=15).strip()


def load_predictions(folder):
    import polars as pl
    receipt = read(folder / 'receipt.json')
    assert receipt['status'] == 'complete' and receipt['full_population']
    assert receipt['retrained'] is False and all(receipt['qualification'].values())
    frames = []
    for part in receipt['parts']:
        assert sha(folder / part['path']) == part['sha256']
        frames.append(pl.read_parquet(folder / part['path']))
    frame = pl.concat(frames)
    assert frame.height == receipt['prediction_rows'] == receipt['expected_target_count']
    assert frame['target_index'].to_list() == list(range(frame.height))
    targets, truth = hashlib.sha256(), hashlib.sha256()
    for row in frame.iter_rows(named=True):
        targets.update((row['target_id'] + '\n').encode())
        values = [row[k] for k in receipt['truth_digest_fields']]
        truth.update((json.dumps(values, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode())
    assert targets.hexdigest() == receipt['target_identity_sha256']
    assert truth.hexdigest() == receipt['truth_sha256']
    return receipt, frame


def metrics(frame):
    import numpy as np
    error = (frame['predicted_raw_quantity'] - frame['raw_quantity']).to_numpy()
    times = frame['time_nll'].to_numpy()
    assert np.isfinite(error).all() and np.isfinite(times).all()
    return {'count': len(error), 'qty_mae': float(np.abs(error).mean()),
            'qty_rmse': float(np.sqrt(np.square(error).mean())),
            'qty_bias': float(error.mean()), 'time_nll': float(times.mean()),
            'qty_absolute_error_sum': float(np.abs(error).sum()),
            'qty_signed_error_sum': float(error.sum()), 'qty_sse': float(np.square(error).sum()),
            'time_nll_sum': float(times.sum())}


def check_prediction(folder, row, split, contract):
    receipt, frame = load_predictions(folder)
    for key in ('dataset', 'model', 'seed', 'selected_epoch', 'checkpoint_file_sha256', 'state_tensor_sha256'):
        assert receipt[key] == row[key], key
    assert receipt['split'] == split
    assert receipt['contract_sha256'] == sha(HERE / 'execution_contract.json')
    assert receipt['runner_sha256'] == contract['evaluator_sha256']
    assert receipt['registry_sha256'] == contract['registry_sha256']
    assert receipt['dataset_manifest_sha256'] == contract['dataset_manifest_sha256']
    paired = contract['population_references'][row['dataset']][split]
    for key in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'expected_target_count'):
        assert receipt[key] == paired[key], (row['dataset'], split, key)
    result = metrics(frame)
    if split == 'validation':
        reference = contract['validation_references'][identity(row)]
        assert result['count'] == reference['count']
        for key in ('qty_rmse', 'qty_mae', 'time_nll'):
            assert math.isclose(result[key], reference[key], rel_tol=1e-5, abs_tol=1e-5), (identity(row), key, result[key], reference[key])
    limits = contract['resources']
    assert receipt['runtime']['cpu_peak_rss_bytes'] <= limits['rss_limit_gib'] * 1024**3
    assert receipt['runtime']['cuda_peak_reserved_bytes'] <= limits['gpu_memory_limit_gib'] * 1024**3
    return {'receipt_sha256': sha(folder / 'receipt.json'), 'metrics': result,
            'target_identity_sha256': receipt['target_identity_sha256'], 'truth_sha256': receipt['truth_sha256']}


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_condition(row, split, contract, deadline):
    limits = contract['resources']
    folder = HERE / 'runs' / split / identity(row)
    assert not folder.exists(), 'Existing output is preserved; automatic retries are forbidden'
    assert not output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader']), 'GPU occupied'
    command = [sys.executable, str(HERE / 'evaluate.py'), '--root', str(ROOT),
        '--registry', str(HERE / 'evaluation_registry.json'), '--dataset-manifest', str(HERE / 'dataset_manifest.json'),
        '--contract', str(HERE / 'execution_contract.json'), '--dataset', row['dataset'],
        '--model', row['model'], '--seed', str(row['seed']), '--split', split, '--device', 'cuda',
        '--output', str(folder), '--deadline-seconds', str(limits['condition_timeout_seconds'])]
    log = HERE / 'logs' / f'{split}__{identity(row)}.log'
    log.parent.mkdir(exist_ok=True)
    process = None
    started, sampled = time.monotonic(), 0.0
    try:
        with log.open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                assert time.time() < deadline, 'Campaign time limit exceeded'
                assert time.monotonic() - started < limits['condition_timeout_seconds'] + 2, 'Condition time limit exceeded'
                if time.monotonic() - sampled >= 10:
                    sampled = time.monotonic()
                    ps = output(['ps', '-eo', 'pgid=,rss='])
                    rss = sum(int(line.split()[1]) * 1024 for line in ps.splitlines()
                              if len(line.split()) == 2 and int(line.split()[0]) == process.pid)
                    assert rss <= limits['rss_limit_gib'] * 1024**3, 'Owned process RSS limit exceeded'
                    gpu = int(output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits']))
                    assert gpu * 1024**2 <= limits['gpu_memory_limit_gib'] * 1024**3, 'GPU memory limit exceeded'
                    size = 0
                    for path in (HERE / 'runs').rglob('*'):
                        try:
                            if path.is_file():
                                size += path.stat().st_size
                        except FileNotFoundError:
                            pass
                    assert size <= limits['output_limit_gib'] * 1024**3, 'Output size limit exceeded'
                time.sleep(1)
            assert process.returncode == 0, f'Evaluator failed; inspect {log}'
    finally:
        if process is not None and process.poll() is None:
            stop_group(process)
    return check_prediction(folder, row, split, contract)


def main():
    assert not (HERE / 'pipeline_status.json').exists(), 'Never overwrite an earlier campaign'
    contract = read(HERE / 'execution_contract.json')
    for relative, expected in read(HERE / 'code_seal.json')['files'].items():
        assert sha(ROOT / relative) == expected, relative
    assert sha(HERE / 'evaluation_registry.json') == contract['registry_sha256']
    assert sha(HERE / 'dataset_manifest.json') == contract['dataset_manifest_sha256']
    assert output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader']) == contract['resources']['gpu_uuid']
    rows = [row for row in read(HERE / 'evaluation_registry.json')['rows'] if identity(row) not in contract['reuse_conditions']]
    assert len(rows) == contract['validation_gate']['new_conditions']
    rows.sort(key=lambda row: (contract['approval']['datasets'].index(row['dataset']), row['seed']))
    os.environ.update(contract['environment'])
    started = time.time()
    deadline = started + contract['resources']['campaign_timeout_seconds']
    completed = []
    try:
        for split in ('validation', 'test'):
            for row in rows:
                write('pipeline_status.json', {'status': 'running', 'split': split, 'condition': identity(row),
                    'completed': completed, 'started_unix': started, 'deadline_unix': deadline, 'pid': os.getpid()})
                result = run_condition(row, split, contract, deadline)
                completed.append({'condition': identity(row), 'split': split, **result})
                print(json.dumps({'completed': len(completed), 'split': split, 'condition': identity(row)}), flush=True)
            if split == 'validation':
                write('validation_gate.json', {'passed': True, 'conditions': len(rows), 'completed_before_test_unix': time.time(),
                    'contract_sha256': sha(HERE / 'execution_contract.json'), 'rows': completed.copy()})
        result = {'status': 'complete', 'completed': completed, 'new_split_count': len(rows) * 2,
            'reused_split_count': contract['reuse_split_count'], 'contract_sha256': sha(HERE / 'execution_contract.json'),
            'elapsed_seconds': time.time() - started, 'training': False, 'selection_changed': False}
        write('inference_completion.json', result)
        write('pipeline_status.json', result)
    except BaseException as error:
        write('pipeline_status.json', {'status': 'failed', 'completed': completed,
            'exception': type(error).__name__, 'message': str(error), 'elapsed_seconds': time.time() - started})
        raise


if __name__ == '__main__':
    with (HERE / 'campaign.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
