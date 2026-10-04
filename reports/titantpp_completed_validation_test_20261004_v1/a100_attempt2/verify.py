"""Bounded prediction verification and CPU-only selected-checkpoint qualification."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
METRICS = ('qty_mae', 'qty_rmse', 'time_nll')
TRUTH_FIELDS = ('target_index', 'target_id', 'entity_id', 'seq', 'split', 'recorded_date',
                'time_bucket', 'site_id', 'raw_quantity', 'recorded_gap', 'history_length',
                'last_quantity', 'mean_quantity')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4 << 20), b''):
            digest.update(data)
    return digest.hexdigest()


def compact(value):
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def save(path, value):
    path = Path(path)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def finish(sums):
    count, absolute, squared, time_sum, bias = sums
    return {'count': count, 'qty_mae': absolute / count if count else None,
            'qty_rmse': math.sqrt(squared / count) if count else None,
            'time_nll': time_sum / count if count else None,
            'qty_bias': bias / count if count else None}


def inspect_predictions(folder, threshold):
    """Read one parquet part at a time; recompute identity/truth and all metrics."""
    import numpy as np
    import polars as pl
    folder = Path(folder)
    receipt = read(folder / 'receipt.json')
    require(receipt.get('status') == 'complete' and receipt.get('full_population') is True and
            receipt.get('retrained') is False and receipt.get('max_batches') is None,
            'Receipt is not a complete immutable evaluation')
    require(tuple(receipt['truth_digest_fields']) == TRUTH_FIELDS, 'Truth schema changed')
    require(receipt['qualification'] and all(v is True for v in receipt['qualification'].values()),
            'Qualification did not pass')
    require(receipt['split'] in ('validation', 'test') and
            receipt['heldout_read'] == (receipt['split'] == 'test'), 'Receipt split mismatch')
    sums = {'overall': [0, 0., 0., 0., 0.], 'tail': [0, 0., 0., 0., 0.]}
    identities, truth = hashlib.sha256(), hashlib.sha256()
    count, seen, previous = 0, set(), None
    require(bool(receipt['parts']), 'No prediction parts')
    for part in receipt['parts']:
        path = (folder / part['path']).resolve()
        require(path.is_relative_to(folder.resolve()) and path not in seen,
                'Unsafe or duplicate prediction path')
        seen.add(path)
        require(sha(path) == part['sha256'], 'Prediction file SHA mismatch')
        require(0 < part['rows'] <= 65536, 'Prediction part exceeds bounded contract')
        frame = pl.read_parquet(path)
        require(len(frame) == part['rows'], 'Part count changed')
        require(frame['target_index'].to_list() == list(range(count, count + len(frame))),
                'Missing or duplicate target index')
        require(frame['split'].eq(receipt['split']).all(), 'Mixed target split')
        part_ids, part_truth = hashlib.sha256(), hashlib.sha256()
        for row in frame.iter_rows(named=True):
            key = (row['entity_id'], row['seq'])
            require(previous is None or key > previous, 'Duplicate or unordered target')
            previous = key
            expected_id = hashlib.sha256(compact([receipt['dataset'], receipt['data_file_sha256'],
                row['entity_id'], row['seq'], receipt['split'], row['recorded_date']])).hexdigest()
            require(row['target_id'] == expected_id, 'Target identity formula mismatch')
            encoded_id = (row['target_id'] + '\n').encode()
            encoded_truth = compact([row[k] for k in TRUTH_FIELDS]) + b'\n'
            identities.update(encoded_id); part_ids.update(encoded_id)
            truth.update(encoded_truth); part_truth.update(encoded_truth)
        require(part_ids.hexdigest() == part['target_identity_sha256'], 'Part target digest mismatch')
        require(part_truth.hexdigest() == part['truth_sha256'], 'Part truth digest mismatch')
        q = frame['raw_quantity'].to_numpy()
        pred = frame['predicted_raw_quantity'].to_numpy()
        nll = frame['time_nll'].to_numpy()
        require(np.isfinite(q).all() and np.isfinite(pred).all() and np.isfinite(nll).all(),
                'Nonfinite prediction or target')
        require((q >= 0).all(), 'Negative target quantity')
        for name, selected in [('overall', np.ones(len(frame), dtype=bool)), ('tail', q > threshold)]:
            error = pred[selected] - q[selected]
            totals = sums[name]
            totals[0] += int(selected.sum())
            totals[1] += float(np.abs(error).sum())
            totals[2] += float(np.square(error).sum())
            totals[3] += float(nll[selected].sum())
            totals[4] += float(error.sum())
        count += len(frame)
    require(count == receipt['prediction_rows'] == receipt['expected_target_count'], 'Incomplete population')
    require(identities.hexdigest() == receipt['target_identity_sha256'], 'Overall identity digest mismatch')
    require(truth.hexdigest() == receipt['truth_sha256'], 'Overall truth digest mismatch')
    return receipt, {name: finish(value) for name, value in sums.items()}


def compare_identity(candidate, baseline):
    for key in ('dataset', 'split', 'data_file_sha256', 'expected_target_count', 'prediction_rows',
                'target_identity_sha256', 'truth_sha256', 'loader', 'truth_digest_fields'):
        require(candidate[key] == baseline[key], 'Paired target/history identity differs: ' + key)
    return True


def validation_gate(metrics, reference, rtol, atol):
    require(metrics['count'] == reference['count'], 'Validation replay target count mismatch')
    checks = {key: math.isclose(metrics[key], reference[key], rel_tol=rtol, abs_tol=atol)
              for key in METRICS}
    require(all(checks.values()), 'Validation replay mismatch: ' + json.dumps(checks))
    return checks


def check_receipt(receipt, row, contract, root):
    for field in ('dataset', 'model', 'seed', 'selected_epoch', 'checkpoint_file_sha256',
                  'state_tensor_sha256', 'evaluator_source_bundle'):
        require(receipt[field] == row[field], 'Receipt registry mismatch: ' + field)
    require(receipt['registry_sha256'] == contract['registry_sha256'] and
            receipt['dataset_manifest_sha256'] == contract['dataset_manifest_sha256'] and
            receipt['contract_sha256'] == sha(HERE / 'execution_contract.json') and
            receipt['runner_sha256'] == contract['evaluator']['sha256'], 'Receipt provenance changed')
    registry = read(HERE / 'evaluation_registry.json')
    bundle = registry['bundles'][row['evaluator_source_bundle']]
    require(receipt['source_closure_sha256'] == bundle['source_closure_sha256'], 'Receipt source closure mismatch')
    require(sha(Path(root) / row['checkpoint_path']) == row['checkpoint_file_sha256'],
            'Selected checkpoint changed after evaluation')
    limits = contract['resources']
    require(receipt['runtime']['cpu_peak_rss_bytes'] <= limits['rss_limit_gib'] * 1024**3,
            'RSS resource cap exceeded')
    require(receipt['runtime']['cuda_peak_reserved_bytes'] <= limits['gpu_memory_limit_gib'] * 1024**3,
            'CUDA resource cap exceeded')


def checkpoint_worker(root, index):
    """Each source bundle imports in a fresh CPU process; no dataset is opened."""
    import os
    root = Path(root).resolve()
    registry = read(HERE / 'evaluation_registry.json')
    row = registry['rows'][index]
    bundle = registry['bundles'][row['evaluator_source_bundle']]
    source = (root / bundle['source_root']).resolve()
    for relative, digest in bundle['source_files'].items():
        require(sha(source / relative) == digest, 'Frozen source preflight mismatch')
    checkpoint = root / row['checkpoint_path']
    require(sha(checkpoint) == row['checkpoint_file_sha256'], 'Checkpoint file preflight mismatch')
    sys.path[:] = [str(source)] + [p for p in sys.path if p and not Path(p).resolve().is_relative_to(root)]
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    factory = importlib.import_module('models.TPPs.CountAwareFactory')
    runner = importlib.import_module('simple_lab_test.search.common.runner')
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(cp['backbone'] == row['model'] and cp['seed'] == row['seed'] and
            cp['best_epoch'] == row['selected_epoch'], 'Checkpoint identity mismatch')
    factory.validate_checkpoint_route(cp, row['model'])
    require(runner.canonical_state_dict_sha256(cp['model_state_dict']) == row['state_tensor_sha256'],
            'Checkpoint tensor SHA mismatch')
    data = next(d for d in bundle['datasets'] if d['dataset_id'] == row['dataset'])
    config = {k: v for k, v in data['model'].items()
              if k not in ('backbone', 'lambda_log_qty', 'lambda_tail', 'time_head_lr_multiplier')}
    model, _ = factory.build_count_aware_model(row['model'], **config,
        train_log_mean=data['statistics']['train_log_mean'], train_log_std=data['statistics']['train_log_std'],
        max_seq_len=data['loader']['max_seq_len'])
    model.load_state_dict(cp['model_state_dict'], strict=True)
    require(runner.canonical_state_dict_sha256(model.state_dict()) == row['state_tensor_sha256'],
            'Strict-loaded tensor SHA mismatch')
    for name, module in sys.modules.items():
        if name.split('.')[0] in ('models', 'paper', 'data_loader', 'simple_lab_test'):
            file = getattr(module, '__file__', None)
            if file:
                path = Path(file).resolve()
                require(path.is_relative_to(source) and str(path.relative_to(source)) in bundle['source_files'],
                        'Nonfrozen preflight import: ' + name)
    print(json.dumps({'status': 'passed', 'dataset': row['dataset'], 'model': row['model'], 'seed': row['seed'],
        'selected_epoch': row['selected_epoch'], 'state_tensor_sha256': row['state_tensor_sha256'],
        'checkpoint_file_sha256': row['checkpoint_file_sha256'], 'source_bundle': row['evaluator_source_bundle'],
        'strict_load': True, 'data_read': False, 'inference_calls': 0, 'device': 'cpu'}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--checkpoint-worker', type=int)
    parser.add_argument('--checkpoint-preflight', action='store_true')
    args = parser.parse_args()
    if args.checkpoint_worker is not None:
        checkpoint_worker(args.root, args.checkpoint_worker)
    elif args.checkpoint_preflight:
        rows = []
        for index in range(len(read(HERE / 'evaluation_registry.json')['rows'])):
            result = subprocess.run([sys.executable, str(HERE / 'verify.py'), '--root', str(args.root),
                '--checkpoint-worker', str(index)], capture_output=True, text=True, timeout=120, check=True)
            rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
        save(HERE / 'local_checkpoint_preflight.json', {'status': 'passed', 'checkpoint_count': len(rows),
            'rows': rows, 'dataset_read': False, 'inference_calls': 0,
            'selected_binary_cpu_identity_audit': 'passed', 'last_binary_cpu_audit': 'not in this scope'})
        print(json.dumps({'status': 'passed', 'checkpoint_count': len(rows)}))
    else:
        parser.error('Select --checkpoint-preflight or --checkpoint-worker')


if __name__ == '__main__':
    main()
