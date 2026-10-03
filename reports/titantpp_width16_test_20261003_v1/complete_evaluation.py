"""Finish two frozen validation gates, then two CPU-only existing-test evaluations."""
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
OLD = ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1'
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts')
MODEL = 'titantpp_history_mlp_width16'


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, value):
    with (B / name).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def load_predictions(folder):
    receipt = read(folder / 'receipt.json')
    assert receipt['status'] == 'complete' and receipt['full_population']
    assert receipt['retrained'] is False
    parts = []
    for part in receipt['parts']:
        path = folder / part['path']
        assert sha(path) == part['sha256']
        parts.append(pl.read_parquet(path))
    frame = pl.concat(parts)
    assert len(frame) == receipt['prediction_rows'] == receipt['expected_target_count']
    assert frame['target_index'].to_list() == list(range(len(frame)))
    # Recompute identity and truth digests, rather than trust receipt equality alone.
    ids, truth = hashlib.sha256(), hashlib.sha256()
    for row in frame.iter_rows(named=True):
        ids.update((row['target_id'] + '\n').encode())
        payload = [row[k] for k in receipt['truth_digest_fields']]
        truth.update((json.dumps(payload, separators=(',', ':'), ensure_ascii=False,
                                 allow_nan=False) + '\n').encode())
    assert ids.hexdigest() == receipt['target_identity_sha256']
    assert truth.hexdigest() == receipt['truth_sha256']
    return receipt, frame


def metrics(frame):
    error = (frame['predicted_raw_quantity'] - frame['raw_quantity']).to_numpy()
    time = frame['time_nll'].to_numpy()
    assert len(error) > 0 and np.isfinite(error).all() and np.isfinite(time).all()
    return {'count': len(frame), 'qty_mae': float(np.abs(error).mean()),
            'qty_rmse': float(np.sqrt(np.square(error).mean())),
            'qty_bias': float(error.mean()), 'time_nll': float(time.mean())}


def run(dataset, split):
    output = B / 'runs' / split / dataset
    assert not output.exists(), 'Never silently retry or reuse a failed inference'
    command = [sys.executable, str(B / 'evaluate.py'), '--root', str(ROOT),
               '--registry', str(B / 'evaluation_registry.json'),
               '--dataset-manifest', str(B / 'dataset_manifest.json'),
               '--contract', str(B / 'execution_contract.json'),
               '--dataset', dataset, '--model', MODEL, '--seed', '42',
               '--split', split, '--device', 'cpu', '--output', str(output),
               '--deadline-seconds', '1800']
    subprocess.run(command, check=True, timeout=1805)


def main():
    contract = read(B / 'execution_contract.json')
    assert sha(B / 'evaluate.py') == contract['evaluator']['sha256']
    assert sha(B / 'evaluation_registry.json') == contract['registry_sha256']
    registry = read(B / 'evaluation_registry.json')
    original_registry_path = ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json'
    original_registry = read(original_registry_path)
    binding = {}
    for dataset in DATASETS:
        folder = OLD / 'runs/test/attempt1' / f'{dataset}__titantpp_history_mlp__seed42'
        receipt, frame = load_predictions(folder)
        row = next(r for r in original_registry['rows'] if
                   (r['dataset'], r['model'], r['seed']) == (dataset, 'titantpp_history_mlp', 42))
        assert receipt['checkpoint_file_sha256'] == row['checkpoint_file_sha256']
        assert receipt['state_tensor_sha256'] == row['state_tensor_sha256']
        assert receipt['selected_epoch'] == row['selected_epoch']
        assert receipt['split'] == 'test' and receipt['heldout_read']
        binding[dataset] = {'folder': str(folder.relative_to(ROOT)),
                            'receipt_sha256': sha(folder / 'receipt.json'),
                            'checkpoint_file_sha256': receipt['checkpoint_file_sha256'],
                            'state_tensor_sha256': receipt['state_tensor_sha256'],
                            'selected_epoch': receipt['selected_epoch'],
                            'target_identity_sha256': receipt['target_identity_sha256'],
                            'truth_sha256': receipt['truth_sha256'],
                            'parts': receipt['parts']}
    write('comparison_binding.json', binding)
    # Taxi validation was invoked explicitly before this orchestrator was written.
    assert (B / 'runs/validation/yellow_trip_hourly/receipt.json').is_file()
    run('raf_spare_parts', 'validation')
    gates = {}
    for dataset in DATASETS:
        receipt, frame = load_predictions(B / 'runs/validation' / dataset)
        assert receipt['contract_sha256'] == sha(B / 'execution_contract.json')
        assert receipt['runner_sha256'] == contract['evaluator']['sha256']
        assert all(receipt['qualification'].values())
        result = metrics(frame)
        reference = contract['validation_gate']['references'][dataset]
        checks = {k: math.isclose(result[k], reference[k], rel_tol=1e-5, abs_tol=1e-5)
                  for k in ('qty_rmse', 'qty_mae', 'time_nll')}
        checks['count'] = result['count'] == reference['count']
        row = next(r for r in registry['rows'] if r['dataset'] == dataset)
        history = read((ROOT / row['checkpoint_path']).with_name('history.json'))['history']
        finite = [r for r in history if math.isfinite(r['val_qty_rmse'])]
        checks['first_validation_minimum_epoch'] = min(finite, key=lambda r: r['val_qty_rmse'])['epoch'] == row['selected_epoch']
        assert all(checks.values()), (dataset, checks, result, reference)
        gates[dataset] = {'passed': True, 'metrics': result, 'reference': reference,
                          'checks': checks, 'receipt_sha256': sha(B / 'runs/validation' / dataset / 'receipt.json')}
    write('validation_gate.json', {'passed': True, 'datasets': gates,
                                  'completed_before_test_utc': datetime.now(timezone.utc).isoformat()})
    for dataset in DATASETS:
        run(dataset, 'test')
    results = []
    for dataset in DATASETS:
        new_receipt, candidate = load_predictions(B / 'runs/test' / dataset)
        old_folder = ROOT / binding[dataset]['folder']
        assert sha(old_folder / 'receipt.json') == binding[dataset]['receipt_sha256']
        old_receipt, baseline = load_predictions(old_folder)
        for k in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256'):
            assert new_receipt[k] == old_receipt[k], (dataset, k)
        assert new_receipt['split'] == 'test' and new_receipt['heldout_read']
        assert new_receipt['contract_sha256'] == sha(B / 'execution_contract.json')
        assert new_receipt['runner_sha256'] == contract['evaluator']['sha256']
        assert all(new_receipt['qualification'].values())
        assert candidate.select(old_receipt['truth_digest_fields']).equals(baseline.select(old_receipt['truth_digest_fields']))
        threshold = contract['tail_thresholds'][dataset]
        row = {'dataset': dataset, 'seed': 42, 'split': 'test', 'paired_population_verified': True,
               'tail_threshold': threshold, 'baseline_selected_epoch': old_receipt['selected_epoch'],
               'candidate_selected_epoch': new_receipt['selected_epoch'], 'groups': {}}
        for group in ('overall', 'tail'):
            a = baseline if group == 'overall' else baseline.filter(pl.col('raw_quantity') > threshold)
            b = candidate if group == 'overall' else candidate.filter(pl.col('raw_quantity') > threshold)
            old_metrics, new_metrics = metrics(a), metrics(b)
            row['groups'][group] = {'width4': old_metrics, 'width16': new_metrics,
                'width16_minus_width4': {k: new_metrics[k] - old_metrics[k] for k in old_metrics if k != 'count'},
                'relative_change_percent': {k: 100 * (new_metrics[k] / old_metrics[k] - 1)
                   for k in ('qty_rmse', 'qty_mae', 'time_nll') if old_metrics[k] != 0}}
        results.append(row)
    write('results.json', {'scope': 'exploratory existing test split, seed42, not independent untouched evaluation',
                           'training': False, 'selection_changed': False, 'rows': results})
    write('completion_receipt.json', {'completed_utc': datetime.now(timezone.utc).isoformat(),
        'validation_gate_passed': True, 'test_conditions_complete': 2, 'all_target_truth_history_pairs_match': True,
        'results_sha256': sha(B / 'results.json'), 'contract_sha256': sha(B / 'execution_contract.json'),
        'baseline_predictions_reused': True, 'device': 'local_cpu', 'remote_training_untouched': True,
        'last_checkpoint_binary_audit': 'not in this evaluation scope'})
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
