"""Local stored-statistics audit. No model imports, inference, training or network."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import tarfile
from datetime import datetime
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
B = ROOT / 'search_artifacts/titantpp_history_taxi6_test_20261015_v1'
C = ROOT / 'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1'
FILES = {}
CHECKS = 0
ARMS = {'CNN': 'titantpp_cnn_gru54_cnn_only',
        'Adaptive': 'titantpp_cnn_history_adaptive_v1',
        'Capacity': 'titantpp_cnn_history_capacity_v1'}
LABELS = ['q≤7', '7<q≤686', '686<q≤1562', '1562<q≤3449', 'q>3449']
FIELDS = ['qty_sse', 'qty_absolute_error_sum', 'qty_signed_error_sum', 'time_nll_sum']

def pin(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert h == expected, (str(path), h, expected)
    FILES[str(path.relative_to(ROOT))] = h
    return path

def load(path, expected=None):
    return json.loads(pin(path, expected).read_text())

def close(a, b):
    global CHECKS
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8), (a, b)
    CHECKS += 1

def write(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def derived(cell):
    n = cell['count']
    a, s = cell['qty_absolute_error_sum'], cell['qty_signed_error_sum']
    assert n > 0 and a >= abs(s) - 1e-8 and cell['qty_sse'] >= 0
    close(math.sqrt(cell['qty_sse'] / n), cell['qty_rmse'])
    close(a / n, cell['qty_mae'])
    close(s / n, cell['qty_bias'])
    close(cell['time_nll_sum'] / n, cell['time_nll'])
    return {k: cell[k] for k in ['count', *FIELDS, 'qty_rmse', 'qty_mae', 'qty_bias', 'time_nll']} | {
        'overprediction_ae_mass': (a + s) / 2,
        'underprediction_ae_mass': (a - s) / 2,
    }

def delta(a, b):
    assert a['count'] == b['count']
    keys = [*FIELDS, 'qty_rmse', 'qty_mae', 'qty_bias', 'time_nll',
            'overprediction_ae_mass', 'underprediction_ae_mass']
    return {k: a[k] - b[k] for k in keys} | {
        'relative_percent': {k: 100 * (a[k] / b[k] - 1) if b[k] else None
                             for k in ['qty_rmse', 'qty_mae']}}

assert not (OUT / 'RESULT.json').exists(), 'Do not overwrite an analysis result'
pin(__file__)
pin(OUT / 'PLAN.md')
receipt = load(B / 'final_receipt.json', '4f1aae8e3de997e5dd50e1bae839225e4e32abc060d8237313d7002464e3bc95')
assert receipt['status'] == 'complete' and receipt['candidate_test_count'] == 6
for p, h in receipt['files'].items():
    pin(p, h)
contract = load(B / 'contract.json')
training_receipt = load(B / contract['training_receipt_path'], contract['training_receipt_sha256'])
assert training_receipt['status'] == 'complete' and training_receipt['new_fit_count'] == 6
for p, h in training_receipt['files'].items():
    pin(p, h)
train_contract = load(B / contract['training_contract_path'], contract['training_contract_sha256'])
inventory = load(B / 'preparation_inventory.json')
manifest_path = C / 'evaluation_results_manifest.json'
cnn_manifest = load(manifest_path, inventory['external_dependencies'][str(manifest_path)])
prior_dir = ROOT / 'reports/titantpp_taxi_raf_integrated_review_20261016_v1'
prior_receipt = load(prior_dir / 'RECEIPT.json')
prior = load(prior_dir / 'RESULT.json', prior_receipt['result_sha256'])
raw, rows, history, contrasts = {}, [], [], []
availability = []

for split in ['validation', 'test']:
    for arm, full_arm in ARMS.items():
        for seed in [42, 52, 62]:
            jid = f'yellow_trip_hourly__{seed}__{full_arm}'
            filename = 'validation_replay.json' if split == 'validation' else 'test_metrics.json'
            if arm == 'CNN' and split == 'validation':
                rel = f'results/{jid}/{filename}'
                doc = load(C / rel, cnn_manifest['files'][rel])
            else:
                p = B / ('references' if arm == 'CNN' else 'results') / jid / filename
                doc = load(p, receipt['files'][str(p)])
            assert doc['status'] == 'complete' and doc['full_population'] is True
            assert doc['dataset'] == 'yellow_trip_hourly' and doc['seed'] == seed
            assert doc['split'] == split and doc['arm'] == full_arm
            assert doc['raw_predictions_written'] is False
            m = doc['metrics']
            assert m['quantity_boundaries'] == [7, 686, 1562, 3449]
            assert doc['prediction_rows'] == doc['expected_target_count'] == m['count']
            values = [derived(m), *[derived(x) for x in m['quantity_cells']]]
            assert sum(x['count'] for x in m['quantity_cells']) == m['count']
            for k in FIELDS:
                close(math.fsum(x[k] for x in m['quantity_cells']), m[k])
                close(m['tail'][k], m['quantity_cells'][-1][k])
            raw[split, arm, seed] = doc
            old = next(x for x in prior['seed_rows'] if (x['dataset'], x['split'], x['arm'], x['seed']) == ('Taxi', split, arm, seed))
            for key, new in [('RMSE', m['qty_rmse']), ('MAE', m['qty_mae']), ('TimeNLL', m['time_nll'])]:
                close(old['metrics'][key], new)
            assert old['selected_epoch'] == doc['selected_epoch']
            for label, value in zip(['overall', *LABELS], values):
                rows.append({'split': split, 'arm': arm, 'seed': seed,
                             'selected_epoch': doc['selected_epoch'], 'stratum': label, **value})
            availability.append({'split': split, 'arm': arm, 'seed': seed,
                'source': str(p.relative_to(ROOT)) if not (arm == 'CNN' and split == 'validation') else str((C / rel).relative_to(ROOT)),
                'raw_predictions_written': False, 'available': 'overall and quantity-bin sufficient statistics'})

for split in ['validation', 'test']:
    base = raw[split, 'CNN', 42]
    for arm in ARMS:
        for seed in [42, 52, 62]:
            x = raw[split, arm, seed]
            for k in ['target_identity_sha256', 'truth_sha256', 'time_sha256', 'data_sha256', 'truth_digest_fields', 'loader']:
                assert x[k] == base[k], (split, arm, seed, k)
            for k in ['checkpoint_sha256', 'state_sha256', 'selected_epoch', 'source_closure']:
                assert x[k] == raw['validation', arm, seed][k]
    for a, b in [('Capacity', 'CNN'), ('Adaptive', 'CNN'), ('Capacity', 'Adaptive')]:
        for seed in [42, 52, 62]:
            dd = []
            for stratum in ['overall', *LABELS]:
                aa = next(x for x in rows if (x['split'], x['arm'], x['seed'], x['stratum']) == (split, a, seed, stratum))
                bb = next(x for x in rows if (x['split'], x['arm'], x['seed'], x['stratum']) == (split, b, seed, stratum))
                dd.append({'stratum': stratum, 'count': aa['count'], **delta(aa, bb)})
            for k in [*FIELDS, 'overprediction_ae_mass', 'underprediction_ae_mass']:
                close(math.fsum(x[k] for x in dd[1:]), dd[0][k])
            contrasts.append({'split': split, 'candidate': a, 'reference': b, 'seed': seed, 'strata': dd})

# Stored selected-checkpoint Validation history strata; no new forwards.
for arm, full_arm in ARMS.items():
    for seed in [42, 52, 62]:
        if arm == 'CNN':
            j = next(j for j in train_contract['baseline_jobs'] if j['seed'] == seed)
            v = j['original']['validation_reference']
        else:
            j = next(j for j in contract['jobs'] if j['seed'] == seed and j['arm'] == full_arm)
            v = j['validation_reference']
        m = raw['validation', arm, seed]['metrics']
        assert v['state_sha256'] == raw['validation', arm, seed]['state_sha256']
        assert v['history_boundaries'] == [64, 128]
        assert sum(c['count'] for c in v['history_cells']) == m['count']
        close(v['qty_rmse'], m['qty_rmse'])
        close(math.fsum(c['qty_sse'] for c in v['history_cells']), m['qty_sse'])
        close(math.fsum(c['qty_mae'] * c['count'] for c in v['history_cells']), m['qty_absolute_error_sum'])
        for label, c in zip(['history≤64', '64<history≤128', 'history>128'], v['history_cells']):
            history.append({'arm': arm, 'seed': seed, 'stratum': label, **c})

# Audit only the local exported archive inventories; never contact a server.
archives = []
for name in ['5080_test_results.tar.gz', '5080_validation_gate.tar.gz']:
    path = pin(B / name, receipt['files'][str(B / name)])
    with tarfile.open(path, 'r:gz') as tf:
        names = [m.name for m in tf.getmembers() if m.isfile()]
    suspects = [n for n in names if n.endswith(('.parquet', '.csv', '.jsonl', '.npz', '.npy'))]
    archives.append({'path': str(path.relative_to(ROOT)), 'file_count': len(names), 'possible_row_payloads': suspects, 'members': names})
    assert not suspects

aggregates = []
for split in ['validation', 'test']:
    for arm in ARMS:
        for stratum in ['overall', *LABELS]:
            rr = [x for x in rows if (x['split'], x['arm'], x['stratum']) == (split, arm, stratum)]
            aggregates.append({'split': split, 'arm': arm, 'stratum': stratum,
                'metrics': {k: {'mean': statistics.mean(x[k] for x in rr), 'sample_sd': statistics.stdev(x[k] for x in rr)}
                            for k in ['qty_rmse', 'qty_mae', 'qty_bias', 'time_nll']}})
for p, h in FILES.items():
    assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
result = {'status': 'aggregate_decomposition_complete_row_analysis_unavailable',
    'created_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    'verification_status': 'ANALYZED', 'dataset': 'yellow_trip_hourly',
    'rows': rows, 'contrasts': contrasts, 'aggregates': aggregates,
    'validation_history_strata': history, 'prediction_availability': availability,
    'archive_inventory': archives, 'source_files': FILES, 'scalar_checks': CHECKS,
    'all_targets_truth_inputs_matched_within_split': True,
    'row_level_paired_analysis_performed': False,
    'new_training': 0, 'new_inference': 0, 'remote_execution': 0, 'baseline_modified': False,
    'limitations': ['No individual predictions in these completed evaluation exports',
       'No candidate selected-checkpoint Train predictions: training losses are not substitutes',
       'Target quantity strata are retrospective and cannot be deployed as input rules',
       'AE mass by sign is not case count; sign transitions and paired residual covariance unavailable',
       'Marginal history bins do not control quantity or entity composition',
       'Cannot test top-case concentration, per-entity weighting, or observable history condition consistency',
       'Three seeds, descriptive exploratory reuse of Test; no causal or significance claim']}
write('RESULT.json', result)
for name, data in [('metrics_by_stratum.csv', rows), ('validation_history_strata.csv', history)]:
    with (OUT / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader(); writer.writerows(data)
write('RECEIPT.json', {'status': result['status'], 'source_file_count': len(FILES),
    'scalar_checks': CHECKS, 'outputs': {n: hashlib.sha256((OUT/n).read_bytes()).hexdigest()
        for n in ['RESULT.json', 'metrics_by_stratum.csv', 'validation_history_strata.csv']},
    'new_training': 0, 'new_inference': 0, 'baseline_modified': False})
print(json.dumps({'status': result['status'], 'source_file_count': len(FILES), 'scalar_checks': CHECKS}, ensure_ascii=False))
