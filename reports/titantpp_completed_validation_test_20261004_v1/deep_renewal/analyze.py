"""Verified event-weighted metrics for all 24 native Deep Renewal dataset/seed/split rows."""
import csv
import json
from pathlib import Path

import numpy as np
import polars as pl

from pipeline import HERE, ROOT, identity, load_predictions, metrics, read, sha, write


def main():
    registry = read(HERE / 'evaluation_registry.json')
    contract = read(HERE / 'execution_contract.json')
    assert read(HERE / 'inference_completion.json')['status'] == 'complete'
    assert read(HERE / 'validation_gate.json')['passed'] is True
    reuse = {(r['dataset'], r['seed'], r['split']): r for r in read(HERE / 'reused_predictions.json')['rows']}
    result_rows, evidence = [], []
    for row in registry['rows']:
        spec = next(d for d in registry['bundles'][row['evaluator_source_bundle']]['datasets'] if d['dataset_id'] == row['dataset'])
        quantity_boundaries = spec['quantity_boundaries_all_train_rows']
        history_boundaries = spec['history_boundaries']
        for split in ('validation', 'test'):
            key = (row['dataset'], row['seed'], split)
            folder = ROOT / reuse[key]['folder'] if key in reuse else HERE / 'runs' / split / identity(row)
            if key in reuse:
                assert sha(folder / 'receipt.json') == reuse[key]['receipt_sha256']
            receipt, frame = load_predictions(folder)
            for field in ('dataset', 'model', 'seed', 'selected_epoch', 'checkpoint_file_sha256', 'state_tensor_sha256'):
                assert receipt[field] == row[field], (key, field)
            assert receipt['split'] == split
            expected = contract['population_references'][row['dataset']][split]
            for field in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'expected_target_count'):
                assert receipt[field] == expected[field], (key, field)
            if key not in reuse:
                assert receipt['contract_sha256'] == sha(HERE / 'execution_contract.json')
                assert receipt['runner_sha256'] == contract['evaluator_sha256']
            threshold = quantity_boundaries[-1]
            groups = [('overall', frame, None), ('tail', frame.filter(pl.col('raw_quantity') > threshold), threshold),
                      ('body', frame.filter(pl.col('raw_quantity') <= threshold), threshold)]
            for name, column, boundaries in [('quantity', 'raw_quantity', quantity_boundaries),
                                              ('history', 'history_length', history_boundaries)]:
                bins = np.searchsorted(boundaries, frame[column].to_numpy(), side='left')
                for index in range(len(boundaries) + 1):
                    subset = frame.filter(pl.Series(bins == index))
                    groups.append((f'{name}_bin_{index}', subset, json.dumps(boundaries)))
            common = {'dataset': row['dataset'], 'model': row['model'], 'seed': row['seed'], 'split': split,
                'selected_epoch': row['selected_epoch'], 'receipt_path': str((folder / 'receipt.json').relative_to(ROOT)),
                'receipt_sha256': sha(folder / 'receipt.json'), 'reused_predictions': key in reuse,
                'comparison_family': 'native_deep_renewal_research_only',
                'time_nll_definition': 'recorded_positive_integer_shifted_negative_binomial_mass_nll; top-coded survival where configured'}
            for view, subset, threshold in groups:
                values = metrics(subset) if len(subset) else {k: 0 if k == 'count' else None for k in
                    ('count', 'qty_mae', 'qty_rmse', 'qty_bias', 'time_nll', 'qty_absolute_error_sum',
                     'qty_signed_error_sum', 'qty_sse', 'time_nll_sum')}
                result_rows.append({**common, 'view': view, 'threshold': threshold, **values})
            evidence.append({**common, 'target_identity_sha256': receipt['target_identity_sha256'],
                'truth_sha256': receipt['truth_sha256'], 'parts': receipt['parts']})
    assert len(evidence) == 24
    for filename, rows in [('metrics.csv', result_rows), ('overall.csv', [r for r in result_rows if r['view'] == 'overall'])]:
        with (HERE / filename).open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write('analysis_receipt.json', {'status': 'complete', 'split_count': 24, 'scientific_conditions': 12,
        'new_split_count': 24, 'reused_split_count': 0, 'selection_changed': False, 'training': False,
        'contract_sha256': sha(HERE / 'execution_contract.json'), 'metrics_sha256': sha(HERE / 'metrics.csv'),
        'overall_sha256': sha(HERE / 'overall.csv'), 'metric_rows': len(result_rows), 'evidence': evidence,
        'quantity_bins': 'searchsorted(side=left); equality belongs to lower bin',
        'tail': 'raw_quantity > final fixed TRAIN quantity boundary',
        'scope': 'Retrospective existing Validation/Test populations; not untouched independent evaluation'})
    print(json.dumps({'status': 'complete', 'split_count': 24, 'metric_rows': len(result_rows)}))


if __name__ == '__main__':
    main()
