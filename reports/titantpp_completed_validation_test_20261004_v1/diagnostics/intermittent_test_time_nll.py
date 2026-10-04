"""Predefined descriptive decomposition of immutable paired Test predictions."""
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
WIDTH = HERE.parent / 'width16'
LEGACY = ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1'
DATASET = 'intermittent_frozen_5000'
GAP_LABELS = ('1', '2–4', '5–13', '14–52', '>52')
QUANTITY_THRESHOLD = 187


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_write(name, rows):
    with (HERE / name).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    specification = importlib.util.spec_from_file_location('width16_verified_reader', WIDTH / 'pipeline.py')
    reader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reader)
    registry_paths = {4: ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json',
                      16: WIDTH / 'evaluation_registry.json'}
    registries = {width: read(path) for width, path in registry_paths.items()}
    evidence = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                str((WIDTH / 'pipeline.py').relative_to(ROOT)): sha(WIDTH / 'pipeline.py')}
    evidence.update({str(path.relative_to(ROOT)): sha(path) for path in registry_paths.values()})
    rows, extremes, paired, arrays = [], [], [], {}
    for seed in (42, 52, 62):
        frames = {}
        for width in (4, 16):
            model = 'titantpp_history_mlp' if width == 4 else 'titantpp_history_mlp_width16'
            row = next(r for r in registries[width]['rows'] if (r['dataset'], r['model'], r['seed']) == (DATASET, model, seed))
            bundle = registries[width]['bundles'][row['evaluator_source_bundle']]
            for relative, expected in bundle['source_files'].items():
                path = ROOT / bundle['source_root'] / relative
                assert sha(path) == expected
                evidence[str(path.relative_to(ROOT))] = expected
            folder = (LEGACY / 'runs/test/attempt1' if width == 4 else WIDTH / 'runs/test') / f'{DATASET}__{model}__seed{seed}'
            receipt, frame = reader.load_predictions(folder)
            for key in ('checkpoint_file_sha256', 'state_tensor_sha256', 'selected_epoch', 'seed', 'model', 'dataset'):
                assert receipt[key] == row[key]
            assert receipt['split'] == 'test'
            evidence[str((folder / 'receipt.json').relative_to(ROOT))] = sha(folder / 'receipt.json')
            for part in receipt['parts']:
                evidence[str((folder / part['path']).relative_to(ROOT))] = part['sha256']
            frames[width] = (receipt, frame)
            gap, quantity, values = (frame[name].to_numpy() for name in ('recorded_gap', 'raw_quantity', 'time_nll'))
            assert np.isfinite(values).all() and np.isfinite(gap).all()
            assert (gap >= 1).all() and np.equal(gap, np.rint(gap)).all()
            gap_bin = np.searchsorted([1, 4, 13, 52], gap, side='left')
            tail = quantity > QUANTITY_THRESHOLD
            total = float(values.sum())
            arrays[(width, seed)] = (values, gap_bin, tail)
            groups = [('overall', 'all', 'all', np.ones(len(values), dtype=bool))]
            groups += [('quantity', 'all', q, tail if q == 'tail' else ~tail) for q in ('body', 'tail')]
            for index, label in enumerate(GAP_LABELS):
                groups.append(('gap', label, 'all', gap_bin == index))
                for q in ('body', 'tail'):
                    groups.append(('gap_quantity', label, q, (gap_bin == index) & (tail if q == 'tail' else ~tail)))
            for view, gap_label, q, mask in groups:
                n, subtotal = int(mask.sum()), float(values[mask].sum())
                rows.append({'dataset': DATASET, 'split': 'test', 'width': width, 'seed': seed, 'view': view,
                    'gap_weeks': gap_label, 'quantity_group': q, 'quantity_tail_threshold': QUANTITY_THRESHOLD,
                    'count': n, 'mean_nll': subtotal / n if n else None, 'sum_nll': subtotal,
                    'total_nll_share': subtotal / total, 'population_share': n / len(values),
                    'contribution_to_overall_mean_nll': subtotal / len(values)})
            top_n = math.ceil(len(values) * 0.01)
            order = np.argsort(-values, kind='stable')[:top_n]
            extremes.append({'dataset': DATASET, 'split': 'test', 'width': width, 'seed': seed,
                'count': len(values), 'mean_nll': total / len(values), 'max_nll': float(values.max()),
                'p99_nll': float(np.quantile(values, .99, method='linear')), 'top_one_percent_count': top_n,
                'top_one_percent_nll_share': float(values[order].sum()) / total,
                'top_one_percent_gap_over52_fraction': float((gap[order] > 52).mean()),
                'top_one_percent_quantity_tail_fraction': float(tail[order].mean()),
                'top_one_percent_min_gap_weeks': float(gap[order].min()),
                'top_one_percent_max_gap_weeks': float(gap[order].max())})
        a, b = frames[4], frames[16]
        for key in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'truth_digest_fields'):
            assert a[0][key] == b[0][key], (seed, key)
        assert a[1].select(a[0]['truth_digest_fields']).equals(b[1].select(b[0]['truth_digest_fields']))
        paired.append({'seed': seed, 'count': len(a[1]), 'target_identity_sha256': a[0]['target_identity_sha256'],
                       'truth_sha256': a[0]['truth_sha256'], 'exact_truth_fields_equal': True})
    delta_rows = []
    for seed in (42, 52, 62):
        base, bins, tail = arrays[(4, seed)]
        candidate = arrays[(16, seed)][0]
        delta = candidate - base
        groups = [('overall', 'all', 'all', np.ones(len(delta), dtype=bool))]
        groups += [('quantity', 'all', q, tail if q == 'tail' else ~tail) for q in ('body', 'tail')]
        groups += [('gap', label, 'all', bins == i) for i, label in enumerate(GAP_LABELS)]
        for view, label, q, mask in groups:
            subtotal = float(delta[mask].sum())
            delta_rows.append({'seed': seed, 'view': view, 'gap_weeks': label, 'quantity_group': q,
                'count': int(mask.sum()), 'mean_nll_delta_within_group': float(delta[mask].mean()) if mask.any() else None,
                'sum_nll_delta': subtotal, 'contribution_to_overall_mean_nll_increase': subtotal / len(delta),
                'share_of_total_nll_increase': subtotal / float(delta.sum())})
    csv_write('intermittent_time_nll_groups.csv', rows)
    csv_write('intermittent_time_nll_extremes.csv', extremes)
    csv_write('intermittent_time_nll_paired_delta.csv', delta_rows)
    result = {'status': 'verified', 'dataset': DATASET, 'split': 'test', 'paired': paired,
        'metric': 'recorded positive-integer round/clamp lognormal PMF NLL; unit week; no top code',
        'fixed_gap_boundaries': [1, 4, 13, 52], 'quantity_tail': 'raw_quantity > 187; frozen TRAIN boundary',
        'top_one_percent': 'ceil(0.01*N) largest NLL observations separately per fixed model/seed; ties use target order',
        'p99': 'numpy linear quantile', 'reselection': False, 'training': False, 'gpu_execution': False,
        'source_and_inputs_sha256': evidence,
        'outputs_sha256': {name: sha(HERE / name) for name in ('intermittent_time_nll_groups.csv',
            'intermittent_time_nll_extremes.csv', 'intermittent_time_nll_paired_delta.csv')}}
    with (HERE / 'intermittent_time_nll_verification.json').open('x') as stream:
        json.dump(result, stream, indent=2);stream.write('\n')
    print(json.dumps({'extremes': extremes, 'delta': [r for r in delta_rows if r['view'] in ('gap', 'quantity')]}))


if __name__ == '__main__':
    main()
