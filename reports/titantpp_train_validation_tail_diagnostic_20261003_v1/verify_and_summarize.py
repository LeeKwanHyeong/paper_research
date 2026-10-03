"""Independently recompute the diagnostic with scalar math.fsum reductions."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
read = lambda p: json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


contract = read(OUT / 'contract.json')
receipts = [read(p) for p in sorted((OUT / 'runs').glob('*/receipt.json'))]
assert len(receipts) == 8
verified = 0
rows, populations, paired, calibration, receipts_by_key = [], [], [], [], {}
frames = {}
for receipt in receipts:
    assert receipt['status'] == 'complete' and receipt['parameters_unchanged']
    assert not receipt['new_training'] and not receipt['heldout_read']
    ds, arm = receipt['dataset'], receipt['model']
    receipts_by_key[ds, arm] = receipt
    bounds = receipt['quantity_bounds']
    for split, spec in receipt['splits'].items():
        path = OUT / spec['prediction_path']
        assert sha(path) == spec['prediction_sha256']
        f = pl.read_parquet(path).sort('target_index')
        frames[ds, arm, split] = f
        assert f['target_index'].n_unique() == f.height == spec['sample_n']
        assert set(f['split']) == {split}
        records = f.to_dicts()
        assert math.isclose(math.fsum(x['weight'] for x in records), spec['population']['target_count'])
        for meta in spec['strata']:
            populations.append(dict(dataset=ds, model=arm, split=split, **meta))
        for stored in spec['metrics']:
            g = stored['bin']
            group = records if g == -1 else [x for x in records if sum(x['raw_quantity'] > b for b in bounds) == g]
            assert len(group) == stored['n_evaluated']
            total_w = math.fsum(x['weight'] for x in group)
            avg = lambda fn: math.fsum(x['weight'] * fn(x) for x in group) / total_w
            calc = {
                'rmse': math.sqrt(avg(lambda x: (x['predicted_raw_quantity'] - x['raw_quantity']) ** 2)),
                'mae': avg(lambda x: abs(x['predicted_raw_quantity'] - x['raw_quantity'])),
                'bias': avg(lambda x: x['predicted_raw_quantity'] - x['raw_quantity']),
                'log_mse': avg(lambda x: (math.log1p(x['predicted_raw_quantity']) - math.log1p(x['raw_quantity'])) ** 2),
                'q_mean': avg(lambda x: x['raw_quantity']),
                'prediction_mean': avg(lambda x: x['predicted_raw_quantity']),
                'last_rmse': math.sqrt(avg(lambda x: (x['last_quantity'] - x['raw_quantity']) ** 2)),
                'mean_rmse': math.sqrt(avg(lambda x: (x['mean_quantity'] - x['raw_quantity']) ** 2)),
            }
            for name, value in calc.items():
                assert math.isclose(value, stored[name], rel_tol=1e-10, abs_tol=1e-10), (ds, split, g, name)
                verified += 1
            # Sampling-design SE for finite fixed targets only; excludes seed,
            # entity, model-fitting, or future-population uncertainty.
            numerator = 0.
            for meta in spec['strata']:
                if g != -1 and g != meta['bin']:
                    continue
                cell = [x for x in group if sum(x['raw_quantity'] > b for b in bounds) == meta['bin']]
                n, N = len(cell), meta['population_n']
                if n > 1 and n < N:
                    e2 = np.array([(x['predicted_raw_quantity'] - x['raw_quantity']) ** 2 for x in cell])
                    numerator += N ** 2 * (1 - n / N) * float(e2.var(ddof=1)) / n
            mse_se = math.sqrt(numerator) / total_w
            rows.append(dict(dataset=ds, model=arm, seed=42, epoch=receipt['selected_epoch'], split=split,
                             **stored, mse_sampling_se=mse_se,
                             rmse_sampling_se_delta=mse_se / (2 * stored['rmse']) if stored['rmse'] else 0.))
        # Prediction-defined buckets distinguish conditional calibration from
        # the mechanical regression-to-the-mean effect of selecting large y.
        for g in range(len(bounds) + 1):
            cell = [x for x in records if sum(x['predicted_raw_quantity'] > b for b in bounds) == g]
            if not cell:
                continue
            sw = math.fsum(x['weight'] for x in cell)
            calibration.append({'dataset': ds, 'model': arm, 'split': split, 'prediction_bin': g,
                'sample_n': len(cell), 'represented_n': sw,
                'prediction_mean': math.fsum(x['weight'] * x['predicted_raw_quantity'] for x in cell) / sw,
                'truth_mean': math.fsum(x['weight'] * x['raw_quantity'] for x in cell) / sw})

for ds in contract['datasets']:
    for split in ['train', 'validation']:
        a, b = [frames[ds, arm, split] for arm in contract['models']]
        for col in ['target_index', 'entity_id', 'seq', 'raw_quantity', 'history_length', 'weight']:
            assert a[col].equals(b[col]), (ds, split, col)
        # These columns are independently formed from the identical prefix.
        for col in ['last_quantity', 'mean_quantity']:
            assert np.allclose(a[col].to_numpy(), b[col].to_numpy(), atol=1e-5, rtol=1e-6)
        for g in [-1, 0, 1, 2, 3, 4]:
            rr = [next(x for x in rows if x['dataset'] == ds and x['model'] == arm and x['split'] == split and x['bin'] == g) for arm in contract['models']]
            paired.append({'dataset': ds, 'split': split, 'bin': g,
                           'mlp_rmse': rr[0]['rmse'], 'all_available_rmse': rr[1]['rmse'],
                           'rmse_change_pct': 100 * (rr[1]['rmse'] / rr[0]['rmse'] - 1)})

for name, data in [('metrics', rows), ('population', populations), ('paired', paired), ('prediction_calibration', calibration)]:
    columns = list(dict.fromkeys(k for row in data for k in row))
    with (OUT / f'{name}.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
protected = {p: sha(ROOT / p) == digest for p, digest in contract['protected_sha256_before'].items()}
verification = {'status': 'passed', 'independent_scalar_metric_comparisons': verified,
                'paired_target_checks': 8, 'protected_files_same_as_start': protected,
                'new_training': False, 'heldout_read': False, 'remote_calls': 0,
                'source_code_sha256': {p.name: sha(p) for p in OUT.glob('*.py')}}
(OUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
(OUT / 'analysis.json').write_text(json.dumps({'metrics': rows, 'paired': paired, 'prediction_calibration': calibration}, indent=2) + '\n')
print(json.dumps(verification, indent=2))
for ds in contract['datasets']:
    print(ds)
    for model in contract['models']:
        for split in ['train', 'validation']:
            rr = [x for x in rows if x['dataset'] == ds and x['model'] == model and x['split'] == split]
            total, tail = rr[0], rr[-1]
            print(model, split, {k: total[k] for k in ['rmse', 'log_mse', 'rmse_sampling_se_delta']},
                  'tail', {k: tail[k] for k in ['n_evaluated', 'rmse', 'bias', 'q_mean', 'log_mse', 'last_rmse', 'mean_rmse', 'rmse_sampling_se_delta']})
