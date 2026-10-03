"""Independent NumPy/PyArrow oracle for MLP/S2P2 and deterministic references.

Does not import either analysis implementation. Uses explicit frozen upper time
thresholds, boolean masks and direct sums rather than tick division/bincount.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

MODELS = ['titantpp_history_mlp', 's2p2_matched_head']
METRICS = ['qty_mae', 'qty_rmse', 'time_nll', 'bias']
SUMS = ['absolute_error_sum', 'squared_error_sum', 'time_nll_sum', 'signed_error_sum']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def independent_tick(date, kind):
    if kind == 'relative_day':
        return int(date)
    fmt = '%Y%m%d%H%M%S' if kind == 'calendar_hour' else '%Y%m%d'
    dt = datetime.strptime(date, fmt)
    if kind == 'calendar_hour':
        assert dt.minute == dt.second == 0
        return int((dt - datetime(1970, 1, 1)).total_seconds() // 3600)
    if kind == 'calendar_week':
        return (dt - datetime(1970, 1, 1)).days // 7
    if kind == 'calendar_month':
        assert dt.day == 1
        return 12 * dt.year + dt.month - 1
    raise ValueError(kind)


def groups_for(truth, spec):
    dates = truth['recorded_date']
    mapping = {d: independent_tick(d, spec['time_axis']['kind']) for d in set(dates)}
    ticks = np.array([mapping[d] for d in dates], dtype=np.int64)
    axis = spec['time_axis']
    assert ticks.min() == axis['min_tick'] and ticks.max() == axis['max_tick']
    c1, c2 = axis['upper_ticks_first_two_bins']
    groups = {'all': [np.ones(len(ticks), dtype=bool)],
              'time': [ticks <= c1, (ticks > c1) & (ticks <= c2), ticks > c2]}
    for view, field in [('quantity', 'raw_quantity'), ('history', 'history_length'),
                        ('history_detail', 'history_length')]:
        values = np.asarray(truth[field])
        bounds = spec[view + '_boundaries']
        groups[view] = [values <= bounds[0]]
        groups[view] += [(values > a) & (values <= b) for a, b in zip(bounds[:-1], bounds[1:])]
        groups[view] += [values > bounds[-1]]
    for masks in groups.values():
        assert np.all(np.sum(masks, axis=0) == 1)
    return groups


def read_vector(directory, truth=False):
    receipt = json.loads((directory / 'receipt.json').read_text())
    fields = ['target_id', 'raw_quantity', 'predicted_raw_quantity', 'time_nll']
    if truth:
        fields += ['recorded_date', 'history_length', 'last_quantity', 'mean_quantity', 'entity_id', 'site_id']
    lists = {f: [] for f in fields}
    for part in receipt['parts']:
        data = pq.read_table(directory / part['path'], columns=fields).to_pydict()
        for f in fields:
            lists[f].extend(data[f])
    return lists


def direct_metrics(truth, predicted, nll, masks):
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    result = {}
    for view, bins in masks.items():
        result[view] = {}
        for bi, mask in enumerate(bins):
            e = error[mask]
            count = int(mask.sum())
            sae, sse, signed = float(np.sum(np.abs(e))), float(np.dot(e, e)), float(np.sum(e))
            ns = None if nll is None else float(np.asarray(nll)[mask].sum())
            result[view][str(bi)] = dict(count=count, absolute_error_sum=sae,
                squared_error_sum=sse, signed_error_sum=signed, time_nll_sum=ns,
                qty_mae=sae/count if count else None,
                qty_rmse=float(np.sqrt(sse/count)) if count else None,
                bias=signed/count if count else None,
                time_nll=ns/count if count and ns is not None else None)
    return result


def verify(contract_path, result_path, output_path):
    contract = json.loads(contract_path.read_text())
    result = json.loads(result_path.read_text())
    assert result['status'] == 'complete' and result['split'] == 'test'
    assert result['analysis_contract_sha256'] == sha(contract_path)
    manifest_path = Path(contract['source_run_manifest'])
    manifest = json.loads(manifest_path.read_text())
    roots = {(c['dataset'], c['model'], c['seed']): manifest_path.parent / c['output_dir']
             for c in manifest['conditions']}
    comparisons = 0
    max_abs_metric_difference = 0.0
    datasets = {}

    def equal(a, b, key):
        nonlocal comparisons, max_abs_metric_difference
        comparisons += 1
        if a is None or b is None:
            assert a is b, (key, a, b)
        else:
            assert np.isclose(a, b, rtol=1e-10, atol=1e-10), (key, a, b)
            if key[-1] in METRICS:
                max_abs_metric_difference = max(max_abs_metric_difference, abs(a-b))

    for spec in contract['datasets']:
        dataset = spec['dataset']
        truth = read_vector(roots[dataset, MODELS[0], 42], truth=True)
        masks = groups_for(truth, spec)
        local = {}
        for model in MODELS + contract['simple_models']:
            local[model] = {}
            deterministic = model in contract['simple_models']
            seeds = ['deterministic'] if deterministic else contract['seeds']
            for seed in seeds:
                if deterministic:
                    field = 'last_quantity' if model == contract['simple_models'][0] else 'mean_quantity'
                    pred, nll = truth[field], None
                else:
                    vector = truth if model == MODELS[0] and seed == 42 else read_vector(roots[dataset, model, seed])
                    assert vector['target_id'] == truth['target_id']
                    assert vector['raw_quantity'] == truth['raw_quantity']
                    pred, nll = vector['predicted_raw_quantity'], vector['time_nll']
                metrics = direct_metrics(truth['raw_quantity'], pred, nll, masks)
                local[model][str(seed)] = metrics
                for view, bins in metrics.items():
                    for bi, values in bins.items():
                        saved = result['results'][dataset][view][bi]['seed_metrics'][model][str(seed)]
                        for key, value in values.items():
                            equal(value, saved[key], (dataset, view, bi, model, str(seed), key))
            for view, bins in masks.items():
                for bi in range(len(bins)):
                    saved = result['results'][dataset][view][str(bi)]
                    for metric in METRICS:
                        values = [local[model][str(s)][view][str(bi)][metric] for s in seeds]
                        mean = None if values[0] is None else float(np.mean(values))
                        sd = None if deterministic or mean is None else float(np.std(values, ddof=1))
                        equal(mean, saved['means'][model][metric], (dataset, view, str(bi), model, 'mean', metric))
                        equal(sd, saved['sample_sd'][model][metric], (dataset, view, str(bi), model, 'sd', metric))
        # Counts and cohort descriptions are independently reconstructed too.
        description = {}
        tick_map = {d: independent_tick(d, spec['time_axis']['kind']) for d in set(truth['recorded_date'])}
        for view, bins in masks.items():
            description[view] = {}
            for bi, mask in enumerate(bins):
                idx = np.flatnonzero(mask)
                entities = {truth['entity_id'][i] for i in idx}
                sites = {truth['site_id'][i] for i in idx if truth['site_id'][i] is not None}
                dates = [truth['recorded_date'][i] for i in idx]
                equal(len(idx), result['results'][dataset][view][str(bi)]['count'],
                      (dataset, view, str(bi), 'population', 'count'))
                description[view][str(bi)] = {'count': len(idx), 'entity_count': len(entities),
                    'site_count': len(sites), 'recorded_date_min': min(dates, key=tick_map.get) if dates else None,
                    'recorded_date_max': max(dates, key=tick_map.get) if dates else None}
        datasets[dataset] = {'learned_vectors': 6, 'deterministic_vectors': 2,
                            'target_count': len(truth['target_id']), 'cohort_descriptions': description}
        print('independent verified: ' + dataset, flush=True)
    evidence = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'independent recomputation of MLP and S2P2 all seeds plus both deterministic references; not all108 learned vectors',
        'learned_vectors': 24, 'deterministic_vectors': 8, 'comparisons': comparisons,
        'max_absolute_metric_difference': max_abs_metric_difference,
        'rtol': 1e-10, 'atol': 1e-10,
        'algorithm': 'PyArrow column reads; explicit upper time thresholds; boolean stratum masks; direct NumPy sum/dot; no analyzer imports',
        'analysis_contract_sha256': sha(contract_path), 'analysis_sha256': sha(result_path),
        'oracle_script_sha256': sha(__file__), 'datasets': datasets}
    with output_path.open('x') as f:
        json.dump(evidence, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    verify(a.contract, a.analysis, a.output)
