"""Read stored Train/Validation rows only; no model imports or inference."""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import defaultdict

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
SAMPLE = ROOT / 'reports/titantpp_train_validation_tail_diagnostic_20261003_v1'
W4 = 'titantpp_history_mlp'
W16 = 'titantpp_history_mlp_width16'
ALL = 'titantpp_all_available_history_mlp'
INPUTS = {}
CHECKS = []
ROWS = []
PAIRS = []
EXAMPLES = []
COVERAGE = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, expected=None):
    path = Path(path)
    assert '/runs/test/' not in str(path) and path.name not in ('metrics_per_seed.csv', 'REPORT.md')
    actual = sha(path)
    if expected is not None:
        assert actual == expected, (str(path), actual, expected)
    INPUTS[str(path.relative_to(ROOT))] = actual
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(name, values):
    fields = list(dict.fromkeys(k for row in values for k in row))
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(values)


def arrays(df):
    assert set(df['split'].unique().to_list()) <= {'train', 'validation'}
    assert df['target_index'].n_unique() == len(df)
    cols = ('raw_quantity', 'predicted_raw_quantity', 'last_quantity', 'mean_quantity', 'time_nll', 'history_length')
    a = {k: df[k].to_numpy().astype(float) for k in cols}
    assert all(np.isfinite(x).all() for x in a.values())
    a['weight'] = df['weight'].to_numpy() if 'weight' in df.columns else np.ones(len(df))
    assert (a['weight'] > 0).all()
    a['entity_id'] = np.array(df['entity_id'].to_list())
    a['error'] = a['predicted_raw_quantity'] - a['raw_quantity']
    return a


def metrics(a, idx, weighting):
    w = a['weight'][idx].copy()
    entity = a['entity_id'][idx]
    if weighting == 'entity_equal':
        totals = defaultdict(float)
        for e, x in zip(entity, w):
            totals[e] += x
        w /= np.array([totals[e] for e in entity])
    error = a['error'][idx]
    return dict(n=int(idx.sum()), entities=len(set(entity)), weight_sum=float(w.sum()),
                rmse=float(np.sqrt(np.average(error**2, weights=w))),
                mae=float(np.average(np.abs(error), weights=w)),
                bias=float(np.average(error, weights=w)),
                time_nll=float(np.average(a['time_nll'][idx], weights=w)),
                sse=float(np.dot(w, error**2)), sae=float(np.dot(w, np.abs(error))),
                underprediction_share=float(np.average(error < 0, weights=w)))


def groups(a, bounds):
    out = [('overall', 'all', np.ones(len(a['error']), dtype=bool))]
    qb = np.searchsorted(bounds, a['raw_quantity'], side='left')
    # Explicit half-open bins; labels reflect exact inequalities in the output.
    hb = np.digitize(a['history_length'], [2,4,8,16,32,64,128], right=False)
    for axis, values in [('quantity_bin', qb), ('history_bin', hb)]:
        for k in sorted(set(values)):
            out.append((axis, str(int(k)), values == k))
    deviation = np.log1p(a['last_quantity']) - np.log1p(a['mean_quantity'])
    recent = np.where(deviation < -math.log(2), 'below_half', np.where(deviation > math.log(2), 'above_double', 'middle'))
    for label in ('below_half', 'middle', 'above_double'):
        out.append(('observed_last_vs_mean', label, recent == label))
    for label, mask in [('tail', a['raw_quantity'] > bounds[-1]), ('non_tail', a['raw_quantity'] <= bounds[-1])]:
        out.append(('tail', label, mask))
    for k in sorted(set(qb)):
        for label in ('below_half', 'middle', 'above_double'):
            out.append(('quantity_by_observed_deviation', f'{k}:{label}', (qb == k) & (recent == label)))
    return [(axis, label, idx) for axis, label, idx in out if idx.any()]


def describe(df, dataset, model, seed, split, scope, bounds):
    a = arrays(df)
    for axis, label, idx in groups(a, bounds):
        for weighting in ('row', 'entity_equal'):
            ROWS.append(dict(dataset=dataset, model=model, seed=seed, split=split, scope=scope,
                             axis=axis, group=label, weighting=weighting, **metrics(a, idx, weighting)))


def align(a, b):
    keys = ['target_index', 'entity_id', 'seq', 'split']
    assert a.select(keys).n_unique() == len(a) and b.select(keys).n_unique() == len(b)
    common = a.select(keys).join(b.select(keys), on=keys, how='inner').sort('target_index')
    aa = common.join(a, on=keys, how='left').sort('target_index')
    bb = common.join(b, on=keys, how='left').sort('target_index')
    assert len(aa) == min(len(a), len(b)), 'Unexpected partial overlap'
    for c in ['raw_quantity', 'last_quantity', 'mean_quantity', 'history_length']:
        # Legacy Train means were saved float32; compared rows use the same scheme.
        assert np.allclose(aa[c].to_numpy(), bb[c].to_numpy(), rtol=1e-7, atol=1e-6), c
    if 'weight' in aa.columns and 'weight' not in bb.columns:
        bb = bb.with_columns(aa['weight'])
    if 'weight' in bb.columns and 'weight' not in aa.columns:
        aa = aa.with_columns(bb['weight'])
    assert np.array_equal(arrays(aa)['weight'], arrays(bb)['weight'])
    return aa, bb


def compare(candidate, reference, dataset, pair, split, scope, bounds):
    ca, re = align(candidate, reference)
    a, b = arrays(ca), arrays(re)
    count = len(ca)
    CHECKS.append(dict(dataset=dataset, pair=pair, split=split, scope=scope, exact_paired_rows=count))
    full_delta_sse = float(np.dot(a['weight'], a['error']**2-b['error']**2))
    full_delta_sae = float(np.dot(a['weight'], np.abs(a['error'])-np.abs(b['error'])))
    for axis, label, idx in groups(a, bounds):
        for weighting in ('row', 'entity_equal'):
            am, bm = metrics(a, idx, weighting), metrics(b, idx, weighting)
            item = dict(dataset=dataset, pair=pair, seed=42, split=split, scope=scope, axis=axis,
                        group=label, weighting=weighting, n=am['n'], entities=am['entities'],
                        weight_sum=am['weight_sum'])
            for metric in ('rmse','mae','bias','time_nll','sse','sae'):
                item['candidate_'+metric], item['reference_'+metric] = am[metric], bm[metric]
                item['delta_'+metric] = am[metric]-bm[metric]
            fraction_weights = a['weight'][idx].copy()
            if weighting == 'entity_equal':
                entity_totals = defaultdict(float)
                for entity, weight in zip(a['entity_id'][idx], fraction_weights):
                    entity_totals[entity] += weight
                fraction_weights /= np.array([entity_totals[e] for e in a['entity_id'][idx]])
            item['improved_row_fraction'] = float(np.average(np.abs(a['error'][idx]) < np.abs(b['error'][idx]), weights=fraction_weights))
            PAIRS.append(item)
    for axis in ('quantity_bin', 'history_bin', 'observed_last_vs_mean', 'tail'):
        sub = [x for x in PAIRS if x['dataset']==dataset and x['pair']==pair and x['split']==split and x['scope']==scope and x['axis']==axis and x['weighting']=='row']
        assert math.isclose(sum(x['delta_sse'] for x in sub),full_delta_sse,rel_tol=1e-8,abs_tol=1e-6)
        assert math.isclose(sum(x['delta_sae'] for x in sub),full_delta_sae,rel_tol=1e-8,abs_tol=1e-6)
    # Largest changes are explanatory examples, not a fitted routing policy.
    ds = a['error']**2-b['error']**2
    for direction, ids in [('improvement', np.argsort(ds)[:5]), ('worsening', np.argsort(ds)[-5:][::-1])]:
        for i in ids:
            row = ca.row(int(i), named=True)
            EXAMPLES.append(dict(dataset=dataset,pair=pair,split=split,scope=scope,direction=direction,
                                 target_index=row['target_index'],entity_id=row['entity_id'],seq=row['seq'],
                                 history_length=row['history_length'],raw_quantity=row['raw_quantity'],
                                 last_quantity=row['last_quantity'],mean_quantity=row['mean_quantity'],
                                 candidate_prediction=float(a['predicted_raw_quantity'][i]),
                                 reference_prediction=float(b['predicted_raw_quantity'][i]),
                                 delta_se=float(ds[i]),delta_ae=float(abs(a['error'][i])-abs(b['error'][i]))))


def full_rows(path):
    receipt = read_json(path/'receipt.json')
    assert receipt['status']=='complete' and receipt['split']=='validation' and not receipt['heldout_read']
    frames=[]
    for part in receipt['parts']:
        file=record(path/part['path'],part['sha256'])
        frame=pl.read_parquet(file)
        assert len(frame)==part['rows']
        frames.append(frame)
    df=pl.concat(frames).sort('target_index')
    assert len(df)==receipt['prediction_rows']==receipt['expected_target_count']
    COVERAGE.append(dict(dataset=receipt['dataset'],model=receipt['model'],seed=receipt['seed'],split='validation',
                         scope='full_rows',n=len(df),selected_epoch=receipt['selected_epoch'],
                         checkpoint_sha256=receipt['checkpoint_file_sha256'],state_sha256=receipt['state_tensor_sha256'],
                         source_sha256=receipt['source_closure_sha256'],data_sha256=receipt['data_file_sha256'],
                         receipt=str(path.relative_to(ROOT)/'receipt.json'),runtime=receipt['runtime']))
    return df


def main():
    bounds_map={}
    for dataset in DATASETS:
        samples={}
        for model in (W4,ALL):
            path=SAMPLE/'runs'/f'{dataset}__{model}__seed42'
            r=read_json(path/'receipt.json')
            assert r['model']==model and r['seed']==42 and not r['heldout_read']
            bounds=r['quantity_bounds'];bounds_map[dataset]=bounds
            for split in ('train','validation'):
                s=r['splits'][split]
                df=pl.read_parquet(record(path/f'{split}_sample.parquet',s['prediction_sha256'])).sort('target_index')
                assert len(df)==s['sample_n'] and abs(df['weight'].sum()-s['population']['target_count'])<1e-6
                for cell in s['strata']:
                    assert cell['sample_n']<=cell['population_n']
                samples[model,split]=df
                describe(df,dataset,model,42,split,'weighted_stratified_sample',bounds)
                COVERAGE.append(dict(dataset=dataset,model=model,seed=42,split=split,scope='weighted_stratified_sample',n=len(df),
                                     represented_n=s['population']['target_count'],selected_epoch=r['selected_epoch'],
                                     checkpoint_sha256=r['checkpoint_sha256'],receipt=str(path.relative_to(ROOT)/'receipt.json'),runtime=r['runtime']))
            assert np.allclose(arrays(samples[model,'train'])['weight'].sum(),r['splits']['train']['population']['target_count'])
        for split in ('train','validation'):
            compare(samples[ALL,split],samples[W4,split],dataset,'all_available-width4',split,'weighted_stratified_sample',bounds)
        base_path=ROOT/'reports/titantpp_legacy_evaluation_20261003_v1/runs/validation_full/attempt1'/f'{dataset}__{W4}__seed42'
        base=full_rows(base_path)
        describe(base,dataset,W4,42,'validation','full_rows',bounds)
        for seed in (42,52,62):
            if seed==42 and dataset!='intermittent_frozen_5000':
                path=ROOT/'reports/titantpp_width16_test_20261003_v1/runs/validation'/dataset
            else:
                path=ROOT/'reports/titantpp_completed_validation_test_20261004_v1/width16/runs/validation'/f'{dataset}__{W16}__seed{seed}'
            df=full_rows(path)
            describe(df,dataset,W16,seed,'validation','full_rows',bounds)
            if seed==42:
                compare(df,base,dataset,'width16-width4','validation','full_rows',bounds)
                compare(df,samples[ALL,'validation'],dataset,'width16-all_available','validation','weighted_stratified_sample',bounds)
    write_csv('ROW_METRICS.csv',ROWS)
    write_csv('PAIRED_ROW_DELTAS.csv',PAIRS)
    write_csv('EXAMPLE_ROWS.csv',EXAMPLES)
    write_json('ROW_COVERAGE.json',COVERAGE)
    write_json('ROW_CHECKS.json',{'status':'complete','paired_checks':CHECKS,'disjoint_error_sum_checks':'passed',
                                'quantity_boundaries':bounds_map,'history_bin_edges':[2,4,8,16,32,64,128],
                                'history_bins':'<2; [2,4); [4,8); [8,16); [16,32); [32,64); [64,128); >=128',
                                'new_training':False,'new_inference':False,'scope':'Train/Validation only',
                                'missing':'width16 Train rows and width4/controls seed52/62 rows are not available in located stored diagnostics.'})
    write_json('ROW_INPUT_MANIFEST.json',INPUTS)
    print(json.dumps({'input_files':len(INPUTS),'coverage':len(COVERAGE),'row_metrics':len(ROWS),'paired_metrics':len(PAIRS),'example_rows':len(EXAMPLES)}))


if __name__=='__main__':
    main()
