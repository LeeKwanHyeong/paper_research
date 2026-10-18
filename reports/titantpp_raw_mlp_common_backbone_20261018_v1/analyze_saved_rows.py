"""Descriptive, allowlisted uncapped Train/Validation analysis. No model imports."""
from pathlib import Path
from collections import defaultdict
import csv
import hashlib
import json
import math

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'reports/titantpp_mlp_train_validation_mechanism_20261017_v1'
W4 = 'titantpp_history_mlp'
W16 = 'titantpp_history_mlp_width16'
ALL = 'titantpp_all_available_history_mlp'
CURRENT = 'titantpp_current_only_param_matched'
INPUTS = {}
ALLOW = {}
METRICS, MATCHES, PAIRS, CHECKS = [], [], [], []


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def record(rel, expected=None, bootstrap=False):
    p = (ROOT / rel).resolve()
    rel = str(p.relative_to(ROOT))
    assert '/runs/test/' not in rel and '/runs/heldout/' not in rel
    actual = digest(p)
    if not bootstrap:
        assert rel in ALLOW, ('not allowlisted', rel)
        assert actual == ALLOW[rel]
    if expected:
        assert actual == expected, rel
    INPUTS[rel] = actual
    return p


def read(rel):
    return json.loads(record(rel).read_text())


def write_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metrics(error, w):
    assert len(error) and np.isfinite(error).all() and np.isfinite(w).all() and (w > 0).all()
    return dict(rmse=float(np.sqrt(np.average(error ** 2, weights=w))),
                mae=float(np.average(abs(error), weights=w)),
                bias=float(np.average(error, weights=w)))


def entity_weights(w, entities):
    totals = defaultdict(float)
    for e, v in zip(entities, w):
        totals[e] += v
    return w / np.array([totals[e] for e in entities])


def arrays(df, bounds):
    assert df['target_index'].n_unique() == len(df)
    assert set(df['split'].unique().to_list()) <= {'train', 'validation'}
    a = {k: df[k].to_numpy().astype(float) for k in (
        'raw_quantity', 'predicted_raw_quantity', 'last_quantity', 'mean_quantity', 'history_length')}
    assert all(np.isfinite(v).all() for v in a.values())
    assert (a['history_length'] >= 1).all() and (a['raw_quantity'] >= 0).all()
    a['entity'] = np.array(df['entity_id'].to_list())
    a['weight'] = df['weight'].to_numpy().astype(float) if 'weight' in df.columns else np.ones(len(df))
    assert np.isfinite(a['weight']).all() and (a['weight'] > 0).all()
    z = np.log1p(a['last_quantity']) - np.log1p(a['mean_quantity'])
    a['regime'] = np.where(z < -math.log(2), 'drop', np.where(z > math.log(2), 'rise', 'middle'))
    a['qbin'] = np.searchsorted(bounds, a['raw_quantity'], side='left')
    a['branches'] = sum(a['history_length'] > t for t in [1, 2, 4, 8, 16, 32, 64, 128])
    a['errors'] = {
        'model': a['predicted_raw_quantity'] - a['raw_quantity'],
        'observed_last_reference': a['last_quantity'] - a['raw_quantity'],
        'observed_mean_reference': a['mean_quantity'] - a['raw_quantity']}
    return a


def groups(a, bounds):
    n = len(a['weight'])
    yield 'overall', 'all', np.ones(n, dtype=bool)
    for label in ('drop', 'middle', 'rise'):
        yield 'regime', label, a['regime'] == label
    for q in range(len(bounds) + 1):
        yield 'quantity_bin', str(q), a['qbin'] == q
        for label in ('drop', 'middle', 'rise'):
            yield 'quantity_regime', f'{q}:{label}', (a['qbin'] == q) & (a['regime'] == label)
    yield 'tail', 'tail', a['raw_quantity'] > bounds[-1]
    yield 'tail', 'non_tail', a['raw_quantity'] <= bounds[-1]


def common_weights(keys, regime, source_w, entities, other, weighting):
    """Overlap mass per cell; equal entity mass only among supported entities."""
    if weighting == 'entity_equal':
        # Entity must be part of the matching cell, otherwise reweighting entities
        # after coarse matching would break equal truth-bin composition.
        keys = list(zip(entities, keys))
    cells = defaultdict(lambda: [[], []])
    for i, (key, r) in enumerate(zip(keys, regime)):
        if r in (other, 'middle'):
            cells[key][int(r == 'middle')].append(i)
    left, right = np.zeros(len(regime)), np.zeros(len(regime))
    common = 0
    for ix, iy in cells.values():
        if not ix or not iy:
            continue
        sx, sy = source_w[ix].sum(), source_w[iy].sum()
        mass = min(sx, sy)
        left[ix] = source_w[ix] * mass / sx
        right[iy] = source_w[iy] * mass / sy
        common += 1
    if common and weighting == 'entity_equal':
        # Match entity composition as well as the requested cell composition.
        ex, ey = defaultdict(float), defaultdict(float)
        for i, e in enumerate(entities):
            ex[e] += left[i]
            ey[e] += right[i]
        for i, e in enumerate(entities):
            if ex[e] and ey[e]:
                mass = min(ex[e], ey[e])
                left[i] *= mass / ex[e]
                right[i] *= mass / ey[e]
            else:
                left[i] = right[i] = 0
        ex2 = defaultdict(float)
        for e, v in zip(entities, left):
            ex2[e] += v
        for i, e in enumerate(entities):
            if ex2[e]:
                left[i] /= ex2[e]
                right[i] /= ex2[e]
    return left, right, common


def describe(df, meta, bounds):
    a = arrays(df, bounds)
    for axis, group, idx in groups(a, bounds):
        if not idx.any():
            continue
        n, entities = int(idx.sum()), len(set(a['entity'][idx]))
        for weighting in ('row', 'entity_equal'):
            w = a['weight'][idx].copy()
            if weighting == 'entity_equal':
                w = entity_weights(w, a['entity'][idx])
            for predictor, error in a['errors'].items():
                METRICS.append(dict(**meta, axis=axis, group=group, weighting=weighting,
                                    predictor=predictor, n=n, entities=entities,
                                    sparse_support=n < 30 or entities < 5,
                                    weight_sum=float(w.sum()), **metrics(error[idx], w)))
    scheme_keys = {
        'truth_bin': list(a['qbin']),
        'exact_truth': list(a['raw_quantity']),
        'entity_truthbin_branches': list(zip(a['entity'], a['qbin'], a['branches'])),
        'entity_exacttruth_branches': list(zip(a['entity'], a['raw_quantity'], a['branches']))}
    for focus, subset in [('all', np.ones(len(df), bool)), ('lowest_truth_bin', a['qbin'] == 0)]:
        for scheme, keys in scheme_keys.items():
            # Restrict support before matching; excluded values cannot enter either regime.
            regimes = np.where(subset, a['regime'], 'excluded')
            for other in ('drop', 'rise'):
                for weighting in ('row', 'entity_equal'):
                    x, y, cells = common_weights(keys, regimes, a['weight'], a['entity'], other, weighting)
                    ix, iy = x > 0, y > 0
                    base = dict(**meta, focus=focus, scheme=scheme, comparison=other + '-middle',
                                weighting=weighting, common_cells=cells,
                                n_other=int(ix.sum()), n_middle=int(iy.sum()),
                                entities_other=len(set(a['entity'][ix])), entities_middle=len(set(a['entity'][iy])),
                                available_other=int((regimes == other).sum()),
                                available_middle=int((regimes == 'middle').sum()),
                                matched_mass_other=float(x.sum()), matched_mass_middle=float(y.sum()))
                    if not ix.any() or not iy.any():
                        MATCHES.append(dict(**base, status='N/A_no_common_support', predictor='all'))
                        continue
                    assert np.isclose(x.sum(), y.sum(), rtol=1e-10, atol=1e-7)
                    base['sparse_support'] = min(ix.sum(), iy.sum()) < 30 or min(base['entities_other'], base['entities_middle']) < 5
                    for predictor, error in a['errors'].items():
                        xm, ym = metrics(error[ix], x[ix]), metrics(error[iy], y[iy])
                        row = dict(**base, status='computed', predictor=predictor)
                        for k in xm:
                            row['other_' + k], row['middle_' + k] = xm[k], ym[k]
                            row['delta_' + k] = xm[k] - ym[k]
                        MATCHES.append(row)
    return a


def pair(df_a, df_b, meta, bounds):
    assert len(df_a) == len(df_b)
    for k in ('target_index', 'entity_id', 'seq', 'split'):
        assert df_a[k].to_list() == df_b[k].to_list(), (meta, k)
    # Legacy Train shards have heterogeneous optional metadata. Never invent IDs;
    # the four mandatory identity columns and saved truths/history must match.
    if 'target_id' in df_a.columns and 'target_id' in df_b.columns:
        assert df_a['target_id'].to_list() == df_b['target_id'].to_list()
    for k in ('raw_quantity', 'history_length', 'last_quantity', 'mean_quantity'):
        assert np.allclose(df_a[k].to_numpy(), df_b[k].to_numpy(), rtol=1e-7, atol=1e-6)
    a, b = arrays(df_a, bounds), arrays(df_b, bounds)
    assert np.array_equal(a['weight'], b['weight'])
    CHECKS.append(dict(**meta, exact_target_pair_count=len(df_a)))
    for axis, group, idx in groups(a, bounds):
        if not idx.any():
            continue
        for weighting in ('row', 'entity_equal'):
            w = a['weight'][idx].copy()
            if weighting == 'entity_equal':
                w = entity_weights(w, a['entity'][idx])
            am, bm = metrics(a['errors']['model'][idx], w), metrics(b['errors']['model'][idx], w)
            row = dict(**meta, axis=axis, group=group, weighting=weighting, n=int(idx.sum()), entities=len(set(a['entity'][idx])))
            for k in am:
                row['candidate_' + k], row['reference_' + k] = am[k], bm[k]
                row['delta_' + k] = am[k] - bm[k]
            PAIRS.append(row)


def self_checks():
    assert metrics(np.array([1., 3.]), np.array([3., 1.]))['mae'] == 1.5
    keys, regimes = ['a', 'a', 'b', 'b', 'c'], np.array(['drop', 'middle', 'drop', 'middle', 'drop'])
    w, e = np.array([2., 4., 3., 1., 9.]), np.array(['x', 'x', 'y', 'y', 'z'])
    x, y, n = common_weights(keys, regimes, w, e, 'drop', 'row')
    assert n == 2 and np.allclose(x, [2, 0, 1, 0, 0]) and np.allclose(y, [0, 2, 0, 1, 0])
    x, y, _ = common_weights(keys, regimes, w, e, 'drop', 'entity_equal')
    assert np.allclose(x, [1, 0, 1, 0, 0]) and np.allclose(y, [0, 1, 0, 1, 0])
    x, y, n = common_weights(['a', 'b'], np.array(['drop', 'middle']), np.ones(2), np.array(['x', 'y']), 'drop', 'row')
    assert n == 0 and x.sum() == y.sum() == 0


def main():
    self_checks()
    plan_path = OUT / 'ANALYSIS_PLAN.json'
    plan = json.loads(plan_path.read_text())
    record(str(plan_path.relative_to(ROOT)), bootstrap=True)
    for rel, expected in plan['input_manifests'].items():
        obj = json.loads(record(rel, expected, bootstrap=True).read_text())
        if isinstance(obj, dict):
            ALLOW.update(obj)
    audit_path = OUT / 'COVERAGE_AUDIT.json'
    audit = json.loads(record(str(audit_path.relative_to(ROOT)), 'cde7f6157fb2f8162eb3ce49d6b069312977d2504140d939ebf6e2e0794fb6e8', bootstrap=True).read_text())
    ALLOW.update(audit['qualification_input_sha256'])
    for rel, expected in ALLOW.items():
        record(rel, expected)
    coverage = json.loads((OLD / 'ROW_COVERAGE.json').read_text())
    cache, bounds_map = {}, {}
    for item in coverage:
        r = read(item['receipt'])
        assert r['status'] == 'complete' and not r['heldout_read']
        directory = Path(item['receipt']).parent
        if item['scope'] == 'weighted_stratified_sample':
            df = pl.read_parquet(record(str(directory / (item['split'] + '_sample.parquet'))))
            bounds_map[item['dataset']] = r['quantity_bounds']
            assert np.isclose(df['weight'].sum(), item['represented_n'])
        else:
            assert r['split'] == 'validation'
            df = pl.concat([pl.read_parquet(record(str(directory / p['path']), p['sha256'])) for p in r['parts']])
            assert len(df) == r['prediction_rows'] == r['expected_target_count']
        df = df.sort('target_index')
        assert len(df) == item['n']
        key = item['dataset'], item['model'], item['seed'], item['split'], item['scope']
        assert key not in cache
        cache[key] = df
    for item in audit['qualification_entries']:
        if not item.get('usable_for_same_native_qualification_comparison'):
            continue
        r = read(item['receipt_path'])
        assert r['status'] == 'complete' and r['split'] == 'validation' and not r['heldout_read']
        df = pl.concat([pl.read_parquet(record(p['path'], p['sha256'])) for p in item['parts']]).sort('target_index')
        assert len(df) == 512
        cache[item['dataset'], item['model'], 42, 'validation', 'qualification_first512'] = df
    for (ds, model, seed, split, scope), df in cache.items():
        describe(df, dict(dataset=ds, model=model, seed=seed, split=split, scope=scope), bounds_map[ds])
    for ds in bounds_map:
        for split in ('train', 'validation'):
            scope = 'weighted_stratified_sample'
            pair(cache[ds, ALL, 42, split, scope], cache[ds, W4, 42, split, scope],
                 dict(dataset=ds, pair='all_available-width4', seed=42, split=split, scope=scope), bounds_map[ds])
        scope = 'full_rows'
        pair(cache[ds, W16, 42, 'validation', scope], cache[ds, W4, 42, 'validation', scope],
             dict(dataset=ds, pair='width16-width4', seed=42, split='validation', scope=scope), bounds_map[ds])
        scope = 'qualification_first512'
        for candidate, reference in [(ALL, W4), (CURRENT, W4), (ALL, CURRENT)]:
            pair(cache[ds, candidate, 42, 'validation', scope], cache[ds, reference, 42, 'validation', scope],
                 dict(dataset=ds, pair=candidate + '-' + reference, seed=42, split='validation', scope=scope), bounds_map[ds])
    write_csv('REFERENCE_AND_REGIME_METRICS.csv', METRICS)
    write_csv('STANDARDIZED_REGIME_COMPARISONS.csv', MATCHES)
    write_csv('PAIRED_MODEL_DELTAS.csv', PAIRS)
    write_json('INPUT_MANIFEST.json', INPUTS)
    write_json('ANALYSIS_CHECKS.json', dict(status='complete_descriptive_saved_rows', input_sha_count=len(ALLOW),
        row_cohort_count=len(cache), metric_rows=len(METRICS), standardized_rows=len(MATCHES),
        no_common_support_rows=sum(r['status'] != 'computed' for r in MATCHES), paired_checks=CHECKS,
        self_checks='weighted metrics, overlap masses, equal entity masses, no-support handling passed',
        quantity_bounds=bounds_map, new_training_count=0, new_inference_count=0, remote_calls=0,
        heldout_read=False, cnn_modified=False,
        limits=['No missing Train rows generated', 'No causal interpretation of truth-conditioned standardization',
                'First512 qualification rows are auxiliary, not population representative', 'No dataset pooling']))
    print(json.dumps(dict(cohorts=len(cache), metric_rows=len(METRICS), standardized_rows=len(MATCHES), pairs=len(CHECKS))))


if __name__ == '__main__':
    main()
