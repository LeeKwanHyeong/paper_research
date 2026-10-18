"""Saved 상한 적용 전 Train/Validation rows only; no model imports or inference."""
from pathlib import Path
import csv
import hashlib
import json
import math
from datetime import datetime, timezone

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PRIOR = ROOT / 'reports/titantpp_raw_mlp_common_backbone_20261018_v1'
OLD = ROOT / 'reports/titantpp_mlp_train_validation_mechanism_20261017_v1'
W4 = 'titantpp_history_mlp'
W16 = 'titantpp_history_mlp_width16'
ALL = 'titantpp_all_available_history_mlp'
Q3_MANIFESTS = {
    'yellow_trip_hourly': 'benchmark_data/Cap/data/taxi/yellow_trip_hourly_split_manifest.json',
    'raf_spare_parts': 'benchmark_data/Cap/data/raf/raf_spare_parts_split_manifest.json',
    'intermittent_frozen_5000': 'benchmark_data/Cap/data/intermittent/intermittent_frozen_5000_split_manifest.json'}
INPUTS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path, expected=None):
    path = path.resolve()
    relative = str(path.relative_to(ROOT))
    assert '/runs/test/' not in relative and '/runs/heldout/' not in relative
    digest = sha(path)
    if expected is not None:
        assert digest == expected, relative
    INPUTS[relative] = digest
    return path


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def metric(pred, truth, weights, take, full_sse):
    count = int(take.sum())
    if not count:
        return dict(n=0, represented_weight=0., represented_share_percent=0.,
                    rmse=None, mae=None, bias=None, underprediction_share=None,
                    sse=0., sse_share_percent=0., true_quantity_mean=None,
                    predicted_quantity_mean=None)
    w, y, p = weights[take], truth[take], pred[take]
    error = p - y
    sse = float(np.sum(w * error ** 2))
    return dict(n=count, represented_weight=float(w.sum()),
                represented_share_percent=float(100 * w.sum() / weights.sum()),
                rmse=float(np.sqrt(np.average(error ** 2, weights=w))),
                mae=float(np.average(abs(error), weights=w)),
                bias=float(np.average(error, weights=w)),
                underprediction_share=float(np.average(error < 0, weights=w)),
                sse=sse, sse_share_percent=100 * sse / full_sse,
                true_quantity_mean=float(np.average(y, weights=w)),
                predicted_quantity_mean=float(np.average(p, weights=w)))


def pair_check(a, b):
    assert len(a) == len(b)
    for key in ('target_index', 'entity_id', 'seq', 'split'):
        assert a[key].to_list() == b[key].to_list(), key
    if 'target_id' in a.columns and 'target_id' in b.columns:
        assert a['target_id'].to_list() == b['target_id'].to_list()
    for key in ('raw_quantity', 'last_quantity', 'mean_quantity', 'history_length'):
        np.testing.assert_allclose(a[key].to_numpy(), b[key].to_numpy(), rtol=1e-7, atol=1e-6)
    if 'weight' in a.columns and 'weight' in b.columns:
        np.testing.assert_array_equal(a['weight'].to_numpy(), b['weight'].to_numpy())


def main():
    # Read a fixed input list; no glob of scientific results or Test reports.
    allow_path = record(PRIOR / 'INPUT_MANIFEST.json')
    allow = json.loads(allow_path.read_text())
    record(PRIOR / 'analyze_saved_rows.py')
    reference_path = record(PRIOR / 'REFERENCE_AND_REGIME_METRICS.csv')
    reference_metrics = list(csv.DictReader(reference_path.open()))
    coverage_path = record(OLD / 'ROW_COVERAGE.json', allow[str((OLD / 'ROW_COVERAGE.json').relative_to(ROOT))])
    coverage = json.loads(coverage_path.read_text())
    q3 = {}
    for ds, relative in Q3_MANIFESTS.items():
        path = record(ROOT / relative)
        transformation = json.loads(path.read_text())['transformation']
        assert transformation['fitted_on'] == 'train'
        assert transformation['quantile'] == .75
        assert transformation['quantile_fit_population'] == 'all_train_events_including_first_event'
        assert transformation['test_used_for_threshold_selection'] is False
        q3[ds] = float(transformation['cap_value'])
    selected = [c for c in coverage if
                (c['split'] == 'validation' and c['scope'] == 'full_rows'
                 and c['model'] in (W4, W16)) or
                (c['split'] == 'train' and c['scope'] == 'weighted_stratified_sample'
                 and c['model'] in (W4, ALL))]
    assert len(selected) == 18
    cache, bounds, cohorts = {}, {}, []
    for item in selected:
        rp = record(ROOT / item['receipt'], allow[item['receipt']])
        receipt = json.loads(rp.read_text())
        assert receipt['status'] == 'complete' and not receipt['heldout_read']
        if item['scope'] == 'weighted_stratified_sample':
            pp = rp.parent / 'train_sample.parquet'
            df = pl.read_parquet(record(pp, allow[str(pp.relative_to(ROOT))]))
            bounds[item['dataset']] = receipt['quantity_bounds']
            assert np.isclose(df['weight'].sum(), item['represented_n'])
        else:
            assert receipt['split'] == 'validation'
            parts = []
            for part in receipt['parts']:
                pp = rp.parent / part['path']
                assert allow[str(pp.relative_to(ROOT))] == part['sha256']
                parts.append(pl.read_parquet(record(pp, part['sha256'])))
            df = pl.concat(parts)
            assert len(df) == receipt['expected_target_count'] == receipt['prediction_rows']
        df = df.sort('target_index')
        assert len(df) == item['n'] and df['target_index'].n_unique() == len(df)
        assert set(df['split'].unique().to_list()) == {item['split']}
        key = item['dataset'], item['model'], int(item['seed']), item['split'], item['scope']
        assert key not in cache
        cache[key] = df
        cohorts.append({**item, 'train_q3': q3[item['dataset']]})
    pair_checks = []
    for ds in q3:
        base = cache[ds, W4, 42, 'validation', 'full_rows']
        for seed in (42, 52, 62):
            other = cache[ds, W16, seed, 'validation', 'full_rows']
            pair_check(base, other)
            pair_checks.append(dict(dataset=ds, pair=f'MLP16seed{seed}-MLP4seed42', split='validation', n=len(base),
                                    purpose='same target/truth/history support only; cross-seed errors are not a paired architecture effect'))
        train_a = cache[ds, W4, 42, 'train', 'weighted_stratified_sample']
        train_b = cache[ds, ALL, 42, 'train', 'weighted_stratified_sample']
        pair_check(train_a, train_b)
        pair_checks.append(dict(dataset=ds, pair='AllAvailable-MLP4seed42', split='train', n=len(train_a)))
    summary, history, observable = [], [], []
    reference_checks = 0
    for (ds, model, seed, split, scope), df in cache.items():
        y, p, last, mean = [df[k].to_numpy().astype(float) for k in
                            ('raw_quantity', 'predicted_raw_quantity', 'last_quantity', 'mean_quantity')]
        w = df['weight'].to_numpy().astype(float) if 'weight' in df.columns else np.ones(len(df))
        assert all(np.isfinite(a).all() for a in (y, p, last, mean, w)) and (w > 0).all()
        q99 = float(bounds[ds][-1])
        meta = dict(dataset=ds, model=model, seed=seed, split=split, scope=scope,
                    approximation='N_over_n_weighted_stratified_estimate' if scope != 'full_rows' else 'full_population',
                    train_q3=q3[ds], train_q99=q99, population_rows_or_represented_count=float(w.sum()))
        predictions = {'model': p, 'observed_last_reference': last, 'observed_mean_reference': mean}
        partitions = {'all': np.ones(len(y), dtype=bool), 'q_le_train_q3': y <= q3[ds],
                      'train_q3_lt_q_le_train_q99': (y > q3[ds]) & (y <= q99), 'q_gt_train_q99': y > q99}
        for predictor, pred in predictions.items():
            full_sse = float(np.sum(w * (pred - y) ** 2))
            for group, take in partitions.items():
                stats = metric(pred, y, w, take, full_sse)
                summary.append(dict(**meta, predictor=predictor, quantity_group=group,
                                    entities=int(df.filter(pl.Series(take))['entity_id'].n_unique()), **stats))
                if group in ('all', 'q_gt_train_q99'):
                    match = [r for r in reference_metrics if r['dataset'] == ds and r['model'] == model
                             and r['seed'] == str(seed) and r['split'] == split and r['scope'] == scope
                             and r['weighting'] == 'row' and r['predictor'] == predictor
                             and r['axis'] == ('overall' if group == 'all' else 'tail')
                             and r['group'] == ('all' if group == 'all' else 'tail')]
                    assert len(match) == 1
                    for k in ('rmse', 'mae', 'bias'):
                        assert np.isclose(stats[k], float(match[0][k]), rtol=1e-10, atol=1e-10)
                        reference_checks += 1
            assert np.isclose(sum(float(np.sum(w[t] * (pred[t] - y[t]) ** 2)) for g, t in partitions.items()
                                  if g != 'all'), full_sse, rtol=1e-12, atol=1e-9)
        # Fixed diagnostic thresholds, no fitting of a predictive routing rule.
        for label, threshold in (('train_q3', q3[ds]), ('train_q99', q99)):
            large = y > threshold
            conditions = {
                'all_large': large,
                'previous_also_above_same_threshold': large & (last > threshold),
                'crossed_from_previous_at_or_below_same_threshold': large & (last <= threshold),
                'previous_at_or_below_train_q3': large & (last <= q3[ds]),
                'history_mean_at_or_below_train_q3': large & (mean <= q3[ds]),
                'both_previous_and_mean_at_or_below_train_q3': large & (last <= q3[ds]) & (mean <= q3[ds]),
                'target_log1p_jump_over_2x_previous': large & ((y + 1) > 2 * (last + 1))}
            total_large_weight = float(w[large].sum())
            for predictor, pred in predictions.items():
                full_sse = float(np.sum(w * (pred - y) ** 2))
                for group, take in conditions.items():
                    stats = metric(pred, y, w, take, full_sse)
                    history.append(dict(**meta, predictor=predictor, threshold_name=label, threshold=threshold,
                                        history_group=group, percent_among_large=100 * stats['represented_weight'] / total_large_weight
                                        if total_large_weight else None, **stats))
        if model == W4 and seed == 42:
            signals = {'last_gt_train_q3': last > q3[ds], 'mean_gt_train_q3': mean > q3[ds],
                       'last_gt_train_q99': last > q99, 'mean_gt_train_q99': mean > q99}
            for label, threshold in (('train_q3', q3[ds]), ('train_q99', q99)):
                truth_large = y > threshold
                for signal, predicted_large in signals.items():
                    tp = float(w[truth_large & predicted_large].sum())
                    signal_w, true_w = float(w[predicted_large].sum()), float(w[truth_large].sum())
                    observable.append(dict(**meta, target_threshold_name=label, target_threshold=threshold,
                        observed_signal=signal, target_large_n=int(truth_large.sum()),
                        signal_n=int(predicted_large.sum()), joint_n=int((truth_large & predicted_large).sum()),
                        represented_true_large=true_w, represented_signal_positive=signal_w,
                        represented_true_positive=tp, represented_false_positive=signal_w - tp,
                        precision=tp / signal_w if signal_w else None, recall=tp / true_w if true_w else None,
                        base_large_rate=true_w / float(w.sum()),
                        interpretation='descriptive fixed observable support; not a trained predictor or proof of knowability'))
    assert len(summary) == 216 and len(history) == 756 and len(observable) == 48
    outputs = {'UNCAPPED_QUANTITY_METRICS.csv': summary, 'UNCAPPED_LARGE_HISTORY.csv': history,
               'UNCAPPED_OBSERVABLE_SUPPORT.csv': observable}
    for name, rows in outputs.items():
        write_csv(name, rows)
    assert all(sha(ROOT / path) == digest for path, digest in INPUTS.items())
    receipt = dict(schema='large_quantity_saved_uncapped_train_validation_v1', status='PASS',
        created_utc=datetime.now(timezone.utc).isoformat(), terminology='상한 적용 전 데이터; Cap은 사용자 지정 공식 기준 데이터',
        analysis_only=True, new_training=False, new_inference=False, remote_access=False, heldout_read=False,
        cnn_changed=False, input_sha256=INPUTS, thresholds={ds:dict(train_q3=q3[ds], train_q99=float(bounds[ds][-1])) for ds in q3},
        threshold_scope='Q3 from stored official Cap transformation metadata fitted on all Train events; Q99 frozen Train bounds from original diagnostic receipts. No Validation/Test quantile fit.',
        cohort_count=18, cohorts=cohorts, target_truth_history_pair_checks=pair_checks,
        prior_reference_metric_checks=reference_checks, quantity_partition_additivity='PASS',
        outputs={name:sha(OUT / name) for name in outputs},
        limitations=['All outcomes here are saved Train/Validation predictions on 상한 적용 전 quantities; not official Cap outcomes.',
                    'Train sample N/n-weighted quantities are estimates; sparse cells have large uncertainty and are retained.',
                    'Full Validation MLP4 rows cover seed42 only; MLP16 covers42/52/62. CurrentOnly Train/full Validation rows were not available in this input inventory.',
                    'Q3 and Q99 use previously fixed Train populations/conventions; Q3 includes the first Train event, while evaluation targets use observed-history eligibility.',
                    'Previous-at-or-below-threshold is a threshold crossing, not proof of a surprising or unpredictable event.',
                    'Large targets and conditional signed bias are retrospective; they do not identify a causal training failure or prove that input information is sufficient.',
                    'Observed last/mean references are deterministic saved-input comparisons, not external models or optimized forecasts.',
                    'Same target support across different seeds is checked but is not a within-seed architecture attribution.',
                    'CPU torch2.14 and CUDA torch2.11/cuDNN91900 source provenance remain in cohort metadata; tiny numerical differences are possible.',
                    'No independence, statistical significance, irreducible-error, data-cleansing or model-selection claim is made.'])
    receipt['analysis_code_sha256'] = sha(Path(__file__))
    with (OUT / 'UNCAPPED_RECEIPT.json').open('x') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(status='PASS', cohorts=len(cache), reference_checks=reference_checks,
                         outputs=receipt['outputs']), indent=2))


if __name__ == '__main__':
    main()
