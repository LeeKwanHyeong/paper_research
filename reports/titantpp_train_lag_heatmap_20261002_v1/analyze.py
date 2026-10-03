"""Train-only event-lag correlations, matching the existing Appendix B statistic.

No model imports, inference, training, remote access, or evaluation-result reads.
All parquet materialization is predicate-filtered to chronological_split=train.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import math
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LAGS = (1, 2, 4, 8, 16)
DATASETS = {
    'yellow_trip_hourly': 'Taxi',
    'intermittent_frozen_5000': 'Intermittent',
    'raf_spare_parts': 'RAF',
    'insta_market_basket': 'Instacart',
}
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def track(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return path


def load(path):
    return json.loads(track(path).read_text())


def local(path):
    return ROOT / str(path).split('paper_research/', 1)[1]


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def correlation(x, y, ids, series_count):
    n = np.bincount(ids, minlength=series_count)
    sx = np.bincount(ids, weights=x, minlength=series_count).astype('float64')
    sy = np.bincount(ids, weights=y, minlength=series_count).astype('float64')
    mx = np.divide(sx, n, out=np.zeros_like(sx), where=n > 0)
    my = np.divide(sy, n, out=np.zeros_like(sy), where=n > 0)
    dx, dy = x - mx[ids], y - my[ids]
    xx = np.bincount(ids, weights=dx * dx, minlength=series_count)
    yy = np.bincount(ids, weights=dy * dy, minlength=series_count)
    xy = np.bincount(ids, weights=dx * dy, minlength=series_count)
    denom = math.sqrt(float(xx.sum() * yy.sum()))
    value = float(xy.sum() / denom) if denom else None
    # Independent residual-vector Pearson identity; no averaging of per-series r.
    reference_denom = math.sqrt(float(np.dot(dx, dx) * np.dot(dy, dy)))
    reference = float(np.dot(dx, dy) / reference_denom) if reference_denom else None
    assert value is None and reference is None or math.isclose(value, reference, abs_tol=1e-12)
    if value is not None:
        assert np.isfinite(value) and abs(value) <= 1 + 1e-12
    eligible = (n >= 4) & (xx > 1e-12) & (yy > 1e-12)
    eligible_denom = math.sqrt(float(xx[eligible].sum() * yy[eligible].sum()))
    return dict(correlation=value, pairs=int(n.sum()), series_with_pairs=int((n > 0).sum()),
                series_min4_pairs_nonconstant=int(eligible.sum()),
                pairs_min4_pairs_nonconstant=int(n[eligible].sum()),
                correlation_min4_pairs_nonconstant=float(xy[eligible].sum() / eligible_denom) if eligible_denom else None,
                centered_x_sum_squares=float(xx.sum()), centered_y_sum_squares=float(yy.sum()),
                centered_cross_product=float(xy.sum()))


def main():
    profiles = load('reports/titantpp_dataset_appendix_20261002_v1/train_profiles.json')
    verification = load('reports/titantpp_dataset_appendix_20261002_v1/verification.json')
    assert verification['status'] == 'passed'
    core = load('search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json')
    raf = load('search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/original/execution_contract.json')
    specs = {s['dataset_id']: s for s in core['datasets'] + raf['datasets']}
    results, checks = [], []
    for ds, label in DATASETS.items():
        profile, spec = profiles[ds], specs[ds]
        identity = spec['inherited_data_identity']
        file_identity = spec['parent_data_identity'] if ds == 'raf_spare_parts' else identity
        path = track(profile['source'])
        assert SOURCES[profile['source']] == profile['sha256'] == file_identity['data']['sha256']
        manifest = track(local(file_identity['split_manifest']['path']))
        assert sha(manifest) == file_identity['split_manifest']['sha256']
        frame = (pl.scan_parquet(path).filter(pl.col('chronological_split') == 'train')
                 .select(['oper_part_no', 'seq', 'demand_qty', 'chronological_split'])
                 .collect().sort(['oper_part_no', 'seq']))
        assert frame['chronological_split'].unique().to_list() == ['train']
        assert frame.null_count().sum_horizontal().item() == 0
        assert frame.select(pl.struct(['oper_part_no', 'seq']).is_duplicated().sum()).item() == 0
        groups = frame.group_by('oper_part_no', maintain_order=True).len()
        parts = groups['oper_part_no'].to_list()
        sizes = groups['len'].to_numpy().astype('int64')
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        ids = np.repeat(np.arange(len(sizes)), sizes)
        pos = np.arange(frame.height) - starts[ids]
        seq = frame['seq'].to_numpy().astype('int64')
        q = frame['demand_qty'].to_numpy().astype('float64')
        assert np.isfinite(q).all() and (q >= 0).all()
        targets = np.flatnonzero(pos > 0)
        pop = identity['populations']['train']
        h = hashlib.sha256(b'hard_lmm_target_identity_v1\0train\0')
        h.update(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode())
        for arr in [ids[targets], pos[targets], seq[targets]]:
            h.update(arr.astype('<i8').tobytes())
        target_q = q[targets].astype('float32').astype('<f8')
        qh = hashlib.sha256(b'hard_lmm_target_quantity_v1\0' + target_q.tobytes()).hexdigest()
        assert h.hexdigest() == pop['target_identity_sha256']
        assert qh == pop['target_quantity_sha256']
        assert len(q) == profile['train_rows'] and len(sizes) == profile['train_series']
        assert len(targets) == profile['train_targets'] == pop['target_count']
        logq = np.log1p(q)
        counts = []
        for lag in LAGS:
            target = np.flatnonzero(pos >= lag)
            prev = target - lag
            assert np.array_equal(ids[prev], ids[target])
            assert np.all(seq[prev] < seq[target])
            assert len(target) == np.maximum(sizes - lag, 0).sum()
            c = correlation(logq[prev], logq[target], ids[target], len(sizes))
            row = dict(dataset=ds, label=label, split='train', event_lag=lag, **c)
            # Same-series sensitivity: retain >=20-event series for every lag,
            # so each retained series has >=4 pairs even at lag16.
            common = sizes[ids[target]] >= max(LAGS) + 4
            row['common_cohort_min20_train_events'] = correlation(
                logq[prev[common]], logq[target[common]], ids[target[common]], len(sizes))
            results.append(row)
            counts.append(c['pairs'])
            if lag == 1:
                expected = profile['adjacent_log_quantity_correlation']
                assert math.isclose(c['correlation'], expected['within_entity_demeaned'], abs_tol=1e-12)
                assert c['pairs'] == expected['pairs']
                assert c['series_with_pairs'] == expected['entities_with_pairs']
                assert c['series_min4_pairs_nonconstant'] == expected['eligible_entities']
        assert counts == sorted(counts, reverse=True)
        checks.append(dict(dataset=ds, train_rows=len(q), train_series=len(sizes),
                           target_identity_sha256=h.hexdigest(), target_quantity_sha256=qh,
                           data_sha_matches=True, train_identity_matches=True, lag1_matches_existing=True,
                           pair_counts_match_series_lengths=True, no_cross_series_pairs=True))
        print(label, [(r['event_lag'], round(r['correlation'], 6) if r['correlation'] is not None else None,
                       r['pairs'], r['series_with_pairs']) for r in results if r['dataset'] == ds], flush=True)
    methodology = dict(split='train only', transform='log1p demand_qty', lags=list(LAGS),
        pairing='all within-series train pairs j-lag,j; event order, not calendar time; no loader window restriction',
        centering='separate per-series per-lag means for predecessor and successor members',
        aggregation='pooled Pearson of centered pairs, pair weighted; not mean of individual series correlations',
        zero_variance='undefined correlation is null; single-pair series contribute zero centered sums',
        sensitivities='min4 nonconstant series at each lag, and fixed cohort of series with >=20 train events',
        interpretation='descriptive association; composition changes with lag; short-series centering can affect negative values',
        architecture='lags are descriptive offsets; all eight correction branches read the immediate predecessor')
    dump('analysis.json', dict(created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        methodology=methodology, rows=results, identities=checks,
        new_training=False, new_inference=False, evaluation_predictions_or_metrics_read=False))
    flat = [{k: v for k, v in r.items() if not isinstance(v, dict)} for r in results]
    with (OUT / 'correlations.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    track(__file__)
    dump('sources.json', SOURCES)
    dump('verification.json', dict(status='passed', datasets=4, lag_cells=len(results),
        identity_checks=checks, existing_lag1_agreement_tolerance=1e-12,
        all_pairs_stay_in_series=True, all_counts_match_lengths=True,
        independent_centered_sum_and_dot_correlations_match=True,
        train_filter_before_materialization=True, heldout_rows_or_results_materialized=False,
        new_model_fit_forward_or_replay=False))


if __name__ == '__main__':
    main()
