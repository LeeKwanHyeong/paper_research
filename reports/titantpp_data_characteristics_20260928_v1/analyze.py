"""CPU-only description of frozen training rows and saved validation aggregates.

No model loading, fitting, prediction replay, remote calls, or held-out row reads.
Run from anywhere with /usr/local/bin/python3 <this file>.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'search_artifacts/titantpp_core_ablation_20260928_v1'
CONTRACT = CAMPAIGN / 'frozen_execution/execution_contract.json'
# Pin the inspected snapshot: later monitoring must not silently change this analysis.
LEDGER = CAMPAIGN / 'aggregation/20260928T015549834592Z/ledger.json'
EXPECTED = 'eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d'
LAGS = np.array([1, 2, 4, 8, 16, 32, 64, 128])


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def distribution(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std()),
                **{k: float(v) for k, v in zip(
                    ['min', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99', 'max'],
                    np.quantile(a, [0, .05, .25, .5, .75, .95, .99, 1], method='nearest'))})


def correlation(x, y, ids):
    """Pooled and pair-wise within-entity demeaned descriptive correlations."""
    n = np.bincount(ids)
    sx, sy = np.bincount(ids, x), np.bincount(ids, y)
    mx = np.divide(sx, n, out=np.zeros_like(sx), where=n > 0)
    my = np.divide(sy, n, out=np.zeros_like(sy), where=n > 0)
    dx, dy = x - mx[ids], y - my[ids]
    xx, yy, xy = np.bincount(ids, dx*dx), np.bincount(ids, dy*dy), np.bincount(ids, dx*dy)
    valid = (n >= 4) & (xx > 1e-12) & (yy > 1e-12)
    den = math.sqrt(float(np.dot(dx, dx) * np.dot(dy, dy)))
    pooled = float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else None
    return {'pairs': len(x), 'pooled': pooled,
            'within_entity_demeaned': float(np.dot(dx, dy)/den) if den else None,
            'entity_correlations_min4_pairs_nonconstant': distribution(xy[valid]/np.sqrt(xx[valid]*yy[valid])),
            'entities_with_pairs': int((n > 0).sum()),
            'eligible_entities': int(valid.sum()), 'excluded_entities': int(((n > 0) & ~valid).sum())}


def profile(spec):
    identity = spec['inherited_data_identity']
    path = ROOT / identity['data']['path'].split('paper_research/')[1]
    manifest = ROOT / identity['split_manifest']['path'].split('paper_research/')[1]
    assert digest(path) == identity['data']['sha256']
    assert digest(manifest) == identity['split_manifest']['sha256']
    # Predicate pushdown; validation and held-out rows never materialized.
    frame = (pl.scan_parquet(path).filter(pl.col('chronological_split') == 'train')
             .collect().sort(['oper_part_no', 'seq']))
    required = ['oper_part_no', 'seq', 'delta_t', 'demand_qty', 'chronological_split']
    assert frame.select(required).null_count().sum_horizontal().item() == 0
    assert frame.select(pl.struct(['oper_part_no','seq']).is_duplicated().sum()).item() == 0
    assert frame['chronological_split'].unique().to_list() == ['train']
    groups = frame.group_by('oper_part_no', maintain_order=True).len()
    parts, sizes = groups['oper_part_no'].to_list(), groups['len'].to_numpy().astype('int64')
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    ends = starts + sizes
    ids = np.repeat(np.arange(len(sizes)), sizes)
    pos = np.arange(frame.height) - starts[ids]
    seq = frame['seq'].to_numpy().astype('int64')
    q = frame['demand_qty'].to_numpy().astype('float64')
    dt = frame['delta_t'].to_numpy().astype('float64')
    assert np.isfinite(q).all() and np.isfinite(dt).all() and (q >= 0).all() and (dt >= 0).all()
    target = np.flatnonzero(pos > 0)
    tid, prev = ids[target], target-1
    tq = q[target].astype('float32').astype('float64')  # loader's exact cast
    h = hashlib.sha256(b'hard_lmm_target_identity_v1\0train\0')
    h.update(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode())
    for a in [tid, pos[target], seq[target]]:
        h.update(a.astype('<i8').tobytes())
    qh = hashlib.sha256(b'hard_lmm_target_quantity_v1\0'+tq.astype('<f8').tobytes()).hexdigest()
    pop = identity['populations']['train']
    assert len(target) == pop['target_count']
    assert h.hexdigest() == pop['target_identity_sha256']
    assert qh == pop['target_quantity_sha256']
    W, cap = spec['loader']['lookback_weeks'], spec['loader']['max_seq_len']-1
    stride = int(seq.max()-seq.min()+W+1)
    keyed = seq + ids * stride
    left = np.maximum(np.searchsorted(keyed, keyed[prev]-(W-1)), starts[tid])
    left = np.maximum(left, target-cap)
    hist = target-left
    assert (hist >= 1).all() and (hist <= cap).all()
    for z in np.unique(np.r_[0, len(target)-1, np.linspace(0, len(target)-1, 30, dtype=int)]):
        s, e, i = starts[tid[z]], ends[tid[z]], pos[prev[z]]
        j = int(np.searchsorted(seq[s:e], seq[prev[z]]-(W-1), side='left'))
        assert int(hist[z]) == min(i-j+1, cap)
    assert np.array_equal(dt[target], seq[target]-seq[prev])
    logq = np.log1p(q)
    assert abs(float(logq.mean()) - spec['statistics']['train_log_mean']) < 1e-12
    assert abs(float(logq.std()) - spec['statistics']['train_log_std']) < 1e-12
    means = np.bincount(ids, q)/sizes
    variances = np.maximum(np.bincount(ids, (q-means[ids])**2)/sizes, 0)
    cvs = np.sqrt(variances)/means
    spans = seq[ends-1]-seq[starts]+1
    sq_by_series = np.bincount(tid, tq*tq, minlength=len(sizes))
    topn = max(1, math.ceil(.01*len(sizes)))
    allq99 = float(np.quantile(q, .99, method='nearest'))
    two = hist >= 2
    observed_log_change = np.abs(logq[prev[two]]-logq[prev[two]-1])
    three = pos[target] >= 2
    d1 = q[prev[three]]-q[prev[three]-1]
    d2 = tq[three]-q[prev[three]]
    both = (d1 != 0) & (d2 != 0)
    # These are descriptive, non-fitted, causal train examples, never evaluation results.
    smooth_rows = hist >= 4
    t = target[smooth_rows]
    cs = np.r_[0., q.cumsum()]
    mean4 = (cs[t]-cs[t-4])/4
    last = q[t-1]
    costs = {}
    for label, pred in [('last_observed', last), ('past4_mean', mean4)]:
        err = pred-q[t]
        costs[label] = dict(count=len(t), rmse=float(np.sqrt(np.mean(err*err))),
                            mae=float(np.mean(abs(err))),
                            log_rmse=float(np.sqrt(np.mean((np.log1p(pred)-logq[t])**2))))
    drift = []
    for s,e in zip(starts, ends):
        n=e-s
        if n >= 10:
            k=n//3
            drift.append(float(logq[e-k:e].mean()-logq[s:s+k].mean()))
    result = {
        'source': str(path.relative_to(ROOT)), 'sha256': digest(path),
        'split': 'train', 'quality_checks_passed': True,
        'target_identity_matches_frozen_contract': True,
        'train_rows': len(q), 'train_series': len(sizes), 'train_targets': len(target),
        'quantity_all_train_rows': distribution(q), 'quantity_canonical_targets': distribution(tq),
        'zero_quantity_rows': int((q==0).sum()), 'quantity_one_row_share': float(np.mean(q==1)),
        'integer_quantity_share': float(np.mean(q==np.floor(q))),
        'quantity_tail_gt_train_p99': {'threshold': allq99, 'target_share': float(np.mean(tq>allq99)),
            'share_of_target_q_squared': float(np.sum(tq[tq>allq99]**2)/np.sum(tq*tq))},
        'top1pct_entities_by_target_q_squared_share': float(np.sort(sq_by_series)[-topn:].sum()/sq_by_series.sum()),
        'top1pct_entity_count': topn,
        'per_entity_train_events': distribution(sizes), 'per_entity_quantity_mean': distribution(means),
        'per_entity_quantity_cv_population_sd': distribution(cvs),
        'constant_quantity_entities_share': float(np.mean(variances < 1e-12)),
        'observed_span_active_bucket_fraction_per_entity': distribution(sizes/spans),
        'time_unit': spec['model']['time_observation_contract']['unit'],
        'target_delta_t': distribution(dt[target]),
        'target_delta_t_one_share': float(np.mean(dt[target]==1)),
        'target_delta_t_gt_one_share': float(np.mean(dt[target]>1)),
        'top_code': spec['model']['time_observation_contract']['top_code'],
        'target_delta_t_eq_30_share': float(np.mean(dt[target]==30)),
        'target_delta_t_gt_30_count': int(np.sum(dt[target]>30)),
        'first_history_dt_zero_count_clipped_to_one': int((dt[starts]==0).sum()),
        'lookback_in_recorded_time_units': W, 'max_history_cap': cap,
        'actual_target_history_events': distribution(hist),
        'actual_target_history_recorded_span': distribution(seq[prev]-seq[left]+1),
        'target_history_le_shares': {str(k):float(np.mean(hist<=k)) for k in [1,3,7,15,31,63,127]},
        'branch_availability_at_final_observed_position': {str(k):float(np.mean(hist>k)) for k in LAGS},
        'mean_available_branches': float(sum(np.mean(hist>k) for k in LAGS)),
        'adjacent_equal_quantity_share': float(np.mean(tq==q[prev])),
        'adjacent_abs_log_quantity_change': distribution(abs(logq[target]-logq[prev])),
        'past_only_abs_log_change_for_targets_with_two_history_events': distribution(observed_log_change),
        'adjacent_quantity_correlation': correlation(q[prev], tq, tid),
        'adjacent_log_quantity_correlation': correlation(logq[prev], logq[target], tid),
        'adjacent_log_quantity_correlation_long_entities': {
            str(n):correlation(logq[prev[sizes[tid]>=n]], logq[target[sizes[tid]>=n]], tid[sizes[tid]>=n])
            for n in [20, 40]},
        'quantity_current_gap_log_correlation': correlation(np.log1p(dt[target]), logq[target], tid),
        'two_successive_nonzero_changes': int(both.sum()),
        'sign_reversal_share_given_two_nonzero_changes': float(np.mean(d1[both]*d2[both]<0)),
        'train_only_naive_descriptions_same_history_ge4_population': costs,
        'early_vs_late_third_train_mean_log_quantity_difference': distribution(drift),
        'early_late_eligible_entities': len(drift),
    }
    if spec['dataset_id']=='yellow_trip_hourly':
        ts = frame['time_bucket'].cast(pl.Int64).to_numpy()
        assert np.all(ts[target]-ts[prev] == (seq[target]-seq[prev])*3600*1_000_000)
        result['hourly_seq_matches_wall_clock_within_train'] = True
        result['train_dates'] = [str(frame['time_bucket'].min()), str(frame['time_bucket'].max())]
        result['time_bucket_and_grid_metadata'] = {'grid_degrees':frame['grid_size_deg'].unique().to_list(),
                                                   'min_active_buckets':frame['min_active_buckets'].unique().to_list()}
        lag_rows = {}
        for lag in [24,168]:
            at = np.searchsorted(keyed,keyed-lag)
            valid = (at < len(keyed)) & (ids[np.minimum(at,len(keyed)-1)]==ids)
            valid &= keyed[np.minimum(at,len(keyed)-1)]==keyed-lag
            lag_rows[str(lag)] = correlation(q[at[valid]],q[valid],ids[valid])
        result['exact_clock_lag_quantity_correlations_hours'] = lag_rows
        hours = frame['time_bucket'].dt.hour().to_numpy()
        ratio = q / means[ids]
        result['hour_of_day_entity_normalized_quantity_mean_event_weighted'] = {
            str(k):float(ratio[hours==k].mean()) for k in range(24)}
    return result


def aggregate_validation(ledger):
    complete = [r for r in ledger['conditions'] if r['execution_status']=='complete']
    out=[]
    for r in complete:
        s=r['selected']
        assert r['audit_status']=='passed'
        assert s['evaluation_scope']=='validation_only' and s['held_out_test_evaluated'] is False
        assert abs(sum(x['qty_sse'] for x in s['quantity_cells'])-s['qty_sse']) < max(.01,s['qty_sse']*1e-6)
        assert sum(x['count'] for x in s['quantity_cells'])==s['count']
        out.append({k:r[k] for k in ['dataset','arm','seed','best_epoch','completed_epochs','parameters','reused']}
                   | {'selected':s, 'source':r['source']})
    summary=[]
    for dataset in sorted({r['dataset'] for r in out}):
        for arm in sorted({r['arm'] for r in out if r['dataset']==dataset}):
            rows=[r for r in out if r['dataset']==dataset and r['arm']==arm]
            entry={'dataset':dataset,'arm':arm,'n_seeds':len(rows),'seeds':[r['seed'] for r in rows]}
            for metric in ['qty_rmse','qty_mae','time_nll']:
                vals=[r['selected'][metric] for r in rows]
                entry[metric+'_mean']=float(np.mean(vals))
                entry[metric+'_sample_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
            summary.append(entry)
    return {'completed_conditions':len(out), 'conditions':out, 'summary':summary,
            'uncompleted_conditions':[{'dataset':r['dataset'],'arm':r['arm'],'seed':r['seed'],
                                       'execution_status':r['execution_status']} for r in ledger['conditions']
                                      if r['execution_status']!='complete']}


def main():
    contract=json.loads(CONTRACT.read_text())
    canonical=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert canonical==EXPECTED
    relevant=['data_loader/event_seq_data_module.py','models/TPPs/CountAwareTPP.py',
              'models/TPPs/CountAwareTitanMultiLagDetail.py','models/TPPs/CountAwareTitanCoreAblation.py',
              'paper/scripts/count_aware_tpp_backbone/core.py']
    for rel in relevant:
        assert digest(ROOT/rel)==contract['source']['files'][rel], rel
    result={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'analysis_scope':'train-only descriptive; saved validation-only selected-checkpoint aggregates',
            'contract_canonical_sha256':canonical,'frozen_source_closure_sha256':contract['source']['files_sha256'],
            'inspected_model_sources_match_frozen':True, 'model_sources':{p:digest(ROOT/p) for p in relevant},
            'ledger_path':str(LEDGER.relative_to(ROOT)),'ledger_sha256':digest(LEDGER),
            'validation_snapshot_utc':'2026-09-28T01:55:42Z',
            'quantile_method':'nearest', 'datasets':{}}
    for spec in contract['datasets']:
        result['datasets'][spec['dataset_id']]=profile(spec)
        print('Profile verified:',spec['dataset_id'],flush=True)
    result['validation']=aggregate_validation(json.loads(LEDGER.read_text()))
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Saved',OUT/'analysis.json',flush=True)
    return result


if __name__=='__main__':
    main()
