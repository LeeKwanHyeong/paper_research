"""Read-only Train/Validation history cohorts, aligned to preserved feature caches.

No model forward, optimizer, held-out rows, or per-event output is permitted.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import numpy as np
import polars as pl


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def load_context(source_root, dataset):
    root = Path(source_root).resolve()
    sys.path.insert(0, str(root))
    core = importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
    loader = importlib.import_module('paper.scripts.run_taxi_quantity_interface_ablation')
    population = importlib.import_module('paper.scripts.run_count_aware_tpp_backbone_control')
    for module in (core, loader, population):
        require(Path(module.__file__).resolve().is_relative_to(root), 'Current/frozen module mixing')
    identity = dataset['inherited_data_identity']
    data_path = root / identity['data']['path']
    split_path = root / identity['split_manifest']['path']
    require(sha(data_path) == identity['data']['sha256'], 'Dataset SHA changed')
    require(sha(split_path) == identity['split_manifest']['sha256'], 'Split SHA changed')
    frame = (pl.scan_parquet(data_path)
             .filter(pl.col('chronological_split').is_in(['train', 'validation']))
             .collect().sort(['oper_part_no', 'seq']))
    require(set(frame['chronological_split'].unique()) == {'train', 'validation'}, 'Illegal materialized split')
    raw_gaps = {(r['oper_part_no'],r['seq']):r['delta_t'] for r in frame.select('oper_part_no','seq','delta_t').iter_rows(named=True)}
    frame = core.prepare_count_frame(frame)
    contexts = {}
    for split in ('train', 'validation'):
        _, observed = population.exact_target_population(frame, target_split=split,
            lookback_weeks=dataset['loader']['lookback_weeks'], max_seq_len=dataset['loader']['max_seq_len'])
        wanted = identity['populations'][split]
        require(all(observed[k] == wanted[k] for k in wanted), 'Canonical target population changed')
        ds = loader.make_loader(frame, target_split=split, batch_size=128,
            lookback_weeks=dataset['loader']['lookback_weeks'], max_seq_len=dataset['loader']['max_seq_len'],
            shuffle=False, generator=None).dataset
        contexts[split] = context_arrays(ds)
        contexts[split]['raw_dt'] = np.asarray([raw_gaps[(ds.parts[p],ds.seq_lists[p][i+1])] for p,i in ds.index],dtype=np.float64)
        contexts[split]['population_receipt'] = {k: observed[k] for k in wanted}
    return contexts


def context_arrays(ds):
    names = ('dt', 'qty', 'history_count', 'previous_dt', 'previous_qty',
             'history_mean_qty', 'history_max_dt', 'history_fraction_dt_gt1', 'seq', 'series_index')
    columns = {k: [] for k in names}
    sequences = [np.asarray(x, dtype=np.int32) for x in ds.seq_lists]
    gaps = [np.asarray(x, dtype=np.float32) for x in ds.dt_lists]
    quantities = [np.asarray(x, dtype=np.float32) for x in ds.val_lists]
    for p, i in ds.index:
        seq, dt, qty = sequences[p], gaps[p], quantities[p]
        j = max(int(np.searchsorted(seq, int(seq[i]) - (ds.W - 1), side='left')),
                i - (ds.max_len - 2))
        hdt, hqty = dt[j:i+1], qty[j:i+1]
        values = (dt[i+1], qty[i+1], len(hdt), dt[i], qty[i], float(hqty.mean()),
                  float(hdt.max()), float((hdt > 1).mean()), seq[i+1], p)
        for key, value in zip(names, values):
            columns[key].append(value)
    return {k: np.asarray(v) for k, v in columns.items()}


def align_context(context, cache):
    for key in ('dt', 'qty'):
        require(np.array_equal(np.asarray(context[key], dtype=np.float32), cache[key].numpy()),
                'Context/cache target order or value mismatch: ' + key)
    require(len(context['history_count']) == len(cache['dt']), 'Context/cache row count mismatch')


def masks_for(context, boundaries):
    q = np.asarray(boundaries)
    def bands(values, cuts):
        return np.searchsorted(np.asarray(cuts), values, side='left')
    result = {}
    for name, values, cuts, labels in (
        ('history_count', context['history_count'], [4,16,64,128], ['1–4','5–16','17–64','65–128','129+']),
        ('previous_dt', context['previous_dt'], [1,3,7], ['1','2–3','4–7','8+']),
        ('history_max_dt', context['history_max_dt'], [1,3,7], ['1','2–3','4–7','8+']),
        ('raw_target_dt', context['raw_dt'], [0,1,3,7], ['raw0','raw1','raw2–3','raw4–7','raw8+']),
        ('previous_quantity', context['previous_qty'], q, [f'≤{q[0]:g}',*[f'({a:g},{b:g}]' for a,b in zip(q[:-1],q[1:])],f'>{q[-1]:g}']),
        ('history_mean_quantity', context['history_mean_qty'], q, [f'≤{q[0]:g}',*[f'({a:g},{b:g}]' for a,b in zip(q[:-1],q[1:])],f'>{q[-1]:g}']),
        ('target_quantity_descriptive', context['qty'], q, [f'≤{q[0]:g}',*[f'({a:g},{b:g}]' for a,b in zip(q[:-1],q[1:])],f'>{q[-1]:g}']),
    ):
        ids = bands(values, cuts)
        result[name] = [(label, ids == i) for i,label in enumerate(labels)]
    low=context['previous_qty'] <= q[0]
    long=context['dt'] > 1
    result['previous_quantity_by_target_dt_descriptive']=[
        (f'previous≤{q[0]:g}, targetdt1',low & ~long),
        (f'previous≤{q[0]:g}, targetdt≥2',low & long),
        (f'previous>{q[0]:g}, targetdt1',~low & ~long),
        (f'previous>{q[0]:g}, targetdt≥2',~low & long)]
    return result


def decompose(e0, last, cohorts):
    e0, last = np.asarray(e0,dtype=np.float64), np.asarray(last,dtype=np.float64)
    require(e0.shape == last.shape and e0.ndim == 1 and len(e0)>0, 'Bad loss arrays')
    require(np.isfinite(e0).all() and np.isfinite(last).all(), 'Nonfinite losses')
    n = len(e0)
    membership = np.stack([m for _,m in cohorts]).sum(axis=0)
    require(np.all(membership == 1), 'Cohorts overlap or omit rows')
    records = []
    for label, m in cohorts:
        count = int(m.sum())
        records.append({'label':label,'count':count,'population_fraction':count/n,
            'e0_mean_nll':float(e0[m].mean()) if count else None,
            'last_mean_nll':float(last[m].mean()) if count else None,
            'e0_contribution_to_overall_mean':float(e0[m].sum()/n),
            'last_contribution_to_overall_mean':float(last[m].sum()/n),
            'delta_contribution_to_overall_mean':float((last[m]-e0[m]).sum()/n)})
    residuals={key: float(sum(r[key] for r in records)-total) for key,total in (
        ('e0_contribution_to_overall_mean',float(e0.mean())),
        ('last_contribution_to_overall_mean',float(last.mean())),
        ('delta_contribution_to_overall_mean',float((last-e0).mean())))}
    require(max(abs(r) for r in residuals.values()) < 1e-10, 'Contribution reconciliation failed')
    return {'rows':records,'reconciliation_residuals':residuals}


def symmetric_mix(train_rows, val_rows):
    require([r['label'] for r in train_rows] == [r['label'] for r in val_rows], 'Misaligned strata')
    mix = within = unmatched = 0.
    for t,v in zip(train_rows,val_rows):
        pt,pv=t['population_fraction'],v['population_fraction']
        lt,lv=t['e0_mean_nll'],v['e0_mean_nll']
        if lt is None or lv is None:
            unmatched += v['e0_contribution_to_overall_mean']-t['e0_contribution_to_overall_mean']
        else:
            mix += (pv-pt)*(lv+lt)/2
            within += (pv+pt)*(lv-lt)/2
    return {'mix_component':mix,'within_stratum_component':within,'unmatched_component':unmatched,
            'sum':mix+within+unmatched,'causal_estimate':False,
            'method':'Symmetric two-population accounting decomposition, original E0 losses'}


def quantiles(values):
    values=np.asarray(values,dtype=np.float64)
    return {str(p):float(np.quantile(values,p)) for p in (0,.01,.1,.5,.9,.99,1)}


def context_analysis(contexts, inputs, losses, boundaries):
    output={'splits':{},'train_validation_e0_gap_decomposition':{},
            'interpretation':'Separate marginal cuts overlap; do not add contributions across different dimensions.'}
    for split,context in contexts.items():
        align_context(context, inputs['caches'][split])
        e0=losses[split]['E0'];last=losses[split]['last']
        masks=masks_for(context,boundaries)
        output['splits'][split]={'count':len(e0),
            'observed_features':{k:quantiles(context[k]) for k in ('dt','history_count','previous_dt','previous_qty','history_mean_qty','history_max_dt','history_fraction_dt_gt1')},
            'aligned_target_values':True,'population_receipt':context['population_receipt'],
            'cohorts':{k:decompose(e0,last,v) for k,v in masks.items()}}
    for dim in output['splits']['train']['cohorts']:
        output['train_validation_e0_gap_decomposition'][dim]=symmetric_mix(
            output['splits']['train']['cohorts'][dim]['rows'],output['splits']['validation']['cohorts'][dim]['rows'])
    return output
