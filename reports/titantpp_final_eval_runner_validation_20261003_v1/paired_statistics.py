"""Paired cluster/calendar-block statistics for validated prediction records.

This version accepts synthetic/validation records only. It cannot unlock test data.
"""
import hashlib
import math
import numpy as np

METRICS=('qty_mae','qty_rmse','time_nll')

def align_panel(records,models,seeds,deterministic_models=()):
    if not records or len(set(models))!=len(models) or len(set(seeds))!=len(seeds):
        raise ValueError('Empty or duplicate model/seed panel')
    if len({r['dataset'] for r in records})!=1:raise ValueError('Do not pool datasets')
    if any(r.get('evaluation_scope') not in ('validation_smoke','validation','synthetic') or
           r.get('split') not in ('validation','synthetic') for r in records):
        raise ValueError('Held-out or unspecified evaluation scope is locked')
    deterministic=set(deterministic_models)
    if not deterministic.issubset(models) or len(deterministic)!=len(deterministic_models):
        raise ValueError('Invalid deterministic model declaration')
    expected={(m,s) for m in models for s in ([None] if m in deterministic else seeds)}
    groups={}
    for r in records:
        key=(r['model'],r['seed'])
        if key not in expected:raise ValueError('Unexpected model or seed')
        g=groups.setdefault(key,{})
        if r['target_id'] in g:raise ValueError('Duplicate target ID')
        g[r['target_id']]=r
    if set(groups)!=expected:raise ValueError('Missing model or seed')
    first=groups[(models[0],None if models[0] in deterministic else seeds[0])];ids=sorted(first)
    losses=np.empty((len(ids),len(models),len(seeds),3),dtype=np.float64)
    for mi,m in enumerate(models):
        for si,s in enumerate(seeds):
            # Repeat a deterministic loss algebraically for paired seed means;
            # it is still one prediction vector, never three sampled replicates.
            g=groups[(m,None if m in deterministic else s)]
            if set(g)!=set(ids):raise ValueError('Target populations differ; no intersection fallback')
            for i,key in enumerate(ids):
                r=g[key];ref=first[key]
                for col in ['dataset','entity_id','seq','split','raw_quantity','recorded_gap','site_id','time_bucket','recorded_event_key','data_file_sha256']:
                    if r.get(col)!=ref.get(col):raise ValueError('Target identity/truth metadata mismatch: '+col)
                q=float(r['raw_quantity']);pred=float(r['predicted_raw_quantity'])
                if m in deterministic:
                    if r['time_nll'] is not None:raise ValueError('Deterministic quantity baseline has no time likelihood')
                    nll=np.nan
                else:nll=float(r['time_nll'])
                if not np.isfinite([q,pred,(pred-q)**2]).all() or (m not in deterministic and not np.isfinite(nll)):
                    raise ValueError('Nonfinite predictions/targets/losses')
                losses[i,mi,si]=[abs(pred-q),(pred-q)**2,nll]
    return [first[k] for k in ids],losses

def sufficient_statistics(metadata,losses,kind,cluster_key='entity_id'):
    if kind=='cluster':
        raw=[r.get(cluster_key) for r in metadata]
        if any(x is None or str(x)=='' for x in raw):raise ValueError('Missing cluster mapping')
        labels=sorted({str(x) for x in raw});lookup={x:i for i,x in enumerate(labels)}
        index=np.array([lookup[str(x)] for x in raw],dtype=np.int64)
    elif kind=='calendar':
        if any(r.get('time_bucket') is None for r in metadata):raise ValueError('Missing Taxi timestamp; no target-order fallback')
        exact_times=np.array([np.datetime64(r['time_bucket'],'ns') for r in metadata])
        times=exact_times.astype('datetime64[h]')
        if np.isnat(times).any():raise ValueError('Invalid timestamp')
        if not np.array_equal(exact_times,times.astype('datetime64[ns]')):raise ValueError('Taxi timestamps must already be aligned to recorded hours')
        # The recorded hour grid is used; absent hours carry zero targets.
        index=(times-times.min()).astype('timedelta64[h]').astype(np.int64)
        labels=list(range(int(index.max())+1))
    else:raise ValueError('Unknown resampling unit')
    counts=np.bincount(index,minlength=len(labels)).astype(np.int64)
    sums=np.zeros((len(labels),)+losses.shape[1:],dtype=np.float64)
    np.add.at(sums,index,losses)
    return labels,counts,sums

def metrics_from_weights(counts,sums,weights):
    n=float(weights@counts)
    if n<=0:raise ValueError('Empty resampled population')
    values=np.tensordot(weights,sums,axes=(0,0))/n
    values[...,1]=np.sqrt(values[...,1])
    return values  # model, seed, metric; mean over seeds only afterwards

def draw_weights(rng,size,kind,length=None):
    if kind=='cluster':index=rng.integers(0,size,size=size)
    else:
        starts=rng.integers(0,size,size=math.ceil(size/length))
        index=((starts[:,None]+np.arange(length)[None,:])%size).ravel()[:size]
    return np.bincount(index,minlength=size)

def paired_bootstrap(records,models,seeds,*,representative='titantpp_history_mlp',kind='cluster',
                     cluster_key='entity_id',draws=10000,rng_seed=20261001,block_hours=168,anchor=None,
                     deterministic_models=()):
    if draws<2:raise ValueError('Need at least two draws')
    if kind=='calendar' and (not isinstance(block_hours,int) or block_hours<1):raise ValueError('Positive integer block length required')
    if len(seeds)<2:raise ValueError('Sample SD requires at least two fixed seeds')
    if representative not in models or representative in deterministic_models:raise ValueError('Representative must be a learned model in the panel')
    if anchor is not None and anchor not in models:raise ValueError('Unknown fixed anchor')
    metadata,losses=align_panel(records,models,seeds,deterministic_models)
    labels,counts,sums=sufficient_statistics(metadata,losses,kind,cluster_key)
    values=metrics_from_weights(counts,sums,np.ones(len(labels),dtype=np.int64))
    means=values.mean(axis=1);sd=values.std(axis=1,ddof=1)
    def metric_dict(array):return {k:float(v) if np.isfinite(v) else None for k,v in zip(METRICS,array)}
    result={'scope':records[0]['evaluation_scope'],'count':len(metadata),'units':len(labels),'kind':kind,'cluster_key':cluster_key if kind=='cluster' else None,
            'seed_metrics':{m:({'deterministic':metric_dict(values[mi,0])} if m in deterministic_models else {str(s):metric_dict(values[mi,si]) for si,s in enumerate(seeds)}) for mi,m in enumerate(models)},
            'means':{m:metric_dict(means[mi]) for mi,m in enumerate(models)},
            'sample_sd':{m:({k:None for k in METRICS} if m in deterministic_models else metric_dict(sd[mi])) for mi,m in enumerate(models)},
            'deterministic_models':list(deterministic_models),'deterministic_prediction_vectors_per_model':1,
            'uncertainty_scope':'conditional_on_fixed_trained_seeds','draws_requested':draws,'rng':'PCG64','rng_seed':rng_seed,'numpy':np.__version__}
    if len(labels)<2 or (kind=='calendar' and len(labels)<2*block_hours):
        return {**result,'interval_status':'not_estimable_insufficient_units_or_span','intervals':None,'draws_completed':0}
    effective=len(labels) if kind=='cluster' else len(labels)//block_hours
    rng=np.random.Generator(np.random.PCG64(rng_seed));rep=models.index(representative)
    diffs=np.full((draws,len(models),3),np.nan);ratios=np.full((draws,len(models)),np.nan)
    invalid=0;digest=hashlib.sha256()
    for b in range(draws):
        weights=draw_weights(rng,len(labels),kind,block_hours)
        digest.update(weights.astype('<i8').tobytes())
        try:avg=metrics_from_weights(counts,sums,weights).mean(axis=1)
        except ValueError:invalid+=1;continue
        diffs[b]=avg[rep]-avg
        np.divide(100*(avg[:,1]-avg[rep,1]),avg[:,1],out=ratios[b],where=avg[:,1]!=0)
    result.update({'draws_completed':draws,'invalid_empty_draws':invalid,'resampling_weights_sha256':digest.hexdigest(),'nominal_nonoverlapping_units':effective,'small_sample_descriptive_only':effective<20})
    if invalid:return {**result,'interval_status':'not_finalized_invalid_draws','intervals':None}
    intervals={}
    for mi,m in enumerate(models):
        if mi==rep:continue
        entry={metric:({'difference':float(means[rep,k]-means[mi,k]),'pointwise_95_percentile':np.quantile(diffs[:,mi,k],[.025,.975],method='linear').tolist()} if not (m in deterministic_models and metric=='time_nll') else None) for k,metric in enumerate(METRICS)}
        valid=np.isfinite(ratios[:,mi]).all() and means[mi,1]!=0
        entry['relative_RMSE_improvement_percent']={'point':float(100*(means[mi,1]-means[rep,1])/means[mi,1]) if means[mi,1]!=0 else None,'pointwise_95_percentile':np.quantile(ratios[:,mi],[.025,.975],method='linear').tolist() if valid else None,'status':'defined' if valid else 'undefined_zero_denominator'}
        if m==anchor:entry['qty_rmse']['four_anchor_bonferroni_98_75_percentile']=np.quantile(diffs[:,mi,1],[.00625,.99375],method='linear').tolist()
        intervals[m]=entry
    return {**result,'interval_status':'computed_descriptive' if effective<20 else 'computed_conditional','intervals':intervals}
