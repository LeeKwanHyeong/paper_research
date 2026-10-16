"""Read-only analysis of stored RAF predictions. No ML model/remote imports."""
from pathlib import Path
import hashlib
import json
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
B = ROOT / 'search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3'
N = B / 'native/5080'
CNN = 'titantpp_cnn_gru54_cnn_only'
MODELS = [CNN, 'rmtpp', 's2p2_matched_head', 'nhp', 'thp', 'sahp', 'attnhp_matched_head']
PRIMARY = ['s2p2_matched_head', 'rmtpp']
SEEDS = [42,52,62]
SPLITS = ['train','validation']
FILES = {}

def pin(p, expected=None):
    p=Path(p).resolve()
    assert '/test/' not in str(p) and not p.name.startswith('THREE_SEED_TEST'), p
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    assert expected is None or h==expected, (str(p),h,expected)
    FILES[str(p.relative_to(ROOT))]=h
    return p

def read(p, expected=None):
    return json.loads(pin(p,expected).read_text())

def write(name, data):
    with (OUT/name).open('x') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def stats(x):
    x=np.asarray(x,dtype=float)
    return {'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None}

def metrics(e,t,w=None):
    if w is None:w=np.ones(len(e))
    w=np.asarray(w,dtype=float);w=w/w.sum()
    return dict(rmse=float(np.sqrt(np.dot(w,e*e))),mae=float(np.dot(w,abs(e))),bias=float(np.dot(w,e)),time_nll=float(np.dot(w,t)))

def features(raw, split, original):
    f=raw[raw.chronological_split.isin(['train'] if split=='train' else SPLITS)].sort_values(['oper_part_no','seq'])
    output=[];parts=[];pis=[];pos=[];seqs=[];qs=[]
    for p,(entity,g) in enumerate(f.groupby('oper_part_no',sort=False)):
        entity=str(entity);parts.append(entity)
        s=g.seq.to_numpy();q=g.demand_qty.to_numpy().astype(np.float32).astype(float)
        labels=g.chronological_split.to_numpy();gap=g.delta_t.to_numpy();date=g.demand_dt.to_numpy()
        assert np.all(np.diff(s)>0) and np.array_equal(np.diff(s),gap[1:])
        for i in range(1,len(g)):
            if labels[i]!=split:continue
            start=max(int(np.searchsorted(s,s[i-1]-83,side='left')),i-83)
            h=q[start:i];z=np.log1p(h);h_len=len(h)
            d1=float(z[-1]-z[-2]) if h_len>=2 else None
            d2=float(z[-2]-z[-3]) if h_len>=3 else None
            pattern='insufficient_history' if h_len<3 else 'stagnation' if d1==0 or d2==0 else 'same_sign' if d1*d2>0 else 'reversal'
            output.append(dict(entity_id=entity,seq=int(s[i]),raw_quantity=float(q[i]),history_length=h_len,last_quantity=float(h[-1]),mean_quantity=float(h.mean()),history_quantity_std=float(h.std()),history_quantity_max=float(h.max()),d1=d1,d2=d2,pattern=pattern,magnitude=max(abs(d1),abs(d2)) if h_len>=3 else None,cv=float(h.std()/h.mean()) if h.mean()>0 else 0.,history_first_date=str(date[start]),history_last_date=str(date[i-1])))
            pis.append(p);pos.append(i);seqs.append(int(s[i]));qs.append(float(q[i]))
    digest=hashlib.sha256(b'hard_lmm_target_identity_v1\0'+split.encode()+b'\0')
    digest.update(json.dumps(parts,ensure_ascii=False,separators=(',',':')).encode())
    for a in [pis,pos,seqs]:digest.update(np.asarray(a,dtype='<i8').tobytes())
    assert digest.hexdigest()==original['target_identity_sha256']
    assert hashlib.sha256(b'hard_lmm_target_quantity_v1\0'+np.asarray(qs,dtype='<f8').tobytes()).hexdigest()==original['target_quantity_sha256']
    return pd.DataFrame(output)

def cell_masks(f,cuts):
    cells={'whole':np.ones(len(f),dtype=bool)}
    for p in ['insufficient_history','stagnation','same_sign','reversal']:
        m=(f.pattern==p).to_numpy();cells['pattern:'+p]=m
        if p=='insufficient_history':cells['pattern_magnitude:'+p+':unavailable']=m
        else:
            for b,z in [('low',f.magnitude<=cuts['magnitude'][0]),('middle',(f.magnitude>cuts['magnitude'][0])&(f.magnitude<=cuts['magnitude'][1])),('high',f.magnitude>cuts['magnitude'][1])]:cells['pattern_magnitude:'+p+':'+b]=m&z.to_numpy()
    for name,m in [('1-2',f.history_length<=2),('3-10',(f.history_length>=3)&(f.history_length<=10)),('>10',f.history_length>10)]:cells['history_length:'+name]=m.to_numpy()
    for field in ['last_quantity','cv']:
        code=np.searchsorted(cuts[field],f[field].to_numpy(),side='left')
        for i in range(4):cells[field+':Q'+str(i+1)]=code==i
    dev=f.last_quantity-f.mean_quantity
    for name,m in [('below',dev<0),('equal',dev==0),('above',dev>0)]:cells['last_vs_mean:'+name]=m.to_numpy()
    return cells

def main():
    assert not (OUT/'RESULT.json').exists(),'Existing result: do not overwrite'
    plan=read(OUT/'PLAN.json');pin(Path(__file__))
    execution=read(B/'execution_receipt.json');assert execution['status']=='complete'
    bindings=read(B/'bindings.json',execution['bindings']['sha256'])
    contract=read(B/'contract_5080.json',next(x['sha256'] for x in execution['contracts'] if x['host']=='5080'))
    manifest=read(N/'retrieval_manifest.json')
    jobs={x['id']:x for x in contract['jobs']};evidence={};frames={};receipts={};checks=[]
    for model in MODELS:
        for seed in SEEDS:
            job=f'raf_spare_parts__{seed}__{model}';j=jobs[job]
            for split in SPLITS:
                ref=next(x for x in execution['jobs'] if x['job_id']==job and x['split']==split)
                rec=read(B/ref['receipt']['path'],ref['receipt']['sha256'])
                ex=read(B/ref['worker_exit']['path'],ref['worker_exit']['sha256'])
                assert ex['exit_code']==0 and ex['actual_child_exit_observed'] and ex['split']==split
                assert rec['status']=='complete' and rec['split']==split and rec['seed']==seed and rec['arm']==model
                assert rec['state_sha256_before']==rec['state_sha256_after']
                runtime=rec['runtime'];assert runtime==rec['runtime_after']
                assert runtime['gpu']['uuid']=='GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6' and runtime['cudnn']==91900
                if model!=CNN:
                    lr=j['legacy_registry_row']
                    assert rec['selected_epoch']==lr['selected_epoch'] and rec['checkpoint_file_sha256']==lr['checkpoint_file_sha256'] and rec['state_sha256_before']==lr['state_tensor_sha256'] and rec['source_closure']==lr['source_closure_sha256']
                    # Hash frozen checkpoint bytes; never deserialize/import a model.
                    pin(B/'external_parent'/lr['checkpoint_path'],lr['checkpoint_file_sha256'])
                else: assert rec['selected_epoch']==dict(zip(SEEDS,[9,5,8]))[seed]
                d=N/'results'/job/split
                pm=read(d/'predictions_manifest.json',rec['predictions_manifest_sha256'])
                rows=[]
                for part in pm['parts']:
                    path=pin(d/'predictions'/part['path'],part['sha256'])
                    assert manifest[str(path.relative_to(N))]==part['sha256']
                    t=pq.read_table(path).to_pandas();assert len(t)==part['rows'];rows.append(t)
                f=pd.concat(rows,ignore_index=True)
                assert len(f)==pm['prediction_rows']==rec['prediction_rows']==plan['expected_counts'][split]
                assert not f.target_id.duplicated().any() and set(f.split)=={split} and set(f.seed)=={seed} and set(f.arm)=={model}
                assert set(f.checkpoint_state_sha256)=={rec['state_sha256_before']} and set(f.selected_epoch)=={rec['selected_epoch']}
                err=f.predicted_raw_quantity.to_numpy()-f.raw_quantity.to_numpy();assert np.isfinite(err).all() and np.isfinite(f.time_nll).all()
                np.testing.assert_allclose(err,f.raw_error,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(abs(err),f.absolute_error,rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(err**2,f.squared_error,rtol=1e-10,atol=1e-10)
                for scope,mask in [('whole',np.ones(len(f),bool)),('tail',f.raw_quantity.to_numpy()>200)]:
                    m=metrics(err[mask],f.time_nll.to_numpy()[mask]);original=rec['metrics'] if scope=='whole' else rec['metrics']['tail']
                    for key,source in [('rmse','qty_rmse'),('mae','qty_mae'),('bias','qty_bias'),('time_nll','time_nll')]:assert math.isclose(m[key],original[source],rel_tol=1e-9,abs_tol=1e-8),(job,split,key)
                frames[(model,seed,split)]=f;receipts[(model,seed,split)]=rec
                evidence[f'{job}/{split}']={k:rec[k] for k in ['selected_epoch','checkpoint_file_sha256','source_closure','data_sha256','population']}
                checks.append(dict(job=job,split=split,rows=len(f),receipt_metrics_reproduced=True,actual_worker_exit0=True))
    identity=['target_id','entity_id','seq','split','recorded_date','raw_quantity','recorded_gap','history_length','last_quantity','mean_quantity','target_index','history_first_date','history_last_date','history_quantity_std','history_quantity_max']
    for split in SPLITS:
        base=frames[CNN,42,split]
        for model in MODELS:
            for seed in SEEDS:
                pd.testing.assert_frame_equal(base[identity],frames[model,seed,split][identity],check_exact=True)
                assert receipts[CNN,42,split]['population']==receipts[model,seed,split]['population']
    rawpath=pin(ROOT/'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/data/raf_spare_parts.parquet','d805cad0242078fedfbb87eed18bde9b73d37bfc9b93c6b75aac49087244efa2')
    raw=pq.read_table(rawpath).to_pandas();assert set(raw.chronological_split)==set(SPLITS)
    fs={}
    for split in SPLITS:
        f=features(raw,split,receipts[CNN,42,split]['original_population_identity']);base=frames[CNN,42,split]
        for c in ['entity_id','seq','history_first_date','history_last_date']:assert np.array_equal(f[c],base[c]),c
        for c in ['raw_quantity','history_length','last_quantity','mean_quantity','history_quantity_std','history_quantity_max']:np.testing.assert_allclose(f[c],base[c],rtol=1e-6,atol=1e-6,err_msg=c)
        f['target_id']=base.target_id;fs[split]=f
    cuts={c:[float(x) for x in np.quantile(fs['train'][c].dropna(),q,method='linear')] for c,q in [('magnitude',[.25,.75]),('last_quantity',[.25,.5,.75]),('cv',[.25,.5,.75])]}
    cutoff=float(np.quantile(np.concatenate([frames[CNN,s,'train'].absolute_error.to_numpy() for s in SEEDS]),.9,method='linear'))
    masks={s:cell_masks(fs[s],cuts) for s in SPLITS}
    grouped=[];pairs=[];cells=[];partitions=[];stable=[];supports=[];examples=[]
    for split in SPLITS:
        base=frames[CNN,42,split];y=base.raw_quantity.to_numpy();ent=base.entity_id.to_numpy()
        for model in MODELS:
            for seed in SEEDS:
                f=frames[model,seed,split];e=f.raw_error.to_numpy();t=f.time_nll.to_numpy()
                grouped.append(dict(model=model,seed=seed,split=split,whole=metrics(e,t),tail=metrics(e[y>200],t[y>200])))
        for model in MODELS[1:]:
            ds=[];ads=[]
            for seed in SEEDS:
                a=frames[CNN,seed,split];b=frames[model,seed,split];ae=a.absolute_error.to_numpy();be=b.absolute_error.to_numpy();da=ae-be;dsq=ae**2-be**2
                assert np.array_equal(np.sign(da),np.sign(dsq))
                ds.append(dsq);ads.append(da)
                top=max(1,int(math.ceil(.01*len(a))));p=np.maximum(dsq,0);q=np.maximum(da,0)
                pairs.append(dict(model=model,seed=seed,split=split,n=len(a),external_win_count=int((da>0).sum()),cnn_win_count=int((da<0).sum()),tie_count=int((da==0).sum()),delta_mae=float(da.mean()),delta_mse=float(dsq.mean()),delta_rmse=float(np.sqrt(np.mean(ae**2))-np.sqrt(np.mean(be**2))),gross_external_gain_ae=float(q.sum()),gross_cnn_gain_ae=float(np.maximum(-da,0).sum()),gross_external_gain_se=float(p.sum()),gross_cnn_gain_se=float(np.maximum(-dsq,0).sum()),top_one_percent_positive_se_share=float(np.sort(p)[-top:].sum()/p.sum()) if p.sum() else None,top_one_percent_positive_ae_share=float(np.sort(q)[-top:].sum()/q.sum()) if q.sum() else None,prediction_correlation=float(np.corrcoef(a.predicted_raw_quantity,b.predicted_raw_quantity)[0,1]),signed_error_correlation=float(np.corrcoef(a.raw_error,b.raw_error)[0,1])))
                partitions_here={}
                qcode=np.searchsorted([2,30,60,200],y,side='left')
                for i in range(5):partitions_here['target_quantity:'+str(i)]=qcode==i
                ah=ae>cutoff;bh=be>cutoff
                for label,m in [('both_high',ah&bh),('cnn_only_high',ah&~bh),('external_only_high',~ah&bh),('neither_high',~ah&~bh)]:partitions_here['high_error:'+label]=m
                for label,m in [('both_under',(a.raw_error<0)&(b.raw_error<0)),('both_over',(a.raw_error>0)&(b.raw_error>0)),('other',~(((a.raw_error<0)&(b.raw_error<0))|((a.raw_error>0)&(b.raw_error>0))))]:partitions_here['direction:'+label]=m.to_numpy() if hasattr(m,'to_numpy') else m
                for label,m in partitions_here.items():
                    partitions.append(dict(split=split,model=model,seed=seed,cell=label,rows=int(m.sum()),delta_ae_sum=float(da[m].sum()),delta_se_sum=float(dsq[m].sum()),cnn_ae_sum=float(ae[m].sum()),cnn_se_sum=float((ae[m]**2).sum()),external_ae_sum=float(be[m].sum()),external_se_sum=float((be[m]**2).sum())))
                for family in ['target_quantity:','high_error:','direction:']:
                    ms=[m for k,m in partitions_here.items() if k.startswith(family)];assert np.all(np.sum(ms,axis=0)==1)
                    assert math.isclose(sum(dsq[m].sum() for m in ms),dsq.sum(),abs_tol=1e-6)
            da=np.asarray(ads);dse=np.asarray(ds);allwin=np.all(da>0,axis=0);allloss=np.all(da<0,axis=0)
            stable.append(dict(split=split,model=model,all3_external_better=int(allwin.sum()),all3_cnn_better=int(allloss.sum()),mixed_or_tied=int((~allwin&~allloss).sum()),rows=len(y),external_stable_delta_ae=float(da[:,allwin].mean(axis=0).sum()),external_stable_delta_se=float(dse[:,allwin].mean(axis=0).sum())))
            if model in PRIMARY:
                for label,score in [('external_stable_se',np.where(allwin,dse.mean(axis=0),-np.inf)),('cnn_stable_se',np.where(allloss,-dse.mean(axis=0),-np.inf))]:
                    inds=np.argsort(-score)[:10]
                    for ix in inds:
                        if not math.isfinite(score[ix]):continue
                        row=base.iloc[ix];fr=fs[split].iloc[ix]
                        examples.append(dict(split=split,model=model,category=label,target_id=row.target_id,entity_id=row.entity_id,seq=int(row.seq),truth=float(y[ix]),cnn_predictions=[float(frames[CNN,s,split].predicted_raw_quantity.iloc[ix]) for s in SEEDS],external_predictions=[float(frames[model,s,split].predicted_raw_quantity.iloc[ix]) for s in SEEDS],mean_delta_ae=float(da[:,ix].mean()),mean_delta_se=float(dse[:,ix].mean()),history_length=int(fr.history_length),last_quantity=float(fr.last_quantity),history_mean=float(fr.mean_quantity),pattern=fr.pattern,magnitude=None if pd.isna(fr.magnitude) else float(fr.magnitude)))
    for cell in masks['train']:
        common=set(fs['train'].loc[masks['train'][cell],'entity_id'])&set(fs['validation'].loc[masks['validation'][cell],'entity_id'])
        for split in SPLITS:
            mask=masks[split][cell];f=fs[split];cm=mask&f.entity_id.isin(common).to_numpy()
            supports.append(dict(cell=cell,split=split,rows=int(mask.sum()),entities=int(f.loc[mask,'entity_id'].nunique()),common_entities=len(common),common_rows=int(cm.sum()),excluded_rows=int(mask.sum()-cm.sum())))
            for weighting,m in [('row',mask),('common_entity_equal',cm)]:
                if not m.any():continue
                w=np.ones(m.sum()) if weighting=='row' else 1/f.loc[m,'entity_id'].map(f.loc[m,'entity_id'].value_counts()).to_numpy(dtype=float)
                w=w/w.sum()
                for model in PRIMARY:
                    for seed in SEEDS:
                        a=frames[CNN,seed,split];b=frames[model,seed,split];ea=a.raw_error.to_numpy()[m];eb=b.raw_error.to_numpy()[m]
                        cells.append(dict(cell=cell,split=split,weighting=weighting,model=model,seed=seed,rows=int(m.sum()),entities=int(f.loc[m,'entity_id'].nunique()),cnn=metrics(ea,a.time_nll.to_numpy()[m],w),external=metrics(eb,b.time_nll.to_numpy()[m],w),delta_ae=float(w@(abs(ea)-abs(eb))),delta_se=float(w@(ea**2-eb**2)),delta_signed_error=float(w@(ea-eb))))
    signals=[]
    for cell in masks['train']:
        support=[s for s in supports if s['cell']==cell]
        enough=all(s['rows']>=100 and s['common_rows']>=100 and s['common_entities']>=30 for s in support)
        for model in PRIMARY:
            selected=[x for x in cells if x['cell']==cell and x['model']==model]
            signals.append(dict(cell=cell,model=model,support=enough,positive_ae_cells=sum(x['delta_ae']>0 for x in selected),positive_se_cells=sum(x['delta_se']>0 for x in selected),required_cells=12,descriptive_signal=enough and len(selected)==12 and all(x['delta_ae']>0 and x['delta_se']>0 for x in selected)))
    # Independent aggregate summaries, not pooled predictions across seeds.
    summaries=[]
    for split in SPLITS:
        for model in MODELS:
            g=[x for x in grouped if x['split']==split and x['model']==model]
            summaries.append(dict(split=split,model=model,whole={m:stats([x['whole'][m] for x in g]) for m in ['rmse','mae','bias','time_nll']},tail={m:stats([x['tail'][m] for x in g]) for m in ['rmse','mae','bias','time_nll']}))
    # Serialized flat paired rows for independently reproducible checks; no prediction generation.
    for split in SPLITS:
        f=fs[split].copy()
        for model in MODELS:
            for seed in SEEDS:
                a=frames[model,seed,split]
                f[f'{model}__{seed}__prediction']=a.predicted_raw_quantity.to_numpy()
                f[f'{model}__{seed}__time_nll']=a.time_nll.to_numpy()
        f.to_parquet(OUT/f'paired_{split}.parquet',index=False)
    for rel,h in FILES.items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
    result=dict(status='complete_local_stored_prediction_analysis',created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),scope=plan['scope'],plan_sha256=FILES[str((OUT/'PLAN.json').relative_to(ROOT))],cutoffs=dict(history=cuts,high_error_cnn_train_AE90=cutoff),models=MODELS,seeds=SEEDS,split_counts=plan['expected_counts'],cohort_checks=checks,cohort_provenance=evidence,summary=summaries,seed_metrics=grouped,pairwise=pairs,partitions=partitions,stable_winners=stable,history_support=supports,history_cells=cells,descriptive_signals=signals,examples=examples,source_files=FILES,new_training=0,new_inference=0,remote_calls=0,test_payload_read=False,originals_unchanged=True)
    write('RESULT.json',result)
    write('INPUT_MANIFEST.json',FILES)
    print(json.dumps({'status':result['status'],'cohorts':len(checks),'files':len(FILES),'signals':[x for x in signals if x['descriptive_signal']]},ensure_ascii=False))

if __name__=='__main__':main()
