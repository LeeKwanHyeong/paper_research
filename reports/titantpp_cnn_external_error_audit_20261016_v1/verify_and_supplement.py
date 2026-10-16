"""Second arithmetic implementation; only stored rows and local byte hashes."""
import bisect
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import pyarrow.parquet as pq

O=Path(__file__).resolve().parent
R=O.parents[1]
B=R/'search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3'
P=R/'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1'
result=json.loads((O/'RESULT.json').read_text())
files=dict(result['source_files'])
checks=Counter()
def pin(p,h=None):
    p=Path(p);actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert h is None or h==actual,(str(p),h,actual)
    files[str(p.relative_to(R))]=actual
    return p
def read(p,h=None):return json.loads(pin(p,h).read_text())
def equal(a,b):
    assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-7),(a,b)
    checks['scalar_equalities']+=1
def mean(xs):return math.fsum(xs)/len(xs)
def metric(e,t,w=None):
    if w is None:w=[1.]*len(e)
    den=math.fsum(w)
    return {'rmse':math.sqrt(math.fsum(v*v*z for v,z in zip(e,w))/den),
            'mae':math.fsum(abs(v)*z for v,z in zip(e,w))/den,
            'bias':math.fsum(v*z for v,z in zip(e,w))/den,
            'time_nll':math.fsum(v*z for v,z in zip(t,w))/den}

assert not (O/'VERIFICATION.json').exists()
contract=read(B/'contract_5080.json')
parent=read(P/'contract.json','918d4d5ae5c2440171bb4929d6049c60fe31a500b4a115b5a37dd42059498d39')
cnn=result['models'][0]
for seed in result['seeds']:
    jid=f'raf_spare_parts__{seed}__{cnn}'
    j=next(x for x in parent['jobs'] if x['id']==jid)
    c=next(x for x in contract['jobs'] if x['id']==jid)
    pin(P/j['checkpoint']['relative_path'],j['checkpoint']['sha256'])
    ref=c['reference_receipts']['validation']
    val=read(P/ref['path'],ref['sha256'])
    for split in ['train','validation']:
        rec=read(B/'native/5080/results'/jid/split/'receipt.json')
        assert rec['selected_epoch']==j['selected_epoch']==val['selected_epoch']
        assert rec['checkpoint_file_sha256']==j['checkpoint']['sha256']==val['checkpoint_sha256']
        assert rec['state_sha256_before']==j['checkpoint_state_sha256']==val['state_sha256']
        assert rec['source_closure']==j['frozen_source_closure']==val['source_closure']
        if split=='validation':
            for a,b in [('rmse','qty_rmse'),('mae','qty_mae'),('time_nll','time_nll')]:
                m=next(x for x in result['seed_metrics'] if x['model']==cnn and x['seed']==seed and x['split']==split)
                equal(m['whole'][a],val['metrics'][b]);equal(m['tail'][a],val['metrics']['tail'][b])
    checks['cnn_checkpoint_lineages']+=1
src=parent['sources']['components']
for path,h in src['files'].items():pin(P/src['root']/path,h)
checks['cnn_source_files']=len(src['files'])
ej=next(x for x in contract['jobs'] if x['id']=='raf_spare_parts__42__s2p2_matched_head')
src=ej['source_bundle']
for path,h in src['source_files'].items():pin(B/'external_parent'/src['source_root']/path,h)
checks['external_source_files']=len(src['source_files'])
pin(R/'reports/titantpp_backbone_value_difference_raf3_20261016_v1/diagnostics_paired_regret_v1/PLAN.json','0708d991be70c3792af95e6266005de2c9cea14d6851fc5aaa731351727ed8b3')
pin(O/'RESULT.json');pin(Path(__file__))

frames={s:pq.read_table(pin(O/f'paired_{s}.parquet')).to_pydict() for s in ['train','validation']}
arrays={};calculated={}
for split,f in frames.items():
    y=f['raw_quantity'];assert len(set(f['target_id']))==len(y)==result['split_counts'][split]
    for model in result['models']:
        for seed in result['seeds']:
            pred=f[f'{model}__{seed}__prediction'];t=f[f'{model}__{seed}__time_nll']
            e=[p-q for p,q in zip(pred,y)];assert all(math.isfinite(x) for x in e+t)
            arrays[split,model,seed]=(e,t)
            found=next(x for x in result['seed_metrics'] if (x['split'],x['model'],x['seed'])==(split,model,seed))
            for scope in ['whole','tail']:
                ids=list(range(len(y))) if scope=='whole' else [i for i,q in enumerate(y) if q>200]
                m=metric([e[i] for i in ids],[t[i] for i in ids]);calculated[split,model,seed,scope]=m
                for k,v in m.items():equal(v,found[scope][k])
            checks['cohort_metrics']+=1
for x in result['summary']:
    for scope in ['whole','tail']:
        for k in ['rmse','mae','bias','time_nll']:
            vals=[calculated[x['split'],x['model'],seed,scope][k] for seed in result['seeds']]
            equal(st.mean(vals),x[scope][k]['mean']);equal(st.stdev(vals),x[scope][k]['sample_sd'])
for x in result['pairwise']:
    e,_=arrays[x['split'],cnn,x['seed']];b,_=arrays[x['split'],x['model'],x['seed']]
    a=[abs(u)-abs(v) for u,v in zip(e,b)];s=[u*u-v*v for u,v in zip(e,b)]
    equal(mean(a),x['delta_mae']);equal(mean(s),x['delta_mse'])
    assert sum(v>0 for v in a)==x['external_win_count']
    assert sum(v<0 for v in a)==x['cnn_win_count']
    for name,v in [('ae',a),('se',s)]:
        pos=[max(z,0.) for z in v];neg=[max(-z,0.) for z in v]
        equal(math.fsum(pos),x['gross_external_gain_'+name]);equal(math.fsum(neg),x['gross_cnn_gain_'+name])
        equal(math.fsum(sorted(pos,reverse=True)[:math.ceil(len(pos)*.01)])/math.fsum(pos),x['top_one_percent_positive_'+name+'_share'])
    checks['paired_cohorts']+=1

cuts=result['cutoffs']['history']
def match(f,i,cell):
    if cell=='whole':return True
    fields=cell.split(':');p=f['pattern'][i];mag=f['magnitude'][i]
    if fields[0]=='pattern':return p==fields[1]
    if fields[0]=='pattern_magnitude':
        band='unavailable' if mag is None or not math.isfinite(mag) else 'low' if mag<=cuts['magnitude'][0] else 'middle' if mag<=cuts['magnitude'][1] else 'high'
        return p==fields[1] and band==fields[2]
    if fields[0]=='history_length':
        n=f['history_length'][i];return fields[1]==('1-2' if n<=2 else '3-10' if n<=10 else '>10')
    if fields[0] in ['last_quantity','cv']:
        return int(fields[1][1:])-1==bisect.bisect_left(cuts[fields[0]],f[fields[0]][i])
    delta=f['last_quantity'][i]-f['mean_quantity'][i]
    return fields[1]==('below' if delta<0 else 'above' if delta>0 else 'equal')
masks={}
for cell in sorted(set(x['cell'] for x in result['history_support'])):
    for split,f in frames.items():masks[split,cell]=[i for i in range(len(f['target_id'])) if match(f,i,cell)]
    common=set(frames['train']['entity_id'][i] for i in masks['train',cell])&set(frames['validation']['entity_id'][i] for i in masks['validation',cell])
    for split,f in frames.items():
        ids=masks[split,cell];ci=[i for i in ids if f['entity_id'][i] in common]
        s=next(x for x in result['history_support'] if x['split']==split and x['cell']==cell)
        assert s['rows']==len(ids) and s['common_rows']==len(ci) and s['common_entities']==len(common)
        for weighting,ii in [('row',ids),('common_entity_equal',ci)]:
            counts=Counter(f['entity_id'][i] for i in ii)
            w=[1. if weighting=='row' else 1/counts[f['entity_id'][i]] for i in ii]
            for x in result['history_cells']:
                if (x['split'],x['cell'],x['weighting'])!=(split,cell,weighting):continue
                e,t=arrays[split,cnn,x['seed']];b,u=arrays[split,x['model'],x['seed']]
                aa=metric([e[i] for i in ii],[t[i] for i in ii],w);bb=metric([b[i] for i in ii],[u[i] for i in ii],w)
                for k in aa:equal(aa[k],x['cnn'][k]);equal(bb[k],x['external'][k])
                equal(aa['mae']-bb['mae'],x['delta_ae']);equal(aa['rmse']**2-bb['rmse']**2,x['delta_se'])
                checks['history_cell_cohorts']+=1

# Explicit post-result sensitivity appendix; it does not change the fixed signal rule.
cell='pattern_magnitude:stagnation:middle';sensitivity=[];case_rows=[]
for split,f in frames.items():
    ii=masks[split,cell]
    for seed in result['seeds']:
        e,_=arrays[split,cnn,seed];b,_=arrays[split,'rmtpp',seed]
        a=[abs(e[i])-abs(b[i]) for i in ii];s=[e[i]**2-b[i]**2 for i in ii]
        sensitivity.append(dict(split=split,seed=seed,rows=len(ii),delta_mae=mean(a),delta_mse=mean(s),
            delta_mae_after_removing_single_largest_ae_gain=(math.fsum(a)-max(a))/(len(a)-1),
            delta_mse_after_removing_single_largest_se_gain=(math.fsum(s)-max(s))/(len(s)-1),
            top_one_percent_gross_positive_se_share=math.fsum(sorted([max(x,0.) for x in s],reverse=True)[:math.ceil(len(s)*.01)])/math.fsum(max(x,0.) for x in s),
            global_mae_contribution=math.fsum(a)/len(f['target_id'])))
    # Largest three-seed mean AE gains and losses in this cell, not selectively omitted.
    for category,reverse in [('external_ae_gain',True),('cnn_ae_gain',False)]:
        vals=[]
        for i in ii:
            es=[arrays[split,cnn,z][0][i] for z in result['seeds']];bs=[arrays[split,'rmtpp',z][0][i] for z in result['seeds']]
            vals.append((mean([abs(a)-abs(b) for a,b in zip(es,bs)]),i))
        for value,i in sorted(vals,reverse=reverse)[:3]:
            case_rows.append(dict(split=split,category=category,target_id=f['target_id'][i],entity_id=f['entity_id'][i],seq=f['seq'][i],truth=f['raw_quantity'][i],history_length=f['history_length'][i],last_quantity=f['last_quantity'][i],history_mean=f['mean_quantity'][i],d1=f['d1'][i],d2=f['d2'][i],mean_delta_ae=value,
                cnn_predictions=[f[f'{cnn}__{z}__prediction'][i] for z in result['seeds']],rmtpp_predictions=[f[f'rmtpp__{z}__prediction'][i] for z in result['seeds']]))
for path,h in files.items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==h,path
payload={'status':'PASS_local_second_implementation','checks':dict(checks),'files_verified_unchanged':len(files),
    'new_model_execution':0,'test_payload_read':False,'independent_agent_review':False,
    'sensitivity_status':'post_result_descriptive_appendix_not_new_gate','signal_sensitivity':sensitivity,'signal_examples':case_rows,
    'source_files':files,'limitations':['Second implementation in same agent, not an independent reviewer','Source/checkpoint byte validation does not rerun models','History groups do not jointly match quantity, amplitude and entity distributions']}
with (O/'VERIFICATION.json').open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({k:v for k,v in payload.items() if k in ['status','checks','files_verified_unchanged','signal_sensitivity']},ensure_ascii=False))
