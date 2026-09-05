from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, platform, sys, re
import numpy as np
import torch

ROOT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
RES=ROOT/'paper/results/hard_lmm_information_access_20260905'
RAW=ROOT/'search_artifacts/hard_lmm_information_access_20260905'
CON=ROOT/'paper/contracts/hard_lmm_information_access_v1.json'
checks=0; hash_count=0; max_error={}; cells={}; findings=[]

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def check(v,msg):
    global checks
    checks+=1
    if not v: raise AssertionError(msg)
def equal(a,b,label):
    global checks
    checks+=1
    if isinstance(a,dict):
        if set(a)!=set(b): raise AssertionError(f'{label}: keys {set(a)^set(b)}')
        for k in a: equal(a[k],b[k],f'{label}.{k}')
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError(f'{label}: list length')
        for i,(x,y) in enumerate(zip(a,b)): equal(x,y,f'{label}.{i}')
    elif a is None or isinstance(a,(str,bool)):
        if a!=b: raise AssertionError(f'{label}: {a!r} != {b!r}')
    else:
        a=np.asarray(a); b=np.asarray(b)
        if a.shape!=b.shape: raise AssertionError(f'{label}: shape')
        if not np.allclose(a,b,rtol=1e-9,atol=1e-10,equal_nan=False):
            raise AssertionError(f'{label}: maxdiff {np.max(np.abs(a-b))}')
        if a.size:
            group='bootstrap' if 'bootstrap' in label else 'fit_metadata' if '.fit.' in label else 'metrics'
            max_error[group]=max(max_error.get(group,0.),float(np.max(np.abs(a-b))))
def hashes(d):
    global hash_count
    for p,h in d.items():
        check(sha(ROOT/p)==h,'hash '+p); hash_count+=1

def ar(x): return x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x,dtype=np.float64)
def mse(v,fold): return {'pooled':float(v.mean()),'folds':{str(f):float(v[fold==f].mean()) for f in (0,1)}}
def raw(logp,q,fold,b,t):
    pred=np.expm1(np.minimum(20.,np.maximum(0.,logp)))
    ae=np.abs(pred-q); se=(pred-q)**2
    def view(sel):
        body=sel&(q<=b); tail=sel&(q>t)
        return dict(n=int(sel.sum()),body_n=int(body.sum()),tail_n=int(tail.sum()),body_mae=float(ae[body].mean()) if body.any() else None,rmse=float(np.sqrt(se[sel].mean())),tail_mae=float(ae[tail].mean()) if tail.any() else None)
    return dict(pooled=view(np.ones(len(q),bool)),folds={str(f):view(fold==f) for f in (0,1)},clamps=dict(below_zero=int((logp<0).sum()),above_twenty=int((logp>20).sum())))

def boot(v,series,fold,repeats,seed,quant):
    rng=np.random.default_rng(seed); groups=[]
    for f in (0,1):
        rows=np.flatnonzero(fold==f); ids,inv=np.unique(series[rows],return_inverse=True)
        sums=np.zeros((len(ids),v.shape[1])); counts=np.zeros(len(ids),dtype=np.int64)
        np.add.at(sums,inv,v[rows]); np.add.at(counts,inv,1)
        groups.append((sums,counts))
    draws=np.zeros((repeats,v.shape[1]))
    for start in range(0,repeats,32):
        stop=min(start+32,repeats); total=np.zeros((stop-start,v.shape[1])); denominator=np.zeros(stop-start)
        for sums,counts in groups:
            selected=rng.integers(len(sums),size=(stop-start,len(sums)))
            total+=np.einsum('bsc->bc',sums[selected]); denominator+=counts[selected].sum(1)
        draws[start:stop]=total/denominator[:,None]
    return [{'lower_p05':float(np.quantile(draws[:,i],.05)), 'lower_family_quantile':float(np.quantile(draws[:,i],quant)), 'median':float(np.median(draws[:,i]))} for i in range(v.shape[1])]

import csv
import xml.etree.ElementTree as ET

def audit_documents():
    document_checks=0; csv_rows={}
    def doccheck(v,msg):
        nonlocal document_checks
        document_checks+=1
        if not v: raise AssertionError('document: '+msg)
    def val(actual,expected,name):
        if expected is None: doccheck(actual=='',name)
        elif isinstance(expected,bool): doccheck(actual==str(expected),name)
        elif isinstance(expected,str): doccheck(actual==expected,name)
        else: doccheck(np.isclose(float(actual),expected,atol=1e-10,rtol=1e-9),name)
    features=c['history_features']['names']
    for fn in ('decoder_metrics.csv','feature_reconstruction.csv','contrast_checks.csv','history_usefulness.csv'):
        with (RES/fn).open() as handle: records=list(csv.DictReader(handle))
        csv_rows[fn]=len(records)
        for rr in records:
            ds,model=rr['dataset'],rr['model']; a=reported[ds][model]
            if fn=='decoder_metrics.csv':
                fam,pack=rr['family'],rr['pack']; z=a['decoders'][fam][pack]; baseline=a['decoders'][fam]['constant']['residual_mse']['pooled']
                expected=dict(input_dimensions=z['input_dimensions'],residual_mse=z['residual_mse']['pooled'],gain_vs_constant_pct=100*(1-z['residual_mse']['pooled']/baseline),fold0_mse=z['residual_mse']['folds']['0'],fold1_mse=z['residual_mse']['folds']['1'],**z['raw_metrics']['pooled'],**z['raw_metrics']['clamps'])
            elif fn=='feature_reconstruction.csv':
                z=a['reconstruction'][rr['stage']]; i=features.index(rr['feature']); expected=dict(oof_r2=z['per_feature_oof_r2'][i],mse=z['per_feature_mse'][i],constant_mse=z['per_feature_constant_mse'][i])
            elif fn=='history_usefulness.csv':
                z=a['history_usefulness'][rr['family']]; expected=dict(gain_vs_constant_pct=100*z['relative_gain'],fold0_mse_improvement=z['fold_mean_improvements']['0'],fold1_mse_improvement=z['fold_mean_improvements']['1'],family_passes=z['passes'],both_families_pass=a['history_usefulness_passes'])
            else:
                contrast=rr['contrast']; z=a['contrasts'][contrast]['families'][rr['family']]['comparisons'][rr['reference']]
                expected=dict(candidate=a['contrasts'][contrast]['candidate'],residual_mse_gain_pct=100*z['relative_gain'],candidate_mse=z['candidate_mse'],reference_mse=z['reference_mse'],fold0_mse_improvement=z['fold_mean_improvements']['0'],fold1_mse_improvement=z['fold_mean_improvements']['1'],bootstrap_lower_family_quantile=z['bootstrap']['lower_family_quantile'],**z['conditions'],comparison_passes=z['passes'],eligible_cell_passes=decision['contrasts'][contrast]['cells'][f'{ds}/{model}'],common_passes=decision['contrasts'][contrast]['passes'])
            for k,v in expected.items(): val(rr[k],v,fn+'.'+k)
    doccheck(csv_rows=={'decoder_metrics.csv':144,'feature_reconstruction.csv':480,'contrast_checks.csv':64,'history_usefulness.csv':8},'CSV expected row counts')
    md=(RES/'README.md').read_text(); blocks=[]; current=[]
    for line in md.splitlines()+['']:
        if line.startswith('|'): current.append([x.strip() for x in line.split('|')[1:-1]])
        elif current: blocks.append(current);current=[]
    doccheck(len(blocks)==5,'five README tables')
    # Input table counts and thresholds: all values fixed in saved reports/manifests.
    scope=blocks[0][2:]
    for col,ds in enumerate(c['datasets'],1):
        a=reported[ds]['original']; m=manifest['datasets'][ds]
        expect=[f"{a['n_targets']:,}",f"{m['series_count']:,}",f"{a['folds']['0']['rows']:,} / {a['folds']['1']['rows']:,}",f"{a['body_threshold']:,.0f}",f"{a['tail_threshold']:,.0f}",f"{a['base_raw_metrics']['pooled']['body_n']:,} / {a['base_raw_metrics']['pooled']['tail_n']:,}"]
        doccheck([row[col] for row in scope]==expect,'README cohort '+ds)
    rfeatures=['mean_log_quantity','last_log_quantity','last_minus_mean_log_quantity','normalized_rank_log_quantity_slope','log_internal_span','age_distortion']; stages=['layer1_last','layer2_mean','h','fused']
    for row,feature in zip(blocks[1][2:],rfeatures):
        i=features.index(feature)
        for j,stage in enumerate(stages,1):
            ex=reported['insta_market_basket']['separate_key']['reconstruction'][stage]['per_feature_oof_r2'][i]
            doccheck(row[j]==f'{ex:.4f}','README reconstruction '+feature+'/'+stage)
    combinations=[('yellow_trip_hourly','original'),('yellow_trip_hourly','separate_key'),('insta_market_basket','original'),('insta_market_basket','separate_key')]
    contrasts_order=['F','history_at_h','history_at_fused','fusion_accessibility','head_accessibility']
    for row,name in zip(blocks[2][2:],contrasts_order):
        for j,(ds,model) in enumerate(combinations,1):
            nums=[float(x.strip().replace('−','-')) for x in row[j].split('/')]
            a=reported[ds][model]
            for idx,fam in enumerate(('linear','random128')):
                ex=a['history_usefulness'][fam]['relative_gain'] if name=='F' else a['contrasts'][name]['families'][fam]['comparisons'][c['primary_contrasts'][name]['references'][0]]['relative_gain']
                doccheck(np.isclose(nums[idx],round(100*ex,3),atol=1e-12,rtol=0),'README MSE gain '+name)
    for row,name in zip(blocks[3][2:],c['primary_contrasts']):
        for j,(ds,model) in enumerate(combinations,1):
            ex='통과' if decision['contrasts'][name]['cells'][f'{ds}/{model}'] else '불충족'; doccheck(row[j]==ex,'README eligible gate '+name)
        doccheck(row[5]==('있음' if decision['contrasts'][name]['passes'] else '없음'),'README common gate')
    for row,(ds,model) in zip(blocks[4][2:],combinations):
        for j,k in enumerate(('body_mae','rmse','tail_mae'),1): doccheck(row[j]==f"{reported[ds][model]['base_raw_metrics']['pooled'][k]:.6f}",'README baseline '+k)
    taxi=reported['yellow_trip_hourly']['original']['decoders']
    for v in (taxi['linear']['constant']['raw_metrics']['pooled']['body_mae'],taxi['linear']['fused']['raw_metrics']['pooled']['body_mae'],taxi['random128']['fused']['raw_metrics']['pooled']['body_mae']): doccheck(f'{v:.6f}' in md,'README Taxi body prose')
    rv=reported['insta_market_basket']['separate_key']['reconstruction']['r']['per_feature_oof_r2'][features.index('mean_log_quantity')];doccheck(f'{rv:.4f}' in md,'README memory reconstruction prose')
    for model in c['models']:
        a=reported['insta_market_basket'][model]
        for fam in ('linear','random128'):
            constant=a['decoders'][fam]['constant']['residual_mse']['pooled']
            for s in ('input_last','input_mean','layer1_last','layer1_mean','layer2_last','layer2_mean','h','fused'):
                doccheck(a['decoders'][fam][s]['residual_mse']['pooled']>constant,'README Instacart no stage beats constant')
    for ds in c['datasets']:
        for model in c['models']: doccheck(all(v==0 for v in manifest['datasets'][ds]['models'][model]['cache_parity_max_abs'].values()),'README exact prior parity')
    tests=ET.parse(RES/'pytest.xml').getroot(); suites=[tests] if tests.tag=='testsuite' else list(tests.iter('testsuite')); totals={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    doccheck(totals==dict(tests=49,failures=0,errors=0,skipped=0),'README 49 tests')
    doccheck('49 passed' in (RES/'pytest.log').read_text(),'pytest log')
    doccheck('회고적 train 내부 진단' in md and '정식 유의성 검정이 아니다' in md and 'decoder를 다시 적합하거나 원본 parquet에서 이력을 재추출하지는 않았다' in md,'README interpretation caveats')
    links=re.findall(r'\]\(([^)]+)\)',md); local_links=[]
    for link in links:
        if link.startswith(('http://','https://')): continue
        p=(Path(link) if link.startswith('/') else RES/link).resolve(); doccheck(p.exists(),'README local link '+link); local_links.append(link)
    files=['README.md','decoder_metrics.csv','feature_reconstruction.csv','contrast_checks.csv','history_usefulness.csv','build_summary_tables.py','pytest.xml','pytest.log','extraction.log','analysis.log']
    return dict(status='passed',checks=document_checks,csv_rows=csv_rows,readme_tables=5,pytest=totals,local_links_verified=len(local_links),file_sha256={f:sha(RES/f) for f in files},interpretation_review='README correctly limits claims to retrospective frozen-checkpoint access via F12 and fixed probes; no common architecture bottleneck or causal information deletion claimed. Lower reconstruction alone is not treated as target-predictive loss. Independent audit scope limitations are stated.',numeric_or_interpretation_findings=[])


c=read(CON); manifest=read(RES/'execution_manifest.json'); reported=read(RES/'analysis.json'); decision=read(RES/'evidence_decision.json')
check(manifest['status']=='complete','completed execution required')
check(sha(CON)==manifest['contract_sha256'],'contract hash')
equal(read(RES/'execution_contract.json'),c,'contract snapshot')
hashes(manifest['source_hashes']); hashes(c['frozen_documents']); hashes(manifest['oof_hashes'])
check(sha(RES/'analysis.json')==manifest['analysis_sha256'],'analysis hash'); check(sha(RES/'evidence_decision.json')==manifest['decision_sha256'],'decision hash')
check(not manifest['validation_rows_materialized'] and not manifest['held_out_rows_materialized'] and not manifest['backbone_parameters_updated'] and not manifest['server_accessed'],'execution boundary metadata')
registry=read(ROOT/'paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json'); kvm=read(ROOT/'paper/results/hard_lmm_key_value_screening_5090_20260904/final_audit.json')
STAGES=list(c['stages']); FAMILIES=('linear','random128'); contrasts=c['primary_contrasts']
bootstrap_config=c['evidence_gate']['paired_series_bootstrap']; reps=bootstrap_config['repeats']; quant=bootstrap_config['lower_quantile']; seed=bootstrap_config['seed']
check(reps==10000 and quant==.05/32,'frozen bootstrap')
for ds in c['datasets']:
    meta=manifest['datasets'][ds]; hashes(meta['input_hashes']); row=next(x for x in registry['datasets'] if x['dataset']==ds)
    first_cache=None; cells[ds]={}
    for model in c['models']:
        print('independent auditing',ds,model,flush=True)
        rec=meta['models'][model]; path=ROOT/rec['cache_path']; check(sha(path)==rec['cache_sha256'],'cache '+str(path)); hash_count+=1
        oldrec=c['prior_caches'][ds][model]; check(sha(ROOT/oldrec['path'])==oldrec['sha256'],'prior cache'); hash_count+=1
        cache=torch.load(path,map_location='cpu',weights_only=False); old=torch.load(ROOT/oldrec['path'],map_location='cpu',weights_only=False)
        npz=np.load(RAW/ds/f'{model}_oof.npz'); report=reported[ds][model]
        n=len(cache['quantity']); check(n==8192,'cohort 8192')
        for key,oldkey in [('target_index','target_index'),('series_id','series_index'),('fold','fold'),('context_end','context_end'),('quantity','quantity'),('history_length','history_length')]:
            check(torch.equal(cache[key],old[oldkey]),'old paired identity '+key)
        for key in ('h','z','prediction'):
            check(torch.allclose(cache[key],old[key],rtol=1e-5,atol=1e-6),'old state parity '+key)
        expected_state=(row['checkpoint_state_sha256'] if model=='original' else kvm['runs'][ds]['checkpoint_state_sha256'])
        check(rec['state_sha256_unchanged']==expected_state,'state unchanged metadata'); check(rec['official_prediction_parity_first_batch'] and rec['layer2_equals_h'],'official parity metadata')
        q=ar(cache['quantity']); logq=ar(cache['log_quantity']); base=ar(cache['base_logpred']); F=ar(cache['history_features']); fold=ar(cache['fold']).astype(int); series=ar(cache['series_id']).astype(int)
        for key in ('quantity','log_quantity','base_logpred','history_features','fold','series_id'): equal(npz[key],ar(cache[key]),f'{ds}.{model}.npz.{key}')
        check(not set(series[fold==0])&set(series[fold==1]),'series separation')
        expected_fold=np.array([int.from_bytes(hashlib.sha256(f'20260905:{i}'.encode()).digest()[:8],'big')%2 for i in series])
        check(np.array_equal(fold,expected_fold),'fold hash')
        check(all(len(np.unique(series[fold==f]))>=10 for f in (0,1)),'series minimum')
        check(torch.equal(cache['h'],cache['layer2_last']),'layer2 equals h')
        check(torch.equal(cache['fused'],cache['h']+cache['r']),'fusion exact')
        check(torch.equal(cache['concat_hr'],torch.cat((cache['h'],cache['r']),dim=1)),'concat exact')
        if first_cache is None: first_cache={k:cache[k] for k in ('history_features','target_index','series_id','context_end','fold','quantity')}
        else:
            for k in first_cache: check(torch.equal(first_cache[k],cache[k]),'models paired '+k)
        b=float(cache['body_threshold']); t=float(cache['tail_threshold']); residual=logq-base
        equal(report['uncorrected_residual_mse'],mse(residual**2,fold),f'{ds}.{model}.uncorrected'); equal(report['base_raw_metrics'],raw(base,q,fold,b,t),f'{ds}.{model}.base')
        packs={k:ar(cache[k]) for k in STAGES}; sham=np.random.default_rng(c['controls']['sham_seed']).normal(size=(n,64))
        packs.update(F=F,constant=np.empty((n,0)),sham64=sham)
        for st in ('h','fused'):
            packs[st+'+F']=np.column_stack((packs[st],F)); packs[st+'+shamF']=np.column_stack((packs[st],sham[:,:12]))
        packs['fused+sham64']=np.column_stack((packs['fused'],sham))
        errors={}; raw_views={}; usefulness={}
        for family in FAMILIES:
            errors[family]={}; raw_views[family]={}
            for pack,x in packs.items():
                pred=npz[f'{family}__{pack}__residual_prediction']; err=(residual-pred)**2; errors[family][pack]=err
                label=f'{ds}.{model}.{family}.{pack}'
                equal(report['decoders'][family][pack]['residual_mse'],mse(err,fold),label+'.MSE')
                raw_views[family][pack]=raw(base+pred,q,fold,b,t); equal(report['decoders'][family][pack]['raw_metrics'],raw_views[family][pack],label+'.raw')
                for f in (0,1):
                    tr=x[fold!=f]; m=tr.mean(0); sd=tr.std(0); sd[sd==0]=1.; z=(tr-m)/sd
                    fm=report['decoders'][family][pack]['fit'][str(f)]
                    equal(fm['input_mean'],m,label+'.fit.input_mean'); equal(fm['input_scale'],sd,label+'.fit.input_scale'); equal(fm['target_mean'],residual[fold!=f].mean(),label+'.fit.target_mean')
                    if family=='random128' and x.shape[1]:
                        g=np.random.default_rng(seed); W=g.normal(size=(x.shape[1],128))/np.sqrt(x.shape[1]); bias=g.normal(size=128); z=np.column_stack((z,np.tanh(z@W+bias)))
                    equal(fm['design_mean'],z.mean(0),label+'.fit.design_mean'); check(fm['train_rows']==len(tr),'fit train rows')
            ce=errors[family]['constant']; fe=errors[family]['F']; gain=float((ce.mean()-fe.mean())/ce.mean()) if ce.mean()>0 else None
            fd={str(f):float((ce-fe)[fold==f].mean()) for f in (0,1)}; passed=gain is not None and gain>=.01 and all(v>0 for v in fd.values())
            usefulness[family]=bool(passed)
            equal(report['history_usefulness'][family],dict(relative_gain=gain,fold_mean_improvements=fd,passes=bool(passed)),f'{ds}.{model}.{family}.Fusefulness')
        check(report['history_usefulness_passes']==all(usefulness.values()),'history usefulness conjunction')
        keys=[]; values=[]
        for contrast,config in contrasts.items():
            for fam in FAMILIES:
                for ref in config['references']:
                    keys.append((contrast,fam,ref)); values.append(errors[fam][ref]-errors[fam][config['candidate']])
        boot_values=boot(np.column_stack(values),series,fold,reps,seed,quant); boots=dict(zip(keys,boot_values)); contrast_passes={}
        for contrast,config in contrasts.items():
            family_passes=[]
            for fam in FAMILIES:
                reference_passes=[]
                for ref in config['references']:
                    er=errors[fam][ref]; ec=errors[fam][config['candidate']]; imp=er-ec; gain=float((er.mean()-ec.mean())/er.mean()) if er.mean()>0 else None
                    fd={str(f):float(imp[fold==f].mean()) for f in (0,1)}; bv=boots[(contrast,fam,ref)]
                    rb=raw_views[fam][ref]['pooled']['body_mae']; cb=raw_views[fam][config['candidate']]['pooled']['body_mae']; conflict=gain is not None and gain>0 and rb is not None and cb is not None and cb>rb
                    conditions=dict(pooled_gain_at_least_one_percent=bool(gain is not None and gain>=.01),both_folds_positive=all(v>0 for v in fd.values()),bootstrap_p05_positive=bv['lower_p05']>0,family_quantile_positive=bv['lower_family_quantile']>0,no_log_vs_body_direction_conflict=not conflict)
                    rr=report['contrasts'][contrast]['families'][fam]['comparisons'][ref]; label=f'{ds}.{model}.{contrast}.{fam}.{ref}'
                    for k,v in bv.items(): equal(rr['bootstrap'][k],v,label+'.bootstrap.'+k)
                    equal(rr['reference_mse'],er.mean(),label+'.reference_mse'); equal(rr['candidate_mse'],ec.mean(),label+'.candidate_mse'); equal(rr['relative_gain'],gain,label+'.gain'); equal(rr['fold_mean_improvements'],fd,label+'.fold'); equal(rr['conditions'],conditions,label+'.conditions'); check(rr['passes']==all(conditions.values()),label+'.passes')
                    reference_passes.append(all(conditions.values()))
                check(report['contrasts'][contrast]['families'][fam]['passes']==all(reference_passes),'family pass'); family_passes.append(all(reference_passes))
            check(report['contrasts'][contrast]['passes']==all(family_passes),'contrast pass'); contrast_passes[contrast]=all(family_passes) and (not contrast.startswith('history_') or all(usefulness.values()))
        constantF=np.empty_like(F)
        for f in (0,1): constantF[fold==f]=F[fold!=f].mean(0)
        denom=((F-constantF)**2).mean(0)
        for stage in STAGES:
            pred=npz['reconstruct__'+stage]; e=(F-pred)**2; err=e.mean(0); recon=report['reconstruction'][stage]
            equal(recon['per_feature_mse'],err,f'{ds}.{model}.{stage}.reconstruction.mse'); equal(recon['per_feature_constant_mse'],denom,f'{ds}.{model}.{stage}.reconstruction.constant')
            r2=[float((v-ei)/v) if v>0 else None for ei,v in zip(err,denom)]
            for i,(a,bb) in enumerate(zip(recon['per_feature_oof_r2'],r2)): equal(a,bb,f'{ds}.{model}.{stage}.reconstruction.r2.{i}')
            for f in (0,1):
                equal(recon['folds'][str(f)]['per_feature_mse'],e[fold==f].mean(0),f'{ds}.{model}.{stage}.reconstruction.fold{f}')
        equal(npz['reconstruct__Ftruth'],F,f'{ds}.{model}.Ftruth')
        cells[ds][model]=dict(n=8192,fold_series=[len(np.unique(series[fold==f])) for f in (0,1)],history_usefulness=usefulness,eligible_contrasts=contrast_passes,bootstrap_comparisons=len(keys),bootstrap_repeats=reps)
        npz.close()
common={}
for contrast in contrasts:
    cc={f'{ds}/{m}':cells[ds][m]['eligible_contrasts'][contrast] for ds in c['datasets'] for m in c['models']}
    common[contrast]=dict(cells=cc,passes=all(cc.values()))
equal(decision['contrasts'],common,'common decision'); check(decision['any_common_evidence']==any(v['passes'] for v in common.values()),'common any')
result=dict(status='passed',completed_at=datetime.now(timezone.utc).isoformat(),method='Independent saved-artifact NumPy audit; no production analysis helper imports, no decoder refits, no model inference, no training, no remote access, no held-out or validation rows',numeric_tolerance=dict(atol=1e-10,rtol=1e-9),checks=checks,hash_checks=hash_count,max_absolute_difference=max_error,datasets=cells,common_evidence=common,manifest_sha256=sha(RES/'execution_manifest.json'),analysis_sha256=sha(RES/'analysis.json'),decision_sha256=sha(RES/'evidence_decision.json'),contract_sha256=sha(CON),auditor_script_sha256=sha(__file__),limits=['Underlying history feature extraction was not rerun from parquet; stage cache integrity, prior identity/parity and construction identities were audited.','Checkpoint unchanged state is reconciled to pinned prior state hashes and execution metadata; no model object was restored.','Fit metadata is checked against opposite-fold inputs and labels; decoders are not independently refitted.','Bootstrap reproduces fixed OOF conditional stability, not sample-selection or fitting uncertainty.','No information-theoretic deletion, causal architecture effect, independent generalization or new model adoption is established.'])
result['document_audit']=audit_documents()
import argparse
parser=argparse.ArgumentParser(description='Independently audit saved Hard-LMM information diagnostic artifacts; no refits or inference.')
parser.add_argument('--output',type=Path,default=RES/'independent_audit.json',help='Fresh output path; refuses overwrite. For reruns choose a path under /private/tmp.')
path=parser.parse_args().output; check(not path.exists(),'Refusing independent audit overwrite'); path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'); print(json.dumps(result,ensure_ascii=False),flush=True)
