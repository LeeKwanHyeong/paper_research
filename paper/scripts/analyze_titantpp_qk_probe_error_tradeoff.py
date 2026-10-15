"""Pinned saved Train/Validation on/off arithmetic; no neural imports or inference."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from zoneinfo import ZoneInfo

PROJECT=Path(__file__).resolve().parents[2]
REPORT=PROJECT/'reports/titantpp_backbone_qk_branch_probe_20261016_v1'
PRIOR=REPORT/'diagnostics_branch_probe_v1'
ROOT=REPORT/'tradeoff_decomposition_v1'
OUTPUT=ROOT/'result_v1'
BUNDLE=PROJECT/'search_artifacts/titantpp_backbone_qk_branch_probe_20261016_v1'
SPEC_SHA='85bb16175995a2444c01fa3b09934d7fe2fda43fc0ddb76a72c05ff3ecf095d9'
RECEIPT_SHA='21f9b868cc9d8833e6540704a8ea76b72bf197eb02b8f603f4c57b114fb4cacb'
BOOT_SHA='ca00f33e3f73944110211776867a1b6ddd7c76f995a7d0b4a4fc07ea31d9216f'
PRIOR_RECEIPT_SHA='dd4881811a3d27b5fb7b7c759563d101933fa1fb42be089af4e6ed7ddc885bac'
RESULT_SHA='2155f65eed92298a882f5fa152d9dd2faca78b08ec234e351c950443230b4534'
COMPARISON_SHA='0e83b9062249090c8db124f3795947e6299451dab5cb4d55b85a030fb776e753'
SEEDS=(42,52,62)
SPLITS=('train','validation')
EPOCHS={42:5,52:5,62:22}
RTOL=1e-9
ATOL=1e-7
PARTITIONS={'effect':('both_improve','both_worsen','equal'),
 'on_error_sign':('negative','zero','positive'),
 'prediction_shift_sign':('negative','zero','positive'),
 'zero_crossing':('negative_to_positive','positive_to_negative','arrives_zero','leaves_zero','zero_to_zero','no_crossing'),
 'quantity':('non_tail_q_le_200','tail_q_gt_200')}
ADDITIVE=('on_ae','off_ae','on_se','off_se','on_signed','off_signed','shift','cross_term','squared_shift','delta_ae','delta_se','delta_signed','on_time','off_time','delta_time')
CORRECTION='Same-row AE and SE changes have the same sign: deltaSE=deltaAE*(AEoff+AEon). Earlier INTERPRETATION_REVIEW proposal of substantive same-row SE-benefit/AE-harm categories is corrected; original bytes preserved. Aggregate tradeoffs concern different row contributions, not opposite same-row AE/SE directions.'


def need(v,m):
    if not v:raise ValueError(m)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sign(v):return -1 if v<0 else 1 if v>0 else 0
def sign_name(v):return ('zero','positive','negative')[sign(v)]


def close_identity(left,terms):
    need(math.isfinite(left) and all(math.isfinite(x) for x in terms),'Nonfinite identity')
    residual=left-math.fsum(terms)
    need(abs(residual)<=ATOL+RTOL*max(abs(left),math.fsum(abs(x) for x in terms)),'Arithmetic identity failed')
    return residual


def row_terms(row):
    """Internal legacy pure-function names: candidate=off, cnn=on, never historical rows."""
    e=float(row['cnn_error']);o=float(row['candidate_error'])
    delta=float(row['candidate_prediction'])-float(row['cnn_prediction'])
    close_identity(o-e,(delta,))
    a=abs(e);b=abs(o);da=b-a;ds=o*o-e*e
    cross=2*e*delta;square=delta*delta
    vals=(e,o,delta,a,b,da,ds,cross,square,float(row['cnn_time_nll']),float(row['candidate_time_nll']))
    need(all(math.isfinite(x) for x in vals),'Nonfinite row arithmetic')
    res1=close_identity(ds,(cross,square));res2=close_identity(ds,(da*(a+b),))
    mismatch=sign(ds)!=sign(da)
    if mismatch:need(abs(ds)<=ATOL+RTOL*max(e*e,o*o) or abs(da)<=ATOL+RTOL*max(a,b),'Substantive AE/SE sign mismatch')
    effect='both_improve' if da<0 else 'both_worsen' if da>0 else 'equal'
    crossing=('zero_to_zero' if e==0 and o==0 else 'arrives_zero' if o==0 else 'leaves_zero' if e==0 else
              'negative_to_positive' if e<0<o else 'positive_to_negative' if e>0>o else 'no_crossing')
    return {'on_ae':a,'off_ae':b,'on_se':e*e,'off_se':o*o,'on_signed':e,'off_signed':o,
            'shift':delta,'cross_term':cross,'squared_shift':square,'delta_ae':da,'delta_se':ds,'delta_signed':o-e,
            'on_time':float(row['cnn_time_nll']),'off_time':float(row['candidate_time_nll']),
            'delta_time':float(row['candidate_time_nll'])-float(row['cnn_time_nll']),
            'effect':effect,'on_error_sign':sign_name(e),'prediction_shift_sign':sign_name(delta),
            'zero_crossing':crossing,'quantity':'tail_q_gt_200' if row['raw_quantity']>200 else 'non_tail_q_le_200',
            'numerical_sign_mismatch':mismatch,'max_identity_residual':max(abs(res1),abs(res2))}


def total(rows,weights,terms,parent_mass):
    need(len(rows)==len(weights)==len(terms),'Weight count mismatch')
    need(all(math.isfinite(w) and w>0 for w in weights),'Invalid weight')
    sums={k:math.fsum(w*t[k] for w,t in zip(weights,terms)) for k in ADDITIVE}
    mass=math.fsum(weights)
    return {'count':len(rows),'entities':len({str(r['entity_id']) for r in rows}),'weight_sum':mass,
            'sums':sums,'contribution_per_parent_weight':{k:v/parent_mass if parent_mass else None for k,v in sums.items()},
            'mean_on_AE':sums['on_ae']/mass if mass else None,'mean_off_AE':sums['off_ae']/mass if mass else None,
            'on_rmse':math.sqrt(sums['on_se']/mass) if mass else None,'off_rmse':math.sqrt(sums['off_se']/mass) if mass else None,
            'numerical_sign_mismatch_count':sum(t['numerical_sign_mismatch'] for t in terms),
            'max_identity_residual':max((t['max_identity_residual'] for t in terms),default=0.)}


def one_cell(rows,weights):
    terms=[row_terms(r) for r in rows];mass=math.fsum(weights);whole=total(rows,weights,terms,mass);partitions={};closures=[]
    for family,labels in PARTITIONS.items():
        groups={}
        for label in labels:
            ix=[i for i,t in enumerate(terms) if t[family]==label]
            groups[label]=total([rows[i] for i in ix],[weights[i] for i in ix],[terms[i] for i in ix],mass)
        need(sum(g['count'] for g in groups.values())==len(rows),'Partition coverage differs')
        close_identity(mass,tuple(g['weight_sum'] for g in groups.values()))
        for key in ADDITIVE:close_identity(whole['sums'][key],tuple(g['sums'][key] for g in groups.values()))
        partitions[family]=groups;closures.append({'partition':family,'passed':True})
    close_identity(whole['sums']['delta_se'],(whole['sums']['cross_term'],whole['sums']['squared_shift']))
    close_identity(whole['sums']['delta_ae'],(whole['sums']['off_ae'],-whole['sums']['on_ae']))
    close_identity(whole['sums']['delta_se'],(whole['sums']['off_se'],-whole['sums']['on_se']))
    return {'total':whole,'partitions':partitions,'closures':closures}


def decompose(cohorts,cuts,pure):
    need(set(cohorts)=={(s,p) for s in SEEDS for p in SPLITS},'Incomplete seed/split cohort')
    out=[];support=[]
    for family,pattern,mag in pure.cell_list():
        cell='|'.join(str(x) for x in (family,pattern,mag));chosen={}
        for key,rows in cohorts.items():
            chosen[key]=[r for r in rows if (pattern is None or pure.fixed_features(r,cuts)[0]==pattern) and (mag is None or pure.fixed_features(r,cuts)[1]==mag)]
        for seed in SEEDS:
            common={str(r['entity_id']) for r in chosen[seed,'train']}&{str(r['entity_id']) for r in chosen[seed,'validation']}
            for split in SPLITS:
                full=chosen[seed,split];kept=[r for r in full if str(r['entity_id']) in common];counts=Counter(str(r['entity_id']) for r in kept)
                weights=[1/counts[str(r['entity_id'])] for r in kept]
                support.append({'cell':cell,'seed':seed,'split':split,'original_rows':len(full),'retained_rows':len(kept),'excluded_rows':len(full)-len(kept),'common_entities':len(common)})
                for weighting,rows,ws in [('row',full,[1.]*len(full)),('common_entity_equal',kept,weights)]:
                    out.append({'cell':cell,'family':family,'seed':seed,'split':split,'weighting':weighting,**one_cell(rows,ws)})
    means=[]
    for cell in dict.fromkeys(x['cell'] for x in out):
        for split in SPLITS:
            for weighting in ('row','common_entity_equal'):
                xs=[x['total'] for x in out if x['cell']==cell and x['split']==split and x['weighting']==weighting]
                need(len(xs)==3,'Incomplete aggregation')
                metrics={}
                for key in ADDITIVE:
                    values=[x['contribution_per_parent_weight'][key] for x in xs]
                    metrics[key]=({'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)} if all(x is not None for x in values) else None)
                means.append({'cell':cell,'split':split,'weighting':weighting,'per_parent_weight_contributions':metrics})
    return {'per_seed':out,'support':support,'three_seed':means}


def load_pinned(path,name,digest,pins):
    pins.add(path,digest);spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def read_cohorts(receipt_sha):
    """Read only current diagnostic rows and pure Train/Validation input; byte-pin all dependencies."""
    need(receipt_sha==RECEIPT_SHA,'Unapproved native receipt')
    helper_path=PROJECT/'paper/scripts/compare_titantpp_value_difference_rows.py'
    need(sha(helper_path)=='3844e5f6dd335aa0a0b7421f0d45780784b1f1b4b5f4e980f84827cd47ce475e','Helper changed')
    spec=importlib.util.spec_from_file_location('_tradeoff_saved',helper_path);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    pins=h.Pins(PROJECT);pins.add(helper_path,'3844e5f6dd335aa0a0b7421f0d45780784b1f1b4b5f4e980f84827cd47ce475e')
    pins.add(Path(__file__),sha(__file__));pins.add(ROOT/'PAIRED_DECOMPOSITION_SPEC.json',SPEC_SHA)
    c=load_pinned(PROJECT/'paper/scripts/compare_titantpp_qk_branch_probe.py','_tradeoff_cmp',COMPARISON_SHA,pins)
    pure=load_pinned(PROJECT/'paper/scripts/analyze_titantpp_value_difference_rows.py','_tradeoff_pure',c.PURE_SHA,pins)
    pr=read(pins.add(PRIOR/'RECEIPT.json',PRIOR_RECEIPT_SHA));prior=read(pins.add(PRIOR/'RESULT.json',RESULT_SHA))
    need(pr['status']=='complete' and pr['native_receipt_sha256']==receipt_sha and prior['native_receipt_sha256']==receipt_sha,'Prior analysis binding differs')
    need(pr['output_files']['RESULT.json']==RESULT_SHA and pr['paired_split_count']==6 and prior['allowed_splits']==list(SPLITS),'Prior analysis scope differs')
    final=read(pins.add(BUNDLE/'retrieval/final_receipt.json',receipt_sha));boot=read(pins.add(BUNDLE/'bootstrap_integrity.json',BOOT_SHA))
    need(final['status']=='complete' and final['bootstrap_integrity_sha256']==BOOT_SHA and final['data_worker_count']==12 and final['actual_data_worker_exit0_count']==12,'Native receipt incomplete')
    need(final['new_training'] is False and final['historical_CNN_inference'] is False and final['held_out_test_evaluated'] is False,'Foreign native scope')
    for rel,digest in boot['files'].items():pins.add(h.safe(BUNDLE,rel),digest)
    for rel,digest in final['files'].items():
        path=h.safe(PROJECT,rel);need(path.is_relative_to(BUNDLE),'Foreign final artifact');pins.add(path,digest)
    contract=read(pins.add(BUNDLE/'contract.json',final['contract_raw_sha256']))
    need(boot['contract_raw_sha256']==final['contract_raw_sha256'] and contract['contract_sha256']==final['contract_sha256'],'Contract binding differs')
    need(contract['source']['closure']==c.SOURCE and len(contract['source']['files'])==143,'Source143 changed')
    jobs=contract['jobs'];need(len(jobs)==3 and {j['seed'] for j in jobs}==set(SEEDS),'Wrong jobs')
    need(all(j['arm']=='titantpp_cnn_qk_only_v1' and j['dataset']=='raf_spare_parts' and j['selected_epoch']==EPOCHS[j['seed']] for j in jobs),'Wrong candidate/epoch')
    orig=h.safe(PROJECT,final['originals_path']);need(orig==BUNDLE/'retrieval/originals','Foreign originals')
    audit=read(pins.add(orig/'ARCHIVE_MANIFEST.json',final['manifest_sha256']))
    for rel,item in audit['files'].items():
        path=pins.add(h.safe(orig,rel),item['sha256']);need(path.stat().st_size==item['bytes'],'Original bytes differ')
    pins.add(BUNDLE/'retrieval/originals.tar.gz',final['archive_sha256'])
    reconstruction=load_pinned(h.PRIOR/'analyze.py','_tradeoff_reconstruction',h.RECONSTRUCTION_SHA,pins)
    pins.add(h.PRIOR/'PLAN.json',h.PLAN_SHA)
    frozen=read(pins.add(h.PRIOR/'RESULT.json',h.PRIOR_RESULT_SHA));cuts=(frozen['cuts']['q25'],frozen['cuts']['q75'])
    need(frozen['plan_sha256']==h.PLAN_SHA and frozen['cuts']['uses_target_quantity'] is False and list(cuts)==prior['cuts'],'Frozen cuts differ')
    import numpy as np
    import polars as pl
    data=contract['datasets']['raf_spare_parts'];raw=pl.read_parquet(pins.add(h.safe(BUNDLE,data['path']),h.INPUT_SHA)).sort(['oper_part_no','seq'])
    need(set(raw['chronological_split'].unique())==set(SPLITS) and raw.height==37469,'Foreign split/count')
    histories={s:reconstruction.build_rows(raw,s,data['populations'][s])[0] for s in SPLITS}
    eligible=[max(abs(r['d1']),abs(r['d2'])) for r in histories['train'] if r['history_length']>=3]
    need(tuple(float(x) for x in np.quantile(eligible,[.25,.75],method='linear'))==cuts and len(eligible)==frozen['cuts']['eligible_train_targets'],'Train quantiles differ')
    contexts=h.reconstruct_contexts(raw,histories,np);del raw
    columns=['dataset','seed','split','entity_id','seq',*h.COMMON,'prediction','time_nll','recorded_date','recorded_gap','target_id','observed_history_sha256','signed_error','absolute_error','squared_error']
    cohorts={}
    for job in sorted(jobs,key=lambda j:j['seed']):
        h.verify_selection(BUNDLE,job,pins,c.SOURCE,EPOCHS[job['seed']])
        for split in SPLITS:
            groups={};identities={};states={}
            for role in ('on','off'):
                stage=role+'_'+split;entry=final['data_receipts'][job['id']+'/'+stage]
                receipt=read(pins.add(h.safe(orig,entry['path']),entry['sha256']))
                need(receipt['job_id']==job['id'] and receipt['stage']==stage and receipt['split']==split and receipt['condition']==role,'Wrong condition receipt')
                need(receipt['selected_epoch']==job['selected_epoch'] and receipt['checkpoint_sha256']==job['checkpoint']['sha256'] and receipt['source_closure']==c.SOURCE and receipt['data_sha256']==h.INPUT_SHA,'Selection/science mismatch')
                need(receipt['predictive_state_before']==receipt['predictive_state_after'] and receipt['parameters_unchanged'] is True and receipt['state_including_buffers_unchanged'] is True and receipt['runtime']==receipt['runtime_after'],'State/runtime drift')
                mp=pins.add(h.safe(orig,receipt['manifest_path']),receipt['manifest_sha256']);manifest=read(mp)
                groups[role]=h.read_shards(mp.parent,manifest,pins,columns,pl)
                identities[role]=h.verify_candidate_row_identity(groups[role],receipt,contexts[split],job,split)
                states[role]=(receipt['predictive_state_before'],receipt['runtime'])
                metrics=h.aggregate_rows(groups[role],'prediction','time_nll');h.check_metrics(metrics,receipt['metrics'])
                evidence=next(x for x in prior['evidence'] if x['seed']==job['seed'] and x['split']==split)
                need(entry['sha256']==evidence[role+'_receipt_sha256'],'Prior condition receipt changed');h.check_metrics(metrics,evidence[role+'_metrics'])
            need(states['on']==states['off'] and identities['on']==identities['off'],'Paired states/targets differ')
            cohorts[job['seed'],split]=c.paired(groups['off'],groups['on'],histories[split],pure,job['seed'],split)
    summary=c.rename_roles(pure.summarize(cohorts,cuts));summary['limitation']=prior['summary']['limitation']
    need(summary==prior['summary'],'Original15conditions/support/weights no longer reproduce exactly')
    pins.again();return cohorts,cuts,pure,pins,prior,h


def render(result):
    lines=['# 저장된 Q/K on/off 오류 기여 분해','',CORRECTION,'',
           '동일행 절대오차와 제곱오차 증감 방향은 같다. 집계에서 두 지표가 엇갈리는 이유를 개선행과 악화행의 기여 규모로 나눴다. 부호·정답 수량 분류는 사후 설명이며 입력 gate나 학습 선택 기준이 아니다.','',
           '| 구간 | seed | 행 분류 | 행수 | on AE 평균 | ΔAE 합 | ΔSE 합 | 2eδ 합 | δ² 합 |','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for cell in result['decomposition']['per_seed']:
        if cell['family']!='whole' or cell['weighting']!='row':continue
        for label,g in cell['partitions']['effect'].items():
            s=g['sums'];ae='N/A' if g['mean_on_AE'] is None else f"{g['mean_on_AE']:.6f}"
            lines.append('| '+' | '.join([cell['split'],str(cell['seed']),label,str(g['count']),ae,*[f'{s[k]:+.6f}' for k in ('delta_ae','delta_se','cross_term','squared_shift')]])+' |')
    lines+=['','모든 seed·15개 이력 조건·두 가중 방식의 양/음/0, 부호·교차·큰수량·시간 기여는 JSON에 보존했다. 새 분할 안에서 공통 개체나 가중치를 다시 정하지 않았다. 조건별 공통 개체 집합은 달라 서로 합산하지 않는다.',
            '기존 CNN·외부모델·Test·새 추론·원격 작업은 없다. 이 분석은 Q/K의 크기·방향·상수성분·메모리 배분 원인을 식별하거나 다음 구조를 자동 선택하지 않는다.']
    return '\n'.join(lines)+'\n'


def run(receipt_sha):
    need(not OUTPUT.exists(),'Existing output; no overwrite/retry')
    cohorts,cuts,pure,pins,prior,h=read_cohorts(receipt_sha)
    decomposition=decompose(cohorts,cuts,pure)
    need(decomposition['support']==prior['summary']['support'],'Original common-entity support changed')
    for cell in decomposition['per_seed']:
        old=next(x for x in prior['summary']['per_seed'] if all(x[k]==cell[k] for k in ('cell','seed','split','weighting')))
        if old['metrics'] is None:need(cell['total']['count']==0,'Empty support changed');continue
        for new,key in [('delta_ae','ae_sum'),('delta_se','se_sum'),('delta_signed','signed_sum')]:close_identity(cell['total']['sums'][new],(old['metrics']['difference'][key],))
    result={'status':'complete','created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'native_receipt_sha256':receipt_sha,'prior_result_sha256':RESULT_SHA,'spec_sha256':SPEC_SHA,'mathematical_correction':CORRECTION,'decomposition':decomposition,'inference_count':0,'remote_count':0,'held_out_test_read':False,'historical_model_rows_loaded':False,'new_training':False,'interpretation':'Descriptive paired contribution decomposition only; no internal Q/K cause, input gate or next training automatically established.'}
    pins.again();OUTPUT.mkdir(exist_ok=False)
    h.write(OUTPUT/'RESULT.json',result)
    with (OUTPUT/'RESULT.md').open('x') as f:f.write(render(result))
    pins.again()
    receipt={'status':'complete','native_receipt_sha256':receipt_sha,'inputs':pins.files,'outputs':{n:sha(OUTPUT/n) for n in ('RESULT.json','RESULT.md')},'input_SHA_unchanged':True,'cohort_count':6,'fixed_cells_per_cohort':15,'weighting_count':2,'partition_closure_count':sum(len(x['closures']) for x in decomposition['per_seed']),'inference_count':0,'remote_count':0,'held_out_test_read':False,'new_training':False}
    h.write(OUTPUT/'RECEIPT.json',receipt)
    return {'status':'complete','output':str(OUTPUT),'receipt_sha256':sha(OUTPUT/'RECEIPT.json'),'outputs':receipt['outputs']}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt-sha',required=True);args=parser.parse_args();print(json.dumps(run(args.receipt_sha),ensure_ascii=False))
