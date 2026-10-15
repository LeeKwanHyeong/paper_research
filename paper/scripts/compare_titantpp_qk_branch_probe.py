"""Read-only off-minus-on comparison of fixed current-candidate Train/Validation rows.

No neural imports, old CNN access, forward inference, Test access or remote operation.
"""
from __future__ import annotations
import argparse
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
BUNDLE=PROJECT/'search_artifacts/titantpp_backbone_qk_branch_probe_20261016_v1'
OUTPUT=REPORT/'diagnostics_branch_probe_v1'
SOURCE='74e79d12538ac005665043fd20f875a9883a436c6a5d02f2768cfd538a8566f7'
EPOCHS={42:5,52:5,62:22}
HELPER_SHA='3844e5f6dd335aa0a0b7421f0d45780784b1f1b4b5f4e980f84827cd47ce475e'
PURE_SHA='c689918ad4f64913cf1b15894a3e4e177667711675a59e9056e72bfa6400cb33'
SPEC_SHA='10371800c7e0f4095defad1380344a5bef19a9b31578dd599bc163779b1af7ee'
ACCEPTANCE_SHA='5bc011e4473f71db5fd520af8ff3763753c767ceeb98d12cfdc1a6f2c4063193'
SEEDS=(42,52,62)
SPLITS=('train','validation')
COUNTS={'train':25779,'validation':6690}
INTERPRETATION='Fixed jointly trained checkpoint sensitivity to both Q/K local residuals; off is neither original CNN nor a model retrained without Q/K local residuals. No generalization, causal or significance claim.'


def need(value,message):
    if not value:raise ValueError(message)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def load(p,name,expected):
    need(sha(p)==expected,'Analysis dependency changed: '+str(p))
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def rename_roles(value):
    """Explicit labels for the frozen arithmetic module; never call off a CNN baseline."""
    if isinstance(value,list):return [rename_roles(v) for v in value]
    if isinstance(value,dict):return {('off' if k=='candidate' else 'on' if k=='cnn' else k):rename_roles(v) for k,v in value.items()}
    return value


def identical_targets(off,on):
    fields=('dataset','seed','split','entity_id','seq','target_id','raw_quantity','recorded_gap','recorded_date','observed_history_sha256')
    index={r['target_id']:r for r in on}
    need(len(index)==len(on)==len(off),'Duplicate/unequal on/off targets')
    seen=set()
    for r in off:
        key=r['target_id'];need(key in index and key not in seen,'Foreign/duplicate off target');seen.add(key)
        need(all(r[f]==index[key][f] for f in fields),'On/off target truth or history differs')
        for a in (r,index[key]):
            e=float(a['prediction'])-float(a['raw_quantity'])
            for f,v in [('signed_error',e),('absolute_error',abs(e)),('squared_error',e*e)]:
                need(math.isfinite(v) and math.isclose(float(a[f]),v,rel_tol=1e-12,abs_tol=1e-12),'Saved row error differs')
    return True


def paired(off,on,history,pure,seed,split):
    identical_targets(off,on)
    reference=[{**r,'prediction_1':r['prediction'],'time_nll_1':r['time_nll']} for r in on]
    return pure.paired_rows(off,reference,history,seed=seed,split=split,expected_count=COUNTS[split])


def metric_groups(evidence):
    out=[]
    for split in SPLITS:
        group=[e for e in evidence if e['split']==split]
        need(len(group)==3 and {e['seed'] for e in group}==set(SEEDS),'Incomplete six-metric cohort')
        for population in ('whole','tail'):
            for metric in ('qty_rmse','qty_mae','time_nll'):
                row={'split':split,'population':population,'metric':metric}
                vals={role:[(e[role+'_metrics'] if population=='whole' else e[role+'_metrics']['tail'])[metric] for e in group] for role in ('on','off')}
                vals['off_minus_on']=[a-b if a is not None and b is not None else None for a,b in zip(vals['off'],vals['on'])]
                for role,v in vals.items():row[role]=None if any(x is None for x in v) else {'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)}
                out.append(row)
    return out


def render(result):
    lines=['# RAF 고정 모델 Q/K 지역 보정 켬/끔 비교','',
      '같은 선택 checkpoint의 Q/K 지역 잔차를 함께 끈 상태(off)와 원래 상태(on)를 비교했다. 양수 off−on은 보정을 끄면 악화함을 뜻한다.',
      '이 비교의 off는 기존 CNN이나 Q/K 지역 보정 없이 새로 학습한 모델이 아니다. 기존 CNN 추론, 새 학습, Test 평가는 하지 않았다.','',
      '| 구간 | seed | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['summary']['per_seed']:
        if r['family']!='whole' or r['weighting']!='row':continue
        m=r['metrics'];lines.append('| '+r['split']+' | '+str(r['seed'])+' | '+' | '.join(f'{v:.6f}' for v in (m['on']['rmse'],m['off']['rmse'],m['difference']['rmse'],m['on']['mae'],m['off']['mae'],m['difference']['mae']))+' |')
    lines+=['','## 세 seed 평균 ± 표본 SD','', '| 구간 | 가중 | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |','|---|---|---:|---:|---:|---:|---:|---:|']
    def stat(v):return f"{v['mean']:.6f} ± {v['sample_sd']:.6f}"
    for r in result['summary']['three_seed']:
        if r['cell']!='whole|None|None':continue
        m=r['metrics'];lines.append('| '+r['split']+' | '+r['weighting']+' | '+' | '.join(stat(m[k][v]) for k,v in [('on','rmse'),('off','rmse'),('difference','rmse'),('on','mae'),('off','mae'),('difference','mae')])+' |')
    lines+=['','## 전체·큰 수량·시간 지표','', '| 구간 | 대상 | 지표 | on | off | off−on |','|---|---|---|---:|---:|---:|']
    for r in result['six_metric_groups']:
        lines.append('| '+' | '.join([r['split'],r['population'],r['metric'],*[stat(r[k]) if r[k] else 'N/A' for k in ('on','off','off_minus_on')]])+' |')
    lines+=['','## 고정 이력 조건','', '| 조건 | 구간 | 가중 | 평균 ΔRMSE | 평균 ΔMAE | 평균 ΔSE합 | 평균 ΔAE합 | 평균 Δ부호오차합 |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in result['summary']['three_seed']:
        m=r['metrics']['difference'];lines.append('| '+' | '.join([r['cell'].replace('|',' / '),r['split'],r['weighting'],*[f"{m[k]['mean']:+.6f}" for k in ('rmse','mae','se_sum','ae_sum','signed_sum')]])+' |')
    lines+=['','15개 이력 조건과 Train Q25/Q75를 미리 고정했고, 모든 seed별 지표·악화·예외와 공통 개체 가중의 포함/제외 행을 JSON에 보존했다. 각 조건의 공통 개체 집합은 달라 가중 조건들의 합계를 전체로 해석하지 않는다.',
      '행 가중의 ΔAE·ΔSE·부호오차 합은 각 분할 합계와 일치하는지 검증했다. RMSE 자체를 가산 분해하지 않았다.',
      '같은 5090 Runtime·공동 학습된 가중치·입력·head·메모리를 유지한 분기 민감도다. 분기 없이 재학습했을 때의 효과, 일반화 개선 또는 오류 원인 확정으로 해석하지 않는다. 기존 Test를 참고한 탐색적 후속 진단이다.']
    return '\n'.join(lines)+'\n'


def run(receipt_sha):
    need(not OUTPUT.exists(),'Existing analysis output; no overwrite or retry')
    h=load(PROJECT/'paper/scripts/compare_titantpp_value_difference_rows.py','_probe_saved_helpers',HELPER_SHA)
    pure=load(PROJECT/'paper/scripts/analyze_titantpp_value_difference_rows.py','_probe_fixed_arithmetic',PURE_SHA)
    pins=h.Pins(PROJECT)
    for p,d in [(Path(__file__),sha(__file__)),(PROJECT/'paper/scripts/compare_titantpp_value_difference_rows.py',HELPER_SHA),(PROJECT/'paper/scripts/analyze_titantpp_value_difference_rows.py',PURE_SHA),(REPORT/'SCIENCE_SPEC.json',SPEC_SHA),(REPORT/'ACCEPTANCE.json',ACCEPTANCE_SHA)]:pins.add(p,d)
    final=h.read(pins.add(BUNDLE/'retrieval/final_receipt.json',receipt_sha))
    boot=h.read(pins.add(BUNDLE/'bootstrap_integrity.json',final['bootstrap_integrity_sha256']))
    for p,d in boot['external_dependencies'].items():pins.add(p,d)
    for p,d in boot['files'].items():pins.add(h.safe(BUNDLE,p),d)
    for p,d in final['files'].items():pins.add(h.safe(PROJECT,p),d)
    guard_path=PROJECT/'paper/scripts/advance_titantpp_qk_branch_probe.py'
    guard=load(guard_path,'_probe_completed_verifier',boot['external_dependencies'][str(guard_path)])
    verified=guard.verify_complete(final['bootstrap_integrity_sha256'])
    need(verified==final,'Completed native evidence differs')
    c=h.read(pins.add(BUNDLE/'contract.json',final['contract_raw_sha256']));orig=h.safe(PROJECT,final['originals_path'])
    need(final['historical_CNN_inference'] is False and final['held_out_test_evaluated'] is False and final['new_training'] is False,'Foreign native scope')
    need(c['source']['closure']==SOURCE and len(c['source']['files'])==143,'QKOnly source lineage differs')
    prior=h.read(pins.add(h.PRIOR/'RESULT.json',h.PRIOR_RESULT_SHA));pins.add(h.PRIOR/'PLAN.json',h.PLAN_SHA)
    reconstruction=load(pins.add(h.PRIOR/'analyze.py',h.RECONSTRUCTION_SHA),'_probe_frozen_histories',h.RECONSTRUCTION_SHA)
    cuts=(prior['cuts']['q25'],prior['cuts']['q75'])
    need(prior['plan_sha256']==h.PLAN_SHA and prior['cuts']['uses_target_quantity'] is False,'Frozen history plan differs')
    import numpy as np
    import polars as pl
    data=c['datasets']['raf_spare_parts'];raw=pl.read_parquet(pins.add(h.safe(BUNDLE,data['path']),h.INPUT_SHA)).sort(['oper_part_no','seq'])
    need(set(raw['chronological_split'].unique())==set(SPLITS) and raw.height==37469,'Foreign input split/count')
    histories={};audits={}
    for split in SPLITS:histories[split],audits[split]=reconstruction.build_rows(raw,split,data['populations'][split])
    eligible=[max(abs(r['d1']),abs(r['d2'])) for r in histories['train'] if r['history_length']>=3]
    calculated=tuple(float(v) for v in np.quantile(eligible,[.25,.75],method='linear'))
    need(calculated==cuts and len(eligible)==prior['cuts']['eligible_train_targets'],'Frozen Train cuts changed')
    contexts=h.reconstruct_contexts(raw,histories,np);del raw
    columns=['dataset','seed','split','entity_id','seq',*h.COMMON,'prediction','time_nll','recorded_date','recorded_gap','target_id','observed_history_sha256','signed_error','absolute_error','squared_error']
    rows_by={};evidence=[]
    for j in sorted(c['jobs'],key=lambda j:j['seed']):
        need(j['selected_epoch']==EPOCHS[j['seed']],'Selected epoch changed')
        h.verify_selection(BUNDLE,j,pins,SOURCE,EPOCHS[j['seed']])
        for split in SPLITS:
            groups={};receipts={};ids={}
            for role,root,f,key in [('on',orig,final,j['id']+'/on_'+split),('off',orig,final,j['id']+'/off_'+split)]:
                entry=f['data_receipts'][key];r=h.read(pins.add(h.safe(root,entry['path']),entry['sha256']))
                need(r['split']==split and r['job_id']==j['id'] and r['selected_epoch']==j['selected_epoch'] and r['checkpoint_sha256']==j['checkpoint']['sha256'],'Condition receipt selection differs')
                need(r['predictive_state_before']==r['predictive_state_after'] and r['parameters_unchanged'] is True and r['state_including_buffers_unchanged'] is True,'Condition mutated model state')
                need(r['runtime']==r['runtime_after'] and r['data_sha256']==h.INPUT_SHA and r['source_closure']==SOURCE,'Condition scientific/runtime binding differs')
                mp=pins.add(h.safe(root,r['manifest_path']),r['manifest_sha256']);manifest=h.read(mp)
                groups[role]=h.read_shards(mp.parent,manifest,pins,columns,pl);receipts[role]=r
                ids[role]=h.verify_candidate_row_identity(groups[role],r,contexts[split],j,split)
                ids[role]['target_hash_domain']='Same pure Train/Validation input and target definition for both current-candidate conditions'
                h.check_metrics(h.aggregate_rows(groups[role],'prediction','time_nll'),r['metrics'])
            need(receipts['on']['predictive_state_before']==receipts['off']['predictive_state_before'] and receipts['on']['runtime']==receipts['off']['runtime'],'On/off initial state/runtime mismatch')
            need(ids['on']==ids['off'],'On/off target fingerprints differ')
            rows_by[j['seed'],split]=paired(groups['off'],groups['on'],histories[split],pure,j['seed'],split)
            evidence.append({'seed':j['seed'],'split':split,'count':COUNTS[split],'on_metrics':h.aggregate_rows(groups['on'],'prediction','time_nll'),'off_metrics':h.aggregate_rows(groups['off'],'prediction','time_nll'),'on_receipt_sha256':final['data_receipts'][j['id']+'/on_'+split]['sha256'],'off_receipt_sha256':final['data_receipts'][j['id']+'/off_'+split]['sha256'],'target_identity':ids['on'],'initial_state_identical':True})
    summary=rename_roles(pure.summarize(rows_by,cuts));summary['limitation']=INTERPRETATION
    result={'schema':'fixed_current_candidate_off_minus_on_v1','created_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'candidate_count':3,'selected_epochs':EPOCHS,'allowed_splits':list(SPLITS),'cuts':list(cuts),'summary':summary,'evidence':evidence,'six_metric_groups':metric_groups(evidence),'reconstruction':audits,'native_receipt_sha256':receipt_sha,'on_rows_from_same_diagnostic':True,'analysis_inference_count':0,'analysis_remote_count':0,'held_out_test_read':False,'historical_CNN_inference':False,'CNN_prediction_rows_loaded':False,'interpretation':INTERPRETATION}
    pins.again();OUTPUT.mkdir(parents=True,exist_ok=False)
    h.write(OUTPUT/'RESULT.json',result)
    with (OUTPUT/'RESULT.md').open('x') as f:f.write(render(result))
    pins.again()
    receipt={'status':'complete','candidate_count':3,'paired_split_count':6,'native_receipt_sha256':receipt_sha,'inputs':pins.files,'output_files':{n:sha(OUTPUT/n) for n in ('RESULT.json','RESULT.md')},'all_inputs_unchanged_after':True,'historical_CNN_inference':False,'CNN_prediction_rows_loaded':False,'held_out_test_read':False,'inference_count':0,'remote_count':0,'additive_closure_count':len(summary['additive_closure'])}
    h.write(OUTPUT/'RECEIPT.json',receipt)
    return {'status':'complete','output':str(OUTPUT),'receipt_sha256':sha(OUTPUT/'RECEIPT.json'),'outputs':receipt['output_files']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipt-sha',required=True);a=p.parse_args();print(json.dumps(run(a.receipt_sha),ensure_ascii=False))
