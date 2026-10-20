"""Reaggregate immutable selected-checkpoint records; no model construction or inference."""
from pathlib import Path
import csv,hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
SHARED='titantpp_history_mlp_shared15_v1';ABLATION='titantpp_history_mlp_width16'
EXTERNAL=('rmtpp','thp','nhp','sahp','s2p2_matched_head','attnhp_matched_head');MODELS=(SHARED,*EXTERNAL,ABLATION)
DS=('raf_spare_parts','yellow_trip_hourly','intermittent_frozen_5000');SEEDS=(42,52,62)
BINDINGS={}
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;BINDINGS[str(p.relative_to(ROOT))]=sha(p);return json.loads(p.read_text())
def bind(p,digest):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;require(sha(p)==digest,'Source SHA differs: '+str(p));BINDINGS[str(p.relative_to(ROOT))]=digest

def main():
 gate=load('reports/titantpp_shared_external_extra42_20261020_v1/PAPER_START_GATE.json');require(gate['status']=='PASS' and gate['paper_may_start'],'Final selection not confirmed')
 frozen=load('reports/titantpp_shared_external_extra42_20261020_v1/FROZEN_CONDITIONS.json')
 for p,d in frozen['source_bindings'].items():bind(p,d)
 raw=load('reports/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1/THREE_SEED_VALIDATION.json')
 final=load('search_artifacts/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1/retrieval/final_receipt.json')
 require(final['status']=='complete' and final['new_fit_count']==9 and final['validation_replay_count']==9,'Raw9 not complete')
 for key in ('source_bindings','archive_bindings','report_files'):
  for p,d in final[key].items():bind(p,d)
 test=load('reports/titantpp_shared_raf_taxi6_test_20261020_v1/THREE_SEED_TEST.json')
 intertest=load('reports/titantpp_shared_inter3_interim_test_20261020_v1/THREE_SEED_TEST.json')
 interval=load('reports/titantpp_shared_inter3_interim_test_20261020_v1/THREE_SEED_VALIDATION.json')
 registry=load('reports/titantpp_shared_raf_taxi6_test_20261020_v1/REFERENCES.json')
 spec=load('reports/titantpp_shared_external_extra42_20261020_v1/SCIENCE_SPEC.json')
 records={}
 def put(r,split,source):
  key=(r['dataset'],r['model'],r['seed'],split);require(r['dataset'] in DS and r['model'] in MODELS and r['seed'] in SEEDS,'Unexpected endpoint')
  rec={'dataset':r['dataset'],'model':r['model'],'model_role':'internal_ablation' if r['model']==ABLATION else 'candidate' if r['model']==SHARED else 'external_matched_head','seed':r['seed'],'split':split,'selected_epoch':r['selected_epoch'],'metrics':r['metrics'],'source':source}
  if key in records:
   old=records[key]
   require(old['selected_epoch']==rec['selected_epoch'],'Duplicate epoch mismatch')
   for metric in ('qty_rmse','qty_mae','time_nll'):
    require(math.isclose(old['metrics']['overall'][metric],rec['metrics']['overall'][metric],rel_tol=1e-5,abs_tol=1e-5),'Duplicate native replay metric mismatch')
   return
  records[key]=rec
 for r in raw['records']:put(r,'validation','raw9 final native Validation replay')
 for ds in DS[:2]:
  for r in test['datasets'][ds]['records']:put(r,'test','completed RAF/Taxi native Test and verified stored references')
 for r in intertest['records']:put(r,'test','frozen Inter Test; now final-best identity verified by PAPER_START_GATE')
 for r in interval['records']:put(r,'validation','Inter native Validation and fixed stored references; final-best verified')
 # For RAF/Taxi external Validation, use the SAME selected checkpoint as the stored Test endpoint.
 for ds in DS[:2]:
  for arm in EXTERNAL:
   for seed in SEEDS:
    ref=next(r for r in registry['references'] if (r['dataset'],r['model'],r['seed'])==(ds,arm,seed))
    bind(ref['checkpoint_path'],ref['checkpoint_sha256']);bind(ref['history_path'],ref['history_sha256'])
    path=Path(ref['checkpoint_path']).parent/'summary.json';md=load(path)
    binding=next(x for x in frozen['models'][arm]['historical_summary_bindings'] if x['dataset']==ds and x['seed']==seed)
    require(sha(ROOT/path)==binding['sha256'],'Frozen summary mismatch')
    h=load(ref['history_path']);h=h['history'] if isinstance(h,dict) else h
    finite=[x for x in h if isinstance(x.get('val_qty_rmse'),(int,float)) and math.isfinite(x['val_qty_rmse'])]
    best=min(finite,key=lambda x:x['val_qty_rmse'])
    require(best['epoch']==md['best_epoch']==ref['selected_epoch'],'Validation/Test selection differs')
    require(md['checkpoint_state_sha256']==ref['state_tensor_sha256'],'Summary/Test state differs')
    for name in ('qty_rmse','qty_mae','time_nll'):
     require(math.isclose(md['best_val_'+name],best['val_'+name],rel_tol=1e-10,abs_tol=1e-10),'Same-epoch metric mismatch')
    metrics={'overall':{'count':spec['datasets'][ds]['populations']['validation']['target_count'],**{k:md['best_val_'+k] for k in ('qty_rmse','qty_mae','time_nll')}},'tail':{'count':None,'qty_rmse':None,'qty_mae':None,'time_nll':None,'status':'not_available_in_verified_historical_summary'}}
    put({'dataset':ds,'model':arm,'seed':seed,'selected_epoch':md['best_epoch'],'metrics':metrics},'validation','historical selected checkpoint summary + earliest strict minimum history + fixed Test checkpoint SHA')
 require(len(records)==144,'Expected144 selected model/dataset/seed/split endpoints')
 aggregates=[];deltas=[];bestexternal=[]
 for ds in DS:
  for split in ('validation','test'):
   for model in MODELS:
    rr=[records[(ds,model,s,split)] for s in SEEDS]
    for cohort in ('overall','tail'):
     for metric in ('qty_rmse','qty_mae','time_nll'):
      vv=[r['metrics'].get(cohort,{}).get(metric) for r in rr];available=all(isinstance(v,(int,float)) and math.isfinite(v) for v in vv)
      aggregates.append({'dataset':ds,'split':split,'model':model,'model_role':rr[0]['model_role'],'cohort':cohort,'metric':metric,'seed_count':3,'mean':statistics.mean(vv) if available else None,'sample_sd_ddof1':statistics.stdev(vv) if available else None,'status':'available' if available else 'historical_metric_unavailable'})
   for metric in ('qty_rmse','qty_mae'):
    aa=[x for x in aggregates if x['dataset']==ds and x['split']==split and x['cohort']=='overall' and x['metric']==metric]
    ext=min((x for x in aa if x['model'] in EXTERNAL),key=lambda x:x['mean']);cand=next(x for x in aa if x['model']==SHARED)
    bestexternal.append({'dataset':ds,'split':split,'metric':metric,'external_model':ext['model'],'external_mean':ext['mean'],'shared_mean':cand['mean'],'shared_minus_external':cand['mean']-ext['mean'],'relative_percent':100*(cand['mean']-ext['mean'])/ext['mean'],'shared_lower':cand['mean']<ext['mean']})
   for arm in EXTERNAL:
    for seed in SEEDS:
     a,b=records[(ds,SHARED,seed,split)],records[(ds,arm,seed,split)]
     for cohort in ('overall','tail'):
      for metric in ('qty_rmse','qty_mae','time_nll'):
       av=a['metrics'].get(cohort,{}).get(metric);bv=b['metrics'].get(cohort,{}).get(metric);available=isinstance(av,(int,float)) and isinstance(bv,(int,float)) and math.isfinite(av) and math.isfinite(bv)
       deltas.append({'dataset':ds,'split':split,'seed':seed,'reference_model':arm,'cohort':cohort,'metric':metric,'shared':av,'reference':bv,'absolute_difference':av-bv if available else None,'relative_percent':100*(av-bv)/bv if available and bv!=0 else None,'status':'available' if available else 'historical_metric_unavailable'})
 result={'schema':'shared_final_three_seed_paper_evidence_v1','status':'verified_existing_results_reaggregated','data_variant':'original_uncapped','seeds':list(SEEDS),'models':list(MODELS),'external_comparison_primary':True,'mlp16_role':'internal_ablation_only','intermittent_final_best_verified':True,'paper_start_gate':gate,'endpoint_count':len(records),'records':list(records.values()),'aggregates':aggregates,'seed_differences':deltas,'external_metric_bests':bestexternal,'source_bindings':BINDINGS,'new_training':0,'new_inference':0,'remote_calls':0,'extra_seeds_included':False,'cross_dataset_raw_error_pooled':False,'independent_new_test':False,'limitations':['Existing Test informed exploratory design.','External models use matched common quantity/time heads, not original native intensity objectives.','Historical3 Shared training5090/cuDNN92000; external historical fits5080/cuDNN91900 and some saved CPU replay evaluations.','Additionalseed72/82 are in progress and are not included.','RAF/Taxi external Validation tail metrics absent from verified historical summary are N/A; no zero imputation.','All metrics at first strict finite minimum fullValidation raw quantity RMSE; MAE uses the same checkpoint.','No statistical significance, universal dominance, or unmeasured runtime speedup claim.']}
 target=OUT/'RESULT_TABLES.json';require(not target.exists(),'Preserve existing table artifact')
 with target.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (OUT/'RESULT_TABLES.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(aggregates[0]));writer.writeheader();writer.writerows(aggregates)
 main_deltas=[x for x in deltas if x['cohort']=='overall' and x['metric'] in ('qty_rmse','qty_mae')]
 require(len(main_deltas)==216,'Missing main endpoint seed differences')
 with (OUT/'SEED_DIFFERENCES.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(main_deltas[0]));writer.writeheader();writer.writerows(main_deltas)
 render_markdown(result)
 print(json.dumps({'status':'complete','endpoints':len(records),'aggregate_cells':len(aggregates),'main_seed_differences':len(main_deltas),'all_seed_metric_differences':len(deltas),'source_file_bindings':len(BINDINGS),'new_inference':0,'external_metric_bests':bestexternal},ensure_ascii=False))
def render_markdown(result):
 aggregates=result['aggregates'];bestexternal=result['external_metric_bests']
 lines=['# 확정3seed·원본 데이터 연구용 수치표','','생성형 원고 본문이 아닌 기존 실험 자료의 집계다. 각 값은42/52/62 평균±표본SD(ddof1). Intermittent 최종 checkpoint는 이전 Test 고정본과 epoch/tensor SHA가 동일하다. 추가72/82는 포함하지 않았다.','']
 def fmt(v):return f'{v:.6e}' if v!=0 and abs(v)<1e-6 else f'{v:.6f}'
 def cell(v):return 'N/A' if v['mean'] is None else fmt(v['mean'])+' ± '+fmt(v['sample_sd_ddof1'])
 for ds in DS:
  for split in ('validation','test'):
   lines += [f'## {ds} · {split}','','| 모델 | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰수량 RMSE | 큰수량 MAE | 큰수량 TimeNLL |','|---|---:|---:|---:|---:|---:|---:|']
   for model in MODELS:
    aa=[next(x for x in aggregates if (x['dataset'],x['split'],x['model'],x['cohort'],x['metric'])==(ds,split,model,cohort,metric)) for cohort in ('overall','tail') for metric in ('qty_rmse','qty_mae','time_nll')]
    lines.append('| '+model+(' (내부 대조)' if model==ABLATION else '')+' | '+' | '.join(map(cell,aa))+' |')
   lines.append('')
 lines+=['## 외부6종 지표별 최저 평균과의 차이','','음수는 공유형 오차가 낮다. 지표별 비교군을 하나의 가상 모델로 합치지 않는다.','','| 데이터셋 | split | 지표 | 외부 최저 모델 | 공유형−외부 | 상대차이 |','|---|---|---|---|---:|---:|']
 for x in bestexternal:lines.append(f"| {x['dataset']} | {x['split']} | {x['metric']} | {x['external_model']} | {x['shared_minus_external']:+.6f} | {x['relative_percent']:+.4f}% |")
 lines+=['','모든 seed 차이는 SEED_DIFFERENCES.csv와 RESULT_TABLES.json에 보존했다. 원본3seed 결과 파일은 수정하지 않았다. RAF/Taxi 외부 Validation의 큰수량 지표는 기존 검증 summary에 없어 N/A다. 시간 NLL은 recorded positive-integer TimeNLL이며 원래 비교군 논문의 native intensity NLL이 아니다.','']
 with (OUT/'RESULT_TABLES.md').open('w') as f:f.write('\n'.join(lines))

if __name__=='__main__':
 import sys
 if '--render-only' in sys.argv:render_markdown(json.loads((OUT/'RESULT_TABLES.json').read_text()))
 else:main()
