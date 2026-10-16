"""Merge fixed native5090 Test8 with native5080 Test4; new inference0."""
import datetime,hashlib,json,math,statistics,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from paper.scripts import evaluate_titantpp_components_inter8_frozen as eval8
from paper.scripts import evaluate_titantpp_components_inter62_frozen as eval4
B8=PROJECT/'search_artifacts/titantpp_components_inter8_test_20261014_v1'
B4=PROJECT/'search_artifacts/titantpp_components_inter62_test_20261014_v1'
R=PROJECT/'reports/titantpp_components_inter62_test_20261014_v1'
LABELS={'titantpp_cnn_gru54':'FULL','titantpp_cnn_gru54_correction_disabled':'NONE','titantpp_cnn_gru54_cnn_only':'CNN','titantpp_cnn_gru54_gru_only':'GRU'}
METRICS=(('전체 RMSE',('qty_rmse',)),('전체 MAE',('qty_mae',)),('큰 수량 RMSE',('tail','qty_rmse')),('큰 수량 MAE',('tail','qty_mae')),('전체 TimeNLL',('time_nll',)),('큰 수량 TimeNLL',('tail','time_nll')))
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def value(m,path):
 for k in path:m=m[k]
 require(type(m) in (int,float) and math.isfinite(m),'Nonfinite metric');return float(m)
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def verify_bundle(b,evaluator,expected,host):
 c=evaluator.verify_contract(b);require(len(c['jobs'])==expected and set(c['runtimes'])=={host},'Contract scope/host differs')
 manifest=read(b/'evaluation_results_manifest.json');recovery=read(b/'evaluation_results_recovery_receipt.json')
 require(recovery['status']=='complete' and sha(b/'evaluation_results_archive.tar.gz')==recovery['archive_sha256'] and sha(b/'evaluation_results_manifest.json')==recovery['manifest_sha256'],'Recovery archive/manifest differs')
 for name,d in manifest['files'].items():require(sha(b/name)==d,'Result bytes changed: '+name)
 actual=read(b/'pipeline_actual_exit.json');require(actual['actual_child_exit_observed'] is True and type(actual['exit_code']) is int and actual['exit_code']==0,'Actual evaluation pipeline exit0 missing')
 evaluator.verify_global_validation_gate(b,c)
 data=c['datasets']['intermittent_frozen_5000'];rows=[]
 for j in c['jobs']:
  require(j['evaluation_host']==j['training_host']==host,'Borrowed Runtime/model host')
  require(sha(b/j['checkpoint']['relative_path'])==j['checkpoint']['sha256'],'Selected checkpoint changed')
  for split in ('validation','test'):
   folder=b/'results'/j['id'];p=folder/('validation_replay.json' if split=='validation' else 'test_metrics.json');o=read(p);x=read(folder/(split+'_worker_exit.json'))
   require(x['actual_child_exit_observed'] is True and type(x['exit_code']) is int and x['exit_code']==0 and x['job_id']==j['id'] and x['contract_sha256']==c['contract_sha256'],'Worker actual exit0 differs')
   require(o['status']=='complete' and o['id']==o['job_id']==j['id'] and o['split']==split and o['contract_sha256']==c['contract_sha256'] and o['selected_epoch']==j['selected_epoch'] and o['checkpoint_sha256']==j['checkpoint']['sha256'] and o['state_sha256']==j['checkpoint_state_sha256'],'Selected model/result identity differs')
   require(o['retrained'] is False and o['reselected'] is False and o['raw_predictions_written'] is False and o['native_runtime_checked'] is True and o['test_independent'] is False,'Unapproved inference semantics')
   require(set(o['qualification'])==evaluator.QUALIFICATION_KEYS and all(v is True for v in o['qualification'].values()),'Causal/parameter qualification failed')
   require(o['runtime']==o['runtime_after'],'Runtime changed');evaluator.check_runtime(o['runtime'],c['runtimes'][host],'cuda',current_process=False)
   pop=data[split+'_population'];require(o['metrics']['count']==o['prediction_rows']==pop['count'],'Target population count changed')
   for k in ('target_identity_sha256','truth_sha256','time_sha256'):require(o[k]==pop[k],'Target population digest changed')
   if split=='validation':evaluator.verify_validation_receipt(o,c,j)
   else:
    require(o['held_out_test_evaluated'] is True and o['validation_gate_sha256']==sha(b/'validation_gate.json') and o['test_started_unix']>=read(b/'validation_gate.json')['created_unix'],'Test started before Validation gate')
   rows.append({'split':split,'seed':j['seed'],'arm':LABELS[j['arm']],'id':j['id'],'host':host,'selected_epoch':j['selected_epoch'],'checkpoint_sha256':j['checkpoint']['sha256'],'state_sha256':j['checkpoint_state_sha256'],'contract_sha256':c['contract_sha256'],'receipt_sha256':sha(p),'metrics':{n:value(o['metrics'],path) for n,path in METRICS},'tail_count':o['metrics']['tail']['count'],'simple_quantity_baselines':o['simple_quantity_baselines'],'elapsed_seconds':o['elapsed_seconds'],'inference_seconds':o['inference_seconds'],'parameter_counts':o['physical_parameter_counts']})
 return c,rows

def summarize(rows):
 expected={(seed,arm) for seed in (42,52,62) for arm in ('NONE','CNN','GRU','FULL')}
 lookup={}
 for row in rows:
  key=row['split'],row['seed'],row['arm'];require(key not in lookup,'Duplicate evaluation ID');lookup[key]=row
 for split in ('validation','test'):require({(seed,arm) for (s,seed,arm) in lookup if s==split}==expected,'Missing/foreign three-seed condition')
 summary={split:{arm:{name:{'mean':statistics.mean(lookup[split,seed,arm]['metrics'][name] for seed in (42,52,62)),'sample_sd':statistics.stdev(lookup[split,seed,arm]['metrics'][name] for seed in (42,52,62)),'n':3} for name,path in METRICS} for arm in ('NONE','CNN','GRU','FULL')} for split in ('validation','test')}
 paired=[{'split':split,'seed':seed,'comparison':arm,'FULL_minus_comparator':{n:lookup[split,seed,'FULL']['metrics'][n]-lookup[split,seed,arm]['metrics'][n] for n,path in METRICS},'FULL_RMSE_improvement_percent':100*(lookup[split,seed,arm]['metrics']['전체 RMSE']-lookup[split,seed,'FULL']['metrics']['전체 RMSE'])/lookup[split,seed,arm]['metrics']['전체 RMSE']} for split in ('validation','test') for seed in (42,52,62) for arm in ('NONE','CNN','GRU')]
 return summary,paired

def main():
 c8,r8=verify_bundle(B8,eval8,8,'5090');c4,r4=verify_bundle(B4,eval4,4,'5080')
 a,b=c8['datasets']['intermittent_frozen_5000'],c4['datasets']['intermittent_frozen_5000']
 for k in ('data_sha256','split_manifest_sha256','tail_threshold','definition','validation_population','test_population'):require(a[k]==b[k],'Cross-seed fixed input semantics differ: '+k)
 rows=r8+r4;summary,paired=summarize(rows)
 baselines={}
 for split in ('validation','test'):
  items=[r['simple_quantity_baselines'] for r in rows if r['split']==split];require(all(x==items[0] for x in items),'Same-input simple baseline differs');baselines[split]=items[0]
 out={'schema':'titantpp_components_Intermittent_three_seed_final_v1','status':'complete','test_job_count':12,'seed_count':3,'seeds':[42,52,62],'new_training':False,'prior_Test8_reexecuted':False,'test_independent':False,'rows':rows,'summary_mean_sample_sd':summary,'seed_paired_differences':paired,'simple_quantity_baselines':baselines,'cross_seed_Runtime':{'42/52':'5090 / cuDNN92000','62':'5080 / cuDNN91900'},'runtime_timing_pooled':False,'recorded_unix':time.time(),'original36_other_datasets_Test_completed_claimed':False}
 write(R/'THREE_SEED_ANALYSIS.json',out)
 fmt=lambda x:f'{x:.6g}'
 lines=['# Intermittent 네 구조의 3seed 최종 비교','', 'Material Passport: 승인된 원12개 구성요소 조건; 고정 Validation 선택 checkpoint, 기존 chronological Test. seed42·52는5090/cuDNN92000,seed62는5080/cuDNN91900이다. cross-seed GPU/Runtime 차이가 있으며 시간은 host별로 분리한다. 새 학습0, 기존Test8 재실행0, checkpoint 재선택0. 새 독립 평가·통계 유의성·외부 신경망 우월성을 주장하지 않는다.','']
 for split in ('validation','test'):
  lines+=['## '+split+' 평균 ± 표본SD (n=3)','', '| 구조 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | 전체 TimeNLL | 큰 수량 TimeNLL |','|---|---:|---:|---:|---:|---:|---:|']
  for arm in ('NONE','CNN','GRU','FULL'):lines.append('| '+arm+' | '+' | '.join(fmt(summary[split][arm][n]['mean'])+' ± '+fmt(summary[split][arm][n]['sample_sd']) for n,p in METRICS)+' |')
  lines+=['', '| 기준선 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL |','|---|---:|---:|---:|---:|---:|']
  for key,label in [('last','직전 수량 반복'),('mean','관측 이력 평균')]:lines.append('| '+label+' | '+' | '.join(fmt(value(baselines[split][key],p)) for n,p in METRICS[:4])+' | N/A |')
  lines+=['']
 lines+=['## Seed별 수량 판정','', '| split | seed | 구조 | 선택 epoch | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE |','|---|---:|---|---:|---:|---:|---:|---:|']
 for row in rows:lines.append('| '+row['split']+' | '+str(row['seed'])+' | '+row['arm']+' | '+str(row['selected_epoch'])+' | '+' | '.join(fmt(row['metrics'][n]) for n,p in METRICS[:4])+' |')
 lines+=['','## FULL의 전체 수량 RMSE 비교','']
 for arm in ('NONE','CNN','GRU'):
  comp=summary['test'][arm]['전체 RMSE']['mean'];full=summary['test']['FULL']['전체 RMSE']['mean'];change=100*(full-comp)/comp;wins=sum(x['FULL_minus_comparator']['전체 RMSE']<0 for x in paired if x['split']=='test' and x['comparison']==arm)
  lines.append(f'- {arm} 대비 평균 변화 {change:+.3f}%; 음수는 FULL 개선. seed별 strict 개선 {wins}/3개.')
 best=min(summary['test'],key=lambda arm:summary['test'][arm]['전체 RMSE']['mean']);lines+=['',f'Test 평균 전체 raw 수량 RMSE가 가장 낮은 구조는 {best}다. 평균 우세와 모든seed의 일관된 우세는 구분한다. 큰 수량은 고정 Train 기준 q>187이며 모든 보조 지표와 불리한 조건을 표에서 보존한다.','', '## 남은 작업','', '**다른 데이터셋의 최종 Test와 주장을 맞춘다 — 다음 작업**','- Taxi·RAF의 최종 결과와 외부 비교군까지 포함한 범위에서 논문 기여를 판정한다. 이번 보고서는 Intermittent 원12개만 완료했다. 현재 main.tex는 변경하지 않았다.']
 with (R/'THREE_SEED_ANALYSIS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 write(B4/'final_receipt.json',{'status':'complete','test_job_count':12,'seed_count':3,'native_seed62_Test4_actual_exit0':True,'native_seed62_Validation4_PASS':True,'prior_Test8_reexecuted':False,'new_training':False,'test_independent':False,'analysis_sha256':sha(R/'THREE_SEED_ANALYSIS.json'),'report_sha256':sha(R/'THREE_SEED_ANALYSIS.md'),'seed62_archive_sha256':sha(B4/'evaluation_results_archive.tar.gz'),'seed62_manifest_sha256':sha(B4/'evaluation_results_manifest.json'),'completed_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()})
 print(json.dumps({'status':'complete','test_job_count':12,'seed_count':3,'report_path':str(R/'THREE_SEED_ANALYSIS.md')},ensure_ascii=False))

if __name__=='__main__':main()
