import hashlib,json,math,statistics,time
from pathlib import Path
b=Path('search_artifacts/titantpp_components_inter8_test_20261014_v1');r=Path('reports/titantpp_components_inter8_test_20261014_v1');c=json.loads((b/'contract.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((b/'evaluation_results_manifest.json').read_text())
for n,d in manifest['files'].items():assert sha(b/n)==d
assert manifest['actual_native_worker_exit0_count']==16 and manifest['actual_pipeline_exit']==0
actual=json.loads((b/'pipeline_actual_exit.json').read_text());assert actual['exit_code']==0 and actual['actual_child_exit_observed'] is True
labels={'titantpp_cnn_gru54':'FULL','titantpp_cnn_gru54_cnn_only':'CNN','titantpp_cnn_gru54_gru_only':'GRU','titantpp_cnn_gru54_correction_disabled':'NONE'}
keys=[('전체 RMSE',('qty_rmse',)),('전체 MAE',('qty_mae',)),('큰 수량 RMSE',('tail','qty_rmse')),('큰 수량 MAE',('tail','qty_mae')),('전체 TimeNLL',('time_nll',)),('큰 수량 TimeNLL',('tail','time_nll'))]
def val(o,p):
 for k in p:o=o[k]
 assert math.isfinite(o);return o
records={};baselines={};rows=[]
for split in ('validation','test'):
 base=None
 for j in c['jobs']:
  folder=b/'results'/j['id'];o=json.loads((folder/('validation_replay.json' if split=='validation' else 'test_metrics.json')).read_text());e=json.loads((folder/(split+'_worker_exit.json')).read_text())
  assert e['exit_code']==0 and e['actual_child_exit_observed'] is True
  assert o['contract_sha256']==c['contract_sha256'] and o['job_id']==j['id'] and o['selected_epoch']==j['selected_epoch'] and o['checkpoint_sha256']==j['checkpoint']['sha256'] and o['state_sha256']==j['checkpoint_state_sha256']
  assert o['retrained'] is False and o['reselected'] is False and o['qualification']['parameters_unchanged'] is True and o['runtime']==o['runtime_after'] and all(o['qualification'].values())
  pop=c['datasets'][j['dataset']][split+'_population'];assert o['metrics']['count']==o['prediction_rows']==pop['count']
  for k in ('target_identity_sha256','truth_sha256','time_sha256'):assert o[k]==pop[k]
  if split=='validation':assert o['validation_match']=='PASS'
  else:assert o['test_independent'] is False and o['held_out_test_evaluated'] is True
  metrics={name:val(o['metrics'],path) for name,path in keys};arm=labels[j['arm']]
  records[split,j['seed'],arm]=metrics
  rows.append({'split':split,'seed':j['seed'],'arm':arm,'selected_epoch':j['selected_epoch'],'count':pop['count'],'tail_count':o['metrics']['tail']['count'],'metrics':metrics,'receipt_raw_sha256':sha(folder/('validation_replay.json' if split=='validation' else 'test_metrics.json')),'physical_parameter_counts':o['physical_parameter_counts']})
  simple=o['simple_quantity_baselines']
  if base is None:base=simple
  else:assert simple==base,'Simple baselines differ across fixed same population'
 baselines[split]=base
summary={};paired=[]
for split in ('validation','test'):
 summary[split]={}
 for arm in ('NONE','CNN','GRU','FULL'):
  summary[split][arm]={name:{'mean':statistics.mean(records[split,seed,arm][name] for seed in (42,52)),'sample_sd':statistics.stdev(records[split,seed,arm][name] for seed in (42,52)),'n':2} for name,path in keys}
 for seed in (42,52):
  for arm in ('NONE','CNN','GRU'):
   a=records[split,seed,arm];f=records[split,seed,'FULL']
   paired.append({'split':split,'seed':seed,'comparison':arm,'FULL_minus_comparator':{k:f[k]-a[k] for k in a},'FULL_RMSE_improvement_percent':100*(a['전체 RMSE']-f['전체 RMSE'])/a['전체 RMSE']})
out={'schema':'titantpp_inter8_test_partial_analysis_v1','status':'complete','contract_sha256':c['contract_sha256'],'split_scope':['validation','test'],'seed_count':2,'seeds':[42,52],'job_count':8,'test_independent':False,'new_training':False,'all8Validation_native_replay_PASS':True,'all16_actual_worker_exit0':True,'actual_pipeline_exit0':True,'result_archive_sha256':sha(b/'evaluation_results_archive.tar.gz'),'result_manifest_sha256':sha(b/'evaluation_results_manifest.json'),'rows':rows,'summary_mean_sample_sd':summary,'seed_paired_differences':paired,'simple_quantity_baselines':baselines,'recorded_unix':time.time()}
with (r/'analysis.json').open('x') as f:json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
def fmt(x):return f'{x:.6g}'
lines=['# Intermittent42·52 네 구조의 Test 부분 비교','', '같은5090에서 학습한 고유8개 선택 모델을 평가했다. 새 학습0, checkpoint 재선택0이다. 원36 중 seed62는 제외한 **2seed 잠정 분석**이며 기존 Test의 접근 이력을 보존한다. 새 독립 평가·3seed 최종 결과·통계 유의성으로 확대하지 않는다.','', '## Validation 재현과 실행 증적','', 'Validation8개 모두 동일 선택 binary에서 원 여섯 지표와 대상 지문을 재현했다. 이후 별도 gate8을 통과해 Test8개를 실행했다. 실제 native worker16개와 평가 pipeline의 exit0, 평가 전후 tensor·input·source SHA 불변을 확인했다. 원본120개 및 result archive/manifest의 모든 로컬 파일 SHA를 검증했다.','']
for split,title in [('validation','Validation'),('test','Test')]:
 lines += ['## '+title+' 평균 ± 표본SD (seed42·52)','', '| 구조 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | 전체 TimeNLL | 큰 수량 TimeNLL |','|---|---:|---:|---:|---:|---:|---:|']
 for arm in ('NONE','CNN','GRU','FULL'):
  lines.append('| '+arm+' | '+' | '.join(fmt(summary[split][arm][name]['mean'])+' ± '+fmt(summary[split][arm][name]['sample_sd']) for name,_ in keys)+' |')
 lines += ['', '| 기준선 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL |','|---|---:|---:|---:|---:|---:|']
 for name,label in [('last','직전 수량 반복'),('mean','관측 이력 평균')]:
  m=baselines[split][name];lines.append('| '+label+' | '+' | '.join(fmt(val(m,p)) for _,p in keys[:4])+' | N/A |')
 lines += ['', '큰 수량은 기존 Train 기준 q>187이다. 비학습 기준선은 같은 대상의 결정적 예측 하나이며 seed SD를 부여하지 않았다.','']
lines += ['## Seed별 수량과 선택 epoch','', '| split | seed | 구조 | 선택 epoch | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE |','|---|---:|---|---:|---:|---:|---:|---:|']
for row in rows:
 lines.append('| '+row['split']+' | '+str(row['seed'])+' | '+row['arm']+' | '+str(row['selected_epoch'])+' | '+' | '.join(fmt(row['metrics'][name]) for name,_ in keys[:4])+' |')
lines += ['', '## FULL 대비 seed별 개선률','', '양수는 FULL의 전체 raw 수량 RMSE 개선, 음수는 악화다.','', '| split | seed | NONE 대비 | CNN 대비 | GRU 대비 |','|---|---:|---:|---:|---:|']
for split in ('validation','test'):
 for seed in (42,52):
  vals=[next(p['FULL_RMSE_improvement_percent'] for p in paired if p['split']==split and p['seed']==seed and p['comparison']==arm) for arm in ('NONE','CNN','GRU')]
  lines.append('| '+split+' | '+str(seed)+' | '+' | '.join(f'{x:+.3f}%' for x in vals)+' |')
lines += ['', '## 해석의 범위','', '이 비교는 고정된 Titantpp 경로에서 CNN·GRU 구성요소의 기여를 점검한다. 등록 파라미터와 활성 파라미터 수 차이는 analysis.json의 실제 계측을 함께 확인해야 한다. 외부 신경망 비교군의 우월성이나 모든 seed·데이터셋의 일반화는 이8개로 판정하지 않는다. 시간 지표는 보조 결과로 함께 보존한다.','', '## 다음 순서','', '**Intermittent62 학습을 확정한다 — 외부 작업 대기**','- 5080의 FULL62가 종료되면 원36 고유ID의 종료·원본을 확정한다.','', '**3seed 최종 비교를 완성한다 — 다음 작업**','- 승인된 최종 평가 계약에62를 연결하고 동일 선택·지표 규칙으로 비교한다. cross-seed GPU/Runtime 차이를 명시한다.','', '**논문 결과와 주장 범위를 맞춘다 — 다음 작업**','- Taxi·RAF·Intermittent와 기존 비교군의 최종 평가를 함께 검토한 뒤 Results·Methods·Contribution을 정합한다. 현재 main.tex는 변경하지 않았다.','', '원본: `search_artifacts/titantpp_components_inter8_test_20261014_v1/results/<job_id>/{validation_replay,test_metrics}.json`. 전체 실제 SHA 및 seed별 여섯 수치는 analysis.json에 보존한다.']
with (r/'VALIDATION_TEST_ANALYSIS.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps({'summary':summary,'paired':paired,'baselines':baselines},ensure_ascii=False))
