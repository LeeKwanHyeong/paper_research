import csv, json, math, statistics, hashlib
from pathlib import Path
from collections import defaultdict
R=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent
sources={}
def load(p):
 p=R/p;sources[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def dump(p,d): (O/p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
base_path='reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv'
p=R/base_path;sources[base_path]=hashlib.sha256(p.read_bytes()).hexdigest();assert sources[base_path]=='3501b98f0f3c5d250c6ddb7f8309b3455ec1bb908f0844928ec504d71c826765'
metrics=['qty_mae','qty_rmse','time_nll'];rows=[]
for r in csv.DictReader(p.open()):
 if r['endpoint']!='selected':continue
 rows.append(dict(dataset=r['dataset'],model=r['arm'],seed=int(r['seed']),epoch=int(r['epoch']),count=int(r['count']),**{m:float(r[m]) for m in metrics},campaign=r['campaign'],status='completed_selected_validation',audit='existing_final_core_audit',source=base_path))
additional_path='search_artifacts/titantpp_additional_tpp_20260930_v1/hourly_comparison/20260930T220247154140Z/comparison.json';a=load(additional_path)
for r in a['rows']:
 if r['status']!='completed_replay_sha_verified':continue
 rows.append(dict(dataset=r['dataset'],model=r['arm'],seed=r['seed'],epoch=r['best_epoch'],count=r['baseline']['count'],**r['metrics'],campaign='additional_tpp',status=r['status'],audit=r['terminal_checkpoint_binary_audit'],source=additional_path))
norm_path='reports/titantpp_active_branch_norm_runpod4090_20260930_v1/comparison.json';n=load(norm_path)
for r in n['conditions']:
 if not r['scientific_success']:continue
 s=r['endpoint_replays']['selected'];rows.append(dict(dataset=r['job']['dataset'],model=r['job']['arm'],seed=42,epoch=r['best_recorded_epoch'],count=s['count'],**{m:s[m] for m in metrics},campaign='normalization_runpod4090',status='completed_selected_validation',audit='passed_cpu_terminal_audit',source=norm_path))
inst_path='search_artifacts/titantpp_instacart_norm_5080_20260930_v1/monitor/20260930T220711276848Z/snapshot.json';ins=load(inst_path);ir=ins['remote'];j=next(iter(ir['jobs'].values()));assert ir['status.json']['status']=='complete' and j['terminal_manifest.json']['scientific_success'] and j['endpoint_replays.json']['status']=='complete'
s=j['endpoint_replays.json']['selected'];rows.append(dict(dataset='insta_market_basket',model='titantpp_history_mlp_active_norm',seed=42,epoch=j['summary.json']['best_epoch'],count=s['count'],**{m:s[m] for m in metrics},campaign='normalization_4090_to_5080_recovery',status='completed_selected_validation',audit='server_manifest_recorded_local_checkpoint_cpu_audit_pending',source=inst_path))
gate_path='reports/titantpp_mlp_gate_execution_20260929_v1/comparison.json';g=load(gate_path)
for r in g['MLP_Gate_new_conditions']+g['Full_Gate_completed_separate']:
 s=r['endpoint_replays']['selected'];rows.append(dict(dataset=r['dataset'],model=r['arm'],seed=42,epoch=r['best_epoch'],count=s['count'],**{m:s[m] for m in metrics},campaign='gate_seed42',status='completed_selected_validation',audit='passed_cpu_terminal_audit',source=gate_path))
assert len({(r['dataset'],r['model'],r['seed']) for r in rows})==len(rows)
assert all(math.isfinite(r[m]) for r in rows for m in metrics)
expected_count={'yellow_trip_hourly':8268,'intermittent_frozen_5000':86285,'insta_market_basket':503733}
assert all(r['count']==expected_count[r['dataset']] for r in rows)
lookup={(r['dataset'],r['model'],r['seed']):r for r in rows};groups=defaultdict(list)
for r in rows:groups[(r['dataset'],r['model'])].append(r)
stats=[]
for (d,m),rr in groups.items():
 if sorted(r['seed'] for r in rr)!=[42,52,62]:continue
 stats.append(dict(dataset=d,model=m,n=3,**{v:dict(mean=statistics.mean(r[v] for r in rr),sample_sd=statistics.stdev(r[v] for r in rr)) for v in metrics}))
external=['rmtpp','thp','nhp','sahp','s2p2_matched_head','attnhp_matched_head'];plain='titantpp_history_mlp';norm='titantpp_history_mlp_active_norm';sl={(r['dataset'],r['model']):r for r in stats}
paired=[]
for d in expected_count:
 for m in external:
  rr=[(lookup[d,plain,seed],lookup[d,m,seed]) for seed in [42,52,62] if (d,m,seed) in lookup]
  paired.append(dict(dataset=d,comparator=m,completed_seed_count=len(rr),seeds=[x['seed'] for x,y in rr],**{v:dict(plain_wins=sum(x[v]<y[v] for x,y in rr),ties=sum(x[v]==y[v] for x,y in rr),comparator_wins=sum(x[v]>y[v] for x,y in rr),reduction_percent_of_3seed_mean=(100*(sl[d,m][v]['mean']-sl[d,plain][v]['mean'])/sl[d,m][v]['mean'] if (d,m) in sl else None)) for v in metrics}))
labels={'yellow_trip_hourly':'Taxi','intermittent_frozen_5000':'Intermittent','insta_market_basket':'Instacart',plain:'TitanTPP MLP',norm:'활성 분기 정규화', 'rmtpp':'RMTPP','thp':'THP','nhp':'NHP','sahp':'SAHP','s2p2_matched_head':'S2P2 (공통 head)','attnhp_matched_head':'AttNHP (공통 head)','titantpp':'B: 이력 보완 없음','titantpp_local_detail':'Full: 수준·변화 분리','titantpp_level_only':'수준만','titantpp_change_only':'변화만','titantpp_no_static_lmm':'Full 정적 검색 제거','titantpp_mlp_conditional':'MLP 조건부 Gate','titantpp_mlp_constant':'MLP 상수 Gate','titantpp_change_conditional':'Full 조건부 Gate','titantpp_change_constant':'Full 상수 Gate'}
lines=['# TitanTPP 외부 비교: 두 데이터의 수량 우위와 Instacart의 한계','', '증거 기준: 2026-10-01 07:07 KST. 추가 TPP는 07:02:47 KST 관측을 사용한다. 현재 실행 중인 조건을 완료 결과에 포함하지 않는다.','', '## 결론','', '**대표 모델은 기존 TitanTPP History MLP를 유지한다.** Taxi·Intermittent에서는 공통 head로 비교한 외부 TPP 6개 모두보다 3seed 평균 MAE·RMSE가 낮으며, 같은 seed별 비교 36쌍 모두에서 두 수량 지표가 낮다. 시간 NLL까지 일관되게 우월하지는 않다. Instacart에서는 최상위 외부 모델 대비 수량 우위를 확보하지 못했다.','', '활성 분기 정규화는 seed42에서 Taxi의 수량 지표를 개선했으나 시간 NLL은 악화됐고, Intermittent는 세 지표 모두 악화됐다. Instacart의 기존 MLP 대비 개선은 MAE 0.025%, RMSE 0.120%로 작으며 모든 외부 모델을 앞서지는 못한다. 대표 구조를 정규화로 교체할 근거는 부족하다.','', '## 동일 조건의 3seed 전체 비교','', '아래 값은 seed42·52·62 평균 ± 표본표준편차(ddof=1). 모든 지표는 낮을수록 좋다. MAE·시간 NLL도 RMSE로 선택한 동일 checkpoint 값이다. 서로 다른 데이터의 절대 오차나 시간 NLL을 직접 비교하지 않는다.']
for d in expected_count:
 lines+=['',f'### {labels[d]}','', '| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |','|---|---:|---:|---:|']
 for m in [plain]+external:
  if (d,m) not in sl:continue
  q=sl[d,m];lines.append('| '+labels[m]+' | '+' | '.join(f"{q[v]['mean']:.6f} ± {q[v]['sample_sd']:.6f}" for v in metrics)+' |')
 if d=='insta_market_basket':lines+=['','S2P2·AttNHP의 Instacart 3seed는 미완료여서 이 평균표에 포함하지 않는다. 아래 seed42 표에서만 확정된 동일 seed 결과를 비교한다.']
lines+=['','## 외부 비교군 대비 대표 MLP의 개선율과 seed별 승패','','개선율 = (비교군 평균 − MLP 평균) / 비교군 평균 × 100. 음수는 MLP 악화. 승수는 완료된 동일 seed 비교만 센다.','', '| 데이터 | 비교군 | 완료 seed | MAE 개선율 | RMSE 개선율 | MAE 승 | RMSE 승 | 시간 NLL 승 |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in paired:
 nseed=r['completed_seed_count'];fmt=lambda m:'3seed 미완료' if r[m]['reduction_percent_of_3seed_mean'] is None else f"{r[m]['reduction_percent_of_3seed_mean']:.2f}%"
 lines.append(f"| {labels[r['dataset']]} | {labels[r['comparator']]} | {nseed} | {fmt('qty_mae')} | {fmt('qty_rmse')} | {r['qty_mae']['plain_wins']}/{nseed} | {r['qty_rmse']['plain_wins']}/{nseed} | {r['time_nll']['plain_wins']}/{nseed} |")
lines+=['','## 활성 분기 정규화와 외부 모델: seed42만 비교','','Taxi·Intermittent 정규화는 RTX4090, Instacart 정규화는 RTX4090 2epoch 이후 개인 RTX5080으로 승인된 복구를 수행했다. 하드웨어가 다르고 한 seed뿐이므로 일반적인 우월성이나 속도 차이로 해석하지 않는다.']
for d in expected_count:
 lines+=['',f'### {labels[d]} — seed42','', '| 모델 | 선택 epoch | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |','|---|---:|---:|---:|---:|']
 for m in [plain,norm]+external:
  r=lookup[d,m,42];lines.append(f"| {labels[m]} | {r['epoch']} | "+' | '.join(f'{r[v]:.6f}' for v in metrics)+' |')
 x=lookup[d,plain,42];y=lookup[d,norm,42];lines+=['', '정규화의 기존 MLP 대비 수량 개선율: '+', '.join(f'{v} {100*(x[v]-y[v])/x[v]:.4f}%' for v in metrics[:2])+f". 시간 NLL 변화(정규화−MLP): {y['time_nll']-x['time_nll']:+.6f}."]
lines+=['','## 내부 구조 비교와 Gate 탐색 — 외부 비교와 분리','','기존의 내부 대안과 불리한 탐색 결과도 보존한다. Full이 Taxi 평균 수량 지표에서 MLP보다 조금 좋았다는 사실은 유지하며, MLP 채택을 모든 내부 대안 대비 수치상 1위라는 주장으로 바꾸지 않는다.','', '| 데이터 | 구조 | seed 수 | MAE 평균 | RMSE 평균 | 시간 NLL 평균 |','|---|---|---:|---:|---:|---:|']
for r in stats:
 if r['model'] not in external: lines.append(f"| {labels[r['dataset']]} | {labels[r['model']]} | 3 | "+' | '.join(f"{r[v]['mean']:.6f}" for v in metrics)+' |')
lines+=['','Gate 6완료조건은 seed42 탐색이며 위 3seed 표에 합산하지 않는다. Full Taxi C/D는 미시작 보류다.','', '| 데이터 | 변형 | 선택 epoch | MAE | RMSE | 시간 NLL |','|---|---|---:|---:|---:|---:|']
for r in rows:
 if r['campaign']=='gate_seed42':lines.append(f"| {labels[r['dataset']]} | {labels[r['model']]} | {r['epoch']} | "+' | '.join(f'{r[v]:.6f}' for v in metrics)+' |')
lines+=['','## 해석과 다음 실험','','- 논문에서 방어 가능한 현재 주장: 동일한 관측 정보·공통 예측 head·RMSE 선택 규칙 아래, TitanTPP MLP는 Taxi와 Intermittent의 validation 다음 사건 수량 예측에서 외부 6개 TPP 비교군보다 반복적으로 낮은 MAE·RMSE를 보였다.','- Instacart에서 모든 비교군보다 우월하다는 주장은 지지되지 않는다. MAE와 RMSE의 비교 순위가 다르다. 정규화의 작은 수치 개선은 통계적 유의성이나 동등성 검증이 아니다.','- 시간 NLL 악화는 제외하지 않는다. 수량 예측 개선과 발생 간격 분포 적합도의 상충을 보고한다.','- 비교군은 공통 head·loss·입력으로 연결한 통제 비교다. 원 논문의 모든 native head 및 각 모델별 최적 튜닝 결과를 이겼다는 주장이 아니다.','- 다음 RAF 24조건은 대표 MLP, 정규화 MLP, 외부 6개를 같은 새로운 데이터에서 비교한다. 결과가 불리해도 모델·seed를 제외하지 않는다.','', '## 검증 범위와 남은 감사','','모든 표는 validation-only이다. held-out/test 성능을 읽거나 추가 평가하지 않았다. 선택 규칙은 최초 최소 raw 수량 RMSE이며 MAE 최저 epoch를 별도로 고르지 않았다.','', '기존 core·외부4종 최종 감사와 Gate/RunPod 정규화 CPU 감사는 기존 증적을 재사용한다. 추가 TPP는 terminal/replay 소형 기록 SHA 검증 완료 결과이며 checkpoint binary CPU 감사는 아직 미완료다. 방금 완료한 5080 Instacart 정규화는 성공 manifest와 selected/last 재평가를 확인했으나 로컬 binary 회수·CPU 감사가 남았다. 이 비교표 작성이 그 감사를 대체하지 않는다.','', '추가 TPP Instacart는 관측시점 S2P2 seed42·52, AttNHP seed42가 완료, AttNHP seed52가 진행 중, seed62 두 조건은 미시작이다. 진행 중 best는 이 완료 성능표에서 제외했다.','', '개인5080 추가 cloud 임대료는 0이고 전기료는 미측정이다. RunPod 정규화의 계정잔액 감소 $3.036663은 Pod별 최종 청구서가 아니다. 이종 GPU·복구·다른 종료 epoch의 학습 시간을 단독 속도 우위로 비교하지 않는다.','', '## 원본 및 재현','','`selected_conditions.csv`는 조건별 원본 수치, `comparison.json`은 집계·승패, `verification.json`은 검증 및 파일 SHA를 담는다. `build_analysis.py`로 저장된 원본에서 재생성할 수 있다.','']
for f,h in sources.items():lines.append(f'- [{f}]({R/f}) — SHA256 `{h}`')
(O/'report.md').write_text('\n'.join(lines)+'\n')
fields=list(rows[0]);
with (O/'selected_conditions.csv').open('w') as f:
 w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
dump('comparison.json',dict(as_of_kst='2026-10-01T07:07:15+09:00',evaluation_scope='validation_only',representative=plain,rows=rows,three_seed=stats,paired=paired,additional_observation=a['observed_kst'],additional_pending=[{k:r.get(k) for k in ['dataset','arm','seed','status','completed_epochs','best_epoch']} for r in a['rows'] if r['status']!='completed_replay_sha_verified']))
wins=[r for r in paired if r['dataset']!='insta_market_basket'];assert sum(r['qty_mae']['plain_wins'] for r in wins)==36 and sum(r['qty_rmse']['plain_wins'] for r in wins)==36
verification=dict(passed=True,unique_completed_conditions=len(rows),complete_three_seed_groups=len(stats),same_seed_pairs_taxi_intermittent=36,mae_wins=36,rmse_wins=36,selection='strict_first_min_validation_raw_qty_rmse',sample_sd_ddof=1,validation_counts=expected_count,finite_metrics=True,source_files=sources,new_training_or_replay_during_analysis=False,additional_binary_cpu_audit='pending',instacart_norm_local_binary_cpu_audit='pending')
dump('verification.json',verification)
# Reviewed snapshot for the local interactive report; exact numeric values remain inspectable.
queries=[]
for d in expected_count:
 rr=[]
 for m in [plain]+external:
  if (d,m) in sl:
   t=sl[d,m];rr.append(dict(model=labels[m],seed_count=3,**{v:t[v]['mean'] for v in metrics},**{v+'_sd':t[v]['sample_sd'] for v in metrics}))
 queries.append(dict(id=d,rows=rr,source={'name':labels[d]+' completed validation', 'files':list(sources),'description':'同一 validation RMSE checkpoint; 3 seeds 42/52/62; mean and sample SD','caveats':['Validation only; additional comparator checkpoint CPU audit pending.'],'metricDefinitions':[{'label':v,'definition':'Three-seed arithmetic mean at each seed minimum raw validation RMSE checkpoint; lower is better','componentIds':[d+'-table']} for v in metrics]}))
snapshot=dict(title='TitanTPP는 두 데이터에서 수량 예측 우위를 유지한다',buildStatus='complete',report={'title':'TitanTPP는 두 데이터에서 수량 예측 우위를 유지한다','asOf':'2026-10-01 07:07 KST'},queries={q['id']:q for q in queries})
dump('reviewed_snapshot.json',snapshot)
print(json.dumps(verification,ensure_ascii=False));print('report',O/'report.md')
