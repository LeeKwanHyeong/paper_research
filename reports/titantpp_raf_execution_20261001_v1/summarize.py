"""Aggregate the audited RAF validation records; no fitting, prediction or data access."""
import csv,json,hashlib,statistics,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
D=R/'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2';P=D/'original'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(D/'terminal_audit.json');c=read(P/'execution_contract.json');assert a['status']=='passed'
metrics=['qty_mae','qty_rmse','time_nll'];plain='titantpp_history_mlp';norm='titantpp_history_mlp_active_norm'
labels={plain:'TitanTPP MLP',norm:'活性分岐'};labels[norm]='활성 분기 정규화'
labels.update(rmtpp='RMTPP',thp='THP',nhp='NHP',sahp='SAHP',s2p2_matched_head='S2P2 (공통 head)',attnhp_matched_head='AttNHP (공통 head)')
rows=[];roles=[];cells=[];epoch_started=[];epoch_finished=[]
for ar in a['conditions']:
 j=ar['job'];p=P/'run'/j['id']/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"
 e=read(p/'endpoint_replays.json');t=read(p/'epoch_timing.json')
 epoch_started += [x['started_at_unix'] for x in t['epochs']];epoch_finished += [x['finished_at_unix'] for x in t['epochs']]
 for role,epoch in [('selected',ar['selected_epoch']),('last',ar['completed_epochs'])]:
  v=e[role];r=dict(dataset=j['dataset'],model=j['arm'],seed=j['seed'],epoch=epoch,count=v['count'],**{k:v[k] for k in metrics},campaign='raf_5080',status='completed_selected_validation' if role=='selected' else 'completed_last_validation',audit='passed_cpu_terminal_audit',source=str((p/'endpoint_replays.json').relative_to(R)))
  roles.append(dict(r,endpoint=role,state_sha256=v['state_sha256']))
  if role=='selected':rows.append(r)
  for part in ['quantity','history','additional_history']:
   for cell in v[part+'_cells']:cells.append(dict(dataset=j['dataset'],model=j['arm'],seed=j['seed'],endpoint=role,partition=part,**cell))
assert len(rows)==24 and len(roles)==48 and len({(r['model'],r['seed']) for r in rows})==24
assert all(r['count']==6690 and all(math.isfinite(r[k]) for k in metrics) for r in roles)
stats=[]
for model in c['arms']:
 rr=[r for r in rows if r['model']==model];assert sorted(r['seed'] for r in rr)==[42,52,62]
 stats.append(dict(dataset='raf_spare_parts',model=model,n=3,**{k:dict(mean=statistics.mean(r[k] for r in rr),sample_sd=statistics.stdev(r[k] for r in rr)) for k in metrics}))
lookup={(r['model'],r['seed']):r for r in rows};sl={r['model']:r for r in stats};paired=[]
for reference in [plain,norm]:
 for model in c['arms']:
  if model==reference:continue
  x=dict(dataset='raf_spare_parts',reference=reference,comparator=model,completed_seed_count=3,seeds=[42,52,62])
  for k in metrics:
   wins=[s for s in [42,52,62] if lookup[reference,s][k]<lookup[model,s][k]]
   losses=[s for s in [42,52,62] if lookup[reference,s][k]>lookup[model,s][k]]
   x[k]=dict(reference_wins=len(wins),winning_seeds=wins,losing_seeds=losses,ties=3-len(wins)-len(losses),reduction_percent_of_3seed_mean=100*(1-sl[reference][k]['mean']/sl[model][k]['mean']))
  paired.append(x)
kst=lambda x:datetime.fromtimestamp(x,timezone(timedelta(hours=9))).isoformat()
cost=dict(first_fit_epoch_started_kst=kst(min(epoch_started)),last_fit_epoch_finished_kst=kst(max(epoch_finished)),queue_completed_kst=kst(read(P/'status.json')['updated_unix']),first_epoch_to_queue_terminal_seconds=read(P/'status.json')['updated_unix']-min(epoch_started),fit_elapsed_seconds_sum=sum(x['fit_elapsed_seconds'] for x in a['conditions']),epoch_timing_seconds_sum=sum(x['epoch_timing_seconds_sum'] for x in a['conditions']),cloud_rental_usd=0,electricity_cost=None,scope='Personal RTX5080; actual recorded times, no imputation. Synthetic qualification excluded from fit.')
sources={str(p.relative_to(R)):sha(p) for p in [D/'terminal_audit.json',D/'retrieval_receipt.json',P/'execution_contract.json',P/'collection_manifest.json',Path(__file__)]}
result=dict(evaluation_scope='validation_only',representative=plain,rows=rows,three_seed=stats,paired=paired,endpoint_roles=roles,partition_cells=cells,cost=cost,sources_sha256=sources,history='User interruption after qualification and pre-fit parent-guard correction preserved. All 24 scientific fits completed. Collector v1 rejected an operational test Python filename before streaming; read-only collection v2 completed. No scientific rerun.')
(O/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for filename,rr in [('selected_conditions.csv',rows),('endpoint_roles.csv',roles)]:
 with (O/filename).open('w') as f:w=csv.DictWriter(f,list(rr[0]));w.writeheader();w.writerows(rr)
lines=['# RAF 24조건 결과 — TitanTPP와 외부 TPP 비교','','**24조건 학습·48개 selected/last validation 재평가와 원본 CPU 감사를 완료했다.** 2026-10-01 08:55:56 KST 큐 종료. 원본 회수는 '+kst(a['collected_unix'])+'이다.','','## 결과의 의미','','대표 TitanTPP MLP의 3seed 평균 RMSE는 외부 최저 S2P2보다 0.130% 낮지만 MAE는 외부 최저 RMTPP보다 1.252% 높다. 정규화 변형은 8모델 중 평균 RMSE가 가장 낮으며 기존 MLP보다 0.155% 낮다. RAF는 큰 폭의 수량 우위를 추가 입증하는 결과가 아니라 지표에 따라 장단점이 갈리는 추가 데이터 결과다.','','모든 값은 seed42·52·62의 평균 ± 표본표준편차이며 낮을수록 좋다. 세 지표는 동일한 최초 최소 validation raw 수량 RMSE checkpoint에서 얻었다.','','| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |','|---|---:|---:|---:|']
for r in stats:lines.append('| '+labels[r['model']]+' | '+' | '.join(f"{r[k]['mean']:.6f} ± {r[k]['sample_sd']:.6f}" for k in metrics)+' |')
lines+=['','## 같은 seed의 비교','','승수는 세 seed 중 기준 모델의 지표가 더 낮은 횟수다. 평균 차이나 승수는 통계적 유의성 검정이 아니다.','','| 기준 모델 | 비교군 | MAE 승 | RMSE 승 | 시간 NLL 승 |','|---|---|---:|---:|---:|']
for r in paired:lines.append('| '+labels[r['reference']]+' | '+labels[r['comparator']]+' | '+' | '.join(str(r[k]['reference_wins'])+'/3' for k in metrics)+' |')
lines+=['','## 조건별 선택과 성능','','| 모델 | seed | 선택 / 종료 epoch | MAE | RMSE | 시간 NLL |','|---|---:|---:|---:|---:|---:|']
for ar,r in zip(a['conditions'],rows):lines.append(f"| {labels[r['model']]} | {r['seed']} | {r['epoch']} / {ar['completed_epochs']} | "+' | '.join(f'{r[k]:.6f}' for k in metrics)+' |')
lines+=['','## 데이터와 해석 범위','','RAF는 월별 양의 부품 수요를 사건열로 표현했다. train 25,779개, validation 6,690개 target을 사용하며 최대 길이84에는 target이 포함된다. lookback84의 단위는 월이고 시간 스케일은6개월이다. 월별 합계나 미래 고정기간 수요를 직접 예측한 결과와 구분한다. held-out/test는 읽거나 평가하지 않았다.','','정규화는 기존 고정 /8을 max(활성 분기 수,1)로 바꾼 변형이다. 기존 MLP와 같은 GPU·split·head·loss·seed에서 비교했지만 데이터에 따라 모델을 교체하지 않는다. 다른 데이터의 정규화 seed52·62 보류와 RAF에서 승인된 3seed를 구분한다.','','quantity/history/additional_history별 모든 구간, 빈 구간, selected/last 기록은 comparison.json과 endpoint_roles.csv에 보존했다. B·Full·Gate는 이 RAF24조건에 포함되지 않아 RAF의 보완모듈 유무 효과로 해석할 수 없다.','','## 원본 감사와 실측 비용','',f"473개 원본 파일·104개 동결 source·48개 checkpoint를 SHA와 CPU strict loading으로 검증했다. optimizer step, RNG/shuffle, 공통 초기화 증거, 전체 표본 노출과 seed별 batch prefix, 최초 RMSE 선택과 early stop, 재평가 기록 및 구간 합계를 확인했다. 새 학습·forward·GPU 재평가는 수행하지 않았다. CPU와 학습 Runtime 차이는 terminal_audit.json에 기록했다.",'',f"첫 본학습 epoch 시작부터 큐 종료까지 {cost['first_epoch_to_queue_terminal_seconds']/60:.3f}분, train_one 실측합 {cost['fit_elapsed_seconds_sum']:.3f}초, epoch 기록합 {cost['epoch_timing_seconds_sum']:.3f}초다. 원래 중단·준비 시간까지 포함한 전체 운영 시간과 구분한다. 개인 RTX5080 추가 cloud 임대료는 $0, 전기료는 미측정이다. 서로 다른 모델의 종료 epoch가 달라 전체 fit 시간으로 속도 우위를 주장하지 않는다.",'',result['history'],'','## 증거','']
for p,h in sources.items():lines.append(f'- [{Path(p).name}]({R/p}) — SHA256 `{h}`')
(O/'report.md').write_text('\n'.join(lines)+'\n')
verification=dict(status='passed',conditions=24,replay_roles=48,models=8,seeds=[42,52,62],all_results_finite=True,validation_count=6690,sample_sd_ddof=1,terminal_cpu_audit_sha256=sha(D/'terminal_audit.json'),source_hashes=sources,heldout_read=False,new_fit_or_replay=False)
(O/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps({'cost':cost,'paired':paired[:3]},ensure_ascii=False))
