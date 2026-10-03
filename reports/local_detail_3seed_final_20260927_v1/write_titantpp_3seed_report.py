import json, hashlib, shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
ROOT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
OUT=ROOT/'reports/local_detail_3seed_final_20260927_v1'
read=lambda name:json.loads((OUT/name).read_text())
audit=read('terminal_audit.json');rows=read('condition_results.json');aggs=read('three_seed_summary.json');changes=read('paired_seed_changes.json')
names={'yellow_trip_hourly':'Taxi','intermittent_frozen_5000':'Intermittent','insta_market_basket':'Instacart'}
models={'titantpp':'기존 B','titantpp_local_detail':'이력 보완 TitanTPP','rmtpp':'RMTPP','thp':'THP','nhp':'NHP','sahp':'SAHP'}
order=list(names);arms=list(models)
get=lambda ds,arm:next(r for r in aggs if r['dataset']==ds and r['model']==arm)
fmt=lambda s:f"{s['mean']:.4f} ± {s['sample_sd']:.4f}"
lines=[]
def add(s=''):lines.append(s)
def table(headers,body):
 add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
 for row in body:add('| '+' | '.join(map(str,row))+' |')
 add()
completed=datetime.fromtimestamp(audit['recovery_terminal']['completed_at_unix'],timezone(timedelta(hours=9)))
add('# 이력 보완 TitanTPP — 3-seed validation 최종 결과')
add()
add(f"최종 학습 종료: **{completed:%Y-%m-%d %H:%M:%S} KST**. 범위: 세 데이터셋 × 여섯 모델 × seed42·52·62, **54조건·108개 selected/last validation 재평가**.")
add()
add('**판단 — 제한을 명시하면 사용 가능**')
add('- Taxi 수량 예측에서는 이력 보완 TitanTPP를 이어갈 근거가 강화됐다. 세 seed 모두 여섯 모델 중 RMSE 1위이며, B 대비 평균 RMSE 12.96%, MAE 11.80% 감소했다.')
add('- 전체 데이터셋의 B 대체 모델로 확정할 근거는 부족하다. Intermittent는 평균 RMSE가 1.88% 증가했고, Instacart는 세 seed 모두 B보다 RMSE가 소폭 높았다(평균 +0.108%).')
add('- 모든 지표를 함께 보는 사전 기준은 B 대비 9개 데이터셋·seed 조합 모두 미통과다. Taxi의 시간 NLL 악화와 다른 데이터셋의 수량·구간·안정성 반례를 유지한다. 실행 성공과 성능 기준 통과는 별도다.')
add('- 연구 방향은 **Taxi의 수량 개선을 중심 근거로 삼고, 시간 예측 및 데이터셋별 한계를 명시한 후보**로 정리하는 것이 타당하다. “여러모로 항상 더 좋다” 또는 “범용적으로 B보다 우월하다”는 주장은 이 결과가 지지하지 않는다.')
add()
add('## 검증 범위와 현재 기준선 — 완료')
add()
table(['출처','조건','재평가','기록된 optimizer step'],[[g,v['conditions'],v['endpoint_replays'],f"{v['optimizer_steps']:,}"] for g,v in audit['groups'].items()]+[['합계',54,108,f"{audit['total_recorded_optimizer_steps']:,}"]])
add('복구 seed62 paired 파일은 원본 B/local 2조건·4재평가와 복구 4조건·8재평가를 결합한 6모델·12재평가다. 원본 B/local을 복구 신규 조건으로 중복 계수하지 않았다. 정지한 RMTPP 첫 시도는 별도 실패 이력으로 보존되며, 저장되지 않은 부분 update 수는 위 23,106,229 step 합계에 포함할 수 없다.')
add()
add('다음 항목을 54조건 모두의 소형 JSON 증적으로 검증했다: 첫 허용 조기 종료 시점, 최초 strict raw-RMSE 최소 checkpoint, 실제 epoch·step·train/validation 표본수, 초기 상태 SHA, 동일 seed의 공통 학습 batch prefix, validation 순서, selected/last 재평가와 history 일치, 구간 count/SSE/MAE/time NLL 합산, first40/last30 평균·표본 표준편차, paired 비교와 복구 partition 집계. 신규 복구는 네 개의 독립 worker PID와 60회 CUDA qualification, 고정 Runtime·공통 deadline, 정상 프로세스/tmux 종료를 확인했다.')
add()
add('5090 원본 완료 결과·입력·체크포인트 61개를 원격 바이트 SHA로 재검증했다. 기존 16개 checkpoint 파일 SHA와 복구 8개 checkpoint SHA를 보존했다. 로컬 감사는 checkpoint를 역직렬화하거나 데이터·held-out 예측을 읽지 않았으며, 이미 수행된 validation 재평가 증적을 검증했다. B/local 공통 초기 tensor 검증은 동결 qualification/실행 증적에 근거하고, 이번 감사에서는 초기 상태 SHA의 일치를 확인했다.')
add()
add('## 세 seed의 동일 선택 checkpoint 비교 — 완료')
add()
add('모든 수량·시간·구간 지표는 각 조건의 **동일한 validation raw-RMSE 선택 checkpoint**에서 나온다. 아래 ±는 seed42·52·62 간 표본 표준편차(ddof=1)이며 신뢰구간이 아니다. 세 seed를 동일 가중치로 평균했다. RMSE·MAE·시간 NLL은 모두 낮을수록 좋다. 데이터셋 간 수량 단위와 척도가 다르므로 하나의 RMSE로 합산하지 않는다. RMTPP·THP·NHP·SAHP는 공통 head/loss 및 고정 학습 규칙을 적용한 비교 구현이며, 각 원논문의 최적 튜닝 결과에 대한 순위가 아니다.')
add()
for ds in order:
 add(f"### {names[ds]}");add()
 table(['모델','수량 RMSE','수량 MAE','시간 NLL'],[[models[arm]]+[fmt(get(ds,arm)['metrics'][m]) for m in ['qty_rmse','qty_mae','time_nll']] for arm in arms])
add('## B 대비 seed별 차이와 반례 — 완료');add()
add('수량 변화율은 (local/B − 1) × 100이며 음수가 개선이다. 시간 NLL은 local − B다. 사전 기준을 사후 수정하지 않았다.');add()
table(['데이터','seed','RMSE 변화','MAE 변화','시간 NLL 차이','수량 RMSE 순위'],[[names[x['dataset']],x['seed'],f"{x['rmse_change_percent']:+.3f}%",f"{x['mae_change_percent']:+.3f}%",f"{x['time_nll_change']:+.6f}",f"{x['quantity_rank_among_six']}/6"] for x in changes])
add('Taxi의 세 seed 모두 수량 기준은 통과했으나 시간 NLL 허용 증가 +0.01을 초과했다. 평균 시간 NLL은 B 0.724678 → local 1.182236이다. Intermittent는 seed42·52의 RMSE가 나빠졌고 seed62에서만 개선됐다. Instacart는 시간 NLL 차이는 작고 허용 범위 이내지만, 전체·tail RMSE 개선 조건은 모든 seed에서 미통과했다.');add()
add('### 사전 기준의 미통과 항목 전체');add()
for x in changes:add(f"- {names[x['dataset']]} seed{x['seed']}: "+', '.join('`'+k+'`' for k in x['failed_prespecified_checks'])+'.')
add();add('기준: 전체 RMSE 및 tail RMSE strict 개선, 전체 MAE ≤ B×1.01, body/tail MAE 및 각 중간 수량 구간 MAE/RMSE ≤ B×1.02, 시간 NLL ≤ B+0.01, last30 RMSE 평균·표본 SD ≤ B×1.05. 모든 비교 모델 대비 전체 판정은 [terminal_audit.json](terminal_audit.json)의 jobs에 보존했다.');add()
add('## 수량·이력 구간의 개선과 반례 — 완료');add()
add('아래는 B와 local의 선택 checkpoint에 대한 세 seed 평균이다. 경계는 오른쪽 값을 포함한다. 각 표본수는 seed마다 같은 validation 대상 수이며, 세 seed를 더한 수가 아니다. 모든 모델·구간의 평균과 표본 SD는 [three_seed_summary.json](three_seed_summary.json)에 있다. 빈 구간은 측정 불가로 유지하며 0점으로 처리하지 않았다.');add()
add('- Taxi: 다섯 수량 구간의 평균 RMSE가 모두 개선됐다. 그러나 이력 65–128 구간(n=1,475)의 RMSE는 2.5126 → 2.7916으로 악화됐다. 시간 NLL도 전체에서 악화됐으므로 수량 개선을 모든 과제의 개선으로 일반화할 수 없다.')
add('- Intermittent: 수량 46 초과–187 이하 구간(n=3,405)의 RMSE는 4.3077 → 4.7665, tail 187 초과(n=1,141)는 8.6230 → 9.4091로 악화됐다. 이력 64 이하(n=25,088)도 1.4217 → 1.5840으로 악화됐다.')
add('- Instacart: 수량 20 이하 두 구간은 평균 RMSE가 개선됐지만, 20 초과 세 구간은 악화됐다. tail 35 초과(n=6,036) RMSE는 24.4246 → 24.7895다. 이력 64 초과 구간은 관측이 없고, 추가 세부 이력 구간에서도 개선과 악화가 섞여 있다.')
add()
def label(bounds,i):return f'≤ {bounds[0]}' if i==0 else f'> {bounds[-1]}' if i==len(bounds) else f'({bounds[i-1]}, {bounds[i]}]'
for kind,title in [('quantity','수량'),('additional_history','관측 이력 길이')]:
 add(f'### {title} 구간 전체');add();body=[]
 for ds in order:
  b=get(ds,'titantpp')['strata'][kind];l=get(ds,'titantpp_local_detail')['strata'][kind]
  for x,y in zip(b,l):
   if not x['count']:continue
   body.append([names[ds],label(x['boundaries'],x['bin']),f"{x['count']:,}",fmt(x['metrics']['qty_rmse']),fmt(y['metrics']['qty_rmse']),f"{x['metrics']['qty_mae']['mean']:.4f} → {y['metrics']['qty_mae']['mean']:.4f}",f"{x['metrics']['time_nll']['mean']:.6f} → {y['metrics']['time_nll']['mean']:.6f}"])
 table(['데이터','구간','표본수','B RMSE ± SD','local RMSE ± SD','MAE B → local','시간 NLL B → local'],body)
add('## 안정성과 계산 비용 — 완료');add()
add('last30은 각 조건의 실제 종료 전 30개 epoch라 서로 다른 학습 위치를 비교한다. 같은 학습 구간인 first40도 함께 제시한다. 아래 값은 각 seed의 epoch별 RMSE 평균·SD를 계산한 뒤 그 통계를 세 seed 평균한 것이며, 위 선택 checkpoint의 seed 간 SD와 다르다.');add()
body=[]
for ds in order:
 for arm in arms[:2]:
  x=get(ds,arm)
  body.append([names[ds],models[arm]]+[f"{x['stability'][w][m]['mean']:.4f}" for w,m in [('first40','mean'),('first40','sd'),('last30','mean'),('last30','sd')]])
table(['데이터','모델','first40 평균','first40 내 SD','last30 평균','last30 내 SD'],body)
add('Taxi local의 last30 평균·변동은 B보다 낮지만 실제 학습 epoch가 훨씬 길다. Intermittent local은 seed42의 불안정성을 포함해 평균 last30 변동이 크게 나빴다. Instacart local은 first40/last30 통계가 개선됐어도 선택 checkpoint 전체 RMSE 우위로 이어지지 않았다.');add()
body=[]
for ds in order:
 b=get(ds,'titantpp');l=get(ds,'titantpp_local_detail')
 body.append([names[ds],f"{int(b['parameters']['mean']):,} → {int(l['parameters']['mean']):,}",f"{b['mean_epoch_seconds']['mean']:.2f} → {l['mean_epoch_seconds']['mean']:.2f}",f"{b['peak_allocated_bytes']['mean']/1024**2:.1f} → {l['peak_allocated_bytes']['mean']/1024**2:.1f}",f"{l['training_elapsed_seconds']['mean']/b['training_elapsed_seconds']['mean']:.2f}×"])
table(['데이터','파라미터 B → local','평균 epoch 초 B → local','평균 peak allocated MiB B → local','실제 학습시간 local/B'],body)
add('local 파라미터는 +6,144개다. epoch당 시간은 Taxi +8.47%, Intermittent +9.12%, Instacart +21.30%였다. 실제 학습시간 비율은 조기 종료 시점 차이를 포함한다. 데이터셋 내부의 같은 서버 비교이며, 전용 성능 벤치마크나 동등 학습량·동등 용량의 인과 실험은 아니다. 학습시간은 대기·정지한 첫 시도·별도 endpoint 재평가 시간을 제외한다.');add()
add('## 실제 종료·선택·학습량 전체 목록 — 완료');add()
table(['데이터','seed','모델','실제 종료 epoch','선택 epoch','step','train/epoch','validation/replay'],[[names[r['dataset']],r['seed'],models[r['model']],r['completed_epochs'],r['best_epoch'],f"{r['optimizer_steps']:,}",f"{r['train_targets_per_epoch']:,}",f"{r['validation_targets_per_replay']:,}"] for r in rows])
assert all(r['stopped_early'] for r in rows)
add('54조건 모두 max300/min40/patience40 규칙의 첫 허용 시점에서 조기 종료됐다. 조건별 원본 경로·source/contract SHA·초기 및 선택 상태 SHA·selected/last 지표는 [condition_results.json](condition_results.json)에 있다.');add()
add('## 한계와 원본 보존 — 완료');add()
add('- 이번 결론은 validation에 한정된다. held-out 평가나 새로운 seed/모델별 튜닝을 하지 않았으며, seed 세 개만으로 통계적 유의성·범용성을 주장하지 않는다.')
add('- 기존 RMTPP seed62 첫 시도의 native 정지 원인은 미확정이다. 저장된 진척이 없어 부분 update 수는 알 수 없다. 각 모델의 새 프로세스 실행은 격리 복구이며 확정된 라이브러리 버그 수정이 아니다.')
add('- 이전 Taxi 완료 후 Intermittent 시작 전 중단된 v2 wrapper와 Intermittent v4 복구도 출처로 보존했다. 사용한 완료 partition만을 계수했으며 이번 RMTPP 정지와 혼동하지 않는다.')
add('- 동결 Runtime의 THP attention backward 비결정성 경고를 보존했다. 동일 checkpoint의 selected/last 재평가와 기록 지표 일치는 통과했지만, 학습 전체의 비트 단위 결정성을 새로 입증한 것은 아니다.')
add('- B/local 간 추가 파라미터 용량 효과를 분리하지 않았다. 현재 결과는 이력 보완 구조의 독립 인과 효과를 증명하지 않는다.')
add()
add('## 남은 작업 순서')
add()
add('**승인된 학습·증적 감사·최종 비교 — 완료**')
add('- 5080 신규24조건과 5090 원본8조건·복구4조건, 재사용 seed42 18조건을 모두 통합했다. 이번 범위에 남은 학습이나 자동 재실행은 없다.')
add()
add('**시간별 모니터링 종료 — 완료**')
add('- 최종 보고서를 작성하고 heartbeat `5080-5090`을 삭제해 시간별 모니터링을 종료했다. 기존 `vnc-hard-lmm-5090-screening`은 PAUSED로 유지했으며 이 작업은 archive하지 않았다.')
add()
add('**후속 연구 범위 확정 — 다음 작업, 새 실행은 승인 필요**')
add('- Taxi 수량 개선을 중심으로 논문 주장 범위를 정리할 수 있다. 범용 B 대체를 목표로 한다면 시간·tail 손실과 동일 용량/학습량 비교를 다루는 별도 설계가 필요하다. 추가 학습·held-out·Runtime/head/loss 변경은 이번 승인에 포함되지 않는다.')
add()
add('## 증거와 재계산')
add()
add('- [전체 실행 감사](terminal_audit.json), [54조건의 원본 연결 및 상세 지표](condition_results.json), [세 seed 평균·표본 SD와 전 구간](three_seed_summary.json), [seed별 B 대비 차이](paired_seed_changes.json).')
add('- [순수 Python 감사 코드](audit_titantpp_3seed_final.py): `--verify-only`로 저장 결과와 재계산 값을 대조한다. 모델·checkpoint·실제 데이터를 로드하지 않는다.')
add('- [복구 완료 원본 사본]('+str((ROOT/'search_artifacts/local_detail_instacart_remaining_5090_20260926_v1/monitor/20260927T130927Z/terminal_5090_recovery/collection_manifest.json'))+'), [5080 완료 감사]('+str(ROOT/'search_artifacts/local_detail_replication_5080_20260924_v1/monitor/20260925T204245Z/terminal_5080/terminal_audit.json')+').')
add('- 모든 source root와 canonical/source SHA는 전체 실행 감사의 source_receipts에 보존했다. 원본 파일을 수정·삭제하지 않았고 커밋·Push·외부 공개를 수행하지 않았다.')
with (OUT/'final_report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
for name in ['audit_titantpp_3seed_final.py','collect_instacart_remaining_terminal.py','write_titantpp_3seed_report.py']:
 target=OUT/name
 with target.open('xb') as f:f.write((Path('/private/tmp')/name).read_bytes())
print(json.dumps({'report':str(OUT/'final_report.md'),'completed_kst':completed.isoformat(),'table_rows':len(rows),'bytes':(OUT/'final_report.md').stat().st_size},ensure_ascii=False))
