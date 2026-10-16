"""Render the checked stored-statistics audit; no evaluation."""
from pathlib import Path
import json
import hashlib

O=Path(__file__).resolve().parent
x=json.loads((O/'RESULT.json').read_text())
v=json.loads((O/'VERIFICATION.json').read_text())
def contrast(split, seed, candidate='Capacity', reference='CNN'):
    return next(c['strata'] for c in x['contrasts'] if (c['split'],c['seed'],c['candidate'],c['reference']) == (split,seed,candidate,reference))
def row(split,seed,arm,stratum='overall'):
    return next(r for r in x['rows'] if (r['split'],r['seed'],r['arm'],r['stratum'])==(split,seed,arm,stratum))
L=[]
def add(s=''):L.append(s)
add('# Taxi 용량 대조군의 seed별 오류 분해')
add('\n**수량 구간별 오류가 갈리는 위치는 확인했다. 다만 개별 예측이 저장되지 않아, 이력·개체별 원인을 분리하거나 다음 CNN 구조를 선정할 근거는 아직 부족하다.**')
add('\n## Material Passport')
add('\n- Workflow: academic-research-suite / experiment-agent / validate. Verification Status: ANALYZED.\n- 대상: Taxi 기존 CNN·이력 적응 CNN(Adaptive)·용량 대조군(Capacity), seed42/52/62의 기존 선택 checkpoint.\n- 분석: 완료된 Validation 8,268개·Test 8,327개 대상의 저장 오류 합계. 새 학습·추론·원격 실행·checkpoint 재선택은 0이다.\n- 상태: 집계 분석·검산 완료. 요청한 행별 이력·개체 비교는 개별 예측 미저장으로 미완료다.\n- 한계: 반복 Test 참고 탐색이다. 인과관계·통계 유의성·새로운 독립 일반화를 주장하지 않는다.')
add('\n## 저장 형식과 분석 가능 범위 — 확인 완료')
add('\n앞선 답변에서는 저장된 개별 예측으로 바로 비교할 수 있는 듯 설명했으나, 실제 해당 평가 결과 18개에는 `raw_predictions_written=false`가 기록돼 있다. 후보의 Validation/Test 원본 archive에도 행 예측 파일이 없다. 저장된 `prediction_rows`는 평가 대상 수를 뜻하며, 예측 행 파일의 존재를 뜻하지 않는다. 기존 CNN의 다른 Train·Validation 행 자료가 있더라도 후보의 대응 예측이 없으면 요청한 세 모델의 행별 비교를 완성할 수 없다.')
add('\n현재 가능한 분석은 전체·수량 구간별 제곱오차(SSE), 절대오차 합계(AE), 부호 오차 합계 및 Validation 이력 길이 구간 집계다. 같은 split에서 평가 대상·정답·시간·입력 해시와 loader가 모두 같고, Validation/Test의 선택 epoch·checkpoint·모델 상태가 연결됨을 확인했다. 소수 사례의 영향, 오류가 난 개체, 이력의 상승·하락 패턴, 같은 행이 과대에서 과소로 바뀌었는지는 확인하지 못했다. 원본을 재생성하려고 기존 CNN을 다시 실행하지 않았다.')
add('\n## 전체 성능과 split별 방향 — 분석 완료')
add('\n변화는 Capacity−CNN이다. 음수는 개선이다. seed52의 Test는 RMSE만 악화하고 MAE는 개선하므로 두 지표가 모두 악화한 것으로 묶지 않는다.')
add('\n| seed | Validation ΔRMSE / ΔMAE | Test ΔRMSE / ΔMAE | Test 해석 |\n|---:|---:|---:|---|')
for seed,desc in [(42,'모든 수량 구간에서 두 지표 개선'),(52,'구간별 이득·손실 상쇄, RMSE와 MAE 방향 다름'),(62,'낮은 수량 이득보다 큰 수량 손실이 큼')]:
    a,b=contrast('validation',seed)[0],contrast('test',seed)[0]
    add(f"| {seed} | {a['qty_rmse']:+.6f} / {a['qty_mae']:+.6f} | {b['qty_rmse']:+.6f} / {b['qty_mae']:+.6f} | {desc} |")
add('\nseed42는 Validation에서 두 지표가 악화했지만 Test에서 크게 개선했다. seed52·62는 Validation에서 두 지표가 개선했으나 Test에서는 같은 방향이 유지되지 않았다. 이를 학습 실패·선택 오류·과적합 원인으로 확정하지 않는다.')
add('\n## Test의 어느 수량 구간에서 갈리는가 — 분석 완료')
add('\n아래는 Capacity−CNN의 ΔRMSE / ΔMAE다. 경계 7·686·1562·3449는 기존 Train 기준을 그대로 사용했다. 실제 정답 수량 q에 따른 사후 구분이므로, 예측 시 사용할 수 있는 입력 분기 조건은 아니다.')
add('\n| 실제 수량 구간 | 대상 수 | seed42 | seed52 | seed62 |\n|---|---:|---:|---:|---:|')
for i in range(1,6):
    cs=[contrast('test',seed)[i] for seed in [42,52,62]]
    add(f"| {cs[0]['stratum']} | {cs[0]['count']} | "+' | '.join(f"{c['qty_rmse']:+.3f} / {c['qty_mae']:+.3f}" for c in cs)+' |')
add('\n**seed42: 큰 수량에서 기존 CNN의 큰 오류가 많이 줄었다.** 전체 제곱오차 감소 141,393,995.863 중 q>1562의 421개 대상(전체의 5.06%) 구간에서 124,864,510.504가 감소해 88.31%를 차지했다. q>3449뿐 아니라 1562<q≤3449에서도 큰 이득이 있었다. 다섯 구간 모두 RMSE·MAE가 개선했지만, 그 안의 모든 행이 개선했다는 뜻은 아니다.')
add('\n**seed52: 중간보다 큰 수량의 이득을 다른 구간의 손실이 상쇄했다.** 1562<q≤3449에서는 SSE가 8,299,394.371 줄었지만 q>3449에서 4,568,371.756, 7<q≤686에서 2,534,764.769, 686<q≤1562에서 1,633,298.494 늘었다. 모든 구간을 합치면 SSE +419,689.515로 RMSE가 소폭 악화했다. 반면 절대오차 합계는 3,191.590 줄어 MAE는 개선했다. RMSE는 큰 오류에 더 민감하므로 제곱오차 합계가 늘면서 절대오차 합계는 줄 수 있다. 개별 행의 과대·과소 방향이 바뀌었는지는 확인할 수 없다.')
add('\n**seed62: 낮은 수량에서 얻은 이득보다 큰 수량의 손실이 컸다.** q≤686의 SSE는 4,331,949.827 줄었으나 q>1562에서 15,055,713.397 늘었다. 특히 1562<q≤3449의 SSE +11,425,016.762, MAE +35.335가 큰 부분이다. 전체 SSE는 11,521,710.055 증가했다. 따라서 가장 큰 수량 83개만의 문제로 좁히면 중간 이상 수량 구간의 손실을 놓친다.')
add('\n## 과대·과소예측과 평균 편향을 구분한다 — 분석 완료')
add('\n부호 오차 e=예측−정답, A=Σ|e|, S=Σe일 때 과대예측 오차량은 (A+S)/2, 과소예측 오차량은 (A−S)/2다. 아래는 해당 오차량의 Capacity−CNN 변화다. 양수는 그 방향의 오류량 증가이며, 예측 건수나 동일 사례의 이동을 뜻하지 않는다.')
add('\n| seed | 과대예측 오차량 변화 | 과소예측 오차량 변화 | CNN 평균 편향 → Capacity 평균 편향 |\n|---:|---:|---:|---:|')
for seed in [42,52,62]:
    d=contrast('test',seed)[0];a=row('test',seed,'CNN');b=row('test',seed,'Capacity')
    add(f"| {seed} | {d['overprediction_ae_mass']:+,.3f} | {d['underprediction_ae_mass']:+,.3f} | {a['qty_bias']:+.3f} → {b['qty_bias']:+.3f} |")
add('\nseed42는 과대예측 오차량 감소가 과소예측 증가를 크게 웃돈다. seed52는 반대로 과소예측 감소가 과대예측 증가를 조금 웃돌아 MAE가 개선했다. seed62는 과대예측이 줄었지만 과소예측 증가가 더 커서 MAE가 악화했다. 세 seed에서 동일한 방향으로 예측을 보정했다고 설명할 수 없다.')
add('\n평균 편향만 보면 세 seed 모두 절댓값이 줄었다. 그러나 MSE=평균 편향²+부호 오차의 모집단 분산으로 분해하면 seed52·62는 편향 항이 줄어도 분산 항이 늘어 전체 MSE가 증가했다. 이 분산은 **같은 split 안의 행별 잔차 분산**이며, seed 사이의 분산이나 예측 불확실성 추정치가 아니다.')
add('\n| seed | ΔMSE | Δ평균 편향² | Δ잔차 분산 |\n|---:|---:|---:|---:|')
for d in v['moment_deltas']:
    if d['split']=='test' and d['arm_minus_CNN']=='Capacity':
        add(f"| {d['seed']} | {d['mse']:+,.3f} | {d['squared_bias']:+,.3f} | {d['residual_population_variance']:+,.3f} |")
add('\n따라서 평균 수량 수준을 더 낮추거나 올리는 일괄 보정만으로 문제를 해결할 근거는 없다. 위 분해는 오류의 대수적 설명이며 CNN 내부 기전의 인과 분해가 아니다.')
add('\n## 이력 적응 CNN과의 차이 — 분석 완료')
add('\n| seed | CNN Test RMSE / MAE | Adaptive Test RMSE / MAE | Capacity Test RMSE / MAE |\n|---:|---:|---:|---:|')
for seed in [42,52,62]:
    rr=[row('test',seed,arm) for arm in ['CNN','Adaptive','Capacity']]
    add(f'| {seed} | '+' | '.join(f"{r['qty_rmse']:.3f} / {r['qty_mae']:.3f}" for r in rr)+' |')
add('\nAdaptive는 seed42·62에서 Capacity보다 전체 두 지표가 좋고, seed52에서 크게 나쁘다. Capacity의 더 좋은 Test 평균은 특히 Adaptive52의 큰 손실을 피한 결과를 포함한다. Adaptive52는 7<q≤686부터 q>3449까지 모든 해당 구간에서 CNN보다 RMSE·MAE가 나빴다. q>3449의 평균 편향은 CNN−239.988, Adaptive+646.424, Capacity+129.655로 다르다. 이는 큰 수량 구간의 예측 수준 차이이지만, 어떤 관측 이력이 이를 유발했는지는 저장 집계에서 알 수 없다.')
add('\nAdaptive62는 q>3449에서 CNN보다 악화했지만 나머지 구간의 이득으로 전체 성능은 개선했다. Capacity62는 686<q≤1562와 1562<q≤3449에서도 악화했다. 같은 seed62라고 해서 두 후보의 손실 위치가 같지는 않다.')
add('\n## Validation의 이력 길이 집계와 한계 — 분석 완료')
add('\n세 개의 기존 이력 길이 구간을 확인했다. 아래는 Capacity−CNN이다. 이 표는 수량·개체를 동시에 맞춘 대조가 아니다.')
add('\n| 이력 길이 | 대상 수 | seed42 ΔRMSE / ΔMAE | seed52 | seed62 |\n|---|---:|---:|---:|---:|')
for group in ['history≤64','64<history≤128','history>128']:
    vals=[]
    for seed in [42,52,62]:
        aa=next(r for r in x['validation_history_strata'] if (r['arm'],r['seed'],r['stratum'])==('Capacity',seed,group))
        bb=next(r for r in x['validation_history_strata'] if (r['arm'],r['seed'],r['stratum'])==('CNN',seed,group))
        vals.append(f"{aa['qty_rmse']-bb['qty_rmse']:+.6f} / {aa['qty_mae']-bb['qty_mae']:+.6f}")
    add(f"| {group} | {aa['count']} | "+' | '.join(vals)+' |')
add('\n이력이 64개 이하인 구간은 Capacity의 MAE가 세 seed 모두 조금 악화했다. 65~128개 구간에서는 RMSE·MAE가 모두 악화했다. 128개 초과에서는 seed42가 악화하고52·62가 개선했다. Adaptive 역시 이력이128개 이하인 두 구간의 MAE는 세 seed 모두 조금 악화했다. 반면 실제 q≤7 구간에서는 Capacity의 RMSE·MAE가 양 split 세 seed 모두 개선했다. 이처럼 이력 길이와 실제 수량의 주변 집계는 서로 바꿔 해석할 수 없다. 관측 수량·개체·변화 폭을 맞춘 행 자료 없이 “짧은 이력용 gate”나 “더 긴 CNN”을 채택할 근거로 쓰지 않는다.')
add('\n같은 선택 checkpoint의 Train 예측을 후보에서 저장하지 않았으므로 학습 중 loss를 가져와 Train·Validation의 행별 일관성을 확인했다고 하지 않는다. Test의 이력 길이별 집계도 없다.')
add('\n## 보조 지표와 전체 기록 — 완료')
add('\n세 모델의 전체·큰 수량 RMSE/MAE 및 전체·큰 수량 recorded positive-integer TimeNLL을 RESULT.json과 metrics_by_stratum.csv에 보존했다. Taxi Test 전체 TimeNLL 3seed 평균은 CNN0.767969, Adaptive1.239752, Capacity0.908386으로 두 후보 모두 악화했다. 작은 TimeNLL 값의 상대 변화율은 해석하지 않는다. 모든 seed·구간·불리한 차이를 남겼고 기존 성능이나 checkpoint를 재선택하지 않았다.')
add('\n## 다음 백본 개선의 근거를 판정한다 — 판단 완료')
add('\n**새 학습은 보류한다.** 이번에는 손실의 위치와 상쇄를 더 구체적으로 확인했지만, 입력으로 식별할 수 있는 반복 오류와 CNN 내부 처리의 결함을 연결하지 못했다. 용량 대조군이 가능성 없는 구조라고 결론 내리는 것도 아니다.')
add('\n- 검토할 질문: 큰 오류의 일부를 줄이면서 다른 사례의 과소예측·오류 분산을 키우지 않는 표현이 가능한가? 이는 목표이며 검증된 방법론이 아니다.\n- 현재 근거로 채택하지 않는 변경: 일정 배율로 예측·잔차 줄이기, 실제 미래 수량에 따른 gate, 평균 편향만 맞추기, 단순히 이력 범위를 늘리기.\n- 필요한 증거: 후보와 기존 CNN의 같은 target별 예측, 예측 전 관측 이력, 큰 오차 사례의 집중도와 seed 간 겹침, Train·Validation에서 반복되는 조건. 현 저장 결과만으로 이 연결은 복원되지 않는다.\n- 남은 작업 상태: 개별 행 비교는 저장 자료 부족으로 차단됨. 기존 CNN 재추론이나 신규 후보 진단 실행으로 범위를 넓히지 않았다. 자동 학습 루프를 재개하지 않는다.')
add('\n## 검산과 해석 오류 점검 — 완료')
add(f"\n원본 {len(x['source_files'])}개 SHA 보존과 분석 내 {x['scalar_checks']}개 수치 일치를 확인했다. 별도 Decimal 산식 구현으로 {v['scalar_equalities']}개 등식을 다시 검산했다. 같은 에이전트의 두 계산이며 독립 에이전트 검토나 모델 재현 평가로 표현하지 않는다. 원본은 전혀 수정하지 않았다.")
add('\n11개 통계 해석 오류 항목을 점검했다. Simpson형 집계 반전은 전체/구간/seed를 함께 공개해 구분했고, 집계에서 개별 사례를 추론하는 생태학적 오류는 행별 주장 금지로 처리했다. 선택 표본/Berkson 및 collider 우려는 사후 정답 구간의 인과 해석을 하지 않는 것으로 제한했다. 기저율은 구간별 대상 수로 표시했다. 극단적인 CNN42를 고른 평균회귀 우려는 모든 seed를 유지하고 원인으로 확정하지 않았다. 생존자 편향은 예정된 세 모델×세 seed 전부 포함 여부를 확인했다. 다중 탐색과 분석 경로 선택은 기존 다섯 수량 구간·세 이력 구간을 모두 보고하며 탐색적이라고 밝혔다. 상관의 인과 오인과 역인과는 미래 실제 수량을 입력 조건으로 쓰지 않는 것으로 제한했다. 이 점검은 자료가 없는 기전을 검증한 것이 아니다.')
add('\n## Validation 수량 구간의 세부 차이')
add('\nCapacity−CNN, ΔRMSE / ΔMAE. Test와 같은 경계다.')
add('\n| 실제 수량 구간 | 대상 수 | seed42 | seed52 | seed62 |\n|---|---:|---:|---:|---:|')
for i in range(1,6):
    cs=[contrast('validation',seed)[i] for seed in [42,52,62]]
    add(f"| {cs[0]['stratum']} | {cs[0]['count']} | "+' | '.join(f"{c['qty_rmse']:+.3f} / {c['qty_mae']:+.3f}" for c in cs)+' |')
with (O/'RESULT.md').open('x') as f:f.write('\n'.join(L)+'\n')
r=json.loads((O/'RECEIPT.json').read_text())
r['outputs'].update({n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['RESULT.md','VERIFICATION.json']})
r['auxiliary_scripts']={n:hashlib.sha256((O/n).read_bytes()).hexdigest() for n in ['verify.py','render_report.py']}
r['row_level_analysis_complete']=False
r['followup_training_justified']=False
(O/'RECEIPT.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print('Report written; result calculations preserved.')
