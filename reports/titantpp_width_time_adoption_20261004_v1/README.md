# 폭 대조 채택 기준과 train·validation 시간 진단

2026-10-04. **채택 기준 확정·고정 checkpoint 진단 완료.** 현재 후속 기준선은 사용자가 확정한 **폭16**이고 폭4는 비교군으로 보존한다. 이번 새 폭8·12 학습은 Taxi·Intermittent·RAF만 대상으로 하며 Instacart는 후속으로 미룬다. 이 문서는 기존 기준선이나 선택 checkpoint를 변경하지 않는다.

## 먼저 구분해야 할 사실

원래 선택 checkpoint의 **전수 validation 시간 NLL이 폭16에서 일괄 악화한 것은 아니다.** 아래 값은 새로운 표본 추정이 아니라 기존 validation-only selected endpoint를 파일 SHA·state SHA·선택 epoch로 대조한 3seed 평균이다. 표본 표준편차와 개별 seed는 `full_validation_metrics.csv` 및 `full_validation_seed_summary.csv`에 있다.

| 데이터 | 전체 RMSE 폭4 → 폭16 | MAE 폭4 → 폭16 | 큰 수량 RMSE 폭4 → 폭16 | 시간 NLL 폭4 → 폭16 | 전체 RMSE 개선 seed | 시간 NLL 개선 seed |
|---|---:|---:|---:|---:|---:|---:|
| Taxi | 79.711 → 77.826 | 25.624 → 25.178 | 384.982 → 376.664 | 1.0387 → 0.9949 | 2/3 | 2/3 |
| Intermittent | 1.673 → 1.687 | 0.701 → 0.714 | 8.189 → 7.924 | 0.4971 → 0.4045 | 1/3 | 1/3 |
| RAF | 33.951 → 34.083 | 9.251 → 9.264 | 325.837 → 329.857 | 3.5446 → 3.4992 | 0/3 | 3/3 |

큰 수량은 최초 TRAIN 경계의 초과 구간(Taxi >3449, Intermittent >187, RAF >200)이다. 경계와 같은 값은 아래 구간에 포함한다. 시간 NLL은 데이터별 원래 단위의 **양의 정수 관측 확률 질량** 점수이며 연속 밀도 점수가 아니다. 다른 데이터의 절댓값을 서로 순위 비교하지 않는다.

- Taxi에서는 네 평균 지표가 좋아졌지만 모든 seed에서 좋아진 것은 아니다.
- Intermittent의 큰 수량 RMSE는 세 seed 모두 개선됐다. 전체 RMSE는 세 seed 중 두 개가 악화하며 MAE 평균도 악화한다. 시간 NLL 평균 개선은 seed62의 큰 개선 영향이므로 안정적인 공통 개선으로 표현하지 않는다.
- RAF는 시간 NLL이 세 seed 모두 개선됐지만 전체·큰 수량 RMSE는 세 seed 모두 악화한다. 용량 증가가 모든 목적을 함께 해결했다는 주장에 해당하지 않는다.
- 이전 다른 split에서 본 시간 악화와 이 validation 값을 혼합하지 않는다. **이번 진단은 Test 예측·성능 또는 혼합 결과표를 열지 않았다.**

## 채택 기준

[adoption_criteria.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_width_time_adoption_20261004_v1/adoption_criteria.json)에 폭8·12의 새 결과가 나오기 전에 다음 기준을 고정했다.

1. **전체 validation raw 수량 RMSE**로 최초 최소 epoch를 선택한다. min40/max300/patience40을 유지하고, MAE·큰 수량·시간 점수마다 다른 checkpoint를 고르지 않는다.
2. 동일 checkpoint의 전체 RMSE, MAE, body/tail RMSE, 정수시간 NLL을 함께 기록한다. Train은 고정 checkpoint의 같은 표본에서 진단하며 학습 epoch 중간의 평균 train loss와 구분한다.
3. 각 데이터에서 3seed 평균·표본 표준편차(ddof1)뿐 아니라 seed별 짝지은 차이·개선 seed 수·가장 불리한 차이를 함께 표시한다. 평균만으로 seed 불안정성을 가리지 않는다.
4. 전체 RMSE 평균이 내려가고 최소2/3seed가 좋아져야 해당 데이터의 개선 후보로 우선 검토한다. MAE·큰 수량·시간 평균도 나빠지지 않으면 함께 개선된 후보, 일부가 나빠지면 **상충 관계가 있는 후보**로 분리한다. 이는 통계적 유의성 판정이 아니다.
5. Train만 좋아지거나 개선이 한 seed에만 의존하면 안정적인 용량 이득으로 채택하지 않는다. 누락된 seed·실패·비유한 수치·SHA 검증 실패가 있으면 3seed 결론을 보류한다.
6. 사용자 기준선 폭16은 자동으로 교체하지 않는다. 시간 악화를 허용하는 새로운 임의 수치 문턱이나 손실·선택 기준 변경을 이번 폭 대조에 넣지 않는다. 불리한 결과가 있으면 명시적인 상충 관계로 검토한다.

폭8·12는 입력·출력부·손실·초기화·분기 가용성·고정 /8을 그대로 두고 보정 폭만 바꾸므로, 다음 결과로 단순 용량의 이득과 손해를 구분할 수 있다. 폭 변화 자체를 새로운 Backbone 기여라고 부르지 않는다.

## 시간 예측에서 실제로 확인한 범위

폭4/16 × 3데이터 × 3seed의 **기존 Validation-selected checkpoint 18개**를 각각 원래 동결 소스에서 읽었다. 로컬 CPU2스레드에서 train·validation만 필터링한 다음 동일 수량×시간간격 층화 표본을 추론했다. 각 층에서 최대64건을 비복원 추출하고 `N층/n표본`으로 가중했다. 시간간격 경계 1/4/13/52는 원래 데이터별 시간 단위다. 어떤 시간간격 범위를 가리키는지 데이터마다 단위로 해석해야 한다.

| 데이터 | split | 고정 표본 수 / 전체 적격 target 수 |
|---|---|---:|
| Taxi | train / validation | 552 / 38,393 · 521 / 8,268 |
| Intermittent | train / validation | 768 / 393,824 · 747 / 86,285 |
| RAF | train / validation | 1,210 / 25,779 · 948 / 6,690 |

모든 모델과 seed가 같은 target·시간·수량·가중치·이력 길이를 사용한다. 시간 출력부의 **log-duration location μ와 조건부 sigma σ**, 실제 정수시간 NLL, 표준화 잔차를 함께 기록했다. 미래 target을 바꿔도 수량 예측과 μ·σ가 그대로인지도 검사했다. 모델 파라미터는 실행 전후 같았다.

**확인된 사실:** Intermittent의 조건부 sigma는 두 폭 모두 작으며, 폭16에서 모든 seed의 sigma가 더 작아진 것이 아니다.

| Intermittent seed | train sigma 평균 폭4 → 폭16 | validation sigma 평균 폭4 → 폭16 | 전수 validation NLL 폭4 → 폭16 |
|---|---:|---:|---:|
| 42 | 0.10964 → 0.12094 | 0.09822 → 0.10395 | 0.34683 → 0.35094 |
| 52 | 0.11489 → 0.11206 | 0.10331 → 0.09755 | 0.40812 → 0.48235 |
| 62 | 0.09436 → 0.10142 | 0.08146 → 0.09011 | 0.73638 → 0.38013 |

위 sigma 평균은 층화 **표본 가중 추정**이다. 전체 모집단이 아니다. 원래 공통 초기 sigma는0.681642이며 floor는0.001이다. 조건부 분포가 초기보다 좁아졌다는 사실만으로 과신이나 보정 실패를 확정하지 않는다. 시간간격1은 `T≤1.5`의 누적 확률 질량을 받으므로 단순 `|log(d)-μ|/σ`가 커도 실제 NLL이 반드시 커지지 않는다. 정수 관측 확률을 실제로 재평가해야 한다.

**수학적 분해:** 같은 표본에서 폭4의 `(μ4,σ4)`, 폭16의 `(μ16,σ16)`과 교차 조합을 정수 확률 질량으로 평가했다.

\[
\Delta L=[L(\mu_{16},\sigma_4)-L(\mu_4,\sigma_4)]
+[L(\mu_4,\sigma_{16})-L(\mu_4,\sigma_4)]+I.
\]

`I`는 두 변화의 상호작용이다. 이 항을 생략하면 location과 sigma가 독립적으로 원인을 설명한다고 오해하기 쉽다. 18개 dataset·seed·split 조합에서 합이 실제 짝지은 NLL 차이를 복원했고, float64 정수 mass 재계산도 원래 forward 결과와 일치했다.

Intermittent validation 표본의 경우 seed42는 location 교체 항 +0.17381, sigma 교체 항 −0.00513, 상호작용 −0.09030으로 총 +0.07838이었다. seed52는 sigma 교체 항이 양수였지만 location과 상호작용이 이를 상쇄했고, seed62는 location 교체 항이 주된 개선 방향이었다. **‘폭16이 sigma를 일괄 축소시켜 시간 NLL을 망쳤다’는 단일 설명은 이 진단으로 지지되지 않는다.**

표본에는 희귀한 큰 NLL 사건이 있을 수 있다. 실제로 Intermittent seed52의 표본 NLL 차이는 −0.04999인데 전수 차이는 +0.07422다. 따라서 표본 분해를 전수 원인 기여율로 보고하지 않는다. `paired_diagnostics.csv`에 유한모집단 층화 표본설계의 시간 NLL 표준오차도 함께 기록했다. 이는 seed·개체·미래 일반화 불확실성의 구간이 아니며 미관측 극단치를 완전히 해결하지 않는다.

**미확인:** 특정 split에서 시간 악화가 발생한 학습 원인, 시간·수량 gradient 충돌의 인과성, 더 넓은 MLP의 과적합 여부, CNN/GRU가 이 문제를 해결하는지. 현재 선택이 수량 RMSE 한 지표에 따른다는 사실 때문에 시간의 최적 epoch와 같을 필요가 없지만, 그것만으로 악화 원인을 확정하지 않는다. 해당 문제를 분리하려면 동일 고정 checkpoint에서 전수 시간 파라미터 진단 또는 별도 시간 head 대조가 필요하며 이번 새 폭 학습의 계약을 바꾸지 않는다.

## 검증·원본 연결

- 동결 진단 계약: [contract.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_width_time_adoption_20261004_v1/contract.json)
- 원래 **전수 validation**: [full_validation_metrics.csv](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_width_time_adoption_20261004_v1/full_validation_metrics.csv), [3seed 집계](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_width_time_adoption_20261004_v1/full_validation_seed_summary.csv), [원본 SHA·선택 연결](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_width_time_adoption_20261004_v1/validation_source_bindings.json)
- **표본 train·validation**: [paired_diagnostics.csv](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_width_time_adoption_20261004_v1/paired_diagnostics.csv), 각 `runs/<dataset>__width<width>__seed<seed>/receipt.json`와 train/validation parquet. 원자료와 checkpoint는 변경하지 않았다.
- 코드: [diagnose_titantpp_width_time_train_validation.py](/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/diagnose_titantpp_width_time_train_validation.py)
- 검증: [verification.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_width_time_adoption_20261004_v1/verification.json). 표본 경계·희귀 시간구간 보존·가중치·상호작용 합·비유한 값 거부·유한모집단 표준오차의 **계약 테스트9개 통과**.

## 남은 작업

**폭8·12의 실제 용량 대조를 완료한다 — 진행 중인 캠페인과 연결**
- 대상: 5080·5090의 Taxi·Intermittent·RAF × 폭8/12 × seed42/52/62. 이 보고서의 원격 실행 상태는 별도 launch receipt가 기준이다. 이미 완료한 폭4/16 학습은 재사용한다.
- 동일 selected checkpoint의 전체·body·큰 수량 RMSE, MAE, 시간 NLL과 각 seed를 대조하고 불리한 조건도 남긴다. 완료 조건은18조건의 원본·선택·validation 재평가 검증이다.

**CNN·GRU 대조에서 무엇을 해결할지 연결한다 — 다음 작업**
- 대상: Encoder1 및 중간 보정의 설계 계약. 수량 표현을 개선하면서 시간 성능과 seed 안정성을 숨기지 않는 것을 확인 목표로 둔다.
- CNN만·GRU만·둘의 결합을 분리하고, 참조 범위·state reset·파라미터·연산비용을 기록한다. 손실·출력부 변경은 이번 폭 결과와 섞지 않는다.
