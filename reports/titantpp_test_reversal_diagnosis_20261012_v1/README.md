# 기존 Test 반전 원인 진단과 최소 후속 구성

2026-10-12. 대상은 paper_research의 CNN+GRU54 이력 보정27이다. **기존 결과의 오류 위치·입력 구성·학습 기록 분석은 완료했고, 새 학습·모델 추론·원격 호출은 0회다.** 기존 Test는 생성·접근 이력이 있는 평가이며 새 독립 확인이 아니다. 열린 논문 `main.tex`는 이번 계획 작업에서 수정하지 않았다.

권고는 추가 검색이나 seed 확대보다 **동결된 선택 checkpoint27개의 진단 계측을 먼저 하는 것**이다. 이후에도 CNN·GRU 각각의 필요성을 논문 기여로 주장하려면 RAF에서 네 구조×3seed=12개의 새 학습을 조건부로 검토한다. 이 12개는 현재 실행 대상으로 확정한 것이 아니다.

## 반전이 어디서 발생했는지 구분한다 — 완료

아래 차이는 동일 seed에서 full−대조다. 양수는 악화다. ‘보정 제거’는 CNN·GRU 결합 경로를 함께 우회한 조건이다. 정확한 전체18쌍과 여섯 지표의 출처는 [수량 진단](QUANTITY_DIAGNOSIS.md), [원래 최종 비교](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/analysis/TEST_FINAL_COMPARISON.md)에 있다.

| 대상 | 확인된 오류 위치 | Test 전체 RMSE 차이: 제거 / current-only | 현재 설명할 수 있는 범위 |
|---|---|---:|---|
| Taxi seed52 | 실제 수량>3449의83건(전체0.997%)에서 과소예측. full 평균 오차−888.407, 제거 대조+128.055 | +0.859 / +33.359 | 제거 대조 대비 큰 수량 손해가 나머지 구간 이득을 뒤집음. current-only 대비 손해도 큰 두 구간에 집중 |
| Intermittent seed42 | 주로2<실제 수량≤31의33,203건. full 평균 오차+1.684, 대조+1.172/+1.312 | +0.334 / +0.107 | 큰 수량보다 중간 수량 과대예측과 오차분산 증가가 중심. 큰 수량 RMSE는 양대조보다 개선 |
| RAF seed42/52/62 | 실제 수량>200의39건(전체0.746%)에서 대조보다 과소예측. full 평균 오차−354.648/−369.052/−355.821 | +0.228/+0.822/+0.510 / +0.277/+0.712/+0.436 | 세 seed 모두 전체 RMSE 악화. 큰 수량이 양의 구간별 총손해49–100%를 차지하지만 RAF42의 작은 수량 손해도 함께 존재 |

Taxi52의 제거 대조 대비 ΔMSE는 `+4620.563(큰 수량)−4354.452(나머지)=+266.111`이다. RMSE는 구간별로 합산할 수 없어 MSE로 분해했다. 큰 수량에서는 오차분산이 줄어도 과소예측의 편향² 증가가 더 컸다. RAF의 여섯 큰 수량 비교도 같은 방향이다. 여기서 편향은 실제 수량에 조건을 건 평균 오차이며 예측값 calibration이나 seed 간 학습 분산을 측정한 것이 아니다.

Taxi42·62와 Intermittent52·62는 같은 대상을 평가했는데도 두 대조보다 개선했다. 따라서 특정 자료의 모든 seed 실패나 ‘큰 수량 실패’ 하나로 두 반전을 묶을 수 없다. RAF52는 Validation에서도 두 대조보다 나빴으므로 RAF의 세 seed를 모두 Validation→Test 반전이라고 쓰지 않는다.

## 입력 변화와 학습·선택 기록을 함께 확인한다 — 완료

| 자료 | Validation→Test에서 확인된 변화 | 원인 후보와 아직 필요한 구분 |
|---|---|---|
| Taxi | 동일131series, series별 대상 비중 거리0.0037. 큰 수량은 두 split 모두4series에서 발생 | 특정 큰 수량 series·입력 상황과 seed별 보정 출력의 상호작용 후보. 전체 구성 변화만으로 seed52만 실패한 이유를 설명하지 못함 |
| Intermittent | 수량 평균−37.76%, recorded gap 평균+30.96%, 이력 평균113.91→135.72. Test3,655개만 sequence cap에 의해 과거가 잘림 | 긴 과거·현재 수량 불일치, truncation, 시간 head 취약성을 각각 계측해야 함. 구성 변화가 seed42만 반전시킨 인과 원인이라고 확정하지 않음 |
| RAF | 큰 수량 비중0.7474%→0.7463%로 거의 동일. series 비중 거리0.5030. Test879series는 Validation에 없지만 모두 Train에는 존재. 이력 최대14/16/19(Train/Val/Test) | 평가 series 구성과 학습 지원 밖의 recurrent 입력 길이 후보. 큰 수량39건이 이력>14 또는879series에 속하는지는 현재 저장 오류표로 연결되지 않음 |

모든 Test 악화10쌍은 Test를 Validation의 고정5개 수량 구간 비중으로 재가중해도 악화한다. **수량 구간 비중 변화 하나만으로는 설명이 부족하다.** 구간 내부의 수량·개체·시점·이력 변화는 남아 있으므로 이것을 ‘분포 변화가 없다’는 증거로 해석하지 않는다. 입력 주변분포 변화와 조건부 관계 변화는 별개다. [Ryan & Culp, JMLR 2015](https://www.jmlr.org/papers/v16/ryan15a.html)

27개 원본 학습은 finite이며 최초 strict 최소 Validation raw 수량 RMSE 선택과 실제 `마지막 epoch=선택+40`이 일치했다. Taxi 반전 full52는epoch74로, 개선 full42의126보다 이르다. Intermittent full은15/11/13, RAF는9/5/20에서 선택됐다. RAF42의 세 arm은 모두epoch9다. 늦은 선택 하나로 악화를 설명할 근거는 없다. 그렇다고 Validation 선택의 민감성이나 과적합 가능성이 배제된 것은 아니다. 유한한 Validation에서 선택을 반복하면 선택 기준 자체에 맞춰질 수 있으므로 Test로 좋은 epoch를 다시 고르지 않는다. [Cawley & Talbot, JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

동일 dataset/seed의 입력·초기화 map·공통 학습 인자는 일치했다. Test36의 공통5080 Runtime에서 원래 Validation을 재현했고 history27 전체 지표의 최대 상대오차는2.65e-7이다. RAF52의 같은 A100 학습과 RAF62의 같은5080 학습도 악화했으므로 RAF42 물리 A100 이관만으로 전체 악화를 설명할 수 없다. 이 검증은 모든 구현 버그의 부재를 증명하지 않는다.

시간 손해도 분리해야 한다. Intermittent42 full은 선택 Validation TimeNLL0.6187로 양대조0.3197보다 이미 불리했고 Test는62.5721이다. 시간 위치·scale과 recorded gap별 기여가 없어서 수량 반전과 같은 원인인지 미확인이다. raw 수량 선택은 시간 성능을 보장하지 않는다. 학습 log-domain 수량 loss 감소를 raw 수량 과적합의 직접 증거로 쓰지 않는다.

입력 대상·history digest와 모든 계산 출처는 [입력 구성 진단](POPULATION_DIAGNOSIS.md), 선택·Runtime·loss 증적은 [학습 진단](TRAINING_DIAGNOSIS.md)에 있다.

## 구조에 대한 잘못된 원인 설명을 제거한다 — 완료

- 동결 CNN+GRU54의54는 GRU hidden 폭이다. GRU 보정은 유효 관측2개 이상에서 적용되며 lag16/32/64/128 활성화 gate나 count 정규화가 없다. RAF Train 최대14→Test19는 recurrent state·깊이의 입력 지원 차이 후보이며 ‘lag16 경로가 새로 켜진 탓’이 아니다.
- CNN은 관측 current/lag1/lag2의 Q/K/V 보정이다. GRU는 현재 forward의 유효 prefix를 순환 처리한다. 저장 결과에는 두 경로의 실제 보정 크기나 부호가 없어 어느 경로가 손해를 유발했는지는 미확인이다.
- current-only의 hidden에도 backbone prefix 이력이 있다. full과 활성 보정25,308개는 같지만 계산 경로·깊이·활성화가 다르므로 이 비교가 정보량 하나만 완벽하게 분리하지는 않는다.
- 기존 `titantpp_cnn_mlp16`은 CNN+MLP16이며 진짜 CNN-only 대조가 아니다. 네 구조 학습을 한다면 CNN만/GRU만/둘 다/둘 다 제거를 별도로 정의해야 한다.

근거는 원래 Test bundle의 [동결 구조](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/sources/history/models/TPPs/CountAwareTitanCNNGRU.py)와 [동결 보정 대조](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/sources/history/models/TPPs/CountAwareTitanCNNHistoryControls.py)다.

## 논문을 보완하는 최소 후속 작업을 결정한다 — 다음 작업

1. **선택27개 진단 계측, 신규 fit0:** 같은 대상에서 오류·시간 head·이력·CNN/GRU 출력 관계를 확인한다. 계측 전후 전체 지표·대상 digest가 재현돼야 한다. Validation/Test 각27개, 총54 split pass다.
2. **필요할 때만 경로 우회 진단, 신규 fit0:** full9의 CNN만 우회/GRU만 우회18설정은 저장 모델 출력 의존성을 보여주는 개입이다. 제거 구조를 독립 학습한 실험이나 구성요소 필요성 증명으로 부르지 않는다.
3. **구성요소 기여를 주장할 때만 RAF12fresh:** 네 구조×3seed를 같은 GPU/Runtime에서 새로 학습한다. 기존 A100/RTX full·제거 결과를 재사용해 ‘6개 추가로 완성한 동일 환경 대조’라고 부르지 않는다. 더 좁은 결합 경로의 조건부 관찰로 논문을 쓰면 이 단계는 생략 가능하다.
4. **최종 모델 동결 뒤 미노출 확인:** 추가 설계에 기존 Test를 이용하면 후속 개발 자료로 표시한다. 새 자료의 사용 이력·호환성과 평가 계약을 먼저 확정하고 최종 성과를 확인한다. 모델을 바꾸지 않았다면 기존 선택 모델의 추론만으로 가능한지 검토한다.

상세 대상·승인·중단 조건은 [최소 실험 계획](MINIMUM_EXPERIMENT_PLAN.md), 기계 판독 계획은 [plan.json](plan.json)에 있다. 검색·loss/LR sweep·실패 seed만 재학습·Test epoch 재선택은 이번 최소 구성에 포함하지 않는다. 이 계획은 실행 승인이 아니다.

## 원본과 검증 범위를 남긴다 — 완료

- Test 계약 canonical: `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c`.
- history source129 closure: `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, frozen revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`.
- 완료 receipt rawSHA: `d21d137190e98552d07c5b9e2d8fc31151846e39c782ad907ff9a1a3cf644dbd`.
- 원본 archive rawSHA: `bf7aa41942420966dc612cd1b8a8806733cf38abeb373dea127558782dbfb3ca`; manifest rawSHA: `b25c0e2c6e39bc8f15d2f7569ad81ed1587c8d1fdfcb907fd1bb451ef135ff27`.
- [수량 계산 JSON](quantity_diagnosis.json), [대상 구성 JSON](population_diagnosis.json), [학습 기록 JSON](training_diagnosis.json)에 절대 원본 경로·SHA와 계산 정의를 남겼다. [독립 검토](DIAGNOSIS_QA.md)와 최종 SHA 목록은 검토 완료 후 함께 봉인한다.

이번 분석은 오류 위치를 설명하는 회계적 분해와 원인 후보의 축소다. 구조의 인과적 필요성, 통계 유의성, 참신성, 외부 방법보다의 Test 우월성 또는 PAKDD 채택 가능성을 입증한 것으로 확대하지 않는다. 외부 인코더 비교의 full9는 별도 학습 기준선이므로 이번 보정 full9로 외부 비교 결과를 교체하지 않는다.
