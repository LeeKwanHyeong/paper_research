# 2026년 6월 28일 이후 TitanTPP 진행 결과 및 면담 자료

- 실험 결과 기준일: 2026년 7월 19일
- 문서 개정일: 2026년 7월 21일
- 보고 범위: 2026년 6월 28일 이후 완료한 분석과 모델 강화 실험
- 현재 비교 기준: Demand·Instacart는 TitanTPP V2, Taxi는 TitanTPP V3b
- 남은 실험: RMTPP·TitanTPP·THP의 최종 e800 공정 비교는 모델과 조건만 확정했으며 아직 실행하지 않음

## 1. 먼저 읽을 용어

이 보고서의 V1, V2, V3b와 `small_lmm`, `mid_lmm`은 논문에서 일반적으로 쓰이는 명칭이 아니라 이번 연구에서 실험을 구분하기 위해 붙인 내부 이름이다. 번호가 높다고 항상 우수한 모델이라는 뜻은 아니며, 사전에 정한 검증 기준을 통과한 모델만 현재 비교 기준으로 유지했다.

| 용어 | 이 보고서에서의 의미 |
|---|---|
| RMTPP | 과거 사건을 순환신경망으로 요약하고 다음 사건의 종류(mark)와 발생 시간(time)을 확률적으로 예측하는 기준 모델 |
| TitanTPP | RMTPP의 mark·time 확률 예측 구조는 유지하면서, 과거 사건을 요약하는 encoder를 Titan 계열 backbone으로 바꾼 모델 |
| THP | Transformer Hawkes Process. 과거 사건을 Transformer로 요약하는 temporal point process 비교 모델 |
| GRU h64 / h128 | RMTPP encoder로 쓰는 GRU의 hidden dimension이 각각 64 또는 128이라는 뜻 |
| d_model 64 / 128 | THP 또는 Titan encoder가 사건 하나를 표현하는 내부 벡터의 크기. 값이 클수록 표현 용량과 계산량이 함께 증가 |
| mark | TPP가 예측하는 다음 사건의 범주. 이번 실험에서는 수량 규모를 구간화한 클래스로 사용 |
| residual value | 정답 수량이 해당 mark 구간 안에서 어느 위치에 있는지를 나타내는 연속값 |
| V1 | 과거 residual value를 encoder 입력에 추가하고, 정답 mark 구간 내부의 residual 오차만 학습하는 초기 TitanTPP |
| V2 | V1에 실제 수량 오차를 직접 반영하는 quantity loss를 추가한 모델. Demand와 Instacart의 현재 기준이며 Taxi에서는 구조 비교용 기준 |
| V3b | mark마다 별도의 residual 예측기를 두고, quantity loss가 mark 분류기를 직접 흔들지 않도록 gradient 경로를 분리한 모델. 현재 Taxi에서만 채택 |
| LMM | Local Memory Matching. 학습 가능한 memory bank에서 현재 이력과 가까운 표현을 찾아 encoder 표현에 보강하는 구성 |
| `small_lmm` | 작은 용량의 Titan preset. hidden dimension 64, 2개 encoder layer, memory 64개와 top-k 4를 사용 |
| `mid_lmm` | 중간 용량의 Titan preset. hidden dimension 128, 2개 encoder layer, memory 128개와 top-k 8을 사용 |
| candidate / preset | encoder 크기와 memory 설정을 묶은 사전 정의 구성. `small_lmm`과 `mid_lmm`은 서로 다른 preset이므로 둘을 비교하면 순수한 단일 요인 ablation은 아님 |
| e50, e200, e800 | 각각 최대 50, 200, 800 epoch를 학습하는 실험 예산 |
| seed | 가중치 초기화와 학습 순서를 재현하기 위한 난수값. 여러 seed에서 같은 방향이 나오는지 확인해 우연한 결과를 줄임 |
| lookback | DataLoader가 입력 창을 만들 때 사용하는 과거 범위. 단위와 구성은 데이터 경로에 따라 달라짐 |
| `max_seq_len` | 한 입력에서 모델이 실제로 받을 수 있는 사건 수의 상한. lookback과 같은 개념이 아님 |
| marker NLL / time NLL | 각각 다음 mark와 다음 발생 시간에 대한 negative log-likelihood. 두 값을 합한 것이 total NLL이며 낮을수록 좋음 |
| best validation NLL | validation total NLL이 가장 낮은 epoch의 checkpoint. 모델 선택은 validation으로만 하고 test 결과를 보고 다시 고르지 않음 |
| 종합 score | 기존 실험에서 NLL과 수량 오차 등을 함께 보기 위해 사용한 보조 지표. 단독 모델 선택 기준은 아니며 주 선택 기준은 best validation NLL |

## 2. 핵심 결론

1. 학습률을 `1e-3`에서 `5e-3`으로 높이면 일부 설정에서 수렴이 빨라졌지만, TitanTPP의 일부 preset에서는 NaN이 발생했다. 따라서 낮은 학습률은 장기 학습의 일부 원인이지만 학습률만 높여 해결되는 문제는 아니었다.
2. Taxi e1000 분석에서 RMTPP와 TitanTPP 모두 최적 epoch 이후 validation 성능이 악화되는 구간이 확인됐다. 다만 time NLL이 계속 낮아지는 동안 marker NLL은 나빠질 수 있어, 이후에는 total NLL을 marker와 time 항으로 나누어 해석했다.
3. V2는 V1보다 세 데이터셋 모두에서 quantity MAE를 낮췄다. 반면 Instacart와 Taxi의 test NLL은 각각 0.04%, 1.23% 악화됐으므로 V2를 모든 지표에서 우수한 모델로 해석하지 않는다.
4. Taxi V3b는 V2와 모델 용량, 학습 예산, seed, 입력 길이와 checkpoint 기준을 맞춘 비교에서 세 seed 모두 total NLL, marker NLL, quantity MAE와 mark accuracy를 개선했다. 따라서 Taxi에서만 V3b를 채택하고 Demand와 Instacart는 V2를 유지한다.
5. 현재 결과는 외생변수를 추가한 실험이 아니다. 확인된 기여는 RMTPP의 확률 예측 구조를 유지하면서 사건 이력 표현과 수량 예측 구조를 확장한 것까지다.

## 3. 교수님 의견에 대한 확인 결과

### 3.1 학습률을 5배·10배 높였을 때

![F1. 학습률 민감도](./F1_learning_rate_stability.png)

F1은 e50, seed 42에서 수행한 학습률 진단이다. Instacart RMTPP의 best validation NLL epoch는 학습률 `1e-3`, `5e-3`, `1e-2`에서 각각 26, 5, 26이었다. `5e-3`에서는 수렴이 빨라졌지만 `1e-2`까지 높였을 때 같은 효과가 유지되지는 않았다.

TitanTPP는 encoder preset과 quantity loss 설정에 따라 안정성이 달랐다. 일부 조합은 높은 학습률에서도 완료됐지만 일부는 epoch 4, 10, 28 등에서 NaN이 발생했다. 이 결과를 바탕으로 이후 기준 학습률은 안정성이 확인된 `1e-3`으로 유지했다.

따라서 교수님께서 제시하신 “학습률이 너무 작아 수렴이 늦을 수 있다”는 가설은 일부 확인됐다. 다만 TitanTPP에서는 높은 학습률이 수렴 속도와 학습 안정성 사이의 trade-off를 만들었으므로, 최종 비교에서는 epoch 수뿐 아니라 checkpoint 선택과 NaN 여부를 함께 관리해야 한다.

### 3.2 e1000에서 손실을 분리해서 본 결과

![F2. e1000 NLL 분해](./F2_e1000_nll_decomposition.png)

Taxi e1000에서 선택한 RMTPP h128과 TitanTPP `mid_lmm`의 best validation NLL epoch는 각각 861과 987이었다. 마지막 epoch의 validation NLL은 최적 지점보다 RMTPP에서 0.0873, TitanTPP에서 0.0406 높았다. 최소한 이 설정에서는 두 모델 모두 과적합할 수 있는 충분한 용량이 있었다.

중요한 점은 total NLL만으로 이 변화를 설명하기 어렵다는 것이다. 최적 epoch 이후 time NLL이 더 낮아져도 marker NLL이 악화될 수 있었다. 이후에는 total NLL, marker NLL, time NLL을 분리하고 quantity MAE, mark accuracy, time error를 함께 확인했다. 또한 best validation checkpoint와 마지막 epoch를 구분해 기록했다.

F1과 F2는 교수님 의견에 답하기 위한 seed-42 진단 실험이다. 최종 모델 우위를 주장하는 근거로는 사용하지 않는다.

## 4. RMTPP에서 TitanTPP V2·V3b로의 변경

![F3. 모델 설계 변화](./F3_model_architecture_evolution.png)

| 모델 | 확률 예측 구조 | 사건 이력과 수량 처리 | 현재 역할 |
|---|---|---|---|
| RMTPP | mark와 time likelihood | 순환신경망으로 과거 사건 요약 | 외부 기준 모델 |
| TitanTPP V1 | RMTPP의 mark·time head 유지 | Titan encoder와 residual 입력·예측 사용 | 초기 Titan 기준 |
| TitanTPP V2 | V1과 동일 | residual loss에 실제 quantity loss 추가 | Demand·Instacart 기준, Taxi control |
| TitanTPP V3b | V2와 동일 | mark별 residual 예측과 quantity-to-mark gradient 분리 | Taxi 채택 모델 |

V2는 가능한 mark별 수량과 예측 mark 확률을 결합해 다음 수량의 기댓값을 계산하고, 그 값과 실제 수량의 차이를 학습에 반영한다. 이 quantity loss가 mark 분류 확률에도 영향을 주기 때문에 수량 회귀와 mark 분류가 충돌할 수 있다.

V3b는 mark별로 서로 다른 residual을 예측하고, quantity loss를 계산할 때만 mark 확률의 gradient를 차단한다. marker NLL은 기존대로 mark head를 학습하므로 분류 학습 자체를 멈추는 구조는 아니다. 즉 V3b는 RMTPP의 확률 모형을 바꾸기보다 mark와 quantity 사이의 학습 간섭을 줄이는 변경이다.

## 5. 완료한 모델 비교

### 5.1 V1 대비 V2

![F4. V1 대비 V2](./F4_v1_v2_dataset_comparison.png)

V1과 V2는 e200, seeds 42·52·62로 학습했다. 각 버전 안에서 mean best validation NLL이 가장 낮은 encoder preset을 선택한 뒤 test split을 한 번 평가했다.

| 데이터셋 | V1 → V2 선택 preset | 종합 score 변화 | test NLL 변화 | quantity MAE 변화 |
|---|---|---|---|---|
| Instacart | `mid_lmm` → `small_lmm` | +0.000377 | +0.040% | -1.041% |
| Demand | `small_lmm` → `small_lmm` | +0.007727 | -0.316% | -9.943% |
| Taxi | `small_lmm` → `mid_lmm` | +0.016058 | +1.233% | -25.858% |

V2는 세 데이터셋에서 quantity MAE를 모두 낮췄다. 그러나 Instacart와 Taxi의 test NLL은 소폭 악화됐고, 두 데이터셋에서는 V1과 V2의 선택 preset도 달랐다. 따라서 이 결과는 objective 하나만 바꾼 순수 ablation이 아니라 각 버전의 validation 선택 결과를 비교한 것이다. V2는 quantity-aware 기준으로 유지하되 “V1보다 모든 면에서 우수하다”고 표현하지 않는다.

### 5.2 Taxi V2 대비 V3b

![F5. Taxi V2 대비 V3b](./F5_taxi_v2_v3b_multiseed.png)

Taxi 비교에서는 V2와 V3b 모두 `mid_lmm`, e50, seeds 42·52·62, lookback 168, `max_seq_len` 256, batch size 128을 사용했다. checkpoint도 동일하게 best validation NLL로 선택했다. 즉 이 비교에서는 모델 용량과 학습 조건을 고정하고 value head와 gradient 경로의 차이만 확인했다.

| 지표 | V3b의 V2 대비 변화 |
|---|---|
| total NLL | -2.335% |
| marker NLL | -16.448% |
| time NLL | +0.181% |
| quantity MAE | -49.086% |
| residual value MAE | -27.303% |
| mark accuracy | +0.729%p |

total NLL, marker NLL, quantity MAE와 mark accuracy는 세 seed 모두 같은 방향으로 개선됐다. 반면 time NLL은 평균 0.181% 악화됐다. 따라서 V3b의 이득은 time model의 개선이 아니라 mark와 quantity를 함께 학습하는 방식을 바꾼 데서 나온 것으로 해석한다. 이 결과는 Taxi에만 적용하며 Demand와 Instacart로 일반화하지 않는다.

## 6. 모델 강화 결과의 현재 상태

![F6. 모델 선택 과정](./F6_model_selection_flow.png)

V2 이후에는 새로운 구조를 계속 누적하지 않고, 각 가설이 현재 기준 모델을 정해진 성능 기준으로 넘어서는지 확인했다.

| 데이터셋 또는 가설 | 현재 판단 | 근거 |
|---|---|---|
| Demand | V2 `small_lmm` 유지 | mark loss 변경과 gradient 분리 후보가 mark 성능을 회복하지 못함 |
| Instacart | V2 `small_lmm` 유지 | 후속 e1 실험은 구현 확인용이었으며 품질 승격 근거가 아님 |
| Taxi | V3b `mid_lmm` 채택 | 동일 조건의 3-seed 비교에서 V2 대비 marker와 quantity 지표 개선 |
| 직접 수량 회귀·RevIN 계열 | 미채택 | 큰 수량 오차는 줄어든 경우가 있었지만 표본이 많은 작은 수량 구간과 mark 예측이 함께 손상됨 |
| mark class-prior 보정 V5b | 설계만 완료 | 아직 구현·성능 실험을 하지 않아 현재 모델로 볼 수 없음 |
| lookback 이전 이력 추가 V6·V7 | 종료 | train-only 사전 진단이 구현 진행 기준을 충족하지 못함 |

RevIN 계열 결과는 “RevIN이 수량 예측에 일반적으로 효과가 없다”는 뜻이 아니다. 현재 sparse event 구성과 시험한 normalization 방식이 V2보다 안전한 개선을 만들지 못했다는 범위로만 해석한다.

## 7. RMTPP·TitanTPP·THP 최종 비교 조건

최종 비교에서는 Titan encoder의 효과와 quantity input·loss의 효과를 구분하기 위해 기존 RMTPP와 quantity 조건을 맞춘 RMTPP를 함께 둔다.

| 비교군 | 비교 목적 |
|---|---|
| RMTPP-R0 | 기존 RMTPP 구성 자체를 외부 기준으로 사용 |
| RMTPP-matched | TitanTPP와 동일한 quantity input·loss를 적용해 encoder 차이를 분리 |
| THP-matched | Transformer 계열 history encoder와 비교 |
| TitanTPP | Demand·Instacart는 V2, Taxi는 V3b를 평가 |

공통 실행 조건은 epochs 800, seeds 42·52·62, learning rate `1e-3`, batch size 128과 고정 train/validation/test split이다. checkpoint는 best validation NLL만 사용하며 test 결과를 본 뒤 모델 설정이나 epoch를 다시 고르지 않는다.

| 데이터셋 | RMTPP | THP | TitanTPP | lookback | `max_seq_len` |
|---|---|---|---|---|---|
| Demand | GRU h64 | d_model 64 | V2 `small_lmm` | 52 | 16 |
| Instacart | GRU h64 | d_model 64 | V2 `small_lmm` | 52 | 64 |
| Taxi | GRU h128 | d_model 128 | V3b `mid_lmm`, V2 병기 | 168 | 256 |

이 절의 모델과 실행 조건은 확정했지만 e800 실험은 아직 수행하지 않았다. 따라서 RMTPP·TitanTPP·THP의 최종 우위는 해당 실험이 끝난 뒤에만 판단한다.

## 8. 현재 제안하는 연구 기여와 면담 안건

현재 실험이 직접 뒷받침하는 기여는 다음과 같이 정리할 수 있다.

> RMTPP의 next-mark와 next-time likelihood는 유지하면서 Titan 계열 history encoder와 quantity-aware objective를 결합하고, mark 분류와 수량 회귀 사이의 gradient interference를 완화하는 구조를 검증했다.

현재 모델 입력은 이미 관측된 사건 이력의 mark, inter-event time과 quantity다. 날씨, 프로모션, 공급 차질과 같은 별도 외생변수는 포함하지 않았다. 따라서 현 단계에서 “외생변수의 영향을 반영한 모델”이라고 표현하는 것은 정확하지 않다.

면담에서는 다음 두 가지를 결정받고자 한다.

1. 현재 범위에서 RMTPP의 사건 이력 표현과 수량 예측 구조를 개선한 연구로 논문을 마무리할지 확인한다.
2. 외부 충격이나 튀는 수요를 핵심 기여로 삼는다면 별도 exogenous feature branch와 새로운 대조 실험까지 연구 범위를 확장할지 결정한다.

## 9. 해석 시 유지할 제한

- F1과 F2는 seed-42 진단이므로 최종 모델 비교 결과가 아니다.
- F4의 Instacart와 Taxi는 V1과 V2의 선택 preset이 달라 순수한 단일 요인 ablation이 아니다.
- F5의 V3b 결과는 Taxi의 세 seed에서만 확인됐다.
- V5b는 설계만 완료했으며 성능 근거가 없다.
- RMTPP·TitanTPP·THP 최종 e800 비교는 실행 전이다.
- 관측된 사건 이력을 외생변수라고 부르지 않는다.
- total NLL 개선만으로 marker와 quantity 성능도 좋아졌다고 단정하지 않는다.

## 10. 관련 기록

### 1. Summarize of Core & Similar Concept

- [1. Summarize of Core & Similar Concept](https://app.notion.com/p/2e4bbe4056138158a039ec8a9d3c7db8): RMTPP와 temporal point process의 기본 개념
- [Summarize RMTPP](https://app.notion.com/p/2e4bbe40561381c5a442d6e1dace7e18): RMTPP의 mark 확률과 time intensity

### 2. Confirm and Refine Topic

- [2. Confirm and Refine Topic](https://app.notion.com/p/2e4bbe40561380c19f09f2a0799efc6e): 연구 대상과 구현 범위
- [RMTPP 구현 정리](https://app.notion.com/p/2f9bbe40561380c289a8c47827bc1efc): TitanTPP에서도 유지하는 RMTPP head와 NLL
- [Data preparation and split](https://app.notion.com/p/2e4bbe4056138097a53fc95468c8887e): 사건 sequence와 데이터 분할
- [Hyperparameter and RMTPP comparison](https://app.notion.com/p/358bbe40561380c2a20ac7b602e1012d): 장기 학습과 하이퍼파라미터 검토

### 3. Identify Similar Papers and Specify Contribution

- [3. Identify Similar Papers and Specify Contribution](https://app.notion.com/p/2e4bbe40561380a3bbccfbeeae08902e): 관련 연구와 기여 정리
- [교수님 피드백 기반 Contribution 정리](https://app.notion.com/p/390bbe405613817abf93e494caf963e1): TitanTPP 자체보다 RMTPP 구조 개선으로 표현하는 근거

### 4. Fixed Experimental Design

- [4. Fixed Experimental Design](https://app.notion.com/p/2e8bbe4056138004bc28d9ce611e9e16): 데이터와 비교 규칙
- [NLL 분해](https://app.notion.com/p/37abbe40561381f7be8ff0a911a182e9): total NLL을 marker와 time으로 나눈 이유
- [Taxi e1000 overfit](https://app.notion.com/p/37abbe4056138179975bfd003bace1f3): 모델 용량과 최적 epoch 이후 악화
- [Instacart TitanTPP learning-rate test](https://app.notion.com/p/391bbe40561381ce8689fd447c8d9a36): 높은 학습률에서 TitanTPP의 불안정성
- [Demand and Taxi learning-rate test](https://app.notion.com/p/393bbe405613819d8fdee93027304b62): 실행 완료와 일반화 성능의 구분
- [Instacart RMTPP learning-rate test](https://app.notion.com/p/394bbe4056138118bff8d3f20826cd4d): `5e-3`에서 RMTPP 수렴이 빨라진 근거

### 5. Model Design Enhancement

- [5. Model Design Enhancement](https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e): 2026년 7월 모델 강화와 gate 결과
- [TitanTPP model status](https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a): V2 유지, Taxi V3b 채택과 후보 상태
- [V1/V2 e200](https://app.notion.com/p/394bbe405613819bae3ed954fa08b2dc): V1·V2의 세 데이터셋·세 seed 비교
- [V3 design](https://app.notion.com/p/399bbe40561381c3bd56e686555d4492): mark별 value 예측과 V3b gradient 분리
- [Taxi V2/V3b e50](https://app.notion.com/p/39abbe405613816aad2be79e6f7f0702): 동일 조건의 Taxi V3b 비교
- [Q0/Q1/Q2 result](https://app.notion.com/p/39cbbe405613812b8a44eba91ea82e92): 큰 수량 오차와 작은 수량·marker 성능의 trade-off
- [V6 result](https://app.notion.com/p/3a0bbe4056138182bae2c5241cb4cea8): lookback 이전 이력 추가를 중단한 근거
- [V7 result](https://app.notion.com/p/3a2bbe40561381caa49ce6022be5992d): time-only 과거 이력 확장을 중단한 근거
- [V5b design](https://app.notion.com/p/3a2bbe40561381c3b087c53a945c0a63): 아직 실행하지 않은 class-prior correction 설계

## 11. 파일과 실험 근거

- 실행 규칙: `TEST_SESSION_PROTOCOL.md`
- 모델 상태 기준: `.agents/results/architecture/titantpp-model-status-baseline-registry.md`
- 학습률: `inter_yt_lr_sensitivity_e50`, `insta_lr_sensitivity_e50`, `insta_rmtpp_lr_sensitivity_e50`
- e1000 분해: `search_artifacts/nll_decomposition_yellow_overfit_e1000`
- V1: `search_artifacts/model_enhancement_v1_residual_e200_0705`
- V2: `search_artifacts/model_enhancement_v2_hybrid_e200_0705`
- Taxi V2: `search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710`
- Taxi V3b: `search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`

모든 최종 수치는 manifest, summary, test summary, histories와 scale-wise 결과 순서로 다시 대조했다. 원천 레코드 수와 전처리 후 사건 수는 같은 수치처럼 제시하지 않는다.
