# 교수님 면담용 핵심 보고서: TitanTPP 모델 강화 결과

- 작성일: 2026년 7월 23일
- 실험 결과 기준일: 2026년 7월 19일
- 보고 범위: 2026년 6월 28일 이후 수행한 학습 진단과 모델 강화 실험
- 면담 목적: 현재까지 확인된 결과를 공유하고 논문의 기여 범위와 최종 비교 실험을 확정

> **한 문장 요약:** RMTPP의 mark·time 확률 예측 구조는 유지하면서 TitanTPP의 수량 예측 방식과 손실함수 사이의 충돌을 단계적으로 개선했다. 학습률 증가는 일부 실험의 수렴을 앞당겼지만 안정적인 해결책은 아니었고, 현재는 Demand·Instacart에서 TitanTPP V2, Taxi에서 V3b를 사용하고 있다.

## 1. 이번 면담에서 먼저 말씀드릴 결론

| 확인할 질문 | 현재 답변 | 핵심 근거 |
|---|---|---|
| 장기 학습의 원인이 작은 learning rate인가? | 일부는 맞지만 유일한 원인은 아님 | RMTPP는 `5e-3`에서 빨리 수렴했지만 `1e-2`에서는 효과가 유지되지 않았고, TitanTPP 일부 설정은 높은 learning rate에서 NaN 발생 |
| 모델 용량이 너무 작아 overfitting이 늦은가? | 선택한 설정에서는 가능성이 낮음 | Taxi e1000에서 RMTPP와 TitanTPP 모두 최적 epoch 이후 validation NLL이 다시 악화 |
| TitanTPP 강화가 실제로 도움이 됐는가? | 수량 예측은 개선됐고, Taxi에서는 mark 성능까지 함께 개선 | V2는 세 데이터셋의 quantity MAE를 낮췄고, Taxi V3b는 동일 조건의 3-seed 비교에서 marker NLL과 quantity MAE를 함께 개선 |
| 현재 결과로 주장할 수 있는 논문 기여는 무엇인가? | RMTPP의 사건 이력 표현과 수량 예측 구조 개선 | 별도 날씨·프로모션·외부 충격 변수는 아직 입력하지 않았으므로 exogenous-variable 모델로 주장하지 않음 |

## 2. 모델 버전과 실험 데이터

### 2.1 보고서에서 사용하는 모델 이름

이 보고서에서 V1, V2, V3b는 논문에서 통용되는 모델명이 아니라 TitanTPP의 변경 단계를 구분하기 위한 내부 이름이다. 번호가 높다고 항상 우수하다는 뜻은 아니다.

![RMTPP에서 TitanTPP로의 설계 변화](./F3_model_architecture_evolution.png)

| 모델 | 이번 연구에서의 의미 | 현재 용도 |
|---|---|---|
| RMTPP | 과거 사건을 GRU로 요약하고 다음 mark와 발생 시간을 확률적으로 예측 | 기본 비교 모델 |
| TitanTPP V1 | Titan 인코더에 과거 수량의 residual 정보를 입력하고 다음 residual을 예측 | 초기 TitanTPP 비교 모델 |
| TitanTPP V2 | V1의 residual 학습에 실제 수량 오차를 직접 반영하는 손실함수를 추가 | Demand·Instacart에서 현재 사용하는 모델 |
| TitanTPP V3b | mark별 수량 예측기를 두고 수량 오차의 학습 신호가 mark 분류부를 직접 바꾸지 않도록 분리 | Taxi에서 현재 사용하는 모델 |

여기서 mark는 다음 사건의 수량 규모를 구간화한 클래스이고, residual value는 해당 구간 안에서의 세부 위치다. `small_lmm`과 `mid_lmm`은 새로운 모델명이 아니라 인코더와 memory 크기를 각각 64와 128 수준으로 묶은 실행 설정이다.

### 2.2 데이터 출처와 사건 시퀀스 변환

세 데이터셋의 원래 단위는 서로 다르다. 비교 가능한 TPP 입력을 만들기 위해 각 데이터셋을 `하나의 개체에서 시간순으로 발생하는 수요 사건` 형태로 변환했다.

| 데이터셋 | 출처와 원자료 단위 | 하나의 시퀀스 | 사건과 수량을 만든 방법 |
|---|---|---|---|
| Demand | 사내 부품별 주간 수요 이력 | 하나의 부품 번호 | 누락 주차를 채운 뒤 간헐 수요 부품을 선별하고, 8 이하의 소량 수요를 제외한 후 3주 이내에 이어진 수요를 하나의 수요 묶음으로 합산 |
| Instacart | Kaggle Instacart Market Basket Analysis의 주문·상품 이력 | 하나의 사용자 | 실제 날짜 대신 `days_since_prior_order`를 누적한 상대 일자를 사용하고, 같은 날 구매한 상품 수를 합쳐 하나의 장바구니 수요 사건으로 변환 |
| Taxi | NYC TLC Yellow Taxi의 개별 승차 기록 | 하나의 승차 위치 격자 | 승차 좌표를 0.02도 격자로 나누고 1시간 단위로 집계하여, 활성 시간대가 72개 이상인 격자의 시간당 승차 건수를 수요량으로 사용 |

변환 후 세 데이터셋은 같은 입력 형식을 사용한다. `delta_t`는 직전 수요 사건 이후의 시간 간격이고, 수량 `q`는 `mark = floor(log2(q))`와 `residual = log2(q) - mark`로 나눈다. Instacart의 상품 분류나 Taxi의 결제 방식처럼 원자료에 있던 범주를 최종 mark로 사용한 것이 아니라, 세 데이터셋 모두 수량 규모를 기준으로 mark를 다시 계산했다. 데이터 분할은 각 시퀀스의 시간 순서를 유지한 train·validation·test 고정 분할을 사용했다.

## 3. TitanTPP를 개선하며 다룬 문제

Titan 인코더를 사용하는 기본 방향은 이미 정해져 있었기 때문에, 이후 실험은 인코더 교체 자체보다 **수량을 어떻게 표현하고 학습할 것인지**에 집중했다.

| 확인한 문제 | 적용한 변경 | 확인하려 한 효과 |
|---|---|---|
| residual 오차가 작아도 수량으로 복원하면 오차가 크게 증폭될 수 있음 | V2에서 residual 손실과 실제 복원 수량의 오차를 함께 학습 | 학습 목표와 최종 quantity MAE 사이의 차이 축소 |
| 모든 mark 구간에 하나의 공통 수량 예측기를 사용하면 규모별 패턴 차이를 놓칠 수 있음 | V3 계열에서 mark별 수량 예측기를 분리 | 작은 수량과 큰 수량 구간의 서로 다른 오차 구조 반영 |
| 수량 오차를 줄이는 학습 신호가 mark 분류 확률까지 바꾸면서 두 목표가 충돌할 수 있음 | V3b에서 수량 손실을 계산할 때 mark 확률 경로만 분리하고, marker NLL은 기존대로 학습 | mark 분류를 유지하면서 수량 예측을 개선 |

이후에는 mark 클래스 불균형을 고려한 손실함수, 직접 수량 회귀와 RevIN, 공용 인코더의 학습 신호 분리, 현재 입력 구간보다 앞선 이력의 추가도 검토했다. 그러나 현재 실험에서는 mark 성능과 수량 성능을 함께 안정적으로 개선하지 못해 최종 비교 모델에는 포함하지 않았다.

## 4. 교수님 의견에 대한 확인 결과

### 4.1 Learning rate를 5배·10배 높인 결과

![Learning rate 민감도](./F1_learning_rate_stability.png)

Instacart RMTPP의 seed-42 e50 실험에서 best validation NLL epoch는 learning rate `1e-3`, `5e-3`, `1e-2`일 때 각각 26, 5, 26이었다. `5e-3`에서는 수렴이 빨라졌지만 `1e-2`까지 높이면 같은 효과가 유지되지 않았다.

TitanTPP는 설정에 따라 결과가 달랐다. 일부 조합은 높은 learning rate에서도 완료됐지만 일부는 epoch 4, 10, 28 등에서 NaN이 발생했다. 따라서 낮은 learning rate가 장기 학습의 일부 원인이라는 가설은 확인됐지만, learning rate를 일괄적으로 5배 또는 10배 높이는 방법은 안정적인 해결책이 아니었다. 최종 비교의 기준 learning rate는 `1e-3`으로 유지한다.

### 4.2 모델 용량과 overfitting을 확인한 결과

![Taxi e1000 NLL 분해](./F2_e1000_nll_decomposition.png)

Taxi e1000에서 RMTPP와 TitanTPP의 best validation NLL epoch는 각각 861과 987이었다. 마지막 epoch의 validation NLL은 최적 지점보다 RMTPP에서 0.0873, TitanTPP에서 0.0406 높았다. 선택한 두 모델 모두 최적점 이후 성능이 악화됐으므로, 과적합할 수 없을 정도로 모델 용량이 작다고 보기는 어렵다.

또한 total NLL만 보면 학습 상태를 잘못 해석할 수 있었다. 최적 epoch 이후 time NLL이 계속 낮아지는 동안 marker NLL은 악화될 수 있었다. 이후에는 total NLL을 marker NLL과 time NLL로 분리하고, quantity MAE와 mark accuracy도 함께 확인했다.

F1과 F2는 교수님 의견에 답하기 위한 seed-42 진단 실험이다. 최종 모델의 우위를 주장하는 근거로는 사용하지 않는다.

## 5. 모델 강화에서 확인된 핵심 결과

### 5.1 V2는 수량 예측을 개선했지만 모든 지표에서 우세하지는 않았다

V1과 V2는 e200, seeds 42·52·62로 학습하고, 각 버전에서 mean best validation NLL이 가장 낮은 설정을 선택한 뒤 test split을 평가했다.

| 데이터셋 | V2의 test NLL 변화 | V2의 quantity MAE 변화 | 해석 |
|---|---|---|---|
| Instacart | +0.040% | -1.041% | 수량 오차는 감소했지만 NLL은 소폭 악화 |
| Demand | -0.316% | -9.943% | NLL과 수량 오차가 함께 개선 |
| Taxi | +1.233% | -25.858% | 수량 오차는 크게 감소했지만 NLL은 악화 |

세 데이터셋 모두 quantity MAE는 낮아졌다. 다만 Instacart와 Taxi에서는 V1과 V2의 선택 설정이 서로 달라, 이 결과를 손실함수 하나만 바꾼 단일 요인 비교로 해석하지 않는다. V2는 수량 정보를 반영한 공통 비교 모델로 사용하되 V1보다 모든 면에서 우수하다고 주장하지 않는다.

### 5.2 Taxi에서는 V3b가 동일 조건의 V2를 안정적으로 개선했다

![Taxi V2와 V3b의 3-seed 비교](./F5_taxi_v2_v3b_multiseed.png)

Taxi 비교는 V2와 V3b 모두 `mid_lmm`, e50, seeds 42·52·62, lookback 168, `max_seq_len` 256, batch size 128로 맞췄다. checkpoint도 동일하게 best validation NLL로 선택했다.

| Validation 지표 | V3b의 V2 대비 변화 |
|---|---|
| total NLL | -2.335% |
| marker NLL | -16.448% |
| time NLL | +0.181% |
| quantity MAE | -49.086% |
| residual value MAE | -27.303% |
| mark accuracy | +0.729%p |

total NLL, marker NLL, quantity MAE와 mark accuracy는 세 seed에서 모두 같은 방향으로 개선됐다. time NLL은 평균 0.181% 악화됐으므로, V3b의 이득은 시간 예측 개선이 아니라 mark 분류와 수량 회귀 사이의 학습 간섭을 줄인 결과로 해석한다. 이 결론은 현재 Taxi에만 적용한다.

## 6. 현재 모델 선택과 추가 검토 결과

| 범위 | 현재 선택 | 판단 이유 |
|---|---|---|
| Demand | TitanTPP V2 | 후속 구조가 mark 성능을 안정적으로 넘지 못해 V2 유지 |
| Instacart | TitanTPP V2 | 추가 실험은 구현 확인 수준이며 현재 모델을 변경할 근거가 부족 |
| Taxi | TitanTPP V3b | 동일 조건 3-seed 비교에서 V2보다 marker와 quantity 지표 개선 |
| 직접 수량 회귀·RevIN | 현재 모델에서 사용하지 않음 | 큰 수량 오차가 줄어든 경우에도 표본이 많은 작은 수량 구간과 mark 예측이 함께 손상 |
| 현재 입력 구간보다 앞선 이력 추가 | 추가 개발 보류 | 학습 데이터만 사용한 사전 분석에서 추가 이력의 예측력이 구현 기준을 충족하지 못함 |

RevIN이 일반적으로 효과가 없다는 결론은 아니다. 현재의 희소 사건 구성과 시험한 정규화 방식이 V2보다 안정적인 개선을 만들지 못했다는 범위로만 판단했다.

## 7. 현재 증거가 뒷받침하는 논문 기여

> **제안하는 기여 문장:** RMTPP의 다음 mark와 다음 사건 시간에 대한 확률모형을 유지하면서 Titan 계열 인코더와 수량 오차를 직접 반영하는 학습 목표를 결합하고, mark 분류와 수량 회귀 사이의 학습 신호 충돌을 완화하는 구조를 검증했다.

현재 입력은 관측된 사건 이력의 mark, inter-event time과 quantity다. 날씨, 프로모션, 공급 차질과 같은 별도 외생변수는 포함하지 않았다. 따라서 현재 결과를 “외생변수에 의한 튀는 영향을 반영한 모델”이라고 표현하는 것은 정확하지 않다.

## 8. 면담에서 결정받고 싶은 사항

### 결정 1. 논문의 기여 범위

| 선택지 | 내용 | 영향 |
|---|---|---|
| A. 현재 범위로 마무리 | RMTPP의 사건 이력 표현과 수량 학습 목표 개선을 중심으로 논문화 | 현재 실험 근거와 직접 연결되며 최종 공정 비교 후 결과 정리 가능 |
| B. 외생변수 모델로 확장 | 날씨·프로모션·공급 충격 등을 받는 별도 입력 경로 추가 | 데이터 정합, 새로운 비교 모델과 단일 요인 실험이 필요해 연구 범위와 일정 확대 |

현재 증거만 기준으로는 **A안을 우선 제안**한다. 교수님께서 외생변수 반영을 핵심 기여로 보시는 경우에만 B안으로 범위를 확장하는 것이 타당하다.

### 결정 2. 최종 공정 비교 실행

최종 비교는 Titan 인코더의 효과와 수량 입력·손실함수의 효과를 구분하도록 다음 네 모델을 사용한다.

| 비교군 | 비교 목적 |
|---|---|
| 기존 RMTPP | 원래 구조와의 비교 |
| 수량 조건을 맞춘 RMTPP | TitanTPP와 같은 수량 입력·손실함수를 적용해 인코더 차이를 분리 |
| 수량 조건을 맞춘 THP | Transformer 계열 인코더와 비교 |
| TitanTPP | Demand·Instacart는 V2, Taxi는 V3b 평가 |

공통 조건은 epochs 800, seeds 42·52·62, learning rate `1e-3`, batch size 128, 고정 train/validation/test split이다. checkpoint는 best validation NLL만 사용하고 test 결과를 본 뒤 설정이나 epoch를 다시 선택하지 않는다.

이 비교 조건은 확정했지만 e800 실험은 아직 실행하지 않았다. 따라서 RMTPP·TitanTPP·THP의 최종 우위는 이 실험이 완료된 뒤에 판단한다.

## 9. 상세 근거

- [기존 July Meeting Preparation](https://app.notion.com/p/3a2bbe405613803180c4ea4eef6ccdba): 전체 실험 히스토리와 도표
- [RMTPP 기본 개념](https://app.notion.com/p/2e4bbe40561381c5a442d6e1dace7e18): mark likelihood와 time intensity
- [교수님 피드백 기반 Contribution 정리](https://app.notion.com/p/390bbe405613817abf93e494caf963e1): 논문 방향의 출발점
- [NLL 분해](https://app.notion.com/p/37abbe40561381f7be8ff0a911a182e9): total NLL을 marker와 time으로 분리한 근거
- [Taxi e1000 overfit](https://app.notion.com/p/37abbe4056138179975bfd003bace1f3): 장기 학습과 최적 epoch 분석
- [TitanTPP V1·V2 e200 비교](https://app.notion.com/p/394bbe405613819bae3ed954fa08b2dc): 세 데이터셋·세 seed 결과
- [Taxi V2·V3b e50 비교](https://app.notion.com/p/39abbe405613816aad2be79e6f7f0702): 동일 조건의 Taxi 최종 비교
- [현재 모델 상태](https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a): 데이터셋별로 현재 사용하는 모델과 추가 검토 결과
