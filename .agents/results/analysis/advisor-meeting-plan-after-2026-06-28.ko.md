# 2026년 6월 28일 이후 지도교수 미팅 자료 구성안

- 작성일: 2026-07-19
- 기준 시점: 2026-06-28 지도교수 메일
- 반영 범위: 2026-06-28 이후부터 2026-07-19까지 완료한 분석과 실험
- 상태: 발표 자료의 목적·구조·근거·그림 구성 확정, 실제 슬라이드는 아직 작성 전

## 1. 미팅의 목적

이번 미팅은 지금까지 수행한 실험을 시간순으로 모두 보고하는 자리가 아니다. 핵심은
교수님께서 6월 28일 메일에서 제시한 질문에 답하고, 그 답을 근거로 논문의 주장을
어디까지 가져갈지 결정하는 것이다.

미팅에서 결정받아야 할 사항은 다음 세 가지다.

1. 학습률을 5배 또는 10배로 높였을 때 수렴이 빨라지는지, 그리고 안정성이 깨지는지
2. 같은 조건에서 확인된 TitanTPP V2와 V3b의 개선이 정확히 무엇인지
3. 논문을 `RMTPP의 이력·수량 반영 구조 개선`으로 정리할지, 실제 외생변수를 추가하는
   새 실험까지 확장할지

본 발표는 8개 구간, 핵심 그림 6개를 기준으로 구성한다. 세부 실험 로그와 실패한
후보의 전체 수치는 부록으로 분리한다.

### 포함할 내용

- 학습률 `1e-3`, `5e-3`, `1e-2` 비교와 1000 epoch 실험의 해석
- total NLL을 marker NLL과 time NLL로 나누어 봐야 하는 이유
- RMTPP에서 TitanTPP V2, Taxi V3b로 이어지는 구조 변화
- seed, epoch, candidate 등 비교 조건을 맞춘 V1/V2 및 V2/V3b 결과
- 실패한 실험이 현재 모델 선택을 어떻게 좁혔는지에 대한 요약
- 교수님께 결정받을 두 가지 논문 방향

### 본문에서 제외할 내용

- V1부터 V7, M/Q 계열의 모든 실행 로그 나열
- e1 smoke test나 seed-42 단일 실험을 최종 성능 근거로 사용하는 것
- 현재 결과만으로 RevIN 자체가 효과 없다고 일반화하는 것
- 아직 설계만 끝난 V5b를 완료된 모델처럼 설명하는 것
- 과거 수량 이력을 외생변수라고 바꾸어 부르는 것

## 2. 현재 결과로 주장할 수 있는 핵심

현재 구현과 실험이 직접 뒷받침하는 설명은 다음과 같다.

> RMTPP의 mark·time 확률 모형은 유지하되, 사건 이력을 인코딩하는 부분과 수량을
> 예측하는 부분을 확장하여 시간·mark·수량을 함께 안정화하려고 한 모델이다.

V2는 과거 사건의 mark, 시간 간격, 수량 표현을 Titan 계열 encoder에 넣고 수량
오차를 직접 반영한다. V3b는 가능한 mark마다 서로 다른 수량 잔차를 예측하고,
수량 loss가 mark 확률을 직접 흔드는 경로를 차단한다.

여기서 과거 수량은 같은 사건 이력 안에 이미 관측된 값이다. 날씨, 프로모션,
공휴일, 공급 차질과 같은 외부 설명변수는 아직 모델에 입력하지 않았다. 따라서
현재 단계에서 `외생변수의 영향을 반영한 모델`이라고 쓰는 것은 정확하지 않다.

## 3. 교수님 질문에 대한 답

| 질문 | 현재까지의 답 | 해석 제한 |
| --- | --- | --- |
| 1000 epoch 이상 필요했던 이유가 학습률이 너무 작았기 때문인가 | 일부는 맞다. Instacart RMTPP에서 best NLL epoch가 `1e-3`의 26에서 `5e-3`의 5로 빨라졌다. 다만 `1e-2`에서는 다시 26이었다. | e50, seed-42 결과이므로 최종 비교가 아니라 원인 확인용이다. |
| 학습률을 5배·10배 높이면 발산하는가 | RMTPP는 세 학습률을 모두 완료했지만 TitanTPP 일부 구성은 `5e-3` 또는 `1e-2`에서 3~28 epoch 사이 NaN이 발생했다. Taxi와 수요 데이터는 끝까지 실행됐어도 marker 성능이 악화된 경우가 있었다. | 실행 완료와 좋은 일반화 성능은 다른 조건이다. |
| 모델 크기가 작아서 과적합까지 도달하지 못한 것인가 | Taxi e1000에서 여러 RMTPP/TitanTPP 구성이 best epoch 이후 악화됐으므로, 최소한 해당 설정에서는 과적합 가능한 용량이 확인됐다. | Taxi, seed-42, `1e-3` 분석이며 최종 우열 근거가 아니다. |
| total NLL만으로 판단하면 왜 안 되는가 | time NLL이 크게 내려가 total NLL을 지배하는 동안 marker NLL과 mark accuracy는 악화될 수 있었다. | 각 항의 스케일이 다르므로 항별 추이와 예측 지표를 같이 봐야 한다. |
| 6월 28일 이후 실제로 개선된 것은 무엇인가 | V2는 V1 대비 세 데이터셋에서 수량 MAE를 개선했다. Taxi V3b는 같은 e50·seed `42,52,62` 비교에서 V2보다 NLL, marker NLL, 수량 MAE, mark accuracy를 모두 개선했다. | V3b는 Taxi에서만 확정했으며 전체 데이터셋 공통 모델은 아니다. |

교수님 질문에 대한 결론은 `학습률이 전혀 문제가 아니었다`도 아니고, `학습률만
올리면 해결된다`도 아니다. `5e-3`은 일부 모델에서 수렴을 빠르게 했지만,
TitanTPP의 안정성은 candidate와 수량 loss 구성에 따라 달라졌다. 이후 최종 비교는
epoch 수를 무조건 늘리기보다 checkpoint selection과 학습률 안정성을 함께 관리해야 한다.

## 4. 근거의 등급

### A. 최종 성능 설명에 사용할 수 있는 근거

- V1 대 V2: e200, seeds `42,52,62`, 세 데이터셋, 전체 `18/18` 완료
- Taxi V2 대 V3b: e50, seeds `42,52,62`, dataset·candidate·lookback·max sequence·loss·selection을 맞춘 비교

### B. 교수님 질문에 답하기 위한 진단 근거

- `1e-3`, `5e-3`, `1e-2` 학습률 비교
- Taxi e1000의 train loss, total NLL, marker NLL, time NLL 분해
- 동일 실행 재현성을 확인한 Q2 A/B 실험

### C. 현재 선택을 좁혀 준 근거

- V3a/V3c, V4a/V4b, V5a의 탈락
- M0와 Q0-Q3의 수량 개선·저구간 성능 악화 trade-off
- V6와 V7의 사전 진단 실패
- 아직 실행하지 않은 V5b 설계

A 등급만 모델의 최종 개선 근거로 사용한다. B는 학습 해석, C는 왜 현재 모델을
유지했는지 설명할 때만 사용한다.

## 5. 발표 자료 구성

### 구간 1. 6월 28일 이후 확인할 질문

- 교수님 메일의 두 질문을 그대로 제시한다.
- `학습률과 수렴`, `RMTPP 구조 개선 방향`을 이번 보고의 축으로 고정한다.
- 발표 첫 장에서 결론을 미리 제시한다.

### 구피 2. 학습률을 높였을 때의 수렴과 안정성

- RMTPP는 `1e-3/5e-3/1e-2`를 모두 완료했음을 보여 준다.
- TitanTPP는 Instacart 일부 구성에서 `5e-3`부터 NaN이 발생했음을 보여 준다.
- 빠른 수렴과 안정적인 일반화가 같은 의미가 아니라는 점을 설명한다.

### 구간 3. total NLL을 나누어 본 이유

- train loss, total NLL, marker NLL, time NLL을 한 화면에서 비교한다.
- best validation epoch와 마지막 epoch를 표시한다.
- time NLL만으로 total NLL이 좋아 보일 수 있음을 설명한다.

### 구간 4. RMTPP에서 V2·V3b로 무엇을 바꿨는가

- RMTPP: recurrent encoder, mark head, time head
- V2: Titan 계열 encoder, 과거 수량 입력, 직접 수량 loss
- V3b: mark별 value 예측, 수량 loss의 mark 확률 경로 차단
- 유지된 부분과 바뀐 부분을 그림에서 분리한다.

### 구간 5. V2에서 확인된 공통 개선

- Instacart, 수요 데이터, Taxi에서 V1 대비 수량 MAE 변화를 비교한다.
- Taxi에서 test NLL이 `1.23%` 악화된 사실도 함께 제시한다.
- V2는 이후 실험의 공통 기준선이지 모든 지표에서 항상 최적인 모델은 아니라고 설명한다.

### 구간 6. Taxi V3b의 동일 조건 비교

- e50, seeds `42,52,62`의 V2/V3b를 비교한다.
- NLL 개선이 time NLL이 아니라 marker NLL에서 나온 것임을 보여 준다.
- 세 seed 모두 통과했지만 Taxi 전용 결과임을 명시한다.

### 구간 7. 중단한 실험에서 얻은 결론

- V3c/V5a: mark class 불균형과 gradient 충돌이 단순한 분리만으로 해결되지 않음
- M/Q 계열: 큰 수량 오차를 줄이는 동안 표본 대부분인 `1-9` 구간과 marker가 손상됨
- V4: time head 개선 폭이 통과 기준에 미달함
- V6/V7: lookback 이전 이력을 추가할 충분한 근거가 사전 진단에서 나오지 않음

### 구간 8. 교수님께 결정받을 항목

1. 현재 근거로 논문을 마무리할지, 실제 외생변수 실험으로 확장할지
2. V2를 세 데이터셋의 공통 기여로 둘지, V3b의 Taxi 개선을 중심으로 둘지
3. 최종 RMTPP/TitanTPP/THP 비교의 모델과 학습률 범위
4. V5b를 추가 확인한 뒤 모델을 닫을지, 현재 V2/V3b에서 모델을 동결할지

## 6. 필요한 그림

| ID | 그림 | 전달할 메시지 | 작성 방식과 주의점 |
| --- | --- | --- | --- |
| F1 | 학습률별 best epoch와 NaN 발생 시점 | `5e-3`은 일부 수렴을 빠르게 했지만 TitanTPP에서 항상 안전하지 않았다. | RMTPP의 epoch `26/5/26`과 TitanTPP candidate별 완료·실패를 함께 표시한다. e50·seed-42 진단임을 표기한다. |
| F2 | e1000 loss 분해 | total NLL만으로 학습을 해석할 수 없다. | train loss, total NLL, marker NLL, time NLL을 그리고 best epoch를 표시한다. |
| F3 | RMTPP → V2 → V3b 구조도 | 확률 모형은 유지하고 encoder와 quantity branch를 바꿨다. | 일반 forward는 실선, V3b에서 차단한 gradient는 별도 기호로 나타낸다. 외생변수라는 표현은 쓰지 않는다. |
| F4 | 세 데이터셋 V1 → V2 변화 | V2가 수량 MAE를 `1.04%`, `9.94%`, `25.86%` 개선했다. | score와 NLL 변화도 작은 주석으로 표시한다. V1 대비 결과임을 명시한다. |
| F5 | Taxi V2와 V3b seed별 비교 | V3b 개선이 세 seed에서 재현됐고 marker 개선이 중심이다. | seed별 점과 평균을 함께 표시하고 time NLL `+0.181%` 악화도 숨기지 않는다. |
| F6 | 후보 선택 흐름 | 현재 V2/V3b는 여러 후보의 gate 결과로 남은 모델이다. | V2 유지, Taxi V3b 승격, V3c/V4/V5a/M/Q/V6/V7 중단을 한 장에 요약한다. |

## 7. 본문에서 사용할 핵심 수치

### V1 대비 V2

| Dataset | Score 변화 | Quantity MAE 변화 | Test NLL 변화 |
| --- | ---: | ---: | ---: |
| Instacart | `+0.000377` | `-1.04%` | `+0.04%` |
| Demand | `+0.007727` | `-9.94%` | `-0.32%` |
| Taxi | `+0.016058` | `-25.86%` | `+1.23%` |

### Taxi V2 대비 V3b

| Metric | 변화 |
| --- | ---: |
| Total NLL | `-2.335%` |
| Marker NLL | `-16.448%` |
| Time NLL | `+0.181%` |
| Quantity MAE | `-49.086%` |
| Value MAE | `-27.303%` |
| Mark accuracy | `+0.729%p` |

V3b 결과는 dataset, candidate, e50, seeds `42,52,62`를 맞춘 비교이므로 Taxi에서의
개선 근거로 사용할 수 있다. 다른 데이터셋에 대한 근거로 확장하지 않는다.

## 8. 표현 기준

### 사용할 수 있는 표현

- `5e-3`은 일부 설정에서 수렴 시점을 앞당겼다.
- TitanTPP는 candidate와 loss 구성에 따라 높은 학습률에서 불안정했다.
- total NLL을 marker NLL과 time NLL로 나누면서 기존 해석을 수정했다.
- V2는 V1 대비 세 데이터셋에서 quantity MAE를 개선했다.
- Taxi V3b는 같은 조건의 V2보다 marker·quantity 성능을 개선했다.
- 현재 모델은 관측된 사건 이력과 과거 수량을 사용한다.

### 사용하면 안 되는 표현

- TitanTPP가 모든 데이터셋과 지표에서 RMTPP보다 우수하다.
- 1000 epoch가 최적 학습 길이다.
- V3b가 세 데이터셋의 공통 최종 모델이다.
- RevIN은 수량 예측에 효과가 없다.
- 현재 모델이 외생변수의 영향을 학습했다.
- V5b가 class imbalance를 해결했다.

## 9. 교수님께 제안할 두 가지 논문 방향

### 방향 A. 현재 근거로 정리

논문의 중심을 `RMTPP의 mark·time 모형을 유지하면서 사건 이력 encoder와 수량
예측 구조를 확장한 방법`으로 둔다. V2를 세 데이터셋 공통 기준으로, V3b를 Taxi에서
검증된 추가 개선으로 설명한다.

이 방향은 현재 완료된 실험과 직접 대응한다. 추가로 필요한 작업은 최종 baseline
표와 그림을 정리하는 일에 가깝다.

### 방향 B. 실제 외생변수까지 확장

교수님께서 말한 외생변수를 엄밀한 의미로 구현하려면 날씨, 공휴일, 프로모션,
공급 차질 등 사건 외부의 변수를 추가해야 한다. RMTPP와 TitanTPP 모두 같은 변수를
받게 하고, 외생변수 유무와 변수별 ablation을 새로 비교해야 한다.

이 방향은 현재 모델의 이름만 바꾸는 작업이 아니라 데이터, 모델 입력, 비교 기준을
새로 여는 연구 범위 확장이다. 어느 방향을 택할지는 추가 실험 전에 확정해야 한다.

## 10. Notion 근거 지도: 1단계부터 5단계까지

### 1. Summarize of Core &

- [1. Summarize of Core &](https://app.notion.com/p/2e4bbe4056138158a039ec8a9d3c7db8): RMTPP와 point process의 기초 학습이 정리된 상위 단계
- [Summarize RMTPP](https://app.notion.com/p/2e4bbe40561381c5a442d6e1dace7e18): RMTPP의 mark 확률과 time intensity를 설명할 때 참고

### 2. Confirm and Refine Topic

- [2. Confirm and Refine Topic](https://app.notion.com/p/2e4bbe40561380c19f09f2a0799efc6e): 연구 대상과 구현 범위를 좁힌 단계
- [RMTPP 구현 정리](https://app.notion.com/p/2f9bbe40561380c289a8c47827bc1efc): 현재 TitanTPP에서도 유지하는 RMTPP head와 NLL의 기준
- [Data preparation and split](https://app.notion.com/p/2e4bbe4056138097a53fc95468c8887e): 사건 sequence와 데이터 분할을 설명할 때 참고
- [Hyperparameter and RMTPP comparison](https://app.notion.com/p/358bbe40561380c2a20ac7b602e1012d): 과적합, 모델 크기, RMTPP/TitanTPP 비교 문제의 출발점

### 3. Identify Similar Papers and Contribution

- [3. Identify Similar Papers and Contribution](https://app.notion.com/p/2e4bbe40561380a3bbccfbeeae08902e): contribution을 정리한 상위 단계
- [교수님 피드백 기반 Contribution 정리](https://app.notion.com/p/390bbe405613817abf93e494caf963e1): TitanTPP 자체보다 RMTPP 구조 개선으로 표현해야 한다는 논문 방향의 직접 근거

### 4. Data, protocol and validation

- [4. Data, Protocol, Validation](https://app.notion.com/p/2e8bbe4056138004bc28d9ce611e9e16): 데이터와 비교 규칙의 상위 단계
- [NLL 분해](https://app.notion.com/p/37abbe40561381f7be8ff0a911a182e9): total NLL을 marker/time으로 분리한 이유
- [Taxi e1000 overfit](https://app.notion.com/p/37abbe4056138179975bfd003bace1f3): 모델 용량과 best epoch 이후 악화를 설명하는 근거
- [Instacart TitanTPP learning-rate test](https://app.notion.com/p/391bbe40561381ce8689fd447c8d9a36): 높은 학습률에서 TitanTPP의 불안정성을 설명하는 근거
- [Demand and Taxi learning-rate test](https://app.notion.com/p/393bbe405613819d8fdee93027304b62): 완료 여부와 성능 악화를 구분하는 근거
- [Instacart RMTPP learning-rate test](https://app.notion.com/p/394bbe4056138118bff8d3f20826cd4d): `5e-3`에서 RMTPP 수렴이 빨라진 비교 근거

### 5. Model Design Enhancement

- [5. Model Design Enhancement](https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e): 2026년 7월의 모델 강화와 gate 결과 전체
- [TitanTPP model status](https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a): V2 유지, Taxi V3b 승격, 나머지 후보 중단 상태
- [V1/V2 e200](https://app.notion.com/p/394bbe405613819bae3ed954fa08b2dc): V2를 공통 기준으로 정한 세 데이터셋·세 seed 근거
- [V3 design](https://app.notion.com/p/399bbe40561381c3bd56e686555d4492): mark별 value 예측과 V3b gradient 차단의 설계 근거
- [Taxi V2/V3b e50](https://app.notion.com/p/39abbe405613816aad2be79e6f7f0702): 같은 조건에서 확인한 Taxi V3b의 최종 비교 근거
- [Q0/Q1/Q2 result](https://app.notion.com/p/39cbbe405613812b8a44eba91ea82e92): 큰 수량 오차 감소와 `1-9`·marker 손상 trade-off의 대표 근거
- [V6 result](https://app.notion.com/p/3a0bbe4056138182bae2c5241cb4cea8): lookback 이전 이력을 추가하지 않기로 한 근거
- [V7 result](https://app.notion.com/p/3a2bbe40561381caa49ce6022be5992d): time-only 과거 이력 확장을 중단한 근거
- [V5b design](https://app.notion.com/p/3a2bbe40561381c3b087c53a945c0a63): 아직 실행 전인 class imbalance 후속안

## 11. 파일과 실험 근거

- 실행 규칙: `TEST_SESSION_PROTOCOL.md`
- 모델 강화 해석: `search_artifacts` 및 `TITANTPP_MODEL_STATUS.md`
- 학습률: `inter_yt_lr_sensitivity_e50`, `insta_lr_sensitivity_e50`, `insta_rmtpp_lr_sensitivity_e50`
- e1000 분해: `search_artifacts/nll_decomposition_yellow_overfit_e1000`
- V1: `search_artifacts/model_enhancement_v1_residual_e200_0705`
- V2: `search_artifacts/model_enhancement_v2_hybrid_e200_0705`
- Taxi V2: `search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710`
- Taxi V3b: `search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`

모든 최종 수치는 발표 자료를 만들 때 manifest, summary, test summary, histories,
scale-wise 결과 순서로 다시 대조한다. 원천 레코드 수와 전처리 후 사건 수는 같은
수치처럼 제시하지 않는다.

## 12. 다음 작업

1. 이 구성에 따라 교수님 미팅용 본문 초안을 작성한다.
2. F1-F6을 실제 실험 파일에서 생성한다.
3. RMTPP/TitanTPP/THP 최종 비교표에 넣을 모델과 selection을 확정한다.
4. 발표 문장을 수치와 다시 대조하고, 과장된 표현을 제거한다.
5. 미팅 자료가 완성되면 교수님께 보낼 짧은 메일을 작성한다.
