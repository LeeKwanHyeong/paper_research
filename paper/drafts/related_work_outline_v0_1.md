# Related Work outline v0.1

> 목적: TitanTPP가 어떤 연구 흐름을 잇고 무엇을 새로 다루는지 정리한다. 선행연구가 제시한
> 일반적 동기와 본 연구의 실험 결과를 구분한다.

## 3.1 Recurrent neural temporal point processes

- RMTPP는 사건 이력을 recurrent state로 요약해 다음 사건의 시간과 mark를 함께 예측하는
  초기 neural marked TPP에 해당한다.
- Neural Hawkes Process는 continuous-time LSTM을 사용해 event intensity의 recurrent history
  representation을 확장한다.
- 두 연구는 recurrent TPP 계열의 출발점으로 사용한다. 긴 이력에서의 상대 성능은 선행연구의
  결론으로 단정하지 않고 본 연구의 history-length breakdown에서 확인한다.

## 3.2 Attention- and memory-based history encoders

- SAHP와 THP는 self-attention과 시간 정보를 결합해 과거 사건의 영향을 표현한다. 이 가운데
  THP는 본 연구의 직접 비교군이다.
- Titans는 attention과 장기 메모리를 결합하려는 설계 배경으로 사용한다.
- TitanTPP는 Titans에서 영향을 받았지만 원 논문의 test-time learning을 구현한 모델은 아니다.
  frozen model은 causal memory attention, learnable persistent memory, static LMM을 사용하며
  validation과 test 중에는 memory를 갱신하지 않는다.

## 3.3 Event marks and continuous quantity

- marked TPP의 mark space는 원칙적으로 범주형에 한정되지 않지만, 다수의 neural TPP 구현과
  benchmark는 유한한 event type을 예측한다.
- Factorial marked TPP는 하나의 사건을 여러 discrete marker로 분해한다.
- mixed-type event model은 discrete attribute와 continuous attribute에 서로 다른 prediction
  head를 사용한다.
- 본 연구는 수요량을 log-magnitude mark와 continuous residual로 분해하고 두 값을 함께 사용해
  원래 수량을 복원한다. 구체적인 factorization과 differentiable decoder가 본 연구의 차별점이다.

## 3.4 Intermittent-demand forecasting as event prediction

- Croston 계열 접근은 간헐적 수요의 발생 간격과 수요 크기를 분리해 추정한다.
- Deep Renewal Processes는 비영 수요의 도착 간격과 크기를 확률적으로 함께 모델링하며,
  간헐적 수요를 renewal 또는 point-process 관점으로 연결한다.
- 본 연구는 다음 positive-demand event의 시간과 수량을 동시에 예측하며, 동일한 event
  formulation을 Intermittent, Taxi, Instacart에 적용한다.

## 3.5 Position of this work

| 비교 축 | 선행연구의 역할 | 본 연구의 위치 |
| :--- | :--- | :--- |
| 사건 이력 | RMTPP/NHP의 recurrent state, SAHP/THP의 self-attention | causal memory attention과 static LMM을 사용하는 TitanTPP encoder |
| mark | finite event type 또는 여러 discrete marker | log-magnitude class와 continuous residual의 결합 |
| 수요 예측 | 발생 간격과 크기의 분리 또는 joint probabilistic modeling | TPP likelihood와 differentiable quantity reconstruction의 공동 학습 |
| 검증 | 모델별 고유 decoder와 protocol | fixed split, matched quantity interface, history/quantity breakdown |

## Baseline terminology

- `RMTPP-matched`와 `THP-matched`는 원 논문의 결과를 그대로 재현한 모델명이 아니다. 각 원
  논문의 history encoder를 유지하면서 본 연구의 quantity input, decoder, loss와 평가 interface를
  맞춘 adapted baseline이다.
- Intermittent와 Instacart에서는 두 matched baseline과 TitanTPP V2가 shared residual head와
  coupled quantity objective를 공유한다.
- Taxi의 순수 encoder 비교는 TitanTPP V2 control을 사용한다. Taxi primary인 V3b는
  mark-conditioned experts와 detached quantity-to-mark gradient까지 포함한 최종 설계이다.

## Writing guardrails

- RMTPP가 긴 시퀀스에서 항상 실패한다고 쓰지 않는다.
- 범주형 mark로 연속값을 모델링할 수 없다고 쓰지 않는다. 범주형 head만으로는 같은 범주 안의
  수량 차이를 복원할 수 없다고 한정한다.
- TitanTPP를 원본 Titans 또는 test-time learning 모델이라고 부르지 않는다.
- THP-matched를 원 논문 THP의 완전한 재현이라고 부르지 않는다.
- TitanTPP의 우위는 held-out test와 breakdown이 확정된 범위에서만 서술한다.

