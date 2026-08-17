# Adapted NHP·SAHP Count-aware 비교 계약

## 목적

현재 mark-free count-aware 실험에 NHP와 SAHP 계열의 history encoder를 추가한다. 이 단계의 목적은 TitanTPP-T1의 성능을 서로 다른 시퀀스 표현 방식과 비교하는 것이며, 손실 함수나 평가 규칙을 동시에 바꾸는 것이 아니다.

## 공통 조건

- 입력은 `log1p(delta_t)`와 `log1p(raw_quantity)` 두 연속 특성으로 고정한다.
- target quantity는 history encoder 입력에서 0으로 가려 target leakage를 방지한다.
- 다음 사건 시간은 기존 RMTPP형 continuous-time density head로 예측한다.
- 수량은 기존 positive `log1p(quantity)` regression head로 예측한다.
- 학습 손실, point prediction, validation-only selection, fixed split, seed, epoch, batch size, learning rate는 기존 count-aware backbone control과 동일하게 유지한다.
- NHP와 SAHP의 원 논문별 고유 likelihood를 이번 비교에 도입하지 않는다. 이를 도입하면 encoder뿐 아니라 time objective도 달라져 현재 backbone control과 직접 비교할 수 없기 때문이다.

## Variant 계약

### Adapted NHP

- history encoder를 continuous-time LSTM으로 교체한다.
- 이전 사건 이후 경과시간에 따라 cell state가 연속적으로 decay한 뒤 현재 사건을 반영한다.
- padding 위치에서는 recurrent state를 갱신하지 않고 출력도 0으로 유지한다.
- 논문과 결과표의 명칭은 `Adapted NHP` 또는 `Count-aware NHP`로 제한한다.

### Adapted SAHP

- history encoder를 causal self-attention과 continuous-time decay modulation으로 구성한다.
- attention은 미래 사건과 padding key를 보지 못해야 한다.
- attention history에서 계산한 base state와 event state 사이를 경과시간에 따라 decay하여 시간 간격 정보를 반영한다.
- 논문과 결과표의 명칭은 `Adapted SAHP` 또는 `Count-aware SAHP`로 제한한다.

## 구현 수용 조건

- 모든 backbone이 동일한 `encode(dts, history_quantities, mask)` 인터페이스를 제공한다.
- 출력 shape은 `(batch, sequence, hidden_dim)`이고 padding 출력은 0이다.
- target quantity 변경이 예측 입력 hidden state를 바꾸지 않는다.
- 극단적인 delta-time과 quantity에서도 forward, loss, gradient가 finite하다.
- NHP의 masked step은 recurrent state를 바꾸지 않는다.
- SAHP의 attention은 미래 및 padding key를 차단한다.
- 기존 RMTPP, THP, TitanTPP 계약 테스트를 통과한다.

## 실험 범위

구현 후 첫 단계는 CPU focused test와 e1 partial smoke이다. 현재 5080에서 실행 중인 TitanTPP-T1 multi-seed 학습이 끝나기 전에는 새 GPU 학습을 시작하지 않는다. 이후 동일한 Intermittent fixed split에서 seed 42 screening을 먼저 수행하고, 결과가 유효할 때만 seeds 52·62로 확장한다.

## 해석 제한

이 비교는 NHP와 SAHP의 원 논문 구현을 재현하는 실험이 아니다. 공통 time·quantity head 아래에서 continuous-time recurrent encoder와 self-attentive Hawkes encoder가 count-aware next-event prediction에 기여하는지를 확인하는 통제 실험이다.
