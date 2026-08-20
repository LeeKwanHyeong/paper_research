# H0/H3 Gradient Attribution Contract v1

## 상태 경계

H0 scaled exact는 현재 F0/F1 비교에서만 유지한다. H0의 train time loss와 gradient
폭증은 해결되지 않았으므로 안정적인 최종 time head 또는 논문 대표 모델로 확정하지
않는다. H3 log-normal duration은 quantity safety gate 실패로 미채택 상태를 유지한다.

## 진단 범위

- Intermittent fixed train split과 seed 42만 사용한다.
- 기존 F0/F1의 initial, best, final state를 동일한 32개 train batch에서 비교한다.
- Validation과 held-out test를 새로 읽거나 selection에 사용하지 않는다.
- H3 상수, learning rate, checkpoint를 결과에 맞춰 변경하지 않는다.

## Gradient attribution

Train mode에서 batch별 dropout seed를 `42 + batch_index`로 고정한다. Time loss와 T1
quantity loss의 gradient를 각각 구한 뒤 shared encoder, time head, quantity head로
나누어 raw norm, joint norm, squared-norm share와 global clipping scale을 계산한다.

Clipping fraction `>=95%`이면 persistent clipping으로 기록한다. Joint squared-gradient
norm share `>=50%`인 group이 있으면 직접 driver로 분류한다. 기준을 넘는 group이
없으면 distributed clipping으로 기록한다.

## Quantity 손상 위치

Best checkpoint에서 다음 네 조합을 train-only fixed batch로 평가한다.

| 조합 | Encoder | Quantity head |
| --- | --- | --- |
| 기준선 | H0 | H0 |
| 실제 H3 | H3 | H3 |
| Encoder transfer | H3 | H0 |
| Head transfer | H0 | H3 |

기준선 대비 MAE 악화 `2%`를 진단 경계로 사용한다. Encoder transfer만 악화되면
encoder-dominant, head transfer만 악화되면 head-dominant, 둘 다 악화되면 mixed,
실제 H3만 악화되면 encoder-head coupling으로 기록한다. Checkpoint crossing은 원인
위치에 대한 진단이며 독립적인 인과 증명으로 해석하지 않는다.
