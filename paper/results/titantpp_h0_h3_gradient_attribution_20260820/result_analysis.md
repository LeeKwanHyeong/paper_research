# H0/H3 Gradient Clipping and Quantity Damage Audit 결과

## 판정

- 범위: Intermittent train split only, seed 42, fixed 32 batches
- H0 상태: 비교 기준선 유지, rare-batch instability 미해결
- H3 clipping driver: time head
- H3 quantity 손상: time-scale dominance와 log/raw objective 불일치
- Gradient 방향 충돌: 주원인 아님
- Validation 및 held-out test: 사용하지 않음

## Clipping 원인

| Variant/state | Clipped batches | Median clip scale | Dominant group | Group share |
| --- | ---: | ---: | --- | ---: |
| H0 initial | 100.00% | 0.0194 | time head | 62.55% |
| H0 best | 3.125% | 1.0000 | quantity head | 44.18% |
| H0 final | 3.125% | 1.0000 | shared encoder | 45.88% |
| H3 initial | 100.00% | 0.0286 | quantity head | 81.45% |
| H3 best | 100.00% | 0.5727 | time head | 93.30% |
| H3 final | 100.00% | 0.2418 | time head | 86.53% |

H3는 초기에는 quantity-head 초기 오차가 clipping을 만들지만, 학습된 best/final
checkpoint에서는 time head가 명확한 driver로 바뀐다. Best checkpoint의 time-head
gradient norm 중앙값은 `1.6756`, quantity-head gradient는 `0.1119`였다. Time head가
global norm을 지배해 모든 batch가 clipping됐다.

Shared encoder에서도 H3 best의 time gradient norm은 `0.4699`, quantity gradient는
`0.0666`으로 약 `7.06배` 차이가 났다. 그러나 cosine 중앙값은 `-0.0066`이므로 두
objective가 지속적으로 반대 방향인 것이 아니라 time objective의 크기가 encoder
update를 지배하는 상태다. H3 final에서도 비율은 약 `4.36배`였다.

H0 best/final은 고정 32개 batch 중 한 개만 clipping됐고 median scale은 `1.0`이었다.
따라서 기존 full-epoch history의 train loss 폭증은 상시 불안정이라기보다 드문 batch나
extreme duration에 대한 민감성으로 관리한다. 이번 audit가 H0 안정성을 해결한 것은
아니다.

## Quantity 손상 해석

Train-mode fixed batch에서 H3 best는 H0 best보다 quantity MAE가 `22.15%` 높았지만
quantity train loss는 `3.15%` 낮았다. Eval-mode best checkpoint 비교에서도 H3는
raw MAE `+31.09%`, RMSE `+53.68%`인 반면 log-quantity MSE는 `-2.94%`였다.

즉, H3가 quantity objective 자체를 전혀 학습하지 못한 것은 아니다. Time-dominant
joint optimization 아래에서 log-domain 오차는 유지됐지만 raw quantity와 tail에 필요한
표현이 보존되지 않았다. Validation에서 나타난 raw MAE/RMSE 악화와 같은 방향이다.

| Encoder / quantity head | Train MAE | Train RMSE | Log-MSE |
| --- | ---: | ---: | ---: |
| H0 / H0 | 1.3241 | 2.1043 | 0.01224 |
| H3 / H3 | 1.7358 | 3.2339 | 0.01188 |
| H3 / H0 | 23.7513 | 42.2835 | 0.76389 |
| H0 / H3 | 16.4306 | 23.6715 | 0.43654 |

교차 조합은 양방향 모두 크게 실패했다. 이는 H3 encoder와 quantity head가 서로 강하게
공동 적응했고 H0 구성요소와 바로 교환할 수 없다는 증거다. 다만 crossed checkpoint는
학습 분포 밖 조합이므로 encoder와 head가 각각 독립적으로 손상됐다는 인과 증명으로
사용하지 않는다.

## 결론

H3의 100% clipping 원인은 time head이며, 이전 audit에서 확인한 gradient 방향 충돌이
아니라 gradient 규모 불균형이 핵심이다. 이로 인해 shared encoder가 time objective에
더 크게 맞춰지고, joint checkpoint가 log-domain quantity loss는 유지하면서 raw quantity
오차를 악화시키는 상태를 선택했다.

따라서 H3를 재개하려면 sigma나 learning rate를 단순 조정하기보다 time-head 전용
clipping 또는 optimizer group, shared encoder로 전달되는 time gradient의 norm routing,
그리고 raw quantity safety와 분리된 checkpoint 계약을 먼저 설계해야 한다. H0는 현재
비교 기준선으로 남지만 rare-batch 폭증이 해결되지 않았으므로 최종 안정 모델로
표현하지 않는다.
