# Notion Result Source: H0/H3 Gradient Clipping and Quantity Damage Audit

## 상태

- 완료
- H0: 비교 기준선 유지, rare-batch instability 미해결
- H3: 미채택 유지
- train only, validation 및 held-out test 미사용

## 결과

| Variant/state | Clipped batches | Median clip scale | Dominant group | Group share |
| --- | ---: | ---: | --- | ---: |
| H0 best | 3.125% | 1.0000 | quantity head | 44.18% |
| H0 final | 3.125% | 1.0000 | shared encoder | 45.88% |
| H3 best | 100.00% | 0.5727 | time head | 93.30% |
| H3 final | 100.00% | 0.2418 | time head | 86.53% |

H3 best의 shared encoder에서 time gradient는 quantity gradient보다 약 `7.06배` 컸다.
Cosine 중앙값은 `-0.0066`으로 강한 반대 방향 충돌은 아니었다. H3 clipping의 직접
원인은 time-head gradient 규모다.

H3 best의 train quantity loss는 H0보다 `3.15%` 낮았지만 raw MAE는 `22.15%`
높았다. Eval mode에서도 log-MSE는 `2.94%` 낮고 raw MAE는 `31.09%` 높았다.
Time-dominant joint optimization이 log-domain quantity loss는 유지하면서 raw quantity
오차를 악화시킨 것으로 해석한다.

H0/H3 encoder와 quantity head를 교차 적용하면 양방향 모두 큰 오차가 발생했다. 이는
강한 co-adaptation을 보여주지만 각 구성요소의 독립적 손상을 증명하지는 않는다.

**판정:** H3는 재개하지 않는다. H0는 비교 기준선으로만 유지하며 기존 full-epoch의
rare-batch train loss 폭증을 미해결 위험으로 남긴다. 다음 time-head 후보는 전용
clipping/optimizer 또는 shared encoder time-gradient norm routing 계약을 먼저 가져야 한다.
