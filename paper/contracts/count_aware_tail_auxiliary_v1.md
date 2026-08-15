# Count-aware Log-MSE + Tail-aware Auxiliary Contract v1

## 상태와 목적

Intermittent train-only audit를 통과했으며 `lambda_tail`의 train-only gradient
calibration만 남아 있다. 기존 mark-free `log1p(quantity)` MSE와 point prediction은
바꾸지 않고, 상위 수량 구간의 raw 오차만 bounded auxiliary loss로 보완한다.

이 실험의 목적은 log-MSE의 중·저수량 안정성을 유지하면서 RMSE와 상위 수량 MAE를
개선할 수 있는지 확인하는 것이다. K=1 log-normal NLL은 다시 사용하지 않는다.

## Train-only audit 결과

| 항목 | 값 |
| --- | ---: |
| Train events | 398,824 |
| p90 / p95 / p99 | 31 / 46 / 187 |
| `q > p95` 표본 비율 | 4.9423% |
| `q > p95` log-MSE loss 비율 | 25.8091% |
| `q > p95` absolute log-location gradient 비율 | 12.2645% |
| 사전 중단선 | 50% |

Tail gradient가 기존 log-MSE를 지배하지 않으므로 보조 손실 구현을 진행한다.

## 손실 계약

기준선은 다음과 같다.

```text
mu = softplus(quantity_head(h))
q_hat = expm1(mu)
L_log = MSE(mu, log1p(q))
```

Tail auxiliary는 train split에서 고정한 값만 사용한다.

```text
tail = indicator(q > 46)
q_hat_norm = clamp(q_hat, 0, 187) / 46
q_norm = clamp(q, 0, 187) / 46
L_tail = tail * Huber(q_hat_norm, q_norm, delta=1)
L_quantity = L_log + lambda_tail * L_tail
```

`L_tail`은 body sample에서 0이며 전체 target event 평균으로 줄인다. Target quantity와
padding은 통계 및 encoder history에서 제외한다. 출력 activation과 MAE/RMSE용 point
prediction은 기준선과 동일하다.

## Variant 계약

| Variant | Tail gradient 경로 | 역할 |
| --- | --- | --- |
| T0-logMSE | 없음 | fresh matched 기준선 |
| T1-tail-shared | quantity head + Titan encoder | shared gradient 영향 확인 |
| T2-tail-head-only | quantity head only | 주 후보 |

T2는 동일한 quantity head를 `stop_gradient(h)`에 한 번 더 적용해 auxiliary loss를
계산한다. 따라서 forward prediction과 parameter count는 T0/T1과 같고, tail loss만
encoder로 전달되지 않는다. Time head에는 두 tail variant 모두 직접 gradient를
보내지 않는다.

## Lambda calibration

Validation을 사용하지 않는다. Seed 42 TitanTPP를 train split 128 batch로 warm-up한 뒤,
별도 고정 loader seed `10042`로 섞은 train 64 batch에서 quantity-head gradient norm을
계산한다. 이 shuffle은 시계열 정렬 순서에 따른 tail 표본 누락을 막기 위한 것이며
validation/test row는 읽지 않는다.

```text
lambda_tail
= 0.10 * mean(||grad_head L_log||) / mean(||grad_head L_tail||)
```

수치 안전 범위 `[1e-4, 100]` 안에서 한 번만 고정하고 T1/T2에 동일하게 사용한다.

## 기존 실험과의 차이

- Q3b/Q3c는 mark가 있는 `direct_raw_qty + causal shrinkage RevIN` 구조에 log2 Huber를
  추가했다. 이번 실험은 mark와 RevIN이 없는 log1p-MSE decoder에 raw Huber를 더한다.
- K=1은 log-normal NLL이 log-MSE를 대체했다. 이번 실험은 log-MSE를 그대로 유지해
  K=1에서 발생한 distribution loss의 shared-encoder 지배를 피한다.

## Validation-only 판정

- Fresh T0 대비 전체 RMSE 또는 `>p99` MAE가 5% 이상 개선되어야 한다.
- 전체 MAE와 `<=p95` MAE 악화는 각각 2% 이하여야 한다.
- Time NLL 악화는 0.01 이하여야 한다.
- 모든 loss, prediction, gradient가 finite여야 한다.
- 통과하기 전까지 multi-seed와 held-out test는 실행하지 않는다.
