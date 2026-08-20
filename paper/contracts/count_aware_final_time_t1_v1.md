# TitanTPP Final Time Head + T1 Contract v1

## 결정

H1 실패 원인은 time과 quantity gradient의 지속적인 충돌이 아니라 RMTPP slope 상한과
수치 안정성의 trade-off로 판정한다. 따라서 gradient를 분리하지 않고 Hard-LMM
encoder를 두 objective가 함께 학습하는 현재 구조를 유지한다.

Time head 후보는 H3 log-normal duration으로 교체한다. H3는 slope와 `w * delta_t`
항을 사용하지 않으므로 H0의 폭발과 H1의 slope 포화를 구조적으로 제거한다.

## H3 density

Train target만 이용해 다음 통계를 고정한다.

```text
time_scale = median(delta_t_train)
z = log(delta_t / time_scale)
mu_init = mean(z_train)
sigma_init = std(z_train)
```

모델은 다음 density를 사용한다.

```text
mu(h) = linear_time(h)
sigma = 0.001 + softplus(sigma_raw)
log f(delta_t | h)
  = -0.5 * ((log(delta_t / time_scale) - mu(h)) / sigma)^2
    - log(sigma)
    - log(delta_t)
    - 0.5 * log(2*pi)
```

`-log(delta_t)`는 원 시간 단위 density를 위한 Jacobian이다. Point prediction은
`time_scale * exp(mu(h))`인 conditional median을 사용한다. Target duration은 양수여야
하며 density 내부에서 duration 또는 exponent를 clamp하지 않는다.

## 통합 Variant

| Variant | Titan memory | Quantity loss | Time head | 역할 |
| --- | --- | --- | --- | --- |
| F0 | Hard-LMM | T1 tail-shared | H0 scaled exact | fresh matched control |
| F1 | Hard-LMM | T1 tail-shared | H3 log-normal | candidate |

나머지 조건은 seed 42, 최대 300 epoch, min epoch 40, patience 40, batch 128,
learning rate `1e-3`, lookback 520주, max sequence length 256, hidden dimension 64로
고정한다. Minimum validation joint objective checkpoint를 선택하며 held-out test는
사용하지 않는다.

## 채택 기준

- F1 Time NLL 악화가 F0 대비 `0.01` 이하여야 한다.
- 전체 quantity MAE와 RMSE 악화가 각각 `2%` 이하여야 한다.
- `<=p95` 및 `>p99` quantity MAE 악화가 각각 `2%` 이하여야 한다.
- 모든 loss, prediction, gradient와 artifact metric이 finite여야 한다.

한 조건이라도 실패하면 H3는 채택하지 않고 H0를 유지한다.
