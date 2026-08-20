# TitanTPP Time-Quantity Gradient Audit 결과

## 판정

- 범위: Intermittent train split only, seed 42
- Backbone/quantity: Hard-LMM + T1 tail-shared
- validation 및 held-out test: 사용하지 않음
- H1 slope 계약: 실패
- time-quantity gradient 간섭: 강한 간섭으로 판정하지 않음
- 다음 계약: slope family 교체, shared encoder gradient 유지

H1은 epoch 3에서 slope 상한의 `98.95%`까지 도달했고, audit한 모든 event에서
time NLL이 slope를 더 키우는 방향의 미분을 보였다. 반면 최종 gradient cosine
중앙값은 `+0.0466`, 강한 충돌 batch 비율은 `25%`였다. 사전에 고정한 충돌 기준은
cosine 중앙값 `<= -0.10`과 강한 충돌 비율 `>= 50%`를 동시에 요구하므로 gradient
분리 조건은 충족하지 않았다.

## Train-only 지표

| Variant | Epoch | Slope ratio | Upward pressure | Gradient cosine | Strong conflict | Train joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | 1 | 0.5960 | 0.9900 | 0.1024 | 9.38% | 10.4195 |
| H0 | 2 | 0.8762 | 0.9932 | -0.1463 | 84.38% | 170,733.4038 |
| H0 | 3 | 0.9565 | 0.9890 | 0.1359 | 3.13% | 18,381,199.4976 |
| H1 | 1 | 0.8383 | 1.0000 | 0.1144 | 28.13% | 1.7487 |
| H1 | 2 | 0.9627 | 1.0000 | -0.0554 | 28.13% | 1.6471 |
| H1 | 3 | 0.9895 | 1.0000 | 0.0466 | 25.00% | 1.6313 |

H0는 slope 상한을 넓히면 표현력은 확보하지만 loss와 gradient가 폭증했다. H1은
수치적으로 안정적이지만 slope 상한 때문에 likelihood가 요구하는 영역에 도달하지
못했다. 따라서 상한을 8과 40 사이에서 다시 고르는 방식은 같은 trade-off를 반복할
가능성이 높다.

## 결정된 후보 계약

H3는 `log(delta_t / train_median)`에 대한 log-normal duration density를 사용한다.
Location은 hidden state에서 예측하고 sigma는 positive softplus scalar로 학습한다.
원 시간 단위의 `-log(delta_t)` Jacobian을 포함하며 slope, `w * delta_t`, duration
clamp는 사용하지 않는다.

초기 location과 sigma는 Intermittent train target의 log-scaled mean/std로 고정한다.
Time loss와 T1 quantity loss는 모두 Hard-LMM encoder를 계속 학습한다. Gradient
분리는 audit gate를 통과하지 않았으므로 추가하지 않는다.

## 다음 검증

동일한 source revision에서 다음 두 run을 fresh seed 42로 비교한다.

| Run | Backbone | Quantity | Time head |
| --- | --- | --- | --- |
| F0 | TitanTPP Hard-LMM | T1 tail-shared | H0 scaled exact RMTPP |
| F1 | TitanTPP Hard-LMM | T1 tail-shared | H3 log-normal duration |

F1은 F0 대비 Time NLL 악화 `<=0.01`, 전체 MAE/RMSE와 `<=p95`, `>p99` MAE 악화
각각 `<=2%`, 모든 값 finite를 만족해야 채택한다. 이 단계에서도 held-out test는
사용하지 않는다.
