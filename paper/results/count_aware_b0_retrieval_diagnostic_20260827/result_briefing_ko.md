# B0 Hard-Prototype Retrieval 진단 결과

## 결론

동일한 frozen B0 checkpoint에서 local encoder와 quantity head를 그대로 두고
`local state + memory residual`과 `local state`를 event 단위로 비교했다. 따라서
두 경로의 예측 및 오차 차이는 추론 시점의 additive hard-prototype residual에
직접 귀속된다.

다만 기존 가설인 "B0 memory가 body를 손상하고 extreme tail을 개선한다"는
보편적 현상으로 확인되지 않았다. Intermittent와 Instacart에서는 residual이 body와
`>p99`를 모두 개선했고, RAF에서는 효과가 작고 seed 방향이 흔들렸다. Taxi만 평균상
body 손상과 `>p99` 개선을 보였지만 body 손상은 3개 seed 중 seed 52에서만
`MAE`와 `MSE`가 함께 악화됐다.

따라서 확인된 결론은 다음과 같다.

- B0 residual은 event별 prediction shift의 직접 원인이다.
- body 손상은 B0 전체의 보편적 결함이 아니라 데이터 구간과 seed에 따른 조건부 현상이다.
- extreme-tail 개선은 Intermittent, Instacart, Taxi에서 3-seed 일관성이 있다.
- residual norm 크기만으로 개선과 손상을 설명할 수 없다. retrieval similarity,
  선택 prototype의 집중도, quantity logit 방향을 함께 봐야 한다.

## 실행 및 계약 검증

- 실행 서버와 device: 5080, CUDA
- source revision: `e346b23a98c0f51c15a18e840fba39c052d9e88b`
- 대상: Intermittent v2, Taxi, RAF Spare Parts, Instacart
- checkpoint: 4개 데이터셋 x seeds 42, 52, 62, 총 12개
- 평가 범위: full validation-only, held-out test 미사용
- event 수: seed 전체 합계 1,814,928개
- memory-on 예측과 공식 evaluator의 최대 절대 차이: `0.0`
- artifact 대비 MAE 최대 차이: `2.23e-8`
- artifact 대비 RMSE 최대 차이: `9.61e-8`
- 모든 event의 retrieval 적용률과 nonzero residual 비율: 각각 `100%`
- 모든 prediction, error delta, residual norm, similarity: finite

음수 delta는 memory-on이 memory-off보다 오차를 줄였다는 뜻이다. memory-off는
별도로 재학습한 모델이 아니라 memory와 함께 학습된 checkpoint에서 residual만 제거한
반사실 경로다.

## 전체 오차 변화

| Dataset | On MAE | Off MAE | MAE delta | On RMSE | Off RMSE | RMSE delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Instacart | 4.045027 | 4.755033 | -0.710007 | 6.015904 | 7.096644 | -1.080740 |
| Intermittent v2 | 0.746917 | 4.087909 | -3.340992 | 1.919518 | 9.514090 | -7.594571 |
| RAF Spare Parts | 9.148543 | 9.156757 | -0.008214 | 35.900488 | 35.909517 | -0.009029 |
| Taxi | 42.580526 | 44.078351 | -1.497825 | 143.993141 | 150.914089 | -6.920948 |

동일 checkpoint에서는 네 데이터셋 모두 residual을 제거하면 전체 MAE와 RMSE가
악화됐다. 특히 Intermittent의 큰 차이는 local encoder와 quantity head가 memory가
존재하는 조건에 맞춰 공동 적응했음을 보여준다. 이 값으로 retrained no-memory
baseline보다 B0가 우수하다고 주장할 수는 없다.

## Body와 Extreme Tail

| Dataset | Body MAE delta | Body MSE delta | Body harm seeds | >p99 MAE delta | >p99 MSE delta | >p99 improve seeds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Instacart | -0.520299 | -7.369082 | 0/3 | -4.819282 | -235.709926 | 3/3 |
| Intermittent v2 | -2.048602 | -15.778705 | 0/3 | -53.659897 | -3934.427602 | 3/3 |
| RAF Spare Parts | -0.005428 | +0.293247 | 0/3 | -0.206673 | -114.839741 | 2/3 |
| Taxi | +0.381862 | +299.152336 | 1/3 | -113.546402 | -182798.819120 | 3/3 |

Body는 train-only `p95` 이하, extreme tail은 train-only `p99` 초과로 정의했다.
`Body harm seeds`는 body MAE와 MSE가 동시에 증가한 seed 수다. Taxi의 평균
body 손상은 seed 52의 큰 악화가 주도했고 seeds 42와 62에서는 body MAE와 MSE가
모두 개선됐다. 따라서 Taxi에서도 평균 방향은 확인되지만 seed-stable한 구조적
현상으로 확정할 수 없다.

세부 quantity 구간은 원인을 더 분명하게 보여준다.

- Instacart `<=p50`은 세 seed 모두 손상됐다. 평균 MAE delta는 `+0.663524`,
  MSE delta는 `+6.109630`이었다. 반면 `p50-p90`부터 `>p99`까지는 모두 개선됐다.
- Intermittent는 모든 quantity 구간에서 평균 MAE와 MSE가 개선됐고, 개선 폭은
  `>p99`에서 가장 컸다.
- RAF는 대부분의 delta가 전체 MAE 0.01 수준에 비해 작고, `>p99` 표본도
  seed당 50개뿐이다. seed 52에서는 tail이 반대로 악화돼 방향을 동결할 수 없다.
- Taxi `p50-p90`은 세 seed 모두 MAE와 MSE가 손상됐고, `p90-p95`의 평균 손상은
  seed 52가 주도했다. `p95-p99`와 `>p99`는 세 seed 모두 크게 개선됐다.

## Retrieval 작동 방식

| Dataset | Residual norm | Top-k similarity | Active prototypes | Effective prototypes | Top-4 selection share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Instacart | 0.837209 | 0.458971 | 51.7 | 5.47 | 90.0% |
| Intermittent v2 | 1.522293 | 0.375224 | 61.7 | 16.89 | 44.1% |
| RAF Spare Parts | 0.502915 | 0.649559 | 13.3 | 7.63 | 69.7% |
| Taxi | 0.278426 | 0.465239 | 29.0 | 15.48 | 40.2% |

`Active prototypes`는 한 번 이상 top-k에 포함된 64개 prototype의 수이고,
`Effective prototypes`는 selection-share entropy의 지수다. 모든 event에서 정확히
4개 prototype이 선택되지만 실제 사용은 균등하지 않다.

- Instacart는 평균 51.7개가 한 번 이상 사용됐지만 top 4가 전체 선택의 90.0%를
  차지했다. 즉, 실질적으로는 약 5.47개 prototype에 집중됐다.
- Taxi `p95-p99`와 `>p99`에서는 각 seed가 항상 같은 4개 prototype만 선택했다.
  두 구간의 active prototype 수와 effective prototype 수는 모두 정확히 4였다.
- Intermittent `>p99`도 top 4 선택 비중이 88.4%였고 effective prototype 수는
  4.96이었다.
- RAF `>p99`의 top 4 비중은 95.3%지만 표본 수가 작아 일반화하지 않는다.

Residual norm 자체는 오차 방향을 설명하지 못한다. Instacart의 norm은 `<=p50`부터
`>p99`까지 약 0.836-0.838로 거의 같지만, MAE delta는 `+0.663524`에서
`-4.819282`로 바뀌었다. 같은 구간에서 top-k similarity는 0.406에서 0.558로
상승했다.

Taxi는 더 직접적인 반례다. `<=p50` residual norm은 0.359로 가장 컸지만
`>p99` norm은 0.198에 불과했다. 그런데 가장 큰 개선은 `>p99`에서 발생했다.
Quantity-logit residual 평균도 `<=p50`의 -0.081에서 `>p99`의 +0.040으로
방향이 바뀌었다. 따라서 gate는 residual 크기 하나가 아니라 similarity, local
state와 residual의 결합, quantity-logit projection을 사용해야 한다.

## History 구간

- Intermittent MAE delta는 history `<=64`에서 -3.826, `65-128`에서 -6.616,
  `>128`에서 -0.123이었다. 가장 긴 history에서 residual 이득이 거의 사라졌다.
- Taxi MAE delta는 `<=64`에서 -0.034, `65-128`에서 -0.026, `>128`에서
  -2.216이었다. 그러나 `>128`의 residual norm과 similarity는 오히려 낮아,
  이 개선은 long-history 자체보다 tail event 구성의 영향을 함께 받는다.
- RAF와 Instacart validation target은 모두 history `<=64`라 long-history memory
  근거로 사용할 수 없다.

## 최종 원인 판정과 설계 반영

추론 시점의 event별 오차 변화가 additive memory residual 때문이라는 점은
확정됐다. 그러나 "body 손상, extreme-tail 개선"은 전체 B0의 보편적 작동 원리가
아니며 Instacart low-body와 Taxi middle-body에서만 명확히 관측됐다.

B2 TPP-specific Gated Memory에는 다음 기준을 적용한다.

- Memory를 전 event에 강제 적용하지 않고 null-memory 경로를 허용한다.
- Confidence gate는 residual norm 단독 임계값을 사용하지 않는다.
- Top-k similarity, similarity margin, local state, retrieved residual을 함께 사용한다.
- Hard top-k 평균 대신 similarity-weighted sparse retrieval을 유지한다.
- Gate 사용률, null 선택률, residual projection과 error delta를 quantity/history
  구간별로 계속 기록한다.
- Instacart `<=p50`과 Taxi `p50-p90`을 body 보호 진단 구간으로 사용하되,
  Intermittent처럼 memory 의존성이 큰 데이터에서는 무차별 null 선택을 방지한다.

다음 작업은 5080에서 B1과 B2의 CUDA model-test 및 네 데이터셋 e1 smoke를 수행한
뒤, B0/B1/B2 seed-42 matched validation screening으로 넘어가는 것이다. Held-out
test는 최종 backbone과 checkpoint 선택 규칙을 동결할 때까지 계속 잠근다.
