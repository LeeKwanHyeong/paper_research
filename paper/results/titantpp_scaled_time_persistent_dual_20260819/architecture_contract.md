# TitanTPP Scaled-Time Persistent/Dual Memory 계약

## 상태

- Scaled exact RMTPP time head 구현 및 로컬 검증: 완료
- Persistent-matched memory와 dual-route 구현 및 로컬 검증: 완료
- 5080 CUDA model-test와 Instacart e1 smoke: 실행 전
- Intermittent seed-42 e300 validation screening: 실행 전
- Held-out test: 미사용

## 목적

이 실험은 이전 memory 비교의 두 가지 혼선을 제거한다. 첫째, 모든 Titan 후보가
persistent token 16개를 동일하게 사용하도록 하여 persistent token 제거 효과와 memory
교체 효과를 분리한다. 둘째, 기존의 포화된 time objective 대신 train split에서 고정한
scaled exact RMTPP density를 모든 후보에 적용한다.

최종 질문은 Surprise memory가 수량 표현을 개선하면서도 정상화된 time likelihood를
유지할 수 있는지, 그리고 time과 quantity에 서로 다른 memory state를 제공할 때 두
목표의 충돌이 줄어드는지이다.

## Time Head 계약

- Mode: `scaled_exact_rmtpp`
- Train-only `time_scale`: 3
- Bounded slope: `0 < w <= 10/3`
- Intercept bound: `[-30, 30]`
- `w * scaled_delta_t` clamp: 사용하지 않음
- Density unit: 원래 delta-time 단위이며 `-log(time_scale)` Jacobian 포함
- 누적 hazard와 `expm1`: float64 계산 후 모델 dtype으로 복원
- Train target profile: count 393,824, median 3, p99 7, max 36

이 time head는 RMTPP, THP, TitanTPP에 동일하게 사용할 수 있다. 이번 memory screening은
Titan 후보만 비교하지만, 이후 backbone 비교에서도 같은 head 계약을 유지한다.

## Variant 계약

| ID | Persistent token | Time state | Quantity state | Quantity gradient 경로 | 역할 |
| --- | :---: | --- | --- | --- | --- |
| M0 `titantpp_persistent_only` | 16 | Titan base | Titan base | shared | Persistent token 단독 진단 |
| M1 `titantpp` | 16 | Hard-LMM | Hard-LMM | shared | 비교 기준 |
| M2 `titantpp_persistent_surprise_memory` | 16 | Surprise | Surprise | shared | Persistent 조건을 맞춘 Surprise 효과 |
| M3a `titantpp_dual_memory_shared` | 16 | Hard-LMM | Hard-LMM + Surprise residual | encoder와 두 memory에 전달 | Dual-route 표현 효과와 gradient 간섭 포함 |
| M3b `titantpp_dual_memory_adapter_only` | 16 | Hard-LMM | Hard-LMM(detached) + Surprise residual | Surprise adapter와 quantity head에만 전달 | Dual-route에서 quantity 간섭 제거 |

M2와 M3의 Surprise residual gate는 0에서 시작한다. 따라서 초기 forward에서 M2는 M0와,
M3a/M3b는 M1과 동일하다. M3의 time loss는 Titan encoder, Hard-LMM, time head만 갱신하며
Surprise adapter를 갱신하지 않는다.

## 고정 조건

- Dataset: `intermittent_frozen_5000`
- Split: 기존 chronological fixed split
- Evaluation: validation only
- Seed: 42
- Maximum epoch: 300
- Early stopping: minimum 40 epochs, patience 40
- Batch size: 128
- Learning rate: `1e-3`
- Gradient clipping: 1
- Lookback: 520 weeks
- Maximum sequence length: 256
- Hidden dimension: 64
- Quantity target: mark-free `log1p(raw quantity)`
- Quantity loss: log-MSE
- Checkpoint selection: minimum validation joint objective
- Held-out test: 사용하지 않음

## Acceptance Gate

M1 Hard-LMM 대비 M2, M3a, M3b에 각각 적용한다.

1. 모든 metric, loss, gradient가 finite여야 한다.
2. 전체 quantity MAE 또는 RMSE가 5% 이상 개선되어야 한다.
3. p95 이하 quantity MAE 악화가 2% 이하여야 한다.
4. Scaled exact Time NLL 악화가 0.01 이하여야 한다.
5. 통과 후보가 없으면 M1을 유지한다.
6. 통과 후보가 있으면 quantity MAE, RMSE, Time NLL 순으로 선택한다.

M0는 persistent token 단독 효과를 해석하는 진단군이며 최종 선택 후보에는 포함하지
않는다. Seed 42 gate를 통과한 구조에만 seeds 52와 62를 추가한다.

## 완료 조건

- Local focused test와 CPU loader smoke 통과
- 5080 CUDA focused test 통과
- Intermittent CUDA model-test 5/5 완료
- Instacart top-20 e1 smoke 5/5 완료
- Intermittent seed-42 e300 5/5 완료
- Comparator가 source, data, split, time head, persistent token, validation-only 계약 검증
- Held-out test artifact가 생성되지 않음
