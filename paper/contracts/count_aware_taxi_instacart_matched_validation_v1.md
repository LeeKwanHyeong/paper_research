# Taxi·Instacart mark-free T0·TitanTPP-T1 matched validation 계약 v1

- 동결일: 2026-08-24
- 실행 서버: 5080 only
- 평가 범위: validation-only
- held-out test: 사용하지 않음

## 목적

기존 Taxi·Instacart e300 결과는 quantity-derived mark를 함께 학습한 과거 계약이므로,
현재 mark-free count-aware 결과와 직접 결합하지 않는다. 두 데이터셋을
`event time + continuous quantity` 예측으로 다시 고정하고, T0 backbone 차이와
TitanTPP-T1 tail-aware objective 효과를 분리해서 비교한다.

## Mark-free 입력 계약

원본 parquet에 mark 및 residual 열이 남아 있어도 모델 입력에는 사용하지 않는다.
공통 runner의 `prepare_count_frame()`이 `demand_qty`를 연속 수량 target으로 사용하고
runtime mark를 모두 0으로 덮어쓴다. 따라서 이번 실행의 mark는 제품 종류나 수량
구간을 의미하지 않으며 mark loss도 학습 목적에 포함되지 않는다.

## 데이터와 문맥 범위

| Dataset | 고정 parquet | 시간 단위 | Lookback | Max sequence length | T1 threshold / normalization / cap |
| --- | --- | --- | ---: | ---: | --- |
| Taxi | `yellow_trip_hourly_with_split.parquet` | hour | 168 | 256 | 1,562 / 1,562 / 3,449 |
| Instacart | `instacart_marked_target_with_split.parquet` | day | 52 | 64 | 25 / 25 / 35 |

Lookback은 target 이전에 포함할 시간 범위이고, max sequence length는 그 범위에서
모델에 전달할 최근 event 수의 상한이다. 데이터와 split manifest의 SHA-256은 JSON
계약과 runner registry에서 함께 검증한다. T1 상수는 train split의 nearest
`p95/p95/p99`로 미리 고정하며 validation을 보고 바꾸지 않는다.

## Factorial 계약

| 역할 | Backbone | Quantity objective | Time head |
| --- | --- | --- | --- |
| T0 common control | RMTPP, THP, NHP, SAHP, TitanTPP | direct `log1p(quantity)` MSE | `legacy_clamped_rmtpp` |
| TitanTPP-T1 | TitanTPP Hard-LMM | T0 + train-only tail raw-quantity Huber | `legacy_clamped_rmtpp` |

T0는 backbone 효과를 비교하고, TitanTPP-T0와 T1의 차이는 objective 효과를
비교한다. T1 결과만으로 Titan backbone 자체의 우월성을 주장하지 않는다.

## 공통 학습 조건

- seeds: 42, 52, 62
- maximum/minimum epochs: 300 / 40
- early-stopping patience: 40
- batch size: 128
- learning rate: 0.001
- hidden dimension: 64
- gradient clipping: 1.0
- checkpoint: minimum validation joint objective
- T1 `lambda_tail`: 0.09111380335463036
- held-out test artifact: 생성 금지

e1 smoke는 seed 42, 상위 20 series, 최대 train/validation batch 2만 사용한다. 이는
CUDA forward/backward, checkpoint와 artifact 경로 확인용이며 성능 근거로 사용하지
않는다.
