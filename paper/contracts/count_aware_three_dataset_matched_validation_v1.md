# 세 데이터셋 Count-aware T0·TitanTPP-T1 matched validation 계약 v1

- 동결일: 2026-08-24
- 실행 서버: 5080 only
- 평가 범위: validation-only
- held-out test: 사용하지 않음

## 비교 목적

Intermittent v2에서 선정한 TitanTPP-T1이 서로 다른 수량 분포와 이력 길이에서도
같은 개선 방향을 보이는지 확인한다. 모델 구조, time head와 loss 식은 변경하지
않는다. T0는 backbone 차이, T1은 TitanTPP에 tail-aware auxiliary loss를 추가한
현재 대표 방법을 나타낸다.

## 공통 비교 축

| 역할 | Backbone | Quantity objective | Time head |
| --- | --- | --- | --- |
| T0 common control | RMTPP, THP, NHP, SAHP, TitanTPP | direct `log1p(quantity)` MSE | `legacy_clamped_rmtpp` |
| TitanTPP-T1 | TitanTPP Hard-LMM | T0 + train-only tail raw-quantity Huber | `legacy_clamped_rmtpp` |

T1의 `lambda_tail=0.09111380335463036`, Huber delta `1.0`, gradient route
`quantity head + Titan encoder`는 세 데이터셋에서 동일하다. 데이터셋별 tail
상수만 각 train split의 nearest `p95/p95/p99`로 고정한다. Validation 결과를
보고 상수를 바꾸지 않는다.

## 데이터셋별 고정 조건

| Dataset | 시간 단위 | Lookback | Max sequence length | T1 threshold / normalization / cap |
| --- | --- | ---: | ---: | --- |
| Intermittent v2 | week | 520 | 256 | 46 / 46 / 187 |
| Online Retail II | hour | 8,760 | 256 | 40 / 40 / 144 |
| RAF Spare Parts | month | 84 | 84 | 60 / 60 / 200 |

Lookback은 시간 범위이고 max sequence length는 그 범위에서 모델에 전달할 최근
event 수의 상한이다. 따라서 데이터셋의 시간 해상도에 맞춰 lookback은 달라지지만,
각 데이터셋 안에서는 모든 backbone과 T0/T1에 동일하게 적용한다.

## 공통 학습과 선택 규칙

- seeds: 42, 52, 62
- maximum/minimum epochs: 300 / 40
- early-stopping patience: 40
- batch size: 128
- learning rate: 0.001
- hidden dimension: 64
- gradient clipping: 1.0
- checkpoint: minimum validation joint objective
- 공식 비교 전 e1 smoke는 seed 42, 최대 train/validation batch 2의 partial run으로만 사용
- smoke 및 validation에서 held-out test artifact를 생성하지 않음

## 해석 경계

세 데이터셋에서 T1이 개선되면 tail-aware objective의 범용성을 지지한다. 그 결과만으로
Titan backbone 자체의 우월성을 주장하지는 않는다. TitanTPP-T0와 다른 T0 backbone의
matched 결과가 backbone 효과의 근거이고, TitanTPP-T0 대비 T1 차이는 objective 효과의
근거로 분리한다.
