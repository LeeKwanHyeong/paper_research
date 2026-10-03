# TitanTPP(B) · RMTPP · THP: seed42 validation 비교

세 모델 모두 validation raw RMSE의 가장 이른 최솟값으로 checkpoint를 선택했다. 기존 결과를 재사용하고 빠진 네 baseline만 추가했다. 학습은 최대300/minimum40/patience40, 같은 데이터·대상 표본·수량 loss 조건이다.

## Raw RMSE ↓

| 데이터셋 | TitanTPP(B) | RMTPP | THP |
|---|---:|---:|---:|
| Intermittent | **1.499555** | 1.833607 | 1.649723 |
| Taxi | **88.194997** | 93.286440 | 100.023882 |
| Instacart | 5.872217 | **5.836961** | 5.852342 |

## 전체 MAE ↓

| 데이터셋 | TitanTPP(B) | RMTPP | THP |
|---|---:|---:|---:|
| Intermittent | 0.604340 | 0.669522 | **0.598333** |
| Taxi | 28.674020 | **28.196301** | 31.557175 |
| Instacart | 3.993781 | **3.985841** | 3.991003 |

TitanTPP(B)는 Intermittent와 Taxi의 RMSE에서 두 baseline보다 낮다. Instacart에서는 RMTPP와 THP가 더 낮다. 전체 MAE의 최솟값은 Intermittent에서 THP, Taxi와 Instacart에서 RMTPP다.

현재 증적은 단일 seed의 validation 비교이며 다중 seed 또는 held-out 우위는 확정하지 않는다. NHP·SAHP의 raw-RMSE selector 정렬 비교는 이 표에 포함되지 않았다. 신규 dual-timescale Backbone은 이 표의 범위 밖이다.

CSV/JSON에는 body·>p99 MAE, legacy clamped time loss, 선택·종료 epoch와 출처 SHA도 기록했다. 이 시간 점수를 정상화된 Time NLL이라고 해석하지 않는다. 9개 행의 대상 identity·quantity hash, 처리 건수, 선택 규칙과 지표 재구성 검증을 통과했다.

B 증적은 원래 B/C 공용 launcher에서 개별 B run을 선별했다. 공용 adaptive strength=1과 Instacart launcher의 running 상태를 그대로 보존했으며, B 자체의 success·비적응형 log-regression variant·완료 이력과 고정 summary/history/launch/checkpoint-state SHA를 확인했다. JSON의 passed는 증적 감사 통과를 뜻하며 모델 성능 gate 통과를 뜻하지 않는다.
