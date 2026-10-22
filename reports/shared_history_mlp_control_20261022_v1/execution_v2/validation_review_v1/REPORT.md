# B·단일 MLP·D Validation 비교

## Material Passport

- 범위: execution_v2, Validation-only 중간 분석. 비용 결과와 Test는 포함하지 않는다.
- Verification Status: ANALYZED. 원예측 27개의 수량 지표와 recorded TimeNLL 집계 재계산 PASS; 새로운 모델 추론은 0회다.
- seed42/52/62의 산술평균 ± 표본SD(ddof=1). 모든 오차·NLL은 낮을수록 좋다.
- B=공유15차원 이력 보정, MLP=단일 보정 MLP, D=공통 encoder. B·MLP 보정 파라미터 각9600이며 활성 수와 함수 집합까지 같지는 않다.
- 선택: 전체 Validation raw 수량 RMSE의 첫 strict minimum checkpoint. 시간 NLL 최적 checkpoint의 비교가 아니다.

## 전체 Validation

| 데이터 | 모델 | 수량 RMSE | 수량 MAE | recorded TimeNLL |
|---|---|---:|---:|---:|
| RAF | B | 33.949746 ± 0.168160 | 9.253158 ± 0.085728 | 3.516363 ± 0.073861 |
| RAF | 단일 MLP | 33.927819 ± 0.215592 | 9.274166 ± 0.070174 | 3.596959 ± 0.114142 |
| RAF | D | 34.333827 ± 0.079408 | 9.262833 ± 0.050869 | 3.584953 ± 0.126723 |
| Taxi | B | 78.612703 ± 1.014493 | 25.410971 ± 0.375703 | 1.312046 ± 0.325451 |
| Taxi | 단일 MLP | 78.102753 ± 1.055853 | 25.129440 ± 0.110200 | 1.164093 ± 0.192476 |
| Taxi | D | 88.796038 ± 0.274725 | 28.313275 ± 0.096985 | 0.798373 ± 0.071117 |
| Intermittent | B | 1.745321 ± 0.157812 | 0.734613 ± 0.049950 | 0.371252 ± 0.116563 |
| Intermittent | 단일 MLP | 1.668509 ± 0.022268 | 0.679059 ± 0.022625 | 0.335987 ± 0.066705 |
| Intermittent | D | 1.724491 ± 0.043399 | 0.708979 ± 0.017106 | 0.365529 ± 0.031384 |

## 큰 수량 Validation

| 데이터 | 모델 | 수량 RMSE | 수량 MAE | recorded TimeNLL |
|---|---|---:|---:|---:|
| RAF | B | 325.333234 ± 2.160558 | 279.313824 ± 2.122709 | 3.422817 ± 0.238698 |
| RAF | 단일 MLP | 320.054948 ± 5.301622 | 272.977684 ± 5.870411 | 3.506596 ± 0.203994 |
| RAF | D | 328.434987 ± 4.595031 | 281.758457 ± 5.172433 | 3.455149 ± 0.160237 |
| Taxi | B | 357.636543 ± 6.389255 | 281.102156 ± 10.481469 | 1.48280e-38 ± 2.56828e-38 |
| Taxi | 단일 MLP | 384.677209 ± 7.946635 | 306.588376 ± 9.676862 | 5.15580e-30 ± 8.93010e-30 |
| Taxi | D | 384.523640 ± 8.409337 | 301.464080 ± 4.744608 | 1.96247e-47 ± 3.39910e-47 |
| Intermittent | B | 8.408825 ± 1.183614 | 6.812036 ± 1.034617 | 3.642358 ± 5.981930 |
| Intermittent | 단일 MLP | 7.941313 ± 0.690679 | 6.167618 ± 0.627202 | 3.70761e-07 ± 3.48260e-07 |
| Intermittent | D | 8.038297 ± 0.265628 | 6.149397 ± 0.188906 | 0.336525 ± 0.509768 |

## 큰 수량 정의

| 데이터 | 전체 수 | 큰 수량 조건 | 큰 수량 수 |
|---|---:|---|---:|
| RAF | 6690 | q > 200 | 50 |
| Taxi | 8268 | q > 3449 | 79 |
| Intermittent | 86285 | q > 187 | 1141 |

## 모든 seed의 악화 위치

아래는 왼쪽 모델의 원래 값이 오른쪽보다 큰 seed다. 극소 TimeNLL 차이까지 산술적으로 포함하므로 이 표를 승패 점수로 합산하지 않는다.

| 데이터·범위·지표 | B−MLP 악화 seed | B−D 악화 seed | MLP−D 악화 seed |
|---|---|---|---|
| RAF·overall·qty_rmse | 62 | 없음 | 없음 |
| RAF·overall·qty_mae | 52 | 42 | 42 |
| RAF·overall·time_nll | 없음 | 42, 52 | 42, 52 |
| RAF·large·qty_rmse | 42, 52, 62 | 52 | 52 |
| RAF·large·qty_mae | 42, 52, 62 | 52 | 52 |
| RAF·large·time_nll | 42, 52 | 42, 52 | 42, 52, 62 |
| Taxi·overall·qty_rmse | 42, 62 | 없음 | 없음 |
| Taxi·overall·qty_mae | 42, 62 | 없음 | 없음 |
| Taxi·overall·time_nll | 42, 52 | 42, 52, 62 | 42, 52, 62 |
| Taxi·large·qty_rmse | 없음 | 없음 | 42 |
| Taxi·large·qty_mae | 없음 | 없음 | 42, 62 |
| Taxi·large·time_nll | 62 | 62 | 52 |
| Intermittent·overall·qty_rmse | 62 | 62 | 없음 |
| Intermittent·overall·qty_mae | 42, 52, 62 | 62 | 62 |
| Intermittent·overall·time_nll | 52, 62 | 62 | 42, 62 |
| Intermittent·large·qty_rmse | 62 | 52, 62 | 42, 52 |
| Intermittent·large·qty_mae | 62 | 62 | 42, 52 |
| Intermittent·large·time_nll | 42, 52, 62 | 42, 62 | 42 |

## 원본 연결

모든 27개 원 receipt/prediction 경로·SHA·split·선택 epoch는 REPORT.json의 records/evidence_sha256에 저장한다. PAIRED_SEED_DELTAS.csv는 162개 모든 seed·지표 차이를 포함한다. 차이는 왼쪽−오른쪽, relative는 그 차이/오른쪽이며 양수는 악화다. 평균 차이는 평균들의 차이로 계산하며 seed별 상대차이의 평균과 구분한다.

- 전체 읽은 예측 행: 911187
- receipt 대비 최대 절대 차이: 2.984279490192421e-13
- Taxi 큰 수량의 극소 TimeNLL을 반올림된 정확한 0이나 완벽한 시간 예측으로 해석하지 않는다.
- 같은 Validation으로 checkpoint를 선택하고 구조를 비교했다. held-out 일반화·통계적 유의성·동등성은 이 표로 확정하지 않는다.
