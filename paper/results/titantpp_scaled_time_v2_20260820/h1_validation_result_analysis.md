# TitanTPP Time Head v2 H1 Validation 결과 분석

## 판정

- 실험 상태: 완료
- 범위: Intermittent seed 42, validation only
- 학습 종료: epoch 62 early stopping, best epoch 22
- H1 safety gate: 실패
- 유지 time head: H0 scaled exact reference
- held-out test: 사용하지 않음
- memory 재비교: 열지 않음

H1은 train-only 안정성 기준은 통과했지만 validation Time NLL과 전체 quantity 오차를
보존하지 못했다. 따라서 안정적인 최적화만으로 H1을 공통 time head로 채택할 수
없으며, 이번 H1/H2 강화 경로는 종료한다.

## Matched 계약

H0와 H1은 Intermittent frozen-5000 fixed split, TitanTPP Hard-LMM, mark-free log-MSE
quantity head, seed 42, 최대 300 epoch, minimum epoch 40, patience 40, batch 128,
learning rate `1e-3`, lookback 520주, maximum sequence length 256, hidden dimension 64를
동일하게 사용했다. Checkpoint는 minimum validation joint objective로 선택했다.

변경 축은 time head뿐이다.

| Head | `w * tau` budget | Intercept | 초기값 |
| --- | ---: | --- | --- |
| H0 | 40 | hard clamp `±30` | `log(time_scale)` |
| H1 | 8 | smooth tanh `±6` | train mean event rate |

## Best Validation 결과

| Head | Best epoch | Joint | Time NLL | Log quantity MSE | MAE | RMSE | `<=p95` MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H0 | 22 | 0.826078 | 0.820917 | 0.005161 | 0.645580 | 1.733665 | 0.464500 |
| H1 | 22 | 1.570377 | 1.563867 | 0.006510 | 0.884585 | 2.895075 | 0.473229 |

| Gate | H1 변화 | 허용 범위 | 결과 |
| --- | ---: | ---: | --- |
| Time NLL | `+0.742950` | `<= +0.01` | 실패 |
| Quantity MAE | `+37.02%` | `<= +2%` | 실패 |
| Quantity RMSE | `+66.99%` | `<= +2%` | 실패 |
| `<=p95` MAE | `+1.88%` | `<= +2%` | 통과 |

## Quantity 구간별 결과

| Quantity 구간 | H0 MAE | H1 MAE | 변화 |
| --- | ---: | ---: | ---: |
| `<=2` | 0.077671 | 0.093153 | `+19.93%` |
| `(2,31]` | 0.862097 | 0.863955 | `+0.22%` |
| `(31,46]` | 1.259104 | 1.253659 | `-0.43%` |
| `(46,187]` | 2.852034 | 5.379289 | `+88.61%` |
| `>187` | 7.033215 | 16.940126 | `+140.86%` |

Body의 중간 수량은 대체로 유지됐지만 low quantity와 tail이 악화됐다. 특히 p95를
초과하는 수량에서 오차가 크게 증가해 전체 MAE와 RMSE 실패를 설명한다.

## History 길이별 결과

| History 구간 | H0 MAE | H1 MAE | 변화 |
| --- | ---: | ---: | ---: |
| `<=64` | 0.803287 | 0.815873 | `+1.57%` |
| `65-128` | 1.116519 | 1.712261 | `+53.36%` |
| `>128` | 0.115431 | 0.217881 | `+88.75%` |

H1은 short history에서는 오차를 거의 보존했지만 중간·긴 history에서는 quantity
오차가 커졌다. 안정화된 time head가 긴 history 표현의 성능을 개선했다는 근거는
확인되지 않았다.

## 학습 안정성과 성능의 분리

H1의 train joint objective는 `1.6186-1.7486`, epoch 평균 gradient clipping 비율은
약 `1.15%`, 최대 clipping 비율은 `3.41%`로 안정적이었다. 62개 epoch의 모든 수치는
finite였고 best epoch 22 이후 patience 40을 정확히 채워 종료됐다.

반면 best epoch에서 learned time slope는 H1 상한 `0.666667`에 포화됐다. 확인된
사실은 H1이 수치적으로 안정적이지만 Time NLL과 tail quantity에서 실패했다는 점이다.
Slope 포화는 `w * tau <= 8` 제약이 이 데이터의 time likelihood에 지나치게 강했을
가능성을 뒷받침한다. Shared backbone에서 time objective의 적합도 저하가 quantity
tail 표현에도 영향을 줬을 수 있지만, 이 인과 해석은 추가 분리 실험 전까지 추론으로
한정한다.

## Artifact 검증

- Launch contract는 H1 stable exact, seed 42, validation-only 계약과 일치했다.
- 학습은 정상 완료됐고 NaN 또는 non-finite metric이 없었다.
- Summary, quantity/history scale-wise CSV의 모든 수치는 finite였다.
- Best checkpoint의 embedded, summary, recomputed state SHA-256이 일치했다.
- held-out test artifact와 plot은 생성되지 않았다.
- 초기 comparator import 오류는 postprocessing에만 발생했고 `5cc862b`에서 복구했다.

## 결론

H1은 H0의 폭발적인 train loss를 제거했지만, 그 대가로 validation Time NLL과 quantity
tail 성능을 크게 잃었다. 따라서 H1과 조건부 H2를 채택하지 않고 H0를 유지한다.
Stable time head가 validation safety를 통과하지 않았으므로 계획했던 memory 구조
재비교도 시작하지 않는다.

다음 time-head 가설을 열 경우에는 slope budget을 단순히 낮추는 방식보다 time과
quantity의 shared-backbone gradient 간섭, slope saturation, time-loss weighting을
먼저 분리 진단해야 한다.
