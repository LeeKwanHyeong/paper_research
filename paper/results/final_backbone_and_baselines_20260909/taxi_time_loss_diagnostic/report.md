# Taxi 시간 loss 증가: 학습 이력과 기존 head 진단

2026-09-10. 신규 학습·checkpoint 재선택·held-out 평가 없이 저장된 JSON 이력과 train/validation 입력을 확인했다. 소스는 `ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c`이다. 이번 점수는 정상화된 Time NLL이 아닌 기존 `legacy_clamped_rmtpp` 시간 loss다.

## 확인된 핵심

신규 후보만 갑자기 불안정해진 것이 아니다. 기존 B도 학습 후반 validation 시간 loss가 커졌다. B는 수량 RMSE 최적 epoch45를 선택했지만 후보는 epoch83을 선택해 후반 시간 손실 악화가 최종 비교에 포함됐다. 후보에서 같은 epoch의 시간 loss도 더 높아지므로 선택 시점만으로 차이 전체를 설명할 수는 없다.

| Epoch | B validation 시간 loss | 후보 validation 시간 loss |
|---|---:|---:|
| 30 | 1.3827 | 1.3749 |
| 45 | 1.4734 (B 선택) | 1.4401 |
| 50 | 1.5084 | 1.4770 |
| 60 | 1.6582 | 2.5199 |
| 70 | 5.8463 | 5.4477 |
| 83 | 8.0080 | 10.8259 (후보 선택) |

후보 epoch83의 train 시간 loss는 1.2601이며 validation은 10.8259다. 두 run의 모든 epoch에 `train_all_finite=true`가 기록돼 있다. NaN/Inf 또는 checkpoint 복원 오류로 확인된 현상이 아니라, 유한한 시간 loss의 train/validation 격차다. 큰 유한 gradient까지 없었다는 뜻은 아니다.

## 어느 집단에서 증가했는가

동일 validation 8,268개에 대해 B 선택 epoch45와 후보 선택 epoch83을 비교했다.

| 이력 길이 | 표본 수 | B 시간 loss | 후보 시간 loss | 전체 평균 증가분 기여 |
|---|---:|---:|---:|---:|
| 64개 이하 | 1,241 | 3.3072 | 64.9782 | +9.2566 |
| 65–128개 | 1,475 | 1.5786 | 2.2673 | +0.1229 |
| 128개 초과 | 5,552 | 1.0355 | 0.9953 | −0.0270 |

전체 증가 +9.3525의 **98.975%**가 이력 64개 이하 집단에서 발생했다. 이는 표본의 15.01%다. 별도의 수량 구간 분해에서도 수량 7 이하 집단의 증가가 +9.4012이고, 나머지 수량 구간은 감소한다. 두 분해는 동일 표본을 서로 다른 기준으로 나눈 것이므로 기여를 합산하지 않는다.

이력은 168시간 window 내 관측 사건 수이며 이번 Taxi 입력의 실제 최댓값은168개다. 따라서 이력 64개 이하가 반드시 전체 series의 시작 부분을 의미하지는 않는다. 입력 재구성으로 해당 validation 집단의 수량 평균은 약1.408, 시간 간격 평균은 약4.990시간이며, validation의 15시간 초과 간격 86개 중84개가 이 집단에 속함을 확인했다.

단, train의 초기 window 형성 구간을 제외하고 비교하면 유사한 희소 집단이 train에도 있다. 전체 short-history train/validation 평균 차이를 곧바로 조건부 분포 이동으로 해석하면 안 된다. 재현 가능한 계산과 집단별 통계는 `target_duration_profile.py/json`에 기록한다.

| 관측 기간168시간이 확보된 이력64개 이하 집단 | Train | Validation |
|---|---:|---:|
| 표본 수 | 3,477 | 1,241 |
| 평균 시간 간격 | 5.0946시간 | 4.9895시간 |
| 1시간 간격 비중 | 32.50% | 35.70% |
| 간격 p99 | 31시간 | 31시간 |
| 평균 수량 | 1.4032 | 1.4077 |

이 비교는 기본 분포가 유사함을 보여주며, 모든 조건부 분포가 같다고 입증하는 검정은 아니다. 전체 validation 간격 최댓값67시간은 train 최댓값115시간보다 작다. 새로운 극단 간격이 validation에만 나타났다는 설명도 맞지 않는다.

## 수식과 학습 경로가 보여주는 설명

실제 legacy 시간 loss는 다음과 같다.

`w = softplus(w_raw) + 0.001`

`a = min(v_t(h) + b_t, 300)`

`z = min(w * target_duration, 10)`

`loss = -a - z + exp(a) / w * expm1(z)`

- `w`에는 실질적인 상한이 없다. 메타데이터의 `time_scale=3`, `time_w_max=10/3`은 이 legacy 분기의 계산에 적용되지 않는다. B와 후보 모두 같은 계산이므로 양 모델 간 단위 불일치는 아니다.
- `z`의 상한10은 최종 loss의 상한이 아니다. 긴 간격이나 큰 intercept에서 누적위험 항이 매우 커질 수 있다.
- 후보의 기록된 slope는 epoch45의0.0821에서 epoch83의0.2931로 증가했다. B도 같은 기간0.0772에서0.1929로 증가했다. 긴 간격에 대한 민감도가 커지는 수식과 시간 loss 궤적이 부합한다.
- 후보 memory는 encoder layer1과2 사이에서 표현을 바꾸고, 최종 표현을 수량·시간 head가 함께 사용한다. 수량 경로가 좋아지는 동안 시간 intercept도 달라질 수 있다.
- Train joint loss에는 시간 loss가 포함돼 있다. 다만 checkpoint 선택과 patience 갱신에는 validation raw RMSE만 사용한다. 따라서 validation 시간 악화가 좋은 수량 checkpoint의 선택을 막지 않는다.

소스 근거: `/tmp/paper_research_dual_timescale/models/TPPs/CountAwareTPP.py:436` 및 `:551`, `:1216`; `models/TPPs/CountAwareTitanDualTimescale.py:269`; `paper/scripts/count_aware_tpp_backbone/core.py:69`; `training.py:56`, `:877`.

이 내용은 가능한 증폭 경로를 설명한다. 개별 표본의 hidden/intercept와 시간 slope를 분리한 개입 실험을 하지 않았으므로 **memory가 원인인지, slope 학습이 원인인지, 양자의 상호작용인지 기여율을 확정한 것은 아니다.** 시간 loss 증가를 시간 median MAE/RMSE 악화와 동일시하지도 않는다.

## 수량 개선의 해석과 기존 판정

| 기록된 시점 | Raw RMSE | 전체 MAE | 기존 시간 loss |
|---|---:|---:|---:|
| B 선택 epoch45 | 88.1950 | 28.6740 | 1.4734 |
| 후보 epoch45 | 80.7813 | 25.7188 | 1.4401 |
| 후보 선택 epoch83 | 78.2554 | 25.0483 | 10.8259 |

후보 epoch45는 B의 선택 결과보다 RMSE8.41%, MAE10.31% 개선되고 시간 loss도 낮았다. 이는 학습 이력에서 확인한 탐색적 관측이다. 이 시점의 body/tail 전체 gate를 평가하거나 checkpoint를 채택한 것이 아니며, 결과를 본 뒤 선택 규칙을 변경하지 않는다.

따라서 이 실험에서 수량 개선 가능성이 없었다고 결론 내릴 수 없다. 확인된 실패는 **사전 RMSE selector가 선택한 최종 후보 checkpoint의 시간 guardrail 실패**다. 기존 실패 기록, TitanTPP(B) 유지, 후속 학습 중단 결정은 변경하지 않는다.

검증: B85개·후보123개 epoch, summary와 earliest RMSE 최솟값 일치, 시간 loss summary/history 일치, 양 모델의 수량·이력 구간별 가중 평균 재현을 확인했다. 입력 SHA와 계산 결과는 `log_diagnostic.json`, 전체208행 궤적은 `epoch_trajectory.csv`에 저장했다.
