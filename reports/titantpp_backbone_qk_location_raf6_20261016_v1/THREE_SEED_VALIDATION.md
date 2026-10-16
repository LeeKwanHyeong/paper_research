# RAF Q-only/K-only CNN: 고정 선택 checkpoint의 Validation 비교

평균±표본SD, seed42·52·62. 모든 지표는 낮을수록 좋음. 큰 수량은 Train 고정 q>200. 기존 CNN3는 저장된 Validation 결과를 재사용하며 새 학습은 Q-only/K-only 각3seed 총6개다.

QueryOnly/KeyOnly candidates use5090/cuDNN92000; historical CNN3 uses5080/cuDNN91900. This runtime difference is not removed by retraining the baseline.; 기존 CNN 비교는 환경 차이를 포함한다.
QueryOnly and KeyOnly each retain192 local coefficients; their comparison tests location at equal capacity. Historical CNN576/QKOnly384 differences include reduced capacity and joint optimization. Query filtering also changes memory-key scores; not memory-independent or causal proof.

| 구조 | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 34.4314 ± 0.0732011 | 9.21827 ± 0.0280019 | 330.758 ± 2.554 | 283.555 ± 3.01648 | 3.49746 ± 0.138992 | 3.36318 ± 0.168363 |
| QueryOnly | 34.4764 ± 0.102057 | 9.2332 ± 0.0324379 | 328.99 ± 5.27804 | 281.392 ± 6.41865 | 3.41765 ± 0.0384377 | 3.29344 ± 0.0141059 |
| KeyOnly | 34.3126 ± 0.0666695 | 9.25429 ± 0.0372557 | 327.486 ± 3.60274 | 280.6 ± 3.7997 | 3.55216 ± 0.0981021 | 3.41161 ± 0.124605 |

| seed | 구조 | 선택 epoch | RMSE | MAE |
|---|---|---:|---:|---:|
| 42 | CNN | 9 | 34.4205 | 9.23111 |
| 42 | KeyOnly | 9 | 34.3556 | 9.21164 |
| 42 | QueryOnly | 5 | 34.4944 | 9.19887 |
| 52 | CNN | 5 | 34.3643 | 9.18615 |
| 52 | KeyOnly | 5 | 34.2358 | 9.27077 |
| 52 | QueryOnly | 5 | 34.3665 | 9.23739 |
| 62 | CNN | 8 | 34.5095 | 9.23755 |
| 62 | KeyOnly | 26 | 34.3465 | 9.28047 |
| 62 | QueryOnly | 22 | 34.5682 | 9.26334 |

차이는 앞 구조−뒤 구조. 음수는 개선, 양수는 악화.

| seed | 비교 | RMSE 차이 | RMSE 상대차이 | MAE 차이 | MAE 상대차이 |
|---|---|---:|---:|---:|---:|
| 42 | QueryOnly−CNN | +0.0739274 | +0.215% | -0.0322414 | -0.349% |
| 42 | KeyOnly−CNN | -0.064864 | -0.188% | -0.0194718 | -0.211% |
| 42 | KeyOnly−QueryOnly | -0.138791 | -0.402% | +0.0127696 | +0.139% |
| 52 | QueryOnly−CNN | +0.0021992 | +0.006% | +0.0512395 | +0.558% |
| 52 | KeyOnly−CNN | -0.128482 | -0.374% | +0.0846153 | +0.921% |
| 52 | KeyOnly−QueryOnly | -0.130681 | -0.380% | +0.0333758 | +0.361% |
| 62 | QueryOnly−CNN | +0.0587468 | +0.170% | +0.0257855 | +0.279% |
| 62 | KeyOnly−CNN | -0.163009 | -0.472% | +0.0429171 | +0.465% |
| 62 | KeyOnly−QueryOnly | -0.221756 | -0.642% | +0.0171316 | +0.185% |

전체 지표의 seed별 차이는 JSON에도 모두 보존한다. 기존 Test를 본 뒤 설계한 탐색적 실험이며, 이번 결과는 Test 개선이나 오류 원인 확정의 근거가 아니다.
