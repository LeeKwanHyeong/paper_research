# RAF V 관측차분 CNN: 고정 선택 checkpoint의 Validation 비교

평균±표본SD, seed42·52·62. 모든 지표는 낮을수록 좋음. 큰 수량은 Train 고정 q>200. 기존 CNN3는 저장된 Validation 결과를 재사용하며 새 학습은 V 관측차분 후보3개뿐이다.

V-difference candidates use5090/cuDNN92000; historical CNN3 uses5080/cuDNN91900. This runtime difference is not removed by retraining the baseline.; 기존 CNN 비교는 환경 차이를 포함한다.
V-only observed differences change degrees of freedom (576 to512), prefix boundaries and optimizer parameterization together. No isolated causal attribution.

| 구조 | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 34.4314 ± 0.0732011 | 9.21827 ± 0.0280019 | 330.758 ± 2.554 | 283.555 ± 3.01648 | 3.49746 ± 0.138992 | 3.36318 ± 0.168363 |
| ValueDifference | 34.4458 ± 0.0677886 | 9.25062 ± 0.0323647 | 327.392 ± 2.52347 | 279.351 ± 3.54612 | 3.47172 ± 0.0651615 | 3.33541 ± 0.0925756 |

| seed | 구조 | 선택 epoch | RMSE | MAE |
|---|---|---:|---:|---:|
| 42 | CNN | 9 | 34.4205 | 9.23111 |
| 42 | ValueDifference | 9 | 34.41 | 9.26609 |
| 52 | CNN | 5 | 34.3643 | 9.18615 |
| 52 | ValueDifference | 5 | 34.4034 | 9.21342 |
| 62 | CNN | 8 | 34.5095 | 9.23755 |
| 62 | ValueDifference | 22 | 34.524 | 9.27235 |

차이는 앞 구조−뒤 구조. 음수는 개선, 양수는 악화.

| seed | 비교 | RMSE 차이 | RMSE 상대차이 | MAE 차이 | MAE 상대차이 |
|---|---|---:|---:|---:|---:|
| 42 | ValueDifference−CNN | -0.0104366 | -0.030% | +0.0349803 | +0.379% |
| 52 | ValueDifference−CNN | +0.0391112 | +0.114% | +0.0272721 | +0.297% |
| 62 | ValueDifference−CNN | +0.0145192 | +0.042% | +0.0347957 | +0.377% |

전체 지표의 seed별 차이는 JSON에도 모두 보존한다. 기존 Test를 본 뒤 설계한 탐색적 실험이며, 이번 결과는 Test 개선이나 오류 원인 확정의 근거가 아니다.
