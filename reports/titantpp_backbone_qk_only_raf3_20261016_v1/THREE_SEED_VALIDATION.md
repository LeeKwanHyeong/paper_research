# RAF Q/K 전용 CNN: 고정 선택 checkpoint의 Validation 비교

평균±표본SD, seed42·52·62. 모든 지표는 낮을수록 좋음. 큰 수량은 Train 고정 q>200. 기존 CNN3는 저장된 Validation 결과를 재사용하며 새 학습은 Q/K 전용 후보3개뿐이다.

QK-only candidates use5090/cuDNN92000; historical CNN3 uses5080/cuDNN91900. This runtime difference is not removed by retraining the baseline.; 기존 CNN 비교는 환경 차이를 포함한다.
Q/K-only local CNN removes the V local filter (576 to384 coefficients), including capacity and jointly learned path changes. Not a causal proof or a continuation of the fixed-weight off checkpoint.

| 구조 | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 34.4314 ± 0.0732011 | 9.21827 ± 0.0280019 | 330.758 ± 2.554 | 283.555 ± 3.01648 | 3.49746 ± 0.138992 | 3.36318 ± 0.168363 |
| QKOnly | 34.4754 ± 0.0838275 | 9.2319 ± 0.0251877 | 329.204 ± 4.44247 | 281.59 ± 5.57598 | 3.42413 ± 0.0395749 | 3.29239 ± 0.0174984 |

| seed | 구조 | 선택 epoch | RMSE | MAE |
|---|---|---:|---:|---:|
| 42 | CNN | 9 | 34.4205 | 9.23111 |
| 42 | QKOnly | 5 | 34.458 | 9.20926 |
| 52 | CNN | 5 | 34.3643 | 9.18615 |
| 52 | QKOnly | 5 | 34.4016 | 9.22741 |
| 62 | CNN | 8 | 34.5095 | 9.23755 |
| 62 | QKOnly | 22 | 34.5665 | 9.25903 |

차이는 앞 구조−뒤 구조. 음수는 개선, 양수는 악화.

| seed | 비교 | RMSE 차이 | RMSE 상대차이 | MAE 차이 | MAE 상대차이 |
|---|---|---:|---:|---:|---:|
| 42 | QKOnly−CNN | +0.0375632 | +0.109% | -0.0218524 | -0.237% |
| 52 | QKOnly−CNN | +0.0373006 | +0.109% | +0.0412562 | +0.449% |
| 62 | QKOnly−CNN | +0.0570672 | +0.165% | +0.0214777 | +0.233% |

전체 지표의 seed별 차이는 JSON에도 모두 보존한다. 기존 Test를 본 뒤 설계한 탐색적 실험이며, 이번 결과는 Test 개선이나 오류 원인 확정의 근거가 아니다.
