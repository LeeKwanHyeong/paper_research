# RAF Current/Temporal FFN CNN: 고정 선택 checkpoint의 Validation 비교

평균±표본SD, seed42·52·62. 모든 지표는 낮을수록 좋음. 큰 수량은 Train 고정 q>200. 기존 CNN3는 저장된 Validation 결과를 재사용하며 새 학습은 Current/Temporal FFN 각3seed 총6개다.

Current/Temporal candidates use5090/cuDNN92000; historical CNN3 uses5080/cuDNN91900. This runtime difference is not removed by retraining the baseline.; 기존 CNN 비교는 환경 차이를 포함한다.
Current and Temporal each add1152 FFN parameters with the same initialization/availability. Current uses an attention-prefix state, not history-free input. Equal counts do not imply equal function spaces or optimization; common trainable weights and memory retrieval may change. Stored CNN has fewer parameters and different Runtime.

| 구조 | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 34.4314 ± 0.0732011 | 9.21827 ± 0.0280019 | 330.758 ± 2.554 | 283.555 ± 3.01648 | 3.49746 ± 0.138992 | 3.36318 ± 0.168363 |
| Current | 34.466 ± 0.0715678 | 9.24984 ± 0.0122929 | 327.184 ± 3.32353 | 279.016 ± 4.17458 | 3.47686 ± 0.0664997 | 3.33634 ± 0.0926826 |
| Temporal | 34.4721 ± 0.0666906 | 9.2236 ± 0.0151901 | 329.294 ± 4.6148 | 281.774 ± 5.66158 | 3.43876 ± 0.0259639 | 3.30276 ± 0.0220471 |

| seed | 구조 | 선택 epoch | RMSE | MAE |
|---|---|---:|---:|---:|
| 42 | CNN | 9 | 34.4205 | 9.23111 |
| 42 | Current | 9 | 34.4737 | 9.25032 |
| 42 | Temporal | 5 | 34.4714 | 9.20613 |
| 52 | CNN | 5 | 34.3643 | 9.18615 |
| 52 | Current | 5 | 34.391 | 9.23731 |
| 52 | Temporal | 5 | 34.4057 | 9.23097 |
| 62 | CNN | 8 | 34.5095 | 9.23755 |
| 62 | Current | 22 | 34.5335 | 9.26188 |
| 62 | Temporal | 22 | 34.5391 | 9.23369 |

차이는 앞 구조−뒤 구조. 음수는 개선, 양수는 악화.

| seed | 비교 | RMSE 차이 | RMSE 상대차이 | MAE 차이 | MAE 상대차이 |
|---|---|---:|---:|---:|---:|
| 42 | Current−CNN | +0.0532329 | +0.155% | +0.0192075 | +0.208% |
| 42 | Temporal−CNN | +0.0509758 | +0.148% | -0.0249835 | -0.271% |
| 42 | Temporal−Current | -0.00225708 | -0.007% | -0.044191 | -0.478% |
| 52 | Current−CNN | +0.0266559 | +0.078% | +0.0511608 | +0.557% |
| 52 | Temporal−CNN | +0.0414065 | +0.120% | +0.0448168 | +0.488% |
| 52 | Temporal−Current | +0.0147507 | +0.043% | -0.00634392 | -0.069% |
| 62 | Current−CNN | +0.0240106 | +0.070% | +0.0243329 | +0.263% |
| 62 | Temporal−CNN | +0.0296158 | +0.086% | -0.00385874 | -0.042% |
| 62 | Temporal−Current | +0.00560524 | +0.016% | -0.0281917 | -0.304% |

전체 지표의 seed별 차이는 JSON에도 모두 보존한다. 기존 Test를 본 뒤 설계한 탐색적 실험이며, 이번 결과는 Test 개선이나 오류 원인 확정의 근거가 아니다.
