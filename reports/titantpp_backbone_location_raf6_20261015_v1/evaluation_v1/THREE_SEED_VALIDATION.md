# RAF Query·Value 및 고정 기준선 3seed Validation 비교

신규 Query3·Value3은 선택 checkpoint의 native Validation 재현값이다. 기존 CNN3·QKV Channel3은 원학습의 선택 epoch Validation 값을 재사용하며, Channel은 당시 native 재현 증적과도 대조했다.
같은 split·대상을 확인했다. CNN 대비는 source·Runtime, 저장된 Channel 대비는 source·추가 용량 차이가 있어 Value−Query와 구분한다.

| 구조 | RMSE 평균±표본SD | MAE 평균±표본SD |
|---|---:|---:|
| titantpp_cnn | 34.431408±0.073201 | 9.218271±0.028002 |
| titantpp_cnn_query_nonlinear_v1 | 34.557754±0.088215 | 9.241208±0.035806 |
| titantpp_cnn_value_nonlinear_v1 | 34.452348±0.056591 | 9.267819±0.040926 |
| titantpp_cnn_channel_nonlinear_v1 | 34.561703±0.072617 | 9.238867±0.037861 |

| seed | 구조 | 선택 epoch | RMSE | MAE |
|---:|---|---:|---:|---:|
| 42 | titantpp_cnn | 9 | 34.420453 | 9.231110 |
| 42 | titantpp_cnn_channel_nonlinear_v1 | 9 | 34.545787 | 9.280974 |
| 42 | titantpp_cnn_query_nonlinear_v1 | 9 | 34.542090 | 9.280118 |
| 42 | titantpp_cnn_value_nonlinear_v1 | 9 | 34.460564 | 9.261440 |
| 52 | titantpp_cnn | 5 | 34.364303 | 9.186152 |
| 52 | titantpp_cnn_channel_nonlinear_v1 | 5 | 34.498364 | 9.207630 |
| 52 | titantpp_cnn_query_nonlinear_v1 | 5 | 34.478421 | 9.209645 |
| 52 | titantpp_cnn_value_nonlinear_v1 | 5 | 34.392098 | 9.230458 |
| 62 | titantpp_cnn | 8 | 34.509470 | 9.237552 |
| 62 | titantpp_cnn_channel_nonlinear_v1 | 22 | 34.640958 | 9.227997 |
| 62 | titantpp_cnn_query_nonlinear_v1 | 22 | 34.652752 | 9.233861 |
| 62 | titantpp_cnn_value_nonlinear_v1 | 26 | 34.504382 | 9.311560 |

| seed | 비교(앞−뒤) | ΔRMSE | ΔMAE |
|---:|---|---:|---:|
| 42 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn | +0.040112 | +0.030330 |
| 42 | titantpp_cnn_query_nonlinear_v1−titantpp_cnn | +0.121638 | +0.049008 |
| 42 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn_query_nonlinear_v1 | -0.081526 | -0.018678 |
| 42 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn_channel_nonlinear_v1 | -0.085223 | -0.019534 |
| 42 | titantpp_cnn_query_nonlinear_v1−titantpp_cnn_channel_nonlinear_v1 | -0.003697 | -0.000856 |
| 52 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn | +0.027796 | +0.044306 |
| 52 | titantpp_cnn_query_nonlinear_v1−titantpp_cnn | +0.114118 | +0.023493 |
| 52 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn_query_nonlinear_v1 | -0.086322 | +0.020813 |
| 52 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn_channel_nonlinear_v1 | -0.106266 | +0.022828 |
| 52 | titantpp_cnn_query_nonlinear_v1−titantpp_cnn_channel_nonlinear_v1 | -0.019943 | +0.002014 |
| 62 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn | -0.005087 | +0.074009 |
| 62 | titantpp_cnn_query_nonlinear_v1−titantpp_cnn | +0.143283 | -0.003690 |
| 62 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn_query_nonlinear_v1 | -0.148370 | +0.077699 |
| 62 | titantpp_cnn_value_nonlinear_v1−titantpp_cnn_channel_nonlinear_v1 | -0.136576 | +0.083563 |
| 62 | titantpp_cnn_query_nonlinear_v1−titantpp_cnn_channel_nonlinear_v1 | +0.011794 | +0.005864 |

큰수량 q>200 및 전체/큰수량 TimeNLL 등 여섯 지표와 seed별 절대·상대 차이는 JSON에 보존했다. 새 학습 6개만 평가했으며, 기준선 6개를 재평가하지 않았다.
