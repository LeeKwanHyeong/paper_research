# 원본 공유형 MLP와 기존 MLP16의 3seed Validation

상한 적용 전 원본·log1p 입력·log-MSE·시간 가중치1. 평균±표본SD(ddof1).
선택은 전체 Validation 실수량 RMSE 최초 strict finite 최소epoch이며 MAE도 같은 checkpoint 값이다.

| 데이터 | 모델 | RMSE | MAE | 큰수량 RMSE | 큰수량 MAE | 전체 TimeNLL | 큰수량 TimeNLL |
|---|---|---:|---:|---:|---:|---:|---:|
| raf_spare_parts | titantpp_history_mlp_width16 | 34.083038 ± 0.086861 | 9.264040 ± 0.080681 | 329.857034 ± 1.086698 | 286.320619 ± 0.922707 | 3.499205 ± 0.049762 | 3.421059 ± 0.230740 |
| raf_spare_parts | titantpp_history_mlp_shared15_v1 | 33.967318 ± 0.110590 | 9.246784 ± 0.064196 | 325.510354 ± 0.761764 | 279.209741 ± 0.803666 | 3.479586 ± 0.020886 | 3.390979 ± 0.196686 |
| yellow_trip_hourly | titantpp_history_mlp_width16 | 77.825957 ± 4.160967 | 25.177656 ± 1.412918 | 376.664264 ± 27.047948 | 299.874169 ± 25.717675 | 0.994909 ± 0.297782 | 0.000000 ± 0.000000 |
| yellow_trip_hourly | titantpp_history_mlp_shared15_v1 | 77.854623 ± 1.182722 | 25.158257 ± 0.451432 | 368.560517 ± 15.317712 | 291.917974 ± 18.432080 | 1.193874 ± 0.143149 | 0.000000 ± 0.000000 |
| intermittent_frozen_5000 | titantpp_history_mlp_width16 | 1.686698 ± 0.047520 | 0.713525 ± 0.011969 | 7.923900 ± 0.468866 | 6.049223 ± 0.467381 | 0.404473 ± 0.069004 | 1.585811 ± 1.680808 |
| intermittent_frozen_5000 | titantpp_history_mlp_shared15_v1 | 1.848139 ± 0.057006 | 0.786280 ± 0.026005 | 9.421829 ± 1.072300 | 7.301882 ± 1.380460 | 0.418578 ± 0.160455 | 4.290097 ± 7.108585 |

| 데이터 | seed | ΔRMSE / 상대% | ΔMAE / 상대% |
|---|---:|---:|---:|
| raf_spare_parts | 42 | -0.063368 / -0.186% | -0.026244 / -0.281% |
| raf_spare_parts | 52 | -0.026234 / -0.077% | -0.028628 / -0.308% |
| raf_spare_parts | 62 | -0.257557 / -0.754% | +0.003105 / +0.034% |
| yellow_trip_hourly | 42 | -5.297603 / -6.421% | -1.466881 / -5.483% |
| yellow_trip_hourly | 52 | +4.697237 / +6.303% | +1.513087 / +6.299% |
| yellow_trip_hourly | 62 | +0.686364 / +0.898% | -0.104400 / -0.422% |
| intermittent_frozen_5000 | 42 | +0.153191 / +9.384% | +0.062101 / +8.698% |
| intermittent_frozen_5000 | 52 | +0.140364 / +8.156% | +0.114489 / +16.324% |
| intermittent_frozen_5000 | 62 | +0.190769 / +11.179% | +0.041675 / +5.746% |

차이는 공유형−기존 MLP16이며 양수는 악화다. 큰수량은 Taxi q>3449, RAF q>200, Intermittent q>187이다.

- Validation selects checkpoints; not independent evaluation or significance.
- Original baseline Validation contains CPU/CUDA cohorts; new candidate uses5090/cuDNN92000.
- Capalpha0 and originalalpha1 differ by data and objective; do not attribute their metric difference to backbone.
- Existing original MLP16 is a stored reference, not a new fit or reevaluation.
