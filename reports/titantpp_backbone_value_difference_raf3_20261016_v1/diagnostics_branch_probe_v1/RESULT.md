# RAF 고정 모델 V 보정 켬/끔 비교

같은 선택 checkpoint의 V 차분 잔차만 끈 상태(off)와 원래 상태(on)를 비교했다. 양수 off−on은 보정을 끄면 악화함을 뜻한다.
이 비교의 off는 기존 CNN이나 V 없이 새로 학습한 모델이 아니다. 기존 CNN 추론, 새 학습, Test 평가는 하지 않았다.

| 구간 | seed | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 42 | 45.832364 | 45.853808 | 0.021445 | 10.646399 | 10.652477 | 0.006077 |
| validation | 42 | 34.410016 | 34.410553 | 0.000537 | 9.266090 | 9.246075 | -0.020015 |
| train | 52 | 45.907763 | 45.920067 | 0.012304 | 10.584884 | 10.589041 | 0.004156 |
| validation | 52 | 34.403414 | 34.395934 | -0.007480 | 9.213424 | 9.207849 | -0.005575 |
| train | 62 | 45.511447 | 45.546162 | 0.034715 | 10.604026 | 10.598683 | -0.005343 |
| validation | 62 | 34.523989 | 34.530130 | 0.006141 | 9.272347 | 9.237629 | -0.034718 |

## 세 seed 평균 ± 표본 SD

| 구간 | 가중 | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---|---:|---:|---:|---:|---:|---:|
| train | row | 45.750525 ± 0.210451 | 45.773346 ± 0.199517 | 0.022821 ± 0.011269 | 10.611770 ± 0.031480 | 10.613400 ± 0.034183 | 0.001630 ± 0.006115 |
| train | common_entity_equal | 45.897560 ± 0.198998 | 45.922460 ± 0.186696 | 0.024899 ± 0.012328 | 10.236481 ± 0.027886 | 10.234454 ± 0.029415 | -0.002026 ± 0.007704 |
| validation | row | 34.445806 ± 0.067789 | 34.445539 ± 0.073622 | -0.000267 ± 0.006846 | 9.250621 ± 0.032365 | 9.230518 ± 0.020081 | -0.020103 ± 0.014572 |
| validation | common_entity_equal | 34.717211 ± 0.107743 | 34.709956 ± 0.108305 | -0.007255 ± 0.001245 | 9.507647 ± 0.051356 | 9.485068 ± 0.036358 | -0.022578 ± 0.015837 |

## 전체·큰 수량·시간 지표

| 구간 | 대상 | 지표 | on | off | off−on |
|---|---|---|---:|---:|---:|
| train | whole | qty_rmse | 45.750525 ± 0.210451 | 45.773346 ± 0.199517 | 0.022821 ± 0.011269 |
| train | whole | qty_mae | 10.611770 ± 0.031480 | 10.613400 ± 0.034183 | 0.001630 ± 0.006115 |
| train | whole | time_nll | 3.053891 ± 0.002213 | 3.054076 ± 0.002233 | 0.000185 ± 0.000291 |
| train | tail | qty_rmse | 460.431817 ± 3.978888 | 460.732941 ± 3.789015 | 0.301124 ± 0.208899 |
| train | tail | qty_mae | 351.167186 ± 5.868693 | 351.712419 ± 5.502074 | 0.545233 ± 0.390696 |
| train | tail | time_nll | 2.936466 ± 0.037616 | 2.936612 ± 0.037227 | 0.000146 ± 0.000640 |
| validation | whole | qty_rmse | 34.445806 ± 0.067789 | 34.445539 ± 0.073622 | -0.000267 ± 0.006846 |
| validation | whole | qty_mae | 9.250621 ± 0.032365 | 9.230518 ± 0.020081 | -0.020103 ± 0.014572 |
| validation | whole | time_nll | 3.471717 ± 0.065161 | 3.492234 ± 0.075992 | 0.020517 ± 0.015242 |
| validation | tail | qty_rmse | 327.392464 ± 2.523467 | 327.667184 ± 2.271116 | 0.274720 ± 0.310479 |
| validation | tail | qty_mae | 279.351157 ± 3.546118 | 279.855931 ± 3.146384 | 0.504774 ± 0.442631 |
| validation | tail | time_nll | 3.335413 ± 0.092576 | 3.349744 ± 0.096468 | 0.014330 ± 0.011364 |

## 고정 이력 조건

| 조건 | 구간 | 가중 | 평균 ΔRMSE | 평균 ΔMAE | 평균 ΔSE합 | 평균 ΔAE합 | 평균 Δ부호오차합 |
|---|---|---|---:|---:|---:|---:|---:|
| whole / None / None | train | row | +0.022821 | +0.001630 | +53766.866573 | +42.025180 | -933.599861 |
| whole / None / None | train | common_entity_equal | +0.024899 | -0.002026 | +9408.684449 | -8.349659 | -166.918674 |
| whole / None / None | validation | row | -0.000267 | -0.020103 | -119.554695 | -134.489170 | -381.010124 |
| whole / None / None | validation | common_entity_equal | -0.007255 | -0.022578 | -2075.346774 | -93.045497 | -258.564480 |
| pattern / insufficient_history / None | train | row | +0.013931 | +0.004881 | +12457.940168 | +48.108467 | -159.713032 |
| pattern / insufficient_history / None | train | common_entity_equal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pattern / insufficient_history / None | validation | row | +0.005937 | -0.007517 | +22.655828 | -1.022281 | -11.362305 |
| pattern / insufficient_history / None | validation | common_entity_equal | +0.005937 | -0.007517 | +22.655828 | -1.022281 | -11.362305 |
| pattern / stagnation / None | train | row | +0.012981 | +0.002031 | +6745.491320 | +15.170114 | -225.273510 |
| pattern / stagnation / None | train | common_entity_equal | +0.006771 | +0.000746 | +351.321983 | +1.211578 | -29.351783 |
| pattern / stagnation / None | validation | row | -0.036650 | -0.014575 | -4343.202150 | -45.735173 | -81.335049 |
| pattern / stagnation / None | validation | common_entity_equal | -0.010570 | -0.001702 | -695.017285 | -2.766337 | -28.978935 |
| pattern / same_sign / None | train | row | +0.052835 | -0.007653 | +13699.699984 | -17.327024 | -241.984463 |
| pattern / same_sign / None | train | common_entity_equal | -0.040266 | -0.041626 | -1542.116261 | -15.776091 | -31.085170 |
| pattern / same_sign / None | validation | row | +0.028351 | -0.019890 | +1904.581189 | -17.642617 | -122.073206 |
| pattern / same_sign / None | validation | common_entity_equal | +0.082451 | -0.025587 | +2193.152934 | -9.697388 | -51.976996 |
| pattern / reversal / None | train | row | +0.032097 | -0.000634 | +20863.735101 | -3.926376 | -306.628855 |
| pattern / reversal / None | train | common_entity_equal | +0.036241 | +0.005742 | +4874.782445 | +7.821225 | -83.675298 |
| pattern / reversal / None | validation | row | +0.009706 | -0.027714 | +2296.410438 | -70.089099 | -166.239563 |
| pattern / reversal / None | validation | common_entity_equal | +0.031184 | -0.015303 | +4629.998692 | -20.842839 | -79.960112 |
| pattern_x_magnitude / insufficient_history / unavailable | train | row | +0.013931 | +0.004881 | +12457.940168 | +48.108467 | -159.713032 |
| pattern_x_magnitude / insufficient_history / unavailable | train | common_entity_equal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | row | +0.005937 | -0.007517 | +22.655828 | -1.022281 | -11.362305 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | common_entity_equal | +0.005937 | -0.007517 | +22.655828 | -1.022281 | -11.362305 |
| pattern_x_magnitude / stagnation / low | train | row | +0.025499 | +0.007927 | +6089.384934 | +40.143819 | -37.462655 |
| pattern_x_magnitude / stagnation / low | train | common_entity_equal | +0.010814 | +0.008733 | +277.950477 | +9.257315 | +2.093012 |
| pattern_x_magnitude / stagnation / low | validation | row | -0.013405 | -0.001992 | -883.977172 | -4.297009 | -16.825204 |
| pattern_x_magnitude / stagnation / low | validation | common_entity_equal | +0.022095 | +0.008274 | +559.214640 | +8.770898 | +1.731276 |
| pattern_x_magnitude / stagnation / middle | train | row | -0.009568 | -0.023991 | -1782.431444 | -41.049316 | -101.581582 |
| pattern_x_magnitude / stagnation / middle | train | common_entity_equal | -0.069616 | -0.009932 | -590.500262 | -2.443173 | -1.183825 |
| pattern_x_magnitude / stagnation / middle | validation | row | -0.130840 | -0.036169 | -3510.004629 | -25.462793 | -56.234331 |
| pattern_x_magnitude / stagnation / middle | validation | common_entity_equal | -0.184008 | -0.044414 | -1832.107980 | -10.925838 | -20.986422 |
| pattern_x_magnitude / stagnation / high | train | row | +0.043346 | +0.023197 | +2438.537829 | +16.075611 | -86.229273 |
| pattern_x_magnitude / stagnation / high | train | common_entity_equal | -0.175259 | -0.065939 | -475.183308 | -4.022258 | -5.027373 |
| pattern_x_magnitude / stagnation / high | validation | row | +0.002511 | -0.057673 | +50.779650 | -15.975370 | -8.275513 |
| pattern_x_magnitude / stagnation / high | validation | common_entity_equal | -0.221034 | -0.044163 | -708.404940 | -2.693944 | +0.719116 |
| pattern_x_magnitude / same_sign / low | train | row | -0.036958 | -0.037169 | -571.170681 | -8.548824 | -10.329921 |
| pattern_x_magnitude / same_sign / low | train | common_entity_equal | -0.007948 | +0.007070 | -1.929026 | +0.120188 | +0.569816 |
| pattern_x_magnitude / same_sign / low | validation | row | -0.051197 | -0.024742 | -89.543321 | -2.350474 | -1.436444 |
| pattern_x_magnitude / same_sign / low | validation | common_entity_equal | -0.018247 | +0.029310 | -6.353149 | +0.498262 | +0.103603 |
| pattern_x_magnitude / same_sign / middle | train | row | +0.057434 | -0.010456 | +6626.638710 | -14.115102 | -110.124118 |
| pattern_x_magnitude / same_sign / middle | train | common_entity_equal | -0.000478 | -0.054001 | -4.996653 | -9.342154 | -23.813185 |
| pattern_x_magnitude / same_sign / middle | validation | row | -0.054770 | -0.042998 | -1723.028004 | -23.777729 | -65.051495 |
| pattern_x_magnitude / same_sign / middle | validation | common_entity_equal | -0.131574 | -0.091357 | -1254.479264 | -15.804767 | -23.409385 |
| pattern_x_magnitude / same_sign / high | train | row | +0.067387 | +0.007802 | +7644.231955 | +5.336903 | -121.530424 |
| pattern_x_magnitude / same_sign / high | train | common_entity_equal | -0.306458 | -0.112770 | -5414.325763 | -7.781119 | -12.903520 |
| pattern_x_magnitude / same_sign / high | validation | row | +0.130742 | +0.035505 | +3717.152514 | +8.485586 | -55.585267 |
| pattern_x_magnitude / same_sign / high | validation | common_entity_equal | +0.468153 | +0.092033 | +4435.485235 | +6.350273 | -25.176960 |
| pattern_x_magnitude / reversal / low | train | row | +0.096942 | +0.014824 | +3381.228623 | +15.669186 | +7.591761 |
| pattern_x_magnitude / reversal / low | train | common_entity_equal | +0.172505 | +0.011942 | +330.480507 | +1.444997 | +4.213325 |
| pattern_x_magnitude / reversal / low | validation | row | -0.065525 | -0.015108 | -941.343433 | -6.798393 | -7.663759 |
| pattern_x_magnitude / reversal / low | validation | common_entity_equal | -0.004945 | -0.002091 | -2.592261 | -0.252981 | +0.688405 |
| pattern_x_magnitude / reversal / middle | train | row | +0.032260 | -0.001920 | +5326.429146 | -4.986216 | -124.780549 |
| pattern_x_magnitude / reversal / middle | train | common_entity_equal | +0.015879 | -0.006684 | +198.693074 | -2.880863 | -12.713068 |
| pattern_x_magnitude / reversal / middle | validation | row | -0.005895 | -0.015547 | -539.012571 | -16.558010 | -67.967808 |
| pattern_x_magnitude / reversal / middle | validation | common_entity_equal | -0.142529 | -0.042308 | -2406.618789 | -18.234864 | -23.704052 |
| pattern_x_magnitude / reversal / high | train | row | +0.032096 | -0.005761 | +12156.077332 | -14.609346 | -189.440067 |
| pattern_x_magnitude / reversal / high | train | common_entity_equal | +0.053146 | +0.010289 | +3867.306688 | +4.949188 | -51.922449 |
| pattern_x_magnitude / reversal / high | validation | row | +0.032002 | -0.046087 | +3776.766442 | -46.732695 | -90.607996 |
| pattern_x_magnitude / reversal / high | validation | common_entity_equal | +0.055549 | -0.020889 | +3857.513849 | -10.047635 | -36.170555 |

15개 이력 조건과 Train Q25/Q75를 미리 고정했고, 모든 seed별 지표·악화·예외와 공통 개체 가중의 포함/제외 행을 JSON에 보존했다. 각 조건의 공통 개체 집합은 달라 가중 조건들의 합계를 전체로 해석하지 않는다.
행 가중의 ΔAE·ΔSE·부호오차 합은 각 분할 합계와 일치하는지 검증했다. RMSE 자체를 가산 분해하지 않았다.
같은 5090 Runtime·공동 학습된 가중치·입력·head·메모리를 유지한 분기 민감도다. 분기 없이 재학습했을 때의 효과, 일반화 개선 또는 오류 원인 확정으로 해석하지 않는다. 기존 Test를 참고한 탐색적 후속 진단이다.
