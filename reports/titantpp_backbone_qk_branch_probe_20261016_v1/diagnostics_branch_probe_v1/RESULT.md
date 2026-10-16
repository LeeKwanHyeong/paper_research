# RAF 고정 모델 Q/K 지역 보정 켬/끔 비교

같은 선택 checkpoint의 Q/K 지역 잔차를 함께 끈 상태(off)와 원래 상태(on)를 비교했다. 양수 off−on은 보정을 끄면 악화함을 뜻한다.
이 비교의 off는 기존 CNN이나 Q/K 지역 보정 없이 새로 학습한 모델이 아니다. 기존 CNN 추론, 새 학습, Test 평가는 하지 않았다.

| 구간 | seed | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 42 | 46.260198 | 46.506487 | 0.246289 | 10.618848 | 10.649086 | 0.030237 |
| validation | 42 | 34.458016 | 34.629938 | 0.171923 | 9.209258 | 9.215155 | 0.005897 |
| train | 52 | 45.899858 | 46.039325 | 0.139466 | 10.591315 | 10.606566 | 0.015251 |
| validation | 52 | 34.401603 | 34.336470 | -0.065134 | 9.227408 | 9.241294 | 0.013886 |
| train | 62 | 45.503850 | 45.362741 | -0.141109 | 10.607674 | 10.827851 | 0.220177 |
| validation | 62 | 34.566537 | 33.932187 | -0.634350 | 9.259029 | 9.669254 | 0.410225 |

## 세 seed 평균 ± 표본 SD

| 구간 | 가중 | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---|---:|---:|---:|---:|---:|---:|
| train | row | 45.887969 ± 0.378314 | 45.969518 ± 0.575060 | 0.081549 ± 0.200088 | 10.605946 ± 0.013848 | 10.694501 ± 0.117425 | 0.088555 ± 0.114234 |
| train | common_entity_equal | 46.032480 ± 0.345331 | 46.113289 ± 0.493771 | 0.080808 ± 0.150595 | 10.230968 ± 0.015273 | 10.308885 ± 0.112015 | 0.077917 ± 0.099369 |
| validation | row | 34.475385 ± 0.083828 | 34.299532 ± 0.350339 | -0.175854 ± 0.414383 | 9.231898 ± 0.025188 | 9.375234 ± 0.254964 | 0.143336 ± 0.231167 |
| validation | common_entity_equal | 34.792925 ± 0.100232 | 34.687006 ± 0.298887 | -0.105918 ± 0.375347 | 9.487700 ± 0.052628 | 9.626854 ± 0.281111 | 0.139154 ± 0.228755 |

## 전체·큰 수량·시간 지표

| 구간 | 대상 | 지표 | on | off | off−on |
|---|---|---|---:|---:|---:|
| train | whole | qty_rmse | 45.887969 ± 0.378314 | 45.969518 ± 0.575060 | 0.081549 ± 0.200088 |
| train | whole | qty_mae | 10.605946 ± 0.013848 | 10.694501 ± 0.117425 | 0.088555 ± 0.114234 |
| train | whole | time_nll | 3.055072 ± 0.001360 | 3.058206 ± 0.004205 | 0.003134 ± 0.004470 |
| train | tail | qty_rmse | 462.164180 ± 6.817789 | 464.197265 ± 8.824345 | 2.033085 ± 2.097185 |
| train | tail | qty_mae | 353.746781 ± 9.860132 | 356.908573 ± 12.645788 | 3.161793 ± 2.960218 |
| train | tail | time_nll | 2.936615 ± 0.037397 | 2.942425 ± 0.045270 | 0.005810 ± 0.007878 |
| validation | whole | qty_rmse | 34.475385 ± 0.083828 | 34.299532 ± 0.350339 | -0.175854 ± 0.414383 |
| validation | whole | qty_mae | 9.231898 ± 0.025188 | 9.375234 ± 0.254964 | 0.143336 ± 0.231167 |
| validation | whole | time_nll | 3.424133 ± 0.039575 | 3.437131 ± 0.071480 | 0.012998 ± 0.036109 |
| validation | tail | qty_rmse | 329.203797 ± 4.442475 | 329.924828 ± 9.613117 | 0.721031 ± 5.492392 |
| validation | tail | qty_mae | 281.590498 ± 5.575979 | 283.723815 ± 10.673003 | 2.133317 ± 5.337816 |
| validation | tail | time_nll | 3.292393 ± 0.017498 | 3.294250 ± 0.014708 | 0.001857 ± 0.002852 |

## 고정 이력 조건

| 조건 | 구간 | 가중 | 평균 ΔRMSE | 평균 ΔMAE | 평균 ΔSE합 | 평균 ΔAE합 | 평균 Δ부호오차합 |
|---|---|---|---:|---:|---:|---:|---:|
| whole / None / None | train | row | +0.081549 | +0.088555 | +196330.853038 | +2282.858437 | +5464.312040 |
| whole / None / None | train | common_entity_equal | +0.080808 | +0.077917 | +31027.809902 | +321.096339 | +768.442885 |
| whole / None / None | validation | row | -0.175854 | +0.143336 | -80394.888030 | +958.916752 | +2584.050012 |
| whole / None / None | validation | common_entity_equal | -0.105918 | +0.139154 | -30109.399487 | +573.452542 | +1572.423378 |
| pattern / insufficient_history / None | train | row | +0.021853 | +0.015721 | +19845.434949 | +154.961110 | +285.401753 |
| pattern / insufficient_history / None | train | common_entity_equal | -0.017113 | -0.011342 | -60.869323 | -1.542548 | -0.052875 |
| pattern / insufficient_history / None | validation | row | +0.048801 | +0.065025 | +186.266218 | +8.843462 | +18.986449 |
| pattern / insufficient_history / None | validation | common_entity_equal | +0.048801 | +0.065025 | +186.266218 | +8.843462 | +18.986449 |
| pattern / stagnation / None | train | row | +0.032695 | +0.079864 | +17822.656549 | +596.426696 | +1115.237827 |
| pattern / stagnation / None | train | common_entity_equal | +0.022350 | +0.092101 | +1247.936628 | +149.664489 | +335.698151 |
| pattern / stagnation / None | validation | row | +0.049777 | +0.123678 | +5961.332467 | +388.102568 | +777.793032 |
| pattern / stagnation / None | validation | common_entity_equal | +0.040525 | +0.091962 | +2760.386698 | +149.438950 | +314.378947 |
| pattern / same_sign / None | train | row | +0.142929 | +0.165635 | +38365.598600 | +374.997538 | +1042.261970 |
| pattern / same_sign / None | train | common_entity_equal | -0.405272 | -0.036575 | -15302.755831 | -13.861820 | +247.520452 |
| pattern / same_sign / None | validation | row | -0.304324 | +0.133807 | -20376.730525 | +118.687149 | +410.428843 |
| pattern / same_sign / None | validation | common_entity_equal | -0.253511 | +0.223229 | -6606.954409 | +84.603730 | +155.359301 |
| pattern / reversal / None | train | row | +0.181985 | +0.186829 | +120297.162941 | +1156.473093 | +3021.410490 |
| pattern / reversal / None | train | common_entity_equal | +0.286323 | +0.176820 | +38920.764363 | +240.829286 | +683.574511 |
| pattern / reversal / None | validation | row | -0.282462 | +0.175280 | -66165.756190 | +443.283573 | +1376.841689 |
| pattern / reversal / None | validation | common_entity_equal | -0.351428 | +0.126046 | -51700.580760 | +171.674663 | +796.638167 |
| pattern_x_magnitude / insufficient_history / unavailable | train | row | +0.021853 | +0.015721 | +19845.434949 | +154.961110 | +285.401753 |
| pattern_x_magnitude / insufficient_history / unavailable | train | common_entity_equal | -0.017113 | -0.011342 | -60.869323 | -1.542548 | -0.052875 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | row | +0.048801 | +0.065025 | +186.266218 | +8.843462 | +18.986449 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | common_entity_equal | +0.048801 | +0.065025 | +186.266218 | +8.843462 | +18.986449 |
| pattern_x_magnitude / stagnation / low | train | row | +0.194324 | +0.065521 | +47647.005870 | +331.799490 | +288.203067 |
| pattern_x_magnitude / stagnation / low | train | common_entity_equal | +0.074421 | +0.083338 | +1926.610190 | +88.337821 | +116.870641 |
| pattern_x_magnitude / stagnation / low | validation | row | +0.137615 | +0.080473 | +9252.245918 | +173.580807 | +332.032259 |
| pattern_x_magnitude / stagnation / low | validation | common_entity_equal | -0.004363 | +0.047467 | -95.661622 | +50.314599 | +107.985110 |
| pattern_x_magnitude / stagnation / middle | train | row | -0.078757 | +0.070674 | -14380.654480 | +120.923181 | +349.101933 |
| pattern_x_magnitude / stagnation / middle | train | common_entity_equal | -0.031565 | +0.096954 | -252.413027 | +23.850613 | +98.519918 |
| pattern_x_magnitude / stagnation / middle | validation | row | -0.138633 | +0.183620 | -3663.169239 | +129.268384 | +218.297738 |
| pattern_x_magnitude / stagnation / middle | validation | common_entity_equal | -0.972952 | +0.096716 | -8925.764428 | +23.792024 | +57.761642 |
| pattern_x_magnitude / stagnation / high | train | row | -0.270327 | +0.207365 | -15443.694842 | +143.704025 | +477.932827 |
| pattern_x_magnitude / stagnation / high | train | common_entity_equal | -0.522364 | +0.013596 | -1370.979848 | +0.829327 | +48.223709 |
| pattern_x_magnitude / stagnation / high | validation | row | +0.014388 | +0.307774 | +372.255788 | +85.253377 | +227.463034 |
| pattern_x_magnitude / stagnation / high | validation | common_entity_equal | -0.428978 | +0.031454 | -1334.514468 | +1.918686 | +43.217813 |
| pattern_x_magnitude / same_sign / low | train | row | +0.265120 | +0.091709 | +4319.990080 | +21.093164 | -22.284028 |
| pattern_x_magnitude / same_sign / low | train | common_entity_equal | +0.260631 | +0.140504 | +67.107760 | +2.388565 | +4.859697 |
| pattern_x_magnitude / same_sign / low | validation | row | +0.159440 | +0.168804 | +280.979129 | +16.036413 | +34.903644 |
| pattern_x_magnitude / same_sign / low | validation | common_entity_equal | -0.034066 | -0.013834 | -11.621573 | -0.235174 | +4.608739 |
| pattern_x_magnitude / same_sign / middle | train | row | +0.283792 | +0.161572 | +33450.929013 | +218.122166 | +482.925721 |
| pattern_x_magnitude / same_sign / middle | train | common_entity_equal | +0.173931 | -0.021669 | +1741.237584 | -3.748788 | +95.300787 |
| pattern_x_magnitude / same_sign / middle | validation | row | -0.397282 | +0.109555 | -12062.415645 | +60.584114 | +159.469411 |
| pattern_x_magnitude / same_sign / middle | validation | common_entity_equal | +0.360334 | +0.343351 | +3481.201990 | +59.399702 | +62.636826 |
| pattern_x_magnitude / same_sign / high | train | row | +0.000344 | +0.198512 | +594.679506 | +135.782208 | +581.620276 |
| pattern_x_magnitude / same_sign / high | train | common_entity_equal | -0.595153 | -0.145132 | -10356.998347 | -10.014079 | +78.568886 |
| pattern_x_magnitude / same_sign / high | validation | row | -0.318750 | +0.176011 | -8595.294009 | +42.066622 | +216.055787 |
| pattern_x_magnitude / same_sign / high | validation | common_entity_equal | -0.758893 | +0.027917 | -6686.314975 | +1.926263 | +78.444176 |
| pattern_x_magnitude / reversal / low | train | row | +0.370811 | +0.055178 | +13513.077152 | +58.322829 | +95.480751 |
| pattern_x_magnitude / reversal / low | train | common_entity_equal | -0.394965 | +0.027329 | -638.465412 | +3.306794 | +7.300725 |
| pattern_x_magnitude / reversal / low | validation | row | -0.028969 | +0.072901 | -404.306080 | +32.805531 | +61.088742 |
| pattern_x_magnitude / reversal / low | validation | common_entity_equal | +0.366864 | +0.099191 | +210.998763 | +12.002100 | +2.142032 |
| pattern_x_magnitude / reversal / middle | train | row | +0.433518 | +0.177120 | +73914.492422 | +459.981186 | +754.010345 |
| pattern_x_magnitude / reversal / middle | train | common_entity_equal | +0.161736 | +0.125516 | +2090.715928 | +54.097601 | +159.997473 |
| pattern_x_magnitude / reversal / middle | validation | row | -0.194776 | +0.123901 | -17644.872846 | +131.954893 | +177.082466 |
| pattern_x_magnitude / reversal / middle | validation | common_entity_equal | -0.415166 | +0.085929 | -6722.570047 | +37.035392 | +157.547532 |
| pattern_x_magnitude / reversal / high | train | row | +0.084969 | +0.251644 | +32869.593366 | +638.169078 | +2171.919394 |
| pattern_x_magnitude / reversal / high | train | common_entity_equal | +0.323372 | +0.283207 | +23659.137707 | +136.222385 | +413.619696 |
| pattern_x_magnitude / reversal / high | validation | row | -0.411531 | +0.274678 | -48116.577265 | +278.523148 | +1138.670481 |
| pattern_x_magnitude / reversal / high | validation | common_entity_equal | -0.540337 | +0.137055 | -37087.385516 | +65.923452 | +545.382754 |

15개 이력 조건과 Train Q25/Q75를 미리 고정했고, 모든 seed별 지표·악화·예외와 공통 개체 가중의 포함/제외 행을 JSON에 보존했다. 각 조건의 공통 개체 집합은 달라 가중 조건들의 합계를 전체로 해석하지 않는다.
행 가중의 ΔAE·ΔSE·부호오차 합은 각 분할 합계와 일치하는지 검증했다. RMSE 자체를 가산 분해하지 않았다.
같은 5090 Runtime·공동 학습된 가중치·입력·head·메모리를 유지한 분기 민감도다. 분기 없이 재학습했을 때의 효과, 일반화 개선 또는 오류 원인 확정으로 해석하지 않는다. 기존 Test를 참고한 탐색적 후속 진단이다.
