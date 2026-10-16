# RAF 고정 KeyOnly 모델의 가용 계수 합 보존 내용 대체 비교

같은 선택 checkpoint의 원래 상태(on)와 가용 과거 K 내용을 현재 K로 대체한 상태(off)를 비교했다. 계수와 가용성은 그대로 두며 양수 off−on은 내용 대체 시 악화함을 뜻한다.
off는 기존 CNN이나 새로 학습한 모델이 아니다. 원본 w0는 같지만 유효 현재계수는 가용 계수 합으로 바뀐다. 잠재 상수 응답을 보존하며 실제 K의 크기·방향이나 attention 배분은 보존하지 않는다. 투영 흡수는 함수 수준의 관계이며 학습 효과까지 같다는 뜻은 아니다. 기존 CNN·QueryOnly 추론, 새 학습, Test 평가는 하지 않았다.

| 구간 | seed | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 42 | 45.916470 | 45.896188 | -0.020281 | 10.623514 | 10.629308 | 0.005793 |
| validation | 42 | 34.355589 | 34.338116 | -0.017473 | 9.211638 | 9.214841 | 0.003203 |
| train | 52 | 45.692657 | 45.669245 | -0.023412 | 10.619962 | 10.630982 | 0.011021 |
| validation | 52 | 34.235820 | 34.184580 | -0.051240 | 9.270767 | 9.288029 | 0.017262 |
| train | 62 | 45.976797 | 45.951302 | -0.025495 | 10.577064 | 10.591915 | 0.014851 |
| validation | 62 | 34.346460 | 34.294573 | -0.051887 | 9.280469 | 9.347191 | 0.066722 |

## 세 seed 평균 ± 표본 SD

| 구간 | 가중 | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---|---:|---:|---:|---:|---:|---:|
| train | row | 45.861975 ± 0.149704 | 45.838912 ± 0.149498 | -0.023063 ± 0.002624 | 10.606847 ± 0.025853 | 10.617402 ± 0.022088 | 0.010555 ± 0.004547 |
| train | common_entity_equal | 46.004479 ± 0.127198 | 45.988039 ± 0.124712 | -0.016440 ± 0.003401 | 10.229347 ± 0.028783 | 10.238452 ± 0.025952 | 0.009105 ± 0.003325 |
| validation | row | 34.312623 ± 0.066669 | 34.272423 ± 0.079128 | -0.040200 ± 0.019685 | 9.254291 ± 0.037256 | 9.283354 ± 0.066299 | 0.029062 ± 0.033363 |
| validation | common_entity_equal | 34.627564 ± 0.042100 | 34.594204 ± 0.030344 | -0.033360 ± 0.016057 | 9.499713 ± 0.028335 | 9.525874 ± 0.052161 | 0.026161 ± 0.030323 |

## 전체·큰 수량·시간 지표

| 구간 | 대상 | 지표 | on | off | off−on |
|---|---|---|---:|---:|---:|
| train | whole | qty_rmse | 45.861975 ± 0.149704 | 45.838912 ± 0.149498 | -0.023063 ± 0.002624 |
| train | whole | qty_mae | 10.606847 ± 0.025853 | 10.617402 ± 0.022088 | 0.010555 ± 0.004547 |
| train | whole | time_nll | 3.051487 ± 0.005823 | 3.052265 ± 0.005428 | 0.000778 ± 0.000485 |
| train | tail | qty_rmse | 462.371436 ± 3.325266 | 462.029641 ± 3.359421 | -0.341795 ± 0.034420 |
| train | tail | qty_mae | 353.989516 ± 4.844426 | 353.485103 ± 4.845079 | -0.504413 ± 0.066493 |
| train | tail | time_nll | 2.914509 ± 0.002953 | 2.915486 ± 0.003436 | 0.000977 ± 0.001025 |
| validation | whole | qty_rmse | 34.312623 ± 0.066669 | 34.272423 ± 0.079128 | -0.040200 ± 0.019685 |
| validation | whole | qty_mae | 9.254291 ± 0.037256 | 9.283354 ± 0.066299 | 0.029062 ± 0.033363 |
| validation | whole | time_nll | 3.552156 ± 0.098102 | 3.634788 ± 0.171145 | 0.082632 ± 0.073043 |
| validation | tail | qty_rmse | 327.486483 ± 3.602741 | 326.619615 ± 3.791754 | -0.866868 ± 0.371529 |
| validation | tail | qty_mae | 280.599659 ± 3.799705 | 279.539623 ± 3.978882 | -1.060036 ± 0.477517 |
| validation | tail | time_nll | 3.411615 ± 0.124605 | 3.449359 ± 0.159891 | 0.037745 ± 0.039537 |

## 고정 이력 조건

| 조건 | 구간 | 가중 | 평균 ΔRMSE | 평균 ΔMAE | 평균 ΔSE합 | 평균 ΔAE합 | 평균 Δ부호오차합 |
|---|---|---|---:|---:|---:|---:|---:|
| whole / None / None | train | row | -0.023063 | +0.010555 | -54520.185717 | +272.090402 | +1603.495912 |
| whole / None / None | train | common_entity_equal | -0.016440 | +0.009105 | -6234.005412 | +37.521361 | +225.316134 |
| whole / None / None | validation | row | -0.040200 | +0.029062 | -18437.085173 | +194.426279 | +838.815908 |
| whole / None / None | validation | common_entity_equal | -0.033360 | +0.026161 | -9518.705850 | +107.811124 | +508.572716 |
| pattern / insufficient_history / None | train | row | -0.008614 | +0.002407 | -7733.450040 | +23.724319 | +226.622508 |
| pattern / insufficient_history / None | train | common_entity_equal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pattern / insufficient_history / None | validation | row | -0.006828 | +0.022544 | -26.371223 | +3.065979 | +9.955875 |
| pattern / insufficient_history / None | validation | common_entity_equal | -0.006828 | +0.022544 | -26.371223 | +3.065979 | +9.955875 |
| pattern / stagnation / None | train | row | -0.009065 | +0.010553 | -4730.365571 | +78.808068 | +345.895466 |
| pattern / stagnation / None | train | common_entity_equal | -0.017715 | +0.008746 | -926.046464 | +14.212618 | +63.684674 |
| pattern / stagnation / None | validation | row | -0.005309 | +0.022607 | -633.499586 | +70.939806 | +196.492484 |
| pattern / stagnation / None | validation | common_entity_equal | -0.001288 | +0.025632 | -86.075024 | +41.651805 | +97.381197 |
| pattern / same_sign / None | train | row | -0.066352 | +0.018588 | -17246.204928 | +42.082315 | +300.871997 |
| pattern / same_sign / None | train | common_entity_equal | -0.120380 | -0.010006 | -4579.265448 | -3.792352 | +56.534119 |
| pattern / same_sign / None | validation | row | -0.114818 | +0.038904 | -7704.548248 | +34.508116 | +178.557329 |
| pattern / same_sign / None | validation | common_entity_equal | -0.218238 | +0.037318 | -5759.309630 | +14.143357 | +97.426992 |
| pattern / reversal / None | train | row | -0.037994 | +0.020594 | -24810.165179 | +127.475699 | +730.105940 |
| pattern / reversal / None | train | common_entity_equal | -0.021399 | +0.020265 | -2889.328785 | +27.600499 | +158.459470 |
| pattern / reversal / None | validation | row | -0.042949 | +0.033971 | -10072.666116 | +85.912378 | +453.810220 |
| pattern / reversal / None | validation | common_entity_equal | -0.050539 | +0.035697 | -7443.105658 | +48.619787 | +266.705564 |
| pattern_x_magnitude / insufficient_history / unavailable | train | row | -0.008614 | +0.002407 | -7733.450040 | +23.724319 | +226.622508 |
| pattern_x_magnitude / insufficient_history / unavailable | train | common_entity_equal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | row | -0.006828 | +0.022544 | -26.371223 | +3.065979 | +9.955875 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | common_entity_equal | -0.006828 | +0.022544 | -26.371223 | +3.065979 | +9.955875 |
| pattern_x_magnitude / stagnation / low | train | row | -0.013764 | +0.005936 | -3328.072201 | +30.061435 | +125.850990 |
| pattern_x_magnitude / stagnation / low | train | common_entity_equal | -0.000839 | +0.008747 | -22.109595 | +9.271382 | +23.585870 |
| pattern_x_magnitude / stagnation / low | validation | row | -0.003779 | +0.016987 | -251.145338 | +36.641825 | +79.680753 |
| pattern_x_magnitude / stagnation / low | validation | common_entity_equal | -0.006956 | +0.012100 | -176.559440 | +12.826309 | +31.096206 |
| pattern_x_magnitude / stagnation / middle | train | row | -0.010845 | +0.013494 | -2023.223885 | +23.088227 | +125.829507 |
| pattern_x_magnitude / stagnation / middle | train | common_entity_equal | +0.012491 | +0.013457 | +103.888402 | +3.310354 | +15.812439 |
| pattern_x_magnitude / stagnation / middle | validation | row | +0.052800 | +0.040213 | +1395.211394 | +28.309666 | +72.484105 |
| pattern_x_magnitude / stagnation / middle | validation | common_entity_equal | +0.145254 | +0.074764 | +1392.998526 | +18.392029 | +29.045461 |
| pattern_x_magnitude / stagnation / high | train | row | +0.011140 | +0.037025 | +620.930515 | +25.658406 | +94.214969 |
| pattern_x_magnitude / stagnation / high | train | common_entity_equal | +0.008230 | +0.037412 | +21.304791 | +2.282143 | +2.947679 |
| pattern_x_magnitude / stagnation / high | validation | row | -0.088898 | +0.021618 | -1777.565642 | +5.988315 | +44.327626 |
| pattern_x_magnitude / stagnation / high | validation | common_entity_equal | +0.021656 | +0.095097 | +69.673363 | +5.800888 | +10.493657 |
| pattern_x_magnitude / same_sign / low | train | row | -0.008706 | +0.019313 | -134.054951 | +4.441955 | +15.963195 |
| pattern_x_magnitude / same_sign / low | train | common_entity_equal | +0.033638 | +0.014076 | +7.889098 | +0.239287 | +0.374021 |
| pattern_x_magnitude / same_sign / low | validation | row | -0.002902 | +0.037247 | -4.876784 | +3.538465 | +7.692126 |
| pattern_x_magnitude / same_sign / low | validation | common_entity_equal | +0.014973 | +0.043776 | +5.180495 | +0.744198 | +1.112699 |
| pattern_x_magnitude / same_sign / middle | train | row | -0.036582 | +0.012648 | -4236.180364 | +17.074748 | +142.624258 |
| pattern_x_magnitude / same_sign / middle | train | common_entity_equal | -0.083002 | -0.013355 | -723.353507 | -2.310417 | +18.829047 |
| pattern_x_magnitude / same_sign / middle | validation | row | +0.034732 | +0.059681 | +1054.749149 | +33.003817 | +97.707447 |
| pattern_x_magnitude / same_sign / middle | validation | common_entity_equal | +0.056043 | +0.099156 | +536.113240 | +17.153957 | +37.763562 |
| pattern_x_magnitude / same_sign / high | train | row | -0.113249 | +0.030067 | -12875.969613 | +20.565612 | +142.284545 |
| pattern_x_magnitude / same_sign / high | train | common_entity_equal | -0.262290 | +0.000141 | -4575.699938 | +0.009716 | +21.012894 |
| pattern_x_magnitude / same_sign / high | validation | row | -0.308029 | -0.008511 | -8754.420613 | -2.034166 | +73.157757 |
| pattern_x_magnitude / same_sign / high | validation | common_entity_equal | -0.746280 | -0.160146 | -7061.064014 | -11.050073 | +32.847136 |
| pattern_x_magnitude / reversal / low | train | row | -0.021635 | +0.002866 | -793.088304 | +3.029039 | +33.026345 |
| pattern_x_magnitude / reversal / low | train | common_entity_equal | +0.026741 | +0.006370 | +45.514917 | +0.770822 | +2.529883 |
| pattern_x_magnitude / reversal / low | validation | row | +0.007209 | +0.011642 | +97.243410 | +5.239077 | +17.951370 |
| pattern_x_magnitude / reversal / low | validation | common_entity_equal | -0.031555 | -0.000580 | -17.434690 | -0.070190 | +2.199391 |
| pattern_x_magnitude / reversal / middle | train | row | -0.031762 | +0.012492 | -5333.186111 | +32.441714 | +188.736529 |
| pattern_x_magnitude / reversal / middle | train | common_entity_equal | -0.012449 | +0.009505 | -156.369135 | +4.096488 | +31.946595 |
| pattern_x_magnitude / reversal / middle | validation | row | +0.003194 | +0.028889 | +293.061511 | +30.766781 | +147.641577 |
| pattern_x_magnitude / reversal / middle | validation | common_entity_equal | +0.097339 | +0.055163 | +1607.767149 | +23.775329 | +57.419222 |
| pattern_x_magnitude / reversal / high | train | row | -0.049243 | +0.036280 | -18683.890763 | +92.004947 | +508.343067 |
| pattern_x_magnitude / reversal / high | train | common_entity_equal | -0.022677 | +0.057299 | -1654.061727 | +27.560908 | +99.714611 |
| pattern_x_magnitude / reversal / high | validation | row | -0.089652 | +0.049217 | -10462.971037 | +49.906520 | +288.217273 |
| pattern_x_magnitude / reversal / high | validation | common_entity_equal | -0.121369 | +0.029677 | -8327.919413 | +14.274858 | +156.021829 |

15개 이력 조건과 Train Q25/Q75를 미리 고정했고, 모든 seed별 지표·악화·예외와 공통 개체 가중의 포함/제외 행을 JSON에 보존했다. 각 조건의 공통 개체 집합은 달라 가중 조건들의 합계를 전체로 해석하지 않는다.
행 가중의 ΔAE·ΔSE·부호오차 합은 각 분할 합계와 일치하는지 검증했다. RMSE 자체를 가산 분해하지 않았다.
같은 5090 Runtime·공동 학습된 가중치·입력·head·메모리를 유지한 과거 K 내용 차이에 대한 민감도다. 이전 tap 삭제 진단과의 오류 차이를 가산적인 원인 분해로 해석하지 않는다. 메모리 텐서는 같아도 attention 배분은 달라질 수 있다. 분기 없이 재학습했을 때의 효과, 일반화 개선 또는 오류 원인 확정으로 해석하지 않는다. 기존 Test를 참고한 탐색적 후속 진단이다.
