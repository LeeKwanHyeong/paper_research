# RAF 고정 KeyOnly 모델의 과거 K 필터 켬/끔 비교

같은 선택 checkpoint에서 현재 K tap은 유지하고 과거 두 tap만 끈 상태(off)를 원래 상태(on)와 비교했다. 양수 off−on은 과거 두 tap을 끄면 악화함을 뜻한다.
off는 기존 CNN이나 현재 tap만으로 새로 학습한 모델이 아니다. 현재 tap의 선형 채널 배율은 공통 K 투영에 흡수할 수 있어 새로운 CNN 기여로 해석하지 않는다. 기존 CNN·QueryOnly 추론, 새 학습, Test 평가는 하지 않았다.

| 구간 | seed | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 42 | 45.916470 | 45.894156 | -0.022313 | 10.623514 | 10.631272 | 0.007758 |
| validation | 42 | 34.355589 | 34.349242 | -0.006346 | 9.211638 | 9.219955 | 0.008317 |
| train | 52 | 45.692657 | 45.691877 | -0.000780 | 10.619962 | 10.622373 | 0.002411 |
| validation | 52 | 34.235820 | 34.210915 | -0.024905 | 9.270767 | 9.263162 | -0.007606 |
| train | 62 | 45.976797 | 45.928628 | -0.048169 | 10.577064 | 10.592578 | 0.015514 |
| validation | 62 | 34.346460 | 34.260171 | -0.086290 | 9.280469 | 9.320774 | 0.040305 |

## 세 seed 평균 ± 표본 SD

| 구간 | 가중 | on RMSE | off RMSE | ΔRMSE | on MAE | off MAE | ΔMAE |
|---|---|---:|---:|---:|---:|---:|---:|
| train | row | 45.861975 ± 0.149704 | 45.838220 ± 0.127904 | -0.023754 ± 0.023728 | 10.606847 ± 0.025853 | 10.615408 ± 0.020266 | 0.008561 ± 0.006588 |
| train | common_entity_equal | 46.004479 ± 0.127198 | 45.986640 ± 0.108852 | -0.017839 ± 0.018626 | 10.229347 ± 0.028783 | 10.236359 ± 0.024782 | 0.007013 ± 0.004910 |
| validation | row | 34.312623 ± 0.066669 | 34.273443 ± 0.070112 | -0.039180 ± 0.041840 | 9.254291 ± 0.037256 | 9.267964 ± 0.050581 | 0.013672 ± 0.024400 |
| validation | common_entity_equal | 34.627564 ± 0.042100 | 34.589324 ± 0.005896 | -0.038240 ± 0.036850 | 9.499713 ± 0.028335 | 9.512086 ± 0.037606 | 0.012373 ± 0.024334 |

## 전체·큰 수량·시간 지표

| 구간 | 대상 | 지표 | on | off | off−on |
|---|---|---|---:|---:|---:|
| train | whole | qty_rmse | 45.861975 ± 0.149704 | 45.838220 ± 0.127904 | -0.023754 ± 0.023728 |
| train | whole | qty_mae | 10.606847 ± 0.025853 | 10.615408 ± 0.020266 | 0.008561 ± 0.006588 |
| train | whole | time_nll | 3.051487 ± 0.005823 | 3.052045 ± 0.005771 | 0.000558 ± 0.000205 |
| train | tail | qty_rmse | 462.371436 ± 3.325266 | 462.107922 ± 3.017205 | -0.263515 ± 0.426240 |
| train | tail | qty_mae | 353.989516 ± 4.844426 | 353.663493 ± 4.346967 | -0.326023 ± 0.708508 |
| train | tail | time_nll | 2.914509 ± 0.002953 | 2.914850 ± 0.003021 | 0.000341 ± 0.000102 |
| validation | whole | qty_rmse | 34.312623 ± 0.066669 | 34.273443 ± 0.070112 | -0.039180 ± 0.041840 |
| validation | whole | qty_mae | 9.254291 ± 0.037256 | 9.267964 ± 0.050581 | 0.013672 ± 0.024400 |
| validation | whole | time_nll | 3.552156 ± 0.098102 | 3.580227 ± 0.102972 | 0.028071 ± 0.015712 |
| validation | tail | qty_rmse | 327.486483 ± 3.602741 | 327.052335 ± 3.359827 | -0.434148 ± 0.938981 |
| validation | tail | qty_mae | 280.599659 ± 3.799705 | 280.134597 ± 3.389643 | -0.465062 ± 1.126524 |
| validation | tail | time_nll | 3.411615 ± 0.124605 | 3.419743 ± 0.130892 | 0.008128 ± 0.006398 |

## 고정 이력 조건

| 조건 | 구간 | 가중 | 평균 ΔRMSE | 평균 ΔMAE | 평균 ΔSE합 | 평균 ΔAE합 | 평균 Δ부호오차합 |
|---|---|---|---:|---:|---:|---:|---:|
| whole / None / None | train | row | -0.023754 | +0.008561 | -56257.374176 | +220.693135 | +1548.955800 |
| whole / None / None | train | common_entity_equal | -0.017839 | +0.007013 | -6774.425126 | +28.898623 | +211.234900 |
| whole / None / None | validation | row | -0.039180 | +0.013672 | -17975.446104 | +91.466238 | +454.403990 |
| whole / None / None | validation | common_entity_equal | -0.038240 | +0.012373 | -10912.395165 | +50.989795 | +287.574597 |
| pattern / insufficient_history / None | train | row | -0.008119 | +0.002233 | -7290.757461 | +22.006568 | +217.191278 |
| pattern / insufficient_history / None | train | common_entity_equal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pattern / insufficient_history / None | validation | row | +0.003570 | +0.022341 | +13.507200 | +3.038346 | +6.832009 |
| pattern / insufficient_history / None | validation | common_entity_equal | +0.003570 | +0.022341 | +13.507200 | +3.038346 | +6.832009 |
| pattern / stagnation / None | train | row | -0.016637 | +0.007488 | -8675.493773 | +55.916881 | +316.803442 |
| pattern / stagnation / None | train | common_entity_equal | -0.023868 | +0.003622 | -1245.134985 | +5.886529 | +55.129918 |
| pattern / stagnation / None | validation | row | -0.020546 | +0.007543 | -2437.554272 | +23.670320 | +91.262840 |
| pattern / stagnation / None | validation | common_entity_equal | -0.025936 | +0.007690 | -1708.475957 | +12.496066 | +42.394191 |
| pattern / same_sign / None | train | row | -0.061798 | +0.013688 | -16105.407670 | +30.990352 | +291.534841 |
| pattern / same_sign / None | train | common_entity_equal | -0.090344 | -0.000945 | -3448.549758 | -0.358008 | +57.293669 |
| pattern / same_sign / None | validation | row | -0.049153 | +0.022686 | -3304.618939 | +20.122213 | +98.117626 |
| pattern / same_sign / None | validation | common_entity_equal | -0.110501 | +0.016688 | -2930.683279 | +6.324844 | +52.441579 |
| pattern / reversal / None | train | row | -0.036947 | +0.018058 | -24185.715271 | +111.779333 | +723.426239 |
| pattern / reversal / None | train | common_entity_equal | -0.031428 | +0.012643 | -4255.962773 | +17.219411 | +163.188654 |
| pattern / reversal / None | validation | row | -0.052186 | +0.017649 | -12246.780092 | +44.635360 | +258.191514 |
| pattern / reversal / None | validation | common_entity_equal | -0.059981 | +0.016053 | -8851.767669 | +21.863790 | +161.612771 |
| pattern_x_magnitude / insufficient_history / unavailable | train | row | -0.008119 | +0.002233 | -7290.757461 | +22.006568 | +217.191278 |
| pattern_x_magnitude / insufficient_history / unavailable | train | common_entity_equal | +0.000000 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | row | +0.003570 | +0.022341 | +13.507200 | +3.038346 | +6.832009 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | common_entity_equal | +0.003570 | +0.022341 | +13.507200 | +3.038346 | +6.832009 |
| pattern_x_magnitude / stagnation / low | train | row | -0.015384 | +0.002270 | -3741.879414 | +11.494790 | +111.744858 |
| pattern_x_magnitude / stagnation / low | train | common_entity_equal | -0.007465 | +0.002093 | -193.503534 | +2.218242 | +16.335104 |
| pattern_x_magnitude / stagnation / low | validation | row | -0.002327 | +0.004979 | -154.342379 | +10.739724 | +32.704676 |
| pattern_x_magnitude / stagnation / low | validation | common_entity_equal | +0.005402 | +0.003727 | +137.527653 | +3.950888 | +11.248129 |
| pattern_x_magnitude / stagnation / middle | train | row | -0.028730 | +0.010262 | -5357.833281 | +17.558584 | +138.510133 |
| pattern_x_magnitude / stagnation / middle | train | common_entity_equal | -0.022434 | +0.007860 | -191.556827 | +1.933549 | +21.387061 |
| pattern_x_magnitude / stagnation / middle | validation | row | -0.092882 | +0.017668 | -2474.093794 | +12.438340 | +37.614583 |
| pattern_x_magnitude / stagnation / middle | validation | common_entity_equal | -0.175300 | +0.033077 | -1675.093386 | +8.136935 | +12.423655 |
| pattern_x_magnitude / stagnation / high | train | row | +0.007551 | +0.038764 | +424.218922 | +26.863507 | +66.548451 |
| pattern_x_magnitude / stagnation / high | train | common_entity_equal | +0.089029 | +0.072747 | +234.015035 | +4.437576 | +3.350448 |
| pattern_x_magnitude / stagnation / high | validation | row | +0.009476 | +0.001777 | +190.881901 | +0.492256 | +20.943581 |
| pattern_x_magnitude / stagnation / high | validation | common_entity_equal | +0.070829 | +0.038607 | +225.136828 | +2.355036 | +4.184139 |
| pattern_x_magnitude / same_sign / low | train | row | -0.048943 | -0.003763 | -754.578365 | -0.865584 | +14.301179 |
| pattern_x_magnitude / same_sign / low | train | common_entity_equal | +0.068344 | +0.018753 | +16.446973 | +0.318807 | +0.794790 |
| pattern_x_magnitude / same_sign / low | validation | row | -0.025026 | +0.022085 | -42.751606 | +2.098061 | +6.600520 |
| pattern_x_magnitude / same_sign / low | validation | common_entity_equal | +0.014020 | +0.024119 | +4.849822 | +0.410026 | +0.430446 |
| pattern_x_magnitude / same_sign / middle | train | row | -0.037171 | +0.012920 | -4333.480314 | +17.441822 | +159.244841 |
| pattern_x_magnitude / same_sign / middle | train | common_entity_equal | -0.032914 | +0.001580 | -303.870519 | +0.273408 | +21.234897 |
| pattern_x_magnitude / same_sign / middle | validation | row | -0.023336 | +0.024031 | -713.388105 | +13.288912 | +56.921826 |
| pattern_x_magnitude / same_sign / middle | validation | common_entity_equal | +0.007203 | +0.061319 | +67.719153 | +10.608157 | +22.193559 |
| pattern_x_magnitude / same_sign / high | train | row | -0.096760 | +0.021073 | -11017.348990 | +14.414115 | +117.988820 |
| pattern_x_magnitude / same_sign / high | train | common_entity_equal | -0.212462 | -0.009337 | -3704.947348 | -0.644268 | +15.077680 |
| pattern_x_magnitude / same_sign / high | validation | row | -0.088911 | +0.019813 | -2548.479228 | +4.735240 | +34.595279 |
| pattern_x_magnitude / same_sign / high | validation | common_entity_equal | -0.202481 | -0.027685 | -1971.748235 | -1.910270 | +13.394419 |
| pattern_x_magnitude / reversal / low | train | row | -0.005320 | +0.005636 | -205.712074 | +5.956769 | +34.504299 |
| pattern_x_magnitude / reversal / low | train | common_entity_equal | +0.057510 | +0.007709 | +108.884414 | +0.932737 | +1.930457 |
| pattern_x_magnitude / reversal / low | validation | row | -0.044500 | +0.001157 | -636.914457 | +0.520847 | +9.860775 |
| pattern_x_magnitude / reversal / low | validation | common_entity_equal | +0.006931 | +0.001498 | +3.545508 | +0.181235 | +0.413397 |
| pattern_x_magnitude / reversal / middle | train | row | -0.026173 | +0.012988 | -4403.179718 | +33.729190 | +230.793493 |
| pattern_x_magnitude / reversal / middle | train | common_entity_equal | -0.011346 | +0.014923 | -142.800311 | +6.431764 | +39.769542 |
| pattern_x_magnitude / reversal / middle | validation | row | -0.052777 | +0.012121 | -4830.437601 | +12.908672 | +88.170528 |
| pattern_x_magnitude / reversal / middle | validation | common_entity_equal | +0.013118 | +0.034500 | +211.543163 | +14.869432 | +37.941412 |
| pattern_x_magnitude / reversal / high | train | row | -0.051487 | +0.028428 | -19576.823479 | +72.093374 | +458.128447 |
| pattern_x_magnitude / reversal / high | train | common_entity_equal | -0.037675 | +0.039007 | -2756.880151 | +18.762407 | +90.600265 |
| pattern_x_magnitude / reversal / high | validation | row | -0.058001 | +0.030775 | -6779.428034 | +31.205840 | +160.160212 |
| pattern_x_magnitude / reversal / high | validation | common_entity_equal | -0.067074 | +0.020617 | -4606.502078 | +9.916765 | +90.906193 |

15개 이력 조건과 Train Q25/Q75를 미리 고정했고, 모든 seed별 지표·악화·예외와 공통 개체 가중의 포함/제외 행을 JSON에 보존했다. 각 조건의 공통 개체 집합은 달라 가중 조건들의 합계를 전체로 해석하지 않는다.
행 가중의 ΔAE·ΔSE·부호오차 합은 각 분할 합계와 일치하는지 검증했다. RMSE 자체를 가산 분해하지 않았다.
같은 5090 Runtime·공동 학습된 가중치·입력·head·메모리를 유지한 과거 K 혼합 민감도다. 메모리 텐서는 같아도 attention 배분은 달라질 수 있다. 분기 없이 재학습했을 때의 효과, 일반화 개선 또는 오류 원인 확정으로 해석하지 않는다. 기존 Test를 참고한 탐색적 후속 진단이다.
