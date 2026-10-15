# RAF 현재 ValueDifference−기존 CNN 행별 오류 분석

회수한 현재 후보와 기존 CNN의 저장된 Train·Validation 예측을 같은 대상별로 비교했다. 새 추론·학습·원격 호출·Test 열람은 없다.
양수 Δ는 현재 후보의 악화, 음수는 개선이다. 평균과 표본 SD는 세 seed의 요약이며 독립 대상을 세 배로 세지 않았다.

고정 Train 변화 폭 경계: Q25=0.405465108108, Q75=1.2527629685. 기존 진단의 15개 조건과 공통 개체 가중을 그대로 적용했다.

| split | seed | CNN RMSE | 후보 RMSE | ΔRMSE | CNN MAE | 후보 MAE | ΔMAE | ΔSSE | ΔAE | Δ편향합 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 42 | +45.908764 | +45.832364 | -0.076401 | +10.652033 | +10.646399 | -0.005634 | -180686.936595 | -145.226920 | +1900.975602 |
| validation | 42 | +34.420453 | +34.410016 | -0.010437 | +9.231110 | +9.266090 | +0.034980 | -4805.805394 | +234.017990 | +997.363261 |
| train | 52 | +45.914484 | +45.907763 | -0.006721 | +10.567263 | +10.584884 | +0.017621 | -15908.266466 | +454.251892 | +1597.719397 |
| validation | 52 | +34.364303 | +34.403414 | +0.039111 | +9.186152 | +9.213424 | +0.027272 | +17993.330786 | +182.450077 | +130.501862 |
| train | 62 | +46.033589 | +45.511447 | -0.522142 | +10.640680 | +10.604026 | -0.036653 | -1232224.014648 | -944.889744 | +10934.169762 |
| validation | 62 | +34.509470 | +34.523989 | +0.014519 | +9.237552 | +9.272347 | +0.034796 | +6705.454606 | +232.783060 | +2194.858034 |

## 세 seed 평균과 표본 SD

| split | 가중 | CNN RMSE | 후보 RMSE | ΔRMSE | CNN MAE | 후보 MAE | ΔMAE |
|---|---|---:|---:|---:|---:|---:|---:|
| train | row | 45.952279 ± 0.070475 | 45.750525 ± 0.210451 | -0.201754 ± 0.279643 | 10.619992 ± 0.046016 | 10.611770 ± 0.031480 | -0.008222 ± 0.027230 |
| train | common_entity_equal | 46.078507 ± 0.057492 | 45.897560 ± 0.198998 | -0.180947 ± 0.249219 | 10.237136 ± 0.041259 | 10.236481 ± 0.027886 | -0.000655 ± 0.016446 |
| validation | row | 34.431408 ± 0.073201 | 34.445806 ± 0.067789 | 0.014398 ± 0.024774 | 9.218271 ± 0.028002 | 9.250621 ± 0.032365 | 0.032349 ± 0.004398 |
| validation | common_entity_equal | 34.678683 ± 0.066536 | 34.717211 ± 0.107743 | 0.038528 ± 0.047703 | 9.459073 ± 0.031530 | 9.507647 ± 0.051356 | 0.048573 ± 0.020244 |

## 사전 고정 조건별 기여

| 조건 | split | 가중 | 평균 ΔRMSE | 평균 ΔMAE | 평균 ΔSSE | 평균 ΔAE | 평균 Δ편향합 |
|---|---|---|---:|---:|---:|---:|---:|
| whole / None / None | train | row | -0.201754 | -0.008222 | -476273.072570 | -211.954924 | +4810.954920 |
| whole / None / None | train | common_entity_equal | -0.180947 | -0.000655 | -68485.112271 | -2.701003 | +775.878433 |
| whole / None / None | validation | row | +0.014398 | +0.032349 | +6630.993333 | +216.417042 | +1107.574385 |
| whole / None / None | validation | common_entity_equal | +0.038528 | +0.048573 | +11038.055226 | +200.170930 | +812.896047 |
| pattern / insufficient_history / None | train | row | -0.214717 | -0.021137 | -192550.879581 | -208.344072 | +1989.654251 |
| pattern / insufficient_history / None | train | common_entity_equal | +0.129990 | +0.055828 | +454.228400 | +7.592638 | +8.698726 |
| pattern / insufficient_history / None | validation | row | -0.085269 | -0.063642 | -325.597204 | -8.655267 | -2.944501 |
| pattern / insufficient_history / None | validation | common_entity_equal | -0.085269 | -0.063642 | -325.597204 | -8.655267 | -2.944501 |
| pattern / stagnation / None | train | row | -0.098637 | -0.012726 | -51294.773445 | -95.035579 | +1026.990923 |
| pattern / stagnation / None | train | common_entity_equal | -0.054055 | -0.039299 | -2807.031473 | -63.860261 | +89.452757 |
| pattern / stagnation / None | validation | row | +0.018557 | +0.017426 | +2195.835496 | +54.683421 | +310.689368 |
| pattern / stagnation / None | validation | common_entity_equal | -0.010298 | +0.013018 | -672.947108 | +21.153718 | +160.880861 |
| pattern / same_sign / None | train | row | -0.212335 | +0.035866 | -55147.657099 | +81.201285 | +602.363449 |
| pattern / same_sign / None | train | common_entity_equal | +0.077297 | +0.119665 | +2951.756164 | +45.353105 | +79.611115 |
| pattern / same_sign / None | validation | row | +0.246726 | +0.106617 | +16631.281223 | +94.569099 | +226.855880 |
| pattern / same_sign / None | validation | common_entity_equal | +0.149083 | +0.049388 | +3953.679432 | +18.718083 | +66.774520 |
| pattern / reversal / None | train | row | -0.272197 | +0.001652 | -177279.762445 | +10.223443 | +1191.946297 |
| pattern / reversal / None | train | common_entity_equal | -0.210794 | +0.024964 | -28395.552609 | +34.000981 | +241.823502 |
| pattern / reversal / None | validation | row | -0.050197 | +0.029980 | -11870.526182 | +75.819789 | +572.973639 |
| pattern / reversal / None | validation | common_entity_equal | -0.044807 | +0.043191 | -6655.742325 | +58.826723 | +241.215717 |
| pattern_x_magnitude / insufficient_history / unavailable | train | row | -0.214717 | -0.021137 | -192550.879581 | -208.344072 | +1989.654251 |
| pattern_x_magnitude / insufficient_history / unavailable | train | common_entity_equal | +0.129990 | +0.055828 | +454.228400 | +7.592638 | +8.698726 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | row | -0.085269 | -0.063642 | -325.597204 | -8.655267 | -2.944501 |
| pattern_x_magnitude / insufficient_history / unavailable | validation | common_entity_equal | -0.085269 | -0.063642 | -325.597204 | -8.655267 | -2.944501 |
| pattern_x_magnitude / stagnation / low | train | row | -0.271715 | -0.054534 | -65311.039690 | -276.162026 | +294.565135 |
| pattern_x_magnitude / stagnation / low | train | common_entity_equal | -0.038234 | -0.046403 | -984.509377 | -49.186776 | -10.018287 |
| pattern_x_magnitude / stagnation / low | validation | row | -0.102457 | +0.010053 | -6794.838532 | +21.685373 | +96.608868 |
| pattern_x_magnitude / stagnation / low | validation | common_entity_equal | -0.054477 | -0.011004 | -1380.447901 | -11.663931 | +23.475383 |
| pattern_x_magnitude / stagnation / middle | train | row | -0.138481 | +0.065574 | -25836.737179 | +112.197216 | +463.147568 |
| pattern_x_magnitude / stagnation / middle | train | common_entity_equal | -0.122019 | -0.058814 | -1037.845515 | -14.468357 | +40.840233 |
| pattern_x_magnitude / stagnation / middle | validation | row | +0.335132 | +0.027279 | +8935.041490 | +19.204377 | +175.563652 |
| pattern_x_magnitude / stagnation / middle | validation | common_entity_equal | +0.914026 | +0.114928 | +8858.699504 | +28.272325 | +59.370338 |
| pattern_x_magnitude / stagnation / high | train | row | +0.711592 | +0.099465 | +39853.003423 | +68.929231 | +269.278219 |
| pattern_x_magnitude / stagnation / high | train | common_entity_equal | +0.373634 | +0.087724 | +1004.415422 | +5.351156 | -2.336617 |
| pattern_x_magnitude / stagnation / high | validation | row | +0.002475 | +0.049797 | +55.632537 | +13.793672 | +38.516848 |
| pattern_x_magnitude / stagnation / high | validation | common_entity_equal | +0.678738 | +0.269200 | +2157.603171 | +16.421176 | +11.942298 |
| pattern_x_magnitude / same_sign / low | train | row | -0.483399 | +0.088671 | -7516.086234 | +20.394406 | +71.959458 |
| pattern_x_magnitude / same_sign / low | train | common_entity_equal | +0.043080 | -0.015341 | +12.763418 | -0.260790 | -1.724145 |
| pattern_x_magnitude / same_sign / low | validation | row | +0.086588 | +0.061860 | +151.592172 | +5.876740 | +3.084392 |
| pattern_x_magnitude / same_sign / low | validation | common_entity_equal | -0.020676 | -0.047990 | -7.187115 | -0.815825 | -1.535365 |
| pattern_x_magnitude / same_sign / middle | train | row | -0.210365 | +0.009478 | -24359.220467 | +12.795956 | +326.393017 |
| pattern_x_magnitude / same_sign / middle | train | common_entity_equal | -0.699027 | -0.025029 | -6171.756060 | -4.330013 | +37.910264 |
| pattern_x_magnitude / same_sign / middle | validation | row | +0.698795 | +0.169912 | +21583.684586 | +93.961130 | +156.392496 |
| pattern_x_magnitude / same_sign / middle | validation | common_entity_equal | +0.259336 | +0.076518 | +2471.711899 | +13.237688 | +25.467559 |
| pattern_x_magnitude / same_sign / high | train | row | -0.205387 | +0.070191 | -23272.350397 | +48.010922 | +204.010974 |
| pattern_x_magnitude / same_sign / high | train | common_entity_equal | +0.060466 | +0.566029 | +1068.268181 | +39.056023 | +45.009629 |
| pattern_x_magnitude / same_sign / high | validation | row | -0.178672 | -0.022045 | -5103.995535 | -5.268771 | +67.378991 |
| pattern_x_magnitude / same_sign / high | validation | common_entity_equal | -0.335734 | -0.046297 | -3199.318323 | -3.194501 | +29.141411 |
| pattern_x_magnitude / reversal / low | train | row | -0.521402 | -0.024776 | -18429.789294 | -26.188131 | +44.645782 |
| pattern_x_magnitude / reversal / low | train | common_entity_equal | +0.155217 | +0.006226 | +292.001887 | +0.753326 | -3.877500 |
| pattern_x_magnitude / reversal / low | validation | row | +0.156966 | +0.046679 | +2234.671561 | +21.005550 | +29.542739 |
| pattern_x_magnitude / reversal / low | validation | common_entity_equal | +0.023148 | +0.002471 | +13.108455 | +0.299011 | +0.532569 |
| pattern_x_magnitude / reversal / middle | train | row | -0.506158 | -0.056117 | -84498.390829 | -145.736685 | +439.587400 |
| pattern_x_magnitude / reversal / middle | train | common_entity_equal | -0.126775 | +0.012465 | -1588.360517 | +5.372481 | +13.845379 |
| pattern_x_magnitude / reversal / middle | validation | row | -0.102943 | -0.023452 | -9439.103675 | -24.976872 | +247.123548 |
| pattern_x_magnitude / reversal / middle | validation | common_entity_equal | +0.306309 | +0.069253 | +5175.460723 | +29.847999 | +14.561369 |
| pattern_x_magnitude / reversal / high | train | row | -0.196219 | +0.071825 | -74351.582323 | +182.148258 | +707.713114 |
| pattern_x_magnitude / reversal / high | train | common_entity_equal | -0.099096 | +0.078401 | -7215.088523 | +37.710712 | +136.373217 |
| pattern_x_magnitude / reversal / high | validation | row | -0.039364 | +0.078689 | -4666.094069 | +79.791111 | +296.307352 |
| pattern_x_magnitude / reversal / high | validation | common_entity_equal | -0.141631 | +0.087209 | -9867.156471 | +41.947682 | +171.648872 |

모든 seed별 수량·시간 지표, 공통 개체 조건에서 남은 행/제외 행과 가산 합계 검증은 RESULT.json에 보존했다.
기존 CNN 행에는 target_id·날짜·시간간격이 없다. 같은 순수 입력 SHA, split·개체·순번, 정답 수량·관측 이력을 결속했다. 후보의 별도 target/truth hash는 자체 재계산했으며 서로 다른 hash domain을 같은 것으로 취급하지 않았다.
기존 CNN은 5080/cuDNN91900, 현재 후보는 5090/cuDNN92000이다. 같은 대상의 오차 차이는 확인할 수 있지만 Runtime·최적화·V 필터 자유도 192→128(전체 576→512)·경계/매개화 차이의 원인을 분리하지는 못한다.
15개 조건의 공통 개체 집합은 조건마다 달라 가중 지표를 조건 간 더하지 않는다. 행 가중 ΔAE·ΔSSE·부호 있는 오차 합은 분할별 합계가 보존되는지 검증했다.
기존 Test 결과를 참고한 탐색적 후속 진단이다. Test 개선·독립 일반화·통계 유의성·원인 확정을 주장하지 않는다.
