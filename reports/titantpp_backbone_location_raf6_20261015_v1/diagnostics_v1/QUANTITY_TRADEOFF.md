# RAF 루프2 Validation 수량 구간 오류 분해

고정된 native Validation 6,690개와 저장 CNN 결과만 사용했다. 신규 추론·CNN 재평가·Test 열람은 없다. 제곱오차합과 절대오차합을 분해했으며 평균 RMSE 차이를 가산 분해한 것이 아니다.

| 비교 | q≤200 평균 ΔSSE | q>200 평균 ΔSSE | 전체 평균 ΔSSE | 전체 평균 Δ절대오차합 |
|---|---:|---:|---:|---:|
| titantpp_cnn_value_nonlinear_v1 − titantpp_cnn_query_nonlinear_v1 | -55240.310412 | +6556.261614 | -48684.048799 | +178.031364 |
| titantpp_cnn_value_nonlinear_v1 − titantpp_cnn | +76674.164493 | -67033.939117 | +9640.225376 | +331.477537 |
| titantpp_cnn_query_nonlinear_v1 − titantpp_cnn | +131914.474905 | -73590.200730 | +58324.274175 | +153.446173 |

V−Q에서는 q≤2 구간이 seed62의 MAE 악화에 크게 기여한다. 그러나 V−기존 CNN의 seed62 MAE 악화는 2<q≤30 구간의 기여가 가장 크다. 두 대조의 해석을 혼합하지 않는다.

현재 집계로 실제 경과시간이나 긴 이력의 처리 문제가 원인이라고 확정할 수 없다. 다음 모델을 정하기 전에 저장된 Train·Validation 개별 예측과 관측 이력의 연결 가능 여부를 확인한다. 기존 source·checkpoint·공통 조건은 유지한다.

원본 SHA: `e0f628b627acef88fc5c239f387343f8241af328490b46b687af4a853f637767`
