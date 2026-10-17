# Cap36 완료 결과 독립 검산

판정: PASS. 원본 수집본과 저장 예측만 사용했고 추론·학습·재평가·SSH·테스트 실행·소스 수정은 하지 않았다.

증적 1130개와 보고서4개 SHA를 재검증했다. 원본72개 receipt의216개 평균·표본SD,810개 seed별 차이,270개 차이 요약을 독립 계산했다.
저장 예측72개 파일의 총2,433,780행에서 전체·cap_hit RMSE/MAE/TimeNLL을 math.fsum으로 재계산했고432개 지표가 허용오차 안에서 일치했다.
모든 구조의 데이터셋별 대상 수·cap_hit 수가 동일하며 q>cap는0개이고 관련 지표는 N/A다.

| split | dataset | arm | RMSE 평균±표본SD | MAE 평균±표본SD |
|---|---|---|---:|---:|
| validation | intermittent_frozen_5000 | MLP4 | 0.437231 ± 0.011914 | 0.267663 ± 0.012770 |
| validation | intermittent_frozen_5000 | MLP16 | 0.449956 ± 0.027381 | 0.273093 ± 0.011208 |
| validation | intermittent_frozen_5000 | AllAvailable | 0.414230 ± 0.004588 | 0.261810 ± 0.005689 |
| validation | intermittent_frozen_5000 | CurrentOnly | 0.461327 ± 0.026300 | 0.279615 ± 0.019263 |
| validation | yellow_trip_hourly | MLP4 | 6.043985 ± 0.064436 | 3.090209 ± 0.084464 |
| validation | yellow_trip_hourly | MLP16 | 6.097848 ± 0.063789 | 3.233059 ± 0.251964 |
| validation | yellow_trip_hourly | AllAvailable | 6.065132 ± 0.134592 | 3.220171 ± 0.202191 |
| validation | yellow_trip_hourly | CurrentOnly | 6.065107 ± 0.106210 | 3.026535 ± 0.077972 |
| validation | raf_spare_parts | MLP4 | 2.606551 ± 0.011188 | 1.863357 ± 0.020546 |
| validation | raf_spare_parts | MLP16 | 2.607568 ± 0.010649 | 1.835865 ± 0.046722 |
| validation | raf_spare_parts | AllAvailable | 2.611237 ± 0.008253 | 1.853601 ± 0.037354 |
| validation | raf_spare_parts | CurrentOnly | 2.612903 ± 0.009025 | 1.885278 ± 0.015499 |
| test | intermittent_frozen_5000 | MLP4 | 1.527533 ± 0.147389 | 0.663201 ± 0.034684 |
| test | intermittent_frozen_5000 | MLP16 | 1.537937 ± 0.049181 | 0.675620 ± 0.035439 |
| test | intermittent_frozen_5000 | AllAvailable | 1.485032 ± 0.045988 | 0.633899 ± 0.019446 |
| test | intermittent_frozen_5000 | CurrentOnly | 1.454467 ± 0.156889 | 0.677866 ± 0.048885 |
| test | yellow_trip_hourly | MLP4 | 6.165356 ± 0.206079 | 3.437163 ± 0.141957 |
| test | yellow_trip_hourly | MLP16 | 6.107686 ± 0.313573 | 3.504514 ± 0.140926 |
| test | yellow_trip_hourly | AllAvailable | 6.203778 ± 0.281107 | 3.540543 ± 0.230343 |
| test | yellow_trip_hourly | CurrentOnly | 6.324196 ± 0.199449 | 3.570002 ± 0.096323 |
| test | raf_spare_parts | MLP4 | 2.702453 ± 0.023393 | 1.915335 ± 0.013455 |
| test | raf_spare_parts | MLP16 | 2.693792 ± 0.013963 | 1.887946 ± 0.036415 |
| test | raf_spare_parts | AllAvailable | 2.702397 ± 0.026733 | 1.902875 ± 0.037319 |
| test | raf_spare_parts | CurrentOnly | 2.702016 ± 0.017717 | 1.937649 ± 0.009035 |

발견한 수치·SHA 불일치: 없음.
Q3 캡 수량을 대상으로 한 결과이며 원래 수량 예측의 개선을 입증하지 않는다. 기존 CNN·raw 결과는 재학습·재평가하지 않았다. Supervisor 자신의 OS exit는 직접 관측하지 않았다.
