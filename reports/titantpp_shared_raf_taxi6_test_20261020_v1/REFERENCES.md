# 원본 RAF·Taxi 저장 Test 참조 검증

기존 MLP16·외부6종 × 3seed × 2데이터셋, 총42개 저장 결과다. 재학습·모델 재추론·원격 호출은 없다.
RAF21개는 기존 검증 레코드와 원본 SHA를 그대로 재사용했다. Taxi21개는 저장 예측에서 여섯 지표를 재집계했다.
Checkpoint는 각 학습의 첫 유한 최소 Validation RMSE epoch로 고정했고, Test로 재선택하지 않았다.

| 데이터 | 모델 | RMSE 평균±표본SD | MAE 평균±표본SD |
|---|---|---:|---:|
| raf_spare_parts | titantpp_history_mlp_width16 | 40.247819±0.168908 | 10.148947±0.090736 |
| raf_spare_parts | rmtpp | 40.463166±0.241581 | 9.906347±0.037018 |
| raf_spare_parts | thp | 40.555091±0.295255 | 10.122950±0.230475 |
| raf_spare_parts | nhp | 42.176330±0.300481 | 9.969145±0.093177 |
| raf_spare_parts | sahp | 41.049344±0.156112 | 10.261643±0.103236 |
| raf_spare_parts | s2p2_matched_head | 39.772935±0.160914 | 10.095805±0.145493 |
| raf_spare_parts | attnhp_matched_head | 40.203624±0.204402 | 9.983934±0.028442 |
| yellow_trip_hourly | titantpp_history_mlp_width16 | 113.392687±16.501995 | 36.039467±4.009202 |
| yellow_trip_hourly | rmtpp | 246.647279±82.506426 | 65.276984±16.103485 |
| yellow_trip_hourly | thp | 202.174002±36.218246 | 58.348636±11.075891 |
| yellow_trip_hourly | nhp | 381.076245±26.234870 | 116.332384±9.712797 |
| yellow_trip_hourly | sahp | 148.249483±30.215681 | 40.900788±4.923976 |
| yellow_trip_hourly | s2p2_matched_head | 115.541513±6.402269 | 36.835852±2.381474 |
| yellow_trip_hourly | attnhp_matched_head | 131.319808±2.427940 | 38.752772±1.056452 |

전체·큰수량(q>RAF200/Taxi3449) RMSE·MAE·시간NLL, 모든 seed 원값은 REFERENCES.json에 보존했다.
표본SD는 ddof=1이다. 데이터셋 간 원시 오류는 평균내지 않았다.
Legacy ID와 현대 Window population ID는 다른 계약이다. 개별 개체·seq·수량·간격·이력·순서를 대조해 연결했다.
MLP16 seed42 CPU와 나머지 기존5080, 새Shared5090의 Runtime 차이는 남는다.
외부 비교군은 공통 수량·시간 head를 맞춘 결과다. 반복 Test 참고 탐색이며 독립 일반화·통계적 유의성의 증거가 아니다.
