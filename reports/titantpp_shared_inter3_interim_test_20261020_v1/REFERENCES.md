# 원본 Intermittent 저장 Validation·Test 참조 검증

MLP16·외부6종 × 3seed × 2split, 총42개 기존 결과다. 재학습·모델 재추론·원격 호출은 없다.
Test21과 MLP16 Validation3은 저장 예측의 대상·수량·시간·이력·순서를 원본 데이터와 대조하고 여섯 지표를 재집계했다.
외부 Validation18은 기존 selected endpoint 집계다. checkpoint·state·첫 유한 최소 Validation RMSE epoch·전체 지표·모집단·tail 개수를 확인했으나 개별 예측/행 순서는 검증하지 못했다.
각 모델의 동일 Validation-selected checkpoint를 유지했으며 Test로 재선택하지 않았다.

| split | 모델 | RMSE 평균±표본SD | MAE 평균±표본SD |
|---|---|---:|---:|
| validation | titantpp_history_mlp_width16 | 1.686698±0.047520 | 0.713525±0.011969 |
| validation | rmtpp | 2.549928±0.160402 | 0.894885±0.049405 |
| validation | thp | 2.753352±0.178427 | 0.899331±0.066103 |
| validation | nhp | 12.875455±1.040504 | 4.518767±0.170590 |
| validation | sahp | 7.029832±0.767268 | 1.861945±0.089926 |
| validation | s2p2_matched_head | 2.575842±0.076563 | 0.935886±0.008569 |
| validation | attnhp_matched_head | 3.079937±0.298175 | 1.041263±0.093761 |
| test | titantpp_history_mlp_width16 | 2.935730±0.082972 | 1.222756±0.070504 |
| test | rmtpp | 4.211471±0.212624 | 1.692320±0.180771 |
| test | thp | 3.483944±0.123200 | 1.312281±0.027045 |
| test | nhp | 5.906457±0.850743 | 2.327751±0.159345 |
| test | sahp | 12.252441±1.518177 | 3.676981±0.689162 |
| test | s2p2_matched_head | 3.215355±0.274250 | 1.319690±0.026354 |
| test | attnhp_matched_head | 12.289099±2.689065 | 4.349284±0.701125 |

큰수량은 Train 고정 q>187이다. 전체·큰수량 RMSE·MAE·recorded positive-integer TimeNLL과 모든 seed 원값은 JSON에 보존했다.
표본SD는 ddof=1이다. 기존 Runtime 차이와 외부 공통 head 적응 범위가 남는다.
