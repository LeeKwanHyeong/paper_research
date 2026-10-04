# 폭8·12의 Taxi·RAF Test 완료 결과

**고정 체크포인트 12개에서 전체 Validation 12개를 재현한 뒤 Test 12개를 완료했다. 기존 폭4·16과 모든 비교군의 Test는 재실행하지 않고 검증된 집계를 재사용했다.**

- 평가 시간: 5080에서 96.514초. 새 학습·선택 epoch 변경·원시 예측 저장은 없다.
- 각 seed 안에서 event별 오차를 집계하고 seed42·52·62의 산술 평균 ± 표본 표준편차를 표시한다. Ensemble 점수가 아니다.
- 같은 Test target·truth·causal history를 사용했다. Taxi 전체 8,327개 / tail 83개, RAF 전체 5,226개 / tail 39개다.
- tail: 원래 학습 자료의 고정 경계를 엄격히 초과한 raw quantity. Taxi >3,449, RAF >200. Time NLL 단위는 Taxi 시간·RAF 월이며, 기록 양의 정수 간격에 대한 음의 로그 확률질량이다.

| 데이터 | 폭 | Test 전체 RMSE | 전체 MAE | Time NLL | tail RMSE | tail MAE |
|---|---:|---:|---:|---:|---:|---:|
| Taxi | 4 | 118.624 ± 19.570 | 37.200 ± 4.974 | 1.0292 ± 0.2727 | 646.423 ± 197.467 | 525.763 ± 208.139 |
| Taxi | 8 | 139.415 ± 16.272 | 41.684 ± 3.609 | 1.2100 ± 0.1012 | 849.404 ± 54.131 | 722.273 ± 71.405 |
| Taxi | 12 | 103.916 ± 9.105 | 34.119 ± 2.489 | 0.8882 ± 0.1603 | 468.622 ± 44.440 | 369.280 ± 46.231 |
| Taxi | 16 | 113.393 ± 16.502 | 36.039 ± 4.009 | 0.9803 ± 0.2311 | 594.683 ± 121.973 | 481.486 ± 110.864 |
| RAF | 4 | 39.795 ± 0.075 | 10.056 ± 0.088 | 4.0468 ± 0.4139 | 406.965 ± 1.253 | 358.753 ± 1.593 |
| RAF | 8 | 39.927 ± 0.412 | 10.134 ± 0.100 | 4.0671 ± 0.4584 | 408.309 ± 7.281 | 361.294 ± 9.960 |
| RAF | 12 | 40.011 ± 0.245 | 10.102 ± 0.086 | 4.0972 ± 0.5817 | 409.415 ± 4.986 | 362.988 ± 5.974 |
| RAF | 16 | 40.248 ± 0.169 | 10.149 ± 0.091 | 3.8865 ± 0.2132 | 414.097 ± 2.615 | 369.903 ± 2.547 |

**Taxi 관측 결과**
- 폭12는 폭4 대비 평균 전체 RMSE 12.40%, MAE 8.28%, tail RMSE 27.51% 감소했다. 평균 Time NLL도 1.0292→0.8882로 낮았다.
- 전체 RMSE의 폭12−폭4 차이는 seed42 −33.459, seed52 +8.399, seed62 −19.065다. 세 seed 모두 같은 방향이라는 주장은 지원하지 않는다.
- 폭8의 평균 전체 RMSE는 폭4보다 17.53% 높았고 tail과 시간 점수도 악화했다. 폭 증가에 따른 단조 개선은 관측되지 않았다.
- 기존 모든 가용 분기 MLP의 평균 Test RMSE 99.304는 폭12의 103.916보다 낮다. 폭12의 장점을 전체 비교군 단독 우월성으로 해석하지 않는다.

**RAF 관측 결과**
- 폭8·12의 평균 전체 RMSE는 폭4보다 각각 0.33%·0.54% 높았다. 평균 MAE·tail RMSE/MAE·Time NLL도 소폭 높았다.
- 폭12의 전체 RMSE는 세 seed 모두 폭4보다 높았다. 폭8은 seed62에서만 낮았다. 폭16은 수량 지표가 더 높았지만 평균 Time NLL은 3.8865로 네 폭 중 가장 낮았다.
- 폭 확대의 수량·시간 효과는 별개로 보고한다. 기존 분할을 이용해 대표 폭이나 선택 epoch를 다시 정하지 않는다.

**증적과 전체 비교 — 완료**
- [모든 기존 비교군과 폭4·8·12·16의 Test 전체 표](TEST_TABLES.md) / [Test tail 표](TEST_TAIL_TABLES.md)
- [Test 전용 per-seed 비교](comparison_test_per_seed.csv) / [새 폭8·12 Validation·Test 지표](new_width_metrics_per_seed.csv)
- [평균·표본 표준편차와 완료 범위 JSON](summary.json) / [집계 검증 영수증](analysis_receipt.json)
- `../taxi_raf_attempt3/validation_gate.json`: 모든 Validation 통과 시각이 Test 시작보다 앞선다. `inference_completion.json`: 24개 full population split 완료.
- 원본 checkpoint binary/tensor·selected epoch·source·데이터·loader·target/truth·평가 영수증 SHA를 검증했다. 기존 Test 참조는 162지표행·77개 aggregate JSON receipt를 검증해 재사용했고, 총 Test 비교는 186지표행이다.

**실행 오류와 원본 보존 — 완료**
- 첫 시도는 full parquet가 없는 `source/` 경로에 바인딩되어 행 로딩 전 중단됐다. 두 번째 시도는 project-root 탐색에 필요한 `frozen_source/sample_data/.keep` 배포 누락으로 import 단계에서 중단됐다. 두 시도 모두 Test와 추론을 시작하지 않았다.
- 실패 폴더·계약·기록은 보존했다. 세 번째 시도는 실제 parquet/manifest SHA, ancillary marker, 별도 staged frozen import를 확인한 뒤 실행했다. 과학 source 114개와 selected checkpoint는 그대로다.
- 초기 세 계약에 대응하는 prepare 원본은 `../historical_code/prepare_v1.py`와 해시 manifest로 보존했다. 후속 prepare는 서버 이동에 따른 데이터 경로만 정규화하며 과학 조건은 일치해야 한다.

**Intermittent 여섯 조건을 마친 뒤 같은 절차로 Test를 평가한다 — 외부 작업 대기**
- 대상: 폭8·12 × seed42·52·62. 모든 여섯 학습의 terminal success와 원본 SHA가 확인되어야 `followup.py`가 새 batch를 준비한다.
- 5090 부모 계약과 5080 이동 계약의 과학 조건을 일치시키고, 운영 host와 경로는 별도 provenance로 남긴다. 준비·실행·회수는 승인된 hourly driver가 수행하며 자동 재학습이나 재시도는 없다.

**Intermittent Test를 기존 표에 별도 집계한다 — 다음 작업 / 학습·평가 완료 이후**
- 이번 Taxi·RAF 결과를 보존하고 새 3dataset 집계 폴더를 만든다. seed 평균과 tail·시간 trade-off를 함께 기록한다.
- 기존 Test에 대한 사후 개발 평가로 구분한다. 새로운 독립 Test 결과로 주장하지 않는다. 원고와 대표 모델 선택은 변경하지 않았다.
