# 원본 세 데이터셋 공유형 MLP의 Test 비교

각 데이터셋의3seed 평균이다. 서로 다른 데이터셋의 raw 오류를 평균내지 않는다. RAF·Taxi는 완료된 학습의 Validation 선택 checkpoint이며 Intermittent는14:07:56 KST 캡처한 best5/5/11 고정본이다.

| 데이터 | 공유형 RMSE | 외부 최우수 RMSE | 차이(%) | 공유형 MAE | 외부 최우수 MAE | 차이(%) |
|---|---:|---|---:|---:|---|---:|
| RAF | 39.775389 | s2p2_matched_head 39.772935 | +0.0062 | 10.150475 | rmtpp 9.906347 | +2.4644 |
| Taxi | 107.084863 | s2p2_matched_head 115.541513 | -7.3191 | 34.323697 | s2p2_matched_head 36.835852 | -6.8199 |
| Intermittent (중간 고정본) | 2.966841 | s2p2_matched_head 3.215355 | -7.7290 | 1.197329 | thp 1.312281 | -8.7597 |

- Taxi와 현재 Intermittent 고정본은 평균 RMSE·MAE 두 지표가 외부 지표별 최우수보다 낮다. RAF는 RMSE가 근접하고 MAE는 외부 최우수보다 높다.
- MLP16은 내부 구조 대조이며 외부 성능 판정과 구분한다. 이력 공유와 용량 감소 효과는 이 비교만으로 분리되지 않는다.
- 큰수량·TimeNLL·모든 seed별 차이와 악화는 각 데이터셋의 원본 결과표에 보존했다. 수량 평균의 우세를 시간 성능이나 모든 seed 우세로 확대하지 않는다.
- 신규 모델은5090에서 학습하고5080에서 고정본을 평가했다. 외부6종은 공통head구현이며 일부 과거 결과의 Runtime이 다르다. 반복 Test를 참고한 탐색적 후속 결과이며 독립 검증·통계적 유의성을 뜻하지 않는다.

출처:

- `reports/titantpp_shared_raf_taxi6_test_20261020_v1/THREE_SEED_TEST.json`
- `reports/titantpp_shared_inter3_interim_test_20261020_v1/THREE_SEED_TEST.json`
