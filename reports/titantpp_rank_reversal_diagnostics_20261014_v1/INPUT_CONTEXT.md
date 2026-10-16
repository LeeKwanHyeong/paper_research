# 고정 입력의 구간별 분포 맥락

## Material Passport

Origin Skill: academic-research-suite / experiment-agent validate. Verification Status: ANALYZED. 기존 입력·영수증만 읽었으며 학습·모델 추론·원격 호출·모델 통계 재추정은 없다. Train 표는 첫 시점 포함 전체 행이며 Train 예측오류가 아니다. Validation/Test 표의 전체·5개 수량bin 개수는 실제 평가 대상과 일치한다.

| 데이터셋 | split | 행 수 | 수량 평균 | p95 | p99 | 최대 | 큰 수량 비율 | 양의 저장 gap 평균 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| yellow_trip_hourly | train | 38524 | 228.97 | 1561.8 | 3448.8 | 6322 | 0.9994% | 1.7399 |
| yellow_trip_hourly | validation | 8268 | 213.46 | 1420.9 | 3424.6 | 6489 | 0.9555% | 1.7788 |
| yellow_trip_hourly | test | 8327 | 229.71 | 1576.1 | 3446.7 | 5952 | 0.9968% | 1.8462 |
| raf_spare_parts | train | 30779 | 14.504 | 60 | 200 | 2062 | 0.7635% | 8.1646 |
| raf_spare_parts | validation | 6690 | 12.765 | 50 | 200 | 1000 | 0.7474% | 10.082 |
| raf_spare_parts | test | 5226 | 14.152 | 50 | 200 | 1000 | 0.7463% | 10.084 |
| intermittent_frozen_5000 | train | 398824 | 15.065 | 46 | 187 | 477 | 0.9959% | 2.9969 |
| intermittent_frozen_5000 | validation | 86285 | 17.741 | 49 | 212 | 464 | 1.3224% | 2.7672 |
| intermittent_frozen_5000 | test | 88019 | 11.042 | 33 | 106 | 350 | 0.2761% | 3.6238 |

세 데이터셋 모두 각 entity의 Train→Validation→Test 순서 위반은 0이다. 일부 dataset은 entity별 분할이므로 global 날짜 범위가 겹친다. 이 겹침만으로 data leakage라고 해석하지 않는다. 날짜/개체별 모델오류 위치는 신규CNN raw prediction 미저장으로 이 맥락 표에서 알 수 없다.

수량·gap·bin share의 변화가 관측돼도 과적합·분포변화의 인과적 기전은 입증되지 않는다. 조건부 모델오류 변화와 함께 해석해야 한다.
