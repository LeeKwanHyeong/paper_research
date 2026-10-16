# 같은 개체·수량·변화 폭을 맞춘 CNN 오류

Train·Validation 저장 예측만 사용. 새 추론·학습·원격·Test 열람 없음.

## Taxi 상위 관측 수량 구간: 하락−상승 bias 차이

양수는 하락 뒤 예측−실제 오차가 상승 뒤보다 더 크다는 뜻이다. 수량 단위이며 RMSE 차이가 아니다.

| seed | Train 매칭 | Validation 매칭 | Train 공통 개체 | Validation 공통 개체 |
|---|---:|---:|---:|---:|
| 42 | +4.8798 | +35.5957 | +2.7487 | +25.8353 |
| 52 | +20.3841 | +56.7940 | +16.1194 | +43.1934 |
| 62 | +1.0213 | +36.8625 | -0.7153 | +27.1530 |

## 매칭 범위와 잔여 차이

| 데이터셋/구간 | split | 방식 | 하락/상승 행 | 개체 | 방향별 가중치 합 | 남은 비율 | 수량 SMD | 변화 폭 SMD |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Intermittent/-1 | train | matched | 19661/32360 | 3103 | 16970.00 | 0.3521 | -0.0125 | -0.0585 |
| Intermittent/3 | train | matched | 13119/23835 | 2116 | 11862.00 | 0.5593 | -0.0077 | -0.0111 |
| Intermittent/-1 | train | common_entity | 10325/18831 | 1647 | 1647.00 | 0.1973 | -0.0157 | -0.0571 |
| Intermittent/3 | train | common_entity | 8397/16839 | 1364 | 1364.00 | 0.3819 | -0.0111 | -0.0500 |
| Intermittent/-1 | validation | matched | 5386/5087 | 1771 | 3974.00 | 0.3847 | -0.0369 | 0.0063 |
| Intermittent/3 | validation | matched | 4640/4403 | 1488 | 3333.00 | 0.5379 | -0.0303 | 0.0371 |
| Intermittent/-1 | validation | common_entity | 5171/4812 | 1647 | 1647.00 | 0.3667 | -0.0355 | -0.0023 |
| Intermittent/3 | validation | common_entity | 4428/4131 | 1364 | 1364.00 | 0.5091 | -0.0301 | 0.0365 |
| RAF/-1 | train | matched | 18/19 | 18 | 18.00 | 0.0027 | -0.0177 | 0.0262 |
| RAF/3 | train | matched | 6/7 | 6 | 6.00 | 0.0035 | -0.0191 | 0.0097 |
| RAF/-1 | train | common_entity | 0/0 | 0 | 0.00 | 0.0000 | N/A | N/A |
| RAF/3 | train | common_entity | 0/0 | 0 | 0.00 | 0.0000 | N/A | N/A |
| RAF/-1 | validation | matched | 3/3 | 3 | 3.00 | 0.0014 | -2.0000 | 0.0539 |
| RAF/3 | validation | matched | 0/0 | 0 | 0.00 | 0.0000 | N/A | N/A |
| RAF/-1 | validation | common_entity | 0/0 | 0 | 0.00 | 0.0000 | N/A | N/A |
| RAF/3 | validation | common_entity | 0/0 | 0 | 0.00 | 0.0000 | N/A | N/A |
| Taxi/-1 | train | matched | 4704/5115 | 68 | 3521.00 | 0.3218 | -0.0011 | -0.0029 |
| Taxi/3 | train | matched | 2624/2560 | 29 | 1903.00 | 0.5471 | -0.0021 | -0.0017 |
| Taxi/-1 | train | common_entity | 4442/4731 | 56 | 66.00 | 0.3006 | -0.0014 | -0.0149 |
| Taxi/3 | train | common_entity | 2579/2518 | 21 | 21.00 | 0.5379 | -0.0025 | -0.0001 |
| Taxi/-1 | validation | matched | 423/444 | 56 | 347.00 | 0.1330 | -0.0012 | 0.0024 |
| Taxi/3 | validation | matched | 243/243 | 21 | 191.00 | 0.2521 | -0.0024 | 0.0095 |
| Taxi/-1 | validation | common_entity | 423/444 | 56 | 66.00 | 0.1330 | -0.0006 | -0.0071 |
| Taxi/3 | validation | common_entity | 243/243 | 21 | 21.00 | 0.2521 | -0.0001 | 0.0068 |

후속 구현 근거: True; 반복 seed: [42, 52]

개체 및 이력 조건을 맞춘 표본에 대한 기술적 비교다. 제외된 표본·시간 의존성·미측정 조건의 영향을 배제하지 못한다.
앞선 10% 기준은 미충족 그대로다. 이번 매칭의 차이 유지 여부는 별도 질문이며 모델 성능 개선이나 인과관계가 아니다.
전체 및 각 사분위의 seed별 RMSE·MAE·bias·TimeNLL, 포함률·ESS·balance는 ANALYSIS.json에 모두 기록했다.
공통 개체의 가중치는 사분위마다 동일하다. 전체(-1)는 사분위별 공통 개체 비교를 합친 보조 요약이다.
