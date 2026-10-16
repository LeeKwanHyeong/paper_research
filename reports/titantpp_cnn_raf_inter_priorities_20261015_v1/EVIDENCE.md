# RAF·Intermittent 기존 CNN Train·Validation 수치 근거

Material Passport: academic-research-suite / experiment-agent / validate / v1. Verification Status: ANALYZED.

저장 예측만 재집계했다. 6개 선택 checkpoint × Train·Validation = 12묶음. 1,537,734 seed별 행이며, 중복 seed를 제거한 대상은 512,578행이다. 243개 입력 SHA, 세 seed 대상 일치, 기존 여섯 지표 12묶음 일치, 구간 합계와 MSE 분해를 검증했다. 새 추론·학습·원격 호출·Test 열람 없음.

## 전체 오류와 단순 기준선

CNN은 3seed 평균 ± 표본SD(ddof=1). 직전 수량·관측 이력 평균은 같은 대상의 결정적 기준선으로 seed별 동일하므로 SD=0이다. 별도 학습 3회가 아니다.

| 데이터셋 | split | 예측 | RMSE | MAE | bias |
|---|---|---|---:|---:|---:|
| RAF | train | CNN | 45.9523 ± 0.0705 | 10.6200 ± 0.0460 | -5.0447 ± 0.3200 |
| RAF | train | last_observed | 57.6911 ± 0.0000 | 13.4251 ± 0.0000 | 0.0543 ± 0.0000 |
| RAF | train | history_mean | 47.7120 ± 0.0000 | 12.2489 ± 0.0000 | -0.0802 ± 0.0000 |
| RAF | validation | CNN | 34.4314 ± 0.0732 | 9.2183 ± 0.0280 | -3.1708 ± 0.1247 |
| RAF | validation | last_observed | 44.5742 ± 0.0000 | 11.3136 ± 0.0000 | -0.0819 ± 0.0000 |
| RAF | validation | history_mean | 36.8499 ± 0.0000 | 10.8564 ± 0.0000 | 0.5857 ± 0.0000 |
| Intermittent | train | CNN | 1.5393 ± 0.0775 | 0.6654 ± 0.0455 | -0.0769 ± 0.1535 |
| Intermittent | train | last_observed | 1.6448 ± 0.0000 | 0.6669 ± 0.0000 | -0.2828 ± 0.0000 |
| Intermittent | train | history_mean | 14.3252 ± 0.0000 | 5.0023 ± 0.0000 | -4.7178 ± 0.0000 |
| Intermittent | validation | CNN | 1.7230 ± 0.0730 | 0.7356 ± 0.0611 | -0.2084 ± 0.1178 |
| Intermittent | validation | last_observed | 1.7078 ± 0.0000 | 0.5647 ± 0.0000 | 0.1932 ± 0.0000 |
| Intermittent | validation | history_mean | 13.7821 ± 0.0000 | 4.5537 ± 0.0000 | -2.2784 ± 0.0000 |

## CNN의 여섯 지표

큰 수량: RAF q>200, Intermittent q>187. TimeNLL은 기존 recorded positive-integer 정의다. 단순 수량 기준선 TimeNLL은 N/A.

| 데이터셋 | split | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |
|---|---|---:|---:|---:|---:|---:|---:|
| Intermittent | train | 1.53929 ± 0.077459 | 0.665434 ± 0.0455459 | 8.78919 ± 0.392124 | 6.67895 ± 0.413909 | 0.257447 ± 0.0305408 | 9.27867e-08 ± 1.60711e-07 |
| Intermittent | validation | 1.72297 ± 0.0730006 | 0.735638 ± 0.0611067 | 7.97078 ± 0.155409 | 6.24562 ± 0.192856 | 0.346241 ± 0.0505996 | 1.94824e-06 ± 3.37446e-06 |
| RAF | train | 45.9523 ± 0.0704749 | 10.62 ± 0.0460159 | 464.616 ± 1.6692 | 357.395 ± 2.57153 | 3.05499 ± 0.000623584 | 2.92205 ± 0.00858845 |
| RAF | validation | 34.4314 ± 0.0732011 | 9.21827 ± 0.0280019 | 330.758 ± 2.554 | 283.555 ± 3.01648 | 3.49746 ± 0.138992 | 3.36318 ± 0.168363 |

## 큰 수량 및 관측 이력 조건별 CNN 오류

SSE는 제곱오차 합계다. target_tail은 결과를 보고 설명하는 구간이며 입력 또는 gate로 사용하지 않는다. 나머지 구간은 예측 전에 관측 가능한 이력이다. 마지막 관측 수량 경계: RAF 1/2/10, Intermittent 1/2/20.

| 데이터셋 | split | 조건 | 행/개체 | 비중 | RMSE | MAE | bias | SSE 비중 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| RAF | train | target_tail_descriptive_only:tail | 205/126 | 0.80% | 464.6157 ± 1.6692 | 357.3946 ± 2.5715 | -357.3890 ± 2.5798 | 81.29% |
| RAF | train | target_tail_descriptive_only:non_tail | 25574/4998 | 99.20% | 19.9529 ± 0.1839 | 7.8403 ± 0.0332 | -2.2203 ± 0.3344 | 18.71% |
| RAF | train | observed_direction:falling | 6932/3716 | 26.89% | 56.2142 ± 0.0757 | 12.5347 ± 0.0657 | -6.1972 ± 0.5228 | 40.24% |
| RAF | train | observed_direction:flat | 6906/3098 | 26.79% | 33.8303 ± 0.1091 | 5.1864 ± 0.0597 | -2.4670 ± 0.1780 | 14.52% |
| RAF | train | observed_direction:rising | 6941/3685 | 26.93% | 45.0397 ± 0.1536 | 13.8740 ± 0.0881 | -5.3974 ± 0.4424 | 25.87% |
| RAF | train | observed_direction:unavailable | 5000/5000 | 19.40% | 45.9249 ± 0.1964 | 10.9531 ± 0.0492 | -6.5175 ± 0.1839 | 19.37% |
| RAF | train | last_quantity:3 | 5136/1877 | 19.92% | 90.4370 ± 0.2521 | 34.6509 ± 0.1192 | -15.9135 ± 0.8251 | 77.17% |
| RAF | validation | target_tail_descriptive_only:tail | 50/45 | 0.75% | 330.7576 ± 2.5540 | 283.5554 ± 3.0165 | -283.5554 ± 3.0165 | 68.97% |
| RAF | validation | target_tail_descriptive_only:non_tail | 6640/4103 | 99.25% | 19.2508 ± 0.1998 | 7.1525 ± 0.0150 | -1.0594 ± 0.1353 | 31.03% |
| RAF | validation | observed_direction:falling | 2201/1914 | 32.90% | 41.1464 ± 0.0680 | 11.8970 ± 0.0601 | -4.3621 ± 0.2578 | 46.98% |
| RAF | validation | observed_direction:flat | 2270/1746 | 33.93% | 19.2341 ± 0.0849 | 4.0833 ± 0.0443 | -1.4696 ± 0.0822 | 10.59% |
| RAF | validation | observed_direction:rising | 2219/1983 | 33.17% | 38.9415 ± 0.1323 | 11.8143 ± 0.0180 | -3.7293 ± 0.1993 | 42.43% |
| RAF | validation | observed_direction:unavailable | 0/0 | 0.00% | N/A | N/A | N/A | 0.00% |
| RAF | validation | last_quantity:3 | 1230/970 | 18.39% | 71.7281 ± 0.2414 | 30.9558 ± 0.1280 | -9.7025 ± 0.8707 | 79.79% |
| Intermittent | train | target_tail_descriptive_only:tail | 3972/132 | 1.01% | 8.7892 ± 0.3921 | 6.6790 ± 0.4139 | 1.0431 ± 5.5871 | 32.90% |
| Intermittent | train | target_tail_descriptive_only:non_tail | 389852/5000 | 98.99% | 1.2675 ± 0.0674 | 0.6042 ± 0.0418 | -0.0883 ± 0.1072 | 67.10% |
| Intermittent | train | observed_direction:falling | 45558/4109 | 11.57% | 2.0907 ± 0.1949 | 1.1013 ± 0.0569 | -0.1355 ± 0.2714 | 21.63% |
| Intermittent | train | observed_direction:flat | 241070/4971 | 61.21% | 0.8285 ± 0.0727 | 0.3352 ± 0.0366 | -0.0742 ± 0.0536 | 17.71% |
| Intermittent | train | observed_direction:rising | 102196/4112 | 25.95% | 2.2399 ± 0.1653 | 1.2116 ± 0.0727 | -0.0378 ± 0.3552 | 54.93% |
| Intermittent | train | observed_direction:unavailable | 5000/5000 | 1.27% | 3.2694 ± 0.1243 | 1.4538 ± 0.0241 | -0.4729 ± 0.2117 | 5.74% |
| Intermittent | train | last_quantity:3 | 97170/2395 | 24.67% | 2.8390 ± 0.1571 | 1.6870 ± 0.1534 | -0.2421 ± 0.5516 | 83.90% |
| Intermittent | validation | target_tail_descriptive_only:tail | 1141/99 | 1.32% | 7.9708 ± 0.1554 | 6.2456 ± 0.1929 | -0.5929 ± 0.9315 | 28.37% |
| Intermittent | validation | target_tail_descriptive_only:non_tail | 85144/4936 | 98.68% | 1.4684 ± 0.0804 | 0.6618 ± 0.0594 | -0.2033 ± 0.1162 | 71.63% |
| Intermittent | validation | observed_direction:falling | 15532/3390 | 18.00% | 2.8930 ± 0.1427 | 1.6606 ± 0.1461 | -0.6038 ± 0.2243 | 50.74% |
| Intermittent | validation | observed_direction:flat | 59064/4833 | 68.45% | 1.0662 ± 0.0707 | 0.4105 ± 0.0410 | -0.1263 ± 0.0685 | 26.19% |
| Intermittent | validation | observed_direction:rising | 11689/3165 | 13.55% | 2.2450 ± 0.0118 | 1.1496 ± 0.0699 | -0.0981 ± 0.2355 | 23.07% |
| Intermittent | validation | observed_direction:unavailable | 0/0 | 0.00% | N/A | N/A | N/A | 0.00% |
| Intermittent | validation | last_quantity:3 | 28466/2158 | 32.99% | 2.9393 ± 0.1294 | 1.7591 ± 0.1713 | -0.6676 ± 0.3731 | 96.00% |

## Intermittent seed별 직전 수량 대비 차이

차이는 CNN−직전 수량이며 양수이면 CNN 오류가 더 크다. RMSE에서 seed42 예외를 보존한다.

| seed | split | CNN RMSE | 직전 수량 RMSE | 차이 | CNN MAE | 직전 수량 MAE | 차이 |
|---|---|---:|---:|---:|---:|---:|---:|
| 42 | train | 1.458483 | 1.644770 | -0.186286 | 0.620737 | 0.666854 | -0.046117 |
| 42 | validation | 1.645086 | 1.707803 | -0.062717 | 0.676601 | 0.564733 | +0.111868 |
| 52 | train | 1.612900 | 1.644770 | -0.031870 | 0.711784 | 0.666854 | +0.044930 |
| 52 | validation | 1.733973 | 1.707803 | +0.026171 | 0.731692 | 0.564733 | +0.166959 |
| 62 | train | 1.546474 | 1.644770 | -0.098296 | 0.663782 | 0.666854 | -0.003072 |
| 62 | validation | 1.789837 | 1.707803 | +0.082034 | 0.798623 | 0.564733 | +0.233890 |

## Intermittent: 조건을 맞춘 하락−상승 차이

기존 공통 개체·수량·변화 폭 매칭의 마지막 관측 수량 상위 사분위(>20)를 그대로 재사용한다. 1,364 공통 개체, Validation 방향성 행의 50.91% 포함. 전체 모집단의 효과가 아니다.

| seed | split | bias 차이 | RMSE 차이 | MAE 차이 |
|---|---|---:|---:|---:|
| 42 | train | -0.354455 | +0.435238 | +0.192871 |
| 52 | train | -0.415608 | +0.197801 | +0.344052 |
| 62 | train | -0.426328 | +0.685773 | +0.412691 |
| 42 | validation | -0.415504 | +0.082105 | +0.234285 |
| 52 | validation | -0.393855 | +0.213748 | +0.394785 |
| 62 | validation | -0.398843 | +0.479456 | +0.601851 |

## 미래 target 이동은 사후 설명에만 사용

아래 비율은 세 seed에서 동일한 실제 대상의 분포이며, 한 번만 계산한 값이다. 예측 시 사용할 수 없다. 관측 상승 후 재상승 비율의 변화만으로 concept drift나 CNN 원인을 확정하지 않는다.

| 데이터셋 | split | 관측 방향 | 다음 실제 방향 | 조건 내 비율 |
|---|---|---|---|---:|
| RAF | train | falling | falling | 20.17% |
| RAF | train | falling | flat | 29.34% |
| RAF | train | falling | rising | 50.49% |
| RAF | train | rising | falling | 66.23% |
| RAF | train | rising | flat | 11.91% |
| RAF | train | rising | rising | 21.86% |
| RAF | validation | falling | falling | 19.85% |
| RAF | validation | falling | flat | 32.35% |
| RAF | validation | falling | rising | 47.80% |
| RAF | validation | rising | falling | 66.83% |
| RAF | validation | rising | flat | 13.74% |
| RAF | validation | rising | rising | 19.42% |
| Intermittent | train | falling | falling | 13.67% |
| Intermittent | train | falling | flat | 37.75% |
| Intermittent | train | falling | rising | 48.58% |
| Intermittent | train | rising | falling | 27.58% |
| Intermittent | train | rising | flat | 29.33% |
| Intermittent | train | rising | rising | 43.09% |
| Intermittent | validation | falling | falling | 27.01% |
| Intermittent | validation | falling | flat | 34.66% |
| Intermittent | validation | falling | rising | 38.33% |
| Intermittent | validation | rising | falling | 60.46% |
| Intermittent | validation | rising | flat | 33.07% |
| Intermittent | validation | rising | rising | 6.48% |

## 방향별 MSE 차이의 대수 분해

ΔMSE = Σ(Validation 비중−Train 비중)×Train 조건 MSE + ΣValidation 비중×(Validation 조건 MSE−Train 조건 MSE). 기준 순서에 의존하는 서술적 분해이며 인과 기여가 아니다. 같은 방향 내부에도 개체·수량·시간 차이가 남는다.

| 데이터셋 | seed | 전체 ΔMSE | 방향 구성비 항 | 방향 내부 MSE 항 |
|---|---:|---:|---:|
| RAF | 42 | -922.847064 | -11.623686 | -911.223378 |
| RAF | 52 | -927.234503 | -7.338011 | -919.896492 |
| RAF | 62 | -928.187847 | -13.329643 | -914.858204 |
| Intermittent | 42 | +0.579135 | -0.373794 | +0.952929 |
| Intermittent | 52 | +0.405218 | -0.582526 | +0.987743 |
| Intermittent | 62 | +0.811936 | -0.326533 | +1.138470 |

## 보존한 한계

- 현재 모델의 Train은 학습에, Validation은 checkpoint 선택에 사용됐다. 새로운 독립 평가가 아니다.
- 세 seed는 동일 대상의 학습 반복이다. 행 사이 시간 의존성·개체 내 의존성이 있으며 유의성 검정을 하지 않았다.
- Intermittent seed42/52는 5090/cuDNN92000, seed62는 5080/cuDNN91900이다. seed 차이에 장비·Runtime 차이가 섞인다. 이번에는 실행시간 비교를 하지 않았다.
- RAF의 기존 매칭 공통 표본 0개는 차이가 없음을 뜻하지 않는다. 작은 tail50 및 오류 집중 개체를 일반 모집단으로 확대하지 않는다.
- 과거 간격이 저장된 진단 행에 없어 간격 적응 후보를 근거 있는 1순위로 두지 않았다. 원 모델이 시간 입력을 안 받는다는 뜻이 아니다.
- 메모리 감쇠는 고정 가중치 추론 개입이다. 재학습한 메모리 gate의 효과나 모든 memory 설계를 판정하지 않는다.
- 모든 seed·구간·단순 기준선·메모리 여섯 지표는 RESULTS.json에 보존했다.
