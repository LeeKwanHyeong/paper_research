# 동결된 Train·Validation·Test 대상의 구성 진단

**상태: 완료.** 선택 checkpoint나 모델을 읽지 않고, 원래 frozen loader의 적격 next-event target만 CPU로 집계했다. 새 학습·추론·원격 명령은 0회다.

Train/Validation의 target 수·identity·quantity SHA 및 target duration SHA가 원계약과 일치한다. Validation/Test 모두 native target digest와 모든 대상의 truth/history digest가 저장된 영수증에 정확히 일치한다. 따라서 이력 길이·직전 수량·이력 평균의 재구성도 native 평가와 일치한다. 직접 loader sample 검사도 통과했다. 첫 관측행을 평가 대상으로 잘못 포함하지 않았다.

| 자료·split | 평가 대상/series | 수량 평균·p99·max | 큰 수량 비중 | gap 평균·p99 | 이력 길이 평균·p50 |
|---|---:|---:|---:|---:|---:|
| Taxi train | 38,393/131 | 229.027 · 3448.080 · 6322 | 0.9976% (383) | 1.740 · 15.000 | 111.25 · 129 |
| Taxi validation | 8,268/131 | 213.456 · 3424.620 · 6489 | 0.9555% (79) | 1.779 · 16.000 | 132.64 · 157 |
| Taxi test | 8,327/131 | 229.713 · 3446.660 · 5952 | 0.9968% (83) | 1.846 · 16.000 | 126.44 · 150 |
| Intermittent train | 393,824/5,000 | 15.238 · 188.000 · 477 | 1.0086% (3,972) | 2.997 · 7.000 | 51.95 · 40 |
| Intermittent validation | 86,285/5,000 | 17.741 · 212.000 · 464 | 1.3224% (1,141) | 2.767 · 5.000 | 113.91 · 93 |
| Intermittent test | 88,019/5,000 | 11.042 · 106.000 · 350 | 0.2761% (243) | 3.624 · 12.000 | 135.72 · 110 |
| RAF train | 25,779/5,000 | 14.661 · 200.000 · 2062 | 0.7952% (205) | 8.165 · 29.000 | 3.61 · 3 |
| RAF validation | 6,690/4,121 | 12.765 · 200.000 · 1000 | 0.7474% (50) | 10.082 · 31.000 | 6.82 · 7 |
| RAF test | 5,226/3,307 | 14.152 · 200.000 · 1000 | 0.7463% (39) | 10.084 · 31.000 | 8.24 · 8 |

**Validation→Test에서 확인된 변화**

- Taxi: 수량 평균 +7.62%, gap 평균 +3.79%, 큰 수량 빈도 +0.0413%p. 수량 구간 구성 거리(TV) 0.0133, 이력 길이 구성 거리 0.0519, series별 대상 비중 거리 0.0037. Train에 없던 Test series는 0개다.
  Validation 기록 기간 2015-01-19T09:00:00–2015-01-29T05:00:00; Test 2015-01-24T16:00:00–2015-01-31T23:00:00. 분할은 series별이어서 자료 전체의 기간은 겹칠 수 있다.
- Intermittent: 수량 평균 -37.76%, gap 평균 +30.96%, 큰 수량 빈도 -1.0463%p. 수량 구간 구성 거리(TV) 0.0711, 이력 길이 구성 거리 0.2044, series별 대상 비중 거리 0.0121. Train에 없던 Test series는 0개다.
  Validation 기록 기간 2020-11-02T00:00:00–2029-02-26T00:00:00; Test 2021-09-13T00:00:00–2030-03-18T00:00:00. 분할은 series별이어서 자료 전체의 기간은 겹칠 수 있다.
- RAF: 수량 평균 +10.86%, gap 평균 +0.02%, 큰 수량 빈도 -0.0011%p. 수량 구간 구성 거리(TV) 0.0284, 이력 길이 구성 거리 0.1933, series별 대상 비중 거리 0.5030. Train에 없던 Test series는 0개다.
  Validation 기록 기간 2000-11-01T00:00:00–2001-11-01T00:00:00; Test 2001-12-01T00:00:00–2002-12-01T00:00:00. 분할은 series별이어서 자료 전체의 기간은 겹칠 수 있다.

**이력과 대상 구성에서 확인한 진단 단서**

- RAF 이력 최대값은 Train 14, Validation 16, Test 19개다. 이력 16개 이상인 대상은 각각 0, 3, 41개다. 이는 학습 때보다 깊은 recurrent state 입력을 받는 외삽 후보이며, lag16 branch가 새로 활성화되는 현상이 아니다. 동결된 CNN+GRU54의 GRU 이용 조건은 관측 이력 2개 이상이고, 54는 hidden width다. lag16·32·64·128 활성화 gate나 sqrt(n) 보정 정규화는 이 구조에 없다.
- JSON의 `available_lag_target_counts`와 `mean_boundary_available_counts`는 과거 구조와 비교할 수 있도록 집계한 이력 개수 기준이다. 선택된 CNN+GRU54의 실제 gate나 활성 경로로 해석하지 않는다. 해당 계산값은 변경하지 않았다.
- RAF Test의 879 series는 Validation에 없고, Validation의 1,693 series는 Test에 없다. 모두 Train에 있던 series다. 따라서 새 entity 문제보다 평가 대상 series 구성과 성숙한 이력의 차이를 먼저 구분해야 한다.
- Intermittent Test의 3,655개 대상은 sequence cap으로 과거를 왼쪽에서 잘라낸다. Train/Validation에서는 이 경우가 0개다. 이력 평균과 target 수량의 기술적 Pearson 상관은 Validation 0.938에서 Test 0.805로 달라진다. 이는 loss-free association 요약이며 반전 원인 확정이나 독립 표본 검정이 아니다.
- Taxi 큰 수량은 Validation/Test 모두 4개 series에서 나온다. 전체 구성이 비교적 비슷하므로 seed 반전을 강한 자료 분포 변화 하나로 설명할 근거는 약하다.
- 날짜는 파일의 기록 날짜다. 특히 Intermittent 날짜 범위가 미래 연도까지 뻗는다는 사실을 실제 미래 관측 자료나 독립 미래 평가의 증거로 해석하지 않는다.

**원인 해석의 한계**

- 구성 차이는 실제로 확인되었지만, 이것만으로 보정 경로의 손해 원인이나 조건부 분포 변화가 증명되지는 않는다.
- 세 seed의 평가 대상은 같으므로, 자료 구성 변화만으로 특정 seed만 반전하는 이유를 설명할 수 없다. 학습 경로·선택 epoch·모델별 오류 분해와 함께 보아야 한다.
- 큰 수량의 squared quantity share는 오차 SSE의 비중과 다르다. 모델별 SSE 분해는 저장된 native 결과에서 별도로 확인해야 한다.
- 이력 길이와 legacy lag 개수 기준은 계산 입력의 특성이다. 실제 learned correction의 크기·부호·recurrent state 변화는 직접 관측한 것이 아니다.
- 불변 Train 통계를 새로 맞추지 않았다. 모든 duration 단위(hour/week/month)와 top-code 없음은 원계약을 그대로 따랐다.

**추가 기록의 최소 범위 — 다음 작업 / 별도 평가 실행 전 계약 필요**

- 기존 선택 모델을 바꾸지 않고, 사전에 고정한 series·기간·이력 길이·수량 구간별 signed error/SSE/MAE/TimeNLL 및 correction 크기·부호를 같은 대상에 함께 기록한다. 모델별 반전이 적은 수의 series나 특정 이력 조건에 집중되는지 확인한다.
- split별 전체 구성을 공통 고정 가중치로 재집계하되, 원래 전체 Test 수치도 함께 남긴다. 가중치 실험은 성능 개선 주장이 아니라 구성 변화 민감도 진단이다.
- Test를 이미 보고 정한 후속 변경은 개발 조치다. 독립 일반화 확인에는 이후에 동결된 새 평가 계약과 미노출 자료가 필요하다.

계약 canonical SHA: `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c`. Source closure: `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`.
각 입력·loader·native receipt의 경로와 raw SHA, 모든 분포 통계 및 정체성 검증은 `population_diagnosis.json`에 있다.
