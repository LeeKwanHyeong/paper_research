# 저장 행 분석 독립 검산 — PASS_independent_descriptive_row_analysis

생산 분석 코드를 import하거나 재실행하지 않고, 원본 저장 예측의 가중 모멘트와 공통 cell 충분통계로 별도 계산했다.

- 33개 cohort의 기준선/관측참조 지표 4314행, 표준화 지표 2788행, 모델 간 차이 708행을 검산했다.
- 수치 52,206개 비교에서 불일치 0건. 최대 절대차 1.81898940355e-12.
- 공통 support cell 40,586개 및 개체 동일질량 16,453개를 검증했다. 양 regime의 cell 구성은 동일하다.
- 18개 paired cohort, 149,231개 target에서 필수 identity와 저장 truth/history·가중치 일치를 확인했다.
- 입력 145개 SHA가 일치했다.

공통 cell 질량은 양 regime 가중질량의 최솟값이다. entity_equal은 entity를 cell에 포함하고 공통 개체 질량으로 정규화하므로 각 개체의 총질량이 1이며 양쪽의 동일 cell 구성을 유지한다. support가 없으면 N/A를 보존하고, 각 side 30행 미만 또는 5개체 미만은 sparse로 표시한다.

표본 Train은 가중 층화표본이며 전수 Train으로 해석할 수 없다. target_id가 없는 legacy 자료에서는 이를 만들지 않고 target_index/entity_id/seq/split을 비교한다. truth/history 수치 비교는 고정 허용오차(rtol=1e-7, atol=1e-6)를 사용한다. 이 사후 조건별 비교는 원인이나 새 백본 효과를 입증하지 않는다.

새 학습·추론·원격 조회·Test 열람·CNN 변경은 수행하지 않았다.
