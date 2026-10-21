## Review Result: PASS

독립 pandas/numpy 재계산으로 저장 Train/Validation 자료를 검산했습니다. 원 분석 스크립트를 재실행하지 않았습니다.

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

검산 개수와 입력 SHA, 범위 한계는 QA.json에 기록했습니다. 코드/입력 수정은 없고 QA.json/.md만 작성했습니다.

완료한 검산: 122 입력 SHA, 36 Validation endpoint, 336 endpoint cell, 112 평균·표본 SD 그룹, 336 seed 차이, 112 차이 요약, 24 행 자료군·60 표본 층, 1,416 행 지표·708 쌍 비교, 총 35,096 수치 비교. 전체 Validation 행 12자료군은 endpoint와 선택 epoch·state SHA가 일치하며 RMSE·MAE·TimeNLL 36개도 허용오차 내 일치했습니다.

초기 MEDIUM 발견은 담당자가 수정했고 독립 재계산에서 해소를 확인했습니다. `analyze_rows.py:150`부터 개체별 동일 가중을 개선된 행 비율에도 적용합니다.

### Acceptance criteria
- [x] Train/Validation 입력 SHA 및 선택 epoch·state 결속
- [x] 3seed 평균·표본 SD(ddof1), seed별 차이
- [x] 같은 target·truth·history 대응과 표본 층별 가중치
- [x] 행·개체별 동일 가중 RMSE/MAE/편향/시간손실 및 개선 비율
- [x] 수량·이력 그룹의 행 수·가중치·오차 가산
- [x] MLP16 Train 부재, seed42 행 비교 한정, current-only 행 부재 명시

표본의 개체별 동일 가중치는 각 그룹에서 관측된 표본 개체의 대표 가중치 합을 정규화한 추정값입니다. 모든 개체의 전수 오류를 직접 측정했다는 뜻은 아닙니다. 과거 혼합 보고서 노출은 계획에 기록돼 있으며 이번 분석은 blinded 분석이 아닙니다.
