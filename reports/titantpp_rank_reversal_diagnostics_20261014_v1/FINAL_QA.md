# Review Result: PASS

Material Passport: academic-research-suite / experiment-agent validate; 독립 로컬 QA. 새 학습·추론·원격 호출·모델 역직렬화 0.

현재 INTERNAL_ANALYSIS, EXTERNAL_ANALYSIS, INPUT_CONTEXT, 종합 DIAGNOSIS와 생성 스크립트의 증적·계산·해석을 검증했다. 원인식별 완료를 뜻하지 않는다.

### CRITICAL
- 없음

### HIGH
- 없음

### MEDIUM
- 없음

### LOW
- 없음

## 검증 결과

- [x] baseline original/source/evaluation SHA closures: QA_BASELINE.json; old395 and final19 reverified at final review
- [x] internal independent numerical reconstruction: {'source_sha': 144, 'seed_split_pairs': 18, 'bin_contributions': 90, 'histories': 18, 'symmetric_decompositions': 3}
- [x] external independent stored-parquet recomputation: {'source_sha': 380, 'Test_fits': 54, 'Validation_endpoints': 54, 'mean_sampleSD_six_metric_groups': 216, 'seed_MSE_MAE_decompositions': 108}
- [x] input context independent parquet review: {'files': 3, 'split_profiles': 9, 'bin_counts': 45, 'Test_entities_seen_in_Train': True, 'per_entity_sequence_order': True}
- [x] focused automated tests: {'internal_guard_tests': 3, 'external_guard_tests': 4}
- [x] no inference or remote side effects in reviewed analysis scripts: imports and execution paths are local JSON/parquet/saved logs; no torch deserialize/SSH/training; exclusive new report writes
- [x] all11 statistical fallacies reviewed: descriptive risks retained; no formal p-value, CI or significance tests were conducted

## 통계·방법론 오류 검토 — 11/11

- **Simpson paradox (checked_descriptive)**: 구간 비중·조건부 오차를 분해하고 seed별 방향과 평균을 함께 유지한다. 산술기여는 인과 설명이 아니다.
- **Ecological fallacy (observed_risk_controlled)**: 집계 bias를 개별 대상 모두의 과소예측이나 과소예측 건수로 해석하지 않는다. 개체·기간별 모델오류는 미확인이다.
- **Berkson paradox (not_identifiable_here)**: 기존 frozen chronological population을 그대로 사용한다. 표본선택의 인과 편향을 확인하거나 배제하는 설계는 없다.
- **Collider bias (not_applicable_to_reported_estimator)**: 공변량 인과 조정이나 mediation 추정을 수행하지 않는다. 정답 기준 bin은 기술 층화로만 사용한다.
- **Base rate neglect (checked_controlled)**: 각 bin count/share를 제시하고 기여를 SSE/N로 계산한다. 조건부 RMSE와 전체 기여를 구별한다.
- **Regression to mean (observed_risk_controlled)**: Validation 최소 epoch 선택으로 선택 낙관 위험이 있다. 최소 주변 값과 마지막 값이 더 높은 사실만으로 과적합을 확정할 수 없다.
- **Survivorship bias (checked_with_scope_limit)**: 내부 FULL/CNN18 fit·36 split rows와 CNN9 및 외부54 각 split 모두 포함한다. 더 넓은 과거 탐색 캠페인의 선택 이력은 제거되지 않는다.
- **Look elsewhere effect (observed_risk_controlled)**: 외부6 전체와 모든seed·6지표를 보존한다. 사후 bin 해석은 탐색적이며 p값/통계유의성은 제시하지 않는다.
- **Garden of forking paths (observed_risk_controlled)**: Test 접근 후 요청된 진단이다. Train 경계 고정, checkpoint 재선택 없음, 원고/학습 변경 없음. 대칭식 분해 기준을 명시한다.
- **Correlation versus causation (checked_controlled)**: 오류 위치·bias·입력분포 변화·학습로그 관찰과 원인규명을 구분한다. GRU/용량/과적합/분포변화 원인 확정 없음.
- **Reverse causality (not_identifiable_here)**: 성능 반전과 이력·품목·기간 조건의 인과방향을 추정하지 않는다. per-entity chronology만 검증했다.

## 한계 및 변경 파일

- 외부Validation 원시 대상 지문은 저장 endpoint로 복구되지 않으므로 완전한 Validation 행별 대응을 추가 검증했다고 주장하지 않음
- CNN 원시 예측이 없어 기간/품목 오류 대응과 과소예측 건수는 미확인
- Train online log1p MSE와 frozen model raw RMSE는 동일 적합도 지표가 아님
- 3seed와 반복 Test 접근·선택 이력은 분석의 한계이며 QA PASS가 독립검증/인과확정을 뜻하지 않음
- .agents 디렉터리 변경 금지 준수; 지정 execution protocol 파일은 없음

QA 소유 파일: QA_BASELINE.json, FINAL_QA.json, FINAL_QA.md. 기존 source·원본·원고는 수정하지 않았다. 상세 파일 SHA는 FINAL_QA.json의 reviewed_artifacts에 있다.

## 종합 보고서 추가 검토

- [x] DIAGNOSIS.json/md의 원본3분석 SHA 연결·수치·seed예외 일치.
- [x] 큰 수량의 Test 격차 기여와 Validation→Test 격차 변화 기여를 구분했다.
- [x] Taxi 외부 S2P2·AttNHP 대비 seed42·62 반전과 seed52 우세 유지, q>1562 변화 기여 89.86059%를 재확인했다.
- [x] 입력 분포 변화·log/raw 목적 차이·선택 주변 변동은 관찰 사실이며 근본 원인 확정이 아님을 명시했다.
- [x] 원시행 예측·고정 모델 Train raw 오차 부재 및 사후 분석·3seed·cross-campaign 한계를 유지했다.

최종 심각도별 수정 필요 사항 없음. DIAGNOSIS.json SHA: `475b836294032bba0ef68dfd2881c8f4911db297a283381ab338548342ee5764`; DIAGNOSIS.md SHA: `e9fdf9cb38b82f7e13966751ed3da3d43bd508293968f5fa028c7f1f6681668b`.

최종 마무리 변경 검토: 작업 순서·링크만 되돌린 메모리 사본이 직전 JSON/Markdown SHA와 정확히 일치한다. 수치·연구 주장 불변을 확인했다. 추가 자료 계약→승인 범위의 고정 모델 추론(현재 미실행)→논문 반영 순서와 모든 절대경로 링크를 확인했다.
