# 고정 모델 행별 진단: QA 실행 전 계획

Material Passport: academic-research-suite / experiment-agent validate. 상태: **검토 준비 완료, 실제 추론 승인 판정은 계약·평가 도구 검토 후**. QA는 원격 호출·모델 추론·checkpoint 역직렬화를 수행하지 않는다.

## 범위와 기준선

- 후보 신규 추론: source131 FULL/CNN 18개 모델 × Train/Validation/Test = 54개 작업, 외부 54개 모델 × Train/Validation = 108개 작업. 합계 **162개**다. 최종 계약의 고유 ID·split 목록과 일치해야 한다.
- 기존 외부 Test54 저장 예측은 SHA 검증 후 재사용한다. 새 추론 목록에 포함하지 않는다. 원본·기존 과학 source·학습·claim·Runtime·원고를 변경하지 않는다.
- 이전 source/input/result 봉인 395개 및 최종 구성요소36 파일 연결을 재검증했다. 앞선 반전 진단의 최종 QA에 결속된 모든 산출물 SHA도 일치한다.
- 이 문서는 실행 전 요구사항이다. 아직 없는 신규 계약·도구·실제 실행을 PASS로 표시하지 않는다.

## 실행을 허용하기 위한 필수 검증

1. **모델 신원:** 캠페인·source·dataset·arm·seed·선택 epoch·binary SHA·전체 state_dict SHA·전처리·입력을 묶는다. 이미 선택한 checkpoint를 사용하며 Test 재선택이나 학습을 하지 않는다.
2. **Train 모집단:** 고정 loader가 실제 예측 가능한 대상(최소 한 개의 과거 관측)을 그대로 사용한다. 전체 Train 통계 행과 예측 대상은 별개다. 첫 시점 포함 통계 및 경계는 기존 값을 유지한다. eligible count·정렬·target ID·truth·recorded time 지문을 사전 고정한다.
3. **과거 관측 경계:** Train target은 Train 과거만, Validation target은 허용된 Train/Validation 과거만 사용한다. Test는 원 계약의 관측 이력 창을 유지한다. 현재 target 수량·미래 시점·패딩이 예측에 들어가지 않는지 합성 입력으로 검증한다.
4. **구조 및 누출 검증:** target 수량을 바꿔도 수량 예측과 시간 점수가 같고, target gap을 바꿔도 수량 예측이 같아야 한다. 시간 점수는 관측 gap에 의존하므로 gap 변조 불변을 요구하지 않는다. batch/single 예측 일치 및 구조별 전달 경로를 확인한다.
5. **추론 불변성:** eval mode, no_grad/inference_mode, gradient 비활성화. parameter뿐 아니라 buffer를 포함한 전체 모델 tensor SHA를 전후 확인한다. checkpoint/data/source/계약/Runtime도 전후 일치해야 한다. optimizer update나 상태 학습을 호출하지 않는다.
6. **원 결과 재현:** 기존 선택 Validation 및 source131 Test의 전체·큰 수량 여섯 지표와 모든 수량 구간의 count/SSE/absolute sum을 사전 허용오차로 재현한다. 처음 접하는 Train에는 과거 raw 예측 결과가 없음을 유지한다.
7. **행별 출력 완전성:** target_index 연속, ID 중복·누락 없음, 기대 행 수, target/truth/time SHA, 각 part SHA, 전체 manifest를 확인한다. target 정보와 예측의 batch 순서가 일치해야 한다. 오류 발생 시 부분 파일을 완료 결과로 쓰지 않는다.
8. **실행 가드:** 승인 host/GPU UUID/Runtime·job 목록·기한을 확인한다. 새 격리 경로에 exclusive claim/영수증을 기록하고 실제 supervised worker exit0를 검증한다. 기존 작업을 종료하거나 claim/failure를 삭제하지 않으며 자동 재시도하지 않는다.
9. **외부 기존 Test 재사용:** 모든 part SHA 및 10개 공통 truth 필드와 recorded time 지문을 확인한다. 신규 외부 Validation row의 대상 검증과, 원래 저장되지 않았던 과거 Validation row 지문의 복구를 구분한다.
10. **집계 및 해석:** 동일 행·동일 seed 비교, 표본SD(ddof=1), raw MSE와 평균RMSE 구분, 고정 Train 수량 경계 유지. seed 간 시간은 host별 분리한다. 행별 진단을 추가해도 사후 분석·인과 미확정 경계는 유지한다.

## 완료 조건과 다음 순서

- 계약·평가 도구와 합성 테스트를 검토해 QA_PREFLIGHT.json에 기록한다. 필수 가드 미충족 시 근거 위치·심각도·수정안을 보고한다.
- 실행 후 실제 worker 종료·모든 원본 및 출력 SHA·162개 신규 작업과54개 재사용 범위를 검증한다. 실제 승인 계약이 더 작은 범위로 확정되면 그 계약 범위를 기준으로 검증한다.
- 마지막에 동일 대상 수치와 해석을 독립 재계산하고 FINAL_QA.json/md를 기록한다. 원본 및 과학 코드는 QA가 수정하지 않는다.

## 현재 심각도별 발견사항

- CRITICAL: 없음(아직 신규 구현 검토 전).
- HIGH: 없음(아직 신규 구현 검토 전).
- MEDIUM: 없음(아직 신규 구현 검토 전).
- LOW: 없음(아직 신규 구현 검토 전).
