# 이력 보정27·Instacart9의 기존 Test 최종 비교

사용자의 2026-10-12 “ㅇㅇ 그거 진행하자”는 기존 Test 최종 비교36조건의
접근·추론·검증·기록을 승인한다. 모델은 Validation에서 이미 선택한 checkpoint를
사용한다. 재학습, Test에 따른 epoch·하이퍼파라미터 재선택, 검색 실험, 새 데이터,
공용 Runtime·드라이버·인증 변경은 이 작업에 포함하지 않는다.

2026-10-12 09:32:08 KST 실제 평가 종료, Validation36·Test36·worker72 exit0 및
원본929 SHA 검증·독립 집계 QA까지 완료했다. [최종 관측](FINAL_OBSERVATION.md)과
[최종 QA](FINAL_TEST_QA.md)를 확인한다. Test에서 이력 보정의 양대조·모든3seed
일관 개선 데이터는0개이며, 평균 이득과 적용 예외를 함께 보존했다. 원고 반영이
다음 작업이다. [재정렬한 논문 작업 순서](../titantpp_history_rtx_handoff_20261010_v1/PAPER_WORK_ORDER.md).

## 현재 기준선 — 완료

- 이력 보정27과 Instacart9의 학습·Validation 선택·원본 회수가 완료됐다.
- 과거 Test 접근·생성 기록을 유지한다. 이번 결과는 기존 Test 최종 비교이며
  새 독립 평가라고 부르지 않는다.
- 선택 checkpoint36개, source129/126, 원래 입력 및 분할, 초기화·선택 epoch를
  `search_artifacts/titantpp_history_instacart_test_20261012_v1/contract.json`에 동결했다.
- canonical contract SHA: `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c`.
- 입력·원본의 raw SHA는 별도 `input_integrity.json`에 봉인했다. 과학 원본과
  운영 평가 코드를 별도 봉인하며 기존 원본은 수정하지 않는다.

## 저장 모델이 원래 Validation 결과를 재현하는지 확인 — 완료

- 대상은 5080의 기존 `ai_env`와 지정 GPU UUID이며, 별도 평가 root에만 배포한다.
- six synthetic route probe에서 실제 코드 경로·인과성·단일/배치 일치를 먼저 확인한다.
  이 probe는 실제 데이터·선택 checkpoint를 읽지 않고 학습하지 않는다.
- 36조건 각각의 전체 Validation 여섯 지표를 재추론한다. `math.isclose`의
  relative/absolute tolerance는 각각 `1e-5`이며, 전체/큰 수량 대상 수는 정확히 일치해야 한다.
- RAF의 전체 분할 부모 파일은 원래 Train/Validation export와 행·열 값의 일치를
  확인한다. 학습 통계를 다시 적합하거나 원래 전처리를 바꾸지 않는다.
- 36개 실제 worker exit0와 원본·모델·입력 일치를 확인한 후에만 전역 gate를 만든다.

## 선택 모델의 기존 Test를 비교 — 완료

- 전역 Validation gate 이전에는 Test 추론을 시작하지 않는다.
- 같은 dataset/seed의 세 arm을 같은 5080 Runtime에서 최대3개 동시에 추론한다.
- 전체/큰 수량 RMSE·MAE·TimeNLL과 직전 수량 반복·현재 입력 이력 평균 기준선을
  같은 Test 대상에 계산한다. 수량 기준선에 TimeNLL을 만들어 넣지 않는다.
- Test-only 입력으로 이력을 잘라내지 않고 원래 시간순 입력·관측된 과거 문맥을
  유지한다. 현재 target와 미래 사건은 예측 문맥으로 사용하지 않는다.
- 개별 예측은 저장하지 않고 전체 수량별 충분통계와 대상·정답 digest를 보존한다.
- 조건별3시간, 전체12시간 상한을 둔다. 실패·timeout·claim은 보존하며 자동 retry/resume하지 않는다.

## 실제 종료·원본을 회수하고 변화 방향을 정리 — 완료

- worker·supervisor 실제 종료, native receipt, gate raw SHA, binary input/checkpoint
  불변성, 출력 manifest/archive SHA를 검증한다.
- 3seed 평균·표본 SD와 full 대 대조의 seed별 변화, Validation→Test 방향 반전,
  큰 수량·시간 손해 및 단순 기준선 우위를 빠짐없이 보고한다.
- 원래 A100/RTX 학습 구간 차이는 유지하며, 이번 추론의 5080 Runtime과 구분한다.
- 전체 CPU binary 감사·미노출 기간 확인·논문 구조 원인 분리 실험은 이번 평가의
  완료 조건이 아니다. 이번 결과를 새 독립 일반화·통계 유의성 증명으로 확대하지 않는다.

## 논문에 반영할 주장 범위를 다시 결정 — 다음 작업

- 이번 Test가 끝나면 보정 이득의 유지 여부와 예외를 기준으로 Results·Methods·주장을
  맞춘다. Test 성과를 보고 모델을 재조정하면 해당 결과가 개발에 사용됐음을 기록한다.
- 현재 열린 `main.tex`는 이번 평가 실행 중 수정하지 않는다. 원고 수정은 결과
  검증이 끝난 뒤 별도 작업으로 이어간다.

평가 구현과 결과 분석은 서로 다른 파일에서 병렬 검증한다. 공통 계약·배포·전역
Validation gate·Test 실행·원본 회수는 의존관계에 따라 현재 세션에서 직렬로 처리한다.
