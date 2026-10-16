# CAP9 평가 연결 오류 수정과 별도 실행

## 현재 기준선 — 완료

CAP9 학습·원본 회수·3seed Validation 집계는 완료됐다. 기존 학습 source134·선택 checkpoint·실패 기록과 기존 CNN/FULL18 Test는 보존한다. 이번 사용자 승인은 Validation 참조 연결 오류 수정·검증과 별도 경로의 Validation9 → Test9 평가다.

## 연결 오류 수정 — 완료

원본 `selected` 객체의 `evaluation_scope=validation_only`, `held_out_test_evaluated=false`를 확인하고, 명시적인 다른 split이 없을 때만 복사본에 `split=validation`을 추가한다. 원본 객체·수치·평가 대상은 바꾸지 않는다. evaluator와 native pipeline의 바이트는 기존 버전과 같다.

14개 계약·회귀검증을 통과했다. 실제 checkpoint9의 로컬 결속 검사도 전부 통과했으며 선택 epoch·binary SHA·tensor SHA·초기 state·선택 metric·history SHA는 기존 실패 경로와 정확히 일치했다. 이 사전검사에는 모델 추론이나 원격 호출이 없다.

수정본 커밋: paper_research, codex/contextual-retrieval-pilot-20261007, `04b04ebf`. 원격 배포 대상은 별도 평가 root뿐이며 원 학습 경로는 수정하지 않는다.

새 bundle: `search_artifacts/titantpp_gru_cap9_test_20261015_v1_repair2`.
새 native root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_gru_cap9_test_20261015_v1_repair2_5080` 및 `_5090`.
새 bootstrap SHA: `b090f31bcf4ae56b2fffdd9f10ea36d19550223808f75f889e86e6a49c4b20cc`.

기존 실패 bundle 전체의 파일 SHA를 `PRESERVED_ORIGINALS.json`에 보존했다. 이 목록은 신규 bootstrap에도 포함돼 원격 실행 전에 재검증된다. 기존 bootstrap과 코드 봉인도 유지한다.

## 실행과 완료 조건

실제 Validation9의 동일 대상·여섯 지표·native Runtime·worker exit0가 모두 통과한 global gate 뒤에만 Test9가 실행된다. 새 학습·선택 변경·Test 기반 조정은 없다. 실패 시 같은 경로에서 자동 재시도하지 않는다.

최종 `final_receipt.json`의 Test9·Validation9·baseline18·3seed·worker exit0 18개·pipeline exit0 4개 및 모든 로컬 SHA를 검증한다. supervisor wrapper 자체 OS exit를 직접 관측했다고 주장하지 않는다.

결과는 전체 raw RMSE·MAE의 3seed 평균·표본SD와 seed별 CAP−CNN/CAP−FULL 절대·상대 차이로 보고한다. 큰 수량 및 시간 지표도 함께 보존·보고한다. 이전 Test 반전 이후 설계된 탐색적 후속 평가이며 독립 평가·보편적 개선·반전 원인 확정으로 해석하지 않는다.
