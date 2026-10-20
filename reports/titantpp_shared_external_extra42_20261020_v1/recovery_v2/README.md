# 5080의 짧은 종료 확인과 자동 회수

**기존 학습과 기록을 보존한다 — 완료**
- 새 사용자 승인 원문은 AUTHORIZATION.json에 보존한다. 기존 학습·checkpoint·완료 결과·원본 실패·v1 복구 실패는 모두 원문과 SHA를 유지한다. 기존 local long-wait PID35139/1564/1567 부재를 확인했다. 원격 학습·평가 명령, Runtime, 5090 작업은 변경하지 않는다.

**연결을 유지하지 않고 종료 여부만 확인한다 — 구현·독립 검증 완료**
- 신규 합성 검사37개와 독립 검사37개가 실제exit0으로 통과했다. 독립검토에서 기존1,013파일의 검사전후SHA가 같았고 열린 발견은0이다. TESTS.json과 INDEPENDENT_QA.json에 근거와 실제로 보존하지 않은 시험시각·stdout/stderr 분리의 한계를 기록했다.
- 기존 자동화2seed의15분 간격을 유지한다. 한 회차에 최대45초의 읽기 전용 SSH를 한 번 열어 같은 contract와 dispatcher의 종료 여부를 확인하고 즉시 닫는다. sleep/무한대기/백그라운드 SSH를 만들지 않는다.
- 정상 대기는 waiting이다. 새로 승인된 단기 확인의 SSH255/transport timeout만 원문을 보존하고 다음 정규 회차에 재확인한다. 같은 회차의 재접속은0, 연속3회까지이며3회 연속이면 중단·보고한다. 정상 확인 성공 시 연속횟수는0으로 하되 과거 라운드는 삭제하지 않는다.
- 신원 불명·과학실패·회수실패·불완전 회수는 첫 발생부터 중단한다. 학습·평가·회수 재시도나 resume은 허용하지 않는다.

**기존 평가 완료를 검증하고 한 번 회수한다 — 종료 대기**
- 기존 dispatcher가 Train42→Validation42 전체 재현 gate→선택 고정 Test42를 수행한다. 종료 확인 도구는 이 과학 실행을 다시 호출하지 않는다.
- 종료 exit0가 확인돼야 기존 수집 도구로126worker/dispatcher 실제exit0·owner부재·선택/원본/archive/manifest SHA를 확인하고 기존의 아직 없는 retrieval 경로에 최초 회수한다. 파일 전송은 기존900초 timeout이며 종료 확인45초와 구분한다.
- 회수 뒤 기존 비교기를 local-only check-only→새5seed 보고 순서로 수행한다. 완료 캐시는 같은 원본과 결과 SHA를 로컬 재검증하며 SSH0이다.

**검증된 진입점을 실행하고 후속 자동화를 갱신한다 — 완료**
- paper_research 저장소의 codex/contextual-retrieval-pilot-20261007 브랜치에 코드 commit727840c4737a773246aa0879fd2fee578a04cbc5를 만들었다. CONTROL_BINDINGS rawSHA는7b59a7fa0dc810c2aa35c2004d69d094a1c4775749936b0442bb4f0918fbf168, AUTHORIZATION rawSHA는fbf6befa8e0b39fe19286b564bdc81dabfb1e87580843e329d61a8c3dc95c9d2다. local verify는 실제exit0·remote0이다.
- 최초 advance는2026-10-21 07:56:00.050136KST에 waiting·실제transportexit0·timeoutfalse로 끝났다. 원dispatcher616624 신원이 일치했고 stdout1,372bytes/stderr0의 원문과SHA를 보존했다. FIRST_CHECK.json과 당시 ROUND/TRANSPORT에 연결했다. 이는 전체 종료나GPU유휴 확인이 아니다.
- 기존 자동화2seed만 ACTIVE·15분 간격·동일 채팅을 유지한 채 v2 advance로 갱신했다. 저장된 prompt가 요청문과 정확히 같고 id/name/kind/간격/대상/생성시각이 유지됐음을 확인했다. AUTOMATION_BEFORE/AFTER.json에 기록했다. Notion의 '7 남은 실험과 서버 분할 계획' 해당 항목만 수정하고 읽어 확인했다.
- COLLECTION_COMPLETE는 회수 검증, COMPLETE는5seed 집계까지의 완료다. 어느 것도 최초 waiting이나 과거 READY로 대신하지 않는다. 최초 RAF/Taxi 학습 완료를 전체 Test 완료로 표현하지 않는다.

**남은 검증과 회수 — 외부 작업 대기**
- 기존5080 학습·Validation 전체 재현·고정Test 종료를 다음 정규 회차에서 짧게 확인한다.
- 실제exit0와 전체 종료 증거가 확인되면 기존 collector로 한 번 회수하고5seed Validation/Test를 별도 집계한다. 현재 retrieval/COMPLETE는 아직 없다. 5090 작업과 기존 과학 실행은 그대로 둔다.
