# 5090 Intermittent4 복구와 Validation 회수

**현재 원본 보존 — 완료**

- 원격 원본1126파일과 완료 RAF6, 중단3개 전체 epoch checkpoint를 v1 origin_archive에 보존했다. 원본 archive와 모든 파일 SHA를 검증했다. 원 중단 worker의 OS 종료 코드는 미확정 그대로 둔다.
- 2026-10-19 17:24:09 KST 재부팅을 확인했다. 재부팅의 세부 유발 원인은 일반 계정 로그로 확인되지 않았다.
- 복구 v1 native 사전 검사는 빈 sample_data 폴더 누락으로 import 단계에서 실패했다. 실제 학습0이며 실패 원문·계약·코드·원격 경로를 그대로 둔다.

**복구 배포 구성 수정 — 진행 중**

- paper_research 저장소 codex/contextual-retrieval-pilot-20261007 브랜치에 별도 v2 manager와 배포 환경 회귀 검사를 추가한다. 실제 tar 추출 뒤 동결된 경로 탐색 함수가 새 source를 찾는지 확인한다.
- 기존 과학 source963, 입력6, runtime, worker와 공통 조건은 동일 SHA를 유지한다. 배포본에는 빈 source/sample_data만 추가한다.
- 독립 검토와 신규 파일 커밋 후, 별도 5090 recovery_v2 경로에서 native GPU·Runtime·3동시 합성 검사를 한다. v1 경로는 재시작하지 않는다.

**미완료 Intermittent4만 처리 — 다음 작업**

- seed52 관측 전달: epoch15 다음부터 전체 상태 재개.
- seed52 용량 대조: epoch13 다음부터 전체 상태 재개.
- seed62 관측 전달: epoch13 다음부터 전체 상태 재개.
- seed62 용량 대조: 미시작이므로 fresh fit. 최대3병렬과 기존 기한을 유지한다.
- 모델·optimizer·난수·loader 상태를 복원한다. 기존 checkpoint와 선택 기준, head·손실·입력은 변경하지 않는다.

**종료 후 Validation과 원본 회수 — 다음 작업**

- 위 학습4개 성공 뒤, 고정 checkpoint로 Intermittent4와 완료 RAF6의 native Validation10을 실행한다. RAF 학습·기존 CNN/MLP16 기준선 재평가·Test는 하지 않는다.
- 실제 worker/dispatcher exit0와 원본 불변·새 checkpoint·재현 결과·소유 프로세스 부재를 확인한 뒤 새 exclusive retrieval 경로에 회수한다. archive·manifest·모든 파일 SHA를 로컬에서 다시 검증한다.
- 별도 반복 자동화는 만들지 않는다. 이번 4개→Validation10→회수로 끝나는 유한한 완료 대기 절차만 연결한다. wrapper 자신의 OS 종료 관측과 직접 확인한 child 종료를 구분한다.
- 실패·재부팅·미확정 종료가 발생하면 원문을 보존하고 자동 재시작하지 않는다. 5080 작업과 기존 외부 비교군 중단 작업은 변경하지 않는다.
