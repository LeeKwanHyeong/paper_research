# 5080 결과 회수 연결 복구

**기존 학습과 원본을 보존한다 — 완료**
- 사용자는 학습을 그대로 두고 종료 대기 연결만 복구한 뒤 Validation/Test 확인과 결과 회수까지 완료하도록 승인했다. AUTHORIZATION.json에 원문과 기존 실패·대기 기록 여섯 파일의 SHA를 보존했다.
- 기존 SSH는 `Connection reset by peer` / `Broken pipe`, 실제 transport exit255로 끊겼다. 같은 학습 dispatcher는 후속 관측에서 계속 실행 중이었다. 네트워크 연결 종료의 더 구체적인 원인은 미확인이다.
- 기존 로컬 대기 PID35139는 이번 복구 전 읽기 전용 ps 확인에서 부재했다. 기존 프로세스 자신의 OS exit를 직접 관측했다는 뜻은 아니다. 5090 작업은 변경하지 않는다.

**새 제어 경로를 검증한다 — 완료**
- 신규 합성 검사26개와 독립검토, 기존5seed 비교기16개 검사를 통과했다. 검사 전후 기존 소스·실패기록993개 SHA는 동일하다. 실제 연결 성공은 별도 READY 증거로 판단한다.
- 기존 봉인된 학습·운영 source987과 계약은 그대로 두고, 새 recover_titantpp_shared_external_extra42 모듈만 사용한다. 복구 제어·로그는 recovery_v1 아래에 보존한다.
- 단 한 번의 SSH 연결에서 기존 contract와 dispatcher 신원을 확인하고 READY를 출력한 뒤 원래 절대 기한까지 종료를 기다린다. SSH keepalive는 연결 생존 확인이며 자동 재접속이 아니다. 학습·평가·resume·kill 명령은 호출하지 않는다.
- 기존 FINISH_STARTED/QUEUE_FAILURE/로그는 삭제하거나 덮어쓰지 않는다. 새 복구 attempt도 한 번만 허용한다. 추가 실패나 불완전 회수는 그대로 보존하고 자동 재시도하지 않는다.

**종료 검증과 결과 회수를 이어간다 — 학습 종료 대기**
- 원래 dispatcher가 Train42→Validation42 전체 재현 gate→고정 Test42를 수행한다. 새 연결은 이 순서나 선택 checkpoint를 변경하지 않는다.
- 원래 수집 도구로 126 scientific worker와 dispatcher 실제 exit0, 소유 PID/GPU 부재, 기존 중단 작업 보존, 선택·원본·archive·manifest SHA를 검증한 뒤 아직 없는 회수 경로를 최초 생성한다.
- 검증된 원본 회수 후 기존 비교 도구를 check-only로 확인하고 별도 five_seed_aggregation_v1에 기존3seed+신규2seed 결과를 기록한다. 과거3seed 결과는 수정하지 않는다.

**완료 증거를 확인한다 — 다음 작업**
- STARTED는 연결 요청, 새 wait.stdout의 READY는 해당 시각의 실제 연결·신원 검증이다. 어느 것도 학습/평가 완료가 아니다.
- COLLECTION_COMPLETE는 원본 회수 검증 완료, COMPLETE는 추가로 로컬5seed 비교 완료다. FAILURE에는 단계와 원문을 보존한다. 기존 실패 기록과 새 해결 기록은 함께 남긴다.
