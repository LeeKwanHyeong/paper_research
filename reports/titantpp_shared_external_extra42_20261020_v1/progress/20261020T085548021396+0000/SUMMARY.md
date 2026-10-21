# 추가 2seed 관측 — 2026-10-20 17:55:48 KST

학습완료21(RAF14·Taxi7), 진행3(Taxi), 대기18(Taxi4·Intermittent14), 미확정0·과학실패0이다. native Validation/Test는 각각0이며 최종 원본 회수 전이다.

종료 대기 SSH가17:55:13 KST actual transport exit255로 끊겼다. stderr 원문은 `Connection reset by peer`와 `Broken pipe`다. 이후17:55:48 KST 한 번의 상태 조회는 actualexit0였고 실제 dispatcher/worker/GPU 신원이 확인됐다. 학습 실패와 종료 대기·자동 회수 연결 실패를 구분한다. finisher 자신의 OS exit는 직접 관측하지 않았다.

| 현재 Taxi 학습 | 저장/best epoch |
|---|---:|
| NHP seed82 |47/35|
| SAHP seed72 |96/94|
| SAHP seed82 |31/28|

자동 재시도하지 않았다. 기존 학습·완료 결과·checkpoint·실패 원문은 보존한다. 종료 대기·검증·회수 연결을 새 제어 경로에서 복구하려면 사용자 승인이 필요하다. 같은 미해결 실패 중 원격 작업을 반복하지 않는다. 기존 PAPER_START_GATE PASS와 논문 연구자료 준비 완료 상태는 유지되며 재생성하지 않았다.
