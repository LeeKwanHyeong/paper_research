# 5080 Cap 실행 사용자 중지 — 완료

사용자의 중지 요청에 따라 2026-10-20 13:33:17 KST에 Cap dispatcher 504213과 남은 학습 worker 560475에 SIGTERM을 보냈다. dispatcher의 실제 OS 종료값은 기존 supervisor가 -15로 기록했다. worker와 wrapper 자체의 OS 종료값은 직접 관측하지 않았다. 기존 연구 실패로 분류하지 않는다.

2026-10-20 13:34:07.623 KST 읽기 전용 재확인에서 dispatcher 504213, worker 560475, supervisor 504203, pane 504202의 /proc 항목과 해당 tmux 세션이 없었다. 로컬 전용 finisher 16494도 시작시각·명령행을 확인하고 종료했다.

기존 중단 외부 모델 PID 351034/351048/351049/353905의 신원과 T 상태는 보존됐다. GPU에는 중단 외부 모델 351049/351048/353905만 남았으며 메모리 여유는 11,257 MiB였다. 전역 GPU가 비어 있다는 뜻은 아니다. 5090 및 다른 finisher는 접근·변경하지 않았다.

checkpoint 35개와 receipt 35개의 SHA를 POST_STOP_VERIFIED.json에 보존했다. 원본 파일을 삭제하거나 덮어쓰지 않았다. 로컬 중지 범위·PID 재사용 차단 등 8개 회귀 검사가 통과했다.

첫 종료 직후 조회에는 이미 종료된 worker의 GPU 문맥이 잠시 남아 RESULT.json에 action_required와 실제 transport exit 1을 보존했다. 종료 동작을 재실행하지 않고, 추가 신호 0회의 읽기 전용 후속 확인으로 GPU 정리를 확인했다. 후속 transport exit는 0이다.

완료 증적: FINAL_RECEIPT.json 및 POST_STOP_VERIFIED.json. 다음 작업은 부모 세션의 RAF·Taxi 평가 준비와 실행 직전 fresh GPU 검사다.
