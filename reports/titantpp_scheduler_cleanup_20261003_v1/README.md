# TitanTPP 스케줄러 정리 — 2026-10-03

사용자 요청에 따라 기존 titantpp-36을 5090의 잔여 학습 관측으로 축소한다. 새 자동화를 만들지 않고 기존1시간 주기와 알림 의도를 유지한다. 이전 설정은 previous_automation.json과 previous_scheduler.md에 보존했다.

- A100 6/6 및 PRO4500 3/3은 terminal·소형기록·원본 SHA회수·비용보고·Pod 삭제 확인 완료. RunPod 반복 관측 종료.
- 5080은24/24 완료 증거를 재사용한다.
- 5090은2026-10-03 07:08:25 KST 관측 기준8/9 완료. 마지막 Instacart Deep Renewal seed62, 저장38/best26, best 미갱신 가정 학습 종료08:32:35 KST. endpoint 재평가·회수·CPU 감사는 별도.
- 새 학습·재시작·Pod 또는 서버프로세스 변경은 하지 않았다. 학습 완료와 binary CPU 감사 완료를 구분한다. A100 Intermittent2조건 및 PRO4500 Deep Renewal1조건 CPU 감사가 별도로 남아 있다.
- 5090 종료 확인 후36조건 및A1006조건의 최종 관측보고와 보존된RunPod 회수·비용·삭제 증거를 연결하고 이자동화만 삭제한다.
