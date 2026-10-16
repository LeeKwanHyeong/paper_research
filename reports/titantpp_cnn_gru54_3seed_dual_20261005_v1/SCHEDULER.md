# CNN+GRU54 3seed 매시간 관측

기존 자동화 `titantpp-gru-3seed`를 새6조건 관측으로 갱신해 ACTIVE 상태로 확인했습니다. 이름은 “TitanTPP CNN+GRU54 3seed”이며 기존 매시간 주기를 유지합니다. 기록은 scheduler_receipt.json, 실제 학습 진입은 launch_confirmation.json을 참조합니다.

두 서버에서 승인된 monitor.py를 각각 한 번만 호출하고, 이후 finalize_once.py의 terminal gate를 확인합니다. 미완료이면 로컬 pending만 남기며 추가 평가·학습·원격 회수를 하지 않습니다. 종료된 이전33조건 및 삭제된 A100은 다시 관측하지 않습니다. 원래 신규6조건과 재사용3조건의 Validation·원본SHA 회수·실제 서버 종료 및 추가5080 시간 출력부6조건의 원본SHA회수·종료 확인이 모두 끝난 뒤 이 자동화만 종료합니다. Test3seed·새 독립 평가·CPU 재추론 감사는 별도 후속입니다.

2026-10-06 사용자 승인으로 5080 수량 동결·시간 bias2개 재적합6조건을 같은 자동화의 별도 집계로 연결했습니다. 새 head는 완료·SHA회수·서버종료를09:57:54 KST에 확인했으므로 terminal cache를 재사용하고 반복SSH를 하지 않습니다. 원래5090 Intermittent 조건의 관측/회수 gate는 유지합니다. 상세: reports/titantpp_time_head_bias_refit_5080_20261006_v1/SCHEDULER.md.

**통합 관측 종료 — 완료 / 2026-10-06 13:53:09 KST**

원래CNN+GRU54 신규6+재사용3조건, bias2/full130 각각6조건의 scientific terminal·원본archiveSHA·실제서버종료와 Intermittent 최종Validation 연결을 검증했습니다. 자동화 `titantpp-gru-3seed`만 앱 도구로 삭제 완료했고 다른 자동화와 서버작업은 변경하지 않았습니다. 최종 보고서: [회차 종료 증적](hourly_monitor/20261006T044538Z/README.md). Test3seed·CPU binary 재추론 감사·새 독립평가는 별도 미완료입니다.
