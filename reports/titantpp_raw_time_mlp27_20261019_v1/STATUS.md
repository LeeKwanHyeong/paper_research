# 원시간 입력 대조27 — 현재 상태

**대조 구현·로컬 검증 — 완료**

사용자는 세 수량 손실 각각에 대응하는 신규27개 학습을 승인했다.
Cap 세 데이터셋·MLP16·seed42/52/62·시간 가중치0과 수량 log1p 입력을 유지하고,
encoder의 관측 시간 입력만 원래 간격으로 바꾼다. 추가 정규화·스케일 조절은 없다.
시간 head와 시간 정답은 기존대로 둔다.
로컬 통합70tests와9subtests가 통과했고, 별도 source955 묶음의 CPU 합성 검사도 통과했다.
기존 source952는 같은 바이트로 보존했다. 실제 native 검증은 부모 완료 후 새 경로에서 수행한다.

**현재 손실 비교 종료 후 실행 — 대기**

실행 중인 quantity_loss_mlp18 v2는 변경하지 않았다.
양 서버 학습·Validation·회수·최종27개 집계와 SHA 검증이 끝난 후에만 후속 작업이 시작된다.
후속 학습은5080에12개,5090에15개이며 각 서버 최대3병렬이다.
준비 완료와 실제 학습 시작은 구분해 기록한다. 현재 이 문서 작성 시 신규 원시간 학습은 시작하지 않았다.
연결 작업의 시작 증적은 FOLLOWUP_STARTED.json, 실제 후속 dispatcher 관측은 START_RECEIPT.json,
최종 성공은 followup_v1/complete.json으로 구분한다. 이 문서를 새로운 현재 관측으로 재사용하지 않는다.

**최종 비교 — 다음 작업**

동일 손실·dataset·seed·host의 저장 로그 시간27개와 신규 원시간27개,
총54개를 Validation RMSE·MAE와 seed별 차이로 비교한다. 새 Test는 없다.
자세한 조건은 [실험 계획](EXPERIMENT_PLAN.md)과 [고정 명세](SCIENCE_SPEC.json)에 기록한다.
