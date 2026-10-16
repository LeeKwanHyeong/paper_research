# 63조건 시간별 관측 — 2026-10-08 16:30:51 KST

|주제|완료 여부|
|---|---|
|Instacart 9조건|진행 중: 완료2 / 실패0 / 진행2 / 대기5 / 미확정0|
|CNN+GRU54 이력 보정27조건|대기: Instacart 전체 원본·종료·최종 Validation 확인 후|
|고정 검색27조건|대기: 이력 보정 완료 확인 후|

|GPU|실제 관측시각|완료/실패/진행/대기/미확정|진행 조건|저장 / best|조건부 잔여 / E300|
|---|---|---|---|---|---|
|5080|2026-10-08 16:30:51 KST|1/0/1/4/0|Instacart / MLP12 / seed42|E26 / E6|1.34h / 18.29h|
|5090|2026-10-08 16:30:51 KST|1/0/1/1/0|Instacart / MLP12 / seed52|E41 / E15|0.93h / 17.13h|

ETA는 해당 fit 최근10완료 epoch 중앙값으로 계산했다. best 미갱신·속도 유지 가정이며, 전체큐 ETA는 미확정이다. endpoint/원본 회수/CPU 감사는 별도다.

|GPU·모델·seed|상태|저장/best|전체 RMSE/MAE/TimeNLL|큰수량 RMSE/MAE/TimeNLL|
|---|---|---|---|---|
|5080 / MLP8 / 42|완료 terminal|70/30|5.872632 / 3.988631 / 2.804357|24.284939 / 21.576828 / 2.810068|
|5080 / MLP12 / 42|진행 중 best-so-far|26/6|5.882473 / 3.986131 / 2.803413|현 관측 미산출|
|5090 / MLP8 / 52|완료 terminal|55/15|5.884102 / 3.994055 / 2.811975|24.665215 / 22.087655 / 2.899737|
|5090 / MLP12 / 52|진행 중 best-so-far|41/15|5.868342 / 3.991334 / 2.812495|현 관측 미산출|

현재 3seed·동일 구조 비교가 미완료이므로 수량 우위나 시간 손해의 최종 결론은 보류한다. TimeNLL은 수량 선택 epoch의 값이며, 다른 GPU의 실행 시간은 합치지 않는다. 큰 수량 경계는 >35, Validation503733/큰수량6036이다.

로컬 봉인 검증 PASS: source126/129/129, input3/7/7, operation각1, project control각6. permit의 start·approval는 canonical SHA, qualification은 raw SHA로 정의에 맞춰 확인했다.
양쪽 실제 supervisor/timeout/worker argv·PID·PPID, GPU UUID/PID, lease/claim, checkpoint rawSHA·저장 epoch 검증 통과. advance 도우미1회, live 관측SSH 각1회; activation already_launched로 원격0, finalization pending으로 원격0.
Instacart 전체 scientific terminal과 실제 종료가 아직 충족되지 않아 원본 최종 회수·후속 stage 실행은 대기한다. 자동화 유지. CPU binary 재추론 감사·Test·새 독립 평가 미완료. 원고는 이번 질문 검토에서 수정하지 않았다.
