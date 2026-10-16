# CNN+GRU54 매시간 관측

회차:2026-10-06 11:43:40 KST heartbeat. 실제 관측시각은 아래와 같이 별도로 보존합니다.

**원래 CNN+GRU54 3seed — 진행 중**

신규6 중5완료·1진행, 재사용seed42 원본3완료로 canonical9 중8완료·1진행입니다. 학습 실패·대기·미확정0. 시간 출력부 단계는 별도 집계합니다.

| 서버 | 실제 관측(KST) | 완료/실패/진행/대기/미확정 | 현재 조건 | 저장/best epoch | 조건부 학습 종료 예상 |
|---|---|---|---|---|---|
|5080|10/06 04:37:49|4/0/0/0/0|Taxi·RAF 신규4조건 종료|조건별 아래|종료 검증·보존 terminal 재사용|
|5090|10/06 11:45:04|1/0/1/0/0|Intermittent CNN+GRU54 seed62|40/13|E53, 10/06 13:30:00 KST (약1.75시간)|

5090 E13 best-so-far 전체 full Validation: RMSE1.744248351, MAE0.712498927, TimeNLL0.322681223. 최근10 완료epoch 중앙값484.343111초, 조건부 E53까지13epoch. best 미갱신·현 단독worker 속도 유지 가정입니다. best 갱신·속도 변화 시 예상도 바뀝니다.
300epoch 별도 시나리오는 약34.98시간, 2026-10-07 22:43:53 KST입니다. 종료 보장/예측구간이 아니며 endpoint replay·회수·CPU 감사 시간은 제외됩니다.

5090 source/operation SHA·GPU UUID·supervisor/worker timeout argv와 PPID 연결·GPU PID2983365·저장 checkpoint/receipt/history/timing이 일치합니다. 서버에 새 학습·재시작·retry·평가를 시작하지 않았습니다. native36h/168h와 저장된 lease를 변경하지 않았습니다.

| 신규 조건 | 상태 | 저장/best epoch | 같은 best epoch RMSE | MAE | TimeNLL |
|---|---|---|---:|---:|---:|
|Taxi seed52|complete|197/157|76.846316|24.595004|2.163854|
|Taxi seed62|complete|94/54|82.022244|26.012447|0.875252|
|RAF seed52|complete|45/5|34.440158|9.136572|3.479681|
|RAF seed62|complete|60/20|33.983366|9.197878|3.611548|
|Intermittent seed52|complete|51/11|1.676304|0.688762|0.287827|
|Intermittent seed62|running|40/13|1.744248|0.712499|0.322681|

**5080 bias2 시간 출력부 — 완료**

실제 terminal 관측:10/06 09:57:54 KST. 6완료/0실패/0진행/0대기/0미확정. 원본119개/archiveSHA 회수와 소유 CPU/GPU PID 부재·exit0 검증됨. monitor SSH0/cache재사용, finalizer already_finalized. 원래9조건에 중복 합산하지 않습니다.

| 원본 조건 | 원본 selected epoch | 새 head 저장/best epoch | 선택 epoch Validation 전체 TimeNLL | 큰 수량 TimeNLL |
|---|---:|---:|---:|---:|
|RAF seed42|9|18/8|3.51066227|3.47733445|
|RAF seed52|5|14/4|3.43986874|3.24387642|
|RAF seed62|20|10/0|3.61154773|3.7055267|
|Taxi seed42|126|10/0|1.36597862|9.05631951e-123|
|Taxi seed52|157|10/0|2.16385431|4.10181824e-19|
|Taxi seed62|54|10/0|0.875251769|2.80380963e-32|

**5080 full130 시간 출력부 — 완료**

실제 terminal 관측:10/06 10:22:28 KST. 6완료/0실패/0진행/0대기/0미확정. 원본119개/archiveSHA 회수와 소유 CPU/GPU PID 부재·exit0 검증됨. monitor SSH0/cache재사용, finalizer already_finalized. 원래9조건에 중복 합산하지 않습니다.

| 원본 조건 | 원본 selected epoch | 새 head 저장/best epoch | 선택 epoch Validation 전체 TimeNLL | 큰 수량 TimeNLL |
|---|---:|---:|---:|---:|
|RAF seed42|9|11/1|3.50558773|3.26997276|
|RAF seed52|5|12/2|3.46070384|3.29741411|
|RAF seed62|20|17/7|3.56346182|3.4916749|
|Taxi seed42|126|10/0|1.36597862|9.05631951e-123|
|Taxi seed52|157|10/0|2.16385431|4.10181824e-19|
|Taxi seed62|54|10/0|0.875251769|2.80380963e-32|

두 head 단계 모두 Taxi3seed의 E0가 유지됩니다. full130 RAF3seed 전체 TimeNLL평균은3.546413→3.509918(1.0291%감소)입니다. 수량 예측 byteSHA는 전체 Train/Validation에서 그대로입니다. 큰 수량 경계는 원래 Train Taxi3449·RAF200초과입니다.

**Intermittent 최종 연결·원본 회수 — 외부 작업 대기**

원래 finalizer는 pending_verified_terminals, 진단 후속 연결은 pending_original_terminal_and_retrieval입니다. 5090 seed62가 미완료이므로 원래 새6조건 archive 회수와 원래9조건3seed 최종 요약은 시작하지 않았습니다. 세 stage 및 진단 링크의 종료gate가 모두 완료될 때까지 titantpp-gru-3seed 자동화를 유지합니다.

**평가 범위와 남은 작업**

- 이번 회차: Train/Validation-only. 신규 Test·예측 행·Validation/Test혼합 성과를 열지 않았습니다. head재적합은 원래 Train을 본 encoder의 사후 개발 선택이며 독립 calibration/독립 Test가 아닙니다.
- 다음:5090 완료 후 기존 승인된 selected/last fullValidation·원본manifest/archiveSHA 회수·실제 서버종료를 검증해 원래9조건 및 Intermittent 최종 링크를 완성합니다.
- 별도 미완료:Test3seed·Intermittent head재적합·S2P2 동일재적합·CPU binary 재추론 감사·새 독립 평가. 전체 연구 감사나 외부 모델 우월성을 확정하지 않습니다.
- 문서 보존 메모:bias2 README의 full130 “다음 작업/미실행”은 당시 종료 시점의 계획이며 현재 full130 완료증거(10:22:28 KST)와 구분합니다. 이번 읽기 전용 회차에서는 과거 문서를 수정하지 않았습니다.
