# Instacart·A100 정기관측

작성시각: 2026-10-09T03:03:22.780586+09:00. 아래 시각은 실제 관측 원본을 유지한다.

|주제|상태|완료/실패/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|4/0/2/3/0|
|CNN+GRU54 이력 보정27|진행 중|4/0/2/21/0|
|고정 검색27|외부 작업 대기|0/0/0/27/0|

|GPU|실제 관측시각|조건|저장/best|best 미갱신 잔여 / E300 시나리오|
|---|---|---|---|---|
|5080|2026-10-09 03:01:23 KST|Instacart CNN+GRU54 seed42|E34/E30|8.08h / 59.72h|
|5090|2026-10-09 03:01:23 KST|Instacart CNN+GRU54 seed52|E42/E15|2.91h / 57.74h|
|A100|2026-10-09 02:52:03 KST|Taxi 이력 보정 seed52|E22/E11|1.30h / 12.50h|
|A100|2026-10-09 02:52:03 KST|Taxi 보정 없음 seed52|E73/E68|0.32h / 2.10h|

Taxi seed52 현재 상태 전용은 E59/best19로 완료됐다. 최대3개 병렬 중 이 대조가 먼저 끝나 실제 활성 GPU PID는23051(full)·23057(보정 없음)의2개다. supervisor1253, timeout23050/23056, GPU UUID와 argv·PPID·lease·checkpoint가 관측시 일치한다. 같은 batch의 나머지 두 조건 종료 후 다음 batch로 넘어간다.

ETA는 각 fit의 실제 완료 epoch 최근 최대10개 시간 중앙값과 best 미갱신·속도 유지 가정이다. 전체 큐 ETA 및 $50 내 전체 완료 여부는 미확정이다. endpoint·회수·CPU 감사는 별도다. Instacart E300 시나리오는 남은 조건별 제한시간을 초과하므로 승인된 완료 예측이 아니다.

## 완료된 A100 조건의 같은 선택 epoch Validation

|조건|저장/best|전체 RMSE / MAE / TimeNLL|큰수량 RMSE / MAE / TimeNLL|fit 관측 wall시간|
|---|---|---|---|---|
|Taxi 이력 보정 seed42|E166/E126|78.3198 / 25.2315 / 1.36598|370.0794 / 276.9118 / 9.05606e-123|7.92h|
|Taxi 보정 없음 seed42|E181/E141|87.9050 / 28.4746 / 1.0233|392.4853 / 311.4418 / 1.2641e-15|3.22h|
|Taxi 현재 상태 전용 seed42|E122/E82|86.5854 / 28.0762 / 1.07993|418.8414 / 328.7055 / 5.93166e-53|2.53h|
|Taxi 현재 상태 전용 seed52|E59/E19|89.3333 / 28.5679 / 0.6649|357.0626 / 278.9636 / 9.1469e-33|0.91h|

전체8268·큰수량79(>3449)의 fullValidation이며, 최초 strict finite 최소 전체 raw 수량RMSE epoch의 여섯 지표다. Taxi42 full의 수량 개선과 전체 TimeNLL 손해는 이전 결과와 같다. Taxi52는 세 조건 중 하나만 완료됐으므로 최종 비교를 하지 않는다. 세seed 평균·표본SD 및 Q1은 아직 판정할 수 없다. 동시실행 wall시간은 단독 구조효율로 해석하지 않는다.

비용 manager heartbeat: 2026-10-09 03:03:13 KST. 누적 추정 $15.20, 총 상한$50(준비·학습45+회수·정리5) 유지. 확정 청구액은 아니다.
Terminal service heartbeat: 2026-10-09 02:53:46 KST. 10분 간격 관측·미검증 live backup 유지. 최종 binary 원본 archive/manifest 검증은 미완료다.
기존 작업마감 2026-10-09 21:42:46 KST / compute stop 2026-10-09 22:42:46 KST 유지. 마감은 ETA가 아니다.

Instacart 도우미1회·각 host 읽기 SSH1회, finalization local pending/remote0. A100은 15분 이내 원본 SHA 검증 관측을 재사용해 추가SSH0. 실패·중단·정리·최종완료 기록은 감사시 없었다. 새 실행·retry·resume·계약/Runtime/인증/Pod/자동화 변경은 하지 않았다.

**학습과 원본 보존 — 진행 중**
- 현 비용·소유권 계약 안에서 Instacart9와 A10027을 관측한다.

**이력 보정 기여의 재현성 판정 — 다음 작업**
- 세seed 결과가 모이면 전체·큰수량 여섯 지표의 평균·표본SD와 손해 조건을 함께 보고한다.

**고정 검색 실행조건 정리 — 외부 작업 대기**
- A100 full9를 RTX 같은GPU mean으로 혼합하지 않는다. 후속 자원·기준선 계약 정리 전 자동 시작하지 않는다.

CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다. Validation 개발 증거이며 통계 유의성·외부 우월성·새 독립 평가의 증거로 확대하지 않는다.

근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1/hourly_monitor/20261009_025203KST.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5080_1791482483133615000.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791482483483385000.json`
