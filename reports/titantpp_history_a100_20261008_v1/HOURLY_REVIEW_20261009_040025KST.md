# Instacart·A100 정기관측

작성: 2026-10-09T04:00:25.283143+09:00. 실제 관측시각은 각 원본 시각을 유지한다.

|주제|상태|완료/실패/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|4/0/2/3/0|
|이력 보정27|진행 중|5/0/1/21/0|
|고정 검색27|외부 작업 대기|0/0/0/27/0|

|GPU·실제 KST 관측시각|조건|저장/best|best 미갱신 잔여 / E300 시나리오|
|---|---|---|---|
|5080 · 2026-10-09 03:58:43 KST|Instacart CNN+GRU54 seed42|E38/E30|7.18h / 58.82h|
|5090 · 2026-10-09 03:58:43 KST|Instacart CNN+GRU54 seed52|E47/E15|1.79h / 56.74h|
|A100 · 2026-10-09 03:49:24 KST|yellow_trip_hourly 이력 보정 seed52|E32/E29|1.55h / 11.25h|

A100은 최대3개 동시 실행이며 현재 활성 조건은 1개다. 먼저 끝난 대조는 재실행하지 않고 같은 dataset/seed의 남은 조건 종료를 기다린다. 관측 GPU PID: `GPU-fb877d58-ff1d-9b4b-f543-c5fc743f58de, 23051`. 정확 argv·PID·PPID·UUID·lease·checkpoint receipt 검증이 통과했다.

ETA는 각 fit의 실제 완료 epoch 최근 최대10개 시간 중앙값과 best 미갱신·속도 유지 가정이다. 전체 큐 ETA 및 $50 내 전체 완료 여부는 미확정이다. endpoint·회수·CPU 감사는 별도다. Instacart의 E300 시나리오는 남은 조건별 시간 한도를 초과하므로 승인된 완료 예측이 아니다.

## 완료 조건의 같은 선택 epoch Validation

|조건|저장/best|전체 RMSE / MAE / TimeNLL|큰수량 RMSE / MAE / TimeNLL|fit 관측 wall시간|
|---|---|---|---|---|
|yellow_trip_hourly 이력 보정 seed42|E166/E126|78.3198 / 25.2315 / 1.36598|370.0794 / 276.9118 / 9.05606e-123|7.92h|
|yellow_trip_hourly 보정 없음 seed42|E181/E141|87.9050 / 28.4746 / 1.0233|392.4853 / 311.4418 / 1.2641e-15|3.22h|
|yellow_trip_hourly 현재 상태 전용 seed42|E122/E82|86.5854 / 28.0762 / 1.07993|418.8414 / 328.7055 / 5.93166e-53|2.53h|
|yellow_trip_hourly 보정 없음 seed52|E108/E68|86.0713 / 27.7408 / 0.82572|412.8328 / 316.8310 / 1.48889e-196|1.56h|
|yellow_trip_hourly 현재 상태 전용 seed52|E59/E19|89.3333 / 28.5679 / 0.6649|357.0626 / 278.9636 / 9.1469e-33|0.91h|

현재 완료된 Taxi 조건은 전체8268·큰수량79(>3449)의 fullValidation이다. 최초 strict finite 최소 전체 raw 수량RMSE epoch에서 여섯 지표를 함께 읽었다. Taxi42 full은 보정 없음 대비 전체/큰수량RMSE 10.90%/5.71% 개선, 전체 TimeNLL 33.49% 악화했다. 현재 상태 전용 대비 RMSE는9.55%/11.64% 개선, 전체 TimeNLL은26.49% 악화했다. Taxi52 full은 아직 진행 중이므로 seed52 최종 대조를 판정하지 않는다. 세seed 평균·표본SD 및 Q1은 미확정이다. 동시 실행 wall시간은 단독 구조효율로 해석하지 않는다.

## 비용과 보존

- 비용 manager heartbeat 2026-10-09 04:00:13 KST. 누적 추정 $16.72; 확정 청구액은 아니다. 총$50(준비·학습45+회수·정리5) 상한 유지.
- Terminal service heartbeat 2026-10-09 04:00:01 KST. 10분 간격 관측과 미검증 live backup 유지. 감사 당시 실패·중단·정리·최종완료 파일 없음.
- 기존 작업마감 2026-10-09 21:42:46 KST / compute stop 2026-10-09 22:42:46 KST 유지. 마감은 ETA가 아니다.
- Instacart 도우미1회·각 host SSH1회, finalization local pending/remote0. A100은15분 이내 SHA 검증 관측 재사용으로 추가SSH0.
- 신규학습·retry·resume·Pod 생성·예산·공용 Runtime·인증·원본·다른 자동화 변경0. 현 자동화 유지.

**학습과 원본 보존 — 진행 중**
- Instacart9와 A100 이력 보정27을 승인된 소유권·비용 안에서 관측한다.

**이력 보정의 재현성 판정 — 다음 작업**
- 데이터셋별 세seed가 모이면 여섯 지표의 평균·표본SD와 불리한 결과를 포함해 Q1을 판정한다.

**고정 검색의 공통 실행조건 정리 — 외부 작업 대기**
- A100 full9를 RTX 같은GPU mean으로 혼합하지 않는다. 후속 자원·기준선 계약 정리 전 자동 시작하지 않는다.

최종 binary archive/manifest 검증·CPU binary 재추론 감사·Test·새 독립 평가는 미완료다. Validation 개발 증거이며 외부 우월성·통계 유의성·새 독립 평가를 입증하지 않는다.

근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1/hourly_monitor/20261009_034924KST.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5080_1791485923383663000.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791485924019868000.json`
