# Instacart·A100 정기관측

작성: 2026-10-09T12:08:54.607801+09:00. 실제 관측시각은 각 원본 시각을 유지한다.

|주제|상태|완료/실패/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|6/0/1/2/0|
|이력 보정27|진행 중|9/0/3/15/0|
|고정 검색27|외부 작업 대기|0/0/0/27/0|

|GPU·실제 KST 관측시각|조건|저장/best|best 미갱신 잔여 / E300 시나리오|
|---|---|---|---|
|5080 · 2026-10-09 12:08:29 KST|Instacart MLP8 seed62|E16/E1|1.66h / 18.91h|
|5090 · 2026-10-09 06:05:11 KST|소유 조건 3개 완료, supervisor exit0·소유/GPU PID 없음|CNN+GRU54 seed52 E55/best15|완료|
|A100 · 2026-10-09 12:00:40 KST|raf_spare_parts 이력 보정 seed42|E11/E9|0.46h / 3.50h|
|A100 · 2026-10-09 12:00:40 KST|raf_spare_parts 보정 없음 seed42|E24/E9|0.14h / 1.59h|
|A100 · 2026-10-09 12:00:40 KST|raf_spare_parts 현재 상태 전용 seed42|E23/E9|0.16h / 1.69h|

A100은 최대3개 동시 실행이며 현재 활성 조건은 3개다. 먼저 끝난 대조는 재실행하지 않고 같은 dataset/seed의 남은 조건 종료를 기다린다. 관측 GPU PID: `GPU-fb877d58-ff1d-9b4b-f543-c5fc743f58de, 52409
GPU-fb877d58-ff1d-9b4b-f543-c5fc743f58de, 52415
GPU-fb877d58-ff1d-9b4b-f543-c5fc743f58de, 52421`. 정확 argv·PID·PPID·UUID·lease·checkpoint receipt 검증이 통과했다.

ETA는 각 fit의 실제 완료 epoch 최근 최대10개 시간 중앙값과 best 미갱신·속도 유지 가정이다. 전체 큐 ETA 및 $50 내 전체 완료 여부는 미확정이다. endpoint·회수·CPU 감사는 별도다.

## 완료 조건의 같은 선택 epoch Validation

|조건|저장/best|전체 RMSE / MAE / TimeNLL|큰수량 RMSE / MAE / TimeNLL|fit 관측 wall시간|
|---|---|---|---|---|
|yellow_trip_hourly 이력 보정 seed42|E166/E126|78.3198 / 25.2315 / 1.36598|370.0794 / 276.9118 / 9.05606e-123|7.92h|
|yellow_trip_hourly 보정 없음 seed42|E181/E141|87.9050 / 28.4746 / 1.0233|392.4853 / 311.4418 / 1.2641e-15|3.22h|
|yellow_trip_hourly 현재 상태 전용 seed42|E122/E82|86.5854 / 28.0762 / 1.07993|418.8414 / 328.7055 / 5.93166e-53|2.53h|
|yellow_trip_hourly 이력 보정 seed52|E114/E74|79.0817 / 24.8757 / 1.36855|370.7605 / 286.1259 / 1.05634e-40|5.06h|
|yellow_trip_hourly 보정 없음 seed52|E108/E68|86.0713 / 27.7408 / 0.82572|412.8328 / 316.8310 / 1.48889e-196|1.56h|
|yellow_trip_hourly 현재 상태 전용 seed52|E59/E19|89.3333 / 28.5679 / 0.6649|357.0626 / 278.9636 / 9.1469e-33|0.91h|
|yellow_trip_hourly 이력 보정 seed62|E124/E84|83.8251 / 26.7968 / 1.08183|443.4424 / 355.0581 / 2.4182e-15|4.99h|
|yellow_trip_hourly 보정 없음 seed62|E132/E92|88.8586 / 28.6116 / 1.01427|400.9703 / 318.6005 / 4.39011e-79|1.88h|
|yellow_trip_hourly 현재 상태 전용 seed62|E83/E43|88.0836 / 27.9462 / 0.717079|384.2245 / 302.0490 / 5.00681e-24|1.42h|

## Instacart 완료 조건의 같은 선택 epoch Validation

|GPU·조건|저장/best|전체 RMSE / MAE / TimeNLL|큰수량 RMSE / MAE / TimeNLL|fit wall시간|
|---|---|---|---|---|
|5080 MLP8 seed42|E70/E30|5.872632 / 3.988631 / 2.804357|24.284939 / 21.576828 / 2.810068|4.74h|
|5080 MLP12 seed42|E68/E28|5.872392 / 3.987816 / 2.802520|24.643427 / 22.057825 / 2.807190|4.55h|
|5080 CNN+GRU54 seed42|E70/E30|5.866795 / 3.983202 / 2.801382|24.206924 / 21.493974 / 2.800584|15.73h|
|5090 MLP8 seed52|E55/E15|5.884102 / 3.994055 / 2.811975|24.665215 / 22.087655 / 2.899737|3.71h|
|5090 MLP12 seed52|E55/E15|5.868342 / 3.991334 / 2.812495|24.457013 / 21.864049 / 2.898764|3.73h|
|5090 CNN+GRU54 seed52|E55/E15|5.880147 / 3.991407 / 2.811694|24.533208 / 21.905814 / 2.888800|12.34h|

보존된 5090 seed52 CNN+GRU54 완료 결과는 E55/best15이다. 같은 seed52 MLP8 대비 전체/큰수량 RMSE가 0.067%/0.535% 감소했고, MLP12 대비는 0.201%/0.312% 증가했다. 전체 TimeNLL은 양쪽보다 소폭 낮다. 세 구조 모두 E55 종료/best15지만 CNN+GRU54의 fit wall시간은 12.34h로 MLP8 3.71h·MLP12 3.73h의 약3.3배다. 이는 같은 5090에서 관측한 학습시간이며 추론 비용 측정은 아니다. fullValidation503733/큰수량6036(>35)의 단일seed 결과로, 3seed 평균·표본SD와 구조 채택 판단은 미완료다.
5090의 Instacart supervisor exit0, 소유 프로세스 및 GPU PID 부재를 검증했다. 5080은 진행 중이므로 전체9 원본 회수/집계 gate는 local pending이다.

## Taxi 3seed 평균 ± 표본 SD

|조건|전체 RMSE / MAE / TimeNLL|큰수량 RMSE / MAE / TimeNLL|
|---|---|---|
|이력 보정|80.409 ± 2.983 / 25.635 ± 1.022 / 1.2721 ± 0.1648|394.76 ± 42.161 / 306.03 ± 42.707 / 8.0607e-16 ± 1.3962e-15|
|보정 없음|87.612 ± 1.4166 / 28.276 ± 0.46826 / 0.95443 ± 0.11156|402.1 ± 10.22 / 315.62 ± 3.7287 / 4.2137e-16 ± 7.2983e-16|
|현재 상태 전용|88.001 ± 1.3758 / 28.197 ± 0.32788 / 0.82064 ± 0.22606|386.71 ± 30.964 / 303.24 ± 24.892 / 1.6689e-24 ± 2.8907e-24|

현재 완료된 Taxi 조건은 전체8268·큰수량79(>3449)의 fullValidation이다. 최초 strict finite 최소 전체 raw 수량RMSE epoch에서 여섯 지표를 함께 읽었다. Taxi42 full은 보정 없음 대비 전체/큰수량RMSE 10.90%/5.71% 개선, 전체 TimeNLL 33.49% 악화했다. 현재 상태 전용 대비 RMSE는9.55%/11.64% 개선, 전체 TimeNLL은26.49% 악화했다. 보존된 Taxi52 full 완료 결과는 E114/best74이다. 보정 없음 대비 전체/큰수량 RMSE는8.12%/10.19% 개선했고 전체 TimeNLL은65.74% 악화했다. 현재 상태 전용 대비 전체 RMSE는11.48% 개선했지만 큰수량 RMSE는3.84%, 큰수량 MAE는2.57%, 전체 TimeNLL은105.83% 악화했다. 따라서 완료된 두seed에서 전체수량 이득은 관측됐으나 큰수량의 일관된 우위는 확인되지 않았다. Taxi62 full은 E124/best84로 완료됐다. 전체 RMSE83.8251·MAE26.7968·TimeNLL1.081834, 큰수량 RMSE443.4424·MAE355.0581·TimeNLL2.4182e-15이다. 전체 RMSE는 두 대조보다 낮지만 큰수량 RMSE와 전체 TimeNLL은 모두 악화했다. Taxi 세seed 전체 RMSE 우위는 확인됐으나 큰수량의 일관된 우위는 없다. Q1은 같은 두 개 이상 데이터셋이 필요하므로 아직 미확정이다. 동시 실행 wall시간은 단독 구조효율로 해석하지 않는다.

## 비용과 보존

- 비용 manager heartbeat 2026-10-09 12:08:51 KST. 누적 추정 $29.72; 확정 청구액은 아니다. 총$50(준비·학습45+회수·정리5) 상한 유지.
- Terminal service heartbeat 2026-10-09 12:00:49 KST. 10분 간격 관측과 미검증 live backup 유지. 감사 당시 실패·중단·정리·최종완료 파일 없음.
- 기존 작업마감 2026-10-09 21:42:46 KST / compute stop 2026-10-09 22:42:46 KST 유지. 마감은 ETA가 아니다.
- Instacart 도우미1회·5080 SSH1회·5090 완료cache SSH0회(실제 관측시각06:05:11 보존), finalization local pending/remote0. A100은15분 이내 SHA 검증 관측 재사용으로 추가SSH0.
- 신규학습·retry·resume·Pod 생성·예산·공용 Runtime·인증·원본·다른 자동화 변경0. 현 자동화 유지.

**학습과 원본 보존 — 진행 중**
- Instacart9와 A100 이력 보정27을 승인된 소유권·비용 안에서 관측한다.

**이력 보정의 재현성 판정 — 다음 작업**
- 데이터셋별 세seed가 모이면 여섯 지표의 평균·표본SD와 불리한 결과를 포함해 Q1을 판정한다.

**고정 검색의 공통 실행조건 정리 — 외부 작업 대기**
- A100 full9를 RTX 같은GPU mean으로 혼합하지 않는다. 후속 자원·기준선 계약 정리 전 자동 시작하지 않는다.

최종 binary archive/manifest 검증·CPU binary 재추론 감사·Test·새 독립 평가는 미완료다. Validation 개발 증거이며 외부 우월성·통계 유의성·새 독립 평가를 입증하지 않는다.

근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1/hourly_monitor/20261009_120040KST.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5080_1791515309129890000.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791493511276949000.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_execution_20261008_v1/rounds/instacart_only_1791515309163751000.json`
