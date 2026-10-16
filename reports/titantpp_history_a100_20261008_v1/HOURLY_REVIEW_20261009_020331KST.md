# Instacart·A100 정기관측

작성: 2026-10-09T02:03:31.540114+09:00. 원 관측시각을 유지한다.

|주제|완료 여부|완료/실패/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|4/0/2/3/0|
|CNN+GRU54 이력 보정27|진행 중|3/0/3/21/0|
|고정 검색27|외부 작업 대기|0/0/0/27/0|

|실제 KST 관측·GPU|조건|저장/best|best 미갱신 잔여 / E300 시나리오|
|---|---|---|---|
|2026-10-09 02:01:29 KST · 5080|Instacart CNN+GRU54 seed42|E30/E30|8.98h / 60.61h|
|2026-10-09 02:01:29 KST · 5090|Instacart CNN+GRU54 seed52|E38/E15|3.81h / 58.64h|
|2026-10-09 02:00:41 KST · A100|Taxi 이력 보정 full seed52|E4/E4|1.82h / 13.44h|
|2026-10-09 02:00:41 KST · A100|Taxi 보정 없음 seed52|E13/E10|0.55h / 4.30h|
|2026-10-09 02:00:41 KST · A100|Taxi 현재 상태 전용 seed52|E13/E10|0.58h / 4.47h|

A100 full을 포함한 진행3조건 모두 첫 checkpoint 저장을 확인했다.
ETA는 각 GPU/fit 자체 최근 최대10완료epoch 시간 중앙값이며 A100 full은4epoch 표본이다. best미갱신·속도유지 가정이고 전체큐ETA·상한내전체완료여부는 미확정이다. endpoint평가/회수/CPU감사는 별도다. Instacart E300 시나리오는 남은 조건별36h 예산을 초과하므로 승인된 완료예측으로 해석하지 않는다.

## 완료된 Taxi seed42 결과 — 이전 원본 유지

|보정|저장/best|전체 RMSE / MAE / TimeNLL|큰수량 RMSE / MAE / TimeNLL|관측 fit wall시간|
|---|---|---|---|---|
|이력 보정 full|E166/E126|78.3198 / 25.2315 / 1.36598|370.0794 / 276.9118 / 9.05606e-123|7.92h|
|보정 없음|E181/E141|87.9050 / 28.4746 / 1.0233|392.4853 / 311.4418 / 1.2641e-15|3.22h|
|현재 상태 전용|E122/E82|86.5854 / 28.0762 / 1.07993|418.8414 / 328.7055 / 5.93166e-53|2.53h|

같은 최초 strict finite 최소 전체raw수량RMSE epoch의 fullValidation8268/큰수량79(>3449)다. full은 보정없음 대비 전체/큰수량RMSE10.90%/5.71%개선, 전체TimeNLL33.49%악화. 현재상태전용 대비9.55%/11.64%개선, 전체TimeNLL26.49%악화. 공유GPU 동시fit wall시간은 구조효율 비교에 사용하지 않는다. 세seed평균·표본SD와 Q1은 아직 미확정이다.

## 비용·증적

- Manager heartbeat 2026-10-09 02:03:12 KST; 누적비용 추정$13.60. 확정 청구액은 아니다. 총$50(준비·학습45+회수·정리5) 유지.
- Terminal service heartbeat 2026-10-09 02:00:48 KST; 10분간격 관측·미검증livebackup 유지. 새 오류/실패/stop/cleanup/completion 증거 없음.
- 원래 작업마감 2026-10-09 21:42:46 KST / compute stop 2026-10-09 22:42:46 KST 유지. 마감은 ETA가 아니다.
- 실제 A100 worker GPU PID23051/23057/23063, supervisor1253, timeout23050/23056/23062의 argv·PPID·UUID·lease·checkpoint receipt일치. 세학습프로세스 동시확인.
- Instacart supervisor/worker/timeout/GPU 소유권과 checkpoint SHA일치. 관측도우미1회·각host SSH1회, finalization local pending/remote0. A100은15분이내서비스관측을재사용해추가SSH0.
- 다른자동화/Pod/공용Runtime/인증/원본변경0, 신규학습/retry/resume0. 현재자동화유지.

**남은 실험과 원본 확인 — 진행 중**
- Instacart9와 A100 이력보정27을 현 계약 안에서 관측하며 최종 원본 회수·실제종료를 확인한다.

**이력 보정의 재현성 판정 — 다음 작업**
- 세seed 결과가 모이면 여섯지표 평균·표본SD와 불리한 조건을 함께 보고하고 Q1을판정한다.

**고정 검색의 공통 실행조건 정리 — 외부 작업 대기**
- A100 full9를 RTX 같은GPUmean으로혼합하지않는다. 후속 자원·기준선계약정리전자동시작하지않는다.

CPU binary재추론감사·Test·새독립평가는 별도미완료다. Validation개발증거이며 외부우월성·통계유의성·새독립평가를입증하지않는다.

근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1/hourly_monitor/20261009_020041KST.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5080_1791478889416316000.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791478889568872000.json`
