# Instacart 및 A100 이력 보정 정기관측

보고서 작성: 2026-10-09T01:59:33.551591+09:00. 실제 관측시각은 아래 원본 기록을 유지한다.

|주제|완료 여부|완료/실패/진행/대기/미확정|
|---|---|---|
|Instacart 9조건|진행 중|4/0/2/3/0|
|CNN+GRU54 이력 보정 27조건|진행 중|3/0/3/21/0|
|고정 검색 27조건|외부 작업 대기|0/0/0/27/0|

## 학습 관측

|GPU·실제 관측시각|데이터·방식·seed|저장/best epoch|best 미갱신 시 잔여 / E300 시나리오|
|---|---|---|---|
|5080 · 2026-10-09 01:55:24 KST|Instacart · CNN+GRU54 · seed42|E29/E28|8.75h / 60.82h|
|5090 · 2026-10-09 01:55:25 KST|Instacart · CNN+GRU54 · seed52|E37/E15|4.03h / 58.83h|
|A100 · 2026-10-09 01:50:28 KST|Taxi · 이력 보정 full · seed52|첫 checkpoint 저장 전|미확정|
|A100 · 2026-10-09 01:50:28 KST|Taxi · 보정 없음 · seed52|E2/E2|0.60h / 4.46h|
|A100 · 2026-10-09 01:50:28 KST|Taxi · 현재 상태 전용 · seed52|E2/E1|0.61h / 4.64h|

ETA는 해당 fit의 실제 완료 epoch 최근 최대10개 시간 중앙값만 사용한다. A100 seed52 두 대조는 완료2epoch만 있어 불확실성이 크다. full은 첫 저장 전이므로 ETA를 제시하지 않는다. 전체 큐 ETA는 미확정이며 endpoint 평가·원본 회수·CPU 감사는 제외한다.

Instacart의 E300 잔여 시나리오 60.82h/58.83h는 남은 조건별 36h 제한을 넘는다. 따라서 해당 시나리오는 허용된 실행 완료 약속이 아니다. 예산/마감은 변경하지 않았다.

## 완료 조건의 같은 선택 epoch Validation

|주제/GPU|조건|저장/best|전체 RMSE|전체 MAE|전체 TimeNLL|큰수량 RMSE|큰수량 MAE|큰수량 TimeNLL|fit 관측 wall시간|
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
|history/A100|이력 보정 full · seed42|E166/E126|78.31980|25.23154|1.36598|370.07944|276.91183|9.05606e-123|7.92h|
|history/A100|보정 없음 · seed42|E181/E141|87.90504|28.47464|1.0233|392.48530|311.44184|1.2641e-15|3.22h|
|history/A100|현재 상태 전용 · seed42|E122/E82|86.58542|28.07621|1.07993|418.84139|328.70549|5.93166e-53|2.53h|
|instacart/5080|MLP8 · seed42|E70/E30|5.87263|3.98863|2.80436|24.28494|21.57683|2.81007|4.74h|
|instacart/5080|MLP12 · seed42|E68/E28|5.87239|3.98782|2.80252|24.64343|22.05783|2.80719|4.55h|
|instacart/5090|MLP8 · seed52|E55/E15|5.88410|3.99406|2.81198|24.66522|22.08765|2.89974|3.71h|
|instacart/5090|MLP12 · seed52|E55/E15|5.86834|3.99133|2.8125|24.45701|21.86405|2.89876|3.73h|

Taxi는 전체8268·큰수량79(>3449), Instacart는 전체503733·큰수량6036(>35)이다. 각 조건의 최초 strict finite 최소 전체 raw 수량 RMSE epoch에서 여섯 지표를 함께 제시했다.

Taxi seed42의 full은 보정 없음 대비 전체 RMSE 10.90%, 큰수량 RMSE 5.71% 개선했지만 전체 TimeNLL은 33.49% 악화했다. 동일 활성 보정 파라미터 예산의 현재 상태 전용 대비 각각 9.55%, 11.64% 개선했고 전체 TimeNLL은 26.49% 악화했다. 단일 seed의 Validation 개발 관찰이다. Q1은 같은 두 개 이상 데이터셋의 각 세 seed에서 두 대조를 모두 이겨야 하므로 아직 판정하지 않는다.

full fit 관측 wall시간은 7.92h, 보정 없음3.22h, 현재 상태 전용2.53h였다. epoch 수와 동시 GPU 자원 공유 조건이 다르므로 이 비율을 구조 효율로 해석하지 않는다.

## 소유권·비용·보존

- A100 동시 GPU worker PID23051/23057/23063과 supervisor1253, UUID/정확 argv/PPID/lease를 원 관측에서 확인했다. Instacart worker PID44495/4045194와 supervisor41573/3911190도 일치한다.
- A100 비용 감시 heartbeat: 2026-10-09 01:59:11 KST. 누적 비용 추정 $13.50; 청구 확정액은 아니다. 총 상한$50(준비·학습45+회수·정리5), 최초 생성 시각부터의 비용 시계 유지.
- A100 terminal service heartbeat: 2026-10-09 01:50:38 KST; 10분 관측/미검증 live backup 지속. 원본 최종 archive/manifest 검증은 아직 완료되지 않았다.
- 작업 마감 2026-10-09 21:42:46 KST / compute stop 2026-10-09 22:42:46 KST 유지. 마감은 ETA가 아니다. 전체27조건의 상한 내 완료 가능성은 아직 미확정이다.
- Instacart 승인 도우미1회(각 live host SSH1회), finalization local pending·remote0. A100은 15분 이내 검증 snapshot을 재사용하여 추가 SSH0.
- 새 학습 활성화·retry·resume·Pod 생성·예산 변경0. 다른 자동화와 기존 서버 작업은 변경하지 않았다. 현재 자동화를 유지한다.

## 남은 작업 순서

**진행 중인 원본과 학습 상태 확인 — 진행 중**
- Instacart9와 A100 이력 보정27의 현 소유권·비용 상한을 유지하며 완료/실패/원본 보존을 확인한다.

**세 seed 이력 보정 효과 판정 — 다음 작업**
- 완료된 같은 선택epoch 여섯 지표의 평균·표본SD와 불리한 결과를 함께 집계하고 Q1을 판정한다.

**고정 검색의 공통 GPU·Runtime 기준선 정리 — 외부 작업 대기**
- A100 full9를 RTX mean과 같은 환경의 결과로 섞지 않는다. 후속 자원·기준선 계약을 정리하기 전 자동 시작하지 않는다.

CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다. 이 보고서는 외부 우월성·통계 유의성·새 독립 평가를 입증하지 않는다.

근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1/hourly_monitor/20261009_015028KST.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5080_1791478524930737000.json`
근거: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_instacart_20261008_v1/hourly_monitor/observation_5090_1791478525653473000.json`
