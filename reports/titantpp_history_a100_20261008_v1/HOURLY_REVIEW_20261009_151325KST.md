# 정기관측

작성 2026-10-09T15:13:25.598045+09:00. 봉인 계약·source/input/operation/control SHA 및 A100 부분 원본393파일·archive/manifest 재검증 PASS.

|주제|상태|완료/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|6/1/2/0|
|이력 보정27|차단됨·부분 종료|10/0/15/2|
|고정 검색27|외부 작업 대기|0/0/27/0|

5080 실제 관측 2026-10-09 15:12:55 KST: Instacart MLP8 seed62 E61/best35, worker48897·timeout48896·supervisor41573 소유권 PASS. 이전 E47/best35에서14epoch 진행. best 미갱신·속도 유지 시 E75까지 59.5분, E300 시나리오 16.93시간. 자체 최근10완료epoch 중앙값255.037초. 전체큐 ETA 미확정; endpoint/회수/CPU감사 별도.

5090 seed52 세 조건 완료, CNN+GRU54 E55/best15. 실제 06:05:11KST terminalcache 시각 유지·SSH0.

A100 마지막 실제 관측12:23:25KST·Pod 삭제 확인12:23:55KST 유지. `Foreign GPU process group` 운영검사 실패·dispatcher exit1로 중단, 수치 학습실패로 판정하지 않음. Taxi9·RAF seed42 보정없음 E49/best9 총10완료; RAF full E23/best9·현재상태전용 E46/best9 부분중단(-15)·snapshot 미확정2 보존, 미시작15. 삭제 Pod 접속/재시작0. 원인 미확인, 학습 ETA 없음. 마지막 비용추정$30.12/$50, 확정청구 미확정. 종료 후 heartbeat 정지는 정상.

완료 Taxi3seed 전체RMSE full 우위·큰수량 불일치·TimeNLL 손해 유지. 같은2개 이상dataset 각3seed 필요하므로 Q1 미확정. CPU binary재추론·Test·새독립평가 미완료.

이번회차 Instacart wrapper1·5080 SSH1·5090 SSH0·A100 원격0. 신규fit/retry/resume/Pod/충전/다른자동화 변경0. 자동화 유지.

**Instacart 학습·보존 — 진행 중**
- 기존 승인 실행만 관측한다.

**A100 운영 실패 원인 확인 — 다음 작업 / 차단됨**
- 보존된 로컬 증거로 원인을 확인해야 한다. 자동재실행은 수행하지 않는다.

**고정 검색 공통조건 정리 — 외부 작업 대기**
- A100 full을 RTX 같은GPU 기준선으로 혼합하지 않는다.
