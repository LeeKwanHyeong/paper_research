# 정기관측

작성 2026-10-09T16:15:37.145945+09:00. 계약·source/input/operation/control 봉인 SHA 및 A100 부분 원본393파일·archive/manifest 로컬 재검증 PASS.

|주제|상태|완료/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|7/1/1/0|
|이력 보정27|차단됨·부분 종료|10/0/15/2|
|고정 검색27|외부 작업 대기|0/0/27/0|

5080 실제 관측 2026-10-09 16:14:59 KST: Instacart MLP8 seed62 E75/best35 완료. MLP12 seed62로 진입, 첫 checkpoint 저장 전·ETA 미확정. worker50828/timeout50827/supervisor41573 소유권 검증 통과. CNN+GRU54 seed62는 대기. 5090 seed52 세 조건 완료·CNN+GRU54 E55/best15, 실제06:05:11KST cache 유지·SSH0.

MLP8 seed62 같은 선택 epoch Validation: 전체 RMSE5.899461/MAE3.990417/TimeNLL2.806123; 큰수량 RMSE24.956328/MAE22.406917/TimeNLL2.863785. 실제 fit시간5.16시간. split 전체503733·큰수량6036(>35). 신규 CPU재추론/Test 아님. 다른 미시작 조건 속도를 복사하지 않는다. 전체 큐 ETA 미확정.

A100 마지막 실제관측12:23:25KST, Pod 삭제확인12:23:55KST 유지. `Foreign GPU process group` 운영검사 실패·dispatcher exit1, 수치학습실패로 판정하지 않음. Taxi9+RAF seed42 보정없음 E49/best9 총10완료. full E23/best9·현재상태전용 E46/best9 부분중단(-15)·미확정2, 미시작15. 원인 미확인·학습 ETA 없음. 마지막 비용추정$30.12/$50·확정청구 미확정. 종료후 heartbeat정지는 정상.

이번회차 Instacart wrapper1·5080 SSH1·5090 SSH0·A100 원격0. 신규fit/retry/resume/Pod생성/충전/다른자동화 변경0. 기존 dispatcher의 승인큐 진척만 관측. 자동화 유지. Q1·CPU binary재추론·Test·새독립평가 미완료.

**Instacart 학습·보존 — 진행 중**
- MLP12·CNN+GRU54 seed62 완료 뒤3seed 집계와 최종 원본 검증.

**A100 운영 실패 원인 확인 — 다음 작업 / 차단됨**
- 로컬 증거로 원인을 확인하며 자동재실행은 하지 않는다.

**고정 검색 공통조건 정리 — 외부 작업 대기**
- A100 full을 RTX 같은GPU 기준선으로 혼합하지 않는다.
