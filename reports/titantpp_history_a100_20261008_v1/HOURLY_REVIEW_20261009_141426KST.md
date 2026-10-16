# 정기관측

작성 2026-10-09T14:14:26.453457+09:00. 봉인 source/input/operation/control SHA와 A100 부분 원본393파일·archive/manifest SHA 재검증 PASS.

|주제|상태|완료/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|6/1/2/0|
|이력 보정27|차단됨·부분 종료|10/0/15/2|
|고정 검색27|외부 작업 대기|0/0/27/0|

5080 실제 관측 2026-10-09 14:13:57 KST: Instacart MLP8 seed62 E47/best35, worker48897·timeout48896·supervisor41573 소유권 PASS. 직전 E31/best1에서 진척됐다. best 미갱신·현재 속도 유지 시 잔여 118.9분, E300 시나리오 17.90시간. 자체 최근10완료epoch 중앙값254.699초. best 갱신으로 종료 예상 epoch가41에서75로 바뀌었다. 전체 큐 ETA 미확정; endpoint/회수/CPU 감사 별도.

5090 seed52 세 조건 완료, CNN+GRU54 E55/best15. 실제 06:05:11KST terminal cache 유지, SSH0.

A100 실제 마지막 관측12:23:25KST·Pod 삭제확인12:23:55KST 보존. 운영 `Foreign GPU process group` 검사 실패·supervisor exit1. Taxi9+RAF seed42 보정없음 E49/best9 총10완료, RAF full E23/best9·현재상태전용 E46/best9 부분중단(-15), snapshot 미확정2 유지. 실제 외부프로세스 유입인지 검사 경쟁인지 원인 미확인. 수치실패와 구분한다. ETA 적용불가. 마지막 비용추정$30.12/$50, 확정청구 미확정. 종료 후 heartbeat 정지는 정상이다. 현재 학습 없음, 종료 Pod 접속0.

Taxi3seed 전체RMSE full 우위·큰수량 불일치·TimeNLL 손해 기록 유지. Q1은 같은2개 이상 dataset 각3seed 필요하므로 미확정. Test·CPU binary 재추론·새 독립평가 미완료.

이번 회차 Instacart wrapper1·5080 SSH1·5090 SSH0·A100 원격0. 자동retry/resume/Pod생성/충전/다른자동화 변경0. 자동화 유지.

**Instacart 학습·보존 — 진행 중**
- 승인된 기존실행을 관측한다.

**A100 운영 실패 원인 확인 — 다음 작업 / 차단됨**
- 보존된 로컬 증거로 원인을 확인하며 신규 실행은 자동 수행하지 않는다.

**고정 검색 공통 조건 정리 — 외부 작업 대기**
- A100 full을 RTX 같은GPU 기준선으로 혼합하지 않는다.
