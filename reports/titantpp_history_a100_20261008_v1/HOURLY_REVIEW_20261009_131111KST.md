# 정기관측: A100 부분 종료

작성 2026-10-09T13:11:11.723329+09:00. A100 마지막 관측 2026-10-09 12:23:25 KST, Pod 삭제 확인 12:23:55 KST. 종료 기록은 현재 시각으로 갱신하지 않았다.

|주제|상태|완료/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|6/1/2/0|
|이력 보정27|차단됨·부분 종료|10/0/15/2|
|고정 검색27|외부 작업 대기|0/0/27/0|

Instacart 5080 실제 관측 2026-10-09 13:10:01 KST: MLP8 seed62 E31/best1, best 미갱신 잔여 39.9분, E300 시나리오 17.90시간. 자체 최근10epoch 중앙값, 속도 유지 가정. 5090은 06:05:11 KST terminal cache 재사용, seed52 세 조건 완료·exit0·소유/GPU PID 부재.

A100 dispatcher는 12:14:47 KST `Foreign GPU process group` 검사로 실패했고 supervisor exit1이다. 실제 외부 프로세스 유입인지 종료 직후 검사 경쟁인지 원인은 미확정이다. 수치 학습 실패로 판정하지 않는다. 보정 없음은 RAF seed42 E49/best9로 완료, full은 E23/best9·현재 상태 전용은 E46/best9에서 소유 worker TERM(-15)로 중단됐다. snapshot 상태 unconfirmed2를 보존하고 부분 checkpoint를 완료 성능으로 비교하지 않는다. 학습 ETA는 중단되어 적용하지 않는다.

Taxi9와 RAF 보정 없음1 총10조건 완료. 나머지15조건 미시작. 동일 두 데이터셋 각3seed 기준 Q1은 미확정이다. RAF 완료 대조만으로 보정 효과를 판단하지 않는다.

부분 원본393파일의 크기·rawSHA, archive rawSHA `83d0d3f4fabd89554a0cc5acc86dc447a9919b22d5247e27a1aaea65ecc01235`, manifest rawSHA `d25ca3393c1c9a295ebf6b37d0695f8b700adf8af68b777a4e09b57c245f2091`를 로컬 재검증했다. 회수 뒤 소유/GPU PID 부재·Pod 삭제 확인 receipt를 보존한다. 실제 종료로 heartbeat 정지는 정상이며 진행 중인 감시 장애로 보고하지 않는다. 마지막 누적 비용 추정 $30.12, 확정 청구액은 미확정이다. $50 상한 초과가 종료 원인은 아니다.

이번 회차 Instacart wrapper1·5080 SSH1·5090 SSH0·A100 원격0. 종료 Pod 접속·retry·resume·새 Pod·충전·자동화 변경0. CPU binary 재추론 감사·Test·새 독립 평가 미완료.

**Instacart 학습과 보존 — 진행 중**
- 기존 승인 실행만 계속 관측한다.

**A100 운영 실패 원인 확인 — 다음 작업 / 차단됨**
- 로컬 보존 증거로 GPU 소유권 검사 실패 원인을 확인해야 한다. 새 실행·재개는 자동 수행하지 않는다.

**고정 검색 공통 실행조건 정리 — 외부 작업 대기**
- A100 full을 RTX 같은 GPU 기준선으로 혼합하지 않는다.
