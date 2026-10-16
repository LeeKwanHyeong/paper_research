# 정기관측

작성 2026-10-09T17:18:01.643196+09:00. 승인 계약·source/input/operation/control 봉인 SHA와 A100 부분원본393파일·archive/manifest 재검증 PASS.

|주제|상태|완료/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|7/1/1/0|
|이력 보정27|차단됨·부분 종료|10/0/15/2|
|고정 검색27|외부 작업 대기|0/0/27/0|

5080 실제 2026-10-09 17:17:28 KST: Instacart MLP12 seed62 E16/best1, worker50828/timeout50827/supervisor41573 소유권 PASS. 최초저장전에서16epoch 저장됨. 자체최근10완료epoch 중앙값239.319초. best미갱신·속도유지 시 E41까지 99.7분, E300 18.88시간. CNN+GRU54 seed62 대기. 전체큐ETA 미확정, endpoint/회수/CPU 감사 별도.

5090 seed52 세조건 완료·CNN+GRU54 E55/best15, 실제06:05:11KST cache 보존·SSH0. Instacart 실패·미확정0. 기존완료의 같은선택epoch Validation 수량·TimeNLL 및학습시간 기록유지, 새완료0.

A100 마지막 실제12:23:25KST·Pod 삭제12:23:55KST 유지. 운영 Foreign GPU process group 검사실패·dispatcher exit1. 수치실패로판정하지않음. Taxi9와RAF seed42 보정없음 E49/best9 완료10. RAF full E23/best9·현재상태전용 E46/best9는TERM(-15) 부분중단·snapshot미확정2. 미시작15. 운영원인미확인·학습ETA없음. 마지막비용추정$30.12/$50·확정청구미확정. 종료후heartbeat정지는정상. 삭제Pod 접속0.

회차 Instacart wrapper1·5080 SSH1·5090 SSH0·A100 원격0. 자동재실행·retry·resume·Pod·충전·다른자동화 변경0. 현자동화유지. CPU binary재추론/Test/새독립평가 미완료, Q1은두dataset각3seed를충족하지못해미확정.

**Instacart 학습과 보존 — 진행 중**
- MLP12와CNN+GRU54 seed62 종료후3seed집계·최종원본검증.

**A100 운영 실패 원인 확인 — 다음 작업 / 차단됨**
- 로컬보존증거로원인을확인해야하며자동재실행하지않음.

**고정 검색 공통 실행조건 — 외부 작업 대기**
- A100 full을RTX같은GPU기준선으로혼합하지않음.
