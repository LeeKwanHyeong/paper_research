# 정기관측

작성 2026-10-09T22:19:19.597201+09:00. 봉인 계약/source/input/op/control와A100 부분원본393파일·archive/manifest SHA 재검증PASS.

|주제|상태|완료/진행/대기/미확정|
|---|---|---|
|Instacart9|진행 중|8/1/0/0|
|이력 보정27|차단됨·부분 종료|10/0/15/2|
|고정 검색27|외부 작업 대기|0/0/27/0|

5080 실제 2026-10-09 22:18:59 KST: MLP12 seed62 E75/best35 완료, 마지막CNN+GRU54 seed62 E4/best1·checkpoint저장됨. 자체4완료epoch 중앙값790.306초, best미갱신/속도유지 시E41잔여8.12시간·E30064.98시간. E300 시나리오는조건36시간한도를초과하며승인된완료예측아님. worker52338/timeout52337/supervisor41573 소유권PASS. 5090은seed52 세조건완료·CNN+GRU54 E55/best15·실제06:05:11KST cache보존·SSH0.

신규 MLP12 seed62 같은선택epoch Validation 전체 RMSE5.899456/MAE3.992747/TimeNLL2.806623, 큰수량 RMSE24.904408/MAE22.387578/TimeNLL2.861771. fit wall 5.00시간. 전체503733·큰수량6036(>35). 신규평가0, CPU재추론/Test아님. 다른arm속도복사0·전체큐ETA미확정.

A100 마지막 실제12:23:25KST·Pod삭제12:23:55KST 유지. Foreign GPU process group 운영검사실패·dispatcherexit1. 수치학습실패로판정하지않음. Taxi9+RAF seed42보정없음 E49/best9 총10완료. RAF full E23/best9·현재전용 E46/best9는TERM(-15)부분중단·snapshot미확정2·미시작15. 원인미확인·ETA없음. 마지막비용추정$30.12/$50·확정청구미확정. 삭제Pod접속0·종료후heartbeat정지는정상.

회차 Instacart wrapper1·5080SSH1·5090SSH0·A100원격0. 재실행/retry/resume/Pod/충전/다른자동화변경0. 기존승인dispatcher큐진척만관측. 현자동화유지. Q1·CPU binary감사·Test·새독립평가미완료.

**Instacart 최종 학습·보존 — 진행 중**
- CNN+GRU54 seed62 종료·소유PID부재·원본회수검증후3seed집계.

**A100 운영 오류 원인 확인 — 다음 작업 / 차단됨**
- 로컬보존증거로원인확인필요·자동재실행금지.

**고정 검색 공통조건 정리 — 외부 작업 대기**
- A100 full을RTX같은GPU기준선으로혼합하지않음.
