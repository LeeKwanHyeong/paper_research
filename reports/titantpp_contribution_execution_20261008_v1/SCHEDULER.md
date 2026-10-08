# 승인된 단계 실행과 시간별 관측

각 회차는 로컬 current/canonical/source/input/operation/control SHA를 먼저 대조한다. 봉인이 다르면 원격에 접속하지 않는다.

`/usr/local/bin/python3 paper/scripts/advance_titantpp_contribution_campaigns.py`를 프로젝트 root에서 회차당 한 번 실행한다. 이 명령은 승인된 stage 하나의 미시도 qualification/학습 시작, live host별 읽기 전용 관측 한 번, 완료 gate의 원본 보존·집계를 연결한다. 완료된 stage는 원격0회로 로컬 원본 SHA만 확인한다.

학습 관측과 아직 미시도인 승인 실행·qualification 회수·최종 원본 회수는 별도 작업이다. incomplete intent/claim/failure는 보존하고 자동 retry/resume/kill 또는 예산 연장을 하지 않는다. 새 worker의 timeout과 자기 소유 group만 운영 예산으로 제어한다.

매시간 한국어 표에 host 실제 KST 관측시각, 완료/실패/진행/대기/미확정, dataset/arm/seed, 실제 저장/best epoch를 표시한다. 첫 저장 전을 명시한다. qualification 단계는 학습 진행으로 세지 않는다. 조회 실패는 학습 실패로 세지 않는다.

ETA는 해당 fit 최근 최대10 완료 epoch의 finite 양수 시간 중앙값만 사용한다. best 미갱신·속도 유지 가정의 min(300,max(40,best+40)) 잔여와 E300 시나리오를 분리한다. 미시작 속도 추정과 전체 큐 ETA는 미확정이다. endpoint/원본 회수/CPU 감사는 별도다.

세 stage 63개·실제 host 종료·모든 원본 archive/manifest rawSHA·최종 Validation 보고가 확인되면 최종 상태를 기록하고 이 단계 실행 자동화만 삭제한다. 다른 자동화·과거 root·Pod·서버 Runtime·인증·chat archive는 변경하지 않는다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다.
