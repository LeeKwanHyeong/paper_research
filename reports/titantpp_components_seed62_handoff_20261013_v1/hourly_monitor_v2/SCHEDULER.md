# CNN·GRU36 관측 도구 v2

**기존 실험과 승인 범위를 유지한다 — 완료**

- 원래36개 ID·source131·input·operation·local40·기존 monitor 봉인은 그대로 유지한다. Intermittent42/52의8개는 이미 승인된 기존 학습이다. 추가 학습·재실행·claim/failure 변경은 하지 않는다.
- 원5080 Taxi/RAF24는 원hourly_monitor_v2 종료 캐시로 SSH0, 원관측2026-10-13 08:27:20 KST·dispatcher 종료06:50:48 KST를 유지한다. 현재5080 전역 idle 증명은 아니다.

**오류 출력과 실제 종료 증거를 검증한다 — 완료**

- 사용자의2026-10-14 관측 도구 수정·검증 및 실제 종료 확인 승인을 별도approval.json에 기록했다. 신규 local tool 버전만 추가했다. 정상 failed+active_jobs=[]를 처리하고 잘못된 shape/PID는 거부한다.
- status만 허용한다. 기존 pinned Python/environment에서 source/input/op/계약/approval SHA를 먼저 검증한다. 같은 SSH 내 fixed GPU UUID inventory 및 compute PID, dispatcher·wrapper·worker /proc PID/PPID/argv/start_ticks를 읽는다.
- 처음 stat의 부재와 중간 소멸/query_error/PID 재사용을 구분한다. 진행 판정은 안정된live, 종료 판정은 명시적인owner 부재가 필요하다. stdout/stderr 원문은diagnostics에exclusive 보존하며 전체SHA·보존SHA·byte수·truncation·actual transport exit/timeout을 기록한다.
- 부모5090은 8scientificworker 실제exit0/모든terminalmanifestSHA/qualification bridge 및계획된FileExistsError·barrier claimSHA·dispatcher 실제exit1·owner/GPU부재를 모두검증한뒤만행정종료cache로기록한다. 이를original12완료나수치실패로바꾸지않는다. pipeline wrapper OSexit는직접관측하지않았다.
- 새5080 terminal도worker·dispatcher 실제exit0·train_complete·terminalbinarySHA·owner/GPU부재를 모두검증한다. 기존claimed원본회수0,자동retry/resume0,kill0이다.

**매시간 checkpoint와 ETA를 보고한다 — 진행 중**

- 원래36개를중복없이집계하고host별관측시각·dataset/seed/arm·상태·저장/best epoch를보고한다. parent5090검증cache는SSH0,진행중new5080은statusSSH1,미launchwaiting은SSH0다. helper회차당1회·별도SSH/observer/polling0. 최신오류receipt와원문SHA를보고하며조회오류를학습실패로바꾸지않는다.
- 먼저TEST_SESSION_PROTOCOL0·1·9·12와이문서를읽고새seal과기존seal·current/계약·승인·source/input/op/ownership·deployment/start/trainingpermits/nativequalification을검증한다. mismatch는추가원격0·덮어쓰기0·사용자조치다.
- 실행명령: `PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.titantpp_components_parallel_monitor_v2 --monitor-integrity-sha 4f2ab78970d862cf146365de6a75999950964c2d809c27183d1a970f37fedb40`
- v2 monitor_integrity rawSHA: `4f2ab78970d862cf146365de6a75999950964c2d809c27183d1a970f37fedb40`. legacy parallel97bf43492d069a9c41e157d1b69b1cfbb0b8182f1c49af9cd5ac073f4b2ebc09 및원local40/monitor/source봉인은변경하지않는다.
- 자체최근최대10완료epoch finite양수elapsed_seconds중앙값으로best미갱신·현재속도유지의min(300,max(40,best+40))잔여와300epoch잔여를구분한다. best는전체fullValidation raw수량RMSE의최초strictfinite최소epoch다. 미시작/다른fit/GPU속도차용0.36h조건/절대168hhost기한연장0,300시나리오가남은기한을넘으면기한내보장불가로표시한다. 전체큐ETA미확정.
- Intermittent42/52는5090/cuDNN92000,62는5080/cuDNN91900이며cross-seed환경차이를명시하고시간은host별계산한다.

**모든36개 종료 뒤 원본과 최종Validation을 확정한다 — 다음 작업**

- all_training_terminal_verified=true이면최종관측보고후자동화cnn-gru36-checkpoint-eta 하나만삭제한다. 미확정/partial/failed이면유지한다. 로컬immutablearchive/manifest/binary원본회수·최종Validation집계는별도다.
- 이자동화는관측/조건부ETA/별도보고만허용한다. deploy/qualify/train/새학습/retry/resume/kill/claimfailure삭제/Runtime driver auth변경/Test검색/원본회수0. Test·heldout성과·ValidationTest혼합파일·main.tex·다른자동화는읽거나변경하지않는다.
