# GRU cap9 최대3개 동시 실행 후속 관측

사용자 2026-10-14 승인: “그럼 기존 계약처럼 서버당 3개씩 돌려.. 너 요즘 좀 삐꾸가되간다? 새로운 세션을 열어야하냐”. 이전 순차 운영 지침만 대체한다. 고정9개, rho0.25, source134·입력·모델·손실·head·선택 기준·조건36h/기존host168h·기존CNN/FULL18·과거실패원본을 보존한다. 새 seed/rho/Test/재학습/자동retry는 없다.

현재 root는 기존 execution_manual2의 host별 root이며 새 운영 확장만 parallel3_v2/에 추가됐다. 현재 학습은 원실행기에서 자연 종료하고 미시작 후보만 새실행기가 맡는다. host별 최대3개이며 5090은 배정2개가 전부다. handoff.json의 retained_job_ids와 remaining_jobs의 합이 원host7/2개를 정확히 이루고 겹치지 않아야 한다. 첫 미시작 작업의 exclusive claim은 원순차 실행기의 재시작을 막는 소유권 경계다. 원실행기는 현재 scientific worker 성공 뒤 다음 작업에서 계획된 GPU점유/fresh-run/claim 충돌로 OSexit1 행정 종료하며 이를 수치학습 실패나 exit0로 바꾸지 않는다. failure/claim을 삭제하지 않는다. 새 supervisor는 각자 독립 lease로 최대3개를 관리하고 완료/실패를 실제 timeout child exit로 기록한다. source 모델과 frozen trainer는 그대로이며, 신규 fit은 운영 lease와 progress 경로만 작업별로 분리한다.

고정운영봉인:
- integrity: 64f143793b2134e0db2fc9dd2e9ca80c782f00b5304c32989d5305e6c4c32ef7
- monitor: 86d0aa52d35e2844716ed4aa0e50abf077d52f05b3b0b8c2bb4c7d386e11890c
- collection: 56740fb9797c6ad076ed5ba156c393043bd194ae196cbb45dbb057a7681b0251
- runtime: 21ab3df8615fdc53dc777faf52ad88823847139d2d44afad55dc59b0a4fcd498
- 원contract rawSHA274ff80ef771788dc4652749bd2875716c7ddeb59ba0a1a0e497358545fe1fdc와 canonical9a7e553730e36879fa89f3732c5d4a5493ed60f6bfcb1a4f826cb0990693622f는 그대로다.
- runtime commit e192dda9 / observer commit cabb5ee6. 기존 manual2의 monitor/collection/qualification/source/input 원본봉인도 함께 검증한다.

후속 자동화 gru-cap-9-validation은 다음 명령만 프로젝트root에서 회차당1회 사용한다.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_gru_cap_parallel3_v3 --mode observe --integrity-sha 64f143793b2134e0db2fc9dd2e9ca80c782f00b5304c32989d5305e6c4c32ef7
```

install은 이미 두host에서1회 시작했으므로 다시 실행하지 않는다. livehost SSH1·검증terminalcache SSH0, 별도SSH/반복polling/재실행0. 변경된회차JSON/Markdown/latest rawSHA 및포인터 확인. 완료/진행/대기/미확정을 원9고유ID로만 집계하며 과거시각을 현재로 바꾸지 않는다. 오류/미확정은 학습실패로 추정하지 않는다. 실패가해결되지않으면 원격반복0이며 사용자조치를보고한다. 학습중에는 회수0. 종료후 원본회수는host별1회·SHA검증·exclusive로 진행한다.

실제scientific9 exit0와 모든terminalbinary SHA, 원순차실행기의 계획된행정종료 및 새parallel dispatcher actualexit0, 모든소유PID/GPU부재를 검증해야 학습종료다. pipeline wrapper 자체OSexit는 직접관측하지 않았다고 유지한다. 이후 retrieval/final_receipt statuscomplete·candidate_fit_count9·seed_count3·baseline_reference_count18와 모든archive/manifest/로컬SHA 및 THREE_SEED_VALIDATION.json/md 검증 후 최종3seed Validation 비교를 보고하고 이 자동화 하나만삭제한다.

전체raw수량RMSE 주지표, 전체MAE·큰수량RMSE/MAE·전체/큰수량 recorded positive-integer TimeNLL을 같은선택checkpoint에서 보고한다. 3seed 평균·표본SD와 seed별 CAP−CNN/CAP−FULL 차이와 악화를모두 포함한다. 목적은 GRU보정량 제한의 탐색적 Validation 검증이며 Test일반화 개선/원인확정은아니다. Inter42/52=5090cuDNN92000,62=5080cuDNN91900 차이명시. ETA는각fit최근최대10 finite양수완료epoch중앙값으로best미갱신 min(300,max(40,best+40)) 및300epoch를별도표시한다. 기한연장0·전체큐ETA미확정. 병렬부하변화로이전ETA를그대로연장하지않는다.

기존 관측 주기1시간을 유지하고 의미있는진행변화·완료·실패·사용자조치때만보고한다. Test·mixed결과·원고·다른자동화·Runtime·인증정보·학습조건 변경은허용하지않는다.


2026-10-14 전환 증적: 첫 parallel3 시작은 GPU PID set 인덱싱 오류로 handoff/claim/추가 scientific fit 생성 전에 종료했다. 원격 runtime/approval/pipeline.log와 로컬 stdout/stderr를 보존했다. 수정 운영 코드는 별도 parallel3_v2 경로에만 적용했다. 다음 조회 도구의 /proc 인자 오류도 원문을 보존하고 로컬 관측 v3에서 수정했다. 이미 실행 중인 native parallel3_v2와 기존 학습은 다시 시작하지 않는다. 검증 14개 통과 및 실제 PID/PPID/argv/GPU/lease/qualification bridge 확인 완료.

실제 관측 기준은 2026-10-14 18:36:56 KST다. 완료0·진행5·대기4·미확정0. 5080 Taxi42/52/62 동시3, 5090 Intermittent42/52 동시2. Taxi42 과거 PID185488 및 Inter42 과거 PID4052999는 그대로 유지됐다. Taxi42 저장90/best89, Inter42 저장10/best3, 신규3fit은 저장 전. 이 시각을 새 관측시각으로 바꾸지 않는다. 18:36:56 원문을 로컬 재검증해 보고서를 만들었고 재검증 때 추가 SSH는 하지 않았다. 출력 각host remote_calls1은 원관측 취득 호출수다.

관측 round: rounds/8c42908570834ffb8ad0f93d040dc637.json rawSHA f81c8b741287c0b5f8b54636333cfa6e4d4c1e02eec8a24a4edf5ef75110af6a
Markdown rawSHA130d791fdcf548d5581b833f3ec59bfb13b4cf71b86feafe4ca9fef6bb39a1a3
latest rawSHA9c49806172972fde4721f949a1a1c5d7068aefc4959ce7e5d5d351f22a362243

앞으로 관측 경로는 hourly_monitor_parallel3_v3, 운영제어봉인은 execution_manual2/parallel3_v3_control/integrity.json, 원격 실행경로는 parallel3_v2다. 기존 v1/v2 실패·조회 원본과 봉인은 유지한다. 새학습·재시작·운영코드 추가수정·재봉인·새로운 전환은 후속 자동화에 허용하지 않는다. 이번 병렬 전환 직후 ETA에는 순차 실행 당시 timing이 섞여 있다. 최근 완료 epoch 최대10개 중앙값 조건을 지키되 병렬부하가 정착된 완료 epoch를 확보하기 전 추정의 한계를 밝힌다.
