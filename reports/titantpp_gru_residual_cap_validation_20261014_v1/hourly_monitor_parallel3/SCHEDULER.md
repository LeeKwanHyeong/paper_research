# GRU cap9 최대3개 동시 실행 후속 관측

사용자 2026-10-14 승인: “그럼 기존 계약처럼 서버당 3개씩 돌려.. 너 요즘 좀 삐꾸가되간다? 새로운 세션을 열어야하냐”. 이전 순차 운영 지침만 대체한다. 고정9개, rho0.25, source134·입력·모델·손실·head·선택 기준·조건36h/기존host168h·기존CNN/FULL18·과거실패원본을 보존한다. 새 seed/rho/Test/재학습/자동retry는 없다.

현재 root는 기존 execution_manual2의 host별 root이며 새 운영 확장만 parallel3/에 추가됐다. 현재 학습은 원실행기에서 자연 종료하고 미시작 후보만 새실행기가 맡는다. host별 최대3개이며 5090은 배정2개가 전부다. handoff.json의 retained_job_ids와 remaining_jobs의 합이 원host7/2개를 정확히 이루고 겹치지 않아야 한다. 첫 미시작 작업의 exclusive claim은 원순차 실행기의 재시작을 막는 소유권 경계다. 원실행기는 현재 scientific worker 성공 뒤 다음 작업에서 계획된 GPU점유/fresh-run/claim 충돌로 OSexit1 행정 종료하며 이를 수치학습 실패나 exit0로 바꾸지 않는다. failure/claim을 삭제하지 않는다. 새 supervisor는 각자 독립 lease로 최대3개를 관리하고 완료/실패를 실제 timeout child exit로 기록한다. source 모델과 frozen trainer는 그대로이며, 신규 fit은 운영 lease와 progress 경로만 작업별로 분리한다.

고정운영봉인:
- integrity: 0b2fdbdf25c007017672e13e8c5b54b8c758ef8fd6a839f4f963f6f98674b665
- monitor: 4556e7855f69d6a66143257eec44161531da310bce397efac801b46a57169e91
- collection: 2c104289a400b0e51cedfb5f30f9506d27a5636569814445c6bc8c73456819bd
- runtime: 021352e23ec41002ba1d7a11e4df9ff47a93332bedfb40bacad27b5f253b3300
- 원contract rawSHA274ff80ef771788dc4652749bd2875716c7ddeb59ba0a1a0e497358545fe1fdc와 canonical9a7e553730e36879fa89f3732c5d4a5493ed60f6bfcb1a4f826cb0990693622f는 그대로다.
- code commit79730014. 기존 manual2의 monitor/collection/qualification/source/input 원본봉인도 함께 검증한다.

후속 자동화 gru-cap-9-validation은 다음 명령만 프로젝트root에서 회차당1회 사용한다.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_gru_cap_parallel3 --mode observe --integrity-sha 0b2fdbdf25c007017672e13e8c5b54b8c758ef8fd6a839f4f963f6f98674b665
```

install은 이미 두host에서1회 시작했으므로 다시 실행하지 않는다. livehost SSH1·검증terminalcache SSH0, 별도SSH/반복polling/재실행0. 변경된회차JSON/Markdown/latest rawSHA 및포인터 확인. 완료/진행/대기/미확정을 원9고유ID로만 집계하며 과거시각을 현재로 바꾸지 않는다. 오류/미확정은 학습실패로 추정하지 않는다. 실패가해결되지않으면 원격반복0이며 사용자조치를보고한다. 학습중에는 회수0. 종료후 원본회수는host별1회·SHA검증·exclusive로 진행한다.

실제scientific9 exit0와 모든terminalbinary SHA, 원순차실행기의 계획된행정종료 및 새parallel dispatcher actualexit0, 모든소유PID/GPU부재를 검증해야 학습종료다. pipeline wrapper 자체OSexit는 직접관측하지 않았다고 유지한다. 이후 retrieval/final_receipt statuscomplete·candidate_fit_count9·seed_count3·baseline_reference_count18와 모든archive/manifest/로컬SHA 및 THREE_SEED_VALIDATION.json/md 검증 후 최종3seed Validation 비교를 보고하고 이 자동화 하나만삭제한다.

전체raw수량RMSE 주지표, 전체MAE·큰수량RMSE/MAE·전체/큰수량 recorded positive-integer TimeNLL을 같은선택checkpoint에서 보고한다. 3seed 평균·표본SD와 seed별 CAP−CNN/CAP−FULL 차이와 악화를모두 포함한다. 목적은 GRU보정량 제한의 탐색적 Validation 검증이며 Test일반화 개선/원인확정은아니다. Inter42/52=5090cuDNN92000,62=5080cuDNN91900 차이명시. ETA는각fit최근최대10 finite양수완료epoch중앙값으로best미갱신 min(300,max(40,best+40)) 및300epoch를별도표시한다. 기한연장0·전체큐ETA미확정. 병렬부하변화로이전ETA를그대로연장하지않는다.

기존 관측 주기1시간을 유지하고 의미있는진행변화·완료·실패·사용자조치때만보고한다. Test·mixed결과·원고·다른자동화·Runtime·인증정보·학습조건 변경은허용하지않는다.
