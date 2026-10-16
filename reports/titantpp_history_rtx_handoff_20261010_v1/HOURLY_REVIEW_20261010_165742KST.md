# CNN+GRU54 이력 보정 정기관측 — 2026-10-10 16:57:42 KST

5090의 실제 관측시각은 **2026-10-10 16:57:42 KST**다. 5080은 **2026-10-10 13:53:36 KST**의 검증된 종료 cache를 SSH0으로 재사용했다. A100과 Instacart는 보존한 로컬 원본 SHA만 재검증했으며 원관측시각을 변경하지 않았다.

| 주제 | 상태 | 확인 내용 |
|---|---|---|
| Instacart | 완료 | 9/9, 원본614개·archive/manifest SHA 로컬 검증, 기존 observer/SSH/재학습0 |
| CNN+GRU54 이력 보정 | 진행 중 | 완료19·진행2·대기6·실패0·미확정0. 원래27 execution ID로 한 번씩 집계 |
| 고정 검색 | 대기 | 공통 mean·자원 계약 정리 전 자동 실행0 |

| GPU / 데이터·seed | 방식 | 상태 | 실제 저장/best epoch | best 미갱신 시 잔여 | 300epoch 잔여 | epoch 시간 중앙값 |
|---|---|---|---|---|---|---|
| 5080 / RAF·62 | 이력 보정 full | 완료 | 60/20 | 종료 | 종료 | — |
| 5080 / RAF·62 | 보정 제거 | 완료 | 66/26 | 종료 | 종료 | — |
| 5080 / RAF·62 | 현재 상태 전용 | 완료 | 62/22 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 이력 보정 full | 진행 중 | 26/15 | 5.02시간 (종료 E55) | 47.38시간 | 622.57초 / 10epoch |
| 5090 / Intermittent·42 | 보정 제거 | 진행 중 | 61/30 | 0.45시간 (종료 E70) | 11.98시간 | 180.39초 / 10epoch |
| 5090 / Intermittent·42 | 현재 상태 전용 | 완료 | 46/6 | 종료 | 종료 | — |

5090 Intermittent seed52·62 × 세 방식6개는 미시작 대기이며 저장/best epoch 및 ETA가 없다. 전체 큐 ETA는 미확정이다. 각 fit 자체의 최근10 완료 epoch finite elapsed_seconds 중앙값을 사용해 best 미갱신·현재 속도 유지 시 min(300,max(40,best+40)) 종료와300epoch 시나리오를 분리했다. full의300epoch47.38시간 시나리오는 조건36시간 lease를 초과하며 허용시간·예산을 연장하지 않는다. endpoint·원본 회수·CPU 감사는 ETA와 별도다. 동시 실행 수가 바뀐 실제 구간의 시간이므로 구조 효율로 해석하지 않는다.

## 이번 변화와 검증된 증거

- Intermittent seed42 현재 상태 전용은 E46/best6으로 scientific terminal이 확인됐다. terminal manifest SHA는 `070546c7082a8db8ff6ee3c6bfa11314ea92d8402370acf0abdc44ba53d8c324`다. 이 한 조건의 과학 완료와5090 host 전체 종료·immutable 원본 회수는 구분한다. 5090 전체9개가 종료되지 않았으므로 원본 회수는 pending/remote0이다.
- 5090 진행 worker PID13255/13261의 정확 argv·PPID·고정 GPU UUID/PID·lease·claim·checkpoint/history/timing을 검증했다. admission의3worker 증거를 현재3fit 실행으로 재사용하지 않으며 현재 active fit은2개다. qualification/admission은 과학 fit 수에 포함하지 않는다.
- canonical/source129/operation/science integrity/local_tools/양host control/monitor 및 봉인 project_files·source/input/op/control raw SHA 사전 검증 PASS. current/contract/approval/ownership_release/source와 기존 start/launch/deploy receipts, native qualification/admission, hourly snapshot을 대조했다.
- 5080은3/3 terminal, 실제 worker/supervised dispatcher exit0·actual owner/GPU PID부재·pipeline_complete·마지막 phase complete의 검증된 원관측을 재사용한다. pipeline OS exit코드를 직접 관측하지 않았음을 유지한다. 원본233개와 selected/last binary3쌍의 archive/manifest/raw SHA 로컬 검증 PASS. receipt SHA `0ed8a4fb0fe853d461ac2cabd91c5e833228a0e265596ce614242253924deefd` 유지, 추가 회수0.
- A100 완료15와 parent393 보존 원본 SHA 검증 PASS. actual terminal 원관측12:17:18 KST와 cached epoch 원관측12:25:00 KST를 유지한다. 행정 인계 FileExistsError와 실제 dispatcher/pipeline exit1·failure·claim을 보존하며 새로운 수치 실패나 부모27 완료로 바꾸지 않는다.
- 삭제된 두 Pod와 전용 volume의 로컬 cleanup receipt를 확인했고 추가 SSH/API조회0이다. 누적 추정비용$32.36242173899348, 최종 청구 미확인, 기존50달러 상한 유지.
- 이번 승인 도우미 실행1회, 5090 읽기 전용 관측SSH1회, 5080/A100/InstacartSSH0·PodAPI0. 내부 finalizer는 pending/remote0/user_action_required:false. 추가 deploy/qualification/launch/observer/반복poll/retry/resume/kill/claim삭제/업데이트/재부팅0.

## 미완료 판정과 다음 순서

전체27 종료·원본 확정과 최종 Validation 집계는 대기다. 같은 최초 strict finite 최소 fullValidation 전체 raw수량RMSE 선택 epoch에서 전체/큰수량 RMSE·MAE·TimeNLL 여섯 지표,3seed 평균/표본SD와 Q1을 판정한다. 불리한 조건·큰수량·시간 손해는 제외하지 않는다. 같은 두 개 이상 데이터셋 각각의 세 seed 모두에서 full이 두 대조군보다 전체 수량 RMSE를 엄격히 개선할 때만 Q1 개발 단계 지지로 판정한다. RAF42 full/current_only의 물리 A100 이관과 seed 간 GPU/Runtime 차이, 깊이·활성화·계산 경로의 대조 한계를 유지하고 GPU 시간을 혼합하지 않는다.

다음은5090 종료·원본 확정 → 보정 Q1/여섯 지표 판정 → CNN+GRU54 중심 Results·Instacart·Methods와 주장 정합 순서다. 이후 검색 포함 여부와 새 공통 계약, 미노출 평가 가능성과 CPU 감사 계약, 관련연구/References/Appendix, 컴파일·투고 점검은 PAPER_WORK_ORDER.md를 따른다. 고정 검색 자동 실행0. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료이며 Test/held-out·Validation/Test 혼합 성과를 열지 않았다.

전체 완료 조건 미충족으로 titantpp-instacart-63을 유지한다. 현재 사용자 조치는 필요없다.

## 관측 원본 SHA 연결

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791619060532390000.json`
- 관측 묶음 raw SHA256: `cafcbed927e942c83e0f4c805126beadb78a6bbd81b23dc75c1b1f2a5d8eac03`
- 5080 snapshot raw SHA256: `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`
- 5090 snapshot raw SHA256: `78bfda3dd667a7def4dd19a7628503bc20ac1505988e1c89d81f2c7f832cb848`
