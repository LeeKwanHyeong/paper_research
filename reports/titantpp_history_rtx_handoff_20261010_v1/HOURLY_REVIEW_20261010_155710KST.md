# CNN+GRU54 이력 보정 정기관측 — 2026-10-10 15:57:10 KST

5090 실제 관측: **2026-10-10 15:57:10 KST**. 5080은 **2026-10-10 13:53:36 KST** 검증 terminal cache를 SSH0으로 재사용한다. A100 완료15·Instacart 완료9는 로컬 원본 SHA를 재검증했으며 원관측시각을 바꾸지 않았다.

| 주제 | 상태 | 확인 내용 |
|---|---|---|
| Instacart | 완료 | 9/9·원본614 SHA 로컬 검증·기존 observer/SSH/재학습0 |
| CNN+GRU54 이력 보정 | 진행 중 | 완료18·진행3·대기6·실패0·미확정0, 원래27 execution ID 한 번씩 집계 |
| 고정 검색 | 대기 | 공통 mean/자원 계약 확정 전 자동 실행0 |

| GPU / 데이터·seed | 방식 | 상태 | 실제 저장/best epoch | best 미갱신 시 잔여 | 300epoch 잔여 | epoch 시간 중앙값 |
|---|---|---|---|---|---|---|
| 5080 / RAF·62 | 이력 보정 full | 완료 | 60/20 | 종료 | 종료 | — |
| 5080 / RAF·62 | 보정 제거 | 완료 | 66/26 | 종료 | 종료 | — |
| 5080 / RAF·62 | 현재 상태 전용 | 완료 | 62/22 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 이력 보정 full | 진행 중 | 20/15 | 6.10시간 (종료E55) | 48.79시간 | 627.26초 / 10epoch |
| 5090 / Intermittent·42 | 보정 제거 | 진행 중 | 43/30 | 2.20시간 (종료E70) | 20.96시간 | 293.61초 / 10epoch |
| 5090 / Intermittent·42 | 현재 상태 전용 | 진행 중 | 42/6 | 0.33시간 (종료E46) | 21.50시간 | 299.99초 / 10epoch |

5090 Intermittent seed52·62 × 세 arm6개는 미시작 대기이며 저장/best epoch와 ETA는 없다. 전체 큐 ETA는 미확정이다. 현재 상태 전용은 E42/best6로 여전히 running이며 best 미갱신/현재 속도 유지 때 E46까지 약20분이다. 이를 완료로 미리 집계하지 않는다. 각 fit 자체 최근10 완료epoch finite elapsed_seconds 중앙값만 사용했고, min(300,max(40,best+40)) 종료와300epoch 시나리오를 분리했다. full의300epoch48.79시간 시나리오는 조건36시간 lease를 초과하며 실행 허용시간·예산을 연장하지 않는다. endpoint·원본 회수·CPU 감사는 ETA와 별도다. 동시 실행시간을 구조 효율로 해석하지 않는다.

## 확인된 실행·원본 증거

- canonical/source129/op/science integrity/local_tools/양host control/monitor 및 봉인 project_files·source/input/op/control raw SHA 사전 검증 PASS. current/contract/approval/ownership_release/source·기존 start/launch/deploy receipts·native qualification/admission·hourly snapshot를 대조했다.
- 5090 worker PID13255/13261/13267의 정확 argv·PPID·고정GPU UUID/PID 동시 존재·lease·claim·checkpoint/history/timing 검증 PASS. 첫checkpoint 저장 전이 아니며 qualification/admission은 학습 수량에 포함하지 않는다.
- 5080 scientific terminal3·actual worker/supervised dispatcher exit0·owner/GPU PID부재·pipeline_complete/마지막phase complete는13:53:36 KST 원관측 증거로 재사용한다. pipeline OS exit코드는 직접 관측하지 않았음을 유지한다.
- 5080233 원본·selected/last binary3쌍·archive/manifest SHA 로컬 검증 PASS. receipt raw SHA `0ed8a4fb0fe853d461ac2cabd91c5e833228a0e265596ce614242253924deefd` 유지. 이미 회수한 원본의 추가 회수0.
- A10015·parent393 보존 archive/manifest/모든 원본 SHA 검증 PASS. actual terminal 원관측12:17:18 KST·cached epoch 원관측12:25:00 KST를 유지한다. 부모의 의도된 행정 인계 FileExistsError·실제 dispatcher/pipeline exit1·failure/claim을 새로운 수치 실패나 전체27 완료로 조작하지 않는다.
- 삭제된 두 Pod·전용volume의 로컬cleanup증거 유지·재SSH/API조회0. 누적 추정비용$32.36242173899348, 최종청구 미확인,50달러상한 유지.
- 이번 도우미1회·5090 읽기전용관측SSH1회·5080/A100/InstacartSSH0·PodAPI0. finalizer pending/remote0/user_action_required:false. 추가 deploy/qualification/launch/observer/반복poll/retry/resume/kill/claim삭제/업데이트/재부팅0.

## 미완료 판정과 남은 순서

최종27실행 원본·종료 확정과 Validation 집계는 대기다. 동일 최초 strict finite 최소 fullValidation 전체raw수량RMSE 선택epoch에서 전체/큰수량 RMSE·MAE·TimeNLL 여섯지표·3seed 평균/표본SD를 모으며 불리한 조건·큰수량·시간손해를 제외하지 않는다. Q1은 같은 두 개 이상 데이터셋 각각의 세 seed 모두에서 full이 두 대조군보다 전체 수량 RMSE를 엄격히 개선할 때만 개발 단계 지지로 판정한다. RAF42 full/current_only의 물리 A100 이관, seed 간 GPU/Runtime 차이와 깊이·활성화·계산 경로의 대조 한계를 유지하며 GPU 시간을 혼합하지 않는다.

다음은 5090 종료·원본 확정 → 보정 Q1/여섯 지표 판정 → CNN+GRU54 중심 Results·Instacart·Methods와 주장 정합 순서다. 이후 검색 포함 여부와 새 공통 계약, 미노출 평가 가능성과 CPU 감사 계약, 관련연구/References/Appendix, 컴파일·투고 점검은 PAPER_WORK_ORDER.md의 순서를 따른다. 고정 검색 자동 실행은 0이다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료이며 Test/held-out·Validation/Test 혼합 성과는 열지 않았다.

전체 완료조건 미충족으로 titantpp-instacart-63을 유지한다. 현재 사용자 조치 필요없음.

## 관측 원본 SHA 연결

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791615430925197000.json`
- 관측 묶음 raw SHA256: `466ee1b71f1c5d43fffe3e82b0fabf276be9b72c8cc108613fc2ff4a302cf672`
- 5080 snapshot raw SHA256: `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`
- 5090 snapshot raw SHA256: `c8ae05383cafa18c0be64a2dbb75e1feb1541c276b401aa69f54a242798f4cfc`
