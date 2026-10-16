# 이력 보정 정기관측 — 2026-10-10 14:58:15 KST

5090 실제 관측은 **2026-10-10 14:58:15 KST**다. 5080은 **2026-10-10 13:53:36 KST**의 검증된 종료 cache를 SSH0으로 재사용했다. A100 완료15와 Instacart 완료9는 로컬 원본을 재검증했으며 원관측시각을 바꾸지 않았다.

| 주제 | 상태 | 확인 내용 |
|---|---|---|
| Instacart | 완료 | 9/9, 원본614개·archive/manifest SHA 로컬 검증, 기존 observer/SSH/재학습0 |
| CNN+GRU54 이력 보정 | 진행 중 | 완료18·진행3·대기6·실패0·미확정0, 원래27 ID를 한번씩 집계 |
| 고정 검색 | 대기 | 공통 mean/자원 계약 정리 전 자동 실행0 |

| host·데이터·seed | 방식 | 상태 | 실제 저장/best epoch | best 미갱신 시 잔여 | 300epoch 시나리오 | 완료 epoch 시간 중앙값 |
|---|---|---|---|---|---|---|
| 5080·RAF·62 | 이력 보정 full | 완료 | 60/20 | 종료 | 종료 | — |
| 5080·RAF·62 | 보정 제거 | 완료 | 66/26 | 종료 | 종료 | — |
| 5080·RAF·62 | 현재 상태 전용 | 완료 | 62/22 | 종료 | 종료 | — |
| 5090·Intermittent·42 | 이력 보정 full | 진행 중 | 15/15 | 6.96시간 (E55) | 49.59시간 | 626.34초 / 10epoch |
| 5090·Intermittent·42 | 보정 제거 | 진행 중 | 31/30 | 3.22시간 (E70) | 22.23시간 | 297.45초 / 10epoch |
| 5090·Intermittent·42 | 현재 상태 전용 | 진행 중 | 31/6 | 1.27시간 (E46) | 22.78시간 | 304.86초 / 10epoch |

5090 Intermittent seed52/62의 세 arm씩6개는 미시작 대기다. 저장/best epoch 및 ETA는 없다. 전체 큐 ETA는 미확정이다. 각 fit 자체 최근10 완료 epoch의 finite 시간 중앙값만 사용하며 best 미갱신/현재 속도 유지 시 min(300,max(40,best+40)) 종료와300epoch 시나리오를 구분했다. full과 보정 제거는 best가 각각15/30으로 갱신돼 조건부 종료 epoch가55/70으로 이동했다. 300epoch full49.59시간은 조건36시간 lease를 초과하는 시나리오이며 실행 허용시간이나 예산을 늘리지 않는다. endpoint·원본 회수·CPU 감사는 ETA에 포함하지 않는다.

## 원본·실행 증거

- 봉인 canonical/source129/operation/science integrity/local_tools/양host control/monitor 및 project_files·source/input/op/control raw SHA 검증 PASS. current/approval/root/UUID·launch/start permit과 native qualification/admission bridge를 대조했다.
- 5090 정확 worker argv·PID13255/13261/13267·PPID·고정 GPU UUID/PID 동시 존재·lease·claim·checkpoint/history/timing 검증 PASS. 세 fit이 같은GPU/Runtime에서 동시 진행 중이다. admission을 학습 수량에 포함하지 않는다.
- 5080 scientific terminal3·actual worker/supervised dispatcher exit0·owner/GPU 부재·pipeline_complete/마지막 phase complete 증거는 원관측 cache를 유지했다. pipeline OS exit코드는 직접 관측하지 않았으므로 exit0으로 단정하지 않는다.
- 5080 원본233개·선택/최종 binary3쌍의 archive/manifest/raw SHA를 로컬 재검증했다. receipt SHA `0ed8a4fb0fe853d461ac2cabd91c5e833228a0e265596ce614242253924deefd` 유지. 추가 회수0.
- A10015 보존 원본과 parent393은 로컬 검증했다. actual terminal 원관측12:17:18 KST·cached epoch 원관측12:25:00 KST 유지. 의도된 행정 인계의 실제 dispatcher/pipeline exit1·FileExistsError/failure/claim을 수치 실패로 바꾸지 않는다.
- 삭제된 두 Pod와 전용 volume의 재접속·API조회0. 기존 cleanup 증거와 누적 추정비용$32.36242173899348을 유지하며 최종 청구액은 미확인이다.
- 도우미 실행1회, 5090 읽기 전용 관측SSH1회, 5080·A100·InstacartSSH0. finalizer pending/remote0/user_action_required:false. 추가 deploy/qualification/launch/retry/resume/kill/claim삭제/업데이트/재부팅0.

## 성능 판정과 다음 순서

전체27 원본·종료 확정과 최종 Validation 보고는 대기다. 같은 최초 strict finite 최소 fullValidation 전체 raw수량RMSE 선택 epoch에서 전체/큰수량 RMSE·MAE·TimeNLL 여섯 지표,3seed 평균/표본SD와Q1을 판정한다. 큰수량·시간 손해와 불리한 조건을 제외하지 않는다. RAF42 full/current_only 물리A100 이관·cross-seed GPU/Runtime 차이를 표시하고 시간은 물리GPU 구간별로 분리한다. 성능 파일을 임의로 Test 또는 새 독립 평가 증거로 해석하지 않는다.

다음은5090 종료·원본 확정 → Q1/여섯 지표 판정 → CNN+GRU54 중심 Results와Instacart/Methods/주장 정합이다. 검색 포함 여부/공통 계약, 미노출 평가 가능성·CPU 감사 계약, 관련연구/References/Appendix, 컴파일·투고 점검은 PAPER_WORK_ORDER.md의 순서를 유지한다. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다. Test/held-out 및Validation/Test 혼합 성과는 열지 않았다.

전체 완료 조건 미충족으로 titantpp-instacart-63을 유지한다. 현재 사용자 조치 필요없음.

## SHA로 연결된 관측 원본

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791611895361823000.json`
- 관측 묶음 raw SHA256: `bd03bc596a0e65c26c92ba4befaee51c193f3b6410d98d676426b8aa26c420c6`
- 5080 snapshot raw SHA256: `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`
- 5090 snapshot raw SHA256: `44dc1b4017595464ccad1660fff1fec0cfeb241676a7ae03a9eb3a62bb3c37fa`
