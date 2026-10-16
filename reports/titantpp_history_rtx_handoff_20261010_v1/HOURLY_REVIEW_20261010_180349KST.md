# CNN+GRU54 이력 보정 정기관측 — 2026-10-10 18:03:49 KST

5090의 실제 관측시각은 **2026-10-10 18:03:49 KST**다. 5080은 **2026-10-10 13:53:36 KST**의 검증된 종료 캐시를 SSH0으로 재사용했다. A100과 Instacart는 보존한 로컬 원본 SHA만 재검증했으며 원관측시각을 변경하지 않았다.

| 주제 | 상태 | 확인 내용 |
|---|---|---|
| Instacart | 완료 | 9/9, 원본614개·archive/manifest SHA 로컬 검증, 기존 observer/SSH/재학습0 |
| CNN+GRU54 이력 보정 | 진행 중 | 완료20·진행1·대기6·실패0·미확정0. 원래27 execution ID로 한 번씩 집계 |
| 고정 검색 | 대기 | 공통 mean·자원 계약 정리 전 자동 실행0 |

| GPU / 데이터·seed | 방식 | 상태 | 실제 저장/best epoch | best 미갱신 시 잔여 | 300epoch 잔여 | epoch 시간 중앙값 |
|---|---|---|---|---|---|---|
| 5080 / RAF·62 | 이력 보정 full | 완료 | 60/20 | 종료 | 종료 | — |
| 5080 / RAF·62 | 보정 제거 | 완료 | 66/26 | 종료 | 종료 | — |
| 5080 / RAF·62 | 현재 상태 전용 | 완료 | 62/22 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 이력 보정 full | 진행 중 | 34/15 | 3.15시간 (종료 E55) | 39.90시간 | 539.98초 / 10epoch |
| 5090 / Intermittent·42 | 보정 제거 | 완료 | 70/30 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 현재 상태 전용 | 완료 | 46/6 | 종료 | 종료 | — |

5090의 Intermittent seed52·62 × 세 방식6개는 미시작 대기다. 저장/best epoch와 ETA는 없다. 전체 큐 ETA는 미확정이다. 진행 중인 full 자체의 최근10 완료 epoch finite elapsed_seconds 중앙값만 사용해 best 미갱신·현재 속도 유지 시 min(300,max(40,best+40)) 종료와300epoch 시나리오를 분리했다. full의300epoch 39.90시간 시나리오는 조건36시간 lease를 초과하며 허용시간·예산을 연장하지 않는다. endpoint·원본 회수·CPU 감사는 ETA와 별도다. 동시 실행 수가 바뀐 구간이 포함된 시간이므로 구조 효율로 해석하지 않는다.

## 이번 변화와 검증된 증거

- Intermittent seed42 보정 제거가 E70/best30으로 종료됐다. 실제 worker 종료 returncode0 및 terminal manifest SHA `f863cdca3b75b27db00b723cb0c3d6ac2d0392d91aa2bd25340de8b360ed4233`을 검증했다. 현재 상태 전용 E46/best6도 완료 상태를 유지하며 manifest SHA `070546c7082a8db8ff6ee3c6bfa11314ea92d8402370acf0abdc44ba53d8c324`다. 두 조건의 과학 완료와5090 host 전체 종료·immutable 원본 회수는 구분한다. 전체9개가 종료되지 않았으므로 원본 회수는 pending/remote0이다.
- full은 이전 E26에서 실제 저장 E34로 진행했고 best는 E15다. worker PID13255/PPID13254, timeout PID13254/PPID13235, dispatcher PID13235/PPID13234, pipeline PID12696/PPID12695의 정확 argv와 계약·root를 대조했다. 고정 GPU UUID `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`에서 GPU PID13255만 실제 관측됐다. 현재 active fit은1개이며, admission의3worker 증거를 현재3fit 동시 실행으로 재사용하지 않는다.
- lease·claim의 job/canonical/supervisor/worker group과 관측시각 내 유효성을 검증했다. qualification 및 concurrency 실제 exit0·idle 관측,3worker 합성 admission과 training_permit 연결이 일치한다. qualification/admission은 과학 fit 수에 포함하지 않는다.
- checkpoint receipt raw SHA는 `5f13c8bcc0bb297b72c24e014e840c20a2c620d6afd3478d87aabb38437bac4b`이며 E34의 checkpoint/history/timing 원본 SHA와 안정성을 검증했다. snapshot의 rawSHA_verified 및 pointer→관측 묶음→raw snapshot SHA 연결이 통과했다.
- canonical/source129/operation/science integrity/local_tools/양host control/monitor 및 봉인 project_files·source/input/op/control raw SHA 사전 검증 PASS. current/contract/approval/ownership_release/source와 기존 start/deploy/launch receipts, native qualification/admission, hourly snapshot을 대조했다.
- 5080은3/3 terminal, 실제 worker/supervised dispatcher exit0·actual owner/GPU PID부재·pipeline_complete·마지막 phase complete의 검증된 원관측을 재사용했다. pipeline OS exit코드를 직접 관측하지 않았음을 유지한다. 원본233개와 selected/last binary3쌍의 archive/manifest/raw SHA 로컬 검증 PASS. receipt SHA `0ed8a4fb0fe853d461ac2cabd91c5e833228a0e265596ce614242253924deefd` 유지, 추가 회수0.
- A100 고유 완료15와 parent393 archive 및 추출·연결 파일163개 SHA 검증 PASS. actual terminal 원관측12:17:18 KST와 cached epoch 원관측12:25:00 KST를 유지한다. 행정 인계 FileExistsError와 실제 dispatcher/pipeline exit1·failure·claim을 보존하며 새로운 수치 실패나 부모27 완료로 바꾸지 않는다.
- 삭제된 두 Pod와 전용 volume의 로컬 cleanup receipt를 확인했고 추가 SSH/API조회0이다. 누적 추정비용$32.36242173899348, 최종 청구 미확인, 기존50달러 상한 유지.
- 이번 승인 도우미 실행1회, 5090 읽기 전용 관측SSH1회, 5080/A100/InstacartSSH0·PodAPI0. 내부 finalizer는 pending/remote0/user_action_required:false. 추가 deploy/qualification/launch/observer/반복poll/retry/resume/kill/claim삭제/업데이트/재부팅0.

## 미완료 판정과 다음 순서

전체27 종료·원본 확정과 최종 Validation 집계는 대기다. 같은 최초 strict finite 최소 fullValidation 전체 raw수량RMSE 선택 epoch에서 전체/큰수량 RMSE·MAE·TimeNLL 여섯 지표,3seed 평균/표본SD와 Q1을 판정한다. 불리한 조건·큰수량·시간 손해는 제외하지 않는다. 같은 두 개 이상 데이터셋 각각의 세 seed 모두에서 full이 두 대조군보다 전체 수량 RMSE를 엄격히 개선할 때만 Q1 개발 단계 지지로 판정한다. RAF42 full/current_only의 물리 A100 이관과 seed 간 GPU/Runtime 차이, 깊이·활성화·계산 경로의 대조 한계를 유지하고 GPU 시간을 혼합하지 않는다.

다음은5090 종료·원본 확정 → 보정 Q1/여섯 지표 판정 → CNN+GRU54 중심 Results·Instacart·Methods와 주장 정합 순서다. 이후 검색 포함 여부와 새 공통 계약, 미노출 평가 가능성과 CPU 감사 계약, 관련연구/References/Appendix, 컴파일·투고 점검은 PAPER_WORK_ORDER.md를 따른다. 고정 검색 자동 실행0. CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료이며 Test/held-out·Validation/Test 혼합 성과를 열지 않았다.

전체 완료 조건 미충족으로 titantpp-instacart-63을 유지한다. 현재 사용자 조치는 필요없다.

## 관측 원본 SHA 연결

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791623027650143000.json`
- 관측 묶음 raw SHA256: `340a6f9a671ee103322066fc676a3a164a927be474100750765438c4ef31c46f`
- 5080 snapshot raw SHA256: `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`
- 5090 snapshot raw SHA256: `1c11dd38736dc1bbc4cce5280bc8861d71cc5fd9ee2b377c9bcb3ab8a434e5b6`
- canonical: `518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b`
- 진행 full lease: `{"contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b", "expires_unix": 1791623118.4286702, "issued_unix": 1791623028.4286702, "job": "intermittent_frozen_5000__42__titantpp_cnn_gru54", "supervisor_pid": 13235, "worker_group_pid": 13254}`
- 진행 full claim: `{"contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b", "deadline_unix": 1791732104.1937063, "job": {"arm": "titantpp_cnn_gru54", "dataset": "intermittent_frozen_5000", "host": "5090", "id": "intermittent_frozen_5000__42__titantpp_cnn_gru54", "seed": 42}, "maximum_concurrency": 3}`

## 실제 관측 process argv

다음은 원격에서 수집한 읽기 전용 snapshot의 정확 argv다. 명령 실행 예시나 신규 실행 승인이 아니다.

PID 12696 / PPID 12695
```text
/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/control/pipeline.py --root /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090 --contract-sha256 518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b --integrity-sha256 a2821c35778c29dea84044ad77316a5555dc834739f0e6831a8b22beea67fcca
```

PID 13234 / PPID 12696
```text
timeout --signal=TERM --kill-after=15s 604742 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode dispatch
```

PID 13235 / PPID 13234
```text
/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode dispatch
```

PID 13254 / PPID 13235
```text
timeout --signal=TERM --kill-after=15s 129599 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__42__titantpp_cnn_gru54 --deadline 1791732104.1937063
```

PID 13255 / PPID 13254
```text
/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__42__titantpp_cnn_gru54 --deadline 1791732104.1937063
```
