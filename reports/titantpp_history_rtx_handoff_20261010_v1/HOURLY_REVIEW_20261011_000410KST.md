# CNN+GRU54 이력 보정 정기관측 — 2026-10-11 00:04:10 KST

5090의 실제 관측시각은 **2026-10-11 00:04:10 KST**다. 5080은 **2026-10-10 13:53:36 KST**의 검증된 종료 캐시를 SSH0으로 재사용했다. A100과 Instacart는 로컬 원본 SHA만 재검증했으며 원관측시각을 바꾸지 않았다.

| 주제 | 상태 | 확인 내용 |
|---|---|---|
| Instacart | 완료 | 9/9,614원본·archive/manifest SHA 검증 |
| CNN+GRU54 이력 보정 | 진행 중 | 완료21·진행3·대기3·실패0·미확정0 |
| 고정 검색 | 대기 | 공통 mean·자원 계약 정리 전 자동 실행0 |

| GPU / 데이터·seed | 방식 | 상태 | 저장/best epoch | best 미갱신 시 잔여 | 300epoch 잔여 | epoch 시간 중앙값 |
|---|---|---|---|---|---|---|
| 5080 / RAF·62 | 이력 보정 full | 완료 | 60/20 | 종료 | 종료 | — |
| 5080 / RAF·62 | 보정 제거 | 완료 | 66/26 | 종료 | 종료 | — |
| 5080 / RAF·62 | 현재 상태 전용 | 완료 | 62/22 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 이력 보정 full | 완료 | 55/15 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 보정 제거 | 완료 | 70/30 | 종료 | 종료 | — |
| 5090 / Intermittent·42 | 현재 상태 전용 | 완료 | 46/6 | 종료 | 종료 | — |
| 5090 / Intermittent·52 | 이력 보정 full | 진행 중 | 18/11 | 5.82시간 (종료 E51) | 49.74시간 | 634.95초 / 10epoch |
| 5090 / Intermittent·52 | 보정 제거 | 진행 중 | 39/13 | 1.15시간 (종료 E53) | 21.52시간 | 296.79초 / 10epoch |
| 5090 / Intermittent·52 | 현재 상태 전용 | 진행 중 | 39/25 | 2.17시간 (종료 E65) | 21.75시간 | 300.07초 / 10epoch |

미시작 대기3개는 저장/best epoch와 ETA가 없다. 다른 arm·GPU의 속도를 복사하지 않았다. 전체 큐 ETA는 미확정이다. 각 fit 자체 최근 최대10 완료 epoch finite elapsed_seconds 중앙값으로 두 시나리오를 분리했다. ETA 표본 수는 각 행에 표시했다. 이후 best 갱신이나 속도 변화에 따라 달라질 수 있다. 조건36시간·host168시간 lease와 예산을 유지한다. endpoint·원본 회수·CPU 감사는 ETA와 별도다. 동시 실행 구간의 시간을 구조 효율로 해석하지 않는다.

- seed52 이력 보정 full: best 미갱신·속도 유지 시 잔여 5.82시간,300epoch 잔여 49.74시간. 관측 당시 조건 deadline까지 남은 허용시간은 32.71시간이다. deadline은 ETA가 아니다.
- seed52 이력 보정 full의 300epoch 시나리오는 조건 기한을 약 17.02시간 초과한다. 완료를 보장하지 않으며 기존 lease·기한·예산을 변경하지 않는다.
- seed52 보정 제거: best 미갱신·속도 유지 시 잔여 1.15시간,300epoch 잔여 21.52시간. 관측 당시 조건 deadline까지 남은 허용시간은 32.71시간이다. deadline은 ETA가 아니다.
- seed52 현재 상태 전용: best 미갱신·속도 유지 시 잔여 2.17시간,300epoch 잔여 21.75시간. 관측 당시 조건 deadline까지 남은 허용시간은 32.71시간이다. deadline은 ETA가 아니다.

## 이전 관측 이후의 변화

- seed52 이력 보정 full: 이전 running / 저장13 → 현재 running / 저장18,best11.
- seed52 보정 제거: 이전 running / 저장27 → 현재 running / 저장39,best13.
- seed52 현재 상태 전용: 이전 running / 저장27 → 현재 running / 저장39,best25.

현재 active fit은3개이며 raw snapshot의 실제 GPU PID는 `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, 184984
GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, 184990
GPU-c9adc246-ef66-7906-bc07-6e152fa9952f, 184996`다. admission의 과거3worker 증거를 현재 fit 실행으로 오인하지 않았다. qualification/admission은 학습 수에 포함하지 않는다. 완료된 각 worker의 실제 exit0·terminal manifest를 대조했고, 전체5090의9개 종료·실제 dispatcher exit·owner/GPU PID부재가 아직 모두 확인되지 않아 최종 원본 회수는 pending/remote0이다. pipeline OS exit코드를 직접 관측하지 않았다는 사실을 유지한다.

## 봉인·원본·운영 증거

- canonical/source129/operation/science integrity/local_tools/양host control/monitor와 봉인 project_files·source/input/op/control stdlib raw SHA 사전 검증 PASS. current/contract/approval/ownership_release/source 및 기존 start/deploy/launch receipts, native qualification/admission/training permit, hourly pointer→관측 묶음→raw snapshot 연결을 확인했다.
- 정확 argv·PID·PPID·GPU UUID/PID·lease·claim과 실제 checkpoint/history/timing의 raw SHA를 확인했다. 진행 fit의 checkpoint receipt는 아래 원본 연결에 보존했다.
- 5080의3 terminal, 실제 worker/supervised dispatcher exit0·actual owner/GPU PID부재·pipeline_complete·마지막 phase complete는 원관측을 재사용한다. 원본233개·selected/last binary3쌍의 archive/manifest/raw SHA 검증 PASS. receipt `0ed8a4fb0fe853d461ac2cabd91c5e833228a0e265596ce614242253924deefd` 유지, 추가 회수0.
- A100 고유 완료15 및 parent393 archive·추출 연결163개 SHA 검증 PASS. actual terminal2026-10-10 12:17:18 KST와 cached epoch2026-10-10 12:25:00 KST를 유지한다. 행정 인계 FileExistsError·실제 dispatcher/pipeline exit1·failure·claim을 보존했고 새로운 수치 실패나 부모27 완료로 바꾸지 않았다.
- 삭제된 두 Pod와 전용 volume의 로컬 cleanup receipt 정상, 추가 SSH/API조회0. 누적 추정비용$32.36242173899348, 최종 청구 미확인, 총50달러 상한 유지.
- 승인 advance 도우미1회,5090 읽기 전용 관측SSH1회,5080/A100/InstacartSSH0·PodAPI0. 내부 finalizer1회/pending/remote0/user_action_required:false. 추가 deploy/qualification/launch/observer/반복poll/retry/resume/kill/claim삭제/업데이트/재부팅0.

## 남은 작업

**원본과 전체27 종료를 확정한다 — 진행 중**
- 5090의 남은 학습과 실제 종료 증거를 확인하고, 모든 terminal 조건이 충족된 뒤만 immutable 원본을 회수한다. 원래 execution ID를 한 번씩 집계한다. 자동화 titantpp-instacart-63을 유지하며 현재 사용자 조치는 필요없다.

**같은 선택 epoch로 보정 기여를 판정한다 — 다음 작업 / 원본 확정 후**
- fullValidation 전체 raw수량RMSE의 최초 strict finite 최소 epoch에서 동일 전체/큰수량 RMSE·MAE·TimeNLL 여섯 지표와3seed 평균/표본SD를 집계한다. 같은 두 개 이상 dataset의 각3seed에서 full이 두 대조보다 전체 수량RMSE를 strict 개선할 때만 Q1 개발지지다. 불리한 조건과 큰 수량·시간 손해를 제외하지 않는다.
- RAF42 full/current_only 물리 A100 이관과 seed 간 GPU/Runtime 차이를 명시한다. full/current_only 활성 보정 파라미터25308은 같지만 깊이·활성화·계산 경로가 다르고,현재 hidden에 prefix 이력이 포함된다는 한계를 유지한다. GPU 시간을 혼합하지 않으며 통계유의성·참신성·외부우월성·새독립평가 증명으로 확대하지 않는다.

**확정 결과를 CNN+GRU54 중심 본문에 반영한다 — 다음 작업 / 판정 후**
- Results·Instacart·Methods와 주장 정합 → 검색 포함 여부·필요시 공통 계약 → 미노출 평가·CPU 감사 계약 → 관련연구/References/Appendix → 컴파일·투고 점검은 PAPER_WORK_ORDER.md를 따른다. 고정 검색 자동 실행0,CPU binary 재추론 감사·Test·새 독립 평가는 별도 미완료다. Test/held-out·Validation/Test 혼합 성과 열람0.

## 관측 원본 SHA 연결

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791644650629332000.json`
- 관측 묶음 raw SHA256: `3e48fe533771d76e3816253b37eb087bbe84225571f081ebc65651d67642e91f`
- 5080 snapshot raw SHA256: `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`
- 5090 snapshot raw SHA256: `65d515ae638f80a486e5955b1fef97abd0e0741adedfb62279b523bc8ccdc944`
- canonical: `518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b`
- 진행 fit receipt·lease·claim 연결:

```json
[
  {
    "job": "intermittent_frozen_5000__52__titantpp_cnn_gru54",
    "worker_pid": 184984,
    "worker_ppid": 184983,
    "timeout_pid": 184983,
    "supervisor_pid": 13235,
    "lease": {
      "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
      "expires_unix": 1791644738.2573645,
      "issued_unix": 1791644648.2573645,
      "job": "intermittent_frozen_5000__52__titantpp_cnn_gru54",
      "supervisor_pid": 13235,
      "worker_group_pid": 184983
    },
    "claim": {
      "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
      "deadline_unix": 1791762424.1470754,
      "job": {
        "arm": "titantpp_cnn_gru54",
        "dataset": "intermittent_frozen_5000",
        "host": "5090",
        "id": "intermittent_frozen_5000__52__titantpp_cnn_gru54",
        "seed": 52
      },
      "maximum_concurrency": 3
    },
    "checkpoint_receipt_raw_sha256": "e06c58cf0f782b0fdb3d24c9581516f8c3a2532059c6ea1f371cfea50a02520f",
    "checkpoint_files": {
      "epoch_timing.json": "82deb4ba6b39ccf61141357f82261899d797f5e56da56dc992c74b84e8fbc83b",
      "exposure.json": "03ddcc7af664a848d7cb8c25304b183c7264b6521b56a72d2e5df62e39b29870",
      "history.json": "6bd12bb8a4defd8ada3a402b4dc0e306ae1ee2024e3c91c7ce096d158f6934b4",
      "last_epoch_state.pt": "0aa339c03a45b889f59ef64460dcb8ce0f5765d22fcf649016a88e66257be3c3",
      "train.log": "297b65f7aad161298a9c7fa61f360675aa625fc5b4ddfc2dd7bdb83ec4cd9225"
    }
  },
  {
    "job": "intermittent_frozen_5000__52__titantpp_cnn_gru54_correction_disabled",
    "worker_pid": 184990,
    "worker_ppid": 184989,
    "timeout_pid": 184989,
    "supervisor_pid": 13235,
    "lease": {
      "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
      "expires_unix": 1791644738.2573645,
      "issued_unix": 1791644648.2573645,
      "job": "intermittent_frozen_5000__52__titantpp_cnn_gru54_correction_disabled",
      "supervisor_pid": 13235,
      "worker_group_pid": 184989
    },
    "claim": {
      "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
      "deadline_unix": 1791762424.1784754,
      "job": {
        "arm": "titantpp_cnn_gru54_correction_disabled",
        "dataset": "intermittent_frozen_5000",
        "host": "5090",
        "id": "intermittent_frozen_5000__52__titantpp_cnn_gru54_correction_disabled",
        "seed": 52
      },
      "maximum_concurrency": 3
    },
    "checkpoint_receipt_raw_sha256": "8a0e611a3e164825820a1cf0167cc1d260ee3b17665c56fd96e6bf66e790acc3",
    "checkpoint_files": {
      "epoch_timing.json": "d5e41f0aac1244b4bc0f130d4994e4dbf187c42f49fb9baa855a2206d2e5c1a9",
      "exposure.json": "5089b0feaadde3ec72ed1bce3125ce6df52e4ce305ec773ca85bcf83e688eb2e",
      "history.json": "ce949b47e34e01c1edf5cf85b003690dbf427a3852379f4c3e0f3ff5c4dfddd2",
      "last_epoch_state.pt": "5fc29a235c5fbbd848d5d75804f5505314f9418d08489e24a8c678c240817d36",
      "train.log": "493a07f32c8e2c49755122c49fa24234f34af761b89ffdea470bb0f2e8847665"
    }
  },
  {
    "job": "intermittent_frozen_5000__52__titantpp_cnn_gru54_current_only",
    "worker_pid": 184996,
    "worker_ppid": 184995,
    "timeout_pid": 184995,
    "supervisor_pid": 13235,
    "lease": {
      "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
      "expires_unix": 1791644738.2573645,
      "issued_unix": 1791644648.2573645,
      "job": "intermittent_frozen_5000__52__titantpp_cnn_gru54_current_only",
      "supervisor_pid": 13235,
      "worker_group_pid": 184995
    },
    "claim": {
      "contract_sha256": "518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b",
      "deadline_unix": 1791762424.2140381,
      "job": {
        "arm": "titantpp_cnn_gru54_current_only",
        "dataset": "intermittent_frozen_5000",
        "host": "5090",
        "id": "intermittent_frozen_5000__52__titantpp_cnn_gru54_current_only",
        "seed": 52
      },
      "maximum_concurrency": 3
    },
    "checkpoint_receipt_raw_sha256": "5a8efec9b2e31eee42f823f6a44f128419cf5863ecf41c141b8aa3d60100d917",
    "checkpoint_files": {
      "epoch_timing.json": "93f68dca39c667368dc8440b8d157e5bfda18b65788f61bcdcc253c2b456f2ed",
      "exposure.json": "5089b0feaadde3ec72ed1bce3125ce6df52e4ce305ec773ca85bcf83e688eb2e",
      "history.json": "9457fcc8e80ef3a577f834c930f8b1ab61a461509e6e9ec158f9b0b00aae73ee",
      "last_epoch_state.pt": "2e48f0b073f7c4e622ebab73bc559999451adf05e8257a6392d48deca0f95c37",
      "train.log": "eaaf50245fd0503287c50614f6acc916a03a849822d4416a0d3ee56073c4e53f"
    }
  }
]
```

## 실제 관측 process argv

아래는 읽기 전용 snapshot의 실제 argv이며 실행 예시나 신규 실행 승인이 아니다.

PID 12695 / PPID 1
```text
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d -s titantpp_contribution_history_rtx_handoff_20261010_v1_5090 cd /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/source '&&' exec env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib MKL_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42 SOURCE_REVISION=57da14a2a7a5687e1bdef7d3164f3e8c0051da59 PYTHONPATH=/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/source PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/cache/matplotlib XDG_CACHE_HOME=/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/cache /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/control/pipeline.py --root /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090 --contract-sha256 518467643d248ba7384482cdb9c8806877392be0eaa81b961992cfa1a0381d6b --integrity-sha256 a2821c35778c29dea84044ad77316a5555dc834739f0e6831a8b22beea67fcca '>' /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/logs/pipeline.log '2>&1'
```

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

PID 184983 / PPID 13235
```text
timeout --signal=TERM --kill-after=15s 129599 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__52__titantpp_cnn_gru54 --deadline 1791762424.1470754
```

PID 184984 / PPID 184983
```text
/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__52__titantpp_cnn_gru54 --deadline 1791762424.1470754
```

PID 184989 / PPID 13235
```text
timeout --signal=TERM --kill-after=15s 129599 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__52__titantpp_cnn_gru54_correction_disabled --deadline 1791762424.1784754
```

PID 184990 / PPID 184989
```text
/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__52__titantpp_cnn_gru54_correction_disabled --deadline 1791762424.1784754
```

PID 184995 / PPID 13235
```text
timeout --signal=TERM --kill-after=15s 129599 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__52__titantpp_cnn_gru54_current_only --deadline 1791762424.2140381
```

PID 184996 / PPID 184995
```text
/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/operation/campaign.py --contract /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1_5090/execution_contract.json --host 5090 --mode fit --job intermittent_frozen_5000__52__titantpp_cnn_gru54_current_only --deadline 1791762424.2140381
```
