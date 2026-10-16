# CNN+GRU54 이력 보정 정기관측 — 2026-10-11 01:07:12 KST

5090의 실제 관측시각은 **2026-10-11 01:07:12 KST**다. 5080은 **2026-10-10 13:53:36 KST**의 검증된 종료 캐시를 SSH0으로 재사용했다. A100과 Instacart는 로컬 원본 SHA만 재검증했으며 원관측시각을 바꾸지 않았다.

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
| 5090 / Intermittent·52 | 이력 보정 full | 진행 중 | 24/11 | 4.76시간 (종료 E51) | 48.71시간 | 635.30초 / 10epoch |
| 5090 / Intermittent·52 | 보정 제거 | 진행 중 | 52/13 | 0.08시간 (종료 E53) | 20.37시간 | 295.66초 / 10epoch |
| 5090 / Intermittent·52 | 현재 상태 전용 | 진행 중 | 51/25 | 1.17시간 (종료 E65) | 20.90시간 | 302.13초 / 10epoch |

미시작 대기3개는 저장/best epoch와 ETA가 없다. 다른 arm·GPU의 속도를 복사하지 않았다. 전체 큐 ETA는 미확정이다. 각 fit 자체 최근 최대10 완료 epoch finite elapsed_seconds 중앙값으로 두 시나리오를 분리했다. ETA 표본 수는 각 행에 표시했다. 이후 best 갱신이나 속도 변화에 따라 달라질 수 있다. 조건36시간·host168시간 lease와 예산을 유지한다. endpoint·원본 회수·CPU 감사는 ETA와 별도다. 동시 실행 구간의 시간을 구조 효율로 해석하지 않는다.

- seed52 이력 보정 full: best 미갱신·속도 유지 시 잔여 4.76시간,300epoch 잔여 48.71시간. 관측 당시 조건 deadline까지 남은 허용시간은 31.66시간이다. deadline은 ETA가 아니다.
- seed52 이력 보정 full의 300epoch 시나리오는 조건 기한을 약 17.04시간 초과한다. 완료를 보장하지 않으며 기존 lease·기한·예산을 변경하지 않는다.
- seed52 보정 제거: best 미갱신·속도 유지 시 잔여 0.08시간,300epoch 잔여 20.37시간. 관측 당시 조건 deadline까지 남은 허용시간은 31.66시간이다. deadline은 ETA가 아니다.
- seed52 현재 상태 전용: best 미갱신·속도 유지 시 잔여 1.17시간,300epoch 잔여 20.90시간. 관측 당시 조건 deadline까지 남은 허용시간은 31.66시간이다. deadline은 ETA가 아니다.

## 이전 관측 이후의 변화

- seed52 이력 보정 full: 이전 running / 저장18 → 현재 running / 저장24,best11.
- seed52 보정 제거: 이전 running / 저장39 → 현재 running / 저장52,best13.
- seed52 현재 상태 전용: 이전 running / 저장39 → 현재 running / 저장51,best25.

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

- 관측 묶음: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1/hourly_monitor/advance_1791648433069240000.json`
- 관측 묶음 raw SHA256: `883aeb15984b80f39c2de909eb5b13f0c264f1f6e179f978226410741afe78c4`
- 5080 snapshot raw SHA256: `2b0da5b6afbf78eddfb66e02cb00be051be02c28e0fb8eb18830f7219ad21a15`
- 5090 snapshot raw SHA256: `713a1e1ac8be10e7ac9a8bf63a89c2331b89210bd9c9eeb10ad96a80e7513cd8`
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
      "expires_unix": 1791648519.851763,
      "issued_unix": 1791648429.851763,
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
    "checkpoint_receipt_raw_sha256": "7d590869406fd5b8908b13442a872b8a47a50008e9bbfdb1850459b9b08b19c0",
    "checkpoint_files": {
      "epoch_timing.json": "c745ba2569b1a13a6770f8720e093a00783f22791cce464a63a869fb4d03a11c",
      "exposure.json": "f75f7b3af33000b4c20b555a248276e5eb3371abb96afd1ac4b70eed3ab9a7d6",
      "history.json": "cd00c0cec4b5f1195099074a14dce59aea2c61637d95c786baf4b35e573a61fc",
      "last_epoch_state.pt": "3356d91e3cc08f45b0fec81ecb64492151e7cd142340072a7946b899c4aedc52",
      "train.log": "88558f0e6d07b4088e7f7d27c5f5f0e2031019009b462563aca15c04eeff850d"
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
      "expires_unix": 1791648519.851763,
      "issued_unix": 1791648429.851763,
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
    "checkpoint_receipt_raw_sha256": "2822ae05b60a2d1d461aa997fefdf27e704f67aa79ecedc216f10e93b1ddb19d",
    "checkpoint_files": {
      "epoch_timing.json": "a7d0194bf52e7731030c35394c3e032283b88e9df7a40eb40834f45baf9bcc74",
      "exposure.json": "7af71e70c0c1321b787c697c35fa5f8b9ae33d5ad1da5c64db95fefdb9e2bebb",
      "history.json": "c89c891134b4e120ba5f17face5691caac00631647834cd86abef897837f0d6b",
      "last_epoch_state.pt": "b142bc67f14053ae67102403d2f86c14b823bcbc3d30fd85cf996d54b47ebe07",
      "train.log": "2e2a856ec6ab75c43aee2ceb09365b5a0b77c822a59a40b98f84628ff29db3f6"
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
      "expires_unix": 1791648519.851763,
      "issued_unix": 1791648429.851763,
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
    "checkpoint_receipt_raw_sha256": "b55b4e08f0c3afface6f6a71039c1e951ad488d8c3b204d5a0b5a29985047930",
    "checkpoint_files": {
      "epoch_timing.json": "212143dd41fe132045c3839fbb37d60efcec79adad1255d97959253a7cdab3b0",
      "exposure.json": "199ace5708c453dc61ee5091e6a33f44b9ec1433d3ec1a7d1a0604f4db863c5b",
      "history.json": "2ba64c98a5ce20c59d000d257c879d1965f498820f5ba12617956354ca864f16",
      "last_epoch_state.pt": "ef147d1e8e19d7c2c21db5fbfd33d486e1249ea991b19bdb2a971de86b09f2dc",
      "train.log": "38be37d605e154d382e14d0d9d0428c99a456dc144a09610d2a84db580ae3db4"
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
