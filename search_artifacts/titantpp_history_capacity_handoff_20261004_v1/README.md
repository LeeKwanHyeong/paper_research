# Intermittent seed62 width8/12 운영 이관

5090에서 시작하지 않은 seed62 width8/12 두 조건을 전용 5080 디렉터리에서 실행한다. 원래 5080의 완료 12조건과 5090의 진행 중 학습은 그대로 보존한다. 이 문서의 원격 명령은 검토 가능한 실행 방법이며, 로컬 준비 작업이 원격 실행을 수행하지 않는다.

- 부모 계약: `8434041a6a301995a8d22baa77d5ca0da4eb2cd942d091a8aa8cef379406c48d`
- 과학 소스 114개: `41a4a4ff79e5e17d907b9ee8966723004516ee2e42608d31c7f1a871f0acd3e1`
- 운영 계약: `e98a59a9f32c04be44f61eba0653920a564b6696ed0be134cb9a07b518603f80`
- 전용 5080 경로: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_capacity_handoff_20261004_v1_5080`
- 원래 5090 경로: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_width8_12_dual_20261004_v1_5090`
- 절대 마감: **2026-10-11 15:41:55.994061 KST**, 부모 시작 시각부터 604800초를 그대로 상속한다. 각 조건은 36시간과 남은 전체 시간 중 작은 값으로 제한한다.
- 범위: Intermittent seed62의 두 width, fresh fit, Train/Validation only, 학습 재시도 없음. 실제 GPU와 Runtime은 5080으로 기록한다. 서로 다른 GPU의 시간으로 논문 효율성을 비교하지 않는다.

`handoff_contract.json`에는 정확한 변경 경로와 전후 값이 있다. 변경은 새 실행 경로, 두 job의 실행 host, 해당 참조의 host/path, 데이터 path 및 그 path를 포함한 dataset 구조 SHA에 한정한다. 데이터 파일 SHA, population, 모델, loss, optimizer, checkpoint 선택, 114개 원본 바이트와 부모 계약은 변경하지 않는다.

**로컬 구현과 검증 — 완료**

`paper/scripts/capacity_handoff_runtime.py`는 표준 라이브러리만으로 초기 진입한다. source SHA와 부모/운영 계약을 검증한 뒤 원본 엔진의 `qualify`와 `run_fit`을 그대로 호출한다. Qualification에 한해서 Intermittent와 seed62를 필터링하며, 원본 계약 검증 시에는 원래 전체 host/seed 범위를 복구한다. 사용자 승인 문구는 별도 승인 기록의 원문을 그대로 사용한다.

```sh
python -m pytest paper/tests/test_capacity_handoff_runtime.py -q
python -m py_compile paper/scripts/capacity_handoff_runtime.py search_artifacts/titantpp_history_capacity_handoff_20261004_v1/control/prepare.py
```

경합, 부분 예약, 중복 시작, Runtime/소스/학습 변경, 원본 terminal SHA, 절대 마감, 두 조건만 실행, 읽기 전용 관측을 로컬에서 검증한다. 원격 qualification과 이관 실행은 별도 증적이 있기 전에는 미확인이다.

**전용 5080 패키지 배치와 native qualification — 다음 작업**

로컬 생성기는 SSH나 GPU 호출을 하지 않는다. 이미 생성된 파일은 내용이 같아야 재사용한다.

```sh
python search_artifacts/titantpp_history_capacity_handoff_20261004_v1/control/prepare.py --user-instruction '승인 기록의 실제 사용자 원문'
```

생성 결과의 `deployment.tar.gz`와 `deployment_manifest.json`을 사용하여 **새 전용 경로만** 생성·전송·추출하고 모든 manifest SHA를 확인한다. 기존 5080 완료 폴더에는 배치하지 않는다. 새 root에 파일을 추출한 후 다음 command를 root가 승인 범위 안에서 실행한다.

```sh
cd /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_capacity_handoff_20261004_v1_5080/source
timeout --signal=TERM --kill-after=15s 5400 env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/home/leekwanhyeong/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib MKL_NUM_THREADS=4 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONHASHSEED=42 SOURCE_REVISION=3bec316f2eb0319cf52e062dcf17b3f9518dcb7a MPLCONFIGDIR=/tmp/titantpp-capacity-mpl /home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12 ../control/capacity_handoff_runtime.py --bundle /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_capacity_handoff_20261004_v1_5080 --mode qualify
```

Native qualification은 원래 seed62 width4 checkpoint의 역사적 initialization, 동일 입력/Train·Validation population, 원래 전체 Validation replay, 각 width의 synthetic gradient·메모리 검증을 실행한다. 이 관문이 통과하기 전에 source 예약이나 destination permit을 만들지 않는다.

**5090 두 조건 예약과 5080 실행 허가 — 다음 작업**

5090의 원래 root 아래 `handoff_seed62/prepared/`에 제어 JSON 8개, `control/capacity_handoff_runtime.py`, 새 5080의 `qualification/receipt.json` 및 `qualification/intermittent_frozen_5000__62_baseline_diagnostic.json`을 SHA 확인 후 복사한다. 원래 114 source는 새 staging에 복사하지 않으며 예약 검증은 원래 5090 source를 읽는다.

```sh
python3 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_width8_12_dual_20261004_v1_5090/handoff_seed62/prepared/control/capacity_handoff_runtime.py --bundle /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_width8_12_dual_20261004_v1_5090/handoff_seed62/prepared --mode reserve --source-root /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_width8_12_dual_20261004_v1_5090
```

예약은 source의 `claims/<job>.json` 두 개를 O_EXCL로 생성한다. 원래 source가 claim을 먼저 획득하면 예약이 실패하고 destination 허가를 발급하지 않는다. 첫 claim만 작성된 부분 상태는 유지하며 자동 rollback 또는 재시도하지 않는다. 전체 예약 완료 후 같은 명령을 다시 호출하면 기존 claim의 내용과 SHA를 검증하고 같은 receipt를 반환한다. 현재 5090 fit을 정지하거나 lease/status를 바꾸지 않는다. 원래 dispatcher는 seed42·52의 네 조건을 마친 뒤 seed62 첫 예약 claim에서 의도한 `FileExistsError`로 종료된다.

완료된 `handoff_seed62/reservation.json`을 5080 새 root의 `source_reservation.json`으로 회수·전송한 뒤:

```sh
python3 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_capacity_handoff_20261004_v1_5080/control/capacity_handoff_runtime.py --bundle /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_capacity_handoff_20261004_v1_5080 --mode permit
```

허가는 5080 자신의 native receipt와 두 source claim의 원본 SHA에 결합한다. 위 qualification과 같은 cwd/interpreter/environment에서 `--mode dispatch`를 실행한다. 외부 `timeout`은 부모 절대 마감까지의 남은 초로 제한하고, 별도 tmux 이름 `titantpp_history_capacity_handoff_20261004_v1_5080`을 사용한다. dispatcher는 지정된 두 조건만 실행하며 각 fit은 별도 process group·native timeout·90초 server lease를 가진다. 자동 retry/resume은 없다.

**관측과 증적 통합 — 다음 작업**

```sh
python3 NEW_ROOT/control/capacity_handoff_runtime.py --bundle NEW_ROOT --mode observe
python3 SOURCE_ROOT/handoff_seed62/prepared/control/capacity_handoff_runtime.py --bundle SOURCE_ROOT/handoff_seed62/prepared --mode observe-source --source-root SOURCE_ROOT
```

`observe`는 기존 snapshot과 호환되는 `host/root/contract_sha256/source_closure_sha256/files/file_sha256/runs/gpu/compute/ps_returncode/processes`를 반환하며, 새 두 조건만 포함한다. named Validation JSON과 binary 존재/크기만 읽고 binary SHA나 CPU replay는 실행하지 않는다. 마감 뒤에도 읽을 수 있다. fit argv는 pinned 5080 Python + `NEW_ROOT/control/capacity_handoff_runtime.py --bundle NEW_ROOT --mode fit --job ID --deadline UNIX`이다.

`observe-source`는 기존 5090 SSH 관측 안에서 호출할 수 있는 보충 함수다. 예약 receipt와 두 claim을 실제 내용·원본 SHA로 검증하고, 네 source terminal manifest admission과 정확한 예약 `FileExistsError`가 확인되었을 때만 `intentional_queue_stop_verified/source_effective_complete`를 true로 기록한다. 실제 fit failure를 행정 종료로 바꾸지 않는다. 부모의 기존 5080 완료 12조건 증적은 보존하되 새 두 조건 때문에 시간별 monitor에서 과거 완료 cache만으로 5080 전체 완료를 선언하면 안 된다. root가 전체 18개의 canonical job ID를 한 번씩 집계하도록 monitor를 갱신한다.
