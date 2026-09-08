# BOUNDED-QK 5090 실행 기록

2026-09-08 · `paper_research / codex/hard-lmm-causal-qkv`

## 현재 기준선과 실행 목적

모델 구현 `be92f9a`를 유지한 채 GPU 실행·감사·중단 순서를 추가했다.
첫 encoder block의 Q/K causal residual 크기를 head별로 제한하고 기존 causal-V
연산을 유지하는 단일 후보다. Loss, raw-RMSE selector, B/FULL의 기존 결과와
과거 FULL의 실패 판정은 변경하지 않는다.

실행 계약은 `paper/contracts/hard_lmm_bounded_qk_screening_v1.json`이다.
새 모델과 기존 B/FULL의 수량 비교, 동일 K=1 head를 사용한 정상화 시간 비교를
구분한다. 신규 B 학습·B 시간 head 재학습·추가 seed·held-out 평가는 실행하지 않는다.

## 완료 / 로컬·5090 읽기 전용 준비

- `runtime_inventory.json`: Runtime/GPU/기존 artifact/세 data·split 검사 **43/43 통과**.
- 5090: `RTX5090-server`, RTX 5090, Python3.12.13, PyTorch2.11.0+cu130,
  CUDA13.0. Python은 `/opt/miniconda3/envs/ai_env/bin/python`을 사용한다.
  Runtime과 패키지를 설치하거나 재배포하지 않는다.
- 세 raw B와 세 FULL checkpoint·summary·history, 세 aligned-B checkpoint·summary를
  합친 **24개 파일, 4,242,642 bytes**를 별도 소스 패키지에 포함한다. 기존 benchmark
  데이터는 checksum 검증 후 읽고, 약 900MiB의 aligned-B hidden cache는 전송하지 않는다.
- `local_tests.txt`, `local_verification.json`: **163 passed, CUDA 1 skipped**.
  `be92f9a` archive에 이번 소스만 복사해 검증했다. 별도 `CountAwareTPP.py` 변경과
  미추적 `scripts/`, `.agents/`는 포함하지 않았다.
- 캠페인 파일 `run_hard_lmm_bounded_qk_campaign.py`는 전체 committed source
  SHA와 source population, Runtime·GPU UUID·data·checkpoint·history를 검증한다.
  실행 manifest와 각 job의 receipt를 해당 출력이 생기기 전에 저장한다.
- 새 `profile_hard_lmm_bounded_qk.py`는 B/FULL/BOUNDED × L8/64/256 × 반복3,
  총 27개 독립 CUDA worker를 사용한다. Batch128, hidden64, warmup5, measured15,
  FP32를 고정한다. 새 후보/B step≤1.5배·peak allocated memory≤1.25배를 각 길이에
  적용하며 FULL 대비 비용도 보고한다. 비용 결과를 본 뒤 기준을 바꾸지 않는다.

## 고정된 실행 순서

1. **CUDA 계약 테스트와 비용 측정**: 모델·gradient·optimizer 복원 및 dtype 계약을
   CUDA에서 확인한다. 실행 오류 또는 비용 기준 미달이면 후속 queue를 중단한다.
2. **BOUNDED full-data e1**: Taxi → Intermittent → Instacart 순서로 처리 건수,
   target identity, 실제 lag 학습, selected/last/optimizer strict restore를 감사한다.
   e1 수치에는 성능 gate를 적용하지 않는다.
3. **기존 FULL의 정상화 head full-data e1**: 같은 세 데이터셋 순서로 자기 hidden
   cache를 생성하고 K=1 head 1 epoch를 실행한다. Cache와 head 저장·복원, 수량
   bitwise identity를 검사한다. 이 단계도 성능 판정에 사용하지 않는다.
4. **Seed42 screening**: Instacart → Taxi → Intermittent 순서다. 후보 Backbone을
   max300/min40/patience40으로 학습한 후 B/FULL 수량 gate를 적용한다. 통과하면
   FULL의 K=1 full fit과 후보의 K=1 e1/full fit을 수행해 정상화 시간 gate를 적용한다.
   어느 데이터셋에서든 공통 gate에 실패하면 뒤의 학습을 시작하지 않는다.

| 판정 | 사전 고정 조건 |
|---|---|
| B 대비 수량 개선 | raw RMSE strictly 감소; 전체 MAE≤+1%, body/>p99 MAE 각각≤+2% |
| FULL 이득 보존 | raw RMSE·전체 MAE 각각≤+1%; body/>p99 MAE 각각≤+2% |
| 공통 시간 guardrail | candidate normalized NLL≤aligned-B+0.01 |
| Taxi 시간 일반화 주장 | 같은 K=1 조건에서 FULL보다 NLL≥0.005 개선; 이 주장만 실패하면 queue를 중단하지 않음 |

Time median MAE/RMSE도 함께 기록한다. Legacy time score는 보조 보고만 하며
정상화 NLL과 숫자를 직접 비교하거나 이번 채택 gate에 사용하지 않는다.

## GPU 실행 상태와 증적

**진행 중 / 5090**: 14:31:39 KST부터 Instacart seed42 screening을 실행 중이다.
후보 Backbone과 FULL 정상화 시간 head의 full-data e1은 각각 세 데이터셋 모두
통과했다. 아직 seed42 결과에 대한 성능 판정을 하지 않았다.

14:34 KST에 첫 epoch 완료를 확인했다. Train 1,991,192건을 처리했고 모든 학습
값이 유한했으며 첫 history row는 원본 e1과 정확히 일치했다
(`seed42_initial_progress.json`). 이 기록은 실행 정상성 확인이며 성능 채택
판정에 사용하지 않는다.

학습 source는 **`89cd700c28dd169cd59bfccbd694ef8967bac823`**이며, 복구 실행기는
**`91c00899e61572bf84884e048a4165fda0c98a26`**이다. 후자는 실행 재개와 감사만
담당한다. 새 실행은 14:29:46 KST에 시작했고 `launch.json`, `recovery_manifest.json`,
`recovery_deployment_receipt.json`에 별도 경로·SHA·명령을 기록했다. 완료된 CUDA·
비용·후보 e1을 재감사해 재사용하고, 실패한 시간 head의 cache는 재사용하지 않았다.
학습은 `max300/min40/patience40`이며 Instacart → Taxi → Intermittent 순서를 유지한다.

`normalized_e1_recovery_audit.json`과 `remote_recovery1/`에는 완료한 NVRTC 확인 및
FULL 정상화 e1 세 건의 소형 증적을 회수했다. 원격·로컬 SHA 18/18이 일치하며,
세 데이터셋의 전체 처리 건수, 학습 가능한 시간 head 130개 파라미터, 수량 예측
bitwise identity, selected restore 감사와 held-out 미사용을 확인했다.

원본 source89 실행은 13:46:47 KST에 별도 경로에서 시작했다. CUDA **67/67**과 비용 **27 worker**가
통과했다. 로컬에서도 비용 비율을 원 측정값으로 재계산했다
(`cuda_cost_local_audit.json`).

후보 Backbone의 Taxi·Intermittent·Instacart full-data e1도 세 건 모두 통과했다.
회수한 selected/last checkpoint, history, receipt를 동일 source의 CPU 감사로
재검증했다. `e1_local_audit.json`의 독립 검사 9/9가 통과했다. e1은 실행·복원
계약의 증적이며 성능 채택 판단에 사용하지 않는다.

| e1 데이터셋 | train 처리 수 | validation 처리 수 | peak allocated bytes | 전체 e1 실행 초 |
|---|---:|---:|---:|---:|
| Taxi | 38,393 | 8,268 | 2,009,344,512 | 8.86 |
| Intermittent | 393,824 | 86,285 | 2,009,344,512 | 73.32 |
| Instacart | 1,991,192 | 503,733 | 404,801,024 | 159.88 |

실행 초는 각 summary의 `elapsed_seconds`이며 초기화와 결과 저장을 포함한다.
장기 학습 종료 시간은 seed42의 실제 epoch 진행 속도와 조기 종료 시점을 함께
확인해야 한다. 재사용 대상 46개 파일은 원격·로컬 SHA가 모두 일치한다
(`reuse_artifact_audit.json`).

| 사건 길이 | 후보/B step | 후보/B peak allocated | 후보/FULL step |
|---:|---:|---:|---:|
| 8 | 1.292 | 1.083 | 1.100 |
| 64 | 1.291 | 1.105 | 1.102 |
| 256 | 1.103 | 1.081 | 1.032 |

위 값은 합성 입력의 실행 비용이며 예측 성능이 아니다. 세 길이 모두 시간≤1.5,
메모리≤1.25의 기존 B 대비 한도를 충족한다. 실제 모델 파라미터 수는
B 89,795개, FULL과 후보 각각 90,371개다.

13:52:22 KST에 첫 `normalized_e1_FULL_yellow_trip_hourly`가 CUDA JIT의
`libnvrtc-builtins.so.13.0` 로딩 오류로 중단됐다. 시간 head의 epoch 0 평가
도중 발생한 실행 오류이며 seed42 학습이나 성능 gate 실패가 아니다.
`attempt_89cd700_nvrtc/`에 원본 manifest·status·launch와 원본 오류 로그 archive를
보존한다. 설치된 CUDA 13 라이브러리 경로를 해당 프로세스에만 지정한 합성
float64 `log_ndtr`와 gradient 검증은 통과했다. 모델·loss·selector·패키지 변경
없이, 완료 증적을 재검증하여 재사용하는 별도 복구 실행을 시작했다.

`nvrtc_runtime_recovery_probe.json`은 실제 5090에서 연속·정수·상한 관측의
float64 likelihood와 gradient가 유한함을 확인했다. CPU/CUDA 출력 차이는 최대
5.15e-14, gradient 차이는 0이었다. 정수 PMF와 나머지 survival의 합은 1.0이며
Taxi 첫 구간과 Instacart 30일 survival 정의도 통과했다. 같은 source89의 합성
B/FULL/BOUNDED 실행을 경로 지정 전후의 독립 프로세스에서 비교한 결과 출력·
전체 parameter gradient SHA가 모두 bitwise 일치했다. Backbone의 실제 NVIDIA
라이브러리 로딩 경로 목록도 같았다. 이 검증은 기존 CUDA·비용·e1 증적의 재사용을
뒷받침하며, 장기 학습의 성능 개선을 뜻하지 않는다.

첫 source `a18614b`는 13:40 KST에 시작했고 CUDA 66개 통과 후 optimizer
상태를 서로 다른 장치에서 비교하는 테스트 오류 1개로 중단됐다. 복원 후 모델
파라미터는 bitwise 일치했으며, 비교 대상의 AdamW step scalar만 CPU/CUDA 위치가
달랐다. 정확한 값 비교를 CPU에서 수행하도록 테스트를 수정했다. 모델·loss·selector
및 비용 기준은 유지했다. 이 시도에서는 비용 측정과 실제 데이터 학습을 시작하지
않았다. 실패 증적은 `attempt_a18614b/diagnosis.json` 및 원격 첫 실행 경로에 보존한다.

이 문서를 처음 커밋한 시점에는 GPU 실행 전이다. 실제 source commit·별도 경로·
archive SHA·시작 명령과 사전 manifest는 `deployment_manifest.json`, `launch.json`에
기록한다. 진행/완료 상태는 원격 `campaign_status.json`과 회수한 상태 snapshot을
기준으로 갱신한다. Source는 독립 경로에서 고정하고 결과는
`/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/` 아래에 저장한다.
`paper_research/develop`, `paper_research/master` 병합 또는 push는 포함하지 않는다.

## 실행 후 남는 작업

- **진행 중 / 5090**: Instacart seed42 screening. 기존 B/FULL 수량 gate를 통과하면
  동일 K=1 조건의 정상화 시간 평가를 이어간다.
- **다음 작업 / 5090**: Instacart의 공통 수량·시간 기준이 모두 통과할 때만 Taxi,
  이어서 Intermittent를 같은 순서와 기준으로 진행한다. 실패하면 뒤의 학습을 중단한다.
- **진행 중 / 시간별 Scheduler**: 상태·GPU/tmux 생존·현재 history를 읽고, 의미 있는 변화나 완료,
  오류가 있을 때 알린다. 실행 오류나 성능 기준 미달 뒤 자동 재시도는 하지 않는다.
  `monitor.json`에 갱신된 자동화와 후속 처리 범위를 기록했다.
- **다음 작업 / 로컬**: 완료 artifact를 회수해 동일 source/data/selector/quantity identity를 재감사한다.
  세 데이터셋 seed42 통과는 추가 검토 근거이며 최종 채택이나 전 모델 대비 우위를
  뜻하지 않는다. 추가 seed와 held-out은 이번 queue에 포함하지 않는다.
