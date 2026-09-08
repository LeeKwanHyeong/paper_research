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

이 문서를 처음 커밋한 시점에는 GPU 실행 전이다. 실제 source commit·별도 경로·
archive SHA·시작 명령과 사전 manifest는 `deployment_manifest.json`, `launch.json`에
기록한다. 진행/완료 상태는 원격 `campaign_status.json`과 회수한 상태 snapshot을
기준으로 갱신한다. Source는 독립 경로에서 고정하고 결과는
`/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/` 아래에 저장한다.
`paper_research/develop`, `paper_research/master` 병합 또는 push는 포함하지 않는다.

## 실행 후 남는 작업

- CUDA·비용·e1이 통과하면 승인된 queue가 seed42를 자동으로 이어서 실행한다.
- 시간별 확인은 상태·GPU/tmux 생존·현재 history를 읽고, 의미 있는 변화나 완료,
  오류가 있을 때 알린다. 실행 오류나 성능 기준 미달 뒤 자동 재시도는 하지 않는다.
- 완료 artifact를 회수해 동일 source/data/selector/quantity identity를 재감사한다.
  세 데이터셋 seed42 통과는 추가 검토 근거이며 최종 채택이나 전 모델 대비 우위를
  뜻하지 않는다. 추가 seed와 held-out은 이번 queue에 포함하지 않는다.
