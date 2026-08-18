# TitanTPP Memory Backbone CUDA and Instacart Smoke Start Record

## 상태

- 상태: dependency 보완 후 fresh 재실행 준비
- 시작 기록 시각: `2026-08-18 11:40:06 KST`
- 재실행 기록 시각: `2026-08-18 11:46:22 KST`
- 실행 서버: `5080`
- tmux session: `titan_memory_backbone_smoke_0818`
- Source revision: `8b0adb36d70db50e4dcec92df74111065766adc4`
- Artifact: `search_artifacts/count_aware_titan_memory_backbone_cuda_smoke_20260818_rerun`
- Held-out test: 미사용

## 목적

Count-aware TitanTPP의 기존 hard-LMM, no-memory, gated soft-memory,
surprise-memory가 실제 CUDA에서 forward와 backward를 완료하는지 확인한다. 이어서
Instacart top-20 fixed split에서 각 Variant를 1 epoch 학습해 checkpoint, history,
validation summary와 scale-wise artifact 생성 경로를 검증한다. 이 smoke 결과는 모델
품질 판정이나 논문 성능표에 사용하지 않는다.

## Variant 계약

| Backbone ID | Memory 차이 | 역할 |
| --- | --- | --- |
| `titantpp` | persistent token + hard top-k LMM | 기존 기준선 |
| `titantpp_no_memory` | memory 제거 | 순수 causal encoder 진단 |
| `titantpp_gated_soft_memory` | dense soft retrieval + zero-init gate | 정적 memory 후보 |
| `titantpp_surprise_memory` | causal rank-16 fast weight + chunk 32 | 동적 memory 후보 |

## 고정 조건

- Quantity interface: mark-free direct log1p-MSE
- Hidden dimension: 64
- Time/quantity head: 네 Variant 공통
- CUDA model-test: Intermittent fixed split, 2 train batches, 2 validation batches
- Instacart smoke: top-20 series, fixed split, seed 42, 1 epoch
- Instacart batch / max sequence length: 16 / 64
- Checkpoint selection: validation joint objective
- Evaluation: validation only

## 실행 계획

1. 원격 source checksum과 CUDA·데이터 preflight를 확인한다.
2. Focused pytest를 5080 환경에서 재실행한다.
3. Intermittent partial CUDA model-test 4 runs를 수행한다.
4. Instacart top-20 e1 4 runs를 수행한다.
5. Manifest, log, summary, histories, scale-wise artifact 순서로 검증한다.

## 실행 명령어

```bash
ssh 5080 '/usr/bin/tmux new-session -d -s titan_memory_backbone_smoke_0818 \
  "env SOURCE_REVISION=8b0adb36d70db50e4dcec92df74111065766adc4 \
  PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
  OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_titan_memory_backbone_cuda_smoke_20260818_rerun \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_count_aware_titan_memory_cuda_smoke_20260818.sh"'
```

## 결과

첫 실행은 학습 진입 전 pytest collection에서 종료됐다. 5080에 최근 package
centralization의 `paper/scripts/count_aware_tpp_backbone/core.py`, `training.py`,
`reporting.py`가 없었던 것이 원인이며 모델 forward는 실행되지 않았다. 실패 artifact는
보존하고 dependency를 source manifest에 추가한 fresh 경로에서 재실행한다. 완료 후
상태, 종료 시각, run 수, NaN·Inf·Traceback 여부와 artifact 검증 결과를 작성한다.
