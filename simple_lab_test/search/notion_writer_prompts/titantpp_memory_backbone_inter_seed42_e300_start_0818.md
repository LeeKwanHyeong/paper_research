# Notion Update Source: TitanTPP Memory Backbone Seed-42 Screening

작성 위치: `5. Model Design Enhancement`

## 2026-08-18

### TitanTPP Memory Backbone Intermittent Seed-42 Screening

#### 상태

- 실험 중
- 실험 시작 시각: 2026-08-18 13:36:45 KST
- 실행 서버 / tmux: 5080 / `titan_memory_inter_e300_5080_0818`
- Artifact: `search_artifacts/count_aware_titan_memory_backbone_screening_e300_20260818`
- Fresh hard-LMM epoch 1 진입과 finite loss를 확인했으며 이후 자동 polling은 하지 않는다.

#### 목적

Time head와 quantity head, T0 log-MSE를 그대로 둔 상태에서 TitanTPP memory
backbone만 바꿔 기존 hard-LMM의 기여와 신규 memory 구조의 개선 여부를 확인한다.
이 단계에서 validation 기준을 통과한 backbone에만 T1 tail-aware loss를 적용한다.

#### Variant 계약

| Backbone | 차이 | 역할 |
| --- | --- | --- |
| hard-LMM | persistent token과 hard top-k LMM 사용 | fresh 기준선 |
| no-memory | 두 memory 경로 제거 | 기존 memory 기여 진단 |
| gated-soft | dense soft retrieval과 학습형 residual gate | 정적 memory 후보 |
| optimized-surprise | 관측 이력으로 갱신되는 causal fast weight | 동적 memory 후보 |

네 Variant는 같은 데이터, Titan encoder 크기, time/quantity head, T0 loss, optimizer,
seed와 checkpoint selection을 사용한다.

#### 고정 조건

| 항목 | 값 |
| --- | --- |
| Dataset | Intermittent frozen-5000 fixed split |
| Epoch / seed | 최대 300 / 42 |
| Early stopping | 최소 40 epoch, patience 40 |
| Batch / LR | 128 / 1e-3 |
| Context | lookback 520주, max sequence length 256 |
| Model size | hidden 64, 2 layers, 4 heads |
| Quantity loss | T0 log1p quantity MSE |
| Selection | best validation joint objective |
| Evaluation | validation only; held-out test 미사용 |

#### 실행 명령어

```bash
SOURCE_REVISION=c65c0da68d49150d5d25b8ef4665cb64b065503c \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
EXECUTION_ROLE=primary_5080_seed42_validation_screening \
OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_titan_memory_backbone_screening_e300_20260818 \
bash simple_lab_test/search/scripts/run_titantpp_memory_backbone_screening_e300_20260818.sh
```

#### 결과
