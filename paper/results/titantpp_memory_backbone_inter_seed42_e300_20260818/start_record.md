# TitanTPP Memory Backbone Intermittent Seed-42 Screening

## 상태

- 상태: 완료, strict gate 미통과
- 준비 시각: 2026-08-18 13:30:32 KST
- 실험 시작 시각: 2026-08-18 13:36:45 KST
- 실행 서버: 5080
- tmux: `titan_memory_inter_e300_5080_0818`
- source revision: `c65c0da68d49150d5d25b8ef4665cb64b065503c`
- artifact: `search_artifacts/count_aware_titan_memory_backbone_screening_e300_20260818`
- 평가 범위: validation only
- held-out test: 미사용
- 초기 진입: fresh hard-LMM epoch 1 완료, NaN/Traceback 없음

## 목적

Count-aware TitanTPP의 time head, quantity head와 T0 log-MSE를 고정하고 memory
backbone만 바꿔 기존 hard-LMM의 실질 기여와 두 신규 memory 구조의 개선 여부를
확인한다. 이 실험에서 gate를 통과한 backbone에만 후속 T1 tail-aware loss를
적용한다.

## Variant 계약

| Backbone | Memory 구성 | 역할 |
| --- | --- | --- |
| `titantpp` | persistent token 16 + hard top-k LMM 64/4 | fresh control |
| `titantpp_no_memory` | persistent token과 LMM 제거 | hard-LMM 기여 진단 |
| `titantpp_gated_soft_memory` | dense soft retrieval + zero-init residual gate | 정적 memory 후보 |
| `titantpp_surprise_memory` | causal rank-16 fast weight + chunk 32 + zero-init gate | 동적 memory 후보 |

비교 축 이외의 데이터, encoder width/depth, time head, quantity head, loss, optimizer,
seed, checkpoint selection은 동일하게 유지한다.

## 고정 조건

| 항목 | 값 |
| --- | --- |
| Dataset | Intermittent frozen-5000 fixed split |
| Quantity objective | T0 log1p quantity MSE |
| Epoch / seed | 최대 300 / 42 |
| Early stopping | 최소 40 epoch, patience 40 |
| Batch / LR | 128 / 1e-3 |
| Context | lookback 520주, max sequence length 256 |
| Hidden / layers / heads | 64 / 2 / 4 |
| Selection | best validation joint objective |
| Evaluation | validation only; held-out test 미사용 |

## Acceptance Gate

- Fresh hard-LMM 대비 전체 MAE 또는 RMSE가 5% 이상 개선되어야 한다.
- `<=p95` 수량 MAE 악화는 2% 이하여야 한다.
- time NLL 악화는 0.01 이하여야 한다.
- 모든 metric과 학습 값은 finite여야 한다.
- 여러 후보가 통과하면 validation quantity MAE, RMSE, joint objective 순으로 선택한다.
- 아무 후보도 통과하지 못하면 hard-LMM을 유지한다.

## 실행 명령어

```bash
tmux new-session -d -s titan_memory_inter_e300_5080_0818 \
  "cd /home/leekwanhyeong/workspace/paper_research && \
  SOURCE_REVISION=c65c0da68d49150d5d25b8ef4665cb64b065503c \
  PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
  EXECUTION_ROLE=primary_5080_seed42_validation_screening \
  OUTPUT_ROOT=/home/leekwanhyeong/workspace/paper_research/search_artifacts/count_aware_titan_memory_backbone_screening_e300_20260818 \
  bash simple_lab_test/search/scripts/run_titantpp_memory_backbone_screening_e300_20260818.sh"
```

## Artifact Reading Order

1. `source_manifest.txt`와 `launch_contract.json`
2. `logs/launcher.log`와 run별 `train.log`
3. `run_summaries.csv`
4. test summary 미생성 및 held-out lock 확인
5. run별 `history.json`과 `history_summary.csv`
6. `quantity_seed_metrics.csv`와 `quantity_summary.csv`
7. 생성된 plot이 있으면 마지막에 확인

## 결과

- 완료 runs: 4/4
- 원래 acceptance gate 선택: `titantpp` (hard-LMM 유지)
- quantity 기준 후보: `titantpp_surprise_memory`
- Surprise는 hard-LMM 대비 validation quantity MAE 23.24%, RMSE 11.35%,
  log1p quantity MSE 38.14%를 개선했다.
- Time NLL은 0.03869 악화되어 사전 정의 gate를 통과하지 못했다.
- 해당 Time NLL 차이의 99.98%는 `history > 128` 구간에서 발생했다.
- 공통 time head의 모든 validation target이 `w * delta_t` clamp 구간에 포함되는
  수치 계약 문제가 확인되어, 현재 Time NLL은 확률적 NLL로 최종 해석하지 않는다.
- held-out test는 사용하지 않았다.
- 상세 분석: [result.md](result.md)
