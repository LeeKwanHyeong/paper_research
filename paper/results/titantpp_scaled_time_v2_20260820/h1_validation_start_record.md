# TitanTPP Time Head v2 H1 Validation Safety Screening

## 상태

- 상태: 실행 준비 중
- 준비 시각: 2026-08-20 09:27:23 KST
- 실행 서버: 5080
- tmux: `inter_timehead_v2_h1_val_e300_0820`
- source revision: `12c85cdb4b06d8e55652eb7aee5cde57bd7f8ce6`
- artifact: `search_artifacts/count_aware_time_head_v2_h1_validation_e300_20260820`
- 평가 범위: Intermittent seed 42 validation only
- held-out test: 사용하지 않음

## 목적

Train-only 안정성 gate에서 선택된 H1 stable exact time head가 기존 H0 scaled exact
head의 validation Time NLL을 보존하면서 quantity MAE와 RMSE를 훼손하지 않는지
확인한다. 이 단계는 memory 구조 비교를 재개하기 전 time head의 validation 안전성을
판정한다.

## Variant 계약

| Variant | 역할 | Time head | `w * tau` budget | Intercept | 초기값 |
| --- | --- | --- | ---: | --- | --- |
| H0 | 기존 완료 결과 | scaled exact | 40 | hard clamp `±30` | `log(time_scale)` |
| H1 | validation 후보 | stable exact | 8 | smooth tanh `±6` | train mean event rate |

두 Variant는 Intermittent fixed split, TitanTPP Hard-LMM backbone, mark-free log-MSE
quantity head, seed 42, 최대 300 epoch, batch 128, learning rate `1e-3`, lookback 520주,
maximum sequence length 256, hidden dimension 64, early stopping과 checkpoint 선택을
동일하게 사용한다.

## Acceptance gate

- 모든 validation metric이 finite여야 한다.
- H1 Time NLL의 H0 대비 악화는 `0.01` 이하여야 한다.
- H1 전체 quantity MAE와 RMSE의 악화는 각각 `2%` 이하여야 한다.
- H1 `<=p95` quantity MAE의 악화는 `2%` 이하여야 한다.
- H1이 실패하면 memory 비교를 열지 않고 H0를 유지한다.

## 실행 명령어

```bash
SOURCE_REVISION=12c85cdb4b06d8e55652eb7aee5cde57bd7f8ce6 \
  bash simple_lab_test/search/scripts/run_count_aware_time_head_v2_h1_validation_e300_20260820.sh
```

## Artifact reading order

1. `launch_contract.json`
2. `logs/launcher.log`와 run-local `train.log`
3. `run_summaries.csv`
4. held-out `test_summary`: 생성되지 않아야 함
5. run-local `history.json`
6. `quantity_seed_metrics.csv`, `quantity_summary.csv`, `history_seed_metrics.csv`
7. plot: 현재 runner는 별도 plot을 생성하지 않음

## 결과

실험 완료 후 작성한다.
