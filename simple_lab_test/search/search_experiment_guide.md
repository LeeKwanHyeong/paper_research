# Search Experiment Guide

이 문서는 실제 실험을 실행할 때 참고하는 레시피 중심 가이드입니다. 구현 상세와
configuration 위치는 `search_experiment_info.md`를 확인합니다.

## One CLI Rule

현재 반복 실험의 기본 진입점은 하나입니다.

```bash
python simple_lab_test/search/tpp_experiment.py {subcommand} ...
```

지원 subcommand:

| subcommand | 목적 |
| --- | --- |
| `model-test` | synthetic batch로 RMTPP/TitanTPP/TransformerHawkesTPP interface 검증 |
| `long-epoch` | 장기 학습, best validation NLL checkpoint, scale-wise quantity error |
| `overfit` | capacity/overfitting diagnostic |
| `qty-ablation` | quantity supervision loss ablation |
| `yellow-resolution` | yellow-trip daily/hourly event sequence benchmark |

## 1. Model Interface Test

새 모델을 추가하거나 THP config를 수정한 뒤 가장 먼저 실행합니다. 실제 parquet를
읽지 않고 synthetic batch로 `forward`, `nll`, `mark_head`, `value_head`,
`sample_next_dt`, `reconstruct_qty`가 모두 finite인지 확인합니다.

THP 단독:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models TransformerHawkesTPP \
  --thp-candidates small \
  --device cpu \
  --left-pad
```

세 모델 동시 확인:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm \
  --thp-candidates small \
  --device cpu \
  --left-pad
```

결과:

```text
search_artifacts/tpp_model_test/
  model_test_summary.csv
  model_test_summary.json
```

## 2. Long-Epoch Main Comparison

논문/본실험 관점의 핵심 비교는 `long-epoch`입니다. RMTPP, TitanTPP,
TransformerHawkesTPP를 같은 marked dataset, split, value reconstruction metric에서
비교합니다.

장기 GPU 실행 중 프로세스가 끊겨도 같은 `--base-dir`, `--datasets`, `--models`,
`--epochs`, `--seeds`로 다시 실행하면 완료된 run은 skip되고, 미완료 run은
`checkpoints/last_epoch_state.pt`에서 마지막 완료 epoch 다음부터 이어집니다.
단, `--force-rerun`을 붙이면 resume하지 않고 처음부터 다시 학습합니다.

Marked target에서 RMTPP/TitanTPP/THP 비교:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/marked_target_long_epoch_with_thp \
  --datasets intermittent \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm \
  --thp-candidates small,base \
  --epochs 300 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --eval-selections best_val_nll,best_score,final
```

Marked target에서 Titan 후보를 더 비교:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/marked_target_titan_candidate_sweep \
  --datasets intermittent \
  --models rmtpp,titantpp \
  --titan-candidates small_deep_lmm,mid_lmm \
  --rmtpp-hidden-dim 64 \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --eval-selections best_val_nll,best_score,final
```

Yellow-trip weekly 기준 비교:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/yellow_trip_weekly_long_epoch_with_thp \
  --datasets yellow_trip \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm \
  --thp-candidates base \
  --epochs 300 \
  --seeds 42,52,62 \
  --lr 1e-3
```

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/marked_target_all_models_e800 \
  --datasets intermittent \
  --models thp,rmtpp,titantpp \
  --titan-candidates small_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/yellow_trip_weekly_all_models_e300 \
  --datasets yellow_trip \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm \
  --thp-candidates base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 256 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```


해석 기준:

| 지표 | 의미 |
| --- | --- |
| `best_val_nll` | 교수님이 말한 sweet spot 기준. 낮을수록 좋음 |
| `best_val_nll_qty_mae` | best NLL checkpoint에서 quantity MAE |
| `best_val_nll_mark_acc` | mark prediction accuracy |
| `best_val_nll_dt_mae` | next delta-time MAE |
| `scale_wise_summary.csv` | true quantity scale별 MAE/WAPE/median AE |

## 3. Overfitting Diagnostic

목적은 성능표를 예쁘게 만드는 것이 아니라, 모델이 train distribution을 충분히
학습할 수 있는지 확인하는 것입니다. 일부 고용량 설정에서 train loss가 내려가고,
validation NLL이 어느 시점 이후 나빠지면 학습 자체는 가능한 것으로 해석합니다.

Marked target small stress:

```bash
python simple_lab_test/search/tpp_experiment.py overfit \
  --base-dir search_artifacts/tpp_overfit_marked_target \
  --datasets intermittent \
  --models rmtpp,titantpp \
  --epochs 100 \
  --lr 1e-3 \
  --seeds 42 \
  --max-seq-lens 16 \
  --rmtpp-rnn-types rnn,gru,lstm \
  --rmtpp-hidden-dims 64,128,256 \
  --titan-candidates small_no_lmm,small_lmm \
  --force-rerun
```

Yellow-trip full long preset:

```bash
python simple_lab_test/search/tpp_experiment.py overfit \
  --preset yellow_trip_full_long
```

Yellow-trip subset stress preset:

```bash
python simple_lab_test/search/tpp_experiment.py overfit \
  --preset yellow_trip_subset_stress
```

확인 파일:

```text
paper_outputs/overfit_diagnostic_report.md
leaderboard/overfit_runs.csv
leaderboard/overfit_summary.csv
leaderboard/overfit_histories.csv
paper_outputs/plots/
```

## 4. Quantity Loss Ablation

현재 magnitude-factorized formulation은 quantity 자체가 아니라 `scale_residual`을
주로 학습합니다. 따라서 복원식 `qty = base^(mark + residual)` 때문에 log-space
오차가 quantity-space에서 크게 증폭될 수 있습니다.

비교 모드:

| loss mode | 학습 목적 |
| --- | --- |
| `residual_only` | mark CE + time NLL + residual Huber |
| `hybrid` | residual_only + direct quantity loss |
| `qty_only` | mark CE + time NLL + direct quantity loss |

TitanTPP만 ablation:

```bash
python simple_lab_test/search/tpp_experiment.py qty-ablation \
  --base-dir search_artifacts/tpp_qty_loss_ablation \
  --datasets intermittent,yellow_trip \
  --models titantpp \
  --loss-modes residual_only,hybrid,qty_only \
  --epochs 30 \
  --seeds 42,52,62 \
  --force-rerun
```

확인 파일:

```text
paper_outputs/qty_loss_analysis_summary.md
paper_outputs/paper_table_metrics.csv
paper_outputs/paper_table_deltas.csv
paper_outputs/paper_table_best_modes.csv
paper_outputs/plots/
```

## 5. Yellow-Trip Resolution Benchmark

기존 weekly yellow-trip은 series 길이가 짧아 RMTPP가 강하게 보일 수 있습니다.
TitanTPP 같은 expressive encoder가 충분한 sequence context를 활용하는지 보려면
daily/hourly event sequence를 따로 확인합니다.

Daily/hourly benchmark:

```bash
python simple_lab_test/search/tpp_experiment.py yellow-resolution \
  --base-dir search_artifacts/yellow_trip_resolution_ab_test \
  --resolutions daily,hourly \
  --models rmtpp,titantpp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --epochs 100 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --max-series 1000
```

Hourly long run:

```bash
python simple_lab_test/search/tpp_experiment.py yellow-resolution \
  --base-dir search_artifacts/yellow_trip_resolution_ab_test_hourly_e800 \
  --resolutions hourly \
  --models rmtpp,titantpp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --max-series 0 \
  --hourly-min-active-buckets 72 \
  --hourly-lookback-buckets 168 \
  --max-seq-len 256
```

## 6. Still-Standalone Scripts

`tpp_experiment.py`로 통합하지 않은 standalone script도 남아 있습니다.

| 스크립트 | 언제 쓰는가 |
| --- | --- |
| `compare_log_bases_distribution.py` | raw demand의 `log10/log4/log2` mark 분포만 볼 때 |
| `titan_hparam_search.py` | TitanConfig와 scale base 후보를 넓게 탐색할 때 |
| `titan_rmtpp_ab_test.py` | 기존 30 epoch RMTPP vs TitanTPP A/B 결과를 재현할 때 |

## 7. Result Reading Order

1. `*_manifest.json`에서 effective dataset/model config 확인
2. `leaderboard/*runs.csv`에서 실패 run과 seed별 raw metric 확인
3. `leaderboard/*summary.csv`에서 seed 평균 비교
4. `leaderboard/*histories.csv`에서 train/validation curve 확인
5. `leaderboard/*scale_wise*.csv`에서 quantity scale별 오류 확인
6. `paper_outputs/*.md`와 `paper_outputs/plots/`를 보고 회의/논문용 해석 작성
