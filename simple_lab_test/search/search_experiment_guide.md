# Search Experiment Guide

이 문서는 실제 실험을 실행할 때 참고하는 레시피 중심 가이드입니다. 구현 상세와
configuration 위치는 `search_experiment_info.md`를 확인합니다.

기본 원칙은 하나입니다. 앞으로 본 실험, 보조 실험, smoke test는 모두 아래
통합 CLI에서 시작합니다.

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

## Dataset Profiles

아래 표는 `sample_data/dataset_analysis_report.md`와 현재 parquet 파일을 기준으로
정리한 실험 입력 특성입니다. 실제 CLI dataset 이름은 세 개입니다.

| CLI dataset | 입력 파일 | rows | series | sequence length | quantity scale |
| --- | --- | ---:| ---:| --- | --- |
| `intermittent` | `sample_data/head_office/marked_target_df.parquet` | 242,888 | 23,387 | mean 10.39, median 6, p95 35, max 110 | median 2, p95 16, max 5,000 |
| `yellow_trip_hourly` | `sample_data/new_york_taxi/yellow_trip_hourly.parquet` | 55,119 | 131 | mean 420.76, median 405, p95 743, max 744 | median 7, p95 1,547, max 6,489 |
| `insta_market_basket` | `sample_data/insta_market_basket/instacart_marked_target_with_split.parquet` | 3,279,521 | 206,209 | mean 15.90, median 10, p95 50, max 100 | median 8, p95 25, max 177 |

주의할 점:

- 대화에서는 `insta_market`이라고 줄여 부를 수 있지만, CLI 값은 `insta_market_basket`입니다.
- Instacart 원본 `instacart_marked_target_df.parquet`의 `mark`는 department id이므로 직접 학습에 쓰지 않습니다.
- Instacart fixed split 실험은 `mark=0..7`, `scale_residual=[0,1)`, `qty=2^(mark+scale_residual)` 계약을 만족하는 `instacart_marked_target_with_split.parquet`을 사용합니다.
- `yellow_trip_hourly`는 raw `yellow_trip.parquet`을 직접 학습하지 않습니다.
- `yellow_trip_hourly`는 `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`에서 미리 생성해야 합니다.
- 세 데이터셋 모두 최종 학습 target은 `demand_qty` 기반 magnitude mark와 residual입니다.
- 논문용 최종 검증은 `--split-mode fixed`를 사용해 미리 생성한 `*_with_split.parquet`의
  `chronological_split`을 따라 train/validation/test를 고정합니다.
- fixed split에서는 validation target만 checkpoint 선택에 사용하고, test target은 학습 완료 후
  선택 checkpoint별 held-out 평가에만 사용합니다.

## Parameter Policy

세 데이터셋은 sequence 길이와 quantity tail이 완전히 다릅니다. 따라서 같은 모델을
비교하더라도 `max_seq_len`, RMTPP hidden size, Titan/THP candidate를 데이터셋별로
다르게 잡는 것이 맞습니다.

| dataset | 추천 scale base | 추천 sequence setting | 추천 RMTPP | 추천 TitanTPP | 추천 THP | 이유 |
| --- | ---:| --- | --- | --- | --- | --- |
| `intermittent` | 2.0 | effective `max_seq_len=16`, CLI `--batch-size` 존중 | GRU, hidden 64 | `small_lmm` | `small`, `base` | series는 많지만 개별 sequence가 짧고 median length가 6이라 큰 encoder보다 small capacity가 안정적입니다. 짧은 sequence 때문에 큰 GPU에서는 batch size를 크게 잡아 throughput을 확보하는 편이 좋습니다. |
| `yellow_trip_hourly` | 10.0 | effective `lookback=168`, `max_seq_len=256`, `batch_size=128` | GRU, hidden 128 | `mid_lmm`, `mid_deep_lmm` | `small`, `base` | series는 131개뿐이지만 각 sequence가 매우 길고 hourly weekly pattern이 있으므로 168시간 이상 context와 중간 이상 capacity가 필요합니다. |
| `insta_market_basket` | 2.0 | `max_seq_len=64`, `batch_size=128` or `256` | GRU, hidden 128 | `mid_lmm` | `small`, `base` | user 수가 206k로 매우 많고 p95 length가 50이라 `max_seq_len=64`가 대부분의 history를 커버합니다. quantity tail은 taxi보다 약하므로 log2 + medium capacity가 균형적입니다. |

현재 코드의 dataset-specific override:

| dataset | 자동 적용 |
| --- | --- |
| `intermittent` | `lookback=52`, `max_seq_len=16`. `batch_size`는 CLI 값을 그대로 사용 |
| `yellow_trip_hourly` | `lookback >= 168`, `max_seq_len >= 256` |
| `insta_market_basket` | 별도 override 없음. 실험 명령에서 `--max-seq-len 64`를 명시하는 것을 권장 |

실험 설계상 중요한 포인트:

- `--titan-profile dataset_best`를 쓰고 `--titan-candidates`를 비워두면 dataset별 기본 Titan 후보가 자동 선택됩니다.
- `--titan-candidates a,b,c`를 직접 주면 모든 선택 dataset에 동일한 후보 목록이 적용됩니다.
- dataset별로 다른 Titan 후보를 sweep하려면 아래처럼 dataset별 명령을 분리하는 것이 안전합니다.
- RMTPP hidden size는 `--rmtpp-hidden-dim`을 생략하면 해당 dataset profile의 Titan `d_model`에 맞춰집니다.

TitanTPP memory-mode 후보:

| memory mode | 의미 | 후보 |
| --- | --- | --- |
| `none` | pure causal Titan encoder | `small_no_lmm`, `mid_no_lmm` |
| `static_lmm` | learnable persistent/static memory + LMM | `small_lmm`, `small_deep_lmm`, `mid_lmm`, `mid_deep_lmm` |
| `contextual_ttm` | window 내부 online contextual memory update | `small_contextual_ttm`, `mid_contextual_ttm` |
| `series_lmm` | per-series retrieved memory hook. 기본 long-epoch에서는 memory 미주입 시 fallback | `small_series_lmm`, `mid_series_lmm` |
| `hybrid_lmm_ttm` | contextual TTM + LMM retrieval | `small_hybrid_lmm_ttm`, `mid_hybrid_lmm_ttm` |

최종 RMTPP/TitanTPP/THP 비교 전에 TitanTPP만 먼저 screening하는 것이 좋습니다.
이 단계에서는 test 결과를 논문 주장에 직접 쓰기보다, validation 기준으로 TitanTPP
대표 후보를 고르는 용도로 사용합니다.

## 1. Model Interface Test

새 모델을 추가하거나 model registry/candidate config를 수정한 뒤 가장 먼저 실행합니다.
실제 parquet를 읽지 않고 synthetic batch로 `forward`, `nll`, `mark_head`,
`value_head`, `sample_next_dt`, `reconstruct_qty`가 모두 finite인지 확인합니다.

기본 세 모델 smoke test:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm \
  --thp-candidates small \
  --device cpu \
  --left-pad
```

`intermittent` 형태의 짧은 sequence smoke test:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm \
  --thp-candidates small \
  --seq-len 16 \
  --num-marks 14 \
  --rmtpp-hidden-dim 64 \
  --device cpu \
  --left-pad
```

`yellow_trip_hourly` 형태의 긴 sequence smoke test:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm \
  --thp-candidates base \
  --seq-len 128 \
  --num-marks 5 \
  --rmtpp-hidden-dim 128 \
  --device cpu \
  --left-pad
```

`insta_market_basket` 형태의 중간 길이 sequence smoke test:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm \
  --thp-candidates small \
  --seq-len 64 \
  --num-marks 9 \
  --rmtpp-hidden-dim 128 \
  --device cpu \
  --left-pad
```

결과 파일:

```text
search_artifacts/tpp_model_test/
  model_test_summary.csv
  model_test_summary.json
```

## 2. Long-Epoch Main Comparison

논문/본실험 관점의 핵심 비교는 `long-epoch`입니다. RMTPP, TitanTPP,
TransformerHawkesTPP를 같은 marked dataset, split, quantity reconstruction metric에서
비교합니다.

장기 GPU 실행 중 프로세스가 끊겨도 같은 `--base-dir`, `--datasets`, `--models`,
`--epochs`, `--seeds`로 다시 실행하면 완료된 run은 skip되고, 미완료 run은
`checkpoints/last_epoch_state.pt`에서 마지막 완료 epoch 다음부터 이어집니다.
단, `--force-rerun`을 붙이면 resume하지 않고 처음부터 다시 학습합니다.

세 데이터셋 전체 본 비교:

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/all_three_datasets_all_models_e800 \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models rmtpp,titantpp,thp \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 64 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

Fixed split TitanTPP memory-mode screening:

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/fixed_split_titantpp_memory_mode_screening_e500 \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models titantpp \
  --titan-candidates small_no_lmm,small_lmm,small_contextual_ttm,small_hybrid_lmm_ttm,mid_no_lmm,mid_lmm,mid_contextual_ttm,mid_hybrid_lmm_ttm \
  --epochs 500 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 64 \
  --eval-selections best_val_nll,best_score,final \
  --split-mode fixed \
  --value-head-activation identity \
  --device cuda
```

이 screening 결과에서 dataset별 TitanTPP 후보를 고른 뒤, 아래 fixed split 본 비교에
선택 후보만 넣는 흐름을 권장합니다. 이렇게 해야 최종 비교가 임의 후보가 아니라
validation으로 선택된 TitanTPP configuration 기준이 됩니다.

논문용 fixed train/validation/test split 본 비교:

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/fixed_split_all_three_all_models_e800 \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm,mid_no_lmm,mid_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 64 \
  --eval-selections best_val_nll,best_score,final \
  --split-mode fixed \
  --value-head-activation identity \
  --device cuda
```

`--value-head-activation identity`를 권장하는 이유:

- `head_office` fixed split은 log2 tail mark merge 때문에 `scale_residual > 1`인 row가 존재합니다.
- sigmoid value head는 출력 범위가 `[0, 1]`에 묶이므로 tail-merged quantity를 구조적으로 과소추정할 수 있습니다.
- identity head는 residual-only, hybrid, qty-only loss 실험을 모두 같은 출력 범위 제약 없이 비교하게 해줍니다.

이 명령의 effective setting:

| dataset | effective setting |
| --- | --- |
| `intermittent` | 자동으로 `max_seq_len=16`, Titan `small_lmm`, RMTPP hidden 64. `batch_size`는 CLI 값 사용 |
| `yellow_trip_hourly` | 자동으로 `lookback=168`, `max_seq_len=256`, Titan `mid_lmm`, RMTPP hidden 128 |
| `insta_market_basket` | 명령값 기준 `max_seq_len=64`, `batch_size=128`, Titan `mid_lmm`, RMTPP hidden 128 |

해석 기준:

| 지표 | 의미 |
| --- | --- |
| `best_val_nll` | 교수님이 말한 sweet spot 기준. 낮을수록 좋음 |
| `best_val_nll_marker` | mark/magnitude-class NLL. 낮을수록 mark 분포 예측이 좋음 |
| `best_val_nll_time` | time-intensity NLL. 낮을수록 next time 예측이 좋음 |
| `best_val_nll_qty_mae` | best NLL checkpoint에서 quantity MAE |
| `best_val_nll_mark_acc` | mark prediction accuracy |
| `best_val_nll_dt_mae` | next delta-time MAE |
| `scale_wise_summary.csv` | true quantity scale별 MAE/WAPE/median AE |

fixed split 실행 시 추가로 확인할 파일:

| 파일 | 의미 |
| --- | --- |
| `leaderboard/test_metrics.csv` | 선택 checkpoint별 held-out test NLL/Qty/Mark/DT 지표 |
| `leaderboard/test_summary.csv` | seed 평균 test 지표 |
| `leaderboard/test_scale_wise_summary.csv` | test set scale-wise Qty MAE/WAPE |
| `paper_outputs/paper_table_test_metrics.csv` | 논문 표 후보 test metric |
| `paper_outputs/paper_table_test_scale_wise_mae.csv` | 논문 표 후보 test scale-wise metric |
| `paper_outputs/plots/test/` | test scale-wise plot |

fixed split에서 NLL은 target event 하나만 평가합니다. 즉 validation/test sample의 context로 이전
train history를 사용할 수는 있지만, NLL 집계에는 마지막 target transition만 들어갑니다.
이렇게 해야 test NLL에 train-context transition이 반복 집계되는 문제를 피할 수 있습니다.

## 3. Dataset-Specific Candidate Sweeps

세 데이터셋은 추천 candidate가 다르므로, 논문용 최종 후보를 고를 때는 dataset별로
명령을 분리해 돌리는 것이 좋습니다.

`intermittent`: 짧고 sparse한 episode sequence 확인

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/intermittent_candidate_sweep_e800 \
  --datasets intermittent \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm,small_deep_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

왜 이렇게 잡는가:

- median sequence length가 6이라 지나치게 큰 encoder는 대부분 padding/짧은 context에서 낭비됩니다.
- p95 length는 35지만 현재 loader는 recent-event 중심이므로 `max_seq_len=16`이 안정적 baseline입니다.
- RMTPP hidden 64와 Titan small 계열이 서로 비슷한 capacity 비교를 만듭니다.
- THP는 attention 구조가 short sequence에서 과한지 확인하기 위해 `small`, `base`까지만 봅니다.

`yellow_trip_hourly`: 긴 hourly sequence와 강한 quantity tail 확인

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/yellow_trip_hourly_candidate_sweep_e800 \
  --datasets yellow_trip_hourly \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 256 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

왜 이렇게 잡는가:

- 각 grid-cell sequence가 median 405, p95 743으로 길어서 RMTPP hidden 128과 Titan mid 계열이 필요합니다.
- hourly 데이터는 하루/주간 주기가 있으므로 `lookback=168`이 최소 기준입니다.
- quantity p95가 1,547이고 max가 6,489라 quantity MAE는 scale-wise로 반드시 함께 해석해야 합니다.
- Titan `mid_deep_lmm`은 복잡한 시간 패턴을 더 잘 학습하는지 확인하기 위한 후보입니다.

`insta_market_basket`: 매우 많은 user와 중간 길이 sequence 확인

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/insta_market_basket_candidate_sweep_e800 \
  --datasets insta_market_basket \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm,mid_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 64 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

왜 이렇게 잡는가:

- user/entity 수가 206,209로 많아서 train sample은 풍부하지만 개별 sequence는 p95 50입니다.
- `max_seq_len=64`면 대부분의 user history를 커버하면서 THP/Titan memory cost를 막을 수 있습니다.
- quantity max가 177로 taxi보다 tail이 약해 `log2 + mid_lmm`이 해석과 학습 안정성의 균형점입니다.
- 대량 entity 데이터라 batch size를 256까지 올릴 수 있지만, THP `base`를 같이 돌릴 때는 128이 더 안전합니다.

## 4. Overfitting Diagnostic

목적은 성능표를 예쁘게 만드는 것이 아니라, 모델이 train distribution을 충분히
학습할 수 있는지 확인하는 것입니다. 일부 고용량 설정에서 train loss가 내려가고,
validation NLL이 어느 시점 이후 나빠지면 학습 자체는 가능한 것으로 해석합니다.

`intermittent` stress:

```bash
python simple_lab_test/search/tpp_experiment.py overfit \
  --base-dir search_artifacts/tpp_overfit_intermittent \
  --datasets intermittent \
  --models rmtpp,titantpp \
  --epochs 300 \
  --lr 1e-3 \
  --seeds 42 \
  --max-seq-lens 16 \
  --rmtpp-rnn-types rnn,gru,lstm \
  --rmtpp-hidden-dims 64,128,256 \
  --titan-candidates small_no_lmm,small_lmm,small_deep_lmm \
  --force-rerun
```

`yellow_trip_hourly` full long preset:

```bash
python simple_lab_test/search/tpp_experiment.py overfit \
  --preset yellow_trip_full_long
```

`insta_market_basket`는 sample이 매우 많으므로 먼저 subset으로 capacity를 확인합니다.

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/tpp_overfit_insta_market_subset \
  --datasets insta_market_basket \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm \
  --thp-candidates base \
  --insta-max-series 5000 \
  --epochs 300 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 64 \
  --force-rerun
```

확인 파일:

```text
paper_outputs/overfit_diagnostic_report.md
leaderboard/overfit_runs.csv
leaderboard/overfit_summary.csv
leaderboard/overfit_histories.csv
paper_outputs/plots/
```

## 5. Quantity Loss Ablation

현재 magnitude-factorized formulation은 quantity 자체가 아니라 `scale_residual`을
주로 학습합니다. 따라서 복원식 `qty = base^(mark + residual)` 때문에 log-space
오차가 quantity-space에서 크게 증폭될 수 있습니다.

비교 모드:

| loss mode | 학습 목적 |
| --- | --- |
| `residual_only` | mark CE + time NLL + residual Huber |
| `hybrid` | residual_only + direct quantity loss |
| `qty_only` | mark CE + time NLL + direct quantity loss |

세 데이터셋 TitanTPP quantity objective ablation:

```bash
python simple_lab_test/search/tpp_experiment.py qty-ablation \
  --base-dir search_artifacts/tpp_qty_loss_ablation_all_three \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models titantpp \
  --loss-modes residual_only,hybrid,qty_only \
  --epochs 100 \
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

### Value-Conditioned Marked TPP Ablation

Instacart plateau 분석을 위해 baseline은 유지하되, event history 입력에 과거 quantity state를
추가하는 optional ablation을 열어두었습니다. 이 옵션은 TPP를 fixed-interval forecasting으로
바꾸지 않습니다. 여전히 next mark, next time, next value residual을 예측하며, encoder input에
이미 관측된 과거 `scale_residual_t` 또는 `log_base(qty_t)`만 추가합니다.

누수 방지 규칙:

- week-lookback loader는 context 뒤에 target event를 append합니다.
- value-conditioned forward에서는 마지막 target row의 value를 입력에서 제거합니다.
- 허용되는 입력은 이미 관측된 history value뿐입니다.

권장 variant:

| variant | CLI option |
| --- | --- |
| `baseline` | `--value-input-mode none --train-loss-scope all --loss-mode residual_only` |
| `target_only` | `--value-input-mode none --train-loss-scope target_only --loss-mode residual_only` |
| `value_conditioned` | `--value-input-mode residual` 또는 `log_qty`, `--train-loss-scope target_only --loss-mode residual_only` |
| `value_conditioned_hybrid` | `--value-input-mode residual` 또는 `log_qty`, `--train-loss-scope target_only --loss-mode hybrid` |

Instacart value-conditioned hybrid smoke/full run 예시:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/insta_value_conditioned_hybrid_e200 \
  --datasets insta_market_basket \
  --models titantpp \
  --titan-candidates small_no_lmm,small_lmm,mid_no_lmm,mid_lmm \
  --epochs 200 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 512 \
  --max-seq-len 64 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

추가 산출물:

```text
runs/.../metrics/validation_mark_confusion_{selection}.csv
runs/.../metrics/test_mark_confusion_{selection}.csv
runs/.../metrics/scale_wise_{selection}.csv
runs/.../metrics/test_scale_wise_{selection}.csv
```

## 6. Preprocessing And Unified Entrypoint

`long-epoch --datasets yellow_trip_hourly`는
`simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`로 저장한 hourly grid-cell
event table만 읽습니다. raw `yellow_trip.parquet`을 `simple_lab_test/search` 내부에서
daily/hourly로 다시 변환하는 경로는 제거했습니다.

Yellow-trip 변환 조건을 바꿔야 할 때:

1. `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`에서 grid size, min active bucket 등을 수정합니다.
2. `sample_data/new_york_taxi/yellow_trip_hourly.parquet`을 다시 저장합니다.
3. `tpp_experiment.py long-epoch --datasets yellow_trip_hourly`를 실행합니다.

standalone search script는 `tpp_experiment.py`로 통합했습니다. 기존
`titan_hparam_search.py`, `titan_rmtpp_ab_test.py`,
`compare_log_bases_distribution.py`의 실행 역할은 제거했고, 필요한 공통 함수는
`common/experiment_utils.py`와 `common/benchmark_utils.py`로 이동했습니다.

| 목적 | 현재 실행 방식 |
| --- | --- |
| 본 비교 | `tpp_experiment.py long-epoch` |
| Titan 후보 sweep | `tpp_experiment.py long-epoch --titan-candidates ...` |
| RMTPP/TitanTPP/THP 전체 비교 | `tpp_experiment.py long-epoch --models rmtpp,titantpp,thp` |
| overfit 진단 | `tpp_experiment.py overfit` |
| quantity loss ablation | `tpp_experiment.py qty-ablation` |

## 7. Result Reading Order

결과를 볼 때는 final epoch만 보지 말고, best validation NLL checkpoint 기준으로
해석합니다.

1. `experiment_manifest.json`에서 effective dataset/model config 확인
2. `leaderboard/runs.csv`에서 실패 run, seed별 raw metric, effective `max_seq_len`, `batch_size`, `scale_base` 확인
3. `leaderboard/summary.csv`에서 seed 평균 기준 model 비교
4. `leaderboard/deltas.csv`에서 RMTPP 대비 TitanTPP/THP 개선량 확인
5. `leaderboard/histories.csv`에서 train loss, validation NLL, split NLL curve 확인
6. `leaderboard/scale_wise_summary.csv`에서 quantity scale별 MAE/WAPE 확인
7. `paper_outputs/report.md`와 `paper_outputs/plots/`를 회의/논문용 해석 초안으로 사용

데이터셋별 해석 포인트:

| dataset | 먼저 볼 것 | 이유 |
| --- | --- | --- |
| `intermittent` | `best_val_nll`, `best_val_nll_marker`, scale-wise Qty MAE | 짧은 sparse sequence라 mark/magnitude class 예측이 성능을 크게 좌우합니다. |
| `yellow_trip_hourly` | `best_val_nll_time`, learning curve, high-scale Qty MAE | 긴 hourly sequence와 heavy quantity tail 때문에 time pattern과 대형 수요 scale을 분리해서 봐야 합니다. |
| `insta_market_basket` | `best_val_nll`, `best_val_nll_qty_mae`, seed variance | entity 수가 매우 많아 평균 성능은 안정적일 수 있지만 user별 sequence가 짧아 seed/candidate 차이를 확인해야 합니다. |
