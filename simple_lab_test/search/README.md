# Search Experiments

이 폴더는 TitanTPP/RMTPP/TransformerHawkesTPP 실험을 재현하기 위한 실행 코드와
실험 문서를 모아둔 곳입니다. 현재 기준의 권장 진입점은
`simple_lab_test/search/tpp_experiment.py`입니다.

## Current Structure

| 경로 | 역할 |
| --- | --- |
| `tpp_experiment.py` | 통합 CLI. `long-epoch`, `model-test`, `overfit`, `qty-ablation` 실행 |
| `common/configs.py` | 공통 experiment/run dataclass |
| `common/models.py` | RMTPP, TitanTPP, TransformerHawkesTPP registry와 후보 preset |
| `common/runner.py` | `long-epoch` 공통 train/eval/report runner |
| `common/experiment_utils.py` | dataset spec, marked-cache, model config, serialization helper |
| `common/benchmark_utils.py` | dataset profile, cache preparation, table helper |
| `common/modes/model_test.py` | synthetic batch 기반 model interface smoke test |
| `common/modes/overfit.py` | overfitting/capacity diagnostic |
| `common/modes/qty_loss_ablation.py` | `residual_only`, `hybrid`, `qty_only` loss ablation |
| `../../TEST_SESSION_PROTOCOL.md` | 5090/5080 GPU 서버 실행, tmux, artifact sync, Notion 기록 템플릿 |

아래 root-level 파일들은 삭제되었습니다. 기능은 `tpp_experiment.py` subcommand와
`common/` 모듈로 이동했습니다.

| 이전 파일 | 새 실행 방식 |
| --- | --- |
| `titan_hparam_search.py` | `python simple_lab_test/search/tpp_experiment.py long-epoch --titan-candidates ...` |
| `titan_rmtpp_ab_test.py` | `python simple_lab_test/search/tpp_experiment.py long-epoch --models rmtpp,titantpp` |
| `compare_log_bases_distribution.py` | `common/experiment_utils.py`의 marked-cache distribution artifacts 확인 |
| `titan_rmtpp_long_epoch_scale_eval.py` | `python simple_lab_test/search/tpp_experiment.py long-epoch` |
| `tpp_overfit_diagnostic.py` | `python simple_lab_test/search/tpp_experiment.py overfit` |
| `tpp_qty_loss_ablation.py` | `python simple_lab_test/search/tpp_experiment.py qty-ablation` |

## Dataset Assumptions

`intermittent`라는 CLI 이름은 유지하지만, search 실험에서는
`sample_data/marked_target_df.parquet`를 읽습니다. 이 파일은 episode-level
간헐 수요 table이어야 하며, `demand_qty` 기준으로 `scale_base=2.0`의
`mark/scale_residual`을 다시 계산합니다.

| 항목 | 현재 설정 |
| --- | --- |
| intermittent input | `sample_data/head_office/marked_target_df.parquet` |
| intermittent kind | `marked_target` |
| intermittent scale base | `2.0` 고정 |
| intermittent lookback | `52` |
| intermittent max seq len | `16` |
| intermittent batch size | CLI `--batch-size` 값을 사용 |
| intermittent fixed split | `sample_data/head_office/marked_target_with_split.parquet` |
| yellow_trip raw input | `sample_data/new_york_taxi/yellow_trip.parquet` |
| yellow_trip preprocessing notebook | `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb` |
| yellow_trip_hourly training input | `sample_data/new_york_taxi/yellow_trip_hourly.parquet` |
| yellow_trip_hourly fixed split | `sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet` |
| yellow_trip long-epoch dataset | `yellow_trip_hourly` |
| insta_market_basket fixed split | `sample_data/insta_market_basket/instacart_marked_target_with_split.parquet` |

## TitanTPP Memory Modes

TitanTPP는 이제 `TitanConfig.memory_mode`를 기준으로 memory ablation을 명시합니다.
기존 `small_lmm`, `mid_no_lmm` 후보명은 유지하지만, 결과 metadata에는
`memory_mode`가 함께 저장됩니다.

| memory mode | 의미 | 대표 후보 |
| --- | --- | --- |
| `none` | pure causal Titan encoder. attention-side memory와 LMM을 끕니다. | `small_no_lmm`, `mid_no_lmm` |
| `static_lmm` | learnable persistent/static memory와 LMM을 사용합니다. | `small_lmm`, `mid_lmm`, `mid_deep_lmm` |
| `contextual_ttm` | window 내부에서 과거 token을 online contextual memory로 업데이트합니다. | `small_contextual_ttm`, `mid_contextual_ttm` |
| `series_lmm` | runner가 주입하는 per-series memory를 사용하기 위한 hook입니다. 현재 기본 long-epoch에서는 memory를 주입하지 않으면 encoder-only fallback으로 동작합니다. | `small_series_lmm`, `mid_series_lmm` |
| `hybrid_lmm_ttm` | contextual TTM과 LMM retrieval을 함께 사용합니다. | `small_hybrid_lmm_ttm`, `mid_hybrid_lmm_ttm` |

## Main Commands

TransformerHawkesTPP smoke test:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models TransformerHawkesTPP \
  --thp-candidates small \
  --device cpu \
  --left-pad
```

RMTPP/TitanTPP/TransformerHawkesTPP long-epoch comparison:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets intermittent \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm \
  --thp-candidates small,base \
  --epochs 300 \
  --seeds 42,52,62 \
  --lr 1e-3
```

Yellow-trip hourly RMTPP/TitanTPP/TransformerHawkesTPP comparison:

```bash
# Run simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb first to create yellow_trip_hourly.parquet.
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets yellow_trip_hourly \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3
```

Fixed train/validation/test split comparison for paper tables:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm,mid_no_lmm,mid_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --split-mode fixed \
  --value-head-activation identity
```

TitanTPP memory-mode screening before the final model comparison:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/fixed_split_titantpp_memory_mode_screening_e500 \
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

`--split-mode fixed` consumes the precomputed `*_with_split.parquet` files and
keeps held-out test metrics separate from validation checkpoint selection.
The runner writes `leaderboard/test_metrics.csv`, `leaderboard/test_summary.csv`,
`leaderboard/test_scale_wise_summary.csv`, and matching paper tables under
`paper_outputs/`.

TitanTPP V5a keeps categorical CE and adds normalized ordinal RPS without
changing the marker head or inference path. The default `ce` mode keeps the
legacy V2 path; V5a uses the distinct `markloss_ce_rps/lambdaord_0p1` run path.

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models titantpp \
  --titan-candidates small_lmm \
  --device cpu \
  --seq-len 16 \
  --num-marks 12 \
  --rmtpp-hidden-dim 64 \
  --marker-loss-mode ce_rps \
  --lambda-ordinal 0.10 \
  --left-pad
```

The corresponding Intermittent screening keeps the confirmed V2 architecture
and changes only the marker objective:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/model_enhancement_v5a_inter_short_e50 \
  --datasets intermittent \
  --models titantpp \
  --titan-candidates small_lmm \
  --epochs 50 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 128 \
  --lookback-weeks 52 \
  --max-seq-len 16 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-head-mode shared \
  --qty-mark-gradient-mode coupled \
  --value-encoder-gradient-mode coupled \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --marker-loss-mode ce_rps \
  --lambda-ordinal 0.10 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

`nll_marker` remains categorical CE, `nll` remains marker CE plus time NLL,
and `ordinal_marker_loss` is stored separately. Validation/test outputs also
include balanced accuracy, macro F1, mark MAE, adjacent accuracy, mark-0/1
diagnostics, and per-class marker metric artifacts.

TitanTPP M0 keeps marker/time likelihood heads but replaces the legacy
mark-residual quantity decoder with exclusive direct `log2(qty)` regression.
The first activation supports TitanTPP-only, fixed split, target-only loss,
plain marker CE, and train-global normalization.

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models titantpp \
  --titan-candidates small_lmm \
  --device cpu \
  --scale-base 2 \
  --qty-decoder-mode direct_log_qty \
  --magnitude-norm-mode global \
  --left-pad \
  --stop-on-error
```

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/model_enhancement_m0_inter_short_e50 \
  --datasets intermittent \
  --models titantpp \
  --titan-candidates small_lmm \
  --epochs 50 \
  --seeds 42 \
  --split-mode fixed \
  --qty-decoder-mode direct_log_qty \
  --magnitude-norm-mode global \
  --value-input-mode none \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --marker-loss-mode ce \
  --lambda-ordinal 0 \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

M0 computes global mean/population variance from train events only and stores
them in the run manifest, checkpoint config, and summary. Direct runs export
`magnitude_loss`, `log_qty_mae`, and `log_qty_rmse`; legacy `value_mae` is not
reinterpreted as a log-quantity metric.

TitanTPP TTM-Lite evaluation can be enabled on the same long-epoch runner:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets yellow_trip_hourly \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --test-time-memory contextual
```

`--test-time-memory contextual` adds series-wise online contextual-memory
metrics for TitanTPP under `metrics/ttm_contextual_<selection>.*`.

Overfitting diagnostic:

```bash
python simple_lab_test/search/tpp_experiment.py overfit \
  --datasets intermittent \
  --models rmtpp,titantpp \
  --epochs 100 \
  --lr 1e-3 \
  --seeds 42
```

Quantity loss ablation:

```bash
python simple_lab_test/search/tpp_experiment.py qty-ablation \
  --datasets intermittent \
  --models titantpp \
  --loss-modes residual_only,hybrid,qty_only \
  --epochs 30 \
  --seeds 42,52,62
```

## Outputs

대부분의 실험은 아래 구조로 저장됩니다.

```text
search_artifacts/{experiment_name}/
  cache/
  runs/
  leaderboard/
  paper_outputs/
  *.log
  *_manifest.json
```

확인 우선순위는 `*_manifest.json`, `leaderboard/*runs.csv`,
`leaderboard/*summary.csv`, `leaderboard/*histories.csv`,
`paper_outputs/*.md`, `paper_outputs/plots/` 순서입니다.

`long-epoch`는 매 epoch마다 아래 resume checkpoint를 갱신합니다. 실행이 중간에
끊기면 같은 명령을 `--force-rerun` 없이 다시 실행하면 마지막 완료 epoch 다음부터
이어 학습합니다.

```text
runs/{dataset}/{model}/.../checkpoints/last_epoch_state.pt
```

## Recommended Workflow

1. 새 모델/후보를 추가하면 `model-test`로 interface를 먼저 확인합니다.
2. `long-epoch`에서 RMTPP/TitanTPP/THP를 같은 split과 metric으로 비교합니다.
3. 학습 가능성이 의심되면 `overfit`으로 train loss와 validation divergence를 확인합니다.
4. quantity MAE가 불안정하면 `qty-ablation`으로 objective 설계를 비교합니다.
5. yellow-trip 변환 조건을 바꿔야 하면 `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`에서
   `yellow_trip_hourly.parquet`을 다시 생성한 뒤 `long-epoch`를 실행합니다.
