# Search Experiments

이 폴더는 TitanTPP/RMTPP/TransformerHawkesTPP 실험을 재현하기 위한 실행 코드와
실험 문서를 모아둔 곳입니다. 현재 기준의 권장 진입점은
`simple_lab_test/search/tpp_experiment.py`입니다.

## Current Structure

| 경로 | 역할 |
| --- | --- |
| `tpp_experiment.py` | 통합 CLI. `long-epoch`, `model-test`, `overfit`, `qty-ablation`, `yellow-resolution` 실행 |
| `common/configs.py` | 공통 experiment/run dataclass |
| `common/models.py` | RMTPP, TitanTPP, TransformerHawkesTPP registry와 후보 preset |
| `common/runner.py` | `long-epoch` 공통 train/eval/report runner |
| `common/modes/model_test.py` | synthetic batch 기반 model interface smoke test |
| `common/modes/overfit.py` | overfitting/capacity diagnostic |
| `common/modes/qty_loss_ablation.py` | `residual_only`, `hybrid`, `qty_only` loss ablation |
| `common/modes/yellow_trip_resolution.py` | yellow-trip daily/hourly resolution benchmark |
| `titan_hparam_search.py` | TitanConfig와 scale base 탐색 |
| `titan_rmtpp_ab_test.py` | 기본 RMTPP vs TitanTPP A/B benchmark |
| `compare_log_bases_distribution.py` | raw demand의 log base별 mark 분포 sanity check |

아래 root-level 파일들은 삭제되었습니다. 기능은 `tpp_experiment.py` subcommand와
`common/modes/`로 이동했습니다.

| 이전 파일 | 새 실행 방식 |
| --- | --- |
| `titan_rmtpp_long_epoch_scale_eval.py` | `python simple_lab_test/search/tpp_experiment.py long-epoch` |
| `tpp_overfit_diagnostic.py` | `python simple_lab_test/search/tpp_experiment.py overfit` |
| `tpp_qty_loss_ablation.py` | `python simple_lab_test/search/tpp_experiment.py qty-ablation` |
| `yellow_trip_resolution_ab_test.py` | `python simple_lab_test/search/tpp_experiment.py yellow-resolution` |

## Dataset Assumptions

`intermittent`라는 CLI 이름은 유지하지만, search 실험에서는
`sample_data/marked_target_df.parquet`를 읽습니다. 이 파일은 episode-level
간헐 수요 table이어야 하며, `demand_qty` 기준으로 `scale_base=2.0`의
`mark/scale_residual`을 다시 계산합니다.

| 항목 | 현재 설정 |
| --- | --- |
| intermittent input | `sample_data/marked_target_df.parquet` |
| intermittent kind | `marked_target` |
| intermittent scale base | `2.0` 고정 |
| intermittent lookback | `52` |
| intermittent max seq len | `16` |
| intermittent batch size | `64` |
| yellow_trip input | `sample_data/yellow_trip.parquet` |
| yellow_trip default benchmark | 기존 weekly grid-cell event setup |
| yellow_trip resolution benchmark | `daily`, `hourly` 재구성 가능 |

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

Yellow-trip daily/hourly benchmark:

```bash
python simple_lab_test/search/tpp_experiment.py yellow-resolution \
  --resolutions daily,hourly \
  --models rmtpp,titantpp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --epochs 100 \
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
5. yellow-trip weekly sequence가 너무 짧으면 `yellow-resolution`으로 daily/hourly를 확인합니다.
