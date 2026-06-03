# Search Folder Guide

이 문서는 `simple_lab_test/search` 폴더의 실험 파일을 빠르게 다시 잡기 위한
짧은 인덱스입니다. 더 자세한 설정 설명은 `search_experiment_info.md`를 보면
됩니다.

## 현재 기본 전제

`intermittent`라는 CLI 이름은 유지하지만, search 학습에서는
`sample_data/marked_target_df.parquet`를 읽습니다. 이 파일은 notebook에서
간헐 수요 burst를 episode 단위로 collapse한 table이어야 합니다. search loader는
`demand_qty` 기준으로 `scale_base=2.0`의 `mark/scale_residual`을 다시 계산합니다.

Marked target 기본값:

| 항목 | 값 |
| --- | --- |
| scale base | `2.0` |
| lookback | `52` |
| max seq len | `16` |
| batch size | `64` |
| Titan 기본 profile | `small_lmm` |

`yellow_trip` 설정은 기존 값 그대로 둡니다.

## 현재 파일별 역할

| 파일 | 목적 | 언제 쓰는가 |
| --- | --- | --- |
| `compare_log_bases_distribution.py` | raw intermittent의 `log10/log4/log2` 분포 비교 | mark/binning sanity check |
| `titan_hparam_search.py` | TitanTPP scale base와 TitanConfig 자동 탐색 | 후보군을 넓게 확인할 때 |
| `titan_rmtpp_ab_test.py` | best profile로 RMTPP vs TitanTPP 본 A/B 비교 | marked target 논문용 기본 비교표가 필요할 때 |
| `titan_rmtpp_long_epoch_scale_eval.py` | long epoch, best NLL, scale-wise quantity error 분석 | 30 epoch 부족 여부와 scale별 MAE를 볼 때 |
| `tpp_overfit_diagnostic.py` | 모델이 train data를 충분히 학습/과적합할 수 있는지 stress test | 학습 가능성과 capacity를 검증할 때 |
| `tpp_qty_loss_ablation.py` | `residual_only`, `hybrid`, `qty_only` loss ablation | quantity loss 설계 후보를 비교할 때 |
| `yellow_trip_resolution_ab_test.py` | yellow-trip daily/hourly 재구성 A/B benchmark | weekly yellow-trip이 너무 짧은지 확인할 때 |

## 권장 실행 순서

1. notebook에서 `sample_data/marked_target_df.parquet`를 최신 episode-level 기준으로 저장
2. marked target A/B smoke test
3. quantity loss ablation 또는 long epoch로 안정성 확인
4. yellow-trip은 기존 weekly 결과와 별도로 daily/hourly benchmark 확인

Marked target A/B:

```bash
python simple_lab_test/search/titan_rmtpp_ab_test.py \
  --datasets intermittent \
  --epochs 30 \
  --seeds 42,52,62 \
  --force-rerun
```

Marked target long epoch:

```bash
python simple_lab_test/search/titan_rmtpp_long_epoch_scale_eval.py \
  --datasets intermittent \
  --epochs 100 \
  --seeds 42,52,62 \
  --force-rerun
```

Marked target quantity loss ablation:

```bash
python simple_lab_test/search/tpp_qty_loss_ablation.py \
  --datasets intermittent \
  --epochs 30 \
  --seeds 42,52,62 \
  --force-rerun
```

Yellow-trip daily/hourly benchmark:

```bash
python simple_lab_test/search/yellow_trip_resolution_ab_test.py \
  --resolutions daily,hourly \
  --models rmtpp,titantpp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --grid-size-deg 0.02 \
  --max-series 1000 \
  --epochs 100 \
  --seeds 42,52,62
```

## 결과 확인 순서

대부분의 search 스크립트는 아래 구조로 결과를 저장합니다.

```text
search_artifacts/{experiment_name}/
  cache/
  runs/
  leaderboard/
  paper_outputs/
  *_manifest.json
```

먼저 볼 파일:

1. `*_manifest.json`: dataset별 effective config 확인
2. `leaderboard/*runs.csv`: run-level 결과
3. `leaderboard/*summary.csv`: seed 평균 결과
4. `leaderboard/*histories.csv`: epoch별 learning curve
5. `paper_outputs/*.md`: 분석 리포트
6. `paper_outputs/plots/`: figure 확인
