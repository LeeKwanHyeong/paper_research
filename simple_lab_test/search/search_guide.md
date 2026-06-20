# Search Folder Guide

이 문서는 `simple_lab_test/search` 폴더의 실험 파일을 빠르게 다시 잡기 위한
짧은 인덱스입니다. 더 자세한 설정 설명은 `search_experiment_info.md`를 보면
됩니다.

## 현재 기본 전제

`intermittent`라는 CLI 이름은 유지하지만, search 학습에서는
`sample_data/head_office/marked_target_df.parquet`를 읽습니다. 이 파일은 notebook에서
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

`yellow_trip.parquet`은 raw taxi log로만 보관합니다. 본 학습에서는
`simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`로 생성한
`sample_data/new_york_taxi/yellow_trip_hourly.parquet`을 `yellow_trip_hourly`
dataset 이름으로 사용합니다.

## 현재 파일별 역할

| 파일 | 목적 | 언제 쓰는가 |
| --- | --- | --- |
| `tpp_experiment.py` | 통합 실험 CLI | 앞으로 모델/데이터셋이 늘어나는 본 실험 진입점 |
| `common/modes/model_test.py` | synthetic batch 기반 모델 인터페이스 테스트 | 새 TPP 모델 추가 후 forward/NLL/head smoke test |
| `common/runner.py` | `long-epoch` 공통 train/eval/report runner | 본 비교, THP 포함 장기 학습, scale-wise MAE |
| `common/models.py` | RMTPP/TitanTPP/THP model registry | 새 비교 모델 추가 시 먼저 수정할 곳 |
| `common/experiment_utils.py` | dataset/cache/config helper | dataset 추가, marked-cache 생성, log base distribution 확인 |
| `common/benchmark_utils.py` | profile/table helper | dataset별 best profile, paper table 유틸 |
| `common/modes/` | 기존 overfit, qty-ablation 실행 모듈 | 이전 실험을 통합 CLI 아래에서 유지할 때 |

## 권장 실행 순서

1. notebook에서 `sample_data/marked_target_df.parquet`를 최신 episode-level 기준으로 저장
2. marked target A/B smoke test
3. quantity loss ablation 또는 long epoch로 안정성 확인
4. `simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`에서 `yellow_trip_hourly.parquet` 생성
5. yellow-trip은 `long-epoch --datasets yellow_trip_hourly` 기준으로 확인

## 통합 CLI 사용법

앞으로 새 모델을 추가하거나 후보군을 늘릴 때는 우선 `tpp_experiment.py`의
`long-epoch` 모드를 기준으로 실행하는 것을 권장합니다. 이 모드는 기존
long-epoch 전용 스크립트의 핵심 기능을 공통 모듈로 옮긴 버전입니다.

지원되는 모델 이름:

| 모델 이름 | 설명 |
| --- | --- |
| `rmtpp` | recurrent RMTPP baseline |
| `titantpp` | Titan encoder 기반 TPP |
| `thp` | Transformer Hawkes Process 스타일 causal Transformer baseline |

Marked target + Titan candidate sweep:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/unified_marked_target_long_epoch \
  --datasets intermittent \
  --models rmtpp,titantpp \
  --titan-candidates small_deep_lmm,mid_lmm \
  --rmtpp-hidden-dim 64 \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --eval-selections best_val_nll,best_score,final \
  --force-rerun
```

THP baseline까지 함께 비교:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/unified_marked_target_with_thp \
  --datasets intermittent \
  --models rmtpp,titantpp,thp \
  --titan-candidates mid_lmm \
  --thp-candidates small,base \
  --epochs 300 \
  --seeds 42,52,62 \
  --lr 1e-3
```

TransformerHawkesTPP 모델 smoke test:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models TransformerHawkesTPP \
  --thp-candidates small \
  --device cpu \
  --left-pad
```

세 모델 인터페이스를 한 번에 확인:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm \
  --thp-candidates small \
  --device cpu \
  --left-pad
```

현재 `overfit`, `qty-ablation`은 통합 CLI에서
`common/modes/` 모듈로 직접 위임됩니다. root-level 개별 실행 파일은 제거했으므로,
아래처럼 새 진입점에서 실행하면 됩니다.

```bash
python simple_lab_test/search/tpp_experiment.py overfit --help
python simple_lab_test/search/tpp_experiment.py qty-ablation --help
```

다음 refactor step에서는 이 모드들의 내부 중복 로직도 `common/runner.py`로
조금씩 흡수하면 됩니다.

Marked target A/B는 이제 통합 CLI에서 실행합니다.

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets intermittent \
  --models rmtpp,titantpp \
  --epochs 30 \
  --seeds 42,52,62 \
  --force-rerun
```

Marked target long epoch:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets intermittent \
  --epochs 100 \
  --seeds 42,52,62 \
  --force-rerun
```

Marked target Titan candidate sweep:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets intermittent \
  --titan-candidates small_deep_lmm,mid_lmm \
  --rmtpp-hidden-dim 64 \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --eval-selections best_val_nll,best_score,final \
  --force-rerun
```

`--titan-candidates`를 비우면 기존처럼 `--titan-profile`의 단일 후보를 쓰고,
값을 넣으면 지정한 Titan preset들을 같은 marked dataset/cache에서 순회합니다.
`--rmtpp-hidden-dim`은 후보 sweep 중 RMTPP baseline capacity가 후보별로 흔들리지
않도록 고정하는 옵션입니다.

Marked target quantity loss ablation:

```bash
python simple_lab_test/search/tpp_experiment.py qty-ablation \
  --datasets intermittent \
  --epochs 30 \
  --seeds 42,52,62 \
  --force-rerun
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
