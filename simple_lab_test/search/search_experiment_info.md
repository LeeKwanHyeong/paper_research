# Search Experiment Guide

이 문서는 `simple_lab_test/search` 폴더의 실험 스크립트들이 어떤 질문을
해결하는지, 그리고 각 파일에서 Configuration이 어디서 어떻게 확정되는지
정리한 상세 가이드입니다.

## 현재 기준

`intermittent`라는 CLI 데이터셋 이름은 유지하지만, search 실험에서는 이제
`sample_data/marked_target_df.parquet`를 읽습니다. 이 파일은 notebook에서
정상적인 간헐 수요 burst를 episode 단위로 collapse한 table이어야 합니다.
search loader는 episode row와 `delta_t`를 그대로 쓰고, `demand_qty` 기준으로
`scale_base=2.0`의 `mark/scale_residual`을 다시 정규화합니다.

`marked_target_df.parquet` 필수 컬럼:

- `oper_part_no`
- `demand_dt`
- `seq`
- `demand_qty`
- `delta_t`
`mark`, `z`, `scale_residual`이 이미 있어도 search loader에서 `scale_base=2.0`
기준으로 다시 계산합니다.

공통 marked target 설정:

| 항목 | 값 |
| --- | --- |
| 데이터셋 label | `intermittent` |
| 데이터셋 kind | `marked_target` |
| 입력 파일 | `sample_data/marked_target_df.parquet` |
| scale base | `2.0` 고정 |
| lookback | `52` |
| max sequence length | `16` |
| batch size | `64` |
| A/B 및 qty ablation epoch | `30` 기본 |
| Titan profile | `small_lmm` 기본 |
| Titan hparam 후보 | `small_no_lmm`, `small_lmm` |

`yellow_trip` 관련 Configuration은 기존 기본값을 유지합니다. 즉 공통 CLI의
`batch_size=128`, `max_seq_len=64` 또는 각 yellow 전용 스크립트의 자체 기본값을
그대로 사용합니다.

## 공통 구현 위치

`titan_hparam_search.py`가 공통 dataset/cache/config builder 역할을 합니다.

중요 함수:

- `default_dataset_specs()`: `intermittent` label을
  `sample_data/marked_target_df.parquet`, kind `marked_target`으로 매핑합니다.
- `prepare_marked_target_events()`: episode-level table의 필수 컬럼을 검증하고
  `demand_qty`에서 `scale_base=2.0` 기준 `mark/scale_residual`을 계산합니다.
  여기서 다시 weekly raw event를 만들지는 않습니다.
- `prepare_marked_dataset()`: `marked_target`이면 `scale_base=2.0`만 허용하고
  cache key를 `marked_parts_*`로 분리합니다.
- `search_config_for_dataset()`: kind가 `marked_target`이면
  `lookback=52`, `max_seq_len=16`, `batch_size=64`를 적용합니다. 그 외
  데이터셋은 입력 CLI/default 값을 그대로 둡니다.
- `scale_bases_for_dataset()`: `marked_target`은 `(2.0,)`, `yellow_trip`은
  기존 `SearchConfig.log_bases=(10.0, 4.0, 2.0)`를 사용합니다.
- `candidate_allowed_for_dataset()`: `marked_target`은 `small_no_lmm`,
  `small_lmm`만 허용하고, `yellow_trip`은 기존 후보 전체를 유지합니다.

## 파일별 역할 및 Configuration

### `compare_log_bases_distribution.py`

역할:
- raw intermittent table에서 `log10/log4/log2` mark 분포를 빠르게 비교하는
  standalone utility입니다.
- 학습이나 테스트 Configuration을 만들지 않습니다.

Configuration 위치:
- `main()`에서 `sample_data/intermittent_df.parquet`를 읽습니다.
- `compare_scale_bases(..., log_bases=(10.0, 4.0, 2.0), min_count=100,
  min_coverage=0.999)`로 분포만 출력합니다.

주의:
- 현재 학습 기준 marked target은 episode-level `marked_target_df.parquet`를
  읽은 뒤 `scale_base=2.0`으로 mark/residual을 정규화합니다.
- 이 파일은 raw 분포 sanity check 용도이며 최종 training cache를 만들지
  않습니다.

### `titan_hparam_search.py`

역할:
- TitanTPP scale base와 Titan architecture preset을 stage1/stage2로 탐색합니다.

Configuration 위치:
- `SearchConfig`: 전역 search/runtime 기본값을 정의합니다.
- `parse_args()`: CLI 기본값을 정의합니다.
- `main()`: CLI 값을 `SearchConfig`로 변환합니다.
- `search_config_for_dataset()`: 실제 학습 직전에 데이터셋별 effective config를
  확정합니다.
- `build_training_config()`, `build_rmtpp_config()`, `build_titan_config()`: trainer와
  model config 객체를 생성합니다.

현재 marked target 설정:
- `intermittent` label이 `marked_target_df.parquet`를 읽습니다.
- `scale_base=2.0`만 cache/training에 사용합니다.
- Titan 후보는 `small_no_lmm`, `small_lmm`만 사용합니다.
- loader는 `lookback=52`, `max_seq_len=16`, `batch_size=64`로 override됩니다.
- stage epoch는 search 용도라 CLI 기본값 `stage1_epochs=3`,
  `stage2_epochs=8`을 유지합니다.

현재 yellow-trip 설정:
- 기존 `sample_data/yellow_trip.parquet`를 weekly grid-cell count로 집계합니다.
- `log_bases=(10.0, 4.0, 2.0)`를 모두 탐색합니다.
- Titan 후보 전체를 탐색합니다.
- CLI/default `lookback_weeks=52`, `max_seq_len=64`, `batch_size=128`을 유지합니다.

대표 실행:

```bash
python simple_lab_test/search/titan_hparam_search.py
```

### `titan_rmtpp_ab_test.py`

역할:
- RMTPP baseline과 TitanTPP를 동일 split에서 비교하는 기본 A/B benchmark입니다.

Configuration 위치:
- `ABConfig`: A/B runtime 기본값을 정의합니다.
- `parse_args()`: CLI 기본값을 정의합니다.
- `default_profile_map()`: dataset별 Titan profile을 결정합니다.
- `make_search_cfg(ab_cfg, dataset_kind)`: `SearchConfig`를 만들고
  `search_config_for_dataset()`을 적용합니다.
- `make_training_cfg(ab_cfg, dataset_kind)`: trainer config를 생성합니다.
- `train_one_model()`: RMTPP/TitanTPP config와 manifest를 저장합니다.

현재 marked target 설정:
- `BEST_TITAN_BY_DATASET["intermittent"] = scale_base 2.0 + small_lmm`
- `BEST_TITAN_OVERALL["intermittent"] = scale_base 2.0 + small_lmm`
- `BEST_TITAN_SCORE_PRIORITY["intermittent"] = scale_base 2.0 + small_lmm`
- 기본 epoch는 `30`, seed는 `42,52,62`입니다.
- effective loader config는 `lookback=52`, `max_seq_len=16`, `batch_size=64`입니다.

현재 yellow-trip 설정:
- `dataset_best`: `scale_base=10.0`, `mid_lmm`
- `overall`: `scale_base=10.0`, `mid_lmm`
- `score_priority`: `scale_base=4.0`, `mid_deep_lmm`
- CLI/default `max_seq_len=64`, `batch_size=128`을 유지합니다.

대표 실행:

```bash
python simple_lab_test/search/titan_rmtpp_ab_test.py
```

### `tpp_qty_loss_ablation.py`

역할:
- `residual_only`, `hybrid`, `qty_only` quantity supervision 방식을 비교합니다.

Configuration 위치:
- `QtyAblationConfig`: ablation runtime 기본값을 정의합니다.
- `parse_args()`: CLI 기본값을 정의합니다.
- `default_profile_map()`: dataset별 Titan profile을 결정합니다.
- `make_search_cfg(cfg, dataset_kind)`: dataset별 effective `SearchConfig`를 만듭니다.
- `make_training_cfg(cfg, dataset_kind)`: trainer config를 생성합니다.
- `instantiate_model(..., dataset_kind)`: RMTPP/TitanTPP model config를 생성합니다.
- `train_one_run()`: loss mode와 effective config를 manifest/summary에 저장합니다.

현재 marked target 설정:
- Titan profile은 `scale_base=2.0 + small_lmm`입니다.
- 기본 epoch는 `30`, seed는 `42,52,62`입니다.
- effective loader config는 `lookback=52`, `max_seq_len=16`, `batch_size=64`입니다.

현재 yellow-trip 설정:
- A/B 파일과 동일하게 `dataset_best=log10 + mid_lmm`,
  `score_priority=log4 + mid_deep_lmm`을 유지합니다.
- CLI/default `max_seq_len=64`, `batch_size=128`을 유지합니다.

대표 실행:

```bash
python simple_lab_test/search/tpp_qty_loss_ablation.py
```

### `titan_rmtpp_long_epoch_scale_eval.py`

역할:
- 30 epoch가 부족했는지 확인하고, best validation NLL checkpoint 기준
  scale-wise quantity error를 분석합니다.

Configuration 위치:
- `LongEpochConfig`: long-run runtime 기본값을 정의합니다.
- `parse_args()`: CLI 기본값을 정의합니다.
- `make_search_cfg()`는 `titan_rmtpp_ab_test.py`에서 가져옵니다.
- `make_training_cfg(long_cfg, dataset_kind)`: dataset별 effective trainer config를
  생성합니다.
- `build_model()`: dataset별 effective search config로 RMTPP/TitanTPP model config를
  생성합니다.
- `train_one_long_run()`: actual `batch_size/lookback/max_seq_len`을 summary에
  저장합니다.

현재 marked target 설정:
- profile은 A/B와 동일하게 `scale_base=2.0 + small_lmm`입니다.
- 기본 long epoch는 `100`입니다.
- learning rate 기본값은 `1e-3`입니다.
- effective loader config는 `lookback=52`, `max_seq_len=16`, `batch_size=64`입니다.

현재 yellow-trip 설정:
- 기존 profile map과 CLI/default를 유지합니다.
- CLI 기본값은 `epochs=100`, `lr=1e-3`, `max_seq_len=64`, `batch_size=128`입니다.
- `analysis_scale_base=10.0`, `analysis_tail_order=4`는 scale-wise report용이며
  training mark base와 별개입니다.

대표 실행:

```bash
python simple_lab_test/search/titan_rmtpp_long_epoch_scale_eval.py \
  --epochs 100 \
  --seeds 42,52,62
```

### `tpp_overfit_diagnostic.py`

역할:
- 모델이 train distribution을 강하게 학습하거나 과적합할 수 있는지 확인하는
  stress test입니다.

Configuration 위치:
- `parse_args()`: diagnostic CLI 기본값을 정의합니다.
- `apply_experiment_preset()`: yellow-trip 전용 preset을 적용합니다.
- `LongEpochConfig`: 실제 training config의 base object로 사용합니다.
- `max_seq_lens_for_spec()`: 기본 실행에서 marked target만 `max_seq_len=16`으로
  줄입니다.
- `titan_candidate_names_for_spec()`: 기본 실행에서 marked target Titan 후보를
  small preset으로 제한합니다.
- `train_one_long_run()`은 `titan_rmtpp_long_epoch_scale_eval.py`의 dataset별
  effective config 적용을 그대로 사용합니다.

현재 marked target 설정:
- 사용자가 `--max-seq-lens`를 직접 넘기지 않으면 `16`만 사용합니다.
- 사용자가 `--titan-candidates`를 직접 넘기지 않으면 `small_no_lmm`,
  `small_lmm`만 사용합니다.
- 실제 loader config는 long-run helper를 통해 `lookback=52`,
  `max_seq_len=16`, `batch_size=64`로 적용됩니다.

현재 yellow-trip 설정:
- `yellow_trip_full_long`, `yellow_trip_subset_stress` preset은 기존 값을 유지합니다.
- preset의 `max_seq_lens`, Titan 후보, epoch, subset 크기 설정을 건드리지
  않았습니다.

대표 실행:

```bash
python simple_lab_test/search/tpp_overfit_diagnostic.py \
  --datasets intermittent \
  --epochs 100 \
  --seeds 42
```

### `yellow_trip_resolution_ab_test.py`

역할:
- yellow-trip을 weekly가 아니라 daily/hourly event sequence로 재구성해
  RMTPP와 TitanTPP를 비교합니다.

Configuration 위치:
- `YellowResolutionSpec`: 해상도별 데이터 생성 설정을 정의합니다.
- `ResolutionRuntimeConfig`: 학습 runtime 설정을 정의합니다.
- `parse_args()`: CLI 기본값을 정의합니다.

현재 설정:
- 이 파일은 yellow-trip 전용이며 이번 marked target 변경에서 건드리지 않았습니다.
- 기본값은 `resolutions=daily,hourly`, `models=rmtpp,titantpp`,
  `titan_candidates=mid_lmm,mid_deep_lmm`, `epochs=100`, `lr=1e-3`,
  `batch_size=128`, `scale_base=10.0`, `grid_size_deg=0.02`,
  `max_series=1000`, `max_seq_len=256`입니다.

대표 실행:

```bash
python simple_lab_test/search/yellow_trip_resolution_ab_test.py \
  --resolutions daily,hourly \
  --models rmtpp,titantpp \
  --titan-candidates mid_lmm,mid_deep_lmm \
  --epochs 100 \
  --seeds 42,52,62
```

## 어떤 질문에 어떤 파일을 쓰는가

| 질문 | 사용할 파일 |
| --- | --- |
| raw intermittent log base별 mark 분포만 보고 싶은가? | `compare_log_bases_distribution.py` |
| marked target과 yellow-trip에서 Titan 후보를 탐색하고 싶은가? | `titan_hparam_search.py` |
| marked target 기준 RMTPP vs TitanTPP 기본 비교가 필요한가? | `titan_rmtpp_ab_test.py` |
| epoch를 길게 늘렸을 때 수렴/scale-wise error가 어떤가? | `titan_rmtpp_long_epoch_scale_eval.py` |
| 모델이 train data를 충분히 학습 가능한지 확인할 것인가? | `tpp_overfit_diagnostic.py` |
| quantity direct loss가 필요한지 비교할 것인가? | `tpp_qty_loss_ablation.py` |
| yellow-trip weekly가 너무 짧아 daily/hourly로 바꿔볼 것인가? | `yellow_trip_resolution_ab_test.py` |

## 산출물 확인 순서

대부분의 search 스크립트는 아래 구조를 따릅니다.

```text
search_artifacts/{experiment_name}/
  cache/
  runs/
  leaderboard/
  paper_outputs/
  *.log
  *_manifest.json
```

확인 우선순위:

1. `*_manifest.json`: 입력 CLI config와 dataset별 effective config 확인
2. `leaderboard/*runs.csv`: run-level 결과 확인
3. `leaderboard/*histories.csv`: epoch별 learning curve 확인
4. `leaderboard/*summary.csv`: seed 평균 집계 확인
5. `paper_outputs/*.md`: 사람이 읽기 좋은 분석 리포트 확인
6. `paper_outputs/plots/`: 발표/논문용 시각화 확인

## 해석 메모

marked target은 episode-level event 수가 약 21,100건 수준이므로, 대형 Titan
설정을 돌리기보다 `small_lmm` 같은 작은 설정으로 sanity/prototype 성능을 보는
것이 맞습니다. 이 데이터에서 중요한 것은 weekly positive event가 아니라
burst episode 간의 `delta_t`이므로, raw weekly intermittent table을 다시
marking하면 `delta_t`가 다시 1 근처로 무너질 수 있습니다.

yellow-trip weekly 실험은 series 길이가 짧고 interval이 단순해 RMTPP가 충분히
강하게 보일 수 있습니다. TitanTPP의 encoder capacity를 더 공정하게 보려면
`yellow_trip_resolution_ab_test.py`의 daily/hourly benchmark를 함께 확인해야
합니다.
