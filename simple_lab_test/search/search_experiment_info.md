# Search Experiment Info

이 문서는 `simple_lab_test/search`의 최신 구현 구조와 configuration 확정 위치를
정리합니다. 실행 레시피는 `search_experiment_guide.md`, 빠른 인덱스는
`README.md`를 봅니다.

## Architecture

```text
simple_lab_test/search/
  tpp_experiment.py
  common/
    configs.py
    models.py
    runner.py
    modes/
      model_test.py
      overfit.py
      qty_loss_ablation.py
      yellow_trip_resolution.py
      long_epoch_legacy.py
  titan_hparam_search.py
  titan_rmtpp_ab_test.py
  compare_log_bases_distribution.py
```

`tpp_experiment.py`가 통합 CLI입니다. `long-epoch`는 `common/runner.py`의 공통
runner를 사용합니다. `overfit`, `qty-ablation`, `yellow-resolution`은 기존
실험 구현을 `common/modes/` 아래로 이동한 상태이며, 앞으로 중복 함수를
`common/runner.py`로 더 흡수하면 됩니다.

## Model Registry

모델 추가/수정의 중심은 `common/models.py`입니다.

| 함수/객체 | 역할 |
| --- | --- |
| `canonical_model_name()` | CLI alias를 내부 모델명으로 정규화 |
| `default_thp_candidates()` | TransformerHawkesTPP preset 정의 |
| `build_project_rmtpp_config()` | 모든 encoder family가 공유하는 decoder/time/value config 생성 |
| `build_model()` | `rmtpp`, `titantpp`, `thp` 인스턴스 생성 |
| `model_run_label()` | plot/report용 model label 생성 |

현재 지원 모델:

| CLI name | 실제 모델 |
| --- | --- |
| `rmtpp` | `models.RMTPPs.RMTPP.RMTPP` |
| `titantpp` | `models.RMTPPs.TitanTPP.TitanTPP` |
| `thp` | `models.RMTPPs.TransformerHawkesTPP.TransformerHawkesTPP` |

THP alias:

```text
thp
transformer_hawkes
transformer_hawkes_process
transformer_hawkes_tpp
TransformerHawkesTPP
```

## TransformerHawkesTPP Integration

`models/RMTPPs/TransformerHawkesTPP.py`는 official Transformer Hawkes Process의
encoder idea를 프로젝트 공통 decoder에 맞춘 adapter입니다.

구현 원칙:

| 항목 | 내용 |
| --- | --- |
| event embedding | mark id embedding |
| temporal encoding | cumulative time 기반 sinusoidal temporal encoding |
| attention | causal self-attention |
| padding | left-padded batch를 위한 safe attention mask |
| decoder | RMTPP/TitanTPP와 동일한 mark/time/value heads |
| objective | 공통 `nll_marker + lambda_dt*nll_time + lambda_value*value_loss` |

빠른 interface 검증은 `model-test`를 사용합니다.

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --models TransformerHawkesTPP \
  --thp-candidates small \
  --device cpu \
  --left-pad
```

## Dataset Configuration

### Intermittent

CLI에서는 계속 `intermittent`라고 부르지만, search 실험에서는
`sample_data/marked_target_df.parquet`를 읽습니다.

필수 컬럼:

| column | 의미 |
| --- | --- |
| `oper_part_no` | series id |
| `demand_dt` | event date/time |
| `seq` | event order |
| `demand_qty` | positive demand quantity |
| `delta_t` | previous event로부터의 elapsed time |

effective config:

| 항목 | 값 |
| --- | --- |
| dataset kind | `marked_target` |
| scale base | `2.0` |
| lookback | `52` |
| max seq len | `16` |
| batch size | `64` |
| allowed Titan candidates in hparam search | `small_no_lmm`, `small_lmm` |

중요 구현 위치:

| 함수 | 역할 |
| --- | --- |
| `default_dataset_specs()` | `intermittent`를 marked target parquet로 매핑 |
| `prepare_marked_target_events()` | 필수 컬럼 검증 및 mark/residual 재계산 |
| `prepare_marked_dataset()` | marked target cache 생성 |
| `search_config_for_dataset()` | marked target runtime override 적용 |

### Yellow Trip

기본 A/B와 long-epoch는 기존 weekly grid-cell event setup을 유지합니다.
sequence가 너무 짧은 문제를 검증하기 위해 `yellow-resolution`에서 daily/hourly
event sequence를 별도로 만듭니다.

주요 옵션:

| 옵션 | 의미 |
| --- | --- |
| `--resolutions daily,hourly` | daily/hourly event sequence 생성 |
| `--grid-size-deg` | pickup location grid 크기 |
| `--max-series` | eligible series 수 제한. `0`이면 전체 |
| `--hourly-lookback-buckets` | hourly sequence lookback |
| `--daily-lookback-buckets` | daily sequence lookback |

## Unified CLI Details

### `model-test`

위치:

```text
common/modes/model_test.py
```

역할:

| 단계 | 확인 내용 |
| --- | --- |
| synthetic batch 생성 | mark, delta time, residual, mask |
| forward | hidden state shape와 finite 값 |
| nll | `nll`, `nll_marker`, `nll_time`, `value_loss` |
| heads | mark logits, value prediction |
| reconstruction | quantity 복원과 next dt sampling |

저장 파일:

```text
search_artifacts/tpp_model_test/model_test_summary.csv
search_artifacts/tpp_model_test/model_test_summary.json
```

### `long-epoch`

위치:

```text
common/runner.py
```

역할:

| 기능 | 내용 |
| --- | --- |
| model sweep | `--models rmtpp,titantpp,thp` |
| Titan sweep | `--titan-candidates ...` |
| THP sweep | `--thp-candidates small,base,deep,wide` |
| checkpoint | `best_val_nll`, `best_score`, `final` |
| scale-wise eval | true quantity scale별 MAE/WAPE/median AE |
| report | CSV, parquet, markdown, plot 저장 |

주요 출력:

```text
leaderboard/runs.csv
leaderboard/summary.csv
leaderboard/deltas.csv
leaderboard/histories.csv
leaderboard/scale_wise_metrics.csv
leaderboard/scale_wise_summary.csv
paper_outputs/report.md
paper_outputs/plots/
```

Resume checkpoint:

```text
runs/{dataset}/{model}/.../checkpoints/last_epoch_state.pt
```

`last_epoch_state.pt`에는 마지막 완료 epoch, model state, optimizer state,
history, best score/NLL state, RNG state가 저장됩니다. 완전 종료된 run은 기존
`summary.json`과 scale-wise metric 파일로 skip되고, 미완료 run은 이 checkpoint로
이어 학습합니다.

### `overfit`

위치:

```text
common/modes/overfit.py
```

역할:

| 항목 | 내용 |
| --- | --- |
| RMTPP capacity | `rnn`, `gru`, `lstm`, hidden dim, mark emb dim |
| Titan capacity | Titan preset, mark emb dim, max seq len |
| 목적 | train loss 감소와 validation NLL 악화 여부 확인 |
| preset | `yellow_trip_full_long`, `yellow_trip_subset_stress` |

### `qty-ablation`

위치:

```text
common/modes/qty_loss_ablation.py
```

loss mode:

| mode | objective |
| --- | --- |
| `residual_only` | mark CE + time NLL + residual Huber |
| `hybrid` | residual_only + direct quantity loss |
| `qty_only` | mark CE + time NLL + direct quantity loss |

이 실험은 본 논문의 main comparison이 아니라 quantity objective 설계를 검증하는
ablation입니다.

### `yellow-resolution`

위치:

```text
common/modes/yellow_trip_resolution.py
```

역할:

| 항목 | 내용 |
| --- | --- |
| 목적 | weekly yellow-trip sequence가 너무 짧은지 검증 |
| daily | day bucket 기준 event sequence |
| hourly | hour bucket 기준 event sequence |
| 비교 모델 | 현재 RMTPP/TitanTPP 중심 |

## Standalone Scripts

| 스크립트 | 유지 이유 |
| --- | --- |
| `titan_hparam_search.py` | TitanConfig와 scale base 탐색용. dataset/cache builder도 제공 |
| `titan_rmtpp_ab_test.py` | 기존 30 epoch RMTPP vs TitanTPP benchmark 재현용 |
| `compare_log_bases_distribution.py` | raw log base 분포 sanity check |

## Removed Root-Level Experiment Scripts

아래 파일들은 root-level에서 삭제되었고, 기능은 통합 CLI로 이동했습니다.

| removed | replacement |
| --- | --- |
| `titan_rmtpp_long_epoch_scale_eval.py` | `tpp_experiment.py long-epoch` |
| `tpp_overfit_diagnostic.py` | `tpp_experiment.py overfit` |
| `tpp_qty_loss_ablation.py` | `tpp_experiment.py qty-ablation` |
| `yellow_trip_resolution_ab_test.py` | `tpp_experiment.py yellow-resolution` |

## Interpretation Notes

`best_val_nll`은 sweet spot 기준입니다. final epoch metric만 보면 under-training과
overfitting을 구분하기 어렵습니다.

전체 `qty_mae`는 큰 수요 이벤트에 끌릴 수 있습니다. 따라서 `scale_wise_summary.csv`
에서 true quantity scale별 MAE/WAPE를 함께 봐야 합니다.

marked target은 episode-level event 수가 제한적입니다. 대형 Titan/THP 후보가
항상 유리하다고 가정하지 말고 `model-test`, `overfit`, `long-epoch` 순서로
학습 가능성과 generalization을 분리해서 확인합니다.
