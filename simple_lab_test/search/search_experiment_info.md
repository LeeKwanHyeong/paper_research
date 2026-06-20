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
      long_epoch_legacy.py
  tpp_experiment.py
```

`tpp_experiment.py`가 통합 CLI입니다. `long-epoch`는 `common/runner.py`의 공통
runner를 사용합니다. `overfit`, `qty-ablation`은 기존 실험 구현을
`common/modes/` 아래로 이동한 상태이며, 앞으로 중복 함수를
`common/runner.py`로 더 흡수하면 됩니다.

## Model Registry

모델 추가/수정의 중심은 `common/models.py`입니다.

| 함수/객체 | 역할 |
| --- | --- |
| `canonical_model_name()` | CLI alias를 내부 모델명으로 정규화 |
| `default_thp_candidates()` | TransformerHawkesTPP preset 정의 |
| `default_titan_candidates()` | TitanTPP preset 정의 |
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

## TitanTPP Memory Modes

TitanTPP는 `TitanConfig.memory_mode`를 공식 실험 축으로 사용합니다. 기존에는
candidate 이름의 `use_lmm=True/False`로 memory 사용 여부를 간접 표현했지만, 이제
각 run metadata에 memory mode가 직접 저장됩니다.

| memory mode | 구현 내용 |
| --- | --- |
| `none` | attention-side memory와 LMM을 끈 pure causal Titan encoder |
| `static_lmm` | learnable persistent/static memory와 LMM memory bank 사용 |
| `contextual_ttm` | window 내부에서 token을 순차 처리하며 과거 token을 contextual memory로 업데이트 |
| `series_lmm` | runner가 per-series memory를 주입할 수 있는 hook. 기본 long-epoch에서는 memory 미주입 시 fallback |
| `hybrid_lmm_ttm` | contextual TTM path와 LMM retrieval을 함께 사용 |

대표 candidate:

```text
small_no_lmm              -> memory_mode=none
small_lmm                 -> memory_mode=static_lmm
small_contextual_ttm      -> memory_mode=contextual_ttm
small_hybrid_lmm_ttm      -> memory_mode=hybrid_lmm_ttm
mid_no_lmm                -> memory_mode=none
mid_lmm                   -> memory_mode=static_lmm
mid_contextual_ttm        -> memory_mode=contextual_ttm
mid_hybrid_lmm_ttm        -> memory_mode=hybrid_lmm_ttm
```

`series_lmm`은 모델 hook은 존재하지만, 현재 공통 long-epoch runner는 아직
series-specific memory tensor를 구성해 주입하지 않습니다. 따라서 논문용 본 비교에서는
`none`, `static_lmm`, `contextual_ttm`, `hybrid_lmm_ttm`을 우선 screening 대상으로 둡니다.

## TTM-Lite Evaluation

`long-epoch`는 TitanTPP에 대해 선택적 Test-Time Memory Lite 평가도 지원합니다.
`--test-time-memory contextual`을 켜면 validation/test metric export 시 series별
contextual memory를 reset하고, 관측된 이벤트만 online update합니다. 이 기능은
checkpoint 추가 평가용이며, 학습 loss나 RMTPP/THP 모델 구조는 바꾸지 않습니다.

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --datasets yellow_trip_hourly \
  --models rmtpp,titantpp,thp \
  --test-time-memory contextual
```

결과 파일:

```text
runs/.../metrics/ttm_contextual_best_val_nll.json
runs/.../metrics/ttm_contextual_best_val_nll.csv
runs/.../metrics/ttm_contextual_best_val_nll.parquet
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
`sample_data/head_office/marked_target_df.parquet`를 읽습니다.

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
| batch size | CLI `--batch-size` 값을 사용 |
| allowed Titan candidates in hparam search | `small_no_lmm`, `small_lmm` |

중요 구현 위치:

| 함수 | 역할 |
| --- | --- |
| `default_dataset_specs()` | `intermittent`를 marked target parquet로 매핑 |
| `prepare_marked_target_events()` | 필수 컬럼 검증 및 mark/residual 재계산 |
| `prepare_marked_dataset()` | marked target cache 생성 |
| `search_config_for_dataset()` | marked target runtime override 적용 |

### Yellow Trip

`tpp_experiment.py long-epoch --datasets yellow_trip_hourly`은
`simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb`로 생성한 hourly grid-cell event table을 읽습니다.
raw `yellow_trip.parquet`을 simple_lab_test 내부에서 다시 변환하지 않습니다.
변환 조건을 바꿔야 하면 root-level preprocessing notebook에서
`yellow_trip_hourly.parquet`을 재생성합니다.

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

## Unified Entrypoint

이제 search root의 standalone 실험 파일은 `tpp_experiment.py`로 통합합니다.
기존 `titan_hparam_search.py`, `titan_rmtpp_ab_test.py`,
`compare_log_bases_distribution.py`에서 쓰던 공통 로직은
`common/experiment_utils.py`와 `common/benchmark_utils.py`로 이동했습니다.

| 목적 | 현재 실행 방식 |
| --- | --- |
| Titan/RMTPP/THP 본 비교 | `tpp_experiment.py long-epoch` |
| Titan 후보 비교 | `tpp_experiment.py long-epoch --titan-candidates ...` |
| RMTPP vs TitanTPP A/B | `tpp_experiment.py long-epoch --models rmtpp,titantpp` |
| log base별 mark 분포 확인 | cache 하위 `raw_dist_base_*.parquet`, `marked_dist_base_*.parquet` 확인 |

## Removed Root-Level Experiment Scripts

아래 파일들은 root-level에서 삭제되었고, 기능은 통합 CLI로 이동했습니다.

| removed | replacement |
| --- | --- |
| `titan_rmtpp_long_epoch_scale_eval.py` | `tpp_experiment.py long-epoch` |
| `tpp_overfit_diagnostic.py` | `tpp_experiment.py overfit` |
| `tpp_qty_loss_ablation.py` | `tpp_experiment.py qty-ablation` |

## Interpretation Notes

`best_val_nll`은 sweet spot 기준입니다. final epoch metric만 보면 under-training과
overfitting을 구분하기 어렵습니다.

전체 `qty_mae`는 큰 수요 이벤트에 끌릴 수 있습니다. 따라서 `scale_wise_summary.csv`
에서 true quantity scale별 MAE/WAPE를 함께 봐야 합니다.

marked target은 episode-level event 수가 제한적입니다. 대형 Titan/THP 후보가
항상 유리하다고 가정하지 말고 `model-test`, `overfit`, `long-epoch` 순서로
학습 가능성과 generalization을 분리해서 확인합니다.
