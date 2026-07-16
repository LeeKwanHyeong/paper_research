# TitanTPP Model Enhancement Strategy

Notion documentation rule (2026-07-12): Model Enhancement의 설계, ablation,
screening, matched comparison, multi-seed 결과는 모두
`5. Model Design Enhancement` 하위에 기록한다. Validation 지표를 다루더라도
primary scope가 model enhancement이면 `2. Confirm and Refine Topic`을 사용하지 않는다.

Notion execution rule (2026-07-13): `notion_writer_prompts/`는 source draft로
보존하고, Model Enhancement Session이 해당 draft를 사용해 Notion 작성과 재조회
검증까지 직접 완료한다. 상위 history는 실험 시작 날짜를 제목 2로, 세부 목표와
Step을 제목 3으로 구성한다.

이 문서는 Model Enhancement Session에서 사용할 TitanTPP 강화 기준선을 고정하기 위한 작업 노트다. Main Developer Session의 실험 runner와 산출물은 그대로 follow-up하되, 이 세션의 초점은 hyperparameter search가 아니라 TitanTPP 모델 구조 강화에 둔다.

## 1. Baseline Contract

TitanTPP 강화 실험은 아래 세 기준선을 항상 함께 비교한다.

| Label | Role | Model / Setting | Purpose |
| --- | --- | --- | --- |
| R0 | Reference baseline | RMTPP | TitanTPP가 최소한 넘어야 할 recurrent TPP 기준선 |
| L0 | Legacy TitanTPP baseline | TitanTPP + `small_lmm`, `memory_mode=static_lmm`, `value_input_mode=none`, `loss_mode=residual_only` | 기존 TitanTPP 대비 구조 개선 효과 확인 |
| S0 | Strong TitanTPP baseline | TitanTPP V2 + `memory_mode=static_lmm`, `value_input_mode=residual`, `train_loss_scope=target_only`, `loss_mode=hybrid`; candidate는 intermittent/Instacart `small_lmm`, taxi `mid_lmm` | validation으로 선택된 dataset별 strong baseline을 넘는지 확인 |

### Interpretation

- `R0`는 강화 대상이 아니라 외부 기준선이다. TitanTPP 계열 모델은 최소한 RMTPP 대비 의미 있는 이득을 보여야 한다.
- `L0`는 legacy TitanTPP 기준선이다. 새 구조가 기존 TitanTPP보다 좋아졌는지 확인하는 최소 기준이다.
- `S0`는 multi-seed V1/V2 비교로 확정한 strong baseline이다. 이후 강화 모델은 단순히 L0만 이기는 것이 아니라, 가능하면 S0 대비 개선을 보여야 한다.
- V1 `residual_only`는 보조 guardrail로 유지한다. 특히 taxi에서 V2의 quantity 이득과 함께 marker NLL/mark accuracy가 회복되는지 확인할 때 사용한다.

## 2. Evaluation Protocol

강화 실험은 기본적으로 fixed split과 held-out test를 사용한다.

| Item | Default |
| --- | --- |
| split mode | `fixed` |
| eval selections | `best_val_nll,best_score,final` |
| primary checkpoint | `best_val_nll` |
| secondary checkpoint | `best_score` |
| recommended seeds | `42,52,62` |
| default train loss scope | `target_only` |
| auxiliary train loss scope | `all` |
| test evaluation | `target_only_nll=True` |

`target_only`를 기본으로 두는 이유는 weekly window 내부의 dense supervision 효과와 모델 구조 개선 효과가 섞이는 것을 줄이기 위해서다. `all`은 Titan encoder가 window 내부 transition supervision을 받을 때 추가 이득이 있는지 확인하는 보조 실험으로 둔다.

## 3. Primary Metrics

강화 모델의 성능은 아래 순서로 판단한다.

1. `test_score`
2. `test_qty_mae`
3. `test_nll`
4. `test_mark_acc`
5. `test_dt_mae`
6. scale-wise quantity MAE

공통 score는 현재 runner 기준을 따른다.

```text
test_score = mark_acc - 0.01 * dt_mae - 0.001 * qty_mae
```

단, score 하나만으로 모델을 판단하지 않는다. TitanTPP 강화의 핵심 목표는 quantity reconstruction, mark prediction, time intensity modeling을 함께 안정화하는 것이므로 `qty_mae`, `nll`, `mark_acc`, scale-wise MAE를 함께 본다.

## 4. Decision Rule

새 TitanTPP 강화 모델은 아래 조건을 기준으로 통과 여부를 판단한다.

- 최소한 `L0` 대비 `test_score` 또는 `test_qty_mae`에서 개선되어야 한다.
- 가능하면 `S0` 대비 최소 한 개 주요 데이터셋에서 개선되어야 한다.
- `test_nll`이 크게 악화되면 안 된다.
- high-scale bucket에서 scale-wise quantity MAE 개선이 관찰되면 긍정적인 구조 개선으로 판단한다.

## 5. Enhancement Ladder

앞으로의 구조 강화는 아래 순서로 누적 비교한다.

| Stage | Enhancement | Goal |
| --- | --- | --- |
| V1 | Value-conditioned input 정식화 | 과거 quantity state를 encoder input에 반영 |
| V2 | Hybrid quantity objective 정리 | residual loss와 final quantity MAE 간 mismatch 완화 |
| V3 | Mark-conditioned value head | mark distribution 정보를 residual prediction에 반영 |
| V4 | Mark-conditioned time head | demand scale과 inter-event time의 의존성 반영 |
| V5 | Ordinal mark modeling | log-scale mark의 순서성을 반영해 class imbalance 완화 |
| V6 | Series-aware memory | 부품/시계열별 local pattern memory 활용 |

V1/V2는 이미 코드에 상당 부분 준비된 기능이므로 기준선 정리와 실험 조건 고정에 가깝다. 실제 새 모델 구조 개발은 V3, 즉 mark-conditioned value head부터 시작한다.

## 6. Default CLI Template

기본 강화 실험 템플릿은 아래 형태를 따른다.

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/model_enhancement_<name> \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models rmtpp,titantpp \
  --titan-candidates small_lmm,mid_lmm \
  --epochs 200 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 512 \
  --max-seq-len 64 \
  --split-mode fixed \
  --value-head-activation identity \
  --train-loss-scope target_only \
  --eval-selections best_val_nll,best_score,final \
  --device cuda
```

S0 또는 value-conditioned hybrid 계열은 아래 옵션을 추가한다.

```bash
--value-input-mode residual \
--loss-mode hybrid
```

## 7. Current Status

- Step 1, baseline contract is fixed in this document.
- Step 2, code-state verification is complete.
- Step 3, S0 smoke reproduction is complete.
- Step 4, V1/V2 multi-seed baseline experiment is complete.
- Step 5, artifact analysis and pre-V3 baseline lock are complete.
- Step 6, V3 mark-conditioned value-head design is complete.
- Next step is V3 implementation and focused model tests.

## 8. Code-State Verification

2026-07-05 기준으로 모델 강화 실험에 필요한 주요 코드 경로는 연결되어 있다.

| Area | Status | Evidence |
| --- | --- | --- |
| Titan memory modes | Ready | `TitanTPP` supports `none`, `static_lmm`, `contextual_ttm`, `series_lmm`, `hybrid_lmm_ttm`. |
| Value-conditioned input | Ready | `value_input_mode` supports `none`, `residual`, `log_qty`; appended target value is masked before forward. |
| Loss scope | Ready | `train_loss_scope` supports `all` and `target_only`; runner passes it into `model.nll(...)`. |
| Quantity objective | Ready | `loss_mode` supports `residual_only`, `hybrid`, `qty_only`; training loss includes optional `qty_loss`. |
| Fixed split evaluation | Ready | Fixed-split validation/test uses `eval_next_event_week_lookback(..., target_only_nll=True)`. |
| Scale-wise diagnostics | Ready | Runner writes validation/test scale-wise quantity metrics for selected checkpoints. |
| Classic event-count path | Auxiliary | Classic loaders/trainers exist in `utils.training`, but the official `tpp_experiment.py long-epoch` path currently uses weekly/fixed-split loaders. |

Important caveat: `yellow_trip_hourly` applies dataset-specific overrides and forces at least `lookback_weeks=168` and `max_seq_len=256`. If a future experiment needs smaller yellow-trip windows, the override policy must be changed or a separate dataset/loader path should be used.

Interface smoke test passed with:

```bash
python simple_lab_test/search/tpp_experiment.py model-test \
  --output-dir /private/tmp/tpp_model_test_step2 \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm,mid_lmm \
  --thp-candidates small \
  --device cpu \
  --batch-size 4 \
  --seq-len 12 \
  --num-marks 6 \
  --left-pad \
  --stop-on-error
```

Observed model-test outputs:

| Model | Candidate | Hidden shape | Synthetic NLL |
| --- | --- | --- | --- |
| RMTPP | `rmtpp_gru_h64` | `[4, 12, 64]` | `3.858925` |
| TitanTPP | `small_lmm` | `[4, 12, 64]` | `3.814501` |
| TitanTPP | `mid_lmm` | `[4, 12, 128]` | `3.815739` |
| THP | `small` | `[4, 12, 64]` | `3.862728` |

## 9. S0 Smoke Reproduction

S0 경로가 공식 `long-epoch` runner에서 실제 학습, validation, test export까지 통과하는지 확인했다.

Smoke command:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/model_enhancement_s0_smoke_e1 \
  --datasets insta_market_basket \
  --models titantpp \
  --titan-candidates mid_lmm \
  --epochs 1 \
  --seeds 42 \
  --lr 1e-3 \
  --batch-size 16 \
  --lookback-weeks 10 \
  --max-seq-len 16 \
  --insta-max-series 20 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-input-mode residual \
  --loss-mode hybrid \
  --train-loss-scope target_only \
  --eval-selections best_val_nll,best_score,final \
  --device cpu \
  --force-rerun \
  --stop-on-error
```

Run scope:

| Item | Value |
| --- | --- |
| dataset | `insta_market_basket` |
| subset | top `20` series |
| rows | `2,000` |
| train / validation / test samples | `1,380 / 300 / 300` |
| model | TitanTPP `mid_lmm` |
| S0 options | `value_input_mode=residual`, `loss_mode=hybrid`, `train_loss_scope=target_only` |

Smoke metrics:

| Split | Selection | Score | NLL | Mark Acc | DT MAE | Qty MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | epoch 1 | `0.463217` | `3.345388` | `0.483333` | `1.443142` | `5.685064` |
| test | `best_val_nll` | `0.505584` | `3.228712` | `0.523333` | `1.236994` | `5.379689` |
| test | `best_score` | `0.505584` | `3.228712` | `0.523333` | `1.236994` | `5.379689` |
| test | `final` | `0.505584` | `3.228712` | `0.523333` | `1.236994` | `5.379689` |

Generated artifacts include:

- `leaderboard/summary.csv`
- `leaderboard/test_summary.csv`
- `leaderboard/scale_wise_summary.csv`
- `leaderboard/test_scale_wise_summary.csv`
- `checkpoints/best_val_nll_model.pt`
- `checkpoints/best_score_model.pt`
- `checkpoints/final_model.pt`

Interpretation: this is a smoke reproduction only. Because it uses 1 epoch and 20 Instacart series, the metrics should not be used as model-quality evidence. The useful conclusion is that S0 is wired correctly through fixed split data preparation, value-conditioned TitanTPP input, hybrid quantity loss, target-only training scope, checkpoint export, and held-out test evaluation.

## 10. V1/V2 Baseline Finalization

V1/V2는 새 model head를 추가하기 전의 immediate baseline이다.

| Stage | Meaning | CLI options | Role |
| --- | --- | --- | --- |
| L0 | Legacy TitanTPP | `value_input_mode=none`, `train_loss_scope=all`, `loss_mode=residual_only` | 과거 TitanTPP 기준선 |
| V1 | Value-conditioned input | `value_input_mode=residual`, `train_loss_scope=target_only`, `loss_mode=residual_only` | score/NLL 중심 strong baseline |
| V2 | Value-conditioned hybrid objective | `value_input_mode=residual`, `train_loss_scope=target_only`, `loss_mode=hybrid` | quantity MAE guardrail baseline |

Existing Instacart e200 single-seed artifacts suggest the following:

| Variant | Best checkpoint view | Candidate | Test score | Test NLL | Test qty MAE | Test mark acc |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| L0 | `best_score` | `small_no_lmm` | `0.433301` | `4.534142` | `4.478428` | `0.499309` |
| V1 | `best_val_nll` / `best_score` | `mid_lmm` | `0.437442` | `4.458443` | `4.432567` | `0.504012` |
| V2 | `best_val_nll` | `mid_lmm` | `0.436217` | `4.467828` | `4.407311` | `0.503090` |
| V2 | `best_score` | `small_lmm` | `0.436809` | `4.468350` | `4.411538` | `0.503830` |

Interpretation before multi-seed confirmation:

- V1 is the strongest pre-V3 baseline for score, total NLL, marker NLL, and mark accuracy.
- V2 improves direct quantity MAE relative to V1, but can slightly trade off score/NLL.
- Therefore V3 should be compared against both V1 and V2: V1 as the primary score/NLL baseline, V2 as the quantity-MAE guardrail.
- The existing e200 artifacts are useful but not final because they are seed-42 only.

Multi-seed baseline confirmation completed on `5090`.

| Item | Value |
| --- | --- |
| start time | `2026-07-05 20:34:30 KST` |
| V1 end time | `2026-07-07 16:01:47 KST` |
| V2 end time | `2026-07-09 12:14:48 KST` |
| status | V1/V2 each `18/18` runs complete; no NaN, Traceback, or runtime ERROR |
| server | `5090` |
| tmux session | `titantpp_v1_v2_baseline_e200_0705` |
| conda env | `ai_env` |
| V1 base dir | `search_artifacts/model_enhancement_v1_residual_e200_0705` |
| V2 base dir | `search_artifacts/model_enhancement_v2_hybrid_e200_0705` |
| datasets | `intermittent,yellow_trip_hourly,insta_market_basket` |
| candidates | `small_lmm,mid_lmm` |
| epochs / seeds | `200` / `42,52,62` |
| batch size | `128` |
| split mode | `fixed` |

V1 command:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/model_enhancement_v1_residual_e200_0705 \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models titantpp \
  --titan-candidates small_lmm,mid_lmm \
  --epochs 200 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 64 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode residual_only \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --stop-on-error
```

V2 command:

```bash
python simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir search_artifacts/model_enhancement_v2_hybrid_e200_0705 \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models titantpp \
  --titan-candidates small_lmm,mid_lmm \
  --epochs 200 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 128 \
  --max-seq-len 64 \
  --split-mode fixed \
  --value-head-activation identity \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --loss-mode hybrid \
  --eval-selections best_val_nll,best_score,final \
  --device cuda \
  --stop-on-error
```

Final V1/V2 lock rule:

- Read artifacts in the protocol order: manifest, log, summary, test_summary, histories, scale-wise metrics, plots.
- If V1 keeps better `test_score` and `test_nll`, use V1-mid_lmm as the primary pre-V3 baseline.
- If V2 keeps better `test_qty_mae` without large NLL degradation, use V2 as the quantity-focused guardrail.
- If dataset-specific candidate behavior diverges, lock candidate per dataset instead of forcing one candidate globally.

## 11. Multi-Seed Result Interpretation And Pre-V3 Lock

Candidate selection used validation mean best NLL only. Held-out test metrics were read after each dataset/variant candidate was fixed, avoiding test-driven candidate selection.

| Dataset | V1 selected | V2 selected | V1 val NLL | V2 val NLL |
| --- | --- | --- | ---: | ---: |
| `insta_market_basket` | `mid_lmm` | `small_lmm` | `4.379298` | `4.381901` |
| `intermittent` | `small_lmm` | `small_lmm` | `5.629190` | `5.604595` |
| `yellow_trip_hourly` | `small_lmm` | `mid_lmm` | `1.568134` | `1.576568` |

Held-out test comparison uses the `best_val_nll` checkpoint. Delta is V2 minus V1; positive score and negative error are improvements.

| Dataset | V1 score | V2 score | Score delta | V1 qty MAE | V2 qty MAE | Qty MAE change | Test NLL change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `insta_market_basket` | `0.436657` | `0.437034` | `+0.000377` | `4.452236` | `4.405873` | `-1.04%` | `+0.04%` |
| `intermittent` | `0.279412` | `0.287139` | `+0.007727` | `3.365281` | `3.030658` | `-9.94%` | `-0.32%` |
| `yellow_trip_hourly` | `0.824321` | `0.840379` | `+0.016058` | `79.172297` | `58.700291` | `-25.86%` | `+1.23%` |

### Interpretation

- `intermittent`: V2 is a clear win. All three seeds improve score and quantity MAE, and every populated scale bucket improves. The improvement is broad rather than tail-only.
- `yellow_trip_hourly`: V2 strongly improves mean quantity MAE and keeps time NLL stable over long training. V1 reaches best NLL at epochs `4-13` for the selected `small_lmm` and then degrades, whereas V2 `mid_lmm` reaches best NLL at epochs `148-184` with only `0.012-0.017` final-minus-best degradation. However, V2 test marker NLL worsens by about `13.2%`, mark accuracy falls by `0.004764`, and seed 42 regresses in score and quantity MAE. This is promising but not fully seed-stable.
- `insta_market_basket`: V2 gives a small but consistent aggregate improvement. All three seeds improve score and quantity MAE, but the gain is concentrated in the `1-9` bucket (`-4.87%` MAE); the `10-99` bucket worsens by `1.47%`. The `100-999` bucket is only `0.0036%` of test rows and is not reliable evidence.
- V1 taxi instability is driven mainly by time NLL growth after the early optimum. V2 removes this late-epoch collapse, so hybrid quantity supervision also acts as an optimization stabilizer in the long weekly sequence regime.

### Baseline Lock

- Primary pre-V3 baseline: V2 hybrid TitanTPP.
- Dataset candidates: `intermittent=small_lmm`, `insta_market_basket=small_lmm`, `yellow_trip_hourly=mid_lmm`.
- Auxiliary guardrail: V1 `residual_only`, especially taxi marker NLL and mark accuracy.
- V3 success target: preserve V2 quantity and stability gains while recovering taxi marker NLL/accuracy and reducing seed variance.

Reproducible analysis notebook:

```text
simple_lab_test/notebooks/experiments/titantpp_v1_v2_baseline_analysis.ipynb
```

## 12. V3 Mark-Conditioned Value Head Design

V3는 현재 V2의 shared residual head를 유지하면서 mark별 residual delta expert를 추가한다.

```text
r_shared = value_head(h_t)
delta_by_mark = value_mark_delta_head(h_t)
r_by_mark[k] = activation(r_shared + delta_by_mark[k])
```

Training에서는 true next mark에 해당하는 residual expert만 residual Huber loss로 감독하고, hybrid quantity loss는 모든 real-mark expert의 quantity를 predicted mark probability로 가중한다.

```text
value_loss = Huber(r_by_mark[m_true], residual_true)
expected_qty = sum_k p(k | h_t) * base^(k + r_by_mark[k])
```

Inference에서는 predicted mark에 해당하는 residual을 선택한다. Ground-truth target mark/value는 encoder 입력에 사용하지 않는다.

V3 설계 결정:

- config: `value_head_mode=shared|mark_conditioned_experts`
- CLI: `--value-head-mode shared|mark_conditioned_experts`
- default: `shared`, 기존 V2와 backward compatible
- V3: `mark_conditioned_experts`
- shared `value_head`는 유지하고 mark delta head만 추가
- mark delta head는 zero initialization하여 V3 시작점을 V2와 동일하게 고정
- first V3 ablation에서는 quantity-to-mark gradient coupling을 V2와 동일하게 유지
- marker NLL trade-off가 지속될 때만 detached-gate V3b를 후속 비교
- run path와 manifest에 `value_head_mode`를 포함해 V2/V3 artifact 충돌 방지
- evaluation residual metric은 true-mark branch, reconstructed quantity는 predicted-mark branch 사용
- time head, Titan memory, lookback, dataset split은 변경하지 않음

상세 ADR:

```text
.agents/results/architecture/adr-titantpp-v3-mark-conditioned-value-head.md
```

## 13. V3a Screening Result And V3b Detached-Gate Decision

V3a seed-42 e50 screening은 mark-conditioned experts가 quantity/value
modeling에는 유효하지만 marker guardrail은 통과하지 못했음을 보여준다.

`best_val_nll` held-out test 기준:

| Dataset | Quantity MAE | Value MAE | Total NLL | Marker NLL | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `intermittent` | `-13.315%` | `-23.748%` | `-0.194%` | `-1.004%` | `-1.023%p` |
| `yellow_trip_hourly` | `-6.988%` | `-11.075%` | `+2.484%` | `+17.501%` | `-0.745%p` |

Scale-wise result도 일관되지 않았다.

- Intermittent: 전체의 `88.67%`인 `1-9` bucket MAE가 `9.21%` 악화됐고,
  더 큰 quantity bucket에서 개선됐다.
- Taxi: `1-9`, `1000-9999`는 개선됐지만 `10-999`는 악화됐다.
- 두 데이터셋 모두 낮은 mark의 예측 비중/정확도가 증가하고 일부 중간
  mark accuracy가 감소했다.

따라서 V3a는 `partial success`로 판정한다.

- mark-conditioned expert architecture는 유지
- V3a를 V2 baseline replacement로 채택하지 않음
- full multi-seed 본실험으로 바로 확장하지 않음
- next architecture는 V3b detached quantity gate

V3b는 forward quantity expectation을 바꾸지 않고 mark-probability gate의
gradient만 차단한다.

```text
mark_probs = softmax(mark_logits_real)

if qty_mark_gradient_mode == "coupled":
    qty_gate = mark_probs
else:
    qty_gate = stop_gradient(mark_probs)

qty_per_mark[k] = base^(k + r_by_mark[k])
expected_qty = sum_k qty_gate[k] * qty_per_mark[k]
```

공식 variant contract:

| Variant | `value_head_mode` | `qty_mark_gradient_mode` |
| --- | --- | --- |
| V2 | `shared` | `coupled` |
| V3a | `mark_conditioned_experts` | `coupled` |
| V3b | `mark_conditioned_experts` | `detached` |

V3b gradient contract:

- isolated quantity loss에서 `mark_head` gradient는 zero/None
- quantity loss에서 shared value head와 populated mark experts gradient는 유지
- shared encoder는 value branch를 통해 quantity gradient를 계속 받음
- marker CE/full loss에서는 `mark_head` gradient가 유지
- V3a/V3b forward value와 loss value는 동일하고 backward graph만 다름

Configuration and artifact decision:

- config/CLI: `qty_mark_gradient_mode=coupled|detached`
- CLI name: `--qty-mark-gradient-mode`
- default: `coupled`, 기존 V2/V3a behavior와 run path 유지
- detached path: `qtymarkgrad_detached`
- manifest, cache identity, run/test/scale/confusion rows에 mode 기록
- first V3b implementation은 TitanTPP-only; detached non-Titan 실행은 명시적으로 거부

V3b seed-42 e50 short gate:

- total NLL이 V2보다 `0.5%` 이상 악화되지 않을 것
- Taxi marker NLL regression을 현재 `17.5%`에서 최대 `2%` 이내로 축소
- 두 데이터셋 mark accuracy gap을 V2 대비 `0.25%p` 이내로 축소
- V3a aggregate quantity-MAE gain의 절반 이상 유지
- test share `5%` 이상 quantity bucket에서 V2 대비 `5%` 초과 regression 없음
- gate 통과 후에만 seeds `42,52,62`로 확장

상세 ADR:

```text
.agents/results/architecture/adr-titantpp-v3b-detached-quantity-gate.md
```

## 14. V3b Screening Result And Dataset-Specific Decision

V3b implementation과 seed-42 e50 screening은 2026-07-10에 완료됐다.
Focused test는 gradient-routing contract와 V3a/V3b forward/loss exact
equivalence를 확인했고, Instacart top-20 e1 smoke도 통과했다.

`best_val_nll` held-out test에서 V3b와 V2를 비교한 결과:

| Dataset | Quantity MAE | Value MAE | Total NLL | Marker NLL | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `intermittent` | `-1.833%` | `-1.770%` | `-0.268%` | `-1.193%` | `-1.093%p` |
| `yellow_trip_hourly` | `-36.128%` | `-23.879%` | `-1.990%` | `-16.124%` | `+1.057%p` |

Intermittent 판정:

- total NLL과 marker NLL은 V2보다 개선
- `1-9` bucket MAE는 `+3.640%`, `10-99`는 `-1.621%`로 주요
  bucket regression gate는 통과
- V2 대비 mark accuracy gap `-1.093%p`로 accuracy gate 실패
- 데이터셋 단독으로는 V3a quantity gain의 `13.77%`만 유지
- validation quantity MAE는 V2보다 악화됐지만 test에서는 소폭 개선돼
  quantity 방향도 완전히 일치하지 않음

Taxi 판정:

- total NLL, marker NLL, quantity MAE, value MAE, mark accuracy gate를 모두 통과
- 데이터셋 단독으로 V3a quantity gain의 `517.01%`를 달성해 V3a보다도
  quantity MAE가 크게 개선
- test share가 `5%` 이상인 모든 bucket에서 V2 대비 개선:
  `1-9 -14.165%`, `10-99 -24.917%`, `100-999 -32.027%`,
  `1000-9999 -40.248%`
- validation과 held-out test 모두 V2보다 NLL/quantity 방향이 개선돼 방향 일치

Decision:

- V3b는 `dataset-specific success`, 전체 공통 gate는 `failed`로 판정
- 두 데이터셋 평균 quantity 개선폭은 V3a `10.151%`, V3b `18.981%`로
  aggregate quantity-retention은 `186.98%`이며 공식 retention gate는 통과
- 전체 공통 gate의 유일한 실패 항목은 Intermittent mark accuracy gap
- V2는 전체 데이터셋 공통 baseline으로 유지
- V3b는 Taxi 전용 승격 후보로 유지하고 seeds `42,52,62` 확인을 우선
- Intermittent에는 V3b를 승격하지 않고 shared encoder에 전달되는 value/quantity
  gradient까지 조절하는 후속 gradient-routing 설계를 검토
- seed-42 e50 결과만으로 최종 우월성을 주장하지 않음

Artifacts:

```text
search_artifacts/model_enhancement_v3b_insta_smoke_e1_0710
search_artifacts/model_enhancement_v3b_inter_short_e50_0710
search_artifacts/model_enhancement_v3b_taxi_short_e50_0710
```

## 15. V3b Taxi Multi-Seed Confirmation

Taxi V3b seeds `42,52,62` e50 confirmation은 2026-07-10에 완료됐다.
5090은 GPU가 비어 있었지만 tmux가 설치되어 있지 않아 표준 장시간 실행 조건을
만족하지 못했고, tmux와 `ai_env`가 검증된 5080에서 실행했다.

`best_val_nll` held-out test 기준 V3b 절대값:

| Metric | Mean | Std | CV |
| --- | ---: | ---: | ---: |
| Total NLL | `1.611892` | `0.000590` | `0.037%` |
| Quantity MAE | `38.312184` | `4.635932` | `12.100%` |
| Marker NLL | `0.208618` | - | - |
| Mark accuracy | `91.902%` | - | - |
| Value MAE | `0.151485` | - | - |

기존 V2 multi-seed e200 `mid_lmm`과 비교한 평균 변화:

| Total NLL | Marker NLL | Time NLL | Quantity MAE | Value MAE | Mark accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `-2.357%` | `-17.015%` | `+0.277%` | `-34.733%` | `-24.247%` | `+1.209%p` |

Seed-matched V2 e200 대비 변화:

| Seed | Total NLL | Marker NLL | Quantity MAE | Mark accuracy | Simultaneous gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | `-2.174%` | `-15.637%` | `-51.918%` | `+0.733%p` | PASS |
| 52 | `-2.278%` | `-16.105%` | `+13.017%` | `+1.045%p` | FAIL: quantity |
| 62 | `-2.618%` | `-19.219%` | `-40.092%` | `+1.849%p` | PASS |

Scale-wise test mean은 share `5%` 이상인 모든 bucket에서 V2보다 개선됐다.

- `1-9`: `-8.095%`
- `10-99`: `-27.444%`
- `100-999`: `-37.833%`
- `1000-9999`: `-34.612%`

Stability interpretation:

- `3/3` run 완료, NaN/Traceback 없음
- test NLL std는 V2보다 `86.99%`, quantity MAE std는 `80.81%` 감소
- total NLL 개선은 time NLL이 아니라 marker NLL 개선이 주도
- seed-matched NLL/marker/quantity 동시개선은 `2/3`으로 confirmation gate 통과
- final test NLL은 best-val-NLL checkpoint보다 평균 `4.758%` 악화
- 특히 seed 52는 final validation NLL이 best보다 `12.913%` 악화해
  best-checkpoint selection 또는 early stopping이 필수
- 동일 seed 42 재실행도 quantity MAE가 달라 CUDA-level non-determinism을
  완전히 배제할 수 없음

Decision:

- V3b는 Taxi 전용 `confirmed next-stage candidate`로 승격
- V2는 전체 데이터셋 공통 baseline으로 유지
- V3b를 모든 데이터셋의 공통 replacement로 승격하지 않음
- 논문 최종 비교 전에는 V2/V3b의 동일 e50 multi-seed budget 비교가 필요
- V3b 장기학습은 final checkpoint가 아니라 validation-selected checkpoint와
  early stopping을 전제로 설계

Artifact:

```text
search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710
```

## 16. V2/V3b Taxi Strict Matched-Budget Confirmation

Taxi V2 seeds `42,52,62` e50 실험은 2026-07-10에 5090에서 완료됐으며,
V3b e50과 dataset, split, seed, epoch, learning rate, batch size, lookback,
`max_seq_len`, loss, selection 기준을 일치시킨 strict comparison을 구성했다.
세 run 모두 정상 종료됐고 NaN, Traceback, ERROR는 없었다.

`best_val_nll` checkpoint의 held-out test 평균 비교:

| Metric | V2 e50 | V3b e50 | Change |
| --- | ---: | ---: | ---: |
| Total NLL | `1.650430` | `1.611892` | `-2.335%` |
| Marker NLL | `0.249688` | `0.208618` | `-16.448%` |
| Time NLL | `1.400742` | `1.403274` | `+0.181%` |
| Quantity MAE | `75.249345` | `38.312184` | `-49.086%` |
| Value MAE | `0.208378` | `0.151485` | `-27.303%` |
| Mark accuracy | `91.173%` | `91.902%` | `+0.729%p` |

Seed-matched 변화:

| Seed | Total NLL | Marker NLL | Quantity MAE | Mark accuracy | Simultaneous gate |
| ---: | ---: | ---: | ---: | ---: | --- |
| 42 | `-1.664%` | `-12.924%` | `-16.446%` | `+0.540%p` | PASS |
| 52 | `-2.545%` | `-16.633%` | `-30.574%` | `+0.877%p` | PASS |
| 62 | `-2.788%` | `-19.543%` | `-71.956%` | `+0.769%p` | PASS |

Scale-wise held-out test quantity MAE 변화:

| Scale | V2 e50 | V3b e50 | Change |
| --- | ---: | ---: | ---: |
| `1-9` | `1.646924` | `1.561491` | `-5.187%` |
| `10-99` | `12.996297` | `11.434228` | `-12.019%` |
| `100-999` | `167.416043` | `81.922968` | `-51.066%` |
| `1000-9999` | `635.561996` | `308.954200` | `-51.389%` |

Stability:

- test NLL std는 V2 `0.010487`에서 V3b `0.000590`으로 `94.378%` 감소
- quantity MAE std는 V2 `37.070037`에서 V3b `4.635932`로 `87.494%` 감소
- NLL CV는 V2 `0.635%`, V3b `0.037%`
- quantity MAE CV는 V2 `49.263%`, V3b `12.100%`
- V2 seed 62의 quantity MAE `117.585907`이 평균과 분산을 크게 높였지만,
  V3b는 해당 seed뿐 아니라 세 seed 각각에서 quantity MAE를 개선

Interpretation:

- V3b의 Taxi 성능 향상은 epoch budget mismatch나 단일 seed에 의한 결과가 아님
- total NLL 개선은 time head가 아니라 marker head 개선이 주도하며,
  time NLL은 `0.181%` 소폭 악화
- quantity 개선은 모든 scale에서 재현됐고 특히 `100-999`, `1000-9999`
  대수요 bucket에서 약 `51%` 개선
- V2 validation best epoch가 `48-50`에 있어 e50 이후 추가 개선 가능성은 남지만,
  동일 budget에서의 V3b 우위와 seed 안정성은 명확함

Decision:

- V3b를 Taxi 전용 `confirmed model enhancement`로 유지
- Taxi 후속 본실험의 기본 value head는 `mark_conditioned_experts + detached`로 설정
- V2는 Intermittent를 포함한 전체 데이터셋 공통 baseline으로 유지
- V3b를 모든 데이터셋의 공통 replacement로 승격하지 않음
- Taxi 장기학습 비교는 validation-selected checkpoint와 early stopping을 적용한
  V2/V3b 동일 e100 또는 e200 조건으로 진행

Artifacts:

```text
search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710
search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710
```

## 17. V3c Intermittent Shared-Encoder Gradient Routing Design

V3b가 Intermittent mark accuracy gate를 회복하지 못했으므로 남아 있는
value/quantity-to-encoder gradient path를 다음 격리 대상으로 정했다.

동일 seed-42 e50 `best_val_nll` held-out test 근거:

| Variant | Total NLL | Marker NLL | Quantity MAE | Value MAE | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 | `5.071916` | `1.016321` | `3.528298` | `0.153685` | `54.460%` |
| V3a | `5.062077` | `1.006122` | `3.058517` | `0.117189` | `53.437%` |
| V3b | `5.058310` | `1.004198` | `3.463607` | `0.150965` | `53.367%` |
| V3c | `5.143232` | `1.023542` | `3.613536` | `0.128854` | `53.674%` |

V3a와 V3b는 V2보다 total/marker NLL이 낮지만 mark accuracy는 각각
`1.023%p`, `1.093%p` 낮다. V3b의 mark-probability gate detachment가
Intermittent accuracy를 회복하지 못했으므로, direct gate보다 shared encoder로
전달되는 value/quantity gradient가 다음 검증 가설이다. 이는 아직 원인으로
확정된 것이 아니며 class imbalance와 checkpoint 변동성도 남아 있다.

Design-twice 비교:

| Option | Forward/parameter 유지 | 가설 분리력 | 비용 | 첫 V3c 판정 |
| --- | --- | --- | --- | --- |
| value hidden 완전 detach | 예 | 높음 | 낮음 | 선택 |
| encoder gradient partial scaling | 예 | 중간 | scale sweep 필요 | 후속 후보 |
| detached value adapter | 아니오 | 중간 | parameter/추론 증가 | 보류 |
| PCGrad/GradNorm | 예 | 중간 | optimizer 복잡도 증가 | 보류 |

V3c 공식 contract:

| Variant | `value_head_mode` | `qty_mark_gradient_mode` | `value_encoder_gradient_mode` |
| --- | --- | --- | --- |
| V2 | `shared` | `coupled` | `coupled` |
| V3a | `mark_conditioned_experts` | `coupled` | `coupled` |
| V3b | `mark_conditioned_experts` | `detached` | `coupled` |
| V3c | `mark_conditioned_experts` | `detached` | `detached` |

Forward/backward decision:

```text
h_main = h_j
h_value = h_j                         # coupled
h_value = stop_gradient(h_j)          # detached V3c

mark_logits = mark_head(h_main)
time_density = time_head(h_main)
value_by_mark = value_heads(h_value)
```

- V3b와 V3c의 parameter, initialization, forward output, loss value는 동일
- V3c에서 marker/time loss는 Titan encoder를 계속 학습
- value/quantity loss는 shared value head와 mark-delta experts를 계속 학습
- value/quantity loss의 encoder, embedding, LMM gradient는 zero/None
- V3b의 detached mark gate를 유지하므로 quantity loss의 mark-head gradient도 없음
- inference와 public value prediction helper는 변경하지 않음

Configuration and artifact identity:

- config/CLI: `value_encoder_gradient_mode=coupled|detached`
- CLI: `--value-encoder-gradient-mode`
- default `coupled`; 기존 V2/V3a/V3b behavior와 run path 유지
- V3c path segment: `valueencgrad_detached`
- V3c detached는 TitanTPP-only,
  `mark_conditioned_experts + qtymarkgrad_detached` 조합에서만 허용
- manifest, checkpoint, cache identity, histories, validation/test/scale/confusion,
  model-test, report grouping에 mode 기록

Intermittent seed-42 e50 short gate:

- mark accuracy gap vs V2 `>= -0.25%p`
- total NLL regression vs V2 `<= 0.5%`
- marker NLL regression vs V2 `<= 2%`
- quantity MAE와 value MAE regression vs V2 각각 `<= 2%`
- test share `5%` 이상 quantity bucket regression vs V2 `<= 5%`
- validation/test의 marker accuracy와 quantity 방향 일치
- gate 통과 후에만 seeds `42,52,62` e50으로 확장

Decision branches:

- marker와 quantity safety 모두 통과: V3c Intermittent multi-seed 진행
- marker 회복, quantity safety 실패: V2 유지 후 partial gradient 설계 검토
- marker 회복 실패: 추가 encoder detachment는 중단하고 shared-encoder gradient
  가설의 우선순위를 낮춘 뒤 class imbalance 또는 ordinal marker objective 분석으로 전환
- Taxi는 strict matched confirmation을 통과한 V3b를 유지하고 V3c를 적용하지 않음

상세 ADR:

```text
.agents/results/architecture/adr-titantpp-v3c-detached-value-encoder-route.md
```

Implementation status (2026-07-12):

- `value_encoder_gradient_mode=coupled|detached` config/CLI/model/artifact propagation 완료
- V3c는 `h_j.detach()`를 value branch에만 적용하며 marker/time branch는 원본 hidden 유지
- V3b/V3c parameter, initialization, forward, loss exact-equivalence test 통과
- isolated value/quantity loss의 encoder gradient 차단과 value-head gradient 유지 확인
- marker/time/full loss의 encoder 및 해당 head gradient 유지 확인
- V3c distinct path와 invalid combination fail-fast 확인
- static LMM을 포함한 focused pytest `18/18` 통과
- V3c `small_lmm` CPU model-test 및 기본 RMTPP/TitanTPP/THP regression model-test 통과
- 5090 CUDA preflight와 Instacart top-20 e1 integration smoke 통과
- Intermittent seed-42 e50 `50/50` epochs 완료, runtime/artifact 오류 없음

Screening result (2026-07-12):

| Gate | V3c vs V2 | Result |
| --- | ---: | --- |
| mark accuracy gap `>= -0.25%p` | `-0.786%p` | FAIL |
| total NLL regression `<= 0.5%` | `+1.406%` | FAIL |
| marker NLL regression `<= 2%` | `+0.710%` | PASS |
| quantity MAE regression `<= 2%` | `+2.416%` | FAIL |
| value MAE regression `<= 2%` | `-16.157%` | PASS |
| share `>= 5%` bucket regression `<= 5%` | `1-9: +12.935%` | FAIL |
| validation/test direction agreement | aggregate quantity 불일치 | FAIL |

Decision:

- V3c는 V3b 대비 mark accuracy를 `0.307%p` 회복했지만 V2 허용 gap에 도달하지 못함
- primary `best_val_nll` checkpoint에서 marker NLL과 value MAE만 gate 통과
- final checkpoint도 total NLL `+0.861%`, mark accuracy `-0.295%p`,
  `1-9` MAE `+13.093%`로 strict gate 실패
- V3c를 Intermittent multi-seed로 승격하지 않고 V2 baseline 유지
- additional encoder detachment를 중단하고 class imbalance 또는 ordinal marker
  objective를 다음 모델 강화 후보로 전환
- Taxi V3b confirmed decision은 유지

Artifact:

```text
search_artifacts/model_enhancement_v3c_inter_short_e50_0712
```

## 18. Intermittent Class Imbalance And Mark Confusion Diagnostic

V3c short gate 실패 후 fixed split의 next-event target 분포와 V2/V3a/V3b/V3c
`best_val_nll` confusion artifact를 동일한 grain으로 재계산했다. 첫 event는
`RMTPPWeekLookbackDataset`과 동일하게 target에서 제외했다.

Source validation:

- actual target count는 train `136,256`, validation `41,901`, test `41,344`
- 모든 variant의 validation/test confusion true-class count가 fixed split과 일치
- confusion에서 재계산한 mark accuracy가 각 leaderboard 값과 일치
- 비교 조건은 Intermittent, seed `42`, e50, held-out test `best_val_nll`

Fixed-split target distribution:

| Split | Majority share | Marks 0-2 | Marks 4+ | Effective classes | TV vs train |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | `39.24%` | `86.60%` | `5.62%` | `4.31` | `0.0000` |
| validation | `41.18%` | `87.19%` | `4.63%` | `4.18` | `0.0233` |
| test | `41.67%` | `87.39%` | `4.74%` | `4.17` | `0.0253` |

Test에는 mark `0/1/2`가 각각 `41.675% / 28.914% / 16.801%`를 차지한다.
반면 mark `6-10`은 각각 `1%` 미만이고 mark `10`은 `16`개뿐이다. 따라서
imbalance는 분명하지만 validation/test drift는 작아 split 변화가 V3 계열 간 차이를
설명하는 주원인은 아니다.

Held-out test confusion diagnostics:

| Variant | Accuracy | Balanced acc. | Macro F1 | Adjacent error share | Mark MAE | Pred mark-0 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | `54.460%` | `36.400%` | `0.3733` | `86.658%` | `0.5216` | `42.529%` |
| V3a | `53.437%` | `41.703%` | `0.4329` | `82.604%` | `0.5540` | `59.443%` |
| V3b | `53.367%` | `42.697%` | `0.4448` | `83.470%` | `0.5507` | `57.617%` |
| V3c | `53.674%` | `36.373%` | `0.3701` | `83.110%` | `0.5483` | `52.545%` |

Confirmed findings:

- V3a/V3b는 overall accuracy가 V2보다 낮지만 balanced accuracy와 macro F1은 높다.
  일부 low-support class recall을 얻는 대신 high-support mark `1`을 잃는
  macro/micro trade-off다.
- V3c의 핵심 regression은 mark `0/1` decision boundary에 집중된다.
  `1 -> 0` confusion은 V2 `40.11%`에서 V3c `56.16%`로 증가했고,
  V3c 전체 error의 `35.05%`를 차지한다.
- V3c accuracy delta를 class별로 분해하면 mark `0`은 `+3.916%p`, mark `1`은
  `-5.316%p`, mark `2`는 `+0.588%p`, mark `3`은 `-0.508%p`, mark `4`는
  `+0.551%p`를 기여해 net `-0.786%p`가 된다.
- 모든 모델에서 error 대부분이 인접 mark 사이에 있다. V2는 `86.66%`, V3c는
  `83.11%`로, unordered tail failure보다 ordered boundary error가 지배적이다.

Interpretation, not causal proof:

- class imbalance는 `0/1` 경계를 민감하게 만드는 조건이지만, 동일 target에서
  variant별 경계가 다르게 움직였으므로 imbalance만으로 regression을 설명할 수 없다.
- raw inverse-frequency weighting은 mark `10`처럼 support가 극단적으로 작은 class의
  gradient를 과도하게 증폭할 수 있어 첫 변경으로 사용하지 않는다.
- 첫 ordinal prototype은 기존 marker CE를 유지하고 작은 ordered-distance auxiliary를
  추가한다. prior correction은 capped effective-number 또는 logit adjustment로 별도
  ablation해 ordinal 구조 효과와 섞지 않는다.

Decision:

- Intermittent baseline은 V2로 유지하고 V3c multi-seed는 실행하지 않음
- 추가 gradient-routing variant 개발은 중단
- Intermittent 다음 구조 실험은 V5 ordinal marker objective를 우선
- Taxi는 V3b confirmed model을 유지하고 V4 mark-conditioned time head를 별도 트랙으로 진행
- V5 gate에는 overall accuracy와 marker NLL뿐 아니라 balanced accuracy, macro F1,
  mark `0/1` recall, adjacent error share를 함께 포함

Artifacts:

```text
simple_lab_test/search/analyze_intermittent_mark_diagnostics.py
search_artifacts/model_enhancement_inter_mark_diagnostics_0712
```

## 19. V5a Ordinal Marker Loss Contract And Acceptance Gate

Intermittent의 다음 구조는 실패한 V3c에 누적하지 않고 확정 baseline인 V2에서
분기한다. V5a는 marker head parameter나 inference를 바꾸지 않고 기존 categorical
CE에 normalized Ranked Probability Score(RPS)를 보조항으로 추가한다.

Design-twice 비교:

| Option | 순서 반영 | Parameter/inference 변경 | 확률 분포 해석 | 첫 V5 판정 |
| --- | --- | --- | --- | --- |
| CE + normalized RPS | cumulative class order | 없음 | CE 유지, RPS 별도 | 선택 |
| CE + expected absolute distance | 직접 거리 | 없음 | point-estimate 편향 위험 | 보류 |
| CORAL cumulative head | threshold order | 있음 | 새 decoding 필요 | 보류 |
| class-balanced/focal/logit adjustment | 빈도 중심 | 없음 또는 inference 변경 | ordinal 가설과 혼합 | V5b 별도 |

V5a 공식 variant contract:

| Variant | `value_head_mode` | `qty_mark_gradient_mode` | `value_encoder_gradient_mode` | `marker_loss_mode` | `lambda_ordinal` |
| --- | --- | --- | --- | --- | ---: |
| V2 | `shared` | `coupled` | `coupled` | `ce` | `0.0` |
| V5a | `shared` | `coupled` | `coupled` | `ce_rps` | `0.10` |

V5a는 V3a/V3b/V3c value expert와 detachment를 사용하지 않는다. 동일 weight와
input에서는 V2와 parameter, state dictionary, forward logits, inference output이
동일하고 training objective만 달라진다.

Loss definition:

```text
C = num_marks - 1                       # real marks; PAD 제외
p_c = softmax(real_logits)_c
F_k = sum_{c=0..k} p_c                  # k = 0..C-2
O_k = 1[target <= k]

RPS = mean_k (F_k - O_k)^2
marker_train_loss = nll_marker + lambda_ordinal * RPS
```

Deterministic prediction에서는 `RPS = abs(pred-target)/(C-1)`이므로 인접 error보다
먼 mark error를 더 크게 벌점화한다. RPS는 `[0,1]`로 정규화하고 첫 coefficient는
`0.10`으로 고정한다.

Metric identity:

- `nll_marker`: 기존 categorical CE만 기록
- `nll_time`: 기존 continuous-time NLL만 기록
- `nll = nll_marker + nll_time`: 기존 likelihood 의미 유지
- `ordinal_marker_loss`: CE와 별도 기록
- `marker_train_loss = nll_marker + lambda_ordinal * ordinal_marker_loss`
- quantity loss mode와 관계없이 ordinal 항은 shared loss composer에서 정확히 한 번만 가산

PAD/mask contract:

- CE는 기존 full logits를 사용해 PAD probability를 계속 억제
- RPS는 `logits[..., :pad_id]`만 softmax하고 real mark에 대해 재정규화
- CE와 RPS는 동일한 `step_mask`와 `train_loss_scope` 사용
- `C=1`이면 ordinal threshold가 없으므로 scalar zero 반환

Configuration and artifact identity:

- config/CLI: `marker_loss_mode=ce|ce_rps`
- config/CLI: `lambda_ordinal`, default `0.0`
- `ce`는 `lambda_ordinal=0`, `ce_rps` 실험은 `lambda_ordinal>0`을 요구
- 첫 V5a activation은 TitanTPP-only
- run path: `markloss_ce_rps/lambdaord_0p1`
- manifest, checkpoint, resume/cache, history, validation/test, scale/confusion,
  model-test, report grouping에 두 필드 기록
- V2 legacy path와 behavior는 변경하지 않음

Evaluation additions:

- validation/test normalized RPS
- balanced accuracy와 macro F1
- mark MAE와 adjacent accuracy (`abs(pred-true) <= 1`)
- per-class support/recall/precision/F1
- mark-0 prediction share와 mark `0/1` recall
- `adjacent_share_of_errors`는 denominator가 총 error 수에 따라 바뀌므로 monitor만 하고
  hard gate에는 mark MAE와 adjacent accuracy를 사용

Checkpoint and test-lock:

- primary checkpoint는 계속 `best_val_nll`
- lambda와 candidate 선택은 validation만 사용
- current runner가 test artifact를 생성하더라도 V5a coefficient와 multi-seed candidate가
  고정되기 전에는 읽지 않음
- held-out test audit 실패 후 같은 test 결과로 V5a lambda를 다시 조정하지 않음

Focused contract gate:

- correct deterministic RPS `0`, adjacent/distant RPS 거리 비례 확인
- RPS finite, `[0,1]`, PAD 제외 확인
- CE/RPS transition mask 및 `all|target_only` equivalence 확인
- V2 default objective/gradient/path exact regression 확인
- V2/V5a parameter, initialization, forward exact equivalence 확인
- isolated RPS가 mark head와 Titan encoder만 update하는지 확인
- 모든 quantity loss mode에서 weighted RPS가 한 번만 포함되는지 확인
- invalid config/CLI fail-fast와 artifact identity 확인

5090 integration gate:

1. local focused CPU test
2. 5090 CUDA `small_lmm` model-test
3. 5090 Instacart top-20 e1 smoke
4. finite CE/RPS/time/value/quantity/full loss, manifest/path/report 생성 확인
5. NaN, Traceback, ERROR가 없을 때만 Intermittent screening 진행

Intermittent seed-42 e50 validation reference:

| Metric | V2 |
| --- | ---: |
| Total NLL | `5.666520` |
| Marker NLL | `0.991274` |
| Mark accuracy | `57.249%` |
| Balanced accuracy | `42.664%` |
| Macro F1 | `43.302%` |
| Mark MAE | `0.487411` |
| Adjacent accuracy | `94.377%` |
| Mark-0 recall | `75.543%` |
| Mark-1 recall | `49.616%` |
| Quantity MAE | `3.060182` |

Seed-42 validation-only gate:

Ordinal benefit:

- V2 재평가 대비 normalized RPS `>= 1%` 개선
- mark MAE `>= 1%` 개선
- balanced accuracy 또는 macro F1 중 하나 `>= +0.50%p`, 다른 하나 `>= -0.25%p`

Classification safety:

- mark accuracy gap `>= -0.25%p`
- mark-1 recall gap `>= -1.00%p`
- mark-0 recall gap `>= -2.00%p`
- adjacent accuracy gap `>= -0.25%p`

Likelihood/task safety:

- marker NLL regression `<= 1%`
- total NLL regression `<= 0.5%`
- time NLL regression `<= 0.5%`
- quantity MAE/value MAE regression 각각 `<= 2%`
- validation share `>= 5%` quantity bucket regression `<= 5%`

Validation-only lambda branch:

- `0.10`에서 전체 통과: coefficient 고정 후 multi-seed
- safety 통과, ordinal benefit 실패: `0.20` 한 번만 추가 screening
- ordinal benefit 통과, safety 실패: `0.05` 한 번만 추가 screening
- benefit/safety 동시 실패 또는 추가 한 번도 실패: V5a 중단
- 이 분기 동안 held-out test 결과는 읽지 않음

Strict matched multi-seed gate:

- V2와 V5a 모두 seeds `42,52,62`, e50 동일 budget 필요
- 현재 V2 Intermittent e50은 seed 42만 있으므로 seeds 52/62 matched baseline 추가 필요
- `3/3` 완료, mean RPS/mark MAE 각각 `>= 1%` 개선
- seed-matched RPS/mark MAE 동시개선 `>= 2/3`
- mean mark accuracy gap `>= -0.25%p`, 어떤 seed도 `< -0.75%p`가 아님
- mean marker NLL regression `<= 1%`
- balanced/macro, mark `0/1`, quantity/value/time safety는 seed mean에서 유지

Frozen held-out test audit:

- multi-seed validation으로 coefficient/model을 고정한 뒤 protocol 순서로 test 확인
- mean RPS/mark MAE 각각 `>= 1%` 개선, seed-matched 개선 `>= 2/3`
- mean mark accuracy gap `>= -0.25%p`, worst seed `>= -0.75%p`
- marker NLL `<= 1%`, total/time NLL safety 통과
- quantity/value mean regression `<= 2%`, share `>= 5%` bucket `<= 5%`
- 실패하면 V2 baseline 유지; 동일 test를 이용한 V5a 재튜닝 금지

Decision:

- V5a는 Intermittent V2 기반 `CE + normalized RPS` objective-only enhancement
- 첫 구현 coefficient는 `0.10`
- class-prior correction은 V5a에 포함하지 않고 V5b 별도 ablation으로 유지
- Taxi V3b confirmed decision과 V4 time-head 트랙은 변경하지 않음
- 최종 backbone 우월성 주장 전에는 같은 CE+RPS objective를 RMTPP/THP에도 적용하거나,
  V5a를 TitanTPP system-level enhancement로 명시해야 함

Implementation status (`2026-07-12`):

- normalized RPS helper와 `marker_loss_mode=ce|ce_rps`, `lambda_ordinal` 구현 완료
- `nll_marker`와 `nll`의 기존 likelihood 의미를 유지하고
  `ordinal_marker_loss`, `marker_train_loss`를 별도 반환
- 공통 loss composer에서 `residual_only|hybrid|qty_only` 모두 weighted RPS를 한 번만 가산
- CLI, model config, manifest/checkpoint/resume/cache, V5 run path, history,
  validation/test, scale/confusion, per-class artifact, report grouping 연결 완료
- validation/test normalized RPS, balanced accuracy, macro F1, mark MAE,
  adjacent accuracy, mark-0 prediction share, mark `0/1` recall 추가 완료
- local focused tests `20 passed`; 기존 V3 계열 `18 passed`; 진단 tests `4 passed`
- CPU default model-test(RMTPP/TitanTPP/THP)와 V5a TitanTPP model-test 통과
- 5090 CUDA model-test 통과: total NLL `4.944328`, CE `2.500342`, RPS
  `0.185512`, marker train loss `2.518893`
- Instacart top-20 e1 smoke 완료: validation NLL/RPS `3.275237/0.078961`,
  test NLL/RPS `3.161041/0.077667`; runtime/artifact gate `PASS`
- e1에서 validation/test prediction은 모두 mark `3`; 미학습 smoke 상태이므로 V5a
  성능 결론이나 lambda 선택 근거로 사용하지 않음
- Intermittent seed-42 e50 validation-only screening은 `2026-07-12 21:16:02 KST`
  5090 tmux에서 시작해 `2026-07-12 21:22:38 KST`에 50 epoch 정상 완료
- e50 준비 완료: V2 `best_val_nll` checkpoint validation-only RPS reference를 먼저
  고정한 뒤 V5a를 실행하는 순서로 5090 script와 CUDA preflight 통과
- held-out test metric은 coefficient/multi-seed candidate 고정 전까지 읽지 않음
- V5a 학습 전 V2 validation reference 고정 완료: RPS `0.035283`, total/marker
  NLL `5.666520/0.991274`, mark accuracy/MAE `0.572492/0.487411`
- V5a `best_val_nll`은 epoch `30`, validation total/marker/time NLL은
  `5.655732/0.991668/4.664064`; quantity/value MAE는 `2.889382/0.130431`
- V5a RPS `0.035371`은 V2보다 `0.251%` 악화됐고 mark MAE `0.527028`은
  `8.128%` 악화되어 ordinal benefit gate 실패
- mark accuracy `-2.430%p`, balanced accuracy `-0.997%p`, macro F1
  `-2.139%p`, adjacent accuracy `-1.401%p`로 classification safety 실패
- mark-0 recall은 `+10.919%p` 개선됐지만 mark-0 prediction share가
  `45.030% -> 58.750%`로 증가했고 mark-1 recall은 `-24.953%p` 하락
- validation share `88.67%`인 `1-9` quantity bucket MAE가 `+5.291%`
  악화되어 허용치 `5%`를 소폭 초과; `10-99` bucket은 `-6.423%` 개선
- ordinal benefit과 safety가 동시에 실패했으므로 사전 규칙에 따라 lambda
  `0.05/0.20` 추가 screening 없이 V5a 중단, multi-seed로 승격하지 않음
- held-out test metric은 읽지 않았고 V2를 Intermittent baseline으로 유지

Detailed ADR:

```text
.agents/results/architecture/adr-titantpp-v5a-ordinal-marker-rps-loss.md
```

## 20. Parallel Direct Magnitude Decoder And Causal Shrinkage RevIN Contract

V5a 결과는 Intermittent mark imbalance를 ordered-distance loss만으로 해결하기
어렵다는 점을 보여줬다. 다음 우선 트랙은 V5b prior correction보다 direct
`log2(qty)` magnitude regression을 먼저 검증한다. 단, mark head를 제거하면 현재
TitanTPP의 marked-event likelihood와 RMTPP 비교 계약이 깨지므로 첫 단계에서는
marker/time probabilistic head를 유지한다.

Selected architecture:

```text
Titan encoder
  + categorical marker head       -> 기존 CE/NLL 유지
  + continuous-time head          -> 기존 time NLL 유지
  + exclusive direct magnitude head -> log2(qty)와 qty 예측
```

`parallel`은 세 task가 encoder를 공유한다는 뜻이다. Legacy mark-residual decoder와
direct magnitude decoder를 동시에 활성화하지 않는다. Run마다
`qty_decoder_mode=mark_residual|direct_log_qty` 중 하나만 quantity prediction과
quantity loss를 소유한다.

Direct target:

```text
z = mark + scale_residual = log2(demand_qty)
z_hat = center + scale * z_hat_norm
qty_hat = 2 ^ z_hat
```

Domain scope correction (`2026-07-13`): this target makes M0-M4 a
**log-domain** family. M0 uses fixed train-global statistics and is not RevIN.
The planned M2-M4 compute instance/shrinkage statistics in log2 space, so they
must be called log-domain RevIN variants rather than canonical raw-quantity RevIN.

첫 track은 `scale_base=2`, fixed split, `train_loss_scope=target_only`,
`marker_loss_mode=ce`, `lambda_ordinal=0`만 지원한다. Appended target과 padding은
normalization 통계와 encoder magnitude input에서 제외한다.

Variant contract:

| Variant | Normalization | Stat context | Role |
| --- | --- | --- | --- |
| M0 | train-global | no | direct regression baseline |
| M1 | per-series train-only, global fallback | no | fixed-series ablation |
| M2 | causal window RevIN | no | plain RevIN ablation |
| M3 | causal shrinkage RevIN | no | shrinkage effect |
| M4 | causal shrinkage RevIN | yes | primary candidate |

기존 composite M3를 M3/M4로 분리해 shrinkage와 statistic context의 효과가 섞이지
않도록 한다. M4 context는 `[center, log(scale), log1p(history_count)]`이며
magnitude head에만 전달한다.

Shrinkage는 표준편차를 직접 평균하지 않고 train-global/history first/second moment를
혼합한다.

```text
alpha = n / (n + k)
mu = alpha * mu_history + (1-alpha) * mu_global
m2 = alpha * (var_history + mu_history^2)
   + (1-alpha) * (var_global + mu_global^2)
var = max(m2 - mu^2, sigma_floor^2)
```

`k`, `sigma_floor`, exp2 clamp는 train-only history/variance audit으로 정한 뒤
validation 전에 고정한다. M2는 one-event variance가 zero가 되는 plain RevIN의
failure mode를 그대로 측정하고, M3/M4가 global shrinkage로 이를 안정화하는지 본다.

Loss contract:

```text
magnitude_loss = Huber(z_hat_norm, z_target_norm)
direct_qty_loss = Huber(qty_hat / qty_scale, qty / qty_scale)

total_loss = marker_train_loss
           + lambda_dt * nll_time
           + 1.0 * magnitude_loss
           + 0.25 * direct_qty_loss
```

`nll_marker`, `nll_time`, `nll=nll_marker+nll_time` 의미는 변경하지 않는다.
Legacy `value_loss/value_mae`를 log-quantity metric으로 재사용하지 않고
`magnitude_loss`, `log_qty_mae`, `log_qty_rmse`를 별도 export한다.

Checkpoint and gate:

- primary checkpoint는 계속 `best_val_nll`
- `best_val_qty_mae`는 magnitude under-training diagnostic이며 단독 승격 근거가 아님
- M0는 V2 대비 `best_val_nll` quantity/log-quantity MAE 각각 `>=3%` 개선 필요
- M3/M4는 M0 대비 전체 quantity/log-quantity MAE 각각 `>=2%` 개선 필요
- history count `<=4`에서 M3/M4는 M0 대비 두 metric 각각 `>=3%` 개선 필요
- M4가 M3를 개선하지 못하면 더 단순한 M3를 선택
- marker NLL `<=1%`, total/time NLL `<=0.5%`, mark accuracy `>=-0.25%p`,
  DT MAE `<=2%` regression safety 유지
- M0가 실패하면 M0에 종속된 log-domain M1-M4 branch를 중단한다. 이 gate는
  raw-quantity RevIN의 성능 결론으로 확장하지 않는다.
- M0만 통과하면 direct regression 효과로만 기록하고 RevIN 개선을 주장하지 않음
- held-out test는 multi-seed candidate와 constants 고정 전까지 잠금

Train-only audit completed on `2026-07-13` using the exact fixed-split
`RMTPPWeekLookbackDataset` context contract:

- source quality gate `PASS`; train target `136,256` and context-length
  distribution exactly match the DataLoader
- context count median/p95/max `3/11/12`; `67.63%` have at most four events
- zero-variance context `35.23%`; zero-variance train series `38.61%`
- train series with at most four events `61.06%`
- between-series level variance share `73.23%`
- target outside current history range `28.70%`
- window half-shift median/p95 `0.5000/1.4534` log2 units
- series early/late shift median/p95 `0.2925/1.3838` log2 units
- validation/test rows were not read for constant selection

Frozen train-only constants:

```text
shrinkage_k=4
magnitude_sigma_floor=0.0014535461338152059
magnitude_exp_clamp_min=-2
magnitude_exp_clamp_max=15
```

At `k=4`, target absolute normalized residual p99 is `1.7773`, one-event median
scale is `1.3953`, and normalized residual share above three is `0.0073%`.
Plain M2 RevIN is diagnostic only, M1 requires global fallback, and M3/M4 remain
the primary shrinkage candidates. This audit is normalization feasibility
evidence, not model-performance evidence.

M0 implementation status (`2026-07-13`):

- `qty_decoder_mode=mark_residual|direct_log_qty` exclusive decoder contract implemented
- shared `MagnitudeContext` supplies the same train-global center/scale to normalized
  history input, target normalization, and direct decoder denormalization
- train-global mean/population variance are recomputed from `chronological_split=train`
  only and persisted in manifest, RMTPP config/checkpoint, and summary
- appended target and padding are excluded from normalized magnitude input
- direct quantity prediction bypasses predicted mark and legacy residual/value head
- marker CE/NLL and RMTPP time NLL remain unchanged; M0 exports separate
  `magnitude_loss`, `log_qty_mae`, and `log_qty_rmse`
- CLI rejects mixed-model, non-fixed-split, non-target-only, non-log2, V3/V5, and
  contextual-TTM combinations for the first M0 activation
- run/cache/report grouping includes decoder and normalization identity, preventing
  M0 artifacts from merging with legacy mark-residual runs
- local synthetic M0 model-test passed; focused and existing enhancement regression
  suite passed (`58 passed`)
- 5090 CUDA model-test passed on RTX 5090: hidden shape `[4,16,64]`, total/marker/time
  NLL `3.902806/2.495406/1.407400`, magnitude loss `0.212801`, all outputs finite
- first CUDA attempt exposed a server `libnvrtc-builtins.so.13.0` lookup issue;
  adding ai_env `nvidia/cu13/lib` to `LD_LIBRARY_PATH` resolved it without model changes
- 5090 Instacart top-20 e1 fixed-split smoke passed: train/validation/test samples
  `1,380/300/300`; train-only magnitude event count/mean/std
  `1,400/3.611660/0.733396`
- e1 validation NLL/quantity MAE/log2 MAE/magnitude loss are
  `3.252680/5.215548/0.567363/0.398124`; held-out metrics were read only for
  artifact integrity, not candidate selection
- direct checkpoint contains `magnitude_head` and `magnitude_input_proj` but no
  legacy `value_head`; runtime/artifact integration gate passed
- Intermittent seed-42 e50 completed on 5090 at `2026-07-13 11:34:38 KST` with
  `SCREENING_SUCCESS`; the primary `best_val_nll` checkpoint is epoch `24`
- frozen V2 validation-only reference records `held_out_test_read=false`, checkpoint
  epoch `19`, and `41,901` targets
- M0 versus V2 validation total/marker/time NLL changed by
  `-1.631%/+0.872%/-2.162%`; NLL safety passed and the total gain was led by time NLL
- raw quantity MAE improved `9.791%`, but log2 quantity MAE regressed `9.700%` and
  mark accuracy fell `3.635%p`; the simultaneous quantity and marker gate failed
- the dominant `1-9` bucket (`88.666%` share) regressed `8.623%` in quantity MAE and
  `12.155%` in log absolute error, while `100-999` and `1000-9999` quantity MAE
  improved `13.737%` and `41.977%`
- the `1-9` result also fails the predeclared rule that no quantity bucket with
  share at least `5%` may regress by more than `5%`
- M0 predicted mark-0 share increased `45.030% -> 63.368%`; mark-0 recall rose
  `14.500%p`, but mark-1 recall fell `31.383%p`
- M0 remains a log-domain negative ablation: no matched multi-seed promotion and
  no activation of the existing log-domain M1-M4 branch
- validation-only audit caveat: the M0 runner co-locates test columns in
  `leaderboard/runs.csv`, and those columns were exposed during schema inspection
  but were not used in the predeclared validation gate; future blind gates must
  avoid merged run/report files until the decision is recorded

Post-result domain reclassification (`2026-07-13`):

- M0 is `log2(qty)` direct regression with train-global normalization; it never
  computes per-window instance statistics and is not a RevIN experiment
- TitanTPP constructs `MemoryEncoder` directly, so the standalone Titan wrapper's
  `use_revin` path is not active in the current TitanTPP execution path
- the M0 failure supports rejecting the completed log-domain direct/global setting
  and stopping its dependent log-domain M1-M4 candidates under the original gate
- it does not support the claim that raw-quantity RevIN is ineffective
- the raw Q0 global, Q1 causal masked RevIN, and Q2 causal shrinkage path passed
  Instacart integration and has now been evaluated on Intermittent seed 42
- matched Intermittent seed-42 e50 validation-only screening completed on 5090 at
  `2026-07-13 17:37:12 KST`; Q0/Q1/Q2 all completed `50/50` epochs with exit code 0
- Q0/Q1/Q2 improved overall raw quantity MAE by `7.835/13.740/14.827%`, but all
  failed the predeclared candidate gate; held-out test remains locked
- V5b remains a separate fallback rather than the only valid next model track

Next execution order:

1. 5090 CUDA M0 model-test - completed
2. 5090 Instacart top-20 e1 fixed-split smoke and artifact contract check - completed
3. Intermittent M0 seed-42 e50 validation-only screening - completed, gate failed
4. Reclassify M0 as a log-domain negative ablation and close only log-domain M1-M4 - completed
5. Define raw-quantity Q0/Q1/Q2 RevIN contract and acceptance gate - completed
6. Run the Intermittent train-only raw history/variance/tail audit and freeze Q2 constants - completed, `k=8`
7. Implement the matched Q0/Q1/Q2 `direct_raw_qty` contract - completed
8. Run focused equivalence/leakage tests and the 5090 CUDA Q0/Q1/Q2 model-test - completed
9. Run an Instacart top-20 e1 fixed-split smoke before Intermittent e50 screening - completed
10. Run matched Q0/Q1/Q2 Intermittent seed-42 e50 validation-only screening - completed, all candidates failed gate
11. Keep held-out test locked and do not start Q0/Q1/Q2 matched multi-seed - completed
12. Design a detached and low-quantity-protected direct-raw follow-up before any new screening
13. Keep V5b class-prior correction as a separate fallback branch

Audit artifacts:

```text
search_artifacts/model_enhancement_magnitude_revin_audit_0713
search_artifacts/model_enhancement_raw_quantity_revin_audit_0713
```

Detailed ADR:

```text
.agents/results/architecture/adr-titantpp-parallel-magnitude-shrinkage-revin.md
```

## 21. Raw-Quantity Q0/Q1/Q2 RevIN Contract

The raw-domain branch is separate from the failed log-domain M0-M4 design. The
shared architecture keeps TitanTPP marker/time heads and introduces the exclusive
`direct_raw_qty` decoder. All candidates consume normalized raw quantity history,
predict a normalized raw target, and denormalize directly into quantity space.

Variant lock:

| Variant | Normalization | Role |
| --- | --- | --- |
| Q0 | fixed train-global raw moments | raw-domain control; not RevIN |
| Q1 | causal masked history mean/std | canonical masked RevIN diagnostic |
| Q2 | causal history/global moment shrinkage | primary short-context candidate |

Q0 is not a prerequisite. Q0, Q1, and Q2 enter the same seed-42 e50 screening
after train-only audit and focused implementation gates. Q1 failure does not veto
Q2 because one-event and zero-variance contexts are the failure mode Q2 is
designed to stabilize.

Shared model constraints:

- fixed split, `scale_base=2`, `train_loss_scope=target_only`
- plain CE marker objective and unchanged RMTPP time likelihood
- no V3/V5 expert, ordinal, detached-gradient, contextual-TTM, statistic-context,
  or learnable RevIN affine in the first comparison
- identical parameters and initialization across Q0/Q1/Q2
- appended target and padding excluded from all statistics and magnitude input
- stateless masked normalization in TitanTPP, not the standalone Titan wrapper

Raw loss:

```text
raw_norm_loss = Huber(u_hat, u_target)
raw_qty_loss = Huber(q_affine, q_target)

total_loss = marker_ce
           + 1.0 * time_nll
           + 1.0 * raw_norm_loss
           + 0.25 * raw_qty_loss
```

No log transform enters training. Evaluation keeps raw quantity MAE as primary
and exports log2 quantity MAE only as a low-scale balance guardrail. The primary
checkpoint remains `best_val_nll`; best-quantity checkpoints are diagnostic only.

Before implementation, an exact fixed-split train-only raw audit must freeze Q2
constants. It evaluates `k={1,2,4,8,16}` and uses
`sigma_floor_raw=max(0.001*global_raw_std,1e-4)`. Eligible candidates must remain
finite, retain one-event scale, keep median local weight, and not worsen Q0's
normalized-target tail. Validation/test rows are not used.

Train-only raw audit outcome (`2026-07-13`, 5090):

- source and exact `RMTPPWeekLookbackDataset` target/context-distribution gates
  passed; `159,643` train events, `23,387` series, and `136,256` train targets
- raw quantity mean/median/p95/p99/max are
  `6.8459/2/17/65/5000`; top `1%` accounts for `42.59%` of total quantity
- between-series raw variance share is `78.10%`, confirming substantial level
  heterogeneity
- context median/p95/max is `3/11/12`; one-event, `n<=4`, and zero-variance
  context shares are `22.66%`, `67.63%`, and `35.23%`
- Q1 plain masked RevIN is diagnostic only: scale p01 is `0.003162`, target
  absolute normalized p99 is `2846.0499`, and `|u|>3` share is `22.0636%`
- Q0 target absolute normalized p99 and `|u|>3` share are `1.1844` and `0.4073%`
- Q2 `k=8` passed every train-only gate and was selected with p99 `0.7968`,
  `|u|>3` share `0.0514%`, median `alpha=0.2727`, and one-event median scale at
  `94.32%` of global raw std
- frozen Q2 constants are `shrinkage_k=8`,
  `sigma_floor_raw=0.0550124034288891`, `global_mean_raw=6.8458560663480394`,
  `global_var_raw=3026.3645310228494`, and
  `global_std_raw=55.0124034288891`
- validation and held-out test rows were not read; this is normalization
  feasibility evidence, not predictive-performance or RevIN-benefit evidence

Seed-42 candidate eligibility versus V2 requires:

- overall and history-count-`<=4` raw quantity MAE each improve `>=3%`
- log2 quantity MAE regression `<=2%`
- no share-`>=5%` quantity bucket regresses more than `5%`
- marker NLL `<=1%`, total/time NLL `<=0.5%`, mark accuracy `>=-0.25%p`,
  and DT MAE `<=2%` regression safety
- all values finite and pre-clamp negative prediction share `<=1%`

A RevIN claim additionally requires Q1/Q2 to beat Q0 by `>=2%` overall raw MAE
and `>=3%` for history count `<=4`, with log2 MAE regression `<=1%`. If Q1 and
Q2 both pass, Q2 is selected only when it adds at least `1%` overall or
short-context benefit; otherwise the simpler Q1 wins. Quantity benefit with
marker-safety failure is not promoted and may motivate a separate Q2b detached
gradient-routing design.

After seed-42 selection, strict matched V2/Q0/candidate seeds `42,52,62` e50 are
required before held-out test unlock. A failed frozen test audit returns to V2
without test-driven retuning.

Implementation status (`2026-07-13`):

- architecture, loss, normalization, audit, focused-test, validation, multi-seed,
  and held-out acceptance contracts are complete
- raw-domain audit and Q2 constants freeze are complete; local audit formula tests
  passed (`5 passed`) and the 5090 runtime audit completed without non-finite values
- matched `direct_raw_qty` Q0/Q1/Q2 is implemented with stateless raw global,
  causal masked, and causal moment-shrinkage contexts
- raw normalized Huber and unclamped affine raw-quantity Huber are wired without
  changing marker/time likelihood identity; non-negative clamp is evaluation-only
- train-only raw moments, effective floor, full run/manifest/checkpoint/history/
  cache/resume identity, raw/context diagnostics, and CLI paths are connected
- dedicated raw contract tests passed `22/22`; the complete search suite passed
  `85/85`; local CPU model-tests for Q0/Q1/Q2 completed with finite outputs
- 5090 CUDA Q0/Q1/Q2 model-test passed on RTX 5090: all variants returned
  `status=success`, finite outputs, hidden shape `[4,16,64]`, and identical
  parameter count `78,111`; normalization mode was the only model-config change
- the 5090 Instacart top-20 e1 actual-data integration smoke passed for all three
  variants with the same fixed split, train/validation/test sample counts
  `1380/300/300`, and train-only raw moments; checkpoint, history, summary,
  test-summary, scale-wise, report, and plot artifacts were generated
- Q1 remained finite but reproduced the expected scale-collapse diagnostic:
  test scale p01 reached `sqrt(1e-5)=0.003162`, normalized-target absolute p99
  reached `1268.0743`, and epoch train loss was `98.8030`; Q2 kept validation/test
  normalized-target p99 at `2.7290/2.4759` with epoch train loss `4.9314`
- the e1 smoke is integration evidence only: Q2 improved the populated `10-99`
  quantity bucket but regressed the `1-9` bucket, the top-20 subset contained no
  `100+` targets, and no accuracy or RevIN-benefit conclusion is available before
  the in-progress Intermittent seed-42 e50 validation-only gate is completed
- the matched screening started at `2026-07-13 17:14:05 KST`; the frozen V2
  validation reference completed with `41,901` samples and all three candidates
  finished at `17:37:12 KST` without runtime failure
- Q0/Q1/Q2 overall raw quantity MAE improved by `7.835/13.740/14.827%`, and
  history-count `<=4` MAE improved by `3.133/15.265/14.838%`
- Q0 failed log2, dominant `1-9`, and mark-accuracy gates; Q1 failed marker/total
  NLL, mark accuracy, and DT gates; Q2 failed log2, dominant `1-9`, and
  mark-accuracy gates
- Q1 reproduced scale collapse (`scale p01=0.003162`, normalized-target p99
  `1897.3666`, magnitude loss `183.5540`); Q2 stabilized those diagnostics but
  did not satisfy common V2 eligibility
- no candidate advances to multi-seed and held-out test remains locked; V2 stays
  the Intermittent baseline

Detailed ADR:

```text
.agents/results/architecture/adr-titantpp-raw-quantity-revin-q0-q1-q2.md
```

## 22. Q3 Factorial Gradient Routing And Dual-Domain Loss Contract

Q2 is retained only as a normalization foundation. Its seed-42 result improved
overall and history-count-`<=4` raw MAE by `14.827%/14.838%` versus V2, but
failed log2 MAE, dominant `1-9`, and mark-accuracy safety. Q3 separates the two
remaining hypotheses with a no-new-parameter `2 x 2` factorial design:

| Variant | Magnitude encoder gradient | Log2 auxiliary | Role |
| --- | --- | --- | --- |
| Q2 control | coupled | off | fresh matched control |
| Q3a | detached | off | gradient-isolation effect |
| Q3b | coupled | on | low-quantity-loss effect |
| Q3c | detached | on | combined interaction |

All variants keep `direct_raw_qty`, Q2 `causal_shrinkage_revin`, `k=8`, frozen
raw moments/floor, small LMM, plain CE, target-only loss, fixed split, and the
same parameter tensors and forward outputs. The new direct-magnitude route is
separate from V3's `value_encoder_gradient_mode`:

```text
magnitude_encoder_gradient_mode = coupled | detached
h_mag = h_j                                      # coupled
h_mag = stop_gradient(h_j)                       # detached
u_hat = magnitude_head(h_mag)
```

In detached mode, magnitude losses train only the magnitude head. Marker/time
NLL still trains the encoder and magnitude input projection, so observed raw
quantity remains available to the TPP representation.

Q3b/Q3c add an auxiliary error term without changing raw normalization or
denormalization:

```text
L_norm = Huber(u_hat, u_target)
L_raw  = Huber(q_affine, q_target)
L_log  = Huber(log2(max(q_affine, 1)), log2(max(q_target, 1)))

L_total = marker_ce + time_nll
        + 1.00 * L_norm
        + 0.25 * L_raw
        + 0.25 * L_log
```

Q2/Q3a use `L_log=0`. This remains raw-domain RevIN because log2 appears only
in an auxiliary training error. The raw losses remain unclamped and continue to
recover negative or sub-one affine predictions.

Implementation gate:

- exact parameter/init/forward equivalence across Q2/Q3a/Q3b/Q3c
- Q2/Q3a and Q3b/Q3c scalar-loss equivalence
- detached magnitude losses have zero encoder/input-projection/marker/time-head
  gradients while the magnitude head remains trainable
- marker/time NLL routes remain unchanged
- masked log-loss formula, negative-affine raw gradient, target/padding isolation
- full config/path/manifest/checkpoint/cache/resume/history/summary identity
- local focused/full tests, 5090 CUDA model-test, Instacart top-20 e1 smoke

Seed-42 Intermittent screening reruns Q2 with all three Q3 variants. Fresh Q2
must reproduce frozen Q2 within `1%` for NLL/raw/log metrics and `0.25%p` for
mark accuracy before attribution.

Full candidate gate at `best_val_nll`:

- overall raw MAE `<=2.736781`
- history-count-`<=4` raw MAE `<=2.053191`
- log2 MAE `<=0.600517`
- `1-9` raw MAE `<=0.999348`; other share-`>=5%` bucket regression `<=5%`
- marker/total/time NLL `<=1.001186/5.694853/4.698623`
- mark accuracy `>=56.999%`, DT MAE `<=42.905873`
- predicted mark-0 absolute share error `<=5.850%p`
- mark-1 recall `>=44.616%`
- all values finite and pre-clamp negative share `<=1%`

Q3a/Q3b mechanism results do not early-stop Q3c. Select the simplest full-gate
candidate, preferring a single intervention over Q3c. If none passes, retain V2.
Only a frozen selected candidate advances to strict matched V2/Q2/candidate
seeds `42,52,62`; held-out test remains locked until that gate passes.

Implementation status (`2026-07-15`):

- architecture, loss, gradient, artifact, focused-test, seed-42, multi-seed, and
  held-out contracts are complete
- independent magnitude-to-encoder routing and raw-domain log2 Huber auxiliary
  are implemented without new parameters or state-dict changes
- config, CLI, model construction, training/evaluation loss, path, manifest,
  checkpoint, cache/resume, history, summary, and scale-wise identity are wired
- focused Q3 tests passed `19/19`; the complete search suite passed `104/104`
- local CPU Q2/Q3a/Q3b/Q3c model-tests all passed with identical parameter count
  `78,111`, NLL, magnitude loss, and quantity predictions; Q3b/Q3c alone report
  the active log auxiliary
- preparation commit `f4cc223` was checksum-verified on 5090 and the RTX 5090 /
  PyTorch `2.11.0+cu130` CUDA preflight passed
- tmux `titantpp_q3_cuda_0713` started at `2026-07-13 23:04:19 KST`; the one-time
  initial check observed Q2/Q3a/Q3b success and Q3c model-test entry
- the run completed at `2026-07-13 23:04:26 KST` with success sentinel, aggregate
  exit code `0`, and all four variant exit codes equal to `0`
- all 13 artifact files are synced locally; parameter/shape/config identity,
  paired scalar-loss equality, finite values, and JSON/CSV consistency passed
- Q2/Q3a are exact no-aux scalar matches and Q3b/Q3c are exact positive-log-aux
  scalar matches; total-loss recomputation error is at most `1.58e-6`
- the 5090 CUDA runtime and artifact identity gate passed; this is not an
  actual-data performance result
- the matched Instacart top-20 e1 runner and start record are prepared with the
  prior `1380/300/300` fixed-split sample contract, identical Q2 normalization
  and training budget, and only the two Q3 factorial axes varying
- the runner records independent variant status, root/variant manifests and
  logs, cache-safe paths, and success/failure sentinels; the gate is explicitly
  limited to actual-data integration rather than performance ranking
- preparation revision `d552b77` and its four file checksums are verified on the
  non-Git 5090 working copy and preserved in `source_sync_manifest.json`
- RTX 5090 / PyTorch `2.11.0+cu130` CUDA allocation, fixed-split files, top-20
  quantity contract, exact loader samples `1380/300/300`, and all four CLI
  contracts passed preflight
- tmux `titantpp_q3_insta_e1_0714` started at `2026-07-14 08:45:33 KST`; the
  one-time check observed Q2 epoch 1 completion and an active CUDA process
- the requested one-time completion check confirmed root `SMOKE_SUCCESS`, no
  failure sentinel, aggregate exit code `0`, and Q2/Q3a/Q3b/Q3c exit codes `0`;
  the run ended at `2026-07-14 08:45:53 KST`
- all `388` artifact files (`18M`) are synced locally and a checksum dry-run found
  no remote/local differences; root metadata and each variant's manifest,
  summary, test summary, history, validation/test scale-wise metrics, report,
  plots, and best-validation-NLL checkpoint satisfy the availability contract
- protocol-order analysis confirmed finite runtime, loss, prediction, summary,
  history, checkpoint/resume, scale-wise, report, and plot contracts; Q2/Q3a
  keep zero log auxiliary while Q3b/Q3c record positive finite auxiliary values
- the four checkpoints share 40 tensor keys/shapes and actual parameter count
  `77,626`, with direct magnitude modules and no legacy value head; all tensors
  are finite and e1 selection/resume states reconcile exactly
- root `expected_parameter_count=78,111` was a non-blocking manifest metadata
  defect inherited from the synthetic `num_marks=12` gate; actual Instacart
  `num_marks=7` explains the exact 485-parameter difference, and the runner now
  records actual and synthetic-reference counts separately without rewriting the
  immutable artifact
- requested sigma floor `0.0550124034288891` and actual train-derived effective
  floor `0.0067776913473542024` are correctly separated and identical across
  variants
- validation/test scale counts reconcile to 300 targets, weighted scale metrics
  reconcile within `1.03e-7`, and expected N/A cells are limited to the legacy
  direct-head value metric and empty 100+ buckets
- the Instacart actual-data integration gate passed; e1 validation differences
  and held-out test exports were not used for performance ranking or selection
- the Intermittent seed-42 runner now fixes fresh Q2/Q3a/Q3b/Q3c at e50,
  seed 42, batch 128, lookback 52, and max sequence 16; the V2 checkpoint,
  all five fixed-split source files, and frozen Q2 summary SHA are verified
  before training
- the runner writes unrounded Q2 reproduction, full candidate, mechanism,
  selection, and held-out rules to `acceptance_contract.json`; Q3c cannot be
  skipped based on Q3a/Q3b outcomes
- preparation `a0a65e5`, recovery `f5851ff`, source manifests, and all runtime
  dependencies are checksum-verified on 5090;
  CUDA/data/reference/CLI preflight passed
- the first launch stopped before training on a legacy V2 inactive-metric
  evaluator contract; the fix preserves active finite checks and passed focused
  `25/25` plus full search `110/110`
- the second launch started at `2026-07-15 08:15:07 KST`; V2 validation-only
  reference and fresh Q2 epoch 1 passed on the frozen split
- the four-variant Intermittent run completed at `2026-07-15 08:46:16 KST`;
  root `SCREENING_SUCCESS` exists, the failure sentinel is absent, and fresh
  Q2/Q3a/Q3b/Q3c all exited with code zero
- all `562` artifact files (`27,179,501` bytes) are synced locally and a checksum
  dry-run found no remote/local differences; manifest/log/status identity is
  verified without reading held-out result files
- formal seed-42 acceptance is complete with no promoted Q3 candidate;
  multi-seed and held-out Q3 experiments have not started
- protocol-order validation analysis is complete: all allowed artifacts
  reconcile to `41,901` samples, all active metrics are finite, and no held-out
  output was read
- fresh Q2 does not reproduce frozen Q2 on raw MAE (`+6.00%`), log2 MAE
  (`+8.21%`), or mark accuracy (`+1.172%p`), despite total NLL staying within
  `1%`; causal factorial attribution is therefore not yet reliable
- Q3b has the best total/time NLL and overall/short raw MAE, but its NLL gain is
  time-driven, dominant `1-9` MAE and log2 MAE do not improve, and mark-0
  overprediction worsens
- Q3c supplies a strong non-additive marker-balance interaction, reducing the
  Q2 mark-0 share error by `9.592%p` and raising mark-1 recall by `17.722%p`,
  but raw/short/log2 quantity errors worsen materially
- no Q3 variant is a balanced winner; the frozen full gate rejects Q3a/Q3b/Q3c
  and retains V2 without unlocking multi-seed or held-out evaluation
- a full deterministic e50 Q3 rerun is not required for the current
  non-promotion decision because Q3b still fails log2, dominant `1-9`, mark
  accuracy, mark-0 share, and mark-1 recall protections, while Q3a/Q3c fail
  additional quantity protections
- fresh/frozen Q2 drift still blocks causal attribution; the historical Q3
  screening runner lacked strict CUDA, dedicated shuffle-generator, and
  grouped-order contracts
- explicit `standard|strict` execution is now implemented: strict mode validates
  launcher environment, enables deterministic Torch/cuDNN, isolates and
  checkpoints loader RNG, sorts grouped series, records source/data/runtime
  identity, and writes canonical selection-state digests
- focused reproducibility tests passed `8/8`, Q3 plus reproducibility tests
  passed `27/27`, and the complete local search suite passed `118/118`
- the Q3 reopen infrastructure gate uses two independent 5090 Intermittent Q2
  e3 strict runs with exact histories, selected epochs, and canonical
  tensor-state digests
- the strict e3 runner now isolates Run A/Run B into fresh sequential Python
  processes and separate artifact roots; it requires a matching source-sync
  manifest plus CUDA and fixed-split hash preflight before training
- the exact comparator validates strict runtime/config/data identity, compares
  history JSON bytes and selected epochs, and recomputes all three checkpoint
  state digests without reading held-out metrics; focused tests passed `3/3`
- the concise Notion source draft is maintained under the `5. Model Design
  Enhancement` structure and now records the completed gate result
- revision `f6da9af9193f6f5bcd6dd60a711b9e8921593829` is checksum-synced from
  the verified 5090 baseline `f5851ff`; all 16 changed files match and the
  checksum dry-run returned zero changes
- source-manifest identity, five fixed-split hashes, runner mode, RTX 5090 CUDA
  allocation, and strict deterministic runtime preflight passed
- tmux `titantpp_q2_strict_e3_0715` started at `2026-07-15 22:53:50 KST`;
  the one-time initial check observed Run A fixed-split preparation and an
  active CUDA process, and no continuous polling is active
- Run A, Run B, and the exact comparator completed with exit code `0` at
  `2026-07-15 22:56:00 KST`; no NaN, traceback, or runtime error was found
- all `202` artifact files (`13,653,382` bytes) are synced locally and the
  checksum dry-run found no remote/local differences
- the exact report and an independent local rerun both passed all `22/22`
  checks with zero mismatches; history bytes, selected epochs, and all three
  canonical checkpoint-state digests match exactly
- the strict Q2 reproducibility gate was formally closed as `PASS` on
  `2026-07-16`; no active experiment or monitoring remains for this probe
- strict reproducibility infrastructure is accepted, but this does not promote
  Q2, establish equality with historical nondeterministic Q2, or change the V2
  retention and Q3 non-promotion decision
- only an explicitly reopened Q3 track may proceed from a passing e3 probe to a
  newly matched deterministic V2/Q2/Q3a/Q3b/Q3c e50 comparison; the historical
  nondeterministic Q2 artifact is context, not its exact numeric target

Next execution order:

1. Treat the strict Q2 reproducibility gate as closed and keep Q3 closed.
2. Move to the next model hypothesis by default.
3. Only if Q3 is explicitly reopened, prepare a fresh deterministic
   V2/Q2/Q3a/Q3b/Q3c e50 comparison using the accepted strict infrastructure.
4. Keep multi-seed and held-out execution locked until a new validation gate
   explicitly unlocks them.

Detailed ADR:

```text
.agents/results/architecture/adr-titantpp-q3-factorial-gradient-dual-domain.md
```
