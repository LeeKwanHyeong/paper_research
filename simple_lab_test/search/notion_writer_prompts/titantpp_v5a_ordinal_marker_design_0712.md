다음 TitanTPP V5a ordinal marker loss 설계와 acceptance gate를 Notion에 정리해주세요.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- Model Enhancement 설계이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.
- 같은 제목의 페이지가 있으면 업데이트하고, 없을 때만 새 페이지를 만듭니다.
- 아래 페이지와 상호 링크합니다.
  - `Intermittent Mark Imbalance And Confusion Diagnostic: V2-V3c`
  - `TitanTPP V3c Detached Value-Encoder Route Smoke And Intermittent Screening e50`
  - `TitanTPP Model Enhancement Strategy`

## 페이지 제목

- `TitanTPP V5a Ordinal Marker Loss Contract And Acceptance Gate`

## 작성 원칙

- 상태는 `design confirmed / implementation not started`로 기록합니다.
- 실제 구현이나 실험을 완료했다고 쓰지 않습니다.
- 사람이 구현 전에 남기는 모델 설계 노트처럼 간결하게 작성합니다.
- confirmed evidence, selected design, acceptance rule, remaining risk를 구분합니다.
- class imbalance가 V3 regression의 단독 원인이라고 쓰지 않습니다.
- `획기적인`, `강력한`, `유의미한`, `주목할 만한`, `명확히 입증`,
  `종합적으로` 같은 홍보성 표현을 사용하지 않습니다.
- 아래 contract와 threshold를 임의로 바꾸지 않습니다.

## 설계 배경

Intermittent fixed-split diagnostic에서 확인한 내용:

- held-out test target의 marks `0-2` share: `87.39%`
- V3c `1 -> 0` confusion: `56.16%`; V2는 `40.11%`
- V3c mark-1 accuracy contribution vs V2: `-5.316%p`
- V3c mark-0 contribution vs V2: `+3.916%p`
- adjacent share of errors: V2 `86.66%`, V3c `83.11%`
- 모든 variant는 같은 fixed-split target을 사용하고 split drift는 작음

해석:

- imbalance는 경계를 민감하게 만드는 조건이지만 variant 차이의 단독 원인은 아닙니다.
- 주요 regression은 support가 큰 mark `0/1` 경계에서 발생했습니다.
- error 대부분이 인접 class에 있어 순서를 반영하는 objective를 먼저 검증합니다.
- support `16`인 mark `10`이 있으므로 raw inverse-frequency weighting은 첫 변경에서 제외합니다.

## V5a Variant Contract

V5a는 실패한 V3c가 아니라 Intermittent의 확정 baseline V2에서 분기합니다.

| Variant | `value_head_mode` | `qty_mark_gradient_mode` | `value_encoder_gradient_mode` | `marker_loss_mode` | `lambda_ordinal` |
| --- | --- | --- | --- | --- | ---: |
| V2 | `shared` | `coupled` | `coupled` | `ce` | `0.0` |
| V5a | `shared` | `coupled` | `coupled` | `ce_rps` | `0.10` |

- V3 mark-conditioned experts를 사용하지 않음
- quantity gate/value encoder detachment를 사용하지 않음
- parameter, initialization, forward logits, inference는 V2와 동일
- training objective만 변경

## Selected Loss

Normalized Ranked Probability Score(RPS)를 기존 CE에 보조항으로 추가합니다.

```text
C = num_marks - 1                       # real marks; PAD 제외
p_c = softmax(real_logits)_c
F_k = sum_{c=0..k} p_c                  # k = 0..C-2
O_k = 1[target <= k]

RPS = mean_k (F_k - O_k)^2
marker_train_loss = nll_marker + lambda_ordinal * RPS
```

Deterministic prediction에서는 다음 관계를 갖습니다.

```text
RPS(one_hot(pred), target) = abs(pred - target) / (C - 1)
```

따라서 adjacent error보다 먼 mark error가 더 큰 비용을 받습니다. RPS는 `[0,1]`로
정규화하고 첫 coefficient는 `0.10`으로 고정합니다.

## Metric Identity

- `nll_marker`: 기존 categorical CE만 유지
- `nll_time`: 기존 continuous-time NLL 유지
- `nll = nll_marker + nll_time`: 기존 likelihood 및 checkpoint 의미 유지
- `ordinal_marker_loss`: 별도 diagnostic과 auxiliary loss로 기록
- primary checkpoint: 계속 `best_val_nll`
- RPS는 checkpoint selection metric으로 사용하지 않음

Quantity loss mode별 objective에는 아래처럼 ordinal 항을 정확히 한 번만 추가합니다.

```text
residual_only:
  CE + lambda_ordinal * RPS + lambda_value * value_loss + lambda_dt * time_NLL

hybrid:
  CE + lambda_ordinal * RPS + lambda_value * value_loss
     + lambda_dt * time_NLL + lambda_qty * qty_loss

qty_only:
  CE + lambda_ordinal * RPS + lambda_dt * time_NLL + lambda_qty * qty_loss
```

## PAD And Mask

- CE는 기존 full logits를 사용해 PAD probability를 억제
- RPS는 `logits[..., :pad_id]`만 사용하고 real mark에서 softmax 재정규화
- CE와 RPS는 동일한 transition `step_mask` 사용
- `all`과 `target_only` scope를 동일하게 적용
- real class가 하나뿐이면 RPS는 scalar zero

## Configuration And Artifact Identity

- `marker_loss_mode=ce|ce_rps`, default `ce`
- `lambda_ordinal`, default `0.0`
- `ce`는 `lambda_ordinal=0`, `ce_rps` 실험은 `lambda_ordinal>0`
- 첫 activation은 TitanTPP-only
- path: `markloss_ce_rps/lambdaord_0p1`
- manifest, checkpoint, resume/cache, history, validation/test, scale/confusion,
  model-test, report에 두 필드 기록
- legacy V2 path와 behavior 유지

## Evaluation Metrics

- normalized RPS
- mark accuracy
- balanced accuracy
- macro F1
- mark MAE
- adjacent accuracy: `abs(pred-true) <= 1`
- per-class support/recall/precision/F1
- mark-0 prediction share
- mark `0/1` recall
- marker/time/total NLL
- quantity/value/scale-wise safety

`adjacent_share_of_errors`는 total error 수에 따라 denominator가 바뀌므로 monitor로만
사용하고 hard gate는 mark MAE와 adjacent accuracy를 사용합니다.

## Focused Contract Gate

- correct deterministic prediction RPS `0`
- adjacent/distant deterministic error가 거리 비례
- RPS finite 및 `[0,1]`
- PAD logit 변화는 RPS에는 영향이 없고 CE에는 영향이 있음
- CE/RPS mask 및 `all|target_only` scope 일치
- default CE objective/gradient/path exact regression
- V2/V5a parameter, initialization, forward exact equivalence
- isolated RPS는 mark head와 Titan encoder만 update
- 모든 quantity loss mode에서 RPS가 한 번만 가산
- invalid CLI/config fail-fast

## 5090 Integration Gate

1. local focused CPU tests
2. 5090 CUDA `small_lmm` model-test
3. 5090 Instacart top-20 e1 smoke
4. CE/RPS/time/value/quantity/full loss finite 확인
5. manifest/path/report 생성 및 NaN, Traceback, ERROR 부재 확인

모든 실험은 별도 지시가 있기 전까지 5090에서만 실행합니다. 초기 config, GPU,
첫 학습 진입까지만 확인하고 지속 polling하지 않습니다.

## Seed-42 Validation Reference

| Metric | V2 value |
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

## Seed-42 Validation-Only Gate

Ordinal benefit:

- normalized RPS `>= 1%` 개선
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
- quantity/value MAE regression 각각 `<= 2%`
- validation share `>= 5%` quantity bucket regression `<= 5%`

## Lambda Branch

Validation만 사용합니다.

- `0.10` 전체 통과: coefficient 고정 후 multi-seed
- safety 통과, ordinal benefit 실패: `0.20` 한 번만 screening
- ordinal benefit 통과, safety 실패: `0.05` 한 번만 screening
- benefit/safety 동시 실패 또는 추가 한 번 실패: V5a 중단
- 이 과정에서 held-out test 결과는 읽지 않음

## Strict Multi-Seed And Test Gate

- V2/V5a 모두 seeds `42,52,62`, e50 strict matched budget
- 현재 V2 e50은 seed 42만 있으므로 seeds 52/62 baseline 추가 필요
- `3/3` 완료
- validation mean RPS/mark MAE 각각 `>= 1%` 개선
- seed-matched RPS/mark MAE 개선 `>= 2/3`
- mean mark accuracy gap `>= -0.25%p`, worst seed `>= -0.75%p`
- mean marker NLL regression `<= 1%`
- balanced/macro, mark `0/1`, quantity/value/time safety 유지
- validation으로 coefficient/model을 고정한 뒤 held-out test artifact 확인
- test 실패 시 V2 유지하고 동일 test로 V5a lambda를 다시 조정하지 않음

## Non-Goals And Risks

- class-balanced CE, focal loss, logit adjustment, label smoothing은 V5a에 포함하지 않음
- CORAL 또는 새 marker head를 추가하지 않음
- V3 value experts, time head, Titan memory, lookback, split을 바꾸지 않음
- RPS가 class prior를 직접 보정하거나 rare-class recall을 보장하지 않음
- final backbone superiority claim 전에는 같은 objective를 RMTPP/THP에 적용하거나
  V5a를 TitanTPP system-level enhancement로 범위를 제한해야 함

## Local Design Evidence

```text
/Users/igwanhyeong/PycharmProjects/paper_research/.agents/results/architecture/adr-titantpp-v5a-ordinal-marker-rps-loss.md
/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/model_enhancement_strategy.md
/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_inter_mark_diagnostics_0712/report.md
```

마지막에는 다음 작업을 남깁니다.

```text
Next: marker_loss_mode 및 normalized RPS 구현, focused loss/gradient test 작성
```
