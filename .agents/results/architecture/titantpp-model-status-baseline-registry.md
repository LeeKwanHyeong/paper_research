# TitanTPP Model Status And Baseline Registry

- Date: 2026-07-17
- Scope: Model Enhancement Session
- Canonical role: 현재 모델 승격 상태와 후속 비교 기준의 단일 기준 문서

이 문서의 `승격`은 구현 완료나 smoke 통과가 아니라, 사전에 고정한 성능 gate를
통과해 후속 실험의 incumbent로 사용할 수 있다는 뜻이다. 현재 lock은 다음 구조
실험을 위한 기준이며 논문 최종 모델 확정을 의미하지 않는다.

## 1. Status Semantics

| Status | Meaning |
| --- | --- |
| `ACTIVE_BASELINE` | 해당 데이터셋의 기본 control이자 새 후보가 넘어야 하는 incumbent |
| `PROMOTED_DATASET` | 특정 데이터셋에서만 승격됐으며 공통 baseline을 대체하지 않음 |
| `REFERENCE_ONLY` | 외부 또는 과거 비교 기준이며 새 구조의 기본 시작점은 아님 |
| `NOT_PROMOTED` | 구현·screening은 완료했지만 acceptance gate를 통과하지 못함 |
| `CLOSED` | prerequisite 또는 screening 실패로 해당 실험 계열을 종료함 |
| `SELECTED_HYPOTHESIS` | 다음 설계·audit 대상으로 선택됐지만 구현·성능 승격은 아직 없음 |
| `DEFERRED` | 설계·scaffold만 있고 승격 판단에 필요한 모델 품질 실험이 없음 |
| `INFRA_ONLY` | 실행·재현성·artifact 계약만 검증했으며 모델 품질 근거가 아님 |

## 2. Frozen Model Identities

### V2 Common TitanTPP Baseline

| Axis | Frozen value |
| --- | --- |
| memory | `memory_mode=static_lmm` |
| decoder | `qty_decoder_mode=mark_residual` |
| encoder value input | `value_input_mode=residual` |
| value head | `value_head_mode=shared` |
| quantity-to-mark route | `qty_mark_gradient_mode=coupled` |
| value-to-encoder route | `value_encoder_gradient_mode=coupled` |
| time head | `time_head_mode=shared` |
| marker objective | `marker_loss_mode=ce`, `lambda_ordinal=0` |
| quantity objective | `loss_mode=hybrid`, `lambda_qty=0.25` |
| supervision | `train_loss_scope=target_only` |

V2의 dataset별 Titan candidate는 V1/V2 e200 multi-seed baseline lock을 따른다:
Intermittent `small_lmm`, Taxi `mid_lmm`, Instacart `small_lmm`.

### V3b Taxi-Specific Model

V3b는 Taxi V2와 같은 `mid_lmm` 및 학습 계약을 사용하고 아래 두 축만 바꾼다.

| Axis | V2 | Taxi V3b |
| --- | --- | --- |
| value head | `shared` | `mark_conditioned_experts` |
| quantity-to-mark route | `coupled` | `detached` |

V3b는 Taxi의 dataset-specific incumbent다. Intermittent와 Instacart의 공통
baseline을 대체하지 않는다.

## 3. Dataset Baseline Lock

| Dataset | Common/attribution control | Active incumbent | Current decision |
| --- | --- | --- | --- |
| Intermittent | V2 `small_lmm` | V2 `small_lmm` | V3/V5/M/Q 후보 미승격; 새 후보는 V2와 비교 |
| Yellow Trip Hourly / Taxi | V2 `mid_lmm` | V3b `mid_lmm` | 대체 후보는 V3b를 넘어야 하며 V2는 구조 효과 분리용 control로 함께 유지 |
| Instacart | V2 `small_lmm` | V2 `small_lmm` | 현재 enhancement 품질 승격 후보 없음; top-20 e1은 integration 근거만 제공 |

`BEST_TITAN_BY_DATASET`의 Instacart `mid_lmm`는 generic full-model recommendation이다.
모델 강화 트랙의 V2 e200 baseline lock은 `small_lmm`이므로 matched architecture
comparison에서 둘을 자동 교체하지 않는다. Instacart `mid_lmm`를 쓰려면 별도
capacity ablation과 fresh matched control로 취급한다.

## 4. Model Promotion Registry

| Label | Structural change | Implementation/evidence | Status | Active scope |
| --- | --- | --- | --- | --- |
| R0 | RMTPP | 기존 recurrent TPP reference | `REFERENCE_ONLY` | 최종 외부 비교 |
| L0 | Legacy TitanTPP: no value input, residual-only | 과거 TitanTPP reference | `REFERENCE_ONLY` | legacy 개선폭 확인 |
| V1 | residual value input + residual-only loss | e200 multi-seed 비교 완료 | `REFERENCE_ONLY` | score/NLL 보조 guardrail |
| V2 | residual value input + hybrid quantity objective | e200 multi-seed baseline 확정 | `ACTIVE_BASELINE` | 전 데이터셋 common control |
| V3a | mark-conditioned value experts, coupled gate | Intermittent/Taxi e50 screening 완료 | `NOT_PROMOTED` | coupled architecture ablation |
| V3b | V3a + detached quantity mark gate | Taxi strict matched e50, seeds 42/52/62 통과 | `PROMOTED_DATASET` | Taxi only |
| V3c | V3b + detached value-to-encoder route | Intermittent seed-42 e50 gate 실패 | `NOT_PROMOTED` | code/gradient ablation only |
| V4a/V4b | mark-conditioned time intercept | Taxi strict validation-only e50 gate 실패 | `NOT_PROMOTED` | experimental code only |
| V5a | CE + normalized RPS ordinal auxiliary | Intermittent seed-42 e50 gate 실패 | `NOT_PROMOTED` | objective ablation only |
| V5b | class-prior correction | fallback idea only | `DEFERRED` | no active experiment |
| M0 | direct `log2(qty)` + train-global normalization | Intermittent validation gate 실패 | `NOT_PROMOTED` | log-domain negative ablation |
| M1-M4 | log-domain RevIN family | M0 prerequisite 실패 후 미실행 | `CLOSED` | no active experiment |
| Q0 | direct raw quantity + global moments | Intermittent validation gate 실패 | `NOT_PROMOTED` | raw-domain control |
| Q1 | direct raw quantity + canonical causal RevIN | scale collapse 및 validation gate 실패 | `NOT_PROMOTED` | normalization diagnostic |
| Q2 | direct raw quantity + shrinkage RevIN | raw MAE 개선, low-scale/mark safety 실패 | `NOT_PROMOTED` | normalization foundation only |
| Q3a/Q3b/Q3c | Q2 gradient routing/log auxiliary factorial | Intermittent validation gate 미통과 | `CLOSED` | implementation retained |
| V6 | causal pre-window series memory adapter | train-only final primary와 bootstrap gate 실패; adapter 미구현 | `CLOSED` | no active experiment |

## 5. Decision Evidence

- V2는 V1 대비 Intermittent quantity MAE `9.94%`, Taxi quantity MAE `25.86%`
  개선을 포함한 e200 multi-seed 결과를 근거로 공통 strong baseline이 됐다.
- Taxi V3b는 V2와 동일 e50·seeds `42,52,62` 비교에서 total NLL `2.335%`,
  marker NLL `16.448%`, quantity MAE `49.086%`, value MAE `27.303%` 개선과
  mark accuracy `+0.729%p`를 보였다. Time NLL은 `0.181%` 악화됐지만 사전
  guardrail 안이었고 세 seed가 동시에 gate를 통과했다.
- V4a/V4b는 Taxi validation-only에서 paired time NLL 개선이 각각
  `0.415%/0.321%`로 `0.5%` primary gate에 미달했다. V4a는 DT MAE도
  `4.172%` 악화돼 multi-seed와 held-out 평가를 열지 않았다.
- V5a, M0, Q0/Q1/Q2, Q3는 일부 quantity 지표가 좋아져도 marker 또는
  dominant low-quantity safety를 동시에 만족하지 못했으므로 승격하지 않았다.
- strict Q2 e3 A/B exact comparator의 `22/22` 일치는 deterministic runner의
  `INFRA_ONLY` 통과다. Q2 자체의 성능 승격이나 과거 standard artifact의 exact
  재현을 뜻하지 않는다.
- V6 train-only audit은 충분한 coverage를 확인했지만, 선택 primary인 marker CE의
  최종 개선이 `0.6235%`로 `1%` threshold에 미달했고 series-bootstrap 95% CI
  `[-1.7265%, 2.9784%]`가 0을 포함했다. `M64/topk4`를 동결하지 않고 adapter를
  구현하지 않은 채 V6를 종료했으며 Taxi incumbent는 V3b로 유지한다.

## 6. Comparison And Unlock Rules

1. Intermittent와 Instacart의 새 구조 후보는 frozen V2와 fresh matched 비교한다.
2. Taxi replacement 후보는 V3b를 primary incumbent로 넘고, V2와도 함께 비교해
   value-head 효과와 새 구조 효과를 분리한다.
3. Candidate, split, seed, epoch, optimizer, lookback, `max_seq_len`, objective,
   checkpoint rule, reproducibility mode를 pair 안에서 동일하게 유지한다.
4. Smoke/CUDA/Instacart top-20 e1 통과는 구현·integration 근거일 뿐 승격 근거가 아니다.
5. 첫 quality screen은 strict fixed-split validation-only로 수행한다. 사전 gate를
   통과한 후보만 multi-seed로 확장하고, 구조와 coefficient를 freeze한 뒤에만
   held-out test를 한 번 연다.
6. 기존 e200 V1/V2와 Taxi V2/V3b matched held-out 결과는 역사적 결정 근거로
   유지한다. 새 인과 비교에서는 과거 수치를 exact control로 재사용하지 않고
   현재 strict runner로 fresh matched baseline을 함께 실행한다.

## 7. Source Of Truth

- Strategy: `simple_lab_test/search/model_enhancement_strategy.md`
- Experiment guide: `simple_lab_test/search/search_experiment_guide.md`
- Architecture ADRs: `.agents/results/architecture/adr-titantpp-*.md`
- Notion source draft:
  `simple_lab_test/search/notion_writer_prompts/titantpp_model_status_baseline_registry_0717.md`
