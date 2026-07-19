# Notion Update: Post-V6 Candidate Selection

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 날짜 구역: `2026-07-19 | Post-V6 Model Candidate Selection`
- 세부 Step: `Step 1. V5b Class-Prior vs V7 Causal Time-History Adapter`
- Model Enhancement 내용이므로 `2. Confirm and Refine Topic`에는 작성하지 않는다.
- 같은 제목의 페이지가 있으면 업데이트하고, 없을 때만 새 페이지를 만든다.

## 페이지 제목

`TitanTPP Post-V6 Candidate Selection: V5b vs V7 Time History`

## 작성 원칙

- 연구 실험 노트처럼 간결하게 작성한다.
- 상태, 목적, Factorial 계약, 고정 조건, 실행 명령어, 결과 순서로 정리한다.
- V6의 실패 판정을 바꾸거나 secondary time signal을 모델 성능으로 쓰지 않는다.
- V7은 구현·승격 완료가 아니라 Stage-0 audit 대상으로 선택된 상태라고 쓴다.
- `획기적`, `강력한`, `유의미한`, `명확히 입증`, `종합적으로` 같은 표현은 쓰지 않는다.
- checksum, preflight 세부 로그, Frozen Reference 섹션은 넣지 않는다.

## 상태

- 날짜: `2026-07-19 KST`
- 상태: `candidate selected; model implementation locked pending train-only audit`
- 실행 서버: 후속 작업은 `5080`
- active baseline: Intermittent V2 `small_lmm`, Taxi V3b `mid_lmm`
- V5b: `DEFERRED`
- V6: `CLOSED`
- V7: `SELECTED_HYPOTHESIS`

## 목적

V6 종료 후 다음 강화 후보를 V5b class-prior correction과 별도 Taxi
time-history architecture 사이에서 비교하고, 한 후보만 다음 audit 대상으로
선정한다.

## 비교 결과

| 후보 | 해결 대상 | 확인 근거 | 제한 | 판정 |
| --- | --- | --- | --- | --- |
| V5b class-prior correction | Intermittent marker imbalance | train marks `0-2` 비중 `86.60%`, effective classes `4.31` | 기존 실패는 rare tail보다 high-support mark `0/1` 경계에 집중; posterior calibration과 marker NLL 계약 필요 | fallback으로 보류 |
| V7 causal time-history adapter | Taxi time modeling | V6 final train-only `log1p(dt)` MAE `2.4696%` 개선, 95% CI `[1.5236%, 3.5050%]`, series improved share `67.176%` | V6 probe가 time/mark/quantity feature를 함께 사용했으므로 time-only source 분리 필요 | Stage-0 audit 대상으로 선정 |

V6 primary였던 marker CE는 `0.6235%` 개선에 그쳤고 95% CI
`[-1.7265%, 2.9784%]`가 0을 포함했다. 따라서 V6는 계속 종료 상태이며
`M=64/topk=4`를 V7 상수로 재사용하지 않는다.

## V7 모델 계약

```text
h_base = LMM_static(TitanEncoder(x_active_context))
r_time = MaskedTimeRetrieve(stop_gradient(h_base), temporal_pre_window, mask)
a_v7   = v_t(h_base) + b_t + tanh(alpha_time) * delta_time(r_time)
alpha_time = 0
```

- pre-window 입력은 observed `delta_t`, event age, 24/168-hour phase만 사용한다.
- mark, quantity, series ID, target/future/current-window row는 외부 history에서 제외한다.
- adapter는 time intercept만 변경하며 marker/value head에 직접 연결하지 않는다.
- zero gate와 empty history에서는 paired V2/V3b와 정확히 같은 함수로 fallback한다.

## Factorial 계약

Stage 0 train-only source audit:

| Probe | 추가 source | 역할 |
| --- | --- | --- |
| P0 | 없음 | active-window control |
| P1 | pre-window temporal fields only | V7 primary source test |
| P2 | temporal + mark + quantity | 기존 V6 time signal attribution only |

Stage 0 통과 후 모델 비교:

| Variant | Value head | Qty-mark gradient | Time-history adapter | 역할 |
| --- | --- | --- | --- | --- |
| V2 | shared | coupled | off | attribution control |
| V3b | mark-conditioned experts | detached | off | Taxi incumbent |
| V7a | shared | coupled | on | isolated time-history effect |
| V7b | mark-conditioned experts | detached | on | Taxi replacement candidate |

## 고정 조건

- Stage 0은 Taxi fixed-split train parquet만 읽고 validation/test는 읽지 않는다.
- rolling-origin fold, 동일 target, 동일 scaler/probe 계약을 사용한다.
- P1 pooled `log1p(dt)` MAE 개선 `>=1%`, 개선 fold `>=2/3`, series-bootstrap
  CI lower `>0`을 모두 요구한다.
- target coverage `>=35%`, series coverage `>=80%`를 유지한다.
- P2 결과로 P1 실패를 대체하지 않는다.
- Stage 0 통과 후 strict Taxi seed-42 e50 validation-only 2x2를 수행한다.
- 모든 model variant에서 `mid_lmm`, static LMM, plain CE, residual input,
  hybrid quantity loss, target-only scope, lookback `168`, max sequence `256`,
  optimizer, batch, checkpoint rule을 동일하게 둔다.
- V7b는 V3b 대비 overall/eligible time NLL `>=0.5%/>=1%`, total NLL
  `>=0.25%` 개선과 marker/DT/quantity/scale/series guardrail을 모두 통과해야 한다.

## 실행 명령어

아직 없음. 이번 단계는 후보 선정과 계약 고정이며, Stage-0 audit runner 구현 후
5080 tmux 실행 명령을 별도 시작 기록에 추가한다.

## 결과

- Post-V6 primary: V7 causal time-history adapter
- 현재 범위: train-only time-source isolation audit
- V5b: V7 Stage 0 실패 시 재검토할 Intermittent fallback
- V6 generic memory: 종료 유지
- active model: Intermittent/Instacart V2, Taxi V3b 유지

## 다음 작업

`Taxi train-only P0/P1/P2 time-source isolation audit와 focused causal/rolling-fold test 구현`

## Local Source

```text
.agents/results/architecture/adr-titantpp-v7-causal-time-history-adapter.md
.agents/results/architecture/titantpp-model-status-baseline-registry.md
simple_lab_test/search/model_enhancement_strategy.md
search_artifacts/model_enhancement_titantpp_v6_taxi_train_memory_audit_0717
```

## Direct Notion Update

- detail page:
  `https://app.notion.com/p/3a2bbe405613813d8854f11a4701b8fb`
- parent history:
  `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- strategy page:
  `https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a`
- refetch verification: `3/3 PASS`
