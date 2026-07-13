# TitanTPP Raw-Quantity Q0/Q1/Q2 RevIN Contract

## 작성 위치

- `5. Model Design Enhancement > 2026-07-13 | Direct Magnitude Regression과 RevIN Track`
- 세부 페이지 제목: `TitanTPP Raw-Quantity Q0 Q1 Q2 RevIN Contract And Acceptance Gate`
- 상위 history에는 제목 3 `Step 6. Raw Q0/Q1/Q2 모델·Loss·Normalization Contract`로 연결
- `2. Confirm and Refine Topic`에는 작성하지 않음
- 동일 제목이 있으면 새 페이지를 만들지 않고 업데이트

## 상태

- 설계일: `2026-07-13 KST`
- 상태: `design completed; raw-domain audit and implementation not started`
- 실행 서버/tmux: 없음. 이번 단계는 architecture contract 설계이며 실험을 시작하지 않음

## 배경

- M0는 `log2(qty)` direct regression과 train-global normalization을 사용한
  log-domain negative ablation이며 RevIN 실험이 아님
- Intermittent context count median/p95/max는 `3/11/12`
- context `<=4` share는 `67.63%`, one-event share는 `22.66%`, zero-variance
  context share는 `35.23%`
- 따라서 raw/global, plain raw RevIN, short-context shrinkage RevIN을 분리해
  비교해야 하며 Q0 실패가 Q1/Q2 중단 조건이 되어서는 안 됨

## 선택 구조

```text
raw observed quantity history
  -> causal masked normalization
  -> Titan MemoryEncoder
       |- categorical marker head   (기존 CE/NLL)
       |- continuous-time head      (기존 time NLL)
       `- direct raw-quantity head  (normalized raw target)
```

공통 조건:

- `qty_decoder_mode=direct_raw_qty`
- `train_loss_scope=target_only`, fixed split, `scale_base=2`
- plain marker CE, V3/V5/ordinal/detached route 비활성
- marker/time head와 likelihood 의미 유지
- Q0/Q1/Q2 parameter와 initialization 동일
- statistic context와 learnable RevIN affine은 첫 비교에서 사용하지 않음
- standalone Titan wrapper RevIN을 호출하지 않고 stateless masked context builder 사용

## Variant

| Variant | Normalization | 역할 |
| --- | --- | --- |
| Q0 | train-global raw mean/std | raw-domain control, RevIN 아님 |
| Q1 | causal masked raw mean/std | canonical RevIN diagnostic |
| Q2 | causal raw history/global moment shrinkage | short-context primary candidate |

Raw target:

```text
q = 2 ** (mark + scale_residual)
u = (q - center) / scale
u_hat = magnitude_head(h)
q_affine = center + scale * u_hat
q_hat = max(q_affine, 0)  # evaluation/inference only
```

Appended target과 padding은 center, scale, normalized history에서 제외한다.
Training raw quantity Huber에는 clamp 전 `q_affine`을 사용해 음수 예측도 gradient를
유지하고, evaluation에서는 non-negative `q_hat`과 clamp 전 음수 비율을 함께 기록한다.

## Normalization

Q0:

```text
center = train_raw_mean
scale = max(train_raw_std, sigma_floor_raw)
```

Q1:

```text
center = masked_history_mean
scale = sqrt(masked_population_variance + 1e-5)
```

- `affine=false`, `subtract_last=false`, `history_count>=1`
- one-event/constant context의 scale collapse를 fallback 없이 진단

Q2:

```text
alpha = n / (n + k)
center = alpha * history_mean + (1-alpha) * global_mean
m2 = alpha * (history_var + history_mean^2)
   + (1-alpha) * (global_var + global_mean^2)
scale = sqrt(max(m2 - center^2, sigma_floor_raw^2))
```

- 표준편차를 직접 평균하지 않고 first/second moment를 혼합
- 추가 parameter나 statistic context 없음

## Loss

```text
raw_norm_loss = Huber(u_hat, u_target)
raw_qty_loss = Huber(q_affine, q_target)

total_loss = marker_ce
           + 1.0 * time_nll
           + 1.0 * raw_norm_loss
           + 0.25 * raw_qty_loss
```

- training loss에 log transform 없음
- `nll = nll_marker + nll_time` 의미 유지
- log2 quantity MAE는 low-scale balance 확인용 evaluation metric만 사용
- primary checkpoint는 계속 `best_val_nll`; best quantity checkpoint는 diagnostic only

## Train-Only Audit Gate

구현 전 exact weekly context를 raw quantity로 다시 audit한다.

- validation/test row 미사용
- raw global mean/std, median, p95/p99/max 기록
- Q1 scale/normalized-target tail 기록
- Q2 `k={1,2,4,8,16}` 비교
- `sigma_floor_raw=max(0.001*global_raw_std, 1e-4)`
- all finite, one-event median scale `>=0.5*global std`, median alpha `>=0.25`
- target normalized p99와 `abs(u)>3` share가 Q0보다 나쁘지 않은 후보만 eligible
- eligible 중 p99가 가장 낮고, 동률이면 작은 k 선택
- 통과 후보가 없으면 validation tuning 없이 Q2 구현을 보류

## Focused Test Gate

- raw reconstruction 정확성
- target mutation 시 context 불변, target만 변경
- padding mutation 및 left/right padding equivalence
- Q0/Q1/Q2 수식과 normalize-denormalize round trip
- Q0/Q1/Q2 parameter/state initialization exact match
- direct raw prediction의 mark logits 독립성
- raw loss gradient는 magnitude head/shared encoder로만 직접 전달
- negative affine prediction이 raw loss gradient를 유지하는지 확인
- invalid decoder/split/loss/V3/V5/TTM combination fail-fast
- manifest/checkpoint/cache/run path identity 분리

## Seed-42 e50 Validation Gate

Q0/Q1/Q2를 모두 동일 budget으로 실행한다. Q0 실패는 Q1/Q2 취소 조건이 아니다.

V2 대비 quantity benefit:

- overall raw quantity MAE `>=3%` 개선
- history count `<=4` raw quantity MAE `>=3%` 개선
- log2 quantity MAE regression `<=2%`
- validation share `>=5%` quantity bucket regression `<=5%`

V2 대비 marker/time safety:

- marker NLL `<=1%`, total/time NLL 각각 `<=0.5%` regression
- mark accuracy gap `>=-0.25%p`
- DT MAE regression `<=2%`

Numeric safety:

- 모든 loss/prediction/stat finite
- pre-clamp negative prediction share `<=1%`
- target/padding leakage 없음

RevIN benefit은 Q1/Q2가 위 V2 gate를 통과하면서 Q0 대비 다음을 만족할 때만 인정:

- overall raw quantity MAE `>=2%` 개선
- history count `<=4` raw quantity MAE `>=3%` 개선
- log2 quantity MAE regression `<=1%`
- marker/time/numeric safety 유지

Q1과 Q2가 모두 통과하고 Q2가 overall 또는 short-context에서 Q1보다 `1%` 이상
좋지 않으면 단순한 Q1을 선택한다. Q2가 quantity는 통과하지만 marker safety를
실패하면 승격하지 않고, 별도 Q2b gradient-routing 설계 후보로만 남긴다.

## Multi-Seed와 Test Lock

- seed-42 선택 후 V2/Q0/선택 Q1 또는 Q2를 seeds `42,52,62`, e50으로 strict match
- mean gate 유지, V2와 Q0 대비 seed-matched quantity 개선 각각 `>=2/3`
- mean mark accuracy gap `>=-0.25%p`, worst seed `>=-0.75%p`
- candidate/constants 고정 전 held-out test와 merged test artifact 미열람
- frozen held-out gate 실패 시 동일 test를 이용한 재튜닝 없이 V2 유지

## 결정

- Q0는 raw-domain control이며 prerequisite가 아님
- Q1은 canonical-method diagnostic, Q2는 short-context primary candidate
- 세 후보는 동일 parameter count, loss coefficient, budget으로 비교
- RevIN claim은 V2와 Q0를 모두 이기고 short-context/marker safety를 통과할 때만 가능
- 다음 작업은 Intermittent train-only raw history/variance/tail audit

## 로컬 근거

```text
.agents/results/architecture/adr-titantpp-raw-quantity-revin-q0-q1-q2.md
simple_lab_test/search/model_enhancement_strategy.md
models/RMTPPs/magnitude_normalization.py
models/RMTPPs/TitanTPP.py
```
