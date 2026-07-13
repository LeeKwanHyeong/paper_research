# ADR: TitanTPP Raw-Quantity Q0/Q1/Q2 RevIN Contract

- Date: 2026-07-13 KST
- Status: seed-42 validation gate failed for Q0/Q1/Q2; held-out test locked
- Scope: TitanTPP Intermittent fixed-split model enhancement
- Supersedes: no prior raw-domain contract
- Related: `adr-titantpp-parallel-magnitude-shrinkage-revin.md`
- Successor: `adr-titantpp-q3-factorial-gradient-dual-domain.md`

## Context

The completed M0 experiment predicted `log2(qty)` with fixed train-global
normalization. It did not compute per-window instance statistics and therefore
was not a raw-quantity RevIN experiment. M0 failed its validation gate, so the
dependent log-domain M1-M4 branch is closed, but raw-quantity RevIN remains
untested.

The Intermittent train-only loader audit establishes the operating constraints:

- context count median/p95/max: `3/11/12`;
- `67.63%` of targets have at most four history events;
- one-event context share: `22.66%`;
- zero-variance context share: `35.23%`;
- target outside the observed history range: `28.70%`.

These observations make unmodified per-window standardization a useful method
check but a risky primary candidate. They also show why raw/global normalization
must not be used as a prerequisite that can veto instance normalization: the
purpose of RevIN is precisely to address cross-series and local level/scale
shift that a global transform cannot remove.

## Decision Drivers

1. Test raw-quantity RevIN without a hidden log transform.
2. Exclude the appended target and padding from every normalization statistic.
3. Keep marker and time probabilistic heads, split, budget, and parameter count
   matched so the effect is attributable to normalization.
4. Protect the short-context majority from zero-variance scale collapse.
5. Preserve the existing `best_val_nll` checkpoint and held-out test lock.
6. Avoid claiming a RevIN benefit unless an instance-normalized candidate beats
   the matched raw/global control.

## Non-Goals

- Removing the categorical marker head or changing the RMTPP time intensity.
- Reopening the failed log-domain M1-M4 branch.
- Adding statistic context, learnable RevIN affine parameters, last-value
  centering, detached encoder gradients, or a new checkpoint criterion in the
  first Q0/Q1/Q2 comparison.
- Tuning coefficients or shrinkage constants using validation or held-out test.
- Applying the standalone Titan forecasting wrapper's mutable `RevIN` buffers to
  TitanTPP. The TPP path requires a stateless masked context builder.

## Alternatives Considered

### A. Stop after log-domain M0

Rejected because M0 did not test instance normalization or raw quantity. This
would generalize beyond the evidence.

### B. Run only raw/global direct regression

Rejected as a complete answer. It is required as a domain control but cannot
establish whether RevIN handles local distribution shift.

### C. Run only canonical raw RevIN

Rejected because `35.23%` zero-variance contexts and a `22.66%` one-event share
make the expected failure mode too common to leave without a stabilized matched
candidate.

### D. Matched Q0/Q1/Q2 comparison

Selected. Q0 isolates the raw target domain, Q1 checks masked mean/std RevIN, and
Q2 tests whether train-global moment shrinkage stabilizes short contexts. Q0 is
a control, not a prerequisite; all three enter the first seed-42 screen if the
raw train-only audit and implementation gates pass.

## Shared Model Contract

All candidates use the following architecture:

```text
observed event history
  -> mark embedding
  -> log1p(delta-time)
  -> normalized raw-quantity feature
  -> Titan MemoryEncoder
       |- categorical marker head       (unchanged CE/NLL)
       |- continuous-time RMTPP head     (unchanged time NLL)
       `- direct raw-quantity head       (normalized raw target)
```

Configuration identity:

```text
qty_decoder_mode = direct_raw_qty
magnitude_norm_mode = global | causal_revin | causal_shrinkage_revin
magnitude_center_mode = mean
magnitude_revin_affine = false
magnitude_stat_context_mode = none
train_loss_scope = target_only
marker_loss_mode = ce
lambda_ordinal = 0
value_head_mode = shared
qty_mark_gradient_mode = coupled
value_encoder_gradient_mode = coupled
```

`direct_raw_qty` is exclusive with `mark_residual` and `direct_log_qty`. It owns
the continuous history feature, direct target, quantity prediction, and quantity
loss. The legacy value head is not constructed. Predicted marks and mark
probabilities are not used to reconstruct the direct quantity prediction.

Q0/Q1/Q2 have identical learnable parameters and initialization. Their only
difference is how the stateless `MagnitudeContext` computes `center` and `scale`.
No candidate receives `[center, scale, count]` as an extra head input in this
first comparison.

## Raw Quantity Contract

The loader continues to provide magnitude-factorized marks and residuals. Raw
quantity is reconstructed without using a log-domain training target:

```text
q_t = 2 ** (mark_t + scale_residual_t)
```

The exponent reconstruction is a data-interface operation. The model target and
prediction domain are raw quantity. Train-global aggregation uses float64; model
features and losses use the model dtype.

For each sample, `H` is the valid-token mask after removing the final valid
token, which the weekly loader reserves as the target. Padding and the appended
target never enter `center`, `scale`, or `normalized_history`.

```text
u_i      = (q_i - center) / scale, i in H
u_target = (q_target - center) / scale
u_hat    = magnitude_head(h_last_history)
q_affine = center + scale * u_hat
q_hat    = max(q_affine, 0)                 # evaluation/inference only
```

The unclamped `q_affine` enters the raw quantity Huber loss so negative outputs
retain a corrective gradient. Evaluation uses non-negative `q_hat` and exports
the pre-clamp negative-prediction share. There is no upper semantic clamp; any
non-finite output fails the runtime gate.

## Variant Contract

| Variant | `magnitude_norm_mode` | Statistics | Role |
| --- | --- | --- | --- |
| Q0 | `global` | fixed train-global raw moments | raw-domain control; not RevIN |
| Q1 | `causal_revin` | masked history mean/population variance | canonical masked RevIN diagnostic |
| Q2 | `causal_shrinkage_revin` | history/global mixed moments | primary short-context candidate |

### Q0: Raw Train-Global Control

```text
center = mu_global_raw
scale  = max(sigma_global_raw, sigma_floor_raw)
```

`mu_global_raw` and population variance are computed from fixed-split train
events only. Q0 does not use instance statistics and cannot support a RevIN
claim.

### Q1: Causal Masked Mean/Std RevIN

```text
n       = sum(H)
mu_h    = sum(H_i * q_i) / n
var_h   = sum(H_i * (q_i - mu_h)^2) / n
center  = mu_h
scale   = sqrt(var_h + revin_eps)
```

The first activation fixes `revin_eps=1e-5`, `affine=false`,
`subtract_last=false`, and requires `n>=1`. Mean centering and no learnable
affine keep the method attributable and match the existing one-channel
projection/head capacity. The one-event/constant-context collapse is measured,
not silently replaced with global statistics.

### Q2: Causal Raw-Moment Shrinkage RevIN

```text
alpha  = n / (n + k)
center = alpha * mu_h + (1 - alpha) * mu_g
m2     = alpha * (var_h + mu_h^2)
       + (1 - alpha) * (var_g + mu_g^2)
var    = max(m2 - center^2, sigma_floor_raw^2)
scale  = sqrt(var)
```

Mixing first and second moments preserves within- and between-level variance;
directly averaging standard deviations is not allowed. Q2 uses no extra model
parameters and no statistic-context feature.

## Loss Contract

No log transform enters the training objective.

```text
L_raw_norm = mean_masked Huber(u_hat, u_target)
L_raw_qty  = mean_masked Huber(q_affine / qty_scale,
                               q_target / qty_scale)

L_total = CE_marker
        + 1.0 * NLL_time
        + 1.0 * L_raw_norm
        + 0.25 * L_raw_qty
```

The first comparison fixes `qty_scale=1`, `lambda_magnitude=1.0`, and
`lambda_qty=0.25`, matching the existing direct-decoder coefficient contract.
Huber bounds the per-sample tail gradient. Coefficient tuning is out of scope
until a normalization candidate passes the fixed gate.

Likelihood identity is unchanged:

```text
nll_marker = categorical CE only
nll_time   = continuous-time NLL only
nll        = nll_marker + nll_time
```

`L_raw_norm` and `L_raw_qty` never enter `nll`. `log2_qty_mae` remains an
evaluation-only low-scale balance metric computed after non-negative quantity
reconstruction; it is not a loss.

## Raw Train-Only Audit Gate

Before implementation constants are frozen, rerun the exact fixed-split weekly
context audit in raw quantity space. Validation and test rows remain unread.

Required outputs:

- train raw event count, mean, population variance/std, min, median, p95, p99,
  and max;
- context count and zero-variance distributions;
- Q1 scale and absolute normalized-target distributions;
- Q0 normalized-target reference distribution;
- Q2 candidates for `k in {1,2,4,8,16}`;
- scale and normalized-target distributions by `n=1`, `n<=4`, and all contexts;
- target-outside-history-range and early/late raw level-shift diagnostics.

Q2 freezes:

```text
sigma_floor_raw = max(0.001 * sigma_global_raw, 1e-4)
```

A `k` candidate is eligible only when:

- all centers, scales, and normalized targets are finite;
- median one-event scale is at least `0.50 * sigma_global_raw`;
- median `alpha` is at least `0.25`;
- target absolute normalized p99 does not exceed Q0 raw/global p99;
- share `abs(u_target)>3` does not exceed the Q0 raw/global share.

Choose the eligible candidate with the lowest target absolute normalized p99,
then the smaller `k`. If no candidate is eligible, Q2 is blocked before model
implementation and the audit result is recorded rather than tuning on
validation.

### Frozen Audit Result

The 5090 audit completed on `2026-07-13` using `159,643` fixed-split train rows,
`23,387` series, and `136,256` exact weekly train-target contexts. Source quality,
target count, and context-length distribution gates passed. Validation and
held-out test rows were not read.

Raw quantity is strongly right-tailed: mean/median/p99/max are
`6.8459/2/65/5000`, the top `1%` contributes `42.59%` of quantity, and
between-series level differences explain `78.10%` of raw variance. The
short-context constraint remains dominant: `67.63%` have at most four events and
`35.23%` have zero population variance.

Q1 confirms the expected plain-RevIN failure mode. Its scale p01 is `0.003162`,
target absolute normalized p99 is `2846.0499`, and `22.0636%` of targets exceed
absolute normalized magnitude three. It remains a diagnostic ablation rather
than the primary candidate.

Q2 `k=8` passed all predeclared gates and minimizes target absolute normalized
p99 among eligible candidates:

```text
shrinkage_k=8
revin_eps=0.00001
sigma_floor_raw=0.0550124034288891
global_mean_raw=6.8458560663480394
global_var_raw=3026.3645310228494
global_std_raw=55.0124034288891
```

At `k=8`, median `alpha` is `0.2727`, one-event median scale is `0.9432` of the
global standard deviation, target absolute normalized p99 is `0.7968`, and the
share above three is `0.0514%`. Q0 references are `1.1844` and `0.4073%`,
respectively. These values freeze implementation constants; they do not establish
model accuracy or a RevIN benefit.

## Checkpoint And Artifact Contract

- Primary checkpoint remains `best_val_nll`.
- `best_val_qty_mae` is diagnostic only and cannot select Q0/Q1/Q2.
- Normalization constants and `k` are selected from train only; model candidates
  are selected with validation only.
- Unified files that co-locate test columns are not opened before the validation
  decision is recorded.
- Manifest, checkpoint, cache identity, history, summary, scale-wise, and report
  rows include decoder mode, normalization mode, raw domain, `revin_eps`, `k`,
  sigma floor, affine mode, center mode, and statistic-context mode.
- Run path includes
  `qtydecoder_direct_raw_qty/magnorm_<mode>/k_<value>` where applicable.

Required raw-specific metrics:

- quantity MAE/RMSE/WAPE;
- evaluation-only log2 quantity MAE/RMSE;
- context-count-wise quantity and log2 MAE for `1`, `2-4`, `5-8`, `9+`;
- quantity scale-wise metrics;
- pre-clamp negative-prediction share;
- center/scale p01/p50/p95/p99 and scale-floor activation share;
- Q1/Q2 normalized-target p95/p99 and non-finite count.

Normalized loss values are not compared directly across Q0/Q1/Q2 because their
centers and scales differ.

## Focused Implementation Gate

1. Raw reconstruction matches `2 ** (mark + residual)` for real marks.
2. Mutating the appended target changes the target but not context statistics or
   normalized history.
3. Mutating padding leaves context and predictions unchanged.
4. Left- and right-padded representations produce equivalent contexts.
5. Q0 uses only frozen train-global raw moments.
6. Q1 masked mean/population variance and `n=1` behavior match the equation.
7. Q2 moment mixing and `k -> 0` / large-`k` limits match the equation.
8. `denorm(norm(q_history))` and target normalize/denormalize round trips pass
   within dtype tolerance.
9. Q0/Q1/Q2 parameter keys, parameter count, and seeded initialization are exact
   matches.
10. Direct quantity prediction is invariant to mark-logit mutation.
11. Isolated raw losses update the magnitude head and shared encoder but not the
    marker/time heads directly; marker/time losses do not update the magnitude
    head.
12. Negative affine predictions retain gradients through `L_raw_qty`; only
    evaluation/inference applies the non-negative clamp.
13. Invalid mixed decoders, non-fixed split, non-target-only scope, ordinal/V3
    combinations, and contextual TTM fail fast.
14. Artifact path, manifest, checkpoint, resume, and cache identity distinguish
    all three variants from M0 and legacy V2.

Local implementation completed on `2026-07-13`. The dedicated raw contract suite
passed `22/22`, the complete search test suite passed `85/85`, and Q0/Q1/Q2 CPU
model-tests all completed with finite forward/loss outputs. The gate includes
exact seeded parameter-state equality, target/padding isolation, equations and
round trips, gradient routing, negative-affine gradients, and cache/resume
identity checks.

The matched 5090 CUDA model-test then passed for Q0, Q1, and Q2. All variants
reported `status=success`, `device=cuda`, hidden shape `[4,16,64]`, finite
contract fields, and the same `78,111` parameters. The model-test derived its
global moments and effective floor from synthetic history, so this is CUDA
runtime evidence rather than actual-data or RevIN-benefit evidence.

### Actual-Data Integration Smoke Result

The matched Instacart top-20 e1 fixed-split smoke completed on `2026-07-13` for
Q0, Q1, and Q2. All three variants returned exit code zero and `status=success`,
used identical train/validation/test sample counts `1380/300/300`, and persisted
the same train-only raw mean/variance/std `13.7700/45.9371/6.7777`. Required
checkpoints, histories, validation/test summaries, scale-wise tables, reports,
and plots were generated without runtime non-finite values.

Q1 remained finite but reproduced the expected canonical RevIN failure mode. Its
test scale p01 reached `sqrt(1e-5)=0.003162`, normalized-target absolute p99
reached `1268.0743`, and epoch train loss reached `98.8030`. Q2 shrinkage kept
validation/test normalized-target p99 at `2.7290/2.4759` and epoch train loss at
`4.9314`, comparable to Q0's `4.9894`. This confirms stabilization of the
normalization path, not a performance benefit.

The smoke subset had targets only in the `1-9` and `10-99` buckets. Q2 improved
the majority `10-99` bucket but regressed `1-9`, and there were no `100+` targets.
The single epoch is not a convergence study, held-out test values were inspected
only for artifact integrity, and no Q0/Q1/Q2 selection is made from this result.

The Intermittent seed-42 e50 validation-only screening ran on 5090 from
`2026-07-13 17:14:05` to `17:37:12 KST`. The frozen V2 validation reference
completed first with `41,901` samples, and Q0/Q1/Q2 then completed `50/50`
epochs with exit code zero. Held-out test artifacts were not opened.

## Seed-42 e50 Validation-Only Gate

Q0, Q1, and Q2 run with the same Intermittent seed, candidate, epoch, learning
rate, batch size, lookback, max sequence length, split, and checkpoint policy.
Q0 failure does not cancel Q1 or Q2.

Eligibility versus the frozen V2 validation reference:

### Quantity benefit

- raw quantity MAE improvement `>=3%`;
- history count `<=4` raw quantity MAE improvement `>=3%`;
- evaluation log2 quantity MAE regression `<=2%`;
- every quantity bucket with validation share `>=5%` has MAE regression `<=5%`.

### Mark/time safety

- marker NLL regression `<=1%`;
- total NLL regression `<=0.5%`;
- time NLL regression `<=0.5%`;
- mark accuracy gap `>=-0.25%p`;
- DT MAE regression `<=2%`.

### Numeric safety

- all losses, predictions, centers, and scales are finite;
- pre-clamp negative-prediction share `<=1%`;
- no target/padding leakage test failure.

An instance-normalized candidate supports a RevIN benefit only when it is also
eligible versus V2 and improves Q0 by:

- overall raw quantity MAE `>=2%`;
- history count `<=4` raw quantity MAE `>=3%`;
- evaluation log2 quantity MAE regression `<=1%`;
- all mark/time and numeric safety gates remain satisfied.

Selection rule:

- If Q1 and Q2 both pass and Q2 improves neither overall nor `n<=4` quantity MAE
  by at least `1%` versus Q1, select simpler Q1.
- Select Q2 only when shrinkage adds measurable benefit or Q1 fails its numeric
  or task gate while Q2 passes.
- If Q0 passes V2 but Q1/Q2 do not beat Q0, retain only a raw direct-regression
  result and make no RevIN claim.
- If no candidate is eligible versus V2, stop the raw branch.
- If Q1/Q2 pass quantity gates but fail marker safety, do not promote them; a
  later Q2b detached magnitude-to-encoder gradient route may be designed as a
  separate ablation.

## Seed-42 Screening Result

All values use each model's `best_val_nll` checkpoint. Q0, Q1, and Q2 selected
epochs `48/42/46` respectively.

| Metric | V2 | Q0 | Q1 | Q2 |
| --- | ---: | ---: | ---: | ---: |
| Raw quantity MAE | `3.060182` | `2.820415` | `2.639700` | `2.606458` |
| Raw MAE improvement | baseline | `7.835%` | `13.740%` | `14.827%` |
| History `<=4` raw MAE improvement | baseline | `3.133%` | `15.265%` | `14.838%` |
| Log2 MAE regression | baseline | `8.484%` | `1.391%` | `7.310%` |
| Total NLL regression | baseline | `-0.976%` | `1.602%` | `-0.723%` |
| Marker NLL regression | baseline | `0.296%` | `7.109%` | `0.018%` |
| Mark accuracy gap | baseline | `-2.809%p` | `-2.814%p` | `-3.253%p` |

Q0 and Q2 regressed the dominant `1-9` quantity bucket by `6.266%` and
`7.616%`; Q1 stayed within the bucket gate at `1.518%`. Q1 nevertheless failed
marker/total NLL, mark accuracy, and DT safety. Its scale p01 reached `0.003162`,
normalized-target p99 reached `1897.3666`, and magnitude loss reached `183.5540`.
Q2 stabilized those diagnostics to `35.3003`, `0.6668`, and `0.00673`, but did
not repair log2, low-quantity, or mark-accuracy safety.

The seed-42 eligibility decision is `FAIL` for Q0, Q1, and Q2. No candidate is
frozen for multi-seed, no RevIN benefit is claimed, and held-out test artifacts
remain locked.

## Strict Multi-Seed And Held-Out Gate

After seed-42 selection, freeze one candidate and all constants. Run matched V2,
Q0, and selected Q1/Q2 with seeds `42,52,62` and e50.

- `3/3` runs complete without non-finite values or runtime errors;
- mean V2 benefit and Q0 RevIN-benefit thresholds remain satisfied;
- seed-matched overall quantity improvement versus V2 and Q0 occurs in at least
  `2/3` seeds;
- mean mark accuracy gap versus V2 is `>=-0.25%p` and no seed is below
  `-0.75%p`;
- mean marker/total/time NLL, DT MAE, log2 quantity, context, bucket, and numeric
  safety gates remain satisfied.

Only then unlock held-out test artifacts in protocol order. The frozen test
audit uses the same mean and seed-safety rules. A failed test audit returns to V2
without tuning Q0/Q1/Q2 on the observed test result.

## Risks And Mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Raw train-global moments are tail dominated | Q0 underfits low quantities | Q0 is a control; require low-scale/log2 and bucket safety |
| Q1 zero variance creates extreme targets | Slow or unstable learning | Keep Q1 diagnostic; report scale/target tails; compare Q2 |
| Q2 hides absolute level | Magnitude head misses scale effects | Denormalization restores level; no stat context in first attribution test |
| Raw loss harms mark/time encoder | Quantity gain with mark collapse | Keep strict safety gate; route only later to Q2b detachment |
| Clamp hides invalid negative forecasts | Artificially improved metrics | Train on unclamped affine output and report negative share |
| Best NLL misses best quantity epoch | Understates quantity potential | Export best-qty diagnostic but keep checkpoint policy fixed |
| Validation constants overfit | False RevIN benefit | Select all statistics and `k` from train-only audit |

## Consequences

The raw branch is now a completed seed-42 matched normalization study rather
than a continuation of the failed log-domain M0-M4 ladder. It shows that raw
normalization can reduce overall and short-context absolute quantity error, and
that shrinkage removes Q1's numeric collapse. It does not provide an eligible
replacement for V2 because low-quantity/log2 and marker safety are not jointly
preserved.

## Next Step

Do not run Q0/Q1/Q2 multi-seed or unlock held-out test. The accepted Q3 successor
contract crosses magnitude-to-encoder gradient isolation with a low-quantity
log2 auxiliary in a complete `2 x 2` design. Implement and validate that contract
before another direct-raw screening. Keep V5b class-prior correction as a
separate fallback.
