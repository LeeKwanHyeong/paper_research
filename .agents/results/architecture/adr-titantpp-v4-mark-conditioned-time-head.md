# ADR: TitanTPP V4 Mark-Conditioned Time Head

- Status: Implemented; local focused gates passed; 5090 integration pending
- Date: 2026-07-16
- Scope: TitanTPP probabilistic next-time head
- Baselines: V2 common baseline and Taxi V3b confirmed enhancement

## Context

V2 remains the common TitanTPP baseline, while V3b is retained only for Taxi.
V3, V5, and Q-series experiments changed quantity, marker, or gradient-routing
paths, but the RMTPP time head remained shared across every next mark:

```text
a_t = v_time(h_t) + b_time
f(dt | h_t)
```

This imposes conditional independence between the next mark and next time once
the history representation is fixed. The matched Taxi V2/V3b comparison also
showed that V3b's total-NLL gain was marker-driven while time NLL regressed by
`0.181%`. That historical held-out result motivates the hypothesis but is not
used to tune V4 or unlock a new held-out evaluation.

The next model hypothesis is therefore:

> A mark-conditioned additive time-intensity expert can model the remaining
> dependence between demand scale and inter-event time while preserving the
> RMTPP likelihood and the established V2/V3b quantity paths.

## Decision Drivers

- Change one untested model axis rather than reopen failed quantity variants.
- Preserve a valid marked point-process joint likelihood.
- Keep V2 behavior exact by default and start V4 from the same function.
- Avoid direct time-loss gradients into the marker probability gate.
- Share most time parameters because low-support marks cannot train independent
  RMTPP heads reliably.
- Keep the first screening validation-only and deterministic.
- Defer series-aware memory until the common runner can construct and inject
  per-series memory without future-event leakage.

## Options Considered

| Option | Benefit | Main cost / risk | Decision |
| --- | --- | --- | --- |
| Keep the shared time head | No implementation cost | Leaves the mark/time conditional-independence assumption untested | Reject as the next enhancement |
| Additive mark-conditioned time intercept | Nested extension of V2, low parameter cost, exact joint likelihood | Predicted-mark errors can propagate into deterministic dt prediction | Select as V4 |
| Soft mixture of mark-specific time densities | Marginalizes mark uncertainty | Couples time loss to mark logits and requires mixture-CDF inversion for median dt | Defer |
| V6 series-aware memory | Can learn persistent per-series patterns | Runner does not inject `series_memory`; changes data, state, leakage, and model contracts together | Defer |

## Decision

Add `time_head_mode=shared|mark_conditioned` to TitanTPP. The default remains
`shared`. V4 uses an additive, zero-initialized real-mark delta over the current
RMTPP log-intensity intercept while keeping the positive time slope shared.

Let `C` be the number of real marks, excluding PAD:

```text
a_shared(h_t) = v_time(h_t) + b_time
delta(h_t) = W_mark_time h_t + b_mark_time       # shape [..., C]
a_k(h_t) = a_shared(h_t) + delta_k(h_t)
w = softplus(w_raw) + w_min

log f(dt | h_t, k)
  = a_k + w * dt - exp(a_k) / w * (exp(w * dt) - 1)

log p(k, dt | h_t)
  = log p(k | h_t) + log f(dt | h_t, k)
```

`W_mark_time` and `b_mark_time` are initialized to zero. V4 therefore has the
same initial time-density and deterministic time prediction as its paired
baseline for every real mark. The existing shared `v_time`, `b_time`, and
`w_raw` remain trainable and receive all valid time-loss observations.

The mark-specific delta changes only the intensity intercept. A mark-specific
`w` is not included in the first version because it increases numerical risk,
adds sparse parameters, and changes the hazard shape rather than only its
history- and mark-dependent level.

### Frozen implementation constants (2026-07-16)

- use one hidden-dependent additive delta for each of the `C` real marks
- implement the delta as `Linear(d_model, C)` with both weight and bias zero-init
- keep the existing scalar positive slope `w` shared across all marks
- do not transfer the audit's fitted intercepts or boundary-selected `w=0.001`
- exclude PAD from the expert table and predicted-mark selection
- support only V4a `shared/coupled/coupled` and V4b
  `mark_conditioned_experts/detached/coupled` value routes in the first activation
- keep marker CE, mark-residual quantity decoding, target-only training, and
  contextual test-time memory disabled for the focused V4 screen

## Training And Inference Contract

### Training likelihood

For an observed next event `(k_true, dt_true)`, the true mark selects the time
expert used by the conditional time NLL:

```text
nll_marker = CE(mark_logits, k_true)
nll_time = -log f(dt_true | h_t, k_true)
```

This is supervised branch selection, not target leakage. `k_true` is not added
to the encoder input and does not affect `h_t`. It plays the same target role as
the label used by marker CE.

The isolated conditional time loss updates:

- the shared Titan encoder,
- shared `v_time`, `b_time`, and `w_raw`,
- only the selected real-mark row of `W_mark_time` and `b_mark_time`.

It does not update `mark_head` parameters because the marker probabilities are
not used to choose the training branch.

### Likelihood evaluation

Validation NLL uses the observed mark in `log_f_dt(..., marks=k_true)`. V2 is a
nested special case where all mark deltas are zero, so total and time NLL remain
directly comparable between paired variants.

### Deterministic prediction

Deployment-style dt prediction first chooses the model-predicted real mark and
then uses that mark's time branch:

```text
k_pred = argmax p(k | h_t)
dt_hat = conditional_median(h_t, k_pred)
```

The existing evaluation `u=0.5` policy is retained. An oracle-true-mark dt MAE
may be exported only as a diagnostic and must not be an acceptance metric.

### Stochastic joint sampling

If stochastic sampling is used later, sample the mark first and pass that same
sampled mark into the conditional time sampler. Independently sampling a mark
after sampling time is not allowed because it would violate the joint model.

## Factorial Contract

Taxi uses a 2x2 design so the time-head effect is identifiable both on the
common V2 baseline and on the confirmed V3b value head:

| Variant | `value_head_mode` | `qty_mark_gradient_mode` | `time_head_mode` | Role |
| --- | --- | --- | --- | --- |
| V2 | `shared` | `coupled` | `shared` | Common control |
| V3b | `mark_conditioned_experts` | `detached` | `shared` | Taxi value-head control |
| V4a | `shared` | `coupled` | `mark_conditioned` | Isolated time-head effect on V2 |
| V4b | `mark_conditioned_experts` | `detached` | `mark_conditioned` | Time-head effect on Taxi V3b |

All four variants keep the same Titan candidate, input features, marker CE,
quantity loss, split, lookback, maximum sequence length, optimizer, seed,
epoch budget, and checkpoint policy.

## Configuration And Artifact Identity

- Config/CLI: `time_head_mode=shared|mark_conditioned`
- Default: `shared`
- Initial support: TitanTPP only; non-Titan activation fails before execution
- Non-default run path: `timehead_mark_conditioned`
- Record the mode in root/run manifests, checkpoints, resume/cache identity,
  histories, validation summaries, and model-test outputs
- Preserve RNG state around construction of the zero-initialized delta layer so
  shared parameters start identically in paired variants
- PAD is never a time expert and cannot receive a delta-row gradient

Proposed model interfaces:

```text
log_f_dt(h, dt, marks=None)
sample_next_dt(h, u=None, marks=None)
```

In `mark_conditioned` mode, `log_f_dt` requires explicit observed marks and
fails fast if they are missing. `sample_next_dt` uses explicit marks when
provided and otherwise selects the predicted real mark. Shared mode ignores the
optional mark argument and preserves current behavior.

## Focused Acceptance Gates

Before any dataset training:

1. Shared-mode parameters, forward output, NLL, gradients, and sampling remain
   exact against the current V2 implementation.
2. Zero-initialized V4a/V4b time log-density and explicit-mark sampling are
   exact against their paired controls.
3. An isolated time loss gives no `mark_head` gradient and updates only selected
   mark-delta rows plus the shared time/encoder path.
4. PAD, left padding, `target_only`, and all-masked edge cases remain finite and
   select no invalid expert.
5. The conditional log-density matches the closed-form RMTPP equation and its
   sampled median is monotonic with the selected intercept.
6. Config, CLI, run path, cache, manifest, checkpoint, and resume identities
   distinguish shared and mark-conditioned modes.

### Local implementation result (2026-07-16)

- `time_head_mode=shared|mark_conditioned` is connected through RMTPP config,
  unified CLI, TitanTPP construction, model-test, run path, cache identity,
  checkpoints, summaries, histories, and validation artifacts
- conditional likelihood uses the observed real mark, while deterministic
  sampling without an explicit mark uses the model-predicted real mark and
  excludes PAD
- padded and unselected transitions are replaced only for safe expert lookup
  and remain excluded by the existing transition mask
- `evaluation_scope=validation_only` creates a distinct run identity and
  disables held-out test metric export during candidate screening
- focused and related regression suites passed `96` tests
- CPU V4a/V4b model-tests both passed with identical zero-init NLL `3.862866`;
  parameter counts were `297,516` and `298,032`

These checks establish implementation and initial-function equivalence only.
They do not establish CUDA integration or learned V4 performance.

## Data And Screening Gates

The first data step is a Taxi train-only audit. Report real-mark support and the
mean, median, IQR, and dispersion of `log1p(dt)` by next mark. Validation and
held-out targets are not read for this audit. The audit is diagnostic: it can
stop V4 before implementation if supported marks show no material conditional
time separation.

### Train-only audit outcome (2026-07-16)

The audit reconstructed `38,393` fixed-split train next-event targets from
`131` series and matched the loader target count and decoded mark/delta-time
values exactly. Validation and held-out targets were not read. Source-quality,
loader-equivalence, support, and temporal-holdout checks all passed.

All four real marks met the global support rule. Their target counts and shares
were `20,512/53.426%`, `9,555/24.887%`, `5,475/14.260%`, and
`2,851/7.426%`. The delta-time median was `1` for every mark, so the signal is
not a median shift. It is concentrated in the upper tail of mark `0`: its
`dt > 1` share was `32.61%` in the diagnostic fit partition and `33.36%` in
the eval partition, versus about `0.2%` for mark `1` and `0%` for marks `2`
and `3`.

The train-only distribution and likelihood diagnostics were material:

- eta-squared for next mark versus `log1p(dt)` was `0.123197`
  (`omega-squared=0.123126`)
- the per-series temporal `80/20` eval NLL changed from `1.577575` for the
  global intercept to `1.485791` for mark-conditioned intercepts, a `5.818%`
  improvement
- the `2,000`-replicate series bootstrap 95% interval was
  `[4.273%, 7.378%]`
- `98/131` series (`74.809%`) improved and no eval mark was unseen in fit

All `10/10` predeclared gates passed. This unlocks V4 constants freeze and
focused implementation, but it is not evidence that a trained V4 improves
validation or deployment metrics. The diagnostic used the observed next mark
to evaluate the conditional density; deployment-style delta-time prediction
must still use the predicted mark. No fitted audit intercept or slope is
transferred to model initialization. Mark `3` spans only `13` train series and
`33/131` series worsened in the diagnostic, so sparse-mark and per-series
guardrails remain required.

After focused tests and a 5090 CUDA model-test plus Instacart top-20 e1 smoke,
run a strict Taxi seed-42 e50 2x2 validation-only screen. Do not read held-out
outputs. At the validation-selected checkpoint, a V4 pair passes only if:

- time NLL improves by at least `0.5%` versus its paired control,
- total NLL regresses by no more than `0.5%`,
- dt MAE regresses by no more than `1.0%`,
- marker NLL regresses by no more than `2.0%`,
- mark accuracy regresses by no more than `0.25%p`,
- quantity MAE regresses by no more than `5.0%`.

V4b is the Taxi promotion candidate. V4a is an attribution control and may be
retained independently if it passes on V2 while V4b fails on V3b. Multi-seed
and held-out evaluation remain locked until a validation pair passes.

## Assumptions

- The next mark and delta time are jointly observed for every valid target.
- Fixed-split target construction and mark definitions remain unchanged.
- The observed target mark is available for joint-likelihood evaluation but is
  not available as an encoder feature at prediction time.
- V2 and Taxi V3b configurations remain frozen controls during V4 screening.
- Historical held-out metrics are motivation only; V4 thresholds and constants
  are frozen before any new held-out output is read.

## Consequences And Risks

Positive consequences:

- V4 tests a new model hypothesis without reopening failed quantity tracks.
- The joint marked-TPP likelihood remains explicit and probabilistically valid.
- V2 and V3b are nested controls with exact zero-delta starting functions.
- Additive sharing limits the cost of sparse next-mark classes.

Risks:

- Predicted-mark errors can worsen deployment dt MAE even when conditional time
  NLL improves with the true mark.
- Rare-mark delta rows may remain poorly estimated.
- Time gradients still update the shared encoder and can indirectly move marker
  or quantity metrics.
- A single seed can only screen the hypothesis, not confirm it.

Mitigations are the 2x2 controls, zero initialization, shared base head and
slope, deployment-style predicted-mark dt metric, guardrails, strict execution,
and validation-first promotion rule.

## Non-Goals

- No mark-specific `w` in the first V4 implementation
- No soft mark-probability mixture in the first V4 implementation
- No class-prior, ordinal, RevIN, direct-quantity, or Q3 changes
- No V6 series-memory implementation in the same change
- No automatic Q3 reopen
- No held-out or multi-seed execution before the validation gate

## Next Steps

1. Treat the Taxi train-only audit as complete and keep validation/test locked.
2. Treat V4 constants freeze and local focused implementation as complete.
3. Run the 5090 CUDA model-test and Instacart top-20 e1 smoke.
4. Run the strict Taxi V2/V3b/V4a/V4b seed-42 e50 validation-only screen.
5. Keep multi-seed and held-out evaluation locked until a validation pair passes.
