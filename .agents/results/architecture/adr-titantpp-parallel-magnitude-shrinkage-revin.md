# ADR: TitanTPP Parallel Direct Magnitude Decoder With Causal Shrinkage RevIN

- Date: 2026-07-13
- Status: M0 log-domain gate failed; raw-domain contract moved to a separate ADR
- Scope: Intermittent-first TitanTPP magnitude regression track
- Method: Design-Twice followed by ADR
- Successor: `adr-titantpp-raw-quantity-revin-q0-q1-q2.md`

## Context

Intermittent train targets are concentrated in magnitude marks 0-2 (`86.60%`).
The V2 validation prediction already favors mark 0, and V5a CE plus normalized
RPS increased the mark-0 prediction share from `45.03%` to `58.75%` while
mark-1 recall fell from `49.62%` to `24.66%`.

Class-prior correction can move this boundary, but it cannot remove the
discretization or the dependency of reconstructed quantity on a marker argmax.
The fixed-split data already contains an exact continuous log-magnitude target:

```text
z_t = log2(qty_t) = mark_t + scale_residual_t
```

This remains exact for tail-merged rows because the residual may exceed one.
A direct `z_(t+1)` decoder can therefore predict quantity without passing
through the imbalanced mark decision.

The individual Intermittent histories are short: mean sequence length is about
`10.39`, median `6`, and the effective model window is at most `16` tokens
including the appended target. Plain per-window RevIN is consequently unstable
for one- or two-event contexts and degenerates to zero variance for one event.

## Problem

Add direct continuous magnitude prediction and test causal reversible
normalization without silently changing TitanTPP into an incomparable unmarked
regression model.

The design must preserve:

- the categorical marker probability head and marker CE likelihood;
- the continuous-time RMTPP likelihood;
- the meaning of `nll_marker`, `nll_time`, and `nll`;
- target-only fixed-split evaluation and held-out test discipline;
- one unambiguous quantity prediction source per run;
- causal input construction with no appended-target or padding leakage.

## Options Considered

### Option A: Mark/Time Heads Plus One Direct Magnitude Decoder

Keep the probabilistic marker and time heads. In direct-magnitude mode, use one
new continuous `log2(qty)` decoder as the only quantity prediction path.

Advantages:

- Preserves marked-TPP likelihood and paper-comparable NLL.
- Bypasses marker argmax for quantity reconstruction.
- Isolates direct regression and normalization effects.
- Requires no change to event-time inference or marker inference.

Disadvantages:

- Adds a decoder and continuous input feature.
- Marker and magnitude losses still share the encoder and may conflict.
- The magnitude output is deterministic and is not itself a probability
  density.

### Option B: Train Legacy Residual And Direct Magnitude Decoders Together

Keep both quantity decoders active and combine their losses or predictions.

Advantages:

- Retains the legacy path as an internal fallback.
- Could be used as an ensemble.

Disadvantages:

- Creates two competing quantity definitions.
- Makes quantity metrics and gradient routing ambiguous.
- Prevents a clean attribution to direct regression or RevIN.
- Adds dead or confounding parameters in ablation runs.

### Option C: Replace The Categorical Marker With A Continuous Mark Density

Remove the marker classifier and model `p(log2(qty) | history)` with a Gaussian,
Student-t, or mixture density.

Advantages:

- Removes discrete magnitude marks from the prediction task.
- Gives a fully probabilistic continuous-mark TPP.

Disadvantages:

- Changes the likelihood, model task, and baseline comparison.
- Requires density-scale Jacobian handling after reversible normalization.
- Invalidates direct comparison with current RMTPP/TitanTPP marker NLL.

### Option D: V5b Class-Prior Correction

Keep mark-plus-residual reconstruction and alter marker CE weighting or logits.

Advantages:

- Smallest implementation change.
- Directly addresses the mark 0/1 prior boundary.

Disadvantages:

- Keeps magnitude discretization and argmax reconstruction.
- Can trade mark-0 errors for mark-1 or tail errors without improving quantity.

## Decision

Select Option A for the first magnitude track. Option C is a later model family,
not an incremental TitanTPP enhancement. Option D remains the fallback if the
direct magnitude track fails. Option B is rejected.

The term *parallel* means that the marker, time, and magnitude tasks share the
Titan encoder. It does **not** mean that legacy mark-residual and direct
magnitude quantity decoders are active together.

### Post-Result Domain Reclassification: 2026-07-13

The first decision conflated the transform domain with the normalization method.
M0 reconstructs `z=log2(qty)` and uses fixed train-global statistics. It does not
compute per-instance context statistics and therefore is not a RevIN experiment.
The original M1-M4 definitions also operate on `z`, making them log-domain
normalization variants.

The current TitanTPP path instantiates `MemoryEncoder` directly. It does not call
the standalone Titan forecasting wrapper that owns `use_revin` and the
`RevIN(norm/denorm)` calls. A serialized `TitanConfig.use_revin=true` value is not
evidence that RevIN was active in M0.

Revised scope:

- retain M0 as a log-domain direct/global negative ablation;
- stop M0 matched multi-seed and the dependent log-domain M1-M4 branch;
- do not infer anything about raw-quantity RevIN performance;
- use the separate Q0 raw/global, Q1 raw/canonical-RevIN, and Q2 raw/shrinkage-
  RevIN successor contract, with a raw train-only audit before implementation;
- retain V5b as an independent fallback rather than the sole next path.

## Model Boundary Contract

```text
observed history
  mark_t, delta_t, scale_residual_t
                |
                v
      causal magnitude context
                |
                v
          Titan encoder
          /     |      \
         /      |       \
   mark head  time head  direct magnitude head
      CE       time NLL      normalized z_hat
                                |
                                v
                     z_hat = center + scale * z_hat_norm
                                |
                                v
                           qty_hat = 2^z_hat
```

Run-level quantity decoder modes:

| Mode | Quantity prediction | Marker/time heads |
| --- | --- | --- |
| `mark_residual` | predicted mark plus residual/value head | unchanged |
| `direct_log_qty` | direct denormalized `log2(qty)` head | unchanged |

Only one quantity decoder contributes to training and evaluation in a run.
`direct_log_qty` does not use predicted marks, mark probabilities, or the legacy
residual head to form `qty_hat`.

The first direct track uses:

```text
marker_loss_mode=ce
lambda_ordinal=0
train_loss_scope=target_only
```

V5a RPS, V5b prior correction, V3 experts, and gradient detachment are not
combined with this first experiment.

## Target And Input Contract

For every valid real event:

```text
z = mark + scale_residual = log2(demand_qty)
```

Requirements:

- `scale_base` must equal `2.0` for the first track.
- Reconstructed `z` and `demand_qty` must match within the fixed-split quantity
  contract tolerance before training.
- Non-positive or non-finite quantity targets fail before artifact creation.
- The appended next-event target is used for loss only.
- Target `z`, target residual, and target-derived statistics never enter the
  encoder input.
- Left padding is excluded from every count, mean, variance, and loss.

The first implementation supports `target_only` only. Supporting `all` would
require a distinct prefix statistic for every transition and is outside the
first activation.

## Magnitude Context Contract

A single stateless batch builder constructs a `MagnitudeContext` from marks,
residuals, mask, and train-only normalization metadata.

```text
MagnitudeContext:
  normalized_history: [B, L, 1]
  center:             [B, 1]
  scale:              [B, 1]
  context_count:      [B, 1]
  stat_features:      [B, 3]
```

The final valid token is treated as the appended target and removed from the
history mask before statistics or encoder features are built. Its normalized
input feature is zero. The same context object must be used by the encoder and
the magnitude head so training and evaluation cannot compute different
statistics.

## Normalization Variants

The earlier composite M3 is split so shrinkage and statistic context can be
attributed separately.

| Variant | Center/scale | Statistic context | Role |
| --- | --- | --- | --- |
| M0 | train-global | no | direct-regression baseline |
| M1 | per-series train-only, global fallback | no | fixed-series ablation |
| M2 | causal window RevIN | no | plain RevIN ablation |
| M3 | causal shrinkage RevIN | no | shrinkage effect |
| M4 | causal shrinkage RevIN | yes | primary candidate |

### M0: Global

Use global mean and population variance computed from train events only:

```text
center = mu_global
scale  = max(sigma_global, sigma_floor)
```

### M1: Per-Series

Use train-only statistics keyed by `oper_part_no`. A series with insufficient
training observations or no training statistics falls back to global values.
Validation or test targets never update the stored scaler.

### M2: Causal Window RevIN

Use only valid context events in the current sample:

```text
center = mean(z_history)
scale  = max(population_std(z_history), sigma_floor)
```

For one context event, the population variance is zero. M2 keeps only the
numeric floor and is expected to expose the short-history failure mode rather
than hide it.

### M3/M4: Causal Shrinkage RevIN

Let `n` be the number of valid history events and:

```text
alpha = n / (n + shrinkage_k)
```

Blend first and second moments rather than directly averaging standard
deviations:

```text
mu = alpha * mu_history + (1 - alpha) * mu_global

m2 = alpha * (var_history + mu_history^2)
   + (1 - alpha) * (var_global + mu_global^2)

var = max(m2 - mu^2, sigma_floor^2)
scale = sqrt(var)
```

This includes the between-mean contribution and remains finite for `n=1`.
`shrinkage_k` and `sigma_floor` are determined from train-only history and
variance diagnostics, then frozen before validation metrics are read.

M4 projects these head-side features:

```text
[center, log(scale), log1p(context_count)]
```

They are concatenated with the final context hidden state. The normalized
magnitude sequence enters the encoder, but the statistic features do not.

## Direct Magnitude Head Contract

```text
M0-M3:
  z_hat_norm = Linear(h_last)

M4:
  stat_emb   = stat_context_proj([center, log(scale), log1p(n)])
  z_hat_norm = magnitude_head(concat(h_last, stat_emb))

z_hat   = center + scale * z_hat_norm
qty_hat = exp2(z_hat)
```

The exponent input may be clamped only for numeric safety in quantity-loss and
metric reconstruction. Unclamped `z_hat` must remain available for log-scale
metrics and diagnostics. Clamp bounds are configuration and artifact identity.

## Loss And Likelihood Contract

The direct normalized target is:

```text
z_target_norm = (z_target - center) / scale
```

Losses:

```text
magnitude_loss = Huber(z_hat_norm, z_target_norm)

direct_qty_loss = Huber(
    qty_hat / qty_scale_value,
    qty_target / qty_scale_value
)

total_loss = marker_train_loss
           + lambda_dt * nll_time
           + lambda_magnitude * magnitude_loss
           + lambda_qty * direct_qty_loss
```

Initial coefficients:

```text
lambda_magnitude = 1.0
lambda_qty       = 0.25
```

These match the existing unit weight for value supervision and the existing
hybrid quantity coefficient. They are fixed for M0-M4 seed-42 screening.

Likelihood identity remains:

```text
nll_marker = raw categorical CE
nll_time   = continuous-time NLL
nll        = nll_marker + nll_time
```

`magnitude_loss` and `direct_qty_loss` do not enter `nll`. The legacy
`value_loss` is not reused or relabeled in direct mode; it is not applicable.

## Evaluation Contract

Direct mode exports explicit metrics without changing legacy meanings:

- `log_qty_mae` and `log_qty_rmse` in base-2 units;
- `magnitude_loss` in normalized target space;
- raw `qty_mae`, `qty_rmse`, and `qty_wape`;
- signed quantity bias;
- scale-wise quantity metrics;
- context-length metrics for `1`, `2-4`, `5-8`, and `9+` history events;
- marker NLL, marker accuracy, time NLL, and DT MAE safety metrics;
- `qty_decoder_mode` and `magnitude_norm_mode` in every result row.

`value_mae` continues to mean scale-residual error for legacy runs. It is `NaN`
or omitted for direct mode and is never overwritten with log-quantity MAE.

## Checkpoint Contract

- `best_val_nll` remains the primary paper-comparable checkpoint.
- M0-M4 promotion decisions use metrics at `best_val_nll`.
- `best_val_qty_mae` may be added as a diagnostic checkpoint to distinguish an
  under-trained magnitude head from an incompatible shared representation.
- A candidate that improves only at `best_val_qty_mae` does not pass the
  TitanTPP enhancement gate.
- `best_score` and `final` remain secondary diagnostics.

## Initial Acceptance Gate

All comparisons use seed 42, e50, fixed split, `small_lmm`, and the V2-matched
training budget before multi-seed promotion.

M0 direct-regression benefit versus V2 at `best_val_nll`:

- quantity MAE improves by at least `3%`;
- log-quantity MAE improves by at least `3%` after exporting the same metric for
  V2;
- no validation quantity bucket with share at least `5%` regresses by more than
  `5%`.

RevIN benefit:

- M3 or M4 improves overall quantity MAE and log-quantity MAE by at least `2%`
  versus M0;
- for context count at most four, both improve by at least `3%` versus M0;
- M4 must improve over M3 to attribute an additional benefit to statistic
  context. Otherwise M3 is preferred as the simpler candidate.

Marked-TPP safety versus V2:

- marker NLL regression at most `1%`;
- total NLL regression at most `0.5%`;
- time NLL regression at most `0.5%`;
- mark accuracy gap at least `-0.25%p`;
- DT MAE regression at most `2%`;
- every evaluated context-length bucket has finite predictions and loss.

An M0 failure stops the **log-domain M1-M4 branch defined in this ADR** because
those variants depend on the same log2 direct decoder. It does not reject a
raw-quantity RevIN family, which changes the input/target domain and requires a
new baseline and gate. If M0 passes but M3/M4 fail, direct log2 regression may
proceed without a RevIN claim. If M3/M4 pass, only the simplest passing candidate
proceeds to matched seeds `42,52,62`.

## Test Lock

- All variant and normalization decisions use validation only.
- Held-out test metric content is not read during M0-M4 seed-42 screening.
- The runner may generate test files, but existence checks are the only allowed
  access before the candidate and constants are frozen.
- Multi-seed validation freezes the candidate before held-out audit.
- A failed held-out audit rejects the candidate and does not trigger retuning.

## Configuration And Artifact Identity

Expected configuration fields:

```text
qty_decoder_mode: mark_residual | direct_log_qty
magnitude_norm_mode: global | per_series | causal_revin | shrinkage_revin
magnitude_use_stat_context: bool
magnitude_input_emb_dim: int
magnitude_stat_emb_dim: int
lambda_magnitude: float
shrinkage_k: float
magnitude_sigma_floor: float
magnitude_exp_clamp_min: float
magnitude_exp_clamp_max: float
```

Defaults preserve the legacy `mark_residual` path exactly. Direct mode fails
fast unless `scale_base=2`, `train_loss_scope=target_only`, and
`marker_loss_mode=ce` with `lambda_ordinal=0`.

Persist in manifest, checkpoint config, resume/cache identity, run path,
history, summaries, scale/context metrics, and reports:

- decoder and normalization modes;
- statistic-context flag;
- global train count, mean, variance, and source split;
- per-series statistic artifact hash for M1;
- shrinkage and numeric-stability constants;
- magnitude/quantity loss coefficients;
- quantity prediction source.

Suggested path identity:

```text
.../qtydecoder_direct_log_qty/magnorm_shrinkage_revin/statctx_on/k_<value>/...
```

## Implementation Boundary

Expected implementation surface:

- new `models/RMTPPs/magnitude_normalization.py` for context construction and
  reversible transforms;
- `models/RMTPPs/config.py` for backward-compatible fields;
- `models/RMTPPs/TitanTPP.py` for exclusive decoder construction and direct
  loss outputs;
- `data_loader/event_seq_data_module.py` only if stable series keys or metadata
  need to accompany `part_idx`;
- `utils/training.py` and `simple_lab_test/search/common/runner.py` for direct
  prediction routing and new metrics;
- unified CLI, model builder, manifest, cache, aggregation, report, and plotting
  paths;
- focused tests under `simple_lab_test/search/tests/`.

## Required Focused Tests

1. `mark + scale_residual` reconstructs the fixed-split log2 quantity target.
2. Appended target changes do not change normalized history, center, or scale.
3. Padding changes do not change statistics or predictions.
4. M2 is finite for one event and exposes the configured variance floor.
5. M3/M4 are finite and reversible for one event and constant histories.
6. Normalize then denormalize recovers valid history values within tolerance.
7. Global and per-series statistics use train rows only.
8. Missing per-series statistics fall back to global values.
9. Direct quantity predictions do not depend on predicted marks or legacy value
   head parameters.
10. Legacy decoder outputs, paths, and state loading are unchanged by defaults.
11. Marker/time likelihood values retain their existing definitions.
12. `marker_train_loss`, magnitude loss, and quantity loss are each composed
    exactly once.
13. M4 statistic features receive no target gradient or target-derived value.
14. Invalid mode combinations fail before run-directory reuse.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Target included in RevIN statistics | Direct leakage | Final-valid-token exclusion and mutation tests |
| Short context yields zero variance | Collapsed denormalization | M2 floor; M3/M4 train-global shrinkage |
| Global statistics use validation/test | Selection leakage | Train-only artifact with source split and hash |
| Two quantity decoders become active | Ambiguous metrics and gradients | Exclusive `qty_decoder_mode` fail-fast |
| Direct loss damages marker/time modeling | TPP regression | Shared-encoder safety gates at `best_val_nll` |
| Quantity exponent overflows | NaN/Inf | Explicit exp2 clamp identity and finite tests |
| M3/M4 improvement is due only to extra parameters | Wrong RevIN claim | M3 shrinkage-only versus M4 context comparison |
| Direct regressor is claimed as probabilistic mark model | Invalid likelihood claim | Keep marker CE; scope direct output as auxiliary quantity decoder |
| Per-series scaler fails on unseen series | Undefined normalization | Train-only map with global fallback |

## Non-Goals

- No removal of the marker head in the first track.
- No Gaussian, Student-t, or mixture continuous-mark likelihood yet.
- No simultaneous legacy and direct quantity decoder ensemble.
- No V5a RPS, V5b class-prior weighting, V3 expert, or detachment combination.
- No `train_loss_scope=all` until prefix-wise causal statistics are implemented.
- No test-driven selection of normalization mode or shrinkage constants.
- No claim that RevIN improves Titan memory until M0 isolates direct regression.

## Consequences

Positive:

- Quantity prediction no longer depends on an imbalanced marker argmax.
- Marked-TPP likelihood remains available and comparable.
- M0 separates direct-regression benefit from RevIN benefit.
- M3/M4 separate shrinkage from statistic-context benefit.
- Short-history stability becomes an explicit metric and test target.

Negative:

- The model has an additional task and shared-encoder gradient path.
- Evaluation and artifact schemas require explicit decoder routing.
- M0-M4 screening adds more runs than a single composite M3 experiment.
- A deterministic magnitude head is not a continuous mark probability model.

## Follow-Up

1. Keep M0 and its artifacts as a log-domain negative ablation.
2. Do not implement or run the current log-domain M1-M4 after the failed M0 gate.
3. Do not promote M0 to strict matched multi-seed.
4. Define raw-domain Q0/Q1/Q2 and a new acceptance gate before implementation.
5. Keep V5b class-prior correction as an independent fallback branch.

## Train-Only Audit Result: 2026-07-13

The normalization audit completed locally against
`sample_data/head_office/marked_target_with_split.parquet`. It reconstructed
the same `136,256` train targets and context-length distribution as
`RMTPPWeekLookbackDataset` with lookback 52 and maximum sequence length 16.
No validation or held-out test rows were read for constant selection.

Source and contract checks passed:

- decoded train rows/series: `159,643 / 23,387`;
- train targets: `136,256`;
- required nulls, duplicate part/sequence keys, non-positive quantities,
  decoded non-train rows, and non-train context events: zero;
- maximum `log2(qty)` versus `mark+scale_residual` error: zero;
- DataLoader target count and context-length distribution: exact match.

Short-history evidence:

- median/p95/max context count: `3 / 11 / 12`;
- one-event contexts: `22.66%`;
- contexts with at most two/four events: `41.50% / 67.63%`;
- zero-variance contexts: `35.23%`;
- train series with at most four events: `61.06%`;
- zero-variance train series: `38.61%`.

Scale and shift evidence:

- train global log2 mean/std: `1.2662 / 1.4535`;
- between-series level differences explain `73.23%` of total train variance;
- median/p95 absolute window-mean versus global-mean gap: `0.7662 / 2.6514`;
- `28.70%` of next targets fall outside their current history range;
- median/p95 within-window half-level shift: `0.5000 / 1.4534` log2 units;
- median/p95 per-series early/late shift: `0.2925 / 1.3838` log2 units.

The predeclared train-only stability gate selected:

```text
shrinkage_k = 4
magnitude_sigma_floor = 0.0014535461338152059
magnitude_exp_clamp_min = -2
magnitude_exp_clamp_max = 15
```

At `k=4`, median local weight is `0.4286`, one-event median scale is `1.3953`,
target absolute normalized residual p99 is `1.7773`, and only `0.0073%` exceed
three. This is the lowest p99 among candidates satisfying the scale and local
weight gates.

Pre-M0 design decision (superseded by the completed screening below):

- M0 was the prerequisite log-domain direct/global baseline.
- M1-M4 were dependent log-domain normalization variants; M2 was diagnostic and
  M3/M4 were the proposed shrinkage path under that original contract.
- The constants above remain an audit record of the train-only analysis. They do
  not activate M1-M4 after M0 failed and do not define a raw-quantity RevIN run.
- M0 plus the shared magnitude-context contract is implemented and passed the
  local synthetic model-test and focused regression suite (`58 passed`).
- The 5090 CUDA model-test passed after adding ai_env `nvidia/cu13/lib` to the
  runtime linker path, and the Instacart top-20 e1 data smoke passed.
- Intermittent M0 seed-42 e50 completed on 5090 with `SCREENING_SUCCESS`; the
  `best_val_nll` checkpoint was epoch `24`.
- Versus the frozen V2 validation-only reference, total/time NLL and raw quantity
  MAE improved `1.631%/2.162%/9.791%`, while marker NLL regressed `0.872%` within
  its safety allowance.
- Log2 quantity MAE regressed `9.700%`, mark accuracy fell `3.635%p`, and the
  dominant `1-9` bucket (`88.666%` share) quantity MAE regressed `8.623%`.
- M0 therefore failed three predeclared conditions: log-quantity improvement,
  mark-accuracy safety, and the share-at-least-5% bucket safety rule.
- M0 is not promoted to multi-seed and the current log-domain M1-M4 are closed
  without execution. Raw-domain RevIN remains untested and requires a separate
  Q0/Q1/Q2 contract; V5b remains an independent fallback.
- During artifact schema inspection, merged test columns in `leaderboard/runs.csv`
  were displayed but were not used in this predeclared validation decision. Future
  blind gates avoid merged run/report artifacts until the validation decision is
  recorded.

Audit artifacts:

```text
search_artifacts/model_enhancement_magnitude_revin_audit_0713
```
