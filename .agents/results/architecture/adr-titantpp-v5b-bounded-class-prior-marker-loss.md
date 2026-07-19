# ADR: TitanTPP V5b Bounded Class-Prior Marker Loss

- Date: 2026-07-19
- Status: Accepted design; implementation not started
- Scope: Intermittent-only TitanTPP marker-objective enhancement
- Baseline: V2 `small_lmm`
- Method: Evidence review followed by frozen acceptance contract

## Context

Intermittent next-event targets are imbalanced, but the relevant failure is not
an unconstrained rare-tail problem. The fixed train split contains `136,256`
targets and marks `0-2` account for `86.60%` of them. Marks `6-10` each have
less than `1%` support.

| Mark | Train count | Train share |
| ---: | ---: | ---: |
| 0 | 53,469 | 39.242% |
| 1 | 41,110 | 30.171% |
| 2 | 23,416 | 17.185% |
| 3 | 10,609 | 7.786% |
| 4 | 4,282 | 3.143% |
| 5 | 1,847 | 1.356% |
| 6-10 | 1,523 | 1.118% combined |

The validation and test priors remain close to train: total-variation distance
is `0.0233` and `0.0253`, respectively. This rules out a large deployment-prior
shift as the motivation for V5b.

V2 is the active Intermittent baseline. At its seed-42 e50 validation
`best_val_nll` checkpoint it predicts mark 0 for `45.030%` of targets, versus a
true share of `41.180%`. Mark-0 recall is `75.543%`, mark-1 recall is `49.616%`,
and `40.289%` of true mark-1 targets are predicted as mark 0.

V5a shows why a broad auxiliary is unsafe. It increased mark-0 prediction share
to `58.750%` and mark-0 recall to `86.462%`, but mark-1 recall collapsed to
`24.664%`. Overall accuracy fell by `2.430%p`. V5a is rejected and is not a
component of V5b.

These observations support a small optimization correction around supported
classes, with explicit protection against rare-tail amplification. They do not
show that ordinary CE is statistically wrong for the observed deployment
prior.

## Problem

Test whether a bounded train-only class-prior weight can reduce the high-support
mark `1 -> 0` error without changing:

- TitanTPP parameters or inference;
- the reported ordinary marker CE and TPP likelihood;
- the time, value, quantity, memory, or input paths;
- the fixed split, target-only supervision, or checkpoint rule;
- the active Intermittent baseline unless a strict matched gate passes.

## Scope

V5b applies only to:

- dataset: `intermittent`;
- model: TitanTPP V2 `small_lmm`;
- objective: marker training CE;
- prior source: exact fixed-split train targets emitted by
  `RMTPPWeekLookbackDataset(target_splits={"train"})`;
- supported classes: train prior at least `1%`, which freezes marks `0-5` for
  the current dataset.

V5b does not apply to:

- marks `6-10`, whose weights remain exactly `1.0`;
- PAD, padding transitions, context-only rows, validation, or test targets;
- Taxi V3b, Instacart quality comparison, RMTPP, or THP;
- V3 value experts, V4 time heads, V5a RPS, direct magnitude, or RevIN paths;
- oversampling, focal loss, label smoothing, or inference-time prior shifts.

An Instacart top-20 e1 run is allowed later only as an integration smoke. It is
not V5b quality evidence.

## Options Considered

### Option A: Raw Inverse-Frequency Weighted CE

This directly raises low-support classes but gives mark 10 approximately
`938` times the raw weight of mark 0. A few tail targets could dominate the
marker gradient. Rejected.

### Option B: Effective-Number Or Unbounded Power Weights

These are smoother than raw inverse frequency, but without a support exclusion
and hard cap they still spend most of the intervention on marks `6-10`, while
the observed regression is the mark `0/1` boundary. Rejected for the first
V5b screen.

### Option C: Train-Time Or Inference-Time Logit Adjustment

Logit adjustment deliberately changes the class-prior semantics of the learned
or decoded posterior. Train, validation, and test priors are already close, so
there is no observed prior shift that justifies decoding toward a balanced
prior. It would also make raw calibration and marker-NLL interpretation harder.
Rejected for V5b.

### Option D: Support-Aware Bounded Prior-Weighted CE

Use smoothed train priors, square-root inverse-prior ratios, a neutral rare
tail, a hard weight cap, and unit expected weight. This is selected because it
targets optimization pressure without changing logits or inference.

## Decision

V5b is Option D, exposed as:

```text
marker_loss_mode=ce_bounded_prior
```

It branches directly from V2. V2 and V5b have identical model parameters,
initialization, forward logits, decoding, and all non-marker settings.

### Train-Only Prior Contract

Let `C=11` be the real mark count, `n_k` the exact train target count, and
`N=sum_k n_k`. Laplace smoothing is frozen at `alpha=1`:

```text
pi_k = (n_k + alpha) / (N + alpha * C)
```

Only classes with `pi_k >= 0.01` are eligible for correction. For an eligible
class, the unnormalized factor is square-root inverse prior:

```text
u_k = pi_k ** -0.5
```

Rare-tail classes receive `w_k=1`. Eligible weights use:

```text
w_k = clip(c * u_k, 0.75, 1.25)
```

The positive scalar `c` is solved by monotonic bisection so the smoothed
train-prior expectation remains one:

```text
sum_k pi_k * w_k = 1
```

This preserves the expected marker-loss scale relative to time and quantity
losses. For the audited train counts, `c=0.527704246214` and the frozen expected
weights are:

| Mark | Eligible | Weight |
| ---: | --- | ---: |
| 0 | yes | 0.842424 |
| 1 | yes | 0.960742 |
| 2 | yes | 1.250000 |
| 3 | yes | 1.250000 |
| 4 | yes | 1.250000 |
| 5 | yes | 1.250000 |
| 6-10 | no | 1.000000 |

The absolute scale is normalized; the relevant mark-1 to mark-0 weight ratio is
`1.140450`. Counts, priors, eligibility, scalar `c`, weights, and their source
split must be persisted in the manifest and checkpoint identity.

### Loss And Metric Contract

For each valid transition, compute ordinary full-logit CE first:

```text
ce_i = CE(logits_i, target_i, reduction="none")
nll_marker = sum(mask_i * ce_i) / sum(mask_i)

prior_weighted_marker_loss =
    sum(mask_i * w[target_i] * ce_i) / sum(mask_i)
```

The denominator remains the valid transition count. The global unit-expectation
normalization, not batch-local weight normalization, preserves expected scale.

```text
V2 marker_train_loss  = nll_marker
V5b marker_train_loss = prior_weighted_marker_loss
```

The project likelihood names do not change:

```text
nll_marker = ordinary unweighted categorical CE
nll_time   = existing continuous-time NLL
nll        = nll_marker + nll_time
```

`prior_weighted_marker_loss` and the effective class weights are reported
separately. The weighted loss does not enter reported NLL or checkpoint
selection.

### Train And Inference Logit Semantics

- Training computes both losses from the same unadjusted logits.
- V5b scales per-target CE only; it never adds or subtracts prior logits.
- Forward, argmax, sampling, validation, and test use unadjusted logits.
- No temperature scaling or post-hoc calibration is allowed before the V5b
  acceptance decision.
- Primary checkpoint remains the minimum ordinary validation `nll`.

This preserves the original posterior interface while acknowledging that
weighted training can still alter calibration through optimization. Therefore
calibration is an explicit safety gate rather than an assumed invariant.

### Calibration Contract

Evaluation adds metrics over real-mark probabilities
`softmax(logits[..., :pad_id])`:

- multiclass Brier score;
- top-label ECE with 15 fixed equal-width confidence bins;
- mean confidence;
- predicted share for marks `6-10`.

The bin boundaries are fixed before validation and empty bins contribute zero.
Calibration is reported from raw V2/V5b probabilities without fitting a
calibrator.

### Configuration And Artifact Identity

Expected implementation fields:

```text
marker_loss_mode: ce | ce_rps | ce_bounded_prior
marker_prior_smoothing: 1.0
marker_prior_power: 0.5
marker_prior_min_support: 0.01
marker_weight_min: 0.75
marker_weight_max: 1.25
marker_class_counts: train-derived immutable vector
marker_class_weights: train-derived immutable vector
```

The default remains `marker_loss_mode=ce`; all new scalar defaults are inert in
that mode. V5b gets a distinct run-path identity. Resume must fail on any prior
count, weight, split, or option mismatch.

## Acceptance Gates

### G0: Focused Contract Tests

1. Prior counts exactly match the `136,256` audited train targets and never
   read validation/test targets.
2. Laplace priors, eligibility, bisection constant, and expected weights match
   the frozen values above.
3. Every weight is finite and in `[0.75,1.25]`; marks `6-10` and PAD handling
   follow the scope contract.
4. `sum(pi * w)=1` within numerical tolerance.
5. V2 default loss, gradients, path, state dict, and outputs remain unchanged.
6. V2 and V5b have identical parameters and epoch-zero logits.
7. `nll_marker` remains ordinary CE while `marker_train_loss` equals bounded
   weighted CE exactly once in all quantity loss modes.
8. Weighted CE updates the marker head and shared Titan encoder, but not time or
   value heads in isolation.
9. `all` and `target_only` use the same transition mask for weighted and
   ordinary CE.
10. Invalid modes, stale priors, unsupported datasets, and resume mismatches
    fail before training.
11. Brier and 15-bin ECE match hand-computed fixtures.

### G1: 5080 Integration Smoke

1. Run local focused CPU tests.
2. Run 5080 CUDA TitanTPP `small_lmm` model-test.
3. Run an Instacart top-20 fixed-split e1 integration smoke only.
4. Require finite ordinary CE, weighted CE, time, value, quantity, total loss,
   Brier, and ECE.
5. Require complete manifest/path/history/confusion/calibration artifacts and no
   NaN, Traceback, or runtime error.

### G2: Strict Seed-42 Validation-Only Screen

Run fresh matched V2 and V5b on Intermittent with `small_lmm`, seed `42`, e50,
fixed split, strict reproducibility, target-only scope, residual input, hybrid
quantity loss, and ordinary `best_val_nll` checkpoint selection.

V5b passes only when all groups pass.

Primary boundary benefit:

- mark-1 recall improves by at least `1.00%p`;
- true mark `1 -> 0` confusion rate falls by at least `2.00%p`;
- balanced accuracy or macro F1 improves by at least `0.50%p`, and the other is
  no worse than `-0.25%p`.

Classification and tail safety:

- overall mark-accuracy gap is at least `-0.25%p`;
- mark-0 recall gap is at least `-2.00%p`;
- mark MAE regression is at most `1%`;
- adjacent-accuracy gap is at least `-0.25%p`;
- predicted marks `6-10` share increases by no more than `0.50%p` versus V2.

Likelihood, calibration, and task safety:

- ordinary marker NLL regression is at most `0.5%`;
- total and time NLL regression are each at most `0.5%`;
- multiclass Brier regression is at most `1%`;
- ECE increase is at most `0.50%p`;
- quantity MAE and value MAE regression are each at most `2%`;
- no quantity bucket with validation share at least `5%` regresses by more than
  `5%`.

There is no gamma, cap, support-threshold, or logit-adjustment branch after G2.
If G2 fails, V5b closes and V2 remains active. A changed weighting contract is
a new hypothesis, not a continuation selected from the same validation result.

### G3: Strict Matched Multi-Seed Validation

If G2 passes, rerun fresh V2/V5b seeds `42,52,62` with the same e50 budget.
Promotion to held-out audit requires:

- all six runs complete;
- mean G2 benefit and safety thresholds pass;
- mark-1 recall and `1 -> 0` rate improve in at least `2/3` matched seeds;
- balanced accuracy or macro F1 improves in at least `2/3` seeds;
- mean accuracy gap is at least `-0.25%p` and no seed is below `-0.75%p`;
- mean likelihood, calibration, quantity, value, and tail safety gates pass.

### G4: Frozen Held-Out Test Audit

Only after G3 freezes V5b, read held-out test artifacts in protocol order. Apply
the same mean thresholds and `2/3` direction rules against fresh matched V2.
Failure keeps V2 as the Intermittent incumbent and does not reopen weight tuning
from held-out evidence.

## Consequences

Positive:

- directly tests whether moderate class competition, rather than a new head or
  ordinal loss, can recover mark 1;
- prevents very rare marks from receiving amplified gradients;
- preserves ordinary marker NLL, inference, parameter count, and checkpoint
  semantics;
- keeps expected marker-loss scale aligned with V2;
- makes calibration risk measurable before promotion.

Negative:

- weighted CE is not maximum likelihood under the observed class prior;
- the support cutoff and caps are Intermittent-specific frozen constants;
- even bounded weighting can move errors from `1 -> 0` to `1 -> 2`;
- a positive single-seed result is not sufficient for promotion;
- the first V5b claim is TitanTPP system-level, not backbone-only, unless the
  same objective is later ported to RMTPP and THP.

## Non-Goals

- No claim that class imbalance is the sole root cause of V3/V5a failures.
- No balanced-test or uniform-prior decoding objective.
- No rare-tail recall optimization in the first V5b screen.
- No logit adjustment, focal loss, resampling, or broad weight sweep.
- No combination with V5a, V3, V4, Q, RevIN, V6, or V7.
- No held-out test use before strict multi-seed freeze.
- No active-model status change from this design document alone.

## Current Decision

- V5b contract: accepted and frozen.
- Implementation and model-quality evidence: not started.
- Registry status: `SELECTED_HYPOTHESIS`.
- Intermittent incumbent: V2 `small_lmm` remains unchanged.
- Next action: implement the train-prior helper, weighted marker loss,
  calibration metrics, artifact identity, and focused tests.
