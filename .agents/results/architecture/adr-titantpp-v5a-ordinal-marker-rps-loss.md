# ADR: TitanTPP V5a CE Plus Ranked Probability Score Marker Objective

- Date: 2026-07-12
- Status: Accepted as implemented; Intermittent V5a seed-42 candidate rejected
- Scope: Intermittent-first TitanTPP ordinal marker enhancement
- Method: Design-Twice followed by ADR

## Context

The Intermittent fixed-split diagnostic reconciled every V2/V3a/V3b/V3c
confusion matrix with the actual next-event targets and the reported leaderboard
accuracy. The held-out test distribution is concentrated in marks 0-2
(`87.39%`), but all variants use the same targets and split drift is small.

The differentiating V3c failure is concentrated at the high-support mark 0/1
boundary:

- V3c `1 -> 0` confusion is `56.16%`, versus `40.11%` for V2.
- The mark-1 recall loss contributes `-5.316%p` to V3c accuracy relative to V2.
- The mark-0 recall gain offsets only `+3.916%p`.
- Adjacent classes account for `83.11%` of V3c errors and `86.66%` of V2 errors.

This evidence supports an order-aware marker objective. It does not support raw
inverse-frequency weighting as the first intervention: test support falls to
`16` targets for mark 10, so inverse-frequency weights could let a few tail
events dominate the marker gradient.

The current TitanTPP marker path is ordinary categorical cross-entropy:

```text
logits = mark_head(h_j)
nll_marker = CE(logits, y_next)
nll = nll_marker + nll_time
```

`nll_marker` and `nll` are used for likelihood reporting and validation-NLL
checkpoint selection. V5 must not silently redefine either metric.

## Problem

Add ordinal information to marker training so that predictions farther from the
true log-scale mark incur a larger penalty, while preserving:

- the existing categorical probability head;
- the probabilistic marker NLL reported by the project;
- V2 model parameters, forward inference, and quantity/time paths;
- validation-selected checkpoint comparability;
- backward-compatible default behavior and artifact identity.

The first experiment must isolate the ordinal objective. It must not combine
ordinal supervision with class reweighting, logit adjustment, a new marker head,
or a failed V3 value-head route.

## Constraints And Quality Attributes

- Intermittent V5 branches from the confirmed V2 baseline, not V3a/V3b/V3c.
- V2 and V5a have identical parameters, initialization, forward logits, and
  inference for identical weights and inputs.
- Padding is not an ordinal class and must be excluded from the ordinal term.
- Existing CE continues to include the full marker head and suppress PAD
  probability on valid targets.
- `nll_marker` remains pure CE and `nll` remains CE plus time NLL.
- The ordinal term uses the same transition mask and `loss_scope` as CE.
- No dataset, mark definition, lookback, maximum sequence length, time head,
  value head, Titan memory, optimizer, or checkpoint rule changes.
- The first coefficient is fixed before reading held-out test results.
- Validation is used for coefficient/candidate decisions; held-out test is an
  audit after the V5a contract is frozen.
- Default `marker_loss_mode=ce` must reproduce all existing runs and paths.

## Options Considered

### Option A: CE Plus Normalized Ranked Probability Score

Use the existing real-mark probabilities and compare their cumulative
distribution with the cumulative one-hot target.

Advantages:

- Encodes the complete class order without adding parameters.
- A deterministic error receives a cost proportional to ordinal distance.
- Retains CE as the primary categorical likelihood.
- Uses a bounded, normalized auxiliary in `[0, 1]`.
- Keeps inference, argmax decoding, and parameter count unchanged.
- Gives a direct probability-distribution diagnostic at validation/test time.

Disadvantages:

- Adds one coefficient and `O(C)` cumulative operations per valid transition.
- Does not directly correct class priors or guarantee better rare-class recall.
- A small coefficient may have no visible effect; a large coefficient may hurt
  CE calibration or high-support class accuracy.

### Option B: CE Plus Expected Absolute Mark Distance

Use `sum_c p(c) * abs(c - y)` as the auxiliary.

Advantages:

- Simple and directly aligned with mark MAE.
- No parameters or inference changes.

Disadvantages:

- By itself it is not a proper probability scoring rule and tends to emphasize
  a point estimate rather than the full ordered distribution.
- Its scale grows with the number of classes unless separately normalized.
- It can encourage probability concentration around a median-like class.

### Option C: Cumulative Ordinal Head Such As CORAL

Replace or augment the categorical head with `C-1` threshold classifiers.

Advantages:

- Makes ordinal thresholds explicit.
- Can yield interpretable cumulative exceedance probabilities.

Disadvantages:

- Changes parameters, decoding, checkpoint compatibility, and inference.
- Requires monotonicity handling and a new conversion to categorical
  probabilities for the TPP marker likelihood.
- Confounds the first loss-only diagnostic.

### Option D: Class-Balanced CE, Focal Loss, Or Logit Adjustment

Correct class frequency through weights, focusing, or prior-adjusted logits.

Advantages:

- Directly targets class imbalance.
- May improve macro recall for low-support marks.

Disadvantages:

- Does not encode that mark 1 is closer to mark 0 than mark 10.
- Raw weights are unstable for the extreme Intermittent tail.
- Changes class-prior behavior and would confound the ordinal hypothesis.
- Logit adjustment affects inference unless train/test handling is specified.

## Decision

Use Option A for V5a: retain categorical CE and add a normalized Ranked
Probability Score (RPS) auxiliary computed from the same marker logits.

V5a is an objective-only branch from V2:

| Variant | Value head | Quantity gate gradient | Value encoder gradient | Marker objective |
| --- | --- | --- | --- | --- |
| V2 | `shared` | `coupled` | `coupled` | `ce` |
| V5a | `shared` | `coupled` | `coupled` | `ce_rps` |

V5a does not inherit V3 mark-conditioned experts or either V3 detachment.

### Mathematical Contract

Let `C = num_marks - 1` be the number of real ordered marks. PAD is index `C`.
For a valid transition with real logits `z_0, ..., z_(C-1)` and target `y`:

```text
p_c = softmax(z_real)_c
F_k = sum_{c=0..k} p_c                  for k = 0..C-2
O_k = 1[y <= k]

RPS(p, y) = (1 / (C - 1)) * sum_{k=0..C-2} (F_k - O_k)^2
```

The batch ordinal loss uses the existing transition mask:

```text
ordinal_marker_loss =
    sum(step_mask * RPS) / max(sum(step_mask), 1)
```

For deterministic probability at predicted class `j`, normalized RPS becomes:

```text
RPS(one_hot(j), y) = abs(j - y) / (C - 1)
```

Therefore an adjacent mistake is cheaper than a distance-two or tail mistake.
For `C=1`, the implementation returns scalar zero because no ordinal threshold
exists.

### Training Objective Contract

The project-standard training composer adds the auxiliary exactly once:

```text
marker_train_loss = nll_marker + lambda_ordinal * ordinal_marker_loss

residual_only:
    loss = marker_train_loss
         + lambda_value * value_loss
         + lambda_dt * nll_time

hybrid:
    loss = marker_train_loss
         + lambda_value * value_loss
         + lambda_dt * nll_time
         + lambda_qty * qty_loss

qty_only:
    loss = marker_train_loss
         + lambda_dt * nll_time
         + lambda_qty * qty_loss
```

Metric identity remains:

```text
nll_marker = categorical CE only
nll_time   = continuous-time NLL only
nll        = nll_marker + nll_time
```

`ordinal_marker_loss` is always available as a diagnostic, including for V2,
but contributes to gradients only when `marker_loss_mode=ce_rps`.

### Configuration Contract

Add backward-compatible fields:

```text
marker_loss_mode: ce | ce_rps       # default: ce
lambda_ordinal: float               # default: 0.0
```

Initial V5a setting:

```text
marker_loss_mode=ce_rps
lambda_ordinal=0.10
```

RPS is normalized to `[0, 1]`; `0.10` therefore bounds its direct scalar
contribution at `0.10` per averaged objective, while Intermittent marker CE is
approximately `1.0`. This is a fixed first screening coefficient, not a broad
hyperparameter sweep.

Fail-fast rules:

- `marker_loss_mode=ce` requires `lambda_ordinal=0`.
- `marker_loss_mode=ce_rps` requires `lambda_ordinal>0` for experiment CLI runs.
- `lambda_ordinal` must be finite and non-negative.
- Initial V5a screening supports TitanTPP-only runs.
- Invalid combinations fail before run-directory or checkpoint reuse.

### PAD And Mask Contract

- CE keeps the current full logits, including the PAD output.
- RPS uses `softmax(logits[..., :pad_id])`, renormalized over real marks.
- PAD has no RPS threshold and receives no direct RPS gradient.
- Invalid/padded targets are clamped only for safe tensor construction and then
  removed by the same `step_mask` used by CE.
- `target_only` and `all` use exactly the same transition selection for CE and
  RPS.

### Output And Metric Contract

`model.nll(...)` adds:

```text
ordinal_marker_loss
marker_train_loss
```

The following names stay unchanged and retain their old meaning:

```text
nll
nll_marker
nll_time
value_loss
qty_loss
steps
```

Validation/test reporting adds:

- `val_ordinal_marker_loss`
- `mark_balanced_accuracy`
- `mark_macro_f1`
- `mark_mae`
- `mark_adjacent_accuracy` (`abs(pred - true) <= 1`)
- per-class support, recall, precision, and F1 in a separate table
- mark-0 prediction share and mark `0/1` recall in the gate summary

`adjacent_share_of_errors` remains a diagnostic, not a hard gate. Its
denominator changes when total errors change, so a larger value is not always
worse. `mark_mae` and `mark_adjacent_accuracy` are the monotonic ordinal gates.

### Checkpoint And Test-Lock Contract

- Primary checkpoint remains `best_val_nll` using CE plus time NLL.
- RPS does not enter checkpoint selection in V5a.
- Lambda decisions use validation only.
- Held-out test artifacts may be generated by the current runner, but they are
  not read until the V5a coefficient and multi-seed candidate are frozen.
- A failed held-out audit rejects V5a; it does not trigger another V5a lambda
  adjustment from the same test evidence.

### Artifact Identity Contract

Persist `marker_loss_mode` and `lambda_ordinal` in:

- experiment manifest and effective model config;
- checkpoint metadata and resume/cache identity;
- run path, summary, history, validation/test rows;
- scale-wise and confusion metadata;
- model-test output and report grouping.

V5a uses a distinct path segment:

```text
.../markloss_ce_rps/lambdaord_0p1/...
```

Legacy CE paths remain unchanged.

### Implementation Boundary

Expected implementation surface:

- new shared helper: `models/RMTPPs/marker_losses.py`
- `models/RMTPPs/config.py`
- `models/RMTPPs/TitanTPP.py`
- `simple_lab_test/search/common/configs.py`
- `simple_lab_test/search/common/models.py`
- `simple_lab_test/search/common/runner.py`
- `simple_lab_test/search/common/modes/model_test.py`
- `simple_lab_test/search/tpp_experiment.py`
- focused tests under `simple_lab_test/search/tests/`

The helper is model-agnostic, but V5a activation is TitanTPP-only for the first
screening. A final claim about backbone superiority requires either applying the
same CE+RPS objective to RMTPP/THP or describing V5a as a system-level TitanTPP
enhancement rather than an encoder-only improvement.

## Consequences

Positive:

- Adds ordinal supervision without adding model parameters or inference cost.
- Keeps categorical NLL and checkpoint semantics comparable with V2.
- Directly reflects the adjacent-error pattern found in Intermittent.
- Separates order-aware learning from prior/class-frequency correction.
- Gives a reusable ordinal diagnostic for both V2 and V5a.

Negative:

- Adds one training coefficient and artifact dimension.
- RPS may improve mark distance while leaving argmax accuracy unchanged.
- It does not directly solve low-support class calibration.
- Validation/test evaluators and leaderboard schemas need new marker metrics.
- The current fixed-split runner produces test artifacts during screening, so
  test-lock compliance initially depends on not reading those files.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| PAD is treated as the largest ordinal class | Meaningless distance gradients | Slice and renormalize real logits; retain full-logit CE |
| RPS replaces CE in `nll_marker` | Likelihood and checkpoint results become incomparable | Keep CE names unchanged; return RPS separately |
| Auxiliary is added twice | Wrong effective coefficient | Add only in the shared training composer; exact objective tests |
| Mask differs between CE and RPS | Weekly/all versus target-only mismatch | Reuse the same `step_mask`; focused mask tests |
| Lambda is too weak | No ordinal behavior change | One validation-only increase to `0.20` if all safety gates pass |
| Lambda is too strong | Accuracy or CE NLL regression | One validation-only decrease to `0.05` if ordinal benefit appears but safety fails |
| Tail imbalance remains | Macro metrics do not improve | Reserve capped prior correction as separate V5b ablation |
| Test informs lambda selection | Held-out leakage | Freeze lambda on validation; do not inspect test before multi-seed freeze |
| V5 is claimed as Titan-backbone-only gain | Unfair objective comparison | Port the same loss to baselines or scope the paper claim explicitly |

## Non-Goals

- No inverse-frequency or effective-number class weights in V5a.
- No focal loss, logit adjustment, label smoothing, or prior correction.
- No CORAL/cumulative marker head or parameter change.
- No V3 mark-conditioned value experts or gradient detachment.
- No change to time likelihood, value/quantity objective, Titan memory, input
  features, lookback, split, or checkpoint selection.
- No broad lambda sweep.
- No held-out test-driven coefficient selection.
- No Taxi V3b or V4 time-head decision change.

## Follow-Up Validation

### G0: Focused Contract Tests

1. RPS is zero for a correct deterministic prediction.
2. Deterministic adjacent and distant errors equal `1/(C-1)` and
   `distance/(C-1)` within tolerance.
3. RPS is finite and bounded in `[0, 1]`.
4. PAD-logit changes do not change RPS but do change full categorical CE.
5. `all` and `target_only` apply identical CE/RPS transition masks.
6. Default CE objective, gradients, run paths, and state dictionaries are
   unchanged.
7. V2 and V5a have identical parameter keys, initialization, and forward
   outputs before training.
8. Isolated RPS gives gradients to the mark head and Titan encoder, but not the
   time or value heads.
9. The composed objective contains exactly one weighted RPS term in every
   quantity loss mode.
10. Invalid CLI/config combinations fail before artifact reuse.

### G1: Integration Smoke

All execution uses 5090 until the user changes the server policy.

1. Local focused CPU tests.
2. 5090 CUDA `small_lmm` model-test with `ce_rps`, `lambda_ordinal=0.10`.
3. 5090 Instacart top-20 e1 integration smoke.
4. Require finite CE, RPS, time, value, quantity, and full loss values.
5. Require manifest/path/report identity and no NaN, Traceback, or ERROR.

### G2: Seed-42 Validation-Only Screening

Use Intermittent V2/V5a with matched `small_lmm`, fixed split, seed 42, e50,
learning rate, batch size, lookback, `max_seq_len`, residual input, hybrid
quantity objective, target-only scope, and `best_val_nll` checkpoint.

The current V2 seed-42 validation reference is:

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

V5a seed-42 passes only when all groups pass:

Ordinal benefit:

- normalized validation RPS improves by at least `1%` relative to reevaluated V2;
- validation mark MAE improves by at least `1%`;
- either balanced accuracy or macro F1 improves by at least `0.50%p`, and the
  other is no worse than `-0.25%p`.

Classification safety:

- mark accuracy gap is at least `-0.25%p`;
- mark-1 recall gap is at least `-1.00%p`;
- mark-0 recall gap is at least `-2.00%p`;
- adjacent accuracy gap is at least `-0.25%p`.

Likelihood and task safety:

- marker NLL regression is at most `1%`;
- total NLL regression is at most `0.5%`;
- time NLL regression is at most `0.5%`;
- quantity MAE and value MAE regression are each at most `2%`;
- no quantity bucket with validation share at least `5%` regresses by more than
  `5%`.

Lambda branch, using validation only:

- All groups pass at `0.10`: freeze V5a coefficient and continue.
- Safety passes but ordinal benefit fails: allow one `0.20` screening.
- Ordinal benefit passes but a classification/likelihood safety gate fails:
  allow one `0.05` screening.
- Both benefit and safety fail, or the one allowed adjustment fails: stop V5a.

No held-out test result is read during this branch.

### G3: Strict Matched-Budget Multi-Seed Validation

Before final confirmation, produce matched V2 and V5a e50 runs for seeds
`42,52,62`. V2 seeds 52 and 62 are not currently available under the strict
Intermittent e50 budget and must be run for a fair comparison.

Promotion requires:

- `3/3` runs complete without runtime or artifact errors;
- mean validation RPS and mark MAE each improve by at least `1%`;
- both improve in at least `2/3` seed-matched comparisons;
- mean mark accuracy gap is at least `-0.25%p`, with no seed below `-0.75%p`;
- mean marker NLL regression is at most `1%`;
- the balanced-accuracy/macro-F1 and mark-0/1 recall safety rules from G2 hold
  on the seed mean;
- mean quantity/value/time safety rules from G2 hold;
- the coefficient and all model/data settings are frozen before test review.

### G4: Frozen Held-Out Test Audit

After G3 freezes the candidate, read held-out test artifacts in protocol order.
Use the same ordinal-benefit, classification, likelihood, and task-safety
thresholds against strict matched V2 e50 seeds.

- Require mean RPS and mark MAE improvement of at least `1%` and seed-matched
  improvement in at least `2/3` seeds.
- Require mean mark accuracy gap at least `-0.25%p` and no seed below `-0.75%p`.
- Require mean marker NLL regression at most `1%` and total/time NLL safety.
- Require mean quantity/value regression at most `2%` and dominant-bucket
  regression at most `5%`.
- Report balanced accuracy, macro F1, mark-0/1 recall, prediction share, and
  confusion even when the gate fails.

If G4 fails, V2 remains the Intermittent baseline. Do not retune V5a from the
same held-out evidence. A later prior-correction V5b must be a separately
specified experiment.

## Assumptions

- Intermittent marks are ordered log2 magnitude classes and PAD is the final
  non-semantic embedding/head index.
- The existing V2 seed-42 e50 validation artifact is the correct first matched
  reference.
- A normalized RPS coefficient of `0.10` is small enough to behave as an
  auxiliary rather than replace CE; this remains a validation-tested assumption.
- Mark MAE and adjacent accuracy are more reliable ordinal gates than adjacent
  share among errors.
- Single-seed screening is only a resource gate, not final evidence.

## Implementation Status: 2026-07-12

The local implementation now follows this ADR:

- `models/RMTPPs/marker_losses.py` implements normalized RPS over real marks
  only, including masked averaging and the one-real-class zero case.
- `RMTPPConfig`, `ExperimentConfig`, the unified CLI, model construction,
  manifests, checkpoints, cache identity, run paths, histories, and report
  grouping carry `marker_loss_mode` and `lambda_ordinal`.
- `TitanTPP.nll(...)` keeps `nll_marker` as categorical CE and `nll` as CE plus
  time NLL. It returns `ordinal_marker_loss` and `marker_train_loss` separately.
- The shared loss composer uses `marker_train_loss` once in `residual_only`,
  `hybrid`, and `qty_only` modes.
- Validation/test evaluation now exports normalized RPS, balanced accuracy,
  macro F1, mark MAE, adjacent accuracy, mark-0 prediction share, mark-0/1
  recall, and per-class support/precision/recall/F1.
- V5a uses the distinct `markloss_ce_rps/lambdaord_0p1` path while the default
  V2 `ce` path is unchanged.

Local verification completed:

- 20 V5a focused tests passed, covering the RPS formula, PAD exclusion,
  transition masks, V2/V5a equivalence, gradient routing, single loss
  composition, invalid configuration, artifact identity, legacy aggregation,
  and learning-curve output.
- 18 existing V3/V3b/V3c focused tests and 4 Intermittent diagnostic tests
  passed without regression.
- Default CPU model-test passed for RMTPP, TitanTPP, and THP.
- V5a CPU `small_lmm` model-test passed with finite marker CE `2.457612`, RPS
  `0.194920`, weighted marker train loss `2.477104`, and total NLL `4.772340`.

5090 integration verification completed on 2026-07-12:

- CUDA `small_lmm` model-test passed with finite CE `2.500342`, normalized RPS
  `0.185512`, marker train loss `2.518893`, and total NLL `4.944328`.
- Instacart top-20 fixed-split e1 completed with no runtime or artifact error.
- Manifest, checkpoints, validation/test summaries, histories, scale-wise,
  confusion, per-class metrics, report, and plots were generated under the
  distinct V5a path.
- All epoch-1 validation/test predictions were mark 3. This is treated as an
  untrained smoke state, not evidence for or against ordinal benefit.

The Intermittent seed-42 e50 screening ran on 5090 from `2026-07-12 21:16:02
KST` to `2026-07-12 21:22:38 KST` and completed all 50 epochs without NaN,
Traceback, runtime error, or failed artifact export. The primary
`best_val_nll` checkpoint is epoch 30.

The frozen V2 and V5a validation-only comparison is:

| Metric | V2 | V5a | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Normalized RPS | `0.035283` | `0.035371` | `+0.251%` regression | Fail |
| Mark MAE | `0.487411` | `0.527028` | `+8.128%` regression | Fail |
| Balanced accuracy | `42.664%` | `41.667%` | `-0.997%p` | Fail |
| Macro F1 | `43.302%` | `41.163%` | `-2.139%p` | Fail |
| Mark accuracy | `57.249%` | `54.820%` | `-2.430%p` | Fail |
| Mark-0 recall | `75.543%` | `86.462%` | `+10.919%p` | Pass |
| Mark-1 recall | `49.616%` | `24.664%` | `-24.953%p` | Fail |
| Adjacent accuracy | `94.377%` | `92.976%` | `-1.401%p` | Fail |
| Marker NLL | `0.991274` | `0.991668` | `+0.040%` regression | Pass |
| Time NLL | `4.675246` | `4.664064` | `-0.239%` improvement | Pass |
| Total NLL | `5.666520` | `5.655732` | `-0.190%` improvement | Pass |
| Quantity MAE | `3.060182` | `2.889382` | `-5.581%` improvement | Pass |
| Value MAE | `0.146300` | `0.130431` | `-10.847%` improvement | Pass |

For validation buckets with at least 5% share, the `1-9` quantity MAE regressed
from `0.979752` to `1.031592` (`+5.291%`) and narrowly failed the 5% safety
limit. The `10-99` bucket improved from `9.318595` to `8.720051` (`-6.423%`).

V5a predicted mark 0 for `58.750%` of validation targets versus the true mark-0
share of `41.180%`; the V2 prediction share was `45.030%`. The resulting
mark-0 recall gain did not compensate for the mark-1 recall collapse. The
ordinal-benefit group and the safety group therefore both failed. Per the
predeclared branch rule, no `lambda_ordinal=0.05` or `0.20` follow-up is allowed,
V5a does not proceed to multi-seed validation, and V2 remains the Intermittent
baseline. Held-out test metric content was not read and remains locked.
