# ADR: TitanTPP Q3 Factorial Gradient Routing And Dual-Domain Quantity Loss

- Status: Intermittent seed-42 validation artifact analysis complete; formal acceptance pending
- Date: 2026-07-13
- Scope: Intermittent TitanTPP direct raw-quantity branch
- Predecessor: `adr-titantpp-raw-quantity-revin-q0-q1-q2.md`

## Context

Q2 uses raw-quantity causal moment-shrinkage RevIN with `k=8`. In the frozen
seed-42 validation-only comparison, Q2 improved overall raw quantity MAE by
`14.827%` and history-count-`<=4` raw MAE by `14.838%` versus V2. It also
removed Q1's scale-collapse failure.

Q2 is not eligible for promotion because it regressed log2 quantity MAE by
`7.310%`, regressed the dominant `1-9` quantity bucket by `7.616%`, and reduced
mark accuracy by `3.253%p` versus V2. Its predicted mark-0 share rose to
`61.235%` although the true share is `41.180%`, while mark-1 recall fell to
`17.730%` from V2's `49.616%`.

The observed failures support two separate hypotheses:

1. Direct magnitude losses interfere with the shared representation used by
   marker and time heads.
2. Raw-domain losses are dominated by absolute error and do not sufficiently
   protect the low-quantity region measured by log2 error.

One seed does not establish either hypothesis causally. The next experiment
must isolate both factors without changing Q2 normalization, decoder output,
parameter count, initialization, data order, or training budget.

## Decision Drivers

- Preserve Q2's raw and short-history quantity gains.
- Restore V2-level marker/time behavior, including mark-0/mark-1 confusion.
- Protect `1-9` and log2 quantity error without reopening log-domain RevIN.
- Keep a clean, no-new-parameter ablation.
- Preserve the probabilistic TPP identity
  `nll = nll_marker + nll_time`.
- Keep held-out test artifacts locked until a validation-selected candidate
  passes strict matched multi-seed comparison.

## Non-Goals

- Reopen M2-M4 log-domain RevIN.
- Change Q2 `k=8`, raw moments, sigma floor, or normalized input.
- Change marker CE, time likelihood, Titan profile, memory mode, or checkpoint
  selection.
- Add class-prior correction, ordinal RPS, statistic context, learnable RevIN
  affine parameters, a positive-link decoder, a second encoder, or PCGrad.
- Tune loss coefficients on held-out data.

## Alternatives Considered

### Reuse `value_encoder_gradient_mode`

Rejected. That option belongs to the V3 mark-conditioned residual branch and
currently requires detached mark gating and expert heads. Reusing it would mix
two model families and weaken artifact identity.

### Add a separate magnitude encoder

Rejected for the first test. It changes capacity and parameter count, so marker
recovery could not be attributed to gradient isolation alone.

### Use gradient surgery or learnable task weighting

Deferred. PCGrad, GradNorm, and learned weights add optimization state and more
hyperparameters before the simpler stop-gradient hypothesis is tested.

### Use bucket weights or inverse-frequency quantity weights

Rejected. Bucket weighting is discontinuous at arbitrary quantity boundaries
and can trade away tail performance while overfitting the known validation
distribution.

### Change the decoder to a positive link

Deferred. Softplus or another positive link changes forward predictions as well
as the loss. Q3 must keep Q2 forward behavior fixed to preserve attribution.

### Add a log2 Huber auxiliary to the existing raw losses

Selected. It directly targets the failed log2 guardrail, adds no parameters,
keeps raw RevIN as the input/normalization domain, and can be crossed with the
gradient-routing factor in a complete `2 x 2` design.

## Factorial Variant Contract

All four runs use `direct_raw_qty`, Q2 causal shrinkage normalization, the same
state dictionary, and the same forward computation.

| Variant | Magnitude-to-encoder route | Log2 auxiliary | Role |
| --- | --- | --- | --- |
| Q2 control | coupled | off | fresh matched control |
| Q3a | detached | off | gradient-isolation main effect |
| Q3b | coupled | on | low-quantity-loss main effect |
| Q3c | detached | on | combined interaction candidate |

No variant adds trainable parameters. Q3a and Q2 must have identical scalar
forward and loss values for the same weights and batch. Q3b and Q3c must also
have identical scalar forward and loss values; only their backward routes differ.

## Shared Model Contract

```text
raw observed quantity history
  -> Q2 causal shrinkage normalization
  -> magnitude input projection
  -> Titan MemoryEncoder h
       |- marker head       -> CE / marker NLL
       |- time head         -> RMTPP time NLL
       `- magnitude head    -> normalized raw target
```

Frozen settings:

```text
dataset=intermittent
model=titantpp
titan_candidate=small_lmm
qty_decoder_mode=direct_raw_qty
magnitude_norm_mode=causal_shrinkage_revin
magnitude_shrinkage_k=8
magnitude_sigma_floor=0.0550124034288891
scale_base=2
split_mode=fixed
train_loss_scope=target_only
marker_loss_mode=ce
lambda_ordinal=0
loss_mode=hybrid
lambda_magnitude=1.0
lambda_qty=0.25
qty_scale_value=1.0
lr=1e-3
batch_size=128
lookback=52
max_seq_len=16
epochs=50
seed=42
checkpoint_selection=best_val_nll
```

The appended target and padding remain excluded from the magnitude context and
encoder input. The fresh Q2 control is rerun in the same code revision as Q3 so
implementation drift cannot be mistaken for a treatment effect.

## Gradient-Routing Contract

Add a direct-magnitude-only option:

```text
magnitude_encoder_gradient_mode = coupled | detached
```

For `coupled`, preserve Q2:

```text
h_mag = h_j
u_hat = magnitude_head(h_mag)
```

For `detached`:

```text
h_mag = stop_gradient(h_j)
u_hat = magnitude_head(h_mag)
```

This detaches only the hidden state consumed by the magnitude head. Observed raw
quantity remains an encoder input. Marker/time losses may therefore still train
the magnitude input projection and encoder to use observed quantity, while
`raw_norm_loss`, `raw_qty_loss`, and `log_qty_aux_loss` cannot update either.

Expected direct routes:

| Loss | Magnitude head | Encoder | Magnitude input projection | Marker head | Time head |
| --- | ---: | ---: | ---: | ---: | ---: |
| magnitude losses, coupled | yes | yes | yes | no | no |
| magnitude losses, detached | yes | no | no | no | no |
| marker + time NLL | no | yes | yes | yes | yes |

Do not reuse `value_encoder_gradient_mode`; the two fields must remain separate
in config, CLI, path, manifest, checkpoint, cache, resume validation, histories,
and summaries.

## Dual-Domain Loss Contract

Q2 raw normalization and raw prediction remain unchanged:

```text
u_target = (q_target - center) / scale
u_hat = magnitude_head(h_mag)
q_affine = center + scale * u_hat
q_hat = max(q_affine, 0)              # evaluation/inference only
```

The existing raw objectives remain primary:

```text
L_norm = Huber(u_hat, u_target, delta=1)
L_raw  = Huber(q_affine, q_target, delta=1)
```

Q3b/Q3c add:

```text
q_log_floor = 1.0
z_hat = log2(max(q_affine, q_log_floor))
z     = log2(max(q_target, q_log_floor))
L_log = Huber(z_hat, z, delta=1)

L_total = marker_ce
        + time_nll
        + 1.00 * L_norm
        + 0.25 * L_raw
        + 0.25 * L_log
```

Q2/Q3a set `L_log=0`. The log floor is fixed at `1.0` because the marked demand
contract has positive quantity with minimum one. Below the floor, `L_log` has no
gradient; the unclamped `L_norm` and `L_raw` continue to recover negative or
sub-one affine predictions. Huber bounds the log-residual derivative, and the
`0.25` coefficient keeps this term auxiliary.

This is still raw-domain RevIN. A log transform appears only in an auxiliary
error term; it is not used for context statistics, encoder input, decoder target,
or denormalization.

New configuration identity:

```text
magnitude_aux_loss_mode = none | log_huber
lambda_log_qty = 0.25
log_qty_huber_delta = 1.0
log_qty_floor = 1.0
```

The non-default path is valid only for `direct_raw_qty` with
`causal_shrinkage_revin`, plain CE, fixed split, target-only loss, and no V3/V5
feature. Invalid mixed contracts fail before model construction.

## Artifact Contract

Run identity must include at least:

```text
magencgrad_<coupled|detached>
magaux_<none|log_huber>
lambdalogqty_<value>
logqtyfloor_<value>
```

Manifest, model config, checkpoint, cache key, resume validation, history,
summary, and scale-wise outputs must persist the same fields. Training and
validation logs add `log_qty_aux_loss`; this metric is never added to likelihood
NLL. Existing raw/log2 MAE, context-count metrics, pre-clamp negative share,
marker confusion, and scale-wise metrics remain unchanged.

## Focused Implementation Gate

Implementation cannot advance to actual-data screening until all checks pass:

1. Q2/Q3a/Q3b/Q3c have exactly matching parameter keys, counts, and seeded
   tensors.
2. All four variants produce identical hidden states, marker/time outputs,
   normalized magnitude predictions, and quantities for the same state.
3. Q2 and Q3a scalar component and total losses are exactly equal.
4. Q3b and Q3c scalar component and total losses are exactly equal.
5. Detached magnitude losses update only `magnitude_head`; encoder,
   magnitude-input projection, marker head, and time head gradients are zero.
6. Coupled magnitude losses preserve Q2 encoder/input-projection gradients.
7. Marker/time NLL still trains encoder, magnitude-input projection, marker head,
   and time head in every variant without training `magnitude_head` directly.
8. `L_log` equals the hand-computed masked formula and does not include padding
   or appended-target context.
9. Negative and sub-one `q_affine` remain finite and receive gradients from both
   raw losses even when the log branch is floored.
10. Default config loads legacy Q2 checkpoints with coupled/no-aux behavior and
    unchanged state-dict keys.
11. Artifact paths and cache/resume checks distinguish all four variants.
12. CPU focused tests, complete search regression tests, and 5090 CUDA
    model-test return finite outputs.

## Actual-Data Integration Gate

Run Q2/Q3a/Q3b/Q3c on the same Instacart top-20 fixed split for one epoch. This
gate verifies actual-data backward, checkpoint loading, resume/cache identity,
new loss logging, summary/scale-wise generation, and finite values. It is not a
performance ranking and does not unlock held-out Intermittent data.

## Seed-42 Validation-Only Design

Run the fresh Q2 control and all Q3 variants on 5090 with the frozen Intermittent
e50 contract. Do not early-stop Q3c based on Q3a or Q3b because an interaction
may exist. Read artifacts under the validation-only lock.

The fresh Q2 control must first reproduce the frozen Q2 within:

- total NLL, raw MAE, and log2 MAE: `<=1%` relative difference;
- mark accuracy: `<=0.25%p` absolute difference;
- matching data counts, train moments, config, and checkpoint policy.

If reproduction fails, stop attribution and investigate drift before comparing
Q3 variants.

## Mechanism Diagnostics

These diagnostics explain effects but do not replace the full candidate gate.

### Q3a: Gradient-Isolation Evidence

Let the mark-accuracy recovery ratio be:

```text
(acc_Q3a - acc_Q2) / (acc_V2 - acc_Q2)
```

- `>=0.50`: supports substantial shared-gradient interference;
- `<0.25`: gradient interference alone is weak evidence, and the changed raw
  input representation remains a likely source;
- raw overall and short-history MAE must each remain within `5%` of fresh Q2 for
  the routing change to be practically useful.

### Q3b: Low-Quantity-Loss Evidence

- log2 MAE improves at least `5%` versus fresh Q2;
- `1-9` MAE is no more than `2%` worse than V2;
- overall and short-history raw MAE remain within `5%` of fresh Q2.

### Q3c: Interaction

For each metric, report the factorial interaction:

```text
interaction = (Q3c - Q3a) - (Q3b - Q2)
```

The sign is interpreted according to whether lower or higher is better. It is a
diagnostic, not an additional promotion threshold.

## Candidate Acceptance Gate

Every Q3 candidate is evaluated at its own `best_val_nll` checkpoint. A candidate
is eligible only if all sections pass.

### Retain Q2 Quantity Benefit

- overall raw MAE is at least `10%` better than V2 and no more than `5%` worse
  than fresh Q2;
- history-count-`<=4` raw MAE satisfies the same rule;
- frozen seed-42 numeric ceilings are `2.736781` overall and `2.053191` for
  history count `<=4`.

### Protect Low Quantity

- log2 quantity MAE regression versus V2 is `<=2%` (`<=0.600517`);
- `1-9` raw MAE regression versus V2 is `<=2%` (`<=0.999348`);
- every other quantity bucket with validation share `>=5%` regresses `<=5%`;
  for `10-99`, the ceiling is `9.784525`.

### Marker And Time Safety

- marker NLL regression versus V2 `<=1%` (`<=1.001186`);
- total NLL regression `<=0.5%` (`<=5.694853`);
- time NLL regression `<=0.5%` (`<=4.698623`);
- mark accuracy gap `>=-0.25%p` (`>=56.999%`);
- DT MAE regression `<=2%` (`<=42.905873`);
- absolute predicted mark-0 share error is no more than V2's error plus `2%p`
  (`<=5.850%p` from the true share);
- mark-1 recall is no more than `5%p` below V2 (`>=44.616%`).

### Numeric Safety

- all losses, predictions, context statistics, and gradients are finite;
- pre-clamp negative prediction share `<=1%`;
- normalized-target non-finite count is zero;
- no target or padding leakage.

Q3a and Q3b remain valid mechanism diagnostics even when they fail full
eligibility. Q3c can pass through an interaction even if neither single-factor
variant passes, so all four runs are completed before selection.

## Selection Rule

1. Discard every candidate that fails any common acceptance section.
2. Prefer a one-factor candidate over Q3c when it passes the full gate; this
   avoids an unnecessary intervention and hyperparameter.
3. If both Q3a and Q3b pass, prefer Q3a because it preserves the Q2 loss and has
   no new loss coefficient.
4. Select Q3c only when the combination is required to pass the full gate.
5. If no Q3 candidate passes, retain V2 and do not tune Q3 from held-out results.

## Strict Multi-Seed And Held-Out Gate

Freeze the selected candidate and all constants. Run strict matched V2, fresh
Q2, and the selected candidate for seeds `42,52,62`, e50. Checksum-verified
frozen V2 artifacts may be reused only when their data/config/code contract is
identical; otherwise rerun V2.

Required before held-out unlock:

- `3/3` runs complete without non-finite values or runtime errors;
- mean candidate acceptance gates remain satisfied;
- overall and short-history raw improvement versus both V2 and Q2 occurs in at
  least `2/3` seed-matched comparisons;
- mean mark accuracy gap versus V2 is `>=-0.25%p` and no seed is below
  `-0.75%p`;
- mean log2 and `1-9` gates pass, with no seed exceeding `5%` regression versus
  V2;
- mean marker/total/time NLL, DT, confusion, bucket, and numeric safety gates
  remain satisfied.

Only then read held-out artifacts in protocol order. A failed frozen held-out
audit returns to V2 without test-driven Q3 retuning.

## Risks And Mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| Detaching h removes useful magnitude representation learning | Q3a/Q3c lose Q2 raw gain | Require within-5% Q2 retention and keep coupled controls |
| Log floor has zero gradient below one | Very poor predictions rely on raw loss | Preserve unclamped normalized/raw losses and negative-share gate |
| Log auxiliary worsens encoder conflict in Q3b | Marker degradation | Q3b is a main-effect diagnostic; Q3c crosses it with detachment |
| Validation reuse overfits thresholds | Optimistic seed-42 selection | Freeze the full contract now, require multi-seed, keep held-out locked |
| Fresh Q2 differs after implementation | False treatment attribution | Reproduction gate blocks comparison |
| Marker accuracy hides class collapse | Unsafe apparent recovery | Add predicted mark-0 error and mark-1 recall gates |

## Consequences

Q3 is a complete factorial attribution study rather than three unrelated model
versions. It can distinguish shared-gradient interference, low-scale loss
imbalance, and their interaction while preserving Q2's raw-domain RevIN
semantics. The cost is one fresh Q2 control and three Q3 e50 runs before any
multi-seed promotion.

## Implementation Evidence

- Added independent `magnitude_encoder_gradient_mode` and
  `magnitude_aux_loss_mode` axes without adding parameters or changing state-dict
  keys.
- Added the masked log2 Huber auxiliary as a separate metric and training term;
  likelihood `nll` remains marker CE plus time NLL.
- Propagated Q3 identity through CLI, model config, run paths, manifests,
  checkpoints, cache/resume checks, histories, summaries, and scale-wise rows.
- Focused Q3 contract tests passed `19/19`; the complete search suite passed
  `104/104`.
- Local CPU model-tests for Q2/Q3a/Q3b/Q3c all succeeded with parameter count
  `78,111` and identical NLL, magnitude loss, and quantity predictions. Only
  Q3b/Q3c reported the active log auxiliary loss, as designed.
- Preparation commit `f4cc223` was checksum-verified on the non-Git 5090 working
  copy. `source_sync_manifest.json` preserves the full revision because the
  remote experiment manifest cannot resolve Git metadata.
- The CUDA preflight passed on RTX 5090 with PyTorch `2.11.0+cu130`. Tmux session
  `titantpp_q3_cuda_0713` started at `2026-07-13 23:04:19 KST`; the initial check
  observed Q2/Q3a/Q3b exit successfully and Q3c enter its model-test.
- The run ended at `2026-07-13 23:04:26 KST` with `MODEL_TEST_SUCCESS`, aggregate
  exit code `0`, and all four variant exit codes equal to `0`.
- All 13 artifact files were synced locally. The four variants have identical
  parameter count `78,111`, hidden shape `[4,16,64]`, NLL components,
  magnitude/raw losses, and quantity/time prediction summaries.
- Q2/Q3a are exact scalar matches with zero log auxiliary. Q3b/Q3c are exact
  scalar matches with the same positive log auxiliary `3.797908067703247`.
  CLI/RMTPP config differs only in the intended gradient/aux factors and output
  path; encoder config is identical.
- Total-loss recomputation agrees within `1.58e-6` FP32 accumulation error. The
  5090 CUDA runtime and artifact identity gate passed.
- A matched Instacart top-20 e1 runner now fixes the prior Q0/Q1/Q2 data and
  training budget while crossing only magnitude encoder gradient routing and
  log-auxiliary mode. It records a root manifest, independent variant status,
  per-variant logs, and success/failure sentinels.
- The start record freezes tmux `titantpp_q3_insta_e1_0714`, artifact root
  `model_enhancement_titantpp_q3_insta_smoke_e1_0714`, expected sample counts
  `1380/300/300`, artifact reading order, and the integration-only gate.
- Preparation revision `d552b77` was checksum-verified on 5090. The source
  manifest records four exact file hashes and the prior Q3 implementation
  revision `14c2978` for the non-Git remote working copy.
- The CUDA/data preflight passed on RTX 5090 with PyTorch `2.11.0+cu130`, exact
  top-20 loader samples `1380/300/300`, and all four factorial CLI contracts.
- Tmux `titantpp_q3_insta_e1_0714` started at `2026-07-14 08:45:33 KST`. The
  one-time initial check observed Q2 complete epoch 1 on the expected split and
  train-only raw statistics with an active CUDA process.
- The requested one-time completion check found the tmux session closed, no active
  GPU process, root `SMOKE_SUCCESS`, no failure sentinel, aggregate exit code `0`,
  and Q2/Q3a/Q3b/Q3c exit codes all equal to `0`. The run ended at
  `2026-07-14 08:45:53 KST`.
- The complete artifact root was checksum-synced locally: `388` files, about
  `18M`, with no changes reported by a checksum dry-run. Root metadata parses,
  and every variant contains its manifest, summary, test summary, history,
  validation/test scale-wise metrics, report, plots, and best-validation-NLL
  checkpoint.
- Protocol-order analysis found no runtime, loss, prediction, checkpoint, resume,
  summary, history, scale-wise, report, or plot failure. Q2/Q3a log auxiliary is
  exactly zero and Q3b/Q3c auxiliary is positive finite in train, validation,
  and test export.
- All four actual-data checkpoints have the same 40 tensor keys and shapes,
  `77,626` parameters, finite tensors, direct magnitude modules, and no legacy
  value head. Their e1 best-validation-NLL, best-score, final, and resume states
  are internally exact matches as expected from a one-epoch run.
- The root manifest's `expected_parameter_count=78,111` was a non-blocking
  metadata defect copied from the synthetic CUDA gate with `num_marks=12`.
  Instacart uses `num_marks=7`; the exact `485` difference is the five removed
  rows in the 32-dimensional mark embedding and 64-dimensional mark head plus
  bias. The source runner now records actual `77,626` separately from the
  synthetic `78,111` reference; the immutable run artifact was not rewritten.
- The requested sigma-floor identity `0.0550124034288891` is distinct from the
  actual Instacart effective floor `0.0067776913473542024`, computed from the
  train-only global raw standard deviation. All variants persist the same value.
- Scale counts reconcile to 300 targets per split and weighted scale metrics
  reconcile to overall metrics within `1.03e-7`. Legacy direct-head value MAE
  and empty 100+ buckets are intentional N/A cells, not non-finite model output.
- All 16 PNG plots are valid. The one-epoch learning curves have no visible line
  because a single point is rendered without a marker; this is a presentation
  limitation rather than a training failure.
- The actual-data integration gate passed. E1 validation differences and held-out
  test exports were not used for performance ranking or candidate selection.
  Intermittent, multi-seed, and held-out Q3 experiments remain unstarted.
- The Intermittent runner now freezes fresh Q2/Q3a/Q3b/Q3c at e50, seed 42,
  batch 128, lookback 52, and max sequence 16. It validates the exact V2
  checkpoint, all five fixed-split source files, and frozen Q2 summary SHA before
  training. Test-file hashing verifies identity only and does not inspect metrics.
- The runner emits an unrounded machine-readable `acceptance_contract.json`
  containing Q2 reproduction, full candidate, mechanism, selection, and
  held-out-lock rules. Q3a or Q3b failure cannot short-circuit Q3c.
- Preparation commit `a0a65e5` and all runtime dependencies were checksum-verified
  on 5090. CUDA/data/reference/CLI preflight passed on the idle RTX 5090.
- The first launch stopped before training because the validation-reference
  evaluator treated legacy V2's structurally inapplicable
  `val_log_qty_aux_loss=NaN` as an active metric. Held-out data was not read, and
  the failed root log was preserved.
- Recovery commit `f5851ff` exports inactive mark-residual magnitude metrics as
  JSON null while retaining finite checks for every active metric. Focused tests
  passed `25/25` and the full search suite passed `110/110`.
- The second tmux launch started at `2026-07-15 08:15:07 KST`. V2 validation-only
  reference completed with exit code zero, and fresh Q2 completed epoch 1 on the
  frozen split and train-only raw statistics. Continuous monitoring stopped at
  `08:21:16 KST`.
- The run completed at `2026-07-15 08:46:16 KST` with root
  `SCREENING_SUCCESS`; fresh Q2/Q3a/Q3b/Q3c all exited with code zero. All `562`
  artifact files (`27,179,501` bytes) are synced locally, and checksum dry-run
  found no remote/local differences.
- Test-file hashing remained identity-only. No held-out row, metric, report, or
  plot was inspected.

## Seed-42 Validation Artifact Analysis (2026-07-15)

### Integrity And Scope

- Root/variant manifests, logs, summaries, histories, validation scale-wise
  summaries, validation mark class/confusion files, and validation plots were
  read in protocol order.
- After excluding `base_dir` and the two intended factorial axes, all four
  `ExperimentConfig` objects are identical. Every variant has one summary row,
  exactly 50 history rows with epochs `1..50`, and finite active metrics.
- Scale and confusion counts each reconcile to `41,901` validation targets.
  Weighted scale MAE agrees with overall MAE within floating-point tolerance.
- Normalized-target non-finite count is zero for every variant. Q3a alone has a
  small pre-clamp negative share (`0.2745%`); the other variants are zero.
  The `>=10000` scale bucket has zero targets, so its N/A values are expected.
- The V2 values below come from the validation-only reference generated from
  checkpoint SHA `1a901eb2...`. No held-out file was read.

### Best-Validation-NLL Comparison

| Variant | Epoch | Total NLL | Marker NLL | Time NLL | Mark acc. | DT MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | 19 | 5.666520 | 0.991274 | 4.675246 | 57.249% | 42.0646 |
| Fresh Q2 | 32 | 5.670936 | 0.988991 | 4.681945 | 55.168% | 42.3018 |
| Q3a | 27 | 5.660492 | 0.989653 | 4.670839 | 55.178% | 42.4310 |
| Q3b | 40 | 5.634060 | 1.001103 | 4.632958 | 54.963% | 41.5708 |
| Q3c | 32 | 5.665948 | 0.988030 | 4.677917 | 55.853% | 42.2033 |

| Variant | Raw qty MAE | Log2 qty MAE | History `<=4` raw MAE | Mark-0 share error | Mark-1 recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 | 3.060182 | 0.588742 | 2.296124 | 3.850%p | 49.616% |
| Fresh Q2 | 2.762922 | 0.683652 | 2.122701 | 16.026%p | 24.020% |
| Q3a | 3.333413 | 0.875693 | 2.759409 | 18.288%p | 19.331% |
| Q3b | 2.649259 | 0.686737 | 2.002956 | 20.205%p | 20.957% |
| Q3c | 3.386220 | 0.738020 | 2.817617 | 6.434%p | 41.742% |

Q3b has the best total NLL, time NLL, overall raw MAE, short-history raw
MAE, and DT MAE. Its total-NLL gain versus fresh Q2 (`-0.036875`) is entirely
time-driven: time NLL improves by `-0.048987` while marker NLL worsens by
`+0.012112`. Its raw/short MAE improve by `4.11%/5.64%`, but log2 MAE worsens
by `0.45%`, mark accuracy falls by `0.205%p`, and mark-0 share error grows by
`4.179%p`.

Q3a does not support the standalone gradient-isolation hypothesis. Relative to
fresh Q2, marker accuracy changes by only `+0.010%p`, while raw, log2, and
short-history MAE worsen by `20.65%`, `28.09%`, and `30.00%`.

Q3c shows a different trade-off. Relative to fresh Q2, mark accuracy improves by
`0.685%p`, mark-0 share error drops by `9.592%p`, and mark-1 recall rises by
`17.722%p`. Raw, log2, and short-history MAE nevertheless worsen by `22.56%`,
`7.95%`, and `32.74%`. It approaches V2's mark distribution but remains below
V2 accuracy/recall and materially worse in quantity prediction.

### Fresh-Q2 Reproduction Warning

| Metric | Frozen Q2 | Fresh Q2 | Difference |
| --- | ---: | ---: | ---: |
| Total NLL | 5.625528 | 5.670936 | +0.807% |
| Raw qty MAE | 2.606458 | 2.762922 | +6.003% |
| Log2 qty MAE | 0.631778 | 0.683652 | +8.211% |
| Mark accuracy | 53.996% | 55.168% | +1.172%p |
| Best epoch | 46 | 32 | -14 epochs |

Only total NLL is inside the frozen `1%` reproduction tolerance. Raw/log MAE
and the `0.25%p` mark-accuracy tolerance do not reproduce. The effective model,
data, and training settings match; the only manifest differences are output
path and new Q3 fields that are no-ops for Q2. Training losses become very close
after the first epochs, but validation trajectories and selected epochs remain
volatile. The current seed helper seeds Python/NumPy/PyTorch but does not enable
deterministic CUDA algorithms, so GPU nondeterminism is a plausible contributor,
not a proven sole cause.

This drift is decision-material. Q3b's NLL/raw-MAE effects and Q3c's accuracy
effect are no larger than the fresh-versus-frozen Q2 movement on the same metric.
The within-run factorial differences remain useful diagnostics, but they are not
yet reliable causal effects or promotion evidence.

### History And Stability

- All variants converge sharply during the first `5-10` epochs and then
  oscillate. Final NLL is worse than best NLL by `0.116-0.192`, so final-epoch
  comparison would be misleading.
- Best NLL epochs differ substantially: Q2/Q3a/Q3b/Q3c are `32/27/40/32`.
  Quantity-optimal, log-optimal, accuracy-optimal, and NLL-optimal epochs also
  differ within every variant.
- Log2 quantity MAE has finite but large spikes: maxima are Q2 `1.063`, Q3a
  `4.463`, Q3b `1.779`, and Q3c `8.869`. Detachment is associated with the two
  largest spikes; the log-Huber auxiliary does not remove Q3c instability.
- Q3b's final NLL is `0.191616` above its epoch-40 optimum even though final raw
  MAE is nearly unchanged, confirming that its late degradation is mainly in
  the probabilistic heads rather than raw quantity magnitude.

### Scale-Wise Quantity Behavior

Validation shares are `88.666%` for `1-9`, `10.723%` for `10-99`, `0.527%` for
`100-999`, and `0.084%` for `1000-9999`.

| Variant | `1-9` MAE | `10-99` MAE | `100-999` MAE | `1000-9999` MAE |
| --- | ---: | ---: | ---: | ---: |
| V2 | 0.9798 | 9.3186 | 99.9079 | 796.4802 |
| Fresh Q2 | 1.0833 | 9.2759 | 82.2982 | 447.4009 |
| Q3a | 1.2486 | 8.2773 | 114.3084 | 880.9077 |
| Q3b | 1.0921 | 8.5389 | 81.1012 | 404.1697 |
| Q3c | 1.1365 | 9.0235 | 120.9641 | 925.3099 |

Q3b's overall raw-MAE advantage is not a low-quantity improvement. The dominant
`1-9` bucket is `11.46%` worse than V2 and `0.81%` worse than fresh Q2. Its gain
comes from `10-99` and the much rarer tail buckets. Q3a is best on `10-99` but
regresses both the dominant bucket and tail. Q3c improves `10-99` slightly but
regresses the dominant `1-9` bucket and both tail buckets versus fresh Q2. Tail
results, especially the 35-sample `1000-9999` bucket, should not be treated as
stable single-seed evidence.

### Mark Confusion And Factorial Interaction

The true mark-0 share is `41.180%`. Predicted mark-0 shares are Q2 `57.206%`,
Q3a `59.469%`, Q3b `61.385%`, and Q3c `47.615%`. Q2/Q3a/Q3b therefore collapse
toward mark 0 and underpredict the true mark in `32.23%/32.85%/35.27%` of
validation cases. Q3c lowers underprediction to `28.43%` and has the best Q3
mark MAE/adjacent accuracy (`0.5038/94.148%`), but V2 remains better
(`0.4874/94.377%`).

The 2x2 interaction is strongly non-additive. For lower-is-better metrics, the
interaction is unfavorable for total NLL (`+0.04233`) and raw MAE (`+0.16647`).
For marker balance it is favorable: mark-0 share-error interaction is
`-16.033%p`, mark-1-recall interaction is `+25.473%p`, and accuracy interaction
is `+0.881%p`. The combined variant is therefore a marker-balance mechanism,
not a balanced quantity-and-TPP winner.

### Plot Review And Assessment

- Learning-curve plots agree with the CSV: rapid initial convergence, persistent
  validation oscillation, and quantity/log2 spikes are visible.
- Scale-wise MAE plots use a linear axis dominated by rare high-quantity buckets,
  so differences in the `1-9` bucket are visually compressed. The table and WAPE
  panel are required for interpretation.
- Per-variant plots use independent y-axis ranges and should not be compared by
  visual slope or bar height alone.

Artifact integrity is ready for the formal gate, but causal attribution and
promotion are **not ready to share** because fresh Q2 failed the frozen
reproduction contract and only one seed is available.

## Next Step

Apply the frozen Q2 reproduction, candidate, mechanism, and selection gates.
Decide whether the failed reproduction gate requires deterministic controls and
a matched rerun before any multi-seed promotion. Keep held-out execution locked.
