# ADR: TitanTPP Q3 Factorial Gradient Routing And Dual-Domain Quantity Loss

- Status: Accepted design; implementation not started
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

## Next Step

Implement the two orthogonal config axes, focused forward/loss/gradient tests,
and full artifact identity. Do not start Intermittent e50 before local tests,
5090 CUDA model-test, and Instacart e1 integration pass.
