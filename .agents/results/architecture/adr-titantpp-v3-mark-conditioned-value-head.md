# ADR: TitanTPP V3 Mark-Conditioned Value Head

- Date: 2026-07-10
- Status: Accepted for implementation
- Scope: TitanTPP value prediction head and its training/evaluation adapters
- Method: Design-Twice followed by ADR

## Context

TitanTPP V2 predicts one residual value from the history representation and reuses it for every possible next mark:

```text
h_t -> mark logits p(m | h_t)
h_t -> one residual r(h_t)
E[q | h_t] = sum_m p(m | h_t) * base^(m + r(h_t))
```

The V1/V2 multi-seed experiment showed that V2 hybrid supervision improves quantity MAE, especially on intermittent demand and Yellow Trip, but taxi marker NLL and seed stability remain weak. V3 must model the dependency between next mark and value residual without changing the Titan backbone, RMTPP time likelihood, dataset, split, or lookback policy.

The intended factorization is:

```text
p(m, r, dt | h) = p(m | h) * p(r | m, h) * p(dt | h)
```

## Constraints And Quality Attributes

- Preserve V2 behavior when V3 is disabled.
- Do not feed the ground-truth target mark or value into the encoder.
- Keep mark CE and RMTPP time intensity unchanged.
- Keep `loss_mode=hybrid`, `lambda_qty=0.25`, and target-only supervision fixed for the first V3 comparison.
- Exclude the PAD mark from value experts and quantity reconstruction.
- Minimize runner/API changes and keep RMTPP/THP behavior unchanged.
- Make V3 start from a V2-equivalent initialization to reduce optimization risk.
- Keep the design parameter-efficient for both `small_lmm` and `mid_lmm`.

## Options Considered

### Option A: Soft Mark-Embedding Fusion

Compute a soft expected mark embedding from predicted mark probabilities and concatenate it with `h_t` before an MLP value head.

```text
c_t = softmax(mark_logits) @ E_mark
r_hat = MLP([h_t, c_t])
```

Advantages:
- Same predicted condition can be used in train and inference.
- Small API and parameter increase.
- Easy to add a residual gate initialized to zero.

Disadvantages:
- Still emits one residual shared by all mark branches.
- An uncertain mark distribution produces an averaged condition that may blur scale-specific residuals.
- Does not directly implement `p(r | m, h)`.

### Option B: Independent Per-Mark Experts

Emit one independently parameterized residual for each real mark.

Advantages:
- Direct class-conditional residual modeling.
- Straightforward training and inference semantics.

Disadvantages:
- Rare marks receive little supervision.
- Discards useful sharing between adjacent quantity scales.
- Cannot reproduce V2 exactly at initialization without extra initialization logic.

### Option C: Shared Residual Plus Per-Mark Delta Experts

Keep the existing V2 residual head as a shared base and add a zero-initialized mark-specific delta head.

```text
r_shared = value_head(h_t)
delta = value_mark_delta_head(h_t)          # K_real values
r_by_mark[k] = activation(r_shared + delta[k])
```

Advantages:
- Directly models `p(r | m, h)`.
- Shares statistical strength through `r_shared` while allowing scale-specific corrections.
- Zero-initialized deltas make every expert equal to V2 at initialization.
- Adds only `d_model * K_real + K_real` parameters.
- Supports differentiable expected quantity without argmax.

Disadvantages:
- Evaluation must select the residual associated with the predicted mark.
- Rare mark deltas can remain weakly trained.
- Hybrid quantity loss can still send conflicting gradients to the mark head.

## Decision

Use Option C, shared residual plus per-mark delta experts.

### Model Interface

Add the following configuration while keeping `shared` as the backward-compatible default:

```text
value_head_mode: shared | mark_conditioned_experts
```

The option must be exposed as `--value-head-mode`, persisted in the experiment manifest/run rows, and included in the run directory key such as `valuehead_shared` or `valuehead_mark_conditioned_experts`. Otherwise V2 and V3 runs with the same dataset/candidate/seed can collide.

TitanTPP adds:

```text
value_mark_delta_head: Linear(d_model, num_marks - 1)
predict_value_by_mark(h): [B, L, K_real]
```

The existing `value_head` remains the shared V2 head. `value_mark_delta_head` weights and biases are initialized to zero.

### Training Path

For target mark `m_true` and target residual `r_true`:

```text
mark_logits = mark_head(h_t)
r_by_mark = predict_value_by_mark(h_t)
r_selected = gather(r_by_mark, m_true)
value_loss = Huber(r_selected, r_true)
```

The ground-truth mark is used only to select which conditional expert receives residual supervision. It is not passed to the encoder or used to generate mark logits. This is conditional likelihood training, not target leakage into the sequence representation.

The hybrid quantity estimate becomes:

```text
E[q | h_t] = sum_k softmax(mark_logits_real)[k] * base^(k + r_by_mark[k])
qty_loss = Huber(E[q | h_t], q_true)
```

### Evaluation Path

```text
m_pred = argmax(mark_logits_real)
r_true_branch = gather(r_by_mark, m_true)
r_pred_branch = gather(r_by_mark, m_pred)
value_mae = abs(r_true_branch - r_true)
q_pred = base^(m_pred + r_pred_branch)
```

The true-mark branch is used only for conditional residual diagnostics. The predicted-mark branch is used for actual quantity prediction. Generic evaluation code should call a helper that uses `predict_value_by_mark` when available and falls back to the existing `predict_value` for RMTPP/THP/shared TitanTPP.

### Gradient Policy

V3a keeps mark probabilities coupled to quantity loss, matching V2 and isolating the architecture change. If taxi marker NLL remains degraded, V3b may detach mark probabilities only inside the quantity-loss expectation. Gradient detachment is not part of the first V3 implementation because combining both changes would confound the ablation.

## Consequences

Positive:
- The model can learn different residual corrections for low-, medium-, and high-scale marks.
- V3 starts numerically equivalent to V2 before delta experts learn.
- The Titan backbone and probabilistic time head remain untouched.
- V3 directly targets the quantity/marker mismatch observed in taxi.

Negative:
- The official runner and legacy prediction helpers need a small compatibility adapter.
- Per-mark metrics and gradient checks become necessary.
- Tail experts may require regularization or additional seeds if their support is sparse.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Rare mark expert undertraining | High-scale residuals remain noisy | Shared base head, zero-init deltas, scale-wise metrics |
| Quantity gradient continues to hurt marker NLL | Taxi trade-off remains | Keep V3a coupled for clean ablation; add V3b detached-gate only if needed |
| PAD expert is accidentally trained | Invalid quantity reconstruction | Allocate exactly `num_marks - 1` experts and mask/clamp targets defensively |
| Train/inference mismatch | Residual quality drops after mark errors | Train conditional expert with true mark, evaluate with predicted mark, and report mark-confusion-conditioned quantity errors |
| API regression in RMTPP/THP callers | Existing evaluations break | Central compatibility helper with shared-head fallback |
| V2/V3 artifact collision | Results overwrite or resume the wrong run | Include `value_head_mode` in run path, manifest, and leaderboard rows |
| Exponential quantity overflow | NaN/Inf in tail | Reuse current stability checks and add finite assertions in model-test |

## Non-Goals

- No mark-conditioned time head; that remains V4.
- No ordinal mark objective; that remains V5.
- No Titan memory-mode or series-memory change.
- No lookback, max sequence length, split, scale-base, or dataset preprocessing change.
- No ground-truth target mark/value input to the encoder.

## Follow-Up Validation

### Model Tests

- Shared mode reproduces current TitanTPP output shapes and losses.
- V3 returns `[B, L, K_real]` mark-conditioned residuals.
- Zero-initialized V3 experts equal the V2 shared residual for every mark.
- V3 expected quantity equals V2 expected quantity at initialization.
- Residual loss routes gradients to the true-mark delta expert.
- Quantity loss produces finite gradients for all populated experts.
- Changing target labels does not change encoder states or mark logits before loss computation.
- Evaluation uses the true-mark branch for residual MAE and the predicted-mark branch for reconstructed quantity.
- V2 and V3 resolve to distinct run directories and manifest configurations.
- Left padding and appended-target value masking remain valid.

### Experiment Sequence

1. CPU synthetic `model-test` for shared and V3 modes.
2. One-epoch fixed-split smoke on a small Instacart subset.
3. Short V3a screening on `intermittent-small_lmm` and `yellow_trip_hourly-mid_lmm`.
4. Full fixed-split V3a comparison on all three datasets with seeds `42,52,62`.
5. Run V3b detached-gate only if taxi marker NLL or seed variance remains worse than V2.

### Decision Metrics

- Compare against V2 S0 using the same candidates and `best_val_nll` checkpoint.
- Preserve or improve V2 test score and quantity MAE where possible.
- Taxi must improve marker NLL/mark accuracy relative to V2 or clearly reduce seed variance while retaining most of the quantity gain.
- Report scale-wise quantity MAE and seed-wise direction; do not rely on aggregate score alone.

## V3a Screening Outcome

The seed-42 e50 screening completed on 2026-07-10. V3a improved quantity and
value MAE on Intermittent and Yellow Trip Hourly, but mark accuracy decreased on
both datasets and Taxi marker NLL worsened by `17.501%`. The detached-gate
follow-up condition in this ADR is therefore satisfied.

V3a remains the coupled architecture ablation and is not selected as the final
V2 replacement. The accepted follow-up decision is documented in:

```text
.agents/results/architecture/adr-titantpp-v3b-detached-quantity-gate.md
```
