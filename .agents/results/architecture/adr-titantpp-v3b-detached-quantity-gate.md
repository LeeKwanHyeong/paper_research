# ADR: TitanTPP V3b Detached Quantity Gate

- Date: 2026-07-10
- Status: Accepted for implementation
- Scope: TitanTPP hybrid quantity-loss gradient routing and experiment identity
- Method: Design-Twice followed by ADR

## Context

TitanTPP V3a keeps the V2 hybrid quantity objective and replaces the shared
residual prediction with shared-plus-mark-delta experts:

```text
p_k = softmax(mark_logits)_k
r_k = activation(r_shared + delta_k)
q_k = base^(k + r_k)
q_hat = sum_k p_k * q_k
```

This path sends quantity-loss gradients through two routes:

```text
L_qty -> p_k -> mark_logits -> mark_head and encoder
L_qty -> q_k -> value experts and encoder
```

The seed-42 e50 screening confirms that V3a improves quantity and residual
prediction but does not preserve the marker guardrails.

| Dataset | Quantity MAE | Value MAE | Marker NLL | Mark accuracy |
| --- | ---: | ---: | ---: | ---: |
| Intermittent | `-13.315%` | `-23.748%` | `-1.004%` | `-1.023%p` |
| Yellow Trip Hourly | `-6.988%` | `-11.075%` | `+17.501%` | `-0.745%p` |

Intermittent also shifts many true mark-1 and mark-2 events toward mark 0.
Taxi improves mark 0 while reducing mark-1 and mark-2 accuracy. These results
do not prove that the direct quantity-to-mark gradient is the sole cause, but
they satisfy the V3a ADR condition for evaluating a detached-gate variant.

## Problem

Preserve V3a's mark-conditioned quantity/value learning while preventing the
hybrid quantity objective from directly optimizing the mark-probability gate.
The change must isolate gradient routing without changing the forward quantity
estimate, parameter initialization, probabilistic time head, data path, or
evaluation contract.

## Constraints And Quality Attributes

- V3a and V3b must produce identical forward values for identical weights and
  inputs; only their backward graphs may differ.
- Marker CE must continue to train the mark head and shared encoder.
- Residual loss must continue to train the true-mark expert and shared encoder.
- Quantity loss must continue to train all value experts through `q_k`.
- Quantity loss must not directly train mark logits in V3b.
- V2/V3a behavior must remain the backward-compatible default.
- No new model parameters, optimizer groups, or target inputs may be added.
- RMTPP, THP, Titan memory, lookback, split, and preprocessing remain unchanged.
- V3a and V3b artifacts must never share a run directory or resume state.
- The first V3b experiment remains a TitanTPP-only architecture ablation.

## Options Considered

### Option A: Reduce `lambda_qty`

Keep the coupled gate and reduce the quantity-loss coefficient.

Advantages:
- No new gradient mode or code path.
- May reduce marker interference.

Disadvantages:
- Weakens both the harmful mark-gate gradient and the useful value-expert
  gradient at the same time.
- Does not isolate whether direct gate coupling caused the V3a trade-off.
- Risks discarding the quantity gains that motivated V3.

### Option B: Use The Ground-Truth Mark For Quantity Loss

Build the quantity prediction only from the target-mark expert during training.

```text
q_hat_train = base^(m_true + r_by_mark[m_true])
```

Advantages:
- Removes quantity gradients from predicted mark probabilities.
- Gives a simple conditional quantity target.

Disadvantages:
- Creates a train/inference mismatch because inference uses predicted marks.
- Removes uncertainty-aware expected quantity training.
- Makes quantity training optimistic under mark errors and partially duplicates
  residual supervision.

### Option C: Detach Only The Mark-Probability Gate

Keep the same forward expectation but stop gradients through the gate weights.

```text
p_k = softmax(mark_logits)_k
g_k = stop_gradient(p_k)
q_hat = sum_k g_k * q_k
```

Advantages:
- Forward values, loss values, and parameter count are identical to V3a.
- Directly tests the gradient-coupling hypothesis.
- Keeps quantity gradients for shared and mark-specific value experts.
- Keeps marker CE as the direct mark-head objective.
- Small implementation and testing surface.

Disadvantages:
- Quantity loss still changes the shared encoder through the value branch, so
  marker predictions are not fully isolated from quantity learning.
- Detached probabilities can freeze a poor gate distribution within each
  quantity-loss step; CE must improve the gate separately.
- If class imbalance is the main cause, detachment alone may not recover mark
  accuracy.

### Option D: Separate Or Freeze The Quantity Encoder Branch

Use a separate hidden projection/encoder or stop quantity gradients before the
shared encoder.

Advantages:
- Strongest isolation between marker and quantity objectives.

Disadvantages:
- Adds parameters and an additional architecture change.
- Removes useful shared-representation learning.
- Confounds the first gradient-routing ablation and increases maintenance cost.

## Decision

Use Option C: detach only the mark-probability gate inside TitanTPP's hybrid
quantity expectation.

The official variants are:

| Variant | `value_head_mode` | `qty_mark_gradient_mode` |
| --- | --- | --- |
| V2 | `shared` | `coupled` |
| V3a | `mark_conditioned_experts` | `coupled` |
| V3b | `mark_conditioned_experts` | `detached` |

### Configuration Contract

Add a backward-compatible training option:

```text
qty_mark_gradient_mode: coupled | detached
```

- Default: `coupled`
- V3b CLI: `--qty-mark-gradient-mode detached`
- The first implementation consumes `detached` only in TitanTPP.
- Official detached runs must use `--models titantpp`; non-Titan detached runs
  should fail clearly rather than silently record an ineffective option.

### Forward Contract

```text
mark_probs = softmax(logits_real)

if qty_mark_gradient_mode == "coupled":
    qty_gate = mark_probs
else:
    qty_gate = stop_gradient(mark_probs)

qty_per_mark = base^(mark_grid + value_by_mark)
expected_qty = sum_k qty_gate[k] * qty_per_mark[k]
```

For the same weights and batch:

```text
expected_qty_V3a == expected_qty_V3b
qty_loss_V3a == qty_loss_V3b
total_loss_value_V3a == total_loss_value_V3b
```

Only the gradient graph differs.

### Backward Contract

For an isolated `qty_loss.backward()`:

| Parameter/path | V3a coupled | V3b detached |
| --- | --- | --- |
| `mark_head` through mark probabilities | gradient | no gradient |
| `value_head` | gradient | gradient |
| `value_mark_delta_head` | gradient | gradient |
| encoder through `value_by_mark` | gradient | gradient |

For the full hybrid loss, `mark_head` must still receive gradients from marker
cross-entropy. Detaching the quantity gate must not freeze the mark head.

The quantity loss can still affect the shared encoder through value experts.
V3b therefore removes the direct gate gradient but does not claim complete task
decoupling.

### Implementation Boundary

TitanTPP should resolve the mode once during construction and centralize the
gate policy in a small helper used by `expected_qty_from_logits`. The caller and
loss composition should not branch on the mode.

Expected implementation surface:

- `models/RMTPPs/config.py`
- `models/RMTPPs/TitanTPP.py`
- `simple_lab_test/search/common/configs.py`
- `simple_lab_test/search/common/models.py`
- `simple_lab_test/search/common/runner.py`
- `simple_lab_test/search/common/modes/model_test.py`
- `simple_lab_test/search/tpp_experiment.py`
- focused V3b gradient-routing tests

RMTPP and TransformerHawkesTPP quantity paths are non-goals for V3b.

### Artifact Identity Contract

Persist `qty_mark_gradient_mode` in:

- experiment manifest
- run summary and cache identity
- histories
- validation/test leaderboard rows
- scale-wise and confusion metadata
- model-test summary

Keep legacy `coupled` run paths unchanged. A detached run adds a distinct path
segment after the value-head segment:

```text
.../valuehead_mark_conditioned_experts/qtymarkgrad_detached/...
```

This prevents V3a/V3b checkpoint collision while preserving existing V2/V3a
resume behavior.

### Evaluation Contract

Evaluation is unchanged:

- marker NLL and accuracy use ordinary mark logits/probabilities
- residual diagnostics gather the true-mark expert
- reconstructed quantity gathers the predicted-mark expert
- time likelihood is unchanged
- `best_val_nll` remains the primary checkpoint

Detachment is a training-gradient policy, not an inference operation.

## Assumptions

- The V2/V3a seed-42 e50 runs are comparable because dataset, split, candidate,
  initialization policy, objective coefficients, and checkpoint rule were fixed.
- Direct quantity-to-mark coupling is a plausible contributor to the observed
  marker trade-off, but it is not assumed to be the only cause.
- `softmax(logits).detach()` preserves the exact forward gate values while
  removing their autograd connection to mark logits.
- The current evaluation path does not depend on training-time gradient mode.
- If V3b fails, class imbalance or shared-encoder interference becomes a higher
  priority hypothesis than further detachment of the same gate.

## Consequences

Positive:
- Clean V3a-to-V3b ablation with identical forward values.
- No additional parameters or inference cost.
- Preserves direct quantity supervision for value experts.
- Gives marker CE exclusive direct ownership of mark-head updates.
- Keeps existing architecture and evaluation semantics stable.

Negative:
- Shared-encoder interference can remain through the value branch.
- One more training-mode field must be propagated through artifacts.
- Titan-only detached execution requires a clear runner contract.
- Detachment may recover marker metrics at the cost of some quantity gain.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Detachment accidentally removes all quantity gradients | V3b becomes residual-only in practice | Isolated gradient test for value heads and encoder |
| Mark head still receives qty gradient | Ablation is invalid | Assert zero/None `mark_head` grad from isolated `qty_loss` |
| Full hybrid loss freezes mark head | Marker learning fails | Assert non-zero mark-head grad from marker NLL/full loss |
| Shared encoder still harms marker metrics | V3b does not recover accuracy | Treat as expected residual risk; consider V3c branch projection only after V3b |
| V3a/V3b checkpoint collision | Wrong resume or overwritten results | Distinct run-path segment and cache identity check |
| Detached option silently ignored by non-Titan models | Misleading experiment metadata | Reject detached non-Titan execution in the first implementation |
| Tail gains dominate aggregate MAE | Misleading success claim | Report scale share, MAE, WAPE, median AE, and mark confusion |

## Non-Goals

- No `lambda_qty` tuning in the first V3b comparison.
- No class-balanced or ordinal marker loss.
- No separate quantity encoder or branch projection.
- No mark-conditioned time head.
- No value-expert grouping or expert-count reduction.
- No change to evaluation reconstruction or checkpoint selection.
- No full multi-seed run before the seed-42 short gate passes.

## Follow-Up Validation

### Focused Tests

1. V3a and V3b produce identical expected quantity, quantity loss, and total
   loss values for identical weights and inputs.
2. Isolated V3a quantity loss produces a non-zero mark-head gradient.
3. Isolated V3b quantity loss produces zero/None mark-head gradient.
4. Isolated V3b quantity loss produces non-zero shared value-head and populated
   mark-delta gradients.
5. V3b marker NLL/full hybrid loss still produces a non-zero mark-head gradient.
6. Shared encoder receives quantity gradients through the value branch.
7. V3b uses a distinct run directory and cache identity from V3a.
8. Shared/V3a default paths and model tests remain unchanged.
9. Detached execution with unsupported non-Titan models fails clearly.

### Short Screening Sequence

1. CPU synthetic focused tests.
2. Instacart top-20 one-epoch V3b integration smoke.
3. Intermittent `small_lmm`, e50, seed 42: V2 vs V3a vs V3b.
4. Yellow Trip Hourly `mid_lmm`, e50, seed 42: V2 vs V3a vs V3b.
5. Proceed to seeds `42,52,62` only if the short gate passes.

### Proposed Short-Gate Criteria

- No NaN, Traceback, or artifact collision.
- V3b total NLL is no worse than V2 by more than `0.5%`.
- Taxi marker NLL regression relative to V2 is reduced from `17.5%` to at most
  `2%`; matching or improving V2 is the target.
- Mark-accuracy gap relative to V2 is within `0.25%p` on both datasets.
- V3b retains at least half of V3a's aggregate quantity-MAE gain relative to
  V2.
- No quantity bucket with at least `5%` test share regresses by more than `5%`
  relative to V2.

These are screening gates, not final paper claims. Multi-seed validation remains
required before adopting V3b.
