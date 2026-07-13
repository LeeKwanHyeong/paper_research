# ADR: TitanTPP V3c Detached Value-to-Encoder Route

- Date: 2026-07-12
- Status: Implemented; focused validation passed
- Scope: Intermittent-specific TitanTPP shared-encoder gradient routing
- Method: Design-Twice followed by ADR

## Context

V3a introduced mark-conditioned residual experts. V3b kept the same forward
quantity estimate and detached only the predicted-mark probability gate from
the quantity loss. The remaining V3b auxiliary gradient path is:

```text
L_value -> value heads -> shared Titan encoder
L_qty   -> value heads -> shared Titan encoder
```

The direct quantity path into `mark_head` is already blocked in V3b:

```text
L_qty -X-> detached mark probabilities -X-> mark_head
```

The matched seed-42 e50 Intermittent artifacts use the same `small_lmm`, fixed
split, learning rate, batch size, lookback, maximum sequence length, hybrid
loss, residual input, and target-only training scope.

`best_val_nll` held-out test results are:

| Variant | Total NLL | Marker NLL | Quantity MAE | Value MAE | Mark accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 | `5.071916` | `1.016321` | `3.528298` | `0.153685` | `54.460%` |
| V3a | `5.062077` | `1.006122` | `3.058517` | `0.117189` | `53.437%` |
| V3b | `5.058310` | `1.004198` | `3.463607` | `0.150965` | `53.367%` |

V3a and V3b improve total and marker NLL relative to V2 but reduce mark
accuracy by `1.023%p` and `1.093%p`, respectively. Detaching the direct
quantity-to-mark gate did not recover Intermittent mark accuracy. This does not
prove shared-encoder conflict, but it makes the remaining value-to-encoder path
the next isolated hypothesis.

The dominant `1-9` held-out quantity bucket has `88.67%` share. Its MAE changes
relative to V2 are `+9.21%` for V3a and `+3.64%` for V3b. V3c must therefore
guard both marker accuracy and the dominant low-quantity bucket.

## Problem

Determine whether gradients from the residual and expected-quantity objectives
alter the shared Titan representation in a way that harms Intermittent mark
classification. The experiment must isolate this route without changing model
parameters, forward predictions, loss values, inference, data, or the
probabilistic time head.

## Constraints And Quality Attributes

- V3b and V3c must have identical parameters and initialization.
- Identical weights and inputs must produce identical hidden states, logits,
  residual predictions, quantity estimates, and scalar loss values.
- Only the backward graph may differ.
- Marker NLL and time NLL must continue to train the Titan encoder.
- Value and quantity losses must continue to train the shared value head and
  populated mark-delta experts.
- Value and quantity losses must not train `mark_head` or the Titan encoder in
  V3c.
- The historical value input remains available to the encoder; V3c changes
  gradient ownership, not input information.
- V2, V3a, and V3b behavior and run paths remain backward-compatible defaults.
- V3c artifacts cannot share a run directory or resume state with V3b.
- No additional inference cost, model parameters, optimizer groups, or target
  inputs may be introduced.
- Taxi remains on V3b; V3c is an Intermittent-specific experiment.

## Options Considered

### Option A: Fully Detach The Value-Branch Hidden State

Use the ordinary hidden state for marker/time heads and a stopped-gradient view
for the value branch.

```text
h_main = h_j
h_value = stop_gradient(h_j)

mark_logits = mark_head(h_main)
time_density = time_head(h_main)
value_by_mark = value_heads(h_value)
```

Advantages:

- Exact V3b/V3c forward and loss-value equivalence.
- No parameter, inference, optimizer, or data-path change.
- Directly isolates all value/quantity gradients at the encoder boundary.
- Small implementation and focused-test surface.

Disadvantages:

- Removes useful value representation learning from the encoder as well as
  potentially harmful gradients.
- Value heads must follow an encoder representation trained only by marker and
  time likelihood.
- A failed accuracy result would leave class imbalance and checkpoint
  volatility as stronger hypotheses.

### Option B: Scale The Value-to-Encoder Gradient

Use a straight-through gradient scale `alpha` while preserving forward values.

```text
h_value = stop_gradient(h_j) + alpha * (h_j - stop_gradient(h_j))
```

Advantages:

- Can trade marker isolation against quantity representation learning.
- Still adds no inference cost or model parameters.

Disadvantages:

- Adds a continuous hyperparameter and a sweep before the causal hypothesis is
  established.
- A positive result cannot distinguish route removal from coefficient tuning.
- Increases artifact and reporting complexity.

### Option C: Add A Detached Value Adapter

Feed `stop_gradient(h_j)` through a trainable projection before the value
experts.

Advantages:

- Isolates the encoder while giving the value branch extra adaptation capacity.

Disadvantages:

- Changes parameter count and initialization together with gradient routing.
- Confounds the first V3c ablation and adds inference cost.

### Option D: Apply PCGrad Or Another Multi-Task Optimizer

Compute separate task gradients and project conflicting encoder components.

Advantages:

- Preserves compatible auxiliary gradients instead of removing all of them.

Disadvantages:

- Requires multiple backward passes or custom optimizer logic.
- Changes training cost and optimizer semantics.
- Is difficult to validate before confirming that encoder-gradient conflict is
  the actual cause.

## Decision

Use Option A for the first V3c implementation: detach the hidden state only at
the value-branch boundary during loss construction.

Official variant identity becomes:

| Variant | `value_head_mode` | `qty_mark_gradient_mode` | `value_encoder_gradient_mode` |
| --- | --- | --- | --- |
| V2 | `shared` | `coupled` | `coupled` |
| V3a | `mark_conditioned_experts` | `coupled` | `coupled` |
| V3b | `mark_conditioned_experts` | `detached` | `coupled` |
| V3c | `mark_conditioned_experts` | `detached` | `detached` |

### Configuration Contract

Add a backward-compatible training option:

```text
value_encoder_gradient_mode: coupled | detached
```

- Default: `coupled`
- V3c CLI: `--value-encoder-gradient-mode detached`
- `detached` is accepted only for TitanTPP-only runs.
- The first V3c contract also requires
  `value_head_mode=mark_conditioned_experts` and
  `qty_mark_gradient_mode=detached`.
- Unsupported combinations fail before training rather than recording a
  partially effective variant.

### Forward Contract

Inside `TitanTPP.nll`, resolve the value-branch hidden state once:

```text
if value_encoder_gradient_mode == "coupled":
    h_value = h_j
else:
    h_value = stop_gradient(h_j)

logits = mark_head(h_j)
logf_dt = time_density(h_j)
value_by_mark = predict_value_by_mark(h_value)
expected_qty = expected_qty_from_logits(logits_real, value_by_mark)
```

The routing policy applies only while constructing training losses. Public
inference helpers continue to consume normal encoder hidden states.

For identical weights and inputs:

```text
hidden_V3b == hidden_V3c
mark_logits_V3b == mark_logits_V3c
value_by_mark_V3b == value_by_mark_V3c
expected_qty_V3b == expected_qty_V3c
all_loss_values_V3b == all_loss_values_V3c
```

### Backward Contract

For isolated losses:

| Parameter/path | Marker NLL | Time NLL | Value loss | Quantity loss |
| --- | --- | --- | --- | --- |
| `mark_head` | gradient | none | none | none |
| time parameters | none | gradient | none | none |
| shared value head | none | none | gradient | gradient |
| mark-delta experts | none | none | selected experts | populated experts |
| Titan encoder/LMM/input embeddings | gradient | gradient | no gradient | no gradient |

For the full hybrid objective, the encoder remains trainable through marker and
time likelihood, while the value heads remain trainable through value and
quantity losses. V3c does not freeze the model or remove quantity supervision.

### Implementation Boundary

Expected implementation surface:

- `models/RMTPPs/config.py`
- `models/RMTPPs/TitanTPP.py`
- `simple_lab_test/search/common/configs.py`
- `simple_lab_test/search/common/models.py`
- `simple_lab_test/search/common/runner.py`
- `simple_lab_test/search/common/modes/model_test.py`
- `simple_lab_test/search/tpp_experiment.py`
- `simple_lab_test/search/tests/test_titantpp_mark_conditioned_value_head.py`

The loss composer remains unchanged. TitanTPP owns the hidden routing policy so
training and evaluation callers do not branch on V3c.

### Artifact Identity Contract

Persist `value_encoder_gradient_mode` in:

- experiment manifest and checkpoint config
- run summary and cache identity
- histories and validation/test leaderboard rows
- scale-wise and confusion metadata
- model-test output and paper report grouping

Legacy `coupled` paths remain unchanged. V3c adds:

```text
.../valuehead_mark_conditioned_experts/qtymarkgrad_detached/
valueencgrad_detached/...
```

## Consequences

Positive:

- Clean V3b-to-V3c backward-only ablation.
- No parameter count, inference latency, or forward prediction change.
- Marker/time objectives receive exclusive ownership of encoder updates.
- Quantity and residual objectives continue to train the conditional experts.

Negative:

- Quantity performance may fall because the encoder no longer adapts to value
  prediction.
- The value heads see a moving representation produced by another objective.
- One more training mode must be propagated through artifact schemas.
- A seed-42 result cannot establish general stability.

## Implementation Outcome

The backward-only routing and experiment identity contract were implemented on
2026-07-12.

- `value_encoder_gradient_mode=coupled|detached` is available in model config,
  experiment config, long-epoch CLI, and model-test CLI.
- V3c routes only the value branch through `h_j.detach()`.
- Run paths use `valueencgrad_detached`; cache, checkpoint, history, test,
  scale-wise, confusion, and report grouping metadata include the new mode.
- Invalid detached combinations fail before training or model-test execution.
- Focused V3a/V3b/V3c tests passed: `18/18`, including the static-LMM
  auxiliary/marker gradient boundary.
- V3c CPU `small_lmm` model-test passed and recorded all three variant fields.
- Default coupled RMTPP, TitanTPP, and THP model-tests all passed.
- No 5090 integration smoke or dataset training was run in this implementation
  step.

## Assumptions

- The V2, V3a, and V3b seed-42 e50 artifacts are comparable because their
  dataset, fixed split, candidate, seed, optimizer settings, sequence settings,
  input features, objective coefficients, and checkpoint rule are matched.
- `h_j.detach()` preserves exact tensor values while removing only the autograd
  connection to upstream encoder components.
- Marker and time objectives provide enough signal to keep the shared encoder
  trainable during V3c screening.
- Value heads can continue to learn from detached hidden features even though
  those features change as marker/time training updates the encoder.
- Shared-encoder interference is a testable contributor, not an established
  cause of the Intermittent mark-accuracy gap.
- A seed-42 short result is a screening signal only; multi-seed confirmation is
  required before selecting V3c.

## Risks And Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Detach is applied to `h_j` globally | Marker/time encoder training stops | Use a separate `h_value`; focused marker/time gradient tests |
| Value heads accidentally receive no gradient | V3c cannot learn quantity | Isolated value/quantity gradient assertions |
| Quantity loss still reaches encoder through another path | Ablation is invalid | Assert zero encoder, embedding, and LMM auxiliary gradients |
| V3c reuses V3b checkpoint path | Wrong resume or overwritten artifact | Add `valueencgrad_detached` and cache identity check |
| Mark accuracy does not recover | Shared-encoder conflict is not supported as the next primary route | Stop further detachment; prioritize imbalance/ordinal marker analysis |
| Mark accuracy recovers but quantity regresses | Full isolation is too strong | Consider a later partial-gradient variant only after recording V3c result |
| Single-seed noise drives the result | False promotion | Require seed-42 short gate before seeds `42,52,62` confirmation |

## Non-Goals

- No change to Taxi V3b.
- No class-balanced, focal, or ordinal marker loss.
- No `lambda_value`, `lambda_qty`, or `lambda_dt` tuning.
- No projection, adapter, separate encoder, or additional expert parameters.
- No PCGrad, GradNorm, optimizer-group, or scheduler change.
- No time-head, Titan memory, lookback, split, preprocessing, or checkpoint-rule
  change.
- No full multi-seed execution before the seed-42 gate passes.

## Follow-Up Validation

### Focused Tests

1. V3b and V3c state dictionaries and initialization are identical.
2. V3b and V3c forward outputs and every scalar/tensor loss output are exactly
   equal for identical inputs.
3. Isolated V3c `value_loss` gives non-zero value-head gradients and zero
   encoder/embedding/LMM gradients.
4. Isolated V3c `qty_loss` gives non-zero value-head gradients and zero
   mark-head/encoder/embedding/LMM gradients.
5. Isolated marker NLL gives non-zero mark-head and encoder gradients.
6. Isolated time NLL gives non-zero time-head and encoder gradients.
7. Full hybrid loss gives non-zero marker, time, encoder, and value-head
   gradients.
8. V3c uses a distinct run path and cache identity.
9. V2/V3a/V3b defaults remain unchanged.
10. Unsupported V3c CLI combinations fail before a run directory is reused.

### Experiment Sequence

All experiment execution uses 5090 until the user changes the server policy.

1. Local CPU focused gradient-routing tests.
2. 5090 synthetic CPU/GPU `model-test` for V3c.
3. 5090 Instacart top-20 e1 integration smoke.
4. 5090 Intermittent `small_lmm`, seed 42, e50 short screening.
5. Run Intermittent seeds `42,52,62` e50 only if the short gate passes.

### Seed-42 Short Gate

Use the `best_val_nll` held-out test and the matched V2 e50 baseline.

- No NaN, Traceback, ERROR, or artifact collision.
- Mark accuracy gap relative to V2 is no worse than `-0.25%p`.
- Total NLL is no worse than V2 by more than `0.5%`.
- Marker NLL is no worse than V2 by more than `2%`.
- Quantity MAE is no worse than V2 by more than `2%`; matching or improving
  V2 is the target.
- Value MAE is no worse than V2 by more than `2%`.
- No quantity bucket with at least `5%` test share regresses by more than `5%`
  relative to V2.
- Validation and held-out test must agree on the direction of marker accuracy
  and quantity safety before multi-seed promotion.

### Decision Rules

- Marker gate and all safety gates pass: promote V3c to Intermittent multi-seed.
- Marker gate passes but quantity safety fails: keep V2 and consider a later
  partial-gradient design; do not promote full-detach V3c.
- Marker gate fails: do not continue encoder detachment; lower this hypothesis'
  priority and move to marker imbalance/ordinal-objective analysis.
- Quantity improves but marker gate fails: keep V2; quantity gain alone is not
  sufficient for Intermittent V3c promotion.
