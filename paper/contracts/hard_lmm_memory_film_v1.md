# Hard-LMM Memory-FiLM v1 contract

## Candidate and unchanged baseline scope

The candidate is `CountAwareTitanMemoryFiLM`; its reference is B, the existing
mark-free `CountAwareTitanTPP` with `static_hard_lmm`. The encoder width and two
layers, 16 persistent tokens per layer, one 64-by-hidden prototype bank, top-4
arithmetic-mean retrieval, final Hard-LMM addition, time and quantity heads,
losses, data split, optimizer, budget, and raw-RMSE checkpoint selector remain
unchanged. This is a backbone candidate, not a selector or output-head change.

The same prototype tensor is read at the intermediate and final locations. No
second bank, query/key/value projections, online writes, dataset conditions, or
extra input features are allowed. The candidate adds exactly `2 * hidden_dim`
trainable scalars.

## Structural hypothesis

After encoder block 1, its hidden state retrieves a residual `r` from the
existing Hard-LMM bank. Before encoder block 2, each hidden feature is changed
as follows:

```text
scale_delta = tanh(r * scale_gain)
shift       = tanh(r * shift_gain)
h_film      = h * (1 + scale_delta) + shift
```

Both gains are feature vectors. Their initial values are zero, so the complete
candidate output must exactly match a same-seed B model in evaluation mode.
The transform is bounded to keep the new multiplicative and additive routes
finite for finite inputs.

## Local implementation gates

- All inherited state-dict tensors must equal same-seed B tensors at
  construction. Loading a B checkpoint may omit only the two FiLM gain vectors.
- Identity initialization must preserve encoded states, time scores, quantity
  predictions, and target losses exactly in evaluation mode.
- The intermediate and final retrievals must share the same `lmm.mem` parameter.
- Opening either gain must change a valid hidden state for a nonzero retrieval.
- A backward pass with an open quantity head must produce finite, nonzero
  gradients for both gain vectors and finite gradients for all participating
  parameters.
- Changing future or padded input values must not change an earlier valid state.
  Every padded output and modulation diagnostic must be zero; padded retrieval
  indices must be `-1`.
- Finite nonnegative durations and quantities must produce finite hidden states,
  predictions, losses, gradients, and modulation diagnostics.

Passing these gates establishes only implementation and learning feasibility.
It is not evidence of validation or held-out performance.

## Remote evaluation boundary

Remote CUDA e1 and performance training require a runner that records this
contract, source commit, data hashes, parameter count, retrieval traces,
checkpoint selector, and held-out-test lock. They are outside this local contract
task. The candidate can be called a Backbone improvement only if it
beats B under the same loss, selector, seed, and budget; retrieval changes or
nonzero gradients alone are insufficient.
