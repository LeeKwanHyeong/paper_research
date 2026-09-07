# Hard-LMM inter-layer memory candidate v1

## Purpose

This candidate tests one backbone hypothesis: allowing the existing Hard-LMM
prototype read to affect the second encoder layer can improve the learned
history representation compared with reading the prototypes only after both
encoder layers.

## Frozen comparison

- Control: B, the original mark-free static Hard-LMM with the validation raw
  quantity RMSE selector.
- Candidate: the same model with one shared-bank read after encoder layer 1.
- Both use the same input features, two encoder layers, 16 persistent tokens
  per attention block, 64 Hard-LMM prototypes, top-4 mean aggregation, final
  Hard-LMM read, time and quantity heads, objectives, data splits, optimizer,
  training budget, selector, and early stopping rule.
- The candidate adds one scalar parameter.  It does not add a prototype bank,
  prediction head, auxiliary loss, dataset branch, or online memory update.

## Candidate route

For first-layer state `u`, shared prototype read `R_M`, and second encoder layer
`E_2`, the candidate computes

```
u = E_1(x)
u_tilde = u + tanh(alpha_raw) * R_M(u)
h = E_2(u_tilde)
z = h + R_M(h)
```

`alpha_raw` is initialized to zero.  Therefore the effective alpha is exactly
zero and the candidate reduces to B at initialization.  The bounded gate keeps
the inter-layer residual coefficient in `(-1, 1)`.

## Local contracts

- With tied common state and matching dropout RNG, alpha zero must preserve B
  outputs and gradients for every common parameter exactly.
- The alpha gradient must be finite and nonzero on a non-degenerate batch.
- Opening alpha must alter the second-layer input and final prediction while
  retaining the original final Hard-LMM read.
- Changing a future valid event must not alter earlier states.
- Padded states must remain exactly zero, and extreme finite inputs must retain
  finite outputs and gradients.
- Loading a B checkpoint uses `strict=False`; the only permitted missing key is
  `interlayer_alpha_raw`, and there may be no unexpected keys.

These checks establish initialization, causality, and trainability only.  They
do not establish a performance improvement.

## Remote evaluation boundary

GPU execution is a later step.  The 5090 run must begin with CUDA/full-data e1,
then compare the fixed candidate with B under the same raw-RMSE checkpoint rule.
Performance adoption requires real-data results and is not decided by local
contract tests.
