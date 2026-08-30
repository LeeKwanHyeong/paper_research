# Count-aware TitanTPP-MAC Semantic Optimization Result

## Scope

This is an RTX 5080 execution audit of the frozen TitanTPP-MAC equations. The
adapter omits unused write diagnostics from the standard training forward and
uses a fixed batch-128, chunk-16 compiled state-only recurrence. It does not
change model parameters, heads, losses, checkpoint selection, segment size,
predict-before-write order, or held-out data status.

## Semantic and checkpoint acceptance

- Maximum prediction/loss difference: `1.1921e-7`.
- Maximum gradient difference: `2.3842e-7`.
- Frozen B1 checkpoint strict-load state digest:
  `00097caf2610b4a10b6f27458035ab5d4f435153ded98a81d02745a221e03c0d`.
- Checkpoint save/restore prediction difference: `0.0`.
- Parameter keys are unchanged.
- Held-out test data were not evaluated.

All semantic thresholds in the optimization contract pass.

## RTX 5080 cost

| Sequence length | B0 steady step | B1 steady step | B1/B0 | B1 cold step | B1 peak allocated |
|---:|---:|---:|---:|---:|---:|
| 64 | 3.316 ms | 26.499 ms | 7.990x | 40.689 s | 898 MiB |
| 84 | 3.577 ms | 39.959 ms | 11.172x | 42.728 s | 1,305 MiB |
| 256 | 15.744 ms | 107.408 ms | 6.822x | 41.627 s | 3,467 MiB |

The 3x compute guardrail is not met at any required sequence length. Fixed
shapes prevent final-batch/chunk recompilation and the state-only graph preserves
semantics, but the sequential segment-wise MAC and neural-memory recurrence
remain the dominant steady cost.

## Rejected full-forward probe

Compiling the bound standard encoder forward at length 64 reduced B1 steady
time from 26.499 ms to 20.103 ms, a 24.14% improvement, but the ratio remained
6.062x B0. Cold compile rose to 204.77 seconds and peak allocation to 1,406 MiB.
The frozen position-bound check also caused a graph break. Removing that safety
boundary cannot plausibly supply the remaining greater-than-2x reduction, so
this probe is not integrated.

## Decision

The semantics-preserving adapter is implemented and verified, but it is not an
accepted basis for the official three-seed expansion because the frozen 3x cost
gate fails. Seeds 52 and 62 remain unseen under the primary-candidate contract.
Meeting the gate now requires a dedicated fused recurrence kernel or a revised
compute contract; an architecture or equation change would be a new model
ablation rather than this optimization pass.
