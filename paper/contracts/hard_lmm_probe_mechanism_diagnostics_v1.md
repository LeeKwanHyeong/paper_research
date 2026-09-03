# Frozen Hard-LMM Probe Mechanism Diagnostics v1

## Status and scope

This prospective diagnostic protocol follows the known frozen-probe results;
it is not a preregistration of the original experiment. The original eight-run
record was independently verified and committed as `a846758` on
`paper_research/master`. Original training code, checkpoints, loss, thresholds,
selection, and installed runtimes must not be modified.

Inputs are the hash-verified caches and selected calibration checkpoint under
`search_artifacts/hard_lmm_frozen_probe_20260903`, execution revision `474ebdd`.
No feature extraction, raw data loading, backbone training, GPU allocation,
held-out evaluation, or service change is authorized by this protocol.

## A. Train-only shrinkage trace

- Replay the original shrinkage MLP on the exact saved train sample for ten
  fixed epochs on CPU. Use seed/shuffle 42, Adam lr .001, batch 128, clip 1,
  zero output layer, and the original one-sided clamp gate. No early stopping,
  validation evaluation, parameterization change, or replacement checkpoint.
- Record every minibatch's score/gate distribution, gradient norm, zero-gradient
  fraction, before/after gate activity, and actual quantity-logit residual
  reduction. Record full-train quantity/history strata after each epoch.
- Relative vector-residual norm reduction is exactly `1-gate`. Absolute vector
  norms were not cached; do not claim to measure them. Absolute projected
  quantity-logit reduction is exactly `abs((gate-1)*projection)`.
- Compare replay train objectives against historical train objectives as a
  reproducibility diagnostic, not as a selection criterion. Runtime differences
  must be recorded. A local replay alone is not an original execution trace.
- Report per-event direction of the unconstrained shrinkage derivative on
  train. This is an oracle diagnostic only: next quantity never becomes a gate
  input or a feature. Separate desired shrinkage from a gate unable to move.
- Keep the original accepted/rejected decisions unchanged. No new gate is fitted.

## B. Taxi constant-calibration control

- Fit a single constant quantity-logit offset on the original Taxi train cache
  only, using direct log-MSE. Enumerate 1,001 fixed uniformly spaced offsets in
  `[-0.05, 0.05]`, including zero and the endpoints; choose minimum train loss.
  Ties prefer the offset closest to zero. No validation fitting or retuning.
- Compare original predictions, the existing selected MLP checkpoint, and this
  fixed constant on train and full validation. Time NLL remains identical.
- Report overall, pinned train-quantity bins, pinned history bins, signed bias,
  correction dispersion/bound saturation, MLP-versus-constant prediction gap,
  and disjoint-bin contributions to absolute/squared error reduction.
- The constant uses a deterministic one-dimensional grid, not a budget-matched
  MLP optimizer. The MLP was selected on validation whereas the constant is
  train-selected. This asymmetry must be disclosed; neither is a new backbone
  result, generalization proof, or causal attribution to specific features.
- If the constant explains most of the gain, prefer the simple bias-correction
  explanation. Even if MLP is better, do not infer retrieval-specific necessity
  without a separate feature ablation. Do not enlarge the .05 bound.

## Execution and completion

The branches are independent after shared input validation. Run the bounded
cache-only diagnostic in this session, without a recurring training scheduler
unless execution genuinely becomes long-running. Default verification runtime
is `/usr/local/bin/python3 -s` (PyTorch 2.7.1); the original server fit used
PyTorch 2.11.0+cu130 on CPU. Record the actual runtime and clamp boundary gradient.

Reject digest mismatch, non-finite output, reused output directories, and any
request to load a held-out cache. Preserve original artifacts byte-for-byte.
Focused tests, input/output digests, a readable analysis, Notion update, and
separate local `paper_research/master` commits complete this task. Do not push or
promote any model to fresh/multiseed training.
