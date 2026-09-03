# Smooth Shrinkage and Matched Constant Control v1

## Decision recorded before this comparison

Original frozen-probe results and the mechanism diagnosis are already known.
This is a prospective follow-up contract, not an unseen-validation experiment.
The current baseline is the original four seed-42 Hard-LMM checkpoints, with
train and full-validation caches from execution `474ebdd`. The encoder,
prototypes, retrieval, quantity head, time head and original probe remain frozen.

Only one adaptive candidate is tested: **Smooth Shrinkage**. A one-parameter
constant gate and exact original predictions are controls, not new backbones.
The constant is a residual multiplier, NOT the earlier Taxi +0.05 logit offset.

## Changed initialization contract

Both fitted models use `g = 1 - 0.2 * sigmoid(score)`, initial `g = 0.99`.
The adaptive MLP keeps the original features, hidden size 16 and tanh, with
zero final weight and bias `log(0.05 / 0.95)`. The scalar starts at the same
score and ignores all event features. Neither model changes the time route.

A smooth one-sided gate at an exact finite-parameter maximum cannot also
have a nonzero ordinary derivative there. Therefore we explicitly abandon
exact-identity initialization for these models. Original `g=1` is a separate
baseline/selection option, not a silently modified checkpoint. No clamp,
straight-through estimator, bidirectional amplification or auxiliary loss is used.
Sigmoid saturation remains possible and must be reported, not presumed absent.

## Matched fitting and verification

- Run on local CPU with the existing PyTorch 2.7.1 selected by
  `/usr/local/bin/python3 -s`; do not install/change runtimes or contact GPU servers.
- Verify the original source, cache, baseline-state and contract digests.
- First fit each of the eight dataset/control pairs for one full cached train
  epoch. Do not load validation caches until all preflights pass. Require finite
  numerics, a nonzero first gradient, changed parameters and changed train gates.
- Discard every preflight model and optimizer. Main fits are fresh seed-42
  initializations with the same shuffle for scalar and adaptive candidates.
- Both use direct log-MSE plus the unchanged constant Time NLL, Adam lr .001,
  batch 128, gradient clip 1, maximum 40, minimum 10, patience 10, no weight decay.
- Selection compares original identity (epoch -1), initial candidate (epoch 0),
  and trained epochs using full-validation joint objective. Strict improvement
  is required; ties retain identity or the earliest checkpoint. An identity or
  initialization-only selection does not count as a trained candidate success.
- Save histories, gradient/saturation diagnostics, selected and final parameter
  checkpoints, full quantity/history tables and event-level validation deltas.
  A selected-identity payload must restore exact original predictions.
- Verify save/load replay and input immutability. Use fresh output directories;
  fail durably rather than automatically retrying or resuming.

## Fixed decision

The original per-dataset body <=p95 MAE improvement >=5%, RMSE and >p99 MAE
regression <=2%, Time NLL increase <=.01 and finite conditions are unchanged.
The adaptive candidate must additionally beat the selected constant control's
body MAE with no greater validation joint objective to support adaptivity.
This is a conservative extra comparison, not a relaxation of the original gate.

General four-dataset support requires all four datasets to pass both checks.
Report partial passes explicitly; do not substitute a successful dataset for
the common contract. Report lower body bins separately but do not change the
primary threshold after seeing these results.

This frozen, bounded-train-sample, single-seed comparison is exploratory and
does not establish fresh-training or held-out generalization. It cannot alone
establish retrieval-specific causality without a future feature ablation.
No candidate is automatically promoted to e300, multiple seeds, or held-out test.

## Completion

Freeze code/contract on local `paper_research/master` before real-cache fitting.
Then run focused tests, train-only preflight, the eight matched fits, independent
artifact reconciliation, documentation and Notion updates, and a separate result
commit. Do not push. The cached CPU workload is expected to be short; create a
monitor only if execution actually becomes long-running rather than leaving an
unnecessary recurring automation after completion.
