# Frozen cached-information readout factorial, v1

## Purpose and baseline

Determine whether information already cached from the frozen Hard-LMM checkpoint
can support more accurate body quantity predictions. This is a seed-42 diagnostic,
not fresh backbone training, a benchmark replacement, or an official T0 amendment.
The original checkpoint, source files, train subset, validation set, time scores,
and train-derived quantity boundaries remain immutable and checksum verified.

## Registered cells

The main factorial is linear versus one-hidden-layer MLP readout crossed with
log-MSE versus raw quantity MAE. A constant-offset control is fitted under each
objective to distinguish information-dependent prediction from simple bias repair.
All six cells run on Intermittent, Taxi, RAF, and Instacart: 24 fits in total.

Readout inputs are the existing 138 features plus the original logit and exact
memory projection (140 scalars). These are normalized hidden vectors and compressed
norm statistics, not the original unnormalized hidden representation. No claim of
refitting the original quantity head on raw hidden states is permitted.
Standardization is fitted on the cached train subset only and frozen in checkpoints.
Target values, future events, and target-defined bins cannot enter the readout.

The original logit is retained as a skip connection. The new offset is unbounded;
constant/linear outputs and the MLP final layer initialize to zero, giving the exact
original predictions. MLP width is 16 with Tanh. Predictions retain the original
softplus then expm1 mapping. Nonfinite values fail the experiment rather than being
silently clipped. Original time scores do not depend on the probe parameters.

## Optimization and selection

Every cell uses Adam, lr .001, batch 1024, clip norm 1, seed/shuffle seed 42,
40 fixed epochs, no early stopping, and the same cached train sample (at most
65,536 events). Log-MSE and MAE use all train events with equal weight, not a
target-dependent body subset. No dataset-specific hyperparameter adjustment.

All 24 train-only one-epoch preflights must pass before validation caches are
loaded. Their state is discarded, and each main fit starts from the same identity.

Two selectors are registered for each training trajectory, with strict improvement
and earliest-epoch tie breaking over epochs 0 through 40:

- `joint`: original validation Time NLL + direct log-MSE.
- `body`: validation quantity MAE at or below the train-derived p95.

Both are always reported separately. Only the log-MSE/joint branch preserves the
official objective/selection semantics. MAE training and body selection are
explicit diagnostic departures, never silently mixed into official T0 results.
Identity selection at epoch 0 is reported as no selected improvement.

## Decision and limits

The prior gate is unchanged: body MAE improvement at least 5%, overall RMSE and
above-p99 MAE worsening at most 2%, Time NLL worsening at most .01, finite values,
and a positive trained selected epoch. Passing is exploratory only, not permission
to promote a model, expand seeds, or start fresh training. Universal support needs
the same cell to pass all four datasets, not dataset-wise cherry-picked winners.

Compare feature heads to the constant control within the same objective/selector;
compare MLP to linear for capacity and MAE to log-MSE for objective alignment.
Report overall, all quantity bins, body, tail, and history bins, including failures.
This known validation set has already informed research decisions; improvements
are not unbiased generalization estimates. One seed, a train subset, compressed
features, finite capacity, and a fixed optimization budget limit negative findings:
failure does not prove that the original encoder lacks predictive information.

## Execution

Use the existing `/usr/local/bin/python3 -s` CPU runtime (torch 2.7.1). No package
installation, remote server access, GPU work, original artifact overwrite, or
held-out data access. Freeze this contract, runner, validator, and tests in a
`paper_research/master` commit before opening caches for fits. Run:

```bash
/usr/local/bin/python3 -s paper/scripts/run_hard_lmm_readout_factorial.py --output search_artifacts/hard_lmm_readout_factorial_20260903
/usr/local/bin/python3 -s paper/scripts/validate_hard_lmm_readout_factorial.py --artifact search_artifacts/hard_lmm_readout_factorial_20260903 --output paper/results/hard_lmm_readout_factorial_20260903
```
