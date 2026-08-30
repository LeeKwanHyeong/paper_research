# Count-aware TitanTPP-MAC Primary Contract v1

## Status and naming

The paper model is **Count-aware TitanTPP-MAC**. Its implementation backbone
remains `titantpp_titans_mac` and its historical B0/B1/B2 identifier remains
`B1`, so existing seed-42 artifacts stay reproducible. The earlier B0/B1/B2
screening contract remains immutable; this contract changes B1's forward-looking
status from reference-only to the primary candidate.

This decision used only seed-42 validation results from Intermittent, Taxi, and
RAF. The acceptance rule below is frozen before seed 52, seed 62, or held-out
test results are inspected.

## Causal memory boundary

- A prediction is completed before the corresponding observed event is written.
- Only valid observed history events can update neural memory. Padding, the
  current target, and future targets never write.
- An explicit online memory state belongs to one series. Rows whose series ID
  changes are reset before memory is read, and state is never shared across
  unrelated validation or test series.
- Persistent memory is an outer-loop task parameter. It is fixed during
  validation and test; only long-term neural-memory state follows observations.

## Shared T0 training contract

Inputs are mark-free observed time and quantity histories. Quantity uses direct
MSE on `log1p(quantity)` with weight 1 and no tail auxiliary loss. Time uses the
common `legacy_clamped_rmtpp` head. Checkpoints are selected only by minimum
validation joint objective. Dataset-specific context length is allowed, but
memory hyperparameters are shared. Held-out test remains locked until the
validation contract and final model are frozen.

## Pre-registered three-seed acceptance rule

The paired validation comparison uses seeds 42, 52, and 62 on Intermittent,
Taxi, RAF, and Instacart. The primary metric is MAE at or below each dataset's
train-only p95 boundary.

- Macro-average relative body-MAE improvement must be at least 5%.
- At least three of four dataset means and eight of twelve dataset-seed pairs
  must improve; no dataset mean may regress by more than 5%.
- Macro overall MAE, RMSE, and above-p99 MAE may not regress. Per-dataset RMSE
  and above-p99 MAE regressions are capped at 15% and 25%, respectively, and
  Time NLL may regress by at most 0.01.
- A hierarchical paired bootstrap interval is reported but is not a hard gate
  with only twelve paired observations.
- Every value must be finite, artifacts and contract digests must match, held-out
  test must remain unused, and optimized epoch cost must be at most 3x B0 on
  each official dataset under the same device and training contract.

Count-aware TitanTPP-MAC is selected only when the primary rule and every
guardrail pass. A failure triggers architecture or positioning review rather
than threshold or dataset-specific hyperparameter changes.
