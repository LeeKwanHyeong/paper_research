# Count-aware Titan B0/B1/B2 Screening Contract v1

## Purpose

This contract compares only the TitanTPP backbone mechanism. B0 is the current
Hard Local Memory Matcher control, B1 is the faithful Titans-MAC reference, and
B2 is the TPP-specific gated-memory candidate. Quantity and time heads remain
unchanged so that validation differences can be attributed to the backbone.

## Shared T0 Boundary

- Inputs contain only observed event time and raw quantity history; no mark or
  future target can be written to memory.
- All variants use direct MSE on `log1p(quantity)`, `lambda_log_qty=1`, and no
  tail auxiliary loss.
- All variants use the `legacy_clamped_rmtpp` time head, hidden size 64, batch
  size 128, learning rate 0.001, gradient clipping at 1.0, and checkpoint
  selection by minimum validation joint objective.
- Dataset-specific context length is the only allowed structural difference.
- Held-out test data remains locked throughout preflight and screening.

## 5080 Preflight

CUDA model checks cover finite ordinary and extreme forward/backward passes,
model checkpoint prediction replay, and B1/B2 online memory-state continuation
after serialization. Median optimizer-step time for B1 and B2 must not exceed
three times B0 under the same synthetic batch. Intermittent, Taxi, RAF, and
Instacart then run an e1 smoke with two train and two validation batches. Every
run must produce finite validation artifacts and a restorable checkpoint.
The compiled CUDA recurrence must first match the eager reference state,
diagnostics, outputs, and gradients within the frozen numerical tolerance.

## Seed-42 Screening

Fresh B0, B1, and B2 runs use seed 42 on Intermittent, Taxi, and RAF with a
maximum of 300 epochs, minimum 40 epochs, and patience 40. Memory
hyperparameters are identical across datasets and are not changed after
viewing validation results.

For every dataset, B2 must improve validation MAE at or below the train-only
p95 quantity boundary by at least 5% relative to fresh B0. Overall quantity
RMSE and MAE above train-only p99 may regress by at most 2%, Time NLL may
regress by at most 0.01, and all metrics must be finite. B2 is selected only if
all three datasets pass. B1 is retained as a faithful reference and is not a
selectable final candidate in this gate.
