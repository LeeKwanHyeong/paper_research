# TitanTPP M0 Log-Domain Reclassification And RevIN Scope Correction

Notion target:

- `5. Model Design Enhancement > 2026-07-13 | Direct Magnitude Regression과 RevIN Track`
- Update the existing M0 screening and direct-magnitude design pages.
- Update the parent history and current-decision summary.
- Do not create a duplicate page under `2. Confirm and Refine Topic`.

## Reason For Correction

The completed M0 experiment was described as a prerequisite for the whole RevIN
track. That scope is too broad. Canonical RevIN computes instance statistics on
the model input feature itself. M0 instead reconstructs `log2(qty)`, applies fixed
train-global statistics, and maps the output through `exp2`. It therefore does not
contain instance normalization and is not a direct test of raw-quantity RevIN.

## Current Execution Path

- `TitanTPP` instantiates `MemoryEncoder` directly rather than calling the standalone
  Titan forecasting wrapper that owns `use_revin` and `RevIN(norm/denorm)`.
- `magnitude_norm_mode=global` uses fixed train-only mean and standard deviation.
- The normalized feature and direct target are `z=log2(qty)`, reconstructed as
  `mark + scale_residual`.
- M0 is therefore `log-domain direct regression + train-global normalization`.
- The originally proposed M1-M4 also use `z=log2(qty)` and are log-domain variants.

## Result That Does Not Change

- M0 seed-42 e50 still fails its predeclared validation gate.
- Raw quantity MAE improves `9.791%`, while log2 quantity MAE regresses `9.700%`.
- Mark accuracy falls `3.635%p`.
- The `1-9` bucket, representing `88.666%` of validation targets, regresses
  `8.623%` in quantity MAE.
- M0 is not promoted to matched multi-seed.

## Corrected Interpretation

| Statement | Correct status |
| --- | --- |
| M0 is a RevIN experiment | Incorrect; M0 uses fixed global statistics |
| M0 rejects direct log2 regression with global normalization | Supported |
| M0 rejects log-domain M1-M4 under the original prerequisite gate | Supported as a design stop |
| M0 proves RevIN is ineffective for quantity forecasting | Not supported |
| Raw-quantity RevIN has been tested in TitanTPP | False; still untested |

The M0 artifact is retained as a **log-domain negative ablation**. The current
log-domain M1-M4 branch remains stopped under its original dependency rule, but
this decision must not be generalized to a raw-quantity RevIN branch.

## Raw-Domain Track Boundary

The raw-domain comparison now has a separate contract and acceptance gate; raw
audit, implementation, and model performance remain untested:

| Candidate | Target / input domain | Normalization role |
| --- | --- | --- |
| Q0 | raw `qty` | train-global baseline |
| Q1 | raw `qty` | causal masked canonical RevIN |
| Q2 | raw `qty` | causal shrinkage RevIN for short contexts |
| L0 | `log2(qty)` | completed M0 negative ablation |

Q1 is the canonical-method check. Q2 is a separate practical candidate because
the train-only audit found context median `3`, `67.63%` with at most four events,
and `35.23%` zero-variance contexts. These facts motivate Q2 but do not establish
its model performance.

## Corrected Decision

- Keep M0 as `L0`, a log-domain negative ablation.
- Do not run M0 matched multi-seed.
- Do not activate the existing log-domain M1-M4 branch.
- Reopen only the methodological question of raw-domain RevIN.
- Use the completed Q0/Q1/Q2 validation contract and finish the raw train-only
  audit before implementation or execution.
- Keep V5b class-prior correction as a separate fallback, not as the only remaining
  model-enhancement path.
