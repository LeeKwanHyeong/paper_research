# Separate-Key Retrieval 5090 Seed42 e300 Result

Status: **complete and locally audited; not accepted for expansion**.
Repository/branch: local `paper_research/master`. This result does not change
the model, loss, selector, threshold, or held-out lock.

## Contract and execution

- Candidate: `titantpp_key_value_static_memory`, role
  `t0_key_value_static_retrieval`.
- Official reference: frozen original Hard-LMM at
  `c5fc6bc6b271f52324ea3090a6edf15c265b9251`.
- Execution source: `2fc49bcb3aef88fd6b14cbad89825ee8ee4bc018`;
  source manifest SHA-256
  `9996f470856fd03401a28ce568f84642a0d032dc3e2ba6308c2d885b40c4527f`.
- Server/runtime: `5090`, Python 3.12.13, PyTorch 2.11.0+cu130,
  CUDA 13.0, Polars 1.39.3.
- Scope: Taxi then Instacart, seed 42, fresh initialization, maximum 300,
  minimum 40, patience 40, batch 128, learning rate 0.001, gradient clip 1.
- Objective: direct log1p quantity MSE, no tail loss, common
  `legacy_clamped_rmtpp` time head, earliest minimum validation joint objective.
- Started 2026-09-04 21:29:24 KST and completed 2026-09-05 00:13:24 KST.
  Taxi stopped at epoch 46 with best epoch 6; Instacart stopped at epoch 66
  with best epoch 26.

## Validation result

| Dataset | Model | Body <=p95 MAE | Quantity MAE | Quantity RMSE | >p99 MAE | Time NLL | Gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Taxi | Original Hard-LMM | 23.167407 | 51.767732 | 181.537594 | 1166.282421 | 1.366430 | Reference |
| Taxi | Separate-key | 16.966216 | 33.836108 | 119.390595 | 696.404859 | 1.365848 | Pass |
| Instacart | Original Hard-LMM | 3.424546 | 4.026535 | 5.974160 | 23.000999 | 3.206440 | Reference |
| Instacart | Separate-key | 3.422799 | 4.036542 | 5.996654 | 23.312413 | 3.206540 | Fail body |

Taxi improves body MAE by **26.77%**, overall MAE by **34.64%**, RMSE by
**34.23%**, and >p99 MAE by **40.29%**; Time NLL also decreases by 0.000582.
All prospective gates pass. The earlier weighted-only Taxi control improved body
MAE by only 0.59%, so the separate-key run provides a strong seed-42 signal that
untied search keys matter on Taxi rather than similarity weighting alone.

Instacart improves body MAE by only **0.05%**, below the fixed 5% requirement.
Overall MAE, RMSE, >p99 MAE, and Time NLL worsen by 0.25%, 0.38%, 1.35%, and
0.000100 respectively. The latter three remain inside their guardrails, but the
mandatory body gate fails. The candidate therefore passes **1/2 datasets** and
is not expanded to more seeds or datasets under the frozen rule.

## Audit

- Both server audits and independent local audits passed. Candidate checkpoint,
  last-best state, history, optimizer tensors, summary metrics, and CSV metrics
  are finite.
- Every epoch contains the full training population: Taxi 38,393 targets and
  Instacart 1,991,192 targets. Validation strata sum to 8,268 and 503,733.
- History is fresh and consecutive. The first legal patience stop and earliest
  validation-joint best selection were reproduced exactly.
- Checkpoint replay is exact on CPU. Keys moved away from values with norms
  2.096357 for Taxi and 10.848063 for Instacart, confirming the new search path
  was active rather than remaining at initialization.
- No held-out test artifact exists and `held_out_test_evaluated` is false.
- Launcher and run logs contain no Traceback, CUDA OOM, floating-point error, or
  runtime error. The server source manifest had no hash mismatches.
- Ten quantity/history/run-summary CSV files contain no non-finite numeric value.
- This runner does not generate plots; no plot is claimed as inspected.

The local checkout does not retain six package-local `references/` copies from
the isolated server snapshot. Their exact launch contract, summary, checkpoint
file, and checkpoint-state digests were instead verified against the canonical
frozen registry artifacts by `audit_run`; all other packaged source files match.

## Decision and limits

This is a useful **dataset-specific Taxi signal**, not evidence of a generally
better Hard-LMM backbone. The prospective cross-dataset gate was fixed before
training and is not relaxed after seeing Instacart. Instacart has no completed
weighted-only e300 control, so its result cannot isolate key/value separation
from similarity weighting. The experiment covers two validation datasets and
one seed, with no held-out test or statistical inference.

Structured evidence is in `final_audit.json`, `validation_comparison.json`, and
`validation_metrics.csv`. The synchronized raw artifact is ignored under
`search_artifacts/hard_lmm_key_value_seed42_e300_5090_20260904`.

## Remaining order

1. Keep the original Hard-LMM as the current general baseline; do not expand
   this candidate automatically.
2. If the Taxi-specific mechanism is worth pursuing, first define a new
   cross-dataset hypothesis that explains the Instacart null result. Any new
   model or training requires a separate prospective contract and approval.
3. Additional seeds, datasets, or held-out evaluation remain locked.

Notion: https://app.notion.com/p/3d1bbe40561381c2a67afd4e5da20c67
