# Frozen Hard-LMM Smooth Shrinkage: Matched Scalar Comparison

## Decision

**Complete. Smooth Shrinkage passes 0/4 datasets; the scalar control passes
0/4. Do not promote either to fresh e300 or multiseed training.**
The train-only preflight passed 8/8, but learning activity did not deliver the
required body MAE improvement. This separates the previous clamp dead-zone
problem from the present negative performance result.

## Frozen Scope

- Contract and implementation were committed on local `paper_research/master`
  as `bff9722e8066b7739c809fa801cf06b83aa742b7` before real-cache fitting.
- The new contract is `paper/contracts/hard_lmm_smooth_shrinkage_v1.json`.
  Both fitted gates start at 0.99 and use `1 - 0.2 * sigmoid(score)`.
  Exact original gate 1 is a separate selection control. No hard clamp or
  bidirectional amplification was introduced.
- Adaptive gate: original 138 features, hidden 16, tanh, 2,241 parameters.
  Constant residual gate: one parameter, no features. This is NOT the earlier
  Taxi +0.05 logit-offset diagnostic. The two fits are not combined.
- Original four seed-42 checkpoints, frozen encoder/prototypes/heads,
  unchanged direct log-MSE and time route. Adam lr .001, batch 128, clip 1,
  max40/min10/patience10; identical seed and shuffle. No dataset-specific tuning.
- Reused original train caches (65,536 / 38,393 / 25,779 / 65,536 targets) and
  full validation (86,285 / 8,268 / 6,690 / 503,733) in Intermittent, Taxi, RAF,
  Instacart order. Original execution revision: `474ebdd`.
- Eight one-epoch train-only preflights preceded loading validation caches.
  All preflight parameters and optimizer state were discarded before main fits.
- Selection minimizes validation joint objective over original identity
  (epoch -1), candidate initialization (epoch 0), and trained epochs. Strict
  ties keep the earlier option. Identity/initialization do not count as trained
  candidate successes. Selecting the original is not a missing or failed run.
- Local CPU PyTorch 2.7.1 via `/usr/local/bin/python3 -s`; no package changes,
  raw data access, GPU server commands, held-out access or original model edits.

## Selected Validation Results

| Dataset | Selected role | Epoch | Quantity MAE | Quantity RMSE | Body <=p95 MAE | >p99 MAE | Joint objective |
|---|---|---:|---:|---:|---:|---:|---:|
| Intermittent | Original / scalar / smooth | -1 | 0.762085 | 1.722404 | 0.603369 | 6.067986 | -3.585706 |
| Taxi | Original / scalar / smooth | -1 | 51.767734 | 181.537602 | 23.167407 | 1166.282480 | 1.557602 |
| RAF | Original | -1 | 9.086492 | 36.123388 | 4.225942 | 313.510813 | 3.842684 |
| RAF | Constant shrinkage | 40 | 9.086723 | 36.124210 | 4.225947 | 313.519675 | 3.842665 |
| RAF | Smooth shrinkage | 21 | 9.088511 | 36.123203 | 4.228068 | 313.510166 | 3.842384 |
| Instacart | Original / scalar / smooth | -1 | 4.026535 | 5.974160 | 3.424546 | 23.001000 | 3.450390 |

Full precision and Time NLL are in `comparison.csv`; quantity and history
strata are in `scope_metrics.csv`. All selected time metrics are unchanged.
These are frozen diagnostic results, not fresh backbone training results and
not the later B0/B1/B2 screening checkpoints.

### What the gate learned

All **40,822 main-fit minibatches had nonzero finite gradient norm**; no run
reproduced the earlier zero-gradient clamp lock. This does not mean sigmoid
saturation or individual zero parameter gradients are impossible.

| Dataset | Smooth fit epochs | Selected epoch | Final train mean gate | Final mean residual reduction | Interpretation |
|---|---:|---:|---:|---:|---|
| Intermittent | 10 | -1 | 0.99999621 | 0.000379% | Learned back toward almost no shrinkage; original joint remains better. |
| Taxi | 10 | -1 | 0.99998684 | 0.001316% | Also approached the original; no selected body gain. |
| RAF | 31 | 21 | 0.87323526 | 12.6765% | Active/selective gate, but selected body MAE worsens 0.0503%. |
| Instacart | 10 | -1 | 0.86051845 | 13.9482% | Active gate; train log-MSE improves, validation does not. |

Final and selected parameters are distinct. RAF's selected epoch-21 train
mean gate is 0.878903; its final epoch-31 mean is 0.873235. Instacart's final
gate is active but was NOT selected: final validation MAE/RMSE are
4.059080/6.054761 versus original 4.026535/5.974160; joint 3.451171 versus
3.450390. Train log-MSE falls from 0.249775 to 0.247867 while train MAE/RMSE
worsen. These final-epoch observations explain the rejection, not an alternative
checkpoint selection or a new acceptance test.

On RAF, selected smooth RMSE improves only 0.000513% and >p99 MAE only
0.000206%; body MAE worsens 0.050305%. Its lower joint objective does not meet
the body target and its body MAE is worse than the matched scalar control.

## Fixed Acceptance and Limits

The parent rules remain body MAE improvement >=5%, overall RMSE and >p99 MAE
regression <=2%, Time NLL increase <=.01, finite values and a trained selected
checkpoint. Smooth must additionally beat the selected constant's body MAE
without worsening joint objective. Both original-gate and adaptive-value counts
are 0/4; thresholds were not relaxed after observing outcomes.

Interpretation: under these checkpoints, bounded train samples, direct log-MSE
and fixed optimization budget, simply shrinking the quantity memory residual
has not produced the desired body improvement. This is evidence against
promoting this candidate, NOT proof that every memory gate or fresh jointly
trained variant is ineffective. There is no empirical success probability,
fresh-training claim, multiseed significance claim or held-out generalization
claim. Previous validation results were already known when this follow-up was
designed; the experiment is exploratory, not a new confirmatory validation set.

## Verification and Reproduction

- 54 tests passed before execution; 57 pass after adding independent validator
  regression tests. These cover initialization, negative-score gradients,
  finite bounded output, feature independence of the scalar, train-only
  preflight, fresh reset, exact identity fallback, checkpoint provenance,
  forbidden/partial/corrupted cache rejection and unchanged acceptance rules.
- Independent verification: original 221 source files, own committed source,
  input/output/cache/checkpoint digests, full validation counts, train sample
  counts, all histories, minimum-joint selection and exact early stopping.
- Selected and final checkpoints replay successfully. Event predictions match
  restored selected checkpoints exactly. NumPy reaggregation of quantity/history
  metrics differs by at most **3.0223e-11**. All metrics are finite; quantity and
  history partitions reconcile; no held-out artifact exists.
- Main artifact status records 2026-09-03 **09:36:46 to 09:36:57 KST**
  (11.15 seconds after imports). This was a small cached CPU fit, not e300.
  It finished in this turn, so no recurring monitoring job was created.
- Source artifact: `search_artifacts/hard_lmm_frozen_probe_20260903`.
  New artifact: `search_artifacts/hard_lmm_smooth_shrinkage_20260903`.
  Checkpoints and event parquet remain in the artifact, not the git commit.
- Existing Notion page:
  [Hard-LMM Frozen Diagnostic](https://app.notion.com/p/3cfbbe4056138105aab2f39e3dd99749).

```sh
/usr/local/bin/python3 -s paper/scripts/run_hard_lmm_smooth_shrinkage.py --output search_artifacts/hard_lmm_smooth_shrinkage_NEW
/usr/local/bin/python3 -s paper/scripts/validate_hard_lmm_smooth_shrinkage.py --artifact search_artifacts/hard_lmm_smooth_shrinkage_20260903 --source-artifact search_artifacts/hard_lmm_frozen_probe_20260903 --output paper/results/hard_lmm_smooth_shrinkage_NEW
```

Both commands refuse an existing output destination. Do not rerun the fits to
obtain a more favorable outcome or modify the frozen contract in place.

## Remaining Work

1. **Current baseline:** retain the original Hard-LMM and frozen benchmark
   artifacts; archive smooth/scalar shrinkage as negative readout ablations.
2. **Next decision, not launched:** reassess the body-error hypothesis and model
   contribution before defining another candidate. Do not automatically start
   bidirectional gating, combine calibration and shrinkage, or expand datasets.
3. **Conditional:** only a separately authorized, prospectively frozen candidate
   that meets the unchanged body/tail conditions may proceed to fresh training.

All implementation and results are scoped to local `paper_research/master`.
No server deployment, service change or push is part of this task.
