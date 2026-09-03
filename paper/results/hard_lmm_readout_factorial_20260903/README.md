# Frozen Hard-LMM readout capacity and objective diagnostic

## Decision

**Existing cached information can support better body predictions on Intermittent,
but no registered cell passes the unchanged body/tail gate on all four datasets.**
No original model, official T0 objective/selection rule, or benchmark is replaced.
No fresh backbone training, seed expansion, server change, or held-out evaluation
was performed or authorized by this diagnostic.

There were 24 fits, not 48 independent experiments: each trajectory has two
predeclared checkpoint selections. Joint selection gives **0/24 gate passes**;
diagnostic body selection gives **3/24**, all on Intermittent. The three are
constant/log-MSE, linear/raw-MAE, and MLP/log-MSE. No single cell passes all datasets.

## Fixed-cell comparison

The following is the **same linear/raw-MAE/body-selected cell** on all datasets,
not a mixture of per-dataset winners. Body means quantity at or below train p95;
tail means strictly above train p99. Negative changes indicate improvement.

| Dataset | Original body MAE | Readout body MAE | Body change | Original RMSE | Readout RMSE | p99 MAE change | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Intermittent | 0.603369 | 0.488757 | -18.995% | 1.722404 | 1.637333 | -1.180% | Exploratory pass |
| Taxi | 23.167407 | 22.284781 | -3.810% | 181.537602 | 126.762615 | -56.294% | Body gain below 5% |
| RAF | 4.225942 | 4.225942 | 0.000% | 36.123388 | 36.123388 | 0.000% | Identity epoch 0 |
| Instacart | 3.424546 | 3.424546 | 0.000% | 5.974160 | 5.974160 | 0.000% | Identity epoch 0 |

Time NLL is exactly unchanged by construction. For Intermittent this cell selects
epoch 9, overall MAE 0.645117 (original 0.762085), and p99 MAE 5.996411
(original 6.067986). It changes both the training objective and checkpoint selector;
it is **not an improvement under the unchanged official T0 contract**.

## What the factorial separates

### Intermittent: readout information and selection both matter

| Head / training objective | Selector / epoch | Body change | RMSE change | p99 MAE change | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| Constant / log-MSE | Joint / 0 | 0.000% | 0.000% | 0.000% | Fail |
| Constant / log-MSE | Body / 26 | -11.020% | -0.369% | -0.633% | Pass |
| Linear / log-MSE | Joint / 19 | -17.584% | +0.042% | +11.081% | Fail |
| Linear / raw-MAE | Joint / 32 | -18.203% | -2.168% | +2.887% | Fail |
| Linear / raw-MAE | Body / 9 | -18.995% | -4.939% | -1.180% | Pass |
| MLP / log-MSE | Joint / 1 | -14.054% | +17.240% | +51.705% | Fail |
| MLP / log-MSE | Body / 36 | -16.306% | -0.923% | -1.648% | Pass |
| MLP / raw-MAE | Body / 34 | -19.480% | +4.976% | +14.627% | Fail |

Even log-MSE with joint selection can improve body substantially using a linear
readout of existing information, but that checkpoint harms extreme-tail MAE.
Thus body error cannot be attributed simply to absence of cached predictive
information, nor does reducing body error automatically preserve the tail.
The unbounded constant/log-MSE control already improves body 11.02% with body
selection; the readout result must not all be credited to sophisticated memory.
The MLP has 2,273 trainable parameters versus linear 141 and scalar 1, but is not
consistently better. More head capacity alone is not the demonstrated solution.

The same linear/raw-MAE trajectory passes at body-selected epoch 9 but fails the
p99 guardrail at joint-selected epoch 32. Selection misalignment is observed, not
inferred from different models. Reporting only the successful selector would hide
an essential contract change. These are known-validation observations, not proof
that body-based selection generalizes.

### Taxi: strong tail gains are not strong body gains

Body-selected linear/raw-MAE improves body 3.810%, exceeding the same-objective
constant control's 3.129% but not the 5% gate. It reduces RMSE 30.173% and p99 MAE
56.294%. The joint-selected linear/log-MSE cell reduces RMSE 28.134% while making
body MAE 0.762% worse. The MLP/log-MSE joint checkpoint worsens body 4.350%.
Overall improvements must not be described as solving Taxi body prediction.

### RAF and Instacart: no selected body benefit in this diagnostic

Every body-selected cell retains the original epoch-0 prediction on both datasets.
This is not a dead-gradient failure: all 46,080 main-fit minibatches across the
experiment have nonzero finite gradient norm. It does not prove no better predictor
exists, only that these cells and this fixed budget did not produce one.

At epoch 40 on RAF, MLP/raw-MAE reduces **train overall MAE** from 10.689104 to
10.429426, while **train body MAE worsens 10.462%** and validation body worsens
9.059%. Optimizing global MAE is not equivalent to optimizing the body stratum;
failure here cannot be explained purely by validation overfitting.

On Instacart, MLP/log-MSE lowers train log-MSE 0.249775 to 0.245379 and train body
MAE by 0.801%, but validation body worsens 0.585%. MLP/raw-MAE lowers train overall
MAE 3.897466 to 3.849591, but validation body worsens 0.749%. The small train-body
benefit does not transfer in this cached-feature experiment. These observations do
not establish a unique root cause or prove that the encoder lacks information.

## Scope, verification, and reproducibility

- Contract: `paper/contracts/hard_lmm_readout_factorial_v1.json` and its Markdown companion.
- Implementation frozen before fits on `paper_research/master`: `5d81e0ed65ec6739e4cf47a63f02584f06da3dbe`.
- Parent frozen-cache execution: `474ebdde2598e4a2509a83d41fc8d404ca99fce0`; original 221 source files unchanged.
- Existing CPU runtime: `/usr/local/bin/python3 -s`, torch 2.7.1, one thread; no installation changes.
- Train targets: 65,536 / 38,393 / 25,779 / 65,536; full validation: 86,285 / 8,268 / 6,690 / 503,733 (Intermittent / Taxi / RAF / Instacart).
- All 24 train-only preflights passed before validation caches were loaded; their states were discarded.
- Identical initialization to original predictions, train-only standardization, fixed 40 epochs, Adam .001, batch 1024, clip norm 1; no dataset-specific tuning or retry.
- Checkpoint replay verified for 48 selected views and 24 final checkpoints; original input/source/output/checkpoint digests verified.
- Independent NumPy event/scope reconciliation passed the pre-existing absolute/relative tolerance 1e-6; maximum absolute metric discrepancy 0.000040175 on raw-scale metrics. No threshold decision is near that discrepancy.
- All metrics finite; quantity/history partitions and immutable time scores verified; no held-out artifact present.
- Local regression suite: 77 passed, including 20 new factorial tests, final exit 0.
- Fit status interval: 2026-09-03 10:04:28 to 10:05:22 KST (54.17 seconds, excluding initial imports and later independent verification). No long-running scheduler was needed.

Full checkpoints, epoch histories and event deltas are preserved in
`search_artifacts/hard_lmm_readout_factorial_20260903` (not committed as large binaries).
The compact verified result files are committed separately from implementation:

- `comparison.csv`: every cell under both selectors, not just highlighted cases.
- `factorial_contrasts.csv`: paired capacity, objective, and constant-control contrasts.
- `scope_metrics.csv`: original and candidate quantity/history metrics and counts.
- `cross_dataset_decision.json`: fixed-cell cross-dataset gate decisions.
- `artifact_verification.json` and manifests: verification and provenance.

Inputs contain normalized local and residual vectors, compressed statistics,
original logit, and memory projection. Raw hidden vectors are not rematerialized.
These are unrestricted auxiliary logit readouts, not fresh original-head refits.
The train sample is capped and validation has repeatedly informed prior decisions.
One seed and these compressed features cannot establish benchmark superiority,
unbiased generalization, or an information-theoretic limit.

## Next step, not executed

Keep the original benchmark models and contracts frozen. Do not launch another
e300 backbone modification or adopt a dataset-wise mixture of successful cells.
The supported follow-up is to preregister replication of the simple Intermittent
constant and linear readouts on the remaining frozen checkpoints, keeping loss,
selector, and tail guardrails explicit and unchanged within each diagnostic arm.
This can test seed sensitivity, not remove repeated-validation selection bias.
RAF/Instacart still require a separate explanation before claiming a universal
readout or memory fix. Final held-out test remains locked.

Notion: [Frozen Readout Capacity and Objective Diagnostic](https://app.notion.com/p/3d0bbe40561381508e5bf20e321eca31)
