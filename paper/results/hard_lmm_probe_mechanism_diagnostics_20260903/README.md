# Hard-LMM Probe Mechanism Diagnostics

## Status

Completed locally on 2026-09-03, 09:11:07 to 09:11:15 KST (8.25 seconds
inside the diagnostic runner, excluding interpreter/font-cache startup).
No GPU, remote server, installed package, backbone, original probe, or service
was changed. No held-out data was loaded. No new candidate was promoted.

The original eight-run experiment and final 31-test verification were committed
as `a846758` on `paper_research/master`. The new diagnostic contract and runner
are `21d5d17`, followed by `736efd8`, which ensures the constant grid includes
exact zero. The latter fix was caught with synthetic tests before real-cache
execution; 37 focused tests passed before the diagnostic.

This is a local CPU reproduction with PyTorch 2.7.1 selected explicitly using
`/usr/local/bin/python3 -s`. The original probe was fitted on CPU with server
PyTorch 2.11.0+cu130. Historical train-objective discrepancies are at most
`4.025e-8`; the reproduction is numerically close, not bit-identical, and is
not mislabeled as a recovered original per-step trace.

## 1. Shrinkage did not meaningfully train on three datasets

The original zero-initialized `1 - 0.2 * clamp(score, 0, 1)` gate was replayed
for ten fixed epochs on the same train targets with unchanged optimizer/shuffle.
Every minibatch was recorded, without validation evaluation or early stopping.

| Dataset | Nonzero-gradient batches / total | Last nonzero-gradient step | Final mean gate | Mean residual-norm reduction |
| --- | ---: | ---: | ---: | ---: |
| Intermittent | 1 / 5,120 | 1 | 1.0000 | 0% |
| Taxi | 5 / 3,000 | 5 | 1.0000 | 0% |
| RAF | 1 / 2,020 | 1 | 1.0000 | 0% |
| Instacart | 5,120 / 5,120 | 5,120 | 0.8558 | 14.42% |

Confirmed in this replay: Intermittent and RAF scores move below zero after
the first update; Taxi becomes inactive after five gradient-bearing updates.
At the end of epoch 1 and thereafter, every train event in those three datasets
has gate exactly one, zero correction and negative score. The clamp blocks
subsequent gradients. Adam momentum can still move parameters, but cannot
restore a useful gate in this trace. This explains the unchanged predictions;
it is not a valid rejection of the entire confidence-shrinkage idea.

Train-only oracle derivatives indicate that infinitesimal shrinkage would help
log-MSE on 41.60% / 38.65% / 40.20% of Intermittent / Taxi / RAF targets,
respectively. These label-based diagnostics never enter the gate. Event fractions
do not establish a learnable decision rule or an overall gain: the mean
shrinkage derivative is unfavorable for Intermittent and Taxi, and only weakly
favorable for RAF. Selectivity would matter if another gate were attempted.

Instacart is a different failure mode: its gate genuinely learns. Train log-MSE
falls from `0.249775` to `0.247822`, while train MAE rises from `3.89747` to
`3.90768` and RMSE rises from `5.66755` to `5.72658`. Its previous validation
selection returned identity. Thus it is not explained by a globally dead gate,
and lower train log-MSE does not establish the desired MAE/RMSE improvement.

The residual-norm reduction is the exact relative scaling `1-gate`; original
absolute vector norms were not cached. Instacart's mean absolute reduction in
the quantity-logit residual projection is `0.0387844`. Do not confuse these
two measurements or claim an unobserved absolute vector norm.

## 2. Taxi aggregate gains are largely constant bias correction

A single offset was chosen on Taxi train only by a fixed 1,001-point grid in
`[-0.05, 0.05]` with the original direct log-MSE. Train selects `+0.05`.
The existing MLP was not retrained or reselected.

| Full Taxi validation | Quantity MAE | Quantity RMSE | Body <=p95 MAE | Joint objective |
| --- | ---: | ---: | ---: | ---: |
| Original Hard-LMM | 51.767734 | 181.537602 | 23.167407 | 1.557602 |
| Existing selected MLP | 47.063531 | 163.508461 | 22.514055 | 1.554193 |
| Train-fitted constant +0.05 | 47.087029 | 163.508324 | 22.538685 | 1.559201 |

Time NLL is unchanged at `1.366430` in every row. The constant recovers 99.5005%
of the MLP's total absolute-error reduction and 100.0007% of its squared-error
reduction. At quantities above train p90, both corrections are near +0.05 and
their raw predictions are identical at stored FP32 precision.

This supports a simple underprediction-correction explanation for most of the
aggregate MAE/RMSE gain, rather than a demonstrated retrieval-specific gain.
It does NOT mean the MLP is constant on every event: at <=p50 it often applies
a negative correction, whereas the scalar always increases the prediction.
The MLP consequently has a better validation joint objective than the scalar.
The scalar actually worsens that official objective relative to the baseline;
it is a mechanism control, not an eligible replacement checkpoint.

The MLP was historically selected on validation, while the scalar uses a
train-only one-dimensional search rather than a budget-matched optimizer. This
asymmetry and the already-viewed validation results make the comparison
exploratory. Neither result attributes causality to individual retrieval features.

## 3. The remaining body problem is real

MLP validation MAE changes relative to baseline:

- <=p50: **0.97% worse** (4,364 events).
- p50-p90: **0.94% worse** (3,138 events).
- p90-p95: **7.09% better** (386 events).
- >p99: **13.76% better** (79 events).

The <=p95 body aggregate improves 2.82%, but that hides deterioration in the
lower two quantity bins. The 380 events above p95 (4.60% of validation) account
for 86.75% of total absolute-error reduction and 96.24% of squared-error reduction.
History >128 accounts for 99.83% of absolute-error reduction, but history and
quantity are correlated slices; these contributions must not be added together
or interpreted as a causal history effect.

## Verification and retained evidence

- All 15,260 minibatch records are finite and their epoch losses, gradient
  counts and residual scaling independently reconcile.
- Constant train-grid losses were independently recomputed with NumPy.
- Taxi validation MAE/RMSE and error sums reconcile against the original
  exported event predictions and the new scalar predictions.
- Input and output digests pass; original source and caches are unchanged.
- `final_tests.xml` records the final focused retrieval/probe/diagnostic suite.
- Full per-step traces and grid curve remain in the local ignored artifact:
  `search_artifacts/hard_lmm_probe_mechanism_diagnostics_20260903` (about 21 MiB).
- Compact epoch/stratum summaries, CSVs, provenance and verification are retained
  here. Existing raw data and held-out partitions were not accessed.

Reproduce:

```bash
/usr/local/bin/python3 -s paper/scripts/diagnose_hard_lmm_frozen_probes.py \
  --artifact search_artifacts/hard_lmm_frozen_probe_20260903 \
  --output /tmp/hard_lmm_mechanism_fresh_output
```

## Next decision

Keep both current candidates unpromoted and retain the original body/tail gates.
If continuing, define a separate gate parameterization contract and first prove
that it learns on real train features without collapsing; preserve the scalar
control when assessing any new calibration module. No gate redesign, new loss,
fresh e300, seed expansion or held-out evaluation was executed in this task.

Notion: https://app.notion.com/p/3cfbbe4056138105aab2f39e3dd99749
