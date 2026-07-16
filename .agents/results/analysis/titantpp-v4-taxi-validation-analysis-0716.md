# TitanTPP V4 Taxi Validation Analysis

- Date: 2026-07-16
- Scope: Taxi V2/V3b/V4a/V4b seed-42 e50, validation only
- Selection: each variant's `best_val_nll` checkpoint
- Source revision: `c5e9cca4241a5579ba0af655c884d6692484ba5a`
- Decision: retain V2 and Taxi V3b; do not promote V4a or V4b

## Scope And Integrity

The four histories contain exactly epochs `1..50`. Scale, class, and confusion
artifacts each reconcile to `8,268` validation targets, and their weighted
quantity MAE and mark accuracy reconcile to the selected history row. No
`test_*`, test plot, or mixed paper-table artifact exists in the experiment
root. Held-out evaluation therefore remains locked.

The comparison is a single-seed screening, not confirmation. Candidate and
control checkpoints were selected independently by validation NLL, so the
predeclared gate is the decision view while same-epoch comparisons are used to
assess whether the time-head effect is structurally persistent.

## Selected Checkpoints

| Variant | Best epoch | Val NLL | Marker NLL | Time NLL | DT MAE | Mark acc. | Qty MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | 43 | 1.595000 | 0.228486 | 1.366514 | 0.719334 | 91.630% | 95.550 |
| V3b | 29 | 1.557672 | 0.188676 | 1.368996 | 0.782815 | 92.465% | 30.872 |
| V4a | 49 | 1.582770 | 0.221934 | 1.360837 | 0.749344 | 91.981% | 70.072 |
| V4b | 33 | 1.553379 | 0.188776 | 1.364603 | 0.743973 | 92.296% | 27.029 |

The frozen pairwise gate rejects both candidates:

- V4a versus V2 improves time NLL by `0.415%`, below the required `0.5%`, and
  worsens DT MAE by `4.172%`, beyond the `1%` guardrail.
- V4b versus V3b improves time NLL by `0.321%`, below the required `0.5%`.
  Its total NLL, DT MAE, marker, accuracy, and quantity guardrails pass, but the
  primary model-axis gate does not.

## History Analysis

### V4a Versus V2

- Time NLL is better in `26/50` epochs and reaches the `0.5%` threshold in only
  `10/50`. The median gain is `0.061%`; over epochs 41-50 the mean gain is only
  `0.117%`, with `2/10` epochs meeting the threshold.
- Total NLL is better in `31/50` epochs. DT MAE is better in `29/50`, but only
  `4/10` late epochs; its late mean change is a `1.373%` regression.
- At V4a's selected epoch 49, the same-epoch V2 comparison gives a `0.414%`
  time-NLL gain. At V2's selected epoch 43, V4a is `0.322%` worse in time NLL
  and `0.708%` worse in total NLL.
- V4a reaches its minimum time NLL at epoch 49, then its final DT MAE worsens
  `8.921%` and time NLL worsens `1.065%` at epoch 50. The isolated late optimum
  is not a stable plateau.

### V4b Versus V3b

- Time NLL is better in `27/50` epochs and reaches the threshold in `10/50`.
  The median gain is `0.034%`, the 50-epoch mean is a `0.217%` regression, and
  only `1/10` late epochs reaches the threshold.
- Total NLL is better in `28/50` epochs, but the late-ten mean is a `0.118%`
  regression. DT MAE improves in only `18/50` epochs and has a `3.132%` mean
  regression over all epochs.
- At V4b's selected epoch 33, its same-epoch time-NLL gain over V3b is only
  `0.037%`; the larger own-best comparison is partly checkpoint timing. At
  V3b's selected epoch 29, V4b's time gain is `0.070%` while total NLL is
  `0.618%` worse.
- V4b's minimum time NLL occurs at epoch 49, not its NLL-selected epoch 33.
  From epoch 33 to 50, marker NLL worsens `15.008%` and mark accuracy drops
  `0.750` percentage points, showing objective tradeoff rather than a joint
  late-training improvement.

### History Conclusion

The mark-conditioned time head occasionally finds a lower conditional time
NLL, but the effect is small, epoch-sensitive, and not accompanied by a stable
deployment-style DT improvement. This is inconsistent with promotion as a
robust time-model enhancement.

## Validation Scale-Wise Analysis

Validation raw-quantity buckets contain `4,577/2,008/1,119/564` targets, or
`55.36/24.29/13.53/6.82%`, for `1-9`, `10-99`, `100-999`, and `1000-9999`.

| Pair | 1-9 | 10-99 | 100-999 | 1000-9999 | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| V4a vs V2 qty-MAE change | +0.42% | -4.85% | +20.83% | +31.33% | +26.66% |
| V4b vs V3b qty-MAE change | +8.57% | +6.65% | -2.34% | +21.41% | +12.45% |

Positive values mean lower MAE for V4.

- V4a's overall reduction is tail-driven: `76.77%` of absolute-error reduction
  comes from `1000-9999` and `23.79%` from `100-999`. The `10-99` bucket gets
  worse, so the improvement is not scale-uniform.
- V4b's `1000-9999` bucket contributes `98.11%` of total absolute-error
  reduction. The `100-999` bucket offsets `5.61%` of that gain.
- V4b reduces the `1000-9999` mean prediction bias from `-197.78` to `-18.41`,
  indicating useful tail calibration. That is a quantity-side secondary effect,
  not evidence that the primary time-head objective passed.
- Median and log errors expose further non-uniformity. For example, V4b lowers
  `10-99` mean MAE but worsens its median absolute error by `1.94%`; V4a lowers
  `1-9` mean MAE slightly while worsening both median and log absolute error.

## Confusion And Class Metrics

More than `99%` of mark errors are adjacent for all variants, so neither V4
candidate introduces a catastrophic ordinal jump pattern.

### V4a Versus V2

- Accuracy rises `0.351` percentage points (`+29` correct), with adjacent
  errors reduced by `28` and non-adjacent errors reduced by `1`.
- The gain is concentrated in mark 0: recall rises `0.961` points (`+44`
  correct). Mark 1 recall falls `0.647` points (`-13`) and mark 3 recall falls
  `0.713` points (`-4`).
- Balanced accuracy falls from `89.755%` to `89.700%`, despite higher raw
  accuracy. Predicted-share total-variation error also worsens from `0.351` to
  `0.581` percentage points.
- The main redistribution is `0->1: 216 to 173`, offset by
  `1->0: 188 to 221`. V4a favors the majority mark rather than improving all
  classes uniformly.

### V4b Versus V3b

- Accuracy falls `0.169` points (`-14` correct); adjacent errors increase by
  `16` while non-adjacent errors decrease by `2`.
- Mark 1 recall rises `0.946` points and mark 3 recall rises `1.604` points,
  but mark 2 recall falls `3.565` points (`-40` correct).
- The main mark-2 shifts are `2->1: 56 to 71` and `2->3: 33 to 59`.
  Predicted-share total-variation error worsens from `0.750` to `1.137` points.
- The mark-2 degradation is directionally consistent with the `100-999`
  quantity-bucket regression, but mark labels and raw scale buckets differ by
  three samples here. This is an association, not a per-sample causal proof.

Time loss does not directly update the mark head, but it does update the shared
Titan encoder. The class shifts are therefore compatible with indirect shared-
encoder interference; this run does not isolate or prove that mechanism.

## Plot Review

The four learning-curve plots confirm isolated early/mid-training DT spikes and
late convergence of time NLL near `1.36`. The V2/V4a and V3b/V4b curves do not
show a durable visual separation in time NLL. V3b/V4b quantity MAE is visibly
far below V2/V4a, reinforcing that the established value-head change remains
the dominant Taxi enhancement. V3b and V4b marker NLL also turn upward after
their selected checkpoints.

The scale plots correctly show raw-error dominance by the high-scale bucket,
but each file uses its own y-axis. Bar height must not be compared across
variant files; the reconciled CSV values above are the decision evidence. The
empty `>=10000` bucket and blank inactive direct-magnitude panels are expected,
but add no analytical value. Separate one-line variant plots, moving legends,
and missing best-epoch markers also limit visual attribution.

## Final Decision

V4a and V4b are not promoted. The V4 implementation remains available as an
experimental mode, but no V4 multi-seed or held-out run is justified by this
screen. V2 remains the common TitanTPP baseline and V3b remains the confirmed
Taxi-specific model. The held-out set stays locked.

Confidence is sufficient for a screening decision, with the caveat that this
is one seed. A multi-seed run would reduce uncertainty but is not warranted
after failure of the predeclared primary gate and weak full-history effect.

Reproducible outputs are generated by:

```bash
python simple_lab_test/search/analyze_titantpp_v4_taxi_validation.py
```

The generated `validation_analysis/analysis_summary.json` and pairwise CSVs are
kept under the ignored experiment artifact root.
