# Independent audit of J/Q/T selected-checkpoint gradient results

## Status
PASS: 425/425 receipt and arithmetic checks. Only the requested JSON artifacts were read; no original data, checkpoint tensors, held-out source, model inference or training were accessed. This verifies recorded calculations and integrity links, not an independent rerun of the probe.

## Confirmed calculations

| Dataset | Model | Selected epoch | Joint clipped /16 | Q-only clipped /16 | Median encoder cosine | Median joint/Q-only clip factor |
|---|---|---:|---:|---:|---:|---:|
| intermittent_frozen_5000 | B | 77 | 0 | 0 | -0.029923 | 1.000000 |
| intermittent_frozen_5000 | candidate | 37 | 0 | 0 | -0.069911 | 1.000000 |
| yellow_trip_hourly | B | 45 | 4 | 0 | 0.039945 | 1.000000 |
| yellow_trip_hourly | candidate | 83 | 13 | 0 | 0.027845 | 0.652544 |
| insta_market_basket | B | 72 | 6 | 3 | -0.021314 | 1.000000 |
| insta_market_basket | candidate | 72 | 6 | 3 | 0.008784 | 1.000000 |

- Each dataset has one shared 2,048-index sample, 16 batches of 128, unique in-range indices and verified little-endian index SHA. The model records have matching batch/count/mode structure, and status population hashes agree with each dataset result.
- All six models record exact pinned-source objective and all-parameter gradient parity on batch 0, and unchanged model states. Do not describe this as exact source-gradient parity individually rechecked on all 16 batches.
- All six summary rows point to matching SHA256 of the corresponding dataset result. All reported medians, extrema, negative counts, clipping counts, norm ratios and time-head squared-norm shares recompute.
- Recorded norms/clip coefficients/cosines independently satisfy the formulas. No coefficient is interpreted as an actual optimizer update; status explicitly records optimizer_updates=0.

## Interpretation

The proposed cautious conclusion is supported: these endpoint snapshots do not establish a common severe directional-gradient-conflict mechanism across the three datasets. They also do not establish absence of interference during training.

Taxi candidate has joint clipping on 13/16 batches versus B 4/16, while quantity-only gradients do not clip on any of the 16 batches for either model. In this specific fixed-checkpoint analytic comparison, adding the time task introduces clipping that Q alone would not require. Candidate median clip coefficient is 0.652544, about 34.75% scaling reduction relative to the unclipped Q-only coefficient. This does not quantify actual AdamW update reduction or explain benchmark performance causally.

Intermittent has no clipping for either model on this sample. Its worse candidate performance cannot be attributed to currently observed endpoint clipping. Mild negative cosine medians alone neither prove destructive representation learning nor justify treating separation as an established remedy.

Instacart has identical clipping counts (joint6/16, Q-only3/16) but different per-batch values; identical counts are not identical gradients or mechanisms.

Selected epochs differ (Intermittent77/37, Taxi45/83), one training seed was used, histories overlap, and this is CPU train-mode replay of frozen CUDA-trained endpoints. No trajectory-causal, confidence-interval, population-general or significance claim is warranted. Same sampling/seed supports controlled descriptive comparison, not independent observations or identical dropout masks across architectures.

Controlled fresh J/Q/T training would address a different question and remains a next-stage experiment. Its selector and compute contract must be kept separate from this gradient snapshot.

## Cost verification

All recorded step and proxy-hour arithmetic passes. 120 epochs x3 arms gives total6,816,240 optimizer steps over the three datasets. The sum22.13518048688625 hours is historical5090 full joint-run wall time per epoch, extrapolated to all arms. It is not measured5080 runtime, new J/Q/T runtime, pure GPU kernel time, electricity or monetary cost. It is neither a bound nor a guarantee; Q/T may differ from joint and hardware/software changes can alter speed.

The input metadata says the120-epoch choice predates gradient results. That chronology is a receipt assertion; these six files alone do not independently prove freeze timing. No scientific performance result should be presented as previously fixed merely from this assertion without the frozen-contract receipt.
