# Taxi quantity-interface ablation

## Contract

- Encoder family: RMTPP with matched hidden size and training budget
- Bin fitting and raw-target normalization: fixed training split only
- Evaluation: fixed validation split only
- Checkpoint selection: best validation event NLL
- Held-out test evaluated: false
- Seeds: 42, 52, 62
- Train-derived boundaries: p50=7, p90=686, p95=1562, p99=3449

## Decision

- Classification: `diagnostic_only`
- The interface ranking changes across upper-tail ranges; retain the model-level Taxi quantile chart as Figure 2.

## Validation quantity MAE

| Range | Interface | MAE | RMSE | Bias |
|---|---|---:|---:|---:|
| (686, 1562] | Uniform-bin categorical | 986.98 +/- 4.45 | 1012.01 +/- 4.79 | -851.44 +/- 26.03 |
| (686, 1562] | Quantile-bin categorical | 578.00 +/- 0.00 | 625.06 +/- 0.00 | -578.00 +/- 0.00 |
| (686, 1562] | Direct raw-scale MSE | 143.00 +/- 19.87 | 188.90 +/- 28.27 | 15.16 +/- 80.41 |
| (686, 1562] | Magnitude + residual | 566.86 +/- 32.85 | 618.69 +/- 27.18 | 96.36 +/- 50.12 |
| (1562, 3449] | Uniform-bin categorical | 471.00 +/- 14.55 | 635.34 +/- 25.25 | -75.31 +/- 25.30 |
| (1562, 3449] | Quantile-bin categorical | 1824.76 +/- 0.00 | 1889.64 +/- 0.00 | -1824.76 +/- 0.00 |
| (1562, 3449] | Direct raw-scale MSE | 228.76 +/- 25.13 | 303.27 +/- 32.95 | 10.42 +/- 144.62 |
| (1562, 3449] | Magnitude + residual | 388.33 +/- 41.42 | 481.64 +/- 58.23 | 383.37 +/- 41.57 |
| > 3449 | Uniform-bin categorical | 386.22 +/- 11.98 | 545.22 +/- 20.93 | -175.99 +/- 74.53 |
| > 3449 | Quantile-bin categorical | 3840.25 +/- 0.00 | 3911.65 +/- 0.00 | -3840.25 +/- 0.00 |
| > 3449 | Direct raw-scale MSE | 293.83 +/- 41.96 | 373.80 +/- 40.23 | 0.61 +/- 208.82 |
| > 3449 | Magnitude + residual | 402.21 +/- 97.62 | 479.08 +/- 111.68 | 363.71 +/- 136.41 |
| Above train p90 | Uniform-bin categorical | 722.27 +/- 7.73 | 839.95 +/- 9.88 | -476.80 +/- 24.92 |
| Above train p90 | Quantile-bin categorical | 1404.36 +/- 0.00 | 1782.71 +/- 0.00 | -1404.36 +/- 0.00 |
| Above train p90 | Direct raw-scale MSE | 192.25 +/- 23.80 | 261.82 +/- 31.16 | 11.80 +/- 116.33 |
| Above train p90 | Magnitude + residual | 479.72 +/- 26.98 | 556.03 +/- 28.38 | 236.71 +/- 36.19 |
| Above train p95 | Uniform-bin categorical | 453.38 +/- 11.22 | 617.80 +/- 19.50 | -96.24 +/- 28.24 |
| Above train p95 | Quantile-bin categorical | 2243.77 +/- 0.00 | 2451.41 +/- 0.00 | -2243.77 +/- 0.00 |
| Above train p95 | Direct raw-scale MSE | 242.28 +/- 28.61 | 319.22 +/- 34.59 | 8.38 +/- 157.02 |
| Above train p95 | Magnitude + residual | 391.21 +/- 50.84 | 481.63 +/- 67.28 | 379.28 +/- 59.10 |

## Magnitude-plus-residual paired MAE changes

Negative values indicate lower error than the alternative.

| Range | Alternative | Relative change | Better seeds |
|---|---|---:|---:|
| (686, 1562] | Uniform-bin categorical | -42.57% | 3/3 |
| (686, 1562] | Quantile-bin categorical | -1.93% | 1/3 |
| (686, 1562] | Direct raw-scale MSE | +296.40% | 0/3 |
| (1562, 3449] | Uniform-bin categorical | -17.55% | 3/3 |
| (1562, 3449] | Quantile-bin categorical | -78.72% | 3/3 |
| (1562, 3449] | Direct raw-scale MSE | +69.76% | 0/3 |
| > 3449 | Uniform-bin categorical | +4.14% | 1/3 |
| > 3449 | Quantile-bin categorical | -89.53% | 3/3 |
| > 3449 | Direct raw-scale MSE | +36.89% | 0/3 |
| Above train p90 | Uniform-bin categorical | -33.58% | 3/3 |
| Above train p90 | Quantile-bin categorical | -65.84% | 3/3 |
| Above train p90 | Direct raw-scale MSE | +149.52% | 0/3 |
| Above train p95 | Uniform-bin categorical | -13.71% | 3/3 |
| Above train p95 | Quantile-bin categorical | -82.56% | 3/3 |
| Above train p95 | Direct raw-scale MSE | +61.47% | 0/3 |
