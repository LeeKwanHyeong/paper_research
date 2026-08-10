# Instacart validation quantity error by train-derived quantiles

- Quantile source: fixed-split train quantities only
- Evaluation target: validation events only
- Checkpoint: best validation event NLL
- Held-out test evaluated: false
- Seeds: 42, 52, 62
- Boundaries: p50=8, p90=20, p95=25, p99=35

## <= 8 (n=247,651, 49.16%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 3.0637 +/- 0.0789 | 4.4809 +/- 0.1253 | 0.6474 +/- 0.0167 | 1.9761 +/- 0.1237 |
| Adapted THP | 2.9605 +/- 0.0936 | 4.2210 +/- 0.1379 | 0.6256 +/- 0.0198 | 1.8776 +/- 0.1312 |
| TitanTPP | 2.9555 +/- 0.0347 | 4.1867 +/- 0.0997 | 0.6246 +/- 0.0073 | 1.9286 +/- 0.0632 |

## (8, 20] (n=202,534, 40.21%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 4.4490 +/- 0.0053 | 5.6588 +/- 0.0114 | 0.3379 +/- 0.0004 | -1.4660 +/- 0.2317 |
| Adapted THP | 4.3170 +/- 0.0429 | 5.4844 +/- 0.0609 | 0.3279 +/- 0.0033 | -1.9350 +/- 0.2646 |
| TitanTPP | 4.2640 +/- 0.0715 | 5.4188 +/- 0.0547 | 0.3239 +/- 0.0054 | -2.0297 +/- 0.2318 |

## (20, 25] (n=27,322, 5.42%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 6.5990 +/- 0.2285 | 8.9200 +/- 0.1888 | 0.2903 +/- 0.0101 | -6.0872 +/- 0.2853 |
| Adapted THP | 7.2761 +/- 0.3204 | 9.3720 +/- 0.2462 | 0.3201 +/- 0.0141 | -6.9155 +/- 0.3834 |
| TitanTPP | 7.4248 +/- 0.4602 | 9.4425 +/- 0.3524 | 0.3267 +/- 0.0202 | -7.1478 +/- 0.4165 |

## (25, 35] (n=20,190, 4.01%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 11.0302 +/- 0.2186 | 12.5198 +/- 0.2158 | 0.3763 +/- 0.0075 | -10.0680 +/- 0.2495 |
| Adapted THP | 11.6515 +/- 0.2796 | 13.0595 +/- 0.2548 | 0.3974 +/- 0.0095 | -10.8488 +/- 0.3688 |
| TitanTPP | 11.8416 +/- 0.4604 | 13.2022 +/- 0.3962 | 0.4039 +/- 0.0157 | -11.1832 +/- 0.3633 |

## > 35 (n=6,036, 1.20%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 20.2746 +/- 0.2319 | 23.2803 +/- 0.2617 | 0.4685 +/- 0.0054 | -19.1975 +/- 0.3623 |
| Adapted THP | 21.0128 +/- 0.4481 | 23.9496 +/- 0.3020 | 0.4855 +/- 0.0104 | -20.3016 +/- 0.6074 |
| TitanTPP | 21.5132 +/- 0.6978 | 24.2820 +/- 0.4605 | 0.4971 +/- 0.0161 | -20.8923 +/- 1.1189 |

## TitanTPP paired MAE deltas

Negative delta means lower MAE for TitanTPP.

| Stratum | Baseline | Delta mean | Relative delta | Better seeds |
|---|---|---:|---:|---:|
| <= 8 | Adapted RMTPP | -0.1082 | -3.53% | 3/3 |
| <= 8 | Adapted THP | -0.0050 | -0.17% | 2/3 |
| (8, 20] | Adapted RMTPP | -0.1850 | -4.16% | 3/3 |
| (8, 20] | Adapted THP | -0.0530 | -1.23% | 3/3 |
| (20, 25] | Adapted RMTPP | +0.8257 | +12.51% | 0/3 |
| (20, 25] | Adapted THP | +0.1486 | +2.04% | 1/3 |
| (25, 35] | Adapted RMTPP | +0.8114 | +7.36% | 0/3 |
| (25, 35] | Adapted THP | +0.1902 | +1.63% | 1/3 |
| > 35 | Adapted RMTPP | +1.2387 | +6.11% | 0/3 |
| > 35 | Adapted THP | +0.5004 | +2.38% | 1/3 |
