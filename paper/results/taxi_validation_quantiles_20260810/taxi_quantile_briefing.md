# Taxi validation quantity error by train-derived quantiles

- Quantile source: fixed-split train quantities only
- Evaluation target: validation events only
- Checkpoint: best validation event NLL
- Held-out test evaluated: false
- Seeds: 42, 52, 62
- Boundaries: p50=7, p90=686, p95=1562, p99=3449

## <= 7 (n=4,364, 52.78%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 1.6240 +/- 0.2250 | 5.5080 +/- 3.8521 | 0.6974 +/- 0.0966 | 0.2448 +/- 0.1447 |
| Adapted THP | 3.4345 +/- 1.7195 | 29.3145 +/- 27.9500 | 1.4748 +/- 0.7384 | 1.7847 +/- 1.6868 |
| TitanTPP | 1.5909 +/- 0.1330 | 5.0936 +/- 1.4787 | 0.6831 +/- 0.0571 | 0.2225 +/- 0.1875 |

## (7, 686] (n=3,136, 37.93%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 53.8742 +/- 0.5196 | 120.9940 +/- 2.7543 | 0.5146 +/- 0.0050 | -38.9920 +/- 2.4715 |
| Adapted THP | 42.4257 +/- 9.9574 | 125.3262 +/- 66.7236 | 0.4053 +/- 0.0951 | -12.0127 +/- 2.2021 |
| TitanTPP | 23.6915 +/- 1.3890 | 56.3149 +/- 10.9963 | 0.2263 +/- 0.0133 | 3.3354 +/- 7.5078 |

## (686, 1562] (n=388, 4.69%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 566.5368 +/- 32.6431 | 618.1592 +/- 27.0331 | 0.5489 +/- 0.0316 | 93.2594 +/- 49.8366 |
| Adapted THP | 699.1134 +/- 100.9607 | 753.0177 +/- 84.5325 | 0.6773 +/- 0.0978 | -34.4725 +/- 204.2642 |
| TitanTPP | 119.5106 +/- 14.1528 | 172.1420 +/- 15.2772 | 0.1158 +/- 0.0137 | 8.2562 +/- 15.7807 |

## (1562, 3449] (n=300, 3.63%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 388.2278 +/- 41.9865 | 481.7585 +/- 58.7943 | 0.1705 +/- 0.0184 | 383.2532 +/- 42.1332 |
| Adapted THP | 761.1342 +/- 85.8728 | 826.7795 +/- 87.5347 | 0.3343 +/- 0.0377 | 710.9143 +/- 82.8487 |
| TitanTPP | 167.5828 +/- 14.7662 | 242.1038 +/- 24.2730 | 0.0736 +/- 0.0065 | -10.9526 +/- 64.2091 |

## > 3449 (n=80, 0.97%)

| Model | Quantity MAE | Quantity RMSE | WAPE | Bias |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 402.4089 +/- 95.1816 | 478.8508 +/- 108.7357 | 0.0939 +/- 0.0222 | 364.3867 +/- 133.4546 |
| Adapted THP | 973.6582 +/- 374.4039 | 1065.7402 +/- 386.8124 | 0.2272 +/- 0.0874 | 955.9087 +/- 389.4870 |
| TitanTPP | 233.3025 +/- 38.8108 | 317.2422 +/- 34.2301 | 0.0544 +/- 0.0091 | -86.0179 +/- 125.3130 |

## TitanTPP paired MAE deltas

Negative delta means lower MAE for TitanTPP.

| Stratum | Baseline | Delta mean | Relative delta | Better seeds |
|---|---|---:|---:|---:|
| <= 7 | Adapted RMTPP | -0.0331 | -2.04% | 2/3 |
| <= 7 | Adapted THP | -1.8436 | -53.68% | 3/3 |
| (7, 686] | Adapted RMTPP | -30.1827 | -56.02% | 3/3 |
| (7, 686] | Adapted THP | -18.7342 | -44.16% | 3/3 |
| (686, 1562] | Adapted RMTPP | -447.0261 | -78.91% | 3/3 |
| (686, 1562] | Adapted THP | -579.6027 | -82.91% | 3/3 |
| (1562, 3449] | Adapted RMTPP | -220.6450 | -56.83% | 3/3 |
| (1562, 3449] | Adapted THP | -593.5514 | -77.98% | 3/3 |
| > 3449 | Adapted RMTPP | -169.1065 | -42.02% | 3/3 |
| > 3449 | Adapted THP | -740.3557 | -76.04% | 3/3 |
