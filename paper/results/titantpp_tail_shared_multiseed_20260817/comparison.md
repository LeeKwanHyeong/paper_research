# TitanTPP-T1 Three-seed Validation Comparison

- Scope: Intermittent validation only
- Seeds: 42, 52, 62
- Held-out test: not evaluated
- Table role: T0 common controls and the T1 incumbent
- H0/H3 time heads: diagnostic-only and excluded from this model table

| Model | Joint objective | Time NLL | Log quantity MSE | Quantity MAE | Quantity RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Adapted RMTPP | -3.588978 +/- 0.000855 | -3.599494 +/- 0.000000 | 0.010517 +/- 0.000854 | 2.902523 +/- 0.170252 | 10.578742 +/- 0.604664 |
| Adapted THP | -3.595301 +/- 0.000150 | -3.599485 +/- 0.000004 | 0.004184 +/- 0.000153 | 0.666380 +/- 0.081872 | 2.150737 +/- 0.555155 |
| TitanTPP-T0 | -3.586028 +/- 0.001290 | -3.593078 +/- 0.000661 | 0.007050 +/- 0.000633 | 0.746917 +/- 0.068508 | 1.919518 +/- 0.293766 |
| TitanTPP-T1 | -3.586923 +/- 0.001086 | -3.593170 +/- 0.000919 | 0.006229 +/- 0.000180 | 0.698884 +/- 0.059655 | 1.799715 +/- 0.182126 |

## TitanTPP-T1 deltas

### Versus Adapted RMTPP

- MAE improvement: `75.9215%`
- RMSE improvement: `82.9874%`
- Time NLL absolute regression: `0.00632452`
- Lower MAE seeds: `3/3`
- Lower RMSE seeds: `3/3`

### Versus Adapted THP

- MAE improvement: `-4.8778%`
- RMSE improvement: `16.3210%`
- Time NLL absolute regression: `0.00631496`
- Lower MAE seeds: `1/3`
- Lower RMSE seeds: `2/3`

### Versus TitanTPP-T0

- MAE improvement: `6.4309%`
- RMSE improvement: `6.2413%`
- Time NLL absolute regression: `-0.00009233`
- Lower MAE seeds: `2/3`
- Lower RMSE seeds: `2/3`

