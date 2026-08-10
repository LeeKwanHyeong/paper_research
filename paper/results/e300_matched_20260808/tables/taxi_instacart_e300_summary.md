# Taxi and Instacart e300 validation summary

All values are mean +/- sample standard deviation over seeds 42, 52, and 62. Lower is better except for mark accuracy.

## Taxi

| Model | Val NLL | Qty MAE | Delta-t MAE | Mark acc. | Best epoch |
|---|---:|---:|---:|---:|---:|
| Adapted RMTPP | 1.5803 +/- 0.0032 | 65.8580 +/- 2.4748 | 0.7326 +/- 0.0085 | 91.800% +/- 0.117%p | 92.7 +/- 40.3 |
| Adapted THP | 1.5998 +/- 0.0087 | 87.7508 +/- 2.6771 | 0.7528 +/- 0.0224 | 91.461% +/- 0.202%p | 36.3 +/- 10.7 |
| TitanTPP | 1.5458 +/- 0.0048 | 23.7722 +/- 1.0929 | 0.7374 +/- 0.0151 | 92.606% +/- 0.134%p | 220.3 +/- 69.3 |

## Instacart

| Model | Val NLL | Qty MAE | Delta-t MAE | Mark acc. | Best epoch |
|---|---:|---:|---:|---:|---:|
| Adapted RMTPP | 4.3809 +/- 0.0007 | 4.3379 +/- 0.0131 | 5.6690 +/- 0.0094 | 49.940% +/- 0.034%p | 135.3 +/- 107.0 |
| Adapted THP | 4.3881 +/- 0.0009 | 4.3046 +/- 0.0081 | 5.7063 +/- 0.0059 | 49.793% +/- 0.091%p | 173.3 +/- 38.0 |
| TitanTPP | 4.3827 +/- 0.0012 | 4.3025 +/- 0.0070 | 5.6827 +/- 0.0027 | 49.809% +/- 0.034%p | 159.7 +/- 22.5 |
