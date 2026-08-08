# Intermittent and Taxi e300 validation summary

Lower is better for Val NLL, Qty MAE, and Delta-t MAE. Higher is better for Mark acc.

| Dataset | Model | n | Val NLL | Qty MAE | Delta-t MAE | Mark acc | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| Intermittent | RMTPP-matched | 3 | 5.6683 +/- 0.0115 | 2.7408 +/- 0.0493 | 41.8872 +/- 0.5030 | 55.183% +/- 0.236%p | 42.3 |
| Intermittent | THP-matched | 3 | 5.6417 +/- 0.0305 | 2.8812 +/- 0.0177 | 40.5947 +/- 0.3284 | 54.235% +/- 0.637%p | 24.7 |
| Intermittent | TitanTPP | 3 | 5.6171 +/- 0.0158 | 2.7188 +/- 0.1336 | 41.4268 +/- 0.5581 | 55.194% +/- 1.293%p | 45.7 |
| Taxi | RMTPP-matched | 3 | 1.5803 +/- 0.0032 | 65.8580 +/- 2.4748 | 0.7326 +/- 0.0085 | 91.800% +/- 0.117%p | 92.7 |
| Taxi | THP-matched | 3 | 1.5998 +/- 0.0087 | 87.7508 +/- 2.6771 | 0.7528 +/- 0.0224 | 91.461% +/- 0.202%p | 36.3 |
| Taxi | TitanTPP | 3 | 1.5458 +/- 0.0048 | 23.7722 +/- 1.0929 | 0.7374 +/- 0.0151 | 92.606% +/- 0.134%p | 220.3 |
