# T1 v0.7. Frozen dataset statistics

> Population and split counts come from the frozen split identity. Sequence-length and quantity distributions use train rows only; no held-out target statistic or model performance is included.

| Dataset | Sequences | Event rows | Train / validation / test | Train sequence length med. / p95 / max | Train quantity med. / p95 / max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Intermittent-5000 | 5,000 | 573,128 | 398,824 / 86,285 / 88,019 | 62 / 182 / 193 | 2 / 46 / 477 |
| Taxi | 131 | 55,119 | 38,524 / 8,268 / 8,327 | 283 / 520 / 520 | 7 / 1562 / 6322 |
| Instacart | 206,209 | 3,279,521 | 2,197,401 / 503,733 / 578,387 | 7 / 35 / 70 | 8 / 25 / 175 |

The Intermittent row refers to the frozen 5,000-series experiment population. It supersedes the earlier full-corpus row for v0.7 model-result reporting.
