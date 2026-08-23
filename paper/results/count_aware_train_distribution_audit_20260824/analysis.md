# Count-aware Benchmark Train-only Data Audit

- Scope: explicit train parquet files only
- Quantiles: nearest interpolation
- Validation/test rows: not read
- Held-out test: not evaluated

## Quantity Distribution

| Dataset | Train events | p50 | p95 | p99 | Max | >p95 share | >p99 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intermittent v2 | 398,824 | 2 | 46 | 187 | 477 | 4.942% | 0.996% |
| Online Retail II | 567,063 | 4 | 40 | 144 | 19152 | 4.940% | 0.795% |
| RAF Spare Parts | 30,779 | 2 | 60 | 200 | 2062 | 4.916% | 0.764% |

## Train Sequence Structure

| Dataset | Series | Events/series p50 | p95 | p99 | Max | History p50 | p95 | p99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Intermittent v2 | 5,000 | 62 | 182 | 191 | 193 | 40 | 144 | 174 | 192 |
| Online Retail II | 3,125 | 105 | 613 | 1041 | 2422 | 130 | 698 | 1089 | 2421 |
| RAF Spare Parts | 5,000 | 6 | 10 | 12 | 15 | 3 | 8 | 10 | 14 |

## Tail Severity

| Dataset | Mean / p50 | Max / p99 | Interpretation |
| --- | ---: | ---: | --- |
| Intermittent v2 | 7.53x | 2.55x | Moderate-length histories and a bounded frozen quantity tail |
| Online Retail II | 3.28x | 133.00x | Extreme tail far beyond p99; an uncapped raw loss would be high risk |
| RAF Spare Parts | 7.25x | 10.31x | Heavy quantity tail with very short event histories |

## Interpretation

All three native-count datasets are right-skewed, but their absolute p50/p95/p99 thresholds and sequence structures differ. A shared absolute quantity threshold is therefore not portable. Any follow-up body/tail objective must derive thresholds from each dataset's train split using one frozen quantile rule.

Online Retail II has the longest histories and the most extreme outliers: its maximum quantity is more than 100 times its train p99. RAF has the shortest histories, with only six train events per series at the median. Intermittent v2 lies between them. These differences let the next matched validation distinguish a long-history retail setting from a short intermittent-demand setting.

This audit establishes data compatibility only. It does not establish that the Intermittent body-MAE/tail-RMSE trade-off repeats on Online Retail II or RAF; that requires matched validation model runs.

## Decision

Do not implement the new mid-body balanced objective yet. First run the frozen T0 and TitanTPP-T1 validation comparison on all three datasets. If the same body-MAE/tail-RMSE trade-off appears on multiple native-count datasets, design one train-quantile-adaptive objective; otherwise retain the behavior as dataset-specific evidence.
