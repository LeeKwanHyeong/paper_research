# Taxi Positive-Regression Control Briefing

## Qualification

The two added controls completed all three seeds under the frozen Taxi split. Both use train-fitted transforms, produce nonnegative quantities by construction, and were evaluated on validation data only.

The fair log-scale regression baseline has the lowest overall MAE. It reduces overall MAE by 47.8% relative to exponent + residual and is also better from p90 through p99. However, exponent + residual is better above p99. The Taxi control therefore does not support a general claim that exponent + residual is more accurate than a properly constrained regression baseline.

Raw MSE remains diagnostic only because its unclipped output violated nonnegative support. It is retained in the table to explain the earlier observation, not as the fair final baseline.

## Mean Validation Error Across Three Seeds

| Interface | Overall MAE | Overall RMSE | p90-p95 MAE | p95-p99 MAE | >p99 MAE |
|---|---:|---:|---:|---:|---:|
| Uniform categorical | 109.204 | 288.347 | 986.982 | 471.002 | 386.219 |
| Quantile categorical | 171.753 | 553.429 | 577.997 | 1824.761 | 3840.253 |
| Raw MSE (diagnostic) | 35.491 | 90.982 | 143.002 | 228.756 | 293.830 |
| Min-max + sigmoid | 44.674 | 120.792 | 188.024 | 323.895 | 429.270 |
| Log-scale regression | 34.399 | 113.319 | 179.596 | 275.291 | 525.710 |
| Exponent + residual | 65.858 | 185.161 | 566.856 | 388.326 | 402.214 |

## Manuscript Decision

Use log-scale regression as the primary fair quantity baseline. Do not claim that the exponent-residual interface solves long-tail quantity prediction on Taxi. A narrower statement is defensible: the representation guarantees valid support and changes the error trade-off, but its advantage is not uniform and disappears against log-scale regression.
