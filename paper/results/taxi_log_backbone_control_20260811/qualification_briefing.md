# Taxi Log-Regression Backbone Control

## Qualification

- Status: **qualified validation-only backbone control**.
- All three backbones use the same fixed Taxi split, seeds 42/52/62, log1p-MSE quantity target, softplus output, expm1 reconstruction, and best-validation-NLL checkpoint rule.
- Held-out test evaluation remains locked; no test artifacts are present.
- Fixed data SHA-256: `b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46`.
- THP/TitanTPP source revision: `e3c577ed78752ef6d6023801c00dc42ee98fc1c9`; RMTPP reference source revision: `19ee084161887a1247819c8d535aedb5b5e4aa38`.
- RMTPP marker/time NLL components were not persisted by the earlier runner, so those two cells remain unavailable. Total NLL is directly comparable.

## Overall Validation Results

| Model | NLL | Quantity MAE | Quantity RMSE | Mark accuracy |
|---|---:|---:|---:|---:|
| Adapted RMTPP | 1.5528 +/- 0.0002 | 34.399 +/- 0.507 | 113.319 +/- 5.262 | 0.9244 +/- 0.0010 |
| Adapted THP | 1.6127 +/- 0.0102 | 44.169 +/- 6.354 | 151.156 +/- 8.243 | 0.9151 +/- 0.0007 |
| TitanTPP | 1.5712 +/- 0.0076 | 40.200 +/- 3.296 | 137.526 +/- 13.651 | 0.9216 +/- 0.0042 |

## Validation NLL Decomposition

| Model | Marker NLL | Time NLL |
|---|---:|---:|
| Adapted RMTPP | Not persisted | Not persisted |
| Adapted THP | 0.2341 +/- 0.0188 | 1.3786 +/- 0.0089 |
| TitanTPP | 0.2069 +/- 0.0075 | 1.3643 +/- 0.0026 |

## Quantity MAE by Train-Derived Stratum

| Model | <=p50 | p50-p90 | p90-p95 | p95-p99 | >p99 |
|---|---:|---:|---:|---:|---:|
| Adapted RMTPP | 1.668 +/- 0.265 | 26.583 +/- 3.202 | 179.596 +/- 2.477 | 275.291 +/- 35.571 | 525.710 +/- 153.212 |
| Adapted THP | 2.664 +/- 1.017 | 36.011 +/- 1.975 | 222.301 +/- 67.484 | 319.990 +/- 146.056 | 739.675 +/- 237.392 |
| TitanTPP | 1.732 +/- 0.257 | 27.246 +/- 2.342 | 184.161 +/- 36.381 | 401.490 +/- 61.267 | 599.816 +/- 362.210 |

## Decision

TitanTPP improves overall quantity MAE by `9.0%` and NLL by `2.6%` relative to Adapted THP.
The NLL improvement over THP is consistent in all three seeds; the quantity MAE and RMSE improvements occur in two of three seeds.
However, Adapted RMTPP remains better than TitanTPP: TitanTPP has `16.9%` higher overall quantity MAE and `1.2%` higher NLL.
RMTPP is better than TitanTPP on NLL, quantity MAE, and quantity RMSE in all three paired seeds.
The stratum results do not rescue a broad Taxi backbone claim: Adapted RMTPP has the lowest mean quantity MAE in every train-derived quantity stratum.
Therefore, Taxi does **not** support the claim that the Titan backbone is generally superior under a controlled log-regression head. It only supports the narrower statement that TitanTPP outperforms the tested THP backbone on this dataset.
A central long-sequence contribution requires the same head-controlled comparison on a dataset where long histories are the defining condition. The new Intermittent dataset is the remaining go/no-go experiment.

## Manuscript Use

- Do not present Taxi as evidence of universal Titan-backbone superiority.
- The result may be reported as a controlled negative/mixed finding or retained as an internal qualification result.
- Do not combine this result with the exponent-plus-residual runs as if the quantity heads were identical.
- This is a three-seed validation comparison, not a held-out test result or a statistical significance claim.
