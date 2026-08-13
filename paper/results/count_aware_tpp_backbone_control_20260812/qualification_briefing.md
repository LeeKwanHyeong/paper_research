# Count-aware TPP Backbone Control Qualification

## Decision

- General count-prediction gate: **NO-GO**.
- Long-history gate: **NO-GO**.
- Held-out test remains locked and was not evaluated.

## Overall validation results

| Model | Time NLL | Log-count MSE | Quantity MAE | Quantity RMSE |
|---|---:|---:|---:|---:|
| RMTPP | -3.599494 +/- 0.000000 | 0.010517 +/- 0.000854 | 2.9025 +/- 0.1703 | 10.5787 +/- 0.6047 |
| THP | -3.599485 +/- 0.000004 | 0.004184 +/- 0.000153 | 0.6664 +/- 0.0819 | 2.1507 +/- 0.5552 |
| TITANTPP | -3.593078 +/- 0.000661 | 0.007050 +/- 0.000633 | 0.7469 +/- 0.0685 | 1.9195 +/- 0.2938 |

TitanTPP reduces MAE and RMSE against RMTPP by 74.3% and 81.9%, respectively. Against THP, TitanTPP has 12.1% higher MAE but 10.8% lower RMSE. TitanTPP is lower than THP in MAE for 0/3 seeds and lower in RMSE for 3/3 seeds, so the preregistered general gate fails.

## History-length result

| History | Model | Quantity MAE | Quantity RMSE | Time NLL |
|---|---|---:|---:|---:|
| <=64 | RMTPP | 2.3582 | 6.7272 | -3.599494 |
| <=64 | THP | 0.7628 | 1.6511 | -3.599485 |
| <=64 | TITANTPP | 0.8598 | 1.5755 | -3.599454 |
| 65-128 | RMTPP | 6.5632 | 17.2763 | -3.599494 |
| 65-128 | THP | 1.2241 | 3.3815 | -3.599486 |
| 65-128 | TITANTPP | 1.3069 | 2.8722 | -3.599415 |
| >128 | RMTPP | 0.1381 | 0.8775 | -3.599495 |
| >128 | THP | 0.1078 | 0.3595 | -3.599484 |
| >128 | TITANTPP | 0.1737 | 0.7532 | -3.582683 |

For history >128, TitanTPP has higher MAE than both RMTPP and THP in all three seeds, and higher RMSE than THP in all three seeds. Its RMSE reduction against RMTPP is 14.2% in the >128 stratum, smaller than the 76.6% reduction in the <=64 stratum. The long-history gate therefore fails independently of the overall gate.

## Manuscript boundary

This validation experiment supports a narrow statement that TitanTPP is substantially stronger than the GRU-based RMTPP count baseline and reduces RMSE relative to THP. It does not support superiority over both baselines, nor a claim that TitanTPP benefits more from long histories. The held-out test must remain unevaluated for this configuration.
