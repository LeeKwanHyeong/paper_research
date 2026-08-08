# e300 matched baseline result briefing

## Verdict

- RMTPP-matched and THP-matched e300 results are final-comparison ready for validation-only baseline reporting.
- Existing TitanTPP artifacts remain draft-only because their epoch budget/source-contract does not yet match the new e300 baseline contract. They can support the August 14 draft as preliminary evidence, but not the final fair comparison table.
- The current evidence supports a cautious claim: TitanTPP is promising for long-history and continuous quantity modeling, especially on Taxi quantity prediction, but a fresh strict e300 TitanTPP rerun is needed before making a final superiority claim.

# Preliminary validation summary

Lower is better for Val NLL, Qty MAE, and Delta-t MAE. Higher is better for Mark acc. TitanTPP rows are draft-only because the epoch budget/source-contract does not yet match the new e300 baseline contract.

| Dataset | Model | n | Budget | Val NLL | Qty MAE | Delta-t MAE | Mark acc | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Intermittent | RMTPP-matched | 3 | e300 | 5.6683 +/- 0.0115 | 2.7408 +/- 0.0493 | 41.8872 +/- 0.5030 | 55.183% +/- 0.236%p | 42.3 |
| Intermittent | THP-matched | 3 | e300 | 5.6417 +/- 0.0305 | 2.8812 +/- 0.0177 | 40.5947 +/- 0.3284 | 54.235% +/- 0.637%p | 24.7 |
| Intermittent | TitanTPP V2 (draft e200) | 3 | e200 | 5.6046 +/- 0.0097 | 2.7162 +/- 0.0720 | 41.1990 +/- 0.4479 | 54.697% +/- 0.577%p | 95.0 |
| Taxi | RMTPP-matched | 3 | e300 | 1.5803 +/- 0.0032 | 65.8580 +/- 2.4748 | 0.7326 +/- 0.0085 | 91.800% +/- 0.117%p | 92.7 |
| Taxi | THP-matched | 3 | e300 | 1.5998 +/- 0.0087 | 87.7508 +/- 2.6771 | 0.7528 +/- 0.0224 | 91.461% +/- 0.202%p | 36.3 |
| Taxi | TitanTPP V3b (draft e50) | 3 | e50 | 1.5555 +/- 0.0019 | 31.0775 +/- 2.8184 | 0.7591 +/- 0.0206 | 92.267% +/- 0.194%p | 36.7 |
| Instacart | RMTPP-matched | 3 | e300 | 4.3809 +/- 0.0007 | 4.3379 +/- 0.0131 | 5.6690 +/- 0.0094 | 49.940% +/- 0.034%p | 135.3 |
| Instacart | THP-matched | 3 | e300 | 4.3881 +/- 0.0009 | 4.3046 +/- 0.0081 | 5.7063 +/- 0.0059 | 49.793% +/- 0.091%p | 173.3 |
| Instacart | TitanTPP V2 (draft e200) | 3 | e200 | 4.3819 +/- 0.0009 | 4.3199 +/- 0.0084 | 5.6768 +/- 0.0129 | 49.941% +/- 0.027%p | 52.0 |

## Figure files

- `paper/results/e300_matched_20260808/figures/validation_nll_comparison.png`
- `paper/results/e300_matched_20260808/figures/quantity_mae_comparison.png`
- `paper/results/e300_matched_20260808/figures/paper_applicability_matrix.png`
