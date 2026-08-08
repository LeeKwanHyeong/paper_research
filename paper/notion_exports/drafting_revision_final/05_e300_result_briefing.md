# TitanTPP e300 matched baseline result briefing and draft applicability

Notion page: https://app.notion.com/p/3b6bbe4056138179a753cebe04509b32

Local working file: `paper/results/e300_matched_20260808/result_briefing.md`

## Purpose

This page summarizes the completed RMTPP/THP matched e300 validation-only baseline run on the 5080 server and compares it with existing TitanTPP draft artifacts. The goal is to decide what can be used in the August 14 paper draft and what still needs rerun before final tables.

## Source artifacts

- New e300 matched baselines: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/final_fair_matched_rmtpp_thp_e300_20260805`
- Local analysis output: `paper/results/e300_matched_20260808/result_briefing.md`
- Summary table: `paper/results/e300_matched_20260808/tables/preliminary_summary.md`
- Delta table: `paper/results/e300_matched_20260808/tables/preliminary_titan_deltas.csv`

## Figures

- Validation NLL comparison: `paper/results/e300_matched_20260808/figures/validation_nll_comparison.png`
- Quantity MAE comparison: `paper/results/e300_matched_20260808/figures/quantity_mae_comparison.png`
- Paper applicability matrix: `paper/results/e300_matched_20260808/figures/paper_applicability_matrix.png`

## Verdict

The result is usable for the August 14 draft, but not yet for the final fair comparison table.

- RMTPP-matched and THP-matched e300 results are final-ready validation baselines.
- Existing TitanTPP results are draft-only because epoch budget and/or run contract differ from the frozen e300 baseline contract.
- The safe draft claim is that TitanTPP is promising for long-history representation and continuous quantity modeling, with the strongest preliminary effect on Taxi.
- Instacart is mixed and should not be used for a strong superiority claim yet.

## Main validation summary

Lower is better for Val NLL, Qty MAE, and Delta-t MAE. Higher is better for Mark acc.

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

## Dataset-level interpretation

### Intermittent

TitanTPP V2 draft e200 has lower Val NLL than RMTPP and THP. Quantity MAE improves over THP but only modestly over RMTPP. Use this as joint-likelihood evidence, not as a strong quantity superiority claim.

### Taxi

Taxi is the strongest current evidence. TitanTPP V3b draft e50 reduces Quantity MAE by about 52.8% vs RMTPP and 64.6% vs THP. Val NLL and Mark acc are also better. Delta-t MAE is slightly worse, so avoid “all-metric superiority.”

### Instacart

Instacart is mixed. TitanTPP V2 draft e200 is very close to RMTPP and only slightly better than THP in Val NLL. It should remain inconclusive until fresh TitanTPP e300 rerun.

## Draft claim policy

Allowed:

> Preliminary validation results suggest that TitanTPP can improve joint event-demand modeling on datasets where quantity prediction and long event histories are important. The improvement is most visible on the Taxi demand dataset, while the Instacart results remain mixed under the current draft-only TitanTPP artifacts.

Avoid:

- TitanTPP consistently outperforms all baselines across all datasets.
- TitanTPP is superior on every metric.
- Instacart confirms TitanTPP's advantage.
- Held-out test results show improvement.

## Paper applicability

- RMTPP-matched: final-ready for Intermittent, Taxi, Instacart.
- THP-matched: final-ready for Intermittent, Taxi, Instacart.
- TitanTPP V2: draft-only for Intermittent and Instacart.
- TitanTPP V3b: draft-only for Taxi.
- Final fair comparison requires fresh TitanTPP e300 runs under the same frozen contract.

## Next work

- Taxi TitanTPP V3b frozen e300 rerun.
- Intermittent TitanTPP V2 frozen e300 rerun.
- Instacart TitanTPP V2 frozen e300 rerun.
- Regenerate T3 main validation result table.
- Decide e800 continuation only after validation convergence review.
- Keep held-out test locked.
