# Instacart validation quantile result qualification

## Decision

- Keep Instacart only as a bounded secondary result in the current manuscript.
- Do not use the candidate Figure 2 as evidence that TitanTPP improves long-tail quantity prediction.
- The safe claim is narrower: TitanTPP has the lowest overall validation quantity MAE, but the small aggregate gain is concentrated below the train-derived p90 boundary.
- Use Taxi or the controlled quantity-interface ablation for the submission Figure 2 if either shows consistent tail improvement.

## Evaluation contract

- Dataset: Instacart fixed split
- Data SHA-256: `06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2`
- Source revision: `726aa64ab0b5478646d11be36fc19dcb224d417e`
- Models: Adapted RMTPP, Adapted THP, and TitanTPP
- Seeds: 42, 52, and 62
- Checkpoints: nine `best_val_nll` checkpoints from the completed e300 runs
- Quantile boundaries: computed from train quantities only
- Evaluation targets: 503,733 validation events
- Held-out test evaluated: false

The train-derived boundaries are p50 = 8, p90 = 20, p95 = 25, and p99 = 35. Ties around the discrete boundaries make the realized validation shares differ slightly from the nominal percentiles.

## Quantity MAE by stratum

Values are the mean over three seeds. A positive Titan delta means that TitanTPP has higher error than the baseline.

| True-quantity stratum | Validation events | Adapted RMTPP | Adapted THP | TitanTPP | Titan vs. RMTPP | Titan vs. THP |
|---|---:|---:|---:|---:|---:|---:|
| `q <= 8` | 247,651 (49.16%) | 3.0637 | 2.9605 | 2.9555 | -3.53% (3/3 seeds) | -0.17% (2/3) |
| `8 < q <= 20` | 202,534 (40.21%) | 4.4490 | 4.3170 | 4.2640 | -4.16% (3/3) | -1.23% (3/3) |
| `20 < q <= 25` | 27,322 (5.42%) | 6.5990 | 7.2761 | 7.4248 | +12.51% (0/3) | +2.04% (1/3) |
| `25 < q <= 35` | 20,190 (4.01%) | 11.0302 | 11.6515 | 11.8416 | +7.36% (0/3) | +1.63% (1/3) |
| `q > 35` | 6,036 (1.20%) | 20.2746 | 21.0128 | 21.5132 | +6.11% (0/3) | +2.38% (1/3) |

Across all validation events, TitanTPP records MAE 4.3025, compared with 4.3379 for Adapted RMTPP and 4.3046 for Adapted THP. The aggregate improvements are 0.82% and 0.05%, respectively.

## Interpretation

The aggregate ranking hides a clear change at `q = 20`. TitanTPP improves MAE throughout the lower 89.37% of validation events, with all seeds improving over both baselines in the `8 < q <= 20` range. In the upper 10.63%, however, its mean MAE is worse in every stratum. The degradation against Adapted RMTPP occurs in all three seeds, while comparisons with Adapted THP are variable but unfavorable on average.

The signed errors point in the same direction. Every model underpredicts the upper strata, and TitanTPP has the most negative mean bias in each stratum above p90. At `q > 35`, its bias is -20.8923, compared with -19.1975 for Adapted RMTPP and -20.3016 for Adapted THP. The current Instacart configuration therefore does not solve the tail underprediction problem.

With only three seeds, the small overall difference from Adapted THP should not be described as a robust improvement. No significance claim is made.

## Manuscript boundary

The current Table 3 wording may retain the exact overall result if it also states that the gain over Adapted THP is negligible relative to seed variation. The manuscript must not infer that Instacart demonstrates long-tail improvement. The candidate quantile figure remains an internal diagnostic and should not replace Figure 2 in the submission.

A manuscript-safe sentence is:

> On Instacart, TitanTPP achieves a marginally lower aggregate quantity MAE, but a validation-stratified analysis shows that this gain is confined to quantities at or below the train-derived p90 boundary; errors increase relative to both adapted baselines in the upper-quantity strata.

## Next evidence task

1. Apply the same train-derived validation stratification to the completed Taxi checkpoints.
2. Use the controlled uniform-bin, quantile-bin, raw-regression, and mark-residual comparison to attribute any tail improvement to the quantity interface.
3. Select the submission Figure 2 only after checking seed consistency in the p90-p95, p95-p99, and above-p99 strata.
4. Keep the held-out test locked until the model and figure selection are frozen.
