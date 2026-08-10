# Taxi validation quantile result qualification

## Decision

- Qualify Taxi as the main-paper result for quantity reconstruction.
- Qualify the Taxi quantile chart as the current submission Figure 2 candidate.
- Use the result to claim that the complete TitanTPP configuration reduces upper-quantity validation error relative to both adapted baselines.
- Do not attribute the difference solely to the encoder or the mark-residual interface until the controlled quantity-interface ablation is complete.

## Evaluation contract

- Dataset: Taxi (`yellow_trip_hourly`) fixed split
- Data SHA-256: `b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46`
- Source revision: `726aa64ab0b5478646d11be36fc19dcb224d417e`
- Models: Adapted RMTPP, Adapted THP, and TitanTPP
- Seeds: 42, 52, and 62
- Checkpoints: nine `best_val_nll` checkpoints from the completed e300 runs
- Quantile boundaries: computed from train quantities only
- Evaluation targets: 8,268 validation events
- Held-out test evaluated: false

The train-derived boundaries are p50 = 7, p90 = 686, p95 = 1,562, and p99 = 3,449. Because quantity is discrete, the realized validation shares differ slightly from the nominal percentiles.

## Quantity MAE by stratum

Values are the mean over three seeds. A negative Titan delta denotes lower error than the baseline.

| True-quantity stratum | Validation events | Adapted RMTPP | Adapted THP | TitanTPP | Titan vs. RMTPP | Titan vs. THP |
|---|---:|---:|---:|---:|---:|---:|
| `q <= 7` | 4,364 (52.78%) | 1.6240 | 3.4345 | 1.5909 | -2.04% (2/3 seeds) | -53.68% (3/3) |
| `7 < q <= 686` | 3,136 (37.93%) | 53.8742 | 42.4257 | 23.6915 | -56.02% (3/3) | -44.16% (3/3) |
| `686 < q <= 1,562` | 388 (4.69%) | 566.5368 | 699.1134 | 119.5106 | -78.91% (3/3) | -82.91% (3/3) |
| `1,562 < q <= 3,449` | 300 (3.63%) | 388.2278 | 761.1342 | 167.5828 | -56.83% (3/3) | -77.98% (3/3) |
| `q > 3,449` | 80 (0.97%) | 402.4089 | 973.6582 | 233.3025 | -42.02% (3/3) | -76.04% (3/3) |

## Cumulative tail check

The p90-and-above group contains 768 validation events, or 9.29% of the validation targets.

| Range | Adapted RMTPP | Adapted THP | TitanTPP | Titan vs. RMTPP | Titan vs. THP |
|---|---:|---:|---:|---:|---:|
| `q > 686` (p90+) | 479.79 +/- 26.91 | 751.94 +/- 21.88 | 150.14 +/- 11.27 | -68.71% (3/3 seeds) | -80.03% (3/3) |
| `q > 1,562` (p95+) | 391.21 +/- 50.84 | 805.88 +/- 146.40 | 181.42 +/- 18.54 | -53.63% (3/3) | -77.49% (3/3) |

Quantity RMSE gives the same upper-tail ranking. Above p90, TitanTPP has lower RMSE than each baseline in all three seeds and in every reported tail stratum. The result is therefore not caused by a small number of MAE cancellations.

## Interpretation boundary

The improvement is strongest where Taxi quantities span hundreds to thousands of units. This directly supports a result-level claim that TitanTPP predicts large validation quantities more accurately than the adapted recurrent and Transformer Hawkes baselines under the frozen contract.

The comparison does not isolate one architectural component. Taxi TitanTPP uses a memory-oriented encoder, mark-conditioned residual experts, and detached quantity-to-mark gradients, while the adapted baselines use their own history encoders with the shared quantity-aware interface. A controlled interface ablation is still required before claiming that the mark-residual representation itself causes the improvement.

## Manuscript-safe text

> The aggregate Taxi improvement is concentrated in the upper quantity range rather than being an artifact of common small events. For validation targets above the train-derived p90 boundary, TitanTPP reduces quantity MAE by 68.7% relative to Adapted RMTPP and 80.0% relative to Adapted THP, with the same ranking in all three seeds. The advantage remains present above p95 and p99.

## Figure decision

`F2_taxi_validation_quantile_mae` is suitable as the current Figure 2 candidate because it adds information not present in the aggregate results table. Panel A shows absolute error by true-quantity range; Panel B makes the relative tail reduction and its direction explicit. The caption must identify the values as validation results and state that boundaries come from the training split.

Suggested caption:

> **Figure 2.** Taxi validation quantity MAE across strata defined by training-split quantity quantiles. Points and error bars show the mean and sample standard deviation over three seeds. TitanTPP retains lower error in every stratum above p90; the held-out test split is not used.

## Next evidence task

1. Run the controlled quantity-interface ablation with uniform bins, train-quantile bins, direct raw-scale regression, and the proposed mark-residual representation.
2. Report the same p90-p95, p95-p99, and above-p99 validation strata for each interface.
3. Retain this figure as the result-level Figure 2 unless the ablation yields a clearer mechanism-focused replacement.
4. Keep the held-out test locked until the interface and figure selection are frozen.
