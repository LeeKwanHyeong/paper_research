# Taxi quantity-interface result qualification

## Decision

- Classification: `diagnostic_only`
- The interface ranking changes across upper-tail ranges; retain the model-level Taxi quantile chart as Figure 2.

## Evidence boundary

This experiment holds the RMTPP encoder family, hidden size, optimizer, seed set, fixed split, and epoch budget constant. It changes the quantity interface, so it can support a representation-level comparison within RMTPP. It does not isolate TitanTPP's history encoder.

## Above-p90 paired comparison

| Alternative | Relative MAE change | Better seeds |
|---|---:|---:|
| Uniform-bin categorical | -33.58% | 3/3 |
| Quantile-bin categorical | -65.84% | 3/3 |
| Direct raw-scale MSE | +149.52% | 0/3 |

## Figure rule

Keep `F2_taxi_validation_quantile_mae` as the main Figure 2. Report this interface ablation as an auxiliary table or sensitivity analysis without claiming universal superiority.

## Integrity checks

- Source revision: `32eb40317d208b050c5dc1b83cabc07e5c99864b`
- Data SHA-256: `b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46`
- Evaluation scope: validation only
- Held-out test evaluated: false
