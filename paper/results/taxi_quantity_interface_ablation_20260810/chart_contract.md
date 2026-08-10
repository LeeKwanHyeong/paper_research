# Chart contract: Taxi quantity-interface tail analysis

## Analytical question

Does the magnitude-plus-residual representation reduce upper-tail quantity error relative to uniform bins, train-quantile bins, and direct raw-scale MSE when the RMTPP encoder family is held fixed?

## Decision rule

The chart becomes the main mechanism figure only if magnitude plus residual has lower mean MAE than every alternative in the p90-p95, p95-p99, above-p99, cumulative above-p90, and cumulative above-p95 ranges, with the same ranking in all three seeds.

## Current qualification

- Classification: `diagnostic_only`
- The interface ranking changes across upper-tail ranges; retain the model-level Taxi quantile chart as Figure 2.
- Keep the model-level Taxi quantile figure as Figure 2 and use this chart only as an auxiliary analysis.

## Data and visual form

- Quantity boundaries are fitted on the fixed training split.
- Metrics use fixed validation targets only; the held-out test is not evaluated.
- Panel A compares absolute upper-tail MAE for four quantity interfaces.
- Panel B reports the paired relative MAE change of magnitude plus residual against each alternative.
- Error bars show sample standard deviation over seeds 42, 52, and 62.
- PNG, PDF, and SVG are emitted for review and publication workflows.
