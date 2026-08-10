# Chart contract: Taxi validation quantity error

## Analytical question

Does TitanTPP's aggregate Taxi quantity-MAE improvement remain present for large quantities, especially above the train-derived p90, p95, and p99 boundaries?

## Required takeaway

Yes. TitanTPP has lower mean quantity MAE than both adapted baselines in every stratum above p90, and the ranking is consistent across all three seeds.

## Data and comparison

- Quantile boundaries come only from the fixed training split.
- Error metrics use only fixed-split validation targets.
- The figure compares Adapted RMTPP, Adapted THP, and TitanTPP over seeds 42, 52, and 62.
- Error bars show sample standard deviation across seeds.
- Held-out test data are not evaluated.

## Visual form

- Panel A: point-and-line chart of absolute quantity MAE by true-quantity stratum.
- Panel B: grouped bars for TitanTPP's relative MAE change against each adapted baseline.
- A zero reference separates improvement from degradation in Panel B.
- Validation-event shares appear in the x-axis labels so the reader can distinguish common and tail ranges.

## Rendering and palette

- Renderer: Matplotlib with deterministic static export.
- Formats: PNG for review, PDF and SVG for publication workflows.
- Model colors remain stable across panels.
- Distinct markers in Panel A and labeled comparison groups in Panel B reduce reliance on color alone.

## QA checks

- Five strata are present in train-quantile order.
- Stratum counts sum to 8,268 validation events.
- Each plotted value is the mean of three checkpoint evaluations.
- Every p90-and-above MAE comparison favors TitanTPP in all three seeds.
- Reconstructed overall MAE matches each checkpoint's stored validation MAE within the analyzer tolerance.
- Labels, legends, error bars, and the footer remain visible in PNG, PDF, and SVG exports.
