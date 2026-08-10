# Chart contract: Instacart validation quantity error

## Analytical question

Does TitanTPP's small overall Instacart quantity-MAE advantage remain present in the upper true-quantity strata?

## Required takeaway

No. TitanTPP improves the lower 89.37% of validation events but has higher mean MAE above the train-derived p90 boundary. The figure is diagnostic evidence, not a submission figure supporting tail improvement.

## Data and comparison

- Quantile boundaries come only from the fixed training split.
- Error metrics use only fixed-split validation targets.
- The figure compares Adapted RMTPP, Adapted THP, and TitanTPP over seeds 42, 52, and 62.
- Error bars show sample standard deviation across seeds.
- Held-out test data are not evaluated.

## Visual form

- Panel A: point-and-line chart of absolute quantity MAE by true-quantity stratum.
- Panel B: grouped bars for TitanTPP's relative MAE change against each adapted baseline.
- A zero line separates improvement from degradation in Panel B.
- Validation-event shares appear in the x-axis labels to prevent small tail strata from being mistaken for equal-sized groups.

## Rendering and palette

- Renderer: Matplotlib with deterministic static export.
- Formats: PNG for review, PDF and SVG for publication workflows.
- Model colors remain stable across panels; the comparison bars use separate blue and green hues.
- The chart avoids relying on color alone by using distinct markers in Panel A and a labeled zero reference in Panel B.

## QA checks

- Five strata are present in train-quantile order.
- Stratum counts sum to 503,733 validation events.
- Each plotted value is the mean of three checkpoint evaluations.
- The overall reconstructed MAE matches each checkpoint's stored validation MAE within the analyzer tolerance.
- Labels, legends, error bars, and the explanatory footer remain visible in PNG, PDF, and SVG exports.
