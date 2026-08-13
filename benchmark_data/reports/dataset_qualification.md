# Dataset Qualification for the Count-aware TPP Study

## Decision

The primary benchmark consists of Intermittent v2 and Online Retail II. RAF Spare
Parts is structurally suitable and may be used for non-commercial academic analysis
with repository and provenance citations. The public distribution does not include
an explicit data license, so the raw workbook must not be redistributed with paper
artifacts unless separate permission is obtained.
NYC Taxi Hourly is auxiliary because its quantity is a derived grid-hour pickup
count. Instacart remains optional auxiliary evidence because its target is a derived
basket size rather than a native line-item quantity.

## Main datasets

### Intermittent v2

- Native `order_qty` with part and order date identifiers.
- Frozen 5,000-series sample with 573,128 positive-demand events.
- Sequence length median 89 and p95 261.
- Quantity median 2, p95 43, p99 181, and maximum 477.
- Existing fixed chronological train/validation/test split and hashes are retained.

### Online Retail II

- Native transaction `Quantity`, SKU (`StockCode`), and exact `InvoiceDate`.
- Cancellations, returns, invalid timestamps, nonpositive prices, and service or
  adjustment codes are removed by a fixed row-level contract.
- Events sharing a SKU and timestamp are summed.
- Global chronological boundaries define train, validation, and test.
- SKU eligibility uses only training-period event counts, preventing future leakage.
- Extreme positive quantities remain in the data and are reported, not silently clipped.

## Main candidate

### RAF Spare Parts

- 5,000 unique parts and 84 complete monthly observations from January 1996 to
  December 2002.
- 42,695 positive-demand events; 89.83% of monthly cells are zero.
- Positive quantity median 2, p95 55, p99 200, and maximum 2,062.
- Every item has at least three positive-demand months, so positive-event TPP
  conversion is structurally possible.
- The dataset is a strong intermittent-demand benchmark. Its public repository is
  explicitly intended for benchmarking and supplies a citation file, which supports
  academic analysis and reporting of aggregate results.
- The repository has no explicit data license. This is treated as a redistribution
  restriction, not as a blocker to running and reporting a non-commercial academic
  experiment. The raw workbook is excluded from public paper artifacts.
- The frozen model table contains 42,695 positive-demand events under a global
  chronological split: 30,779 train, 6,690 validation, and 5,226 held-out test
  events. Because the first event of each of the 5,000 item histories cannot serve
  as a next-event target, the train loader exposes 25,779 training targets. All
  validation events have prior context and expose 6,690 validation targets.
- The common count-aware loader smoke test passed with an 84-month lookback and
  84-token maximum sequence length. Model-facing timestamps, month indices,
  inter-event intervals, quantities, and split labels are therefore experiment-ready.

## Auxiliary datasets

### NYC Taxi Hourly

The target is a derived number of pickups per grid cell and hour. It is useful for
long sequence and derived-count robustness, but it is not evidence based on a native
quantity field.

### Instacart

The target is the number of product rows in a user order. It is defensible as basket
size, but it does not contain native per-line item quantity. It should remain optional
and should not carry the primary count-representation claim.

## Excluded datasets

- The earlier Head Office Intermittent set is superseded by Intermittent v2 and has
  an unstable tail (`p95=16`, `max=5000`).
- GlucoBench predicts continuous glucose measurements, not demand counts.
- Amazon, Retweet, StackOverflow, Taobao, and Volcano provide event time and event
  type but no quantity target.
- The small Walmart Weekly Sales dataset reports revenue, not unit counts.
- M5 remains pending until Kaggle access and competition-rule acceptance are complete.

## Paper-facing configuration

1. Main table: Intermittent v2 and Online Retail II.
2. Add RAF to the main table with the repository and original RAF-study citations;
   publish only derived statistics and model results, not the raw workbook.
3. Put NYC Taxi Hourly in robustness or supplementary analysis.
4. Use Instacart only when basket-count generalization adds a necessary argument.

## Sources and usage notes

- Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii
  (CC BY 4.0; DOI 10.24432/C5CG6D).
- RAF data distribution: Daniel de Haan, *GitHub repository for benchmarking spare
  parts demand forecasting for intermittent demand*, version 1.0.0, 20 September
  2021, https://github.com/danieldehaan96/spdf. The repository's `CITATION.cff`
  requests this citation, but the repository contains no explicit data license.
- Original RAF study: R. H. Teunter and L. Duncan, "Forecasting intermittent
  demand: a comparative study," *Journal of the Operational Research Society*,
  60(3), 321-329, 2009. https://doi.org/10.1057/palgrave.jors.2602569.
- Dataset-characteristics reference: A. A. Syntetos, M. Z. Babai, and N. Altay,
  "On the demand distributions of spare parts," *International Journal of
  Production Research*, 50(8), 2101-2117, 2012.
  https://doi.org/10.1080/00207543.2011.562561. This paper reports the RAF panel as
  5,000 SKUs with 84 monthly observations, matching the audited workbook.
- Usage boundary: use the workbook internally for non-commercial academic analysis,
  cite both the distribution and provenance, and publish only aggregate statistics,
  transformations, code, and model results. Obtain explicit permission before
  redistributing the raw workbook or a row-level derivative that substantially
  reproduces it.
