# Dataset Qualification for the Count-aware TPP Study

## Decision

The primary benchmark consists of Intermittent v2 and Online Retail II. RAF Spare
Parts is structurally suitable and is retained as a main-dataset candidate, but it
must not be described as publication-cleared until reuse permission is confirmed.
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
- The dataset is a strong intermittent-demand benchmark, but the downloaded GitHub
  repository has no explicit license. Publication reuse permission remains a blocker.

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
2. Add RAF to the main table only after permission and citation verification.
3. Put NYC Taxi Hourly in robustness or supplementary analysis.
4. Use Instacart only when basket-count generalization adds a necessary argument.

## Sources and usage notes

- Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii
  (CC BY 4.0; DOI 10.24432/C5CG6D).
- RAF repository: https://github.com/danieldehaan96/spdf. The repository asks
  users to cite Daniel de Haan's benchmark repository but does not include an
  explicit software or data license.
- Published RAF description: https://pmc.ncbi.nlm.nih.gov/articles/PMC8629246/
  reports 5,000 parts and 84 monthly periods, consistent with the audited file.
