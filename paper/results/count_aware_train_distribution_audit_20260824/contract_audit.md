# Train-only Audit Contract

- Status: **PASS**
- Inputs: three explicit `_train.parquet` files
- Observed split value: `train` only
- Required key: unique `(oper_part_no, seq)`
- Quantity: finite and strictly positive
- Quantile interpolation: `nearest`
- Dataset count: 3
- Validation/test parquet files: not read
- Held-out test: not evaluated
