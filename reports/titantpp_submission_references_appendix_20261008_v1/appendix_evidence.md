# Appendix restoration evidence

## Completed scope

Prepared `appendix_additions.tex` with three sections. The original full-validation MLP metrics appendix remains the parent editor's responsibility and is not replaced. No current experiment result or completion status was updated.

## Sources and claim map

| Appendix content | Original evidence | Reuse basis |
| --- | --- | --- |
| Event definitions and train populations | Earlier manuscript safe data-definition section; `reports/titantpp_dataset_appendix_20261002_v1/train_profiles.json` | Descriptive data quantities, independent of architecture. |
| Train table and usable-history figure | The same `train_profiles.json`; profiling algorithm in `reports/titantpp_data_characteristics_20260928_v1/analyze.py:61` | Split is train. Quantity quantiles include all training events; gaps/history exclude each first event. |
| Lag formula, figure and composition caveats | `reports/titantpp_train_lag_heatmap_20261002_v1/analysis.json`, `figure.tex`, `verification.json`, and `analyze.py` | The archived computation filters train before materialization, checks identity, and processes complete within-series train sequences without loader windows. |
| Partition boundaries | Current contribution stage `execution_contract.json` dataset entries | Exact boundaries match the historical definitions; no historical error table was copied. |
| MAE/RMSE and additive contributions | Mathematical definitions, adapted from the prior validation-diagnostic method | Generalized focal model/comparator and seed notation, without empirical values. |

## Contract checks

Across all dataset entries in the current Instacart, history, and retrieval contracts, the archived train profile matches the source-data ancestry SHA, train target identity SHA, train target quantity SHA, train target count, lookback, history cap, native time unit, and top code. The contracts' canonical SHA-256 values match `reports/titantpp_contribution_execution_20261008_v1/execution_plan.json`. Canonical SHA and raw file SHA are separate values; the JSON records raw source hashes and canonical contract hashes explicitly.

RAF requires an ancestry distinction. Its current prepared train/validation file differs from the original all-split parquet, while `parent_data_identity` binds the original file and the train population identity/quantity hashes agree. This is metadata verification, not a fresh materialization or recomputation of the dataset.

## Preserved boundaries

The figure describes unconditional log-quantity association, not a CNN+GRU correction effect. A longer event lag changes the pair population. Single-pair series contribute zero centered variance, and the fixed cohort sensitivity is explicitly identified. No interpretation connecting the old model's validation or held-out performance to this figure was carried over.

Old width-4 MLP structural tables, stratum errors, numerical contributions, and all held-out/mixed performance content are excluded. Current MLP8, CNN+GRU54, and prior width-4 MLP evidence remain distinct. Model-specific stratification is described as a method requiring that model's selected validation records.

## Validation and limitations

Source hashes, seven stage/dataset bindings, all twenty lag cells, fixed-cohort prose, and label uniqueness were checked. Raw parquet files and prediction rows were not opened. No training, inference, remote access, or main manuscript edit was performed by this worker. Final LaTeX compilation and page-layout inspection belong to the parent integration step.

The requested length was “총 약3–4쪽 분량(부록에는조금길어도됨)”. The fragment includes a data table and two inline TikZ figures; final pagination depends on the surrounding manuscript. Academic writing review checked precise scope, natural English, and claim-evidence alignment.
