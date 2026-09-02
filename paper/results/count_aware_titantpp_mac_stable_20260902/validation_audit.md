# Count-aware TitanTPP-MAC Stable Seed 52/62 Validation Audit

## Scope

This record covers the fresh stable-policy validation runs for seed 52 on RTX 5090 and seed 62 on RTX 5080. It does not combine the historical seed 42 result and therefore is not a final three-seed table. Held-out test data remained locked.

## Provenance

- Training revision: `c4dbf856c32e6502acc660ffac23c3e2f68e5375`
- Seed 52 orchestration revision: `dd9790ef190784bf0084246efb5f8614016a5411`
- Seed 62 orchestration revision: `f814e03f8a9f842937bd0de3a8dc4d78a96d61e7`
- Seed 52 artifact: `search_artifacts/count_aware_titantpp_mac_stable_seed52_e300_20260831_5090`
- Seed 62 artifact: `search_artifacts/count_aware_titantpp_mac_stable_seed62_e300_20260831_5080`
- Model: Count-aware TitanTPP-MAC (`titantpp_titans_mac`)
- Evaluation: fixed split, validation-only, seeds 52 and 62

## Contract Audit

- Both artifacts report `complete` with 4/4 long-validation runs and three context e1 preflights.
- The source manifests match all 24 frozen training-file checksums and the expected training and orchestration revisions.
- All eight long-validation launch contracts use the mark-free count interface, direct MSE on `log1p(raw_quantity)`, `legacy_clamped_rmtpp`, time scale 3, time slope cap 10/3, batch size 128, learning rate 0.001, hidden size 64, inner and outer gradient clipping at 1, and validation joint-objective checkpoint selection.
- Maximum epoch 300, minimum epoch 40, and patience 40 are consistent across datasets. Dataset-specific context lengths are the only intended context difference.
- All JSON and quantity/history scale-wise CSV metrics are finite and non-empty.
- Every checkpoint state digest agrees between the run summary and validator report. Strict prediction replay and observed-history memory replay are exact for all eight runs.
- Summary best epochs equal the minimum validation joint-objective epochs in the corresponding histories.
- No held-out test or test-summary artifact exists, and every report records `held_out_test_evaluated=false`.
- This single-model stable runner did not emit plot files; plots are therefore not part of this artifact contract rather than missing validation evidence.

## Two-Seed Validation Summary

Values are mean +/- sample standard deviation over seeds 52 and 62. These values are descriptive only until a separately qualified seed 42 run is audited under the same stable inner-gradient policy.

| Dataset | Quantity MAE | Quantity RMSE | Time NLL | Joint objective |
| --- | ---: | ---: | ---: | ---: |
| Instacart | 4.0653 +/- 0.0059 | 6.0669 +/- 0.0103 | 3.2064 +/- 0.0016 | 3.4510 +/- 0.0016 |
| Intermittent v2 | 0.7280 +/- 0.0457 | 2.0804 +/- 0.2486 | -3.5994 +/- 0.0001 | -3.5948 +/- 0.0002 |
| RAF Spare Parts | 9.2651 +/- 0.2422 | 36.9737 +/- 1.5612 | 3.2437 +/- 0.0116 | 3.8093 +/- 0.0060 |
| Taxi | 43.7063 +/- 6.7956 | 154.3737 +/- 23.8637 | 1.3678 +/- 0.0044 | 1.5590 +/- 0.0086 |

## Interpretation Boundary

The stable inner-memory gradient policy completed reproducibly on both GPUs without opening held-out test data. The two fresh seeds are highly consistent on Instacart and Intermittent time metrics, while Taxi quantity error has materially larger seed spread. Benchmark superiority and the final Count-aware TitanTPP-MAC positioning must be decided only after an equivalent seed 42 result is admitted and the frozen comparison table is rebuilt.
