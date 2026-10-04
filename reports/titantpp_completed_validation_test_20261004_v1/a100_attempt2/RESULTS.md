# A100 completed-candidate Validation/Test reevaluation

## Completion and provenance

- Completed at 2026-10-04 14:45:22 KST: Validation 12/12, Test 12/12; first full Validation gates passed before any Test inference.
- Owned controller and GPU processes absent at 14:45:40 KST. No training, checkpoint selection, or source change occurred.
- Retrieved 273 original files (59,675,920 bytes), checked each SHA, and recomputed every metric from raw prediction parts locally.
- Source closures and checkpoint binaries match the two original A100 campaigns. Full Validation overall and strict-tail reference metrics passed; Test targets, truth and loader identities match the existing MLP seed42 comparator.
- Results are seed42 exploratory, A100-trained / RTX5080-evaluated. No 3-seed SD or A100 efficiency claim; Instacart was not scheduled.
- The original first attempt remains in ../a100: missing process library path caused zero prediction rows and zero Test rows. Attempt2 changed only its workspace and existing LD_LIBRARY_PATH; inference/scientific code is identical.
- Selected checkpoint CPU identity/strict-load audit is available for all 12; last checkpoint binary CPU audit was not part of this evaluation.

## Overall metrics

All metrics below use the same Validation-selected epoch. MAE/RMSE use raw quantity. Time NLL uses the recorded positive-integer lognormal round/clamp mass, including top-coded survival where configured. Test is the previously accessed legacy split, not new independent untouched data.

| Dataset | Candidate | Epoch | Val MAE | Val RMSE | Val Time NLL | Test MAE | Test RMSE | Test Time NLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Taxi | Cross-product | 111 | 26.654181 | 82.435187 | 1.121980 | 33.530819 | 103.925584 | 1.129912 |
| Taxi | Post-block | 93 | 28.094473 | 86.990678 | 1.009135 | 48.406115 | 162.127856 | 0.975861 |
| Taxi | Recent-four-attention | 104 | 25.977915 | 81.613785 | 1.037235 | 40.471469 | 129.563258 | 1.060173 |
| Taxi | Recent-four-mean | 95 | 27.646762 | 87.371090 | 1.115452 | 36.530394 | 115.522856 | 1.097308 |
| RAF | Cross-product | 53 | 9.362425 | 34.200336 | 3.640114 | 10.833809 | 41.650488 | 5.089499 |
| RAF | Post-block | 9 | 9.310979 | 34.094356 | 3.575759 | 10.095533 | 39.767177 | 4.011698 |
| RAF | Recent-four-attention | 9 | 9.261268 | 34.634227 | 3.590718 | 9.947273 | 39.763668 | 4.122593 |
| RAF | Recent-four-mean | 16 | 9.151169 | 34.479365 | 3.520884 | 9.999305 | 39.752789 | 4.138451 |
| Intermittent | Cross-product | 1 | 0.885260 | 1.993651 | 0.507090 | 1.095308 | 2.714679 | 6.990244 |
| Intermittent | Post-block | 100 | 0.685003 | 1.659139 | 0.530265 | 1.236374 | 2.922316 | 23.015568 |
| Intermittent | Recent-four-attention | 39 | 0.653685 | 1.613457 | 0.305608 | 1.057728 | 2.514618 | 21.878336 |
| Intermittent | Recent-four-mean | 68 | 0.677873 | 1.634570 | 0.524067 | 1.185370 | 2.681114 | 41.095059 |

## Strict upper-tail metrics

The tail is raw quantity strictly greater than the frozen training threshold; no Test-fitted thresholds. Every split includes all eligible target rows.

| Dataset | Candidate | Threshold | Val count | Val RMSE | Test count | Test RMSE |
|---|---|---:|---:|---:|---:|---:|
| Taxi | Cross-product | 3449 | 79 | 386.618780 | 83 | 539.118944 |
| Taxi | Post-block | 3449 | 79 | 387.724187 | 83 | 1007.099534 |
| Taxi | Recent-four-attention | 3449 | 79 | 398.536173 | 83 | 598.493675 |
| Taxi | Recent-four-mean | 3449 | 79 | 413.635718 | 83 | 600.203097 |
| RAF | Cross-product | 200.0 | 50 | 319.976380 | 39 | 410.654738 |
| RAF | Post-block | 200.0 | 50 | 325.578932 | 39 | 404.956392 |
| RAF | Recent-four-attention | 200.0 | 50 | 335.550660 | 39 | 409.606339 |
| RAF | Recent-four-mean | 200.0 | 50 | 331.542248 | 39 | 405.333792 |
| Intermittent | Cross-product | 187 | 1141 | 10.440513 | 243 | 8.280346 |
| Intermittent | Post-block | 187 | 1141 | 8.359532 | 243 | 9.442828 |
| Intermittent | Recent-four-attention | 187 | 1141 | 7.976492 | 243 | 10.057390 |
| Intermittent | Recent-four-mean | 187 | 1141 | 7.616932 | 243 | 11.030470 |

## Evidence

- `metrics_per_seed.csv`: 48 rows including overall and tail MAE, RMSE, Time NLL and quantity bias for every split/condition.
- `completion_receipt.json`, `validation_gate.json`, `results.json`: remote completion and full Validation-before-Test gates.
- `retrieval_receipt.json`: all original file SHA and byte sizes.
- `analysis_receipt.json`: independent local raw-part calculations, target/truth checks.
- `final_verification.json`: unchanged sealed files and scientific bindings, result SHA and GPU absence.
