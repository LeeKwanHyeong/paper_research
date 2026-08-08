# Artifact qualification manifest

> Generated against frozen source revision `726aa64ab0b5478646d11be36fc19dcb224d417e`.
> This report contains validation provenance only. It does not qualify held-out test results.

## Current status

- Expected run contracts: 39
- Final-comparison ready: 13
- Draft only: 5
- Pending active run: 5
- No matching artifact: 16

## Qualification by model

| Model | Ready | Draft only | Pending | Missing |
| :--- | ---: | ---: | ---: | ---: |
| RMTPP-matched | 7 | 0 | 2 | 0 |
| RMTPP-original | 0 | 0 | 0 | 9 |
| THP-matched | 6 | 0 | 3 | 0 |
| TitanTPP V2 | 0 | 1 | 0 | 5 |
| TitanTPP V2 control | 0 | 1 | 0 | 2 |
| TitanTPP V3b | 0 | 3 | 0 | 0 |

## Run-level decision

| Dataset | Model | Seed | Qualification | Epochs | Best epoch | Reasons |
| :--- | :--- | ---: | :--- | ---: | ---: | :--- |
| Intermittent | RMTPP-original | 42 | rerun_required | -1 | -1 | no_matching_artifact |
| Intermittent | RMTPP-original | 52 | rerun_required | -1 | -1 | no_matching_artifact |
| Intermittent | RMTPP-original | 62 | rerun_required | -1 | -1 | no_matching_artifact |
| Intermittent | RMTPP-matched | 42 | final_comparison_ready | 300 | 16 | - |
| Intermittent | RMTPP-matched | 52 | final_comparison_ready | 300 | 33 | - |
| Intermittent | RMTPP-matched | 62 | final_comparison_ready | 300 | 78 | - |
| Intermittent | THP-matched | 42 | final_comparison_ready | 300 | 14 | - |
| Intermittent | THP-matched | 52 | final_comparison_ready | 300 | 25 | - |
| Intermittent | THP-matched | 62 | final_comparison_ready | 300 | 35 | - |
| Intermittent | TitanTPP V2 | 42 | draft_only | 50 | 19 | initial_budget_met, strict, source_revision, validation_only, held_out_test_locked, best_val_nll_state_hash, dataset_hashes_missing |
| Intermittent | TitanTPP V2 | 52 | rerun_required | -1 | -1 | no_matching_artifact |
| Intermittent | TitanTPP V2 | 62 | rerun_required | -1 | -1 | no_matching_artifact |
| Taxi | RMTPP-original | 42 | rerun_required | -1 | -1 | no_matching_artifact |
| Taxi | RMTPP-original | 52 | rerun_required | -1 | -1 | no_matching_artifact |
| Taxi | RMTPP-original | 62 | rerun_required | -1 | -1 | no_matching_artifact |
| Taxi | RMTPP-matched | 42 | final_comparison_ready | 300 | 61 | - |
| Taxi | RMTPP-matched | 52 | final_comparison_ready | 300 | 79 | - |
| Taxi | RMTPP-matched | 62 | final_comparison_ready | 300 | 138 | - |
| Taxi | THP-matched | 42 | final_comparison_ready | 300 | 43 | - |
| Taxi | THP-matched | 52 | final_comparison_ready | 300 | 24 | - |
| Taxi | THP-matched | 62 | final_comparison_ready | 300 | 42 | - |
| Taxi | TitanTPP V3b | 42 | draft_only | 50 | 42 | initial_budget_met, strict, source_revision, validation_only, held_out_test_locked, best_val_nll_state_hash, dataset_hashes_missing |
| Taxi | TitanTPP V3b | 52 | draft_only | 50 | 32 | initial_budget_met, strict, source_revision, validation_only, held_out_test_locked, best_val_nll_state_hash, dataset_hashes_missing |
| Taxi | TitanTPP V3b | 62 | draft_only | 50 | 49 | initial_budget_met, strict, source_revision, validation_only, held_out_test_locked, best_val_nll_state_hash, dataset_hashes_missing |
| Taxi | TitanTPP V2 control | 42 | draft_only | 50 | 42 | initial_budget_met, strict, source_revision, validation_only, held_out_test_locked, best_val_nll_state_hash, dataset_hashes_missing |
| Taxi | TitanTPP V2 control | 52 | rerun_required | -1 | -1 | no_matching_artifact |
| Taxi | TitanTPP V2 control | 62 | rerun_required | -1 | -1 | no_matching_artifact |
| Instacart | RMTPP-original | 42 | rerun_required | -1 | -1 | no_matching_artifact |
| Instacart | RMTPP-original | 52 | rerun_required | -1 | -1 | no_matching_artifact |
| Instacart | RMTPP-original | 62 | rerun_required | -1 | -1 | no_matching_artifact |
| Instacart | RMTPP-matched | 42 | final_comparison_ready | 300 | 21 | - |
| Instacart | RMTPP-matched | 52 | pending_active_run | 300 | -1 | active_run_not_complete |
| Instacart | RMTPP-matched | 62 | pending_active_run | 300 | -1 | active_run_not_complete |
| Instacart | THP-matched | 42 | pending_active_run | 300 | -1 | active_run_not_complete |
| Instacart | THP-matched | 52 | pending_active_run | 300 | -1 | active_run_not_complete |
| Instacart | THP-matched | 62 | pending_active_run | 300 | -1 | active_run_not_complete |
| Instacart | TitanTPP V2 | 42 | rerun_required | -1 | -1 | no_matching_artifact |
| Instacart | TitanTPP V2 | 52 | rerun_required | -1 | -1 | no_matching_artifact |
| Instacart | TitanTPP V2 | 62 | rerun_required | -1 | -1 | no_matching_artifact |

## Decision rules

A run is final-comparison ready only when its model contract matches, the epoch budget is 300 or an approved continuation to 800, strict reproducibility and validation-only evaluation are recorded, the source revision and fixed-split hashes match, no held-out test evidence exists, and the best-validation-NLL checkpoint and state hash are present.

A draft-only artifact may support preliminary discussion, but it must be rerun before entering the final comparison table. Pending rows belong to the active frozen launcher and are reclassified after completion.
