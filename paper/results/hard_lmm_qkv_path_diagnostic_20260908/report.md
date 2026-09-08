# Hard-LMM Q/K/V Path Diagnostic

This train-only fixed-checkpoint counterfactual diagnoses Q/K/V paths. It does not prove that a retrained QK model will or will not work.

## Prospective QK gate

- Outcome: **failed**
- Classification: `current_counterfactual_does_not_support_QK_candidate_selection`
- Time-triggered folds: 2
- Raw contractual numerical failures: 27
- Diagnostically meaningful numerical failures: 23
- Support limitations: 2

## Pooled metrics

| Dataset | Variant | Raw RMSE | MAE | Log MSE | Legacy time loss | w | wd saturated | Clamp floor | Excess over floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yellow_trip_hourly | B | 77.7832 | 24.3482 | 0.115731 | 1.28286 | 0.0771737 | 0 | N/A | N/A |
| yellow_trip_hourly | FULL | 76.7014 | 24.0568 | 0.105543 | 1.02619 | 0.659674 | 0.00830078 | N/A | N/A |
| yellow_trip_hourly | QK | 103.844 | 32.427 | 0.165006 | 1.65971 | 0.659674 | 0.00830078 | N/A | N/A |
| yellow_trip_hourly | QV | 107.44 | 32.8376 | 0.121334 | 2.22357 | 0.659674 | 0.00830078 | N/A | N/A |
| yellow_trip_hourly | KV | 229.057 | 67.8888 | 0.161198 | 3.95006 | 0.659674 | 0.00830078 | N/A | N/A |
| yellow_trip_hourly | Q | 134.741 | 41.8818 | 0.173905 | 2.157 | 0.659674 | 0.00830078 | N/A | N/A |
| yellow_trip_hourly | K | 394.67 | 114.495 | 0.22366 | 2.98031 | 0.659674 | 0.00830078 | N/A | N/A |
| yellow_trip_hourly | V | 109.322 | 33.1884 | 0.144758 | 10.3236 | 0.659674 | 0.00830078 | N/A | N/A |
| yellow_trip_hourly | ZERO | 108.674 | 34.2508 | 0.174925 | 4.12192 | 0.659674 | 0.00830078 | N/A | N/A |
| intermittent_frozen_5000 | B | 1.4049 | 0.580115 | 0.00973674 | -3.49429 | 89.5077 | 1 | -3.49437 | 8.09088e-05 |
| intermittent_frozen_5000 | FULL | 1.40473 | 0.577745 | 0.0101628 | -1.60267 | 13.5038 | 1 | -1.60302 | 0.00034396 |
| intermittent_frozen_5000 | QK | 3.03131 | 1.29893 | 0.0175904 | -1.6005 | 13.5038 | 1 | -1.60302 | 0.00252225 |
| intermittent_frozen_5000 | QV | 1.56288 | 0.711847 | 0.0125318 | -1.60275 | 13.5038 | 1 | -1.60302 | 0.00027391 |
| intermittent_frozen_5000 | KV | 10.6331 | 3.09348 | 0.0241909 | -1.60029 | 13.5038 | 1 | -1.60302 | 0.0027265 |
| intermittent_frozen_5000 | Q | 4.23334 | 1.73326 | 0.0189897 | -1.59915 | 13.5038 | 1 | -1.60302 | 0.00387161 |
| intermittent_frozen_5000 | K | 24.2407 | 6.77393 | 0.0611164 | -1.59789 | 13.5038 | 1 | -1.60302 | 0.00512642 |
| intermittent_frozen_5000 | V | 10.3254 | 3.21806 | 0.0297211 | -1.60076 | 13.5038 | 1 | -1.60302 | 0.00226334 |
| intermittent_frozen_5000 | ZERO | 17.7115 | 4.56617 | 0.0443965 | -1.59823 | 13.5038 | 1 | -1.60302 | 0.00478595 |
| insta_market_basket | B | 5.76405 | 3.88227 | 0.250571 | 3.14802 | 0.0563324 | 0 | N/A | N/A |
| insta_market_basket | FULL | 5.77142 | 3.89406 | 0.250347 | 3.1485 | 0.0564059 | 0 | N/A | N/A |
| insta_market_basket | QK | 5.79687 | 3.94254 | 0.255115 | 3.14924 | 0.0564059 | 0 | N/A | N/A |
| insta_market_basket | QV | 5.86654 | 3.92706 | 0.252698 | 3.15032 | 0.0564059 | 0 | N/A | N/A |
| insta_market_basket | KV | 5.92705 | 3.96356 | 0.255403 | 3.15406 | 0.0564059 | 0 | N/A | N/A |
| insta_market_basket | Q | 5.81504 | 3.92945 | 0.25452 | 3.15038 | 0.0564059 | 0 | N/A | N/A |
| insta_market_basket | K | 5.8189 | 3.95561 | 0.25836 | 3.153 | 0.0564059 | 0 | N/A | N/A |
| insta_market_basket | V | 6.06456 | 4.01091 | 0.255835 | 3.15387 | 0.0564059 | 0 | N/A | N/A |
| insta_market_basket | ZERO | 5.97459 | 3.97848 | 0.256238 | 3.15285 | 0.0564059 | 0 | N/A | N/A |

## Fold gate audit

| Dataset | Fold | Criterion | Observed | Rule | Passed |
|---|---:|---|---:|---:|---:|
| yellow_trip_hourly | 0 | QK_raw_rmse_max_ratio_FULL | 1.20794 | <= 1.01 | False |
| yellow_trip_hourly | 0 | QK_mae_max_ratio_FULL | 1.27444 | <= 1.01 | False |
| yellow_trip_hourly | 0 | QK_body_mae_max_ratio_FULL | 1.34576 | <= 1.02 | False |
| yellow_trip_hourly | 0 | QK_legacy_time_max_increase_FULL | 0.850293 | <= 0.01 | False |
| yellow_trip_hourly | 0 | QK_raw_rmse_max_ratio_ZERO | 0.792219 | <= 1.01 | True |
| yellow_trip_hourly | 0 | QK_body_mae_max_ratio_ZERO | 0.866038 | <= 1.02 | True |
| yellow_trip_hourly | 0 | QK_legacy_time_max_increase_ZERO | -3.2295 | <= 0.01 | True |
| yellow_trip_hourly | 0 | QK_raw_rmse_improvement_over_ZERO | 0.207781 | >= 0.01 | True |
| yellow_trip_hourly | 1 | QK_raw_rmse_max_ratio_FULL | 1.45007 | <= 1.01 | False |
| yellow_trip_hourly | 1 | QK_mae_max_ratio_FULL | 1.40531 | <= 1.01 | False |
| yellow_trip_hourly | 1 | QK_body_mae_max_ratio_FULL | 1.25906 | <= 1.02 | False |
| yellow_trip_hourly | 1 | QK_legacy_time_max_increase_FULL | 0.416749 | <= 0.01 | False |
| yellow_trip_hourly | 1 | QK_raw_rmse_max_ratio_ZERO | 1.08736 | <= 1.01 | False |
| yellow_trip_hourly | 1 | QK_body_mae_max_ratio_ZERO | 0.844443 | <= 1.02 | True |
| yellow_trip_hourly | 1 | QK_legacy_time_max_increase_ZERO | -1.69491 | <= 0.01 | True |
| yellow_trip_hourly | 1 | QK_raw_rmse_improvement_over_ZERO | -0.0873559 | >= 0.01 | False |
| intermittent_frozen_5000 | 0 | QK_raw_rmse_max_ratio_FULL | 1.96499 | <= 1.01 | False |
| intermittent_frozen_5000 | 0 | QK_mae_max_ratio_FULL | 2.26423 | <= 1.01 | False |
| intermittent_frozen_5000 | 0 | QK_body_mae_max_ratio_FULL | 2.22521 | <= 1.02 | False |
| intermittent_frozen_5000 | 0 | QK_legacy_time_max_increase_FULL | 0.00222595 | <= 0.01 | True |
| intermittent_frozen_5000 | 0 | QK_raw_rmse_max_ratio_ZERO | 0.1502 | <= 1.01 | True |
| intermittent_frozen_5000 | 0 | QK_body_mae_max_ratio_ZERO | 0.455742 | <= 1.02 | True |
| intermittent_frozen_5000 | 0 | QK_legacy_time_max_increase_ZERO | -0.00211151 | <= 0.01 | True |
| intermittent_frozen_5000 | 0 | QK_tail_mae_max_ratio_FULL | 2.12112 | <= 1.02 | False |
| intermittent_frozen_5000 | 0 | QK_tail_mae_max_ratio_ZERO | 0.0659356 | <= 1.02 | True |
| intermittent_frozen_5000 | 0 | minimum_absolute_improvement_FULL_to_QK | -0.00222595 | >= 0.005 | False |
| intermittent_frozen_5000 | 0 | minimum_fraction_of_positive_excess_removed | -0.00117675 | >= 0.2 | False |
| intermittent_frozen_5000 | 0 | QK_raw_rmse_improvement_over_ZERO | 0.8498 | >= 0.01 | True |
| intermittent_frozen_5000 | 1 | QK_raw_rmse_max_ratio_FULL | 2.35395 | <= 1.01 | False |
| intermittent_frozen_5000 | 1 | QK_mae_max_ratio_FULL | 2.23304 | <= 1.01 | False |
| intermittent_frozen_5000 | 1 | QK_body_mae_max_ratio_FULL | 2.13399 | <= 1.02 | False |
| intermittent_frozen_5000 | 1 | QK_legacy_time_max_increase_FULL | 0.00213062 | <= 0.01 | True |
| intermittent_frozen_5000 | 1 | QK_raw_rmse_max_ratio_ZERO | 0.195194 | <= 1.01 | True |
| intermittent_frozen_5000 | 1 | QK_body_mae_max_ratio_ZERO | 0.430599 | <= 1.02 | True |
| intermittent_frozen_5000 | 1 | QK_legacy_time_max_increase_ZERO | -0.00241589 | <= 0.01 | True |
| intermittent_frozen_5000 | 1 | QK_tail_mae_max_ratio_FULL | 2.11686 | <= 1.02 | False |
| intermittent_frozen_5000 | 1 | QK_tail_mae_max_ratio_ZERO | 0.0624718 | <= 1.02 | True |
| intermittent_frozen_5000 | 1 | minimum_absolute_improvement_FULL_to_QK | -0.00213062 | >= 0.005 | False |
| intermittent_frozen_5000 | 1 | minimum_fraction_of_positive_excess_removed | -0.00112635 | >= 0.2 | False |
| intermittent_frozen_5000 | 1 | QK_raw_rmse_improvement_over_ZERO | 0.804806 | >= 0.01 | True |
| insta_market_basket | 0 | QK_raw_rmse_max_ratio_FULL | 1.0261 | <= 1.01 | False |
| insta_market_basket | 0 | QK_mae_max_ratio_FULL | 1.0225 | <= 1.01 | False |
| insta_market_basket | 0 | QK_body_mae_max_ratio_FULL | 1.04329 | <= 1.02 | False |
| insta_market_basket | 0 | QK_legacy_time_max_increase_FULL | 0.00270678 | <= 0.01 | True |
| insta_market_basket | 0 | QK_raw_rmse_max_ratio_ZERO | 1.00351 | <= 1.01 | True |
| insta_market_basket | 0 | QK_body_mae_max_ratio_ZERO | 1.03717 | <= 1.02 | False |
| insta_market_basket | 0 | QK_legacy_time_max_increase_ZERO | -0.00148237 | <= 0.01 | True |
| insta_market_basket | 0 | QK_tail_mae_max_ratio_FULL | 0.824866 | <= 1.02 | True |
| insta_market_basket | 0 | QK_tail_mae_max_ratio_ZERO | 0.723122 | <= 1.02 | True |
| insta_market_basket | 0 | QK_raw_rmse_improvement_over_ZERO | -0.00351466 | >= 0.01 | False |
| insta_market_basket | 1 | QK_raw_rmse_max_ratio_FULL | 0.986888 | <= 1.01 | True |
| insta_market_basket | 1 | QK_mae_max_ratio_FULL | 1.0029 | <= 1.01 | True |
| insta_market_basket | 1 | QK_body_mae_max_ratio_FULL | 1.03017 | <= 1.02 | False |
| insta_market_basket | 1 | QK_legacy_time_max_increase_FULL | -0.00123376 | <= 0.01 | True |
| insta_market_basket | 1 | QK_raw_rmse_max_ratio_ZERO | 0.94426 | <= 1.01 | True |
| insta_market_basket | 1 | QK_body_mae_max_ratio_ZERO | 1.02644 | <= 1.02 | False |
| insta_market_basket | 1 | QK_legacy_time_max_increase_ZERO | -0.00573765 | <= 0.01 | True |
| insta_market_basket | 1 | QK_tail_mae_max_ratio_FULL | 0.880993 | <= 1.02 | True |
| insta_market_basket | 1 | QK_tail_mae_max_ratio_ZERO | 0.776277 | <= 1.02 | True |
| insta_market_basket | 1 | QK_raw_rmse_improvement_over_ZERO | 0.0557404 | >= 0.01 | True |
| all | all | minimum_datasets_with_both_folds_QK_contribution | 1 | >= 1 | True |

## Interpretation limits

- Raw contractual gate flags are retained separately from the interpreted outcome.
- B metric comparisons are descriptive; hidden coordinates and prototype row IDs are not comparable across the independently trained checkpoints.
- Gate-validity warning: Cross-B time trigger/recovery compares different analytic slope floors under complete wd clamping. Retain the frozen raw gate flags, but do not use those folds to infer representation quality or architecture selection. Within-checkpoint QK/FULL/ZERO time comparisons remain identified because their slope is shared.
- No alternate mask is selected from QV, KV, Q, K, or V results.
- Failure of this off-manifold counterfactual does not prove QK retraining is impossible.
- No validation or held-out test rows are analyzed.
