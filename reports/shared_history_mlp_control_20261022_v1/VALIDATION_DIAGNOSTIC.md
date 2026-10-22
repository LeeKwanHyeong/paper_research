# A/B/D Validation 시간·수량 진단

상태: 완료. 기존 Validation 27개 예측을 독립 집계했고 원본 receipt와 수량 RMSE·MAE·TimeNLL이 일치한다. 원본은 변경하지 않았다.

## 시간 점수의 실제 정의

이번 TimeNLL은 양의 정수 시간 간격에 부여한 확률의 −log다. 이분산 로그정규분포를 정수 관측 구간으로 적분한다. 첫 구간은 (0,1.5], 이후 구간은 (d−0.5,d+0.5]이며 세 데이터 모두 top_code=None이다. sigma에는 0.001 하한이 있지만 확률 clipping, intercept cap, wd clamp는 이 실행 경로에 없다. 정규화된 확률질량이므로 음수 NLL은 허용되지 않는다. 과거 clamped RMTPP 점수와 구분해야 한다.

## 데이터별 요약

평균±표본 SD(ddof=1), 공통 3seed. 분위수는 각 seed 내부에서 계산한 뒤 요약한다. 상위 1% 비중은 가장 큰 ceil(N×0.01)개 NLL의 합/전체 NLL합이다.

| 데이터 | 모델 | 수량 RMSE | TimeNLL | median NLL | p99 NLL | 상위 1% 비중 |
|---|---|---:|---:|---:|---:|---:|
| raf_spare_parts | A | 33.9299 ± 0.147 | 3.50973 ± 0.0729 | 3.42782 ± 0.15 | 6.71508 ± 0.25 | 2.00988 ± 0.0289% |
| raf_spare_parts | B | 33.9497 ± 0.168 | 3.51636 ± 0.0739 | 3.44153 ± 0.148 | 6.74874 ± 0.264 | 2.02022 ± 0.0278% |
| raf_spare_parts | D | 34.3338 ± 0.0794 | 3.58495 ± 0.127 | 3.53588 ± 0.24 | 6.97123 ± 0.431 | 2.16066 ± 0.266% |
| yellow_trip_hourly | A | 79.8591 ± 3.09 | 1.04648 ± 0.347 | 0.00140877 ± 0.00243 | 12.7112 ± 5.76 | 23.1564 ± 9.81% |
| yellow_trip_hourly | B | 78.6127 ± 1.01 | 1.31205 ± 0.325 | 2.13978e-06 ± 3.69e-06 | 18.394 ± 5.95 | 29.4421 ± 6.74% |
| yellow_trip_hourly | D | 88.796 ± 0.275 | 0.798373 ± 0.0711 | 0.00125259 ± 0.00206 | 7.60057 ± 1.77 | 20.9045 ± 0.399% |
| intermittent_frozen_5000 | A | 2.06854 ± 0.328 | 0.709715 ± 0.468 | 0.0541766 ± 0.0484 | 12.7738 ± 10.6 | 33.8606 ± 10.9% |
| intermittent_frozen_5000 | B | 1.74532 ± 0.158 | 0.371252 ± 0.117 | 0.00899582 ± 0.00316 | 5.80667 ± 1.93 | 26.3889 ± 4.15% |
| intermittent_frozen_5000 | D | 1.72449 ± 0.0434 | 0.365529 ± 0.0314 | 0.0500803 ± 0.0531 | 4.41338 ± 1.18 | 19.171 ± 5.62% |

## 모든 seed 분포

| 데이터/모델/seed | N | 평균 | median | p90 | p99 | max | 상위1% 합 | 비중 | 음수 개수 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raf_spare_parts/A/42 | 6690 | 3.59387 | 3.56812 | 5.74301 | 6.96566 | 8.31373 | 489.234 | 2.035% | 0 |
| raf_spare_parts/B/42 | 6690 | 3.6016 | 3.58374 | 5.75287 | 7.01299 | 8.41729 | 493.043 | 2.046% | 0 |
| raf_spare_parts/D/42 | 6690 | 3.59074 | 3.5827 | 5.74219 | 6.8861 | 8.20737 | 483.85 | 2.014% | 0 |
| raf_spare_parts/A/52 | 6690 | 3.46455 | 3.26946 | 5.55779 | 6.713 | 7.92575 | 467.422 | 2.017% | 0 |
| raf_spare_parts/B/52 | 6690 | 3.4712 | 3.28854 | 5.56789 | 6.74878 | 7.99641 | 469.878 | 2.023% | 0 |
| raf_spare_parts/D/52 | 6690 | 3.45544 | 3.27634 | 5.52636 | 6.58917 | 7.90142 | 462.405 | 2.000% | 0 |
| raf_spare_parts/A/62 | 6690 | 3.47078 | 3.44587 | 5.46151 | 6.46658 | 8.74101 | 459.314 | 1.978% | 0 |
| raf_spare_parts/B/62 | 6690 | 3.47629 | 3.4523 | 5.47545 | 6.48446 | 9.38923 | 463.033 | 1.991% | 0 |
| raf_spare_parts/D/62 | 6690 | 3.70869 | 3.7486 | 5.89965 | 7.43843 | 18.2554 | 612.215 | 2.468% | 0 |
| yellow_trip_hourly/A/42 | 8268 | 0.672932 | 0.00421743 | 2.40416 | 6.09368 | 29.1824 | 761.359 | 13.684% | 0 |
| yellow_trip_hourly/B/42 | 8268 | 1.40254 | 2.42307e-08 | 3.65261 | 22.163 | 113.876 | 3496.25 | 30.150% | 0 |
| yellow_trip_hourly/D/42 | 8268 | 0.741628 | 0.00362606 | 2.43089 | 5.97686 | 66.3056 | 1300.39 | 21.207% | 0 |
| yellow_trip_hourly/A/52 | 8268 | 1.35911 | 1.53106e-07 | 3.54357 | 16.5473 | 127.018 | 3739.34 | 33.277% | 0 |
| yellow_trip_hourly/B/52 | 8268 | 1.58267 | 2.03948e-10 | 3.86301 | 21.4874 | 295.287 | 4684.67 | 35.800% | 0 |
| yellow_trip_hourly/D/52 | 8268 | 0.77534 | 6.2944e-05 | 2.5377 | 7.33326 | 63.1558 | 1311.11 | 20.452% | 0 |
| yellow_trip_hourly/A/62 | 8268 | 1.10741 | 8.72918e-06 | 3.24909 | 15.4928 | 59.0935 | 2060.89 | 22.508% | 0 |
| yellow_trip_hourly/B/62 | 8268 | 0.950925 | 6.39491e-06 | 2.96759 | 11.5316 | 67.0787 | 1759.25 | 22.376% | 0 |
| yellow_trip_hourly/D/62 | 8268 | 0.878153 | 6.87532e-05 | 2.79792 | 9.4916 | 63.3808 | 1528.62 | 21.054% | 0 |
| intermittent_frozen_5000/A/42 | 86285 | 1.22163 | 0.102996 | 1.64371 | 24.7859 | 197.756 | 48408.5 | 45.925% | 0 |
| intermittent_frozen_5000/B/42 | 86285 | 0.301995 | 0.0100879 | 0.750439 | 4.62843 | 34.7388 | 6091.07 | 23.375% | 0 |
| intermittent_frozen_5000/D/42 | 86285 | 0.38366 | 0.0251645 | 0.907319 | 5.49624 | 32.9343 | 7768.6 | 23.467% | 0 |
| intermittent_frozen_5000/A/52 | 86285 | 0.305323 | 0.00616232 | 0.792817 | 4.83079 | 33.3409 | 6484.89 | 24.615% | 0 |
| intermittent_frozen_5000/B/52 | 86285 | 0.305934 | 0.00543018 | 0.793024 | 4.75668 | 40.0358 | 6513.48 | 24.675% | 0 |
| intermittent_frozen_5000/D/52 | 86285 | 0.383637 | 0.11109 | 1.12516 | 3.15614 | 11.8928 | 4242.53 | 12.816% | 0 |
| intermittent_frozen_5000/A/62 | 86285 | 0.602192 | 0.053371 | 1.28542 | 8.70467 | 70.8874 | 16129.2 | 31.042% | 0 |
| intermittent_frozen_5000/B/62 | 86285 | 0.505829 | 0.0114694 | 1.09887 | 8.03489 | 70.2007 | 13581 | 31.117% | 0 |
| intermittent_frozen_5000/D/62 | 86285 | 0.32929 | 0.0139863 | 0.879314 | 4.58777 | 24.894 | 6031.82 | 21.229% | 0 |

## 동일 target의 비교

차이는 후보−기준이다. 양수는 악화다. 후보 상위1% 기여와 나머지 기여의 합은 전체 평균 NLL 차이다. 이는 진단용 분해이며 해당 target을 평가에서 제거하지 않는다.

| 데이터 | 비교 | seed | ΔRMSE | ΔMAE | ΔNLL | NLL 악화 target | 후보 상위1% 기여 | 나머지 기여 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| raf_spare_parts | B−A | 42 | +0.00735799 | -0.00569327 | +0.00772807 | 48.16% | +0.000576844 | +0.00715123 |
| raf_spare_parts | B−D | 42 | -0.592473 | +0.130287 | +0.0108643 | 53.41% | +0.00445655 | +0.00640774 |
| raf_spare_parts | A−D | 42 | -0.599831 | +0.13598 | +0.00313621 | 53.59% | +0.0040145 | -0.000878284 |
| raf_spare_parts | B−A | 52 | +0.0447689 | -0.0135555 | +0.00665543 | 62.81% | +0.000377286 | +0.00627814 |
| raf_spare_parts | B−D | 52 | -0.111703 | -0.0168344 | +0.0157654 | 55.14% | +0.00195901 | +0.0138063 |
| raf_spare_parts | A−D | 52 | -0.156472 | -0.00327889 | +0.00910993 | 53.83% | +0.00143722 | +0.00767271 |
| raf_spare_parts | B−A | 62 | +0.00739736 | -0.00596517 | +0.0055072 | 50.96% | +0.000567335 | +0.00493987 |
| raf_spare_parts | B−D | 62 | -0.448068 | -0.142478 | -0.232401 | 35.17% | -0.00330127 | -0.229099 |
| raf_spare_parts | A−D | 62 | -0.455465 | -0.136512 | -0.237908 | 35.25% | -0.00377005 | -0.234138 |
| yellow_trip_hourly | B−A | 42 | -3.645 | -1.16696 | +0.729607 | 20.20% | +0.342768 | +0.386839 |
| yellow_trip_hourly | B−D | 42 | -9.2438 | -2.65335 | +0.660911 | 19.67% | +0.28435 | +0.376561 |
| yellow_trip_hourly | A−D | 42 | -5.5988 | -1.48639 | -0.068696 | 65.84% | -0.0501201 | -0.0185758 |
| yellow_trip_hourly | B−A | 52 | -1.6705 | -0.518702 | +0.223565 | 26.33% | +0.158941 | +0.0646246 |
| yellow_trip_hourly | B−D | 52 | -11.512 | -3.20698 | +0.807334 | 21.95% | +0.419944 | +0.38739 |
| yellow_trip_hourly | A−D | 52 | -9.84153 | -2.68828 | +0.583769 | 28.30% | +0.306233 | +0.277536 |
| yellow_trip_hourly | B−A | 62 | +1.57624 | +0.800881 | -0.156487 | 62.03% | -0.00396334 | -0.152524 |
| yellow_trip_hourly | B−D | 62 | -9.79418 | -2.84658 | +0.0727718 | 51.78% | +0.056567 | +0.0162048 |
| yellow_trip_hourly | A−D | 62 | -11.3704 | -3.64746 | +0.229259 | 34.41% | +0.087997 | +0.141262 |
| intermittent_frozen_5000 | B−A | 42 | -0.798751 | -0.244461 | -0.919635 | 25.65% | -0.0126248 | -0.90701 |
| intermittent_frozen_5000 | B−D | 42 | -0.060196 | -0.0119668 | -0.0816657 | 42.33% | +0.0123687 | -0.0940344 |
| intermittent_frozen_5000 | A−D | 42 | +0.738555 | +0.232494 | +0.837969 | 58.04% | +0.560022 | +0.277947 |
| intermittent_frozen_5000 | B−A | 52 | -0.200975 | -0.0872617 | +0.000610904 | 46.83% | +0.0241177 | -0.0235068 |
| intermittent_frozen_5000 | B−D | 52 | -0.113201 | -0.000123299 | -0.0777032 | 19.42% | +0.0496677 | -0.127371 |
| intermittent_frozen_5000 | A−D | 52 | +0.0877744 | +0.0871384 | -0.0783141 | 18.13% | +0.03822 | -0.116534 |
| intermittent_frozen_5000 | B−A | 62 | +0.0300831 | +0.020341 | -0.0963638 | 36.85% | -0.0120007 | -0.0843631 |
| intermittent_frozen_5000 | B−D | 62 | +0.235888 | +0.0889912 | +0.176539 | 44.45% | +0.128855 | +0.0476834 |
| intermittent_frozen_5000 | A−D | 62 | +0.205805 | +0.0686502 | +0.272902 | 64.54% | +0.159476 | +0.113426 |

## 해석 범위

수량 RMSE로 checkpoint를 선택했으므로 시간 NLL이 함께 최소가 되는 조건은 보장하지 않는다. 수량 개선과 시간 악화가 함께 나타나는 seed는 교환관계의 관측 증거이며, 선택 기준이나 모델 구조가 원인이라는 인과 증거는 아니다. 저장 예측에는 실제 dt·예측 duration·mu·sigma가 없어 시간 간격별 분석과 sigma 이상 여부는 미확인이다. 새 추론이나 모델 변경은 수행하지 않았다.

## 원본 증거

각 수치는 아래 Validation 예측과 receipt에 연결되며, 상세 SHA·선택 epoch·checkpoint SHA·분해는 VALIDATION_DIAGNOSTIC.json에 있다.

- raf_spare_parts__full__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/raf_spare_parts__full__seed42/validation_predictions.jsonl.gz` SHA256 `9d6ca9709b6985585fefe8d86873b035d0202fc8e312eddbf36f65061bb430d2`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/raf_spare_parts__full__seed42/validation_receipt.json` SHA256 `7831157384f4e5aa0dd93001de022b5f64f3c42ab053cc8c120f41d340e55375`
- raf_spare_parts__no_prototype__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/raf_spare_parts__no_prototype__seed42/validation_predictions.jsonl.gz` SHA256 `f23bffcbe848341ded233cbd94bb2f32c75e9db4d1920c1b236b86d180772885`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/raf_spare_parts__no_prototype__seed42/validation_receipt.json` SHA256 `e6cc9d0ea42191826e1312e2e5423710c40532e6b72124b3e99104db37cf092e`
- raf_spare_parts__encoder_only__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/raf_spare_parts__encoder_only__seed42/validation_predictions.jsonl.gz` SHA256 `e4f58951254ecdbada9bd305d80a5c3caa628ee751e94463da45e49028985d0b`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/raf_spare_parts__encoder_only__seed42/validation_receipt.json` SHA256 `09a12322b761a09939573138ab536e59f1858f7e0df57b67c558ef13489fac15`
- raf_spare_parts__full__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__full__seed52/validation_predictions.jsonl.gz` SHA256 `19ebbc10271cdb217c28caf9b9477b32dfd195660371d1dca4fccd9bd68fc410`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__full__seed52/validation_receipt.json` SHA256 `4b360eb1661782bd4e528fff256b8aa24b473fcca290c1fabc312b4f38b37834`
- raf_spare_parts__no_prototype__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__no_prototype__seed52/validation_predictions.jsonl.gz` SHA256 `6d123a4aaf342209c7392940d085978f62a233dee2f66e106fabd303d5786355`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__no_prototype__seed52/validation_receipt.json` SHA256 `0a653756719d32839661704a02abf97cd9d04b0527bd26e0d6d35996207c2579`
- raf_spare_parts__encoder_only__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__encoder_only__seed52/validation_predictions.jsonl.gz` SHA256 `2c2b8ce35d542d30c7cab038e23f87f3b88a579e510d08486ef7cef4becc9d9f`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__encoder_only__seed52/validation_receipt.json` SHA256 `d23496c2ba27f55e946596392b6768bd57d4f6697bc4645985467bd0bc77b289`
- raf_spare_parts__full__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__full__seed62/validation_predictions.jsonl.gz` SHA256 `b29f64acb9ca7e61fc1217f224cce851be0c50249c88a5a531350faec3ee672b`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__full__seed62/validation_receipt.json` SHA256 `78ee558fdab8b725ae898d2ca3a7c503d5e194ca7d27ffba1a09a96e25c47cb5`
- raf_spare_parts__no_prototype__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__no_prototype__seed62/validation_predictions.jsonl.gz` SHA256 `6fa310793ff976def087974e0a8884d2012304d0ae15ec0b51616cf9e723ffee`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__no_prototype__seed62/validation_receipt.json` SHA256 `a504bdd36a9b4a7f0625e21d5d8fae69e6d2f358bc19a36a7e96e3172d9cdfa2`
- raf_spare_parts__encoder_only__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__encoder_only__seed62/validation_predictions.jsonl.gz` SHA256 `7fbae5df98e6c1d9875522f80bfcd87897b642c8dd593d03c8783eaa69208c96`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/raf_spare_parts__encoder_only__seed62/validation_receipt.json` SHA256 `a381be6b39293ec80077b0b93fef0e108e74ffcd81a1daade7cf4166b70d7c69`
- yellow_trip_hourly__full__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/yellow_trip_hourly__full__seed42/validation_predictions.jsonl.gz` SHA256 `aeb454d5752f4c55f8712162288d6c748aa91459f9f254c6fde3f1bf5dee5711`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/yellow_trip_hourly__full__seed42/validation_receipt.json` SHA256 `83067e022b60fb0f9a4dedba8bb59654e0117194e666a595eece2b6112394d5b`
- yellow_trip_hourly__no_prototype__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/yellow_trip_hourly__no_prototype__seed42/validation_predictions.jsonl.gz` SHA256 `94e909ede92b590763d1255c7a993b36de63fcd6cbc56330de753f40f0ed9190`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/yellow_trip_hourly__no_prototype__seed42/validation_receipt.json` SHA256 `cdfc0bddb615644d51db4f97c603371ee10fc637527a2f4c4dce06f776fe051a`
- yellow_trip_hourly__encoder_only__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/yellow_trip_hourly__encoder_only__seed42/validation_predictions.jsonl.gz` SHA256 `2759459c4f2298518089a74180cffe55430b7a687f72b2010d218539a6c5a5f4`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/yellow_trip_hourly__encoder_only__seed42/validation_receipt.json` SHA256 `d566a82da0b703e7d896d9b8d17161660044dc0c7701db9bfe4ccc5799cc50f7`
- yellow_trip_hourly__full__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__full__seed52/validation_predictions.jsonl.gz` SHA256 `063e789816b2c07fead61ea226ab0e740ecdfbb661502912275cbfffcdfa99ae`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__full__seed52/validation_receipt.json` SHA256 `e57b8277646f33e5f71b212a1816065b0030137c0f0fe7963e2adabc46aa386d`
- yellow_trip_hourly__no_prototype__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__no_prototype__seed52/validation_predictions.jsonl.gz` SHA256 `b25bc1cb566c3cad5db1fc03a5a398365eff5a0d35d129a43749a4d5856bd7ba`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__no_prototype__seed52/validation_receipt.json` SHA256 `e493d1db339c37769abb3ceeb0bc81c5544cb7720a920185a728bca4739e6512`
- yellow_trip_hourly__encoder_only__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__encoder_only__seed52/validation_predictions.jsonl.gz` SHA256 `88fad4a4414b015c35347c9a2bee448607c98efbe9c60ff15466ed3cc8d268d1`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__encoder_only__seed52/validation_receipt.json` SHA256 `6e805d10b5f624ebf678f01af04bcf0390abe93b851e99a07973b6a714b542ef`
- yellow_trip_hourly__full__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__full__seed62/validation_predictions.jsonl.gz` SHA256 `4761ebb4457b1e2366df516011fafff772a426810927417a75cc2d6ae568068b`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__full__seed62/validation_receipt.json` SHA256 `475d2fcacc1dfbf06b86afd5780750800e3d24f34ba83766d9cd278b9a8d0c47`
- yellow_trip_hourly__no_prototype__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__no_prototype__seed62/validation_predictions.jsonl.gz` SHA256 `4bb82affa9896c6b1a0eaade9124de88c922d0d72a780ef2c645271b9e86d5ab`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__no_prototype__seed62/validation_receipt.json` SHA256 `9a0bc986d248dfed1a17b1be1b857c185751908d4497df5c0fff22d79310d491`
- yellow_trip_hourly__encoder_only__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__encoder_only__seed62/validation_predictions.jsonl.gz` SHA256 `6c665a0cfb6a516b77c8fcf207c4c48f5eafd8381bcb4f41a1cee4a78b7e5487`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/yellow_trip_hourly__encoder_only__seed62/validation_receipt.json` SHA256 `a45202033cd9f56678dbc7689d26f51dcd223373532e45162b4914e378489d0e`
- intermittent_frozen_5000__full__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/intermittent_frozen_5000__full__seed42/validation_predictions.jsonl.gz` SHA256 `1119ec198e6908cb53666c5e4b21d7b5427a5f0016eb443555d88ce6778a299d`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/intermittent_frozen_5000__full__seed42/validation_receipt.json` SHA256 `65d90427e04e3e593fefbfd7e8743eb317ae356aca9c7010e9f0890642ae896b`
- intermittent_frozen_5000__no_prototype__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/intermittent_frozen_5000__no_prototype__seed42/validation_predictions.jsonl.gz` SHA256 `3f40a8b007ba34372105049c276bebd446110a9d41a41774fa8b5e3e678b0944`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/intermittent_frozen_5000__no_prototype__seed42/validation_receipt.json` SHA256 `55bef184a9c405faed27c514813b6e1026c6ce71a878e9c0a52b9c520020d7c3`
- intermittent_frozen_5000__encoder_only__seed42 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/intermittent_frozen_5000__encoder_only__seed42/validation_predictions.jsonl.gz` SHA256 `dd989bc484b0aafdb46ea2077552a427d63c45784417b6ea71883da8772e0152`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5080/originals/run/intermittent_frozen_5000__encoder_only__seed42/validation_receipt.json` SHA256 `0a8a5bf38dcfb2a5032ac50013c661097b9a6fb57e515a3bdfcf7aeacbb81dd4`
- intermittent_frozen_5000__full__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__full__seed52/validation_predictions.jsonl.gz` SHA256 `46f8711b9503f40db046a8e87c6301ccd74bfe02de87c09d0e8509c9ff59c627`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__full__seed52/validation_receipt.json` SHA256 `a633e3e0518e5b6c4b43bd51fc8b1e5162c6fd92ff079c7e5132f9f4805af4ef`
- intermittent_frozen_5000__no_prototype__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__no_prototype__seed52/validation_predictions.jsonl.gz` SHA256 `4f9d52a7c7af7e71f9b673baab89ad219606d227e2cc0ac7f7f9a71df84bb227`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__no_prototype__seed52/validation_receipt.json` SHA256 `bfe1346fddcef955c1898d7f47a33549d1ef1217ac4a36af78d84f09f0e8b5b7`
- intermittent_frozen_5000__encoder_only__seed52 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__encoder_only__seed52/validation_predictions.jsonl.gz` SHA256 `684ad91a04df7f10cc6eb6fd930e4361a81d1eff65c3f15091998b8525ecd48b`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__encoder_only__seed52/validation_receipt.json` SHA256 `606ae433a5b712d9059914a160969bf070fd3a7c9a9e991b0d02d53319f8cae1`
- intermittent_frozen_5000__full__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__full__seed62/validation_predictions.jsonl.gz` SHA256 `34778790d5631b3a34d552736f775813fe9212036c753744c8dfce1bfdc30691`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__full__seed62/validation_receipt.json` SHA256 `19bd6cf7045fcf69de946f8b606c25a9810d151a428156d173af7bc70bc14acb`
- intermittent_frozen_5000__no_prototype__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__no_prototype__seed62/validation_predictions.jsonl.gz` SHA256 `3d92fb3ddcb359d4c751f4e550128993168e0295142e07a23067bc8d96314fae`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__no_prototype__seed62/validation_receipt.json` SHA256 `7c2af0f199b2ae883208e850f87f2462a3d527c220fbbe502cdfceb8ca3f350e`
- intermittent_frozen_5000__encoder_only__seed62 (split=validation):
  - 예측 `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__encoder_only__seed62/validation_predictions.jsonl.gz` SHA256 `9a526d1d1ce8d03ded85f89c8e8be8bfb2efe5dc4a7a6cbc2e479067562021b8`
  - receipt `search_artifacts/shared_component_ablation_20261022_v1/retrieval/fit_validation/5090/originals/run/intermittent_frozen_5000__encoder_only__seed62/validation_receipt.json` SHA256 `18e6d084e1410b5c29a979a42d7dae6d9ddfc17a2b1fd9904f23e9ed86a5df0d`

## 검증 및 남은 작업

- [x] 27개 원본 예측 SHA·split·행수·target 수량 SHA 확인
- [x] 같은 seed·데이터의 A/B/D target 및 이력 길이 일치 확인
- [x] 전체·큰수량 RMSE/MAE/TimeNLL 원 receipt 재현
- [x] 실제 시간 점수 경로 6개 소스 SHA를 동결 실행 계약과 대조
- [x] 분석 뒤 모든 읽은 예측·receipt·소스 SHA 재확인
- [ ] 별도 단순 보정기 실험과 과학적 기여 판단은 상위 작업에서 진행

학습 소스 수정이나 버그 수정은 하지 않았으므로 회귀 테스트 대상 변경은 없다. 진단 스크립트의 --check-only는 동일 원본에서 두 산출물을 다시 계산해 바이트 일치를 검사한다.
