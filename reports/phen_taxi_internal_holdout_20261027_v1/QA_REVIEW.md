## Review Result: PASS — 로컬 사전 검증 범위

실제 원격 native 검증·첫 epoch·전체 학습 결과는 아직 확인하지 않았다. 이 PASS는 해당 단계의 성공이나 과적합/분포 이동의 원인 확정을 뜻하지 않는다.

### CRITICAL
- 미해결 사항 없음.

### HIGH
- 미해결 사항 없음.

### MEDIUM
- 미해결 사항 없음.

### LOW
- 미해결 사항 없음.

### 완료 범위와 남은 경계
- 독립 CPU 합성 테스트 24개와 본진의 최종 24개 실행이 모두 실제 종료값 0이다. native CPU 검증은 합성 4사건·optimizer 1회이며 실제 Train 학습은 아니다.
- 실제 Train/Validation 원본의 공통 시간 원점, metadata 신원, fit/dev/audit/purge 수와 fit context의 미학습 정답 교차 0을 확인했다.
- prepare_v2의 canonical input, context-only 131건의 target 제외, fit-only 수량·시간 초기화 통계와 DATA→INTERFACE→factory 연결을 확인했다.
- 추가 fit 평가와 native dev 평가 모두 난수·loader generator·model mode를 복구한다. 유한 RMSE와 비유한 시간 점수가 공존하는 경우도 즉시 실패한다.
- 사전 audit 격자와 dev-selected/last만 허용하고, 기존 Validation은 selected/last 두 상태로 한정한다. `SELECTION_CLOSED` 기록 이후에만 원본 Validation과 audit 모델 평가 경로에 진입한다.
- 기존 300/40/40 및 earliest-strict-minimum 선택을 유지한다. 재개/재시도 없는 단일 claim과 실패 기록 보존을 확인했다.
- 지원 부족은 counts를 남기고 metrics=None으로 구분한다. 같은 구간의 과거 관측 context 공유는 허용되지만 미학습 정답의 fit context 유입은 차단한다.
- build_payload/native_smoke 정적 검토상 과학 범위 차단 사유는 없다. SOURCE_REVISION은 builder가 실행 env에 git HEAD로 고정한다. builder가 runner의 git 추적 여부와 HEAD blob/local SHA 일치를 강제 검사한다. 원격 prelaunch native 검증은 아직 미확인이다.
- 한 seed·줄어든 학습량·dev 선택 재사용·과거 구간 결과 노출의 한계가 남는다. 수량 raw RMSE 선택과 시간 NLL 판독을 구분하며, 과적합과 분포 차이의 인과 기여율을 식별하지 않는다.

### 사전 발견 사항과 수정 확인
1. native dev 평가의 RNG 보호 누락: 새 runner의 `evaluate_preserving_rng`로 수정. 정상/예외 및 loader generator 회귀 테스트 통과.
2. 평가 시간 손실 비유한 값이 선택 RMSE만으로 가려질 가능성: wrapper의 필수 지표 finite/count 검사로 수정. 회귀 테스트 통과.
3. 기존 grid validator의 context_only 거부: 기존 실패 prepare_01을 보존하고 검증용 shadow만 허용. 실제 frame/loader target 불변·Test 거부·두 참조의 예외 복구 확인 후 prepare_02 통과.
4. QA raw-column 검사에서 canonical mark 변환을 빠뜨린 오류는 아래 실패 증적에 별도로 보존했다. 이는 생산 결함이나 새 학습 실패가 아니다.

### 완료 조건
- [x] 분할 경계·누락·중복·정답 노출 차단
- [x] native loader 순서·shuffle·context truncation
- [x] 같은 상태의 원본 수식/평가 지표와 RNG 보존
- [x] 평가 시점·내부 선택·빈/희소 지원 처리
- [x] 실제 fit 전용 통계 및 모델 초기화 전달 경로
- [x] 입력/소스 및 실제 CPU 실행 증적
- [ ] 원격 native 검증·첫 epoch·전체 run 후 독립 결과 QA — 본진의 이후 단계

### 파일 변경
QA는 `tests/test_taxi_internal_holdout_diagnostic.py`와 이 보고서만 변경했다. runner·기존 결과·.agents 파일은 수정하지 않았다. 추가 모델 학습·원격 호출·실제 Test 데이터 열람은 하지 않았다. 프로젝트 QA 역할은 Astra/high로 설정됐지만 현재 모델의 정확한 런타임 식별자는 노출되지 않았다.

## 검토한 최종 소스와 실행 증적

```json
{
  "reviewed_files": {
    "/Users/igwanhyeong/PycharmProjects/paper_research/paper/scripts/run_taxi_internal_holdout_diagnostic.py": "fd5cd048bfdf94a8b56fe062425053514efb0a1baabd6d917280edb76d7eb7fe",
    "/Users/igwanhyeong/PycharmProjects/paper_research/tests/test_taxi_internal_holdout_diagnostic.py": "e75a676739f630b038ba83d665eede5c30a42a807c8bb0e4bc6726e7184ffb24",
    "/Users/igwanhyeong/PycharmProjects/paper_research/reports/phen_taxi_internal_holdout_20261027_v1/SCIENCE_SPEC.json": "5ac466826ac7473c186b7379251809ba5b05ef7697a7f7199540d838f6ced501",
    "/Users/igwanhyeong/PycharmProjects/paper_research/reports/phen_taxi_internal_holdout_20261027_v1/build_payload.py": "f01f3ee5866495f4ccd33066562233e99cc1f7b5f8af41d58310c60322df71c2",
    "/Users/igwanhyeong/PycharmProjects/paper_research/reports/phen_taxi_internal_holdout_20261027_v1/native_smoke.py": "ec49fbaaac253967cdc0c12a0c48ef0e9bab7a2f5aeef47377c4499e0f9bd493"
  },
  "verified_receipts": {
    "tests_final": {
      "receipt_sha256": "f946b61123c686f50e93ee2972642053fbc8c304b0dd13d9c958ceaf0028a3cb",
      "actual_child_exit": 0,
      "wrapper_exit": 0,
      "streams_verified": true
    },
    "native_cpu_01": {
      "receipt_sha256": "58efc691ea802f540597cfb934601ade19a420263559eef0010866fe68b42afc",
      "actual_child_exit": 0,
      "wrapper_exit": 0,
      "streams_verified": true
    },
    "prepare_02": {
      "receipt_sha256": "62cbf952288bcc59c887086e6045c32a6d9c7c3c757597ec6c31f074e2599bb8",
      "actual_child_exit": 0,
      "wrapper_exit": 0,
      "streams_verified": true
    }
  },
  "native_cpu_result_sha256": "c2d1cccfc03a04f1cb457b8712d073145db46ee9d9012190872b7ba250967c13",
  "independent_test_tool_chunk": "d1057b",
  "independent_input_tool_chunk": "e2f155",
  "independent_prepare_stats_tool_chunk": "612c79"
}
```

## Train/Validation input and calendar audit

Local read-only checks; no model training/inference or Test input. Actual command exit is observed separately in the tool output.

```json
{
  "status": "PASS",
  "source_SHA": {
    "search_artifacts/shared_history_mlp_control_20261022_v1/execution_v2/input_snapshot/inputs/data/yellow_trip_hourly/train_validation.parquet": "54ce04a41e3cde688ce85f7ccda83773f47194c4f7ffbaf02001ea2390e881ab",
    "reports/phen_case_time_retrieval_20261027_v1/taxi/train_metadata.parquet": "64f1df6c43491d2650917e9626f9fcd652fa067f524b5e6614ee452822d4d471",
    "reports/phen_case_time_retrieval_20261027_v1/taxi/validation_metadata.parquet": "7b89e5cd4cb506da09e721b97fcaf9dc3eb3e6257c9da7cd357edde8a1515e43"
  },
  "raw_TrainValidation_rows": 46792,
  "common_time_origin": "2014-12-31 23:00:00",
  "seq_equals_hours_since_common_origin_all": true,
  "metadata_label_time_equals_raw_seq_all": true,
  "metadata_origin_equals_previous_raw_seq_all": true,
  "metadata_target_quantity_gap_exact": true,
  "target_counts": {
    "train": 38393,
    "validation": 8268
  },
  "partition_counts": {
    "fit": 23108,
    "dev": 7536,
    "audit": 7487,
    "purged": 262
  },
  "fit_actual_context_heldout_target_overlap": 0,
  "fit_context_distinct_raw_events": 23108,
  "maximum_fit_context_length": 168,
  "Test_or_full_data_read": false,
  "model_training_or_inference": false
}
```

## Recorded independent synthetic run

```json
{
  "started_utc": "2026-10-27T05:33:50.626776+00:00",
  "finished_utc": "2026-10-27T05:34:01.145440+00:00",
  "actual_child_exit": 0,
  "before_sha256": {
    "paper/scripts/run_taxi_internal_holdout_diagnostic.py": "fcd48087e0cdf4455f7f73b9bfd8864fc21e55f0b929721a392d317ad407cd15",
    "tests/test_taxi_internal_holdout_diagnostic.py": "e75a676739f630b038ba83d665eede5c30a42a807c8bb0e4bc6726e7184ffb24"
  },
  "after_sha256": {
    "paper/scripts/run_taxi_internal_holdout_diagnostic.py": "fcd48087e0cdf4455f7f73b9bfd8864fc21e55f0b929721a392d317ad407cd15",
    "tests/test_taxi_internal_holdout_diagnostic.py": "e75a676739f630b038ba83d665eede5c30a42a807c8bb0e4bc6726e7184ffb24"
  },
  "stdout_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "stderr_sha256": "0153a270f311cedec2010c5afc70cc0239ad868acc5a72762e00679cb4e19b8c",
  "synthetic_only": true,
  "real_training": false,
  "remote": false,
  "Test_data_read": false
}
```

stdout:
```text

```

stderr:
```text
test_actual_input_reader_rejects_synthetic_Test_rows (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_actual_input_reader_rejects_synthetic_Test_rows) ... /Users/igwanhyeong/.matplotlib is not a writable directory
Matplotlib created a temporary cache directory at /var/folders/w7/ybvrl2jd4hl24bpzjmfk9qh80000gn/T/matplotlib-oj0n4loc because there was an issue with the default path (/Users/igwanhyeong/.matplotlib); it is highly recommended to set the MPLCONFIGDIR environment variable to a writable directory, in particular to speed up the import of Matplotlib and to better support multiprocessing.

Fontconfig error: No writable cache directories
	/opt/homebrew/var/cache/fontconfig
	/Users/igwanhyeong/.cache/fontconfig
	/Users/igwanhyeong/.fontconfig
Matplotlib is building the font cache; this may take a moment.
ok
test_actual_model_factory_consumes_explicit_fit_statistics (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_actual_model_factory_consumes_explicit_fit_statistics) ... ok
test_aggregate_is_event_weighted_and_objectives_separate (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_aggregate_is_event_weighted_and_objectives_separate) ... ok
test_exclusive_writes_never_repair_failed_evidence (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_exclusive_writes_never_repair_failed_evidence) ... ok
test_grid_and_endpoint_only_validation (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_grid_and_endpoint_only_validation) ... ok
test_invalid_epoch_contract_fails_closed (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_invalid_epoch_contract_fails_closed) ... ok
test_native_metric_accumulation_matches_independent_event_arithmetic (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_native_metric_accumulation_matches_independent_event_arithmetic) ... ok
test_original_early_stopping_minimum_patience_and_earliest_ties (test_taxi_internal_holdout_diagnostic.EvaluationPolicyTests.test_original_early_stopping_minimum_patience_and_earliest_ties) ... ok
test_both_grid_validator_bindings_restore_after_failure (test_taxi_internal_holdout_diagnostic.LoaderExposureTests.test_both_grid_validator_bindings_restore_after_failure) ... ok
test_context_and_target_exposure_are_both_blocked (test_taxi_internal_holdout_diagnostic.LoaderExposureTests.test_context_and_target_exposure_are_both_blocked) ... ok
test_context_shadow_only_changes_validator_copy_not_loader_targets (test_taxi_internal_holdout_diagnostic.LoaderExposureTests.test_context_shadow_only_changes_validator_copy_not_loader_targets) ... ok
test_metadata_matches_actual_native_context_truncation (test_taxi_internal_holdout_diagnostic.LoaderExposureTests.test_metadata_matches_actual_native_context_truncation) ... ok
test_metadata_order_target_quantity_and_history_mismatch_fail (test_taxi_internal_holdout_diagnostic.LoaderExposureTests.test_metadata_order_target_quantity_and_history_mismatch_fail) ... ok
test_subset_order_and_seeded_shuffle_preserve_target_population (test_taxi_internal_holdout_diagnostic.LoaderExposureTests.test_subset_order_and_seeded_shuffle_preserve_target_population) ... ok
test_truncated_out_old_context_is_not_false_leakage (test_taxi_internal_holdout_diagnostic.LoaderExposureTests.test_truncated_out_old_context_is_not_false_leakage) ... ok
test_boundaries_and_crossing_queries_are_purged (test_taxi_internal_holdout_diagnostic.PartitionTests.test_boundaries_and_crossing_queries_are_purged) ... ok
test_duplicate_missing_target_and_invalid_chronology_fail (test_taxi_internal_holdout_diagnostic.PartitionTests.test_duplicate_missing_target_and_invalid_chronology_fail) ... ok
test_full_contract_counts_with_synthetic_identities (test_taxi_internal_holdout_diagnostic.PartitionTests.test_full_contract_counts_with_synthetic_identities) ... ok
test_all_CPU_random_streams_loader_generator_and_mixed_modes_restore (test_taxi_internal_holdout_diagnostic.RngTests.test_all_CPU_random_streams_loader_generator_and_mixed_modes_restore) ... ok
test_evaluation_wrapper_preserves_rng_and_rejects_nonfinite_time (test_taxi_internal_holdout_diagnostic.RngTests.test_evaluation_wrapper_preserves_rng_and_rejects_nonfinite_time) ... ok
test_exception_and_initialized_CUDA_state_are_restored (test_taxi_internal_holdout_diagnostic.RngTests.test_exception_and_initialized_CUDA_state_are_restored) ... ok
test_fit_only_cells_exact_support_boundary_and_missing_distinct (test_taxi_internal_holdout_diagnostic.SupportTests.test_fit_only_cells_exact_support_boundary_and_missing_distinct) ... ok
test_past_feature_change_is_log_difference_and_no_future_quantity (test_taxi_internal_holdout_diagnostic.SupportTests.test_past_feature_change_is_log_difference_and_no_future_quantity) ... ok
test_small_and_empty_report_groups_retain_counts_but_no_metrics (test_taxi_internal_holdout_diagnostic.SupportTests.test_small_and_empty_report_groups_retain_counts_but_no_metrics) ... ok

----------------------------------------------------------------------
Ran 24 tests in 9.830s

OK

```

## Preserved QA checker failure (not a prepare/training failure)

Tool chunk d3e2e1, actual exit 1: `AssertionError: mark` at the raw-column equality assertion. The checker incorrectly omitted the existing canonical `prepare_count_frame` transform (`mark=0`, `scale_residual=demand_qty`). No production defect is established by this assertion. Correction: compare against that original canonical transform, preserving all other columns. No prepare, model, fit, or inference rerun.

## Independent prepare_v2 statistics and input audit

```json
{
  "status": "PASS",
  "prepared_path": "search_artifacts/phen_taxi_internal_holdout_20261027_v1/local_qualification_v2/prepared",
  "prepared_inventory_sha256": "dc73c28e8cfacd66ee779dc4ab0a82d421d93c8ddcf179d9ccce0fed9fa52aa1",
  "prepared_artifact_count": 12,
  "data_sha256": "1da22d3a7978604e96eb2c97899a0dd73264e98a36d682c741be71769fb67cf0",
  "interface_sha256": "164212f4079fd2cbe05ac9a693912b44b804e94b96320caafa22d27e15e186d6",
  "original_raw_input_sha256": "54ce04a41e3cde688ce85f7ccda83773f47194c4f7ffbaf02001ea2390e881ab",
  "independent_arithmetic": "math.fsum mean and population variance; no model construction/fit",
  "fit_raw_rows": 23239,
  "fit_targets": 23108,
  "fitdev_raw_rows": 30906,
  "context_only_rows": 131,
  "canonical_input_values_equal_except_internal_split_labels": true,
  "canonical_transform": "Original prepare_count_frame mark=0, scale_residual=demand_qty",
  "fit_dev_target_IDs_unchanged": true,
  "fit_statistics_expected": {
    "train_log_mean": 2.8273515281958352,
    "train_log_std": 2.1733695104577553,
    "raw_scale": 222.49748267997762,
    "time_initial_location": 0.23476376121943576,
    "time_initial_scale": 0.5877281187709201
  },
  "maximum_differences": {
    "train_log_mean": 4.440892098500626e-16,
    "train_log_std": 0.0,
    "raw_scale": 0.0,
    "time_initial_location": 0.0,
    "time_initial_scale": 0.0
  },
  "interface_and_model_fields_bound_to_fit_statistics": true,
  "time_scale": 1,
  "time_sigma_floor": 0.001,
  "Test_reads": 0,
  "real_training": 0,
  "inference": 0
}
```
