"""Record preparation status without authorizing held-out evaluation."""
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'reports/titantpp_independent_evaluation_protocol_20261001_v1'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def main():
    now = datetime.now(timezone.utc).isoformat()
    protocol = read(PRIOR / 'protocol.json')
    registry = read(OUT / 'evaluation_registry.json')
    lineage = read(OUT / 'lineage_review.json')
    protected = read(PRIOR / 'preserved_inputs.json')
    assert all(sha(ROOT / p) == h for p, h in protected.items())
    prior_files = ['protocol.json', 'protocol.md', 'checkpoint_manifest.json', 'checkpoint_manifest.csv',
                   'dataset_manifest.json', 'split_access_ledger.json', 'verification.json', 'preserved_inputs.json']
    protected.update({str((PRIOR / p).relative_to(ROOT)): sha(PRIOR / p) for p in prior_files})
    write('preserved_inputs.json', protected)
    amendment = {
        'created_utc': now, 'status': 'analysis_design_amendment_before_heldout_performance_access',
        'parent_protocol': str((PRIOR / 'protocol.json').relative_to(ROOT)),
        'parent_protocol_sha256': sha(PRIOR / 'protocol.json'),
        'authorization_scope': 'User requested execution preparation; this is not held-out execution authorization.',
        'evidence': {'path': 'lineage_review.json', 'sha256': sha(OUT / 'lineage_review.json'),
                     'entity_site_mapping_sha256': sha(OUT / 'intermittent_entity_site_mapping.csv')},
        'changes': [{
            'dataset': 'intermittent_frozen_5000',
            'old_primary_unit': 'whole_oper_part_no_cluster',
            'new_primary_unit': 'whole_site_cd_cluster',
            'sensitivity_unit': 'whole_oper_part_no_cluster',
            'reason': 'Metadata confirms 5000 series nested within 50 sites; resample sites together to retain within-site dependence.',
            'coverage_caveat': 'Conditional intervals assume suitable independence across sampled sites and do not remove cross-site common shocks.'}],
        'selection_model_checkpoint_metrics_changed': False,
        'heldout_metrics_predictions_or_target_values_read': False,
        'prior_protocol_preserved': True,
    }
    write('amendment.json', amendment)
    effective = copy.deepcopy(protocol)
    effective.update({
        'schema': 'titantpp_independent_evaluation_design_preparation_v2',
        'updated_utc': now, 'status': 'CPU_preparation_passed_population_and_execution_contract_pending',
        'execution_authorized': False,
        'parent_protocol': amendment['parent_protocol'],
        'parent_protocol_sha256': amendment['parent_protocol_sha256'],
        'amendment': 'amendment.json', 'amendment_sha256': sha(OUT / 'amendment.json'),
        'checkpoint_manifest': 'evaluation_registry.json',
        'checkpoint_manifest_sha256': sha(OUT / 'evaluation_registry.json'),
        'dataset_manifest': str((PRIOR / 'dataset_manifest.json').relative_to(ROOT)),
        'split_access_ledger': str((PRIOR / 'split_access_ledger.json').relative_to(ROOT)),
        'lineage_followup': 'lineage_review.json',
        'remaining_requirements': [
            'resolve_researcher_access_attestation_and_split_lineage_or_label_retrospective_reevaluation',
            'freeze_eligible_population_and_target_identity_manifest',
            'complete_full_population_wrapper_and_native_runtime_validation',
            'freeze_actual_host_runtime_resource_caps_deadline_outputs',
            'obtain_explicit_evaluation_execution_and_performance_access_authorization'],
    })
    effective['resampling']['units']['intermittent_frozen_5000'] = 'whole_site_cd_cluster'
    effective['resampling']['intermittent_sensitivity_unit'] = 'whole_oper_part_no_cluster'
    effective['resampling']['analysis_runtime'] = {'python': '3.12.10', 'numpy': '2.3.1', 'device': 'cpu',
                                                   'status': 'validated_preparation_runtime_to_bind_in_execution_contract'}
    effective['resampling']['implementation_and_real_data_CIs_status'] = '14_tests_passed_and_validation_sample_pipeline_checked_no_benchmark_or_heldout_CIs'
    effective['resampling']['sort'] = 'primary_cluster_key_string_ascending_or_calendar_hour_ascending'
    write('protocol_effective.json', effective)
    runtime = read(OUT / 'runtime_5080_readonly.json')
    draft = {
        'schema': 'titantpp_evaluation_execution_contract_draft_v1', 'created_utc': now,
        'status': 'non_executable_draft', 'execution_authorized': False,
        'purpose': 'Evaluate the original 84 selected models after population eligibility is decided.',
        'evaluation_registry': 'evaluation_registry.json', 'registry_sha256': sha(OUT / 'evaluation_registry.json'),
        'protocol': 'protocol_effective.json', 'protocol_sha256': sha(OUT / 'protocol_effective.json'),
        'main_conditions': 84, 'representative': protocol['representative'],
        'models': protocol['models'], 'datasets': protocol['datasets'], 'seeds': protocol['seeds'],
        'selection': protocol['selection'], 'endpoint': 'selected_only_no_reselection',
        'source_isolation': 'One child process per original frozen source bundle, checked against registry file SHA.',
        'candidate_host': {'host': '5080', 'gpu_uuid': runtime['expected_gpu_uuid'],
                           'python': runtime['expected_python'], 'inventory': 'runtime_5080_readonly.json',
                           'inventory_sha256': sha(OUT / 'runtime_5080_readonly.json'),
                           'native_qualification': 'pending', 'runtime_redeployment': False},
        'analysis_runtime': effective['resampling']['analysis_runtime'],
        'preparation_entrypoint': 'evaluate_validation.py',
        'preparation_entrypoint_sha256': sha(OUT / 'evaluate_validation.py'),
        'preparation_scope': 'CPU_validation_only_first_up_to_8_targets_from_2_sorted_entities_per_dataset',
        'statistics_entrypoint': 'paired_statistics.py', 'statistics_sha256': sha(OUT / 'paired_statistics.py'),
        'statistics_scope_lock': ['synthetic', 'validation_smoke', 'validation'],
        'cpu_smoke_passed_conditions': 84, 'prediction_rows': 462, 'unit_tests_passed': 14,
        'population': {'status': 'pending', 'split_route': None, 'target_manifest_sha256': None,
                       'researcher_attestation': lineage['researcher_attestation'],
                       'independence': lineage['independence_status'],
                       'existing_dataset_identities': effective['dataset_manifest']},
        'execution_entrypoint': None, 'server_output_root': None,
        'resource_caps': {'wall_seconds': None, 'cpu_seconds': None, 'gpu_memory_bytes': None,
                          'deadline_utc': None, 'status': 'must_be_measured_and_frozen_for_selected_population'},
        'prediction_protocol': protocol['prediction_protocol'],
        'no_training_or_parameter_updates': True, 'automatic_retry': False,
        'failure_policy': 'Preserve original outputs and errors; no population intersections, checkpoint changes, or favorable exclusions.',
        'required_outputs': ['execution_receipt', 'target_population_manifest', 'per_target_validation_or_approved_split_predictions',
                             'per_seed_metrics', 'paired_resampling_weights_digest', 'comparison_intervals',
                             'failure_and_missing_condition_records', 'runtime_and_resource_measurements', 'file_SHA_manifest'],
        'blocking_fields': ['eligible_population_and_label', 'target_manifest', 'researcher_access_record',
                            'full_population_execution_entrypoint', 'native_runtime_validation', 'resource_caps',
                            'deadline_and_output_root', 'explicit_performance_execution_authorization'],
    }
    write('execution_contract_draft.json', draft)
    write('preparation_issue.json', {
        'scope': 'validation_smoke_test_fixture', 'resolved': True,
        'initial_failure': 'Instacart target-gap perturbation exceeded original top-code observation support of 30 days.',
        'correction': 'Use in-domain target gap 1 or 2; original scientific code, data, observation contract and checkpoints unchanged.',
        'verification': 'All 84 condition smoke checks passed after fixture correction.',
        'training_failure': False, 'heldout_access': False,
    })

    tracking = [ROOT/'reports/titantpp_baseline_reset_20261001_v1/README.md',
                ROOT/'reports/titantpp_baseline_reset_20261001_v1/baseline.json',
                ROOT/'reports/titantpp_baseline_reset_20261001_v1/verification.json',
                ROOT/'reports/titantpp_manuscript_integration_20261001_v1/README.md', PRIOR/'README.md']
    for path in tracking:
        dest = OUT/'before'/path.relative_to(ROOT)
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    bp = ROOT/'reports/titantpp_baseline_reset_20261001_v1/baseline.json'
    baseline = read(bp)
    baseline['independent_evaluation'] = {
        'status': effective['status'], 'updated_utc': now,
        'protocol': str((OUT/'protocol_effective.json').relative_to(ROOT)),
        'protocol_sha256': sha(OUT/'protocol_effective.json'),
        'report': str((OUT/'report.md').relative_to(ROOT)),
        'registry': str((OUT/'evaluation_registry.json').relative_to(ROOT)),
        'selected_conditions': 84, 'local_selected_binaries': 84,
        'prior_CPU_audits_reused': 48, 'new_legacy_selected_binary_CPU_audits': 36,
        'legacy_selected_binary_retrieval_and_CPU_audit_pending': 0,
        'CPU_validation_sample_conditions': 84, 'CPU_validation_sample_prediction_rows': 462,
        'statistical_unit_tests_passed': 14, 'full_validation_endpoint_replay': False,
        'untouched_split_independence': 'not_established',
        'verification': str((OUT/'verification.json').relative_to(ROOT)),
        'read_only_remote_retrieval_and_runtime_inventory': True,
        'new_GPU_inference_or_training': False, 'new_CPU_validation_sample_inference': True,
        'heldout_performance_access': False, 'execution_authorized': False,
    }
    baseline['next_action'] = ('Resolve actual prior test viewing/selection use and population lineage; choose independent cohort or retrospectively locked test scope. '
                              'Then complete full-population evaluator/native checks and freeze runtime, resource limits, deadline and outputs for separately authorized evaluation.')
    baseline['parallelism'] = 'Binary retrieval and metadata investigation completed; remaining population/contract/evaluator steps share dependencies and proceed sequentially.'
    bp.write_text(json.dumps(baseline, ensure_ascii=False, indent=2)+'\n')
    banner = ('**실행 준비 보완 — 완료:** 외부4종36개 selected binary 회수·CPU 감사와 기존48개 재검증으로 총84개를 준비했다. '
              '84조건 CPU validation 표본·통계14검사를 통과했다. 실제 test 접근 이력과 평가 모집단은 미확정이며, '
              '이번에는 원고·benchmark 수치·GPU 학습을 변경하지 않았다. '
              '[준비 결과와 다음 작업](../titantpp_independent_evaluation_preparation_20261001_v1/report.md).\n\n')
    for path in [tracking[3], PRIOR/'README.md']:
        if not path.read_text().startswith('**실행 준비 보완 — 완료:**'):
            path.write_text(banner + path.read_text())
    # The original protocol README remains an explicitly dated design-stage history.
    path = PRIOR/'README.md'
    text = path.read_text().replace('# TitanTPP 독립 평가 준비\n', '# TitanTPP 독립 평가 준비 — 최초 설계 당시 기록\n\n아래 48개 준비·36개 미회수 서술은 최초 설계 시점 이력이다. 현재 상태는 위 후속 보고서를 따른다.\n', 1)
    path.write_text(text)
    mi = tracking[3]
    text = mi.read_text()
    text = text.replace('로컬48개checkpoint의 byte SHA를 대조했으며, 기존 외부4종36개 selected binary는 원본 회수·CPU검증이 남았다.',
                        '후속 준비에서 기존 외부4종36개 selected binary의 회수·CPU검증까지 완료해 총84개를 준비했다.')
    text = text.replace('다음은 split 계보·실제 접근 이력과36개binary 준비를 마무리하고, 합성·validation 기반 실행기 검증과 실제 평가 계약을 확정하는 작업이다.',
                        '원본84개 준비와 합성·CPU validation 표본 검증을 완료했다. 다음은 실제 test 접근 이력·평가 모집단을 정하고, 전체 평가 wrapper와 native 검증·자원 상한을 포함한 실제 실행 계약을 확정하는 작업이다.')
    mi.write_text(text)
    bp_md = tracking[0]
    text = bp_md.read_text().replace('— 독립 평가 기준 정리 완료', '— 독립 평가 CPU 실행 준비 완료')
    start = text.index('**split 접근 이력과 원본 준비를 마무리한다')
    text = text[:start] + '''**원본 checkpoint와 CPU 평가 경로를 준비한다 — 완료**

- 외부4종36개 selected checkpoint와 원본755파일을 회수·SHA·CPU 검증했다. 기존48개와 합쳐 총84개 파일을 준비했다. [회수·검증 보고서](../titantpp_independent_evaluation_preparation_20261001_v1/report.md).
- 9개 동결 source에서84조건의 validation 표본462행을 확인하고, 통계 코드14검사를 통과했다. 전체 validation 재평가나 held-out 성능 검증과는 구별한다.
- Intermittent의50사업장 구조를 반영해 기본 bootstrap을 사업장 단위로 보완했다. 원고·기존 benchmark 수치·과학 source는 유지했다.

**평가 데이터의 성격을 확정한다 — 다음 작업 / 외부 작업 대기**

- metadata 계보 조사를 마쳤으나 실제 test 열람·모델 선택 사용 여부는 연구자 확인이 남았다. 네 데이터의 미접근 독립성은 미확정이다. 기존 split의 잠금 재평가와 미사용 코호트의 독립 평가를 구분해 선택한다.

**전체 평가 실행기와 실제 계약을 확정한다 — 다음 작업**

- 적격한 target 모집단을 고정한 뒤 전체 평가 wrapper·native runtime·자원 상한·마감·출력경로를 확정한다. 현재 CPU validation 실행기는 준비용이며 [실행 계약 초안](../titantpp_independent_evaluation_preparation_20261001_v1/execution_contract_draft.json)은 실행 불가 상태다.
- 공통 모집단과 계약에 의존하므로 현재 세션에서 순서대로 진행한다. 원본 회수·Core 감사·효율 측정은 반복할 필요가 없다.

**고정 평가와 제출본을 완성한다 — 이후 작업 / 실제 실행·성능 열람·제출 승인 필요**

- 검토 가능한 계약에 따라 승인된 평가를 실행하고 불리한 결과·실패도 보존한다. 결과의 적용 범위를 초록·결론과 대조한 뒤 제출본을 편집한다.
- 이번 준비에서는 읽기 전용 원본 회수·Runtime inventory와 로컬 CPU validation 표본 추론만 추가했다. GPU 작업·held-out 성능 열람·스케줄 변경·commit/push·외부 게시·제출은 수행하지 않았다.
'''
    bp_md.write_text(text)
    verification_path = tracking[2]
    verification = read(verification_path)
    verification['independent_evaluation_preparation'] = {
        'updated_utc': now, 'evidence': str((OUT/'verification.json').relative_to(ROOT)),
        'selected_binaries': 84, 'new_CPU_audits': 36, 'CPU_validation_sample_conditions': 84,
        'statistical_tests': 14, 'execution_authorized': False,
        'heldout_read': False, 'independence_not_established': True,
        'old_observation_and_verification_fields_are_historical': True,
    }
    verification['plan_sha256'] = sha(bp_md)
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2)+'\n')
    changed = {str(p.relative_to(ROOT)): {'before_sha256':sha(OUT/'before'/p.relative_to(ROOT)),
                                         'after_sha256':sha(p)} for p in tracking}
    write('changes.json', {'updated_utc': now, 'tracking_files': changed,
                           'manuscript_and_scientific_code_unchanged': True,
                           'prior_protocol_and_selection_manifest_preserved': True,
                           'new_evaluation_scope': 'CPU_validation_samples_only'})
    print(json.dumps({'status': 'preparation_recorded', 'selected_binaries': len(registry['rows']),
                      'execution_authorized': False}))

if __name__ == '__main__':
    main()
