"""Check local preparation evidence. No remote access, inference, or metric-file discovery."""
import csv
import hashlib
import itertools
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT / 'reports/titantpp_independent_evaluation_protocol_20261001_v1'

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def main():
    checks = {}
    def check(name, value):
        checks[name] = bool(value)

    audit = read(OUT/'checkpoint_audit.json')
    registry = read(OUT/'evaluation_registry.json')
    effective = read(OUT/'protocol_effective.json')
    draft = read(OUT/'execution_contract_draft.json')
    prior = read(PRIOR/'protocol.json')
    prior_rows = read(PRIOR/'checkpoint_manifest.json')['rows']
    key = lambda r: (r['dataset'], r['model'], r['seed'])
    check('36_new_CPU_audits_passed', audit['status']=='passed' and len(audit['rows'])==36 and
          len({key(r) for r in audit['rows']})==36 and all(r['status']=='passed' and r['strict_cpu_loading'] and
          r['finite_tensors'] and r['first_strict_validation_RMSE_selection_verified'] and
          r['historical_json_records_unchanged'] for r in audit['rows']))
    check('selected_weight_audit_not_optimizer_state_audit', audit['new_forward_calls']==0 and
          all(r['optimizer_state_audit'].startswith('not_applicable_selected_checkpoint') for r in audit['rows']))
    files = checkpoint_count = file_bytes = 0
    bytes_match = archive_match = read_only = True
    for host in ['5080','5090']:
        dest = ROOT/f'search_artifacts/titantpp_independent_evaluation_preparation_20261001_v1/{host}/attempt1'
        manifest = read(dest/'original/collection_manifest.json')
        receipt = read(dest/'retrieval_receipt.json')
        archive_match &= sha(dest/'original.tar') == receipt['archive_sha256']
        read_only &= not manifest['remote_writes'] and not manifest['gpu_calls'] and not manifest['data_rows_or_test_metrics_read']
        files += len(manifest['files']); checkpoint_count += receipt['checkpoints']
        for rel, rec in manifest['files'].items():
            p = dest/'original'/rel
            bytes_match &= p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256']
            file_bytes += p.stat().st_size
    check('755_retrieved_files_byte_SHA_and_sizes_match', bytes_match and files==755 and file_bytes==27812946)
    check('two_archives_SHA_match_and_36_checkpoints', archive_match and checkpoint_count==36)
    check('retrieval_was_read_only_no_GPU_or_data_rows', read_only)
    expected = set(itertools.product(prior['datasets'], prior['models'], prior['seeds']))
    check('complete_84_condition_panel_no_duplicates', len(registry['rows'])==84 and {key(r) for r in registry['rows']}==expected)
    check('all_84_selected_file_SHA_match', all(sha(ROOT/r['checkpoint_path'])==r['checkpoint_file_sha256'] for r in registry['rows']))
    before = {key(r):r for r in prior_rows}
    check('all_original_selections_and_tensor_identities_preserved', all(
        all(r[k]==before[key(r)][k] for k in ['selected_epoch','state_tensor_sha256','contract_canonical_sha256','selection']) and
        r.get('source_closure_sha256',r.get('legacy_source_manifest_sha256')) ==
        before[key(r)].get('source_closure_sha256',before[key(r)].get('legacy_source_manifest_sha256'))
        for r in registry['rows']))
    source_files = 0
    source_match = True
    for bundle in registry['bundles'].values():
        contract = read(ROOT/bundle['contract_path'])
        source_match &= canonical(contract)==bundle['canonical_sha256']
        source_match &= canonical(bundle['source_files'])==bundle['source_closure_sha256']
        source_match &= contract['source']['files']==bundle['source_files']
        for rel,h in bundle['source_files'].items():
            source_files += 1
            source_match &= sha(ROOT/bundle['source_root']/rel)==h
    check('nine_frozen_source_bundles_and_contracts_match', source_match and len(registry['bundles'])==9)
    check('registry_keeps_heldout_execution_locked', not registry['heldout_execution_authorized'])
    smoke = read(OUT/'validation_smoke.json')
    smoke_bundles = [read(p) for p in (OUT/'validation_smoke').glob('*.json')]
    conditions = [x for bundle in smoke_bundles for x in bundle['checks']]
    check('CPU_validation_smoke_84_conditions_passed', len(smoke_bundles)==9 and len(conditions)==84 and
          {key(r) for r in conditions}==expected and all(b['status']=='passed' and b['device']=='cpu' and not b['heldout_read'] for b in smoke_bundles))
    check('target_perturbation_batch_and_parameter_checks_passed', all(
        all(r[k] for k in ['target_quantity_perturbation_preserves_prediction_and_time_loss',
                          'target_gap_perturbation_preserves_quantity_prediction','single_vs_batch_matches',
                          'parameters_unchanged','finite_outputs']) for r in conditions))
    check('462_sample_predictions_only_validation', smoke['prediction_rows']==462 and len(smoke['records'])==462 and
          all(r['split']=='validation' and r['evaluation_scope']=='validation_smoke' for r in smoke['records']) and
          not smoke['heldout_read'] and not smoke['full_endpoint_replay'])
    aligned = True
    for ds in prior['datasets']:
        groups = {}
        for r in smoke['records']:
            if r['dataset']==ds:
                groups.setdefault((r['model'],r['seed']), []).append((r['target_id'],r['raw_quantity'],r['recorded_gap'],r['entity_id'],r['seq']))
        aligned &= len(groups)==21 and all(v==next(iter(groups.values())) for v in groups.values())
    check('actual_sample_target_pairing_rechecked', aligned)
    suite = ET.parse(OUT/'unit_tests.xml').getroot().find('testsuite')
    check('14_unit_tests_no_failures_errors_or_skips', suite.get('tests')=='14' and
          all(suite.get(k)=='0' for k in ['errors','failures','skipped']))
    stats = read(OUT/'validation_statistics_smoke.json')
    check('statistics_smoke_bound_to_current_code', stats['status']=='passed' and not stats['heldout_read'] and
          stats['bootstrap_code_sha256']==sha(OUT/'paired_statistics.py'))
    check('validation_stats_small_unit_guards_worked',
          all(stats['results'][ds]['draws_completed']==0 for ds in ['yellow_trip_hourly','intermittent_frozen_5000']) and
          all(stats['results'][ds]['draws_completed']==10000 and stats['results'][ds]['interval_status']=='computed_descriptive'
              for ds in ['insta_market_basket','raf_spare_parts']))
    lineage = read(OUT/'lineage_review.json')
    check('historical_manifest_78_SHA_rechecked', len(lineage['historical_manifests_rechecked'])==78 and
          all(r['manifest_unchanged'] for r in lineage['historical_manifests_rechecked']))
    check('RAF_25_contracts_validation_only', len(lineage['raf']['historical_launch_contracts'])==25 and
          all(r['metadata']['evaluation_scope']=='validation_only' and r['metadata']['held_out_test_evaluated'] is False
              for r in lineage['raf']['historical_launch_contracts']))
    check('heldout_access_flags_false', not lineage['access_scope']['heldout_metric_or_prediction_files_opened'] and
          not lineage['access_scope']['heldout_quantity_or_gap_columns_read'])
    mapping = list(csv.DictReader((OUT/'intermittent_entity_site_mapping.csv').open()))
    check('5000_entities_in_50_sites_mapping_preserved', len(mapping)==5000 and
          len({r['site_cd'] for r in mapping})==50 and len({r['oper_part_no'] for r in mapping})==5000 and
          sha(OUT/'intermittent_entity_site_mapping.csv')==lineage['intermittent']['mapping_sha256'])
    check('independence_is_explicitly_unresolved', all(v=='not_established' for v in lineage['independence_status'].values()) and
          lineage['researcher_attestation']['status']=='pending')
    amendment = read(OUT/'amendment.json')
    check('site_cluster_amendment_bound_to_parent_and_metadata',
          amendment['parent_protocol_sha256']==sha(PRIOR/'protocol.json') and
          amendment['evidence']['sha256']==sha(OUT/'lineage_review.json') and
          effective['resampling']['units']['intermittent_frozen_5000']=='whole_site_cd_cluster' and
          effective['resampling']['intermittent_sensitivity_unit']=='whole_oper_part_no_cluster')
    check('old_model_metric_selection_and_anchor_protocol_preserved', all(effective[k]==prior[k] for k in
          ['models','datasets','seeds','representative','selection','primary_metric','secondary_metrics','aggregation','fixed_validation_selected_anchors','prediction_protocol']))
    check('execution_draft_non_executable_and_unapproved', draft['status']=='non_executable_draft' and not draft['execution_authorized'] and
          draft['execution_entrypoint'] is None and draft['population']['target_manifest_sha256'] is None and
          draft['resource_caps']['deadline_utc'] is None and not effective['execution_authorized'])
    check('draft_links_actual_registry_protocol_and_code', draft['registry_sha256']==sha(OUT/'evaluation_registry.json') and
          draft['protocol_sha256']==sha(OUT/'protocol_effective.json') and
          draft['preparation_entrypoint_sha256']==sha(OUT/'evaluate_validation.py') and
          draft['statistics_sha256']==sha(OUT/'paired_statistics.py'))
    inventory = read(OUT/'runtime_5080_readonly.json')
    check('runtime_inventory_not_native_GPU_qualification', inventory['returncode']==0 and
          not inventory['result']['runtime_qualification'] and not inventory['result']['gpu_tensor_or_model_calls'] and
          not inventory['result']['remote_writes'])
    protected = read(OUT/'preserved_inputs.json')
    check('prior_protocol_manuscript_and_dirty_scientific_files_unchanged', all(sha(ROOT/p)==h for p,h in protected.items()))
    changes = read(OUT/'changes.json')['tracking_files']
    check('tracking_before_and_after_SHA_preserved', all(sha(OUT/'before'/p)==h['before_sha256'] and
          sha(ROOT/p)==h['after_sha256'] for p,h in changes.items()))
    baseline = read(ROOT/'reports/titantpp_baseline_reset_20261001_v1/baseline.json')
    old_baseline = read(OUT/'before/reports/titantpp_baseline_reset_20261001_v1/baseline.json')
    check('tracking_84_complete_no_old_observation_time_refresh', baseline['independent_evaluation']['local_selected_binaries']==84 and
          baseline['independent_evaluation']['legacy_selected_binary_retrieval_and_CPU_audit_pending']==0 and
          baseline['status_observed_kst']==old_baseline['status_observed_kst'])
    missing = []
    for p in [OUT/'README.md', OUT/'report.md', ROOT/'reports/titantpp_baseline_reset_20261001_v1/README.md']:
        for ref in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if ref.startswith(('https:', 'http:', '#')): continue
            target = ref.split('#',1)[0]
            if target=='verification.json': continue  # produced below
            if not (p.parent/target).exists(): missing.append((str(p),target))
    check('local_report_links_resolve', not missing)
    sources = read(OUT/'sources.json')
    check('preparation_source_evidence_SHA_manifest_matches', all(sha(ROOT/p)==h for p,h in sources['files'].items()))
    result = {'status':'passed' if all(checks.values()) else 'failed',
              'verified_utc':datetime.now(timezone.utc).isoformat(), 'checks_passed':sum(checks.values()),
              'checks_total':len(checks), 'checks':checks, 'retrieved_files':files, 'retrieved_bytes':file_bytes,
              'frozen_source_file_occurrences':source_files, 'selected_conditions':84,
              'heldout_read':False, 'new_inference_during_verification':False,
              'new_remote_calls_during_verification':False, 'missing_local_links':missing,
              'readiness':'CPU_preparation_passed_population_and_execution_contract_pending'}
    (OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
    if not all(checks.values()):
        print('FAILED:',[k for k,v in checks.items() if not v]); raise SystemExit(1)

if __name__=='__main__':
    main()
