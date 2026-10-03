"""Verify evidence bindings and decision invariants; never evaluates a model."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
load = lambda n: json.loads((OUT / n).read_text())
e = load('eligibility.json')
m = load('local_metadata.json')
p = load('public_metadata.json')
s = load('taxi_access_search.json')
ev = load('source_evidence_map.json')
rc = load('effective_rule_reconciliation.json')
checks = []


def check(name, value):
    checks.append({'check': name, 'passed': bool(value)})


check('four_datasets_unique', len(e['datasets']) == len({d['dataset'] for d in e['datasets']}) == 4)
check('none_ready_for_independent_evaluation', not any(d['ready_for_independent_evaluation'] for d in e['datasets']))
check('no_PASS_without_evidence', not any(d['status'] == 'PASS' for d in e['datasets']))
check('status_count_matches', {k: sum(d['status'] == k for d in e['datasets']) for k in e['counts']} == e['counts'])
check('no_execution_permit', e['execution_permit'] is False)
check('no_data_acquisition_or_new_eval', e['new_data_acquired'] is e['new_evaluation'] is False)
check('no_heldout_values_or_predictions', e['heldout_metric_or_prediction_files_opened'] is e['heldout_quantity_or_gap_columns_materialized'] is False)
check('table_read_allowlist', m['materialized_column_allowlist'] == ['site_cd', 'oper_part_no'])
check('mapping_5000_in_50_sites', m['intermittent']['selected_mapping']['rows'] == 5000 and m['intermittent']['selected_mapping']['unique_sites'] == 50)
check('raw_inter_source_absent_recorded', m['intermittent']['raw_source_exists'] is False)
check('inter_future_date_not_hidden', m['intermittent']['events']['calendar_date_footer_only']['demand_dt']['max'] == '20300318' and '2030' in e['datasets'][1]['source_period_metadata']['local_derived_max'])
check('site_cluster_amendment_restored', e['datasets'][1]['resampling']['primary'] == 'whole_site_cd_cluster')
check('taxi_source_january_only_metadata', m['taxi'][0]['calendar_date_footer_only']['tpep_pickup_datetime']['min'].startswith('2015-01-01') and m['taxi'][0]['calendar_date_footer_only']['tpep_pickup_datetime']['max'].startswith('2015-01-31'))
columns = next(x['retained'] for x in p['entries'] if x['kind'] == 'columns')
check('official_taxi_GPS_columns_verified', {'pickup_datetime', 'pickup_longitude', 'pickup_latitude'} <= {x['fieldName'] for x in columns})
check('public_api_rows_not_requested', all(x['target_rows_requested'] is False for x in p['entries']))
check('taxi_conditional_with_user_unsure', e['datasets'][0]['status'] == 'conditional_candidate' and e['datasets'][0]['candidate']['prior_access_status'] == 'user_unsure')
check('user_statement_preserved', e['datasets'][0]['candidate']['user_statement_provenance']['exact_reply'] == '확인이 필요함')
check('bounded_search_not_nonaccess_attestation', s['nonaccess_attested'] is False and s['prior_access_status'] == 'user_unsure')
check('bounded_search_no_result_files', s['heldout_metric_prediction_or_mixed_result_files_opened'] is False and s['notebook_outputs_read'] is False)
check('bounded_search_clean_parse', not s['errors'])
check('RAF_license_unknown_not_granted', next(x['retained'] for x in p['entries'] if x['kind'] == 'repository')['license'] is None and e['datasets'][3]['rights']['verified_permission_grant'] is False)
check('legacy_taxi_insta_not_untouched', all(e['datasets'][i]['existing_frozen_test']['status'] == 'ineligible' for i in [0, 2]))
check('prior_design_preserved', rc['prior_design_modified'] is False)
ids = {x['id'] for x in ev['evidence']}
check('dataset_evidence_resolves', all(set(d['evidence']) <= ids for d in e['datasets']))
for r in ev['evidence']:
    if 'path' in r and 'sha256' in r:
        check('source_sha:' + r['id'], hashlib.sha256((ROOT / r['path']).read_bytes()).hexdigest() == r['sha256'])
check('README_28day_block_limit', '네 개뿐' in (OUT / 'README.md').read_text())
check('README_user_unsure', 'prior_access_status=user_unsure' in (OUT / 'README.md').read_text())
artifacts = [{'path': str(x.relative_to(ROOT)), 'sha256': hashlib.sha256(x.read_bytes()).hexdigest()}
             for x in sorted(OUT.iterdir()) if x.is_file() and x.name not in ['verification.json', 'independent_review.json']]
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if all(c['passed'] for c in checks) else 'FAIL',
          'meaning': 'Internal evidence/decision consistency passed; no dataset independence PASS implied.',
          'passed': sum(c['passed'] for c in checks), 'total': len(checks), 'checks': checks, 'artifacts': artifacts}
(OUT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'passed': report['passed'], 'total': report['total']}))
if report['status'] != 'PASS':
    raise SystemExit(1)
