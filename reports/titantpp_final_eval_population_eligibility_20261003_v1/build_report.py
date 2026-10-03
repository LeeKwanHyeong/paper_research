"""Assemble metadata-only decisions; does not execute evaluation or acquisition."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


common_gates = [
    'source_owner_rights_and_actual_access_verified',
    'exact_population_period_entities_event_definition_and_asof_rule_frozen',
    'prior_access_attestation_and_bounded_machine_history_reconciled',
    'overlap_with_all_development_populations_checked_using_identifiers_only',
    'raw_and_derived_metadata_manifests_bound_before_outcome_unsealing',
    'unchanged_checkpoint_source_input_vocabulary_scaling_and_context_policy_bound',
    'same_population_and_past_only_input_access_for_every_method',
    'separate_evaluation_execution_permit_host_budget_deadline_and_one_shot_release',
]
taxi = {
    'dataset': 'yellow_trip_hourly', 'name': 'Taxi', 'status': 'conditional_candidate',
    'ready_for_independent_evaluation': False,
    'existing_frozen_test': {'status': 'ineligible', 'for_claim': 'untouched_independent_confirmation',
        'reason': 'Prior accessed test has exactly the same event identifiers; retained descriptive legacy evaluation remains possible under a separate permit.'},
    'source_period_metadata': {'local_raw_min': '2015-01-01 00:00:00', 'local_raw_max': '2015-01-31 23:59:59',
        'derived_min': '2015-01-01 00:00:00', 'derived_max': '2015-01-31 23:00:00',
        'coverage': 'Parquet footer bounds only; completeness or equality with the full official January archive not established.'},
    'event_definition': 'Positive pickup count in each 0.02-degree GPS grid cell/hour; shared sequence is the sorted set of observed global hours.',
    'actual_cohort_rule': 'MIN_ACTIVE_BUCKETS=72 over the complete January source before the split; MAX_SERIES=None; longitude [-75,-72], latitude [40,42].',
    'candidate': {'source': 'https://data.cityofnewyork.us/api/views/2yzn-sicd.json',
        'dataset_id': '2yzn-sicd', 'source_name': '2015 Yellow Taxi Trip Data',
        'metadata_schema_status': 'verified_pickup_datetime_pickup_longitude_pickup_latitude',
        'field_alias': {'pickup_datetime': 'tpep_pickup_datetime'},
        'period_start_inclusive': '2015-02-01T00:00:00', 'period_end_exclusive': '2015-03-01T00:00:00',
        'period_selection_reason': 'First full month after the current source, selected before any February outcomes; do not choose another month on model performance.',
        'asof': '2015-02-01T00:00:00', 'cohort': 'Freeze existing January-selected grid IDs at the cutoff; never reselect cells by February activity.',
        'context_policy': 'Use fixed January observed history as initial context, then reveal February past events sequentially; no parameter or scaler refit.',
        'training_required': False, 'training_required_caveat': 'Only if the verified source and frozen event/input contract are compatible. Replacing GPS grid with taxi zones changes the task and is not covered.',
        'rows_downloaded': False, 'event_coverage_verified': False, 'overlap_verified': False,
        'researcher_prior_access_attestation': 'user_unsure',
        'prior_access_status': 'user_unsure',
        'user_statement': '확인이 필요함 — user did not attest that February2015 was unused.',
        'user_statement_provenance': {'received_via': 'parent/root message relaying direct user reply in this task', 'received_date_kst': '2026-10-03', 'exact_reply': '확인이 필요함', 'exact_user_reply_timestamp': None, 'timestamp_note': 'Exact user response timestamp was not provided; no invented timestamp.'},
        'rights': 'Official publicly accessible TLC/Open Data metadata verified; bind the applicable data-use terms and research access attestation before acquisition.',
        'modern_zone_only_source': 'ineligible_for_direct_GPS_grid_reuse'},
    'remaining_checks': common_gates + ['official_February_coverage_without_outcome_screening',
        'January_source_lineage_and_preexisting_February_copies_or_use',
        'GPS_rounding_grid_boundaries_timestamp_timezone_and_global_hour_index_compatibility',
        'shared_calendar_block_resampling_feasibility_for_28_day_window'],
    'evidence': ['L_taxi_notebook', 'L_taxi_footer', 'L_prior_lineage', 'W_taxi_columns', 'W_taxi_catalog', 'W_tlc_dictionary'],
}
inter = {
    'dataset': 'intermittent_frozen_5000', 'name': 'Intermittent', 'status': 'unknown',
    'ready_for_independent_evaluation': False,
    'existing_frozen_test': {'status': 'unknown', 'for_claim': 'untouched_independent_confirmation',
        'reason': 'Legacy event construction and source IDs are not bound to the current source; literal non-overlap is insufficient.'},
    'source_period_metadata': {'local_derived_min': '2018-01-01', 'local_derived_max': '2030-03-18',
        'calendar_note': 'Dates extend beyond the 2026-10-03 review date. This does not prove synthetic generation; real, shifted, or generated date semantics require source-owner evidence.',
        'source_path': '/Users/igwanhyeong/data/demand_engine/data_v2/intermittent.parquet',
        'source_available_local': False},
    'event_definition': 'Each positive source order_qty row; site_cd::part_no entity; floor weeks from each series first date +1; no legacy burst merging.',
    'actual_cohort_rule': 'At least20 positive events over full source; sample5000 stratified within50 sites by full-source event_count and a nominal first70%-event train_qty_p95; deterministic seed20260810.',
    'asof_risk': 'Full-source event count, eligibility and event quantile strata use future event existence. A 70%-of-total boundary is retrospective. Do not reuse these rules as prospective cutoff eligibility.',
    'candidate': {'source': None, 'status': 'unknown', 'required_owner_deliverable':
        'Original extraction/generator revision, raw SHA, date semantics, site/part mapping to legacy data, use authorization, availability cutoffs and a separately sealed cohort.',
        'conditional_route': 'After provenance resolution, use the same pre-cutoff5000 IDs on newly observed later source events, or independently sourced sites chosen without future activity/quantity criteria.',
        'new_dates_not_assumed_available': True,
        'if_synthetic': 'Bind a predeclared generator and independent random stream/replicate; describe simulation generalization, not unseen real-world follow-up.',
        'training_required': 'not_for_compatible_new_events_with_frozen_inputs; a changed event definition or feature/normalization contract requires a separately authorized study'},
    'resampling': {'primary': 'whole_site_cd_cluster', 'sensitivity': 'whole_oper_part_no_cluster',
        'evidence': '10/1 amendment and rechecked5000-series/50-site identifier mapping',
        'caveat': 'Cross-site common shocks may remain; site resampling does not guarantee independence.'},
    'remaining_checks': common_gates + ['recover_raw_source_and_generation_or_extraction_provenance',
        'resolve_2018_to2030_date_semantics', 'bind_legacy_entity_and_event_mapping', 'freeze_asof_eligibility_without_full_horizon_event_counts'],
    'evidence': ['L_inter_builder', 'L_inter_metadata', 'L_prior_lineage', 'L_preparation_amendment'],
}
insta = {
    'dataset': 'insta_market_basket', 'name': 'Instacart', 'status': 'unknown',
    'ready_for_independent_evaluation': False,
    'existing_frozen_test': {'status': 'ineligible', 'for_claim': 'untouched_independent_confirmation',
        'reason': 'Available legacy test event keys equal current test event keys and prior access was confirmed.'},
    'source_period_metadata': {'release': '2017', 'calendar_period': None,
        'reason': 'Official release contains relative capped inter-order days, not real calendar dates; no genuine later calendar partition can be inferred.'},
    'event_definition': 'Current preprocessing source counts product rows per order, joins prior+train source labels, then sums these basket sizes per user and floor(cumulative capped relative days). Quantity is item-row count per reconstructed active day, not physical item units or necessarily one order. Historical build-source identity must still be bound.',
    'official_source_semantics': 'Orders provide relative days capped at30; anonymized biased sample; original competition test labels are not established as available research labels.',
    'candidate': {'source': None, 'status': 'unknown', 'new_official_release_verified': False,
        'same_release_mirror': 'ineligible_as_new_independent_population',
        'cornell_uchoice_derivative': 'ineligible_as_new_independent_population; explicitly derived from2017 source',
        'required_owner_deliverable': 'A genuinely separate consented basket-history extract with cross-release user/event deduplication, common item-row and relative-time semantics, and valid research-use rights.',
        'training_required': 'not_if_all_frozen_inputs_and_event_units_are_compatible; a different retailer or altered definition is a separately scoped external-domain study'},
    'rights': {'primary_terms_url': 'https://gist.github.com/jeremystan/582eba13d6ee27ed465c43dc78934700',
        'source_terms_date': '2017-05-15', 'verified_terms_summary': 'Non-commercial, no third-party redistribution without terms compliance; publisher may modify/suspend access.',
        'current_entitlement_verified': False, 'reason': 'Terms contain availability-dependent language; current dataset download page could not be resolved by web tool and competition is disabled at host request. Do not infer permission from a mirror.'},
    'remaining_checks': common_gates + ['independent_dataset_exists_and_allowed_use_current',
        'no_overlap_with_2017_release_or_derivatives', 'capped_relative_day_and_daily_aggregation_compatibility'],
    'evidence': ['L_insta_notebook', 'L_insta_stat', 'L_prior_lineage', 'W_insta_release', 'W_insta_dictionary', 'W_insta_terms', 'W_insta_competition', 'W_insta_derivative'],
}
raf = {
    'dataset': 'raf_spare_parts', 'name': 'RAF', 'status': 'unknown',
    'ready_for_independent_evaluation': False,
    'existing_frozen_test': {'status': 'unknown', 'for_claim': 'untouched_independent_confirmation',
        'reason': 'Bounded local25 launch-contract review found validation-only evaluation but is not complete human/machine access history; target-summary exposure was previously logged.'},
    'source_period_metadata': {'start': '1996-01', 'end': '2002-12', 'months': 84,
        'train': '1996-01..2000-10', 'validation': '2000-11..2001-11', 'test': '2001-12..2002-12'},
    'event_definition': 'Positive monthly demand events for archived parts; month index retained from JAN96 through DEC02. No later months exist in this workbook.',
    'candidate': {'source': None, 'status': 'unknown', 'new_official_followup_verified': False,
        'same_archive_resplit': 'ineligible_as_new_independent_population',
        'required_owner_deliverable': 'Rights-cleared independent parts or later-month archive, source IDs/date mapping/units, availability-at-cutoff and a prior-access attestation.',
        'existing_holdout_option': 'A separately permitted legacy fixed-temporal evaluation with explicit access caveat; not automatically a pristine confirmation.',
        'training_required': 'not_for_compatible_independent_monthly_events; new features/event units require separate study'},
    'rights': {'repository': 'https://github.com/danieldehaan96/spdf', 'github_license': None,
        'verified_permission_grant': False,
        'interpretation': 'Public availability and citations are verified; no explicit data-license grant was recovered. Local research_use_status is a project assertion, not owner permission.'},
    'remaining_checks': common_gates + ['rights_for_actual_source', 'exhaustive_practical_access_attestation',
        'new_cohort_or_period_beyond_the_existing_archive_if_clean_confirmation_required'],
    'evidence': ['L_raf_contract', 'L_raf_manifest', 'L_preparation_lineage', 'W_raf_repository'],
}
eligibility = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'metadata_eligibility_review_complete_no_population_passed',
    'verdict_vocabulary': {'PASS': 'All declared clean population and access gates evidenced.',
        'conditional_candidate': 'Specific compatible source candidate identified; independence and acquisition gates unresolved.',
        'ineligible': 'Specified route cannot support the specified independence claim.',
        'unknown': 'Required provenance/access/source evidence not established.'},
    'status_scope': 'Readiness for an untouched independent confirmation, not technical ability to rerun a legacy holdout.',
    'datasets': [taxi, inter, insta, raf],
    'counts': {'PASS': 0, 'conditional_candidate': 1, 'ineligible': 0, 'unknown': 3},
    'execution_permit': False, 'new_data_acquired': False, 'new_evaluation': False,
    'heldout_metric_or_prediction_files_opened': False, 'heldout_quantity_or_gap_columns_materialized': False,
    'model_selection_reopened': False, 'retraining_started_or_authorized_by_this_report': False,
    'source_version_caveat': 'Current preprocessing code and present file metadata were inspected. A matching filename alone does not prove that current notebook code produced historical frozen bytes; historical source/transform binding remains a gate.',
}
write('eligibility.json', eligibility)
write('effective_rule_reconciliation.json', {
    'status': 'prior_artifacts_preserved_corrections_recorded_for_effective_contract',
    'prior_design': 'reports/titantpp_independent_final_evaluation_design_20261003_v1/design.json',
    'later_evidence_preparation': 'reports/titantpp_independent_evaluation_preparation_20261001_v1/',
    'corrections': [
        {'topic': 'Intermittent bootstrap', 'old_10_3_design': 'whole_oper_part_no primary',
         'effective_rule': 'whole_site_cd primary; whole_oper_part_no sensitivity',
         'source': '10/1 preparation/amendment.json, supported by50site/5000series identifier mapping',
         'reason': 'The10/3 design omitted an already approved10/1 amendment.'},
        {'topic': 'Taxi cohort filters', 'old_interpretation': 'min_count100/min_coverage0.999 as cohort selection',
         'effective_rule': 'These are train-only max-order construction thresholds. Actual source cohort filter is full-January active_buckets>=72.',
         'source': 'yellow_trip.ipynb code and tpp_split_utils.py::fit_max_order_from_train'},
        {'topic': 'Checkpoint/evaluator readiness', 'old_10_3_design': '36 legacy original paths absent and evaluator readiness not established',
         'effective_rule': '10/1 preparation already reports84 original bindings and CPU validation-only preparation. Current binding/evaluator agents verify effective sources; old-path absence is not current absence.',
         'source': '10/1 preparation README/protocol_effective; current task binding report is authoritative once verified'},
        {'topic': 'Independent population readiness', 'effective_rule': 'Unchanged: no clean evaluation population is certified merely by artifact or evaluator completion.'},
    ], 'model_checkpoint_selection_changed': False, 'prior_design_modified': False,
})
local = json.loads((OUT / 'local_metadata.json').read_text())
local_map = {x['path']: x for x in local['source_files']}
evidence = []
links = {
    'L_taxi_notebook': 'simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb',
    'L_inter_builder': 'paper/scripts/build_intermittent_frozen_subset.py',
    'L_insta_notebook': 'simple_lab_test/notebooks/preprocessing/insta_market_basket.ipynb',
    'L_prior_lineage': 'reports/titantpp_pakdd_extension_preparation_20261001_v1/lineage.json',
    'L_preparation_amendment': 'reports/titantpp_independent_evaluation_preparation_20261001_v1/amendment.json',
    'L_preparation_lineage': 'reports/titantpp_independent_evaluation_preparation_20261001_v1/lineage_review.json',
    'L_raf_contract': 'benchmark_data/contracts/raf_spare_parts_v1.json',
    'L_raf_manifest': 'benchmark_data/manifests/raf_spare_parts_v1.json',
}
for key, path in links.items():
    evidence.append({'id': key, 'type': 'local_metadata_or_code', **local_map[path]})
for key, pointer in [('L_taxi_footer', '/taxi'), ('L_inter_metadata', '/intermittent'), ('L_insta_stat', '/instacart_raw_stat_only')]:
    evidence.append({'id': key, 'type': 'metadata_only_extraction', 'path': str((OUT / 'local_metadata.json').relative_to(ROOT)),
                     'sha256': sha(OUT / 'local_metadata.json'), 'json_pointer': pointer})
web = [
    ('W_taxi_columns', 'https://data.cityofnewyork.us/api/views/2yzn-sicd/columns.json', 'Official schema exposes pickup_datetime and pickup GPS coordinates; no rows queried.'),
    ('W_taxi_catalog', 'https://data.cityofnewyork.us/api/views/2yzn-sicd.json', 'Official2015 dataset attributed toTLC; source description uses generic zone wording, but exact column API still exposes GPS.'),
    ('W_tlc_dictionary', 'https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf', '2025 dictionary uses PULocationID/DOLocationID; no equivalence to old GPS cells asserted.'),
    ('W_tlc_access', 'https://www.nyc.gov/site/tlc/about/trip_record_user_guide.page', 'Use official TLC trip-record landing page/user guide for access; this convenience URL itself was not verified.'),
    ('W_insta_release', 'https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2', 'Publisher2017release identifies relative time and links terms; no independent later release verified.'),
    ('W_insta_dictionary', 'https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b', 'Publisher-linked dictionary documents capped30day relative interval and prior/train/test source meanings.'),
    ('W_insta_terms', 'https://gist.github.com/jeremystan/582eba13d6ee27ed465c43dc78934700', 'Publisher-linked terms restrict noncommercial use and third-party redistribution; current entitlement not certified.'),
    ('W_insta_competition', 'https://www.kaggle.com/competitions/basket-analysis', 'Official competition says disabled at host request; not evidence that all previously acquired research use is revoked.'),
    ('W_insta_derivative', 'https://www.cs.cornell.edu/~arb/data/uchoice-Instacart/', 'Authors describe derivative of2017Instacart, therefore not evidence of a new independent release.'),
    ('W_raf_repository', 'https://api.github.com/repos/danieldehaan96/spdf', 'Original distributor repository metadata is public and license=null; no license grant inferred.'),
]
for key, url, supports in web:
    if key == 'W_tlc_access':
        url = 'https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf'
        supports = 'Official user guide describes publicly available monthly trip records; source access does not establish nonaccess history.'
    evidence.append({'id': key, 'type': 'primary_public_source', 'url': url, 'checked_date': '2026-10-03', 'supports': supports})
write('source_evidence_map.json', {'evidence': evidence, 'api_metadata_snapshot': 'public_metadata.json',
    'bounded_search_limit': 'Officialpublisher/NYCschema/RAFdistributor metadata checked; absence of a verified new source is not proof that none exists.'})
print(json.dumps({'datasets': [(x['name'], x['status']) for x in eligibility['datasets']], 'independent_population_passes': 0}))
