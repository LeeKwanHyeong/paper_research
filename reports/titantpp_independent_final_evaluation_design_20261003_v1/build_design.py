"""Local design and identity inventory only; never loads data rows or checkpoints.

No model imports, inference, training, remote commands, or provider operations.
"""
import copy
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREV = ROOT / 'reports/titantpp_independent_evaluation_protocol_20261001_v1'

def read(p):
    return json.loads((ROOT / p).read_text())

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

now = datetime.now(timezone.utc).isoformat()
previous = json.loads((PREV / 'protocol.json').read_text())
inventory = json.loads((PREV / 'checkpoint_manifest.json').read_text())
data = json.loads((PREV / 'dataset_manifest.json').read_text())
lineage = read('reports/titantpp_pakdd_extension_preparation_20261001_v1/lineage.json')
for row in inventory['rows']:
    p = ROOT / row['checkpoint_path']
    row['current_byte_check_utc'] = now
    row['current_local_path_present'] = p.is_file()
    row['current_file_sha256'] = sha(p) if p.is_file() else None
    row['current_matches_recorded_sha256'] = (row['current_file_sha256'] == row['checkpoint_file_sha256']) if p.is_file() else None
    if p.is_file():
        assert row['current_matches_recorded_sha256'], row['checkpoint_path']
inventory['scope'] = '84 frozen selected identities; local byte hashes only, prior CPU audits reused; no binary deserialization or data rows read'
inventory['refreshed_utc'] = now
inventory['summary'] = {
    'selected_conditions': len(inventory['rows']),
    'local_byte_verified': sum(r['current_local_path_present'] for r in inventory['rows']),
    'missing_at_recorded_local_path': sum(not r['current_local_path_present'] for r in inventory['rows']),
    'representative_local_byte_verified': sum(r['current_local_path_present'] for r in inventory['rows'] if r['model'] == 'titantpp_history_mlp'),
    'new_cpu_deserializations': 0,
    'missing_does_not_establish_loss_or_absence_at_other_locations': True,
}
write('checkpoint_manifest.json', inventory)
write('dataset_freeze.json', data)

registry_path = ROOT / 'reports/titantpp_structural3_final_audit_20261003_v1/condition_registry.csv'
registry = list(csv.DictReader(registry_path.open()))
control_rows = []
audit_paths = set()
for row in registry:
    if row['model'] == 'titantpp_history_mlp':
        continue
    ap = Path(row['audit'])
    audit = json.loads(ap.read_text())
    assert audit['status'] == 'passed' and audit['evaluation_scope'] == 'validation_only'
    cond = next(c for c in audit['conditions'] if (c['job']['dataset'], c['job']['arm'], c['job']['seed']) == (row['dataset'], row['model'], int(row['seed'])))
    assert cond['status'] == 'passed' and cond['strict_cpu_loading']
    assert cond['selected_epoch'] == int(row['selected_epoch'])
    job = Path(row['original_job_path'])
    cp = job / 'runs' / row['model'] / 'count_only_log_regression' / ('seed_' + row['seed']) / 'best_val_qty_rmse_model.pt'
    digest = sha(cp)
    assert digest == cond['checkpoint_files']['best_val_qty_rmse_model.pt']
    control_rows.append({
        'dataset': row['dataset'], 'model': row['model'], 'seed': int(row['seed']),
        'host': row['host'], 'endpoint': 'selected',
        'selected_epoch': cond['selected_epoch'], 'completed_epochs': cond['completed_epochs'],
        'selection': 'strict_first_min_validation_raw_quantity_rmse',
        'checkpoint_path': str(cp.relative_to(ROOT)), 'checkpoint_file_sha256': digest,
        'state_tensor_sha256': cond['model_tensor_sha256']['selected'],
        'contract_canonical_sha256': audit['contract_sha256'],
        'source_closure_sha256': audit['source_closure_sha256'],
        'frozen_source_directory': str((job.parent.parent / 'source').relative_to(ROOT)),
        'prior_cpu_audit_reused': str(ap.relative_to(ROOT)), 'audit_sha256': sha(ap),
        'native_runtime': audit['native_runtime'],
        'selected_validation_replay_path': str((cp.parent / 'endpoint_replays.json').relative_to(ROOT)),
        'current_byte_check_utc': now, 'new_cpu_deserialization': False,
    })
    audit_paths.add(str(ap.relative_to(ROOT)))
assert len(control_rows) == 24
write('structural_checkpoint_manifest.json', {'created_utc': now, 'rows': control_rows, 'summary': {'selected_conditions': 24, 'local_byte_verified': 24, 'prior_cpu_audits_reused': 24, 'new_cpu_deserializations': 0}, 'registry_path': str(registry_path.relative_to(ROOT)), 'registry_sha256': sha(registry_path)})

routes = {
    'yellow_trip_hourly': {
        'legacy_label': 'previously_exposed_legacy_test',
        'clean_route': 'Later non-overlapping raw calendar period after the latest timestamp in every previously used partition; same frozen spatial cells and hourly aggregation.',
        'minimum_metadata': ['raw release identifier and immutable SHA', 'full previously used calendar end cutoff', 'cell mapping and unit definitions', 'pre-cutoff entity eligibility', 'event-key overlap report', 'data rights and prior-access attestation'],
        'currently_unresolved': ['new raw period not acquired or bound', 'retrospective min_count=100 and min_coverage=0.999 need cutoff-only eligibility audit', 'common temporal shocks across spatial cells'],
        'scope': 'future-period generalization conditional on the frozen region/cell cohort, not new-city generalization',
    },
    'intermittent_frozen_5000': {
        'legacy_label': 'legacy_holdout_exposure_unresolved',
        'clean_route': 'A new timestamped extraction after all prior per-site cutoffs, or independently sourced sites/parts with verified source mapping; freeze a cohort using only pre-cutoff metadata.',
        'minimum_metadata': ['upstream owner and redistribution/evaluation permission', 'site_cd::part_no identity map', 'extraction as-of timestamp and per-site cutoffs', 'current 5000-series selection and strata code', 'event definition and quantity/gap units', 'historical extract event-key deduplication'],
        'currently_unresolved': ['raw current and legacy source unavailable in recorded lineage', 'legacy demand>8 and burst merging versus current positive rows not reconciled', '5000-series retrospective stratification may depend on future information', 'cross-series site dependency needs metadata audit'],
        'scope': 'same-site temporal transfer or new-site transfer must be named separately before labels are released',
    },
    'insta_market_basket': {
        'legacy_label': 'previously_exposed_legacy_test',
        'clean_route': 'Separately sourced, previously unused basket histories with the same event/quantity definition and documented historical timestamps; no verified later release is available locally.',
        'minimum_metadata': ['dataset license and release SHA', 'customer identity provenance and non-overlap', 'order chronology and observed-gap coding', 'basket-size definition', 'as-of customer eligibility', 'whether a real timestamp or a reconstructed relative clock is supplied'],
        'currently_unresolved': ['no clean dataset bound', 'same archive relabeling or Kaggle eval_set reassignment does not create independence', 'relative dates cannot be represented as a verified later calendar cohort', 'if uncapped new gaps are recoded to 30, report the same recorded observation task'],
        'scope': 'external basket-history transfer, not proof of within-archive unseen final test',
    },
    'raf_spare_parts': {
        'legacy_label': 'fixed_temporal_holdout_with_incomplete_access_history',
        'clean_route': 'An independently sourced parts/monthly-demand archive or a demonstrably unused later release; the known 84-month archive provides no later period.',
        'minimum_metadata': ['archive identifier and SHA', 'part identity and source-system provenance', 'coverage months and pre-cutoff part eligibility', 'quantity and monthly-gap definitions', 'duplicate/derived archive audit', 'prior-access attestation'],
        'currently_unresolved': ['2001-12..2002-12 fixed holdout is not proven untouched by bounded access search', 'independent archive not bound', 'zero-demand months are not additional next-positive-event targets', 'shared supplier/system shocks may cross part clusters'],
        'scope': 'external parts transfer if new data is obtained; otherwise transparently named fixed legacy-holdout reevaluation',
    },
}
population = {'created_utc': now, 'clean_population_verified': False, 'rows': []}
for ds in previous['datasets']:
    old = lineage['datasets'][ds]
    population['rows'].append({
        'dataset': ds,
        'independence_status': 'not_established',
        'lineage_classification': old['classification'],
        'historical_access_record': old.get('historical_test_access', 'mapping unresolved; do not infer absence from disjoint literal keys'),
        'same_event_key_overlap': old.get('overlap'),
        'clean_population_path': None, 'clean_population_sha256': None,
        'eligible_target_ids_sha256': None, 'eligible_entity_ids_sha256': None,
        'approved_cutoff': None, 'approved_evaluation_period': None,
        **routes[ds],
    })
population['access_caveat'] = 'Prior incidental held-out target-distribution summary exposure is recorded; no new held-out quantities, gaps, predictions, performance, or mixed result files were read in this work.'
write('population_eligibility.json', population)

design = copy.deepcopy(previous)
design.update({
    'schema': 'titantpp_independent_final_evaluation_design_20261003_v1',
    'created_utc': now,
    'status': 'design_frozen_population_eligibility_and_execution_incomplete',
    'execution_authorized': False,
    'inference_started': False,
    'registration': 'local design freeze after validation development and A100 exploration; not a public or pre-development preregistration',
    'predecessor_design': str(PREV.relative_to(ROOT) / 'protocol.json'),
    'checkpoint_manifest': 'checkpoint_manifest.json',
    'dataset_manifest': 'dataset_freeze.json',
    'split_access_ledger': 'population_eligibility.json',
    'representative_architecture': {
        'model': 'titantpp_history_mlp', 'hidden_dim': 64,
        'branches': 8, 'bottleneck_width': 4, 'correction_parameters': 6144,
        'source_indices': 'same immediate observed predecessor in all eight branches; thresholds are not lag retrieval offsets',
        'availability_thresholds': [1, 2, 4, 8, 16, 32, 64, 128],
        'availability_test': 'observed row and contiguous observed count > threshold; withheld valid row resets eligibility; padding does not',
        'residual_divisor': 8, 'output_projection_initialization': 'zero',
        'insertion': 'between two frozen encoder blocks',
        'checkpoint_source_revision': '1de2c31e8e3febda7ff20d0527402ea9959b4098',
        'new_current_worktree_source_is_not_an_evaluation_substitute': True,
    },
    'simple_quantity_baselines': [
        {'model': 'last_observed_quantity', 'definition': 'last raw quantity in exactly the same admitted observed history window', 'seed': None, 'time_nll': None},
        {'model': 'mean_quantity_in_same_observed_window', 'definition': 'arithmetic mean of raw quantities over exactly the same admitted observed history window, excluding target and padding', 'seed': None, 'time_nll': None},
    ],
    'baseline_protocol': {'trained': False, 'fit_to_test': False, 'same_target_ids': True, 'repeated_as_three_seeds': False, 'empty_history': 'ineligible under common next-event target contract; no method-specific dropping or tuned fallback', 'uncertainty_pairing': 'one deterministic baseline prediction vector reused in the three fixed model-seed differences without adding independent replicates'},
    'structural_secondary_panel': {
        'models': ['titantpp_current_only_param_matched', 'titantpp_all_available_history_mlp'],
        'conditions': 24, 'seeds': [42, 52, 62],
        'role': 'prespecified supporting contrasts, not alternative representatives',
        'binary_and_source_binding': 'all 24 exact selected identities bound to original file/tensor/source SHA with passed CPU-audit reuse and local byte recheck; preserve heterogeneous GPU provenance',
        'manifest': 'structural_checkpoint_manifest.json', 'identity_binding_status': 'complete',
    },
    'comparison_scope': {
        'claim': 'numerical-history common-head TPP encoder comparison plus the two simple quantity references and two structural controls',
        'deep_renewal': 'excluded from manuscript comparison at user request after its validation results existed; preserve all runs and research records; this scope is not evidence of dominance over native distributional demand predictors',
        'selection_transparency': 'Do not invent a pre-result exclusion rationale or describe native methods as invalid because they outperform TitanTPP. Method-family scope limits the claim and does not erase retrospective comparator-selection concerns.',
        'A100': 'cross-product and recent-four-mean remain separately reported single-seed exploration; neither replaces the representative',
    },
    'input_access': {
        'all_learned_methods': ['log1p of observed past gaps with frozen nonnegative handling', 'log1p of observed past raw quantities with frozen nonnegative handling', 'valid/observed masks and chronology/positions required by the frozen adapter'],
        'simple_baselines': ['same past raw quantities', 'same history masks/order/window'],
        'bookkeeping_only': ['entity ID', 'site/part or user mapping', 'target ID', 'split', 'timestamp for chronology/Taxi resampling'],
        'forbidden_model_inputs': ['current target quantity', 'current target elapsed gap when forecasting it', 'future rows', 'product category/item IDs', 'new calendar covariates', 'test-fitted normalization or clipping'],
        'asof': 'Only information observed by the forecast origin; no target-dependent history length/window, no future-based admission, no online parameter updates.',
    },
    'evaluation_routes': {
        'existing_checkpoints_no_retraining': 'Allowed in a separately authorized evaluation if event definitions, unit/observation coding, numerical input schema and train-fitted transformations match; exact frozen checkpoint/source and causal target mask remain usable.',
        'external_domain_interpretation': 'Any frozen-weight application to newly sourced sites, users or archives is external transfer. Poor calibration is a result, not permission to refit on evaluation outcomes.',
        'new_training_requires_new_design_and_approval': ['required input dimension or semantic/unit change not handled by frozen observation coding', 'benchmark is redesigned with a different train/cohort construction to answer a different population claim', 'frozen selected checkpoint cannot be recovered', 'new model or tuned preprocessing/normalization/head is introduced'],
        'fresh_data_after_redesign': 'After label/result exposure, changed architecture or tuning uses those data only as development evidence; independent confirmation needs another genuinely unused cohort.',
        'not_authorized_now': ['new training', 'new validation/test forward pass', 'held-out performance or prediction reading', 'remote execution or data extraction', 'paid resources'],
    },
    'one_shot_release': {
        'before_release': ['verify identity/lineage and pre-cutoff cohort rules without labels/performance', 'bind manifests, code/runtime, hardware/budget/deadline', 'pass synthetic/validation-only evaluator checks under separate permission', 'obtain exact population and performance-access execution approval', 'seal manifests and analysis script hashes'],
        'release_event': 'One approved locked evaluation batch writes all models/seeds, failure records, target IDs and hashes; no sequential model-result peeking to choose which runs to retain.',
        'after_release': 'Run the frozen analyses and publish all prespecified results and failures, including negative effects. Record every access timestamp and any broken gate.',
        'technical_failure': 'Preserve first attempt; rerun the identical scientific specification only under the applicable execution authorization with diagnosis based on infrastructure/identity checks. Any scientific change makes same-cohort findings exploratory.',
        'stop_before_release': ['unresolved source/target overlap', 'post-cutoff cohort selection', 'hash/config mismatch', 'missing selected checkpoint', 'unexpected input/target alignment or NaN/Inf', 'unapproved data rights or environment/budget', 'new result exposure before freeze'],
    },
    'required_execution_approval_fields': ['route: legacy reevaluation or genuinely new population', 'each dataset raw/split/target-ID/entity-ID SHA and source/license', 'calendar or source cutoffs, forecast origin, eligibility rule and allowed history', 'complete selected-checkpoint/source/runtime manifests for 84+24 learned conditions and two deterministic baselines', 'named host/GPU or CPU, maximum cost, deadline, no training flag', 'exact output directory, sealed result access owner/time', 'explicit held-out prediction/performance generation and reading permission'],
    'remaining_requirements': ['bind a genuinely unused eligible population or choose transparent legacy reevaluation', 'resolve pre-cutoff cohort construction and source rights', 'locate and byte/CPU bind missing legacy comparator checkpoint originals', 'implement/verify evaluator on synthetic and authorized validation only', 'freeze runtime, output paths, cost/deadline and explicit evaluation/access authorization'],
})
design['exploratory_variants_not_promoted'] += ['titantpp_history_mlp_cross_product', 'titantpp_history_mlp_recent4_mean']
design['resampling']['primary_unit'] = 'whole entity/series clusters, not overlapping windows or individual events; Taxi shared calendar blocks preserve cross-cell temporal shocks'
design['resampling']['paired_entity_cluster_sensitivity_for_taxi'] = '10,000 complete-cell cluster draws, reported as a sensitivity conditional on cells; cannot substitute for shared-time primary uncertainty'
design['resampling']['upper_level_dependency_gate'] = 'If site or system clustering violates the declared unit, freeze a metadata-justified higher-level plan before outcome release or limit inference; never select a favorable unit from CIs.'
design['resampling']['minimum_cluster_count_to_compute'] = 2
design['readiness'] = {'methodological_design_complete': True, 'clean_population_eligible': False, 'all_checkpoint_binaries_bound': False, 'evaluation_implementation_verified': False, 'execution_authorized': False}
write('design.json', design)

source_roles = {
    'TEST_SESSION_PROTOCOL.md': 'approval and held-out access boundaries sections 0/9/12',
    'reports/titantpp_pakdd_extension_preparation_20261001_v1/lineage.json': 'newer event-key and exposure lineage; no target values or performance accessed',
    'reports/titantpp_independent_evaluation_protocol_20261001_v1/protocol.json': 'inherited 84-model/seed panel, anchors, paired bootstrap policy',
    'reports/titantpp_independent_evaluation_protocol_20261001_v1/checkpoint_manifest.json': 'frozen 84 selected epoch/tensor/file/source identities',
    'reports/titantpp_independent_evaluation_protocol_20261001_v1/dataset_manifest.json': 'frozen train-derived loader/head/observation settings',
    'reports/titantpp_independent_evaluation_protocol_20261001_v1/split_access_ledger.json': 'bounded historical access audit and nonaccess caveats',
    'reports/titantpp_independent_evaluation_protocol_20261001_v1/verification.json': 'prior manifest and design verification only; not a new CPU audit',
    'reports/titantpp_pakdd_extension_preparation_20261001_v1/nonlearned_comparison.json': 'validation-only deterministic baseline identities, already used in manuscript',
    'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/frozen_source/paper/scripts/run_pakdd_extension.py': 'frozen simple-baseline formulas only; script not executed',
    'search_artifacts/titantpp_core_ablation_20260928_v1/final_audit_20260930_v1/terminal_5090/source/models/TPPs/CountAwareTitanCoreAblation.py': 'frozen representative architecture source; no imports',
    'search_artifacts/titantpp_core_ablation_20260928_v1/final_audit_20260930_v1/terminal_5090/source/models/TPPs/CountAwareTPP.py': 'frozen shared numerical input feature source; no imports',
    'reports/titantpp_structural3_final_audit_20261003_v1/condition_registry.csv': 'final 36-row MLP+structural registry; metadata allowlist used for control identity binding',
    'reports/titantpp_structural3_final_audit_20261003_v1/verification.json': 'additional three conditions passed original CPU audit; original21 reused',
}
source_roles.update({p: 'prior passed validation-only CPU audit: selected checkpoint/source identity fields reused, no new replay or binary load' for p in audit_paths})
write('source_evidence_map.json', {
    'created_utc': now,
    'local_sources': [{'path': p, 'sha256': sha(ROOT / p), 'evidence_role': role} for p, role in source_roles.items()],
    'manuscript_context': {'path': 'paper/titantpp_pakdd_2027_draft/main.tex', 'sections': ['Bottleneck History Correction', 'Discussion and Limitations'], 'role': 'development use, retrospective cohorts, common-head scope; concurrently edited by parent, not hashed as a frozen source'},
    'external_method_sources': [
        {'title': 'Bootstrapping clustered data', 'authors': 'Field and Welsh', 'year': 2007, 'url': 'https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2007.00593.x', 'access': 'publisher abstract checked 2026-10-03 KST', 'supports': 'cluster bootstrap validity depends on dependence/model assumptions; not a generic guarantee'},
        {'title': 'The Jackknife and the Bootstrap for General Stationary Observations', 'authors': 'Künsch', 'year': 1989, 'url': 'https://doi.org/10.1214/aos/1176347265', 'access': 'primary DOI resolved; procedure inherited from prior design, full article not newly reviewed', 'supports': 'background citation for dependence-preserving block resampling, not exact chosen block length'},
        {'title': 'Common pitfalls and recommended practices', 'publisher': 'scikit-learn', 'url': 'https://scikit-learn.org/stable/common_pitfalls.html', 'access': 'official documentation checked 2026-10-03 KST', 'supports': 'keep test information out of fitting/preprocessing/model selection'},
    ],
    'new_data_rows_or_heldout_predictions_or_metrics_accessed': False,
    'source_modules_imported': False,
    'new_gpu_remote_or_forward_calls': False,
})
print(json.dumps({'output': str(OUT), 'status': design['status'], 'inventory': inventory['summary']}, ensure_ascii=False))
