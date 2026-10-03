"""Validate local design gates and identity bindings, without inference or data access."""
import ast
import hashlib
import itertools
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import stdev

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
def load(name):
    return json.loads((OUT / name).read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

d = load('design.json')
base = load('checkpoint_manifest.json')
controls = load('structural_checkpoint_manifest.json')
pop = load('population_eligibility.json')
src = load('source_evidence_map.json')
checks = {}
def check(name, condition):
    checks[name] = bool(condition)

key = lambda r: (r['dataset'], r['model'], r['seed'])
expected = set(itertools.product(d['datasets'], d['models'], d['seeds']))
check('84_unique_fixed_selected_conditions', len(base['rows']) == 84 and {key(r) for r in base['rows']} == expected)
expected_control = set(itertools.product(d['datasets'], d['structural_secondary_panel']['models'], d['seeds']))
check('24_unique_control_selected_conditions', len(controls['rows']) == 24 and {key(r) for r in controls['rows']} == expected_control)
check('no_overlap_between_main_and_structural_conditions', not ({key(r) for r in base['rows']} & {key(r) for r in controls['rows']}))
check('12_representative_checkpoints_byte_bound', len([r for r in base['rows'] if r['model'] == d['representative'] and r['current_local_path_present']]) == 12)
check('48_main_and24_control_local_bytes_match', sum(r['current_local_path_present'] for r in base['rows']) == 48 and all(sha(ROOT / r['checkpoint_path']) == r['checkpoint_file_sha256'] for r in base['rows'] if r['current_local_path_present']) and all(sha(ROOT / r['checkpoint_path']) == r['checkpoint_file_sha256'] for r in controls['rows']))
check('36_missing_are_explicit_not_invented', sum(not r['current_local_path_present'] for r in base['rows']) == 36 and all(r['current_file_sha256'] is None for r in base['rows'] if not r['current_local_path_present']))
check('selected_epoch_tensor_and_source_identity_bound', all(r['selected_epoch'] >= 1 and len(r['state_tensor_sha256']) == 64 for r in base['rows'] + controls['rows']) and all(len(r['source_closure_sha256']) == 64 for r in controls['rows']))
check('control_original_source_and_endpoint_paths_resolve', all((ROOT / r['frozen_source_directory']).is_dir() and (ROOT / r['selected_validation_replay_path']).is_file() for r in controls['rows']))
check('control_prior_cpu_audits_immutable', all(sha(ROOT / r['prior_cpu_audit_reused']) == r['audit_sha256'] for r in controls['rows']))
check('all_immutable_source_evidence_hashes_match', all(sha(ROOT / s['path']) == s['sha256'] for s in src['local_sources']))
check('frozen_original_architecture', d['representative_architecture']['residual_divisor'] == 8 and d['representative_architecture']['output_projection_initialization'] == 'zero' and d['representative_architecture']['correction_parameters'] == 6144 and d['representative_architecture']['availability_thresholds'] == [1,2,4,8,16,32,64,128])
check('all_four_clean_populations_explicitly_unestablished', len(pop['rows']) == 4 and not pop['clean_population_verified'] and all(r['independence_status'] == 'not_established' and r['clean_population_sha256'] is None for r in pop['rows']))
check('legacy_exposure_not_erased', {r['dataset']:r['legacy_label'] for r in pop['rows']} == {'yellow_trip_hourly':'previously_exposed_legacy_test','intermittent_frozen_5000':'legacy_holdout_exposure_unresolved','insta_market_basket':'previously_exposed_legacy_test','raf_spare_parts':'fixed_temporal_holdout_with_incomplete_access_history'})
check('no_execution_permission_inferred', not d['execution_authorized'] and not d['inference_started'] and d['readiness']['methodological_design_complete'] and not any(v for k,v in d['readiness'].items() if k != 'methodological_design_complete'))
check('two_deterministic_baselines_not_fake_three_seeds', len(d['simple_quantity_baselines']) == 2 and all(r['seed'] is None and r['time_nll'] is None for r in d['simple_quantity_baselines']) and not d['baseline_protocol']['repeated_as_three_seeds'])
check('fixed_validation_anchors_all_datasets', d['fixed_validation_selected_anchors'] == {'yellow_trip_hourly':'rmtpp','intermittent_frozen_5000':'rmtpp','insta_market_basket':'s2p2_matched_head','raf_spare_parts':'s2p2_matched_head'})
check('paired_10000_draws_no_seed_resampling', d['resampling']['draws'] == 10000 and d['resampling']['paired_across_all_models_and_seeds'] and not d['resampling']['seed_resampling'])
check('95percent_and_four_comparison_multiplicity_prespecified', d['resampling']['pointwise_interval_quantiles'] == [.025,.975] and d['resampling']['fixed_four_RMSE_anchor_interval_quantiles'] == [.00625,.99375] and not d['resampling']['pvalues'])
check('taxi_shared_time_dependencies_not_independent_windows', d['resampling']['taxi_block_hours'] == 168 and d['resampling']['units']['yellow_trip_hourly'] == 'shared_calendar_hour_circular_blocks_across_all_spatial_cells' and d['resampling']['unavailable_timestamp'] == 'no_target_order_fallback')
check('new_training_and_heldout_forward_not_in_design_scope', {'new training','new validation/test forward pass','held-out performance or prediction reading','remote execution or data extraction','paid resources'} <= set(d['evaluation_routes']['not_authorized_now']))
check('explicit_execution_approval_and_stop_gates_present', len(d['required_execution_approval_fields']) == 7 and len(d['one_shot_release']['stop_before_release']) >= 7)
check('source_access_boundary_explicit', src['new_data_rows_or_heldout_predictions_or_metrics_accessed'] is False and src['source_modules_imported'] is False and src['new_gpu_remote_or_forward_calls'] is False)

# Unequal clusters verify the estimand arithmetic without using any real data.
clusters = [{'n':1,'se':9.0,'ae':3.0}, {'n':3,'se':3.0,'ae':3.0}]
multiplicity = [2,1]
n = sum(w*c['n'] for w,c in zip(multiplicity, clusters))
rmse = math.sqrt(sum(w*c['se'] for w,c in zip(multiplicity, clusters))/n)
mae = sum(w*c['ae'] for w,c in zip(multiplicity, clusters))/n
expanded = [3.0,3.0,1.0,1.0,1.0]
check('synthetic_cluster_multiplicity_keeps_event_weighting', math.isclose(rmse,math.sqrt(sum(e*e for e in expanded)/len(expanded))) and math.isclose(mae,sum(expanded)/len(expanded)))
seed_rmse = [1.0,2.0,4.0]
check('synthetic_seed_mean_rmse_not_sqrt_mean_mse', not math.isclose(sum(seed_rmse)/3, math.sqrt(sum(x*x for x in seed_rmse)/3)))
check('synthetic_seed_SD_is_sample_SD_not_SE', math.isclose(stdev([1,2,3]),1.0) and not math.isclose(stdev([1,2,3]),1/math.sqrt(3)))

bad_links = []
for rel in re.findall(r'\]\(([^)]+)\)', (OUT/'README.md').read_text()):
    if rel.startswith('https://'):
        continue
    if rel == 'verification.json':
        continue  # This verification receipt is created at the end of this script.
    if not (OUT/rel).resolve().exists():
        bad_links.append(rel)
check('local_report_links_resolve', not bad_links)
for filename in ['build_design.py','verify.py']:
    tree = ast.parse((OUT/filename).read_text())
    imported = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import)}
    imported |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module}
    check(filename+'_no_gpu_data_loader_network_process_imports', not imported & {'torch','pandas','pyarrow','requests','subprocess','socket','paramiko','runpod'})

result = {
    'status': 'passed' if all(checks.values()) else 'failed',
    'verified_utc': datetime.now(timezone.utc).isoformat(),
    'checks': checks, 'passed_checks': sum(checks.values()), 'total_checks': len(checks),
    'broken_links': bad_links,
    'execution_readiness': False, 'independent_population_eligible': False,
    'heldout_rows_predictions_or_performance_read': False,
    'new_binary_cpu_loads': 0, 'new_model_forward_training_remote_calls': 0,
    'actual_evaluator_implemented_or_integration_tested': False,
    'real_data_confidence_intervals_computed': False,
    'scope': 'design consistency, previously audited selected identity byte binding, stop gates and synthetic metric arithmetic only',
    'artifacts_sha256': {p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'},
}
(OUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','passed_checks','total_checks','execution_readiness']},ensure_ascii=False))
if not all(checks.values()):
    raise SystemExit([k for k,v in checks.items() if not v])
