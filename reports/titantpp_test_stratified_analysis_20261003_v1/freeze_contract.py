"""Freeze additive retrospective Test analysis from existing metadata only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1'
REGISTRY = ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tick(value, kind):
    value = str(value)
    if kind == 'relative_day':
        return int(value)
    dt = datetime.strptime(value, '%Y%m%d%H%M%S' if kind == 'calendar_hour' else '%Y%m%d')
    if kind == 'calendar_month':
        return dt.year * 12 + dt.month - 1
    days = (dt - datetime(1970, 1, 1)).days
    return days * 24 + dt.hour if kind == 'calendar_hour' else days // 7


def main():
    manifest_path = SOURCE / 'dataset_manifest.json'
    registry = read(REGISTRY)
    manifest = read(manifest_path)
    old_contract = read(SOURCE / 'execution_contract.json')
    routing_path = ROOT / 'search_artifacts/titantpp_routing_placement_a100_20261003_v1/execution_contract.json'
    routing = read(routing_path)
    activations = routing['architecture']['titantpp_history_recent4_attention']['multilag_detail_availability_lags']
    assert activations == [1, 2, 4, 8, 16, 32, 64, 128]
    sources = [manifest_path, REGISTRY, SOURCE / 'execution_contract.json', SOURCE / 'analysis.json',
               SOURCE / 'analyze.py', SOURCE / 'runs/test/attempt1/run_manifest.json',
               SOURCE / 'retrieval/original/runs/test/attempt1/terminal_manifest.json',
               SOURCE / 'final_artifact_audit.json', routing_path]
    kinds = ['calendar_hour', 'calendar_week', 'relative_day', 'calendar_month']
    descriptions = [
        'Recorded calendar-hour coordinate; original per-entity Test membership retained; not a newly global-cutoff forecast.',
        'Recorded shifted-date week coordinate. User-confirmed uniform date shift; original dates and seasonality not independently recovered.',
        'User-relative elapsed recorded days from purchase history start. Not a shared calendar, not within-user Test progress thirds.',
        'Recorded common calendar month coordinate. Original fixed Test membership retained.',
    ]
    rows = []
    for dataset, kind, semantics in zip(manifest['datasets'], kinds, descriptions):
        name = dataset['dataset']
        baseline = next(r for r in registry['rows'] if r['dataset'] == name and r['model'] == 'titantpp_history_mlp' and r['seed'] == 42)
        replay_path = ROOT / baseline['validation_replay_path']
        replay = read(replay_path)
        selected = replay['selected']
        assert selected.get('evaluation_scope') == 'validation_only'
        assert selected.get('held_out_test_evaluated') is False
        sources.append(replay_path)
        population = dataset['populations']['test']
        t0, t1 = tick(population['recorded_date_min'], kind), tick(population['recorded_date_max'], kind)
        upper = [t0 + ((i * (t1 - t0 + 1) + 2) // 3) - 1 for i in [1, 2]]
        rows.append({
            'dataset': name, 'target_count': population['target_count'],
            'data_file_sha256': dataset['sha256'], 'population': population,
            'quantity_boundaries': selected['quantity_boundaries'],
            'history_boundaries': selected.get('additional_history_boundaries', selected['history_boundaries']),
            'history_detail_boundaries': [n - 1 for n in activations[1:]],
            'boundary_source': str(replay_path),
            'time_axis': {'kind': kind, 'min_tick': t0, 'max_tick': t1,
                          'upper_ticks_first_two_bins': upper,
                          'recorded_date_min': population['recorded_date_min'],
                          'recorded_date_max': population['recorded_date_max'],
                          'semantics': semantics},
        })
    models = ['titantpp_history_mlp', 'rmtpp', 'thp', 'nhp', 'sahp', 's2p2_matched_head',
              'attnhp_matched_head', 'titantpp_current_only_param_matched', 'titantpp_all_available_history_mlp']
    assert set(models) == set(old_contract['approval']['models'])
    contract = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'approval': {'user_text': '진행하자', 'scope': 'Additive full, time-period, quantity and observed-history Test diagnostics on existing predictions.',
                     'test_access_authorized': True, 'retraining': False, 'new_inference': False,
                     'new_rental': False, 'remote_access': False, 'manuscript_edit': False},
        'interpretation': 'post_hoc_descriptive_stratified_legacy_test_analysis_not_new_independent_test',
        'source_bundle': str(SOURCE),
        'source_run_manifest': str(SOURCE / 'runs/test/attempt1/run_manifest.json'),
        'source_analysis': str(SOURCE / 'analysis.json'),
        'source_hashes': {str(p): sha(p) for p in sources},
        'models': models, 'seeds': [42, 52, 62],
        'simple_models': ['last_observed_quantity', 'mean_quantity_in_same_observed_window'],
        'datasets': rows,
        'time_rule': 'min(2, 3 * (tick - min_tick) // (max_tick - min_tick + 1)); equal native-time-grid thirds, not equal target counts; ties never split.',
        'boundary_rule': 'quantity and history bins use searchsorted(side=left): equality in lower bin; first <=b0, then (previous,b], final >last.',
        'history_definition': 'Actual input observed-event count, excluding target, truncated by original window/max_seq_len; not total lifetime history or elapsed time.',
        'history_detail_rule': 'Supplementary fixed [1,3,7,15,31,63,127] cutoffs derived from pre-existing branch activation thresholds [1,2,4,8,16,32,64,128]. All datasets, all bins retained.',
        'aggregation': {'within_seed': 'event_weighted', 'across_seeds': 'mean_of_seed_metrics',
                        'seed_sd': 'sample_ddof1_not_confidence_interval', 'prediction_ensemble': False,
                        'raw_cross_dataset_pooling': False, 'deterministic_seed_sd': None,
                        'deterministic_time_nll': None, 'empty_bin_metrics': None,
                        'paired_difference': 'TitanTPP_MLP_minus_comparator; lower_is_better',
                        'sse_share': 'mean_seed_bin_SSE / mean_seed_total_SSE, within one dataset/model only'},
        'inference': {'new_pvalues': False, 'new_confidence_intervals': False,
                      'reason': 'Exploratory multiple strata; 3seed SD is training variation, not evaluation-sampling uncertainty. Existing full-Test intervals remain in source report.'},
        'invariants': ['All108 learned conditions and8 deterministic dataset-reference vectors retained.',
                       'Exact same Test identities/truth/history for all models and seeds; no favorable intersection.',
                       'No target exclusion, split membership, source dataset, checkpoint or selection-epoch change.',
                       'Every declared partition count and error sums restore full Test within numerical tolerance.',
                       'Full seed metrics/means/sampleSD reconcile to original analysis rtol=atol=1e-10.',
                       'Group rules frozen before new stratified metric computation; original overall Test results already known.'],
        'limitations': ['Temporal slices change cohort composition; arithmetic differences do not identify causal distribution shift.',
                        'Instacart relative time is not calendar time.',
                        'Intermittent dates have user-confirmed uniform anonymization offset.',
                        'RAF coarse history [64,128] may have empty bins; do not omit these.',
                        'Prior comparator scope excludes Deep Renewal post-results at user request; no claim against all native demand models.',
                        'A100 exploratory candidates and separate B exploration are not added to this frozen108-condition panel.'],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'analysis_contract.json'
    with path.open('x') as f:
        json.dump(contract, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'contract': str(path), 'sha256': sha(path), 'datasets': rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
