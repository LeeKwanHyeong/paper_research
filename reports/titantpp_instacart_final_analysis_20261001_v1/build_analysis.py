"""Reconcile saved validation endpoints and summarize three seeds; no inference."""
import csv
import hashlib
import json
import math
import statistics as st
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

R = Path(__file__).resolve().parents[2]
O = Path(__file__).resolve().parent
A = R / 'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5090_20261001_v1'
METRICS = ('qty_mae', 'qty_rmse', 'time_nll')
PLAIN = 'titantpp_history_mlp'
MODELS = [PLAIN, 'rmtpp', 'thp', 'nhp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head']
LABELS = dict(zip(MODELS, ['TitanTPP MLP', 'RMTPP', 'THP', 'NHP', 'SAHP', 'S2P2', 'AttNHP']))
sources = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def track(p):
    p = Path(p)
    if not p.is_absolute():
        p = R / p
    sources[str(p.relative_to(R))] = sha(p)
    return p


def load(p):
    return json.loads(track(p).read_text())


def dump(name, value):
    (O / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-8)


def stats(values):
    assert len(values) == 3 and all(math.isfinite(v) for v in values)
    return {'mean': st.mean(values), 'sample_sd': st.stdev(values)}


audit = load(A / 'terminal_audit.json')
assert audit['status'] == 'passed' and len(audit['conditions']) == 6
assert audit['verified_terminal_checkpoints'] == 12 and audit['verified_source_files'] == 106
assert audit['new_model_forward_calls'] == audit['new_training_updates'] == audit['new_replay_calls'] == 0
receipt = load(A / 'retrieval_receipt.json')
assert sha(track(A / 'original.tar')) == receipt['archive_sha256'] == audit['archive_sha256']
collection = load(A / 'original/collection_manifest.json')
for rel, info in collection['files'].items():
    assert sha(A / 'original' / rel) == info['sha256'], rel
assert not collection['owned_processes']
core_ver = load('reports/titantpp_core_ablation_execution_20260928_v1/verification.json')
assert core_ver['status'] == 'passed' and core_ver['final_core_binary_source_audit_complete']
core_path = track('reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv')
assert sha(core_path) == '3501b98f0f3c5d250c6ddb7f8309b3455ec1bb908f0844928ec504d71c826765'
core = list(csv.DictReader(core_path.open()))
previous = load('reports/titantpp_completed_external_comparison_20261001_v1/comparison.json')
latest = load('search_artifacts/titantpp_additional_tpp_20260930_v1/hourly_comparison/20261001T104225874418Z/comparison.json')
assert latest['all_queues_terminal'] and not latest['errors']

rows, endpoints, strata = [], {}, []
for model in MODELS:
    for seed in (42, 52, 62):
        if model in MODELS[:5]:
            cr = [r for r in core if r['dataset'] == 'insta_market_basket' and r['arm'] == model and int(r['seed']) == seed]
            assert sorted(r['endpoint'] for r in cr) == ['last', 'selected']
            selected = next(r for r in cr if r['endpoint'] == 'selected')
            path = R / selected['source'] / 'endpoint_replays.json'
            evidence = 'existing_core_and_external_audit_reused'
        else:
            job = f'insta_market_basket__{seed}__{model}'
            path = A / 'original/run' / job / 'runs' / model / 'count_only_log_regression' / f'seed_{seed}' / 'endpoint_replays.json'
            evidence = 'new_5090_cpu_source_checkpoint_audit_passed'
        replay = load(path)
        for role in ('selected', 'last'):
            ep = replay[role]
            assert ep['evaluation_scope'] == 'validation_only' and not ep['held_out_test_evaluated']
            assert ep['count'] == 503733 and all(math.isfinite(ep[m]) for m in METRICS)
            assert close(ep['qty_sse'] / ep['count'], ep['qty_rmse'] ** 2)
            epoch = replay['best_epoch'] if role == 'selected' else replay['completed_epochs']
            if model in MODELS[:5]:
                ref = next(r for r in cr if r['endpoint'] == role)
                assert int(ref['epoch']) == epoch and ref['state_sha256'] == ep['state_sha256']
                assert all(close(float(ref[m]), ep[m]) for m in METRICS)
            else:
                ar = next(r for r in audit['conditions'] if r['job']['arm'] == model and r['job']['seed'] == seed)
                assert ar['model_tensor_sha256'][role] == ep['state_sha256']
                assert all(close(ar[role + '_validation'][m], ep[m]) for m in METRICS)
                if role == 'selected':
                    obs = next(r for r in latest['rows'] if r['arm'] == model and r['seed'] == seed and r['host'] == '5090')
                    assert all(close(obs['metrics'][m], ep[m]) for m in METRICS)
            if role == 'selected':
                old = [r for r in previous['rows'] if r['dataset'] == 'insta_market_basket' and r['model'] == model and r['seed'] == seed]
                if old:
                    assert len(old) == 1 and all(close(old[0][m], ep[m]) for m in METRICS)
            row = {'model': model, 'seed': seed, 'endpoint': role, 'epoch': epoch, 'count': ep['count'],
                   **{m: ep[m] for m in METRICS}, 'state_sha256': ep['state_sha256'],
                   'source': str(path.relative_to(R)), 'audit': evidence}
            rows.append(row)
            endpoints[model, seed, role] = ep
            for partition in ('quantity', 'history', 'additional_history'):
                cells = ep[partition + '_cells']
                assert sum(c['count'] for c in cells) == ep['count']
                assert close(sum(c['qty_sse'] for c in cells), ep['qty_sse'])
                for m in ('qty_mae', 'time_nll'):
                    assert close(sum(c['count'] * c[m] for c in cells if c['count']) / ep['count'], ep[m])
                for cell in cells:
                    strata.append({'model': model, 'seed': seed, 'endpoint': role, 'partition': partition,
                                   'boundaries': ep[partition + '_boundaries'], **cell})
assert len(rows) == 42 and len({(r['model'], r['seed'], r['endpoint']) for r in rows}) == 42

groups = [{'model': model, 'endpoint': role, 'n': 3,
           **{m: stats([endpoints[model, seed, role][m] for seed in (42, 52, 62)]) for m in METRICS}}
          for role in ('selected', 'last') for model in MODELS]
g = {(x['model'], x['endpoint']): x for x in groups}
paired = []
for model in MODELS[1:]:
    result = {'comparator': model, 'seeds': [42, 52, 62]}
    for m in METRICS:
        deltas = [endpoints[PLAIN, s, 'selected'][m] - endpoints[model, s, 'selected'][m] for s in (42, 52, 62)]
        result[m] = {'plain_wins': sum(d < 0 for d in deltas), 'ties': sum(d == 0 for d in deltas),
                     'comparator_wins': sum(d > 0 for d in deltas), 'plain_minus_comparator_by_seed': deltas,
                     'paired_difference': stats(deltas),
                     'plain_reduction_percent': 100 * (g[model, 'selected'][m]['mean'] - g[PLAIN, 'selected'][m]['mean']) / g[model, 'selected'][m]['mean']}
    paired.append(result)

stratum_groups = []
for model in MODELS:
    for partition in ('quantity', 'additional_history'):
        for cell in endpoints[model, 42, 'selected'][partition + '_cells']:
            rr = [next(c for c in endpoints[model, s, 'selected'][partition + '_cells'] if c['bin'] == cell['bin']) for s in (42, 52, 62)]
            assert len({c['count'] for c in rr}) == 1
            boundaries = endpoints[model, 42, 'selected'][partition + '_boundaries']
            assert all(endpoints[model, s, 'selected'][partition + '_boundaries'] == boundaries for s in (42, 52, 62))
            assert boundaries == endpoints[PLAIN, 42, 'selected'][partition + '_boundaries']
            assert cell['count'] == next(c['count'] for c in endpoints[PLAIN, 42, 'selected'][partition + '_cells'] if c['bin'] == cell['bin'])
            if not cell['count']:
                continue
            stratum_groups.append({'model': model, 'partition': partition, 'bin': cell['bin'],
                                   'boundaries': boundaries, 'count': cell['count'],
                                   **{m: stats([c[m] for c in rr]) for m in METRICS},
                                   'mean_sse': st.mean(c['qty_sse'] for c in rr)})
decomposition = []
for model in MODELS[1:]:
    for p in ('quantity', 'additional_history'):
        d = []
        for x in [z for z in stratum_groups if z['model'] == PLAIN and z['partition'] == p]:
            y = next(z for z in stratum_groups if z['model'] == model and z['partition'] == p and z['bin'] == x['bin'])
            d.append({'bin': x['bin'], 'count': x['count'],
                      'plain_minus_comparator_mse_contribution': (x['mean_sse'] - y['mean_sse']) / 503733,
                      'plain_minus_comparator_mae_contribution': (x['qty_mae']['mean'] - y['qty_mae']['mean']) * x['count'] / 503733})
        expected_mse = st.mean(endpoints[PLAIN, s, 'selected']['qty_rmse']**2 - endpoints[model, s, 'selected']['qty_rmse']**2 for s in (42, 52, 62))
        assert close(sum(x['plain_minus_comparator_mse_contribution'] for x in d), expected_mse)
        decomposition.append({'comparator': model, 'partition': p, 'cells': d, 'mean_mse_difference': expected_mse})

norm_audit = load('search_artifacts/titantpp_instacart_norm_5080_20260930_v1/retrieved/terminal_20261001_v1/terminal_audit.json')
assert norm_audit['status'] == 'passed' and len(norm_audit['conditions']) == 1
norm = norm_audit['conditions'][0]['selected_validation']
seed42 = {'model': 'titantpp_history_mlp_active_norm', 'seed': 42, 'metrics': norm,
          'plain_reduction_percent': {m: 100*(endpoints[PLAIN,42,'selected'][m]-norm[m])/endpoints[PLAIN,42,'selected'][m] for m in METRICS},
          'same_seed_external_comparison': {model: {m: norm[m]-endpoints[model,42,'selected'][m] for m in METRICS} for model in MODELS[1:]},
          'three_seed_result': False, 'execution': 'RTX4090 2 epochs, approved recovery on RTX5080; original failure preserved'}
cost = [{k: c[k] for k in ('job', 'completed_epochs', 'selected_epoch', 'fit_elapsed_seconds',
                         'epoch_timing_seconds_sum', 'parameter_count', 'peak_allocated_bytes', 'peak_reserved_bytes')} for c in audit['conditions']]
cost_summary = {'conditions': cost, 'fit_elapsed_seconds_sum': sum(r['fit_elapsed_seconds'] for r in cost),
                'epoch_timing_seconds_sum': sum(r['epoch_timing_seconds_sum'] for r in cost),
                'additional_cloud_rental_usd': 0, 'electricity_cost': 'not_measured',
                'scope': 'Recorded train_one and epoch durations; no correction for overhead; not a matched speed comparison against MLP.'}
rankings = {m: [x['model'] for x in sorted([x for x in groups if x['endpoint']=='selected'],key=lambda x:x[m]['mean'])] for m in METRICS}
result = {'completed_analysis_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), 'dataset': 'insta_market_basket',
          'evaluation_scope': 'validation_only', 'selection': 'first strict minimum raw validation quantity RMSE',
          'seeds': [42,52,62], 'std_ddof': 1, 'validation_targets_per_seed': 503733,
          'representative': PLAIN, 'models': MODELS, 'rows': rows, 'three_seed': groups,
          'rankings': rankings, 'paired': paired, 'selected_strata': stratum_groups,
          'squared_error_decomposition': decomposition, 'normalization_seed42_separate': seed42,
          'recorded_cost': cost_summary, 'new_5090_audit': {'path':str((A/'terminal_audit.json').relative_to(R)), 'sha256':sha(A/'terminal_audit.json')},
          'limits': ['Validation used in development, not independent held-out evaluation.',
                     'Three seeds provide descriptive variability, not a significance or equivalence test.',
                     'Common-input/head/loss adaptations, not original native-head results or per-model HPO.',
                     'CPU strict loading and saved-record reconciliation; no new GPU replay or raw-data read.',
                     'Normalization is a separate one-seed, cross-GPU recovery experiment.']}
dump('analysis.json', result)
dump('source_manifest.json', sources)
for name, content in [('conditions.csv',rows), ('strata.csv',strata)]:
    with (O/name).open('w',newline='') as f:
        w=csv.DictWriter(f,list(content[0])); w.writeheader(); w.writerows(content)
verification = {'status':'passed', 'completed_kst':result['completed_analysis_kst'],
                'new_5090_conditions_cpu_audited':6,'retrieved_files':receipt['files'],'source_files':106,
                'checkpoints_cpu_audited':12,'replay_roles_audited':12,
                'comparison_models':7,'comparison_selected_conditions':21,'comparison_endpoint_roles':42,
                'all_three_seed_groups_complete':True,'sample_sd_ddof':1,
                'original_endpoint_metrics_match_prior_frozen_tables':True,
                'selected_last_tensor_sha_reconciled':True,'strata_count_mae_sse_time_nll_reconciled':True,
                'same_seed_pairs':18,'all_unfavorable_seeds_preserved':True,'held_out_read':False,
                'new_training_updates':0,'new_gpu_or_forward_or_replay':False,'manuscript_changed':False,
                'source_manifest_sha256':sha(O/'source_manifest.json'),'analysis_sha256':sha(O/'analysis.json'),
                'analysis_script_sha256':sha(__file__), 'binary_audit_script_sha256':audit['audit_script_sha256']}
dump('verification.json',verification)
print(json.dumps({'groups':[x for x in groups if x['endpoint']=='selected'],'paired':paired,'ranking':rankings,
                  'key_strata':[x for x in stratum_groups if x['model'] in [PLAIN,'s2p2_matched_head','attnhp_matched_head'] and x['partition']=='quantity'],
                  'cost_hours':cost_summary['fit_elapsed_seconds_sum']/3600},ensure_ascii=False,indent=2))
