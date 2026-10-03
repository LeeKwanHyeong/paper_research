"""Read saved validation evidence only; never import a model or launch a run."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {}
CHECKS = []


def read(path):
    p = ROOT / path
    assert not any(x.startswith('test_') for x in p.parts), p
    raw = p.read_bytes()
    SOURCES[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return raw.decode()


def js(path):
    return json.loads(read(path))


def check(name, value):
    CHECKS.append({'check': name, 'passed': bool(value)})
    if not value:
        raise AssertionError(name)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def aggregate(rows, model_key='model'):
    groups = defaultdict(list)
    for r in rows:
        groups[(r['dataset'], r[model_key])].append(r)
    result = []
    for (dataset, model), group in sorted(groups.items()):
        row = {'dataset': dataset, 'model': model, 'n_seeds': len(group),
               'seeds': ';'.join(map(str, sorted(x['seed'] for x in group)))}
        for metric in ('qty_rmse', 'qty_mae', 'time_nll'):
            vals = [x[metric] for x in group]
            row[metric + '_mean'] = statistics.mean(vals)
            row[metric + '_sample_sd'] = statistics.stdev(vals) if len(vals) > 1 else None
        result.append(row)
    return result


core_root = Path('search_artifacts/titantpp_core_ablation_20260928_v1')
contract = js(core_root / 'frozen_execution/execution_contract.json')
canonical = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
check('current canonical execution contract', canonical == 'eff125f587f7a9a8097d5d43b3d688eac481abb78bb6006b5aa4ab1b0ec2a88d')
current_datasets = {d['dataset_id']: d for d in contract['datasets']}
# Freeze the reviewed local snapshot. Do not silently switch to a newer aggregation.
ledger_path = core_root / 'aggregation/20260928T015549834592Z/ledger.json'
ledger = js(ledger_path)
check('ledger scope', ledger['evaluation_scope'] == 'validation_only')
check('ledger contract', ledger['contract_sha256'] == canonical)

historical_root = Path('search_artifacts/count_aware_b012_seed42_screening_e300_20260828_recovery1')
historical = list(csv.DictReader(read(historical_root / 'comparison/metrics.csv').splitlines()))
costs = list(csv.DictReader(read('paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/historical_cost.csv').splitlines()))
legacy_rows = []
for dataset in sorted({x['dataset'] for x in historical}):
    group = {x['backbone']: x for x in historical if x['dataset'] == dataset}
    b0, b1 = group['titantpp'], group['titantpp_titans_mac']
    summary = js(historical_root / f'shards/{dataset}/titantpp_titans_mac/titan_b012_screening/runs/titantpp_titans_mac/count_only_log_regression/seed_42/summary.json')
    check(f'legacy B1 {dataset} validation only', summary['evaluation_scope'] == 'validation_only' and summary['held_out_test_evaluated'] is False)
    check(f'legacy B1 {dataset} unclipped policy', summary['encoder_config'].get('titans_memory_gradient_clip') is None)
    check(f'legacy B1 {dataset} table vs summary', close(summary['best_val_qty_rmse'], b1['overall_qty_rmse']))
    c = next(x for x in costs if x['dataset'] == dataset)
    check(f'legacy cost ratio {dataset}', close(float(c['b1_seconds_per_completed_epoch']) / float(c['b0_seconds_per_completed_epoch']), c['b1_b0_epoch_cost_ratio']))
    legacy_rows.append({'dataset': dataset, 'seed': 42, 'policy': 'historical_unbounded_inner_gradient',
                        'b0_qty_rmse': float(b0['overall_qty_rmse']), 'b1_qty_rmse': float(b1['overall_qty_rmse']),
                        'b1_vs_b0_rmse_change_pct': (float(b1['overall_qty_rmse']) / float(b0['overall_qty_rmse']) - 1) * 100,
                        'b0_qty_mae': float(b0['overall_qty_mae']), 'b1_qty_mae': float(b1['overall_qty_mae']),
                        'b1_vs_b0_tail_mae_change_pct': (float(b1['gt_p99_mae']) / float(b0['gt_p99_mae']) - 1) * 100,
                        'historical_epoch_cost_ratio': float(c['b1_b0_epoch_cost_ratio'])})

stable_csv = list(csv.DictReader(read('paper/results/count_aware_titantpp_mac_stable_20260902/seed_results.csv').splitlines()))
stable_rows, compatibility = [], []
for seed, host in [(52, 5090), (62, 5080)]:
    base = Path(f'search_artifacts/count_aware_titantpp_mac_stable_seed{seed}_e300_20260831_{host}')
    status, execution = js(base / 'status.json'), js(base / 'execution_contract.json')
    check(f'MAC seed{seed} four completed', status['completed_run_count'] == status['expected_run_count'] == 4)
    check(f'MAC seed{seed} clipped policy', execution['training']['inner_gradient_clip'] == 1)
    for launch in sorted((ROOT / base / 'validation_e300').glob(f'*/seed_{seed}/launch_contract.json')):
        relative = launch.relative_to(ROOT)
        l = js(relative)
        run = relative.parent / f'runs/titantpp_titans_mac/count_only_log_regression/seed_{seed}'
        s = js(run / 'summary.json')
        h = js(run / 'history.json')['history']
        dataset = l['dataset']
        tag = f'MAC {dataset} seed{seed}'
        check(tag + ' validation only', s['evaluation_scope'] == l['evaluation_scope'] == 'validation_only' and s['held_out_test_evaluated'] is False and l['held_out_test_evaluated'] is False)
        check(tag + ' success', s['status'] == 'success' and l['status'] == 'complete')
        check(tag + ' inner clipping', l['titans_memory_gradient_clip'] == s['encoder_config']['titans_memory_gradient_clip'] == 1)
        check(tag + ' finite history', all(math.isfinite(r[m]) for r in h for m in ['val_joint_objective', 'val_qty_rmse', 'val_qty_mae', 'val_time_nll']))
        joint = min(h, key=lambda r: r['val_joint_objective'])
        raw = min(h, key=lambda r: r['val_qty_rmse'])
        check(tag + ' original joint selector', joint['epoch'] == s['best_epoch'] and close(joint['val_qty_rmse'], s['best_val_qty_rmse']))
        check(tag + ' history coverage', len(h) == s['completed_epochs'])
        tab = next(x for x in stable_csv if x['dataset'] == dataset and int(x['seed']) == seed)
        check(tag + ' aggregate CSV correspondence', all(close(tab[k], s[v]) for k,v in [('quantity_rmse','best_val_qty_rmse'),('quantity_mae','best_val_qty_mae'),('time_nll','best_val_time_nll')]) and tab['checkpoint_state_sha256'] == s['checkpoint_state_sha256'])
        stable_rows.append({'dataset': dataset, 'seed': seed, 'model': 'Titans-MAC-innerclip1-legacy-joint-selection',
                            'qty_rmse': s['best_val_qty_rmse'], 'qty_mae': s['best_val_qty_mae'],
                            'time_nll': s['best_val_time_nll'], 'best_epoch': s['best_epoch'],
                            'completed_epochs': s['completed_epochs'], 'fit_seconds': s['elapsed_seconds'],
                            'parameters': s['parameter_count'], 'history_raw_rmse_min_epoch_diagnostic_only': raw['epoch'],
                            'history_raw_rmse_min_value_diagnostic_only': raw['val_qty_rmse'],
                            'source': str(run / 'summary.json')})
        if dataset in current_datasets:
            cur = current_datasets[dataset]
            comp = {'dataset': dataset, 'seed': seed,
                    'same_data_sha': l['data_sha256'] == cur['inherited_data_identity']['data']['sha256'],
                    'same_split_sha': l['split_manifest_sha256'] == cur['inherited_data_identity']['split_manifest']['sha256'],
                    'same_max_seq_len': l['max_seq_len'] == cur['loader']['max_seq_len'],
                    'same_lookback': l['lookback_weeks'] == cur['loader']['lookback_weeks'],
                    'same_train_count': l['time_head']['train_time_statistics']['target_count'] == cur['time_statistics']['expected_train_targets'],
                    'same_validation_count': sum(r['count'] for r in s['quantity_rows']) == cur['time_statistics']['expected_validation_targets'],
                    'same_batch': l['batch_size'] == cur['loader']['batch_size'],
                    'same_lr': l['lr'] == cur['optimizer']['lr'],
                    'old_time_head': l['time_head']['mode'], 'current_time_head': cur['model']['time_head_mode'],
                    'old_selector': l['early_stopping']['monitor'], 'current_selector': contract['training']['monitor'],
                    'old_time_scale': l['time_head']['time_scale'], 'current_time_scale': cur['model']['time_scale'],
                    'direct_matched_backbone_comparison_admissible': False}
            compatibility.append(comp)
check('eight stable MAC summaries', len(stable_rows) == 8)

previous = js('reports/local_detail_3seed_final_20260927_v1/condition_results.json')
audit = js('reports/local_detail_3seed_final_20260927_v1/terminal_audit.json')
check('previous complete terminal audit', audit['status'] == 'passed' and audit['conditions'] == 54 and audit['endpoint_replays'] == 108 and audit['validation_only'] is True)
current_rows = []
for r in previous:
    tag = f"current {r['dataset']}/{r['seed']}/{r['model']}"
    base = Path(r['source'])
    s, h, endpoint = js(base / 'summary.json'), js(base / 'history.json')['history'], js(base / 'endpoint_replays.json')
    best = min(h, key=lambda x: x['val_qty_rmse'])
    check(tag + ' raw selector', best['epoch'] == r['best_epoch'] == s['best_epoch'])
    check(tag + ' observation head', s['encoder_config']['time_head']['mode'] == 'heteroscedastic_lognormal_duration' and s['encoder_config']['time_head']['reported_metric'] == 'recorded_positive_integer_time_nll')
    check(tag + ' source and replay', endpoint['selected']['state_sha256'] == r['selected']['state_sha256'] == s['checkpoint_state_sha256'] and all(close(endpoint['selected'][k], r['selected'][k]) and close(best['val_' + k], r['selected'][k]) for k in ['qty_rmse','qty_mae','time_nll']))
    check(tag + ' validation scope', s['held_out_test_evaluated'] is False and endpoint['selected']['held_out_test_evaluated'] is False)
    cur = current_datasets[r['dataset']]
    check(tag + ' count', r['selected']['count'] == cur['time_statistics']['expected_validation_targets'])
    current_rows.append({'dataset': r['dataset'], 'seed': r['seed'], 'model': r['model'],
                         **{k:r['selected'][k] for k in ['qty_rmse','qty_mae','time_nll']},
                         'source': str(base), 'cohort': 'completed_3seed_shared_head'})
bykey = {(x['dataset'], x['seed'], x['model']): x for x in previous}
core_rows = []
for r in ledger['conditions']:
    if r['execution_status'] != 'complete':
        continue
    check(f"core {r['dataset']}/{r['seed']}/{r['arm']} audited", r['audit_status'] == 'passed')
    if r['reused']:
        old = bykey[(r['dataset'], r['seed'], r['arm'])]
        check(f"reuse identical {r['dataset']}/{r['seed']}/{r['arm']}", old['selected']['state_sha256'] == r['selected']['state_sha256'] and all(close(old['selected'][m],r['selected'][m]) for m in ['qty_rmse','qty_mae','time_nll']))
    core_rows.append({'dataset': r['dataset'], 'seed': r['seed'], 'model': r['arm'],
                      **{k:r['selected'][k] for k in ['qty_rmse','qty_mae','time_nll']},
                      'source': r['source'], 'cohort': 'core_saved_snapshot'})
unique = {(r['dataset'],r['seed'],r['model']):r for r in current_rows}
for r in core_rows:
    unique.setdefault((r['dataset'],r['seed'],r['model']),r)
check('unique complete union 71',len(unique)==71)
check('existing external encoder fits 36',sum(r['model'] in ['rmtpp','thp','nhp','sahp'] for r in current_rows)==36)

paired = []
for dataset in current_datasets:
    for model in ['titantpp_history_mlp','titantpp_level_only','titantpp_change_only','titantpp_no_static_lmm']:
        pairs = []
        for seed in [42,52,62]:
            cand = unique.get((dataset,seed,model))
            full = unique.get((dataset,seed,'titantpp_local_detail'))
            if cand and full:
                pairs.append((full,cand))
        if pairs:
            fm = statistics.mean(x[0]['qty_rmse'] for x in pairs)
            cm = statistics.mean(x[1]['qty_rmse'] for x in pairs)
            paired.append({'dataset':dataset,'comparison':model,'seeds':';'.join(str(x[0]['seed']) for x in pairs),'n_seeds':len(pairs),'full_rmse_matched_mean':fm,'comparison_rmse_matched_mean':cm,'comparison_vs_full_pct':100*(cm/fm-1),'full_time_matched_mean':statistics.mean(x[0]['time_nll'] for x in pairs),'comparison_time_matched_mean':statistics.mean(x[1]['time_nll'] for x in pairs)})

for p in ['paper/contracts/count_aware_titans_backbone_reproduction_v1.json',
          'paper/contracts/count_aware_titantpp_mac_stable_seed52_5090_v1.md',
          'paper/results/titantpp_mac_inner_stability_incident_20260831/README.md',
          'paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/decision.json',
          'reports/titantpp_followup_contract_5080_20260927_v1/experiment_contract.md']:
    read(p)

write_csv('historical_seed42.csv',legacy_rows)
write_csv('stable_mac_conditions.csv',stable_rows)
write_csv('stable_mac_two_seed_summary.csv',aggregate(stable_rows))
write_csv('mac_current_compatibility.csv',compatibility)
write_csv('current_comparable_summary.csv',aggregate(list(unique.values())))
write_csv('full_component_pairs.csv',paired)
write_json('analysis.json',{'evaluation_scope':'validation_only','generated_at_utc':datetime.now(timezone.utc).isoformat(),
 'source_snapshot_kst':'2026-09-28 10:55 KST; saved evidence, no live remote check',
 'historical_seed42':legacy_rows,'stable_mac':stable_rows,'stable_mac_summary':aggregate(stable_rows),
 'mac_current_compatibility':compatibility,'current_summary':aggregate(list(unique.values())),
 'full_component_pairs':paired,'counts':{'previous_comparable_complete':54,'existing_external_encoder_fits':36,'core_complete':len(core_rows),'core_pending':54-len(core_rows),'complete_unique_union':len(unique),'mac_stable_seeds':[52,62],'mac_stable_conditions':len(stable_rows)},
 'decision':{'external_benchmarks_alone_sufficient_for_current_backbone_claim':False,
 'reuse_existing_36_external_encoder_fits':True,'reuse_B_and_Full_18':True,
 'matched_stable_MAC_fits_if_retaining_original_Titans_comparison_claim':9,
 'new_MAC_fits_authorized_or_launched':False,
 'no_new_Full_or_gate_required_by_this_audit':True},
 'limitations':['MAC historical seed42 uses unbounded inner gradients; do not pool with clipped seeds52/62.',
 'Legacy time loss is clamped; its time_nll is not comparable to current recorded-time probability-mass NLL.',
 'Historical minima over stored epochs are diagnostics, not new selected checkpoints or replays.',
 'Data/split byte hashes, context limits and counts are compared; legacy target/prefix identity hashes are not independently reconstructed.',
 'Prior replay reports are inspected as evidence; checkpoint binaries are not loaded and no GPU inference is performed.',
 'Current new-arm results are a partial local snapshot; pending and unsuccessful earlier attempts are not dropped from the research record.',
 'No held-out performance, raw held-out rows, current remote state or paper acceptance probability is assessed.']})
write_json('verification.json',{'status':'passed','checks':CHECKS,'checked_source_files':len(SOURCES),'model_or_training_code_changed':False,'new_training_or_replay':False})
write_json('sources.json',{'files_sha256':SOURCES})
print(json.dumps({'status':'passed','checks':len(CHECKS),'sources':len(SOURCES),'old_mac':len(stable_rows),'current_unique':len(unique),'paired_rows':len(paired),'output':str(OUT)},ensure_ascii=False))
