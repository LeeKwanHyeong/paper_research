"""Reuse three train profiles, compute RAF train only, reconcile 84 saved endpoints.

No model construction, inference, training, remote call, or held-out metric read.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy
import csv
import hashlib
import importlib.util
import json
import math
import statistics as st

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATASETS = ['yellow_trip_hourly', 'intermittent_frozen_5000', 'raf_spare_parts', 'insta_market_basket']
LABELS = dict(zip(DATASETS, ['Taxi', 'Intermittent', 'RAF', 'Instacart']))
MODELS = ['titantpp_history_mlp', 'rmtpp', 'nhp', 'thp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head']
ANCHORS = dict(zip(DATASETS, ['rmtpp', 'rmtpp', 's2p2_matched_head', 's2p2_matched_head']))
METRICS = ['qty_mae', 'qty_rmse', 'time_nll']
sources = {}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def track(p):
    p = Path(p)
    if not p.is_absolute(): p = ROOT / p
    sources[str(p.relative_to(ROOT))] = sha(p)
    return p


def load(p): return json.loads(track(p).read_text())


def rows(p):
    with track(p).open() as f: return list(csv.DictReader(f))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csvout(name, data):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)


def close(a, b): return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-7)


def summarize(values):
    assert len(values) == 3
    return {'mean': st.mean(values), 'sample_sd': st.stdev(values)}


def binlabel(bounds, i):
    f = lambda x: f'{x:g}'
    if i == 0: return '<=' + f(bounds[0])
    if i == len(bounds): return '>' + f(bounds[-1])
    return '(' + f(bounds[i - 1]) + ',' + f(bounds[i]) + ']'


def main():
    OUT.mkdir(exist_ok=True)
    old_path = 'reports/titantpp_data_characteristics_20260928_v1/analysis.json'
    old = load(old_path)
    profiles = copy.deepcopy(old['datasets'])  # do not reuse its partial validation snapshot
    core_contract = load('search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json')
    for spec in core_contract['datasets']:
        p = profiles[spec['dataset_id']]
        identity = spec['inherited_data_identity']
        assert p['sha256'] == identity['data']['sha256'] == sha(track(p['source']))
        assert p['train_targets'] == identity['populations']['train']['target_count']
        assert p['quality_checks_passed'] and p['target_identity_matches_frozen_contract']
    raf_contract_path = 'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/original/execution_contract.json'
    raf_contract = load(raf_contract_path)
    spec = copy.deepcopy(raf_contract['datasets'][0])
    # Same parent population, checked by target identity and exact quantities.
    spec['inherited_data_identity']['data'] = copy.deepcopy(spec['parent_data_identity']['data'])
    spec['inherited_data_identity']['split_manifest'] = copy.deepcopy(spec['parent_data_identity']['split_manifest'])
    helper = track('reports/titantpp_data_characteristics_20260928_v1/analyze.py')
    module_spec = importlib.util.spec_from_file_location('frozen_train_description', helper)
    helper_module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(helper_module)
    original_correlation = helper_module.correlation
    def empty_safe_correlation(x, y, ids):
        # RAF has no entities in one optional long-history subgroup. Preserve
        # its empty status rather than fabricate a correlation or edit the old helper.
        if not len(x):
            return dict(pairs=0, pooled=None, within_entity_demeaned=None,
                entity_correlations_min4_pairs_nonconstant={'n':0},
                entities_with_pairs=0, eligible_entities=0, excluded_entities=0)
        return original_correlation(x,y,ids)
    helper_module.correlation = empty_safe_correlation
    profiles['raf_spare_parts'] = helper_module.profile(spec)
    for key in ('data', 'split_manifest'): track(spec['inherited_data_identity'][key]['path'])
    dump('train_profiles.json', profiles)
    profile_rows = []
    for ds in DATASETS:
        p = profiles[ds]
        profile_rows.append(dict(dataset=ds, split='train', train_series=p['train_series'],
            train_rows=p['train_rows'], train_targets=p['train_targets'],
            quantity_p50=p['quantity_all_train_rows']['p50'], quantity_p95=p['quantity_all_train_rows']['p95'],
            quantity_p99=p['quantity_all_train_rows']['p99'], quantity_max=p['quantity_all_train_rows']['max'],
            gap_p50=p['target_delta_t']['p50'], gap_p95=p['target_delta_t']['p95'], time_unit=p['time_unit'],
            history_p05=p['actual_target_history_events']['p05'], history_p25=p['actual_target_history_events']['p25'],
            history_p50=p['actual_target_history_events']['p50'], history_p75=p['actual_target_history_events']['p75'],
            history_p95=p['actual_target_history_events']['p95'], history_le7_percent=100*p['target_history_le_shares']['7'],
            within_entity_log_lag1_corr=p['adjacent_log_quantity_correlation']['within_entity_demeaned'],
            adjacent_equal_percent=100*p['adjacent_equal_quantity_share'], mean_active_branches=p['mean_available_branches']))
    csvout('train_profiles.csv', profile_rows)

    audit_paths = ['reports/titantpp_core_ablation_execution_20260928_v1/verification.json',
                   'reports/titantpp_completed_5080_audit_20261001_v1/verification.json',
                   'reports/titantpp_instacart_final_analysis_20261001_v1/verification.json',
                   'reports/titantpp_raf_execution_20261001_v1/verification.json']
    for p in audit_paths: assert load(p)['status'] == 'passed'
    core = rows('reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv')
    ext = load('reports/titantpp_completed_external_comparison_20261001_v1/comparison.json')
    insta = rows('reports/titantpp_instacart_final_analysis_20261001_v1/conditions.csv')
    raf = load('reports/titantpp_raf_execution_20261001_v1/comparison.json')
    endpoints, cells, verified_paths = {}, [], []
    for ds in DATASETS:
        for model in MODELS:
            for seed in (42, 52, 62):
                if ds == 'raf_spare_parts':
                    ref = next(r for r in raf['rows'] if r['model'] == model and r['seed'] == seed)
                    path = ROOT / ref['source']
                elif model in MODELS[:5]:
                    ref = next(r for r in core if r['dataset'] == ds and r['arm'] == model and int(r['seed']) == seed and r['endpoint'] == 'selected')
                    path = ROOT / ref['source'] / 'endpoint_replays.json'
                else:
                    host = '5090' if ds == 'insta_market_basket' else '5080'
                    base = ROOT / f'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_{host}_20261001_v1/original/run'
                    path = base / f'{ds}__{seed}__{model}' / 'runs' / model / 'count_only_log_regression' / f'seed_{seed}' / 'endpoint_replays.json'
                    candidates = insta if host == '5090' else ext['rows']
                    ref = next(r for r in candidates if r['model'] == model and int(r['seed']) == seed and (host == '5090' and r['endpoint'] == 'selected' or host == '5080' and r['dataset'] == ds))
                replay = load(path); ep = replay['selected']
                assert ep['evaluation_scope'] == 'validation_only' and not ep['held_out_test_evaluated']
                assert int(ref['epoch']) == replay['best_epoch']
                assert all(close(float(ref[m]), ep[m]) for m in METRICS)
                assert close(ep['qty_sse'] / ep['count'], ep['qty_rmse'] ** 2)
                endpoints[(ds, model, seed)] = ep
                verified_paths.append(dict(dataset=ds, model=model, seed=seed, selected_epoch=replay['best_epoch'],
                    source=str(path.relative_to(ROOT)), source_sha256=sources[str(path.relative_to(ROOT))],
                    model_tensor_sha256=ep['state_sha256'], **{k:ep[k] for k in ['count','qty_sse']+METRICS}))
                for partition, field in [('quantity', 'quantity'), ('history', 'additional_history')]:
                    bounds = ep[field + '_boundaries']; group = ep[field + '_cells']
                    assert len(group) == len(bounds) + 1
                    assert sum(c['count'] for c in group) == ep['count']
                    assert close(sum(c['qty_sse'] for c in group), ep['qty_sse'])
                    for metric in ('qty_mae', 'time_nll'):
                        assert close(sum(c['count']*c[metric] for c in group if c['count'])/ep['count'], ep[metric])
                    for cell in group:
                        if cell['count']:
                            assert close(cell['qty_sse']/cell['count'], cell['qty_rmse']**2)
                        else:
                            assert cell['qty_sse'] == 0 and all(cell[k] is None for k in METRICS)
                        cells.append(dict(dataset=ds, model=model, seed=seed, selected_epoch=replay['best_epoch'],
                            partition=partition, bin=cell['bin'], boundaries=json.dumps(bounds),
                            interval=binlabel(bounds,cell['bin']), count=cell['count'],
                            **{k:cell[k] for k in METRICS+['qty_sse']}, source=str(path.relative_to(ROOT))))
    assert len(endpoints) == 84
    csvout('selected_conditions.csv', verified_paths); csvout('strata_by_seed.csv', cells)
    grouped=[]
    for ds in DATASETS:
        for model in MODELS:
            for partition in ('quantity','history'):
                group = [r for r in cells if r['dataset']==ds and r['model']==model and r['partition']==partition]
                for i in sorted({r['bin'] for r in group}):
                    rs=[r for r in group if r['bin']==i]
                    assert sorted(r['seed'] for r in rs)==[42,52,62]
                    assert len({(r['boundaries'],r['count'],r['interval']) for r in rs})==1
                    n=rs[0]['count']; summary=dict(dataset=ds,model=model,partition=partition,bin=i,
                        boundaries=rs[0]['boundaries'],interval=rs[0]['interval'],count=n,
                        share_percent=100*n/endpoints[(ds,model,42)]['count'],mean_sse=st.mean(r['qty_sse'] for r in rs))
                    for m in METRICS:
                        s=summarize([r[m] for r in rs]) if n else dict(mean=None,sample_sd=None)
                        summary[m+'_mean']=s['mean'];summary[m+'_sample_sd']=s['sample_sd']
                    grouped.append(summary)
    csvout('strata_three_seed.csv',grouped)
    anchor_rows=[]; decompositions=[]
    for ds in DATASETS:
        anchor=ANCHORS[ds]; total=endpoints[(ds,anchor,42)]['count']
        for partition in ('quantity','history'):
            plain=[r for r in grouped if r['dataset']==ds and r['model']==MODELS[0] and r['partition']==partition]
            comp=[r for r in grouped if r['dataset']==ds and r['model']==anchor and r['partition']==partition]
            denom=st.mean(endpoints[(ds,anchor,s)]['qty_sse']/total for s in (42,52,62))
            for x,y in zip(plain,comp):
                assert (x['bin'],x['boundaries'],x['count'])==(y['bin'],y['boundaries'],y['count'])
                n=x['count']; dmse=(x['mean_sse']-y['mean_sse'])/total
                dmae=(x['qty_mae_mean']-y['qty_mae_mean'])*n/total if n else 0.
                anchor_rows.append(dict(dataset=ds,anchor=anchor,partition=partition,bin=x['bin'],
                    interval=x['interval'],count=n,share_percent=x['share_percent'],
                    mlp_mae=x['qty_mae_mean'],mlp_rmse=x['qty_rmse_mean'],mlp_time_nll=x['time_nll_mean'],
                    anchor_mae=y['qty_mae_mean'],anchor_rmse=y['qty_rmse_mean'],anchor_time_nll=y['time_nll_mean'],
                    mse_contribution=dmse,mae_contribution=dmae,relative_mse_contribution_percent=100*dmse/denom))
            rs=[r for r in anchor_rows if r['dataset']==ds and r['partition']==partition]
            expected=st.mean((endpoints[(ds,MODELS[0],s)]['qty_sse']-endpoints[(ds,anchor,s)]['qty_sse'])/total for s in (42,52,62))
            assert close(sum(r['mse_contribution'] for r in rs),expected)
            expected_mae=st.mean(endpoints[(ds,MODELS[0],s)]['qty_mae']-endpoints[(ds,anchor,s)]['qty_mae'] for s in (42,52,62))
            assert close(sum(r['mae_contribution'] for r in rs),expected_mae)
            decompositions.append(dict(dataset=ds,anchor=anchor,partition=partition,
                mae_difference=expected_mae,mse_difference=expected,
                low_quantity_mse_contribution=sum(r['mse_contribution'] for r in rs if r['bin']<=1) if partition=='quantity' else None,
                high_quantity_mse_contribution=sum(r['mse_contribution'] for r in rs if r['bin']>1) if partition=='quantity' else None))
    csvout('anchor_strata.csv',anchor_rows);dump('error_decomposition.json',decompositions)
    dump('analysis.json',dict(created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
        scope='train descriptors; completed selected validation endpoints; no heldout',
        train_profiles=profiles,anchors=ANCHORS,selected_conditions=verified_paths,
        strata=grouped,anchor_strata=anchor_rows,decompositions=decompositions,
        methodology=dict(train_reused=DATASETS[:2]+[DATASETS[3]],new_train_profile='raf_spare_parts',
            quantile_method='nearest',train_quantity_population='all train event rows',
            gap_and_history_population='canonical train targets excluding first event per entity',
            correlation='pooled residual Pearson correlation of adjacent log1p quantity pairs after separate within-entity means for predecessor and successor',
            validation_selection='same earliest strict minimum raw quantity RMSE checkpoint for every metric and stratum',
            validation_bins='frozen dataset-specific boundaries, right endpoint included, empty cells preserved',
            history_partition='saved additional_history; Instacart [1,3,7,15,31], others [64,128]',
            aggregation='event weighted within seed; arithmetic mean and sample SD across seeds42/52/62',
            anchor_rule='previously fixed strongest external mean validation RMSE model, unchanged by strata',
            mse_difference='mean over seeds of (SSE_MLP-SSE_anchor)/total_targets; not square of mean RMSE'),
        new_training=False,new_model_inference=False,heldout_read=False))
    dump('sources.json',sources)
    dump('verification.json',dict(status='passed',selected_conditions=84,three_seed_groups=28,
        source_endpoints=84,seed_stratum_rows=len(cells),aggregate_stratum_rows=len(grouped),
        empty_aggregate_strata=sum(r['count']==0 for r in grouped),
        three_train_profiles_reused_unchanged=all(profiles[d]==old['datasets'][d] for d in old['datasets']),
        raf_train_identity_quantity_and_initialization_verified=True,
        all_selected_metrics_match_final_tables=True,all_partition_counts_and_sums_reconcile=True,
        aggregate_RMSE_not_computed_from_mean_bin_RMSE=True,all_empty_cells_preserved=True,
        audits_reused=audit_paths,heldout_predictions_or_metrics_read=False,new_fit_or_forward_or_replay=False))
    print(json.dumps({'status':'passed','profiles':profile_rows,'conditions':len(endpoints),'strata':len(grouped),'decompositions':decompositions},ensure_ascii=False))


if __name__ == '__main__': main()
