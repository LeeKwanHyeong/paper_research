"""Stored-result analysis only. No torch, inference, remote access or model edits."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics as st
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/titantpp_three_dataset_final_comparison_20261005_v1'
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts')
MODELS = ('titantpp_history_mlp_width16', 'titantpp_history_mlp',
          'titantpp_history_mlp_width8', 'titantpp_history_mlp_width12',
          'titantpp_all_available_history_mlp', 'titantpp_current_only_param_matched', 'titantpp')
EXT = ('rmtpp', 'thp', 'nhp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head')
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')
verified = {}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify(path, expected):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    actual = digest(path)
    assert actual == expected, (str(path), actual, expected)
    verified[str(path.relative_to(ROOT))] = actual

def load_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def write_json(name, data):
    with (OUT / name).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def write_csv(name, rows):
    with (OUT / name).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

receipt = json.loads((BASE / 'comparison_receipt.json').read_text())
assert receipt['status'] == 'complete' and receipt['baseline_changed'] is False
verified[str((BASE / 'comparison_receipt.json').relative_to(ROOT))] = digest(BASE / 'comparison_receipt.json')
for path, expected in receipt['source_files_sha256'].items():
    verify(path, expected)
for path, expected in receipt['outputs'].items():
    verify(BASE / path, expected)

rows = [r for r in load_csv(BASE / 'metrics_per_seed.csv')
        if r['dataset'] in DATASETS and r['model'] in MODELS + EXT]
index = {}
for row in rows:
    key = tuple(row[k] for k in ('dataset', 'model', 'split', 'view', 'seed'))
    assert key not in index
    index[key] = row
    verify(row['source'], row['source_sha256'])
    for metric in METRICS:
        assert math.isfinite(float(row[metric]))

# Independent recomputation from per-seed values; cross-check the frozen table.
old_summary = {tuple(r[k] for k in ('dataset', 'model', 'split', 'view')): r
               for r in load_csv(BASE / 'metrics_aggregated.csv')}
groups = defaultdict(list)
for key, row in index.items():
    groups[key[:4]].append(row)
summary = []
for key, group in sorted(groups.items()):
    assert {r['seed'] for r in group} == {'42', '52', '62'}
    assert len({r['count'] for r in group}) == 1
    entry = dict(zip(('dataset', 'model', 'split', 'view'), key))
    entry.update(seed_count=3, count_per_seed=int(group[0]['count']))
    for metric in METRICS:
        values = [float(r[metric]) for r in group]
        entry[metric + '_mean'], entry[metric + '_sample_sd'] = st.mean(values), st.stdev(values)
        for suffix in ('_mean', '_sample_sd'):
            assert math.isclose(entry[metric + suffix], float(old_summary[key][metric + suffix]), rel_tol=1e-11, abs_tol=1e-11)
    summary.append(entry)

pairs = [(m, MODELS[0]) for m in MODELS[1:]] + [
    ('titantpp_all_available_history_mlp', 'titantpp_history_mlp'),
    ('titantpp_current_only_param_matched', 'titantpp_history_mlp')]
deltas = []
for dataset in DATASETS:
    for a, b in pairs:
        if dataset == 'raf_spare_parts' and a == 'titantpp':
            continue  # No historical TitanB RAF3 in this frozen comparison.
        for split in ('validation', 'test'):
            for view in ('overall', 'tail'):
                for seed in ('42', '52', '62'):
                    x, y = index[dataset, a, split, view, seed], index[dataset, b, split, view, seed]
                    assert x['count'] == y['count']
                    for metric in METRICS:
                        av, bv = float(x[metric]), float(y[metric])
                        deltas.append(dict(dataset=dataset, candidate=a, reference=b, split=split,
                                           view=view, seed=int(seed), metric=metric,
                                           candidate_value=av, reference_value=bv,
                                           delta=av-bv, relative_percent=100*(av/bv-1),
                                           improved=av < bv))

# Reconstruct non-tail sums from disjoint full/tail aggregates, never individual rows.
decomposition = []
for dataset in DATASETS:
    for a, b in pairs:
        if dataset == 'raf_spare_parts' and a == 'titantpp':
            continue
        for split in ('validation', 'test'):
            for seed in ('42', '52', '62'):
                result = dict(dataset=dataset, candidate=a, reference=b, split=split, seed=int(seed))
                for model, label in ((a, 'candidate'), (b, 'reference')):
                    full, tail = index[dataset, model, split, 'overall', seed], index[dataset, model, split, 'tail', seed]
                    n, nt = int(full['count']), int(tail['count'])
                    assert 0 < nt < n
                    result.update(full_n=n, tail_n=nt, non_tail_n=n-nt)
                    se = n*float(full['qty_rmse'])**2
                    setail = nt*float(tail['qty_rmse'])**2
                    ae = n*float(full['qty_mae'])
                    aetail = nt*float(tail['qty_mae'])
                    assert se >= setail and ae >= aetail
                    for k, v in {'full_sse':se, 'tail_sse':setail, 'non_tail_sse':se-setail,
                                 'full_sae':ae, 'tail_sae':aetail, 'non_tail_sae':ae-aetail,
                                 'non_tail_rmse':math.sqrt((se-setail)/(n-nt)),
                                 'non_tail_mae':(ae-aetail)/(n-nt)}.items():
                        result[label+'_'+k] = v
                for k in ('full_sse','tail_sse','non_tail_sse','full_sae','tail_sae','non_tail_sae','non_tail_rmse','non_tail_mae'):
                    result['delta_'+k] = result['candidate_'+k]-result['reference_'+k]
                for suffix in ('sse', 'sae'):
                    assert math.isclose(result['delta_full_'+suffix], result['delta_tail_'+suffix]+result['delta_non_tail_'+suffix], rel_tol=1e-9, abs_tol=1e-6)
                decomposition.append(result)

selected = [r for r in load_csv(BASE/'selected_epochs.csv') if r['dataset'] in DATASETS and r['model'] in MODELS]
for r in selected:
    verify(r['checkpoint_path'], r['checkpoint_file_sha256'])
    for split in ('validation','test'):
        for view in ('overall','tail'):
            assert int(index[r['dataset'],r['model'],split,view,r['seed']]['selected_epoch']) == int(r['selected_epoch'])

# Bind historical source code, not today's differing width implementation.
regpath = ROOT/'reports/titantpp_completed_validation_test_20261004_v1/width16/evaluation_registry.json'
registry = json.loads(regpath.read_text())
structures = []
for name, bundle in registry['bundles'].items():
    for path in ('models/TPPs/CountAwareTitanCoreAblation.py', 'models/TPPs/CountAwareTitanHistoryWidth.py',
                 'models/TPPs/CountAwareTitanHistoryControls.py', 'models/TPPs/CountAwareTitanMultiLagDetail.py'):
        frozen = ROOT/bundle['source_root']/path
        verify(frozen, bundle['source_files'][path])
        structures.append(dict(bundle=name, path=str(frozen.relative_to(ROOT)), sha256=digest(frozen),
                               current_worktree_identical=digest(ROOT/path)==digest(frozen)))

# Train diagnostic is a weighted fixed sample for widths8/12 vs4, not comparable to16's sample.
cap = ROOT/'reports/titantpp_history_capacity_comparison_20261004_v1/snapshots/20261004T234019109234Z'
bindings = json.loads((cap/'source_bindings.json').read_text())
train = [r for r in load_csv(cap/'train_fixed_sample_metrics.csv') if r['dataset'] in DATASETS]
for row in train:
    fit = next(f for f in bindings['candidate_fits'] if f['job']['dataset']==row['dataset'] and f['job']['seed']==int(row['seed']) and f['job']['arm']=='titantpp_history_mlp_width'+row['width'])
    source = Path(row['source'])
    rel = str(source.relative_to(Path(fit['folder'])))
    verify(source, fit['original_files_sha256'][rel])
    assert fit['train_comparator_width16_available'] is False
    row['source_sha256'] = digest(source)

external = []
for dataset in DATASETS:
    for split in ('validation','test'):
        for metric in ('qty_rmse','qty_mae'):
            best = min((r for r in summary if r['dataset']==dataset and r['split']==split and r['view']=='overall' and r['model'] in EXT), key=lambda r:(r[metric+'_mean'],r['model']))
            anchor = next(r for r in summary if r['dataset']==dataset and r['split']==split and r['view']=='overall' and r['model']==MODELS[0])
            external.append(dict(dataset=dataset,split=split,metric=metric,best_external_model=best['model'],
                                 best_external_mean=best[metric+'_mean'],width16_mean=anchor[metric+'_mean'],
                                 width16_minus_external=anchor[metric+'_mean']-best[metric+'_mean'],
                                 relative_percent=100*(anchor[metric+'_mean']/best[metric+'_mean']-1)))

write_csv('SUMMARY.csv',summary)
write_csv('SEED_DELTAS.csv',deltas)
write_csv('TAIL_NONTAIL_DECOMPOSITION.csv',decomposition)
write_csv('SELECTED_CHECKPOINTS.csv',selected)
write_csv('TRAIN_FIXED_SAMPLE.csv',train)
write_json('RESULT.json',dict(verification_status='ANALYZED',summary=summary,external_best=external,
                              historical_source=structures,baseline=MODELS[0],baseline_reselected=False,
                              new_training=0,new_inference=0,remote_calls=0,
                              note='Stored fixed-checkpoint aggregates; retrospective Test-informed analysis. No row-level causal attribution.'))
write_json('INPUT_MANIFEST.json',dict(verified_files=verified,
                                     additional_analysis_inputs={str(p.relative_to(ROOT)):digest(p) for p in (cap/'source_bindings.json',cap/'train_fixed_sample_metrics.csv')}))
print(json.dumps(dict(verified_file_count=len(verified),summary_groups=len(summary),paired_metric_rows=len(deltas),
                      decomposition_rows=len(decomposition),selected_binary_count=len(selected),train_sample_rows=len(train),external=external),ensure_ascii=False,indent=2))
