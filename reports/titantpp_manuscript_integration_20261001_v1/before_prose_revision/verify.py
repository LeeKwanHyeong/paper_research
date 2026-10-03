"""Check the manuscript against existing records only; no model execution or network."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MAIN = ROOT / 'paper/titantpp_history_mlp_manuscript_20261001_v1.md'


def read(relative):
    return json.loads((ROOT / relative).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


text = MAIN.read_text()
checks = {}
method_root = 'reports/titantpp_method_efficiency_20260930_v1/'
eff_root = 'reports/titantpp_efficiency_5080_20260930_v1/'
cmp_root = 'reports/titantpp_completed_external_comparison_20261001_v1/'
mv = read(method_root + 'verification.json')
eff = read(eff_root + 'analysis.json')
comp = read(cmp_root + 'comparison.json')
contract_path = 'search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json'
contract = read(contract_path)
frozen = ROOT / 'search_artifacts/titantpp_core_ablation_20260928_v1/final_audit_20260930_v1/terminal_5090/source'
source_checks = [{
    'path': str((frozen / row['path']).relative_to(ROOT)),
    'expected': row['expected'],
    'actual': sha(frozen / row['path']),
} for row in mv['locally_verified_method_source_files']]
checks['eight_retrieved_frozen_method_source_hashes_match'] = (
    len(source_checks) == 8 and all(r['actual'] == r['expected'] for r in source_checks))
checks['contract_file_matches_prior_verified_method_source'] = (
    sha(ROOT / contract_path) == read(method_root + 'sources.json')[contract_path])
old = (ROOT / (method_root + 'method_draft.md')).read_text()
blocks = lambda s: re.findall(r'\$\$\n(.*?)\n\$\$', s, re.S)
checks['all_nine_equations_reused_exactly'] = blocks(text) == blocks(old) and len(blocks(text)) == 9
checks['equation_tags_unique_one_to_nine'] = re.findall(r'\\tag\{(\d+)\}', text) == list(map(str, range(1, 10)))
checks['cpu_equation_evidence_reused'] = (
    mv['status'] == 'passed'
    and mv['cpu_equation_checks']['parameter_count'] == 6144
    and mv['cpu_equation_checks']['explicit_equation_max_absolute_error'] < 1e-12)
checks['matched_cost_audit_reused'] = read(eff_root + 'verification.json')['status'] == 'passed'

names = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent', 'insta_market_basket': 'Instacart'}
metrics = ['qty_mae', 'qty_rmse', 'time_nll']
agg = {(r['dataset'], r['model']): r for r in comp['three_seed']}
external = ['rmtpp', 'thp', 'nhp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head']
numeric_failures = []
for dataset in names:
    for model, label in [('titantpp', 'B: no correction'), ('titantpp_history_mlp', 'TitanTPP')]:
        rows = [r for r in comp['rows'] if r['dataset'] == dataset and r['model'] == model]
        assert sorted(r['seed'] for r in rows) == [42, 52, 62]
        group = agg[dataset, model]
        cells = []
        for metric in metrics:
            values = [r[metric] for r in rows]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            if not (close(mean, group[metric]['mean']) and close(sd, group[metric]['sample_sd'])):
                numeric_failures.append((dataset, model, metric))
            cells.append(f'{mean:.4f} ± {sd:.4f}')
        assert '| ' + ' | '.join([names[dataset], label] + cells) + ' |' in text
checks['table3_recomputed_from_selected_seed_rows'] = not numeric_failures
paired = []
for dataset in list(names)[:2]:
    for metric in ['qty_mae', 'qty_rmse']:
        assert min(external, key=lambda m: agg[dataset, m][metric]['mean']) == 'rmtpp'
    mlp = {r['seed']: r for r in comp['rows'] if r['dataset'] == dataset and r['model'] == 'titantpp_history_mlp'}
    for model in external:
        other = {r['seed']: r for r in comp['rows'] if r['dataset'] == dataset and r['model'] == model}
        assert sorted(other) == [42, 52, 62]
        for seed in [42, 52, 62]:
            paired.append(all(mlp[seed][m] < other[seed][m] for m in ['qty_mae', 'qty_rmse']))
    a, b = agg[dataset, 'titantpp_history_mlp'], agg[dataset, 'rmtpp']
    ratios = [100 * (1 - a[m]['mean'] / b[m]['mean']) for m in ['qty_mae', 'qty_rmse']]
    line = f"| {names[dataset]} | {ratios[0]:.2f}% | {ratios[1]:.2f}% | {a['time_nll']['mean']:.4f} | {b['time_nll']['mean']:.4f} |"
    assert line in text
checks['table4_recomputed_and_strongest_external_verified'] = True
checks['same_seed_both_quantity_wins_36_of_36'] = len(paired) == 36 and all(paired)

for row in eff['comparison']:
    ds, mode = row['dataset'], row['mode']
    by_arm = {}
    for prefix, arm in [('mlp', 'titantpp_history_mlp'), ('mac', 'titantpp_titans_mac')]:
        reps = sorted([r for r in eff['repetitions'] if r['dataset'] == ds and r['mode'] == mode and r['arm'] == arm], key=lambda r: r['rep'])
        assert len(reps) == 3, (ds, mode, arm)
        by_arm[prefix] = reps
        vals = [r['median_compute_ms'] for r in reps]
        assert close(statistics.mean(vals), row[prefix + '_median_compute_ms_mean'])
        assert close(statistics.stdev(vals), row[prefix + '_median_compute_ms_sd'])
        for metric in ['peak_allocated_mib', 'peak_reserved_mib', 'parameters']:
            assert close(statistics.mean(r[metric] for r in reps), row[f'{prefix}_{metric}_mean'])
    ratios = [m['median_compute_ms'] / p['median_compute_ms'] for m, p in zip(by_arm['mac'], by_arm['mlp'])]
    assert close(statistics.mean(ratios), row['paired_mac_over_mlp_ratio_mean'])
    assert close(statistics.stdev(ratios), row['paired_mac_over_mlp_ratio_sd'])
    path = 'Training step' if mode == 'train' else 'Evaluation batch'
    line = f"| {names[ds]} | {path} | {row['mlp_median_compute_ms_mean']:.3f} ± {row['mlp_median_compute_ms_sd']:.3f} | {row['mac_median_compute_ms_mean']:.3f} ± {row['mac_median_compute_ms_sd']:.3f} | {row['paired_mac_over_mlp_ratio_mean']:.2f} ± {row['paired_mac_over_mlp_ratio_sd']:.2f} |"
    assert line in text
    cells = [f'{row[k]:.1f}' for k in ['mlp_peak_allocated_mib_mean', 'mac_peak_allocated_mib_mean', 'mlp_peak_reserved_mib_mean', 'mac_peak_reserved_mib_mean']]
    assert '| ' + ' | '.join([names[ds], 'Training' if mode == 'train' else 'Evaluation'] + cells) + ' |' in text
checks['timing_and_memory_tables_recomputed_from_24_repetitions'] = True
for dataset in contract['datasets']:
    p = dataset['inherited_data_identity']['populations']
    stem = f"| {names[dataset['dataset_id']]} | {p['train']['target_count']:,} | {p['validation']['target_count']:,} | {dataset['loader']['max_seq_len']} |"
    assert stem in text
checks['dataset_table_matches_frozen_contract'] = True

refs = json.loads((OUT / 'references.json').read_text())['records']
bib = (OUT / 'references.bib').read_text()
checks['nine_primary_reference_records_and_bib_entries'] = (
    len(refs) == 9 and all('@' in bib and '{' + r['key'] + ',' in bib and r['title'] in text for r in refs))
checks['two_existing_figures_and_six_tables'] = len(re.findall(r'^!\[', text, re.M)) == 2 and re.findall(r'\*\*Table (\d)\.', text) == list('123456')
checks['balanced_display_math'] = text.count('$$') == 18
checks['scope_and_counterexamples_present'] = all(s in text for s in [
    'same immediate predecessor', 'divisor is always eight', 'not a fitted probability density',
    'not claimed to be a full joint marked-event log-likelihood',
    '3.98 times', 'not pure deployment inference', 'retrospective', 'not a significance test',
    'not establish causation', 'not presented as final results', 'no new training',
])

inputs = [
    method_root + x for x in ['method_draft.md', 'code_equation_map.md', 'verification.json', 'sources.json', 'architecture.png', 'architecture.pdf', 'architecture.svg']
] + [eff_root + x for x in ['report.md', 'analysis.json', 'verification.json', 'repetitions.csv', 'per_batch.csv', 'paired_efficiency.png', 'paired_efficiency.pdf', 'paired_efficiency.svg']]
inputs += [cmp_root + x for x in ['comparison.json', 'report.md', 'audit_completion.json']]
inputs += [contract_path,
    'reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv',
    'reports/titantpp_core_ablation_execution_20260928_v1/verification.json',
    'reports/titantpp_completed_5080_audit_20261001_v1/report.md',
    'reports/titantpp_completed_5080_audit_20261001_v1/verification.json',
    'reports/titantpp_data_characteristics_20260928_v1/README.md',
    'search_artifacts/titantpp_additional_tpp_20260930_v1/README.md',
    'search_artifacts/titantpp_additional_tpp_20260930_v1/launch_layout_fix_v3/upstream_receipt.json']
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
    'scope': 'Document integration; original validation and matched-cost evidence reused',
    'inputs_sha256': {p: sha(ROOT / p) for p in inputs}, 'method_frozen_source_checks': source_checks}
(OUT / 'source_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
# Create a pending receipt so all document links can be checked before finalization.
(OUT / 'verification.json').write_text('{"status":"checking"}\n')
missing = []
for doc in [MAIN, OUT / 'README.md', OUT / 'claim_evidence_and_writing_notes.md']:
    for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', doc.read_text()):
        if link.startswith(('http://', 'https://', '#')):
            continue
        target = Path(link) if link.startswith('/') else doc.parent / link
        if not target.exists():
            missing.append({'document': str(doc.relative_to(ROOT)), 'link': link})
checks['all_local_links_resolve'] = not missing
result = {
    'status': 'passed' if all(checks.values()) else 'failed',
    'verified_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks,
    'missing_links': missing, 'manuscript_sha256': sha(MAIN),
    'word_count_whitespace_including_tables_references': len(text.split()),
    'source_manifest_sha256': sha(OUT / 'source_manifest.json'),
    'existing_figures_visually_reviewed': True,
    'existing_cpu_equation_checks_reused_not_rerun': True,
    'new_model_or_checkpoint_execution': False,
    'new_remote_observation_or_control': False,
    'held_out_or_mixed_results_read': False,
    'new_training_gpu_measurement_scheduler_commit_push_or_publication': False,
    'scientific_source_or_frozen_contract_edited': False,
    'not_claimed': ['independent_test_validation', 'exhaustive_novelty_review', 'final_four_dataset_results', 'submission_ready'],
}
(OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'checks': checks, 'missing_links': missing}, ensure_ascii=False, indent=2))
assert result['status'] == 'passed'
