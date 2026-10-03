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
    assert f"{ratios[0]:.2f}%/{ratios[1]:.2f}% on {names[dataset]}" in text
checks['quantity_reduction_prose_and_strongest_external_verified'] = True
checks['same_seed_both_quantity_wins_36_of_36'] = len(paired) == 36 and all(paired)

table4 = text.split('**Table 4.**', 1)[1].split('\n\nOn Taxi', 1)[0]
common_names = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent',
                'raf_spare_parts': 'RAF', 'insta_market_basket': 'Instacart'}
model_names = {'titantpp_history_mlp': 'TitanTPP', 'rmtpp': 'RMTPP', 'thp': 'THP',
               'nhp': 'NHP', 'sahp': 'SAHP', 's2p2_matched_head': 'S2P2',
               'attnhp_matched_head': 'AttNHP'}
completed_groups, pending_groups = 0, []
for dataset, label in common_names.items():
    for model, model_label in model_names.items():
        rows = [r for r in comp['rows'] if r['dataset'] == dataset and r['model'] == model]
        if sorted(r['seed'] for r in rows) == [42, 52, 62]:
            cells = []
            for metric in metrics:
                mean, sd = statistics.mean(r[metric] for r in rows), statistics.stdev(r[metric] for r in rows)
                assert close(mean, agg[dataset, model][metric]['mean'])
                assert close(sd, agg[dataset, model][metric]['sample_sd'])
                cells.append(f'{mean:.4f} ± {sd:.4f}')
            completed_groups += 1
        else:
            cells = ['Pending'] * 3
            pending_groups.append((dataset, model))
        line = '| ' + ' | '.join([label, model_label] + cells) + ' |'
        assert line in table4
checks['table4_four_datasets_26_complete_groups_recomputed'] = completed_groups == 26 and len(re.findall(r'^\|', table4, re.M)) == 30
checks['table4_two_incomplete_groups_explicit_no_partial_means'] = pending_groups == [
    ('insta_market_basket', 's2p2_matched_head'), ('insta_market_basket', 'attnhp_matched_head')]
checks['table4_one_representative_no_exploratory_variant'] = 'active normalization' not in table4

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
checks['two_figures_and_seven_tables'] = len(re.findall(r'^!\[', text, re.M)) == 2 and re.findall(r'\*\*Table (\d)\.', text) == list('1234567')
checks['balanced_display_math'] = text.count('$$') == 18
method = text.split('## 3. Method', 1)[1].split('## 4.', 1)[0]
protocol = text.split('## 4. Evaluation protocol', 1)[1].split('## 5.', 1)[0]
results = text.split('## 5.', 1)[1].split('## 6.', 1)[0]
cost = text.split('## 6.', 1)[1].split('## 7.', 1)[0]
limits = text.split('## 7. Limitations', 1)[1].split('## Appendix', 1)[0]
checks['scope_and_counterexamples_present'] = all([
    'same immediate predecessor' in method,
    'divisor is always eight' in method,
    'nonnegative quantity point estimate' in method,
    'validation, which also informed architecture development' in protocol,
    '**common-head adaptations**' in protocol,
    'same RMSE-selected checkpoint' in protocol,
    'Mean time NLL increases on all three datasets' in results,
    '0.16% higher' in results and '0.42% higher' in results,
    '3.98 times' in cost and 'target loss' in cost,
    'not a full joint marked-event likelihood' in limits,
    'not a significance test' in limits,
    'do not establish causation' in limits,
    'retrospective cohort selection' in limits,
    'remaining Instacart comparator results' in limits,
])
previous = (OUT / 'before_prose_revision' / MAIN.name).read_text()
# The old reduction-only Table 4 is now a four-dataset table. Its values are
# checked above against the same source and retained in the comparison prose.
previous_without_old_table4 = previous.split('**Table 4.**', 1)[0] + previous.split('## 6.', 1)[1]
checks['all_other_preexisting_numeric_table_rows_preserved'] = all(
    row in re.findall(r'^\|.*$', text, re.M)
    for row in re.findall(r'^\|.*$', previous_without_old_table4, re.M))
checks['prose_revision_preserves_all_display_equations'] = blocks(text) == blocks(previous)
checks['efficiency_figure_preserved'] = (
    re.findall(r'^!\[.*$', text, re.M)[1] == re.findall(r'^!\[.*$', previous, re.M)[1])
checks['prose_revision_preserves_reference_section'] = (
    text.split('## References', 1)[1].split('## Internal', 1)[0]
    == previous.split('## References', 1)[1].split('## Internal', 1)[0])
checks['intro_contributions_use_direct_declarations'] = (
    'Our contributions are:' in text and 'The contributions supported by the current development evidence are:' not in text)

raf = read('reports/titantpp_raf_execution_20261001_v1/comparison.json')
raf_audit_path = 'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/terminal_audit.json'
raf_audit = read(raf_audit_path)
raf_names = {'titantpp_history_mlp':'TitanTPP', 'titantpp_history_mlp_active_norm':'TitanTPP: active normalization', 'rmtpp':'RMTPP', 'thp':'THP', 'nhp':'NHP', 'sahp':'SAHP', 's2p2_matched_head':'S2P2', 'attnhp_matched_head':'AttNHP'}
for model,label in raf_names.items():
    rows = [r for r in raf['rows'] if r['model'] == model]
    assert sorted(r['seed'] for r in rows) == [42,52,62]
    cells = [f"{statistics.mean(r[k] for r in rows):.4f} ± {statistics.stdev(r[k] for r in rows):.4f}" for k in metrics]
    prefix = '| ' if model == 'titantpp_history_mlp_active_norm' else '| RAF | '
    target = text if model == 'titantpp_history_mlp_active_norm' else table4
    assert prefix+label+' | '+' | '.join(cells)+' |' in target
checks['raf_values_recomputed_from_all_24_selected_rows'] = True
checks['raf_original_cpu_audit_passed'] = raf_audit['status']=='passed' and raf_audit['verified_terminal_checkpoints']==48
checks['raf_dataset_population_and_units'] = '| RAF | 25,779 | 6,690 | 84 | 84 months | 6 months |' in text
checks['raf_not_presented_as_uniform_metric_or_seed_win'] = all(x in text for x in ['two of the three paired seeds', 'RMTPP has the lowest mean MAE', 'NHP has the lowest mean time NLL', 'exploratory variant'])
section52 = text.split('### 5.2 ', 1)[1].split('### 5.3 ', 1)[0]
section53 = text.split('### 5.3 ', 1)[1].split('## 6.', 1)[0]
checks['raf_one_interpretive_paragraph_in_common_comparison'] = (
    len([p for p in section52.split('\n\n') if p.startswith('On RAF spare-parts demand,')]) == 1
    and 'Extension to RAF spare-parts demand' not in text)
checks['normalization_details_in_structural_comparison'] = (
    section53.startswith('Structural alternatives') and '**Table 5.**' in section53
    and '33.8979 ± 0.0804' in section53 and '3.5446 to 3.5631' in section53
    and 'two of the three seeds' in section53 and 'exploratory variant' in section53)
before_layout = (OUT / 'before_raf_repositioning' / MAIN.name).read_text()
checks['layout_revision_preserves_efficiency_section_exactly'] = (
    text.split('## 6.', 1)[1].split('## 7.', 1)[0]
    == before_layout.split('## 6.', 1)[1].split('## 7.', 1)[0])
checks['raf_duration_scale_present_in_method'] = '6 months for RAF' in method
figure_root = 'reports/titantpp_architecture_figure_20261001_v2/'
figure_receipt = read(figure_root + 'verification.json')
checks['hierarchical_figure_outputs_verified'] = (
    figure_receipt['status'] == 'passed'
    and all(sha(ROOT / figure_root / name) == expected for name, expected in figure_receipt['output_sha256'].items()))
checks['hierarchical_figure_caption_matches_method'] = all(s in text for s in [
    figure_root + 'architecture.png', 'Eight independent bottleneck branches',
    'same current–predecessor pair', 'divided by eight', 'ellipsis denotes branches 3–7',
    'With no predecessor, all branches are unavailable', 'matrices initialize to zero'])
before_figure = (OUT / 'before_architecture_revision' / MAIN.name).read_text()
without_figure1 = lambda value: re.sub(r'!\[TitanTPP.*?\n\n### 3\.7', '### 3.7', value, flags=re.S)
checks['figure_revision_changes_only_first_figure_and_caption'] = without_figure1(text) == without_figure1(before_figure)

inputs = [
    method_root + x for x in ['method_draft.md', 'code_equation_map.md', 'verification.json', 'sources.json', 'architecture.png', 'architecture.pdf', 'architecture.svg']
] + [eff_root + x for x in ['report.md', 'analysis.json', 'verification.json', 'repetitions.csv', 'per_batch.csv', 'paired_efficiency.png', 'paired_efficiency.pdf', 'paired_efficiency.svg']]
inputs += [cmp_root + x for x in ['comparison.json', 'report.md', 'audit_completion.json']]
inputs += [figure_root + x for x in ['draw.py', 'architecture.png', 'architecture.svg', 'architecture.pdf', 'verification.json']]
inputs += [raf_audit_path, 'reports/titantpp_raf_execution_20261001_v1/comparison.json', 'reports/titantpp_raf_execution_20261001_v1/report.md', contract_path,
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
    'efficiency_figure_reused_and_new_architecture_visually_reviewed': True,
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
