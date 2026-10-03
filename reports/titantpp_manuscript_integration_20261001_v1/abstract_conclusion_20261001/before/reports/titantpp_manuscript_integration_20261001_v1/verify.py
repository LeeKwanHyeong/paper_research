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
inst_revision_root = OUT / 'instacart_final_integration_20261001'
inst_revision = json.loads((inst_revision_root / 'changes.json').read_text())
pre_instacart_text = text
for row in reversed(inst_revision['manuscript_replacements']):
    assert pre_instacart_text.count(row['after']) == 1
    pre_instacart_text = pre_instacart_text.replace(row['after'], row['before'], 1)
checks['instacart_revision_only_recorded_manuscript_edits'] = (
    pre_instacart_text == (inst_revision_root / 'before' / MAIN.relative_to(ROOT)).read_text())
checks['instacart_revision_original_files_preserved'] = all(
    sha(inst_revision_root / 'before' / rel) == expected
    for rel, expected in json.loads((inst_revision_root / 'before_sha256.json').read_text()).items())
audit_root = OUT / 'reference_doublecheck_20261001'
audit = json.loads((audit_root / 'audit.json').read_text())
audit_changes = json.loads((audit_root / 'changes.json').read_text())
links_root = OUT / 'official_links_20261001'
link_changes = json.loads((links_root / 'changes.json').read_text())['reference_changes']
before_audit = (audit_root / 'before/paper' / MAIN.name).read_text()
audit_reverted_text = pre_instacart_text
for row in link_changes:
    audit_reverted_text = audit_reverted_text.replace(row['after'], row['before'])
for prior_url, revised_url in audit_changes['body_url_replacements'].items():
    audit_reverted_text = audit_reverted_text.replace('](' + revised_url + ')', '](' + prior_url + ')')
method_root = 'reports/titantpp_method_efficiency_20260930_v1/'
eff_root = 'reports/titantpp_efficiency_5080_20260930_v1/'
cmp_root = 'reports/titantpp_completed_external_comparison_20261001_v1/'
mv = read(method_root + 'verification.json')
eff = read(eff_root + 'analysis.json')
comp = read(cmp_root + 'comparison.json')
inst_root = 'reports/titantpp_instacart_final_analysis_20261001_v1/'
inst = read(inst_root + 'analysis.json')
inst_verification = read(inst_root + 'verification.json')
inst_audit = read(inst['new_5090_audit']['path'])
checks['final_instacart_analysis_and_source_audit_verified'] = (
    inst_verification['status'] == 'passed'
    and inst_verification['new_5090_conditions_cpu_audited'] == 6
    and inst_verification['checkpoints_cpu_audited'] == 12
    and inst_audit['status'] == 'passed'
    and sha(ROOT / inst['new_5090_audit']['path']) == inst['new_5090_audit']['sha256']
    and sha(ROOT / (inst_root + 'analysis.json')) == inst_verification['analysis_sha256'] == inst_revision['source_analysis_sha256']
    and sha(ROOT / (inst_root + 'report.md')) == inst_verification['report_sha256']
    and sha(ROOT / (inst_root + 'source_manifest.json')) == inst_verification['source_manifest_sha256']
    and sha(ROOT / (inst_root + 'conditions.csv')) == inst_verification['conditions_csv_sha256']
    and sha(ROOT / (inst_root + 'strata.csv')) == inst_verification['strata_csv_sha256'])
# Overlay the completed Instacart panel locally; retain the historical report.
is_old_instacart = lambda row: row['dataset'] == inst['dataset'] and row['model'] in inst['models']
comp['rows'] = [r for r in comp['rows'] if not is_old_instacart(r)] + [
    dict(r, dataset=inst['dataset']) for r in inst['rows'] if r['endpoint'] == 'selected']
comp['three_seed'] = [r for r in comp['three_seed'] if not is_old_instacart(r)] + [
    dict(r, dataset=inst['dataset']) for r in inst['three_seed'] if r['endpoint'] == 'selected']
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
checks['table4_four_datasets_28_complete_groups_recomputed'] = completed_groups == 28 and len(re.findall(r'^\|', table4, re.M)) == 30
checks['table4_all_84_seed_runs_complete_no_pending_or_partial_means'] = (
    not pending_groups and 'Pending' not in text and '84 runs' in text)
checks['table4_one_representative_no_exploratory_variant'] = 'active normalization' not in table4
table_data = json.loads((OUT / 'tables.json').read_text())['common_external_comparison']
checks['machine_readable_table_matches_all_28_groups'] = (
    len(table_data) == 28
    and {(r['dataset'], r['model']) for r in table_data}
        == {(d, m) for d in common_names for m in model_names}
    and all(r['status'] == 'complete_three_seed' and r['completed_seeds'] == [42, 52, 62]
            and all(close(r[m][stat], agg[r['dataset'], r['model']][m][stat])
                    for m in metrics for stat in ['mean', 'sample_sd']) for r in table_data))
inst_pairs = {r['comparator']: r for r in inst['paired']}
inst_groups = {r['model']: r for r in inst['three_seed'] if r['endpoint'] == 'selected'}
for model in external:
    left = {r['seed']: r for r in inst['rows'] if r['model'] == 'titantpp_history_mlp' and r['endpoint'] == 'selected'}
    right = {r['seed']: r for r in inst['rows'] if r['model'] == model and r['endpoint'] == 'selected'}
    for metric in metrics:
        assert sum(left[s][metric] < right[s][metric] for s in [42, 52, 62]) == inst_pairs[model][metric]['plain_wins']
        reduction = 100 * (1 - inst_groups['titantpp_history_mlp'][metric]['mean'] / inst_groups[model][metric]['mean'])
        assert close(reduction, inst_pairs[model][metric]['plain_reduction_percent'])
checks['instacart_paired_wins_and_percentage_claims_recomputed'] = (
    all(inst_pairs[m]['qty_mae']['plain_wins'] == 2 and inst_pairs[m]['qty_rmse']['plain_wins'] == 0
        for m in ['s2p2_matched_head', 'attnhp_matched_head'])
    and inst_pairs['s2p2_matched_head']['time_nll']['plain_wins'] == 0
    and all(s in text for s in ['0.18% and 0.11%', '0.52% and 0.31%', 'lower RMSE in all three seeds']))
quantity = {(r['model'], r['bin']): r for r in inst['selected_strata'] if r['partition'] == 'quantity'}
mlp, s2p2 = 'titantpp_history_mlp', 's2p2_matched_head'
low_fraction = sum(quantity[mlp, b]['count'] for b in [0, 1]) / inst['validation_targets_per_seed'] * 100
decomp = next(r for r in inst['squared_error_decomposition'] if r['partition'] == 'quantity' and r['comparator'] == s2p2)
checks['instacart_quantity_strata_interpretation_verified'] = (
    f'{low_fraction:.2f}%' in text
    and all(quantity[mlp, b][m]['mean'] < quantity[s2p2, b][m]['mean'] for b in [0, 1] for m in ['qty_mae', 'qty_rmse'])
    and all(quantity[mlp, b][m]['mean'] > quantity[s2p2, b][m]['mean'] for b in [2, 3, 4] for m in ['qty_mae', 'qty_rmse'])
    and all(f"{quantity[model, 4]['qty_rmse']['mean']:.4f}" in text for model in [mlp, s2p2])
    and sum(r['plain_minus_comparator_mse_contribution'] for r in decomp['cells']) > 0)

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
body = text.split('## References', 1)[0]
bib_keys = re.findall(r'^@\w+\{([^,]+),', bib, re.M)
body_urls = re.findall(r'\]\((https?://[^)]+)\)', body)
ref_keys = [r['key'] for r in refs]
checks['32_unique_references_bib_and_body_citations_match'] = (
    len(refs) == len(set(ref_keys)) == 32 and bib_keys == ref_keys
    and len(set(r['title'].casefold() for r in refs)) == 32
    and all(r['title'] in text and r['url'] in body_urls for r in refs)
    and set(body_urls) == {r['url'] for r in refs})
checks['reference_authors_years_titles_present_in_bib'] = all(
    'author = {' + r.get('bibtex_author', ' and '.join(r['authors'])) + '}' in bib
    and 'title = {{' + r['title'] + '}}' in bib
    and 'year = {' + str(r['year']) + '}' in bib for r in refs)
checks['bibtex_braces_balanced'] = bib.count('{') == bib.count('}')
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
    '0.16% higher' in results and '0.52% and 0.31%' in results,
    '3.98 times' in cost and 'target loss' in cost,
    'not a full joint marked-event likelihood' in limits,
    'not a significance test' in limits,
    'do not establish causation' in limits,
    'retrospective cohort selection' in limits,
    'higher RMSE in every paired seed' in limits,
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
before_references = (OUT / 'before_reference_expansion' / MAIN.name).read_text()
checks['historical_prose_revision_preserved_reference_section'] = (
    before_references.split('## References', 1)[1].split('## Internal', 1)[0]
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
checks['historical_layout_revision_preserved_efficiency_section_exactly'] = (
    before_references.split('## 6.', 1)[1].split('## 7.', 1)[0]
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
checks['historical_figure_revision_changed_only_first_figure_and_caption'] = (
    without_figure1(before_references) == without_figure1(before_figure))

# The current citation revision is checked separately from the historical edits.
revision = read('reports/titantpp_manuscript_integration_20261001_v1/reference_expansion_changes.json')
restored = audit_reverted_text
for old_phrase, new_phrase in revision['literal_replacements'].items():
    assert restored.count(new_phrase) == 1, new_phrase
    restored = restored.replace(new_phrase, old_phrase, 1)
section = lambda value, start, end: value.split(start, 1)[1].split(end, 1)[0]
checks['reference_revision_method_protocol_cost_only_recorded_edits'] = all(
    section(restored, start, end) == section(before_references, start, end)
    for start, end in [('## 3. Method', '## 4.'), ('## 4. Evaluation protocol', '## 5.'), ('## 6.', '## 7.')])
checks['reference_revision_preserves_results_limits_notation_contributions'] = all(
    section(pre_instacart_text, start, end) == section(before_references, start, end)
    for start, end in [('## 5.', '## 6.'), ('## 7.', '## Appendix'), ('## Appendix', '## References'), ('Our contributions are:', '## 2.')])
checks['reference_revision_preserves_all_equations_tables_and_figures'] = (
    blocks(pre_instacart_text) == blocks(before_references)
    and re.findall(r'^\|.*$', pre_instacart_text, re.M) == re.findall(r'^\|.*$', before_references, re.M)
    and re.findall(r'^!\[.*$', pre_instacart_text, re.M) == re.findall(r'^!\[.*$', before_references, re.M))
old_refs = json.loads((OUT / 'before_reference_expansion/references.json').read_text())['records']
current_refs = {r['key']: r for r in refs}
preaudit_refs = json.loads((audit_root / 'before' / (OUT / 'references.json').relative_to(ROOT)).read_text())['records']
preaudit_by_key = {r['key']: r for r in preaudit_refs}
checks['historical_expansion_preserved_original_nine_metadata'] = all(
    all(preaudit_by_key[r['key']].get(k) == v for k,v in r.items()) for r in old_refs)
expected_refs = json.loads(json.dumps(preaudit_refs))
expected_by_key = {r['key']: r for r in expected_refs}
for row in audit_changes['reference_changes']:
    for field, values in row['fields'].items():
        assert expected_by_key[row['key']].get(field) == values['before']
        expected_by_key[row['key']][field] = values['after']
for row in link_changes:
    assert expected_by_key[row['key']]['url'] == row['before']
    expected_by_key[row['key']]['url'] = row['after']
checks['bibliography_audit_only_documented_metadata_edits'] = expected_refs == refs
before_links = links_root / 'before'
link_reverted_text, link_reverted_bib = pre_instacart_text, bib
for row in link_changes:
    assert text.count(row['after']) == row['manuscript_occurrences']
    assert bib.count(row['after']) == row['bibtex_occurrences']
    link_reverted_text = link_reverted_text.replace(row['after'], row['before'])
    link_reverted_bib = link_reverted_bib.replace(row['after'], row['before'])
checks['official_links_only_five_reviewed_urls_changed_in_manuscript_and_bib'] = (
    len(link_changes) == 5
    and link_reverted_text == (before_links / MAIN.relative_to(ROOT)).read_text()
    and link_reverted_bib == (before_links / (OUT / 'references.bib').relative_to(ROOT)).read_text())
before_link_refs = json.loads((before_links / (OUT / 'references.json').relative_to(ROOT)).read_text())['records']
reverted_refs = json.loads(json.dumps(refs))
reverted_by_key = {r['key']: r for r in reverted_refs}
for row in link_changes:
    assert reverted_by_key[row['key']]['url'] == row['after']
    reverted_by_key[row['key']]['url'] = row['before']
checks['official_links_preserve_all_other_reference_metadata'] = reverted_refs == before_link_refs
checks['official_links_original_files_preserved'] = all(
    sha(before_links / rel) == expected
    for rel, expected in json.loads((links_root / 'before_sha256.json').read_text()).items())
checks['bibliography_audit_preserves_body_prose_except_reviewed_links'] = (
    audit_reverted_text.split('## References', 1)[0] == before_audit.split('## References', 1)[0]
    and pre_instacart_text.split('## Internal', 1)[1] == before_audit.split('## Internal', 1)[1])
checks['bibliography_audit_all_32_have_primary_evidence_and_claim_review'] = (
    [r['key'] for r in audit['records']] == ref_keys
    and all(r['identity_verified'] and r['authors_and_order_verified']
            and r['publication_year_verified'] and r['evidence_urls']
            and r['citation_claim_check'] == 'consistent_with_current_passage' for r in audit['records']))
checks['bibliography_audit_preserves_all_before_files'] = all(
    sha(audit_root / 'before' / rel) == expected
    for rel, expected in json.loads((audit_root / 'before_sha256.json').read_text()).items())
checks['bibliography_audit_preserves_titles_authors_years_and_order'] = (
    [{k: r[k] for k in ['key', 'title', 'authors', 'year']} for r in refs]
    == [{k: r[k] for k in ['key', 'title', 'authors', 'year']} for r in preaudit_refs])
checks['bibliography_audit_compound_family_name_explicit'] = 'De Laroussilhe, Quentin' in bib
additions = json.loads((OUT / 'reference_additions.json').read_text())['records']
checks['23_added_references_have_primary_evidence_and_claim_locations'] = (
    len(additions) == 23 and all(r.get('verified_claim') and r.get('verification_extent') and r.get('used_in') and r.get('url') for r in additions)
    and set(current_refs) == {r['key'] for r in old_refs + additions})

inputs = [
    method_root + x for x in ['method_draft.md', 'code_equation_map.md', 'verification.json', 'sources.json', 'architecture.png', 'architecture.pdf', 'architecture.svg']
] + [eff_root + x for x in ['report.md', 'analysis.json', 'verification.json', 'repetitions.csv', 'per_batch.csv', 'paired_efficiency.png', 'paired_efficiency.pdf', 'paired_efficiency.svg']]
inputs += [cmp_root + x for x in ['comparison.json', 'report.md', 'audit_completion.json']]
inputs += [inst_root + x for x in ['analysis.json', 'report.md', 'verification.json', 'source_manifest.json', 'conditions.csv', 'strata.csv']]
inputs += [inst['new_5090_audit']['path']]
inputs += [str((inst_revision_root / name).relative_to(ROOT)) for name in ['changes.json', 'before_sha256.json', 'report.md']]
inputs += [str((OUT / 'tables.json').relative_to(ROOT))]
inputs += [figure_root + x for x in ['draw.py', 'architecture.png', 'architecture.svg', 'architecture.pdf', 'verification.json']]
inputs += ['reports/titantpp_manuscript_integration_20261001_v1/' + x for x in
           ['references.json', 'references.bib', 'reference_additions.json', 'reference_expansion_changes.json', 'reference_expansion_review.md']]
inputs += [str((audit_root / name).relative_to(ROOT)) for name in
           ['audit.json', 'changes.json', 'report.md', 'attachment_identity.json', 'before_sha256.json',
            'crossref_metadata_verified_tls.json', 's2p2_crossref.json']]
inputs += [str((links_root / name).relative_to(ROOT)) for name in
           ['changes.json', 'before_sha256.json', 'report.md']]
inputs += [str((audit_root / 'gemini_feedback_review.md').relative_to(ROOT))]
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
for doc in [MAIN, OUT / 'README.md', OUT / 'claim_evidence_and_writing_notes.md',
            OUT / 'reference_expansion_review.md', OUT / 'reference_coverage_review.md', audit_root / 'report.md',
            links_root / 'report.md', inst_revision_root / 'report.md']:
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
    'four_dataset_validation_table_complete': True,
    'not_claimed': ['independent_test_validation', 'exhaustive_novelty_review', 'submission_ready'],
}
(OUT / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'checks': checks, 'missing_links': missing}, ensure_ascii=False, indent=2))
assert result['status'] == 'passed'
