"""Check this draft's tables and source structure without training or inference.

These checks are NOT a TeX compilation, PDF layout review, or a new binary audit.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics as stats
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DRAFT = ROOT / 'paper/titantpp_pakdd_2027_draft'
TEX = (DRAFT / 'main.tex').read_text()
SOURCES = [
    'paper/titantpp_history_mlp_manuscript_20261001_v1.md',
    'reports/titantpp_manuscript_integration_20261001_v1/references.bib',
    'reports/titantpp_completed_external_comparison_20261001_v1/comparison.json',
    'reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv',
    'reports/titantpp_instacart_final_analysis_20261001_v1/conditions.csv',
    'reports/titantpp_instacart_final_analysis_20261001_v1/verification.json',
    'reports/titantpp_efficiency_5080_20260930_v1/analysis.json',
    'reports/titantpp_pakdd_extension_preparation_20261001_v1/nonlearned_comparison.csv',
    'search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/hourly_monitor/20261001T221422059513Z/analysis.json',
]
DATA = {'Taxi': 'yellow_trip_hourly', 'Intermittent': 'intermittent_frozen_5000',
        'Instacart': 'insta_market_basket', 'RAF': 'raf_spare_parts'}
MODELS = {'TitanTPP': 'titantpp_history_mlp', 'RMTPP': 'rmtpp', 'THP': 'thp',
          'NHP': 'nhp', 'SAHP': 'sahp', 'S2P2': 's2p2_matched_head',
          'AttNHP': 'attnhp_matched_head', 'B': 'titantpp'}
METRICS = ('qty_mae', 'qty_rmse', 'time_nll')
checks = {}


def read_json(rel):
    return json.loads((ROOT / rel).read_text())


def read_csv(rel):
    with (ROOT / rel).open() as f:
        return list(csv.DictReader(f))


def check(name, condition):
    checks[name] = bool(condition)
    assert condition, name


def table(label):
    for block in re.findall(r'\\begin\{table\}.*?\\end\{table\}', TEX, re.S):
        if '\\label{' + label + '}' in block:
            return block
    raise AssertionError(label)


def number_tokens(line):
    return re.findall(r'\d+\.\d+', line)


def grouped_rows(label):
    dataset = None
    result = {}
    for line in table(label).splitlines():
        match = re.search(r'\\textit\{([^}]+)\}', line)
        if match:
            dataset = DATA[match[1]]
        if dataset and '&' in line:
            model = line.split('&')[0].strip()
            result[(dataset, MODELS[model])] = number_tokens(line)
    return result


def means_sd(rows, precision):
    result = []
    for key in METRICS:
        values = [float(row[key]) for row in rows]
        result.extend([f'{stats.mean(values):.{precision}f}',
                       f'{stats.stdev(values):.{precision}f}'])
    return result


comparison = read_json(SOURCES[2])
machine = {(r['dataset'], r['model']): [f'{r[m][k]:.4f}' for m in METRICS
            for k in ('mean', 'sample_sd')] for r in comparison['three_seed']}
inst = [r for r in read_csv(SOURCES[4]) if r['endpoint'] == 'selected']
for model in set(r['model'] for r in inst):
    group = [r for r in inst if r['model'] == model]
    check('Instacart three seeds: ' + model, sorted(int(r['seed']) for r in group) == [42, 52, 62])
    machine[(DATA['Instacart'], model)] = means_sd(group, 4)
main = grouped_rows('tab:main')
check('main panel has 28 model/dataset groups', len(main) == 28)
check('main panel matches machine records at reported precision', all(v == machine[k] for k, v in main.items()))
check('Instacart source audit passed', read_json(SOURCES[5])['status'] == 'passed')

core = [r for r in read_csv(SOURCES[3]) if r['endpoint'] == 'selected']
ablation = grouped_rows('tab:ablation')
check('six correction control rows match selected validation CSV', len(ablation) == 6 and all(
    values == means_sd([r for r in core if (r['dataset'], r['arm']) == key], 4)
    for key, values in ablation.items()))

naive = read_csv(SOURCES[7])
expected = {}
for r in naive:
    label = 'Last quantity' if r['model'] == 'last_observed_quantity' else 'Window mean'
    expected[(r['dataset'], label)] = [f"{float(r[m]):.4f}" for m in METRICS[:2]]
    expected[(r['dataset'], 'TitanTPP')] = [f"{float(r['mlp_3seed_mean_' + m]):.4f}" for m in ('mae', 'rmse')]
observed = {}
dataset = None
for line in table('tab:naive').splitlines():
    if not number_tokens(line) or '&' not in line:
        continue
    cells = line.split('&')
    if cells[0].strip():
        dataset = DATA[cells[0].strip()]
    observed[(dataset, cells[1].strip())] = number_tokens(line)
check('all 12 nonlearned comparison rows match source CSV', observed == expected)

eff = read_json(SOURCES[6])
dataset = None
cost_rows = {}
for line in table('tab:cost').splitlines():
    if not number_tokens(line) or '&' not in line:
        continue
    cells = line.split('&')
    if cells[0].strip():
        dataset = DATA[cells[0].strip()]
    cost_rows[(dataset, 'train' if cells[1].strip() == 'Train' else 'eval')] = number_tokens(line)
for r in eff['comparison']:
    expected = [f'{r[k]:.3f}' for k in ('mlp_median_compute_ms_mean', 'mlp_median_compute_ms_sd',
                'mac_median_compute_ms_mean', 'mac_median_compute_ms_sd')]
    expected += [f'{r[k]:.2f}' for k in ('paired_mac_over_mlp_ratio_mean', 'paired_mac_over_mlp_ratio_sd')]
    check('matched cost row: ' + r['dataset'] + '/' + r['mode'], cost_rows[(r['dataset'], r['mode'])] == expected)
check('existing efficiency verification passed', eff['verification']['status'] == 'passed')

extension = read_json(SOURCES[8])
complete = [r for host in extension['hosts'].values() for r in host['rows']
            if r['state'] == 'complete_small_records_verified']
check('appendix snapshot has 11 completed records', len(complete) == 11)
EXT_MODELS = {'Original MLP': 'titantpp_history_mlp',
              'Current-only': 'titantpp_current_only_param_matched',
              'All-available': 'titantpp_all_available_history_mlp',
              'Deep Renewal': 'deep_renewal_event_native_nb'}
for line in table('tab:extension').splitlines():
    if '&' not in line or not number_tokens(line):
        continue
    name = line.split('&')[0].strip()
    arm = EXT_MODELS[name]
    rows = [r['metrics'] for r in complete if r['job']['dataset'] == DATA['Taxi'] and r['job']['arm'] == arm]
    if name == 'Original MLP':
        rows = [r for r in core if r['dataset'] == DATA['Taxi'] and r['arm'] == arm]
    check('appendix Taxi row: ' + name, len(rows) == 3 and number_tokens(line) == means_sd(rows, 3))

for dataset in (DATA['Intermittent'], DATA['Instacart']):
    new = [r for r in complete if r['job']['dataset'] == dataset]
    check('appendix single-seed identity: ' + dataset, len(new) == 1 and new[0]['job']['seed'] == 42)
    old = next(r for r in core if r['dataset'] == dataset and r['arm'] == 'titantpp_history_mlp' and r['seed'] == '42')
    for metrics in (new[0]['metrics'], old):
        check('appendix metric triple: ' + dataset + '/' + str(metrics['qty_mae']),
              '/'.join(f'{float(metrics[m]):.5f}' for m in METRICS) in TEX)

# Structural checks are deliberately independent of compilation.
source = re.sub(r'(?<!\\)%[^\n]*', '', TEX)
depth = 0
for token in re.finditer(r'\\(?:[A-Za-z]+|.)|([{}])', source):
    if token[1]:
        depth += 1 if token[1] == '{' else -1
        assert depth >= 0, 'negative brace nesting'
check('balanced source braces', depth == 0)
stack = []
for mode, env in re.findall(r'\\(begin|end)\{([^}]+)\}', source):
    if mode == 'begin':
        stack.append(env)
    else:
        assert stack and stack.pop() == env, env
check('balanced LaTeX environments', not stack)
labels = re.findall(r'\\label\{([^}]+)\}', source)
refs = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', source)
check('all cross-references resolve and labels are unique', set(refs) <= set(labels) and len(labels) == len(set(labels)))
cites = set(','.join(re.findall(r'\\cite\{([^}]+)\}', source)).split(','))
bibkeys = re.findall(r'\\bibitem\{([^}]+)\}', source)
check('32 bibliography entries all cited with no missing keys', len(bibkeys) == 32 and len(set(bibkeys)) == 32 and cites == set(bibkeys))
check('source has anonymous author/institute and no local identifiers',
      all(v in source for v in ('\\author{Anonymous Authors}', '\\institute{Anonymous Institution}'))
      and not any(v in source.lower() for v in ('/users/', '/home/', 'igwanhyeong', 'leekwanhyeong', 'notion.com')))
check('nine tables and four embedded figures', source.count('\\begin{table}') == 9 and source.count('\\begin{figure}') == 4)
check('no external figure or bibliography file dependency', not re.search(r'\\(?:includegraphics|input|include|bibliography)\{', source))
check('no unfinished content placeholders', not re.search(r'\b(?:TODO|TBD|FIXME)\b|BIBLIOGRAPHY_INSERT', source))
check('copied bibliography unchanged', (DRAFT / 'references.bib').read_bytes() == (ROOT / SOURCES[1]).read_bytes())
receipt = json.loads((OUT / 'template_receipt.json').read_text())
for name, sha in receipt['files'].items():
    check('official template unchanged: ' + name, hashlib.sha256((DRAFT / name).read_bytes()).hexdigest() == sha)
check('core frozen CSV identity', hashlib.sha256((ROOT / SOURCES[3]).read_bytes()).hexdigest() == '3501b98f0f3c5d250c6ddb7f8309b3455ec1bb908f0844928ec504d71c826765')

compilation_path = ROOT / 'reports/titantpp_dataset_appendix_20261002_v1/compilation_receipt.json'
compilation = json.loads(compilation_path.read_text())
check('current source compiled successfully in native editor',
      compilation['status'] == 'success' and compilation['source_sha256'] == hashlib.sha256((DRAFT / 'main.tex').read_bytes()).hexdigest())

result = {
    'status': 'source_and_numeric_checks_passed_native_compilation_passed',
    'checked_kst': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
    'checks': checks,
    'evaluation_scope': 'development_validation_only',
    'heldout_predictions_or_metrics_read': False,
    'new_gpu_training_or_replay': False,
    'main_selected_run_count': 84,
    'extension_completed_record_count_at_cutoff': 11,
    'extension_cutoff_kst': '2026-10-02T07:14:23+09:00',
    'extension_binary_cpu_audit': 'pending_at_cutoff',
    'compilation': {'status': 'passed', 'tool': 'compile_latex_document',
                    'calls_this_edit': len(compilation['attempts']),
                    'receipt': str(compilation_path.relative_to(ROOT)),
                    'native_class_version': compilation['native_class_version'],
                    'packaged_class_version': compilation['packaged_class_version']},
    'pdf_exported': False,
    'visual_layout_review': 'full_document_not_reviewed; new_standalone_figures_reviewed_separately',
    'page_count': None,
    'source_sha256': hashlib.sha256((DRAFT / 'main.tex').read_bytes()).hexdigest(),
    'evidence_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
}
(OUT / 'verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': result['status'], 'checks': len(checks), 'pdf_exported': False}))
