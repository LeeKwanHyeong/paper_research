"""Verify the manuscript revision against audited validation-only records."""
from pathlib import Path
from collections import Counter
import csv
import datetime
import difflib
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAPER = ROOT / 'paper/titantpp_pakdd_2027_draft'
AUDIT = ROOT / 'reports/titantpp_structural3_final_audit_20261003_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    text = (PAPER / 'main.tex').read_text()
    before = (OUT / 'main.tex.before').read_text()
    comparison = json.loads((AUDIT / 'comparison.json').read_text())
    with (AUDIT / 'condition_registry.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 36
    appendix = text.split(r'\section{Completed Structural Comparisons}', 1)[1].split(
        r'\section{Data Definitions and Training Distributions}', 1)[0]
    datasets = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent',
                'raf_spare_parts': 'RAF', 'insta_market_basket': 'Instacart'}
    models = {'titantpp_history_mlp': 'Original MLP',
              'titantpp_current_only_param_matched': 'Current-only',
              'titantpp_all_available_history_mlp': 'All-available'}
    checks = []
    for group in comparison['groups']:
        assert group['seeds'] == [42, 52, 62] and group['three_seed_complete']
        subset = [r for r in rows if r['dataset'] == group['dataset'] and r['model'] == group['model']]
        assert sorted(int(r['seed']) for r in subset) == [42, 52, 62]
        cells = []
        for metric in ('qty_mae', 'qty_rmse', 'time_nll'):
            values = [float(r[metric]) for r in subset]
            assert math.isclose(statistics.mean(values), group['mean'][metric], rel_tol=1e-12)
            assert math.isclose(statistics.stdev(values), group['sample_sd'][metric], rel_tol=1e-12)
            cells.append(f"${group['mean'][metric]:.4f}\\pm{group['sample_sd'][metric]:.4f}$")
        expected = models[group['model']] + ' & ' + ' & '.join(cells) + r'\\'
        assert appendix.count(expected) == 1, expected
        checks.append({'dataset': datasets[group['dataset']], 'model': models[group['model']],
                       'three_seed_mean_and_sample_sd_match': True})
    assert len(checks) == 12
    old_figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', before, re.S)
    figures = re.findall(r'\\begin\{figure\}.*?\\end\{figure\}', text, re.S)
    assert old_figures == figures
    old_tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', before, re.S)
    tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', text, re.S)
    kept = [table for table in old_tables if r'\label{tab:extension' not in table]
    assert all(table in tables for table in kept)
    marker = r'\section{Data Definitions and Training Distributions}'
    assert text.split(marker, 1)[1] == before.split(marker, 1)[1]
    assert text.split(r'\documentclass', 1)[1].split(r'\begin{abstract}', 1)[0] == before.split(r'\documentclass', 1)[1].split(r'\begin{abstract}', 1)[0]
    labels = re.findall(r'\\label\{([^}]+)\}', text)
    refs = re.findall(r'\\(?:ref|eqref)\{([^}]+)\}', text)
    assert not [key for key, count in Counter(labels).items() if count > 1]
    assert not set(refs) - set(labels)
    for stale in ('extension-partial', '21 structural', 'incomplete groups', 'Completing the structural'):
        assert stale not in text
    sources = []
    for item in json.loads((AUDIT / 'verification.json').read_text())['sources']:
        assert sha(Path(item['path'])) == item['sha256']
        sources.append(item)
    for name in ('main.tex', 'README.md'):
        original = (OUT / (name + '.before')).read_text()
        current = (PAPER / name).read_text()
        (OUT / (name + '.diff')).write_text(''.join(difflib.unified_diff(
            original.splitlines(keepends=True), current.splitlines(keepends=True),
            fromfile=name + '.before', tofile=name)))
    result = {'status': 'passed', 'verified_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'split': 'development_validation_only', 'new_structural_conditions_audited': 3,
              'total_structural_conditions_audited': 24, 'reused_MLP_conditions': 12,
              'table_checks': checks, 'source_audit_receipts': sources,
              'source_sha256': sha(PAPER / 'main.tex'), 'readme_sha256': sha(PAPER / 'README.md'),
              'comparison_sha256': sha(AUDIT / 'comparison.json'),
              'registry_sha256': sha(AUDIT / 'condition_registry.csv'),
              'figures_unchanged': len(figures), 'non_structural_tables_unchanged': len(kept),
              'tables_after_revision': len(tables), 'appendices_B_C_and_bibliography_unchanged': True,
              'author_and_preamble_unchanged': True, 'references_resolve': True,
              'native_compilation': 'success; final source checked with compile_latex_document',
              'separate_pdf_export': False, 'visual_layout_review': 'pending',
              'new_training_or_inference': False, 'new_heldout_reads': False}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'source_sha256', 'figures_unchanged',
          'non_structural_tables_unchanged', 'tables_after_revision')}, indent=2))


if __name__ == '__main__':
    main()
