"""Render the approved exploratory B test analysis without reading predictions.

Usage: python render_b.py --analysis analysis.json --output-dir review
The output directory must be new. This module formats existing statistics; it
does not run inference, bootstrap resampling, or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


DATASETS = {
    'yellow_trip_hourly': 'Taxi',
    'intermittent_frozen_5000': 'Intermittent',
    'insta_market_basket': 'Instacart',
}
MODELS = {
    'titantpp': 'B (TitanTPP)',
    'titantpp_history_mlp': 'MLP representative',
    'rmtpp': 'RMTPP',
    'thp': 'THP',
    'nhp': 'NHP',
    'sahp': 'SAHP',
    's2p2_matched_head': 'S2P2',
    'attnhp_matched_head': 'AttNHP',
}
B_MODEL = 'titantpp'
REPRESENTATIVE = 'titantpp_history_mlp'
COMPARATORS = tuple(model for model in MODELS if model != B_MODEL)
SEEDS = ('42', '52', '62')
METRICS = ('qty_mae', 'qty_rmse', 'time_nll')
METRIC_LABELS = {'qty_mae': 'MAE', 'qty_rmse': 'RMSE', 'time_nll': 'Time NLL'}
PANEL_SCOPE = 'exploratory_b_after_legacy_test_review'
PRIMARY = {
    'yellow_trip_hourly': 'primary_168_hour',
    'intermittent_frozen_5000': 'primary_site',
    'insta_market_basket': 'primary_entity',
}
PANELS = {
    'yellow_trip_hourly': ('primary_168_hour', 'sensitivity_24_hour',
                          'sensitivity_336_hour', 'sensitivity_whole_cell'),
    'intermittent_frozen_5000': ('primary_site', 'sensitivity_series'),
    'insta_market_basket': ('primary_entity',),
}
PANEL_LABELS = {
    'primary_168_hour': 'Shared 168-hour blocks',
    'sensitivity_24_hour': 'Shared 24-hour blocks',
    'sensitivity_336_hour': 'Shared 336-hour blocks',
    'sensitivity_whole_cell': 'Whole spatial cells',
    'primary_site': 'Whole sites',
    'sensitivity_series': 'Whole series',
    'primary_entity': 'Whole users',
}
STATUSES = {
    'computed_conditional': ('산출: 고정 학습 시드에 조건부', 'Computed conditional'),
    'computed_descriptive': ('산출: 단위 수가 적어 기술적으로만 해석', 'Descriptive only'),
    'not_estimable_insufficient_units_or_span': ('미산출: 재표집 단위 또는 기간 부족', 'Not estimable'),
    'not_finalized_invalid_draws': ('미확정: 빈 재표집 발생', 'Not finalized'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value),
            'Missing or nonfinite numeric result')
    return float(value)


def close(left, right):
    return math.isclose(number(left), number(right), rel_tol=1e-9, abs_tol=1e-10)


def valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def validate_interval(value):
    require(isinstance(value, list) and len(value) == 2,
            'Missing confidence interval endpoints')
    require(number(value[0]) <= number(value[1]), 'Reversed confidence interval')


def validate_analysis(analysis):
    """Return analysis after checking the fixed scope and already-computed data."""
    require(isinstance(analysis, dict), 'Analysis must be a JSON object')
    require(analysis.get('status') == 'complete' and analysis.get('split') == 'test',
            'Only a complete test analysis can be rendered')
    require(analysis.get('scope') == PANEL_SCOPE, 'Unexpected exploratory analysis scope')
    for field, expected in (('learned_conditions', 72), ('new_b_conditions', 9),
                            ('legacy_comparison_conditions', 63)):
        require(type(analysis.get(field)) is int and analysis[field] == expected,
                'Incomplete condition scope: ' + field)
    require(analysis.get('exploratory') is True, 'B test addition must remain exploratory')
    require(analysis.get('representative_model') == REPRESENTATIVE and
            analysis.get('representative_replaced') is False,
            'The frozen MLP representative must not be replaced')
    require(analysis.get('prediction_ensemble') is False, 'Unexpected prediction ensemble')
    require(analysis.get('retrained') is False, 'Unexpected retraining in frozen evaluation')
    interpretation = analysis.get('interpretation', '')
    require(isinstance(interpretation, str) and
            all(word in interpretation.lower() for word in ('exploratory', 'legacy', 'test')),
            'Missing exploratory legacy-test interpretation')
    require(isinstance(analysis.get('results'), dict) and
            isinstance(analysis.get('validation'), dict) and
            set(analysis['results']) == set(DATASETS) == set(analysis['validation']),
            'The exact three-dataset panel is required')

    for dataset in DATASETS:
        evidence = analysis['validation'][dataset]
        require(isinstance(evidence, dict), 'Missing population evidence')
        for key in ('exact_common_population', 'frozen_population_identity_verified',
                    'truth_sha256_independently_verified'):
            require(evidence.get(key) is True, 'Unverified population evidence: ' + key)
        for key in ('target_identity_sha256', 'truth_sha256'):
            require(valid_sha(evidence.get(key)), 'Missing population hash: ' + key)
        conditions = evidence.get('conditions', [])
        require(isinstance(conditions, list) and len(conditions) == 24 and
                all(isinstance(row, dict) and type(row.get('seed')) is int for row in conditions),
                'The exact 24 learned conditions per dataset are required')
        expected = {(model, int(seed)) for model in MODELS for seed in SEEDS}
        require({(row.get('model'), row['seed']) for row in conditions} == expected,
                'Missing, duplicate, or unexpected condition evidence')
        count = evidence.get('target_count')
        require(type(count) is int and count > 0 and
                all(type(row.get('prediction_rows')) is int and row['prediction_rows'] == count
                    for row in conditions), 'Incomplete prediction counts')
        panels = analysis['results'][dataset]
        require(isinstance(panels, dict) and set(panels) == set(PANELS[dataset]),
                'Missing primary or sensitivity panel')
        for name, panel in panels.items():
            require(isinstance(panel, dict), 'Missing analysis panel')
            require(panel.get('scope') == PANEL_SCOPE and panel.get('count') == count,
                    'Panel population or exploratory scope mismatch')
            require(panel.get('exploratory') is True and
                    panel.get('difference_reference') == B_MODEL and
                    panel.get('representative_model') == REPRESENTATIVE and
                    panel.get('representative_replaced') is False and
                    panel.get('multiplicity') == 'unadjusted_exploratory_pointwise_95_percentile',
                    'Panel must preserve B exploration and the frozen MLP representative')
            require(panel.get('uncertainty_scope') == 'conditional_on_fixed_trained_seeds' and
                    panel.get('estimate') == 'mean_of_per_seed_metrics_not_prediction_ensemble',
                    'Unexpected uncertainty or aggregation definition')
            require(panel.get('time_nll_definition') ==
                    'frozen_recorded_integer_bin_heteroscedastic_lognormal_NLL',
                    'Unexpected duration NLL definition')
            for field in ('means', 'sample_sd', 'seed_metrics'):
                require(isinstance(panel.get(field), dict) and set(panel[field]) == set(MODELS),
                        'The exact eight learned model results are required: ' + field)
            for model in MODELS:
                seeds = panel['seed_metrics'][model]
                require(set(seeds) == set(SEEDS), 'Missing fixed learned seed metrics')
                for metric in METRICS:
                    values = [number(seeds[seed][metric]) for seed in SEEDS]
                    require(close(panel['means'][model][metric], statistics.mean(values)) and
                            close(panel['sample_sd'][model][metric], statistics.stdev(values)),
                            'Mean/sample SD does not match fixed seed metrics')
            point_differences = panel.get('point_differences')
            require(isinstance(point_differences, dict) and set(point_differences) == set(COMPARATORS),
                    'Incomplete B-minus-comparator point differences')
            for model in COMPARATORS:
                for metric in METRICS:
                    delta = panel['means'][B_MODEL][metric] - panel['means'][model][metric]
                    require(close(point_differences[model][metric], delta),
                            'B-minus-comparator point difference does not match means')
            status = panel.get('interval_status')
            require(status in STATUSES, 'Unknown interval status')
            if status.startswith('computed_'):
                require(panel.get('draws_requested') == panel.get('draws_completed') == 10000 and
                        panel.get('invalid_empty_draws') == 0, 'Incomplete bootstrap draws')
                intervals = panel.get('intervals')
                require(isinstance(intervals, dict) and set(intervals) == set(COMPARATORS),
                        'Incomplete B-minus-comparator intervals')
                for model, comparison in intervals.items():
                    for metric in METRICS:
                        entry = comparison[metric]
                        require(isinstance(entry, dict) and
                                set(entry) == {'difference', 'pointwise_95_percentile'},
                                'Only pointwise exploratory intervals are supported')
                        delta = panel['means'][B_MODEL][metric] - panel['means'][model][metric]
                        require(close(entry['difference'], delta),
                                'B-minus-comparator difference does not match means')
                        validate_interval(entry['pointwise_95_percentile'])
            else:
                require(panel.get('intervals') is None,
                        'An unavailable interval must not have endpoints')
                if status == 'not_estimable_insufficient_units_or_span':
                    require(panel.get('draws_completed') == 0,
                            'Unexpected draws for an unestimable interval')
                else:
                    require(panel.get('invalid_empty_draws', 0) > 0,
                            'Missing invalid-draw reason')
            require(type(panel.get('units')) is int and panel['units'] > 0 and
                    type(panel.get('nominal_nonoverlapping_units')) is int and
                    panel['nominal_nonoverlapping_units'] >= 0,
                    'Missing resampling unit counts')
            primary = panels[PRIMARY[dataset]]
            for field in ('means', 'sample_sd'):
                for model in MODELS:
                    for metric in METRICS:
                        require(close(panel[field][model][metric], primary[field][model][metric]),
                                'Sensitivity analysis changes the point estimates')
        taxi = panels.get('primary_168_hour')
        if taxi is not None:
            require(taxi.get('kind') == 'calendar' and taxi.get('block_hours') == 168 and
                    taxi['units'] < 336 and
                    taxi['interval_status'] == 'not_estimable_insufficient_units_or_span',
                    'The frozen Taxi primary interval must remain unavailable')
    return analysis


def fmt(value, digits=4):
    return f'{number(value):.{digits}f}'


def estimate(panel, model, metric, tex=False):
    digits = 5 if metric == 'time_nll' else 4
    mean = fmt(panel['means'][model][metric], digits)
    sd = fmt(panel['sample_sd'][model][metric], digits)
    return f'${mean} \\pm {sd}$' if tex else f'{mean} ± {sd}'


def comparison_values(panel, model, metric):
    delta = panel['point_differences'][model][metric]
    entry = panel['intervals'][model][metric] if panel['intervals'] is not None else None
    return delta, entry['pointwise_95_percentile'] if entry else None


def ci(value, tex=False):
    if value is None:
        return '---' if tex else '미산출'
    return '[' + ', '.join(fmt(endpoint, 6) for endpoint in value) + ']'


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows])


def latex_table(caption, label, headers, rows, columns):
    return '\n'.join([r'\begin{table}[p]', r'\centering\small',
                      r'\caption{' + caption + '}', r'\label{' + label + '}',
                      r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{' + columns + '}',
                      r'\toprule', ' & '.join(headers) + r' \\', r'\midrule'] +
                     [' & '.join(row) + r' \\' for row in rows] +
                     [r'\bottomrule', r'\end{tabular}}', r'\end{table}', ''])


def comparison_table(panel, tex=False):
    rows = []
    for model in COMPARATORS:
        for metric in METRICS:
            delta, endpoints = comparison_values(panel, model, metric)
            rows.append([MODELS[model], METRIC_LABELS[metric], f'{delta:+.6f}', ci(endpoints, tex)])
    return rows


def render(analysis, source_sha):
    """Return (summary Markdown, insertion-ready LaTeX) from analysis JSON only."""
    validate_analysis(analysis)
    require(valid_sha(source_sha), 'Missing source analysis SHA-256')
    rows, tex_rows = [], []
    for dataset, label in DATASETS.items():
        panel = analysis['results'][dataset][PRIMARY[dataset]]
        for model, name in MODELS.items():
            rows.append([label, name] + [estimate(panel, model, metric) for metric in METRICS])
            tex_rows.append([label, name] + [estimate(panel, model, metric, True) for metric in METRICS])
    summary = [
        '# B 모델의 탐색적 test 추가 비교',
        '기존 legacy test 결과를 확인한 뒤 B 모델(`titantpp`)을 추가한 탐색적 분석입니다. '
        '독립 미접근 평가나 사전 확증 분석이 아닙니다. 고정 대표 모델은 '
        '`titantpp_history_mlp`이며 이 결과로 대표 모델을 자동 교체하지 않습니다.',
        '세 데이터셋 × 여덟 학습 모델 × 시드 42·52·62의 72조건을 포함합니다. '
        'B의 새 평가 9조건과 기존 MLP·외부 여섯 모델의 63조건을 구분합니다. '
        '아래 값은 시드별 지표의 평균 ± 표본 표준편차이며 예측 앙상블이 아닙니다. '
        '시간 NLL은 고정 이분산 lognormal 모형의 기록된 정수 시간 구간 확률에 대한 음의 로그입니다.',
        '차이는 **B − 비교 모델**입니다. 음수이면 B의 오류가 작습니다. '
        '신뢰구간은 고정된 세 학습 시드에 조건부인 개별 95% paired bootstrap 구간이며, '
        '시드 자체는 재표집하지 않습니다. 다중 비교 보정이나 비교군 전체의 오류율 보장을 주장하지 않습니다.',
        '**Taxi의 168시간 블록 기본 신뢰구간은 기간 부족으로 미산출입니다.** '
        '24·336시간 블록이나 전체 셀 민감도 결과로 기본 구간을 대체하지 않습니다. '
        'Intermittent의 기본 단위는 전체 site, Instacart는 전체 사용자입니다.',
        '## 전체 학습 모델의 test 결과',
        markdown_table(['데이터셋', '모델', 'MAE', 'RMSE', '시간 NLL'], rows),
        '## 기본 재표집 단위에서 B와 비교 모델의 차이',
    ]
    common = ('Exploratory addition after viewing legacy test results; not an untouched independent evaluation. '
              'The frozen MLP representative is not replaced. ')
    tables = ('% Requires booktabs and graphicx. Preserve existing validation and legacy-test tables.\n'
              '% Source analysis JSON SHA-256: ' + source_sha + '\n\n')
    tables += latex_table(common + 'All eight learned models across three datasets. '
                          'Entries are mean $\\pm$ sample standard deviation over seeds 42, 52, and 62; '
                          'they are not prediction ensembles.', 'tab:b-test-exploratory-models',
                          ['Dataset', 'Model', 'MAE', 'RMSE', 'Time NLL'], tex_rows, 'llrrr')
    status_rows = []
    for sensitivity in (False, True):
        if sensitivity:
            summary.extend(['## 별도로 제시하는 탐색적 민감도 분석',
                            '아래 결과는 모두 기술적 민감도 비교이며 기본 구간을 대체하지 않습니다. '
                            '재표집 단위가 적은 결과는 특히 제한적으로 해석합니다.'])
        for dataset, label in DATASETS.items():
            for name in PANELS[dataset]:
                if (name != PRIMARY[dataset]) != sensitivity:
                    continue
                panel = analysis['results'][dataset][name]
                status = STATUSES[panel['interval_status']]
                heading = f'### {label}: {PANEL_LABELS[name]}'
                summary.extend([heading,
                                f"상태: {status[0]}. 단위/시간 수 {panel['units']:,}, "
                                f"명목 비중첩 단위 수 {panel['nominal_nonoverlapping_units']:,}.",
                                markdown_table(['비교 모델', '지표', 'B − 비교 모델', '개별 95% 구간'],
                                               comparison_table(panel))])
                status_rows.append([label, PANEL_LABELS[name], '민감도' if sensitivity else '기본',
                                    str(panel['units']), str(panel['nominal_nonoverlapping_units']), status[0]])
                caption = (common + f'{label}: {PANEL_LABELS[name]}. '
                           + ('Descriptive sensitivity only; never a substitute for the primary interval. '
                              if sensitivity else 'Primary resampling unit. ')
                           + 'Differences are B minus comparator. Intervals are pointwise 95\\% and '
                           'conditional on fixed trained seeds, without multiplicity adjustment. '
                           + f'Status: {status[1]}. Dashes denote an unavailable interval.')
                if dataset == 'yellow_trip_hourly':
                    caption += ' The 168-hour primary interval is unavailable because the calendar span is too short.'
                tables += latex_table(caption, f'tab:b-test-{dataset.replace("_", "-")}-{name.replace("_", "-")}',
                                      ['Comparator', 'Metric', '$\\Delta$ (B minus comparator)', 'Pointwise 95\\% CI'],
                                      comparison_table(panel, True), 'llrr')
    summary.extend(['## 재표집 상태',
                    markdown_table(['데이터셋', '단위', '용도', '단위/시간 수', '명목 비중첩 단위 수', '상태'], status_rows),
                    '## 원본 연결', f'- 분석 JSON SHA-256: `{source_sha}`\n'
                    '- 이 formatter는 저장된 분석 JSON만 읽으며 추가 추론, 재표집, 대표 모델 선택을 수행하지 않습니다.'])
    return '\n\n'.join(summary) + '\n', tables


def write_outputs(analysis_path, output_dir):
    """Create a new directory with summary.md, tables.tex, and a hash receipt."""
    analysis_path, output_dir = Path(analysis_path), Path(output_dir)
    raw = analysis_path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    summary, tables = render(json.loads(raw), source_sha)
    output_dir.mkdir(parents=True, exist_ok=False)
    created = []
    try:
        output_hashes = {}
        for name, content in (('summary.md', summary), ('tables.tex', tables)):
            path = output_dir / name
            with path.open('x', encoding='utf-8') as handle:
                created.append(path)
                handle.write(content)
            output_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipt = {
            'status': 'rendered', 'split': 'test', 'exploratory': True,
            'representative_model': REPRESENTATIVE, 'representative_replaced': False,
            'learned_conditions': 72, 'new_b_conditions': 9, 'legacy_comparison_conditions': 63,
            'analysis_path': str(analysis_path.resolve()), 'analysis_sha256': source_sha,
            'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'output_sha256': output_hashes, 'prediction_files_read': False,
            'inference_performed': False, 'bootstrap_performed': False,
        }
        path = output_dir / 'render_receipt.json'
        with path.open('x', encoding='utf-8') as handle:
            created.append(path)
            json.dump(receipt, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        return receipt
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = write_outputs(args.analysis, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'Refusing to render: {error}\n')
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
