"""Format a complete, verified legacy-test analysis; never read prediction parts.

Usage: python render_results.py --analysis analysis.json --output-dir review
Existing outputs are preserved. This formatter is separate from the sealed
evaluator/analyzer and performs no inference, selection, or new statistics.
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
    'raf_spare_parts': 'RAF',
}
MODELS = {
    'titantpp_history_mlp': 'TitanTPP',
    'rmtpp': 'RMTPP',
    'thp': 'THP',
    'nhp': 'NHP',
    'sahp': 'SAHP',
    's2p2_matched_head': 'S2P2',
    'attnhp_matched_head': 'AttNHP',
    'titantpp_current_only_param_matched': 'Current-only control',
    'titantpp_all_available_history_mlp': 'All-available-history control',
}
SIMPLE = {
    'last_observed_quantity': 'Last observed quantity',
    'mean_quantity_in_same_observed_window': 'Observed-window mean',
}
SEEDS = ('42', '52', '62')
METRICS = ('qty_mae', 'qty_rmse', 'time_nll')
REPRESENTATIVE = 'titantpp_history_mlp'
CONTROLS = tuple(MODELS)[-2:]
PRIMARY = dict(zip(DATASETS, ('primary_168_hour', 'primary_site',
                              'primary_entity', 'primary_entity')))
ANCHORS = dict(zip(DATASETS, ('rmtpp', 'rmtpp', 's2p2_matched_head',
                              's2p2_matched_head')))
PANELS = {
    'yellow_trip_hourly': ('primary_168_hour', 'sensitivity_24_hour',
                          'sensitivity_336_hour', 'sensitivity_whole_cell'),
    'intermittent_frozen_5000': ('primary_site', 'sensitivity_series'),
    'insta_market_basket': ('primary_entity',),
    'raf_spare_parts': ('primary_entity',),
}
STATUSES = {
    'computed_conditional': ('산출: 고정 학습 시드에 조건부', 'Computed conditional'),
    'computed_descriptive': ('산출: 단위 수가 적어 기술적으로만 해석', 'Descriptive only'),
    'not_estimable_insufficient_units_or_span': ('미산출: 재표집 단위 또는 기간 부족', 'Not estimable'),
    'not_finalized_invalid_draws': ('미확정: 빈 재표집 발생', 'Not finalized'),
}
BONFERRONI = 'four_anchor_bonferroni_98_75_percentile'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value),
            'Missing or nonfinite numeric result')
    return float(value)


def close(left, right):
    return math.isclose(number(left), number(right), rel_tol=1e-9, abs_tol=1e-10)


def interval(value):
    require(isinstance(value, list) and len(value) == 2,
            'Missing confidence interval endpoints')
    require(number(value[0]) <= number(value[1]), 'Reversed confidence interval')


def validate_analysis(analysis):
    """Reject pending/failure, missing panels, and unverified population evidence."""
    require(analysis.get('status') == 'complete', 'Only a complete analysis can be rendered')
    require(analysis.get('split') == 'test', 'Only the approved legacy test analysis is accepted')
    require(analysis.get('learned_conditions') == 108 and
            analysis.get('deterministic_reference_vectors') == 8,
            'The complete 108-condition/eight-reference panel is required')
    require(analysis.get('prediction_ensemble') is False and
            analysis.get('v0_7_validation_baseline_replaced') is False,
            'Unexpected ensemble or baseline replacement')
    for key in ('run_manifest_sha256', 'analysis_script_sha256'):
        require(isinstance(analysis.get(key), str) and len(analysis[key]) == 64 and
                all(c in '0123456789abcdef' for c in analysis[key]), 'Missing source hash: ' + key)
    require(set(analysis['results']) == set(DATASETS) and
            set(analysis['validation']) == set(DATASETS), 'Incomplete dataset coverage')
    for dataset in DATASETS:
        evidence = analysis['validation'][dataset]
        for key in ('exact_common_population', 'frozen_population_identity_verified',
                    'truth_sha256_independently_verified'):
            require(evidence.get(key) is True, 'Unverified population evidence: ' + key)
        expected = {(model, int(seed)) for model in MODELS for seed in SEEDS}
        conditions = evidence['conditions']
        require(len(conditions) == 27 and
                {(row['model'], row['seed']) for row in conditions} == expected,
                'Incomplete or duplicate condition evidence')
        count = evidence['target_count']
        require(type(count) is int and count > 0 and
                all(row['prediction_rows'] == count for row in conditions), 'Incomplete prediction counts')
        panels = analysis['results'][dataset]
        require(set(panels) == set(PANELS[dataset]), 'Missing primary or sensitivity panel')
        for panel in panels.values():
            require(panel['count'] == count and panel['scope'] == 'legacy_test_reevaluation',
                    'Panel population/scope mismatch')
            require(panel['uncertainty_scope'] == 'conditional_on_fixed_trained_seeds' and
                    panel['estimate'] == 'mean_of_per_seed_metrics_not_prediction_ensemble',
                    'Unexpected uncertainty or aggregation definition')
            require(panel['time_nll_definition'] ==
                    'frozen_recorded_integer_bin_heteroscedastic_lognormal_NLL', 'Unexpected duration NLL')
            require(set(panel['means']) == set(MODELS) | set(SIMPLE), 'Incomplete model results')
            for model in MODELS:
                seeds = panel['seed_metrics'][model]
                require(set(seeds) == set(SEEDS), 'Missing learned seed metrics')
                for metric in METRICS:
                    values = [number(seeds[seed][metric]) for seed in SEEDS]
                    require(close(panel['means'][model][metric], statistics.mean(values)) and
                            close(panel['sample_sd'][model][metric], statistics.stdev(values)),
                            'Mean/sample SD does not match fixed seed metrics')
            require(panel['deterministic_models'] == list(SIMPLE) and
                    panel['deterministic_prediction_vectors_per_model'] == 1,
                    'Deterministic reference must be one prediction vector')
            for model in SIMPLE:
                require(set(panel['seed_metrics'][model]) == {'deterministic'}, 'Replicated deterministic reference')
                require(panel['means'][model]['time_nll'] is None and
                        panel['seed_metrics'][model]['deterministic']['time_nll'] is None and
                        all(panel['sample_sd'][model][key] is None for key in METRICS),
                        'Deterministic reference has an invented NLL or seed SD')
                for metric in METRICS[:2]:
                    require(close(panel['means'][model][metric],
                                  panel['seed_metrics'][model]['deterministic'][metric]),
                            'Deterministic reference estimate mismatch')
            status = panel['interval_status']
            require(status in STATUSES, 'Unknown confidence interval status')
            if status.startswith('computed_'):
                require(panel['draws_requested'] == panel['draws_completed'] == 10000 and
                        panel.get('invalid_empty_draws') == 0, 'Incomplete bootstrap draws')
                require(set(panel['intervals']) == (set(MODELS) | set(SIMPLE)) - {REPRESENTATIVE},
                        'Incomplete paired intervals')
                for model, comparison in panel['intervals'].items():
                    for metric in METRICS:
                        if model in SIMPLE and metric == 'time_nll':
                            require(comparison[metric] is None, 'Invented baseline NLL interval')
                            continue
                        entry = comparison[metric]
                        delta = panel['means'][REPRESENTATIVE][metric] - panel['means'][model][metric]
                        require(close(entry['difference'], delta), 'Paired difference does not match means')
                        interval(entry['pointwise_95_percentile'])
                interval(panel['intervals'][ANCHORS[dataset]]['qty_rmse'][BONFERRONI])
            else:
                require(panel['intervals'] is None, 'Unestimable interval must not have endpoints')
                if status == 'not_estimable_insufficient_units_or_span':
                    require(panel['draws_completed'] == 0, 'Unexpected draws for unestimable interval')
                else:
                    require(panel.get('invalid_empty_draws', 0) > 0, 'Missing failed-draw reason')
        primary = panels[PRIMARY[dataset]]
        for panel in panels.values():
            for field in ('means', 'sample_sd'):
                for model in (*MODELS, *SIMPLE):
                    for metric in METRICS:
                        left, right = panel[field][model][metric], primary[field][model][metric]
                        require((left is None and right is None) or
                                (left is not None and right is not None and close(left, right)),
                                'Sensitivity panel changes point estimates')
    return analysis


def fmt(value, digits=4):
    return f'{number(value):.{digits}f}'


def estimate(panel, model, metric, tex=False):
    if model in SIMPLE:
        return ('---' if tex else '—') if metric == 'time_nll' else fmt(panel['means'][model][metric])
    digits = 5 if metric == 'time_nll' else 4
    mean, sd = fmt(panel['means'][model][metric], digits), fmt(panel['sample_sd'][model][metric], digits)
    return f'${mean} \\pm {sd}$' if tex else f'{mean} ± {sd}'


def ci(value, tex=False):
    if value is None:
        return '---' if tex else '미산출'
    return '[' + ', '.join(fmt(v, 6) for v in value) + ']'


def anchor_values(panel, dataset):
    anchor = ANCHORS[dataset]
    delta = panel['means'][REPRESENTATIVE]['qty_rmse'] - panel['means'][anchor]['qty_rmse']
    entry = panel['intervals'][anchor]['qty_rmse'] if panel['intervals'] is not None else None
    return (delta, entry['pointwise_95_percentile'] if entry else None,
            entry[BONFERRONI] if entry else None)


def markdown_table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows])


def latex_table(caption, label, headers, rows, columns):
    return '\n'.join([r'\begin{table}[p]', r'\centering\small',
                      r'\caption{' + caption + '}', r'\label{' + label + '}',
                      r'\resizebox{\linewidth}{!}{%',
                      r'\begin{tabular}{' + columns + '}', r'\toprule',
                      ' & '.join(headers) + r' \\', r'\midrule'] +
                     [' & '.join(row) + r' \\' for row in rows] +
                     [r'\bottomrule', r'\end{tabular}}', r'\end{table}', ''])


def render(analysis, source_sha):
    """Return review prose and insertion-ready booktabs tables as strings."""
    validate_analysis(analysis)
    primary = {d: analysis['results'][d][PRIMARY[d]] for d in DATASETS}
    rows, tex_rows, simple_rows, simple_tex, structural, structural_tex = [], [], [], [], [], []
    for dataset, label in DATASETS.items():
        panel = primary[dataset]
        for model, name in MODELS.items():
            row = [label, name] + [estimate(panel, model, metric) for metric in METRICS]
            tex_row = [label, name] + [estimate(panel, model, metric, True) for metric in METRICS]
            rows.append(row); tex_rows.append(tex_row)
            if model in (REPRESENTATIVE, *CONTROLS):
                structural.append(row); structural_tex.append(tex_row)
        for model, name in SIMPLE.items():
            simple_rows.append([label, name] + [estimate(panel, model, metric) for metric in METRICS])
            simple_tex.append([label, name] + [estimate(panel, model, metric, True) for metric in METRICS])
    anchors, anchors_tex, status_rows = [], [], []
    for dataset, label in DATASETS.items():
        panel = primary[dataset]
        delta, pointwise, adjusted = anchor_values(panel, dataset)
        status = STATUSES[panel['interval_status']]
        anchors.append([label, MODELS[ANCHORS[dataset]], fmt(delta, 6), ci(pointwise), ci(adjusted), status[0]])
        anchors_tex.append([label, MODELS[ANCHORS[dataset]], fmt(delta, 6), ci(pointwise, True), ci(adjusted, True), status[1]])
        for name in PANELS[dataset]:
            item = analysis['results'][dataset][name]
            status_rows.append([label, name, str(item['units']),
                                str(item['nominal_nonoverlapping_units']), STATUSES[item['interval_status']][0]])
    headers = ['데이터셋', '모델', 'MAE', 'RMSE', '시간 NLL']
    scope = ('기존 test 분할의 고정 재평가이며, 독립 미접근 평가가 아닙니다. '
             '학습 모델 108조건(4개 데이터셋 × 9개 모델 × 3개 시드)과 결정적 기준 8개 벡터를 포함합니다. '
             '기존 validation 표와 v0.7 기준선은 보존합니다.')
    summary = '\n\n'.join([
        '# 기존 test 분할 재평가 검토 자료', scope,
        '학습 모델은 시드 42·52·62의 지표 평균 ± 표본 표준편차입니다. 예측 앙상블이 아닙니다. '
        '결정적 기준에는 시드 표준편차와 시간 NLL이 없습니다. 시간 NLL은 고정된 이분산 lognormal 모형이 '
        '기록된 정수 시간 구간에 부여한 확률의 음의 로그입니다.',
        '신뢰구간은 고정된 세 학습 시드에 조건부인 paired 재표집 결과입니다. 모델·시드를 함께 유지하며 '
        '시드 자체를 재표집하지 않습니다. 95% 구간은 개별 비교 구간이고, 98.75% 구간은 '
        'validation에서 고정한 네 RMSE 비교에 대한 Bonferroni 구간입니다. '
        'TitanTPP − 비교 모델의 차이가 음수이면 TitanTPP의 오류가 작습니다.',
        '**Taxi 기본 분석은 공유 달력 시간의 168시간 블록입니다. 기간이 짧아 기본 구간이 미산출이면 '
        '24·336시간 또는 공간 셀 민감도 분석으로 대체하지 않습니다.** '
        'Intermittent 기본 단위는 전체 site이며 시리즈 단위는 민감도 분석입니다. '
        '단위 수가 적은 구간은 기술적으로만 해석합니다.',
        '## 학습 모델 전체', markdown_table(headers, rows),
        '## 결정적 수량 기준', markdown_table(headers, simple_rows),
        '## 고정 비교 기준의 RMSE 차이와 기본 신뢰구간',
        markdown_table(['데이터셋', '고정 비교 모델', 'RMSE 차이', '95% 구간', '98.75% 구간', '상태'], anchors),
        '## 구조 비교', '대표 모델과 두 구조 비교를 다시 모은 표입니다. 전체 조건에 추가되는 실험은 아닙니다.',
        markdown_table(headers, structural),
        '## 재표집 상태', markdown_table(['데이터셋', '분석', '단위/시간 수', '명목 비중첩 단위 수', '상태'], status_rows),
        '## 원본 연결',
        f'- 분석 JSON SHA-256: `{source_sha}`\n'
        f'- 실행 manifest SHA-256: `{analysis["run_manifest_sha256"]}`\n'
        f'- 집계 코드 SHA-256: `{analysis["analysis_script_sha256"]}`\n'
        '- 이 문서는 검증된 집계 값의 표시 형식만 바꾸며 추가 추론·모델 선택·유의성 판정을 수행하지 않습니다.',
    ]) + '\n'
    common = ('Retrospective reevaluation of existing test splits, not an untouched independent evaluation. '
              'Learned-model entries are mean $\\pm$ sample standard deviation over the three fixed seeds.')
    tex_headers = ['Dataset', 'Model', 'MAE', 'RMSE', 'Time NLL']
    tables = '% Requires booktabs and graphicx. Preserve the existing development-validation tables.\n'
    tables += '% Source analysis JSON SHA-256: ' + source_sha + '\n\n'
    tables += latex_table(common + ' All nine learned models are shown.',
                          'tab:legacy-test-main', tex_headers, tex_rows, 'llrrr')
    tables += latex_table('Deterministic quantity references on the identical legacy-test targets and observed windows. '
                          'Each reference has one prediction vector; seed dispersion and time NLL are not defined.',
                          'tab:legacy-test-simple', tex_headers, simple_tex, 'llrrr')
    tables += latex_table('TitanTPP minus the frozen validation-selected anchor in legacy-test RMSE. '
                          'Intervals are conditional on the three fixed trained seeds. The 95\\% intervals are pointwise; '
                          '98.75\\% intervals adjust the four prespecified anchor comparisons. '
                          'Dashes denote an unavailable interval. Taxi uses shared 168-hour blocks; '
                          'sensitivity analyses never replace an unestimable primary interval.',
                          'tab:legacy-test-anchors', ['Dataset', 'Anchor', '$\\Delta$ RMSE', '95\\% CI', '98.75\\% CI', 'Status'],
                          anchors_tex, 'llrrrl')
    tables += latex_table(common + ' The representative and two structural controls are repeated from the complete panel.',
                          'tab:legacy-test-structural', tex_headers, structural_tex, 'llrrr')
    return summary, tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    created = []
    try:
        raw = args.analysis.read_bytes()
        analysis = json.loads(raw)
        summary, tables = render(analysis, hashlib.sha256(raw).hexdigest())
        outputs = [(args.output_dir / 'summary.md', summary), (args.output_dir / 'tables.tex', tables)]
        if any(path.exists() for path, _ in outputs):
            raise ValueError('Existing outputs are preserved; choose a new output directory')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, content in outputs:
            with path.open('x', encoding='utf-8') as handle:
                created.append(path)
                handle.write(content)
    except (OSError, ValueError, KeyError, TypeError) as error:
        for path in created:
            path.unlink(missing_ok=True)
        parser.exit(2, f'Refusing to render: {error}\n')
    print(json.dumps({'status': 'rendered', 'summary': str(outputs[0][0]),
                      'tables': str(outputs[1][0])}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
