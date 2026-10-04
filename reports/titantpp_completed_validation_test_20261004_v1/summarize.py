"""Source-bound descriptive comparison; no inference or model selection."""
import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = {
    'yellow_trip_hourly': 'Taxi',
    'intermittent_frozen_5000': 'Intermittent',
    'insta_market_basket': 'Instacart',
    'raf_spare_parts': 'RAF',
}
LABELS = {
    'titantpp_history_mlp': 'TitanTPP MLP 폭4',
    'titantpp_history_mlp_width16': 'TitanTPP MLP 폭16',
    'titantpp': 'TitanTPP B',
    'rmtpp': 'RMTPP', 'thp': 'THP', 'nhp': 'NHP', 'sahp': 'SAHP',
    's2p2_matched_head': 'S2P2 (공통 출력부)',
    'attnhp_matched_head': 'AttNHP (공통 출력부)',
    'titantpp_all_available_history_mlp': '모든 가용 분기 MLP',
    'titantpp_current_only_param_matched': '현재 상태만 · 파라미터 대조',
    'titantpp_history_cross_product': 'Cross-product · A100 seed42',
    'titantpp_history_recent4_mean': 'Recent-four mean · A100 seed42',
    'titantpp_history_recent4_attention': 'Recent-four attention · A100 seed42',
    'titantpp_history_post_block': 'Post-block · A100 seed42',
    'deep_renewal_event_native_nb': 'Deep Renewal (native NB · 연구 기록)',
    'last_observed_quantity': '마지막 관측 수량',
    'mean_quantity_in_same_observed_window': '관측 이력 평균',
}
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')
MODEL_ORDER = list(LABELS)
THRESHOLDS = dict(zip(DATASETS, (3449, 187, 35, 200)))
BASE_MODELS = ('titantpp_history_mlp', 'rmtpp', 'thp', 'nhp', 'sahp',
               's2p2_matched_head', 'attnhp_matched_head',
               'titantpp_all_available_history_mlp', 'titantpp_current_only_param_matched',
               'titantpp_history_mlp_width16', 'deep_renewal_event_native_nb')
A100_MODELS = ('titantpp_history_cross_product', 'titantpp_history_recent4_mean',
               'titantpp_history_recent4_attention', 'titantpp_history_post_block')
REFERENCES = ('last_observed_quantity', 'mean_quantity_in_same_observed_window')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def normalize(row):
    row = dict(row)
    row['view'] = row.get('view') or 'overall'
    if row['view'] == 'all':
        row['view'] = 'overall'
    row['seed'] = int(row['seed']) if row.get('seed') not in ('', None, 'null') else None
    row['count'] = int(row['count'])
    row['threshold'] = float(row['threshold']) if row.get('threshold') not in ('', None) else None
    if row['view'] == 'tail':
        rule = row.get('threshold_rule')
        assert rule in (None, '', 'raw_quantity > threshold', 'raw_quantity_strictly_greater_than_train_threshold'), rule
        row['threshold_rule'] = 'raw_quantity > frozen_train_threshold'
        assert row['threshold'] == THRESHOLDS[row['dataset']], row
    else:
        row['threshold_rule'] = ''
    if not row.get('time_nll_definition'):
        row['time_nll_definition'] = ('not_applicable_quantity_only' if row['seed'] is None else
            'recorded_positive_integer_shifted_negative_binomial_mass_nll; top-coded survival where configured' if row['model'] == 'deep_renewal_event_native_nb' else
            'recorded_positive_integer_lognormal_round_clamp_mass_nll; top-coded survival where configured')
    for key in METRICS:
        value = row.get(key)
        row[key] = float(value) if value not in ('', None, 'null') else None
        assert row[key] is None or math.isfinite(row[key]), (row, key)
    assert row['count'] > 0 and row['qty_rmse'] is not None and row['qty_mae'] is not None
    assert 0 <= row['qty_mae'] <= row['qty_rmse'] + 1e-8
    assert row['time_nll'] is not None if row['seed'] is not None else row['time_nll'] is None
    return row


def validate_scope(rows):
    expected = {(dataset, model, seed, split, view)
                for dataset in DATASETS for model in BASE_MODELS for seed in (42, 52, 62)
                for split in ('validation', 'test') for view in ('overall', 'tail')}
    expected.update((dataset, 'titantpp', seed, split, view)
                    for dataset in DATASETS if dataset != 'raf_spare_parts'
                    for seed in (42, 52, 62) for split in ('validation', 'test')
                    for view in ('overall', 'tail'))
    expected.update((dataset, model, 42, split, view)
                    for dataset in DATASETS if dataset != 'insta_market_basket'
                    for model in A100_MODELS for split in ('validation', 'test')
                    for view in ('overall', 'tail'))
    expected.update((dataset, model, None, split, view)
                    for dataset in DATASETS for model in REFERENCES
                    for split in ('validation', 'test') for view in ('overall', 'tail'))
    actual = [(r['dataset'], r['model'], r['seed'], r['split'], r['view']) for r in rows]
    assert len(actual) == len(set(actual)), 'Duplicate model/seed/split/view'
    assert set(actual) == expected, {'missing': sorted(map(str, expected - set(actual))),
                                     'unexpected': sorted(map(str, set(actual) - expected))}
    for dataset in DATASETS:
        for split in ('validation', 'test'):
            for view in ('overall', 'tail'):
                counts = {r['count'] for r in rows if (r['dataset'], r['split'], r['view']) == (dataset, split, view)}
                assert len(counts) == 1, (dataset, split, view, counts)
    return {'exact_condition_split_view_coverage': True, 'population_counts_match_all_models': True,
            'tail_thresholds_and_strict_rule_match': True, 'duplicate_rows': False}


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['dataset'], row['model'], row['split'], row['view'])].append(row)
    result = []
    for (dataset, model, split, view), values in sorted(groups.items()):
        seeds = [v['seed'] for v in values]
        assert len(set(seeds)) == len(seeds), (dataset, model, split, view, seeds)
        assert len(set(v['count'] for v in values)) == 1, 'Different seed populations'
        for field in ('threshold', 'threshold_rule', 'time_nll_definition'):
            assert len({v.get(field) for v in values}) == 1, (dataset, model, split, view, field)
        assert len(values) in (1, 3), 'Never report an incomplete two-seed group'
        if len(values) == 3:
            assert set(seeds) == {42, 52, 62}
        row = {'dataset': dataset, 'model': model, 'split': split, 'view': view,
               'seeds': ','.join(str(s) for s in sorted(seeds) if s is not None),
               'n_seeds': len(values) if seeds[0] is not None else 0, 'count_per_seed': values[0]['count'],
               'threshold': values[0].get('threshold'), 'threshold_rule': values[0].get('threshold_rule'),
               'time_nll_definition': values[0].get('time_nll_definition'),
               'comparison_family': values[0].get('comparison_family')}
        for key in METRICS:
            numbers = [v[key] for v in values]
            assert all(n is None for n in numbers) or all(n is not None for n in numbers)
            row[key + '_mean'] = statistics.mean(numbers) if numbers[0] is not None else None
            row[key + '_sample_sd'] = statistics.stdev(numbers) if len(numbers) > 1 and numbers[0] is not None else None
        result.append(row)
    return result


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def formatted(row, metric):
    mean, sd = row[metric + '_mean'], row[metric + '_sample_sd']
    if mean is None:
        return '—'
    digits = 4 if metric == 'time_nll' else 3
    return f'{mean:.{digits}f}' + (f' ± {sd:.{digits}f}' if sd is not None else '')


def table(rows):
    lines = ['| 모델 | seed 수 | RMSE ↓ | MAE ↓ | Time NLL ↓ |',
             '|---|---:|---:|---:|---:|']
    for row in sorted(rows, key=lambda r: MODEL_ORDER.index(r['model'])):
        lines.append('| ' + ' | '.join([
            LABELS.get(row['model'], row['model']), str(row['n_seeds']) if row['n_seeds'] else '—',
            *(formatted(row, m) for m in METRICS),
        ]) + ' |')
    return '\n'.join(lines)


def write_panels(path, aggregate_rows, split, view='overall', singles=False):
    lines = [f'# {split.title()} · {view}', '',
             '3seed는 seed42·52·62의 평균 ± 표본 표준편차입니다. 단일seed 및 결정적 기준선에는 표준편차를 만들지 않습니다.', '',
             '수량 지표는 동일한 평가 대상의 raw 수량으로 계산합니다. tail은 학습 자료에서 고정한 경계를 엄격히 초과한 동일 대상입니다.', '',
             'Time NLL은 동일한 기록 정수 간격과 상한 코딩에 대한 음의 로그 확률질량입니다. 공통 출력부 모델은 round/clamp lognormal, Deep Renewal은 native shifted-NB입니다. 같은 데이터·split에서 예측 점수를 비교하되 native 모델과의 차이를 출력부로 통제된 backbone 효과로 해석하지 않습니다.', '']
    for dataset, name in DATASETS.items():
        selected = [r for r in aggregate_rows if r['dataset'] == dataset and r['split'] == split and r['view'] == view]
        if singles:
            selected = [r for r in selected if r['seeds'] in ('42', '')]
        else:
            selected = [r for r in selected if r['n_seeds'] == 3 or r['seeds'] == '']
        if selected:
            lines.extend([f'## {name}', '', table(selected), ''])
    path.write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    args = parser.parse_args()
    rows, inputs = [], []
    for value in args.inputs:
        path = Path(value)
        inputs.append({'path': str(path), 'sha256': sha(path)})
        rows.extend(normalize(row) for row in read_rows(path)
                    if (row.get('view') or 'overall') in ('all', 'overall', 'tail'))
    overall = [r for r in rows if r['view'] == 'overall']
    assert len(overall) == 322, f'Expected 306 learned split rows + 16 deterministic references, got {len(overall)}'
    for split in ('validation', 'test'):
        learned = [r for r in overall if r['split'] == split and r['seed'] is not None]
        assert len(learned) == 153, (split, len(learned))
        width = [r for r in learned if r['model'] == 'titantpp_history_mlp_width16']
        assert {(r['dataset'], r['seed']) for r in width} == {(d, s) for d in DATASETS for s in (42, 52, 62)}
    scope_checks = validate_scope(rows)
    result = aggregate(rows)
    output = HERE / 'results'
    output.mkdir(exist_ok=True)
    write_csv(output / 'metrics_per_seed.csv', rows)
    write_csv(output / 'metrics_aggregated.csv', result)
    for split in ('validation', 'test'):
        write_panels(HERE / f'{split.upper()}_TABLES.md', result, split)
        paired42 = [r for r in rows if r['seed'] == 42 and r['view'] == 'overall' and r['split'] == split]
        write_panels(HERE / f'{split.upper()}_SEED42_TABLES.md', aggregate(paired42), split, singles=True)
    tail_rows = [r for r in result if r['view'] == 'tail']
    if tail_rows:
        for split in ('validation', 'test'):
            write_panels(HERE / f'{split.upper()}_TAIL_TABLES.md', result, split, 'tail')
            paired_tail42 = [r for r in rows if r['seed'] == 42 and r['view'] == 'tail' and r['split'] == split]
            write_panels(HERE / f'{split.upper()}_SEED42_TAIL_TABLES.md', aggregate(paired_tail42), split, 'tail', singles=True)
    deltas = []
    for dataset in DATASETS:
        for split in ('validation', 'test'):
            selected = {r['model']: r for r in result if r['dataset'] == dataset and r['split'] == split and r['view'] == 'overall'}
            candidate, baseline = selected['titantpp_history_mlp_width16'], selected['titantpp_history_mlp']
            row = {'dataset': dataset, 'split': split}
            for metric in METRICS:
                a, b = candidate[metric + '_mean'], baseline[metric + '_mean']
                row[metric + '_width16_minus_width4'] = a - b
                row[metric + '_relative_change_percent'] = 100 * (a / b - 1) if b != 0 else None
            deltas.append(row)
    write_csv(output / 'width16_vs_width4.csv', deltas)
    width_seed_deltas = []
    row_lookup = {(r['dataset'], r['model'], r['seed'], r['split'], r['view']): r for r in rows}
    for dataset in DATASETS:
        for seed in (42, 52, 62):
            for split in ('validation', 'test'):
                for view in ('overall', 'tail'):
                    reference = row_lookup[(dataset, 'titantpp_history_mlp', seed, split, view)]
                    candidate = row_lookup[(dataset, 'titantpp_history_mlp_width16', seed, split, view)]
                    row = {'dataset': dataset, 'seed': seed, 'split': split, 'view': view,
                           'count': reference['count']}
                    for metric in METRICS:
                        a, b = candidate[metric], reference[metric]
                        row[metric + '_width4'] = b
                        row[metric + '_width16'] = a
                        row[metric + '_difference'] = a - b
                        row[metric + '_relative_change_percent'] = 100 * (a / b - 1) if b != 0 else None
                    width_seed_deltas.append(row)
    write_csv(output / 'width16_vs_width4_per_seed.csv', width_seed_deltas)
    paired_deltas = []
    for split in ('validation', 'test'):
        for view in ('overall', 'tail'):
            for dataset in DATASETS:
                pool = [r for r in result if (r['dataset'], r['split'], r['view']) == (dataset, split, view)
                        and r['n_seeds'] == 3]
                baseline = next(r for r in pool if r['model'] == 'titantpp_history_mlp')
                candidates = [(r, baseline, 'three_seed_mean') for r in pool if r['model'] != baseline['model']]
                single_pool = aggregate([r for r in rows if (r['dataset'], r['split'], r['view'], r['seed']) ==
                                         (dataset, split, view, 42)])
                single_baseline = next(r for r in single_pool if r['model'] == 'titantpp_history_mlp')
                candidates.extend((r, single_baseline, 'paired_seed42') for r in single_pool if r['model'] in A100_MODELS)
                for candidate, reference, comparison in candidates:
                    row = {'dataset': dataset, 'split': split, 'view': view, 'model': candidate['model'],
                           'reference': reference['model'], 'comparison': comparison}
                    for metric in METRICS:
                        a, b = candidate[metric + '_mean'], reference[metric + '_mean']
                        row[metric + '_difference'] = a - b
                        row[metric + '_relative_change_percent'] = 100 * (a / b - 1) if b != 0 else None
                    paired_deltas.append(row)
    write_csv(output / 'all_vs_width4.csv', paired_deltas)
    receipt = {'status': 'complete', 'learned_conditions': 153, 'learned_split_rows': 306,
               'deterministic_split_rows': 16, 'inputs': inputs,
               'source': 'verified per-condition metrics and prediction receipts; original source identities remain in input files',
               'aggregation': 'event-weighted within seed; seed arithmetic mean and sample SD ddof1; no ensemble',
               'independent_test': False, 'model_selection_changed': False,
               'deep_renewal_manuscript_exclusion_preserved': True,
               'single_seed_A100_not_mixed_with_three_seed_results': True,
               'scope_checks': scope_checks,
               'tail_thresholds': THRESHOLDS,
               'outputs': {str(p.relative_to(HERE)): sha(p) for p in sorted(output.glob('*.csv'))}}
    receipt['tables'] = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.glob('*TABLES.md'))}
    (HERE / 'comparison_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(deltas, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
