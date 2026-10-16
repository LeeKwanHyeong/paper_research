"""Independent CSV calculations and presentation/source audit; local only."""
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def equal(a, b):
    if a is None or b in ('', None):
        assert a is None and b in ('', None)
    else:
        assert math.isclose(a, float(b), abs_tol=1e-12, rel_tol=1e-12), (a, b)


def mean_sd(values):
    mu = sum(values) / len(values)
    sd = math.sqrt(sum((v - mu)**2 for v in values) / (len(values) - 1)) if len(values) > 1 else None
    return mu, sd


def printed(value, metric):
    if value is None:
        return '—'
    if metric == 'time_nll' and value != 0 and abs(value) < 0.0001:
        return f'{value:.3e}'
    return f'{value:.{4 if metric == "time_nll" else 3}f}'


def main():
    receipt = json.loads((HERE / 'comparison_receipt.json').read_text())
    assert receipt['status'] == 'complete'
    assert receipt['baseline'] == 'titantpp_history_mlp_width16' and not receipt['baseline_changed']
    sources = receipt['source_files_sha256']
    for path, expected in sources.items():
        assert not any('a100' in p.lower() for p in Path(path).parts)
        assert digest(ROOT / path) == expected, path
    for path, expected in receipt['outputs'].items():
        assert digest(HERE / path) == expected, path
    raw = csv_rows('metrics_per_seed.csv')
    aggregate = csv_rows('metrics_aggregated.csv')
    assert len(raw) == 516 and len(aggregate) == 188
    keys = [(r['dataset'], r['model'], r['seed'], r['split'], r['view']) for r in raw]
    assert len(keys) == len(set(keys))
    assert not any(r['model'] in receipt['excluded_single_seed_exploration_arms'] for r in raw)
    by_key = dict(zip(keys, raw))
    source_values = {}
    reference = ROOT / 'reports/titantpp_completed_validation_test_20261004_v1/results/metrics_per_seed.csv'
    for path in [reference, HERE / 'test_analysis/new_width_metrics_per_seed.csv']:
        with path.open() as stream:
            for row in csv.DictReader(stream):
                if row['model'] in receipt['excluded_single_seed_exploration_arms'] or row['dataset'] not in receipt['datasets']:
                    continue
                key = tuple(row[k] for k in ('dataset', 'model', 'seed', 'split', 'view'))
                assert key not in source_values
                source_values[key] = row
    assert set(source_values) == set(by_key)
    for key, row in by_key.items():
        original = source_values[key]
        for metric in METRICS:
            equal(float(original[metric]) if original[metric] else None, row[metric])
        assert row['count'] == original['count']
        assert row['selected_epoch'] == original['selected_epoch']
    for row in aggregate:
        key = row['dataset'], row['model'], row['split'], row['view']
        members = [r for r in raw if tuple(r[k] for k in ('dataset', 'model', 'split', 'view')) == key]
        assert len({r['count'] for r in members}) == 1
        assert len({r['threshold'] for r in members}) == 1
        assert int(row['count_per_seed']) == int(members[0]['count'])
        if row['seed_count'] != '0':
            assert len(members) == 3 and {r['seed'] for r in members} == {'42', '52', '62'}
        for metric in METRICS:
            values = [float(r[metric]) for r in members if r[metric]]
            assert len(values) in (0, len(members))
            mu, sd = mean_sd(values) if values else (None, None)
            equal(mu, row[metric + '_mean'])
            equal(sd, row[metric + '_sample_sd'])
            filename = f'{row["split"].upper()}_{"TAIL_" if row["view"] == "tail" else ""}TABLES.md'
            expected = printed(mu, metric) + (' ± ' + printed(sd, metric) if sd is not None else '')
            assert expected in (HERE / filename).read_text(), (filename, expected)
    paired = csv_rows('paired_deltas_per_seed.csv')
    deltas = csv_rows('paired_deltas_aggregated.csv')
    assert len(paired) == 1368 and len(deltas) == 456
    for row in paired:
        key = row['dataset'], row['model'], row['seed'], row['split'], row['view']
        base = row['dataset'], row['baseline'], row['seed'], row['split'], row['view']
        value, baseline = float(by_key[key][row['metric']]), float(by_key[base][row['metric']])
        equal(value - baseline, row['delta'])
        equal(value, row['model_value'])
        equal(baseline, row['baseline_value'])
    for row in deltas:
        members = [r for r in paired if all(row[k] == r[k] for k in ('dataset', 'model', 'split', 'view', 'metric'))]
        assert len(members) == 3 and {r['seed'] for r in members} == {'42', '52', '62'}
        values = [float(r['delta']) for r in members]
        mu, sd = mean_sd(values)
        equal(mu, row['delta_mean'])
        equal(sd, row['delta_sample_sd'])
        assert int(row['improved_seeds']) == sum(v < 0 for v in values)
        assert int(row['worsened_seeds']) == sum(v > 0 for v in values)
        assert int(row['unchanged_seeds']) == sum(v == 0 for v in values)
        worst = max(members, key=lambda r: float(r['delta']))
        assert row['worst_delta_seed'] == worst['seed']
        equal(float(worst['delta']), row['worst_delta'])
        absolute_worst = max(members, key=lambda r: float(r['model_value']))
        assert row['worst_metric_seed'] == absolute_worst['seed']
        equal(float(absolute_worst['model_value']), row['worst_metric_value'])
        equal(100 * mu / (sum(float(r['baseline_value']) for r in members) / 3), row['relative_mean_change_percent'])
    # Only width8/12 aggregate receipts contain event sums; audit those directly.
    direct = 0
    for evidence in receipt['source_evidence']:
        if evidence['model'] not in {'titantpp_history_mlp_width8', 'titantpp_history_mlp_width12'} or evidence['view'] != 'overall':
            continue
        source = json.loads((ROOT / evidence['source']).read_text())
        metrics = source['metrics']
        cells = metrics['quantity_cells']
        assert sum(c['count'] for c in cells) == metrics['count']
        for total_name in ('qty_sse', 'qty_absolute_error_sum', 'time_nll_sum'):
            equal(sum(c[total_name] for c in cells), metrics[total_name])
        for part in (metrics, metrics['tail']):
            equal(math.sqrt(part['qty_sse'] / part['count']), part['qty_rmse'])
            equal(part['qty_absolute_error_sum'] / part['count'], part['qty_mae'])
            equal(part['time_nll_sum'] / part['count'], part['time_nll'])
        assert source['raw_predictions_written'] is False and source['parts'] == []
        direct += 1
    assert direct == 36
    result = {'status': 'passed', 'metric_rows': len(raw), 'summary_groups': len(aggregate), 'paired_summary_rows': len(deltas), 'paired_seed_rows': len(paired), 'SHA_verified_source_files': len(sources), 'direct_width8_12_population_sum_checks': direct, 'all_original_metric_values_preserved': True, 'split_separated': True, 'exact_condition_seed_coverage': True, 'ddof1_independent_formula_verified': True, 'paired_deltas_and_worst_seeds_verified': True, 'overall_tail_counts_fixed_thresholds_verified': True, 'presentation_metric_format_verified': True, 'baseline_width16_preserved': True, 'A100_scope_not_accessed': True, 'new_inference_calls': 0, 'remote_calls': 0, 'comparison_receipt_sha256': digest(HERE / 'comparison_receipt.json'), 'validation_digest_limitation_preserved': True}
    (HERE / 'final_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
