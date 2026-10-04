"""Verify aggregate receipts and compare pure Test rows; no model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT / 'reports/titantpp_completed_validation_test_20261004_v1'
DATASETS = {'yellow_trip_hourly': 'Taxi', 'raf_spare_parts': 'RAF', 'intermittent_frozen_5000': 'Intermittent'}
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')
LABELS = {'titantpp_history_mlp': 'TitanTPP MLP 폭4', 'titantpp_history_mlp_width8': 'TitanTPP MLP 폭8',
          'titantpp_history_mlp_width12': 'TitanTPP MLP 폭12', 'titantpp_history_mlp_width16': 'TitanTPP MLP 폭16',
          'titantpp': 'TitanTPP B', 'rmtpp': 'RMTPP', 'thp': 'THP', 'nhp': 'NHP', 'sahp': 'SAHP',
          's2p2_matched_head': 'S2P2 · 공통 출력부', 'attnhp_matched_head': 'AttNHP · 공통 출력부',
          'titantpp_all_available_history_mlp': '모든 가용 분기 MLP',
          'titantpp_current_only_param_matched': '현재 상태만 · 파라미터 대조',
          'last_observed_quantity': '마지막 관측 수량', 'mean_quantity_in_same_observed_window': '같은 관측 이력 평균',
          'titantpp_history_cross_product': 'A100 Cross-product · seed42',
          'titantpp_history_recent4_mean': 'A100 Recent4 mean · seed42',
          'titantpp_history_recent4_attention': 'A100 Recent4 attention · seed42',
          'titantpp_history_post_block': 'A100 Post-block · seed42',
          'deep_renewal_event_native_nb': 'Deep Renewal native NB · 연구 기록 / 원고 제외'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def csv_write(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric(row):
    result = {key: row.get(key) for key in ('dataset', 'model', 'split', 'view', 'count', 'threshold',
              'selected_epoch', 'receipt_path', 'receipt_sha256', 'source', 'source_sha256', 'comparison_family')}
    result['seed'] = int(row['seed']) if row.get('seed') not in ('', None) else None
    result['count'] = int(row['count'])
    result['threshold'] = float(row['threshold']) if row.get('threshold') not in ('', None) else None
    result['source'] = row.get('source') or row.get('receipt_path')
    result['source_sha256'] = row.get('source_sha256') or row.get('receipt_sha256')
    for key in METRICS:
        result[key] = float(row[key]) if row.get(key) not in ('', None) else None
        require(result[key] is None or math.isfinite(result[key]), f'Nonfinite source metric: {key}')
    return result


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['model'], row['view']), []).append(row)
    output = []
    for (dataset, model, view), members in groups.items():
        require(len({member['seed'] for member in members}) == len(members), 'Duplicate model/seed Test row')
        require(len({member['count'] for member in members}) == 1, 'Seed population counts differ')
        require(len(members) == 1 or {member['seed'] for member in members} == {42, 52, 62}, 'Partial 3seed result must not be averaged')
        record = {'dataset': dataset, 'model': model, 'split': 'test', 'view': view,
                  'seed_count': len(members) if members[0]['seed'] is not None else 0,
                  'count_per_seed': members[0]['count'], 'threshold': members[0]['threshold']}
        for key in METRICS:
            values = [member[key] for member in members]
            require(all(v is None for v in values) or all(v is not None for v in values), 'Missing metric within model seeds')
            record[key + '_mean'] = statistics.mean(values) if values[0] is not None else None
            record[key + '_sample_sd'] = statistics.stdev(values) if values[0] is not None and len(values) > 1 else None
        output.append(record)
    return output


def formatted(row, metric):
    mean, sd = row[metric + '_mean'], row[metric + '_sample_sd']
    if mean is None:
        return '—'
    precision = 4 if metric == 'time_nll' else 3
    return f'{mean:.{precision}f}' + (f' ± {sd:.{precision}f}' if sd is not None else '')


def main(args):
    spec = importlib.util.spec_from_file_location('width8_12_test_pipeline_analysis', HERE / 'pipeline.py')
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    new_rows, evidence = [], []
    for campaign in args.campaign:
        campaign = Path(campaign).resolve()
        contract, registry = read(campaign / 'execution_contract.json'), read(campaign / 'evaluation_registry.json')
        completion = read(campaign / 'inference_completion.json')
        require(completion['status'] == 'complete' and completion['contract_sha256'] == sha(campaign / 'execution_contract.json'), 'Batch incomplete or changed')
        require(read(campaign / 'validation_gate.json')['passed'] is True, 'Full Validation gate missing')
        require(len(completion['completed']) == len(registry['rows']) * 2, 'Unexpected batch split count')
        for condition in registry['rows']:
            for split in ('validation', 'test'):
                folder = campaign / 'runs' / split / pipeline.identity(condition)
                checked = pipeline.check_receipt(folder, condition, split, contract, campaign)
                committed = [r for r in completion['completed'] if r['condition'] == checked['condition'] and r['split'] == split]
                require(len(committed) == 1 and committed[0] == checked, 'Completion/receipt aggregate differs')
                receipt = read(folder / 'receipt.json')
                evidence.append({'campaign': str(campaign.relative_to(ROOT)), **{k: checked[k] for k in ('condition', 'split', 'receipt_sha256', 'target_identity_sha256', 'truth_sha256')}})
                for view, metrics in [('overall', checked['metrics']), ('tail', checked['metrics']['tail'])]:
                    new_rows.append(numeric({'dataset': condition['dataset'], 'model': condition['model'], 'seed': condition['seed'],
                        'split': split, 'view': view, 'count': metrics['count'], 'threshold': receipt['metrics']['quantity_boundaries'][-1] if view == 'tail' else None,
                        'selected_epoch': condition['selected_epoch'], 'receipt_path': str((folder / 'receipt.json').relative_to(ROOT)),
                        'receipt_sha256': checked['receipt_sha256'], 'source': str((folder / 'receipt.json').relative_to(ROOT)),
                        'source_sha256': checked['receipt_sha256'], 'comparison_family': 'frozen_width8_12_existing_split', **{key: metrics[key] for key in METRICS}}))
    datasets = {row['dataset'] for row in new_rows}
    old_path = REFERENCE / 'results/metrics_per_seed.csv'
    prior_receipt = read(REFERENCE / 'comparison_receipt.json')
    require(prior_receipt['status'] == 'complete', 'Existing benchmark comparison incomplete')
    require(sha(old_path) == prior_receipt['outputs']['results/metrics_per_seed.csv'], 'Existing benchmark CSV SHA differs')
    with old_path.open() as stream:
        old_rows = [numeric(row) for row in csv.DictReader(stream) if row['split'] == 'test' and row['view'] in ('overall', 'tail') and row['dataset'] in datasets]
    reference_evidence = {}
    for row in old_rows:
        source = ROOT / row['source']
        require(source.suffix == '.json' and source.is_file() and sha(source) == row['source_sha256'], 'Original Test aggregate receipt missing or changed')
        reference_evidence[str(source.relative_to(ROOT))] = row['source_sha256']
        receipt = read(source)
        require(receipt['split'] == 'test' and receipt['status'] == 'complete' and receipt['full_population'], 'Original reference is not complete Test')
        population = next(read(Path(campaign) / 'execution_contract.json')['population_references'][row['dataset']]['test']
                          for campaign in args.campaign if row['dataset'] in read(Path(campaign) / 'execution_contract.json')['population_references'])
        for field in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'expected_target_count'):
            require(receipt[field] == population[field], f'Original/new Test population or history differs: {field}')
    test_rows = old_rows + [row for row in new_rows if row['split'] == 'test']
    require(all(row['split'] == 'test' for row in test_rows), 'Comparison must be pure Test')
    keys = [(row['dataset'], row['model'], row['seed'], row['view']) for row in test_rows]
    require(len(keys) == len(set(keys)), 'Duplicate existing/new Test condition')
    for dataset in datasets:
        for view in ('overall', 'tail'):
            matching = [row for row in test_rows if row['dataset'] == dataset and row['view'] == view]
            require(len({row['count'] for row in matching}) == 1, 'Existing/new eligible Test populations differ')
            if view == 'tail':
                require(len({row['threshold'] for row in matching}) == 1, 'Existing/new fixed TRAIN tail threshold differs')
    summaries = aggregate(test_rows)
    output = Path(args.output).resolve()
    require(output.is_relative_to(HERE), 'Analysis outputs must remain in the new report directory')
    output.mkdir(parents=True, exist_ok=False)
    csv_write(output / 'new_width_metrics_per_seed.csv', new_rows)
    csv_write(output / 'comparison_test_per_seed.csv', test_rows)
    csv_write(output / 'comparison_test_aggregated.csv', summaries)
    for view, filename in [('overall', 'TEST_TABLES.md'), ('tail', 'TEST_TAIL_TABLES.md')]:
        lines = [f'# Test · {view}', '', '같은 원래 Test target과 causal history에 대한 체크포인트 평가다. 3seed는 seed42·52·62의 평균 ± 표본 표준편차이며 ensemble 점수가 아니다.', '',
                 'tail은 원래 학습 자료의 고정 경계를 엄격히 초과한 target이다. Time NLL은 데이터별 기록 정수 간격의 확률질량 점수다. Native NB와 공통 lognormal 출력부의 차이는 출력부로 통제된 backbone 효과가 아니다.', '']
        for dataset, label in DATASETS.items():
            if dataset not in datasets:
                continue
            lines.extend([f'## {label}', '', '| 모델 | seed 수 | Test target 수 | RMSE ↓ | MAE ↓ | Time NLL ↓ |', '|---|---:|---:|---:|---:|---:|'])
            subset = [row for row in summaries if row['dataset'] == dataset and row['view'] == view]
            width_order = ['titantpp_history_mlp', 'titantpp_history_mlp_width8', 'titantpp_history_mlp_width12', 'titantpp_history_mlp_width16']
            subset.sort(key=lambda row: (width_order.index(row['model']) if row['model'] in width_order else 10, row['model']))
            for row in subset:
                lines.append('| ' + ' | '.join([LABELS.get(row['model'], row['model']), str(row['seed_count']) if row['seed_count'] else '—', str(row['count_per_seed']), *[formatted(row, key) for key in METRICS]]) + ' |')
            lines.append('')
        lines.extend(['기존 비교군은 기존 집계 CSV의 `split=test` 행만 재사용했다. 폭8·12는 선택 epoch를 바꾸지 않았다. 새 독립 Test나 Test 기반 폭 선택으로 주장하지 않는다.', '',
                      f'원본: `{old_path.relative_to(ROOT)}` (SHA `{sha(old_path)}`). 신규 per-seed 근거: `new_width_metrics_per_seed.csv`; Test 전용 비교 근거: `comparison_test_per_seed.csv`.'])
        with (output / filename).open('x') as stream:
            stream.write('\n'.join(lines) + '\n')
    receipt = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(), 'new_conditions': len(new_rows) // 4,
               'new_population_splits': len(evidence), 'new_raw_metric_rows': len(new_rows), 'test_comparison_rows': len(test_rows),
               'reference_source': str(old_path.relative_to(ROOT)), 'reference_source_sha256': sha(old_path),
               'reference_comparison_receipt_sha256': sha(REFERENCE / 'comparison_receipt.json'),
               'reference_aggregate_receipts': reference_evidence,
               'new_evidence': evidence, 'comparison_split': 'test', 'new_training': False,
               'new_inference_calls_here': 0, 'selection_changed': False, 'new_independent_test_claim': False,
               'outputs': {path.name: sha(path) for path in output.iterdir() if path.is_file()}}
    with (output / 'analysis_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'complete', 'output': str(output), 'new_conditions': receipt['new_conditions']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', action='append', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    main(parser.parse_args())
