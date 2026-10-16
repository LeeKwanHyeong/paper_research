"""Reuse completed cohorts separately; local evidence/aggregation only, inference0."""
from __future__ import annotations
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MAIN = PROJECT / 'reports/titantpp_cnn_gru54_test_3seed_comparison_20261006_v1'
HISTORY = PROJECT / 'search_artifacts/titantpp_history_instacart_test_20261012_v1'
FULL = 'titantpp_cnn_gru54'
EXTERNAL = ('rmtpp', 'nhp', 'thp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head')
BASELINES = ('last_observed_quantity', 'mean_quantity_in_same_observed_window')
METRICS = ('overall_qty_rmse', 'overall_qty_mae', 'tail_qty_rmse', 'tail_qty_mae',
           'overall_time_nll', 'tail_time_nll')
LABELS = {'yellow_trip_hourly': 'Taxi', 'raf_spare_parts': 'RAF',
          'intermittent_frozen_5000': 'Intermittent', 'insta_market_basket': 'Instacart',
          FULL: 'CNN+GRU (FULL)', 'titantpp_cnn_gru54_correction_disabled': '보정 없음',
          'titantpp_cnn_gru54_current_only': '현재 상태만',
          'titantpp_history_mlp_width8': 'MLP8', 'titantpp_history_mlp_width12': 'MLP12',
          'last_observed_quantity': '직전 수량', 'mean_quantity_in_same_observed_window': '관측 이력 평균',
          'last': '직전 수량', 'mean': '관측 이력 평균'}
TAILS = {'yellow_trip_hourly': 3449, 'raf_spare_parts': 200,
         'intermittent_frozen_5000': 187, 'insta_market_basket': 35}
EXPECTED_COUNTS = {'yellow_trip_hourly': {'validation': 8268, 'test': 8327},
                   'raf_spare_parts': {'validation': 6690, 'test': 5226},
                   'intermittent_frozen_5000': {'validation': 86285, 'test': 88019}}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, relative, digest):
    name = Path(relative)
    path = root / name
    require(not name.is_absolute() and '..' not in name.parts and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()), 'Unsafe evidence path')
    require(sha(path) == digest, 'Evidence SHA mismatch: ' + str(path))
    return path


def close(actual, expected):
    require(actual is not None and expected is not None
            and math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12),
            'Reused metric/aggregate mismatch: ' + str((actual, expected)))


def statistics3(values):
    require(len(values) == 3 and all(math.isfinite(x) for x in values), 'Exactly3 finite seed metrics required')
    return {'n': 3, 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values), 'ddof': 1}


def flatten(metric):
    tail = metric['tail']
    return {prefix + key: section[key] for prefix, section in (('overall_', metric), ('tail_', tail))
            for key in ('qty_rmse', 'qty_mae', 'time_nll')}


def validate_metric(metric, dataset, split):
    require(metric['split'] == split and metric['count'] > 0 and metric['tail']['count'] > 0,
            'Metric split/population differs')
    if dataset in EXPECTED_COUNTS:
        require(metric['count'] == EXPECTED_COUNTS[dataset][split], 'Recorded target count differs')
    require(float(metric['quantity_boundaries'][-1]) == TAILS[dataset], 'Frozen Train tail threshold differs')
    require(all(math.isfinite(float(v)) for v in flatten(metric).values()), 'Nonfinite recorded metric')
    for section in (metric, metric['tail']):
        close(section['qty_rmse'], math.sqrt(section['qty_sse'] / section['count']))
        close(section['qty_mae'], section['qty_absolute_error_sum'] / section['count'])
        close(section['time_nll'], section['time_nll_sum'] / section['count'])


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['arm'], row['split']), []).append(row)
    out = []
    for (dataset, arm, split), values in sorted(groups.items()):
        require(sorted(row['seed'] for row in values) == [42, 52, 62], 'Seed set differs')
        require(len({row['count'] for row in values}) == len({row['tail_count'] for row in values}) == 1,
                'Cross-seed target counts differ')
        out.append({'dataset': dataset, 'arm': arm, 'split': split, 'count': values[0]['count'],
                    'tail_count': values[0]['tail_count'], 'tail_threshold': TAILS[dataset],
                    'metrics': {key: statistics3([row['metrics'][key] for row in values]) for key in METRICS}})
    return out


def paired(rows, controls):
    index = {(row['dataset'], row['arm'], row['seed'], row['split']): row for row in rows}
    out = []
    for dataset in sorted({row['dataset'] for row in rows}):
        for split in ('validation', 'test'):
            for control in controls:
                seed_rows = []
                for seed in (42, 52, 62):
                    left, right = index[dataset, FULL, seed, split], index[dataset, control, seed, split]
                    require(left['count'] == right['count'] and left['tail_count'] == right['tail_count'],
                            'Paired target population differs')
                    deltas = {key: left['metrics'][key] - right['metrics'][key] for key in METRICS}
                    seed_rows.append({'seed': seed, 'full_selected_epoch': left['selected_epoch'],
                                      'control_selected_epoch': right['selected_epoch'], 'deltas': deltas})
                out.append({'dataset': dataset, 'split': split, 'full_arm': FULL, 'control_arm': control,
                            'delta_definition': 'FULL minus control; negative is lower error/NLL',
                            'seed_rows': seed_rows,
                            'metrics': {key: {**statistics3([r['deltas'][key] for r in seed_rows]),
                                              'strict_improvement_seeds': [r['seed'] for r in seed_rows if r['deltas'][key] < 0],
                                              'strict_worsening_seeds': [r['seed'] for r in seed_rows if r['deltas'][key] > 0]}
                                        for key in METRICS}})
    return out


def main_cohort(qa):
    receipt_path = MAIN / 'comparison_receipt.json'
    require(sha(receipt_path) == '95e877f505047cd5de58eec9974c2dfc05431ac4abf776b63f96508b3054bd86', 'Representative9 receipt differs')
    receipt = read(receipt_path)
    require(receipt['status'] == 'complete' and receipt['candidate_conditions'] == 9
            and receipt['independent_untouched_Test'] is False and receipt['checkpoint_reselection'] is False,
            'Representative9 completion differs')
    for name, digest in receipt['reference_sources'].items():
        checked(PROJECT, name, digest)
    for name in ('metrics_per_seed.csv', 'metrics_aggregated.csv', 'TEST_TABLES.md', 'CONTRIBUTION_DECISION.md'):
        checked(MAIN, name, receipt['outputs'][name])
    csv_rows = list(csv.DictReader((MAIN / 'metrics_per_seed.csv').open()))
    prior_aggregated = list(csv.DictReader((MAIN / 'metrics_aggregated.csv').open()))
    rows, baseline_rows, source_refs = [], [], {}
    allowed = {FULL, *EXTERNAL, *BASELINES}
    index = {(r['dataset'], r['model'], r['seed'], r['split'], r['view']): r for r in csv_rows if r['model'] in allowed}
    for raw in (r for r in csv_rows if r['model'] in allowed and r['view'] == 'overall'):
        source_path = checked(PROJECT, raw['source'], raw['source_sha256'])
        source = read(source_path)
        source_refs[raw['source']] = raw['source_sha256']
        tail = index[raw['dataset'], raw['model'], raw['seed'], raw['split'], 'tail']
        require(float(tail['threshold']) == TAILS[raw['dataset']], 'Main table tail differs')
        is_baseline = raw['model'] in BASELINES
        metric = {prefix + key: (float(section[key]) if section[key] else None)
                  for prefix, section in (('overall_', raw), ('tail_', tail))
                  for key in ('qty_rmse', 'qty_mae', 'time_nll')}
        if raw['model'] == FULL:
            require(source['dataset'] == raw['dataset'] and source['model'] == FULL
                    and source['seed'] == int(raw['seed']) and source['split'] == raw['split']
                    and source['status'] == 'complete' and source['full_population'] is True,
                    'Representative actual receipt identity differs')
            for key, value in flatten(source['metrics']).items():
                close(metric[key], value)
        elif raw['split'] == 'validation' and not is_baseline:
            require(source.get('evaluation_scope', source['selected'].get('evaluation_scope')) == 'validation_only'
                    and source.get('held_out_test_evaluated', source['selected'].get('held_out_test_evaluated')) is False,
                    'External selected Validation endpoint differs')
            for key, value in flatten(source['selected']).items():
                close(metric[key], value)
        elif raw['split'] == 'test' and not is_baseline:
            require(source['dataset'] == raw['dataset'] and source['model'] == raw['model']
                    and source['seed'] == int(raw['seed']) and source['split'] == 'test'
                    and source['status'] == 'complete' and source['full_population'] is True,
                    'External full-Test receipt differs')
        if is_baseline:
            require(metric['overall_time_nll'] is None and metric['tail_time_nll'] is None,
                    'Nonlearned baseline TimeNLL must remain N/A')
        item = {'dataset': raw['dataset'], 'arm': raw['model'], 'seed': None if is_baseline else int(raw['seed']),
                'split': raw['split'], 'count': int(raw['count']), 'tail_count': int(tail['count']),
                'selected_epoch': None if is_baseline else int(raw['selected_epoch']), 'metrics': metric,
                'source_path': raw['source'], 'source_raw_sha256': raw['source_sha256'],
                'provenance': raw['provenance'], 'time_nll_definition': raw['time_nll_definition'],
                'training_source': ('source117 seed42' if raw['seed'] == '42' else 'source123 seed52,62')
                    if raw['model'] == FULL else 'existing matched-head external cohort',
                'checkpoint_file_sha256': source.get('checkpoint_file_sha256'),
                'tensor_sha256': source.get('state_tensor_sha256'), 'target_identity_sha256': source.get('target_identity_sha256'),
                'truth_sha256': source.get('truth_sha256'), 'new_inference_performed': False}
        (baseline_rows if is_baseline else rows).append(item)
    summaries = aggregate(rows)
    for row in summaries:
        for view, prefix in (('overall', 'overall_'), ('tail', 'tail_')):
            original = next(r for r in prior_aggregated if r['dataset'] == row['dataset'] and r['model'] == row['arm']
                            and r['split'] == row['split'] and r['view'] == view)
            require(original['seed_count'] == '3', 'Original aggregate n differs')
            for key in ('qty_rmse', 'qty_mae', 'time_nll'):
                close(row['metrics'][prefix + key]['mean'], original[key + '_mean'])
                close(row['metrics'][prefix + key]['sample_sd'], original[key + '_sample_sd'])
    require(len([r for r in rows if r['arm'] == FULL and r['split'] == 'test']) == 9,
            'Representative9 Test identity count differs')
    return {'cohort': 'representative9_source117_123', 'test_condition_count': 9,
            'external_comparator_condition_count': 54, 'training_source': receipt['training_source_provenance'],
            'receipt_path': str(receipt_path.relative_to(PROJECT)), 'receipt_raw_sha256': sha(receipt_path),
            'rows': rows, 'summary_mean_sample_sd': summaries, 'paired_full_minus_external': paired(rows, EXTERNAL),
            'simple_quantity_baselines': baseline_rows, 'source_references': source_refs,
            'Validation_origin': 'CNNGRU9 native full replay; external6 original selected endpoint reused, no fresh external Validation replay',
            'Test_origin': 'Previously exposed chronological Test; CSV preserves SHA-verified full-population reaggregation',
            'new_inference_performed': False, 'new_training': False, 'test_independent': False}


def history_instacart_cohorts():
    completion_path = HISTORY / 'analysis/completion_receipt.json'
    require(sha(completion_path) == 'd21d137190e98552d07c5b9e2d8fc31151846e39c782ad907ff9a1a3cf644dbd', 'History/Instacart completion differs')
    completed = read(completion_path)
    require(completed['status'] == 'complete' and completed['distinct_conditions'] == 36
            and completed['actual_worker_exit0_verified'] is True and completed['reselected'] is False
            and completed['retrained'] is False and completed['test_is_independent'] is False,
            'History/Instacart completion scope differs')
    for table in ('source_files', 'analysis_files'):
        for name, digest in completed[table].items():
            checked(HISTORY, name, digest)
    analysis = read(HISTORY / 'analysis/test_conditions.json')
    contract = read(HISTORY / 'contract.json')
    results = []
    for study, label, expected, controls in (('history', 'history27_source129', 27,
            ('titantpp_cnn_gru54_correction_disabled', 'titantpp_cnn_gru54_current_only')),
            ('instacart', 'instacart9_source126', 9,
            ('titantpp_history_mlp_width8', 'titantpp_history_mlp_width12'))):
        conditions = [c for c in analysis['conditions'] if c['study'] == study]
        require(len(conditions) == expected and len({c['id'] for c in conditions}) == expected,
                'Reused cohort condition count differs')
        spec = contract['sources'][study]
        require(len(spec['files']) == (129 if study == 'history' else 126), 'Cohort source size differs')
        for name, digest in spec['files'].items():
            checked(HISTORY / spec['root'], name, digest)
        rows, baselines = [], {}
        for c in conditions:
            require(c['frozen_source_closure'] == spec['closure'] and c['reselected'] is False
                    and c['retrained'] is False and c['test_is_independent'] is False,
                    'History/Instacart source or selected condition differs')
            actual = read(HISTORY / 'results' / c['id'] / 'receipt.json')
            require(actual['job_id'] == c['id'] and actual['test'] == c['test']
                    and actual['validation_replay'] == c['validation_replay']
                    and actual['checkpoint_sha256'] == c['checkpoint_sha256'], 'Condition metrics differ from actual receipt')
            for split, exit_name, field in (('validation', 'validation_worker_exit.json', 'validation_replay'),
                                           ('test', 'worker_exit.json', 'test')):
                worker = read(HISTORY / 'results' / c['id'] / exit_name)
                require(worker['actual_child_exit_observed'] is True and type(worker['exit_code']) is int
                        and worker['exit_code'] == 0 and worker['job_id'] == c['id']
                        and worker['split'] == split, 'Actual scientific worker exit0 differs')
                metric = c[field]
                validate_metric(metric, c['dataset'], split)
                rows.append({'id': c['id'], 'study': study, 'dataset': c['dataset'], 'arm': c['arm'],
                    'seed': c['seed'], 'split': split, 'selected_epoch': c['selected_epoch'],
                    'checkpoint_file_sha256': c['checkpoint_sha256'], 'source_closure': c['frozen_source_closure'],
                    'source_revision': c['source_revision'], 'training_host': c['training_host'],
                    'count': metric['count'], 'tail_count': metric['tail']['count'], 'metrics': flatten(metric),
                    'validation_reference_metrics': flatten(c['validation_reference']) if split == 'validation' else None,
                    'data_identity': c['data_identity'], 'new_inference_performed': False})
            for name, metric in c['simple_quantity_baselines'].items():
                key = c['dataset'], name
                item = {'dataset': c['dataset'], 'arm': name, 'split': 'test', 'count': metric['count'],
                        'tail_count': metric['tail']['count'],
                        'metrics': {**{prefix + k: value[k] for prefix, value in (('overall_', metric), ('tail_', metric['tail']))
                                      for k in ('qty_rmse', 'qty_mae')},
                                    'overall_time_nll': None, 'tail_time_nll': None}, 'n': 1, 'sample_sd': None}
                require(key not in baselines or baselines[key] == item, 'Deterministic baseline differs across arms/seeds')
                baselines[key] = item
        summaries = aggregate(rows)
        for row in summaries:
            original = next(r for r in analysis['seed_summary'] if r['study'] == study
                            and r['dataset'] == row['dataset'] and r['arm'] == row['arm'])
            values = original['test' if row['split'] == 'test' else 'validation_replay']
            for key in METRICS:
                close(row['metrics'][key]['mean'], values[key]['mean'])
                close(row['metrics'][key]['sample_sd'], values[key]['sample_sd'])
        comparisons = paired(rows, controls)
        for row in comparisons:
            for seed_row in row['seed_rows']:
                original = next(p for p in analysis['paired_comparisons'] if p['study'] == study
                                and p['dataset'] == row['dataset'] and p['control_arm'] == row['control_arm']
                                and p['seed'] == seed_row['seed'])
                if row['split'] == 'test':
                    for key in METRICS:
                        close(seed_row['deltas'][key], original['test_delta'][key])
        results.append({'cohort': label, 'study': study, 'test_condition_count': expected,
            'source_file_count': len(spec['files']), 'source_closure': spec['closure'], 'source_revision': spec['revision'],
            'receipt_path': str(completion_path.relative_to(PROJECT)), 'receipt_raw_sha256': sha(completion_path),
            'rows': rows, 'summary_mean_sample_sd': summaries, 'paired_full_minus_controls': comparisons,
            'simple_quantity_baselines': list(baselines.values()), 'Validation_origin': 'Native replay metrics; original reference also preserved separately',
            'Test_origin': 'Previously exposed chronological Test with original actual worker exit0',
            'new_inference_performed': False, 'new_training': False, 'test_independent': False})
    return results


def number(value):
    return 'N/A' if value is None else f'{value:.7g}'


def format_stats(value):
    return f"{number(value['mean'])} ± {number(value['sample_sd'])}"


def render(value):
    lines = ['## Material Passport', '', '- Origin Skill: experiment-agent', '- Origin Mode: validate',
             '- Origin Date: ' + value['recorded_KST'], '- Verification Status: ANALYZED',
             '- Version Label: validation_v1', '', '# 완료된 기존 실험 결과의 별도 재사용표', '',
             '기존 대표 CNN+GRU9, 이력 보정27, Instacart9를 각 실험의 source·checkpoint·split 그대로 재사용했다. ',
             '신규 source131 구성요소36은 이 문서에 넣지 않았다. 서로 다른 실험의 FULL 값은 합치거나 교체하지 않는다. ',
             '이 작업은 원본 SHA와 기존 집계의 수치 검증이며, 학습·추론·checkpoint 재선택은 하지 않았다.', '',
             '모든 학습 모델은 seed42·52·62의 평균 ± 표본SD(ddof=1)다. n=3의 기술 통계이며 통계 유의성이나 새 독립 Test를 뜻하지 않는다. ',
             '큰 수량은 기존 Train 기준 Taxi q>3449, Intermittent q>187, RAF q>200, Instacart q>35를 유지한다. ',
             'TimeNLL은 기록된 양의 정수 시간의 lognormal round/clamp 확률 질량 점수다. 단순 수량 기준선의 TimeNLL은 N/A다.', '']
    for cohort in value['cohorts']:
        lines += ['## ' + cohort['cohort'], '',
                  f"원래 Test 조건 수: {cohort['test_condition_count']}. source: " + cohort.get('training_source', cohort.get('source_closure', '')),
                  '', 'Validation 출처: ' + cohort['Validation_origin'], 'Test 출처: ' + cohort['Test_origin'], '']
        for dataset in sorted({r['dataset'] for r in cohort['summary_mean_sample_sd']}):
            for split in ('validation', 'test'):
                rows = [r for r in cohort['summary_mean_sample_sd'] if r['dataset'] == dataset and r['split'] == split]
                lines += [f"### {LABELS[dataset]} / {split}", '',
                          f"전체 대상 {rows[0]['count']}, 큰 수량 대상 {rows[0]['tail_count']} (q>{TAILS[dataset]}).", '',
                          '| 모델 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | 전체 TimeNLL | 큰 수량 TimeNLL |',
                          '|---|---:|---:|---:|---:|---:|---:|']
                for row in rows:
                    lines.append('| ' + LABELS.get(row['arm'], row['arm']) + ' | ' + ' | '.join(format_stats(row['metrics'][m]) for m in METRICS) + ' |')
                lines += ['']
        comparisons = cohort.get('paired_full_minus_controls', cohort.get('paired_full_minus_external'))
        lines += ['### FULL 대비 seed별 전체 RMSE 차이', '',
                  '차이는 FULL − 대조군이다. 음수는 FULL의 오류가 더 작다는 뜻이다. 여섯 지표의 모든 seed별 차이와 평균·표본SD는 JSON에 보존했다.', '',
                  '| dataset | split | 대조군 | seed42 차이 | seed52 차이 | seed62 차이 | FULL 개선 seed |',
                  '|---|---|---|---:|---:|---:|---|']
        for row in comparisons:
            wins = row['metrics']['overall_qty_rmse']['strict_improvement_seeds']
            lines.append('| ' + ' | '.join([LABELS[row['dataset']], row['split'], LABELS.get(row['control_arm'], row['control_arm']),
                         *[number(r['deltas']['overall_qty_rmse']) for r in row['seed_rows']],
                         ', '.join(map(str, wins)) if wins else '없음']) + ' |')
        lines += ['', '### 단순 수량 기준선', '',
                  '기준선은 같은 관측 이력에서 결정되는 값이므로 3개의 독립 seed 반복으로 부풀리지 않았다. 누락된 Validation 기준선은 새로 추론하지 않았다.', '',
                  '| dataset | split | 기준선 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | 전체/큰 수량 TimeNLL |',
                  '|---|---|---|---:|---:|---:|---:|---|']
        for row in cohort['simple_quantity_baselines']:
            lines.append('| ' + ' | '.join([LABELS[row['dataset']], row['split'], LABELS.get(row['arm'], row['arm']),
                         *[number(row['metrics'][m]) for m in METRICS[:4]], 'N/A']) + ' |')
        lines += ['']
    lines += ['## 결과가 지지하는 범위와 한계', '',
              '- 대표 모델9: Test 전체 수량 RMSE에서 S2P2보다 낮은 평균은 Intermittent다. Taxi·RAF에서는 S2P2보다 높다. 외부 여섯 모델 전체 또는 원 논문 방법 전체의 우월성으로 확대할 수 없다.',
              '- 이력 보정27: FULL의 Test 평균 전체 RMSE는 Taxi·Intermittent에서 두 대조군보다 낮고 RAF에서는 높다. 모든 dataset/seed에서 보정 효과가 유지된다는 주장은 지지되지 않는다.',
              '- Instacart9: FULL의 Test 평균 전체 RMSE는 MLP8·MLP12보다 소폭 낮다. seed52에서는 MLP12보다 높아 모든 seed 우월성으로 표현할 수 없다.',
              '- 전체 오류·큰 수량 오류·시간 점수는 별도 결과다. 하나의 지표 이득으로 나머지 지표의 이득을 대신 주장하지 않는다.',
              '- 신규 구성요소36의 결과와 결론은 별도 파일에서 확인해야 한다. 이 문서의 FULL을 신규 캠페인의 FULL 값으로 바꾸지 않는다.', '',
              '검증: main9의 기존 source reference263개와 출력 CSV SHA, history/Instacart completion의 source/analysis SHA, 선택 checkpoint와 실제 Validation/Test worker exit0, seed별 원본 지표와 기존 집계·Test paired 차이를 확인했다. ',
              '본 작업에서 원본 예측을 다시 만들지는 않았으며, 기존 외부 Test 재집계 CSV의 수치는 봉인된 결과를 재사용했다.', '',
              '원본 출처 및 SHA, 선택 epoch·checkpoint, 원본 Validation reference, 3seed의 모든 지표와 대조 차이는 REUSED_COHORT_TABLES.json에 들어 있다.', '']
    return '\n'.join(lines)


def main():
    outputs = [HERE / 'REUSED_COHORT_TABLES.json', HERE / 'REUSED_COHORT_TABLES.md']
    require(not any(path.exists() for path in outputs), 'Existing immutable reused-cohort output; no overwrite')
    qa_path = HERE / 'QA_BASELINE.json'
    qa = read(qa_path)
    require(qa['status'] == 'PASS' and all(row['passed'] for row in qa['checks']), 'Reused baseline QA did not pass')
    cohorts = [main_cohort(qa), *history_instacart_cohorts()]
    value = {'schema': 'titantpp_reused_cohort_tables_v1', 'status': 'complete',
             'recorded_KST': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
             'material_passport': {'origin_skill': 'experiment-agent', 'origin_mode': 'validate',
                                  'verification_status': 'ANALYZED', 'version_label': 'validation_v1'},
             'scope': 'Reuse separate completed main9/history27/Instacart9; no source131 rows',
             'source131_component36_included': False, 'FULL_values_pooled_or_substituted': False,
             'new_inference_performed': False, 'new_training': False, 'remote_calls': 0,
             'checkpoint_reselection': False, 'test_independent': False,
             'statistical_significance_established': False, 'summary_definition': 'mean and sample SD ddof1 across seeds42,52,62; no ensemble',
             'QA_BASELINE_path': str(qa_path.relative_to(PROJECT)), 'QA_BASELINE_raw_sha256': sha(qa_path),
             'cohorts': cohorts}
    for path, raw in ((outputs[0], (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()),
                      (outputs[1], render(value).encode())):
        with path.open('xb') as handle:
            handle.write(raw)
    print(json.dumps({'status': 'complete', 'test_conditions': [c['test_condition_count'] for c in cohorts],
                      'files': {str(p.relative_to(PROJECT)): sha(p) for p in outputs},
                      'new_inference_performed': False, 'remote_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
