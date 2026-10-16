"""Audit and summarize fixed source131 component36; read-only reuse of Inter12.

No inference, training, source changes, checkpoint selection, or automatic retry.
The final outputs are exclusive and appear only after all native receipts pass.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from paper.scripts import evaluate_titantpp_components_taxi_raf24_frozen as eval24
from paper.scripts import evaluate_titantpp_components_inter8_frozen as eval8
from paper.scripts import evaluate_titantpp_components_inter62_frozen as eval4
B24 = PROJECT / 'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1'
B8 = PROJECT / 'search_artifacts/titantpp_components_inter8_test_20261014_v1'
B4 = PROJECT / 'search_artifacts/titantpp_components_inter62_test_20261014_v1'
REPORT = PROJECT / 'reports/titantpp_components_taxi_raf24_test_20261014_v1'
PRIOR_REPORT = PROJECT / 'reports/titantpp_components_inter62_test_20261014_v1'
SOURCE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
PRIOR_PINS = {
    'analysis': 'de3236b990fa65fbffcf44176604b085f17beec2b2d65d3fc7bb9e9bdf7d60cd',
    'report': '9eed8f11b7788991dde060cb8c12092da2cebd9d31e742cd142e68149072c2b0',
    'receipt': 'd8843fd158289d4167f8a0faad9b244d2e7ebffc211e167bdbaccf7562a8888b',
}
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
NAMES = dict(zip(DATASETS, ('Taxi', 'RAF', 'Intermittent')))
ARMS = ('NONE', 'CNN', 'GRU', 'FULL')
SEEDS = (42, 52, 62)
LABELS = {'titantpp_cnn_gru54': 'FULL', 'titantpp_cnn_gru54_correction_disabled': 'NONE',
          'titantpp_cnn_gru54_cnn_only': 'CNN', 'titantpp_cnn_gru54_gru_only': 'GRU'}
METRICS = (('전체 RMSE', ('qty_rmse',)), ('전체 MAE', ('qty_mae',)),
           ('큰 수량 RMSE', ('tail', 'qty_rmse')), ('큰 수량 MAE', ('tail', 'qty_mae')),
           ('전체 TimeNLL', ('time_nll',)), ('큰 수량 TimeNLL', ('tail', 'time_nll')))
TAILS = dict(zip(DATASETS, (3449, 200, 187)))
READY_FILES = ('evaluation_results_manifest.json', 'evaluation_results_recovery_receipt.json',
               'evaluation_results_archive.tar.gz', 'pipeline_actual_exit.json',
               'pipeline_complete.json', 'validation_gate.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def confined(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Unsafe artifact name')
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Artifact escapes approved root')
    return path


def metric_value(metrics, path):
    for key in path:
        metrics = metrics[key]
    require(type(metrics) in (int, float) and math.isfinite(metrics), 'Nonfinite or nonnumeric metric')
    return float(metrics)


def verify_manifest(root, manifest):
    files = manifest.get('files')
    require(isinstance(files, dict) and bool(files), 'Missing artifact manifest')
    for relative, expected in files.items():
        require(sha(confined(root, relative)) == expected, 'Artifact SHA changed: ' + relative)
    return len(files)


def ready(root=B24):
    missing = [name for name in READY_FILES if not (Path(root) / name).is_file()]
    return {'status': 'ready' if not missing else 'waiting_for_native_results',
            'missing': missing, 'new_training': False, 'new_inference': False, 'outputs_written': False}


def verify_originals24(root):
    manifest = read(root / 'originals_manifest.json')
    require(manifest.get('all_file_SHA_verified') is True and manifest.get('source131_SHA_verified') is True
            and manifest.get('checkpoint_selected_last_pairs') == 24
            and manifest.get('remote_originals_changed') is False, 'Original24 recovery verification missing')
    require(verify_manifest(root, manifest) == 360, 'Expected exact360 original files')
    require(sha(root / 'originals_archive_verified.tar.gz') == manifest['archive_raw_sha256'],
            'Verified original archive changed')
    return {'file_count': 360, 'manifest_sha256': sha(root / 'originals_manifest.json'),
            'archive_sha256': sha(root / 'originals_archive_verified.tar.gz')}


def verify_bundle(root, evaluator, count, host, expected_result_files):
    c = evaluator.verify_contract(root)
    require(len(c['jobs']) == count and set(c['runtimes']) == {host}
            and c['sources']['components']['closure'] == SOURCE, 'Foreign cohort/source/native host')
    manifest = read(root / 'evaluation_results_manifest.json')
    recovery = read(root / 'evaluation_results_recovery_receipt.json')
    require(recovery.get('status') == 'complete'
            and sha(root / 'evaluation_results_archive.tar.gz') == recovery.get('archive_sha256')
            and sha(root / 'evaluation_results_manifest.json') == recovery.get('manifest_sha256'),
            'Result archive/manifest recovery differs')
    require(verify_manifest(root, manifest) == expected_result_files, 'Result manifest file count differs')
    require(manifest.get('contract_sha256') == c['contract_sha256'], 'Result manifest contract differs')
    actual = read(root / 'pipeline_actual_exit.json')
    require(actual.get('actual_child_exit_observed') is True and type(actual.get('exit_code')) is int
            and actual['exit_code'] == 0, 'Actual native pipeline exit0 missing')
    complete = read(root / 'pipeline_complete.json')
    expected_completions = {(split, job['id']) for split in ('validation', 'test') for job in c['jobs']}
    completed = complete.get('completed', [])
    require(complete.get('status') == 'complete' and complete.get('contract_sha256') == c['contract_sha256']
            and complete.get('new_training') is False and len(completed) == count * 2
            and {tuple(item) for item in completed} == expected_completions, 'Pipeline incomplete/duplicate conditions')
    evaluator.verify_global_validation_gate(root, c)
    gate = read(root / 'validation_gate.json')
    rows = []
    for j in c['jobs']:
        require(j['evaluation_host'] == j['training_host'] == host, 'Borrowed model Runtime')
        require(sha(confined(root, j['checkpoint']['relative_path'])) == j['checkpoint']['sha256'], 'Selected binary changed')
        for split in ('validation', 'test'):
            folder = root / 'results' / j['id']
            path = folder / ('validation_replay.json' if split == 'validation' else 'test_metrics.json')
            o = read(path)
            x = read(folder / (split + '_worker_exit.json'))
            require(x.get('actual_child_exit_observed') is True and type(x.get('exit_code')) is int
                    and x['exit_code'] == 0 and x.get('job_id') == j['id']
                    and x.get('contract_sha256') == c['contract_sha256'], 'Scientific evaluation worker actual exit0 missing')
            require(o.get('status') == 'complete' and o.get('schema') == evaluator.KIND
                    and o.get('id') == o.get('job_id') == j['id'] and o.get('split') == split
                    and o.get('dataset') == j['dataset'] and o.get('cohort') == evaluator.COHORT
                    and o.get('contract_sha256') == c['contract_sha256']
                    and o.get('selected_epoch') == j['selected_epoch']
                    and o.get('checkpoint_sha256') == j['checkpoint']['sha256']
                    and o.get('state_sha256') == j['checkpoint_state_sha256']
                    and o.get('source_closure') == SOURCE and o.get('frozen_source_closure') == SOURCE,
                    'Selected scientific result identity differs')
            require(o.get('retrained') is False and o.get('reselected') is False
                    and o.get('raw_predictions_written') is False and o.get('native_runtime_checked') is True
                    and o.get('test_independent') is False, 'Unapproved result interpretation')
            require(set(o.get('qualification', {})) == evaluator.QUALIFICATION_KEYS
                    and all(flag is True for flag in o['qualification'].values()), 'Causal/batch/state qualification failed')
            require(o['runtime'] == o['runtime_after'], 'Runtime changed during evaluation')
            evaluator.check_runtime(o['runtime'], c['runtimes'][host], 'cuda', current_process=False)
            population = c['datasets'][j['dataset']][split + '_population']
            require(o['metrics']['count'] == o['prediction_rows'] == population['count'], 'Target population count changed')
            for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256'):
                require(o.get(key) == population[key], 'Target population digest changed: ' + key)
            if split == 'validation':
                evaluator.verify_validation_receipt(o, c, j)
            else:
                require(o.get('held_out_test_evaluated') is True
                        and o.get('validation_gate_sha256') == sha(root / 'validation_gate.json')
                        and type(o.get('test_started_unix')) in (int, float)
                        and o['test_started_unix'] >= gate['created_unix'], 'Test started before qualified Validation gate')
            rows.append({'dataset': j['dataset'], 'split': split, 'seed': j['seed'], 'arm': LABELS[j['arm']],
                'id': j['id'], 'host': host, 'selected_epoch': j['selected_epoch'],
                'checkpoint_sha256': j['checkpoint']['sha256'], 'state_sha256': j['checkpoint_state_sha256'],
                'contract_sha256': c['contract_sha256'], 'receipt_sha256': sha(path),
                'metrics': {name: metric_value(o['metrics'], path) for name, path in METRICS},
                'count': o['metrics']['count'], 'tail_count': o['metrics']['tail']['count'],
                'simple_quantity_baselines': o['simple_quantity_baselines'],
                'population': {key: o[key] for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256')},
                'elapsed_seconds': metric_value(o, ('elapsed_seconds',)),
                'inference_seconds': metric_value(o, ('inference_seconds',)),
                'parameter_counts': o['physical_parameter_counts']})
    return c, rows


def verify_prior_inter12():
    paths = {'analysis': PRIOR_REPORT / 'THREE_SEED_ANALYSIS.json',
             'report': PRIOR_REPORT / 'THREE_SEED_ANALYSIS.md', 'receipt': B4 / 'final_receipt.json'}
    for key, path in paths.items():
        require(sha(path) == PRIOR_PINS[key], 'Pinned previous Inter12 ' + key + ' changed')
    prior = read(paths['analysis'])
    receipt = read(paths['receipt'])
    require(prior.get('status') == receipt.get('status') == 'complete'
            and prior.get('test_job_count') == receipt.get('test_job_count') == 12
            and prior.get('seed_count') == receipt.get('seed_count') == 3
            and prior.get('prior_Test8_reexecuted') is False and prior.get('new_training') is False
            and prior.get('test_independent') is False
            and receipt.get('analysis_sha256') == PRIOR_PINS['analysis']
            and receipt.get('report_sha256') == PRIOR_PINS['report'], 'Prior Inter12 scope differs')
    c8, rows8 = verify_bundle(B8, eval8, 8, '5090', 86)
    c4, rows4 = verify_bundle(B4, eval4, 4, '5080', 46)
    for key in ('data_sha256', 'split_manifest_sha256', 'tail_threshold', 'definition', 'validation_population', 'test_population'):
        require(c8['datasets']['intermittent_frozen_5000'][key] == c4['datasets']['intermittent_frozen_5000'][key],
                'Inter12 cross-seed fixed input differs: ' + key)
    rows = rows8 + rows4
    require(len(prior['rows']) == len(rows) == 24, 'Prior Inter12 Validation/Test row count differs')
    lookup = {(r['split'], r['id']): r for r in prior['rows']}
    require(len(lookup) == 24, 'Duplicate prior Inter12 row')
    for r in rows:
        p = lookup[r['split'], r['id']]
        for key in ('metrics', 'selected_epoch', 'checkpoint_sha256', 'state_sha256', 'receipt_sha256',
                    'contract_sha256', 'simple_quantity_baselines', 'parameter_counts'):
            require(r[key] == p[key], 'Reused prior Inter12 row differs: ' + key)
    return rows


def summarize(rows):
    lookup = {}
    for row in rows:
        key = row['dataset'], row['split'], row['seed'], row['arm']
        require(key not in lookup, 'Duplicate scientific evaluation ID')
        require(row['dataset'] in DATASETS and row['split'] in ('validation', 'test')
                and row['seed'] in SEEDS and row['arm'] in ARMS, 'Foreign scientific evaluation ID')
        require(row['id'] == f"{row['dataset']}__{row['seed']}__{next(k for k,v in LABELS.items() if v == row['arm'])}",
                'Campaign ID/structure differs')
        for name, _ in METRICS:
            metric_value(row['metrics'], (name,))
        lookup[key] = row
    expected = {(d, split, seed, arm) for d in DATASETS for split in ('validation', 'test') for seed in SEEDS for arm in ARMS}
    require(set(lookup) == expected and len(rows) == 72, 'Need exact36 unique fits with Validation36/Test36')
    summary, paired, wins, baselines, counts, populations = {}, [], [], {}, {}, {}
    for d in DATASETS:
        summary[d], baselines[d], populations[d], counts[d] = {}, {}, {}, {}
        for split in ('validation', 'test'):
            scoped = [lookup[d, split, seed, arm] for seed in SEEDS for arm in ARMS]
            require(all(r['simple_quantity_baselines'] == scoped[0]['simple_quantity_baselines'] for r in scoped), 'Deterministic baseline differs')
            require(all(r['population'] == scoped[0]['population'] and r['count'] == scoped[0]['count']
                        and r['tail_count'] == scoped[0]['tail_count'] for r in scoped), 'Cross-arm/seed target population differs')
            baselines[d][split] = scoped[0]['simple_quantity_baselines']
            populations[d][split] = {'count': scoped[0]['count'], 'tail_count': scoped[0]['tail_count'], **scoped[0]['population']}
            summary[d][split] = {arm: {name: {'mean': statistics.mean(lookup[d, split, seed, arm]['metrics'][name] for seed in SEEDS),
                'sample_sd': statistics.stdev(lookup[d, split, seed, arm]['metrics'][name] for seed in SEEDS), 'n': 3}
                for name, _ in METRICS} for arm in ARMS}
            for seed in SEEDS:
                full = lookup[d, split, seed, 'FULL']
                for arm in ('NONE', 'CNN', 'GRU'):
                    other = lookup[d, split, seed, arm]
                    paired.append({'dataset': d, 'split': split, 'seed': seed, 'comparison': arm,
                        'FULL_minus_comparator': {name: full['metrics'][name] - other['metrics'][name] for name, _ in METRICS}})
            for arm in ('NONE', 'CNN', 'GRU'):
                for name, _ in METRICS:
                    deltas = [lookup[d, split, seed, 'FULL']['metrics'][name] - lookup[d, split, seed, arm]['metrics'][name] for seed in SEEDS]
                    comparator = summary[d][split][arm][name]['mean']
                    full = summary[d][split]['FULL'][name]['mean']
                    wins.append({'dataset': d, 'split': split, 'comparison': arm, 'metric': name,
                        'FULL_mean': full, 'comparator_mean': comparator, 'FULL_minus_comparator_mean': full - comparator,
                        'FULL_improvement_percent': 100 * (comparator - full) / comparator if comparator else None,
                        'strict_wins': sum(x < 0 for x in deltas), 'strict_losses': sum(x > 0 for x in deltas), 'ties': sum(x == 0 for x in deltas), 'n': 3})
        for arm in ARMS:
            specs = [lookup[d, split, seed, arm]['parameter_counts'] for split in ('validation', 'test') for seed in SEEDS]
            require(all(spec == specs[0] for spec in specs), 'Cross-seed/phase parameter count differs')
            expected_counts = dict(zip(ARMS, (78851, 79427, 103583, 104159) if d == 'raf_spare_parts' else (89859, 90435, 114591, 115167)))
            expected_registered = 104159 if d == 'raf_spare_parts' else 115167
            require(specs[0]['actual_registered_parameters'] == expected_registered
                    and specs[0]['actual_active_parameters_before_evaluation_freeze'] == expected_counts[arm], 'Physical structure parameter count differs')
            counts[d][arm] = specs[0]
    flips = []
    for d in DATASETS:
        for seed in SEEDS:
            for arm in ('NONE', 'CNN', 'GRU'):
                for name, _ in METRICS:
                    val = lookup[d, 'validation', seed, 'FULL']['metrics'][name] - lookup[d, 'validation', seed, arm]['metrics'][name]
                    test = lookup[d, 'test', seed, 'FULL']['metrics'][name] - lookup[d, 'test', seed, arm]['metrics'][name]
                    flips.append({'dataset': d, 'seed': seed, 'comparison': arm, 'metric': name,
                        'Validation_FULL_minus_comparator': val, 'Test_FULL_minus_comparator': test,
                        'strict_direction_reversal': val * test < 0})
    return {'summary_mean_sample_sd': summary, 'seed_paired_differences': paired, 'FULL_metric_comparisons': wins,
            'simple_quantity_baselines': baselines, 'physical_parameter_counts': counts,
            'population_bindings': populations, 'Validation_Test_direction_comparisons': flips}


def markdown(out):
    f = lambda value: f'{value:.6g}'
    stats = out['summary_mean_sample_sd']
    lines = ['# CNN·GRU 구성요소36개 최종 Validation/Test 비교', '',
      'Material Passport: source131 구성요소 캠페인만 포함한다. 네 구조 × Taxi·RAF·Intermittent × seed42·52·62의 선택 checkpoint36개이며, 전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch를 고정했다. 이전 대표 모델·이력 보정27·Instacart9의 FULL과 합치거나 교체하지 않는다. 새 학습·checkpoint 재선택·기존 Inter12 Test 재실행은 없다. 기존 접근 이력이 있는 chronological Test이며 새 독립 평가로 부르지 않는다.', '',
      '여섯 지표는 같은 선택 모델에서 계산한다. 큰 수량은 Taxi q>3449, RAF q>200, Intermittent q>187이다. 평균±표본SD는 seed3개의 기술통계이며 통계적 유의성을 뜻하지 않는다. 단순 기준선은 동일 대상의 직전 수량 반복·관측 이력 평균이고 TimeNLL은 N/A다.', '']
    for split in ('validation', 'test'):
        lines += ['## '+split+' 평균 ± 표본SD (n=3)', '', '| 데이터셋 | 구조 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | 전체 TimeNLL | 큰 수량 TimeNLL |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for d in DATASETS:
            for arm in ARMS:
                lines.append('| '+NAMES[d]+' | '+arm+' | '+' | '.join(f(stats[d][split][arm][name]['mean'])+' ± '+f(stats[d][split][arm][name]['sample_sd']) for name, _ in METRICS)+' |')
        lines += ['', '| 데이터셋 | 기준선 | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL |', '|---|---|---:|---:|---:|---:|---:|']
        for d in DATASETS:
            for key, label in (('last', '직전 수량 반복'), ('mean', '관측 이력 평균')):
                baseline = out['simple_quantity_baselines'][d][split][key]
                lines.append('| '+NAMES[d]+' | '+label+' | '+' | '.join(f(metric_value(baseline, path)) for _, path in METRICS[:4])+' | N/A |')
        lines += ['']
    lines += ['## FULL의 전체 raw 수량 RMSE 판정', '', '| 데이터셋 | Validation 최저 평균 구조 | Test 최저 평균 구조 | 비교군 | Test FULL 변화율 | FULL seed별 개선 |', '|---|---|---|---|---:|---:|']
    for d in DATASETS:
        best_val = min(ARMS, key=lambda arm: stats[d]['validation'][arm]['전체 RMSE']['mean'])
        best_test = min(ARMS, key=lambda arm: stats[d]['test'][arm]['전체 RMSE']['mean'])
        for item in out['FULL_metric_comparisons']:
            if item['dataset'] == d and item['split'] == 'test' and item['metric'] == '전체 RMSE':
                change = -item['FULL_improvement_percent']
                lines.append(f"| {NAMES[d]} | {best_val} | {best_test} | {item['comparison']} | {change:+.3f}% | {item['strict_wins']}/3 |")
    lines += ['', '변화율이 음수면 FULL 개선이다. 평균 우세와 모든 seed의 우세를 구분한다. Validation의 이득이 Test에서 유지되지 않는 데이터셋은 결합 구조의 일반적인 우월성 근거로 사용하지 않는다.', '',
      '## Seed별 선택 모델과 여섯 지표', '', '| 데이터셋 | split | seed | 구조 | 선택 epoch | 전체 RMSE | 전체 MAE | 큰 수량 RMSE | 큰 수량 MAE | 전체 TimeNLL | 큰 수량 TimeNLL |', '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(out['rows'], key=lambda r:(DATASETS.index(r['dataset']), r['split'], r['seed'], ARMS.index(r['arm']))):
        lines.append('| '+NAMES[r['dataset']]+' | '+r['split']+' | '+str(r['seed'])+' | '+r['arm']+' | '+str(r['selected_epoch'])+' | '+' | '.join(f(r['metrics'][name]) for name, _ in METRICS)+' |')
    lines += ['', '## 여섯 지표의 seed별 비교와 Validation/Test 방향 반전', '', '| 데이터셋 | 비교군 | 지표 | Validation FULL 개선 | Test FULL 개선 | 방향 반전 seed |', '|---|---|---|---:|---:|---:|']
    lookup = {(x['dataset'],x['split'],x['comparison'],x['metric']):x for x in out['FULL_metric_comparisons']}
    for d in DATASETS:
        for arm in ('NONE','CNN','GRU'):
            for name, _ in METRICS:
                v,t=lookup[d,'validation',arm,name],lookup[d,'test',arm,name]
                flips=sum(x['strict_direction_reversal'] for x in out['Validation_Test_direction_comparisons'] if x['dataset']==d and x['comparison']==arm and x['metric']==name)
                lines.append(f"| {NAMES[d]} | {arm} | {name} | {v['strict_wins']}/3 | {t['strict_wins']}/3 | {flips}/3 |")
    lines += ['', '## 파라미터 수와 Runtime 범위', '', '| 데이터셋 | 구조 | 등록 파라미터 | 실제 활성 파라미터 | 최대 시퀀스 길이 |', '|---|---|---:|---:|---:|']
    for d in DATASETS:
        for arm in ARMS:
            c=out['physical_parameter_counts'][d][arm]
            lines.append(f"| {NAMES[d]} | {arm} | {c['actual_registered_parameters']} | {c['actual_active_parameters_before_evaluation_freeze']} | {c['maximum_sequence_length']} |")
    lines += ['', '활성 파라미터 수가 서로 달라 동일 파라미터 예산의 대조는 아니다. RAF의 최대 길이는84, Taxi·Intermittent는256이다. Taxi·RAF 전체와 Intermittent62는5080/cuDNN91900이며 Intermittent42·52는5090/cuDNN92000이다. Intermittent의 seed 간 GPU·Runtime 차이를 보존하고 추론 시간은 host별로 분리한다.', '',
      '## 재사용과 남은 원고 작업', '', '**완료 결과를 캠페인별로 원고에 연결한다 — 다음 작업**',
      '- 구성요소36의 결과는 이 표를 사용한다. 기존 대표9, 이력 보정27, Instacart9는 각 원본 결과표를 별도로 사용한다.',
      '- Test의 반전과 큰 수량·시간 지표의 불리한 결과를 함께 기록하고, 데이터셋별로 기여 범위를 제한한다.',
      '- 교수님 보고용 main.reviewed.tex와 별도 PAKDD12페이지 원고를 다음 단계에서 정리한다. 이번 집계는 원고를 변경하지 않는다.']
    return '\n'.join(lines)+'\n'


def write_json_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-ready', action='store_true')
    args=parser.parse_args(argv)
    availability=ready()
    if args.check_ready or availability['status'] != 'ready':
        print(json.dumps(availability,ensure_ascii=False));return availability
    originals=verify_originals24(B24)
    contract24,rows24=verify_bundle(B24,eval24,24,'5080',246)
    inter12=verify_prior_inter12()
    rows=rows24+inter12
    analysis={'schema':'titantpp_components36_source131_final_Validation_Test_v1','status':'complete',
        'source131_closure':SOURCE,'test_job_count':36,'validation_job_count':36,'new_test_job_count':24,
        'reused_test_job_count':12,'seed_count':3,'seeds':list(SEEDS),'dataset_count':3,
        'new_training':False,'prior_Inter12_Test_reexecuted':False,'checkpoint_reselection':False,
        'test_independent':False,'universal_claim':False,'rows':rows,**summarize(rows),
        'originals24':originals,'prior_Inter12_pins':PRIOR_PINS,
        'contracts':{'Taxi_RAF24':contract24['contract_sha256'],
                     'Inter42_52':read(B8/'contract.json')['contract_sha256'],
                     'Inter62':read(B4/'contract.json')['contract_sha256']},
        'cross_seed_Runtime':{'Taxi_RAF42_52_62':'5080 / cuDNN91900',
                             'Intermittent42_52':'5090 / cuDNN92000','Intermittent62':'5080 / cuDNN91900'},
        'runtime_timing_pooled':False,'host_timing_rows':[{'dataset':r['dataset'],'seed':r['seed'],
             'arm':r['arm'],'split':r['split'],'host':r['host'],'elapsed_seconds':r['elapsed_seconds'],
             'inference_seconds':r['inference_seconds']} for r in rows],
        'other_campaign_FULL_pooled_or_substituted':False,'recorded_unix':time.time()}
    paths=(REPORT/'COMPONENTS36_ANALYSIS.json',REPORT/'COMPONENTS36_ANALYSIS.md',B24/'components36_receipt.json')
    require(all(not p.exists() for p in paths),'Existing final output prohibits automatic overwrite/repeat')
    write_json_exclusive(paths[0],analysis)
    with paths[1].open('x',encoding='utf-8') as stream:stream.write(markdown(analysis))
    receipt={'status':'complete','source131_closure':SOURCE,'test_job_count':36,'validation_job_count':36,
        'new_native5080_Test24_actual_exit0':True,'new_native5080_Validation24_PASS':True,
        'actual_native_worker_exit0_count_new24':48,'prior_Inter12_Test_reexecuted':False,
        'new_training':False,'checkpoint_reselection':False,'test_independent':False,
        'other_campaign_FULL_pooled_or_substituted':False,'analysis_sha256':sha(paths[0]),'report_sha256':sha(paths[1]),
        'new24_archive_sha256':sha(B24/'evaluation_results_archive.tar.gz'),
        'new24_manifest_sha256':sha(B24/'evaluation_results_manifest.json'),
        'originals24':originals,'prior_Inter12_pins':PRIOR_PINS,
        'completed_KST':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat()}
    write_json_exclusive(paths[2],receipt)
    print(json.dumps({'status':'complete','test_job_count':36,'new_test_job_count':24,
                     'reused_test_job_count':12,'report_path':str(paths[1]),'receipt_sha256':sha(paths[2])},ensure_ascii=False))
    return receipt


if __name__=='__main__':main()
