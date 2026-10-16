"""Build a local, split-separated comparison from sealed aggregate evidence.

No training, inference, GPU, SSH, dataset access, or checkpoint deserialization.
Historical single-seed A100 rows are filtered before numeric or source access.
"""
from __future__ import annotations

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
PRIOR = ROOT / 'reports/titantpp_completed_validation_test_20261004_v1'
WIDTH = ROOT / 'reports/titantpp_width8_12_test_20261004_v1'
DATASETS = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent', 'raf_spare_parts': 'RAF'}
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')
SEEDS = {42, 52, 62}
WIDTHS = ['titantpp_history_mlp', 'titantpp_history_mlp_width8', 'titantpp_history_mlp_width12', 'titantpp_history_mlp_width16']
BASELINE = WIDTHS[-1]
A100 = {'titantpp_history_cross_product', 'titantpp_history_recent4_mean', 'titantpp_history_recent4_attention', 'titantpp_history_post_block'}
SIMPLE = {'last_observed_quantity', 'mean_quantity_in_same_observed_window'}
ORDER = WIDTHS + ['titantpp', 'rmtpp', 'thp', 'nhp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head', 'titantpp_all_available_history_mlp', 'titantpp_current_only_param_matched', 'deep_renewal_event_native_nb', 'last_observed_quantity', 'mean_quantity_in_same_observed_window']
THRESHOLDS = {'yellow_trip_hourly': 3449.0, 'intermittent_frozen_5000': 187.0, 'raf_spare_parts': 200.0}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    if any('a100' in part.lower() for part in path.parts):
        raise ValueError(f'A100 source access outside this report scope: {path}')
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-5)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


REFERENCE = module(HERE / 'analyze_reference.py', 'final_reference')
LABELS = dict(REFERENCE.LABELS)
LABELS[BASELINE] += ' · 사용자 기준선'


def write_csv(path, rows):
    require(bool(rows), 'Empty output CSV')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(row):
    result = REFERENCE.numeric(row)
    result['selected_epoch'] = int(row['selected_epoch']) if row.get('selected_epoch') not in ('', None) else None
    result['provenance'] = row.get('provenance') or 'full_population_aggregate_receipt'
    result['time_nll_definition'] = row.get('time_nll_definition') or 'recorded_positive_integer_lognormal_round_clamp_mass_nll'
    result['threshold_rule'] = 'raw_quantity > threshold' if result['view'] == 'tail' else None
    return result


def check_coverage(rows):
    keys = [(r['dataset'], r['model'], r['seed'], r['split'], r['view']) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate condition/split/view')
    for dataset in DATASETS:
        expected_models = set(ORDER) - ({'titantpp'} if dataset == 'raf_spare_parts' else set())
        require({r['model'] for r in rows if r['dataset'] == dataset} == expected_models, f'Model coverage differs: {dataset}')
        for model in expected_models:
            members = [r for r in rows if r['dataset'] == dataset and r['model'] == model]
            expected_seeds = {None} if model in SIMPLE else SEEDS
            expected = {(seed, split, view) for seed in expected_seeds for split in ('validation', 'test') for view in ('overall', 'tail')}
            actual = {(r['seed'], r['split'], r['view']) for r in members}
            require(actual == expected, f'Incomplete population/seed coverage: {dataset}/{model}')
            for seed in expected_seeds:
                require(len({r['selected_epoch'] for r in members if r['seed'] == seed}) == 1, 'Selected epoch changed across split or view')
        for split in ('validation', 'test'):
            for view in ('overall', 'tail'):
                members = [r for r in rows if (r['dataset'], r['split'], r['view']) == (dataset, split, view)]
                require(len({r['count'] for r in members}) == 1, f'Population counts differ: {dataset}/{split}/{view}')
                if view == 'tail':
                    require({r['threshold'] for r in members} == {THRESHOLDS[dataset]}, 'Tail is not the fixed TRAIN threshold')


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['model'], row['split'], row['view']), []).append(row)
    result = []
    for (dataset, model, split, view), members in groups.items():
        record = {'dataset': dataset, 'model': model, 'split': split, 'view': view, 'seed_count': 0 if model in SIMPLE else len(members), 'count_per_seed': members[0]['count'], 'threshold': members[0]['threshold']}
        for metric in METRICS:
            values = [r[metric] for r in members]
            require(all(v is None for v in values) or all(v is not None for v in values), 'Partial metric coverage')
            record[metric + '_mean'] = statistics.mean(values) if values[0] is not None else None
            record[metric + '_sample_sd'] = statistics.stdev(values) if values[0] is not None and len(values) > 1 else None
        result.append(record)
    return result


def paired(rows):
    lookup = {(r['dataset'], r['model'], r['seed'], r['split'], r['view']): r for r in rows}
    result, raw = [], []
    for dataset in DATASETS:
        models = [model for model in ORDER if model not in SIMPLE and model != BASELINE and (dataset, model, 42, 'validation', 'overall') in lookup]
        for split in ('validation', 'test'):
            for view in ('overall', 'tail'):
                for model in models:
                    for metric in METRICS:
                        pairs = [(seed, lookup[(dataset, model, seed, split, view)][metric], lookup[(dataset, BASELINE, seed, split, view)][metric]) for seed in sorted(SEEDS)]
                        deltas = [v - b for seed, v, b in pairs]
                        require(all(math.isfinite(v) for v in deltas), 'Nonfinite paired delta')
                        worst = max(pairs, key=lambda x: x[1])
                        worst_delta = max(zip(pairs, deltas), key=lambda x: x[1])
                        base_mean = statistics.mean(p[2] for p in pairs)
                        result.append({'dataset': dataset, 'model': model, 'baseline': BASELINE, 'split': split, 'view': view, 'metric': metric, 'seed_count': len(pairs), 'baseline_mean': base_mean, 'delta_mean': statistics.mean(deltas), 'delta_sample_sd': statistics.stdev(deltas), 'relative_mean_change_percent': 100 * statistics.mean(deltas) / base_mean if base_mean != 0 else None, 'improved_seeds': sum(v < 0 for v in deltas), 'worsened_seeds': sum(v > 0 for v in deltas), 'unchanged_seeds': sum(v == 0 for v in deltas), 'worst_metric_seed': worst[0], 'worst_metric_value': worst[1], 'worst_delta_seed': worst_delta[0][0], 'worst_delta': worst_delta[1]})
                        for (seed, value, baseline), delta in zip(pairs, deltas):
                            raw.append({'dataset': dataset, 'model': model, 'baseline': BASELINE, 'seed': seed, 'split': split, 'view': view, 'metric': metric, 'model_value': value, 'baseline_value': baseline, 'delta': delta, 'relative_change_percent': 100 * delta / baseline if baseline != 0 else None})
    return result, raw


def collect_and_verify():
    reference_path = PRIOR / 'results/metrics_per_seed.csv'
    prior_receipt = read(PRIOR / 'comparison_receipt.json')
    require(prior_receipt['status'] == 'complete' and sha(reference_path) == prior_receipt['outputs']['results/metrics_per_seed.csv'], 'Original comparison SHA/status mismatch')
    with reference_path.open() as stream:
        old = [normalize(row) for row in csv.DictReader(stream) if row['model'] not in A100 and row['dataset'] in DATASETS and row['split'] in ('validation', 'test') and row['view'] in ('overall', 'tail')]
    new_path = HERE / 'test_analysis/new_width_metrics_per_seed.csv'
    test_receipt = read(HERE / 'test_analysis/analysis_receipt.json')
    require(sha(new_path) == test_receipt['outputs'][new_path.name], 'Derived new metrics SHA mismatch')
    with new_path.open() as stream:
        new = [normalize(row) for row in csv.DictReader(stream)]
    rows = old + new
    check_coverage(rows)
    sources = {str(reference_path.relative_to(ROOT)): sha(reference_path), str((PRIOR / 'comparison_receipt.json').relative_to(ROOT)): sha(PRIOR / 'comparison_receipt.json')}
    registries, populations, new_registry = {}, {}, {}
    for campaign in [WIDTH / 'taxi_raf_attempt3', WIDTH / 'intermittent']:
        contract = read(campaign / 'execution_contract.json')
        for dataset, values in contract['population_references'].items():
            populations[dataset] = values
        registry = read(campaign / 'evaluation_registry.json')
        new_registry.update({(r['dataset'], r['model'], int(r['seed'])): r for r in registry['rows']})
        for name in ('execution_contract.json', 'evaluation_registry.json', 'selected_binding.json', 'validation_gate.json', 'inference_completion.json'):
            path = campaign / name
            sources[str(path.relative_to(ROOT))] = sha(path)
        require(sha(campaign / 'evaluation_registry.json') == contract['registry_sha256'], 'Registry SHA mismatch')
        require(sha(campaign / 'selected_binding.json') == contract['selected_binding_sha256'], 'Selected binding SHA mismatch')
    for path in [ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json', ROOT / 'reports/titantpp_b_test_evaluation_20261003_v1/evaluation_registry.json', PRIOR / 'width16/evaluation_registry.json', PRIOR / 'deep_renewal/evaluation_registry.json']:
        sources[str(path.relative_to(ROOT))] = sha(path)
        registries.update({(r['dataset'], r['model'], int(r['seed'])): r for r in read(path)['rows'] if r['dataset'] in DATASETS})
    registries.update(new_registry)
    binding_path = PRIOR / 'existing/source_bindings.json'
    bindings = read(binding_path)
    sources[str(binding_path.relative_to(ROOT))] = sha(binding_path)
    binding_lookup = {(r['dataset'], r['model'], int(r['seed'])): r for r in bindings['selected_endpoint_records'] if r['dataset'] in DATASETS}
    evidence, selected = [], {}
    receipt_cache = {}
    test_rows = {(r['dataset'], r['model'], r['seed']): r for r in old if r['split'] == 'test' and r['view'] == 'overall'}
    for row in rows:
        source = ROOT / row['source']
        require(not any('a100' in p.lower() for p in source.parts), 'Forbidden A100 source')
        require(sha(source) == row['source_sha256'], f'Source SHA mismatch: {source}')
        sources[row['source']] = row['source_sha256']
        if row['source'] not in receipt_cache:
            receipt_cache[row['source']] = read(source)
        receipt = receipt_cache[row['source']]
        population = populations[row['dataset']][row['split']]
        identity = (row['dataset'], row['model'], row['seed'])
        registry = registries.get(identity)
        if receipt.get('full_population'):
            require(receipt['status'] == 'complete' and receipt['split'] == row['split'], 'Full population status/split mismatch')
            for field in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'expected_target_count'):
                require(receipt[field] == population[field], f'Population/history mismatch: {row["source"]} {field}')
            require(receipt['prediction_rows'] == population['expected_target_count'], 'Full population row count mismatch')
            if registry:
                for field in ('selected_epoch', 'checkpoint_file_sha256', 'state_tensor_sha256'):
                    require(receipt[field] == registry[field], f'Checkpoint binding mismatch: {field}')
                require(row['selected_epoch'] == registry['selected_epoch'], 'CSV selected epoch differs')
            if 'metrics' in receipt and row['model'] not in SIMPLE:
                metrics = receipt['metrics'] if row['view'] == 'overall' else receipt['metrics']['tail']
                require(metrics['count'] == row['count'], 'Receipt metric count mismatch')
                for metric in METRICS:
                    require(near(metrics[metric], row[metric]), 'Receipt/CSV metric mismatch')
            kind = 'direct_full_population_receipt_SHA_target_truth_loader_verified'
            scientific_receipt = receipt
        else:
            require(row['split'] == 'validation' and 'selected' in receipt, 'Source is neither full receipt nor frozen Validation endpoint')
            endpoint = receipt['selected']
            require(endpoint['evaluation_scope'] == 'validation_only' and endpoint['held_out_test_evaluated'] is False, 'Endpoint split mismatch')
            require(receipt['best_epoch'] == row['selected_epoch'] == registry['selected_epoch'], 'Frozen Validation selected epoch mismatch')
            require(endpoint['state_sha256'] == registry['state_tensor_sha256'], 'Frozen Validation state mismatch')
            bound = binding_lookup[identity]
            require(bound['source'] == row['source'] and bound['source_sha256'] == row['source_sha256'] and bound['selected_state_binding'], 'Prior selected endpoint binding mismatch')
            metric_record = endpoint if row['view'] == 'overall' else endpoint['tail']
            require(metric_record['count'] == row['count'], 'Endpoint count mismatch')
            for metric in METRICS:
                require(near(metric_record[metric], row[metric]), 'Endpoint/CSV metric mismatch')
            require(endpoint['count'] == registry.get('validation_count', population['expected_target_count']) == population['expected_target_count'], 'Validation population size mismatch')
            require(endpoint['quantity_boundaries'][-1] == THRESHOLDS[row['dataset']], 'Endpoint tail threshold mismatch')
            counterpart = test_rows[identity]
            scientific_receipt = read(ROOT / counterpart['source'])
            require(scientific_receipt['state_tensor_sha256'] == endpoint['state_sha256'], 'Validation/Test checkpoint identity differs')
            for field in ('loader', 'data_file_sha256'):
                require(scientific_receipt[field] == population[field], 'Frozen endpoint loader/data history differs')
            require(scientific_receipt['source_closure_sha256'] == registry['source_closure_sha256'], 'Frozen source closure differs')
            kind = 'frozen_Validation_endpoint_SHA_state_count_tail_verified; target_truth_from_shared_original_population_reference_not_endpoint_digest'
        if registry:
            require(scientific_receipt['selected_epoch'] == row['selected_epoch'], 'Split selection differs')
            selected[identity] = {'dataset': row['dataset'], 'model': row['model'], 'seed': row['seed'], 'selected_epoch': row['selected_epoch'], 'checkpoint_file_sha256': scientific_receipt['checkpoint_file_sha256'], 'state_tensor_sha256': scientific_receipt['state_tensor_sha256'], 'source_closure_sha256': scientific_receipt['source_closure_sha256'], 'original_source_revision': registry.get('original_source_revision') or registry.get('source_revision') or 'not_explicit_in_registry; frozen_source_closure_recorded', 'checkpoint_path': registry['checkpoint_path']}
        evidence.append({'dataset': row['dataset'], 'model': row['model'], 'seed': row['seed'], 'split': row['split'], 'view': row['view'], 'source': row['source'], 'source_sha256': row['source_sha256'], 'verification': kind})
    for dataset, values in populations.items():
        for split, population in values.items():
            path = ROOT / population['source_receipt']
            require(sha(path) == population['source_receipt_sha256'], 'Original population reference SHA changed')
            receipt = read(path)
            for field in ('target_identity_sha256', 'truth_sha256', 'loader', 'data_file_sha256', 'expected_target_count'):
                require(receipt[field] == population[field], 'Original population reference differs')
            sources[str(path.relative_to(ROOT))] = sha(path)
    require(len(selected) == 123, 'Expected 123 learned conditions')
    return rows, sources, evidence, list(selected.values()), populations


def fmt(value, metric, signed=False):
    if value is None:
        return '—'
    if metric == 'time_nll' and value != 0 and abs(value) < 0.0001:
        return f'{value:+.3e}' if signed else f'{value:.3e}'
    precision = 4 if metric == 'time_nll' else 3
    return f'{value:+.{precision}f}' if signed else f'{value:.{precision}f}'


def cell(row, metric):
    if row is None:
        return '—'
    mean, sd = row[metric + '_mean'], row[metric + '_sample_sd']
    return fmt(mean, metric) + (' ± ' + fmt(sd, metric) if sd is not None else '')


def tables(aggregate, populations):
    lookup = {(r['dataset'], r['model'], r['split'], r['view']): r for r in aggregate}
    for split in ('validation', 'test'):
        for view in ('overall', 'tail'):
            filename = f'{split.upper()}_{"TAIL_" if view == "tail" else ""}TABLES.md'
            lines = [f'# {split.title()} · {"전체 사건" if view == "overall" else "TRAIN 기준 큰 수량 tail"}', '', '학습 모델은 seed42·52·62의 산술 평균 ± 표본 표준편차(ddof=1)다. 사건별 원시 수량 오차를 seed 안에서 집계한 값이며 ensemble 결과가 아니다. ↓는 작을수록 좋다는 뜻이다.', '', '사용자 기준선은 MLP 폭16으로 유지한다. 폭4는 기존 비교군이며, 결과 순위에 따라 모델·폭·epoch를 다시 선택하지 않았다.', '']
            for dataset, name in DATASETS.items():
                count = lookup[(dataset, BASELINE, split, view)]['count_per_seed']
                population_n = populations[dataset][split]['expected_target_count']
                scope = f'전체 target {count:,}개' if view == 'overall' else f'raw quantity > {THRESHOLDS[dataset]:g}: {count:,}/{population_n:,} target ({100*count/population_n:.2f}%)'
                lines.extend([f'## {name}', '', scope + '. 모든 비교군과 seed에 동일한 사건 집합을 적용했다.', '', '| 모델 | seed 수 | target 수/seed | RMSE ↓ | MAE ↓ | Time NLL ↓ |', '|---|---:|---:|---:|---:|---:|'])
                for model in ORDER:
                    row = lookup.get((dataset, model, split, view))
                    cells = [LABELS[model], str(row['seed_count']) if row and row['seed_count'] else '—', f'{row["count_per_seed"]:,}' if row else '—'] + [cell(row, metric) for metric in METRICS]
                    lines.append('| ' + ' | '.join(cells) + ' |')
                lines.append('')
            lines.extend(['수량만 예측하는 통계 기준선은 결정적 1회 값이며 seed 평균·SD와 Time NLL이 없다. RAF의 TitanTPP B는 비교 가능한 원본이 없어 —로 표시했다.', '', 'Time NLL은 기록된 양의 정수 간격의 확률질량 점수이며 Taxi는 시간, Intermittent는 주, RAF는 월 단위다. 공통 lognormal 출력부와 Deep Renewal native shifted-NB는 likelihood가 다르므로 출력부를 통제한 backbone 효과로 해석할 수 없다. Deep Renewal은 연구 기록을 보존하며 원고 제외 상태를 유지한다.', '', '소수점 표시는 가독성을 위한 반올림이며 작은 Time NLL은 과학 표기로 보존했다. 원값·SHA·선택 epoch는 [지표 CSV](metrics_per_seed.csv), [출처 영수증](comparison_receipt.json), [선택 기록](selected_epochs.csv)에 있다.', '', '과거 A100 4개 탐색 arm과 현재 A100 CNN/GRU Validation-only 비교는 별도 범위다. [범위와 검증 한계](README.md)를 함께 읽는다.'])
            (HERE / filename).write_text('\n'.join(lines) + '\n')


def paired_tables(deltas, raw):
    for split in ('validation', 'test'):
        lines = [f'# {split.title()} · 같은 seed의 사용자 기준선 폭16 대비 차이', '', 'Δ = 비교 모델 − MLP 폭16. 음수는 개선이고 양수는 악화다. 평균 ± SD는 seed42·52·62의 paired Δ의 평균과 표본 표준편차(ddof=1)다. 개별 모델의 SD를 빼서 계산하지 않았다. 개선 seed 수와 가장 불리한 Δ를 함께 보여 준다.', '', '비율은 평균 Δ / 폭16 평균 × 100이다. Time NLL 기준 평균의 절대값이 1e-6 미만이면 비율은 해석이 불안정하므로 표시만 —(기준≈0)로 하고 절대 Δ·원래 비율·분모를 [paired 요약](paired_deltas_aggregated.csv)에 모두 보존했다. 모든 개별 Δ는 [paired seed 원값](paired_deltas_per_seed.csv)에 있다.', '']
        for dataset, name in DATASETS.items():
            for view in ('overall', 'tail'):
                lines.extend([f'## {name} · {view}', '', '| 모델 | 지표 | paired Δ 평균 ± SD | 평균 변화율 | 개선 seed | 최대 Δ seed / Δ | 최악 성능 seed / 값 | seed42 Δ | seed52 Δ | seed62 Δ |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
                for model in ORDER:
                    for metric in METRICS:
                        row = next((r for r in deltas if (r['dataset'], r['model'], r['split'], r['view'], r['metric']) == (dataset, model, split, view, metric)), None)
                        if row is None:
                            continue
                        values = [next(r['delta'] for r in raw if (r['dataset'], r['model'], r['seed'], r['split'], r['view'], r['metric']) == (dataset, model, seed, split, view, metric)) for seed in sorted(SEEDS)]
                        ratio = '— (기준≈0)' if metric == 'time_nll' and abs(row['baseline_mean']) < 1e-6 else (f'{row["relative_mean_change_percent"]:+.2f}%' if row['relative_mean_change_percent'] is not None else '—')
                        cells = [LABELS[model], {'qty_rmse': 'RMSE', 'qty_mae': 'MAE', 'time_nll': 'Time NLL'}[metric], fmt(row['delta_mean'], metric, True) + ' ± ' + fmt(row['delta_sample_sd'], metric), ratio, f'{row["improved_seeds"]}/3', f'{row["worst_delta_seed"]} / {fmt(row["worst_delta"], metric, True)}', f'{row["worst_metric_seed"]} / {fmt(row["worst_metric_value"], metric)}'] + [fmt(v, metric, True) for v in values]
                        lines.append('| ' + ' | '.join(cells) + ' |')
                lines.append('')
        lines.extend(['통계 기준선에는 paired seed가 없어 이 표의 paired 통계 범위에서 제외했다. 절대 성능은 전체·tail 비교표에 그대로 포함했다. 어느 지표의 개선도 다른 지표·seed·데이터에서의 개선을 뜻하지 않는다.'])
        (HERE / f'{split.upper()}_PAIRED_DELTAS.md').write_text('\n'.join(lines) + '\n')


def report(aggregate, deltas, populations, evidence):
    lookup = {(r['dataset'], r['model'], r['split'], r['view']): r for r in aggregate}
    lines = ['# Taxi·Intermittent·RAF 최종 Validation·Test 비교', '', '**로컬 집계 완료:** 폭4·8·12·16과 기존 벤치마크·구조 대조군 123학습 조건을 연결했다. seed42·52·62의 두 split × 전체·TRAIN 기준 tail은 학습 모델 492지표행이며 결정적 수량 기준선 24행을 더해 총 516행이다.', '', '**MLP 폭16은 사용자 기준선으로 유지한다.** 폭4는 비교군이다. 아래 결과는 이미 접근했던 기존 Test split에 대한 사후 비교이며, 새로운 독립 Test나 Test를 사용한 모델 선택으로 주장하지 않는다.', '', '- [Validation 전체](VALIDATION_TABLES.md) / [Validation tail](VALIDATION_TAIL_TABLES.md) / [Validation paired Δ](VALIDATION_PAIRED_DELTAS.md)', '- [Test 전체](TEST_TABLES.md) / [Test tail](TEST_TAIL_TABLES.md) / [Test paired Δ](TEST_PAIRED_DELTAS.md)', '- [모든 seed 지표](metrics_per_seed.csv) / [평균·표본 SD](metrics_aggregated.csv) / [선택 epoch와 동결 source](selected_epochs.csv)', '', '## 폭4·8·12·16의 전체·tail RMSE', '', '| 데이터 | split | 범위 | 폭4 | 폭8 | 폭12 | 폭16 · 기준선 |', '|---|---|---|---:|---:|---:|---:|']
    for dataset, name in DATASETS.items():
        for split in ('validation', 'test'):
            for view in ('overall', 'tail'):
                lines.append('| ' + ' | '.join([name, split.title(), view] + [cell(lookup[(dataset, model, split, view)], 'qty_rmse') for model in WIDTHS]) + ' |')
    lines.extend(['', '## 폭8·12와 기준선의 이득·손해를 함께 읽기', '', '수량 열의 변화율은 각 split에서 폭16 평균 대비다. 전체·tail Time NLL은 절대 Δ(비교 모델 평균 − 폭16 평균)로 표시한다. 특히 거의 0인 tail Time NLL을 분모로 한 비율은 해석을 왜곡하므로 절대 Δ와 과학 표기를 사용했다. 계산된 모든 비율과 분모는 CSV에 보존했다.', '', '| 데이터 | split | 모델 | 전체 RMSE 변화율 | 전체 MAE 변화율 | 전체 Time NLL Δ | tail RMSE 변화율 | tail MAE 변화율 | tail Time NLL Δ | 전체 RMSE 개선 seed | 전체 RMSE 최대 Δ seed / Δ |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for dataset, name in DATASETS.items():
        for split in ('validation', 'test'):
            for model in WIDTHS[1:3]:
                changes = [next(r for r in deltas if (r['dataset'], r['model'], r['split'], r['view'], r['metric']) == (dataset, model, split, view, metric)) for view in ('overall', 'tail') for metric in METRICS]
                cells = [name, split.title(), LABELS[model]] + [fmt(r['delta_mean'], 'time_nll', True) if r['metric'] == 'time_nll' else (f'{r["relative_mean_change_percent"]:+.2f}%' if r['relative_mean_change_percent'] is not None else '—') for r in changes] + [f'{changes[0]["improved_seeds"]}/3', f'{changes[0]["worst_delta_seed"]} / {changes[0]["worst_delta"]:+.3f}']
                lines.append('| ' + ' | '.join(cells) + ' |')
    lines.extend(['', '각 지표의 부호와 seed 수를 별도로 판단한다. 평균 RMSE가 낮아도 모든 seed가 좋아졌다는 뜻은 아니며, tail은 일부 사건 집합이므로 전체 성능을 대신하지 않는다. 최대 Δ가 음수라면 세 seed 모두 개선한 상태에서 가장 작은 개선을 뜻한다. 전체 수량·tail 수량·시간이 같은 방향인지 paired 표에서 확인한다.', ''])
    for dataset, name in DATASETS.items():
        full = {model: lookup[(dataset, model, 'test', 'overall')] for model in WIDTHS + ['s2p2_matched_head']}
        tail = {model: lookup[(dataset, model, 'test', 'tail')] for model in WIDTHS}
        if name == 'Taxi':
            lines.extend([f'**Taxi:** 폭12 Test RMSE {cell(full[WIDTHS[2]], "qty_rmse")}와 tail RMSE {cell(tail[WIDTHS[2]], "qty_rmse")}는 폭16 평균보다 낮다. 전체 RMSE 개선은 2/3 seed이며 seed42 Δ는 +11.072다. 폭8은 폭16 대비 전체 RMSE가 세 seed 모두 악화했다. 폭12의 Time NLL 평균 {full[WIDTHS[2]]["time_nll_mean"]:.4f}는 S2P2 {full["s2p2_matched_head"]["time_nll_mean"]:.4f}보다 높으며, 모든 가용 분기 MLP의 Test RMSE {lookup[(dataset, "titantpp_all_available_history_mlp", "test", "overall")]["qty_rmse_mean"]:.3f}도 보존한다.', ''])
        elif name == 'Intermittent':
            lines.extend([f'**Intermittent:** 폭12 Test RMSE {cell(full[WIDTHS[2]], "qty_rmse")}와 tail RMSE {cell(tail[WIDTHS[2]], "qty_rmse")}는 폭16 평균보다 낮지만 전체 RMSE 개선은 2/3 seed다. 폭12의 전체 Time NLL 평균 {full[WIDTHS[2]]["time_nll_mean"]:.3f}는 폭16 {full[BASELINE]["time_nll_mean"]:.3f}보다 낮고, 폭4 {full[WIDTHS[0]]["time_nll_mean"]:.3f}와 S2P2 {full["s2p2_matched_head"]["time_nll_mean"]:.3f}보다 높다. 폭4의 전체 RMSE 평균 {full[WIDTHS[0]]["qty_rmse_mean"]:.3f}도 폭12보다 낮다. Validation에서 폭12의 전체·tail RMSE는 폭16보다 높은 방향이므로 split 간 방향 차이를 함께 기록한다.', ''])
        else:
            lines.extend([f'**RAF:** 폭8·12의 Test RMSE 평균은 {full[WIDTHS[1]]["qty_rmse_mean"]:.3f}·{full[WIDTHS[2]]["qty_rmse_mean"]:.3f}로 폭16 {full[BASELINE]["qty_rmse_mean"]:.3f}보다 세 seed 모두 낮지만, 폭4 {full[WIDTHS[0]]["qty_rmse_mean"]:.3f}와 S2P2 {full["s2p2_matched_head"]["qty_rmse_mean"]:.3f}보다 평균이 높다. 폭8·12의 Time NLL 평균 {full[WIDTHS[1]]["time_nll_mean"]:.3f}·{full[WIDTHS[2]]["time_nll_mean"]:.3f}는 폭16 {full[BASELINE]["time_nll_mean"]:.3f}보다 높다. 수량 이득과 시간 손해를 함께 유지한다.', ''])
    lines.extend(['## 전체·tail 평가 집합과 출처', '', '| 데이터 | split | 전체 target | tail target | TRAIN threshold · 엄격 초과 | tail 비율 |', '|---|---|---:|---:|---:|---:|'])
    for dataset, name in DATASETS.items():
        for split in ('validation', 'test'):
            n = populations[dataset][split]['expected_target_count']
            tail = lookup[(dataset, BASELINE, split, 'tail')]['count_per_seed']
            lines.append(f'| {name} | {split.title()} | {n:,} | {tail:,} | > {THRESHOLDS[dataset]:g} | {100*tail/n:.2f}% |')
    frozen = sum('frozen_Validation' in row['verification'] for row in evidence)
    lines.extend(['', '폭8·12의 신규 평가 18조건·36 split은 원본 계약·checkpoint binary/tensor SHA·선택 epoch를 유지한 전체 Validation gate 이후 Test 결과다. 이 보고서에서 실행한 추론은 0회다. 기존 폭4·16과 비교군은 기존 공통 집계 CSV 및 개별 원본 SHA를 검증해 재사용했다.', '', f'Validation 원본에는 전체 population receipt와 frozen selected endpoint가 함께 있다. frozen endpoint는 {frozen}지표행(전체·tail 포함)이다. endpoint의 split·state·epoch·count·TRAIN threshold·수치를 검증했고, 원래 전체 Validation reference의 target/truth SHA 및 loader/data를 동일 checkpoint의 Test receipt와 동결 source 계약에 연결했다. endpoint 자체에는 개별 target/truth digest가 없으므로 모든 Validation 조건을 새 raw 예측으로 재집계한 것으로 표현하지 않는다. full-population receipt가 있는 행에서는 target/truth/loader SHA를 직접 대조했다.', '', '원GPU 학습 endpoint와 후속 full-population Validation replay는 같은 선택 epoch·source·loader를 사용해도 runtime에 따른 미세한 수치차가 있다. 독립 검토의 전체·tail 180지표셀 중 33셀에서 차이가 관측됐지만 사전 허용오차 abs/rel 1e-5 안에 있었다. 가장 큰 예는 Taxi 폭16 seed42 tail RMSE 406.191034904 → 406.191874862(차이 약 0.000840, 상대차 약 2.07e-6)다. 이번 표는 후속 replay 원본을 유지하며, 두 보고서의 값이 byte 수준에서 동일하다고 주장하지 않는다. 선택 epoch·코드·loader·선택 규칙 변경은 없다.', '', '원래 frozen 결과와 현재 코드 재실행을 구분했다. source revision이 registry에 직접 기록된 조건은 그대로 기재하고, 없는 조건은 source closure SHA와 원본 checkpoint 연결을 기록했다. 현재 runtime 기본값으로 결과를 다시 만들지 않았다.', '', 'Time NLL은 기록 정수 간격의 확률질량 점수이며 데이터 간 NLL을 합산해 순위를 만들지 않는다. Deep Renewal native shifted-NB는 공통 출력부 비교와 다른 연구군으로 포함했으며 원고 제외를 유지한다. 결정적 수량 기준선은 seed SD·시간 likelihood가 없고, RAF의 TitanTPP B는 비교 가능한 원본이 없어 —로 표시했다.', '', '## 사전에 구분한 비교 범위', '', '과거 A100 단일 seed42 탐색 arm `titantpp_history_cross_product`(cross-product), `titantpp_history_recent4_mean`(recent-four mean), `titantpp_history_recent4_attention`(recent-four attention/routing), `titantpp_history_post_block`(post-block/placement)는 이번 3seed 폭·벤치마크 비교의 입력에서 제외했다. 서로 다른 seed 범위·탐색 계약을 구분한 것이며 성능 값에 따라 불리한 조건을 제거한 것이 아니다. 과거 보고서는 보존한다. 현재 A100 CNN/GRU는 별도의 Validation-only 분석 범위이고 합치지 않았다. Instacart 폭8·12는 후속 작업 범위로 남긴다.', '', '이번 집계는 A100 개별 Validation/Test 파일·예측을 열람하지 않았다. 모든 포함된 비교군의 불리한 결과, Time NLL, 모든 seed를 보존했다.', '', '## 검증과 남은 작업 순서', '', '**3개 데이터의 split별 최종 비교와 지표 검증 — 완료**', '', '- 대상: paper_research의 이 보고서. 516지표행·123조건, 3seed ddof=1, population·threshold·선택 epoch·원본 SHA와 paired Δ를 확인했다.', '- [원본·선택·계산 영수증](comparison_receipt.json), [독립 집계 검토](final_verification.json), [재생성 스크립트](build_report.py)를 보존한다.', '', '**현재 A100 잔여 조건과 원본 회수를 기다린다 — 외부 작업 대기**', '', '- 대상: paper_research의 별도 [A100 CNN·GRU Validation-only 보고서](../titantpp_cnn_gru_completed_review_20261005_v1/README.md). 잔여 조건 종료와 원본 binary 회수·CPU 검증은 해당 작업에서 추적한다.', '- A100의 Test는 미접근 상태를 유지하고, 이 3seed Validation·Test 표와 결합하지 않는다.', '', '**Instacart 폭8·12의 후속 연결은 보류한다 — 다음 작업 / 현재 우선 범위 밖**', '', '- 대상: paper_research의 Instacart 후속 실험. 현재 사용자 우선 작업 이후 별도 범위에서 기존 원본·Validation 선택을 유지해 연결한다.', '- 이번 세 데이터 결과에서 Instacart 폭8·12 수치를 만들거나 기존 기준선을 교체하지 않는다.', '', '**CNN·GRU와 원고 주장 연결 — 다음 작업 / 별도 검증 이후**', '', '- 대상: paper_research의 A100 Validation-only 구조 대조와 논문 원고. 구조 효과·용량 효과를 구분하고 모든 확인된 이득·손해를 반영한다.', '- 새 학습·원격 평가·commit·push는 이 로컬 집계 작업에서 수행하지 않았다.'])
    (HERE / 'README.md').write_text('\n'.join(lines) + '\n')


def main():
    rows, sources, evidence, selected, populations = collect_and_verify()
    aggregate = summarize(rows)
    delta, delta_raw = paired(rows)
    rows.sort(key=lambda r: (list(DATASETS).index(r['dataset']), ORDER.index(r['model']), str(r['seed']), r['split'], r['view']))
    write_csv(HERE / 'metrics_per_seed.csv', rows)
    write_csv(HERE / 'metrics_aggregated.csv', aggregate)
    write_csv(HERE / 'paired_deltas_aggregated.csv', delta)
    write_csv(HERE / 'paired_deltas_per_seed.csv', delta_raw)
    write_csv(HERE / 'selected_epochs.csv', selected)
    tables(aggregate, populations)
    paired_tables(delta, delta_raw)
    report(aggregate, delta, populations, evidence)
    receipt = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(), 'datasets': DATASETS, 'baseline': BASELINE, 'baseline_changed': False, 'width4_role': 'comparison_arm', 'learned_conditions': len(selected), 'metric_rows': len(rows), 'summary_groups': len(aggregate), 'paired_summary_rows': len(delta), 'paired_seed_rows': len(delta_raw), 'source_files_sha256': sources, 'population_references': populations, 'source_evidence': evidence, 'selected_epochs': selected, 'aggregation': 'within-seed event aggregate; across-seed arithmetic mean and sample SD ddof1; paired delta per identical seed; no ensemble', 'thresholds_from_original_TRAIN': THRESHOLDS, 'tail_rule': 'raw_quantity > threshold', 'source_metrics_reaggregated_here': False, 'source_metrics_reused_from_SHA_verified_aggregates': True, 'new_training': False, 'new_inference_calls': 0, 'remote_calls': 0, 'selected_epoch_changed': False, 'model_selection_changed': False, 'independent_test_claim': False, 'excluded_single_seed_exploration_arms': sorted(A100), 'exclusion_reason': 'Separate historical single-seed exploration scope, retained in original report; not metric-value filtering', 'current_A100_CNN_GRU_scope': 'separate_validation_only_not_accessed_or_merged', 'missing_models': {'raf_spare_parts': ['titantpp']}, 'quantity_only_models_without_seed_SD_or_time_NLL': sorted(SIMPLE), 'deep_renewal_manuscript_exclusion_preserved': True, 'frozen_validation_digest_limitation': 'Selected endpoint files do not include target/truth digests. Shared original Validation population SHA is verified, selected state/count/threshold/loader/source closure linked; no claim of fresh full-prediction verification for those endpoints.', 'original_GPU_endpoint_vs_followup_Validation_replay': {'review_provided_by_root_QA': True, 'metric_cells_compared': 180, 'nonzero_difference_cells': 33, 'within_predefined_abs_and_rel_tolerance': 1e-5, 'largest_reported_example': {'dataset': 'yellow_trip_hourly', 'model': BASELINE, 'seed': 42, 'view': 'tail', 'metric': 'qty_rmse', 'original_endpoint': 406.191034904, 'followup_replay': 406.191874862, 'approx_abs_difference': 0.000840, 'approx_relative_difference': 2.07e-6}, 'same_selected_epoch': True, 'same_source_and_loader_contract': True, 'byte_identical_numeric_claim': False, 'interpretation': 'runtime numerical difference within predeclared tolerance; scientific condition and conclusions unchanged'}, 'presentation_rules': {'summary_Time_NLL': 'absolute signed mean delta for overall and tail; scientific notation below 1e-4', 'paired_Time_NLL_relative_percent': 'display em dash only when abs(baseline_mean)<1e-6; all original percent/denominator/delta values retained in CSV', 'worst_seed': 'largest absolute model metric and largest paired delta both retained; negative maximum delta means all seeds improved'}, 'outputs': {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.csv', '.md', '.py', '.json') and p.name not in ('comparison_receipt.json', 'final_verification.json') and '__pycache__' not in p.parts}}
    (HERE / 'comparison_receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'status': 'complete', 'learned_conditions': len(selected), 'metric_rows': len(rows), 'summary_groups': len(aggregate)}))


if __name__ == '__main__':
    main()
