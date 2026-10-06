"""Freeze original quantity-selected, full Validation metrics. No inference or Test access.

The historical receipt is used only as an index of checkpoint/source bindings;
its mixed-split numeric CSV and reports are never opened. All metric values come
from original Validation-only endpoint_replays.json, never a new replay.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAMPAIGN = ROOT / 'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1'
DESIGN = ROOT / 'reports/titantpp_cnn_gru54_3seed_dual_20261005_v1'
PRIOR = ROOT / 'reports/titantpp_three_dataset_final_comparison_20261005_v1'
CNN = 'titantpp_cnn_gru54'
MLP16 = 'titantpp_history_mlp_width16'
S2P2 = 's2p2_matched_head'
SEEDS = (42, 52, 62)
DATASETS = {'yellow_trip_hourly': 'Taxi', 'intermittent_frozen_5000': 'Intermittent', 'raf_spare_parts': 'RAF'}
THRESHOLDS = {'yellow_trip_hourly': 3449, 'intermittent_frozen_5000': 187, 'raf_spare_parts': 200}
TAIL_COUNTS = {'yellow_trip_hourly': 79, 'intermittent_frozen_5000': 1141, 'raf_spare_parts': 50}
EXTERNAL = ('rmtpp', 'thp', 'nhp', 'sahp', S2P2, 'attnhp_matched_head')
INTERNAL = ('titantpp_history_mlp', 'titantpp_history_mlp_width8', 'titantpp_history_mlp_width12', MLP16, 'titantpp_all_available_history_mlp', 'titantpp_current_only_param_matched', 'titantpp')
ORDER = (CNN,) + INTERNAL + EXTERNAL
METRICS = ('qty_rmse', 'qty_mae', 'time_nll')
LABELS = {CNN: 'CNN+GRU54 · original', 'titantpp_history_mlp': 'TitanTPP MLP 폭4', 'titantpp_history_mlp_width8': 'TitanTPP MLP 폭8', 'titantpp_history_mlp_width12': 'TitanTPP MLP 폭12', MLP16: 'TitanTPP MLP 폭16 · 사용자 기준선', 'titantpp_all_available_history_mlp': '모든 가용 분기 MLP', 'titantpp_current_only_param_matched': '현재 상태만 · 파라미터 대조', 'titantpp': 'TitanTPP B', 'rmtpp': 'RMTPP', 'thp': 'THP', 'nhp': 'NHP', 'sahp': 'SAHP', S2P2: 'S2P2 · 공통 출력부', 'attnhp_matched_head': 'AttNHP · 공통 출력부'}
SOURCES: dict[str, str] = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approved_path(path):
    """Deny performance files outside our explicit Validation/source allowlist."""
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Source outside repository')
    require(not any(p in ('test', 'test_full', 'predictions', 'test_analysis') or p.startswith('test__') for p in path.parts), 'Held-out performance path is locked')
    allowed_names = {'endpoint_replays.json', 'history.json', 'input_receipt.json', 'terminal_manifest.json', 'comparison_receipt.json', 'completion_receipt.json', 'Validation_comparison.json', 'source_rows.json', 'reuse_seed42_registry.json', 'execution_contract.json', 'analysis_contract.json', 'server_checkpoint_receipt.json', 'summary.json'}
    require(path.name in allowed_names, f'File not on Validation/source allowlist: {path.name}')
    return path


def read(path, expected_sha=None):
    path = approved_path(path)
    digest = sha(path)
    require(expected_sha is None or digest == expected_sha, f'SHA mismatch: {rel(path)}')
    SOURCES[rel(path)] = digest
    return json.loads(path.read_text())


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-9)


def verify_population(receipt, spec):
    identity = spec['inherited_data_identity']
    require(receipt['dataset_id'] == spec['dataset_id'], 'Input dataset differs')
    require(receipt.get('held_out_materialized') is False, 'Input materializes held-out')
    canonical = {'data_sha256': identity['data']['sha256'], 'split_manifest_sha256': identity['split_manifest']['sha256']}
    known_packaging = {'data_sha256': '0638d0fdc2fb6dda6e32d1cdc62fe282a8da912da420e0295431478fe38225ea', 'split_manifest_sha256': '1a88913ed828b0f57b013a763a7713383bd9dae2edfff7d5ede67fbcd6937fdc'}
    require(receipt['input_identity'] == canonical or (spec['dataset_id'] == 'intermittent_frozen_5000' and receipt['input_identity'] == known_packaging), 'Data/split identity differs without recorded package equivalence')
    for split in ('train', 'validation'):
        require(receipt['populations'][split] == identity['populations'][split], f'{split} population differs')
    for name in ('train_log_mean', 'train_log_std', 'raw_scale'):
        require(math.isclose(receipt[name], spec['statistics'][name], rel_tol=0, abs_tol=1e-12), f'TRAIN statistic differs: {name}')
    require(receipt['all_train_rows']['quantity_sha256'] == spec['statistics']['all_train_quantity_sha256'], 'TRAIN statistic population differs')


def verify_endpoint(endpoint, history, binding, dataset):
    selected = endpoint['selected']
    require(selected['evaluation_scope'] == 'validation_only' and selected['held_out_test_evaluated'] is False, 'Endpoint scope differs')
    require(endpoint['best_epoch'] == binding['selected_epoch'], 'Selected epoch binding differs')
    state = binding.get('state_tensor_sha256', binding.get('state_sha256'))
    require(selected['state_sha256'] == state, 'Selected state binding differs')
    records = history['history'] if isinstance(history, dict) else history
    require([r['epoch'] for r in records] == list(range(1, len(records) + 1)), 'Non-sequential history')
    require(endpoint['completed_epochs'] == len(records), 'Completion/history length differs')
    finite = [r for r in records if math.isfinite(r['val_qty_rmse'])]
    first_min = min(finite, key=lambda r: r['val_qty_rmse'])
    require(first_min['epoch'] == binding['selected_epoch'], 'Original quantity selector changed')
    same_epoch = records[binding['selected_epoch'] - 1]
    for metric in METRICS:
        equal = close(selected[metric], same_epoch['val_' + metric])
        if binding.get('reused_cross_GPU_tolerance'):
            equal = math.isclose(selected[metric], same_epoch['val_' + metric], rel_tol=1e-5, abs_tol=1e-5)
        require(equal, f'Endpoint/history same-epoch metric differs: {metric}')
        require(math.isfinite(selected[metric]), 'Nonfinite metric')
        require(math.isfinite(selected['tail'][metric]), 'Nonfinite tail metric')
    require(selected['quantity_boundaries'][-1] == THRESHOLDS[dataset], 'TRAIN tail threshold differs')
    require(selected['tail']['count'] == TAIL_COUNTS[dataset], 'Tail population count differs')
    return records, {m: selected[m] - same_epoch['val_' + m] for m in METRICS}


def input_path(endpoint_path):
    for parent in list(Path(endpoint_path).parents)[:8]:
        path = parent / 'input_receipt.json'
        if path.exists():
            return path
    raise ValueError(f'Original input receipt missing: {endpoint_path}')


def verify_packaged_time(ep, spec):
    original_root = next(p for p in ep.parents if (p / 'execution_contract.json').exists())
    packaged = read(original_root / 'execution_contract.json')
    dataset = next(r for r in packaged['datasets'] if r['dataset_id'] == spec['dataset_id'])
    require(dataset['time_statistics'] == spec['time_statistics'], 'Packaged TRAIN/Validation dt SHA contract differs')
    require(dataset['loader'] == spec['loader'], 'Packaged loader differs')
    require(dataset['model']['time_observation_contract'] == spec['model']['time_observation_contract'], 'Packaged likelihood/unit differs')
    manifest_path = input_path(ep).parent / 'terminal_manifest.json'
    manifest = read(manifest_path)
    summary_path = ep.parent / 'summary.json'
    summary = read(summary_path, manifest['files'][str(summary_path.relative_to(manifest_path.parent))])
    require(summary['evaluation_scope'] == 'validation_only' and summary['held_out_test_evaluated'] is False, 'Packaged summary scope differs')
    interface = summary['interface_meta']
    require(interface['data_scope'] == 'verified_train_validation_only', 'Packaged runtime data verification absent')
    require(interface['backbone_design']['time_statistics'] == spec['time_statistics'], 'Packaged runtime ordered dt SHA differs')
    require(interface['time_head']['observation_likelihood'] == spec['model']['time_observation_contract'], 'Packaged runtime likelihood/unit differs')
    require(interface['source_files_sha256'] == packaged['source']['files_sha256'], 'Packaged runtime/source closure differs')
    code_paths = ('paper/scripts/run_multilag_detail_execution.py', 'paper/scripts/run_pakdd_extension.py')
    code_evidence = {}
    for name in code_paths:
        path = original_root / 'source' / name
        require(sha(path) == packaged['source']['files'][name], 'Packaged frozen time-check source SHA differs')
        SOURCES[rel(path)] = sha(path)
        code_evidence[name] = sha(path)
    return {'status': 'passed', 'runtime_summary': rel(summary_path), 'runtime_summary_sha256': SOURCES[rel(summary_path)], 'original_contract': rel(original_root / 'execution_contract.json'), 'expected_train_target_dt_sha256': spec['time_statistics']['expected_train_target_dt_sha256'], 'expected_validation_target_dt_sha256': spec['time_statistics']['expected_validation_target_dt_sha256'], 'observation_likelihood': spec['model']['time_observation_contract'], 'frozen_source_checks': code_evidence, 'verification': 'Successful runtime interface is returned only after ordered TRAIN/Validation raw dt hashes match; frozen code SHA and runtime interface/contract verified.'}


def collect():
    SOURCES.clear()
    contract = read(DESIGN / 'execution_contract.json')
    analysis_contract = read(DESIGN / 'analysis_contract.json')
    receipt = read(CAMPAIGN / 'analysis/completion_receipt.json')
    final = read(CAMPAIGN / 'analysis/Validation_comparison.json', receipt['outputs']['Validation_comparison.json'])
    require(final['scope'] == 'Validation_only' and final['held_out_test_evaluated'] is False, 'Final scope differs')
    require(receipt['status'] == 'nine_conditions_Validation_originals_complete' and receipt['canonical_conditions'] == 9, 'CNNGRU final9 incomplete')
    require(receipt['contract_sha256'] == analysis_contract['scientific_contract_sha256'], 'Frozen contract mismatch')
    prior = read(PRIOR / 'comparison_receipt.json')
    require(prior['status'] == 'complete' and prior['baseline'] == MLP16, 'Historical baseline index differs')
    reuse = read(DESIGN / 'reuse_seed42_registry.json')
    require(reuse['status'] == 'PASS', 'Seed42 reuse audit did not pass')
    specs = {r['dataset_id']: r for r in contract['datasets']}
    safe = {}
    for seed in (52, 62):
        folder = CAMPAIGN / f'comparisons/seed{seed}_20261006_v1'
        index_receipt = read(folder / 'comparison_receipt.json')
        index = read(folder / 'source_rows.json', index_receipt['outputs']['source_rows.json'])
        require(index['scope'].lower().startswith('validation_only'), 'Unsafe comparison index')
        safe.update({(r['dataset'], r['model'], seed): r for r in index['rows'] if r['model'] != CNN})
    indexed = {(r['dataset'], r['model'], r['seed']): r for r in prior['selected_epochs'] if r['model'] in INTERNAL + EXTERNAL and r['dataset'] in DATASETS}
    evidence = {(r['dataset'], r['model'], r['seed']): r for r in prior['source_evidence'] if r['split'] == 'validation' and r['view'] == 'overall'}
    selected_sources = []
    for identity, frozen in indexed.items():
        dataset, model, seed = identity
        if identity in safe:
            original = safe[identity]
            ep = Path(original['endpoint_path'])
            ep_sha = original['endpoint_sha256']
        else:
            indexed_source = evidence[identity]
            ep = ROOT / indexed_source['source'] if Path(indexed_source['source']).name == 'endpoint_replays.json' else ROOT / Path(frozen['checkpoint_path']).parent / 'endpoint_replays.json'
            ep_sha = prior['source_files_sha256'].get(rel(ep))
            if model == 'titantpp':
                peer = evidence[dataset, 'rmtpp', seed]
                ep = ROOT / peer['source'].replace('/runs/rmtpp/', '/runs/titantpp/')
                ep_sha = prior['source_files_sha256'].get(rel(ep))
        selected_sources.append((dataset, model, seed, ep, ep_sha, frozen, None))
    reuse_lookup = {r['dataset']: r for r in reuse['rows']}
    for row in final['rows']:
        dataset, seed = row['dataset'], row['seed']
        if seed == 42:
            frozen = reuse_lookup[dataset]
            ep = Path(frozen['endpoint_replays']['path'])
            binding = {**frozen['selected'], 'selected_epoch': frozen['selected_epoch'], 'reused_cross_GPU_tolerance': True}
            checkpoint = {**frozen['selected'], 'GPU_segments': frozen['GPU_segments'], 'cross_GPU_selected_replay_gate_events': frozen['cross_GPU_selected_replay_gate_events'], 'source_revision': reuse['source_revision'], 'source_closure_sha256': reuse['scientific_source']['closure_sha256']}
        else:
            host = '5090' if dataset == 'intermittent_frozen_5000' else '5080'
            job_root = CAMPAIGN / f'retrieved/{host}/original/run/{dataset}__{seed}__{CNN}'
            native = next(r for r in receipt['originals'][host]['manifests'] if r['id'] == job_root.name)
            manifest = read(job_root / 'terminal_manifest.json', native['terminal_manifest_sha256'])
            require(manifest['scientific_success'] is True and manifest['status'] == 'complete', 'Native training did not finish')
            ep = job_root / f'runs/{CNN}/count_only_log_regression/seed_{seed}/endpoint_replays.json'
            require(manifest['files'][str(ep.relative_to(job_root))] == row['endpoint_sha256'], 'Manifest/endpoint binding differs')
            checkpoint = read(ep.parent / 'server_checkpoint_receipt.json')
            binding = {'selected_epoch': row['best_epoch'], 'state_tensor_sha256': checkpoint['selected']['state_tensor_sha256']} if 'selected' in checkpoint else {'selected_epoch': row['best_epoch'], 'state_tensor_sha256': read(ep)['selected']['state_sha256']}
            checkpoint = {**checkpoint, 'native_manifest': rel(job_root / 'terminal_manifest.json'), 'native_manifest_sha256': native['terminal_manifest_sha256'], 'selected_file_sha256': manifest['files'][str((ep.parent / 'best_val_qty_rmse_model.pt').relative_to(job_root))], 'path': rel(ep.parent / 'best_val_qty_rmse_model.pt'), 'source_revision': contract['source']['git_revision'], 'source_closure_sha256': contract['source']['files_sha256'], 'host': host}
        selected_sources.append((dataset, CNN, seed, ep, row['endpoint_sha256'], binding, {'final': row, 'checkpoint': checkpoint}))
    rows, bindings = [], []
    for dataset, model, seed, ep, digest, frozen, cnn_info in selected_sources:
        endpoint = read(ep, digest)
        history = read(ep.parent / 'history.json')
        inp_path = input_path(ep)
        inp = read(inp_path)
        verify_population(inp, specs[dataset])
        records, history_differences = verify_endpoint(endpoint, history, frozen, dataset)
        selected = endpoint['selected']
        require(selected['count'] == specs[dataset]['inherited_data_identity']['populations']['validation']['target_count'], 'Full Validation count differs')
        if cnn_info:
            for view in ('overall', 'tail'):
                for metric in METRICS:
                    key = metric if view == 'overall' else 'tail_' + metric
                    source = selected if view == 'overall' else selected['tail']
                    require(source[metric] == cnn_info['final']['metrics'][key], 'Final9/source original metric differs')
        identity = {'dataset': dataset, 'model': model, 'seed': seed, 'selected_epoch': endpoint['best_epoch']}
        for view in ('overall', 'tail'):
            m = selected if view == 'overall' else selected['tail']
            rows.append({**identity, 'split': 'validation', 'view': view, 'count': m['count'], 'threshold': THRESHOLDS[dataset] if view == 'tail' else None, **{k: m[k] for k in METRICS}, 'source': rel(ep), 'source_sha256': SOURCES[rel(ep)], 'state_tensor_sha256': selected['state_sha256']})
        canonical_input = {'data_sha256': specs[dataset]['inherited_data_identity']['data']['sha256'], 'split_manifest_sha256': specs[dataset]['inherited_data_identity']['split_manifest']['sha256']}
        require(model != CNN or inp['input_identity'] == canonical_input, 'CNNGRU canonical input file SHA differs')
        time_evidence = verify_packaged_time(ep, specs[dataset]) if inp['input_identity'] != canonical_input else {'status': 'canonical_data_split_SHA_exact', 'time_unit': specs[dataset]['model']['time_observation_contract']['unit']}
        bindings.append({**identity, 'completed_epochs': len(records), 'endpoint': {'path': rel(ep), 'sha256': SOURCES[rel(ep)]}, 'history': {'path': rel(ep.parent / 'history.json'), 'sha256': SOURCES[rel(ep.parent / 'history.json')]}, 'input_receipt': {'path': rel(inp_path), 'sha256': SOURCES[rel(inp_path)]}, 'state_tensor_sha256': selected['state_sha256'], 'checkpoint_binding': cnn_info['checkpoint'] if cnn_info else frozen, 'population': inp['populations']['validation'], 'input_identity': inp['input_identity'], 'canonical_input_identity': canonical_input, 'input_file_SHA_exactly_canonical': inp['input_identity'] == canonical_input, 'population_equivalence_verified': 'Exact ordered TRAIN/Validation target identity+quantity SHA/counts and TRAIN statistic hash+values', 'time_input_equivalence': time_evidence, 'same_epoch_history_numeric_differences': history_differences, 'same_epoch_numeric_tolerance': {'rel_tol': 1e-5, 'abs_tol': 1e-5} if frozen.get('reused_cross_GPU_tolerance') else {'rel_tol': 1e-7, 'abs_tol': 1e-9}, 'selector_verified': 'strict_earliest_finite_Validation_raw_quantity_RMSE_minimum', 'time_nll_at_same_quantity_selected_epoch': True, 'original_GPU_training_endpoint': True, 'new_replay': False})
    check_coverage(rows)
    return rows, bindings, contract, receipt, final


def check_coverage(rows):
    keys = [(r['dataset'], r['model'], r['seed'], r['view']) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate metric row')
    for dataset in DATASETS:
        models = set(ORDER) - ({'titantpp'} if dataset == 'raf_spare_parts' else set())
        expected = {(dataset, model, seed, view) for model in models for seed in SEEDS for view in ('overall', 'tail')}
        require({k for k in keys if k[0] == dataset} == expected, f'Model/seed/view coverage incomplete: {dataset}')
    require(all(r['split'] == 'validation' for r in rows), 'Non-Validation metric row')


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['model'], row['view']), []).append(row)
    output = []
    for (dataset, model, view), members in groups.items():
        require(sorted(r['seed'] for r in members) == list(SEEDS), 'Summary needs matched three seeds')
        output.append({'dataset': dataset, 'model': model, 'split': 'validation', 'view': view, 'seed_count': 3, 'count_per_seed': members[0]['count'], **{f'{m}_{suffix}': function([r[m] for r in members]) for m in METRICS for suffix, function in [('mean', statistics.mean), ('sample_sd', statistics.stdev)]}})
    return output


def paired(rows):
    lookup = {(r['dataset'], r['model'], r['seed'], r['view']): r for r in rows}
    raw, summary = [], []
    for dataset in DATASETS:
        for model in INTERNAL + EXTERNAL:
            if (dataset, model, 42, 'overall') not in lookup:
                continue
            for view in ('overall', 'tail'):
                for metric in METRICS:
                    pairs = []
                    for seed in SEEDS:
                        candidate = lookup[dataset, CNN, seed, view][metric]
                        reference = lookup[dataset, model, seed, view][metric]
                        delta = candidate - reference
                        row = {'dataset': dataset, 'candidate': CNN, 'reference': model, 'seed': seed, 'split': 'validation', 'view': view, 'metric': metric, 'candidate_value': candidate, 'reference_value': reference, 'delta': delta, 'relative_change_percent': 100 * delta / reference if abs(reference) >= 1e-6 else None}
                        raw.append(row)
                        pairs.append(row)
                    values = [r['delta'] for r in pairs]
                    base = statistics.mean(r['reference_value'] for r in pairs)
                    summary.append({'dataset': dataset, 'candidate': CNN, 'reference': model, 'split': 'validation', 'view': view, 'metric': metric, 'seed_count': 3, 'reference_mean': base, 'delta_mean': statistics.mean(values), 'delta_sample_sd': statistics.stdev(values), 'relative_mean_change_percent': 100 * statistics.mean(values) / base if abs(base) >= 1e-6 else None, 'improved_seeds': sum(v < 0 for v in values), 'worsened_seeds': sum(v > 0 for v in values), 'seed42_delta': values[0], 'seed52_delta': values[1], 'seed62_delta': values[2]})
    return raw, summary


def scientific_gate(pairs):
    criteria = [('overall', 'qty_rmse', True), ('overall', 'qty_mae', False), ('tail', 'qty_rmse', False), ('tail', 'qty_mae', False), ('overall', 'time_nll', False), ('tail', 'time_nll', False)]
    results = []
    for dataset in DATASETS:
        for reference in (S2P2, MLP16):
            failures = []
            for view, metric, strict in criteria:
                r = next(r for r in pairs if (r['dataset'], r['reference'], r['view'], r['metric']) == (dataset, reference, view, metric))
                deltas = {s: r[f'seed{s}_delta'] for s in SEEDS}
                failed = {s: d for s, d in deltas.items() if d >= 0 if strict} if strict else {s: d for s, d in deltas.items() if d > 0}
                if failed:
                    failures.append({'view': view, 'metric': metric, 'requirement': 'all3 strictly improve' if strict else 'no seed worse', 'failed_seed_deltas': failed})
            results.append({'dataset': dataset, 'reference': reference, 'strong_dominance': not failures, 'status': 'PASS' if not failures else 'FAIL_TRADEOFF', 'failures': failures})
    return results


def write_json(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, metric):
    if value != 0 and abs(value) < 1e-4:
        return f'{value:.6e}'
    return f'{value:.6f}' if metric == 'time_nll' else f'{value:.3f}'


def table(summary, view):
    text = ['# Original CNN+GRU54 · 최종 3seed Validation · ' + ('전체 사건' if view == 'overall' else 'TRAIN 기준 큰 수량 tail'), '', 'seed42·52·62의 산술 평균 ± 표본 표준편차(ddof=1). 모든 값은 원래 전체 Validation 수량 RMSE로 선택된 checkpoint의 학습 당시 Validation endpoint다. Time NLL도 같은 epoch 값이다. ↓는 작을수록 좋다.', '']
    lookup = {(r['dataset'], r['model'], r['view']): r for r in summary}
    for dataset, label in DATASETS.items():
        count = lookup[dataset, CNN, view]['count_per_seed']
        text += [f'## {label}', '', f'{count:,} target/seed.' + (f' TRAIN 고정 raw quantity > {THRESHOLDS[dataset]}.' if view == 'tail' else ''), '', '| 모델 | seed 수 | target 수/seed | RMSE ↓ | MAE ↓ | Time NLL ↓ |', '|---|---:|---:|---:|---:|---:|']
        for model in ORDER:
            r = lookup.get((dataset, model, view))
            if r is None:
                text.append(f'| {LABELS[model]} | — | — | — | — | — |')
            else:
                cells = [fmt(r[m + '_mean'], m) + ' ± ' + fmt(r[m + '_sample_sd'], m) for m in METRICS]
                text.append(f'| {LABELS[model]} | 3 | {count:,} | ' + ' | '.join(cells) + ' |')
        text.append('')
    text += ['RAF TitanTPP B는 비교 가능한 원본이 없어 결측을 유지했다. 외부 six는 RMTPP/THP/NHP/SAHP/S2P2/AttNHP이다. S2P2/AttNHP는 공통 출력부 대조다. n=3 기술 통계이며 통계적 유의성·독립 held-out 성능·앙상블 성능을 뜻하지 않는다.', '', '수량과 간격 단위가 데이터셋마다 다르므로 raw RMSE/MAE/Time NLL을 세 데이터셋 간 평균하지 않았다. Time NLL은 양의 정수 간격 lognormal round/clamp 확률질량 점수이며 Taxi=시간, Intermittent=주, RAF=월이다. 극소 tail NLL에는 백분율을 계산하지 않는다.', '', '원값·원본 경로·SHA·선택 epoch는 [metrics_per_seed.csv](metrics_per_seed.csv), [source_bindings.json](source_bindings.json), [comparison_receipt.json](comparison_receipt.json)에 보존했다. 이전 최종 표 일부가 후속 Validation replay를 사용한 것과 달리 이 표는 모두 original 학습 endpoint를 사용하므로 미세한 수치 차이가 가능하다.']
    return '\n'.join(text) + '\n'


def report(summary, paired_summary, gates):
    lookup = {(r['dataset'], r['reference'], r['view'], r['metric']): r for r in paired_summary}
    text = ['# Original CNN+GRU54 최종 3seed Validation 비교', '', '**원본 결속 검증은 통과했으나, 고정된 강한 지배 기준은 S2P2와 MLP16에 대해 세 데이터셋 모두 통과하지 못했다.** CNN+GRU54를 통합 승자로 선택하거나 Time NLL 악화를 숨기지 않는다.', '', '주 지표는 전체 Validation raw quantity RMSE다. 전체/tail MAE와 Time NLL을 같은 quantity-selected epoch에서 함께 읽는다. 후보는 original CNN+GRU54 하나이며 seed42/52/62를 보존하고 폭·epoch·지표를 사후 재선택하지 않았다.', '']
    for dataset, label in DATASETS.items():
        text += [f'## {label}', '', '| 기준 | view/지표 | CNNGRU − 기준 평균 ± 표본 SD | 개선 seed | seed42 / 52 / 62 delta |', '|---|---|---:|---:|---|']
        for reference in (S2P2, MLP16):
            for view in ('overall', 'tail'):
                for metric in METRICS:
                    r = lookup[dataset, reference, view, metric]
                    values = ' / '.join(fmt(r[f'seed{s}_delta'], metric) for s in SEEDS)
                    text.append(f'| {LABELS[reference]} | {view}/{metric} | {fmt(r["delta_mean"], metric)} ± {fmt(r["delta_sample_sd"], metric)} | {r["improved_seeds"]}/3 | {values} |')
        text.append('')
    text += ['## 사전에 고정된 기준과 해석', '', '실행 계약의 `reporting.adoption`은 3seed 모두 전체 RMSE가 개선되고 전체 MAE·tail RMSE/MAE·Time NLL이 어느 seed에서도 나빠지지 않을 때만 강한 지배로 인정한다. 이 보고서는 원래 문구의 Time NLL에 전체와 tail 양쪽을 포함하는 보수적 해석을 적용한다. 전체 Time NLL만 적용해도 결론은 동일하다. `strict_validation_gate.json`은 각 실패 지표와 seed delta를 보존한다.', '', 'Taxi에서는 외부 six 모두보다 수량 RMSE 평균이 낮지만 S2P2보다 Time NLL이 높고, 강한 내부 MLP 폭8/16 및 모든 가용 분기 MLP보다 전체 RMSE 평균이 높다. Intermittent에서도 외부 six보다 RMSE 평균은 낮으나 S2P2보다 Time NLL이 높고 MLP 폭4/8/16보다 RMSE가 높다. RAF에서는 S2P2와 MLP 폭4/8/12/16보다 전체 RMSE가 높다. 따라서 수량 예측의 외부 대조 이점과 시간/내부 대조 약점을 함께 보고한다.', '', 'n=3의 평균/표본 SD와 같은 seed 차이는 기술 통계다. 통계적 유의성, recurrence/CNN 상호작용의 독립 기여, 동일 GPU 효율 우월성, 독립 held-out 일반화 주장은 뒷받침하지 않는다.', '', '## 원본과 평가 범위', '', '새 fit은 5080의 Taxi/RAF seed52/62 네 개와 5090의 Intermittent seed52/62 두 개다. seed42 세 개는 A100 및 후속 GPU 연속 학습 이력이 있는 기존 원본을 재사용한다. 정확한 GPU segment, source revision/closure, checkpoint file/state SHA, endpoint/history/input SHA는 source_bindings에 있다. seed42를 이번 새 GPU fit이나 동일 GPU 반복으로 재명명하지 않았다.', '', '이 작업은 로컬 JSON만 읽어 결과를 재집계했다. remote/GPU/optimizer/training/inference/evaluation, checkpoint deserialization, Test 성능·행·예측, Validation/Test 혼합 숫자 CSV·보고서 열람은 수행하지 않았다. 이미 개발에 노출된 Test를 향후 평가하더라도 독립 held-out 증거로 다시 이름 붙일 수 없다.', '', 'benchmark 수량과 NLL의 source는 기존 frozen registry에 결속된 original Validation endpoint이며 현재 코드 기본값으로 재현한 결과가 아니다. TRAIN/Validation identity·quantity hash와 TRAIN 통계, 고정 tail threshold, 전체 target 수, 첫 유한 수량 RMSE 최소 epoch, 같은 epoch NLL, selected state를 직접 검증했다. endpoint digest와 기존 평가 receipt digest는 서로 다른 namespace이므로 동일 digest처럼 비교하지 않았다.', '', '## 다음 작업의 승인 경계', '', '**기준선과 비교 기록 확정 — 완료**', '- 이 디렉터리의 수량/시간 전체 및 tail 표, 같은 seed 차이, 원본 결속 영수증과 gate를 고정했다.', '', '**Test9 범위 판단 — 승인 필요**', '- 원본 결속 PASS는 강한 지배 PASS와 다르다. 현재 후보는 강한 지배 FAIL_TRADEOFF다.', '- 사용자 승인이 허용한 gate가 원본 결속인지 성능 채택인지 구분하고 root 세션이 결정한다. 이 보고서는 승인 자체를 만들지 않는다.', '- 실행할 경우 original CNN+GRU54의 3 dataset × seed42/52/62 checkpoint/epoch를 그대로 쓰며 재학습·재선택·대체 fit은 제외한다. 기존 Test의 개발 노출 한계를 유지한다.']
    return '\n'.join(text) + '\n'


def main():
    rows, bindings, contract, completion, final = collect()
    summary = summarize(rows)
    seed_deltas, paired_summary = paired(rows)
    gates = scientific_gate(paired_summary)
    for name, data in [('metrics_per_seed.csv', rows), ('summary.csv', summary), ('paired_deltas_per_seed.csv', seed_deltas), ('paired_deltas_summary.csv', paired_summary)]:
        write_csv(name, data)
    write_json('source_bindings.json', {'scope': 'original_full_validation_only', 'conditions': bindings})
    gate = {'schema': 'cnn_gru54_original_validation_strict_gate_v1', 'scope': 'Validation_only', 'status': 'PASS_EVIDENCE_FAIL_STRONG_DOMINANCE', 'evidence_integrity_gate': 'PASS', 'evidence_checks': ['all9_final_originals_complete', 'original_endpoint_SHA_matches_completion', 'matched_seed42_52_62', 'same_TRAIN_Validation_data_split_population_quantity_SHA_and_statistics', 'same_full_Validation_targets_and_fixed_TRAIN_tail', 'strict_first_finite_quantity_RMSE_minimum_epoch_preserved', 'NLL_same_quantity_selected_epoch', 'selected_state_matches_frozen_checkpoint_registry', 'no_Test_performance_or_mixed_numeric_source_access'], 'scientific_adoption_gate': 'FAIL_TRADEOFF', 'frozen_adoption_source': {'path': rel(DESIGN / 'execution_contract.json'), 'sha256': SOURCES[rel(DESIGN / 'execution_contract.json')], 'original_rule': contract['reporting']['adoption']}, 'dominance_results': gates, 'checkpoint_reselection': False, 'test_unlock': {'performed_here': False, 'automatic_authorization': False, 'decision': 'User-authorized root workflow must distinguish evidence integrity from strong-dominance adoption; this gate grants neither remote execution nor Test access.', 'eligible_frozen_condition_count': 9, 'candidate_model': CNN, 'seeds': list(SEEDS), 'datasets': list(DATASETS), 'checkpoint_binding_source': 'source_bindings.json', 'quantity_selector_and_epoch_must_remain_fixed': True, 'Test_is_development_exposed_not_independent': True}, 'held_out_performance_read': False, 'new_training': False, 'new_inference': False}
    gate.update(passed=True, evaluation_scope='validation_only', held_out_test_evaluated=False, conditions=9)
    gate['evidence_checks'][3] = 'matched TRAIN/Validation ordered identity+quantity SHA/statistics; canonical file SHA exact for CNN9; benchmark packaging alternate dt SHA/runtime likelihood equivalence verified'
    gate['test_unlock'].update(decision='Existing user authorization covers original CNNGRU Test9 regardless of adoption superiority; root may execute only after evidence integrity and a new native full Validation9 replay gate pass. Scientific FAIL_TRADEOFF restricts claims, not authorized Test execution.', authorization_scope='Existing user instruction; no new permission requested here.', native_Validation9_replay_gate_required=True)
    gate['rows'] = []
    for binding in sorted((b for b in bindings if b['model'] == CNN), key=lambda b: (b['dataset'], b['seed'])):
        cp = binding['checkpoint_binding']
        gate['rows'].append({'dataset': binding['dataset'], 'seed': binding['seed'], 'model': CNN, 'selected_epoch': binding['selected_epoch'], 'checkpoint_path': cp['path'], 'checkpoint_file_sha256': cp.get('file_sha256', cp.get('selected_file_sha256')), 'state_tensor_sha256': binding['state_tensor_sha256'], 'endpoint_sha256': binding['endpoint']['sha256'], 'endpoint_path': binding['endpoint']['path'], 'source_closure_sha256': cp['source_closure_sha256'], 'source_revision': cp['source_revision'], 'population': binding['population'], 'same_epoch_time_nll': True})
    write_json('strict_validation_gate.json', gate)
    (HERE / 'VALIDATION_TABLES.md').write_text(table(summary, 'overall'))
    (HERE / 'VALIDATION_TAIL_TABLES.md').write_text(table(summary, 'tail'))
    prose = report(summary, paired_summary, gates)
    prose = prose.replace('**Test9 범위 판단 — 승인 필요**', '**Test9 실행 — 다음 작업 / 기존 승인 범위**')
    prose = prose.replace('- 사용자 승인이 허용한 gate가 원본 결속인지 성능 채택인지 구분하고 root 세션이 결정한다. 이 보고서는 승인 자체를 만들지 않는다.', '- 이번 사용자 승인은 우월성 여부와 별개로 original 9 checkpoint를 평가하는 범위다. root 세션은 원본 결속과 새 native 전체 Validation9 replay gate를 통과한 뒤 기존 승인 범위의 Test9를 실행한다. 채택 FAIL_TRADEOFF는 과학적 주장만 제한하며 Test 평가 승인을 취소하지 않는다.')
    prose += '\nIntermittent 모든 가용 분기 MLP와 현재 상태만 seed62의 Runpod train/validation 파일 포장 SHA는 canonical 파일과 다르다. 두 조건 모두 ordered TRAIN/Validation target identity·quantity SHA 및 TRAIN 통계가 동일하고, frozen runtime interface의 TRAIN/Validation raw dt SHA 검사와 주 단위 lognormal 관측 계약도 동일함을 source_bindings에서 검증했다. 파일 SHA를 동일한 것으로 보고하지 않는다. 재사용 seed42의 cross GPU endpoint와 같은 epoch 학습 이력 사이에는 원래 gate 허용오차 이내 차이가 있으며 원차이·허용오차를 보존했다.\n'
    (HERE / 'README.md').write_text(prose)
    output_names = ['metrics_per_seed.csv', 'summary.csv', 'paired_deltas_per_seed.csv', 'paired_deltas_summary.csv', 'source_bindings.json', 'strict_validation_gate.json', 'VALIDATION_TABLES.md', 'VALIDATION_TAIL_TABLES.md', 'README.md']
    write_json('comparison_receipt.json', {'schema': 'cnn_gru54_final_original_validation_comparison_v1', 'status': 'complete', 'scope': 'Validation_only_original_quantity_selected_endpoints', 'learned_conditions': len(bindings), 'candidate_conditions': 9, 'benchmark_conditions': len(bindings) - 9, 'metric_rows': len(rows), 'summary_groups': len(summary), 'paired_seed_metric_rows': len(seed_deltas), 'paired_summary_metric_rows': len(paired_summary), 'aggregation': 'datasetwise arithmetic mean and sample SD ddof1; same-seed CNNGRU minus each frozen reference; no cross-dataset raw metric average', 'baseline': MLP16, 'baseline_changed': False, 'source_files_sha256': SOURCES, 'original_GPU_endpoints_only': True, 'historical_benchmark_manifest_index': 'Prior receipt used for checkpoint/source metadata only; no mixed performance CSV/report opened.', 'comparison_namespace': 'original training Validation target identity/quantity; evaluation receipt namespace not substituted', 'seed42_reused_heterogeneous_GPU': True, 'same_epoch_time_nll': True, 'source_metrics_reaggregated': False, 'new_training': False, 'new_inference': False, 'new_evaluation': False, 'remote_calls': 0, 'held_out_performance_read': False, 'mixed_performance_file_read': False, 'raw_prediction_read': False, 'checkpoint_binary_deserialized': False, 'missing_models': {'raf_spare_parts': ['titantpp']}, 'Deep_Renewal': 'Outside external-six/common-lognormal-head comparison; manuscript exclusion preserved.', 'frozen_contract_sha256': completion['contract_sha256'], 'builder_sha256': sha(__file__), 'outputs': {name: sha(HERE / name) for name in output_names}})
    comparison = json.loads((HERE / 'comparison_receipt.json').read_text())
    comparison['held_out_test_evaluated'] = False
    comparison['evidence_integrity_gate_passed'] = True
    comparison['scientific_adoption_gate'] = 'FAIL_TRADEOFF'
    comparison['seed42_original_endpoint_history_tolerance'] = 'Cross GPU original same-epoch replay allowed by original reuse audit; exact differences and tolerances recorded.'
    comparison['packaging_equivalence_conditions'] = [{k: b[k] for k in ('dataset', 'model', 'seed', 'input_identity', 'canonical_input_identity', 'time_input_equivalence')} for b in bindings if not b['input_file_SHA_exactly_canonical']]
    write_json('comparison_receipt.json', comparison)
    print(json.dumps({'status': 'complete', 'conditions': len(bindings), 'candidate_conditions': 9, 'metric_rows': len(rows), 'strong_dominance': [r['status'] for r in gates]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
