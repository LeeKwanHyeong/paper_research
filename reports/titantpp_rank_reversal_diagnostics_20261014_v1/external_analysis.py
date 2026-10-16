"""Stored-evidence-only CNN versus six external encoders, no inference/imports.

Test parquet is reaggregated after each receipt/part SHA and target/truth check.
Validation uses the stored selected endpoint quantity cells. Its unavailable
signed-error sums and raw target rows are never reconstructed from assumptions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import unittest

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'reports/titantpp_components_taxi_raf24_test_20261014_v1'
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
SEEDS = (42, 52, 62)
EXTERNAL = ('s2p2_matched_head', 'attnhp_matched_head', 'rmtpp', 'thp', 'sahp', 'nhp')
BOUNDS = {'yellow_trip_hourly': [7., 686., 1562., 3449.],
          'raf_spare_parts': [2., 30., 60., 200.],
          'intermittent_frozen_5000': [2., 31., 46., 187.]}
COMMON_TRUTH = ('target_id', 'entity_id', 'seq', 'split', 'recorded_date',
                'raw_quantity', 'recorded_gap', 'history_length', 'last_quantity', 'mean_quantity')
NAMES = {'yellow_trip_hourly': 'Taxi', 'raf_spare_parts': 'RAF', 'intermittent_frozen_5000': 'Intermittent'}
SOURCES = {}


def require(flag, message):
    if not flag:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path, expected=None):
    path = Path(path)
    actual = sha(path)
    require(expected is None or actual == expected, 'Source SHA differs: ' + str(path))
    SOURCES[str(path.relative_to(ROOT))] = actual
    return json.loads(path.read_text())


def compact(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()


def close(a, b, label, *, rel=2e-9, absolute=1e-8):
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=rel, abs_tol=absolute),
            f'{label}: {a} != {b}')


def normalized_cells(metrics, dataset, *, signed_available):
    require(metrics['quantity_boundaries'] == BOUNDS[dataset], 'Quantity boundaries changed')
    cells = metrics['quantity_cells']
    require([c['bin'] for c in cells] == list(range(5)), 'Missing/duplicate quantity cells')
    result = []
    for c in cells:
        n = c['count']
        require(type(n) is int and n > 0, 'Empty/noninteger fixed bin')
        absolute = c.get('qty_absolute_error_sum', c['qty_mae'] * n)
        close(c['qty_sse'], c['qty_rmse'] ** 2 * n, 'Bin RMSE/SSE')
        close(absolute, c['qty_mae'] * n, 'Bin MAE/absolute sum')
        result.append({'bin': c['bin'], 'count': n, 'share': n / metrics['count'],
                       'sse': c['qty_sse'], 'absolute_error_sum': absolute,
                       'mse': c['qty_sse'] / n, 'mae': absolute / n,
                       'signed_error_sum': c.get('qty_signed_error_sum') if signed_available else None,
                       'bias': c.get('qty_signed_error_sum', 0) / n if signed_available else None})
    require(sum(c['count'] for c in result) == metrics['count'], 'Bin counts do not conserve population')
    close(sum(c['sse'] for c in result), metrics['qty_rmse'] ** 2 * metrics['count'], 'Overall SSE conservation')
    close(sum(c['absolute_error_sum'] for c in result), metrics['qty_mae'] * metrics['count'], 'Overall absolute conservation')
    return result


def component_row(row):
    bundle = ('titantpp_components_taxi_raf24_test_20261014_v1' if row['dataset'] != DATASETS[2]
              else 'titantpp_components_inter62_test_20261014_v1' if row['seed'] == 62
              else 'titantpp_components_inter8_test_20261014_v1')
    filename = 'validation_replay.json' if row['split'] == 'validation' else 'test_metrics.json'
    path = ROOT / 'search_artifacts' / bundle / 'results' / row['id'] / filename
    receipt = read(path, row['receipt_sha256'])
    require(receipt['raw_predictions_written'] is False and receipt['status'] == 'complete'
            and receipt['checkpoint_sha256'] == row['checkpoint_sha256']
            and receipt['selected_epoch'] == row['selected_epoch']
            and receipt['retrained'] is False and receipt['reselected'] is False, 'CNN receipt identity/selection changed')
    require(receipt['native_runtime_checked'] is True and receipt['runtime'] == receipt['runtime_after'], 'CNN Runtime receipt differs')
    for key, expected in row['population'].items():
        require(receipt[key] == expected, 'CNN population fingerprint differs')
    cells = normalized_cells(receipt['metrics'], row['dataset'], signed_available=True)
    return {**row, 'quantity_cells': cells, 'source_path': str(path.relative_to(ROOT)),
            'data_sha256': receipt['data_sha256'], 'runtime': receipt['runtime'],
            'population_verification': 'native evaluation receipt SHA and fixed population fingerprints',
            'raw_predictions_available': False}


def external_validation(row, test_receipt):
    receipt = read(ROOT / row['source_path'], row['source_raw_sha256'])
    selected = receipt['selected']
    require(receipt.get('status') in (None, 'complete') and receipt['best_epoch'] == row['selected_epoch']
            and receipt['completed_epochs'] >= receipt['best_epoch'] >= 1,
            'External selected epoch differs')
    require(selected['evaluation_scope'] == 'validation_only' and selected['held_out_test_evaluated'] is False,
            'External selected endpoint scope differs')
    require(selected['state_sha256'] == test_receipt['state_tensor_sha256'], 'Validation/Test selected tensors differ')
    close(selected['qty_rmse'], row['metrics']['overall_qty_rmse'], 'External Validation RMSE')
    close(selected['qty_mae'], row['metrics']['overall_qty_mae'], 'External Validation MAE')
    cells = normalized_cells(selected, row['dataset'], signed_available=False)
    return {**row, 'quantity_cells': cells, 'population_verification':
            'stored fullValidation endpoint counts, Train-fixed boundaries and selected tensor; raw Validation truth unavailable',
            'raw_predictions_available': False, 'signed_error_available': False,
            'selected_tensor_sha256': selected['state_sha256'],
            'training_endpoint': {'selected_epoch': receipt['best_epoch'], 'completed_epochs': receipt['completed_epochs'],
                'last_validation_rmse': receipt['last']['qty_rmse'], 'selected_validation_rmse': selected['qty_rmse'],
                'last_minus_selected_validation_rmse': receipt['last']['qty_rmse'] - selected['qty_rmse'],
                'first40': receipt.get('first40'), 'last30': receipt.get('last30'),
                'actual_train_raw_quantity_errors_available': False}}


def external_test(row, cnn):
    path = ROOT / row['source_path']
    receipt = read(path, row['source_raw_sha256'])
    require(receipt['status'] == 'complete' and receipt['full_population'] is True and receipt['retrained'] is False
            and receipt['selected_epoch'] == row['selected_epoch'], 'External Test receipt incomplete/identity differs')
    require(receipt['data_file_sha256'] == cnn['data_sha256'], 'CNN/external data bytes differ')
    require(receipt['target_identity_sha256'] == cnn['population']['target_identity_sha256'], 'Stored CNN/external target IDs differ')
    sums = np.zeros((5, 5), dtype=np.float64)
    target, old_truth, common_truth = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    time_hash = hashlib.sha256(b'titantpp_components_recorded_target_time_v1\0test\0')
    count = 0
    all_parts = []
    for part in receipt['parts']:
        require(not Path(part['path']).is_absolute() and '..' not in Path(part['path']).parts, 'Unsafe prediction path')
        p = path.parent / part['path']
        require(sha(p) == part['sha256'], 'Stored Test prediction bytes differ')
        SOURCES[str(p.relative_to(ROOT))] = part['sha256']
        frame = pl.read_parquet(p)
        require(frame.height == part['rows'] and frame['target_index'].to_list() == list(range(count, count + frame.height)),
                'Missing/duplicate/out-of-order stored Test targets')
        require(frame['split'].unique().to_list() == ['test'], 'Mixed prediction split')
        q = frame['raw_quantity'].to_numpy()
        prediction = frame['predicted_raw_quantity'].to_numpy()
        temporal = frame['time_nll'].to_numpy()
        require(np.isfinite(q).all() and np.isfinite(prediction).all() and np.isfinite(temporal).all(), 'Nonfinite stored vectors')
        bins = np.searchsorted(BOUNDS[row['dataset']], q, side='left')
        errors = prediction - q
        for i in range(5):
            mask = bins == i
            err = errors[mask]
            sums[i] += [int(mask.sum()), np.sum(err * err), np.sum(np.abs(err)), np.sum(err), np.sum(temporal[mask])]
        pt, po = hashlib.sha256(), hashlib.sha256()
        for raw in frame.iter_rows(named=True):
            id_bytes = (raw['target_id'] + '\n').encode()
            old = compact([raw[k] for k in receipt['truth_digest_fields']]) + b'\n'
            common = compact([raw[k] for k in COMMON_TRUTH]) + b'\n'
            target.update(id_bytes); pt.update(id_bytes)
            old_truth.update(old); po.update(old); common_truth.update(common)
        require(pt.hexdigest() == part['target_identity_sha256'] and po.hexdigest() == part['truth_sha256'], 'Part row truth fingerprints differ')
        time_hash.update(frame['recorded_gap'].to_numpy().astype('<f8').tobytes())
        count += frame.height
        all_parts.append({'path': str(p.relative_to(ROOT)), 'sha256': part['sha256'], 'count': frame.height})
    require(count == receipt['prediction_rows'] == cnn['count'], 'CNN/external Test target count differs')
    require(target.hexdigest() == receipt['target_identity_sha256'] == cnn['population']['target_identity_sha256'], 'Recomputed target order differs')
    require(old_truth.hexdigest() == receipt['truth_sha256'], 'Legacy Test full truth digest differs')
    require(common_truth.hexdigest() == cnn['population']['truth_sha256'], 'CNN/external common truth fields differ')
    require(time_hash.hexdigest() == cnn['population']['time_sha256'], 'CNN/external recorded time differs')
    metrics = {'count': count, 'quantity_boundaries': BOUNDS[row['dataset']], 'quantity_cells': [],
               'qty_rmse': math.sqrt(float(sums[:, 1].sum()) / count), 'qty_mae': float(sums[:, 2].sum()) / count}
    for i, (n, sse, absolute, signed, temporal) in enumerate(sums):
        n = int(n)
        metrics['quantity_cells'].append({'bin': i, 'count': n, 'qty_sse': float(sse), 'qty_absolute_error_sum': float(absolute),
            'qty_signed_error_sum': float(signed), 'qty_rmse': math.sqrt(sse / n), 'qty_mae': absolute / n, 'time_nll': temporal / n})
    close(metrics['qty_rmse'], row['metrics']['overall_qty_rmse'], 'External reaggregated Test RMSE')
    close(metrics['qty_mae'], row['metrics']['overall_qty_mae'], 'External reaggregated Test MAE')
    close(float(sums[:, 4].sum()) / count, row['metrics']['overall_time_nll'], 'External reaggregated Test TimeNLL')
    close(math.sqrt(sums[-1, 1] / sums[-1, 0]), row['metrics']['tail_qty_rmse'], 'External Test tail RMSE')
    close(sums[-1, 2] / sums[-1, 0], row['metrics']['tail_qty_mae'], 'External Test tail MAE')
    close(sums[-1, 4] / sums[-1, 0], row['metrics']['tail_time_nll'], 'External Test tail TimeNLL')
    return ({**row, 'quantity_cells': normalized_cells(metrics, row['dataset'], signed_available=True),
            'population_verification': 'all part SHA + target order + legacy full truth + common CNN truth + recorded time SHA',
            'common_truth_sha256': common_truth.hexdigest(), 'common_time_sha256': time_hash.hexdigest(),
            'raw_predictions_available': True, 'signed_error_available': True,
            'runtime': receipt['runtime'], 'prediction_parts': all_parts}, receipt)


def difference_cells(cnn, external, metric):
    result = []
    for a, b in zip(cnn['quantity_cells'], external['quantity_cells']):
        require(a['bin'] == b['bin'] and a['count'] == b['count'], 'CNN/external quantity cell populations differ')
        value = a[metric] - b[metric]
        result.append({'bin': a['bin'], 'count': a['count'], 'share': a['share'],
                       'CNN_minus_external_conditional_error': value,
                       'contribution_to_overall_difference': a['share'] * value,
                       'CNN_conditional_error': a[metric], 'external_conditional_error': b[metric],
                       'CNN_bias': a['bias'], 'external_bias': b['bias']})
    return result


def decompose(validation, test):
    """Symmetric, exact arithmetic decomposition; not a causal mediation model."""
    parts = []
    for v, t in zip(validation, test):
        require(v['bin'] == t['bin'], 'Reordered strata')
        pv, pt = v['share'], t['share']
        dv, dt = v['CNN_minus_external_conditional_error'], t['CNN_minus_external_conditional_error']
        mix = (pt - pv) * (dt + dv) / 2
        conditional = (pt + pv) * (dt - dv) / 2
        delta = t['contribution_to_overall_difference'] - v['contribution_to_overall_difference']
        close(mix + conditional, delta, 'Cell symmetric decomposition')
        parts.append({'bin': v['bin'], 'validation_share': pv, 'test_share': pt,
                      'composition_change_contribution': mix, 'within_bin_error_change_contribution': conditional,
                      'total_difference_change_contribution': delta})
    change = sum(t['contribution_to_overall_difference'] for t in test) - sum(v['contribution_to_overall_difference'] for v in validation)
    close(sum(p['composition_change_contribution'] + p['within_bin_error_change_contribution'] for p in parts), change,
          'Overall symmetric decomposition')
    return {'difference_change': change, 'composition_change': sum(p['composition_change_contribution'] for p in parts),
            'within_bin_error_change': sum(p['within_bin_error_change_contribution'] for p in parts), 'cells': parts,
            'causal_attribution': False, 'decomposition_reference': 'symmetric midpoint; alternative references alter components, not their sum'}


def metric_map(row, cnn=False):
    if cnn:
        names = ('전체 RMSE', '전체 MAE', '큰 수량 RMSE', '큰 수량 MAE', '전체 TimeNLL', '큰 수량 TimeNLL')
        keys = ('overall_qty_rmse', 'overall_qty_mae', 'tail_qty_rmse', 'tail_qty_mae', 'overall_time_nll', 'tail_time_nll')
        return dict(zip(keys, [row['metrics'][n] for n in names]))
    return row['metrics']


def make_report(result):
    lines = ['# CNN과 외부 비교군의 Validation→Test 순위 변화', '',
      'Material Passport: 기존 저장 endpoint·Test 예측만 재집계했다. 신규 학습·추론·원격 조회·checkpoint 재선택은 없다. CNN은 source131 구성요소 캠페인, 외부 6개는 기존 공통 head encoder 캠페인이다. 기존 접근 이력이 있는 chronological Test이다.', '',
      '## 평균 결과와 seed별 방향', '',
      '각 차이는 CNN−외부 모델의 3seed 평균이다. 음수는 CNN 우세다. RMSE 평균과 평균 MSE는 별도 값이며, 아래 오류 분해는 MSE에 수행했다.', '',
      '| 데이터셋 | 비교 모델 | Validation RMSE 차이 | Test RMSE 차이 | Validation MAE 차이 | Test MAE 차이 | RMSE seed 순위 반전 수 |',
      '|---|---|---:|---:|---:|---:|---:|']
    for x in result['comparisons']:
        lines.append(f"| {NAMES[x['dataset']]} | {x['external_arm']} | {x['validation']['overall_qty_rmse']['mean_difference']:.4f} | {x['test']['overall_qty_rmse']['mean_difference']:.4f} | {x['validation']['overall_qty_mae']['mean_difference']:.4f} | {x['test']['overall_qty_mae']['mean_difference']:.4f} | {x['metric_direction_reversals']['overall_qty_rmse']} / 3 |")
    lines += ['', '## 공통 기준선 S2P2 대비 오류 반전의 산술 위치', '',
      '전체 차이 변화는 Train 고정 수량 구간의 구성비 변화와 각 구간 내 상대 오류 변화로 정확히 분해된다. 이 분해는 오류 위치를 설명하며 원인의 인과적 증명이 아니다. 구성비 항에는 구간 내부 값·관측 이력의 분포 변화가 포함되지 않는다.', '',
      '| 데이터셋 | 평균 MSE 차이 변화 | 구성비 변화 항 | 구간 내 오류 변화 항 | q>Train 경계의 변화 기여 |',
      '|---|---:|---:|---:|---:|']
    for x in result['comparisons']:
        if x['external_arm'] != 's2p2_matched_head':
            continue
        d = x['decomposition_mean_of_seed_mse']
        lines.append(f"| {NAMES[x['dataset']]} | {d['difference_change']:.5f} | {d['composition_change']:.5f} | {d['within_bin_error_change']:.5f} | {d['cells'][-1]['total_difference_change_contribution']:.5f} |")
    lines += ['', '## 확인된 사실과 해석의 한계', '',
      '- Taxi의 CNN은 Validation RMSE·MAE에서 외부 6개 평균보다 좋았지만 Test에서는 S2P2와 AttNHP에 두 지표 모두 뒤졌다. 다른 네 외부 모델에는 Test 평균 두 지표가 여전히 좋다.',
      '- Intermittent의 CNN은 Validation과 Test에서 외부 6개보다 평균 RMSE·MAE가 낮다. 외부 비교 기준의 전체 순위 반전으로 일반화할 수 없다.',
      '- RAF는 Validation RMSE에서 S2P2에 뒤졌고, Test에서는 CNN의 평균 RMSE가 외부 6개 중 최저다. 이 방향은 악화 반전과 다르며 MAE도 별도 비교해야 한다.',
      '- Test의 모든 외부 parquet 부분 SHA·대상 순서·truth SHA를 검증했다. CNN과 공통 10개 truth 필드 및 recorded time SHA도 재검증했다. Validation은 고정 선택 tensor·대상 수·5개 구간별 대상 수를 확인했으나 외부의 원시 Validation 대상 지문은 endpoint에서 복구할 수 없다.',
      '- 5개 수량 구간의 구성비 변화가 작아도 구간 내부의 수량·이력·개체·시간 조건 변화는 남을 수 있다. 구성비 항이 작다는 사실은 모든 분포 변화가 없다는 증거가 아니다.',
      '- CNN의 원시행 예측이 저장되지 않아 두 모델의 같은 행 오류를 시간·개체별로 연결할 수 없다. 외부 예측만으로 CNN의 특정 시점 실패를 추정하지 않았다.',
      '- 외부 Validation의 signed error가 없어 외부의 Validation→Test 과소·과대 예측 변화는 계산할 수 없다. Test에서만 구간별 bias를 확인했다.',
      '- Train 로그의 log 수량 손실은 raw 수량 RMSE와 직접 비교할 수 없다. 이 자료만으로 과적합, GRU 특성 또는 분포 변화 중 하나를 확정할 수 없다.',
      '- 외부 학습 source·캠페인·Runtime과 CNN source131은 다르다. 같은 입력·선택·공통 head 비교이며 동일 Runtime 재학습 대조가 아니다. Intermittent seed42/52는 5090, seed62는 5080이므로 시간은 합치지 않았다.',
      '- 3seed 평균 순위와 seed별 예외는 기술 통계다. 통계적 유의성·모든 데이터셋 우월성·원인 규명 완료를 주장하지 않는다.', '',
      '전체 six-metric seed별 차이, 모든 외부 모델의 5개 bin 기여, 검증 SHA 및 보존 endpoint 학습 말기 지표는 EXTERNAL_ANALYSIS.json에 기록했다.']
    return '\n'.join(lines) + '\n'


def run():
    component = read(BASE / 'COMPONENTS36_ANALYSIS.json')
    reused = read(BASE / 'REUSED_COHORT_TABLES.json')
    require(component['test_job_count'] == 36 and component['seed_count'] == 3 and component['new_training'] is False, 'Component36 incomplete')
    cohort = next(c for c in reused['cohorts'] if c['cohort'] == 'representative9_source117_123')
    cnn = {(r['dataset'], r['seed'], r['split']): component_row(r) for r in component['rows'] if r['arm'] == 'CNN'}
    require(len(cnn) == 18, 'CNN9 Validation/Test coverage differs')
    inputs = {(r['dataset'], r['arm'], r['seed'], r['split']): r for r in cohort['rows'] if r['arm'] in EXTERNAL}
    require(len(inputs) == 108, 'External54 Validation/Test coverage differs')
    external = {}
    for dataset in DATASETS:
        for arm in EXTERNAL:
            for seed in SEEDS:
                test, receipt = external_test(inputs[dataset, arm, seed, 'test'], cnn[dataset, seed, 'test'])
                validation = external_validation(inputs[dataset, arm, seed, 'validation'], receipt)
                external[dataset, arm, seed, 'test'] = test
                external[dataset, arm, seed, 'validation'] = validation
    comparisons = []
    for dataset in DATASETS:
        for arm in EXTERNAL:
            item = {'dataset': dataset, 'external_arm': arm, 'seed_rows': [], 'metric_direction_reversals': {}}
            for split in ('validation', 'test'):
                item[split] = {}
                for metric in metric_map(external[dataset, arm, 42, split]):
                    values = [metric_map(cnn[dataset, s, split], cnn=True)[metric] - metric_map(external[dataset, arm, s, split])[metric] for s in SEEDS]
                    item[split][metric] = {'mean_difference': statistics.mean(values), 'sample_sd_difference': statistics.stdev(values),
                        'strict_CNN_wins': sum(v < 0 for v in values), 'seed_differences': dict(zip(map(str, SEEDS), values))}
            for metric in item['validation']:
                item['metric_direction_reversals'][metric] = sum(item['validation'][metric]['seed_differences'][str(s)] *
                    item['test'][metric]['seed_differences'][str(s)] < 0 for s in SEEDS)
            for seed in SEEDS:
                seed_row = {'seed': seed}
                for metric in ('mse', 'mae'):
                    v = difference_cells(cnn[dataset, seed, 'validation'], external[dataset, arm, seed, 'validation'], metric)
                    t = difference_cells(cnn[dataset, seed, 'test'], external[dataset, arm, seed, 'test'], metric)
                    seed_row[metric] = {'validation_difference': sum(c['contribution_to_overall_difference'] for c in v),
                        'test_difference': sum(c['contribution_to_overall_difference'] for c in t),
                        'validation_cells': v, 'test_cells': t, 'decomposition': decompose(v, t)}
                item['seed_rows'].append(seed_row)
            for metric in ('mse', 'mae'):
                decs = [s[metric]['decomposition'] for s in item['seed_rows']]
                avg = {key: statistics.mean(d[key] for d in decs) for key in ('difference_change', 'composition_change', 'within_bin_error_change')}
                avg['cells'] = [{key: statistics.mean(d['cells'][i][key] for d in decs) for key in decs[0]['cells'][i]} for i in range(5)]
                for i in range(5):
                    avg['cells'][i]['bin'] = i
                close(avg['composition_change'] + avg['within_bin_error_change'], avg['difference_change'], 'Mean decomposition')
                item['decomposition_mean_of_seed_' + metric] = avg
            comparisons.append(item)
    anchors = {}
    for dataset in DATASETS:
        anchors[dataset] = {'fixed_shared_anchor': 's2p2_matched_head', 'metric_best_external_descriptive_only': {}}
        for split in ('validation', 'test'):
            anchors[dataset]['metric_best_external_descriptive_only'][split] = {}
            for metric in ('overall_qty_rmse', 'overall_qty_mae'):
                means = {arm: statistics.mean(metric_map(external[dataset, arm, s, split])[metric] for s in SEEDS) for arm in EXTERNAL}
                anchors[dataset]['metric_best_external_descriptive_only'][split][metric] = {'arm': min(means, key=means.get), 'means': means,
                    'selected_or_retrained_model': False, 'post_test_selection_for_followup': False}
    result = {'schema': 'CNN_external6_stored_evidence_rank_reversal_v1', 'status': 'complete',
        'material_passport': {'origin_skill': 'academic-research-suite', 'mode': 'stored-evidence diagnostics',
            'source131_CNN_vs_prior_matched_head_external': True},
        'new_training': False, 'new_inference': False, 'remote_calls': 0, 'checkpoint_reselection': False,
        'Test_independent': False, 'statistical_significance_established': False, 'causal_identification_established': False,
        'CNN_conditions_per_split': 9, 'external_conditions_per_split': 54, 'Train_fixed_quantity_boundaries': BOUNDS,
        'MSE_definition': 'mean of per-seed squared RMSE; distinct from squared mean RMSE',
        'comparisons': comparisons, 'anchors': anchors, 'CNN_rows': list(cnn.values()), 'external_rows': list(external.values()),
        'Validation_external_raw_truth_fingerprints_available': False, 'CNN_raw_prediction_rows_available': False,
        'source_references': SOURCES, 'script_sha256': sha(__file__)}
    json_path = OUT / 'EXTERNAL_ANALYSIS.json'
    md_path = OUT / 'EXTERNAL_ANALYSIS.md'
    with json_path.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2); stream.write('\n')
    with md_path.open('x') as stream:
        stream.write(make_report(result))
    print(json.dumps({'status': result['status'], 'json': str(json_path), 'json_sha256': sha(json_path),
                      'md': str(md_path), 'md_sha256': sha(md_path), 'verified_source_files': len(SOURCES)}))


class FocusedTests(unittest.TestCase):
    def test_midpoint_decomposition_conserves_with_offsetting_bins(self):
        def cell(i, p, d):
            return {'bin': i, 'share': p, 'CNN_minus_external_conditional_error': d, 'contribution_to_overall_difference': p * d}
        d = decompose([cell(0, .9, -10), cell(1, .1, 30)], [cell(0, .5, 20), cell(1, .5, -5)])
        self.assertAlmostEqual(d['difference_change'], 13.5)
        self.assertAlmostEqual(d['composition_change'] + d['within_bin_error_change'], 13.5)

    def test_cell_population_mismatch_blocked(self):
        with self.assertRaises(ValueError):
            difference_cells({'quantity_cells': [{'bin': 0, 'count': 1}]}, {'quantity_cells': [{'bin': 0, 'count': 2}]}, 'mse')

    def test_bin_boundary_is_closed_on_right(self):
        self.assertEqual(np.searchsorted([2, 30, 60, 200], [2, 3, 30, 200, 201], side='left').tolist(), [0, 1, 1, 3, 4])

    def test_nonfinite_decomposition_rejected(self):
        def cell(v):
            return {'bin': 0, 'share': 1., 'CNN_minus_external_conditional_error': v, 'contribution_to_overall_difference': v}
        with self.assertRaises(ValueError):
            decompose([cell(1)], [cell(float('nan'))])


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        run()
