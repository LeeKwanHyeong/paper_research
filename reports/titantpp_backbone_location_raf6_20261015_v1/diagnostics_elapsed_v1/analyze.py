"""Saved RAF CNN Train/Validation rows only; no model imports or inference."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
BUNDLE = PROJECT / 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3'
CONTRACT_SHA = 'd51ab06365a7165e00cc18849b84caba4ee2dd8af6a36aee84a5c1d62e01fab3'
MANIFEST_SHA = '9b7044490727a42add4b9e05538fdf315137f0a956e25d8b1422f59a37430819'
PLAN_SHA = '1e07538aa13aab244457cde636abf9be68fc3bd92fee65c23ca897bc3df31d16'
SOURCE_CLOSURE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
SEEDS = (42, 52, 62)
SPLITS = ('train', 'validation')
FEATURES = {'last_observed_gap': 2, 'last_three_observed_span': 3}
KEYS = ('entity_id', 'target_quantity_bin', 'history_length', 'observed_change_magnitude_bin')
BOUNDARIES = (2., 30., 60., 200.)
INPUTS = {}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path, expected):
    path = Path(path)
    actual = sha(path)
    need(actual == expected, 'SHA differs: ' + str(path))
    INPUTS[str(path.relative_to(PROJECT))] = actual
    return path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def score(rows, predictions, weights=None):
    if not rows:
        return dict(rows=0, entities=0, weight_sum=0., ess=0., rmse=None, mae=None, bias=None,
                    weighted_sse=0., weighted_absolute_error_sum=0., weighted_error_sum=0.)
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=np.float64)
    e = np.asarray([predictions[(r['entity_id'], r['seq'])] - r['raw_quantity'] for r in rows])
    mass = float(w.sum())
    sse, ae, signed = float(np.dot(w, e * e)), float(np.dot(w, np.abs(e))), float(np.dot(w, e))
    return dict(rows=len(rows), entities=len({r['entity_id'] for r in rows}), weight_sum=mass,
                ess=mass * mass / float(np.dot(w, w)), rmse=math.sqrt(sse / mass), mae=ae / mass,
                bias=signed / mass, weighted_sse=sse, weighted_absolute_error_sum=ae,
                weighted_error_sum=signed)


def matching(rows, feature, cuts):
    low, high = cuts
    groups = defaultdict(lambda: {'low': [], 'high': []})
    for row in rows:
        value = row[feature]
        if value is None:
            continue
        side = 'low' if value <= low else 'high' if value >= high else None
        if side is not None:
            groups[tuple(row[k] for k in KEYS)][side].append(row)
    matched = {side: [] for side in ('low', 'high')}
    weights = {side: [] for side in ('low', 'high')}
    audit = []
    for key, sides in sorted(groups.items()):
        if not all(sides.values()):
            continue
        mass = min(len(sides['low']), len(sides['high']))
        audit.append(dict(zip(KEYS, key), low_n=len(sides['low']), high_n=len(sides['high']),
                          matched_mass_per_group=mass))
        for side in matched:
            matched[side].extend(sides[side])
            weights[side].extend([mass / len(sides[side])] * len(sides[side]))
    return matched, weights, audit


def weighted_balance(rows, weights):
    result = {}
    for field in ('raw_quantity', 'absolute_observed_log_change', 'last_quantity', 'history_length'):
        data = []
        for side in ('low', 'high'):
            values, w = np.asarray([r[field] for r in rows[side]]), np.asarray(weights[side])
            if not len(values):
                data.append(None)
                continue
            mean = float(np.dot(values, w) / w.sum())
            variance = float(np.dot((values - mean) ** 2, w) / w.sum())
            data.append((mean, variance))
        if any(v is None for v in data):
            result[field] = dict(low_mean=None, high_mean=None, high_minus_low=None, smd=None)
        else:
            difference = data[1][0] - data[0][0]
            scale = math.sqrt((data[0][1] + data[1][1]) / 2)
            result[field] = dict(low_mean=data[0][0], high_mean=data[1][0], high_minus_low=difference,
                                 smd=difference / scale if scale else 0. if difference == 0 else None)
    return result


def build_rows(frame, split, expected):
    """Mirror the frozen loader's target order and past-only context, without importing it."""
    need(split in SPLITS, 'Forbidden split')
    allowed = ('train',) if split == 'train' else SPLITS
    frame = frame.filter(pl.col('chronological_split').is_in(allowed)).sort(['oper_part_no', 'seq'])
    groups = frame.partition_by('oper_part_no', maintain_order=True)
    parts = [str(g['oper_part_no'][0]) for g in groups]
    part_indices, positions, sequences, quantities, output = [], [], [], [], []
    raw_gap_checks = 0
    for p, group in enumerate(groups):
        seq = group['seq'].to_numpy()
        q = group['demand_qty'].to_numpy().astype(np.float32).astype(np.float64)
        gaps = group['delta_t'].to_numpy()
        dates = group['demand_dt'].to_numpy()
        months = dates // 10000 * 12 + (dates // 100) % 100
        labels = group['chronological_split'].to_list()
        need(np.all(np.diff(seq) > 0), 'Duplicate/reversed within-entity sequence')
        need(np.array_equal(np.diff(seq), gaps[1:]), 'Recorded past gaps differ from sequence coordinate')
        need(np.array_equal(np.diff(months), gaps[1:]), 'Recorded past gaps differ from calendar month intervals')
        raw_gap_checks += len(seq) - 1
        for target in range(1, len(seq)):
            if labels[target] != split:
                continue
            end = target - 1
            start = max(int(np.searchsorted(seq, seq[end] - 83, side='left')), target - 83)
            indices = np.arange(start, target)
            need(len(indices) >= 1 and int(indices.max()) < target, 'Future or target in features')
            hq, hs = q[indices], seq[indices]
            history_length = len(indices)
            logq = np.log1p(hq)
            change = float(logq[-1] - logq[-2]) if history_length >= 2 else 0.
            row = dict(entity_id=parts[p], seq=int(seq[target]), split=split,
                       raw_quantity=float(q[target]), history_length=history_length,
                       last_quantity=float(hq[-1]), mean_quantity=float(hq.mean()),
                       history_log_mean=float(logq.mean()), history_log_std=float(logq.std()),
                       last_log_deviation=float(logq[-1] - logq.mean()),
                       last_observed_log_change=change, previous_observation_available=history_length >= 2,
                       absolute_observed_log_change=abs(change),
                       last_observed_gap=float(gaps[end]) if history_length >= 2 else None,
                       last_three_observed_span=float(gaps[end] + gaps[end - 1]) if history_length >= 3 else None,
                       target_quantity_bin=sum(q[target] > b for b in BOUNDARIES),
                       observed_change_magnitude_bin=math.floor(abs(change) / .05),
                       last_observed_seq=int(hs[-1]), first_observed_seq=int(hs[0]),
                       target_gap_used_for_feature=False)
            if history_length >= 3:
                need(row['last_three_observed_span'] == float(hs[-1] - hs[-3]), 'Span differs')
            output.append(row)
            part_indices.append(p); positions.append(target); sequences.append(int(seq[target])); quantities.append(float(q[target]))
    digest = hashlib.sha256(b'hard_lmm_target_identity_v1\0' + split.encode() + b'\0')
    digest.update(json.dumps(parts, ensure_ascii=False, separators=(',', ':')).encode())
    for values in (part_indices, positions, sequences):
        digest.update(np.asarray(values, dtype='<i8').tobytes())
    population = dict(target_count=len(output), target_identity_sha256=digest.hexdigest(),
                      target_quantity_sha256=hashlib.sha256(b'hard_lmm_target_quantity_v1\0' + np.asarray(quantities, dtype='<f8').tobytes()).hexdigest())
    need(population == expected, 'Population or quantity identity differs: ' + split)
    return output, dict(population=population, calendar_and_sequence_gap_pairs_verified=raw_gap_checks,
                        features_use_only_indices_before_target=True, loader_source_imported=False)


def self_checks():
    rows = [dict(entity_id='x', seq=i, raw_quantity=1., target_quantity_bin=0, history_length=3,
                 observed_change_magnitude_bin=0, last_observed_gap=g) for i, g in enumerate([1., 1., 10.])]
    m, w, cells = matching(rows, 'last_observed_gap', (1., 10.))
    need(w == {'low': [.5, .5], 'high': [1.]} and len(cells) == 1, 'Synthetic weights failed')
    s = score(m['low'], {('x', 0): 2., ('x', 1): 4.}, w['low'])
    need(s['weight_sum'] == 1. and s['mae'] == 2. and s['rmse'] == math.sqrt(5.), 'Synthetic scores failed')
    return {'matched_cell_equal_mass': True, 'hand_computed_weighted_rmse_mae_bias': True}


def main():
    need(not (OUTPUT / 'RESULT.json').exists(), 'Result exists; no retry/overwrite')
    pin(OUTPUT / 'PLAN.json', PLAN_SHA)
    plan = read(OUTPUT / 'PLAN.json')
    contract = read(pin(BUNDLE / 'contract.json', CONTRACT_SHA))
    need(contract['allowed_splits'] == list(SPLITS), 'Contract is not Train/Validation-only')
    need(contract['source']['closure'] == canonical(contract['source']['files']) == SOURCE_CLOSURE, 'Source closure differs')
    for rel, expected in contract['source']['files'].items():
        pin(BUNDLE / 'source' / rel, expected)
    for rel in ('operation/evaluate.py', 'operation/helper_taxi_raf24.py'):
        pin(BUNDLE / rel, contract['operation_files'][rel])
    manifest = read(pin(BUNDLE / 'retrieval/manifest.json', MANIFEST_SHA))
    need(manifest['contract_sha256'] == CONTRACT_SHA, 'Retrieval contract differs')
    host_path = BUNDLE / 'retrieval/runtime/5080/receipt.json'
    host = read(pin(host_path, manifest['files']['runtime/5080/receipt.json']))
    need(host['status'] == 'complete', 'Original inference incomplete')
    dataset = contract['datasets']['raf_spare_parts']
    path = pin(BUNDLE / dataset['path'], dataset['sha256'])
    raw = pl.scan_parquet(path).filter(pl.col('chronological_split').is_in(SPLITS)).collect()
    need(raw.height == 37469 and set(raw['chronological_split']) == set(SPLITS), 'Unexpected permitted data scope')
    definition = dataset['definition']
    need(definition['loader']['lookback_weeks'] == definition['loader']['max_seq_len'] == 84, 'Loader contract differs')
    rows, reconstruction = {}, {}
    for split in SPLITS:
        rows[split], reconstruction[split] = build_rows(raw, split, definition['inherited_data_identity']['populations'][split])
    predictions, evidence = {}, []
    jobs = [j for j in contract['jobs'] if j['dataset'] == 'raf_spare_parts']
    need(sorted(j['seed'] for j in jobs) == list(SEEDS), 'Missing CNN seed')
    for job in jobs:
        original = job['original']
        ckpt = original['checkpoint']
        pin(BUNDLE / ckpt['relative_path'], ckpt['sha256'])
        for name in ('checkpoint_metadata', 'selection_history', 'worker_exit', 'terminal_manifest'):
            pin(BUNDLE / original[name + '_path'], original[name + '_sha256'])
        meta = read(BUNDLE / original['checkpoint_metadata_path'])
        history = read(BUNDLE / original['selection_history_path'])['history']
        finite = [h for h in history if math.isfinite(h['val_qty_rmse'])]
        chosen = min(finite, key=lambda h: h['val_qty_rmse'])
        need(chosen['epoch'] == meta['best_epoch'] == original['selected_epoch'], 'First strict minimum differs')
        need(meta['model_state_sha256'] == original['checkpoint_state_sha256'], 'State binding differs')
        need(meta['resume_identity']['interface_meta']['source_files_sha256'] == SOURCE_CLOSURE, 'Metadata source differs')
        need(meta['resume_identity']['arguments']['data_sha256'] == dataset['sha256'], 'Selected input differs')
        for split in SPLITS:
            rel = f"results/{job['id']}/{split}/receipt.json"
            rec = read(pin(BUNDLE / 'retrieval' / rel, manifest['files'][rel]))
            worker = next(w for w in host['workers'] if w['job_id'] == job['id'] and w['split'] == split)
            need(worker['actual_worker_exit_code'] == 0 and worker['receipt']['sha256'] == manifest['files'][rel], 'Native worker evidence differs')
            need(rec['status'] == 'complete' and rec['contract_sha256'] == CONTRACT_SHA and rec['split'] == split
                 and rec['host'] == '5080' and rec['no_training'] and rec['parameters_unchanged']
                 and not rec['held_out_test_evaluated'], 'Prediction provenance differs')
            need(rec['checkpoint_sha256'] == ckpt['sha256'] and rec['selected_epoch'] == original['selected_epoch']
                 and rec['state_sha256'] == original['checkpoint_state_sha256'], 'Prediction checkpoint differs')
            need(rec['population'] == reconstruction[split]['population'], 'Prediction target identity differs')
            need(split != 'validation' or rec['baseline_validation_reproduced'], 'Original replay gate failed')
            paths = []
            for part in rec['prediction_manifest']['parts']:
                p = BUNDLE / 'retrieval/results' / job['id'] / split / 'predictions' / part['path']
                pin(p, part['sha256']); need(p.stat().st_size == part['bytes'], 'Prediction bytes differ'); paths.append(p)
            frame = pl.read_parquet(paths)
            need(frame.height == len(rows[split]) and set(frame['seed']) == {job['seed']}
                 and set(frame['dataset']) == {'raf_spare_parts'} and set(frame['split']) == {split}, 'Prediction scope differs')
            need(frame.select(pl.struct('entity_id', 'seq').n_unique()).item() == frame.height, 'Duplicate prediction target')
            records = {(r['entity_id'], r['seq']): r for r in frame.iter_rows(named=True)}
            common = ('raw_quantity', 'history_length', 'last_quantity', 'mean_quantity', 'history_log_mean',
                      'history_log_std', 'last_log_deviation', 'last_observed_log_change', 'previous_observation_available')
            for row in rows[split]:
                observed = records[(row['entity_id'], row['seq'])]
                for field in common:
                    need(math.isclose(row[field], observed[field], rel_tol=1e-12, abs_tol=1e-12), 'History reconstruction differs: ' + field)
                need(math.isfinite(observed['prediction_1']), 'Nonfinite baseline prediction')
            predictions[job['seed'], split] = {k: r['prediction_1'] for k, r in records.items()}
            baseline = score(rows[split], predictions[job['seed'], split])
            expected = rec['metrics']['1.0']
            need(math.isclose(baseline['rmse'], expected['qty_rmse'], rel_tol=1e-12, abs_tol=1e-12)
                 and math.isclose(baseline['mae'], expected['qty_mae'], rel_tol=1e-12, abs_tol=1e-12), 'Stored scores differ')
            evidence.append(dict(seed=job['seed'], split=split, receipt=rel, receipt_sha256=manifest['files'][rel],
                                 selected_epoch=original['selected_epoch'], binary_sha256=ckpt['sha256'],
                                 state_sha256=original['checkpoint_state_sha256'], parts=rec['prediction_manifest']['parts'],
                                 population=rec['population'], runtime=rec['runtime'], baseline_scores=baseline))
    cuts, results, samples = {}, [], []
    for feature in FEATURES:
        train_values = np.asarray([r[feature] for r in rows['train'] if r[feature] is not None])
        cuts[feature] = dict(q25=float(np.quantile(train_values, .25)), q75=float(np.quantile(train_values, .75)),
                             eligible_train_targets=len(train_values), tied_cuts=False)
        lower, upper = cuts[feature]['q25'], cuts[feature]['q75']
        cuts[feature]['tied_cuts'] = lower >= upper
        for split in SPLITS:
            original = rows[split]
            eligible = [r for r in original if r[feature] is not None]
            low_high = {side: [r for r in eligible if r[feature] <= lower] if side == 'low'
                        else [r for r in eligible if r[feature] >= upper] for side in ('low', 'high')}
            matched, weights, cells = matching(original, feature, (lower, upper)) if lower < upper else ({'low': [], 'high': []}, {'low': [], 'high': []}, [])
            support = dict(feature=feature, split=split, targets=len(original), eligible=len(eligible),
                           insufficient_history=len(original) - len(eligible), middle_excluded=len(eligible)-sum(map(len, low_high.values())) if lower < upper else None,
                           low_n=len(low_high['low']), high_n=len(low_high['high']), matched_cells=len(cells),
                           matched_low=len(matched['low']), matched_high=len(matched['high']),
                           matched_unique_targets=len(matched['low']) + len(matched['high']),
                           matched_entities=len({r['entity_id'] for side in matched for r in matched[side]}),
                           matched_population_fraction=(len(matched['low']) + len(matched['high'])) / len(original),
                           matched_low_high_fraction=(len(matched['low']) + len(matched['high'])) / max(1, sum(map(len, low_high.values()))),
                           matched_weight_per_group=sum(weights['low']), balance=weighted_balance(matched, weights),
                           common_cells=cells, tied_cuts=lower >= upper)
            samples.append(support)
            for seed in SEEDS:
                prediction = predictions[seed, split]
                for method, subset, w in (('unmatched', low_high, None), ('matched', matched, weights)):
                    scores = {side: score(subset[side], prediction, None if w is None else w[side]) for side in ('low', 'high')}
                    contrast = {m: scores['high'][m] - scores['low'][m] if all(scores[s][m] is not None for s in scores) else None
                                for m in ('rmse', 'mae', 'bias')}
                    results.append(dict(feature=feature, split=split, seed=seed, method=method, **scores, high_minus_low=contrast))
    aggregates = []
    for feature in FEATURES:
        for split in SPLITS:
            for method in ('unmatched', 'matched'):
                values = [r for r in results if (r['feature'], r['split'], r['method']) == (feature, split, method)]
                stats = {}
                for side in ('low', 'high', 'high_minus_low'):
                    stats[side] = {}
                    for metric in ('rmse', 'mae', 'bias'):
                        v = [r[side][metric] for r in values]
                        stats[side][metric] = dict(mean=statistics.mean(v), sample_sd=statistics.stdev(v)) if all(x is not None for x in v) else dict(mean=None, sample_sd=None)
                aggregates.append(dict(feature=feature, split=split, method=method, **stats))
    result = dict(schema='raf_cnn_observed_elapsed_saved_rows_v1', created_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                  plan_sha256=PLAN_SHA, source_closure=SOURCE_CLOSURE, contract_sha256=CONTRACT_SHA,
                  cuts=cuts, reconstruction=reconstruction, evidence=evidence, support=samples,
                  per_seed=results, three_seed=aggregates, inputs=INPUTS,
                  self_checks=self_checks(), inference_count=0, remote_count=0, held_out_test_read=False,
                  interpretation=plan['interpretation'])
    write(OUTPUT / 'RESULT.json', result)
    lines = ['# RAF CNN 저장 예측의 관측 간격 진단', '', 'Train·Validation만 사용했다. 새 추론·학습·원격 호출은 없다.', '',
             '간격은 예측 직전까지 관측된 이력에서 계산했다. 다음 target까지의 간격을 특징으로 쓰지 않았다.',
             '수량 구간은 정답 수량으로 정의한 기술적 비교 조건이다. 모델 입력이나 사전 예측 규칙이 아니다.', '',
             '| 조건 | split | 전체/유효 | 매칭 low/high | 개체/셀 | 전체 포함률 | 방향별 가중치 |', '|---|---|---:|---:|---:|---:|---:|']
    for r in samples:
        lines.append(f"| {r['feature']} | {r['split']} | {r['targets']}/{r['eligible']} | {r['matched_low']}/{r['matched_high']} | {r['matched_entities']}/{r['matched_cells']} | {r['matched_population_fraction']:.3%} | {r['matched_weight_per_group']:.1f} |")
    lines += ['', 'Train Q25/Q75: ' + json.dumps(cuts, ensure_ascii=False), '', '| 조건 | split | seed | matched high−low RMSE | MAE | bias |', '|---|---|---:|---:|---:|---:|']
    for r in results:
        if r['method'] == 'matched':
            values = [f'{r["high_minus_low"][k]:+.6f}' if r['high_minus_low'][k] is not None else 'N/A' for k in ('rmse', 'mae', 'bias')]
            lines.append(f"| {r['feature']} | {r['split']} | {r['seed']} | " + ' | '.join(values) + ' |')
    lines += ['', '전체 low/high 원시 비교, 매칭 전후 RMSE·MAE·bias·SSE, 가중 표본수·ESS·잔여 균형, 3seed 평균·표본SD는 RESULT.json에 보존했다.',
              'seed 3개는 같은 대상의 세 모델이다. 관측 표본수를 세 배로 세지 않는다.',
              '같은 개체·정답 수량 구간·정확한 이력 길이·관측 변화 폭 구간이 모두 같아야 비교에 남는다. 좁은 지원 범위는 RAF 전체의 근거가 아니다.',
              '동일 bin 안의 연속 수량·변화 폭 및 마지막 관측 수량 차이가 남을 수 있다. 경과시간이나 CNN 필터가 원인이라고 확정하지 않는다.',
              '표본 부족을 이유로 bin이나 분위수를 바꿔 재탐색하지 않았다. 이 결과만으로 새 학습을 정당화하지 않는다.']
    with (OUTPUT / 'RESULT.md').open('x') as stream:
        stream.write('\n'.join(lines) + '\n')
    receipt = dict(status='complete', schema='raf_cnn_observed_elapsed_receipt_v1', allowed_splits=list(SPLITS),
                   baseline_model_count=3, baseline_evaluation_runs=0, inference_count=0, remote_count=0,
                   held_out_test_read=False, source_files_verified=len(contract['source']['files']),
                   selected_binary_count=3, prediction_splits_verified=6, inputs=INPUTS,
                   artifacts={p.name: sha(p) for p in (OUTPUT / 'PLAN.json', Path(__file__), OUTPUT / 'RESULT.json', OUTPUT / 'RESULT.md')})
    write(OUTPUT / 'RECEIPT.json', receipt)
    print(json.dumps({'status': 'complete', 'cuts': cuts, 'support': [{k: v for k, v in r.items() if k not in ('common_cells', 'balance')} for r in samples],
                      'result_sha256': sha(OUTPUT / 'RESULT.json'), 'receipt_sha256': sha(OUTPUT / 'RECEIPT.json')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
