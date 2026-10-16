"""Read-only source131 FULL versus CNN diagnostics from sealed saved sums.

Stdlib only: no inference, checkpoint deserialization, source modifications,
remote access, learning, or Test-based checkpoint reselection.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics

PROJECT = Path(__file__).resolve().parents[2]
REPORT = Path(__file__).resolve().parent
SOURCE = '15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'
DATASETS = ('yellow_trip_hourly', 'raf_spare_parts', 'intermittent_frozen_5000')
NAMES = dict(zip(DATASETS, ('Taxi', 'RAF', 'Intermittent')))
SEEDS = (42, 52, 62)
ARMS = {'titantpp_cnn_gru54': 'FULL', 'titantpp_cnn_gru54_cnn_only': 'CNN'}
BUNDLES = ('taxi_raf24', 'inter8', 'inter62')
CONTRACTS = ('96f6c12ea77ed0bc35447c8d363f1e2229d288a5552c8908e91c5ed24f44a5c0',
             '432b300eab9025fc4a0b7c5670c0e8ba3f64c51eda50811d678f5363b9307615',
             '85c5dc1e1ac91d6b138058ea1838c0da750674279151a2dba3bb3c087a08482a')
SOURCES = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    require(expected is None or actual == expected, 'SHA mismatch: ' + str(path))
    SOURCES[str(path)] = actual
    return json.loads(path.read_bytes())


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8)


def signed_share(part, total):
    # A signed gap is not an error fraction. Undefined when its denominator
    # is zero or numerically too small; values >1 and negatives are retained.
    if abs(total) <= max(1e-10, abs(part) * 1e-10):
        return None
    return part / total


def summarize(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values),
            'seed_values': dict(zip(map(str, SEEDS), values))}


def check_metrics(m):
    cells = m['quantity_cells']
    require([c['bin'] for c in cells] == list(range(5)), 'Quantity bins changed')
    require(sum(c['count'] for c in cells) == m['count'], 'Bin count does not conserve population')
    for total in ('qty_sse', 'qty_absolute_error_sum', 'qty_signed_error_sum', 'time_nll_sum'):
        require(near(math.fsum(c[total] for c in cells), m[total]), 'Bin sum mismatch: ' + total)
    require(near(m['qty_rmse'] ** 2, m['qty_sse'] / m['count']), 'RMSE/SSE mismatch')
    require(near(m['qty_mae'], m['qty_absolute_error_sum'] / m['count']), 'MAE sum mismatch')
    require(near(m['qty_bias'], m['qty_signed_error_sum'] / m['count']), 'Bias sum mismatch')
    for c in cells:
        require(c['count'] > 0 and c['qty_sse'] >= 0 and c['qty_absolute_error_sum'] >= 0,
                'Invalid quantity cell')
        require(abs(c['qty_signed_error_sum']) <= c['qty_absolute_error_sum'] + 1e-7,
                'Signed errors exceed absolute errors')
        require(near(c['qty_rmse'] ** 2, c['qty_sse'] / c['count']), 'Bin RMSE/SSE mismatch')
        require(near(c['qty_mae'], c['qty_absolute_error_sum'] / c['count']), 'Bin MAE mismatch')
        require(near(c['qty_bias'], c['qty_signed_error_sum'] / c['count']), 'Bin bias mismatch')
    require(m['tail'] == {k: v for k, v in cells[-1].items() if k != 'bin'}, 'Tail differs from final bin')


def loss_definition(root, contract):
    files = contract['sources']['components']['files']
    verified = {}
    for relative in ('models/TPPs/CountAwareTPP.py', 'paper/scripts/count_aware_tpp_backbone/core.py',
                     'paper/scripts/count_aware_tpp_backbone/training.py'):
        path = root / 'source' / relative
        require(sha(path) == files[relative], 'Frozen loss source changed')
        SOURCES[str(path.resolve())] = sha(path)
        verified[relative] = sha(path)
    text = (root / 'source/models/TPPs/CountAwareTPP.py').read_text()
    require('target = torch.log1p(true_quantity.clamp_min(0.0))' in text
            and 'log_mse = F.mse_loss(location, target, reduction="none")' in text
            and 'train_loss = log_mse' in text, 'Loss definition changed')
    return verified


def load():
    b24 = PROJECT / 'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1'
    previous = PROJECT / 'reports/titantpp_components_taxi_raf24_test_20261014_v1'
    receipt = read(b24 / 'components36_receipt.json',
        '697a260c7b9dda176bcb7159454542d5c666e13eca9f6eb28623b770c67127bd')
    require(receipt['status'] == 'complete' and receipt['test_job_count'] == 36
            and receipt['source131_closure'] == SOURCE, 'Component36 scope changed')
    read(previous / 'COMPONENTS36_ANALYSIS.json', receipt['analysis_sha256'])
    qa = read(previous / 'QA_FINAL.json',
        'b1a84a55ad1998b8824a75f369421c49d0969b83f9048f5a62721fed17eb95b5')
    require(qa['status'] == 'PASS' and qa['components36_receipt_sha256'] == sha(b24 / 'components36_receipt.json'),
            'Independent prior QA scope changed')
    rows, histories, loss_sources = [], [], {}
    for bundle, canonical_sha in zip(BUNDLES, CONTRACTS):
        root = PROJECT / f'search_artifacts/titantpp_components_{bundle}_test_20261014_v1'
        c = read(root / 'contract.json')
        require(c['contract_sha256'] == canonical_sha
            == canonical({k: v for k, v in c.items() if k != 'contract_sha256'}), 'Contract changed')
        loss_sources[bundle] = loss_definition(root, c)
        manifest = read(root / 'evaluation_results_manifest.json')
        require(manifest['contract_sha256'] == canonical_sha and manifest['actual_pipeline_exit'] == 0,
                'Evaluation manifest scope differs')
        for j in c['jobs']:
            if j['arm'] not in ARMS:
                continue
            require(j['frozen_source_closure'] == SOURCE, 'Source131 changed')
            h = read(root / j['history_path'], j['history_sha256'])['history']
            require([r['epoch'] for r in h] == list(range(1, len(h) + 1)), 'History epochs incomplete')
            best = None
            for r in h:
                metric = r['val_qty_rmse']
                if math.isfinite(metric) and (best is None or metric < best['val_qty_rmse']):
                    best = r
            require(best and best['epoch'] == j['selected_epoch']
                    and near(best['val_qty_rmse'], j['selected_metric_value']), 'Selected epoch changed')
            require(c['datasets'][j['dataset']]['definition']['model']['quantity_variant']
                    == 'count_only_log_regression', 'Quantity objective differs')
            require(all(near(r['val_quantity_train_loss'], r['val_log_qty_mse']) for r in h),
                    'Validation training-loss definition differs')
            terminal = read(root / j['terminal_manifest_path'], j['terminal_manifest_sha256'])
            require(terminal['scientific_success'] is True and terminal['status'] == 'complete',
                    'Training terminal success absent')
            trainlog_rel = j['history_original_relative_path'].rsplit('/', 1)[0] + '/train.log'
            trainlog_path = root / 'originals/run' / j['id'] / trainlog_rel
            require(sha(trainlog_path) == terminal['files'][trainlog_rel], 'Training text log changed')
            SOURCES[str(trainlog_path.resolve())] = sha(trainlog_path)
            window = [r for r in h if abs(r['epoch'] - best['epoch']) <= 5]
            end = h[-1]
            fields = ('train_quantity_loss', 'val_log_qty_mse', 'val_qty_rmse', 'val_qty_mae',
                      'train_time_nll', 'val_time_nll', 'train_time_sigma', 'train_gradient_clip_fraction',
                      'train_pre_clip_grad_norm_mean', 'train_pre_clip_grad_norm_max')
            histories.append({'dataset': j['dataset'], 'seed': j['seed'], 'arm': ARMS[j['arm']],
                'id': j['id'], 'host': j['training_host'], 'selected_epoch': best['epoch'],
                'stopped_epoch': end['epoch'], 'selected': {k: best[k] for k in fields},
                'last': {k: end[k] for k in fields}, 'all_saved_train_finite': all(r['train_all_finite'] is True for r in h),
                'early_best_le_20': best['epoch'] <= 20,
                'late_best_ge_100': best['epoch'] >= 100,
                'selected_raw_RMSE_epoch_is_log_MSE_minimum': best['epoch'] == min(h, key=lambda r:r['val_log_qty_mse'])['epoch'],
                'val_log_MSE_minimum_epoch': min(h, key=lambda r:r['val_log_qty_mse'])['epoch'],
                'train_log_loss_change_selected_to_last': end['train_quantity_loss'] - best['train_quantity_loss'],
                'val_log_MSE_change_selected_to_last': end['val_log_qty_mse'] - best['val_log_qty_mse'],
                'val_raw_RMSE_change_selected_to_last': end['val_qty_rmse'] - best['val_qty_rmse'],
                'neighborhood_pm5': {'count': len(window), 'first_epoch': window[0]['epoch'], 'last_epoch': window[-1]['epoch'],
                    'RMSE_median': statistics.median(r['val_qty_rmse'] for r in window),
                    'RMSE_max': max(r['val_qty_rmse'] for r in window),
                    'median_excess_over_selected_percent': 100*(statistics.median(r['val_qty_rmse'] for r in window)/best['val_qty_rmse']-1)},
                'saved_curve': [{k:r[k] for k in ('epoch',)+fields} for r in h]})
            for split, filename in (('validation','validation_replay.json'), ('test','test_metrics.json')):
                relative = f'results/{j["id"]}/{filename}'
                r = read(root / relative, manifest['files'][relative])
                relative_exit = f'results/{j["id"]}/{split}_worker_exit.json'
                exit_row = read(root / relative_exit, manifest['files'][relative_exit])
                require(exit_row['actual_child_exit_observed'] is True and exit_row['exit_code'] == 0, 'Worker exit0 absent')
                require(r['status'] == 'complete' and r['source_closure'] == SOURCE
                        and r['selected_epoch'] == j['selected_epoch'] and r['id'] == j['id']
                        and r['split'] == split and r['contract_sha256'] == canonical_sha
                        and r['checkpoint_sha256'] == j['checkpoint']['sha256']
                        and r['state_sha256'] == j['checkpoint_state_sha256'], 'Model identity changed')
                require(r['retrained'] is False and r['reselected'] is False
                        and r['raw_predictions_written'] is False, 'Unapproved execution recorded')
                population = c['datasets'][j['dataset']][split + '_population']
                for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256'):
                    require(r[key] == population[key], 'Population changed')
                check_metrics(r['metrics'])
                rows.append({'id': j['id'], 'dataset': j['dataset'], 'seed': j['seed'], 'arm': ARMS[j['arm']],
                    'split': split, 'selected_epoch': j['selected_epoch'], 'host': j['training_host'],
                    'checkpoint_sha256': j['checkpoint']['sha256'], 'tensor_sha256': j['checkpoint_state_sha256'],
                    'receipt_path': str((root / relative).resolve()), 'receipt_sha256': sha(root / relative),
                    'target_identity_sha256': r['target_identity_sha256'], 'truth_sha256': r['truth_sha256'],
                    'time_sha256': r['time_sha256'], 'metrics': r['metrics'],
                    'parameters': r['physical_parameter_counts'], 'runtime_cudnn': r['runtime']['cudnn']})
    require(len(histories) == 18 and len(rows) == 36, 'Internal exact18 fit /36 result scope differs')
    require(len({(r['dataset'],r['seed'],r['arm'],r['split']) for r in rows}) == 36, 'Duplicate internal result')
    return rows, histories, loss_sources


def decompose(rows):
    lookup = {(r['dataset'],r['seed'],r['arm'],r['split']): r for r in rows}
    summaries, pairs, shifts = [], [], []
    for dataset in DATASETS:
        split_summaries = {}
        for split in ('validation', 'test'):
            armrows = {arm:[lookup[dataset,seed,arm,split] for seed in SEEDS] for arm in ('CNN','FULL')}
            for seed in SEEDS:
                f, c = lookup[dataset,seed,'FULL',split],lookup[dataset,seed,'CNN',split]
                require([f[k] for k in ('target_identity_sha256','truth_sha256','time_sha256')]
                        == [c[k] for k in ('target_identity_sha256','truth_sha256','time_sha256')], 'Paired populations differ')
                require(f['metrics']['quantity_boundaries'] == c['metrics']['quantity_boundaries'], 'Paired bins differ')
                n=f['metrics']['count']; bins=[]
                for fc,cc in zip(f['metrics']['quantity_cells'],c['metrics']['quantity_cells']):
                    require(fc['count']==cc['count'], 'Paired bin counts differ')
                    bins.append({'bin':fc['bin'],'count':fc['count'],'population_weight':fc['count']/n,
                        'MSE_gap_within_bin':(fc['qty_sse']-cc['qty_sse'])/fc['count'],
                        'MSE_gap_population_contribution':(fc['qty_sse']-cc['qty_sse'])/n,
                        'MAE_gap_population_contribution':(fc['qty_absolute_error_sum']-cc['qty_absolute_error_sum'])/n,
                        'FULL_bias':fc['qty_bias'],'CNN_bias':cc['qty_bias'],
                        'FULL_MAE':fc['qty_mae'],'CNN_MAE':cc['qty_mae'],
                        'FULL_RMSE':fc['qty_rmse'],'CNN_RMSE':cc['qty_rmse'],
                        'FULL_negative_error_absolute_magnitude_fraction':(1-fc['qty_signed_error_sum']/fc['qty_absolute_error_sum'])/2,
                        'CNN_negative_error_absolute_magnitude_fraction':(1-cc['qty_signed_error_sum']/cc['qty_absolute_error_sum'])/2})
                gap=(f['metrics']['qty_sse']-c['metrics']['qty_sse'])/n
                mae=f['metrics']['qty_mae']-c['metrics']['qty_mae']
                require(near(math.fsum(b['MSE_gap_population_contribution'] for b in bins),gap), 'Paired MSE conservation failed')
                require(near(math.fsum(b['MAE_gap_population_contribution'] for b in bins),mae), 'Paired MAE conservation failed')
                pairs.append({'dataset':dataset,'split':split,'seed':seed,'count':n,'FULL_minus_CNN_MSE':gap,
                    'FULL_minus_CNN_RMSE':f['metrics']['qty_rmse']-c['metrics']['qty_rmse'],
                    'FULL_minus_CNN_MAE':mae,'bins':bins,'tail_signed_share_of_MSE_gap':signed_share(bins[-1]['MSE_gap_population_contribution'],gap)})
            selected=[p for p in pairs if p['dataset']==dataset and p['split']==split]
            gap=statistics.mean(p['FULL_minus_CNN_MSE'] for p in selected)
            bins=[]
            for b in range(5):
                entries=[p['bins'][b] for p in selected]
                require(len({p['count'] for p in entries})==1, 'Seed-dependent target bin count')
                bins.append({k:statistics.mean(p[k] for p in entries) for k in entries[0]})
            arm_metrics={arm:{k:summarize([r['metrics'][k] for r in armrows[arm]])
                for k in ('qty_rmse','qty_mae','qty_bias','time_nll')} for arm in ('CNN','FULL')}
            summary={'dataset':dataset,'split':split,'arm_metrics':arm_metrics,
                'mean_MSE_gap_FULL_minus_CNN':gap,
                'mean_RMSE_gap_FULL_minus_CNN':statistics.mean(p['FULL_minus_CNN_RMSE'] for p in selected),
                'mean_MAE_gap_FULL_minus_CNN':statistics.mean(p['FULL_minus_CNN_MAE'] for p in selected),
                'FULL_RMSE_wins':sum(p['FULL_minus_CNN_RMSE']<0 for p in selected),
                'FULL_MAE_wins':sum(p['FULL_minus_CNN_MAE']<0 for p in selected),
                'bins':bins,'tail_signed_share_of_mean_MSE_gap':signed_share(bins[-1]['MSE_gap_population_contribution'],gap)}
            summaries.append(summary);split_summaries[split]=summary
        v,t=split_summaries['validation'],split_summaries['test'];bins=[]
        for vb,tb in zip(v['bins'],t['bins']):
            pv,pt=vb['population_weight'],tb['population_weight'];dv,dt=vb['MSE_gap_within_bin'],tb['MSE_gap_within_bin']
            mix=(pt-pv)*(dt+dv)/2;within=(pt+pv)*(dt-dv)/2
            require(near(mix+within,tb['MSE_gap_population_contribution']-vb['MSE_gap_population_contribution']), 'Symmetric decomposition failed')
            bins.append({'bin':vb['bin'],'quantity_mix_component':mix,'within_recorded_bin_gap_component':within,
                'total_reversal_contribution':mix+within,'Val_weight':pv,'Test_weight':pt,
                'Val_MSE_gap_within_bin':dv,'Test_MSE_gap_within_bin':dt})
        change=t['mean_MSE_gap_FULL_minus_CNN']-v['mean_MSE_gap_FULL_minus_CNN']
        require(near(math.fsum(b['total_reversal_contribution'] for b in bins),change), 'Reversal decomposition does not conserve')
        mix=math.fsum(b['quantity_mix_component'] for b in bins);within=math.fsum(b['within_recorded_bin_gap_component'] for b in bins)
        shifts.append({'dataset':dataset,'MSE_gap_change_Test_minus_Validation':change,'quantity_mix_component':mix,
            'within_recorded_bin_gap_component':within,'mix_signed_share':signed_share(mix,change),
            'within_bin_signed_share':signed_share(within,change),'tail_signed_share_of_reversal':signed_share(bins[-1]['total_reversal_contribution'],change),
            'bins':bins,'interpretation':'Exact descriptive symmetric decomposition; within-bin includes unmeasured covariates and within-bin population shifts, not a causal mechanism.'})
    return summaries,pairs,shifts


def markdown(out):
    lines=['# CNN+GRU(FULL) 대 CNN: Validation → Test 순위 반전 진단', '',
        '> Material Passport: source131 구성요소 실험만 사용. 로컬 로그·봉인된 누적오차 분석. 새 학습·추론·원격 조회·checkpoint 역직렬화·원고 변경 0.', '',
        'FULL−CNN이 음수면 FULL이 좋고, 양수면 CNN이 좋습니다. 3seed 평균 RMSE의 차이와 평균 MSE(제곱오차)의 차이는 서로 다른 값입니다.', '',
        '| 데이터셋 | split | FULL−CNN 평균 RMSE | FULL−CNN 평균 MAE | FULL RMSE 승리 | 평균 MSE 차이 | 큰 수량의 MSE 차이 기여 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for s in out['summaries']:
        share=s['tail_signed_share_of_mean_MSE_gap'];txt='미정의' if share is None else f'{share*100:.2f}%'
        lines.append(f"| {NAMES[s['dataset']]} | {s['split']} | {s['mean_RMSE_gap_FULL_minus_CNN']:+.6f} | {s['mean_MAE_gap_FULL_minus_CNN']:+.6f} | {s['FULL_RMSE_wins']}/3 | {s['mean_MSE_gap_FULL_minus_CNN']:+.6f} | {txt} |")
    lines += ['', '기여율은 signed gap의 분해입니다. 음수나 100% 초과는 다른 구간의 상쇄를 의미합니다. RMSE 개선율·오류 발생 비율로 해석하지 않습니다.', '',
        '## 반전은 구간 비중 변화 때문인가, 구간 안의 오차 차이 변화 때문인가?', '',
        '각 split에서 D=Σ pᵦ·dᵦ입니다. pᵦ는 대상 수량 구간 비중, dᵦ는 FULL−CNN의 구간 MSE 차이입니다. Test−Validation의 D 변화는 대칭식 Σ(pT−pV)(dT+dV)/2 + Σ(pT+pV)(dT−dV)/2로 정확히 분해했습니다.', '',
        '| 데이터셋 | MSE 차이의 반전 크기 | 구간 비중 변화 기여 | 구간 내 차이 변화 기여 | 큰 수량 구간의 반전 기여 |',
        '|---|---:|---:|---:|---:|']
    for s in out['reversal_decomposition']:
        lines.append(f"| {NAMES[s['dataset']]} | {s['MSE_gap_change_Test_minus_Validation']:+.6f} | {100*s['mix_signed_share']:.2f}% | {100*s['within_bin_signed_share']:.2f}% | {100*s['tail_signed_share_of_reversal']:.2f}% |")
    lines += ['', '이는 넓은 수량 구간의 비중 변화만으로 순위 반전을 설명할 수 있는지를 보는 산술 분해입니다. 구간 내부의 수량·이력·기간·품목 변화는 분리하지 못하므로 “분포 변화가 없다”거나 “GRU가 원인이다”라는 인과 결론은 아닙니다.', '',
        '## Test 구간별 평균 편향', '', '편향은 예측−정답입니다. 음수는 순 과소 예측이며, 모든 대상의 과소 예측이나 과소 예측 건수 비율을 뜻하지 않습니다.', '',
        '| 데이터셋 | 수량 구간 | 대상 수 | FULL RMSE | CNN RMSE | FULL 편향 | CNN 편향 | FULL−CNN 전체 MSE 기여 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in out['summaries']:
        if s['split'] != 'test': continue
        boundaries=next(r['metrics']['quantity_boundaries'] for r in out['rows'] if r['dataset']==s['dataset'])
        labels=[f'≤{boundaries[0]:g}']+[f'({boundaries[i-1]:g},{boundaries[i]:g}]' for i in range(1,4)]+[f'>{boundaries[-1]:g}']
        for b,label in zip(s['bins'],labels):
            lines.append(f"| {NAMES[s['dataset']]} | {label} | {int(b['count'])} | {b['FULL_RMSE']:.6f} | {b['CNN_RMSE']:.6f} | {b['FULL_bias']:+.6f} | {b['CNN_bias']:+.6f} | {b['MSE_gap_population_contribution']:+.6f} |")
    lines += ['', '## Seed별 반전과 학습·선택 로그', '',
        '| 데이터셋 | seed | Validation FULL−CNN RMSE | Test FULL−CNN RMSE | FULL 선택/종료 epoch | CNN 선택/종료 epoch |',
        '|---|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        for seed in SEEDS:
            p={r['split']:r for r in out['seed_pairs'] if r['dataset']==d and r['seed']==seed}
            h={r['arm']:r for r in out['training_histories'] if r['dataset']==d and r['seed']==seed}
            lines.append(f"| {NAMES[d]} | {seed} | {p['validation']['FULL_minus_CNN_RMSE']:+.6f} | {p['test']['FULL_minus_CNN_RMSE']:+.6f} | {h['FULL']['selected_epoch']}/{h['FULL']['stopped_epoch']} | {h['CNN']['selected_epoch']}/{h['CNN']['stopped_epoch']} |")
    lines += ['', '| 데이터셋 | seed | arm | 선택 Train log MSE | 선택 Val log MSE | 선택후 Train 감소 | 선택후 Val raw RMSE 증가 | ±5epoch 중앙값의 선택값 대비 증가 | 선택 epoch clip 비율 |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for h in out['training_histories']:
        lines.append(f"| {NAMES[h['dataset']]} | {h['seed']} | {h['arm']} | {h['selected']['train_quantity_loss']:.6f} | {h['selected']['val_log_qty_mse']:.6f} | {h['train_log_loss_change_selected_to_last']:+.6f} | {h['val_raw_RMSE_change_selected_to_last']:+.6f} | {h['neighborhood_pm5']['median_excess_over_selected_percent']:.2f}% | {100*h['selected']['train_gradient_clip_fraction']:.2f}% |")
    lines += ['', '## 확인된 사실과 원인 해석의 경계', '']
    for x in out['confirmed']:
        lines.append('- '+x)
    lines += ['', '## 아직 확인하지 못한 원인·제한', '']
    for x in out['limitations']:
        lines.append('- '+x)
    lines += ['', '상세 36개 raw 누적오차 기록·18개 전 epoch 로그·seed별 분해·SHA·기존 QA 연결은 INTERNAL_ANALYSIS.json에 있습니다. 각 숫자는 해당 split의 고정 선택 checkpoint에서 나온 저장값입니다.', '']
    return '\n'.join(lines)


def main():
    rows,histories,loss_sources=load();summaries,pairs,shifts=decompose(rows)
    require(all(h['all_saved_train_finite'] for h in histories), 'A saved epoch reports nonfinite training')
    mismatch=sum(not h['selected_raw_RMSE_epoch_is_log_MSE_minimum'] for h in histories)
    train_down_val_up=sum(h['train_log_loss_change_selected_to_last']<0 and h['val_raw_RMSE_change_selected_to_last']>0 for h in histories)
    out={'schema':'titantpp_internal_rank_reversal_diagnostics_v1','status':'complete',
        'scope':'source131 fixed selected FULL versus CNN; three datasets x three seeds x two arms; 36 Validation/Test rows',
        'new_training':False,'new_inference':False,'remote_access':False,'checkpoint_deserialization':False,
        'checkpoint_reselection':False,'test_independent':False,'causal_mechanism_established':False,
        'source131_closure':SOURCE,'summaries':summaries,'seed_pairs':pairs,'reversal_decomposition':shifts,
        'training_histories':histories,'rows':rows,'frozen_loss_source_SHA':loss_sources,
        'confirmed':[
            '18개 fit의 선택 epoch는 저장 history 전체 Validation raw 수량 RMSE의 최초 strict finite 최소와 일치했습니다. 실제 Val/Test 결과의 binary·tensor·대상 지문·source131 및 worker exit0가 계약과 일치합니다.',
            '18개 fit의 모든 저장 epoch에서 train_all_finite=true입니다. 비유한 수치 실패나 잘못된 종료 checkpoint를 선택한 흔적은 없습니다.',
            f'선택 raw RMSE epoch와 Validation log MSE 최소 epoch가 다른 fit은 {mismatch}/18개입니다. 학습 목적은 log1p 수량 MSE이고 선택·주 평가는 raw 수량 RMSE로, 크기 가중이 다릅니다.',
            f'선택 epoch 이후 종료까지 Train 온라인 log MSE가 감소하고 Validation raw RMSE가 증가한 fit은 {train_down_val_up}/18개입니다. 마지막 모델이 아닌 선택 모델로 Test를 평가했습니다.',
            'FULL 대 CNN의 Test 평균 MSE 악화는 Taxi·RAF에서 큰 수량 구간의 signed 기여가 크고, Intermittent에서는 주로 큰 수량 이외 구간에서 발생합니다. 데이터셋별 분해와 seed 예외를 함께 보존했습니다.',
        ],
        'limitations':[
            'Train raw 수량 RMSE·MAE·구간별 raw 오차는 기존 로그에 없습니다. Train 단계부터 큰 수량을 못 맞췄는지, 고정 모델의 Train 적합과 미래 기간 일반화 중 무엇이 주된 원인인지 확정할 수 없습니다.',
            'train_quantity_loss는 매 batch 업데이트 전·train mode(dropout 포함)의 온라인 log1p MSE이고 val_log_qty_mse는 epoch 끝 고정 모델·eval mode의 log1p MSE입니다. 함수 정의는 같지만 고정 모델의 동일조건 Train/Validation gap은 아닙니다.',
            '높은 gradient clip 비율은 joint time+quantity 전체 gradient의 norm 기록입니다. 이를 GRU만의 gradient 문제나 시간 head가 수량 오류를 유발한 증거로 해석할 수 없습니다.',
            'σ와 TimeNLL telemetry는 시간 head 관련 관찰입니다. 시간 분포 폭과 수량 일반화의 인과 연결을 검증하지 않았습니다.',
            '예측 행별 파일이 저장되지 않아 시간 구간·품목별 오차, residual scatter, 오류 건수·과소 예측 건수 비율, input-condition별 GRU 활성화 효과를 확인하지 못했습니다.',
            '넓은 Train 수량 bin 내부의 distribution과 conditioning은 남아 있습니다. Exact mix/within-bin 분해는 원인 실험이 아닌 기술적 오차 분해입니다.',
            'FULL과 CNN은 활성 파라미터 수가 다릅니다. source131 구성 비교만으로 용량 효과와 GRU 구조 효과를 분리할 수 없습니다.',
            'Intermittent seed42/52는 5090/cuDNN92000, seed62는 5080/cuDNN91900입니다. 같은 seed의 FULL/CNN은 같은 host이고 cross-seed 시간을 합치지 않았습니다.',
            '3seed는 기술 통계입니다. 통계적 유의성·인과적 원인 확정·독립 Test 검증으로 확대하지 않습니다. 기존 다른 캠페인의 FULL 값을 대체하거나 합치지 않았습니다.',
        ], 'source_files_SHA':SOURCES,'script_sha256':sha(__file__)}
    for path,value in ((REPORT/'INTERNAL_ANALYSIS.json',json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),
                       (REPORT/'INTERNAL_ANALYSIS.md',markdown(out))):
        with path.open('x',encoding='utf-8') as stream:stream.write(value)
    print(json.dumps({'status':'complete','fit_count':18,'result_count':36,
        'report':str(REPORT/'INTERNAL_ANALYSIS.md'),'json_sha256':sha(REPORT/'INTERNAL_ANALYSIS.json'),
        'report_sha256':sha(REPORT/'INTERNAL_ANALYSIS.md')},ensure_ascii=False))


if __name__=='__main__':
    main()
